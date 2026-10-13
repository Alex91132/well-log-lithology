"""
Артефакт модели: бустинг LightGBM + метаданные (версия, признаки, категории, эталон для дрейфа).
Один формат для обучения, сервиса и тестов.
"""

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import CODES, NAMES, N_CLASSES
from .features import CURVES, build
from .models import bayes_decision

PSI_BINS = 10
MEDIAN_MARGIN = 0.5      # допуск к диапазону медиан обучающих скважин — доля ширины диапазона
COVERAGE_WARN = 0.5      # ключевая кривая есть меньше чем на половине интервала
REQUIRED = ["GR"]                         # без GR литологию не определить; остальные кривые — по возможности
KEY_CURVES = ["GR", "RHOB", "NPHI", "DTC", "RDEP", "PEF"]


def reference_bins(df):
    """
    Эталон входных данных по каждой кривой.

    * Децили по всей обучающей выборке — для PSI. PSI возвращается для мониторинга, но тревогу не
      поднимает: одна скважина всегда отличается от смеси 98 (свой диаметр долота, свой интервал),
      и даже порог по 99-му перцентилю PSI обучающих скважин срабатывал на нормальной короткой скважине.
    * Диапазон медиан по обучающим скважинам — для проверки единиц измерения: медиана RHOB в кг/м³
      или NPHI в процентах лежит далеко за пределами любой обучающей скважины.
    """
    ref = {}
    for c in CURVES:
        v = pd.to_numeric(df[c], errors="coerce").dropna().values
        if len(v) < 1000:
            continue
        edges = np.unique(np.quantile(v, np.linspace(0, 1, PSI_BINS + 1)[1:-1]))
        counts = np.bincount(np.searchsorted(edges, v, side="right"), minlength=len(edges) + 1)
        med = pd.to_numeric(df[c], errors="coerce").groupby(df["WELL"]).median().dropna()
        lo, hi = float(med.quantile(0.01)), float(med.quantile(0.99))
        pad = MEDIAN_MARGIN * max(hi - lo, abs(hi) * 0.05)
        ref[c] = {"edges": edges.tolist(), "share": (counts / counts.sum()).tolist(),
                  "median_range": [lo - pad, hi + pad]}
    return ref


def psi(values, ref):
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna().values
    if len(v) == 0:
        return None
    edges, q = np.array(ref["edges"]), np.array(ref["share"])
    p = np.bincount(np.searchsorted(edges, v, side="right"), minlength=len(q)) / len(v)
    p, q = np.clip(p, 1e-4, None), np.clip(q, 1e-4, None)
    return float(np.sum((p - q) * np.log(p / q)))


def temperature_scale(p, T):
    """Калибровка температурой: p_k ∝ p_k^(1/T). T > 1 смягчает завышенную уверенность."""
    if T == 1.0:
        return p
    L = np.log(np.clip(p, 1e-9, 1)) / T
    L -= L.max(axis=1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(axis=1, keepdims=True)


def fit_temperature(p, y):
    from scipy.optimize import minimize_scalar
    nll = lambda T: -np.log(np.clip(temperature_scale(p, T)[np.arange(len(y)), y], 1e-9, 1)).mean()  # noqa: E731
    return float(minimize_scalar(nll, bounds=(0.5, 5), method="bounded").x)


class Artifact:
    def __init__(self, booster, meta, A):
        self.booster, self.meta, self.A = booster, meta, np.asarray(A)

    @classmethod
    def from_model(cls, model, A, reference, version, metrics):
        meta = {"version": version, "feature_set": model.feature_set, "columns": model.columns,
                "categories": model.categories, "classes": [int(c) for c in model.model.classes_],
                "reference": reference, "metrics": metrics, "lightgbm_params": {
                    k: v for k, v in model.params.items() if k not in ("verbose", "n_jobs")}}
        return cls(model.model.booster_, meta, A)

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        with gzip.open(folder / "model.txt.gz", "wt", encoding="utf-8") as f:
            f.write(self.booster.model_to_string())
        meta = {**self.meta, "penalty_matrix": self.A.tolist()}
        (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, folder):
        import lightgbm as lgb
        folder = Path(folder)
        with gzip.open(folder / "model.txt.gz", "rt", encoding="utf-8") as f:
            booster = lgb.Booster(model_str=f.read())
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        return cls(booster, meta, meta.pop("penalty_matrix"))

    def proba(self, df):
        X, _ = build(df, self.meta["feature_set"], self.meta["categories"])
        X = X.loc[df.index, self.meta["columns"]]
        p = np.zeros((len(df), N_CLASSES))
        p[:, self.meta["classes"]] = self.booster.predict(X)
        return temperature_scale(p, self.meta.get("temperature", 1.0))

    def check(self, df, proba=None):
        """
        Предупреждения о входных данных и о самой скважине:
        пропуски ключевых кривых, медиана кривой вне диапазона обучающих скважин (единицы измерения),
        низкая средняя уверенность модели — скважина не похожа на обучающие.
        """
        warnings, drift = [], {}
        missing = [c for c in KEY_CURVES if c not in df or df[c].isna().all()]
        if missing:
            warnings.append(f"нет кривых {', '.join(missing)} — точность ниже, чем на валидации")
        partial = {c: df[c].notna().mean() for c in KEY_CURVES if c in df and c not in missing}
        partial = {c: s for c, s in partial.items() if s < COVERAGE_WARN}
        if partial:
            warnings.append("кривые есть не на всём интервале: " +
                            ", ".join(f"{c} — {s:.0%}" for c, s in partial.items()))
        for c, ref in self.meta["reference"].items():
            if c in df and df[c].notna().any():
                drift[c] = round(psi(df[c], ref), 3)
                lo, hi = ref["median_range"]
                med = float(pd.to_numeric(df[c], errors="coerce").median())
                if not lo <= med <= hi:
                    warnings.append(f"{c}: медиана {med:.4g} вне диапазона обучающих скважин "
                                    f"[{lo:.3g}; {hi:.3g}] — проверьте единицы измерения и калибровку")
        thr = self.meta.get("confidence_alert")
        if proba is not None and thr is not None:
            conf = float(proba.max(axis=1).mean())
            if conf < thr:
                warnings.append(f"средняя уверенность модели {conf:.2f} ниже, чем у 95 % скважин валидации "
                                f"({thr:.2f}) — скважина не похожа на обучающие, результат проверить вручную")
        return warnings, drift

    def predict(self, df):
        """Таблица: глубина, код и название литологии, уверенность, вероятности по классам."""
        P = self.proba(df)
        k = bayes_decision(P, self.A) if self.meta.get("decision", "bayes") == "bayes" else P.argmax(1)
        out = pd.DataFrame({"WELL": df.WELL.values, "DEPTH_MD": df.DEPTH_MD.values,
                            "lithology": np.array(CODES)[k], "name": np.array(NAMES)[k],
                            "confidence": P[np.arange(len(P)), k]})
        return out, P
