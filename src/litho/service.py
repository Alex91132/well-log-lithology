"""
HTTP-сервис: каротаж скважины (LAS или CSV) → литология по глубине.

    uvicorn litho.service:app --app-dir src            # http://127.0.0.1:8000/docs
"""

import io
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from .artifact import REQUIRED, Artifact
from .data import NAMES
from .features import CURVES

MODEL_DIR = Path(os.environ.get("LITHO_MODEL_DIR", Path(__file__).resolve().parents[2] / "models"))
MAX_ROWS = 200_000
ALIASES = {"DEPT": "DEPTH_MD", "DEPTH": "DEPTH_MD", "MD": "DEPTH_MD"}

app = FastAPI(title="Литология по каротажу",
              description="LightGBM, обученный на 98 скважинах FORCE 2020. Возвращает литологию, "
                          "уверенность и предупреждения о качестве входных данных.",
              version="1.0")


@lru_cache(maxsize=1)
def model() -> Artifact:
    return Artifact.load(MODEL_DIR)


class Point(BaseModel):
    lithology: int
    name: str
    depth: float
    confidence: float = Field(description="вероятность выбранного класса")
    proba: dict[str, float] | None = None


class Prediction(BaseModel):
    model_version: str
    well: str
    n_points: int
    warnings: list[str]
    drift_psi: dict[str, float]
    summary: dict[str, float] = Field(description="доля интервала по литологиям")
    points: list[Point]


class LogsIn(BaseModel):
    """Каротаж в JSON: глубины и кривые одинаковой длины."""
    well: str = "WELL"
    depth: list[float]
    curves: dict[str, list[float | None]]
    group: list[str | None] | None = None
    formation: list[str | None] | None = None


def _frame(df, well):
    df = df.rename(columns={c: ALIASES.get(c.upper(), c.upper()) for c in df.columns})
    if "DEPTH_MD" not in df:
        raise HTTPException(422, "нет столбца глубины (DEPTH_MD / DEPT)")
    missing = [c for c in REQUIRED if c not in df or df[c].isna().all()]
    if missing:
        raise HTTPException(422, f"нет обязательных кривых: {', '.join(missing)}")
    if len(df) > MAX_ROWS:
        raise HTTPException(413, f"больше {MAX_ROWS:,} точек — разбейте скважину на части")
    df = df.copy()
    df["WELL"] = df["WELL"].astype(str) if "WELL" in df else well
    for c in CURVES + ["X_LOC", "Y_LOC", "Z_LOC"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").replace([-999.25, -9999], np.nan)
    for c in ("GROUP", "FORMATION"):
        if c not in df:
            df[c] = np.nan
    return df.reset_index(drop=True)


def _predict(df, with_proba):
    m = model()
    out, P = m.predict(df)
    warnings, drift = m.check(df, P)
    if df.GROUP.isna().all():
        warnings.append("нет стратиграфии (GROUP/FORMATION) — модель работает, но точность ниже")
    summary = out.name.value_counts(normalize=True).round(4).to_dict()
    points = [Point(lithology=int(r.lithology), name=r.name, depth=float(r.DEPTH_MD),
                    confidence=round(float(r.confidence), 4),
                    proba=({NAMES[j]: round(float(P[i, j]), 4) for j in np.nonzero(P[i] > 0.01)[0]}
                           if with_proba else None))
              for i, r in enumerate(out.itertuples())]
    return Prediction(model_version=m.meta["version"], well=str(df.WELL.iloc[0]), n_points=len(out),
                      warnings=warnings, drift_psi=drift, summary=summary, points=points)


@app.get("/health")
def health():
    return {"status": "ok", "model_version": model().meta["version"]}


@app.get("/model")
def model_info():
    meta = model().meta
    return {k: meta[k] for k in ("version", "feature_set", "metrics", "lightgbm_params")} | {
        "n_features": len(meta["columns"]), "classes": NAMES}


@app.post("/predict", response_model=Prediction)
async def predict_file(file: UploadFile = File(..., description="LAS 2.0 или CSV (разделитель , или ;)"),
                       proba: bool = Query(False, description="вернуть вероятности классов")):
    raw = await file.read()
    name = (file.filename or "").lower()
    try:
        if name.endswith(".las"):
            import lasio
            las = lasio.read(io.StringIO(raw.decode("utf-8", errors="replace")))
            df = las.df().reset_index()
            well = str(las.well.WELL.value or Path(name).stem) if "WELL" in las.well else Path(name).stem
        else:
            df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python")
            well = Path(name).stem or "WELL"
    except Exception as e:  # noqa: BLE001 — любая ошибка разбора — это плохой файл
        raise HTTPException(400, f"не удалось прочитать файл: {e}") from e
    return _predict(_frame(df, well), proba)


@app.post("/predict/json", response_model=Prediction)
def predict_json(body: LogsIn, proba: bool = Query(False)):
    n = len(body.depth)
    bad = [k for k, v in body.curves.items() if len(v) != n]
    if bad:
        raise HTTPException(422, f"длина кривых {', '.join(bad)} не совпадает с глубиной")
    df = pd.DataFrame({"DEPTH_MD": body.depth, **body.curves})
    if body.group:
        df["GROUP"] = body.group
    if body.formation:
        df["FORMATION"] = body.formation
    return _predict(_frame(df, body.well), proba)
