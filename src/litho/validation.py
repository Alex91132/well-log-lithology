"""Кросс-валидация по точкам и по скважинам, метрики, калибровка."""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, KFold

from .data import N_CLASSES, score
from .models import bayes_decision


def stride_mask(df, k):
    """Каждая k-я точка внутри скважины: соседние точки почти одинаковы, обучение быстрее в k раз."""
    return (df.groupby("WELL", sort=False).cumcount() % k == 0).values


def folds(df, kind, n=5, seed=0):
    if kind == "wells":
        return list(GroupKFold(n).split(df, groups=df.WELL))
    if kind == "points":
        return list(KFold(n, shuffle=True, random_state=seed).split(df))
    raise ValueError(kind)


def oof_proba(df, make_model, splits, X=None, stride=4, log=print):
    """
    Вероятности «вне выборки»: каждая точка предсказана моделью, которая её не видела.
    X — готовые признаки (для LightGBM), иначе модель получает строки df.
    """
    P = np.zeros((len(df), N_CLASSES), dtype=np.float32)
    sub = stride_mask(df, stride)
    for i, (tr, te) in enumerate(splits):
        tr = tr[sub[tr]]
        m = make_model()
        if X is not None:
            m.fit_features(X.iloc[tr], df.y.values[tr])
            P[te] = m.proba_features(X.iloc[te])
        else:
            m.fit(df.iloc[tr])
            P[te] = m.predict_proba(df.iloc[te])
        log(f"  fold {i + 1}/{len(splits)}: score {score(df.y.values[te], P[te].argmax(1)):.4f}")
    return P


def metrics(y, proba, A):
    pa, pb = proba.argmax(1), bayes_decision(proba, A)
    return {"score": score(y, pa, A), "score_bayes": score(y, pb, A),
            "accuracy": float((pa == y).mean()),
            "macro_f1": float(f1_score(y, pa, average="macro", labels=np.unique(y)))}


def per_well(df, proba, A, decision="argmax"):
    pb = bayes_decision(proba, A) if decision == "bayes" else proba.argmax(1)
    return (pd.DataFrame({"well": df.WELL.values, "pen": A[df.y.values, pb], "ok": pb == df.y.values})
            .groupby("well").agg(score=("pen", lambda s: -s.mean()), accuracy=("ok", "mean"), n=("ok", "size"))
            .reset_index())


def reliability(y, proba, bins=10):
    """Диаграмма надёжности по уверенности (max p): заявленная уверенность против доли верных."""
    conf, pred = proba.max(1), proba.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(conf, edges) - 1, 0, bins - 1)
    t = (pd.DataFrame({"b": b, "conf": conf, "ok": pred == y}).groupby("b")
         .agg(conf=("conf", "mean"), acc=("ok", "mean"), n=("ok", "size")).reset_index())
    ece = float((t.n * (t.conf - t.acc).abs()).sum() / t.n.sum())
    return t, ece


def coverage_curve(y, proba, A, qs=(1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5)):
    """Если отдать геологу на ручную проверку наименее уверенные точки — насколько точнее остальное."""
    conf = proba.max(1)
    pb = proba.argmax(1)
    rows = []
    for q in qs:
        thr = np.quantile(conf, 1 - q) if q < 1 else -1
        m = conf > thr if q < 1 else np.ones(len(y), bool)
        rows.append({"coverage": q, "threshold": float(max(thr, 0)), "accuracy": float((pb[m] == y[m]).mean()),
                     "score": score(y[m], pb[m], A)})
    return pd.DataFrame(rows)
