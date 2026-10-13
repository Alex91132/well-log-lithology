"""
Признаки из каротажа. Всё, что зависит от соседних точек, считается внутри одной скважины
по возрастанию глубины — признаки скважины не зависят от того, какие ещё скважины в выборке.
"""

import numpy as np
import pandas as pd

CURVES = ["CALI", "RSHA", "RMED", "RDEP", "RHOB", "GR", "NPHI", "PEF", "DTC", "SP", "BS",
          "ROP", "DTS", "DCAL", "DRHO", "MUDWEIGHT", "RMIC", "RXO"]
RESISTIVITY = ["RSHA", "RMED", "RDEP", "RMIC", "RXO"]
POSITION = ["DEPTH_MD", "X_LOC", "Y_LOC", "Z_LOC"]
STRAT = ["GROUP", "FORMATION"]
WINDOW_CURVES = ["GR", "RHOB", "NPHI", "DTC", "PEF", "RDEP_log"]
WINDOWS = (5, 21)          # точек; шаг ≈ 15 см → окна ≈ 0,75 и 3 м

# ступени модели из PLAN.md
FEATURE_SETS = ("raw", "strat", "full")


def _base(df):
    out = pd.DataFrame(index=df.index)
    for c in CURVES:
        out[c] = pd.to_numeric(df[c], errors="coerce") if c in df else np.nan
    for c in RESISTIVITY:
        out[c] = np.log10(out[c].clip(lower=1e-3))
    for c in POSITION:
        out[c] = pd.to_numeric(df[c], errors="coerce") if c in df else np.nan
    return out


def build(df, feature_set="full", categories=None):
    """
    df — строки одной или нескольких скважин (столбцы как в данных FORCE).
    categories — словарь {GROUP: [...], FORMATION: [...]} из обучающей выборки, чтобы коды
    категорий совпадали при обучении и в сервисе.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(feature_set)
    df = df.sort_values(["WELL", "DEPTH_MD"], kind="stable")
    X = _base(df)
    X["n_missing"] = X[CURVES].isna().sum(axis=1).astype("int8")
    if feature_set == "raw":
        return X, df.index

    for c in STRAT:
        cats = (categories or {}).get(c)
        col = df[c] if c in df else pd.Series(np.nan, index=df.index)
        X[c] = (pd.Categorical(col.where(col.isin(cats)), categories=cats) if cats is not None
                else col.astype("category"))
    if feature_set == "strat":
        return X, df.index

    # физически осмысленные комбинации
    X["CALI_BS"] = X.CALI - X.BS                                  # каверны
    X["ND_sep"] = X.NPHI - (2.65 - X.RHOB) / 1.65                 # расхождение нейтрон — плотность
    X["RDEP_RMED"] = X.RDEP - X.RMED                              # проникновение (в log10)
    X["RDEP_log"] = X.RDEP
    g = X.groupby(df.WELL, sort=False)
    q05, q95 = g.GR.transform(lambda s: s.quantile(0.05)), g.GR.transform(lambda s: s.quantile(0.95))
    X["GR_norm"] = (X.GR - q05) / (q95 - q05).replace(0, np.nan)  # GR в масштабе своей скважины

    new = {}
    for c in WINDOW_CURVES:
        s = X[c].groupby(df.WELL, sort=False)
        new[f"{c}_d1"] = s.diff()
        new[f"{c}_d1b"] = -s.diff(-1)
        for w in WINDOWS:
            r = s.rolling(w, center=True, min_periods=1)
            new[f"{c}_m{w}"] = r.mean().reset_index(level=0, drop=True)
            new[f"{c}_s{w}"] = r.std().reset_index(level=0, drop=True)
    X = pd.concat([X.drop(columns="RDEP_log"), pd.DataFrame(new, index=X.index)], axis=1)
    return X, df.index


def categories_of(X):
    return {c: list(X[c].cat.categories) for c in STRAT if c in X}
