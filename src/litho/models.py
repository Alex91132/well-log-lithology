"""Модели-ступени из PLAN.md и решение с учётом штрафной матрицы."""

import numpy as np
import pandas as pd

from .data import N_CLASSES
from .features import CURVES, build, categories_of

LGB_PARAMS = dict(objective="multiclass", num_class=N_CLASSES, learning_rate=0.08, n_estimators=400,
                  num_leaves=63, min_child_samples=200, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.7, reg_lambda=1.0, verbose=-1, n_jobs=8)


def bayes_decision(proba, A):
    """Класс с минимальным ожидаемым штрафом: argmin_k Σ_j p_j · A[j, k] (A[истина, прогноз])."""
    return np.argmin(proba @ A, axis=1)


class Constant:
    """«Везде глина» — самый частый класс."""
    name = "Константа «глина»"

    def fit(self, df):
        return self

    def predict_proba(self, df):
        p = np.zeros((len(df), N_CLASSES))
        p[:, 2] = 1.0
        return p


class LogReg:
    """Мультиномиальная логистическая регрессия на сырых кривых (медианы + индикаторы пропусков)."""
    name = "Логистическая регрессия"

    def fit(self, df):
        from sklearn.linear_model import LogisticRegression
        X = self._X(df, fit=True)
        self.model = LogisticRegression(max_iter=300, C=1.0)
        self.model.fit(X, df.y.values)
        return self

    def _X(self, df, fit=False):
        X, _ = build(df, "raw")
        X = X.loc[df.index]
        miss = X[CURVES].isna().astype(float).add_suffix("_na")
        if fit:
            self.med = X.median()
            self.mu = X.fillna(self.med).mean()
            self.sd = X.fillna(self.med).std().replace(0, 1)
        Z = ((X.fillna(self.med) - self.mu) / self.sd).clip(-6, 6)
        return pd.concat([Z, miss], axis=1).values

    def predict_proba(self, df):
        p = np.zeros((len(df), N_CLASSES))
        p[:, self.model.classes_] = self.model.predict_proba(self._X(df))
        return p


class LGBM:
    """LightGBM на наборе признаков raw | strat | full."""
    NAMES = {"raw": "LightGBM: сырые кривые", "strat": "LightGBM: + стратиграфия",
             "full": "LightGBM: + признаки по глубине"}

    def __init__(self, feature_set="full", **params):
        self.feature_set = feature_set
        self.params = {**LGB_PARAMS, **params}
        self.name = self.NAMES[feature_set]

    def fit(self, df, train_mask=None):
        """train_mask — какие строки брать в обучение (признаки по окнам считаются по всем строкам скважины)."""
        X, _ = build(df, self.feature_set)
        m = np.ones(len(df), bool) if train_mask is None else np.asarray(train_mask)
        return self.fit_features(X.loc[df.index][m], df.y.values[m])

    def fit_features(self, X, y):
        """Обучение на готовых признаках — в кросс-валидации признаки строятся один раз."""
        import lightgbm as lgb
        self.categories = categories_of(X)
        self.columns = list(X.columns)
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(X, y)
        return self

    def features(self, df):
        X, _ = build(df, self.feature_set, self.categories)
        return X.loc[df.index, self.columns]

    def proba_features(self, X):
        p = np.zeros((len(X), N_CLASSES))
        p[:, self.model.classes_] = self.model.predict_proba(X[self.columns])
        return p

    def predict_proba(self, df):
        return self.proba_features(self.features(df))
