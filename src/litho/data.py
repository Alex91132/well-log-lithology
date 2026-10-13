"""
Данные FORCE 2020: загрузка с проверкой контрольных сумм, метки, штрафная матрица и метрика.

Источник: Bormann P. et al. 2020, FORCE 2020 Machine Learning Competition, CC-BY 4.0
(https://doi.org/10.5281/zenodo.4351156, https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition).
"""

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
BASE = ("https://raw.githubusercontent.com/bolgebrygg/Force-2020-Machine-Learning-competition/"
        "master/lithology_competition/data/")
FILES = {
    "train.zip": "e996c8466f2d0dc7116daa71485cc90c31fcfac6345a5bb33165026478b642e9",
    "leaderboard_test_features.csv": "4688d93728feab7e86855b9315ddfc8b498387927954235f7dbbfa2216550781",
    "leaderboard_test_target.csv": "4fe5e59809d98eb9b8b0a3836776bd52e5c7e63ffefa31476fa17011a27ef382",
    "hidden_test.csv": "c75e817bfefcabd2bacdefd9e1c74dbd829d24543646bf0b85ef8fbf5c06f069",
    "penalty_matrix.npy": "c312481af94d3557616b7ed79b5d1a12afee380fd471188785ab31874f159d43",
}
TARGET = "FORCE_2020_LITHOFACIES_LITHOLOGY"
CONFIDENCE = "FORCE_2020_LITHOFACIES_CONFIDENCE"

# порядок классов — как в штрафной матрице соревнования
CODES = [30000, 65030, 65000, 80000, 74000, 70000, 70032, 88000, 86000, 99000, 90000, 93000]
NAMES = ["Песчаник", "Алевритистая глина", "Глина", "Мергель", "Доломит", "Известняк",
         "Мел", "Галит", "Ангидрит", "Туф", "Уголь", "Фундамент"]
NAMES_EN = ["Sandstone", "Sandstone/Shale", "Shale", "Marl", "Dolomite", "Limestone",
            "Chalk", "Halite", "Anhydrite", "Tuff", "Coal", "Basement"]
CODE2IDX = {c: i for i, c in enumerate(CODES)}
N_CLASSES = len(CODES)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name, dest=None):
    """Скачивает файл, если его нет, и сверяет SHA-256 — анализ идёт ровно на опубликованных данных."""
    dest = Path(dest or RAW / name)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"скачиваю {name} …")
        urllib.request.urlretrieve(BASE + name, dest)
    if sha256(dest) != FILES[name]:
        dest.unlink()
        raise RuntimeError(f"{name}: контрольная сумма не совпала — файл удалён")
    return dest


def penalty_matrix():
    return np.load(fetch("penalty_matrix.npy"), allow_pickle=False)


def _read(f):
    return pd.read_csv(f, sep=";", low_memory=False)


def load(split):
    """train | open | hidden. Метка — индекс класса в столбце `y`."""
    if split == "train":
        cache = RAW.parent / "train.parquet"
        if cache.exists():
            return pd.read_parquet(cache)
        with zipfile.ZipFile(fetch("train.zip")) as z:
            df = _read(io.BytesIO(z.read("train.csv")))
        df["y"] = df[TARGET].map(CODE2IDX).astype("int8")
        df.to_parquet(cache)
        return df
    if split == "open":
        df = _read(fetch("leaderboard_test_features.csv"))
        y = _read(fetch("leaderboard_test_target.csv"))
        assert (df.WELL.values == y.WELL.values).all() and np.allclose(df.DEPTH_MD, y.DEPTH_MD)
        df["y"] = y[TARGET].map(CODE2IDX).astype("int8").values
        return df
    if split == "hidden":
        df = _read(fetch("hidden_test.csv"))
        df["y"] = df[TARGET].map(CODE2IDX).astype("int8")
        return df
    raise ValueError(split)


def score(y_true, y_pred, A=None):
    """Метрика соревнования: −(средний штраф). 0 — все точки верны."""
    A = penalty_matrix() if A is None else A
    return -float(A[np.asarray(y_true, int), np.asarray(y_pred, int)].mean())
