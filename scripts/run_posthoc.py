"""
Сверх плана, ПОСЛЕ финальной оценки на hidden test. Гипотеза из разбора скважины 17/4-1:
модель переобучается на категориальный признак FORMATION (галит в свите Smith Bank, которая
в большинстве скважин — глина). Варианты сравниваются только на CV по скважинам и open test.

    python scripts/run_posthoc.py        # ~25 минут
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from litho import data as D  # noqa: E402
from litho import validation as V  # noqa: E402
from litho.features import build  # noqa: E402
from litho.models import LGBM  # noqa: E402

STRIDE = 4
CAT_REG = dict(cat_smooth=50, min_data_per_group=500, cat_l2=10, max_cat_threshold=16)


class NoFormation(LGBM):
    def __init__(self, **kw):
        super().__init__("full", **kw)
        self.name = "без FORMATION"

    def fit_features(self, X, y):
        return super().fit_features(X.drop(columns="FORMATION"), y)


VARIANTS = {
    "v1": ("v1: как в финале", lambda: LGBM("full")),
    "no_formation": ("без FORMATION", NoFormation),
    "cat_reg": ("регуляризация категорий", lambda: LGBM("full", **CAT_REG)),
}


def halite_recall(y, P):
    m = y == D.CODE2IDX[88000]
    return float((P[m].argmax(1) == y[m]).mean())


def main():
    t0 = time.time()
    A = D.penalty_matrix()
    tr, op = D.load("train"), D.load("open")
    X = build(tr, "full")[0].loc[tr.index]
    sub = V.stride_mask(tr, STRIDE)
    rows = []
    for key, (name, make) in VARIANTS.items():
        if key == "v1":
            P = np.load(ROOT / "data" / "oof_full.npy")
        else:
            P = V.oof_proba(tr, make, V.folds(tr, "wells"), X=X, stride=STRIDE)
            np.save(ROOT / "data" / f"oof_{key}.npy", P)
        m = make().fit(tr, train_mask=sub)
        Po = m.predict_proba(op)
        np.save(ROOT / "data" / f"proba_open_{key}.npy", Po.astype(np.float32))
        for split, yy, PP in (("CV по скважинам", tr.y.values, P), ("open test", op.y.values, Po)):
            rows.append({"key": key, "variant": name, "split": split, **V.metrics(yy, PP, A),
                         "halite_recall": halite_recall(yy, PP) if (yy == 7).any() else np.nan})
        print(pd.DataFrame(rows[-2:]).round(4).to_string(index=False), f"· {time.time() - t0:.0f} с")
    pd.DataFrame(rows).to_csv(ROOT / "results" / "posthoc.csv", index=False)


if __name__ == "__main__":
    main()
