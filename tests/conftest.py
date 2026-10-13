import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from litho.features import CURVES  # noqa: E402


def make_wells(n_wells=4, n=400, seed=0):
    """Синтетические скважины: литология слоями, кривые зависят от литологии + шум."""
    rng = np.random.default_rng(seed)
    rows = []
    for w in range(n_wells):
        depth = 1000 + 0.152 * np.arange(n)
        y = np.repeat(rng.choice([0, 2, 5], n // 20 + 1, p=[0.3, 0.5, 0.2]), 20)[:n]
        df = pd.DataFrame({"WELL": f"W-{w}", "DEPTH_MD": depth, "X_LOC": 4e5 + w, "Y_LOC": 6.4e6,
                           "Z_LOC": -depth, "GROUP": np.where(depth < depth.mean(), "A GP.", "B GP."),
                           "FORMATION": "F1 Fm."})
        for c in CURVES:
            df[c] = np.nan
        df["GR"] = np.select([y == 0, y == 2], [40, 110], 20) + rng.normal(0, 8, n)
        df["RHOB"] = np.select([y == 0, y == 2], [2.3, 2.45], 2.65) + rng.normal(0, 0.03, n)
        df["RDEP"] = 10 ** (np.select([y == 0, y == 2], [1.0, 0.3], 1.5) + rng.normal(0, 0.1, n))
        df["DTC"] = np.select([y == 0, y == 2], [90, 120], 60) + rng.normal(0, 5, n)
        df["y"] = y.astype("int8")
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def wells():
    return make_wells()
