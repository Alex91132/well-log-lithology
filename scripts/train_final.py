"""
Обучает выбранную модель на всех 98 скважинах train и сохраняет артефакт в models/.

    python scripts/train_final.py            # full + argmax — выбор по правилу из PLAN.md
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from litho import data as D  # noqa: E402
from litho import validation as V  # noqa: E402
from litho.artifact import Artifact, fit_temperature, reference_bins, temperature_scale  # noqa: E402
from litho.models import LGBM  # noqa: E402

STRIDE = 4          # как в экспериментах — модель в сервисе та же, что прошла валидацию


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", default="full")
    ap.add_argument("--decision", choices=["bayes", "argmax"], default="argmax")
    a = ap.parse_args()

    tr = D.load("train")
    m = LGBM(a.feature_set).fit(tr, train_mask=V.stride_mask(tr, STRIDE))
    lad = pd.read_csv(ROOT / "results" / "ladder.csv")
    lad = lad[(lad.key == "lgbm_" + a.feature_set) & (lad.split != "CV по точкам")]
    col = "score_bayes" if a.decision == "bayes" else "score"
    metrics = {f"{s} · {col}": round(float(v), 4) for s, v in zip(lad.split, lad[col])}
    art = Artifact.from_model(m, D.penalty_matrix(), reference_bins(tr),
                              f"1.0.0+{date.today():%Y%m%d}", metrics)
    # температура подбирается по вероятностям «вне выборки» из CV по скважинам (run_experiments.py)
    oof = np.load(ROOT / "data" / f"oof_{a.feature_set}.npy").astype(float)
    T = fit_temperature(oof, tr.y.values)
    # порог «скважина не похожа на обучающие»: 5-й перцентиль средней уверенности по скважинам CV
    conf = pd.Series(temperature_scale(oof, T).max(axis=1)).groupby(tr.WELL.values).mean()
    art.meta.update(temperature=round(T, 3), confidence_alert=round(float(conf.quantile(0.05)), 3), decision=a.decision, train_wells=int(tr.WELL.nunique()),
                    stride=STRIDE, data="FORCE 2020, CC-BY 4.0, doi:10.5281/zenodo.4351156")
    art.save(ROOT / "models")
    size = (ROOT / "models" / "model.txt.gz").stat().st_size / 1e6
    print(json.dumps(metrics, ensure_ascii=False), f"→ models/ ({size:.1f} МБ)")


if __name__ == "__main__":
    main()
