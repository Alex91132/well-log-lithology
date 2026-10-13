"""
Единственная оценка на 10 скрытых скважинах финала — после того, как модель выбрана (PLAN.md).
Повторный запуск без --force отказывается перезаписывать результат.

    python scripts/evaluate_hidden.py
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from litho import data as D  # noqa: E402
from litho import validation as V  # noqa: E402
from litho.artifact import Artifact  # noqa: E402

# итоговая таблица соревнования (final test score), README репозитория FORCE 2020
FINALISTS = [("Olawale Ibrahim", -0.4690), ("GIR Team", -0.4792), ("Lab.ICA-Team / Smith A.", -0.4954),
             ("H3G", -0.5045), ("ISPL Team", -0.5084), ("Jiampiers C.", -0.5087), ("José Bermúdez", -0.5091),
             ("Bohdan Pavlyshenko", -0.5171), ("Jeremy Zhao", -0.5173), ("Campbell Hutcheson", -0.5221),
             ("David P.", -0.5256), ("SoftServe Team", -0.5263), ("Dapo Awolayo", -0.9441)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    out = ROOT / "results" / "hidden.json"
    if out.exists() and not ap.parse_args().force:
        sys.exit(f"{out.name} уже есть — оценка на скрытых скважинах делается один раз")

    art = Artifact.load(ROOT / "models")
    hd = D.load("hidden")
    pred, P = art.predict(hd)
    y = hd.y.values
    k = np.vectorize(D.CODE2IDX.get)(pred.lithology.values)
    A = np.asarray(art.A)
    res = {"model_version": art.meta["version"], "decision": art.meta.get("decision"),
           "score": D.score(y, k, A), "accuracy": float((k == y).mean()),
           "all_metrics": V.metrics(y, P, A)}
    better = [n for n, s in FINALISTS if s > res["score"]]
    res["rank_among_finalists"] = len(better) + 1
    res["finalists"] = FINALISTS
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    np.save(ROOT / "data" / "proba_hidden.npy", P.astype(np.float32))
    V.per_well(hd, P, A, art.meta.get("decision", "argmax")).to_csv(ROOT / "results" / "per_well_hidden.csv", index=False)
    print(f"hidden: score {res['score']:.4f}, accuracy {res['accuracy']:.3f} → "
          f"место {res['rank_among_finalists']} из {len(FINALISTS) + 1}")


if __name__ == "__main__":
    main()
