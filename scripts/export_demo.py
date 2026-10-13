"""
Данные для страницы docs/: результаты экспериментов, финальная оценка и каротаж скрытых скважин
с прогнозом модели (каждая 4-я точка ≈ 60 см).

    python scripts/export_demo.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from litho import data as D  # noqa: E402

R = ROOT / "results"
STEP = 4


def _r(a, nd):
    a = np.round(np.asarray(a, float), nd)
    return [None if np.isnan(v) else (int(v) if nd == 0 else float(v)) for v in a]


CONCLUSION = ("Гипотеза не подтвердилась: без свиты модель хуже и в целом, и на галите, а регуляризация категорий "
              "меняет CV в пределах шума — по правилу плана модель не меняется. Причина не в переобучении, "
              "а в данных: значимых примеров галита — две скважины, и никакая настройка этого не исправит. "
              "Правильный ответ — не прятать проблему, а помечать такие скважины: сервис предупреждает о низкой "
              "уверенности модели.")


def api_example():
    """Настоящий ответ сервиса на примере скважины из открытого теста (точки сокращены до двух)."""
    from fastapi.testclient import TestClient

    from litho import service
    las = ROOT / "examples" / "well_25_11-24.las"
    r = TestClient(service.app).post("/predict", files={"file": (las.name, las.read_bytes())}).json()
    r["points"] = r["points"][:2] + ["…"]
    r["summary"] = {k: v for k, v in list(r["summary"].items())[:4]}
    r["drift_psi"] = {k: v for k, v in list(r["drift_psi"].items())[:4]}
    return r


def test_well_warnings(A, hd, P_hidden):
    """Какие из 20 тестовых скважин сервис пометил бы — и насколько они плохи на самом деле."""
    from litho.artifact import Artifact
    art = Artifact.load(ROOT / "models")
    op = D.load("open")
    rows = []
    for split, df, P in (("открытый", op, art.proba(op)), ("скрытый", hd, P_hidden)):
        pred = P.argmax(1)
        for w, g in df.groupby("WELL"):
            i = g.index.values
            warn, _ = art.check(g, P[i])
            rows.append({"well": w, "split": split, "score": -float(A[df.y.values[i], pred[i]].mean()),
                         "warnings": warn})
    return sorted(rows, key=lambda r: r["score"])


def main():
    A = D.penalty_matrix()
    hd = D.load("hidden")
    P = np.load(ROOT / "data" / "proba_hidden.npy")
    pred = P.argmax(1)                                   # решение финальной модели (meta.json: argmax)
    conf = P[np.arange(len(P)), pred]
    wells = []
    for w, g in hd.groupby("WELL", sort=True):
        g = g.sort_values("DEPTH_MD")
        i = g.index.values[::STEP]
        wells.append({"well": w, "depth": _r(hd.DEPTH_MD[i], 1), "GR": _r(hd.GR[i], 1),
                      "RHOB": _r(hd.RHOB[i], 3), "NPHI": _r(hd.NPHI[i], 3), "RDEP": _r(hd.RDEP[i], 2),
                      "DTC": _r(hd.DTC[i], 1), "true": hd.y.values[i].tolist(), "pred": pred[i].tolist(),
                      "conf": _r(conf[i], 2), "group": hd.GROUP.fillna("").values[i].tolist()})
    pw = (pd.DataFrame({"well": hd.WELL, "pen": A[hd.y.values, pred], "ok": pred == hd.y.values})
          .groupby("well").agg(score=("pen", lambda x: -x.mean()), accuracy=("ok", "mean")).reset_index())
    worst = pw.sort_values("score").well.iloc[0]
    w = (hd.WELL == worst).values
    hal = w & (hd.y.values == D.CODE2IDX[88000])
    as_shale = hal & (pred == D.CODE2IDX[65000])
    tr = D.load("train")
    info = {"worst": worst, "score_without_worst": -float(A[hd.y.values[~w], pred[~w]].mean()),
            "worst_zechstein_share": float((hd.GROUP[w] == "ZECHSTEIN GP.").mean()),
            "halite_train_wells": int(tr[tr.y == D.CODE2IDX[88000]].WELL.nunique()),
            "halite_as_shale": int(as_shale.sum()), "halite_gr_median": round(float(hd.GR[as_shale].median())),
            "halite_rhob_share": float(hd.RHOB[as_shale].notna().mean()), "conclusion": CONCLUSION}
    out = {
        "names": D.NAMES,
        "ladder": pd.read_csv(R / "ladder.csv").to_dict("records"),
        "experiments": json.loads((R / "experiments.json").read_text(encoding="utf-8")),
        "hidden": json.loads((R / "hidden.json").read_text(encoding="utf-8")),
        "per_well_hidden": pw.to_dict("records"),
        "posthoc": pd.read_csv(R / "posthoc.csv").to_dict("records") if (R / "posthoc.csv").exists() else [],
        "posthoc_info": info,
        "api_example": api_example(),
        "warnings": test_well_warnings(A, hd, P),
        "reliability": pd.read_csv(R / "reliability.csv").to_dict("records"),
        "coverage": pd.read_csv(R / "coverage.csv").to_dict("records"),
        "interp": pd.read_csv(R / "interpreter_confidence.csv").to_dict("records"),
        "confusion": pd.read_csv(R / "confusion_cv.csv", index_col=0).values.tolist(),
        "wells": wells,
    }
    (ROOT / "docs").mkdir(exist_ok=True)
    js = "window.DATA = " + json.dumps(out, ensure_ascii=False, separators=(",", ":")) + ";\n"
    (ROOT / "docs" / "data.js").write_text(js, encoding="utf-8")
    print(f"docs/data.js — {len(js) / 1e6:.1f} МБ, {sum(len(w['depth']) for w in wells):,} точек")


if __name__ == "__main__":
    main()
