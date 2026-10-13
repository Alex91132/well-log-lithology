"""
Вопросы 1–5 из PLAN.md на train (кросс-валидация) и open test. Hidden test здесь не читается.

    python scripts/run_experiments.py        # ~30–40 минут на 8 ядрах
"""

import json
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
from litho.artifact import fit_temperature, temperature_scale  # noqa: E402
from litho.models import LGBM, Constant, LogReg  # noqa: E402

OUT = ROOT / "results"
STRIDE = 4


def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    A = D.penalty_matrix()
    tr, op = D.load("train"), D.load("open")
    print(f"train {len(tr):,} точек / {tr.WELL.nunique()} скважин; open {len(op):,} / {op.WELL.nunique()}")
    wells = V.folds(tr, "wells")
    X = {fs: build(tr, fs)[0].loc[tr.index] for fs in ("raw", "strat", "full")}
    print(f"признаки: {X['full'].shape[1]} столбцов, {time.time() - t0:.0f} с")

    ladder = [("const", Constant, None), ("logreg", LogReg, None),
              ("lgbm_raw", lambda: LGBM("raw"), X["raw"]), ("lgbm_strat", lambda: LGBM("strat"), X["strat"]),
              ("lgbm_full", lambda: LGBM("full"), X["full"])]
    rows, oof = [], {}
    sub = V.stride_mask(tr, STRIDE)
    for key, make, Xf in ladder:
        name = make().name
        print(f"\n{name} — CV по скважинам")
        P = V.oof_proba(tr, make, wells, X=Xf, stride=STRIDE)
        oof[key] = P
        m = make()
        m.fit(tr, train_mask=sub) if Xf is not None else m.fit(tr[sub])
        Po = m.predict_proba(op)
        rows.append({"key": key, "model": name, "split": "CV по скважинам", **V.metrics(tr.y.values, P, A)})
        rows.append({"key": key, "model": name, "split": "open test", **V.metrics(op.y.values, Po, A)})
        print(f"  CV {rows[-2]['score']:.4f} · open {rows[-1]['score']:.4f} · {time.time() - t0:.0f} с")
        if key == "lgbm_full":
            np.save(ROOT / "data" / "proba_open_full.npy", Po.astype(np.float32))

    print("\nLightGBM full — CV по случайным точкам (как делать не надо)")
    P_pts = V.oof_proba(tr, lambda: LGBM("full"), V.folds(tr, "points"), X=X["full"], stride=STRIDE)
    rows.append({"key": "lgbm_full", "model": LGBM("full").name, "split": "CV по точкам",
                 **V.metrics(tr.y.values, P_pts, A)})
    ladder_df = pd.DataFrame(rows)
    ladder_df.to_csv(OUT / "ladder.csv", index=False)
    print(ladder_df.round(4).to_string(index=False))

    P = oof["lgbm_full"]
    np.save(ROOT / "data" / "oof_full.npy", P)
    summary = {"n_train": len(tr), "wells_train": int(tr.WELL.nunique()), "n_open": len(op),
               "wells_open": int(op.WELL.nunique()), "n_features": int(X["full"].shape[1]),
               "stride": STRIDE, "minutes": round((time.time() - t0) / 60, 1)}
    analysis(tr, op, P, A, summary)


def analysis(tr, op, P, A, summary):
    """Разбор лучшей модели на вероятностях «вне выборки» (CV по скважинам)."""
    y = tr.y.values
    pb = P.argmax(1)             # решение по правилу из PLAN.md: штрафной выбор улучшил CV меньше чем на 0,005
    rel, ece = V.reliability(y, P)
    rel.to_csv(OUT / "reliability.csv", index=False)
    V.coverage_curve(y, P, A).to_csv(OUT / "coverage.csv", index=False)
    pw = V.per_well(tr, P, A)
    pw.to_csv(OUT / "per_well_cv.csv", index=False)
    cm = pd.crosstab(pd.Series(y, name="true"), pd.Series(pb, name="pred")).reindex(
        index=range(D.N_CLASSES), columns=range(D.N_CLASSES), fill_value=0)
    cm.to_csv(OUT / "confusion_cv.csv")
    conf = (pd.DataFrame({"interp_conf": tr[D.CONFIDENCE].values, "ok": pb == y})
            .dropna().groupby("interp_conf").agg(accuracy=("ok", "mean"), n=("ok", "size")).reset_index())
    conf.to_csv(OUT / "interpreter_confidence.csv", index=False)

    # насколько score зависит от того, какие 10 скважин попали в тест
    rng = np.random.default_rng(0)
    sub = [np.average(pw.score.values[i], weights=pw.n.values[i])
           for i in (rng.choice(len(pw), 10, replace=False) for _ in range(5000))]
    # калибровка температурой (сверх плана): T по OOF, проверка на open test
    T = fit_temperature(P.astype(float), y)
    Po = np.load(ROOT / "data" / "proba_open_full.npy").astype(float)
    summary.update(ece=ece, temperature=T, ece_cv_scaled=V.reliability(y, temperature_scale(P, T))[1],
                   ece_open=V.reliability(op.y.values, Po)[1],
                   ece_open_scaled=V.reliability(op.y.values, temperature_scale(Po, T))[1],
                   subset10_p05=float(np.quantile(sub, 0.05)), subset10_p95=float(np.quantile(sub, 0.95)),
                   halite_recall_cv=float((pb[y == 7] == 7).mean()))
    (OUT / "experiments.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(conf.to_string(index=False))


if __name__ == "__main__":
    if "--analysis-only" in sys.argv:          # пересчитать разбор по сохранённым OOF, без обучения
        A = D.penalty_matrix()
        tr, op = D.load("train"), D.load("open")
        old = json.loads((OUT / "experiments.json").read_text(encoding="utf-8"))
        keep = ("n_train", "wells_train", "n_open", "wells_open", "n_features", "stride", "minutes")
        analysis(tr, op, np.load(ROOT / "data" / "oof_full.npy"), A, {k: old[k] for k in keep})
    else:
        main()
