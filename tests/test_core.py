import numpy as np
import pandas as pd
import pytest

from litho import data as D
from litho.features import build
from litho.models import LGBM, bayes_decision
from litho.validation import folds, oof_proba, stride_mask

from conftest import make_wells


def _official_score(y_true, y_pred, A):
    """Функция из стартового ноутбука соревнования — эталон."""
    S = 0.0
    for i in range(y_true.shape[0]):
        S -= A[y_true[i], y_pred[i]]
    return S / y_true.shape[0]


def test_score_matches_official_implementation():
    rng = np.random.default_rng(0)
    A = rng.uniform(0, 4, (12, 12))
    np.fill_diagonal(A, 0)
    y, p = rng.integers(0, 12, 5000), rng.integers(0, 12, 5000)
    assert D.score(y, p, A) == pytest.approx(_official_score(y, p, A))
    assert D.score(y, y, A) == 0


def test_bayes_decision_avoids_expensive_mistake():
    # класс 0 чуть вероятнее, но ошибиться им дорого; класс 1 — дешёвый компромисс
    A = np.array([[0, 1, 4], [1, 0, 1], [4, 1, 0]], float)
    p = np.array([[0.40, 0.25, 0.35]])
    assert p.argmax() == 0
    assert bayes_decision(p, A)[0] == 1          # ожидаемый штраф: 1,65 / 0,75 / 1,85


def test_window_features_do_not_leak_between_wells():
    df = make_wells(3)
    alone = build(df[df.WELL == "W-0"], "full")[0]
    together = build(df, "full")[0].loc[alone.index]
    pd.testing.assert_frame_equal(alone.drop(columns=["GROUP", "FORMATION"]),
                                  together.drop(columns=["GROUP", "FORMATION"]))


def test_feature_order_does_not_depend_on_row_order():
    df = make_wells(2)
    a = build(df, "full")[0].sort_index()
    b = build(df.sample(frac=1, random_state=1), "full")[0].sort_index()
    pd.testing.assert_frame_equal(a.drop(columns=["GROUP", "FORMATION"]),
                                  b.drop(columns=["GROUP", "FORMATION"]))


def test_unseen_formation_becomes_missing():
    df = make_wells(1)
    X = build(df, "full", categories={"GROUP": ["A GP."], "FORMATION": ["Other Fm."]})[0]
    assert X.FORMATION.isna().all()
    assert X.GROUP.isna().sum() == (df.GROUP == "B GP.").sum()


def test_group_folds_keep_wells_together():
    df = make_wells(5)
    for tr, te in folds(df, "wells"):
        assert not set(df.WELL.iloc[tr]) & set(df.WELL.iloc[te])


def test_stride_mask_is_per_well():
    df = make_wells(2, n=10)
    m = stride_mask(df, 4)
    assert m.sum() == 2 * 3                      # точки 0, 4, 8 в каждой скважине


def test_model_learns_synthetic_lithology():
    df = make_wells(5)
    P = oof_proba(df, lambda: LGBM("full", n_estimators=50, min_child_samples=20),
                  folds(df, "wells"), X=build(df, "full")[0].loc[df.index], stride=1, log=lambda *_: None)
    assert (P.argmax(1) == df.y.values).mean() > 0.9


def test_checksum_mismatch_is_rejected(tmp_path):
    fake = tmp_path / "penalty_matrix.npy"
    fake.write_bytes(b"not the real file")
    with pytest.raises(RuntimeError, match="контрольная сумма"):
        D.fetch("penalty_matrix.npy", dest=fake)
    assert not fake.exists()
