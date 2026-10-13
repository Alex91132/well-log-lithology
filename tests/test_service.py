import io

import numpy as np
import pytest
from fastapi.testclient import TestClient

from litho import service
from litho.artifact import Artifact, psi, reference_bins
from litho.data import CODES
from litho.models import LGBM

from conftest import make_wells


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    df = make_wells(6, n=1500)
    m = LGBM("full", n_estimators=30, min_child_samples=20).fit(df)
    A = np.full((12, 12), 3.0)
    np.fill_diagonal(A, 0)
    folder = tmp_path_factory.mktemp("model")
    Artifact.from_model(m, A, reference_bins(df), "test-1", {"cv_score": -0.1}).save(folder)
    service.MODEL_DIR = folder
    service.model.cache_clear()
    return TestClient(service.app)


def _csv(df):
    return df.drop(columns=["y"]).to_csv(index=False).encode()


def test_health_and_model_info(client):
    assert client.get("/health").json() == {"status": "ok", "model_version": "test-1"}
    info = client.get("/model").json()
    assert info["version"] == "test-1" and len(info["classes"]) == 12


def test_predict_csv_recovers_lithology(client):
    df = make_wells(1, n=600, seed=42)
    r = client.post("/predict", files={"file": ("w.csv", _csv(df))}, params={"proba": True})
    assert r.status_code == 200
    body = r.json()
    assert body["n_points"] == 600 and body["model_version"] == "test-1"
    pred = np.array([p["lithology"] for p in body["points"]])
    true = np.array(CODES)[df.y.values]
    assert (pred == true).mean() > 0.85
    assert abs(sum(body["points"][0]["proba"].values()) - 1) < 0.05
    assert not [w for w in body["warnings"] if "PSI" in w]


def test_predict_las_file(client):
    lasio = pytest.importorskip("lasio")
    df = make_wells(1, n=300, seed=3)
    las = lasio.LASFile()
    las.well.WELL.value = "TEST-1"
    las.append_curve("DEPT", df.DEPTH_MD.values, unit="m")
    for c in ("GR", "RHOB", "RDEP", "DTC"):
        las.append_curve(c, df[c].values)
    buf = io.StringIO()
    las.write(buf, version=2.0)
    r = client.post("/predict", files={"file": ("test.las", buf.getvalue().encode())})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["well"] == "TEST-1" and body["n_points"] == 300
    assert any("стратиграфии" in w for w in body["warnings"])


def test_unit_error_is_flagged_as_drift(client):
    df = make_wells(1, n=600, seed=5)
    df["RHOB"] *= 1000                                # кг/м³ вместо г/см³
    body = client.post("/predict", files={"file": ("w.csv", _csv(df))}).json()
    assert body["drift_psi"]["RHOB"] > 0.25
    assert any(w.startswith("RHOB") and "единицы" in w for w in body["warnings"])
    assert not any(w.startswith(("GR", "DTC")) for w in body["warnings"])


def test_json_endpoint_and_validation(client):
    df = make_wells(1, n=100, seed=7)
    body = {"well": "J-1", "depth": df.DEPTH_MD.tolist(),
            "curves": {"GR": df.GR.tolist(), "RHOB": df.RHOB.tolist()}}
    r = client.post("/predict/json", json=body)
    assert r.status_code == 200 and r.json()["well"] == "J-1"
    body["curves"]["RHOB"] = body["curves"]["RHOB"][:-1]
    assert client.post("/predict/json", json=body).status_code == 422
    del body["curves"]["GR"]
    body["curves"]["RHOB"] = df.RHOB.tolist()
    assert "GR" in client.post("/predict/json", json=body).json()["detail"]


def test_bad_file_gives_400(client):
    r = client.post("/predict", files={"file": ("x.las", b"\x00\x01 not a las")})
    assert r.status_code in (400, 422)


def test_psi_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    import pandas as pd
    from litho.features import CURVES
    df = pd.DataFrame({c: rng.normal(size=20000) for c in CURVES})
    df["WELL"] = np.repeat([f"W{i}" for i in range(10)], 2000)
    ref = reference_bins(df)
    assert psi(rng.normal(size=5000), ref["GR"]) < 0.02
    assert psi(rng.normal(1.0, 1, size=5000), ref["GR"]) > 0.25
    lo, hi = ref["GR"]["median_range"]
    assert lo < 0 < hi and hi - lo < 1           # медианы скважин N(0, 1) около нуля


def test_partial_curves_and_low_confidence_are_flagged(client):
    m = service.model()
    df = make_wells(1, n=400, seed=9)
    df.loc[df.index[:300], "RHOB"] = np.nan
    P = np.full((len(df), 12), 1 / 12)
    m.meta["confidence_alert"] = 0.5
    warnings, _ = m.check(df, P)
    assert any("RHOB — 25%" in w for w in warnings)
    assert any("уверенность" in w for w in warnings)
