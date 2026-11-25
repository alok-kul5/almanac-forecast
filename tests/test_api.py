import pytest
from fastapi.testclient import TestClient

from almanac.config import HORIZON
from almanac.models import Forecaster


@pytest.fixture
def client(panel, tmp_path, monkeypatch):
    Forecaster(num_rounds=30).fit(panel).save(tmp_path / "model")
    panel.to_parquet(tmp_path / "panel.parquet")
    monkeypatch.setenv("ALMANAC_MODEL_DIR", str(tmp_path / "model"))
    monkeypatch.setenv("ALMANAC_PANEL", str(tmp_path / "panel.parquet"))
    monkeypatch.setenv("ALMANAC_REPORTS", str(tmp_path))
    from almanac.api import app

    with TestClient(app) as c:
        yield c


def test_health_and_skus(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/skus").json() == ["SKU00", "SKU01", "SKU02", "SKU03"]


def test_forecast_shape(client):
    r = client.post("/forecast", json={"skus": ["SKU01"], "horizon": 7})
    assert r.status_code == 200
    pts = r.json()["points"]
    assert len(pts) == 7
    assert all(p["p10"] <= p["forecast"] <= p["p90"] for p in pts)
    assert all(p["forecast"] == 0 for p in pts if not p["open"])


def test_forecast_validation(client):
    assert client.post("/forecast", json={"skus": ["NOPE"]}).status_code == 404
    assert client.post("/forecast", json={"horizon": HORIZON + 1}).status_code == 422
    assert client.post("/forecast", json={"as_of": "1999-01-01"}).status_code == 422
    assert client.get("/drift").status_code == 404
