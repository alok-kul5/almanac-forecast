import numpy as np
import pandas as pd

from almanac.backtest import run_backtest, summarize
from almanac.config import HORIZON
from almanac.data import build_panel, clean_lines
from almanac.drift import accuracy_status, psi
from almanac.features import FEATURES, LAG_FEATURES, add_features, cap_outliers, sku_code_map
from almanac.metrics import bias, coverage, wape
from almanac.models import Forecaster, future_frame, moving_average, seasonal_naive


def test_clean_lines_drops_cancellations_and_fees():
    raw = pd.DataFrame({
        "Invoice": ["1", "C2", "3", "4", "5"],
        "StockCode": ["85123A", "85123A", "POST", "22138", "22138"],
        "Quantity": [5, -5, 1, 3, -1],
        "InvoiceDate": ["2010-01-04 10:00"] * 5,
        "Price": [2.5, 2.5, 18, 4.95, 4.95],
    })
    out = clean_lines(raw)
    assert list(out["StockCode"]) == ["85123A", "22138"]


def test_build_panel_zero_fills_missing_days():
    lines = pd.DataFrame({
        "StockCode": ["11111", "11111"],
        "Quantity": [3, 4],
        "date": pd.to_datetime(["2010-01-01", "2010-01-03"]),
    })
    p = build_panel(lines, top_n=5)
    assert len(p) == 3
    assert p["units"].tolist() == [3, 0, 4]
    assert p["open"].tolist() == [1, 0, 1]


def test_features_never_see_the_forecast_window(panel):
    """Changing demand inside the forecast window must not change its features."""
    cut = panel["date"].max() - pd.Timedelta(days=HORIZON)
    capped, _ = cap_outliers(panel)
    codes = sku_code_map(panel)
    a = add_features(capped, codes)
    tampered = capped.copy()
    tampered.loc[tampered["date"] > cut, "units_capped"] = 9999
    b = add_features(tampered, codes)
    win = a["date"] > cut
    lag_cols = [c for c in FEATURES if c in LAG_FEATURES]
    assert "dow_mean_8w" in lag_cols and "open_mean_28" in lag_cols
    pd.testing.assert_frame_equal(a.loc[win, lag_cols], b.loc[win, lag_cols])


def test_metrics():
    y = np.array([10, 0, 10])
    assert wape(y, y) == 0
    assert wape(y, [0, 0, 0]) == 1
    assert bias(y, [11, 0, 11]) == 0.1
    assert coverage(y, [9, 0, 11], [11, 1, 12]) == 2 / 3


def test_psi_detects_shift():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 5000)
    assert psi(ref, rng.normal(0, 1, 5000)) < 0.05
    assert psi(ref, rng.normal(1.5, 1, 5000)) > 0.25
    assert accuracy_status(0.5, 0.5) == "ok"
    assert accuracy_status(0.7, 0.5) == "alert"


def test_baselines_respect_closed_days(panel):
    cut = panel["date"].max() - pd.Timedelta(days=HORIZON)
    hist = panel[panel["date"] <= cut]
    fut = future_frame(hist, cut)
    for f in (seasonal_naive, moving_average):
        pred = f(hist, fut)
        assert len(pred) == len(fut)
        assert (pred[fut["open"].to_numpy() == 0] == 0).all()


def test_forecaster_beats_noise_and_round_trips(panel, tmp_path):
    cut = panel["date"].max() - pd.Timedelta(days=HORIZON)
    hist = panel[panel["date"] <= cut]
    truth = panel[panel["date"] > cut]
    model = Forecaster(num_rounds=80).fit(hist)
    fut = future_frame(hist, cut, open_flags=panel[["date", "open"]].drop_duplicates("date"))
    pred = model.predict(hist, fut).merge(truth[["sku", "date", "units"]], on=["sku", "date"])
    assert len(pred) == 4 * HORIZON
    assert (pred["q10"] <= pred["yhat"]).all() and (pred["yhat"] <= pred["q90"]).all()
    assert wape(pred["units"], pred["yhat"]) < 0.35

    model.save(tmp_path / "m")
    again = Forecaster.load(tmp_path / "m").predict(hist, fut)
    np.testing.assert_allclose(again["yhat"], model.predict(hist, fut)["yhat"])


def test_backtest_runs(panel):
    preds, folds = run_backtest(panel, n_folds=2, num_rounds=40, log=lambda *_: None)
    assert len(folds) == 2
    s = summarize(preds)
    assert set(s["wape_by_horizon"]) == set(range(1, HORIZON + 1))


def test_monitor_flags_a_demand_shift():
    from almanac.monitor import replay

    from .conftest import make_panel

    clean = replay(make_panel(n_skus=20), weeks=4, params={"num_rounds": 150})
    assert all(w["status"] == "ok" for w in clean["weeks"]), "no false alarms on stable demand"

    shifted = make_panel(n_skus=20)
    late = shifted["date"] > shifted["date"].max() - pd.Timedelta(days=28)
    shifted.loc[late, "units"] *= 4
    rep = replay(shifted, weeks=4, params={"num_rounds": 150})
    assert len(rep["weeks"]) == 4
    assert rep["weeks"][0]["features_alerting"] < 3
    assert rep["weeks"][-1]["status"] == "alert"
    assert rep["recommendation"] == "retrain"


def test_build_panel_drops_partial_last_day():
    lines = pd.DataFrame({
        "StockCode": ["11111"] * 3,
        "Quantity": [3, 4, 50],
        "InvoiceDate": ["2010-01-01 17:00", "2010-01-02 16:30", "2010-01-03 11:00"],
    })
    lines["date"] = pd.to_datetime(lines["InvoiceDate"]).dt.normalize()
    p = build_panel(lines, top_n=5)
    assert p["date"].max() == pd.Timestamp("2010-01-02")
