"""Replay the last weeks of data as if the model were live, and check for drift.

    python -m almanac.monitor [--weeks 8]

A model is frozen `weeks` weeks before the end of the data. Each replayed week
it forecasts the next HORIZON days from that week's start, then once the week's
actuals land we score it and compare its feature distribution against the
training reference window. Alerts recommend a retrain.
"""
from __future__ import annotations

import argparse
import json

import mlflow
import numpy as np
import pandas as pd

from .config import EXPERIMENT, HORIZON, PANEL, REPORTS, TRACKING_URI
from .drift import accuracy_status, calibrate_thresholds, feature_drift
from .features import CALENDAR
from .metrics import bias, wape
from .models import Forecaster, future_frame, moving_average

# Fast-moving demand signals, scaled by each SKU's reference level so SKUs of very
# different sizes are comparable. Slow features (91-day means, expanding means) are
# left out: inside a short window they barely move, so PSI flags them on clean data.
LEVEL_FEATURES = ["lag_14", "lag_21", "lag_28", "rmean_7", "dow_mean_8w", "dow_median_8w"]
WATCHED = LEVEL_FEATURES + ["rstd_7", "rzero_28"]
REFERENCE_DAYS = 182


def _scaled(df: pd.DataFrame, level: pd.Series) -> pd.DataFrame:
    out = df.copy()
    lv = out["sku"].map(level).replace(0, np.nan)
    for f in LEVEL_FEATURES + ["rstd_7"]:
        out[f] = out[f] / lv
    return out


def replay(panel: pd.DataFrame, weeks: int = 8, params: dict | None = None) -> dict:
    end = panel["date"].max()
    freeze = end - pd.Timedelta(days=7 * weeks)
    train = panel[panel["date"] <= freeze]
    model = Forecaster(params).fit(train)
    calendar = panel[["date", "open"]].drop_duplicates("date")

    design = model._design(panel)
    ref = design[(design["date"] > freeze - pd.Timedelta(days=REFERENCE_DAYS))
                 & (design["date"] <= freeze) & (design["open"] == 1)]
    level = ref.groupby("sku")["units_capped"].mean()
    ref = _scaled(ref, level)
    thresholds = calibrate_thresholds(ref, WATCHED)

    # what "normal" error looks like: the frozen model's own fit window, scored like a backtest
    bt_wape = _reference_wape(train, params)

    out = []
    for k in range(weeks):
        start = freeze + pd.Timedelta(days=7 * k)
        hist = panel[panel["date"] <= start]
        fut = future_frame(hist, start, HORIZON, open_flags=calendar)
        pred = model.predict(hist, fut)
        week = pred[pred["date"] <= start + pd.Timedelta(days=7)].merge(
            panel[["sku", "date", "units"]], on=["sku", "date"])
        ma = moving_average(hist, fut)[: len(fut)]
        fut = fut.assign(ma=ma)
        week = week.merge(fut[["sku", "date", "ma"]], on=["sku", "date"])

        # trailing 14 days: one week of 50 SKUs is too few rows for a stable PSI
        cur = design[(design["date"] > start - pd.Timedelta(days=7))
                     & (design["date"] <= start + pd.Timedelta(days=7)) & (design["open"] == 1)]
        cur = _scaled(cur, level)
        drift = feature_drift(ref, cur, WATCHED, thresholds)
        live = wape(week["units"], week["yhat"])
        n_alert = sum(d["status"] == "alert" for d in drift)
        acc = accuracy_status(live, bt_wape)
        status = "alert" if acc == "alert" or n_alert >= 3 else (
            "warn" if acc == "warn" or n_alert >= 1 else "ok")
        out.append({
            "week_start": (start + pd.Timedelta(days=1)).date().isoformat(),
            "units": float(week["units"].sum()),
            "forecast": round(float(week["yhat"].sum()), 1),
            "wape": round(live, 4),
            "baseline_wape": round(wape(week["units"], week["ma"]), 4),
            "bias": round(bias(week["units"], week["yhat"]), 4),
            "accuracy_status": acc,
            "features_alerting": n_alert,
            "top_drift": drift[:5],
            "status": status,
        })

    latest = out[-1]
    return {
        "frozen_at": freeze.date().isoformat(),
        "reference_window_days": REFERENCE_DAYS,
        "reference_wape": round(bt_wape, 4),
        "weeks": out,
        "latest_status": latest["status"],
        "first_alert": next((w["week_start"] for w in out if w["status"] == "alert"), None),
        "recommendation": "retrain" if latest["status"] == "alert" else "keep",
        "watched_features": WATCHED,
        "calendar_features_not_watched": CALENDAR,
    }


def _reference_wape(train: pd.DataFrame, params: dict | None, folds: int = 4) -> float:
    from .backtest import run_backtest

    preds, _ = run_backtest(train, n_folds=folds, params=params, log=lambda *_: None)
    return float(wape(preds["units"], preds["yhat"]))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weeks", type=int, default=8)
    args = ap.parse_args(argv)
    panel = pd.read_parquet(PANEL)
    report = replay(panel, args.weeks)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="monitor-replay"):
        mlflow.log_param("frozen_at", report["frozen_at"])
        for i, w in enumerate(report["weeks"]):
            mlflow.log_metrics({"live_wape": w["wape"], "live_bias": w["bias"],
                                "features_alerting": w["features_alerting"]}, step=i)
        mlflow.set_tag("recommendation", report["recommendation"])

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "monitoring.json").write_text(json.dumps(report, indent=2, default=_np))
    for w in report["weeks"]:
        print(f"{w['week_start']}  wape {w['wape']:.3f} (ma28 {w['baseline_wape']:.3f})  "
              f"drifting {w['features_alerting']:2d}  {w['status']}")
    print(f"reference wape {report['reference_wape']:.3f}; recommendation: {report['recommendation']}")


def _np(o):
    return o.item() if isinstance(o, np.generic) else str(o)


if __name__ == "__main__":
    main()
