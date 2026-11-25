"""Rolling-origin backtest: refit at each cutoff, forecast the next HORIZON days."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import HORIZON
from .metrics import bias, coverage, mase, wape
from .models import Forecaster, future_frame, moving_average, seasonal_naive


def cutoffs(panel: pd.DataFrame, n_folds: int, step: int = HORIZON, horizon: int = HORIZON):
    last = panel["date"].max() - pd.Timedelta(days=horizon)
    return [last - pd.Timedelta(days=step * i) for i in reversed(range(n_folds))]


def run_backtest(panel: pd.DataFrame, n_folds: int = 12, params: dict | None = None,
                 num_rounds: int | None = None, log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per-row forecasts, per-fold metrics) for LightGBM and two baselines."""
    rows, folds = [], []
    calendar = panel[["date", "open"]].drop_duplicates("date")
    for i, cut in enumerate(cutoffs(panel, n_folds)):
        hist = panel[panel["date"] <= cut]
        truth = panel[(panel["date"] > cut) & (panel["date"] <= cut + pd.Timedelta(days=HORIZON))]
        fut = future_frame(hist, cut, open_flags=calendar)

        kw = {"num_rounds": num_rounds} if num_rounds else {}
        model = Forecaster(params, **kw).fit(hist)
        pred = model.predict(hist, fut)
        pred["seasonal_naive"] = seasonal_naive(hist, fut)
        pred["moving_avg"] = moving_average(hist, fut)
        pred = pred.merge(truth[["sku", "date", "units"]], on=["sku", "date"], how="left")
        pred["cutoff"] = cut
        pred["h"] = (pred["date"] - cut).dt.days
        rows.append(pred)

        y = pred["units"].to_numpy()
        y_train = hist.sort_values(["sku", "date"])["units"].to_numpy()
        fold = {"fold": i, "cutoff": cut.date().isoformat()}
        for name, col in [("lgbm", "yhat"), ("seasonal_naive", "seasonal_naive"), ("moving_avg", "moving_avg")]:
            fold[f"{name}_wape"] = wape(y, pred[col])
            fold[f"{name}_bias"] = bias(y, pred[col])
            fold[f"{name}_mase"] = mase(y, pred[col], y_train)
        fold["lgbm_p10_p90_coverage"] = coverage(y, pred["q10"], pred["q90"])
        folds.append(fold)
        log(f"fold {i:2d} cutoff {fold['cutoff']}  lgbm {fold['lgbm_wape']:.3f}  "
            f"snaive {fold['seasonal_naive_wape']:.3f}  ma28 {fold['moving_avg_wape']:.3f}")
    return pd.concat(rows, ignore_index=True), pd.DataFrame(folds)


def summarize(preds: pd.DataFrame) -> dict:
    """Pooled metrics over every backtest row."""
    y = preds["units"]
    out = {}
    for name, col in [("lgbm", "yhat"), ("seasonal_naive", "seasonal_naive"), ("moving_avg", "moving_avg")]:
        out[f"{name}_wape"] = wape(y, preds[col])
        out[f"{name}_bias"] = bias(y, preds[col])
    out["lgbm_p10_p90_coverage"] = coverage(y, preds["q10"], preds["q90"])
    out["lgbm_vs_best_baseline"] = 1 - out["lgbm_wape"] / min(out["seasonal_naive_wape"], out["moving_avg_wape"])
    by_h = preds.groupby("h").apply(lambda d: wape(d["units"], d["yhat"]), include_groups=False)
    # NaN where a horizon only ever lands on closed days (no demand to score), stored as null
    out["wape_by_horizon"] = {int(k): (None if np.isnan(v) else float(v)) for k, v in by_h.items()}
    out["n_rows"] = int(len(preds))
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in out.items()}
