<p align="left"><img src="docs/brand/mark.svg" width="64" alt="Almanac"></p>

# Almanac

Daily demand forecasts for 50 products of a UK online retailer, 14 days ahead, with P10–P90 bands.
Everything needed to keep a model useful after launch is included: rolling backtests against honest
baselines, MLflow tracking, a registry that only promotes models that win, a FastAPI service, and drift
monitoring that knows the difference between noise and a real shift.

**Report:** https://alok-kul5.github.io/almanac-forecast/

[![ci](https://github.com/alok-kul5/almanac-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/alok-kul5/almanac-forecast/actions/workflows/ci.yml)

## Results

12 rolling folds, one every 14 days from 2011-06-23 to 2011-11-24 (8,400 SKU-days). Each fold refits on
data before its cutoff only.

| Model | WAPE | Bias |
|---|---:|---:|
| **LightGBM (Tweedie, global)** | **0.786** | −11.5% |
| 28-day moving average | 0.841 | −4.4% |
| Same weekday last week | 1.034 | −3.8% |

LightGBM wins 10 of 12 folds and cuts error by 6.5% against the best baseline. Its P10–P90 band
holds 78.8% of actual days (the aim is 80%).

WAPE is high in absolute terms because this is wholesale-heavy, lumpy SKU-day data: one customer can
order 1,000 units on an otherwise quiet day. That's why the comparison against baselines matters more
than the raw number.

### Choosing the objective

| Objective | WAPE | Bias | Notes |
|---|---:|---:|---|
| L1 / Huber | 0.71 | −32% | Best WAPE, but it forecasts the median, so it would leave shelves empty |
| Tweedie, 99th pct cap | 0.82 | +4% | Unbiased, but no better than the moving average |
| **Tweedie, 95th pct cap** | **0.77** | **−8%** | Chosen: real gain with small bias |

(6-fold sweep; the full 12-fold numbers are above.) The cap clips one-off bulk orders in the training
target only. Evaluation always uses the raw actuals.

## How it works

```
UCI Online Retail II (1.07M invoice lines)
  └─ clean: drop cancellations, returns, postage/fee codes, the partial last day
     └─ panel: 50 SKUs x 738 days, zero-filled, shop-open flag
        └─ features: lags >= 14 days, same-weekday means, rolling stats, UK holidays, days to Christmas
           ├─ backtest: 12 folds vs moving average and seasonal naive  ─┐
           ├─ train: Tweedie point model + P10/P90 quantile models      ├─ MLflow (sqlite) runs + registry
           │                    champion alias only if it beats baseline ┘
           ├─ serve: FastAPI /forecast, /skus, /health, /drift
           └─ monitor: replay 8 weeks as live -> live WAPE + calibrated PSI -> keep / retrain
```

- **No leakage, one model for the whole horizon.** Every demand feature is lagged at least 14 days,
  so the same features are available for day +1 and day +14. A test changes demand inside the forecast
  window and checks that no feature moves.
- **Gated registry.** `python -m almanac.train` logs params, per-fold metrics and artifacts, registers
  a new version, and gives it the `champion` alias only if it beats the best baseline. Otherwise it's
  tagged `challenger`.
- **Drift that doesn't cry wolf.** Smooth time-series features give a high PSI for any short window,
  so a fixed 0.25 cut-off raises alarms on clean data (I hit exactly that). The monitor scales demand
  features by each SKU's normal level, then calibrates each feature's threshold from 14-day slices of
  the reference window itself. A test checks for no false alarms on stable data and for an alert after
  a 4x demand shift.

### Monitoring replay

The model is frozen on 2011-10-13 and the next 8 weeks are replayed as if live:

| Week of | Live WAPE | 28-day avg | Bias | Features drifting | Status |
|---|---:|---:|---:|---:|---|
| 2011-10-14 | 0.763 | 0.880 | −7.4% | 0 | ok |
| 2011-10-21 | 0.965 | 1.150 | +22.4% | 0 | alert |
| 2011-10-28 | 0.789 | 0.803 | −30.5% | 0 | ok |
| 2011-11-04 | 0.710 | 0.709 | −34.4% | 0 | ok |
| 2011-11-11 | 0.681 | 0.699 | −30.4% | 0 | ok |
| 2011-11-18 | 0.680 | 0.629 | −21.4% | 0 | ok |
| 2011-11-25 | 0.712 | 0.784 | +8.7% | 0 | ok |
| 2011-12-02 | 0.870 | 0.736 | +13.2% | 2 | warn |

Reference WAPE is 0.771, and the accuracy alert fires at 1.25x that. The one alert (week of 2011-10-21) was a week when the moving average broke down too (WAPE 1.15): a demand spike, not a broken model, so the monitor recovered on its own the next week. The December warning is the Christmas run-up showing in `rmean_7` and the lag features. Recommendation at the end of the replay: **keep**.

## Run it

Python 3.11. Everything runs locally and for free.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
python scripts/fetch_data.py      # downloads UCI Online Retail II, builds data/processed/panel.parquet
python -m almanac.train           # backtest + train + register (about 2 minutes on a laptop)
python -m almanac.monitor         # 8-week drift replay -> reports/monitoring.json
uvicorn almanac.api:app           # http://127.0.0.1:8000/docs
mlflow ui --backend-store-uri sqlite:///mlruns.db
pytest -q
```

```bash
curl -X POST localhost:8000/forecast -H "content-type: application/json" \
     -d '{"skus": ["85123A"], "horizon": 7}'
```

Docker: `docker build -t almanac .`, then mount `models/`, `data/` and `reports/` (see the Dockerfile).

## Layout

```
almanac/
  data.py       invoice lines -> clean daily panel
  features.py   leakage-safe features
  models.py     baselines + LightGBM forecaster (save/load)
  backtest.py   rolling-origin evaluation
  metrics.py    WAPE, bias, MASE, interval coverage
  train.py      MLflow tracking + gated registry
  drift.py      PSI with self-calibrated thresholds
  monitor.py    live replay, keep/retrain decision
  api.py        FastAPI service
docs/           GitHub Pages report (reads docs/data/*.json)
tests/          pytest on synthetic data, so CI needs no download
```

## Data & licence

[Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii), UCI Machine Learning
Repository, by Daqing Chen (CC BY 4.0). Code is MIT licensed.
