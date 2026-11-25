"""Forecast error metrics."""
from __future__ import annotations

import numpy as np


def wape(y, yhat) -> float:
    """Weighted absolute percentage error: sum|e| / sum|y|. Robust to zero days."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    denom = np.abs(y).sum()
    return float(np.abs(y - yhat).sum() / denom) if denom else float("nan")


def bias(y, yhat) -> float:
    """Signed share of over (+) or under (-) forecasting."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    denom = np.abs(y).sum()
    return float((yhat - y).sum() / denom) if denom else float("nan")


def mase(y, yhat, y_train, season: int = 7) -> float:
    """MAE scaled by the in-sample seasonal-naive MAE."""
    y, yhat, y_train = (np.asarray(a, float) for a in (y, yhat, y_train))
    scale = np.mean(np.abs(y_train[season:] - y_train[:-season]))
    return float(np.mean(np.abs(y - yhat)) / scale) if scale else float("nan")


def coverage(y, lo, hi) -> float:
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    return float(np.mean((y >= lo) & (y <= hi)))
