"""Drift checks: input drift (PSI per feature) and accuracy drift (live WAPE vs backtest)."""
from __future__ import annotations

import numpy as np
import pandas as pd

PSI_WARN, PSI_ALERT = 0.1, 0.25
WAPE_ALERT_RATIO = 1.25  # live WAPE this much worse than the backtest WAPE triggers an alert


def psi(reference, current, bins: int = 10) -> float:
    """Population stability index, quantile bins taken from the reference sample."""
    ref = np.asarray(reference, float)
    cur = np.asarray(current, float)
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def level(value: float, warn: float, alert: float) -> str:
    if np.isnan(value):
        return "n/a"
    return "alert" if value >= alert else "warn" if value >= warn else "ok"


def feature_drift(ref: pd.DataFrame, cur: pd.DataFrame, features: list[str]) -> list[dict]:
    out = []
    for f in features:
        v = psi(ref[f], cur[f])
        out.append({"feature": f, "psi": round(v, 4), "status": level(v, PSI_WARN, PSI_ALERT)})
    return sorted(out, key=lambda d: -np.nan_to_num(d["psi"]))


def accuracy_status(live_wape: float, backtest_wape: float) -> str:
    ratio = live_wape / backtest_wape if backtest_wape else float("nan")
    return level(ratio, 1.10, WAPE_ALERT_RATIO)
