"""Feature engineering.

Every demand feature is lagged by at least HORIZON days, so one model can
forecast any day 1..HORIZON ahead from the same cutoff with no leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import HORIZON

# England & Wales bank holidays inside the data window
UK_HOLIDAYS = pd.to_datetime([
    "2009-12-25", "2009-12-28", "2010-01-01", "2010-04-02", "2010-04-05", "2010-05-03",
    "2010-05-31", "2010-08-30", "2010-12-27", "2010-12-28", "2011-01-03", "2011-04-22",
    "2011-04-25", "2011-04-29", "2011-05-02", "2011-05-30", "2011-08-29", "2011-12-26",
    "2011-12-27", "2012-01-02",
])

LAGS = [HORIZON, HORIZON + 7, 28, 35, 364]
WINDOWS = [7, 28, 91]

CALENDAR = ["dow", "month", "weekofyear", "dayofyear", "holiday", "days_to_xmas", "open"]
LAG_FEATURES = (
    [f"lag_{lag}" for lag in LAGS]
    + [f"rmean_{w}" for w in WINDOWS]
    + [f"rstd_{w}" for w in WINDOWS[:2]]
    + ["rzero_28", "sku_mean_units", "dow_mean_8w", "dow_median_8w", "open_mean_28"]
)
DOW_LAGS = [HORIZON + 7 * k for k in range(8)]  # same weekday, 2..9 weeks back
FEATURES = ["sku_code"] + CALENDAR + LAG_FEATURES
CATEGORICAL = ["sku_code", "dow"]


def cap_outliers(panel: pd.DataFrame, caps: pd.Series | None = None, q: float = 0.99):
    """Clip one-off wholesale orders per SKU so a single bulk order doesn't swamp training.

    Returns the panel with a `units_capped` column, and the caps used (fit them on
    training data only, then reuse).
    """
    if caps is None:
        caps = panel[panel["units"] > 0].groupby("sku")["units"].quantile(q)
    out = panel.copy()
    out["units_capped"] = np.minimum(out["units"], out["sku"].map(caps).fillna(np.inf))
    return out, caps


def sku_code_map(panel: pd.DataFrame) -> dict[str, int]:
    return {s: i for i, s in enumerate(sorted(panel["sku"].unique()))}


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    d = df["date"]
    df["dow"] = d.dt.day_of_week
    df["month"] = d.dt.month
    df["weekofyear"] = d.dt.isocalendar().week.astype(int).to_numpy()
    df["dayofyear"] = d.dt.day_of_year
    df["holiday"] = d.isin(UK_HOLIDAYS).astype(int)
    xmas = pd.to_datetime(d.dt.year.astype(str) + "-12-25")
    df["days_to_xmas"] = (xmas - d).dt.days.clip(lower=-10, upper=120)
    return df


def add_features(panel: pd.DataFrame, sku_codes: dict[str, int]) -> pd.DataFrame:
    """Expects columns sku, date, open and units_capped (future rows may hold NaN units)."""
    df = panel.sort_values(["sku", "date"]).reset_index(drop=True)
    df["sku_code"] = df["sku"].map(sku_codes).fillna(-1).astype(int)
    df = add_calendar(df)

    g = df.groupby("sku", sort=False)["units_capped"]
    for lag in LAGS:
        df[f"lag_{lag}"] = g.shift(lag)
    base = g.shift(HORIZON)
    gb = base.groupby(df["sku"], sort=False)
    for w in WINDOWS:
        df[f"rmean_{w}"] = gb.transform(lambda s, w=w: s.rolling(w, min_periods=1).mean())
    for w in WINDOWS[:2]:
        df[f"rstd_{w}"] = gb.transform(lambda s, w=w: s.rolling(w, min_periods=2).std())
    df["rzero_28"] = (base == 0).astype(float).where(base.notna())
    df["rzero_28"] = df.groupby("sku", sort=False)["rzero_28"].transform(
        lambda s: s.rolling(28, min_periods=1).mean()
    )
    df["sku_mean_units"] = gb.transform(lambda s: s.expanding().mean())

    same_dow = np.column_stack([g.shift(lag).to_numpy() for lag in DOW_LAGS])
    with np.errstate(all="ignore"):
        df["dow_mean_8w"] = np.nanmean(same_dow, axis=1)
        df["dow_median_8w"] = np.nanmedian(same_dow, axis=1)
    # average over trading days only, so Saturday zeros don't drag the level down
    open_units = base.where(df.groupby("sku", sort=False)["open"].shift(HORIZON) == 1)
    df["open_mean_28"] = open_units.groupby(df["sku"], sort=False).transform(
        lambda s: s.rolling(28, min_periods=1).mean())
    return df
