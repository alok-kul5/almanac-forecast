"""Baselines and the LightGBM forecaster."""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import HORIZON, QUANTILES
from .features import CATEGORICAL, FEATURES, add_features, cap_outliers, sku_code_map

DEFAULT_PARAMS = {
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 7,
}
NUM_ROUNDS = 600


def future_frame(history: pd.DataFrame, cutoff: pd.Timestamp, horizon: int = HORIZON,
                 open_flags: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rows for cutoff+1 .. cutoff+horizon for every SKU.

    `open_flags` (date, open) gives the trading calendar; without it we assume the
    shop's usual pattern: closed on Saturdays and over Christmas to New Year.
    """
    dates = pd.date_range(cutoff + pd.Timedelta(days=1), periods=horizon, freq="D")
    skus = sorted(history["sku"].unique())
    fut = pd.MultiIndex.from_product([skus, dates], names=["sku", "date"]).to_frame(index=False)
    if open_flags is not None:
        fut = fut.merge(open_flags, on="date", how="left")
        fut["open"] = fut["open"].fillna(1).astype(int)
    else:
        d = fut["date"]
        closed = (d.dt.day_of_week == 5) | ((d.dt.month == 12) & (d.dt.day >= 24)) | (
            (d.dt.month == 1) & (d.dt.day <= 3))
        fut["open"] = (~closed).astype(int)
    fut["units"] = np.nan
    return fut


# ---------------------------------------------------------------- baselines

def seasonal_naive(history: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
    """Same weekday in the most recent observed week."""
    cutoff = history["date"].max()
    lookup = history.set_index(["sku", "date"])["units"]
    k = np.ceil((future["date"] - cutoff).dt.days / 7).astype(int)
    ref = future["date"] - pd.to_timedelta(7 * k, unit="D")
    vals = lookup.reindex(pd.MultiIndex.from_arrays([future["sku"], ref])).to_numpy()
    return np.nan_to_num(vals) * future["open"].to_numpy()


def moving_average(history: pd.DataFrame, future: pd.DataFrame, window: int = 28) -> np.ndarray:
    """Mean of the last `window` trading days, zero on closed days."""
    cutoff = history["date"].max()
    recent = history[(history["date"] > cutoff - pd.Timedelta(days=window)) & (history["open"] == 1)]
    means = recent.groupby("sku")["units"].mean()
    return future["sku"].map(means).fillna(0).to_numpy() * future["open"].to_numpy()


# ---------------------------------------------------------------- LightGBM

class Forecaster:
    """Global LightGBM model across SKUs: a Tweedie point model plus P10/P90 quantile models."""

    def __init__(self, params: dict | None = None, num_rounds: int = NUM_ROUNDS,
                 quantiles: tuple[float, ...] = QUANTILES):
        params = {**DEFAULT_PARAMS, **(params or {})}
        # 0.95 picked in the objective sweep (see README): biggest WAPE gain without starving stock
        self.cap_q = params.pop("cap_q", 0.95)
        self.params = params
        self.num_rounds = num_rounds
        self.quantiles = quantiles
        self.boosters: dict[str, lgb.Booster] = {}
        self.sku_codes: dict[str, int] = {}
        self.caps: pd.Series | None = None
        self.train_end: pd.Timestamp | None = None

    def _design(self, panel: pd.DataFrame) -> pd.DataFrame:
        capped, _ = cap_outliers(panel, self.caps)
        return add_features(capped, self.sku_codes)

    def fit(self, train: pd.DataFrame) -> Forecaster:
        self.sku_codes = sku_code_map(train)
        _, self.caps = cap_outliers(train, q=self.cap_q)
        self.train_end = train["date"].max()
        df = self._design(train)
        # skip a warm-up stretch where the rolling features are still filling in
        df = df[(df["date"] >= df["date"].min() + pd.Timedelta(days=60)) & (df["open"] == 1)]
        ds = lgb.Dataset(df[FEATURES], df["units_capped"], categorical_feature=CATEGORICAL,
                         free_raw_data=False)
        self.boosters["point"] = lgb.train(
            {"objective": "tweedie", "tweedie_variance_power": 1.2, **self.params},
            ds, self.num_rounds)
        for q in self.quantiles:
            self.boosters[f"q{int(q * 100)}"] = lgb.train(
                {**self.params, "objective": "quantile", "alpha": q}, ds, self.num_rounds // 2)
        return self

    def predict(self, history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
        """Forecast `future` rows (sku, date, open) using `history` up to its last date."""
        cols = ["sku", "date", "open", "units"]
        both = pd.concat([history[cols], future[cols]], ignore_index=True)
        df = self._design(both)
        df = df.merge(future[["sku", "date"]], on=["sku", "date"], how="inner")
        out = df[["sku", "date", "open"]].copy()
        is_open = out["open"].to_numpy()
        out["yhat"] = np.clip(self.boosters["point"].predict(df[FEATURES]), 0, None) * is_open
        for q in self.quantiles:
            key = f"q{int(q * 100)}"
            if key in self.boosters:
                out[key] = np.clip(self.boosters[key].predict(df[FEATURES]), 0, None) * is_open
        if {"q10", "q90"} <= set(out):
            lo = np.minimum(out["q10"], out["yhat"])
            hi = np.maximum(out["q90"], out["yhat"])
            out["q10"], out["q90"] = lo, hi
        return out

    def feature_importance(self) -> pd.Series:
        b = self.boosters["point"]
        return pd.Series(b.feature_importance("gain"), index=b.feature_name()).sort_values(ascending=False)

    # ------------------------------------------------------------ persistence
    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for name, b in self.boosters.items():
            b.save_model(str(path / f"{name}.txt"))
        meta = {
            "params": {**self.params, "cap_q": self.cap_q}, "num_rounds": self.num_rounds,
            "quantiles": list(self.quantiles),
            "sku_codes": self.sku_codes, "caps": self.caps.to_dict(),
            "train_end": str(self.train_end.date()), "features": FEATURES,
        }
        (path / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: Path) -> Forecaster:
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        f = cls(meta["params"], meta["num_rounds"], tuple(meta["quantiles"]))
        f.sku_codes = meta["sku_codes"]
        f.caps = pd.Series(meta["caps"], dtype=float)
        f.train_end = pd.Timestamp(meta["train_end"])
        for p in path.glob("*.txt"):
            f.boosters[p.stem] = lgb.Booster(model_file=str(p))
        return f
