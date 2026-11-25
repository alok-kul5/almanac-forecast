"""Turn raw invoice lines into a clean daily SKU x date demand panel."""
from __future__ import annotations

import pandas as pd

from .config import TOP_SKUS

PRODUCT_CODE = r"^\d{5}[A-Za-z]{0,2}$"


def load_raw(path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        usecols=["Invoice", "StockCode", "Quantity", "InvoiceDate", "Price"],
        dtype={"Invoice": str, "StockCode": str, "Quantity": "int32", "Price": "float32"},
    )


def clean_lines(df: pd.DataFrame) -> pd.DataFrame:
    """Drop cancellations, returns, non-product codes (postage, fees) and bad rows."""
    df = df[~df["Invoice"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    df = df[df["StockCode"].astype(str).str.match(PRODUCT_CODE)]
    return df.assign(
        StockCode=df["StockCode"].astype(str).str.upper(),
        date=pd.to_datetime(df["InvoiceDate"]).dt.normalize(),
    )


def build_panel(lines: pd.DataFrame, top_n: int = TOP_SKUS) -> pd.DataFrame:
    """Daily units per SKU for the top-N SKUs by selling days, zero-filled."""
    # the export stops at lunchtime on its last day; a final day that ends before 15:00 is partial, so drop it
    if "InvoiceDate" in lines and pd.to_datetime(lines["InvoiceDate"]).max().hour < 15:
        lines = lines[lines["date"] < lines["date"].max()]
    last = lines["date"].max()

    daily = (
        lines.groupby(["StockCode", "date"], as_index=False)["Quantity"].sum()
        .rename(columns={"StockCode": "sku", "Quantity": "units"})
    )
    active = daily.groupby("sku")["date"].nunique().sort_values(ascending=False, kind="stable")
    keep = sorted(active.head(top_n).index)
    daily = daily[daily["sku"].isin(keep)]
    dates = pd.date_range(lines["date"].min(), last, freq="D")
    full = pd.MultiIndex.from_product([keep, dates], names=["sku", "date"])
    panel = daily.set_index(["sku", "date"])["units"].reindex(full, fill_value=0).reset_index()
    # the shop trades when anything sold that day; in production this comes from the trading calendar
    open_days = pd.DatetimeIndex(lines["date"].unique())
    panel["open"] = panel["date"].isin(open_days).astype(int)
    panel["units"] = panel["units"].astype(float)
    return panel
