import numpy as np
import pandas as pd
import pytest


def make_panel(n_skus: int = 4, days: int = 500, seed: int = 0) -> pd.DataFrame:
    """Synthetic panel with weekly seasonality, a Christmas ramp and Saturday closures."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-01", periods=days, freq="D")
    rows = []
    for s in range(n_skus):
        base = 10 + 5 * s
        weekly = 1 + 0.3 * np.sin(2 * np.pi * dates.day_of_week / 7)
        xmas = 1 + 0.8 * np.exp(-((dates.day_of_year - 330) ** 2) / 400)
        lam = base * weekly * xmas
        units = rng.poisson(lam).astype(float)
        open_ = (dates.day_of_week != 5).astype(int)
        units *= open_
        rows.append(pd.DataFrame({"sku": f"SKU{s:02d}", "date": dates, "units": units, "open": open_}))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def panel():
    return make_panel()
