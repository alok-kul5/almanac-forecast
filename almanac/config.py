from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "raw" / "online_retail_ii.csv"
PANEL = ROOT / "data" / "processed" / "panel.parquet"
MODEL_DIR = ROOT / "models" / "champion"
REPORTS = ROOT / "reports"
SITE_DATA = ROOT / "docs" / "data"  # the GitHub Pages report reads copies from here
TRACKING_URI = f"sqlite:///{(ROOT / 'mlruns.db').as_posix()}"
EXPERIMENT = "almanac-demand"
REGISTERED_MODEL = "almanac-demand"

HORIZON = 14  # days ahead we forecast; every demand feature is lagged at least this far
TOP_SKUS = 50
QUANTILES = (0.1, 0.9)
