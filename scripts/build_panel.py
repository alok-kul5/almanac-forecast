"""raw CSV -> data/processed/panel.parquet"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from almanac.config import PANEL, RAW_CSV  # noqa: E402
from almanac.data import build_panel, clean_lines, load_raw  # noqa: E402


def main() -> None:
    lines = clean_lines(load_raw(RAW_CSV))
    panel = build_panel(lines)
    PANEL.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL, index=False)
    print(f"{panel['sku'].nunique()} skus, {panel['date'].nunique()} days, "
          f"{panel['date'].min().date()} -> {panel['date'].max().date()}, "
          f"zero share {(panel['units'] == 0).mean():.2f}")


if __name__ == "__main__":
    main()
