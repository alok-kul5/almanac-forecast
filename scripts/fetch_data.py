"""Download UCI Online Retail II and build the daily panel.

    python scripts/fetch_data.py
"""
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from almanac.config import RAW_CSV  # noqa: E402

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"


def main() -> None:
    raw = RAW_CSV.parent
    raw.mkdir(parents=True, exist_ok=True)
    zpath, xlsx = raw / "online_retail_ii.zip", raw / "online_retail_II.xlsx"
    if not xlsx.exists():
        if not zpath.exists():
            print("downloading", URL)
            urllib.request.urlretrieve(URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(raw)
    if not RAW_CSV.exists():
        print("converting workbook to csv (a few minutes)")
        from xlsx_to_csv import convert

        convert(xlsx, RAW_CSV)
    from build_panel import main as build

    build()


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
