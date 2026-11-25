"""Stream the Online Retail II workbook (two yearly sheets) into one CSV, low memory."""
import csv
import sys

import openpyxl


def convert(src, dst) -> None:
    wb = openpyxl.load_workbook(src, read_only=True)
    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header_written = False
        for ws in wb.worksheets:
            for i, row in enumerate(ws.iter_rows(values_only=True)):
                if i == 0:
                    if not header_written:
                        w.writerow(row)
                        header_written = True
                    continue
                w.writerow(row)


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
