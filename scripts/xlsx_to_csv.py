"""Stream the UCI Online Retail II workbook to one CSV (low memory)."""
import csv
import sys

import openpyxl

src, dst = sys.argv[1], sys.argv[2]
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
