"""
q3_market_data.py
--------------------------------------------------------------------------------
Reads the IRS CURVE snapshot from MarketData202526.xlsx and returns structured
EURIBOR deposit rates and EUR IRS (AB6E) par rates for 5 November 2025.

Data source : ../common/input/MarketData202526.xlsx  (sheet: 'IRS CURVE')
Snapshot    : 5 November 2025

Returns
-------
dict with keys
    'eval_date'  : datetime.date
    'deposits'   : OrderedDict {tenor_label: rate_%}   e.g. {"ON": 1.9, "1W": 1.921, ...}
    'irs'        : OrderedDict {tenor_label: rate_%}   e.g. {"1Y": 2.13, "2Y": 2.142, ...}

Output
------
    output/q3_market_snapshot.csv
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import sys
from collections import OrderedDict
from pathlib import Path

import openpyxl

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).parent
COMMON_DIR  = SCRIPT_DIR.parent / "common"
MARKET_FILE = COMMON_DIR / "input" / "MarketData202526.xlsx"
OUTPUT_DIR  = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Row-label → standardised tenor key
# ---------------------------------------------------------------------------
DEPOSIT_LABEL_MAP = {
    "EUR CASH DEPOSIT O/N (TP) - MIDDLE RATE": "ON",
    "EBF EURIBOR 1W DELAYED - OFFERED RATE":   "1W",
    "EBF EURIBOR 1M DELAYED - OFFERED RATE":   "1M",
    "EBF EURIBOR 3M DELAYED - OFFERED RATE":   "3M",
    "EBF EURIBOR 6M DELAYED - OFFERED RATE":   "6M",
    "EBF EURIBOR 12M DELAYED - OFFERED RATE":  "12M",
}

# AB6E = Annual Bond / 6M EURIBOR (standard EUR plain-vanilla IRS)
IRS_LABEL_MAP = {
    "EUR 1Y AB6E IRS":  "1Y",
    "EUR 2Y AB6E IRS":  "2Y",
    "EUR 3Y AB6E IRS":  "3Y",
    "EUR 4Y AB6E IRS":  "4Y",
    "EUR 5Y AB6E IRS":  "5Y",
    "EUR 6Y AB6E IRS":  "6Y",
    "EUR 7Y AB6E IRS":  "7Y",
    "EUR 8Y AB6E IRS":  "8Y",
    "EUR 9Y AB6E IRS":  "9Y",
    "EUR 10Y AB6E IRS": "10Y",
}


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def load_market_data() -> dict:
    """
    Parse the IRS CURVE sheet and return structured market data.

    Returns
    -------
    dict
        eval_date : datetime.date
        deposits  : OrderedDict {tenor: rate_%}
        irs       : OrderedDict {tenor: rate_%}
    """
    wb = openpyxl.load_workbook(MARKET_FILE, read_only=True, data_only=True)
    ws = wb["IRS CURVE"]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    # First row contains the snapshot datetime
    eval_date = rows[0][0].date()

    deposits: OrderedDict[str, float] = OrderedDict()
    irs:      OrderedDict[str, float] = OrderedDict()

    for row in rows[1:]:
        label, rate = row[0], row[1]
        if label is None or rate is None:
            continue
        if label in DEPOSIT_LABEL_MAP:
            deposits[DEPOSIT_LABEL_MAP[label]] = float(rate)
        elif label in IRS_LABEL_MAP:
            irs[IRS_LABEL_MAP[label]] = float(rate)

    return {"eval_date": eval_date, "deposits": deposits, "irs": irs}


# ---------------------------------------------------------------------------
# Display & export
# ---------------------------------------------------------------------------
def print_market_data(data: dict) -> None:
    w = 62
    print("=" * w)
    print(f"  MARKET DATA SNAPSHOT  —  {data['eval_date']}")
    print("=" * w)
    print(f"\n  {'EURIBOR CASH DEPOSITS':^58}")
    print(f"  {'Tenor':<10}  {'Rate (%)':>10}")
    print(f"  {'-'*10}  {'-'*10}")
    for tenor, rate in data["deposits"].items():
        print(f"  {tenor:<10}  {rate:>10.4f}")
    print(f"\n  {'EUR IRS  (AB6E: Annual fixed 30/360 vs 6M EURIBOR)':^58}")
    print(f"  {'Tenor':<10}  {'Rate (%)':>10}")
    print(f"  {'-'*10}  {'-'*10}")
    for tenor, rate in data["irs"].items():
        print(f"  {tenor:<10}  {rate:>10.4f}")
    print()


def save_market_csv(data: dict) -> Path:
    out = OUTPUT_DIR / "q3_market_snapshot.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Type", "Tenor", "Rate (%)"])
        for tenor, rate in data["deposits"].items():
            writer.writerow(["Deposit", tenor, rate])
        for tenor, rate in data["irs"].items():
            writer.writerow(["IRS (AB6E)", tenor, rate])
    print(f"  Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Standalone usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mkt = load_market_data()
    print_market_data(mkt)
    save_market_csv(mkt)
