"""
fetch_euribor_rates.py
================================================================================
Loads historical EURIBOR daily fixings from a local Excel file exported from:
  Bank of Finland (Suomen Pankki) – euriborkorot_pv_chrt_en report

Source file
-----------
  common/input/euriborkorot_pv_chrt_en.xlsx
  (Downloaded from https://www.suomenpankki.fi/en/statistics/data-and-charts/
   interest-rates/charts/korot_kuviot_en/euriborkorot_pv_chrt_en/)

Expected Excel layout
---------------------
  The file has a header row containing:
    "Date", "1 week", "1 month", "3 month", "6 month", "12 month"
  followed by daily rows.  Blank rows and the leading blank column are skipped
  automatically.

Usage
-----
  # from the command line:
  python fetch_euribor_rates.py

  # as a module:
  from fetch_euribor_rates import get_rate
  r = get_rate("2025-09-08", tenor="3M")   # returns float, e.g. 2.029

Outputs (written to common/output/)
------------------------------------
  euribor_rates.csv
  euribor_rates.xlsx

================================================================================
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
INPUT_DIR  = SCRIPT_DIR / "input"
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

OUT_CSV  = OUTPUT_DIR / "euribor_rates.csv"
OUT_XLSX = OUTPUT_DIR / "euribor_rates.xlsx"

COLUMNS = ["Date", "1W", "1M", "3M", "6M", "12M"]

_CACHE: pd.DataFrame | None = None   # in-process cache after first load

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_excel(path: Path) -> pd.DataFrame:
    """
    Parse the Bank of Finland EURIBOR Excel export.

    The file may have a blank leading column and the header row somewhere near
    the top.  This function locates the header row dynamically, drops blank
    columns, renames to standard names (1W/1M/3M/6M/12M), and returns a clean
    DataFrame sorted by Date.
    """
    raw = pd.read_excel(path, header=None)

    # Find the header row: must contain "date" AND at least one "month" in non-NaN cells
    header_idx = None
    for i, row in raw.iterrows():
        non_null = [str(v).strip().lower() for v in row.values if pd.notna(v) and str(v).strip()]
        if any("date" == v for v in non_null) and any("month" in v for v in non_null):
            header_idx = int(i)
            break

    if header_idx is None:
        raise ValueError(
            f"Could not locate a header row in {path.name}. "
            "Expected a row containing 'Date' and 'month'."
        )

    # Promote header row, take data below it
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.iloc[header_idx]]
    df = df.iloc[header_idx + 1:].reset_index(drop=True)

    # Drop columns with blank / NaN / "nan" names
    df = df[[c for c in df.columns if c.strip() not in ("", "nan")]]

    # Rename to standard short names
    rename = {}
    for col in df.columns:
        cl = col.strip().lower()
        if cl == "date":
            rename[col] = "Date"
        elif "week" in cl:
            rename[col] = "1W"
        elif cl == "1 month":
            rename[col] = "1M"
        elif cl == "3 month":
            rename[col] = "3M"
        elif cl == "6 month":
            rename[col] = "6M"
        elif "12 month" in cl or "1 year" in cl:
            rename[col] = "12M"
    df = df.rename(columns=rename)

    if "Date" not in df.columns:
        raise ValueError(
            f"'Date' column not found after parsing {path.name}. "
            f"Columns present: {list(df.columns)}"
        )

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    for col in ["1W", "1M", "3M", "6M", "12M"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Date"])
    existing_cols = [c for c in COLUMNS if c in df.columns]
    return df[existing_cols].sort_values("Date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_all(save: bool = True) -> pd.DataFrame:
    """
    Load and combine EURIBOR rates from all dated Excel files in common/input/.

    Reads every euriborkorot_pv_chrt_en_*.xlsx file, combines them,
    deduplicates by date, and sorts chronologically.

    Parameters
    ----------
    save : if True, write euribor_rates.csv and euribor_rates.xlsx to common/output/

    Returns
    -------
    pd.DataFrame  with columns: Date, 1W, 1M, 3M, 6M, 12M
    """
    global _CACHE

    files = sorted(INPUT_DIR.glob("euriborkorot_pv_chrt_en*.xlsx"))
    if not files:
        sys.exit(
            f"ERROR: No input files found in {INPUT_DIR}\n"
            "Expected files matching: euriborkorot_pv_chrt_en*.xlsx"
        )

    frames: list[pd.DataFrame] = []
    for f in files:
        print(f"  [LOAD] Reading {f.name} ...")
        part = _parse_excel(f)
        print(f"    {len(part)} rows  "
              f"({part['Date'].min().strftime('%d %b %Y')} "
              f"to {part['Date'].max().strftime('%d %b %Y')})")
        frames.append(part)

    df = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["Date"])
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if save:
        df.to_csv(OUT_CSV, index=False, encoding="utf-8")
        print(f"\n  [OK] Saved {len(df)} rows to {OUT_CSV}")
        df.to_excel(OUT_XLSX, index=False)
        print(f"  [OK] Saved to {OUT_XLSX}")

    _CACHE = df
    return df


def get_rate(
    date: str | datetime,
    tenor: str = "3M",
    df: pd.DataFrame | None = None,
    tolerance_days: int = 5,
) -> float:
    """
    Return the EURIBOR fixing for a given date and tenor.

    Parameters
    ----------
    date           : "YYYY-MM-DD" string or datetime object (e.g. "2025-09-08")
    tenor          : one of "1W", "1M", "3M", "6M", "12M"  (default "3M")
    df             : pre-loaded DataFrame; if None, loads from CSV or fetches live
    tolerance_days : if exact date not found, accept nearest within this many days

    Returns
    -------
    float  rate in percent, e.g. 2.629

    Raises
    ------
    ValueError if no data within tolerance_days of the requested date
    """
    global _CACHE

    if df is None:
        if _CACHE is not None:
            df = _CACHE
        elif OUT_CSV.exists():
            df = pd.read_csv(OUT_CSV, parse_dates=["Date"])
            _CACHE = df
            print(f"  [INFO] Loaded rates from {OUT_CSV}")
        else:
            print("  [INFO] No cached data found; loading from local Excel file...")
            df = fetch_all()

    if tenor not in COLUMNS:
        raise ValueError(f"tenor must be one of {COLUMNS[1:]}; got {tenor!r}")

    target = pd.to_datetime(date)
    diff   = (df["Date"] - target).abs()
    idx    = diff.idxmin()

    if diff[idx].days > tolerance_days:
        raise ValueError(
            f"No EURIBOR data within {tolerance_days} days of {target.date()}. "
            f"Nearest available: {df.loc[idx, 'Date'].date()}. "
            f"Run fetch_all() to refresh data."
        )

    rate = df.loc[idx, tenor]
    if pd.isna(rate):
        raise ValueError(f"Rate for {tenor} on {df.loc[idx, 'Date'].date()} is NaN.")

    return float(rate)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Coupon history (IT0005599110 UniCredit Variable Rate Bond 2034)
# Reset date = 2nd TARGET Settlement Day prior to each coupon period start.
# Coupon = clip(1.60 x 3M EURIBOR, 0%, 5.45%),  30/360, quarterly.
# ---------------------------------------------------------------------------
COUPON_SCHEDULE = [
    # (coupon_no, reset_date,   period_start, payment_date,  paid)
    (1,  "2024-06-10", "2024-06-12", "2024-09-12", True),
    (2,  "2024-09-10", "2024-09-12", "2024-12-12", True),
    (3,  "2024-12-10", "2024-12-12", "2025-03-12", True),
    (4,  "2025-03-10", "2025-03-12", "2025-06-12", True),
    (5,  "2025-06-10", "2025-06-12", "2025-09-12", True),
    (6,  "2025-09-10", "2025-09-12", "2025-12-12", True),
    (7,  "2025-12-10", "2025-12-12", "2026-03-12", False),  # due 12 Mar 2026
]
PARTICIPATION = 1.60
CAP_RATE      = 5.45
FLOOR_RATE    = 0.00


def print_coupon_history(df: pd.DataFrame) -> None:
    """Print historical coupon resets with EURIBOR fixings from combined data."""
    print()
    print("=" * 78)
    print("  COUPON HISTORY  --  UniCredit Variable Rate Bond 2034  (IT0005599110)")
    print("  Coupon = clip(1.60 x 3M EURIBOR,  0%,  5.45%)  |  30/360  |  Quarterly")
    print("=" * 78)
    hdr = (f"  {'#':>2}  {'Reset Date':<12}  {'Pay Date':<12}  "
           f"{'3M EURIBOR':>10}  {'x1.60':>7}  {'Coupon':>7}  {'Status':<10}")
    print(hdr)
    print("  " + "-" * 74)
    for (no, reset_str, _ps, pay_str, paid) in COUPON_SCHEDULE:
        euribor = get_rate(reset_str, tenor="3M", df=df, tolerance_days=3)
        leveraged = PARTICIPATION * euribor
        coupon    = max(FLOOR_RATE, min(leveraged, CAP_RATE))
        status    = "PAID" if paid else "DUE 12 Mar 26"
        print(f"  {no:>2}  {reset_str:<12}  {pay_str:<12}  "
              f"{euribor:>10.4f}%  {leveraged:>6.4f}%  {coupon:>6.4f}%  {status}")
    print()


def main() -> None:
    print()
    print("=" * 68)
    print("  EURIBOR Rate Loader  --  Bank of Finland (Suomen Pankki)")
    print("=" * 68)
    print(f"  Source  : {INPUT_DIR}")
    print()

    df = fetch_all()

    # Key trade dates
    key_dates = {
        "Sep-10, 2025 (coupon 6 reset)": "2025-09-10",
        "Nov-5, 2025 (trade date)"     : "2025-11-05",
        "Nov-7, 2025 (settlement date)": "2025-11-07",
        "Dec-10, 2025 (coupon 7 reset)": "2025-12-10",
    }

    print()
    print("  KEY DATES FOR SMM269 COURSEWORK")
    print("  " + "-" * 54)
    print(f"  {'Date':<38}  {'1W':>6}  {'1M':>6}  {'3M':>6}  {'6M':>6}  {'12M':>6}")
    print("  " + "-" * 54)
    for label, d in key_dates.items():
        try:
            row = df[df["Date"] == pd.to_datetime(d)]
            if row.empty:
                idx = (df["Date"] - pd.to_datetime(d)).abs().idxmin()
                row = df.iloc[[idx]]
                actual = row["Date"].values[0]
                label += f"  [nearest: {pd.Timestamp(actual).strftime('%d %b %Y')}]"
            r = row.iloc[0]
            print(f"  {label:<38}  "
                  f"{r['1W']:>6.3f}  {r['1M']:>6.3f}  "
                  f"{r['3M']:>6.3f}  {r['6M']:>6.3f}  {r['12M']:>6.3f}")
        except Exception as e:
            print(f"  {label:<38}  ERROR: {e}")

    print()
    print(f"  Total rows : {len(df)}")
    print(f"  Date range : {df['Date'].min().strftime('%d %b %Y')} "
          f"to {df['Date'].max().strftime('%d %b %Y')}")

    print_coupon_history(df)



if __name__ == "__main__":
    main()
