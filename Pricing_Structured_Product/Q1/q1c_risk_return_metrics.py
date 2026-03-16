"""
q1c_risk_return_metrics.py
================================================================================
Called from  : q1_main.py
Reads        : bond_details.txt (via the pre-parsed details dict)
               ql_pvs dict  (from ql_replication_pv, optional but recommended)
Outputs      : console table  +  output/q1c_risk_return_metrics.txt
               output/q1c_coupon_history.png
================================================================================
"""

from __future__ import annotations

import csv
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless backend — no GUI window
import matplotlib.pyplot as plt
import numpy as np
import QuantLib as ql

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Bond parameters
# ---------------------------------------------------------------------------
_ISSUE_DATE   = ql.Date(12, 6, 2024)
_MATURITY     = ql.Date(12, 6, 2034)
_PARTICIPATION = 1.60
_FLOOR_PCT    = 0.0        # 0 %
_CAP_PCT      = 5.45       # 5.45 %  (values in the CSV are already in percent)

_EURIBOR_CSV  = Path(__file__).parent.parent / "common" / "output" / "euribor_rates.csv"

_COL_HEADERS = ["#", "Reset Date", "Payment Date", "3M EURIBOR", "\u00d71.60", "Coupon", "Status"]


# ---------------------------------------------------------------------------
# Dynamic coupon schedule builder
# ---------------------------------------------------------------------------

def _load_euribor_3m(csv_path: Path) -> dict[str, float]:
    """Load {date_str: 3M_rate} from the Bank of Finland CSV."""
    rates: dict[str, float] = {}
    with open(csv_path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rates[row["Date"]] = float(row["3M"])
    return rates


def _ql_to_iso(d: ql.Date) -> str:
    return f"{d.year()}-{d.month():02d}-{d.dayOfMonth():02d}"


def _ql_to_display(d: ql.Date) -> str:
    months = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return f"{d.dayOfMonth():02d} {months[d.month()]} {d.year()}"


def _build_coupon_rows() -> list[tuple[str, ...]]:
    """
    Build coupon rows dynamically from the QuantLib schedule and the
    EURIBOR CSV.  Only includes periods whose reset-date EURIBOR is
    available in the CSV.
    """
    calendar = ql.TARGET()
    schedule = ql.MakeSchedule(
        effectiveDate=_ISSUE_DATE,
        terminationDate=_MATURITY,
        tenor=ql.Period(ql.Quarterly),
        calendar=calendar,
        convention=ql.ModifiedFollowing,
        terminalDateConvention=ql.ModifiedFollowing,
        rule=ql.DateGeneration.Forward,
        endOfMonth=False,
    )
    dates = list(schedule)

    euribor_map = _load_euribor_3m(_EURIBOR_CSV)

    today = ql.Date(datetime.now().day, datetime.now().month, datetime.now().year)

    rows: list[tuple[str, ...]] = []
    for i, (start, end) in enumerate(zip(dates[:-1], dates[1:])):
        # Reset date = 2 TARGET business days before period start
        reset = calendar.advance(start, -2, ql.Days, ql.Preceding)
        iso_reset = _ql_to_iso(reset)

        if iso_reset not in euribor_map:
            break  # no more data available

        euribor = euribor_map[iso_reset]
        euribor_pct = euribor_map[iso_reset]          # already in percent
        leveraged_pct = _PARTICIPATION * euribor_pct
        effective_pct = max(_FLOOR_PCT, min(leveraged_pct, _CAP_PCT))

        # Status
        if end <= today:
            status = "PAID"
        else:
            status = f"DUE {_ql_to_display(end)}"

        # Format coupon string
        if abs(effective_pct - _CAP_PCT) < 1e-4:
            coupon_str = f"{effective_pct:.4f} (capped)"
        else:
            coupon_str = f"{effective_pct:.4f}"

        rows.append((
            str(i + 1),
            _ql_to_display(reset),
            _ql_to_display(end),
            f"{euribor_pct:.4f}",
            f"{leveraged_pct:.4f}",
            coupon_str,
            status,
        ))

    return rows


# ---------------------------------------------------------------------------
# Coupon history table PNG


def _save_coupon_table_png(output_dir: Path) -> Path:
    """
    Render _COUPON_ROWS as a styled matplotlib table and save to
    output_dir/q1c_coupon_history.png.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_png = output_dir / "q1c_coupon_history.png"

    HEADER_BG  = "#1a3a5c"
    HEADER_FG  = "white"
    GRID_COLOR = "#b0b8c1"
    TITLE_COLOR = "#1a3a5c"
    CAP_COLOR  = "#cc0000"     # red text for (capped) rows

    col_widths = [0.04, 0.14, 0.14, 0.13, 0.10, 0.18, 0.16]
    fig_w = 11.0
    row_h = 0.38
    coupon_rows = _build_coupon_rows()
    n_rows = len(coupon_rows)
    fig_h = row_h * (n_rows + 1) + 0.65

    fig = plt.figure(figsize=(fig_w, fig_h))
    # axes fills the full figure; we'll place everything in data coords
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title line — just above the table
    subtitle_y = 1.0 - (0.30 / fig_h)   # ~0.30 in for two text lines
    title_y    = 1.0 - (0.05 / fig_h)

    ax.text(
        0.5, title_y,
        "Coupon History  \u2014  UniCredit Variable Rate Bond 2034  (ISIN IT0005599110)",
        ha="center", va="top", transform=ax.transAxes,
        fontsize=12, fontweight="bold", color=TITLE_COLOR,
    )
    ax.text(
        0.5, subtitle_y,
        "Coupon = clip( 1.60 \u00d7 3M EURIBOR,  0%,  5.45% )   |   "
        "Day count: 30/360   |   Frequency: Quarterly",
        ha="center", va="top", transform=ax.transAxes,
        fontsize=9, color="#444444",
    )

    cell_text = [list(r) for r in coupon_rows]

    table_top = 1.0 - (0.62 / fig_h)
    tbl = ax.table(
        cellText=cell_text,
        colLabels=_COL_HEADERS,
        cellLoc="center",
        bbox=[0, 0, 1, table_top],   # [left, bottom, width, height] in axes coords
        colWidths=col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)

    # Header row
    for col_idx in range(len(_COL_HEADERS)):
        cell = tbl[0, col_idx]
        cell.set_facecolor(HEADER_BG)
        cell.set_text_props(color=HEADER_FG, fontweight="bold")
        cell.set_edgecolor(GRID_COLOR)

    for row_idx, r in enumerate(coupon_rows, start=1):
        for col_idx in range(len(_COL_HEADERS)):
            cell = tbl[row_idx, col_idx]
            cell.set_facecolor("white")
            cell.set_edgecolor(GRID_COLOR)
            if col_idx == 5 and "capped" in r[5]:
                cell.set_text_props(color=CAP_COLOR)
            else:
                cell.set_text_props(color="black")

    fig.savefig(out_png, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_float(details: dict, key: str, default: float = 0.0) -> float:
    raw = details.get(key, str(default))
    numeric = re.sub(r"[^0-9.\-]", "", raw.split()[0])
    try:
        return float(numeric)
    except ValueError:
        return default


def _ql_date(d: int, m: int, y: int) -> ql.Date:
    return ql.Date(d, m, y)


def _project_cashflows(
    schedule: ql.Schedule,
    pricing_date: ql.Date,
    current_euribor: float,
    participation: float,
    floor_rate: float,
    cap_rate: float,
    notional: float,
    dc: ql.DayCounter,
    future_euribor: float | None = None,
) -> list[tuple[float, float]]:
    """
    Project future cash flows from pricing_date onward.

    The first coupon period that is still open on pricing_date uses
    current_euribor (the rate that was already fixed on the last reset date).
    All subsequent periods use future_euribor (the rate observed on the
    trade date, used as a flat forward projection).  If future_euribor is
    None, current_euribor is used uniformly for all periods.

    Returns a list of (time_in_years, cashflow) tuples; the final entry
    includes the notional redemption.
    """
    if future_euribor is None:
        future_euribor = current_euribor
    dates = list(schedule)
    cashflows: list[tuple[float, float]] = []
    is_first_open = True

    for start, end in zip(dates[:-1], dates[1:]):
        if end <= pricing_date:
            continue
        # Use the already-fixed rate for the current open period;
        # trade-date EURIBOR (flat forward) for all future periods.
        rate      = current_euribor if is_first_open else future_euribor
        is_first_open = False
        eff_coupon = min(max(participation * rate, floor_rate), cap_rate)
        t     = dc.yearFraction(pricing_date, end)
        alpha = dc.yearFraction(start, end)
        cf    = eff_coupon * alpha * notional
        cashflows.append((t, cf))

    # Add redemption to the final period's cash flow
    if cashflows:
        t_last, cf_last = cashflows[-1]
        cashflows[-1] = (t_last, cf_last + notional)

    return cashflows


def _price_from_yield(cashflows: list[tuple[float, float]], y: float) -> float:
    """Dirty price from a flat continuous yield."""
    return float(sum(cf * np.exp(-y * t) for t, cf in cashflows))


def _solve_ytm(
    cashflows: list[tuple[float, float]],
    price: float,
    guess: float = 0.03,
) -> float:
    """Newton-Raphson solver for the flat continuous YTM."""
    y = guess
    for _ in range(300):
        pv  = sum(cf * np.exp(-y * t) for t, cf in cashflows)
        dpv = sum(-t * cf * np.exp(-y * t) for t, cf in cashflows)
        diff = pv - price
        if abs(diff) < 1e-10 or abs(dpv) < 1e-14:
            break
        y -= diff / dpv
    return float(y)


def _macaulay_duration(
    cashflows: list[tuple[float, float]],
    ytm: float,
    price: float,
) -> float:
    return float(
        sum(t * cf * np.exp(-ytm * t) for t, cf in cashflows) / price
    )


def _convexity(
    cashflows: list[tuple[float, float]],
    ytm: float,
    price: float,
) -> float:
    return float(
        sum(t ** 2 * cf * np.exp(-ytm * t) for t, cf in cashflows) / price
    )


# ---------------------------------------------------------------------------
# Main function — PNG output only
# ---------------------------------------------------------------------------

def print_risk_return_metrics(
    details: dict[str, str],
    ql_pvs: dict[str, float] | None = None,
    output_dir: Path | None = None,
) -> None:
    """Generate q1c_coupon_history.png (no text output)."""
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        _save_coupon_table_png(output_dir)
