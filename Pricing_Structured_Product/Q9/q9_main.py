"""
q9_main.py
--------------------------------------------------------------------------------
Produces a multi-panel chart of the bond's expected cash flows:

  Panel 1: Nominal (undiscounted) quarterly coupon cash flows
            - Historical fixings (already paid) in grey
            - Base-case (forward-projected) future coupons in navy
            - Par redemption at maturity shown as stacked bar
  Panel 2: Present values of each cash flow (discounted at Q3 curve)
            - Cumulative PV line showing value build-up over time
  Panel 3: EURIBOR forward curve with floor / cap breakeven overlays

Outputs (output/)
-----------------
  q9_expected_cashflows.png   Three-panel cash flow chart
  q9_cashflow_schedule.csv    Detailed cash flow schedule
  Console report
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import QuantLib as ql
from datetime import date as pydate

# ---------------------------------------------------------------------------
# Paths and imports
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT_DIR))
from Q3.q3_market_data import load_market_data
from Q3.q3_main import build_curve

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Bond parameters (consistent with Q4/Q5/Q6/Q8)
# ---------------------------------------------------------------------------
ISSUE_DATE = ql.Date(12, 6, 2024)
MATURITY_DATE = ql.Date(12, 6, 2034)
PARTICIPATION = 1.60
FLOOR_RATE = 0.00
CAP_RATE = 0.0545
NOTIONAL = 1000.0

CALENDAR = ql.TARGET()
COUPON_DC = ql.Thirty360(ql.Thirty360.BondBasis)
FLOAT_DC = ql.Actual360()

# Historical 3M EURIBOR fixings (consistent with Q4/Q6/Q8)
EURIBOR_FIXINGS = {
    ql.Date(10, 6, 2024): 0.037430,
    ql.Date(10, 9, 2024): 0.034600,
    ql.Date(10, 12, 2024): 0.028720,
    ql.Date(10, 3, 2025): 0.025470,
    ql.Date(10, 6, 2025): 0.019540,
    ql.Date(10, 9, 2025): 0.020290,
    ql.Date(10, 12, 2025): 0.020820,
}

# Colours
NAVY = "#1a3a5c"
GREY = "#888888"
GREEN = "#1a7a3a"
RED = "#cc2200"
AMBER = "#c07800"
TEAL = "#0088aa"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_bond_schedule() -> ql.Schedule:
    return ql.MakeSchedule(
        effectiveDate=ISSUE_DATE,
        terminationDate=MATURITY_DATE,
        tenor=ql.Period(ql.Quarterly),
        calendar=CALENDAR,
        convention=ql.ModifiedFollowing,
        terminalDateConvention=ql.ModifiedFollowing,
        rule=ql.DateGeneration.Forward,
        endOfMonth=False,
    )


def _get_reset_date(period_start: ql.Date) -> ql.Date:
    return CALENDAR.advance(period_start, -2, ql.Days, ql.Preceding)


def _get_historical_euribor(reset_date: ql.Date) -> float | None:
    if reset_date in EURIBOR_FIXINGS:
        return EURIBOR_FIXINGS[reset_date]
    candidates = sorted(
        [d for d in EURIBOR_FIXINGS if d <= reset_date], reverse=True
    )
    if candidates:
        return EURIBOR_FIXINGS[candidates[0]]
    return None


def _ql_to_pydate(d: ql.Date) -> pydate:
    return d.to_date()


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------
def compute_cashflows(curve: ql.YieldTermStructure, eval_date: ql.Date) -> list[dict]:
    """Build the complete cash flow schedule with PVs."""
    schedule = _build_bond_schedule()
    dates = list(schedule)

    rows: list[dict] = []

    for i in range(len(dates) - 1):
        p_start = dates[i]
        p_end = dates[i + 1]
        reset = _get_reset_date(p_start)
        alpha = COUPON_DC.yearFraction(p_start, p_end)

        # Determine EURIBOR
        hist = _get_historical_euribor(reset)
        if reset <= eval_date and hist is not None:
            euribor = hist
        else:
            euribor = curve.forwardRate(
                p_start, p_end, FLOAT_DC, ql.Simple
            ).rate()

        leveraged = PARTICIPATION * euribor
        coupon_rate = max(FLOOR_RATE, min(leveraged, CAP_RATE))
        coupon_amt = NOTIONAL * coupon_rate * alpha

        is_paid = p_end <= eval_date
        df = curve.discount(p_end) if not is_paid else 0.0
        pv = coupon_amt * df if not is_paid else 0.0

        # Is this a future period?
        is_future = not is_paid

        rows.append({
            "period": i + 1,
            "start": _ql_to_pydate(p_start),
            "end": _ql_to_pydate(p_end),
            "reset": _ql_to_pydate(reset),
            "euribor": euribor,
            "coupon_rate": coupon_rate,
            "coupon_amt": coupon_amt,
            "df": df,
            "pv": pv,
            "is_paid": is_paid,
            "is_future": is_future,
        })

    # Add par redemption as a special row
    mat_df = curve.discount(MATURITY_DATE)
    rows.append({
        "period": len(dates),
        "start": _ql_to_pydate(dates[-1]),
        "end": _ql_to_pydate(MATURITY_DATE),
        "reset": _ql_to_pydate(MATURITY_DATE),
        "euribor": 0.0,
        "coupon_rate": 0.0,
        "coupon_amt": NOTIONAL,  # par redemption
        "df": mat_df,
        "pv": NOTIONAL * mat_df,
        "is_paid": False,
        "is_future": True,
    })

    return rows


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------
def save_cashflow_chart(
    rows: list[dict],
    eval_date: ql.Date,
    output_dir: Path,
) -> Path:
    """Three-panel expected cash flow chart."""
    out_png = output_dir / "q9_expected_cashflows.png"

    # Separate coupon rows from redemption
    coupon_rows = [r for r in rows if r["period"] <= 40]
    redemption_row = rows[-1] if rows[-1]["coupon_rate"] == 0.0 else None

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    fig.patch.set_facecolor("#f9f9f9")

    # ── Panel 1: Nominal cash flows ─────────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor("#f9f9f9")

    paid = [r for r in coupon_rows if r["is_paid"]]
    future = [r for r in coupon_rows if r["is_future"]]

    # Paid coupons
    if paid:
        x_paid = [r["end"] for r in paid]
        y_paid = [r["coupon_amt"] for r in paid]
        ax1.bar(x_paid, y_paid, width=60, color=GREY, alpha=0.6,
                label="Paid (actual)", zorder=2)

    # Future coupons
    if future:
        x_fut = [r["end"] for r in future]
        y_fut = [r["coupon_amt"] for r in future]
        ax1.bar(x_fut, y_fut, width=60, color=NAVY, alpha=0.75,
                label="Future (forward)", zorder=2)

    # Redemption marker
    if redemption_row:
        ax1.bar(redemption_row["end"], redemption_row["coupon_amt"],
                width=60, color=TEAL, alpha=0.8,
                label="Par redemption", zorder=2)

    eval_pydate = _ql_to_pydate(eval_date)
    ax1.axvline(eval_pydate, color=RED, ls="--", lw=1, alpha=0.7,
                label=f"Eval date ({eval_pydate.isoformat()})")

    ax1.set_xlabel("Payment Date", fontsize=9)
    ax1.set_ylabel("Cash Flow (EUR per 1,000)", fontsize=9)
    ax1.set_title("Nominal Cash Flows", fontsize=10, fontweight="bold", color=NAVY)
    ax1.legend(fontsize=7, loc="upper left")
    ax1.grid(axis="y", ls="--", alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=7)

    # ── Panel 2: Present values + cumulative PV ─────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#f9f9f9")

    future_all = [r for r in rows if r["is_future"]]
    x_pv = [r["end"] for r in future_all]
    y_pv = [r["pv"] for r in future_all]
    cum_pv = np.cumsum(y_pv)

    # Bar for individual PVs
    colours_pv = [TEAL if r.get("coupon_rate", 0) == 0 else NAVY for r in future_all]
    ax2.bar(x_pv, y_pv, width=60, color=colours_pv, alpha=0.7,
            label="Period PV", zorder=2)

    # Cumulative PV line on twin axis
    ax2b = ax2.twinx()
    ax2b.plot(x_pv, cum_pv, color=RED, lw=1.8, marker=".", ms=3,
              label="Cumulative PV", zorder=3)
    ax2b.set_ylabel("Cumulative PV (EUR)", fontsize=9, color=RED)
    ax2b.tick_params(axis="y", colors=RED)

    ax2.set_xlabel("Payment Date", fontsize=9)
    ax2.set_ylabel("Cash Flow PV (EUR per 1,000)", fontsize=9)
    ax2.set_title("Discounted Cash Flows", fontsize=10, fontweight="bold", color=NAVY)

    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")

    ax2.grid(axis="y", ls="--", alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=7)

    # ── Panel 3: EURIBOR forward curve with floor/cap ───────────────────
    ax3 = axes[2]
    ax3.set_facecolor("#f9f9f9")

    # Plot EURIBOR for all coupon periods
    x_eur = [r["end"] for r in coupon_rows]
    y_eur = [r["euribor"] * 100 for r in coupon_rows]

    paid_x = [r["end"] for r in paid]
    paid_y = [r["euribor"] * 100 for r in paid]
    fut_x = [r["end"] for r in future]
    fut_y = [r["euribor"] * 100 for r in future]

    if paid_x:
        ax3.plot(paid_x, paid_y, "o-", color=GREY, ms=4, lw=1.2,
                 label="Historical fixing")
    if fut_x:
        ax3.plot(fut_x, fut_y, "s-", color=NAVY, ms=3, lw=1.2,
                 label="Forward rate")

    # Floor and cap breakeven lines
    cap_break = CAP_RATE / PARTICIPATION * 100
    ax3.axhline(cap_break, color=RED, ls="--", lw=1,
                label=f"Cap break-even ({cap_break:.3f}%)")
    ax3.axhline(0.0, color=GREEN, ls="--", lw=1,
                label="Floor (0%)")

    # Shade the linear regime
    ax3.axhspan(0.0, cap_break, color=NAVY, alpha=0.04)
    ax3.annotate("Linear regime\n(coupon = 1.60 × r)", fontsize=7,
                 xy=(coupon_rows[len(coupon_rows)//2]["end"], cap_break / 2),
                 ha="center", color=NAVY, alpha=0.5)

    ax3.axvline(eval_pydate, color=RED, ls=":", lw=0.8, alpha=0.5)

    ax3.set_xlabel("Payment Date", fontsize=9)
    ax3.set_ylabel("3M EURIBOR (%)", fontsize=9)
    ax3.set_title("EURIBOR Path (Historical + Forward)",
                  fontsize=10, fontweight="bold", color=NAVY)
    ax3.legend(fontsize=7, loc="upper right")
    ax3.grid(axis="both", ls="--", alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax3.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=7)

    plt.tight_layout(pad=1.5)
    fig.savefig(out_png, dpi=200, bbox_inches="tight",
                pad_inches=0.05, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK] Saved {out_png.name}")
    return out_png


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------
def save_cashflow_csv(rows: list[dict], output_dir: Path) -> Path:
    out_csv = output_dir / "q9_cashflow_schedule.csv"
    csv_rows = []
    for r in rows:
        csv_rows.append({
            "period": r["period"],
            "start": r["start"].isoformat(),
            "end": r["end"].isoformat(),
            "reset": r["reset"].isoformat(),
            "euribor": round(r["euribor"], 6),
            "coupon_rate": round(r["coupon_rate"], 6),
            "cashflow_eur": round(r["coupon_amt"], 4),
            "df": round(r["df"], 8),
            "pv_eur": round(r["pv"], 4),
            "is_paid": r["is_paid"],
        })

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)

    print(f"  [OK] Saved {out_csv.name}")
    return out_csv


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    data = load_market_data()
    curve, eval_date, _ = build_curve(data)

    rows = compute_cashflows(curve, eval_date)
    coupon_rows = [r for r in rows if r["period"] <= 40]

    # Summary stats
    future_coupon = [r for r in coupon_rows if r["is_future"]]
    paid_coupon = [r for r in coupon_rows if r["is_paid"]]
    total_nominal_future = sum(r["coupon_amt"] for r in future_coupon)
    total_pv_future = sum(r["pv"] for r in future_coupon)
    redemption_pv = rows[-1]["pv"]
    total_pv = total_pv_future + redemption_pv

    save_cashflow_chart(rows, eval_date, OUTPUT_DIR)
    save_cashflow_csv(rows, OUTPUT_DIR)

    print("\n" + "=" * 76)
    print("  QUESTION 9 -- Expected Cash Flows")
    print("=" * 76)
    print(f"  Eval date                : {eval_date.to_date().isoformat()}")
    print(f"  Total periods            : {len(coupon_rows)}")
    print(f"  Paid periods             : {len(paid_coupon)}")
    print(f"  Future periods           : {len(future_coupon)}")
    print("-" * 76)
    print(f"  Total nominal future CF  : EUR {total_nominal_future:.4f}")
    print(f"  Total coupon PV          : EUR {total_pv_future:.4f}")
    print(f"  Par redemption PV        : EUR {redemption_pv:.4f}")
    print(f"  Total bond PV            : EUR {total_pv:.4f}")
    print("-" * 76)
    avg_coupon = np.mean([r["coupon_rate"] * 100 for r in future_coupon])
    max_coupon = max(r["coupon_rate"] * 100 for r in future_coupon)
    min_coupon = min(r["coupon_rate"] * 100 for r in future_coupon)
    print(f"  Avg future coupon rate   : {avg_coupon:.4f}%")
    print(f"  Min future coupon rate   : {min_coupon:.4f}%")
    print(f"  Max future coupon rate   : {max_coupon:.4f}%")
    print("=" * 76)


if __name__ == "__main__":
    main()
