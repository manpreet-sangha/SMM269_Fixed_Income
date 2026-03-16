"""
q15_main.py
--------------------------------------------------------------------------------
Measures the bond's exposure to changes in the UniCredit CDS spread by
bumping the spread from its base value of 75 bps and observing the change
in fair (risky) clean price.

Outputs (output/)
-----------------
  q15_cs_sensitivity.csv       Bump-by-bump results
  q15_cs_chart.png             Price–spread profile
  Console report
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import QuantLib as ql

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
from Q12.q12_main import _load_and_strip_vols, reprice_bond

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Bond parameters (consistent with Q6 / Q12)
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
OPTION_DC = ql.Actual365Fixed()

EURIBOR_FIXINGS = {
    ql.Date(10, 6, 2024): 0.037430,
    ql.Date(10, 9, 2024): 0.034600,
    ql.Date(10, 12, 2024): 0.028720,
    ql.Date(10, 3, 2025): 0.025470,
    ql.Date(10, 6, 2025): 0.019540,
    ql.Date(10, 9, 2025): 0.020290,
    ql.Date(10, 12, 2025): 0.020820,
}

# Base credit parameters
BASE_CDS_BPS = 75.0
RECOVERY_RATE = 0.40
LGD = 1.0 - RECOVERY_RATE


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


# ---------------------------------------------------------------------------
# CVA with variable hazard rate
# ---------------------------------------------------------------------------
def compute_cva_for_spread(
    curve: ql.YieldTermStructure,
    eval_date: ql.Date,
    future_cf: list[tuple[ql.Date, float]],
    cds_bps: float,
) -> float:
    """
    Compute CVA using a flat hazard rate derived from the given CDS spread.

    Parameters
    ----------
    curve       : discount curve (unchanged across CDS bumps)
    eval_date   : valuation date
    future_cf   : list of (date, amount) for all future cashflows
    cds_bps     : CDS spread in basis points

    Returns
    -------
    total CVA in EUR
    """
    hazard_rate = (cds_bps / 10_000.0) / LGD
    future_dates = sorted({d for d, _ in future_cf}, key=lambda d: d.serialNumber())

    total_cva = 0.0
    for idx, ti in enumerate(future_dates):
        remaining = sum(amt * curve.discount(d) for d, amt in future_cf if d >= ti)
        t_i = OPTION_DC.yearFraction(eval_date, ti)
        t_prev = (OPTION_DC.yearFraction(eval_date, future_dates[idx - 1])
                  if idx > 0 else 0.0)
        q_prev = math.exp(-hazard_rate * t_prev)
        q_i = math.exp(-hazard_rate * t_i)
        marginal_pd = q_prev - q_i
        total_cva += LGD * remaining * marginal_pd
    return total_cva


def build_future_cashflows(
    curve: ql.YieldTermStructure,
    eval_date: ql.Date,
) -> list[tuple[ql.Date, float]]:
    """
    Generate the collared coupon + par cashflow vector for all future dates.
    """
    schedule = _build_bond_schedule()
    dates = list(schedule)
    future_cf: list[tuple[ql.Date, float]] = []

    for i in range(len(dates) - 1):
        p_start = dates[i]
        p_end = dates[i + 1]
        if p_end <= eval_date:
            continue

        reset = _get_reset_date(p_start)
        alpha = COUPON_DC.yearFraction(p_start, p_end)

        hist = _get_historical_euribor(reset)
        if reset <= eval_date and hist is not None:
            euribor = hist
        else:
            euribor = curve.forwardRate(
                p_start, p_end, FLOAT_DC, ql.Simple
            ).rate()

        frn_rate = PARTICIPATION * euribor
        collared_rate = max(FLOOR_RATE, min(frn_rate, CAP_RATE))
        coupon_amt = NOTIONAL * collared_rate * alpha
        future_cf.append((p_end, coupon_amt))

    # Par redemption
    future_cf.append((MATURITY_DATE, NOTIONAL))
    return future_cf


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------
def make_chart(rows: list[dict], base_cds: float, base_clean: float,
               out_path: Path):
    """Price-spread profile with CS-DV01 tangent line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    NAVY = "#1a3a5c"
    RED = "#cc2200"

    spreads = [r["cds_bps"] for r in rows]
    prices = [r["risky_clean"] for r in rows]

    # CS-DV01 for tangent line
    cs_dv01 = rows[0]["cs_dv01"]  # stored in first row (computed centrally)

    fig, ax = plt.subplots(figsize=(3.8, 3.2), dpi=400,
                           facecolor="#f9f9f9")
    ax.set_facecolor("#f9f9f9")

    # Main curve
    ax.plot(spreads, prices, "o-", color=NAVY, linewidth=1.4,
            markersize=4, zorder=3, label="Risky clean price")

    # Base point
    ax.plot(base_cds, base_clean, "s", color=RED, markersize=7,
            zorder=4, label=f"Base ({base_cds:.0f} bps)")

    # Tangent line (CS-DV01 slope)
    x_tang = np.linspace(min(spreads), max(spreads), 50)
    y_tang = base_clean - cs_dv01 * (x_tang - base_cds)
    ax.plot(x_tang, y_tang, "--", color=RED, alpha=0.5, linewidth=0.8,
            label=f"CS-DV01 tangent ({cs_dv01:+.4f})")

    ax.set_xlabel("CDS Spread (bps)", fontsize=7)
    ax.set_ylabel("Risky Clean Price (EUR)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    data = load_market_data()
    curve, eval_date, _ = build_curve(data)
    stripped_vols = _load_and_strip_vols(curve, eval_date)

    # ── Base-case pricing ────────────────────────────────────────────────
    base_result = reprice_bond(curve, eval_date, stripped_vols)
    base_rf_dirty = base_result["risk_free_dirty"]
    base_accrued = base_result["accrued_interest"]
    base_cva = base_result["cva"]
    base_clean = base_result["risky_clean"]

    # Build future cashflows once (they don't change with CDS bumps)
    future_cf = build_future_cashflows(curve, eval_date)

    # ── CDS spread sweep ────────────────────────────────────────────────
    bumps_bps = [-25, -10, -5, -1, 0, +1, +5, +10, +25]
    results: list[dict] = []

    for bump in bumps_bps:
        cds_bps = BASE_CDS_BPS + bump
        cva = compute_cva_for_spread(curve, eval_date, future_cf, cds_bps)
        risky_dirty = base_rf_dirty - cva
        risky_clean = risky_dirty - base_accrued
        delta_clean = risky_clean - base_clean
        delta_cva = cva - base_cva

        results.append({
            "bump_bps": bump,
            "cds_bps": cds_bps,
            "hazard_rate_bps": round((cds_bps / 10_000.0) / LGD * 10_000, 2),
            "cva": round(cva, 4),
            "delta_cva": round(delta_cva, 4),
            "risky_clean": round(risky_clean, 4),
            "delta_clean": round(delta_clean, 4),
        })

    # ── CS-DV01 (central difference at ±1 bp) ───────────────────────────
    price_up1 = next(r["risky_clean"] for r in results if r["bump_bps"] == +1)
    price_dn1 = next(r["risky_clean"] for r in results if r["bump_bps"] == -1)
    cs_dv01 = -(price_up1 - price_dn1) / 2.0  # positive = loses value on spread widening

    # CS-Convexity (second derivative at 1 bp)
    price_base = next(r["risky_clean"] for r in results if r["bump_bps"] == 0)
    cs_convexity = (price_up1 + price_dn1 - 2 * price_base) / (1.0 ** 2)

    # Store CS-DV01 in all rows for chart access
    for r in results:
        r["cs_dv01"] = round(cs_dv01, 6)
        r["cs_convexity"] = round(cs_convexity, 6)

    # ── Save CSV ─────────────────────────────────────────────────────────
    out_csv = OUTPUT_DIR / "q15_cs_sensitivity.csv"
    fieldnames = ["bump_bps", "cds_bps", "hazard_rate_bps", "cva",
                  "delta_cva", "risky_clean", "delta_clean",
                  "cs_dv01", "cs_convexity"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)

    # ── Chart ────────────────────────────────────────────────────────────
    out_chart = OUTPUT_DIR / "q15_cs_chart.png"
    make_chart(results, BASE_CDS_BPS, base_clean, out_chart)

    # ── Console report ───────────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("  QUESTION 15 -- Credit Spread Sensitivity (CS-DV01)")
    print("=" * 76)
    print(f"  Eval date          : {eval_date}")
    print(f"  Base CDS spread    : {BASE_CDS_BPS:.0f} bps")
    print(f"  Recovery rate      : {RECOVERY_RATE:.0%}")
    print(f"  Base risky clean   : EUR {base_clean:.4f}")
    print("-" * 76)
    print(f"{'Bump':>6s}  {'CDS':>6s}  {'λ (bps)':>8s}  {'CVA':>10s}  "
          f"{'ΔCVA':>8s}  {'Clean':>10s}  {'ΔClean':>8s}")
    print("-" * 76)
    for r in results:
        marker = " <--" if r["bump_bps"] == 0 else ""
        print(f"{r['bump_bps']:>+5d}   {r['cds_bps']:>6.0f}  "
              f"{r['hazard_rate_bps']:>8.2f}  {r['cva']:>10.4f}  "
              f"{r['delta_cva']:>+8.4f}  {r['risky_clean']:>10.4f}  "
              f"{r['delta_clean']:>+8.4f}{marker}")
    print("-" * 76)
    print(f"  CS-DV01     : {cs_dv01:+.4f} EUR per 1 bp spread widening")
    print(f"  CS-Convexity: {cs_convexity:+.6f} EUR per bp²")
    print("-" * 76)
    print(f"  Interpretation: A 1 bp widening in UniCredit CDS spread")
    print(f"  reduces the bond fair value by EUR {abs(cs_dv01):.4f}")
    print(f"  ({abs(cs_dv01)/NOTIONAL*100:.4f}% of notional).")
    print("=" * 76)
    print(f"  Saved: {out_csv}")
    print(f"  Saved: {out_chart}")


if __name__ == "__main__":
    main()
