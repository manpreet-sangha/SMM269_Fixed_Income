"""
q8_main.py
--------------------------------------------------------------------------------
Assembles all pricing components into a comprehensive results table showing:
  1. Leveraged FRN leg PV  (1.60 × forward coupons, uncollared)
  2. Long floor PV         (from Q5)
  3. Short cap PV          (from Q5, entered as negative)
  4. Par redemption PV     (NOTIONAL × DF(maturity))
  5. Already-paid coupons  (historical fixings, discounted at eval)
  6. Risk-free (gross) dirty price
  7. CVA                   (from Q6)
  8. Risky dirty price
  9. Accrued interest
 10. Risky clean price

Outputs (output/)
-----------------
  q8_fair_value_table.csv   Full decomposition table
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Bond parameters (consistent with Q4/Q5/Q6)
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

# Historical 3M EURIBOR fixings (consistent with Q4/Q6)
EURIBOR_FIXINGS = {
    ql.Date(10, 6, 2024): 0.037430,
    ql.Date(10, 9, 2024): 0.034600,
    ql.Date(10, 12, 2024): 0.028720,
    ql.Date(10, 3, 2025): 0.025470,
    ql.Date(10, 6, 2025): 0.019540,
    ql.Date(10, 9, 2025): 0.020290,
    ql.Date(10, 12, 2025): 0.020820,
}

# Credit parameters (consistent with Q6)
CDS_SPREAD_5Y_BPS = 75.0
RECOVERY_RATE = 0.40
LGD = 1.0 - RECOVERY_RATE
HAZARD_RATE = (CDS_SPREAD_5Y_BPS / 10_000.0) / LGD


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


def _survival_prob(t: float) -> float:
    return math.exp(-HAZARD_RATE * t)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------
def compute_fair_value(curve: ql.YieldTermStructure, eval_date: ql.Date) -> dict:
    """
    Compute the full fair value decomposition of the bond.

    Returns a dict with all pricing components.
    """
    schedule = _build_bond_schedule()
    dates = list(schedule)

    # ── Leveraged FRN leg (uncollared) ───────────────────────────────────
    frn_pv_future = 0.0       # PV of future uncollared coupon flows
    collared_pv_future = 0.0  # PV of future collared coupon flows (base case)
    paid_coupon_pv = 0.0      # PV at eval_date of already-paid coupons

    # Accrued interest: find the current period containing eval_date
    accrued_interest = 0.0
    current_coupon_rate = 0.0

    # Full cash flow list for CVA recomputation
    future_cf_dates: list[tuple[ql.Date, float]] = []

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

        # Uncollared (leveraged FRN) coupon
        frn_coupon_rate = PARTICIPATION * euribor
        frn_coupon_amt = NOTIONAL * frn_coupon_rate * alpha

        # Collared coupon (actual bond coupon)
        collared_rate = max(FLOOR_RATE, min(frn_coupon_rate, CAP_RATE))
        collared_amt = NOTIONAL * collared_rate * alpha

        if p_end <= eval_date:
            # Already paid -- not part of current valuation
            continue
        elif reset <= eval_date:
            # Current or near-current period: coupon is fixed (reset done)
            # but payment is in the future
            df = curve.discount(p_end)

            # Both FRN and collared use the actual fixing
            frn_pv_future += frn_coupon_amt * df
            collared_pv_future += collared_amt * df
            future_cf_dates.append((p_end, collared_amt))

            # Accrued interest: if eval_date falls within this period
            if p_start <= eval_date < p_end:
                accrued_frac = COUPON_DC.yearFraction(p_start, eval_date)
                accrued_interest = NOTIONAL * collared_rate * accrued_frac
                current_coupon_rate = collared_rate
        else:
            # Future period: use forward rates
            df = curve.discount(p_end)
            frn_pv_future += frn_coupon_amt * df
            collared_pv_future += collared_amt * df
            future_cf_dates.append((p_end, collared_amt))

    # Par redemption
    par_redemption_pv = NOTIONAL * curve.discount(MATURITY_DATE)
    future_cf_dates.append((MATURITY_DATE, NOTIONAL))

    # ── Read Q5 option component PVs ────────────────────────────────────
    # We recompute from the CSV to keep this module self-contained
    q5_csv = ROOT_DIR / "Q5" / "output" / "q5_option_components.csv"
    floor_pv = 0.0
    cap_pv = 0.0
    if q5_csv.exists():
        with open(q5_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                floor_pv += float(row["floorlet_pv"])
                cap_pv += float(row["caplet_pv"])

    # ── CVA recomputation (consistent with Q6) ──────────────────────────
    future_dates_sorted = sorted(
        {d for d, _ in future_cf_dates},
        key=lambda d: d.serialNumber(),
    )

    total_cva = 0.0
    for idx, ti in enumerate(future_dates_sorted):
        remaining_pv = sum(
            amt * curve.discount(d)
            for d, amt in future_cf_dates
            if d >= ti
        )
        t_i = OPTION_DC.yearFraction(eval_date, ti)
        t_prev = (
            OPTION_DC.yearFraction(eval_date, future_dates_sorted[idx - 1])
            if idx > 0
            else 0.0
        )
        marginal_pd = _survival_prob(t_prev) - _survival_prob(t_i)
        total_cva += LGD * remaining_pv * marginal_pd

    # ── Assemble results ────────────────────────────────────────────────
    # Decomposition approach:
    #   Risk-free dirty price = FRN leg + Floor - Cap + Par redemption
    #   (equivalently = Collared coupon PV + Par redemption PV)
    risk_free_dirty = collared_pv_future + par_redemption_pv
    risky_dirty = risk_free_dirty - total_cva
    risky_clean = risky_dirty - accrued_interest

    # Also compute via decomposition for the table
    # FRN leg PV = uncollared coupon PV
    # Verify: FRN + Floor - Cap ≈ Collared (should match closely)
    decomp_check = frn_pv_future + floor_pv - cap_pv

    results = {
        # Decomposition components
        "leveraged_frn_pv": frn_pv_future,
        "long_floor_pv": floor_pv,
        "short_cap_pv": cap_pv,
        "net_option_pv": floor_pv - cap_pv,
        "par_redemption_pv": par_redemption_pv,
        # Aggregates
        "collared_coupon_pv": collared_pv_future,
        "risk_free_dirty": risk_free_dirty,
        "cva": total_cva,
        "risky_dirty": risky_dirty,
        "accrued_interest": accrued_interest,
        "risky_clean": risky_clean,
        # Percentage of par
        "risk_free_dirty_pct": risk_free_dirty / NOTIONAL * 100,
        "risky_dirty_pct": risky_dirty / NOTIONAL * 100,
        "risky_clean_pct": risky_clean / NOTIONAL * 100,
        # Supporting info
        "eval_date": eval_date.to_date().isoformat(),
        "current_coupon_rate_pct": current_coupon_rate * 100,
        "decomp_check": decomp_check,
        "collared_vs_decomp_diff": collared_pv_future - decomp_check,
    }
    return results


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def _save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    data = load_market_data()
    curve, eval_date, _ = build_curve(data)

    r = compute_fair_value(curve, eval_date)

    # Build the results table rows
    table_rows = [
        {"component": "A. Leveraged FRN leg (1.60 * fwd coupons)",
         "pv_eur": round(r["leveraged_frn_pv"], 4),
         "pct_par": round(r["leveraged_frn_pv"] / NOTIONAL * 100, 4)},
        {"component": "B. Long floor strip (K = 0%)",
         "pv_eur": round(r["long_floor_pv"], 4),
         "pct_par": round(r["long_floor_pv"] / NOTIONAL * 100, 4)},
        {"component": "C. Short cap strip (K = 3.406%)",
         "pv_eur": round(-r["short_cap_pv"], 4),
         "pct_par": round(-r["short_cap_pv"] / NOTIONAL * 100, 4)},
        {"component": "   Net option value (B + C)",
         "pv_eur": round(r["net_option_pv"], 4),
         "pct_par": round(r["net_option_pv"] / NOTIONAL * 100, 4)},
        {"component": "D. Par redemption (100% at maturity)",
         "pv_eur": round(r["par_redemption_pv"], 4),
         "pct_par": round(r["par_redemption_pv"] / NOTIONAL * 100, 4)},
        {"component": "---", "pv_eur": "---", "pct_par": "---"},
        {"component": "Risk-free dirty price (A+B+C+D)",
         "pv_eur": round(r["risk_free_dirty"], 4),
         "pct_par": round(r["risk_free_dirty_pct"], 4)},
        {"component": "E. CVA (credit adjustment)",
         "pv_eur": round(-r["cva"], 4),
         "pct_par": round(-r["cva"] / NOTIONAL * 100, 4)},
        {"component": "---", "pv_eur": "---", "pct_par": "---"},
        {"component": "Risky dirty price (A+B+C+D+E)",
         "pv_eur": round(r["risky_dirty"], 4),
         "pct_par": round(r["risky_dirty_pct"], 4)},
        {"component": "F. Accrued interest",
         "pv_eur": round(-r["accrued_interest"], 4),
         "pct_par": round(-r["accrued_interest"] / NOTIONAL * 100, 4)},
        {"component": "---", "pv_eur": "---", "pct_par": "---"},
        {"component": "RISKY CLEAN PRICE",
         "pv_eur": round(r["risky_clean"], 4),
         "pct_par": round(r["risky_clean_pct"], 4)},
    ]

    out_table = OUTPUT_DIR / "q8_fair_value_table.csv"
    _save_csv(out_table, table_rows, ["component", "pv_eur", "pct_par"])

    # Console report
    print("\n" + "=" * 76)
    print("  QUESTION 8 -- Fair Value Decomposition Table")
    print("=" * 76)
    print(f"  Eval date                : {r['eval_date']}")
    print(f"  Current coupon rate      : {r['current_coupon_rate_pct']:.4f}%")
    print("-" * 76)
    print(f"  {'Component':<45s} {'PV (EUR)':>12s} {'% par':>8s}")
    print("-" * 76)
    for row in table_rows:
        if row["pv_eur"] == "---":
            print("-" * 76)
        else:
            print(f"  {row['component']:<45s} {row['pv_eur']:>12.4f} {row['pct_par']:>8.4f}")
    print("=" * 76)
    print(f"\n  Decomposition check:")
    print(f"    FRN + Floor - Cap           = {r['decomp_check']:.4f}")
    print(f"    Collared coupon PV          = {r['collared_coupon_pv']:.4f}")
    print(f"    Difference (collar effect)  = {r['collared_vs_decomp_diff']:.4f}")
    print(f"    Accrued interest            = {r['accrued_interest']:.4f}")
    print(f"                                  ({r['current_coupon_rate_pct']:.4f}% coupon, "
          f"30/360 accrual to {r['eval_date']})")
    print(f"\n  Saved: {out_table}")
    print("=" * 76)


if __name__ == "__main__":
    main()
