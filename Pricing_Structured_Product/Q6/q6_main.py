"""
q6_main.py
--------------------------------------------------------------------------------
  CVA = (1-R) * sum_i  EE(t_i) * [Q(t_{i-1}) - Q(t_i)]

where:
  R          = recovery rate (40%, Basel III senior unsecured)
  Q(t)       = exp(-lambda * t), survival probability from flat hazard rate
  lambda     = CDS_spread / (1 - R)
  EE(t_i)    = expected exposure at coupon date t_i
             = PV of all remaining cash flows from t_i onward, discounted to t_0

Inputs
------
  - Q3 bootstrapped EUR discount curve (5 Nov 2025)
  - Q5 option-like component pricing (caplet/floorlet PVs)
  - CDS spread: 75 bps (UniCredit 5Y senior, approx. Nov 2025)
  - Recovery rate: 40%

Outputs (output/)
-----------------
  q6_cva_detail.csv          Period-level CVA breakdown
  q6_cva_summary.csv         Summary metrics
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
Q3_DIR = ROOT_DIR / "Q3"
Q5_DIR = ROOT_DIR / "Q5"
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT_DIR))
from Q3.q3_market_data import load_market_data
from Q3.q3_main import build_curve

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Bond parameters (consistent with Q4 / Q5)
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

# Historical 3M EURIBOR fixings (consistent with Q4)
EURIBOR_FIXINGS = {
    ql.Date(10, 6, 2024): 0.037430,
    ql.Date(10, 9, 2024): 0.034600,
    ql.Date(10, 12, 2024): 0.028720,
    ql.Date(10, 3, 2025): 0.025470,
    ql.Date(10, 6, 2025): 0.019540,
    ql.Date(10, 9, 2025): 0.020290,
    ql.Date(10, 12, 2025): 0.020820,
}


# ---------------------------------------------------------------------------
# Credit parameters
# ---------------------------------------------------------------------------
CDS_SPREAD_5Y_BPS = 75.0          # UniCredit 5Y senior CDS spread (bps)
RECOVERY_RATE = 0.40               # Basel III senior unsecured
LGD = 1.0 - RECOVERY_RATE         # 60%
HAZARD_RATE = (CDS_SPREAD_5Y_BPS / 10_000.0) / LGD  # lambda = s / (1-R)


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


def survival_prob(t: float) -> float:
    """Q(t) = exp(-lambda * t) under flat hazard rate."""
    return math.exp(-HAZARD_RATE * t)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------
def compute_cva(curve: ql.YieldTermStructure, eval_date: ql.Date):
    """
    Compute CVA for the bond from the investor's perspective.

    Steps:
    1. Build the full cash flow schedule (coupons + redemption)
    2. For each future cash flow date t_i, compute EE(t_i) = PV at t_0 of
       all remaining cash flows from t_i onward
    3. CVA = (1-R) * sum_i EE(t_i) * [Q(t_{i-1}) - Q(t_i)]

    Returns: (detail_rows, summary_dict)
    """
    schedule = _build_bond_schedule()
    dates = list(schedule)

    # ── Step 1: Build the complete cash flow vector ──────────────────────
    # Each entry: (date, cashflow_amount_undiscounted, is_coupon)
    cashflows: list[tuple[ql.Date, float, bool]] = []

    for i in range(len(dates) - 1):
        p_start = dates[i]
        p_end = dates[i + 1]
        reset = _get_reset_date(p_start)
        alpha = COUPON_DC.yearFraction(p_start, p_end)

        # Determine EURIBOR: historical fixing or forward rate
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
        cashflows.append((p_end, coupon_amt, True))

    # Add par redemption at maturity
    cashflows.append((MATURITY_DATE, NOTIONAL, False))

    # ── Step 2: Compute expected exposure at each unique future date ─────
    # Unique future dates where default could crystallise a loss
    # We use coupon payment dates as the CVA grid
    future_cf_dates = sorted(
        {d for d, _, _ in cashflows if d > eval_date},
        key=lambda d: d.serialNumber(),
    )

    # For each date, EE = sum of DF(t_0, t_j) * CF_j for all j >= i
    # (i.e., PV at t_0 of remaining flows from date t_i onward)
    exposure_rows: list[dict] = []

    for idx, ti in enumerate(future_cf_dates):
        # Remaining cash flows: all flows at or after ti
        remaining_pv = 0.0
        for cf_date, cf_amt, _ in cashflows:
            if cf_date >= ti:
                remaining_pv += cf_amt * curve.discount(cf_date)

        # Time from eval_date to ti (for survival probability)
        t_i = OPTION_DC.yearFraction(eval_date, ti)
        t_prev = OPTION_DC.yearFraction(eval_date, future_cf_dates[idx - 1]) if idx > 0 else 0.0

        q_prev = survival_prob(t_prev)
        q_i = survival_prob(t_i)
        marginal_pd = q_prev - q_i  # probability of default in [t_{i-1}, t_i]

        cva_contrib = LGD * remaining_pv * marginal_pd

        exposure_rows.append({
            "date": ti.to_date().isoformat(),
            "time_yrs": round(t_i, 6),
            "surv_prob": round(q_i, 8),
            "marginal_pd": round(marginal_pd, 8),
            "expected_exposure": round(remaining_pv, 4),
            "cva_contribution": round(cva_contrib, 6),
        })

    # ── Step 3: Aggregate ────────────────────────────────────────────────
    total_cva = sum(r["cva_contribution"] for r in exposure_rows)

    # Risk-free bond value (base case from Q4 logic)
    risk_free_pv = sum(
        cf_amt * curve.discount(cf_date)
        for cf_date, cf_amt, _ in cashflows
        if cf_date > eval_date
    )

    risky_pv = risk_free_pv - total_cva

    summary = {
        "eval_date": eval_date.to_date().isoformat(),
        "cds_spread_bps": CDS_SPREAD_5Y_BPS,
        "recovery_rate": RECOVERY_RATE,
        "hazard_rate_bps": round(HAZARD_RATE * 10_000, 2),
        "risk_free_pv": round(risk_free_pv, 4),
        "cva": round(total_cva, 4),
        "risky_pv": round(risky_pv, 4),
        "cva_pct_of_notional": round(total_cva / NOTIONAL * 100, 4),
        "cva_pct_of_risk_free": round(total_cva / risk_free_pv * 100, 4),
    }

    return exposure_rows, summary


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

    detail_rows, summary = compute_cva(curve, eval_date)

    # Save detail
    out_detail = OUTPUT_DIR / "q6_cva_detail.csv"
    _save_csv(
        out_detail,
        detail_rows,
        ["date", "time_yrs", "surv_prob", "marginal_pd",
         "expected_exposure", "cva_contribution"],
    )

    # Save summary
    out_summary = OUTPUT_DIR / "q6_cva_summary.csv"
    _save_csv(
        out_summary,
        [summary],
        list(summary.keys()),
    )

    # Console report
    print("\n" + "=" * 76)
    print("  QUESTION 6 -- Credit Valuation Adjustment (CVA)")
    print("=" * 76)
    print(f"  Eval date              : {summary['eval_date']}")
    print(f"  CDS spread (5Y)        : {summary['cds_spread_bps']:.0f} bps")
    print(f"  Recovery rate           : {summary['recovery_rate']:.0%}")
    print(f"  Hazard rate (lambda)    : {summary['hazard_rate_bps']:.2f} bps/yr")
    print("-" * 76)
    print(f"  Risk-free bond PV       : EUR {summary['risk_free_pv']:.4f}")
    print(f"  CVA                     : EUR {summary['cva']:.4f}")
    print(f"  Risky bond PV           : EUR {summary['risky_pv']:.4f}")
    print("-" * 76)
    print(f"  CVA as % of notional    : {summary['cva_pct_of_notional']:.4f}%")
    print(f"  CVA as % of risk-free PV: {summary['cva_pct_of_risk_free']:.4f}%")
    print("-" * 76)
    print(f"  Saved: {out_detail}")
    print(f"  Saved: {out_summary}")
    print("=" * 76)

    # Print first/last few detail rows
    print("\n  Period-level detail (first 5):")
    print(f"  {'Date':>12s} {'t(yr)':>8s} {'Q(t)':>10s} {'marg PD':>10s} "
          f"{'EE (EUR)':>12s} {'CVA contr':>10s}")
    for r in detail_rows[:5]:
        print(f"  {r['date']:>12s} {r['time_yrs']:8.4f} {r['surv_prob']:10.6f} "
              f"{r['marginal_pd']:10.6f} {r['expected_exposure']:12.4f} "
              f"{r['cva_contribution']:10.6f}")
    if len(detail_rows) > 5:
        print("  ...")
    for r in detail_rows[-3:]:
        print(f"  {r['date']:>12s} {r['time_yrs']:8.4f} {r['surv_prob']:10.6f} "
              f"{r['marginal_pd']:10.6f} {r['expected_exposure']:12.4f} "
              f"{r['cva_contribution']:10.6f}")


if __name__ == "__main__":
    main()
