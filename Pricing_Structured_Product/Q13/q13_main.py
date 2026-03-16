"""
q13_main.py
--------------------------------------------------------------------------------
Proposes and quantifies a hedging strategy for the UniCredit structured bond's
yield-curve exposure using quoted EUR interest-rate swaps (AB6E convention).

Outputs (output/)
-----------------
  q13_hedge_ratios.csv      Hedge instrument details and notionals
  q13_factor_dv01s.csv      Bond and IRS factor DV01s
  Console report
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import copy
import csv
import math
import sys
from pathlib import Path

import QuantLib as ql
import numpy as np

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
# Constants
# ---------------------------------------------------------------------------
NOTIONAL = 1000.0
CALENDAR = ql.TARGET()
OPTION_DC = ql.Actual365Fixed()
FLOAT_DC = ql.Actual360()
COUPON_DC = ql.Thirty360(ql.Thirty360.BondBasis)

# IRS notional for DV01 computation
IRS_NOTIONAL = 1_000_000.0  # standard 1M EUR notional for DV01

# Hedge instruments: 2Y and 10Y EUR IRS (AB6E)
HEDGE_TENORS = ["2Y", "10Y"]
HEDGE_PERIODS = {"2Y": ql.Period(2, ql.Years), "10Y": ql.Period(10, ql.Years)}


# ---------------------------------------------------------------------------
# Load PCA data from Q11
# ---------------------------------------------------------------------------
def load_pca_data():
    """Load eigenvalues and loadings from Q11 output."""
    q11_dir = ROOT_DIR / "Q11" / "output"

    eigenvals = []
    with open(q11_dir / "q11_pca_eigenvalues.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            eigenvals.append(float(row["eigenvalue_bps2"]))
            if len(eigenvals) == 3:
                break

    tenors = []
    loadings = [[], [], []]
    with open(q11_dir / "q11_pca_loadings.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tenors.append(row["tenor"])
            loadings[0].append(float(row["PC1"]))
            loadings[1].append(float(row["PC2"]))
            loadings[2].append(float(row["PC3"]))

    return tenors, eigenvals, loadings


# ---------------------------------------------------------------------------
# Bond repricing (import from Q12)
# ---------------------------------------------------------------------------
def _get_bond_repricer():
    """Import the reprice_bond function and vol loader from Q12."""
    from Q12.q12_main import reprice_bond, _load_and_strip_vols
    return reprice_bond, _load_and_strip_vols


# ---------------------------------------------------------------------------
# Apply a 1-bps factor shock to market data
# ---------------------------------------------------------------------------
def apply_factor_shock(base_data, pca_tenors, eigenvector, bump_bps=1.0):
    """
    Apply a +1 bps shock along a given eigenvector direction.

    The shock is: Δr(tenor) = bump_bps × loading(tenor)  [in bps]
    Converted to percentage points for the market data dict.
    """
    shocked = copy.deepcopy(base_data)
    for idx, pca_tenor in enumerate(pca_tenors):
        if pca_tenor in shocked["irs"]:
            delta_pct = bump_bps * eigenvector[idx] / 100.0
            shocked["irs"][pca_tenor] += delta_pct

    # Also perturb deposits proportionally to 1Y loading
    if pca_tenors and pca_tenors[0] == "1Y":
        delta_1y_pct = bump_bps * eigenvector[0] / 100.0
        dep_fracs = {"6M": 0.80, "3M": 0.60, "1M": 0.40, "1W": 0.20, "ON": 0.20}
        for dep, frac in dep_fracs.items():
            if dep in shocked["deposits"]:
                shocked["deposits"][dep] += frac * delta_1y_pct

    return shocked


# ---------------------------------------------------------------------------
# Compute bond factor DV01s via central difference
# ---------------------------------------------------------------------------
def compute_bond_factor_dv01s(base_data, base_curve, eval_date, stripped_vols,
                               pca_tenors, loadings, reprice_fn):
    """
    Compute the bond's DV01 along each PC factor using ±1 bps central diff.

    Returns dict: {pc_index: dv01_eur_per_bps}
    """
    bump = 1.0  # bps
    dv01s = {}

    for k in range(3):
        # Up shock
        up_data = apply_factor_shock(base_data, pca_tenors, loadings[k], +bump)
        up_curve, _, _ = build_curve(up_data)
        up_price = reprice_fn(up_curve, eval_date, stripped_vols)["risky_clean"]

        # Down shock
        dn_data = apply_factor_shock(base_data, pca_tenors, loadings[k], -bump)
        dn_curve, _, _ = build_curve(dn_data)
        dn_price = reprice_fn(dn_curve, eval_date, stripped_vols)["risky_clean"]

        dv01s[k + 1] = (up_price - dn_price) / (2.0 * bump)

    return dv01s


# ---------------------------------------------------------------------------
# Compute IRS DV01 along each PC factor
# ---------------------------------------------------------------------------
def compute_irs_factor_dv01(base_data, eval_date, pca_tenors, loadings,
                             swap_tenor_label, swap_period):
    """
    Compute an IRS's DV01 along each PC factor.

    We value a par swap (receiver) under base and shocked curves and
    compute the change per 1 bps factor move.

    Returns dict: {pc_index: dv01_eur_per_bps_per_1M_notional}
    """
    base_curve, _, _ = build_curve(base_data)
    euribor6m = ql.Euribor6M(ql.YieldTermStructureHandle(base_curve))

    settlement = CALENDAR.advance(eval_date, 2, ql.Days)
    maturity = CALENDAR.advance(settlement, swap_period, ql.ModifiedFollowing)

    # Get par rate for this swap
    par_rate = base_curve.forwardRate(
        settlement, maturity, ql.Thirty360(ql.Thirty360.BondBasis), ql.Annual
    ).rate()

    # Build receiver swap at par (receiver = receive fixed, pay float)
    schedule_fixed = ql.MakeSchedule(
        effectiveDate=settlement,
        terminationDate=maturity,
        tenor=ql.Period(ql.Annual),
        calendar=CALENDAR,
        convention=ql.ModifiedFollowing,
        terminalDateConvention=ql.ModifiedFollowing,
        rule=ql.DateGeneration.Forward,
        endOfMonth=False,
    )
    schedule_float = ql.MakeSchedule(
        effectiveDate=settlement,
        terminationDate=maturity,
        tenor=ql.Period(ql.Semiannual),
        calendar=CALENDAR,
        convention=ql.ModifiedFollowing,
        terminalDateConvention=ql.ModifiedFollowing,
        rule=ql.DateGeneration.Forward,
        endOfMonth=False,
    )

    bump = 1.0
    dv01s = {}

    for k in range(3):
        # Value the swap under up-shocked curve
        up_data = apply_factor_shock(base_data, pca_tenors, loadings[k], +bump)
        up_curve, _, _ = build_curve(up_data)

        dn_data = apply_factor_shock(base_data, pca_tenors, loadings[k], -bump)
        dn_curve, _, _ = build_curve(dn_data)

        # Swap PV = sum of fixed leg cashflows discounted - sum of float leg
        # For a par receiver swap, PV change ≈ -Annuity × Δr
        # We compute via DFs directly for accuracy

        pv_up = _swap_pv(up_curve, schedule_fixed, schedule_float, par_rate, eval_date)
        pv_dn = _swap_pv(dn_curve, schedule_fixed, schedule_float, par_rate, eval_date)

        dv01s[k + 1] = (pv_up - pv_dn) / (2.0 * bump)

    return dv01s


def _swap_pv(curve, schedule_fixed, schedule_float, fixed_rate, eval_date):
    """
    Compute the PV of a receiver swap (receive fixed, pay float) under a curve.

    Fixed leg: annual payments at `fixed_rate`, 30/360 BondBasis.
    Float leg: semi-annual forward-rate payments, ACT/360.
    Notional: IRS_NOTIONAL.
    """
    dc_fixed = ql.Thirty360(ql.Thirty360.BondBasis)
    dc_float = ql.Actual360()

    dates_f = list(schedule_fixed)
    pv_fixed = 0.0
    for i in range(len(dates_f) - 1):
        alpha = dc_fixed.yearFraction(dates_f[i], dates_f[i + 1])
        df = curve.discount(dates_f[i + 1])
        pv_fixed += IRS_NOTIONAL * fixed_rate * alpha * df

    dates_fl = list(schedule_float)
    pv_float = 0.0
    for i in range(len(dates_fl) - 1):
        alpha = dc_float.yearFraction(dates_fl[i], dates_fl[i + 1])
        fwd = curve.forwardRate(dates_fl[i], dates_fl[i + 1], dc_float, ql.Simple).rate()
        df = curve.discount(dates_fl[i + 1])
        pv_float += IRS_NOTIONAL * fwd * alpha * df

    return pv_fixed - pv_float  # receiver: +fixed -float


# ---------------------------------------------------------------------------
# Solve 2x2 hedge ratio system
# ---------------------------------------------------------------------------
def solve_hedge_ratios(bond_dv01s, irs_dv01s_2y, irs_dv01s_10y):
    """
    Solve for hedge notional multipliers N_2Y, N_10Y such that:

        N_2Y × IRS_DV01^{2Y}_{k} + N_10Y × IRS_DV01^{10Y}_{k} = -Bond_DV01_{k}

    for k = 1 (level) and k = 2 (slope).

    Returns (n_2y, n_10y) in units of bond notional per IRS_NOTIONAL.
    """
    # Build the 2×2 matrix: A × [N_2Y, N_10Y]^T = -b
    A = np.array([
        [irs_dv01s_2y[1], irs_dv01s_10y[1]],  # PC1 row
        [irs_dv01s_2y[2], irs_dv01s_10y[2]],  # PC2 row
    ])
    b = np.array([-bond_dv01s[1], -bond_dv01s[2]])

    try:
        n = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        print("  WARNING: Singular matrix — cannot solve hedge ratios.")
        n = np.array([0.0, 0.0])

    return n[0], n[1]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    reprice_fn, load_vols = _get_bond_repricer()

    # 1. Load market data and build base curve
    base_data = load_market_data()
    base_curve, eval_date, _ = build_curve(base_data)
    stripped_vols = load_vols(base_curve, eval_date)

    # 2. Price base case
    base_result = reprice_fn(base_curve, eval_date, stripped_vols)
    base_clean = base_result["risky_clean"]

    # 3. Load PCA data
    pca_tenors, eigenvals, loadings = load_pca_data()

    # 4. Compute bond factor DV01s
    bond_dv01s = compute_bond_factor_dv01s(
        base_data, base_curve, eval_date, stripped_vols,
        pca_tenors, loadings, reprice_fn
    )

    # 5. Compute IRS factor DV01s for each hedge tenor
    irs_dv01s = {}
    par_rates = {}
    for tenor in HEDGE_TENORS:
        irs_dv01s[tenor] = compute_irs_factor_dv01(
            base_data, eval_date, pca_tenors, loadings,
            tenor, HEDGE_PERIODS[tenor]
        )
        # Get par rate
        settlement = CALENDAR.advance(eval_date, 2, ql.Days)
        mat = CALENDAR.advance(settlement, HEDGE_PERIODS[tenor], ql.ModifiedFollowing)
        par_rates[tenor] = base_curve.forwardRate(
            settlement, mat, ql.Thirty360(ql.Thirty360.BondBasis), ql.Annual
        ).rate() * 100

    # 6. Solve hedge ratios (using 2Y and 10Y)
    n_2y, n_10y = solve_hedge_ratios(
        bond_dv01s, irs_dv01s["2Y"], irs_dv01s["10Y"]
    )

    # Convert to notional amounts: n is in units of (bond_notional / IRS_NOTIONAL)
    # The hedge notional for 1 bond (NOTIONAL=1000) is n × IRS_NOTIONAL
    hedge_notional_2y = n_2y * IRS_NOTIONAL
    hedge_notional_10y = n_10y * IRS_NOTIONAL

    # 7. Compute residual (unhedged curvature risk)
    hedged_dv01_pc3 = (bond_dv01s[3]
                       + n_2y * irs_dv01s["2Y"][3]
                       + n_10y * irs_dv01s["10Y"][3])

    # ── Console report ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  QUESTION 13 -- Hedging Strategy Using Plain-Vanilla IRS")
    print("=" * 80)
    print(f"  Eval date          : {eval_date.to_date().isoformat()}")
    print(f"  Base risky clean   : EUR {base_clean:.2f}")
    print(f"  Bond notional      : EUR {NOTIONAL:,.0f}")
    print()

    # Factor DV01s
    pc_names = ["Level (PC1)", "Slope (PC2)", "Curvature (PC3)"]
    print(f"  {'Factor DV01 (EUR per 1 bps factor move)':<55s}")
    print(f"  {'Factor':<20s} {'Bond':>10s} {'2Y IRS':>12s} {'10Y IRS':>12s}")
    print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*12}")
    for k in range(3):
        print(f"  {pc_names[k]:<20s} {bond_dv01s[k+1]:>10.4f} "
              f"{irs_dv01s['2Y'][k+1]:>12.4f} {irs_dv01s['10Y'][k+1]:>12.4f}")
    print()

    # Hedge solution
    print(f"  Hedge Solution (neutralise PC1 + PC2):")
    print(f"  {'-'*60}")
    print(f"  {'Instrument':<20s} {'Direction':<12s} {'Notional (EUR)':>15s} {'Par Rate':>10s}")
    print(f"  {'-'*20} {'-'*12} {'-'*15} {'-'*10}")

    dir_2y = "Receiver" if hedge_notional_2y > 0 else "Payer"
    dir_10y = "Receiver" if hedge_notional_10y > 0 else "Payer"
    print(f"  {'2Y EUR IRS':<20s} {dir_2y:<12s} {abs(hedge_notional_2y):>15,.0f} "
          f"{par_rates['2Y']:>9.3f}%")
    print(f"  {'10Y EUR IRS':<20s} {dir_10y:<12s} {abs(hedge_notional_10y):>15,.0f} "
          f"{par_rates['10Y']:>9.3f}%")
    print()

    # Hedge verification
    print(f"  Hedge Verification (residual factor DV01):")
    print(f"  {'-'*60}")
    for k in range(3):
        hedged_k = (bond_dv01s[k+1]
                    + n_2y * irs_dv01s["2Y"][k+1]
                    + n_10y * irs_dv01s["10Y"][k+1])
        status = "HEDGED" if abs(hedged_k) < 0.001 else f"residual = {hedged_k:+.4f}"
        print(f"  {pc_names[k]:<20s}  {status}")
    print()

    # ── Save CSVs ───────────────────────────────────────────────────────
    # Factor DV01s
    dv01_rows = []
    for k in range(3):
        dv01_rows.append({
            "factor": pc_names[k],
            "pc_index": k + 1,
            "bond_dv01_eur": round(bond_dv01s[k + 1], 6),
            "irs_2y_dv01_eur": round(irs_dv01s["2Y"][k + 1], 6),
            "irs_10y_dv01_eur": round(irs_dv01s["10Y"][k + 1], 6),
        })
    dv01_path = OUTPUT_DIR / "q13_factor_dv01s.csv"
    with open(dv01_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dv01_rows[0].keys()))
        w.writeheader()
        w.writerows(dv01_rows)

    # Hedge ratios
    hedge_rows = [
        {
            "instrument": "2Y EUR IRS (AB6E)",
            "direction": dir_2y,
            "notional_eur": round(abs(hedge_notional_2y), 2),
            "par_rate_pct": round(par_rates["2Y"], 4),
            "hedge_ratio": round(n_2y, 8),
        },
        {
            "instrument": "10Y EUR IRS (AB6E)",
            "direction": dir_10y,
            "notional_eur": round(abs(hedge_notional_10y), 2),
            "par_rate_pct": round(par_rates["10Y"], 4),
            "hedge_ratio": round(n_10y, 8),
        },
    ]
    hedge_path = OUTPUT_DIR / "q13_hedge_ratios.csv"
    with open(hedge_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(hedge_rows[0].keys()))
        w.writeheader()
        w.writerows(hedge_rows)

    print(f"  Saved: {dv01_path}")
    print(f"  Saved: {hedge_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
