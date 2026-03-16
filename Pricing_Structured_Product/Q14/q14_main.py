"""
q14_main.py
--------------------------------------------------------------------------------
Implementation of the Hedging Strategy

Constructs the hedged portfolio (bond + 2Y receiver + 10Y receiver) at the
notionals determined in Q13, then reprices under the Q12 PCA scenarios to
demonstrate hedge effectiveness and quantify residual risk.

Outputs (output/)
-----------------
  q14_hedge_performance.csv   Scenario-by-scenario results
  q14_hedge_chart.png         Grouped bar chart (hedged vs unhedged)
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
from Q12.q12_main import reprice_bond, _load_and_strip_vols, apply_pca_shock, HORIZON_SCALE
from Q13.q13_main import load_pca_data, IRS_NOTIONAL

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NOTIONAL = 1000.0
CALENDAR = ql.TARGET()


# ---------------------------------------------------------------------------
# Load hedge ratios from Q13
# ---------------------------------------------------------------------------
def load_hedge_ratios():
    """Load hedge notionals and par rates from Q13 CSV."""
    q13_dir = ROOT_DIR / "Q13" / "output"
    ratios = {}
    with open(q13_dir / "q13_hedge_ratios.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = "2Y" if "2Y" in row["instrument"] else "10Y"
            ratios[key] = {
                "direction": row["direction"],
                "notional": float(row["notional_eur"]),
                "par_rate": float(row["par_rate_pct"]) / 100.0,
                "hedge_ratio": float(row["hedge_ratio"]),
            }
    return ratios


# ---------------------------------------------------------------------------
# Swap PV under a given curve
# ---------------------------------------------------------------------------
def compute_swap_pv(curve, eval_date, swap_period, par_rate, notional, direction):
    """
    Compute the PV of a receiver or payer swap under a given curve.

    Receiver: +fixed -float;  Payer: -fixed +float.
    Returns PV in EUR.
    """
    dc_fixed = ql.Thirty360(ql.Thirty360.BondBasis)
    dc_float = ql.Actual360()
    settlement = CALENDAR.advance(eval_date, 2, ql.Days)
    maturity = CALENDAR.advance(settlement, swap_period, ql.ModifiedFollowing)

    sched_fixed = ql.MakeSchedule(
        effectiveDate=settlement, terminationDate=maturity,
        tenor=ql.Period(ql.Annual), calendar=CALENDAR,
        convention=ql.ModifiedFollowing,
        terminalDateConvention=ql.ModifiedFollowing,
        rule=ql.DateGeneration.Forward, endOfMonth=False,
    )
    sched_float = ql.MakeSchedule(
        effectiveDate=settlement, terminationDate=maturity,
        tenor=ql.Period(ql.Semiannual), calendar=CALENDAR,
        convention=ql.ModifiedFollowing,
        terminalDateConvention=ql.ModifiedFollowing,
        rule=ql.DateGeneration.Forward, endOfMonth=False,
    )

    dates_f = list(sched_fixed)
    pv_fixed = 0.0
    for i in range(len(dates_f) - 1):
        alpha = dc_fixed.yearFraction(dates_f[i], dates_f[i + 1])
        df = curve.discount(dates_f[i + 1])
        pv_fixed += notional * par_rate * alpha * df

    dates_fl = list(sched_float)
    pv_float = 0.0
    for i in range(len(dates_fl) - 1):
        alpha = dc_float.yearFraction(dates_fl[i], dates_fl[i + 1])
        fwd = curve.forwardRate(dates_fl[i], dates_fl[i + 1], dc_float, ql.Simple).rate()
        df = curve.discount(dates_fl[i + 1])
        pv_float += notional * fwd * alpha * df

    if direction == "Receiver":
        return pv_fixed - pv_float
    else:  # Payer
        return pv_float - pv_fixed


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------
def make_hedge_chart(rows, out_path):
    """Grouped bar chart: unhedged vs hedged P&L for each scenario."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    NAVY  = "#1a3a5c"
    RED   = "#cc2200"
    GREEN = "#1a7a3a"
    GREY  = "#888888"

    labels = [r["scenario"] for r in rows]
    unhedged = [r["delta_bond"] for r in rows]
    hedged   = [r["delta_portfolio"] for r in rows]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(5, 4.5))
    fig.patch.set_facecolor("#f9f9f9")
    ax.set_facecolor("#f9f9f9")

    bars1 = ax.bar(x - width / 2, unhedged, width, label="Unhedged",
                   color=RED, edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x + width / 2, hedged, width, label="Hedged",
                   color=GREEN, edgecolor="white", linewidth=0.8)

    ax.axhline(0, color=GREY, linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6, rotation=35, ha="right")
    ax.set_ylabel("Δ Portfolio (EUR)", fontsize=8)
    ax.legend(fontsize=7, loc="best", framealpha=0.9)
    ax.tick_params(labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#cccccc")

    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, dpi=400, bbox_inches="tight", facecolor="#f9f9f9")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # 1. Load base data
    base_data = load_market_data()
    base_curve, eval_date, _ = build_curve(base_data)
    stripped_vols = _load_and_strip_vols(base_curve, eval_date)

    # 2. Base prices
    base_bond = reprice_bond(base_curve, eval_date, stripped_vols)
    base_bond_clean = base_bond["risky_clean"]

    hedge_ratios = load_hedge_ratios()

    swap_periods = {"2Y": ql.Period(2, ql.Years), "10Y": ql.Period(10, ql.Years)}
    base_swap_pvs = {}
    for tenor in ["2Y", "10Y"]:
        hr = hedge_ratios[tenor]
        base_swap_pvs[tenor] = compute_swap_pv(
            base_curve, eval_date, swap_periods[tenor],
            hr["par_rate"], hr["notional"], hr["direction"]
        )

    base_portfolio = base_bond_clean + base_swap_pvs["2Y"] + base_swap_pvs["10Y"]

    # 3. Load PCA data
    pca_tenors, eigenvals, loadings = load_pca_data()
    pc_names = ["Level (PC1)", "Slope (PC2)", "Curvature (PC3)"]
    sigma_mults = [-2, -1, 1, 2]

    # 4. Run scenarios
    rows = []

    print("\n" + "=" * 95)
    print("  QUESTION 14 -- Hedging Strategy Implementation & Performance")
    print("=" * 95)
    print(f"  Eval date          : {eval_date.to_date().isoformat()}")
    print(f"  Base bond clean    : EUR {base_bond_clean:.2f}")
    print(f"  Base swap 2Y PV    : EUR {base_swap_pvs['2Y']:.4f}")
    print(f"  Base swap 10Y PV   : EUR {base_swap_pvs['10Y']:.4f}")
    print(f"  Base portfolio     : EUR {base_portfolio:.2f}")
    print()

    hdr = (f"  {'Scenario':<22s} {'ΔBond':>9s} {'ΔSwap2Y':>9s} {'ΔSwap10Y':>9s} "
           f"{'ΔHedged':>9s} {'Reduction':>10s}")
    print(hdr)
    print(f"  {'-'*22} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*10}")

    for pc_k in range(3):
        for n_sig in sigma_mults:
            # Shock the curve
            shocked_data = apply_pca_shock(
                base_data, pca_tenors, loadings[pc_k], eigenvals[pc_k], n_sig
            )
            try:
                shocked_curve, _, _ = build_curve(shocked_data)
            except RuntimeError:
                continue

            # Reprice bond
            shocked_bond = reprice_bond(shocked_curve, eval_date, stripped_vols)
            delta_bond = shocked_bond["risky_clean"] - base_bond_clean

            # Reprice swaps
            delta_swaps = {}
            for tenor in ["2Y", "10Y"]:
                hr = hedge_ratios[tenor]
                shocked_pv = compute_swap_pv(
                    shocked_curve, eval_date, swap_periods[tenor],
                    hr["par_rate"], hr["notional"], hr["direction"]
                )
                delta_swaps[tenor] = shocked_pv - base_swap_pvs[tenor]

            delta_portfolio = delta_bond + delta_swaps["2Y"] + delta_swaps["10Y"]

            # Reduction %
            if abs(delta_bond) > 1e-8:
                reduction = (1.0 - abs(delta_portfolio) / abs(delta_bond)) * 100
            else:
                reduction = 0.0

            label = f"{pc_names[pc_k]} {n_sig:+d}σ"
            print(f"  {label:<22s} {delta_bond:+9.2f} {delta_swaps['2Y']:+9.2f} "
                  f"{delta_swaps['10Y']:+9.2f} {delta_portfolio:+9.2f} "
                  f"{reduction:9.1f}%")

            rows.append({
                "scenario": label,
                "pc_index": pc_k + 1,
                "n_sigma": n_sig,
                "delta_bond": round(delta_bond, 4),
                "delta_swap_2y": round(delta_swaps["2Y"], 4),
                "delta_swap_10y": round(delta_swaps["10Y"], 4),
                "delta_portfolio": round(delta_portfolio, 4),
                "reduction_pct": round(reduction, 2),
            })

    # 5. Hedge effectiveness summary
    unhedged_var = np.var([r["delta_bond"] for r in rows])
    hedged_var = np.var([r["delta_portfolio"] for r in rows])
    effectiveness = (1.0 - hedged_var / unhedged_var) * 100 if unhedged_var > 0 else 0

    print()
    print(f"  Hedge Effectiveness:")
    print(f"    Unhedged P&L variance  : {unhedged_var:.4f}")
    print(f"    Hedged P&L variance    : {hedged_var:.4f}")
    print(f"    Variance reduction     : {effectiveness:.1f}%")

    # 6. Save CSV
    out_csv = OUTPUT_DIR / "q14_hedge_performance.csv"
    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # 7. Chart
    out_chart = OUTPUT_DIR / "q14_hedge_chart.png"
    make_hedge_chart(rows, out_chart)

    print(f"\n  Saved: {out_csv}")
    print(f"  Saved: {out_chart}")
    print("=" * 95)


if __name__ == "__main__":
    main()
