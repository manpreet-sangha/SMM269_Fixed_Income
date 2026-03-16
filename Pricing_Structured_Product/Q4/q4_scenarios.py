"""
q4_scenarios.py
--------------------------------------------------------------------------------
Point 4 -- Best- and Worst-Case Scenario Analysis

Computes three valuation scenarios for the UniCredit Variable Rate Bond 2034
(ISIN IT0005599110) per EUR 1,000 face value:

  Best case  : EURIBOR >= cap break-even (3.406%) at every future reset
               -> every coupon pays the maximum 5.45% p.a.
  Base case  : Forward EURIBOR from the bootstrapped Q3 curve at each reset
               -> coupons follow the leveraged-collar formula
  Worst case : EURIBOR <= 0% at every future reset
               -> every coupon pays the floor (0% p.a.)

For each scenario the script:
  1. Projects the quarterly coupon cash flows (30/360, Mod-Following, TARGET)
  2. Discounts them using the bootstrapped EUR curve from Q3
  3. Adds the par redemption PV
  4. Reports total PV, coupon PV, and annualised yield

Uses historical fixings for already-paid periods.

Outputs
-------
  output/q4_scenarios.png     Three-panel comparison chart
  output/q4_scenarios.csv     Summary table
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
import numpy as np
import QuantLib as ql

# ---------------------------------------------------------------------------
# Paths & imports from Q3
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
Q3_DIR     = SCRIPT_DIR.parent / "Q3"
sys.path.insert(0, str(Q3_DIR))
from Q3.q3_market_data import load_market_data
from Q3.q3_main import build_curve

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Bond parameters
# ---------------------------------------------------------------------------
ISSUE_DATE    = ql.Date(12, 6, 2024)
MATURITY_DATE = ql.Date(12, 6, 2034)
PARTICIPATION = 1.60
FLOOR_RATE    = 0.00       # 0% p.a.
CAP_RATE      = 0.0545     # 5.45% p.a.
NOTIONAL      = 1000.0     # EUR per bond
DC            = ql.Thirty360(ql.Thirty360.BondBasis)
CALENDAR      = ql.TARGET()

# Historical 3M EURIBOR fixings (Bank of Finland)
EURIBOR_FIXINGS = {
    ql.Date(10, 6, 2024):  0.037430,
    ql.Date(10, 9, 2024):  0.034600,
    ql.Date(10, 12, 2024): 0.028720,
    ql.Date(10, 3, 2025):  0.025470,
    ql.Date(10, 6, 2025):  0.019540,
    ql.Date(10, 9, 2025):  0.020290,
    ql.Date(10, 12, 2025): 0.020820,
}


# ---------------------------------------------------------------------------
# Schedule builder
# ---------------------------------------------------------------------------
def build_schedule() -> ql.Schedule:
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


def get_reset_date(period_start: ql.Date) -> ql.Date:
    """Reset = 2nd TARGET business day before period start."""
    reset = period_start - ql.Period(2, ql.Days)
    while not CALENDAR.isBusinessDay(reset):
        reset += ql.Period(1, ql.Days)
    return reset


def get_historical_euribor(reset_date: ql.Date) -> float | None:
    """Return historical fixing if available, else None."""
    if reset_date in EURIBOR_FIXINGS:
        return EURIBOR_FIXINGS[reset_date]
    candidates = sorted(
        [d for d in EURIBOR_FIXINGS if d <= reset_date], reverse=True
    )
    if candidates:
        return EURIBOR_FIXINGS[candidates[0]]
    return None


# ---------------------------------------------------------------------------
# Scenario engine
# ---------------------------------------------------------------------------
def compute_scenarios(curve, eval_date: ql.Date) -> dict:
    """
    For each scenario, compute quarterly cash flows and PVs.

    Returns dict with keys: 'best', 'base', 'worst', each containing:
        periods    : list of period dicts
        coupon_pv  : float
        redemption_pv : float
        total_pv   : float
    """
    schedule = build_schedule()
    dates = list(schedule)
    today = ql.Date(15, 3, 2026)

    results = {}

    for scenario in ("best", "base", "worst"):
        periods = []
        coupon_pv = 0.0

        for i in range(len(dates) - 1):
            p_start = dates[i]
            p_end   = dates[i + 1]
            reset   = get_reset_date(p_start)
            alpha   = DC.yearFraction(p_start, p_end)

            # Determine EURIBOR for this period
            hist = get_historical_euribor(reset)
            if p_end <= today and hist is not None:
                # Already paid -- use actual fixing regardless of scenario
                euribor = hist
            else:
                # Future period -- scenario-dependent
                if scenario == "best":
                    euribor = CAP_RATE / PARTICIPATION + 0.01  # above cap break-even
                elif scenario == "worst":
                    euribor = -0.005  # below floor break-even
                else:
                    # Base case: forward rate from bootstrapped curve
                    fwd = curve.forwardRate(
                        p_start, p_end, ql.Actual360(), ql.Simple
                    ).rate()
                    euribor = fwd

            # Apply collar
            leveraged = PARTICIPATION * euribor
            coupon_rate = max(FLOOR_RATE, min(leveraged, CAP_RATE))
            coupon_amt = NOTIONAL * coupon_rate * alpha

            # Discount factor -- only for future cash flows
            if p_end <= eval_date:
                # Already in the past relative to curve; no PV contribution
                df = 0.0
                pv = 0.0
            else:
                df = curve.discount(p_end)
                pv = coupon_amt * df

            coupon_pv += pv

            periods.append({
                "number":      i + 1,
                "start":       p_start,
                "end":         p_end,
                "reset":       reset,
                "euribor":     euribor,
                "coupon_rate": coupon_rate,
                "coupon_amt":  coupon_amt,
                "df":          df,
                "pv":          pv,
                "is_paid":     p_end <= today,
            })

        redemption_pv = NOTIONAL * curve.discount(MATURITY_DATE)
        total_pv = coupon_pv + redemption_pv

        results[scenario] = {
            "periods":       periods,
            "coupon_pv":     coupon_pv,
            "redemption_pv": redemption_pv,
            "total_pv":      total_pv,
        }

    return results


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------
def save_scenario_chart(results: dict, output_dir: Path) -> Path:
    """Three-panel figure: cash flow profiles + PV comparison bar chart."""
    out_png = output_dir / "q4_scenarios.png"
    NAVY  = "#1a3a5c"
    GREEN = "#1a7a3a"
    RED   = "#cc2200"
    AMBER = "#c07800"

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.patch.set_facecolor("#f9f9f9")

    scenario_meta = {
        "best":  ("Best Case (Cap-Binding)", GREEN),
        "base":  ("Base Case (Forward Curve)", NAVY),
        "worst": ("Worst Case (Floor-Binding)", RED),
    }

    # -- Panels 1 & 2: Cash flow timelines for best and worst ----------------
    for idx, scen in enumerate(["best", "worst"]):
        ax = axes[idx]
        ax.set_facecolor("#f9f9f9")
        label, colour = scenario_meta[scen]
        periods = results[scen]["periods"]

        # Only future periods for scenario difference; paid are same
        future = [p for p in periods if not p["is_paid"]]
        paid   = [p for p in periods if p["is_paid"]]

        # Paid periods (grey)
        if paid:
            x_paid = range(1, len(paid) + 1)
            y_paid = [p["coupon_amt"] for p in paid]
            ax.bar(x_paid, y_paid, color="grey", alpha=0.5, label="Paid (actual)")

        # Future periods (scenario colour)
        if future:
            x_fut = range(len(paid) + 1, len(periods) + 1)
            y_fut = [p["coupon_amt"] for p in future]
            ax.bar(x_fut, y_fut, color=colour, alpha=0.75, label=f"Future ({scen})")

        ax.set_xlabel("Coupon Period", fontsize=9)
        ax.set_ylabel("Coupon (EUR per 1,000)", fontsize=9)
        ax.set_title(label, fontsize=10, fontweight="bold", color=colour)
        ax.legend(fontsize=7.5, loc="upper right")
        ax.grid(axis="y", ls="--", alpha=0.3)
        ax.set_ylim(0, 16)

    # -- Panel 3: PV comparison bar chart -------------------------------------
    ax3 = axes[2]
    ax3.set_facecolor("#f9f9f9")

    scenarios = ["best", "base", "worst"]
    labels = ["Best\n(Cap)", "Base\n(Forward)", "Worst\n(Floor)"]
    colours = [GREEN, NAVY, RED]

    coupon_pvs     = [results[s]["coupon_pv"] for s in scenarios]
    redemption_pvs = [results[s]["redemption_pv"] for s in scenarios]
    total_pvs      = [results[s]["total_pv"] for s in scenarios]

    x = np.arange(len(scenarios))
    w = 0.35

    ax3.bar(x - w/2, coupon_pvs, w, label="Coupon PV", color=colours, alpha=0.7)
    ax3.bar(x + w/2, redemption_pvs, w, label="Redemption PV", color="steelblue", alpha=0.5)

    # Total PV markers
    for i, (tv, col) in enumerate(zip(total_pvs, colours)):
        ax3.plot(i, tv, "D", color=col, ms=8, zorder=5)
        ax3.annotate(f"EUR {tv:.2f}", xy=(i, tv), xytext=(i, tv + 15),
                     fontsize=8, ha="center", fontweight="bold", color=col)

    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=9)
    ax3.set_ylabel("Present Value (EUR per 1,000)", fontsize=9)
    ax3.set_title("Fair Value Comparison", fontsize=10, fontweight="bold", color=NAVY)
    ax3.legend(fontsize=8)
    ax3.grid(axis="y", ls="--", alpha=0.3)

    plt.tight_layout(pad=1.5)
    fig.savefig(out_png, dpi=200, bbox_inches="tight",
                pad_inches=0.05, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK] Saved {out_png.name}")
    return out_png


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------
def save_scenario_csv(results: dict, output_dir: Path) -> Path:
    out_csv = output_dir / "q4_scenarios.csv"
    rows = []
    for scen in ("best", "base", "worst"):
        r = results[scen]
        rows.append({
            "Scenario":       scen.title(),
            "Coupon PV (EUR)": f"{r['coupon_pv']:.4f}",
            "Redemption PV (EUR)": f"{r['redemption_pv']:.4f}",
            "Total PV (EUR)": f"{r['total_pv']:.4f}",
            "Total PV (% par)": f"{r['total_pv'] / NOTIONAL * 100:.4f}",
        })
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"  [OK] Saved {out_csv.name}")
    return out_csv


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------
def print_report(results: dict) -> None:
    print()
    print("=" * 72)
    print("  POINT 4 -- Best / Base / Worst Scenario Analysis")
    print("  UniCredit Variable Rate Bond 2034  (per EUR 1,000 face)")
    print("=" * 72)

    for scen in ("best", "base", "worst"):
        r = results[scen]
        tag = {"best": "BEST (all coupons at cap 5.45%)",
               "base": "BASE (forward EURIBOR from curve)",
               "worst": "WORST (all coupons at floor 0%)"}[scen]
        print(f"\n  --- {tag} ---")
        print(f"  Coupon PV      : EUR {r['coupon_pv']:>10.4f}")
        print(f"  Redemption PV  : EUR {r['redemption_pv']:>10.4f}")
        print(f"  Total PV       : EUR {r['total_pv']:>10.4f}  "
              f"({r['total_pv']/NOTIONAL*100:.2f}% of par)")

        # Count future periods and show scenario coupon rate
        future = [p for p in r["periods"] if not p["is_paid"]]
        if future:
            avg_rate = np.mean([p["coupon_rate"] for p in future]) * 100
            print(f"  Future periods : {len(future)}")
            print(f"  Avg coupon rate: {avg_rate:.4f}% p.a.")

    # Spread
    best_pv  = results["best"]["total_pv"]
    worst_pv = results["worst"]["total_pv"]
    base_pv  = results["base"]["total_pv"]
    print(f"\n  Range (best - worst) : EUR {best_pv - worst_pv:.2f}  "
          f"({(best_pv - worst_pv)/NOTIONAL*100:.2f}% of par)")
    print(f"  Base vs par          : {(base_pv/NOTIONAL - 1)*100:+.2f}%")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    data = load_market_data()
    curve, eval_date, _ = build_curve(data)

    print("  Computing scenarios ...")
    results = compute_scenarios(curve, eval_date)

    print_report(results)
    save_scenario_csv(results, OUTPUT_DIR)
    save_scenario_chart(results, OUTPUT_DIR)


if __name__ == "__main__":
    main()
