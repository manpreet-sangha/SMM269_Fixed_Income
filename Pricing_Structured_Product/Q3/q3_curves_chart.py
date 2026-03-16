"""
q3_curves_chart.py
--------------------------------------------------------------------------------
Question 3 – Term Structure Visualisation

Generates a 3-panel figure:
  Panel 1 : Zero-coupon spot curve  (annually compounded zero rates, %)
  Panel 2 : 3-month forward EURIBOR curve (%)
  Panel 3 : Discount factor curve

Input pillar nodes (deposits + IRS) are overlaid as scatter points on the
spot curve panel to show where observed market quotes anchor the curve.

Output
------
  output/q3_curves.png
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import QuantLib as ql

sys.path.insert(0, str(Path(__file__).parent))
from q3_market_data import load_market_data
from q3_main   import build_curve, extract_curve_data, _ql_date

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Colours (match Q1 house style)
# ---------------------------------------------------------------------------
NAVY   = "#1a3a5c"
RED    = "#c0392b"
GREEN  = "#1a7a4a"
GOLD   = "#d4a017"


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------
def plot_curves(data: dict, curve, eval_date: ql.Date) -> Path:
    """
    Build and save the 3-panel term structure chart.

    Parameters
    ----------
    data       : dict from q3_market_data.load_market_data()
    curve      : bootstrapped ql.YieldTermStructure
    eval_date  : ql.Date

    Returns
    -------
    Path to saved PNG
    """
    rows     = extract_curve_data(curve, eval_date)
    years    = [r["years"]          for r in rows]
    zero_ann = [r["zero_rate_ann_%"] for r in rows]
    fwd_3m   = [r["fwd_3m_%"]        for r in rows]
    df_vals  = [r["discount_factor"] for r in rows]

    # --- Pillar mark coordinates ---
    cal = ql.TARGET()
    dc  = ql.Actual365Fixed()
    settlement = cal.advance(eval_date, 2, ql.Days)

    # Map each deposit/IRS tenor to its QuantLib period and fixing days
    from q3_main import _TENOR_PERIOD, _DEPOSIT_TENORS
    pillar_x, pillar_y = [], []

    # Deposits (skip O/N — matures before settlement, invisible on 10Y chart)
    for tenor in data["deposits"]:
        if tenor not in _DEPOSIT_TENORS or tenor == "ON":
            continue
        start = cal.advance(eval_date, 2, ql.Days)
        mat   = cal.advance(start, _TENOR_PERIOD[tenor], ql.ModifiedFollowing)
        t     = dc.yearFraction(eval_date, mat)
        if t <= 0:
            continue
        z     = curve.zeroRate(mat, dc, ql.Compounded, ql.Annual).rate() * 100
        pillar_x.append(t)
        pillar_y.append(z)

    # IRS
    for tenor in data["irs"]:
        mat = cal.advance(settlement, _TENOR_PERIOD[tenor], ql.ModifiedFollowing)
        t   = dc.yearFraction(eval_date, mat)
        z   = curve.zeroRate(mat, dc, ql.Compounded, ql.Annual).rate() * 100
        pillar_x.append(t)
        pillar_y.append(z)

    # -----------------------------------------------------------------------
    # Build figure — 1 row × 3 columns (wide & short for LaTeX embedding)
    # figsize=(6.3, 2.0): matches LaTeX textwidth, ~1/5 page height at 200dpi
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(
        1, 3, figsize=(6.3, 2.0), dpi=200,
        gridspec_kw={"wspace": 0.42},
    )

    xlabel_kw = dict(fontsize=6.5)
    ylabel_kw = dict(fontsize=6.5)
    title_kw  = dict(fontsize=7, fontweight="bold", pad=3)
    tick_kw   = dict(labelsize=6)

    # ------------------------------------------------------------------
    # Panel 1 – Zero (spot) curve
    # ------------------------------------------------------------------
    ax0 = axes[0]
    ax0.plot(years, zero_ann, color=NAVY, linewidth=1.2)
    ax0.scatter(pillar_x, pillar_y, color=RED, s=12, zorder=5,
                label="Pillars")
    ax0.set_ylabel("Rate (%)", **ylabel_kw)
    ax0.set_xlabel("Maturity (yrs)", **xlabel_kw)
    ax0.set_title("(a) Zero-Coupon Spot Curve", **title_kw)
    ax0.legend(fontsize=5.5, loc="lower right", handlelength=1)
    ax0.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax0.tick_params(**tick_kw)
    ax0.set_xlim(0, 10.5)
    ax0.grid(True, linestyle="--", alpha=0.4, linewidth=0.5)

    # ------------------------------------------------------------------
    # Panel 2 – 3M Forward curve
    # ------------------------------------------------------------------
    ax1 = axes[1]
    ax1.plot(years, fwd_3m, color=GREEN, linewidth=1.2)
    ax1.set_ylabel("Rate (%)", **ylabel_kw)
    ax1.set_xlabel("Maturity (yrs)", **xlabel_kw)
    ax1.set_title("(b) 3M Forward EURIBOR", **title_kw)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax1.tick_params(**tick_kw)
    ax1.set_xlim(0, 10.5)
    ax1.grid(True, linestyle="--", alpha=0.4, linewidth=0.5)
    ax1.text(0.03, 0.06, "Step = piecewise-const. fwds",
             transform=ax1.transAxes, fontsize=5, color="grey")

    # ------------------------------------------------------------------
    # Panel 3 – Discount factor curve
    # ------------------------------------------------------------------
    ax2 = axes[2]
    ax2.plot(years, df_vals, color=NAVY, linewidth=1.2)
    ax2.set_ylabel("Discount Factor", **ylabel_kw)
    ax2.set_xlabel("Maturity (yrs)", **xlabel_kw)
    ax2.set_title("(c) Discount Factor Curve", **title_kw)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax2.tick_params(**tick_kw)
    ax2.set_xlim(0, 10.5)
    ax2.grid(True, linestyle="--", alpha=0.4, linewidth=0.5)

    # 10Y DF annotation
    mat10 = cal.advance(eval_date, 10, ql.Years)
    df10  = curve.discount(mat10)
    ax2.annotate(
        f"10Y: {df10:.4f}",
        xy=(10, df10), xytext=(7.2, df10 + 0.012),
        fontsize=5.5, color=NAVY,
        arrowprops=dict(arrowstyle="-", color="grey", lw=0.6),
    )

    fig.suptitle(
        f"EUR Term Structure — Single-Curve Bootstrap  ({data['eval_date']})  "
        "| PiecewiseLogLinearDiscount | TARGET",
        fontsize=6.5, y=1.01,
    )

    out = OUTPUT_DIR / "q3_curves.png"
    plt.savefig(out, bbox_inches="tight", pad_inches=0.05, dpi=200)
    plt.close()
    print(f"  Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Standalone usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data                    = load_market_data()
    curve, eval_date, _     = build_curve(data)
    plot_curves(data, curve, eval_date)
