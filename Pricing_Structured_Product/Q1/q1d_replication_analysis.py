"""
q1d_replication_analysis.py
================================================================================
Called from  : q1_main.py
Inputs       : details dict + ql_pvs dict (from ql_replication_pv)
Outputs      : console report  +  output/q1d_replication_analysis.txt
               output/q1d_replication_payoff.png
================================================================================
"""

from __future__ import annotations

import sys
import re
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_float(details: dict, key: str, default: float = 0.0) -> float:
    raw = details.get(key, str(default))
    numeric = re.sub(r"[^0-9.\-]", "", raw.split()[0])
    try:
        return float(numeric)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def _save_replication_png(
    participation: float,
    floor_rate: float,
    cap_rate: float,
    current_euribor: float,
    ql_pvs: dict,
    output_dir: Path,
) -> Path:
    """
    Two-panel figure:
      Left  — per-period coupon payoff decomposed into its three components
              (FRN leg, floor payoff, cap cost) as a function of 3M EURIBOR.
      Right — QuantLib PV bar chart of the four replicating instruments.
    """
    out_png = output_dir / "q1d_replication_payoff.png"

    NAVY    = "#1a3a5c"
    GREEN   = "#2a7a2a"
    RED     = "#cc2200"
    ORANGE  = "#e07000"
    BLUE    = "#3060b0"
    LGREY   = "#f3f3f3"
    GRID    = "#d0d0d0"

    r = np.linspace(-0.005, 0.065, 800)
    frn_leg  = participation * r                           # uncapped FRN coupon
    floor_payout = np.maximum(-frn_leg, 0)                # long floor payoff
    cap_cost     = np.maximum(frn_leg - cap_rate, 0)      # short cap cost (positive = loss)
    net_coupon   = np.clip(frn_leg, floor_rate, cap_rate)

    be_cap  = cap_rate   / participation
    be_floor = floor_rate / participation

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor("white")

    # ── LEFT: payoff decomposition ──────────────────────────────────────────
    ax1.set_facecolor(LGREY)
    ax1.grid(True, color=GRID, linewidth=0.6, zorder=0)

    ax1.plot(r * 100, frn_leg * 100, "--", lw=1.5, color=BLUE, alpha=0.7,
             label=f"(1) FRN leg  = {participation:.2f}×r  [uncapped]")
    ax1.fill_between(r * 100, 0, floor_payout * 100,
                     alpha=0.18, color=GREEN,
                     label=f"(2) Long Floor payoff  (strike 0%)")
    ax1.fill_between(r * 100, 0, -cap_cost * 100,
                     alpha=0.18, color=RED,
                     label=f"(3) Short Cap cost  (strike {cap_rate*100:.2f}%)")
    ax1.plot(r * 100, net_coupon * 100, "-", lw=2.5, color=NAVY,
             label=r"Net coupon $c(r)$ = (1)+(2)−(3)", zorder=4)

    ax1.axvline(be_floor * 100, color=GREEN,  ls=":", lw=1.2,
                label=f"Floor break-even = {be_floor*100:.3f}%")
    ax1.axvline(be_cap   * 100, color=RED,    ls=":", lw=1.2,
                label=f"Cap break-even   = {be_cap*100:.4f}%")
    ax1.axvline(current_euribor * 100, color=ORANGE, ls="-.", lw=1.2,
                label=f"Current EURIBOR  = {current_euribor*100:.3f}%  (Sep-10 reset)")
    ax1.scatter([current_euribor * 100],
                [np.clip(participation * current_euribor, floor_rate, cap_rate) * 100],
                color=ORANGE, s=70, zorder=5)

    ax1.set_xlabel("3M EURIBOR (%)", fontsize=10)
    ax1.set_ylabel("Per-period coupon rate (%)", fontsize=10)
    ax1.set_title("Coupon Payoff Decomposition\nc(r) = FRN leg + Floor − Cap",
                  fontsize=11, fontweight="bold", color=NAVY)
    ax1.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    ax1.set_xlim(r[0] * 100, r[-1] * 100)

    # ── RIGHT: QuantLib PV bar chart ─────────────────────────────────────────
    frn_pv  = ql_pvs["leveraged_frn_pv"]
    flo_pv  = ql_pvs["long_floor_pv"]
    cap_pv  = ql_pvs["short_cap_pv"]
    red_pv  = ql_pvs["redemption_pv"]
    tot_pv  = ql_pvs["total_bond_pv"]
    notional = ql_pvs["notional"]

    labels = [
        "(1) Leveraged\nFRN  (1.6×r)",
        "(2) Long\nFloor  0%",
        "(3) Short\nCap  5.45%",
        "(4) Redemption\n@ par",
    ]
    values = [frn_pv, flo_pv, -cap_pv, red_pv]   # cap cost is negative to bond value
    colours = [BLUE, GREEN, RED, ORANGE]

    ax2.set_facecolor(LGREY)
    ax2.grid(True, color=GRID, linewidth=0.6, axis="y", zorder=0)

    bars = ax2.bar(labels, values, color=colours, alpha=0.82, edgecolor="white",
                   linewidth=0.8, zorder=3)

    # annotate each bar
    for bar, val in zip(bars, values):
        yoff = 4 if val >= 0 else -14
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + yoff,
                 f"EUR {val:+.2f}", ha="center", va="bottom", fontsize=8.5,
                 fontweight="bold")

    # total line
    ax2.axhline(tot_pv, color=NAVY, ls="--", lw=1.5, zorder=4,
                label=f"Total Fair Value = EUR {tot_pv:.2f}  "
                      f"({tot_pv/notional*100:.2f}% of par)")
    ax2.legend(fontsize=8, loc="lower right", framealpha=0.9)

    ax2.set_ylabel("Present Value (EUR per EUR 1,000 bond)", fontsize=10)
    ax2.set_title(
        f"QuantLib PV by Component\n"
        f"Valuation: {ql_pvs['pricing_date']}  |  "
        f"3M EURIBOR {ql_pvs['implied_euribor_pct']:.4f}%  |  "
        f"Normal vol {ql_pvs['normal_vol_bps']:.0f} bps",
        fontsize=11, fontweight="bold", color=NAVY,
    )

    plt.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
# Main function — PNG output only
# ---------------------------------------------------------------------------

def print_replication_analysis(
    details: dict[str, str],
    ql_pvs: dict[str, float],
    output_dir: Path | None = None,
) -> None:
    """Generate q1d_replication_payoff.png (no text output)."""
    if output_dir is not None:
        participation  = _get_float(details, "Participation Factor", 1.60)
        floor_rate     = _get_float(details, "Floor (Min Rate)",     0.00) / 100
        cap_rate       = _get_float(details, "Cap (Max Rate)",       5.45) / 100
        current_coupon = _get_float(details, "Current Coupon Rate",  3.2464)
        current_euribor = current_coupon / participation / 100

        output_dir.mkdir(parents=True, exist_ok=True)
        _save_replication_png(
            participation, floor_rate, cap_rate, current_euribor, ql_pvs, output_dir
        )
