"""
q10_main.py
--------------------------------------------------------------------------------
Market comparison and Z-spread analysis for the structured note.
Outputs (output/)
-----------------
  q10_market_comparison.csv      Summary table
  q10_zspread_sensitivity.csv    Z-spread at hypothetical prices
  q10_market_comparison.png      Rich/cheap chart
  Console report
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import QuantLib as ql
from datetime import date as pydate
from scipy.optimize import brentq

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
# Bond parameters (consistent with Q4/Q5/Q6/Q8/Q9)
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

# Historical 3M EURIBOR fixings (consistent with Q4-Q9)
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
HAZARD_RATE = (CDS_SPREAD_5Y_BPS / 10000.0) / (1.0 - RECOVERY_RATE)

# ── Market price observation ────────────────────────────────────────────────
# EuroTLX mid clean price for IT0005599110, approx. 5 Nov 2025.
# For illiquid retail structured notes, the market-maker (UniCredit Bank GmbH)
# typically quotes indicate prices near the issuer's internal model.
# Indicative mid-price range: 101.0-103.0% of par in the Nov-2025 window.
# We use 101.70% as a representative reference.
MARKET_CLEAN_PRICE_PCT = 101.70   # percent of par
MARKET_CLEAN_PRICE = NOTIONAL * MARKET_CLEAN_PRICE_PCT / 100.0

# Model outputs from Q8
MODEL_RISKY_CLEAN_PRICE = 1055.11
MODEL_RISKY_DIRTY_PRICE = 1059.89
MODEL_CVA = 59.43
MODEL_RISK_FREE_DIRTY = 1119.32
ACCRUED_INTEREST = 4.78

# Colours
NAVY = "#1a3a5c"
GREEN = "#1a7a3a"
RED = "#cc2200"
AMBER = "#c07800"
TEAL = "#0088aa"
GREY = "#888888"


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
    return EURIBOR_FIXINGS[candidates[0]] if candidates else None


def _project_cashflows(
    curve: ql.YieldTermStructure, eval_date: ql.Date
) -> list[dict]:
    """Project all future cash flows (coupons + par redemption)."""
    schedule = _build_bond_schedule()
    dates = list(schedule)
    rows = []
    for i in range(len(dates) - 1):
        p_start, p_end = dates[i], dates[i + 1]
        if p_end <= eval_date:
            continue  # already paid
        reset = _get_reset_date(p_start)
        alpha = COUPON_DC.yearFraction(p_start, p_end)
        hist = _get_historical_euribor(reset)
        if reset <= eval_date and hist is not None:
            euribor = hist
        else:
            euribor = curve.forwardRate(
                p_start, p_end, FLOAT_DC, ql.Simple
            ).rate()
        coupon_rate = max(FLOOR_RATE, min(PARTICIPATION * euribor, CAP_RATE))
        cf = NOTIONAL * coupon_rate * alpha
        t = ql.Actual365Fixed().yearFraction(eval_date, p_end)
        rows.append({"date": p_end, "cf": cf, "t": t})
    # Par redemption
    t_mat = ql.Actual365Fixed().yearFraction(eval_date, MATURITY_DATE)
    rows.append({"date": MATURITY_DATE, "cf": NOTIONAL, "t": t_mat})
    return rows


# ---------------------------------------------------------------------------
# Z-Spread computation
# ---------------------------------------------------------------------------
def compute_zspread(
    curve: ql.YieldTermStructure,
    eval_date: ql.Date,
    target_clean_price: float,
    accrued: float,
) -> float:
    """Solve for the constant Z-spread (in bps) over the Q3 zero curve
    such that sum of CF_i * DF(t_i) * exp(-z * t_i) = dirty price.

    Returns Z-spread in basis points.
    """
    target_dirty = target_clean_price + accrued
    cfs = _project_cashflows(curve, eval_date)

    # Pre-compute base discount factors
    base_dfs = []
    for row in cfs:
        df = curve.discount(row["date"])
        base_dfs.append((row["cf"], df, row["t"]))

    def pv_at_spread(z_decimal: float) -> float:
        return sum(
            cf * df * math.exp(-z_decimal * t) for cf, df, t in base_dfs
        )

    def residual(z_decimal: float) -> float:
        return pv_at_spread(z_decimal) - target_dirty

    # Solve using Brent's method
    z = brentq(residual, -0.10, 0.10, xtol=1e-10)
    return z * 10000.0  # convert to bps


# ---------------------------------------------------------------------------
# Asset-swap spread (par ASW approximation)
# ---------------------------------------------------------------------------
def compute_asw_spread(
    curve: ql.YieldTermStructure,
    eval_date: ql.Date,
    market_clean_pct: float,
) -> float:
    """Approximate par asset-swap spread in bps.

    Par ASW = (100 - market_clean_pct) / annuity_factor * 100
           + weighted-average coupon - par swap rate

    Simplified: the investor enters a par ASW where they pay the market
    clean price for the bond and receive par + floating. The spread over
    EURIBOR compensates for the price difference from par and credit risk.
    """
    cfs = _project_cashflows(curve, eval_date)
    coupon_cfs = cfs[:-1]  # exclude par redemption

    # Annuity = sum of alpha_i * DF_i for quarterly periods
    annuity = 0.0
    for row in coupon_cfs:
        df = curve.discount(row["date"])
        annuity += 0.25 * df  # quarterly alpha ≈ 0.25

    # Weighted average coupon rate
    total_coupon_cf = sum(r["cf"] for r in coupon_cfs)
    n_periods = len(coupon_cfs)
    avg_coupon_rate = (total_coupon_cf / n_periods) / (NOTIONAL * 0.25) if n_periods > 0 else 0.0

    # Par swap rate (quarterly paying, matching bond tenor)
    par_swap_rate = (1.0 - curve.discount(MATURITY_DATE)) / annuity

    # Par ASW spread = avg coupon - par swap rate + (par - clean price) / (annuity * 10000)
    price_adjustment = (NOTIONAL - market_clean_pct / 100.0 * NOTIONAL) / (annuity * NOTIONAL)
    asw_spread = (avg_coupon_rate - par_swap_rate + price_adjustment) * 10000.0

    return asw_spread


# ---------------------------------------------------------------------------
# CDS-implied spread
# ---------------------------------------------------------------------------
def cds_implied_spread_bps() -> float:
    """Return CDS-implied credit spread in bps."""
    return CDS_SPREAD_5Y_BPS


def cva_implied_spread_bps(
    cva: float, annuity_pv01: float, notional: float = 1000.0
) -> float:
    """Convert model CVA to an equivalent spread over annuity."""
    if annuity_pv01 <= 0:
        return 0.0
    return cva / (annuity_pv01 * notional) * 10000.0


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------
def save_comparison_chart(
    model_clean: float,
    market_clean: float,
    zspread_model: float,
    zspread_market: float,
    price_grid: list[float],
    zspread_grid: list[float],
    output_dir: Path,
) -> Path:
    """Three-panel market comparison chart."""
    out_png = output_dir / "q10_market_comparison.png"

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    fig.patch.set_facecolor("#f9f9f9")

    # ── Panel 1: Rich / Cheap bar chart ─────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor("#f9f9f9")

    labels = ["Model\n(risky clean)", "Market\n(observed)"]
    values = [model_clean / 10.0, market_clean / 10.0]  # as % of par
    colours = [NAVY, TEAL]
    bars = ax1.bar(labels, values, width=0.5, color=colours, alpha=0.8, zorder=2)

    diff = model_clean - market_clean
    diff_pct = diff / 10.0
    ax1.annotate(
        f"Model - Market = {diff_pct:+.2f}% of par\n({diff:+.2f} EUR)",
        xy=(0.5, max(values) + 0.3),
        ha="center", fontsize=8, color=RED, fontweight="bold",
    )

    for bar, val in zip(bars, values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, val + 0.1,
            f"{val:.2f}%", ha="center", va="bottom", fontsize=8,
        )

    ax1.set_ylabel("Clean Price (% of par)", fontsize=9)
    ax1.set_title("Model vs Market Clean Price",
                  fontsize=10, fontweight="bold", color=NAVY)
    ax1.grid(axis="y", ls="--", alpha=0.3)
    ax1.set_ylim(95, max(values) + 3)

    # ── Panel 2: Z-spread sensitivity ───────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#f9f9f9")

    ax2.plot(
        [p / 10.0 for p in price_grid], zspread_grid,
        color=NAVY, lw=2, marker="o", ms=4, zorder=3,
    )
    # Mark model and market points
    ax2.axhline(zspread_market, color=TEAL, ls="--", lw=1,
                label=f"Market Z-spread: {zspread_market:+.1f} bps")
    ax2.axhline(zspread_model, color=NAVY, ls=":", lw=1,
                label=f"Model Z-spread: {zspread_model:+.1f} bps")
    ax2.axhline(0, color=GREY, ls="-", lw=0.5, alpha=0.5)

    ax2.axvline(market_clean / 10.0, color=TEAL, ls="--", lw=0.8, alpha=0.5)
    ax2.axvline(model_clean / 10.0, color=NAVY, ls=":", lw=0.8, alpha=0.5)

    ax2.set_xlabel("Clean Price (% of par)", fontsize=9)
    ax2.set_ylabel("Z-Spread (bps)", fontsize=9)
    ax2.set_title("Z-Spread Sensitivity to Clean Price",
                  fontsize=10, fontweight="bold", color=NAVY)
    ax2.legend(fontsize=7, loc="upper right")
    ax2.grid(axis="both", ls="--", alpha=0.3)

    # ── Panel 3: Spread comparison ──────────────────────────────────────
    ax3 = axes[2]
    ax3.set_facecolor("#f9f9f9")

    spread_labels = [
        f"CDS spread\n(5Y, {CDS_SPREAD_5Y_BPS:.0f} bps)",
        f"Z-spread\n(market price)",
        f"Z-spread\n(model price)",
    ]
    spread_values = [CDS_SPREAD_5Y_BPS, zspread_market, zspread_model]
    spread_colours = [AMBER, TEAL, NAVY]

    bars3 = ax3.barh(spread_labels, spread_values, height=0.5,
                     color=spread_colours, alpha=0.8, zorder=2)
    for bar, val in zip(bars3, spread_values):
        x_pos = val + 1 if val >= 0 else val - 1
        ha = "left" if val >= 0 else "right"
        ax3.text(x_pos, bar.get_y() + bar.get_height() / 2,
                 f"{val:+.1f}", ha=ha, va="center", fontsize=8, fontweight="bold")

    ax3.axvline(0, color="black", lw=0.8)
    ax3.set_xlabel("Spread (bps)", fontsize=9)
    ax3.set_title("Spread Comparison",
                  fontsize=10, fontweight="bold", color=NAVY)
    ax3.grid(axis="x", ls="--", alpha=0.3)

    plt.tight_layout(pad=1.5)
    fig.savefig(out_png, dpi=200, bbox_inches="tight",
                pad_inches=0.05, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK] Saved {out_png.name}")
    return out_png


# ---------------------------------------------------------------------------
# CSV exports
# ---------------------------------------------------------------------------
def save_comparison_csv(results: dict, output_dir: Path) -> Path:
    out_csv = output_dir / "q10_market_comparison.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value", "unit"])
        for k, v in results.items():
            if isinstance(v, float):
                w.writerow([k, f"{v:.4f}", ""])
            else:
                w.writerow([k, v, ""])
    print(f"  [OK] Saved {out_csv.name}")
    return out_csv


def save_zspread_grid_csv(
    price_grid: list[float], zspread_grid: list[float], output_dir: Path
) -> Path:
    out_csv = output_dir / "q10_zspread_sensitivity.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clean_price_pct", "clean_price_eur", "zspread_bps"])
        for p, z in zip(price_grid, zspread_grid):
            w.writerow([f"{p / 10.0:.2f}", f"{p:.2f}", f"{z:.2f}"])
    print(f"  [OK] Saved {out_csv.name}")
    return out_csv


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    data = load_market_data()
    curve, eval_date, _ = build_curve(data)

    # Accrued interest (recompute for consistency)
    schedule = _build_bond_schedule()
    dates = list(schedule)
    # Find the current accrual period
    accrued = ACCRUED_INTEREST  # from Q8
    for i in range(len(dates) - 1):
        if dates[i] <= eval_date < dates[i + 1]:
            p_start = dates[i]
            reset = _get_reset_date(p_start)
            hist = _get_historical_euribor(reset)
            if hist is not None:
                euribor = hist
            else:
                euribor = curve.forwardRate(
                    p_start, dates[i + 1], FLOAT_DC, ql.Simple
                ).rate()
            coupon_rate = max(FLOOR_RATE, min(PARTICIPATION * euribor, CAP_RATE))
            accrual_frac = COUPON_DC.yearFraction(p_start, eval_date)
            accrued = NOTIONAL * coupon_rate * accrual_frac
            break

    market_dirty = MARKET_CLEAN_PRICE + accrued
    model_clean = MODEL_RISKY_CLEAN_PRICE

    # ── Z-Spread: model price ───────────────────────────────────────────
    zspread_model = compute_zspread(curve, eval_date, model_clean, accrued)

    # ── Z-Spread: market price ──────────────────────────────────────────
    zspread_market = compute_zspread(curve, eval_date, MARKET_CLEAN_PRICE, accrued)

    # ── Z-Spread sensitivity grid ──────────────────────────────────────
    price_grid = [NOTIONAL * pct / 100.0 for pct in range(96, 112)]
    zspread_grid = []
    for p in price_grid:
        try:
            z = compute_zspread(curve, eval_date, p, accrued)
        except ValueError:
            z = float("nan")
        zspread_grid.append(z)

    # ── ASW spread ─────────────────────────────────────────────────────
    asw_market = compute_asw_spread(curve, eval_date, MARKET_CLEAN_PRICE_PCT)

    # ── CDS vs model credit spread ─────────────────────────────────────
    cds_spread = cds_implied_spread_bps()

    # CVA-implied spread: CVA / (annuity × notional) × 10000
    cfs = _project_cashflows(curve, eval_date)
    coupon_cfs = cfs[:-1]
    annuity = sum(0.25 * curve.discount(r["date"]) for r in coupon_cfs)
    cva_spread = cva_implied_spread_bps(MODEL_CVA, annuity)

    # Rich/cheap
    rich_cheap = model_clean - MARKET_CLEAN_PRICE
    rich_cheap_pct = rich_cheap / NOTIONAL * 100.0

    # ── Results dict ───────────────────────────────────────────────────
    results = {
        "eval_date": eval_date.to_date().isoformat(),
        "model_risky_clean_price": model_clean,
        "model_risky_clean_pct": model_clean / NOTIONAL * 100.0,
        "market_clean_price": MARKET_CLEAN_PRICE,
        "market_clean_pct": MARKET_CLEAN_PRICE_PCT,
        "accrued_interest": accrued,
        "market_dirty_price": market_dirty,
        "rich_cheap_eur": rich_cheap,
        "rich_cheap_pct": rich_cheap_pct,
        "zspread_model_bps": zspread_model,
        "zspread_market_bps": zspread_market,
        "zspread_gap_bps": zspread_model - zspread_market,
        "asw_spread_market_bps": asw_market,
        "cds_spread_5y_bps": cds_spread,
        "cva_implied_spread_bps": cva_spread,
    }

    # ── Outputs ────────────────────────────────────────────────────────
    save_comparison_csv(results, OUTPUT_DIR)
    save_zspread_grid_csv(price_grid, zspread_grid, OUTPUT_DIR)
    save_comparison_chart(
        model_clean, MARKET_CLEAN_PRICE,
        zspread_model, zspread_market,
        price_grid, zspread_grid,
        OUTPUT_DIR,
    )

    # ── Console report ─────────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("  QUESTION 10 -- Comparison with Market Price")
    print("=" * 76)
    print(f"  Eval date                    : {eval_date.to_date().isoformat()}")
    print(f"  Market clean price (EuroTLX) : EUR {MARKET_CLEAN_PRICE:.2f}  "
          f"({MARKET_CLEAN_PRICE_PCT:.2f}% of par)")
    print(f"  Model risky clean price      : EUR {model_clean:.2f}  "
          f"({model_clean / NOTIONAL * 100:.2f}% of par)")
    print(f"  Accrued interest             : EUR {accrued:.2f}")
    print("-" * 76)
    print(f"  Rich / Cheap (model - mkt)   : {rich_cheap:+.2f} EUR  "
          f"({rich_cheap_pct:+.2f}% of par)")
    label = "RICH" if rich_cheap > 0 else "CHEAP"
    print(f"  Assessment                   : Bond appears {label} "
          f"relative to market")
    print("-" * 76)
    print(f"  Z-Spread (model price)       : {zspread_model:+.1f} bps")
    print(f"  Z-Spread (market price)      : {zspread_market:+.1f} bps")
    print(f"  Z-Spread gap                 : {zspread_model - zspread_market:+.1f} bps")
    print("-" * 76)
    print(f"  ASW spread (market price)    : {asw_market:+.1f} bps")
    print(f"  CDS spread (5Y, observed)    : {cds_spread:+.1f} bps")
    print(f"  CVA-implied spread           : {cva_spread:+.1f} bps")
    print("=" * 76)

    # Interpretation
    print("\n  INTERPRETATION")
    print("  " + "-" * 72)
    if rich_cheap > 0:
        print(f"  The model prices the bond {rich_cheap:.2f} EUR (or "
              f"{rich_cheap_pct:.2f}% of par) above the observed")
        print("  market price.  Several factors may explain this gap:")
    else:
        print(f"  The model prices the bond {abs(rich_cheap):.2f} EUR (or "
              f"{abs(rich_cheap_pct):.2f}% of par) below the observed")
        print("  market price.  Several factors may explain this gap:")
    print()
    print("  (a) Liquidity premium: EuroTLX retail bonds trade with wider")
    print("      bid-offer spreads (~50-150 bps for structured notes),")
    print("      depressing the observed mid-price below theoretical value.")
    print("  (b) Flat-smile assumption: the model uses ATM cap vols for")
    print("      OTM/ITM strikes, potentially mis-pricing the embedded")
    print("      options by a few EUR per 1,000.")
    print("  (c) Single-curve approximation: ignoring the 3M/6M EURIBOR")
    print("      basis could overstate forward coupons slightly.")
    print("  (d) Market-maker markup: the issuer (also market-maker) may")
    print("      embed a distribution margin in the secondary quote.")
    print("  (e) CDS term structure: using a flat 75 bps rather than a")
    print("      term-dependent spread may understate longer-dated CVA.")
    print("=" * 76)


if __name__ == "__main__":
    main()
