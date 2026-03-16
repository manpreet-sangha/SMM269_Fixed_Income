"""
q12_main.py
--------------------------------------------------------------------------------
Uses the principal components from Q11 (level / slope / curvature) to apply
historically calibrated shocks to the EUR IRS curve, then re-prices the bond
under each scenario.


Outputs (output/)
-----------------
  q12_sensitivity_table.csv     Full scenario results
  q12_sensitivity_chart.png     Bar chart of price changes
  Console report
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import copy
import csv
import math
import sys
from collections import OrderedDict
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
# Bond parameters (consistent with Q5/Q6/Q8)
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

# Historical 3M EURIBOR fixings
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

# Scaling: daily 1-σ → 1-month horizon (≈21 business days)
HORIZON_DAYS = 21
HORIZON_SCALE = math.sqrt(HORIZON_DAYS)


# ---------------------------------------------------------------------------
# Load PCA eigenvectors and eigenvalues from Q11 CSV outputs
# ---------------------------------------------------------------------------
def load_pca_results() -> tuple[list[str], list[float], list[list[float]]]:
    """
    Load PCA eigenvalues and eigenvector loadings from Q11 output CSVs.

    Returns
    -------
    tenors     : list of tenor labels, e.g. ["1Y", "2Y", ..., "10Y"]
    eigenvals  : list of eigenvalues (bps²) for first 3 PCs
    loadings   : list of 3 lists, each being the eigenvector loadings per tenor
    """
    q11_dir = ROOT_DIR / "Q11" / "output"

    # Eigenvalues
    eigenvals: list[float] = []
    with open(q11_dir / "q11_pca_eigenvalues.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            eigenvals.append(float(row["eigenvalue_bps2"]))
            if len(eigenvals) == 3:
                break

    # Loadings
    tenors: list[str] = []
    loadings: list[list[float]] = [[], [], []]  # PC1, PC2, PC3
    with open(q11_dir / "q11_pca_loadings.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tenors.append(row["tenor"])
            loadings[0].append(float(row["PC1"]))
            loadings[1].append(float(row["PC2"]))
            loadings[2].append(float(row["PC3"]))

    return tenors, eigenvals, loadings


# ---------------------------------------------------------------------------
# Apply a PCA shock to market data
# ---------------------------------------------------------------------------
def apply_pca_shock(
    base_data: dict,
    pca_tenors: list[str],
    eigenvector: list[float],
    eigenvalue_bps2: float,
    n_sigma: float,
) -> dict:
    """
    Apply an additive shock to IRS par rates based on a single PC.

    Shock per tenor (in percentage points):
        Δr = n_sigma × sqrt(eigenvalue_bps2) × HORIZON_SCALE × loading / 100

    The division by 100 converts bps→percentage points (since IRS rates
    in the market data dict are in % units, e.g. 2.13 for 2.13%).
    """
    shocked = copy.deepcopy(base_data)
    sigma_daily_bps = math.sqrt(eigenvalue_bps2)

    # Map PCA tenor labels to IRS tenor keys
    for idx, pca_tenor in enumerate(pca_tenors):
        irs_key = pca_tenor  # e.g. "1Y", "2Y" etc.
        if irs_key in shocked["irs"]:
            delta_bps = n_sigma * sigma_daily_bps * HORIZON_SCALE * eigenvector[idx]
            delta_pct = delta_bps / 100.0  # bps → percentage points
            shocked["irs"][irs_key] += delta_pct

    # Also shock deposits proportionally to the 1Y loading (short-end proxy)
    if pca_tenors and pca_tenors[0] == "1Y":
        delta_1y_bps = n_sigma * sigma_daily_bps * HORIZON_SCALE * eigenvector[0]
        delta_1y_pct = delta_1y_bps / 100.0
        # Apply a fraction of the 1Y shock to deposits
        # 6M gets ~80% of the 1Y shock, 3M ~60%, 1M ~40%, 1W/ON ~20%
        dep_fracs = {"6M": 0.80, "3M": 0.60, "1M": 0.40, "1W": 0.20, "ON": 0.20}
        for dep_tenor, frac in dep_fracs.items():
            if dep_tenor in shocked["deposits"]:
                shocked["deposits"][dep_tenor] += frac * delta_1y_pct

    return shocked


# ---------------------------------------------------------------------------
# Bond schedule helpers (consistent with Q8)
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
# Option pricing (reuses Q5 logic inline for self-containment)
# ---------------------------------------------------------------------------
MARKET_FILE = ROOT_DIR / "common" / "input" / "MarketData202526.xlsx"


def _load_and_strip_vols(curve, eval_date):
    """Load Bachelier ATM quotes and strip caplet vols (reusing Q5 logic)."""
    import openpyxl

    wb = openpyxl.load_workbook(MARKET_FILE, read_only=True, data_only=True)
    ws = wb["Bachelier"]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    header = rows[4]
    stk_col = atm_col = None
    for idx, value in enumerate(header):
        if isinstance(value, str) and value.strip().upper() == "STK":
            stk_col = idx
        if isinstance(value, str) and value.strip().upper() == "ATM":
            atm_col = idx

    quotes = []  # (tenor_period, atm_strike, atm_norm_vol)
    for row in rows[5:]:
        tenor_raw = row[0]
        if tenor_raw is None:
            continue
        label = str(tenor_raw).strip().upper()
        if not (label.endswith("M") or label.endswith("Y")):
            continue
        stk = row[stk_col]
        atm = row[atm_col]
        if stk is None or atm is None:
            continue
        if label.endswith("M"):
            period = ql.Period(int(label[:-1]), ql.Months)
            years = int(label[:-1]) / 12.0
        else:
            period = ql.Period(int(label[:-1]), ql.Years)
            years = int(label[:-1])
        if years > 10.0:
            continue
        quotes.append((period, float(stk) / 100.0, float(atm) / 10_000.0))

    # Sort by length
    quotes.sort(key=lambda q: OPTION_DC.yearFraction(
        ql.Date(1, 1, 2025), ql.Date(1, 1, 2025) + q[0]))

    # Strip caplet vols using sequential variance allocation
    settlement = CALENDAR.advance(eval_date, 2, ql.Days)
    sigma_by_key: dict[int, float] = {}

    for cap_period, _, cap_vol in quotes:
        cap_end = CALENDAR.advance(settlement, cap_period, ql.ModifiedFollowing)
        sched = ql.MakeSchedule(
            effectiveDate=settlement,
            terminationDate=cap_end,
            tenor=ql.Period(ql.Semiannual),
            calendar=CALENDAR,
            convention=ql.ModifiedFollowing,
            terminalDateConvention=ql.ModifiedFollowing,
            rule=ql.DateGeneration.Forward,
            endOfMonth=False,
        )
        dts = list(sched)
        cap_weights = []
        for j in range(len(dts) - 1):
            alpha = FLOAT_DC.yearFraction(dts[j], dts[j + 1])
            df = curve.discount(dts[j + 1])
            cap_weights.append((dts[j + 1], alpha * df))

        annuity = sum(w for _, w in cap_weights)
        target = (cap_vol ** 2) * annuity
        known = sum(w * sigma_by_key[d.serialNumber()] ** 2
                     for d, w in cap_weights if d.serialNumber() in sigma_by_key)
        unknown = [(d, w) for d, w in cap_weights if d.serialNumber() not in sigma_by_key]
        if not unknown:
            continue
        rem = max(target - known, 0.0)
        rem_w = sum(w for _, w in unknown)
        sigma_new = math.sqrt(rem / rem_w) if rem_w > 0 else 0.0
        for d, _ in unknown:
            sigma_by_key[d.serialNumber()] = sigma_new

    # Build sorted list of (date_serial, sigma)
    stripped = sorted(sigma_by_key.items())
    return stripped


def _sigma_for_expiry(expiry_serial: int, stripped: list[tuple[int, float]]) -> float:
    if not stripped:
        return 0.0
    for key, sigma in stripped:
        if expiry_serial <= key:
            return sigma
    return stripped[-1][1]


# ---------------------------------------------------------------------------
# Full bond re-pricing under a given curve
# ---------------------------------------------------------------------------
def reprice_bond(curve, eval_date: ql.Date, stripped_vols) -> dict:
    """
    Re-price the full bond (collared coupon PV + options + CVA) under a
    given curve.  Returns key pricing components.
    """
    schedule = _build_bond_schedule()
    dates = list(schedule)

    eff_notional = NOTIONAL * PARTICIPATION
    floor_strike = FLOOR_RATE / PARTICIPATION
    cap_strike = CAP_RATE / PARTICIPATION

    frn_pv = 0.0
    collared_pv = 0.0
    total_floor = 0.0
    total_cap = 0.0
    accrued_interest = 0.0
    future_cf: list[tuple[ql.Date, float]] = []

    for i in range(len(dates) - 1):
        p_start = dates[i]
        p_end = dates[i + 1]
        if p_end <= eval_date:
            continue

        reset = _get_reset_date(p_start)
        alpha = COUPON_DC.yearFraction(p_start, p_end)
        df = curve.discount(p_end)

        hist = _get_historical_euribor(reset)
        if reset <= eval_date and hist is not None:
            euribor = hist
        else:
            euribor = curve.forwardRate(p_start, p_end, FLOAT_DC, ql.Simple).rate()

        # Uncollared FRN coupon
        frn_rate = PARTICIPATION * euribor
        frn_amt = NOTIONAL * frn_rate * alpha
        frn_pv += frn_amt * df

        # Collared coupon
        collared_rate = max(FLOOR_RATE, min(frn_rate, CAP_RATE))
        collared_amt = NOTIONAL * collared_rate * alpha
        collared_pv += collared_amt * df
        future_cf.append((p_end, collared_amt))

        # Accrued interest
        if p_start <= eval_date < p_end:
            accrued_frac = COUPON_DC.yearFraction(p_start, eval_date)
            accrued_interest = NOTIONAL * collared_rate * accrued_frac

        # Option components (only for future resets)
        if reset > eval_date:
            fwd = curve.forwardRate(p_start, p_end, FLOAT_DC, ql.Simple).rate()
            texp = max(OPTION_DC.yearFraction(eval_date, reset), 0.0)
            sigma = _sigma_for_expiry(reset.serialNumber(), stripped_vols)
            std_dev = sigma * math.sqrt(texp)

            caplet_pv = eff_notional * alpha * df * ql.bachelierBlackFormula(
                ql.Option.Call, cap_strike, fwd, std_dev, 1.0)
            floorlet_pv = eff_notional * alpha * df * ql.bachelierBlackFormula(
                ql.Option.Put, floor_strike, fwd, std_dev, 1.0)
            total_cap += caplet_pv
            total_floor += floorlet_pv

    # Par redemption
    par_pv = NOTIONAL * curve.discount(MATURITY_DATE)
    future_cf.append((MATURITY_DATE, NOTIONAL))

    # CVA
    future_dates_sorted = sorted({d for d, _ in future_cf}, key=lambda d: d.serialNumber())
    total_cva = 0.0
    for idx, ti in enumerate(future_dates_sorted):
        remaining = sum(amt * curve.discount(d) for d, amt in future_cf if d >= ti)
        t_i = OPTION_DC.yearFraction(eval_date, ti)
        t_prev = OPTION_DC.yearFraction(eval_date, future_dates_sorted[idx - 1]) if idx > 0 else 0.0
        marginal_pd = _survival_prob(t_prev) - _survival_prob(t_i)
        total_cva += LGD * remaining * marginal_pd

    risk_free_dirty = collared_pv + par_pv
    risky_dirty = risk_free_dirty - total_cva
    risky_clean = risky_dirty - accrued_interest

    return {
        "leveraged_frn_pv": frn_pv,
        "long_floor_pv": total_floor,
        "short_cap_pv": total_cap,
        "net_option_pv": total_floor - total_cap,
        "par_redemption_pv": par_pv,
        "collared_coupon_pv": collared_pv,
        "risk_free_dirty": risk_free_dirty,
        "cva": total_cva,
        "risky_dirty": risky_dirty,
        "accrued_interest": accrued_interest,
        "risky_clean": risky_clean,
        "risky_clean_pct": risky_clean / NOTIONAL * 100,
    }


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------
def make_sensitivity_chart(scenarios: list[dict], base_clean: float, out_path: Path):
    """Create a grouped bar chart of price changes by PC and shock size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    NAVY  = "#1a3a5c"
    GREEN = "#1a7a3a"
    RED   = "#cc2200"
    AMBER = "#c07800"
    TEAL  = "#0088aa"
    GREY  = "#888888"

    # Organise data: group by PC, sorted by n_sigma
    pc_labels = ["PC1 (Level)", "PC2 (Slope)", "PC3 (Curvature)"]
    sigma_vals = [-2, -1, 1, 2]
    sigma_labels = ["-2σ", "-1σ", "+1σ", "+2σ"]
    colors     = [RED, AMBER, GREEN, TEAL]

    # Build matrix: rows=PCs, cols=sigma_vals → price change
    changes = np.zeros((3, 4))
    for sc in scenarios:
        pc_idx = sc["pc_index"] - 1
        sig = sc["n_sigma"]
        col = sigma_vals.index(sig) if sig in sigma_vals else -1
        if 0 <= pc_idx < 3 and col >= 0:
            changes[pc_idx, col] = sc["risky_clean"] - base_clean

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    fig.patch.set_facecolor("#f9f9f9")

    for ax_idx, ax in enumerate(axes):
        ax.set_facecolor("#f9f9f9")
        vals = changes[ax_idx]
        bars = ax.bar(sigma_labels, vals, color=colors, edgecolor="white",
                       linewidth=0.8, width=0.6)
        for bar, val in zip(bars, vals):
            y_off = 0.3 if val >= 0 else -0.3
            va = "bottom" if val >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2, val + y_off,
                    f"{val:+.2f}", ha="center", va=va, fontsize=7,
                    fontweight="bold", color=NAVY)

        ax.axhline(0, color=GREY, linewidth=0.8, linestyle="--")
        ax.set_title(pc_labels[ax_idx], fontsize=9, fontweight="bold", color=NAVY, pad=3)
        ax.set_ylabel("Δ Clean Price (EUR)" if ax_idx == 0 else "", fontsize=8)
        ax.set_xlabel("Shock Size", fontsize=8)
        ax.tick_params(labelsize=7)

        # Symmetric y-limits
        ymax = max(abs(v) for v in vals) * 1.4 if max(abs(v) for v in vals) > 0 else 1
        ax.set_ylim(-ymax, ymax)

        for spine in ax.spines.values():
            spine.set_color("#cccccc")

    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=400, bbox_inches="tight", facecolor="#f9f9f9")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Additional chart: waterfall showing all components under each scenario
# ---------------------------------------------------------------------------
def make_component_chart(scenarios: list[dict], base_result: dict, out_path: Path):
    """Create a stacked-bar chart showing decomposition changes per scenario."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    NAVY  = "#1a3a5c"
    GREEN = "#1a7a3a"
    RED   = "#cc2200"
    TEAL  = "#0088aa"
    AMBER = "#c07800"

    # Only use ±1σ scenarios for the component decomposition chart
    one_sig = [s for s in scenarios if abs(s["n_sigma"]) == 1]
    one_sig.sort(key=lambda s: (s["pc_index"], s["n_sigma"]))

    labels = [f"PC{s['pc_index']} {'+' if s['n_sigma'] > 0 else ''}{s['n_sigma']}σ"
              for s in one_sig]

    # Component deltas relative to base
    d_coupon = [s["collared_coupon_pv"] - base_result["collared_coupon_pv"] for s in one_sig]
    d_par    = [s["par_redemption_pv"] - base_result["par_redemption_pv"]   for s in one_sig]
    d_option = [s["net_option_pv"] - base_result["net_option_pv"]           for s in one_sig]
    d_cva    = [-(s["cva"] - base_result["cva"])                            for s in one_sig]

    x = np.arange(len(labels))
    width = 0.18

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    fig.patch.set_facecolor("#f9f9f9")
    ax.set_facecolor("#f9f9f9")

    ax.bar(x - 1.5 * width, d_coupon, width, label="Δ Collared Coupon PV", color=TEAL, edgecolor="white")
    ax.bar(x - 0.5 * width, d_par,    width, label="Δ Par Redemption PV",  color=GREEN, edgecolor="white")
    ax.bar(x + 0.5 * width, d_option, width, label="Δ Net Option PV",      color=AMBER, edgecolor="white")
    ax.bar(x + 1.5 * width, d_cva,    width, label="Δ CVA (sign-flipped)", color=RED, edgecolor="white")

    ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Δ EUR (relative to base)", fontsize=8)
    ax.legend(fontsize=7, loc="best", framealpha=0.9)
    ax.tick_params(labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#cccccc")

    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=400, bbox_inches="tight", facecolor="#f9f9f9")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # 1. Load base market data and build base curve
    base_data = load_market_data()
    base_curve, eval_date, _ = build_curve(base_data)

    # 2. Load stripped vols (from base curve, kept constant across scenarios)
    stripped_vols = _load_and_strip_vols(base_curve, eval_date)

    # 3. Price base case
    base_result = reprice_bond(base_curve, eval_date, stripped_vols)
    base_clean = base_result["risky_clean"]

    # 4. Load PCA results
    pca_tenors, eigenvals, loadings = load_pca_results()

    # 5. Define scenarios
    pc_names = ["Level (PC1)", "Slope (PC2)", "Curvature (PC3)"]
    sigma_multipliers = [-2, -1, 1, 2]
    scenarios: list[dict] = []

    print("\n" + "=" * 90)
    print("  QUESTION 12 -- Sensitivity to PCA-Based Term Structure Shifts")
    print("=" * 90)
    print(f"  Eval date          : {eval_date.to_date().isoformat()}")
    print(f"  Base risky clean   : EUR {base_clean:.2f}  ({base_clean / NOTIONAL * 100:.4f}% par)")
    print(f"  Horizon            : {HORIZON_DAYS} business days (1 month)")
    print(f"  Scale factor       : sqrt({HORIZON_DAYS}) = {HORIZON_SCALE:.4f}")
    print()

    # Print shock magnitudes
    print(f"  {'PC':<18s} {'σ daily (bps)':<15s} {'σ 1-month (bps)':<18s}")
    print(f"  {'-'*18} {'-'*15} {'-'*18}")
    for k in range(3):
        sigma_d = math.sqrt(eigenvals[k])
        sigma_m = sigma_d * HORIZON_SCALE
        print(f"  {pc_names[k]:<18s} {sigma_d:>10.2f}     {sigma_m:>10.2f}")
    print()

    # 6. Run scenarios
    print(f"  {'Scenario':<25s} {'Clean (EUR)':>12s} {'Δ (EUR)':>10s} "
          f"{'Δ (% par)':>10s} {'Clean %':>10s}")
    print(f"  {'-'*25} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'Base':25s} {base_clean:12.2f} {'---':>10s} {'---':>10s} "
          f"{base_clean / NOTIONAL * 100:10.4f}")

    for pc_k in range(3):
        for n_sig in sigma_multipliers:
            shocked_data = apply_pca_shock(
                base_data, pca_tenors, loadings[pc_k], eigenvals[pc_k], n_sig
            )
            try:
                shocked_curve, _, _ = build_curve(shocked_data)
                result = reprice_bond(shocked_curve, eval_date, stripped_vols)
            except RuntimeError as e:
                print(f"  {pc_names[pc_k]} {n_sig:+d}σ  — bootstrap failed: {e}")
                continue

            delta_eur = result["risky_clean"] - base_clean
            delta_pct = delta_eur / NOTIONAL * 100

            label = f"{pc_names[pc_k]} {n_sig:+d}σ"
            print(f"  {label:<25s} {result['risky_clean']:12.2f} "
                  f"{delta_eur:+10.2f} {delta_pct:+10.4f} "
                  f"{result['risky_clean'] / NOTIONAL * 100:10.4f}")

            scenario_row = {
                "scenario": label,
                "pc_index": pc_k + 1,
                "pc_name": pc_names[pc_k],
                "n_sigma": n_sig,
                "shock_1y_bps": n_sig * math.sqrt(eigenvals[pc_k]) * HORIZON_SCALE * loadings[pc_k][0] if pca_tenors[0] == "1Y" else 0.0,
                "risky_clean": result["risky_clean"],
                "risky_clean_pct": result["risky_clean"] / NOTIONAL * 100,
                "delta_eur": delta_eur,
                "delta_pct_par": delta_pct,
                "leveraged_frn_pv": result["leveraged_frn_pv"],
                "collared_coupon_pv": result["collared_coupon_pv"],
                "long_floor_pv": result["long_floor_pv"],
                "short_cap_pv": result["short_cap_pv"],
                "net_option_pv": result["net_option_pv"],
                "par_redemption_pv": result["par_redemption_pv"],
                "risk_free_dirty": result["risk_free_dirty"],
                "cva": result["cva"],
                "risky_dirty": result["risky_dirty"],
                "accrued_interest": result["accrued_interest"],
            }
            scenarios.append(scenario_row)

    # 7. Save CSV
    out_csv = OUTPUT_DIR / "q12_sensitivity_table.csv"
    fieldnames = [
        "scenario", "pc_index", "pc_name", "n_sigma", "shock_1y_bps",
        "risky_clean", "risky_clean_pct", "delta_eur", "delta_pct_par",
        "leveraged_frn_pv", "collared_coupon_pv", "long_floor_pv",
        "short_cap_pv", "net_option_pv", "par_redemption_pv",
        "risk_free_dirty", "cva", "risky_dirty", "accrued_interest",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(scenarios)

    # 8. Charts
    out_chart = OUTPUT_DIR / "q12_sensitivity_chart.png"
    make_sensitivity_chart(scenarios, base_clean, out_chart)

    out_comp = OUTPUT_DIR / "q12_component_chart.png"
    make_component_chart(scenarios, base_result, out_comp)

    print()
    print(f"  Saved: {out_csv}")
    print(f"  Saved: {out_chart}")
    print(f"  Saved: {out_comp}")
    print("=" * 90)


if __name__ == "__main__":
    main()
