"""
q21_main.py
--------------------------------------------------------------------------------
Hedged-Portfolio VaR Decomposition

Outputs (output/)
-----------------
  q21_hedged_var.csv     Component VaR/ES: standalone vs hedged
  q21_comparison.png     Side-by-side bar chart
  Console report
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

from scipy.stats import norm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
NOTIONAL = 1000.0
CONFIDENCE = 0.99
HORIZON_DAYS = 21
HORIZON_SCALE = math.sqrt(HORIZON_DAYS)
CDS_DAILY_VOL_BPS = 3.0
BASE_CDS_BPS = 75.0
RECOVERY_RATE = 0.40
IRS_NOTIONAL_BASE = 1_000_000.0   # DV01s in Q13 are per EUR 1M


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------
def _load_eigenvalues() -> list[float]:
    path = ROOT_DIR / "Q11" / "output" / "q11_pca_eigenvalues.csv"
    vals = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vals.append(float(row["eigenvalue_bps2"]))
            if len(vals) == 3:
                break
    return vals


def _load_betas() -> dict[str, float]:
    path = ROOT_DIR / "Q17" / "output" / "q17_factor_model.csv"
    betas = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            factor = row.get("factor", "")
            coeff = row.get("coefficient", "")
            if factor and coeff:
                try:
                    betas[factor] = float(coeff)
                except ValueError:
                    break
    return betas


def _load_factor_dv01s() -> list[dict]:
    """Load Q13 factor DV01s (bond and IRS per EUR 1M)."""
    path = ROOT_DIR / "Q13" / "output" / "q13_factor_dv01s.csv"
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "factor": row["factor"],
                "bond_dv01": float(row["bond_dv01_eur"]),
                "irs_2y_dv01": float(row["irs_2y_dv01_eur"]),
                "irs_10y_dv01": float(row["irs_10y_dv01_eur"]),
            })
    return rows


def _load_hedge_ratios() -> dict[str, float]:
    """Load Q13 IRS hedge notionals."""
    path = ROOT_DIR / "Q13" / "output" / "q13_hedge_ratios.csv"
    notionals = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            inst = row["instrument"]
            n = float(row["notional_eur"])
            if "2Y" in inst:
                notionals["2Y"] = n
            elif "10Y" in inst:
                notionals["10Y"] = n
    return notionals


def _compute_rpv01_base() -> float:
    """Recompute RPV01 at base CDS spread (75 bps) using Q16 logic."""
    from Q3.q3_market_data import load_market_data
    from Q3.q3_main import build_curve
    from Q16.q16_main import compute_rpv01

    data = load_market_data()
    curve, eval_date, _ = build_curve(data)
    return compute_rpv01(curve, eval_date, BASE_CDS_BPS)


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------
def make_comparison_chart(standalone: list[dict], hedged: list[dict],
                          total_std: float, total_hdg: float,
                          out_path: Path):
    """Side-by-side horizontal bar chart: standalone vs hedged CVaR."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [r["factor"] for r in standalone]
    cvar_std = [r["component_var"] for r in standalone]
    cvar_hdg = [r["component_var"] for r in hedged]

    y = np.arange(len(labels))
    bar_h = 0.32

    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=300, facecolor="#f9f9f9")
    ax.set_facecolor("#f9f9f9")

    bars_std = ax.barh(y + bar_h / 2, cvar_std, bar_h,
                       label=f"Standalone (VaR = EUR {total_std:.2f})",
                       color="#1a3a5c", edgecolor="white", linewidth=0.6)
    bars_hdg = ax.barh(y - bar_h / 2, cvar_hdg, bar_h,
                       label=f"Hedged (VaR = EUR {total_hdg:.2f})",
                       color="#2a7a4a", edgecolor="white", linewidth=0.6)

    # Annotate values on standalone bars
    for bar, val in zip(bars_std, cvar_std):
        w = bar.get_width()
        if w > 0.5:
            ax.text(w + 0.2, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", ha="left",
                    fontsize=7.5, color="#1a3a5c", fontweight="bold")

    # Annotate values on hedged bars
    for bar, val in zip(bars_hdg, cvar_hdg):
        w = bar.get_width()
        ax.text(max(w, 0) + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left",
                fontsize=7, color="#2a7a4a")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Component VaR (EUR per EUR 1,000 notional)", fontsize=9)
    ax.set_title("Standalone vs Hedged Component VaR (99%, 1-month)",
                 fontsize=11, fontweight="bold", pad=10)

    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax.invert_yaxis()

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved: {out_path}")


# ---------------------------------------------------------------------------
# Decomposition helper
# ---------------------------------------------------------------------------
def _decompose(factors: list[tuple[str, float, float]],
               ) -> tuple[list[dict], float, float]:
    """
    Given [(name, beta, sigma), ...] with independent factors,
    compute CVaR_k and CES_k.
    Returns (results_list, total_var, total_es).
    """
    z99 = norm.ppf(CONFIDENCE)
    es_mult = norm.pdf(z99) / (1 - CONFIDENCE)

    var_components = [(nm, b, s, (b * s) ** 2) for nm, b, s in factors]
    total_variance = sum(v for _, _, _, v in var_components)
    sigma_p = math.sqrt(total_variance) if total_variance > 0 else 0.0

    total_var = z99 * sigma_p
    total_es = es_mult * sigma_p

    results = []
    for nm, beta, sigma, var_k in var_components:
        if sigma_p > 0:
            cvar_k = z99 * var_k / sigma_p
            ces_k = es_mult * var_k / sigma_p
            mvar_k = z99 * beta * sigma ** 2 / sigma_p
        else:
            cvar_k = ces_k = mvar_k = 0.0
        share = var_k / total_variance * 100 if total_variance > 0 else 0.0
        results.append({
            "factor": nm,
            "beta": beta,
            "sigma_bps": sigma,
            "beta_sigma": beta * sigma,
            "variance_share_pct": round(share, 2),
            "marginal_var": round(mvar_k, 6),
            "component_var": round(cvar_k, 4),
            "component_es": round(ces_k, 4),
        })
    return results, round(total_var, 4), round(total_es, 4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # Load data
    eigenvals = _load_eigenvalues()
    betas = _load_betas()
    dv01s = _load_factor_dv01s()
    hedge_n = _load_hedge_ratios()

    sigma_pc1 = math.sqrt(eigenvals[0]) * HORIZON_SCALE
    sigma_pc2 = math.sqrt(eigenvals[1]) * HORIZON_SCALE
    sigma_pc3 = math.sqrt(eigenvals[2]) * HORIZON_SCALE
    sigma_cds = CDS_DAILY_VOL_BPS * HORIZON_SCALE

    # ── Bond standalone betas (from Q17 regression) ──────────────────────
    beta_l = betas["beta_level"]
    beta_s = betas["beta_slope"]
    beta_c = betas["beta_curvature"]
    beta_cds = betas["beta_cds"]

    standalone_factors = [
        ("Level (PC1)",     beta_l,   sigma_pc1),
        ("Slope (PC2)",     beta_s,   sigma_pc2),
        ("Curvature (PC3)", beta_c,   sigma_pc3),
        ("CDS",             beta_cds, sigma_cds),
    ]
    std_results, std_var, std_es = _decompose(standalone_factors)

    # ── Hedged betas ─────────────────────────────────────────────────────
    # IRS hedge contribution per factor (EUR/bps):
    # Δ_irs_k = (N_2Y / 1M) × DV01_2Y_k + (N_10Y / 1M) × DV01_10Y_k
    n2y = hedge_n["2Y"]
    n10y = hedge_n["10Y"]

    # Map factor names from Q13 CSV to (index, beta_bond)
    factor_map = {
        "Level (PC1)":     {"bond_beta": beta_l,   "sigma": sigma_pc1},
        "Slope (PC2)":     {"bond_beta": beta_s,   "sigma": sigma_pc2},
        "Curvature (PC3)": {"bond_beta": beta_c,   "sigma": sigma_pc3},
    }

    hedged_betas_rates = {}
    for row in dv01s:
        fn = row["factor"]
        irs_contribution = (
            (n2y / IRS_NOTIONAL_BASE) * row["irs_2y_dv01"]
            + (n10y / IRS_NOTIONAL_BASE) * row["irs_10y_dv01"]
        )
        bond_beta = factor_map[fn]["bond_beta"]
        hedged_betas_rates[fn] = bond_beta + irs_contribution

    # CDS hedge: buy protection with notional N_cds
    # CDS sensitivity = RPV01 × N_cds / 10,000  (EUR per bps)
    # The protection buyer gains when spreads widen → positive beta
    rpv01_base = _compute_rpv01_base()

    # Hedge notional from Q16: bond CS-DV01 / (RPV01/10000)
    bond_cs_dv01 = abs(beta_cds)  # 0.752925 EUR/bps
    cds_cs_dv01_per_unit = rpv01_base / 10_000.0
    n_cds = bond_cs_dv01 / cds_cs_dv01_per_unit

    cds_hedge_beta = rpv01_base * n_cds / 10_000.0
    hedged_beta_cds = beta_cds + cds_hedge_beta  # should be ≈ 0

    hedged_factors = [
        ("Level (PC1)",     hedged_betas_rates["Level (PC1)"],     sigma_pc1),
        ("Slope (PC2)",     hedged_betas_rates["Slope (PC2)"],     sigma_pc2),
        ("Curvature (PC3)", hedged_betas_rates["Curvature (PC3)"], sigma_pc3),
        ("CDS",             hedged_beta_cds,                       sigma_cds),
    ]
    hdg_results, hdg_var, hdg_es = _decompose(hedged_factors)

    # ── VaR reduction ────────────────────────────────────────────────────
    var_reduction_pct = (1 - hdg_var / std_var) * 100 if std_var > 0 else 0
    es_reduction_pct  = (1 - hdg_es / std_es)  * 100 if std_es  > 0 else 0

    # ── Save CSV ─────────────────────────────────────────────────────────
    out_csv = OUTPUT_DIR / "q21_hedged_var.csv"
    fieldnames = [
        "factor",
        "beta_standalone", "beta_hedged",
        "sigma_bps",
        "cvar_standalone", "cvar_hedged",
        "ces_standalone", "ces_hedged",
        "var_share_std_pct", "var_share_hdg_pct",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for s, h in zip(std_results, hdg_results):
            hdg_share = (
                h["component_var"] / hdg_var * 100
                if hdg_var > 0 else 0.0
            )
            w.writerow({
                "factor": s["factor"],
                "beta_standalone": f"{s['beta']:.6f}",
                "beta_hedged": f"{h['beta']:.6f}",
                "sigma_bps": f"{s['sigma_bps']:.4f}",
                "cvar_standalone": f"{s['component_var']:.4f}",
                "cvar_hedged": f"{h['component_var']:.4f}",
                "ces_standalone": f"{s['component_es']:.4f}",
                "ces_hedged": f"{h['component_es']:.4f}",
                "var_share_std_pct": f"{s['variance_share_pct']:.2f}",
                "var_share_hdg_pct": f"{hdg_share:.2f}",
            })
        # Totals
        w.writerow({
            "factor": "TOTAL",
            "beta_standalone": "",
            "beta_hedged": "",
            "sigma_bps": "",
            "cvar_standalone": f"{std_var:.4f}",
            "cvar_hedged": f"{hdg_var:.4f}",
            "ces_standalone": f"{std_es:.4f}",
            "ces_hedged": f"{hdg_es:.4f}",
            "var_share_std_pct": "100.00",
            "var_share_hdg_pct": "100.00",
        })

    # ── Chart ────────────────────────────────────────────────────────────
    out_chart = OUTPUT_DIR / "q21_comparison.png"
    make_comparison_chart(std_results, hdg_results, std_var, hdg_var, out_chart)

    # ── Console report ───────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  QUESTION 21 -- Hedged-Portfolio VaR Decomposition")
    print("=" * 80)
    print(f"  Confidence   : {CONFIDENCE:.0%}")
    print(f"  Horizon      : {HORIZON_DAYS} business days (1 month)")
    print(f"  IRS notionals: 2Y = EUR {n2y:.2f}, 10Y = EUR {n10y:.2f} (Receiver)")
    print(f"  CDS notional : EUR {n_cds:,.2f} (Buy protection)")
    print(f"  RPV01 (base) : {rpv01_base:.6f} yrs")
    print("-" * 80)

    print(f"\n  {'Factor':<18s} {'β_std':>8s} {'β_hdg':>10s} {'σ':>8s} "
          f"{'CVaR_std':>9s} {'CVaR_hdg':>9s} {'Δ':>8s}")
    print("  " + "-" * 72)
    for s, h in zip(std_results, hdg_results):
        delta = h["component_var"] - s["component_var"]
        print(f"  {s['factor']:<18s} {s['beta']:>+8.4f} {h['beta']:>+10.6f} "
              f"{s['sigma_bps']:>8.2f} {s['component_var']:>9.4f} "
              f"{h['component_var']:>9.4f} {delta:>+8.4f}")
    print("  " + "-" * 72)
    print(f"  {'TOTAL':<18s} {'':>8s} {'':>10s} {'':>8s} "
          f"{std_var:>9.4f} {hdg_var:>9.4f} {hdg_var - std_var:>+8.4f}")

    print(f"\n  VaR reduction : {var_reduction_pct:.1f}%  "
          f"(EUR {std_var:.2f} -> EUR {hdg_var:.2f})")
    print(f"  ES  reduction : {es_reduction_pct:.1f}%  "
          f"(EUR {std_es:.2f} -> EUR {hdg_es:.2f})")

    # Hedged VaR composition
    print(f"\n  Hedged VaR composition:")
    for h in hdg_results:
        hdg_share = h["component_var"] / hdg_var * 100 if hdg_var > 0 else 0
        print(f"    {h['factor']:<18s}: EUR {h['component_var']:.4f}  "
              f"({hdg_share:.1f}%)")

    print(f"\n  Limitation: the linear factor model overstates hedge")
    print(f"  effectiveness.  Q14 showed that under PC1 +2s, the cap's")
    print(f"  negative convexity caused -452% hedge effectiveness.")
    print(f"  True residual VaR is larger than this linear estimate.")
    print("=" * 80)
    print(f"  Saved: {out_csv}")
    print(f"  Saved: {out_chart}")


if __name__ == "__main__":
    main()
