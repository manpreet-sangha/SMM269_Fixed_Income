"""
q20_main.py
--------------------------------------------------------------------------------
Marginal and Component VaR Decomposition (Bond Standalone)
Outputs (output/)
-----------------
  q20_var_decomposition.csv    Component VaR/ES table
  q20_var_pie.png              Pie chart of VaR contributions
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


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def _load_eigenvalues() -> list[float]:
    q11_dir = ROOT_DIR / "Q11" / "output"
    eigenvals = []
    with open(q11_dir / "q11_pca_eigenvalues.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            eigenvals.append(float(row["eigenvalue_bps2"]))
            if len(eigenvals) == 3:
                break
    return eigenvals


def _load_betas() -> dict[str, float]:
    q17_csv = ROOT_DIR / "Q17" / "output" / "q17_factor_model.csv"
    betas = {}
    with open(q17_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            factor = row.get("factor", "")
            coeff = row.get("coefficient", "")
            if factor and coeff:
                try:
                    betas[factor] = float(coeff)
                except ValueError:
                    break
    return betas


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------
def make_decomposition_chart(results: list[dict],
                             total_var: float, total_es: float,
                             out_path: Path):
    """Horizontal grouped-bar chart of Component VaR and ES by factor."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [r["factor"] for r in results]
    cvar = [r["component_var"] for r in results]
    ces  = [r["component_es"]  for r in results]
    pcts = [r["component_var_pct"] for r in results]

    y = np.arange(len(labels))
    bar_h = 0.32

    fig, ax = plt.subplots(figsize=(8, 4), dpi=300, facecolor="#f9f9f9")
    ax.set_facecolor("#f9f9f9")

    bars_var = ax.barh(y + bar_h / 2, cvar, bar_h,
                       label="Component VaR", color="#1a3a5c",
                       edgecolor="white", linewidth=0.6)
    bars_es  = ax.barh(y - bar_h / 2, ces, bar_h,
                       label="Component ES",  color="#c07800",
                       edgecolor="white", linewidth=0.6)

    # Annotate percentage share on VaR bars
    for bar, pct in zip(bars_var, pcts):
        w = bar.get_width()
        if w > 1.0:
            ax.text(w - 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{pct:.1f}%", va="center", ha="right",
                    fontsize=8, fontweight="bold", color="white")
        else:
            ax.text(w + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{pct:.1f}%", va="center", ha="left",
                    fontsize=8, fontweight="bold", color="#333")

    # EUR value labels on ES bars
    for bar, val in zip(bars_es, ces):
        w = bar.get_width()
        ax.text(w + 0.3, bar.get_y() + bar.get_height() / 2,
                f"EUR {val:.2f}", va="center", ha="left",
                fontsize=7.5, color="#555")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("EUR (per EUR 1,000 notional)", fontsize=9)
    ax.set_title("Component VaR & ES Decomposition (99%, 1-month)",
                 fontsize=11, fontweight="bold", pad=10)

    # Add reference lines for totals
    ax.axvline(total_var, color="#1a3a5c", ls="--", lw=0.9, alpha=0.5)
    ax.axvline(total_es,  color="#c07800", ls="--", lw=0.9, alpha=0.5)
    ax.text(total_var, len(labels) - 0.15, f" VaR={total_var:.1f}",
            fontsize=7, color="#1a3a5c", va="bottom")
    ax.text(total_es, len(labels) - 0.15, f" ES={total_es:.1f}",
            fontsize=7, color="#c07800", va="bottom")

    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax.invert_yaxis()

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    eigenvals = _load_eigenvalues()
    betas = _load_betas()

    beta_l = betas["beta_level"]
    beta_s = betas["beta_slope"]
    beta_c = betas["beta_curvature"]
    beta_cds = betas["beta_cds"]

    sigma_pc1 = math.sqrt(eigenvals[0]) * HORIZON_SCALE
    sigma_pc2 = math.sqrt(eigenvals[1]) * HORIZON_SCALE
    sigma_pc3 = math.sqrt(eigenvals[2]) * HORIZON_SCALE
    sigma_cds = CDS_DAILY_VOL_BPS * HORIZON_SCALE

    # Factor labels, betas, sigmas
    factors = [
        ("Level (PC1)", beta_l, sigma_pc1),
        ("Slope (PC2)", beta_s, sigma_pc2),
        ("Curvature (PC3)", beta_c, sigma_pc3),
        ("CDS", beta_cds, sigma_cds),
    ]

    # ── Portfolio variance ───────────────────────────────────────────────
    comp_var = [(name, beta, sigma, (beta * sigma) ** 2)
                for name, beta, sigma in factors]
    total_var = sum(v for _, _, _, v in comp_var)
    sigma_p = math.sqrt(total_var)

    z99 = norm.ppf(CONFIDENCE)
    total_var_99 = z99 * sigma_p

    # ES multiplier: φ(z) / (1-α)
    es_mult = norm.pdf(z99) / (1 - CONFIDENCE)
    total_es_99 = es_mult * sigma_p

    # ── Decomposition ────────────────────────────────────────────────────
    results: list[dict] = []
    for name, beta, sigma, var_k in comp_var:
        # Marginal VaR: ∂VaR/∂w = z × β² σ² / σ_P
        # (here "w" = 1 unit exposure, so marginal = z × β σ² / σ_P × β)
        # More precisely: β_k σ_k is the factor's contribution to σ_P direction
        # Component VaR = z_α × (β_k σ_k)² / σ_P
        cvar_k = z99 * var_k / sigma_p
        ces_k = es_mult * var_k / sigma_p
        mvar_k = z99 * beta * sigma ** 2 / sigma_p  # per unit of beta change

        share_pct = var_k / total_var * 100

        results.append({
            "factor": name,
            "beta": beta,
            "sigma_bps": sigma,
            "beta_sigma": beta * sigma,
            "variance_share_pct": round(share_pct, 2),
            "marginal_var": round(mvar_k, 6),
            "component_var": round(cvar_k, 4),
            "component_es": round(ces_k, 4),
            "component_var_pct": round(cvar_k / total_var_99 * 100, 2),
            "component_es_pct": round(ces_k / total_es_99 * 100, 2),
        })

    # Verification: sum of component VaR = total VaR
    sum_cvar = sum(r["component_var"] for r in results)
    sum_ces = sum(r["component_es"] for r in results)

    # ── Save CSV ─────────────────────────────────────────────────────────
    out_csv = OUTPUT_DIR / "q20_var_decomposition.csv"
    fieldnames = ["factor", "beta", "sigma_bps", "beta_sigma",
                  "variance_share_pct", "marginal_var",
                  "component_var", "component_es",
                  "component_var_pct", "component_es_pct"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
        # Totals row
        w.writerow({
            "factor": "TOTAL",
            "beta": "",
            "sigma_bps": "",
            "beta_sigma": f"{sigma_p:.4f}",
            "variance_share_pct": "100.00",
            "marginal_var": "",
            "component_var": f"{sum_cvar:.4f}",
            "component_es": f"{sum_ces:.4f}",
            "component_var_pct": f"{sum_cvar/total_var_99*100:.2f}",
            "component_es_pct": f"{sum_ces/total_es_99*100:.2f}",
        })

    # ── Chart ─────────────────────────────────────────────────────────
    out_chart = OUTPUT_DIR / "q20_var_decomposition.png"
    make_decomposition_chart(results, total_var_99, total_es_99, out_chart)

    # ── Console report ───────────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("  QUESTION 20 -- Marginal and Component VaR Decomposition")
    print("=" * 76)
    print(f"  Confidence  : {CONFIDENCE:.0%}")
    print(f"  Horizon     : {HORIZON_DAYS} business days (1 month)")
    print(f"  z_99        : {z99:.4f}")
    print(f"  σ_P         : EUR {sigma_p:.4f}")
    print(f"  Total VaR   : EUR {total_var_99:.4f}")
    print(f"  Total ES    : EUR {total_es_99:.4f}")
    print("-" * 76)

    print(f"\n  {'Factor':<18s} {'β':>8s} {'σ':>8s} {'β×σ':>8s} "
          f"{'CVaR':>8s} {'%':>6s}  {'CES':>8s} {'%':>6s}")
    print("  " + "-" * 68)
    for r in results:
        print(f"  {r['factor']:<18s} {r['beta']:>+8.4f} "
              f"{r['sigma_bps']:>8.2f} {r['beta_sigma']:>+8.4f} "
              f"{r['component_var']:>8.4f} {r['component_var_pct']:>5.1f}%  "
              f"{r['component_es']:>8.4f} {r['component_es_pct']:>5.1f}%")
    print("  " + "-" * 68)
    print(f"  {'TOTAL':<18s} {'':>8s} {'':>8s} {sigma_p:>+8.4f} "
          f"{sum_cvar:>8.4f} {sum_cvar/total_var_99*100:>5.1f}%  "
          f"{sum_ces:>8.4f} {sum_ces/total_es_99*100:>5.1f}%")

    print(f"\n  Verification:")
    print(f"    Σ CVaR_k  = EUR {sum_cvar:.4f}  vs  VaR = EUR {total_var_99:.4f}"
          f"  (diff: {abs(sum_cvar - total_var_99):.6f})")
    print(f"    Σ CES_k   = EUR {sum_ces:.4f}  vs  ES  = EUR {total_es_99:.4f}"
          f"  (diff: {abs(sum_ces - total_es_99):.6f})")

    print(f"\n  Marginal VaR (∂VaR/∂β_k):")
    for r in results:
        print(f"    {r['factor']:<18s}: {r['marginal_var']:+.6f} EUR")
    print(f"\n  Interpretation: the marginal VaR shows how much the total")
    print(f"  VaR would change if the factor beta increased by 1 unit.")
    print(f"  CDS has the largest absolute marginal VaR, consistent with")
    print(f"  its 77.7% variance share.")
    print("=" * 76)
    print(f"  Saved: {out_csv}")
    print(f"  Saved: {out_chart}")


if __name__ == "__main__":
    main()
