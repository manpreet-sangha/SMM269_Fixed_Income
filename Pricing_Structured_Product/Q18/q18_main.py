"""
q18_main.py
--------------------------------------------------------------------------------
Monte Carlo VaR and Expected Shortfall (99%, 1-month)

Methodology
-----------
1) Draw N = 100,000 joint samples from N(0, Σ) where Σ = diag(σ²).
2) ΔGP_i = β_l·Δl_i + β_s·Δs_i + β_c·Δc_i + β_cds·Δcds_i
3) VaR_99 = −percentile(ΔGP, 1)
4) ES_99  = −mean(ΔGP | ΔGP ≤ −VaR_99)
5) Generate histogram with VaR/ES markers.

Outputs (output/)
-----------------
  q18_mc_results.csv           Summary statistics
  q18_mc_histogram.png         P&L distribution with VaR/ES
  Console report
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

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
N_SIM = 100_000
CONFIDENCE = 0.99
SEED = 42

# PCA eigenvalues (bps² of daily changes) from Q11
# (loaded from CSV for consistency)
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

# Factor model betas from Q17
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
                    break  # hit the metrics section
    return betas

# CDS daily vol assumption: 3 bps/day for IG European bank
CDS_DAILY_VOL_BPS = 3.0

# Horizon
HORIZON_DAYS = 21  # 1 month ≈ 21 business days
HORIZON_SCALE = math.sqrt(HORIZON_DAYS)


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------
def make_histogram(pnl: np.ndarray, var_99: float, es_99: float,
                   out_path: Path):
    """P&L histogram with VaR and ES markers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    NAVY = "#1a3a5c"
    RED = "#cc2200"
    AMBER = "#c07800"

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300, facecolor="#f9f9f9")
    ax.set_facecolor("#f9f9f9")

    # Histogram
    ax.hist(pnl, bins=200, color=NAVY, alpha=0.7, edgecolor="none",
            density=True, label="P&L distribution")

    # VaR line
    ax.axvline(-var_99, color=RED, linewidth=2.0, linestyle="--",
               label=f"VaR$_{{99\\%}}$ = EUR {var_99:.2f}")

    # ES line
    ax.axvline(-es_99, color=AMBER, linewidth=2.0, linestyle="-.",
               label=f"ES$_{{99\\%}}$ = EUR {es_99:.2f}")

    ax.set_xlabel("ΔGP (EUR per 1,000 notional)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Monte Carlo P&L Distribution (1-month, 99%)",
                 fontsize=11, fontweight="bold", pad=8)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # ── Load parameters ──────────────────────────────────────────────────
    eigenvals = _load_eigenvalues()
    betas = _load_betas()

    beta_l = betas["beta_level"]
    beta_s = betas["beta_slope"]
    beta_c = betas["beta_curvature"]
    beta_cds = betas["beta_cds"]

    # Monthly factor standard deviations (bps)
    sigma_pc1 = math.sqrt(eigenvals[0]) * HORIZON_SCALE
    sigma_pc2 = math.sqrt(eigenvals[1]) * HORIZON_SCALE
    sigma_pc3 = math.sqrt(eigenvals[2]) * HORIZON_SCALE
    sigma_cds = CDS_DAILY_VOL_BPS * HORIZON_SCALE

    print("\n  Factor standard deviations (1-month, bps):")
    print(f"    σ_PC1 (Level)     = {sigma_pc1:.2f}")
    print(f"    σ_PC2 (Slope)     = {sigma_pc2:.2f}")
    print(f"    σ_PC3 (Curvature) = {sigma_pc3:.2f}")
    print(f"    σ_CDS             = {sigma_cds:.2f}")

    # ── Monte Carlo simulation ───────────────────────────────────────────
    rng = np.random.default_rng(SEED)

    # Independent draws (PCA factors are orthogonal; CDS assumed independent)
    z_pc1 = rng.normal(0, sigma_pc1, N_SIM)
    z_pc2 = rng.normal(0, sigma_pc2, N_SIM)
    z_pc3 = rng.normal(0, sigma_pc3, N_SIM)
    z_cds = rng.normal(0, sigma_cds, N_SIM)

    # P&L via factor model
    pnl = (beta_l * z_pc1
            + beta_s * z_pc2
            + beta_c * z_pc3
            + beta_cds * z_cds)

    # ── Risk measures ────────────────────────────────────────────────────
    alpha = 1.0 - CONFIDENCE  # 0.01

    var_99 = -np.percentile(pnl, alpha * 100)  # positive number = loss
    tail = pnl[pnl <= -var_99]
    es_99 = -np.mean(tail)

    # P&L statistics
    pnl_mean = np.mean(pnl)
    pnl_std = np.std(pnl, ddof=1)
    pnl_skew = float(np.mean(((pnl - pnl_mean) / pnl_std) ** 3))
    pnl_kurt = float(np.mean(((pnl - pnl_mean) / pnl_std) ** 4) - 3.0)
    pnl_min = np.min(pnl)
    pnl_max = np.max(pnl)

    # ── Component VaR (analytical, for reporting) ────────────────────────
    # Under normality: VaR_99 = z_99 × σ_portfolio
    # σ² = β_l² σ₁² + β_s² σ₂² + β_c² σ₃² + β_cds² σ_cds²
    var_l = (beta_l * sigma_pc1) ** 2
    var_s = (beta_s * sigma_pc2) ** 2
    var_c = (beta_c * sigma_pc3) ** 2
    var_cds = (beta_cds * sigma_cds) ** 2
    total_var = var_l + var_s + var_c + var_cds
    portfolio_sigma = math.sqrt(total_var)

    # Contribution to variance (%)
    pct_l = var_l / total_var * 100
    pct_s = var_s / total_var * 100
    pct_c = var_c / total_var * 100
    pct_cds = var_cds / total_var * 100

    # ── Save CSV ─────────────────────────────────────────────────────────
    out_csv = OUTPUT_DIR / "q18_mc_results.csv"
    rows = [
        {"metric": "n_simulations", "value": f"{N_SIM}"},
        {"metric": "confidence_level", "value": f"{CONFIDENCE:.2f}"},
        {"metric": "horizon_days", "value": f"{HORIZON_DAYS}"},
        {"metric": "seed", "value": f"{SEED}"},
        {"metric": "sigma_pc1_bps", "value": f"{sigma_pc1:.4f}"},
        {"metric": "sigma_pc2_bps", "value": f"{sigma_pc2:.4f}"},
        {"metric": "sigma_pc3_bps", "value": f"{sigma_pc3:.4f}"},
        {"metric": "sigma_cds_bps", "value": f"{sigma_cds:.4f}"},
        {"metric": "beta_level", "value": f"{beta_l:.6f}"},
        {"metric": "beta_slope", "value": f"{beta_s:.6f}"},
        {"metric": "beta_curvature", "value": f"{beta_c:.6f}"},
        {"metric": "beta_cds", "value": f"{beta_cds:.6f}"},
        {"metric": "portfolio_sigma_eur", "value": f"{portfolio_sigma:.4f}"},
        {"metric": "var_99_eur", "value": f"{var_99:.4f}"},
        {"metric": "es_99_eur", "value": f"{es_99:.4f}"},
        {"metric": "pnl_mean", "value": f"{pnl_mean:.6f}"},
        {"metric": "pnl_std", "value": f"{pnl_std:.4f}"},
        {"metric": "pnl_skew", "value": f"{pnl_skew:.4f}"},
        {"metric": "pnl_kurtosis", "value": f"{pnl_kurt:.4f}"},
        {"metric": "pnl_min", "value": f"{pnl_min:.4f}"},
        {"metric": "pnl_max", "value": f"{pnl_max:.4f}"},
        {"metric": "var_pct_level", "value": f"{pct_l:.2f}"},
        {"metric": "var_pct_slope", "value": f"{pct_s:.2f}"},
        {"metric": "var_pct_curvature", "value": f"{pct_c:.2f}"},
        {"metric": "var_pct_cds", "value": f"{pct_cds:.2f}"},
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "value"])
        w.writeheader()
        w.writerows(rows)

    # ── Chart ────────────────────────────────────────────────────────────
    out_chart = OUTPUT_DIR / "q18_mc_histogram.png"
    make_histogram(pnl, var_99, es_99, out_chart)

    # ── Console report ───────────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("  QUESTION 18 -- Monte Carlo VaR and Expected Shortfall")
    print("=" * 76)
    print(f"  Simulations     : {N_SIM:,}")
    print(f"  Confidence      : {CONFIDENCE:.0%}")
    print(f"  Horizon         : {HORIZON_DAYS} business days (1 month)")
    print(f"  Seed            : {SEED}")
    print("-" * 76)
    print(f"  Portfolio σ     : EUR {portfolio_sigma:.4f}")
    print(f"  VaR (99%)       : EUR {var_99:.4f}")
    print(f"  ES  (99%)       : EUR {es_99:.4f}")
    print(f"  VaR as % notl   : {var_99/NOTIONAL*100:.4f}%")
    print(f"  ES  as % notl   : {es_99/NOTIONAL*100:.4f}%")
    print("-" * 76)
    print(f"  P&L distribution:")
    print(f"    Mean          : EUR {pnl_mean:.4f}")
    print(f"    Std dev       : EUR {pnl_std:.4f}")
    print(f"    Skewness      : {pnl_skew:+.4f}")
    print(f"    Excess kurt.  : {pnl_kurt:+.4f}")
    print(f"    Min           : EUR {pnl_min:.4f}")
    print(f"    Max           : EUR {pnl_max:.4f}")
    print("-" * 76)
    print(f"  Variance decomposition:")
    print(f"    Level (PC1)   : {pct_l:6.2f}%   (β σ = {abs(beta_l*sigma_pc1):.4f})")
    print(f"    Slope (PC2)   : {pct_s:6.2f}%   (β σ = {abs(beta_s*sigma_pc2):.4f})")
    print(f"    Curvature(PC3): {pct_c:6.2f}%   (β σ = {abs(beta_c*sigma_pc3):.4f})")
    print(f"    CDS           : {pct_cds:6.2f}%   (β σ = {abs(beta_cds*sigma_cds):.4f})")
    print("-" * 76)

    # Analytical benchmark
    from scipy.stats import norm
    z99 = norm.ppf(CONFIDENCE)
    analytical_var = z99 * portfolio_sigma
    analytical_es = portfolio_sigma * norm.pdf(z99) / (1 - CONFIDENCE)
    print(f"\n  Analytical benchmark (normal):")
    print(f"    z_99          : {z99:.4f}")
    print(f"    VaR (analyt.) : EUR {analytical_var:.4f}")
    print(f"    ES  (analyt.) : EUR {analytical_es:.4f}")
    print(f"    MC/Analyt VaR : {var_99/analytical_var:.4f}")
    print(f"    MC/Analyt ES  : {es_99/analytical_es:.4f}")
    print("=" * 76)
    print(f"  Saved: {out_csv}")
    print(f"  Saved: {out_chart}")


if __name__ == "__main__":
    main()
