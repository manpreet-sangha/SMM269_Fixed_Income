"""
q19_main.py
--------------------------------------------------------------------------------
Exact (Analytical) VaR and ES vs. Monte Carlo
This script:
1) Derives the exact VaR and ES from the Q17 factor betas and factor σ's.
2) Compares with the MC results from Q18.
3) Analyses convergence: re-runs MC at {1k, 5k, 10k, 50k, 100k, 500k}
   simulations to show convergence to the exact solution.
4) Discusses conditions under which MC ≠ exact (non-linearity, fat tails).

Outputs (output/)
-----------------
  q19_exact_vs_mc.csv          Comparison table
  q19_convergence.csv          MC convergence across sample sizes
  q19_convergence_chart.png    Convergence chart
  Console report
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np
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
SEED = 42
HORIZON_DAYS = 21
HORIZON_SCALE = math.sqrt(HORIZON_DAYS)
CDS_DAILY_VOL_BPS = 3.0


# ---------------------------------------------------------------------------
# Load data from Q11 / Q17 / Q18
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


def _load_mc_results() -> dict[str, str]:
    q18_csv = ROOT_DIR / "Q18" / "output" / "q18_mc_results.csv"
    results = {}
    with open(q18_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results[row["metric"]] = row["value"]
    return results


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------
def make_convergence_chart(conv_data: list[dict], exact_var: float,
                           exact_es: float, out_path: Path):
    """MC convergence to exact VaR/ES as N increases."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    NAVY = "#1a3a5c"
    RED = "#cc2200"
    AMBER = "#c07800"

    ns = [d["n_sim"] for d in conv_data]
    mc_vars = [d["mc_var"] for d in conv_data]
    mc_ess = [d["mc_es"] for d in conv_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), dpi=300,
                                    facecolor="#f9f9f9")
    for ax in (ax1, ax2):
        ax.set_facecolor("#f9f9f9")

    # VaR convergence
    ax1.semilogx(ns, mc_vars, "o-", color=NAVY, linewidth=1.5,
                 markersize=5, label="MC VaR")
    ax1.axhline(exact_var, color=RED, linewidth=1.5, linestyle="--",
                label=f"Exact = {exact_var:.2f}")
    ax1.set_xlabel("Number of simulations", fontsize=9)
    ax1.set_ylabel("VaR₉₉% (EUR)", fontsize=9)
    ax1.set_title("VaR Convergence", fontsize=10, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ES convergence
    ax2.semilogx(ns, mc_ess, "o-", color=NAVY, linewidth=1.5,
                 markersize=5, label="MC ES")
    ax2.axhline(exact_es, color=AMBER, linewidth=1.5, linestyle="--",
                label=f"Exact = {exact_es:.2f}")
    ax2.set_xlabel("Number of simulations", fontsize=9)
    ax2.set_ylabel("ES₉₉% (EUR)", fontsize=9)
    ax2.set_title("ES Convergence", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

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
    mc_results = _load_mc_results()

    beta_l = betas["beta_level"]
    beta_s = betas["beta_slope"]
    beta_c = betas["beta_curvature"]
    beta_cds = betas["beta_cds"]

    # Monthly factor standard deviations (bps)
    sigma_pc1 = math.sqrt(eigenvals[0]) * HORIZON_SCALE
    sigma_pc2 = math.sqrt(eigenvals[1]) * HORIZON_SCALE
    sigma_pc3 = math.sqrt(eigenvals[2]) * HORIZON_SCALE
    sigma_cds = CDS_DAILY_VOL_BPS * HORIZON_SCALE

    # ── Exact (analytical) solution ──────────────────────────────────────
    # σ_P² = Σ (β_k × σ_k)²   (independent factors)
    comp_vars = {
        "Level (PC1)": (beta_l * sigma_pc1) ** 2,
        "Slope (PC2)": (beta_s * sigma_pc2) ** 2,
        "Curvature (PC3)": (beta_c * sigma_pc3) ** 2,
        "CDS": (beta_cds * sigma_cds) ** 2,
    }
    total_var = sum(comp_vars.values())
    sigma_p = math.sqrt(total_var)

    z99 = norm.ppf(CONFIDENCE)
    exact_var = z99 * sigma_p
    exact_es = sigma_p * norm.pdf(z99) / (1 - CONFIDENCE)

    # ── MC results from Q18 ─────────────────────────────────────────────
    mc_var = float(mc_results["var_99_eur"])
    mc_es = float(mc_results["es_99_eur"])
    mc_n = int(mc_results["n_simulations"])

    # ── Convergence study ────────────────────────────────────────────────
    sample_sizes = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000]
    conv_data: list[dict] = []

    beta_vec = np.array([beta_l, beta_s, beta_c, beta_cds])
    sigma_vec = np.array([sigma_pc1, sigma_pc2, sigma_pc3, sigma_cds])

    for n_sim in sample_sizes:
        rng = np.random.default_rng(SEED)
        z = rng.normal(0, 1, size=(n_sim, 4))  # standard normals
        z *= sigma_vec  # scale to factor sigmas
        pnl = z @ beta_vec

        var_n = -np.percentile(pnl, (1 - CONFIDENCE) * 100)
        tail = pnl[pnl <= -var_n]
        es_n = -np.mean(tail) if len(tail) > 0 else var_n

        conv_data.append({
            "n_sim": n_sim,
            "mc_var": round(var_n, 4),
            "mc_es": round(es_n, 4),
            "var_error_pct": round((var_n - exact_var) / exact_var * 100, 4),
            "es_error_pct": round((es_n - exact_es) / exact_es * 100, 4),
        })

    # ── Save comparison CSV ──────────────────────────────────────────────
    out_comparison = OUTPUT_DIR / "q19_exact_vs_mc.csv"
    with open(out_comparison, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["measure", "exact_analytical", "mc_100k", "difference",
                     "ratio"])
        w.writerow(["sigma_P (EUR)", f"{sigma_p:.4f}",
                     mc_results["pnl_std"], "", ""])
        w.writerow(["VaR_99 (EUR)", f"{exact_var:.4f}", f"{mc_var:.4f}",
                     f"{mc_var - exact_var:+.4f}",
                     f"{mc_var / exact_var:.6f}"])
        w.writerow(["ES_99 (EUR)", f"{exact_es:.4f}", f"{mc_es:.4f}",
                     f"{mc_es - exact_es:+.4f}",
                     f"{mc_es / exact_es:.6f}"])
        w.writerow([])
        w.writerow(["component", "beta", "sigma_bps", "beta_sigma",
                     "variance_share_pct"])
        for name, v in comp_vars.items():
            b = {"Level (PC1)": beta_l, "Slope (PC2)": beta_s,
                 "Curvature (PC3)": beta_c, "CDS": beta_cds}[name]
            s = {"Level (PC1)": sigma_pc1, "Slope (PC2)": sigma_pc2,
                 "Curvature (PC3)": sigma_pc3, "CDS": sigma_cds}[name]
            w.writerow([name, f"{b:.6f}", f"{s:.4f}",
                        f"{abs(b*s):.4f}",
                        f"{v/total_var*100:.2f}"])

    # ── Save convergence CSV ─────────────────────────────────────────────
    out_conv = OUTPUT_DIR / "q19_convergence.csv"
    with open(out_conv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["n_sim", "mc_var", "mc_es",
                                          "var_error_pct", "es_error_pct"])
        w.writeheader()
        w.writerows(conv_data)

    # ── Chart ────────────────────────────────────────────────────────────
    out_chart = OUTPUT_DIR / "q19_convergence_chart.png"
    make_convergence_chart(conv_data, exact_var, exact_es, out_chart)

    # ── Console report ───────────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("  QUESTION 19 -- Exact VaR/ES vs. Monte Carlo Comparison")
    print("=" * 76)
    print(f"  Confidence      : {CONFIDENCE:.0%}")
    print(f"  Horizon         : {HORIZON_DAYS} business days (1 month)")
    print(f"  z_{{99%}}         : {z99:.4f}")
    print("-" * 76)

    print(f"\n  Exact (analytical) derivation:")
    print(f"  {'Component':<18s} {'β':>9s} {'σ (bps)':>9s} {'|β×σ|':>9s} "
          f"{'Var share':>9s}")
    print("  " + "-" * 56)
    for name, v in comp_vars.items():
        b = {"Level (PC1)": beta_l, "Slope (PC2)": beta_s,
             "Curvature (PC3)": beta_c, "CDS": beta_cds}[name]
        s = {"Level (PC1)": sigma_pc1, "Slope (PC2)": sigma_pc2,
             "Curvature (PC3)": sigma_pc3, "CDS": sigma_cds}[name]
        print(f"  {name:<18s} {b:>+9.6f} {s:>9.2f} {abs(b*s):>9.4f} "
              f"{v/total_var*100:>8.2f}%")
    print("  " + "-" * 56)
    print(f"  {'Total σ_P':<18s} {'':>9s} {'':>9s} {sigma_p:>9.4f}")

    print(f"\n  Exact risk measures:")
    print(f"    VaR_99 = z_99 × σ_P = {z99:.4f} × {sigma_p:.4f} "
          f"= EUR {exact_var:.4f}")
    print(f"    ES_99  = σ_P × φ(z_99) / (1−α) = {sigma_p:.4f} × "
          f"{norm.pdf(z99):.4f} / 0.01 = EUR {exact_es:.4f}")

    print(f"\n  Comparison with MC ({mc_n:,} simulations):")
    print(f"  {'Measure':<12s} {'Exact':>10s} {'MC':>10s} {'Δ':>8s} {'Ratio':>8s}")
    print("  " + "-" * 50)
    print(f"  {'VaR 99%':<12s} {exact_var:>10.4f} {mc_var:>10.4f} "
          f"{mc_var-exact_var:>+8.4f} {mc_var/exact_var:>8.4f}")
    print(f"  {'ES 99%':<12s} {exact_es:>10.4f} {mc_es:>10.4f} "
          f"{mc_es-exact_es:>+8.4f} {mc_es/exact_es:>8.4f}")

    print(f"\n  Convergence study (seed={SEED}):")
    print(f"  {'N':>10s} {'MC VaR':>10s} {'Err%':>8s} {'MC ES':>10s} {'Err%':>8s}")
    print("  " + "-" * 48)
    for d in conv_data:
        print(f"  {d['n_sim']:>10,} {d['mc_var']:>10.4f} "
              f"{d['var_error_pct']:>+8.2f}% {d['mc_es']:>10.4f} "
              f"{d['es_error_pct']:>+8.2f}%")

    print(f"\n  Conditions for MC ≠ Exact:")
    print(f"    1. Non-linear payoff: the embedded cap creates convexity")
    print(f"       that the linear factor model ignores → MC on the full")
    print(f"       repricing engine would yield different tail behaviour.")
    print(f"    2. Fat-tailed factors: if rate/CDS changes follow Student-t")
    print(f"       or other heavy-tailed distributions, analytical normal")
    print(f"       formulas would understate VaR/ES.")
    print(f"    3. Factor correlation: non-zero credit-rate correlation")
    print(f"       would alter σ_P and shift the risk decomposition.")
    print("=" * 76)
    print(f"  Saved: {out_comparison}")
    print(f"  Saved: {out_conv}")
    print(f"  Saved: {out_chart}")


if __name__ == "__main__":
    main()
