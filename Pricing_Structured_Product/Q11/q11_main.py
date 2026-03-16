"""
q11_main.py
--------------------------------------------------------------------------------
Performs PCA on daily changes of EUR IRS par rates (1Y--10Y) from the
historical panel in MarketData202526.xlsx ("Hist IRS" sheet).
Outputs (output/)
-----------------
  q11_pca_eigenvalues.csv       Eigenvalues and variance explained
  q11_pca_loadings.csv          Eigenvector loadings (PC1--PC3)
  q11_pca_chart.png             Three-panel PCA chart
  Console report
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import openpyxl

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
COMMON_DIR = ROOT_DIR / "common"
MARKET_FILE = COMMON_DIR / "input" / "MarketData202526.xlsx"
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Colours (consistent with Q4/Q9/Q10)
# ---------------------------------------------------------------------------
NAVY = "#1a3a5c"
GREEN = "#1a7a3a"
RED = "#cc2200"
AMBER = "#c07800"
TEAL = "#0088aa"
GREY = "#888888"
PURPLE = "#6a3d9a"


# ---------------------------------------------------------------------------
# Tenor mapping: ICEIB code → display label and numeric maturity (years)
# ---------------------------------------------------------------------------
TENOR_MAP = {
    "ICEIB1Y": ("1Y", 1),
    "ICEIB2Y": ("2Y", 2),
    "ICEIB3Y": ("3Y", 3),
    "ICEIB4Y": ("4Y", 4),
    "ICEIB5Y": ("5Y", 5),
    "ICEIB6Y": ("6Y", 6),
    "ICEIB7Y": ("7Y", 7),
    "ICEIB8Y": ("8Y", 8),
    "ICEIB9Y": ("9Y", 9),
    "ICEIB10": ("10Y", 10),
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _xl_date(val) -> datetime | None:
    """Convert an Excel cell value to a Python datetime."""
    if isinstance(val, datetime):
        return val
    if isinstance(val, (int, float)):
        try:
            return datetime(1899, 12, 30) + __import__("datetime").timedelta(days=int(val))
        except Exception:
            return None
    return None


def load_hist_irs() -> tuple[list[datetime], list[str], np.ndarray]:
    """
    Load historical EUR IRS rates from "Hist IRS" sheet.

    Returns
    -------
    dates : list of datetime (length T)
    tenor_labels : list of str  (length N, e.g. ["1Y", "2Y", ..., "10Y"])
    rates : np.ndarray shape (T, N) — rates in percent
    """
    wb = openpyxl.load_workbook(MARKET_FILE, read_only=True, data_only=True)
    ws = wb["Hist IRS"]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    # Row 1 = ticker codes (ICEIB1Y, ICEIB2Y, ...)
    codes_raw = [str(c).strip() if c else "" for c in rows[1][1:]]

    # Map to the tenors we want (1Y--10Y)
    col_indices = []
    tenor_labels = []
    for idx, code in enumerate(codes_raw):
        if code in TENOR_MAP:
            col_indices.append(idx + 1)  # +1 for Date column
            tenor_labels.append(TENOR_MAP[code][0])

    # Data rows start at index 3
    dates = []
    rate_rows = []
    for row in rows[3:]:
        d = _xl_date(row[0])
        if d is None:
            continue
        vals = []
        skip = False
        for ci in col_indices:
            v = row[ci] if ci < len(row) else None
            if v is None or not isinstance(v, (int, float)):
                skip = True
                break
            vals.append(float(v))
        if skip:
            continue
        dates.append(d)
        rate_rows.append(vals)

    rates = np.array(rate_rows)  # shape (T, N), in percent
    return dates, tenor_labels, rates


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------
def run_pca(
    rates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA on daily first-differences of rates (in bps).

    Returns
    -------
    diffs     : (T-1, N) daily changes in bps
    eigenvals : (N,) eigenvalues (variance in bps^2/day)
    eigenvecs : (N, N) columns are eigenvectors (PC1 in col 0)
    var_expl  : (N,) fraction of variance explained
    """
    diffs = np.diff(rates, axis=0) * 100.0  # percent → bps

    # Remove rows with any NaN
    mask = ~np.isnan(diffs).any(axis=1)
    diffs_clean = diffs[mask]

    cov = np.cov(diffs_clean, rowvar=False)
    eigenvals, eigenvecs = np.linalg.eigh(cov)

    # Sort descending
    order = eigenvals.argsort()[::-1]
    eigenvals = eigenvals[order]
    eigenvecs = eigenvecs[:, order]

    var_expl = eigenvals / eigenvals.sum()

    # Sign convention: PC1 should be positive (level shift up = positive)
    for k in range(min(3, eigenvecs.shape[1])):
        if eigenvecs[:, k].mean() < 0:
            eigenvecs[:, k] *= -1

    return diffs_clean, eigenvals, eigenvecs, var_expl


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------
def save_pca_chart(
    tenor_labels: list[str],
    maturities: list[int],
    eigenvals: np.ndarray,
    eigenvecs: np.ndarray,
    var_expl: np.ndarray,
    dates: list[datetime],
    scores: np.ndarray,
    output_dir: Path,
) -> Path:
    """Three-panel PCA chart."""
    out_png = output_dir / "q11_pca_chart.png"
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    fig.patch.set_facecolor("#f9f9f9")

    n_pcs = min(10, len(eigenvals))

    # ── Panel 1: Scree plot (variance explained) ────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor("#f9f9f9")

    cum_var = np.cumsum(var_expl[:n_pcs]) * 100
    bars = ax1.bar(
        range(1, n_pcs + 1), var_expl[:n_pcs] * 100,
        color=NAVY, alpha=0.7, label="Individual", zorder=2,
    )
    ax1.plot(
        range(1, n_pcs + 1), cum_var,
        "o-", color=RED, lw=2, ms=5, label="Cumulative", zorder=3,
    )

    # Annotate first 3 bars
    for i in range(min(3, n_pcs)):
        ax1.text(
            i + 1, var_expl[i] * 100 + 1.5,
            f"{var_expl[i]*100:.1f}%",
            ha="center", fontsize=8, fontweight="bold", color=NAVY,
        )

    ax1.axhline(95, color=GREY, ls=":", lw=0.8, alpha=0.6)
    ax1.text(n_pcs + 0.3, 95, "95%", fontsize=7, color=GREY, va="center")

    ax1.set_xlabel("Principal Component", fontsize=9)
    ax1.set_ylabel("Variance Explained (%)", fontsize=9)
    ax1.set_title("Scree Plot", fontsize=10, fontweight="bold", color=NAVY)
    ax1.legend(fontsize=7, loc="center right")
    ax1.grid(axis="y", ls="--", alpha=0.3)
    ax1.set_xticks(range(1, n_pcs + 1))
    ax1.set_ylim(0, 105)

    # ── Panel 2: Eigenvector loadings (PC1-PC3) ────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#f9f9f9")

    pc_colors = [NAVY, GREEN, RED]
    pc_styles = ["o-", "s--", "D:"]
    pc_names = [
        f"PC1: Level ({var_expl[0]*100:.1f}%)",
        f"PC2: Slope ({var_expl[1]*100:.1f}%)",
        f"PC3: Curvature ({var_expl[2]*100:.1f}%)",
    ]

    for k in range(3):
        ax2.plot(
            maturities, eigenvecs[:, k],
            pc_styles[k], color=pc_colors[k], lw=2, ms=6,
            label=pc_names[k], zorder=3,
        )

    ax2.axhline(0, color="black", lw=0.5, alpha=0.4)
    ax2.set_xlabel("Maturity (years)", fontsize=9)
    ax2.set_ylabel("Eigenvector Loading", fontsize=9)
    ax2.set_title("Principal Component Loadings",
                  fontsize=10, fontweight="bold", color=NAVY)
    ax2.legend(fontsize=7, loc="best")
    ax2.grid(axis="both", ls="--", alpha=0.3)
    ax2.set_xticks(maturities)
    ax2.set_xticklabels(tenor_labels, fontsize=8)

    # ── Panel 3: Factor scores over time (PC1-PC3) ─────────────────────
    ax3 = axes[2]
    ax3.set_facecolor("#f9f9f9")

    # 60-day rolling mean of scores for smoothing
    window = 60
    for k in range(3):
        s = scores[:, k]
        # Rolling cumulative sum (approximate factor level)
        cum_score = np.cumsum(s)
        # Smooth
        if len(cum_score) > window:
            kernel = np.ones(window) / window
            smoothed = np.convolve(cum_score, kernel, mode="valid")
            t_offset = window // 2
            plot_dates = dates[1 + t_offset: 1 + t_offset + len(smoothed)]
            ax3.plot(
                plot_dates, smoothed,
                color=pc_colors[k], lw=1.2, alpha=0.8,
                label=f"PC{k+1} (cumul., {window}d MA)",
            )

    ax3.axhline(0, color="black", lw=0.5, alpha=0.4)
    ax3.set_xlabel("Date", fontsize=9)
    ax3.set_ylabel("Cumulative Factor Score (bps)", fontsize=9)
    ax3.set_title("Factor Score History",
                  fontsize=10, fontweight="bold", color=NAVY)
    ax3.legend(fontsize=7, loc="best")
    ax3.grid(axis="both", ls="--", alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax3.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=7)

    plt.tight_layout(pad=1.5)
    fig.savefig(out_png, dpi=200, bbox_inches="tight",
                pad_inches=0.05, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK] Saved {out_png.name}")
    return out_png


# ---------------------------------------------------------------------------
# CSV exports
# ---------------------------------------------------------------------------
def save_eigenvalue_csv(
    eigenvals: np.ndarray, var_expl: np.ndarray, output_dir: Path
) -> Path:
    out = output_dir / "q11_pca_eigenvalues.csv"
    cum = np.cumsum(var_expl)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pc", "eigenvalue_bps2", "var_explained_pct", "cumulative_pct"])
        for i in range(len(eigenvals)):
            w.writerow([
                f"PC{i+1}",
                f"{eigenvals[i]:.6f}",
                f"{var_expl[i]*100:.4f}",
                f"{cum[i]*100:.4f}",
            ])
    print(f"  [OK] Saved {out.name}")
    return out


def save_loadings_csv(
    tenor_labels: list[str],
    eigenvecs: np.ndarray,
    output_dir: Path,
) -> Path:
    out = output_dir / "q11_pca_loadings.csv"
    n_pcs = min(5, eigenvecs.shape[1])
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["tenor"] + [f"PC{i+1}" for i in range(n_pcs)]
        w.writerow(header)
        for j, label in enumerate(tenor_labels):
            row = [label] + [f"{eigenvecs[j, k]:.6f}" for k in range(n_pcs)]
            w.writerow(row)
    print(f"  [OK] Saved {out.name}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    dates, tenor_labels, rates = load_hist_irs()
    maturities = [TENOR_MAP[code][1] for code in TENOR_MAP if TENOR_MAP[code][0] in tenor_labels]

    diffs, eigenvals, eigenvecs, var_expl = run_pca(rates)

    # Factor scores: project diffs onto eigenvectors
    scores = diffs @ eigenvecs  # (T-1, N)

    # Chart
    save_pca_chart(
        tenor_labels, maturities,
        eigenvals, eigenvecs, var_expl,
        dates, scores,
        OUTPUT_DIR,
    )

    # CSVs
    save_eigenvalue_csv(eigenvals, var_expl, OUTPUT_DIR)
    save_loadings_csv(tenor_labels, eigenvecs, OUTPUT_DIR)

    # Console report
    print("\n" + "=" * 76)
    print("  QUESTION 11 -- PCA on EUR IRS Swap Rates")
    print("=" * 76)
    print(f"  Data source    : MarketData202526.xlsx, sheet 'Hist IRS'")
    print(f"  Date range     : {dates[0].strftime('%Y-%m-%d')} to "
          f"{dates[-1].strftime('%Y-%m-%d')}")
    print(f"  Observations   : {len(dates)} days ({len(diffs)} daily changes)")
    print(f"  Tenors         : {', '.join(tenor_labels)}")
    print("-" * 76)

    cum = np.cumsum(var_expl) * 100
    print("  EIGENVALUES AND VARIANCE EXPLAINED")
    print(f"  {'PC':<6} {'Eigenvalue':>12} {'Var Expl':>10} {'Cumulative':>12}")
    print("  " + "-" * 42)
    for i in range(min(5, len(eigenvals))):
        print(f"  PC{i+1:<3} {eigenvals[i]:>12.4f} {var_expl[i]*100:>9.2f}% "
              f"{cum[i]:>10.2f}%")
    print("-" * 76)

    print("  EIGENVECTOR LOADINGS (PC1-PC3)")
    print(f"  {'Tenor':<8} {'PC1 (Level)':>12} {'PC2 (Slope)':>12} {'PC3 (Curv)':>12}")
    print("  " + "-" * 46)
    for j, label in enumerate(tenor_labels):
        print(f"  {label:<8} {eigenvecs[j,0]:>12.4f} {eigenvecs[j,1]:>12.4f} "
              f"{eigenvecs[j,2]:>12.4f}")
    print("-" * 76)

    # Interpretation
    print("  INTERPRETATION")
    print("  " + "-" * 72)
    print(f"  PC1 (Level):     {var_expl[0]*100:.1f}% of variance. Near-uniform")
    print(f"                   loadings across tenors => parallel shift.")
    print(f"  PC2 (Slope):     {var_expl[1]*100:.1f}% of variance. Monotonically")
    print(f"                   increasing from short to long end => steepening/")
    print(f"                   flattening.")
    print(f"  PC3 (Curvature): {var_expl[2]*100:.1f}% of variance. Positive at")
    print(f"                   short+long, negative in the belly => butterfly.")
    print(f"  First 3 PCs explain {cum[2]:.1f}% of total variance.")
    print("-" * 76)

    # Bond relevance
    print("  BOND RELEVANCE")
    print("  " + "-" * 72)
    pc1_avg = eigenvecs[:, 0].mean()
    pc2_range = eigenvecs[-1, 1] - eigenvecs[0, 1]
    print(f"  The UniCredit bond has coupons linked to 3M EURIBOR (short end)")
    print(f"  but extends to 10Y maturity (long end). The leveraged coupon")
    print(f"  (1.60x) amplifies PC1 (level) exposure by 60%.")
    print(f"  PC2 (slope) matters because the par redemption discount factor")
    print(f"  depends on 8.6Y rates while coupons reset at the 3M point:")
    print(f"  a steepening (positive PC2 shock) raises coupon rates but")
    print(f"  lowers the redemption PV, partially offsetting each other.")
    print(f"  PC3 (curvature) is less material for this product as the")
    print(f"  coupon payoff is effectively at the very short end.")
    print("=" * 76)


if __name__ == "__main__":
    main()
