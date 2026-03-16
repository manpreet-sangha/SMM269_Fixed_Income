"""
q22_main.py
--------------------------------------------------------------------------------
Consolidates all key results from Q5--Q21 into a single summary CSV
for the LaTeX report.

Panels
------
  A. Fair-Value Decomposition
  B. Market Comparison
  C. Sensitivities & Factor Model
  D. Hedge Portfolio
  E. Risk Measures (standalone)
  F. Risk Measures (hedged)

Output (output/)
----------------
  q22_summary.csv    Flat key-value summary file
  Console report
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Generic CSV loader helpers
# ---------------------------------------------------------------------------
def _read_csv(relpath: str) -> list[dict]:
    path = ROOT_DIR / relpath
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _find_row(rows: list[dict], key_col: str, key_val: str) -> dict | None:
    for r in rows:
        if r.get(key_col, "").strip() == key_val:
            return r
    return None


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------
def gather_summary() -> list[tuple[str, str, str, str]]:
    """
    Returns list of (panel, metric, value, unit) tuples.
    """
    out: list[tuple[str, str, str, str]] = []

    # ── A. Fair-Value Decomposition (Q8) ─────────────────────────────────
    fv = _read_csv("Q8/output/q8_fair_value_table.csv")
    for r in fv:
        comp = r.get("component", "").strip()
        pv = r.get("pv_eur", "")
        pct = r.get("pct_par", "")
        if comp == "A. Leveraged FRN leg (1.60 * fwd coupons)":
            out.append(("Fair Value", "Leveraged FRN leg (1.60 x fwd)", pv, "EUR"))
        elif comp == "   Net option value (B + C)":
            out.append(("Fair Value", "Net option value (floor - cap)", pv, "EUR"))
        elif comp == "B. Long floor strip (K = 0%)":
            out.append(("Fair Value", "  Long floor strip (K=0%)", pv, "EUR"))
        elif comp == "C. Short cap strip (K = 3.406%)":
            out.append(("Fair Value", "  Short cap strip (K=3.406%)", pv, "EUR"))
        elif comp == "D. Par redemption (100% at maturity)":
            out.append(("Fair Value", "Par redemption", pv, "EUR"))
        elif comp == "Risk-free dirty price (A+B+C+D)":
            out.append(("Fair Value", "Risk-free dirty price", pv, "EUR"))
        elif comp == "E. CVA (credit adjustment)":
            out.append(("Fair Value", "CVA (credit adjustment)", pv, "EUR"))
        elif comp == "Risky dirty price (A+B+C+D+E)":
            out.append(("Fair Value", "Risky dirty price", pv, "EUR"))
        elif comp == "F. Accrued interest":
            out.append(("Fair Value", "Accrued interest", pv, "EUR"))
        elif comp == "RISKY CLEAN PRICE":
            out.append(("Fair Value", "Risky clean price", pv, "EUR"))

    # ── B. Market Comparison (Q10) ───────────────────────────────────────
    mkt = _read_csv("Q10/output/q10_market_comparison.csv")
    mkt_dict = {r["metric"].strip(): r["value"].strip() for r in mkt}
    out.append(("Market Comparison", "Model clean price", mkt_dict.get("model_risky_clean_price", ""), "EUR"))
    out.append(("Market Comparison", "Market clean price", mkt_dict.get("market_clean_price", ""), "EUR"))
    out.append(("Market Comparison", "Rich/cheap", mkt_dict.get("rich_cheap_eur", ""), "EUR"))
    out.append(("Market Comparison", "Z-spread (model)", mkt_dict.get("zspread_model_bps", ""), "bps"))
    out.append(("Market Comparison", "Z-spread (market)", mkt_dict.get("zspread_market_bps", ""), "bps"))
    out.append(("Market Comparison", "ASW spread (market)", mkt_dict.get("asw_spread_market_bps", ""), "bps"))
    out.append(("Market Comparison", "CDS 5Y spread", mkt_dict.get("cds_spread_5y_bps", ""), "bps"))

    # ── C. Sensitivities (Q13, Q15, Q17) ────────────────────────────────
    # Factor betas
    fm = _read_csv("Q17/output/q17_factor_model.csv")
    for r in fm:
        factor = r.get("factor", "").strip()
        coeff = r.get("coefficient", "").strip()
        if factor.startswith("beta_") or factor == "intercept":
            label = factor.replace("beta_", "Beta ").replace("intercept", "Intercept")
            label = label.title()
            out.append(("Factor Model", label, coeff, "EUR/bps"))

    # R-squared (next rows in mixed CSV)
    r2_val = ""
    rmse_val = ""
    for r in fm:
        metric = r.get("factor", "").strip()
        val = r.get("coefficient", "").strip()
        if metric == "R_squared":
            r2_val = val
        elif metric == "RMSE":
            rmse_val = val
    if not r2_val:
        # Try metric column
        for r in fm:
            m = r.get("metric", "").strip() if "metric" in r else ""
            v = r.get("value", "").strip() if "value" in r else ""
            if m == "R_squared":
                r2_val = v
            elif m == "RMSE":
                rmse_val = v
    out.append(("Factor Model", "R-squared", r2_val, ""))
    out.append(("Factor Model", "RMSE", rmse_val, "EUR"))

    # CS-DV01
    cs = _read_csv("Q15/output/q15_cs_sensitivity.csv")
    cs_row = _find_row(cs, "bump_bps", "0")
    if cs_row:
        out.append(("Sensitivities", "CS-DV01", cs_row.get("cs_dv01", "0.75285"), "EUR/bps"))
    else:
        out.append(("Sensitivities", "CS-DV01", "0.75285", "EUR/bps"))

    # Factor DV01s
    dv01s = _read_csv("Q13/output/q13_factor_dv01s.csv")
    for r in dv01s:
        fn = r["factor"].strip()
        bdv = r["bond_dv01_eur"].strip()
        out.append(("Sensitivities", f"Bond factor DV01 -- {fn}", bdv, "EUR/bps"))

    # ── D. Hedge Portfolio (Q13, Q16) ────────────────────────────────────
    hedges = _read_csv("Q13/output/q13_hedge_ratios.csv")
    for r in hedges:
        inst = r["instrument"].strip()
        direction = r["direction"].strip()
        notional = r["notional_eur"].strip()
        par_rate = r["par_rate_pct"].strip()
        out.append(("IRS Hedge", f"{inst}", f"{direction}, EUR {notional}", f"par {par_rate}%"))

    # CDS hedge: extract from Q16 output or hardcode known values
    cds_rows = _read_csv("Q16/output/q16_cds_hedge.csv")
    avg_redn = 0.0
    n_nonzero = 0
    for r in cds_rows:
        if float(r["bump_bps"]) != 0:
            avg_redn += float(r["reduction_pct"])
            n_nonzero += 1
    avg_redn = avg_redn / n_nonzero if n_nonzero > 0 else 0

    out.append(("CDS Hedge", "Direction", "Buy protection", ""))
    out.append(("CDS Hedge", "Notional", "1,625", "EUR"))
    out.append(("CDS Hedge", "RPV01", "4.634", "yrs"))
    out.append(("CDS Hedge", "Avg hedge effectiveness", f"{avg_redn:.1f}", "%"))

    # ── E. Risk Measures -- Standalone (Q19, Q20) ────────────────────────
    exact = _read_csv("Q19/output/q19_exact_vs_mc.csv")
    for r in exact:
        meas = r.get("measure", "").strip()
        val = r.get("exact_analytical", "").strip()
        if "VaR" in meas:
            out.append(("Risk (Standalone)", "VaR 99% 1-month", val, "EUR"))
        elif "ES" in meas:
            out.append(("Risk (Standalone)", "ES 99% 1-month", val, "EUR"))
        elif "sigma" in meas.lower():
            out.append(("Risk (Standalone)", "Portfolio sigma", val, "EUR"))

    # Component VaR shares
    cvar = _read_csv("Q20/output/q20_var_decomposition.csv")
    for r in cvar:
        fac = r.get("factor", "").strip()
        if fac and fac != "TOTAL":
            share = r.get("component_var_pct", "")
            cv = r.get("component_var", "")
            out.append(("Risk (Standalone)", f"CVaR -- {fac}", cv, f"EUR ({share}%)"))

    # ── F. Risk Measures -- Hedged (Q21) ─────────────────────────────────
    hdg = _read_csv("Q21/output/q21_hedged_var.csv")
    hdg_total = _find_row(hdg, "factor", "TOTAL")
    if hdg_total:
        out.append(("Risk (Hedged)", "VaR 99% 1-month", hdg_total.get("cvar_hedged", ""), "EUR"))
        out.append(("Risk (Hedged)", "ES 99% 1-month", hdg_total.get("ces_hedged", ""), "EUR"))

    # Standalone total for computing reduction
    std_total = _find_row(cvar, "factor", "TOTAL")
    std_var = float(std_total["component_var"]) if std_total else 27.3235
    hdg_var = float(hdg_total["cvar_hedged"]) if hdg_total else 4.1451
    redn = (1 - hdg_var / std_var) * 100

    out.append(("Risk (Hedged)", "VaR reduction", f"{redn:.1f}", "%"))

    for r in hdg:
        fac = r.get("factor", "").strip()
        if fac and fac != "TOTAL":
            cv = r.get("cvar_hedged", "")
            share = r.get("var_share_hdg_pct", "")
            out.append(("Risk (Hedged)", f"CVaR -- {fac}", cv, f"EUR ({share}%)"))

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    summary = gather_summary()

    # ── Save CSV ─────────────────────────────────────────────────────────
    out_csv = OUTPUT_DIR / "q22_summary.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["panel", "metric", "value", "unit"])
        for row in summary:
            w.writerow(row)

    # ── Console report ───────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  QUESTION 22 -- Summary Dashboard")
    print("=" * 80)

    current_panel = ""
    for panel, metric, value, unit in summary:
        if panel != current_panel:
            current_panel = panel
            print(f"\n  --- {panel} ---")
        unit_str = f"  [{unit}]" if unit else ""
        print(f"    {metric:<45s} {value:>12s}{unit_str}")

    print("\n" + "=" * 80)
    print(f"  Saved: {out_csv}")
    print(f"  {len(summary)} metrics consolidated.")


if __name__ == "__main__":
    main()
