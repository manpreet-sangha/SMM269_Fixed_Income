"""
q3_validation_table.py
--------------------------------------------------------------------------------
Validates the bootstrapped curve by repricing every input instrument and
comparing the implied par rate against the original market quote.

Outputs
-------
  output/q3_validation.png   
  Console summary
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import QuantLib as ql

sys.path.insert(0, str(Path(__file__).parent))
from q3_market_data import load_market_data
from q3_main   import build_curve, _ql_date

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Implied-rate helpers
# ---------------------------------------------------------------------------
def _implied_deposit_rate(curve, start: ql.Date, end: ql.Date) -> float:
    """
    Implied simple ACT/360 deposit rate from the DF curve.
        r = (DF(start)/DF(end) - 1) / alpha
    """
    dc    = ql.Actual360()
    alpha = dc.yearFraction(start, end)
    df_s  = curve.discount(start)
    df_e  = curve.discount(end)
    return (df_s / df_e - 1.0) / alpha * 100.0


def _implied_swap_rate(curve, eval_date: ql.Date, years: int) -> float:
    """
    Implied par swap rate (annual fixed, 30/360 BondBasis) from the DF curve.

    Formula:
        c = (DF(settlement) - DF(T_n)) / Σ alpha_i * DF(T_i)
    where T_i are the annual fixed payment dates and alpha_i uses 30/360.
    """
    cal       = ql.TARGET()
    dc_fixed  = ql.Thirty360(ql.Thirty360.BondBasis)
    settlement = cal.advance(eval_date, 2, ql.Days)

    annuity = 0.0
    for y in range(1, years + 1):
        prev = cal.advance(settlement, y - 1, ql.Years, ql.ModifiedFollowing)
        curr = cal.advance(settlement, y,     ql.Years, ql.ModifiedFollowing)
        alpha = dc_fixed.yearFraction(prev, curr)
        annuity += alpha * curve.discount(curr)

    df_0 = curve.discount(settlement)
    df_n = curve.discount(cal.advance(settlement, years, ql.Years, ql.ModifiedFollowing))

    return (df_0 - df_n) / annuity * 100.0


# ---------------------------------------------------------------------------
# Build validation rows
# ---------------------------------------------------------------------------
def build_validation_rows(data: dict, curve, eval_date: ql.Date) -> list[dict]:
    cal        = ql.TARGET()
    settlement = cal.advance(eval_date, 2, ql.Days)

    rows: list[dict] = []

    # ------------------------------------------------------------------
    # Deposits (O/N skipped — matures before curve settlement date T+2)
    # ------------------------------------------------------------------
    _dep_periods: dict[str, ql.Period] = {
        "1W":  ql.Period(1,  ql.Weeks),
        "1M":  ql.Period(1,  ql.Months),
        "3M":  ql.Period(3,  ql.Months),
        "6M":  ql.Period(6,  ql.Months),
    }

    for tenor, mkt_rate in data["deposits"].items():
        if tenor not in _dep_periods:
            continue                         # skip ON and 12M
        # Deposits settle T+2; start = settlement, end = settlement + tenor
        end = cal.advance(settlement, _dep_periods[tenor], ql.ModifiedFollowing)

        # Implied simple ACT/360 rate:  r = (DF(start)/DF(end) - 1) / alpha
        # DF(settlement) = 1 by construction
        dc_dep   = ql.Actual360()
        alpha    = dc_dep.yearFraction(settlement, end)
        df_end   = curve.discount(end)
        implied  = (1.0 / df_end - 1.0) / alpha * 100.0
        error_bps = (implied - mkt_rate) * 100.0

        rows.append({
            "Instrument": "Deposit",
            "Tenor":      tenor,
            "Input (%)":  f"{mkt_rate:.4f}",
            "Implied (%)": f"{implied:.4f}",
            "Error (bps)": f"{error_bps:+.4f}",
            "_error_abs": abs(error_bps),
        })

    # ------------------------------------------------------------------
    # IRS (1Y–10Y)
    # ------------------------------------------------------------------
    for tenor, mkt_rate in data["irs"].items():
        years   = int(tenor.replace("Y", ""))
        implied = _implied_swap_rate(curve, eval_date, years)
        error_bps = (implied - mkt_rate) * 100.0

        rows.append({
            "Instrument": "IRS (AB6E)",
            "Tenor":      tenor,
            "Input (%)":  f"{mkt_rate:.4f}",
            "Implied (%)": f"{implied:.4f}",
            "Error (bps)": f"{error_bps:+.4f}",
            "_error_abs": abs(error_bps),
        })

    return rows


# ---------------------------------------------------------------------------
# PNG table
# ---------------------------------------------------------------------------
def generate_validation_png(rows: list[dict]) -> Path:
    headers    = ["Instrument", "Tenor", "Input (%)", "Implied (%)", "Error (bps)"]
    table_data = [headers] + [[r[h] for h in headers] for r in rows]
    n_rows     = len(table_data)

    fig, ax = plt.subplots(figsize=(5.0, n_rows * 0.22), dpi=350)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.axis("off")

    table = ax.table(
        cellText=table_data, cellLoc="center",
        bbox=[0, 0, 1, 1],
        colWidths=[0.22, 0.12, 0.20, 0.22, 0.24],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)

    NAVY = "#1a3a5c"
    RED  = "#ffeaea"

    # Header row
    for j in range(len(headers)):
        cell = table[(0, j)]
        cell.set_facecolor(NAVY)
        cell.set_text_props(weight="bold", color="white", fontsize=6.5)

    # Data rows
    for i in range(1, n_rows):
        error_abs = rows[i - 1]["_error_abs"]
        for j in range(len(headers)):
            cell = table[(i, j)]
            if j == 4 and error_abs > 0.1:
                cell.set_facecolor(RED)   # flag large error
            else:
                cell.set_facecolor("#f5f5f5" if i % 2 == 0 else "#ffffff")
            cell.set_text_props(fontsize=6.5)

    out = OUTPUT_DIR / "q3_validation.png"
    plt.savefig(out, bbox_inches="tight", pad_inches=0, dpi=350,
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"  Saved: {out}")
    return out

# ---------------------------------------------------------------------------
# Standalone usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data               = load_market_data()
    curve, eval_date, _ = build_curve(data)
    rows               = build_validation_rows(data, curve, eval_date)

    # Console table
    print(f"\n  {'INSTRUMENT':<12} {'TENOR':<6} {'INPUT%':>8} {'IMPLIED%':>10} {'ERR(bps)':>10}")
    print(f"  {'-'*12} {'-'*6} {'-'*8} {'-'*10} {'-'*10}")
    for r in rows:
        flag = " !" if r["_error_abs"] > 0.1 else ""
        print(f"  {r['Instrument']:<12} {r['Tenor']:<6} "
              f"{r['Input (%)']:>8} {r['Implied (%)']:>10} "
              f"{r['Error (bps)']:>10}{flag}")

    max_err = max(r["_error_abs"] for r in rows)
    status  = "PASS" if max_err < 0.1 else "FAIL"
    print(f"\n  Max |error|: {max_err:.4f} bps  →  {status}")

    generate_validation_png(rows)
