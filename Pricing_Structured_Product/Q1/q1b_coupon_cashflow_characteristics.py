"""
q1b_coupon_cashflow_characteristics.py
================================================================================
Section B -- Coupon & Cash-Flow Characteristics

Called from  : q1_main.py
Reads        : bond_details.txt (via the pre-parsed details dict)
Outputs      : console table  +  output/q1b_coupon_cashflow_characteristics.txt
================================================================================
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _get_float(details: dict, key: str, default: float = 0.0) -> float:
    raw = details.get(key, str(default))
    numeric = re.sub(r"[^0-9.\-]", "", raw.split()[0])
    try:
        return float(numeric)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Main reporting function
# ---------------------------------------------------------------------------

def print_coupon_cashflow_characteristics(
    details: dict[str, str],
    output_dir: Path | None = None,
) -> None:
    """
    Print Section-B coupon & cash-flow characteristics to the console and save
    a plain-text copy to output_dir/q1b_coupon_cashflow_characteristics.txt.

    Parameters
    ----------
    details    : pre-parsed bond_details dict from load_bond_details()
    output_dir : directory where the .txt file is written (created if absent)
    """
    # -- Derived values -------------------------------------------------------
    participation   = _get_float(details, "Participation Factor", 1.60)
    floor_pct       = _get_float(details, "Floor (Min Rate)",     0.00)
    cap_pct         = _get_float(details, "Cap (Max Rate)",       5.45)
    current_coupon  = _get_float(details, "Current Coupon Rate",  3.2464)
    implied_euribor = current_coupon / participation        # ~ 2.029 %
    be_floor        = floor_pct / participation              # EURIBOR floor break-even
    be_cap          = cap_pct   / participation              # EURIBOR cap  break-even

    lines: list[str] = []

    def pr(text: str = "") -> None:
        print(text)
        lines.append(text)

    def wrap(text: str, indent: int = 20) -> None:
        """Word-wrap a note at 72 chars with a hanging indent."""
        words = text.split()
        current = " " * indent
        for word in words:
            if len(current) + len(word) + 1 > 72:
                pr(current)
                current = " " * indent + word
            else:
                current += (" " if current.strip() else "") + word
        if current.strip():
            pr(current)

    sep_thick = "=" * 72
    sep_thin  = "-" * 72

    pr()
    pr(sep_thick)
    pr("  SECTION B  --  COUPON & CASH-FLOW CHARACTERISTICS")
    pr("  UniCredit Variable Rate Bond 2034  |  ISIN IT0005599110")
    pr(sep_thick)
    pr()

    # -------------------------------------------------------------------------
    # 1. Coupon Rate / Formula
    # -------------------------------------------------------------------------
    formula_str = (
        f"c(r) = clip({participation:.2f} x r,  {floor_pct:.2f}%,  {cap_pct:.2f}%)"
    )
    pr(f"  {'CHARACTERISTIC':<32}  VALUE")
    pr(f"  {'Coupon Rate / Formula':<32}  {formula_str}")
    pr(f"  Current coupon (last reset)  :  {current_coupon:.4f}%  "
       f"(implied 3M EURIBOR = {implied_euribor:.4f}%)")
    pr(f"  {sep_thin}")
    pr()

    # -------------------------------------------------------------------------
    # 2. Day Count Convention
    # -------------------------------------------------------------------------
    dc = details.get("Day Count Convention", "30/360").strip()
    pr(f"  {'Day Count Convention':<32}  {dc}")
    pr(f"  {sep_thin}")
    pr()

    # -------------------------------------------------------------------------
    # 3. Payment Frequency
    # -------------------------------------------------------------------------
    freq = details.get("Coupon Frequency", "Quarterly").strip()
    reset_date  = details.get("Reset / Determination Date", "N/A").strip()
    pay_dates   = details.get("Interest Payment Dates", "N/A").strip()
    bdc         = details.get("Business Day Convention", "N/A").strip()
    add_centre  = details.get("Additional Business Centre", "N/A").strip()

    pr(f"  {'Payment Frequency':<32}  {freq}")
    pr(f"  {'Reset / Determination Date':<32}  {reset_date}")
    pr(f"  {'Interest Payment Dates':<32}  {pay_dates}")
    pr(f"  {'Business Day Convention':<32}  {bdc}")
    pr()

    # -------------------------------------------------------------------------
    # 4. Reference Rate
    # -------------------------------------------------------------------------
    ref_rate   = details.get("Reference Rate", "3m Euribor").strip()
    screen_pg  = details.get("Screen Page", "Bloomberg EUR003M Index").strip()
    ref_repl   = details.get("Reference Rate Replacement", "N/A").strip()

    pr(f"  {'Reference Rate':<32}  {ref_rate}")
    pr(f"  {'Screen Page':<32}  {screen_pg}")
    pr(f"  {'Reference Rate Replacement':<32}  {ref_repl}")
    pr(f"  {sep_thin}")
    pr()

    # -------------------------------------------------------------------------
    # 5. Coupon Caps / Floors
    # -------------------------------------------------------------------------
    pr(f"  {'Coupon Cap (Max Rate)':<32}  {cap_pct:.2f}% p.a.  "
       f"(EURIBOR break-even: {be_cap:.4f}%)")
    pr(f"  {'Coupon Floor (Min Rate)':<32}  {floor_pct:.2f}% p.a.  "
       f"(EURIBOR break-even: {be_floor:.4f}%)")
    pr(f"  {'Participation / Leverage':<32}  {participation:.2f}x")
    pr()

    # -------------------------------------------------------------------------
    # 6. Embedded Options
    # -------------------------------------------------------------------------
    call_opt   = details.get("Call Option",           "N/A").strip()
    std_call   = details.get("Issuer Call (Standard)","N/A").strip()
    cleanup    = details.get("Clean-Up Redemption",   "N/A").strip()
    red_type   = details.get("Redemption Type",       "N/A").strip()
    red_basis  = details.get("Redemption Basis",      "N/A").strip()

    pr(f"  {'Call Option':<32}  {call_opt}")
    pr(f"  {'Issuer Call (Standard)':<32}  {std_call}")
    pr(f"  {'Clean-Up Redemption':<32}  {cleanup}")
    pr(f"  {'Redemption Type':<32}  {red_type}")
    if red_basis and red_basis != "N/A":
        pr(f"  {'Redemption Basis':<32}  {red_basis}")
    pr()

    pr(sep_thick)
    pr(f"  Generated: {datetime.now().strftime('%d %b %Y  %H:%M')}")
    pr(sep_thick)
    pr()

    # ------------------------------------------------------------------
    # Save to output file
    # ------------------------------------------------------------------
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / "q1b_coupon_cashflow_characteristics.txt"
        out_file.write_text("\n".join(lines), encoding="utf-8")
        print(f"  [OK] Section B saved -> {out_file}\n")
