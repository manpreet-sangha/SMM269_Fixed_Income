"""
q1a_contractual_characteristics.py
================================================================================
Called from  : q1_main.py
Reads        : bond_details.txt (via the pre-parsed details dict)
Outputs      : console table  +  output/q1a_contractual_characteristics.txt
================================================================================
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Mapping: Section-A fields → (display label, why-it-matters note)
# ---------------------------------------------------------------------------
SECTION_A_FIELDS: list[tuple[str, str, str]] = [
    (
        "ISIN / Ticker",
        "ISIN",
        "Unique identifier for the note across all trading venues and data "
        "providers (Bloomberg: IT0005599110; EuroTLX). Essential for trade "
        "matching, regulatory reporting, and cross-sectional dataset merges.",
    ),
    (
        "Issuer",
        "Issuer Full Name",
        "UniCredit S.p.A. is a pan-European G-SIB rated A2 (Moody's equivalent). "
        "Issuer identity determines the credit-risk component of the yield spread, "
        "drives CVA/DVA inputs, and governs MREL eligibility treatment.",
    ),
    (
        "Seniority / Subordination",
        "Status / Seniority",
        "Senior (MREL-eligible under BRRD). Senior ranking means higher recovery "
        "priority in distress than Tier 2 / AT1 instruments, directly affecting "
        "credit spread and LGD assumptions in pricing models.",
    ),
    (
        "Bond Type",
        "Bond Type",
        "Floating Rate Note with 160% participation, 5.45% cap, and 0% floor. "
        "Bond type drives model selection: a plain vanilla FRN would use simple "
        "discounting; the leveraged collar requires a cap/floor options framework "
        "(Bachelier or Displaced Black).",
    ),
    (
        "Issue Date",
        "Issue Date",
        "12 June 2024. Determines the bond's age and seasoning. Seasoning affects "
        "liquidity (older bonds are often less liquid) and the number of remaining "
        "coupon periods, which directly enter the QuantLib schedule.",
    ),
    (
        "Maturity Date",
        "Maturity Date",
        "12 June 2034 (10-year tenor). The primary determinant of modified duration, "
        "DV01, and yield-curve positioning. Longer tenor increases the value of the "
        "embedded floor while amplifying sensitivity to parallel shifts in EURIBOR.",
    ),
    (
        "Issue Size (Outstanding)",
        "Aggregate Nominal Amount",
        "EUR 20,000,000. A proxy for secondary-market liquidity: smaller issues "
        "trade with wider bid-ask spreads and lower turnover. Relevant when "
        "estimating transaction costs and liquidity-adjusted fair value.",
    ),
]


# ---------------------------------------------------------------------------
# Main reporting function
# ---------------------------------------------------------------------------

def print_contractual_characteristics(
    details: dict[str, str],
    output_dir: Path | None = None,
) -> None:
    """
    Print Section-A contractual & identification characteristics to the console
    and save a plain-text copy to output_dir/q1a_contractual_characteristics.txt.

    Parameters
    ----------
    details    : pre-parsed bond_details dict from load_bond_details()
    output_dir : directory where the .txt file is written (created if absent)
    """
    lines: list[str] = []

    def pr(text: str = "") -> None:
        print(text)
        lines.append(text)

    sep_thick = "=" * 72
    sep_thin  = "-" * 72

    pr()
    pr(sep_thick)
    pr("  SECTION A  –  CONTRACTUAL & IDENTIFICATION CHARACTERISTICS")
    pr("  UniCredit Variable Rate Bond 2034  |  ISIN IT0005599110")
    pr(sep_thick)
    pr()

    for display_label, field_key, why_it_matters in SECTION_A_FIELDS:
        value = details.get(field_key, "N/A").strip()
        # Trim very long raw values to first 80 chars for display
        value_display = value if len(value) <= 80 else value[:77] + "..."

        pr(f"  {'CHARACTERISTIC':<32}  {'VALUE'}")
        pr(f"  {display_label:<32}  {value_display}")
        pr(f"  {sep_thin}")
        # Word-wrap the why-it-matters note at ~68 chars
        words = why_it_matters.split()
        current_line = "  WHY IT MATTERS:  "
        for word in words:
            if len(current_line) + len(word) + 1 > 72:
                pr(current_line)
                current_line = "                    " + word
            else:
                current_line += (" " if current_line.strip() else "") + word
        if current_line.strip():
            pr(current_line)
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
        out_file = output_dir / "q1a_contractual_characteristics.txt"
        out_file.write_text("\n".join(lines), encoding="utf-8")
        print(f"  [OK] Section A saved -> {out_file}\n")
