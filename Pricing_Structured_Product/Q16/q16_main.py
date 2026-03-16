"""
q16_main.py
--------------------------------------------------------------------------------
Proposes and sizes a CDS-based hedge to neutralise the bond's credit-spread
exposure (CS-DV01) measured in Q15.

Outputs (output/)
-----------------
  q16_cds_hedge.csv          Hedge specification and verification
  Console report
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import math
import sys
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
# Credit parameters
# ---------------------------------------------------------------------------
BASE_CDS_BPS = 75.0
RECOVERY_RATE = 0.40
LGD = 1.0 - RECOVERY_RATE
NOTIONAL = 1000.0  # Bond notional

CALENDAR = ql.TARGET()
CDS_DC = ql.Actual360()  # CDS premium leg day-count
OPTION_DC = ql.Actual365Fixed()

# ---------------------------------------------------------------------------
# Bond CS-DV01 from Q15
# ---------------------------------------------------------------------------
BOND_CS_DV01 = 0.75285  # EUR per 1 bp, per EUR 1,000 bond notional


# ---------------------------------------------------------------------------
# CDS valuation helpers
# ---------------------------------------------------------------------------
def compute_rpv01(
    curve: ql.YieldTermStructure,
    eval_date: ql.Date,
    cds_spread_bps: float,
    cds_tenor_years: int = 5,
) -> float:
    """
    Compute the Risky PV01 (risky annuity) of a CDS with quarterly
    premium payments.

    RPV01 = Σᵢ αᵢ × DF(tᵢ) × Q(tᵢ)

    where αᵢ is the ACT/360 day-count fraction for period i,
    DF is the risk-free discount factor, and Q is the survival
    probability under a flat hazard rate derived from the given spread.

    Returns RPV01 per unit notional (in years).
    """
    hazard_rate = (cds_spread_bps / 10_000.0) / LGD

    # Build CDS premium schedule: quarterly, CDS standard conventions
    cds_start = CALENDAR.advance(eval_date, ql.Period(1, ql.Days))
    cds_maturity = CALENDAR.advance(eval_date, ql.Period(cds_tenor_years, ql.Years))
    schedule = ql.MakeSchedule(
        effectiveDate=cds_start,
        terminationDate=cds_maturity,
        tenor=ql.Period(ql.Quarterly),
        calendar=CALENDAR,
        convention=ql.ModifiedFollowing,
        terminalDateConvention=ql.ModifiedFollowing,
        rule=ql.DateGeneration.Forward,
        endOfMonth=False,
    )

    dates = list(schedule)
    rpv01 = 0.0
    for i in range(len(dates) - 1):
        p_end = dates[i + 1]
        alpha = CDS_DC.yearFraction(dates[i], p_end)
        t = OPTION_DC.yearFraction(eval_date, p_end)
        df = curve.discount(p_end)
        q = math.exp(-hazard_rate * t)
        rpv01 += alpha * df * q
    return rpv01


def cds_mtm_protection_buyer(
    curve: ql.YieldTermStructure,
    eval_date: ql.Date,
    initial_spread_bps: float,
    current_spread_bps: float,
    notional: float,
    cds_tenor_years: int = 5,
) -> float:
    """
    Mark-to-market of a CDS protection-buyer position.

    When the market spread moves from s₀ to s₁, the protection buyer
    gains approximately:
        MTM = (s₁ − s₀) / 10,000 × RPV01(s₁) × N

    This uses RPV01 evaluated at the *current* spread (s₁), which
    reflects the updated survival probabilities.
    """
    rpv01_new = compute_rpv01(curve, eval_date, current_spread_bps,
                              cds_tenor_years)
    delta_s = (current_spread_bps - initial_spread_bps) / 10_000.0
    return delta_s * rpv01_new * notional


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    data = load_market_data()
    curve, eval_date, _ = build_curve(data)

    # ── Step 1: RPV01 at base spread ────────────────────────────────────
    rpv01_base = compute_rpv01(curve, eval_date, BASE_CDS_BPS)

    # ── Step 2: CDS CS-DV01 per unit notional ───────────────────────────
    # Bump ±1 bp for central difference
    rpv01_up = compute_rpv01(curve, eval_date, BASE_CDS_BPS + 1)
    rpv01_dn = compute_rpv01(curve, eval_date, BASE_CDS_BPS - 1)

    mtm_up = (1.0 / 10_000.0) * rpv01_up * 1.0   # per unit notional, +1bp
    mtm_dn = (-1.0 / 10_000.0) * rpv01_dn * 1.0  # per unit notional, -1bp
    cds_cs_dv01 = (mtm_up - mtm_dn) / 2.0         # per unit notional per bp

    # ── Step 3: Hedge notional ──────────────────────────────────────────
    hedge_notional = BOND_CS_DV01 / cds_cs_dv01
    hedge_direction = "Buy protection"

    # ── Step 4: Verification under Q15 spread bumps ─────────────────────
    bumps_bps = [-25, -10, -5, -1, 0, +1, +5, +10, +25]

    # Load Q15 bond price changes
    q15_csv = ROOT_DIR / "Q15" / "output" / "q15_cs_sensitivity.csv"
    bond_deltas: dict[int, float] = {}
    with open(q15_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bond_deltas[int(float(row["bump_bps"]))] = float(row["delta_clean"])

    results: list[dict] = []
    for bump in bumps_bps:
        new_spread = BASE_CDS_BPS + bump
        cds_mtm = cds_mtm_protection_buyer(
            curve, eval_date,
            initial_spread_bps=BASE_CDS_BPS,
            current_spread_bps=new_spread,
            notional=hedge_notional,
        )
        bond_delta = bond_deltas.get(bump, 0.0)
        hedged_pnl = bond_delta + cds_mtm

        # Hedge effectiveness
        if abs(bond_delta) > 1e-10:
            reduction_pct = (1.0 - abs(hedged_pnl) / abs(bond_delta)) * 100
        else:
            reduction_pct = 100.0

        results.append({
            "bump_bps": bump,
            "cds_bps": new_spread,
            "bond_delta": round(bond_delta, 4),
            "cds_mtm": round(cds_mtm, 4),
            "hedged_pnl": round(hedged_pnl, 4),
            "reduction_pct": round(reduction_pct, 2),
        })

    # ── Save CSV ─────────────────────────────────────────────────────────
    out_csv = OUTPUT_DIR / "q16_cds_hedge.csv"
    fieldnames = ["bump_bps", "cds_bps", "bond_delta", "cds_mtm",
                  "hedged_pnl", "reduction_pct"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)

    # ── Console report ───────────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("  QUESTION 16 -- CDS Hedging Strategy")
    print("=" * 76)
    print(f"  Eval date           : {eval_date}")
    print(f"  Base CDS spread     : {BASE_CDS_BPS:.0f} bps")
    print(f"  Recovery rate       : {RECOVERY_RATE:.0%}")
    print(f"  CDS tenor           : 5Y (quarterly premium, ACT/360)")
    print("-" * 76)
    print(f"  RPV01 (base spread) : {rpv01_base:.6f} yrs")
    print(f"  CDS CS-DV01 / unit  : {cds_cs_dv01:.8f} EUR per bp per EUR 1 notl")
    print(f"  Bond CS-DV01        : {BOND_CS_DV01:.5f} EUR per bp (EUR 1,000 bond)")
    print("-" * 76)
    print(f"  Hedge direction     : {hedge_direction}")
    print(f"  Hedge notional      : EUR {hedge_notional:,.2f}")
    print(f"  Hedge ratio         : {hedge_notional/NOTIONAL:.4f}x bond notional")
    print("=" * 76)

    print("\n  Verification: hedged portfolio under CDS spread bumps")
    print("-" * 76)
    print(f"  {'Bump':>6s}  {'CDS':>5s}  {'ΔBond':>9s}  {'CDS MTM':>9s}  "
          f"{'ΔHedged':>9s}  {'Redn %':>7s}")
    print("-" * 76)
    for r in results:
        marker = " <--" if r["bump_bps"] == 0 else ""
        print(f"  {r['bump_bps']:>+5d}  {r['cds_bps']:>5.0f}  "
              f"{r['bond_delta']:>+9.4f}  {r['cds_mtm']:>+9.4f}  "
              f"{r['hedged_pnl']:>+9.4f}  {r['reduction_pct']:>7.2f}{marker}")
    print("-" * 76)

    # Summary statistics
    non_zero = [r for r in results if r["bump_bps"] != 0]
    avg_reduction = sum(r["reduction_pct"] for r in non_zero) / len(non_zero)
    max_residual = max(abs(r["hedged_pnl"]) for r in non_zero)
    print(f"  Average reduction   : {avg_reduction:.1f}%")
    print(f"  Max residual P&L    : EUR {max_residual:.4f}")
    print("=" * 76)

    print("\n  Practical limitations:")
    print("  1. Maturity mismatch: bond matures Jun 2034 (~8.6Y) vs 5Y CDS;")
    print("     the credit DV01 profile differs across the term structure.")
    print("  2. CDS-bond basis: CDS spreads and bond credit spreads can")
    print("     diverge, leaving a residual basis risk.")
    print("  3. Roll risk: the 5Y CDS must be rolled before expiry,")
    print("     potentially at a different spread level.")
    print("  4. Counterparty risk: CDS protection is subject to the")
    print("     creditworthiness of the protection seller.")
    print("  5. Liquidity: single-name CDS on UniCredit may have wider")
    print("     bid-offer spreads than index CDS.")
    print("=" * 76)
    print(f"  Saved: {out_csv}")


if __name__ == "__main__":
    main()
