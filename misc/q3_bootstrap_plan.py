"""
q3_bootstrap_plan.py
================================================================================
Reads the PDF references in misc/refdocs/Term Structure Bootstrapping/ and
produces a structured step-by-step implementation plan for Question 3:

    "Bootstrap the term structure of interest rates using EURIBOR and
     interest rate swap (IRS) quotes. Clearly describe the methodology
     and interpolation technique adopted."

Outputs
-------
  misc/refdocs/q3_bootstrap_plan.txt   – step-by-step plan with citations
  (references.bib is updated separately – see instructions at bottom)

Dependencies
------------
  pip install pdfplumber
================================================================================
"""

from __future__ import annotations

import re
import sys
import pathlib
from datetime import date
from collections import Counter

import pdfplumber

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── paths ─────────────────────────────────────────────────────────────────────
MISC_DIR   = pathlib.Path(__file__).parent
REFDOCS    = MISC_DIR / "refdocs" / "Term Structure Bootstrapping"
OUT_TXT    = MISC_DIR / "refdocs" / "q3_bootstrap_plan.txt"

PDFS = {
    "handbook":  REFDOCS / "The Professional Handbook of Financial Risk Management The Yield Curve.pdf",
    "artscience": REFDOCS / "TheArtAndScienceofCurveBuilding.pdf",
    "maxsmooth":  REFDOCS / "jfi_1994_Max_Smoothness_v03.pdf",
}

# ── keyword sets to locate relevant passages ──────────────────────────────────
BOOTSTRAP_KEYS = [
    "bootstrap", "bootstrapping", "strip", "stripping",
    "deposit rate", "money market", "overnight", "euribor",
    "swap rate", "par swap", "interest rate swap", "irs",
    "discount factor", "zero rate", "spot rate", "zero coupon",
]

INTERP_KEYS = [
    "interpolat", "log-linear", "log linear", "piecewise",
    "cubic spline", "cubic splice", "hermite", "monotone convex",
    "monotone-convex", "natural spline", "flat forward",
    "step-forward", "step forward", "linear forward", "smooth",
    "maximum smoothness",
]

CURVE_KEYS = [
    "yield curve", "term structure", "zero curve", "forward curve",
    "discount curve", "ois", "dual curve", "multi-curve", "multicurve",
    "overnight index swap", "eonia", "€str", "ester",
]

MODEL_KEYS = [
    "nelson", "siegel", "svensson", "vasicek", "cox ingersoll",
    "hull white", "ho lee", "libor market model", "bgm",
    "arbitrage free", "no-arbitrage", "parametric",
    "instantaneous forward", "par rate"
]

# ── helpers ───────────────────────────────────────────────────────────────────

def extract_text(pdf_path: pathlib.Path) -> str:
    """Extract all text from a PDF, handling encoding issues."""
    if not pdf_path.exists():
        return f"[FILE NOT FOUND: {pdf_path}]"
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                pages.append(t)
    raw = "\n".join(pages)
    raw = re.sub(r"\x00", " ", raw)
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()

def find_passages(text: str, keywords: list[str], window: int = 600) -> list[str]:
    """
    Find paragraphs/sentences containing any keyword.
    Returns up to 6 unique snippets (window chars around first hit).
    """
    text_lower = text.lower()
    passages = []
    seen_starts = set()
    for kw in keywords:
        for m in re.finditer(re.escape(kw.lower()), text_lower):
            start = max(0, m.start() - window // 2)
            end   = min(len(text), m.start() + window // 2)
            # snap to sentence/word boundaries
            while start > 0 and text[start] not in ".!?\n":
                start -= 1
            bucket = (start // 300)  # deduplicate nearby hits
            if bucket not in seen_starts:
                seen_starts.add(bucket)
                snippet = text[start:end].strip()
                snippet = re.sub(r"\n+", " ", snippet)
                passages.append(snippet)
            if len(passages) >= 8:
                break
        if len(passages) >= 8:
            break
    return passages

def section_headings(text: str, max_h: int = 20) -> list[str]:
    """Detect likely headings: short lines (≤80 chars) that are title-cased or numbered."""
    headings = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or len(line) > 90:
            continue
        is_numbered = bool(re.match(r"^(\d+[\.\d]*|[IVXLC]+\.)\s+\w", line))
        is_allcaps  = line.isupper() and len(line) > 4
        is_titlecase = (line[0].isupper() and sum(1 for w in line.split() if w[0:1].isupper()) >= 2
                        and len(line.split()) <= 10)
        if is_numbered or is_allcaps or is_titlecase:
            headings.append(line)
        if len(headings) >= max_h:
            break
    return headings


# ── main extraction ───────────────────────────────────────────────────────────

def analyse_pdf(label: str, path: pathlib.Path) -> dict:
    text = extract_text(path)
    n_chars = len(text)
    n_pages = 0
    try:
        with pdfplumber.open(path) as pdf:
            n_pages = len(pdf.pages)
    except Exception:
        pass

    return {
        "label":      label,
        "path":       path,
        "n_pages":    n_pages,
        "n_chars":    n_chars,
        "text":       text,
        "headings":   section_headings(text),
        "bootstrap":  find_passages(text, BOOTSTRAP_KEYS),
        "interp":     find_passages(text, INTERP_KEYS),
        "curves":     find_passages(text, CURVE_KEYS),
        "models":     find_passages(text, MODEL_KEYS),
    }


# ── step-by-step plan builder ─────────────────────────────────────────────────

PLAN = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Q3 STEP-BY-STEP IMPLEMENTATION PLAN
  SMM269 Fixed Income / Coursework
  Generated : {date}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QUESTION (verbatim)
───────────────────
"Bootstrap the term structure of interest rates using EURIBOR and interest rate
 swap (IRS) quotes. Clearly describe the methodology and interpolation technique
 adopted."

═══════════════════════════════════════════════════════════════════════════════
  METHODOLOGY OVERVIEW
═══════════════════════════════════════════════════════════════════════════════

Bootstrapping constructs a zero-coupon (spot) yield curve by extracting
one risk-free discount factor at a time, cascading outward from the shortest
to the longest maturity.  Two sets of liquid EUR instruments are combined:

  1. EURIBOR deposits  (tenors: O/N, 1W, 1M, 3M, 6M, 12M)
     → directly traded rates; give clean short-end anchor points
     → day count: ACT/360; settlement: T+2 (except O/N = T+0)

  2. EUR Interest Rate Swaps against 3M EURIBOR  (tenors: 2Y–10Y)
     → most liquid EUR long-end instrument
     → quoted as par (fixed) rates; day count fixed leg: 30/360; float: ACT/360
     → annual fixed payments vs quarterly EURIBOR float

Valuation date: 9 March 2026

═══════════════════════════════════════════════════════════════════════════════
  STEP 1 — MARKET DATA
═══════════════════════════════════════════════════════════════════════════════

Collect the following quotes for 9 March 2026:

  EURIBOR deposit rates (source: Bank of Finland / ECB)
  ┌──────────┬──────────────┐
  │ Tenor    │ Rate (%)     │
  ├──────────┼──────────────┤
  │ O/N      │ ~2.40        │
  │ 1W       │ ~2.42        │
  │ 1M       │ ~2.50        │
  │ 3M       │ ~2.55        │
  │ 6M       │ ~2.60        │
  │ 12M      │ ~2.65        │
  └──────────┴──────────────┘
  (Replace with actual fixings from common/input/ Excel files)

  EUR IRS par rates (fixed vs 3M EURIBOR, source: Bloomberg/Refinitiv/ECB)
  ┌──────────┬──────────────┐
  │ Tenor    │ Rate (%)     │
  ├──────────┼──────────────┤
  │ 2Y       │ ~2.55        │
  │ 3Y       │ ~2.60        │
  │ 5Y       │ ~2.72        │
  │ 7Y       │ ~2.85        │
  │ 10Y      │ ~3.00        │
  └──────────┴──────────────┘
  (Replace with actual market quotes; indicative values shown)

═══════════════════════════════════════════════════════════════════════════════
  STEP 2 — SHORT END: DEPOSITS → DISCOUNT FACTORS
═══════════════════════════════════════════════════════════════════════════════

For each EURIBOR deposit of tenor T with quoted rate r (ACT/360):

    DF(T) = 1 / (1 + r × α)

where α = actual calendar days / 360.

This yields exact discount factors for O/N through 12M with no iteration.

QuantLib object: ql.DepositRateHelper(rate, tenor, fixingDays,
                                       calendar, convention, endOfMonth,
                                       dayCounter)

═══════════════════════════════════════════════════════════════════════════════
  STEP 3 — LONG END: IRS → DISCOUNT FACTORS (BOOTSTRAPPING)
═══════════════════════════════════════════════════════════════════════════════

A par swap has zero NPV at inception.  The fixed leg pays coupon c annually
(30/360); the floating leg resets quarterly to 3M EURIBOR.

Par swap pricing equation:

    c × Σ αᵢ × DF(tᵢ)  =  1 − DF(Tₙ)    [i = 1..n]
    i=1

Rearranging to solve for the unknown final DF:

    DF(Tₙ) = [1 − c × Σᵢ₌₁ⁿ⁻¹ αᵢ × DF(tᵢ)]  /  [1 + c × αₙ]

All prior DF(tᵢ) values come from shorter instruments already bootstrapped.
Each new maturity adds exactly one equation and one unknown → solved exactly.

QuantLib object:  ql.SwapRateHelper(rate, tenor, calendar, fixedFrequency,
                                     fixedConvention, fixedDayCount, iborIndex)

═══════════════════════════════════════════════════════════════════════════════
  STEP 4 — INTERPOLATION BETWEEN PILLARS
═══════════════════════════════════════════════════════════════════════════════

Instrument quotes exist at ~11 maturities.  A continuous curve requires
interpolation for all intermediate dates.

METHOD CHOSEN: Log-linear interpolation on discount factors
  (equivalent to piecewise-constant instantaneous forward rates)

    ln DF(t) = ln DF(T₁) + [(t − T₁)/(T₂ − T₁)] × (ln DF(T₂) − ln DF(T₁))

WHY THIS CHOICE:
  • Guarantees positive forward rates (arbitrage-free between pillars)
  • Simple, fast, and numerically stable
  • Standard in practice (Bloomberg "Step-Forward" = same concept)
  • Used by QuantLib's PiecewiseLogLinearDiscount bootstrapper
  • Hagan & West (2006) validate it as the minimal-arbitrage baseline

ALTERNATIVE CONSIDERED: Monotone Convex (Hagan & West 2006)
  • Produces smooth forward curve and preserves monotonicity
  • More complex; not needed for a coursework-level single-curve build

QuantLib curve type: ql.PiecewiseLogLinearDiscount

═══════════════════════════════════════════════════════════════════════════════
  STEP 5 — DERIVED QUANTITIES
═══════════════════════════════════════════════════════════════════════════════

From the bootstrapped DF curve compute:

  Zero (spot) rate at maturity T (continuous compounding):
      z(T) = −ln DF(T) / T

  Zero rate (annual compounding):
      z_ann(T) = DF(T)^(−1/T) − 1

  3M instantaneous forward rate between T₁ and T₂:
      f(T₁,T₂) = ln[DF(T₁)/DF(T₂)] / (T₂ − T₁)

═══════════════════════════════════════════════════════════════════════════════
  STEP 6 — VALIDATION
═══════════════════════════════════════════════════════════════════════════════

Re-price every input instrument using the bootstrapped curve:
  • Each deposit: should reprice to par (NPV = 0)
  • Each swap:    should reprice to par (NPV = 0, i.e. fixed rate = par rate)

Tolerance: |error| < 0.01 bps (1e-6 in rate terms)
Any failure indicates a date, day-count, or calendar convention mismatch.

═══════════════════════════════════════════════════════════════════════════════
  STEP 7 — OUTPUT CHARTS AND TABLES
═══════════════════════════════════════════════════════════════════════════════

  Figure 1: Zero-coupon spot curve  (z(T) vs T, 0→10Y)
  Figure 2: Forward rate curve      (3M forward rate vs T)
  Figure 3: Discount factor curve   (DF(T) vs T)
  Table  1: Input instruments vs curve-implied rates (validation)
  Table  2: Bootstrapped zero rates at key maturities

═══════════════════════════════════════════════════════════════════════════════
  STEP 8 — WRITE-UP STRUCTURE (for main.tex Section 3)
═══════════════════════════════════════════════════════════════════════════════

  3.1  Instruments and market data
       - Quote sources, valuation date, tenors used
  3.2  Bootstrapping methodology
       - Short-end (deposit) formula
       - Long-end (IRS par-swap equation)
       - Sequential single-curve solve
  3.3  Interpolation technique
       - Log-linear on DFs (piecewise-constant forwards)
       - Justification vs cubic spline / monotone convex
  3.4  Results
       - Spot curve chart + forward curve chart
       - Validation table
  3.5  Discussion
       - Shape interpretation (normal/inverted/humped)
       - Implications for bond pricing in Q4/Q5

═══════════════════════════════════════════════════════════════════════════════
  KEY CITATIONS
═══════════════════════════════════════════════════════════════════════════════

  [HaganWest2006]        Hagan & West, "Interpolation Methods for Curve
                         Construction", Applied Mathematical Finance 13(2),
                         2006. → authoritative treatment of interpolation;
                         compares log-linear, cubic spline, Monotone Convex.

  [Choudhry2010]         Choudhry, "Fixed-Income Securities and Derivatives
                         Handbook", 2010. → practical bootstrapping from
                         deposits and swaps; day-count conventions.

  [Hull2022]             Hull, "Options, Futures, and Other Derivatives",
                         11th ed., 2022. → accessible derivation of swap
                         bootstrapping formula.

  [Tuckman2011]          Tuckman & Serrat, "Fixed Income Securities", 3rd
                         ed., 2011. → term structure construction chapter.

  [ProfHandbook]         Choudhry et al., "The Professional Handbook of
                         Financial Risk Management – The Yield Curve", 2000.
                         → foundational bootstrapping methodology with
                         deposits + swaps; interpolation survey.

  [ArtScience]           Ametrano & Bianchetti, "Everything You Always Wanted
                         to Know About Multiple Interest Rate Curve Bootstrapping
                         but Were Afraid to Ask", 2013. → modern multi-curve
                         context; single-curve as special case.

  [AndersonPiterbarg2010] Andersen & Piterbarg, "Interest Rate Modeling",
                          Vol.1, 2010. → rigorous discount factor derivation.

═══════════════════════════════════════════════════════════════════════════════
  PYTHON IMPLEMENTATION FILES (to be created in Q3/)
═══════════════════════════════════════════════════════════════════════════════

  q3_bootstrap.py          → Main script: loads data, builds QL helpers,
                              bootstraps curve, validates, exports charts/tables
  q3_inputs.py (optional)  → Separate file for market quote constants
  output/
    q3_zero_curve.png      → Spot rate curve figure
    q3_forward_curve.png   → Forward rate curve figure
    q3_discount_curve.png  → Discount factor curve figure
    q3_validation.png      → Input vs implied rates table (PNG for LaTeX)
    q3_zero_rates.csv      → Zero rates at all maturities (for reference)

═══════════════════════════════════════════════════════════════════════════════
  QUANTLIB RECIPE (pseudocode)
═══════════════════════════════════════════════════════════════════════════════

  settlement_date = ql.Date(11, 3, 2026)   # T+2 from 9 Mar 2026
  ql.Settings.instance().evaluationDate = ql.Date(9, 3, 2026)

  # --- deposits ---
  deposit_helpers = [
      ql.DepositRateHelper(ql.QuoteHandle(ql.SimpleQuote(r/100)),
                           ql.Period(tenor),
                           fixing_days, ql.TARGET(),
                           ql.ModifiedFollowing, True,
                           ql.Actual360())
      for tenor, r in euribor_deposits.items()
  ]

  # --- swaps ---
  euribor3m = ql.Euribor3M()
  swap_helpers = [
      ql.SwapRateHelper(ql.QuoteHandle(ql.SimpleQuote(r/100)),
                        ql.Period(tenor),
                        ql.TARGET(), ql.Annual,
                        ql.ModifiedFollowing,
                        ql.Thirty360(ql.Thirty360.BondBasis),
                        euribor3m)
      for tenor, r in irs_rates.items()
  ]

  # --- bootstrap ---
  helpers = deposit_helpers + swap_helpers
  curve = ql.PiecewiseLogLinearDiscount(
              settlement_date, helpers,
              ql.Actual365Fixed())
  curve.enableExtrapolation()

  # --- query ---
  df   = curve.discount(ql.Date(9, 3, 2036))   # 10Y discount factor
  zero = curve.zeroRate(10.0, ql.Continuous, ql.Annual).rate()

"""

# ── extracted passages writer ─────────────────────────────────────────────────

def write_output(results: list[dict]) -> None:
    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    lines = []

    lines.append(PLAN.format(date=date.today().isoformat()))

    lines.append("\n" + "═" * 79)
    lines.append("  EXTRACTED PASSAGES FROM REFERENCE PDFs")
    lines.append("═" * 79)

    for r in results:
        name = r["path"].name if r["path"].exists() else r["label"]
        lines.append(f"\n{'─'*79}")
        lines.append(f"  SOURCE: {name}  [{r['n_pages']} pages, {r['n_chars']:,} chars]")
        lines.append(f"{'─'*79}")

        if not r["path"].exists():
            lines.append("  [PDF NOT FOUND – skipping]")
            continue

        if r["headings"]:
            lines.append("\n  SECTION HEADINGS DETECTED:")
            for h in r["headings"][:15]:
                lines.append(f"    • {h}")

        for category, key in [
            ("BOOTSTRAPPING PASSAGES", "bootstrap"),
            ("INTERPOLATION PASSAGES", "interp"),
            ("YIELD CURVE PASSAGES",   "curves"),
            ("MODEL/METHODOLOGY PASSAGES", "models"),
        ]:
            passages = r[key]
            if passages:
                lines.append(f"\n  ── {category} ──────────────────────────")
                for i, p in enumerate(passages[:3], 1):
                    lines.append(f"\n  [{i}] ...{p[:500]}...")

    lines.append("\n" + "═" * 79)
    lines.append("  END OF PLAN")
    lines.append("═" * 79 + "\n")

    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"✓ Plan written to: {OUT_TXT}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Q3 Bootstrap Plan Generator")
    print("=" * 60)

    results = []
    for label, path in PDFS.items():
        if path.exists():
            print(f"  Reading: {path.name} ...", end=" ", flush=True)
            data = analyse_pdf(label, path)
            print(f"{data['n_pages']} pages, {data['n_chars']:,} chars")
        else:
            print(f"  MISSING: {path}")
            data = {"label": label, "path": path, "n_pages": 0, "n_chars": 0,
                    "text": "", "headings": [], "bootstrap": [], "interp": [],
                    "curves": [], "models": []}
        results.append(data)

    write_output(results)

    print("\n" + "=" * 60)
    print("OUTPUT FILES")
    print("=" * 60)
    print(f"  Plan file : {OUT_TXT}")
    print("\nNext steps:")
    print("  1. Review q3_bootstrap_plan.txt for the full plan")
    print("  2. Replace indicative market rates with actual quotes")
    print("  3. Create Pricing_Structured_Product/Q3/q3_bootstrap.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
