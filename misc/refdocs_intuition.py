"""
refdocs_intuition.py
====================
Reads every PDF under misc/refdocs/ (recursively) and produces:
    misc/refdocs/refdocs_intuition.txt

For each PDF the script reports:
  - File metadata (path, size, page count)
  - First 400 words (abstract/intro)
  - Top 20 content keywords (excluding common stopwords)
  - Detected section headings
  - Relevance tags for bond pricing / this project
"""

import sys, io, re, pathlib, os
from collections import Counter
from datetime import date

import pdfplumber

# ── paths ────────────────────────────────────────────────────────────────────
REFDOCS = pathlib.Path(__file__).parent / "refdocs"
OUT_TXT  = REFDOCS / "refdocs_intuition.txt"

# ── stopwords ────────────────────────────────────────────────────────────────
STOPWORDS = {
    "the","a","an","and","or","but","in","on","at","to","of","for",
    "is","are","was","were","be","been","being","have","has","had",
    "do","does","did","will","would","could","should","may","might",
    "shall","can","it","its","this","that","these","those","from",
    "by","with","as","if","not","no","so","we","our","us","they",
    "their","them","he","she","his","her","i","you","my","your",
    "which","who","whom","what","when","where","how","all","any",
    "such","more","also","than","then","into","after","before",
    "each","between","about","per","based","using","used","use",
    "give","given","see","shown","show","above","below","since",
    "let","set","well","two","three","one","first","second","third",
    "under","over","note","notes","section","page","figure","table",
    "fig","eq","ref","e\\.g","i\\.e","etc","via","both","only",
    "either","however","therefore","thus","hence","further","while",
    "although","whereas","among","within","without","during","through",
    "following","here","there","very","other","new","different","same",
    "general","paper","result","results","method","methods","approach",
    "model","models","using","can","there","while","however",
}

def clean_text(raw: str) -> str:
    """Remove control chars and collapse whitespace."""
    text = re.sub(r"\x00", " ", raw)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def top_keywords(text: str, n: int = 20) -> list:
    """Return top-n content words (lower, alpha >=4 chars, not stopword)."""
    words = re.findall(r"[a-zA-Z]{4,}", text.lower())
    words = [w for w in words if w not in STOPWORDS and not w.isdigit()]
    return Counter(words).most_common(n)

def detect_headings(text: str, max_h: int = 12) -> list:
    """Heuristic: lines that are short, capitalised, and possibly numbered."""
    headings = []
    for line in text.splitlines():
        line = line.strip()
        if not line or len(line) > 120:
            continue
        # numbered heading: starts with digit(s) + dot pattern, or ALL-CAPS short line
        if re.match(r"^\d[\d\.]*\s+\w", line) and len(line) < 90:
            headings.append(line)
        elif line.isupper() and 5 < len(line) < 80:
            headings.append(line)
        elif re.match(r"^[A-Z][A-Z\s\-]{8,60}$", line):
            headings.append(line)
        if len(headings) >= max_h:
            break
    return headings

def relevance_tags(text_lower: str) -> list:
    """Assign relevance tags based on keyword presence."""
    tag_map = {
        "EUR IRS / Curve Bootstrap": ["bootstrapp", "swap curve", "irs curve", "yield curve"],
        "EURIBOR / LIBOR fixing"   : ["euribor", "libor", "ibor", "fixing", "benchmark rate"],
        "Cap/Floor / Swaption"     : ["caplet", "floorlet", "cap floor", "swaption", "optionlet"],
        "Vol Surface / Smile"      : ["volatility surface", "smile", "skew", "sabr", "bachelier",
                                      "black-76", "displaced"],
        "Day Count / Conventions"  : ["day count", "act/360", "30/360", "actual", "accrued",
                                      "convention"],
        "Bloomberg data"           : ["bloomberg", "dsgrid", "icvs"],
        "Discounting / DCF"        : ["discount factor", "zero coupon", "zero rate", "discount curve"],
        "Term Structure Interpolation": ["interpolat", "cubic spline", "monotone convex",
                                         "max smoothness", "linear logdf"],
    }
    tags = []
    for tag, keywords in tag_map.items():
        if any(kw in text_lower for kw in keywords):
            tags.append(tag)
    return tags if tags else ["General reference"]

# ── collect PDFs ─────────────────────────────────────────────────────────────
pdf_paths = sorted(REFDOCS.rglob("*.pdf"))

# ── buffer for output ─────────────────────────────────────────────────────────
buf = io.StringIO()

def pr(*args, **kw):
    msg = " ".join(str(a) for a in args)
    print(msg, **kw)
    buf.write(msg + "\n")

def hl(c="=", n=72):
    return c * n

# ── header ────────────────────────────────────────────────────────────────────
pr(hl())
pr("  REFDOCS INTUITION  –  misc/refdocs/")
pr(f"  Analysis date  : {date.today()}")
pr(f"  PDFs found     : {len(pdf_paths)}")
pr(hl())
pr()

for idx, pdf_path in enumerate(pdf_paths, start=1):
    rel = pdf_path.relative_to(REFDOCS)
    size_kb = pdf_path.stat().st_size / 1024

    pr(f"\n{'#'*72}")
    pr(f"  PDF {idx:02d}/{len(pdf_paths)}  |  {rel}")
    pr(f"  File size : {size_kb:.1f} KB")
    pr("#" * 72)

    # -- Extract text ──────────────────────────────────────────────────────
    all_text   = ""
    page_count = 0
    errors     = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)
            for pg in pdf.pages:
                try:
                    t = pg.extract_text() or ""
                    all_text += t + "\n"
                except Exception as e:
                    errors.append(str(e))
    except Exception as e:
        errors.append(f"Could not open: {e}")

    all_text = clean_text(all_text)
    pr(f"  Pages         : {page_count}")
    pr(f"  Chars extracted: {len(all_text):,}")
    if errors:
        pr(f"  Extraction warnings: {'; '.join(errors[:3])}")

    if not all_text.strip():
        pr("  [WARN] No text extracted (possibly scanned / image-only PDF)")
        pr()
        continue

    # -- First ~400 words (intro) ──────────────────────────────────────────
    words_preview = all_text.split()
    preview_text  = " ".join(words_preview[:400])
    pr()
    pr("  -- FIRST 400 WORDS (abstract / introduction) " + "-" * 26)
    pr(f"  {preview_text}")

    # -- Detected headings ─────────────────────────────────────────────────
    headings = detect_headings(all_text, max_h=12)
    pr()
    pr("  -- DETECTED HEADINGS " + "-" * 49)
    if headings:
        for h in headings:
            pr(f"    > {h}")
    else:
        pr("    (none detected by heuristic)")

    # -- Top keywords ──────────────────────────────────────────────────────
    kws = top_keywords(all_text, n=20)
    pr()
    pr("  -- TOP 20 CONTENT KEYWORDS " + "-" * 43)
    kw_line = ", ".join(f"{w}({c})" for w, c in kws)
    # wrap at ~70 chars
    line = ""
    for chunk in kw_line.split(", "):
        if len(line) + len(chunk) > 68:
            pr(f"    {line.rstrip(', ')}")
            line = chunk + ", "
        else:
            line += chunk + ", "
    if line.strip().rstrip(","):
        pr(f"    {line.rstrip(', ')}")

    # -- Relevance tags ────────────────────────────────────────────────────
    tags = relevance_tags(all_text.lower())
    pr()
    pr("  -- RELEVANCE TAGS " + "-" * 52)
    pr(f"    {', '.join(tags)}")

    # -- Summary (key observations) ────────────────────────────────────────
    pr()
    pr("  -- SUMMARY / KEY OBSERVATIONS " + "-" * 40)
    total_words = len(all_text.split())
    pr(f"    Word count     : ~{total_words:,}")
    pr(f"    Page density   : {total_words // max(page_count,1):,} words/page")

    # Content-aware summary based on top keywords and path
    folder = rel.parts[0] if len(rel.parts) > 1 else "."
    fname  = pdf_path.stem.lower()

    if "bloomberg" in folder.lower() or "bloomberg" in fname:
        pr("    Topic area     : Bloomberg terminal usage for building interest")
        pr("                     rate curves (ICVS function, curve settings,")
        pr("                     DSGRID data extraction, swap math).")

    if "caplet" in folder.lower() or "caplet" in fname or "smile" in fname:
        pr("    Topic area     : Volatility bootstrapping from cap/floor market")
        pr("                     prices to caplet/floorlet vols; smile interpolation")
        pr("                     methods (SABR, cubic spline) used after stripping.")

    if "libor" in folder.lower() or "euribor" in fname or "libor" in fname:
        pr("    Topic area     : IBOR benchmark rate mechanics, fixing methodology,")
        pr("                     panelled submission process, fallback provisions.")

    if "market" in folder.lower() or "convention" in fname or "accrued" in fname:
        pr("    Topic area     : Fixed income market conventions (day count, accrued")
        pr("                     interest, yield calculations, bill pricing, IR options).")

    if "bootstrapping" in folder.lower() or "jfi" in fname or "handbook" in fname.lower() \
       or "curvebuilding" in fname or "artandscience" in fname.lower():
        pr("    Topic area     : Yield curve construction methodologies (bootstrapping,")
        pr("                     interpolation, multi-curve framework, OIS discounting).")

    pr()

# -- Footer ────────────────────────────────────────────────────────────────────
pr(hl())
pr("  END OF REFDOCS INTUITION")
pr(hl())
pr()
pr("  How to use these references in the project:")
pr()
pr("  1. CURVE CONSTRUCTION (Questions 2 & 4):")
pr("       Bootstrap the EUR EURIBOR/OIS curve using IRS CURVE data.")
pr("       Key refs: 'TheArtAndScienceofCurveBuilding.pdf',")
pr("       'The Professional Handbook... Yield Curve.pdf',")
pr("       'jfi_1994_Max_Smoothness_v03.pdf' (max-smoothness interpolation).")
pr("       Bloomberg: 'building_the_bloomberg_irc.pdf',")
pr("       'Bloomberg Recommended Settings for Interest Rate Curve.pdf'.")
pr()
pr("  2. CAP/FLOOR PRICING (Question 3):")
pr("       Understand vol surface structure, caplet stripping, and smile.")
pr("       Key refs: 'caplet-stripping-opengamma.pdf',")
pr("       'SmileInterpolation_OpenGamma.pdf'.")
pr("       Model reference: Bachelier sheet in MarketData202526.xlsx.")
pr()
pr("  3. CONVENTIONS (throughout):")
pr("       Day count (30/360, ACT/360, ACT/ACT): Market Conventions folder.")
pr("       Key refs: 'interest-rate-instruments-and-market-conventions.pdf',")
pr("       'interest rate options conventions.pdf', 'ACT-ACT-ISDA-1999.pdf').")
pr()
pr("  4. EURIBOR/LIBOR REFORM awareness:")
pr("       Bond has 'Reference Rate Replacement: Applicable' clause.")
pr("       Key refs: 'Euribor FAQs Final.pdf', 'IBA_LIBOR_FAQ.pdf'.")
pr(hl())

# ── Write file ─────────────────────────────────────────────────────────────────
with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write(buf.getvalue())

print(f"\n[OK] Intuition file saved -> {OUT_TXT}")
print(f"[OK] PDFs processed       : {len(pdf_paths)}")
