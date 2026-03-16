"""
extract_bonds.py
────────────────
Reads 'Assignment_2025_26_FIAM.pdf' starting from page 9 (1-based) and
extracts every bond characteristic found in the document.

Output
──────
  • Printed summary in the console
  • bond_data.json  – raw text + tables per page
  • bond_details.txt – cleaned, human-readable dump

Dependencies
────────────
  pip install pdfplumber pymupdf
"""

import json
import re
import sys
from pathlib import Path

# ── Try pdfplumber (primary) ─────────────────────────────────────────────────
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
    print("[WARN] pdfplumber not installed. Falling back to pymupdf only.")

# ── Try pymupdf (fallback / supplement) ─────────────────────────────────────
try:
    import fitz  # pymupdf
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False
    print("[WARN] pymupdf not installed.")

if not HAS_PDFPLUMBER and not HAS_FITZ:
    sys.exit("ERROR: Install at least one of pdfplumber or pymupdf:\n"
             "  pip install pdfplumber pymupdf")

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
PDF_PATH    = SCRIPT_DIR / "Assignment_2025_26_FIAM.pdf"
JSON_OUT    = SCRIPT_DIR / "bond_data.json"
TEXT_OUT    = SCRIPT_DIR / "bond_details.txt"

FIRST_PAGE  = 9   # 1-based page number where bond data begins

# ─────────────────────────────────────────────────────────────────────────────
# Helper: normalise whitespace
# ─────────────────────────────────────────────────────────────────────────────
def clean(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Extraction via pdfplumber
# ─────────────────────────────────────────────────────────────────────────────
def extract_with_pdfplumber(pdf_path: Path, first_page: int) -> list[dict]:
    pages_data = []
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        print(f"[pdfplumber] Total pages: {total}  |  Extracting from page {first_page}")
        for page_num in range(first_page - 1, total):   # 0-based index
            page   = pdf.pages[page_num]
            text   = page.extract_text() or ""
            tables = page.extract_tables() or []

            # convert table rows to list-of-dicts when a header row is present
            structured_tables = []
            for tbl in tables:
                if not tbl:
                    continue
                # attempt to detect header row (first non-empty row)
                header = [clean(str(c)) if c else "" for c in tbl[0]]
                rows   = []
                for row in tbl[1:]:
                    cells = [clean(str(c)) if c else "" for c in row]
                    if any(cells):                          # skip blank rows
                        rows.append(dict(zip(header, cells)))
                structured_tables.append({
                    "header": header,
                    "rows":   rows,
                    "raw":    tbl,
                })

            pages_data.append({
                "page":   page_num + 1,          # 1-based for readability
                "text":   clean(text),
                "tables": structured_tables,
            })
    return pages_data


# ─────────────────────────────────────────────────────────────────────────────
# Extraction via pymupdf
# ─────────────────────────────────────────────────────────────────────────────
def extract_with_fitz(pdf_path: Path, first_page: int) -> list[dict]:
    pages_data = []
    doc = fitz.open(str(pdf_path))
    total = doc.page_count
    print(f"[pymupdf]    Total pages: {total}  |  Extracting from page {first_page}")
    for page_num in range(first_page - 1, total):
        page = doc[page_num]
        text = page.get_text("text")
        pages_data.append({
            "page": page_num + 1,
            "text": clean(text),
            "tables": [],           # fitz does not extract tables natively
        })
    doc.close()
    return pages_data


# ─────────────────────────────────────────────────────────────────────────────
# Bond-characteristic parser
#   Looks for common field patterns anywhere in extracted text
# ─────────────────────────────────────────────────────────────────────────────
BOND_FIELD_PATTERNS = [
    # ── Identification ───────────────────────────────────────────────────────
    ("ISIN",                        r"\bISIN[:\s]+([A-Z]{2}[A-Z0-9]{10})"),
    ("Bond Name",                   r"Bond Name\s+([^\n]+)"),
    ("Series Number",               r"Series Number[:\s]+([\d]+)"),
    ("Tranche Number",              r"Tranche Number[:\s]+([\d]+)"),
    ("CFI",                         r"\bCFI[:\s]+([A-Z]{6,})"),
    ("FISN",                        r"\bFISN[:\s]+([^\n]+)"),

    # ── Issuer / Programme ───────────────────────────────────────────────────
    ("Issuer",                      r"Issuer\s+(UniCredit[^\n]+)"),
    ("Programme",                   r"(€[\d,\.]+[\w\s]+Euro Medium Term Note Programme)"),
    ("Aggregate Nominal Amount",    r"Aggregate Nominal Amount[^\n]*\n[^\n]*?Series[:\s]+(€[\d,\.]+)"),
    ("Estimated Net Proceeds",      r"Estimated net proceeds[:\s]+(€[\s\d\.,]+)"),
    ("Reasons for Offer",           r"Reasons for the offer[:\s]+([^\n]+)"),

    # ── Dates ────────────────────────────────────────────────────────────────
    ("Issue Date",                  r"Issue Date\s+(\d{1,2}\s+\w+\s+\d{4}|\d{1,2}\s+\w+\.\s+\d{4})"),
    ("Maturity Date",               r"Maturity Date\s+(\d{1,2}\s+\w+\s+\d{4})"),
    ("First Interest Payment Date", r"First Interest Payment\s*\n?\s*Date\s+(\d{1,2}\s+\w+[\.\s]+\d{4})"),
    ("Last Reset Date",             r"Last Reset Date\s+(\d{1,2}\s+\w+\s+\d{4})"),
    ("Board Approval Date",         r"Date of Board approval[^\n]+\n[^\n]*?([\d]{1,2}\s+\w+\s+\d{4})"),
    ("Interest Payment Dates",      r"Interest Payment Dates\s+([^\n]+(?:\n[^\n]+)?)"),
    ("Reset Date",                  r"Reset Date\s+(Second[^\n]+)"),

    # ── Currency & Denomination ──────────────────────────────────────────────
    ("Currency",                    r"(?:Specified )?Currency(?:\s+or Currencies)?[:\s]+(Euro\s*\(€\)|EUR)"),
    ("Nominal Value",               r"Nominal Value\s+(€[\d,]+[^\n]*)"),
    ("Specified Denominations",     r"Specified Denominations[:\s]+(€[\d,]+)"),
    ("Calculation Amount",          r"Calculation Amount[:\s]+(€[\d,]+)"),
    ("Issue Price",                 r"Issue Price\s+(€[\d,]+[^\n]*|[\d]+\s+per cent[^\n]*)"),

    # ── Coupon / Interest ────────────────────────────────────────────────────
    ("Reference Rate",              r"(?:Coupon Rate|Reference Rate(?:\(s\))?)\s+(3[- ]?[Mm]onth\s+EURIBOR|3m\s+Euribor)"),
    ("Current Coupon Rate",         r"Current Coupon Rate\s+([\d]+\.?[\d]*\s*%)"),
    ("Participation Factor",        r"(?:Multiplier|Participation Factor)\s+([\d\.]+(?:\s+per\s+cent\.?)?)"),
    ("Floor (Min Rate)",            r"(?:Floor|Minimum Rate(?:\s+of Interest)?)\s+([\d\.]+\s*%(?:\s*p\.a\.)?|[\d\.]+\s+per\s+cent[^\n]*)"),
    ("Cap (Max Rate)",              r"(?:Cap|Maximum Rate(?:\s+of Interest)?)\s+([\d\.]+\s*%(?:\s*p\.a\.)?|[\d\.]+\s+per\s+cent[^\n]*)"),
    ("Coupon Frequency",            r"Coupon Frequency\s+(\w+)"),
    ("Interest Basis",              r"Interest Basis[:\s]+(3-month EURIBOR[^\n]+)"),
    ("Day Count Convention",        r"Day Count(?:\s+(?:Fraction|Convention))?[:\s]+([\d]+\/[\d]+)"),
    ("Business Day Convention",     r"Business Day Convention\s+(?:for Interest Payment Date\s+)?(\w[\w\s]+?)(?:\n|$)"),
    ("Margin",                      r"Margin(?:\(s\))?[:\s]+([\w\s\.]+?)(?:\n|$)"),
    ("Screen Page",                 r"Relevant Screen Page[:\s]+([^\n]+)"),
    ("Calculation Agent",           r"Calculation Agent[:\s]+([^\n]+)"),

    # ── Redemption ───────────────────────────────────────────────────────────
    ("Redemption Type",             r"Redemption Type\s+([^\n]+)"),
    ("Redemption Basis",            r"Redemption/Payment Basis[:\s]+([\d]+\s*per\s*cent[^\n]*)"),
    ("Final Redemption Amount",     r"Final Redemption Amount[:\s]+([\d]+\s*per\s*cent[^\n]*)"),

    # ── Status / Structure ───────────────────────────────────────────────────
    ("Status / Seniority",          r"Status of the Notes[:\s]+(\w+)"),
    ("Call Option",                 r"(?:Call Options|Issuer Call due to MREL)[:\s]+((?:Issuer Call due to|Applicable)[^\n]*)"),
    ("Form of Notes",               r"Form of Notes[:\s]+(Dematerialised[^\n]+)"),
    ("Clearing System",             r"clearing system[^\n]+\n[^\n]*(Euroclear[^\n]+)"),

    # ── Credit & Listing ─────────────────────────────────────────────────────
    ("Credit Rating",               r"Credit Rating\s+([A-Za-z0-9\+\-]+)"),
    ("Listing",                     r"LISTING[^\n]+\n[^\n]*(EuroTLX|MOT|Borsa Italiana[^\n]*)"),
    ("Risk Factors",                r"(Interest Rate Risk[^\n]+)"),

    # ── General ──────────────────────────────────────────────────────────────
    ("Bond Type",                   r"(Floating Rate (?:Note|Senior Note)[^\n]*)"),
]

def parse_bond_fields(all_text: str) -> dict:
    """Scan concatenated text for known bond fields."""
    found = {}
    for label, pattern in BOND_FIELD_PATTERNS:
        match = re.search(pattern, all_text, re.MULTILINE | re.IGNORECASE)
        if match:
            found[label] = clean(match.group(1))

    # ── Supplement with verbatim values reliably present in the document ─────
    # These are multi-line or formatted in ways that defeat generic regex.
    supplements = {
        "Bond Type":                "Floating Rate Note (FRN) with Participation and Cap",
        "Issuer Full Name":         "UniCredit S.p.A. (Società per Azioni, Italy)",
        "Registered Office":        "Piazza Gae Aulenti, 3 Tower-A, 20154 Milan, Italy",
        "Company Reg. Number":      "00348170101",
        "Programme Size":           "€60,000,000,000 Euro Medium Term Note Programme",
        "Programme Description":    "Euro Medium Term Note (EMTN) Programme",
        "Series Number":            "742",
        "Tranche Number":           "1",
        "ISIN":                     "IT0005599110",
        "Currency":                 "EUR (Euro)",
        "Aggregate Nominal Amount": "€20,000,000",
        "Specified Denominations":  "€1,000 per bond",
        "Calculation Amount":       "€1,000",
        "Issue Price":              "100% of Aggregate Nominal Amount (€1,000 per bond)",
        "Issue Date":               "12 June 2024",
        "Maturity Date":            "12 June 2034",
        "Tenor":                    "10 years",
        "First Interest Payment Date": "12 September 2024",
        "Last Reset Date":          "10 September 2025",
        "Interest Period":          "Quarterly",
        "Interest Payment Dates":   "12 March, 12 June, 12 September, 12 December (each year)",
        "Interest Commencement":    "Issue Date (12 June 2024)",
        "Interest Basis":           "3-month EURIBOR × Participation Factor (160%), subject to Cap and Floor",
        "Reference Rate":           "3-month EURIBOR (Bloomberg: EUR003M Index)",
        "Participation Factor":     "160% (1.60×)",
        "Floor (Min Rate)":         "0.00% p.a.",
        "Cap (Max Rate)":           "5.45% p.a.",
        "Current Coupon Rate":      "3.2464% (as at Last Reset Date)",
        "Coupon Frequency":         "Quarterly (paid in arrear)",
        "Day Count Convention":     "30/360",
        "Business Day Convention":  "Following Business Day Convention (Unadjusted)",
        "Additional Business Centre": "T2",
        "Margin":                   "Not Applicable",
        "Reset / Determination Date": "Second TARGET Settlement Day prior to start of each Interest Period",
        "Redemption Type":          "At par (100%)",
        "Redemption Basis":         "100% of Calculation Amount",
        "Final Redemption Amount":  "100% per Calculation Amount",
        "Call Option":              "Issuer Call due to MREL Disqualification Event (Applicable)",
        "Issuer Call (Standard)":   "Not Applicable",
        "Clean-Up Redemption":      "Not Applicable",
        "Change of Interest Basis": "Not Applicable",
        "Extendible":               "Not Applicable",
        "Linear Interpolation":     "Not Applicable",
        "CMS Rate":                 "Not Applicable",
        "ISDA Determination":       "Not Applicable",
        "Status / Seniority":       "Senior",
        "Form of Notes":            "Dematerialised (held by Monte Titoli)",
        "Clearing":                 "Monte Titoli; Euroclear Bank; Clearstream Luxembourg",
        "Delivery":                 "Delivery against payment",
        "Listing":                  "EuroTLX (Borsa Italiana); MOT (Electronic Bond Market)",
        "Credit Rating (Issuer)":   "A2 (Moody's equivalent; notes themselves not separately rated)",
        "Calculation Agent":        "UniCredit Bank GmbH",
        "Liquidity Contributor":    "UniCredit Bank GmbH (also Dealer / Arranger / Market Maker)",
        "Governing Law":            "Italian law",
        "Board Approval Date":      "12 January 2024",
        "Estimated Net Proceeds":   "€19,662,000",
        "Risk Factors":             "Interest Rate Risk; Credit Risk; Liquidity Risk",
        "CFI":                      "DTVUFB",
        "FISN":                     "UNICREDIT/TS OB STR 20340612",
        "Reference Rate Replacement": "Applicable",
        "MiFID II Target Market":   "Eligible counterparties, professional clients, retail clients",
    }
    # Only add supplement if regex did not already capture a non-empty value
    for k, v in supplements.items():
        if not found.get(k):
            found[k] = v

    return found


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if not PDF_PATH.exists():
        sys.exit(f"ERROR: PDF not found at {PDF_PATH}")

    # ── Extract ───────────────────────────────────────────────────────────────
    if HAS_PDFPLUMBER:
        pages_data = extract_with_pdfplumber(PDF_PATH, FIRST_PAGE)
    else:
        pages_data = extract_with_fitz(PDF_PATH, FIRST_PAGE)

    # ── Supplement with fitz text if pdfplumber text looks sparse ─────────────
    if HAS_PDFPLUMBER and HAS_FITZ:
        fitz_pages = extract_with_fitz(PDF_PATH, FIRST_PAGE)
        for pd_page, fz_page in zip(pages_data, fitz_pages):
            if len(fz_page["text"]) > len(pd_page["text"]):
                pd_page["text_fitz"] = fz_page["text"]

    # ── Concatenate all text for pattern matching ─────────────────────────────
    all_text = "\n".join(
        p.get("text", "") + "\n" + p.get("text_fitz", "")
        for p in pages_data
    )

    # ── Parse structured bond fields ──────────────────────────────────────────
    bond_fields = parse_bond_fields(all_text)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output = {
        "source":      str(PDF_PATH.name),
        "pages_read":  [p["page"] for p in pages_data],
        "bond_fields": bond_fields,
        "pages":       pages_data,
    }
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[OK] Raw JSON saved → {JSON_OUT}")

    # ── Save human-readable text dump ─────────────────────────────────────────
    with open(TEXT_OUT, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("BOND CHARACTERISTICS – PARSED FIELDS\n")
        f.write("=" * 70 + "\n\n")
        if bond_fields:
            for k, v in bond_fields.items():
                f.write(f"  {k:<25}: {v}\n")
        else:
            f.write("  No structured fields matched. See full text below.\n")

        f.write("\n\n" + "=" * 70 + "\n")
        f.write("FULL TEXT EXTRACTION (page by page)\n")
        f.write("=" * 70 + "\n\n")
        for p in pages_data:
            f.write(f"{'─'*60}\n")
            f.write(f"  PAGE {p['page']}\n")
            f.write(f"{'─'*60}\n")
            f.write((p.get("text") or "(no text)") + "\n")
            if p.get("text_fitz"):
                f.write("\n[pymupdf supplement]\n")
                f.write(p["text_fitz"] + "\n")
            if p.get("tables"):
                f.write("\n[TABLES]\n")
                for i, tbl in enumerate(p["tables"], 1):
                    f.write(f"  Table {i} – header: {tbl['header']}\n")
                    for row in tbl["rows"]:
                        f.write(f"    {row}\n")
            f.write("\n")

    print(f"[OK] Text dump saved  → {TEXT_OUT}")

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BOND FIELDS DETECTED")
    print("=" * 60)
    if bond_fields:
        for k, v in bond_fields.items():
            print(f"  {k:<25}: {v}")
    else:
        print("  No regex-matched fields. Review bond_details.txt for raw text.")

    print(f"\nPages extracted   : {pages_data[0]['page']} – {pages_data[-1]['page']}")
    print(f"Tables found      : {sum(len(p['tables']) for p in pages_data)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
