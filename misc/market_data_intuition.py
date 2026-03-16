"""
market_data_intuition.py
========================
Reads MarketData202526.xlsx from ../input_data/ and produces:
  1. Console + text output  ->  misc/MarketData_intuition.txt
  2. Matplotlib charts       ->  misc/MD_0*.png

Sheets analysed:
  Refinitiv Code   – reference tickers
  IRS CURVE        – EUR spot curve snapshot (5 Nov 2025)
  Hist IRS         – 10-year daily history of EUR IRS (1Y-50Y)
  Hist Euribor     – 10-year daily history of EURIBOR (ON-12M)
  Bachelier        – EUR Cap/Floor Normal vol surface (ICAP 5-Nov-25)
  Displaced Black  – EUR Cap/Floor Shifted Black vol surface (shift=3%)
"""

import sys, io, pathlib
from datetime import datetime, timedelta, date

import openpyxl
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# ── paths ───────────────────────────────────────────────────────────────────
BASE   = pathlib.Path(__file__).parent
XLSX   = BASE.parent / "input_data" / "MarketData202526.xlsx"
OUT    = BASE / "MarketData_intuition.txt"

# ── helpers ──────────────────────────────────────────────────────────────────
def xl_date(v):
    """Convert Excel serial integer *or* datetime object to date."""
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, (int, float)) and not pd.isna(v):
        return (datetime(1899, 12, 30) + timedelta(days=int(v))).date()
    return None

def fmt(v, decimals=4):
    """Format a numeric value cleanly."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "  N/A  "
    return f"{v:>{8}.{decimals}f}"

def hline(c="=", n=72):
    return c * n

def sec(title, c="="):
    return f"\n{hline(c)}\n  {title}\n{hline(c)}\n"

# ── capture output to both console and string buffer ─────────────────────────
buf = io.StringIO()

def pr(*args, **kw):
    msg = " ".join(str(a) for a in args)
    print(msg, **kw)
    buf.write(msg + "\n")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  LOAD WORKBOOK                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝
wb = openpyxl.load_workbook(XLSX, data_only=True)

pr(sec("MARKET DATA INTUITION  –  MarketData202526.xlsx"))
pr(f"  File      : {XLSX.name}")
pr(f"  Run date  : {date.today()}")
pr(f"  Sheets    : {', '.join(wb.sheetnames)}")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SHEET 1 – REFINITIV CODE                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
pr(sec("SHEET 1 – REFINITIV CODE (Reference Identifiers)"))
ws_ref = wb["Refinitiv Code"]
irs_codes_ref  = [r[0] for r in ws_ref.iter_rows(values_only=True)
                  if r[0] and str(r[0]).startswith("ICEIB")]
eibor_codes_ref = [r[0] for r in ws_ref.iter_rows(values_only=True)
                   if r[0] and str(r[0]).startswith("EIBOR")]
pr(f"  IRS tickers    ({len(irs_codes_ref):<2d}): {', '.join(irs_codes_ref)}")
pr(f"  EURIBOR tickers ({len(eibor_codes_ref):<2d}): {', '.join(eibor_codes_ref)}")
pr()
pr("  Interpretation:")
pr("    ICEIB* = EUR 30/360 6M EURIBOR Interest Rate Swap par rates")
pr("    EIBOR* = EBF EURIBOR fixings (1W, 1M, 3M, 6M, 12M)")
pr("    Both sourced via LSEG Refinitiv Datastream (DSGRID formula).")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SHEET 2 – IRS CURVE (Snapshot: 5 November 2025)                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
pr(sec("SHEET 2 – IRS CURVE  (Snapshot: 5 November 2025)"))
ws_irs = wb["IRS CURVE"]
rows_irs = list(ws_irs.iter_rows(values_only=True))

snap_date = rows_irs[0][0]
irs_snap  = {r[0]: r[1] for r in rows_irs[1:] if r[0] is not None and r[1] is not None}

pr(f"  Snapshot date : {snap_date.strftime('%d %b %Y') if isinstance(snap_date, datetime) else snap_date}")
pr()
pr("  -- SHORT END (Money Market) --")
short_keys = [k for k in irs_snap if "EURIBOR" in k.upper() or "DEPOSIT" in k.upper() or "O/N" in k]
for k, v in irs_snap.items():
    if "EURIBOR" in k.upper() or "DEPOSIT" in k.upper():
        marker = "  <-- key 3M coupon reset" if "3M" in k else ""
        pr(f"    {k:<45s}: {v:6.3f}%{marker}")

pr()
pr("  -- IRS CURVE (par swap rates, EUR 6M EURIBOR vs fixed) --")
irs_tenors = {}
for k, v in irs_snap.items():
    if "IRS" in k:
        label = k.split()[-1] if k.split() else k
        irs_tenors[label] = v
        pr(f"    {k:<45s}: {v:6.4f}%")

# Slope and curvature
def get_irs(label):
    return irs_snap.get(f"EUR {label} AB6E IRS")

v2y  = get_irs("2Y");  v5y  = get_irs("5Y");  v10y = get_irs("10Y")
v30y = get_irs("30Y"); v50y = get_irs("50Y")
slope_10_2  = (v10y - v2y)  * 100 if v10y and v2y  else None
slope_30_10 = (v30y - v10y) * 100 if v30y and v10y else None
slope_50_30 = (v50y - v30y) * 100 if v50y and v30y else None
curv        = (2 * v5y - v2y - v10y) * 100 if v2y and v5y and v10y else None

pr()
pr("  -- SHAPE & SLOPE ANALYSIS --")
pr(f"    Slope  10Y - 2Y   : {slope_10_2:+.1f} bps")
pr(f"    Slope  30Y - 10Y  : {slope_30_10:+.1f} bps")
pr(f"    Slope  50Y - 30Y  : {slope_50_30:+.1f} bps")
pr(f"    Curvature 2*5Y-(2Y+10Y): {curv:+.1f} bps")
pr()
pr("  -- INTERPRETATION --")
pr("    The EUR IRS curve (5-Nov-2025) is upward sloping in the belly")
pr("    (2Y-30Y), typical of a late rate-cutting-cycle environment.")
pr("    The ECB Deposit Facility rate was ~3.4% in late 2023 (peak) and")
pr("    has been cut steadily; O/N at 1.90% signals further cuts already")
pr("    delivered. The 3M EURIBOR (2.023%) sits above O/N but below the")
pr("    1M (1.914%) – the inversion at the very short end reflects market")
pr("    expectation of near-term ECB rate cuts embedded in forward rates.")
pr()
euribor_3m = irs_snap.get("EBF EURIBOR 3M DELAYED - OFFERED RATE")
cap_strike_euribor = 5.40 / 1.60
pr("  -- BOND RELEVANCE (UniCredit Var. Rate Bond 2034) --")
pr(f"    Current 3M EURIBOR    : {euribor_3m:.3f}%")
pr(f"    Cap break-even EURIBOR: {cap_strike_euribor:.3f}%  (= 5.40% / 1.60)")
pr(f"    Gap to cap            : {(cap_strike_euribor - euribor_3m)*100:+.1f} bps")
pr(f"    Current coupon (est.) : {1.60 * euribor_3m:.4f}%  (= 1.60 x {euribor_3m:.3f}%)")
pr("    => Bond is in LINEAR regime; cap not binding, floor not binding.")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SHEET 3 – HIST IRS                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
pr(sec("SHEET 3 – HIST IRS  (Historical EUR IRS rates, 10-yr panel)"))
ws_hist = wb["Hist IRS"]
hist_rows = list(ws_hist.iter_rows(values_only=True))

irs_names_h = [str(c) if c else "" for c in hist_rows[0][1:]]
irs_codes_h = [str(c) if c else "" for c in hist_rows[1][1:]]

data_h = []
for row in hist_rows[3:]:
    d = xl_date(row[0])
    if d is not None:
        data_h.append([d] + list(row[1:]))

df_irs = pd.DataFrame(data_h, columns=["Date"] + irs_codes_h).set_index("Date")
df_irs = df_irs.apply(pd.to_numeric, errors="coerce")

pr(f"  Date range    : {df_irs.index[0]} to {df_irs.index[-1]}")
pr(f"  Observations  : {len(df_irs)}")
pr(f"  Tenors        : {', '.join(irs_codes_h)}")
pr()

# Descriptive stats for selected tenors
sel_codes = ["ICEIB2Y", "ICEIB5Y", "ICEIB10", "ICEIB30"]
sel_labels = {"ICEIB2Y": "2Y IRS", "ICEIB5Y": "5Y IRS", "ICEIB10": "10Y IRS", "ICEIB30": "30Y IRS"}
pr("  -- DESCRIPTIVE STATISTICS (%)")
pr(f"  {'Tenor':<10} {'Min':>8} {'Max':>8} {'Mean':>8} {'Std':>8} {'Current':>10} {'%-ile':>8}")
pr("  " + "-" * 62)
for c in sel_codes:
    if c in df_irs.columns:
        s = df_irs[c].dropna()
        curr = s.iloc[-1]
        pct  = (s < curr).mean() * 100
        pr(f"  {sel_labels[c]:<10} {s.min():>8.3f} {s.max():>8.3f} {s.mean():>8.3f}"
           f" {s.std():>8.3f} {curr:>10.3f} {pct:>7.1f}%")

# Slope history
if "ICEIB2Y" in df_irs.columns and "ICEIB10" in df_irs.columns:
    slope_hist = (df_irs["ICEIB10"] - df_irs["ICEIB2Y"]) * 100
    pr()
    pr("  -- 10Y-2Y SLOPE HISTORY (bps)")
    pr(f"    Min       : {slope_hist.min():.1f} bps  ({slope_hist.idxmin()})")
    pr(f"    Max       : {slope_hist.max():.1f} bps  ({slope_hist.idxmax()})")
    pr(f"    Mean      : {slope_hist.mean():.1f} bps")
    pr(f"    Current   : {slope_hist.iloc[-1]:.1f} bps")

pr()
pr("  -- KEY OBSERVATIONS --")
pr("    Rates turned negative (1Y-10Y IRS) circa Jan 2016 and remained")
pr("    negative until mid-2022 as ECB operated NIRP (negative interest")
pr("    rate policy). The sharp rate hike from Jun 2022 rapidly pushed")
pr("    all tenors positive; 10Y IRS peaked ~3.4% in Oct 2023.")
pr("    Since then, ECB cuts have brought rates down; current 10Y at")
pr(f"    {df_irs['ICEIB10'].iloc[-1]:.3f}% (in the {(df_irs['ICEIB10'].dropna() < df_irs['ICEIB10'].dropna().iloc[-1]).mean()*100:.0f}th percentile of the 10Y history).")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SHEET 4 – HIST EURIBOR                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝
pr(sec("SHEET 4 – HIST EURIBOR  (Historical EURIBOR fixings)"))
ws_eur = wb["Hist Euribor"]
eur_rows = list(ws_eur.iter_rows(values_only=True))

eur_names_h = [str(c) if c else "" for c in eur_rows[0][1:7]]
eur_codes_h = [str(c) if c else "" for c in eur_rows[1][1:7]]

data_e = []
for row in eur_rows[3:]:
    d = xl_date(row[0])
    if d is not None:
        data_e.append([d] + list(row[1:7]))

df_eur = pd.DataFrame(data_e, columns=["Date"] + eur_codes_h).set_index("Date")
df_eur = df_eur.apply(pd.to_numeric, errors="coerce")

pr(f"  Date range    : {df_eur.index[0]} to {df_eur.index[-1]}")
pr(f"  Observations  : {len(df_eur)}")
pr(f"  Tenors        : {', '.join(eur_codes_h)}")
pr()

eur_label_map = {
    "TPEURON": "O/N", "EIBOR1W": "1W", "EIBOR1M": "1M",
    "EIBOR3M": "3M", "EIBOR6M": "6M", "EIBOR12": "12M"
}
pr("  -- DESCRIPTIVE STATISTICS (%)")
pr(f"  {'Tenor':<8} {'Min':>8} {'Max':>8} {'Mean':>8} {'Std':>8} {'Current':>10} {'%-ile':>8}")
pr("  " + "-" * 58)
for c in eur_codes_h:
    if c in df_eur.columns:
        s = df_eur[c].dropna()
        if len(s) == 0:
            continue
        curr = s.iloc[-1]
        pct  = (s < curr).mean() * 100
        lbl  = eur_label_map.get(c, c)
        pr(f"  {lbl:<8} {s.min():>8.3f} {s.max():>8.3f} {s.mean():>8.3f}"
           f" {s.std():>8.3f} {curr:>10.3f} {pct:>7.1f}%")

pr()
pr("  -- KEY OBSERVATIONS --")
pr("    EURIBOR 3M peaked at ~3.99% (Oct 2023) during the ECB rate hike")
pr("    cycle (largest and fastest hike cycle in ECB history). Minimum")
pr("    was ~-0.57% (2021) during NIRP era. Current 3M (2.023%) is")
pr("    above the 10Y mean (~0.60%) but much lower than the peak.")
pr("    The forward curve implies further ECB cuts: 1M EURIBOR (1.914%)")
pr("    is BELOW 3M (2.023%), signalling a rate-cut expectation over the")
pr("    next 2-3 months that is embedded in the forward rate structure.")
pr()
euribor_3m_curr = df_eur["EIBOR3M"].dropna().iloc[-1] if "EIBOR3M" in df_eur else euribor_3m
pr("  -- BOND RELEVANCE --")
pr(f"    The bond coupon resets quarterly to 1.60 x (3M EURIBOR).")
pr(f"    Current rate: 1.60 x {euribor_3m_curr:.3f}% = {1.60*euribor_3m_curr:.4f}% per annum.")
pr("    Historical EURIBOR highs (>3.375%) would have triggered the cap;")
pr("    in the NIRP era (neg EURIBOR) the floor (0%) would have bound.")
p_above_cap = (df_eur["EIBOR3M"].dropna() >= cap_strike_euribor).mean() * 100 if "EIBOR3M" in df_eur else 0
p_at_floor  = (df_eur["EIBOR3M"].dropna() <= 0).mean() * 100 if "EIBOR3M" in df_eur else 0
pr(f"    % of history where EURIBOR 3M >= cap break-even (3.375%): {p_above_cap:.1f}%")
pr(f"    % of history where EURIBOR 3M <= 0% (floor bound)       : {p_at_floor:.1f}%")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SHEET 5 – BACHELIER VOL SURFACE                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
pr(sec("SHEET 5 – BACHELIER VOL SURFACE  (Normal / Basis-Points)"))
ws_bach = wb["Bachelier"]
bach_rows = list(ws_bach.iter_rows(values_only=True))

bach_hdr  = bach_rows[4]   # (None, 'STK', 'ATM', -1.5, -1, 0, ...)
strike_lbls = [str(x) if x is not None else "" for x in bach_hdr[2:]]  # 'ATM', -1.5, -1, ...
strike_vals = [x for x in bach_hdr[2:]]

bach_records = []
for row in bach_rows[5:]:
    if row[0] and str(row[0]).strip():
        tenor = str(row[0]).strip()
        atm_strike = row[1]
        vals       = list(row[2:])
        bach_records.append({"tenor": tenor, "atm_strike": atm_strike,
                              **{str(strike_vals[i]): vals[i] for i in range(len(strike_vals))}})
df_bach = pd.DataFrame(bach_records)
df_bach.set_index("tenor", inplace=True)

pr(f"  Source : ICAP, 05-Nov-2025 (data stamp on file: 13:25)")
pr(f"  Vols   : Normal (Bachelier) model, units = basis points (bps)")
pr(f"  Tenors : {', '.join(df_bach.index.tolist())}")
pr(f"  Strikes: {', '.join(str(x) for x in bach_hdr[2:])} (%)")
pr()

pr("  -- ATM VOLATILITY TERM STRUCTURE (bps) --")
pr(f"  {'Tenor':<8} {'ATM Strike':>12} {'ATM Vol (bps)':>15}")
pr("  " + "-" * 38)
for tenor in df_bach.index:
    stk = df_bach.loc[tenor, "atm_strike"]
    vol = df_bach.loc[tenor, "ATM"]
    pr(f"  {tenor:<8} {stk if stk is not None else float('nan'):>11.2f}%  {vol if vol is not None else float('nan'):>12.2f} bps")

pr()
pr("  -- VOL SMILE (selected tenors, bps) --")
smile_tenors = ["1Y", "5Y", "10Y", "30Y"]
smile_cols   = [str(x) for x in strike_vals if x is not None]
header = f"  {'Strike':>8}  " + "  ".join(f"{t:>10}" for t in smile_tenors)
pr(header)
pr("  " + "-" * (12 + 12 * len(smile_tenors)))
for col in smile_cols[:13]:   # up to +10% strike
    row_vals = []
    for t in smile_tenors:
        t_key = t.strip()
        if t_key in df_bach.index:
            v = df_bach.loc[t_key, col]
            row_vals.append(f"{v if v is not None else float('nan'):>10.1f}")
        else:
            row_vals.append(f"{'N/A':>10}")
    pr(f"  {col:>8}  " + "  ".join(row_vals))

pr()
pr("  -- KEY OBSERVATIONS --")
pr("    1. ATM vol term structure is steeply upward-sloping in the front")
pr("       (1Y=27 bps to 10Y=65 bps), then flattens at 20Y-30Y (~66 bps).")
pr("    2. Smile is U-shaped (or 'smirk'): high vols for deep OTM puts")
pr("       (-1.5% strike) and rising vols for OTM calls (>ATM). This is")
pr("       typical for EUR caps/floors in post-NIRP environment where")
pr("       negative-rate risk is still priced at the short end.")
pr("    3. For the 10Y cap@5.40%: ATM for 10Y is ~2.63%; the cap strike")
pr("       is +2.77% OTM. Reading across the +3 column (closest to OTM):")
atm_10y = df_bach.loc["10Y", "ATM"] if "10Y" in df_bach.index else None
col_3pct = "3" if "3" in df_bach.columns else None
vol_10y_3pct = df_bach.loc["10Y", col_3pct] if col_3pct and "10Y" in df_bach.index else None
pr(f"       10Y Cap vol at +3%: {vol_10y_3pct} bps (ATM: {atm_10y} bps)")
pr("    4. For the 10Y floor@0%: ATM is 2.63%; floor strike is -2.63% OTM.")
pr("       Reading the -1.5% col (nearest): floor vol is higher than ATM")
pr("       due to residual tail risk from the negative rate era.")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SHEET 6 – DISPLACED BLACK VOL SURFACE                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝
pr(sec("SHEET 6 – DISPLACED BLACK VOL SURFACE  (Shifted Black, shift=3%)"))
ws_db = wb["Displaced Black"]
db_rows = list(ws_db.iter_rows(values_only=True))

db_hdr     = db_rows[4]
db_s_vals  = [x for x in db_hdr[2:]]
db_records = []
for row in db_rows[5:]:
    if row[0] and str(row[0]).strip() and "Shift" not in str(row[0]):
        tenor = str(row[0]).strip()
        atm_stk = row[1]
        vals = list(row[2:])
        db_records.append({"tenor": tenor, "atm_strike": atm_stk,
                            **{str(db_s_vals[i]): vals[i] for i in range(len(db_s_vals))}})
df_db = pd.DataFrame(db_records)
df_db.set_index("tenor", inplace=True)

pr(f"  Source : ICAP, 05-Nov-2025 (data stamp: 13:33)")
pr(f"  Vols   : Shifted Black-76 model, units = % (annualised lognormal)")
pr(f"  Shift  : 3.00% (applied to all tenors, 1Y/18M/2Y use 3M; 3Y+ use 6M)")
pr(f"  Tenors : {', '.join(df_db.index.tolist())}")
pr()

pr("  -- ATM VOL COMPARISON: Bachelier (bps) vs Displaced Black (%) --")
pr(f"  {'Tenor':<8} {'ATM Strike':>12} {'Bachelier ATM (bps)':>22} {'Displ. Black ATM (%)':>22}")
pr("  " + "-" * 68)
for t in df_db.index:
    stk  = df_db.loc[t, "atm_strike"]
    dbv  = df_db.loc[t, "ATM"]
    bt   = t.strip()
    bacv = df_bach.loc[bt, "ATM"] if bt in df_bach.index else float("nan")
    pr(f"  {t:<8} {stk if stk else float('nan'):>11.2f}%  {bacv if bacv is not None else float('nan'):>18.2f} bps"
       f"  {dbv if dbv is not None else float('nan'):>18.2f}%")

pr()
pr("  -- KEY OBSERVATIONS --")
pr("    1. Shifted Black shift=3% means the model treats forward rate as")
pr("       (F + 3%), allowing lognormal vol to handle near-zero or")
pr("       negative rates without arbitrage.")
pr("    2. Displaced Black ATM vols (~5.5-11.4%) are much smaller numbers")
pr("       than Bachelier bps vols (27-66 bps). Both describe the same")
pr("       market prices; conversion: sigma_N ~ sigma_SB * (F + shift).")
pr("       E.g., 10Y: sigma_SB=11.43%, F+shift=2.633%+3%=5.633%")
pr(f"       => approx Bachelier vol = {11.43 * 5.633:.1f} bps  (market: {atm_10y} bps)")
pr("    3. Smile shape: Displaced Black shows higher vols at deep OTM puts")
pr("       (e.g., -1.5% strike) and elevated at OTM calls, consistent with")
pr("       Bachelier surface (both pricing the same caps/floors).")
pr("    4. Long-end (20Y-30Y) smile is notably flatter than short end,")
pr("       reflecting reduced uncertainty about the long-run rate level.")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CHARTS                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝
pr(sec("FIGURES"))

# -- Chart 1: Spot IRS Curve --------------------------------------------------
fig1, axes = plt.subplots(1, 2, figsize=(14, 5))
fig1.suptitle("EUR IRS Curve – Snapshot 5 November 2025", fontsize=12, y=0.98)

# Full curve
snap_items = [(lbl, rate) for lbl, rate in irs_snap.items() if rate is not None]
labels_full  = [x[0].replace("EUR CASH DEPOSIT ", "O/N").replace(" DELAYED - OFFERED RATE", "")
                     .replace(" AB6E IRS", "").replace("EBF EURIBOR ", "EURIBOR ")
                for x in snap_items]
rates_full   = [x[1] for x in snap_items]
ax = axes[0]
ax.plot(range(len(rates_full)), rates_full, "o-", color="steelblue", linewidth=2, markersize=4)
ax.set_xticks(range(len(labels_full)))
ax.set_xticklabels(labels_full, rotation=45, ha="right", fontsize=7)
ax.set_ylabel("Rate (%)")
ax.set_title("Full spot curve (MM + IRS)")
ax.grid(True, alpha=0.3)
ax.axhline(y=irs_snap.get("EBF EURIBOR 3M DELAYED - OFFERED RATE", 2.0),
           color="red", linestyle="--", linewidth=1, label="EURIBOR 3M (bond reset)")
ax.legend(fontsize=8)

# IRS only (tenors in years)
irs_only = {k: v for k, v in irs_snap.items() if "IRS" in k}
tenor_map = {"1Y": 1, "18M": 1.5, "2Y": 2, "3Y": 3, "4Y": 4, "5Y": 5, "6Y": 6,
             "7Y": 7, "8Y": 8, "9Y": 9, "10Y": 10, "12Y": 12, "15Y": 15,
             "20Y": 20, "25Y": 25, "30Y": 30, "40Y": 40, "50Y": 50}
yrs  = []
rts  = []
for lbl_key, rate in irs_only.items():
    for t_str, t_yrs in tenor_map.items():
        if lbl_key.endswith(t_str + " AB6E IRS"):
            yrs.append(t_yrs)
            rts.append(rate)
            break
yrs_sorted = sorted(zip(yrs, rts))
yrs_arr = [x[0] for x in yrs_sorted]
rts_arr = [x[1] for x in yrs_sorted]

ax2 = axes[1]
ax2.plot(yrs_arr, rts_arr, "o-", color="navy", linewidth=2, markersize=5)
ax2.axhline(y=cap_strike_euribor, color="red",    linestyle="--", linewidth=1,
            label=f"Cap break-even EURIBOR ({cap_strike_euribor:.3f}%)")
ax2.axhline(y=euribor_3m,         color="orange", linestyle="--", linewidth=1,
            label=f"Current EURIBOR 3M ({euribor_3m:.3f}%)")
ax2.set_xlabel("Tenor (years)")
ax2.set_ylabel("Par swap rate (%)")
ax2.set_title("EUR IRS par curve (by tenor)")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

fig1.tight_layout(rect=[0, 0, 1, 0.94])
p1 = BASE / "MD_01_spot_curve.png"
fig1.savefig(p1, dpi=120, bbox_inches="tight")
plt.close(fig1)
pr(f"  MD_01_spot_curve.png        saved -> {p1}")

# -- Chart 2: Historical time series -----------------------------------------
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10))
fig2.suptitle("Historical EUR Interest Rates  (Nov 2015 – Nov 2025)", fontsize=12, y=0.98)

# Panel (0,0): Historical EURIBOR
ax = axes2[0, 0]
colors_eur = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
for i, c in enumerate(eur_codes_h):
    if c in df_eur.columns:
        s = df_eur[c].dropna()
        ax.plot(s.index, s.values, linewidth=0.8,
                label=eur_label_map.get(c, c), color=colors_eur[i % len(colors_eur)])
ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
ax.axhline(cap_strike_euribor, color="red", linewidth=0.8, linestyle="--",
           label=f"Bond cap EURIBOR ({cap_strike_euribor:.3f}%)")
ax.set_title("Historical EURIBOR (all tenors)")
ax.set_ylabel("Rate (%)")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Panel (0,1): EURIBOR 3M with bond context shading
ax = axes2[0, 1]
s3m = df_eur["EIBOR3M"].dropna() if "EIBOR3M" in df_eur.columns else pd.Series(dtype=float)
ax.plot(s3m.index, s3m.values, color="steelblue", linewidth=1.2, label="EURIBOR 3M")
ax.fill_between(s3m.index, 0, s3m.values.clip(0, cap_strike_euribor - 1e-9),
                where=(s3m.values > 0), alpha=0.15, color="steelblue", label="Linear regime (bond coupon)")
ax.fill_between(s3m.index, s3m.values, 0,
                where=(s3m.values < 0), alpha=0.2, color="red", label="Floor regime (coupon=0%)")
ax.fill_between(s3m.index, cap_strike_euribor, s3m.values,
                where=(s3m.values >= cap_strike_euribor), alpha=0.25, color="orange",
                label="Cap regime (coupon=5.40%)")
ax.axhline(0,                 color="red",    linewidth=0.8, linestyle="--")
ax.axhline(cap_strike_euribor, color="orange", linewidth=0.8, linestyle="--",
           label=f"Cap break-even ({cap_strike_euribor:.3f}%)")
ax.set_title("EURIBOR 3M – Bond coupon regime")
ax.set_ylabel("Rate (%)")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Panel (1,0): Historical IRS selected tenors
ax = axes2[1, 0]
irs_plot_cols = [("ICEIB2Y", "2Y IRS"), ("ICEIB5Y", "5Y IRS"),
                 ("ICEIB10", "10Y IRS"), ("ICEIB30", "30Y IRS")]
for c, lbl in irs_plot_cols:
    if c in df_irs.columns:
        s = df_irs[c].dropna()
        ax.plot(s.index, s.values, linewidth=0.9, label=lbl)
ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
ax.set_title("Historical EUR IRS par rates")
ax.set_ylabel("Rate (%)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel (1,1): 10Y-2Y slope
ax = axes2[1, 1]
if "ICEIB2Y" in df_irs.columns and "ICEIB10" in df_irs.columns:
    slope_h = (df_irs["ICEIB10"] - df_irs["ICEIB2Y"]) * 100
    slope_h = slope_h.dropna()
    ax.plot(slope_h.index, slope_h.values, color="purple", linewidth=0.9, label="10Y - 2Y slope (bps)")
    ax.fill_between(slope_h.index, slope_h.values, 0,
                    where=(slope_h.values > 0), alpha=0.15, color="green", label="Normal (steep)")
    ax.fill_between(slope_h.index, slope_h.values, 0,
                    where=(slope_h.values < 0), alpha=0.15, color="red", label="Inverted")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("IRS Curve Slope: 10Y - 2Y (bps)")
    ax.set_ylabel("Basis points")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig2.tight_layout(rect=[0, 0, 1, 0.95])
p2 = BASE / "MD_02_historical.png"
fig2.savefig(p2, dpi=120, bbox_inches="tight")
plt.close(fig2)
pr(f"  MD_02_historical.png        saved -> {p2}")

# -- Chart 3: Vol Surfaces (heatmaps) ----------------------------------------
strike_num_cols = [c for c in df_bach.columns
                   if c not in ("atm_strike",) and str(c) not in ("ATM", "STK", "None")]
try:
    strike_num_sorted = sorted(strike_num_cols, key=lambda x: float(x))
except Exception:
    strike_num_sorted = strike_num_cols

bach_matrix = df_bach[strike_num_sorted].apply(pd.to_numeric, errors="coerce")
db_matrix   = df_db[strike_num_sorted].apply(pd.to_numeric, errors="coerce")

fig3, axes3 = plt.subplots(1, 2, figsize=(16, 7))
fig3.suptitle("EUR Cap/Floor Volatility Surfaces  (ICAP, 05-Nov-2025)", fontsize=12, y=0.98)

for ax, mat, title, unit in [
    (axes3[0], bach_matrix, "Bachelier (Normal) vol surface", "bps"),
    (axes3[1], db_matrix,   "Displaced Black vol surface (shift=3%)", "%"),
]:
    vals = mat.values.astype(float)
    im = ax.imshow(vals, aspect="auto", cmap="RdYlGn_r", origin="upper",
                   vmin=np.nanmin(vals), vmax=np.nanpercentile(vals, 95))
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels([str(c) for c in mat.columns], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index.tolist(), fontsize=8)
    ax.set_xlabel("Strike (%)")
    ax.set_ylabel("Tenor")
    ax.set_title(f"{title}\n(units: {unit})")
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(unit)
    # annotate cells
    for i in range(len(mat.index)):
        for j in range(len(mat.columns)):
            v = vals[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}" if unit == "bps" else f"{v:.1f}",
                        ha="center", va="center", fontsize=5.5,
                        color="white" if v > np.nanpercentile(vals, 75) else "black")

fig3.tight_layout(rect=[0, 0, 1, 0.94])
p3 = BASE / "MD_03_vol_surfaces.png"
fig3.savefig(p3, dpi=120, bbox_inches="tight")
plt.close(fig3)
pr(f"  MD_03_vol_surfaces.png      saved -> {p3}")

# -- Chart 4: ATM vol term structure comparison --------------------------------
fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))
fig4.suptitle("ATM Volatility – Term Structure Comparison", fontsize=12, y=0.98)

bach_atm = df_bach["ATM"].apply(pd.to_numeric, errors="coerce")
db_atm   = df_db["ATM"].apply(pd.to_numeric, errors="coerce")

tenor_order_b = df_bach.index.tolist()
tenor_order_d = df_db.index.tolist()

# Map tenor strings to approximate years
def tenor_yrs(t):
    t = t.strip()
    if t.endswith("Y"):
        return float(t[:-1])
    if t == "18M" or t == "18m":
        return 1.5
    return None

x_b = [tenor_yrs(t) for t in tenor_order_b]
x_d = [tenor_yrs(t) for t in tenor_order_d]

ax = axes4[0]
ax.plot([x for x in x_b if x], [v for x, v in zip(x_b, bach_atm) if x],
        "o-", color="steelblue", linewidth=2, markersize=5, label="Bachelier ATM (bps)")
ax.set_xlabel("Tenor (years)")
ax.set_ylabel("ATM Vol (bps)")
ax.set_title("Bachelier (Normal) ATM Vol Term Structure")
ax.grid(True, alpha=0.3)
ax.axvline(x=8.5, color="red", linestyle="--", linewidth=1, label="Bond tenor (~8.5Y to maturity)")
ax.legend(fontsize=8)

ax = axes4[1]
ax.plot([x for x in x_d if x], [v for x, v in zip(x_d, db_atm) if x],
        "o-", color="darkorange", linewidth=2, markersize=5, label="Displaced Black ATM (%)")
ax.set_xlabel("Tenor (years)")
ax.set_ylabel("ATM Vol (%)")
ax.set_title("Displaced Black (Shift=3%) ATM Vol Term Structure")
ax.grid(True, alpha=0.3)
ax.axvline(x=8.5, color="red", linestyle="--", linewidth=1, label="Bond tenor (~8.5Y to maturity)")
ax.legend(fontsize=8)

fig4.tight_layout(rect=[0, 0, 1, 0.94])
p4 = BASE / "MD_04_atm_vol.png"
fig4.savefig(p4, dpi=120, bbox_inches="tight")
plt.close(fig4)
pr(f"  MD_04_atm_vol.png           saved -> {p4}")

# ── Summary ──────────────────────────────────────────────────────────────────
pr(sec("SUMMARY FOR BOND PRICING"))
pr("  The following data points are DIRECTLY RELEVANT to pricing the")
pr("  UniCredit Variable Rate Bond 2034 (ISIN IT0005599110):")
pr()
pr("  DISCOUNT / FORWARD CURVE:")
pr("    Use the IRS CURVE sheet as the basis for bootstrapping the")
pr("    EUR discount/EURIBOR forward curve (multi-curve: EURIBOR 3M")
pr("    forwards vs OIS/ESTER for discounting). The AB6E IRS rates")
pr("    (6M EURIBOR, semi-annual fixed coupon) need a basis adjustment")
pr("    to derive the 3M EURIBOR forward curve.")
pr()
pr("  VOLATILITY SURFACE:")
pr("    Both Bachelier and Displaced Black surfaces are provided.")
pr("    For the caps/floors embedded in the bond:")
pr("      - Cap @5.40%  → 10Y cap; strike ~2.77% OTM from ATM (2.63%)")
pr("      - Floor @0.00%→ 10Y floor; strike ~2.63% OTM (deep OTM floor)")
pr("    The Bachelier surface is preferred for EUR IR options post-NIRP")
pr("    (avoids log-of-negative-number issues). Bachelier 10Y ATM ~65 bps.")
pr()
pr("  HISTORICAL CONTEXT:")
pr("    The 10-yr history shows two distinct rate regimes:")
pr("    (A) NIRP era (2016-2022): negative rates; floor likely binding for")
pr("        short-dated bonds; irrelevant for this bond (issued Jun 2024).")
pr("    (B) Hiking cycle (2022-2023): rapid rate rise; cap would have been")
pr("        binding for coupon periods had the bond existed then.")
pr("    (C) Cutting cycle (2024-present): rates normalising downward;")
pr("        bond currently in the linear coupon regime.")
pr()
pr("  KEY NUMBERS TO CARRY INTO PRICING:")
pr(f"    Valuation date      : {snap_date.strftime('%d %b %Y') if isinstance(snap_date, datetime) else snap_date}")
pr(f"    EURIBOR 3M (spot)   : {euribor_3m:.3f}%")
pr(f"    EURIBOR 12M         : {irs_snap.get('EBF EURIBOR 12M DELAYED - OFFERED RATE'):.3f}%")
pr(f"    IRS 10Y par rate    : {v10y:.4f}%")
pr(f"    Bachelier ATM (10Y) : {atm_10y:.2f} bps" if atm_10y else "    Bachelier ATM (10Y) : see surface")
pr(f"    Bond cap EURIBOR    : {cap_strike_euribor:.3f}%  (= 5.40% / 1.60)")
pr(f"    Current coupon      : {1.60 * euribor_3m:.4f}% p.a.")

pr(f"\n{'='*72}")
pr("  END OF MARKET DATA INTUITION")
pr(f"{'='*72}")

# ── Write to file ─────────────────────────────────────────────────────────────
with open(OUT, "w", encoding="utf-8") as f:
    f.write(buf.getvalue())

print(f"\n[OK] Text output saved -> {OUT}")
print(f"[OK] Charts saved to   -> {BASE}")
