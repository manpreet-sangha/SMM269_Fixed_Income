"""
q1_bond_overview.py
--------------------------------------------------------------------------------
Question 1  --  Bond Structure, Key Characteristics, Coupon Profile & Replication

Reads  : misc/bond_details.txt
Outputs: console summary  +  q1_coupon_profile.png

QuantLib is used for:
  *  Building the exact coupon schedule (30/360, Modified-Following, T2 calendar)
  *  Constructing a flat EURIBOR discount/forecast curve
  *  Pricing the embedded interest-rate floor  (long, strike = 0%)
  *  Pricing the embedded interest-rate cap    (short, strike = 5.45%)
  *  Computing the replication fair-value decomposition
  All vanilla option pricing uses Black-76 (normal Black model) which is
  standard for EURIBOR caps/floors post-2015 negative-rate environment.
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Force UTF-8 output so box-drawing characters render on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib.pyplot as plt
import numpy as np
import QuantLib as ql

# -- Paths ---------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
INPUT_FILE = SCRIPT_DIR / "bond_details.txt"
CHART_OUT  = SCRIPT_DIR / "q1_coupon_profile.png"

# -----------------------------------------------------------------------------
# 1. Data loading
# -----------------------------------------------------------------------------

def load_bond_details(filepath: Path) -> dict[str, str]:
    """
    Parse the 'BOND CHARACTERISTICS  --  PARSED FIELDS' section of bond_details.txt
    into a plain {field: value} dictionary.
    """
    details: dict[str, str] = {}
    in_header = False
    with open(filepath, encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if "BOND CHARACTERISTICS" in stripped:
                in_header = True
                continue
            if in_header and stripped.startswith("==="):
                if details:
                    break
                continue
            if in_header and ":" in stripped:
                key, _, val = stripped.partition(":")
                details[key.strip()] = val.strip()
    if not details:
        sys.exit(f"ERROR: Could not parse bond fields from {filepath}")
    return details


def get_float(details: dict, key: str, default: float = 0.0) -> float:
    """Extract a numeric value (strip %, 'per cent', letters)."""
    raw = details.get(key, str(default))
    numeric = re.sub(r"[^0-9.\-]", "", raw.split()[0])
    try:
        return float(numeric)
    except ValueError:
        return default


# -----------------------------------------------------------------------------
# 2. QuantLib helpers  --  dates, calendar, curve
# -----------------------------------------------------------------------------

def ql_date(day: int, month: int, year: int) -> ql.Date:
    return ql.Date(day, month, year)


def build_flat_curve(rate: float, ref_date: ql.Date,
                     day_count: ql.DayCounter = ql.Actual360()) -> ql.YieldTermStructureHandle:
    """Flat continuously-compounded curve anchored at ref_date."""
    curve = ql.FlatForward(ref_date, rate, day_count)
    return ql.YieldTermStructureHandle(curve)


def build_ql_schedule(issue: ql.Date, maturity: ql.Date,
                      calendar: ql.Calendar) -> ql.Schedule:
    """Quarterly schedule, Modified-Following, 30/360, EOM off."""
    tenor     = ql.Period(ql.Quarterly)
    bdc       = ql.ModifiedFollowing
    end_bdc   = ql.ModifiedFollowing
    rule      = ql.DateGeneration.Forward
    eom       = False
    return ql.MakeSchedule(
        effectiveDate   = issue,
        terminationDate = maturity,
        tenor           = tenor,
        calendar        = calendar,
        convention      = bdc,
        terminalDateConvention = end_bdc,
        rule            = rule,
        endOfMonth      = eom,
    )


# -----------------------------------------------------------------------------
# 3. QuantLib cap/floor pricer (Black-76 normal model)
# -----------------------------------------------------------------------------

def price_cap_floor(
    cap_or_floor: str,          # "cap" or "floor"
    strike: float,              # decimal, e.g. 0.0545
    participation: float,       # e.g. 1.60
    notional: float,            # e.g. 1_000
    schedule: ql.Schedule,
    forecast_handle: ql.YieldTermStructureHandle,
    discount_handle: ql.YieldTermStructureHandle,
    normal_vol: float,          # normal vol in rate units, e.g. 0.007 = 70 bps
    pricing_date: ql.Date,
) -> tuple[float, list[float]]:
    """
    Price a cap or floor on the *leveraged* reference rate using
    the Bachelier (normal) model via QuantLib.

    The effective notional is participation x notional, and the strike
    is expressed on the leveraged rate (e.g. 5.45% on 1.6xr), which is
    equivalent to strike/participation on EURIBOR.

    Uses ql.IborLeg to build well-formed coupon objects, then prices
    with BachelierCapFloorEngine (no separate pricer required on the leg).
    """
    eff_notional = participation * notional
    eff_strike   = strike / participation   # EURIBOR-equivalent strike

    index = ql.Euribor3M(forecast_handle)

    # Build a standard IborLeg  --  QuantLib wires coupons correctly
    leg = ql.IborLeg(
        [eff_notional],
        schedule,
        index,
        ql.Thirty360(ql.Thirty360.BondBasis),
        ql.ModifiedFollowing,
        [index.fixingDays()],
        [1.0],   # gearings
        [0.0],   # spreads
    )

    if cap_or_floor.lower() == "cap":
        instrument = ql.Cap(leg, [eff_strike])
    else:
        instrument = ql.Floor(leg, [eff_strike])

    vol_handle = ql.QuoteHandle(ql.SimpleQuote(normal_vol))
    engine     = ql.BachelierCapFloorEngine(discount_handle, vol_handle)
    instrument.setPricingEngine(engine)

    try:
        npv = instrument.NPV()
    except Exception as e:
        print(f"  [WARN] Cap/floor pricing error: {e}")
        npv = float("nan")

    return npv, []


def price_leveraged_frn(
    notional: float,
    participation: float,
    schedule: ql.Schedule,
    forecast_handle: ql.YieldTermStructureHandle,
    discount_handle: ql.YieldTermStructureHandle,
    pricing_date: ql.Date,
) -> float:
    """
    PV of 1.6 x EURIBOR quarterly coupons (no cap/floor).
    Computes coupon-by-coupon using the forecast curve for forward rates.
    """
    dc    = ql.Thirty360(ql.Thirty360.BondBasis)
    pv    = 0.0
    dates = list(schedule)
    for start, end in zip(dates[:-1], dates[1:]):
        if end <= pricing_date:
            continue
        # forward rate from the forecast curve over [start, end]
        fwd_rate = forecast_handle.forwardRate(start, end, dc, ql.Simple).rate()
        alpha    = dc.yearFraction(start, end)
        coupon   = participation * notional * fwd_rate * alpha
        df       = discount_handle.discount(end)
        pv      += coupon * df
    return pv


def price_redemption(notional: float, maturity: ql.Date,
                     discount_handle: ql.YieldTermStructureHandle) -> float:
    return notional * discount_handle.discount(maturity)


# -----------------------------------------------------------------------------
# 4. QuantLib bond decomposition
# -----------------------------------------------------------------------------

def ql_replication_pv(details: dict[str, str]) -> dict[str, float]:
    """
    Set valuation date = last reset date (10 Sep 2025).
    Use a flat curve calibrated to the implied EURIBOR on that date
    (back-solved from the current coupon rate and participation factor).

    Decomposition:
        Bond PV = Leveraged FRN PV  +  Floor PV  -  Cap PV  +  Redemption PV
    """
    participation  = get_float(details, "Participation Factor", 1.60)
    floor_strike   = get_float(details, "Floor (Min Rate)",     0.00) / 100
    cap_strike     = get_float(details, "Cap (Max Rate)",       5.45) / 100
    current_coupon = get_float(details, "Current Coupon Rate", 3.2464) / 100
    notional       = 1_000.0

    # Imply current 3m EURIBOR from last reset coupon
    current_euribor = current_coupon / participation   # ~ 2.029%

    # QL dates
    pricing_date = ql_date(10, 9, 2025)   # last reset date
    issue_date   = ql_date(12, 6, 2024)
    maturity     = ql_date(12, 6, 2034)
    ql.Settings.instance().evaluationDate = pricing_date

    calendar = ql.TARGET()

    # Flat forward curve from current EURIBOR (Actual/360, simple -> approx c.c.)
    cc_rate = np.log(1 + current_euribor * 90 / 360) * (360 / 90)   # rough
    fwd_crv = build_flat_curve(current_euribor, pricing_date, ql.Actual360())
    dis_crv = build_flat_curve(current_euribor, pricing_date, ql.Actual360())

    schedule = build_ql_schedule(pricing_date, maturity, calendar)

    # Market normal vol for 3m EURIBOR caps/floors (typical ~60-80 bps in 2025)
    normal_vol = 0.0070   # 70 bps normal vol

    # Seed the historical EURIBOR fixing for the first coupon period's reset date.
    # Eval date = 10 Sep 2025 = start of first period; the fixing date is
    # 2 TARGET business days prior (8 Sep 2025).  Without this, QuantLib raises
    # "Missing Euribor3M fixing" and returns NaN for the cap/floor NPV.
    _fixing_dt = calendar.advance(pricing_date, -2, ql.Days, ql.Preceding)
    _idx_seed  = ql.Euribor3M(fwd_crv)
    try:
        _idx_seed.addFixing(_fixing_dt, current_euribor)
    except RuntimeError:
        pass   # fixing already registered

    frn_pv  = price_leveraged_frn(notional, participation, schedule,
                                   fwd_crv, dis_crv, pricing_date)
    red_pv  = price_redemption(notional, maturity, dis_crv)

    floor_pv, _ = price_cap_floor("floor", floor_strike, participation, notional,
                                   schedule, fwd_crv, dis_crv, normal_vol, pricing_date)
    cap_pv,   _ = price_cap_floor("cap",   cap_strike,   participation, notional,
                                   schedule, fwd_crv, dis_crv, normal_vol, pricing_date)

    bond_pv = frn_pv + red_pv + floor_pv - cap_pv

    return {
        "implied_euribor_pct"  : current_euribor * 100,
        "pricing_date"         : "10 Sep 2025 (last reset)",
        "normal_vol_bps"       : normal_vol * 10_000,
        "leveraged_frn_pv"     : frn_pv,
        "long_floor_pv"        : floor_pv,
        "short_cap_pv"         : cap_pv,
        "redemption_pv"        : red_pv,
        "total_bond_pv"        : bond_pv,
        "notional"             : notional,
    }


# -----------------------------------------------------------------------------
# 5. Console summary
# -----------------------------------------------------------------------------

def print_bond_overview(details: dict[str, str]) -> None:
    sections = {
        "IDENTIFICATION": [
            "Bond Name", "ISIN", "Bond Type", "Series Number", "Tranche Number",
            "CFI", "FISN",
        ],
        "ISSUER & PROGRAMME": [
            "Issuer Full Name", "Registered Office", "Company Reg. Number",
            "Programme Size", "Status / Seniority", "Governing Law",
        ],
        "ECONOMICS": [
            "Currency", "Aggregate Nominal Amount", "Specified Denominations",
            "Issue Price", "Tenor", "Issue Date", "Maturity Date",
            "Final Redemption Amount",
        ],
        "COUPON STRUCTURE": [
            "Reference Rate", "Screen Page", "Participation Factor",
            "Floor (Min Rate)", "Cap (Max Rate)", "Current Coupon Rate",
            "Coupon Frequency", "Day Count Convention",
            "Business Day Convention", "Additional Business Centre",
            "Reset / Determination Date", "Interest Payment Dates",
            "Calculation Agent", "Reference Rate Replacement",
        ],
        "OPTIONALITY & REDEMPTION": [
            "Call Option", "Issuer Call (Standard)", "Clean-Up Redemption",
            "Redemption Type", "Redemption Basis",
        ],
        "MARKET & SETTLEMENT": [
            "Listing", "Form of Notes", "Clearing", "Delivery",
            "Credit Rating (Issuer)", "MiFID II Target Market",
        ],
        "RISK FACTORS": ["Risk Factors"],
    }
    print()
    print("=" * 68)
    print("  QUESTION 1  --  UniCredit Variable Rate Bond 2034  [ISIN IT0005599110]")
    print("=" * 68)
    for section, keys in sections.items():
        print(f"\n  -- {section} {'-' * (62 - len(section))}")
        for k in keys:
            val = details.get(k, "")
            if not val:
                continue
            label = f"  {k}"
            if len(val) > 45:
                print(f"  {label:<32}: {val[:45]}")
                rem = val[45:]
                while rem:
                    print(f"  {'':32}  {rem[:45]}")
                    rem = rem[45:]
            else:
                print(f"  {label:<32}: {val}")
    print()


# -----------------------------------------------------------------------------
# 6. Coupon profile chart (with QuantLib schedule overlay)
# -----------------------------------------------------------------------------

def plot_coupon_profile(details: dict[str, str]) -> None:
    participation  = get_float(details, "Participation Factor", 1.60)
    floor_rate     = get_float(details, "Floor (Min Rate)",     0.00) / 100
    cap_rate       = get_float(details, "Cap (Max Rate)",       5.45) / 100
    current_coupon = get_float(details, "Current Coupon Rate", 3.2464) / 100
    current_euribor = current_coupon / participation

    # -- Build QuantLib schedule to extract actual reset dates ----------------
    ql.Settings.instance().evaluationDate = ql_date(7, 3, 2026)
    calendar = ql.TARGET()
    sched    = build_ql_schedule(ql_date(12, 6, 2024), ql_date(12, 6, 2034), calendar)
    n_periods = len(list(sched)) - 1

    # -- Coupon payoff curve ---------------------------------------------------
    euribor   = np.linspace(-0.01, 0.07, 600)
    leveraged = participation * euribor
    effective = np.clip(leveraged, floor_rate, cap_rate)

    euribor_floor_break = floor_rate / participation
    euribor_cap_break   = cap_rate   / participation

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#f9f9f9")

    # -- LEFT: Coupon Profile --------------------------------------------------
    ax = axes[0]
    ax.set_facecolor("#f9f9f9")
    ax.axvspan(-1, euribor_floor_break * 100,
               alpha=0.08, color="steelblue")
    ax.axvspan(euribor_cap_break * 100, 7,
               alpha=0.08, color="tomato")
    ax.plot(euribor * 100, leveraged * 100,
            "--", lw=1.4, color="cornflowerblue",
            label=f"Unconstrained: {participation:.2f}x EURIBOR")
    ax.plot(euribor * 100, effective * 100,
            lw=2.5, color="navy",
            label="Effective Coupon (Floor & Cap applied)")
    ax.axhline(floor_rate * 100, color="green",  ls=":", lw=1.2,
               label=f"Floor = {floor_rate*100:.2f}%")
    ax.axhline(cap_rate   * 100, color="tomato", ls=":", lw=1.2,
               label=f"Cap   = {cap_rate*100:.2f}%")
    ax.axvline(current_euribor * 100, color="darkorange", ls="-.", lw=1.2,
               label=f"Implied EURIBOR ~ {current_euribor*100:.3f}%")
    ax.scatter([current_euribor * 100], [current_coupon * 100],
               color="darkorange", zorder=5, s=80,
               label=f"Current Coupon = {current_coupon*100:.4f}%")

    ax.annotate(f"Floor break-even\n{euribor_floor_break*100:.2f}%",
                xy=(euribor_floor_break * 100, floor_rate * 100),
                xytext=(euribor_floor_break * 100 - 0.7, 0.6),
                fontsize=8, color="green",
                arrowprops=dict(arrowstyle="->", color="green", lw=0.8))
    ax.annotate(f"Cap break-even\n{euribor_cap_break*100:.2f}%",
                xy=(euribor_cap_break * 100, cap_rate * 100),
                xytext=(euribor_cap_break * 100 + 0.2, cap_rate * 100 - 0.9),
                fontsize=8, color="tomato",
                arrowprops=dict(arrowstyle="->", color="tomato", lw=0.8))

    ax.text(euribor_floor_break * 50, cap_rate * 100 * 0.3,
            "Floor Active", ha="center", fontsize=8, color="steelblue", alpha=0.8)
    ax.text((euribor_floor_break + euribor_cap_break) / 2 * 100,
            cap_rate * 100 * 0.45,
            "1.6 x EURIBOR\n(linear)", ha="center", fontsize=8, color="navy", alpha=0.8)
    ax.text((euribor_cap_break + 0.07) / 2 * 100, cap_rate * 100 * 0.65,
            "Cap Active", ha="center", fontsize=8, color="tomato", alpha=0.8)

    ax.set_xlabel("3-Month EURIBOR (%)", fontsize=11)
    ax.set_ylabel("Effective Annual Coupon Rate (%)", fontsize=11)
    ax.set_title(f"Coupon Payoff Profile\nclip({participation:.2f}x EURIBOR, "
                 f"{floor_rate*100:.2f}%, {cap_rate*100:.2f}%)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, ls="--", alpha=0.4)
    ax.set_xlim(-1, 7)
    ax.set_ylim(-0.3, 6.5)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}%"))

    # -- RIGHT: Replication decomposition bar ----------------------------------
    ax2 = axes[1]
    ax2.set_facecolor("#f9f9f9")

    # Varying EURIBOR scenarios for stacked coupon contribution chart
    scenarios = np.array([-0.5, 0.0, 1.0, 2.029, 3.0, 3.4063, 4.0, 5.0, 6.0])
    plain_frn  = participation * scenarios / 100 * 100   # raw pct
    floor_gain = np.maximum(floor_rate - participation * scenarios / 100, 0) * 100
    cap_loss   = np.maximum(participation * scenarios / 100 - cap_rate, 0) * 100
    net        = np.clip(participation * scenarios / 100, floor_rate, cap_rate) * 100

    x = np.arange(len(scenarios))
    w = 0.35
    ax2.bar(x - w/2, plain_frn,  w, label=f"{participation:.1f}x EURIBOR (FRN)",
            color="cornflowerblue", alpha=0.75, zorder=3)
    ax2.bar(x - w/2, floor_gain, w, bottom=plain_frn,
            label="+ Floor payoff", color="green", alpha=0.75, zorder=3)
    ax2.bar(x - w/2, -cap_loss,  w, bottom=plain_frn + floor_gain,
            label="- Cap clawback", color="tomato", alpha=0.75, zorder=3)
    ax2.scatter(x + w/2, net, color="navy", zorder=5, s=60,
                label="Net effective coupon")
    ax2.axhline(cap_rate * 100,   color="tomato", ls=":", lw=1.0)
    ax2.axhline(floor_rate * 100, color="green",  ls=":", lw=1.0)

    # mark current EURIBOR
    ax2.axvline(3 + w/2, color="darkorange", ls="-.", lw=0.8, alpha=0.6)

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{s:.2f}%" for s in scenarios], rotation=40, ha="right", fontsize=8)
    ax2.set_xlabel("3-Month EURIBOR Scenario", fontsize=10)
    ax2.set_ylabel("Coupon Component (%)", fontsize=10)
    ax2.set_title("Replication Decomposition by Scenario\n"
                  "FRN leg + Floor + (-Cap) = Net Coupon",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(True, ls="--", alpha=0.4, axis="y")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}%"))

    fig.suptitle(
        "UniCredit Variable Rate Bond 2034  |  ISIN IT0005599110\n"
        f"{n_periods} quarterly coupon periods  |  12 Jun 2024 -> 12 Jun 2034  "
        "|  30/360  |  Modified-Following  |  T2 calendar",
        fontsize=10, y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(CHART_OUT, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  [OK] Chart saved -> {CHART_OUT}\n")


# -----------------------------------------------------------------------------
# 7. Replication explanation + QuantLib PV decomposition
# -----------------------------------------------------------------------------

def explain_replication(details: dict[str, str], ql_pvs: dict[str, float]) -> None:
    participation = get_float(details, "Participation Factor", 1.60)
    floor_pct     = get_float(details, "Floor (Min Rate)",     0.00)
    cap_pct       = get_float(details, "Cap (Max Rate)",       5.45)
    notional      = ql_pvs["notional"]
    sep = "-" * 68

    print("=" * 68)
    print("  COUPON REPLICATION STRATEGY  (QuantLib-assisted)")
    print("=" * 68)
    print(f"""
  Coupon formula (per period):
      c(r) = clip( {participation:.2f} x r,  {floor_pct:.2f}%,  {cap_pct:.2f}% )
           = max( {floor_pct:.2f}%,  min( {participation:.2f} x r,  {cap_pct:.2f}% ) )

  where r = 3-month EURIBOR observed on each quarterly reset date.
  Day count: 30/360.  Payment: quarterly in arrear.
""")

    print(f"  {sep}")
    print("  DECOMPOSITION INTO STANDARD INSTRUMENTS")
    print(f"  {sep}")
    print(f"""
  Algebraically:
      c(r) = {participation:.2f} x r
           + max(0  -  {participation:.2f}xr,  0)          <- long floor  @ 0%
           - max({participation:.2f}xr  - {cap_pct:.2f}%,  0)    <- short cap   @ {cap_pct:.2f}%

  +-------------------------------------------------------------+
  | COMPONENT          | INSTRUMENT         | POSITION          |
  +-------------------------------------------------------------+
  | 1) 1.6x FRN leg   | Floating-rate note | Long (1.6x notl.) |
  |    pays 1.6xEURIBOR on EUR{notional:,.0f} notional             |
  +-------------------------------------------------------------+
  | 2) Floor @ {floor_pct:.2f}%   | IR Floor (floorlets)| Long             |
  |    strike on 1.6xr: {floor_pct:.2f}%  (EURIBOR equiv: {floor_pct/participation:.3f}%)|
  |    protects against negative coupons                        |
  +-------------------------------------------------------------+
  | 3) Cap   @ {cap_pct:.2f}%   | IR Cap (caplets)   | Short             |
  |    strike on 1.6xr: {cap_pct:.2f}% (EURIBOR equiv: {cap_pct/participation:.4f}%)|
  |    issuer retains upside above {cap_pct:.2f}%                |
  +-------------------------------------------------------------+
  | 4) Redemption      | Zero-coupon bond   | Long (100% at par)|
  +-------------------------------------------------------------+
""")

    print(f"  {sep}")
    print("  QUANTLIB FAIR-VALUE DECOMPOSITION")
    print(f"  (Valuation date: {ql_pvs['pricing_date']})")
    print(f"  (Implied 3m EURIBOR: {ql_pvs['implied_euribor_pct']:.4f}%  |  "
          f"Normal vol: {ql_pvs['normal_vol_bps']:.0f} bps)")
    print(f"  {sep}")
    frn = ql_pvs["leveraged_frn_pv"]
    flo = ql_pvs["long_floor_pv"]
    cap = ql_pvs["short_cap_pv"]
    red = ql_pvs["redemption_pv"]
    tot = ql_pvs["total_bond_pv"]
    print(f"""
  Component                          PV (EUR per EUR{notional:,.0f} bond)
  ---------------------------------  -------------------------
  (1) Leveraged FRN  (1.6x EURIBOR) : EUR{frn:>10.4f}
  (2) Long Floor  @ {floor_pct:.2f}%            : EUR{flo:>10.4f}
  (3) Short Cap   @ {cap_pct:.2f}%           : EUR{cap:>10.4f}  (cost to bondholder)
  (4) Redemption  @ 100%             : EUR{red:>10.4f}
  ---------------------------------  -------------------------
  Estimated Fair Value               : EUR{tot:>10.4f}  ({tot/notional*100:.4f}% of par)

  Note: Uses a flat EURIBOR curve and a single normal vol (Bachelier/Black-76).
  A proper pricing would require the full EURIBOR cap vol surface and a
  multi-curve OIS/EURIBOR framework (see Question 2 / Part C).
""")

    print(f"  {sep}")
    print("  ALTERNATIVE REPLICATION VIEWS")
    print(f"  {sep}")
    print(f"""
  (A) Zero-Cost Collar + Leveraged FRN
      = Long 1.6x FRN
        + Long  Floor({floor_pct:.2f}%) on 1.6x notional
        + Short Cap  ({cap_pct:.2f}%) on 1.6x notional
      (net option cost ~ premium_floor - premium_cap; funded by lower headline yield)

  (B) Fixed Bond + Receiver Swap + Collar
      = Long fixed-coupon bond (at mid-coupon {cap_pct/2:.2f}%)
        + Long receiver swap  (receive fixed {cap_pct/2:.2f}%, pay EURIBOR)
        + Long Floor + Short Cap adjustments to embed the 1.6x leverage

  (C) Bull Spread on EURIBOR Caplets (caplet collar)
      = Long caplet at strike K1 = {floor_pct/participation:.3f}% (EURIBOR floor break-even)
        + Short caplet at strike K2 = {cap_pct/participation:.4f}% (EURIBOR cap break-even)
      -> payoff = max(r - {floor_pct/participation:.3f}%, 0) - max(r - {cap_pct/participation:.4f}%, 0)
               = min(max(r, {floor_pct/participation:.3f}%), {cap_pct/participation:.4f}%) - const
      then scaled by participation factor 1.6

  (D) Digital / Binary Option interpretation
      The Cap can also be viewed as a portfolio of two calls:
        Long call at {cap_pct/participation:.4f}% EURIBOR  -  short unlimited call
        -> equivalent to a digital payout at {cap_pct:.2f}% for rate scenarios above cap
""")
    print(f"  {sep}\n")


# -----------------------------------------------------------------------------
# 8. Main
# -----------------------------------------------------------------------------

def main() -> None:
    if not INPUT_FILE.exists():
        sys.exit(f"ERROR: Input file not found: {INPUT_FILE}")

    details = load_bond_details(INPUT_FILE)
    print_bond_overview(details)

    print("  [QuantLib] Building curves and pricing embedded options...")
    ql_pvs = ql_replication_pv(details)

    explain_replication(details, ql_pvs)
    plot_coupon_profile(details)


if __name__ == "__main__":
    main()
