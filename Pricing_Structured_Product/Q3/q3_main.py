"""
q3_main.py
--------------------------------------------------------------------------------
Constructs a single-curve EUR term structure using:
  - EURIBOR cash deposits  (O/N, 1W, 1M, 3M, 6M)       short end
  - EUR AB6E IRS par rates (1Y–10Y)                      long end

Interpolation
-------------
  ql.PiecewiseLogLinearDiscount
  ≡ log-linear interpolation on discount factors
  ≡ piecewise-constant instantaneous forward rates between pillar nodes.

Outputs (output/)
-----------------
  q3_zero_rates.csv        zero rates (cc & annual compounding) on monthly grid
  q3_discount_factors.csv  discount factors on monthly grid
  q3_fwd3m_rates.csv       3M forward EURIBOR rates on monthly grid
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import QuantLib as ql

# ---------------------------------------------------------------------------
# Local import
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from q3_market_data import load_market_data

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Tenor string → QuantLib Period
# ---------------------------------------------------------------------------
_TENOR_PERIOD: dict[str, ql.Period] = {
    "ON":  ql.Period(1,  ql.Days),
    "1W":  ql.Period(1,  ql.Weeks),
    "1M":  ql.Period(1,  ql.Months),
    "3M":  ql.Period(3,  ql.Months),
    "6M":  ql.Period(6,  ql.Months),
    "12M": ql.Period(12, ql.Months),
    "1Y":  ql.Period(1,  ql.Years),
    "2Y":  ql.Period(2,  ql.Years),
    "3Y":  ql.Period(3,  ql.Years),
    "4Y":  ql.Period(4,  ql.Years),
    "5Y":  ql.Period(5,  ql.Years),
    "6Y":  ql.Period(6,  ql.Years),
    "7Y":  ql.Period(7,  ql.Years),
    "8Y":  ql.Period(8,  ql.Years),
    "9Y":  ql.Period(9,  ql.Years),
    "10Y": ql.Period(10, ql.Years),
}

# Deposits to include (12M excluded — 1Y IRS takes over from 1Y)
_DEPOSIT_TENORS = {"ON", "1W", "1M", "3M", "6M"}


def _ql_date(d) -> ql.Date:
    """Convert datetime.date → ql.Date."""
    return ql.Date(d.day, d.month, d.year)


# ---------------------------------------------------------------------------
# Core bootstrap function
# ---------------------------------------------------------------------------
def build_curve(data: dict):
    """
    Build the EUR single discount curve from deposits + IRS.

    Parameters
    ----------
    data : dict
        Output of q3_market_data.load_market_data()

    Returns
    -------
    curve      : ql.YieldTermStructure  (PiecewiseLogLinearDiscount)
    eval_date  : ql.Date
    helpers    : list[ql.RateHelper]    (all helpers, for validation)
    """
    eval_date = _ql_date(data["eval_date"])
    ql.Settings.instance().evaluationDate = eval_date

    calendar       = ql.TARGET()
    settlement_date = calendar.advance(eval_date, 2, ql.Days)

    # -----------------------------------------------------------------------
    # Step 1 – EURIBOR deposit helpers  (short end: O/N to 6M)
    # -----------------------------------------------------------------------
    # Day count: ACT/360 (EURIBOR standard)
    # Fixing days: 0 for O/N, 2 for all others
    deposit_helpers = []
    for tenor, rate in data["deposits"].items():
        if tenor not in _DEPOSIT_TENORS:
            continue
        fixing_days = 0 if tenor == "ON" else 2
        helper = ql.DepositRateHelper(
            ql.QuoteHandle(ql.SimpleQuote(rate / 100.0)),
            _TENOR_PERIOD[tenor],
            fixing_days,
            calendar,
            ql.ModifiedFollowing,
            False,            # endOfMonth
            ql.Actual360(),
        )
        deposit_helpers.append(helper)

    # -----------------------------------------------------------------------
    # Step 2 – IRS par-swap helpers  (long end: 1Y to 10Y)
    # -----------------------------------------------------------------------
    # AB6E convention:
    #   Fixed  leg: Annual, 30/360 BondBasis
    #   Float  leg: 6M EURIBOR, semi-annual, ACT/360
    euribor6m = ql.Euribor6M()
    swap_helpers = []
    for tenor, rate in data["irs"].items():
        helper = ql.SwapRateHelper(
            ql.QuoteHandle(ql.SimpleQuote(rate / 100.0)),
            _TENOR_PERIOD[tenor],
            calendar,
            ql.Annual,                              # fixed leg: annual
            ql.ModifiedFollowing,
            ql.Thirty360(ql.Thirty360.BondBasis),   # fixed leg: 30/360
            euribor6m,                              # float index: 6M EURIBOR
        )
        swap_helpers.append(helper)

    all_helpers = deposit_helpers + swap_helpers

    # -----------------------------------------------------------------------
    # Step 3 – Bootstrap
    # Log-linear on DFs  ≡  piecewise-constant instantaneous forwards
    # -----------------------------------------------------------------------
    curve = ql.PiecewiseLogLinearDiscount(
        settlement_date,
        all_helpers,
        ql.Actual365Fixed(),
    )
    curve.enableExtrapolation()

    return curve, eval_date, all_helpers


# ---------------------------------------------------------------------------
# Extract curve data on a monthly grid
# ---------------------------------------------------------------------------
def extract_curve_data(curve, eval_date: ql.Date) -> list[dict]:
    """
    Sample zero rates, discount factors, and 3M forward rates on a
    monthly grid from 1M to 120M (10Y).

    Parameters
    ----------
    curve      : bootstrapped QL yield term structure
    eval_date  : ql.Date  (evaluation / fixing date)

    Returns
    -------
    list of dicts with keys:
        maturity_date, years, discount_factor,
        zero_rate_cc_%, zero_rate_ann_%, fwd_3m_%
    """
    calendar   = ql.TARGET()
    dc         = ql.Actual365Fixed()
    dc_fwd     = ql.Actual360()
    rows       = []

    for months in range(1, 121):
        mat  = calendar.advance(eval_date, months, ql.Months, ql.ModifiedFollowing)
        t    = dc.yearFraction(eval_date, mat)
        df   = curve.discount(mat)
        if df <= 0:
            continue

        z_cc  = curve.zeroRate(mat, dc, ql.Continuous, ql.Annual).rate()
        z_ann = curve.zeroRate(mat, dc, ql.Compounded,  ql.Annual).rate()

        # 3M forward starting at mat
        fwd_end  = calendar.advance(mat, 3, ql.Months, ql.ModifiedFollowing)
        fwd_rate = curve.forwardRate(mat, fwd_end, dc_fwd, ql.Simple).rate()

        rows.append({
            "maturity_date":  mat.to_date().isoformat(),
            "years":          round(t, 4),
            "discount_factor": round(df, 8),
            "zero_rate_cc_%":  round(z_cc  * 100, 6),
            "zero_rate_ann_%": round(z_ann * 100, 6),
            "fwd_3m_%":        round(fwd_rate * 100, 6),
        })

    return rows


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------
def save_curve_csvs(rows: list[dict]) -> None:
    specs = [
        ("q3_zero_rates.csv",
         ["maturity_date", "years", "zero_rate_cc_%", "zero_rate_ann_%"]),
        ("q3_discount_factors.csv",
         ["maturity_date", "years", "discount_factor"]),
        ("q3_fwd3m_rates.csv",
         ["maturity_date", "years", "fwd_3m_%"]),
    ]
    for filename, fields in specs:
        out = OUTPUT_DIR / filename
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows([{k: r[k] for k in fields} for r in rows])
        print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Standalone usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data = load_market_data()

    print("Building EUR yield curve …")
    curve, eval_date, helpers = build_curve(data)

    rows = extract_curve_data(curve, eval_date)
    save_curve_csvs(rows)

    # Quick sanity print at key maturities
    cal = ql.TARGET()
    dc  = ql.Actual365Fixed()
    print(f"\n  Curve nodes  : {len(curve.nodes())}")
    print(f"  {'Maturity':<8}  {'DF':>10}  {'Zero Ann%':>10}  {'Fwd 3M%':>10}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}")
    for yr in [1, 2, 3, 5, 7, 10]:
        mat = cal.advance(eval_date, yr, ql.Years)
        df  = curve.discount(mat)
        z   = curve.zeroRate(mat, dc, ql.Compounded, ql.Annual).rate() * 100
        fwd_end = cal.advance(mat, 3, ql.Months)
        f   = curve.forwardRate(mat, fwd_end, ql.Actual360(), ql.Simple).rate() * 100
        print(f"  {yr}Y      {df:>10.6f}  {z:>10.4f}  {f:>10.4f}")
