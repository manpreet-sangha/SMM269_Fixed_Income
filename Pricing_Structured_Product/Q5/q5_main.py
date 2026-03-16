"""
q5_main.py
--------------------------------------------------------------------------------
This implementation uses the Bachelier (normal) forward-rate model for EURIBOR
caplets/floorlets. The choice is consistent with the market data provided in
the workbook (sheet "Bachelier": EUR cap/floor normal vols), and is suitable in
low/near-zero rate regimes.

Calibration approach
--------------------
1) Read 6M EUR cap ATM normal vol quotes by maturity from the market workbook.
2) Convert cap ATM vols into an implied optionlet (caplet) normal-vol term
   structure via sequential variance allocation:

      sigma_cap(T_n)^2 * A_n = sum_i w_i * sigma_i^2,

   where A_n = sum_i w_i and w_i = alpha_i * DF(t_i).
3) Use the stripped optionlet vols as piecewise-flat in expiry time.
4) Price the bond option-like components (long floorlets @ 0%, short caplets
   @ 5.45% on the leveraged leg) period by period.

Outputs (output/)
-----------------
  q5_cap_atm_quotes.csv
  q5_caplet_vols.csv
  q5_option_components.csv
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import openpyxl
import QuantLib as ql

# ---------------------------------------------------------------------------
# Paths and imports
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
Q3_DIR = ROOT_DIR / "Q3"
COMMON_DIR = ROOT_DIR / "common"
MARKET_FILE = COMMON_DIR / "input" / "MarketData202526.xlsx"
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT_DIR))
from Q3.q3_market_data import load_market_data
from Q3.q3_main import build_curve

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Bond and convention constants
# ---------------------------------------------------------------------------
ISSUE_DATE = ql.Date(12, 6, 2024)
MATURITY_DATE = ql.Date(12, 6, 2034)
PARTICIPATION = 1.60
FLOOR_RATE = 0.00
CAP_RATE = 0.0545
NOTIONAL = 1000.0

CALENDAR = ql.TARGET()
COUPON_DC = ql.Thirty360(ql.Thirty360.BondBasis)
OPTION_DC = ql.Actual365Fixed()
FLOAT_DC = ql.Actual360()


@dataclass
class CapAtmQuote:
    tenor_label: str
    tenor: ql.Period
    atm_strike: float      # decimal
    atm_norm_vol: float    # decimal rate vol, e.g. 0.0027 = 27 bps


@dataclass
class StrippedNode:
    payment_date: ql.Date
    sigma: float
    weight: float


def _parse_tenor_label(label: str) -> ql.Period:
    t = label.strip().upper()
    if t.endswith("M"):
        return ql.Period(int(t[:-1]), ql.Months)
    if t.endswith("Y"):
        return ql.Period(int(t[:-1]), ql.Years)
    raise ValueError(f"Unsupported tenor label: {label}")


def _build_bond_schedule() -> ql.Schedule:
    return ql.MakeSchedule(
        effectiveDate=ISSUE_DATE,
        terminationDate=MATURITY_DATE,
        tenor=ql.Period(ql.Quarterly),
        calendar=CALENDAR,
        convention=ql.ModifiedFollowing,
        terminalDateConvention=ql.ModifiedFollowing,
        rule=ql.DateGeneration.Forward,
        endOfMonth=False,
    )


def _get_reset_date(period_start: ql.Date) -> ql.Date:
    """Reset is 2 TARGET business days before period start."""
    return CALENDAR.advance(period_start, -2, ql.Days, ql.Preceding)


def load_bachelier_atm_quotes(path: Path) -> list[CapAtmQuote]:
    """Load ATM normal vol term quotes from the workbook Bachelier sheet."""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb["Bachelier"]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    header = rows[4]
    stk_col = None
    atm_col = None
    for idx, value in enumerate(header):
        if isinstance(value, str) and value.strip().upper() == "STK":
            stk_col = idx
        if isinstance(value, str) and value.strip().upper() == "ATM":
            atm_col = idx

    if stk_col is None or atm_col is None:
        raise RuntimeError("Could not find STK/ATM columns in Bachelier sheet.")

    out: list[CapAtmQuote] = []
    for row in rows[5:]:
        tenor_raw = row[0]
        if tenor_raw is None:
            continue
        tenor_label = str(tenor_raw).strip().upper()
        if tenor_label in {"", "SHIFT"}:
            continue
        if not (tenor_label.endswith("M") or tenor_label.endswith("Y")):
            continue

        stk = row[stk_col]
        atm = row[atm_col]
        if stk is None or atm is None:
            continue

        quote = CapAtmQuote(
            tenor_label=tenor_label,
            tenor=_parse_tenor_label(tenor_label),
            atm_strike=float(stk) / 100.0,
            atm_norm_vol=float(atm) / 10_000.0,
        )
        out.append(quote)

    # Keep calibration tenors up to 10Y, matching the Q3 curve horizon.
    filtered: list[CapAtmQuote] = []
    for q in out:
        years = q.tenor.length() / (12.0 if q.tenor.units() == ql.Months else 1.0)
        if years <= 10.0:
            filtered.append(q)

    filtered.sort(key=lambda q: OPTION_DC.yearFraction(ql.Date(1, 1, 2025), ql.Date(1, 1, 2025) + q.tenor))
    return filtered


def _six_month_schedule(start_date: ql.Date, end_date: ql.Date) -> ql.Schedule:
    return ql.MakeSchedule(
        effectiveDate=start_date,
        terminationDate=end_date,
        tenor=ql.Period(ql.Semiannual),
        calendar=CALENDAR,
        convention=ql.ModifiedFollowing,
        terminalDateConvention=ql.ModifiedFollowing,
        rule=ql.DateGeneration.Forward,
        endOfMonth=False,
    )


def strip_caplet_normal_vols(
    curve: ql.YieldTermStructure,
    eval_date: ql.Date,
    quotes: list[CapAtmQuote],
) -> list[StrippedNode]:
    """
    Strip implied optionlet normal vols from ATM cap normal vols.

    Uses sequential variance matching with annuity weights:
      sigma_cap^2 * A = sum_i (w_i * sigma_i^2),  w_i = alpha_i * DF_i.
    """
    settlement = CALENDAR.advance(eval_date, 2, ql.Days)
    sigma_by_date: dict[int, float] = {}
    weight_by_date: dict[int, float] = {}

    for q in quotes:
        cap_end = CALENDAR.advance(settlement, q.tenor, ql.ModifiedFollowing)
        sched = _six_month_schedule(settlement, cap_end)
        dts = list(sched)
        periods = list(zip(dts[:-1], dts[1:]))

        cap_weights: list[tuple[ql.Date, float]] = []
        for p_start, p_end in periods:
            alpha = FLOAT_DC.yearFraction(p_start, p_end)
            df = curve.discount(p_end)
            w = alpha * df
            cap_weights.append((p_end, w))

        annuity = sum(w for _, w in cap_weights)
        target_var_weighted = (q.atm_norm_vol ** 2) * annuity

        known_contrib = 0.0
        unknown: list[tuple[ql.Date, float]] = []
        for p_end, w in cap_weights:
            key = p_end.serialNumber()
            if key in sigma_by_date:
                known_contrib += w * sigma_by_date[key] ** 2
            else:
                unknown.append((p_end, w))

        if not unknown:
            continue

        rem = max(target_var_weighted - known_contrib, 0.0)
        rem_weight = sum(w for _, w in unknown)
        sigma_new = math.sqrt(rem / rem_weight) if rem_weight > 0 else 0.0

        for p_end, w in unknown:
            key = p_end.serialNumber()
            sigma_by_date[key] = sigma_new
            weight_by_date[key] = w

    nodes: list[StrippedNode] = []
    for key, sigma in sigma_by_date.items():
        payment_date = ql.Date(key)
        nodes.append(StrippedNode(payment_date=payment_date, sigma=sigma, weight=weight_by_date[key]))

    nodes.sort(key=lambda x: x.payment_date.serialNumber())
    return nodes


def _sigma_for_expiry(expiry: ql.Date, nodes: list[StrippedNode]) -> float:
    if not nodes:
        return 0.0
    for node in nodes:
        if expiry <= node.payment_date:
            return node.sigma
    return nodes[-1].sigma


def price_option_like_components(
    curve: ql.YieldTermStructure,
    eval_date: ql.Date,
    stripped_nodes: list[StrippedNode],
) -> tuple[list[dict], dict[str, float]]:
    """Price long floorlets and short caplets in the coupon decomposition."""
    schedule = _build_bond_schedule()
    dates = list(schedule)

    eff_notional = NOTIONAL * PARTICIPATION
    floor_strike = FLOOR_RATE / PARTICIPATION
    cap_strike = CAP_RATE / PARTICIPATION

    rows: list[dict] = []
    total_floor = 0.0
    total_cap = 0.0

    for i in range(len(dates) - 1):
        p_start = dates[i]
        p_end = dates[i + 1]
        if p_end <= eval_date:
            continue

        reset = _get_reset_date(p_start)
        # If reset has already occurred, the coupon is fixed and has no
        # remaining option value at eval_date.
        if reset <= eval_date:
            continue

        alpha = COUPON_DC.yearFraction(p_start, p_end)
        df = curve.discount(p_end)
        fwd = curve.forwardRate(p_start, p_end, FLOAT_DC, ql.Simple).rate()

        texp = max(OPTION_DC.yearFraction(eval_date, reset), 0.0)
        sigma = _sigma_for_expiry(reset, stripped_nodes)
        std_dev = sigma * math.sqrt(texp)

        caplet_unit = ql.bachelierBlackFormula(ql.Option.Call, cap_strike, fwd, std_dev, 1.0)
        floorlet_unit = ql.bachelierBlackFormula(ql.Option.Put, floor_strike, fwd, std_dev, 1.0)

        caplet_pv = eff_notional * alpha * df * caplet_unit
        floorlet_pv = eff_notional * alpha * df * floorlet_unit

        total_cap += caplet_pv
        total_floor += floorlet_pv

        rows.append(
            {
                "period": i + 1,
                "start": p_start.to_date().isoformat(),
                "end": p_end.to_date().isoformat(),
                "reset": reset.to_date().isoformat(),
                "alpha_30_360": alpha,
                "forward_3m": fwd,
                "sigma_norm": sigma,
                "caplet_pv": caplet_pv,
                "floorlet_pv": floorlet_pv,
            }
        )

    summary = {
        "long_floor_pv": total_floor,
        "short_cap_pv": total_cap,
        "net_option_pv": total_floor - total_cap,
        "cap_equivalent_strike": cap_strike,
    }
    return rows, summary


def _save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    data = load_market_data()
    curve, eval_date, _ = build_curve(data)

    quotes = load_bachelier_atm_quotes(MARKET_FILE)
    stripped = strip_caplet_normal_vols(curve, eval_date, quotes)
    leg_rows, opt_summary = price_option_like_components(curve, eval_date, stripped)

    quote_rows = [
        {
            "tenor": q.tenor_label,
            "atm_strike_%": round(q.atm_strike * 100, 6),
            "atm_norm_vol_bps": round(q.atm_norm_vol * 10_000, 6),
        }
        for q in quotes
    ]

    strip_rows = [
        {
            "payment_date": n.payment_date.to_date().isoformat(),
            "sigma_norm_bps": round(n.sigma * 10_000, 6),
            "weight_alpha_df": round(n.weight, 8),
        }
        for n in stripped
    ]

    out_quotes = OUTPUT_DIR / "q5_cap_atm_quotes.csv"
    out_strip = OUTPUT_DIR / "q5_caplet_vols.csv"
    out_legs = OUTPUT_DIR / "q5_option_components.csv"

    _save_csv(out_quotes, quote_rows, ["tenor", "atm_strike_%", "atm_norm_vol_bps"])
    _save_csv(out_strip, strip_rows, ["payment_date", "sigma_norm_bps", "weight_alpha_df"])
    _save_csv(
        out_legs,
        leg_rows,
        [
            "period",
            "start",
            "end",
            "reset",
            "alpha_30_360",
            "forward_3m",
            "sigma_norm",
            "caplet_pv",
            "floorlet_pv",
        ],
    )

    print("\n" + "=" * 76)
    print("  QUESTION 5 -- Option-Like Component Pricing (Bachelier Model)")
    print("=" * 76)
    print(f"  Eval date                : {eval_date.to_date().isoformat()}")
    print(f"  Calibration quotes used  : {len(quotes)} ATM 6M-cap maturities")
    print(f"  Stripped caplet nodes    : {len(stripped)}")
    print(f"  Floor strike (EURIBOR)   : {FLOOR_RATE / PARTICIPATION * 100:.4f}%")
    print(f"  Cap strike (EURIBOR)     : {CAP_RATE / PARTICIPATION * 100:.4f}%")
    print("-")
    print(f"  Long floor PV            : EUR {opt_summary['long_floor_pv']:.4f}")
    print(f"  Short cap PV             : EUR {opt_summary['short_cap_pv']:.4f}")
    print(f"  Net option PV (floor-cap): EUR {opt_summary['net_option_pv']:.4f}")
    print("-")
    print(f"  Saved                    : {out_quotes}")
    print(f"  Saved                    : {out_strip}")
    print(f"  Saved                    : {out_legs}")
    print("=" * 76)


if __name__ == "__main__":
    main()
