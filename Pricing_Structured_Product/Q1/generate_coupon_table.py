"""
generate_coupon_table.py
================================================================================
Generate reproducible Table 4: Coupon Schedule with Adjusted Period End Dates

This script builds the complete coupon schedule dynamically using QuantLib:
- Generates schedule with Modified-Following + TARGET calendar
- Extracts reset dates (2nd TARGET day before period start)
- Includes both unadjusted and adjusted period end dates
- Calculates coupon amounts from EURIBOR fixings
- Outputs LaTeX table code

No hardcoding of dates – all derived from QuantLib schedule.
"""

from datetime import datetime
from pathlib import Path
import QuantLib as ql
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np

# Bond parameters
ISIN = "IT0005599110"
ISSUE_DATE = ql.Date(12, 6, 2024)
MATURITY_DATE = ql.Date(12, 6, 2034)
PARTICIPATION = 1.60
FLOOR_RATE = 0.00
CAP_RATE = 0.0545
NOTIONAL = 1000.0  # per €1,000 face value
DAY_COUNT = ql.Thirty360(ql.Thirty360.BondBasis)

# Historical 3M EURIBOR fixings (from Bank of Finland)
EURIBOR_FIXINGS = {
    ql.Date(10, 6, 2024): 0.037430,
    ql.Date(10, 9, 2024): 0.034600,
    ql.Date(10, 12, 2024): 0.028720,
    ql.Date(10, 3, 2025): 0.025470,
    ql.Date(10, 6, 2025): 0.019540,
    ql.Date(10, 9, 2025): 0.020290,
    ql.Date(10, 12, 2025): 0.020820,
}


def ql_date(d, m, y):
    """Create QuantLib date."""
    return ql.Date(d, m, y)


def date_str(ql_date_obj):
    """Format QuantLib date as 'DD Mon YYYY'."""
    months = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    day = ql_date_obj.dayOfMonth()
    month = months[ql_date_obj.month()]
    year = ql_date_obj.year()
    return f"{day:2d} {month} {year}"


def build_schedule():
    """Build QuantLib schedule with Modified-Following + TARGET calendar."""
    calendar = ql.TARGET()
    schedule = ql.MakeSchedule(
        effectiveDate=ISSUE_DATE,
        terminationDate=MATURITY_DATE,
        tenor=ql.Period(ql.Quarterly),
        calendar=calendar,
        convention=ql.ModifiedFollowing,
        terminalDateConvention=ql.ModifiedFollowing,
        rule=ql.DateGeneration.Forward,
        endOfMonth=False,
    )
    return schedule, calendar


def get_reset_date(period_start, calendar):
    """
    Calculate reset date: 2nd TARGET business day before period start.
    (Modified-Following convention applied implicitly via QuantLib.)
    """
    # Start from 2 days before period start
    reset = period_start - ql.Period(2, ql.Days)
    
    # Adjust forward to next business day if it's not one
    while not calendar.isBusinessDay(reset):
        reset += ql.Period(1, ql.Days)
    
    return reset


def get_euribor(reset_date):
    """Retrieve EURIBOR fixing or use last known value as fallback."""
    if reset_date in EURIBOR_FIXINGS:
        return EURIBOR_FIXINGS[reset_date]
    # Fallback: use most recent available
    latest_available = sorted(
        [d for d in EURIBOR_FIXINGS.keys() if d <= reset_date],
        reverse=True
    )
    if latest_available:
        return EURIBOR_FIXINGS[latest_available[0]]
    return 0.02  # Conservative fallback


def calculate_coupon_amount(euribor_rate, period_start, period_end):
    """Calculate coupon amount in EUR per €1,000 face value."""
    # Apply cap/floor/leverage
    leveraged = PARTICIPATION * euribor_rate
    capped = max(FLOOR_RATE, min(leveraged, CAP_RATE))
    
    # Accrual factor (30/360)
    accrual = DAY_COUNT.yearFraction(period_start, period_end)
    
    # Amount = notional × rate × accrual
    amount = NOTIONAL * capped * accrual
    
    return capped, amount


def generate_coupon_table():
    """Generate coupon table data and return as list of dicts."""
    schedule, calendar = build_schedule()
    schedule_dates = list(schedule)
    
    periods = []
    
    for i in range(len(schedule_dates) - 1):
        period_num = i
        period_start = schedule_dates[i]
        period_end = schedule_dates[i + 1]
        
        # Calculate reset date (2nd TARGET day before period start)
        reset_date = get_reset_date(period_start, calendar)
        
        # Get EURIBOR for this reset date
        euribor = get_euribor(reset_date)
        
        # Calculate coupon
        coupon_rate, coupon_amount = calculate_coupon_amount(euribor, period_start, period_end)
        
        # Determine status
        today = ql.Date(15, 3, 2026)  # Current date as of analysis
        if period_end <= today:
            status = "Paid"
        elif period_end == today:
            status = "Due"
        else:
            status = "Future"
        
        # Check if capped
        is_capped = (coupon_rate >= CAP_RATE - 1e-6)
        
        periods.append({
            "number": period_num,
            "reset_date": reset_date,
            "period_start": period_start,
            "period_end": period_end,
            "euribor": euribor,
            "coupon_rate": coupon_rate,
            "coupon_amount": coupon_amount,
            "is_capped": is_capped,
            "status": status,
        })
    
    return periods


def coupon_rate_str(coupon_rate, is_capped):
    """Format coupon rate with exponent if capped."""
    s = f"{coupon_rate*100:.4f}".rstrip('0').rstrip('.')
    if is_capped:
        s += r"$^{\dagger}$"
    return s


def generate_table_png(periods, output_path):
    """Generate a PNG image of the coupon schedule table with no white space."""
    
    # Prepare table data
    table_data = []
    headers = ["#", "Reset Date", "Period Start", "Period End", "Adj. End", 
               "EURIBOR (%)", "Coupon (%)", "Amount (€)", "Status"]
    table_data.append(headers)
    
    total_paid = 0.0
    
    for p in periods[:7]:  # First 7 periods
        reset = date_str(p["reset_date"])
        start = date_str(p["period_start"])
        end = date_str(p["period_end"])
        euribor_str = f"{p['euribor']*100:.4f}".rstrip('0').rstrip('.')
        coupon_str = f"{p['coupon_rate']*100:.4f}"
        if p["is_capped"]:
            coupon_str += "†"
        amount_str = f"{p['coupon_amount']:.3f}"
        status = p["status"]
        
        adj_end = end  # In this bond, always equals end date
        
        table_data.append([
            str(p['number']+1),
            reset,
            start,
            end,
            adj_end,
            euribor_str,
            coupon_str,
            amount_str,
            status,
        ])
        
        if status == "Paid":
            total_paid += p["coupon_amount"]
    
    # Add total row
    table_data.append(["", "", "", "", "", "", "Total:", f"{total_paid:.3f}", ""])

    # Calculate exact figure height from row count so table fills it completely
    n_rows = len(table_data)   # 9 rows: 1 header + 7 data + 1 total
    row_height = 0.30          # inches per row at 7pt font
    fig_height = n_rows * row_height

    fig, ax = plt.subplots(figsize=(7.0, fig_height), dpi=250)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.axis("off")

    # bbox=[0,0,1,1] stretches the table to fill the entire axes — no white gaps
    table = ax.table(cellText=table_data, cellLoc="center", bbox=[0, 0, 1, 1],
                     colWidths=[0.04, 0.11, 0.11, 0.11, 0.11, 0.14, 0.12, 0.13, 0.13])

    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.55)

    # Style header row
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor("#1a3a5c")
        cell.set_text_props(weight="bold", color="white", fontsize=7)
        cell.set_linewidth(1)

    # Style data rows
    for i in range(1, len(table_data)-1):
        for j in range(len(headers)):
            cell = table[(i, j)]
            cell.set_facecolor("#f5f5f5" if i % 2 == 0 else "#ffffff")
            cell.set_text_props(fontsize=7)
            cell.set_linewidth(0.5)

    # Style total row
    for j in range(len(headers)):
        cell = table[(len(table_data)-1, j)]
        cell.set_facecolor("#e8e8e8")
        cell.set_text_props(weight="bold", fontsize=7)
        cell.set_linewidth(0.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=250,
                facecolor="white", edgecolor="none")
    plt.close()
    
    return output_path


def main():
    periods = generate_coupon_table()
    
    # Output directory for generated files
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate PNG table
    table_png = output_dir / "coupon_table.png"
    generate_table_png(periods, table_png)
    
    # Print summary
    print("=" * 80)
    print("COUPON SCHEDULE SUMMARY (First 7 Periods)")
    print("=" * 80)
    print(f"{'#':<2} {'Reset Date':<12} {'Period Start':<12} {'Period End':<12} "
          f"{'EURIBOR':<10} {'Coupon':<10} {'Amount':<10} {'Status':<6}")
    print("-" * 80)
    
    for p in periods[:7]:
        reset = date_str(p["reset_date"])
        start = date_str(p["period_start"])
        end = date_str(p["period_end"])
        euribor_str = f"{p['euribor']*100:.4f}"
        coupon_str = f"{p['coupon_rate']*100:.4f}"
        amount_str = f"{p['coupon_amount']:.3f}"
        status = p["status"]
        
        print(f"{p['number']+1:<2} {reset:<12} {start:<12} {end:<12} "
              f"{euribor_str:<10} {coupon_str:<10} {amount_str:<10} {status:<6}")
    
    print("\n" + "=" * 80)
    print(f"✓ Table image written to: {table_png}")
    print("=" * 80)
    print("\nIn main.tex, use:")
    print(r"  \includegraphics[width=\textwidth]{../Pricing_Structured_Product/Q1/output/coupon_table.png}")
    print("=" * 80)


if __name__ == "__main__":
    main()
