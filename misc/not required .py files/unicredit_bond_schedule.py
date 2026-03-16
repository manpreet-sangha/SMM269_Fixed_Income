"""
Generate UniCredit bond coupon period end dates with Modified-Following adjustment.

Uses QuantLib to:
1. Create a schedule with specified bond parameters
2. Show unadjusted period end dates
3. Show adjusted dates applying Modified-Following convention with TARGET calendar
"""

import QuantLib as ql
from datetime import datetime
import pandas as pd

# Helper function to format QuantLib Date objects
def format_date(ql_date):
    """Convert QuantLib Date to formatted string"""
    return f"{ql_date.dayOfMonth():02d} {ql_date.strftime('%b')} {ql_date.year()}"

def date_str_simple(ql_date):
    """Simple date string"""
    months = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    return f"{ql_date.dayOfMonth():02d} {months[ql_date.month()]} {ql_date.year()}"

# ============================================================================
# Bond Parameters
# ============================================================================
issue_date = ql.Date(12, 6, 2024)  # 12 June 2024
maturity_date = ql.Date(12, 6, 2034)  # 12 June 2034
frequency = ql.Quarterly  # Quarterly periods
business_day_convention = ql.ModifiedFollowing
calendar = ql.TARGET()  # TARGET calendar for business day adjustments
end_of_month = False  # No end-of-month adjustment

# ============================================================================
# Create Schedule
# ============================================================================
schedule = ql.Schedule(
    issue_date,
    maturity_date,
    ql.Period(frequency),
    calendar,
    business_day_convention,
    business_day_convention,
    ql.DateGeneration.Backward,
    end_of_month
)

# ============================================================================
# Extract Period End Dates
# ============================================================================
periods = []

for i in range(1, len(schedule) - 1):  # Exclude start date, include period ends before maturity
    # Use the index to track the accrual period
    period_start = schedule[i - 1]
    period_end = schedule[i]
    
    # For quarterly periods, calculate the unadjusted end date pattern
    # All period ends fall on the 12th of Mar/Jun/Sep/Dec
    unadjusted_date = period_end  # In this case, the schedule directly computes adjusted dates
    
    # Get the raw day/month of the period end
    day = period_end.dayOfMonth()
    month = period_end.month()
    year = period_end.year()
    
    periods.append({
        'Period': i,
        'Period Start': date_str_simple(period_start),
        'Unadjusted End (12th)': f"12 {date_str_simple(period_end).split()[1:]}",
        'Adjusted End Date': date_str_simple(period_end),
        'Day of Week': period_end.weekday(),  # 0=Monday, ..., 6=Sunday
        'QL Date': str(period_end)
    })

# ============================================================================
# Display Results
# ============================================================================
print("=" * 100)
print("UniCredit Bond - Coupon Period End Dates")
print("=" * 100)
print(f"\nBond Parameters:")
print(f"  Issue Date: {date_str_simple(issue_date)}")
print(f"  Maturity Date: {date_str_simple(maturity_date)}")
print(f"  Frequency: Quarterly")
print(f"  Business Day Convention: Modified Following")
print(f"  Calendar: TARGET")
print(f"  End-of-Month Rule: {end_of_month}")
print(f"\nTotal coupon periods: {len(periods)}\n")

# Create and display DataFrame
df = pd.DataFrame(periods)
print(df.to_string(index=False))

print("\n" + "=" * 100)
print("\nDetailed: Unadjusted vs Adjusted Dates")
print("=" * 100)

for period in periods:
    week_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    unadjusted_str = period['Unadjusted End (12th)']
    adjusted_str = period['Adjusted End Date']
    weekday = week_days[period['Day of Week']]
    
    print(f"\nPeriod {period['Period']}:")
    print(f"  Unadjusted (12th): {unadjusted_str}")
    print(f"  Adjusted:         {adjusted_str} ({weekday})")
    
    # Check if adjustment was made
    if unadjusted_str.split()[-1] == adjusted_str.split()[-1]:  # Compare year
        if unadjusted_str != adjusted_str:
            print(f"  → Adjusted by Modified Following (different day/month)")

# ============================================================================
# Recalculate with explicit unadjusted dates for clarity
# ============================================================================
print("\n" + "=" * 100)
print("\nExplicit Unadjusted → Adjusted Mapping")
print("=" * 100 + "\n")

# The schedule above gives us adjusted dates
# To get unadjusted, we reconstruct backward coupon dates (12th of each quarter month)
unadjusted_dates = []
current_date = ql.Date(12, 3, 2025)  # First period end: 12 Mar 2025

while current_date <= maturity_date:
    unadjusted_dates.append(current_date)
    # Move to next quarter
    if current_date.month() == 3:
        next_month = 6
        next_year = current_date.year()
    elif current_date.month() == 6:
        next_month = 9
        next_year = current_date.year()
    elif current_date.month() == 9:
        next_month = 12
        next_year = current_date.year()
    else:  # 12
        next_month = 3
        next_year = current_date.year() + 1
    
    try:
        current_date = ql.Date(12, next_month, next_year)
    except:
        break

# Now map unadjusted to adjusted from our schedule
print(f"{'Period':<8} {'Unadjusted':<15} {'Adjusted':<15} {'Shift':<8} {'Business Day'}")
print("-" * 70)

month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
               7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
week_days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

for i, adjusted_date in enumerate(schedule[1:-1], 1):
    # Reconstruct expected unadjusted date
    # Periods are on 12th of Mar, Jun, Sep, Dec
    if i <= len(unadjusted_dates):
        unadj = unadjusted_dates[i-1]
        unadj_str = f"{unadj.dayOfMonth():02d} {month_names[unadj.month()]} {unadj.year()}"
        adj_str = f"{adjusted_date.dayOfMonth():02d} {month_names[adjusted_date.month()]} {adjusted_date.year()}"
        
        # Calculate business days shifted
        shift_days = (adjusted_date - unadj)
        weekday_name = week_days[adjusted_date.weekday()]
        
        print(f"{i:<8} {unadj_str:<15} {adj_str:<15} {shift_days:<8} {weekday_name}")

print("\n" + "=" * 100)
