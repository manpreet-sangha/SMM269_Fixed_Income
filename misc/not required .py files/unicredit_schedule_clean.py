"""
Extract UniCredit bond coupon period end dates with Modified-Following adjustment.

Shows the first 7 periods with:
1. Period number
2. Unadjusted period end date (12th of the quarter month)  
3. Adjusted period end date (after Modified-Following with TARGET calendar)
"""

import QuantLib as ql

# Bond Parameters
issue_date = ql.Date(12, 6, 2024)  # 12 June 2024
maturity_date = ql.Date(12, 6, 2034)  # 12 June 2034
frequency = ql.Quarterly
business_day_convention = ql.ModifiedFollowing
calendar = ql.TARGET()
end_of_month = False

# Create Schedule
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

# Month mapping
months = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
          7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

def ql_to_str(date):
    """Convert QuantLib Date to string format"""
    return f"{date.dayOfMonth():02d} {months[date.month()]} {date.year()}"

# Print header
print("\n" + "="*90)
print("UniCredit Bond - Coupon Period End Dates (First 7 Periods)")
print("="*90)
print(f"\nBond Details:")
print(f"  Issue Date:               {ql_to_str(issue_date)}")
print(f"  Maturity Date:            {ql_to_str(maturity_date)}")
print(f"  Frequency:                Quarterly")
print(f"  Business Day Convention:  Modified Following")
print(f"  Calendar:                 TARGET (Trans-European Automated Real-time Gross-settlement Express Transfer)")
print(f"  End-of-Month Rule:        Disabled")

print(f"\n{'-'*90}")
print(f"{'Period':<8} {'Unadjusted (12th)':<20} {'Adjusted Date':<20} {'Day of Week':<15} {'Shift (days)'}")
print(f"{'-'*90}\n")

# Extract and display first 7 coupon periods
count = 0
for i in range(1, min(len(schedule) - 1, 8)):  # First 7 periods (indices 1-7)
    period_end = schedule[i]
    adjusted_date = period_end
    
    # Construct unadjusted date (always 12th of the month)
    unadjusted_date = ql.Date(12, period_end.month(), period_end.year())
    
    # Format dates
    unadj_str = ql_to_str(unadjusted_date)
    adj_str = ql_to_str(adjusted_date)
    weekday = weekdays[adjusted_date.weekday()]
    
    # Calculate shift in business days
    shift = adjusted_date - unadjusted_date
    shift_str = str(shift) if shift != 0 else "-"
    
    print(f"{i:<8} {unadj_str:<20} {adj_str:<20} {weekday:<15} {shift_str}")
    
    count += 1

print(f"\n{'-'*90}\n")

# Show detailed mapping
print("\nDetailed Unadjusted → Adjusted Mapping (First 7 Periods):\n")
for i in range(1, min(len(schedule) - 1, 8)):
    period_end = schedule[i]
    adjusted_date = period_end
    unadjusted_date = ql.Date(12, period_end.month(), period_end.year())
    
    unadj_str = ql_to_str(unadjusted_date)
    adj_str = ql_to_str(adjusted_date)
    weekday = weekdays[adjusted_date.weekday()]
    shift = adjusted_date - unadjusted_date
    
    print(f"Period {i}:")
    print(f"  Unadjusted:  {unadj_str}")
    print(f"  Adjusted:    {adj_str} ({weekday})")
    if shift != 0:
        direction = "forward" if shift > 0 else "backward"
        print(f"  → Modified Following shifted {abs(shift)} days {direction}")
    else:
        print(f"  → No adjustment needed (already a business day)")
    print()

print("="*90)
