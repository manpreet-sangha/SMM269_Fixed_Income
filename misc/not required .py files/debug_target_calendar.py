"""
Debug script: Verify TARGET calendar business day recognition for the first 7 periods
"""

import QuantLib as ql

# Setup
calendar = ql.TARGET()
months = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
          7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

def ql_to_str(date):
    return f"{date.dayOfMonth():02d} {months[date.month()]} {date.year()}"

# First 7 coupon period end dates (unadjusted)
unadjusted_dates = [
    ql.Date(12, 9, 2024),   # 12 Sep 2024
    ql.Date(12, 12, 2024),  # 12 Dec 2024
    ql.Date(12, 3, 2025),   # 12 Mar 2025
    ql.Date(12, 6, 2025),   # 12 Jun 2025
    ql.Date(12, 9, 2025),   # 12 Sep 2025
    ql.Date(12, 12, 2025),  # 12 Dec 2025
    ql.Date(12, 3, 2026),   # 12 Mar 2026
]

print("\n" + "="*90)
print("TARGET Calendar Analysis - First 7 Periods")
print("="*90 + "\n")

print(f"{'Period':<8} {'Date':<18} {'Day':<12} {'Business Day?':<15} {'Adjusted (Mod Foll.)':<20}")
print("-"*90)

for i, unadj_date in enumerate(unadjusted_dates, 1):
    is_business_day = calendar.isBusinessDay(unadj_date)
    
    # Apply Modified Following
    adjusted_date = calendar.adjust(unadj_date, ql.ModifiedFollowing)
    
    day_name = weekdays[unadj_date.weekday()]
    adj_str = ql_to_str(adjusted_date)
    
    print(f"{i:<8} {ql_to_str(unadj_date):<18} {day_name:<12} {str(is_business_day):<15} {adj_str:<20}")

print("\n" + "="*90)
print("\nDetailed Business Day Information:\n")

for i, unadj_date in enumerate(unadjusted_dates, 1):
    is_business_day = calendar.isBusinessDay(unadj_date)
    adjusted_date = calendar.adjust(unadj_date, ql.ModifiedFollowing)
    shift = adjusted_date - unadj_date
    day_name = weekdays[unadj_date.weekday()]
    adj_day = weekdays[adjusted_date.weekday()]
    
    print(f"Period {i}: {ql_to_str(unadj_date)} ({day_name})")
    print(f"  Is business day: {is_business_day}")
    
    if is_business_day:
        print(f"  → No adjustment needed")
    else:
        print(f"  → Adjusted to: {ql_to_str(adjusted_date)} ({adj_day})")
        print(f"  → Shift: {shift} days")
    
    # Check next few days if not a business day
    if not is_business_day:
        print(f"\n  Next business days from {ql_to_str(unadj_date)}:")
        check_date = ql.Date(unadj_date)
        for day_offset in range(1, 5):
            check_date = ql.Date(check_date.dayOfMonth() + day_offset, check_date.month(), check_date.year())
            if calendar.isBusinessDay(check_date):
                check_day = weekdays[check_date.weekday()]
                print(f"    {ql_to_str(check_date)} ({check_day})")
                break
    
    print()

print("="*90)
