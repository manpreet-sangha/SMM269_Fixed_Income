"""
Verify what day of week September 12, 2024 actually is
"""

import QuantLib as ql
from datetime import datetime

# Create the date
date_12sep = ql.Date(12, 9, 2024)
calendar = ql.TARGET()

# Check day of week using QuantLib
weekdays_ql = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_ql = weekdays_ql[date_12sep.weekday()]

# Check using Python datetime for verification
py_date = datetime(2024, 9, 12)
dow_python = py_date.strftime('%A')

print(f"Date: September 12, 2024")
print(f"  QuantLib day of week: {dow_ql} (weekday index: {date_12sep.weekday()})")
print(f"  Python datetime day:  {dow_python}")
print(f"  Is business day (TARGET): {calendar.isBusinessDay(date_12sep)}")

# Now test some nearby dates
print(f"\n\nNearby dates:")
for day in [10, 11, 12, 13, 14]:
    test_date = ql.Date(day, 9, 2024)
    is_bday = calendar.isBusinessDay(test_date)
    dow = weekdays_ql[test_date.weekday()]
    print(f"  Sep {day:02d}: {dow:<10} - Business day? {is_bday}")

# Check with advanced/adjust functions
print(f"\n\nAdjustment tests on Sep 12 2024:")
date_12 = ql.Date(12, 9, 2024)

print(f"Original date: {date_12}")
print(f"  isBusinessDay: {calendar.isBusinessDay(date_12)}")
print(f"  adjust(ModifiedFollowing): {calendar.adjust(date_12, ql.ModifiedFollowing)}")

# Try adjust with a known non-business day
new_year = ql.Date(1, 1, 2025)
print(f"\nNew Year 2025: {new_year}")
print(f"  isBusinessDay: {calendar.isBusinessDay(new_year)}")
print(f"  adjust(ModifiedFollowing): {calendar.adjust(new_year, ql.ModifiedFollowing)}")
