"""
Direct verification of TARGET calendar configuration
"""

import QuantLib as ql

calendar = ql.TARGET()

# Test a known weekend
saturday = ql.Date(14, 9, 2024)  # Saturday
sunday = ql.Date(15, 9, 2024)    # Sunday
monday = ql.Date(16, 9, 2024)     # Monday

print("Testing known weekend dates:")
print(f"Saturday 14 Sep 2024 - Business day? {calendar.isBusinessDay(saturday)}")
print(f"Sunday 15 Sep 2024 - Business day? {calendar.isBusinessDay(sunday)}")
print(f"Monday 16 Sep 2024 - Business day? {calendar.isBusinessDay(monday)}")

# Test Euro New Year holidays
new_year = ql.Date(1, 1, 2025)
print(f"\n\nNew Year 2025 - Business day? {calendar.isBusinessDay(new_year)}")

# List calendar name and properties
print(f"\nCalendar name: {calendar.name()}")

# Try using adjusted method
test_date = ql.Date(12, 9, 2024)  # Saturday
print(f"\nDirect adjust() call on Saturday 12 Sep 2024:")
print(f"  Using ModifiedFollowing: {calendar.adjust(test_date, ql.ModifiedFollowing)}")
print(f"  Using Following: {calendar.adjust(test_date, ql.Following)}")
print(f"  Using Preceding: {calendar.adjust(test_date, ql.Preceding)}")
