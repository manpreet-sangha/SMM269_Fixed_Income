"""
Investigate QuantLib date construction issue
"""

import QuantLib as ql
from datetime import datetime

print(f"QuantLib version: {ql.__version__}\n")

# Create date using QuantLib
ql_date = ql.Date(12, 9, 2024)

# Create using Python datetime
py_date = datetime(2024, 9, 12)

print(f"QuantLib Date(12, 9, 2024):")
print(f"  String representation: {str(ql_date)}")
print(f"  Day of month: {ql_date.dayOfMonth()}")
print(f"  Month: {ql_date.month()}")
print(f"  Year: {ql_date.year()}")
print(f"  Weekday index: {ql_date.weekday()}")

# Try to get more info
try:
    print(f"  ISO date: {ql_date.ISO()}")
except:
    pass

print(f"\nPython datetime(2024, 9, 12):")
print(f"  String: {py_date}")
print(f"  Day: {py_date.day}")
print(f"  Month: {py_date.month}")
print(f"  Year: {py_date.year}")
print(f"  Weekday (0=Mon): {py_date.weekday()}")
print(f"  ISO Weekday (1=Mon): {py_date.isoweekday()}")

# Check if it's a Leap year issue
print(f"\n\n2024 Leap year: {py_date.year % 4 == 0}")

# Test with different dates to see pattern
print(f"\n\nTesting date construction pattern:\n")
test_dates = [
    (1, 9, 2024),
    (12, 9, 2024),
    (30, 9, 2024),
    (12, 6, 2024),  # The issue date
]

for day, month, year in test_dates:
    try:
        q_date = ql.Date(day, month, year)
        p_date = datetime(year, month, day)
        
        weekdays_map = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        print(f"Date({day}, {month}, {year}):")
        print(f"  QuantLib: {q_date} | Weekday idx: {q_date.weekday()} | Day of month: {q_date.dayOfMonth()}")
        print(f"  Python:   {p_date.strftime('%A %d %B %Y')} | Weekday idx: {p_date.weekday()}")
        print()
    except Exception as e:
        print(f"Error: {e}\n")

# Try using serial number
print(f"\nUsing QL serial number approach:")
ql_date = ql.Date(12, 9, 2024)
print(f"Serial number: {ql_date.serialNumber()}")

# Create from serial 
serial = ql_date.serialNumber()
ql_date2 = ql.Date(serial)
print(f"From serial {serial}: {ql_date2}")
