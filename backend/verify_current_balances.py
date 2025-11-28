"""
Quick Verification Script
========================
Compares the balances shown in the UI with the Excel file data.
"""

import sys
import io

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Excel data (from the image, oldest to newest)
excel_data = [
    {'date': '31-07-2025', 'desc': 'RTR TAX SERVICES SDN BHD', 'inward': 0, 'outward': 626.40, 'balance': 15671.42},
    {'date': '01-08-2025', 'desc': 'LEMBAGA HASIL DALAM NEGERI MAL', 'inward': 0, 'outward': 14196.00, 'balance': 1475.42},
    {'date': '01-08-2025', 'desc': 'LEMBAGA HASIL DALAM NEGERI MAL', 'inward': 0, 'outward': 993.72, 'balance': 481.70},
    {'date': '04-09-2025', 'desc': 'NKR ADVISORS SDN. BHD.', 'inward': 4481.84, 'outward': 0, 'balance': 4963.54},
    {'date': '04-09-2025', 'desc': 'OKL TAXATION SERVICES SDN. BHD.', 'inward': 0, 'outward': 1404.00, 'balance': 3559.54},
    {'date': '07-09-2025', 'desc': 'PREM SUBRAMANIAN PILLAI', 'inward': 0, 'outward': 1130.25, 'balance': 2429.29},
    {'date': '17-09-2025', 'desc': 'NKR ADVISORS SDN. BHD.', 'inward': 18771.22, 'outward': 0, 'balance': 21200.51},
    {'date': '17-09-2025', 'desc': 'BALAJI GOPAL', 'inward': 0, 'outward': 20000.00, 'balance': 1200.51},
]

# UI balances (from the image, newest to oldest)
ui_balances = [
    21200.51,  # Newest: BALAJI GOPAL
    1200.51,   # NKR ADVISORS SDN. BHD.
    2429.29,   # PREM SUBRAMANIAN PILLAI
    3559.54,   # OKL TAXATION SERVICES SDN. BHD.
    4963.54,   # NKR ADVISORS SDN. BHD.
    481.70,    # LEMBAGA HASIL DALAM NEGERI MAL
    1475.42,   # LEMBAGA HASIL DALAM NEGERI MAL
    15671.42,  # Oldest: RTR TAX SERVICES SDN BHD
]

print("=" * 80)
print("BALANCE VERIFICATION")
print("=" * 80)

# Reverse UI balances to match Excel order (oldest to newest)
ui_balances_reversed = list(reversed(ui_balances))

print("\nComparing Excel vs UI Balances (oldest to newest):")
print("-" * 80)
print(f"{'Transaction':<40} {'Excel Balance':<15} {'UI Balance':<15} {'Match':<10}")
print("-" * 80)

all_match = True
for i, excel_txn in enumerate(excel_data):
    excel_bal = excel_txn['balance']
    ui_bal = ui_balances_reversed[i] if i < len(ui_balances_reversed) else 0
    match = "✅" if abs(excel_bal - ui_bal) < 0.01 else "❌"
    if abs(excel_bal - ui_bal) >= 0.01:
        all_match = False
    
    desc = excel_txn['desc'][:38]
    print(f"{desc:<40} ₹{excel_bal:>12.2f}  ₹{ui_bal:>12.2f}  {match}")

print("-" * 80)

if all_match:
    print("\n✅ SUCCESS: All UI balances match Excel file!")
else:
    print("\n❌ MISMATCH: Some balances don't match!")

# Verify calculation logic
print("\n" + "=" * 80)
print("CALCULATION VERIFICATION")
print("=" * 80)

# Calculate opening balance from first transaction
first = excel_data[0]
opening_balance = first['balance'] - first['inward'] + abs(first['outward'])
print(f"\nOpening Balance Calculation:")
print(f"  First transaction balance (after): ₹{first['balance']}")
print(f"  First transaction inward: ₹{first['inward']}")
print(f"  First transaction outward: ₹{first['outward']}")
print(f"  Opening balance (before first): ₹{opening_balance}")

print(f"\nRunning Balance Calculation (verifying Excel balances):")
print(f"{'Transaction':<35} {'Prev':<12} {'Inward':<12} {'Outward':<12} {'Calc':<12} {'Excel':<12} {'Match':<10}")
print("-" * 100)

current_balance = opening_balance
all_calc_match = True

for txn in excel_data:
    prev_balance = current_balance
    inward = txn['inward']
    outward = abs(txn['outward'])  # Make positive for calculation
    # Calculate: Previous + Inward - Outward
    new_balance = prev_balance + inward - outward
    expected = txn['balance']
    match = "✅" if abs(new_balance - expected) < 0.01 else "❌"
    if abs(new_balance - expected) >= 0.01:
        all_calc_match = False
    
    desc = txn['desc'][:33]
    print(f"{desc:<35} ₹{prev_balance:>10.2f}  ₹{inward:>10.2f}  ₹{outward:>10.2f}  ₹{new_balance:>10.2f}  ₹{expected:>10.2f}  {match}")
    current_balance = new_balance

print("-" * 100)

if all_calc_match:
    print("\n✅ SUCCESS: Calculation logic is correct!")
else:
    print("\n❌ FAILURE: Calculation logic has errors!")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"UI vs Excel Match: {'✅ PASSED' if all_match else '❌ FAILED'}")
print(f"Calculation Logic: {'✅ PASSED' if all_calc_match else '❌ FAILED'}")

if all_match and all_calc_match:
    print("\n🎉 All verifications passed! Your balances are correct!")
else:
    print("\n⚠️ Some verifications failed. Please review the output above.")

