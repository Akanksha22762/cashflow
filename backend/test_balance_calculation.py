"""
Test Balance Calculation Logic
==============================
Tests the balance calculation to ensure it matches Excel calculations.
"""

import pandas as pd
import sys
import os

# Fix encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from upload_modules.data_preprocessor import preprocess_dataframe, _recalculate_balances


def test_balance_calculation():
    """Test balance calculation with sample data matching Excel format"""
    
    # Sample data matching the Excel file structure
    # Dates are in DD-MM-YYYY format, sorted oldest to newest
    data = {
        'Date': ['31-07-2025', '01-08-2025', '01-08-2025', '04-09-2025', '04-09-2025', 
                 '07-09-2025', '17-09-2025', '17-09-2025'],
        'Time': ['10:42:26', '07:00:12', '15:15:36', '09:30:00', '14:20:00',
                 '11:45:00', '08:30:00', '16:00:00'],
        'Description': [
            'RTR TAX SERVICES SDN BHD',
            'LEMBAGA HASIL DALAM NEGERI MAL',
            'LEMBAGA HASIL DALAM NEGERI MAL',
            'NKR ADVISORS SDN. BHD.',
            'OKL TAXATION SERVICES SDN. BHD.',
            'PREM SUBRAMANIAN PILLAI',
            'NKR ADVISORS SDN. BHD.',
            'BALAJI GOPAL'
        ],
        'Inward Amount': [0, 0, 0, 4481.84, 0, 0, 18771.22, 0],
        'Outward Amount': [-626.40, -14196.00, -993.72, 0, -1404.00, -1130.25, 0, -20000.00],
        'Balance': [0.00, -14196.00, -15189.72, -10707.88, -12111.88, -13242.13, 5529.09, -14470.91]
    }
    
    df = pd.DataFrame(data)
    
    print("=" * 80)
    print("TEST: Balance Calculation")
    print("=" * 80)
    print("\n📊 Input Data (as from Excel):")
    print(df.to_string(index=False))
    print("\n")
    
    # Test the preprocessing
    try:
        result = preprocess_dataframe(df)
        processed_df = result['dataframe']
        
        print("\n" + "=" * 80)
        print("PROCESSED DATA (After Preprocessing):")
        print("=" * 80)
        
        # Show key columns
        display_cols = ['Date', 'Description', 'Inward_Amount', 'Outward_Amount', 'Closing_Balance']
        if all(col in processed_df.columns for col in display_cols):
            display_df = processed_df[display_cols].copy()
            # Sort by date for display
            display_df = display_df.sort_values('Date', ascending=True)
            print(display_df.to_string(index=False))
        else:
            print("Available columns:", list(processed_df.columns))
            print(processed_df.head(10).to_string())
        
        print("\n" + "=" * 80)
        print("VERIFICATION:")
        print("=" * 80)
        
        # Sort by date ascending for verification
        if 'Date' in processed_df.columns:
            processed_df_sorted = processed_df.sort_values('Date', ascending=True).reset_index(drop=True)
        else:
            processed_df_sorted = processed_df.copy()
        
        # Expected balances (from Excel, oldest to newest)
        expected_balances = [0.00, -14196.00, -15189.72, -10707.88, -12111.88, -13242.13, 5529.09, -14470.91]
        
        print("\nComparing Calculated vs Expected Balances:")
        print("-" * 80)
        print(f"{'Transaction':<40} {'Expected':<15} {'Calculated':<15} {'Match':<10}")
        print("-" * 80)
        
        all_match = True
        if 'Closing_Balance' in processed_df_sorted.columns:
            for idx, row in processed_df_sorted.iterrows():
                if idx < len(expected_balances):
                    desc = str(row.get('Description', ''))[:38]
                    expected = expected_balances[idx]
                    calculated = float(row['Closing_Balance'])
                    match = "✅" if abs(expected - calculated) < 0.01 else "❌"
                    if abs(expected - calculated) >= 0.01:
                        all_match = False
                    print(f"{desc:<40} ₹{expected:>12.2f}  ₹{calculated:>12.2f}  {match}")
        
        print("-" * 80)
        if all_match:
            print("\n✅ SUCCESS: All balances match expected values!")
        else:
            print("\n❌ FAILURE: Some balances don't match expected values!")
            print("\nDebugging info:")
            print(f"Opening balance calculation:")
            if 'Balance' in processed_df_sorted.columns and len(processed_df_sorted) > 0:
                first_balance = float(processed_df_sorted.iloc[0]['Balance'])
                first_inward = float(processed_df_sorted.iloc[0]['Inward_Amount'])
                first_outward = float(processed_df_sorted.iloc[0]['Outward_Amount'])
                print(f"  First transaction balance: ₹{first_balance}")
                print(f"  First transaction inward: ₹{first_inward}")
                print(f"  First transaction outward: ₹{first_outward}")
                opening = first_balance - first_inward + abs(first_outward) if first_outward != 0 else first_balance - first_inward
                print(f"  Calculated opening balance: ₹{opening}")
        
        return all_match
        
    except Exception as e:
        print(f"\n❌ ERROR during processing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_manual_calculation():
    """Manual step-by-step calculation to verify logic"""
    
    print("\n" + "=" * 80)
    print("MANUAL CALCULATION VERIFICATION")
    print("=" * 80)
    
    # Transaction data (oldest to newest)
    transactions = [
        {'desc': 'RTR TAX SERVICES', 'inward': 0, 'outward': 626.40, 'balance_after': 0.00},
        {'desc': 'LEMBAGA 1', 'inward': 0, 'outward': 14196.00, 'balance_after': -14196.00},
        {'desc': 'LEMBAGA 2', 'inward': 0, 'outward': 993.72, 'balance_after': -15189.72},
        {'desc': 'NKR ADVISORS', 'inward': 4481.84, 'outward': 0, 'balance_after': -10707.88},
        {'desc': 'OKL TAXATION', 'inward': 0, 'outward': 1404.00, 'balance_after': -12111.88},
        {'desc': 'PREM SUBRAMANIAN', 'inward': 0, 'outward': 1130.25, 'balance_after': -13242.13},
        {'desc': 'NKR ADVISORS 2', 'inward': 18771.22, 'outward': 0, 'balance_after': 5529.09},
        {'desc': 'BALAJI GOPAL', 'inward': 0, 'outward': 20000.00, 'balance_after': -14470.91},
    ]
    
    # Calculate opening balance from first transaction
    first = transactions[0]
    opening_balance = first['balance_after'] - first['inward'] + first['outward']
    print(f"\nOpening Balance Calculation:")
    print(f"  First transaction balance (after): ₹{first['balance_after']}")
    print(f"  First transaction inward: ₹{first['inward']}")
    print(f"  First transaction outward: ₹{first['outward']}")
    print(f"  Opening balance (before first): ₹{opening_balance}")
    
    print(f"\nRunning Balance Calculation:")
    print(f"{'Transaction':<25} {'Prev Bal':<12} {'Inward':<12} {'Outward':<12} {'New Bal':<12} {'Expected':<12} {'Match':<10}")
    print("-" * 100)
    
    current_balance = opening_balance
    all_match = True
    
    for i, txn in enumerate(transactions):
        prev_balance = current_balance
        inward = txn['inward']
        outward = txn['outward']
        # Calculate new balance: Previous + Inward - Outward
        new_balance = prev_balance + inward - outward
        expected = txn['balance_after']
        match = "✅" if abs(new_balance - expected) < 0.01 else "❌"
        if abs(new_balance - expected) >= 0.01:
            all_match = False
        
        print(f"{txn['desc']:<25} ₹{prev_balance:>10.2f}  ₹{inward:>10.2f}  ₹{outward:>10.2f}  ₹{new_balance:>10.2f}  ₹{expected:>10.2f}  {match}")
        current_balance = new_balance
    
    print("-" * 100)
    if all_match:
        print("\n✅ SUCCESS: Manual calculation matches expected values!")
    else:
        print("\n❌ FAILURE: Manual calculation doesn't match expected values!")
    
    return all_match


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("BALANCE CALCULATION TEST SUITE")
    print("=" * 80)
    
    # Run manual calculation test first
    manual_ok = test_manual_calculation()
    
    # Run full preprocessing test
    preprocessing_ok = test_balance_calculation()
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Manual Calculation Test: {'✅ PASSED' if manual_ok else '❌ FAILED'}")
    print(f"Preprocessing Test: {'✅ PASSED' if preprocessing_ok else '❌ FAILED'}")
    
    if manual_ok and preprocessing_ok:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n⚠️ Some tests failed. Please review the output above.")
        sys.exit(1)

