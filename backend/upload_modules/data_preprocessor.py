"""
Data Preprocessor Module
========================
Handles data preprocessing: sorting, normalization, validation, amount cleaning.
"""

import pandas as pd
import re
from typing import Dict, Any


def clean_amount(amount_str: Any) -> float:
    """
    Clean amount string by removing commas, currency symbols, and formatting.
    Handles negative amounts in strings like "-20000" or "(20000)".
    
    Args:
        amount_str: Amount string or number (e.g., "₹1,23,456.78", "$1,234.56", "-20000", 1234.56)
        
    Returns:
        Cleaned numeric amount
    """
    if pd.isna(amount_str):
        return 0.0
    
    # If already numeric, return as is
    if isinstance(amount_str, (int, float)):
        return float(amount_str)
    
    # Convert to string and strip whitespace
    amount_str = str(amount_str).strip()
    
    if not amount_str or amount_str.lower() in ['nan', 'none', '']:
        return 0.0
    
    # Remove currency symbols (₹, $, €, £, etc.)
    amount_str = re.sub(r'[₹$€£¥]', '', amount_str)
    
    # Remove commas
    amount_str = amount_str.replace(',', '')
    
    # Handle parentheses notation for negative amounts (e.g., "(20000)" -> "-20000")
    if '(' in amount_str and ')' in amount_str:
        amount_str = amount_str.replace('(', '-').replace(')', '')
    
    # Extract numeric value (including decimals and negative signs)
    # This regex handles: -20000, 20000, -20000.50, 20000.50, etc.
    match = re.search(r'-?\d+\.?\d*', amount_str)
    if match:
        try:
            value = float(match.group())
            # Handle case where outward amounts are already negative (keep as is)
            return value
        except ValueError:
            return 0.0
    
    return 0.0


def preprocess_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Preprocess DataFrame: sort, normalize, validate, clean amounts.
    Handles bank statement format with separate Inward Amount and Outward Amount columns.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with processed DataFrame and metadata
    """
    # Remove duplicate columns first to avoid DataFrame issues
    df = df.loc[:, ~df.columns.duplicated()].copy()
    original_count = len(df)
    df = df.copy()
    
    # Remove duplicate columns first (in case of any duplicates)
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Normalize column names (handle spaces and case)
    column_mapping = {}
    for col in df.columns:
        col_lower = str(col).lower().strip()
        # Map common variations
        if 'inward' in col_lower and 'amount' in col_lower:
            column_mapping[col] = 'Inward_Amount'
        elif 'outward' in col_lower and 'amount' in col_lower:
            column_mapping[col] = 'Outward_Amount'
        elif col_lower == 'balance':
            column_mapping[col] = 'Closing_Balance'
        elif 'date' in col_lower and 'time' not in col_lower:
            column_mapping[col] = 'Date'
        elif 'description' in col_lower or 'desc' in col_lower:
            column_mapping[col] = 'Description'
    
    # Apply column mapping
    df = df.rename(columns=column_mapping)
    
    # Remove duplicate columns after renaming (in case multiple columns mapped to same name)
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Ensure Description column exists - use it as-is, don't combine
    if 'Description' not in df.columns:
        # Try to find a description-like column
        for col in df.columns:
            col_lower = str(col).lower()
            if 'description' in col_lower or 'desc' in col_lower:
                df['Description'] = df[col].fillna('').astype(str).str.replace('nan', '', regex=False).str.strip()
                break
    
    # Clean Description column - remove "nan" strings
    if 'Description' in df.columns:
        df['Description'] = df['Description'].fillna('').astype(str).str.replace('nan', '', regex=False).str.replace('NaN', '', regex=False).str.strip()
    
    # Clean amount columns (handle case-insensitive matching)
    amount_columns = ['Amount', 'Inward_Amount', 'Outward_Amount', 'Closing_Balance', 
                     'Credit', 'Debit', 'Balance', 'amount', 'credit', 'debit', 'balance',
                     'inward amount', 'outward amount']
    
    for col in df.columns:
        col_lower = str(col).lower()
        if any(amt_col.lower() in col_lower for amt_col in amount_columns):
            print(f"🧹 Cleaning amounts in column: {col}")
            # Handle empty cells - they should be 0, not NaN
            # First convert to string, then clean
            # Clean NaN values - replace 'nan' strings with empty string
            # Ensure we're working with a Series, not DataFrame
            col_data = df[col]
            if isinstance(col_data, pd.DataFrame):
                # If duplicate columns exist, take the first one
                col_data = col_data.iloc[:, 0]
            df[col] = col_data.fillna('').astype(str).str.replace('nan', '', regex=False).str.replace('NaN', '', regex=False).str.strip()
            df[col] = df[col].apply(clean_amount)
    
    # Normalize Inward_Amount and Outward_Amount column names (handle spaces)
    
    if 'Inward Amount' in df.columns and 'Inward_Amount' not in df.columns:
        df['Inward_Amount'] = df['Inward Amount']
        # Drop the original if it exists
        if 'Inward Amount' in df.columns and 'Inward_Amount' in df.columns:
            df = df.drop(columns=['Inward Amount'])
    
    if 'Outward Amount' in df.columns and 'Outward_Amount' not in df.columns:
        df['Outward_Amount'] = df['Outward Amount']
        # Drop the original if it exists
        if 'Outward Amount' in df.columns and 'Outward_Amount' in df.columns:
            df = df.drop(columns=['Outward Amount'])
    
    # Ensure Inward_Amount and Outward_Amount exist - no fallback to Amount
    # These are required columns - no fallback logic
    if 'Inward_Amount' not in df.columns:
        raise ValueError("Inward_Amount column is required. No fallback to Amount column.")
    if 'Outward_Amount' not in df.columns:
        raise ValueError("Outward_Amount column is required. No fallback to Amount column.")
    
    # Ensure Inward_Amount and Outward_Amount are numeric - no Amount column
    df['Inward_Amount'] = pd.to_numeric(df['Inward_Amount'], errors='coerce').fillna(0.0)
    df['Outward_Amount'] = pd.to_numeric(df['Outward_Amount'], errors='coerce').fillna(0.0)
    df['Outward_Amount'] = df['Outward_Amount'].apply(lambda x: -abs(x))  # force all outward to negative

    
    # CRITICAL: Calculate balance BEFORE sorting (balance is a running total that needs chronological order)
    # Step 1: Ensure Date column exists and normalize it (but don't sort yet)
    if 'Date' not in df.columns:
        # Try to find date column
        for col in df.columns:
            if 'date' in str(col).lower() and 'time' not in str(col).lower():
                df['Date'] = df[col]
                break
    
    # Step 2: Sort ASCENDING (oldest first) ONLY for balance calculation
    # Create a copy sorted ascending for balance calculation
    # Step 2: Sort ASCENDING (oldest first) — use Date + Time if available
    df_asc = df.copy()
    # Combine Date + Time into DateTime for accurate sorting
    if 'Date' in df_asc.columns:
        if 'Time' in df_asc.columns:
            df_asc['DateTime'] = pd.to_datetime(
                df_asc['Date'].astype(str) + ' ' + df_asc['Time'].astype(str),
                errors='coerce',
                dayfirst=True  # DD-MM-YYYY format
            )
        else:
            df_asc['DateTime'] = pd.to_datetime(
                df_asc['Date'],
                errors='coerce',
                dayfirst=True  # DD-MM-YYYY format
            )

    # Sort by combined DateTime
        df_asc = df_asc.sort_values('DateTime', ascending=True).reset_index(drop=True)

        print(f"📊 Sorted {len(df_asc)} transactions ASCENDING using DateTime (oldest first)")
    else:
        print("⚠️ No Date column available; using original order for balance calculation.")

    # Step 3: Calculate opening balance from first transaction (oldest, after ASC sorting)
    # The Excel Balance column contains the closing balance AFTER each transaction
    # We need to calculate the opening balance BEFORE the first transaction
    opening_balance = 0.0
    # Check for Balance column (might be named 'Balance' or 'Closing_Balance')
    balance_col = None
    for col in df_asc.columns:
        if col.lower() in ['balance', 'closing_balance']:
            balance_col = col
            break
    
    if balance_col and len(df_asc) > 0:
        df_asc[balance_col] = pd.to_numeric(df_asc[balance_col], errors='coerce')
        if pd.notna(df_asc[balance_col].iloc[0]):
            first_balance = float(df_asc[balance_col].iloc[0])
            first_inward = float(df_asc.iloc[0]['Inward_Amount'])
            first_outward = float(df_asc.iloc[0]['Outward_Amount'])
            # Get absolute value of outward for calculation
            first_outward_abs = abs(first_outward) if first_outward != 0 else 0.0
            # Calculate opening balance BEFORE first transaction
            # Balance_after = Balance_before + Inward - Outward
            # So: Balance_before = Balance_after - Inward + Outward
            opening_balance = first_balance - first_inward + first_outward_abs

            print(f"📊 First transaction (oldest): Inward=₹{first_inward}, Outward=₹{first_outward} (abs: ₹{first_outward_abs}), Balance_after=₹{first_balance}")
            print(f"📊 Calculated opening balance (before first): ₹{opening_balance}")
    else:
        print(f"⚠️ Warning: No Balance column found. Using opening_balance = 0.0")
    
    # Step 4: Calculate running balance on ASC-sorted data (oldest → newest)
    _recalculate_balances(df_asc, opening_balance)
    
    # Step 5: Map calculated balances back to original DataFrame, then sort DESC for display
    if 'Date' in df.columns:
        # Normalize dates in original df
        normalized_dates = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)  # DD-MM-YYYY format
        if normalized_dates.notna().any():
            df['Date'] = normalized_dates
        
        # Create DateTime column in original df for proper sorting (same as df_asc)
        if 'Time' in df.columns:
            df['DateTime'] = pd.to_datetime(
                df['Date'].astype(str) + ' ' + df['Time'].astype(str),
                errors='coerce',
                dayfirst=True
            )
        else:
            df['DateTime'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
        
        # Sort original by DateTime ASCENDING to match df_asc order exactly
        df_sorted_asc = df.sort_values('DateTime', ascending=True).reset_index(drop=True)
        
        # Map balances from df_asc to df_sorted_asc (they're in same order now)
        df_sorted_asc['Closing_Balance'] = df_asc['Closing_Balance'].values
        if 'Balance' in df_sorted_asc.columns:
            df_sorted_asc['Balance'] = df_asc['Closing_Balance'].values
        
        # NOW sort DESCENDING by DateTime for UI display (newest first)
        df = df_sorted_asc.sort_values('DateTime', ascending=False).reset_index(drop=True)
        print(f"✅ Balance calculated on ASC data, then sorted DESC for display (newest first)")
    else:
        # No date column, use as-is
        df['Closing_Balance'] = df_asc['Closing_Balance'].values
        if 'Balance' in df.columns:
            df['Balance'] = df_asc['Closing_Balance'].values
    
    print(f"🚀 PROCESSING: Dataset size: {len(df)} transactions (full dataset)")
    
    return {
        'dataframe': df,
        'original_count': original_count,
        'processed_count': len(df)
    }


def _recalculate_balances(df: pd.DataFrame, opening_balance: float = 0.0) -> None:
    """
    Recalculate balances from Inward/Outward amounts starting from opening_balance.
    opening_balance is the balance BEFORE the first transaction.
    Formula: New Balance = Previous Balance + Inward + Outward
    (Outward amounts are already negative after cleaning, so adding them subtracts)
    """
    balances = []
    
    for idx in range(len(df)):
        if idx == 0:
            prev_balance = opening_balance
        else:
            prev_balance = balances[-1]
        
        inward = float(df.iloc[idx]['Inward_Amount'])
        outward = float(df.iloc[idx]['Outward_Amount'])
        # Outward amounts are already negative after cleaning (e.g., -626.40)
        # So: balance + inward + (-outward) = balance + inward - outward
        new_balance = prev_balance + inward + outward
        balances.append(new_balance)
        
        # Debug output for first few transactions
        if idx < 3:
            print(f"   Transaction {idx+1}: Prev={prev_balance:.2f}, Inward={inward:.2f}, Outward={outward:.2f}, New={new_balance:.2f}")
    
    df['Closing_Balance'] = balances
    if 'Balance' in df.columns:
        df['Balance'] = balances
    
    print(f"✅ Recalculated {len(balances)} balances starting from opening balance ₹{opening_balance}")
    print(f"   Final balance: ₹{balances[-1] if balances else 0}")

