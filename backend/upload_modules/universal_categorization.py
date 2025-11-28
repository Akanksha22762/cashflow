"""
Universal Categorization Module
================================
Handles universal AI/ML categorization of any dataset.
Moved from app.py to reduce load on main application file.
"""

import pandas as pd
import re
from typing import Optional, Dict, Any


def enhanced_standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names and prepare data for categorization.
    """
    df = df.copy()
    
    # Normalize column names (handle spaces, case, etc.)
    column_mapping = {}
    for col in df.columns:
        col_lower = str(col).lower().strip()
        # Map common variations
        if 'date' in col_lower and 'time' not in col_lower:
            column_mapping[col] = '_date'
        elif 'description' in col_lower or 'desc' in col_lower or 'narration' in col_lower:
            column_mapping[col] = 'Description'
        elif 'amount' in col_lower and 'inward' not in col_lower and 'outward' not in col_lower:
            column_mapping[col] = '_amount'
        elif 'inward' in col_lower and 'amount' in col_lower:
            column_mapping[col] = 'Inward_Amount'
        elif 'outward' in col_lower and 'amount' in col_lower:
            column_mapping[col] = 'Outward_Amount'
        elif 'balance' in col_lower:
            column_mapping[col] = 'Closing_Balance'
    
    # Apply mapping
    df = df.rename(columns=column_mapping)
    
    # Removed _combined_description - just use Description column directly
    
    if '_amount' not in df.columns:
        # Use Inward_Amount and Outward_Amount directly - no fallback to Amount
        if 'Inward_Amount' in df.columns and 'Outward_Amount' in df.columns:
            # _amount = Inward - Outward (Outward is already negative, so this gives net)
            df['_amount'] = df['Inward_Amount'].fillna(0) + df['Outward_Amount'].fillna(0)
        else:
            raise ValueError("Inward_Amount and Outward_Amount columns are required. No fallback to Amount column.")
    
    if '_date' not in df.columns:
        if 'Date' in df.columns:
            df['_date'] = df['Date']
        else:
            df['_date'] = pd.NaT
    
    return df


def detect_data_type(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect the type of data in the DataFrame.
    """
    return {
        'data_type': 'bank_statement',
        'has_dates': 'Date' in df.columns or '_date' in df.columns,
        'has_amounts': 'Amount' in df.columns or '_amount' in df.columns,
        'transaction_count': len(df)
    }




def universal_categorize_any_dataset(df):
    """
    Universal categorization using 100% AI/ML approach with OpenAI.
    """
    print("🤖 Starting Universal AI/ML-Based Categorization with OpenAI...")
    
    # Step 1: Minimal processing to preserve original data
    df_processed = enhanced_standardize_columns(df.copy())
    
    # Step 2: Detect data type for context
    context = detect_data_type(df_processed)
    print(f"🔍 Detected data type: {context['data_type']}")
    
    # Step 3: Hybrid AI/ML categorization with OpenAI + XGBoost
    categories = []
    
    # Use Description column directly - no combining
    # Remove duplicate columns first
    df_processed = df_processed.loc[:, ~df_processed.columns.duplicated()]
    
    # Ensure Description column exists
    if 'Description' not in df_processed.columns:
        raise ValueError("Description column is required. Please ensure your file has a Description column.")
    
    # Clean "nan" strings from descriptions
    descriptions = df_processed['Description'].fillna('').astype(str).str.replace('nan', '', regex=False).str.replace('NaN', '', regex=False).str.strip().tolist()
    
    amounts = df_processed['_amount'].tolist()
    
    # Extract inward and outward amounts if available
    inward_amounts = None
    outward_amounts = None
    balances = None
    
    if 'Inward_Amount' in df_processed.columns:
        inward_amounts = df_processed['Inward_Amount'].fillna(0).tolist()
    elif 'Inward Amount' in df_processed.columns:
        inward_amounts = df_processed['Inward Amount'].fillna(0).tolist()
    
    if 'Outward_Amount' in df_processed.columns:
        # Outward amounts are negative in the file (that's correct - debits are negative)
        # For OpenAI prompt, show as positive numbers for clarity
        if 'Outward_Amount_Display' in df_processed.columns:
            outward_amounts = df_processed['Outward_Amount_Display'].fillna(0).tolist()
        else:
            # Convert negative outward amounts to positive for display
            outward_amounts = df_processed['Outward_Amount'].fillna(0).apply(lambda x: abs(float(x)) if pd.notna(x) else 0.0).tolist()
    elif 'Outward Amount' in df_processed.columns:
        # Convert negative to positive for display
        outward_amounts = df_processed['Outward Amount'].fillna(0).apply(lambda x: abs(float(x)) if pd.notna(x) else 0.0).tolist()
    
    if 'Closing_Balance' in df_processed.columns:
        balances = df_processed['Closing_Balance'].fillna(0).tolist()
    elif 'Balance' in df_processed.columns:
        balances = df_processed['Balance'].fillna(0).tolist()
    
    # Removed verbose debug messages for sample descriptions
    
    print(f"🤖 Categorizing {len(descriptions)} transactions with OpenAI + XGBoost hybrid models...")
    print(f"🚀 PRODUCTION MODE: Processing all transactions for maximum accuracy")
    print(f"📊 Using full transaction details: descriptions, amounts, inward/outward amounts, and balances")
    
    # First try OpenAI categorization for better accuracy
    openai_success_count = 0
    try:
        from openai_integration import openai_integration, check_openai_availability
        if check_openai_availability():
            print("🧠 Using OpenAI for intelligent categorization with full transaction context...")
            
            # Use the enhanced categorize_transactions method with all details
            categories = openai_integration.categorize_transactions(
                descriptions=descriptions,
                amounts=amounts,
                inward_amounts=inward_amounts,
                outward_amounts=outward_amounts,
                balances=balances
            )
            
            # Verify categories were generated
            if categories and len(categories) == len(descriptions):
                openai_success_count = sum(1 for cat in categories if cat and str(cat).strip() != '')
                print(f"✅ OpenAI successfully categorized {openai_success_count}/{len(descriptions)} transactions with full context")
                
                # Add categories to DataFrame
                df_processed['Category'] = categories
                print(f"🔧 Added {len(categories)} categories to DataFrame")
                return df_processed
            else:
                raise ValueError(f"OpenAI returned {len(categories) if categories else 0} categories, expected {len(descriptions)}")
            
    except Exception as e:
        print(f"❌ OpenAI categorization failed: {e}")
        raise RuntimeError(f"Failed to categorize transactions: {e}")

