"""
Comprehensive Cash Flow Report Generator
========================================
Generates detailed cash flow reports with all transactions, classifications, and summaries.
"""

import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime


def generate_comprehensive_cashflow_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive cash flow report from transactions.
    
    Args:
        df: DataFrame with transactions (must have Date, Description, Amount, Category columns)
        
    Returns:
        Dictionary containing full cash flow report
    """
    # Ensure required columns exist - use Inward_Amount and Outward_Amount directly
    # Check for Date column (might be _date after preprocessing)
    if 'Date' not in df.columns and '_date' in df.columns:
        df['Date'] = df['_date']
    
    # Check for Description column (might be _combined_description)
    if 'Description' not in df.columns and '_combined_description' in df.columns:
        df['Description'] = df['_combined_description']
    
    required_columns = ['Date', 'Description', 'Inward_Amount', 'Outward_Amount']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}. Amount column is not used - use Inward_Amount and Outward_Amount directly.")
    
    # Ensure Category column exists
    if 'Category' not in df.columns:
        df['Category'] = 'Operating Activities'  # Default
    
    # Ensure Inward_Amount and Outward_Amount are numeric
    df['Inward_Amount'] = pd.to_numeric(df['Inward_Amount'], errors='coerce').fillna(0.0)
    df['Outward_Amount'] = pd.to_numeric(df['Outward_Amount'], errors='coerce').fillna(0.0)
    
    # Mark transactions needing more information
    if 'Needs_More_Info' not in df.columns:
        df['Needs_More_Info'] = df['Category'].apply(
            lambda x: str(x).strip() == 'More information needed'
        )
    
    # Prepare individual transactions
    transactions = []
    for idx, row in df.iterrows():
        transaction = {
            'date': str(row.get('Date', '')),
            'description': str(row.get('Description', '')),
            'inward_amount': float(row.get('Inward_Amount', 0.0)),
            'outward_amount': float(row.get('Outward_Amount', 0.0)),
            'category': str(row.get('Category', 'Operating Activities')),
            'closing_balance': float(row.get('Closing_Balance', 0.0)),
            'needs_more_info': bool(row.get('Needs_More_Info', False))
        }
        transactions.append(transaction)
    
    # Calculate category-wise totals
    category_totals = {}
    categories = ['Operating Activities', 'Investing Activities', 'Financing Activities', 'More information needed']
    
    for category in categories:
        cat_df = df[df['Category'] == category]
        if len(cat_df) > 0:
            # Use Inward_Amount and Outward_Amount directly instead of Amount
            inflows = float(cat_df['Inward_Amount'].fillna(0).sum())
            outflows = float(cat_df['Outward_Amount'].fillna(0).apply(lambda x: abs(float(x)) if pd.notna(x) else 0.0).sum())
            net = float(inflows - outflows)
            
            category_totals[category] = {
                'inflows': inflows,
                'outflows': outflows,
                'net_cash_flow': net,
                'transaction_count': int(len(cat_df))
            }
        else:
            category_totals[category] = {
                'inflows': 0.0,
                'outflows': 0.0,
                'net_cash_flow': 0.0,
                'transaction_count': 0
            }
    
    # Calculate overall totals using Inward_Amount and Outward_Amount directly
    total_inflows = float(df['Inward_Amount'].fillna(0).sum())
    total_outflows = float(df['Outward_Amount'].fillna(0).apply(lambda x: abs(float(x)) if pd.notna(x) else 0.0).sum())
    overall_net_cash_flow = float(total_inflows - total_outflows)
    
    # Calculate final closing balance
    # Use the last transaction's closing balance if available
    if 'Closing_Balance' in df.columns and df['Closing_Balance'].notna().any():
        # Get the last non-null closing balance (should be the most recent transaction)
        closing_balances = df['Closing_Balance'].dropna()
        if len(closing_balances) > 0:
            final_closing_balance = float(closing_balances.iloc[-1])
        else:
            # Fallback: calculate from transactions if not provided
            final_closing_balance = overall_net_cash_flow
    else:
        # Calculate from transactions if not provided
        final_closing_balance = overall_net_cash_flow
    
    # Also calculate opening balance (first transaction's balance before it, or first closing balance)
    if 'Closing_Balance' in df.columns and df['Closing_Balance'].notna().any():
        closing_balances = df['Closing_Balance'].dropna()
        if len(closing_balances) > 0:
            # Opening balance = first closing balance - first transaction net amount
            first_idx = closing_balances.index[0]
            first_inward = float(df.loc[first_idx, 'Inward_Amount']) if pd.notna(df.loc[first_idx, 'Inward_Amount']) else 0.0
            first_outward = float(df.loc[first_idx, 'Outward_Amount']) if pd.notna(df.loc[first_idx, 'Outward_Amount']) else 0.0
            first_transaction_net = first_inward - abs(first_outward)
            first_closing_balance = float(closing_balances.iloc[0])
            opening_balance = first_closing_balance - first_transaction_net
        else:
            opening_balance = 0.0
    else:
        # If no closing balance column, opening balance is 0 or calculated from net
        opening_balance = final_closing_balance - overall_net_cash_flow if final_closing_balance != overall_net_cash_flow else 0.0
    
    # Get transactions needing more information
    needs_more_info = df[df['Needs_More_Info'] == True] if 'Needs_More_Info' in df.columns else pd.DataFrame()
    items_needing_info = []
    for idx, row in needs_more_info.iterrows():
        inward_amt = row.get('Inward_Amount', None)
        outward_amt = row.get('Outward_Amount', None)
        items_needing_info.append({
            'date': str(row.get('Date', '')),
            'description': str(row.get('Description', '')),
            'inward_amount': float(inward_amt) if pd.notna(inward_amt) and inward_amt != 0 else None,
            'outward_amount': float(outward_amt) if pd.notna(outward_amt) and outward_amt != 0 else None,
            'category': str(row.get('Category', 'More information needed'))
        })
    
    # Build comprehensive report
    report = {
        'generated_at': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
        'transactions': transactions,
        'cash_flow_summary': {
            'operating_activities': category_totals.get('Operating Activities', {}),
            'investing_activities': category_totals.get('Investing Activities', {}),
            'financing_activities': category_totals.get('Financing Activities', {}),
            'more_information_needed': category_totals.get('More information needed', {}),
            'overall': {
                'opening_balance': opening_balance,
                'total_inflows': total_inflows,
                'total_outflows': total_outflows,
                'net_cash_flow': overall_net_cash_flow,
                'closing_balance': final_closing_balance,
                'final_closing_balance': final_closing_balance,  # Keep for backward compatibility
                'total_transactions': int(len(df))
            }
        },
        'items_needing_more_information': items_needing_info,
        'summary_statistics': {
            'total_transactions': int(len(df)),
            'transactions_needing_info': int(len(items_needing_info)),
            'categorized_transactions': int(len(df) - len(items_needing_info)),
            'date_range': {
                'start': str(df['Date'].min()) if 'Date' in df.columns and df['Date'].notna().any() else '',
                'end': str(df['Date'].max()) if 'Date' in df.columns and df['Date'].notna().any() else ''
            }
        }
    }
    
    return report

