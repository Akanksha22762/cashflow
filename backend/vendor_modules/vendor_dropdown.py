import os
from typing import Dict, Any, Tuple, List

import pandas as pd

from .session_utils import resolve_session_ids
from .vendor_cache import get_cached_vendor_data

def _load_bank_df(uploaded_bank_df, uploaded_data, data_folder) -> pd.DataFrame:
    if uploaded_bank_df is not None and not uploaded_bank_df.empty:
        print(f"📁 Using uploaded bank data: {len(uploaded_bank_df)} rows")
        return uploaded_bank_df
    if uploaded_data and uploaded_data.get('bank_df') is not None:
        print(f"📁 Using stored bank data: {len(uploaded_data['bank_df'])} rows")
        return uploaded_data['bank_df']
    bank_path = os.path.join(data_folder, 'bank_data_processed.xlsx')
    if os.path.exists(bank_path):
        print("📁 Using fallback processed file: bank_data_processed.xlsx")
        return pd.read_excel(bank_path)
    return None


def get_vendor_dropdown_data(
    uploaded_bank_df,
    uploaded_data: Dict[str, Any],
    data_folder: str,
    db_manager,
    session,
    database_available: bool,
    state_manager=None,
) -> Tuple[Dict[str, Any], int]:
    bank_df = _load_bank_df(uploaded_bank_df, uploaded_data, data_folder)
    if bank_df is None:
        return ({
            'success': False,
            'error': 'No bank data available. Please upload a bank statement first.'
        }, 400)

    db_vendor_stats = None
    session_id, file_id = resolve_session_ids(session, state_manager, db_manager)
    
    vendors: List[str] = []
    db_vendor_stats = None
    
    # ✅ PRIORITY 1: Check if current uploaded DataFrame has Assigned_Vendor column (most recent data)
    if 'Assigned_Vendor' in bank_df.columns:
        unique_vendors = bank_df['Assigned_Vendor'].dropna().unique().tolist()
        vendors = [v for v in unique_vendors if v and v.strip()]
        if vendors:
            vendors = ['All'] + sorted(vendors)
            print(f"✅ Using {len(vendors)} vendors from CURRENT uploaded data (DataFrame)")
            
            # Still fetch stats from database for vendor details (same session only)
            if database_available and db_manager and session_id:
                try:
                    db_vendor_stats = db_manager.fetch_vendor_entities(session_id)
                except Exception as db_error:
                    print(f"⚠️ Vendor stats lookup failed: {db_error}")
                    db_vendor_stats = None
    
    # ✅ PRIORITY 2: Only check database cache if CURRENT file_id matches
    # This prevents showing vendors from old/previous sessions
    if not vendors and database_available and db_manager and file_id:
        try:
            # Try file-level cache first (most accurate - tied to specific file)
            cached = get_cached_vendor_data(db_manager, file_id)
            if cached and cached.get('summaries'):
                db_vendor_stats = cached['summaries']
                vendor_names = [row['vendor_name'] for row in db_vendor_stats if row.get('vendor_name')]
                vendors = ['All'] + sorted([v for v in vendor_names if v != 'All'])
                print(f"✅ Using {len(vendors)} vendors from CURRENT file cache (file_id: {file_id})")
            elif cached and cached.get('transactions'):
                vendor_names = ['All'] + sorted([v for v in cached['transactions'].keys() if v and v != 'All'])
                vendors = vendor_names
                print(f"✅ Using {len(vendors)} vendors from CURRENT file transactions cache")
        except Exception as cache_error:
            print(f"⚠️ Vendor cache lookup failed: {cache_error}")
    
    # ✅ PRIORITY 3: Only if no file_id match, try session-level (but warn it might be old data)
    if not vendors and database_available and db_manager and session_id:
        try:
            db_vendor_stats = db_manager.fetch_vendor_entities(session_id)
            if db_vendor_stats:
                vendor_names = [row['vendor_name'] for row in db_vendor_stats if row.get('vendor_name')]
                vendors = ['All'] + sorted([v for v in vendor_names if v != 'All'])
                print(f"⚠️ Using {len(vendors)} vendors from database cache (session {session_id}) - may be from previous upload")
        except Exception as db_error:
            print(f"⚠️ Vendor DB lookup failed: {db_error}")
            db_vendor_stats = None
    # ✅ FINAL FALLBACK: If still no vendors, just show 'All' (don't mix old session data)
    if not vendors:
        print("⚠️ No vendors found for current file. Showing 'All' only. Please click 'Extract Vendors' to extract vendors for this file.")
        vendors = ['All']  # Always provide at least "All" option

    # Ensure vendors is never empty - always have "All" option
    if not vendors or len(vendors) == 0:
        print("⚠️ Vendors list was empty, adding 'All' as default")
        vendors = ['All']
    
    # Ensure "All" is always first
    if 'All' not in vendors:
        vendors.insert(0, 'All')
    
    transaction_types = []
    if 'Category' in bank_df.columns:
        transaction_types = bank_df['Category'].dropna().unique().tolist()
    if not transaction_types:
        transaction_types = ['Operating Activities', 'Investing Activities', 'Financing Activities']

    # Ensure response is always valid JSON
    response = {
        'success': True,
        'vendors': vendors if vendors else ['All'],
        'vendor_stats': db_vendor_stats if db_vendor_stats else [],
        'transaction_types': transaction_types if transaction_types else [],
        'total_transactions': len(bank_df) if bank_df is not None else 0
    }
    
    print(f"📊 Returning vendor dropdown data: {len(response['vendors'])} vendors, {response['total_transactions']} transactions")
    
    return (response, 200)

