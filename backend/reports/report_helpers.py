"""
Helpers for building comprehensive cash-flow reports.
"""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from reports import generate_final_report
from vendor_modules import resolve_session_ids


def resolve_active_session_and_file(
    session_obj,
    state_manager,
    db_manager,
    database_available: bool,
    persistent_state_available: bool,
) -> Tuple[Optional[int], Optional[int]]:
    if not (database_available and db_manager):
        return None, None
    try:
        current_session, current_file = resolve_session_ids(
            session_obj if session_obj is not None else {},
            state_manager if persistent_state_available else None,
            db_manager,
        )
        return current_session, current_file
    except Exception as exc:
        print(f"⚠️ Unable to resolve session/file IDs: {exc}")
        return None, None


def fetch_vendor_master_from_db(db_manager, session_id, file_id):
    """Fetch vendor master data from database - non-blocking, returns None on failure"""
    if not db_manager:
        return None
    
    try:
        if session_id:
            vendor_master = db_manager.fetch_vendor_entities(session_id)
        elif file_id:
            vendor_master = db_manager.fetch_vendor_entities_by_file(file_id)
        else:
            vendor_master = None
        return vendor_master
    except Exception as exc:
        # Log but don't fail - vendor master is optional
        print(f"⚠️ Failed to fetch vendor master data (non-critical): {exc}")
        return None


def fetch_vendor_assignments_from_db(db_manager, session_id, file_id):
    """Fetch vendor assignments from database - non-blocking, returns empty list on failure"""
    if not db_manager:
        return []
    
    try:
        if session_id:
            return db_manager.fetch_vendor_assignments(session_id)
        if file_id:
            return db_manager.fetch_vendor_assignments_by_file(file_id)
        return []
    except Exception as exc:
        # Log but don't fail - vendor assignments are optional
        print(f"⚠️ Failed to fetch vendor assignments (non-critical): {exc}")
        return []


def apply_vendor_assignments_to_dataframe(df: pd.DataFrame, assignments: List[Dict[str, Any]]) -> pd.DataFrame:
    if not assignments:
        return df

    df_copy = df.copy()
    if 'Original_Row_Number' not in df_copy.columns:
        df_copy['Original_Row_Number'] = df_copy.reset_index().index + 1

    df_copy['Original_Row_Number'] = pd.to_numeric(
        df_copy['Original_Row_Number'], errors='coerce'
    ).astype('Int64')

    assignment_map = {}
    for row in assignments:
        row_number = row.get('original_row_number')
        vendor_name = row.get('vendor_name')
        if row_number is None or not vendor_name:
            continue
        try:
            assignment_map[int(row_number)] = vendor_name
        except (TypeError, ValueError):
            continue

    if not assignment_map:
        return df_copy

    mapped_vendors = df_copy['Original_Row_Number'].map(assignment_map)
    if 'Vendor' in df_copy.columns:
        df_copy['Vendor'] = mapped_vendors.combine_first(df_copy['Vendor'])
    else:
        df_copy['Vendor'] = mapped_vendors
    df_copy['Vendor'] = df_copy['Vendor'].fillna('Unknown Vendor')
    return df_copy


def build_daily_timeseries(transactions_df: pd.DataFrame) -> List[Dict[str, Any]]:
    if transactions_df.empty or 'Date' not in transactions_df.columns:
        return []

    temp_df = transactions_df.copy()
    temp_df['__date'] = pd.to_datetime(temp_df['Date'], errors='coerce')
    temp_df = temp_df.dropna(subset=['__date'])
    if temp_df.empty:
        return []

    # Use Inward_Amount and Outward_Amount if available, otherwise fall back to Amount
    if 'Inward_Amount' in temp_df.columns and 'Outward_Amount' in temp_df.columns:
        temp_df['_amount'] = pd.to_numeric(temp_df['Inward_Amount'], errors='coerce').fillna(0) + pd.to_numeric(temp_df['Outward_Amount'], errors='coerce').fillna(0)
    elif 'Amount' in temp_df.columns:
        temp_df['_amount'] = pd.to_numeric(temp_df['Amount'], errors='coerce').fillna(0)
    else:
        # Try to find amount column
        amount_col = next((c for c in temp_df.columns if 'amount' in c.lower()), None)
        if amount_col:
            temp_df['_amount'] = pd.to_numeric(temp_df[amount_col], errors='coerce').fillna(0)
        else:
            temp_df['_amount'] = 0.0

    grouped = temp_df.groupby(temp_df['__date'].dt.date)['_amount'].sum().reset_index()
    return [
        {'date': date_value.isoformat(), 'net': float(amount)}
        for date_value, amount in grouped.values
    ]


def build_monthly_patterns(transactions_df: pd.DataFrame) -> List[Dict[str, Any]]:
    if transactions_df.empty or 'Date' not in transactions_df.columns:
        return []

    temp_df = transactions_df.copy()
    temp_df['__date'] = pd.to_datetime(temp_df['Date'], errors='coerce')
    temp_df = temp_df.dropna(subset=['__date'])
    if temp_df.empty:
        return []

    temp_df['__month'] = temp_df['__date'].dt.to_period('M')
    patterns = []
    
    # Use Inward_Amount and Outward_Amount if available, otherwise fall back to Amount
    use_inward_outward = 'Inward_Amount' in temp_df.columns and 'Outward_Amount' in temp_df.columns
    
    for month, group in temp_df.groupby('__month'):
        if use_inward_outward:
            inflows = float(pd.to_numeric(group['Inward_Amount'], errors='coerce').fillna(0).sum())
            outflows = float(pd.to_numeric(group['Outward_Amount'], errors='coerce').fillna(0).apply(lambda x: abs(x) if pd.notna(x) else 0.0).sum())
        else:
            if 'Amount' in group.columns:
                group['Amount'] = pd.to_numeric(group['Amount'], errors='coerce').fillna(0)
                inflows = float(group[group['Amount'] > 0]['Amount'].sum())
                outflows = float(abs(group[group['Amount'] < 0]['Amount'].sum()))
            else:
                inflows = 0.0
                outflows = 0.0
        
        patterns.append({
            'month': str(month),
            'inflows': inflows,
            'outflows': outflows,
            'net': float(inflows - outflows),
            'transaction_count': int(len(group))
        })
    return patterns


def format_currency(value) -> str:
    try:
        amount = float(value or 0)
    except (TypeError, ValueError):
        amount = 0.0
    return f"INR {amount:,.2f}"


def build_comprehensive_report_payload(
    bank_df: pd.DataFrame,
    session_obj,
    state_manager,
    db_manager,
    database_available: bool,
    persistent_state_available: bool,
) -> Dict[str, Any]:
    try:
        print("📊 [REPORTS] Step 1: Resolving session/file IDs...")
        session_id, file_id = resolve_active_session_and_file(
            session_obj=session_obj,
            state_manager=state_manager,
            db_manager=db_manager,
            database_available=database_available,
            persistent_state_available=persistent_state_available,
        )
        print(f"📊 [REPORTS] Step 1 complete: session_id={session_id}, file_id={file_id}")

        print("📊 [REPORTS] Step 2: Fetching vendor data...")
        vendor_master = None
        vendor_assignments = []
        if db_manager:
            print(f"📊 [REPORTS] Step 2a: Fetching vendor master (session_id={session_id}, file_id={file_id})...")
            try:
                vendor_master = fetch_vendor_master_from_db(db_manager, session_id, file_id)
                print(f"📊 [REPORTS] Step 2a complete: Vendor master fetched: {vendor_master is not None}")
            except Exception as e:
                print(f"⚠️ Error fetching vendor master: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
            
            print(f"📊 [REPORTS] Step 2b: Fetching vendor assignments (session_id={session_id}, file_id={file_id})...")
            try:
                vendor_assignments = fetch_vendor_assignments_from_db(db_manager, session_id, file_id)
                print(f"📊 [REPORTS] Step 2b complete: Vendor assignments fetched: {len(vendor_assignments)} items")
            except Exception as e:
                print(f"⚠️ Error fetching vendor assignments: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("📊 [REPORTS] Step 2 skipped: No database manager available")
        print("📊 [REPORTS] Step 2 complete")
        
        print("📊 [REPORTS] Step 3: Applying vendor assignments...")
        if vendor_assignments:
            try:
                bank_df = apply_vendor_assignments_to_dataframe(bank_df, vendor_assignments)
                print(f"📊 [REPORTS] Vendor assignments applied: {len(bank_df)} rows")
            except Exception as e:
                print(f"⚠️ [REPORTS] Step 3 WARNING: Error applying vendor assignments: {e}")
                import traceback
                traceback.print_exc()
                print(f"⚠️ [REPORTS] Continuing without vendor assignments...")
        else:
            print(f"📊 [REPORTS] Step 3 skipped: No vendor assignments to apply")
        print("📊 [REPORTS] Step 3 complete")

        print("📊 [REPORTS] Step 4: Generating final report...")
        print(f"📊 [REPORTS] Step 4: Input DataFrame shape: {bank_df.shape}, columns: {list(bank_df.columns)[:10]}")
        try:
            print("📊 [REPORTS] Step 4a: Calling generate_final_report()...")
            report = generate_final_report(bank_df, vendor_master=vendor_master, include_raw=True)
            print("📊 [REPORTS] Step 4 complete: Final report generated")
        except Exception as report_error:
            print(f"❌ [REPORTS] Step 4 FAILED: {report_error}")
            import traceback
            traceback.print_exc()
            # Return minimal valid payload
            return {
                'success': False,
                'error': f'Failed to generate report: {str(report_error)}',
                'generated_at': None,
                'summary': {},
                'transactions': [],
                'vendor_analysis': [],
                'category_analysis': [],
                'analytics_timeseries': [],
                'monthly_patterns': [],
                'executive_summary': {'error': str(report_error)},
                'source': {}
            }

        print("📊 [REPORTS] Step 5: Processing transactions...")
        transactions = report.get('transactions', [])
        transactions_df = pd.DataFrame(transactions) if transactions else pd.DataFrame()
        print(f"📊 [REPORTS] Step 5 complete: {len(transactions)} transactions processed")

        print("📊 [REPORTS] Step 6: Building category analysis...")
        summary = report.get('summary', {})
        category_totals = summary.get('category_totals', {})
        category_analysis = [
            {
                'category': category,
                'inflows': metrics.get('inflows', 0),
                'outflows': metrics.get('outflows', 0),
                'net': metrics.get('net', 0),
                'transaction_count': metrics.get('count', 0)
            }
            for category, metrics in category_totals.items()
        ]
        print(f"📊 [REPORTS] Step 6 complete: {len(category_analysis)} categories")

        print("📊 [REPORTS] Step 7: Building analytics...")
        # Build analytics with error handling
        try:
            analytics_timeseries = build_daily_timeseries(transactions_df)
            print(f"📊 [REPORTS] Daily timeseries built: {len(analytics_timeseries)} items")
        except Exception as e:
            print(f"⚠️ Error building daily timeseries: {e}")
            analytics_timeseries = []
        
        try:
            monthly_patterns = build_monthly_patterns(transactions_df)
            print(f"📊 [REPORTS] Monthly patterns built: {len(monthly_patterns)} items")
        except Exception as e:
            print(f"⚠️ Error building monthly patterns: {e}")
            monthly_patterns = []

        print("📊 [REPORTS] Step 8: Building final payload...")

        payload = {
            'success': True,
            'generated_at': report.get('generated_at'),
            'summary': summary,
            'transactions': transactions,
            'vendor_analysis': report.get('vendors', []),
            'category_analysis': category_analysis,
            'analytics_timeseries': analytics_timeseries,
            'monthly_patterns': monthly_patterns,
            'cashflow_statement': report.get('cashflow_statement'),  # ✅ Include cash flow statement
            'executive_summary': {
                'total_transactions': summary.get('transaction_count', 0),
                'total_inflows': summary.get('total_inflows', 0),
                'total_outflows': summary.get('total_outflows', 0),
                'net_cash_flow': summary.get('net_cash_flow', 0)
            },
            'source': {
                'session_id': session_id,
                'file_id': file_id
            }
        }
        return payload
    except Exception as e:
        print(f"❌ Error in build_comprehensive_report_payload: {e}")
        import traceback
        traceback.print_exc()
        # Return a minimal error payload
        return {
            'success': False,
            'error': f'Failed to build report: {str(e)}',
            'generated_at': None,
            'summary': {},
            'transactions': [],
            'vendor_analysis': [],
            'category_analysis': [],
            'analytics_timeseries': [],
            'monthly_patterns': [],
            'executive_summary': {},
            'source': {}
        }


