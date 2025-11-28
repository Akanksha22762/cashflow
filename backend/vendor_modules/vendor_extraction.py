import json
import time
from typing import Tuple, Dict, Any

import pandas as pd

from .session_utils import resolve_session_ids
from .vendor_cache import get_cached_vendor_data


def extract_vendors(
    bank_df: pd.DataFrame,
    uploaded_bank_df: pd.DataFrame,
    uploaded_data: Dict[str, Any],
    app_openai_integration,
    db_manager,
    session,
    persistent_state_available: bool,
    state_manager,
    database_available: bool,
) -> Tuple[Dict[str, Any], int, pd.DataFrame]:
    """
    Core vendor extraction logic used by /extract-vendors-for-analysis.

    Returns (response_dict, status_code, updated_bank_df)
    """
    session_id, file_id = resolve_session_ids(session if session else {}, state_manager, db_manager)

    if database_available and db_manager:
        cached = _get_vendor_cache(db_manager, session_id, file_id)
        if cached and cached.get('summaries'):
            return _build_cached_vendor_response(cached, uploaded_data, uploaded_bank_df)

    if bank_df is None or bank_df.empty:
        return ({
            'status': 'error',
            'error': 'No data available. Please upload files first.'
        }, 400, uploaded_bank_df)

    # Find description column
    desc_col = None
    for candidate in ['Description', 'description', 'Transaction Description', 'transaction_description', '_combined_description']:
        if candidate in bank_df.columns:
            desc_col = candidate
            break
    if desc_col is None:
        return ({
            'status': 'error',
            'error': 'No description column found in uploaded data'
        }, 400, uploaded_bank_df)

    descriptions = bank_df[desc_col].astype(str).tolist()

    # Run AI extraction (OpenAI/OpenAI)
    vendors = []
    vendor_patterns = {}
    used_openai = False
    vendor_col = 'Assigned_Vendor'

    try:
        if 'app_openai_integration' in globals():
            pass  # placeholder for lint; actual integration passed below
    except Exception:
        pass

    try:
        if app_openai_integration and getattr(app_openai_integration, 'is_available', False):
            print("🤖 Using OpenAI to extract vendors per transaction...")
            print(f"📊 Processing {len(descriptions)} descriptions...")
            
            try:
                per_tx_vendors = app_openai_integration.extract_vendors_for_transactions(descriptions)
                print(f"✅ OpenAI extraction returned {len(per_tx_vendors) if per_tx_vendors else 0} vendors")
            except Exception as openai_error:
                print(f"❌ OpenAI API call failed: {openai_error}")
                import traceback
                traceback.print_exc()
                raise
            
            bank_df[vendor_col] = per_tx_vendors
            used_openai = True
            vendors = [v for v in bank_df[vendor_col].dropna().unique().tolist() if v and v.strip()]
            vendors.insert(0, "All")
            vendors[1:] = sorted(vendors[1:])
            print(f"✅ OpenAI vendor extraction complete: {len(vendors)} vendors")
    except Exception as e:
        print(f"⚠️ OpenAI vendor extraction failed, will fallback: {e}")
        import traceback
        traceback.print_exc()
        used_openai = False

    if not used_openai:
        return ({
            'status': 'error',
            'error': 'OpenAI vendor extraction is required and is not available. Please configure OPENAI_API_KEY (and OPENAI_PROJECT if needed) and try again.'
        }, 500, uploaded_bank_df)

    # Validate vendors (same logic as original)
    test_descriptions = bank_df[desc_col].astype(str).tolist()

    def test_vendor_transactions(vendor_name, descriptions=None):
        lookup = descriptions if descriptions is not None else test_descriptions
        vendor_clean = vendor_name.strip().lower()
        for suffix in [' vendor', ' supplier', ' services', ' corp', ' inc', ' ltd']:
            if vendor_clean.endswith(suffix):
                vendor_clean = vendor_clean.replace(suffix, '').strip()
        search_terms = [vendor_clean]
        core_parts = vendor_clean.split()
        if len(core_parts) > 1:
            search_terms.extend(core_parts)
        if vendor_name.lower() != vendor_clean:
            search_terms.append(vendor_name.lower())
        for search_term in search_terms:
            search_term = search_term.strip()
            if len(search_term) < 3:
                continue
            matches = [desc for desc in lookup if search_term in desc.lower()]
            if matches:
                return True, len(matches)
        return False, 0

    validated_vendors = []
    vendor_assignments = bank_df[vendor_col].tolist()

    for vendor in vendors:
        if vendor == 'All':
            validated_vendors.append(vendor)
            continue
        has_transactions, count = test_vendor_transactions(vendor, test_descriptions)
        if has_transactions:
            validated_vendors.append(vendor)
            vendor_patterns[vendor] = {'transaction_matches': count}
        else:
            print(f"❌ {vendor}: No transactions found - EXCLUDED from dropdown")

    unique_vendors = validated_vendors
    vendor_assignments = [v for v in vendor_assignments if v and v.strip()]
    print(f"🎯 Final validated vendor list: {len(unique_vendors)} vendors with confirmed transactions (including 'All' option)")

    # Persist to uploaded_bank_df for other features
    updated_bank_df = bank_df
    if uploaded_data is not None:
        uploaded_data['bank_df'] = updated_bank_df

    # Persist to state manager (if available)
    if persistent_state_available and state_manager and state_manager.current_session_id:
        try:
            vendor_analysis_results = {
                f"vendor_extraction_{int(time.time())}": {
                    'analysis_type': 'vendor_extraction',
                    'results': {
                        'success': True,
                        'data': {
                            'vendors': unique_vendors,
                            'total_vendors': len(unique_vendors),
                            'vendor_patterns': vendor_patterns
                        }
                    },
                    'timestamp': time.time(),
                    'extraction_method': 'validated_extraction',
                    'analysis_metadata': {
                        'total_extracted': len(unique_vendors),
                        'validation_passed': True
                    }
                }
            }
            saved = state_manager.save_analysis_results(vendor_analysis_results)
            if saved:
                print(f"✅ PERSISTENCE: Vendor list saved to database for restoration ({len(unique_vendors)} vendors)")
            else:
                print(f"⚠️ PERSISTENCE: Failed to save vendor list")
        except Exception as save_error:
            print(f"⚠️ PERSISTENCE: Error saving vendor list: {save_error}")

    # Prepare transaction preview for frontend (same as original route)
    transactions_with_vendors = []
    for _, row in bank_df.iterrows():
        transactions_with_vendors.append({
            'Description': row.get('Description'),
            'Amount': row.get('Amount'),
            'Date': row.get('Date').isoformat() if hasattr(row.get('Date'), 'isoformat') else str(row.get('Date')),
            'Assigned_Vendor': row.get(vendor_col)
        })

    if database_available and db_manager and session_id and file_id:
        try:
            print("💾 Persisting vendor assignments to database...")
            vendor_df = bank_df.copy()
            amount_col = 'Amount'
            if amount_col not in vendor_df.columns:
                numeric_cols = [c for c in vendor_df.columns if pd.api.types.is_numeric_dtype(vendor_df[c])]
                if numeric_cols:
                    amount_col = numeric_cols[0]
                    vendor_df['Amount'] = pd.to_numeric(vendor_df[amount_col], errors='coerce').fillna(0)
                else:
                    vendor_df['Amount'] = 0
            else:
                vendor_df['Amount'] = pd.to_numeric(vendor_df['Amount'], errors='coerce').fillna(0)
            if 'Original_Row_Number' not in vendor_df.columns:
                vendor_df['Original_Row_Number'] = vendor_df.index + 1

            def classify_frequency(count):
                if count >= 20:
                    return 'High'
                if count >= 10:
                    return 'Medium'
                return 'Low'

            def classify_importance(net_value):
                abs_value = abs(net_value)
                if abs_value >= 1_000_000:
                    return 'Critical'
                if abs_value >= 100_000:
                    return 'Strategic'
                return 'Normal'

            vendor_summaries = []
            transaction_links = []

            vendor_column = vendor_col
            if vendor_column not in vendor_df.columns and 'AssignedVendor' in vendor_df.columns:
                vendor_column = 'AssignedVendor'

            vendor_names_for_storage = [v for v in unique_vendors if v and v != 'All']
            for vendor in vendor_names_for_storage:
                rows = vendor_df[vendor_df[vendor_column] == vendor]
                if rows.empty:
                    continue
                
                # Use Inward_Amount and Outward_Amount directly (same as reports)
                if 'Inward_Amount' in rows.columns and 'Outward_Amount' in rows.columns:
                    inflow = float(rows['Inward_Amount'].fillna(0).sum())
                    outflow = float(rows['Outward_Amount'].fillna(0).apply(lambda x: abs(float(x)) if pd.notna(x) else 0.0).sum())
                    net = float(inflow - outflow)
                else:
                    # Fallback to Amount column if Inward/Outward not available
                    inflow = float(rows[rows['Amount'] > 0]['Amount'].sum())
                    outflow = float(abs(rows[rows['Amount'] < 0]['Amount'].sum()))
                    net = float(rows['Amount'].sum())
                category_totals = {}
                if 'Category' in rows.columns:
                    category_totals = {
                        str(cat): float(val)
                        for cat, val in rows.groupby('Category')['Amount'].sum().to_dict().items()
                    }

                vendor_summaries.append({
                    'vendor_name': vendor,
                    'vendor_category': None,
                    'total_transactions': len(rows),
                    'total_inflow': inflow,
                    'total_outflow': outflow,
                    'net_cash_flow': net,
                    'payment_frequency': classify_frequency(len(rows)),
                    'vendor_importance': classify_importance(net),
                    'cash_flow_categories': category_totals,
                    'ai_summary': {
                        'insights': [
                            f"{len(rows)} transactions linked to {vendor}",
                            f"Net cash flow ${net:,.2f}"
                        ]
                    }
                })

                for _, r in rows.iterrows():
                    row_number = int(r.get('Original_Row_Number') or (r.name + 1))
                    transaction_links.append({
                        'vendor_name': vendor,
                        'row_number': row_number,
                        'assignment_source': 'AI',
                        'ai_confidence': None,
                        'ai_reasoning': r.get('AI_Reasoning')
                    })

            overall_rows = vendor_df  # include all rows for "All" summary
            if not overall_rows.empty:
                # Use Inward_Amount and Outward_Amount directly (same as reports)
                if 'Inward_Amount' in overall_rows.columns and 'Outward_Amount' in overall_rows.columns:
                    inflow = float(overall_rows['Inward_Amount'].fillna(0).sum())
                    outflow = float(overall_rows['Outward_Amount'].fillna(0).apply(lambda x: abs(float(x)) if pd.notna(x) else 0.0).sum())
                    net = float(inflow - outflow)
                else:
                    # Fallback to Amount column if Inward/Outward not available
                    inflow = float(overall_rows[overall_rows['Amount'] > 0]['Amount'].sum())
                    outflow = float(abs(overall_rows[overall_rows['Amount'] < 0]['Amount'].sum()))
                    net = float(overall_rows['Amount'].sum())
                category_totals = {}
                if 'Category' in overall_rows.columns:
                    category_totals = {
                        str(cat): float(val)
                        for cat, val in overall_rows.groupby('Category')['Amount'].sum().to_dict().items()
                    }
                vendor_summaries.append({
                    'vendor_name': 'All',
                    'vendor_category': 'Aggregate',
                    'total_transactions': len(overall_rows),
                    'total_inflow': inflow,
                    'total_outflow': outflow,
                    'net_cash_flow': net,
                    'payment_frequency': classify_frequency(len(overall_rows)),
                    'vendor_importance': classify_importance(net),
                    'cash_flow_categories': category_totals,
                    'ai_summary': {
                        'insights': [
                            f"{len(overall_rows)} vendor-tagged transactions",
                            f"Net cash flow ${net:,.2f}"
                        ]
                    }
                })

            db_manager.store_vendor_data(
                file_id=file_id,
                session_id=session_id,
                vendor_summaries=vendor_summaries,
                transaction_links=transaction_links
            )
            print("✅ Vendor assignments persisted to database")
            
            # Precompute vendor landscape analysis for all vendors (so Details button works instantly)
            print("🧠 Precomputing vendor landscape analysis for all vendors...")
            try:
                import sys
                import os
                import json
                # Add parent directory to path for imports
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from ai_reasoning_engine import get_vendor_landscape_analysis
                AI_REASONING_AVAILABLE = True
            except Exception as import_error:
                print(f"⚠️ AI Reasoning Engine not available: {import_error}")
                AI_REASONING_AVAILABLE = False
            
            if AI_REASONING_AVAILABLE:
                # Include "All" vendor in precomputation (it represents all transactions combined)
                vendor_names_for_analysis = [v for v in unique_vendors if v]  # Include 'All' now
                total_vendors = len(vendor_names_for_analysis)
                
                for idx, vendor_name in enumerate(vendor_names_for_analysis, 1):
                    try:
                        # For "All" vendor, use all transactions; otherwise filter by vendor
                        if vendor_name == 'All':
                            vendor_rows = vendor_df  # All transactions
                        else:
                            vendor_rows = vendor_df[vendor_df[vendor_column] == vendor_name]
                        
                        if vendor_rows.empty:
                            continue
                        
                        # Prepare transaction data for landscape analysis
                        vendor_transactions = []
                        for _, row in vendor_rows.iterrows():
                            vendor_transactions.append({
                                'description': str(row.get('Description', '')),
                                'amount': float(row.get('Amount', 0)),
                                'vendor': str(row.get(vendor_column, vendor_name)) if vendor_name != 'All' else vendor_name
                            })
                        
                        if not vendor_transactions:
                            continue
                        
                        # Generate landscape analysis
                        print(f"   🤖 Generating landscape analysis for vendor {idx}/{total_vendors}: {vendor_name}...")
                        analysis = get_vendor_landscape_analysis([vendor_name], vendor_transactions)
                        
                        # Store in vendor_reasoning column
                        analysis_json = json.dumps(analysis) if not isinstance(analysis, str) else analysis
                        
                        # Update vendor entity with reasoning
                        conn = db_manager.get_connection()
                        cursor = None
                        try:
                            cursor = conn.cursor()
                            cursor.execute("""
                                UPDATE vendor_entities 
                                SET vendor_reasoning = %s, reasoning_generated_at = CURRENT_TIMESTAMP
                                WHERE vendor_name = %s AND session_id = %s
                            """, (analysis_json, vendor_name, session_id))
                            conn.commit()
                        finally:
                            if cursor:
                                cursor.close()
                            if conn:
                                conn.close()
                        
                        if (idx % 5 == 0) or idx == total_vendors:
                            print(f"   ✅ Precomputed analysis for {idx}/{total_vendors} vendors")
                    
                    except Exception as vendor_error:
                        print(f"   ⚠️ Failed to precompute analysis for {vendor_name}: {vendor_error}")
                        continue
                
                print(f"✅ Precomputed vendor landscape analysis for {total_vendors} vendors (including 'All')")
            else:
                print("⚠️ Skipping vendor landscape precomputation (AI Reasoning Engine not available)")
                
        except Exception as persist_error:
            print(f"⚠️ Vendor persistence failed: {persist_error}")
    else:
        print("⚠️ Vendor persistence skipped: missing session_id or file_id in session")

    response = {
        'success': True,
        'vendors': unique_vendors,
        'total_transactions': len(bank_df),
        'vendor_assignments': len(vendor_assignments),
        'transactions_with_vendors': transactions_with_vendors,
        'message': f'OpenAI successfully assigned vendors to {len(bank_df)} transactions'
    }

    return response, 200, updated_bank_df


def _get_vendor_cache(db_manager, session_id, file_id):
    if not db_manager:
        return None
    if file_id:
        cached = get_cached_vendor_data(db_manager, file_id)
        if cached:
            return cached
    if session_id:
        summaries = db_manager.fetch_vendor_entities(session_id)
        if summaries:
            transactions = {}
            for summary in summaries:
                name = summary.get('vendor_name')
                if not name:
                    continue
                transactions[name] = db_manager.fetch_vendor_transactions(session_id, name)
            if file_id and not transactions:
                transactions = db_manager.fetch_vendor_transactions_by_file(file_id)
            return {'summaries': summaries, 'transactions': transactions}
    return None


def _build_cached_vendor_response(cached_data, uploaded_data, uploaded_bank_df):
    summaries = cached_data.get('summaries', [])
    tx_map = cached_data.get('transactions', {})

    vendors = [row['vendor_name'] for row in summaries if row.get('vendor_name')]
    vendors = ['All'] + sorted([v for v in vendors if v and v != 'All']) if vendors else ['All']

    all_entries = tx_map.get('All', [])
    transactions_with_vendors = []
    for entry in all_entries:
        transactions_with_vendors.append({
            'Description': entry.get('description'),
            'Amount': entry.get('amount'),
            'Date': entry.get('date'),
            'Assigned_Vendor': entry.get('vendor')
        })

    total_transactions = next((s.get('total_transactions') for s in summaries if s.get('vendor_name') in ('All', 'All Vendors')), None)
    if total_transactions is None:
        total_transactions = len(all_entries)

    if uploaded_data is not None and transactions_with_vendors:
        try:
            uploaded_bank_df = pd.DataFrame(transactions_with_vendors)
            uploaded_data['bank_df'] = uploaded_bank_df
        except Exception:
            pass

    response = {
        'success': True,
        'vendors': vendors,
        'total_transactions': total_transactions,
        'vendor_assignments': len(all_entries),
        'transactions_with_vendors': transactions_with_vendors,
        'message': 'Vendors loaded from database cache'
    }
    return response, 200, uploaded_bank_df

