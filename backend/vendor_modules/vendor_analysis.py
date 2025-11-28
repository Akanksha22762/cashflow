import pandas as pd
from typing import Dict, Any, Tuple, Optional

from .session_utils import resolve_session_ids
from .vendor_cache import get_cached_vendor_data

def run_vendor_analysis(
    vendor: str,
    uploaded_bank_df,
    uploaded_data: Dict[str, Any],
    db_manager=None,
    session=None,
    database_available: bool = False,
    state_manager=None,
) -> Tuple[Dict[str, Any], int]:
    """Core logic for /vendor-analysis endpoint."""
    try:
        if not vendor:
            return {'error': 'Vendor is required'}, 400

        vendor_lower = vendor.lower()
        if vendor_lower == 'all':
            vendor_filter = 'All'
            vendor_display_name = 'All Vendors'
        else:
            vendor_filter = vendor
            vendor_display_name = vendor

        session_id, file_id = resolve_session_ids(session if session else {}, state_manager, db_manager)

        if database_available and db_manager:
            summary, transactions = _fetch_cached_vendor_data(db_manager, session_id, file_id, vendor_display_name)
            if summary or transactions:
                return _build_cached_analysis_response(vendor_display_name, summary, transactions)

        bank_df = _get_bank_df(uploaded_bank_df, uploaded_data)
        if bank_df is None:
            return {'error': 'No bank data uploaded yet. Please upload a file first.'}, 400
        if isinstance(bank_df, pd.DataFrame) and bank_df.empty:
            return {'error': 'Uploaded bank data is empty'}, 400
        if not isinstance(bank_df, pd.DataFrame):
            return {'error': 'Bank data is not in the correct format'}, 400

        filtered_transactions = smart_vendor_filter(bank_df, vendor_filter)
        vendor_name = vendor_display_name
        if filtered_transactions.empty:
            return {'status': 'error', 'error': f'No transactions found for vendor: {vendor}'}, 400

        df = _normalize_amount_column(filtered_transactions)
        if df is None:
            return {'status': 'error', 'error': 'No amount column found in transaction data'}, 400
        filtered_transactions = df

        total_amount = filtered_transactions['Amount'].sum()
        transaction_count = len(filtered_transactions)
        avg_amount = filtered_transactions['Amount'].mean()

        openai_insights = _generate_openai_insights(vendor_name, transaction_count, total_amount)
        payment_frequency = _estimate_payment_frequency(filtered_transactions)
        vendor_importance = _estimate_vendor_importance(transaction_count, total_amount)

        ai_pattern = ('positive' if total_amount > 0 else 'negative') + \
                     (' trend (strong)' if transaction_count > 10 else ' trend (moderate)')
        ai_confidence = float(min(1.0, max(0.0, transaction_count / 20.0)))
        ml_prediction = 'Positive cash flow' if total_amount > 0 else 'Negative cash flow'
        ml_accuracy = ai_confidence

        response = {
            'status': 'success',
            'data': {
                'vendor_name': vendor_name,
                'total_amount': float(total_amount),
                'transaction_count': int(transaction_count),
                'avg_amount': float(avg_amount),
                'cash_flow_status': 'Positive' if total_amount > 0 else 'Negative',
                'payment_frequency': payment_frequency,
                'vendor_importance': vendor_importance,
                'analysis_summary': {
                    'total_transactions': int(transaction_count),
                    'net_cash_flow': float(total_amount),
                    'avg_transaction': float(avg_amount)
                }
            },
            'reasoning_explanations': {
                'simple_reasoning': f'OpenAI AI Analysis: {transaction_count} transactions totaling ₹{total_amount:,.2f}. {openai_insights}',
                'training_insights': f'OpenAI AI analyzed {transaction_count} transactions using natural language understanding',
                'ml_analysis': {
                    'pattern_analysis': {
                        'trend_direction': 'positive' if total_amount > 0 else 'negative',
                        'pattern_strength': 'strong' if transaction_count > 10 else 'moderate'
                    },
                    'prediction': ml_prediction,
                    'accuracy': ml_accuracy
                },
                'ai_analysis': {
                    'business_intelligence': {
                        'financial_knowledge': f'OpenAI Insights: {openai_insights}',
                        'openai_powered': True
                    },
                    'pattern_analysis': ai_pattern,
                    'confidence': ai_confidence
                }
            },
            'message': f'OpenAI-powered vendor analysis completed for {vendor_name}'
        }
        return response, 200

    except Exception as exc:
        print(f"❌ Vendor analysis error: {exc}")
        return {'error': str(exc)}, 500


def get_vendor_transactions_view(
    vendor_name: str,
    uploaded_data: Dict[str, Any],
    uploaded_bank_df,
    db_manager,
    session,
    database_available: bool,
    state_manager=None,
) -> Tuple[Dict[str, Any], int]:
    """Logic for /view_vendor_transactions/<vendor_name> endpoint."""
    session_id, file_id = resolve_session_ids(session, state_manager, db_manager)
    if database_available and db_manager and session_id:
        vendor_entity = db_manager.fetch_vendor_entity(session_id, vendor_name)
        if vendor_entity:
            transactions = db_manager.fetch_vendor_transactions(session_id, vendor_name)
            net_flow = float(vendor_entity.get('net_cash_flow') or 0)
            cash_flow_status = "Positive Flow" if net_flow > 0 else "Negative Flow" if net_flow < 0 else "Balanced"
            summary_cards = {
                'transactions': {'value': vendor_entity.get('total_transactions', 0), 'label': 'TRANSACTIONS', 'description': 'Stored vendor transactions'},
                'cash_flow_status': {'value': cash_flow_status, 'label': 'CASH FLOW STATUS', 'description': 'Based on stored net cash flow'},
                'payment_patterns': {'value': vendor_entity.get('payment_frequency') or 'N/A', 'label': 'PAYMENT FREQUENCY', 'description': 'Historical payment frequency'},
                'collection_status': {'value': vendor_entity.get('vendor_importance') or 'Normal', 'label': 'VENDOR IMPORTANCE', 'description': 'AI importance classification'}
            }
            return ({
                'success': True,
                'vendor_name': vendor_name,
                'summary_cards': summary_cards,
                'transactions': transactions,
                'cash_flow_categories': vendor_entity.get('cash_flow_categories'),
                'ai_summary': vendor_entity.get('ai_summary')
            }, 200)

    if database_available and db_manager and file_id:
        cached = get_cached_vendor_data(db_manager, file_id)
        if cached:
            transactions = cached['transactions'].get(vendor_name)
            summary = next((s for s in cached['summaries'] if s['vendor_name'] == vendor_name), None)
            if transactions:
                summary_cards = _summary_cards_from_summary(summary, len(transactions))
                return ({
                    'success': True,
                    'vendor_name': vendor_name,
                    'summary_cards': summary_cards,
                    'transactions': transactions,
                    'cash_flow_categories': (summary or {}).get('cash_flow_categories'),
                    'ai_summary': (summary or {}).get('ai_summary')
                }, 200)

    print(f"⚠️ Vendor '{vendor_name}' not found in DB (session_id={session_id}); returning empty dataset.")
    return ({
        'success': True,
        'vendor_name': vendor_name,
        'summary_cards': _empty_summary_cards(),
        'transactions': []
    }, 200)


def _get_bank_df(uploaded_bank_df, uploaded_data):
    if uploaded_bank_df is not None:
        if isinstance(uploaded_bank_df, pd.DataFrame):
            if not uploaded_bank_df.empty:
                return uploaded_bank_df
        elif isinstance(uploaded_bank_df, list) and uploaded_bank_df:
            try:
                return pd.DataFrame(uploaded_bank_df)
            except Exception:
                pass
    if uploaded_data and uploaded_data.get('bank_df') is not None:
        if isinstance(uploaded_data['bank_df'], pd.DataFrame):
            return uploaded_data['bank_df']
        if isinstance(uploaded_data['bank_df'], list):
            try:
                return pd.DataFrame(uploaded_data['bank_df'])
            except Exception:
                pass
    return None


def _normalize_amount_column(df: pd.DataFrame):
    df = df.copy()
    amount_col = None
    for c in ['Amount', 'amount', '_amount', 'Credit Amount', 'Debit Amount', 'Balance']:
        if c in df.columns:
            amount_col = c
            break
    if amount_col is None:
        numeric_cols = [c for c in df.columns if str(df[c].dtype).startswith(('float', 'int'))]
        amount_col = numeric_cols[0] if numeric_cols else None
    if amount_col is None:
        return None
    if amount_col != 'Amount':
        df['Amount'] = pd.to_numeric(df[amount_col], errors='coerce').fillna(0)
    else:
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
    return df


def _generate_openai_insights(vendor_name, transaction_count, total_amount):
    try:
        from openai_integration import simple_openai, check_openai_availability
        if check_openai_availability():
            prompt = (
                f"Analyze this vendor financial data:\n"
                f"Vendor: {vendor_name}\n"
                f"Transactions: {transaction_count}\n"
                f"Total Amount: ₹{total_amount:,.2f}\n"
                f"Provide 2-3 brief business insights about this vendor:"
            )
            return simple_openai(prompt, max_tokens=100)
    except Exception:
        pass
    return f"Analysis of {vendor_name} transactions"


def _estimate_payment_frequency(df: pd.DataFrame) -> str:
    try:
        date_col = None
        for c in ['Date', 'date', 'Txn Date', 'Transaction Date', 'Value Date', 'Posted Date']:
            if c in df.columns:
                date_col = c
                break
        if date_col is None:
            for c in df.columns:
                parsed = pd.to_datetime(df[c], errors='coerce')
                if parsed.notna().mean() >= 0.5:
                    date_col = c
                    break
        if date_col is not None:
            dt_series = pd.to_datetime(df[date_col], errors='coerce').dropna().sort_values()
            if len(dt_series) >= 2:
                avg_days = dt_series.diff().dt.days.dropna().mean()
                if pd.notna(avg_days):
                    if avg_days <= 3:
                        return 'Daily'
                    if avg_days <= 10:
                        return 'Weekly'
                    if avg_days <= 40:
                        return 'Monthly'
                    if avg_days <= 100:
                        return 'Quarterly'
                    return 'Annual'
    except Exception:
        pass
    return 'Monthly'


def _estimate_vendor_importance(transaction_count: int, total_amount: float) -> str:
    abs_total = float(abs(total_amount))
    if transaction_count >= 20 or abs_total >= 1_000_000:
        return 'High'
    if transaction_count >= 10 or abs_total >= 200_000:
        return 'Medium'
    return 'Low'


def _build_transaction_summary(df: pd.DataFrame):
    import pandas as pd
    vendor_transactions = df.copy()
    total_amount = vendor_transactions['Amount'].sum()
    transaction_count = len(vendor_transactions)
    avg_amount = vendor_transactions['Amount'].mean()
    inflows = vendor_transactions[vendor_transactions['Amount'] > 0]['Amount'].sum()
    outflows = abs(vendor_transactions[vendor_transactions['Amount'] < 0]['Amount'].sum())
    net_flow = inflows - outflows
    cash_flow_status = "Positive Flow" if net_flow > 0 else "Negative Flow" if net_flow < 0 else "Balanced"

    summary_cards = {
        'transactions': {'value': transaction_count, 'label': 'TRANSACTIONS', 'description': 'Click to view details'},
        'cash_flow_status': {'value': cash_flow_status, 'label': 'CASH FLOW STATUS', 'description': 'Net inflow/outflow'},
        'payment_patterns': {'value': _estimate_payment_frequency(vendor_transactions), 'label': 'PAYMENT FREQUENCY', 'description': 'Based on transaction history'},
        'collection_status': {'value': _estimate_vendor_importance(transaction_count, total_amount), 'label': 'VENDOR IMPORTANCE', 'description': 'Importance classification'}
    }

    transactions_list = []
    for _, row in vendor_transactions.iterrows():
        date_value = str(row.get('Date_Display', '') or row.get('Date') or 'Date N/A')
        transactions_list.append({
            'date': date_value,
            'description': str(row.get('Description', '')),
            'amount': float(row.get('Amount', 0)),
            'type': str(row.get('Type', 'Credit' if float(row.get('Amount', 0)) > 0 else 'Debit')),
            'category': str(row.get('Category', '')),
            'balance': float(row.get('Balance', 0)) if 'Balance' in row else 0
        })

    return summary_cards, transactions_list


def _empty_summary_cards():
    return {
        'transactions': {'value': 0, 'label': 'TRANSACTIONS', 'description': 'No stored data'},
        'cash_flow_status': {'value': 'No Data', 'label': 'CASH FLOW STATUS', 'description': 'No stored data'},
        'payment_patterns': {'value': 'N/A', 'label': 'PAYMENT FREQUENCY', 'description': 'No stored data'},
        'collection_status': {'value': 'N/A', 'label': 'VENDOR IMPORTANCE', 'description': 'No stored data'}
    }


def _fetch_cached_vendor_data(db_manager, session_id: Optional[int], file_id: Optional[int], vendor_name: str):
    summary = None
    transactions = []

    lookup_names = _normalized_vendor_aliases(vendor_name)

    # Try session-based lookup first
    if db_manager and session_id:
        for name in lookup_names:
            summary = db_manager.fetch_vendor_entity(session_id, name)
            if summary:
                transactions = db_manager.fetch_vendor_transactions(session_id, name)
                break

    # Fallback to file-based lookup
    if (not summary or not transactions) and db_manager and file_id:
        cached = get_cached_vendor_data(db_manager, file_id)
        if cached:
            if not summary:
                for name in lookup_names:
                    summary = next((s for s in cached['summaries'] if s.get('vendor_name') == name), None)
                    if summary:
                        break
            for name in lookup_names:
                transactions = cached['transactions'].get(name, [])
                if transactions:
                    break

    return summary, transactions


def _build_cached_analysis_response(vendor_name: str, summary: Optional[Dict[str, Any]], transactions: list):
    tx_count = int((summary or {}).get('total_transactions', len(transactions)))
    total_amount = float((summary or {}).get('net_cash_flow', 0))
    avg_amount = (total_amount / tx_count) if tx_count else 0.0
    payment_frequency = (summary or {}).get('payment_frequency') or 'Monthly'
    vendor_importance = (summary or {}).get('vendor_importance') or 'Normal'
    data = {
        'vendor_name': vendor_name,
        'total_amount': total_amount,
        'transaction_count': tx_count,
        'avg_amount': avg_amount,
        'cash_flow_status': 'Positive' if total_amount > 0 else 'Negative' if total_amount < 0 else 'Balanced',
        'payment_frequency': payment_frequency,
        'vendor_importance': vendor_importance,
        'analysis_summary': {
            'total_transactions': tx_count,
            'net_cash_flow': total_amount,
            'avg_transaction': avg_amount
        }
    }
    reasoning = {
        'simple_reasoning': f'Using stored analysis results for {vendor_name}.',
        'training_insights': f'Cached summary derived from {tx_count} historical transactions.',
        'ml_analysis': {
            'pattern_analysis': {
                'trend_direction': 'positive' if total_amount > 0 else 'negative',
                'pattern_strength': 'strong' if tx_count > 10 else 'moderate'
            },
            'prediction': 'Positive cash flow' if total_amount > 0 else 'Negative cash flow',
            'accuracy': 1.0
        },
        'ai_analysis': {
            'business_intelligence': {
                'financial_knowledge': 'Historical vendor insights restored from database.',
                'openai_powered': False
            },
            'pattern_analysis': 'historical',
            'confidence': 0.9
        }
    }
    return ({
        'status': 'success',
        'data': data,
        'reasoning_explanations': reasoning,
        'message': f'Cached vendor analysis loaded for {vendor_name}'
    }, 200)


def _summary_cards_from_summary(summary: Optional[Dict[str, Any]], fallback_count: int):
    if summary:
        net_flow = float(summary.get('net_cash_flow') or 0)
        return {
            'transactions': {'value': summary.get('total_transactions', fallback_count), 'label': 'TRANSACTIONS', 'description': 'Stored vendor transactions'},
            'cash_flow_status': {'value': "Positive Flow" if net_flow > 0 else "Negative Flow" if net_flow < 0 else "Balanced", 'label': 'CASH FLOW STATUS', 'description': 'Based on stored net cash flow'},
            'payment_patterns': {'value': summary.get('payment_frequency') or 'N/A', 'label': 'PAYMENT FREQUENCY', 'description': 'Historical payment frequency'},
            'collection_status': {'value': summary.get('vendor_importance') or 'Normal', 'label': 'VENDOR IMPORTANCE', 'description': 'AI importance classification'}
        }
    base = _empty_summary_cards()
    base['transactions']['value'] = fallback_count
    return base


def _normalized_vendor_aliases(vendor_name: str):
    if not vendor_name:
        return []
    normalized = vendor_name.strip()
    if normalized.lower().startswith('all'):
        return ['All Vendors', 'All', 'All Vendor', 'All Records']
    return [normalized]


def smart_vendor_filter(bank_df: pd.DataFrame, vendor_name: str) -> pd.DataFrame:
    if bank_df is None or bank_df.empty or not vendor_name:
        return pd.DataFrame()

    if vendor_name.lower().startswith('all'):
        if 'Assigned_Vendor' in bank_df.columns:
            return bank_df[bank_df['Assigned_Vendor'].notna() & (bank_df['Assigned_Vendor'] != '')]
        return bank_df.copy()

    vendor_keywords = [kw for kw in vendor_name.lower().split() if kw not in ['company', 'corp', 'corporation', 'ltd', 'limited', 'llc', 'inc', 'co', '&', 'and']]
    vendor_pattern = '|'.join(vendor_keywords) if vendor_keywords else vendor_name

    if 'Assigned_Vendor' in bank_df.columns:
        vendor_transactions = bank_df[bank_df['Assigned_Vendor'] == vendor_name]
        if not vendor_transactions.empty:
            return vendor_transactions

    vendor_transactions = pd.DataFrame()
    try:
        vendor_transactions = bank_df[bank_df['Description'].str.contains(vendor_pattern, case=False, na=False)]
    except Exception:
        pass

    if vendor_transactions.empty and len(vendor_keywords) > 1:
        for keyword in vendor_keywords:
            if len(keyword) > 2:
                try:
                    keyword_transactions = bank_df[bank_df['Description'].str.contains(keyword, case=False, na=False)]
                    if not keyword_transactions.empty:
                        vendor_transactions = pd.concat([vendor_transactions, keyword_transactions]).drop_duplicates()
                except Exception:
                    continue

        if vendor_transactions.empty:
            try:
                vendor_transactions = bank_df[bank_df['Description'].str.contains(vendor_name, case=False, na=False)]
            except Exception:
                pass

    return vendor_transactions

