"""
Final Report Generator
======================

Builds a structured cash-flow report that combines:
1. Overall inflow, outflow, and net cash metrics (by category and total)
2. Vendor-level inflow/outflow visibility for reporting pages

The module is intentionally standalone so it can be imported by Flask routes,
Celery jobs, or ad-hoc scripts without dragging the gigantic `app.py`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import pandas as pd


# --------------------------------------------------------------------------- #
# Data contracts
# --------------------------------------------------------------------------- #


CATEGORY_LABELS = (
    "Operating Activities",
    "Investing Activities",
    "Financing Activities",
    "More information needed",
)


@dataclass
class CashflowSummary:
    total_inflows: float
    total_outflows: float
    net_cash_flow: float
    category_totals: Dict[str, Dict[str, float]]
    transaction_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VendorSummary:
    vendor_name: str
    inflows: float
    outflows: float
    net_cash_flow: float
    transaction_count: int
    # ✅ Removed opening_balance and closing_balance - these are account-level, not vendor-level
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        # Drop empty metadata blocks to keep payload clean
        if not self.metadata:
            payload.pop("metadata", None)
        return payload


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


ReportableInput = Union[pd.DataFrame, Sequence[Dict[str, Any]], Dict[str, Sequence[Any]]]


def generate_final_report(
    transactions: ReportableInput,
    vendor_master: Optional[ReportableInput] = None,
    include_raw: bool = False,
) -> Dict[str, Any]:
    """
    Build the final cash-flow report payload used by the report page.

    Parameters
    ----------
    transactions:
        Categorized transaction data. Must ultimately resolve to a DataFrame
        containing `Amount`, `Category`, and (optionally) `Vendor`.
    vendor_master:
        Optional vendor master (DataFrame or list-of-dicts) used to append IDs,
        categories, payment terms, etc., to the vendor breakdown.
    include_raw:
        When True, append the cleaned transaction DataFrame (as records) so the
        frontend can render drill-down tables without recomputing.
    """

    print("📊 [FINAL_REPORT] Step 4b: Coercing dataframes...")
    transaction_df = _coerce_dataframe(transactions, "transactions")
    vendor_df = _coerce_dataframe(vendor_master, "vendor master") if vendor_master is not None else None

    print("📊 [FINAL_REPORT] Step 4c: Preparing transactions...")
    prepared_df = _prepare_transactions(transaction_df)
    
    print("📊 [FINAL_REPORT] Step 4d: Building cashflow summary...")
    summary = _build_cashflow_summary(prepared_df)
    
    print("📊 [FINAL_REPORT] Step 4e: Building vendor section...")
    try:
        vendors = _build_vendor_section(prepared_df, vendor_df)
        print(f"📊 [FINAL_REPORT] Step 4e complete: {len(vendors)} vendors processed")
    except Exception as vendor_error:
        print(f"❌ [FINAL_REPORT] Step 4e FAILED: {vendor_error}")
        import traceback
        traceback.print_exc()
        vendors = []  # Continue with empty vendors list

    print("📊 [FINAL_REPORT] Step 4f: Building cash flow statement...")
    try:
        cashflow_statement = _build_cashflow_statement(prepared_df, summary)
        print("📊 [FINAL_REPORT] Step 4f complete: Cash flow statement generated")
    except Exception as cfs_error:
        print(f"⚠️ [FINAL_REPORT] Step 4f FAILED: {cfs_error}")
        import traceback
        traceback.print_exc()
        cashflow_statement = None  # Continue without cash flow statement
    
    print("📊 [FINAL_REPORT] Step 4g: Building report dict...")
    report = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "summary": summary.to_dict(),
        "vendors": [vendor.to_dict() for vendor in vendors],
        "cashflow_statement": cashflow_statement,
    }

    if include_raw:
        print("📊 [FINAL_REPORT] Step 4h: Adding raw transactions...")
        # ✅ Clean NaN values before converting to dict (NaN is not valid JSON)
        prepared_df_clean = prepared_df.copy()
        # Replace NaN/NaT with None (JSON will serialize None as null)
        prepared_df_clean = prepared_df_clean.where(pd.notna(prepared_df_clean), None)
        report["transactions"] = prepared_df_clean.to_dict(orient="records")

    print("📊 [FINAL_REPORT] Step 4i: Report generation complete")
    return report


# --------------------------------------------------------------------------- #
# Cash-flow calculations
# --------------------------------------------------------------------------- #


def _build_cashflow_summary(df: pd.DataFrame) -> CashflowSummary:
    # Use Inward_Amount and Outward_Amount directly - no fallback
    if 'Inward_Amount' not in df.columns or 'Outward_Amount' not in df.columns:
        raise ValueError("Inward_Amount and Outward_Amount columns are required. Amount column is not used.")
    
    inflows = float(df['Inward_Amount'].fillna(0).sum())
    outflows = float(df['Outward_Amount'].fillna(0).apply(lambda x: abs(float(x)) if pd.notna(x) else 0.0).sum())
    
    category_totals: Dict[str, Dict[str, float]] = {}

    for label in CATEGORY_LABELS:
        cat_df = df[df["Category"] == label]
        if cat_df.empty:
            category_totals[label] = {
                "inflows": 0.0,
                "outflows": 0.0,
                "net": 0.0,
                "count": 0,
            }
            continue

        # Use Inward_Amount and Outward_Amount directly - no fallback
        cat_inflows = float(cat_df['Inward_Amount'].fillna(0).sum())
        cat_outflows = float(cat_df['Outward_Amount'].fillna(0).apply(lambda x: abs(float(x)) if pd.notna(x) else 0.0).sum())
        cat_net = float(cat_inflows - cat_outflows)
        
        category_totals[label] = {
            "inflows": cat_inflows,
            "outflows": cat_outflows,
            "net": cat_net,
            "count": int(len(cat_df)),
        }

    summary = CashflowSummary(
        total_inflows=inflows,
        total_outflows=outflows,
        net_cash_flow=float(inflows - outflows),
        category_totals=category_totals,
        transaction_count=len(df),
    )

    return summary


def _build_vendor_section(df: pd.DataFrame, vendor_df: Optional[pd.DataFrame]) -> List[VendorSummary]:
    if "Vendor" not in df.columns:
        # Safe default when upstream flow does not provide vendors yet.
        return []

    vendor_metadata_lookup = _build_vendor_metadata_lookup(vendor_df) if vendor_df is not None else {}
    vendor_summaries: List[VendorSummary] = []

    # Sort entire dataframe by date for accurate balance calculation
    df_sorted = df.copy()
    if "Date" in df_sorted.columns:
        df_sorted = df_sorted.sort_values("Date").reset_index(drop=False)
        if "index" in df_sorted.columns:
            df_sorted = df_sorted.set_index("index")
    
    for vendor_name, vendor_transactions in df.groupby("Vendor"):
        try:
            vendor_transactions = vendor_transactions.copy()
            
            # ✅ Sort vendor transactions by date to get correct first/last
            if "Date" in vendor_transactions.columns:
                vendor_transactions = vendor_transactions.sort_values("Date", ascending=True).reset_index(drop=True)
            
            # Use Inward_Amount and Outward_Amount directly - no fallback
            if 'Inward_Amount' not in vendor_transactions.columns or 'Outward_Amount' not in vendor_transactions.columns:
                print(f"⚠️ [FINAL_REPORT] Skipping vendor '{vendor_name}': Missing Inward_Amount or Outward_Amount columns")
                continue
            
            inflows = float(vendor_transactions['Inward_Amount'].fillna(0).sum())
            outflows = float(vendor_transactions['Outward_Amount'].fillna(0).apply(lambda x: abs(float(x)) if pd.notna(x) else 0.0).sum())
            
            # ✅ Opening/Closing balances are account-level, not vendor-level
            # Removed per-vendor balance calculation - not meaningful at vendor level
        except Exception as vendor_processing_error:
            print(f"⚠️ [FINAL_REPORT] Error processing vendor '{vendor_name}': {vendor_processing_error}")
            import traceback
            traceback.print_exc()
            continue
        
        vendor_summary = VendorSummary(
            vendor_name=str(vendor_name),
            inflows=inflows,
            outflows=outflows,
            net_cash_flow=float(inflows - outflows),
            transaction_count=int(len(vendor_transactions)),
            # ✅ Removed opening_balance and closing_balance - account-level metrics
            metadata=vendor_metadata_lookup.get(str(vendor_name)),
        )
        vendor_summaries.append(vendor_summary)

    # Sort vendors by absolute net impact so the UI can highlight critical ones
    vendor_summaries.sort(key=lambda vendor: abs(vendor.net_cash_flow), reverse=True)
    return vendor_summaries


def _build_cashflow_statement(df: pd.DataFrame, summary: CashflowSummary) -> Dict[str, Any]:
    """
    Build a formatted Cash Flow Statement (Direct Method) in standard accounting format.
    
    Returns a structured dict with sections for Operating, Investing, and Financing Activities,
    along with opening/closing balances and date range.
    """
    # Get date range
    period_start = None
    period_end = None
    if "Date" in df.columns:
        try:
            dates = pd.to_datetime(df["Date"], errors='coerce').dropna()
            if not dates.empty:
                period_start = dates.min()
                period_end = dates.max()
        except Exception:
            pass
    
    # Get opening and closing balances - find balance column with case-insensitive search
    opening_balance = None
    closing_balance = None
    
    # Find balance column (case-insensitive) - prioritize 'closing_balance' if it exists (more accurate)
    balance_col = None
    # First, try to find 'closing_balance' (most accurate - this is the recalculated balance)
    for col in df.columns:
        col_lower = str(col).lower().strip().replace(' ', '_')
        if col_lower == 'closing_balance':
            balance_col = col
            break
    
    # If not found, fall back to 'balance'
    if not balance_col:
        for col in df.columns:
            col_lower = str(col).lower().strip().replace(' ', '_')
            if col_lower == 'balance':
                balance_col = col
                break
    
    if balance_col and len(df) > 0:
        print(f"📊 [CASHFLOW_STMT] Found balance column: {balance_col}")
        
        # Debug: Show all balance values before sorting
        print(f"📊 [CASHFLOW_STMT] All {balance_col} values (before sort): {df[balance_col].tolist()}")
        if 'closing_balance' in df.columns:
            print(f"📊 [CASHFLOW_STMT] All closing_balance values (before sort): {df['closing_balance'].tolist()}")
        
        # Sort by date to get first (oldest) and last (newest) transactions
        df_sorted = df.copy()
        sort_col = None
        
        # Try to find date column for sorting - prioritize lowercase 'date' (from API/database)
        if 'date' in df_sorted.columns:
            sort_col = 'date'
        elif 'Date' in df_sorted.columns:
            sort_col = 'Date'
        else:
            # Try DateTime column
            for col in df_sorted.columns:
                col_lower = str(col).lower().strip()
                if col_lower == 'datetime':
                    sort_col = col
                    break
        
        if sort_col:
            try:
                # Ensure date column is datetime type for proper sorting
                df_sorted[sort_col] = pd.to_datetime(df_sorted[sort_col], errors='coerce', dayfirst=True)
                
                # Check current sort order by comparing first and last dates
                if len(df_sorted) > 1:
                    first_date_check = df_sorted[sort_col].iloc[0]
                    last_date_check = df_sorted[sort_col].iloc[-1]
                    is_ascending = pd.notna(first_date_check) and pd.notna(last_date_check) and first_date_check <= last_date_check
                    print(f"📊 [CASHFLOW_STMT] Before sort: First date={first_date_check}, Last date={last_date_check}, Currently {'ascending' if is_ascending else 'descending'}")
                
                # Always sort ascending (oldest first) to get chronological order for balance calculation
                df_sorted = df_sorted.sort_values(sort_col, ascending=True, na_position='last').reset_index(drop=True)
                
                # Verify after sorting
                if len(df_sorted) > 1:
                    first_date_after = df_sorted[sort_col].iloc[0]
                    last_date_after = df_sorted[sort_col].iloc[-1]
                    print(f"📊 [CASHFLOW_STMT] After sort ascending: First date={first_date_after}, Last date={last_date_after}")
                
                print(f"📊 [CASHFLOW_STMT] Sorted {len(df_sorted)} rows by {sort_col} ascending (oldest first)")
            except Exception as sort_error:
                print(f"⚠️ [CASHFLOW_STMT] Error sorting: {sort_error}")
                import traceback
                traceback.print_exc()
                # Continue without sorting - use dataframe as-is
        else:
            print(f"⚠️ [CASHFLOW_STMT] No date column found for sorting. Available columns: {list(df_sorted.columns)[:10]}")
        
        # Get balances directly from sorted dataframe rows (not from filtered series)
        # First row (index 0) = oldest transaction, Last row (index -1) = newest transaction
        if len(df_sorted) > 0:
            # Get first (oldest) transaction's balance
            first_balance_raw = df_sorted[balance_col].iloc[0]
            # Get last (newest) transaction's balance  
            last_balance_raw = df_sorted[balance_col].iloc[-1]
            
            if pd.notna(first_balance_raw) and pd.notna(last_balance_raw):
                first_balance_after = float(first_balance_raw)  # Balance AFTER oldest transaction
                last_balance_after = float(last_balance_raw)  # Balance AFTER newest transaction (this is closing balance)
                
                # Calculate opening balance (balance BEFORE first/oldest transaction)
                # Use the SAME formula as preprocessing to ensure consistency
                # Formula: Balance_after = Balance_before + Inward - Outward
                # So: Balance_before = Balance_after - Inward + Outward_abs
                first_inward = float(df_sorted['Inward_Amount'].iloc[0]) if pd.notna(df_sorted['Inward_Amount'].iloc[0]) else 0.0
                first_outward = float(df_sorted['Outward_Amount'].iloc[0]) if pd.notna(df_sorted['Outward_Amount'].iloc[0]) else 0.0
                
                # Outward amounts are already negative after preprocessing (e.g., -626.40)
                # Use same formula as preprocessing: opening_balance = first_balance - first_inward + first_outward_abs
                first_outward_abs = abs(first_outward) if first_outward != 0 else 0.0
                opening_balance = first_balance_after - first_inward + first_outward_abs
                closing_balance = last_balance_after  # This IS the closing balance (newest transaction's balance)
                
                # Also get dates for verification
                first_date = str(df_sorted[sort_col].iloc[0]) if sort_col else "N/A"
                last_date = str(df_sorted[sort_col].iloc[-1]) if sort_col else "N/A"
                
                # Debug: Show all balance values after sorting to verify
                print(f"📊 [CASHFLOW_STMT] All {balance_col} values after sort (oldest to newest): {df_sorted[balance_col].tolist()}")
                if 'closing_balance' in df_sorted.columns and balance_col != 'closing_balance':
                    print(f"📊 [CASHFLOW_STMT] All closing_balance values after sort: {df_sorted['closing_balance'].tolist()}")
                
                print(f"📊 [CASHFLOW_STMT] Oldest transaction ({first_date}): Balance={first_balance_after}, Inward={first_inward}, Outward={first_outward} (abs: {first_outward_abs})")
                print(f"📊 [CASHFLOW_STMT] Newest transaction ({last_date}): Balance={last_balance_after} (this IS the closing balance)")
                print(f"📊 [CASHFLOW_STMT] Calculated opening balance (before oldest tx): {opening_balance}")
                print(f"📊 [CASHFLOW_STMT] Closing balance (from newest tx): {closing_balance}")
            else:
                print(f"⚠️ [CASHFLOW_STMT] Balance values are NaN: first={first_balance_raw}, last={last_balance_raw}")
    else:
        print(f"⚠️ [CASHFLOW_STMT] No balance column found. Available columns: {list(df.columns)[:10]}")
    
    # Calculate net increase in cash
    net_increase = summary.net_cash_flow
    
    # Build Operating Activities section with line items
    operating_df = df[df["Category"] == "Operating Activities"].copy() if "Category" in df.columns else pd.DataFrame()
    operating_inflows = float(operating_df['Inward_Amount'].fillna(0).sum()) if not operating_df.empty else 0.0
    operating_outflows = float(operating_df['Outward_Amount'].fillna(0).apply(lambda x: abs(float(x)) if pd.notna(x) else 0.0).sum()) if not operating_df.empty else 0.0
    
    # Break down operating activities into line items (returns separate inflow/outflow dicts)
    operating_inflow_items, operating_outflow_items = _categorize_operating_activities(operating_df)
    
    # Build Investing Activities section
    investing_df = df[df["Category"] == "Investing Activities"].copy() if "Category" in df.columns else pd.DataFrame()
    investing_inflows = float(investing_df['Inward_Amount'].fillna(0).sum()) if not investing_df.empty else 0.0
    investing_outflows = float(investing_df['Outward_Amount'].fillna(0).apply(lambda x: abs(float(x)) if pd.notna(x) else 0.0).sum()) if not investing_df.empty else 0.0
    investing_inflow_items, investing_outflow_items = _categorize_investing_activities(investing_df)
    
    # Build Financing Activities section
    financing_df = df[df["Category"] == "Financing Activities"].copy() if "Category" in df.columns else pd.DataFrame()
    financing_inflows = float(financing_df['Inward_Amount'].fillna(0).sum()) if not financing_df.empty else 0.0
    financing_outflows = float(financing_df['Outward_Amount'].fillna(0).apply(lambda x: abs(float(x)) if pd.notna(x) else 0.0).sum()) if not financing_df.empty else 0.0
    financing_inflow_items, financing_outflow_items = _categorize_financing_activities(financing_df)
    
    # Format date range string
    period_str = ""
    if period_start and period_end:
        try:
            start_str = period_start.strftime("%d %B %Y") if hasattr(period_start, 'strftime') else str(period_start)
            end_str = period_end.strftime("%d %B %Y") if hasattr(period_end, 'strftime') else str(period_end)
            period_str = f"{start_str} – {end_str}"
        except Exception:
            period_str = f"{period_start} – {period_end}"
    
    return {
        "period": period_str,
        "period_start": period_start.isoformat() if period_start and hasattr(period_start, 'isoformat') else str(period_start) if period_start else None,
        "period_end": period_end.isoformat() if period_end and hasattr(period_end, 'isoformat') else str(period_end) if period_end else None,
        "operating_activities": {
            "inflow_items": operating_inflow_items,
            "total_inflows": operating_inflows,
            "outflow_items": operating_outflow_items,
            "total_outflows": operating_outflows,
            "net_cash_flow": operating_inflows - operating_outflows,
        },
        "investing_activities": {
            "inflow_items": investing_inflow_items,
            "total_inflows": investing_inflows,
            "outflow_items": investing_outflow_items,
            "total_outflows": investing_outflows,
            "net_cash_flow": investing_inflows - investing_outflows,
        },
        "financing_activities": {
            "inflow_items": financing_inflow_items,
            "total_inflows": financing_inflows,
            "outflow_items": financing_outflow_items,
            "total_outflows": financing_outflows,
            "net_cash_flow": financing_inflows - financing_outflows,
        },
        "net_increase_in_cash": net_increase,
        "opening_cash_balance": opening_balance,
        "closing_cash_balance": closing_balance,
    }


def _categorize_operating_activities(df: pd.DataFrame) -> tuple[Dict[str, float], Dict[str, float]]:
    """Break down operating activities into common line items. Returns (inflow_items, outflow_items)."""
    if df.empty or 'Description' not in df.columns:
        return {}, {}
    
    inflow_items = {
        "Cash received from customers": 0.0,
        "Scrap sales received": 0.0,
        "Other operating receipts": 0.0,
    }
    
    outflow_items = {
        "Payments to suppliers": 0.0,
        "Employee payments": 0.0,
        "Utilities and other operating expenses": 0.0,
    }
    
    # Use vendor/description to categorize
    for idx, row in df.iterrows():
        desc = str(row.get('Description', '')).lower()
        vendor = str(row.get('Vendor', '')).lower()
        inward = float(row.get('Inward_Amount', 0) or 0)
        outward = float(row.get('Outward_Amount', 0) or 0)
        outward = abs(outward)
        
        # Categorize inflows
        if inward > 0:
            if any(word in desc for word in ['scrap', 'sale of', 'disposal']):
                inflow_items["Scrap sales received"] += inward
            elif any(word in desc or word in vendor for word in ['customer', 'client', 'revenue', 'income', 'payment received']):
                inflow_items["Cash received from customers"] += inward
            else:
                inflow_items["Other operating receipts"] += inward
        
        # Categorize outflows
        if outward > 0:
            if any(word in desc or word in vendor for word in ['salary', 'wage', 'employee', 'payroll', 'staff']):
                outflow_items["Employee payments"] += outward
            elif any(word in desc or word in vendor for word in ['supplier', 'vendor', 'purchase', 'material', 'inventory']):
                outflow_items["Payments to suppliers"] += outward
            elif any(word in desc or word in vendor for word in ['utility', 'electricity', 'water', 'telephone', 'internet', 'rent', 'maintenance', 'repair']):
                outflow_items["Utilities and other operating expenses"] += outward
            else:
                outflow_items["Payments to suppliers"] += outward  # Default to suppliers
    
    # Remove zero items
    return (
        {k: v for k, v in inflow_items.items() if v > 0},
        {k: v for k, v in outflow_items.items() if v > 0}
    )


def _categorize_investing_activities(df: pd.DataFrame) -> tuple[Dict[str, float], Dict[str, float]]:
    """Break down investing activities into common line items. Returns (inflow_items, outflow_items)."""
    if df.empty:
        return {}, {}
    
    inflow_items = {
        "Proceeds from sale of assets": 0.0,
        "Other investing receipts": 0.0,
    }
    
    outflow_items = {
        "Purchase of machinery": 0.0,
        "Purchase of equipment": 0.0,
        "Other investing activities": 0.0,
    }
    
    for idx, row in df.iterrows():
        desc = str(row.get('Description', '')).lower()
        inward = float(row.get('Inward_Amount', 0) or 0)
        outward = float(row.get('Outward_Amount', 0) or 0)
        outward = abs(outward)
        
        if inward > 0:
            if any(word in desc for word in ['sale', 'proceed', 'disposal', 'realization']):
                inflow_items["Proceeds from sale of assets"] += inward
            else:
                inflow_items["Other investing receipts"] += inward
        
        if outward > 0:
            if any(word in desc for word in ['machinery', 'machine', 'equipment', 'plant']):
                outflow_items["Purchase of machinery"] += outward
            elif any(word in desc for word in ['equipment', 'tool', 'instrument']):
                outflow_items["Purchase of equipment"] += outward
            else:
                outflow_items["Other investing activities"] += outward
    
    return (
        {k: v for k, v in inflow_items.items() if v > 0},
        {k: v for k, v in outflow_items.items() if v > 0}
    )


def _categorize_financing_activities(df: pd.DataFrame) -> tuple[Dict[str, float], Dict[str, float]]:
    """Break down financing activities into common line items. Returns (inflow_items, outflow_items)."""
    if df.empty:
        return {}, {}
    
    inflow_items = {
        "Loan proceeds received": 0.0,
        "Other financing receipts": 0.0,
    }
    
    outflow_items = {
        "Loan repayment": 0.0,
        "Other financing payments": 0.0,
    }
    
    for idx, row in df.iterrows():
        desc = str(row.get('Description', '')).lower()
        inward = float(row.get('Inward_Amount', 0) or 0)
        outward = float(row.get('Outward_Amount', 0) or 0)
        outward = abs(outward)
        
        if inward > 0:
            if any(word in desc for word in ['loan', 'credit', 'borrowing', 'advance']):
                inflow_items["Loan proceeds received"] += inward
            else:
                inflow_items["Other financing receipts"] += inward
        
        if outward > 0:
            if any(word in desc for word in ['loan', 'repayment', 'repay', 'interest']):
                outflow_items["Loan repayment"] += outward
            else:
                outflow_items["Other financing payments"] += outward
    
    return (
        {k: v for k, v in inflow_items.items() if v > 0},
        {k: v for k, v in outflow_items.items() if v > 0}
    )


# --------------------------------------------------------------------------- #
# Preparation helpers
# --------------------------------------------------------------------------- #


def _coerce_dataframe(data: Optional[ReportableInput], label: str) -> pd.DataFrame:
    if data is None:
        raise ValueError(f"{label.title()} data is required to generate the report.")

    if isinstance(data, pd.DataFrame):
        return data.copy()

    if isinstance(data, dict):
        return pd.DataFrame(data)

    if isinstance(data, Sequence):
        # Filter out dataclass instances or other objects that can't be coerced
        try:
            return pd.DataFrame(list(data))
        except ValueError as exc:
            raise ValueError(f"Unable to convert {label} sequence into DataFrame") from exc

    raise TypeError(f"Unsupported {label} input type: {type(data)}")


def _prepare_transactions(df: pd.DataFrame) -> pd.DataFrame:
    normalized_df = df.copy()

    _ensure_amount_column(normalized_df)
    _ensure_inward_outward_columns(normalized_df)  # Ensure Inward_Amount and Outward_Amount exist
    _ensure_category_column(normalized_df)
    _ensure_vendor_column(normalized_df)
    _ensure_date_column(normalized_df)

    # Reorder canonical columns to keep downstream serialization predictable
    canonical_columns = [col for col in ["Date", "Description", "Vendor", "Category", "Amount"] if col in normalized_df.columns]
    remaining_columns = [col for col in normalized_df.columns if col not in canonical_columns]

    normalized_df = normalized_df[canonical_columns + remaining_columns]
    return normalized_df


def _ensure_amount_column(df: pd.DataFrame) -> None:
    # Amount column is not required - we use Inward_Amount and Outward_Amount directly
    # This function is kept for compatibility but doesn't enforce Amount requirement
    if "Amount" in df.columns:
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)
    # If Amount doesn't exist, that's fine - we use Inward_Amount and Outward_Amount


def _ensure_inward_outward_columns(df: pd.DataFrame) -> None:
    """
    Ensure Inward_Amount and Outward_Amount columns exist.
    If they don't exist, derive them from Amount column (for backward compatibility).
    """
    # ✅ FIRST: Normalize column names (handle lowercase variations)
    column_mapping = {}
    for col in df.columns:
        col_lower = str(col).lower().strip()
        if 'inward' in col_lower and 'amount' in col_lower and col != 'Inward_Amount':
            column_mapping[col] = 'Inward_Amount'
        elif 'outward' in col_lower and 'amount' in col_lower and col != 'Outward_Amount':
            column_mapping[col] = 'Outward_Amount'
    
    if column_mapping:
        df.rename(columns=column_mapping, inplace=True)
        print(f"📊 [FINAL_REPORT] Normalized column names: {column_mapping}")
    
    # Check if Inward_Amount and Outward_Amount exist (even if all zeros - that's valid data)
    has_inward = 'Inward_Amount' in df.columns
    has_outward = 'Outward_Amount' in df.columns
    
    # If both exist (regardless of values), use them directly
    if has_inward and has_outward:
        # Ensure they're numeric (convert to float, handle NaN)
        df['Inward_Amount'] = pd.to_numeric(df['Inward_Amount'], errors='coerce').fillna(0.0)
        df['Outward_Amount'] = pd.to_numeric(df['Outward_Amount'], errors='coerce').fillna(0.0)
        print(f"📊 [FINAL_REPORT] Using Inward_Amount and Outward_Amount columns directly")
        print(f"📊 [FINAL_REPORT] Inward_Amount sample: {df['Inward_Amount'].head(3).tolist()}")
        print(f"📊 [FINAL_REPORT] Outward_Amount sample: {df['Outward_Amount'].head(3).tolist()}")
        return
    
    # If missing or all null/zero, derive from Amount if available
    if 'Amount' in df.columns:
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0.0)
        
        if not has_inward:
            # Derive Inward_Amount from Amount: positive amounts = inward
            df['Inward_Amount'] = df['Amount'].apply(lambda x: x if x > 0 else 0.0)
        
        if not has_outward:
            # Derive Outward_Amount from Amount: negative amounts = outward (as positive)
            df['Outward_Amount'] = df['Amount'].apply(lambda x: abs(x) if x < 0 else 0.0)
    else:
        # If Amount doesn't exist and Inward_Amount/Outward_Amount are missing, create empty columns
        if 'Inward_Amount' not in df.columns:
            df['Inward_Amount'] = 0.0
        if 'Outward_Amount' not in df.columns:
            df['Outward_Amount'] = 0.0
    
    # Ensure they're numeric
    df['Inward_Amount'] = pd.to_numeric(df['Inward_Amount'], errors='coerce').fillna(0.0)
    df['Outward_Amount'] = pd.to_numeric(df['Outward_Amount'], errors='coerce').fillna(0.0)


def _ensure_category_column(df: pd.DataFrame) -> None:
    if "Category" not in df.columns:
        for candidate in ("category", "Cash Flow Category", "_category"):
            if candidate in df.columns:
                df.rename(columns={candidate: "Category"}, inplace=True)
                break

    if "Category" not in df.columns:
        df["Category"] = "Operating Activities"

    df["Category"] = df["Category"].apply(_normalize_category)


def _ensure_vendor_column(df: pd.DataFrame) -> None:
    if "Vendor" not in df.columns:
        for candidate in ("vendor", "Vendor Name", "vendor_name"):
            if candidate in df.columns:
                df.rename(columns={candidate: "Vendor"}, inplace=True)
                break

    if "Vendor" not in df.columns:
        # Optional column; fill with placeholder for grouping consistency
        df["Vendor"] = "Unknown Vendor"

    df["Vendor"] = df["Vendor"].fillna("Unknown Vendor")


def _ensure_date_column(df: pd.DataFrame) -> None:
    if "Date" in df.columns:
        return

    for candidate in ("date", "_date", "Transaction Date"):
        if candidate in df.columns:
            df.rename(columns={candidate: "Date"}, inplace=True)
            return

    # Dates are optional for aggregation, but useful for the UI, so stub them
    df["Date"] = pd.NaT


def _build_vendor_metadata_lookup(vendor_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    if vendor_df.empty:
        return {}

    lookup = {}
    column_map = {
        "Vendor Name": "vendor_name",
        "Vendor ID": "vendor_id",
        "Category": "category",
        "Payment Terms": "payment_terms",
    }

    available_columns = [col for col in column_map if col in vendor_df.columns]
    if not available_columns and "vendor_name" not in vendor_df.columns:
        return {}

    for _, row in vendor_df.iterrows():
        vendor_name = (
            row.get("Vendor Name")
            or row.get("vendor_name")
            or row.get("Vendor")
            or "Unknown Vendor"
        )

        metadata = {}
        for source_column, target_field in column_map.items():
            if source_column in vendor_df.columns and pd.notna(row.get(source_column)):
                metadata[target_field] = row[source_column]

        # Include any direct snake_case columns (common in API responses)
        for column in ("vendor_id", "category", "payment_terms"):
            if column in vendor_df.columns and pd.notna(row.get(column)):
                metadata[column] = row[column]

        lookup[str(vendor_name)] = metadata or None

    return lookup


def _normalize_category(raw_category: Any) -> str:
    if not raw_category or not isinstance(raw_category, str):
        return "Operating Activities"

    normalized = raw_category.strip()
    for label in CATEGORY_LABELS:
        if label.lower() in normalized.lower():
            return label

    return "Operating Activities"


# --------------------------------------------------------------------------- #
# Optional CLI usage for quick debugging
# --------------------------------------------------------------------------- #


def preview_report(transactions_csv: str, vendor_csv: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience helper so developers can quickly preview the report from shell:

        python - <<'PY'
        from reports.final_report_generator import preview_report
        report = preview_report("uploads/bank_abcd.xlsx")
        print(report["summary"])
        PY
    """

    tx_df = pd.read_csv(transactions_csv)
    vendor_df = pd.read_csv(vendor_csv) if vendor_csv else None
    return generate_final_report(tx_df, vendor_df)


__all__ = [
    "generate_final_report",
    "preview_report",
    "CashflowSummary",
    "VendorSummary",
]

