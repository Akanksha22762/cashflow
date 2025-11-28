"""
Trend analytics service layer.
Provides reusable helpers for analytics routes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import time
import re

import numpy as np
import pandas as pd

from vendor_modules.session_utils import resolve_session_ids

TREND_TYPES = [
    "historical_revenue_trends",
    "sales_forecast",
    "customer_contracts",
    "pricing_models",
    "ar_aging",
    "operating_expenses",
    "accounts_payable",
    "inventory_turnover",
    "loan_repayments",
    "tax_obligations",
    "capital_expenditure",
    "equity_debt_inflows",
    "other_income_expenses",
    "cash_flow_types",
]

TREND_LABELS = {
    "historical_revenue_trends": "Historical Revenue Trends",
    "sales_forecast": "Sales Forecast",
    "customer_contracts": "Customer Contracts",
    "pricing_models": "Pricing Models",
    "ar_aging": "Accounts Receivable Aging",
    "operating_expenses": "Operating Expenses",
    "accounts_payable": "Accounts Payable",
    "inventory_turnover": "Inventory Turnover",
    "loan_repayments": "Loan Repayments",
    "tax_obligations": "Tax Obligations",
    "capital_expenditure": "Capital Expenditure",
    "equity_debt_inflows": "Equity & Debt Inflows",
    "other_income_expenses": "Other Income & Expenses",
    "cash_flow_types": "Cash Flow Types",
}


def build_trend_types_payload() -> Dict[str, Any]:
    """Return metadata for available trend types."""
    trend_options = [
        {
            "value": trend_type,
            "label": TREND_LABELS.get(trend_type, trend_type.replace("_", " ").title()),
            "category": _categorize_trend(trend_type),
        }
        for trend_type in TREND_TYPES
    ]

    return {
        "success": True,
        "trend_types": TREND_TYPES,
        "trend_options": trend_options,
        "total_available": len(TREND_TYPES),
        "categories": {
            "revenue": [opt for opt in trend_options if opt["category"] == "revenue"],
            "expenses": [opt for opt in trend_options if opt["category"] == "expenses"],
            "cash_flow": [opt for opt in trend_options if opt["category"] == "cash_flow"],
            "financial": [opt for opt in trend_options if opt["category"] == "financial"],
        },
    }


def validate_trend_selection_payload(selected_trends: List[str]) -> Dict[str, Any]:
    """Validate an incoming trend selection list."""
    if not selected_trends:
        return {"valid": False, "error": "At least one trend must be selected for analysis"}

    if len(selected_trends) > len(TREND_TYPES):
        return {"valid": False, "error": "Maximum 14 trends can be analyzed at once"}

    invalid_trends = [trend for trend in selected_trends if trend not in TREND_TYPES]
    if invalid_trends:
        return {"valid": False, "error": f"Invalid trend types: {invalid_trends}"}

    estimated_time = len(selected_trends) * 2.5
    scope = (
        "comprehensive"
        if len(selected_trends) == len(TREND_TYPES)
        else "multiple_specific"
        if len(selected_trends) > 1
        else "single"
    )

    return {
        "valid": True,
        "selected_count": len(selected_trends),
        "estimated_processing_time": f"{estimated_time:.1f} seconds",
        "analysis_scope": scope,
        "selected_trends": selected_trends,
    }


@dataclass
class TrendAnalysisContext:
    uploaded_data: Dict[str, Any]
    uploaded_bank_df: pd.DataFrame
    db_manager: Any
    database_available: bool
    state_manager: Any
    persistent_state_available: bool
    dynamic_trends_analyzer: Any
    analysis_storage_available: bool


def run_dynamic_trends_analysis_service(
    analysis_type: Any,
    vendor_name: str,
    flask_session: Dict[str, Any],
    context: TrendAnalysisContext,
) -> Tuple[Dict[str, Any], int]:
    """
    Execute the dynamic trends analysis. Returns (payload, status_code).
    """

    try:
        start_time = time.time()

        sample_df = _resolve_input_dataframe(context)
        sample_df = _apply_vendor_filter(sample_df, vendor_name)

        trend_types_to_analyze, analysis_scope, error = _determine_trend_scope(analysis_type)
        if error:
            return error, 400

        validation_error = _validate_trend_request(trend_types_to_analyze)
        if validation_error:
            return validation_error, 400

        cached_response = _check_cached_trends(
            trend_types_to_analyze, flask_session, context, vendor_name, analysis_type, analysis_scope, len(sample_df)
        )
        if cached_response:
            return cached_response, 200

        analyzer = context.dynamic_trends_analyzer
        if analyzer is None:
            return {
                "status": "info", 
                "message": "Analytics feature is currently under development. This feature will be available in a future update.",
                "feature_status": "coming_soon",
                "error": "DynamicTrendsAnalyzer not yet implemented"
            }, 503

        trends_results = analyzer.analyze_trends_batch(sample_df, trend_types_to_analyze)
        if "error" in trends_results:
            return {"status": "error", "error": trends_results["error"]}, 400

        processing_time = time.time() - start_time
        results = _build_success_response(
            trends_results, processing_time, sample_df, vendor_name, analysis_type, analysis_scope, trend_types_to_analyze
        )
        clean_results = _clean_nan_values(results)
        return clean_results, 200
    except Exception as exc:  # pragma: no cover - defensive logging
        import traceback

        print(f"❌ Dynamic trends analysis failed: {exc}")
        traceback.print_exc()
        return {"status": "error", "error": f"Dynamic trends analysis failed: {str(exc)}"}, 500


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _categorize_trend(trend_type: str) -> str:
    if "revenue" in trend_type or "sales" in trend_type:
        return "revenue"
    if "expense" in trend_type or "payable" in trend_type:
        return "expenses"
    if "cash" in trend_type:
        return "cash_flow"
    return "financial"


def _resolve_input_dataframe(context: TrendAnalysisContext) -> pd.DataFrame:
    """Determine which dataframe to use for analysis."""
    uploaded_bank_df = context.uploaded_bank_df
    uploaded_data = context.uploaded_data or {}

    bank_df = None
    if uploaded_bank_df is not None and not uploaded_bank_df.empty:
        bank_df = uploaded_bank_df
        print("✅ TRENDS FIX: Using session-restored bank data")
    elif uploaded_data.get("bank_df") is not None:
        bank_df = uploaded_data["bank_df"]
        print("✅ TRENDS FIX: Using fresh upload bank data")

    if bank_df is None or bank_df.empty:
        raise ValueError("No data available. Please upload files first or restore a session.")

    df = bank_df.copy()
    _normalize_dataframe(df)
    return df


def _normalize_dataframe(df: pd.DataFrame) -> None:
    """Normalize column names and data types required for analysis."""
    # First, try to find Inward_Amount and Outward_Amount (new format)
    inward_col = next((c for c in ["Inward_Amount", "Inward Amount", "inward_amount", "inward amount"] if c in df.columns), None)
    outward_col = next((c for c in ["Outward_Amount", "Outward Amount", "outward_amount", "outward amount"] if c in df.columns), None)
    
    # If we have both Inward and Outward, create Amount = Inward - Outward
    if inward_col is not None and outward_col is not None:
        df["Inward_Amount"] = pd.to_numeric(df[inward_col], errors="coerce").fillna(0)
        df["Outward_Amount"] = pd.to_numeric(df[outward_col], errors="coerce").fillna(0)
        # Create Amount column for backward compatibility: Inward - Outward (outward is negative)
        df["Amount"] = df["Inward_Amount"] + df["Outward_Amount"]  # Outward is already negative
    else:
        # Fallback to old Amount column if Inward/Outward not found
        amount_col = next((c for c in ["Amount", "amount", "_amount", "Credit Amount", "Debit Amount", "Balance"] if c in df.columns), None)
        if amount_col is None:
            numeric_cols = [c for c in df.columns if str(df[c].dtype).startswith(("float", "int"))]
            amount_col = numeric_cols[0] if numeric_cols else None

        if amount_col is not None:
            df["Amount"] = pd.to_numeric(df[amount_col], errors="coerce").fillna(0)

    date_col = next((c for c in ["Date", "date", "_date", "Transaction Date", "Transaction_Date"] if c in df.columns), None)
    if date_col is not None:
        df["Date"] = pd.to_datetime(df[date_col], errors="coerce")

    desc_col = next(
        (c for c in ["Description", "description", "_combined_description", "Transaction Description", "Narration"] if c in df.columns),
        None,
    )
    if desc_col is not None:
        df["Description"] = df[desc_col].astype(str)


def _apply_vendor_filter(df: pd.DataFrame, vendor_name: str) -> pd.DataFrame:
    if not vendor_name:
        return df

    print(f"🏢 Filtering data for vendor: {vendor_name}")
    desc_col = next((col for col in df.columns if "desc" in col.lower() or "description" in col.lower() or "narration" in col.lower()), None)
    if not desc_col:
        return df

    # Clean vendor keywords
    clean_vendor = re.sub(r"\s*\(vendor type:.*?\)\s*", "", vendor_name, flags=re.IGNORECASE)
    clean_vendor = re.sub(r"\b(vendor|supplier|company|corp|corporation|ltd|limited|inc|incorporated)\b", "", clean_vendor, flags=re.IGNORECASE)
    words = [word.strip() for word in clean_vendor.split() if len(word.strip()) > 2]
    vendor_keywords = [word for word in words if word not in ["the", "and", "for", "with", "from", "ltd", "inc"]]

    vendor_filtered_df = pd.DataFrame()

    try:
        exact_match = df[df[desc_col].str.contains(vendor_name, case=False, na=False)]
        if not exact_match.empty:
            vendor_filtered_df = exact_match
            print(f"✅ Found {len(exact_match)} transactions with exact vendor match")
    except Exception:
        pass

    if vendor_filtered_df.empty and vendor_keywords:
        for keyword in vendor_keywords:
            try:
                keyword_match = df[df[desc_col].str.contains(keyword, case=False, na=False)]
                if not keyword_match.empty:
                    vendor_filtered_df = pd.concat([vendor_filtered_df, keyword_match]).drop_duplicates()
            except Exception:
                continue

    if vendor_filtered_df.empty:
        raise ValueError(
            f"No transactions found for vendor: {vendor_name}. "
            "Try using a more general vendor name or check if the vendor exists in your data."
        )

    return vendor_filtered_df


def _determine_trend_scope(analysis_type: Any) -> Tuple[List[str], str, Dict[str, Any]]:
    if analysis_type == "all":
        return TREND_TYPES, "comprehensive", None
    if isinstance(analysis_type, list):
        return analysis_type, "multiple_specific" if len(analysis_type) > 1 else "single", None
    if isinstance(analysis_type, str) and "," in analysis_type:
        parsed = [trend.strip() for trend in analysis_type.split(",") if trend.strip()]
        scope = "multiple_specific" if len(parsed) > 1 else "single"
        return parsed, scope, None
    if isinstance(analysis_type, str):
        return [analysis_type], "single", None

    return [], "single", {"status": "error", "error": "Analysis type not specified"}


def _validate_trend_request(trend_types: List[str]) -> Dict[str, Any] | None:
    if not trend_types:
        return {"status": "error", "error": "At least one trend must be selected for analysis"}

    invalid_trends = [trend for trend in trend_types if trend not in TREND_TYPES]
    if invalid_trends:
        return {"status": "error", "error": f"Invalid trend types: {invalid_trends}. Valid types: {TREND_TYPES}"}

    if len(trend_types) > len(TREND_TYPES):
        return {"status": "error", "error": "Maximum 14 trends can be analyzed at once"}
    return None


def _check_cached_trends(
    trend_types: List[str],
    flask_session: Dict[str, Any],
    context: TrendAnalysisContext,
    vendor_name: str,
    analysis_type: Any,
    analysis_scope: str,
    dataset_size: int,
) -> Dict[str, Any] | None:
    if not (context.database_available and context.db_manager):
        return None

    try:
        session_id, _ = resolve_session_ids(
            flask_session or {},
            context.state_manager if context.persistent_state_available else None,
            context.db_manager,
        )

        if not session_id:
            return None

        cached_trends = context.db_manager.restore_multiple_trends_session(session_id)
        if not cached_trends or not cached_trends.get("main_analysis_data"):
            return None

        available_cached_trends = {
            detail.get("trend_type")
            for detail in cached_trends.get("trend_details", [])
            if detail.get("trend_type")
        }
        if not set(trend_types).issubset(available_cached_trends):
            return None

        trends_analysis = {}
        for detail in cached_trends.get("trend_details", []):
            trend_type = detail.get("trend_type")
            if trend_type in trend_types:
                trend_results = detail.get("trend_results_parsed", {})
                if trend_results:
                    trends_analysis[trend_type] = trend_results

        if not trends_analysis:
            return None

        cached_results = cached_trends["main_analysis_data"].get("results", {})
        if "data" not in cached_results:
            cached_results["data"] = {}
        cached_results["data"]["trends_analysis"] = trends_analysis

        summary = cached_results["data"].setdefault("analysis_summary", {})
        summary.update(
            {
                "vendor_filter": vendor_name if vendor_name else "Full Dataset",
                "analysis_type": analysis_type,
                "analysis_scope": analysis_scope,
                "selected_trends": trend_types,
                "trends_count": len(trend_types),
                "dataset_size": dataset_size,
                "cached_result": True,
            }
        )

        return {
            "status": "success",
            "data": cached_results.get("data", {}),
            "message": f"Cached trends analysis loaded for {len(trend_types)} trend types",
            "cached": True,
            "processing_time": 0.0,
        }
    except Exception as cache_error:  # pragma: no cover - best effort caching
        print(f"⚠️ Cache check failed: {cache_error}")
        return None


def _build_success_response(
    trends_results: Dict[str, Any],
    processing_time: float,
    sample_df: pd.DataFrame,
    vendor_name: str,
    analysis_type: Any,
    analysis_scope: str,
    trend_types: List[str],
) -> Dict[str, Any]:
    summary = trends_results.get("_summary", {})
    return {
        "status": "success",
        "message": (
            f"Dynamic trends analysis completed successfully in {processing_time:.2f}s - "
            f"{len(trend_types)} trend type(s) analyzed"
        ),
        "data": {
            "trends_analysis": trends_results,
            "analysis_summary": {
                "total_trends_analyzed": summary.get("total_trends_analyzed", 0),
                "successful_analyses": summary.get("successful_analyses", 0),
                "processing_time": processing_time,
                "dataset_size": len(sample_df),
                "vendor_filter": vendor_name if vendor_name else "Full Dataset",
                "analysis_type": analysis_type,
                "analysis_scope": analysis_scope,
                "selected_trends": trend_types,
                "trends_count": len(trend_types),
                "is_multiple_trends": len(trend_types) > 1,
                "is_comprehensive": analysis_scope == "comprehensive",
                "openai_integration": "Active",
                "caching_enabled": "Yes",
                "batch_processing": "Yes",
            },
        },
    }


def _clean_nan_values(obj: Any) -> Any:
    """Recursively replace NaN/inf values so JSON serialization succeeds."""
    import math

    if isinstance(obj, dict):
        return {key: _clean_nan_values(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_clean_nan_values(value) for value in obj]
    if isinstance(obj, (np.float64, np.float32)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if pd.isna(obj):
        return None
    if obj != obj:  # NaN check
        return None
    return obj


