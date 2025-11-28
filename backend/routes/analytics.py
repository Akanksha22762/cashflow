"""
Analytics Routes
Routes for trend analysis and forecasting
"""
from flask import Blueprint, request, jsonify, session

from analytics_modules import (
    build_trend_types_payload,
    validate_trend_selection_payload,
    run_dynamic_trends_analysis_service,
    TrendAnalysisContext,
)

# Create blueprint
bp = Blueprint('analytics', __name__)


@bp.route('/get-available-trend-types', methods=['GET'])
def get_available_trend_types():
    """Get list of all available trend types for UI selection"""
    try:
        return jsonify(build_trend_types_payload())
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error getting available trend types: {str(e)}'
        }), 500


@bp.route('/validate-trend-selection', methods=['POST'])
def validate_trend_selection():
    """Validate user's trend selection before processing"""
    try:
        data = request.get_json()
        selected_trends = data.get('selected_trends', [])
        return jsonify(validate_trend_selection_payload(selected_trends))
        
    except Exception as e:
        return jsonify({
            'valid': False,
            'error': f'Validation error: {str(e)}'
        }), 500


@bp.route('/run-dynamic-trends-analysis', methods=['POST'])
def run_dynamic_trends_analysis():
    """
    Run dynamic trends analysis with OpenAI integration and intelligent caching
    NOTE: This is a complex route (800+ lines in original). Extracted with helper functions.
    """
    # Import here to avoid circular dependencies
    from app_setup import (
        uploaded_data, uploaded_bank_df, db_manager, DATABASE_AVAILABLE,
        state_manager, PERSISTENT_STATE_AVAILABLE, dynamic_trends_analyzer,
        ANALYSIS_STORAGE_AVAILABLE
    )
    data = request.get_json()
    analysis_type = data.get('analysis_type', 'all')
    vendor_name = data.get('vendor_name', '')

    context = TrendAnalysisContext(
        uploaded_data=uploaded_data,
        uploaded_bank_df=uploaded_bank_df,
        db_manager=db_manager,
        database_available=DATABASE_AVAILABLE,
        state_manager=state_manager,
        persistent_state_available=PERSISTENT_STATE_AVAILABLE,
        dynamic_trends_analyzer=dynamic_trends_analyzer,
        analysis_storage_available=ANALYSIS_STORAGE_AVAILABLE,
    )

    response, status_code = run_dynamic_trends_analysis_service(
        analysis_type=analysis_type,
        vendor_name=vendor_name,
        flask_session=session if session else {},
        context=context,
    )
    return jsonify(response), status_code

