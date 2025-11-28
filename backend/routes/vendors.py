"""
Vendor Routes
Routes for vendor analysis and transactions
"""
from flask import Blueprint, request, jsonify, session
from vendor_modules import (
    extract_vendors,
    run_vendor_analysis,
    get_vendor_transactions_view,
)

# Create blueprint
bp = Blueprint('vendors', __name__)

@bp.route('/vendor-analysis', methods=['POST'])
def vendor_analysis():
    """Simple OpenAI-powered vendor analysis - works exactly like categories"""
    # Import here to avoid circular dependencies
    from app_setup import (
        uploaded_bank_df, uploaded_data, db_manager, DATABASE_AVAILABLE,
        state_manager, PERSISTENT_STATE_AVAILABLE
    )
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400
            
        vendor = data.get('vendor', '')
        response, status_code = run_vendor_analysis(
            vendor=vendor,
            uploaded_bank_df=uploaded_bank_df,
            uploaded_data=uploaded_data,
            db_manager=db_manager if DATABASE_AVAILABLE else None,
            session=session,
            database_available=DATABASE_AVAILABLE,
            state_manager=state_manager if PERSISTENT_STATE_AVAILABLE else None,
        )
        return jsonify(response), status_code
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ Vendor analysis error: {e}")
        print(f"❌ Traceback: {error_trace}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Vendor analysis failed'
        }), 500


@bp.route('/view_vendor_transactions/<vendor_name>', methods=['GET'])
def view_vendor_transactions(vendor_name):
    """View transactions for a specific vendor - DIRECT AND SIMPLE"""
    # Import here to avoid circular dependencies
    from app_setup import (
        uploaded_data, uploaded_bank_df, db_manager, DATABASE_AVAILABLE,
        state_manager, PERSISTENT_STATE_AVAILABLE
    )
    
    try:
        response, status_code = get_vendor_transactions_view(
            vendor_name=vendor_name,
            uploaded_data=uploaded_data,
            uploaded_bank_df=uploaded_bank_df,
            db_manager=db_manager,
            session=session,
            database_available=DATABASE_AVAILABLE,
            state_manager=state_manager if PERSISTENT_STATE_AVAILABLE else None,
        )
        return jsonify(response), status_code
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ View vendor transactions error: {e}")
        print(f"❌ Traceback: {error_trace}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to get vendor transactions'
        }), 500


@bp.route('/extract-vendors-for-analysis', methods=['POST'])
def extract_vendors_for_analysis():
    """Extract vendors from bank data for analysis dropdown - OPTIMIZED FOR SPEED"""
    # Import here to avoid circular dependencies
    from app_setup import (
        uploaded_bank_df, uploaded_data, app_openai_integration, db_manager,
        PERSISTENT_STATE_AVAILABLE, state_manager, DATABASE_AVAILABLE,
        get_unified_bank_data, update_uploaded_data
    )
    
    try:
        # Handle case when no data is uploaded
        bank_df = get_unified_bank_data()
        if bank_df is None or bank_df.empty:
            return jsonify({
                'success': False,
                'error': 'No data available. Please upload a file first.',
                'vendors': []
            }), 400
        
        response, status_code, updated_bank_df = extract_vendors(
            bank_df=bank_df,
            uploaded_bank_df=uploaded_bank_df,
            uploaded_data=uploaded_data,
            app_openai_integration=app_openai_integration,
            db_manager=db_manager,
            session=session,
            persistent_state_available=PERSISTENT_STATE_AVAILABLE,
            state_manager=state_manager,
            database_available=DATABASE_AVAILABLE,
        )
        if updated_bank_df is not None:
            update_uploaded_data(updated_bank_df, None)
        return jsonify(response), status_code
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ Vendor extraction error: {e}")
        print(f"❌ Traceback: {error_trace}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to extract vendors'
        }), 500

