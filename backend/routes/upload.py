"""
Upload Routes
File upload endpoint
"""
from flask import Blueprint, request, jsonify, session
from upload_modules import process_upload

# Create blueprint
bp = Blueprint('upload', __name__)

@bp.route('/upload', methods=['POST'])
def upload_files_with_ml_ai():
    """
    Simplified upload endpoint using modular upload system.
    All upload processing is now handled by upload_modules package.
    """
    # Import here to avoid circular dependencies
    from app_setup import (
        reconciliation_data, db_manager, DATABASE_AVAILABLE,
        DATA_ADAPTER_AVAILABLE, ML_AVAILABLE, state_manager, 
        PERSISTENT_STATE_AVAILABLE, uploaded_bank_df, uploaded_data
    )
    
    bank_file = request.files.get('bank_file')
    
    # Use modular upload orchestrator
    try:
        # Process upload using modular orchestrator
        response_data, status_code = process_upload(
            bank_file=bank_file,
            db_manager=db_manager if DATABASE_AVAILABLE else None,
            session=session,
            data_adapter_available=DATA_ADAPTER_AVAILABLE,
            ml_available=ML_AVAILABLE,
            reconciliation_data=reconciliation_data,
            state_manager=state_manager if PERSISTENT_STATE_AVAILABLE else None,
        )
        
        # Update global variables for compatibility with other endpoints
        if 'transactions' in response_data and len(response_data['transactions']) > 0:
            # Import pandas for DataFrame creation
            import pandas as pd
            # Update globals via app_setup
            from app_setup import update_uploaded_data
            transactions = response_data['transactions']
            if transactions:
                bank_df = pd.DataFrame(transactions)
                update_uploaded_data(bank_df, None)
        
        return jsonify(response_data), status_code

    except Exception as e:
        import traceback
        print(f"❌ Upload error: {str(e)}")
        print("Traceback:\n", traceback.format_exc())
        return jsonify({'error': f'Upload processing failed: {str(e)}'}), 500

