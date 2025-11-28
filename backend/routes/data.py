"""
Data Routes
Routes for getting current data and dropdown data
"""
from flask import Blueprint, jsonify, session, request
from vendor_modules import get_vendor_dropdown_data
import pandas as pd

# Create blueprint
bp = Blueprint('data', __name__)

@bp.route('/get-current-data', methods=['GET'])
def get_current_data():
    """Get current restored data for UI population"""
    # Import here to avoid circular dependencies
    from app_setup import uploaded_bank_df, bank_count, ai_categorized, reconciliation_data
    
    try:
        if uploaded_bank_df is not None and not uploaded_bank_df.empty:
            # Use the formatter to ensure Inward_Amount and Outward_Amount are properly extracted
            from upload_modules.response_formatter import format_transactions_for_frontend
            bank_data_clean = format_transactions_for_frontend(uploaded_bank_df)
            data = {
                'success': True,
                'bank_data': bank_data_clean,
                'bank_columns': list(uploaded_bank_df.columns),
                'transaction_count': len(uploaded_bank_df),
                'bank_count': bank_count if 'bank_count' in globals() else 0,
                'ai_categorized': ai_categorized if 'ai_categorized' in globals() else 0
            }
            
            # Add categories if available
            if 'Category' in uploaded_bank_df.columns:
                categories = uploaded_bank_df['Category'].fillna('Uncategorized').value_counts().to_dict()
                data['categories'] = categories
                
            # Add vendors if available
            if 'Vendor' in uploaded_bank_df.columns:
                vendors = uploaded_bank_df['Vendor'].dropna().unique().tolist()
                vendors_clean = [str(v).strip() for v in vendors if str(v).strip() and str(v) != 'nan']
                data['vendors'] = vendors_clean
            
            # Add reconciliation status
            if reconciliation_data:
                data['has_reconciliation_data'] = True
            
            return jsonify(data)
        else:
            return jsonify({
                'success': False,
                'message': 'No current data available'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error getting current data: {str(e)}'
        }), 500


@bp.route('/get-dropdown-data', methods=['GET', 'OPTIONS'])
def get_dropdown_data():
    """Get real data to populate dropdowns"""
    # Handle OPTIONS preflight request
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response, 200
    
    # Import here to avoid circular dependencies
    from flask import session
    from app_setup import (
        uploaded_bank_df, uploaded_data, DATA_FOLDER, db_manager,
        DATABASE_AVAILABLE, state_manager, PERSISTENT_STATE_AVAILABLE
    )
    
    try:
        # Handle case where no data is uploaded yet
        if uploaded_bank_df is None or uploaded_bank_df.empty:
            response = jsonify({
                'success': True,
                'vendors': [],
                'message': 'No data uploaded yet'
            })
            response.headers['Access-Control-Allow-Origin'] = '*'
            return response, 200
        
        try:
            response, status_code = get_vendor_dropdown_data(
                uploaded_bank_df=uploaded_bank_df,
                uploaded_data=uploaded_data,
                data_folder=DATA_FOLDER,
                db_manager=db_manager,
                session=session,
                database_available=DATABASE_AVAILABLE,
                state_manager=state_manager if PERSISTENT_STATE_AVAILABLE else None,
            )
            json_response = jsonify(response)
            json_response.headers['Access-Control-Allow-Origin'] = '*'
            return json_response, status_code
        except Exception as db_error:
            # If database lookup fails, try to get vendors from DataFrame directly
            print(f"⚠️ Database lookup failed, falling back to DataFrame: {db_error}")
            import traceback
            traceback.print_exc()
            
            # Fallback: extract vendors from DataFrame
            vendors = ['All']  # Always provide at least "All" option
            if 'Assigned_Vendor' in uploaded_bank_df.columns:
                unique_vendors = uploaded_bank_df['Assigned_Vendor'].dropna().unique().tolist()
                unique_vendors = [v for v in unique_vendors if v and v.strip()]
                if unique_vendors:
                    vendors = ['All'] + sorted(unique_vendors)
            
            transaction_types = []
            if 'Category' in uploaded_bank_df.columns:
                transaction_types = uploaded_bank_df['Category'].dropna().unique().tolist()
            if not transaction_types:
                transaction_types = ['Operating Activities', 'Investing Activities', 'Financing Activities']
            
            print(f"📊 Fallback: Returning {len(vendors)} vendors from DataFrame")
            
            response = jsonify({
                'success': True,
                'vendors': vendors if vendors else ['All'],
                'vendor_stats': [],
                'transaction_types': transaction_types if transaction_types else [],
                'total_transactions': len(uploaded_bank_df) if uploaded_bank_df is not None else 0,
                'message': 'Using data from uploaded file (database unavailable)'
            })
            response.headers['Access-Control-Allow-Origin'] = '*'
            return response, 200
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ Get dropdown data error: {e}")
        print(f"❌ Traceback: {error_trace}")
        response = jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to get dropdown data'
        })
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response, 500


@bp.route('/update-transaction-category', methods=['POST', 'OPTIONS'])
def update_transaction_category():
    """Update transaction category when user provides clarification"""
    # Handle OPTIONS preflight request
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response, 200
    
    try:
        from flask import session
        from app_setup import db_manager, DATABASE_AVAILABLE, uploaded_bank_df
        
        data = request.get_json()
        transaction_id = data.get('transaction_id')
        row_number = data.get('row_number')
        description = data.get('description')
        new_category = data.get('category')
        
        if not new_category:
            return jsonify({
                'success': False,
                'error': 'Category is required'
            }), 400
        
        # Validate category
        valid_categories = ["Operating Activities", "Investing Activities", "Financing Activities"]
        if new_category not in valid_categories:
            return jsonify({
                'success': False,
                'error': f'Invalid category. Must be one of: {", ".join(valid_categories)}'
            }), 400
        
        # Get session_id from session or database
        session_id = None
        if session and 'mysql_session_id' in session:
            session_id = session['mysql_session_id']
            print(f"📊 Using session_id from session: {session_id}")
        
        if not session_id and DATABASE_AVAILABLE and db_manager:
            # Try to find session_id from current data or most recent session
            try:
                conn = db_manager.get_connection()
                cursor = conn.cursor()
                # Get most recent session_id from database
                cursor.execute("""
                    SELECT session_id FROM analysis_sessions 
                    WHERE status = 'completed'
                    ORDER BY created_at DESC LIMIT 1
                """)
                result = cursor.fetchone()
                if result:
                    session_id = result[0]
                    print(f"📊 Using session_id from database: {session_id}")
                    # Also try to get from file_id if we have row_number
                    if row_number:
                        cursor.execute("""
                            SELECT session_id FROM transactions 
                            WHERE original_row_number = %s
                            ORDER BY transaction_id DESC LIMIT 1
                        """, (row_number,))
                        txn_result = cursor.fetchone()
                        if txn_result:
                            session_id = txn_result[0]
                            print(f"📊 Using session_id from transaction row_number: {session_id}")
            except Exception as e:
                print(f"⚠️ Error getting session_id: {e}")
                pass
        
        # If still no session_id but we have description, try to find it
        if not session_id and description and DATABASE_AVAILABLE and db_manager:
            try:
                conn = db_manager.get_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT session_id FROM transactions 
                    WHERE description = %s
                    ORDER BY transaction_id DESC LIMIT 1
                """, (description,))
                result = cursor.fetchone()
                if result:
                    session_id = result[0]
                    print(f"📊 Using session_id from transaction description: {session_id}")
            except Exception as e:
                print(f"⚠️ Error getting session_id from description: {e}")
                pass
        
        print(f"📊 Final session_id: {session_id}, row_number: {row_number}, description: {description}")
        
        # If we have row_number or description but no session_id, try to find the transaction first
        if not session_id and (row_number or description) and DATABASE_AVAILABLE and db_manager:
            try:
                conn = db_manager.get_connection()
                cursor = conn.cursor(dictionary=True)
                
                if row_number:
                    cursor.execute("""
                        SELECT session_id, transaction_id FROM transactions 
                        WHERE original_row_number = %s
                        ORDER BY transaction_id DESC LIMIT 1
                    """, (row_number,))
                elif description:
                    cursor.execute("""
                        SELECT session_id, transaction_id FROM transactions 
                        WHERE description = %s
                        ORDER BY transaction_id DESC LIMIT 1
                    """, (description,))
                
                result = cursor.fetchone()
                if result:
                    session_id = result['session_id']
                    if not transaction_id:
                        transaction_id = result['transaction_id']
                    print(f"📊 Found transaction - session_id: {session_id}, transaction_id: {transaction_id}")
            except Exception as e:
                print(f"⚠️ Error finding transaction: {e}")
                import traceback
                traceback.print_exc()
        
        if not DATABASE_AVAILABLE or not db_manager:
            # Update in-memory DataFrame if database not available
            if uploaded_bank_df is not None and not uploaded_bank_df.empty:
                if row_number is not None:
                    idx = int(row_number) - 1  # Convert to 0-based index
                    if 0 <= idx < len(uploaded_bank_df):
                        uploaded_bank_df.at[idx, 'Category'] = new_category
                        print(f"✅ Updated transaction {row_number} category to {new_category} (in-memory)")
                        return jsonify({
                            'success': True,
                            'message': 'Category updated successfully (in-memory)'
                        })
                
                return jsonify({
                    'success': False,
                    'error': 'Could not update transaction (database not available and row_number not provided)'
                }), 400
        
        # Validate we have enough info to identify the transaction
        if not transaction_id and not (session_id and (row_number or description)):
            error_msg = f"Unable to identify transaction. Need either transaction_id OR (session_id + row_number/description). Got: session_id={session_id}, row_number={row_number}, description={description[:50] if description else None}"
            print(f"❌ {error_msg}")
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400
        
        # Update in database
        print(f"📊 Updating transaction with: transaction_id={transaction_id}, session_id={session_id}, row_number={row_number}")
        success = db_manager.update_transaction_category(
            new_category=new_category,
            transaction_id=transaction_id,
            session_id=session_id,
            row_number=row_number,
            description=description
        )
        
        if not success:
            return jsonify({
                'success': False,
                'error': 'Failed to update transaction in database'
            }), 500
        
        # Also update in-memory DataFrame if available
        if uploaded_bank_df is not None and not uploaded_bank_df.empty:
            if row_number is not None:
                idx = int(row_number) - 1
                if 0 <= idx < len(uploaded_bank_df):
                    uploaded_bank_df.at[idx, 'Category'] = new_category
                    print(f"✅ Updated in-memory DataFrame: row {row_number} category → {new_category}")
            elif description:
                # Find by description (less reliable but fallback)
                mask = uploaded_bank_df['Description'] == description
                if mask.any():
                    uploaded_bank_df.loc[mask, 'Category'] = new_category
                    print(f"✅ Updated in-memory DataFrame: description '{description[:50]}' category → {new_category}")
        
        print(f"📊 Reports will be automatically recalculated with updated category on next access")
        print(f"📊 Category totals will reflect: {new_category} (transaction moved from previous category)")
        
        response = jsonify({
            'success': True,
            'message': 'Category updated successfully. Reports will recalculate automatically.',
            'updated_category': new_category
        })
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response, 200
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ Update transaction category error: {e}")
        print(f"❌ Traceback: {error_trace}")
        response = jsonify({
            'success': False,
            'error': str(e)
        })
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response, 500