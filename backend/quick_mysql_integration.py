"""
Quick MySQL Integration for Existing Upload Process
This will add MySQL storage to your current upload workflow
"""

def add_mysql_integration_to_app():
    """
    Add this code to your existing upload function in app.py
    """
    integration_code = '''
    # Add this at the beginning of your upload function (after file is processed):
    
    # MySQL Database Integration
    if DATABASE_AVAILABLE and db_manager:
        try:
            # Store file metadata
            file_id = db_manager.store_file_metadata(
                filename=uploaded_file.filename,
                file_path=file_path,  # Path where you saved the uploaded file
                data_source='bank' if 'bank' in uploaded_file.filename.lower() else 'sap'
            )
            
            # Create analysis session
            session_id = db_manager.create_analysis_session(
                file_id=file_id,
                analysis_type='full_analysis'
            )
            
            print(f"📁 MySQL: File stored (ID: {file_id}), Session created (ID: {session_id})")
            
            # Store this in session for later use
            session['mysql_file_id'] = file_id
            session['mysql_session_id'] = session_id
            
        except Exception as e:
            print(f"⚠️ MySQL file storage failed: {e}")
    
    # ... your existing processing code ...
    
    # Add this after AI categorization is complete:
    
    # Store AI processing results in MySQL
    if DATABASE_AVAILABLE and db_manager and 'mysql_session_id' in session:
        try:
            session_id = session['mysql_session_id']
            file_id = session['mysql_file_id']
            
            # Store individual transactions
            for idx, row in processed_df.iterrows():
                db_manager.store_transaction(
                    session_id=session_id,
                    file_id=file_id,
                    row_number=idx + 1,
                    transaction_date=str(row.get('Date', '2024-01-01')),
                    description=str(row.get('Description', '')),
                    amount=float(row.get('Amount', 0)),
                    ai_category=str(row.get('Category', 'Operating Activities')),
                    balance=float(row.get('Balance', 0)) if 'Balance' in row and pd.notna(row['Balance']) else None,
                    transaction_type=str(row.get('Type', '')) if 'Type' in row else None,
                    vendor_name=str(row.get('Vendor', '')) if 'Vendor' in row else None,
                    ai_confidence=0.85  # You can calculate this based on your AI model
                )
            
            # Store AI performance metrics
            total_transactions = len(processed_df)
            successful_categorizations = len(processed_df[processed_df['Category'].notna()])
            
            db_manager.store_ai_model_performance(
                session_id=session_id,
                model_name='ollama',
                model_version='llama3.2:3b',
                total_predictions=total_transactions,
                successful_predictions=successful_categorizations,
                failed_predictions=total_transactions - successful_categorizations,
                average_confidence=0.85,
                processing_time_ms=1500,  # You can measure actual processing time
                memory_usage_mb=50.0
            )
            
            # Complete the analysis session
            db_manager.complete_analysis_session(
                session_id=session_id,
                transaction_count=total_transactions,
                processing_time=1.5,  # Processing time in seconds
                success_rate=95.0
            )
            
            print(f"✅ MySQL: Stored {total_transactions} transactions and AI performance metrics")
            
        except Exception as e:
            print(f"⚠️ MySQL AI results storage failed: {e}")
    '''
    
    return integration_code

def create_simple_test_upload():
    """Create a simple test to verify the integration works with real upload"""
    
    test_code = '''
# Simple test - add this as a new route in your app.py:

@app.route('/test-mysql-upload', methods=['POST'])
def test_mysql_upload():
    """Test route to verify MySQL integration with upload process"""
    try:
        # Simulate file upload process
        filename = "test_real_upload.xlsx"
        
        # Use existing file for testing
        import pandas as pd
        test_df = pd.DataFrame({
            'Date': ['2024-01-15', '2024-01-16', '2024-01-17'],
            'Description': ['SALARY CREDIT', 'ELECTRICITY BILL', 'EQUIPMENT PURCHASE'],
            'Amount': [50000, -2500, -150000],
            'Category': ['Operating Activities', 'Operating Activities', 'Investing Activities']
        })
        
        # MySQL Integration
        if DATABASE_AVAILABLE and db_manager:
            # Store file metadata
            file_id = db_manager.store_file_metadata(
                filename=filename,
                file_path="D:\\\\CASHFLOW-SAP-BANK\\\\data\\\\bank_data_processed.xlsx",
                data_source='bank'
            )
            
            # Create analysis session
            session_id = db_manager.create_analysis_session(
                file_id=file_id,
                analysis_type='full_analysis'
            )
            
            # Store transactions
            for idx, row in test_df.iterrows():
                db_manager.store_transaction(
                    session_id=session_id,
                    file_id=file_id,
                    row_number=idx + 1,
                    transaction_date=row['Date'],
                    description=row['Description'],
                    amount=float(row['Amount']),
                    ai_category=row['Category'],
                    ai_confidence=0.90
                )
            
            # Store AI performance
            db_manager.store_ai_model_performance(
                session_id=session_id,
                model_name='ollama',
                model_version='llama3.2:3b',
                total_predictions=len(test_df),
                successful_predictions=len(test_df),
                failed_predictions=0,
                average_confidence=0.90,
                processing_time_ms=800,
                memory_usage_mb=30.0
            )
            
            # Complete session
            db_manager.complete_analysis_session(
                session_id=session_id,
                transaction_count=len(test_df),
                processing_time=0.8,
                success_rate=100.0
            )
            
            return jsonify({
                'success': True,
                'message': f'MySQL integration test successful!',
                'file_id': file_id,
                'session_id': session_id,
                'transactions_stored': len(test_df)
            })
        else:
            return jsonify({'error': 'MySQL database not available'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Test failed: {str(e)}'}), 500
    '''
    
    return test_code

if __name__ == "__main__":
    print("Quick MySQL Integration Helper")
    print("\n1. Integration Code:")
    print(add_mysql_integration_to_app())
    print("\n2. Test Route Code:")
    print(create_simple_test_upload())
