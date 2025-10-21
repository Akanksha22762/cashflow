"""
Database Storage Integration for Upload Process
This script shows how to integrate MySQL database storage into your upload workflow.
"""

def integrate_database_storage_into_upload(uploaded_file_path, filename, data_source='bank'):
    """
    Integration function to add to your upload process
    Call this function when a file is uploaded and processed
    """
    try:
        from mysql_database_manager import MySQLDatabaseManager
        
        # Initialize database manager
        db_manager = MySQLDatabaseManager(password="cashflow123")
        
        print(f"📁 Storing file metadata for: {filename}")
        
        # Step 1: Store file metadata
        file_id = db_manager.store_file_metadata(
            filename=filename,
            file_path=uploaded_file_path,
            data_source=data_source
        )
        
        print(f"✅ File stored with ID: {file_id}")
        
        # Step 2: Create analysis session
        session_id = db_manager.create_analysis_session(
            file_id=file_id,
            analysis_type='full_analysis'
        )
        
        print(f"🔄 Analysis session created with ID: {session_id}")
        
        return file_id, session_id, db_manager
        
    except Exception as e:
        print(f"❌ Database storage integration failed: {e}")
        return None, None, None

def store_ai_processing_results(db_manager, session_id, file_id, transactions_data, ai_performance_data):
    """
    Store AI processing results including transactions and performance metrics
    """
    try:
        if not db_manager or not session_id:
            print("⚠️ Database manager or session not available")
            return
        
        print(f"💾 Storing {len(transactions_data)} transactions...")
        
        # Store individual transactions
        for idx, transaction in enumerate(transactions_data):
            transaction_id = db_manager.store_transaction(
                session_id=session_id,
                file_id=file_id,
                row_number=idx + 1,
                transaction_date=transaction.get('date', '2024-01-01'),
                description=transaction.get('description', ''),
                amount=float(transaction.get('amount', 0)),
                ai_category=transaction.get('category', 'Operating Activities'),
                balance=transaction.get('balance'),
                transaction_type=transaction.get('type'),
                vendor_name=transaction.get('vendor'),
                ai_confidence=transaction.get('confidence')
            )
        
        # Store AI model performance
        if ai_performance_data:
            performance_id = db_manager.store_ai_model_performance(
                session_id=session_id,
                model_name=ai_performance_data.get('model_name', 'ollama'),
                model_version=ai_performance_data.get('model_version', 'llama3.2:3b'),
                total_predictions=ai_performance_data.get('total_predictions', len(transactions_data)),
                successful_predictions=ai_performance_data.get('successful_predictions', len(transactions_data)),
                failed_predictions=ai_performance_data.get('failed_predictions', 0),
                average_confidence=ai_performance_data.get('average_confidence', 0.85),
                processing_time_ms=ai_performance_data.get('processing_time_ms', 1000),
                memory_usage_mb=ai_performance_data.get('memory_usage_mb', 50)
            )
            print(f"📊 AI performance stored with ID: {performance_id}")
        
        # Complete the analysis session
        processing_time_seconds = (ai_performance_data.get('processing_time_ms', 1000) / 1000.0) if ai_performance_data else 1.0
        db_manager.complete_analysis_session(
            session_id=session_id,
            transaction_count=len(transactions_data),
            processing_time=processing_time_seconds,
            success_rate=95.0  # You can calculate this based on actual AI performance
        )
        
        print(f"✅ Successfully stored all data for session {session_id}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to store AI processing results: {e}")
        return False

# Example of how to integrate this into your upload function:
"""
def your_upload_function():
    # ... your existing upload code ...
    
    # When file is uploaded:
    file_id, session_id, db_manager = integrate_database_storage_into_upload(
        uploaded_file_path=file_path,
        filename=filename,
        data_source='bank'  # or 'sap'
    )
    
    # ... your AI processing code ...
    # After AI processing is complete:
    
    # Prepare transaction data
    transactions_data = [
        {
            'date': '2024-01-01',
            'description': 'Sample transaction',
            'amount': 1000.0,
            'category': 'Operating Activities',
            'confidence': 0.85
        }
        # ... more transactions
    ]
    
    # Prepare AI performance data
    ai_performance_data = {
        'model_name': 'ollama',
        'model_version': 'llama3.2:3b',
        'total_predictions': len(transactions_data),
        'successful_predictions': len(transactions_data),
        'average_confidence': 0.85,
        'processing_time_ms': 1500
    }
    
    # Store everything
    store_ai_processing_results(
        db_manager, session_id, file_id, 
        transactions_data, ai_performance_data
    )
"""

if __name__ == "__main__":
    print("Database Storage Integration Helper")
    print("Use the functions above to integrate MySQL storage into your upload process")
