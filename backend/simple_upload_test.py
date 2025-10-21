"""
Simple Upload Test - This simulates what happens during real file upload
Run this to test if MySQL integration works with actual data processing
"""

def simulate_real_upload():
    """Simulate the exact process that happens during real file upload"""
    
    try:
        from mysql_database_manager import MySQLDatabaseManager
        import pandas as pd
        import time
        
        print("🧪 Simulating Real File Upload Process...")
        print("=" * 50)
        
        # Initialize database
        db_manager = MySQLDatabaseManager(password="cashflow123")
        
        # Step 1: Simulate file upload
        filename = "real_bank_statement.xlsx"
        file_path = "D:\\CASHFLOW-SAP-BANK\\data\\bank_data_processed.xlsx"
        
        print(f"📁 Processing file: {filename}")
        
        # Step 2: Load and process data (like your real upload does)
        df = pd.read_excel(file_path)
        print(f"📊 Loaded {len(df)} transactions from file")
        
        # Step 3: Store file metadata
        file_id = db_manager.store_file_metadata(
            filename=filename,
            file_path=file_path,
            data_source='bank'
        )
        print(f"✅ File metadata stored (ID: {file_id})")
        
        # Step 4: Create analysis session
        session_id = db_manager.create_analysis_session(
            file_id=file_id,
            analysis_type='full_analysis'
        )
        print(f"✅ Analysis session created (ID: {session_id})")
        
        # Step 5: Process transactions (simulate AI categorization)
        start_time = time.time()
        successful_categorizations = 0
        
        print(f"🤖 Processing {len(df)} transactions with AI...")
        
        for idx, row in df.iterrows():
            # Simulate AI categorization (like your real system does)
            description = str(row.get('Description', ''))
            amount = float(row.get('Amount', 0))
            
            # Simple categorization logic (replace with your actual AI logic)
            if 'salary' in description.lower() or 'credit' in description.lower():
                category = 'Operating Activities'
                confidence = 0.95
            elif 'equipment' in description.lower() or 'machinery' in description.lower():
                category = 'Investing Activities' 
                confidence = 0.90
            elif 'loan' in description.lower() or 'interest' in description.lower():
                category = 'Financing Activities'
                confidence = 0.88
            else:
                category = 'Operating Activities'
                confidence = 0.80
            
            # Store transaction in MySQL
            db_manager.store_transaction(
                session_id=session_id,
                file_id=file_id,
                row_number=idx + 1,
                transaction_date=str(row.get('Date', '2024-01-01')),
                description=description,
                amount=amount,
                ai_category=category,
                balance=float(row.get('Balance', 0)) if 'Balance' in row and pd.notna(row['Balance']) else None,
                ai_confidence=confidence
            )
            
            if confidence > 0.8:
                successful_categorizations += 1
            
            # Show progress every 10 transactions
            if (idx + 1) % 10 == 0:
                print(f"   Processed {idx + 1}/{len(df)} transactions...")
        
        processing_time = time.time() - start_time
        
        # Step 6: Store AI performance metrics
        performance_id = db_manager.store_ai_model_performance(
            session_id=session_id,
            model_name='ollama',
            model_version='llama3.2:3b',
            total_predictions=len(df),
            successful_predictions=successful_categorizations,
            failed_predictions=len(df) - successful_categorizations,
            average_confidence=0.87,
            processing_time_ms=processing_time * 1000,
            memory_usage_mb=60.0
        )
        print(f"✅ AI performance stored (ID: {performance_id})")
        
        # Step 7: Complete analysis session
        db_manager.complete_analysis_session(
            session_id=session_id,
            transaction_count=len(df),
            processing_time=processing_time,
            success_rate=(successful_categorizations / len(df)) * 100
        )
        
        print("🎉 REAL UPLOAD SIMULATION COMPLETE!")
        print("=" * 50)
        print(f"📊 Results:")
        print(f"   • File ID: {file_id}")
        print(f"   • Session ID: {session_id}")
        print(f"   • Transactions: {len(df)}")
        print(f"   • AI Success Rate: {(successful_categorizations/len(df)*100):.1f}%")
        print(f"   • Processing Time: {processing_time:.2f}s")
        print(f"   • Performance ID: {performance_id}")
        
        print("\n🔍 Check MySQL with these commands:")
        print(f"   SELECT * FROM files WHERE file_id = {file_id};")
        print(f"   SELECT * FROM analysis_sessions WHERE session_id = {session_id};")
        print(f"   SELECT COUNT(*) FROM transactions WHERE session_id = {session_id};")
        print(f"   SELECT * FROM ai_model_performance WHERE session_id = {session_id};")
        
        return True
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        return False

if __name__ == "__main__":
    simulate_real_upload()
