"""
Upload Orchestrator Module
==========================
Main orchestrator that coordinates all upload processing steps.
"""

import time
import os
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from werkzeug.datastructures import FileStorage

from .file_validator import validate_uploaded_file, save_uploaded_file
from .file_loader import load_file
from .data_preprocessor import preprocess_dataframe
from .ai_categorizer import categorize_transactions
from .database_storage import store_upload_results
from .response_formatter import (
    format_transactions_for_frontend,
    generate_ai_reasoning_explanations,
    format_upload_response
)


def _precompute_transaction_reasoning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate AI reasoning for each transaction so the frontend Details modal
    can display insights immediately after upload.
    """
    if df is None or df.empty:
        return df
    
    # If reasoning already exists (cache hit), keep it
    if 'AI_Reasoning' in df.columns and df['AI_Reasoning'].notna().any():
        print("🧠 Using existing AI reasoning from dataset")
        return df
    
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from ai_reasoning_engine import (
            ai_reasoning_engine,
            get_categorization_reasoning
        )
    except Exception as import_error:
        print(f"⚠️ AI Reasoning Engine unavailable ({import_error}). Skipping reasoning generation.")
        return df
    
    if not getattr(ai_reasoning_engine, 'is_available', False):
        print("⚠️ AI Reasoning Engine disabled (no API key). Skipping reasoning generation.")
        return df
    
    print(f"🧠 Precomputing AI reasoning for {len(df)} transactions...")
    reasoning_results = []
    for idx, row in df.iterrows():
        description = str(row.get('Description', '') or '').strip()
        category = str(row.get('Category', '') or '').strip() or "Operating Activities"
        
        if not description:
            reasoning_results.append(None)
            continue
        
        try:
            if (idx + 1) % 10 == 0 or idx in (0, len(df) - 1):
                print(f"   🤖 Reasoning {idx + 1}/{len(df)} for '{description[:40]}...'")
            
            reasoning = get_categorization_reasoning(
                transaction_desc=description,
                category=category,
                all_transactions=None
            )
            reasoning_results.append(reasoning)
        except Exception as reasoning_error:
            print(f"   ⚠️ Failed to generate reasoning for row {idx + 1}: {reasoning_error}")
            reasoning_results.append(None)
    
    df['AI_Reasoning'] = reasoning_results
    generated = sum(1 for r in reasoning_results if r)
    print(f"✅ Precomputed AI reasoning for {generated} transactions")
    return df


def process_upload(
    bank_file: FileStorage,
    db_manager=None,
    session=None,
    data_adapter_available: bool = False,
    ml_available: bool = False,
    reconciliation_data: dict = None,
    state_manager=None,
) -> Tuple[Dict[str, Any], int]:
    """
    Main upload processing function - orchestrates all steps.
    
    Args:
        bank_file: Uploaded file from request
        db_manager: MySQLDatabaseManager instance (optional)
        session: Flask session object (optional)
        data_adapter_available: Whether Universal Data Adapter is available
        ml_available: Whether ML is available
        reconciliation_data: Reconciliation data (optional)
        
    Returns:
        Tuple of (response_dict, status_code)
    """
    start_time = time.time()
    
    # Step 1: Validate file
    is_valid, error_message = validate_uploaded_file(bank_file)
    if not is_valid:
        return {'error': error_message}, 400
    
    try:
        print("⚡ ML/AI UPLOAD: Processing files with 100% AI/ML approach...")
        print(f"🔍 ML System Status: {'Available' if ml_available else 'Not Available'}")
        
        # Step 2: Save file temporarily
        temp_file_path = save_uploaded_file(bank_file, file_type='bank')
        
        # Step 3: Check database cache
        using_cached_data = False
        uploaded_bank_df = None
        cache_metadata = None  # ✅ Initialize cache_metadata to make it available in later code
        
        if db_manager:
            try:
                import sys
                import os
                # Add parent directory to path for imports
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from database_cache_manager import check_and_retrieve_cached_file
                cached_result = check_and_retrieve_cached_file(temp_file_path, db_manager)
                
                if cached_result:
                    cached_df, cache_metadata = cached_result
                    print(f"🎯 CACHE HIT: File already processed! Using cached results from database.")
                    print(f"   📊 File ID: {cache_metadata['file_id']}, Session ID: {cache_metadata['session_id']}")
                    print(f"   📈 Transactions: {cache_metadata['transaction_count']}")
                    print(f"   ⏰ Completed: {cache_metadata['completed_at']}")
                    print(f"   💰 Saved: Skipping expensive AI/ML processing!")
                    
                    uploaded_bank_df = cached_df
                    using_cached_data = True
                    
                    # ✅ ALWAYS update session IDs when cache is found (even if they already exist)
                    # This ensures we're using the correct session_id for the current file
                    if session:
                        session['mysql_file_id'] = cache_metadata['file_id']
                        session['mysql_session_id'] = cache_metadata['session_id']
                        print(f"✅ Updated Flask session: file_id={cache_metadata['file_id']}, session_id={cache_metadata['session_id']}")
                    
                    if state_manager:
                        state_manager.set_current_session(
                            cache_metadata['session_id'],
                            cache_metadata['file_id']
                        )
                    
                    print(f"✅ Using cached data - skipping AI processing")
            except Exception as cache_error:
                print(f"⚠️ Cache check failed: {cache_error}. Continuing with normal processing...")
        
        # Step 4: Load file (if not cached)
        if not using_cached_data:
            print(f"📄 CACHE MISS: New file or no completed session found. Processing with AI...")
            uploaded_bank_df = load_file(temp_file_path, bank_file, data_adapter_available)
        
        print(f"📊 File loaded: {len(uploaded_bank_df)} rows, {len(uploaded_bank_df.columns)} columns")
        
        # Step 5: Preprocess data (ONLY if not cached - cached data is already preprocessed)
        if not using_cached_data:
            preprocess_result = preprocess_dataframe(uploaded_bank_df)
            uploaded_bank_df = preprocess_result['dataframe']
        else:
            # ✅ Cached data is already preprocessed - skip preprocessing to preserve correct balances
            print(f"✅ SKIPPING preprocess (cached data already has correct balances from database)")
        
        # Step 6: AI Categorization
        uploaded_bank_df, ai_categorized = categorize_transactions(
            uploaded_bank_df, 
            use_cache=using_cached_data
        )
        
        # Step 7: Precompute AI reasoning (always attempt; function skips if already present)
        uploaded_bank_df = _precompute_transaction_reasoning(uploaded_bank_df)

        # Track original row numbers for downstream persistence (e.g., vendor linking)
        uploaded_bank_df = uploaded_bank_df.reset_index(drop=True)
        uploaded_bank_df['Original_Row_Number'] = uploaded_bank_df.index + 1
        
        bank_count = len(uploaded_bank_df)
        
        # Step 8: Store in database (only if not using cached data)
        db_metadata = None
        if not using_cached_data and db_manager:
            db_metadata = store_upload_results(
                df=uploaded_bank_df,
                bank_file_filename=bank_file.filename,
                temp_file_path=temp_file_path,
                db_manager=db_manager,
                session=session,
                ai_categorized=ai_categorized,
                start_time=start_time,
                reconciliation_data=reconciliation_data
            )
            if state_manager and db_metadata:
                state_manager.set_current_session(
                    db_metadata['session_id'],
                    db_metadata['file_id']
                )
        elif using_cached_data and cache_metadata:
            # ✅ Use cache_metadata directly (already set above) instead of potentially stale session values
            # This ensures we use the correct session_id from the cache, not old session values
            db_metadata = {
                'file_id': cache_metadata['file_id'],
                'session_id': cache_metadata['session_id'],
                'transaction_count': cache_metadata.get('transaction_count', bank_count),
                'processing_time': 0  # Cached, so no processing time
            }
            if state_manager:
                state_manager.set_current_session(
                    cache_metadata['session_id'],
                    cache_metadata['file_id']
                )
            print(f"✅ Using cached session metadata: file_id={cache_metadata['file_id']}, session_id={cache_metadata['session_id']}")
        
        # Step 9: Save processed file (backup)
        try:
            DATA_FOLDER = 'data'
            os.makedirs(DATA_FOLDER, exist_ok=True)
            uploaded_bank_df.to_excel(
                os.path.join(DATA_FOLDER, 'bank_data_processed.xlsx'), 
                index=False
            )
        except Exception as e:
            print(f"⚠️ Failed to save backup file: {e}")
        
        # Step 10: Generate comprehensive cash flow report
        try:
            from .cashflow_report_generator import generate_comprehensive_cashflow_report
            cashflow_report = generate_comprehensive_cashflow_report(uploaded_bank_df)
            print(f"✅ Generated comprehensive cash flow report with {len(cashflow_report.get('transactions', []))} transactions")
        except Exception as e:
            print(f"⚠️ Failed to generate comprehensive cash flow report: {e}")
            cashflow_report = None
        
        # Step 11: Format response
        processing_time = time.time() - start_time
        
        # Format transactions for frontend
        transactions_data = format_transactions_for_frontend(uploaded_bank_df)
        
        # Calculate ML statistics
        ml_count = sum(1 for t in transactions_data if t.get('category', '').strip())
        ml_percentage = (ml_count / len(transactions_data) * 100) if transactions_data else 0
        
        # Generate AI reasoning explanations
        reasoning_explanations = generate_ai_reasoning_explanations(
            transactions_data,
            bank_count,
            ml_percentage
        )
        
        # Format complete response
        response = format_upload_response(
            transactions_data=transactions_data,
            bank_count=bank_count,
            processing_time=processing_time,
            ml_available=ml_available,
            ml_percentage=ml_percentage,
            mode="bank_only_analysis",
            reasoning_explanations=reasoning_explanations
        )
        
        # Add comprehensive cash flow report to response
        if cashflow_report:
            response['cashflow_report'] = cashflow_report
        
        # Add processing summary
        print(f"\n🚀 PRODUCTION MODE PROCESSING SUMMARY:")
        print(f"   📊 Processing Method: 100% AI/ML Approach (PRODUCTION MODE)")
        print(f"   📈 Bank Transactions: {bank_count} processed (All transactions)")
        print(f"   ⏱️ Processing Time: {processing_time:.2f} seconds")
        print(f"   🦙 OpenAI Integration: Active")
        print(f"   {'🎯 CACHE USED: No AI processing needed!' if using_cached_data else '🤖 AI Processing: Completed'}")
        
        return response, 200
        
    except Exception as e:
        print(f"❌ Upload processing error: {e}")
        import traceback
        traceback.print_exc()
        return {'error': f'Upload processing failed: {str(e)}'}, 500

