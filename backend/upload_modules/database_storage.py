"""
Database Storage Module
=======================
Handles storing upload results in MySQL database.
"""

import pandas as pd
import time
from typing import Dict, Any, Optional, Tuple
from datetime import datetime


def store_upload_results(
    df: pd.DataFrame,
    bank_file_filename: str,
    temp_file_path: str,
    db_manager,
    session,
    ai_categorized: int,
    start_time: float,
    reconciliation_data: dict = None
) -> Optional[Dict[str, Any]]:
    """
    Store upload results in MySQL database.
    
    Args:
        df: Processed DataFrame with transactions
        bank_file_filename: Original filename
        temp_file_path: Path to saved file
        db_manager: MySQLDatabaseManager instance
        session: Flask session object
        ai_categorized: Number of AI-categorized transactions
        start_time: Processing start time
        reconciliation_data: Reconciliation data (optional)
        
    Returns:
        Dictionary with file_id, session_id, and metadata, or None if failed
    """
    if not db_manager:
        print("⚠️ MySQL database not available - results not stored permanently")
        return None
    
    try:
        print("💾 Storing results in MySQL database...")
        
        # Store file metadata
        file_id = db_manager.store_file_metadata(
            filename=bank_file_filename,
            file_path=temp_file_path,
            data_source='bank'
        )
        
        # ✅ Check if there's already an incomplete session for this file
        # Reuse incomplete sessions to avoid duplicates, but don't reuse completed ones
        # (Completed sessions should have been caught by cache check)
        session_id = None
        try:
            conn = db_manager.get_connection()
            cursor = conn.cursor()
            # Look for incomplete sessions first (processing, failed, etc.)
            cursor.execute("""
                SELECT session_id 
                FROM analysis_sessions
                WHERE file_id = %s 
                  AND status != 'completed'
                ORDER BY created_at DESC
                LIMIT 1
            """, (file_id,))
            existing_session = cursor.fetchone()
            if existing_session:
                session_id = existing_session[0]
                print(f"✅ Reusing existing incomplete session {session_id} for file_id {file_id}")
                # Reset session status to 'processing'
                cursor.execute("""
                    UPDATE analysis_sessions 
                    SET status = 'processing',
                        created_at = CURRENT_TIMESTAMP
                    WHERE session_id = %s
                """, (session_id,))
                conn.commit()
        except Exception as session_check_error:
            print(f"⚠️ Error checking for existing session: {session_check_error}")
        
        # Create new analysis session only if no incomplete one found
        if not session_id:
            session_id = db_manager.create_analysis_session(
                file_id=file_id,
                analysis_type='full_analysis'
            )
            print(f"✅ Created new analysis session {session_id} for file_id {file_id}")
        
        # Generate AI reasoning for each transaction and store
        print("🧠 Generating AI reasoning for all transactions...")
        try:
            import sys
            import os
            # Add parent directory to path for imports
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from ai_reasoning_engine import get_categorization_reasoning
            AI_REASONING_AVAILABLE = True
        except:
            AI_REASONING_AVAILABLE = False
            print("⚠️ AI Reasoning Engine not available - skipping reasoning generation")
        
        # Store each transaction with AI reasoning
        total_transactions = len(df)
        ai_reasonings = []
        
        # Removed verbose debug messages - only log if there's an actual issue
        
        for idx, row in df.iterrows():
            # Generate AI reasoning for this transaction
            ai_reasoning = None
            precomputed_reasoning = None
            if 'AI_Reasoning' in df.columns:
                precomputed_reasoning = row.get('AI_Reasoning') if 'AI_Reasoning' in row.index else None
            
            if precomputed_reasoning and pd.notna(precomputed_reasoning):
                ai_reasoning = precomputed_reasoning
            elif AI_REASONING_AVAILABLE:
                try:
                    description = str(row.get('Description', '')) if 'Description' in row.index and pd.notna(row.get('Description')) else ''
                    category = str(row.get('Category', '')) if 'Category' in row.index and pd.notna(row.get('Category')) else ''
                    
                    # Generate reasoning (with progress indicator every 10 transactions)
                    if (idx + 1) % 10 == 0 or idx == 0:
                        print(f"   🧠 Generating reasoning for transaction {idx + 1}/{total_transactions}...")
                    
                    reasoning_result = get_categorization_reasoning(
                        transaction_desc=description,
                        category=category,
                        all_transactions=None  # Can pass sample transactions for context if needed
                    )
                    ai_reasoning = reasoning_result
                except Exception as e:
                    print(f"   ⚠️ Failed to generate reasoning for transaction {idx + 1}: {e}")
                    ai_reasoning = None
            
            # Helper function to safely get column value with fallback column names
            def safe_get(column_name, default=None, convert_type=str, fallback_names=None):
                # Try primary column name first
                if column_name in row.index:
                    value = row[column_name]
                    if pd.notna(value) and value is not None:
                        try:
                            if convert_type == str:
                                return str(value)
                            elif convert_type == float:
                                return float(value)
                            elif convert_type == int:
                                return int(value)
                            return value
                        except (ValueError, TypeError):
                            pass
                
                # Try fallback column names if provided
                if fallback_names:
                    for fallback in fallback_names:
                        if fallback in row.index:
                            value = row[fallback]
                            if pd.notna(value) and value is not None:
                                try:
                                    if convert_type == str:
                                        return str(value)
                                    elif convert_type == float:
                                        return float(value)
                                    elif convert_type == int:
                                        return int(value)
                                    return value
                                except (ValueError, TypeError):
                                    pass
                
                return default
            
            # Store transaction with reasoning - properly handle column access and NaN values
            # Use fallback column names in case Universal Data Adapter uses different names
            transaction_date = safe_get('Date', '', str, ['_date', 'Transaction_Date', 'transaction_date'])
            description = safe_get('Description', '', str, ['description', 'Transaction_Description'])
            amount = safe_get('Amount', 0.0, float, ['_amount', 'amount', 'Transaction_Amount'])
            
            # Extract Inward_Amount and Outward_Amount directly from DataFrame
            inward_amount = safe_get('Inward_Amount', None, float, ['inward_amount', 'Inward Amount', 'Credit Amount'])
            outward_amount = safe_get('Outward_Amount', None, float, ['outward_amount', 'Outward Amount', 'Debit Amount'])
            
            # If Inward_Amount/Outward_Amount not found, derive from Amount for backward compatibility
            if inward_amount is None or pd.isna(inward_amount):
                inward_amount = amount if amount > 0 else 0.0
            if outward_amount is None or pd.isna(outward_amount):
                outward_amount = abs(amount) if amount < 0 else 0.0
            
            ai_category = safe_get('Category', '', str, ['category', 'ai_category', 'Transaction_Category'])
            balance = safe_get('Balance', None, float, ['Closing_Balance', 'closing_balance', 'balance', 'Account_Balance'])
            transaction_type = safe_get('Type', None, str, ['type', 'Transaction_Type', '_type'])
            
            # Convert date to string if it's a datetime object
            if transaction_date and not isinstance(transaction_date, str):
                try:
                    if hasattr(transaction_date, 'strftime'):
                        transaction_date = transaction_date.strftime('%Y-%m-%d')
                    else:
                        transaction_date = str(transaction_date)
                except:
                    transaction_date = str(transaction_date) if transaction_date else ''
            
            db_manager.store_transaction(
                session_id=session_id,
                file_id=file_id,
                row_number=idx + 1,
                transaction_date=transaction_date,
                description=description,
                amount=amount,
                ai_category=ai_category,
                balance=balance,
                transaction_type=transaction_type,
                vendor_name=None,  # Will add vendor extraction later
                ai_confidence=None,  # Will add confidence scores later
                ai_reasoning=ai_reasoning,  # Store AI reasoning
                inward_amount=inward_amount,
                outward_amount=outward_amount
            )
            ai_reasonings.append(ai_reasoning)
        
        # Ensure all transactions are committed
        try:
            conn = db_manager.get_connection()
            if conn and conn.is_connected():
                conn.commit()
                print(f"✅ Committed {total_transactions} transactions to database")
        except Exception as commit_error:
            print(f"⚠️ Commit warning (may be using autocommit): {commit_error}")
        
        # Attach reasoning to DataFrame for frontend use
        if len(ai_reasonings) == len(df):
            df['AI_Reasoning'] = ai_reasonings
        
        print(f"✅ Generated and stored AI reasoning for {total_transactions} transactions")
        
        # Complete analysis session
        processing_time = time.time() - start_time
        bank_count = len(df)
        success_rate = (ai_categorized / bank_count * 100) if bank_count > 0 else 0
        
        db_manager.complete_analysis_session(
            session_id=session_id,
            transaction_count=bank_count,
            processing_time=processing_time,
            success_rate=success_rate
        )
        
        print(f"✅ Successfully stored {bank_count} transactions in MySQL database!")
        print(f"📊 File ID: {file_id}, Session ID: {session_id}")
        
        # Store session state if available
        try:
            import sys
            import os
            # Add parent directory to path for imports
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from persistent_state_manager import state_manager as pm_state_manager
            PERSISTENT_STATE_AVAILABLE = pm_state_manager is not None
        except:
            PERSISTENT_STATE_AVAILABLE = False
        
        if PERSISTENT_STATE_AVAILABLE:
            try:
                pm_state_manager.set_current_session(session_id, file_id)
                
                global_data = {
                    'reconciliation_data': reconciliation_data or {},
                    'uploaded_bank_df': df,
                    'uploaded_sap_df': None,
                    'bank_count': bank_count,
                    'sap_count': 0,
                    'ai_categorized': ai_categorized,
                    'processing_time': processing_time,
                    'upload_timestamp': datetime.now().isoformat()
                }
                
                pm_state_manager.save_global_state(global_data)
                print(f"✅ State: Auto-saved session state for session {session_id}")
            except Exception as state_error:
                print(f"⚠️ State auto-save failed: {state_error}")
        
        # Store category insights
        try:
            import sys
            import os
            # Add parent directory to path for imports
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from analysis_storage_integration import store_category_insights
            
            category_breakdown = {}
            for category in ['Operating Activities', 'Investing Activities', 'Financing Activities']:
                category_transactions = df[df['Category'].str.contains(category, na=False)]
                if not category_transactions.empty:
                    # Use Inward_Amount and Outward_Amount - no fallback to Amount
                    if 'Inward_Amount' not in category_transactions.columns or 'Outward_Amount' not in category_transactions.columns:
                        raise ValueError(f"Inward_Amount and Outward_Amount columns are required for category totals. No fallback to Amount.")
                    
                    total_inward = float(category_transactions['Inward_Amount'].fillna(0).sum())
                    total_outward = float(category_transactions['Outward_Amount'].fillna(0).apply(lambda x: abs(float(x)) if pd.notna(x) else 0.0).sum())
                    net_total = total_inward - total_outward
                    
                    category_breakdown[category] = {
                        'count': len(category_transactions),
                        'total': net_total,
                        'average': net_total / len(category_transactions) if len(category_transactions) > 0 else 0.0,
                        'percentage': len(category_transactions) / bank_count * 100
                    }
            
            store_category_insights(db_manager, file_id, session_id, category_breakdown)
            print("✅ Categories analysis stored in database!")
        except Exception as storage_error:
            print(f"⚠️ Categories storage failed: {storage_error}")
        
        # Update session
        session['mysql_file_id'] = file_id
        session['mysql_session_id'] = session_id
        
        return {
            'file_id': file_id,
            'session_id': session_id,
            'transaction_count': bank_count,
            'processing_time': processing_time
        }
        
    except Exception as e:
        print(f"⚠️ MySQL storage failed: {e}")
        return None

