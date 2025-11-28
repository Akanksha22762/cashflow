"""
AI Reasoning Retriever Module
==============================
Handles retrieval of AI reasoning from database or generation if not found.
Moved from app.py to reduce load on main application file.
"""

import json
from typing import Dict, Any, Optional


def get_transaction_reasoning(
    transaction_desc: str,
    category: str,
    db_manager=None,
    all_transactions=None
) -> Dict[str, Any]:
    """
    Get AI reasoning for a transaction from database only.
    No fallback - reasoning must be pre-generated during upload.
    
    Args:
        transaction_desc: Transaction description
        category: Transaction category
        db_manager: MySQLDatabaseManager instance (required)
        all_transactions: Not used (kept for compatibility)
        
    Returns:
        Dictionary with reasoning data from database, or error if not found
    """
    if not db_manager:
        return {
            'status': 'error',
            'error': 'Database not available. Reasoning must be retrieved from database.',
            'source': None
        }
    
    try:
        conn = db_manager.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT ai_reasoning 
            FROM transactions 
            WHERE description = %s AND ai_category = %s 
            AND ai_reasoning IS NOT NULL
            ORDER BY transaction_id DESC
            LIMIT 1
        """, (transaction_desc, category))
        
        result = cursor.fetchone()
        if result and result[0]:
            # Parse JSON if stored as string
            if isinstance(result[0], str):
                try:
                    reasoning = json.loads(result[0])
                except:
                    reasoning = {'stored_reasoning': result[0]}
            else:
                reasoning = result[0]
            
            print(f"✅ Retrieved AI reasoning from database for: {transaction_desc[:50]}...")
            return {
                'status': 'success',
                'reasoning': reasoning,
                'source': 'database'
            }
        else:
            # Not found in database
            return {
                'status': 'error',
                'error': 'AI reasoning not found in database. Please re-upload the file to generate reasoning.',
                'source': None
            }
    except Exception as db_error:
        print(f"❌ Database lookup failed: {db_error}")
        return {
            'status': 'error',
            'error': f'Database error: {str(db_error)}',
            'source': None
        }

