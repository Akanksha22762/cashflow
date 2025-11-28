"""
Database Cache Manager
======================
Checks database for existing file analysis results before processing.
Prevents re-running expensive AI analysis on duplicate file uploads.
"""

import pandas as pd
import hashlib
import os
import logging
import json
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from mysql_database_manager import MySQLDatabaseManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseCacheManager:
    """
    Manages database cache lookups to avoid re-processing duplicate files.
    """
    
    def __init__(self, db_manager: MySQLDatabaseManager = None):
        """
        Initialize the cache manager.
        
        Args:
            db_manager: MySQLDatabaseManager instance (optional, will create if not provided)
        """
        self.db_manager = db_manager
        if not self.db_manager:
            try:
                self.db_manager = MySQLDatabaseManager()
            except Exception as e:
                logger.warning(f"Could not initialize database manager: {e}")
                self.db_manager = None
    
    def calculate_file_hash(self, file_path: str) -> str:
        """
        Calculate SHA-256 hash of file for duplicate detection.
        
        Args:
            file_path: Path to the file
            
        Returns:
            SHA-256 hash string
        """
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating file hash: {e}")
            return ""
    
    def check_file_exists(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Check if file already exists in database by hash.
        
        Args:
            file_path: Path to the uploaded file
            
        Returns:
            Dictionary with file_id and session_id if exists, None otherwise
        """
        if not self.db_manager:
            return None
        
        try:
            file_hash = self.calculate_file_hash(file_path)
            if not file_hash:
                return None
            
            conn = self.db_manager.get_connection()
            cursor = conn.cursor(dictionary=True)
            
            # Check if file exists
            cursor.execute(
                "SELECT file_id FROM files WHERE file_hash = %s",
                (file_hash,)
            )
            file_record = cursor.fetchone()
            
            if not file_record:
                logger.info(f"📄 File not found in database (new file)")
                return None
            
            file_id = file_record['file_id']
            logger.info(f"✅ File found in database (file_id: {file_id})")
            
            # Get latest completed session for this file
            cursor.execute("""
                SELECT session_id, transaction_count, completed_at
                FROM analysis_sessions
                WHERE file_id = %s 
                  AND status = 'completed'
                ORDER BY completed_at DESC
                LIMIT 1
            """, (file_id,))
            
            session_record = cursor.fetchone()
            
            if not session_record:
                logger.info(f"⚠️ File exists but no completed session found")
                return None
            
            logger.info(f"✅ Found completed session (session_id: {session_record['session_id']}, "
                       f"transactions: {session_record['transaction_count']})")
            
            return {
                'file_id': file_id,
                'session_id': session_record['session_id'],
                'transaction_count': session_record['transaction_count'],
                'completed_at': session_record['completed_at']
            }
            
        except Exception as e:
            logger.error(f"Error checking file existence: {e}")
            return None
    
    def retrieve_transactions_from_db(self, session_id: int) -> Optional[pd.DataFrame]:
        """
        Retrieve transactions from database and convert to DataFrame.
        
        Args:
            session_id: Analysis session ID
            
        Returns:
            DataFrame with transactions in the same format as upload processing
        """
        if not self.db_manager:
            return None
        
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor(dictionary=True)
            
            # Get all transactions for this session
            cursor.execute("""
                SELECT 
                    transaction_date as Date,
                    description as Description,
                    amount as Amount,
                    inward_amount as Inward_Amount,
                    outward_amount as Outward_Amount,
                    transaction_type as Type,
                    ai_category as Category,
                    balance as Balance,
                    vendor_name as Vendor,
                    ai_confidence_score as AI_Confidence,
                    ai_reasoning as AI_Reasoning,
                    original_row_number as Original_Row_Number
                FROM transactions
                WHERE session_id = %s
                ORDER BY original_row_number
            """, (session_id,))
            
            transactions = cursor.fetchall()
            
            if not transactions:
                logger.warning(f"No transactions found for session_id: {session_id}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(transactions)
            if 'Vendor' in df.columns and 'Assigned_Vendor' not in df.columns:
                df['Assigned_Vendor'] = df['Vendor']
            
            # Ensure Date is datetime
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            
            # Ensure Amount is numeric
            if 'Amount' in df.columns:
                df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
            
            # Ensure Inward_Amount and Outward_Amount are numeric
            if 'Inward_Amount' in df.columns:
                df['Inward_Amount'] = pd.to_numeric(df['Inward_Amount'], errors='coerce')
            if 'Outward_Amount' in df.columns:
                df['Outward_Amount'] = pd.to_numeric(df['Outward_Amount'], errors='coerce')
            
            # For old cached data: derive Inward_Amount and Outward_Amount from Amount if they are missing/empty
            # This handles old cached data where inward_amount/outward_amount columns exist but are NULL/0
            if 'Amount' in df.columns:
                # Check if Inward_Amount/Outward_Amount are all null or zero
                inward_all_null = 'Inward_Amount' not in df.columns or df['Inward_Amount'].isna().all() or (df['Inward_Amount'] == 0).all()
                outward_all_null = 'Outward_Amount' not in df.columns or df['Outward_Amount'].isna().all() or (df['Outward_Amount'] == 0).all()
                
                if inward_all_null or outward_all_null:
                    # Derive Inward_Amount and Outward_Amount from Amount
                    # Positive amounts = Inward, Negative amounts = Outward
                    if inward_all_null:
                        df['Inward_Amount'] = df['Amount'].apply(lambda x: x if x > 0 else 0.0)
                    if outward_all_null:
                        df['Outward_Amount'] = df['Amount'].apply(lambda x: abs(x) if x < 0 else 0.0)
                    logger.info("Derived Inward_Amount and Outward_Amount from Amount for old cached data")
            
            # Fill NaN with 0.0 only after derivation
            if 'Inward_Amount' in df.columns:
                df['Inward_Amount'] = df['Inward_Amount'].fillna(0.0)
            if 'Outward_Amount' in df.columns:
                df['Outward_Amount'] = df['Outward_Amount'].fillna(0.0)
            
            # Ensure Balance is numeric
            if 'Balance' in df.columns:
                df['Balance'] = pd.to_numeric(df['Balance'], errors='coerce')
            
            # Map Balance to Closing_Balance if needed
            if 'Closing_Balance' not in df.columns and 'Balance' in df.columns:
                df['Closing_Balance'] = df['Balance']
            
            # Parse AI reasoning JSON strings back to objects
            if 'AI_Reasoning' in df.columns:
                def _parse_reasoning(value):
                    if isinstance(value, str):
                        value = value.strip()
                        if not value:
                            return None
                        try:
                            return json.loads(value)
                        except Exception:
                            return value
                    return value
                df['AI_Reasoning'] = df['AI_Reasoning'].apply(_parse_reasoning)
            
            # Add additional columns that might be expected
            if 'Year' not in df.columns and 'Date' in df.columns:
                df['Year'] = df['Date'].dt.year
            if 'Month' not in df.columns and 'Date' in df.columns:
                df['Month'] = df['Date'].dt.month
            if 'Day' not in df.columns and 'Date' in df.columns:
                df['Day'] = df['Date'].dt.day
            
            logger.info(f"✅ Retrieved {len(df)} transactions from database")
            
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving transactions from database: {e}")
            return None
    
    def get_cached_analysis(self, file_path: str) -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
        """
        Check if file exists in database and retrieve cached analysis.
        
        Args:
            file_path: Path to the uploaded file
            
        Returns:
            Tuple of (DataFrame, metadata_dict) if found, None otherwise
        """
        # Check if file exists
        file_info = self.check_file_exists(file_path)
        
        if not file_info:
            return None
        
        # Retrieve transactions
        df = self.retrieve_transactions_from_db(file_info['session_id'])
        
        if df is None:
            return None
        
        # Prepare metadata
        metadata = {
            'file_id': file_info['file_id'],
            'session_id': file_info['session_id'],
            'transaction_count': file_info['transaction_count'],
            'completed_at': file_info['completed_at'],
            'from_cache': True
        }
        
        logger.info(f"🎯 Using cached analysis from database (skipping AI processing)")
        
        return df, metadata


def check_and_retrieve_cached_file(file_path: str, db_manager: MySQLDatabaseManager = None) -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
    """
    Convenience function to check database cache and retrieve existing analysis.
    
    Args:
        file_path: Path to the uploaded file
        db_manager: MySQLDatabaseManager instance (optional)
        
    Returns:
        Tuple of (DataFrame, metadata_dict) if found, None otherwise
    """
    cache_manager = DatabaseCacheManager(db_manager)
    return cache_manager.get_cached_analysis(file_path)

