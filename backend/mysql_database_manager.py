"""
MySQL Database Manager for Cashflow SAP Bank Analysis System
Handles all MySQL database operations for storing and retrieving analysis results
"""

import mysql.connector
import json
import hashlib
import logging
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import pandas as pd
import os
import uuid

class MySQLDatabaseManager:
    """
    Manages MySQL database operations for storing and retrieving analysis results
    """
    
    def __init__(self, host: str = "cashflow.c1womgmu83di.ap-south-1.rds.amazonaws.com", port: int = 3306, user: str = "admin", 
                 password: str = "cashflow123", database: str = "cashflow"):
        """
        Initialize MySQL database manager
        
        Args:
            host: MySQL server host
            port: MySQL server port
            user: MySQL username
            password: MySQL password
            database: Database name
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.connection = None
        self.setup_logging()
    
    def _sanitize_for_json(self, obj):
        """Recursively sanitize object for JSON serialization, handling NaN values"""
        if isinstance(obj, dict):
            return {key: self._sanitize_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize_for_json(item) for item in obj]
        elif isinstance(obj, (np.float64, np.float32)):
            if np.isnan(obj):
                return None
            elif np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        elif obj != obj:  # NaN check
            return None
        return obj
        
    def setup_logging(self):
        """Setup logging for database operations"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('cashflow_app.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_connection(self):
        """Get MySQL database connection"""
        try:
            if self.connection is None or not self.connection.is_connected():
                self.connection = mysql.connector.connect(
                    host=self.host,
                    port=self.port,
                    user=self.user,
                    password=self.password,
                    database=self.database,
                    charset='utf8mb4',
                    collation='utf8mb4_unicode_ci',
                    autocommit=True
                )
                self.logger.info(f"SUCCESS: Connected to MySQL database: {self.database}")
            return self.connection
        except mysql.connector.Error as e:
            self.logger.error(f"ERROR: MySQL connection error: {e}")
            raise
    
    def close_connection(self):
        """Close MySQL database connection"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            self.connection = None
            self.logger.info("INFO: MySQL connection closed")
    
    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file for duplicate detection"""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            self.logger.error(f"Error calculating file hash: {e}")
            return ""
    
    def store_file_metadata(self, filename: str, file_path: str, data_source: str = 'bank') -> int:
        """
        Store file metadata and return file_id
        
        Args:
            filename: Name of the uploaded file
            file_path: Path to the uploaded file
            data_source: Data source ('bank', 'sap', 'other')
            
        Returns:
            file_id: Database ID of the stored file
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Calculate file hash and size
            file_hash = self.calculate_file_hash(file_path)
            file_size = os.path.getsize(file_path)
            file_type = Path(filename).suffix.lower().replace('.', '')
            
            # Check if file already exists (for override system)
            cursor.execute(
                "SELECT file_id FROM files WHERE file_hash = %s",
                (file_hash,)
            )
            existing_file = cursor.fetchone()
            
            if existing_file:
                # Update existing file metadata
                file_id = existing_file[0]
                cursor.execute("""
                    UPDATE files 
                    SET last_processed = CURRENT_TIMESTAMP,
                        processing_status = 'pending'
                    WHERE file_id = %s
                """, (file_id,))
                
                self.logger.info(f"INFO: Updated existing file metadata for file_id: {file_id}")
            else:
                # Insert new file metadata
                cursor.execute("""
                    INSERT INTO files 
                    (filename, file_hash, file_size, file_type, data_source)
                    VALUES (%s, %s, %s, %s, %s)
                """, (filename, file_hash, file_size, file_type, data_source))
                
                file_id = cursor.lastrowid
                self.logger.info(f"INFO: Stored new file metadata with file_id: {file_id}")
            
            return file_id
            
        except mysql.connector.Error as e:
            self.logger.error(f"Error storing file metadata: {e}")
            raise
    
    def create_analysis_session(self, file_id: int, analysis_type: str = 'full_analysis') -> int:
        """
        Create a new analysis session
        
        Args:
            file_id: ID of the file being analyzed
            analysis_type: Type of analysis being performed
            
        Returns:
            session_id: Database ID of the created session
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            session_uuid = str(uuid.uuid4())
            
            cursor.execute("""
                INSERT INTO analysis_sessions 
                (file_id, session_uuid, analysis_type, status)
                VALUES (%s, %s, %s, 'processing')
            """, (file_id, session_uuid, analysis_type))
            
            session_id = cursor.lastrowid
            self.logger.info(f"SUCCESS: Created analysis session with session_id: {session_id}")
            return session_id
            
        except mysql.connector.Error as e:
            self.logger.error(f"Error creating analysis session: {e}")
            raise
    
    def store_transaction(self, session_id: int, file_id: int, row_number: int,
                         transaction_date: str, description: str, amount: float,
                         ai_category: str, balance: float = None, 
                         transaction_type: str = None, vendor_name: str = None,
                         ai_confidence: float = None) -> int:
        """
        Store a single transaction with AI analysis results
        
        Args:
            session_id: Analysis session ID
            file_id: File ID
            row_number: Original row number in file
            transaction_date: Transaction date
            description: Transaction description
            amount: Transaction amount
            ai_category: AI-assigned category
            balance: Account balance (optional)
            transaction_type: Transaction type (optional)
            vendor_name: Extracted vendor name (optional)
            ai_confidence: AI confidence score (optional)
            
        Returns:
            transaction_id: Database ID of the stored transaction
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO transactions 
                (session_id, file_id, original_row_number, transaction_date, 
                 description, amount, balance, transaction_type, ai_category, 
                 ai_confidence_score, vendor_name)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (session_id, file_id, row_number, transaction_date, description, 
                  amount, balance, transaction_type, ai_category, ai_confidence, vendor_name))
            
            transaction_id = cursor.lastrowid
            return transaction_id
            
        except mysql.connector.Error as e:
            self.logger.error(f"Error storing transaction: {e}")
            raise
    
    def complete_analysis_session(self, session_id: int, transaction_count: int, 
                                 processing_time: float, success_rate: float = None):
        """
        Mark analysis session as completed
        
        Args:
            session_id: Analysis session ID
            transaction_count: Number of transactions processed
            processing_time: Processing time in seconds
            success_rate: AI categorization success rate
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE analysis_sessions 
                SET status = 'completed',
                    completed_at = CURRENT_TIMESTAMP,
                    transaction_count = %s,
                    processing_time_seconds = %s,
                    success_rate = %s
                WHERE session_id = %s
            """, (transaction_count, processing_time, success_rate, session_id))
            
            # Also update file status
            cursor.execute("""
                UPDATE files 
                SET processing_status = 'completed',
                    completed_at = CURRENT_TIMESTAMP
                WHERE file_id = (SELECT file_id FROM analysis_sessions WHERE session_id = %s)
            """, (session_id,))
            
            self.logger.info(f"SUCCESS: Completed analysis session {session_id}")
            
        except mysql.connector.Error as e:
            self.logger.error(f"Error completing analysis session: {e}")
            raise

    def store_ai_model_performance(self, session_id: int, model_name: str, model_version: str,
                                 total_predictions: int, successful_predictions: int,
                                 failed_predictions: int = 0, average_confidence: float = None,
                                 processing_time_ms: float = None, memory_usage_mb: float = None) -> int:
        """
        Store AI model performance metrics for analysis tracking
        
        Args:
            session_id: Analysis session ID
            model_name: Name of the AI model (e.g., 'ollama', 'xgboost')
            model_version: Version of the model (e.g., 'llama3.2:3b', 'xgboost_v1.0')
            total_predictions: Total number of predictions made
            successful_predictions: Number of successful predictions
            failed_predictions: Number of failed predictions
            average_confidence: Average confidence score (0.0-1.0)
            processing_time_ms: Processing time in milliseconds
            memory_usage_mb: Memory usage in MB
            
        Returns:
            int: Performance record ID if successful, None otherwise
        """
        try:
            cursor = self.connection.cursor()
            
            # Calculate error rate
            error_rate = (failed_predictions / total_predictions) if total_predictions > 0 else 0.0
            
            insert_query = """
                INSERT INTO ai_model_performance (
                    session_id, model_name, model_version, total_predictions,
                    successful_predictions, failed_predictions, average_confidence,
                    processing_time_ms, memory_usage_mb, error_rate
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(insert_query, (
                session_id, model_name, model_version, total_predictions,
                successful_predictions, failed_predictions, average_confidence,
                processing_time_ms, memory_usage_mb, error_rate
            ))
            
            performance_id = cursor.lastrowid
            self.connection.commit()
            
            self.logger.info(f"SUCCESS: Stored AI model performance: {model_name} v{model_version} - {successful_predictions}/{total_predictions} successful")
            return performance_id
            
        except mysql.connector.Error as e:
            self.logger.error(f"Error storing AI model performance: {e}")
            return None
    
    def get_latest_results_by_filename(self, filename: str) -> Dict[str, Any]:
        """Get latest analysis results for a specific filename"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)
            
            # Get file metadata and latest session
            cursor.execute("""
                SELECT f.*, s.session_id, s.analysis_type, s.completed_at, 
                       s.transaction_count, s.processing_time_seconds, s.success_rate
                FROM files f
                LEFT JOIN analysis_sessions s ON f.file_id = s.file_id
                WHERE f.filename = %s 
                ORDER BY f.last_processed DESC, s.completed_at DESC
                LIMIT 1
            """, (filename,))
            
            file_record = cursor.fetchone()
            if not file_record:
                return None
            
            results = {'file_metadata': file_record}
            
            # Get transactions for this file
            if file_record['session_id']:
                cursor.execute("""
                    SELECT * FROM transactions 
                    WHERE session_id = %s 
                    ORDER BY original_row_number
                """, (file_record['session_id'],))
                
                transactions = cursor.fetchall()
                results['transactions'] = transactions
            
            return results
            
        except mysql.connector.Error as e:
            self.logger.error(f"Error retrieving results by filename: {e}")
            raise
    
    def get_all_files(self) -> List[Dict[str, Any]]:
        """Get all processed files"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)
            
            cursor.execute("""
                SELECT f.*, 
                       COUNT(s.session_id) as session_count,
                       MAX(s.completed_at) as last_analysis,
                       SUM(s.transaction_count) as total_transactions
                FROM files f
                LEFT JOIN analysis_sessions s ON f.file_id = s.file_id
                GROUP BY f.file_id
                ORDER BY f.last_processed DESC
            """)
            
            files = cursor.fetchall()
            return files
            
        except mysql.connector.Error as e:
            self.logger.error(f"Error retrieving all files: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Test MySQL database connection"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            self.logger.info("✅ MySQL connection test successful")
            return True
        except Exception as e:
            self.logger.error(f"❌ MySQL connection test failed: {e}")
            return False
    
    def store_session_state(self, session_id: int, state_type: str, state_data: dict, 
                           user_session_id: str = None) -> int:
        """
        Store session state for persistence across app restarts
        
        Args:
            session_id: Analysis session ID
            state_type: Type of state (ui_state, global_data, analysis_results)
            state_data: Dictionary containing the state data
            user_session_id: Browser session ID for user tracking
            
        Returns:
            state_id: Database ID of stored state
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Convert state data to JSON
            state_json = json.dumps(state_data, default=str, ensure_ascii=False)
            
            # Store or update session state
            cursor.execute("""
                INSERT INTO session_states 
                (session_id, state_type, state_data, user_session_id)
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                state_data = VALUES(state_data),
                updated_at = CURRENT_TIMESTAMP
            """, (session_id, state_type, state_json, user_session_id))
            
            state_id = cursor.lastrowid if cursor.lastrowid else cursor.rowcount
            self.logger.info(f"SUCCESS: Stored session state (type: {state_type}, session_id: {session_id})")
            return state_id
            
        except mysql.connector.Error as e:
            self.logger.error(f"Error storing session state: {e}")
            raise
    
    def get_latest_session_state(self, state_type: str = None) -> Dict[str, Any]:
        """
        Get the latest session state for restoration
        
        Args:
            state_type: Specific state type to retrieve (optional)
            
        Returns:
            Dictionary containing session state data
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)
            
            if state_type:
                cursor.execute("""
                    SELECT ss.*, asession.file_id, asession.analysis_type
                    FROM session_states ss
                    JOIN analysis_sessions asession ON ss.session_id = asession.session_id
                    WHERE ss.state_type = %s
                    ORDER BY ss.updated_at DESC
                    LIMIT 1
                """, (state_type,))
            else:
                cursor.execute("""
                    SELECT ss.*, asession.file_id, asession.analysis_type
                    FROM session_states ss
                    JOIN analysis_sessions asession ON ss.session_id = asession.session_id
                    ORDER BY ss.updated_at DESC
                    LIMIT 1
                """)
            
            result = cursor.fetchone()
            if result and result['state_data']:
                result['state_data'] = json.loads(result['state_data'])
                return result
            
            return None
            
        except mysql.connector.Error as e:
            self.logger.error(f"Error retrieving session state: {e}")
            return None
    
    def get_all_session_states(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get all recent session states for user selection
        
        Args:
            limit: Maximum number of sessions to return
            
        Returns:
            List of session state dictionaries
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)
            
            cursor.execute("""
                SELECT DISTINCT 
                    ss.session_id,
                    asession.file_id,
                    asession.analysis_type,
                    asession.started_at,
                    asession.completed_at,
                    asession.transaction_count,
                    f.filename,
                    f.data_source,
                    MAX(ss.updated_at) as last_updated
                FROM session_states ss
                JOIN analysis_sessions asession ON ss.session_id = asession.session_id
                JOIN files f ON asession.file_id = f.file_id
                GROUP BY ss.session_id
                ORDER BY last_updated DESC
                LIMIT %s
            """, (limit,))
            
            sessions = cursor.fetchall()
            return sessions
            
        except mysql.connector.Error as e:
            self.logger.error(f"Error retrieving all session states: {e}")
            return []
    
    def restore_session_data(self, session_id: int) -> Dict[str, Any]:
        """
        Restore complete session data including all state types
        
        Args:
            session_id: Session ID to restore
            
        Returns:
            Dictionary containing all session data
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)
            
            # Get all state data for this session
            cursor.execute("""
                SELECT state_type, state_data, updated_at
                FROM session_states 
                WHERE session_id = %s
                ORDER BY updated_at DESC
            """, (session_id,))
            
            states = cursor.fetchall()
            restored_data = {}
            
            for state in states:
                try:
                    # Clean legacy NaN values before JSON parsing
                    state_data_str = state['state_data']
                    if isinstance(state_data_str, str):
                        # Clean NaN/Infinity values from JSON string
                        import re
                        state_data_str = re.sub(r'\bNaN\b', 'null', state_data_str)
                        state_data_str = re.sub(r'\bInfinity\b', 'null', state_data_str)
                        state_data_str = re.sub(r'\b-Infinity\b', 'null', state_data_str)
                        state_data_str = re.sub(r':\s*NaN\s*([,}])', r': null\1', state_data_str)
                        state_data_str = re.sub(r':\s*Infinity\s*([,}])', r': null\1', state_data_str)
                        state_data_str = re.sub(r':\s*-Infinity\s*([,}])', r': null\1', state_data_str)
                    
                    state_data = json.loads(state_data_str)
                    restored_data[state['state_type']] = {
                        'data': state_data,
                        'updated_at': state['updated_at']
                    }
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse state data for {state['state_type']}: {e}")
                    # Store error info for debugging but continue processing
                    restored_data[state['state_type']] = {
                        'data': {'error': 'JSON parse failed', 'type': state['state_type']},
                        'updated_at': state['updated_at']
                    }
            
            # Get session metadata
            cursor.execute("""
                SELECT s.*, f.filename, f.data_source
                FROM analysis_sessions s
                JOIN files f ON s.file_id = f.file_id
                WHERE s.session_id = %s
            """, (session_id,))
            
            session_metadata = cursor.fetchone()
            if session_metadata:
                restored_data['session_metadata'] = session_metadata
            
            # Get transactions for this session
            cursor.execute("""
                SELECT * FROM transactions 
                WHERE session_id = %s 
                ORDER BY original_row_number
            """, (session_id,))
            
            transactions = cursor.fetchall()
            if transactions:
                restored_data['transactions'] = transactions
            
            self.logger.info(f"SUCCESS: Restored session data for session_id: {session_id}")
            return restored_data
            
        except mysql.connector.Error as e:
            self.logger.error(f"Error restoring session data: {e}")
            return {}

    def __del__(self):
        """Cleanup when object is destroyed"""
        self.close_connection()
    
    def store_multiple_trends_analysis(self, session_id: int, file_id: int, trends_data: Dict[str, Any], 
                                     analysis_metadata: Dict[str, Any]) -> bool:
        """
        Store multiple trends analysis results with detailed breakdown
        
        Args:
            session_id: Analysis session ID
            file_id: File ID
            trends_data: Complete trends analysis data
            analysis_metadata: Analysis metadata including trend types
            
        Returns:
            Boolean indicating success
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Store main analysis record in session_states
            analysis_parameter = analysis_metadata.get('analysis_parameter', '')
            if isinstance(analysis_parameter, list):
                analysis_parameter = ','.join(analysis_parameter)
            
            # Sanitize data before JSON serialization
            state_data = {
                'analysis_type': 'trends_analysis',
                'analysis_parameter': analysis_parameter,
                'trend_types_analyzed': analysis_metadata.get('trend_types_analyzed', []),
                'analysis_scope': analysis_metadata.get('analysis_scope', 'single'),
                'trends_count': analysis_metadata.get('trends_count', 1),
                'is_multiple_trends': analysis_metadata.get('is_multiple_trends', False),
                'results': trends_data,
                'timestamp': analysis_metadata.get('timestamp', time.time())
            }
            
            # Sanitize all data to prevent NaN/Infinity JSON errors
            sanitized_data = self._sanitize_for_json(state_data)
            
            cursor.execute("""
                INSERT INTO session_states (
                    session_id, state_type, state_data, created_at, updated_at
                ) VALUES (%s, %s, %s, NOW(), NOW())
                ON DUPLICATE KEY UPDATE
                state_data = VALUES(state_data),
                updated_at = NOW()
            """, (
                session_id, 
                'trends_analysis',
                json.dumps(sanitized_data)
            ))
            
            # Check if trends_analysis_details table exists, create if not
            self._ensure_trends_details_table()
            
            # Store individual trend details if multiple trends
            trend_types = analysis_metadata.get('trend_types_analyzed', [])
            if trend_types and len(trend_types) > 0:
                trends_analysis_data = trends_data.get('data', {}).get('trends_analysis', {})
                processing_time = trends_data.get('data', {}).get('analysis_summary', {}).get('processing_time', 0)
                
                for trend_type in trend_types:
                    if trend_type in trends_analysis_data and trend_type != '_summary':
                        trend_result = trends_analysis_data[trend_type]
                        
                        # Sanitize trend result before storing
                        sanitized_trend_result = self._sanitize_for_json(trend_result)
                        
                        cursor.execute("""
                            INSERT INTO trends_analysis_details (
                                session_id, file_id, trend_type, trend_results,
                                trend_direction, confidence_score, pattern_strength,
                                processing_time_seconds, analysis_timestamp
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                        """, (
                            session_id, file_id, trend_type,
                            json.dumps(sanitized_trend_result),
                            trend_result.get('trend_direction', 'stable'),
                            trend_result.get('confidence', 0.75),
                            trend_result.get('pattern_strength', 'Moderate'),
                            processing_time / len(trend_types) if len(trend_types) > 0 else processing_time
                        ))
            
            conn.commit()
            self.logger.info(f"SUCCESS: Stored multiple trends analysis for session {session_id}")
            return True
            
        except mysql.connector.Error as e:
            self.logger.error(f"Error storing multiple trends analysis: {e}")
            if conn:
                conn.rollback()
            return False
        except Exception as e:
            self.logger.error(f"Error storing multiple trends analysis: {e}")
            return False
    
    def _ensure_trends_details_table(self):
        """Ensure trends_analysis_details table exists"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trends_analysis_details (
                    detail_id INT AUTO_INCREMENT PRIMARY KEY,
                    session_id INT NOT NULL,
                    file_id INT NOT NULL,
                    trend_type VARCHAR(100) NOT NULL,
                    trend_results TEXT NOT NULL,
                    analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    trend_direction VARCHAR(50),
                    confidence_score DECIMAL(3,2),
                    pattern_strength VARCHAR(50),
                    processing_time_seconds DECIMAL(8,2),
                    
                    INDEX idx_session_trend (session_id, trend_type),
                    INDEX idx_file_trend (file_id, trend_type),
                    INDEX idx_trend_type (trend_type),
                    INDEX idx_analysis_timestamp (analysis_timestamp)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            
            self.logger.info("SUCCESS: trends_analysis_details table ready")
            
        except mysql.connector.Error as e:
            self.logger.error(f"Error creating trends_analysis_details table: {e}")
    
    def get_trends_analysis_history(self, file_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get history of trends analysis for a file
        
        Args:
            file_id: File ID to get history for
            limit: Maximum number of records to return
            
        Returns:
            List of trends analysis history records
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)
            
            cursor.execute("""
                SELECT 
                    ss.session_id,
                    ss.created_at as analysis_timestamp,
                    JSON_EXTRACT(ss.state_data, '$.analysis_scope') as analysis_scope,
                    JSON_EXTRACT(ss.state_data, '$.trends_count') as trends_count,
                    JSON_EXTRACT(ss.state_data, '$.trend_types_analyzed') as trend_types_analyzed,
                    JSON_EXTRACT(ss.state_data, '$.is_multiple_trends') as is_multiple_trends,
                    COUNT(tad.detail_id) as detailed_trends_count,
                    AVG(tad.confidence_score) as avg_confidence
                FROM session_states ss
                LEFT JOIN trends_analysis_details tad ON ss.session_id = tad.session_id
                WHERE tad.file_id = %s AND ss.state_type = 'trends_analysis'
                GROUP BY ss.session_id
                ORDER BY ss.created_at DESC
                LIMIT %s
            """, (file_id, limit))
            
            results = cursor.fetchall()
            
            # Clean up JSON fields
            for result in results:
                if result.get('trend_types_analyzed'):
                    try:
                        result['trend_types_analyzed'] = json.loads(result['trend_types_analyzed'].replace('"', '"'))
                    except:
                        result['trend_types_analyzed'] = []
                
                result['analysis_scope'] = str(result.get('analysis_scope', 'single')).replace('"', '')
                result['trends_count'] = int(result.get('trends_count', 1) or 1)
                result['is_multiple_trends'] = bool(result.get('is_multiple_trends', False))
            
            return results
            
        except mysql.connector.Error as e:
            self.logger.error(f"Error getting trends analysis history: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Error getting trends analysis history: {e}")
            return []
    
    def restore_multiple_trends_session(self, session_id: int) -> Dict[str, Any]:
        """
        Restore session with multiple trends analysis
        
        Args:
            session_id: Session ID to restore
            
        Returns:
            Dictionary containing restored multiple trends data
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)
            
            # Get main trends analysis data from session_states
            cursor.execute("""
                SELECT state_data FROM session_states 
                WHERE session_id = %s AND state_type = 'trends_analysis'
                ORDER BY updated_at DESC
                LIMIT 1
            """, (session_id,))
            
            main_data = cursor.fetchone()
            if not main_data:
                return {}
            
            try:
                # Clean legacy NaN values before parsing trends data
                state_data_str = main_data['state_data']
                if isinstance(state_data_str, str):
                    import re
                    state_data_str = re.sub(r'\bNaN\b', 'null', state_data_str)
                    state_data_str = re.sub(r'\bInfinity\b', 'null', state_data_str)
                    state_data_str = re.sub(r'\b-Infinity\b', 'null', state_data_str)
                    state_data_str = re.sub(r':\s*NaN\s*([,}])', r': null\1', state_data_str)
                    state_data_str = re.sub(r':\s*Infinity\s*([,}])', r': null\1', state_data_str)
                    state_data_str = re.sub(r':\s*-Infinity\s*([,}])', r': null\1', state_data_str)
                
                analysis_data = json.loads(state_data_str)
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse trends analysis data for session {session_id}: {e}")
                return {}
            
            # Get detailed trend results if available
            cursor.execute("""
                SELECT * FROM trends_analysis_details 
                WHERE session_id = %s
                ORDER BY trend_type
            """, (session_id,))
            
            trend_details = cursor.fetchall()
            
            # Reconstruct complete analysis data
            restored_data = {
                'analysis_metadata': {
                    'analysis_scope': analysis_data.get('analysis_scope', 'single'),
                    'trends_count': analysis_data.get('trends_count', 1),
                    'is_multiple_trends': analysis_data.get('is_multiple_trends', False),
                    'trend_types_analyzed': analysis_data.get('trend_types_analyzed', [])
                },
                'main_analysis_data': analysis_data,
                'trend_details': []
            }
            
            # Add individual trend results with parsed JSON
            for detail in trend_details:
                try:
                    detail['trend_results_parsed'] = json.loads(detail['trend_results'])
                except json.JSONDecodeError:
                    detail['trend_results_parsed'] = {}
                restored_data['trend_details'].append(detail)
            
            self.logger.info(f"SUCCESS: Restored multiple trends session {session_id} with {len(trend_details)} trend details")
            return restored_data
            
        except mysql.connector.Error as e:
            self.logger.error(f"Error restoring multiple trends session: {e}")
            return {}
        except Exception as e:
            self.logger.error(f"Error restoring multiple trends session: {e}")
            return {}
    
    def get_available_trend_types_stats(self) -> Dict[str, Any]:
        """
        Get statistics about trend types usage
        
        Returns:
            Dictionary with trend type usage statistics
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)
            
            cursor.execute("""
                SELECT 
                    trend_type,
                    COUNT(*) as usage_count,
                    AVG(confidence_score) as avg_confidence,
                    MAX(analysis_timestamp) as last_used,
                    COUNT(DISTINCT session_id) as unique_sessions
                FROM trends_analysis_details 
                GROUP BY trend_type
                ORDER BY usage_count DESC
            """)
            
            trend_stats = cursor.fetchall()
            
            # Get overall statistics
            cursor.execute("""
                SELECT 
                    COUNT(DISTINCT session_id) as total_sessions,
                    COUNT(*) as total_trend_analyses,
                    AVG(confidence_score) as overall_avg_confidence
                FROM trends_analysis_details
            """)
            
            overall_stats = cursor.fetchone()
            
            return {
                'trend_statistics': trend_stats,
                'overall_statistics': overall_stats,
                'total_unique_trends': len(trend_stats)
            }
            
        except mysql.connector.Error as e:
            self.logger.error(f"Error getting trend types stats: {e}")
            return {}
        except Exception as e:
            self.logger.error(f"Error getting trend types stats: {e}")
            return {}

# Test function to verify MySQL connectivity
def test_mysql_connection():
    """Test MySQL database connection and basic operations"""
    try:
        print("🧪 Testing MySQL Database Manager...")
        
        # You'll need to provide your MySQL password here
        mysql_password = input("Enter your MySQL root password: ")
        
        # Initialize database manager
        db_manager = MySQLDatabaseManager(password=mysql_password)
        print("✅ MySQL Database manager initialized successfully")
        
        # Test connection
        if db_manager.test_connection():
            print("✅ MySQL connection working")
        else:
            print("❌ MySQL connection failed")
            return False
        
        print("🎉 All MySQL tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ MySQL test failed: {e}")
        return False

if __name__ == "__main__":
    test_mysql_connection()
