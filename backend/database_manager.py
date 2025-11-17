"""
Database Manager for Cashflow SAP Bank Analysis System
Handles all database operations including CRUD, versioning, and override management
"""

import sqlite3
import json
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import pandas as pd
import os

class DatabaseManager:
    """
    Manages SQLite database operations for storing and retrieving analysis results
    """
    
    def __init__(self, db_path: str = "data/analysis_database.db"):
        """
        Initialize database manager
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.connection = None
        self.setup_logging()
        self.ensure_database_exists()
        self.create_tables()
    
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
    
    def ensure_database_exists(self):
        """Ensure database directory exists"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
            self.logger.info(f"Created database directory: {db_dir}")
    
    def get_connection(self):
        """Get database connection with proper configuration"""
        if self.connection is None:
            self.connection = sqlite3.connect(
                self.db_path,
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
            )
            self.connection.row_factory = sqlite3.Row
            # Enable foreign keys
            self.connection.execute("PRAGMA foreign_keys = ON")
        return self.connection
    
    def close_connection(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
    
    def create_tables(self):
        """Create all necessary database tables"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Table 1: File Metadata
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS file_metadata (
                    file_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    file_hash TEXT UNIQUE NOT NULL,
                    file_size INTEGER,
                    file_type TEXT,
                    upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_processed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processing_status TEXT DEFAULT 'completed',
                    industry_context TEXT,
                    analysis_version TEXT DEFAULT '1.0'
                )
            """)
            
            # Table 2: Cash Flow Analysis Results
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cash_flow_results (
                    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER NOT NULL,
                    analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    cash_flow_data TEXT NOT NULL, -- JSON string
                    summary_data TEXT NOT NULL, -- JSON string
                    categories_data TEXT NOT NULL, -- JSON string
                    trends_data TEXT NOT NULL, -- JSON string
                    analysis_metadata TEXT, -- JSON string
                    FOREIGN KEY (file_id) REFERENCES file_metadata(file_id) ON DELETE CASCADE
                )
            """)
            
            # Table 3: Vendor Extraction Results
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS vendor_extraction_results (
                    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER NOT NULL,
                    analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    vendor_data TEXT NOT NULL, -- JSON string
                    vendor_summary TEXT NOT NULL, -- JSON string
                    vendor_categories TEXT NOT NULL, -- JSON string
                    vendor_insights TEXT NOT NULL, -- JSON string
                    analysis_metadata TEXT, -- JSON string
                    FOREIGN KEY (file_id) REFERENCES file_metadata(file_id) ON DELETE CASCADE
                )
            """)
            
            # Table 4: Revenue Analysis Results
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS revenue_analysis_results (
                    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER NOT NULL,
                    analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    revenue_data TEXT NOT NULL, -- JSON string
                    revenue_summary TEXT NOT NULL, -- JSON string
                    revenue_trends TEXT NOT NULL, -- JSON string
                    revenue_insights TEXT NOT NULL, -- JSON string
                    analysis_metadata TEXT, -- JSON string
                    FOREIGN KEY (file_id) REFERENCES file_metadata(file_id) ON DELETE CASCADE
                )
            """)
            
            # Table 5: Industry Context Results
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS industry_context_results (
                    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER NOT NULL,
                    analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    industry_data TEXT NOT NULL, -- JSON string
                    industry_summary TEXT NOT NULL, -- JSON string
                    industry_insights TEXT NOT NULL, -- JSON string
                    analysis_metadata TEXT, -- JSON string
                    FOREIGN KEY (file_id) REFERENCES file_metadata(file_id) ON DELETE CASCADE
                )
            """)
            
            # Table 6: Analysis Performance Data
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_performance (
                    performance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER NOT NULL,
                    analysis_type TEXT NOT NULL,
                    processing_time REAL,
                    memory_usage REAL,
                    cpu_usage REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (file_id) REFERENCES file_metadata(file_id) ON DELETE CASCADE
                )
            """)
            
            # Table 7: Version History (for overrides)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS version_history (
                    version_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER NOT NULL,
                    previous_result_id INTEGER,
                    new_result_id INTEGER NOT NULL,
                    change_type TEXT NOT NULL, -- 'override', 'update', 'correction'
                    change_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    change_description TEXT,
                    user_id TEXT DEFAULT 'system',
                    FOREIGN KEY (file_id) REFERENCES file_metadata(file_id) ON DELETE CASCADE
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_hash ON file_metadata(file_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_filename ON file_metadata(filename)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_upload_timestamp ON file_metadata(upload_timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_id ON cash_flow_results(file_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_vendor_file_id ON vendor_extraction_results(file_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_revenue_file_id ON revenue_analysis_results(file_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_industry_file_id ON industry_context_results(file_id)")
            
            conn.commit()
            self.logger.info("Database tables created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating tables: {e}")
            raise
    
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
    
    def store_file_metadata(self, filename: str, file_path: str, industry_context: str = None) -> int:
        """
        Store file metadata and return file_id
        
        Args:
            filename: Name of the uploaded file
            file_path: Path to the uploaded file
            industry_context: Industry context if available
            
        Returns:
            file_id: Database ID of the stored file
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Calculate file hash and size
            file_hash = self.calculate_file_hash(file_path)
            file_size = os.path.getsize(file_path)
            file_type = Path(filename).suffix.lower()
            
            # Check if file already exists
            cursor.execute(
                "SELECT file_id FROM file_metadata WHERE file_hash = ?",
                (file_hash,)
            )
            existing_file = cursor.fetchone()
            
            if existing_file:
                # Update existing file metadata
                cursor.execute("""
                    UPDATE file_metadata 
                    SET last_processed = CURRENT_TIMESTAMP,
                        processing_status = 'completed',
                        industry_context = COALESCE(?, industry_context)
                    WHERE file_id = ?
                """, (industry_context, existing_file['file_id']))
                
                file_id = existing_file['file_id']
                self.logger.info(f"Updated existing file metadata for file_id: {file_id}")
            else:
                # Insert new file metadata
                cursor.execute("""
                    INSERT INTO file_metadata 
                    (filename, file_hash, file_size, file_type, industry_context)
                    VALUES (?, ?, ?, ?, ?)
                """, (filename, file_hash, file_size, file_type, industry_context))
                
                file_id = cursor.lastrowid
                self.logger.info(f"Stored new file metadata with file_id: {file_id}")
            
            conn.commit()
            return file_id
            
        except Exception as e:
            self.logger.error(f"Error storing file metadata: {e}")
            raise
    
    def store_cash_flow_results(self, file_id: int, cash_flow_data: Dict, 
                               summary_data: Dict, categories_data: Dict, 
                               trends_data: Dict, analysis_metadata: Dict = None) -> int:
        """Store cash flow analysis results"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO cash_flow_results 
                (file_id, cash_flow_data, summary_data, categories_data, trends_data, analysis_metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                file_id,
                json.dumps(cash_flow_data),
                json.dumps(summary_data),
                json.dumps(categories_data),
                json.dumps(trends_data),
                json.dumps(analysis_metadata) if analysis_metadata else None
            ))
            
            result_id = cursor.lastrowid
            conn.commit()
            self.logger.info(f"Stored cash flow results with result_id: {result_id}")
            return result_id
            
        except Exception as e:
            self.logger.error(f"Error storing cash flow results: {e}")
            raise
    
    def store_vendor_extraction_results(self, file_id: int, vendor_data: Dict,
                                      vendor_summary: Dict, vendor_categories: Dict,
                                      vendor_insights: Dict, analysis_metadata: Dict = None) -> int:
        """Store vendor extraction analysis results"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO vendor_extraction_results 
                (file_id, vendor_data, vendor_summary, vendor_categories, vendor_insights, analysis_metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                file_id,
                json.dumps(vendor_data),
                json.dumps(vendor_summary),
                json.dumps(vendor_categories),
                json.dumps(vendor_insights),
                json.dumps(analysis_metadata) if analysis_metadata else None
            ))
            
            result_id = cursor.lastrowid
            conn.commit()
            self.logger.info(f"Stored vendor extraction results with result_id: {result_id}")
            return result_id
            
        except Exception as e:
            self.logger.error(f"Error storing vendor extraction results: {e}")
            raise
    
    def store_revenue_analysis_results(self, file_id: int, revenue_data: Dict,
                                     revenue_summary: Dict, revenue_trends: Dict,
                                     revenue_insights: Dict, analysis_metadata: Dict = None) -> int:
        """Store revenue analysis results"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO revenue_analysis_results 
                (file_id, revenue_data, revenue_summary, revenue_trends, revenue_insights, analysis_metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                file_id,
                json.dumps(revenue_data),
                json.dumps(revenue_summary),
                json.dumps(revenue_trends),
                json.dumps(revenue_insights),
                json.dumps(analysis_metadata) if analysis_metadata else None
            ))
            
            result_id = cursor.lastrowid
            conn.commit()
            self.logger.info(f"Stored revenue analysis results with result_id: {result_id}")
            return result_id
            
        except Exception as e:
            self.logger.error(f"Error storing revenue analysis results: {e}")
            raise
    
    def store_industry_context_results(self, file_id: int, industry_data: Dict,
                                     industry_summary: Dict, industry_insights: Dict,
                                     analysis_metadata: Dict = None) -> int:
        """Store industry context analysis results"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO industry_context_results 
                (file_id, industry_data, industry_summary, industry_insights, analysis_metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                file_id,
                json.dumps(industry_data),
                json.dumps(industry_summary),
                json.dumps(industry_insights),
                json.dumps(analysis_metadata) if analysis_metadata else None
            ))
            
            result_id = cursor.lastrowid
            conn.commit()
            self.logger.info(f"Stored industry context results with result_id: {result_id}")
            return result_id
            
        except Exception as e:
            self.logger.error(f"Error storing industry context results: {e}")
            raise
    
    def get_latest_results_by_filename(self, filename: str) -> Dict[str, Any]:
        """Get latest analysis results for a specific filename"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Get file metadata
            cursor.execute("""
                SELECT * FROM file_metadata 
                WHERE filename = ? 
                ORDER BY last_processed DESC 
                LIMIT 1
            """, (filename,))
            
            file_record = cursor.fetchone()
            if not file_record:
                return None
            
            file_id = file_record['file_id']
            results = {'file_metadata': dict(file_record)}
            
            # Get cash flow results
            cursor.execute("""
                SELECT * FROM cash_flow_results 
                WHERE file_id = ? 
                ORDER BY analysis_timestamp DESC 
                LIMIT 1
            """, (file_id,))
            
            cash_flow_record = cursor.fetchone()
            if cash_flow_record:
                results['cash_flow'] = {
                    'result_id': cash_flow_record['result_id'],
                    'cash_flow_data': json.loads(cash_flow_record['cash_flow_data']),
                    'summary_data': json.loads(cash_flow_record['summary_data']),
                    'categories_data': json.loads(cash_flow_record['categories_data']),
                    'trends_data': json.loads(cash_flow_record['trends_data']),
                    'analysis_metadata': json.loads(cash_flow_record['analysis_metadata']) if cash_flow_record['analysis_metadata'] else None
                }
            
            # Get vendor extraction results
            cursor.execute("""
                SELECT * FROM vendor_extraction_results 
                WHERE file_id = ? 
                ORDER BY analysis_timestamp DESC 
                LIMIT 1
            """, (file_id,))
            
            vendor_record = cursor.fetchone()
            if vendor_record:
                results['vendor_extraction'] = {
                    'result_id': vendor_record['result_id'],
                    'vendor_data': json.loads(vendor_record['vendor_data']),
                    'vendor_summary': json.loads(vendor_record['vendor_summary']),
                    'vendor_categories': json.loads(vendor_record['vendor_categories']),
                    'vendor_insights': json.loads(vendor_record['vendor_insights']),
                    'analysis_metadata': json.loads(vendor_record['analysis_metadata']) if vendor_record['analysis_metadata'] else None
                }
            
            # Get revenue analysis results
            cursor.execute("""
                SELECT * FROM revenue_analysis_results 
                WHERE file_id = ? 
                ORDER BY analysis_timestamp DESC 
                LIMIT 1
            """, (file_id,))
            
            revenue_record = cursor.fetchone()
            if revenue_record:
                results['revenue_analysis'] = {
                    'result_id': revenue_record['result_id'],
                    'revenue_data': json.loads(revenue_record['revenue_data']),
                    'revenue_summary': json.loads(revenue_record['revenue_summary']),
                    'revenue_trends': json.loads(revenue_record['revenue_trends']),
                    'revenue_insights': json.loads(revenue_record['revenue_insights']),
                    'analysis_metadata': json.loads(revenue_record['analysis_metadata']) if revenue_record['analysis_metadata'] else None
                }
            
            # Get industry context results
            cursor.execute("""
                SELECT * FROM industry_context_results 
                WHERE file_id = ? 
                ORDER BY analysis_timestamp DESC 
                LIMIT 1
            """, (file_id,))
            
            industry_record = cursor.fetchone()
            if industry_record:
                results['industry_context'] = {
                    'result_id': industry_record['result_id'],
                    'industry_data': json.loads(industry_record['industry_data']),
                    'industry_summary': json.loads(industry_record['industry_summary']),
                    'industry_insights': json.loads(industry_record['industry_insights']),
                    'analysis_metadata': json.loads(industry_record['analysis_metadata']) if industry_record['analysis_metadata'] else None
                }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error retrieving results by filename: {e}")
            raise
    
    def get_all_analysis_results(self) -> List[Dict[str, Any]]:
        """Get all analysis results from database"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    fm.*,
                    cfr.analysis_timestamp as cash_flow_timestamp,
                    ver.analysis_timestamp as vendor_timestamp,
                    rar.analysis_timestamp as revenue_timestamp,
                    icr.analysis_timestamp as industry_timestamp
                FROM file_metadata fm
                LEFT JOIN cash_flow_results cfr ON fm.file_id = cfr.file_id
                LEFT JOIN vendor_extraction_results ver ON fm.file_id = ver.file_id
                LEFT JOIN revenue_analysis_results rar ON fm.file_id = rar.file_id
                LEFT JOIN industry_context_results icr ON fm.file_id = icr.file_id
                ORDER BY fm.last_processed DESC
            """)
            
            results = []
            for row in cursor.fetchall():
                results.append(dict(row))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error retrieving all results: {e}")
            raise
    
    def delete_file_results(self, file_id: int) -> bool:
        """Delete all results for a specific file"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Delete from all result tables (cascade will handle this)
            cursor.execute("DELETE FROM file_metadata WHERE file_id = ?", (file_id,))
            
            conn.commit()
            self.logger.info(f"Deleted all results for file_id: {file_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting file results: {e}")
            return False
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.close_connection()

# Test function to verify database functionality
def test_database():
    """Test database functionality"""
    try:
        print("🧪 Testing Database Manager...")
        
        # Initialize database manager
        db_manager = DatabaseManager()
        print("✅ Database manager initialized successfully")
        
        # Test table creation
        print("✅ Database tables created/verified")
        
        # Test file metadata storage
        test_filename = "test_file.xlsx"
        test_file_path = "data/test_file.xlsx"
        
        # Create a test file
        with open(test_file_path, 'w') as f:
            f.write("test content")
        
        file_id = db_manager.store_file_metadata(test_filename, test_file_path, "steel_industry")
        print(f"✅ File metadata stored with file_id: {file_id}")
        
        # Test storing analysis results
        test_cash_flow = {"test": "data"}
        test_summary = {"summary": "test"}
        test_categories = {"categories": "test"}
        test_trends = {"trends": "test"}
        
        result_id = db_manager.store_cash_flow_results(
            file_id, test_cash_flow, test_summary, test_categories, test_trends
        )
        print(f"✅ Cash flow results stored with result_id: {result_id}")
        
        # Test retrieval
        results = db_manager.get_latest_results_by_filename(test_filename)
        print(f"✅ Results retrieved successfully: {len(results)} result types")
        
        # Cleanup test file
        os.remove(test_file_path)
        print("✅ Test file cleaned up")
        
        print("🎉 All database tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

if __name__ == "__main__":
    test_database()
