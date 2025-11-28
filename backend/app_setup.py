"""
App Setup - Shared Dependencies and Initialization
Centralized module for all shared dependencies, globals, and managers
"""
import os
import time
import logging
import pandas as pd
from typing import Dict, List, Optional, Any
import warnings
from datetime import datetime

# Suppress pandas warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Define base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===== DATA FOLDER =====
DATA_FOLDER = "data"

# ===== LOGGING SETUP =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===== CACHE SETUP =====
CACHE_TTL = 3600  # 1 hour cache TTL

class AICacheManager:
    """Manages AI response caching with TTL and batch processing"""
    
    def __init__(self):
        self.cache = {}
        self.last_cleanup = time.time()
    
    def get(self, key: str) -> Optional[str]:
        """Get cached response if not expired"""
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry['timestamp'] < CACHE_TTL:
                return entry['response']
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, response: str):
        """Cache a response with timestamp"""
        self.cache[key] = {
            'response': response,
            'timestamp': time.time()
        }
    
    def cleanup_expired(self):
        """Remove expired cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if current_time - entry['timestamp'] > CACHE_TTL
        ]
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

# Initialize cache manager
ai_cache_manager = AICacheManager()

# ===== PERFORMANCE MONITORING =====
class PerformanceMonitor:
    """Monitor system performance and provide health metrics"""
    
    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.start_time = time.time()
        self.processing_times = []
    
    def record_request(self, processing_time: float, success: bool = True):
        """Record a request and its processing time"""
        self.request_count += 1
        if not success:
            self.error_count += 1
        self.processing_times.append(processing_time)
        
        # Keep only last 1000 processing times
        if len(self.processing_times) > 1000:
            self.processing_times = self.processing_times[-1000:]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        uptime = time.time() - self.start_time
        avg_processing_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        error_rate = (self.error_count / self.request_count * 100) if self.request_count > 0 else 0
        
        return {
            'uptime_seconds': uptime,
            'total_requests': self.request_count,
            'error_count': self.error_count,
            'error_rate_percent': error_rate,
            'avg_processing_time_seconds': avg_processing_time,
            'cache_size': len(ai_cache_manager.cache),
            'cache_hit_rate': self._calculate_cache_hit_rate()
        }
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate (simplified)"""
        return 0.0

# Initialize performance monitor
performance_monitor = PerformanceMonitor()

# ===== MODULE IMPORTS AND INITIALIZATION =====
# OpenAI Integration
try:
    from openai_integration import simple_openai as simple_openai, OpenAIIntegration as OpenAISimpleIntegration
    OPENAI_AVAILABLE = True
    print("✅ OpenAI Integration loaded!")
    
    try:
        app_openai_integration = OpenAISimpleIntegration()
        print(f"✅ Global OpenAI integration initialized: {app_openai_integration.is_available}")
        
        if not app_openai_integration.is_available:
            print("\n" + "="*80)
            print("⚠️  OPENAI API KEY NOT FOUND OR INVALID")
            print("="*80)
            print("\n📋 TO FIX:")
            print("   1. Open the '.env' file in the backend directory")
            print("   2. Replace 'your_openai_api_key_here' with your actual OpenAI API key")
            print("   3. Get your API key from: https://platform.openai.com/api-keys")
            print("\n   Example: OPENAI_API_KEY=sk-...")
            print("\n⚠️  Note: Some features may not work without OpenAI API key.")
            print("="*80 + "\n")
            # Don't exit - let the app start but warn that OpenAI features won't work
            # The app will continue running, but AI features won't work
    except SystemExit:
        raise  # Re-raise SystemExit to allow intentional exits
    except Exception as e:
        print(f"⚠️ Failed to initialize global OpenAI integration: {e}")
        app_openai_integration = None
        # Don't exit - let the app start even if OpenAI fails
        # Some features won't work, but the app will run
        print("⚠️ App will start but AI features may not work")
        
except ImportError as e:
    OPENAI_AVAILABLE = False
    app_openai_integration = None
    print(f"⚠️  OpenAI Integration module not found: {e}")
    print("   Some AI features may not be available.")
    # Don't exit - allow app to start without OpenAI

# ML Libraries
try:
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    try:
        import xgboost as xgb
        XGBOOST_AVAILABLE = True
        print("✅ XGBoost loaded successfully!")
    except ImportError:
        XGBOOST_AVAILABLE = False
        print("❌ XGBoost not available.")
    
    ML_AVAILABLE = XGBOOST_AVAILABLE
except ImportError as e:
    ML_AVAILABLE = False
    print(f"⚠️ ML libraries not available: {e}")

# Universal Data Adapter
try:
    from universal_data_adapter import UniversalDataAdapter
    from data_adapter_integration import preprocess_for_analysis, load_and_preprocess_file
    DATA_ADAPTER_AVAILABLE = True
    print("✅ Universal Data Adapter loaded successfully!")
except ImportError as e:
    DATA_ADAPTER_AVAILABLE = False
    print(f"⚠️ Universal Data Adapter not available: {e}")

# MySQL Database Manager
try:
    from mysql_database_manager import MySQLDatabaseManager
    DATABASE_AVAILABLE = True
    print("✅ MySQL Database Manager loaded successfully!")
    db_manager = MySQLDatabaseManager(password="cashflow123")
    print("✅ MySQL Database connection established!")
    try:
        db_manager.ensure_schema_updates()
        db_manager.ensure_vendor_tables()
        print("✅ Database schema verified")
    except Exception as schema_error:
        print(f"⚠️ Database schema verification failed: {schema_error}")
except ImportError as e:
    DATABASE_AVAILABLE = False
    db_manager = None
    print(f"⚠️ MySQL Database Manager not available: {e}")
except Exception as e:
    DATABASE_AVAILABLE = False
    db_manager = None
    print(f"⚠️ MySQL Database connection failed: {e}")

# Persistent State Manager
try:
    from persistent_state_manager import PersistentStateManager, create_auto_save_decorator
    if DATABASE_AVAILABLE and db_manager:
        state_manager = PersistentStateManager(db_manager)
        auto_save = create_auto_save_decorator(state_manager)
        PERSISTENT_STATE_AVAILABLE = True
        print("✅ Persistent State Manager loaded successfully!")
    else:
        state_manager = None
        auto_save = lambda f: f
        PERSISTENT_STATE_AVAILABLE = False
        print("⚠️ Persistent State Manager requires database connection")
except ImportError as e:
    state_manager = None
    auto_save = lambda f: f
    PERSISTENT_STATE_AVAILABLE = False
    print(f"⚠️ Persistent State Manager not available: {e}")

# Report Storage - Using RenderS3Client from s3_functions.py
try:
    from services.report_storage import S3ReportStorageRender
    S3_API_BASE_URL = os.environ.get("S3_API_BASE_URL", "http://15.207.1.40:3000")
    report_storage = S3ReportStorageRender(
        api_base_url=S3_API_BASE_URL,
        db_manager=db_manager if DATABASE_AVAILABLE else None,
    )
    REPORT_STORAGE_AVAILABLE = True
    print(f"✅ Report storage (RenderS3Client) connected to {S3_API_BASE_URL}")
except Exception as storage_error:
    report_storage = None
    REPORT_STORAGE_AVAILABLE = False
    print(f"⚠️ Report storage unavailable: {storage_error}")
    import traceback
    traceback.print_exc()

# Analysis Storage Integration
try:
    from analysis_storage_integration import integrate_analysis_with_database, store_ui_interaction
    ANALYSIS_STORAGE_AVAILABLE = True
    print("✅ Analysis Storage Integration loaded successfully!")
except ImportError as e:
    ANALYSIS_STORAGE_AVAILABLE = False
    print(f"⚠️ Analysis Storage Integration not available: {e}")

# PDF Generation
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# ===== DYNAMIC TRENDS ANALYZER =====
# ✅ Extracted from CashflowDemo and imported
try:
    from analytics_modules.dynamic_trends_analyzer import DynamicTrendsAnalyzer
    dynamic_trends_analyzer = DynamicTrendsAnalyzer(openai_integration=app_openai_integration)
    print("✅ DynamicTrendsAnalyzer initialized successfully")
except Exception as e:
    print(f"⚠️ Failed to initialize DynamicTrendsAnalyzer: {e}")
    dynamic_trends_analyzer = None

# ===== GLOBAL VARIABLES =====
uploaded_bank_df = None
uploaded_sap_df = None
uploaded_data = {}
reconciliation_data = {}
bank_count = 0
sap_count = 0
ai_categorized = 0

# ===== HELPER FUNCTIONS =====
def get_unified_bank_data():
    """Get bank data from unified source - ONLY your uploaded data"""
    global uploaded_data
    try:
        if 'bank_df' in uploaded_data and uploaded_data['bank_df'] is not None:
            return uploaded_data['bank_df']
        else:
            print("⚠️ No bank data uploaded yet")
            return None
    except Exception as e:
        print(f"❌ Error getting unified bank data: {e}")
        return None

def get_unified_sap_data():
    """Get SAP data from unified source - ONLY your uploaded data"""
    global uploaded_data
    try:
        if 'sap_df' in uploaded_data and uploaded_data['sap_df'] is not None:
            return uploaded_data['sap_df']
        else:
            print("⚠️ No SAP data uploaded yet")
            return None
    except Exception as e:
        print(f"❌ Error getting unified SAP data: {e}")
        return None

def update_uploaded_data(bank_df, sap_df=None):
    """Update global uploaded data variables"""
    global uploaded_bank_df, uploaded_sap_df, uploaded_data
    uploaded_bank_df = bank_df
    uploaded_sap_df = sap_df
    if uploaded_data is None:
        uploaded_data = {}
    uploaded_data['bank_df'] = bank_df
    uploaded_data['sap_df'] = sap_df

def _get_active_bank_dataframe():
    """Return the most recent uploaded bank dataframe or raise ValueError."""
    global uploaded_bank_df, uploaded_data
    if uploaded_bank_df is not None and not uploaded_bank_df.empty:
        return uploaded_bank_df.copy()

    if uploaded_data and uploaded_data.get('bank_df') is not None:
        bank_df_cached = uploaded_data['bank_df']
        if isinstance(bank_df_cached, pd.DataFrame):
            return bank_df_cached.copy()
        if isinstance(bank_df_cached, list):
            try:
                return pd.DataFrame(bank_df_cached)
            except Exception:
                pass

    raise ValueError("No bank data available. Please upload a bank statement first.")

# Export all for use in routes
__all__ = [
    # Constants
    'DATA_FOLDER', 'CACHE_TTL', 'BASE_DIR',
    # Managers
    'ai_cache_manager', 'performance_monitor',
    # Integration flags
    'OPENAI_AVAILABLE', 'ML_AVAILABLE', 'DATABASE_AVAILABLE',
    'DATA_ADAPTER_AVAILABLE', 'PERSISTENT_STATE_AVAILABLE',
    'REPORT_STORAGE_AVAILABLE', 'ANALYSIS_STORAGE_AVAILABLE',
    'REPORTLAB_AVAILABLE',
    # Instances
    'app_openai_integration', 'db_manager', 'state_manager',
    'report_storage', 'dynamic_trends_analyzer',
    # Global variables
    'uploaded_bank_df', 'uploaded_sap_df', 'uploaded_data',
    'reconciliation_data', 'bank_count', 'sap_count', 'ai_categorized',
    # Helper functions
    'get_unified_bank_data', 'get_unified_sap_data',
    'update_uploaded_data', '_get_active_bank_dataframe',
    # Utilities
    'logger', 'simple_openai',
]

