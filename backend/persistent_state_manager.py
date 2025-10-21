"""
Persistent State Manager for Cashflow SAP Bank Analysis System
Handles automatic state saving and restoration across app restarts
"""

import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd

class PersistentStateManager:
    """
    Manages persistent state for the Flask application to survive restarts and network issues
    """
    
    def __init__(self, db_manager):
        """
        Initialize persistent state manager
        
        Args:
            db_manager: MySQL database manager instance
        """
        self.db_manager = db_manager
        self.current_session_id = None
        self.current_file_id = None
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for state management"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('cashflow_app.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def save_global_state(self, global_data: Dict[str, Any], user_session_id: str = None) -> bool:
        """
        Save global application state to database
        
        Args:
            global_data: Dictionary containing global variables like reconciliation_data, uploaded_bank_df, etc.
            user_session_id: Browser session ID for tracking
            
        Returns:
            bool: Success status
        """
        try:
            if not self.current_session_id:
                self.logger.warning("No active session ID for saving global state")
                return False
            
            # Prepare state data for storage
            state_data = {}
            
            # Store reconciliation_data if available
            if 'reconciliation_data' in global_data and global_data['reconciliation_data']:
                state_data['reconciliation_data'] = self._serialize_dataframes(global_data['reconciliation_data'])
            
            # Store uploaded dataframes
            if 'uploaded_bank_df' in global_data and global_data['uploaded_bank_df'] is not None:
                state_data['uploaded_bank_df'] = global_data['uploaded_bank_df'].to_dict('records')
                state_data['uploaded_bank_df_columns'] = list(global_data['uploaded_bank_df'].columns)
            
            if 'uploaded_sap_df' in global_data and global_data['uploaded_sap_df'] is not None:
                state_data['uploaded_sap_df'] = global_data['uploaded_sap_df'].to_dict('records')
                state_data['uploaded_sap_df_columns'] = list(global_data['uploaded_sap_df'].columns)
            
            # Store other important global variables
            for key in ['bank_count', 'sap_count', 'ai_categorized', 'processing_time']:
                if key in global_data:
                    state_data[key] = global_data[key]
            
            # Store in database
            self.db_manager.store_session_state(
                session_id=self.current_session_id,
                state_type='global_data',
                state_data=state_data,
                user_session_id=user_session_id
            )
            
            self.logger.info(f"SUCCESS: Global state saved for session {self.current_session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving global state: {e}")
            return False
    
    def save_ui_state(self, ui_data: Dict[str, Any], user_session_id: str = None) -> bool:
        """
        Save UI state to database (form data, user selections, etc.)
        
        Args:
            ui_data: Dictionary containing UI state data
            user_session_id: Browser session ID for tracking
            
        Returns:
            bool: Success status
        """
        try:
            if not self.current_session_id:
                self.logger.warning("No active session ID for saving UI state")
                return False
            
            # Store in database
            self.db_manager.store_session_state(
                session_id=self.current_session_id,
                state_type='ui_state',
                state_data=ui_data,
                user_session_id=user_session_id
            )
            
            self.logger.info(f"SUCCESS: UI state saved for session {self.current_session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving UI state: {e}")
            return False
    
    def save_analysis_results(self, analysis_data: Dict[str, Any], user_session_id: str = None) -> bool:
        """
        Save analysis results to database
        
        Args:
            analysis_data: Dictionary containing analysis results
            user_session_id: Browser session ID for tracking
            
        Returns:
            bool: Success status
        """
        try:
            if not self.current_session_id:
                self.logger.warning("No active session ID for saving analysis results")
                return False
            
            # Store in database
            self.db_manager.store_session_state(
                session_id=self.current_session_id,
                state_type='analysis_results',
                state_data=analysis_data,
                user_session_id=user_session_id
            )
            
            self.logger.info(f"SUCCESS: Analysis results saved for session {self.current_session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving analysis results: {e}")
            return False
    
    def restore_latest_session(self) -> Dict[str, Any]:
        """
        Restore the latest session state from database
        
        Returns:
            Dictionary containing restored session data
        """
        try:
            # Get latest session
            latest_state = self.db_manager.get_latest_session_state()
            
            if not latest_state:
                self.logger.info("No previous session found to restore")
                return {}
            
            session_id = latest_state['session_id']
            self.current_session_id = session_id
            self.current_file_id = latest_state.get('file_id')
            
            # Restore complete session data
            restored_data = self.db_manager.restore_session_data(session_id)
            
            # Process restored data
            processed_data = self._process_restored_data(restored_data)
            
            self.logger.info(f"SUCCESS: Restored session {session_id} with {len(processed_data)} state types")
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error restoring latest session: {e}")
            return {}
    
    def restore_specific_session(self, session_id: int) -> Dict[str, Any]:
        """
        Restore a specific session by ID
        
        Args:
            session_id: Session ID to restore
            
        Returns:
            Dictionary containing restored session data
        """
        try:
            self.current_session_id = session_id
            
            # Restore complete session data
            restored_data = self.db_manager.restore_session_data(session_id)
            
            # Process restored data
            processed_data = self._process_restored_data(restored_data)
            
            self.logger.info(f"SUCCESS: Restored specific session {session_id}")
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error restoring session {session_id}: {e}")
            return {}
    
    def get_available_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get list of available sessions for restoration
        
        Args:
            limit: Maximum number of sessions to return
            
        Returns:
            List of session dictionaries
        """
        try:
            sessions = self.db_manager.get_all_session_states(limit)
            
            # Format sessions for display
            formatted_sessions = []
            for session in sessions:
                formatted_sessions.append({
                    'session_id': session['session_id'],
                    'filename': session['filename'],
                    'data_source': session['data_source'],
                    'analysis_type': session['analysis_type'],
                    'transaction_count': session['transaction_count'],
                    'started_at': session['started_at'],
                    'last_updated': session['last_updated'],
                    'display_name': f"{session['filename']} - {session['transaction_count']} transactions ({session['started_at']})"
                })
            
            return formatted_sessions
            
        except Exception as e:
            self.logger.error(f"Error getting available sessions: {e}")
            return []
    
    def set_current_session(self, session_id: int, file_id: int = None):
        """
        Set the current active session for state saving
        
        Args:
            session_id: Current session ID
            file_id: Current file ID (optional)
        """
        self.current_session_id = session_id
        if file_id:
            self.current_file_id = file_id
        self.logger.info(f"Set current session to {session_id}")
    
    def _serialize_dataframes(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize pandas DataFrames in the data dictionary
        FIXED: Preserve original data during serialization, minimal NaN handling
        
        Args:
            data: Dictionary that may contain DataFrames
            
        Returns:
            Dictionary with serialized DataFrames
        """
        serialized = {}
        
        for key, value in data.items():
            if isinstance(value, pd.DataFrame):
                # FIXED: Minimal data processing to preserve original values
                df_copy = value.copy()
                
                # Only handle critical NaN issues for JSON serialization, preserve original data
                for col in df_copy.columns:
                    # Only convert NaN to None for JSON compatibility, don't change actual values
                    if df_copy[col].dtype == 'object':
                        # Keep original string values, only convert actual NaN to None
                        df_copy[col] = df_copy[col].where(pd.notna(df_copy[col]), None)
                    elif df_copy[col].dtype in ['float64', 'int64']:
                        # For numeric columns, keep NaN as None for JSON serialization
                        df_copy[col] = df_copy[col].where(pd.notna(df_copy[col]), None)
                
                print(f"💾 SERIALIZING DATAFRAME: {key} with {len(df_copy)} rows, {len(df_copy.columns)} columns")
                print(f"   📊 Sample data: {df_copy.head(1).to_dict('records') if not df_copy.empty else 'Empty'}")
                
                serialized[key] = {
                    'data': df_copy.to_dict('records'),
                    'columns': list(df_copy.columns),
                    'index': list(df_copy.index),
                    'is_dataframe': True,
                    'original_dtypes': {col: str(dtype) for col, dtype in df_copy.dtypes.items()}
                }
            elif isinstance(value, dict):
                serialized[key] = self._serialize_dataframes(value)
            else:
                serialized[key] = value
        
        return serialized
    
    def _deserialize_dataframes(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deserialize pandas DataFrames from the data dictionary
        FIXED: Preserve original data types and values during deserialization
        
        Args:
            data: Dictionary with serialized DataFrames
            
        Returns:
            Dictionary with restored DataFrames
        """
        deserialized = {}
        
        for key, value in data.items():
            if isinstance(value, dict) and value.get('is_dataframe', False):
                print(f"🔄 DESERIALIZING DATAFRAME: {key}")
                print(f"   📊 Data rows: {len(value.get('data', []))}")
                print(f"   📋 Columns: {value.get('columns', [])}")
                
                df = pd.DataFrame(value['data'])
                if 'columns' in value:
                    df.columns = value['columns']
                
                # FIXED: Restore original data types if available
                if 'original_dtypes' in value:
                    for col, dtype_str in value['original_dtypes'].items():
                        if col in df.columns:
                            try:
                                # Restore original dtype where possible
                                if 'object' in dtype_str:
                                    df[col] = df[col].astype('object')
                                elif 'float' in dtype_str:
                                    df[col] = pd.to_numeric(df[col], errors='coerce')
                                elif 'int' in dtype_str:
                                    df[col] = pd.to_numeric(df[col], errors='coerce', downcast='integer')
                            except Exception as e:
                                print(f"   ⚠️ Could not restore dtype for {col}: {e}")
                
                # Debug: Check for data integrity after deserialization
                if not df.empty:
                    print(f"   🔍 Sample restored data: {df.head(1).to_dict('records')}")
                    # Count actual None/NaN values vs original data
                    for col in df.columns:
                        none_count = df[col].isnull().sum()
                        if none_count > 0:
                            print(f"   ℹ️ Column '{col}' has {none_count} null values (preserved from original)")
                else:
                    print(f"   ❌ WARNING: Restored DataFrame is empty!")
                
                deserialized[key] = df
            elif isinstance(value, dict):
                deserialized[key] = self._deserialize_dataframes(value)
            else:
                deserialized[key] = value
        
        return deserialized
    
    def _process_restored_data(self, restored_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and format restored data for application use
        
        Args:
            restored_data: Raw restored data from database
            
        Returns:
            Processed data ready for application use
        """
        processed = {}
        
        # Process each state type
        for state_type, state_info in restored_data.items():
            if state_type == 'session_metadata' or state_type == 'transactions':
                processed[state_type] = state_info
            elif isinstance(state_info, dict) and 'data' in state_info:
                state_data = state_info['data']
                
                if state_type == 'global_data':
                    # Deserialize DataFrames in global data
                    processed['global_data'] = self._deserialize_dataframes(state_data)
                    
                    # Convert specific data back to DataFrames - FIXED: Use deserialized data instead
                    if 'uploaded_bank_df' in processed['global_data']:
                        processed['uploaded_bank_df'] = processed['global_data']['uploaded_bank_df']
                        print(f"✅ Restored uploaded_bank_df from global_data: {len(processed['uploaded_bank_df'])} rows")
                    elif 'uploaded_bank_df' in state_data:
                        # Fallback: create from raw data
                        processed['uploaded_bank_df'] = pd.DataFrame(state_data['uploaded_bank_df'])
                        print(f"✅ Restored uploaded_bank_df from raw data: {len(processed['uploaded_bank_df'])} rows")
                    
                    if 'uploaded_sap_df' in processed['global_data']:
                        processed['uploaded_sap_df'] = processed['global_data']['uploaded_sap_df']
                        print(f"✅ Restored uploaded_sap_df from global_data: {len(processed['uploaded_sap_df'])} rows")
                    elif 'uploaded_sap_df' in state_data:
                        # Fallback: create from raw data
                        processed['uploaded_sap_df'] = pd.DataFrame(state_data['uploaded_sap_df'])
                        print(f"✅ Restored uploaded_sap_df from raw data: {len(processed['uploaded_sap_df'])} rows")
                    
                else:
                    processed[state_type] = state_data
        
        return processed

def create_auto_save_decorator(state_manager: PersistentStateManager):
    """
    Create a decorator for automatic state saving after endpoint execution
    
    Args:
        state_manager: PersistentStateManager instance
        
    Returns:
        Decorator function
    """
    def auto_save_decorator(f):
        def wrapper(*args, **kwargs):
            # Execute the original function
            result = f(*args, **kwargs)
            
            # Auto-save global state after successful execution
            try:
                # Import here to avoid circular imports
                import app
                
                global_data = {
                    'reconciliation_data': getattr(app, 'reconciliation_data', None),
                    'uploaded_bank_df': getattr(app, 'uploaded_bank_df', None),
                    'uploaded_sap_df': getattr(app, 'uploaded_sap_df', None),
                    'bank_count': getattr(app, 'bank_count', 0),
                    'sap_count': getattr(app, 'sap_count', 0),
                    'ai_categorized': getattr(app, 'ai_categorized', 0),
                }
                
                state_manager.save_global_state(global_data)
                
            except Exception as e:
                print(f"Warning: Auto-save failed: {e}")
            
            return result
        return wrapper
    return auto_save_decorator
