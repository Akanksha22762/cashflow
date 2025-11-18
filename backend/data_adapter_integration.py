"""
Data Adapter Integration Module
==============================
This module integrates the UniversalDataAdapter with the main application.
It provides functions to preprocess any dataset before passing it to the analysis system.
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from universal_data_adapter import UniversalDataAdapter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataAdapterIntegration:
    """
    Integration class for the Universal Data Adapter.
    Provides methods to preprocess data before analysis.
    """
    
    def __init__(self):
        """Initialize the data adapter integration."""
        self.adapter = UniversalDataAdapter()
        self.dataset_profiles = {}
        self.last_adaptation_stats = {}
    
    def preprocess_dataset(self, data: pd.DataFrame, dataset_name: str = "unknown") -> pd.DataFrame:
        """
        Preprocess a dataset using the universal adapter.
        
        Args:
            data: Input DataFrame
            dataset_name: Name identifier for the dataset
            
        Returns:
            Preprocessed DataFrame ready for analysis
        """
        logger.info(f"Preprocessing dataset '{dataset_name}' with shape {data.shape}")
        
        # Adapt the data
        start_time = pd.Timestamp.now()
        adapted_data = self.adapter.adapt(data)
        processing_time = (pd.Timestamp.now() - start_time).total_seconds()
        
        # Store adaptation statistics
        self.last_adaptation_stats = {
            'dataset_name': dataset_name,
            'original_shape': data.shape,
            'adapted_shape': adapted_data.shape,
            'processing_time_seconds': processing_time,
            'column_mapping': self.adapter.column_mapping,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Store dataset profile
        self.dataset_profiles[dataset_name] = self.adapter.dataset_profile
        
        logger.info(f"Dataset '{dataset_name}' preprocessed in {processing_time:.2f} seconds")
        
        return adapted_data
    
    def preprocess_file(self, file_path: str, test_limit: int = None) -> pd.DataFrame:
        """
        Load and preprocess a file using the universal adapter.
        
        Args:
            file_path: Path to the file
            test_limit: Limit number of transactions for testing (default: None for no limit)
            
        Returns:
            Preprocessed DataFrame ready for analysis
        """
        # Extract dataset name from file path
        dataset_name = os.path.basename(file_path)
        
        logger.info(f"Loading and preprocessing file: {file_path}")
        
        # Load and adapt the data (pass test_limit to avoid default 20 limit)
        adapted_data = UniversalDataAdapter.load_and_adapt(file_path, test_limit=test_limit)
        
        # Store dataset profile
        self.dataset_profiles[dataset_name] = self.adapter.dataset_profile
        
        return adapted_data
    
    def get_adaptation_report(self) -> Dict[str, Any]:
        """
        Get a report of the last adaptation.
        
        Returns:
            Dictionary with adaptation statistics
        """
        return self.last_adaptation_stats
    
    def get_dataset_profile(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get the profile of a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with dataset profile
        """
        return self.dataset_profiles.get(dataset_name, {})
    
    def validate_dataset_compatibility(self, data: pd.DataFrame) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validate if a dataset is compatible with the analysis system.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Tuple of (is_compatible, message, compatibility_details)
        """
        compatibility_details = {
            'missing_required_columns': [],
            'data_quality_issues': [],
            'recommendations': []
        }
        
        # Check for required columns
        required_columns = set(['Date', 'Amount'])
        available_columns = set(data.columns)
        missing_columns = required_columns - available_columns
        
        if missing_columns:
            compatibility_details['missing_required_columns'] = list(missing_columns)
            compatibility_details['recommendations'].append(
                f"Dataset is missing required columns: {', '.join(missing_columns)}. "
                "Consider mapping these columns using the universal adapter."
            )
        
        # Check data quality
        if 'Date' in data.columns:
            date_nulls = data['Date'].isna().sum()
            if date_nulls > 0:
                compatibility_details['data_quality_issues'].append(
                    f"Dataset contains {date_nulls} missing dates."
                )
                compatibility_details['recommendations'].append(
                    "Fill missing dates or filter out rows with missing dates."
                )
        
        if 'Amount' in data.columns:
            amount_nulls = data['Amount'].isna().sum()
            if amount_nulls > 0:
                compatibility_details['data_quality_issues'].append(
                    f"Dataset contains {amount_nulls} missing amounts."
                )
                compatibility_details['recommendations'].append(
                    "Fill missing amounts or filter out rows with missing amounts."
                )
        
        # Check for sufficient data
        if len(data) < 10:
            compatibility_details['data_quality_issues'].append(
                f"Dataset contains only {len(data)} rows, which may be insufficient for analysis."
            )
            compatibility_details['recommendations'].append(
                "Consider using a larger dataset for more reliable analysis."
            )
        
        # Determine overall compatibility
        is_compatible = (
            not compatibility_details['missing_required_columns'] and
            len(compatibility_details['data_quality_issues']) <= 2
        )
        
        message = "Dataset is compatible with the analysis system." if is_compatible else \
                  "Dataset requires adaptation before analysis."
        
        return is_compatible, message, compatibility_details


# Initialize the data adapter integration
data_adapter = DataAdapterIntegration()

def preprocess_for_analysis(data: pd.DataFrame, dataset_name: str = "unknown") -> pd.DataFrame:
    """
    Convenience function to preprocess data for analysis.
    
    Args:
        data: Input DataFrame
        dataset_name: Name identifier for the dataset
        
    Returns:
        Preprocessed DataFrame ready for analysis
    """
    return data_adapter.preprocess_dataset(data, dataset_name)

def load_and_preprocess_file(file_path: str, test_limit: int = None) -> pd.DataFrame:
    """
    Convenience function to load and preprocess a file.
    
    Args:
        file_path: Path to the file
        test_limit: Limit number of transactions for testing (default: None for no limit, set to a number to limit)
        
    Returns:
        Preprocessed DataFrame ready for analysis, limited to test_limit transactions if specified
    """
    # Pass test_limit to preprocess_file, which will pass it to UniversalDataAdapter.load_and_adapt
    result = data_adapter.preprocess_file(file_path, test_limit=test_limit)
    
    # Note: The limit is now applied inside UniversalDataAdapter.load_and_adapt, so we don't need to limit again here
    if result is not None:
        print(f"📊 Final dataset size: {len(result)} transactions")
    
    return result

def get_adaptation_report() -> Dict[str, Any]:
    """
    Get a report of the last adaptation.
    
    Returns:
        Dictionary with adaptation statistics
    """
    return data_adapter.get_adaptation_report()

def validate_dataset(data: pd.DataFrame) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validate if a dataset is compatible with the analysis system.
    
    Args:
        data: Input DataFrame
        
    Returns:
        Tuple of (is_compatible, message, compatibility_details)
    """
    return data_adapter.validate_dataset_compatibility(data)
