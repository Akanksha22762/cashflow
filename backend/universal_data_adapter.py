"""
Universal Data Adapter for Cash Flow Analysis System
====================================================
This module provides automatic adaptation of any dataset format to work with the cash flow analysis system.
It handles column mapping, data type conversion, validation, and preprocessing.
"""

import pandas as pd
import numpy as np
import re
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Union, Optional, Tuple, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UniversalDataAdapter:
    """
    Universal Data Adapter that transforms any dataset to be compatible with the cash flow analysis system.
    """
    
    # Standard column names expected by the system
    STANDARD_COLUMNS = {
        'date': 'Date',
        'description': 'Description',
        'amount': 'Amount',
        'type': 'Type'
    }
    
    # Common column name patterns for detection
    COLUMN_PATTERNS = {
        'date': [
            r'date|dt|day|transaction.*date|posting.*date|value.*date|settlement.*date|entry.*date'
        ],
        'description': [
            r'^description$|^desc$|narration|particulars|details|text|memo|note|remarks|comment'
        ],
        'amount': [
            r'amount|amt|value|sum|total|debit|credit|dr|cr|transaction.*amount|payment'
        ],
        'type': [
            r'type|transaction.*type|entry.*type|category|classification|nature'
        ]
    }
    
    # Common transaction type mappings
    TRANSACTION_TYPE_MAPPINGS = {
        'inward': ['credit', 'cr', 'deposit', 'incoming', 'received', 'income', 'revenue', 'inflow', '+'],
        'outward': ['debit', 'dr', 'withdrawal', 'outgoing', 'paid', 'expense', 'payment', 'outflow', '-']
    }
    
    def __init__(self):
        """Initialize the data adapter."""
        self.column_mapping = {}
        self.dataset_profile = {}
        self.preprocessing_steps = []
    
    def adapt(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Main method to adapt any dataset to the standard format.
        
        Args:
            data: Input DataFrame to adapt
            
        Returns:
            Adapted DataFrame in the standard format
        """
        logger.info(f"Starting data adaptation for DataFrame with shape {data.shape}")
        logger.info(f"Original columns: {list(data.columns)}")
        logger.info(f"Sample data from first row:")
        for col in data.columns:
            logger.info(f"  {col}: {data[col].iloc[0] if len(data) > 0 else 'N/A'}")
        
        # Make a copy to avoid modifying the original
        adapted_data = data.copy()
        
        # Step 1: Analyze and profile the dataset
        self._profile_dataset(adapted_data)
        
        # Step 2: Map columns to standard format
        adapted_data = self._map_columns(adapted_data)
        
        # Step 3: Clean and validate data
        adapted_data = self._clean_data(adapted_data)
        
        # Step 4: Infer and standardize data types
        adapted_data = self._standardize_types(adapted_data)
        
        # Step 5: Add derived columns
        adapted_data = self._add_derived_columns(adapted_data)
        
        logger.info(f"Data adaptation complete. Output shape: {adapted_data.shape}")
        
        return adapted_data
    
    def _profile_dataset(self, data: pd.DataFrame) -> None:
        """
        Analyze and profile the dataset to understand its structure.
        
        Args:
            data: Input DataFrame to profile
        """
        self.dataset_profile = {
            'row_count': len(data),
            'column_count': len(data.columns),
            'column_types': {col: str(dtype) for col, dtype in data.dtypes.items()},
            'missing_values': {col: int(data[col].isna().sum()) for col in data.columns},
            'unique_values': {col: int(data[col].nunique()) for col in data.columns},
            'date_columns': [],
            'numeric_columns': [],
            'categorical_columns': [],
            'text_columns': []
        }
        
        # Identify column types
        for col in data.columns:
            # Check for date columns
            if 'date' in col.lower() or data[col].dtype == 'datetime64[ns]':
                self.dataset_profile['date_columns'].append(col)
            
            # Check for numeric columns
            elif pd.api.types.is_numeric_dtype(data[col]):
                self.dataset_profile['numeric_columns'].append(col)
            
            # Check for categorical columns
            elif data[col].nunique() < 20:
                self.dataset_profile['categorical_columns'].append(col)
            
            # Assume text columns
            else:
                self.dataset_profile['text_columns'].append(col)
        
        logger.info(f"Dataset profiled: {len(data)} rows, {len(data.columns)} columns")
    
    def _map_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Map columns from the input dataset to the standard column names.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with mapped column names
        """
        # Initialize column mapping
        self.column_mapping = {}
        
        # Try to find matches for each standard column
        for std_key, std_col in self.STANDARD_COLUMNS.items():
            # First check for exact matches
            for col in data.columns:
                if col.lower() == std_col.lower():
                    self.column_mapping[std_key] = col
                    break
            
            # If no exact match, try pattern matching
            if std_key not in self.column_mapping:
                for col in data.columns:
                    # Skip Transaction_ columns for description mapping
                    if std_key == 'description' and col.lower().startswith('transaction'):
                        continue
                    for pattern in self.COLUMN_PATTERNS[std_key]:
                        if re.search(pattern, col, re.IGNORECASE):
                            self.column_mapping[std_key] = col
                            break
                    if std_key in self.column_mapping:
                        break
            
            # If still no match, use heuristics
            if std_key not in self.column_mapping:
                if std_key == 'date' and self.dataset_profile['date_columns']:
                    self.column_mapping[std_key] = self.dataset_profile['date_columns'][0]
                elif std_key == 'amount' and self.dataset_profile['numeric_columns']:
                    # Use the numeric column with the highest variance as amount
                    numeric_cols = self.dataset_profile['numeric_columns']
                    if numeric_cols:
                        variances = {col: data[col].var() for col in numeric_cols if not data[col].isna().all()}
                        if variances:
                            self.column_mapping[std_key] = max(variances, key=variances.get)
        
        # Create a new DataFrame with standard column names
        result = pd.DataFrame()
        
        # Map the columns we found
        for std_key, std_col in self.STANDARD_COLUMNS.items():
            if std_key in self.column_mapping:
                result[std_col] = data[self.column_mapping[std_key]]
        
        # Copy over any columns we didn't map
        for col in data.columns:
            if col not in self.column_mapping.values():
                result[col] = data[col]
        
        logger.info(f"Column mapping complete: {self.column_mapping}")
        logger.info(f"Final result columns: {list(result.columns)}")
        logger.info(f"Sample final data from first row:")
        for col in result.columns:
            logger.info(f"  {col}: {result[col].iloc[0] if len(result) > 0 else 'N/A'}")
        
        return result
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate the data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        result = data.copy()
        
        # Handle missing values
        if 'Date' in result.columns:
            result['Date'] = pd.to_datetime(result['Date'], errors='coerce')
            # Fill missing dates with the median date
            if result['Date'].isna().any():
                median_date = result['Date'].dropna().median()
                result['Date'] = result['Date'].fillna(median_date)
        
        if 'Amount' in result.columns:
            # Convert amount strings to numeric
            if not pd.api.types.is_numeric_dtype(result['Amount']):
                # Convert to string first to handle any non-string types
                result['Amount'] = result['Amount'].astype(str)
                # Remove currency symbols and commas
                result['Amount'] = result['Amount'].str.replace(r'[^\d.\-]', '', regex=True)
                # Handle empty strings
                result['Amount'] = result['Amount'].replace('', '0')
                # Convert to numeric
                result['Amount'] = pd.to_numeric(result['Amount'], errors='coerce')
            
            # Fill missing amounts with the median amount
            if result['Amount'].isna().any():
                median_amount = result['Amount'].dropna().median()
                result['Amount'] = result['Amount'].fillna(median_amount)
        
        if 'Description' in result.columns:
            # Fill missing descriptions
            result['Description'] = result['Description'].fillna('Unknown Transaction')
        
        if 'Type' in result.columns:
            # Standardize transaction types
            result['Type'] = self._standardize_transaction_types(result)
        elif 'Amount' in result.columns:
            # Infer transaction type from amount
            result['Type'] = np.where(result['Amount'] >= 0, 'INWARD', 'OUTWARD')
        
        logger.info("Data cleaning complete")
        
        return result
    
    def _standardize_transaction_types(self, data: pd.DataFrame) -> pd.Series:
        """
        Standardize transaction types to INWARD/OUTWARD.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Series with standardized transaction types
        """
        if 'Type' not in data.columns:
            return pd.Series(['UNKNOWN'] * len(data))
        
        # Convert to string
        types = data['Type'].astype(str).str.lower()
        
        # Map to standard types
        standardized = types.copy()
        
        # Check for inward indicators
        for indicator in self.TRANSACTION_TYPE_MAPPINGS['inward']:
            mask = types.str.contains(indicator, na=False)
            standardized = standardized.mask(mask, 'INWARD')
        
        # Check for outward indicators
        for indicator in self.TRANSACTION_TYPE_MAPPINGS['outward']:
            mask = types.str.contains(indicator, na=False)
            standardized = standardized.mask(mask, 'OUTWARD')
        
        # If Amount column exists, use it as fallback
        if 'Amount' in data.columns:
            amount_based_type = np.where(data['Amount'] >= 0, 'INWARD', 'OUTWARD')
            # Apply amount-based type where standardized is still the original value
            mask = (standardized == types)
            standardized = standardized.mask(mask, [amount_based_type[i] for i in range(len(mask)) if mask.iloc[i]])
        
        # Convert to uppercase
        return standardized.str.upper()
    
    def _standardize_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize data types for the columns.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with standardized data types
        """
        result = data.copy()
        
        # Standardize Date column
        if 'Date' in result.columns:
            result['Date'] = pd.to_datetime(result['Date'], errors='coerce')
        
        # Standardize Amount column
        if 'Amount' in result.columns:
            result['Amount'] = pd.to_numeric(result['Amount'], errors='coerce')
        
        # Standardize Type column
        if 'Type' in result.columns:
            result['Type'] = result['Type'].astype(str)
        
        # Standardize Description column
        if 'Description' in result.columns:
            result['Description'] = result['Description'].astype(str)
        
        logger.info("Data type standardization complete")
        
        return result
    
    def _add_derived_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add useful derived columns to the dataset.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with additional derived columns
        """
        result = data.copy()
        
        # Add Year, Month, Day columns if Date exists
        if 'Date' in result.columns:
            result['Year'] = result['Date'].dt.year
            result['Month'] = result['Date'].dt.month
            result['Day'] = result['Date'].dt.day
            result['DayOfWeek'] = result['Date'].dt.dayofweek
            result['Quarter'] = result['Date'].dt.quarter
        
        # Add AbsAmount column
        if 'Amount' in result.columns:
            result['AbsAmount'] = result['Amount'].abs()
        
        logger.info("Added derived columns")
        
        return result
    
    @staticmethod
    def detect_file_type(file_path: str) -> str:
        """
        Detect the type of file based on extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File type as string ('csv', 'excel', 'unknown')
        """
        if file_path.lower().endswith('.csv'):
            return 'csv'
        elif file_path.lower().endswith(('.xlsx', '.xls')):
            return 'excel'
        else:
            return 'unknown'
    
    @classmethod
    def load_and_adapt(cls, file_path: str, test_limit: int = None) -> pd.DataFrame:
        """
        Load a file and adapt it to the standard format.
        
        Args:
            file_path: Path to the file to load
            test_limit: Deprecated parameter (kept for backward compatibility, no longer used)
            
        Returns:
            Adapted DataFrame
        """
        adapter = cls()
        
        # Detect file type
        file_type = cls.detect_file_type(file_path)
        
        # Load data
        if file_type == 'csv':
            try:
                # Try different encodings and separators
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    for sep in [',', ';', '\t', '|']:
                        try:
                            data = pd.read_csv(file_path, encoding=encoding, sep=sep)
                            if len(data.columns) > 1:
                                logger.info(f"CSV loaded successfully with encoding {encoding} and separator '{sep}'")
                                break
                        except Exception:
                            continue
                    if 'data' in locals() and len(data.columns) > 1:
                        break
                
                if 'data' not in locals() or len(data.columns) <= 1:
                    # Fallback to default
                    data = pd.read_csv(file_path)
            except Exception as e:
                logger.error(f"Failed to load CSV file: {str(e)}")
                raise
        elif file_type == 'excel':
            try:
                # Try to find header row (look for row containing "Date" and "Description" or "Inward Amount")
                df_sample = pd.read_excel(file_path, header=None, nrows=10)
                header_row = None
                for idx, row in df_sample.iterrows():
                    row_str = ' '.join([str(val).lower() for val in row.values if pd.notna(val)])
                    if ('date' in row_str or 'dt' in row_str) and ('description' in row_str or 'desc' in row_str or 'inward' in row_str):
                        header_row = idx
                        logger.info(f"Found header row at index {header_row}")
                        break
                
                if header_row is not None:
                    data = pd.read_excel(file_path, header=header_row)
                    # Clean column names (remove extra spaces, normalize)
                    data.columns = [str(col).strip() for col in data.columns]
                    # Remove any rows that are still headers or completely empty
                    first_col = data.columns[0]
                    data = data[data[first_col].notna()]
                    # Remove rows where first column looks like a header (all text, no date/numbers)
                    data = data[data[first_col].astype(str).str.contains(r'\d', na=False, regex=True)]
                else:
                    # No header found, use first row as header
                    logger.warning("No header row detected, using first row as header")
                data = pd.read_excel(file_path)
            except Exception as e:
                logger.error(f"Failed to load Excel file: {str(e)}")
                raise
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        
        # Adapt data
        adapted_data = adapter.adapt(data)
        
        logger.info(f"📊 Final dataset size: {len(adapted_data)} transactions")
        
        return adapted_data


# Example usage
if __name__ == "__main__":
    # Example usage
    try:
        # Test with a sample file
        sample_file = "sample_data.csv"
        adapter = UniversalDataAdapter()
        
        # Create sample data if running as standalone
        sample_data = pd.DataFrame({
            'TransactionDate': pd.date_range(start='2023-01-01', periods=10),
            'TransactionDetails': [f"Transaction {i}" for i in range(10)],
            'TransactionAmount': [1000 * (i - 5) for i in range(10)],
            'EntryType': ['Credit' if i % 2 == 0 else 'Debit' for i in range(10)]
        })
        
        # Adapt the data
        adapted_data = adapter.adapt(sample_data)
        
        print("Original columns:", sample_data.columns.tolist())
        print("Adapted columns:", adapted_data.columns.tolist())
        print("Column mapping:", adapter.column_mapping)
        print("\nSample of adapted data:")
        print(adapted_data.head())
        
    except Exception as e:
        logger.error(f"Error in example: {str(e)}")
