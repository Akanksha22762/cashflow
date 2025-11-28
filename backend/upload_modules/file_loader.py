"""
File Loader Module
==================
Handles loading files using Universal Data Adapter or standard methods.
"""

import pandas as pd
import os
from typing import Optional
from werkzeug.datastructures import FileStorage


def load_file_with_adapter(temp_file_path: str, data_adapter_available: bool) -> Optional[pd.DataFrame]:
    """
    Load file using Universal Data Adapter if available, otherwise use standard loading.
    
    Args:
        temp_file_path: Path to saved file
        data_adapter_available: Whether Universal Data Adapter is available
        
    Returns:
        Loaded DataFrame or None if failed
    """
    if data_adapter_available:
        try:
            import sys
            import os
            # Add parent directory to path for imports
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from data_adapter_integration import load_and_preprocess_file, get_adaptation_report
            
            print(f"🔄 Using Universal Data Adapter for file")
            
            # Load and preprocess file
            df = load_and_preprocess_file(temp_file_path, test_limit=None)
            
            # Check if adapter properly mapped columns
            column_mapping = get_adaptation_report().get('column_mapping', {})
            print(f"🔍 Adapter mapped columns: {column_mapping}")
            
            # Check if we have "Unnamed" columns (indicates header not detected)
            has_unnamed_cols = any('unnamed' in str(col).lower() for col in df.columns)
            
            # If adapter didn't properly map columns or has unnamed columns, fall back to standard loader
            if not column_mapping or has_unnamed_cols:
                print(f"⚠️ Universal Data Adapter didn't properly detect headers (has unnamed columns: {has_unnamed_cols})")
                print(f"🔄 Falling back to standard file loading with header detection...")
                return None  # Return None to trigger standard loader
            
            print(f"✅ Universal Data Adapter successfully processed file")
            return df
            
        except Exception as e:
            print(f"⚠️ Universal Data Adapter failed: {str(e)}. Falling back to standard loading.")
            return None
    
    return None


def load_file_standard(primary_file: FileStorage, temp_file_path: str = None) -> Optional[pd.DataFrame]:
    """
    Load file using standard pandas methods (fallback).
    Supports PDF and text files using OpenAI extraction.
    
    Args:
        primary_file: FileStorage object
        temp_file_path: Path to saved file (required for PDF/text extraction)
        
    Returns:
        Loaded DataFrame or None if failed
    """
    try:
        filename = primary_file.filename.lower()
        
        # Handle PDF files
        if filename.endswith('.pdf'):
            if not temp_file_path:
                raise ValueError("temp_file_path required for PDF extraction")
            from .bank_statement_extractor import extract_from_bank_statement
            print(f"📄 Detected PDF file - using OpenAI extraction...")
            df = extract_from_bank_statement(temp_file_path, 'pdf')
            return df
        
        # Handle text files
        elif filename.endswith('.txt'):
            if not temp_file_path:
                raise ValueError("temp_file_path required for text extraction")
            from .bank_statement_extractor import extract_from_bank_statement
            print(f"📄 Detected text file - using OpenAI extraction...")
            df = extract_from_bank_statement(temp_file_path, 'text')
            return df
        
        # Handle CSV files
        elif filename.endswith('.csv'):
            # Try different encodings and separators for CSV
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                for sep in [',', ';', '\t', '|']:
                    try:
                        primary_file.seek(0)
                        df = pd.read_csv(primary_file, encoding=encoding, sep=sep)
                        if len(df.columns) > 1 and len(df) > 0:
                            print(f"📊 CSV read successfully: {encoding}, separator: '{sep}'")
                            return df
                    except:
                        continue
        else:
            # Excel file - try to find header row
            primary_file.seek(0)
            # Try reading first 10 rows to find header
            df_sample = pd.read_excel(primary_file, header=None, nrows=10)
            primary_file.seek(0)
            
            # Find header row (look for row containing "Date" and "Description" or "Inward Amount")
            header_row = None
            for idx, row in df_sample.iterrows():
                row_str = ' '.join([str(val).lower() for val in row.values if pd.notna(val)])
                if ('date' in row_str or 'dt' in row_str) and ('description' in row_str or 'desc' in row_str or 'inward' in row_str):
                    header_row = idx
                    print(f"📋 Found header row at index {header_row}")
                    break
            
            if header_row is not None:
                # Re-read with proper header
                primary_file.seek(0)
                df = pd.read_excel(primary_file, header=header_row)
                
                # Clean column names (remove extra spaces, normalize)
                df.columns = [str(col).strip() for col in df.columns]
                
                # Remove any rows that are still headers or completely empty
                # Check if first column (usually Date) has valid data
                first_col = df.columns[0]
                df = df[df[first_col].notna()]
                
                # Remove rows where first column looks like a header (all text, no date/numbers)
                # Keep rows that have dates or numbers
                df = df[df[first_col].astype(str).str.contains(r'\d', na=False, regex=True)]
                
                print(f"✅ Loaded {len(df)} data rows after header detection")
            else:
                # No header found, use first row as header
                print("⚠️ No header row detected, using first row as header")
                primary_file.seek(0)
                df = pd.read_excel(primary_file)
            
            return df
            
    except Exception as e:
        print(f"⚠️ Standard file loading failed: {e}")
    
    return None


def load_file(temp_file_path: str, primary_file: FileStorage, 
              data_adapter_available: bool) -> pd.DataFrame:
    """
    Main function to load file - tries adapter first, then standard methods.
    
    Args:
        temp_file_path: Path to saved file
        primary_file: FileStorage object (for fallback)
        data_adapter_available: Whether Universal Data Adapter is available
        
    Returns:
        Loaded DataFrame
    """
    # Try adapter first
    df = load_file_with_adapter(temp_file_path, data_adapter_available)
    
    # Fallback to standard loading if adapter failed or didn't detect headers properly
    if df is None:
        print(f"📄 Using standard file loading with header detection...")
        df = load_file_standard(primary_file, temp_file_path)
    
    if df is None:
        raise ValueError("Failed to load file with both adapter and standard methods")
    
    print(f"✅ File loaded successfully: {len(df)} rows, {len(df.columns)} columns")
    if len(df.columns) <= 10:
        print(f"📋 Columns: {list(df.columns)}")
    else:
        print(f"📋 Columns (first 10): {list(df.columns)[:10]}...")
    
    return df

