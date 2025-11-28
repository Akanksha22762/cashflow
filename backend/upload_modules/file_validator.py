"""
File Validator Module
=====================
Validates uploaded files before processing.
"""

import os
from typing import Tuple, Optional
from werkzeug.datastructures import FileStorage


def validate_uploaded_file(bank_file: Optional[FileStorage]) -> Tuple[bool, Optional[str]]:
    """
    Validate uploaded file.
    
    Args:
        bank_file: Uploaded file from request
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not bank_file:
        return False, 'Please upload a Bank Statement file'
    
    if not bank_file.filename:
        return False, 'Please upload a valid Bank Statement file'
    
    # Check file extension
    allowed_extensions = ['.xlsx', '.xls', '.csv', '.pdf', '.txt']
    file_ext = os.path.splitext(bank_file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        return False, f'Invalid file type. Allowed types: {", ".join(allowed_extensions)}'
    
    return True, None


def save_uploaded_file(bank_file: FileStorage, upload_folder: str = 'uploads', file_type: str = 'bank') -> str:
    """
    Save uploaded file to temporary location.
    
    Args:
        bank_file: Uploaded file
        upload_folder: Folder to save file
        file_type: Type of file (bank, sap, etc.)
        
    Returns:
        Path to saved file
    """
    # Create uploads folder if it doesn't exist
    os.makedirs(upload_folder, exist_ok=True)
    
    # Generate file path
    temp_file_path = os.path.join(upload_folder, f"{file_type.lower()}_{bank_file.filename}")
    
    # Save file
    bank_file.save(temp_file_path)
    
    return temp_file_path

