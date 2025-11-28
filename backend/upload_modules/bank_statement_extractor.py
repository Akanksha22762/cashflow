"""
Bank Statement Extractor Module
================================
Extracts transactions from bank statements (PDF, Excel, CSV, text) using OpenAI.
Handles extraction of date, description, inward/outward amounts, and closing balance.
"""

import pandas as pd
import re
import os
from typing import Dict, List, Optional, Any, Tuple
from werkzeug.datastructures import FileStorage


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from PDF file.
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        Extracted text content
    """
    try:
        import PyPDF2
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
    except ImportError:
        try:
            import pdfplumber
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
            return text
        except ImportError:
            raise ImportError("Please install PyPDF2 or pdfplumber: pip install PyPDF2 pdfplumber")
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {e}")


def extract_text_from_file(file_path: str) -> str:
    """
    Extract text from text file.
    
    Args:
        file_path: Path to text file
        
    Returns:
        File content as string
    """
    try:
        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    return file.read()
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not decode file with any supported encoding")
    except Exception as e:
        raise ValueError(f"Failed to read text file: {e}")


def clean_amount(amount_str: str) -> float:
    """
    Clean amount string by removing commas, currency symbols, and formatting.
    
    Args:
        amount_str: Amount string (e.g., "₹1,23,456.78", "$1,234.56", "1,234.56")
        
    Returns:
        Cleaned numeric amount
    """
    if not amount_str or pd.isna(amount_str):
        return 0.0
    
    # Convert to string and strip whitespace
    amount_str = str(amount_str).strip()
    
    # Remove currency symbols (₹, $, €, £, etc.)
    amount_str = re.sub(r'[₹$€£¥]', '', amount_str)
    
    # Remove commas
    amount_str = amount_str.replace(',', '')
    
    # Remove parentheses (often used for negative amounts)
    if '(' in amount_str and ')' in amount_str:
        amount_str = amount_str.replace('(', '-').replace(')', '')
    
    # Extract numeric value (including decimals and negative signs)
    match = re.search(r'-?\d+\.?\d*', amount_str)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return 0.0
    
    return 0.0


def extract_transactions_with_openai(
    file_content: str,
    file_type: str,
    file_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Extract transactions from bank statement using OpenAI.
    
    Args:
        file_content: Text content of the file
        file_type: Type of file ('pdf', 'text', 'excel', 'csv')
        file_path: Optional file path for context
        
    Returns:
        List of transaction dictionaries
    """
    import sys
    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    from openai_integration import openai_integration
    
    if not openai_integration.is_available:
        raise RuntimeError("OpenAI API is not available. Please configure OPENAI_API_KEY.")
    
    # Prepare prompt for OpenAI
    prompt = f"""You are a financial data extraction expert. Extract all transactions from this bank statement.

REQUIREMENTS:
1. Extract EVERY transaction from the statement
2. For each transaction, identify:
   - Date (in YYYY-MM-DD format if possible, or preserve original format)
   - Description (full transaction description/narration)
   - Inward Amount (credit/deposit - positive number, or 0 if not applicable)
   - Outward Amount (debit/payment - positive number, or 0 if not applicable)
   - Closing Balance (balance after this transaction, if available)

3. Clean amounts: Remove commas, currency symbols (₹, $, €, etc.), and convert to plain numbers
4. If a transaction shows only one amount, determine if it's inward (credit) or outward (debit) based on context
5. If closing balance is not explicitly shown, calculate it if you can determine the pattern

IMPORTANT RULES:
- If description is unclear or too generic (e.g., "TRANSFER", "PAYMENT", "DEBIT", "CREDIT" without details), mark it as needing more information
- Extract dates in any format found (preserve original if not standard)
- Amounts should be positive numbers (use Inward Amount for credits, Outward Amount for debits)
- If balance information is missing, leave Closing Balance as empty or 0

BANK STATEMENT CONTENT:
{file_content[:50000]}  # Limit to first 50k characters to avoid token limits

OUTPUT FORMAT (JSON):
Return a JSON array of transactions, each with:
{{
  "date": "YYYY-MM-DD or original format",
  "description": "Full transaction description",
  "inward_amount": 0.0 or positive number,
  "outward_amount": 0.0 or positive number,
  "closing_balance": 0.0 or number if available,
  "needs_more_info": true/false
}}

Return ONLY valid JSON, no other text."""

    try:
        # Call OpenAI
        response = openai_integration.simple_openai(
            prompt,
            model="gpt-4o",  # Use GPT-4 for better extraction accuracy
            max_tokens=4000
        )
        
        # Parse JSON response
        import json
        # Try to extract JSON from response (might have markdown code blocks)
        response = response.strip()
        if response.startswith('```'):
            # Remove markdown code blocks
            response = re.sub(r'```json\s*', '', response)
            response = re.sub(r'```\s*', '', response)
        
        transactions = json.loads(response)
        
        if not isinstance(transactions, list):
            raise ValueError("OpenAI response is not a list of transactions")
        
        # Clean and validate transactions
        cleaned_transactions = []
        for tx in transactions:
            cleaned_tx = {
                'Date': str(tx.get('date', '')),
                'Description': str(tx.get('description', '')),
                'Inward_Amount': clean_amount(str(tx.get('inward_amount', 0))),
                'Outward_Amount': clean_amount(str(tx.get('outward_amount', 0))),
                'Closing_Balance': clean_amount(str(tx.get('closing_balance', 0))),
                'Needs_More_Info': bool(tx.get('needs_more_info', False))
            }
            
            # Calculate net amount (inward - outward)
            net_amount = cleaned_tx['Inward_Amount'] - cleaned_tx['Outward_Amount']
            cleaned_tx['Amount'] = net_amount
            
            cleaned_transactions.append(cleaned_tx)
        
        return cleaned_transactions
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse OpenAI response as JSON: {e}\nResponse: {response[:500]}")
    except Exception as e:
        raise RuntimeError(f"OpenAI extraction failed: {e}")


def extract_from_bank_statement(
    file_path: str,
    file_type: str
) -> pd.DataFrame:
    """
    Main function to extract transactions from bank statement.
    
    Args:
        file_path: Path to bank statement file
        file_type: Type of file ('pdf', 'text', 'excel', 'csv')
        
    Returns:
        DataFrame with extracted transactions
    """
    print(f"📄 Extracting transactions from {file_type.upper()} file: {file_path}")
    
    # Extract text content based on file type
    if file_type.lower() == 'pdf':
        content = extract_text_from_pdf(file_path)
    elif file_type.lower() == 'text':
        content = extract_text_from_file(file_path)
    else:
        # For Excel/CSV, we'll use standard pandas loading
        # But we can still use OpenAI to enhance extraction if needed
        raise ValueError(f"File type {file_type} should be handled by standard file loader. Use this extractor for PDF/text files.")
    
    if not content or len(content.strip()) < 50:
        raise ValueError("File appears to be empty or could not extract meaningful content")
    
    print(f"📊 Extracted {len(content)} characters from file")
    print(f"🤖 Using OpenAI to extract transactions...")
    
    # Extract transactions using OpenAI
    transactions = extract_transactions_with_openai(content, file_type, file_path)
    
    if not transactions:
        raise ValueError("No transactions extracted from file")
    
    print(f"✅ Extracted {len(transactions)} transactions")
    
    # Convert to DataFrame
    df = pd.DataFrame(transactions)
    
    # Ensure standard column names
    column_mapping = {
        'Date': 'Date',
        'Description': 'Description',
        'Inward_Amount': 'Inward_Amount',
        'Outward_Amount': 'Outward_Amount',
        'Closing_Balance': 'Closing_Balance',
        'Amount': 'Amount',
        'Needs_More_Info': 'Needs_More_Info'
    }
    
    # Rename columns if needed
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})
    
    return df

