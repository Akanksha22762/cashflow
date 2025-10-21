"""
OpenAI Integration Module
Replaces Ollama with OpenAI API for AI-powered analysis
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any
import warnings
from dotenv import load_dotenv
from openai import OpenAI
import asyncio

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIIntegration:
    """
    OpenAI Integration for AI Enhancement
    Provides OpenAI API integration for text processing and analysis
    """
    
    def __init__(self, api_key: str = None):
        """Initialize OpenAI integration"""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        # Optional project support for project-scoped keys
        self.project = os.getenv('OPENAI_PROJECT') or os.getenv('OPENAI_PROJECT_ID')
        if not self.api_key:
            logger.error("❌ No OpenAI API key provided. Please set OPENAI_API_KEY in .env file")
            self.is_available = False
            self.client = None
        else:
            try:
                # Pass project if provided (supports sk-proj-* keys)
                if self.project:
                    self.client = OpenAI(api_key=self.api_key, project=self.project)
                else:
                    self.client = OpenAI(api_key=self.api_key)
                self.is_available = True
                logger.info("✅ OpenAI API initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize OpenAI: {e}")
                self.is_available = False
                self.client = None
        
        self.default_model = "gpt-4o-mini"  # Cost-effective model
        self.available_models = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
        
    def _check_availability(self):
        """Check if OpenAI is available"""
        if not self.client:
            return False
        try:
            # Test with a simple call - DETERMINISTIC
            response = self.client.chat.completions.create(
                model=self.default_model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5,
                temperature=0,  # ✅ DETERMINISTIC
                seed=42  # ✅ REPRODUCIBLE
            )
            logger.info("✅ OpenAI API is available")
            return True
        except Exception as e:
            logger.warning(f"⚠️ OpenAI not available: {e}")
            return False
    
    def simple_openai(self, prompt: str, model: str = None, max_tokens: int = 100) -> str:
        """
        Simple OpenAI API call for text processing
        
        Args:
            prompt: Input prompt for the model
            model: Model name to use (defaults to gpt-4o-mini)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
            
        Raises:
            RuntimeError: If OpenAI is not available or all attempts fail
        """
        if not self.is_available or not self.client:
            raise RuntimeError("OpenAI API is not available. Check your API key and connection.")
        
        model = model or self.default_model
        
        # Retry logic with exponential backoff
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                logger.info(f"🤖 OpenAI API call (attempt {attempt + 1}/{max_retries})")
                
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful financial analysis assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=0,  # ✅ DETERMINISTIC: Same input = Same output
                    seed=42  # ✅ REPRODUCIBLE: Consistent across runs
                )
                
                result = response.choices[0].message.content.strip()
                return result
                
            except Exception as e:
                logger.error(f"OpenAI API error (attempt {attempt + 1}): {e}")
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = min(5, 2 ** attempt)
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
        
        raise RuntimeError(f"OpenAI API failed after {max_retries} attempts. Last error: {last_error}")
    
    def enhance_descriptions(self, descriptions: List[str], model: str = None) -> List[str]:
        """
        Enhance transaction descriptions using OpenAI
        
        Args:
            descriptions: List of transaction descriptions
            model: Model to use for enhancement
            
        Returns:
            List of enhanced descriptions
            
        Raises:
            RuntimeError: If OpenAI is not available
        """
        if not self.is_available:
            raise RuntimeError("OpenAI API is not available. Check your API key and connection.")
        
        model = model or self.default_model
        enhanced_descriptions = []
        
        for desc in descriptions:
            prompt = f"""Enhance this transaction description to be more descriptive and clear:
Original: {desc}

Enhanced description (one line only):"""
            
            enhanced = self.simple_openai(prompt, model, max_tokens=50)
            enhanced_descriptions.append(enhanced)
        
        return enhanced_descriptions
    
    def categorize_transactions(self, descriptions: List[str], model: str = None) -> List[str]:
        """
        Categorize transactions using OpenAI - BATCH MODE for consistency and speed
        
        Args:
            descriptions: List of transaction descriptions
            model: Model to use for categorization
            
        Returns:
            List of categories
            
        Raises:
            RuntimeError: If OpenAI is not available or categorization fails
        """
        if not self.is_available:
            raise RuntimeError("OpenAI API is not available. Check your API key and connection.")
        
        model = model or self.default_model
        
        # ✅ BATCH MODE: Categorize ALL transactions in ONE API call for consistency
        logger.info(f"🚀 BATCH MODE: Categorizing {len(descriptions)} transactions in ONE API call...")
        
        # Create numbered list of all transactions
        transactions_text = "\n".join([f"{i+1}. {desc}" for i, desc in enumerate(descriptions)])
        
        # Single prompt with ALL transactions
        batch_prompt = f"""You are a financial analyst expert in cash flow statement categorization. Analyze ALL these transactions and categorize each into ONE of these three categories:

TRANSACTIONS:
{transactions_text}

CATEGORIZATION RULES:

1. OPERATING ACTIVITIES (Core business operations):
   - Revenue from sales of goods/services (energy sales, service income)
   - Regular operating expenses (salaries, wages, benefits, payroll)
   - Cost of goods sold (raw materials, inventory purchases, coal procurement)
   - Operating supplies and consumables (boiler additives, chemicals, lubricants)
   - Regular maintenance and repairs (not capital improvements)
   - Utilities, rent, insurance, professional services
   - Transmission charges, wheeling fees, grid services
   - Customer payments received, overdue invoice settlements
   - Day-to-day business operations that occur regularly

2. INVESTING ACTIVITIES (Long-term asset transactions):
   - Capital expenditures (CAPEX) on major equipment, machinery, infrastructure
   - Purchase/sale of property, plant, equipment (PPE) - must be capitalized
   - Equipment upgrades, major renovations, capacity expansions
   - Purchase/sale of investments, securities, bonds
   - Acquisition/disposal of subsidiaries or business units
   - Loans made to others or collections of loans
   - Major infrastructure projects, technology investments
   - Transactions that create or dispose of long-term assets

3. FINANCING ACTIVITIES (Capital structure changes):
   - Borrowing money (loans, credit lines, bonds issued)
   - Repayment of debt (principal and interest payments)
   - Equity transactions (issuance of shares, buybacks, equity infusions)
   - Dividend payments to shareholders
   - Capital contributions from owners
   - Any transaction that changes debt levels or equity

CRITICAL DISTINCTIONS:
- Coal procurement = Operating (inventory/raw materials)
- Salaries/wages = Operating (regular employee costs)
- Energy sales = Operating (core revenue)
- Transmission charges = Operating (regular operating expense)
- Capital expenditure on equipment = Investing (long-term asset)
- Loan interest/principal = Financing (debt servicing)
- Equipment upgrades (capitalized) = Investing (long-term asset)

FORMAT YOUR RESPONSE EXACTLY AS:
1: Operating Activities
2: Investing Activities
3: Financing Activities
(etc. for all transactions)

Your answer (one category per line with number):"""
        
        # Single API call for ALL transactions
        response = self.simple_openai(batch_prompt, model, max_tokens=len(descriptions) * 50)
        
        # Parse the batch response
        categories = []
        lines = response.strip().split('\n')
        
        # Create dict to store categories by index
        category_dict = {}
        
        for line in lines:
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    try:
                        index = int(parts[0].strip()) - 1  # Convert to 0-based
                        category_text = parts[1].strip()
                        
                        # Clean and validate category
                        category_upper = category_text.upper()
                        if "OPERATING" in category_upper:
                            category = "Operating Activities"
                        elif "INVESTING" in category_upper:
                            category = "Investing Activities"
                        elif "FINANCING" in category_upper:
                            category = "Financing Activities"
                        else:
                            continue  # Skip invalid lines
                        
                        if 0 <= index < len(descriptions):
                            category_dict[index] = category
                            logger.info(f"OpenAI classified #{index+1} '{descriptions[index][:30]}...' as {category}")
                    except (ValueError, IndexError):
                        continue
        
        # Build final list with all categories
        for i in range(len(descriptions)):
            if i in category_dict:
                categories.append(category_dict[i])
            else:
                # If missing, default to Operating Activities
                logger.warning(f"Missing category for transaction #{i+1}, defaulting to Operating Activities")
                categories.append("Operating Activities")
        
        logger.info(f"✅ BATCH categorization complete: {len(categories)}/{len(descriptions)} transactions categorized")
        
        return categories
    
    def analyze_patterns(self, data: List[Dict[str, Any]], model: str = None) -> Dict[str, Any]:
        """
        Analyze patterns in transaction data using OpenAI
        
        Args:
            data: List of transaction dictionaries
            model: Model to use for analysis
            
        Returns:
            Dictionary containing pattern analysis
            
        Raises:
            RuntimeError: If OpenAI is not available or analysis fails
        """
        if not self.is_available:
            raise RuntimeError("OpenAI API is not available. Check your API key and connection.")
        
        model = model or self.default_model
        
        # Prepare data summary for analysis
        total_transactions = len(data)
        total_amount = sum(float(item.get('amount', 0)) for item in data)
        avg_amount = total_amount / total_transactions if total_transactions > 0 else 0
        
        prompt = f"""Analyze these transaction patterns:
- Total transactions: {total_transactions}
- Total amount: ${total_amount:,.2f}
- Average amount: ${avg_amount:,.2f}

Provide insights about:
1. Revenue patterns
2. Seasonal trends
3. Risk factors
4. Recommendations

Analysis:"""
        
        analysis = self.simple_openai(prompt, model, max_tokens=300)
        
        return {
            "patterns": analysis,
            "confidence": 0.8,
            "total_transactions": total_transactions,
            "total_amount": total_amount,
            "avg_amount": avg_amount
        }
    
    def extract_vendors_for_transactions(self, descriptions: List[str], model: str = None) -> List[str]:
        """
        Extract specific vendor for each transaction using OpenAI
        
        Args:
            descriptions: List of transaction descriptions
            model: Model to use
            
        Returns:
            List of vendor names, one for each transaction
            
        Raises:
            RuntimeError: If OpenAI is not available or extraction fails
        """
        if not self.is_available:
            raise RuntimeError("OpenAI API is not available. Check your API key and connection.")
        
        model = model or self.default_model
        
        logger.info(f"🤖 Extracting vendors for {len(descriptions)} transactions using OpenAI...")
        
        # Process in batches for better performance
        batch_size = 10  # OpenAI can handle larger batches
        all_vendors = []
        
        for i in range(0, len(descriptions), batch_size):
            batch = descriptions[i:i + batch_size]
            batch_vendors = self._extract_vendors_batch(batch, model)
            all_vendors.extend(batch_vendors)
            
            logger.info(f"🔄 Processed batch {i//batch_size + 1}/{(len(descriptions) + batch_size - 1)//batch_size}")
        
        logger.info(f"✅ OpenAI vendor extraction completed: {len(all_vendors)} vendors assigned")
        return all_vendors
    
    def _extract_vendors_batch(self, descriptions: List[str], model: str) -> List[str]:
        """
        Extract vendors for a batch of descriptions
        
        Raises:
            RuntimeError: If vendor extraction fails
        """
        # Create prompt for vendor extraction
        descriptions_text = "\n".join([f"{i+1}. {desc}" for i, desc in enumerate(descriptions)])
        
        prompt = f"""Extract the vendor/company name from each transaction description. Return ONLY the company name, nothing else.

TRANSACTIONS:
{descriptions_text}

For each transaction, identify the vendor company name. If no specific company is mentioned, use "Other Services" for consistency.

Format your response as:
1: [Vendor Name]
2: [Vendor Name]
3: [Vendor Name]
etc.

Examples:
"Coal procurement from Tata Steel" → 1: Tata Steel
"Energy sale to Gujarat DISCOM" → 2: Gujarat DISCOM  
"Loan interest payment to Axis Bank" → 3: Axis Bank
"Salaries and benefits paid to staff" → 4: Other Services
"Capital expenditure on equipment" → 5: Other Services

Your response:"""

        response = self.simple_openai(prompt, model, max_tokens=300)
        
        # Parse response to extract vendors
        vendors = []
        lines = response.strip().split('\n')
        
        # Create a dictionary to store vendors by index
        vendor_dict = {}
        
        for line in lines:
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    index_str = parts[0].strip()
                    vendor = parts[1].strip()
                    
                    # Clean up vendor name
                    vendor = vendor.replace('"', '').replace("'", '').replace('[', '').replace(']', '')
                    vendor = vendor.split('(')[0].strip()
                    
                    try:
                        index = int(index_str) - 1
                        if 0 <= index < len(descriptions) and vendor and len(vendor) > 1:
                            vendor_dict[index] = vendor
                    except ValueError:
                        continue
        
        # Build the final vendor list
        for i in range(len(descriptions)):
            if i in vendor_dict:
                vendors.append(vendor_dict[i])
            else:
                vendors.append("Other Services")
        
        if len(vendors) != len(descriptions):
            raise RuntimeError(f"Vendor extraction mismatch: got {len(vendors)} vendors for {len(descriptions)} descriptions")
        
        return vendors[:len(descriptions)]
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of OpenAI integration"""
        return {
            "available": self.is_available,
            "api_configured": bool(self.api_key),
            "available_models": self.available_models,
            "default_model": self.default_model,
            "status": "healthy" if self.is_available else "unavailable"
        }

# Global instance for easy access
openai_integration = OpenAIIntegration()

# Compatibility functions (to replace Ollama functions)
def simple_openai(prompt: str, model: str = None, max_tokens: int = 100) -> str:
    """
    Simple function to call OpenAI (replaces simple_ollama)
    
    Args:
        prompt: Input prompt
        model: Model name
        max_tokens: Maximum tokens to generate
        
    Returns:
        Generated text
        
    Raises:
        RuntimeError: If OpenAI is not available or call fails
    """
    return openai_integration.simple_openai(prompt, model, max_tokens)

def enhance_descriptions_with_openai(descriptions: List[str]) -> List[str]:
    """
    Enhance descriptions using OpenAI
    
    Args:
        descriptions: List of descriptions to enhance
        
    Returns:
        List of enhanced descriptions
    """
    return openai_integration.enhance_descriptions(descriptions)

def categorize_with_openai(descriptions: List[str]) -> List[str]:
    """
    Categorize transactions using OpenAI
    
    Args:
        descriptions: List of descriptions to categorize
        
    Returns:
        List of categories
    """
    return openai_integration.categorize_transactions(descriptions)

def analyze_patterns_with_openai(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze patterns using OpenAI
    
    Args:
        data: Transaction data to analyze
        
    Returns:
        Pattern analysis results
    """
    return openai_integration.analyze_patterns(data)

def check_openai_availability():
    """Check if OpenAI is available and working"""
    return openai_integration._check_availability()

# No backward compatibility - OpenAI only

# Test function
def test_openai_integration():
    """Test the OpenAI integration"""
    print("Testing OpenAI integration...")
    
    # Test availability
    status = openai_integration.get_health_status()
    print(f"OpenAI status: {status}")
    
    # Test simple call
    if openai_integration.is_available:
        result = simple_openai("Hello, how are you?", max_tokens=20)
        print(f"Test response: {result}")
    else:
        print("OpenAI not available for testing")

if __name__ == "__main__":
    test_openai_integration()
