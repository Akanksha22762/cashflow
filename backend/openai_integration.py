"""
OpenAI Integration Module
Replaces OpenAI with OpenAI API for AI-powered analysis
"""

import os
import json
import time
import logging
import re
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
        
        # Default model for all OpenAI calls (use GPT-3.5-turbo for lower cost testing)
        self.default_model = "gpt-3.5-turbo"
        self.available_models = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"]
        
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
    
    def categorize_transactions(
        self, 
        descriptions: List[str], 
        amounts: List[float] = None,
        inward_amounts: List[float] = None,
        outward_amounts: List[float] = None,
        balances: List[float] = None,
        model: str = None
    ) -> List[str]:
        """
        Categorize transactions using OpenAI - BATCH MODE for consistency and speed
        
        Args:
            descriptions: List of transaction descriptions
            amounts: Optional list of net amounts (inward - outward)
            inward_amounts: Optional list of inward/credit amounts
            outward_amounts: Optional list of outward/debit amounts
            balances: Optional list of closing balances
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
        
        # Create numbered list of all transactions with full details
        transactions_list = []
        for i, desc in enumerate(descriptions):
            transaction_info = f"{i+1}. Description: {desc}"
            
            # Add amount information if available
            if inward_amounts and i < len(inward_amounts) and inward_amounts[i] > 0:
                transaction_info += f" | Inward Amount: {inward_amounts[i]:.2f}"
            if outward_amounts and i < len(outward_amounts) and outward_amounts[i] > 0:
                transaction_info += f" | Outward Amount: {outward_amounts[i]:.2f}"
            elif amounts and i < len(amounts):
                transaction_info += f" | Net Amount: {amounts[i]:.2f}"
            
            # Add balance if available
            if balances and i < len(balances) and balances[i] != 0:
                transaction_info += f" | Closing Balance: {balances[i]:.2f}"
            
            transactions_list.append(transaction_info)
        
        transactions_text = "\n".join(transactions_list)
        
        # Single prompt with ALL transactions
        batch_prompt = f"""You are a financial analyst specializing in cash-flow statement classification. You MUST strictly follow the cash flow classification rules defined in:

- IAS 7 (International Accounting Standard 7 - Statement of Cash Flows) / IFRS
- Ind AS 7 (Indian Accounting Standard 7 - Statement of Cash Flows)
- GAAP (Generally Accepted Accounting Principles) cash flow rules

These standards define three categories of cash flows. Your task is to assign each financial line item to EXACTLY ONE category according to these standards:

1. OPERATING ACTIVITIES – Core business activities + working capital movements.
2. INVESTING ACTIVITIES – Acquisition or disposal of long-term assets and investments.
3. FINANCING ACTIVITIES – Raising or settling capital (equity/debt) and related cash flows.

Use GENERAL rules that work for ANY industry (healthcare, retail, manufacturing, IT, finance, etc.). 
Do NOT assume any specific business model.

IMPORTANT: Classify transactions strictly according to IAS 7 / Ind AS 7 / GAAP definitions below. Do not deviate from these standard accounting rules.

-------------------------------------------------------------------------------
OPERATING ACTIVITIES include:
- Cash inflow from providing goods or services
- Customer receipts, sales revenue, service revenue
- Payments to suppliers, vendors, contractors, employees
- Routine business expenses (rent, utilities, repairs, admin expenses)
- Collections and payments related to working capital
- Interest paid/received (IFRS/Ind AS) 
- Dividends received
- Taxes and statutory payments
- Purchase of consumables, inventory, supplies, raw materials
- Routine maintenance and repairs (not capital improvements)

INVESTING ACTIVITIES include:
- Purchase or sale of property, plant, equipment, or any long-term asset
- Capital expenditure (CAPEX) - purchases meant to be capitalized, not expensed
- Purchase of capital equipment
- Purchase or disposal of investments (equity, bonds, deposits)
- Acquisition or sale of subsidiaries or business units
- Loans given and loans collected (if entity acts as lender)

FINANCING ACTIVITIES include:
- Issuance or buyback of equity
- Raising or repaying borrowings (loans, bonds, credit facilities)
- Loan principal repayment (EMI principal portion)
- Loan interest payments
- Dividends paid to shareholders
- Loan disbursements, loan drawdowns, loan EMI payments

-------------------------------------------------------------------------------
If a description matches multiple possibilities, pick the MOST LIKELY based on:
- The economic intent
- Whether it affects long-term assets
- Whether it affects capital/borrowing structure

⚠️ CRITICAL INSTRUCTION - WHEN TO USE "More information needed":
If the description is too vague, generic, or missing critical details and you CANNOT confidently determine if it's Operating, Investing, or Financing, you MUST respond with:
"More information needed"

**Analyze each description carefully and use "More information needed" when:**
- The description shows signs of being incomplete, unclear, or a placeholder
- You cannot confidently determine the business purpose or economic intent
- The description lacks sufficient context to classify it accurately
- The transaction type is ambiguous even after considering all available information

**Use your judgment to identify unclear descriptions:**
- Descriptions with question marks, uncertainty markers, or placeholders indicate the data itself is incomplete
- Descriptions that are clearly placeholders or require manual verification
- Any description where you cannot confidently determine the business purpose

**You MUST use "More information needed" when:**
- Description is too generic (e.g., "TRANSFER", "PAYMENT", "DEBIT", "CREDIT")
- No details about what the transaction is for or who it's with
- Description lacks context to determine the business purpose
- Description could reasonably belong to multiple categories without more context
- Transaction type is ambiguous and context doesn't clarify it

**Examples of unclear descriptions that MUST be marked "More information needed":**
- "TRANSFER" (no details about what or to whom)
- "PAYMENT" (no details about what payment)
- "DEBIT" or "CREDIT" (no description at all)
- "BANK CHARGES" (could be operating expense or financing fee)
- "REMITTANCE" (unclear purpose)
- "NEFT" or "RTGS" (just transfer type, no purpose)
- "WITHDRAWAL" (unclear what it's for)
- Generic descriptions without any business context
- Descriptions with question marks or uncertainty markers (e.g., "misc exp??", "unknown txn??", "verify later??", "check entry??")
- Descriptions containing words like "unknown", "unclear", "verify", "check", "misc", "manual adj", "rev/pymnt" without clear context
- Any description that ends with "??" or contains "??" - these indicate uncertainty
- Descriptions that are placeholders or require manual review (e.g., "verify later", "check entry", "manual adjustment")

**Do NOT guess!** If you cannot determine the category with reasonable confidence, use "More information needed" so the user can clarify later.

-------------------------------------------------------------------------------
ANALYSIS INSTRUCTIONS:
When categorizing each transaction, consider ALL available information:
1. **Description**: The transaction description/narration
2. **Inward Amount**: Credit/deposit amounts (money coming in)
3. **Outward Amount**: Debit/payment amounts (money going out)
4. **Net Amount**: Inward - Outward (positive = inflow, negative = outflow)
5. **Closing Balance**: Account balance after the transaction

Use the amounts and balance context to better understand the transaction nature:
- Large inward amounts might indicate revenue, loans, or investments
- Large outward amounts might indicate expenses, loan repayments, or asset purchases
- Balance trends can help identify patterns (e.g., regular payments, one-time transactions)

-------------------------------------------------------------------------------
Now categorize the following transactions according to IAS 7 / Ind AS 7 / GAAP cash flow rules:

TRANSACTIONS WITH FULL DETAILS:
{transactions_text}

REMINDER: 
- Classify each transaction strictly according to IAS 7 / Ind AS 7 / GAAP definitions provided above
- Consider the description, amounts (inward/outward), and balance context when classifying
- ⚠️ IMPORTANT: If the description is unclear, generic, or you cannot confidently determine if it's Operating/Investing/Financing, you MUST use "More information needed"
- Do NOT default to "Operating Activities" for unclear transactions - use "More information needed" instead
- The user will provide clarification for "More information needed" transactions later

RESPONSE FORMAT (REQUIRED):
You MUST respond with exactly {len(descriptions)} lines, one per transaction, in this exact format:
1: Operating Activities
2: Investing Activities
3: Financing Activities
4: More information needed
(continue numbering for every transaction in order)

Each line must start with the transaction number, followed by a colon (:), followed by the category name.
Use exactly these category names: "Operating Activities", "Investing Activities", "Financing Activities", or "More information needed".

Answer with category per line only (no explanations, no extra text):
"""
        
        # Single API call for ALL transactions
        response = self.simple_openai(batch_prompt, model, max_tokens=len(descriptions) * 50)
        
        # Parse the batch response
        categories = []
        lines = response.strip().split('\n')
        
        # Create dict to store categories by index
        category_dict = {}
        
        # Try multiple patterns to handle different response formats
        patterns = [
            re.compile(r'^\s*(\d+)\s*[:\.\-\)]\s*(.+)$'),  # 1: Operating Activities, 1. Operating, 1) Operating
            re.compile(r'^\s*(\d+)\s+(\d+)\s*[:\.\-\)]\s*(.+)$'),  # 1 1: Operating Activities (if numbered twice)
            re.compile(r'^\s*(\d+)\s+(.+)$'),  # 1 Operating Activities (space separator)
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            match = None
            index_str = None
            category_text = None
            
            # Try each pattern
            for pattern in patterns:
                match = pattern.match(line)
                if match:
                    groups = match.groups()
                    if len(groups) == 2:
                        index_str, category_text = groups
                    elif len(groups) == 3:
                        # Skip middle number if present
                        index_str, _, category_text = groups
                    break
            
            if not match:
                # Log unparseable lines for debugging
                if any(word in line.upper() for word in ["OPERATING", "INVESTING", "FINANCING"]):
                    logger.debug(f"Unparseable response line: '{line}'")
                continue
            
            try:
                index = int(index_str.strip()) - 1  # Convert to 0-based
                category_text = category_text.strip()
                
                # Clean and validate category
                category_upper = category_text.upper()
                if "MORE INFORMATION" in category_upper or "UNCLEAR" in category_upper or "NEEDED" in category_upper:
                    category = "More information needed"
                elif "OPERATING" in category_upper:
                    category = "Operating Activities"
                elif "INVESTING" in category_upper:
                    category = "Investing Activities"
                elif "FINANCING" in category_upper or "FINANCE" in category_upper:
                    category = "Financing Activities"
                else:
                    continue  # Skip invalid lines
                
                if 0 <= index < len(descriptions):
                    category_dict[index] = category
                    logger.info(f"OpenAI classified #{index+1} '{descriptions[index]}' as {category}")
            except (ValueError, IndexError):
                continue
        
        # Log raw response if many categories are missing (for debugging)
        if len(category_dict) < len(descriptions) * 0.5:  # Less than 50% parsed
            logger.warning(f"⚠️ Low parsing success ({len(category_dict)}/{len(descriptions)}). Raw response preview:\n{response[:500]}")
        
        # Build final list with all categories
        for i in range(len(descriptions)):
            if i in category_dict:
                categories.append(category_dict[i])
            else:
                # If missing or unclear, default to "More information needed" (not Operating Activities)
                desc_preview = descriptions[i][:50] if i < len(descriptions) else "unknown"
                logger.warning(f"Missing category for transaction #{i+1} ('{desc_preview}...'), defaulting to 'More information needed'")
                categories.append("More information needed")
        
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

# Compatibility functions (to replace OpenAI functions)
def simple_openai(prompt: str, model: str = None, max_tokens: int = 100) -> str:
    """
    Simple function to call OpenAI (replaces simple_openai)
    
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
