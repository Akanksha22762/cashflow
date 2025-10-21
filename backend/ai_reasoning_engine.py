"""
AI Reasoning Engine
Provides OpenAI-powered reasoning, insights, and recommendations for all operations
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIReasoningEngine:
    """
    AI-Powered Reasoning Engine for Cash Flow Analysis
    Generates dynamic insights, recommendations, and explains how AI is working
    """
    
    def __init__(self, api_key: str = None):
        """Initialize AI Reasoning Engine with OpenAI"""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.project = os.getenv('OPENAI_PROJECT') or os.getenv('OPENAI_PROJECT_ID')
        
        if not self.api_key:
            logger.error("❌ No OpenAI API key provided")
            self.is_available = False
            self.client = None
        else:
            try:
                if self.project:
                    self.client = OpenAI(api_key=self.api_key, project=self.project)
                else:
                    self.client = OpenAI(api_key=self.api_key)
                self.is_available = True
                logger.info("✅ AI Reasoning Engine initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize AI Reasoning Engine: {e}")
                self.is_available = False
                self.client = None
        
        self.default_model = "gpt-4o-mini"
        self.advanced_model = "gpt-4o"
    
    def _call_openai(self, prompt: str, model: str = None, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """
        Internal method to call OpenAI API
        
        Args:
            prompt: The prompt to send to OpenAI
            model: Model to use
            max_tokens: Maximum tokens to generate
            temperature: Creativity level (0-1, higher = more creative)
            
        Returns:
            Generated response text
        """
        if not self.is_available or not self.client:
            raise RuntimeError("OpenAI API is not available")
        
        model = model or self.default_model
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert financial analyst and AI reasoning specialist. Provide clear, actionable insights."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise RuntimeError(f"OpenAI API call failed: {e}")
    
    # ==================== TRANSACTION CATEGORIZATION REASONING ====================
    
    def explain_categorization_reasoning(self, transaction_desc: str, category: str, all_transactions: List[Dict] = None) -> Dict[str, Any]:
        """
        Generate detailed reasoning about why a transaction was categorized in a specific way
        
        Args:
            transaction_desc: The transaction description
            category: The assigned category
            all_transactions: Optional list of all transactions for context
            
        Returns:
            Dictionary with reasoning, business impact, and recommendations
        """
        if not self.is_available:
            return {"error": "AI Reasoning not available"}
        
        # Build context from all transactions if provided
        context = ""
        if all_transactions:
            total_amount = sum(float(t.get('amount', 0)) for t in all_transactions)
            context = f"\nTotal transactions: {len(all_transactions)}\nTotal amount: ₹{total_amount:,.2f}"
        
        prompt = f"""You are a financial expert explaining AI decision-making to business users.

TRANSACTION ANALYZED:
Description: "{transaction_desc}"
AI-Assigned Category: {category}
{context}

Provide a comprehensive explanation with the following structure:

1. **AI REASONING PROCESS:**
   - How the AI analyzed this transaction
   - What keywords/patterns triggered the classification
   - Why this category was chosen over others
   - Confidence level and rationale

2. **BUSINESS IMPACT:**
   - What this categorization means for cash flow statements
   - Impact on Operating/Investing/Financing activities
   - How this affects financial reporting

3. **VALIDATION CHECKS:**
   - Is this categorization likely correct?
   - What red flags or considerations exist?
   - Alternative interpretations if any

4. **RECOMMENDATIONS:**
   - Should any action be taken?
   - What to monitor related to this transaction
   - Best practices for similar transactions

Format as JSON with keys: reasoning_process, business_impact, validation, recommendations, confidence_score"""

        try:
            response = self._call_openai(prompt, max_tokens=800, temperature=0.3)
            
            # Try to parse as JSON, fallback to structured text
            try:
                # Clean up any JSON prefixes that AI might add
                cleaned_response = response.strip()
                
                # Remove various possible prefixes
                prefixes_to_remove = [
                    '""json ',
                    '"json ',
                    'json ',
                    '```json\n',
                    '```json',
                    '```\n',
                    '```'
                ]
                
                for prefix in prefixes_to_remove:
                    if cleaned_response.startswith(prefix):
                        cleaned_response = cleaned_response[len(prefix):]
                        break
                
                # Also remove any trailing backticks or newlines
                cleaned_response = cleaned_response.rstrip('`\n').strip()
                
                print(f"🔍 Original response: {response[:100]}...")
                print(f"🔍 Cleaned response: {cleaned_response[:100]}...")
                
                result = json.loads(cleaned_response)
                print(f"✅ Successfully parsed JSON with {len(result)} keys")
                
                # If the result has a reasoning_process that's still a string with markdown, parse it
                if 'reasoning_process' in result and isinstance(result['reasoning_process'], str):
                    if result['reasoning_process'].startswith('```json'):
                        try:
                            # Extract JSON from markdown code block
                            json_content = result['reasoning_process']
                            json_content = json_content.replace('```json', '').replace('```', '').strip()
                            parsed_reasoning = json.loads(json_content)
                            result.update(parsed_reasoning)  # Merge the parsed content into result
                            print(f"✅ Successfully parsed nested JSON from reasoning_process")
                        except Exception as nested_error:
                            print(f"⚠️ Failed to parse nested JSON: {nested_error}")
                            
            except Exception as parse_error:
                print(f"❌ JSON parsing failed: {parse_error}")
                print(f"❌ Failed to parse: {cleaned_response[:200]}...")
                result = {
                    "reasoning_process": response,
                    "confidence_score": 0.8
                }
            
            result["timestamp"] = datetime.now().isoformat()
            result["transaction"] = transaction_desc
            result["category"] = category
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate categorization reasoning: {e}")
            return {"error": str(e)}
    
    def explain_batch_categorization(self, transactions: List[Dict], categories: List[str]) -> Dict[str, Any]:
        """
        Explain how batch categorization works and provide insights on the entire dataset
        
        Args:
            transactions: List of transaction dictionaries
            categories: List of assigned categories
            
        Returns:
            Comprehensive analysis of the categorization process
        """
        if not self.is_available:
            return {"error": "AI Reasoning not available"}
        
        # Prepare summary statistics
        category_counts = {}
        for cat in categories:
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        total_amount = sum(float(t.get('amount', 0)) for t in transactions)
        
        # Sample transactions for analysis (first 5 from each category)
        sample_transactions = {}
        for i, txn in enumerate(transactions[:15]):
            cat = categories[i] if i < len(categories) else "Unknown"
            if cat not in sample_transactions:
                sample_transactions[cat] = []
            if len(sample_transactions[cat]) < 5:
                sample_transactions[cat].append(txn.get('description', 'N/A'))
        
        prompt = f"""You are explaining the AI batch categorization process to business stakeholders.

BATCH CATEGORIZATION SUMMARY:
- Total Transactions: {len(transactions)}
- Total Amount: ₹{total_amount:,.2f}
- Categories Distribution:
{json.dumps(category_counts, indent=2)}

SAMPLE TRANSACTIONS BY CATEGORY:
{json.dumps(sample_transactions, indent=2)}

Provide a comprehensive explanation covering:

1. **HOW THE AI WORKS:**
   - The batch processing methodology
   - How GPT-4 analyzes financial transactions
   - Pattern recognition techniques used
   - Why batch processing is more accurate than individual processing

2. **INTERNAL AI MECHANICS:**
   - Natural language understanding for financial terms
   - Context awareness across transactions
   - Consistency checks performed
   - Error handling and fallback logic

3. **BUSINESS INSIGHTS:**
   - Key patterns identified in this dataset
   - Category distribution analysis
   - Unusual or noteworthy transactions
   - Cash flow health indicators

4. **QUALITY METRICS:**
   - Expected accuracy level
   - Potential misclassifications to watch for
   - Data quality observations
   - Confidence in categorization

5. **ACTIONABLE RECOMMENDATIONS:**
   - What actions should be taken based on this data
   - Areas requiring human review
   - Opportunities for optimization
   - Risk mitigation strategies

Format as JSON with keys: ai_methodology, internal_mechanics, business_insights, quality_metrics, recommendations"""

        try:
            response = self._call_openai(prompt, model=self.advanced_model, max_tokens=1500, temperature=0.4)
            
            try:
                # Clean up any JSON prefixes that AI might add
                cleaned_response = response.strip()
                if cleaned_response.startswith('""json '):
                    cleaned_response = cleaned_response[7:]  # Remove '""json '
                elif cleaned_response.startswith('"json '):
                    cleaned_response = cleaned_response[6:]  # Remove '"json '
                elif cleaned_response.startswith('json '):
                    cleaned_response = cleaned_response[5:]  # Remove 'json '
                
                result = json.loads(cleaned_response)
            except:
                result = {
                    "ai_methodology": response,
                    "summary": {
                        "total_transactions": len(transactions),
                        "categories": category_counts
                    }
                }
            
            result["timestamp"] = datetime.now().isoformat()
            result["statistics"] = {
                "total_transactions": len(transactions),
                "total_amount": total_amount,
                "category_distribution": category_counts
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate batch reasoning: {e}")
            return {"error": str(e)}
    
    # ==================== VENDOR EXTRACTION REASONING ====================
    
    def explain_vendor_extraction(self, transaction_desc: str, extracted_vendor: str, all_vendors: List[str] = None) -> Dict[str, Any]:
        """
        Explain how AI extracted vendor information from transaction descriptions
        
        Args:
            transaction_desc: The transaction description
            extracted_vendor: The vendor name extracted by AI
            all_vendors: Optional list of all extracted vendors for context
            
        Returns:
            Detailed explanation of vendor extraction process
        """
        if not self.is_available:
            return {"error": "AI Reasoning not available"}
        
        vendor_context = ""
        if all_vendors:
            unique_vendors = len(set(all_vendors))
            vendor_context = f"\nTotal unique vendors identified: {unique_vendors}"
        
        prompt = f"""You are explaining AI-powered vendor extraction to business users.

VENDOR EXTRACTION ANALYSIS:
Transaction: "{transaction_desc}"
Extracted Vendor: "{extracted_vendor}"
{vendor_context}

Provide detailed explanation covering:

1. **AI EXTRACTION PROCESS:**
   - How the AI identified the vendor in this text
   - Named Entity Recognition (NER) techniques used
   - Pattern matching and context analysis
   - Why this specific name was extracted

2. **CONFIDENCE & VALIDATION:**
   - Confidence level in this extraction
   - Potential ambiguities or alternatives
   - Validation against known vendor databases
   - Quality of extraction

3. **BUSINESS VALUE:**
   - Why accurate vendor tracking matters
   - How this helps in vendor management
   - Spend analysis implications
   - Relationship management opportunities

4. **VENDOR INSIGHTS:**
   - What can be inferred about this vendor
   - Industry categorization
   - Payment patterns to monitor
   - Vendor risk considerations

5. **RECOMMENDATIONS:**
   - Should this vendor be validated?
   - What vendor due diligence is needed?
   - Optimization opportunities
   - Contract management suggestions

Format as JSON with keys: extraction_process, confidence_validation, business_value, vendor_insights, recommendations"""

        try:
            response = self._call_openai(prompt, max_tokens=900, temperature=0.3)
            
            try:
                # Clean up any JSON prefixes that AI might add
                cleaned_response = response.strip()
                if cleaned_response.startswith('""json '):
                    cleaned_response = cleaned_response[7:]  # Remove '""json '
                elif cleaned_response.startswith('"json '):
                    cleaned_response = cleaned_response[6:]  # Remove '"json '
                elif cleaned_response.startswith('json '):
                    cleaned_response = cleaned_response[5:]  # Remove 'json '
                
                result = json.loads(cleaned_response)
            except:
                result = {
                    "extraction_process": response,
                    "vendor": extracted_vendor
                }
            
            result["timestamp"] = datetime.now().isoformat()
            result["transaction"] = transaction_desc
            result["extracted_vendor"] = extracted_vendor
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate vendor extraction reasoning: {e}")
            return {"error": str(e)}
    
    def analyze_vendor_landscape(self, vendors: List[str], transactions: List[Dict]) -> Dict[str, Any]:
        """
        Analyze the entire vendor landscape and provide strategic insights
        
        Args:
            vendors: List of all extracted vendors
            transactions: List of all transactions with vendor information
            
        Returns:
            Comprehensive vendor landscape analysis
        """
        if not self.is_available:
            return {"error": "AI Reasoning not available"}
        
        # Calculate vendor statistics
        vendor_stats = {}
        for i, vendor in enumerate(vendors):
            if i < len(transactions):
                amount = float(transactions[i].get('amount', 0))
                if vendor not in vendor_stats:
                    vendor_stats[vendor] = {"count": 0, "total_amount": 0}
                vendor_stats[vendor]["count"] += 1
                vendor_stats[vendor]["total_amount"] += amount
        
        # Sort vendors by amount
        top_vendors = sorted(vendor_stats.items(), key=lambda x: x[1]["total_amount"], reverse=True)[:10]
        
        prompt = f"""You are providing strategic vendor management insights to executives.

VENDOR LANDSCAPE ANALYSIS:
Total Transactions: {len(transactions)}
Total Unique Vendors: {len(set(vendors))}

TOP 10 VENDORS BY SPEND:
{json.dumps([{"vendor": v[0], "transactions": v[1]["count"], "total_spend": f"₹{v[1]['total_amount']:,.2f}"} for v in top_vendors], indent=2)}

Provide strategic analysis covering:

1. **VENDOR CONCENTRATION ANALYSIS:**
   - Spend concentration risks
   - Dependency on key vendors
   - Diversification opportunities
   - Single point of failure risks

2. **VENDOR SEGMENTATION:**
   - Strategic vs operational vendors
   - High-value vs high-volume vendors
   - Critical vs non-critical relationships
   - Suggested vendor tiers

3. **FINANCIAL INSIGHTS:**
   - Payment pattern analysis
   - Cash flow implications
   - Working capital optimization
   - Payment terms opportunities

4. **RISK ASSESSMENT:**
   - Vendor concentration risks
   - Supply chain vulnerabilities
   - Financial exposure analysis
   - Contingency planning needs

5. **STRATEGIC RECOMMENDATIONS:**
   - Vendor consolidation opportunities
   - Negotiation priorities
   - Relationship management strategies
   - Cost optimization initiatives
   - Contract renegotiation targets

Format as JSON with keys: concentration_analysis, vendor_segmentation, financial_insights, risk_assessment, strategic_recommendations"""

        try:
            response = self._call_openai(prompt, model=self.advanced_model, max_tokens=1500, temperature=0.4)
            
            try:
                # Clean up any JSON prefixes that AI might add
                cleaned_response = response.strip()
                if cleaned_response.startswith('""json '):
                    cleaned_response = cleaned_response[7:]  # Remove '""json '
                elif cleaned_response.startswith('"json '):
                    cleaned_response = cleaned_response[6:]  # Remove '"json '
                elif cleaned_response.startswith('json '):
                    cleaned_response = cleaned_response[5:]  # Remove 'json '
                
                result = json.loads(cleaned_response)
            except:
                result = {
                    "concentration_analysis": response
                }
            
            result["timestamp"] = datetime.now().isoformat()
            result["statistics"] = {
                "total_vendors": len(set(vendors)),
                "total_transactions": len(transactions),
                "top_vendors": [{"vendor": v[0], "spend": v[1]["total_amount"]} for v in top_vendors[:5]]
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate vendor landscape analysis: {e}")
            return {"error": str(e)}
    
    # ==================== ANALYTICS & TREND REASONING ====================
    
    def analyze_trend_with_reasoning(self, trend_type: str, trends_data: Dict = None, analysis_summary: Dict = None, selected_filters: Dict = None) -> Dict[str, Any]:
        """
        Analyze any trend with AI-powered reasoning and insights
        
        Args:
            trend_type: Type of trend analysis (revenue, expenses, cash_flow, etc.)
            trends_data: Pre-computed trends data from the system
            analysis_summary: Summary of the trends analysis
            selected_filters: User-selected filters and parameters
            
        Returns:
            Comprehensive trend analysis with reasoning
        """
        if not self.is_available:
            return {"error": "AI Reasoning not available"}
        
        # Prepare trends summary from the existing data
        trends_summary = ""
        if trends_data:
            trends_summary = f"Available trends: {', '.join(trends_data.keys())}"
            for trend_name, trend_info in trends_data.items():
                if isinstance(trend_info, dict):
                    trend_summary = f"\n{trend_name}:"
                    for key, value in trend_info.items():
                        if key != '_summary':
                            trend_summary += f"\n  - {key}: {value}"
                    trends_summary += trend_summary
        
        # Get analysis summary context
        summary_context = ""
        if analysis_summary:
            summary_context = f"Analysis summary: {analysis_summary}"
        
        filter_context = ""
        if selected_filters:
            filter_context = f"\nUser-Selected Filters: {json.dumps(selected_filters)}"
        
        prompt = f"""You are a financial analytics expert providing deep insights to business executives.

TREND ANALYSIS REQUEST:
Analysis Type: {trend_type}

EXISTING TRENDS DATA:
{trends_summary}

ANALYSIS SUMMARY:
{summary_context}
{filter_context}

Provide comprehensive analysis covering:

1. **AI ANALYSIS METHODOLOGY:**
   - How the AI is processing this trend data
   - Statistical methods and algorithms used
   - Pattern recognition techniques applied
   - Time series analysis approach

2. **TREND IDENTIFICATION:**
   - Key trends identified in the data
   - Growth rates and trajectories
   - Seasonal patterns detected
   - Anomalies or outliers found
   - Cyclical patterns observed

3. **BUSINESS INSIGHTS:**
   - What do these trends reveal about business health?
   - Revenue growth or decline patterns
   - Cost structure analysis
   - Cash flow health indicators
   - Operational efficiency metrics
   - Profitability implications

4. **PREDICTIVE ANALYSIS:**
   - Short-term forecast (next 1-3 months)
   - Medium-term outlook (3-6 months)
   - Key drivers and assumptions
   - Risk factors affecting predictions
   - Confidence intervals

5. **ROOT CAUSE ANALYSIS:**
   - What's driving the observed trends?
   - External factors at play
   - Internal operational impacts
   - Market dynamics influence
   - Strategic decisions impact

6. **ACTIONABLE RECOMMENDATIONS:**
   - Immediate actions required
   - Strategic initiatives to consider
   - Risk mitigation strategies
   - Opportunity capitalization
   - KPIs to monitor
   - Decision-making priorities

7. **RISK & OPPORTUNITY ASSESSMENT:**
   - Downside risks identified
   - Upside opportunities available
   - Scenario planning suggestions
   - Contingency measures needed

Format as detailed JSON with all sections as keys"""

        try:
            response = self._call_openai(prompt, model=self.advanced_model, max_tokens=2000, temperature=0.5)
            
            try:
                # Clean up any JSON prefixes that AI might add
                cleaned_response = response.strip()
                
                # Remove various possible prefixes
                prefixes_to_remove = [
                    '""json ',
                    '"json ',
                    'json ',
                    '```json\n',
                    '```json',
                    '```\n',
                    '```'
                ]
                
                for prefix in prefixes_to_remove:
                    if cleaned_response.startswith(prefix):
                        cleaned_response = cleaned_response[len(prefix):]
                        break
                
                # Also remove any trailing backticks or newlines
                cleaned_response = cleaned_response.rstrip('`\n').strip()
                
                print(f"🔍 Original response: {response[:100]}...")
                print(f"🔍 Cleaned response: {cleaned_response[:100]}...")
                
                result = json.loads(cleaned_response)
                print(f"✅ Successfully parsed JSON with {len(result)} keys")
            except Exception as parse_error:
                print(f"❌ JSON parsing failed: {parse_error}")
                print(f"❌ Failed to parse: {cleaned_response[:200]}...")
                # If JSON parsing fails, create structured result
                result = {
                    "ai_analysis_methodology": response[:500],
                    "trend_identification": "See full analysis",
                    "business_insights": response[500:1000] if len(response) > 500 else response,
                    "recommendations": "See full analysis"
                }
            
            result["timestamp"] = datetime.now().isoformat()
            result["analysis_type"] = trend_type
            result["data_summary"] = {
                "trends_analyzed": list(trends_data.keys()) if trends_data else [],
                "analysis_summary": analysis_summary,
                "filters_applied": selected_filters
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate trend analysis: {e}")
            return {"error": str(e)}
    
    def generate_executive_summary(self, all_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate executive summary with strategic insights
        
        Args:
            all_data: All available data (transactions, vendors, categories, etc.)
            
        Returns:
            Executive-level summary with key insights and recommendations
        """
        if not self.is_available:
            return {"error": "AI Reasoning not available"}
        
        # Extract key metrics
        transactions = all_data.get('transactions', [])
        total_amount = sum(float(t.get('amount', 0)) for t in transactions)
        
        prompt = f"""You are providing an executive summary for C-level stakeholders.

DATA OVERVIEW:
Total Transactions: {len(transactions)}
Total Value: ₹{total_amount:,.2f}
Data Points Available: {', '.join(all_data.keys())}

Create a concise executive summary covering:

1. **KEY FINDINGS (Top 3-5 most important insights)**
2. **FINANCIAL HEALTH SCORE (0-100 with rationale)**
3. **CRITICAL ACTIONS REQUIRED (Top 3 priorities)**
4. **STRATEGIC OPPORTUNITIES (Top 3 opportunities)**
5. **RISK ALERTS (Top 3 risks to address)**
6. **PERFORMANCE METRICS (Key KPIs and their interpretation)**
7. **30-60-90 DAY ACTION PLAN**

Keep each section concise and action-oriented. Format as JSON."""

        try:
            response = self._call_openai(prompt, model=self.advanced_model, max_tokens=1500, temperature=0.4)
            
            try:
                # Clean up any JSON prefixes that AI might add
                cleaned_response = response.strip()
                if cleaned_response.startswith('""json '):
                    cleaned_response = cleaned_response[7:]  # Remove '""json '
                elif cleaned_response.startswith('"json '):
                    cleaned_response = cleaned_response[6:]  # Remove '"json '
                elif cleaned_response.startswith('json '):
                    cleaned_response = cleaned_response[5:]  # Remove 'json '
                
                result = json.loads(cleaned_response)
            except:
                result = {"executive_summary": response}
            
            result["timestamp"] = datetime.now().isoformat()
            result["generated_by"] = "AI Reasoning Engine"
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate executive summary: {e}")
            return {"error": str(e)}
    
    def explain_ai_capabilities(self) -> Dict[str, Any]:
        """
        Explain what the AI can do and how it works - meta-reasoning about the system itself
        
        Returns:
            Explanation of AI capabilities and methodologies
        """
        if not self.is_available:
            return {"error": "AI Reasoning not available"}
        
        prompt = """You are explaining your own AI capabilities to business users who want to understand how you analyze their financial data.

Provide a comprehensive but accessible explanation covering:

1. **WHAT I CAN DO:**
   - Financial transaction categorization
   - Vendor extraction and analysis
   - Trend analysis and forecasting
   - Pattern recognition
   - Anomaly detection
   - Business insights generation
   - Recommendation synthesis

2. **HOW I WORK:**
   - Natural Language Processing techniques
   - Financial domain knowledge
   - Pattern matching algorithms
   - Statistical analysis methods
   - Machine learning approaches
   - Context understanding

3. **MY ACCURACY & LIMITATIONS:**
   - Expected accuracy levels for different tasks
   - What I'm highly confident about
   - What requires human validation
   - Known limitations and edge cases
   - When to trust AI vs human judgment

4. **HOW TO GET THE BEST RESULTS:**
   - Data quality requirements
   - Optimal data formats
   - Context that improves accuracy
   - How to interpret my outputs
   - When to override my recommendations

5. **CONTINUOUS IMPROVEMENT:**
   - How I learn from feedback
   - Model updates and improvements
   - User feedback integration
   - Evolution over time

Format as user-friendly JSON that non-technical executives can understand."""

        try:
            response = self._call_openai(prompt, model=self.advanced_model, max_tokens=1500, temperature=0.5)
            
            try:
                # Clean up any JSON prefixes that AI might add
                cleaned_response = response.strip()
                if cleaned_response.startswith('""json '):
                    cleaned_response = cleaned_response[7:]  # Remove '""json '
                elif cleaned_response.startswith('"json '):
                    cleaned_response = cleaned_response[6:]  # Remove '"json '
                elif cleaned_response.startswith('json '):
                    cleaned_response = cleaned_response[5:]  # Remove 'json '
                
                result = json.loads(cleaned_response)
            except:
                result = {"capabilities_explanation": response}
            
            result["timestamp"] = datetime.now().isoformat()
            result["ai_model"] = self.advanced_model
            result["version"] = "1.0.0"
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate AI capabilities explanation: {e}")
            return {"error": str(e)}


# Global instance
ai_reasoning_engine = AIReasoningEngine()


# Convenience functions
def get_categorization_reasoning(transaction_desc: str, category: str, all_transactions: List[Dict] = None) -> Dict[str, Any]:
    """Get reasoning for transaction categorization"""
    return ai_reasoning_engine.explain_categorization_reasoning(transaction_desc, category, all_transactions)


def get_batch_categorization_insights(transactions: List[Dict], categories: List[str]) -> Dict[str, Any]:
    """Get insights on batch categorization process"""
    return ai_reasoning_engine.explain_batch_categorization(transactions, categories)


def get_vendor_extraction_reasoning(transaction_desc: str, extracted_vendor: str, all_vendors: List[str] = None) -> Dict[str, Any]:
    """Get reasoning for vendor extraction"""
    return ai_reasoning_engine.explain_vendor_extraction(transaction_desc, extracted_vendor, all_vendors)


def get_vendor_landscape_analysis(vendors: List[str], transactions: List[Dict]) -> Dict[str, Any]:
    """Get comprehensive vendor landscape analysis"""
    return ai_reasoning_engine.analyze_vendor_landscape(vendors, transactions)


def get_trend_analysis_with_reasoning(trend_type: str, trends_data: Dict = None, analysis_summary: Dict = None, filters: Dict = None) -> Dict[str, Any]:
    """Get trend analysis with AI reasoning"""
    return ai_reasoning_engine.analyze_trend_with_reasoning(trend_type, trends_data, analysis_summary, filters)


def get_executive_summary(all_data: Dict[str, Any]) -> Dict[str, Any]:
    """Get executive summary with strategic insights"""
    return ai_reasoning_engine.generate_executive_summary(all_data)


def explain_ai_system() -> Dict[str, Any]:
    """Get explanation of AI system capabilities"""
    return ai_reasoning_engine.explain_ai_capabilities()


# Test function
def test_ai_reasoning():
    """Test the AI Reasoning Engine"""
    print("Testing AI Reasoning Engine...")
    
    if ai_reasoning_engine.is_available:
        # Test categorization reasoning
        result = get_categorization_reasoning(
            "Coal procurement from Tata Steel",
            "Operating Activities"
        )
        print(f"\nCategorization Reasoning: {json.dumps(result, indent=2)}")
        
        # Test AI capabilities explanation
        capabilities = explain_ai_system()
        print(f"\nAI Capabilities: {json.dumps(capabilities, indent=2)}")
    else:
        print("AI Reasoning Engine not available")


if __name__ == "__main__":
    test_ai_reasoning()

