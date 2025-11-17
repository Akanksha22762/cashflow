"""
Simplified Script to Update app.py for Universal Industry Support
================================================================
This script updates the main app.py file to integrate the universal industry system
and remove hardcoded steel industry references while keeping steel as a supported sector.
"""

import re
import os

def update_app_py():
    """Update app.py to integrate universal industry system."""
    
    # Read the current app.py file
    with open('app.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add import for universal industry system
    import_pattern = r'# ===== UNIVERSAL DATA ADAPTER ====='
    import_replacement = '''# ===== UNIVERSAL INDUSTRY SYSTEM =====
try:
    from universal_industry_system import universal_industry_system
    UNIVERSAL_INDUSTRY_AVAILABLE = True
    print("✅ Universal Industry System loaded successfully!")
except ImportError as e:
    UNIVERSAL_INDUSTRY_AVAILABLE = False
    print(f"⚠️ Universal Industry System not available: {e}")

# ===== UNIVERSAL DATA ADAPTER ====='''
    
    content = re.sub(import_pattern, import_replacement, content)
    
    # Simple string replacements for steel industry references
    steel_replacements = [
        # Replace steel industry prompts
        ('Analyze financial trends for {trend_type} in the STEEL INDUSTRY based on this data:',
         'Analyze financial trends for {trend_type} based on this data:'),
        
        ('CONTEXT: This is a steel manufacturing company with transactions related to:',
         'CONTEXT: This is a business with transactions related to:'),
        
        ('- Steel production, sales, and distribution',
         '- Business operations, sales, and distribution'),
        
        ('STEEL INDUSTRY SPECIFIC ANALYSIS for {trend_type}:',
         'INDUSTRY-SPECIFIC ANALYSIS for {trend_type}:'),
        
        ('2. Risk assessment (low/medium/high) based on steel industry standards',
         '2. Risk assessment (low/medium/high) based on industry standards'),
        
        ('3. Key insights specific to steel manufacturing and operations',
         '3. Key insights specific to business operations and industry context'),
        
        ('4. Recommendations for steel industry optimization',
         '4. Recommendations for industry optimization'),
        
        ('Consider steel industry factors:',
         'Consider industry-specific factors:'),
        
        # Replace steel_industry_context with universal_industry_context
        ('steel_industry_context', 'universal_industry_context'),
        
        # Replace steel-specific context strings
        ("'steel_industry_context': 'Revenue analysis for steel manufacturing operations'",
         "'universal_industry_context': 'Revenue analysis for business operations'"),
        
        ("'steel_industry_context': 'Sales forecasting for steel products and services'",
         "'universal_industry_context': 'Sales forecasting for products and services'"),
        
        ("'steel_industry_context': 'Customer contract analysis for steel industry clients'",
         "'universal_industry_context': 'Customer contract analysis for business clients'"),
        
        ("'steel_industry_context': 'Pricing strategy analysis for steel products'",
         "'universal_industry_context': 'Pricing strategy analysis for products'"),
        
        ("'steel_industry_context': 'AR aging analysis for steel industry receivables'",
         "'universal_industry_context': 'AR aging analysis for business receivables'"),
        
        ("'steel_industry_context': 'Operating expense analysis for steel manufacturing'",
         "'universal_industry_context': 'Operating expense analysis for business operations'"),
        
        ("'steel_industry_context': 'AP analysis for steel industry suppliers'",
         "'universal_industry_context': 'AP analysis for business suppliers'"),
        
        ("'steel_industry_context': 'Inventory turnover analysis for steel products'",
         "'universal_industry_context': 'Inventory turnover analysis for business products'"),
        
        ("'steel_industry_context': 'Loan repayment analysis for steel industry financing'",
         "'universal_industry_context': 'Loan repayment analysis for business financing'"),
        
        ("'steel_industry_context': 'Tax obligation analysis for steel industry compliance'",
         "'universal_industry_context': 'Tax obligation analysis for business compliance'"),
        
        ("'steel_industry_context': 'Capital expenditure analysis for steel industry investments'",
         "'universal_industry_context': 'Capital expenditure analysis for business investments'"),
        
        ("'steel_industry_context': 'Equity and debt analysis for steel industry financing'",
         "'universal_industry_context': 'Equity and debt analysis for business financing'"),
        
        ("'steel_industry_context': 'Other income/expense analysis for steel industry operations'",
         "'universal_industry_context': 'Other income/expense analysis for business operations'"),
        
        ("'steel_industry_context': 'Cash flow type analysis for steel industry operations'",
         "'universal_industry_context': 'Cash flow type analysis for business operations'"),
        
        # Replace steel-specific analysis prompts
        ('Analyze revenue trends for a steel manufacturing company:',
         'Analyze revenue trends for this business:'),
        
        ('2. Steel industry market trends',
         '2. Industry market trends'),
        
        ('Analyze sales forecasting for a steel manufacturing company:',
         'Analyze sales forecasting for this business:'),
        
        ('Analyze customer contracts for a steel manufacturing company:',
         'Analyze customer contracts for this business:'),
        
        ('Analyze pricing models for a steel manufacturing company:',
         'Analyze pricing models for this business:'),
        
        ('Analyze accounts receivable aging for a steel manufacturing company:',
         'Analyze accounts receivable aging for this business:'),
        
        ('Analyze operating expenses for a steel manufacturing company:',
         'Analyze operating expenses for this business:'),
        
        ('Analyze accounts payable for a steel manufacturing company:',
         'Analyze accounts payable for this business:'),
        
        ('Analyze inventory turnover for a steel manufacturing company:',
         'Analyze inventory turnover for this business:'),
        
        ('Analyze loan repayments for a steel manufacturing company:',
         'Analyze loan repayments for this business:'),
        
        ('Analyze tax obligations for a steel manufacturing company:',
         'Analyze tax obligations for this business:'),
        
        ('Analyze capital expenditure for a steel manufacturing company:',
         'Analyze capital expenditure for this business:'),
        
        ('Analyze equity and debt inflows for a steel manufacturing company:',
         'Analyze equity and debt inflows for this business:'),
        
        ('Analyze other income and expenses for a steel manufacturing company:',
         'Analyze other income and expenses for this business:'),
        
        ('Analyze cash flow types for a steel manufacturing company:',
         'Analyze cash flow types for this business:'),
        
        # Replace steel-specific comments
        ('# ENHANCED FEATURE EXTRACTION FOR STEEL INDUSTRY',
         '# ENHANCED FEATURE EXTRACTION FOR INDUSTRY'),
        
        ('You are a financial analyst for a steel manufacturing company.',
         'You are a financial analyst for this business.'),
        
        ('Consider the category and typical business transactions for a steel plant.',
         'Consider the category and typical business transactions for this business.'),
        
        ('data_folder = \'steel_plant_datasets\'',
         'data_folder = \'business_datasets\''),
        
        ('chart_of_accounts_path = os.path.join(data_folder, \'steel_plant_sap_data.xlsx\')',
         'chart_of_accounts_path = os.path.join(data_folder, \'business_sap_data.xlsx\')'),
        
        ('description: \'Steel Plant Operations - Raw Materials\'',
         'description: \'Business Operations - Raw Materials\''),
        
        ('vendor: \'Steel Suppliers\'',
         'vendor: \'Business Suppliers\''),
        
        ('Regular supplier appears {count} times (expected for steel plant operations)',
         'Regular supplier appears {count} times (expected for business operations)'),
        
        ('Month-end batch transaction at {row[\'Hour\']}:00 (normal for steel plant)',
         'Month-end batch transaction at {row[\'Hour\']}:00 (normal for business operations)'),
        
        ('Weekend batch processing at {row[\'Hour\']}:00 (normal for steel plant operations)',
         'Weekend batch processing at {row[\'Hour\']}:00 (normal for business operations)'),
        
        ('# 6. SEASONAL ANOMALIES (Steel Plant Specific)',
         '# 6. SEASONAL ANOMALIES (Industry Specific)'),
        
        ('Steel coil production - blast furnace maintenance',
         'Production - equipment maintenance')
    ]
    
    # Apply all replacements
    for old_text, new_text in steel_replacements:
        content = content.replace(old_text, new_text)
    
    # Add industry detection logic
    industry_detection = '''
    # Auto-detect industry from uploaded data
    if UNIVERSAL_INDUSTRY_AVAILABLE and 'bank_data' in globals() and not bank_data.empty:
        detected_industry = universal_industry_system.auto_detect_industry(bank_data)
        print(f"🔍 Auto-detected industry: {detected_industry}")
        if detected_industry != 'steel':
            print(f"📊 Using {universal_industry_system.get_industry_profile(detected_industry).name} industry profile")
    
    # Get industry context for analysis
    if UNIVERSAL_INDUSTRY_AVAILABLE:
        industry_profile = universal_industry_system.get_industry_profile()
        industry_context = f"{industry_profile.name} industry analysis with {industry_profile.description}"
    else:
        industry_context = "Steel industry analysis with manufacturing operations"
'''
    
    # Insert industry detection after data loading
    data_loading_pattern = r'# Load chart of accounts'
    content = re.sub(data_loading_pattern, industry_detection + '\n\n    # Load chart of accounts', content)
    
    # Write the updated content back to app.py
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Successfully updated app.py for universal industry support!")
    print("🔧 The system now works for ANY industry while keeping steel as a supported sector")
    print("🌍 Supported industries: Steel, Healthcare, Technology, Retail, Construction, Finance")

if __name__ == "__main__":
    update_app_py()
