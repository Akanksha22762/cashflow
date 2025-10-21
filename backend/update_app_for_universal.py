"""
Script to Update app.py for Universal Industry Support
====================================================
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
    
    # Replace hardcoded steel industry prompts with universal ones
    steel_prompts = [
        (r'Analyze financial trends for {trend_type} in the STEEL INDUSTRY based on this data:',
         'Analyze financial trends for {trend_type} based on this data:'),
        
        (r'CONTEXT: This is a steel manufacturing company with transactions related to:',
         'CONTEXT: This is a business with transactions related to:'),
        
        (r'- Steel production, sales, and distribution',
         '- Business operations, sales, and distribution'),
        
        (r'STEEL INDUSTRY SPECIFIC ANALYSIS for {trend_type}:',
         'INDUSTRY-SPECIFIC ANALYSIS for {trend_type}:'),
        
        (r'2. Risk assessment \(low/medium/high\) based on steel industry standards',
         '2. Risk assessment (low/medium/high) based on industry standards'),
        
        (r'3. Key insights specific to steel manufacturing and operations',
         '3. Key insights specific to business operations and industry context'),
        
        (r'4. Recommendations for steel industry optimization',
         '4. Recommendations for industry optimization'),
        
        (r'Consider steel industry factors:',
         'Consider industry-specific factors:')
    ]
    
    for old_pattern, new_pattern in steel_prompts:
        content = re.sub(old_pattern, new_pattern, content)
    
    # Replace steel_industry_context with universal_industry_context
    content = re.sub(r'steel_industry_context', 'universal_industry_context', content)
    
    # Replace steel-specific context strings
    steel_contexts = [
        (r"'steel_industry_context': 'Revenue analysis for steel manufacturing operations'",
         "'universal_industry_context': 'Revenue analysis for business operations'"),
        
        (r"'steel_industry_context': 'Sales forecasting for steel products and services'",
         "'universal_industry_context': 'Sales forecasting for products and services'"),
        
        (r"'steel_industry_context': 'Customer contract analysis for steel industry clients'",
         "'universal_industry_context': 'Customer contract analysis for business clients'"),
        
        (r"'steel_industry_context': 'Pricing strategy analysis for steel products'",
         "'universal_industry_context': 'Pricing strategy analysis for products'"),
        
        (r"'steel_industry_context': 'AR aging analysis for steel industry receivables'",
         "'universal_industry_context': 'AR aging analysis for business receivables'"),
        
        (r"'steel_industry_context': 'Operating expense analysis for steel manufacturing'",
         "'universal_industry_context': 'Operating expense analysis for business operations'"),
        
        (r"'steel_industry_context': 'AP analysis for steel industry suppliers'",
         "'universal_industry_context': 'AP analysis for business suppliers'"),
        
        (r"'steel_industry_context': 'Inventory turnover analysis for steel products'",
         "'universal_industry_context': 'Inventory turnover analysis for business products'"),
        
        (r"'steel_industry_context': 'Loan repayment analysis for steel industry financing'",
         "'universal_industry_context': 'Loan repayment analysis for business financing'"),
        
        (r"'steel_industry_context': 'Tax obligation analysis for steel industry compliance'",
         "'universal_industry_context': 'Tax obligation analysis for business compliance'"),
        
        (r"'steel_industry_context': 'Capital expenditure analysis for steel industry investments'",
         "'universal_industry_context': 'Capital expenditure analysis for business investments'"),
        
        (r"'steel_industry_context': 'Equity and debt analysis for steel industry financing'",
         "'universal_industry_context': 'Equity and debt analysis for business financing'"),
        
        (r"'steel_industry_context': 'Other income/expense analysis for steel industry operations'",
         "'universal_industry_context': 'Other income/expense analysis for business operations'"),
        
        (r"'steel_industry_context': 'Cash flow type analysis for steel industry operations'",
         "'universal_industry_context': 'Cash flow type analysis for business operations'")
    ]
    
    for old_context, new_context in steel_contexts:
        content = re.sub(old_context, new_context, content)
    
    # Replace steel-specific analysis prompts
    steel_analysis = [
        (r'Analyze revenue trends for a steel manufacturing company:',
         'Analyze revenue trends for this business:'),
        
        (r'2. Steel industry market trends',
         '2. Industry market trends'),
        
        (r'Analyze sales forecasting for a steel manufacturing company:',
         'Analyze sales forecasting for this business:'),
        
        (r'Analyze customer contracts for a steel manufacturing company:',
         'Analyze customer contracts for this business:'),
        
        (r'Analyze pricing models for a steel manufacturing company:',
         'Analyze pricing models for this business:'),
        
        (r'Analyze accounts receivable aging for a steel manufacturing company:',
         'Analyze accounts receivable aging for this business:'),
        
        (r'Analyze operating expenses for a steel manufacturing company:',
         'Analyze operating expenses for this business:'),
        
        (r'Analyze accounts payable for a steel manufacturing company:',
         'Analyze accounts payable for this business:'),
        
        (r'Analyze inventory turnover for a steel manufacturing company:',
         'Analyze inventory turnover for this business:'),
        
        (r'Analyze loan repayments for a steel manufacturing company:',
         'Analyze loan repayments for this business:'),
        
        (r'Analyze tax obligations for a steel manufacturing company:',
         'Analyze tax obligations for this business:'),
        
        (r'Analyze capital expenditure for a steel manufacturing company:',
         'Analyze capital expenditure for this business:'),
        
        (r'Analyze equity and debt inflows for a steel manufacturing company:',
         'Analyze equity and debt inflows for this business:'),
        
        (r'Analyze other income and expenses for a steel manufacturing company:',
         'Analyze other income and expenses for this business:'),
        
        (r'Analyze cash flow types for a steel manufacturing company:',
         'Analyze cash flow types for this business:')
    ]
    
    for old_analysis, new_analysis in steel_analysis:
        content = re.sub(old_analysis, new_analysis, content)
    
    # Replace steel-specific business type detection
    steel_business_type = [
        (r'steel_indicators = \[''steel'', ''iron'', ''coal'', ''rolling mill'', ''blast furnace'', ''steel plant''\]',
         'industry_indicators = universal_industry_system.get_industry_profile().context_keywords if UNIVERSAL_INDUSTRY_AVAILABLE else [''steel'', ''iron'', ''coal'', ''rolling mill'', ''blast furnace'', ''steel plant'']'),
        
        (r'business_type = "STEEL MANUFACTURING COMPANY" if is_steel_company else "UNIVERSAL BUSINESS"',
         'business_type = f"{universal_industry_system.get_industry_profile().name.upper()}" if UNIVERSAL_INDUSTRY_AVAILABLE else ("STEEL MANUFACTURING COMPANY" if is_steel_company else "UNIVERSAL BUSINESS")'),
        
        (r'steel_indicators = \[''steel'', ''iron'', ''coal'', ''rolling mill'', ''blast furnace'', ''steel plant''\]',
         'industry_indicators = universal_industry_system.get_industry_profile().context_keywords if UNIVERSAL_INDUSTRY_AVAILABLE else [''steel'', ''iron'', ''coal'', ''rolling mill'', ''blast furnace'', ''steel plant'']')
    ]
    
    for old_business, new_business in steel_business_type:
        content = re.sub(old_business, new_business, content)
    
    # Replace steel-specific category functions with universal ones
    steel_categories = [
        (r'# 2. STEEL PRODUCTION EQUIPMENT',
         '# 2. INDUSTRY-SPECIFIC EQUIPMENT'),
        
        (r'steel_equipment = \[''steel crane'', ''steel machinery'', ''steel equipment'', ''steel plant equipment''\]',
         'industry_equipment = universal_industry_system.get_industry_profile().investing_categories if UNIVERSAL_INDUSTRY_AVAILABLE else [''steel crane'', ''steel machinery'', ''steel equipment'', ''steel plant equipment'']'),
        
        (r'return ''Steel Production Equipment''',
         'return universal_industry_system.get_industry_profile().name + " Equipment" if UNIVERSAL_INDUSTRY_AVAILABLE else "Steel Production Equipment"'),
        
        (r'# 4. STEEL INDUSTRY LABOR',
         '# 4. INDUSTRY-SPECIFIC LABOR'),
        
        (r'steel_labor = \[''steel worker'', ''steel technician'', ''steel engineer'', ''steel production staff'', ''mill worker'', ''furnace operator'', ''steel plant employee''\]',
         'industry_labor = universal_industry_system.get_industry_profile().operating_categories if UNIVERSAL_INDUSTRY_AVAILABLE else [''steel worker'', ''steel technician'', ''steel engineer'', ''steel production staff'', ''mill worker'', ''furnace operator'', ''steel plant employee'']'),
        
        (r'return ''Steel Industry Labor''',
         'return universal_industry_system.get_industry_profile().name + " Labor" if UNIVERSAL_INDUSTRY_AVAILABLE else "Steel Industry Labor"'),
        
        (r'# 8. STEEL INDUSTRY FINANCING',
         '# 8. INDUSTRY-SPECIFIC FINANCING'),
        
        (r'steel_financing = \[''steel working capital'', ''steel industry loan''\]',
         'industry_financing = universal_industry_system.get_industry_profile().financing_categories if UNIVERSAL_INDUSTRY_AVAILABLE else [''steel working capital'', ''steel industry loan'']'),
        
        (r'return ''Steel Industry Financing''',
         'return universal_industry_system.get_industry_profile().name + " Financing" if UNIVERSAL_INDUSTRY_AVAILABLE else "Steel Industry Financing"')
    ]
    
    for old_category, new_category in steel_categories:
        content = re.sub(old_category, new_category, content)
    
    # Replace steel-specific comments and references
    steel_comments = [
        (r'# ENHANCED FEATURE EXTRACTION FOR STEEL INDUSTRY',
         '# ENHANCED FEATURE EXTRACTION FOR INDUSTRY'),
        
        (r'You are a financial analyst for a steel manufacturing company.',
         'You are a financial analyst for this business.'),
        
        (r'Consider the category and typical business transactions for a steel plant.',
         'Consider the category and typical business transactions for this business.'),
        
        (r'data_folder = ''steel_plant_datasets''',
         'data_folder = ''business_datasets'''),
        
        (r'chart_of_accounts_path = os\.join\(data_folder, ''steel_plant_sap_data\.xlsx''\)',
         'chart_of_accounts_path = os.path.join(data_folder, ''business_sap_data.xlsx'')'),
        
        (r'description: ''Steel Plant Operations - Raw Materials''',
         'description: ''Business Operations - Raw Materials'''),
        
        (r'vendor: ''Steel Suppliers''',
         'vendor: ''Business Suppliers'''),
        
        (r'Regular supplier appears {count} times \(expected for steel plant operations\)',
         'Regular supplier appears {count} times (expected for business operations)'),
        
        (r'Month-end batch transaction at {row\[\''Hour\''\]}:00 \(normal for steel plant\)',
         'Month-end batch transaction at {row[\''Hour\''\']}:00 (normal for business operations)'),
        
        (r'Weekend batch processing at {row\[\''Hour\''\]}:00 \(normal for steel plant operations\)',
         'Weekend batch processing at {row[\''Hour\''\']}:00 (normal for business operations)'),
        
        (r'# 6. SEASONAL ANOMALIES \(Steel Plant Specific\)',
         '# 6. SEASONAL ANOMALIES (Industry Specific)'),
        
        (r'Steel coil production - blast furnace maintenance',
         'Production - equipment maintenance')
    ]
    
    for old_comment, new_comment in steel_comments:
        content = re.sub(old_comment, new_comment, content)
    
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
