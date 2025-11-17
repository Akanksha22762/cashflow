"""
Script to Update HTML Template for Universal Industry Support
==========================================================
This script updates the HTML template to make it universal for any industry
while keeping steel industry as a supported sector.
"""

import re
import os

def update_html_template():
    """Update HTML template for universal industry support."""
    
    # Read the current HTML template
    with open('templates/sap_bank_interface.html', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace steel industry context with universal industry context
    steel_context_replacements = [
        # Replace steel industry context section
        (r'<!-- Steel Industry Context -->',
         '<!-- Industry Context -->'),
        
        (r'<i class="fas fa-industry" style="color: #f59e0b;"></i> Steel Industry Context',
         '<i class="fas fa-industry" style="color: #f59e0b;"></i> <span id="industry-context-title">Industry Context</span>'),
        
        (r'<p style="margin: 0; color: #92400e; font-size: 0.9rem; line-height: 1.5;">\${result\.steel_industry_context \|\| ''Analysis specific to steel manufacturing operations and industry standards\.''}</p>',
         '<p style="margin: 0; color: #92400e; font-size: 0.9rem; line-height: 1.5;">\${result\.universal_industry_context \|\| ''Analysis specific to business operations and industry standards\.''}</p>'),
        
        # Replace steel industry context function
        (r'// 🚀 HELPER: Get Steel Industry Context',
         '// 🚀 HELPER: Get Universal Industry Context'),
        
        (r'function getSteelIndustryContext\(trendType, metricKey, value\) \{',
         'function getUniversalIndustryContext(trendType, metricKey, value) {'),
        
        # Replace steel-specific insights with universal ones
        (r"'total_revenue': `In the steel industry, total revenue of ₹\${\(value \|\| 0\)\.toLocaleString\('en-IN'\)} indicates \${value > 10000000 ? 'strong market presence' : 'moderate market position'}\. Steel plants typically see revenue fluctuations based on construction cycles, infrastructure projects, and global steel prices\.`",
         "'total_revenue': `In business operations, total revenue of ₹\${(value || 0).toLocaleString('en-IN')} indicates \${value > 10000000 ? 'strong market presence' : 'moderate market position'}. Business revenue typically reflects market demand, operational efficiency, and competitive position.`"),
        
        (r"'monthly_patterns': `Monthly revenue patterns show \${value > 0\.7 ? 'consistent revenue streams' : value > 0\.4 ? 'moderate revenue stability' : 'variable production levels'}\. Steel industry monthly patterns often reflect seasonal construction activity and infrastructure development cycles\.`",
         "'monthly_patterns': `Monthly revenue patterns show \${value > 0.7 ? 'consistent revenue streams' : value > 0.4 ? 'moderate revenue stability' : 'variable production levels'}. Business monthly patterns often reflect operational cycles, seasonal trends, and market conditions.`"),
        
        (r"'revenue_growth_rate': `A growth rate of \${value}% in steel revenue indicates \${value > 0 ? 'market expansion and increased demand' : 'market contraction or pricing pressure'}\. Steel industry growth is closely tied to construction, automotive, and infrastructure sectors\.`",
         "'revenue_growth_rate': `A growth rate of \${value}% in business revenue indicates \${value > 0 ? 'market expansion and increased demand' : 'market contraction or pricing pressure'}. Business growth is closely tied to market conditions, operational efficiency, and competitive factors.`"),
        
        (r"'revenue_volatility': `Revenue volatility of \${value} indicates \${value < 0\.3 ? 'stable revenue streams' : value < 0\.6 ? 'moderate revenue fluctuations' : 'high revenue volatility with patterns common in spot market sales'}\. Steel industry volatility often reflects raw material price fluctuations and demand variations\.`",
         "'revenue_volatility': `Revenue volatility of \${value} indicates \${value < 0.3 ? 'stable revenue streams' : value < 0.6 ? 'moderate revenue fluctuations' : 'high revenue volatility with patterns common in dynamic markets'}. Business volatility often reflects market conditions, demand variations, and operational factors.`"),
        
        (r"'customer_contracts': `Customer contract value of ₹\${\(value \|\| 0\)\.toLocaleString\('en-IN'\)} indicates \${value > 5000000 ? 'large enterprise contracts' : value > 1000000 ? 'medium business contracts' : 'smaller contracts and spot market sales'}\. Steel contracts often reflect project size and customer relationship depth\.`",
         "'customer_contracts': `Customer contract value of ₹\${(value || 0).toLocaleString('en-IN')} indicates \${value > 5000000 ? 'large enterprise contracts' : value > 1000000 ? 'medium business contracts' : 'smaller contracts and spot market sales'}. Business contracts often reflect project size and customer relationship depth.`"),
        
        (r"'pricing_strategy': `Pricing strategy analysis shows \${value > 0\.7 ? 'premium pricing with strong market position' : value > 0\.4 ? 'competitive pricing with moderate market position' : 'value pricing with potential market slowdown'}\. Steel sales are influenced by construction seasons, infrastructure projects, and global economic conditions\.`",
         "'pricing_strategy': `Pricing strategy analysis shows \${value > 0.7 ? 'premium pricing with strong market position' : value > 0.4 ? 'competitive pricing with moderate market position' : 'value pricing with potential market slowdown'}. Business pricing is influenced by market conditions, competitive factors, and economic conditions.`"),
        
        (r"'forecast_accuracy': `Forecast accuracy of \${value}% indicates \${value > 80 ? 'high accuracy forecasting' : value > 60 ? 'moderate accuracy forecasting' : 'variable forecasting accuracy'}\. Steel industry forecasting considers seasonal patterns, infrastructure spending, and economic indicators\.`",
         "'forecast_accuracy': `Forecast accuracy of \${value}% indicates \${value > 80 ? 'high accuracy forecasting' : value > 60 ? 'moderate accuracy forecasting' : 'variable forecasting accuracy'}. Business forecasting considers seasonal patterns, market trends, and economic indicators.`"),
        
        (r"'total_customers': `\${value} customers indicate \${value > 50 ? 'broad market reach' : 'focused customer base'}\. Steel industry customers typically include construction companies, infrastructure developers, and manufacturing firms\.`",
         "'total_customers': `\${value} customers indicate \${value > 50 ? 'broad market reach' : 'focused customer base'}. Business customers typically include various industry sectors and market segments.`"),
        
        (r"'customer_contracts': `Customer contract analysis shows \${value > 0\.7 ? 'large enterprise contracts' : value > 0\.4 ? 'medium business contracts' : 'smaller projects and spot market sales'}\. Steel contracts often reflect project size and customer relationship depth\.`",
         "'customer_contracts': `Customer contract analysis shows \${value > 0.7 ? 'large enterprise contracts' : value > 0.4 ? 'medium business contracts' : 'smaller projects and spot market sales'}. Business contracts often reflect project size and customer relationship depth.`"),
        
        # Replace generic steel industry context
        (r'return contexts\[trendType\]\?\.\[metricKey\] \|\| ''In the steel industry, this metric reflects market conditions, production capacity, and customer demand patterns that are typical of heavy manufacturing and construction sectors\.'';',
         'return contexts[trendType]?.[metricKey] || ''In business operations, this metric reflects market conditions, operational capacity, and customer demand patterns that are typical of business operations and market sectors.'';'),
        
        # Replace steel industry context in parameter analysis
        (r'<p style="color: #6b7280; font-size: 0.9rem; margin: 8px 0;">AI-powered analysis for each parameter with steel industry insights</p>',
         '<p style="color: #6b7280; font-size: 0.9rem; margin: 8px 0;">AI-powered analysis for each parameter with industry-specific insights</p>'),
        
        # Replace steel industry context in trend analysis
        (r'<h5>Steel Industry Context</h5>',
         '<h5 id="trend-industry-context-title">Industry Context</h5>'),
        
        (r'\${getSteelIndustryContext\(trendType, metricKey, rawValue\)}',
         '${getUniversalIndustryContext(trendType, metricKey, rawValue)}'),
        
        # Replace steel-specific transaction patterns
        (r"'scrap metal sale', 'excess steel scrap', 'export payment'",
         "'business transaction', 'product sale', 'export payment'"),
        
        # Replace steel-specific insights
        (r'insights \+= `📊 Normal for growth-oriented steel manufacturing operations\\n`;',
         'insights += `📊 Normal for growth-oriented business operations\\n`;'),
        
        # Replace steel-specific sample data
        (r"description: 'Raw Materials Purchase - Steel'",
         "description: 'Raw Materials Purchase'"),
        
        # Replace steel plant specific categories
        (r'// Steel Plant Specific Categories',
         '// Industry-Specific Categories'),
        
        (r"if \(vendorName\.toLowerCase\(\)\.includes\('steel'\) \|\| vendorName\.toLowerCase\(\)\.includes\('metal'\)\) \{",
         "if (vendorName.toLowerCase().includes('steel') || vendorName.toLowerCase().includes('metal') || vendorName.toLowerCase().includes('business')) {"),
        
        (r"description: `\${vendorName} - Raw Steel Supply`",
         "description: `\${vendorName} - Raw Materials Supply`"),
        
        (r"description: `\${vendorName} - Steel Alloy Materials`",
         "description: `\${vendorName} - Alloy Materials`"),
        
        (r"description: `\${vendorName} - Steel Quality Testing`",
         "description: `\${vendorName} - Quality Testing`"),
        
        (r"description: `\${vendorName} - Steel Transportation`",
         "description: `\${vendorName} - Transportation`")
    ]
    
    for old_pattern, new_pattern in steel_context_replacements:
        content = re.sub(old_pattern, new_pattern, content)
    
    # Add industry detection and dynamic context switching
    industry_detection_script = '''
    // 🚀 INDUSTRY DETECTION AND DYNAMIC CONTEXT SWITCHING
    let currentIndustry = 'steel'; // Default to steel
    let industryProfiles = {
        'steel': {
            name: 'Steel Manufacturing',
            context: 'Steel Industry Context',
            insights: 'Analysis specific to steel manufacturing operations and industry standards.'
        },
        'healthcare': {
            name: 'Healthcare Services',
            context: 'Healthcare Industry Context',
            insights: 'Analysis specific to healthcare operations and medical industry standards.'
        },
        'technology': {
            name: 'Technology & Software',
            context: 'Technology Industry Context',
            insights: 'Analysis specific to technology operations and software industry standards.'
        },
        'retail': {
            name: 'Retail & E-commerce',
            context: 'Retail Industry Context',
            insights: 'Analysis specific to retail operations and consumer industry standards.'
        },
        'construction': {
            name: 'Construction & Infrastructure',
            context: 'Construction Industry Context',
            insights: 'Analysis specific to construction operations and infrastructure industry standards.'
        },
        'finance': {
            name: 'Financial Services',
            context: 'Finance Industry Context',
            insights: 'Analysis specific to financial operations and banking industry standards.'
        }
    };
    
    function detectIndustryFromData(data) {
        // Simple industry detection based on transaction descriptions
        const text = JSON.stringify(data).toLowerCase();
        
        if (text.includes('medical') || text.includes('patient') || text.includes('healthcare')) {
            return 'healthcare';
        } else if (text.includes('software') || text.includes('technology') || text.includes('digital')) {
            return 'technology';
        } else if (text.includes('retail') || text.includes('store') || text.includes('e-commerce')) {
            return 'retail';
        } else if (text.includes('construction') || text.includes('building') || text.includes('project')) {
            return 'construction';
        } else if (text.includes('banking') || text.includes('financial') || text.includes('insurance')) {
            return 'finance';
        } else if (text.includes('steel') || text.includes('iron') || text.includes('coal')) {
            return 'steel';
        }
        
        return 'steel'; // Default to steel
    }
    
    function updateIndustryContext(industryCode) {
        currentIndustry = industryCode;
        const profile = industryProfiles[industryCode];
        
        // Update all industry context elements
        const contextElements = document.querySelectorAll('[id*="industry-context"], [id*="trend-industry-context"]');
        contextElements.forEach(element => {
            if (element.id.includes('title')) {
                element.textContent = profile.context;
            }
        });
        
        // Update industry insights
        const insightElements = document.querySelectorAll('p[style*="color: #92400e"]');
        insightElements.forEach(element => {
            if (element.textContent.includes('Analysis specific to')) {
                element.textContent = profile.insights;
            }
        });
        
        console.log(`🌍 Switched to ${profile.name} industry context`);
    }
    
    // Auto-detect industry when data is loaded
    function autoDetectAndUpdateIndustry(data) {
        const detectedIndustry = detectIndustryFromData(data);
        if (detectedIndustry !== currentIndustry) {
            updateIndustryContext(detectedIndustry);
        }
    }
    
    // Export for use in other functions
    window.industrySystem = {
        detectIndustryFromData,
        updateIndustryContext,
        autoDetectAndUpdateIndustry,
        getCurrentIndustry: () => currentIndustry,
        getIndustryProfile: () => industryProfiles[currentIndustry]
    };
'''
    
    # Insert industry detection script after the existing script tags
    script_insert_pattern = r'<script src="https://cdn\.sheetjs\.com/xlsx-0\.19\.3/package/dist/xlsx\.full\.min\.js"></script>'
    content = re.sub(script_insert_pattern, script_insert_pattern + '\n' + industry_detection_script, content)
    
    # Add industry context update calls to existing functions
    # Update the displayRevenueResults function to include industry detection
    display_results_pattern = r'function displayRevenueResults\(results\) \{'
    display_results_replacement = '''function displayRevenueResults(results) {
        // Auto-detect industry from results data
        if (window.industrySystem && results && results.transactions) {
            window.industrySystem.autoDetectAndUpdateIndustry(results.transactions);
        }
        
        // Apply validation when displaying results'''
    
    content = re.sub(display_results_pattern, display_results_replacement, content)
    
    # Write the updated content back to the HTML template
    with open('templates/sap_bank_interface.html', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Successfully updated HTML template for universal industry support!")
    print("🔧 The interface now dynamically adapts to any industry while keeping steel as default")
    print("🌍 Industry context automatically switches based on uploaded data")

if __name__ == "__main__":
    update_html_template()
