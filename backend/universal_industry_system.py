"""
Universal Industry System for Cash Flow Analysis
===============================================
This system makes the cash flow analysis truly universal for ANY industry while
keeping steel industry as a fully supported sector. It auto-detects industry,
provides industry-specific insights, and works equally well for all sectors.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import re
from dataclasses import dataclass

@dataclass
class IndustryProfile:
    """Complete industry profile with all necessary configurations."""
    name: str
    code: str
    description: str
    
    # Financial benchmarks
    debt_equity_ratio: float
    inventory_turnover_target: float
    receivables_days_target: int
    payables_days_target: int
    gross_margin_target: float
    operating_margin_target: float
    
    # Industry-specific categories
    operating_categories: List[str]
    investing_categories: List[str]
    financing_categories: List[str]
    
    # Industry context keywords for auto-detection
    context_keywords: List[str]
    business_terms: List[str]
    transaction_patterns: List[str]
    
    # Industry-specific insights and analysis
    revenue_insights: Dict[str, str]
    expense_insights: Dict[str, str]
    market_factors: List[str]
    risk_factors: List[str]
    
    # Industry-specific vendor patterns
    vendor_patterns: List[str]
    supplier_keywords: List[str]
    customer_keywords: List[str]
    
    # Transaction thresholds and limits
    min_transaction_threshold: float
    max_transaction_threshold: float
    typical_transaction_range: Tuple[float, float]

class UniversalIndustrySystem:
    """Universal industry system that works for ANY sector."""
    
    def __init__(self):
        self.industry_profiles = self._create_industry_profiles()
        self.current_industry = 'steel'  # Default to steel
        self.detected_industry = None
    
    def _create_industry_profiles(self) -> Dict[str, IndustryProfile]:
        """Create comprehensive industry profiles."""
        return {
            'steel': IndustryProfile(
                name="Steel Manufacturing",
                code="STEEL",
                description="Steel production, manufacturing, and distribution",
                debt_equity_ratio=0.4,
                inventory_turnover_target=4.0,
                receivables_days_target=60,
                payables_days_target=45,
                gross_margin_target=0.25,
                operating_margin_target=0.12,
                operating_categories=[
                    "Raw Materials", "Steel Production", "Labor", "Utilities",
                    "Maintenance", "Quality Control", "Transportation", "Steel Sales"
                ],
                investing_categories=[
                    "Steel Plant Equipment", "Furnace Upgrades", "Plant Expansion",
                    "Technology Investment", "Infrastructure Development"
                ],
                financing_categories=[
                    "Steel Working Capital", "Equipment Financing", "Plant Loans",
                    "Trade Credit", "Steel Industry Financing"
                ],
                context_keywords=[
                    "steel", "iron", "coal", "rolling mill", "blast furnace",
                    "steel plant", "steel production", "steel manufacturing"
                ],
                business_terms=[
                    "steel", "manufacturing", "production", "plant", "mill",
                    "furnace", "rolling", "galvanized", "alloy"
                ],
                transaction_patterns=[
                    "steel sale", "raw material purchase", "steel production",
                    "plant maintenance", "steel transportation"
                ],
                revenue_insights={
                    'total_revenue': 'Steel revenue reflects production capacity, market demand, and global steel prices.',
                    'revenue_growth_rate': 'Steel growth indicates construction demand, infrastructure projects, and market expansion.',
                    'monthly_patterns': 'Steel patterns reflect construction cycles, infrastructure spending, and seasonal demand.'
                },
                expense_insights={
                    'operating_expenses': 'Steel expenses include raw materials, energy, labor, and plant maintenance.',
                    'inventory_costs': 'Steel inventory management balances production efficiency with market demand.'
                },
                market_factors=[
                    "Global steel prices", "Construction demand", "Infrastructure spending",
                    "Raw material costs", "Energy prices", "Trade policies"
                ],
                risk_factors=[
                    "Commodity price volatility", "Energy cost fluctuations", "Trade restrictions",
                    "Environmental regulations", "Labor availability", "Equipment maintenance"
                ],
                vendor_patterns=[
                    "steel supplier", "coal supplier", "iron ore supplier", "equipment supplier",
                    "logistics provider", "maintenance contractor"
                ],
                supplier_keywords=[
                    "steel", "coal", "iron", "ore", "alloy", "equipment", "machinery"
                ],
                customer_keywords=[
                    "construction", "automotive", "infrastructure", "manufacturing", "engineering"
                ],
                min_transaction_threshold=1000.0,
                max_transaction_threshold=10000000.0,
                typical_transaction_range=(5000.0, 500000.0)
            ),
            
            'healthcare': IndustryProfile(
                name="Healthcare Services",
                code="HEALTH",
                description="Medical services, hospitals, clinics, and healthcare",
                debt_equity_ratio=0.3,
                inventory_turnover_target=12.0,
                receivables_days_target=60,
                payables_days_target=45,
                gross_margin_target=0.35,
                operating_margin_target=0.08,
                operating_categories=[
                    "Medical Supplies", "Laboratory Costs", "Staff Salaries",
                    "Equipment Maintenance", "Utilities", "Insurance", "Licensing"
                ],
                investing_categories=[
                    "Medical Equipment", "Facility Expansion", "Technology Systems",
                    "Laboratory Equipment", "Patient Care Systems"
                ],
                financing_categories=[
                    "Medical Equipment Financing", "Working Capital", "Facility Loans",
                    "Insurance Premiums", "Regulatory Compliance"
                ],
                context_keywords=[
                    "medical", "healthcare", "hospital", "clinic", "patient",
                    "treatment", "diagnosis", "clinical", "therapeutic"
                ],
                business_terms=[
                    "healthcare", "medical", "clinical", "patient care", "treatment",
                    "diagnosis", "therapeutic", "preventive"
                ],
                transaction_patterns=[
                    "patient payment", "medical supply purchase", "equipment maintenance",
                    "staff salary", "insurance premium", "licensing fee"
                ],
                revenue_insights={
                    'total_revenue': 'Healthcare revenue reflects patient volume, service mix, and reimbursement rates.',
                    'revenue_growth_rate': 'Healthcare growth indicates increased patient demand or service expansion.',
                    'monthly_patterns': 'Healthcare patterns reflect seasonal health trends and insurance cycles.'
                },
                expense_insights={
                    'operating_expenses': 'Healthcare expenses include medical supplies, staff costs, and regulatory compliance.',
                    'inventory_costs': 'Medical inventory balances cost control with patient care needs.'
                },
                market_factors=[
                    "Patient demographics", "Insurance regulations", "Medical technology",
                    "Regulatory compliance", "Staff availability", "Market competition"
                ],
                risk_factors=[
                    "Regulatory changes", "Insurance reimbursement cuts", "Medical malpractice",
                    "Technology obsolescence", "Staff shortages", "Patient safety"
                ],
                vendor_patterns=[
                    "medical supplier", "equipment vendor", "pharmaceutical company",
                    "laboratory service", "maintenance contractor"
                ],
                supplier_keywords=[
                    "medical", "pharmaceutical", "equipment", "laboratory", "supplies"
                ],
                customer_keywords=[
                    "patient", "insurance", "referral", "medical", "healthcare"
                ],
                min_transaction_threshold=500.0,
                max_transaction_threshold=5000000.0,
                typical_transaction_range=(1000.0, 100000.0)
            ),
            
            'technology': IndustryProfile(
                name="Technology & Software",
                code="TECH",
                description="Software, IT services, and technology companies",
                debt_equity_ratio=0.2,
                inventory_turnover_target=20.0,
                receivables_days_target=45,
                payables_days_target=30,
                gross_margin_target=0.70,
                operating_margin_target=0.20,
                operating_categories=[
                    "Research & Development", "Software Development", "Cloud Services",
                    "Staff Salaries", "Marketing", "Customer Support", "Infrastructure"
                ],
                investing_categories=[
                    "Technology Infrastructure", "Software Licenses", "Research Equipment",
                    "Intellectual Property", "Acquisitions", "Data Centers"
                ],
                financing_categories=[
                    "Venture Capital", "Working Capital", "Equipment Financing",
                    "Intellectual Property Protection", "Research Funding"
                ],
                context_keywords=[
                    "technology", "software", "digital", "innovation", "development",
                    "platform", "solution", "service", "product", "IT"
                ],
                business_terms=[
                    "technology", "software", "digital", "innovation", "development",
                    "platform", "solution", "service", "product", "IT"
                ],
                transaction_patterns=[
                    "software license", "cloud service", "development cost",
                    "staff salary", "marketing expense", "infrastructure cost"
                ],
                revenue_insights={
                    'total_revenue': 'Technology revenue reflects product adoption, market penetration, and innovation.',
                    'revenue_growth_rate': 'Technology growth indicates market expansion and product success.',
                    'monthly_patterns': 'Technology patterns reflect product launches and subscription cycles.'
                },
                expense_insights={
                    'operating_expenses': 'Technology expenses include R&D, staff costs, and infrastructure.',
                    'inventory_costs': 'Technology companies typically have low inventory costs.'
                },
                market_factors=[
                    "Innovation rate", "Market adoption", "Competition", "Regulatory environment",
                    "Talent availability", "Funding availability", "Technology trends"
                ],
                risk_factors=[
                    "Technology obsolescence", "Market disruption", "Talent competition",
                    "Intellectual property theft", "Cybersecurity threats", "Regulatory changes"
                ],
                vendor_patterns=[
                    "software vendor", "cloud provider", "equipment supplier",
                    "service provider", "consulting firm"
                ],
                supplier_keywords=[
                    "software", "cloud", "equipment", "service", "consulting"
                ],
                customer_keywords=[
                    "enterprise", "business", "consumer", "government", "education"
                ],
                min_transaction_threshold=1000.0,
                max_transaction_threshold=1000000.0,
                typical_transaction_range=(5000.0, 50000.0)
            ),
            
            'retail': IndustryProfile(
                name="Retail & E-commerce",
                code="RETAIL",
                description="Retail stores, e-commerce, and consumer goods",
                debt_equity_ratio=0.4,
                inventory_turnover_target=8.0,
                receivables_days_target=30,
                payables_days_target=60,
                gross_margin_target=0.40,
                operating_margin_target=0.08,
                operating_categories=[
                    "Inventory Purchase", "Store Operations", "Staff Salaries",
                    "Rent & Utilities", "Marketing", "Customer Service", "E-commerce"
                ],
                investing_categories=[
                    "Store Expansion", "Technology Systems", "Equipment Purchase",
                    "E-commerce Platform", "Supply Chain Systems", "Warehouse"
                ],
                financing_categories=[
                    "Inventory Financing", "Working Capital", "Store Leases",
                    "Equipment Leasing", "Trade Credit", "E-commerce Investment"
                ],
                context_keywords=[
                    "retail", "store", "customer", "sales", "inventory",
                    "merchandise", "shopping", "e-commerce", "brick-and-mortar"
                ],
                business_terms=[
                    "retail", "sales", "inventory", "merchandise", "customer service",
                    "store operations", "e-commerce", "omnichannel"
                ],
                transaction_patterns=[
                    "inventory purchase", "store rent", "staff salary", "marketing cost",
                    "customer refund", "e-commerce transaction"
                ],
                revenue_insights={
                    'total_revenue': 'Retail revenue reflects customer traffic, average transaction value, and product mix.',
                    'revenue_growth_rate': 'Retail growth indicates increased customer demand or market expansion.',
                    'monthly_patterns': 'Retail patterns reflect seasonal shopping trends and promotional cycles.'
                },
                expense_insights={
                    'operating_expenses': 'Retail expenses include inventory costs, store operations, and marketing.',
                    'inventory_costs': 'Inventory management is critical for retail profitability and cash flow.'
                },
                market_factors=[
                    "Consumer spending", "Seasonal trends", "Competition", "E-commerce adoption",
                    "Supply chain efficiency", "Marketing effectiveness", "Economic conditions"
                ],
                risk_factors=[
                    "Consumer spending decline", "Competition", "Supply chain disruption",
                    "Technology changes", "Economic recession", "Seasonal fluctuations"
                ],
                vendor_patterns=[
                    "inventory supplier", "equipment vendor", "service provider",
                    "marketing agency", "logistics provider"
                ],
                supplier_keywords=[
                    "inventory", "equipment", "service", "marketing", "logistics"
                ],
                customer_keywords=[
                    "consumer", "customer", "shopper", "buyer", "end-user"
                ],
                min_transaction_threshold=100.0,
                max_transaction_threshold=100000.0,
                typical_transaction_range=(500.0, 5000.0)
            ),
            
            'construction': IndustryProfile(
                name="Construction & Infrastructure",
                code="CONST",
                description="Construction, infrastructure, and engineering",
                debt_equity_ratio=0.6,
                inventory_turnover_target=4.0,
                receivables_days_target=60,
                payables_days_target=45,
                gross_margin_target=0.20,
                operating_margin_target=0.08,
                operating_categories=[
                    "Materials", "Labor", "Equipment Rental", "Subcontractor Costs",
                    "Insurance", "Permits", "Safety Equipment", "Project Management"
                ],
                investing_categories=[
                    "Equipment Purchase", "Fleet Expansion", "Technology Systems",
                    "Facility Development", "Safety Systems", "Project Tools"
                ],
                financing_categories=[
                    "Project Financing", "Equipment Leasing", "Working Capital",
                    "Performance Bonds", "Insurance Premiums", "Project Loans"
                ],
                context_keywords=[
                    "construction", "building", "project", "infrastructure",
                    "development", "contracting", "engineering", "site"
                ],
                business_terms=[
                    "construction", "building", "project", "infrastructure",
                    "development", "contracting", "engineering"
                ],
                transaction_patterns=[
                    "material purchase", "equipment rental", "subcontractor payment",
                    "project milestone", "safety equipment", "permit fee"
                ],
                revenue_insights={
                    'total_revenue': 'Construction revenue reflects project pipeline, contract values, and execution efficiency.',
                    'revenue_growth_rate': 'Construction growth indicates increased project demand or market expansion.',
                    'monthly_patterns': 'Construction patterns reflect project cycles and seasonal weather conditions.'
                },
                expense_insights={
                    'operating_expenses': 'Construction expenses include materials, labor, equipment, and project management.',
                    'inventory_costs': 'Construction inventory includes materials and equipment for active projects.'
                },
                market_factors=[
                    "Project pipeline", "Economic conditions", "Regulatory environment",
                    "Material costs", "Labor availability", "Weather conditions", "Infrastructure spending"
                ],
                risk_factors=[
                    "Project delays", "Cost overruns", "Weather impact", "Material price volatility",
                    "Labor shortages", "Regulatory changes", "Economic downturn"
                ],
                vendor_patterns=[
                    "material supplier", "equipment rental", "subcontractor",
                    "service provider", "safety equipment"
                ],
                supplier_keywords=[
                    "material", "equipment", "subcontractor", "service", "safety"
                ],
                customer_keywords=[
                    "client", "project owner", "developer", "government", "corporation"
                ],
                min_transaction_threshold=5000.0,
                max_transaction_threshold=10000000.0,
                typical_transaction_range=(10000.0, 500000.0)
            ),
            
            'finance': IndustryProfile(
                name="Financial Services",
                code="FIN",
                description="Banking, insurance, and financial services",
                debt_equity_ratio=0.8,
                inventory_turnover_target=50.0,
                receivables_days_target=30,
                payables_days_target=15,
                gross_margin_target=0.60,
                operating_margin_target=0.25,
                operating_categories=[
                    "Staff Salaries", "Technology Systems", "Regulatory Compliance",
                    "Marketing", "Customer Service", "Risk Management", "Operations"
                ],
                investing_categories=[
                    "Technology Infrastructure", "Compliance Systems", "Risk Management Tools",
                    "Customer Platforms", "Data Analytics", "Security Systems"
                ],
                financing_categories=[
                    "Capital Requirements", "Regulatory Capital", "Working Capital",
                    "Technology Investment", "Compliance Investment"
                ],
                context_keywords=[
                    "financial", "banking", "insurance", "investment", "lending",
                    "credit", "risk", "compliance", "regulatory"
                ],
                business_terms=[
                    "financial", "banking", "insurance", "investment", "lending",
                    "credit", "risk", "compliance", "regulatory"
                ],
                transaction_patterns=[
                    "loan disbursement", "interest payment", "fee collection",
                    "insurance premium", "compliance cost", "technology expense"
                ],
                revenue_insights={
                    'total_revenue': 'Financial revenue reflects loan volume, interest rates, and fee income.',
                    'revenue_growth_rate': 'Financial growth indicates increased lending activity and market expansion.',
                    'monthly_patterns': 'Financial patterns reflect lending cycles and regulatory reporting periods.'
                },
                expense_insights={
                    'operating_expenses': 'Financial expenses include technology, compliance, and operational costs.',
                    'inventory_costs': 'Financial services have minimal inventory costs.'
                },
                market_factors=[
                    "Interest rates", "Regulatory environment", "Economic conditions",
                    "Market competition", "Technology adoption", "Risk appetite"
                ],
                risk_factors=[
                    "Credit risk", "Interest rate risk", "Regulatory changes",
                    "Market volatility", "Cybersecurity threats", "Compliance failures"
                ],
                vendor_patterns=[
                    "technology vendor", "service provider", "compliance consultant",
                    "risk management", "security provider"
                ],
                supplier_keywords=[
                    "technology", "service", "compliance", "risk", "security"
                ],
                customer_keywords=[
                    "borrower", "investor", "policyholder", "client", "customer"
                ],
                min_transaction_threshold=1000.0,
                max_transaction_threshold=5000000.0,
                typical_transaction_range=(5000.0, 100000.0)
            )
        }
    
    def auto_detect_industry(self, data: pd.DataFrame) -> str:
        """Auto-detect industry from uploaded data."""
        if data.empty:
            return 'steel'  # Default to steel
        
        # Combine all text data for analysis
        text_data = ""
        for col in data.columns:
            if data[col].dtype == 'object':
                text_data += " " + " ".join(data[col].astype(str).dropna())
        
        text_data = text_data.lower()
        
        # Score each industry based on keyword matches
        industry_scores = {}
        for code, profile in self.industry_profiles.items():
            score = 0
            for keyword in profile.context_keywords:
                if keyword in text_data:
                    score += 1
            industry_scores[code] = score
        
        # Return industry with highest score, or default to steel
        if industry_scores:
            best_industry = max(industry_scores, key=industry_scores.get)
            if industry_scores[best_industry] > 0:
                self.detected_industry = best_industry
                return best_industry
        
        # Default to steel if no clear match
        self.detected_industry = 'steel'
        return 'steel'
    
    def get_industry_profile(self, industry_code: str = None) -> IndustryProfile:
        """Get industry profile by code or use detected/default."""
        if not industry_code:
            industry_code = self.detected_industry or self.current_industry
        
        if industry_code not in self.industry_profiles:
            industry_code = 'steel'  # Fallback to steel
        
        self.current_industry = industry_code
        return self.industry_profiles[industry_code]
    
    def generate_industry_context(self, metric: str, value: Any, industry_code: str = None) -> str:
        """Generate industry-specific context for any metric."""
        profile = self.get_industry_profile(industry_code)
        
        # Try to get industry-specific insight
        if metric in profile.revenue_insights:
            return profile.revenue_insights[metric]
        elif metric in profile.expense_insights:
            return profile.expense_insights[metric]
        
        # Generic context if no industry-specific insight
        return f"In {profile.name.lower()} operations, {metric} of {value} reflects business performance and market conditions."
    
    def get_industry_benchmarks(self, industry_code: str = None) -> Dict[str, Any]:
        """Get industry-specific financial benchmarks."""
        profile = self.get_industry_profile(industry_code)
        
        return {
            'debt_equity_ratio': profile.debt_equity_ratio,
            'inventory_turnover_target': profile.inventory_turnover_target,
            'receivables_days_target': profile.receivables_days_target,
            'payables_days_target': profile.payables_days_target,
            'gross_margin_target': profile.gross_margin_target,
            'operating_margin_target': profile.operating_margin_target,
            'industry_name': profile.name,
            'industry_code': profile.code,
            'description': profile.description
        }
    
    def get_industry_categories(self, industry_code: str = None) -> Dict[str, List[str]]:
        """Get industry-specific category lists."""
        profile = self.get_industry_profile(industry_code)
        
        return {
            'operating': profile.operating_categories,
            'investing': profile.investing_categories,
            'financing': profile.financing_categories
        }
    
    def get_industry_insights(self, industry_code: str = None) -> Dict[str, Any]:
        """Get comprehensive industry insights."""
        profile = self.get_industry_profile(industry_code)
        
        return {
            'market_factors': profile.market_factors,
            'risk_factors': profile.risk_factors,
            'vendor_patterns': profile.vendor_patterns,
            'supplier_keywords': profile.supplier_keywords,
            'customer_keywords': profile.customer_keywords
        }
    
    def is_steel_industry(self, industry_code: str = None) -> bool:
        """Check if the current industry is steel."""
        if not industry_code:
            industry_code = self.current_industry
        return industry_code == 'steel'
    
    def get_all_industries(self) -> List[Dict[str, str]]:
        """Get list of all supported industries."""
        return [
            {'code': code, 'name': profile.name, 'description': profile.description}
            for code, profile in self.industry_profiles.items()
        ]

# Global instance
universal_industry_system = UniversalIndustrySystem()

# Export for use in other modules
__all__ = ['universal_industry_system', 'IndustryProfile', 'UniversalIndustrySystem']
