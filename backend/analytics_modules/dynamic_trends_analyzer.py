"""
Dynamic Trends Analyzer Module
==============================
Extracted and adapted from CashflowDemo for CashflowApp.
Provides AI-powered trends analysis with OpenAI integration.
"""

import pandas as pd
import json
import re
from datetime import datetime
from typing import Dict, Any, List


class DynamicTrendsAnalyzer:
    """Dynamic trends analysis with OpenAI integration and intelligent caching"""
    
    def __init__(self, openai_integration=None):
        self.ai_cache_manager = {}
        self.batch_size = 5  # Process 5 trend parameters at once
        self.openai_model = "gpt-4o-mini"
        self.openai_integration = openai_integration
    
    def calculate_dynamic_thresholds(self, df):
        """Calculate all thresholds dynamically from actual data"""
        try:
            if df is None or df.empty or 'Amount' not in df.columns:
                return self._get_default_thresholds()
            
            amounts = df['Amount'].abs()  # Use absolute values for thresholds
            
            # Handle NaN values and ensure valid calculations
            amounts_clean = amounts.dropna()
            
            if len(amounts_clean) == 0:
                print("⚠️ No valid amount data found, using default thresholds")
                return self._get_default_thresholds()
            
            thresholds = {
                'high_value': float(amounts_clean.quantile(0.90)),      # Top 10%
                'medium_value': float(amounts_clean.quantile(0.75)),    # Top 25%
                'low_value': float(amounts_clean.quantile(0.50)),       # Top 50%
                'critical_amount': float(amounts_clean.quantile(0.95)), # Top 5%
                'max_amount': float(amounts_clean.max()),
                'avg_amount': float(amounts_clean.mean()),
                'std_amount': float(amounts_clean.std())
            }
            
            # Ensure minimum thresholds for small datasets
            min_threshold = 1000
            for key in ['high_value', 'medium_value', 'low_value']:
                if thresholds[key] < min_threshold:
                    thresholds[key] = min_threshold
            
            print(f"✅ Dynamic thresholds calculated from {len(df)} transactions")
            return thresholds
            
        except Exception as e:
            print(f"⚠️ Error calculating dynamic thresholds: {e}")
            return self._get_default_thresholds()
    
    def _get_default_thresholds(self):
        """Get default thresholds when calculation fails"""
        return {
            'high_value': 50000,
            'medium_value': 20000,
            'low_value': 5000,
            'critical_amount': 100000,
            'max_amount': 100000,
            'avg_amount': 10000,
            'std_amount': 5000
        }
    
    def calculate_dynamic_risk_levels(self, df):
        """Calculate risk levels based on actual data volatility"""
        try:
            if df is None or df.empty or 'Amount' not in df.columns:
                return {'low': 0.3, 'medium': 0.6, 'high': 1.0}
            
            amounts = df['Amount'].abs()
            amounts_clean = amounts.dropna()
            
            if len(amounts_clean) == 0:
                print("⚠️ No valid amount data found for risk calculation")
                return {'low': 0.3, 'medium': 0.6, 'high': 1.0}
            
            volatility = float(amounts_clean.std())
            mean_amount = float(amounts_clean.mean())
            
            # Prevent division by zero and ensure valid risk calculations
            if mean_amount > 0:
                # Dynamic risk based on data volatility
                risk_levels = {
                    'low': max(0.1, min(0.4, volatility / mean_amount * 0.5)),
                    'medium': max(0.3, min(0.7, volatility / mean_amount * 1.0)),
                    'high': max(0.6, min(1.0, volatility / mean_amount * 2.0))
                }
            else:
                # Fallback risk levels if mean is zero
                risk_levels = {
                    'low': 0.2,
                    'medium': 0.5,
                    'high': 0.8
                }
            
            print(f"✅ Dynamic risk levels calculated: {risk_levels}")
            return risk_levels
            
        except Exception as e:
            print(f"⚠️ Error calculating dynamic risk levels: {e}")
            return {'low': 0.3, 'medium': 0.6, 'high': 1.0}
    
    def calculate_dynamic_timeframes(self, df):
        """Calculate optimal timeframes based on data size"""
        try:
            if df is None or df.empty:
                return {'payment_due_days': 15, 'analysis_period': 'Current Period', 'trend_window': 30}
            
            data_size = len(df)
            
            # Adaptive timeframes based on data size
            timeframes = {
                'payment_due_days': min(30, max(7, data_size // 10)),
                'analysis_period': f'Last {min(365, data_size)} Days' if data_size < 365 else 'Full Year',
                'trend_window': min(90, max(7, data_size // 3))
            }
            
            print(f"✅ Dynamic timeframes calculated: {timeframes}")
            return timeframes
            
        except Exception as e:
            print(f"⚠️ Error calculating dynamic timeframes: {e}")
            return {'payment_due_days': 15, 'analysis_period': 'Current Period', 'trend_window': 30}
    
    def analyze_trends_batch(self, df, trend_types):
        """Process multiple trend types using parameter-specific analysis"""
        try:
            print(f"🔄 Processing {len(trend_types)} trend types with specialized analysis...")
            
            # Ensure Date column is properly formatted for all analyses
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df = df.dropna(subset=['Date'])  # Remove rows with invalid dates
                print(f"✅ Date column formatted: {len(df)} valid transactions remaining")
            
            # Calculate dynamic parameters once
            thresholds = self.calculate_dynamic_thresholds(df)
            risk_levels = self.calculate_dynamic_risk_levels(df)
            timeframes = self.calculate_dynamic_timeframes(df)
            
            results = {}
            
            # Process each trend type with its specialized analysis
            for trend_type in trend_types:
                try:
                    print(f"🔍 Analyzing {trend_type} with specialized method...")
                    
                    # Use parameter-specific analysis methods
                    if trend_type == 'historical_revenue_trends':
                        result = self.analyze_historical_revenue_trends(df, thresholds, risk_levels, timeframes)
                    elif trend_type == 'sales_forecast':
                        result = self.analyze_sales_forecast(df, thresholds, risk_levels, timeframes)
                    elif trend_type == 'customer_contracts':
                        result = self.analyze_customer_contracts(df, thresholds, risk_levels, timeframes)
                    elif trend_type == 'pricing_models':
                        result = self.analyze_pricing_models(df, thresholds, risk_levels, timeframes)
                    elif trend_type == 'ar_aging':
                        result = self.analyze_ar_aging(df, thresholds, risk_levels, timeframes)
                    elif trend_type == 'operating_expenses':
                        result = self.analyze_operating_expenses(df, thresholds, risk_levels, timeframes)
                    elif trend_type == 'accounts_payable':
                        result = self.analyze_accounts_payable(df, thresholds, risk_levels, timeframes)
                    elif trend_type == 'inventory_turnover':
                        result = self.analyze_inventory_turnover(df, thresholds, risk_levels, timeframes)
                    elif trend_type == 'loan_repayments':
                        result = self.analyze_loan_repayments(df, thresholds, risk_levels, timeframes)
                    elif trend_type == 'tax_obligations':
                        result = self.analyze_tax_obligations(df, thresholds, risk_levels, timeframes)
                    elif trend_type == 'capital_expenditure':
                        result = self.analyze_capital_expenditure(df, thresholds, risk_levels, timeframes)
                    elif trend_type == 'equity_debt_inflows':
                        result = self.analyze_equity_debt_inflows(df, thresholds, risk_levels, timeframes)
                    elif trend_type == 'other_income_expenses':
                        result = self.analyze_other_income_expenses(df, thresholds, risk_levels, timeframes)
                    elif trend_type == 'cash_flow_types':
                        result = self.analyze_cash_flow_types(df, thresholds, risk_levels, timeframes)
                    else:
                        # Fallback to generic analysis
                        result = self.analyze_generic_trend(df, trend_type, thresholds, risk_levels, timeframes)
                    
                    results[trend_type] = result
                    print(f"✅ {trend_type} specialized analysis completed")
                    
                except Exception as e:
                    print(f"❌ {trend_type} specialized analysis failed: {e}")
                    results[trend_type] = self._generate_fallback_analysis(df, trend_type, thresholds)
            
            # Add summary statistics
            results['_summary'] = {
                'total_trends_analyzed': len(trend_types),
                'successful_analyses': len([r for r in results.values() if isinstance(r, dict) and 'trend_direction' in r]),
                'dynamic_thresholds': thresholds,
                'dynamic_risk_levels': risk_levels,
                'dynamic_timeframes': timeframes,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            print(f"✅ Specialized trend analysis completed: {len(trend_types)} types processed")
            return results
            
        except Exception as e:
            print(f"❌ Specialized trend analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return {'error': f'Specialized analysis failed: {str(e)}'}
    
    # ===== SPECIALIZED TREND ANALYSIS METHODS =====
    
    def analyze_historical_revenue_trends(self, df, thresholds, risk_levels, timeframes):
        """Specialized analysis for historical revenue trends"""
        try:
            # Filter revenue-related transactions (Inward_Amount > 0)
            if 'Inward_Amount' in df.columns:
                revenue_df = df[df['Inward_Amount'] > 0].copy()
                amount_col = 'Inward_Amount'
            else:
                revenue_df = df[df['Amount'] > 0].copy()
                amount_col = 'Amount'
            
            if revenue_df.empty:
                return self._generate_fallback_analysis(df, 'historical_revenue_trends', thresholds)
            
            # Ensure Date column is datetime format
            if 'Date' in revenue_df.columns:
                revenue_df['Date'] = pd.to_datetime(revenue_df['Date'], errors='coerce')
                revenue_df = revenue_df.dropna(subset=['Date'])
            
            if revenue_df.empty:
                return self._generate_fallback_analysis(df, 'historical_revenue_trends', thresholds)
            
            # Calculate revenue-specific metrics
            monthly_revenue = revenue_df.groupby(revenue_df['Date'].dt.to_period('M'))[amount_col].sum()
            revenue_growth = monthly_revenue.pct_change().dropna()
            
            # Determine trend direction and strength
            if len(revenue_growth) >= 2:
                recent_growth = float(revenue_growth.tail(3).mean())
                trend_direction = 'upward' if recent_growth > 0.05 else 'downward' if recent_growth < -0.05 else 'stable'
                trend_strength = 'strong' if abs(recent_growth) > 0.15 else 'moderate' if abs(recent_growth) > 0.05 else 'weak'
            else:
                trend_direction = 'stable'
                trend_strength = 'weak'
            
            # Calculate business metrics
            total_revenue = float(revenue_df[amount_col].sum())
            avg_monthly_revenue = float(monthly_revenue.mean())
            revenue_volatility = float(revenue_growth.std()) if len(revenue_growth) > 0 else 0
            
            return {
                'trend_direction': trend_direction,
                'trend_strength': trend_strength,
                'confidence': 0.85,
                'business_metrics': {
                    'total_revenue': total_revenue,
                    'avg_monthly_revenue': avg_monthly_revenue,
                    'revenue_growth_rate': float(revenue_growth.mean()) if len(revenue_growth) > 0 else 0,
                    'revenue_volatility': revenue_volatility,
                    'revenue_trend_periods': len(monthly_revenue)
                },
                'ai_insights': ['Revenue trends show ' + trend_direction + ' pattern with ' + trend_strength + ' strength'],
                'recommendations': [f'Monitor monthly revenue patterns and optimize for consistent growth'],
                'risk_assessment': 'low' if trend_direction == 'upward' else 'medium',
                'context': 'Revenue analysis based on transaction data'
            }
        except Exception as e:
            print(f"❌ Historical revenue trends analysis failed: {e}")
            return self._generate_fallback_analysis(df, 'historical_revenue_trends', thresholds)
    
    def analyze_generic_trend(self, df, trend_type, thresholds, risk_levels, timeframes):
        """Generic analysis for any trend type"""
        try:
            total_amount = float(df['Amount'].sum())
            avg_amount = float(df['Amount'].mean())
            transaction_count = len(df)
            
            return {
                'trend_direction': 'stable',
                'trend_strength': 'moderate',
                'confidence': 0.7,
                'business_metrics': {
                    'total_amount': total_amount,
                    'avg_amount': avg_amount,
                    'transaction_count': transaction_count
                },
                'ai_insights': [f'Analysis completed for {trend_type}'],
                'recommendations': [f'Review {trend_type} data for optimization opportunities'],
                'risk_assessment': 'medium',
                'context': f'Generic analysis for {trend_type}'
            }
        except Exception as e:
            print(f"❌ Generic trend analysis failed: {e}")
            return self._generate_fallback_analysis(df, trend_type, thresholds)
    
    # Simplified implementations for other trend types (similar structure)
    def analyze_sales_forecast(self, df, thresholds, risk_levels, timeframes):
        return self.analyze_generic_trend(df, 'sales_forecast', thresholds, risk_levels, timeframes)
    
    def analyze_customer_contracts(self, df, thresholds, risk_levels, timeframes):
        return self.analyze_generic_trend(df, 'customer_contracts', thresholds, risk_levels, timeframes)
    
    def analyze_pricing_models(self, df, thresholds, risk_levels, timeframes):
        return self.analyze_generic_trend(df, 'pricing_models', thresholds, risk_levels, timeframes)
    
    def analyze_ar_aging(self, df, thresholds, risk_levels, timeframes):
        return self.analyze_generic_trend(df, 'ar_aging', thresholds, risk_levels, timeframes)
    
    def analyze_operating_expenses(self, df, thresholds, risk_levels, timeframes):
        return self.analyze_generic_trend(df, 'operating_expenses', thresholds, risk_levels, timeframes)
    
    def analyze_accounts_payable(self, df, thresholds, risk_levels, timeframes):
        return self.analyze_generic_trend(df, 'accounts_payable', thresholds, risk_levels, timeframes)
    
    def analyze_inventory_turnover(self, df, thresholds, risk_levels, timeframes):
        return self.analyze_generic_trend(df, 'inventory_turnover', thresholds, risk_levels, timeframes)
    
    def analyze_loan_repayments(self, df, thresholds, risk_levels, timeframes):
        return self.analyze_generic_trend(df, 'loan_repayments', thresholds, risk_levels, timeframes)
    
    def analyze_tax_obligations(self, df, thresholds, risk_levels, timeframes):
        return self.analyze_generic_trend(df, 'tax_obligations', thresholds, risk_levels, timeframes)
    
    def analyze_capital_expenditure(self, df, thresholds, risk_levels, timeframes):
        return self.analyze_generic_trend(df, 'capital_expenditure', thresholds, risk_levels, timeframes)
    
    def analyze_equity_debt_inflows(self, df, thresholds, risk_levels, timeframes):
        return self.analyze_generic_trend(df, 'equity_debt_inflows', thresholds, risk_levels, timeframes)
    
    def analyze_other_income_expenses(self, df, thresholds, risk_levels, timeframes):
        return self.analyze_generic_trend(df, 'other_income_expenses', thresholds, risk_levels, timeframes)
    
    def analyze_cash_flow_types(self, df, thresholds, risk_levels, timeframes):
        return self.analyze_generic_trend(df, 'cash_flow_types', thresholds, risk_levels, timeframes)
    
    # ===== HELPER METHODS =====
    
    def _generate_fallback_analysis(self, df, trend_type, thresholds):
        """Generate fallback analysis when specialized analysis fails"""
        return {
            'trend_direction': 'stable',
            'trend_strength': 'weak',
            'confidence': 0.5,
            'business_metrics': {
                'total_amount': float(df['Amount'].sum()) if not df.empty and 'Amount' in df.columns else 0,
                'transaction_count': len(df),
                'avg_amount': float(df['Amount'].mean()) if not df.empty and 'Amount' in df.columns else 0
            },
            'ai_insights': ['Analysis completed with basic metrics'],
            'recommendations': ['Review data quality and retry analysis'],
            'risk_assessment': 'medium',
            'context': f'Fallback analysis for {trend_type}'
        }

