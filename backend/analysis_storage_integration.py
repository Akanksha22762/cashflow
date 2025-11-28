# Analysis Storage Integration for CASHFLOW-SAP-BANK
# This module integrates analysis results storage with the MySQL database

import json
import time
from datetime import datetime

def store_analysis_results(db_manager, file_id, analysis_session_id, analysis_type, analysis_subtype, selected_filter, result_data):
    """
    Store analysis results in the database for any analysis type
    """
    try:
        print(f"📊 Storing {analysis_type} analysis results in database...")
        
        # Extract key information from result_data
        summary = {
            'status': result_data.get('status', 'success'),
            'data_source': result_data.get('data_source', 'unknown'),
            'total_transactions': result_data.get('transaction_count', 0),
            'processing_time': result_data.get('processing_time', 0.0)
        }
        
        # Get reasoning explanations
        reasoning_explanations = result_data.get('reasoning_explanations', {})
        
        # Extract individual reasoning components
        simple_reasoning = reasoning_explanations.get('simple_reasoning', '')
        training_insights = reasoning_explanations.get('training_insights', '')
        ml_analysis = reasoning_explanations.get('ml_analysis', {})
        ai_analysis = reasoning_explanations.get('ai_analysis', {})
        hybrid_analysis = reasoning_explanations.get('hybrid_analysis', {})
        client_explanations = reasoning_explanations.get('client_explanations', {})
        
        # Get AI insights and recommendations
        ai_insights = result_data.get('insights', '') or result_data.get('simple_reasoning', '')
        recommendations = result_data.get('recommendations', '') or result_data.get('strategic_recommendations', '')
        
        # Get performance metrics
        confidence_score = result_data.get('confidence_score', 0.0)
        success_rate = result_data.get('success_rate', 0.0)
        ai_model_used = result_data.get('ai_model', 'hybrid')
        
        # Store in analysis_results table
        cursor = db_manager.get_connection().cursor()
        
        insert_query = """
        INSERT INTO analysis_results (
            file_id, analysis_session_id, analysis_type, analysis_subtype, selected_filter,
            result_summary, detailed_results, ai_insights, recommendations, reasoning_explanations,
            simple_reasoning, training_insights, ml_analysis, ai_analysis, hybrid_analysis,
            client_explanations, processing_time, confidence_score, success_rate,
            transaction_count, ai_model_used
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        cursor.execute(insert_query, (
            file_id, analysis_session_id, analysis_type, analysis_subtype, selected_filter,
            json.dumps(summary), json.dumps(result_data), ai_insights, recommendations,
            json.dumps(reasoning_explanations), simple_reasoning, training_insights,
            json.dumps(ml_analysis), json.dumps(ai_analysis), json.dumps(hybrid_analysis),
            json.dumps(client_explanations), result_data.get('processing_time', 0.0),
            confidence_score, success_rate, result_data.get('transaction_count', 0),
            ai_model_used
        ))
        
        db_manager.get_connection().commit()
        result_id = cursor.lastrowid
        
        print(f"SUCCESS: Stored {analysis_type} analysis results with ID: {result_id}")
        return result_id
        
    except Exception as e:
        print(f"ERROR: Failed to store analysis results: {e}")
        return None

def store_vendor_analysis(db_manager, file_id, vendor_data):
    """
    Store vendor analysis results in vendor_analysis table
    """
    try:
        print(f"🏢 Storing vendor analysis data in database...")
        
        cursor = db_manager.get_connection().cursor()
        
        for vendor_name, data in vendor_data.items():
            if isinstance(data, dict):
                # Extract vendor metrics
                total_transactions = data.get('transaction_count', 0)
                total_amount = data.get('total_amount', 0.0)
                average_amount = data.get('avg_amount', 0.0)
                max_amount = data.get('max_amount', 0.0)
                min_amount = data.get('min_amount', 0.0)
                
                # Determine vendor category
                vendor_category = data.get('business_type', 'Unknown')
                
                # Calculate risk and reliability scores
                risk_score = data.get('risk_score', 50.0)
                reliability_score = data.get('reliability_score', 75.0)
                ai_confidence = data.get('confidence_score', 0.8) * 100
                
                # Determine business impact
                if total_amount > 1000000:
                    business_impact = 'critical'
                elif total_amount > 500000:
                    business_impact = 'high'
                elif total_amount > 100000:
                    business_impact = 'medium'
                else:
                    business_impact = 'low'
                
                # Extract AI insights
                ai_insights = data.get('simple_reasoning', '') or data.get('insights', '')
                
                insert_query = """
                INSERT INTO vendor_analysis (
                    file_id, vendor_name, vendor_category, total_transactions, total_amount,
                    average_amount, max_amount, min_amount, risk_score, reliability_score,
                    ai_confidence, business_impact, notes
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    total_transactions = VALUES(total_transactions),
                    total_amount = VALUES(total_amount),
                    average_amount = VALUES(average_amount),
                    max_amount = VALUES(max_amount),
                    min_amount = VALUES(min_amount),
                    risk_score = VALUES(risk_score),
                    reliability_score = VALUES(reliability_score),
                    ai_confidence = VALUES(ai_confidence),
                    business_impact = VALUES(business_impact),
                    notes = VALUES(notes),
                    updated_at = CURRENT_TIMESTAMP
                """
                
                cursor.execute(insert_query, (
                    file_id, vendor_name, vendor_category, total_transactions, total_amount,
                    average_amount, max_amount, min_amount, risk_score, reliability_score,
                    ai_confidence, business_impact, ai_insights
                ))
        
        db_manager.get_connection().commit()
        print(f"SUCCESS: Stored vendor analysis for {len(vendor_data)} vendors")
        
    except Exception as e:
        print(f"ERROR: Failed to store vendor analysis: {e}")

def store_category_insights(db_manager, file_id, analysis_session_id, category_data):
    """
    Store category insights in category_insights table
    """
    try:
        print(f"INFO: Storing category insights in database...")
        
        cursor = db_manager.get_connection().cursor()
        
        # If category_data is a dict with categories as keys
        if isinstance(category_data, dict):
            for category_name, data in category_data.items():
                if isinstance(data, dict):
                    transaction_count = data.get('count', 0)
                    total_amount = data.get('total', 0.0)
                    average_amount = total_amount / max(transaction_count, 1)
                    
                    # Calculate percentage of total (approximate)
                    percentage_of_total = data.get('percentage', 0.0)
                    
                    # Extract AI insights
                    ai_insights = data.get('insights', '') or data.get('simple_reasoning', '')
                    recommendations = data.get('recommendations', '')
                    
                    # Determine business criticality based on amount
                    if 'operating' in category_name.lower():
                        business_criticality = 'essential'
                    elif total_amount > 1000000:
                        business_criticality = 'high'
                    elif total_amount > 500000:
                        business_criticality = 'medium'
                    else:
                        business_criticality = 'low'
                    
                    # Fixed: Use only existing columns and proper JSON format
                    insert_query = """
                    INSERT INTO category_insights (
                        file_id, category_name, transaction_count, total_amount,
                        average_amount, percentage_of_total, ai_model_used,
                        ai_confidence, insights
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        transaction_count = VALUES(transaction_count),
                        total_amount = VALUES(total_amount),
                        average_amount = VALUES(average_amount),
                        percentage_of_total = VALUES(percentage_of_total),
                        ai_confidence = VALUES(ai_confidence),
                        insights = VALUES(insights),
                        updated_at = CURRENT_TIMESTAMP
                    """
                    
                    # Prepare JSON insights
                    insights_json = {
                        'business_criticality': business_criticality,
                        'recommendations': recommendations or 'Category analysis completed',
                        'ai_reasoning': ai_insights or f'Analysis of {category_name} transactions',
                        'data_source': 'upload_analysis'
                    }
                    
                    cursor.execute(insert_query, (
                        file_id, category_name, transaction_count, total_amount,
                        average_amount, percentage_of_total, 'openai',
                        85.0, json.dumps(insights_json)
                    ))
        
        db_manager.get_connection().commit()
        print(f"SUCCESS: Stored category insights for {len(category_data)} categories")
        
    except Exception as e:
        print(f"ERROR: Failed to store category insights: {e}")

def store_business_insights(db_manager, file_id, analysis_session_id, insight_type, insight_data):
    """
    Store business insights (cash flow, payment schedule, collection status)
    """
    try:
        print(f"INFO: Storing {insight_type} business insights in database...")
        
        cursor = db_manager.get_connection().cursor()
        
        # Extract key metrics based on insight type
        if insight_type == 'cash_flow_status':
            insight_name = "Cash Flow Analysis"
            insight_value = insight_data.get('net_cash_flow', 0.0)
            insight_text = f"Net cash flow: ${insight_value:,.2f} (Inflow: ${insight_data.get('total_inflow', 0):,.2f}, Outflow: ${insight_data.get('total_outflow', 0):,.2f})"
            risk_level = insight_data.get('risk_level', 'low')
            trend_indicator = 'improving' if insight_value > 0 else 'declining'
        
        elif insight_type == 'payment_schedule':
            insight_name = "Payment Schedule Analysis"
            insight_value = insight_data.get('total_upcoming', 0.0)
            overdue_count = insight_data.get('overdue_count', 0)
            insight_text = f"Upcoming payments: ${insight_value:,.2f}, Overdue payments: {overdue_count}"
            risk_level = 'high' if overdue_count > 5 else 'medium' if overdue_count > 0 else 'low'
            trend_indicator = 'stable'
        
        elif insight_type == 'collection_status':
            insight_name = "Collection Status Analysis"
            insight_value = insight_data.get('total_outstanding', 0.0)
            delayed_count = insight_data.get('delayed_count', 0)
            insight_text = f"Outstanding collections: ${insight_value:,.2f}, Delayed collections: {delayed_count}"
            risk_level = 'high' if delayed_count > 5 else 'medium' if delayed_count > 0 else 'low'
            trend_indicator = 'stable'
        
        else:
            insight_name = f"{insight_type.title()} Analysis"
            insight_value = 0.0
            insight_text = str(insight_data)
            risk_level = 'medium'
            trend_indicator = 'stable'
        
        # Determine action required and priority
        action_required = risk_level in ['high', 'critical']
        priority_level = 'urgent' if risk_level == 'critical' else 'high' if risk_level == 'high' else 'medium'
        
        # Generate action items and recommendations
        recommendations = insight_data.get('recommendations', f'Monitor {insight_type} closely')
        
        insert_query = """
        INSERT INTO business_insights (
            file_id, analysis_session_id, insight_type, insight_name, insight_value,
            insight_text, insight_data, risk_level, priority_level, action_required, 
            action_items, confidence_score, trend_indicator, ai_recommendations
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        cursor.execute(insert_query, (
            file_id, analysis_session_id, insight_type, insight_name,
            insight_value, insight_text, json.dumps(insight_data), risk_level,
            priority_level, action_required, recommendations, 85.0, trend_indicator, recommendations
        ))
        
        db_manager.get_connection().commit()
        insight_id = cursor.lastrowid
        
        print(f"SUCCESS: Stored {insight_type} business insight with ID: {insight_id}")
        return insight_id
        
    except Exception as e:
        print(f"ERROR: Failed to store business insights: {e}")
        return None

def store_ui_interaction(db_manager, file_id, analysis_session_id, user_session, interaction_type, element_clicked, element_value=None):
    """
    Store UI interaction for analytics
    """
    try:
        cursor = db_manager.get_connection().cursor()
        
        insert_query = """
        INSERT INTO ui_interactions (
            file_id, analysis_session_id, user_session, interaction_type,
            element_clicked, element_value, page_section
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        
        # Determine page section from element clicked
        page_section = 'unknown'
        if 'categories' in element_clicked.lower():
            page_section = 'categories_section'
        elif 'vendor' in element_clicked.lower():
            page_section = 'vendors_section'
        elif 'business' in element_clicked.lower():
            page_section = 'business_insights'
        elif 'trend' in element_clicked.lower():
            page_section = 'trends_section'
        
        cursor.execute(insert_query, (
            file_id, analysis_session_id, user_session, interaction_type,
            element_clicked, element_value, page_section
        ))
        
        db_manager.get_connection().commit()
        
    except Exception as e:
        print(f"WARNING: Failed to store UI interaction: {e}")

# Integration helper function
def integrate_analysis_with_database(db_manager, file_id, analysis_session_id, analysis_type, result_data):
    """
    Main integration function to store any analysis results
    """
    try:
        # Store in main analysis_results table
        result_id = store_analysis_results(
            db_manager, file_id, analysis_session_id, analysis_type, 
            result_data.get('analysis_subtype', 'general'),
            result_data.get('selected_filter', 'all'),
            result_data
        )
        
        # Store specific analysis data in specialized tables
        if analysis_type == 'vendors' and 'vendor_analysis' in result_data:
            store_vendor_analysis(db_manager, file_id, result_data['vendor_analysis'])
        
        elif analysis_type == 'categories' and 'category_breakdown' in result_data:
            store_category_insights(db_manager, file_id, analysis_session_id, result_data['category_breakdown'])
        
        # Store business insights if present
        if 'business_insights' in result_data:
            for insight_type, insight_data in result_data['business_insights'].items():
                store_business_insights(db_manager, file_id, analysis_session_id, insight_type, insight_data)
        
        print(f"SUCCESS: Complete analysis integration for {analysis_type}")
        return result_id
        
    except Exception as e:
        print(f"ERROR: Analysis integration failed: {e}")
        return None
