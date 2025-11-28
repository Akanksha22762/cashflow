"""
AI Reasoning Routes
Routes for AI-powered reasoning and analysis
"""
from flask import Blueprint, request, jsonify, session

# Create blueprint
bp = Blueprint('ai_reasoning', __name__)

@bp.route('/ai-reasoning/categorization', methods=['POST'])
def get_ai_categorization_reasoning():
    """
    Get AI reasoning for transaction categorization from database only.
    No fallback - reasoning must be pre-generated during upload.
    
    Request body:
    {
        "transaction_description": "Coal procurement from Tata Steel",
        "category": "Operating Activities",
        "all_transactions": [...] // optional (not used)
    }
    """
    # Import here to avoid circular dependencies
    from app_setup import db_manager, DATABASE_AVAILABLE
    
    try:
        data = request.get_json()
        transaction_desc = data.get('transaction_description', '')
        category = data.get('category', '')
        all_transactions = data.get('all_transactions', None)
        
        if not transaction_desc or not category:
            return jsonify({'error': 'transaction_description and category are required'}), 400
        
        # Use modular reasoning retriever (database only, no fallback)
        from upload_modules.ai_reasoning_retriever import get_transaction_reasoning
        
        result = get_transaction_reasoning(
            transaction_desc=transaction_desc,
            category=category,
            db_manager=db_manager if DATABASE_AVAILABLE else None,
            all_transactions=all_transactions
        )
        
        if result['status'] == 'error':
            error_msg = result.get('error', 'Unknown error')
            return jsonify({'error': error_msg}), 404  # 404 for not found
        
        return jsonify(result)
    
    except Exception as e:
        print(f"❌ Error in categorization reasoning: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/ai-reasoning/vendor-landscape', methods=['POST'])
def get_ai_vendor_landscape():
    """
    Get comprehensive vendor landscape analysis with caching.
    First click generates and stores in database, subsequent clicks use cached data.
    
    Request body:
    {
        "vendors": ["Tata Steel", "Axis Bank", ...],
        "transactions": [
            {"description": "...", "amount": 1000, "vendor": "Tata Steel"},
            ...
        ]
    }
    """
    # Import here to avoid circular dependencies
    try:
        from ai_reasoning_engine import get_vendor_landscape_analysis
        AI_REASONING_AVAILABLE = True
    except ImportError:
        AI_REASONING_AVAILABLE = False
    
    if not AI_REASONING_AVAILABLE:
        return jsonify({'error': 'AI Reasoning Engine not available'}), 503
    
    from app_setup import (
        db_manager, DATABASE_AVAILABLE, state_manager, PERSISTENT_STATE_AVAILABLE
    )
    
    try:
        data = request.get_json()
        vendors = data.get('vendors', [])
        transactions = data.get('transactions', [])
        
        if not vendors or not transactions:
            return jsonify({'error': 'vendors and transactions are required'}), 400
        
        # ✅ Get current session_id for caching - use resolve_session_ids for better reliability
        session_id = None
        try:
            from vendor_modules.session_utils import resolve_session_ids
            resolved_session_id, _ = resolve_session_ids(
                session if session else {},
                state_manager if PERSISTENT_STATE_AVAILABLE else None,
                db_manager if DATABASE_AVAILABLE else None
            )
            if resolved_session_id:
                session_id = resolved_session_id
        except Exception as resolve_error:
            print(f"⚠️ Error resolving session_id: {resolve_error}")
            # Fallback to direct session check
            if session and 'mysql_session_id' in session:
                session_id = session['mysql_session_id']
            elif state_manager and PERSISTENT_STATE_AVAILABLE:
                session_id = getattr(state_manager, 'current_session_id', None)
        
        # For single vendor requests (most common case), check cache first
        if len(vendors) == 1 and session_id and db_manager and DATABASE_AVAILABLE:
            vendor_name = vendors[0]
            print(f"🔍 Checking cache for vendor: {vendor_name}, session_id: {session_id}")
            try:
                conn = db_manager.get_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT vendor_reasoning, ai_summary 
                    FROM vendor_entities 
                    WHERE vendor_name = %s 
                    AND session_id = %s 
                    AND (vendor_reasoning IS NOT NULL AND vendor_reasoning != '' 
                         OR ai_summary IS NOT NULL AND ai_summary != '')
                    ORDER BY vendor_id DESC
                    LIMIT 1
                """, (vendor_name, session_id))
                
                cached_result = cursor.fetchone()
                cached_data = None
                source_column = None
                if cached_result:
                    if cached_result[0]:  # vendor_reasoning has data
                        cached_data = cached_result[0]
                        source_column = 'vendor_reasoning'
                    elif len(cached_result) > 1 and cached_result[1]:  # fallback to ai_summary
                        cached_data = cached_result[1]
                        source_column = 'ai_summary'
                
                if cached_data:
                    import json
                    try:
                        cached_analysis = json.loads(cached_data) if isinstance(cached_data, str) else cached_data
                        expected_keys = ['concentration_analysis', 'vendor_segmentation', 'financial_insights', 'risk_assessment', 'strategic_recommendations']
                        has_expected_structure = isinstance(cached_analysis, dict) and any(key in cached_analysis for key in expected_keys)
                        
                        if has_expected_structure and 'error' not in cached_analysis:
                            print(f"✅ Using cached vendor landscape analysis for: {vendor_name}")
                            return jsonify({
                                'status': 'success',
                                'analysis': cached_analysis,
                                'cached': True
                            })
                    except json.JSONDecodeError:
                        pass
            except Exception as cache_error:
                print(f"⚠️ Cache lookup failed: {cache_error}")
        
        # Generate fresh analysis (cache miss or multi-vendor request)
        print(f"🧠 Generating fresh vendor landscape analysis for: {', '.join(vendors)}")
        analysis = get_vendor_landscape_analysis(vendors, transactions)
        
        # Store in database for future use (single vendor only)
        if len(vendors) == 1 and session_id and db_manager and DATABASE_AVAILABLE:
            try:
                import json
                vendor_name = vendors[0]
                conn = db_manager.get_connection()
                cursor = conn.cursor()
                analysis_json = json.dumps(analysis) if not isinstance(analysis, str) else analysis
                
                cursor.execute("""
                    SELECT vendor_id FROM vendor_entities 
                    WHERE vendor_name = %s AND session_id = %s
                    LIMIT 1
                """, (vendor_name, session_id))
                
                existing = cursor.fetchone()
                if existing:
                    cursor.execute("""
                        UPDATE vendor_entities 
                        SET vendor_reasoning = %s, reasoning_generated_at = CURRENT_TIMESTAMP
                        WHERE vendor_id = %s
                    """, (analysis_json, existing[0]))
                else:
                    cursor.execute("""
                        INSERT INTO vendor_entities 
                        (session_id, vendor_name, vendor_reasoning, reasoning_generated_at) 
                        VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                    """, (session_id, vendor_name, analysis_json))
                
                conn.commit()
                print(f"✅ Stored vendor landscape analysis for: {vendor_name}")
            except Exception as store_error:
                print(f"⚠️ Failed to cache vendor landscape: {store_error}")
        
        return jsonify({
            'status': 'success',
            'analysis': analysis,
            'cached': False
        })
    
    except Exception as e:
        print(f"❌ Error in vendor landscape analysis: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@bp.route('/ai-reasoning/trend-analysis', methods=['POST'])
def get_ai_trend_analysis():
    """
    Get AI-powered trend analysis with reasoning
    
    Request body:
    {
        "trend_type": "revenue_trends",
        "trends_data": {
            "revenue_trends": {...},
            "expense_trends": {...},
            ...
        },
        "analysis_summary": {...},
        "filters": {
            "date_range": "last_6_months",
            "analysis_type": "comprehensive"
        }
    }
    """
    # Import here to avoid circular dependencies
    try:
        from ai_reasoning_engine import get_trend_analysis_with_reasoning
        AI_REASONING_AVAILABLE = True
    except ImportError:
        AI_REASONING_AVAILABLE = False
    
    if not AI_REASONING_AVAILABLE:
        return jsonify({'error': 'AI Reasoning Engine not available'}), 503
    
    from app_setup import (
        db_manager, DATABASE_AVAILABLE, state_manager, PERSISTENT_STATE_AVAILABLE
    )
    
    try:
        data = request.get_json()
        trend_type = data.get('trend_type', 'general_trends')
        trends_data = data.get('trends_data', {})
        analysis_summary = data.get('analysis_summary', {})
        filters = data.get('filters', None)
        
        if not trends_data:
            return jsonify({'error': 'trends_data is required'}), 400
        
        # Check cache first
        if DATABASE_AVAILABLE and db_manager:
            try:
                from vendor_modules.session_utils import resolve_session_ids
                session_id, file_id = resolve_session_ids(
                    session if session else {},
                    state_manager if PERSISTENT_STATE_AVAILABLE else None,
                    db_manager
                )
                
                if session_id:
                    conn = db_manager.get_connection()
                    cursor = conn.cursor(dictionary=True)
                    cursor.execute("""
                        SELECT trend_results, detail_id
                        FROM trends_analysis_details
                        WHERE session_id = %s AND trend_type = %s
                        ORDER BY analysis_timestamp DESC
                        LIMIT 1
                    """, (session_id, trend_type))
                    
                    cached_result = cursor.fetchone()
                    if cached_result:
                        import json
                        trend_results = json.loads(cached_result['trend_results']) if isinstance(cached_result['trend_results'], str) else cached_result['trend_results']
                        
                        if isinstance(trend_results, dict) and 'ai_reasoning' in trend_results:
                            ai_reasoning = trend_results['ai_reasoning']
                            print(f"✅ Found cached AI reasoning for trend type: {trend_type}")
                            return jsonify({
                                'status': 'success',
                                'analysis': ai_reasoning,
                                'cached': True
                            })
            except Exception as cache_error:
                print(f"⚠️ Cache check failed: {cache_error}")
        
        # Generate fresh AI reasoning analysis
        print(f"🧠 Generating fresh AI reasoning for trend type: {trend_type}")
        analysis = get_trend_analysis_with_reasoning(trend_type, trends_data, analysis_summary, filters)
        
        # Store AI reasoning in database for future use
        if DATABASE_AVAILABLE and db_manager:
            try:
                from vendor_modules.session_utils import resolve_session_ids
                session_id, file_id = resolve_session_ids(
                    session if session else {},
                    state_manager if PERSISTENT_STATE_AVAILABLE else None,
                    db_manager
                )
                
                if session_id and file_id:
                    conn = db_manager.get_connection()
                    cursor = conn.cursor()
                    
                    cursor.execute("""
                        SELECT detail_id, trend_results
                        FROM trends_analysis_details
                        WHERE session_id = %s AND trend_type = %s
                        ORDER BY analysis_timestamp DESC
                        LIMIT 1
                    """, (session_id, trend_type))
                    
                    existing = cursor.fetchone()
                    if existing:
                        import json
                        trend_results = json.loads(existing[1]) if isinstance(existing[1], str) else existing[1]
                        if not isinstance(trend_results, dict):
                            trend_results = {}
                        trend_results['ai_reasoning'] = analysis
                        
                        cursor.execute("""
                            UPDATE trends_analysis_details
                            SET trend_results = %s
                            WHERE detail_id = %s
                        """, (json.dumps(trend_results), existing[0]))
                        conn.commit()
                        print(f"✅ Stored AI reasoning for trend type: {trend_type}")
            except Exception as store_error:
                print(f"⚠️ Failed to store AI reasoning: {store_error}")
        
        return jsonify({
            'status': 'success',
            'analysis': analysis,
            'cached': False
        })
    
    except Exception as e:
        print(f"❌ Error in trend analysis reasoning: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

