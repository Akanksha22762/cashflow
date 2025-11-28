"""
System Routes
Simple routes for API home and status
"""
from flask import Blueprint, jsonify, session
from datetime import datetime
import os
import time

# Create blueprint
bp = Blueprint('system', __name__)

@bp.route('/')
def home():
    """API Home - Returns API info and available endpoints"""
    return jsonify({
        'status': 'success',
        'message': 'Cash Flow Analysis API - Backend Server',
        'version': '2.0',
        'frontend_url': 'http://13.126.18.17:3000',
        'api_endpoints': {
            'upload': '/upload',
            'status': '/status',
            'vendor_analysis': '/vendor-analysis',
            'transaction_analysis': '/transaction-analysis',
            'cash_flow_forecast': '/cash-flow-forecast',
            'anomaly_detection': '/anomaly-detection'
        },
        'documentation': 'Visit http://13.126.18.17:3000 for the frontend interface'
    })


@bp.route('/status', methods=['GET'])
def check_status():
    """Enhanced status endpoint with performance metrics and system health"""
    # Import here to avoid circular dependencies
    from app_setup import (
        reconciliation_data, performance_monitor, ai_cache_manager,
        CACHE_TTL, DATA_FOLDER, DATABASE_AVAILABLE, OPENAI_AVAILABLE
    )
    import pandas as pd
    
    start_time = time.time()
    
    try:
        # Check OpenAI API availability
        openai_available = bool(os.getenv('OPENAI_API_KEY'))
        
        # Get performance metrics
        performance_metrics = performance_monitor.get_metrics() if performance_monitor else {}
        
        status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '2.0.0',
            'data_folder_exists': os.path.exists(DATA_FOLDER) if DATA_FOLDER else False,
            'reconciliation_completed': bool(reconciliation_data),
            'available_reports': list(reconciliation_data.keys()) if reconciliation_data else [],
            'openai_api_available': openai_available,
            'openai_status': 'Connected' if openai_available else 'Not configured (set OPENAI_API_KEY environment variable)',
            'performance': performance_metrics,
            'cache_info': {
                'size': len(ai_cache_manager.cache) if ai_cache_manager else 0,
                'ttl_seconds': CACHE_TTL
            },
            'system_info': {
                'python_version': '3.8+',
                'flask_version': '2.0+',
                'pandas_version': pd.__version__
            }
        }
        
        # Record successful request
        processing_time = time.time() - start_time
        if performance_monitor:
            performance_monitor.record_request(processing_time, success=True)
        
        return jsonify(status)
        
    except Exception as e:
        processing_time = time.time() - start_time
        if performance_monitor:
            performance_monitor.record_request(processing_time, success=False)
        
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

