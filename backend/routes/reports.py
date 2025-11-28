"""
Report Routes
Routes for generating comprehensive reports
"""
from flask import Blueprint, jsonify, send_file, session
from datetime import datetime
from io import BytesIO
import pandas as pd
import pandas as pd

# Import report helpers with error handling
try:
    from reports.report_helpers import (
        build_comprehensive_report_payload as build_report_payload,
        format_currency,
    )
    REPORT_HELPERS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Could not import report_helpers: {e}")
    REPORT_HELPERS_AVAILABLE = False
    # Create dummy functions to prevent crashes
    def build_report_payload(*args, **kwargs):
        raise ValueError("Report helpers not available")
    def format_currency(amount):
        return f"${amount:,.2f}"

# Create blueprint
bp = Blueprint('reports', __name__)

def _render_comprehensive_report_pdf(report_payload):
    """Render comprehensive report as PDF"""
    # Import here to avoid circular dependencies
    from app_setup import REPORTLAB_AVAILABLE
    
    if not REPORTLAB_AVAILABLE:
        buffer = BytesIO()
        buffer.write(b"ReportLab is not installed on this server.")
        buffer.seek(0)
        return buffer

    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    summary = report_payload.get('summary', {})
    elements.append(Paragraph("Comprehensive Cash Flow Report", styles['Title']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Generated At: {report_payload.get('generated_at', '')}", styles['Normal']))
    elements.append(Spacer(1, 18))

    elements.append(Paragraph("Executive Summary", styles['Heading2']))
    exec_summary = report_payload.get('executive_summary', {})
    summary_lines = [
        f"Total Transactions: {exec_summary.get('total_transactions', 0):,}",
        f"Total Inflows: {format_currency(exec_summary.get('total_inflows', 0))}",
        f"Total Outflows: {format_currency(exec_summary.get('total_outflows', 0))}",
        f"Net Cash Flow: {format_currency(exec_summary.get('net_cash_flow', 0))}",
    ]
    for line in summary_lines:
        elements.append(Paragraph(line, styles['Normal']))
    elements.append(Spacer(1, 18))

    category_analysis = report_payload.get('category_analysis', [])
    if category_analysis:
        elements.append(Paragraph("Category Breakdown", styles['Heading2']))
        category_table_data = [["Category", "Inflows", "Outflows", "Net", "Transactions"]]
        for row in category_analysis:
            category_table_data.append([
                row.get('category', ''),
                format_currency(row.get('inflows', 0)),
                format_currency(row.get('outflows', 0)),
                format_currency(row.get('net', 0)),
                row.get('transaction_count', 0),
            ])
        category_table = Table(category_table_data, hAlign='LEFT')
        category_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
        ]))
        elements.append(category_table)
        elements.append(Spacer(1, 18))

    vendor_analysis = report_payload.get('vendor_analysis', [])[:10]
    if vendor_analysis:
        elements.append(Paragraph("Top Vendors by Net Impact", styles['Heading2']))
        vendor_table_data = [["Vendor", "Inflows", "Outflows", "Net", "Transactions"]]
        for vendor in vendor_analysis:
            vendor_table_data.append([
                vendor.get('vendor_name', ''),
                format_currency(vendor.get('inflows', 0)),
                format_currency(vendor.get('outflows', 0)),
                format_currency(vendor.get('net_cash_flow', 0)),
                vendor.get('transaction_count', 0),
            ])
        vendor_table = Table(vendor_table_data, hAlign='LEFT')
        vendor_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
        ]))
        elements.append(vendor_table)
        elements.append(Spacer(1, 18))

    # ✅ Add Cash Flow Statement to PDF
    cashflow_stmt = report_payload.get('cashflow_statement')
    if cashflow_stmt:
        elements.append(Paragraph("CASH FLOW STATEMENT – DIRECT METHOD", styles['Heading2']))
        if cashflow_stmt.get('period'):
            elements.append(Paragraph(f"For the Period: {cashflow_stmt.get('period')}", styles['Normal']))
        elements.append(Spacer(1, 12))
        
        # Operating Activities
        op_act = cashflow_stmt.get('operating_activities', {})
        elements.append(Paragraph("A. CASH FLOWS FROM OPERATING ACTIVITIES", styles['Heading3']))
        
        # Operating Inflows
        op_inflows = op_act.get('inflow_items', {})
        for item, amount in op_inflows.items():
            elements.append(Paragraph(f"   {item}: {amount:,.2f}", styles['Normal']))
        if op_inflows:
            elements.append(Paragraph(f"   Total Cash Inflows: {op_act.get('total_inflows', 0):,.2f}", styles['Normal']))
            elements.append(Spacer(1, 6))
        
        # Operating Outflows
        op_outflows = op_act.get('outflow_items', {})
        for item, amount in op_outflows.items():
            elements.append(Paragraph(f"   {item}: {amount:,.2f}", styles['Normal']))
        if op_outflows:
            elements.append(Paragraph(f"   Total Cash Outflows: {op_act.get('total_outflows', 0):,.2f}", styles['Normal']))
        
        net_op = op_act.get('net_cash_flow', 0)
        elements.append(Paragraph(f"   Net Cash from Operating Activities: {net_op:,.2f}", styles['Normal']))
        elements.append(Spacer(1, 12))
        
        # Investing Activities
        inv_act = cashflow_stmt.get('investing_activities', {})
        elements.append(Paragraph("B. CASH FLOWS FROM INVESTING ACTIVITIES", styles['Heading3']))
        inv_outflows = inv_act.get('outflow_items', {})
        for item, amount in inv_outflows.items():
            elements.append(Paragraph(f"   {item}: {amount:,.2f}", styles['Normal']))
        net_inv = inv_act.get('net_cash_flow', 0)
        elements.append(Paragraph(f"   Net Cash from Investing Activities: {net_inv:,.2f}", styles['Normal']))
        elements.append(Spacer(1, 12))
        
        # Financing Activities
        fin_act = cashflow_stmt.get('financing_activities', {})
        elements.append(Paragraph("C. CASH FLOWS FROM FINANCING ACTIVITIES", styles['Heading3']))
        fin_inflows = fin_act.get('inflow_items', {})
        for item, amount in fin_inflows.items():
            elements.append(Paragraph(f"   {item}: {amount:,.2f}", styles['Normal']))
        fin_outflows = fin_act.get('outflow_items', {})
        for item, amount in fin_outflows.items():
            elements.append(Paragraph(f"   {item}: {amount:,.2f}", styles['Normal']))
        net_fin = fin_act.get('net_cash_flow', 0)
        elements.append(Paragraph(f"   Net Cash from Financing Activities: {net_fin:,.2f}", styles['Normal']))
        elements.append(Spacer(1, 12))
        
        # Summary
        net_increase = cashflow_stmt.get('net_increase_in_cash', 0)
        opening_bal = cashflow_stmt.get('opening_cash_balance')
        closing_bal = cashflow_stmt.get('closing_cash_balance')
        
        elements.append(Paragraph(f"Net Increase in Cash: {net_increase:,.2f}", styles['Normal']))
        if opening_bal is not None:
            elements.append(Paragraph(f"Opening Cash Balance: {opening_bal:,.2f}", styles['Normal']))
        if closing_bal is not None:
            elements.append(Paragraph(f"Closing Cash Balance: {closing_bal:,.2f}", styles['Normal']))
        elements.append(Spacer(1, 18))

    doc.build(elements)
    buffer.seek(0)
    return buffer


@bp.route('/comprehensive-report', methods=['GET', 'OPTIONS'])
def get_comprehensive_report():
    """Get comprehensive report as JSON"""
    # Import here to avoid circular dependencies
    from flask import session, request, after_this_request, Response
    import traceback
    
    # Handle OPTIONS preflight request
    if request.method == 'OPTIONS':
        response = Response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Accept'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response.headers['Access-Control-Max-Age'] = '3600'
        return response, 200
    
    # Add CORS headers to prevent browser issues
    @after_this_request
    def add_cors_headers(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Accept'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        return response
    
    try:
        if not REPORT_HELPERS_AVAILABLE:
            return jsonify({
                'success': False, 
                'error': 'Report helpers module not available. Please check backend logs.'
            }), 503
        
        print("📊 [REPORTS] Starting comprehensive report generation...")
        print(f"📊 [REPORTS] Request from: {request.remote_addr}")
        
        from app_setup import (
            state_manager, PERSISTENT_STATE_AVAILABLE,
            db_manager, DATABASE_AVAILABLE, report_storage, REPORT_STORAGE_AVAILABLE,
            _get_active_bank_dataframe
        )
        
        print("📊 [REPORTS] Getting active bank dataframe...")
        bank_df = _get_active_bank_dataframe()
        print(f"📊 [REPORTS] Got bank dataframe: {len(bank_df)} rows")
        
        print("📊 [REPORTS] Building report payload...")
        try:
            payload = build_report_payload(
                bank_df=bank_df,
                session_obj=session,
                state_manager=state_manager if PERSISTENT_STATE_AVAILABLE else None,
                db_manager=db_manager if DATABASE_AVAILABLE else None,
                database_available=DATABASE_AVAILABLE,
                persistent_state_available=PERSISTENT_STATE_AVAILABLE
            )
            print("📊 [REPORTS] Report payload built successfully")
            
            # ✅ CRITICAL: Validate summary contains numeric values (not NaN)
            if 'summary' in payload and isinstance(payload['summary'], dict):
                for key in ['total_inflows', 'total_outflows', 'net_cash_flow', 'closing_balance', 'opening_balance']:
                    if key in payload['summary']:
                        value = payload['summary'][key]
                        if isinstance(value, float):
                            import math
                            if math.isnan(value) or math.isinf(value):
                                print(f"⚠️ [REPORTS] Converting NaN/Inf in summary.{key} to 0")
                                payload['summary'][key] = 0.0
                        
        except Exception as payload_error:
            print(f"❌ Error building report payload: {payload_error}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': f'Failed to build report payload: {str(payload_error)}'
            }), 500
        
        # Ensure payload has success field
        if not isinstance(payload, dict):
            payload = {'success': False, 'error': 'Invalid payload format'}
        elif 'success' not in payload:
            payload['success'] = True
        
        # Clean payload for JSON serialization (handle pandas/numpy types)
        import json
        import math
        import numpy as np
        
        # ✅ ALWAYS clean payload for JSON serialization (don't wait for error)
        print("📊 [REPORTS] Cleaning payload for JSON serialization...")
        
        def clean_for_json(obj):
            """Recursively clean object for JSON serialization, handling NaN, Inf, numpy types, etc."""
            # Handle dictionaries
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            
            # Handle lists
            elif isinstance(obj, list):
                return [clean_for_json(item) for item in obj]
            
            # Handle numpy scalars (int64, float64, etc.)
            elif hasattr(obj, 'item') and hasattr(obj, 'dtype'):
                val = obj.item()
                # Check for NaN/Inf after extracting value
                if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                    return None
                return val
            
            # Handle pandas NaT, NaN
            elif pd.isna(obj) if hasattr(pd, 'isna') else False:
                return None
            
            # Handle float NaN/Inf
            elif isinstance(obj, float):
                if math.isnan(obj) or math.isinf(obj):
                    return None
                return obj
            
            # Handle numpy NaN/Inf directly
            elif isinstance(obj, (np.floating, np.integer)):
                val = float(obj) if isinstance(obj, np.floating) else int(obj)
                if math.isnan(val) or math.isinf(val):
                    return None
                return val
            
            # Handle datetime objects
            elif hasattr(obj, 'isoformat'):
                return obj.isoformat()
            
            # Default: return as-is (will be caught by json.dumps if invalid)
            return obj
        
        # Always clean the payload
        payload = clean_for_json(payload)
        
        # Validate JSON serialization
        try:
            json.dumps(payload)
        except (TypeError, ValueError) as json_error:
            print(f"⚠️ JSON serialization error after cleaning: {json_error}")
            # Last resort: convert remaining invalid types to strings
            def final_clean(obj):
                if isinstance(obj, dict):
                    return {k: final_clean(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [final_clean(item) for item in obj]
                else:
                    try:
                        json.dumps(obj)
                        return obj
                    except (TypeError, ValueError):
                        return str(obj) if obj is not None else None
            payload = final_clean(payload)
        
        # ✅ Store report in background to prevent blocking response
        if REPORT_STORAGE_AVAILABLE and report_storage:
            import threading
            def store_async():
                try:
                    report_storage.store_json_report(
                        payload,
                        metadata={'source': payload.get('source')}
                    )
                except Exception as storage_error:
                    print(f"⚠️ Failed to persist report JSON (background): {storage_error}")
            
            # Start background thread (non-blocking)
            storage_thread = threading.Thread(target=store_async, daemon=True)
            storage_thread.start()
            print("📊 [REPORTS] S3 storage queued in background (non-blocking)")
        
        print("📊 [REPORTS] Returning JSON response...")
        print(f"📊 [REPORTS] Payload keys: {list(payload.keys())}")
        print(f"📊 [REPORTS] Success field: {payload.get('success')}")
        
        # ✅ SIMPLE return like CashflowDemo - Flask handles everything
        return jsonify(payload)
    except ValueError as ve:
        import traceback
        print(f"❌ Comprehensive report ValueError: {ve}")
        traceback.print_exc()
        response = jsonify({'success': False, 'error': str(ve)})
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response, 400
    except Exception as exc:
        import traceback
        print(f"❌ Comprehensive report error: {exc}")
        traceback.print_exc()
        response = jsonify({'success': False, 'error': f'Failed to generate comprehensive report: {str(exc)}'})
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response, 500


@bp.route('/comprehensive-report.pdf', methods=['GET'])
def download_comprehensive_report_pdf():
    """Download comprehensive report as PDF"""
    # Import here to avoid circular dependencies
    from flask import session
    from app_setup import (
        state_manager, PERSISTENT_STATE_AVAILABLE,
        db_manager, DATABASE_AVAILABLE, report_storage, REPORT_STORAGE_AVAILABLE,
        _get_active_bank_dataframe
    )
    
    try:
        bank_df = _get_active_bank_dataframe()
        payload = build_report_payload(
            bank_df=bank_df,
            session_obj=session,
            state_manager=state_manager if PERSISTENT_STATE_AVAILABLE else None,
            db_manager=db_manager if DATABASE_AVAILABLE else None,
            database_available=DATABASE_AVAILABLE,
            persistent_state_available=PERSISTENT_STATE_AVAILABLE
        )
        pdf_buffer = _render_comprehensive_report_pdf(payload)
        # Store PDF report (non-blocking - don't fail the request if storage fails)
        if REPORT_STORAGE_AVAILABLE and report_storage:
            try:
                print("📊 [REPORTS] Storing PDF report to S3...")
                pdf_bytes = pdf_buffer.getvalue()
                report_storage.store_pdf_report(
                    pdf_bytes,
                    metadata={'source': payload.get('source')}
                )
                pdf_buffer = BytesIO(pdf_bytes)
                print("📊 [REPORTS] PDF report stored to S3 successfully")
            except Exception as storage_error:
                # Log error but don't fail the request
                print(f"⚠️ Failed to persist report PDF: {storage_error}")
                import traceback
                traceback.print_exc()
                print("📊 [REPORTS] Continuing without storage (PDF will still be returned)")
                pdf_buffer.seek(0)
        filename = f"comprehensive_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )
    except ValueError as ve:
        import traceback
        print(f"❌ Comprehensive report PDF ValueError: {ve}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(ve)}), 400
    except Exception as exc:
        import traceback
        print(f"❌ Comprehensive report PDF error: {exc}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Failed to generate PDF report: {str(exc)}'}), 500

