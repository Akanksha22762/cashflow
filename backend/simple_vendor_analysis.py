#!/usr/bin/env python3
"""
Simple Vendor Analysis - Works exactly like Categories
"""

from flask import jsonify, request

def simple_vendor_analysis():
    """Simple vendor analysis that works exactly like categories"""
    try:
        data = request.get_json()
        vendor = data.get('vendor', '')
        
        print(f"🏢 Processing simple vendor analysis: {vendor}")
        
        # Load bank data - same as categories
        global uploaded_data
        if not uploaded_data or 'bank_df' not in uploaded_data:
            return jsonify({'error': 'No bank data uploaded yet'}), 400
        
        bank_df = uploaded_data['bank_df']
        if bank_df is None or bank_df.empty:
            return jsonify({'error': 'Uploaded bank data is empty'}), 400
        
        # Simple vendor filtering - just like categories
        if not vendor or vendor == 'all':
            # Show all transactions if no specific vendor
            filtered_transactions = bank_df
            vendor_name = 'All Vendors'
        else:
            # Filter by vendor name in description
            filtered_transactions = bank_df[bank_df['Description'].str.contains(vendor, case=False, na=False)]
            vendor_name = vendor
            
        if filtered_transactions.empty:
            return jsonify({
                'status': 'error',
                'error': f'No transactions found for vendor: {vendor}'
            })
        
        # Calculate simple metrics - exactly like categories
        total_amount = filtered_transactions['Amount'].sum()
        transaction_count = len(filtered_transactions)
        avg_amount = filtered_transactions['Amount'].mean()
        
        # Simple response - same structure as categories
        response_data = {
            'status': 'success',
            'data': {
                'vendor_name': vendor_name,
                'total_amount': float(total_amount),
                'transaction_count': int(transaction_count),
                'avg_amount': float(avg_amount),
                'cash_flow_status': 'Positive' if total_amount > 0 else 'Negative',
                'analysis_summary': {
                    'total_transactions': int(transaction_count),
                    'net_cash_flow': float(total_amount),
                    'avg_transaction': float(avg_amount)
                }
            },
            'reasoning_explanations': {
                'simple_reasoning': f'Analysis of {vendor_name}: {transaction_count} transactions totaling ₹{total_amount:,.2f}',
                'training_insights': f'AI analyzed {transaction_count} transactions for {vendor_name}',
                'ml_analysis': {
                    'pattern_analysis': {
                        'trend_direction': 'positive' if total_amount > 0 else 'negative',
                        'pattern_strength': 'strong' if transaction_count > 10 else 'moderate'
                    }
                },
                'ai_analysis': {
                    'business_intelligence': {
                        'financial_knowledge': f'{vendor_name} shows {"healthy" if total_amount > 0 else "concerning"} cash flow'
                    }
                }
            },
            'message': f'Vendor analysis completed for {vendor_name}'
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Simple vendor analysis error: {e}")
        return jsonify({'error': str(e)}), 500
