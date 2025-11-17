"""
Test Script for AI Reasoning API Endpoints
Run this to test the new AI reasoning functionality
"""

import requests
import json

BASE_URL = "http://localhost:5000"

def test_ai_reasoning_status():
    """Test AI Reasoning Engine status"""
    print("\n" + "="*60)
    print("Testing AI Reasoning Status...")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/ai-reasoning/status")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_categorization_reasoning():
    """Test transaction categorization reasoning"""
    print("\n" + "="*60)
    print("Testing Categorization Reasoning...")
    print("="*60)
    
    data = {
        "transaction_description": "Coal procurement from Tata Steel for power plant operations",
        "category": "Operating Activities"
    }
    
    response = requests.post(f"{BASE_URL}/ai-reasoning/categorization", json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_vendor_extraction_reasoning():
    """Test vendor extraction reasoning"""
    print("\n" + "="*60)
    print("Testing Vendor Extraction Reasoning...")
    print("="*60)
    
    data = {
        "transaction_description": "Payment to Axis Bank for loan interest",
        "extracted_vendor": "Axis Bank"
    }
    
    response = requests.post(f"{BASE_URL}/ai-reasoning/vendor-extraction", json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_trend_analysis():
    """Test trend analysis with reasoning"""
    print("\n" + "="*60)
    print("Testing Trend Analysis...")
    print("="*60)
    
    data = {
        "trend_type": "revenue_trends",
        "transactions": [
            {"description": "Energy sale to Gujarat DISCOM", "amount": 5000000, "date": "2024-01-15", "category": "Operating Activities"},
            {"description": "Coal procurement from Tata Steel", "amount": -2000000, "date": "2024-01-20", "category": "Operating Activities"},
            {"description": "Equipment upgrade payment", "amount": -3000000, "date": "2024-02-10", "category": "Investing Activities"},
            {"description": "Loan interest to Axis Bank", "amount": -500000, "date": "2024-02-25", "category": "Financing Activities"}
        ],
        "filters": {
            "date_range": "last_6_months"
        }
    }
    
    response = requests.post(f"{BASE_URL}/ai-reasoning/trend-analysis", json=data)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Analysis Type: {result.get('status')}")
        if 'analysis' in result:
            analysis = result['analysis']
            print(f"\nTimestamp: {analysis.get('timestamp', 'N/A')}")
            print(f"Analysis Type: {analysis.get('analysis_type', 'N/A')}")
            print("\nKey Insights:")
            for key, value in analysis.items():
                if key not in ['timestamp', 'analysis_type', 'data_summary']:
                    print(f"  - {key}: {str(value)[:200]}...")


def test_ai_system_explanation():
    """Test AI system explanation"""
    print("\n" + "="*60)
    print("Testing AI System Explanation...")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/ai-reasoning/explain-system")
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)[:1000]}...")


def test_full_analysis():
    """Test full AI reasoning analysis"""
    print("\n" + "="*60)
    print("Testing Full AI Reasoning Analysis...")
    print("="*60)
    
    data = {
        "transactions": [
            {
                "description": "Energy sale to Gujarat DISCOM",
                "amount": 5000000,
                "date": "2024-01-15",
                "category": "Operating Activities",
                "vendor": "Gujarat DISCOM"
            },
            {
                "description": "Coal procurement from Tata Steel",
                "amount": -2000000,
                "date": "2024-01-20",
                "category": "Operating Activities",
                "vendor": "Tata Steel"
            },
            {
                "description": "Equipment upgrade payment",
                "amount": -3000000,
                "date": "2024-02-10",
                "category": "Investing Activities",
                "vendor": "Equipment Supplier"
            }
        ]
    }
    
    response = requests.post(f"{BASE_URL}/ai-reasoning/full-analysis", json=data)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"\nTotal Transactions Analyzed: {result.get('full_analysis', {}).get('total_transactions', 0)}")
        print(f"Analyses Available:")
        if 'full_analysis' in result and 'analyses' in result['full_analysis']:
            for analysis_type in result['full_analysis']['analyses'].keys():
                print(f"  ✓ {analysis_type}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("AI REASONING API TEST SUITE")
    print("="*60)
    print("\nMake sure the backend is running on http://localhost:5000")
    print("Press Enter to continue...")
    input()
    
    try:
        # Test 1: Status
        test_ai_reasoning_status()
        
        # Test 2: Categorization Reasoning
        test_categorization_reasoning()
        
        # Test 3: Vendor Extraction Reasoning
        test_vendor_extraction_reasoning()
        
        # Test 4: Trend Analysis
        test_trend_analysis()
        
        # Test 5: AI System Explanation
        test_ai_system_explanation()
        
        # Test 6: Full Analysis
        test_full_analysis()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED!")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to backend server!")
        print("Make sure the Flask backend is running on http://localhost:5000")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")

