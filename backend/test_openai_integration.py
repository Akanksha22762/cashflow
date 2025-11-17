"""
Test script for OpenAI Integration
This script verifies that the OpenAI integration is working correctly
"""

import os
import sys
from dotenv import load_dotenv

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables
load_dotenv()

def test_openai_integration():
    """Test OpenAI integration"""
    print("=" * 60)
    print("TESTING OPENAI INTEGRATION")
    print("=" * 60)
    
    # Check if API key is loaded
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print(f"[OK] API Key loaded: {api_key[:20]}...{api_key[-10:]}")
    else:
        print("[ERROR] API Key not found in environment")
        return False
    
    # Test importing the module
    try:
        from openai_integration import openai_integration, simple_openai
        print("[OK] OpenAI integration module imported successfully")
    except Exception as e:
        print(f"[ERROR] Failed to import OpenAI integration: {e}")
        return False
    
    # Check health status
    try:
        health = openai_integration.get_health_status()
        print(f"\n[INFO] Health Status:")
        print(f"   - Available: {health['available']}")
        print(f"   - API Configured: {health['api_configured']}")
        print(f"   - Default Model: {health['default_model']}")
        print(f"   - Status: {health['status']}")
    except Exception as e:
        print(f"[ERROR] Failed to get health status: {e}")
        return False
    
    # Test simple API call
    try:
        print("\n[TEST] Testing simple API call...")
        response = simple_openai("Say 'Hello, OpenAI integration is working!' in one sentence.", max_tokens=20)
        if response:
            print(f"[OK] API Response: {response}")
        else:
            print("[ERROR] No response from API")
            return False
    except Exception as e:
        print(f"[ERROR] API call failed: {e}")
        return False
    
    # Test transaction categorization
    try:
        print("\n[TEST] Testing transaction categorization...")
        test_descriptions = [
            "Salaries and wages paid to employees",
            "Purchase of machinery for production",
            "Loan repayment to bank"
        ]
        categories = openai_integration.categorize_transactions(test_descriptions)
        print(f"[OK] Categorization Results:")
        for desc, cat in zip(test_descriptions, categories):
            print(f"   - '{desc[:40]}...' => {cat}")
    except Exception as e:
        print(f"[ERROR] Categorization test failed: {e}")
        return False
    
    # Test vendor extraction
    try:
        print("\n[TEST] Testing vendor extraction...")
        test_descriptions = [
            "Payment to Tata Steel for raw materials",
            "Energy sale to Gujarat DISCOM",
            "General office expenses"
        ]
        vendors = openai_integration.extract_vendors_for_transactions(test_descriptions)
        print(f"[OK] Vendor Extraction Results:")
        for desc, vendor in zip(test_descriptions, vendors):
            print(f"   - '{desc[:40]}...' => {vendor}")
    except Exception as e:
        print(f"[ERROR] Vendor extraction test failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_openai_integration()
    if not success:
        print("\n[WARNING] Some tests failed. Please check the error messages above.")
        exit(1)
    else:
        print("\n[SUCCESS] OpenAI integration is ready to use!")
        exit(0)
