#!/usr/bin/env python3
import requests
import time
import subprocess
import sys
import os

def test_vendor_endpoint():
    """Test the vendor endpoint to see if dates are working"""
    try:
        print("Testing vendor endpoint...")
        response = requests.get('http://localhost:5000/view_vendor_transactions/Thermax')
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            transactions = data.get('transactions', [])
            if transactions:
                sample_tx = transactions[0]
                print(f"Sample transaction date: {sample_tx.get('date', 'Not found')}")
                print(f"Sample transaction: {sample_tx.get('description', 'No description')[:50]}...")
                return sample_tx.get('date', 'Date N/A') != 'Date N/A'
            else:
                print("No transactions found")
                return False
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Error testing endpoint: {e}")
        return False

def main():
    print("🔧 Testing date fix for vendor transactions...")
    print("=" * 50)
    
    # Test current state
    print("1. Testing current state...")
    current_working = test_vendor_endpoint()
    
    if current_working:
        print("✅ Dates are already working!")
        return
    
    print("❌ Dates still showing as 'Date N/A'")
    print("\n🔄 The backend server needs to be restarted to pick up the new code changes.")
    print("Please:")
    print("1. Stop your current backend server (Ctrl+C)")
    print("2. Restart it with: python app.py")
    print("3. Then test the vendor analysis again")
    print("\nThe fix will create dates from your year/month columns like:")
    print("- year: 2021, month: 7 → 2021-07-01")
    print("- year: 2025, month: 12 → 2025-12-01")

if __name__ == "__main__":
    main()
