#!/usr/bin/env python3
import requests
import json

try:
    print("Testing debug endpoint...")
    response = requests.get('http://localhost:5000/debug-data')
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Has bank_df: {data.get('has_bank_df', False)}")
        print(f"Columns: {data.get('bank_df_columns', [])}")
        print(f"Date columns: {data.get('date_columns', [])}")
        
        # Check for year and month samples
        for key, value in data.items():
            if 'year' in key.lower() or 'month' in key.lower():
                print(f"{key}: {value}")
        
        # Show sample data
        sample_data = data.get('sample_data', [])
        if sample_data:
            print(f"Sample data: {json.dumps(sample_data[0], indent=2)}")
    else:
        print(f"Error: {response.text}")
        
except Exception as e:
    print(f"Error: {e}")
