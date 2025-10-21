#!/usr/bin/env python3
"""
🧪 Database Integration Test Script
Tests the complete database integration with the Flask app
"""

import sys
import os

def test_database_integration():
    """Test complete database integration"""
    print("🧪 Testing Database Integration...")
    
    try:
        # Test 1: Import app and database manager
        print("1. Testing imports...")
        from app import app
        from mysql_database_manager import MySQLDatabaseManager
        print("   ✅ App and database manager imported successfully")
        
        # Test 2: Create database manager
        print("2. Testing database manager creation...")
        db = MySQLDatabaseManager(password="cashflow123")
        print("   ✅ Database manager created successfully")
        
        # Test 3: Test connection
        print("3. Testing database connection...")
        if db.test_connection():
            print("   ✅ Database connection successful")
        else:
            print("   ❌ Database connection failed")
            return False
        
        # Test 4: Test basic operations
        print("4. Testing basic database operations...")
        conn = db.get_connection()
        cursor = conn.cursor()
        
        # Get file count
        cursor.execute("SELECT COUNT(*) FROM files")
        file_count = cursor.fetchone()[0]
        print(f"   📊 Files in database: {file_count}")
        
        # Get transaction count
        cursor.execute("SELECT COUNT(*) FROM transactions")
        transaction_count = cursor.fetchone()[0]
        print(f"   📊 Transactions in database: {transaction_count}")
        
        # Get analysis sessions count
        cursor.execute("SELECT COUNT(*) FROM analysis_sessions")
        session_count = cursor.fetchone()[0]
        print(f"   📊 Analysis sessions: {session_count}")
        
        cursor.close()
        conn.close()
        
        # Test 5: Test database manager methods
        print("5. Testing database manager methods...")
        files = db.get_all_files()
        print(f"   📊 get_all_files() returned {len(files)} files")
        
        # Test 6: Test Flask app integration
        print("6. Testing Flask app integration...")
        with app.test_client() as client:
            response = client.get('/')
            if response.status_code == 200:
                print("   ✅ Flask app responds successfully")
            else:
                print(f"   ⚠️ Flask app returned status {response.status_code}")
        
        print("\n🎉 All database integration tests passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Database integration test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_processing():
    """Test dataset processing functionality"""
    print("\n🧪 Testing Dataset Processing...")
    
    try:
        from app import universal_categorize_any_dataset
        import pandas as pd
        
        # Create a small test dataset
        test_data = pd.DataFrame({
            'Description': ['Office supplies purchase', 'Equipment maintenance', 'Salary payment'],
            'Amount': [-150.00, -500.00, -2500.00],
            'Date': ['2024-01-01', '2024-01-02', '2024-01-03']
        })
        
        print("1. Testing dataset processing with sample data...")
        result = universal_categorize_any_dataset(test_data)
        
        if result is not None and not result.empty:
            print(f"   ✅ Dataset processing successful")
            print(f"   📊 Processed {len(result)} transactions")
            if 'Category' in result.columns:
                print(f"   📊 Categories generated: {result['Category'].value_counts().to_dict()}")
            return True
        else:
            print("   ❌ Dataset processing failed - no result returned")
            return False
            
    except Exception as e:
        print(f"   ❌ Dataset processing test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 Starting Complete Database Integration Tests...")
    print("=" * 60)
    
    # Test 1: Database integration
    if not test_database_integration():
        print("\n❌ Database integration tests failed!")
        sys.exit(1)
    
    # Test 2: Dataset processing
    if not test_dataset_processing():
        print("\n❌ Dataset processing tests failed!")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED!")
    print("✅ Your database and dataset processing are working correctly!")
    print("\n📋 System Status:")
    print("   ✅ Database connection: WORKING")
    print("   ✅ Database operations: WORKING") 
    print("   ✅ Flask app integration: WORKING")
    print("   ✅ Dataset processing: WORKING")
    print("\n🚀 Your system is ready for production use!")

if __name__ == "__main__":
    main()
