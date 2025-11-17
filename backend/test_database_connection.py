#!/usr/bin/env python3
"""
🧪 Database Connection Test Script
Tests MySQL database connectivity and basic operations
"""

import mysql.connector
from mysql_database_manager import MySQLDatabaseManager
import sys

def test_database_connection():
    """Test basic database connection"""
    print("🧪 Testing Database Connection...")
    
    try:
        # Test 1: Basic connection
        print("1. Testing basic MySQL connection...")
        connection = mysql.connector.connect(
            host='cashflow.c1womgmu83di.ap-south-1.rds.amazonaws.com',
            port=3306,
            user='admin',
            password='cashflow123',
            database='cashflow'
        )
        
        if connection.is_connected():
            print("   ✅ Database connection successful!")
            cursor = connection.cursor()
            cursor.execute("SELECT VERSION()")
            version = cursor.fetchone()
            print(f"   📊 MySQL Version: {version[0]}")
            cursor.close()
            connection.close()
        else:
            print("   ❌ Database connection failed!")
            return False
            
    except mysql.connector.Error as e:
        print(f"   ❌ Database connection error: {e}")
        return False
    
    try:
        # Test 2: Using MySQLDatabaseManager
        print("2. Testing MySQLDatabaseManager...")
        db_manager = MySQLDatabaseManager(password="cashflow123")
        
        if db_manager.connection:
            print("   ✅ MySQLDatabaseManager connection successful!")
            
            # Test basic query
            cursor = db_manager.connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM file_metadata")
            file_count = cursor.fetchone()[0]
            print(f"   📊 Files in database: {file_count}")
            
            cursor.execute("SELECT COUNT(*) FROM analysis_sessions")
            session_count = cursor.fetchone()[0]
            print(f"   📊 Analysis sessions: {session_count}")
            
            cursor.execute("SELECT COUNT(*) FROM transactions")
            transaction_count = cursor.fetchone()[0]
            print(f"   📊 Total transactions: {transaction_count}")
            
            # Test 10-transaction mode verification
            cursor.execute("""
                SELECT session_id, COUNT(*) as transaction_count
                FROM transactions 
                WHERE session_id = (SELECT MAX(session_id) FROM transactions)
                GROUP BY session_id
            """)
            latest_session = cursor.fetchone()
            
            if latest_session:
                session_id, count = latest_session
                print(f"   📊 Latest session ({session_id}): {count} transactions")
                
                if count <= 10:
                    print("   ✅ 10-Transaction Testing Mode: ACTIVE")
                else:
                    print("   ⚠️ Production Mode: ACTIVE (more than 10 transactions)")
            
            cursor.close()
            db_manager.close_connection()
            
        else:
            print("   ❌ MySQLDatabaseManager connection failed!")
            return False
            
    except Exception as e:
        print(f"   ❌ MySQLDatabaseManager error: {e}")
        return False
    
    print("\n🎉 All database tests passed!")
    return True

def test_table_structure():
    """Test table structure and data integrity"""
    print("\n🧪 Testing Table Structure...")
    
    try:
        db_manager = MySQLDatabaseManager(password="cashflow123")
        cursor = db_manager.connection.cursor()
        
        # Test table existence
        tables = ['file_metadata', 'analysis_sessions', 'transactions', 'session_states']
        
        for table in tables:
            cursor.execute(f"SHOW TABLES LIKE '{table}'")
            result = cursor.fetchone()
            if result:
                print(f"   ✅ Table '{table}' exists")
            else:
                print(f"   ❌ Table '{table}' missing")
        
        # Test data integrity
        print("\n📊 Data Integrity Tests:")
        
        # Check for categorized transactions
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(ai_category) as categorized,
                COUNT(*) - COUNT(ai_category) as uncategorized
            FROM transactions
        """)
        result = cursor.fetchone()
        total, categorized, uncategorized = result
        print(f"   📈 Total transactions: {total}")
        print(f"   ✅ Categorized: {categorized}")
        print(f"   ⚠️ Uncategorized: {uncategorized}")
        
        if categorized > 0:
            print(f"   📊 Categorization rate: {(categorized/total)*100:.1f}%")
        
        cursor.close()
        db_manager.close_connection()
        
    except Exception as e:
        print(f"   ❌ Table structure test error: {e}")
        return False
    
    return True

def main():
    """Main test function"""
    print("🚀 Starting Database Tests for 10-Transaction Mode...")
    print("=" * 60)
    
    # Test 1: Basic connection
    if not test_database_connection():
        print("\n❌ Database connection tests failed!")
        sys.exit(1)
    
    # Test 2: Table structure
    if not test_table_structure():
        print("\n❌ Table structure tests failed!")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🎉 ALL DATABASE TESTS PASSED!")
    print("✅ Your database is ready for 10-transaction testing mode!")
    print("\n📋 Next steps:")
    print("   1. Run the SQL queries in 'database_test_queries.sql'")
    print("   2. Upload a file to test 10-transaction processing")
    print("   3. Check the database for new data")

if __name__ == "__main__":
    main()
