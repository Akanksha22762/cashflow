"""
Quick Test: 20 Transaction Limit
"""
import pandas as pd
from universal_data_adapter import UniversalDataAdapter
from data_adapter_integration import load_and_preprocess_file

def test_limit():
    """Test that the 20 transaction limit is working"""
    
    print("=" * 80)
    print("TESTING 20 TRANSACTION LIMIT")
    print("=" * 80)
    
    # Test 1: Check if universal_data_adapter respects the limit
    print("\n📝 Test 1: Testing UniversalDataAdapter with test_limit=20")
    print("-" * 80)
    
    # Find a test file with more than 20 transactions
    test_files = [
        "uploads/bank_enhanced_steel_plant_bank_data.xlsx",
        "uploads/bank_hospital_bank_statement_single_description.xlsx",
        "steel_plant_datasets/steel_plant_bank_data.xlsx"
    ]
    
    for test_file in test_files:
        try:
            print(f"\n🔍 Testing with: {test_file}")
            result = UniversalDataAdapter.load_and_adapt(test_file, test_limit=20)
            
            if result is not None:
                print(f"✅ Result shape: {result.shape}")
                print(f"✅ Number of transactions: {len(result)}")
                
                if len(result) == 20:
                    print("🎉 SUCCESS: Limit is working correctly (20 transactions)")
                elif len(result) < 20:
                    print(f"ℹ️  File has less than 20 transactions ({len(result)} total)")
                else:
                    print(f"❌ FAILED: Expected 20 but got {len(result)} transactions")
                
                break
        except FileNotFoundError:
            print(f"⏭️  File not found, trying next one...")
            continue
        except Exception as e:
            print(f"⚠️  Error: {str(e)}")
            continue
    
    # Test 2: Check if data_adapter_integration respects the limit
    print("\n📝 Test 2: Testing data_adapter_integration with test_limit=20")
    print("-" * 80)
    
    for test_file in test_files:
        try:
            print(f"\n🔍 Testing with: {test_file}")
            result = load_and_preprocess_file(test_file, test_limit=20)
            
            if result is not None:
                print(f"✅ Result shape: {result.shape}")
                print(f"✅ Number of transactions: {len(result)}")
                
                if len(result) == 20:
                    print("🎉 SUCCESS: Limit is working correctly (20 transactions)")
                elif len(result) < 20:
                    print(f"ℹ️  File has less than 20 transactions ({len(result)} total)")
                else:
                    print(f"❌ FAILED: Expected 20 but got {len(result)} transactions")
                
                break
        except FileNotFoundError:
            print(f"⏭️  File not found, trying next one...")
            continue
        except Exception as e:
            print(f"⚠️  Error: {str(e)}")
            continue
    
    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)
    print("\n💡 To test with your app, run: python app.py")
    print("   Then upload a file and look for these messages:")
    print("   - '🧪 TEST MODE: Limiting dataset from X to 20 transactions'")
    print("   - '✅ Dataset limited to first 20 transactions'")
    print("   - '📊 Final dataset size: 20 transactions'")
    print()

if __name__ == "__main__":
    test_limit()
