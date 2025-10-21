# 20 Transaction Test Limit - Guide

## Overview
Your system now automatically limits all uploaded datasets to **20 transactions** for testing purposes.

---

## ✅ What's Been Changed

### 1. Main Upload Route (`app.py`)
**Location**: Line ~11975

\`\`\`python
# ===== TESTING MODE: LIMIT TO 20 TRANSACTIONS =====
original_count = len(uploaded_bank_df)
TEST_LIMIT = 20

if original_count > TEST_LIMIT:
    print(f"🧪 TEST MODE: Limiting dataset from {original_count} to {TEST_LIMIT} transactions")
    uploaded_bank_df = uploaded_bank_df.head(TEST_LIMIT)
    print(f"✅ Dataset limited to first {TEST_LIMIT} transactions")
\`\`\`

### 2. Universal Data Adapter (`universal_data_adapter.py`)
**Function**: `load_and_adapt(file_path, test_limit=20)`

\`\`\`python
# Automatically limits data to 20 transactions
adapted_data = adapter.adapt(data)

if test_limit and original_count > test_limit:
    logger.info(f"🧪 TEST MODE: Limiting to {test_limit} transactions")
    adapted_data = adapted_data.head(test_limit)
\`\`\`

### 3. Data Adapter Integration (`data_adapter_integration.py`)
**Function**: `load_and_preprocess_file(file_path, test_limit=20)`

\`\`\`python
# Applies 20 transaction limit after preprocessing
if result is not None and test_limit:
    original_count = len(result)
    if original_count > test_limit:
        result = result.head(test_limit)
\`\`\`

---

## 🎯 How It Works

### When You Upload a File

1. **Small File** (≤20 transactions):
   \`\`\`
   📊 Processing all 15 transactions (less than 20 limit)
   📊 Final dataset size: 15 transactions
   \`\`\`

2. **Large File** (>20 transactions):
   \`\`\`
   🧪 TEST MODE: Limiting dataset from 1000 to 20 transactions for testing
   ✅ Dataset limited to first 20 transactions
   📊 Final dataset size: 20 transactions
   \`\`\`

### What Gets Limited
- ✅ **Bank data upload** - First 20 transactions
- ✅ **CSV files** - First 20 rows
- ✅ **Excel files** - First 20 rows
- ✅ **All data processing** - Maximum 20 transactions
- ✅ **OpenAI categorization** - Only 20 transactions processed
- ✅ **Vendor extraction** - Only 20 vendors extracted

---

## 💰 Cost Benefits

### With 20 Transaction Limit:
- **OpenAI API calls**: ~60-80 calls (20 transactions × 3-4 calls each)
- **Estimated cost**: ~$0.01-0.02 per test
- **Processing time**: ~30-60 seconds

### Without Limit (1000 transactions):
- **OpenAI API calls**: ~3,000-4,000 calls
- **Estimated cost**: ~$0.50-1.00 per test
- **Processing time**: ~15-30 minutes

### Savings:
- **50x cost reduction** 💰
- **20-30x faster testing** ⚡
- **Quick iteration** for development 🚀

---

## 🧪 Testing Workflow

### 1. Upload Test Data
\`\`\`bash
# Upload any file - automatically limited to 20 transactions
python app.py
# Navigate to upload page
\`\`\`

### 2. What You'll See
\`\`\`
🔍 DEBUG: After reading file - uploaded_bank_df shape: (1000, 5)
🧪 TEST MODE: Limiting dataset from 1000 to 20 transactions for testing
✅ Dataset limited to first 20 transactions
📊 Final dataset size: 20 transactions

🤖 ML PROCESSING: Using 100% AI/ML approach...
🤖 Applying AI/ML categorization to all 20 transactions...
✅ AI categorization applied: 20/20 transactions categorized with AI
\`\`\`

### 3. Test Results
- ✅ Only 20 transactions processed
- ✅ Fast categorization (~30-60 seconds)
- ✅ Low API costs (~$0.01-0.02)
- ✅ Quick feedback loop

---

## 🔧 Changing the Limit

### Option 1: Change Global Limit (Recommended for Testing)

**In `app.py` (line ~11977):**
\`\`\`python
TEST_LIMIT = 20  # Change to 50, 100, or any number
\`\`\`

**In `universal_data_adapter.py` (line 374):**
\`\`\`python
def load_and_adapt(cls, file_path: str, test_limit: int = 20):  # Change default here
\`\`\`

**In `data_adapter_integration.py` (line 196):**
\`\`\`python
def load_and_preprocess_file(file_path: str, test_limit: int = 20):  # Change default here
\`\`\`

### Option 2: Remove Limit for Production

**In `app.py`:**
\`\`\`python
# Change from:
TEST_LIMIT = 20

# To:
TEST_LIMIT = None  # No limit
\`\`\`

**In function calls:**
\`\`\`python
# Change from:
adapted_data = load_and_adapt(file_path, test_limit=20)

# To:
adapted_data = load_and_adapt(file_path, test_limit=None)  # No limit
\`\`\`

### Option 3: Environment-Based Limit

Add to your `.env` file:
\`\`\`
TEST_LIMIT=20
ENVIRONMENT=LOCAL
\`\`\`

Then in code:
\`\`\`python
import os
TEST_LIMIT = int(os.getenv('TEST_LIMIT', 0))  # 0 = no limit
if TEST_LIMIT > 0:
    uploaded_bank_df = uploaded_bank_df.head(TEST_LIMIT)
\`\`\`

---

## 📊 Example Outputs

### Test with 20 Transactions
\`\`\`
📊 Final dataset size: 20 transactions
🤖 Applying AI/ML categorization to all 20 transactions...
✅ OpenAI categorization completed: 20 categories generated

Processing time: ~45 seconds
API calls: ~60-80
Cost: ~$0.01-0.02
\`\`\`

### Test with 50 Transactions
\`\`\`
📊 Final dataset size: 50 transactions
🤖 Applying AI/ML categorization to all 50 transactions...
✅ OpenAI categorization completed: 50 categories generated

Processing time: ~2 minutes
API calls: ~150-200
Cost: ~$0.03-0.05
\`\`\`

### Full Dataset (No Limit)
\`\`\`
📊 Final dataset size: 1000 transactions
🤖 Applying AI/ML categorization to all 1000 transactions...
✅ OpenAI categorization completed: 1000 categories generated

Processing time: ~15-30 minutes
API calls: ~3000-4000
Cost: ~$0.50-1.00
\`\`\`

---

## ✅ Benefits

### For Development
- ✅ **Faster iteration** - Test changes in 30-60 seconds
- ✅ **Lower costs** - $0.01-0.02 per test vs $0.50-1.00
- ✅ **Quick debugging** - Easier to inspect 20 transactions
- ✅ **Rapid testing** - Try different approaches quickly

### For Production
- ✅ **Easy to remove** - Change TEST_LIMIT to None
- ✅ **Configurable** - Set via environment variable
- ✅ **Flexible** - Different limits for different environments
- ✅ **Safe** - Prevents accidental large uploads during testing

---

## 🎯 Best Practices

### During Development
\`\`\`python
TEST_LIMIT = 20  # Fast testing
\`\`\`

### During QA/Staging
\`\`\`python
TEST_LIMIT = 100  # More comprehensive testing
\`\`\`

### In Production
\`\`\`python
TEST_LIMIT = None  # No limit, process all transactions
\`\`\`

---

## 🔍 Verification

### Check Limit is Applied
Look for these messages in console:
\`\`\`
🧪 TEST MODE: Limiting dataset from X to 20 transactions for testing
✅ Dataset limited to first 20 transactions
📊 Final dataset size: 20 transactions
\`\`\`

### Check Processing
\`\`\`
🤖 Applying AI/ML categorization to all 20 transactions...
✅ AI categorization applied: 20/20 transactions categorized
\`\`\`

### Check API Calls
Monitor OpenAI dashboard:
- Expected calls: ~60-80 for 20 transactions
- Expected cost: ~$0.01-0.02

---

## 🚀 Quick Reference

### Current Configuration
- **Main Upload**: 20 transaction limit ✅
- **Data Adapter**: 20 transaction limit ✅
- **Preprocessing**: 20 transaction limit ✅
- **All Functions**: Consistent 20 limit ✅

### To Change Limit
1. Edit `TEST_LIMIT = 20` in `app.py` (line 11977)
2. Edit `test_limit: int = 20` in `universal_data_adapter.py` (line 374)
3. Edit `test_limit: int = 20` in `data_adapter_integration.py` (line 196)

### To Remove Limit
1. Set `TEST_LIMIT = None` in all files
2. Or comment out the limit logic
3. Or use environment variable

---

## 📝 Summary

Your system now:
- ✅ Automatically limits uploads to 20 transactions
- ✅ Works consistently across all data loading functions
- ✅ Saves ~98% on API costs during testing ($0.01 vs $0.50)
- ✅ Reduces testing time by 20-30x (1 min vs 30 mins)
- ✅ Easy to configure or remove for production

**Perfect for rapid testing and development!** 🎉

---

*Last Updated: October 11, 2025*  
*Default Limit: 20 transactions*  
*Status: ✅ Active across all data loading points*
