# Database Storage Summary

## What Gets Stored When You Upload a File

When you upload a file, the following data is stored in the MySQL database:

### 1. **File Metadata** (`files` table)
- ✅ **filename**: Original filename
- ✅ **file_hash**: SHA-256 hash (for duplicate detection)
- ✅ **file_size**: File size in bytes
- ✅ **file_type**: File extension (csv, xlsx, etc.)
- ✅ **data_source**: Source type ('bank', 'sap', 'other')
- ✅ **uploaded_at**: Timestamp when file was uploaded
- ✅ **last_processed**: Last processing timestamp

### 2. **Analysis Session** (`analysis_sessions` table)
- ✅ **file_id**: Link to the uploaded file
- ✅ **session_uuid**: Unique session identifier
- ✅ **analysis_type**: Type of analysis ('full_analysis')
- ✅ **status**: Session status ('processing' → 'completed')
- ✅ **created_at**: Session creation timestamp
- ✅ **completed_at**: Completion timestamp
- ✅ **transaction_count**: Total number of transactions processed
- ✅ **processing_time**: Time taken to process (in seconds)
- ✅ **success_rate**: AI categorization success rate (%)

### 3. **All Transactions** (`transactions` table)
For **EVERY transaction** in your file, the following is stored:
- ✅ **session_id**: Link to analysis session
- ✅ **file_id**: Link to the file
- ✅ **original_row_number**: Original row number in the file
- ✅ **transaction_date**: Transaction date
- ✅ **description**: Transaction description/narration
- ✅ **amount**: Transaction amount
- ✅ **balance**: Account balance (if available)
- ✅ **transaction_type**: Type (Inward/Outward, if available)
- ✅ **ai_category**: AI-assigned category (Operating/Investing/Financing Activities)
- ✅ **ai_confidence_score**: AI confidence (currently None, will be added later)
- ✅ **vendor_name**: Extracted vendor name (currently None, will be added later)

### 4. **Category Insights** (`category_insights` table)
For each category type, summary statistics:
- ✅ **category_name**: Category name (Operating/Investing/Financing Activities)
- ✅ **transaction_count**: Number of transactions in this category
- ✅ **total_amount**: Total amount for this category
- ✅ **average_amount**: Average transaction amount
- ✅ **percentage**: Percentage of total transactions

### 5. **Session State** (if persistent state manager is available)
- ✅ **reconciliation_data**: Reconciliation information
- ✅ **uploaded_bank_df**: Processed DataFrame data
- ✅ **bank_count**: Number of bank transactions
- ✅ **ai_categorized**: Number of AI-categorized transactions
- ✅ **processing_time**: Processing time
- ✅ **upload_timestamp**: Upload timestamp

## Storage Flow

```
Upload File
    ↓
1. Validate & Save File
    ↓
2. Check Database Cache (by file hash)
    ↓
3. If NOT cached:
   - Store File Metadata → files table
   - Create Analysis Session → analysis_sessions table
   - Load & Process File
   - AI Categorization
   - Store ALL Transactions → transactions table (one row per transaction)
   - Store Category Insights → category_insights table
   - Complete Analysis Session → update analysis_sessions table
   - Save Session State (if available)
    ↓
4. If cached:
   - Retrieve cached data from database
   - Skip AI processing (saves time & cost)
```

## Important Notes

✅ **ALL transactions are stored** - Every single transaction from your file is saved to the database

✅ **Caching works** - If you upload the same file again (same hash), it uses cached results from the database instead of re-processing

✅ **Complete data** - All transaction details (date, description, amount, category, etc.) are stored

✅ **Session tracking** - Each upload creates a new analysis session that tracks all processing details

✅ **Category analysis** - Summary statistics for each category are stored separately for quick access

## Database Tables Used

1. `files` - File metadata
2. `analysis_sessions` - Analysis session tracking
3. `transactions` - Individual transaction records
4. `category_insights` - Category-wise summary statistics

## Verification

To verify everything is stored, check:
- `files` table: Should have your file entry
- `analysis_sessions` table: Should have a completed session
- `transactions` table: Should have ALL transactions from your file
- `category_insights` table: Should have category breakdowns

