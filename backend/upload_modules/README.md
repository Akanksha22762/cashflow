# Upload Modules - Modular Upload Processing System

## 📁 Overview

This package breaks down the complex upload process into clean, maintainable modules. Each module handles a specific responsibility, making the code easier to understand, test, and maintain.

## 🏗️ Module Structure

### 1. **file_validator.py** - File Validation
- Validates uploaded files
- Checks file extensions
- Saves files to temporary location

**Functions:**
- `validate_uploaded_file()` - Validates file before processing
- `save_uploaded_file()` - Saves file to temporary location

### 2. **file_loader.py** - File Loading
- Loads files using Universal Data Adapter (if available)
- Falls back to standard pandas loading
- Handles CSV and Excel files

**Functions:**
- `load_file_with_adapter()` - Load using Universal Data Adapter
- `load_file_standard()` - Load using standard pandas methods
- `load_file()` - Main function that tries adapter first, then standard

### 3. **data_preprocessor.py** - Data Preprocessing
- Sorts data by date
- Normalizes date columns
- Validates data structure

**Functions:**
- `preprocess_dataframe()` - Main preprocessing function

### 4. **ai_categorizer.py** - AI Categorization
- Applies AI/ML categorization to transactions
- Skips processing if using cached data
- Uses OpenAI for intelligent categorization

**Functions:**
- `categorize_transactions()` - Categorize transactions with AI

### 5. **database_storage.py** - Database Storage
- Stores file metadata
- Creates analysis sessions
- Stores transactions
- Stores category insights

**Functions:**
- `store_upload_results()` - Store all results in MySQL database

### 6. **response_formatter.py** - Response Formatting
- Formats transactions for frontend
- Generates AI reasoning explanations
- Creates complete response dictionary

**Functions:**
- `format_transactions_for_frontend()` - Format transaction data
- `generate_ai_reasoning_explanations()` - Generate AI explanations
- `format_upload_response()` - Create complete response

### 7. **upload_orchestrator.py** - Main Orchestrator
- Coordinates all upload steps
- Handles cache checking
- Manages the entire upload flow

**Functions:**
- `process_upload()` - Main function that orchestrates all steps

## 🔄 Upload Flow

```
1. File Validation
   ↓
2. Save File Temporarily
   ↓
3. Check Database Cache
   ├─→ Cache Hit: Use cached data (skip AI processing)
   └─→ Cache Miss: Continue to step 4
   ↓
4. Load File
   ├─→ Try Universal Data Adapter
   └─→ Fallback to Standard Loading
   ↓
5. Preprocess Data
   ├─→ Sort by date
   └─→ Normalize columns
   ↓
6. AI Categorization
   ├─→ Use cached categories (if cache hit)
   └─→ Run AI categorization (if cache miss)
   ↓
7. Store in Database
   ├─→ Store file metadata
   ├─→ Create analysis session
   ├─→ Store transactions
   └─→ Store category insights
   ↓
8. Format Response
   ├─→ Format transactions
   ├─→ Generate AI explanations
   └─→ Create response dictionary
```

## 📝 Usage

### Simple Usage (Recommended)

Replace your upload endpoint in `app.py` with:

```python
from upload_modules import process_upload

@app.route('/upload', methods=['POST'])
def upload_files_with_ml_ai():
    global reconciliation_data
    
    bank_file = request.files.get('bank_file')
    
    # Process upload using modular orchestrator
    response_data, status_code = process_upload(
        bank_file=bank_file,
        db_manager=db_manager if DATABASE_AVAILABLE else None,
        session=session,
        data_adapter_available=DATA_ADAPTER_AVAILABLE,
        ml_available=ML_AVAILABLE,
        reconciliation_data=reconciliation_data
    )
    
    return jsonify(response_data), status_code
```

### Advanced Usage (Custom Steps)

If you need to customize specific steps:

```python
from upload_modules import (
    validate_uploaded_file,
    save_uploaded_file,
    load_file,
    preprocess_dataframe,
    categorize_transactions,
    store_upload_results,
    format_upload_response
)

# Use individual modules as needed
is_valid, error = validate_uploaded_file(bank_file)
if not is_valid:
    return jsonify({'error': error}), 400

# ... continue with custom logic
```

## ✅ Benefits

1. **Modularity** - Each module has a single responsibility
2. **Maintainability** - Easy to find and fix issues
3. **Testability** - Each module can be tested independently
4. **Reusability** - Modules can be used in other parts of the application
5. **Readability** - Clear flow and structure
6. **Scalability** - Easy to add new features or modify existing ones

## 🔧 Configuration

The modules use global variables from `app.py`:
- `DATABASE_AVAILABLE` - Whether database is available
- `DATA_ADAPTER_AVAILABLE` - Whether Universal Data Adapter is available
- `ML_AVAILABLE` - Whether ML is available
- `db_manager` - MySQLDatabaseManager instance
- `session` - Flask session object

## 📊 Module Dependencies

```
upload_orchestrator.py
├── file_validator.py
├── file_loader.py
│   └── data_adapter_integration.py (external)
├── data_preprocessor.py
├── ai_categorizer.py
│   └── universal_data_adapter.py (external)
├── database_storage.py
│   ├── mysql_database_manager.py (external)
│   └── analysis_storage_integration.py (external)
└── response_formatter.py
```

## 🚀 Next Steps

1. Replace the old upload endpoint in `app.py` with the simplified version
2. Test the upload functionality
3. Monitor logs to ensure all modules work correctly
4. Customize modules as needed for your specific requirements

