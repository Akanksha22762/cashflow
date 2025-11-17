# Database Schema Documentation

## Overview
This document describes the SQLite database schema for the Cashflow SAP Bank Analysis System. The database stores all analysis results, metadata, and version history to enable instant access to results and automatic override capabilities.

## Database File
- **Location**: `data/analysis_database.db`
- **Type**: SQLite 3
- **Purpose**: Persistent storage for all analysis results

## Table Structure

### 1. file_metadata
Stores information about uploaded files and their processing status.

| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| file_id | INTEGER | Unique identifier for each file | PRIMARY KEY, AUTOINCREMENT |
| filename | TEXT | Original filename | NOT NULL |
| file_hash | TEXT | SHA-256 hash of file content | UNIQUE, NOT NULL |
| file_size | INTEGER | File size in bytes | |
| file_type | TEXT | File extension (e.g., .xlsx) | |
| upload_timestamp | TIMESTAMP | When file was first uploaded | DEFAULT CURRENT_TIMESTAMP |
| last_processed | TIMESTAMP | When file was last processed | DEFAULT CURRENT_TIMESTAMP |
| processing_status | TEXT | Current processing status | DEFAULT 'completed' |
| industry_context | TEXT | Industry context if available | |
| analysis_version | TEXT | Version of analysis system | DEFAULT '1.0' |

**Indexes:**
- `idx_file_hash` on `file_hash`
- `idx_filename` on `filename`
- `idx_upload_timestamp` on `upload_timestamp`

### 2. cash_flow_results
Stores cash flow analysis results for each file.

| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| result_id | INTEGER | Unique identifier for each result | PRIMARY KEY, AUTOINCREMENT |
| file_id | INTEGER | Reference to file_metadata | FOREIGN KEY, NOT NULL |
| analysis_timestamp | TIMESTAMP | When analysis was performed | DEFAULT CURRENT_TIMESTAMP |
| cash_flow_data | TEXT | JSON string of cash flow data | NOT NULL |
| summary_data | TEXT | JSON string of summary data | NOT NULL |
| categories_data | TEXT | JSON string of categories data | NOT NULL |
| trends_data | TEXT | JSON string of trends data | NOT NULL |
| analysis_metadata | TEXT | JSON string of analysis metadata | |

**Indexes:**
- `idx_file_id` on `file_id`

### 3. vendor_extraction_results
Stores vendor extraction analysis results for each file.

| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| result_id | INTEGER | Unique identifier for each result | PRIMARY KEY, AUTOINCREMENT |
| file_id | INTEGER | Reference to file_metadata | FOREIGN KEY, NOT NULL |
| analysis_timestamp | TIMESTAMP | When analysis was performed | DEFAULT CURRENT_TIMESTAMP |
| vendor_data | TEXT | JSON string of vendor data | NOT NULL |
| vendor_summary | TEXT | JSON string of vendor summary | NOT NULL |
| vendor_categories | TEXT | JSON string of vendor categories | NOT NULL |
| vendor_insights | TEXT | JSON string of vendor insights | NOT NULL |
| analysis_metadata | TEXT | JSON string of analysis metadata | |

**Indexes:**
- `idx_vendor_file_id` on `file_id`

### 4. revenue_analysis_results
Stores revenue analysis results for each file.

| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| result_id | INTEGER | Unique identifier for each result | PRIMARY KEY, AUTOINCREMENT |
| file_id | INTEGER | Reference to file_metadata | FOREIGN KEY, NOT NULL |
| analysis_timestamp | TIMESTAMP | When analysis was performed | DEFAULT CURRENT_TIMESTAMP |
| revenue_data | TEXT | JSON string of revenue data | NOT NULL |
| revenue_summary | TEXT | JSON string of revenue summary | NOT NULL |
| revenue_trends | TEXT | JSON string of revenue trends | NOT NULL |
| revenue_insights | TEXT | JSON string of revenue insights | NOT NULL |
| analysis_metadata | TEXT | JSON string of analysis metadata | |

**Indexes:**
- `idx_revenue_file_id` on `file_id`

### 5. industry_context_results
Stores industry context analysis results for each file.

| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| result_id | INTEGER | Unique identifier for each result | PRIMARY KEY, AUTOINCREMENT |
| file_id | INTEGER | Reference to file_metadata | FOREIGN KEY, NOT NULL |
| analysis_timestamp | TIMESTAMP | When analysis was performed | DEFAULT CURRENT_TIMESTAMP |
| industry_data | TEXT | JSON string of industry data | NOT NULL |
| industry_summary | TEXT | JSON string of industry summary | NOT NULL |
| industry_insights | TEXT | JSON string of industry insights | NOT NULL |
| analysis_metadata | TEXT | JSON string of analysis metadata | |

**Indexes:**
- `idx_industry_file_id` on `file_id`

### 6. analysis_performance
Stores performance metrics for analysis operations.

| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| performance_id | INTEGER | Unique identifier for each performance record | PRIMARY KEY, AUTOINCREMENT |
| file_id | INTEGER | Reference to file_metadata | FOREIGN KEY, NOT NULL |
| analysis_type | TEXT | Type of analysis performed | NOT NULL |
| processing_time | REAL | Processing time in seconds | |
| memory_usage | REAL | Memory usage in MB | |
| cpu_usage | REAL | CPU usage percentage | |
| timestamp | TIMESTAMP | When performance was recorded | DEFAULT CURRENT_TIMESTAMP |

### 7. version_history
Tracks changes and overrides for audit purposes.

| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| version_id | INTEGER | Unique identifier for each version record | PRIMARY KEY, AUTOINCREMENT |
| file_id | INTEGER | Reference to file_metadata | FOREIGN KEY, NOT NULL |
| previous_result_id | INTEGER | ID of previous result (if override) | |
| new_result_id | INTEGER | ID of new result | NOT NULL |
| change_type | TEXT | Type of change made | NOT NULL |
| change_timestamp | TIMESTAMP | When change was made | DEFAULT CURRENT_TIMESTAMP |
| change_description | TEXT | Description of the change | |
| user_id | TEXT | Who made the change | DEFAULT 'system' |

## Relationships

### Foreign Key Constraints
- `cash_flow_results.file_id` → `file_metadata.file_id`
- `vendor_extraction_results.file_id` → `file_metadata.file_id`
- `revenue_analysis_results.file_id` → `file_metadata.file_id`
- `industry_context_results.file_id` → `file_metadata.file_id`
- `analysis_performance.file_id` → `file_metadata.file_id`
- `version_history.file_id` → `file_metadata.file_id`

### Cascade Deletes
When a file is deleted from `file_metadata`, all related records in other tables are automatically deleted.

## Data Storage Format

### JSON Fields
Most analysis data is stored as JSON strings to maintain flexibility:
- **cash_flow_data**: Complete cash flow analysis results
- **summary_data**: Summary statistics and key metrics
- **categories_data**: Categorized financial data
- **trends_data**: Trend analysis and patterns
- **vendor_data**: Vendor extraction results
- **revenue_data**: Revenue analysis results
- **industry_data**: Industry context analysis

### Metadata Fields
- **analysis_metadata**: System configuration, model versions, parameters
- **file_hash**: SHA-256 hash for duplicate detection
- **processing_status**: Current status of file processing

## Performance Considerations

### Indexes
- Primary keys are automatically indexed
- Foreign key columns are indexed for faster joins
- File hash is indexed for quick duplicate detection
- Timestamps are indexed for chronological queries

### Query Optimization
- Use `file_id` for joins between tables
- Use `filename` for user-friendly lookups
- Use `file_hash` for duplicate detection
- Use timestamps for chronological ordering

## Backup and Maintenance

### Backup Strategy
- Regular backups of the `.db` file
- Export data to CSV/Excel for analysis
- Version control for schema changes

### Maintenance Tasks
- Regular VACUUM operations
- Index rebuilding if needed
- Cleanup of old version history
- Performance monitoring

## Usage Examples

### Store Analysis Results
\`\`\`python
from database_manager import DatabaseManager

db = DatabaseManager()

# Store file metadata
file_id = db.store_file_metadata("bank_data.xlsx", "/path/to/file", "banking")

# Store cash flow results
result_id = db.store_cash_flow_results(
    file_id, cash_flow_data, summary_data, categories_data, trends_data
)
\`\`\`

### Retrieve Results
\`\`\`python
# Get latest results for a file
results = db.get_latest_results_by_filename("bank_data.xlsx")

# Get all analysis results
all_results = db.get_all_analysis_results()
\`\`\`

## Future Enhancements

### Planned Features
- Data compression for large JSON fields
- Partitioning for very large datasets
- Advanced querying capabilities
- Real-time synchronization
- Multi-user access controls

### Schema Evolution
- Version migration scripts
- Backward compatibility
- Data validation rules
- Performance monitoring
