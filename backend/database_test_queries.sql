-- 🧪 DATABASE TESTING QUERIES FOR 10-TRANSACTION TESTING MODE
-- Use these queries to verify your MySQL database is working properly

-- ==============================================
-- 1. BASIC CONNECTION AND DATABASE TESTS
-- ==============================================

-- Test 1: Check if database exists and is accessible
SHOW DATABASES LIKE 'cashflow';

-- Test 2: Check current database
SELECT DATABASE() as current_database;

-- Test 3: Check database version
SELECT VERSION() as mysql_version;

-- ==============================================
-- 2. TABLE STRUCTURE VERIFICATION
-- ==============================================

-- Test 4: List all tables in cashflow database
SHOW TABLES;

-- Test 5: Check file_metadata table structure
DESCRIBE file_metadata;

-- Test 6: Check analysis_sessions table structure
DESCRIBE analysis_sessions;

-- Test 7: Check transactions table structure
DESCRIBE transactions;

-- Test 8: Check session_states table structure
DESCRIBE session_states;

-- ==============================================
-- 3. DATA VERIFICATION QUERIES
-- ==============================================

-- Test 9: Count total files uploaded
SELECT COUNT(*) as total_files FROM file_metadata;

-- Test 10: Count total analysis sessions
SELECT COUNT(*) as total_sessions FROM analysis_sessions;

-- Test 11: Count total transactions stored
SELECT COUNT(*) as total_transactions FROM transactions;

-- Test 12: Count total session states
SELECT COUNT(*) as total_session_states FROM session_states;

-- ==============================================
-- 4. RECENT DATA QUERIES (Last 5 sessions)
-- ==============================================

-- Test 13: Get recent analysis sessions
SELECT 
    session_id,
    file_id,
    analysis_type,
    transaction_count,
    processing_time,
    success_rate,
    created_at,
    completed_at
FROM analysis_sessions 
ORDER BY created_at DESC 
LIMIT 5;

-- Test 14: Get recent file uploads
SELECT 
    file_id,
    filename,
    file_path,
    data_source,
    file_size,
    upload_timestamp
FROM file_metadata 
ORDER BY upload_timestamp DESC 
LIMIT 5;

-- Test 15: Get recent transactions (last 10)
SELECT 
    transaction_id,
    session_id,
    file_id,
    row_number,
    transaction_date,
    description,
    amount,
    ai_category,
    balance,
    transaction_type,
    vendor_name,
    ai_confidence,
    created_at
FROM transactions 
ORDER BY created_at DESC 
LIMIT 10;

-- ==============================================
-- 5. CATEGORY ANALYSIS QUERIES
-- ==============================================

-- Test 16: Count transactions by category
SELECT 
    ai_category,
    COUNT(*) as transaction_count,
    SUM(amount) as total_amount,
    AVG(amount) as avg_amount
FROM transactions 
WHERE ai_category IS NOT NULL AND ai_category != ''
GROUP BY ai_category
ORDER BY transaction_count DESC;

-- Test 17: Category distribution for latest session
SELECT 
    ai_category,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM transactions WHERE session_id = (SELECT MAX(session_id) FROM transactions)), 2) as percentage
FROM transactions 
WHERE session_id = (SELECT MAX(session_id) FROM transactions)
    AND ai_category IS NOT NULL AND ai_category != ''
GROUP BY ai_category
ORDER BY count DESC;

-- ==============================================
-- 6. VENDOR ANALYSIS QUERIES
-- ==============================================

-- Test 18: Count transactions by vendor
SELECT 
    vendor_name,
    COUNT(*) as transaction_count,
    SUM(amount) as total_amount,
    AVG(amount) as avg_amount
FROM transactions 
WHERE vendor_name IS NOT NULL AND vendor_name != ''
GROUP BY vendor_name
ORDER BY transaction_count DESC;

-- ==============================================
-- 7. SESSION STATE VERIFICATION
-- ==============================================

-- Test 19: Check recent session states
SELECT 
    session_id,
    state_type,
    data_size,
    created_at
FROM session_states 
ORDER BY created_at DESC 
LIMIT 10;

-- Test 20: Check global state data
SELECT 
    session_id,
    state_type,
    LENGTH(state_data) as data_length,
    created_at
FROM session_states 
WHERE state_type = 'global_data'
ORDER BY created_at DESC 
LIMIT 5;

-- ==============================================
-- 8. PERFORMANCE AND INTEGRITY TESTS
-- ==============================================

-- Test 21: Check for any NULL or empty categories
SELECT 
    COUNT(*) as total_transactions,
    COUNT(ai_category) as categorized_transactions,
    COUNT(*) - COUNT(ai_category) as uncategorized_transactions
FROM transactions;

-- Test 22: Check for duplicate transactions
SELECT 
    session_id,
    file_id,
    row_number,
    COUNT(*) as duplicate_count
FROM transactions 
GROUP BY session_id, file_id, row_number
HAVING COUNT(*) > 1;

-- Test 23: Check data integrity - transactions without sessions
SELECT COUNT(*) as orphaned_transactions
FROM transactions t
LEFT JOIN analysis_sessions a ON t.session_id = a.session_id
WHERE a.session_id IS NULL;

-- ==============================================
-- 9. TESTING MODE SPECIFIC QUERIES (10 Transactions)
-- ==============================================

-- Test 24: Verify 10-transaction limit is working
SELECT 
    session_id,
    COUNT(*) as transaction_count,
    CASE 
        WHEN COUNT(*) <= 10 THEN '✅ Testing Mode (≤10 transactions)'
        ELSE '⚠️ Production Mode (>10 transactions)'
    END as mode_status
FROM transactions 
GROUP BY session_id
ORDER BY session_id DESC
LIMIT 5;

-- Test 25: Latest session transaction count
SELECT 
    MAX(session_id) as latest_session_id,
    COUNT(*) as transaction_count_in_latest_session
FROM transactions 
WHERE session_id = (SELECT MAX(session_id) FROM transactions);

-- ==============================================
-- 10. SUMMARY REPORT QUERY
-- ==============================================

-- Test 26: Complete system status summary
SELECT 
    'Database Status' as check_type,
    '✅ Connected' as status,
    CONCAT('MySQL ', VERSION()) as details
UNION ALL
SELECT 
    'Total Files',
    CONCAT(COUNT(*), ' files'),
    CONCAT('Latest: ', MAX(filename))
FROM file_metadata
UNION ALL
SELECT 
    'Total Sessions',
    CONCAT(COUNT(*), ' sessions'),
    CONCAT('Latest: Session ', MAX(session_id))
FROM analysis_sessions
UNION ALL
SELECT 
    'Total Transactions',
    CONCAT(COUNT(*), ' transactions'),
    CONCAT('Latest: ', MAX(created_at))
FROM transactions
UNION ALL
SELECT 
    'Categorized Transactions',
    CONCAT(COUNT(*), ' transactions'),
    CONCAT(ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM transactions), 2), '% categorized')
FROM transactions 
WHERE ai_category IS NOT NULL AND ai_category != ''
UNION ALL
SELECT 
    'Testing Mode Status',
    CASE 
        WHEN (SELECT COUNT(*) FROM transactions WHERE session_id = (SELECT MAX(session_id) FROM transactions)) <= 10 
        THEN '✅ 10-Transaction Mode Active'
        ELSE '⚠️ Production Mode Active'
    END,
    CONCAT('Latest session: ', (SELECT COUNT(*) FROM transactions WHERE session_id = (SELECT MAX(session_id) FROM transactions)), ' transactions');

-- ==============================================
-- USAGE INSTRUCTIONS:
-- ==============================================
-- 1. Run these queries one by one in your MySQL client
-- 2. Check the results to verify database functionality
-- 3. Pay special attention to Test 24 and 25 for 10-transaction verification
-- 4. Use Test 26 for a complete system status overview
-- 5. If any query fails, check your database connection and table structure
