-- ===== CASHFLOW SAP BANK SYSTEM - COMPLETE DATABASE VERIFICATION QUERIES =====
-- Run these queries in MySQL to verify your entire system

-- 1. CHECK DATABASE AND TABLES STRUCTURE
-- =============================================

-- Show all databases
SHOW DATABASES;

-- Use the cashflow database
USE cashflow;

-- Show all tables in the system
    SHOW TABLES;

-- Get detailed table structures
DESCRIBE files;
DESCRIBE analysis_sessions;
DESCRIBE transactions;
DESCRIBE ai_model_performance;
DESCRIBE session_states;

-- 2. CHECK FILE UPLOADS AND SESSIONS
-- ===================================

-- Show all uploaded files
SELECT 
    file_id,
    filename,
    file_hash,
    data_source,
    file_size,
    upload_timestamp,
    processing_status,
    completed_at
FROM files 
ORDER BY upload_timestamp DESC 
LIMIT 10;

-- Show all analysis sessions
SELECT 
    session_id,
    file_id,
    analysis_type,
    started_at,
    completed_at,
    transaction_count,
    processing_time_seconds,
    success_rate,
    status,
    ollama_model
FROM analysis_sessions 
ORDER BY started_at DESC 
LIMIT 10;

-- 3. CHECK TRANSACTION DATA (BANK & SAP)
-- ======================================

-- Count total transactions by data source
SELECT 
    f.data_source,
    COUNT(t.transaction_id) as transaction_count,
    SUM(t.amount) as total_amount
FROM transactions t
JOIN files f ON t.file_id = f.file_id
GROUP BY f.data_source;

-- Show recent transactions with AI categorization
SELECT 
    t.transaction_id,
    t.session_id,
    t.transaction_date,
    t.description,
    t.amount,
    t.ai_category,
    t.vendor_name,
    t.ai_confidence_score,
    f.data_source
FROM transactions t
JOIN files f ON t.file_id = f.file_id
ORDER BY t.created_at DESC 
LIMIT 20;

-- Check AI categorization distribution
SELECT 
    ai_category,
    COUNT(*) as count,
    AVG(ai_confidence_score) as avg_confidence,
    SUM(amount) as total_amount
FROM transactions 
WHERE ai_category IS NOT NULL
GROUP BY ai_category
ORDER BY count DESC;

-- 4. CHECK VENDOR ASSIGNMENTS (OLLAMA)
-- ====================================

-- Show vendor distribution
SELECT 
    vendor_name,
    COUNT(*) as transaction_count,
    SUM(amount) as total_amount,
    AVG(ai_confidence_score) as avg_confidence
FROM transactions 
WHERE vendor_name IS NOT NULL AND vendor_name != ''
GROUP BY vendor_name
ORDER BY transaction_count DESC;

-- Check transactions with vendor assignments
SELECT 
    COUNT(*) as total_transactions,
    SUM(CASE WHEN vendor_name IS NOT NULL AND vendor_name != '' THEN 1 ELSE 0 END) as with_vendors,
    ROUND(
        (SUM(CASE WHEN vendor_name IS NOT NULL AND vendor_name != '' THEN 1 ELSE 0 END) * 100.0 / COUNT(*)), 
        2
    ) as vendor_assignment_percentage
FROM transactions;

-- 5. CHECK AI MODEL PERFORMANCE
-- =============================

-- Show AI model performance metrics
SELECT 
    amp.session_id,
    amp.model_name,
    amp.model_version,
    amp.total_predictions,
    amp.successful_predictions,
    amp.failed_predictions,
    amp.average_confidence,
    amp.processing_time_ms,
    amp.memory_usage_mb,
    amp.created_at
FROM ai_model_performance amp
ORDER BY amp.created_at DESC 
LIMIT 10;

-- Calculate overall AI performance statistics
SELECT 
    model_name,
    COUNT(*) as runs,
    AVG(total_predictions) as avg_predictions,
    AVG(successful_predictions) as avg_successful,
    AVG(average_confidence) as avg_confidence,
    AVG(processing_time_ms) as avg_processing_time,
    AVG(error_rate) as avg_error_rate
FROM ai_model_performance
GROUP BY model_name;

-- 6. CHECK SESSION PERSISTENCE DATA
-- =================================

-- Show all session states
SELECT 
    ss.state_id,
    ss.session_id,
    ss.state_type,
    LENGTH(ss.state_data) as data_size_bytes,
    ss.user_session_id,
    ss.created_at,
    ss.updated_at
FROM session_states ss
ORDER BY ss.updated_at DESC 
LIMIT 15;

-- Count session states by type
SELECT 
    state_type,
    COUNT(*) as count,
    AVG(LENGTH(state_data)) as avg_data_size
FROM session_states
GROUP BY state_type;

-- Show latest session with all state types
SELECT 
    ss.session_id,
    GROUP_CONCAT(ss.state_type) as available_states,
    COUNT(*) as state_count,
    MAX(ss.updated_at) as last_updated
FROM session_states ss
GROUP BY ss.session_id
ORDER BY last_updated DESC
LIMIT 5;

-- 7. CHECK TRENDS ANALYSIS RESULTS
-- ================================

-- Find sessions with trends analysis
SELECT DISTINCT
    ss.session_id,
    ss.state_type,
    ss.updated_at,
    LENGTH(ss.state_data) as data_size
FROM session_states ss
WHERE ss.state_data LIKE '%trends_analysis%' 
   OR ss.state_data LIKE '%Enhanced AI Analysis%'
   OR ss.state_data LIKE '%XGBoost%'
   OR ss.state_data LIKE '%Ollama%'
ORDER BY ss.updated_at DESC;

-- 8. CHECK CATEGORIES ANALYSIS RESULTS
-- ====================================

-- Find sessions with category analysis
SELECT DISTINCT
    ss.session_id,
    ss.state_type,
    ss.updated_at
FROM session_states ss
WHERE ss.state_data LIKE '%categories_analysis%' 
   OR ss.state_data LIKE '%Operating Activities%'
   OR ss.state_data LIKE '%Investing Activities%'
   OR ss.state_data LIKE '%Financing Activities%'
ORDER BY ss.updated_at DESC;

-- 9. CHECK VENDOR EXTRACTION RESULTS
-- ==================================

-- Find sessions with vendor extraction
SELECT DISTINCT
    ss.session_id,
    ss.state_type,
    ss.updated_at
FROM session_states ss
WHERE ss.state_data LIKE '%vendor_extraction%' 
   OR ss.state_data LIKE '%Assigned_Vendor%'
   OR ss.state_data LIKE '%vendor_assignments%'
ORDER BY ss.updated_at DESC;

-- 10. COMPREHENSIVE SYSTEM HEALTH CHECK
-- =====================================

-- Overall system statistics
SELECT 
    'Total Files' as metric,
    COUNT(*) as value
FROM files
UNION ALL
SELECT 
    'Total Sessions' as metric,
    COUNT(*) as value
FROM analysis_sessions
UNION ALL
SELECT 
    'Total Transactions' as metric,
    COUNT(*) as value
FROM transactions
UNION ALL
SELECT 
    'AI Categorized Transactions' as metric,
    COUNT(*) as value
FROM transactions 
WHERE ai_category IS NOT NULL
UNION ALL
SELECT 
    'Vendor Assigned Transactions' as metric,
    COUNT(*) as value
FROM transactions 
WHERE vendor_name IS NOT NULL AND vendor_name != ''
UNION ALL
SELECT 
    'Session States Stored' as metric,
    COUNT(*) as value
FROM session_states
UNION ALL
SELECT 
    'AI Model Runs' as metric,
    COUNT(*) as value
FROM ai_model_performance;

-- 11. LATEST SESSION DETAILED CHECK
-- =================================

-- Get the most recent session details
SELECT 
    'Latest Session ID' as info,
    CAST(MAX(session_id) as CHAR) as value
FROM analysis_sessions
UNION ALL
SELECT 
    'Latest Session Transaction Count' as info,
    CAST(transaction_count as CHAR) as value
FROM analysis_sessions 
WHERE session_id = (SELECT MAX(session_id) FROM analysis_sessions)
UNION ALL
SELECT 
    'Latest Session Success Rate' as info,
    CAST(success_rate as CHAR) as value
FROM analysis_sessions 
WHERE session_id = (SELECT MAX(session_id) FROM analysis_sessions);

-- Show latest session's transactions
SELECT 
    t.transaction_date,
    t.description,
    t.amount,
    t.ai_category,
    t.vendor_name,
    t.ai_confidence_score
FROM transactions t
WHERE t.session_id = (SELECT MAX(session_id) FROM analysis_sessions)
ORDER BY t.original_row_number
LIMIT 10;

-- 12. DATA INTEGRITY CHECKS
-- =========================

-- Check for orphaned transactions (transactions without sessions)
SELECT COUNT(*) as orphaned_transactions
FROM transactions t
LEFT JOIN analysis_sessions s ON t.session_id = s.session_id
WHERE s.session_id IS NULL;

-- Check for sessions without transactions
SELECT COUNT(*) as empty_sessions
FROM analysis_sessions s
LEFT JOIN transactions t ON s.session_id = t.session_id
WHERE t.session_id IS NULL;

-- Check for duplicate transactions
SELECT 
    session_id,
    transaction_date,
    description,
    amount,
    COUNT(*) as duplicate_count
FROM transactions
GROUP BY session_id, transaction_date, description, amount
HAVING COUNT(*) > 1;

-- 13. STORAGE SIZE ANALYSIS
-- ========================

-- Check table sizes
SELECT 
    table_name,
    table_rows,
    ROUND((data_length + index_length) / 1024 / 1024, 2) as size_mb
FROM information_schema.tables 
WHERE table_schema = 'cashflow'
ORDER BY (data_length + index_length) DESC;

-- 14. CHECK BUSINESS INSIGHTS (MySQL SPECIFIC)
-- ============================================

-- Show business insights generated
SELECT 
    insight_type,
    insight_name,
    insight_value,
    risk_level,
    priority_level,
    business_impact_score,
    confidence_score,
    trend_indicator
FROM business_insights
ORDER BY created_at DESC
LIMIT 10;

-- 15. CHECK ANALYSIS RESULTS (MySQL SPECIFIC)
-- ===========================================

-- Show analysis results by type
SELECT 
    analysis_type,
    analysis_subtype,
    COUNT(*) as result_count,
    AVG(confidence_score) as avg_confidence,
    AVG(processing_time) as avg_processing_time
FROM analysis_results
GROUP BY analysis_type, analysis_subtype
ORDER BY result_count DESC;

-- Show latest analysis results
SELECT 
    ar.result_id,
    ar.analysis_type,
    ar.analysis_subtype,
    ar.transaction_count,
    ar.confidence_score,
    ar.ai_model_used,
    ar.created_at
FROM analysis_results ar
ORDER BY ar.created_at DESC
LIMIT 10;

-- 16. CHECK TRENDS ANALYSIS (MySQL SPECIFIC)
-- ==========================================

-- Show trends analysis results
SELECT 
    trend_type,
    trend_period,
    category_name,
    vendor_name,
    transaction_count,
    total_amount,
    trend_direction,
    trend_percentage,
    confidence_score
FROM trends_analysis
ORDER BY created_at DESC
LIMIT 15;

-- 17. CHECK VENDOR ANALYSIS (MySQL SPECIFIC)
-- ==========================================

-- Show vendor analysis results
SELECT 
    vendor_name,
    vendor_category,
    total_transactions,
    total_amount,
    average_amount,
    payment_frequency,
    risk_score,
    reliability_score,
    business_impact
FROM vendor_analysis
ORDER BY total_amount DESC
LIMIT 10;

-- 18. CHECK SYSTEM HEALTH (MySQL SPECIFIC)
-- ========================================

-- Check error logs
SELECT 
    error_type,
    COUNT(*) as error_count,
    MAX(occurred_at) as last_occurrence
FROM error_logs
GROUP BY error_type
ORDER BY error_count DESC;

-- Check system metrics
SELECT 
    metric_name,
    AVG(metric_value) as avg_value,
    MIN(metric_value) as min_value,
    MAX(metric_value) as max_value,
    metric_unit
FROM system_metrics
GROUP BY metric_name, metric_unit
ORDER BY metric_name;

-- 19. VENDOR EXTRACTION VERIFICATION (COMPREHENSIVE)
-- ==================================================

-- Check vendor assignment overview
SELECT 
    COUNT(*) as total_transactions,
    COUNT(vendor_name) as transactions_with_vendors,
    COUNT(DISTINCT vendor_name) as unique_vendors,
    ROUND((COUNT(vendor_name) * 100.0 / COUNT(*)), 2) as vendor_assignment_percentage
FROM transactions;

-- List all extracted vendors with details
SELECT 
    vendor_name,
    COUNT(*) as transaction_count,
    SUM(amount) as total_amount,
    AVG(amount) as avg_amount,
    AVG(ai_confidence_score) as avg_confidence,
    MIN(transaction_date) as first_transaction,
    MAX(transaction_date) as last_transaction
FROM transactions 
WHERE vendor_name IS NOT NULL 
GROUP BY vendor_name
ORDER BY transaction_count DESC;

-- Check vendor extraction progress by session
SELECT 
    session_id,
    COUNT(*) as total_transactions,
    COUNT(vendor_name) as with_vendors,
    COUNT(DISTINCT vendor_name) as unique_vendors,
    ROUND((COUNT(vendor_name) * 100.0 / COUNT(*)), 2) as extraction_rate,
    MAX(created_at) as session_date
FROM transactions
GROUP BY session_id
HAVING COUNT(*) > 0
ORDER BY session_id DESC
LIMIT 15;

-- Check vendor assignments in latest session
SELECT 
    transaction_date,
    description,
    amount,
    vendor_name,
    ai_confidence_score,
    ai_category
FROM transactions 
WHERE session_id = (SELECT MAX(session_id) FROM analysis_sessions)
ORDER BY original_row_number
LIMIT 20;

-- Show vendor extraction failures (transactions without vendors)
SELECT 
    COUNT(*) as transactions_without_vendors,
    ROUND((COUNT(*) * 100.0 / (SELECT COUNT(*) FROM transactions)), 2) as failure_percentage
FROM transactions 
WHERE vendor_name IS NULL OR vendor_name = '';

-- Sample transactions without vendor assignments (for debugging)
SELECT 
    transaction_id,
    description,
    amount,
    ai_category,
    session_id
FROM transactions 
WHERE vendor_name IS NULL OR vendor_name = ''
ORDER BY created_at DESC
LIMIT 10;

-- Check vendor categories distribution
SELECT 
    CASE 
        WHEN vendor_name LIKE '%Bank%' THEN 'Banking'
        WHEN vendor_name LIKE '%Medical%' OR vendor_name LIKE '%Health%' THEN 'Healthcare'
        WHEN vendor_name LIKE '%Equipment%' OR vendor_name LIKE '%Machine%' THEN 'Equipment'
        WHEN vendor_name LIKE '%Corp%' OR vendor_name LIKE '%Ltd%' OR vendor_name LIKE '%Inc%' THEN 'Corporate'
        ELSE 'Other'
    END as vendor_category,
    COUNT(*) as transaction_count,
    COUNT(DISTINCT vendor_name) as unique_vendors,
    SUM(amount) as total_amount
FROM transactions 
WHERE vendor_name IS NOT NULL
GROUP BY vendor_category
ORDER BY transaction_count DESC;

-- 20. SESSION PERSISTENCE VENDOR CHECK
-- ====================================

-- Check if vendor assignments are saved in session states
SELECT 
    ss.session_id,
    ss.state_type,
    LENGTH(ss.state_data) as data_size_bytes,
    ss.created_at,
    ss.updated_at,
    CASE 
        WHEN ss.state_data LIKE '%Assigned_Vendor%' THEN 'Contains Vendor Data'
        WHEN ss.state_data LIKE '%vendor_assignments%' THEN 'Contains Vendor Assignments'
        WHEN ss.state_data LIKE '%vendor%' THEN 'Contains Vendor Info'
        ELSE 'No Vendor Data'
    END as vendor_data_status
FROM session_states ss
WHERE ss.state_data LIKE '%vendor%' 
   OR ss.state_data LIKE '%Assigned_Vendor%'
   OR ss.state_data LIKE '%vendor_assignments%'
ORDER BY ss.updated_at DESC
LIMIT 10;

-- 21. VENDOR ANALYSIS RESULTS CHECK
-- =================================

-- Check dedicated vendor analysis table results
SELECT 
    vendor_name,
    vendor_category,
    total_transactions,
    total_amount,
    average_amount,
    risk_score,
    reliability_score,
    business_impact,
    payment_frequency,
    created_at
FROM vendor_analysis
ORDER BY total_amount DESC
LIMIT 15;

-- Count vendor analysis entries
SELECT 
    COUNT(*) as total_vendor_profiles,
    COUNT(DISTINCT vendor_category) as unique_categories,
    AVG(risk_score) as avg_risk_score,
    AVG(reliability_score) as avg_reliability_score
FROM vendor_analysis;

-- 22. TRENDS ANALYSIS VERIFICATION (COMPREHENSIVE)
-- ================================================

-- Check trends analysis overview
SELECT 
    COUNT(*) as total_trend_records,
    COUNT(DISTINCT trend_type) as unique_trend_types,
    COUNT(DISTINCT file_id) as files_analyzed,
    COUNT(DISTINCT analysis_session_id) as sessions_analyzed,
    MIN(created_at) as first_analysis,
    MAX(created_at) as last_analysis
FROM trends_analysis;

-- Show all trend types and their frequency
SELECT 
    trend_type,
    COUNT(*) as analysis_count,
    COUNT(DISTINCT file_id) as unique_files,
    AVG(confidence_score) as avg_confidence,
    AVG(total_amount) as avg_amount_analyzed
FROM trends_analysis
GROUP BY trend_type
ORDER BY analysis_count DESC;

-- Latest trends analysis results
SELECT 
    ta.trend_id,
    ta.trend_type,
    ta.trend_period,
    ta.category_name,
    ta.vendor_name,
    ta.transaction_count,
    ta.total_amount,
    ta.trend_direction,
    ta.trend_percentage,
    ta.confidence_score,
    ta.created_at
FROM trends_analysis ta
ORDER BY ta.created_at DESC
LIMIT 20;

-- Trends analysis by category
SELECT 
    category_name,
    trend_direction,
    COUNT(*) as trend_count,
    AVG(trend_percentage) as avg_trend_percentage,
    AVG(confidence_score) as avg_confidence,
    SUM(total_amount) as total_amount_analyzed
FROM trends_analysis
WHERE category_name IS NOT NULL
GROUP BY category_name, trend_direction
ORDER BY category_name, trend_count DESC;

-- Vendor-specific trends
SELECT 
    vendor_name,
    trend_direction,
    COUNT(*) as trend_analyses,
    AVG(trend_percentage) as avg_growth_rate,
    AVG(total_amount) as avg_amount,
    MAX(created_at) as latest_analysis
FROM trends_analysis
WHERE vendor_name IS NOT NULL
GROUP BY vendor_name, trend_direction
ORDER BY avg_growth_rate DESC
LIMIT 15;

-- Seasonal and periodic trends summary
SELECT 
    trend_type,
    trend_period,
    COUNT(*) as analysis_count,
    AVG(growth_rate) as avg_growth_rate,
    AVG(volatility_score) as avg_volatility,
    COUNT(DISTINCT CASE WHEN trend_direction = 'increasing' THEN trend_id END) as increasing_trends,
    COUNT(DISTINCT CASE WHEN trend_direction = 'decreasing' THEN trend_id END) as decreasing_trends,
    COUNT(DISTINCT CASE WHEN trend_direction = 'stable' THEN trend_id END) as stable_trends
FROM trends_analysis
GROUP BY trend_type, trend_period
ORDER BY analysis_count DESC;

-- Trend performance by session
SELECT 
    analysis_session_id,
    COUNT(*) as trends_generated,
    COUNT(DISTINCT trend_type) as unique_trend_types,
    AVG(confidence_score) as avg_confidence,
    MAX(created_at) as analysis_date,
    STRING_AGG(DISTINCT trend_type, ', ') as trend_types_analyzed
FROM trends_analysis
GROUP BY analysis_session_id
ORDER BY analysis_session_id DESC
LIMIT 10;

-- High confidence trends (AI quality check)
SELECT 
    trend_type,
    category_name,
    vendor_name,
    trend_direction,
    trend_percentage,
    confidence_score,
    ai_insights,
    created_at
FROM trends_analysis
WHERE confidence_score >= 0.8
ORDER BY confidence_score DESC, created_at DESC
LIMIT 15;

-- Trend analysis failures (low confidence)
SELECT 
    trend_type,
    category_name,
    confidence_score,
    ai_insights,
    created_at
FROM trends_analysis
WHERE confidence_score < 0.5
ORDER BY confidence_score ASC
LIMIT 10;

-- 23. ANALYSIS RESULTS TABLE TRENDS CHECK
-- =======================================

-- Check trends in analysis_results table
SELECT 
    analysis_type,
    analysis_subtype,
    COUNT(*) as analysis_count,
    AVG(confidence_score) as avg_confidence,
    AVG(processing_time) as avg_processing_time,
    AVG(transaction_count) as avg_transactions_analyzed,
    MAX(created_at) as latest_analysis
FROM analysis_results
WHERE analysis_type = 'trends'
GROUP BY analysis_type, analysis_subtype
ORDER BY analysis_count DESC;

-- Latest trends analysis detailed results
SELECT 
    ar.result_id,
    ar.analysis_subtype,
    ar.transaction_count,
    ar.confidence_score,
    ar.processing_time,
    ar.ai_model_used,
    ar.ai_insights,
    ar.created_at
FROM analysis_results ar
WHERE ar.analysis_type = 'trends'
ORDER BY ar.created_at DESC
LIMIT 10;

-- Trends analysis performance metrics
SELECT 
    ai_model_used,
    COUNT(*) as analyses_performed,
    AVG(confidence_score) as avg_confidence,
    AVG(processing_time) as avg_processing_time,
    MIN(processing_time) as min_processing_time,
    MAX(processing_time) as max_processing_time
FROM analysis_results
WHERE analysis_type = 'trends'
GROUP BY ai_model_used
ORDER BY analyses_performed DESC;

-- 24. SESSION PERSISTENCE TRENDS CHECK
-- ====================================

-- Check if trends analysis results are saved in session states
SELECT 
    ss.session_id,
    ss.state_type,
    LENGTH(ss.state_data) as data_size_bytes,
    ss.created_at,
    ss.updated_at,
    CASE 
        WHEN ss.state_data LIKE '%trends_analysis%' THEN 'Contains Trends Analysis'
        WHEN ss.state_data LIKE '%XGBoost%' THEN 'Contains XGBoost Results'
        WHEN ss.state_data LIKE '%Ollama%' THEN 'Contains Ollama Analysis'
        WHEN ss.state_data LIKE '%forecast%' THEN 'Contains Forecast Data'
        ELSE 'Other Trends Data'
    END as trends_data_status
FROM session_states ss
WHERE ss.state_data LIKE '%trend%' 
   OR ss.state_data LIKE '%forecast%'
   OR ss.state_data LIKE '%XGBoost%'
   OR ss.state_data LIKE '%Ollama%'
   OR ss.state_data LIKE '%Enhanced AI Analysis%'
ORDER BY ss.updated_at DESC
LIMIT 15;

-- 25. HYBRID AI MODELS VERIFICATION
-- =================================

-- Check XGBoost + Ollama model performance in AI performance table
SELECT 
    model_name,
    model_version,
    COUNT(*) as runs,
    AVG(total_predictions) as avg_predictions,
    AVG(successful_predictions) as avg_successful,
    AVG(average_confidence) as avg_confidence,
    AVG(processing_time_ms) as avg_processing_time,
    AVG(error_rate) as avg_error_rate,
    MAX(created_at) as last_run
FROM ai_model_performance
WHERE model_name IN ('XGBoost', 'ollama', 'ensemble', 'hybrid')
   OR model_name LIKE '%ollama%'
   OR model_name LIKE '%xgb%'
GROUP BY model_name, model_version
ORDER BY runs DESC;

-- Check latest AI model performance for trends
SELECT 
    amp.session_id,
    amp.model_name,
    amp.model_version,
    amp.total_predictions,
    amp.successful_predictions,
    amp.average_confidence,
    amp.processing_time_ms,
    amp.created_at
FROM ai_model_performance amp
WHERE amp.created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
ORDER BY amp.created_at DESC
LIMIT 15;

-- 26. UI DROPDOWN DATA VERIFICATION
-- =================================

-- Check what populates the VENDOR dropdown
SELECT DISTINCT 
    vendor_name,
    COUNT(*) as transaction_count,
    SUM(amount) as total_amount
FROM transactions 
WHERE vendor_name IS NOT NULL 
  AND vendor_name != ''
  AND vendor_name != 'NULL'
GROUP BY vendor_name
ORDER BY transaction_count DESC;

-- Check what populates the CATEGORY dropdown (AI Classifications)
SELECT DISTINCT 
    ai_category,
    COUNT(*) as transaction_count,
    AVG(ai_confidence_score) as avg_confidence
FROM transactions 
WHERE ai_category IS NOT NULL 
  AND ai_category != ''
GROUP BY ai_category
ORDER BY transaction_count DESC;

-- Check what populates the TRENDS TYPE dropdown
SELECT DISTINCT 
    analysis_subtype as trend_type,
    COUNT(*) as analysis_count,
    MAX(created_at) as last_performed
FROM analysis_results
WHERE analysis_type = 'trends'
GROUP BY analysis_subtype
ORDER BY analysis_count DESC;

-- Check available SESSIONS for session restore dropdown
SELECT 
    session_id,
    started_at,
    completed_at,
    transaction_count,
    status,
    CASE 
        WHEN transaction_count > 0 THEN 'Has Data'
        ELSE 'Empty'
    END as data_status
FROM analysis_sessions
WHERE status = 'completed'
ORDER BY session_id DESC
LIMIT 20;

-- Check available FILES for file selection
SELECT 
    file_id,
    filename,
    data_source,
    file_size,
    upload_timestamp,
    processing_status
FROM files
ORDER BY upload_timestamp DESC
LIMIT 15;

-- 27. FRONTEND DATA VERIFICATION  
-- ==============================

-- Data for DASHBOARD summary cards
SELECT 
    'Total Transactions' as metric,
    COUNT(*) as value
FROM transactions
UNION ALL
SELECT 
    'AI Categorized',
    COUNT(*)
FROM transactions 
WHERE ai_category IS NOT NULL
UNION ALL
SELECT 
    'Vendor Assigned',
    COUNT(*)
FROM transactions 
WHERE vendor_name IS NOT NULL AND vendor_name != ''
UNION ALL
SELECT 
    'Analysis Sessions',
    COUNT(*)
FROM analysis_sessions
UNION ALL
SELECT 
    'Trends Analyses',
    COUNT(*)
FROM analysis_results
WHERE analysis_type = 'trends';

-- Data for CATEGORY BREAKDOWN chart/table
SELECT 
    ai_category,
    COUNT(*) as transaction_count,
    SUM(amount) as total_amount,
    ROUND(AVG(amount), 2) as avg_amount,
    ROUND((COUNT(*) * 100.0 / (SELECT COUNT(*) FROM transactions WHERE ai_category IS NOT NULL)), 2) as percentage
FROM transactions 
WHERE ai_category IS NOT NULL
GROUP BY ai_category
ORDER BY transaction_count DESC;

-- Data for VENDOR ANALYSIS table
SELECT 
    vendor_name,
    COUNT(*) as transactions,
    ROUND(SUM(amount), 2) as total_amount,
    ROUND(AVG(amount), 2) as avg_amount,
    MIN(transaction_date) as first_transaction,
    MAX(transaction_date) as last_transaction
FROM transactions 
WHERE vendor_name IS NOT NULL AND vendor_name != ''
GROUP BY vendor_name
ORDER BY total_amount DESC
LIMIT 20;

-- Data for RECENT ACTIVITY feed
SELECT 
    'Transaction' as activity_type,
    CONCAT('Transaction: ', LEFT(description, 50)) as activity_description,
    amount as activity_value,
    created_at as activity_time
FROM transactions
WHERE created_at >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
UNION ALL
SELECT 
    'Analysis' as activity_type,
    CONCAT('Analysis: ', analysis_type, ' - ', IFNULL(analysis_subtype, 'General')) as activity_description,
    confidence_score as activity_value,
    created_at as activity_time
FROM analysis_results
WHERE created_at >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
ORDER BY activity_time DESC
LIMIT 15;

-- 28. SPECIFIC UI COMPONENT QUERIES
-- =================================

-- TRENDS ANALYSIS dropdown options (what you can select)
SELECT DISTINCT 
    analysis_subtype as available_trend_type,
    COUNT(*) as times_performed,
    AVG(confidence_score) as avg_success_rate,
    MAX(created_at) as last_performed
FROM analysis_results
WHERE analysis_type = 'trends'
  AND analysis_subtype IS NOT NULL
GROUP BY analysis_subtype
ORDER BY times_performed DESC;

-- VENDOR FILTER dropdown (for "Run Analysis" vendor selection)
SELECT 
    vendor_name as dropdown_option,
    COUNT(*) as transaction_count,
    CONCAT('₹', FORMAT(SUM(amount), 2)) as total_value,
    CONCAT(vendor_name, ' (', COUNT(*), ' transactions)') as display_text
FROM transactions 
WHERE vendor_name IS NOT NULL 
  AND vendor_name != ''
  AND vendor_name != 'NULL'
GROUP BY vendor_name
HAVING COUNT(*) >= 1  -- Only vendors with at least 1 transaction
ORDER BY transaction_count DESC;

-- SESSION RESTORE dropdown (sessions available for restore)
SELECT 
    session_id,
    CONCAT('Session ', session_id, ' - ', DATE_FORMAT(started_at, '%Y-%m-%d %H:%i')) as display_text,
    transaction_count,
    CASE 
        WHEN status = 'completed' THEN '✅ Complete'
        WHEN status = 'failed' THEN '❌ Failed'
        ELSE status
    END as status_display,
    started_at
FROM analysis_sessions
WHERE transaction_count > 0
ORDER BY session_id DESC
LIMIT 25;

-- CATEGORY FILTER options (for transaction filtering)
SELECT DISTINCT 
    ai_category as filter_option,
    COUNT(*) as available_transactions,
    CONCAT(ai_category, ' (', COUNT(*), ')') as display_text
FROM transactions 
WHERE ai_category IS NOT NULL 
  AND ai_category != ''
GROUP BY ai_category
ORDER BY available_transactions DESC;

-- FILE UPLOAD history (recent uploads for reference)
SELECT 
    file_id,
    filename,
    CONCAT(filename, ' - ', DATE_FORMAT(upload_timestamp, '%Y-%m-%d')) as display_text,
    data_source,
    processing_status,
    upload_timestamp
FROM files
ORDER BY upload_timestamp DESC
LIMIT 10;
-- 
-- ✅ FIXED ISSUES:
-- - file_path → file_hash (files table)
-- - upload_date → upload_timestamp (files table)  
-- - start_time → started_at (analysis_sessions table)
-- - end_time → completed_at (analysis_sessions table)
-- - processing_time → processing_time_seconds (analysis_sessions table)
-- - ai_confidence → ai_confidence_score (transactions table)
-- - Added error_rate column (ai_model_performance table)
-- - Added MySQL-specific tables queries
-- 
-- USAGE INSTRUCTIONS:
-- 1. Copy each section and run in MySQL Workbench or command line
-- 2. Check that all tables exist and have data
-- 3. Verify that AI categorization and vendor assignments are present
-- 4. Ensure session states are being saved properly
-- 5. Confirm that the latest session matches your current app state
-- 6. Use the MySQL-specific queries to check advanced features
