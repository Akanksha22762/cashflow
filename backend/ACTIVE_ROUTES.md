# Active Routes Used by Frontend

## Routes Actually Used:

### Upload Routes
- ✅ `/upload` (POST) - File upload

### Data Routes  
- ✅ `/get-current-data` (GET) - Get current transaction data
- ✅ `/get-dropdown-data` (GET) - Get vendor dropdown data

### Vendor Routes
- ✅ `/vendor-analysis` (POST) - Analyze vendor
- ✅ `/view_vendor_transactions/<vendor_name>` (GET) - View vendor transactions
- ✅ `/extract-vendors-for-analysis` (POST) - Extract vendors

### Transaction Routes
- ✅ `/transaction-analysis` (POST) - Transaction analysis

### Report Routes
- ✅ `/comprehensive-report` (GET) - Get comprehensive report
- ✅ `/comprehensive-report.pdf` (GET) - Get PDF report

### Analytics Routes
- ✅ `/get-available-trend-types` (GET) - Get trend types
- ✅ `/validate-trend-selection` (POST) - Validate trend selection
- ✅ `/run-dynamic-trends-analysis` (POST) - Run trends analysis

### AI Reasoning Routes
- ✅ `/ai-reasoning/categorization` (POST) - AI reasoning for categorization
- ✅ `/ai-reasoning/vendor-landscape` (POST) - AI reasoning for vendor landscape
- ✅ `/ai-reasoning/trend-analysis` (POST) - AI reasoning for trend analysis

### System Routes
- ✅ `/status` (GET) - Status check
- ✅ `/` (GET) - Root route

## ❌ Routes NOT Used (Dead Code - Skip)
- All other routes in app.py

