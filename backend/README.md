# Flask Backend - Cash Flow Analysis API

## 🚀 Overview
This is the Flask backend API for the Cash Flow Analysis System with AI/ML Enhancement. It provides intelligent financial analysis, vendor categorization, cash flow forecasting, and anomaly detection.

## 📁 Folder Structure
```
backend/
├── app.py                              # Main Flask application
├── requirements.txt                    # Python dependencies
├── templates/                          # Flask templates (if needed)
├── static/                            # Static files (CSS, images)
├── data/                              # Data storage
├── __pycache__/                       # Python cache
│
├── Core Modules:
├── advanced_revenue_ai_system.py      # Advanced revenue AI system
├── analysis_storage_integration.py    # Analysis storage
├── database_manager.py                # Database management
├── mysql_database_manager.py          # MySQL database manager
├── openai_integration.py              # OpenAI integration
├── enhanced_ai_reasoning.py           # Enhanced AI reasoning
├── universal_data_adapter.py          # Universal data adapter
├── universal_industry_system.py       # Industry classification system
│
├── Integration Modules:
├── data_adapter_integration.py
├── integrate_advanced_revenue_system.py
├── integrate_database_storage.py
├── persistent_state_manager.py
│
└── Test Files:
    ├── test_*.py                      # Various test files
    ├── database_test_queries.sql
    └── mysql_verification_queries.sql
```

## 🔧 Installation

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Install Flask-CORS (if not already installed)
```bash
pip install flask-cors
```

### 3. Set up Environment Variables
Create a `.env` file in the backend folder:
```env
OPENAI_API_KEY=your_openai_api_key_here
FLASK_ENV=development
FLASK_APP=app.py
```

## ▶️ Running the Backend

### Development Mode
```bash
cd backend
python app.py
```

The API will start on:
- **Local**: http://127.0.0.1:5000
- **Network**: http://192.168.x.x:5000

### Production Mode
```bash
cd backend
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## 📡 API Endpoints

### Core Endpoints

#### 1. **Root**
- **GET** `/`
- Returns API information and available endpoints

#### 2. **Upload & Analyze**
- **POST** `/upload`
- Upload financial data (CSV/Excel) for analysis
- **Body**: `multipart/form-data` with file upload

#### 3. **Status Check**
- **GET** `/status`
- Check system status and AI/ML model availability

### Vendor Analysis

- **POST** `/vendor-analysis` - Analyze specific vendors
- **POST** `/vendor-analysis-type` - Vendor analysis by type
- **GET** `/vendor_list` - Get all vendors
- **GET** `/view_vendor_transactions/<vendor_name>` - View vendor transactions
- **GET** `/view_vendor_cashflow/<vendor_name>` - View vendor cash flow
- **POST** `/extract-vendors-for-analysis` - Extract vendors using AI

### Transaction Analysis

- **POST** `/transaction-analysis` - Analyze transactions
- **POST** `/transaction-analysis-type` - Transaction analysis by type
- **POST** `/analysis-category` - Category-based analysis

### Advanced AI Features

- **POST** `/run-revenue-analysis` - AI-powered revenue analysis
- **POST** `/run-dynamic-trends-analysis` - Dynamic trend analysis
- **POST** `/run-parameter-analysis` - Parameter analysis with ML
- **POST** `/complete-analysis` - Complete financial analysis

### Forecasting & Predictions

- **POST** `/cash-flow-forecast` - Cash flow forecasting
- **POST** `/time-series-forecast` - Time series forecasting
- **POST** `/anomaly-detection` - Detect financial anomalies

### Data Management

- **GET** `/view/<data_type>` - View different data types
- **GET** `/download/<data_type>` - Download analysis results
- **GET** `/download_vendor_cashflow` - Download vendor cash flow
- **POST** `/save-current-state` - Save current analysis state
- **POST** `/restore-session` - Restore previous session

### Reporting

- **POST** `/generate-report` - Generate analysis reports
- **GET** `/download-anomaly-report` - Download anomaly reports
- **GET** `/download-forecast-report` - Download forecast reports

### ML Model Management

- **POST** `/train-ml-models` - Train ML models
- **GET** `/test-ml-models` - Test ML model performance

## 🔄 Integration with Next.js Frontend

The backend is configured with CORS to allow requests from:
- `http://localhost:3000`
- `http://127.0.0.1:3000`

### Example API Call from Frontend:
```javascript
// Upload file for analysis
const formData = new FormData();
formData.append('bank_file', bankFile);

const response = await fetch('http://localhost:5000/upload', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log(result);
```

## 🧪 Testing

### Test the API
```bash
# Visit in browser
http://localhost:5000/

# Or use curl
curl http://localhost:5000/status
```

### Run Tests
```bash
cd backend
python test_database_connection.py
python test_openai_integration.py
```

## 🤖 AI/ML Features

- **XGBoost** - Machine learning classification
- **OpenAI GPT-4** - Natural language processing
- **Sentence Transformers** - Text embeddings
- **TensorFlow** - Deep learning models

## 📊 What You Get

When uploading files, the API returns:

```json
{
  "status": "success",
  "cash_flow_summary": {
    "Operating Activities": {
      "total": 150000,
      "percentage": 75
    },
    "Investing Activities": {
      "total": 30000,
      "percentage": 15
    },
    "Financing Activities": {
      "total": 20000,
      "percentage": 10
    }
  },
  "vendor_analysis": [...],
  "monthly_trends": [...],
  "ai_insights": [...],
  "predictions": {...}
}
```

## 🛠️ Tech Stack

- **Flask** - Web framework
- **XGBoost** - ML classification
- **OpenAI** - AI reasoning
- **Pandas** - Data processing
- **NumPy** - Numerical computing
- **SQLAlchemy** - Database ORM

## 📝 Notes

- The backend runs on port **5000**
- The Next.js frontend runs on port **3000**
- Make sure both servers are running for full functionality
- OpenAI API key is required for AI features

## 🆘 Troubleshooting

### Error: Template not found
- The backend now works as an API, not serving HTML templates
- Use the Next.js frontend at http://localhost:3000

### Error: CORS issues
- Make sure `flask-cors` is installed
- Check that the frontend URL is in the CORS configuration

### Error: Module not found
- Run `pip install -r requirements.txt` in the backend folder

## 📄 License

All rights reserved.

