# 📁 Project Structure - Full Stack Cash Flow Analysis System

## 🎯 Overview
This is a **Full Stack Application** with:
- **Frontend**: Next.js/React (TypeScript) - Port 3000
- **Backend**: Flask/Python (AI/ML) - Port 5000

---

## 📂 Folder Structure

```
Cashflow-main/
│
├── 🎨 FRONTEND (Next.js/React)
│   ├── app/                        # Next.js App Router
│   │   ├── page.tsx               # Home page
│   │   ├── layout.tsx             # Root layout
│   │   └── upload/                # Upload page
│   │       └── page.tsx
│   │
│   ├── components/                 # React components
│   │   ├── ui/                    # UI components
│   │   │   ├── button.tsx
│   │   │   ├── card.tsx
│   │   │   ├── badge.tsx
│   │   │   └── progress.tsx
│   │   ├── dashboard-layout.tsx
│   │   ├── upload-interface.tsx
│   │   ├── analysis-overview.tsx
│   │   └── theme-provider.tsx
│   │
│   ├── lib/                       # Utilities
│   │   └── utils.ts
│   │
│   ├── styles/                    # Global styles
│   │   └── globals.css
│   │
│   ├── public/                    # Static assets
│   │   ├── placeholder-logo.png
│   │   └── ...
│   │
│   ├── package.json               # Node dependencies
│   ├── tsconfig.json              # TypeScript config
│   ├── next.config.mjs            # Next.js config
│   └── components.json            # shadcn/ui config
│
└── 🔧 BACKEND (Flask/Python)
    ├── backend/
    │   ├── app.py                 # 🚀 Main Flask application
    │   ├── requirements.txt       # Python dependencies
    │   ├── README.md             # Backend documentation
    │   │
    │   ├── 📁 Core Modules
    │   ├── advanced_revenue_ai_system.py
    │   ├── openai_integration.py
    │   ├── database_manager.py
    │   ├── mysql_database_manager.py
    │   ├── enhanced_ai_reasoning.py
    │   ├── universal_data_adapter.py
    │   └── universal_industry_system.py
    │   │
    │   ├── 📁 Integration Modules
    │   ├── analysis_storage_integration.py
    │   ├── data_adapter_integration.py
    │   ├── integrate_advanced_revenue_system.py
    │   └── persistent_state_manager.py
    │   │
    │   ├── 📁 Flask Assets
    │   ├── templates/             # Flask HTML templates
    │   ├── static/                # Static files (CSS, images)
    │   └── data/                  # Data storage
    │   │
    │   ├── 📁 Tests
    │   ├── test_*.py              # Test files
    │   └── *_queries.sql          # SQL test queries
    │   │
    │   └── 📁 Startup Scripts
    │       ├── start_backend.ps1  # Windows startup
    │       └── start_backend.sh   # Linux/Mac startup
    │
    └── 📝 Documentation
        ├── README.md
        ├── DATABASE_SCHEMA.md
        ├── DEPLOYMENT_GUIDE.md
        └── ...

```

---

## 🚀 How to Run the Application

### Option 1: Run Both Servers Separately

#### Terminal 1 - Start Backend (Flask)
```bash
# Windows
cd backend
python app.py

# Or use the startup script
cd backend
.\start_backend.ps1
```

#### Terminal 2 - Start Frontend (Next.js)
```bash
# Development mode
npm run dev

# Production mode
npm run build
npm start
```

### Option 2: Quick Start (Windows)
```powershell
# Terminal 1
cd backend
.\start_backend.ps1

# Terminal 2
npm run dev
```

---

## 🌐 Access the Application

| Service | URL | Port |
|---------|-----|------|
| **Frontend (Next.js)** | http://localhost:3000 | 3000 |
| **Backend API (Flask)** | http://localhost:5000 | 5000 |
| **API Documentation** | http://localhost:5000/ | 5000 |

---

## 🔄 How They Work Together

```
┌─────────────────┐         HTTP/API         ┌──────────────────┐
│                 │    ------------------>    │                  │
│  Next.js        │                           │  Flask Backend   │
│  Frontend       │    <------------------    │  (AI/ML Engine)  │
│  (Port 3000)    │         JSON Response     │  (Port 5000)     │
└─────────────────┘                           └──────────────────┘
        │                                              │
        │                                              │
        v                                              v
  React Components                              AI/ML Processing
  - Upload Interface                            - XGBoost Models
  - Dashboard                                   - OpenAI Integration
  - Analytics Views                             - Data Analysis
  - Charts & Reports                            - Cash Flow Analysis
```

### Example Flow:
1. **User uploads CSV** → Frontend (`/upload`)
2. **Frontend sends file** → `fetch('http://localhost:5000/upload', ...)`
3. **Backend processes** → AI/ML analysis
4. **Backend returns JSON** → Analysis results
5. **Frontend displays** → Beautiful dashboard

---

## 📡 API Integration Example

### From Next.js Frontend to Flask Backend:

```typescript
// components/upload-interface.tsx
const uploadFile = async (file: File) => {
  const formData = new FormData();
  formData.append('bank_file', file);

  try {
    const response = await fetch('http://localhost:5000/upload', {
      method: 'POST',
      body: formData
    });

    const result = await response.json();
    
    if (result.status === 'success') {
      // Display analysis results
      console.log(result.cash_flow_summary);
      console.log(result.vendor_analysis);
      console.log(result.ai_insights);
    }
  } catch (error) {
    console.error('Upload failed:', error);
  }
};
```

---

## 🔧 Key Features

### Frontend (Next.js)
- ⚡ Modern React with TypeScript
- 🎨 Beautiful UI with shadcn/ui components
- 📊 Interactive dashboards
- 📈 Real-time data visualization
- 🌙 Dark mode support
- 📱 Responsive design

### Backend (Flask)
- 🤖 AI-powered analysis (OpenAI GPT-4)
- 🎯 ML classification (XGBoost)
- 💰 Cash flow forecasting
- 🔍 Anomaly detection
- 📊 Vendor analysis
- 📈 Trend analysis
- 💾 Session persistence

---

## 📦 Dependencies

### Frontend
```json
{
  "next": "15.2.4",
  "react": "19.0.0",
  "typescript": "5.0.0",
  "tailwindcss": "^3.4.1"
}
```

### Backend
```txt
flask>=2.0.0
flask-cors>=3.0.10
xgboost==3.0.2
scikit-learn==1.7.0
openai==1.93.1
pandas>=1.3.0
numpy>=1.21.0
```

---

## 🧪 Testing

### Test Backend API
```bash
# Check status
curl http://localhost:5000/status

# View API info
curl http://localhost:5000/

# Run tests
cd backend
python test_database_connection.py
python test_openai_integration.py
```

### Test Frontend
```bash
# Run dev server
npm run dev

# Build for production
npm run build

# Run production build
npm start
```

---

## 📝 Environment Variables

### Backend `.env`
```env
OPENAI_API_KEY=your_openai_api_key_here
FLASK_ENV=development
FLASK_APP=app.py
```

### Frontend `.env.local` (if needed)
```env
NEXT_PUBLIC_API_URL=http://localhost:5000
```

---

## 🔒 CORS Configuration

The backend is configured to accept requests from:
- `http://localhost:3000`
- `http://127.0.0.1:3000`

This allows the Next.js frontend to communicate with the Flask backend seamlessly.

---

## 📚 Documentation

- **Backend API**: See `backend/README.md`
- **Database Schema**: See `DATABASE_SCHEMA.md`
- **Deployment**: See `DEPLOYMENT_GUIDE.md`

---

## 🆘 Troubleshooting

### Issue: Cannot connect to backend
**Solution**: Make sure Flask backend is running on port 5000
```bash
cd backend
python app.py
```

### Issue: CORS errors
**Solution**: Verify flask-cors is installed
```bash
pip install flask-cors
```

### Issue: Port already in use
**Solution**: Kill the process using the port
```bash
# Windows
netstat -ano | findstr :3000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:3000 | xargs kill -9
```

---

## 🎉 Quick Start Checklist

- [ ] Navigate to project root
- [ ] Install backend dependencies: `cd backend && pip install -r requirements.txt`
- [ ] Install frontend dependencies: `npm install`
- [ ] Set up `.env` file in backend with OpenAI API key
- [ ] Start backend: `cd backend && python app.py`
- [ ] Start frontend: `npm run dev`
- [ ] Open browser: http://localhost:3000
- [ ] Upload a financial file and see the magic! ✨

---

## 💡 Pro Tips

1. **Use two terminals** - One for backend, one for frontend
2. **Check logs** - Backend logs show AI processing in real-time
3. **API first** - Test API endpoints before building frontend features
4. **Hot reload** - Both servers support hot reload for development

---

**Happy Coding! 🚀**

