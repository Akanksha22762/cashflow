# How to Run the Cash Flow Application

## Prerequisites

1. **Python 3.8+** installed
2. **Node.js 18+** and **npm** installed
3. **OpenAI API Key** (for AI features)

## Step 1: Install Backend Dependencies

Open a terminal in the project root and run:

```bash
# Navigate to backend directory
cd backend

# Install Python packages
pip install -r requirements.txt

# Optional: Install PDF support (if you want to process PDF files)
pip install PyPDF2
# OR
pip install pdfplumber
```

## Step 2: Install Frontend Dependencies

Open a **new terminal** (keep backend terminal open) and run:

```bash
# Navigate to frontend directory
cd frontend

# Install Node.js packages
npm install
```

## Step 3: Configure Environment Variables

Create a `.env` file in the `backend` directory:

```bash
# In backend directory
cd backend

# Create .env file (Windows PowerShell)
New-Item -ItemType File -Path .env

# Or create manually with these contents:
```

Add these variables to `backend/.env`:

```env
# OpenAI API Key (REQUIRED for AI features)
OPENAI_API_KEY=your-openai-api-key-here

# Optional: Server Configuration
HOST=127.0.0.1
PORT=5000
SECRET_KEY=your-secret-key-here

# Optional: Database Configuration (if using MySQL)
# DB_HOST=localhost
# DB_USER=your_user
# DB_PASSWORD=your_password
# DB_NAME=your_database
```

## Step 4: Run the Backend Server

In the **backend terminal**:

```bash
# Make sure you're in backend directory
cd backend

# Run the Flask server
python app.py

# OR use the run_server script
python run_server.py
```

You should see:
```
✅ Cash Flow Analysis API - All routes registered successfully!
🚀 Starting Cash Flow Analysis API on http://127.0.0.1:5000
```

The backend will run on **http://127.0.0.1:5000**

## Step 5: Run the Frontend Server

In the **frontend terminal** (new terminal window):

```bash
# Make sure you're in frontend directory
cd frontend

# Run the Next.js development server
npm run dev
```

You should see:
```
  ▲ Next.js 15.2.4
  - Local:        http://localhost:3000
```

The frontend will run on **http://localhost:3000**

## Step 6: Access the Application

Open your browser and go to:
- **Frontend**: http://localhost:3000
- **Backend API**: http://127.0.0.1:5000

## Quick Commands Summary

### Backend (Terminal 1)
```bash
cd backend
pip install -r requirements.txt
python app.py
```

### Frontend (Terminal 2)
```bash
cd frontend
npm install
npm run dev
```

## Testing the Upload Feature

1. Go to http://localhost:3000
2. Upload a bank statement file (PDF, Excel, CSV, or text)
3. The system will:
   - Extract all transactions
   - Classify them using AI
   - Generate a comprehensive cash flow report

## Troubleshooting

### Backend Issues

**Port already in use:**
```bash
# Change port in .env file
PORT=5001
```

**Missing dependencies:**
```bash
pip install -r requirements.txt --upgrade
```

**OpenAI API errors:**
- Check your `.env` file has `OPENAI_API_KEY` set
- Verify the API key is valid

### Frontend Issues

**Port already in use:**
```bash
# Next.js will automatically use next available port (3001, 3002, etc.)
```

**Module not found:**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

**Build errors:**
```bash
cd frontend
npm run build
```

## Production Deployment

### Backend (using Gunicorn)
```bash
cd backend
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Frontend (build and serve)
```bash
cd frontend
npm run build
npm start
```

## File Upload Formats Supported

- ✅ **Excel**: `.xlsx`, `.xls`
- ✅ **CSV**: `.csv`
- ✅ **PDF**: `.pdf` (requires PyPDF2 or pdfplumber)
- ✅ **Text**: `.txt`

## API Endpoints

Once backend is running, you can test:

- `GET http://127.0.0.1:5000/` - Health check
- `GET http://127.0.0.1:5000/status` - System status
- `POST http://127.0.0.1:5000/upload` - Upload bank statement
- `GET http://127.0.0.1:5000/get-current-data` - Get current data
- `GET http://127.0.0.1:5000/comprehensive-report` - Get cash flow report

## Need Help?

Check these files:
- `backend/upload_modules/BANK_STATEMENT_PROCESSING.md` - Processing details
- `backend/TROUBLESHOOTING.md` - Common issues
- `backend/ACTIVE_ROUTES.md` - All available routes

