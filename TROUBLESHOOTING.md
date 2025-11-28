# 🔧 Troubleshooting "Failed to fetch" Error

## Problem
Frontend is running but showing "Failed to fetch" errors on Vendors, Analytics, and Reports pages.

## ✅ Solution Checklist

### 1. **Check if Backend is Running**

Open a new terminal and check if backend is running:

```powershell
# Check if port 5000 is in use
netstat -ano | findstr :5000
```

**If backend is NOT running, start it:**

```powershell
cd C:\Users\akank\OneDrive\Desktop\CashflowApp\backend
python app.py
```

You should see:
```
✅ Cash Flow Analysis API - All routes registered successfully!
🚀 Starting Cash Flow Analysis API on http://127.0.0.1:5000
```

### 2. **Verify Backend is Accessible**

Open in browser: `http://127.0.0.1:5000/status`

You should see a JSON response like:
```json
{"status": "ok", "message": "API is running"}
```

### 3. **Check API Configuration**

The frontend needs to know where the backend is. Check if there's an API configuration file.

### 4. **Common Issues**

#### Issue: Backend not started
- **Fix:** Start backend in a separate terminal

#### Issue: Wrong port
- **Fix:** Make sure backend is on port 5000 and frontend is on port 3000

#### Issue: CORS error
- **Fix:** Backend should already have CORS configured, but verify in `backend/app.py`

#### Issue: API endpoint not configured
- **Fix:** Check frontend code for API base URL

## 🚀 Quick Fix Steps

1. **Start Backend:**
   ```powershell
   cd C:\Users\akank\OneDrive\Desktop\CashflowApp\backend
   python app.py
   ```

2. **Keep Frontend Running:**
   ```powershell
   cd C:\Users\akank\OneDrive\Desktop\CashflowApp\frontend
   npm run dev
   ```

3. **Test Connection:**
   - Open: `http://127.0.0.1:5000/status` (should work)
   - Open: `http://127.0.0.1:3000` (frontend should connect)

## 📝 Expected Behavior

- **Backend:** Running on `http://127.0.0.1:5000`
- **Frontend:** Running on `http://127.0.0.1:3000`
- **Connection:** Frontend calls backend API at `http://127.0.0.1:5000`

