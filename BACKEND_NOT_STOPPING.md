# 🔧 Fix Backend Stopping Unexpectedly

## Problem
Backend stops running even when you don't stop it manually.

## ✅ Solutions

### Solution 1: Use run_server.py (Recommended)

Instead of `python app.py`, use:

```powershell
cd C:\Users\akank\OneDrive\Desktop\CashflowApp\backend
python run_server.py
```

This version:
- ✅ Won't auto-reload (prevents unexpected stops)
- ✅ Better error handling
- ✅ Stays running until you press CTRL+C

### Solution 2: Check for Errors

Look at your backend terminal for:
- ❌ Red error messages
- ❌ Import errors
- ❌ Exception tracebacks

If you see errors, share them and I'll help fix them.

### Solution 3: Keep Terminal Open

Make sure:
- ✅ The terminal window stays open
- ✅ Don't close the terminal
- ✅ Don't click "X" on the terminal window

## 🚀 Quick Start (Use This)

**Terminal 1 - Backend:**
```powershell
cd C:\Users\akank\OneDrive\Desktop\CashflowApp\backend
python run_server.py
```

**Terminal 2 - Frontend:**
```powershell
cd C:\Users\akank\OneDrive\Desktop\CashflowApp\frontend
npm run dev
```

## 📝 Why Backend Might Stop

1. **Auto-reload in debug mode** - Flask's debug mode can cause issues
2. **Uncaught exceptions** - Errors crash the server
3. **Terminal closed** - Accidentally closing the window
4. **System exit calls** - Code calling `sys.exit()` somewhere

## ✅ What I Fixed

1. ✅ Disabled auto-reloader (`use_reloader=False`)
2. ✅ Added better error handling
3. ✅ Created `run_server.py` as alternative
4. ✅ Prevented SystemExit calls from stopping the app

## 🔍 Debug Steps

If backend still stops:

1. **Check backend terminal** - Look for error messages
2. **Check logs** - See what happened before it stopped
3. **Use run_server.py** - More stable than app.py
4. **Check .env file** - Make sure API keys are valid

The backend should now stay running! 🚀

