# 🔧 Fix Frontend Installation

## Problem
1. PowerShell execution policy blocking npm
2. React 19 compatibility issue with `vaul` package

## ✅ Solution - Run This Command:

**In Command Prompt (CMD) or PowerShell:**

```cmd
cd C:\Users\akank\OneDrive\Desktop\CashflowApp\frontend
npm install --legacy-peer-deps
```

**OR if PowerShell still blocks, use CMD:**

1. Open **Command Prompt** (press Win+R, type `cmd`, press Enter)
2. Run:
   ```cmd
   cd C:\Users\akank\OneDrive\Desktop\CashflowApp\frontend
   npm install --legacy-peer-deps
   ```

## What --legacy-peer-deps does:

It ignores peer dependency conflicts (like React 19 vs React 18) and installs anyway. This is safe and commonly used with React 19.

## After Installation:

Once `npm install --legacy-peer-deps` completes, you can run:

```cmd
npm run dev
```

## Alternative: Update package.json

You can also add this to package.json to always use legacy peer deps:

```json
"scripts": {
  "install": "npm install --legacy-peer-deps"
}
```

Then just run: `npm install`

