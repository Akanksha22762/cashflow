# 🚀 Quick Deployment Reference

## Your Current Setup

### ✅ Local Development (Working Now)
\`\`\`
File: .env
OPENAI_API_KEY=sk-proj-Fz25BhnBMrXcwhjzuNOFfQ36UvS8N7D...
ENVIRONMENT=LOCAL
\`\`\`
- Runs on: `http://127.0.0.1:5000`
- Status: ✅ **Working**

### 🌍 Production Deployment (EC2)
\`\`\`
File: .env (on EC2 server)
OPENAI_API_KEY=sk-proj-Fz25BhnBMrXcwhjzuNOFfQ36UvS8N7D...
ENVIRONMENT=EC2
\`\`\`
- Runs on: `http://13.204.84.17:5000`
- Status: ⏳ **Needs deployment**

---

## 🎯 Quick Deployment - 3 Simple Steps

### Step 1: Run Deployment Script (Windows)

\`\`\`powershell
.\deploy_to_ec2.ps1
\`\`\`

This will create `.env.production` file with your API key configured for EC2.

### Step 2: Upload to EC2

Choose one method:

**Method A - WinSCP (Easiest):**
1. Download WinSCP: https://winscp.net/
2. Connect to: `13.204.84.17`
3. Upload `.env.production` → Rename to `.env` on server

**Method B - SSH + Manual:**
\`\`\`bash
ssh -i your-key.pem ubuntu@13.204.84.17
nano /path/to/app/.env
# Paste content from .env.production
# Save: Ctrl+O, Enter, Ctrl+X
\`\`\`

### Step 3: Start App on EC2

\`\`\`bash
# On EC2 server
cd /path/to/cashflow-app
python3 test_openai_integration.py  # Verify first
python3 app.py                       # Start app
\`\`\`

---

## 📋 Files Created for You

| File | Purpose | Use |
|------|---------|-----|
| `.env` | Local development | Your current setup ✅ |
| `deploy_to_ec2.ps1` | Deployment script | Run to prepare deployment |
| `.env.production` | Production config | Upload to EC2 |
| `DEPLOYMENT_GUIDE.md` | Full guide | Detailed instructions |

---

## ✅ Verification Commands

### On Your Local Machine:
\`\`\`powershell
# Test local setup
python test_openai_integration.py
\`\`\`

### On EC2 Server:
\`\`\`bash
# Verify .env file exists
cat .env

# Test OpenAI integration
python3 test_openai_integration.py

# Check if app is running
ps aux | grep app.py

# View logs
tail -f cashflow_app.log
\`\`\`

### From Browser:
\`\`\`
Local:      http://localhost:5000
Production: http://13.204.84.17:5000
\`\`\`

---

## 🔐 Security Checklist

- [x] API key in `.env` file (not in code)
- [x] `.env` in `.gitignore` (won't commit)
- [ ] Set OpenAI spending limit
- [ ] Enable billing alerts
- [ ] Secure EC2 security groups
- [ ] Use HTTPS (recommended)

---

## 💰 Cost Monitoring

**Set limits NOW to avoid surprises:**

1. **OpenAI Dashboard**: https://platform.openai.com/account/billing/limits
2. **Set monthly limit**: $50-100 (recommended for production)
3. **Enable alerts**: At 80% usage
4. **Monitor daily**: https://platform.openai.com/usage

**Expected costs:**
- Small usage (1K-10K transactions/month): $0.50-5.00
- Medium usage (10K-100K transactions/month): $5.00-50.00
- High usage (100K+ transactions/month): $50.00+

---

## 🆘 Troubleshooting

### Issue: "OpenAI integration not properly initialized"
**Fix:** Verify `.env` file exists on EC2 with correct API key
\`\`\`bash
cat .env  # Should show OPENAI_API_KEY=sk-proj-...
\`\`\`

### Issue: Can't connect to EC2
**Fix:** Check EC2 security group allows inbound on port 5000

### Issue: App starts but API fails
**Fix:** Verify environment variable is loaded
\`\`\`bash
python3 -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('OPENAI_API_KEY')[:20])"
\`\`\`

---

## 📱 Quick Commands Reference

### Local Development:
\`\`\`powershell
# Start app
python app.py

# Test OpenAI
python test_openai_integration.py

# View logs
Get-Content cashflow_app.log -Tail 20 -Wait
\`\`\`

### Production (EC2):
\`\`\`bash
# SSH to EC2
ssh -i your-key.pem ubuntu@13.204.84.17

# Start app
python3 app.py

# Start with auto-restart (recommended)
nohup python3 app.py > output.log 2>&1 &

# Stop app
pkill -f app.py

# View logs
tail -f cashflow_app.log
\`\`\`

---

## 🎯 Summary

| Environment | Status | URL | API Key Source |
|-------------|--------|-----|----------------|
| **Local** | ✅ Working | localhost:5000 | `.env` (local) |
| **Production** | ⏳ Pending | 13.204.84.17:5000 | `.env` (on EC2) |

**Both environments use the SAME API key - it's already configured!**

Just need to:
1. ✅ Run `deploy_to_ec2.ps1`
2. ⏳ Upload `.env` to EC2
3. ⏳ Start app on EC2

---

**Created**: October 11, 2025  
**Your API Key**: Configured and Ready ✅  
**Status**: Ready for Deployment 🚀

---

## 📚 Need More Help?

- **Full Guide**: See `DEPLOYMENT_GUIDE.md`
- **OpenAI Docs**: `README_OPENAI.md`
- **Migration Info**: `OPENAI_MIGRATION_GUIDE.md`
- **Troubleshooting**: `OPENAI_FIX_COMPLETE.md`
