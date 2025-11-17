# 🚀 Deployment Guide - OpenAI API Key Setup

## Overview

This guide shows you how to deploy your Cash Flow Analysis app with OpenAI API key configured properly for production.

---

## 📋 Deployment Options

### Option 1: AWS EC2 (Recommended for Your App)
### Option 2: Other Cloud Providers (Azure, GCP, DigitalOcean)
### Option 3: Docker Deployment

---

## 🔐 **OPTION 1: AWS EC2 Deployment** (Your Current Setup)

Your app is configured to run on EC2 at: `http://13.204.84.17:5000`

### Step 1: Connect to Your EC2 Instance

\`\`\`bash
# Via SSH (from your local machine)
ssh -i your-key.pem ubuntu@13.204.84.17
\`\`\`

### Step 2: Set Environment Variables on EC2

Once connected to EC2, you have 3 options:

#### **Method A: Create `.env` file on EC2** (Easiest)

\`\`\`bash
# On EC2 server, navigate to your app directory
cd /path/to/your/app

# Create .env file
nano .env

# Add these lines (paste your actual API key):
OPENAI_API_KEY=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
ENVIRONMENT=EC2

# Save: Ctrl+O, Enter, Ctrl+X
\`\`\`

#### **Method B: System Environment Variables** (More Secure)

\`\`\`bash
# On EC2 server, edit your .bashrc or .profile
nano ~/.bashrc

# Add at the end:
export OPENAI_API_KEY="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"
export ENVIRONMENT="EC2"

# Save and reload
source ~/.bashrc
\`\`\`

#### **Method C: Systemd Service** (Best for Production)

\`\`\`bash
# Create systemd service file
sudo nano /etc/systemd/system/cashflow-app.service

# Add this content:
[Unit]
Description=Cash Flow Analysis App
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/path/to/your/app
Environment="OPENAI_API_KEY=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"
Environment="ENVIRONMENT=EC2"
ExecStart=/usr/bin/python3 /path/to/your/app/app.py
Restart=always

[Install]
WantedBy=multi-user.target

# Save and enable service
sudo systemctl daemon-reload
sudo systemctl enable cashflow-app
sudo systemctl start cashflow-app
sudo systemctl status cashflow-app
\`\`\`

### Step 3: Install Dependencies on EC2

\`\`\`bash
# On EC2 server
cd /path/to/your/app

# Install Python dependencies
pip3 install -r requirements.txt

# Or with virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
\`\`\`

### Step 4: Start Your App on EC2

\`\`\`bash
# Simple start
python3 app.py

# Or with production server (gunicorn)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
\`\`\`

### Step 5: Access Your Deployed App

\`\`\`
http://13.204.84.17:5000
\`\`\`

---

## 🔐 **OPTION 2: Environment Variables via AWS Systems Manager**

For better security, store secrets in AWS Systems Manager Parameter Store:

### Step 1: Store API Key in AWS Parameter Store

\`\`\`bash
# From your local machine with AWS CLI configured
aws ssm put-parameter \
    --name "/cashflow-app/openai-api-key" \
    --value "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" \
    --type "SecureString" \
    --region ap-south-1
\`\`\`

### Step 2: Retrieve in Your App

Add this to your `app.py` or create a new file `config.py`:

\`\`\`python
import boto3
import os

def get_openai_key():
    """Get OpenAI API key from AWS Parameter Store or environment"""
    # Try environment variable first
    key = os.getenv('OPENAI_API_KEY')
    if key:
        return key
    
    # If not found, try AWS Parameter Store
    try:
        ssm = boto3.client('ssm', region_name='ap-south-1')
        response = ssm.get_parameter(
            Name='/cashflow-app/openai-api-key',
            WithDecryption=True
        )
        return response['Parameter']['Value']
    except Exception as e:
        print(f"Failed to get API key from AWS: {e}")
        return None
\`\`\`

---

## 🐳 **OPTION 3: Docker Deployment**

### Step 1: Create Dockerfile

\`\`\`dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 5000

# Run application
CMD ["python", "app.py"]
\`\`\`

### Step 2: Create docker-compose.yml

\`\`\`yaml
version: '3.8'

services:
  cashflow-app:
    build: .
    ports:
      - "5000:5000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ENVIRONMENT=EC2
    restart: always
\`\`\`

### Step 3: Deploy with Docker

\`\`\`bash
# On EC2 or any server
export OPENAI_API_KEY="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"
docker-compose up -d
\`\`\`

---

## ✅ **Verification Checklist**

After deployment, verify everything works:

### 1. Check API Key is Loaded

\`\`\`bash
# On your deployment server
python3 -c "import os; from dotenv import load_dotenv; load_dotenv(); print('API Key:', os.getenv('OPENAI_API_KEY')[:20] if os.getenv('OPENAI_API_KEY') else 'NOT FOUND')"
\`\`\`

### 2. Test OpenAI Integration

\`\`\`bash
# On your deployment server
python3 test_openai_integration.py
\`\`\`

Expected output:
\`\`\`
[SUCCESS] OpenAI integration is working!
\`\`\`

### 3. Check App Logs

\`\`\`bash
# Check application logs
tail -f cashflow_app.log

# Look for:
✅ OpenAI Integration loaded!
✅ Global OpenAI integration initialized: True
\`\`\`

### 4. Test API Endpoints

\`\`\`bash
# From your local machine or browser
curl http://13.204.84.17:5000/health

# Should return:
{
  "status": "healthy",
  "openai_available": true
}
\`\`\`

---

## 🔒 **Security Best Practices**

### DO ✅
- ✅ Use environment variables for API keys
- ✅ Use AWS Parameter Store or Secrets Manager for production
- ✅ Set up IAM roles with minimum required permissions
- ✅ Enable HTTPS/SSL for production
- ✅ Set up billing alerts on OpenAI dashboard
- ✅ Monitor API usage regularly
- ✅ Use security groups to restrict access
- ✅ Rotate API keys every 3-6 months

### DON'T ❌
- ❌ Never commit `.env` files to git
- ❌ Never hardcode API keys in source code
- ❌ Never share API keys in chat/email
- ❌ Never expose keys in logs or error messages
- ❌ Never use same key for dev and prod (if possible)

---

## 📊 **Cost Management for Production**

### Set OpenAI Spending Limits

1. Go to: https://platform.openai.com/account/billing/limits
2. Set monthly spending limit (e.g., $50-100)
3. Enable email alerts at 80% usage
4. Monitor daily usage

### Expected Production Costs

| Usage | Estimated Cost/Month |
|-------|---------------------|
| 10K transactions/month | $0.50-1.00 |
| 100K transactions/month | $5.00-10.00 |
| 1M transactions/month | $50.00-100.00 |

---

## 🚨 **Troubleshooting Production Issues**

### Issue: "OpenAI integration not properly initialized"

**Solution:**
\`\`\`bash
# Verify environment variable is set
echo $OPENAI_API_KEY

# Check .env file exists and has correct format
cat .env

# Restart application
sudo systemctl restart cashflow-app  # if using systemd
# OR
pkill -f app.py && python3 app.py &  # manual restart
\`\`\`

### Issue: "Connection refused" or timeout

**Solution:**
\`\`\`bash
# Check if app is running
ps aux | grep app.py

# Check port is listening
netstat -tulpn | grep 5000

# Check firewall/security groups
sudo ufw status
# AWS: Check EC2 Security Group allows inbound on port 5000
\`\`\`

### Issue: High API costs

**Solution:**
- Enable caching in app (already implemented)
- Reduce batch sizes
- Set stricter rate limits
- Review unusual usage patterns
- Check for API key leaks

---

## 📝 **Quick Deployment Commands**

### For EC2 Deployment:

\`\`\`bash
# 1. SSH to EC2
ssh -i your-key.pem ubuntu@13.204.84.17

# 2. Navigate to app
cd /path/to/cashflow-app

# 3. Create/update .env
echo "OPENAI_API_KEY=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" > .env
echo "ENVIRONMENT=EC2" >> .env

# 4. Install dependencies
pip3 install -r requirements.txt

# 5. Test
python3 test_openai_integration.py

# 6. Start app
python3 app.py
# OR with gunicorn:
gunicorn -w 4 -b 0.0.0.0:5000 app:app
\`\`\`

---

## 📚 **Additional Resources**

- **OpenAI Dashboard**: https://platform.openai.com/
- **AWS EC2 Documentation**: https://docs.aws.amazon.com/ec2/
- **Python dotenv**: https://pypi.org/project/python-dotenv/
- **Gunicorn**: https://gunicorn.org/

---

## 🎯 **Summary**

1. ✅ Choose deployment method (EC2, Docker, etc.)
2. ✅ Set `OPENAI_API_KEY` environment variable
3. ✅ Set `ENVIRONMENT=EC2` for production
4. ✅ Install dependencies
5. ✅ Test OpenAI integration
6. ✅ Start application
7. ✅ Verify endpoints work
8. ✅ Set up monitoring and alerts

**Your API key is ready for both local and production use!** 🚀

---

**Last Updated**: October 11, 2025  
**Deployment Target**: AWS EC2 (13.204.84.17)  
**Environment**: Production Ready ✅
