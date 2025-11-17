#!/bin/bash
# Quick Deployment Script for EC2
# This script helps you deploy your Cash Flow app to EC2 with OpenAI API key

echo "================================================"
echo "  Cash Flow App - EC2 Deployment Script"
echo "================================================"
echo ""

# Your EC2 IP
EC2_IP="13.204.84.17"
EC2_USER="ubuntu"
APP_DIR="/home/ubuntu/cashflow-app"  # Change this to your app directory on EC2

# Your OpenAI API Key
OPENAI_KEY="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"

echo "🔧 Step 1: Creating .env file for production..."

# Create .env content
ENV_CONTENT="OPENAI_API_KEY=${OPENAI_KEY}
ENVIRONMENT=EC2"

# Save to temporary file
echo "$ENV_CONTENT" > .env.production

echo "✅ .env.production file created locally"
echo ""

echo "📤 Step 2: Copying .env to EC2 server..."
echo "   (You'll need your SSH key and may be prompted for password)"
echo ""

# Copy .env file to EC2 (you'll need to adjust the path to your SSH key)
# Uncomment and modify the line below with your actual SSH key path:
# scp -i /path/to/your-key.pem .env.production ${EC2_USER}@${EC2_IP}:${APP_DIR}/.env

echo "   Run this command manually with your SSH key:"
echo "   scp -i your-key.pem .env.production ${EC2_USER}@${EC2_IP}:${APP_DIR}/.env"
echo ""

echo "🔄 Step 3: SSH to EC2 and restart app..."
echo "   Run this command manually:"
echo "   ssh -i your-key.pem ${EC2_USER}@${EC2_IP}"
echo ""
echo "   Then on EC2 server:"
echo "   cd ${APP_DIR}"
echo "   python3 test_openai_integration.py  # Test first"
echo "   python3 app.py                       # Start app"
echo ""

echo "================================================"
echo "✅ Deployment preparation complete!"
echo "================================================"
echo ""
echo "📝 Next steps:"
echo "   1. Copy .env.production to EC2 (see command above)"
echo "   2. SSH to EC2"
echo "   3. Test OpenAI integration"
echo "   4. Start your app"
echo ""
echo "🌐 Your app will be available at: http://${EC2_IP}:5000"
echo ""
echo "📊 Monitor OpenAI usage: https://platform.openai.com/usage"
echo "💰 Set billing limits: https://platform.openai.com/account/billing/limits"
echo ""

# Clean up
rm -f .env.production
