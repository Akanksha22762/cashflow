# Quick Deployment Script for EC2 (PowerShell)
# This script helps you deploy your Cash Flow app to EC2 with OpenAI API key

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Cash Flow App - EC2 Deployment Script" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Your EC2 Configuration
$EC2_IP = "13.204.84.17"
$EC2_USER = "ubuntu"
$APP_DIR = "/home/ubuntu/cashflow-app"  # Change this to your app directory on EC2

# Your OpenAI API Key
$OPENAI_KEY = "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"

Write-Host "🔧 Step 1: Creating .env file for production..." -ForegroundColor Green

# Create .env content for production
$envContent = @"
OPENAI_API_KEY=$OPENAI_KEY
ENVIRONMENT=EC2
"@

# Save to file
$envContent | Out-File -FilePath ".env.production" -Encoding UTF8 -NoNewline

Write-Host "   ✅ .env.production file created" -ForegroundColor Green
Write-Host ""

Write-Host "📋 Step 2: Your production environment file is ready!" -ForegroundColor Green
Write-Host ""

Write-Host "Production .env contents:" -ForegroundColor Yellow
Write-Host "------------------------" -ForegroundColor Gray
Get-Content .env.production
Write-Host "------------------------" -ForegroundColor Gray
Write-Host ""

Write-Host "📤 Step 3: Upload to EC2" -ForegroundColor Green
Write-Host ""
Write-Host "   Option A: Using SCP (if you have it installed)" -ForegroundColor Cyan
Write-Host "   scp -i path\to\your-key.pem .env.production ${EC2_USER}@${EC2_IP}:${APP_DIR}/.env" -ForegroundColor White
Write-Host ""
Write-Host "   Option B: Using WinSCP or FileZilla" -ForegroundColor Cyan
Write-Host "   - Open WinSCP/FileZilla" -ForegroundColor White
Write-Host "   - Connect to: $EC2_IP" -ForegroundColor White
Write-Host "   - Upload .env.production to: $APP_DIR/.env" -ForegroundColor White
Write-Host ""
Write-Host "   Option C: Manual copy-paste" -ForegroundColor Cyan
Write-Host "   - SSH to EC2: ssh -i your-key.pem ${EC2_USER}@${EC2_IP}" -ForegroundColor White
Write-Host "   - Create file: nano ${APP_DIR}/.env" -ForegroundColor White
Write-Host "   - Paste the content shown above" -ForegroundColor White
Write-Host "   - Save: Ctrl+O, Enter, Ctrl+X" -ForegroundColor White
Write-Host ""

Write-Host "🚀 Step 4: Deploy on EC2" -ForegroundColor Green
Write-Host ""
Write-Host "   After uploading .env file, SSH to EC2 and run:" -ForegroundColor Cyan
Write-Host "   ssh -i your-key.pem ${EC2_USER}@${EC2_IP}" -ForegroundColor White
Write-Host ""
Write-Host "   Then on EC2 server:" -ForegroundColor Cyan
Write-Host "   cd $APP_DIR" -ForegroundColor White
Write-Host "   cat .env                              # Verify .env file" -ForegroundColor White
Write-Host "   python3 test_openai_integration.py    # Test OpenAI" -ForegroundColor White
Write-Host "   python3 app.py                        # Start app" -ForegroundColor White
Write-Host ""

Write-Host "================================================" -ForegroundColor Green
Write-Host "✅ Deployment files ready!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host ""

Write-Host "📝 Summary:" -ForegroundColor Cyan
Write-Host "   ✅ Local .env: For development (ENVIRONMENT=LOCAL)" -ForegroundColor White
Write-Host "   ✅ .env.production: For EC2 deployment (ENVIRONMENT=EC2)" -ForegroundColor White
Write-Host ""

Write-Host "🌐 After deployment, access your app at:" -ForegroundColor Cyan
Write-Host "   http://${EC2_IP}:5000" -ForegroundColor Yellow
Write-Host ""

Write-Host "📊 Important Links:" -ForegroundColor Cyan
Write-Host "   Monitor usage: https://platform.openai.com/usage" -ForegroundColor White
Write-Host "   Billing limits: https://platform.openai.com/account/billing/limits" -ForegroundColor White
Write-Host "   API keys: https://platform.openai.com/api-keys" -ForegroundColor White
Write-Host ""

Write-Host "💡 Tips:" -ForegroundColor Yellow
Write-Host "   - Set a monthly spending limit on OpenAI ($50-100 recommended)" -ForegroundColor White
Write-Host "   - Enable email alerts at 80% usage" -ForegroundColor White
Write-Host "   - Monitor logs: tail -f cashflow_app.log (on EC2)" -ForegroundColor White
Write-Host ""

$keepFile = Read-Host "Keep .env.production file? (y/n)"
if ($keepFile -ne "y") {
    Remove-Item .env.production -ErrorAction SilentlyContinue
    Write-Host "   Removed .env.production" -ForegroundColor Gray
} else {
    Write-Host "   .env.production saved for manual upload" -ForegroundColor Green
}

Write-Host ""
Write-Host "🎉 Ready to deploy! Follow the steps above." -ForegroundColor Green
Write-Host ""
