#!/bin/bash
# Start Flask Backend Server
# Linux/Mac Bash Script

echo "🚀 Starting Flask Backend Server..."
echo ""

# Navigate to backend directory
cd "$(dirname "$0")"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python is not installed or not in PATH"
    exit 1
fi

# Install/upgrade dependencies
echo "📦 Checking dependencies..."
pip3 install -r requirements.txt --quiet

# Check for .env file
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found"
    echo "Creating sample .env file..."
    cat > .env << EOF
OPENAI_API_KEY=your_openai_api_key_here
FLASK_ENV=development
FLASK_APP=app.py
EOF
    echo "✅ Sample .env file created. Please add your OpenAI API key."
fi

echo ""
echo "✅ Starting Flask Backend on http://localhost:5000"
echo "📡 API Documentation: http://localhost:5000/"
echo "🌐 Frontend: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the Flask app
python3 app.py

