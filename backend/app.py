"""
Main Flask Application
Clean entry point that registers all routes and initializes the application
"""
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, static_folder='static', static_url_path='/static')
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')

# Enable CORS for Next.js frontend - Allow all origins in development
CORS(app, resources={
    r"/*": {
        "origins": "*",  # Allow all origins in development
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
        "supports_credentials": True
    }
})

# Set max file size
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Import app_setup to initialize all dependencies
# This must be imported before routes to ensure globals are initialized
import app_setup

# Register all route blueprints
from routes import system, upload, data, vendors, reports, analytics, ai_reasoning, transactions

app.register_blueprint(system.bp)
app.register_blueprint(upload.bp)
app.register_blueprint(data.bp)
app.register_blueprint(vendors.bp)
app.register_blueprint(reports.bp)
app.register_blueprint(analytics.bp)
app.register_blueprint(ai_reasoning.bp)
app.register_blueprint(transactions.bp)

print("="*80)
print("✅ Cash Flow Analysis API - All routes registered successfully!")
print("="*80)
print(f"📡 Registered {len(app.url_map._rules)} routes")
print("🔗 Available blueprints:")
print("   - system: /, /status")
print("   - upload: /upload")
print("   - data: /get-current-data, /get-dropdown-data")
print("   - vendors: /vendor-analysis, /view_vendor_transactions/<vendor_name>, /extract-vendors-for-analysis")
print("   - reports: /comprehensive-report, /comprehensive-report.pdf")
print("   - analytics: /get-available-trend-types, /validate-trend-selection, /run-dynamic-trends-analysis")
print("   - ai_reasoning: /ai-reasoning/categorization, /ai-reasoning/vendor-landscape, /ai-reasoning/trend-analysis")
print("   - transactions: /transaction-analysis")
print("="*80)

if __name__ == '__main__':
    # Get server URL from environment or use default
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    
    print(f"\n🚀 Starting Cash Flow Analysis API on http://{host}:{port}")
    print("📝 Press CTRL+C to stop\n")
    
    try:
        app.run(host=host, port=port, debug=True, use_reloader=False, threaded=True)
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n\n❌ Server error: {e}")
        import traceback
        traceback.print_exc()
        # Don't exit - let user see the error
        input("\nPress Enter to exit...")

