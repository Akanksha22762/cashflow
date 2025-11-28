"""
Keep-alive script to ensure Flask server stays running
Run this instead of app.py if the server keeps stopping
"""
import os
import sys
import signal
from app import app

def signal_handler(sig, frame):
    """Handle CTRL+C gracefully"""
    print('\n\n🛑 Shutting down gracefully...')
    sys.exit(0)

if __name__ == '__main__':
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    host = os.getenv('HOST', '127.0.0.1')
    port = int(os.getenv('PORT', 5000))
    
    print(f"\n🚀 Starting Cash Flow Analysis API on http://{host}:{port}")
    print("📝 Press CTRL+C to stop")
    print("💡 Server configured to stay running...\n")
    
    try:
        # Use threaded=True for better request handling
        # Use use_reloader=False to prevent auto-restart issues
        app.run(
            host=host,
            port=port,
            debug=True,
            use_reloader=False,
            threaded=True,
            use_debugger=True,
            passthrough_errors=False
        )
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n\n❌ Server error: {e}")
        import traceback
        traceback.print_exc()
        print("\n⚠️ Server crashed. Check the error above.")
        input("\nPress Enter to exit...")

