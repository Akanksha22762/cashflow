"""
Alternative server runner that won't stop unexpectedly
Use this instead of app.py if server keeps stopping
"""
import os
from app import app

if __name__ == '__main__':
    host = os.getenv('HOST', '127.0.0.1')
    port = int(os.getenv('PORT', 5000))
    
    print(f"\n🚀 Starting Cash Flow Analysis API on http://{host}:{port}")
    print("📝 Press CTRL+C to stop\n")
    
    try:
        # Use threaded=True to handle multiple requests
        # Use use_reloader=False to prevent auto-restart issues
        app.run(
            host=host, 
            port=port, 
            debug=True, 
            use_reloader=False,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n\n❌ Server crashed: {e}")
        import traceback
        traceback.print_exc()
        print("\n🔍 Check the error above for details")
        input("\nPress Enter to exit...")

