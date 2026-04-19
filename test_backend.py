
import os

if __name__ == "__main__":
    # Just check if app.py can be initialized without errors
    try:
        import app
        print("Backend imports and initialization OK.")
    except Exception as e:
        print(f"Backend check failed: {e}")
        import traceback
        traceback.print_exc()
