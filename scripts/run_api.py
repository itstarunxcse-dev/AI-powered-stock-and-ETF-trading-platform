"""
Quick start script for the FastAPI Backtesting Service
Run from project root: python scripts/run_api.py
"""
import uvicorn
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    print("🚀 Starting Backtesting API Server...")
    print("📍 API will be available at: http://localhost:8000")
    print("📚 Interactive docs at: http://localhost:8000/docs")
    print("⏹️  Press CTRL+C to stop\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
