# -*- coding: utf-8 -*-
"""
Background API Starter for Streamlit Dashboard
Automatically starts API server when dashboard launches
"""

import subprocess
import sys
import time
import requests
from pathlib import Path
import atexit
import os

# Global process handle
api_process = None

def is_api_running(port=8000, timeout=1):
    """Check if API is already running"""
    try:
        response = requests.get(f"http://127.0.0.1:{port}/health", timeout=timeout)
        return response.status_code == 200
    except:
        return False

def find_available_port(start_port=8000, max_attempts=10):
    """Find an available port"""
    import socket
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    return None

def start_api_background():
    """Start API server in background"""
    global api_process
    
    # Check if API is already running
    if is_api_running():
        print("✅ API server already running on port 8000")
        return True
    
    print("🚀 Starting API server in background...")
    
    # Get the API script path
    signals_dir = Path(__file__).parent.parent / "signals"
    api_script = signals_dir / "api.py"
    
    if not api_script.exists():
        print(f"❌ API script not found: {api_script}")
        return False
    
    # Find available port
    port = find_available_port()
    if not port:
        print("❌ No available ports found")
        return False
    
    try:
        # Start API server as subprocess
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0
        
        api_process = subprocess.Popen(
            [sys.executable, str(api_script)],
            cwd=str(signals_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=creationflags,
            env={**os.environ, 'PYTHONUNBUFFERED': '1'}
        )
        
        # Wait for API to start (max 30 seconds)
        print(f"⏳ Waiting for API to start on port {port}...")
        for i in range(60):
            time.sleep(0.5)
            if is_api_running(port):
                print(f"✅ API server started successfully on http://127.0.0.1:{port}")
                print(f"📚 API docs available at http://127.0.0.1:{port}/docs")
                return True
        
        print("⚠️ API server started but not responding yet")
        return True
        
    except Exception as e:
        print(f"❌ Failed to start API: {e}")
        return False

def stop_api():
    """Stop API server on exit"""
    global api_process
    if api_process:
        try:
            api_process.terminate()
            api_process.wait(timeout=5)
            print("\n✅ API server stopped")
        except:
            try:
                api_process.kill()
            except:
                pass

# Register cleanup function
atexit.register(stop_api)

def ensure_api_running():
    """
    MANUAL MODE: We assume the user is running the API manually.
    Always return True to skip auto-start logic.
    """
    print("ℹ️ Manual Mode: Skipping auto-start check.")
    return True
