
import requests
import sys
import time
from datetime import datetime

OK = "[OK]"
FAIL = "[FAIL]"
WARN = "[WARN]"

def check_url(name, url, expected_status=200):
    try:
        # Use a short timeout to fail fast
        response = requests.get(url, timeout=3)
        if response.status_code == expected_status:
            print(f"{OK} {name}: Online ({url}, Status {response.status_code})")
            return int(True)
        else:
            print(f"{FAIL} {name}: Error ({url}, Status {response.status_code})")
            return int(False)
    except Exception as e:
        print(f"{FAIL} {name}: Unreachable ({url}) - {str(e)}")
        return int(False)

def check_signal_flow(symbol="AAPL"):
    url = "http://localhost:8000/api/v1/ml/signal/live"
    payload = {"ticker": symbol}
    try:
        print(f"   Testing Signal Generation for {symbol}...")
        response = requests.post(url, json=payload, timeout=20)
        if response.status_code == 200:
            data = response.json()
            signal = data.get("signal", "Unknown")
            conf = data.get("confidence", 0)
            print(f"{OK} Signal API Logic: Working. {symbol} -> {signal} ({conf:.1f}%)")
            return True
        else:
            print(f"{FAIL} Signal API Logic: Failed. Status {response.status_code}. Response: {response.text[:100]}")
            return False
    except Exception as e:
        print(f"{FAIL} Signal API Logic: Unreachable or Timeout. {e}")
        return False

def check_backtest_flow(symbol="AAPL"):
    url = "http://localhost:8002/api/v1/backtest/run"
    payload = {"ticker": symbol}
    try:
        print(f"   Testing Backtest for {symbol}...")
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            conf = data.get("confidence_score", 0)
            print(f"{OK} Backtest API Logic: Working. Confidence: {conf}")
            return True
        else:
            print(f"{FAIL} Backtest API Logic: Failed. Status {response.status_code}. Response: {response.text[:100]}")
            return False
    except Exception as e:
        print(f"{FAIL} Backtest API Logic: Unreachable or Timeout. {e}")
        return False

def main():
    print(f"\nSystem Health Check - {datetime.now().strftime('%H:%M:%S')}")
    print("-" * 50)
    
    # 1. Check Ports
    s1 = check_url("Signals API", "http://localhost:8000/health")
    s2 = check_url("Alerts API", "http://localhost:8001/health")
    s3 = check_url("Backtest API", "http://localhost:8002/health")
    s4 = check_url("Streamlit Dashboard", "http://localhost:8501/_stcore/health")
    
    # 2. Check Logic
    logic_ok = True
    
    if s1:
        if not check_signal_flow(): logic_ok = False
    else:
        print(f"{WARN} Skipping Signal Logic (API down)")
        logic_ok = False

    if s3:
         if not check_backtest_flow(): logic_ok = False
    else:
         print(f"{WARN} Skipping Backtest Logic (API down)")
         logic_ok = False

    print("-" * 50)
    total = s1 + s2 + s3 + s4
    print(f"Summary: {total}/4 Services Online.")
    
    if total == 4 and logic_ok:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
