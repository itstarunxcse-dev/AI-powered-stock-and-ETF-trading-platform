import requests
import json
import sys

def test_backtest():
    url = "http://localhost:8002/api/v1/backtest/run"
    payload = {"ticker": "AAPL"}
    
    print(f"Testing {url} with {payload}...")
    try:
        resp = requests.post(url, json=payload, timeout=10)
        print(f"Status Code: {resp.status_code}")
        try:
            print("Response:", json.dumps(resp.json(), indent=2))
        except:
            print("Response Text:", resp.text)
            
    except Exception as e:
        print(f"Connection Failed: {e}")

if __name__ == "__main__":
    test_backtest()
