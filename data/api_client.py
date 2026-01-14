# -*- coding: utf-8 -*-
"""
API Client for Streamlit Dashboard
Updated endpoints from DE team (Aman)
Base URL: http://127.0.0.1:8000
"""

import requests
from typing import Optional, Dict, List
from datetime import datetime
import pandas as pd

class DashboardAPIClient:
    """Client for interacting with the updated API endpoints"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url
        
    def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make HTTP request with error handling"""
        url = f"{self.base_url}{endpoint}"
        try:
            if method == "GET":
                response = requests.get(url, params=params, timeout=10)
            elif method == "POST":
                response = requests.post(url, json=params, timeout=10)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")
    
    # --- System Health ---
    def check_health(self) -> Dict:
        """GET /health - Check system health"""
        return self._make_request("GET", "/health")
    
    # --- Pipeline Control ---
    def run_pipeline(self) -> Dict:
        """POST /run-pipeline - Trigger data pipeline"""
        return self._make_request("POST", "/run-pipeline")
    
    # --- Stock Data & Charts ---
    def get_recent_data(self, ticker: str, days: int = 30) -> Dict:
        """
        GET /supabase/recent/{ticker}?days=30
        Get recent stock data for a ticker
        """
        endpoint = f"/supabase/recent/{ticker}"
        params = {"days": days}
        return self._make_request("GET", endpoint, params)
    
    def get_ticker_data(self, ticker: str, start_date: str = "2024-01-01", limit: int = 100) -> Dict:
        """
        GET /supabase/ticker/{ticker}?start_date=2024-01-01&limit=100
        Get ticker data with date range and limit
        """
        endpoint = f"/supabase/ticker/{ticker}"
        params = {"start_date": start_date, "limit": limit}
        return self._make_request("GET", endpoint, params)
    
    # --- Market Overview ---
    def get_latest_market(self, limit: int = 10) -> Dict:
        """
        GET /supabase/latest?limit=10
        Get latest market data for all tickers
        """
        params = {"limit": limit}
        return self._make_request("GET", "/supabase/latest", params)
    
    def get_top_performers(self, top_n: int = 10) -> Dict:
        """
        GET /supabase/top-performers?top_n=10
        Get top performing stocks
        """
        params = {"top_n": top_n}
        return self._make_request("GET", "/supabase/top-performers", params)
    
    # --- Analysis & Filtering ---
    def get_ticker_stats(self, ticker: str, start_date: str = "2024-01-01") -> Dict:
        """
        GET /supabase/stats/{ticker}?start_date=2024-01-01 (Auto-ends today)
        Get statistical analysis for a ticker
        """
        endpoint = f"/supabase/stats/{ticker}"
        params = {"start_date": start_date}
        return self._make_request("GET", endpoint, params)
    
    def search_by_rsi(self, min_rsi: float = 0, max_rsi: float = 30) -> Dict:
        """
        GET /supabase/rsi-search?min_rsi=0&max_rsi=30
        Search stocks by RSI range
        """
        params = {"min_rsi": min_rsi, "max_rsi": max_rsi}
        return self._make_request("GET", "/supabase/rsi-search", params)
    
    # --- Helper Methods ---
    def get_ticker_dataframe(self, ticker: str, start_date: str = "2024-01-01", limit: int = 100) -> pd.DataFrame:
        """Get ticker data as pandas DataFrame"""
        data = self.get_ticker_data(ticker, start_date, limit)
        if "data" in data and data["data"]:
            return pd.DataFrame(data["data"])
        return pd.DataFrame()
    
    def get_recent_dataframe(self, ticker: str, days: int = 30) -> pd.DataFrame:
        """Get recent data as pandas DataFrame"""
        data = self.get_recent_data(ticker, days)
        if "data" in data and data["data"]:
            return pd.DataFrame(data["data"])
        return pd.DataFrame()


# Example usage
if __name__ == "__main__":
    client = DashboardAPIClient()
    
    # Test health check
    print("Health Check:")
    print(client.check_health())
    
    # Test ticker data
    print("\nTicker Data (AAPL):")
    print(client.get_ticker_data("AAPL", start_date="2024-01-01", limit=10))
    
    # Test top performers
    print("\nTop Performers:")
    print(client.get_top_performers(top_n=5))
    
    # Test RSI search
    print("\nRSI Search (Oversold stocks):")
    print(client.search_by_rsi(min_rsi=0, max_rsi=30))
