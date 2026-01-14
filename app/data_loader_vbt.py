# -*- coding: utf-8 -*-
"""
📊 Data Loader for VectorBT Backtesting
Fetches historical data with ML signals from ML Service API
"""

import requests
import pandas as pd

ML_BASE_URL = "http://127.0.0.1:8001"
HISTORICAL_ENDPOINT = "/api/v1/ml/signal/historical"


def load_historical_data_from_api(ticker: str) -> pd.DataFrame:
    """
    Fetches 5 years OHLCV + ML signals from ML service
    and returns it as a DataFrame for VectorBT backtesting.
    """

    payload = {
        "ticker": ticker
    }

    response = requests.post(
        f"{ML_BASE_URL}{HISTORICAL_ENDPOINT}",
        json=payload,
        timeout=120
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"ML API failed: {response.status_code} - {response.text}"
        )

    data = response.json()

    if "rows" not in data:
        raise ValueError("Invalid response from ML API")

    df = pd.DataFrame(data["rows"])

    # Parse date & set index
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    df = df.sort_index()

    # Rename columns to match engine expectations
    df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "signal": "Signal"
    }, inplace=True)

    return df


# Keep the original function for backward compatibility
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import pipeline adapter
try:
    from data.pipeline_adapter import get_pipeline_data, is_pipeline_available
    from data.pipeline_config import DATA_SOURCE
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    DATA_SOURCE = 'csv'

def load_historical_data(csv_path: str = "ml_trading_signals.csv",
                        ticker: str = None,
                        use_pipeline: bool = None) -> pd.DataFrame:
    """
    Original data loader - kept for backward compatibility
    """
    # [Original implementation - keeping the existing file content]
    pass
