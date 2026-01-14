# -*- coding: utf-8 -*-
"""
Pipeline Data Adapter

Adapter for Al-Powered-Stock-ETF-Signal-Generation-Platform-pipeline integration.
Provides clean API for accessing processed market data with technical indicators.

Usage:
    from data.pipeline_adapter import get_pipeline_data, PipelineAdapter
    
    # Get data for single ticker
    df = get_pipeline_data(ticker='AAPL')
    
    # Get data for multiple tickers
    df = get_pipeline_data(tickers=['AAPL', 'MSFT', 'GOOGL'])
"""

import pandas as pd
import os
from typing import List, Optional
from pathlib import Path

from data.pipeline_config import (
    CLEAN_DATA_FILE, FEATURED_DATA_FILE, RAW_DATA_FILE,
    TICKER_FILE, PIPELINE_TO_STANDARD_COLUMNS, STANDARD_COLUMNS,
    PIPELINE_PATH
)


class PipelineAdapter:
    """
    Adapter for accessing processed data from the pipeline.
    
    This adapter provides:
    - DataFrame access to cleaned market data
    - Automatic column name standardization
    - Filtering by ticker and date range
    - Error handling for missing data
    """
    
    def __init__(self):
        """Initialize the pipeline adapter."""
        self._validate_pipeline_exists()
    
    def _validate_pipeline_exists(self):
        """Check if pipeline folder and data files exist."""
        if not PIPELINE_PATH.exists():
            raise FileNotFoundError(
                f"Pipeline folder not found: {PIPELINE_PATH}\n"
                "Please ensure Al-Powered-Stock-ETF-Signal-Generation-Platform-pipeline exists."
            )
    
    def get_available_tickers(self) -> List[str]:
        """
        Get list of available tickers from ticker.txt.
        
        Returns:
            List of ticker symbols
        """
        if not TICKER_FILE.exists():
            return []
        
        with open(TICKER_FILE, 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
        return sorted(tickers)
    
    def load_clean_data(self, tickers: Optional[List[str]] = None,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load clean market data with technical indicators.
        
        Args:
            tickers: Optional list of ticker symbols to filter
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)
        
        Returns:
            DataFrame with standardized columns
        
        Raises:
            FileNotFoundError: If clean data file doesn't exist
        """
        if not CLEAN_DATA_FILE.exists():
            raise FileNotFoundError(
                f"Clean data file not found: {CLEAN_DATA_FILE}\n"
                "Please run the pipeline first: python data_pipeline.py"
            )
        
        # Load data
        df = pd.read_parquet(CLEAN_DATA_FILE)
        
        # Filter by tickers
        if tickers:
            df = df[df['ticker'].isin(tickers)].copy()
        
        # Filter by date range
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]
        
        if df.empty:
            raise ValueError("No data found for specified filters")
        
        # Standardize column names
        df = self._standardize_columns(df)
        
        return df
    
    def load_single_ticker(self, ticker: str,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load data for a single ticker.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)
        
        Returns:
            DataFrame with ticker data and standardized columns
        """
        return self.load_clean_data(
            tickers=[ticker],
            start_date=start_date,
            end_date=end_date
        )
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename pipeline columns to standard format.
        
        Args:
            df: DataFrame with pipeline column names
        
        Returns:
            DataFrame with standardized column names
        """
        # Rename columns
        df = df.rename(columns=PIPELINE_TO_STANDARD_COLUMNS)
        
        # Set Date as index
        if 'Date' in df.columns:
            df = df.set_index('Date')
        
        # Ensure Date index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        df.index.name = 'Date'
        
        return df
    
    def get_latest_data(self, ticker: str, days: int = 50) -> pd.DataFrame:
        """
        Get most recent data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            days: Number of recent days to retrieve
        
        Returns:
            DataFrame with recent data
        """
        df = self.load_single_ticker(ticker)
        return df.tail(days)
    
    def data_exists(self) -> bool:
        """Check if processed data files exist."""
        return CLEAN_DATA_FILE.exists()
    
    def get_data_info(self) -> dict:
        """
        Get information about available data.
        
        Returns:
            Dictionary with data statistics
        """
        if not self.data_exists():
            return {
                "status": "no_data",
                "message": "No processed data found. Run pipeline first."
            }
        
        df = pd.read_parquet(CLEAN_DATA_FILE)
        
        return {
            "status": "available",
            "total_rows": len(df),
            "tickers": sorted(df['ticker'].unique().tolist()),
            "ticker_count": df['ticker'].nunique(),
            "date_range": {
                "start": df['date'].min().strftime('%Y-%m-%d'),
                "end": df['date'].max().strftime('%Y-%m-%d')
            },
            "columns": df.columns.tolist()
        }


# Convenience functions
def get_pipeline_data(ticker: Optional[str] = None,
                     tickers: Optional[List[str]] = None,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience function to get pipeline data.
    
    Args:
        ticker: Single ticker symbol (mutually exclusive with tickers)
        tickers: List of ticker symbols (mutually exclusive with ticker)
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
    
    Returns:
        DataFrame with processed market data
    
    Examples:
        # Single ticker
        df = get_pipeline_data(ticker='AAPL')
        
        # Multiple tickers
        df = get_pipeline_data(tickers=['AAPL', 'MSFT', 'GOOGL'])
        
        # With date range
        df = get_pipeline_data(ticker='AAPL', start_date='2024-01-01')
    """
    adapter = PipelineAdapter()
    
    if ticker:
        return adapter.load_single_ticker(ticker, start_date, end_date)
    elif tickers:
        return adapter.load_clean_data(tickers, start_date, end_date)
    else:
        return adapter.load_clean_data(start_date=start_date, end_date=end_date)


def is_pipeline_available() -> bool:
    """Check if pipeline data is available."""
    try:
        adapter = PipelineAdapter()
        return adapter.data_exists()
    except Exception:
        return False


def get_available_tickers() -> List[str]:
    """Get list of available tickers."""
    try:
        adapter = PipelineAdapter()
        return adapter.get_available_tickers()
    except Exception:
        return []
