# -*- coding: utf-8 -*-
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from contracts.schema import StockData
from typing import Optional
import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import pipeline integration
try:
    from data.pipeline_adapter import get_pipeline_data, is_pipeline_available
    from data.pipeline_config import DATA_SOURCE, PIPELINE_API_URL
    from data.api_client import DashboardAPIClient
    PIPELINE_AVAILABLE = True
    API_CLIENT = DashboardAPIClient(PIPELINE_API_URL)
except ImportError:
    PIPELINE_AVAILABLE = False
    DATA_SOURCE = 'yfinance'
    API_CLIENT = None

class DataEngine:
    _api_unreachable = False

    @staticmethod
    def fetch_data(symbol: str, period: str = "1y", interval: str = "1d", 
                   use_pipeline: bool = None, use_api: bool = True) -> StockData:
        """
        Fetches market data and calculates technical indicators.
        
        Data sources (priority order):
        1. API endpoints (if available and use_api=True)
        2. Pipeline data (if configured and available)
        3. Yahoo Finance (yfinance) - fallback
        
        Cached for 5 minutes to improve performance.
        """
        import streamlit as st
        
        @st.cache_data(ttl=300, show_spinner=False)
        def _fetch_cached_v2(symbol: str, period: str, interval: str, use_pipeline_flag: bool, use_api_flag: bool) -> dict:
            data = DataEngine._fetch_uncached(symbol, period, interval, use_pipeline_flag, use_api_flag)
            return data.dict()
        
        should_use_pipeline = use_pipeline if use_pipeline is not None else (DATA_SOURCE == 'pipeline')
        data_dict = _fetch_cached_v2(symbol, period, interval, should_use_pipeline, use_api)
        return StockData(**data_dict)
    
    @staticmethod
    def _fetch_uncached(symbol: str, period: str, interval: str, use_pipeline: bool = False, use_api: bool = True) -> StockData:
        """Internal method to fetch data without caching"""
        
        # Check global circuit breaker
        if DataEngine._api_unreachable:
            use_api = False

        # Try API first if available and enabled
        if use_api and API_CLIENT is not None:
            try:
                return DataEngine._fetch_from_api(symbol, period)
            except Exception as e:
                error_str = str(e)
                if "WinError 10061" in error_str or "Connection refused" in error_str:
                    print(f"⚠️ API Connection Failed. Switching to OFFLINE mode for this session.")
                    DataEngine._api_unreachable = True
                else:
                    print(f"⚠️ API Error: {e}. Falling back...")
        
        # Try pipeline second if configured
        if use_pipeline and PIPELINE_AVAILABLE and is_pipeline_available():
            try:
                return DataEngine._fetch_from_pipeline(symbol, period)
            except Exception as e:
                print(f"⚠️ Pipeline Error: {e}. Falling back to yfinance...")
        
        # Fallback to yfinance
        return DataEngine._fetch_from_yfinance(symbol, period, interval)
    
    @staticmethod
    def _fetch_from_api(symbol: str, period: str) -> StockData:
        """
        Fetch data from API endpoints.
        
        Args:
            symbol: Stock ticker symbol
            period: Time period (converted to days)
        
        Returns:
            StockData object
        """
        # Convert period to days
        period_days = {
            '1d': 1, '5d': 5, '1mo': 30, '3mo': 90,
            '6mo': 180, '1y': 365, '2y': 730, '5y': 1825, 'ytd': 365, 'max': 3650
        }
        days = period_days.get(period, 30)
        
        try:
            print(f"📡 [DEBUG] Fetching {symbol} from API (days={days})...")
            
            # Use the new API endpoint
            api_response = API_CLIENT.get_recent_data(symbol, days=days)
            
            # Check for error
            if isinstance(api_response, dict) and "error" in api_response:
                print(f"❌ [ERROR] API returned error: {api_response['error']}")
                raise ValueError(f"API Error: {api_response['error']}")

            # Check if API returned real data
            if api_response.get('data') and len(api_response['data']) > 0:
                print(f"✅ [SUCCESS] Received {len(api_response['data'])} records for {symbol}")
                
                # Process API data into StockData format
                df = pd.DataFrame(api_response['data'])
                return DataEngine._process_api_data(df, symbol)
            else:
                print(f"⚠️ [WARNING] API returned empty data for {symbol}")
                raise ValueError(f"No data returned from API for {symbol}")
                
        except Exception as e:
            # Propagate error to _fetch_uncached for handling/fallback
            # We avoid printing "CRITICAL" here to reduce noise for connection errors
            raise e
    
    @staticmethod
    def _process_api_data(df: pd.DataFrame, symbol: str) -> StockData:
        """Process API response data into StockData format."""
        try:
            # 1. Prepare DataFrame
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
            
            # 2. Extract Basic Data
            dates = df['date'].dt.strftime('%Y-%m-%d').tolist() if 'date' in df.columns else []
            closes = df['close'].tolist() if 'close' in df.columns else []
            opens = df['open'].tolist() if 'open' in df.columns else []
            highs = df['high'].tolist() if 'high' in df.columns else []
            lows = df['low'].tolist() if 'low' in df.columns else []
            volumes = df['volume'].tolist() if 'volume' in df.columns else []
            
            if not closes:
                raise ValueError("No close price data available")

            # 3. Calculate Technical Indicators
            s_close = pd.Series(closes)
            
            # RSI (14)
            delta = s_close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi_series = 100 - (100 / (1 + rs))
            rsi = rsi_series.fillna(50).tolist()
            
            # SMAs
            sma_20 = s_close.rolling(window=20).mean().fillna(s_close).tolist()
            sma_50 = s_close.rolling(window=50).mean().fillna(s_close).tolist()
            
            # EMAs
            ema_12 = s_close.ewm(span=12, adjust=False).mean().tolist()
            ema_26 = s_close.ewm(span=26, adjust=False).mean().tolist()
            
            # MACD
            s_ema_12 = pd.Series(ema_12)
            s_ema_26 = pd.Series(ema_26)
            macd_series = s_ema_12 - s_ema_26
            macd_signal_series = macd_series.ewm(span=9, adjust=False).mean()
            macd_hist_series = macd_series - macd_signal_series
            
            macd = macd_series.fillna(0).tolist()
            macd_signal = macd_signal_series.fillna(0).tolist()
            macd_hist = macd_hist_series.fillna(0).tolist()

            # 4. Summary Stats
            current_price = closes[-1]
            prev_close = closes[-2] if len(closes) > 1 else current_price
            price_change = current_price - prev_close
            price_change_pct = (price_change / prev_close) * 100 if prev_close != 0 else 0.0
            
            market_status = "Open" if datetime.now().weekday() < 5 else "Closed"

            return StockData(
                symbol=symbol.upper(),
                current_price=float(current_price),
                price_change=float(price_change),
                price_change_pct=float(price_change_pct),
                last_updated=datetime.now(),
                market_status=market_status,
                dates=dates,
                opens=opens,
                highs=highs,
                lows=lows,
                closes=closes,
                volumes=volumes,
                rsi=rsi,
                sma_20=sma_20,
                sma_50=sma_50,
                ema_12=ema_12,
                ema_26=ema_26,
                macd=macd,
                macd_signal=macd_signal,
                macd_hist=macd_hist
            )
            
        except Exception as e:
            print(f"Error processing API data: {e}")
            raise ValueError(f"Processing failed: {e}")
    
    @staticmethod
    def _fetch_from_yfinance(symbol: str, period: str, interval: str) -> StockData:
        """Original yfinance implementation"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Calculate Indicators
            df['RSI'] = DataEngine._calculate_rsi(df['Close'])
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
            df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
            
            # MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            
            # Fill NaNs for JSON serialization
            df = df.fillna(0)
            
            # Get latest info
            try:
                info = ticker.info
                current_price = info.get('currentPrice', df['Close'].iloc[-1])
                prev_close = info.get('previousClose', df['Close'].iloc[-2] if len(df) > 1 else df['Close'].iloc[-1])
            except Exception:
                current_price = df['Close'].iloc[-1]
                prev_close = df['Close'].iloc[-2] if len(df) > 1 else df['Close'].iloc[-1]

            price_change = current_price - prev_close
            price_change_pct = (price_change / prev_close) * 100 if prev_close else 0
            
            market_status = "Open" if datetime.now().weekday() < 5 else "Closed"

            return StockData(
                symbol=symbol.upper(),
                current_price=float(current_price),
                price_change=float(price_change),
                price_change_pct=float(price_change_pct),
                last_updated=datetime.now(),
                market_status=market_status,
                dates=df.index.strftime('%Y-%m-%d').tolist(),
                opens=df['Open'].tolist(),
                highs=df['High'].tolist(),
                lows=df['Low'].tolist(),
                closes=df['Close'].tolist(),
                volumes=df['Volume'].astype(int).tolist(),
                rsi=df['RSI'].tolist(),
                sma_20=df['SMA_20'].tolist(),
                sma_50=df['SMA_50'].tolist(),
                ema_12=df['EMA_12'].tolist(),
                ema_26=df['EMA_26'].tolist(),
                macd=df['MACD'].tolist(),
                macd_signal=df['MACD_Signal'].tolist(),
                macd_hist=df['MACD_Hist'].tolist()
            )
            
        except Exception as e:
            print(f"❌ Error fetching {symbol} from Yahoo: {e}")
            return None

    @staticmethod
    def _calculate_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def load_historical_data(csv_path: str = "ml_trading_signals.csv") -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
        df.set_index("Date", inplace=True)
        return df.sort_index().dropna()
    
    @staticmethod
    def load_ml_signals_data(csv_path: str = "ml_trading_signals.csv", 
                             ticker: Optional[str] = None) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
        df.set_index("Date", inplace=True)
        df = df.sort_index().dropna()
        if ticker and 'Ticker' in df.columns:
            df = df[df['Ticker'] == ticker].copy()
        return df
    
    @staticmethod
    def prepare_ml_data_for_backtest(csv_path: str = "ml_trading_signals.csv",
                                     ticker: str = "AAPL") -> pd.DataFrame:
        df = DataEngine.load_ml_signals_data(csv_path, ticker)
        backtest_df = pd.DataFrame({
            'Open': df['Open'],
            'High': df['High'],
            'Low': df['Low'],
            'Close': df['Close'],
            'Volume': df['Volume'].astype(int),
            'Signal': df['Signal'].astype(int)
        })
        return backtest_df
