# -*- coding: utf-8 -*-
import streamlit as st
import sys
import logging
from pathlib import Path
from typing import Optional, List, Protocol
from dataclasses import dataclass, field
import datetime

# --- Configuration & Path Setup ---
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="AI Signals - AI Stock Trading",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

from ui.utils.design import load_design_system

# Load Design System
load_design_system()

# --- Imports ---
from contracts.schema import StockData, MLSignal
from ui.utils.constants import get_common_tickers
from ui.utils.design import load_design_system
from ui.components.controls import render_controls
from ui.components.charts import render_price_chart, render_rsi_chart
from ui.components.prediction_card import render_prediction_card
from ui.components.indicators import render_indicators_panel, render_macd_chart
from data.fetcher import DataEngine
from ml.predictor import MLEngine
from ui.components.prediction_card import render_prediction_card
from ui.components.indicators import render_indicators_panel, render_macd_chart
from data.fetcher import DataEngine
from ml.predictor import MLEngine

def render_prediction_card_stub(signal): 
        st.info(f"🤖 Prediction: **{signal.action}** | Confidence: {signal.confidence:.0f}%")
        st.write(signal.reasoning)

# --- Styles ---
# --- Styles ---
def inject_glassmorphism_css():
    st.markdown("""
        <style>
        /* Page Background: Black -> Dark Gray/Soft Light */
        .stApp {
            background: radial-gradient(circle at 50% 0%, #262626 0%, #000000 100%) !important;
            background-attachment: fixed;
        }

        .glass-header {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.03) 100%);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 24px;
            margin-bottom: 24px;
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-top: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            position: relative;
            overflow: hidden;
        }
        
        /* Shine effect */
        .glass-header::before {
            content: "";
            position: absolute;
            top: 0; left: -50%; width: 50%; height: 100%;
            background: linear-gradient(to right, transparent, rgba(255,255,255,0.05), transparent);
            transform: skewX(-20deg);
            pointer-events: none;
        }

        /* Metric/Card Styles */
        div[data-testid="stMetric"], .metric-card {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 20px;
            transition: all 0.2s ease;
            backdrop-filter: blur(10px);
        }
        
        div[data-testid="stMetric"]:hover {
            background: rgba(255, 255, 255, 0.06);
            border-color: rgba(255, 255, 255, 0.15);
            transform: translateY(-2px);
        }

        /* Responsive Adjustments */
        @media (max-width: 768px) {
            .glass-header {
                padding: 18px;
                margin-bottom: 18px;
            }
            div[data-testid="stMetricValue"] {
                font-size: 22px !important;
            }
            .stMainBlockContainer {
                padding-left: 0.5rem !important;
                padding-right: 0.5rem !important;
            }
        }
        </style>
    """, unsafe_allow_html=True)

class Styles:
    HEADER = """
        <div class="glass-header">
            <div style="font-size: 28px; font-weight: 800; color: #f8fafc; letter-spacing: -0.5px; text-shadow: 0 0 20px rgba(255,255,255,0.1);">📊 AI Signals</div>
            <div style="font-size: 14px; color: #94a3b8; margin-top: 6px;">Real-time stock analysis with AI signals</div>
        </div>
    """

    METRIC_CARD = """
        <div class="metric-card">
            <div style="font-size: 14px; font-weight: 700; color: #cbd5e1; letter-spacing: 0.5px; text-transform: uppercase;">Market Pulse</div>
        </div>
    """

# --- Service Layer ---

@st.cache_resource
def get_ml_engine():
    return MLEngine()

@st.cache_resource(ttl=300)
def get_stock_data(symbol: str, period: str, interval: str) -> Optional[StockData]:
    return DataEngine.fetch_data(symbol, period, interval)

def get_prediction(stock_data: StockData) -> Optional[MLSignal]:
    """Get ML prediction with error handling - NO CACHING for fresh predictions"""
    try:
        model = get_ml_engine()
        # Generate fresh prediction every time - don't cache based on stock data
        return model.predict(stock_data)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return None

# --- Session & Search Management ---

class SessionManager:
    """Handles App State and Search History"""
    
    @staticmethod
    def initialize():
        if 'stock_data' not in st.session_state:
            st.session_state['stock_data'] = None
        if 'ml_signal' not in st.session_state:
            st.session_state['ml_signal'] = None
        if 'search_history' not in st.session_state:
            st.session_state['search_history'] = []
        if 'current_symbol' not in st.session_state:
            st.session_state['current_symbol'] = ""
        if 'period' not in st.session_state:
            st.session_state['period'] = "1y"
        if 'interval' not in st.session_state:
            st.session_state['interval'] = "1d"

    @staticmethod
    def add_to_history(symbol: str):
        symbol = symbol.upper()
        history = st.session_state['search_history']
        if symbol in history:
            history.remove(symbol)
        history.insert(0, symbol)
        st.session_state['search_history'] = history[:5]  # Keep last 5

    @staticmethod
    def set_data(data: StockData, signal: MLSignal):
        st.session_state['stock_data'] = data
        st.session_state['ml_signal'] = signal
        # Update current symbol text input
        st.session_state['current_symbol'] = data.symbol

    @staticmethod
    def get_data():
        return st.session_state.get('stock_data'), st.session_state.get('ml_signal')

# --- Logic Pipelines ---

def run_analysis_pipeline(symbol: str, period: str, interval: str):
    """Centralized pipeline for stock analysis with comprehensive error handling"""
    clean_symbol = symbol.strip().upper()
    
    if not clean_symbol:
        st.warning("⚠️ Please enter a valid stock symbol.")
        return

    with st.status(f"Processing {clean_symbol}...", expanded=True) as status:
        try:
            # Step 1: Data fetch
            st.write("📡 Connecting to market data API...")
            stock_data = get_stock_data(clean_symbol, period, interval)
            
            # Step 2: ML prediction - Force fresh prediction by clearing any cache
            st.write("🧠 Running AI prediction models...")
            # Clear the Streamlit cache for this specific prediction to ensure fresh results
            if hasattr(get_prediction, 'clear'):
                get_prediction.clear()
            ml_signal = get_prediction(stock_data)
            
            if not ml_signal:
                status.update(label="❌ Prediction Failed", state="error")
                st.error("ML model returned no prediction.")
                return

            # Step 3: Update state atomically
            SessionManager.set_data(stock_data, ml_signal)
            SessionManager.add_to_history(clean_symbol)
            
            status.update(label=f"✅ Analysis Complete - {ml_signal.action}", state="complete", expanded=False)
            logger.info(f"Successfully analyzed {clean_symbol}: {ml_signal.action} ({ml_signal.confidence:.1f}%)")
            
        except Exception as e:
            status.update(label="❌ Analysis Failed", state="error")
            st.error(f"Error: {str(e)}")
            logger.error(f"Pipeline error for {clean_symbol}: {e}", exc_info=True)
            return

# --- UI Components ---

def render_sidebar_search():
    """Enhanced Sidebar with History and Quick Actions"""
    st.markdown("""
        <div style='background: #0b1220; border-radius: 14px; padding: 16px; margin-bottom: 14px; border: 1px solid rgba(255,255,255,0.05);'>
            <div style='display: flex; align-items: center; gap: 8px;'>
                <span style='font-size: 16px;'>🔍</span>
                <span style='font-size: 16px; font-weight: 700; color: #e2e8f0;'>Market Search</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # 1. Main Search Input (Dropdown)
    all_tickers = get_common_tickers()
    current_sym = st.session_state.get('current_symbol', 'AAPL')
    
    # Handle custom symbols not in list gracefully
    if current_sym not in all_tickers and current_sym:
        # If we want to allow custom, we might need a way to add it or just default to first
        # For now, let's just default to first if invalid, or maybe append it temporarily?
        # Appending temporarily is better UX
        all_tickers.insert(0, current_sym)
        default_ix = 0
    else:
        default_ix = all_tickers.index(current_sym) if current_sym in all_tickers else 0

    symbol = st.selectbox(
        "Select Ticker", 
        options=all_tickers, 
        index=default_ix,
        key="ticker_select",
        help="Select a stock from the list."
    )

    # 2. Configuration (Collapsible to save space)
    with st.expander("⚙️ Advanced Settings", expanded=False):
        period = st.selectbox(
            "📅 Time Period", 
            ["1mo", "3mo", "6mo", "1y", "2y", "5y"], 
            index=["1mo", "3mo", "6mo", "1y", "2y", "5y"].index(st.session_state.get('period', '1y')),
            key='period'
        )
        interval = st.selectbox(
            "⏱️ Data Interval", 
            ["1d", "1wk", "1mo"], 
            index=["1d", "1wk", "1mo"].index(st.session_state.get('interval', '1d')),
            key='interval'
        )

    # 3. Primary Action
    if st.button("🚀 Analyze Stock", type="primary", use_container_width=True, key="analyze_btn"):
        # Update session state from the selected value before running
        st.session_state['current_symbol'] = symbol
        run_analysis_pipeline(symbol, st.session_state.period, st.session_state.interval)

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. Quick Select (Top Favorites)
    st.markdown("""
        <div style='font-size: 12px; font-weight: 700; color: #94a3b8; 
             margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px;'>
            Quick Favorites
        </div>
    """, unsafe_allow_html=True)
    
    favorites = ["AAPL", "NVDA", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "TSLA"]
    cols = st.columns(3)
    for idx, ticker in enumerate(favorites):
        with cols[idx % 3]:
            if st.button(ticker, key=f"pop_{ticker}", use_container_width=True):
                st.session_state['current_symbol'] = ticker
                run_analysis_pipeline(ticker, st.session_state.period, st.session_state.interval)
                st.rerun()

    # 5. Recent History Section
    if st.session_state['search_history']:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
            <div style='font-size: 12px; font-weight: 700; color: #94a3b8; 
                 margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px;'>
                Recent Searches
            </div>
        """, unsafe_allow_html=True)
        
        for hist_ticker in st.session_state['search_history']:
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(f"📈 {hist_ticker}", key=f"hist_{hist_ticker}", width='stretch'):
                    st.session_state['current_symbol'] = hist_ticker
                    run_analysis_pipeline(hist_ticker, st.session_state.period, st.session_state.interval)
            with col2:
                if st.button("✕", key=f"del_{hist_ticker}", help="Remove from history"):
                    if hist_ticker in st.session_state['search_history']:
                        st.session_state['search_history'].remove(hist_ticker)
                        st.rerun()

    # 6. Utility Actions
    st.markdown("<br>", unsafe_allow_html=True)
    col_clear1, col_clear2 = st.columns(2)
    with col_clear1:
        if st.button("🗑️ Clear History", width='stretch', help="Clear search history"):
            st.session_state['search_history'] = []
            st.success("✅ History cleared")
    with col_clear2:
        if st.button("🔄 Reset Cache", width='stretch', help="Clear all cached data"):
            try:
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("✅ Cache cleared")
            except Exception as e:
                st.error(f"Cache clear failed: {e}")

def render_sidebar_metrics(stock_data: StockData):
    st.markdown(Styles.METRIC_CARD, unsafe_allow_html=True)
    if not stock_data or not stock_data.highs:
        return

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Price", f"₹{stock_data.current_price:.2f}", f"{stock_data.price_change_pct:.2f}%")
        st.metric("High", f"₹{max(stock_data.highs):.2f}")
    with col2:
        val = stock_data.volumes[-1] if stock_data.volumes else 0
        vol_str = f"{val/1e6:.1f}M" if val > 1e6 else f"{val/1e3:.1f}K"
        st.metric("Volume", vol_str)
        st.metric("Low", f"₹{min(stock_data.lows):.2f}")


def inject_glassmorphism_css():
    """Inject reusable glassmorphism CSS once"""
    st.markdown(
        """
        <style>
        /* Glassmorphism for Metrics */
        [data-testid="stMetric"] {
            background: rgba(15, 23, 42, 0.6);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border-radius: 16px;
            padding: 20px 16px;
            border: 1px solid rgba(255, 255, 255, 0.15);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
            transition: all 0.3s ease;
        }
        [data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.25);
        }
        [data-testid="stMetric"] label {
            color: #cbd5e1 !important;
        }
        [data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #f1f5f9 !important;
        }
        
        /* Glassmorphism for Tabs */
        .stTabs [data-baseweb="tab-list"] {
            background: rgba(15, 23, 42, 0.6);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border-radius: 16px;
            padding: 8px;
            gap: 8px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border-radius: 12px;
            padding: 12px 24px;
            color: #cbd5e1;
            font-weight: 600;
            transition: all 0.3s ease;
            border: 1px solid transparent;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(255, 255, 255, 0.08);
            color: #f1f5f9;
        }
        .stTabs [aria-selected="true"] {
            background: rgba(100, 149, 237, 0.25) !important;
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            color: #f1f5f9 !important;
            border: 1px solid rgba(100, 149, 237, 0.4) !important;
        }
        
        /* Glassmorphism for Tab Panels */
        .stTabs [data-baseweb="tab-panel"] {
            background: rgba(15, 23, 42, 0.4);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-radius: 16px;
            padding: 24px;
            margin-top: 16px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }
        
        /* Glassmorphism for Expanders */
        .streamlit-expanderHeader {
            background: rgba(15, 23, 42, 0.6);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_market_pulse(stock_data: Optional[StockData]):
    """Primary market snapshot with glassmorphism styling; falls back to provided sample when data is unavailable."""
    snapshot = {
        "price": 273.40,
        "change_pct": -0.15,
        "high": 288.62,
        "low": 168.63,
        "volume": 21_500_000,
    }

    if stock_data:
        snapshot = {
            "price": stock_data.current_price,
            "change_pct": stock_data.price_change_pct,
            "high": max(stock_data.highs) if stock_data.highs else None,
            "low": min(stock_data.lows) if stock_data.lows else None,
            "volume": stock_data.volumes[-1] if stock_data.volumes else None,
        }

    st.markdown(
        """
        <div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.8), rgba(30, 41, 59, 0.8)); 
             backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
             border-radius: 20px; padding: 24px; border: 1px solid rgba(255, 255, 255, 0.2);
             box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37); margin-bottom: 24px;">
            <div style="font-size: 20px; font-weight: 800; color: #f1f5f9; letter-spacing: 0.5px; 
                 text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3); display: flex; align-items: center; gap: 10px;">
                <span>💎</span><span>Market Pulse</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b, col_c, col_d = st.columns([1.2, 1, 1, 1])
    with col_a:
        st.metric("Price", f"₹{snapshot['price']:.2f}", f"{snapshot['change_pct']:.2f}%")
    with col_b:
        st.metric("High", f"₹{(snapshot['high'] or 0):.2f}")
    with col_c:
        vol = snapshot["volume"] or 0
        vol_str = f"{vol/1e6:.1f}M" if vol >= 1e6 else f"{vol/1e3:.1f}K" if vol >= 1e3 else f"{vol:.0f}"
        st.metric("Volume", vol_str)
    with col_d:
        st.metric("Low", f"₹{(snapshot['low'] or 0):.2f}")




# --- Main Application ---

INFO_BOX = """
<div style="background: linear-gradient(135deg, rgba(15,23,42,0.78), rgba(30,41,59,0.78));
            backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
            border-radius: 18px; padding: 20px 22px; margin-top: 16px;
            border: 1px solid rgba(255, 255, 255, 0.12);
            box-shadow: 0 12px 36px rgba(0,0,0,0.45);">
    {content}
</div>
"""

def rsi_explanation():
    st.markdown(
        """
            <div style="font-size:14px; line-height:1.7; color:rgba(255,255,255,0.85)">
                <strong style="font-size:16px; color:#fff;">
                    RSI (Relative Strength Index)
                </strong>
                <ul style="margin:12px 0 0; padding-left:18px;">
                    <li><span style="color:#ff6347; font-weight:600;">Above 70</span>: Overbought</li>
                    <li><span style="color:#00ffff; font-weight:600;">Below 30</span>: Oversold</li>
                    <li><span style="color:#ffd700; font-weight:600;">30–70</span>: Neutral zone</li>
                </ul>
            </div>
        """,
        unsafe_allow_html=True
    )

def macd_explanation():
    st.markdown(
        """
            <div style="font-size: 14px; color: rgba(255, 255, 255, 0.85); line-height: 1.7;">
                <strong style="font-size: 16px; color: #fff;">MACD (Moving Average Convergence Divergence)</strong>
                <ul style="margin: 12px 0 0 18px; padding: 0; list-style: disc;">
                    <li><span style="color: #00BFFF; font-weight: 600;">MACD Line</span> crosses above <span style="color: #FF6347; font-weight: 600;">Signal Line</span>: Bullish</li>
                    <li><span style="color: #00BFFF; font-weight: 600;">MACD Line</span> crosses below <span style="color: #FF6347; font-weight: 600;">Signal Line</span>: Bearish</li>
                    <li><span style="color: #ffd700; font-weight: 600;">Histogram</span>: Shows momentum strength and direction</li>
                </ul>
            </div>
        """,
        unsafe_allow_html=True
    )

def main():
    SessionManager.initialize()
    inject_glassmorphism_css()
    # load_design_system() # Uncomment if you have the CSS file
    st.markdown(Styles.HEADER, unsafe_allow_html=True)

    # --- Sidebar Layout ---
    with st.sidebar:
        symbol, period, interval = render_controls()
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Analyze Stock", type="primary", width='stretch'):
            run_analysis_pipeline(symbol, period, interval)

    # --- Main Content Layout ---
    stock_data, ml_signal = SessionManager.get_data()

    if stock_data and ml_signal:
        render_market_pulse(stock_data)
        # Improved Navigation using Tabs
        tab1, tab2, tab3 = st.tabs(["📉 Price Action", "🧠 AI Insights", "📊 Technicals"])
        
        with tab1:
            st.markdown(f"### {stock_data.symbol} - Price Action")
            st.markdown("""<div style='margin-top: 20px;'></div>""", unsafe_allow_html=True)
            render_price_chart(stock_data)
        
        with tab2:
            st.markdown("""<div style='margin-top: 10px;'></div>""", unsafe_allow_html=True)
            render_prediction_card(ml_signal)

        with tab3:
            st.markdown("### 🎯 Technical Indicators")
            st.markdown("""<div style='margin-top: 20px;'></div>""", unsafe_allow_html=True)
            
            render_indicators_panel(stock_data)

            tabs = st.tabs(["📊 RSI", "📉 MACD"])
            with tabs[0]:
                render_rsi_chart(stock_data)
                rsi_explanation()
            with tabs[1]:
                render_macd_chart(stock_data)
                macd_explanation()

    else:
        # Empty State Guide
        st.info("👈 Use the search bar or select a quick ticker to begin.")

if __name__ == "__main__":
    main()