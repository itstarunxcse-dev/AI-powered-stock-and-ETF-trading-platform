# -*- coding: utf-8 -*-
import streamlit as st
import sys
import logging
from pathlib import Path
from typing import Optional

# ======================================================
# PATH & LOGGING
# ======================================================
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("AnalysisPage")

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Strategy Analysis - AI Stock Trading",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# IMPORTS (AFTER PATH SETUP)
# ======================================================
from ui.utils.design import load_design_system
from ui.components.controls import render_controls
from ui.components.charts import (
    render_rsi_chart,
    render_equity_curve,
    render_equity_comparison,
    render_drawdown_chart,
    render_price_with_trades_chart,
    render_profit_loss_chart,
    render_volume_chart
)
from ui.components.indicators import (
    render_indicators_panel,
    render_macd_chart
)
from ui.components.metrics import (
    render_metrics,
    render_strategy_config,
    render_signal_logic,
    render_trade_history
)
from ui.components.export import export_data_section
from data.fetcher import DataEngine
from backtesting.engine import BacktestEngine
from ui.utils.constants import get_common_tickers

# ======================================================
# SESSION KEYS
# ======================================================
class SessionKeys:
    STOCK_DATA = "stock_data"
    BACKTEST = "backtest_metrics"
    ML_SIGNAL = "ml_signal"
    LAST_QUERY = "last_query"

# ======================================================
# STYLES
# ======================================================
HEADER_HTML = """
<div class="page-header">
    <div class="page-title">
        📈 Strategy Analysis
    </div>
    <div class="page-subtitle">
        Advanced indicators, strategy validation, and performance insights
    </div>
</div>
"""

INFO_BOX = """
<div class="info-box">
    {content}
</div>
"""

# ======================================================
# CACHE SAFE FUNCTIONS
# ======================================================
@st.cache_data(show_spinner=False)
def fetch_stock_data(symbol: str, period: str, interval: str):
    return DataEngine.fetch_data(symbol, period, interval)

# ======================================================
# HELPERS
# ======================================================
def render_header():
    # Override background to black for high contrast analysis
    st.markdown("""
        <style>
        .stApp {
            background: radial-gradient(circle at 50% 0%, #1c1c1c 0%, #000000 100%) !important;
            animation: none !important;
        }
        </style>
    """, unsafe_allow_html=True)
    st.markdown(HEADER_HTML, unsafe_allow_html=True)

def validate_stock_data(data) -> bool:
    return data is not None and hasattr(data, "closes") and len(data.closes) > 0

def reset_backtest_state():
    st.session_state.pop(SessionKeys.BACKTEST, None)

# ======================================================
# DATA FETCH UI
# ======================================================
def handle_missing_data():
    st.warning("⚠️ No stock data loaded yet.")
    st.markdown("### 🔍 Quick Analysis")

    symbol, period, interval = render_controls()

    if st.button("🚀 Analyze Stock", type="primary", width='stretch'):
        try:
            with st.spinner(f"Fetching {symbol.upper()} data..."):
                data = fetch_stock_data(symbol, period, interval)

            if not validate_stock_data(data):
                st.error("❌ Invalid or empty market data received.")
                return

            st.session_state[SessionKeys.STOCK_DATA] = data
            st.session_state[SessionKeys.LAST_QUERY] = (symbol, period, interval)
            reset_backtest_state()
            st.rerun()

        except Exception as e:
            logger.exception("Data fetch failed")
            st.error(str(e))

# ======================================================
# BACKTEST RENDERING
# ======================================================
def render_backtest(metrics):
    st.markdown("### 📊 Performance Metrics")
    render_metrics(metrics)
    st.markdown("---")
    st.markdown("### 📈 Equity Curve: Market vs ML Strategy")
    # Main comparison chart with trades and volume-style activity
    render_equity_comparison(
        metrics.dates,
        metrics.market_equity,
        metrics.equity_curve,
        metrics.buy_signals,
        metrics.sell_signals,
        metrics.prices
    )
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        render_profit_loss_chart(metrics.dates, metrics.returns)
    with col2:
        render_volume_chart(metrics.dates, metrics.volumes)
    
    render_trade_history(metrics)

# ======================================================
# MAIN
# ======================================================
def main():
    load_design_system()
    render_header()

    # ================= SIDEBAR NAVIGATION =================
    with st.sidebar:
        st.markdown("""
            <div class="sidebar-card">
                <div class="sidebar-title">
                    🔍 Quick Analysis
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        symbol, period, interval = render_controls()
        
        if st.button("🚀 Analyze Stock", type="primary", width='stretch', key="sidebar_analyze"):
            try:
                with st.spinner(f"Fetching {symbol} data..."):
                    data = fetch_stock_data(symbol, period, interval)

                if not validate_stock_data(data):
                    st.error("❌ Invalid or empty market data received.")
                else:
                    st.session_state[SessionKeys.STOCK_DATA] = data
                    st.session_state[SessionKeys.LAST_QUERY] = (symbol, period, interval)
                    reset_backtest_state()
                    st.success(f"✅ Loaded {symbol}")
                    st.rerun()

            except Exception as e:
                logger.exception("Data fetch failed")
                st.error(f"❌ {str(e)}")
        
        # Display current loaded stock
        if SessionKeys.STOCK_DATA in st.session_state:
            stock_data = st.session_state[SessionKeys.STOCK_DATA]
            if stock_data:
                st.markdown("---")
                st.markdown(f"""
                    <div class="status-card">
                        <div style="font-size: 12px; color: rgba(255, 255, 255, 0.6); margin-bottom: 5px;">
                            CURRENTLY LOADED
                        </div>
                        <div style="font-size: 24px; font-weight: 800; color: #00ff7f;">
                            {stock_data.symbol}
                        </div>
                        <div style="font-size: 14px; color: rgba(255, 255, 255, 0.8); margin-top: 5px;">
                            ₹{stock_data.current_price:.2f}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

    # ================= MAIN CONTENT =================
    if SessionKeys.STOCK_DATA not in st.session_state or st.session_state[SessionKeys.STOCK_DATA] is None:
        st.info("👈 Use the sidebar to analyze a stock")
        return

    stock_data = st.session_state[SessionKeys.STOCK_DATA]

    # ================= BACKTEST =================
    st.markdown("---")
    st.markdown("### 🧪 Strategy Backtesting")

    run_disabled = SessionKeys.BACKTEST in st.session_state
    if st.button("▶️ Run Backtest", type="primary", disabled=run_disabled):
        try:
            with st.spinner("Running strategy backtest..."):
                metrics = BacktestEngine.run_backtest(stock_data)
                st.session_state[SessionKeys.BACKTEST] = metrics
                st.success("✅ Backtest completed")
        except Exception as e:
            logger.exception("Backtest failed")
            st.error(str(e))

    if SessionKeys.BACKTEST in st.session_state:
        render_backtest(st.session_state[SessionKeys.BACKTEST])

        st.markdown("---")
        export_data_section(
            stock_data,
            st.session_state.get(SessionKeys.ML_SIGNAL),
            st.session_state[SessionKeys.BACKTEST]
        )

# ======================================================
if __name__ == "__main__":
    main()
