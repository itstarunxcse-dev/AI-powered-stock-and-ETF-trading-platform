import streamlit as st
from typing import Tuple, List
from ui.utils.constants import get_common_tickers

# Define constants for easy maintenance
PERIOD_OPTIONS = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"]
INTERVAL_OPTIONS = ["1d", "1wk", "1mo"]

def render_controls() -> Tuple[str, str, str]:
    """
    Renders unified trading controls using shared ticker list.
    
    Returns:
        Tuple[str, str, str]: (symbol, period, interval)
    """
    # Get available tickers
    all_tickers = get_common_tickers()
    
    # 1. Main Ticker Select
    # Try to persist state or default to AAPL
    current = st.session_state.get('current_msg_symbol', 'AAPL')
    if current not in all_tickers: 
        current = 'AAPL'
        
    idx = all_tickers.index(current) if current in all_tickers else 0
    
    symbol = st.selectbox(
        "Select Ticker", 
        all_tickers, 
        index=idx,
        help="Select a stock from the list to analyze."
    )
    
    # Update local state tracker if needed, or just return
    
    # 2. Advanced Settings
    with st.expander("⚙️ Advanced Settings"):
        col1, col2 = st.columns(2)
        with col1:
            period = st.selectbox("History", PERIOD_OPTIONS, index=3)
        with col2:
            interval = st.selectbox("Interval", INTERVAL_OPTIONS, index=0)

    return symbol.strip().upper(), period, interval