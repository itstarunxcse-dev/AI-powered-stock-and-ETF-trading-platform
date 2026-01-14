# Common Valid Tickers List
# Shared across the application for consistent dropdowns

COMMON_TICKERS = [
    # US Tech / Large Cap
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA",
    "JPM", "V", "MA", "COST", "UNH", "AVGO", "INTC", "IBM", 
    "GE", "F", "GM", "T", "VZ", "WMT", "DIS", "KO", "PYPL", 
    "SNAP", "LYFT", "SPY", "QQQ",
    
    # Indian / NSE
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", 
    "LT.NS", "BHARTIARTL.NS", "ITC.NS", "HINDUNILVR.NS", "ASIANPAINT.NS", 
    "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS", "YESBANK.NS", "IDEA.NS", 
    "SUZLON.NS", "RPOWER.NS", "JPPOWER.NS", "GMRAIRPORT.NS", "SAIL.NS", 
    "BHEL.NS", "ZEEL.NS", "PAYTM.NS", "NYKAA.NS"
]

def get_common_tickers():
    return sorted(COMMON_TICKERS)
