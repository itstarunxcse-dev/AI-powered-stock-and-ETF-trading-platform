# 📊 Supabase Table Schema

## Your Exact Table Structure

Based on your pipeline code in `Al-Powered-Stock-ETF-Signal-Generation-Platform-pipeline/data_pipeline.py`:

### Table Name: `clean_market`

### Columns (Exact Match):

```sql
CREATE TABLE clean_market (
    -- Price Data (OHLCV)
    date DATE NOT NULL,
    open DECIMAL(10, 4),
    high DECIMAL(10, 4),
    low DECIMAL(10, 4),
    close DECIMAL(10, 4),
    "adj close" DECIMAL(10, 4),  -- Note: space in column name
    volume BIGINT,
    
    -- Ticker Info
    ticker TEXT NOT NULL,
    ticker_id INTEGER,
    
    -- Returns & Changes
    daily_return DECIMAL(10, 6),
    volume_change DECIMAL(10, 6),
    
    -- Moving Averages
    ma20 DECIMAL(10, 4),
    ma50 DECIMAL(10, 4),
    close_ma20_ratio DECIMAL(10, 6),
    
    -- Volatility & Momentum
    volatility DECIMAL(10, 6),
    rsi DECIMAL(5, 2),
    
    -- EMA & MACD
    ema12 DECIMAL(10, 4),
    ema26 DECIMAL(10, 4),
    macd DECIMAL(10, 6),
    macd_signal DECIMAL(10, 6),
    
    -- Indexes for performance
    PRIMARY KEY (ticker, date)
);

-- Recommended indexes
CREATE INDEX idx_date ON clean_market(date DESC);
CREATE INDEX idx_ticker ON clean_market(ticker);
CREATE INDEX idx_rsi ON clean_market(rsi);
```

## 🔍 Column Reference

### Price Columns
- `date` - Trading date (YYYY-MM-DD)
- `open`, `high`, `low`, `close` - OHLC prices
- `adj close` - Adjusted closing price (space in name!)
- `volume` - Trading volume

### Identification
- `ticker` - Stock symbol (AAPL, MSFT, etc.)
- `ticker_id` - Numeric encoding (from ticker_encoder.parquet)

### Technical Indicators
- `daily_return` - Daily percentage change
- `volume_change` - Volume percentage change
- `ma20`, `ma50` - Simple Moving Averages (20, 50 days)
- `close_ma20_ratio` - Close price divided by MA20
- `volatility` - 20-day rolling standard deviation
- `rsi` - Relative Strength Index (14-day)
- `ema12`, `ema26` - Exponential Moving Averages
- `macd` - MACD line
- `macd_signal` - MACD signal line

## 📝 Important Notes

1. **Column name with space**: `adj close` has a space - use quotes in SQL
2. **Date format**: YYYY-MM-DD (standard ISO format)
3. **Ticker case**: Usually uppercase (AAPL, MSFT) from yfinance
4. **Missing values**: Pipeline drops NaN rows, so all should be populated
5. **Lookback**: First 50 days per ticker are dropped (warming up indicators)

## 🔧 Supabase Setup

1. Create table using SQL above in Supabase SQL Editor
2. Import your pipeline data:
   - Convert parquet to CSV
   - Use Supabase Table Editor → Import Data

### Convert Pipeline Data to CSV:
```python
import pandas as pd

# Read pipeline output
df = pd.read_parquet('Al-Powered-Stock-ETF-Signal-Generation-Platform-pipeline/data/clean_market.parquet')

# Export for Supabase import
df.to_csv('clean_market.csv', index=False)
print(f"✅ Exported {len(df)} rows to clean_market.csv")
```

## 🚀 Alternative: Direct Python Upload

```python
from supabase import create_client
import pandas as pd

# Connect
url = "YOUR_SUPABASE_URL"
key = "YOUR_SUPABASE_KEY"
supabase = create_client(url, key)

# Read pipeline data
df = pd.read_parquet('Al-Powered-Stock-ETF-Signal-Generation-Platform-pipeline/data/clean_market.parquet')

# Upload in batches (Supabase has row limits per request)
batch_size = 1000
for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i+batch_size].to_dict('records')
    supabase.table('clean_market').insert(batch).execute()
    print(f"✅ Uploaded batch {i//batch_size + 1}")
```

## ✅ Verification Query

After uploading, test with:

```sql
-- Check total records
SELECT COUNT(*) FROM clean_market;

-- Check tickers
SELECT ticker, COUNT(*) as days 
FROM clean_market 
GROUP BY ticker 
ORDER BY days DESC 
LIMIT 10;

-- Check date range
SELECT 
    ticker,
    MIN(date) as first_date,
    MAX(date) as latest_date
FROM clean_market
GROUP BY ticker
LIMIT 5;

-- Sample data
SELECT * FROM clean_market 
WHERE ticker = 'AAPL' 
ORDER BY date DESC 
LIMIT 5;
```
