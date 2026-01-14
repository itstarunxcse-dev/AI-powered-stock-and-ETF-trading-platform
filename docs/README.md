# AI Stock Trading Platform

A comprehensive stock trading platform with AI-powered predictions, backtesting capabilities, interactive visualizations, and integrated data pipeline.

## 📊 Features

- **Real-time Stock Analysis** - Live data fetching with technical indicators
- **ML Trading Signals** - Pre-trained models (Random Forest + XGBoost) with 81.82% win rate
- **Advanced Backtesting** - Vectorbt-powered strategy analysis with comprehensive metrics
- **Interactive Dashboard** - Streamlit-based UI with charts and visualizations
- **REST API** - FastAPI service for programmatic access
- **Pipeline Integration** - 51 stocks (26 US + 25 Indian) with 8+ years of data
- **Multiple Data Sources** - Pipeline (Parquet), CSV, or live yfinance data

## 🚀 Quick Start

### Easy Start (Recommended)

**Windows:**
```bash
# Double-click start.bat or run:
start.bat
```

**Linux/Mac:**
```bash
chmod +x start.sh
./start.sh
```

The start script will:
- ✅ Check if pipeline data exists
- ✅ Offer to generate data if needed
- ✅ Let you choose: Dashboard, API, Demo, or Test
- ✅ Launch your selection automatically

### Manual Installation

```bash
# Clone the repository
cd ProjD-main

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

#### Option 1: Streamlit Dashboard (Interactive UI)
```bash
streamlit run 0_Overview.py
```
Access at: http://localhost:8501

#### Option 2: FastAPI Service (REST API)
```bash
python scripts/run_api.py
```
Access at: http://localhost:8000  
Docs: http://localhost:8000/docs

#### Option 3: Run Demo Scripts
```bash
# Pipeline integration demo (NEW!)
python scripts/demo_pipeline_integration.py

# ML signals demonstration
python scripts/demo_ml_signals.py

# Integration test
python tests/test_integration.py

# API test (start API first)
python tests/test_api.py
```

### Pipeline Data Setup (Optional but Recommended)

```bash
# Generate pipeline data (8+ years, 51 stocks)
cd Al-Powered-Stock-ETF-Signal-Generation-Platform-pipeline
python data_pipeline.py

# Configure to use pipeline
# Edit data/pipeline_config.py: DATA_SOURCE = 'pipeline'

# Test the integration
cd ..
python scripts/demo_pipeline_integration.py
```

## 📁 Project Structure

```
ProjD-main/
├── 0_Overview.py              # Main Streamlit app entry point
├── main.py                    # FastAPI application
├── requirements.txt           # Python dependencies
├── ml_trading_signals.csv     # ML predictions dataset (7,340 rows)
│
├── app/                       # FastAPI application modules
│   ├── __init__.py
│   ├── data_loader.py         # Data loading with pipeline support
│   ├── engine.py              # BacktestEngine wrapper
│   └── schemas.py             # Pydantic response models
│
├── backtesting/               # Backtesting engine
│   └── engine.py              # Vectorbt-based backtest engine
│
├── contracts/                 # Data contracts & schemas
│   └── schema.py              # Pydantic models for data exchange
│
├── data/                      # Data fetching & processing
│   ├── fetcher.py             # Live data with pipeline support
│   ├── ml_signals_loader.py   # ML dataset loader
│   ├── pipeline_config.py     # Pipeline configuration (NEW!)
│   └── pipeline_adapter.py    # Pipeline data access layer (NEW!)
│
├── ml/                        # ML prediction models
│   └── predictor.py           # ML signal generation
│
├── pages/                     # Streamlit multi-page app
│   ├── 1_📊_AI_Signals.py     # Trading signals page
│   ├── 2_📈_Strategy_Analysis.py  # Backtesting page
│   └── 3_⚙️_Alerts_&_Preferences.py  # Settings page
│
├── ui/                        # UI components
│   ├── components/            # Reusable UI components
│   └── utils/                 # UI utilities
│
├── scripts/                   # Utility scripts
│   ├── demo_pipeline_integration.py  # Pipeline demo (NEW!)
│   ├── demo_ml_signals.py     # ML signals demo
│   └── run_api.py             # API server launcher
│
├── docs/                      # Documentation
│   ├── PIPELINE_INTEGRATION_GUIDE.md  # Pipeline guide (NEW!)
│   ├── ML_SIGNALS_GUIDE.md    # ML signals documentation
│   └── API_README.md          # API documentation
│
└── Al-Powered-Stock-ETF-Signal-Generation-Platform-pipeline/  # Data pipeline (NEW!)
    ├── data_pipeline.py       # ETL pipeline for 51 stocks
    ├── data_pipeline_api.py   # Pipeline trigger API
    ├── input_api.py           # Live indicators API
    └── pipeline/data/         # Generated Parquet files
```
│   ├── run_api.py             # API server launcher
│   └── demo_ml_signals.py     # ML signals demo
│
├── tests/                     # Test scripts
│   ├── test_api.py            # API tests
│   └── test_integration.py    # Integration tests
│
└── docs/                      # Documentation
    ├── API_README.md          # API documentation
    ├── ML_SIGNALS_GUIDE.md    # ML dataset guide
    └── INTEGRATION_SUMMARY.md # Integration details
    └── pipeline/data/         # Generated Parquet files
```

## 📊 Dataset Information

### Pipeline Data (Recommended - NEW!)
- **Source**: Al-Powered-Stock-ETF-Signal-Generation-Platform-pipeline
- **Stocks**: 51 tickers (26 US + 25 Indian markets)
- **Period**: 2018-01-01 to present (8+ years)
- **Rows**: ~100,000+ data points
- **Format**: Parquet (fast columnar storage)
- **Indicators**: MA20, MA50, RSI, MACD, EMA12/26, Volatility
- **Generation**: Run `data_pipeline.py` to create/update
- **Tickers**: AAPL, MSFT, GOOGL, NVDA, TSLA, AMZN, RELIANCE.NS, TCS.NS, and more

### ML Signals Dataset (CSV Fallback)
- **Total Rows**: 7,340
- **Date Range**: Feb 2020 - Dec 2025 (5+ years)
- **Tickers**: AAPL, AMZN, GOOGL, MSFT, TSLA
- **Signals**: Buy (7.33%), Hold (85.64%), Sell (7.03%)

### Performance (AAPL Example)
- **ML Strategy Return**: 467.94%
- **Market Return**: 314.08%
- **Outperformance**: +153.86%
- **Win Rate**: 81.82%
- **Sharpe Ratio**: 1.89
- **Max Drawdown**: 20.38%

## 🔧 Key Technologies

- **Backend**: FastAPI, Python 3.13
- **Frontend**: Streamlit
- **Data**: pandas, numpy, yfinance, pyarrow (Parquet)
- **Backtesting**: vectorbt
- **ML Models**: Random Forest, XGBoost (pre-trained)
- **Visualization**: Plotly, Matplotlib
- **API Docs**: OpenAPI/Swagger
- **Pipeline**: Data ETL with technical indicators

## 📚 API Endpoints

**Updated:** January 9, 2026 by Aman (DE Team)  
**Base URL:** `http://127.0.0.1:8000`  
**Full Documentation:** See [API_ENDPOINTS.md](../API_ENDPOINTS.md) in project root

### Quick Reference

- `GET /health` - System health check
- `POST /run-pipeline` - Trigger data pipeline
- `GET /supabase/recent/{ticker}?days=30` - Recent stock data
- `GET /supabase/ticker/{ticker}?start_date=2024-01-01&limit=100` - Ticker data with range
- `GET /supabase/latest?limit=10` - Latest market overview
- `GET /supabase/top-performers?top_n=10` - Top performers
- `GET /supabase/stats/{ticker}?start_date=2024-01-01` - Statistical analysis
- `GET /supabase/rsi-search?min_rsi=0&max_rsi=30` - RSI-based search

## 📚 API Endpoints (Legacy)

### REST API
- `GET /` - Health check
- `POST /api/v1/backtest/run` - Run backtest (supports pipeline & CSV)
- `GET /api/v1/health` - Service status

### Example Usage

#### Using Pipeline Data (Recommended)
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/backtest/run",
    params={"ticker": "AAPL", "use_pipeline": True}
)

results = response.json()
print(f"Return: {results['ml_metrics']['total_return']:.2f}%")
```

#### Using CSV Fallback
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/backtest/run",
    params={"csv_path": "ml_trading_signals.csv", "use_pipeline": False}
)

results = response.json()
print(f"Return: {results['ml_metrics']['total_return_pct']:.2f}%")
```

## 🔌 Pipeline Integration

### Quick Start
```bash
# 1. Generate pipeline data
cd Al-Powered-Stock-ETF-Signal-Generation-Platform-pipeline
python data_pipeline.py

# 2. Configure to use pipeline
cd ..
# Edit data/pipeline_config.py: DATA_SOURCE = 'pipeline'

# 3. Run demo
python scripts/demo_pipeline_integration.py

# 4. Use in your app
streamlit run 0_Overview.py
```

### Usage Examples

#### Load Pipeline Data
```python
from data.pipeline_adapter import get_pipeline_data, get_available_tickers

# Get available tickers
tickers = get_available_tickers()  # Returns all 51 tickers

# Load single ticker
df = get_pipeline_data(ticker='AAPL')

# Load multiple tickers
df = get_pipeline_data(tickers=['AAPL', 'MSFT', 'GOOGL'])
```

#### Use in Backtesting
```python
from app.data_loader import load_historical_data
from app.engine import BacktestEngine

# Load from pipeline (automatically based on config)
df = load_historical_data(ticker='AAPL', use_pipeline=True)

# Run backtest
engine = BacktestEngine(df)
results = engine.run_ml()
print(f"Return: {results['ml_metrics']['total_return']:.2f}%")
```

#### Check Data Info
```python
from data.pipeline_adapter import PipelineAdapter

adapter = PipelineAdapter()
info = adapter.get_data_info()

print(f"Total rows: {info['total_rows']:,}")
print(f"Tickers: {info['ticker_count']}")
print(f"Date range: {info['date_range']}")
```

### Documentation
See [Pipeline Integration Guide](docs/PIPELINE_INTEGRATION_GUIDE.md) for:
- Complete setup instructions
- API usage examples
- Configuration options
- Troubleshooting
- Performance tips

## 🧪 Testing

```bash
# Test pipeline integration
python scripts/demo_pipeline_integration.py

# Run all tests
python tests/test_integration.py
python tests/test_api.py

# Run demo
python scripts/demo_ml_signals.py
```

## 📖 Documentation

Detailed documentation available in `/docs`:
- [Pipeline Integration Guide](docs/PIPELINE_INTEGRATION_GUIDE.md) - NEW! Pipeline setup & usage
- [API Documentation](docs/API_README.md)
- [ML Signals Guide](docs/ML_SIGNALS_GUIDE.md)
- [Integration Summary](docs/INTEGRATION_SUMMARY.md)

## 🎯 Usage Examples

### Load ML Signals Data (CSV)
```python
from data.fetcher import DataEngine

df = DataEngine.prepare_ml_data_for_backtest(ticker='AAPL')
```

### Load Pipeline Data (Recommended)
```python
from data.pipeline_adapter import get_pipeline_data

df = get_pipeline_data(ticker='AAPL')
```

### Run Backtest
```python
from backtesting.engine import BacktestEngine

engine = BacktestEngine(df)
market = engine.run_market()
ml = engine.run_ml()

print(f"ML Return: {ml['ml_metrics']['total_return_pct']:.2f}%")
```

### Use API
```python
import requests

response = requests.post("http://localhost:8000/api/v1/backtest/run")
results = response.json()
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## 📝 License

This project is for educational purposes.

## 🔗 Links

- [Streamlit Documentation](https://docs.streamlit.io)
- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [Vectorbt Documentation](https://vectorbt.dev)

---

**Note**: Make sure to have Python 3.8+ installed and all dependencies from requirements.txt.
