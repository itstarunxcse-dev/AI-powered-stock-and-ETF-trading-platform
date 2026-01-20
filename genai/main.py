
from fastapi import FastAPI, HTTPException
import requests
import os
from dotenv import load_dotenv
from google import genai
import json
import re

load_dotenv()

# Initialize Gemini client with API key from environment
def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    return genai.Client(api_key=api_key)

app = FastAPI(title="Stock Prediction with News + Indicators via Gemini")

# -----------------------------
# INTERNAL HELPER: CALL INDICATORS API
# -----------------------------
def get_indicators_from_api(ticker: str):
    try:
        response = requests.get("http://127.0.0.1:8000/indicators", params={"ticker": ticker}, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to fetch indicators from API: {e}")

# -----------------------------
# FETCH NEWS FROM ALPHA VANTAGE
# -----------------------------
def fetch_news(ticker: str, max_articles=5):
    alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not alpha_vantage_api_key:
        return []  # Return empty list if API key not configured
    
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={alpha_vantage_api_key}"
    response = requests.get(url)
    data = response.json()

    if "feed" not in data:
        return []  # No news found

    articles = data["feed"][:max_articles]
    news_texts = []
    for article in articles:
        if "title" in article and "summary" in article:
            news_texts.append(article["title"] + ". " + article["summary"])
    return news_texts

# -----------------------------
# CALL GEMINI API FOR PREDICTION
# -----------------------------
def get_gemini_prediction(ticker: str, news_texts: list, indicators: dict):
    """
    Sends news + indicators to Gemini API for analysis.
    Gemini returns signal (BUY/HOLD/SELL) and explanation.
    """
    if not news_texts:
        news_texts = ["No recent news available"]

    # Combine news into one text (truncate if too long)
    news_combined = " ".join(news_texts)[:5000]

    prompt = f"""
Analyze the stock {ticker} based on the following news and technical indicators.

News: {news_combined}

Indicators: {indicators}

Provide a trading signal: BUY, HOLD, or SELL, and a brief explanation.

Response format: {{"signal": "BUY", "explanation": "reason"}}
"""

    try:
        # Call Gemini API for prediction
        client = get_gemini_client()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        
        # Parse the response
        try:
            response_text = response.text
            # Extract JSON from the response
            json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
            else:
                # Fallback if JSON not found
                return {
                    "signal": "HOLD",
                    "explanation": response_text
                }
        except Exception as e:
            return {
                "signal": "HOLD",
                "explanation": f"Error parsing Gemini response: {str(e)}"
            }
    except Exception as api_error:
        # Fallback analysis when API call fails - perform basic technical analysis
        print(f"Gemini API error: {api_error}. Using fallback analysis.")
        
        # Basic technical analysis fallback
        rsi = indicators.get("RSI", 50)
        macd = indicators.get("MACD", 0)
        ma_ratio = indicators.get("Close_MA20_Ratio", 1.0)
        
        signal = "HOLD"
        explanation = "Fallback analysis: "
        
        if rsi > 70:
            signal = "SELL"
            explanation += "RSI indicates overbought condition. "
        elif rsi < 30:
            signal = "BUY"
            explanation += "RSI indicates oversold condition. "
        
        if macd > 0 and ma_ratio > 1.01:
            if signal == "HOLD":
                signal = "BUY"
            explanation += "MACD and moving average trends are positive. "
        elif macd < 0 and ma_ratio < 0.99:
            if signal == "HOLD":
                signal = "SELL"
            explanation += "MACD and moving average trends are negative. "
        
        return {
            "signal": signal,
            "explanation": explanation + "Based on technical indicators analysis."
        }

# -----------------------------
# FASTAPI ENDPOINT
# -----------------------------
@app.get("/predict_stock")
def predict_stock(ticker: str):
    try:
        # 1️⃣ Get indicators from input API
        indicators = get_indicators_from_api(ticker)

        # 2️⃣ Get news from Alpha Vantage
        news_texts = fetch_news(ticker)

        # 3️⃣ Send both to Gemini for analysis
        gemini_result = get_gemini_prediction(ticker, news_texts, indicators)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "ticker": ticker,
        "signal": gemini_result["signal"],
        "explanation": gemini_result["explanation"]
    }
