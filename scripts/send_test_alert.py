import smtplib
from email.mime.text import MIMEText
from datetime import datetime
import requests
import sys

# CONFIGURATION
EMAIL_SENDER = "tarunsivasai03@gmail.com"
EMAIL_PASSWORD = "odsi iyiq ywar zvba"
TARGET_EMAIL = "alonewalker07827@gmail.com"
TICKER = "AMZN"

def fetch_live_signal(ticker):
    """Try to get real data from the running ML API"""
    try:
        print(f"Connecting to ML API for {ticker}...")
        url = "http://localhost:8000/api/v1/ml/signal/live"
        response = requests.post(url, json={"ticker": ticker}, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API returned status {response.status_code}")
    except Exception as e:
        print(f"Could not connect to ML API: {e}")
    return None

def send_alert(data):
    try:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        signal = data.get('signal', 'TEST')
        price = data.get('current_price', 0.0)
        confidence = data.get('confidence', 0.0)
        
        # Format confidence
        if isinstance(confidence, (int, float)):
            conf_str = f"{confidence:.2f}%"
        else:
            conf_str = str(confidence)

        subject = f"🚨 TEST ALERT: {signal} {TICKER}"
        body = f"""
        Subject: {subject}

        TRADING ALERT SYSTEM (MANUAL TEST)
        ---------------------------------
        Time:       {current_time}
        Ticker:     {TICKER}
        Signal:     {signal}
        Price:      {price}
        Confidence: {conf_str}
        ---------------------------------
        This is a test alert requested manually.
        """

        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = EMAIL_SENDER
        msg['To'] = TARGET_EMAIL

        print(f"Sending email to {TARGET_EMAIL}...")
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, TARGET_EMAIL, msg.as_string())

        print("✅ Email sent successfully!")
        return True

    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        return False

def main():
    # 1. Try to fetch real data
    data = fetch_live_signal(TICKER)
    
    # 2. If no real data, use dummy data (since this is a requested 'test')
    if not data:
        print("Using dummy data for test...")
        data = {
            "signal": "TEST_BUY",
            "current_price": 185.50,
            "confidence": 99.9
        }
    
    # 3. Send the alert
    send_alert(data)

if __name__ == "__main__":
    main()
