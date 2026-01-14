import requests
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
import sys

# CONFIGURATION
EMAIL_SENDER = "tarunsivasai03@gmail.com"
EMAIL_PASSWORD = "odsi iyiq ywar zvba"
API_URL = "http://localhost:8000/api/v1/ml/signal/live"

def get_stock_details(ticker):
    print(f"Fetching live data for {ticker}...")
    try:
        response = requests.post(API_URL, json={"ticker": ticker}, timeout=15)
        if response.status_code == 200:
            return response.json()
        print(f"API Error: {response.text}")
        return None
    except Exception as e:
        print(f"Connection Error: {e}")
        return None

def send_instant_email(recipient, ticker, data):
    print(f"Sending email to {recipient}...")
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    signal = data.get('signal', 'N/A')
    price = data.get('current_price', 0.0)
    conf = data.get('confidence', 0.0)
    
    subject = f"⚡ INSTANT REPORT: {ticker} ({signal})"
    
    body = f"""
    Subject: {subject}

    INSTANT STOCK REPORT
    ============================
    Ticker:     {ticker}
    Time:       {current_time}
    ----------------------------
    Signal:     {signal}
    Price:      ₹{price}
    Confidence: {conf:.2f}%
    ----------------------------
    
    Expected Return (Model): {data.get('expected_return', 0):.4f}%
    
    This report was generated on demand.
    """

    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = EMAIL_SENDER
        msg['To'] = recipient

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, recipient, msg.as_string())
        
        print("✅ Email sent successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        return False

if __name__ == "__main__":
    print("--- INSTANT STOCK REPORTER ---")
    
    # Get inputs interactively if not passed as args
    if len(sys.argv) >= 3:
        ticker = sys.argv[1]
        email = sys.argv[2]
    else:
        ticker = input("Enter Ticker (e.g. RELIANCE.NS): ").strip().upper()
        email = input("Enter Recipient Email: ").strip()
        
    if ticker and email:
        data = get_stock_details(ticker)
        if data:
            send_instant_email(email, ticker, data)
    else:
        print("Invalid input.")
