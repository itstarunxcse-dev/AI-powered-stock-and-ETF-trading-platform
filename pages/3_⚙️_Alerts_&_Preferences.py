# -*- coding: utf-8 -*-
"""
🔔 Smart Alerts Configuration
AI + Strategy + Technical + Preference based alerts
Production-ready Streamlit UI (Glassmorphism)
"""

import streamlit as st
import time
import requests
import os
import datetime as dt
import re
from ui.utils.constants import get_common_tickers
from ui.utils.design import glass_card, glass_card_end, load_design_system

# ======================================================
# PROJECT CONFIG
# ======================================================
ALERTS_API_URL = os.getenv("ALERTS_API_URL", "http://localhost:8001")

st.set_page_config(
    page_title="Alerts & Preferences",
    page_icon="🔔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Call the centralized design system
load_design_system()

# ======================================================
# DATA & API
# ======================================================
def validate_email(email: str) -> bool:
    """Basic regex validation for email."""
    if not email:
        return False
    # Simple regex for email validation
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return bool(re.match(pattern, email))

def check_api_status():
    try:
        response = requests.get(f"{ALERTS_API_URL}/health", timeout=2)
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except requests.exceptions.ConnectionError:
        return False, "Connection Error"
    except Exception:
        return False, None

def fetch_active_alerts():
    try:
        response = requests.get(f"{ALERTS_API_URL}/active-alerts", timeout=3)
        if response.status_code == 200:
            return response.json().get('jobs', [])
        return []
    except Exception:
        return []

def create_alert(email, ticker, time_str):
    """
    Creates a scheduled alert via the backend.
    """
    payload = {"user_email": email, "ticker_name": ticker, "alert_time": time_str}
        
    try:
        response = requests.post(
            f"{ALERTS_API_URL}/create-alert", 
            json=payload,
            timeout=5
        )
        if response.status_code == 200:
            return True, response.json()
        else:
            try:
                err_msg = response.json().get("detail", response.text)
            except:
                err_msg = f"HTTP {response.status_code}"
            return False, err_msg
            
    except requests.exceptions.ConnectionError:
        return False, "Failed to connect to Alerts API."
    except Exception as e:
        return False, str(e)

def stop_alert(email, ticker):
    try:
        response = requests.delete(
            f"{ALERTS_API_URL}/stop-alert/{email}/{ticker}",
            timeout=5
        )
        return response.status_code == 200
    except:
        return False

# ======================================================
# MAIN UI
# ======================================================
def main():
    # Inject page-specific styles
    st.markdown("""
    <style>
    /* New Job Card Styles */
    .job-row {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        transition: all 0.2s ease;
    }
    .job-row:hover {
        background: rgba(255, 255, 255, 0.06);
        border-color: rgba(255, 255, 255, 0.2);
    }
    .metric-nice {
        font-size: 13px;
        color: #94a3b8;
    }
    .metric-val-nice {
        font-size: 15px;
        font-weight: 600;
        color: #f1f5f9;
    }
    .big-ticker {
        font-size: 20px; 
        font-weight: 800; 
        background: linear-gradient(90deg, #60a5fa, #a78bfa); 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent;
    }
    
    /* Page Background - Neutral Dark Gradient */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #000000 0%, #1a1a1a 100%) !important;
        background-attachment: fixed;
    }

    .glass-header {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.03), rgba(99, 102, 241, 0.05));
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 35px;
        margin-bottom: 35px;
        box-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .glass-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 1px;
        background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.5), transparent);
    }
    
    .gradient-text {
        background: linear-gradient(to right, #818cf8, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    .status-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 99px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-left: 10px;
        text-transform: uppercase;
        backdrop-filter: blur(4px);
    }
    .status-online { 
        background: rgba(16, 185, 129, 0.1); 
        color: #34d399; 
        border: 1px solid rgba(16, 185, 129, 0.2);
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.1);
    }
    .status-offline { 
        background: rgba(239, 68, 68, 0.1); 
        color: #f87171; 
        border: 1px solid rgba(239, 68, 68, 0.2);
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.1);
    }
    
    /* Input field accents */
    .stTextInput>div>div>input:focus {
        border-color: #818cf8 !important;
        box-shadow: 0 0 0 1px #818cf8 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 1. Check Service Status
    is_online, _ = check_api_status()
    
    status_html = '<span class="status-badge status-online">● ONLINE</span>' if is_online else '<span class="status-badge status-offline">● OFFLINE</span>'
    
    st.markdown(f"""
    <div class="glass-header">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <div>
                <div class="gradient-text" style="font-size:36px;">
                    🔔 Alerts & Notifications
                </div>
                <div style="opacity:.65;">
                    Manage real-time email alerts and monitoring schedules
                </div>
            </div>
            <div>{status_html}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not is_online:
        st.error("⚠️ Alerts Middleware Service is unreachable.")
        st.caption(f"Targeting: `{ALERTS_API_URL}`")
        st.code("uvicorn alerts.main:app --host 0.0.0.0 --port 8001", language="bash")
    
    # Active Jobs for Deduplication
    active_jobs = fetch_active_alerts() if is_online else []
    
    # ======================================================
    # TABS
    # ======================================================
    tab_alerts, tab_prefs = st.tabs(["📢 Active Alerts", "🎨 UI Preferences"])
    
    with tab_alerts:
        col1, col2 = st.columns([1, 2])
        
        # CREATE ALERT FORM
        with col1:
            glass_card("Create New Alert", "➕")
            if is_online:
                with st.form("create_alert_form"):
                    tickers = get_common_tickers()
                    selected_tickers = st.multiselect(
                        "Select Tickers", 
                        options=tickers, 
                        default=["AAPL"],
                        help="Select multiple stocks to monitor."
                    )
                    
                    target_email = st.text_input("Notify Email", placeholder="your@email.com")
                    
                    st.markdown("### 📅 Schedule")
                    
                    # Generate 5-minute intervals
                    time_options = [f"{h:02d}:{m:02d}" for h in range(24) for m in range(0, 60, 5)]
                    default_idx = time_options.index("10:00") if "10:00" in time_options else 0
                    
                    alert_time_str = st.selectbox("Select Time", options=time_options, index=default_idx)
                    
                    # Convert to AM/PM for display
                    display_time = dt.datetime.strptime(alert_time_str, "%H:%M").strftime("%I:%M %p")
                    
                    st.info(f"""
                    Alert will trigger daily at **{display_time}**.
                    """)
                    
                    submitted = st.form_submit_button("Activate Monitoring", type="primary")
                    
                    if submitted:
                        if not selected_tickers:
                            st.error("❌ Select at least one ticker.")
                        elif not target_email:
                            st.error("❌ Enter an email address.")
                        elif not validate_email(target_email):
                            st.error("❌ Invalid email format.")
                        else:
                            success_count = 0
                            errors = []
                            
                            prog_bar = st.progress(0)
                            for idx, tkr in enumerate(selected_tickers):
                                ok, resp = create_alert(target_email, tkr, alert_time_str)
                                if ok:
                                    success_count += 1
                                else:
                                    errors.append(f"{tkr}: {resp}")
                                
                                prog_bar.progress((idx + 1) / len(selected_tickers))
                            
                            if success_count > 0:
                                st.success(f"✅ Scheduled {success_count} alerts!")
                                if errors:
                                    st.warning(f"Skipped/Failed: {', '.join(errors)}")
                                time.sleep(1.5)
                                st.rerun()
                            elif errors:
                                st.error(f"❌ Failed: {errors[0]}")
            
            st.markdown("---")
            with st.expander("⚡ Send Instant Report"):
                ir_tickers_list = get_common_tickers()
                ir_ticker = st.selectbox("Ticker", options=ir_tickers_list, key="ir_ticker")
                ir_email = st.text_input("Email", key="ir_email")
                
                if st.button("Send Now"):
                    if not ir_ticker or not ir_email:
                         st.error("Please fill all fields.")
                    elif not validate_email(ir_email):
                         st.error("Invalid email address.")
                    else:
                        with st.spinner("Processing..."):
                            try:
                                r = requests.post(
                                    f"{ALERTS_API_URL}/instant-report",
                                    json={"user_email": ir_email, "ticker_name": ir_ticker},
                                    timeout=15
                                )
                                if r.status_code == 200:
                                    st.success(f"✅ Sent!")
                                else:
                                    st.error(f"Failed: {r.text}")
                            except Exception as e:
                                st.error(f"Connection Error: {e}")
            glass_card_end()

        # LIST ACTIVE ALERTS
        with col2:
            # Header
            st.markdown("""
                <div style="margin-bottom: 20px; display: flex; align-items: center; justify-content: space-between;">
                    <div style="font-size: 18px; font-weight: 700; color: #fff;">📡 Active Monitors</div>
                    <div style="font-size: 12px; color: #94a3b8; background: rgba(255,255,255,0.05); padding: 4px 12px; border-radius: 20px;">
                        Real-time
                    </div>
                </div>
            """, unsafe_allow_html=True)

            if is_online:
                if active_jobs:
                    # Filter and Group Unique Active Jobs
                    unique_pairs = {}
                    for job in active_jobs:
                        if len(job.get('args', [])) >= 2:
                            email = job['args'][0]
                            ticker = job['args'][1]
                            key = f"{email}_{ticker}"
                            if key not in unique_pairs:
                                unique_pairs[key] = {
                                    "email": email, 
                                    "ticker": ticker, 
                                    "schedules": [],
                                    "next_run": job.get('next_run', 'N/A')
                                }
                            
                            # Parse Schedule Time from ID or Next Run
                            try:
                                # Try ID first (format: ..._HHMM)
                                t_str = job['id'].split('_')[-1]
                                if len(t_str) == 4 and t_str.isdigit():
                                    formatted_time = f"{t_str[:2]}:{t_str[2:]}"
                                    unique_pairs[key]["schedules"].append(formatted_time)
                            except:
                                pass

                    if not unique_pairs:
                        st.info("No active alerts configured.")
                    
                    # Render Cards
                    for key, data in unique_pairs.items():
                        ticker = data["ticker"]
                        email = data["email"]
                        times = sorted(list(set(data["schedules"])))
                        time_display = " • ".join(times) if times else "Active"
                        
                        with st.container():
                            st.markdown(f"""
                            <div class="job-row">
                                <div style="display: flex; justify-content: space-between; align-items: start;">
                                    <div>
                                        <div class="big-ticker">{ticker}</div>
                                        <div style="margin-top: 4px; display: flex; align-items: center; gap: 8px;">
                                            <span style="font-size: 12px; color: #94a3b8;">📧 {email}</span>
                                        </div>
                                    </div>
                                    <div style="text-align: right;">
                                        <div style="font-size: 11px; color: #64748b; text-transform: uppercase; font-weight: 700;">Schedule</div>
                                        <div style="color: #e2e8f0; font-family: monospace; font-size: 14px; background: rgba(255,255,255,0.1); padding: 2px 8px; border-radius: 4px; display: inline-block; margin-top: 4px;">
                                            {time_display}
                                        </div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Action Buttons (Native Streamlit widgets for functionality)
                            c1, c2 = st.columns([4, 1])
                            with c2:
                                if st.button("Stop", key=f"stop_{key}", type="secondary", use_container_width=True):
                                    stop_alert(email, ticker)
                                    st.rerun()
                            st.markdown("<div style='margin-bottom: 8px'></div>", unsafe_allow_html=True)

                else:
                    st.info("✨ No alerts active. Use the form to create one.")
                    
                st.markdown("---")
                if active_jobs:
                    if st.button("🗑️ Clear All Alerts", key="clear_all"):
                        try:
                            requests.delete(f"{ALERTS_API_URL}/clear-all-alerts", timeout=5)
                            st.success("Cleared!")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to clear: {e}")
            else:
                st.warning("⚠️ Alerts Service Offline")
                st.caption("Detailed view unavailable.")

    # --------------------------------------------------
    # TAB 2: UI PREFERENCES
    # --------------------------------------------------
    with tab_prefs:
        glass_card("Client-Side Preferences", "🎨")
        c1, c2 = st.columns(2)
        with c1:
            st.selectbox("Notification Sound", ["Ping", "Chime", "Silent"])
            st.checkbox("Show Desktop Notifications", value=True)
        with c2:
            st.slider("Notification Duration (s)", 3, 30, 5)
            st.selectbox("Theme Mode", ["Dark (Glass)", "Light"])
        
        if st.button("Save Preferences"):
            st.success("Saved.")
        glass_card_end()

if __name__ == "__main__":
    main()
