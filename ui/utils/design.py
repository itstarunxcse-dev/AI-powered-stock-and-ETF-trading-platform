# -*- coding: utf-8 -*-
import streamlit as st

def load_design_system():
    """
    Injects custom CSS for the Glassmorphism UI with animations.
    """
    st.markdown("""
        <style>
        /* Main Background with animated gradient */
        .stApp {
            background: linear-gradient(135deg, #0a0e27 0%, #1a1f2e 50%, #2d1b3d 100%);
            background-size: 400% 400%;
            animation: gradientShift 15s ease infinite;
        }
        
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Glassmorphism Card - Enhanced */
        .glass-card {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.06) 0%, rgba(255, 255, 255, 0.02) 100%);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-radius: 24px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-top: 1px solid rgba(255, 255, 255, 0.15);
            border-left: 1px solid rgba(255, 255, 255, 0.12);
            padding: 30px;
            margin-bottom: 25px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            position: relative;
            overflow: hidden;
        }
        
        .glass-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.05), transparent);
            transition: 0.5s;
        }
        
        .glass-card:hover {
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 15px 50px rgba(0, 0, 0, 0.5);
            transform: translateY(-4px);
        }
        
        .glass-card:hover::before {
            left: 100%;
        }
        
        /* Enhanced Metrics */
        .metric-value {
            font-size: 28px;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .metric-label {
            font-size: 13px;
            color: rgba(255, 255, 255, 0.6);
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        /* Enhanced Signal Badges */
        .signal-buy {
            background: linear-gradient(135deg, rgba(0, 255, 127, 0.2), rgba(0, 200, 100, 0.1));
            color: #00ff7f;
            padding: 10px 20px;
            border-radius: 15px;
            border: 2px solid rgba(0, 255, 127, 0.5);
            font-weight: bold;
            box-shadow: 0 4px 15px rgba(0, 255, 127, 0.2);
            animation: pulseGreen 2s infinite;
        }
        .signal-sell {
            background: linear-gradient(135deg, rgba(255, 99, 71, 0.2), rgba(255, 70, 50, 0.1));
            color: #ff6347;
            padding: 10px 20px;
            border-radius: 15px;
            border: 2px solid rgba(255, 99, 71, 0.5);
            font-weight: bold;
            box-shadow: 0 4px 15px rgba(255, 99, 71, 0.2);
            animation: pulseRed 2s infinite;
        }

        /* Metric Box */
        .metric-box {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 20px;
            text-align: center;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        .metric-box:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
            border: 1px solid rgba(255, 255, 255, 0.2);
            background: rgba(255, 255, 255, 0.08);
        }

        /* Section Header */
        .section-header {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.15), rgba(118, 75, 162, 0.1));
            backdrop-filter: blur(15px);
            border-radius: 20px;
            border: 1px solid rgba(102, 126, 234, 0.3);
            padding: 20px 25px;
            margin-bottom: 25px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            display: flex;
            align-items: center;
        }
        .section-title {
            font-size: 22px;
            font-weight: 700;
            color: white;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        /* Page Header */
        .page-header {
            background: linear-gradient(135deg, rgba(0, 255, 127, 0.1), rgba(0, 200, 100, 0.1));
            backdrop-filter: blur(15px);
            -webkit-backdrop-filter: blur(15px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        }
        .page-title {
            font-size: 42px;
            font-weight: 800;
            background: linear-gradient(135deg, #00ff7f 0%, #00cc66 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 10px;
        }
        .page-subtitle {
            font-size: 16px;
            color: rgba(255, 255, 255, 0.7);
            margin-top: 10px;
            line-height: 1.5;
        }

        /* Info Box */
        .info-box {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            margin-top: 15px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        }

        /* Sidebar Card */
        .sidebar-card {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.15), rgba(118, 75, 162, 0.1));
            backdrop-filter: blur(15px);
            -webkit-backdrop-filter: blur(15px);
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        }
        .sidebar-title {
            font-size: 18px;
            font-weight: 700;
            color: #fff;
            margin-bottom: 15px;
        }

        /* Status Card */
        .status-card {
            background: rgba(0, 255, 127, 0.1);
            border: 1px solid rgba(0, 255, 127, 0.3);
            border-radius: 10px;
            padding: 15px;
            text-align: center;
        }

        /* Prediction Card Specifics */
        .glass-signal-buy {
            background: linear-gradient(135deg, rgba(0, 255, 100, 0.2), rgba(0, 200, 80, 0.1));
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 2px solid rgba(0, 255, 100, 0.3);
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            box-shadow: 0 8px 32px 0 rgba(0, 255, 100, 0.2);
            animation: glow-green 2s ease-in-out infinite alternate;
        }
        
        .glass-signal-sell {
            background: linear-gradient(135deg, rgba(255, 50, 50, 0.2), rgba(200, 0, 0, 0.1));
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 2px solid rgba(255, 50, 50, 0.3);
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            box-shadow: 0 8px 32px 0 rgba(255, 50, 50, 0.2);
            animation: glow-red 2s ease-in-out infinite alternate;
        }
        
        .glass-signal-hold {
            background: linear-gradient(135deg, rgba(255, 200, 0, 0.2), rgba(200, 150, 0, 0.1));
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 2px solid rgba(255, 200, 0, 0.3);
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            box-shadow: 0 8px 32px 0 rgba(255, 200, 0, 0.2);
        }
        
        .glass-insight-box {
            background: rgba(100, 150, 255, 0.08);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            border-radius: 12px;
            border-left: 4px solid rgba(100, 150, 255, 0.5);
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 4px 16px 0 rgba(100, 150, 255, 0.1);
        }
        
        .glass-factor-item {
            background: rgba(255, 255, 255, 0.04);
            backdrop-filter: blur(5px);
            -webkit-backdrop-filter: blur(5px);
            border-radius: 10px;
            border-left: 3px solid rgba(100, 150, 255, 0.4);
            padding: 12px;
            margin: 8px 0;
            transition: all 0.3s ease;
        }
        
        .glass-factor-item:hover {
            background: rgba(255, 255, 255, 0.08);
            border-left-color: rgba(100, 150, 255, 0.8);
            transform: translateX(5px);
        }

        @keyframes glow-green {
            from { box-shadow: 0 0 10px rgba(0, 255, 100, 0.2); }
            to { box-shadow: 0 0 20px rgba(0, 255, 100, 0.5); }
        }
        
        @keyframes glow-red {
            from { box-shadow: 0 0 10px rgba(255, 50, 50, 0.2); }
            to { box-shadow: 0 0 20px rgba(255, 50, 50, 0.5); }
        }

        .glass-metric-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 4px 16px 0 rgba(0, 0, 0, 0.2);
        }
        
        .feature-bar {
            background: rgba(100, 150, 255, 0.3);
            height: 8px;
            border-radius: 4px;
            margin-top: 5px;
            transition: width 0.5s ease;
        }
        
        .model-badge {
            display: inline-block;
            background: rgba(100, 150, 255, 0.2);
            backdrop-filter: blur(5px);
            border: 1px solid rgba(100, 150, 255, 0.4);
            border-radius: 20px;
            padding: 5px 15px;
            margin: 5px;
            font-size: 12px;
            color: rgba(255, 255, 255, 0.9);
        }

        /* Navigation */
        .nav-container {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 15px 30px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        }
        .nav-item {
            display: inline-block;
            padding: 10px 20px;
            margin: 0 10px;
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            color: rgba(255, 255, 255, 0.8);
            text-decoration: none;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        .nav-item:hover {
            background: rgba(100, 150, 255, 0.15);
            border: 1px solid rgba(100, 150, 255, 0.3);
            color: rgba(255, 255, 255, 1);
            transform: translateY(-2px);
        }
        .nav-item.active {
            background: rgba(100, 150, 255, 0.2);
            border: 1px solid rgba(100, 150, 255, 0.4);
            color: #fff;
        }





        .signal-hold {
            background: linear-gradient(135deg, rgba(255, 215, 0, 0.2), rgba(255, 200, 0, 0.1));
            color: #ffd700;
            padding: 10px 20px;
            border-radius: 15px;
            border: 2px solid rgba(255, 215, 0, 0.5);
            font-weight: bold;
            box-shadow: 0 4px 15px rgba(255, 215, 0, 0.2);
        }
        
        /* Pulse Animations */
        @keyframes pulseGreen {
            0%, 100% { 
                box-shadow: 0 4px 15px rgba(0, 255, 127, 0.2);
            }
            50% { 
                box-shadow: 0 4px 25px rgba(0, 255, 127, 0.4);
            }
        }
        
        @keyframes pulseRed {
            0%, 100% { 
                box-shadow: 0 4px 15px rgba(255, 99, 71, 0.2);
            }
            50% { 
                box-shadow: 0 4px 25px rgba(255, 99, 71, 0.4);
            }
        }
        
        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
        }
        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.02);
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        }
        
        /* Button Enhancements */
        .stButton>button {
            background: rgba(100, 150, 255, 0.15) !important;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(100, 150, 255, 0.3) !important;
            border-radius: 12px !important;
            color: white !important;
            font-weight: 600 !important;
            padding: 12px 24px !important;
            transition: all 0.3s ease !important;
        }
        .stButton>button:hover {
            background: rgba(100, 150, 255, 0.25) !important;
            border: 1px solid rgba(100, 150, 255, 0.5) !important;
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(100, 150, 255, 0.3) !important;
        }
        
        /* Input Field Enhancements */
        .stTextInput>div>div>input {
            background: rgba(255, 255, 255, 0.05) !important;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 12px !important;
            color: white !important;
            padding: 12px !important;
        }
        .stTextInput>div>div>input:focus {
            border: 1px solid rgba(100, 150, 255, 0.5) !important;
            box-shadow: 0 0 20px rgba(100, 150, 255, 0.2) !important;
        }
        
        /* Selectbox Enhancements */
        .stSelectbox>div>div>div {
            background: rgba(255, 255, 255, 0.05) !important;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 12px !important;
        }

        /* --- RESPONSIVE DESIGN --- */
        @media screen and (max-width: 768px) {
            /* Adjust Main Container */
            .stMainBlockContainer {
                padding-left: 0.5rem !important;
                padding-right: 0.5rem !important;
                padding-top: 1rem !important;
            }
            
            /* Typography Scaling */
            .page-title { font-size: 28px !important; }
            .section-title { font-size: 18px !important; }
            .metric-value { font-size: 24px !important; }
            
            /* Card Adjustments */
            .glass-card, .page-header, .sidebar-card, .nav-container {
                padding: 15px !important;
                margin-bottom: 15px !important;
                border-radius: 16px !important;
            }
            
            /* Full Width Elements on Mobile */
            .stButton>button {
                width: 100% !important;
                padding: 12px 15px !important;
            }
            
            /* Metrics Stacking */
            .metric-box {
                flex-direction: column;
                padding: 12px;
            }
            
            /* Nav Items */
            .nav-item {
                display: block;
                margin: 5px 0;
                text-align: center;
                width: 100%;
            }
        }
        </style>
    """, unsafe_allow_html=True)

def card_container(key=None):
    """
    Returns a container with the glassmorphism class.
    """
    return st.container()

def glass_card(title: str, icon: str = ""):
    """Start a glassmorphism card with a title"""
    st.markdown(f"""
    <div class="glass-card">
        <div class="section-title">
            <span>{icon}</span> {title}
        </div>
    """, unsafe_allow_html=True)

def glass_card_end():
    """End a glassmorphism card"""
    st.markdown("</div>", unsafe_allow_html=True)
