# -*- coding: utf-8 -*-
import streamlit as st
from datetime import datetime

def render_navigation():
    """Renders a modern navigation bar with glassmorphism"""
    st.markdown("""
        <div class="nav-container">
            <span class="nav-item active">📊 Dashboard</span>
            <span class="nav-item">📈 Analysis</span>
            <span class="nav-item">🎯 Signals</span>
            <span class="nav-item">📉 Backtest</span>
            <span class="nav-item">⚙️ Settings</span>
        </div>
    """, unsafe_allow_html=True)

def render_header():
    # Hero section with glassmorphism
    st.markdown("""
        <style>
        .hero-container {
            background: linear-gradient(135deg, rgba(100, 150, 255, 0.1), rgba(150, 100, 255, 0.1));
            backdrop-filter: blur(15px);
            -webkit-backdrop-filter: blur(15px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        }
        .hero-title {
            font-size: 48px;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        .hero-subtitle {
            font-size: 18px;
            color: rgba(255, 255, 255, 0.7);
            margin-bottom: 0;
        }
        .status-badge {
            display: inline-block;
            padding: 8px 16px;
            background: rgba(0, 255, 127, 0.15);
            border: 1px solid rgba(0, 255, 127, 0.3);
            border-radius: 20px;
            color: #00ff7f;
            font-weight: 600;
            font-size: 14px;
            animation: pulse 2s ease-in-out infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Navigation bar
    render_navigation()
    
    # Hero header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
            <div class="hero-container">
                <div class="hero-title">🤖 AI Stock Signal Dashboard</div>
                <div class="hero-subtitle">📊 Institutional-grade analysis powered by Generative AI & Machine Learning</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style="background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px); 
                 border-radius: 15px; border: 1px solid rgba(255, 255, 255, 0.1); 
                 padding: 20px; text-align: center; margin-top: 20px;">
                <div class="status-badge">🟢 MARKET LIVE</div>
                <div style="margin-top: 15px; font-size: 12px; opacity: 0.6;">Last Updated</div>
                <div style="font-size: 14px; font-weight: 600; color: #6496ff;">""" + datetime.now().strftime('%H:%M:%S') + """</div>
            </div>
        """, unsafe_allow_html=True)
