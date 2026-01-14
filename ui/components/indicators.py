# -*- coding: utf-8 -*-
import streamlit as st
import plotly.graph_objects as go
from contracts.schema import StockData

# --- SHARED STYLING CONSTANTS ---
COLORS = {
    'neon_green': '#00ff7f',
    'neon_red': '#ff6347',
    'neon_blue': '#667eea',
    'neon_purple': '#764ba2',
    'gold': '#ffd700',
    'text': '#ffffff',
    'text_secondary': 'rgba(255, 255, 255, 0.7)',
    'grid': 'rgba(255, 255, 255, 0.06)'
}

def _update_fig_layout(fig, height=300):
    """
    Applies the global glassmorphism theme to Plotly figures.
    """
    fig.update_layout(
        template='plotly_dark',
        height=height,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", color=COLORS['text'], size=12),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='rgba(15, 17, 26, 0.9)',
            bordercolor=COLORS['neon_blue'],
            font_size=12,
            font_family="JetBrains Mono, monospace"
        ),
        showlegend=False
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=COLORS['grid'])
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=COLORS['grid'])
    return fig

def _render_metric_card(label, value, status, color, sub_text=None):
    """
    Helper to render a single glass metric card with dynamic neon styling.
    """
    # Create a subtle background tint based on the status color
    bg_color = f"{color}15" # 15 = roughly 8% opacity hex
    border_color = f"{color}40" # 40 = roughly 25% opacity hex
    
    st.markdown(f"""
        <div class="metric-box" style="
            background: linear-gradient(135deg, {bg_color}, rgba(255,255,255,0.02));
            border: 1px solid {border_color};
            box-shadow: 0 4px 20px {color}10;
            height: 100%;
            display: block;
        ">
            <div class="metric-label" style="color: {COLORS['text_secondary']}; margin-bottom: 8px;">
                {label}
            </div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 28px; font-weight: 800; color: {color}; margin-bottom: 4px;">
                {value}
            </div>
            <div style="font-size: 14px; font-weight: 600; color: {COLORS['text']};">
                {status}
            </div>
            {f'<div style="font-size: 12px; color: {COLORS["text_secondary"]}; margin-top: 4px;">{sub_text}</div>' if sub_text else ''}
        </div>
    """, unsafe_allow_html=True)

def render_indicators_panel(data: StockData):
    """
    Render technical indicators summary panel.
    """
    # Header
    st.markdown("""
        <div class="section-header">
            <div class="section-title">
                🎯 Technical Indicators Summary
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3, gap="medium")
    
    # --- 1. RSI Logic ---
    with col1:
        rsi_val = data.rsi[-1] if data.rsi else 50
        if rsi_val > 70:
            rsi_status, rsi_color = "Overbought", COLORS['neon_red']
        elif rsi_val < 30:
            rsi_status, rsi_color = "Oversold", COLORS['neon_green']
        else:
            rsi_status, rsi_color = "Neutral", COLORS['gold']
            
        _render_metric_card("RSI (14)", f"{rsi_val:.1f}", rsi_status, rsi_color)
    
    # --- 2. MACD Logic ---
    with col2:
        macd = data.macd[-1] if data.macd else 0
        macd_sig = data.macd_signal[-1] if data.macd_signal else 0
        
        if macd > macd_sig:
            macd_status, macd_color = "Bullish Cross", COLORS['neon_green']
        else:
            macd_status, macd_color = "Bearish Cross", COLORS['neon_red']
            
        _render_metric_card("MACD Momentum", f"{macd:.2f}", macd_status, macd_color, sub_text=f"Signal: {macd_sig:.2f}")

    # --- 3. Trend Logic ---
    with col3:
        price = data.current_price
        sma_50 = data.sma_50[-1] if data.sma_50 else price
        
        if price > sma_50:
            trend_status, trend_color = "Uptrend", COLORS['neon_green']
            trend_detail = "Price > SMA 50"
        else:
            trend_status, trend_color = "Downtrend", COLORS['neon_red']
            trend_detail = "Price < SMA 50"
            
        _render_metric_card("Market Trend", trend_status, trend_detail, trend_color)

def render_macd_chart(data: StockData):
    """
    Render MACD indicator chart with glass styling.
    """
    st.markdown("""
        <div class="section-header">
            <div class="section-title">
                📉 MACD Analysis
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    fig = go.Figure()
    
    # MACD Line
    fig.add_trace(go.Scatter(
        x=data.dates, y=data.macd, 
        line=dict(color='#00BFFF', width=2), 
        name='MACD'
    ))
    
    # Signal Line
    fig.add_trace(go.Scatter(
        x=data.dates, y=data.macd_signal, 
        line=dict(color=COLORS['gold'], width=2), 
        name='Signal'
    ))
    
    # Histogram
    hist_colors = [COLORS['neon_green'] if val > 0 else COLORS['neon_red'] for val in data.macd_hist]
    fig.add_trace(go.Bar(
        x=data.dates, y=data.macd_hist, 
        marker_color=hist_colors, 
        name='Histogram',
        opacity=0.6,
        marker_line_width=0
    ))
    
    # Apply unified layout
    _update_fig_layout(fig, height=300)
    
    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})