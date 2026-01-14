# -*- coding: utf-8 -*-
import streamlit as st
from contracts.schema import BacktestMetrics
import pandas as pd
import json

def render_strategy_config(metrics: BacktestMetrics):
    """
    Display Strategy Configuration (Static Info)
    """
    st.markdown("""
        <div class="section-header">
            <div class="section-title">
                ⚙️ Strategy Configuration
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Strategy Name</div>
                <div style="font-size: 20px; font-weight: 700; color: #667eea; margin-top: 8px;">
                    {metrics.config.strategy_name}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="metric-box" style="margin-top: 15px;">
                <div class="metric-label">Initial Capital</div>
                <div style="font-size: 20px; font-weight: 700; color: #00ff7f; margin-top: 8px;">
                    ₹{metrics.config.initial_capital:,.0f}
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Commission</div>
                <div style="font-size: 20px; font-weight: 700; color: #ffd700; margin-top: 8px;">
                    {metrics.config.commission * 100:.2f}% per trade
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="metric-box" style="margin-top: 15px;">
                <div class="metric-label">Trade Execution</div>
                <div style="font-size: 16px; font-weight: 600; color: rgba(255, 255, 255, 0.9); margin-top: 8px;">
                    {"✅ Trade on Close" if metrics.config.trade_on_close else "❌ Trade on Open"}
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Position Type</div>
                <div style="font-size: 18px; font-weight: 700; color: #ff6347; margin-top: 8px;">
                    {metrics.config.position_type}
                </div>
            </div>
        """, unsafe_allow_html=True)

def render_signal_logic(metrics: BacktestMetrics):
    """
    Display Signal Execution Logic
    """
    st.markdown("""
        <div class="section-header" style="background: linear-gradient(135deg, rgba(0, 255, 127, 0.1), rgba(0, 200, 100, 0.05)); border-color: rgba(0, 255, 127, 0.2);">
            <div class="section-title">
                🎯 Signal Execution Logic
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
        <div class="glass-card" style="padding: 20px;">
            <div style="margin-bottom: 15px;">
                <div style="font-size: 14px; color: rgba(255, 255, 255, 0.7); margin-bottom: 5px;">Entry Rule:</div>
                <div style="font-size: 16px; color: #00ff7f; font-weight: 600; background: rgba(0, 255, 127, 0.1); padding: 10px; border-radius: 8px; border: 1px solid rgba(0, 255, 127, 0.2);">
                    {metrics.entry_rule}
                </div>
            </div>
            <div style="margin-bottom: 15px;">
                <div style="font-size: 14px; color: rgba(255, 255, 255, 0.7); margin-bottom: 5px;">Exit Rule:</div>
                <div style="font-size: 16px; color: #ff6347; font-weight: 600; background: rgba(255, 99, 71, 0.1); padding: 10px; border-radius: 8px; border: 1px solid rgba(255, 99, 71, 0.2);">
                    {metrics.exit_rule}
                </div>
            </div>
            <div>
                <div style="font-size: 14px; color: rgba(255, 255, 255, 0.7); margin-bottom: 5px;">Position Strategy:</div>
                <div style="font-size: 16px; color: #ffd700; font-weight: 600; background: rgba(255, 215, 0, 0.1); padding: 10px; border-radius: 8px; border: 1px solid rgba(255, 215, 0, 0.2);">
                    {metrics.position_strategy}
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_metrics(metrics):
    """
    Display Backtest Performance Summary - MANDATORY METRICS
    """
    # Helper extractors
    ml = metrics.ml_metrics
    trd = metrics.trading_metrics
    
    st.markdown("""
        <div class="section-header">
            <div class="section-title">
                📊 Performance Summary - Mandatory Metrics
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # First Row - Core Performance Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div class="metric-box" style="background: linear-gradient(135deg, rgba(0, 255, 127, 0.1), rgba(0, 200, 100, 0.05)); border-color: rgba(0, 255, 127, 0.2);">
                <div class="metric-label">Total Return</div>
                <div class="metric-value" style="background: linear-gradient(135deg, #00ff7f, #00cc66); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    {ml['total_return_pct']:.2f}%
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
            <div class="metric-box" style="background: linear-gradient(135deg, rgba(100, 150, 255, 0.1), rgba(80, 130, 255, 0.05)); border-color: rgba(100, 150, 255, 0.2);">
                <div class="metric-label">CAGR</div>
                <div class="metric-value" style="background: linear-gradient(135deg, #6496ff, #5080ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    {ml['cagr_pct']:.2f}%
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
            <div class="metric-box" style="background: linear-gradient(135deg, rgba(255, 99, 71, 0.1), rgba(255, 70, 50, 0.05)); border-color: rgba(255, 99, 71, 0.2);">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value" style="background: linear-gradient(135deg, #ff6347, #ff4632); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    {ml['max_drawdown_pct']:.2f}%
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
            <div class="metric-box" style="background: linear-gradient(135deg, rgba(255, 215, 0, 0.1), rgba(255, 200, 0, 0.05)); border-color: rgba(255, 215, 0, 0.2);">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value" style="background: linear-gradient(135deg, #ffd700, #ffc800); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    {ml['sharpe_ratio']:.2f}
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Second Row - Additional Mandatory Metrics
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Volatility</div>
                <div class="metric-value" style="color: #667eea; background: none; -webkit-text-fill-color: #667eea;">
                    {ml['volatility_pct']:.2f}%
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col6:
        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value" style="color: #764ba2; background: none; -webkit-text-fill-color: #764ba2;">
                    {ml['win_rate_pct']:.1f}%
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col7:
        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Final Equity</div>
                <div class="metric-value" style="font-size: 24px; color: #00ff7f; background: none; -webkit-text-fill-color: #00ff7f;">
                    INR {ml['total_equity_value']:,.0f}
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col8:
        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Total Trades</div>
                <div class="metric-value" style="color: white; background: none; -webkit-text-fill-color: white;">
                    {ml['total_trades']}
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Third Row - Confidence & Market Metrics
    st.markdown("### 🛡️ Risk & Benchmark Analysis")
    
    # Helper for market
    mkt = metrics.market_metrics
    
    col9, col10, col11, col12,col13= st.columns(5)

    with col9:
        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Total Return</div>
                <div class="metric-value" style="color: #00ff7f; background: none; -webkit-text-fill-color: #00ff7f;">
                    {mkt['total_return_pct']:.2f}%
                </div>
            </div>
            
        """, unsafe_allow_html=True)
        
    with col10:
        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">CAGR</div>
                <div class="metric-value" style="color: #ffd700; background: none; -webkit-text-fill-color: #ffd700;">
                    {mkt['cagr_pct']:.2f}%
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    with col11:
        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Volatality</div>
                <div class="metric-value" style="color: white; background: none; -webkit-text-fill-color: white;">
                    {mkt['volatility_pct']:.2f}%
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    with col12:
        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value" style="color: white; background: none; -webkit-text-fill-color: white;">
                    {mkt['sharpe_ratio']:.2f}
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col13:
        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value" style="color: white; background: none; -webkit-text-fill-color: white;">
                    {mkt['max_drawdown_pct']:.2f}%
                </div>
        """, unsafe_allow_html=True)

def render_trade_history(metrics: BacktestMetrics):
    """
    Display Trade History Table
    """
    '''st.markdown("""
        <div class="section-header" style="background: linear-gradient(135deg, rgba(118, 75, 162, 0.15), rgba(102, 126, 234, 0.1)); border-color: rgba(118, 75, 162, 0.3);">
            <div class="section-title">
                📋 Trade History
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    if not metrics.trades:
        st.info("No trades executed during the backtest period.")
        return
    
    # Convert to DataFrame
    trades_data = []
    for trade in metrics.trades:
        trades_data.append({
            'Entry Date': trade.entry_date,
            'Exit Date': trade.exit_date,
            'Entry Price': f"₹{trade.entry_price:.2f}",
            'Exit Price': f"₹{trade.exit_price:.2f}",
            'P/L': f"₹{trade.profit_loss:.2f}",
            'P/L %': f"{trade.profit_loss_pct:.2f}%",
            'Duration (Days)': trade.duration_days,
            'Type': trade.trade_type
        })
    
    df = pd.DataFrame(trades_data)
    
    # Style the dataframe
    st.dataframe(
        df,
        width='stretch',
        hide_index=True,
        height=400
    )
    
    # Summary Stats
    winning_trades = len([t for t in metrics.trades if t.profit_loss > 0])
    losing_trades = len([t for t in metrics.trades if t.profit_loss <= 0])
    avg_win = sum([t.profit_loss for t in metrics.trades if t.profit_loss > 0]) / winning_trades if winning_trades > 0 else 0
    avg_loss = sum([t.profit_loss for t in metrics.trades if t.profit_loss <= 0]) / losing_trades if losing_trades > 0 else 0
    '''
    
    # Calculate trading metrics from trades
    # Calculate trading metrics from trades
    if not metrics.trades:
        return

    wins = [t for t in metrics.trades if t['PnL'] > 0]
    losses = [t for t in metrics.trades if t['PnL'] <= 0]
    
    winning_trades = len(wins)
    losing_trades = len(losses)
    
    total_wins = sum([t['PnL'] for t in wins])
    total_losses = abs(sum([t['PnL'] for t in losses]))
    
    avg_win = total_wins / winning_trades if winning_trades > 0 else 0
    avg_loss = total_losses / losing_trades if losing_trades > 0 else 0
    profit_factor = total_wins / total_losses if total_losses > 0 else 0

    st.markdown("### 📊 Trading Metrics")

    col1, col2, col3, col4,col5= st.columns(5)
    with col1:
        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Winning Trades</div>
                <div class="metric-value" style="color: #00ff7f; background: none; -webkit-text-fill-color: #00ff7f;">{winning_trades}</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Losing Trades</div>
                <div class="metric-value" style="color: #ff6347; background: none; -webkit-text-fill-color: #ff6347;">{losing_trades}</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Avg Win</div>
                <div class="metric-value" style="color: #00ff7f; background: none; -webkit-text-fill-color: #00ff7f;">INR {avg_win:.2f}</div>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Avg Loss</div>
                <div class="metric-value" style="color: #ff6347; background: none; -webkit-text-fill-color: #ff6347;">INR {avg_loss:.2f}</div>
            </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Profit Factor</div>
                <div class="metric-value" style="color: #ffd700; background: none; -webkit-text-fill-color: #ffd700;">{profit_factor:.2f}</div>
            </div>
        """, unsafe_allow_html=True)

def render_data_scope(metrics: BacktestMetrics):
    """
    Display Data Scope & Constraints
    """
    st.markdown(f"""
        <div class="glass-card" style="margin-top: 20px; padding: 20px;">
            <div style="font-size: 18px; font-weight: 700; color: white; margin-bottom: 15px;">
                📊 Data Scope & Constraints
            </div>
            <div style="font-size: 14px; color: rgba(255, 255, 255, 0.8); line-height: 1.8;">
                • <strong>Historical Data Only:</strong> Backtested on past market data<br>
                • <strong>Missing Values:</strong> Removed from dataset<br>
                • <strong>Date Range:</strong> {metrics.date_range}<br>
                • <strong>Data Points:</strong> {metrics.data_points:,} trading periods<br>
                • <strong>Time Series:</strong> Date-indexed chronological data
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_confidence(metrics):
    """
    Display Confidence Analysis based on strategy metrics
    """
    st.markdown("#### Confidence Analysis")

    # Calculate confidence from metrics (0-100 scale) and cap at 100%
    confidence = min(metrics.confidence_score, 100.0)
    
    # Determine confidence level
    if confidence >= 80:
        confidence_level = "Very High"
        color = "#00ff7f"
    elif confidence >= 60:
        confidence_level = "High"
        color = "#6495ed"
    elif confidence >= 40:
        confidence_level = "Medium"
        color = "#ffd700"
    else:
        confidence_level = "Low"
        color = "#ff6347"

    st.markdown(f"""
        <div class="glass-metric-card">
            <div style="font-size:32px; font-weight:bold; color:{color};">
                {confidence:.1f}%
            </div>
            <div style="font-size:14px; opacity:0.8; margin-top:5px;">
                Confidence Level: <strong>{confidence_level}</strong>
            </div>
            <div style="background: rgba(255,255,255,0.1); height:10px; border-radius:5px; margin-top:10px; overflow:hidden;">
                <div class="feature-bar"
                     style="width:{min(confidence, 100)}%;
                            background:linear-gradient(90deg,{color},#764ba2);">
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

