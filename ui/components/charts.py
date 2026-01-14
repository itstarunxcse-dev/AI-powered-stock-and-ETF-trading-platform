# -*- coding: utf-8 -*-
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from contracts.schema import StockData

# --- COLOR PALETTE (Matches CSS Variables) ---
COLORS = {
    'neon_green': '#00ff7f',
    'neon_red': '#ff6347',
    'neon_blue': '#667eea',
    'neon_purple': '#764ba2',
    'gold': '#ffd700',
    'grid': 'rgba(255, 255, 255, 0.06)',
    'text': '#ffffff',
    'text_secondary': 'rgba(255, 255, 255, 0.7)'
}

def _update_fig_layout(fig, height=500):
    """
    Applies the global glassmorphism theme to any Plotly figure.
    """
    fig.update_layout(
        template='plotly_dark',
        height=height,
        margin=dict(l=15, r=15, t=20, b=15),
        paper_bgcolor='rgba(0,0,0,0)',       # Transparent
        plot_bgcolor='rgba(0,0,0,0)',        # Transparent
        font=dict(
            family="Inter, sans-serif",
            color=COLORS['text'],
            size=12
        ),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='rgba(20, 25, 40, 0.95)',
            bordercolor='rgba(255, 255, 255, 0.3)',
            font_size=13,
            font_family="JetBrains Mono, monospace",
            font_color="#ffffff"
        ),
        xaxis_rangeslider_visible=False,
        showlegend=False
    )
    
    # Enhanced Gridlines
    fig.update_xaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='rgba(255, 255, 255, 0.2)', 
        zeroline=False,
        showline=True,
        linecolor='rgba(255, 255, 255, 0.5)'
    )
    fig.update_yaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='rgba(255, 255, 255, 0.2)', 
        zeroline=True,
        zerolinecolor='rgba(255, 255, 255, 0.4)',
        showline=True,
        linecolor='rgba(255, 255, 255, 0.5)',
        automargin=True
    )
    
    return fig

def render_chart_header(title, icon="📈"):
    """
    Renders a styled header inside the glass card.
    """
    st.markdown(f"""
        <div style="
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 15px 20px;
            margin-bottom: 20px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            display: flex; 
            align-items: center;">
            <div style="
                background: rgba(255, 255, 255, 0.1); 
                width: 40px; 
                height: 40px; 
                border-radius: 10px; 
                display: flex; 
                align-items: center; 
                justify-content: center; 
                margin-right: 15px;
                font-size: 20px;">
                {icon}
            </div>
            <span style="
                font-family: 'Inter', sans-serif; 
                font-weight: 700; 
                font-size: 18px; 
                color: white;">
                {title}
            </span>
        </div>
    """, unsafe_allow_html=True)


def render_price_chart(data: StockData):
    """
    Renders the OHLCV chart inside a glass container.
    """
    # Start Glass Card Wrapper
    render_chart_header("Price Action & Volume Analysis", "📈")

    # Create Subplots
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05, 
        row_width=[0.2, 0.7]
    )

    # 1. Candlestick
    fig.add_trace(go.Candlestick(
        x=data.dates,
        open=data.opens, high=data.highs, low=data.lows, close=data.closes,
        increasing_line_color=COLORS['neon_green'],
        decreasing_line_color=COLORS['neon_red'],
        name='OHLC'
    ), row=1, col=1)

    # 2. Moving Averages (Neon Glows)
    fig.add_trace(go.Scatter(
        x=data.dates, y=data.sma_50, 
        line=dict(color=COLORS['gold'], width=1.5), 
        name='SMA 50', opacity=0.8
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=data.dates, y=data.ema_12, 
        line=dict(color=COLORS['neon_blue'], width=1.5), 
        name='EMA 12', opacity=0.8
    ), row=1, col=1)

    # 3. Volume Bar Chart
    # Color logic: Green if Close > Open, else Red
    vol_colors = [COLORS['neon_red'] if c < o else COLORS['neon_green'] 
                  for o, c in zip(data.opens, data.closes)]
    
    fig.add_trace(go.Bar(
        x=data.dates, y=data.volumes, 
        marker_color=vol_colors, 
        name='Volume', opacity=0.5,
        marker_line_width=0
    ), row=2, col=1)

    # Apply Shared Styling
    _update_fig_layout(fig, height=600)
    
    # Render with Streamlit
    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})

def render_rsi_chart(data: StockData):
    """
    Render RSI indicator with styled zones.
    """
    render_chart_header("RSI Momentum", "⚡")
    
    fig = go.Figure()
    
    # RSI Line
    fig.add_trace(go.Scatter(
        x=data.dates, y=data.rsi, 
        line=dict(color=COLORS['neon_purple'], width=2), 
        name='RSI'
    ))
    
    # Overbought/Oversold Zones (Rectangles for cleaner look than filled lines)
    # Overbought Zone
    fig.add_hrect(
        y0=70, y1=100, 
        fillcolor=COLORS['neon_red'], opacity=0.1, 
        line_width=0
    )
    # Oversold Zone
    fig.add_hrect(
        y0=0, y1=30, 
        fillcolor=COLORS['neon_green'], opacity=0.1, 
        line_width=0
    )
    
    # Reference Lines
    fig.add_hline(y=70, line_dash="dash", line_color=COLORS['neon_red'], line_width=1, opacity=0.5)
    fig.add_hline(y=30, line_dash="dash", line_color=COLORS['neon_green'], line_width=1, opacity=0.5)
    
    _update_fig_layout(fig, height=300)
    fig.update_yaxes(range=[0, 100], tickvals=[30, 50, 70])
    
    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})

def render_drawdown_chart(dates, drawdown_curve):
    """
    Renders the Drawdown Chart showing portfolio losses from peak.
    """
    render_chart_header("Drawdown Analysis", "📉")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates, y=drawdown_curve, 
        fill='tozeroy', 
        line=dict(color=COLORS['neon_red'], width=2),
        fillcolor='rgba(255, 99, 71, 0.1)', 
        name='Drawdown'
    ))
    
    _update_fig_layout(fig, height=300)
    
    # Add max drawdown annotation
    if len(drawdown_curve) > 0:
        min_dd = min(drawdown_curve)
        min_idx = drawdown_curve.index(min_dd)
        fig.add_annotation(
            x=dates[min_idx], y=min_dd,
            text=f"Max DD: {min_dd:.2f}%",
            showarrow=True, arrowhead=2, ax=0, ay=30,
            font=dict(color=COLORS['neon_red'], size=11, family="JetBrains Mono"),
            bgcolor="rgba(0,0,0,0.7)", borderpad=4
        )

    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})

def render_price_with_trades_chart(dates, prices, buy_signals, sell_signals):
    """
    Renders price chart with buy/sell markers from backtest.
    """
    render_chart_header("Trade Execution Visualization", "🎯")
    
    fig = go.Figure()
    
    # Price Line
    fig.add_trace(go.Scatter(
        x=dates, y=prices, 
        line=dict(color=COLORS['neon_blue'], width=2), 
        name='Price'
    ))
    
    # Buy Markers
    if buy_signals:
        buy_dates = [dates[i] for i in buy_signals]
        buy_prices = [prices[i] for i in buy_signals]
        fig.add_trace(go.Scatter(
            x=buy_dates, y=buy_prices,
            mode='markers',
            marker=dict(
                symbol='triangle-up',
                size=12,
                color=COLORS['neon_green'],
                line=dict(color='white', width=1)
            ),
            name='Buy'
        ))
    
    # Sell Markers
    if sell_signals:
        sell_dates = [dates[i] for i in sell_signals]
        sell_prices = [prices[i] for i in sell_signals]
        fig.add_trace(go.Scatter(
            x=sell_dates, y=sell_prices,
            mode='markers',
            marker=dict(
                symbol='triangle-down',
                size=12,
                color=COLORS['neon_red'],
                line=dict(color='white', width=1)
            ),
            name='Sell'
        ))
    
    _update_fig_layout(fig, height=400)
    fig.update_layout(showlegend=True, legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    
    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})

def render_equity_curve(dates, equity_curve):
    """
    Renders the Equity Curve with a gradient fill effect.
    """
    render_chart_header("Strategy Performance", "💰")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates, y=equity_curve, 
        fill='tozeroy', 
        line=dict(color=COLORS['neon_green'], width=2),
        # Using a slight transparency for the fill to let gridlines show
        fillcolor='rgba(0, 255, 127, 0.1)', 
        name='Equity'
    ))
    
    _update_fig_layout(fig, height=350)
    
    # Add a "Current Value" annotation at the end
    if len(equity_curve) > 0:
        last_val = equity_curve[-1]
        last_date = dates[-1]
        fig.add_annotation(
            x=last_date, y=last_val,
            text=f"₹{last_val:,.0f}",
            showarrow=True, arrowhead=0, ax=0, ay=-20,
            font=dict(color=COLORS['neon_green'], size=12, family="JetBrains Mono"),
            bgcolor="rgba(0,0,0,0.5)", borderpad=4
        )

    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})

def render_equity_comparison(dates, market_equity, strategy_equity, buy_signals=None, sell_signals=None, prices=None):
    """
    Renders Equity Curve comparison: Market (Buy & Hold) vs ML Strategy
    with optional trade markers overlay.
    """
    render_chart_header("Equity Curve: Market vs ML Strategy", "📈")
    
    # Create subplots: Main equity chart + Volume-style indicator
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.75, 0.25],
        subplot_titles=("", "Trade Activity")
    )
    
    # Market (Buy & Hold) Line
    fig.add_trace(go.Scatter(
        x=dates,
        y=market_equity,
        name="Market (Buy & Hold)",
        line=dict(color="#64748b", width=2, dash="dot"),
        opacity=0.7
    ), row=1, col=1)
    
    # ML Strategy Line
    fig.add_trace(go.Scatter(
        x=dates,
        y=strategy_equity,
        name="ML Strategy",
        line=dict(color=COLORS['neon_green'], width=3),
        fill='tonexty',
        fillcolor='rgba(0, 255, 127, 0.1)'
    ), row=1, col=1)
    
    # Add trade markers if provided
    if buy_signals and sell_signals and prices:
        buy_dates = [dates[i] for i in buy_signals if i < len(dates)]
        buy_equity = [strategy_equity[i] for i in buy_signals if i < len(strategy_equity)]
        
        sell_dates = [dates[i] for i in sell_signals if i < len(dates)]
        sell_equity = [strategy_equity[i] for i in sell_signals if i < len(strategy_equity)]
        
        # Buy markers
        fig.add_trace(go.Scatter(
            x=buy_dates, y=buy_equity,
            mode='markers',
            marker=dict(
                symbol='triangle-up',
                size=10,
                color=COLORS['neon_green'],
                line=dict(color='white', width=1)
            ),
            name='Buy',
            showlegend=True
        ), row=1, col=1)
        
        # Sell markers
        fig.add_trace(go.Scatter(
            x=sell_dates, y=sell_equity,
            mode='markers',
            marker=dict(
                symbol='triangle-down',
                size=10,
                color=COLORS['neon_red'],
                line=dict(color='white', width=1)
            ),
            name='Sell',
            showlegend=True
        ), row=1, col=1)
        
        # Trade activity bars (volume-style visualization)
        trade_activity = [0] * len(dates)
        for idx in buy_signals:
            if idx < len(trade_activity):
                trade_activity[idx] = 1
        for idx in sell_signals:
            if idx < len(trade_activity):
                trade_activity[idx] = -1
        
        colors = ['#00ff7f' if x > 0 else '#ff6347' if x < 0 else 'rgba(255,255,255,0.1)' for x in trade_activity]
        
        fig.add_trace(go.Bar(
            x=dates,
            y=[abs(x) for x in trade_activity],
            marker_color=colors,
            name='Trades',
            showlegend=False,
            opacity=0.6
        ), row=2, col=1)
    
    _update_fig_layout(fig, height=550)
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor=COLORS['neon_blue'],
            borderwidth=1
        )
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Equity (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Activity", row=2, col=1, showticklabels=False)
    
    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})

    
def render_profit_loss_chart(dates, returns):
    """
    Renders a bar chart of daily profit/loss.
    """
    render_chart_header("Daily Profit/Loss", "📊")
    
    colors = [COLORS['neon_green'] if r >= 0 else COLORS['neon_red'] for r in returns]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dates,
        y=returns,
        marker_color=colors,
        name='Daily Return'
    ))
    
    _update_fig_layout(fig, height=350)
    fig.update_yaxes(title_text="Return", tickformat=".2%")
    
    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})

def render_volume_chart(dates, volumes):
    """
    Renders a volume chart.
    """
    render_chart_header("Trading Volume", "📊")
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dates,
        y=volumes,
        marker_color=COLORS['neon_blue'],
        opacity=0.6,
        name='Volume'
    ))
    
    _update_fig_layout(fig, height=300)
    fig.update_yaxes(title_text="Volume")
    
    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
