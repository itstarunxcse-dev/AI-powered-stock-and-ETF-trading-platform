# -*- coding: utf-8 -*-
import streamlit as st
from contracts.schema import MLSignal

def render_prediction_card(signal: MLSignal):
    """
    Enhanced prediction card UI with all 8 ML features displayed beautifully.
    """
    

    
    # ========================================
    # 1️⃣ PRIMARY SIGNAL DISPLAY
    # ========================================
    
    signal_emoji = "🟢" if signal.signal_value == 1 else ("🔴" if signal.signal_value == -1 else "🟡")
    
    st.markdown(f"""
    <div class="glass-card" style="padding: 20px; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="flex: 2;">
                <h3 style="margin: 0; padding: 0; color: white;">🤖 AI Trading Signal</h3>
            </div>
            <div style="flex: 1; text-align: center;">
                <div class="model-badge">{signal_emoji} Signal: {signal.signal_value}</div>
            </div>
            <div style="flex: 1; text-align: right; opacity: 0.7; font-size: 12px; color: #cbd5e1;">
                📅 {signal.prediction_date}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ========================================
    # UNIFIED METRICS ROW
    # ========================================

    # CSS for the new unified card design AND the explanation box
    st.markdown("""
<style>
.metric-row { display: flex; gap: 20px; margin-bottom: 25px; flex-wrap: wrap; }
.metric-box {
  flex: 1;
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 16px;
  padding: 20px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  min-width: 150px;
  backdrop-filter: blur(12px);
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
  transition: transform 0.2s;
}
.metric-box:hover {
  transform: translateY(-2px);
  background: rgba(255, 255, 255, 0.05);
  border-color: rgba(255, 255, 255, 0.15);
}
.metric-label {
  font-size: 13px;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: rgba(255, 255, 255, 0.5);
  margin-bottom: 8px;
}
.metric-value-lg {
  font-size: 32px;
  font-weight: 800;
  background: linear-gradient(135deg, #fff 0%, #cbd5e1 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.confidence-bar-bg {
  width: 100%;
  height: 6px;
  background: rgba(255,255,255,0.1);
  border-radius: 3px;
  margin-top: 10px;
  overflow: hidden;
}
.confidence-bar-fill {
  height: 100%;
  border-radius: 3px;
  transition: width 1s ease-out;
}
.glass-insight-box {
  background: rgba(15, 23, 42, 0.6);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 12px;
  padding: 20px;
  margin-top: 10px;
  backdrop-filter: blur(12px);
}
.glass-factor-item {
  background: rgba(255, 255, 255, 0.05);
  border-radius: 8px;
  padding: 10px 14px;
  margin-bottom: 8px;
  font-size: 14px;
  color: #e2e8f0;
  border-left: 3px solid rgba(100, 149, 237, 0.5);
}
</style>
""", unsafe_allow_html=True)
    
    # Custom color logic
    if signal.action == "BUY":
        action_color = "#00ff88"
        action_icon = "🚀"
        grad_start, grad_end = "#00b09b", "#96c93d"
    elif signal.action == "SELL":
        action_color = "#ff5555"
        action_icon = "🔻"
        grad_start, grad_end = "#ff5f6d", "#ffc371"
    else:
        action_color = "#ffcc00"
        action_icon = "⏸️"
        grad_start, grad_end = "#f7971e", "#ffd200"

    # HTML Structure (No indentation to prevent Code Blocks)
    st.markdown(f"""
<div class="metric-row">
<div class="metric-box" style="border-bottom: 3px solid {action_color};">
<div class="metric-label">Recommendation</div>
<div style="font-size: 36px; font-weight: 800; color: {action_color}; text-shadow: 0 0 20px {action_color}40;">
{action_icon} {signal.action}
</div>
<div style="font-size: 12px; opacity: 0.7; margin-top: 4px;">Strong Signal Momentum</div>
</div>
<div class="metric-box">
<div class="metric-label">AI Confidence</div>
<div class="metric-value-lg">{signal.confidence:.1f}%</div>
<div style="font-size: 12px; color: {action_color}; margin-bottom: 5px;">{signal.confidence_level} Certainty</div>
<div class="confidence-bar-bg">
<div class="confidence-bar-fill" style="width: {signal.confidence}%; background: linear-gradient(90deg, {grad_start}, {grad_end});"></div>
</div>
</div>
<div class="metric-box">
<div class="metric-label">Model Context</div>
<div style="font-size: 18px; font-weight: 600; color: #e2e8f0; margin-bottom: 5px;">
{signal.model_type.split(' ')[0]} v3
</div>
<div style="font-size: 11px; opacity: 0.6; display: flex; align-items: center; gap: 5px;">
<span>⏱ {signal.prediction_frequency}</span>
</div>
<div style="font-size: 11px; opacity: 0.6; margin-top: 2px;">
📅 {signal.last_trained}
</div>
</div>
</div>
""", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ========================================
    # 4️⃣ GEN-AI EXPLANATION
    # ========================================
    
    st.markdown("#### 🧠 AI Explanation")
    st.markdown(f"""
        <div class="glass-insight-box">
            <div style="white-space: pre-wrap; line-height: 1.6; color: rgba(255,255,255,0.9);">
{signal.reasoning}
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ========================================
    # 6️⃣ TOP FEATURES / INDICATORS USED
    # ========================================
    
    col_features1, col_features2 = st.columns(2)
    
    with col_features1:
        st.markdown("#### 📊 Key Factors")
        for i, factor in enumerate(signal.key_factors[:5], 1):
            st.markdown(f"""
                <div class="glass-factor-item">
                    <strong>{i}.</strong> {factor}
                </div>
            """, unsafe_allow_html=True)
    
    with col_features2:
        st.markdown("#### 🎯 Feature Importance")
        
        # Defensive check for feature_importance
        if signal.feature_importance and isinstance(signal.feature_importance, dict) and len(signal.feature_importance) > 0:
            # Sort by importance
            sorted_features = sorted(signal.feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            for feature, importance in sorted_features[:5]:
                st.markdown(f"""
                    <div style="margin-bottom:12px;">
                        <div style="display:flex; justify-content:space-between; font-size:13px; margin-bottom:3px;">
                            <span style="color:rgba(255,255,255,0.9);">{feature}</span>
                            <span style="color:#6495ed; font-weight:bold;">{importance:.1f}%</span>
                        </div>
                        <div style="background:rgba(255,255,255,0.1); height:6px; border-radius:3px; overflow:hidden;">
                            <div style="width:{importance}%; height:100%; background:linear-gradient(90deg,#667eea,#764ba2); border-radius:3px;"></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            # Fallback message when feature importance is not available
            st.markdown("""
                <div class="glass-factor-item" style="text-align:center; opacity:0.7;">
                    ⚠️ Feature importance data not available
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Glassmorphism alerts
    if signal.confidence < 60:
        st.markdown(
            '<div style="background:rgba(255,193,7,0.1); backdrop-filter:blur(10px); '
            'border-left:4px solid rgba(255,193,7,0.6); padding:15px; border-radius:10px; margin:10px 0;">'
            '⚠️ <strong>Caution:</strong> This signal has lower confidence. Consider waiting for stronger signals '
            'or reducing position size to manage risk.'
            '</div>', 
            unsafe_allow_html=True
        )
    elif signal.action == "SELL" and signal.confidence > 75:
        st.markdown(
            '<div style="background:rgba(244,67,54,0.1); backdrop-filter:blur(10px); '
            'border-left:4px solid rgba(244,67,54,0.6); padding:15px; border-radius:10px; margin:10px 0;">'
            '🚨 <strong>Strong Sell Signal:</strong> Consider reviewing your position. '
            'High confidence sell signals may indicate significant downside risk.'
            '</div>', 
            unsafe_allow_html=True
        )
    elif signal.action == "BUY" and signal.confidence > 80:
        st.markdown(
            '<div style="background:rgba(76,175,80,0.1); backdrop-filter:blur(10px); '
            'border-left:4px solid rgba(76,175,80,0.6); padding:15px; border-radius:10px; margin:10px 0;">'
            '✅ <strong>Strong Buy Signal:</strong> High confidence indicates favorable conditions. '
            'Consider this for portfolio entry or position increase.'
            '</div>', 
            unsafe_allow_html=True
        )
