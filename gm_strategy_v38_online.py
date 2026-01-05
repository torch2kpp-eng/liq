import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import requests
import io
import os
from datetime import datetime, timedelta

# -----------------------------------------------------------
# [ENV]
# -----------------------------------------------------------
st.set_page_config(page_title="GM Strategy v3.8", layout="wide")
st.title("🏛️ Grand Master: Adaptive Alpha Engine")
st.caption("v3.8 | HYBRID DATA | Institutional Adaptive Risk Budgeting")

# -----------------------------------------------------------
# [SIDEBAR]
# -----------------------------------------------------------
st.sidebar.header("⚙️ Engine Tuning")
base_target_vol = st.sidebar.slider("🎯 Base Target Vol (%)", 30, 80, 50, 5)
max_lev_cap = st.sidebar.slider("🔒 Max Leverage Limit", 1.0, 3.0, 2.0, 0.1)

# -----------------------------------------------------------
# [DATA LOADER - HYBRID & FAST]
# -----------------------------------------------------------
@st.cache_data(ttl=3600)
def load_data_hybrid():
    data = {}
    start_str = "2013-01-01"
    
    # 1. Fast FRED Downloader (pdr 대신 직접 호출하여 속도 개선)
    def fetch_fred(series_id):
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                df = pd.read_csv(io.StringIO(r.text), index_col=0, parse_dates=True)
                return df.iloc[:, 0]
        except: return pd.Series(dtype=float)

    # 2. BTC Data (YFinance) - Multi-index Robustness
    try:
        with st.spinner("BTC 시세 동기화 중..."):
            btc_raw = yf.download("BTC-USD", start=start_str, progress=False, auto_adjust=True)
            if isinstance(btc_raw.columns, pd.MultiIndex):
                data["btc"] = btc_raw['Close'].iloc[:, 0]
            else:
                data["btc"] = btc_raw['Close']
    except: data["btc"] = pd.Series(dtype=float)

    # 3. Macro Data (FRED)
    with st.spinner("매크로 유동성 지표 동기화 중..."):
        data["m2"] = fetch_fred("M2SL")
        data["spread"] = fetch_fred("BAMLH0A0HYM2")

    return data

# 데이터 로드 실행
raw = load_data_hybrid()

# 밸리데이션
if raw["btc"].empty or raw["m2"].empty:
    st.error("데이터 통신에 문제가 발생했습니다. 잠시 후 다시 시도하거나 로컬 CSV를 확인하십시오.")
    st.stop()

# -----------------------------------------------------------
# [CORE ENGINE] - Vectorized for Speed
# -----------------------------------------------------------
def run_strategy_engine(raw_data, base_vol, max_lev):
    # 1. ALIGNMENT
    common_start = max(raw_data["btc"].index.min(), raw_data["m2"].index.min())
    common_end = raw_data["btc"].index.max()
    idx = pd.date_range(start=common_start, end=common_end, freq="D")

    price = raw_data["btc"].reindex(idx).interpolate("time").ffill().bfill()
    m2 = raw_data["m2"].reindex(idx).ffill().bfill()
    spread = raw_data["spread"].reindex(idx).ffill().bfill() if not raw_data["spread"].empty else pd.Series(3.5, index=idx)

    # 2. REGIME LAG (Shifted to remove bias)
    spread_roll_mean = spread.rolling(365).mean()
    spread_roll_std = spread.rolling(365).std()
    z_spread = (spread - spread_roll_mean) / (spread_roll_std + 1e-9)
    # PM 지시사항 반영: 1일 shift하여 look-ahead 방지
    lag_regime = np.where(z_spread.shift(1) < 0.5, 56, 168)

    # 3. SHIFTED M2 (Vectorized)
    m2_vals = m2.values
    shifted_m2 = np.zeros(len(idx))
    for i in range(len(idx)):
        j = int(i - lag_regime[i])
        shifted_m2[i] = m2_vals[j] if j >= 0 else m2_vals[0]

    # 4. RLS MODEL (Recursive Least Squares)
    x = np.log(shifted_m2)
    y = np.log(price.values)
    
    lam = 0.9995
    theta = np.array([0.0, 1.0])
    P = np.eye(2) * 100
    
    res, beta = np.zeros(len(x)), np.zeros(len(x))
    X_mat = np.column_stack([np.ones(len(x)), x])

    # RLS Loop
    for i in range(len(x)):
        xi = X_mat[i].reshape(-1, 1)
        pred = float(xi.T @ theta)
        err = y[i] - pred
        res[i] = err
        beta[i] = theta[1]

        K = (P @ xi) / (lam + xi.T @ P @ xi)
        theta = theta + (K.flatten() * err)
        P = (P - K @ xi.T @ P) / lam

    # 5. INDICATORS
    df = pd.DataFrame(index=idx)
    df["Price"] = price
    df["Residual"] = res
    df["Z_Gap"] = (df["Residual"] - df["Residual"].rolling(730, min_periods=90).mean()) / df["Residual"].rolling(730, min_periods=90).std()
    df["Beta"] = beta
    df["Beta_Up"] = df["Beta"] > df["Beta"].rolling(60).mean()
    df["MA200"] = df["Price"].rolling(200).mean()
    
    # VOLATILITY
    df["Vol_30"] = price.pct_change().rolling(30).std() * np.sqrt(365) * 100

    # 6. DECISION LOGIC (Adaptive Risk Budgeting)
    last = df.iloc[-1]
    curr_vol = last["Vol_30"] if not np.isnan(last["Vol_30"]) else 50
    z_val = last["Z_Gap"] if not np.isnan(last["Z_Gap"]) else 0
    trend_ok = last["Price"] > last["MA200"]

    status = "NEUTRAL"
    tgt_vol = base_vol
    exposure = 0.0
    val_mod = 1.0

    if not trend_ok:
        status = "HIBERNATION"
        exposure = 0.0
    else:
        # Beta-Driven Adaptive Vol
        if last["Beta_Up"]:
            tgt_vol = base_vol + 10
            status = "HIGH BETA (Aggressive)"
        else:
            tgt_vol = base_vol - 10
            status = "LOW BETA (Defensive)"

        # Vol Scalar (Risk Parity)
        scalar = min(tgt_vol / curr_vol, max_lev)

        # Valuation Modifier
        if z_val > 2:
            val_mod = 0.7
            status += " + TRIM"
        elif z_val < -2:
            val_mod = 1.3
            status += " + BOOST"
        
        exposure = min(scalar * val_mod, max_lev)

    return exposure, status, df, curr_vol, tgt_vol, z_val, val_mod

# -----------------------------------------------------------
# [DISPLAY]
# -----------------------------------------------------------
exp, stat, final_df, c_vol, t_vol, z_gap, v_mod = run_strategy_engine(raw, base_target_vol, max_lev_cap)

st.markdown("### 🧭 Alpha Command Center")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Recommended Exposure", f"{exp*100:.1f}%", stat)
m2.metric("Target vs Current Vol", f"{t_vol}%", f"Live {c_vol:.1f}%")
m3.metric("Valuation Z-Gap", f"{z_gap:.2f} σ", f"Tilt x{v_mod}")
m4.metric("Market Trend", "UP TREND" if exp > 0 else "DOWN TREND", delta_color="normal")

st.divider()

# Charts
dfv = final_df[final_df.index >= "2021-01-01"]
tab1, tab2 = st.tabs(["📊 Risk & Vol", "📈 Valuation Z-Gap"])

with tab1:
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=dfv.index, y=dfv["Vol_30"], name="Realized Vol", line=dict(color='yellow')))
    fig_vol.add_hline(y=t_vol, line_dash="dash", line_color="red", annotation_text="Target Vol")
    fig_vol.update_layout(title="Volatility Management (Targeting Risk)", template="plotly_dark")
    st.plotly_chart(fig_vol, use_container_width=True)

with tab2:
    fig_z = go.Figure()
    # Z-Gap Bar Color Logic
    colors = ['red' if x > 2 else 'lime' if x < -2 else 'gray' for x in dfv["Z_Gap"]]
    fig_z.add_trace(go.Bar(x=dfv.index, y=dfv["Z_Gap"], marker_color=colors))
    fig_z.add_hline(y=2, line_dash="dash", line_color="red")
    fig_z.add_hline(y=-2, line_dash="dash", line_color="lime")
    fig_z.update_layout(title="Valuation Signal (Z-Gap)", template="plotly_dark")
    st.plotly_chart(fig_z, use_container_width=True)
