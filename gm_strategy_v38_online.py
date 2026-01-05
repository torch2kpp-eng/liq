import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import io
from datetime import datetime

# ============================================================
# [APP CONFIG]
# ============================================================
st.set_page_config(page_title="GM Strategy v3.8", layout="wide")
st.title("🏛️ Grand Master: Adaptive Alpha Engine")
st.caption("v3.8 | ONLINE AUTO DATA | Adaptive Risk Budgeting | Valuation Tilt")

# ============================================================
# [SIDEBAR]
# ============================================================
st.sidebar.header("⚙️ Engine Tuning")
base_target_vol = st.sidebar.slider("🎯 Base Target Vol (%)", 30, 80, 50, 5)
max_lev_cap = st.sidebar.slider("🔒 Max Leverage Limit", 1.0, 3.0, 2.0, 0.1)

# ============================================================
# [DATA FETCH]
# ============================================================
@st.cache_data(ttl=3600)
def fetch_data():
    headers = {"User-Agent": "Mozilla/5.0"}
    data = {}

    # --- BTC (Yahoo Finance) ---
    try:
        btc_url = (
            "https://query1.finance.yahoo.com/v7/finance/download/BTC-USD"
            "?period1=1279315200&period2=9999999999&interval=1d&events=history"
        )
        btc = pd.read_csv(btc_url)
        btc['Date'] = pd.to_datetime(btc['Date'])
        btc.set_index('Date', inplace=True)
        data['btc'] = btc['Close'].astype(float)
    except:
        data['btc'] = pd.Series(dtype=float)

    # --- FRED Helper ---
    def get_fred(series_id):
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
            r = requests.get(url, headers=headers, timeout=10)
            df = pd.read_csv(io.StringIO(r.text))
            df['DATE'] = pd.to_datetime(df['DATE'])
            df.set_index('DATE', inplace=True)
            s = pd.to_numeric(df.iloc[:, 0], errors='coerce')
            return s
        except:
            return pd.Series(dtype=float)

    data['m2'] = get_fred("M2SL")
    data['spread'] = get_fred("BAMLH0A0HYM2")

    return data

with st.spinner("📡 Fetching live macro & market data..."):
    raw = fetch_data()

if raw['btc'].empty or raw['m2'].empty:
    st.error("❌ Online BTC or M2 data fetch failed.")
    st.stop()

# ============================================================
# [ENGINE]
# ============================================================
def run_engine(raw, base_vol, max_lev):
    # --- Unified Index ---
    start = max(raw['btc'].index.min(), raw['m2'].index.min())
    end = raw['btc'].index.max()
    idx = pd.date_range(start, end, freq='D')

    price = raw['btc'].reindex(idx).interpolate().ffill()
    m2 = raw['m2'].reindex(idx).ffill().bfill()
    spread = raw['spread'].reindex(idx).ffill().fillna(3.5)

    # --- Spread Regime ---
    spread_z = (spread - spread.rolling(365).mean()) / (spread.rolling(365).std() + 1e-9)
    lag = np.where(spread_z.shift(1) < 0.5, 56, 168)

    # --- Kalman Valuation ---
    N = len(idx)
    shifted_m2 = []
    m2v = m2.values

    for i in range(N):
        j = i - lag[i]
        shifted_m2.append(m2v[j] if j >= 0 else m2v[0])

    x = np.log(np.array(shifted_m2))
    y = np.log(price.values)

    lam = 0.9995
    theta = np.array([0.0, 1.0])
    P = np.eye(2) * 100

    residuals, betas = [], []

    for i in range(N):
        X = np.array([1, x[i]])
        y_hat = X @ theta
        err = y[i] - y_hat

        residuals.append(err)
        betas.append(theta[1])

        K = (P @ X) / (lam + X.T @ P @ X)
        theta = theta + K * err
        P = (P - np.outer(K, X.T @ P)) / lam

    df = pd.DataFrame(index=idx)
    df['Price'] = price
    df['Residual'] = residuals
    df['Z_Gap'] = (df['Residual'] - df['Residual'].rolling(730, min_periods=90).mean()) / df['Residual'].rolling(730, min_periods=90).std()
    df['Beta'] = betas
    df['Beta_MA'] = df['Beta'].rolling(60).mean()
    df['Beta_Up'] = df['Beta'] > df['Beta_MA']
    df['Vol_30'] = price.pct_change().rolling(30).std() * np.sqrt(365) * 100
    df['MA200'] = price.rolling(200).mean()

    last = df.iloc[-1]

    if last['Price'] < last['MA200']:
        return df, 0.0, "❄️ HIBERNATION", 0, 0, 0

    target_vol = base_vol + (10 if last['Beta_Up'] else -10)
    scalar = min(target_vol / max(last['Vol_30'], 1.0), max_lev)

    val_mod = 1.0
    if last['Z_Gap'] > 2:
        val_mod = 0.7
    elif last['Z_Gap'] < -2:
        val_mod = 1.3

    exposure = min(scalar * val_mod, max_lev)

    return df, exposure, "ACTIVE", target_vol, last['Vol_30'], last['Z_Gap']

# ============================================================
# [RUN]
# ============================================================
df, exp, status, tgt_vol, curr_vol, z = run_engine(raw, base_target_vol, max_lev_cap)

# ============================================================
# [DASHBOARD]
# ============================================================
st.markdown("### 🧭 Institutional Command Center")

c1, c2, c3 = st.columns(3)
c1.metric("Final Exposure", f"{exp*100:.0f}%", status)
c2.metric("Target Vol", f"{tgt_vol:.0f}%", f"Current {curr_vol:.1f}%")
c3.metric("Valuation Z", f"{z:.2f} σ")

st.divider()

df_viz = df[df.index >= "2020-01-01"]

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_viz.index, y=df_viz['Price'], name="BTC Price"))
fig.add_trace(go.Scatter(x=df_viz.index, y=df_viz['MA200'], name="MA200"))
fig.update_layout(template="plotly_dark", height=450)
st.plotly_chart(fig, use_container_width=True)
