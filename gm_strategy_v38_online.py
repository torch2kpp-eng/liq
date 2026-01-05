import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from pandas_datareader import data as pdr
from datetime import datetime

# -----------------------------------------------------------
# [ENV]
# -----------------------------------------------------------
st.set_page_config(page_title="GM Strategy v3.8", layout="wide")
st.title("🏛️ Grand Master: Adaptive Alpha Engine")
st.caption("v3.8 | ONLINE DATA | Adaptive Risk Budgeting | Valuation Tilt")

# -----------------------------------------------------------
# [SIDEBAR]
# -----------------------------------------------------------
st.sidebar.header("⚙️ Engine Tuning")
base_target_vol = st.sidebar.slider("🎯 Base Target Vol (%)", 30, 80, 50, 5)
max_lev_cap = st.sidebar.slider("🔒 Max Leverage Limit", 1.0, 3.0, 2.0, 0.1)

# -----------------------------------------------------------
# [DATA LOADER - ONLINE ONLY]
# -----------------------------------------------------------
@st.cache_data(ttl=3600)
def load_online_data():
    data = {}

    start = "2013-01-01"
    end = datetime.today().strftime("%Y-%m-%d")

    # BTC (Yahoo)
    btc = yf.download("BTC-USD", start=start, end=end, progress=False)
    if btc.empty:
        return None
    data["btc"] = btc["Adj Close"]

    # M2 (FRED)
    try:
        m2 = pdr.DataReader("M2SL", "fred", start, end)
        data["m2"] = m2["M2SL"]
    except:
        data["m2"] = pd.Series(dtype=float)

    # Credit Spread (optional)
    try:
        spread = pdr.DataReader("BAMLH0A0HYM2", "fred", start, end)
        data["spread"] = spread["BAMLH0A0HYM2"]
    except:
        data["spread"] = pd.Series(dtype=float)

    return data

raw = load_online_data()

if raw is None or raw["btc"].empty or raw["m2"].empty:
    st.error("❌ BTC 또는 M2 온라인 데이터를 불러오지 못했습니다.")
    st.stop()

# -----------------------------------------------------------
# [CORE ENGINE]
# -----------------------------------------------------------
def run_strategy_engine(raw_data, base_vol, max_lev):

    # 1. ALIGNMENT
    idx = pd.date_range(
        start=max(raw_data["btc"].index.min(), raw_data["m2"].index.min()),
        end=raw_data["btc"].index.max(),
        freq="D"
    )

    price = raw_data["btc"].reindex(idx).interpolate("time").ffill().bfill()
    m2 = raw_data["m2"].reindex(idx).ffill().bfill()

    if raw_data["spread"].empty:
        spread = pd.Series(3.5, index=idx)
    else:
        spread = raw_data["spread"].reindex(idx).ffill().bfill()

    # 2. REGIME LAG
    z = (spread - spread.rolling(365).mean()) / (spread.rolling(365).std() + 1e-9)
    lag = np.where(z.shift(1) < 0.5, 56, 168)

    shifted_m2 = []
    m2v = m2.values
    for i in range(len(idx)):
        j = int(i - lag[i])
        shifted_m2.append(m2v[j] if j >= 0 else m2v[0])

    x = np.log(np.array(shifted_m2))
    y = np.log(price.values)

    # 3. RLS VALUATION MODEL
    lam = 0.9995
    theta = np.array([0.0, 1.0])
    P = np.eye(2) * 100

    res, beta = [], []
    X = np.column_stack([np.ones(len(x)), x])

    for i in range(len(x)):
        xi = X[i].reshape(-1, 1)
        yi = y[i]
        pred = float(xi.T @ theta)
        err = yi - pred

        res.append(err)
        beta.append(theta[1])

        K = (P @ xi) / (lam + xi.T @ P @ xi)
        theta = theta + (K.flatten() * err)
        P = (P - K @ xi.T @ P) / lam

    df = pd.DataFrame(index=idx)
    df["Price"] = price
    df["Residual"] = res
    df["Z_Gap"] = (
        (df["Residual"] - df["Residual"].rolling(730, min_periods=90).mean())
        / df["Residual"].rolling(730, min_periods=90).std()
    )

    df["Beta"] = beta
    df["Beta_MA"] = df["Beta"].rolling(60).mean()
    df["Beta_Up"] = df["Beta"] > df["Beta_MA"]

    ret = price.pct_change()
    df["Vol_30"] = ret.rolling(30).std() * np.sqrt(365) * 100

    df["MA200"] = df["Price"].rolling(200).mean()

    # 4. FINAL DECISION
    last = df.iloc[-1]
    curr_vol = last["Vol_30"] if not np.isnan(last["Vol_30"]) else 50
    z_val = last["Z_Gap"] if not np.isnan(last["Z_Gap"]) else 0
    trend = last["Price"] > last["MA200"]

    status = ""
    adaptive = base_vol
    val_mod = 1.0
    exposure = 0.0

    if not trend:
        status = "HIBERNATION (Cash)"
        exposure = 0.0
    else:
        if last["Beta_Up"]:
            adaptive += 10
            status = "HIGH BETA"
        else:
            adaptive -= 10
            status = "LOW BETA"

        scalar = min(adaptive / curr_vol, max_lev)

        if z_val > 2:
            val_mod = 0.7
            status += " + TRIM"
        elif z_val < -2:
            val_mod = 1.3
            status += " + BOOST"

        exposure = min(scalar * val_mod, max_lev)

    return exposure, status, df, curr_vol, adaptive, z_val, val_mod, last["Beta_Up"]

# -----------------------------------------------------------
# [RUN]
# -----------------------------------------------------------
exp, status, df, vol, tgt, z, vm, bu = run_strategy_engine(
    raw, base_target_vol, max_lev_cap
)

# -----------------------------------------------------------
# [DASHBOARD]
# -----------------------------------------------------------
st.markdown("### 🧭 Institutional Command Center")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Final Exposure", f"{exp*100:.0f}%", status)
c2.metric("Risk Budget", f"{tgt:.0f}%", f"Current {vol:.1f}%")
c3.metric("Valuation Tilt", f"x{vm}", f"Z {z:.2f}")
c4.metric("Liquidity Regime", "🚀" if bu else "🐌")

st.divider()

dfv = df[df.index >= "2020-01-01"]

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=dfv.index, y=dfv["Vol_30"], name="Volatility"))
fig1.add_hline(y=tgt, line_dash="dash", line_color="red")
fig1.update_layout(title="Volatility Control", height=400)
st.plotly_chart(fig1, use_container_width=True)

fig2 = go.Figure()
fig2.add_bar(x=dfv.index, y=dfv["Z_Gap"])
fig2.add_hline(y=2, line_dash="dash", line_color="red")
fig2.add_hline(y=-2, line_dash="dash", line_color="green")
fig2.update_layout(title="Valuation Z-Gap", height=400)
st.plotly_chart(fig2, use_container_width=True)
