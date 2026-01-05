import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os

# -----------------------------------------------------------
# [환경 설정]
# -----------------------------------------------------------
st.set_page_config(page_title="GM Strategy v3.7", layout="wide")
st.title("🏛️ Grand Master: Adaptive Alpha Engine")
st.caption("v3.7 | Local Data | Adaptive Risk Budgeting | Valuation Tilt")

# -----------------------------------------------------------
# [사이드바] Strategy Tuning
# -----------------------------------------------------------
st.sidebar.header("⚙️ Engine Tuning")
base_target_vol = st.sidebar.slider("🎯 Base Target Vol (%)", 30, 80, 50, 5)
max_lev_cap = st.sidebar.slider("🔒 Max Leverage Limit", 1.0, 3.0, 2.0, 0.1)

# -----------------------------------------------------------
# [CORE] 데이터 로더
# -----------------------------------------------------------
@st.cache_data(ttl=3600)
def load_data():
    data = {}

    def read_csv(filenames):
        for f in filenames:
            if os.path.exists(f):
                try:
                    df = pd.read_csv(
                        f,
                        parse_dates=['observation_date'],
                        index_col='observation_date'
                    )
                    s = df.iloc[:, 0].apply(pd.to_numeric, errors='coerce')
                    return s.sort_index()
                except:
                    continue
        return pd.Series(dtype=float)

    data['btc'] = read_csv(['CBBTCUSD 2.csv', 'CBBTCUSD.csv', 'btc.csv'])
    data['m2'] = read_csv(['M2SL 2.csv', 'M2SL.csv', 'm2.csv'])
    data['spread'] = read_csv(['BAMLH0A0HYM2.csv', 'spread.csv'])

    return data

raw = load_data()

if raw['btc'].empty or raw['m2'].empty:
    st.error("❌ BTC 또는 M2 CSV 파일을 찾을 수 없습니다.")
    st.stop()

# -----------------------------------------------------------
# [ENGINE] GM v3.7 Core Logic
# -----------------------------------------------------------
def run_strategy(raw, base_vol, max_lev):
    # -------------------------------------------------------
    # 1. Alignment
    # -------------------------------------------------------
    start_dt = max(raw['btc'].index.min(), raw['m2'].index.min())
    end_dt = raw['btc'].index.max()
    idx = pd.date_range(start=start_dt, end=end_dt, freq='D')

    price = (
        raw['btc']
        .reindex(idx)
        .interpolate(method='time')
        .ffill()
        .bfill()
    )

    m2 = raw['m2'].reindex(idx).ffill().bfill()

    if raw['spread'].empty:
        spread = pd.Series(3.5, index=idx)
    else:
        spread = raw['spread'].reindex(idx).ffill().bfill()

    # -------------------------------------------------------
    # 2. Regime Lag (Spread)
    # -------------------------------------------------------
    spread_z = (spread - spread.rolling(365).mean()) / (spread.rolling(365).std() + 1e-9)
    lag = np.where(spread_z.shift(1) < 0.5, 56, 168)

    # -------------------------------------------------------
    # 3. Valuation (Kalman / RLS)
    # -------------------------------------------------------
    m2_vals = m2.values
    shifted_m2 = []

    for i in range(len(idx)):
        li = i - int(lag[i])
        shifted_m2.append(m2_vals[li] if li >= 0 else m2_vals[0])

    x = np.log(np.array(shifted_m2))
    y = np.log(price.values)

    lam = 0.9995
    theta = np.array([0.0, 1.0])
    P = np.eye(2) * 100

    residuals, betas = [], []

    X = np.column_stack([np.ones(len(idx)), x])

    for i in range(len(idx)):
        xi = X[i].reshape(-1, 1)
        yi = y[i]

        pred = float(xi.T @ theta)
        err = yi - pred

        residuals.append(err)
        betas.append(theta[1])

        if not np.isnan(err):
            K = (P @ xi) / (lam + xi.T @ P @ xi)
            theta = theta + (K.flatten() * err)
            P = (P - K @ xi.T @ P) / lam

    # -------------------------------------------------------
    # 4. Indicators
    # -------------------------------------------------------
    df = pd.DataFrame(index=idx)
    df['Price'] = price
    df['Residual'] = residuals
    df['Z_Gap'] = (
        (df['Residual'] - df['Residual'].rolling(730, min_periods=90).mean()) /
        df['Residual'].rolling(730, min_periods=90).std()
    )

    df['Beta'] = betas
    df['Beta_MA'] = df['Beta'].rolling(60).mean()
    df['Beta_Up'] = df['Beta'] > df['Beta_MA']

    ret = price.pct_change()
    df['Vol_30'] = ret.rolling(30).std() * np.sqrt(365) * 100

    df['MA200'] = df['Price'].rolling(200).mean()

    # -------------------------------------------------------
    # 5. Final Decision (Last Day)
    # -------------------------------------------------------
    last = df.iloc[-1]

    curr_vol = last['Vol_30'] if not np.isnan(last['Vol_30']) else 50.0
    z_val = last['Z_Gap'] if not np.isnan(last['Z_Gap']) else 0.0
    is_trend = last['Price'] > last['MA200']

    status = ""
    color = "normal"
    exposure = 0.0
    target_vol = base_vol
    val_mod = 1.0

    if not is_trend:
        status = "❄️ HIBERNATION (Cash)"
        color = "off"
    else:
        if last['Beta_Up']:
            target_vol = base_vol + 10
            status = "🔥 HIGH BETA (Aggressive)"
            color = "inverse"
        else:
            target_vol = base_vol - 10
            status = "🛡️ LOW BETA (Defensive)"

        scalar = min(target_vol / max(curr_vol, 1.0), max_lev)

        if z_val > 2.0:
            val_mod = 0.7
            status += " + TRIM"
        elif z_val < -2.0:
            val_mod = 1.3
            status += " + BOOST"

        exposure = min(scalar * val_mod, max_lev)

    return {
        "exposure": exposure,
        "status": status,
        "color": color,
        "curr_vol": curr_vol,
        "target_vol": target_vol,
        "z_gap": z_val,
        "beta_up": bool(last['Beta_Up']),
        "df": df
    }

# -----------------------------------------------------------
# [RUN ENGINE]
# -----------------------------------------------------------
res = run_strategy(raw, base_target_vol, max_lev_cap)

# -----------------------------------------------------------
# [DASHBOARD]
# -----------------------------------------------------------
st.markdown("### 🧭 Institutional Command Center")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Final Exposure", f"{res['exposure']*100:.0f}%", res['status'], delta_color=res['color'])
c2.metric("Risk Budget", f"{res['target_vol']}%", f"Current Vol: {res['curr_vol']:.1f}%")
c3.metric("Valuation Tilt", f"x{1 if abs(res['z_gap']) < 2 else ('0.7' if res['z_gap'] > 2 else '1.3')}", f"Z: {res['z_gap']:.2f}")
c4.metric("Liquidity Efficiency", "🚀" if res['beta_up'] else "🐌")

st.divider()

# -----------------------------------------------------------
# [CHARTS] (Visualization-safe)
# -----------------------------------------------------------
df_plot = res['df'].copy()
df_plot = df_plot.dropna(subset=['Vol_30', 'Z_Gap'])

if df_plot.index.max() >= pd.Timestamp("2020-01-01"):
    df_plot = df_plot[df_plot.index >= "2020-01-01"]

tab1, tab2 = st.tabs(["🛡️ Risk Control", "📊 Valuation Signal"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Vol_30'], name="Market Vol"))
    fig.add_hline(y=res['target_vol'], line_dash="dash", annotation_text="Target Vol")
    fig.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    colors = [
        'red' if z > 2 else 'lime' if z < -2 else 'gray'
        for z in df_plot['Z_Gap']
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['Z_Gap'], marker_color=colors))
    fig.add_hline(y=2, line_dash="dash", line_color="red")
    fig.add_hline(y=-2, line_dash="dash", line_color="lime")
    fig.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig, use_container_width=True)
