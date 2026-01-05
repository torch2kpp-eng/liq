import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import io
import warnings
import time
import ccxt
import numpy as np
from datetime import datetime, timedelta

# -----------------------------------------------------------
# [환경 설정]
# -----------------------------------------------------------
warnings.filterwarnings("ignore")
st.set_page_config(page_title="GM Strategy v3.7", layout="wide")

st.title("🏛️ Grand Master: Adaptive Alpha Engine")
st.caption("v3.7 | Adaptive Risk Budgeting | Valuation Tilt | Institutional Grade")

# -----------------------------------------------------------
# [사이드바] Strategy Tuning
# -----------------------------------------------------------
st.sidebar.header("⚙️ Engine Tuning")
base_target_vol = st.sidebar.slider("🎯 Base Target Vol (%)", 30, 80, 50, 5)
max_lev_cap = st.sidebar.slider("🔒 Max Leverage Limit", 1.0, 3.0, 2.0, 0.1)

# -----------------------------------------------------------
# [CORE] 데이터 파이프라인
# -----------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_data():
    d = {}
    headers = {'User-Agent': 'Mozilla/5.0'}

    def get_fred(id):
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"
            r = requests.get(url, headers=headers, timeout=5)
            df = pd.read_csv(io.StringIO(r.text), index_col=0, parse_dates=True)
            s = df.squeeze().apply(pd.to_numeric, errors='coerce')
            ext_idx = pd.date_range(start=s.index.min(), end=datetime.now(), freq='D')
            s_ext = s.reindex(ext_idx)
            return s_ext.fillna(method='ffill').fillna(method='bfill')
        except: return pd.Series(dtype=float)

    def fetch_btc():
        try:
            # Production: Use real exchange data
            bithumb = ccxt.bithumb({'enableRateLimit': True, 'timeout': 3000})
            ohlcv = bithumb.fetch_ohlcv('BTC/KRW', '1d', limit=1000)
            if not ohlcv: return pd.Series(dtype=float)
            df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df.set_index('timestamp')['close'].tz_localize(None)
        except: return pd.Series(dtype=float)

    d['m2'] = get_fred('M2SL')
    d['spread'] = get_fred('BAMLH0A0HYM2')
    d['btc'] = fetch_btc()
    return d

with st.spinner("📡 월가 적응형 엔진(Adaptive Engine) 가동 중..."):
    raw = fetch_data()

# -----------------------------------------------------------
# [ENGINE] GM v3.7 Logic
# -----------------------------------------------------------
def run_strategy_v3_7(price, m2, spread, base_vol, max_lev):
    try:
        # 1. Alignment
        idx = price.index
        m2_d = m2.reindex(idx).fillna(method='ffill')
        spread_d = spread.reindex(idx).fillna(method='ffill').fillna(3.5)
        
        # 2. Regime Lag (Bias Free)
        spread_z = (spread_d - spread_d.rolling(365).mean()) / (spread_d.rolling(365).std() + 1e-9)
        regime_sig = spread_z.shift(1)
        lag_vals = np.where(regime_sig < 0.5, 56, 168)
        
        # 3. Valuation (Kalman Filter)
        m2_vals = m2_d.values
        shifted_m2 = []
        N = len(idx)
        for i in range(N):
            l_idx = int(i - lag_vals[i])
            shifted_m2.append(m2_vals[l_idx] if l_idx >=0 else m2_vals[0])
            
        x_liq = np.log(np.array(shifted_m2))
        y_price = np.log(price.values)
        
        lam = 0.9995
        theta = np.zeros(2); theta[1] = 1.0
        P = np.eye(2) * 100
        residuals, betas = [], []
        
        X = np.column_stack([np.ones(N), x_liq])
        for i in range(N):
            xi = X[i, :].reshape(-1, 1)
            yi = y_price[i]
            pred = np.dot(xi.T, theta).item()
            err = yi - pred
            residuals.append(err)
            betas.append(theta[1])
            if not np.isnan(err):
                num = np.dot(P, xi)
                den = lam + np.dot(xi.T, np.dot(P, xi))
                K = num / den
                theta = theta + (K * err).flatten()
                P = (P - np.dot(K, np.dot(xi.T, P))) / lam
        
        # 4. Signal Construction
        df = pd.DataFrame(index=idx)
        df['Price'] = price
        df['Z_Gap'] = (pd.Series(residuals, index=idx) - pd.Series(residuals, index=idx).rolling(730).mean()) / pd.Series(residuals, index=idx).rolling(730).std()
        
        # Beta Trend (For Adaptive Vol)
        df['Beta'] = betas
        df['Beta_MA'] = df['Beta'].rolling(60).mean()
        df['Beta_Up'] = df['Beta'] > df['Beta_MA']
        
        # Volatility
        daily_ret = price.pct_change()
        vol_30 = daily_ret.rolling(30).std() * np.sqrt(365) * 100
        
        # Trend
        ma200 = price.rolling(200).mean()
        is_trend = price.iloc[-1] > ma200.iloc[-1]
        
        # 5. Final Decision Logic (Last Day)
        curr_vol = vol_30.iloc[-1]
        if pd.isna(curr_vol) or curr_vol < 1.0: curr_vol = 50.0
        
        z_val = df['Z_Gap'].iloc[-1]
        is_beta_up = df['Beta_Up'].iloc[-1]
        
        status_msg = ""
        status_color = "normal"
        base_dir = 1.0
        
        if not is_trend:
            final_exp = 0.0
            status_msg = "❄️ HIBERNATION (Cash)"
            status_color = "off"
            adaptive_target = 0.0
            val_mod = 1.0
        else:
            # A. Adaptive Risk Budgeting
            if is_beta_up:
                adaptive_target = base_vol + 10.0 # Aggressive
                status_msg = "🔥 HIGH BETA (Aggressive)"
                status_color = "inverse"
            else:
                adaptive_target = base_vol - 10.0 # Defensive
                status_msg = "🛡️ LOW BETA (Defensive)"
                status_color = "normal"
            
            # B. Risk Scalar
            risk_scalar = adaptive_target / curr_vol
            risk_scalar = min(risk_scalar, max_lev)
            
            # C. Valuation Modifier (Tilt)
            val_mod = 1.0
            if z_val > 2.0: 
                val_mod = 0.7 # Trim
                status_msg += " + TRIM (Overheat)"
            elif z_val < -2.0: 
                val_mod = 1.3 # Boost
                status_msg += " + BOOST (Value)"
                
            # Final Calculation
            final_exp = base_dir * risk_scalar * val_mod
            final_exp = min(final_exp, max_lev)
            
        return {
            "exposure": final_exp,
            "status": status_msg,
            "color": status_color,
            "z_gap": z_val,
            "beta_up": is_beta_up,
            "curr_vol": curr_vol,
            "target_vol": adaptive_target,
            "val_mod": val_mod,
            "df": pd.DataFrame({'Price': price, 'MA200': ma200, 'Z_Gap': df['Z_Gap'], 'Beta': df['Beta']})
        }
        
    except Exception as e: return None

# -----------------------------------------------------------
# [DASHBOARD]
# -----------------------------------------------------------
if 'btc' in raw and not raw['m2'].empty:
    res = run_strategy_v3_7(raw['btc'], raw['m2'], raw['spread'], base_target_vol, max_lev_cap)
    
    if res:
        st.markdown("### 🧭 Institutional Command Center")
        
        # Top Metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Final Exposure", f"{res['exposure']*100:.0f}%", res['status'], delta_color=res['color'])
        with c2:
            st.metric("Adaptive Target Vol", f"{res['target_vol']}%", f"Current Vol: {res['curr_vol']:.1f}%")
        with c3:
            st.metric("Valuation Tilt", f"x{res['val_mod']}", f"Z-Gap: {res['z_gap']:.2f} σ")
        with c4:
            icon = "🚀" if res['beta_up'] else "🐌"
            st.metric("Transmission (Beta)", icon, "Liquidity Efficiency")
            
        st.divider()
        
        # Charts
        tab1, tab2 = st.tabs(["🛡️ Exposure Logic", "📊 Valuation Signal"])
        
        df_viz = res['df'][res['df'].index >= '2020-01-01']
        
        with tab1:
            # Beta Trend Visual
            st.caption("Beta Trend (Transmission Efficiency): If Rising, we target higher volatility.")
            st.line_chart(df_viz['Beta'])
            
        with tab2:
            fig = go.Figure()
            cols = []
            for i in range(len(df_viz)):
                z = df_viz['Z_Gap'].iloc[i]
                if z > 2.0: cols.append('red')
                elif z < -2.0: cols.append('lime')
                else: cols.append('gray')
            fig.add_trace(go.Bar(x=df_viz.index, y=df_viz['Z_Gap'], marker_color=cols))
            fig.add_hline(y=2.0, line_dash="dash", line_color="red")
            fig.add_hline(y=-2.0, line_dash="dash", line_color="lime")
            fig.update_layout(title="Z-Gap History", template="plotly_dark", height=450)
            st.plotly_chart(fig, use_container_width=True)

    else: st.error("엔진 오류")
else: st.info("데이터 로딩 중...")
