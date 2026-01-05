import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os
from datetime import datetime

# -----------------------------------------------------------
# [환경 설정]
# -----------------------------------------------------------
st.set_page_config(page_title="GM Strategy v3.7", layout="wide")
st.title("🏛️ Grand Master: Adaptive Alpha Engine")
st.caption("v3.7 (Local Data Optimized) | Risk Parity | Valuation Tilt")

# -----------------------------------------------------------
# [사이드바] Strategy Tuning
# -----------------------------------------------------------
st.sidebar.header("⚙️ Engine Tuning")
base_target_vol = st.sidebar.slider("🎯 Base Target Vol (%)", 30, 80, 50, 5)
max_lev_cap = st.sidebar.slider("🔒 Max Leverage Limit", 1.0, 3.0, 2.0, 0.1)

# -----------------------------------------------------------
# [CORE] 데이터 로더 (로컬 파일 우선)
# -----------------------------------------------------------
@st.cache_data(ttl=3600)
def load_data():
    data = {}
    
    # 1. 파일 로드 함수 (공통)
    def read_csv_data(filenames):
        for f in filenames:
            if os.path.exists(f):
                try:
                    df = pd.read_csv(f, parse_dates=['observation_date'], index_col='observation_date')
                    # 숫자로 변환 (에러 방지)
                    s = df.iloc[:, 0].apply(pd.to_numeric, errors='coerce')
                    return s.sort_index()
                except: continue
        return pd.Series(dtype=float)

    # 2. BTC 데이터 로드
    # 업로드해주신 파일명 후보들
    data['btc'] = read_csv_data(['CBBTCUSD 2.csv', 'CBBTCUSD.csv', 'btc.csv'])
    
    # 3. M2 데이터 로드
    data['m2'] = read_csv_data(['M2SL 2.csv', 'M2SL.csv', 'm2.csv'])
    
    # 4. Spread 데이터 로드 (없으면 3.5 고정)
    # Spread 파일이 없으므로, 기본값 생성 혹은 파일이 있다면 로드
    s_spread = read_csv_data(['BAMLH0A0HYM2.csv', 'spread.csv'])
    if s_spread.empty:
        # 데이터가 없으면 BTC 기간에 맞춰 3.5(중립)로 채움
        if not data['btc'].empty:
            idx = data['btc'].index
            data['spread'] = pd.Series(3.5, index=idx)
        else:
            data['spread'] = pd.Series(dtype=float)
    else:
        data['spread'] = s_spread

    return data

# 데이터 로딩 실행
raw = load_data()

# 데이터 상태 점검 메시지
if raw['btc'].empty:
    st.error("❌ 'CBBTCUSD 2.csv' 파일을 찾을 수 없습니다.")
    st.stop()
if raw['m2'].empty:
    st.error("❌ 'M2SL 2.csv' 파일을 찾을 수 없습니다.")
    st.stop()

# -----------------------------------------------------------
# [ENGINE] GM v3.7 Logic (Robust)
# -----------------------------------------------------------
def run_strategy_engine(raw_data, base_vol, max_lev):
    try:
        # 1. Alignment (전체 기간 통합)
        # BTC와 M2 중 더 긴 기간을 커버하도록 인덱스 생성
        start_dt = max(raw_data['btc'].index.min(), raw_data['m2'].index.min())
        end_dt = raw_data['btc'].index.max() # BTC 마지막 날짜 기준
        
        idx = pd.date_range(start=start_dt, end=end_dt, freq='D')
        
        # Reindex & Fill
        # BTC: 주말/휴일은 보간(Time Interpolation)
        price = raw_data['btc'].reindex(idx).interpolate(method='time').fillna(method='ffill').fillna(method='bfill')
        # M2: 월간 데이터이므로 ffill (직전 값 유지)
        m2_d = raw_data['m2'].reindex(idx).fillna(method='ffill').fillna(method='bfill')
        # Spread
        spread_d = raw_data['spread'].reindex(idx).fillna(method='ffill').fillna(3.5)

        # 2. Indicators Calculation
        
        # A. Regime Lag (Bias Free)
        # Spread Z-Score
        roll_mean = spread_d.rolling(365).mean()
        roll_std = spread_d.rolling(365).std()
        spread_z = (spread_d - roll_mean) / (roll_std + 1e-9)
        regime_sig = spread_z.shift(1) # 하루 전 신호 사용
        lag_vals = np.where(regime_sig < 0.5, 56, 168)
        
        # B. Valuation (Kalman Filter)
        m2_vals = m2_d.values
        shifted_m2 = []
        N = len(idx)
        for i in range(N):
            l_idx = int(i - lag_vals[i])
            # 인덱스 범위를 벗어나면 첫 번째 값 사용
            val = m2_vals[l_idx] if l_idx >= 0 else m2_vals[0]
            shifted_m2.append(val)
            
        x_liq = np.log(np.array(shifted_m2))
        y_price = np.log(price.values)
        
        # RLS Algorithm
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

        # DataFrame 구성
        df = pd.DataFrame(index=idx)
        df['Price'] = price
        df['Residual'] = residuals
        df['Z_Gap'] = (df['Residual'] - df['Residual'].rolling(730, min_periods=90).mean()) / df['Residual'].rolling(730, min_periods=90).std()
        df['Beta'] = betas
        
        # C. Beta Trend
        df['Beta_MA'] = df['Beta'].rolling(60).mean()
        df['Beta_Up'] = df['Beta'] > df['Beta_MA']
        
        # D. Volatility
        daily_ret = price.pct_change()
        df['Vol_30'] = daily_ret.rolling(30).std() * np.sqrt(365) * 100
        
        # E. Trend Filter
        df['MA200'] = df['Price'].rolling(200).mean()
        
        # 3. Final Decision (마지막 날짜 기준)
        last = df.iloc[-1]
        
        # 안전장치: 데이터가 NaN이면 기본값 처리
        curr_vol = last['Vol_30'] if not pd.isna(last['Vol_30']) else 50.0
        z_val = last['Z_Gap'] if not pd.isna(last['Z_Gap']) else 0.0
        is_trend = last['Price'] > last['MA200']
        
        # 로직 시작
        status_msg = ""
        status_color = "normal"
        adaptive_target = base_vol
        val_mod = 1.0
        final_exp = 0.0
        
        if not is_trend:
            final_exp = 0.0
            status_msg = "❄️ HIBERNATION (Cash)"
            status_color = "off"
        else:
            # Beta Driven Target Vol
            if last['Beta_Up']:
                adaptive_target = base_vol + 10.0
                status_msg = "🔥 HIGH BETA (Aggressive)"
                status_color = "inverse"
            else:
                adaptive_target = base_vol - 10.0
                status_msg = "🛡️ LOW BETA (Defensive)"
            
            # Risk Scalar
            if curr_vol < 1.0: curr_vol = 50.0 # 0 나누기 방지
            scalar = adaptive_target / curr_vol
            scalar = min(scalar, max_lev)
            
            # Valuation Modifier
            if z_val > 2.0: 
                val_mod = 0.7
                status_msg += " + TRIM"
            elif z_val < -2.0: 
                val_mod = 1.3
                status_msg += " + BOOST"
                
            final_exp = scalar * val_mod
            final_exp = min(final_exp, max_lev)
            
        return {
            "exposure": final_exp,
            "status": status_msg,
            "color": status_color,
            "z_gap": z_val,
            "curr_vol": curr_vol,
            "target_vol": adaptive_target,
            "val_mod": val_mod,
            "beta_up": bool(last['Beta_Up']),
            "df": df
        }
        
    except Exception as e:
        st.error(f"엔진 오류 발생: {e}")
        return None

# -----------------------------------------------------------
# [DASHBOARD]
# -----------------------------------------------------------
res = run_strategy_engine(raw, base_target_vol, max_lev_cap)

if res:
    st.markdown("### 🧭 Institutional Command Center")
    
    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Final Exposure", f"{res['exposure']*100:.0f}%", res['status'], delta_color=res['color'])
    with c2:
        st.metric("Risk Budget", f"{res['target_vol']}% Vol", f"Current: {res['curr_vol']:.1f}%")
    with c3:
        st.metric("Valuation Tilt", f"x{res['val_mod']}", f"Z-Gap: {res['z_gap']:.2f} σ")
    with c4:
        icon = "🚀" if res['beta_up'] else "🐌"
        st.metric("Liquidity Efficiency", icon, "Beta Trend")
        
    st.divider()
    
    # Charts
    tab1, tab2 = st.tabs(["🛡️ Strategy Logic", "📊 Valuation Signal"])
    
    df_viz = res['df'][res['df'].index >= '2020-01-01']
    
    with tab1:
        # Volatility & Beta
        fig_v = go.Figure()
        fig_v.add_trace(go.Scatter(x=df_viz.index, y=df_viz['Vol_30'], name='Market Vol', line=dict(color='yellow')))
        fig_v.add_hline(y=res['target_vol'], line_dash="dash", line_color="magenta", annotation_text="Target Vol")
        fig_v.update_layout(title="Volatility vs Target (Risk Control)", template="plotly_dark", height=400)
        st.plotly_chart(fig_v, use_container_width=True)
        
    with tab2:
        # Z-Gap
        fig_z = go.Figure()
        cols = []
        for i in range(len(df_viz)):
            z = df_viz['Z_Gap'].iloc[i]
            if z > 2.0: cols.append('red')
            elif z < -2.0: cols.append('lime')
            else: cols.append('gray')
        
        fig_z.add_trace(go.Bar(x=df_viz.index, y=df_viz['Z_Gap'], marker_color=cols, name='Z-Gap'))
        fig_z.add_hline(y=2.0, line_dash="dash", line_color="red")
        fig_z.add_hline(y=-2.0, line_dash="dash", line_color="lime")
        fig_z.update_layout(title="Z-Gap Signal History", template="plotly_dark", height=400)
        st.plotly_chart(fig_z, use_container_width=True)

else:
    st.warning("데이터 계산 중 문제가 발생했습니다. 데이터 파일 기간을 확인해주세요.")
