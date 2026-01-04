import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import json
import io
import warnings
import time
import ccxt
import numpy as np
from datetime import date, timedelta

# 1. 환경 설정
warnings.filterwarnings("ignore")
st.set_page_config(page_title="GM Level Hunter", layout="wide")

st.title("🏛️ Grand Master: Analytics Engine")
st.caption("Ver 20.0 | Log-Level Z-Gap 엔진 탑재 | 직관적 사이클 분석 (High Price = High Gap)")

# -----------------------------------------------------------
# [사이드바 설정]
# -----------------------------------------------------------
st.sidebar.header("⚙️ Control Panel")
is_mobile = st.sidebar.checkbox("📱 모바일 모드 (축 공간 최소화)", value=True)

# [Stress Test 옵션]
st.sidebar.markdown("---")
st.sidebar.subheader("📉 Crash Simulation")
spike_threshold = st.sidebar.slider("위험 감지 민감도 (bps)", 5, 50, 15)
look_forward_days = st.sidebar.slider("반응 관찰 기간 (Days)", 1, 30, 7)
col_d1, col_d2 = st.sidebar.columns(2)
with col_d1: sim_start_date = st.date_input("시작일", value=date(2019, 1, 1))
with col_d2: sim_end_date = st.date_input("종료일", value="today")

st.sidebar.markdown("---")
liq_option = st.sidebar.radio(
    "1. 유동성 지표 (Left Axis)",
    ("🇺🇸 Fed Net Liquidity", "🏛️ G3 Central Bank Assets", "🌍 Global M2"),
    index=2
)

st.sidebar.markdown("---")
shift_days = st.sidebar.number_input(
    "2. Time Shift (Days)", min_value=-365, max_value=365, value=112, step=7,
    help="0이면 자산별 최적 시차 자동 적용, 그 외엔 설정값 강제 적용"
)

st.sidebar.markdown("---")
st.sidebar.write("3. 표시할 자산 (Right Axes)")

ASSETS_CONFIG = [
    {'id': 'hy_spread', 'name': '⚡ HY Spread', 'symbol': 'BAMLH0A0HYM2', 'source': 'fred', 'color': '#E040FB', 'type': 'risk', 'default': True},
    {'id': 'nasdaq', 'name': 'Nasdaq', 'symbol': 'IXIC', 'source': 'hybrid', 'color': '#D62780', 'type': 'index', 'default': False},
    {'id': 'btc',    'name': 'BTC',    'symbol': 'BTC/KRW', 'source': 'bithumb', 'color': '#00FFEE', 'type': 'crypto', 'default': True},
    {'id': 'gold',   'name': 'Gold',   'symbol': 'GC=F', 'source': 'hybrid_metal', 'color': '#FFD700', 'type': 'metal', 'default': False},
    {'id': 'silver', 'name': 'Silver', 'symbol': 'SI=F', 'source': 'hybrid_metal', 'color': '#C0C0C0', 'type': 'metal', 'default': False},
    {'id': 'eth',    'name': 'ETH',    'symbol': 'ETH/KRW', 'source': 'bithumb', 'color': '#627EEA', 'type': 'crypto', 'default': False},
    {'id': 'doge',   'name': 'DOGE',   'symbol': 'DOGE/KRW', 'source': 'bithumb', 'color': '#FFA500', 'type': 'crypto', 'default': False},
    {'id': 'link',   'name': 'LINK',   'symbol': 'LINK/KRW', 'source': 'bithumb', 'color': '#2A5ADA', 'type': 'crypto', 'default': False},
    {'id': 'ada',    'name': 'ADA',    'symbol': 'ADA/KRW', 'source': 'bithumb', 'color': '#0033AD', 'type': 'crypto', 'default': False},
    {'id': 'xrp',    'name': 'XRP',    'symbol': 'XRP/KRW', 'source': 'bithumb', 'color': '#00AAE4', 'type': 'crypto', 'default': False},
]

selected_assets = {}
for asset in ASSETS_CONFIG:
    selected_assets[asset['id']] = st.sidebar.checkbox(f"{asset['name']}", value=asset['default'])

# -----------------------------------------------------------
# 데이터 수집 Logic
# -----------------------------------------------------------
def fetch_master_data_logic():
    d = {}
    meta_info = {}
    GLOBAL_START = time.time()
    MAX_EXECUTION_TIME = 30 
    START_YEAR = 2016 
    headers = {'User-Agent': 'Mozilla/5.0'}

    def check_timeout(): return (time.time() - GLOBAL_START > MAX_EXECUTION_TIME)

    def get_fred(id):
        if check_timeout(): return pd.Series(dtype=float)
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"
            r = requests.get(url, headers=headers, timeout=5)
            df = pd.read_csv(io.StringIO(r.text), index_col=0, parse_dates=True)
            s = df.squeeze().apply(pd.to_numeric, errors='coerce')
            return s.resample('D').interpolate(method='time').tz_localize(None)
        except: return pd.Series(dtype=float)

    def get_yahoo(ticker):
        if check_timeout(): return pd.Series(dtype=float)
        try:
            import yfinance as yf
            df = yf.download(ticker, start=f"{START_YEAR}-01-01", progress=False, auto_adjust=True)
            if not df.empty:
                s = df['Close'] if 'Close' in df.columns else df.iloc[:,0]
                if isinstance(s, pd.DataFrame): s = s.iloc[:, 0]
                s = s.squeeze().tz_localize(None)
                if isinstance(s.index, pd.DatetimeIndex):
                    return s.resample('D').interpolate(method='time')
            return pd.Series(dtype=float)
        except: return pd.Series(dtype=float)

    def get_metal_hybrid(symbol):
        if check_timeout(): return pd.Series(dtype=float), "Timeout"
        data = get_yahoo(symbol)
        if not data.empty and len(data) > 10: return data, "Futures"
        backup = "GLD" if "GC" in symbol else "SLV"
        data_b = get_yahoo(backup)
        if not data_b.empty: return data_b, "ETF(Backup)"
        return pd.Series(dtype=float), "Fail"

    bithumb = ccxt.bithumb({'enableRateLimit': True, 'timeout': 3000})
    def fetch_bithumb(symbol_code):
        if check_timeout(): return pd.Series(dtype=float)
        all_data = []
        try:
            since = bithumb.parse8601(f'{START_YEAR}-01-01T00:00:00Z')
            for _ in range(20): 
                if check_timeout(): break
                ohlcv = bithumb.fetch_ohlcv(symbol_code, '1d', since=since, limit=1000)
                if not ohlcv: break
                all_data.extend(ohlcv)
                last_ts = ohlcv[-1][0]
                if last_ts >= (time.time() * 1000) - 86400000: break
                since = last_ts + 1
                time.sleep(0.05)
        except: pass
        if not all_data: return pd.Series(dtype=float)
        df = pd.DataFrame(all_data, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df.drop_duplicates('timestamp').set_index('timestamp')['close'].tz_localize(None)

    status_text = st.empty()
    status_text.text("📡 Initializing Data (2016~)...")

    fred_ids = {
        'fed': 'WALCL', 'tga': 'WTREGEN', 'rrp': 'RRPONTSYD',
        'ecb': 'ECBASSETSW', 'boj': 'JPNASSETS',
        'm2_us': 'M2SL', 'm3_eu': 'MABMM301EZM189S', 'm3_jp': 'MABMM301JPM189S',
        'eur_usd': 'DEXUSEU', 'usd_jpy': 'DEXJPUS',
        'nasdaq_fred': 'NASDAQCOM'
    }
    
    for k, v in fred_ids.items():
        if check_timeout(): break
        d[k] = get_fred(v)

    if not d.get('nasdaq_fred', pd.Series()).empty: d['nasdaq'] = d['nasdaq_fred']
    else: d['nasdaq'] = get_yahoo("^IXIC")

    active_ids = [a['id'] for a in ASSETS_CONFIG if selected_assets[a['id']]]
    for asset in ASSETS_CONFIG:
        if asset['id'] not in active_ids: continue
        if check_timeout(): continue
        if asset['id'] == 'nasdaq': continue
        
        if asset['source'] == 'fred': d[asset['id']] = get_fred(asset['symbol'])
        elif asset['source'] == 'hybrid_metal':
            data, src = get_metal_hybrid(asset['symbol'])
            d[asset['id']] = data
            meta_info[asset['id']] = src
        elif asset['source'] == 'yahoo': d[asset['id']] = get_yahoo(asset['symbol'])
        elif asset['source'] == 'bithumb': d[asset['id']] = fetch_bithumb(asset['symbol'])
        
    status_text.empty()
    return d, meta_info

raw, meta = fetch_master_data_logic()

# -----------------------------------------------------------
# [FUNC 1] Risk Radar Logic
# -----------------------------------------------------------
def check_risk_radar(hy_series):
    if hy_series.empty: return None
    last_val = hy_series.iloc[-1]
    prev_val = hy_series.iloc[-2]
    ma_20 = hy_series.rolling(20).mean().iloc[-1]
    daily_chg_pct = (last_val - prev_val) / prev_val * 100
    daily_chg_bps = (last_val - prev_val) * 100
    trend_break = last_val > ma_20
    is_danger_zone = last_val > 4.0
    status, color, msg = "Normal", "green", "안정 (Risk-On)"
    if daily_chg_pct > 5.0 or (trend_break and daily_chg_pct > 2.0):
        status, color, msg = "Warning", "orange", "⚠️ 급등 감지 (Warning)"
    if is_danger_zone:
        status, color, msg = "Danger", "red", "🚨 위험 지역 (Risk-Off)"
    return {"val": last_val, "daily_chg_bps": daily_chg_bps, "status": status, "color": color, "msg": msg}

# -----------------------------------------------------------
# [FUNC 2] Quant Engine (Log-Level Update)
# -----------------------------------------------------------
def run_quant_analysis_pure(liq_series_raw, asset_series_raw, manual_lag_days):
    try:
        # [NEW Logic] YoY가 아니라 Log Level을 사용
        # 그래야 "가격이 높으면 Z-Gap도 높다"는 직관과 일치함
        
        # 1. 전처리 (Weekly Resample)
        liq_weekly = liq_series_raw.resample('W-WED').last()
        asset_weekly = asset_series_raw.resample('W-WED').last()
        
        # 2. Log 변환 (데이터 정규화를 위해 필수)
        # 0이나 음수가 있을 수 있으므로 처리
        liq_log = np.log(liq_weekly.replace(0, np.nan).dropna())
        asset_log = np.log(asset_weekly.replace(0, np.nan).dropna())
        
        # 3. 데이터 병합
        df = pd.concat([liq_log, asset_log], axis=1).dropna()
        df.columns = ['L_Log', 'P_Log']
        if len(df) < 52: return None
        
        # 4. Smoothing (노이즈 제거)
        df['L_Smooth'] = df['L_Log'].rolling(4).mean()
        df['P_Smooth'] = df['P_Log'].rolling(4).mean()
        df = df.dropna()
        
        # 5. Z-Score 산출 (전체 기간 기준 정규화)
        # 이것이 핵심입니다. "과거 평균 대비 현재 레벨이 얼마나 높은가?"
        df['L_Z'] = (df['L_Smooth'] - df['L_Smooth'].mean()) / (df['L_Smooth'].std() + 1e-9)
        df['P_Z'] = (df['P_Smooth'] - df['P_Smooth'].mean()) / (df['P_Smooth'].std() + 1e-9)

        # 6. Correlation Check (Lag 찾기용으로는 차분(Diff) 데이터 사용이 더 정확함)
        # 레벨 변수끼리는 무조건 상관계수가 높게 나오므로(Spurious Regression),
        # Lag를 찾을 때만 잠시 1주 차분 데이터를 씁니다.
        df_diff = df[['L_Z', 'P_Z']].diff().dropna()
        best_lag_weeks, best_corr = 0, -1.0
        for lag in range(0, 53): 
            corr = df_diff['P_Z'].corr(df_diff['L_Z'].shift(lag))
            if corr > best_corr: best_corr, best_lag_weeks = corr, lag
            
        # 7. Lag 적용 및 Gap 계산
        if manual_lag_days != 0:
            calc_lag_weeks = int(manual_lag_days / 7)
            used_mode = "Manual"
        else:
            calc_lag_weeks = best_lag_weeks
            used_mode = "Auto"
        
        # 유동성 지표를 Shift (미래로 밀어서 현재 가격과 매칭)
        df['L_Z_Shifted'] = df['L_Z'].shift(calc_lag_weeks)
        
        # 최근 데이터 확인
        df_recent = df.iloc[-4:]
        
        # 최근 상관관계 (레벨 기준)
        recent_corr = df_recent['P_Z'].corr(df_recent['L_Z_Shifted'])
        last_val = df.iloc[-1]
        
        # [핵심] Z-Gap = Price Z - Liquidity Z
        gap_z = last_val['P_Z'] - last_val['L_Z_Shifted']
        
        # Regime 판단
        if best_corr < 0: regime = "Inverse" # 장기 역상관
        elif gap_z > 1.5: regime = "Overheat" # 과열
        elif gap_z < -1.5: regime = "Undervalued" # 저평가
        else: regime = "Fair"

        return {
            "optimal_lag": best_lag_weeks * 7, 
            "calc_lag": calc_lag_weeks * 7,    
            "mode": used_mode,
            "global_corr": best_corr,
            "recent_corr": recent_corr,
            "gap_z": gap_z, 
            "regime": regime
        }
    except Exception: return None

# -----------------------------------------------------------
# [FUNC 3] Stress Test
# -----------------------------------------------------------
def run_stress_test(hy_series, btc_series, threshold_bps, look_forward, start_d, end_d):
    try:
        hy, btc = hy_series.copy(), btc_series.copy()
        hy.index, btc.index = hy.index.normalize(), btc.index.normalize()
        s_date, e_date = pd.to_datetime(start_d).normalize(), pd.to_datetime(end_d).normalize()
        df = pd.concat([hy, btc], axis=1).dropna()
        df.columns = ['Spread', 'Price']
        df = df[(df.index >= s_date) & (df.index <= e_date)]
        if df.empty: return pd.DataFrame()
        df['Spread_Chg_Bps'] = df['Spread'].diff() * 100
        events = df[df['Spread_Chg_Bps'] >= threshold_bps].index
        results = []
        for date in events:
            target_date = date + timedelta(days=look_forward)
            if target_date <= btc.index[-1]:
                try:
                    price_at_signal = df.loc[date]['Price']
                    future_data = btc[btc.index >= target_date]
                    if not future_data.empty:
                        price_future = future_data.iloc[0]
                        price_chg_pct = (price_future - price_at_signal) / price_at_signal * 100
                        outcome = "🛡️ 방어 성공" if price_chg_pct < 0 else "🎣 휩쏘 (False)"
                        results.append({
                            "Date": date.strftime("%Y-%m-%d"),
                            "Spike": f"+{df.loc[date]['Spread_Chg_Bps']:.1f} bps",
                            "Raw_Return": price_chg_pct,
                            "BTC Return": f"{price_chg_pct:+.2f}%",
                            "Outcome": outcome
                        })
                except: continue
        return pd.DataFrame(results).sort_values("Date", ascending=False)
    except Exception: return pd.DataFrame()

# -----------------------------------------------------------
# Main Logic
# -----------------------------------------------------------
try:
    # 0. 매크로 데이터 선행 계산
    if not raw.get('fed', pd.Series()).empty:
        base_idx = raw['fed'].resample('W-WED').last().index
        df_m = pd.DataFrame(index=base_idx)
        for k in raw:
            if k not in [a['id'] for a in ASSETS_CONFIG] and k != 'diff':
                try: df_m[k] = raw[k].reindex(df_m.index, method='ffill')
                except: continue
        df_m = df_m.fillna(method='ffill')

        # [Level Data 사용] YoY가 아닌 Raw Trillion Data를 준비
        s_m2_us, s_m3_eu, s_m3_jp = df_m.get('m2_us'), df_m.get('m3_eu'), df_m.get('m3_jp')
        if s_m2_us is not None and s_m3_eu is not None and s_m3_jp is not None:
            global_m2_sum = (s_m2_us/1000) + ((s_m3_eu * df_m.get('eur_usd', 1))/1e12) + ((s_m3_jp / df_m.get('usd_jpy', 1))/1e12)
            df_m['Global_M2_Tril'] = global_m2_sum.interpolate(limit_direction='both')
            # YoY는 차트용으로 남겨둠
            df_m['Global_M2_YoY'] = df_m['Global_M2_Tril'].pct_change(52) * 100
        else: 
            df_m['Global_M2_Tril'] = pd.Series(dtype=float)
            df_m['Global_M2_YoY'] = pd.Series(dtype=float)

        s_fed, s_ecb, s_boj = df_m.get('fed'), df_m.get('ecb'), df_m.get('boj')
        if s_fed is not None and s_ecb is not None and s_boj is not None:
            g3_sum = (s_fed/1e6) + ((s_ecb * df_m.get('eur_usd', 1))/1e6) + ((s_boj * 0.0001) / df_m.get('usd_jpy', 1))
            df_m['G3_Asset_Tril'] = g3_sum.replace(0, np.nan).interpolate()
            df_m['G3_Asset_YoY'] = df_m['G3_Asset_Tril'].pct_change(52) * 100
        else: 
            df_m['G3_Asset_Tril'] = pd.Series(dtype=float)
            df_m['G3_Asset_YoY'] = pd.Series(dtype=float)

        df_m['Fed_Net_Tril'] = (df_m.get('fed',0)/1000 - df_m.get('tga',0)/1000 - df_m.get('rrp',0)/1000000)
        df_m['Fed_Net_YoY'] = df_m['Fed_Net_Tril'].pct_change(52) * 100

    # 1. 상단: Integrated Risk Radar
    st.markdown("### ⚡ Integrated Risk Radar")
    r_cols = st.columns(2)

    # [Radar 1] HY Spread
    if 'hy_spread' in raw and not raw['hy_spread'].empty:
        risk_res = check_risk_radar(raw['hy_spread'])
        if risk_res:
            with r_cols[0]:
                st.markdown("#### 🛡️ HY Spread Monitor")
                c1, c2 = st.columns([1.5, 2])
                with c1: st.metric("Level", f"{risk_res['val']:.2f}%", f"{risk_res['daily_chg_bps']:+.0f} bps", delta_color="inverse")
                with c2: 
                    if risk_res['status'] == "Normal": st.success(f"{risk_res['msg']}")
                    elif risk_res['status'] == "Warning": st.warning(f"{risk_res['msg']}")
                    else: st.error(f"{risk_res['msg']}")

    # [Radar 2] M2 Divergence (Level Based)
    if 'btc' in raw and not raw['btc'].empty and not df_m['Global_M2_Tril'].empty:
        # [수정] Log-Level 엔진에 Raw Price와 Raw M2 전달
        m2_res = run_quant_analysis_pure(df_m['Global_M2_Tril'], raw['btc'], shift_days)
        if m2_res:
            with r_cols[1]:
                st.markdown("#### 🌊 Liquidity Z-Gap (Log-Level)")
                c1, c2 = st.columns([1.5, 2])
                with c1:
                    gap_state = "High" if m2_res['gap_z'] > 1.0 else ("Low" if m2_res['gap_z'] < -1.0 else "Fair")
                    st.metric("Z-Gap", f"{m2_res['gap_z']:+.2f} σ", gap_state, delta_color="inverse")
                with c2:
                    regime = m2_res['regime']
                    lag_msg = f"Lag: {m2_res['calc_lag']}d ({m2_res['mode']})"
                    
                    if "Overheat" in regime: st.error(f"🔴 과열 (Overheat)\n({lag_msg})")
                    elif "Undervalued" in regime: st.success(f"🟢 저평가 (Value)\n({lag_msg})")
                    elif "Fair" in regime: st.info(f"⚪ 적정 (Fair)\n({lag_msg})")
                    else: st.warning(f"⚠️ {regime}\n({lag_msg})")
    
    # [NEW] Z-Gap Trend Chart (Level Based)
    st.markdown("#### 🌊 Z-Gap Trend Monitor (All Selected Assets)")
    
    target_z_assets = [a['id'] for a in ASSETS_CONFIG if selected_assets[a['id']] and a['id'] != 'hy_spread']
    z_chart_data = {}
    
    for t_asset in target_z_assets:
        if t_asset in raw and not raw[t_asset].empty and not df_m['Global_M2_Tril'].empty:
            # [수정] Log Level 로직 직접 구현 (차트용)
            asset_series_daily = raw[t_asset]
            asset_weekly = asset_series_daily.resample('W-WED').last()
            
            # Log 변환
            s_liq = np.log(df_m['Global_M2_Tril'].replace(0, np.nan).dropna())
            s_price = np.log(asset_weekly.replace(0, np.nan).dropna())
            
            df_z = pd.concat([s_liq, s_price], axis=1).dropna()
            df_z.columns = ['L_Log', 'P_Log']
            
            if len(df_z) > 10:
                df_z['L_Smooth'] = df_z['L_Log'].rolling(4).mean()
                df_z['P_Smooth'] = df_z['P_Log'].rolling(4).mean()
                df_z = df_z.dropna()
                
                df_z['L_Z'] = (df_z['L_Smooth'] - df_z['L_Smooth'].mean()) / (df_z['L_Smooth'].std() + 1e-9)
                df_z['P_Z'] = (df_z['P_Smooth'] - df_z['P_Smooth'].mean()) / (df_z['P_Smooth'].std() + 1e-9)
                
                # Hybrid Shift Logic (Chart)
                if shift_days != 0:
                    final_lag_weeks = int(shift_days / 7)
                else:
                    # Auto Lag (Diff Correlation)
                    df_diff = df_z[['L_Z', 'P_Z']].diff().dropna()
                    best_lag_w, best_r = 0, -1.0
                    for lg in range(0, 53):
                        r = df_diff['P_Z'].corr(df_diff['L_Z'].shift(lg))
                        if r > best_r: best_r, best_lag_w = r, lg
                    final_lag_weeks = best_lag_w
                
                df_z['L_Z_Shifted'] = df_z['L_Z'].shift(final_lag_weeks)
                df_z['Gap_Z'] = df_z['P_Z'] - df_z['L_Z_Shifted']
                z_chart_data[t_asset] = df_z['Gap_Z'].dropna()

    if z_chart_data:
        fig_z = go.Figure()
        x_min = min([s.index.min() for s in z_chart_data.values()])
        x_max = max([s.index.max() for s in z_chart_data.values()])
        
        fig_z.add_shape(type="rect", x0=x_min, x1=x_max, y0=-2.0, y1=-6.0, fillcolor="rgba(0, 255, 127, 0.1)", line=dict(width=0), layer="below")
        fig_z.add_shape(type="rect", x0=x_min, x1=x_max, y0=1.5, y1=6.0, fillcolor="rgba(255, 69, 0, 0.1)", line=dict(width=0), layer="below")
        
        fig_z.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
        fig_z.add_hline(y=1.5, line_dash="dash", line_color="red", annotation_text="Overheat (+1.5σ)", annotation_position="top left")
        fig_z.add_hline(y=-2.0, line_dash="dash", line_color="#00FF7F", annotation_text="Deep Value (-2.0σ)", annotation_position="bottom left")

        colors = {
            'btc': '#00FFEE', 'eth': '#627EEA', 'doge': '#FFA500', 'link': '#2A5ADA',
            'nasdaq': '#D62780', 'gold': '#FFD700', 'silver': '#C0C0C0',
            'ada': '#0033AD', 'xrp': '#00AAE4'
        }
        
        for t_asset, series in z_chart_data.items():
            c = colors.get(t_asset, '#FFFFFF')
            name = next((a['name'] for a in ASSETS_CONFIG if a['id'] == t_asset), t_asset.upper())
            fig_z.add_trace(go.Scatter(x=series.index, y=series, name=name, line=dict(color=c, width=2)))

        fig_z.update_layout(
            template="plotly_dark", height=350, margin=dict(l=20, r=20, t=30, b=20),
            yaxis=dict(title="Z-Gap (Sigma)", range=[-4.0, 4.0]),
            xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
            legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center")
        )
        st.plotly_chart(fig_z, use_container_width=True)

    # Z-Gap Guide
    with st.expander("ℹ️ Z-Gap 해석 가이드 (Signal Traffic Light)"):
        st.markdown("""
        | 구간 (Sigma) | 상태 | 의미 (Meaning) | 행동 요령 (Action) |
        | :--- | :--- | :--- | :--- |
        | **+1.5 이상** | 🔴 **High (과열)** | 유동성 대비 가격이 너무 높음. | **매도/관망** |
        | **+1.0 ~ +1.5** | 🟠 **Warn (주의)** | 가격이 유동성을 앞서가기 시작함. | 추격 매수 자제 |
        | **-1.0 ~ +1.0** | ⚪ **Fair (적정)** | 가격과 유동성이 **비슷한 속도**로 동행 중. | **추세 추종 (Hold)** |
        | **-1.5 ~ -1.0** | 🔵 **Low (기회)** | 돈은 풀렸는데 가격이 아직 덜 오름. | **분할 매수 (Buy)** |
        | **-2.0 이하** | 🟢 **Deep Value** | 극심한 공포/투매 구간. 절호의 기회. | **강력 매수 (Strong Buy)** |
        """)

    st.divider()

    # 2. Shift Logic (Main Chart - Still uses YoY for Momentum View)
    if not raw.get('fed', pd.Series()).empty:
        def apply_shift(s, days):
            if s.empty: return pd.Series(dtype=float)
            new_s = s.copy()
            new_s.index = new_s.index - pd.Timedelta(days=days)
            return new_s

        processed = {}
        for asset in ASSETS_CONFIG:
            s = raw.get(asset['id'], pd.Series(dtype=float))
            if isinstance(s.index, pd.DatetimeIndex):
                processed[asset['id']] = apply_shift(s, shift_days)
            else: processed[asset['id']] = pd.Series(dtype=float)

        # 3. Chart
        st.subheader(f"📊 Integrated Strategy Chart (Momentum View)")
        start_viz = pd.to_datetime('2021-06-01') 
        def flt(s): return s[s.index >= start_viz] if not s.empty else s

        if "Global M2" in liq_option:
            liq_v, liq_name, liq_color = flt(df_m['Global_M2_YoY']), "Global M2", "#FF4500"
        elif "G3" in liq_option:
            liq_v, liq_name, liq_color = flt(df_m['G3_Asset_YoY']), "G3 Assets", "#FFD700"
        else:
            liq_v, liq_name, liq_color = flt(df_m['Fed_Net_YoY']), "Fed Net", "#00FF7F"

        liq_v = liq_v.replace([np.inf, -np.inf], np.nan).dropna()
        if not liq_v.empty:
            l_min, l_max = liq_v.min(), liq_v.max()
            if pd.isna(l_min) or pd.isna(l_max): l_rng = [-20, 20]
            else: l_rng = [l_min - (l_max-l_min)*0.1, l_max + (l_max-l_min)*0.1]
        else: l_rng = [-20, 20]

        active_assets = [a for a in ASSETS_CONFIG if selected_assets[a['id']]]
        num_active = len(active_assets)
        if is_mobile: tick_fmt, margin, font_size = "s", 0.03, 10
        else: tick_fmt, margin, font_size = ",", 0.05 if num_active > 5 else 0.08, 12
        if num_active == 0: domain_end = 0.95
        else: domain_end = max(0.5, 1.0 - (num_active * margin))

        layout = go.Layout(
            template="plotly_dark", height=600,
            xaxis=dict(domain=[0.0, domain_end], showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
            yaxis=dict(title=dict(text=liq_name, font=dict(color=liq_color, size=font_size)), tickfont=dict(color=liq_color, size=font_size), range=l_rng, showgrid=False),
            legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"),
            hovermode="x", margin=dict(l=30, r=10, t=80, b=50)
        )

        fig = go.Figure(layout=layout)
        if not liq_v.empty:
            h = liq_color.lstrip('#')
            rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
            if shift_days != 0:
                last_date = liq_v.index.max()
                start_date = last_date - pd.Timedelta(days=abs(shift_days))
                fig.add_shape(type="rect", x0=start_date, x1=last_date, y0=l_rng[0], y1=l_rng[1], fillcolor="rgba(200, 200, 200, 0.15)", line=dict(width=0), layer="below")
                fig.add_annotation(x=last_date, y=l_rng[1], text=f"Lag:{abs(shift_days)}d", showarrow=False, yshift=10, xshift=-40, font=dict(color="rgba(255,255,255,0.7)", size=10))
            fig.add_trace(go.Scatter(x=liq_v.index, y=liq_v, name=liq_name, line=dict(color=liq_color, width=3), fill='tozeroy', fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.15)", yaxis='y', hoverinfo='none'))

        current_pos = domain_end
        for i, asset in enumerate(active_assets):
            data = flt(processed[asset['id']])
            data = data.replace([np.inf, -np.inf], np.nan).dropna()
            if data.empty: continue
            axis_name = f'yaxis{i+2}'
            axis_key = f'y{i+2}'
            d_min, d_max = data.min(), data.max()
            if pd.isna(d_min) or pd.isna(d_max) or d_min <= 0: d_min = 0.0001
            t_type = "linear"
            if asset['id'] == 'hy_spread': rng = [d_min - 0.5, d_max + 0.5]
            elif asset['id'] == 'doge': 
                t_type = "log"
                log_min, log_max = np.log10(d_min), np.log10(d_max)
                span = log_max - log_min
                rng = [log_min - span*0.1, log_max + span*0.1]
            else:
                span = d_max - d_min
                rng = [d_min - span*0.2, d_max + span*0.1]
            fig.update_layout({
                axis_name: dict(
                    title=dict(text=asset['name'], font=dict(color=asset['color'], size=font_size)),
                    tickfont=dict(color=asset['color'], size=font_size),
                    overlaying="y", side="right", anchor="free", position=current_pos,
                    range=rng, type=t_type, showgrid=False, tickformat=tick_fmt
                )
            })
            fig.add_trace(go.Scatter(x=data.index, y=data, name=asset['name'], line=dict(color=asset['color'], width=2), yaxis=axis_key, hoverinfo='none'))
            current_pos += margin

        st.plotly_chart(fig, use_container_width=True, key="main_chart")

        # 4. Stress Test
        st.markdown("---")
        st.subheader("📉 Crash Simulation (Stress Test)")
        st.caption(f"기간: **{sim_start_date} ~ {sim_end_date}** | 감지 조건: Spread Spike **≥ {spike_threshold} bps**")

        if 'hy_spread' in raw and 'btc' in raw:
            res_df = run_stress_test(raw['hy_spread'], raw['btc'], spike_threshold, look_forward_days, sim_start_date, sim_end_date)
            if not res_df.empty:
                success_cases = res_df[res_df['Raw_Return'] < 0]
                fail_cases = res_df[res_df['Raw_Return'] >= 0]
                total_sigs = len(res_df)
                success_rate = (len(success_cases) / total_sigs) * 100
                avg_saved = success_cases['Raw_Return'].mean() if not success_cases.empty else 0
                avg_missed = fail_cases['Raw_Return'].mean() if not fail_cases.empty else 0
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("위험 감지 횟수", f"{total_sigs} 회")
                c2.metric("방어 확률 (Win Rate)", f"{success_rate:.1f}%")
                c3.metric("평균 방어 수익률", f"{avg_saved:.2f}%")
                c4.metric("평균 기회비용", f"{avg_missed:.2f}%")
                st.dataframe(res_df[['Date', 'Spike', 'BTC Return', 'Outcome']].style.map(lambda x: 'color: #00FF7F' if '성공' in str(x) else ('color: #FF4500' if '휩쏘' in str(x) else ''), subset=['Outcome']), use_container_width=True)
            else:
                st.info(f"선택하신 기간 동안 감지된 위험 신호가 없습니다.")

        # 5. Quant Analytics (Table - also Updated)
        st.markdown("---")
        st.subheader("🛰️ Matrix Quant Analytics")
        st.caption("비교 기준: Log Levels (Abs Value) ↔ Recent (Last 30d)")
        
        # [수정] 표도 Log-Level 로직 사용
        if active_assets:
            asset_tabs = st.tabs([f"{a['name']}" for a in active_assets])
            for tab, asset in zip(asset_tabs, active_assets):
                with tab:
                    raw_asset_series = raw.get(asset['id'], pd.Series(dtype=float))
                    if raw_asset_series.empty: continue
                    
                    # 사용할 유동성 소스들 (Raw Level 전달)
                    sources = []
                    if not df_m['Fed_Net_Tril'].empty: sources.append(("🇺🇸 Fed Net Liq", df_m['Fed_Net_Tril']))
                    if not df_m['G3_Asset_Tril'].empty: sources.append(("🏛️ G3 Assets", df_m['G3_Asset_Tril']))
                    if not df_m['Global_M2_Tril'].empty: sources.append(("🌍 Global M2", df_m['Global_M2_Tril']))
                    
                    results = []
                    for liq_label, liq_data in sources:
                        res = run_quant_analysis_pure(liq_data, raw_asset_series, shift_days)
                        if res:
                            res['label'] = liq_label
                            results.append(res)
                    
                    if not results: continue
                    cols = st.columns(len(results))
                    best_res = max(results, key=lambda x: x['global_corr']) # 상관계수는 여전히 Diff 기준
                    
                    for i, res in enumerate(results):
                        with cols[i]:
                            if res == best_res: st.markdown(f"#### ⭐ {res['label']}")
                            else: st.markdown(f"#### {res['label']}")
                            st.metric("Optimal Lag", f"{res['optimal_lag']} days")
                            st.metric("Regime", f"{res['regime']}")
                            gap_state = "High" if res['gap_z'] > 1.0 else ("Low" if res['gap_z'] < -1.0 else "Fair")
                            st.metric("Z-Gap", f"{res['gap_z']:+.2f} σ", gap_state, delta_color="inverse")
                            st.caption(f"Lag Used: {res['calc_lag']}d ({res['mode']})")

        with st.expander("🔍 데이터 연결 리포트"):
            active_ids_report = [a['id'] for a in ASSETS_CONFIG if selected_assets[a['id']]]
            for asset in ASSETS_CONFIG:
                if asset['id'] in active_ids_report:
                    s = processed[asset['id']]
                    if s.empty: st.error(f"❌ {asset['name']}: 로드 실패")
                    else:
                        extra = f" ({meta.get(asset['id'], 'OK')})" if asset['id'] in meta else ""
                        st.success(f"✅ {asset['name']}: 로드 성공{extra}")
    else:
        st.error("❌ 데이터 로드 실패")

except Exception as e:
    st.error(f"⚠️ 시스템 오류: {str(e)}")
