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
from datetime import date, timedelta, datetime

# 1. 환경 설정
warnings.filterwarnings("ignore")
st.set_page_config(page_title="GM Time Machine", layout="wide")

st.title("🏛️ Grand Master: Analytics Engine")
st.caption("Ver 19.3 | 시뮬레이션 기간 설정(Time Machine) 추가 | 구간별 정밀 검증 가능")

# -----------------------------------------------------------
# [사이드바 설정]
# -----------------------------------------------------------
st.sidebar.header("⚙️ Control Panel")
is_mobile = st.sidebar.checkbox("📱 모바일 모드 (축 공간 최소화)", value=True)

# [Stress Test 옵션 - 날짜 선택 추가]
st.sidebar.markdown("---")
st.sidebar.subheader("📉 Crash Simulation")

# 1. 민감도 설정
spike_threshold = st.sidebar.slider(
    "위험 감지 민감도 (Daily Delta bps)", 
    min_value=5, max_value=50, value=15, step=1,
    help="하루에 스프레드가 이 값(bps) 이상 튀어 오르면 '위기'로 간주합니다."
)
look_forward_days = st.sidebar.slider(
    "반응 관찰 기간 (Days)",
    min_value=1, max_value=30, value=7,
    help="신호 발생 후 며칠 뒤의 가격 등락을 확인할까요?"
)

# 2. [NEW] 날짜 구간 설정
st.sidebar.markdown("**검증 기간 설정 (Date Range)**")
col_d1, col_d2 = st.sidebar.columns(2)
with col_d1:
    sim_start_date = st.date_input("시작일", value=date(2019, 1, 1))
with col_d2:
    sim_end_date = st.date_input("종료일", value="today")

st.sidebar.markdown("---")
liq_option = st.sidebar.radio(
    "1. 유동성 지표 (Left Axis)",
    (
        "🇺🇸 Fed Net Liquidity (미국 실질 유동성)",
        "🏛️ G3 Central Bank Assets (본원통화 총량)",
        "🌍 Global M2 (실물 통화량: US+EU+JP)"
    ),
    index=2
)

st.sidebar.markdown("---")
st.sidebar.write("2. Time Shift (Days)")
shift_days = st.sidebar.number_input(
    "자산/지표 이동 (일)", min_value=-365, max_value=365, value=90, step=7
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
# 데이터 수집 (Logic)
# -----------------------------------------------------------
def fetch_master_data_logic():
    d = {}
    meta_info = {}
    GLOBAL_START = time.time()
    MAX_EXECUTION_TIME = 30 
    
    # [설정] 2018년부터 데이터를 가져오되, 실제 분석은 사용자가 지정한 날짜로 자름
    START_YEAR = 2018 
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
            for _ in range(12): 
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
    status_text.text("📡 Initializing Data...")

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
# [FUNC 1] Risk Radar (Real-Time BPS)
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
        
    return {
        "val": last_val, 
        "daily_chg_bps": daily_chg_bps, 
        "status": status, 
        "color": color, 
        "msg": msg
    }

# -----------------------------------------------------------
# [FUNC 2] Stress Test (Date Range Added)
# -----------------------------------------------------------
def run_stress_test(hy_series, btc_series, threshold_bps, look_forward, start_d, end_d):
    try:
        hy = hy_series.copy()
        btc = btc_series.copy()
        hy.index = hy.index.normalize()
        btc.index = btc.index.normalize()
        
        # [NEW] 사용자 지정 날짜 구간으로 필터링
        s_date = pd.to_datetime(start_d).normalize()
        e_date = pd.to_datetime(end_d).normalize()
        
        # 데이터를 병합하기 전에 먼저 자르는 게 아니라, 병합 후 자르는 게 안전
        df = pd.concat([hy, btc], axis=1).dropna()
        df.columns = ['Spread', 'Price']
        
        # 날짜 필터링 적용
        df = df[(df.index >= s_date) & (df.index <= e_date)]
        
        if df.empty: return pd.DataFrame()
        
        df['Spread_Chg_Bps'] = df['Spread'].diff() * 100
        events = df[df['Spread_Chg_Bps'] >= threshold_bps].index
        
        results = []
        for date in events:
            target_date = date + timedelta(days=look_forward)
            # 미래 날짜가 현재 데이터 범위(필터링 된 범위가 아님, 전체 범위) 내에 있어야 함
            # 따라서 미래 가격 조회용으로는 원본(필터 전) 데이터를 쓰는 게 좋을 수도 있으나,
            # 여기서는 편의상 잘린 데이터 내에서 확인하거나, 원본 데이터를 참조해야 함.
            # 정확도를 위해 원본 시리즈에서 미래 가격을 조회
            
            # 원본 시리즈에서 가격 조회
            if target_date <= btc.index[-1]:
                try:
                    price_at_signal = df.loc[date]['Price']
                    # 미래 가격은 btc 원본에서 조회 (필터링된 종료일 이후의 결과도 궁금할 수 있으므로)
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
    except Exception:
        return pd.DataFrame()

# -----------------------------------------------------------
# [FUNC 3] Quant Analytics
# -----------------------------------------------------------
def run_quant_analysis_pure(liq_series, asset_series_daily):
    try:
        asset_weekly = asset_series_daily.resample('W-WED').last()
        asset_yoy = asset_weekly.pct_change(52) * 100
        df = pd.concat([liq_series, asset_yoy], axis=1).dropna()
        df.columns = ['Liquidity_YoY', 'Price_YoY']
        
        if len(df) < 52: return None
        
        df['L_Smooth'] = df['Liquidity_YoY'].rolling(4).mean()
        df['P_Smooth'] = df['Price_YoY'].rolling(4).mean()
        df = df.dropna()
        if df.empty: return None
        
        df['L_Z'] = (df['L_Smooth'] - df['L_Smooth'].mean()) / (df['L_Smooth'].std() + 1e-9)
        df['P_Z'] = (df['P_Smooth'] - df['P_Smooth'].mean()) / (df['P_Smooth'].std() + 1e-9)

        best_lag_weeks, best_corr = 0, -1.0
        for lag in range(0, 53): 
            corr = df['P_Z'].corr(df['L_Z'].shift(lag))
            if corr > best_corr: best_corr, best_lag_weeks = corr, lag
        
        best_lag_days = best_lag_weeks * 7
        recent_window = 4 
        df['L_Z_Shifted'] = df['L_Z'].shift(best_lag_weeks)
        df_recent = df.iloc[-recent_window:]
        if len(df_recent) < recent_window: return None
        
        recent_corr = df_recent['P_Z'].corr(df_recent['L_Z_Shifted'])
        last_val = df.iloc[-1]
        gap_z = last_val['P_Z'] - last_val['L_Z_Shifted']
        
        if best_corr < 0: regime = "Inverse"
        elif recent_corr > 0.5: regime = "Sync"
        elif recent_corr < 0.0: regime = "Divergence" 
        else: regime = "Weak"

        return {
            "optimal_lag": best_lag_days, "global_corr": best_corr,
            "recent_corr": recent_corr, "gap_z": gap_z, "regime": regime
        }
    except Exception: return None

# -----------------------------------------------------------
# Main Logic
# -----------------------------------------------------------
try:
    if 'hy_spread' in raw and not raw['hy_spread'].empty:
        risk_res = check_risk_radar(raw['hy_spread'])
        if risk_res:
            st.markdown("### ⚡ Risk Radar (HY Spread)")
            r_col1, r_col2, r_col3 = st.columns([1, 1, 2])
            with r_col1: 
                st.metric("HY Spread", f"{risk_res['val']:.2f}%", f"{risk_res['daily_chg_bps']:+.0f} bps (Daily)", delta_color="inverse")
            with r_col2: st.metric("Signal", risk_res['msg'])
            with r_col3:
                if risk_res['status'] == "Normal": st.success("안정적 (Risk-On)")
                elif risk_res['status'] == "Warning": st.warning("주의 필요 (Spike)")
                else: st.error("위험 (Risk-Off)")
            st.divider()

    if not raw.get('fed', pd.Series()).empty:
        base_idx = raw['fed'].resample('W-WED').last().index
        df_m = pd.DataFrame(index=base_idx)
        
        for k in raw:
            if k not in [a['id'] for a in ASSETS_CONFIG] and k != 'diff':
                try: df_m[k] = raw[k].reindex(df_m.index, method='ffill')
                except: continue
        
        df_m = df_m.fillna(method='ffill')

        # G3 Calc
        s_fed, s_ecb, s_boj = df_m.get('fed'), df_m.get('ecb'), df_m.get('boj')
        if s_fed is not None and s_ecb is not None and s_boj is not None:
            fed_t = s_fed / 1000000
            ecb_t = (s_ecb * df_m.get('eur_usd', 1)) / 1000000
            boj_t = (s_boj * 0.0001) / df_m.get('usd_jpy', 1)
            g3_sum = fed_t.fillna(0) + ecb_t.fillna(0) + boj_t.fillna(0)
            df_m['G3_Asset_Tril'] = g3_sum.replace(0, np.nan).interpolate()
            df_m['G3_Asset_YoY'] = df_m['G3_Asset_Tril'].pct_change(52) * 100
        else: df_m['G3_Asset_YoY'] = pd.Series(dtype=float)

        # Global M2 Calc
        s_m2_us, s_m3_eu, s_m3_jp = df_m.get('m2_us'), df_m.get('m3_eu'), df_m.get('m3_jp')
        if s_m2_us is not None and s_m3_eu is not None and s_m3_jp is not None:
            m2_us = s_m2_us / 1000
            m3_eu = (s_m3_eu * df_m.get('eur_usd', 1)) / 1e12
            m3_jp = (s_m3_jp / df_m.get('usd_jpy', 1)) / 1e12
            global_m2_sum = m2_us + m3_eu + m3_jp
            df_m['Global_M2_Tril'] = global_m2_sum.interpolate(limit_direction='both')
            df_m['Global_M2_YoY'] = df_m['Global_M2_Tril'].pct_change(52) * 100
        else:
            df_m['Global_M2_YoY'] = pd.Series(dtype=float)

        df_m['Fed_Net_Tril'] = (df_m.get('fed',0)/1000 - df_m.get('tga',0)/1000 - df_m.get('rrp',0)/1000000)
        df_m['Fed_Net_YoY'] = df_m['Fed_Net_Tril'].pct_change(52) * 100

        # Shift
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

        # Chart
        st.subheader(f"📊 Integrated Strategy Chart (Shift: {shift_days}d)")
        
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
        # [NEW] 날짜 표시 추가
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
                c3.metric("평균 방어 수익률", f"{avg_saved:.2f}%", help="성공 시 회피한 하락폭 평균")
                c4.metric("평균 기회비용", f"{avg_missed:.2f}%", help="실패 시 놓친 상승폭 평균")
                
                st.dataframe(res_df[['Date', 'Spike', 'BTC Return', 'Outcome']].style.map(lambda x: 'color: #00FF7F' if '성공' in str(x) else ('color: #FF4500' if '휩쏘' in str(x) else ''), subset=['Outcome']), use_container_width=True)
            else:
                st.info(f"선택하신 기간({sim_start_date} ~ {sim_end_date}) 동안 설정된 민감도({spike_threshold} bps)로 감지된 위험 신호가 없습니다.")

        # 5. Quant Analytics
        st.markdown("---")
        st.subheader("🛰️ Matrix Quant Analytics")
        st.caption("비교 기준: Historical (2021~, 전체 역사) ↔ Recent (Last 30d, 최근 1달)")
        
        status_box = st.empty()
        status_box.info("🚀 Starting Quant Analysis...")
        
        liquidity_sources = [
            ("🇺🇸 Fed Net Liq", df_m['Fed_Net_YoY']),
            ("🏛️ G3 Assets",    df_m.get('G3_Asset_YoY', pd.Series(dtype=float))),
            ("🌍 Global M2",    df_m['Global_M2_YoY'])
        ]

        if active_assets:
            asset_tabs = st.tabs([f"{a['name']}" for a in active_assets])
            for tab, asset in zip(asset_tabs, active_assets):
                with tab:
                    status_box.caption(f"Analyzing {asset['name']}...")
                    raw_asset_series = raw.get(asset['id'], pd.Series(dtype=float))
                    
                    if raw_asset_series.empty:
                        st.warning("데이터 부족")
                        continue
                    
                    results = []
                    for liq_label, liq_data in liquidity_sources:
                        if liq_data.empty: continue
                        res = run_quant_analysis_pure(liq_data, raw_asset_series)
                        if res:
                            res['label'] = liq_label
                            results.append(res)
                    
                    if not results:
                        st.info("분석 데이터 부족")
                        continue

                    cols = st.columns(len(results))
                    best_res = max(results, key=lambda x: x['global_corr'])
                    
                    for i, res in enumerate(results):
                        with cols[i]:
                            if res == best_res: st.markdown(f"#### ⭐ {res['label']}")
                            else: st.markdown(f"#### {res['label']}")

                            st.metric("Optimal Lag", f"{res['optimal_lag']} days")
                            st.metric("Hist. Corr (4y)", f"{res['global_corr']:.2f}")
                            st.metric("Recent Corr (30d)", f"{res['recent_corr']:.2f}", delta=f"{res['recent_corr'] - res['global_corr']:.2f}")
                            
                            regime_icon = "🟢" if "Sync" in res['regime'] else ("⚠️" if "Divergence" in res['regime'] else ("📉" if "Inverse" in res['regime'] else "⚪"))
                            st.metric("Regime", f"{regime_icon} {res['regime']}")
                            
                            gap_state = "High" if res['gap_z'] > 1.0 else ("Low" if res['gap_z'] < -1.0 else "Fair")
                            st.metric("Z-Gap", f"{res['gap_z']:+.2f} σ", gap_state, delta_color="inverse")
                    
                    if best_res['global_corr'] < 0:
                        insight = f"**{asset['name']}**는 유동성과 **역상관(Inverse)** 관계입니다."
                    else:
                        insight = f"**{asset['name']}**는 **{best_res['label']}**와 밀접하며, 최근 **{best_res['regime']}** 상태입니다."
                    st.info(f"**Insight:** {insight}")
        
        status_box.empty()

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
