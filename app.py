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
st.set_page_config(page_title="GM True YoY", layout="wide")

st.title("🏛️ Grand Master: True YoY Analytics")
st.caption("Ver 17.1 | 자산 가격 YoY 변환 적용 | 유동성 vs 자산 등락률 동기화 분석")

# -----------------------------------------------------------
# [사이드바 설정]
# -----------------------------------------------------------
st.sidebar.header("⚙️ Control Panel")

is_mobile = st.sidebar.checkbox("📱 모바일 모드 (축 공간 최소화)", value=True)

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
    "자산 가격 이동 (일)", min_value=-365, max_value=365, value=90, step=7,
    help="차트 시각화용 수동 이동입니다. (하단 퀀트 분석은 자동으로 최적값을 찾습니다)"
)

st.sidebar.markdown("---")
st.sidebar.write("3. 표시할 자산 (Right Axes)")

ASSETS_CONFIG = [
    {'id': 'nasdaq', 'name': 'Nasdaq', 'symbol': 'IXIC', 'source': 'hybrid', 'color': '#D62780', 'type': 'index', 'default': False},
    {'id': 'gold',   'name': 'Gold',   'symbol': 'GC=F', 'source': 'hybrid_metal', 'color': '#FFD700', 'type': 'metal', 'default': False},
    {'id': 'silver', 'name': 'Silver', 'symbol': 'SI=F', 'source': 'hybrid_metal', 'color': '#C0C0C0', 'type': 'metal', 'default': False},
    {'id': 'btc',    'name': 'BTC',    'symbol': 'BTC/KRW', 'source': 'bithumb', 'color': '#00FFEE', 'type': 'crypto', 'default': True},
    {'id': 'doge',   'name': 'DOGE',   'symbol': 'DOGE/KRW', 'source': 'bithumb', 'color': '#FFA500', 'type': 'crypto', 'default': False},
    {'id': 'eth',    'name': 'ETH',    'symbol': 'ETH/KRW', 'source': 'bithumb', 'color': '#627EEA', 'type': 'crypto', 'default': False},
    {'id': 'link',   'name': 'LINK',   'symbol': 'LINK/KRW', 'source': 'bithumb', 'color': '#2A5ADA', 'type': 'crypto', 'default': False},
    {'id': 'ada',    'name': 'ADA',    'symbol': 'ADA/KRW', 'source': 'bithumb', 'color': '#0033AD', 'type': 'crypto', 'default': False},
    {'id': 'xrp',    'name': 'XRP',    'symbol': 'XRP/KRW', 'source': 'bithumb', 'color': '#00AAE4', 'type': 'crypto', 'default': False},
]

selected_assets = {}
for asset in ASSETS_CONFIG:
    selected_assets[asset['id']] = st.sidebar.checkbox(f"{asset['name']}", value=asset['default'])

# -----------------------------------------------------------
# 데이터 수집
# -----------------------------------------------------------
def fetch_master_data_logic():
    d = {}
    meta_info = {}
    
    GLOBAL_START = time.time()
    MAX_EXECUTION_TIME = 25 
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    START_YEAR = 2021
    headers = {'User-Agent': 'Mozilla/5.0'}

    def check_timeout(): return (time.time() - GLOBAL_START > MAX_EXECUTION_TIME)

    def get_fred(id):
        if check_timeout(): return pd.Series(dtype=float)
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"
            r = requests.get(url, headers=headers, timeout=3)
            df = pd.read_csv(io.StringIO(r.text), index_col=0, parse_dates=True)
            return df.squeeze().resample('D').interpolate(method='time').tz_localize(None)
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
            for _ in range(8): 
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

    status_text.text("📡 Initializing...")
    fred_ids = {
        'fed': 'WALCL', 'tga': 'WTREGEN', 'rrp': 'RRPONTSYD',
        'ecb': 'ECBASSETSW', 'boj': 'JPNASSETS',
        'm2_us': 'M2SL', 'm3_eu': 'MABMM301EZM189S', 'm3_jp': 'MABMM301JPM189S',
        'eur_usd': 'DEXUSEU', 'usd_jpy': 'DEXJPUS',
        'nasdaq_fred': 'NASDAQCOM'
    }
    
    total_steps = len(fred_ids) + len(ASSETS_CONFIG)
    current_step = 0
    
    for k, v in fred_ids.items():
        if check_timeout(): break
        d[k] = get_fred(v)
        current_step += 1
        progress_bar.progress(int((current_step / total_steps) * 90))

    if not d.get('nasdaq_fred', pd.Series()).empty: d['nasdaq'] = d['nasdaq_fred']
    else: d['nasdaq'] = get_yahoo("^IXIC")

    status_text.text("💰 Fetching Active Assets...")
    active_ids = [a['id'] for a in ASSETS_CONFIG if selected_assets[a['id']]]
    
    for asset in ASSETS_CONFIG:
        if asset['id'] not in active_ids:
            d[asset['id']] = pd.Series(dtype=float)
            continue
        if check_timeout(): 
            d[asset['id']] = pd.Series(dtype=float)
            continue
        if asset['id'] == 'nasdaq': continue
        
        if asset['source'] == 'hybrid_metal':
            data, src = get_metal_hybrid(asset['symbol'])
            d[asset['id']] = data
            meta_info[asset['id']] = src
        elif asset['source'] == 'yahoo': 
            d[asset['id']] = get_yahoo(asset['symbol'])
        elif asset['source'] == 'bithumb': 
            d[asset['id']] = fetch_bithumb(asset['symbol'])
        
        current_step += 1
        progress_bar.progress(min(int((current_step / total_steps) * 100), 100))

    status_text.text("✅ Data Ready")
    progress_bar.empty()
    status_text.empty()

    try:
        with open('difficulty (1).json', 'r') as f:
            js = json.load(f)['difficulty']
        df_js = pd.DataFrame(js)
        df_js['Date'] = pd.to_datetime(df_js['x'], unit='ms').dt.tz_localize(None)
        d['diff'] = df_js.set_index('Date').sort_index()['y']
    except: d['diff'] = pd.Series(dtype=float)

    return d, meta_info

raw, meta = fetch_master_data_logic()

# -----------------------------------------------------------
# [CORE] Quant Analytics Engine (True YoY)
# -----------------------------------------------------------
def run_quant_analysis(liq_series, asset_series_daily):
    """
    유동성(YoY)과 자산가격(YoY)을 동기화하여 분석
    """
    try:
        # 1. 데이터 동기화 (주간 단위 Resample)
        # 유동성 지표는 주간(Weekly) 데이터이므로, 자산 가격도 주간으로 맞춤
        # W-WED (수요일 기준)으로 맞추면 노이즈가 줄고 추세가 명확해짐
        asset_weekly = asset_series_daily.resample('W-WED').last()
        
        # 2. 자산 가격 YoY 변환 (핵심 수정)
        # 52주 전 가격 대비 변동률 = YoY
        asset_yoy = asset_weekly.pct_change(52) * 100
        
        # 데이터 병합 (교집합)
        df = pd.concat([liq_series, asset_yoy], axis=1).dropna()
        df.columns = ['Liquidity_YoY', 'Price_YoY']
        
        # 데이터가 너무 짧으면 분석 불가 (최소 1년치 데이터 필요)
        if len(df) < 52: return None
        
        # 3. 스무딩 & 정규화 (Z-Score)
        # YoY 데이터는 이미 추세성이 강하므로 가볍게 스무딩 (4주/1달)
        df['L_Smooth'] = df['Liquidity_YoY'].rolling(4).mean()
        df['P_Smooth'] = df['Price_YoY'].rolling(4).mean()
        df = df.dropna()
        
        df['L_Z'] = (df['L_Smooth'] - df['L_Smooth'].mean()) / df['L_Smooth'].std()
        df['P_Z'] = (df['P_Smooth'] - df['P_Smooth'].mean()) / df['P_Smooth'].std()

        # 4. 최적 시차 탐색 (Optimal Lag)
        # 주간 데이터이므로 Lag 1 = 1주일(7일)
        # 0주 ~ 26주(약 6개월) 후행 테스트
        best_lag_weeks = 0
        best_corr = -1.0
        
        for lag in range(0, 27): # 0~26 weeks
            shifted_L = df['L_Z'].shift(lag)
            corr = df['P_Z'].corr(shifted_L)
            if corr > best_corr:
                best_corr = corr
                best_lag_weeks = lag
        
        best_lag_days = best_lag_weeks * 7 # 일수로 변환
        
        # 5. 국면 감지 (Regime)
        # 최근 12주(약 3달) 상관계수 확인
        recent_window = 12
        df['L_Z_Shifted'] = df['L_Z'].shift(best_lag_weeks)
        df_recent = df.iloc[-recent_window:]
        recent_corr = df_recent['P_Z'].corr(df_recent['L_Z_Shifted'])
        
        # 6. 괴리율 (Gap)
        last_val = df.iloc[-1]
        gap_z = last_val['P_Z'] - last_val['L_Z_Shifted']
        
        return {
            "optimal_lag": best_lag_days,
            "max_corr": best_corr,
            "recent_corr": recent_corr,
            "gap_z": gap_z,
            "regime": "Sync" if recent_corr > 0.5 else ("Divergence" if recent_corr < 0.2 else "Weak")
        }

    except Exception:
        return None

# -----------------------------------------------------------
# Main Logic & Rendering
# -----------------------------------------------------------
try:
    if not raw.get('btc', pd.Series()).empty:

        base_idx = raw['fed'].resample('W-WED').last().index
        df_m = pd.DataFrame(index=base_idx)
        
        for k in raw:
            if k not in [a['id'] for a in ASSETS_CONFIG] and k != 'diff':
                series = raw[k]
                if not series.empty and isinstance(series.index, pd.DatetimeIndex):
                    try: df_m[k] = series.reindex(df_m.index, method='ffill')
                    except: continue
        
        df_m = df_m.fillna(method='ffill')

        df_m['Fed_Net_Tril'] = (df_m.get('fed',0)/1000 - df_m.get('tga',0)/1000 - df_m.get('rrp',0)/1000000)
        df_m['Fed_Net_YoY'] = df_m['Fed_Net_Tril'].pct_change(52) * 100

        fed_t = df_m.get('fed',0)/1000
        ecb_t = (df_m.get('ecb',0) * df_m.get('eur_usd',1)) / 1000000
        boj_t = (df_m.get('boj',0) * 0.0001) / df_m.get('usd_jpy',1)
        df_m['G3_Asset_YoY'] = (fed_t + ecb_t + boj_t).pct_change(52) * 100

        m2_us = df_m.get('m2_us',0) / 1000
        m3_eu = (df_m.get('m3_eu',0) * df_m.get('eur_usd',1)) / 1e12
        m3_jp = (df_m.get('m3_jp',0) / df_m.get('usd_jpy',1)) / 1e12
        df_m['Global_M2_YoY'] = (m2_us + m3_eu + m3_jp).pct_change(52) * 100

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
            else:
                processed[asset['id']] = pd.Series(dtype=float)

        st.subheader(f"📊 Integrated Strategy Chart (Shift: {shift_days}d)")
        
        start_viz = pd.to_datetime('2021-06-01') 
        def flt(s): return s[s.index >= start_viz] if not s.empty else s

        if "Global M2" in liq_option:
            liq_v = flt(df_m['Global_M2_YoY'])
            liq_name, liq_color = "Global M2", "#FF4500"
            liq_full_series = df_m['Global_M2_YoY']
        elif "G3" in liq_option:
            liq_v = flt(df_m['G3_Asset_YoY'])
            liq_name, liq_color = "G3 Assets", "#FFD700"
            liq_full_series = df_m['G3_Asset_YoY']
        else:
            liq_v = flt(df_m['Fed_Net_YoY'])
            liq_name, liq_color = "Fed Net", "#00FF7F"
            liq_full_series = df_m['Fed_Net_YoY']

        liq_v = liq_v.replace([np.inf, -np.inf], np.nan).dropna()

        if not liq_v.empty:
            l_min, l_max = liq_v.min(), liq_v.max()
            if pd.isna(l_min) or pd.isna(l_max): l_rng = [-20, 20]
            else:
                l_span = max(l_max - l_min, 10)
                l_rng = [l_min - l_span*0.1, l_max + l_span*0.1]
        else: l_rng = [-20, 20]

        active_assets = [a for a in ASSETS_CONFIG if selected_assets[a['id']]]
        num_active = len(active_assets)
        
        if is_mobile:
            tick_fmt = "s" 
            margin = 0.03  
            font_size = 10
        else:
            tick_fmt = "," 
            margin = 0.05 if num_active > 5 else 0.08
            font_size = 12

        if num_active == 0: domain_end = 0.95
        else:
            domain_end = max(0.5, 1.0 - (num_active * margin))

        spike_settings = dict(showspikes=True, spikemode='across', spikesnap='cursor', spikethickness=1, spikecolor='red', spikedash='dash')

        layout = go.Layout(
            template="plotly_dark", height=800,
            xaxis=dict(domain=[0.0, domain_end], showgrid=True, gridcolor='rgba(128,128,128,0.2)', **spike_settings),
            yaxis=dict(title=dict(text=liq_name, font=dict(color=liq_color, size=font_size)), tickfont=dict(color=liq_color, size=font_size), range=l_rng, showgrid=False, **spike_settings),
            legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"),
            hovermode="x",
            margin=dict(l=30, r=10, t=80, b=50)
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
            
            is_log = (asset['id'] == 'doge')
            if is_log:
                log_min, log_max = np.log10(d_min), np.log10(d_max)
                span = log_max - log_min
                rng = [log_min - span*0.1, log_max + span*0.1]
                t_type = "log"
            else:
                span = d_max - d_min
                rng = [d_min - span*0.2, d_max + span*0.1]
                t_type = "linear"

            fig.update_layout({
                axis_name: dict(
                    title=dict(text=asset['name'], font=dict(color=asset['color'], size=font_size)),
                    tickfont=dict(color=asset['color'], size=font_size),
                    overlaying="y", side="right", anchor="free", position=current_pos,
                    range=rng, type=t_type, showgrid=False, tickformat=tick_fmt, **spike_settings
                )
            })

            fig.add_trace(go.Scatter(x=data.index, y=data, name=asset['name'], line=dict(color=asset['color'], width=2), yaxis=axis_key, hoverinfo='none'))
            current_pos += margin

        st.plotly_chart(fig, use_container_width=True, key="main_chart")

        # ------------------------------------------------------------------
        # [NEW] Quant Analysis (Using Correct YoY Logic)
        # ------------------------------------------------------------------
        st.markdown("---")
        st.subheader("🛰️ Market Quant Analytics (YoY vs YoY)")
        st.caption(f"분석 기준: {liq_name} (YoY) ↔ 자산 가격 등락률 (YoY)")

        target_asset_id = 'btc' if 'btc' in [a['id'] for a in active_assets] else (active_assets[0]['id'] if active_assets else None)

        if target_asset_id:
            raw_asset_series = raw.get(target_asset_id, pd.Series(dtype=float))
            
            if not raw_asset_series.empty and not liq_full_series.empty:
                res = run_quant_analysis(liq_full_series, raw_asset_series)
                
                if res:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("최적 후행 시차 (Optimal Lag)", f"{res['optimal_lag']} days", help="자산의 등락률이 유동성 등락률을 따라가는 시간")
                    with col2:
                        st.metric("최대 상관계수 (Max Corr)", f"{res['max_corr']:.2f}", help="YoY 등락률 간의 상관관계 (1.0 = 완벽 동행)")
                    with col3:
                        if res['regime'] == "Sync": regime_icon = "🟢 동행 (Sync)"
                        elif res['regime'] == "Divergence": regime_icon = "⚠️ 이탈 (Div)"
                        else: regime_icon = "⚪ 약세 (Weak)"
                        st.metric("현재 국면 (Regime)", regime_icon, f"Recent Corr: {res['recent_corr']:.2f}")
                    with col4:
                        gap_val = res['gap_z']
                        gap_state = "High" if gap_val > 1.0 else ("Low" if gap_val < -1.0 else "Fair")
                        st.metric("괴리율 (Z-Gap)", f"{gap_val:+.2f} σ", f"{gap_state}", delta_color="inverse")

                    st.info(f"""
                    **💡 Insight:** **{target_asset_id.upper()}**의 가격 등락률(YoY)은 **{liq_name}**의 변화를 약 **{res['optimal_lag']}일** 후에 따라가는 경향이 있습니다.
                    현재 두 지표의 추세는 **{res['regime']}** 상태이며, 유동성 흐름 대비 가격 모멘텀은 **{gap_val:.2f} Sigma** 수준으로 **{gap_state}** 상태입니다.
                    """)
                else:
                    st.warning("데이터가 부족하여 퀀트 분석을 수행할 수 없습니다.")

        with st.expander("🔍 데이터 연결 리포트"):
            active_ids_report = [a['id'] for a in ASSETS_CONFIG if selected_assets[a['id']]]
            for asset in ASSETS_CONFIG:
                if asset['id'] in active_ids_report:
                    s = processed[asset['id']]
                    if s.empty:
                        st.error(f"❌ {asset['name']}: 로드 실패")
                    else:
                        extra = f" ({meta.get(asset['id'], 'OK')})" if asset['id'] in meta else ""
                        st.success(f"✅ {asset['name']}: 로드 성공{extra}")

    else:
        st.error("❌ 비트코인 로드 실패")

except Exception as e:
    st.error(f"⚠️ 차트 렌더링 중 오류 발생: {str(e)}")
