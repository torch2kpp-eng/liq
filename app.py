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
st.set_page_config(page_title="GM Deep Dive", layout="wide")

st.title("🏛️ Grand Master: Quant Deep Dive")
st.caption("Ver 17.5 | Global(전체 역사) vs Local(최근 90일) 상관관계 비교 | 구조적 이탈 감지")

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
    help="차트 시각화용 수동 이동입니다."
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
# 데이터 수집 (Logic)
# -----------------------------------------------------------
def fetch_master_data_logic():
    d = {}
    meta_info = {}
    
    GLOBAL_START = time.time()
    MAX_EXECUTION_TIME = 60 
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    START_YEAR = 2021
    headers = {'User-Agent': 'Mozilla/5.0'}

    def check_timeout(): return (time.time() - GLOBAL_START > MAX_EXECUTION_TIME)

    def get_fred(id):
        if check_timeout(): return pd.Series(dtype=float)
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"
            r = requests.get(url, headers=headers, timeout=5)
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

    status_text.text("💰 Fetching Assets...")
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
# [CORE] Quant Analytics (Comparison Engine)
# -----------------------------------------------------------
@st.cache_data(show_spinner=False) 
def run_quant_analysis_cached(liq_series, asset_series_daily):
    """
    Global(전체) vs Local(최근) 상관관계 비교 분석
    """
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

        # 1. Global Optimization (전체 기간 최적 시차 탐색)
        best_lag_weeks = 0
        best_corr = -1.0 # Global Max Corr
        
        # 1년(52주) 시차 탐색
        for lag in range(0, 53): 
            shifted_L = df['L_Z'].shift(lag)
            corr = df['P_Z'].corr(shifted_L)
            # 양의 상관관계 우선 (트렌드 추종 확인)
            if corr > best_corr:
                best_corr = corr
                best_lag_weeks = lag
        
        best_lag_days = best_lag_weeks * 7
        
        # 2. Local Correlation (최근 12주, 약 3달)
        # Global에서 찾은 '최적 시차'를 적용했을 때, 최근에도 그 관계가 유효한가?
        recent_window = 12
        df['L_Z_Shifted'] = df['L_Z'].shift(best_lag_weeks)
        df_recent = df.iloc[-recent_window:]
        
        if len(df_recent) < recent_window: return None
        
        recent_corr = df_recent['P_Z'].corr(df_recent['L_Z_Shifted'])
        
        # 3. 괴리율
        last_val = df.iloc[-1]
        gap_z = last_val['P_Z'] - last_val['L_Z_Shifted']
        
        # 4. Regime 판단 (Global과 Recent 비교)
        # Global이 양수인데 Recent가 음수면 -> 이탈
        # Global 자체가 음수면 -> 역상관
        
        if best_corr < 0:
            regime = "Inverse"
        elif recent_corr > 0.6:
            regime = "Strong Sync"
        elif recent_corr > 0.2:
            regime = "Sync"
        elif recent_corr < -0.2:
            regime = "Divergence" # 이탈
        else:
            regime = "Neutral"

        return {
            "optimal_lag": best_lag_days,
            "global_corr": best_corr,   # 전체 기간 (기준)
            "recent_corr": recent_corr, # 최근 90일 (현재)
            "gap_z": gap_z,
            "regime": regime
        }

    except Exception:
        return None

# -----------------------------------------------------------
# Rendering
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
        elif "G3" in liq_option:
            liq_v = flt(df_m['G3_Asset_YoY'])
            liq_name, liq_color = "G3 Assets", "#FFD700"
        else:
            liq_v = flt(df_m['Fed_Net_YoY'])
            liq_name, liq_color = "Fed Net", "#00FF7F"

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

        st.markdown("---")
        st.subheader("🛰️ Matrix Quant Analytics (Global vs Local)")
        st.caption("비교 기준: Historical (2021~, 전체 역사) ↔ Recent (Last 90d, 최근 분기)")
        
        with st.spinner("Calculating Correlations..."):
            
            liquidity_sources = [
                ("🇺🇸 Fed Net Liq", df_m['Fed_Net_YoY']),
                ("🏛️ G3 Assets",    df_m['G3_Asset_YoY']),
                ("🌍 Global M2",    df_m['Global_M2_YoY'])
            ]

            if active_assets:
                asset_tabs = st.tabs([f"{a['name']}" for a in active_assets])
                
                for tab, asset in zip(asset_tabs, active_assets):
                    with tab:
                        raw_asset_series = raw.get(asset['id'], pd.Series(dtype=float))
                        if raw_asset_series.empty:
                            st.warning("데이터 부족")
                            continue
                        
                        results = []
                        for liq_label, liq_data in liquidity_sources:
                            res = run_quant_analysis_cached(liq_data, raw_asset_series)
                            if res:
                                res['label'] = liq_label
                                results.append(res)
                        
                        if not results:
                            st.info("분석 불가")
                            continue

                        cols = st.columns(len(results))
                        best_res = max(results, key=lambda x: x['global_corr'])
                        
                        for i, res in enumerate(results):
                            with cols[i]:
                                if res == best_res: st.markdown(f"#### ⭐ {res['label']}")
                                else: st.markdown(f"#### {res['label']}")

                                st.metric("Optimal Lag", f"{res['optimal_lag']} days")
                                
                                # Global vs Recent 비교
                                st.metric("Hist. Corr (4y)", f"{res['global_corr']:.2f}", help="2021년부터 현재까지의 전체 상관관계")
                                st.metric("Recent Corr (90d)", f"{res['recent_corr']:.2f}", 
                                          delta=f"{res['recent_corr'] - res['global_corr']:.2f}",
                                          delta_color="normal",
                                          help="최근 90일간의 상관관계 (Global 대비 변화량)")
                                
                                regime_icon = "🟢" if res['regime']=="Sync" or res['regime']=="Strong Sync" else ("⚠️" if res['regime']=="Divergence" else "📉")
                                st.metric("Regime Status", f"{regime_icon} {res['regime']}")
                                
                                gap_state = "High" if res['gap_z'] > 1.0 else ("Low" if res['gap_z'] < -1.0 else "Fair")
                                st.metric("Z-Gap", f"{res['gap_z']:+.2f} σ", gap_state, delta_color="inverse")
                        
                        # Insight Text Generation
                        if best_res['global_corr'] < 0:
                            insight_msg = f"**{asset['name']}**는 역사적으로 유동성 지표들과 **역상관(Inverse)** 관계를 보여왔습니다. 유동성이 빠질 때 오히려 오르는 경향이 있거나, 다른 요인에 더 크게 반응합니다."
                        else:
                            insight_msg = f"**{asset['name']}**는 역사적으로 **{best_res['label']}**와 가장 강한 동행(**{best_res['global_corr']:.2f}**)을 보입니다."
                            if best_res['recent_corr'] < 0.2:
                                insight_msg += f" 그러나 최근 90일간은 상관관계가 약화되어 **이탈(Divergence)** 조짐이 보입니다."
                            else:
                                insight_msg += f" 최근에도 이 관계는 견고하게 유지되고 있습니다."

                        st.info(f"**Insight:** {insight_msg}")

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
        st.error("❌ 비트코인 로드 실패")

except Exception as e:
    st.error(f"⚠️ 차트 렌더링 중 오류 발생: {str(e)}")
