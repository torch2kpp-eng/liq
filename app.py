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
st.set_page_config(page_title="GM Visual Lag", layout="wide")

st.title("🏛️ Grand Master: Visual Lag Terminal")
st.caption("Ver 16.0 | Time Shift 구간 시각화(Lag Box) | Futures & ETF Hybrid Engine")

# -----------------------------------------------------------
# [사이드바 설정]
# -----------------------------------------------------------
st.sidebar.header("⚙️ Control Panel")

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
    help="양수(+) 입력 시 자산 차트를 과거로 이동시킵니다. (오른쪽에 생기는 박스는 유동성이 선행하는 구간입니다)"
)

st.sidebar.markdown("---")
st.sidebar.write("3. 표시할 자산 (Right Axes)")

# Gold/Silver Hybrid Source
ASSETS_CONFIG = [
    {'id': 'nasdaq', 'name': 'Nasdaq', 'symbol': 'IXIC', 'source': 'hybrid', 'color': '#D62780', 'type': 'index', 'default': True},
    {'id': 'gold',   'name': 'Gold',   'symbol': 'GC=F', 'source': 'hybrid_metal', 'color': '#FFD700', 'type': 'metal', 'default': True},
    {'id': 'silver', 'name': 'Silver', 'symbol': 'SI=F', 'source': 'hybrid_metal', 'color': '#C0C0C0', 'type': 'metal', 'default': True},
    {'id': 'btc',    'name': 'BTC',    'symbol': 'BTC/KRW', 'source': 'bithumb', 'color': '#00FFEE', 'type': 'crypto', 'default': True},
    {'id': 'doge',   'name': 'DOGE',   'symbol': 'DOGE/KRW', 'source': 'bithumb', 'color': '#FFA500', 'type': 'crypto', 'default': True},
    {'id': 'eth',    'name': 'ETH',    'symbol': 'ETH/KRW', 'source': 'bithumb', 'color': '#627EEA', 'type': 'crypto', 'default': False},
    {'id': 'link',   'name': 'LINK',   'symbol': 'LINK/KRW', 'source': 'bithumb', 'color': '#2A5ADA', 'type': 'crypto', 'default': False},
    {'id': 'ada',    'name': 'ADA',    'symbol': 'ADA/KRW', 'source': 'bithumb', 'color': '#0033AD', 'type': 'crypto', 'default': False},
    {'id': 'xrp',    'name': 'XRP',    'symbol': 'XRP/KRW', 'source': 'bithumb', 'color': '#00AAE4', 'type': 'crypto', 'default': False},
]

selected_assets = {}
for asset in ASSETS_CONFIG:
    selected_assets[asset['id']] = st.sidebar.checkbox(f"{asset['name']}", value=asset['default'])

# -----------------------------------------------------------
# 데이터 수집 (안정화 버전)
# -----------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="데이터 소스 연결 및 시각화 준비 중...")
def fetch_master_data():
    d = {}
    START_YEAR = 2021
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36'
    }

    # Fetchers
    def get_fred(id):
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"
            r = requests.get(url, headers=headers, timeout=5)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text), index_col=0, parse_dates=True)
            return df.squeeze().resample('D').interpolate(method='time').tz_localize(None)
        except: return pd.Series(dtype=float)

    def get_yahoo(ticker):
        try:
            import yfinance as yf
            df = yf.download(ticker, start=f"{START_YEAR}-01-01", progress=False, auto_adjust=True)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    try: s = df.xs('Close', axis=1, level=0)[ticker]
                    except: s = df.iloc[:, 0]
                elif 'Close' in df.columns: s = df['Close']
                else: s = df.iloc[:, 0]
                if isinstance(s, pd.DataFrame): s = s.squeeze()
                return s.tz_localize(None).resample('D').interpolate(method='time')
            return pd.Series(dtype=float)
        except: return pd.Series(dtype=float)

    def get_metal_hybrid(symbol):
        # 1. Futures
        data = get_yahoo(symbol)
        if not data.empty: return data, "Futures"
        # 2. ETF Backup
        backup = "GLD" if "GC" in symbol else "SLV"
        data_b = get_yahoo(backup)
        if not data_b.empty: return data_b, "ETF(Backup)"
        return pd.Series(dtype=float), "Fail"

    bithumb = ccxt.bithumb({'enableRateLimit': True, 'timeout': 3000})
    def fetch_bithumb(symbol_code):
        all_data = []
        try:
            since = bithumb.parse8601(f'{START_YEAR}-01-01T00:00:00Z')
            for _ in range(10): 
                ohlcv = bithumb.fetch_ohlcv(symbol_code, '1d', since=since, limit=1000)
                if not ohlcv: break
                all_data.extend(ohlcv)
                last_ts = ohlcv[-1][0]
                if last_ts >= (time.time() * 1000) - 86400000: break
                since = last_ts + 1
                time.sleep(0.1)
        except: pass
        if not all_data: return pd.Series(dtype=float)
        df = pd.DataFrame(all_data, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df.drop_duplicates('timestamp').set_index('timestamp')['close'].tz_localize(None)

    # Execution
    fred_ids = {
        'fed': 'WALCL', 'tga': 'WTREGEN', 'rrp': 'RRPONTSYD',
        'ecb': 'ECBASSETSW', 'boj': 'JPNASSETS',
        'm2_us': 'M2SL', 'm3_eu': 'MABMM301EZM189S', 'm3_jp': 'MABMM301JPM189S',
        'eur_usd': 'DEXUSEU', 'usd_jpy': 'DEXJPUS',
        'nasdaq_fred': 'NASDAQCOM'
    }
    for k, v in fred_ids.items(): d[k] = get_fred(v)

    if not d['nasdaq_fred'].empty: d['nasdaq'] = d['nasdaq_fred']
    else: d['nasdaq'] = get_yahoo("^IXIC")

    meta_info = {}
    for asset in ASSETS_CONFIG:
        if asset['id'] == 'nasdaq': continue
        if asset['source'] == 'hybrid_metal':
            data, src = get_metal_hybrid(asset['symbol'])
            d[asset['id']] = data
            meta_info[asset['id']] = src
        elif asset['source'] == 'yahoo': d[asset['id']] = get_yahoo(asset['symbol'])
        elif asset['source'] == 'bithumb': d[asset['id']] = fetch_bithumb(asset['symbol'])

    try:
        with open('difficulty (1).json', 'r') as f:
            js = json.load(f)['difficulty']
        df_js = pd.DataFrame(js)
        df_js['Date'] = pd.to_datetime(df_js['x'], unit='ms').dt.tz_localize(None)
        d['diff'] = df_js.set_index('Date').sort_index()['y']
    except: d['diff'] = pd.Series(dtype=float)

    return d, meta_info

raw, meta = fetch_master_data()

# -----------------------------------------------------------
# Logic & Chart
# -----------------------------------------------------------
if not raw.get('btc', pd.Series()).empty:

    # Macro
    base_idx = raw['fed'].resample('W-WED').last().index
    df_m = pd.DataFrame(index=base_idx)
    for k in raw:
        if k not in [a['id'] for a in ASSETS_CONFIG] and k != 'diff':
            df_m[k] = raw[k].reindex(df_m.index, method='ffill')
    df_m = df_m.fillna(method='ffill')

    # Liquidity
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

    # Asset Shift
    def apply_shift(s, days):
        if s.empty: return pd.Series(dtype=float)
        new_s = s.copy()
        new_s.index = new_s.index - pd.Timedelta(days=days)
        return new_s

    processed = {}
    for asset in ASSETS_CONFIG:
        processed[asset['id']] = apply_shift(raw.get(asset['id'], pd.Series(dtype=float)), shift_days)

    # Chart
    st.subheader(f"📊 Integrated Strategy Chart (Shift: {shift_days}d)")
    
    start_viz = pd.to_datetime('2021-06-01') 
    def flt(s): return s[s.index >= start_viz] if not s.empty else s

    if "Global M2" in liq_option:
        liq_v = flt(df_m['Global_M2_YoY'])
        liq_name, liq_color = "🌍 Global M2 YoY", "#FF4500"
    elif "G3" in liq_option:
        liq_v = flt(df_m['G3_Asset_YoY'])
        liq_name, liq_color = "🏛️ G3 Assets YoY", "#FFD700"
    else:
        liq_v = flt(df_m['Fed_Net_YoY'])
        liq_name, liq_color = "🇺🇸 Fed Net Liq YoY", "#00FF7F"

    if not liq_v.empty:
        l_min, l_max = liq_v.min(), liq_v.max()
        l_span = max(l_max - l_min, 10)
        l_rng = [l_min - l_span*0.1, l_max + l_span*0.1]
    else: l_rng = [-20, 20]

    active_assets = [a for a in ASSETS_CONFIG if selected_assets[a['id']]]
    num_active = len(active_assets)
    if num_active == 0: domain_end = 0.95
    else:
        margin = 0.05 if num_active > 5 else 0.08
        domain_end = max(0.5, 1.0 - (num_active * margin))

    spike_settings = dict(
        showspikes=True, spikemode='across', spikesnap='cursor',
        spikethickness=1, spikecolor='red', spikedash='dash'
    )

    layout = go.Layout(
        template="plotly_dark", height=800,
        xaxis=dict(domain=[0.0, domain_end], showgrid=True, gridcolor='rgba(128,128,128,0.2)', **spike_settings),
        yaxis=dict(title=liq_name, title_font_color=liq_color, tickfont_color=liq_color, range=l_rng, showgrid=False, **spike_settings),
        legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"),
        hovermode="x",
        margin=dict(l=50, r=20, t=80, b=50)
    )

    fig = go.Figure(layout=layout)

    # 1. Liquidity Trace
    if not liq_v.empty:
        h = liq_color.lstrip('#')
        rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        fig.add_trace(go.Scatter(
            x=liq_v.index, y=liq_v, name=liq_name,
            line=dict(color=liq_color, width=3),
            fill='tozeroy', fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.15)",
            yaxis='y', hoverinfo='none'
        ))
        
        # -------------------------------------------------------
        # [핵심 기능] Visual Lag Box 추가
        # -------------------------------------------------------
        if shift_days != 0:
            last_date = liq_v.index.max()
            # Shift Days가 양수면, 자산 데이터가 과거로 밀리므로
            # 현재 시점 기준 [최근 N일]은 "자산 가격은 아직 오지 않았고, 유동성은 이미 나와있는" 구간임
            start_date = last_date - pd.Timedelta(days=abs(shift_days))
            
            fig.add_vrect(
                x0=start_date,
                x1=last_date,
                fillcolor="rgba(255, 255, 255, 0.08)", # 아주 연한 투명 회색/흰색
                layer="below",
                line_width=0,
                annotation_text=f"Lag Period: {abs(shift_days)}d",
                annotation_position="top left",
                annotation_font_color="rgba(255,255,255,0.5)"
            )
        # -------------------------------------------------------

    # 2. Assets Trace
    current_pos = domain_end
    for i, asset in enumerate(active_assets):
        data = flt(processed[asset['id']])
        if data.empty: continue

        axis_name = f'yaxis{i+2}'
        axis_key = f'y{i+2}'

        d_min, d_max = data.min(), data.max()
        if d_min <= 0: d_min = 0.0001
        
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
                title=dict(text=asset['name'], font=dict(color=asset['color'])),
                tickfont=dict(color=asset['color']),
                overlaying="y", side="right", anchor="free", position=current_pos,
                range=rng, type=t_type, showgrid=False, tickformat=",",
                **spike_settings
            )
        })

        fig.add_trace(go.Scatter(
            x=data.index, y=data, name=asset['name'],
            line=dict(color=asset['color'], width=2),
            yaxis=axis_key, hoverinfo='none'
        ))
        current_pos += margin

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔍 데이터 연결 리포트"):
        for asset in ASSETS_CONFIG:
            s = processed[asset['id']]
            if s.empty:
                st.error(f"❌ {asset['name']}: 로드 실패")
            else:
                extra = f" ({meta.get(asset['id'], 'OK')})" if asset['id'] in meta else ""
                st.success(f"✅ {asset['name']}: 로드 성공{extra}")

else:
    st.error("데이터 로드 실패")
