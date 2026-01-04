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
st.set_page_config(page_title="GM Final Fix", layout="wide")

st.title("🏛️ Grand Master: Final Stability")
st.caption("Ver 16.4 | 변수 스코프 에러(NameError) 수정 | 모바일 최적화 & 시각화 완료")

# -----------------------------------------------------------
# [사이드바 설정]
# -----------------------------------------------------------
st.sidebar.header("⚙️ Control Panel")

# 모바일 모드 토글
is_mobile = st.sidebar.checkbox("📱 모바일 모드 (축 공간 최소화)", value=True, help="체크 시 숫자를 짧게(150M) 표시하고 여백을 줄여 차트를 넓게 봅니다.")

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
    "자산 가격 이동 (일)", min_value=-365, max_value=365, value=90, step=7
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

    def check_timeout():
        if time.time() - GLOBAL_START > MAX_EXECUTION_TIME: return True
        return False

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
# Logic & Chart
# -----------------------------------------------------------
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

    st.subheader(f"📊
