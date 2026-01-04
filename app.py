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
from datetime import date

# 1. 환경 설정
warnings.filterwarnings("ignore")
st.set_page_config(page_title="GM Commodity Fix", layout="wide")

st.title("🏛️ Grand Master: Commodity Fixed Terminal")
st.caption("Ver 14.1 | Gold/Silver 선물 데이터(COMEX) 연동 | 데이터 소스 최적화")

# -----------------------------------------------------------
# [사이드바 설정]
# -----------------------------------------------------------
st.sidebar.header("⚙️ Control Panel")

# 1. 유동성 지표 선택
liq_option = st.sidebar.radio(
    "1. 유동성 지표 (Left Axis)",
    (
        "🇺🇸 Fed Net Liquidity (미국 실질 유동성)", 
        "🏛️ G3 Central Bank Assets (본원통화 총량)",
        "🌍 Global M2 (실물 통화량: US+EU+JP)"
    ),
    index=2
)

# 2. 타임 시프트
st.sidebar.markdown("---")
st.sidebar.write("2. Time Shift (Days)")
shift_days = st.sidebar.number_input(
    "자산 가격 이동 (일)", 
    min_value=-365, max_value=365, value=90, step=7,
    help="양수(+)는 과거 데이터를 현재와 매칭, 음수(-)는 미래 예측 시뮬레이션."
)

# 3. 자산 선택
st.sidebar.markdown("---")
st.sidebar.write("3. 표시할 자산 (Right Axes)")

ASSETS_CONFIG = [
    {'id': 'nasdaq', 'name': 'Nasdaq', 'symbol': 'IXIC', 'color': '#D62780', 'type': 'index', 'default': True},
    {'id': 'gold',   'name': 'Gold',   'symbol': 'GC=F', 'color': '#FFD700', 'type': 'metal', 'default': True},
    {'id': 'silver', 'name': 'Silver', 'symbol': 'SI=F', 'color': '#C0C0C0', 'type': 'metal', 'default': True},
    {'id': 'btc',    'name': 'BTC',    'symbol': 'BTC',  'color': '#00FFEE', 'type': 'crypto', 'default': True},
    {'id': 'doge',   'name': 'DOGE',   'symbol': 'DOGE', 'color': '#FFA500', 'type': 'crypto', 'default': True},
    {'id': 'eth',    'name': 'ETH',    'symbol': 'ETH',  'color': '#627EEA', 'type': 'crypto', 'default': False},
    {'id': 'link',   'name': 'LINK',   'symbol': 'LINK', 'color': '#2A5ADA', 'type': 'crypto', 'default': False},
    {'id': 'ada',    'name': 'ADA',    'symbol': 'ADA',  'color': '#0033AD', 'type': 'crypto', 'default': False},
    {'id': 'xrp',    'name': 'XRP',    'symbol': 'XRP',  'color': '#00AAE4', 'type': 'crypto', 'default': False},
]

selected_assets = {}
for asset in ASSETS_CONFIG:
    selected_assets[asset['id']] = st.sidebar.checkbox(f"{asset['name']}", value=asset['default'])

# -----------------------------------------------------------
# 2. 데이터 수집
# -----------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="전 자산 데이터 통합 수집 중...")
def fetch_master_data():
    d = {}
    
    # [A] Crypto (Bithumb KRW via ccxt)
    exchange = ccxt.bithumb({'enableRateLimit': True})
    crypto_list = [a for a in ASSETS_CONFIG if a['type'] == 'crypto']
    
    def fetch_ohlcv_ccxt(symbol_code):
        pair = f"{symbol_code}/KRW"
        all_data = []
        since = exchange.parse8601('2017-01-01T00:00:00Z')
        while True:
            try:
                ohlcv = exchange.fetch_ohlcv(pair, '1d', since=since, limit=1000)
                if not ohlcv: break
                all_data.extend(ohlcv)
                since = ohlcv[-1][0] + 1
                time.sleep(0.05)
            except: break
        
        if not all_data: return pd.Series(dtype=float)
        df = pd.DataFrame(all_data, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df.set_index('timestamp')['close'].tz_localize(None)

    for item in crypto_list:
        d[item['id']] = fetch_ohlcv_ccxt(item['symbol'])

    # [B] FRED Data (Liquidity + Nasdaq)
    # Nasdaq(NASDAQCOM)은 FRED가 안정적임
    fred_ids = {
        'fed': 'WALCL', 'tga': 'WTREGEN', 'rrp': 'RRPONTSYD',
        'ecb': 'ECBASSETSW', 'boj': 'JPNASSETS', 
        'm2_us': 'M2SL', 'm3_eu': 'MABMM301EZM189S', 'm3_jp': 'MABMM301JPM189S',
        'eur_usd': 'DEXUSEU', 'usd_jpy': 'DEXJPUS',
        'nasdaq': 'NASDAQCOM'
    }

    def get_fred(id):
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text), index_col=0, parse_dates=True)
            return df.squeeze().resample('D').interpolate(method='time').tz_localize(None)
        except: return pd.Series(dtype=float)

    for key, val in fred_ids.items():
        d[key] = get_fred(val)

    # [C] Commodities (Gold/Silver) via Yahoo Finance
    # FRED 데이터 중단으로 인해 선물 데이터(GC=F, SI=F) 사용
    try:
        import yfinance as yf
        
        def get_yfinance(ticker):
            try:
                # 최근 10년치 데이터 요청
                df = yf.download(ticker, start="2015-01-01", progress=False, auto_adjust=True)
                if not df.empty:
                    if 'Close' in df.columns:
                        s = df['Close']
                    elif isinstance(df.columns, pd.MultiIndex):
                        s = df.xs('Close', axis=1, level=0)
                    else:
                        s = df.iloc[:, 0]
                    
                    if isinstance(s, pd.DataFrame): s = s.squeeze()
                    return s.tz_localize(None).resample('D').interpolate(method='time')
                return pd.Series(dtype=float)
            except: return pd.Series(dtype=float)

        d['gold'] = get_yfinance('GC=F')   # Gold Futures
        d['silver'] = get_yfinance('SI=F') # Silver Futures

    except Exception as e:
        d['gold'] = pd.Series(dtype=float)
        d['silver'] = pd.Series(dtype=float)

    # [D] Difficulty
    try:
        with open('difficulty (1).json', 'r', encoding='utf-8') as f:
            js = json.load(f)['difficulty']
        df_js = pd.DataFrame(js)
        df_js['Date'] = pd.to_datetime(df_js['x'], unit='ms').dt.tz_localize(None)
        d['diff'] = df_js.set_index('Date').sort_index()['y']
    except: d['diff'] = pd.Series(dtype=float)

    return d

raw = fetch_master_data()

# 3. 데이터 가공
if not raw.get('btc', pd.Series()).empty:
    
    # --- 유동성 로직 ---
    df_m = pd.DataFrame(index=raw['fed'].resample('W-WED').last().index)
    for k in list(raw.keys()):
        if k not in [a['id'] for a in ASSETS_CONFIG] and k != 'diff':
            df_m[k] = raw[k].reindex(df_m.index, method='ffill')

    df_m['eur_usd'] = raw['eur_usd'].resample('W-WED').mean().reindex(df_m.index, method='ffill')
    df_m['usd_jpy'] = raw['usd_jpy'].resample('W-WED').mean().reindex(df_m.index, method='ffill')
    df_m = df_m.fillna(method='ffill')

    # Liquidity Calculations
    df_m['Fed_Net_Tril'] = (df_m['fed'] / 1000 - df_m.get('tga', 0) / 1000 - df_m.get('rrp', 0) / 1_000_000)
    df_m['Fed_Net_YoY'] = df_m['Fed_Net_Tril'].pct_change(52) * 100

    fed_t = df_m['fed'] / 1_000_000
    ecb_t = (df_m['ecb'] * df_m['eur_usd']) / 1_000_000
    boj_t = (df_m['boj'] * 0.0001) / df_m['usd_jpy']
    df_m['G3_Asset_Tril'] = fed_t + ecb_t + boj_t
    df_m['G3_Asset_YoY'] = df_m['G3_Asset_Tril'].pct_change(52) * 100

    m2_us_t = df_m['m2_us'] / 1000 
    m3_eu_usd_t = (df_m['m3_eu'] * df_m['eur_usd']) / 1_000_000_000_000 
    m3_jp_usd_t = (df_m['m3_jp'] / df_m['usd_jpy']) / 1_000_000_000_000
    df_m['Global_M2_Tril'] = m2_us_t + m3_eu_usd_t + m3_jp_usd_t
    df_m['Global_M2_YoY'] = df_m['Global_M2_Tril'].pct_change(52) * 100

    # Mining Cost
    df_c = pd.DataFrame(index=raw['btc'].index)
    if not raw['diff'].empty:
        df_c['diff'] = raw['diff'].reindex(df_c.index).interpolate()
        halving_date = date(2024, 4, 20)
        df_c['reward'] = df_c.index.map(lambda x: 3.125 if x.date() >= halving_date else 6.25)
        df_c['cost'] = df_c['diff'] / df_c['reward']
        sub = pd.concat([raw['btc'], df_c['cost']], axis=1).dropna()
        k = (sub.iloc[:,0] / sub.iloc[:,1]).min() if not sub.empty else 0.0000001
        df_c['floor'] = df_c['cost'] * k

    # --- Shift Logic ---
    def apply_shift(s, days):
        if s is None or s.empty: return pd.Series(dtype=float)
        new_s = s.copy()
        new_s.index = new_s.index - pd.Timedelta(days=days)
        return new_s

    processed_assets = {}
    for asset in ASSETS_CONFIG:
        raw_series = raw.get(asset['id'], pd.Series(dtype=float))
        processed_assets[asset['id']] = apply_shift(raw_series, shift_days)
    
    floor_s = apply_shift(df_c.get('floor', pd.Series(dtype=float)), shift_days)

    # 4. 차트 생성
    st.subheader(f"📊 Integrated Strategy Chart (Shift: {shift_days} days)")
    start_viz = pd.to_datetime('2018-01-01')
    def flt(s): return s[s.index >= start_viz] if not s.empty else s

    if "Global M2" in liq_option:
        liq_v = flt(df_m['Global_M2_YoY'])
        liq_name = "🌍 Global M2 YoY"
        liq_color = "#FF4500" 
    elif "G3" in liq_option:
        liq_v = flt(df_m['G3_Asset_YoY'])
        liq_name = "🏛️ G3 Assets YoY"
        liq_color = "#FFD700" 
    else:
        liq_v = flt(df_m['Fed_Net_YoY'])
        liq_name = "🇺🇸 Fed Net Liq YoY"
        liq_color = "#00FF7F" 

    # 유동성 축 범위
    if not liq_v.empty:
        l_min, l_max = liq_v.min(), liq_v.max()
        l_span = l_max - l_min if l_max != l_min else 1
        l_rng = [l_min - (l_span * 0.1), l_max + (l_span * 0.1)]
    else: l_rng = [-20, 20]

    # --- Axes Loop ---
    active_assets = [a for a in ASSETS_CONFIG if selected_assets[a['id']]]
    num_axes = len(active_assets)
    margin_per_axis = 0.06 if num_axes > 4 else 0.08 
    domain_end = 1.0 - (num_axes * margin_per_axis)
    if domain_end < 0.5: domain_end = 0.5 

    common_spike = dict(
        showspikes=True, spikemode='across', spikesnap='cursor',
        spikethickness=1, spikecolor='red', spikedash='dash'
    )

    layout = go.Layout(
        template="plotly_dark", height=800,
        xaxis=dict(
            domain=[0.0, domain_end], 
            showgrid=True, gridcolor='rgba(128,128,128,0.2)',
            **common_spike
        ),
        yaxis=dict(
            title=dict(text=liq_name, font=dict(color=liq_color)),
            tickfont=dict(color=liq_color),
            range=l_rng, showgrid=False,
            **common_spike
        ),
        legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"),
        hovermode="x",
        margin=dict(l=50, r=20, t=80, b=50)
    )
    
    fig = go.Figure(layout=layout)

    # Liquidity
    if not liq_v.empty:
        h = liq_color.lstrip('#')
        rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        fill_rgba = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.15)"
        fig.add_trace(go.Scatter(
            x=liq_v.index, y=liq_v, name=liq_name, 
            line=dict(color=liq_color, width=3), 
            fill='tozeroy', fillcolor=fill_rgba,
            yaxis='y', hoverinfo='none'
        ))

    # Assets Loop
    current_pos = domain_end
    
    for i, asset in enumerate(active_assets):
        data = flt(processed_assets[asset['id']])
        if data.empty: continue
        
        axis_name = f'yaxis{i+2}'
        axis_key = f'y{i+2}'
        
        d_min, d_max = data.min(), data.max()
        if d_min <= 0: d_min = 0.0001
        
        # DOGE만 Log 적용
        is_log = (asset['id'] == 'doge')
        
        if is_log:
            log_min, log_max = np.log10(d_min), np.log10(d_max)
            span = log_max - log_min
            rng = [log_min - (span * 0.1), log_max + (span * 0.2)]
            type_val = "log"
        else:
            span = d_max - d_min
            if span == 0: span = 1
            rng = [max(d_min - (span * 0.4), 0), d_max + (span * 0.1)] # 아래쪽 버퍼 40%
            type_val = "linear"

        fig.update_layout({
            axis_name: dict(
                title=dict(text=asset['name'], font=dict(color=asset['color'])),
                tickfont=dict(color=asset['color']),
                overlaying="y", side="right",
                anchor="free", position=current_pos,
                range=rng, type=type_val,
                showgrid=False, tickformat=",",
                **common_spike
            )
        })
        
        fig.add_trace(go.Scatter(
            x=data.index, y=data, 
            name=f"{asset['name']}", 
            line=dict(color=asset['color'], width=2), 
            yaxis=axis_key,
            hoverinfo='none'
        ))
        
        if asset['id'] == 'btc' and not floor_s.empty:
            f_data = flt(floor_s)
            if not f_data.empty:
                fig.add_trace(go.Scatter(
                    x=f_data.index, y=f_data, name="Cost Floor", 
                    line=dict(color='red', width=1, dash='dot'), 
                    yaxis=axis_key,
                    hoverinfo='none'
                ))

        current_pos += margin_per_axis

    st.plotly_chart(fig, use_container_width=True)
    st.success(f"✅ 설정 완료: {shift_days}일 Shift | Gold/Silver(COMEX) 로드됨")

    # 진단 패널
    with st.expander("🔍 데이터 연결 상태 확인"):
        for asset in ASSETS_CONFIG:
            s = processed_assets[asset['id']]
            status = "✅ 연결됨" if not s.empty else "❌ 로드 실패"
            st.write(f"- {asset['name']}: {status} ({len(s)} rows)")

else:
    st.error("데이터 로드 실패")
