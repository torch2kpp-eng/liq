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
st.set_page_config(page_title="GM Red Dash", layout="wide")

st.title("🏛️ Grand Master: Clean View Terminal")
st.caption("Ver 13.4 | 마우스 추적선 스타일 변경 (얇은 붉은색 파선)")

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

# 2. 자산 선택
st.sidebar.markdown("---")
st.sidebar.write("2. 표시할 자산 (Right Axes)")
show_btc = st.sidebar.checkbox("Bitcoin (BTC)", value=True)
show_doge = st.sidebar.checkbox("Dogecoin (DOGE)", value=True)
show_nasdaq = st.sidebar.checkbox("Nasdaq (IXIC)", value=True)

# -----------------------------------------------------------
# 2. 데이터 수집
# -----------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="데이터 동기화 중...")
def fetch_master_data():
    d = {}
    
    # [A] Crypto
    exchange = ccxt.bithumb({'enableRateLimit': True})
    
    def fetch_ohlcv(symbol, since_year=2017):
        all_data = []
        since = exchange.parse8601(f'{since_year}-01-01T00:00:00Z')
        while True:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, '1d', since=since, limit=1000)
                if not ohlcv: break
                all_data.extend(ohlcv)
                since = ohlcv[-1][0] + 1
                time.sleep(0.1)
            except: break
        
        if not all_data: return pd.Series(dtype=float)
        df = pd.DataFrame(all_data, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df.set_index('timestamp')['close'].tz_localize(None)

    d['btc'] = fetch_ohlcv('BTC/KRW', 2017)
    try: d['doge'] = fetch_ohlcv('DOGE/KRW', 2018)
    except: d['doge'] = pd.Series(dtype=float)

    # [B] FRED Data
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
            df = pd.read_csv(io.StringIO(r.text), index_col=0, parse_dates=True)
            return df.squeeze().resample('D').interpolate(method='time').tz_localize(None)
        except: return pd.Series(dtype=float)

    for key, val in fred_ids.items():
        d[key] = get_fred(val)

    # [C] Difficulty
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
    
    # --- 통합 데이터프레임 ---
    df_m = pd.DataFrame(index=raw['fed'].resample('W-WED').last().index)
    for k in list(raw.keys()):
        if k not in ['btc', 'doge', 'diff']:
            df_m[k] = raw[k].reindex(df_m.index, method='ffill')

    df_m['eur_usd'] = raw['eur_usd'].resample('W-WED').mean().reindex(df_m.index, method='ffill')
    df_m['usd_jpy'] = raw['usd_jpy'].resample('W-WED').mean().reindex(df_m.index, method='ffill')
    df_m = df_m.fillna(method='ffill')

    # Liquidity logics
    # 1. Fed Net
    df_m['Fed_Net_Tril'] = (df_m['fed'] / 1000 - df_m.get('tga', 0) / 1000 - df_m.get('rrp', 0) / 1_000_000)
    df_m['Fed_Net_YoY'] = df_m['Fed_Net_Tril'].pct_change(52) * 100

    # 2. G3 Assets
    fed_t = df_m['fed'] / 1_000_000
    ecb_t = (df_m['ecb'] * df_m['eur_usd']) / 1_000_000
    boj_t = (df_m['boj'] * 0.0001) / df_m['usd_jpy']
    df_m['G3_Asset_Tril'] = fed_t + ecb_t + boj_t
    df_m['G3_Asset_YoY'] = df_m['G3_Asset_Tril'].pct_change(52) * 100

    # 3. Global M2
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
        target = sub[(sub.index >= '2022-11-01') & (sub.index <= '2023-01-31')]
        k = (target.iloc[:,0] / target.iloc[:,1]).min() if not target.empty else 0.0000001
        df_c['floor'] = df_c['cost'] * k

    # Shift -90d
    def shift_90(s):
        if s.empty: return pd.Series(dtype=float)
        new = s.copy()
        new.index = new.index - pd.Timedelta(days=90)
        return new

    btc_s = shift_90(raw['btc'])
    floor_s = shift_90(df_c.get('floor', pd.Series(dtype=float)))
    nasdaq_s = shift_90(raw.get('nasdaq', pd.Series(dtype=float)))
    doge_s = shift_90(raw['doge'])

    # 4. 차트 생성
    st.subheader("📊 Integrated Strategy Chart")
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

    btc_v = flt(btc_s)
    fl_v = flt(floor_s)
    nd_v = flt(nasdaq_s)
    dg_v = flt(doge_s)

    # Ranges
    if not liq_v.empty:
        l_min, l_max = liq_v.min(), liq_v.max()
        l_span = l_max - l_min
        if l_span == 0: l_span = 1
        l_rng = [l_min - (l_span * 0.1), l_max + (l_span * 0.1)]
    else: l_rng = [-20, 20]

    if not btc_v.empty:
        b_min, b_max = btc_v.min(), btc_v.max()
        b_rng = [max(b_min * 0.6, 1_000_000), b_max * 1.4]
    else: b_rng = [0, 1]

    if not dg_v.empty:
        d_min, d_max = dg_v.min(), dg_v.max()
        if d_min <= 0: d_min = 0.0001
        log_min, log_max = np.log10(d_min), np.log10(d_max)
        span = log_max - log_min
        d_rng = [log_min - (span * 0.1), log_max + (span * 0.2)]
    else: d_rng = [-1, 1]

    # Axes Logic
    active_axes = []
    if show_btc: active_axes.append('btc')
    if show_nasdaq: active_axes.append('nasdaq')
    if show_doge: active_axes.append('doge')
    
    num_axes = len(active_axes)
    right_margin_per_axis = 0.08
    domain_end = 1.0 - (num_axes * right_margin_per_axis)
    if domain_end < 0.6: domain_end = 0.6 

    # Layout
    layout = go.Layout(
        template="plotly_dark", height=700,
        # [핵심] Spikes 설정 수정 (세로선 스타일 변경)
        xaxis=dict(
            domain=[0.0, domain_end], 
            showgrid=True, 
            gridcolor='rgba(128,128,128,0.2)',
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikethickness=1,      # 두께는 얇게 유지 (1px)
            spikecolor='red',      # 선 색상: 붉은색
            spikedash='dash'       # 선 스타일: 파선
        ),
        yaxis=dict(
            title=dict(text=liq_name, font=dict(color=liq_color)),
            tickfont=dict(color=liq_color),
            range=l_rng, showgrid=False
        ),
        legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"),
        hovermode="x",
        margin=dict(l=50, r=20, t=80, b=50)
    )
    
    fig = go.Figure(layout=layout)

    # Liquidity Trace (색상 계산 로직 분리 유지)
    if not liq_v.empty:
        h = liq_color.lstrip('#')
        rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        fill_rgba = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.15)"
        
        fig.add_trace(go.Scatter(
            x=liq_v.index, y=liq_v, name=liq_name, 
            line=dict(color=liq_color, width=3), 
            fill='tozeroy', 
            fillcolor=fill_rgba,
            yaxis='y',
            hoverinfo='none'
        ))

    current_pos = domain_end 

    if show_btc and not btc_v.empty:
        fig.update_layout(yaxis2=dict(
            title=dict(text="BTC", font=dict(color="#00FFEE")),
            tickfont=dict(color="#00FFEE"),
            overlaying="y", side="right", 
            anchor="free", position=current_pos,
            range=b_rng, showgrid=False, tickformat=","
        ))
        fig.add_trace(go.Scatter(
            x=btc_v.index, y=btc_v, name="BTC", 
            line=dict(color='#00FFEE', width=3), 
            yaxis='y2',
            hoverinfo='none'
        ))
        if not fl_v.empty:
            fig.add_trace(go.Scatter(
                x=fl_v.index, y=fl_v, name="Cost", 
                line=dict(color='red', width=1, dash='dot'), 
                yaxis='y2',
                hoverinfo='none'
            ))
        current_pos += right_margin_per_axis

    if show_nasdaq and not nd_v.empty:
        fig.update_layout(yaxis3=dict(
            title=dict(text="NDX", font=dict(color="#D62780")),
            tickfont=dict(color="#D62780"),
            overlaying="y", side="right", 
            anchor="free", position=current_pos,
            showgrid=False, tickformat=","
        ))
        fig.add_trace(go.Scatter(
            x=nd_v.index, y=nd_v, name="NDX", 
            line=dict(color='#D62780', width=2), 
            yaxis='y3',
            hoverinfo='none'
        ))
        current_pos += right_margin_per_axis

    if show_doge and not dg_v.empty:
        fig.update_layout(yaxis4=dict(
            title=dict(text="DOGE", font=dict(color="orange")),
            tickfont=dict(color="orange"),
            overlaying="y", side="right", 
            anchor="free", position=current_pos,
            type="log", range=d_rng,
            showgrid=False
        ))
        fig.add_trace(go.Scatter(
            x=dg_v.index, y=dg_v, name="DOGE", 
            line=dict(color='orange', width=2), 
            yaxis='y4',
            hoverinfo='none'
        ))
        current_pos += right_margin_per_axis

    st.plotly_chart(fig, use_container_width=True)
    st.success("✅ Clean View: 붉은색 파선(Dash) 가이드 적용 완료")

else:
    st.error("데이터 로드 실패")
