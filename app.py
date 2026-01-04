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
st.set_page_config(page_title="GM Time-Warp", layout="wide")

st.title("🏛️ Grand Master: Time-Warp Terminal")
st.caption("Ver 14.0 | Time Shift 제어 | Gold, Silver 및 Major Altcoins 통합")

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

# 2. 타임 시프트 설정 (핵심 기능)
st.sidebar.markdown("---")
st.sidebar.write("2. Time Shift (Days)")
shift_days = st.sidebar.number_input(
    "자산 가격 이동 (일)", 
    min_value=-365, max_value=365, value=90, step=7,
    help="양수(+)를 입력하면 차트가 왼쪽으로(과거 데이터를 현재 유동성과 매칭), 음수(-)는 오른쪽으로 이동합니다."
)

# 3. 자산 선택 (순서대로 정의)
st.sidebar.markdown("---")
st.sidebar.write("3. 표시할 자산 (Right Axes)")

# 자산 메타데이터 정의 (순서: 나스닥, GOLD, SILVER, BTC, DOGE, ETH, LINK, ADA, XRP)
ASSETS_CONFIG = [
    {'id': 'nasdaq', 'name': 'Nasdaq', 'symbol': 'IXIC', 'color': '#D62780', 'type': 'index', 'default': True},
    {'id': 'gold',   'name': 'Gold',   'symbol': 'GOLD', 'color': '#FFD700', 'type': 'metal', 'default': False},
    {'id': 'silver', 'name': 'Silver', 'symbol': 'SLV',  'color': '#C0C0C0', 'type': 'metal', 'default': False},
    {'id': 'btc',    'name': 'BTC',    'symbol': 'BTC',  'color': '#00FFEE', 'type': 'crypto', 'default': True},
    {'id': 'doge',   'name': 'DOGE',   'symbol': 'DOGE', 'color': '#FFA500', 'type': 'crypto', 'default': True},
    {'id': 'eth',    'name': 'ETH',    'symbol': 'ETH',  'color': '#627EEA', 'type': 'crypto', 'default': False},
    {'id': 'link',   'name': 'LINK',   'symbol': 'LINK', 'color': '#2A5ADA', 'type': 'crypto', 'default': False},
    {'id': 'ada',    'name': 'ADA',    'symbol': 'ADA',  'color': '#0033AD', 'type': 'crypto', 'default': False},
    {'id': 'xrp',    'name': 'XRP',    'symbol': 'XRP',  'color': '#00AAE4', 'type': 'crypto', 'default': False},
]

# 사용자 선택 받기
selected_assets = {}
for asset in ASSETS_CONFIG:
    selected_assets[asset['id']] = st.sidebar.checkbox(f"{asset['name']}", value=asset['default'])

# -----------------------------------------------------------
# 2. 데이터 수집
# -----------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="전 자산 데이터 통합 수집 중...")
def fetch_master_data():
    d = {}
    
    # [A] Crypto (Bithumb KRW via ccxt) - 루프 처리
    exchange = ccxt.bithumb({'enableRateLimit': True})
    
    crypto_list = [a for a in ASSETS_CONFIG if a['type'] == 'crypto']
    
    def fetch_ohlcv_ccxt(symbol_code):
        # symbol_code: 'BTC', 'ETH' ... -> 'BTC/KRW'
        pair = f"{symbol_code}/KRW"
        all_data = []
        # 알트코인은 상장일이 다를 수 있으므로 넉넉히 2017년부터 시도하되 없으면 빈값 리턴
        since = exchange.parse8601('2017-01-01T00:00:00Z')
        
        while True:
            try:
                ohlcv = exchange.fetch_ohlcv(pair, '1d', since=since, limit=1000)
                if not ohlcv: break
                all_data.extend(ohlcv)
                since = ohlcv[-1][0] + 1
                time.sleep(0.05) # 속도 최적화
            except: break
        
        if not all_data: return pd.Series(dtype=float)
        df = pd.DataFrame(all_data, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df.set_index('timestamp')['close'].tz_localize(None)

    for item in crypto_list:
        d[item['id']] = fetch_ohlcv_ccxt(item['symbol'])

    # [B] FRED Data (Liquidity + Nasdaq + Gold/Silver)
    # Nasdaq: NASDAQCOM
    # Gold: GOLDAMGBD228NLBM (London Bullion Market, PM Fix) - 신뢰도 높음
    # Silver: SLVPRUSD (London Fix)
    fred_ids = {
        'fed': 'WALCL', 'tga': 'WTREGEN', 'rrp': 'RRPONTSYD',
        'ecb': 'ECBASSETSW', 'boj': 'JPNASSETS', 
        'm2_us': 'M2SL', 'm3_eu': 'MABMM301EZM189S', 'm3_jp': 'MABMM301JPM189S',
        'eur_usd': 'DEXUSEU', 'usd_jpy': 'DEXJPUS',
        'nasdaq': 'NASDAQCOM',
        'gold': 'GOLDAMGBD228NLBM',
        'silver': 'SLVPRUSD'
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
    
    # --- 유동성 로직 (기존 유지) ---
    df_m = pd.DataFrame(index=raw['fed'].resample('W-WED').last().index)
    for k in list(raw.keys()):
        # 자산 데이터는 제외하고 거시지표만 병합
        if k not in [a['id'] for a in ASSETS_CONFIG] and k != 'diff':
            df_m[k] = raw[k].reindex(df_m.index, method='ffill')

    df_m['eur_usd'] = raw['eur_usd'].resample('W-WED').mean().reindex(df_m.index, method='ffill')
    df_m['usd_jpy'] = raw['usd_jpy'].resample('W-WED').mean().reindex(df_m.index, method='ffill')
    df_m = df_m.fillna(method='ffill')

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

    # --- Mining Cost (BTC only) ---
    df_c = pd.DataFrame(index=raw['btc'].index)
    if not raw['diff'].empty:
        df_c['diff'] = raw['diff'].reindex(df_c.index).interpolate()
        halving_date = date(2024, 4, 20)
        df_c['reward'] = df_c.index.map(lambda x: 3.125 if x.date() >= halving_date else 6.25)
        df_c['cost'] = df_c['diff'] / df_c['reward']
        sub = pd.concat([raw['btc'], df_c['cost']], axis=1).dropna()
        k = (sub.iloc[:,0] / sub.iloc[:,1]).min() if not sub.empty else 0.0000001
        df_c['floor'] = df_c['cost'] * k

    # -----------------------------------------------------------
    # [핵심] Dynamic Time Shift Function
    # -----------------------------------------------------------
    def apply_shift(s, days):
        if s is None or s.empty: return pd.Series(dtype=float)
        new_s = s.copy()
        # 입력된 일수(days)만큼 index를 뒤로 미룸 (Lag)
        new_s.index = new_s.index - pd.Timedelta(days=days)
        return new_s

    # 자산 데이터 시프트 적용
    processed_assets = {}
    for asset in ASSETS_CONFIG:
        raw_series = raw.get(asset['id'], pd.Series(dtype=float))
        processed_assets[asset['id']] = apply_shift(raw_series, shift_days)
    
    # Cost Floor도 시프트
    floor_s = apply_shift(df_c.get('floor', pd.Series(dtype=float)), shift_days)

    # 4. 차트 생성
    st.subheader(f"📊 Integrated Strategy Chart (Shift: {shift_days} days)")
    
    start_viz = pd.to_datetime('2018-01-01')
    def flt(s): return s[s.index >= start_viz] if not s.empty else s

    # 유동성 데이터
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

    # 유동성 축 범위 계산
    if not liq_v.empty:
        l_min, l_max = liq_v.min(), liq_v.max()
        l_span = l_max - l_min if l_max != l_min else 1
        l_rng = [l_min - (l_span * 0.1), l_max + (l_span * 0.1)]
    else: l_rng = [-20, 20]

    # -----------------------------------------------------------
    # [Dynamic Axis Allocation Loop]
    # -----------------------------------------------------------
    # 활성화된 자산 리스트 필터링 (순서 보장)
    active_assets = [a for a in ASSETS_CONFIG if selected_assets[a['id']]]
    
    # 축 공간 계산
    num_axes = len(active_assets)
    # 축이 많아질수록 마진을 조금 줄여서 차트 공간 확보
    margin_per_axis = 0.06 if num_axes > 4 else 0.08 
    domain_end = 1.0 - (num_axes * margin_per_axis)
    if domain_end < 0.5: domain_end = 0.5 # 최소 50%는 차트 영역

    # 공통 스파이크 스타일
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

    # 1. Liquidity Trace
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

    # 2. Assets Trace Loop
    current_pos = domain_end
    
    # Plotly Y-axes는 yaxis2, yaxis3... 순서로 이름이 붙음
    # i는 0부터 시작하므로 axis_name은 'yaxis' + str(i+2)
    
    for i, asset in enumerate(active_assets):
        data = flt(processed_assets[asset['id']])
        if data.empty: continue
        
        axis_name = f'yaxis{i+2}'
        axis_key = f'y{i+2}'
        
        # 범위 계산
        d_min, d_max = data.min(), data.max()
        if d_min <= 0: d_min = 0.0001
        
        # 스케일링 로직
        # Crypto/Index: Linear가 기본이지만, 진폭 크면 Log 고려 가능. 
        # 사용자의 요청은 '도지코인 스케일'이었음. 도지만 Log로 처리하고 나머지는 Linear?
        # 혹은 자산 타입에 따라 결정. 여기서는 DOGE만 Log, 나머지는 Linear + Buffer
        
        is_log = (asset['id'] == 'doge') # 도지만 로그
        
        if is_log:
            log_min, log_max = np.log10(d_min), np.log10(d_max)
            span = log_max - log_min
            rng = [log_min - (span * 0.1), log_max + (span * 0.2)]
            type_val = "log"
        else:
            span = d_max - d_min
            if span == 0: span = 1
            rng = [max(d_min - (span * 0.4), 0), d_max + (span * 0.1)] # 아래쪽 40% 버퍼 (겹침 방지)
            type_val = "linear"

        # 축 업데이트
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
        
        # 그래프 그리기
        fig.add_trace(go.Scatter(
            x=data.index, y=data, 
            name=f"{asset['name']} ({shift_days}d)", 
            line=dict(color=asset['color'], width=2), 
            yaxis=axis_key,
            hoverinfo='none'
        ))
        
        # BTC인 경우 Cost Floor 추가
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
    st.success(f"✅ 설정 적용: {shift_days}일 이동 | {len(active_assets)}개 자산 표시")

else:
    st.error("데이터 로드 실패")
