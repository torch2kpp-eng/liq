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

warnings.filterwarnings("ignore")

st.set_page_config(page_title="GM Anti-Block", layout="wide")
st.title("🏛️ Grand Master: Anti-Block Terminal")
st.caption("Ver 15.1 | Mining Cost 제거 | Kraken 정상 페이징 | 축 위치 동적 | 로그 안전 처리")

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
    help="양수(+)는 과거 데이터를 현재와 매칭, 음수(-)는 미래 예측 시뮬레이션."
)

st.sidebar.markdown("---")
st.sidebar.write("3. 표시할 자산 (Right Axes)")

ASSETS_CONFIG = [
    {'id': 'nasdaq', 'name': 'Nasdaq', 'symbol': 'IXIC', 'source': 'hybrid', 'color': '#D62780', 'type': 'index', 'default': True},
    {'id': 'gold', 'name': 'Gold', 'symbol': 'XAU/USD', 'source': 'kraken', 'color': '#FFD700', 'type': 'metal', 'default': True},
    {'id': 'silver', 'name': 'Silver', 'symbol': 'XAG/USD', 'source': 'kraken', 'color': '#C0C0C0', 'type': 'metal', 'default': True},
    {'id': 'btc', 'name': 'BTC', 'symbol': 'BTC/KRW', 'source': 'bithumb', 'color': '#00FFEE', 'type': 'crypto', 'default': True},
    {'id': 'doge', 'name': 'DOGE', 'symbol': 'DOGE/KRW', 'source': 'bithumb', 'color': '#FFA500', 'type': 'crypto', 'default': True},
    {'id': 'eth', 'name': 'ETH', 'symbol': 'ETH/KRW', 'source': 'bithumb', 'color': '#627EEA', 'type': 'crypto', 'default': False},
    {'id': 'link', 'name': 'LINK', 'symbol': 'LINK/KRW', 'source': 'bithumb', 'color': '#2A5ADA', 'type': 'crypto', 'default': False},
    {'id': 'ada', 'name': 'ADA', 'symbol': 'ADA/KRW', 'source': 'bithumb', 'color': '#0033AD', 'type': 'crypto', 'default': False},
    {'id': 'xrp', 'name': 'XRP', 'symbol': 'XRP/KRW', 'source': 'bithumb', 'color': '#00AAE4', 'type': 'crypto', 'default': False},
]

selected_assets = {}
for asset in ASSETS_CONFIG:
    selected_assets[asset['id']] = st.sidebar.checkbox(f"{asset['name']}", value=asset['default'])

# -----------------------------------------------------------
# 데이터 수집 (개선 적용)
# -----------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="데이터 보안 접속 중...")
def fetch_master_data():
    d = {}
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36'
    }

    bithumb = ccxt.bithumb({'enableRateLimit': True})
    kraken = ccxt.kraken({'enableRateLimit': True})

    # 1. Bithumb Fetcher (기존 유지)
    def fetch_bithumb(symbol_code):
        all_data = []
        try:
            since = bithumb.parse8601('2017-01-01T00:00:00Z')
            while True:
                ohlcv = bithumb.fetch_ohlcv(symbol_code, '1d', since=since, limit=1000)
                if not ohlcv: break
                all_data.extend(ohlcv)
                last_ts = ohlcv[-1][0]
                if last_ts >= (time.time() * 1000) - 86400000: break
                since = last_ts + 1
                time.sleep(0.08)
        except: pass
        if not all_data: return pd.Series(dtype=float)
        df = pd.DataFrame(all_data, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df.drop_duplicates(subset=['timestamp']).set_index('timestamp')['close'].tz_localize(None)

    # 2. Kraken Fetcher - 정상 페이징 방식 (개선)
    def fetch_kraken(symbol, start_year=2015):
        all_data = []
        since = kraken.parse8601(f'{start_year}-01-01T00:00:00Z')
        max_attempts = 8
        attempt = 0

        while attempt < max_attempts:
            try:
                ohlcv = kraken.fetch_ohlcv(symbol, '1d', since=since, limit=720)
                if not ohlcv: break
                all_data.extend(ohlcv)
                since = ohlcv[-1][0] + 86400000  # 다음 날부터
                time.sleep(1.8)  # Kraken rate limit 엄격
            except Exception as e:
                attempt += 1
                time.sleep(3 + attempt * 2)  # backoff
                continue

        if not all_data: return pd.Series(dtype=float)
        df = pd.DataFrame(all_data, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df.drop_duplicates('timestamp').set_index('timestamp').sort_index()['close'].tz_localize(None)

    # 3. FRED Fetcher - 강화 (재시도 1회)
    def get_fred_secure(fid, retries=1):
        for _ in range(retries + 1):
            try:
                url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fid}"
                r = requests.get(url, headers=headers, timeout=15)
                r.raise_for_status()
                df = pd.read_csv(io.StringIO(r.text), index_col=0, parse_dates=True)
                return df.squeeze().resample('D').interpolate(method='time').tz_localize(None)
            except:
                time.sleep(2)
        return pd.Series(dtype=float)

    fred_ids = {
        'fed': 'WALCL', 'tga': 'WTREGEN', 'rrp': 'RRPONTSYD',
        'ecb': 'ECBASSETSW', 'boj': 'JPNASSETS',
        'm2_us': 'M2SL', 'm3_eu': 'MABMM301EZM189S', 'm3_jp': 'MABMM301JPM189S',
        'eur_usd': 'DEXUSEU', 'usd_jpy': 'DEXJPUS',
        'nasdaq_fred': 'NASDAQCOM'
    }

    for key, val in fred_ids.items():
        d[key] = get_fred_secure(val)

    # Nasdaq hybrid
    if not d['nasdaq_fred'].empty:
        d['nasdaq'] = d['nasdaq_fred']
    else:
        try:
            import yfinance as yf
            df = yf.download("^IXIC", period="max", progress=False)
            d['nasdaq'] = df['Close'].tz_localize(None) if 'Close' in df else pd.Series(dtype=float)
        except:
            d['nasdaq'] = pd.Series(dtype=float)

    # Crypto & Commodities
    for asset in ASSETS_CONFIG:
        if asset['source'] == 'bithumb':
            d[asset['id']] = fetch_bithumb(asset['symbol'])
        elif asset['source'] == 'kraken':
            d[asset['id']] = fetch_kraken(asset['symbol'])

    # Difficulty (필요시 유지, Mining Cost 제거로 사용 안 함)
    try:
        with open('difficulty (1).json', 'r', encoding='utf-8') as f:
            js = json.load(f)['difficulty']
        df_js = pd.DataFrame(js)
        df_js['Date'] = pd.to_datetime(df_js['x'], unit='ms').dt.tz_localize(None)
        d['diff'] = df_js.set_index('Date').sort_index()['y']
    except:
        d['diff'] = pd.Series(dtype=float)

    return d

raw = fetch_master_data()

# -----------------------------------------------------------
# 데이터 가공 & 차트
# -----------------------------------------------------------
if not raw.get('btc', pd.Series()).empty:

    # 유동성 로직 (기존 유지)
    df_m = pd.DataFrame(index=raw['fed'].resample('W-WED').last().index)
    for k in ['fed', 'tga', 'rrp', 'ecb', 'boj', 'm2_us', 'm3_eu', 'm3_jp', 'eur_usd', 'usd_jpy']:
        if k in raw:
            df_m[k] = raw[k].reindex(df_m.index, method='ffill')

    df_m = df_m.fillna(method='ffill')

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

    # Shift 적용
    def apply_shift(s, days):
        if s.empty: return pd.Series(dtype=float)
        new_s = s.copy()
        new_s.index = new_s.index - pd.Timedelta(days=days)
        return new_s

    processed_assets = {}
    for asset in ASSETS_CONFIG:
        processed_assets[asset['id']] = apply_shift(raw.get(asset['id'], pd.Series(dtype=float)), shift_days)

    # 차트
    st.subheader(f"📊 Integrated Strategy Chart (Shift: {shift_days} days)")

    start_viz = pd.to_datetime('2019-01-01')
    def flt(s): return s[s.index >= start_viz] if not s.empty else s

    # 유동성 선택
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

    if not liq_v.empty:
        l_min, l_max = liq_v.min(), liq_v.max()
        l_span = l_max - l_min if l_max != l_min else 20
        l_rng = [l_min - l_span * 0.1, l_max + l_span * 0.1]
    else:
        l_rng = [-20, 20]

    # 개선된 축 위치 계산
    active_assets = [a for a in ASSETS_CONFIG if selected_assets[a['id']]]
    num_active = len(active_assets)
    if num_active == 0:
        domain_end = 0.88
    else:
        spacing = max(0.04, min(0.10, 0.92 / max(1, num_active)))
        domain_end = 1.0 - (num_active * spacing) + spacing / 2
        domain_end = max(0.55, min(0.92, domain_end))

    common_spike = dict(
        showspikes=True, spikemode='across', spikesnap='cursor',
        spikethickness=1, spikecolor='red', spikedash='dash'
    )

    layout = go.Layout(
        template="plotly_dark", height=800,
        xaxis=dict(domain=[0.0, domain_end], showgrid=True, gridcolor='rgba(128,128,128,0.2)', **common_spike),
        yaxis=dict(title=liq_name, title_font_color=liq_color, tickfont_color=liq_color,
                   range=l_rng, showgrid=False, **common_spike),
        legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
        margin=dict(l=50, r=20, t=80, b=50)
    )

    fig = go.Figure(layout=layout)

    # Liquidity Trace
    if not liq_v.empty:
        h = liq_color.lstrip('#')
        rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        fig.add_trace(go.Scatter(
            x=liq_v.index, y=liq_v, name=liq_name,
            line=dict(color=liq_color, width=3),
            fill='tozeroy', fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.15)",
            yaxis='y', hoverinfo='none'
        ))

    # Assets Loop
    current_pos = domain_end
    for i, asset in enumerate(active_assets):
        data = flt(processed_assets[asset['id']])
        if data.empty: continue

        axis_name = f'yaxis{i+2}'
        axis_key = f'y{i+2}'

        # 로그 스케일 안전 처리 (개선)
        is_log = (asset['id'] == 'doge')
        if is_log:
            valid_data = data[data > 0]
            if valid_data.empty:
                continue
            d_min = valid_data.min()
            d_max = valid_data.max()
            log_min = np.log10(d_min)
            log_max = np.log10(d_max)
            span = log_max - log_min
            rng = [log_min - span * 0.15, log_max + span * 0.25]
            type_val = "log"
        else:
            d_min, d_max = data.min(), data.max()
            span = d_max - d_min if d_max != d_min else 1
            rng = [max(d_min - span * 0.4, 0), d_max + span * 0.15]
            type_val = "linear"

        fig.update_layout({
            axis_name: dict(
                title=dict(text=asset['name'], font=dict(color=asset['color'])),
                tickfont=dict(color=asset['color']),
                overlaying="y", side="right", anchor="free", position=current_pos,
                range=rng, type=type_val, showgrid=False, tickformat=",",
                **common_spike
            )
        })

        fig.add_trace(go.Scatter(
            x=data.index, y=data,
            name=asset['name'],
            line=dict(color=asset['color'], width=2),
            yaxis=axis_key,
            hoverinfo='none'
        ))

        current_pos += spacing

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔍 데이터 연결 상태"):
        for asset in ASSETS_CONFIG:
            s = processed_assets[asset['id']]
            if s.empty:
                st.error(f"❌ {asset['name']}: 로드 실패 (Source: {asset['source']})")
            else:
                st.success(f"✅ {asset['name']}: 로드 성공 ({len(s)} rows)")

else:
    st.error("❌ 비트코인(Bithumb) 데이터 로드 실패. 잠시 후 새로고침하세요.")
