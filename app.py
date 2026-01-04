import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import json
import io
import warnings
import time
import ccxt
from datetime import date

# 1. 환경 설정
warnings.filterwarnings("ignore")
st.set_page_config(page_title="GM Dual Core", layout="wide")

st.title("🏛️ Grand Master: Dual Liquidity Terminal")
st.caption("Ver 11.0 | 🇺🇸 Fed Net vs 🌍 G3 Global (Selectable) | Bithumb KRW")

# 사이드바 설정 (유동성 선택)
st.sidebar.header("⚙️ Liquidity Engine")
liq_option = st.sidebar.radio(
    "분석할 유동성 지표 선택:",
    ("🇺🇸 Fed Net Liquidity (미국 실질 유동성)", "🌍 G3 Global Liquidity (미+유+일 총량)"),
    index=0
)

# 2. 통합 데이터 수집 (캐시 적용)
@st.cache_data(ttl=3600, show_spinner="글로벌 금융 데이터 통합 수집 중...")
def fetch_master_data():
    d = {}
    
    # [A] Crypto (Bithumb KRW via ccxt)
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

    # [B] FRED Data (Fed Net + G3 계산에 필요한 모든 소스)
    fred_ids = {
        'fed': 'WALCL',         # Fed Total Assets
        'tga': 'WTREGEN',       # Treasury General Account
        'rrp': 'RRPONTSYD',     # Reverse Repo
        'ecb': 'ECBASSETSW',    # ECB Total Assets
        'boj': 'JPNASSETS',     # BOJ Total Assets
        'eur_usd': 'DEXUSEU',   # EUR/USD
        'usd_jpy': 'DEXJPUS'    # USD/JPY
    }

    def get_fred(id):
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"
            r = requests.get(url, timeout=12)
            df = pd.read_csv(io.StringIO(r.text), index_col=0, parse_dates=True)
            return df.squeeze().resample('D').interpolate(method='time').tz_localize(None)
        except: return pd.Series(dtype=float)

    for key, val in fred_ids.items():
        d[key] = get_fred(val)

    # [C] Nasdaq
    try:
        import yfinance as yf
        ns = yf.download("^IXIC", period="max", progress=False)
        close = ns['Close'] if 'Close' in ns.columns else ns
        d['nasdaq'] = close.tz_localize(None) if not close.empty else pd.Series(dtype=float)
    except: d['nasdaq'] = pd.Series(dtype=float)

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

# 3. 데이터 가공 및 지표 산출
if not raw.get('btc', pd.Series()).empty:
    
    # --- 통합 데이터프레임 생성 (주간 수요일 기준) ---
    df_m = pd.DataFrame(index=raw['fed'].resample('W-WED').last().index)
    
    # 데이터 채우기
    for k in ['fed', 'tga', 'rrp', 'ecb', 'boj']:
        if k in raw: df_m[k] = raw[k].resample('W-WED').last()
    
    df_m['eur_usd'] = raw['eur_usd'].resample('W-WED').mean()
    df_m['usd_jpy'] = raw['usd_jpy'].resample('W-WED').mean()
    df_m = df_m.fillna(method='ffill')

    # [Logic 1] Fed Net Liquidity (기존)
    # Unit: Trillions
    df_m['Fed_Net_Tril'] = (
        df_m['fed'] / 1000 - 
        df_m.get('tga', 0) / 1000 - 
        df_m.get('rrp', 0) / 1_000_000
    )
    df_m['Fed_Net_YoY'] = df_m['Fed_Net_Tril'].pct_change(52) * 100

    # [Logic 2] G3 Global Liquidity (확장)
    # Formula: Fed + (ECB * EUR) + (BOJ / JPY)
    fed_t = df_m['fed'] / 1_000_000
    ecb_t = (df_m['ecb'] * df_m['eur_usd']) / 1_000_000
    boj_t = (df_m['boj'] * 0.0001) / df_m['usd_jpy']
    
    df_m['G3_Tril'] = fed_t + ecb_t + boj_t
    df_m['G3_YoY'] = df_m['G3_Tril'].pct_change(52) * 100

    # --- Mining Cost ---
    df_c = pd.DataFrame(index=raw['btc'].index)
    if not raw['diff'].empty:
        df_c['diff'] = raw['diff'].reindex(df_c.index).interpolate()
        halving_date = date(2024, 4, 20)
        df_c['reward'] = df_c.index.map(lambda x: 3.125 if x.date() >= halving_date else 6.25)
        df_c['cost'] = df_c['diff'] / df_c['reward']
        
        # Calibration (2022-2023 Bottom)
        sub = pd.concat([raw['btc'], df_c['cost']], axis=1).dropna()
        sub.columns = ['btc', 'cost']
        target = sub[(sub.index >= '2022-11-01') & (sub.index <= '2023-01-31')]
        k = (target['btc'] / target['cost']).min() if not target.empty else 0.0000001
        df_c['floor'] = df_c['cost'] * k

    # --- Shift -90d ---
    def shift_90(s):
        if s.empty: return pd.Series(dtype=float)
        new = s.copy()
        new.index = new.index - pd.Timedelta(days=90)
        return new

    btc_s = shift_90(raw['btc'])
    floor_s = shift_90(df_c.get('floor', pd.Series(dtype=float)))
    nasdaq_s = shift_90(raw['nasdaq'])
    doge_s = shift_90(raw['doge'])

    # --- Visualization Selection ---
    start_viz = pd.to_datetime('2018-01-01')
    def flt(s): return s[s.index >= start_viz] if not s.empty else s

    # 사용자 선택에 따른 데이터 스위칭
    if "G3" in liq_option:
        liq_series = flt(df_m['G3_YoY'])
        liq_name = "🌍 G3 Liquidity YoY"
        liq_color = "#FFD700" # Gold
        st.info("💡 **G3 모드:** 미국(Fed), 유럽(ECB), 일본(BOJ)의 총 유동성 합계를 보여줍니다. 거시적 추세 파악에 유리합니다.")
    else:
        liq_series = flt(df_m['Fed_Net_YoY'])
        liq_name = "🇺🇸 Fed Net Liquidity YoY"
        liq_color = "#00FF7F" # Spring Green (구분하기 쉽게 색상 변경)
        st.info("💡 **Fed Net 모드:** 미 연준의 자산에서 TGA/RRP를 뺀 실질 달러 유동성입니다. 비트코인 단기 상관성이 높습니다.")

    btc_v = flt(btc_s)
    fl_v = flt(floor_s)
    nd_v = flt(nasdaq_s)
    dg_v = flt(doge_s)

    # Dynamic Range (KRW)
    if not btc_v.empty:
        b_min, b_max = btc_v.min(), btc_v.max()
        b_min_dyn = max(b_min * 0.6, 1_000_000)
        b_max_dyn = b_max * 1.4
    else: b_min_dyn, b_max_dyn = 0, 1

    # Plotly Chart
    fig = go.Figure(
        layout=go.Layout(
            template="plotly_dark", height=800,
            xaxis=dict(domain=[0.0, 0.88], showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
            
            # Y1: Liquidity (Dynamic Title & Color)
            yaxis=dict(title=liq_name, title_font_color=liq_color, tickfont_color=liq_color, range=[-30, 60]),
            
            # Y2: BTC
            yaxis2=dict(title="BTC (KRW)", title_font_color="#00FFEE", tickfont_color="#00FFEE", overlaying="y", side="right", position=0.88, range=[b_min_dyn, b_max_dyn], showgrid=False, tickformat=","),
            
            # Y3: Nasdaq
            yaxis3=dict(title="Nasdaq", title_font_color="#D62780", tickfont_color="#D62780", overlaying="y", side="right", anchor="free", position=0.96, tickformat=","),
            
            # Y4: Doge
            yaxis4=dict(title="DOGE", title_font_color="orange", tickfont_color="orange", overlaying="y", side="right", anchor="free", position=1.0, type="log"),
            
            legend=dict(orientation="h", y=1.12, x=0.01, bgcolor="rgba(0,0,0,0)"),
            hovermode="x unified",
            margin=dict(l=60, r=140, t=100, b=60)
        )
    )

    # 1. Liquidity Trace (Switched)
    if not liq_series.empty:
        fig.add_trace(go.Scatter(
            x=liq_series.index, y=liq_series, name=liq_name,
            line=dict(color=liq_color, width=3),
            fill='tozeroy', fillcolor=f"rgba{tuple(int(liq_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.15,)}",
            yaxis='y'
        ))

    # 2. Asset Traces
    if not btc_v.empty:
        fig.add_trace(go.Scatter(x=btc_v.index, y=btc_v, name="BTC (-90d)", line=dict(color='#00FFEE', width=4), yaxis='y2'))
    if not fl_v.empty:
        fig.add_trace(go.Scatter(x=fl_v.index, y=fl_v, name="Mining Cost Floor", line=dict(color='red', width=2, dash='dot'), yaxis='y2'))
    if not nd_v.empty:
        fig.add_trace(go.Scatter(x=nd_v.index, y=nd_v, name="Nasdaq (-90d)", line=dict(color='#D62780', width=2), yaxis='y3'))
    if not dg_v.empty:
        fig.add_trace(go.Scatter(x=dg_v.index, y=dg_v, name="DOGE (-90d)", line=dict(color='orange', width=2), yaxis='y4'))

    st.plotly_chart(fig, use_container_width=True)
    st.success(f"✅ 분석 모드 적용 완료: {liq_option}")

else:
    st.error("데이터 로드 실패")
