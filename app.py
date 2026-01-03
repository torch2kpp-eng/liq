import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import yfinance as yf
import json
import io
import warnings
from datetime import date

# 1. 환경 설정
warnings.filterwarnings("ignore")
st.set_page_config(page_title="GM Terminal Final", layout="wide")

st.title("🏛️ Grand Master: Multi-Axis Final")
st.caption("Ver 8.8 | 인덱스 타입 안정성 강화 및 TypeError 방지")

# 2. 데이터 수집
@st.cache_data(ttl=3600)
def fetch_data_final():
    d = {}
    
    # [A] Upbit
    def get_upbit(symbol):
        try:
            url = f"https://api.upbit.com/v1/candles/days?market={symbol}&count=1000"
            r = requests.get(url, timeout=10).json()
            if not r:
                return pd.Series(dtype=float)
            df = pd.DataFrame(r)
            df['Date'] = pd.to_datetime(df['candle_date_time_utc']).dt.tz_localize(None)
            return df.set_index('Date').sort_index()['trade_price'].astype(float)
        except Exception:
            return pd.Series(dtype=float)

    d['btc'] = get_upbit("KRW-BTC")
    d['doge'] = get_upbit("KRW-DOGE")

    # [B] FRED
    def get_fred(series_id):
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
            r = requests.get(url, timeout=10)
            df = pd.read_csv(io.StringIO(r.text), index_col=0, parse_dates=True)
            return df.squeeze().resample('D').interpolate().tz_localize(None)
        except Exception:
            return pd.Series(dtype=float)

    d['fed'] = get_fred('WALCL')
    d['tga'] = get_fred('WTREGEN')
    d['rrp'] = get_fred('RRPONTSYD')

    # [C] Nasdaq
    try:
        ns = yf.download("^IXIC", period="5y", progress=False, auto_adjust=True)
        close = ns['Close'] if 'Close' in ns.columns else ns
        s = close.tz_localize(None)
        if isinstance(s.index, pd.DatetimeIndex):
            d['nasdaq'] = s
        else:
            d['nasdaq'] = pd.Series(dtype=float)
    except Exception:
        d['nasdaq'] = pd.Series(dtype=float)

    # [D] Difficulty
    try:
        with open('difficulty (1).json', 'r') as f:
            js = json.load(f)['difficulty']
        df_js = pd.DataFrame(js)
        df_js['Date'] = pd.to_datetime(df_js['x'], unit='ms').dt.tz_localize(None)
        d['diff'] = df_js.set_index('Date').sort_index()['y']
    except Exception:
        d['diff'] = pd.Series(dtype=float)
    
    return d

raw = fetch_data_final()

# 3. 데이터 가공
if not raw['btc'].empty and isinstance(raw['btc'].index, pd.DatetimeIndex):
    # Liquidity
    df_liq = raw['fed'].resample('W-WED').last().to_frame(name='Fed')
    if not raw['tga'].empty:
        df_liq['TGA'] = raw['tga'].resample('W-WED').mean()
    if not raw['rrp'].empty:
        df_liq['RRP'] = raw['rrp'].resample('W-WED').mean()
    df_liq = df_liq.fillna(method='ffill')

    df_liq['Net_Tril'] = (
        df_liq['Fed'] / 1000 -
        df_liq.get('TGA', 0) / 1000 -
        df_liq.get('RRP', 0) / 1_000_000
    )
    df_liq['YoY'] = df_liq['Net_Tril'].pct_change(52) * 100

    # Mining Cost Floor
    df_c = pd.DataFrame(index=raw['btc'].index)
    if not raw['diff'].empty:
        df_c['diff'] = raw['diff'].reindex(df_c.index).interpolate()
        halving_date = date(2024, 4, 20)
        df_c['reward'] = df_c.index.map(lambda x: 3.125 if x.date() >= halving_date else 6.25)
        df_c['cost_raw'] = df_c['diff'] / df_c['reward']

        sub = pd.concat([raw['btc'], df_c['cost_raw']], axis=1).dropna()
        sub.columns = ['btc', 'cost_raw']
        target = sub[(sub.index >= '2022-11-01') & (sub.index <= '2023-01-31')]
        k = (target['btc'] / target['cost_raw']).min() if not target.empty else 0.0000001
        df_c['floor'] = df_c['cost_raw'] * k
    else:
        df_c['floor'] = pd.Series(dtype=float)

    # -90일 시프트 (안전하게)
    def shift_90(s):
        if s.empty or not isinstance(s.index, pd.DatetimeIndex):
            return pd.Series(dtype=float)
        new_s = s.copy()
        new_s.index = new_s.index - pd.Timedelta(days=90)
        return new_s

    btc_s = shift_90(raw['btc'])
    floor_s = shift_90(df_c.get('floor', pd.Series(dtype=float)))
    nasdaq_s = shift_90(raw['nasdaq'])
    doge_s = shift_90(raw['doge'])

    # 4. 차트 생성
    st.subheader("📊 Grand Master Integrated Strategy Chart")
    
    start_viz_dt = pd.to_datetime('2023-01-01')

    # 안전한 필터링: 인덱스가 DatetimeIndex인지 확인 후 필터
    def safe_filter(s, start_dt):
        if s.empty or not isinstance(s.index, pd.DatetimeIndex):
            return pd.Series(dtype=float)
        return s[s.index >= start_dt]

    liq_v = df_liq[df_liq.index >= start_viz_dt]['YoY'] if not df_liq.empty else pd.Series(dtype=float)
    btc_v = safe_filter(btc_s, start_viz_dt)
    fl_v = safe_filter(floor_s, start_viz_dt)
    nd_v = safe_filter(nasdaq_s, start_viz_dt)
    dg_v = safe_filter(doge_s, start_viz_dt)

    fig = go.Figure(
        layout=go.Layout(
            template="plotly_dark",
            height=720,
            xaxis=dict(domain=[0.0, 0.85], showgrid=False),
            yaxis=dict(
                title=dict(text="Liquidity YoY %", font=dict(color="#FFD700")),
                tickfont=dict(color="#FFD700"),
                range=[-30, 50]
            ),
            yaxis2=dict(
                title=dict(text="BTC (Log)", font=dict(color="white")),
                tickfont=dict(color="white"),
                overlaying="y",
                side="right",
                type="log"
            ),
            yaxis3=dict(
                title=dict(text="Nasdaq", font=dict(color="#D62780")),
                tickfont=dict(color="#D62780"),
                overlaying="y",
                side="right",
                anchor="free",
                position=0.92
            ),
            yaxis4=dict(
                title=dict(text="DOGE (Log)", font=dict(color="orange")),
                tickfont=dict(color="orange"),
                overlaying="y",
                side="right",
                anchor="free",
                position=1.0,
                type="log"
            ),
            legend=dict(orientation="h", y=1.12, x=0.01, bgcolor="rgba(0,0,0,0)"),
            hovermode="x unified",
            margin=dict(l=50, r=120, t=80, b=50)
        )
    )

    fig.add_trace(go.Scatter(x=liq_v.index, y=liq_v, name="Liquidity YoY %",
                             line=dict(color='#FFD700', width=3), fill='tozeroy',
                             fillcolor='rgba(255, 215, 0, 0.15)', yaxis='y'))

    if not btc_v.empty:
        fig.add_trace(go.Scatter(x=btc_v.index, y=btc_v, name="BTC (-90d)",
                                 line=dict(color='white', width=3), yaxis='y2'))

    if not fl_v.empty:
        fig.add_trace(go.Scatter(x=fl_v.index, y=fl_v, name="Mining Cost Floor",
                                 line=dict(color='red', width=2, dash='dot'), yaxis='y2'))

    if not nd_v.empty:
        fig.add_trace(go.Scatter(x=nd_v.index, y=nd_v, name="Nasdaq (-90d)",
                                 line=dict(color='#D62780', width=2), yaxis='y3'))

    if not dg_v.empty:
        fig.add_trace(go.Scatter(x=dg_v.index, y=dg_v, name="DOGE (-90d)",
                                 line=dict(color='orange', width=2), yaxis='y4'))

    st.plotly_chart(fig, use_container_width=True)
    st.success("✅ 시스템 정상 가동: 모든 데이터 및 차트 로딩 완료")

else:
    st.error("❌ 주요 데이터(BTC) 로드 실패 또는 인덱스 오류. 네트워크 확인 후 재시도해주세요.")
