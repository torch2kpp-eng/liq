import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import yfinance as yf
import json
import numpy as np
import io
import warnings

# 1. 환경 설정
warnings.filterwarnings("ignore")
st.set_page_config(page_title="GM Multi-Scale Terminal", layout="wide")

st.title("🏛️ Grand Master: Multi-Scale Analysis")
st.caption("Ver 8.3 | 개별 자산별 독립 스케일 적용 모드")

# 2. 데이터 수집 (캐시 유지)
@st.cache_data(ttl=3600)
def fetch_final_data():
    d = {}
    def get_upbit(symbol):
        try:
            r = requests.get(f"https://api.upbit.com/v1/candles/days?market={symbol}&count=1000", timeout=5).json()
            df = pd.DataFrame(r)
            df['Date'] = pd.to_datetime(df['candle_date_time_utc']).dt.tz_localize(None)
            return df.set_index('Date').sort_index()['trade_price'].astype(float)
        except: return pd.Series(dtype=float)

    d['btc'] = get_upbit("USDT-BTC")
    d['doge'] = get_upbit("USDT-DOGE")

    def get_fred(id):
        try:
            r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}", timeout=5)
            df = pd.read_csv(io.StringIO(r.text), index_col=0, parse_dates=True)
            return df.resample('D').interpolate().tz_localize(None)
        except: return pd.DataFrame()

    d['fed'] = get_fred('WALCL')
    d['tga'] = get_fred('WTREGEN')
    d['rrp'] = get_fred('RRPONTSYD')

    try:
        ns = yf.download("^IXIC", period="3y", progress=False)
        s = ns.xs('Close', axis=1, level=0)["^IXIC"] if isinstance(ns.columns, pd.MultiIndex) else ns['Close']
        d['nasdaq'] = s.tz_localize(None)
    except: d['nasdaq'] = pd.Series(dtype=float)

    try:
        with open('difficulty (1).json', 'r') as f:
            js = json.load(f)['difficulty']
        df_js = pd.DataFrame(js)
        df_js['Date'] = pd.to_datetime(df_js['x'], unit='ms').dt.tz_localize(None)
        d['diff'] = df_js.set_index('Date').sort_index()['y']
    except: d['diff'] = pd.Series(dtype=float)
    return d

raw = fetch_final_data()

# 3. 로직 처리 (시프트 및 유동성 YoY)
if not raw['btc'].empty:
    df_liq = raw['fed'].resample('W-WED').last()
    df_liq.columns = ['Fed']
    if not raw['tga'].empty: df_liq = df_liq.join(raw['tga'].resample('W-WED').mean().rename(columns={raw['tga'].columns[0]:'TGA'}))
    if not raw['rrp'].empty: df_liq = df_liq.join(raw['rrp'].resample('W-WED').mean().rename(columns={raw['rrp'].columns[0]:'RRP'}))
    df_liq = df_liq.fillna(method='ffill')
    df_liq['Net_Tril'] = (df_liq['Fed'] - df_liq.get('TGA', 0) - (df_liq.get('RRP', 0) * 1000)) / 1_000_000
    df_liq['YoY'] = df_liq['Net_Tril'].pct_change(52) * 100

    # 원가 계산
    df_c = pd.DataFrame(index=raw['btc'].index)
    if not raw['diff'].empty:
        df_c['diff'] = raw['diff'].resample('D').interpolate()
        df_c['reward'] = [3.125 if x >= pd.Timestamp('2024-04-20') else 6.25 for x in df_c.index]
        df_c['cost_raw'] = df_c['diff'] / df_c['reward']
        sub = pd.concat([raw['btc'], df_c['cost_raw']], axis=1).dropna()
        target = sub[(sub.index >= '2022-11-01') & (sub.index <= '2023-01-31')]
        k = (target.iloc[:,0] / target.iloc[:,1]).min() if not target.empty else 0.00000008
        df_c['floor'] = df_c['cost_raw'] * k

    # 90일 시프트 함수
    def shift_90(s):
        if s is None or s.empty: return pd.Series(dtype=float)
        new_s = s.copy(); new_s.index = new_s.index - pd.Timedelta(days=90)
        return new_s

    btc_s = shift_90(raw['btc'])
    floor_s = shift_90(df_c['floor']) if 'floor' in df_c else pd.Series(dtype=float)
    nasdaq_s = shift_90(raw['nasdaq'])
    doge_s = shift_90(raw['doge'])

    # 4. [핵심] 다중 축 차트 구성
    st.subheader("📊 Grand Master Integrated Strategy Chart")
    
    fig = go.Figure()
    start_viz = '2023-01-01'

    # [좌측 축: yaxis1] 유동성 YoY (Yellow)
    liq_v = df_liq[df_liq.index >= start_viz]
    fig.add_trace(go.Scatter(x=liq_v.index, y=liq_v['YoY'], name="Liquidity YoY", line=dict(color='gold', width=2), fill='tozeroy', fillcolor='rgba(255, 215, 0, 0.1)', yaxis="y1"))

    # [우측 축 1: yaxis2] BTC & Floor (White/Red)
    btc_v = btc_s[btc_s.index >= start_viz]
    fig.add_trace(go.Scatter(x=btc_v.index, y=btc_v, name="BTC (-90d)", line=dict(color='white', width=2.5), yaxis="y2"))
    if not floor_s.empty:
        fl_v = floor_s[floor_s.index >= start_viz]
        fig.add_trace(go.Scatter(x=fl_v.index, y=fl_v, name="Cost Floor", line=dict(color='red', width=1.5, dash='dot'), yaxis="y2"))

    # [우측 축 2: yaxis3] Nasdaq (Magenta)
    if not nasdaq_s.empty:
        nd_v = nasdaq_s[nasdaq_s.index >= start_viz]
        fig.add_trace(go.Scatter(x=nd_v.index, y=nd_v, name="Nasdaq (-90d)", line=dict(color='#D62780', width=1.5), yaxis="y3"))

    # [우측 축 3: yaxis4] Doge (Orange)
    if not doge_s.empty:
        dg_v = doge_s[doge_s.index >= start_viz]
        fig.add_trace(go.Scatter(x=dg_v.index, y=dg_v, name="DOGE (-90d)", line=dict(color='orange', width=1.2), yaxis="y4"))

    # 레이아웃 정밀 조정 (다중 축 배치)
    fig.update_layout(
        template="plotly_dark",
        height=750,
        margin=dict(l=50, r=200, t=50, b=50),
        xaxis=dict(domain=[0, 0.8]), # 우측 축 공간 확보
        yaxis=dict(title="Liquidity YoY %", titlefont=dict(color="gold"), tickfont=dict(color="gold")),
        yaxis2=dict(title="BTC (Log Scale)", type="log", overlaying="y", side="right", titlefont=dict(color="white"), tickfont=dict(color="white")),
        yaxis3=dict(title="Nasdaq", overlaying="y", side="right", anchor="free", position=0.88, titlefont=dict(color="#D62780"), tickfont=dict(color="#D62780")),
        yaxis4=dict(title="Doge (Log)", type="log", overlaying="y", side="right", anchor="free", position=0.96, titlefont=dict(color="orange"), tickfont=dict(color="orange")),
        legend=dict(orientation="h", y=1.1, x=0),
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("데이터 로딩 중입니다...")
