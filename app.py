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
from datetime import datetime

# 1. 환경 설정
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Grand Master Terminal", layout="wide")

st.title("🏛️ Grand Master: Multi-Asset Liquidity Terminal")
st.caption("Ver 8.2 | G3 유동성 & 자산간 괴리 정밀 분석 모드")

# 2. 고성능 데이터 수집 엔진 (Timeout & Cache 최적화)
@st.cache_data(ttl=3600)
def fetch_all_data_final():
    d = {}
    # [A] Upbit 데이터 (BTC, DOGE)
    def get_upbit(symbol):
        try:
            r = requests.get(f"https://api.upbit.com/v1/candles/days?market={symbol}&count=1000", timeout=3).json()
            df = pd.DataFrame(r)
            df['Date'] = pd.to_datetime(df['candle_date_time_utc']).dt.tz_localize(None)
            return df.set_index('Date').sort_index()['trade_price'].astype(float)
        except: return pd.Series(dtype=float)

    d['btc'] = get_upbit("USDT-BTC")
    d['doge'] = get_upbit("USDT-DOGE")

    # [B] FRED 데이터 (유동성)
    def get_fred(id):
        try:
            r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}", timeout=3)
            df = pd.read_csv(io.StringIO(r.text), index_col=0, parse_dates=True)
            return df.resample('D').interpolate().tz_localize(None)
        except: return pd.DataFrame()

    d['fed'] = get_fred('WALCL')
    d['tga'] = get_fred('WTREGEN')
    d['rrp'] = get_fred('RRPONTSYD')

    # [C] Nasdaq (Yahoo)
    try:
        ns = yf.download("^IXIC", period="3y", progress=False)
        s = ns.xs('Close', axis=1, level=0)["^IXIC"] if isinstance(ns.columns, pd.MultiIndex) else ns['Close']
        d['nasdaq'] = s.tz_localize(None)
    except: d['nasdaq'] = pd.Series(dtype=float)

    # [D] 난이도 (JSON)
    try:
        with open('difficulty (1).json', 'r') as f:
            js = json.load(f)['difficulty']
        df_js = pd.DataFrame(js)
        df_js['Date'] = pd.to_datetime(df_js['x'], unit='ms').dt.tz_localize(None)
        d['diff'] = df_js.set_index('Date').sort_index()['y']
    except: d['diff'] = pd.Series(dtype=float)

    return d

with st.spinner('전 세계 금융 요새로부터 데이터를 동기화 중입니다...'):
    raw = fetch_all_data_final()

# 3. 전략적 지표 계산 (선생님의 로직 완벽 반영)
if not raw['btc'].empty:
    # [유동성 계산]
    df_liq = raw['fed'].resample('W-WED').last()
    df_liq.columns = ['Fed']
    if not raw['tga'].empty: df_liq = df_liq.join(raw['tga'].resample('W-WED').mean().rename(columns={raw['tga'].columns[0]:'TGA'}))
    if not raw['rrp'].empty: df_liq = df_liq.join(raw['rrp'].resample('W-WED').mean().rename(columns={raw['rrp'].columns[0]:'RRP'}))
    df_liq = df_liq.fillna(method='ffill')
    df_liq['Net_Tril'] = (df_liq['Fed'] - df_liq.get('TGA', 0) - (df_liq.get('RRP', 0) * 1000)) / 1_000_000
    df_liq['YoY'] = df_liq['Net_Tril'].pct_change(52) * 100

    # [채굴 원가 계산]
    df_c = pd.DataFrame(index=raw['btc'].index)
    if not raw['diff'].empty:
        df_c['diff'] = raw['diff'].resample('D').interpolate()
        df_c['reward'] = [3.125 if x >= pd.Timestamp('2024-04-20') else 6.25 for x in df_c.index]
        df_c['cost_raw'] = df_c['diff'] / df_c['reward']
        sub = pd.concat([raw['btc'], df_c['cost_raw']], axis=1).dropna()
        target = sub[(sub.index >= '2022-11-01') & (sub.index <= '2023-01-31')]
        k = (target.iloc[:,0] / target.iloc[:,1]).min() if not target.empty else 0.00000008
        df_c['floor'] = df_c['cost_raw'] * k

    # [90일 시프트 데이터 준비]
    def shift_90(s):
        if s is None or s.empty: return pd.Series(dtype=float)
        new_s = s.copy()
        new_s.index = new_s.index - pd.Timedelta(days=90)
        return new_s

    btc_s = shift_90(raw['btc'])
    floor_s = shift_90(df_c['floor']) if 'floor' in df_c else pd.Series(dtype=float)
    nasdaq_s = shift_90(raw['nasdaq'])
    doge_s = shift_90(raw['doge'])

    # 4. 고해상도 멀티축 차트 (Plotly - 모바일 줌 지원)
    st.subheader("📊 The Grand Master Integrated Chart")
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    start_viz = '2023-01-01'

    # [1] 유동성 YoY (왼쪽 축 - 노란색 영역)
    liq_v = df_liq[df_liq.index >= start_viz]
    fig.add_trace(go.Scatter(x=liq_v.index, y=liq_v['YoY'], name="Liquidity YoY", line=dict(color='rgba(255, 215, 0, 0.8)', width=2), fill='tozeroy', fillcolor='rgba(255, 215, 0, 0.1)'), secondary_y=False)

    # [2] BTC 가격 (오른쪽 축 1 - 흰색)
    btc_v = btc_s[btc_s.index >= start_viz]
    fig.add_trace(go.Scatter(x=btc_v.index, y=btc_v, name="BTC (-90d)", line=dict(color='white', width=2.5)), secondary_y=True)

    # [3] 채굴 원가 (오른쪽 축 1 - 빨간 점선)
    if not floor_s.empty:
        fl_v = floor_s[floor_s.index >= start_viz]
        fig.add_trace(go.Scatter(x=fl_v.index, y=fl_v, name="Cost Floor (-90d)", line=dict(color='red', width=1.5, dash='dot')), secondary_y=True)

    # [4] Nasdaq (오른쪽 축 2 - 분홍색)
    if not nasdaq_s.empty:
        nd_v = nasdaq_s[nasdaq_s.index >= start_viz]
        fig.add_trace(go.Scatter(x=nd_v.index, y=nd_s, name="Nasdaq (-90d)", line=dict(color='#D62780', width=1.5), opacity=0.7), secondary_y=True)

    # [5] Doge (오른쪽 축 2 - 주황색)
    if not doge_s.empty:
        dg_v = doge_s[doge_s.index >= start_viz]
        fig.add_trace(go.Scatter(x=dg_v.index, y=dg_v, name="DOGE (-90d)", line=dict(color='orange', width=1.2), opacity=0.6), secondary_y=True)

    # 스타일 및 레이아웃 설정
    fig.update_layout(template="plotly_dark", height=700, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.1), hovermode="x unified")
    fig.update_yaxes(title_text="Liquidity Growth %", secondary_y=False, range=[-20, 40])
    fig.update_yaxes(title_text="Asset Price (Log)", type="log", secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

    # 5. 실시간 진단 사이드바
    st.sidebar.header("📋 Master Diagnosis")
    cur_p = raw['btc'].iloc[-1]
    st.sidebar.metric("BTC 현재가", f"${cur_p:,.0f}")
    if 'floor' in df_c:
        cur_f = df_c['floor'].iloc[-1]
        gap = (cur_p/cur_f - 1)*100
        st.sidebar.metric("채굴 원가", f"${cur_f:,.0f}", f"{gap:.2f}%")
        if gap < 0: st.sidebar.error("🔥 진성 항복 구간 (강력 매수)")
        else: st.sidebar.success("✅ 정상 궤도 진입")

else:
    st.error("데이터 동기화 실패. 다시 시도해 주세요.")
