import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import yfinance as yf
import json
import numpy as np
import io
import warnings

# 1. 환경 설정 및 인터페이스 최적화
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Grand Master Terminal", layout="wide")

st.title("🏛️ Grand Master: Strategic Multi-Axis Terminal")
st.caption("Ver 8.4 | 개별 자산별 독립 척도 시스템 완비")

# 2. 데이터 수집 엔진 (안정성 강화)
@st.cache_data(ttl=3600)
def fetch_master_data():
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

raw = fetch_master_data()

# 3. 로직 처리 (유동성 및 시프트 계산)
if not raw['btc'].empty:
    # G3 유동성 모사 (Fed Net Liquidity)
    df_liq = raw['fed'].resample('W-WED').last()
    df_liq.columns = ['Fed']
    if not raw['tga'].empty: df_liq = df_liq.join(raw['tga'].resample('W-WED').mean().rename(columns={raw['tga'].columns[0]:'TGA'}))
    if not raw['rrp'].empty: df_liq = df_liq.join(raw['rrp'].resample('W-WED').mean().rename(columns={raw['rrp'].columns[0]:'RRP'}))
    df_liq = df_liq.fillna(method='ffill')
    df_liq['Net_Tril'] = (df_liq['Fed'] - df_liq.get('TGA', 0) - (df_liq.get('RRP', 0) * 1000)) / 1_000_000
    df_liq['YoY'] = df_liq['Net_Tril'].pct_change(52) * 100

    # 채굴 원가
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

    # 4. [핵심] 다중 축 차트 생성 (골조 확보)
    st.subheader("📊 Grand Master Multi-Asset Convergence")
    
    fig = go.Figure()
    start_viz = '2023-01-01'

    # [Trace 1] 유동성 YoY (Y1 - Left)
    liq_v = df_liq[df_liq.index >= start_viz]
    fig.add_trace(go.Scatter(x=liq_v.index, y=liq_v['YoY'], name="Liquidity YoY", 
                             line=dict(color='gold', width=2), fill='tozeroy', 
                             fillcolor='rgba(255, 215, 0, 0.1)', yaxis="y1"))

    # [Trace 2] BTC (Y2 - Right 1)
    btc_v = btc_s[btc_s.index >= start_viz]
    fig.add_trace(go.Scatter(x=btc_v.index, y=btc_v, name="BTC (-90d)", 
                             line=dict(color='white', width=2.5), yaxis="y2"))
    
    # [Trace 3] Cost Floor (Y2 - Right 1 공유)
    if not floor_s.empty:
        fl_v = floor_s[floor_s.index >= start_viz]
        fig.add_trace(go.Scatter(x=fl_v.index, y=fl_v, name="Cost Floor", 
                                 line=dict(color='red', width=1.5, dash='dot'), yaxis="y2"))

    # [Trace 4] Nasdaq (Y3 - Right 2)
    if not nasdaq_s.empty:
        nd_v = nasdaq_s[nasdaq_s.index >= start_viz]
        fig.add_trace(go.Scatter(x=nd_v.index, y=nd_v, name="Nasdaq (-90d)", 
                                 line=dict(color='#D62780', width=1.5), yaxis="y3"))

    # [Trace 5] Doge (Y4 - Right 3)
    if not doge_s.empty:
        dg_v = doge_s[doge_s.index >= start_viz]
        fig.add_trace(go.Scatter(x=dg_v.index, y=dg_v, name="DOGE (-90d)", 
                                 line=dict(color='orange', width=1.2), yaxis="y4"))

    # 레이아웃 최종 정렬 (축 정의 포함)
    fig.update_layout(
        template="plotly_dark",
        height=750,
        xaxis=dict(domain=[0.05, 0.8], showgrid=False), # 왼쪽 여백 및 우측 축 공간
        yaxis=dict(title="Liquidity YoY %", titlefont=dict(color="gold"), tickfont=dict(color="gold")),
        yaxis2=dict(title="BTC (Log)", type="log", overlaying="y", side="right", 
                    titlefont=dict(color="white"), tickfont=dict(color="white"), anchor="x"),
        yaxis3=dict(title="Nasdaq", overlaying="y", side="right", anchor="free", 
                    position=0.87, titlefont=dict(color="#D62780"), tickfont=dict(color="#D62780")),
        yaxis4=dict(title="Doge (Log)", type="log", overlaying="y", side="right", anchor="free", 
                    position=0.95, titlefont=dict(color="orange"), tickfont=dict(color="orange")),
        legend=dict(orientation="h", y=1.1, x=0),
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("시장의 맥박을 동기화 중입니다. 잠시만 기다려 주십시오.")
