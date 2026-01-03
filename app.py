import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import yfinance as yf
import json
import numpy as np
import io
import warnings

# 1. 환경 설정
warnings.filterwarnings("ignore")
st.set_page_config(page_title="GM Terminal Final", layout="wide")

st.title("🏛️ Grand Master: Multi-Axis Final")
st.caption("Ver 8.5 | Layout 구조 재설계 및 안정화 패치 완료")

# 2. 데이터 수집 (캐시 적용)
@st.cache_data(ttl=3600)
def fetch_data_ver85():
    d = {}
    
    # [A] Upbit
    def get_upbit(symbol):
        try:
            r = requests.get(f"https://api.upbit.com/v1/candles/days?market={symbol}&count=1000", timeout=5).json()
            df = pd.DataFrame(r)
            df['Date'] = pd.to_datetime(df['candle_date_time_utc']).dt.tz_localize(None)
            return df.set_index('Date').sort_index()['trade_price'].astype(float)
        except: return pd.Series(dtype=float)

    d['btc'] = get_upbit("USDT-BTC")
    d['doge'] = get_upbit("USDT-DOGE")

    # [B] FRED
    def get_fred(id):
        try:
            r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}", timeout=5)
            df = pd.read_csv(io.StringIO(r.text), index_col=0, parse_dates=True)
            return df.resample('D').interpolate().tz_localize(None)
        except: return pd.DataFrame()

    d['fed'] = get_fred('WALCL')
    d['tga'] = get_fred('WTREGEN')
    d['rrp'] = get_fred('RRPONTSYD')

    # [C] Nasdaq
    try:
        ns = yf.download("^IXIC", period="3y", progress=False)
        s = ns.xs('Close', axis=1, level=0)["^IXIC"] if isinstance(ns.columns, pd.MultiIndex) else ns['Close']
        d['nasdaq'] = s.tz_localize(None)
    except: d['nasdaq'] = pd.Series(dtype=float)

    # [D] Difficulty
    try:
        with open('difficulty (1).json', 'r') as f:
            js = json.load(f)['difficulty']
        df_js = pd.DataFrame(js)
        df_js['Date'] = pd.to_datetime(df_js['x'], unit='ms').dt.tz_localize(None)
        d['diff'] = df_js.set_index('Date').sort_index()['y']
    except: d['diff'] = pd.Series(dtype=float)
    
    return d

raw = fetch_data_ver85()

# 3. 데이터 가공
if not raw['btc'].empty:
    # Liquidity
    df_liq = raw['fed'].resample('W-WED').last()
    df_liq.columns = ['Fed']
    if not raw['tga'].empty: df_liq = df_liq.join(raw['tga'].resample('W-WED').mean().rename(columns={raw['tga'].columns[0]:'TGA'}))
    if not raw['rrp'].empty: df_liq = df_liq.join(raw['rrp'].resample('W-WED').mean().rename(columns={raw['rrp'].columns[0]:'RRP'}))
    df_liq = df_liq.fillna(method='ffill')
    df_liq['Net_Tril'] = (df_liq['Fed'] - df_liq.get('TGA', 0) - (df_liq.get('RRP', 0) * 1000)) / 1_000_000
    df_liq['YoY'] = df_liq['Net_Tril'].pct_change(52) * 100

    # Mining Cost
    df_c = pd.DataFrame(index=raw['btc'].index)
    if not raw['diff'].empty:
        df_c['diff'] = raw['diff'].resample('D').interpolate()
        df_c['reward'] = [3.125 if x >= pd.Timestamp('2024-04-20') else 6.25 for x in df_c.index]
        df_c['cost_raw'] = df_c['diff'] / df_c['reward']
        sub = pd.concat([raw['btc'], df_c['cost_raw']], axis=1).dropna()
        target = sub[(sub.index >= '2022-11-01') & (sub.index <= '2023-01-31')]
        k = (target.iloc[:,0] / target.iloc[:,1]).min() if not target.empty else 0.00000008
        df_c['floor'] = df_c['cost_raw'] * k

    # Shift -90d
    def shift_90(s):
        if s is None or s.empty: return pd.Series(dtype=float)
        new_s = s.copy(); new_s.index = new_s.index - pd.Timedelta(days=90)
        return new_s

    btc_s = shift_90(raw['btc'])
    floor_s = shift_90(df_c['floor']) if 'floor' in df_c else pd.Series(dtype=float)
    nasdaq_s = shift_90(raw['nasdaq'])
    doge_s = shift_90(raw['doge'])

    # 4. [수정된 핵심] 안전한 Layout 생성 방식
    st.subheader("📊 Grand Master Integrated Strategy Chart")
    
    start_viz = '2023-01-01'
    
    # Trace 데이터 준비
    liq_v = df_liq[df_liq.index >= start_viz]
    btc_v = btc_s[btc_s.index >= start_viz]
    fl_v = floor_s[floor_s.index >= start_viz] if not floor_s.empty else pd.Series(dtype=float)
    nd_v = nasdaq_s[nasdaq_s.index >= start_viz] if not nasdaq_s.empty else pd.Series(dtype=float)
    dg_v = doge_s[doge_s.index >= start_viz] if not doge_s.empty else pd.Series(dtype=float)

    # Figure 생성 (Layout을 명시적으로 주입)
    fig = go.Figure(
        layout=go.Layout(
            template="plotly_dark",
            height=700,
            # [X축] 우측에 3개의 축이 들어갈 공간(15%) 확보
            xaxis=dict(domain=[0.0, 0.85], showgrid=False),
            
            # [Y축 1: 유동성] 왼쪽
            yaxis=dict(
                title="Liquidity YoY %",
                titlefont=dict(color="#FFD700"),
                tickfont=dict(color="#FFD700"),
                range=[-20, 40]
            ),
            
            # [Y축 2: BTC] 오른쪽 1번
            yaxis2=dict(
                title="BTC (Log)",
                titlefont=dict(color="white"),
                tickfont=dict(color="white"),
                anchor="x",
                overlaying="y",
                side="right",
                type="log"
            ),
            
            # [Y축 3: Nasdaq] 오른쪽 2번 (위치 지정)
            yaxis3=dict(
                title="Nasdaq",
                titlefont=dict(color="#D62780"),
                tickfont=dict(color="#D62780"),
                anchor="free",
                overlaying="y",
                side="right",
                position=0.92  # BTC 축보다 약간 오른쪽
            ),
            
            # [Y축 4: DOGE] 오른쪽 3번 (가장 오른쪽)
            yaxis4=dict(
                title="DOGE (Log)",
                titlefont=dict(color="orange"),
                tickfont=dict(color="orange"),
                anchor="free",
                overlaying="y",
                side="right",
                position=1.0,  # 가장 끝
                type="log"
            ),
            
            legend=dict(orientation="h", y=1.1, x=0),
            hovermode="x unified",
            margin=dict(r=100) # 우측 여백 추가 확보
        )
    )

    # Trace 추가 (정의된 축에 매핑)
    # 1. Liquidity -> y
    fig.add_trace(go.Scatter(x=liq_v.index, y=liq_v['YoY'], name="Liquidity YoY",
                             line=dict(color='#FFD700', width=2), fill='tozeroy', 
                             fillcolor='rgba(255, 215, 0, 0.1)', yaxis='y'))

    # 2. BTC -> y2
    fig.add_trace(go.Scatter(x=btc_v.index, y=btc_v, name="BTC (-90d)",
                             line=dict(color='white', width=2.5), yaxis='y2'))

    # 3. Cost Floor -> y2 (BTC와 공유)
    if not fl_v.empty:
        fig.add_trace(go.Scatter(x=fl_v.index, y=fl_v, name="Cost Floor",
                                 line=dict(color='red', width=1.5, dash='dot'), yaxis='y2'))

    # 4. Nasdaq -> y3
    if not nd_v.empty:
        fig.add_trace(go.Scatter(x=nd_v.index, y=nd_v, name="Nasdaq (-90d)",
                                 line=dict(color='#D62780', width=1.5), yaxis='y3'))

    # 5. Doge -> y4
    if not dg_v.empty:
        fig.add_trace(go.Scatter(x=dg_v.index, y=dg_v, name="DOGE (-90d)",
                                 line=dict(color='orange', width=1.5), yaxis='y4'))

    st.plotly_chart(fig, use_container_width=True)

    # 하단 정보
    st.success("데이터 로드 및 4중 축 차트 생성 완료")

else:
    st.error("데이터 동기화 대기 중...")
