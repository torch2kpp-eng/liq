import streamlit as st
import pandas as pd
import yfinance as yf
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

# 설정
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Grand Master Terminal", layout="wide")
st.title("🏛️ Grand Master Terminal (Diagnostic Mode)")

# 1. 데이터 로드 (가장 단순하고 강력하게)
@st.cache_data(ttl=600)
def get_data_debug():
    data = {}
    
    # [A] 자산 데이터 (yfinance)
    # yfinance가 최근 업데이트로 리턴 형식이 복잡해져서, 가장 단순하게 풉니다.
    def download_safe(ticker):
        try:
            df = yf.download(ticker, start="2020-01-01", progress=False)
            # MultiIndex 컬럼 처리 (Price, Ticker) -> Price만 남김
            if isinstance(df.columns, pd.MultiIndex):
                s = df.xs('Close', axis=1, level=0)[ticker]
            else:
                s = df['Close']
            
            # [핵심] 날짜 형식을 무조건 'UTC제거' 상태로 통일
            s.index = pd.to_datetime(s.index).tz_localize(None)
            return s
        except Exception as e:
            st.error(f"{ticker} 로드 실패: {e}")
            return pd.Series(dtype=float)

    data['btc'] = download_safe("BTC-USD")
    data['nasdaq'] = download_safe("^IXIC")
    data['doge'] = download_safe("DOGE-USD")

    # [B] 난이도 데이터 (JSON)
    try:
        with open('difficulty (1).json', 'r') as f:
            diff_raw = json.load(f)['difficulty']
        df_d = pd.DataFrame(diff_raw)
        # 밀리초(ms) -> 날짜 변환
        df_d['Date'] = pd.to_datetime(df_d['x'], unit='ms')
        # [핵심] 여기도 날짜 통일
        df_d['Date'] = df_d['Date'].dt.tz_localize(None)
        data['diff'] = df_d.set_index('Date').sort_index()['y']
    except Exception as e:
        st.error(f"JSON 로드 실패: {e}")
        data['diff'] = pd.Series(dtype=float)

    return data

d = get_data_debug()

# 2. 데이터 눈으로 확인하기 (디버깅용)
st.subheader("1. 데이터 생존 확인")
col1, col2 = st.columns(2)
with col1:
    st.write("비트코인 데이터 (최근 5일):", d['btc'].tail())
with col2:
    st.write("난이도 데이터 (최근 5일):", d['diff'].tail())

if d['btc'].empty:
    st.error("🚨 비트코인 데이터가 텅 비었습니다. 야후 파이낸스 연결 문제일 수 있습니다.")
    st.stop()

# 3. 계산 및 차트 그리기
# 원가 계산
df_main = pd.DataFrame(index=d['btc'].index)
if not d['diff'].empty:
    # 일별 난이도 채우기
    daily_diff = d['diff'].resample('D').interpolate(method='linear')
    # 인덱스 교집합으로 합치기
    df_main = df_main.join(daily_diff.rename('diff'), how='left').ffill()
    
    # 반감기 로직
    def get_reward(dt):
        if dt < pd.Timestamp('2020-05-11'): return 12.5
        elif dt < pd.Timestamp('2024-04-20'): return 6.25
        else: return 3.125
    
    df_main['reward'] = [get_reward(x) for x in df_main.index]
    df_main['cost'] = df_main['diff'] / df_main['reward']
    
    # Calibration
    try:
        common = pd.concat([d['btc'], df_main['cost']], axis=1).dropna()
        subset = common[(common.index >= '2022-11-01') & (common.index <= '2023-01-31')]
        k = (subset.iloc[:,0] / subset.iloc[:,1]).min() if not subset.empty else 0.00000008
    except: k = 0.00000008
        
    df_main['floor'] = df_main['cost'] * k
else:
    df_main['floor'] = 0

# 차트 그리기
st.subheader("2. 차트 시각화")

# [핵심] 필터링 없이 일단 전체 기간 그리기 (데이터가 짤리는지 확인 위해)
fig = make_subplots(specs=[[{"secondary_y": True}]])

# BTC
fig.add_trace(go.Scatter(x=d['btc'].index, y=d['btc'], name="BTC", line=dict(color='white')), secondary_y=False)

# Floor
if 'floor' in df_main.columns:
    fig.add_trace(go.Scatter(x=df_main.index, y=df_main['floor'], name="Mining Cost", line=dict(color='red', dash='dot')), secondary_y=False)

# Nasdaq (-90d)
if not d['nasdaq'].empty:
    nd_s = d['nasdaq'].shift(90)
    fig.add_trace(go.Scatter(x=nd_s.index, y=nd_s, name="Nasdaq (-90d)", line=dict(color='#D62780'), opacity=0.7), secondary_y=True)

# DOGE (-90d)
if not d['doge'].empty:
    dg_s = d['doge'].shift(90)
    fig.add_trace(go.Scatter(x=dg_s.index, y=dg_s, name="DOGE (-90d)", line=dict(color='orange'), opacity=0.7), secondary_y=True)

fig.update_layout(template="plotly_dark", height=600, title="Grand Master Chart")
fig.update_yaxes(type="log", title_text="BTC (Log)", secondary_y=False)

st.plotly_chart(fig, use_container_width=True)

# 사이드바
if not d['btc'].empty and 'floor' in df_main.columns:
    last_p = d['btc'].iloc[-1]
    last_f = df_main['floor'].iloc[-1]
    if last_f > 0:
        gap = (last_p / last_f - 1) * 100
        st.sidebar.metric("BTC", f"${last_p:,.0f}")
        st.sidebar.metric("Cost", f"${last_f:,.0f}", f"{gap:.2f}%")
