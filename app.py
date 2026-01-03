import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

# 시스템 설정
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Grand Master Terminal", layout="wide")

st.title("🏛️ Grand Master Investment Terminal")

# ---------------------------------------------------------
# 1. 데이터 로드 (Timezone 강제 제거 적용)
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def load_data_plotly():
    data = {}
    
    # [A] 자산 가격 (yfinance)
    def get_ticker(t):
        try:
            df = yf.download(t, start="2019-01-01", progress=False)
            if df.empty: return pd.Series(dtype='float64')
            if isinstance(df.columns, pd.MultiIndex): 
                s = df['Close'][t]
            else: 
                s = df['Close']
            
            # [핵심 해결책] 시간대(Timezone) 정보를 강제로 제거
            s.index = s.index.tz_localize(None)
            return s
        except: return pd.Series(dtype='float64')

    data['btc'] = get_ticker("BTC-USD")
    data['nasdaq'] = get_ticker("^IXIC")
    data['doge'] = get_ticker("DOGE-USD")

    # [B] 채굴 난이도 (JSON)
    try:
        with open('difficulty (1).json', 'r') as f:
            diff_raw = json.load(f)['difficulty']
        df_d = pd.DataFrame(diff_raw)
        df_d['Date'] = pd.to_datetime(df_d['x'], unit='ms')
        
        # [핵심 해결책] 여기도 시간대 제거 (혹시 몰라서)
        df_d['Date'] = df_d['Date'].dt.tz_localize(None)
        
        data['diff'] = df_d.set_index('Date').sort_index()['y']
    except:
        data['diff'] = pd.Series(dtype='float64')

    # [C] 유동성 (FRED)
    def get_fred(id):
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"
            df = pd.read_csv(url, index_col=0, parse_dates=True)
            df.index = df.index.tz_localize(None) # 시간대 제거
            return df
        except: return pd.DataFrame()

    data['fed'] = get_fred('WALCL')
    
    return data

d = load_data_plotly()

# ---------------------------------------------------------
# 2. 지표 계산
# ---------------------------------------------------------
# 원가 계산
df_calc = pd.DataFrame(index=d['btc'].index)

if not d['diff'].empty:
    # 인덱스 정렬 확인
    daily_diff = d['diff'].resample('D').interpolate(method='linear')
    # 같은 날짜끼리 합치기
    df_calc = df_calc.join(daily_diff.rename('diff'), how='left').ffill()
    
    def get_reward(dt):
        if dt < pd.Timestamp('2020-05-11'): return 12.5
        elif dt < pd.Timestamp('2024-04-20'): return 6.25
        else: return 3.125
    
    df_calc['reward'] = [get_reward(x) for x in df_calc.index]
    df_calc['cost'] = df_calc['diff'] / df_calc['reward']
    
    # Calibration
    try:
        # 데이터가 있는 구간만 잘라서 계산
        aligned = pd.concat([d['btc'], df_calc['cost']], axis=1).dropna()
        subset = aligned[(aligned.index >= '2022-11-01') & (aligned.index <= '2023-01-31')]
        if not subset.empty:
            k = (subset.iloc[:,0] / subset.iloc[:,1]).min()
        else:
            k = 0.00000008
    except: k = 0.00000008
        
    df_calc['floor'] = df_calc['cost'] * k
else:
    df_calc['floor'] = np.nan

# ---------------------------------------------------------
# 3. Interactive 차트 그리기 (Plotly)
# ---------------------------------------------------------
st.subheader("📊 Interactive Grand Master Chart")

if not d['btc'].empty:
    # 2023년부터 보기
    start_date = '2023-01-01'
    # 데이터 필터링
    plot_btc = d['btc'][d['btc'].index >= start_date]
    plot_floor = df_calc['floor'][df_calc['floor'].index >= start_date] if 'floor' in df_calc.columns else pd.Series()
    
    # [Plotly] 차트 캔버스 생성 (보조축 사용)
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 1. BTC 가격 (왼쪽 축, 로그, 캔들/선)
    fig.add_trace(
        go.Scatter(x=plot_btc.index, y=plot_btc, name="BTC Price", line=dict(color='white', width=2)),
        secondary_y=False
    )

    # 2. Mining Floor (왼쪽 축, 빨간 점선)
    fig.add_trace(
        go.Scatter(x=plot_floor.index, y=plot_floor, name="Mining Cost", line=dict(color='red', dash='dot')),
        secondary_y=False
    )

    # 3. Nasdaq (-90일, 오른쪽 축)
    if not d['nasdaq'].empty:
        nd_s = d['nasdaq'].shift(90)
        nd_s = nd_s[nd_s.index >= start_date]
        fig.add_trace(
            go.Scatter(x=nd_s.index, y=nd_s, name="Nasdaq (-90d)", line=dict(color='#D62780', width=1.5), opacity=0.7),
            secondary_y=True
        )

    # 4. DOGE (-90일, 오른쪽 축)
    if not d['doge'].empty:
        dg_s = d['doge'].shift(90)
        dg_s = dg_s[dg_s.index >= start_date]
        fig.add_trace(
            go.Scatter(x=dg_s.index, y=dg_s, name="DOGE (-90d)", line=dict(color='orange', width=1.5), opacity=0.7),
            secondary_y=True
        )

    # 스타일 설정 (다크모드, 로그스케일)
    fig.update_layout(
        template="plotly_dark",
        height=600,
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=10, r=10, t=30, b=10)
    )
    # 왼쪽 축 로그 설정
    fig.update_yaxes(type="log", title_text="BTC Price (Log)", secondary_y=False)
    fig.update_yaxes(title_text="Nasdaq / Doge", secondary_y=True)

    # 화면 출력
    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------
    # 4. 사이드바 진단 업데이트
    # ---------------------------------------------------------
    last_price = d['btc'].iloc[-1]
    if not np.isnan(plot_floor.iloc[-1]):
        last_floor = plot_floor.iloc[-1]
        gap = (last_price / last_floor - 1) * 100
        
        st.sidebar.header("📋 시장 진단")
        st.sidebar.metric("BTC Price", f"${last_price:,.0f}")
        st.sidebar.metric("Mining Cost", f"${last_floor:,.0f}", f"{gap:.2f}%")
        
        if gap < 0:
            st.sidebar.error(f"🔥 진성 항복 구간 (Gap: {gap:.1f}%)")
        else:
            st.sidebar.success("✅ 정상 가동 구간")
else:
    st.error("데이터 로드 실패. 잠시 후 다시 시도해주세요.")
