import streamlit as st
import pandas as pd
import requests
import json
import numpy as np
import warnings

# 1. 기본 설정
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Grand Master", layout="mobile") # 모바일 레이아웃

st.title("🏛️ Grand Master Terminal")
st.caption("데이터가 눈에 보이지 않으면 의미가 없습니다. 순정 모드로 가동합니다.")

# 2. 데이터 가져오기 (바이낸스 + JSON)
@st.cache_data(ttl=300)
def load_data():
    data = {}
    
    # [A] BTC 가격 (바이낸스 - 무조건 됨)
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": "BTCUSDT", "interval": "1d", "limit": 1000} # 약 3년치
        r = requests.get(url, params=params).json()
        df = pd.DataFrame(r, columns=["t", "o", "h", "l", "c", "v", "T", "q", "n", "V", "Q", "B"])
        df["Date"] = pd.to_datetime(df["t"], unit='ms')
        df["BTC"] = df["c"].astype(float)
        data['btc'] = df.set_index("Date")[["BTC"]]
    except Exception as e:
        st.error(f"바이낸스 연결 실패: {e}")
        return None

    # [B] 채굴 난이도 (JSON 파일)
    try:
        with open('difficulty (1).json', 'r') as f:
            d_json = json.load(f)['difficulty']
        df_d = pd.DataFrame(d_json)
        df_d['Date'] = pd.to_datetime(df_d['x'], unit='ms')
        data['diff'] = df_d.set_index('Date')['y']
    except:
        # 파일 없으면 그냥 빈 것으로 처리 (차트는 나오게)
        data['diff'] = pd.Series(dtype=float)

    return data

d = load_data()

# 3. 데이터가 비었으면 중단
if d is None or d['btc'].empty:
    st.error("🚨 데이터 로드 실패. 인터넷 연결을 확인하세요.")
    st.stop()

# 4. 원가 계산 및 데이터 합치기
df_chart = d['btc'].copy()
df_chart['Mining Cost'] = np.nan # 기본값

if not d['diff'].empty:
    # 날짜 맞추기
    daily_diff = d['diff'].resample('D').interpolate(method='linear')
    # 인덱스 합치기
    temp = pd.merge(d['btc'], daily_diff.rename('diff'), left_index=True, right_index=True, how='left').ffill()
    
    # 반감기 보정
    def get_reward(dt):
        if dt < pd.Timestamp('2024-04-20'): return 6.25
        else: return 3.125
    
    temp['reward'] = [get_reward(x) for x in temp.index]
    temp['cost'] = temp['diff'] / temp['reward']
    
    # Calibration (2022년 11월 바닥 기준)
    try:
        subset = temp[(temp.index >= '2022-11-01') & (temp.index <= '2023-01-31')]
        k = (subset['BTC'] / subset['cost']).min() if not subset.empty else 0.00000008
    except: k = 0.00000008
    
    df_chart['Mining Cost'] = temp['cost'] * k

# 5. [핵심] 차트 그리기 (Streamlit 내장 차트 사용)
# 복잡한 설정 다 빼고 그냥 그립니다.
st.subheader("1. 가격 vs 원가 (Price & Cost)")
st.line_chart(df_chart[['BTC', 'Mining Cost']], color=["#FFFFFF", "#FF0000"]) 
# 흰색: BTC, 빨간색: 원가

# 6. 데이터 표로 확인 (이게 보이면 데이터는 있는 것임)
with st.expander("🔍 데이터 원본 확인하기 (클릭)"):
    st.dataframe(df_chart.tail(10))

# 7. 사이드바 진단
last_btc = df_chart['BTC'].iloc[-1]
last_cost = df_chart['Mining Cost'].iloc[-1]

st.sidebar.header("시장 진단")
st.sidebar.metric("비트코인", f"${last_btc:,.0f}")
if not np.isnan(last_cost):
    gap = (last_btc / last_cost - 1) * 100
    st.sidebar.metric("채굴 원가", f"${last_cost:,.0f}", f"{gap:.2f}%")
