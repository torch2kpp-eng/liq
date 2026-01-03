import streamlit as st
import pandas as pd
import requests
import json
import numpy as np
import warnings

# 1. 설정
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Grand Master", layout="mobile")

st.title("🏛️ Grand Master Terminal")

# 2. 데이터 가져오기 (바이낸스 + JSON)
@st.cache_data(ttl=300)
def load_data_final():
    data = {}
    
    # [A] BTC 가격 (바이낸스)
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": "BTCUSDT", "interval": "1d", "limit": 1000} 
        r = requests.get(url, params=params).json()
        df = pd.DataFrame(r, columns=["t", "o", "h", "l", "c", "v", "T", "q", "n", "V", "Q", "B"])
        df["Date"] = pd.to_datetime(df["t"], unit='ms')
        df["BTC"] = df["c"].astype(float)
        # 인덱스 대신 컬럼으로 유지 (Streamlit 호환성 극대화)
        data['btc'] = df[["Date", "BTC"]] 
    except Exception as e:
        st.error(f"바이낸스 에러: {e}")
        return None

    # [B] 채굴 난이도 (JSON)
    try:
        with open('difficulty (1).json', 'r') as f:
            d_json = json.load(f)['difficulty']
        df_d = pd.DataFrame(d_json)
        df_d['Date'] = pd.to_datetime(df_d['x'], unit='ms')
        data['diff'] = df_d[['Date', 'y']]
    except:
        data['diff'] = pd.DataFrame(columns=['Date', 'y'])

    return data

d = load_data_final()

if d is None or d['btc'].empty:
    st.error("데이터 로드 실패")
    st.stop()

# 3. 데이터 합치기 및 원가 계산
df_btc = d['btc'].set_index('Date')
df_diff = d['diff'].set_index('Date')['y']

# 날짜 합치기
df_main = df_btc.copy()
daily_diff = df_diff.resample('D').interpolate(method='linear')
df_main = df_main.join(daily_diff.rename('Difficulty'), how='left').ffill()

# 원가 계산
def get_reward(dt):
    if dt < pd.Timestamp('2024-04-20'): return 6.25
    else: return 3.125

df_main['Reward'] = [get_reward(x) for x in df_main.index]
df_main['Raw Cost'] = df_main['Difficulty'] / df_main['Reward']

# Calibration
try:
    subset = df_main[(df_main.index >= '2022-11-01') & (df_main.index <= '2023-01-31')]
    k = (subset['BTC'] / subset['Raw Cost']).min() if not subset.empty else 0.00000008
except: k = 0.00000008

df_main['Mining Cost'] = df_main['Raw Cost'] * k

# [핵심] 차트용 데이터 정리 (NaN 제거 및 인덱스 리셋)
# NaN이 있으면 차트가 끊기므로 채워줍니다.
chart_data = df_main[['BTC', 'Mining Cost']].fillna(method='ffill').dropna()
chart_data = chart_data.reset_index() # 'Date'를 컬럼으로 뺍니다.

# 4. 차트 그리기 (색상 지정 삭제 -> 자동 배색)
st.subheader("📊 Price vs Cost")

# x축을 Date로 명시하고, y축에 그릴 것들을 리스트로 줍니다.
st.line_chart(
    chart_data,
    x="Date",
    y=["BTC", "Mining Cost"]
)

# 5. 데이터 표 확인 (이게 보이면 데이터는 확실함)
with st.expander("데이터 원본 보기 (클릭)"):
    st.dataframe(chart_data.tail(10))

# 6. 사이드바
last_btc = chart_data['BTC'].iloc[-1]
last_cost = chart_data['Mining Cost'].iloc[-1]
gap = (last_btc / last_cost - 1) * 100

st.sidebar.metric("Bitcoin", f"${last_btc:,.0f}")
st.sidebar.metric("Mining Cost", f"${last_cost:,.0f}", f"{gap:.2f}%")
