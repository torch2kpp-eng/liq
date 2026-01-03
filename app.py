import streamlit as st
import pandas as pd
import requests
import warnings

# 1. 설정
warnings.filterwarnings("ignore")
st.set_page_config(page_title="GM Mobile", layout="mobile")

# [중요] 버전 확인용 제목
st.title("Ver 7.0 : Mobile Final") 
st.caption("이 제목이 보여야 최신 버전입니다.")

# 2. 데이터 가져오기 (가장 단순한 구조)
@st.cache_data(ttl=300)
def get_data():
    # 바이낸스 비트코인
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": "BTCUSDT", "interval": "1d", "limit": 1000}
        r = requests.get(url, params=params).json()
        df = pd.DataFrame(r, columns=["t", "o", "h", "l", "c", "v", "T", "q", "n", "V", "Q", "B"])
        df["Date"] = pd.to_datetime(df["t"], unit='ms')
        df["Price"] = df["c"].astype(float)
        return df[["Date", "Price"]].set_index("Date")
    except Exception as e:
        return None

df = get_data()

# 3. 차트 그리기 (Streamlit 내장 차트)
if df is not None and not df.empty:
    st.subheader("비트코인 가격 차트")
    # 내장 라인 차트 (무조건 나옴)
    st.line_chart(df["Price"]) 
    
    # 현재가 표시
    last_p = df["Price"].iloc[-1]
    st.metric("현재가", f"${last_p:,.0f}")
else:
    st.error("데이터 로드 실패")
