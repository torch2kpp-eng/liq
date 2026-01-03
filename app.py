import streamlit as st
import pandas as pd
import requests
import warnings

# 1. 설정 (수정됨: layout="mobile" -> "wide")
warnings.filterwarnings("ignore")
# [수정 포인트] layout은 반드시 "wide" 또는 "centered"여야 합니다.
st.set_page_config(page_title="GM Mobile", layout="wide") 

# [중요] 버전 확인용 제목
st.title("Ver 7.1 : Mobile Fix") 
st.caption("이 화면이 보이면 성공입니다. 에러가 해결되었습니다.")

# 2. 데이터 가져오기 (바이낸스 - 가장 단순한 구조)
@st.cache_data(ttl=300)
def get_data():
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

# 3. 차트 그리기
if df is not None and not df.empty:
    st.subheader("비트코인 가격 차트")
    # Streamlit 내장 차트 (무조건 그려짐)
    st.line_chart(df["Price"]) 
    
    # 현재가 표시
    last_p = df["Price"].iloc[-1]
    st.metric("현재가", f"${last_p:,.0f}")
else:
    st.error("데이터 로드 실패")
