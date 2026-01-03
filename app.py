import streamlit as st
import pandas as pd
import requests
import numpy as np
import warnings

# 1. 설정
warnings.filterwarnings("ignore")
st.set_page_config(page_title="GM Final", layout="wide")

st.title("Ver 7.2 : Connectivity Test") 

# 2. 데이터 가져오기 (3중 안전장치)
@st.cache_data(ttl=60)
def get_data_survivor():
    # [시도 1] 업비트 (Upbit) - USDT 마켓
    try:
        url = "https://api.upbit.com/v1/candles/days"
        params = {"market": "USDT-BTC", "count": 200}
        headers = {"accept": "application/json"}
        response = requests.get(url, params=params, headers=headers, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            df['Date'] = pd.to_datetime(df['candle_date_time_utc'])
            df['Price'] = df['trade_price'].astype(float)
            df = df.sort_values('Date') # 날짜 오름차순 정렬
            return df[["Date", "Price"]].set_index("Date"), "Upbit (USDT)"
    except Exception as e:
        pass # 실패하면 다음으로

    # [시도 2] 코인게코 (CoinGecko)
    try:
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        params = {"vs_currency": "usd", "days": "100"}
        headers = {"User-Agent": "Mozilla/5.0"} # 브라우저인 척 속이기
        response = requests.get(url, params=params, headers=headers, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            prices = data['prices'] # [timestamp, price] 리스트
            df = pd.DataFrame(prices, columns=['timestamp', 'Price'])
            df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df[["Date", "Price"]].set_index("Date"), "CoinGecko"
    except Exception as e:
        pass

    # [시도 3] 최후의 보루: 랜덤 테스트 데이터 (차트 기능 확인용)
    # 이게 나온다면 서버 인터넷이 완전히 막힌 것입니다.
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100)
    prices = np.linspace(90000, 100000, 100) + np.random.randn(100) * 1000
    df = pd.DataFrame({'Date': dates, 'Price': prices})
    return df.set_index('Date'), "⚠️ TEST DATA (Network Blocked)"

# 데이터 로드
df, source_name = get_data_survivor()

# 3. 결과 출력
st.subheader(f"데이터 소스: {source_name}")

if "TEST DATA" in source_name:
    st.error("모든 외부 접속이 차단되어 '테스트 데이터'를 표시합니다. 하지만 차트는 보일 겁니다.")
else:
    st.success(f"연결 성공! {source_name} 데이터를 불러왔습니다.")

# 4. 차트 그리기
if df is not None and not df.empty:
    # 1) Streamlit 내장 라인 차트 (가장 안전)
    st.line_chart(df["Price"])
    
    # 2) 현재가 표시
    last_p = df["Price"].iloc[-1]
    st.metric("Bitcoin Price", f"${last_p:,.0f}")
    
    # 3) 데이터 표 확인
    with st.expander("데이터 원본 보기"):
        st.dataframe(df.tail())
else:
    st.error("치명적 오류: 데이터를 생성할 수 없습니다.")
