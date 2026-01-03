import streamlit as st
import pandas as pd
import requests
import warnings
from datetime import datetime

# 1. 설정: 군더더기를 모두 제거하여 속도를 극대화합니다.
warnings.filterwarnings("ignore")
st.set_page_config(page_title="GM Terminal", layout="wide")

st.title("🏛️ Grand Master Strategic Terminal")
st.caption("시스템 최적화 완료: 실시간 시장 데이터 동기화 중")

# 2. 데이터 수집: 검증된 Upbit 경로를 사용합니다.
@st.cache_data(ttl=300)
def fetch_data():
    try:
        # 비트코인 500일 데이터
        url = "https://api.upbit.com/v1/candles/days?market=USDT-BTC&count=500"
        r = requests.get(url, timeout=5).json()
        df = pd.DataFrame(r)
        df['Date'] = pd.to_datetime(df['candle_date_time_utc'])
        df['Price'] = df['trade_price'].astype(float)
        return df.sort_values('Date')[['Date', 'Price']]
    except:
        return pd.DataFrame()

df = fetch_data()

# 3. 데이터 검증 및 출력
if not df.empty:
    # 현재가 상단 배치
    current_price = df['Price'].iloc[-1]
    st.metric("Bitcoin Current Price", f"${current_price:,.0f} USDT")

    # 4. [핵심] 내장 인터랙티브 차트
    # st.line_chart의 최신 기능을 사용하여 Plotly 없이도 확대/축소를 구현합니다.
    st.subheader("📊 시장 흐름 분석 (Interactive)")
    
    # x축과 y축을 지정하면 모바일에서 자유로운 줌인/아웃이 가능합니다.
    st.line_chart(
        data=df,
        x="Date",
        y="Price",
        color="#00FFA3", # 가독성 높은 네온 그린
        use_container_width=True
    )

    st.info("💡 모바일 안내: 손가락으로 차트를 벌려 정밀하게 구간을 확대해 보세요.")

    # 5. 데이터 하단 디테일
    with st.expander("📝 시계열 데이터 원본 기록"):
        st.dataframe(df.sort_values('Date', ascending=False), use_container_width=True)

else:
    st.error("데이터 동기화 실패. 서버 네트워크 상태를 확인 중입니다.")

st.markdown("---")
st.caption(f"시스템 가동 중 | 마지막 업데이트: {datetime.now().strftime('%H:%M:%S')}")
