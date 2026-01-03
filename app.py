import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime
import warnings

# 1. 고도의 시각적 경험을 위한 설정
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Grand Master Terminal", layout="wide")

# 지적인 비서로서의 서문
st.title("🏛️ Grand Master Strategic Terminal")
st.markdown("---")

# 2. 데이터 수집 엔진 (성공한 Upbit 로직 기반)
@st.cache_data(ttl=300)
def fetch_strategic_data():
    def get_upbit_price(market, count=200):
        try:
            url = f"https://api.upbit.com/v1/candles/days?market={market}&count={count}"
            r = requests.get(url, timeout=5).json()
            df = pd.DataFrame(r)
            df['Date'] = pd.to_datetime(df['candle_date_time_utc'])
            df['Price'] = df['trade_price'].astype(float)
            return df.sort_values('Date')[['Date', 'Price']]
        except:
            return pd.DataFrame()

    # 비트코인 및 도지코인(시장 심리 지표) 수집
    btc = get_upbit_price("USDT-BTC", 500)
    doge = get_upbit_price("USDT-DOGE", 500)
    return btc, doge

with st.spinner('시장 데이터를 정밀 분석 중입니다...'):
    btc_df, doge_df = fetch_strategic_data()

if btc_df.empty:
    st.error("데이터 동기화에 일시적인 장애가 발생했습니다. 잠시 후 다시 시도해 주십시오.")
    st.stop()

# 3. 전략적 분석 레이아웃
col1, col2 = st.columns([3, 1])

with col2:
    st.subheader("📌 Market Status")
    current_btc = btc_df['Price'].iloc[-1]
    st.metric("Bitcoin (USDT)", f"${current_btc:,.0f}")
    
    if not doge_df.empty:
        current_doge = doge_df['Price'].iloc[-1]
        st.metric("Doge (Sentiment Index)", f"${current_doge:.4f}")
    
    st.info("💡 모바일 팁: 차트 영역을 두 손가락으로 벌리면 특정 구간을 정밀하게 탐색할 수 있습니다.")

with col1:
    # 4. 고성능 인터랙티브 차트 (Plotly)
    fig = go.Figure()

    # 비트코인 주력 선 (Log Scale 적용 권장)
    fig.add_trace(go.Scatter(
        x=btc_df['Date'], 
        y=btc_df['Price'],
        mode='lines',
        name='Bitcoin',
        line=dict(color='#00FFA3', width=2.5),
        fill='toself',
        fillcolor='rgba(0, 255, 163, 0.05)'
    ))

    # 차트 레이아웃 최적화 (모바일 줌 기능 활성화)
    fig.update_layout(
        template="plotly_dark",
        height=600,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(
            rangeslider=dict(visible=True), # 하단 기간 조절 슬라이더
            type="date",
            showgrid=False
        ),
        yaxis=dict(
            title="Price (USDT)",
            side="right",
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)',
            exponentformat="none"
        ),
        hovermode="x unified",
        dragmode="zoom" # 기본 드래그 모드를 줌으로 설정
    )

    # 차트 출력
    st.plotly_chart(fig, use_container_width=True, config={
        'scrollZoom': True, # 마우스 휠 및 터치 줌 활성화
        'displayModeBar': False
    })

# 5. 지적 성찰을 위한 하단부
st.markdown("---")
st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} KST | 데이터 제공: Upbit")
