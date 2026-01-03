import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

# 1. 기본 설정 (모바일 최적화)
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Market Pulse", layout="wide")

st.title("📈 Market Pulse: Crypto vs Tech")
st.caption("비트코인 | 도지코인 | 나스닥 (실시간 비교)")

# 2. 데이터 로드 (가장 강력한 하이브리드 방식)
@st.cache_data(ttl=300)
def load_market_data():
    data = {}
    
    # [A] 코인 데이터 (바이낸스 API - 차단 걱정 없음)
    def fetch_binance(symbol):
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {"symbol": symbol, "interval": "1d", "limit": 1000} # 최근 1000일
            r = requests.get(url, params=params).json()
            df = pd.DataFrame(r, columns=["t", "o", "h", "l", "c", "v", "T", "q", "n", "V", "Q", "B"])
            df["Date"] = pd.to_datetime(df["t"], unit='ms')
            df["Close"] = df["c"].astype(float)
            return df.set_index("Date")["Close"]
        except:
            return pd.Series(dtype=float)

    data['BTC'] = fetch_binance("BTCUSDT")
    data['DOGE'] = fetch_binance("DOGEUSDT")

    # [B] 나스닥 데이터 (야후 파이낸스)
    try:
        # 시간대 문제 해결을 위해 tz_localize(None) 필수
        nd = yf.download("^IXIC", period="3y", progress=False)
        if isinstance(nd.columns, pd.MultiIndex):
            s = nd.xs('Close', axis=1, level=0)["^IXIC"]
        else:
            s = nd['Close']
        s.index = s.index.tz_localize(None)
        data['Nasdaq'] = s
    except:
        data['Nasdaq'] = pd.Series(dtype=float)

    return data

# 데이터 로딩
d = load_market_data()

if d['BTC'].empty:
    st.error("🚨 데이터 연결 실패. 잠시 후 새로고침 해주세요.")
    st.stop()

# 3. 옵션 패널 (사이드바 대신 상단 배치로 모바일 접근성 UP)
with st.expander("⚙️ 차트 옵션 설정 (클릭)", expanded=False):
    show_nasdaq = st.checkbox("나스닥 (Nasdaq) 보기", value=True)
    show_doge = st.checkbox("도지코인 (DOGE) 보기", value=True)
    apply_shift = st.checkbox("90일 선행 지표로 변환 (Grand Master Logic)", value=False)
    
    st.info("💡 팁: 차트 위에서 손가락을 벌리면 확대(Zoom), 오므리면 축소됩니다.")

# 4. 차트 그리기 (Plotly)
fig = make_subplots(specs=[[{"secondary_y": True}]])

# [Main] BTC (왼쪽 축, 로그 스케일)
fig.add_trace(
    go.Scatter(x=d['BTC'].index, y=d['BTC'], name="Bitcoin", 
               line=dict(color='white', width=2)), 
    secondary_y=False
)

# [Sub] Nasdaq
if show_nasdaq and not d['Nasdaq'].empty:
    y_data = d['Nasdaq'].shift(90) if apply_shift else d['Nasdaq']
    name_suffix = "(-90d)" if apply_shift else ""
    
    fig.add_trace(
        go.Scatter(x=y_data.index, y=y_data, name=f"Nasdaq {name_suffix}", 
                   line=dict(color='#D62780', width=1.5), opacity=0.8),
        secondary_y=True
    )

# [Sub] DOGE
if show_doge and not d['DOGE'].empty:
    y_data = d['DOGE'].shift(90) if apply_shift else d['DOGE']
    name_suffix = "(-90d)" if apply_shift else ""
    
    fig.add_trace(
        go.Scatter(x=y_data.index, y=y_data, name=f"DOGE {name_suffix}", 
                   line=dict(color='orange', width=1.5), opacity=0.8),
        secondary_y=True
    )

# 5. 스타일링 (모바일 최적화)
fig.update_layout(
    template="plotly_dark",
    height=600, # 모바일에서 시원하게 보이도록 높이 확보
    title_text="",
    legend=dict(
        orientation="h", # 범례 가로 배치
        y=1.1, x=0,
        font=dict(size=10)
    ),
    margin=dict(l=10, r=10, t=40, b=10), # 여백 최소화
    hovermode="x unified" # 터치 시 모든 데이터 한 번에 보기
)

# 축 설정
fig.update_yaxes(title_text="BTC (Log Scale)", type="log", showgrid=True, gridcolor='rgba(255,255,255,0.1)', secondary_y=False)
fig.update_yaxes(title_text="Nasdaq / Doge", showgrid=False, secondary_y=True)

# 차트 출력
st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}) 

# 6. 하단 요약 정보
col1, col2, col3 = st.columns(3)
col1.metric("BTC", f"${d['BTC'].iloc[-1]:,.0f}")
if not d['Nasdaq'].empty:
    col2.metric("Nasdaq", f"{d['Nasdaq'].iloc[-1]:,.0f}")
col3.metric("DOGE", f"${d['DOGE'].iloc[-1]:.4f}")
