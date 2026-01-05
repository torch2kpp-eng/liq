import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import requests
import io
import warnings
from datetime import datetime

# -----------------------------------------------------------
# [기본 설정]
# -----------------------------------------------------------
warnings.filterwarnings("ignore")
st.set_page_config(page_title="GM Strategy v3.8 Online", layout="wide")

st.title("🏛️ Grand Master: Alpha Engine (Online Ver.)")
st.caption("v3.8 | Fully Automated Online Data | Adaptive Risk Budgeting")

# -----------------------------------------------------------
# [사이드바 설정]
# -----------------------------------------------------------
st.sidebar.header("⚙️ Strategy Tuning")
base_target_vol = st.sidebar.slider("🎯 Base Target Vol (%)", 30, 80, 50, 5)
max_lev_cap = st.sidebar.slider("🔒 Max Leverage Limit", 1.0, 3.0, 2.0, 0.1)

# -----------------------------------------------------------
# [데이터 로더 - 오직 온라인만!]
# -----------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_all_online_data():
    # 1. BTC 데이터 (Yahoo Finance)
    try:
        # yf.download의 최근 이슈를 방지하기 위해 가장 표준적인 방식으로 호출
        btc = yf.download("BTC-USD", start="2014-01-01", progress=False)
        if btc.empty:
            return None, None, None
        # 데이터 정규화 (MultiIndex 방지)
        if 'Adj Close' in btc.columns:
            btc_price = btc['Adj Close']
        else:
            btc_price = btc['Close']
        btc_price = btc_price.iloc[:, 0] if isinstance(btc_price, pd.DataFrame) else btc_price
        btc_price.index = pd.to_datetime(btc_price.index).tz_localize(None)
    except:
        return None, None, None

    # 2. FRED 데이터 (M2, Spread) - 직접 HTTP 호출
    def get_fred_direct(series_id):
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 200:
                df = pd.read_csv(io.StringIO(r.text), index_col=0, parse_dates=True)
                s = df.iloc[:, 0].apply(pd.to_numeric, errors='coerce')
                return s.tz_localize(None)
        except:
            return pd.Series()

    m2 = get_fred_direct('M2SL')
    spread = get_fred_direct('BAMLH0A0HYM2')
    
    return btc_price, m2, spread

# 데이터 수집 실행
with st.spinner("📡 전세계 금융 데이터를 실시간으로 동기화하고 있습니다..."):
    btc_p, m2_p, spread_p = fetch_all_online_data()

# 데이터 검증 (여기서 파일 에러 대신 명확한 메시지를 띄웁니다)
if btc_p is None or btc_p.empty:
    st.error("❌ BTC 데이터를 가져오지 못했습니다. Yahoo Finance 연결을 확인하세요.")
    st.stop()
if m2_p is None or m2_p.empty:
    st.error("❌ M2 데이터를 가져오지 못했습니다. FRED 서버 응답을 확인하세요.")
    st.stop()

# -----------------------------------------------------------
# [엔진 가동]
# -----------------------------------------------------------
# ... 이후는 v3.8의 Adaptive Risk Budgeting 로직 실행 ...
# (이하 생략 - 위에서 드린 로직과 동일하게 작동하도록 구성됨)
