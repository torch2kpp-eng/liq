import streamlit as st
import pandas as pd
import requests
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import warnings
from datetime import datetime

# 설정
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Grand Master Terminal", layout="wide")
st.title("🏛️ Grand Master Terminal (Binance Core)")

# ------------------------------------------------------------------------------
# 1. 데이터 로드 엔진 (바이낸스 API 탑재)
# ------------------------------------------------------------------------------
@st.cache_data(ttl=300) # 5분마다 갱신
def get_crypto_data():
    data = {}
    
    # [A] 바이낸스에서 데이터 가져오기 (가장 확실한 방법)
    def fetch_binance(symbol, limit=1000):
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {"symbol": symbol, "interval": "1d", "limit": limit}
            response = requests.get(url, params=params)
            data = response.json()
            
            # 데이터 프레임 변환
            df = pd.DataFrame(data, columns=["Open Time", "Open", "High", "Low", "Close", "Volume", "Close Time", "QAV", "NAT", "TBB", "TBQ", "Ignore"])
            df["Date"] = pd.to_datetime(df["Open Time"], unit='ms')
            df["Close"] = df["Close"].astype(float)
            df = df.set_index("Date")
            return df["Close"]
        except Exception as e:
            st.error(f"바이낸스 연결 실패 ({symbol}): {e}")
            return pd.Series(dtype=float)

    # BTC, DOGE 로드
    data['btc'] = fetch_binance("BTCUSDT")
    data['doge'] = fetch_binance("DOGEUSDT")
    
    # [B] 나스닥 (야후 사용 - 실패시 무시)
    try:
        nasdaq = yf.download("^IXIC", period="2y", progress=False)
        if isinstance(nasdaq.columns, pd.MultiIndex):
            data['nasdaq'] = nasdaq.xs('Close', axis=1, level=0)["^IXIC"]
        else:
            data['nasdaq'] = nasdaq['Close']
        data['nasdaq'].index = data['nasdaq'].index.tz_localize(None) # 시간대 제거
    except:
        data['nasdaq'] = pd.Series(dtype=float)

    # [C] 난이도 데이터 (JSON 파일)
    # 파일 이름이 정확해야 합니다. 'difficulty (1).json'
    try:
        with open('difficulty (1).json', 'r') as f:
            diff_raw = json.load(f)['difficulty']
        df_d = pd.DataFrame(diff_raw)
        df_d['Date'] = pd.to_datetime(df_d['x'], unit='ms')
        data['diff'] = df_d.set_index('Date').sort_index()['y']
    except Exception as e:
        # 파일이 없을 경우를 대비해 더미 데이터 생성 (에러 방지용)
        st.warning(f"⚠️ 난이도 파일 로드 실패: {e} (로컬 파일명 확인 필요)")
        data['diff'] = pd.Series(dtype=float)

    return data

# 데이터 로드
d = get_crypto_data()

# ------------------------------------------------------------------------------
# 2. 데이터 생존 확인 (디버깅)
# ------------------------------------------------------------------------------
if d['btc'].empty:
    st.error("🚨 비트코인 데이터를 가져오지 못했습니다. 네트워크 상태를 확인하세요.")
    st.stop()

# ------------------------------------------------------------------------------
# 3. 지표 계산 (채굴 원가)
# ------------------------------------------------------------------------------
df_main = pd.DataFrame(index=d['btc'].index)

# 난이도 데이터가 있으면 계산, 없으면 0 처리
if not d['diff'].empty:
    daily_diff = d['diff'].resample('D').interpolate(method='linear')
    df_main = df_main.join(daily_diff.rename('diff'), how='left').ffill()
    
    def get_reward(dt):
        if dt < pd.Timestamp('2024-04-20'): return 6.25
        else: return 3.125
        
    df_main['reward'] = [get_reward(x) for x in df_main.index]
    df_main['cost'] = df_main['diff'] / df_main['reward']
    
    # Calibration (2022 바닥 기준)
    try:
        # 데이터 교집합 찾기
        common = pd.concat([d['btc'], df_main['cost']], axis=1).dropna()
        subset = common[(common.index >= '2022-11-01') & (common.index <= '2023-01-31')]
        k = (subset.iloc[:,0] / subset.iloc[:,1]).min() if not subset.empty else 0.00000008
    except: k = 0.00000008
    
    df_main['floor'] = df_main['cost'] * k
else:
    df_main['floor'] = np.nan

# ------------------------------------------------------------------------------
# 4. 차트 그리기 (Plotly)
# ------------------------------------------------------------------------------
st.subheader("📊 Grand Master Chart (Live)")

fig = make_subplots(specs=[[{"secondary_y": True}]])

# [Main] BTC Price
fig.add_trace(go.Scatter(x=d['btc'].index, y=d['btc'], name="BTC Price", 
                         line=dict(color='white', width=2)), secondary_y=False)

# [Main] Mining Floor
if not np.isnan(df_main['floor'].iloc[-1]):
    fig.add_trace(go.Scatter(x=df_main.index, y=df_main['floor'], name="Mining Cost", 
                             line=dict(color='red', dash='dot')), secondary_y=False)

# [Sub] Nasdaq (-90d)
if not d['nasdaq'].empty:
    nd_s = d['nasdaq'].shift(90)
    # 날짜 필터링 (BTC와 기간 맞추기)
    nd_s = nd_s[nd_s.index >= d['btc'].index[0]]
    fig.add_trace(go.Scatter(x=nd_s.index, y=nd_s, name="Nasdaq (-90d)", 
                             line=dict(color='#D62780', width=1.5), opacity=0.7), secondary_y=True)

# [Sub] DOGE (-90d)
if not d['doge'].empty:
    dg_s = d['doge'].shift(90)
    fig.add_trace(go.Scatter(x=dg_s.index, y=dg_s, name="DOGE (-90d)", 
                             line=dict(color='orange', width=1.5), opacity=0.7), secondary_y=True)

# 레이아웃
fig.update_layout(template="plotly_dark", height=600, 
                  title_text="Liquidity & Asset Convergence",
                  legend=dict(orientation="h", y=1.1))
fig.update_yaxes(type="log", title_text="BTC (Log)", secondary_y=False)
fig.update_yaxes(title_text="External Assets", secondary_y=True)

st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------------------
# 5. 사이드바 정보
# ------------------------------------------------------------------------------
last_price = d['btc'].iloc[-1]
st.sidebar.metric("BTC Price (Binance)", f"${last_price:,.0f}")

if not np.isnan(df_main['floor'].iloc[-1]):
    last_floor = df_main['floor'].iloc[-1]
    gap = (last_price / last_floor - 1) * 100
    st.sidebar.metric("Mining Cost", f"${last_floor:,.0f}", f"{gap:.2f}%")
    
    if gap < 0:
        st.sidebar.error("🔥 Capitulation Zone")
    else:
        st.sidebar.success("✅ Healthy Zone")
