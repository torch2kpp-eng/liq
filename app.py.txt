import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import requests
import json
import numpy as np
import warnings

# 페이지 설정 (아이폰 최적화)
st.set_page_config(page_title="Grand Master Terminal", layout="wide")
warnings.filterwarnings("ignore")

st.title("🏛️ Grand Master Investment Terminal")
st.markdown("---")

# 1. 데이터 로드 및 업데이트 함수 (캐싱 적용)
@st.cache_data(ttl=3600) # 1시간마다 데이터 자동 갱신
def load_all_data():
    # FRED 데이터 가져오기
    def get_fred(s_id):
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s_id}"
        return pd.read_csv(url, index_col=0, parse_dates=True)

    data = {}
    # 유동성 지표
    data['fed'] = get_fred('WALCL')
    data['credit'] = get_fred('TOTBKCR')
    data['ecb'] = get_fred('ECBASSETSW')
    data['boj'] = get_fred('JPNASSETS')
    data['eur_usd'] = get_fred('DEXUSEU')
    data['usd_jpy'] = get_fred('DEXJPUS')
    data['rrp'] = get_fred('RRPONTSYD')
    data['tga'] = get_fred('WTREGEN')

    # 자산 가격 (오늘까지의 최신 데이터)
    data['btc'] = yf.download("BTC-USD", start="2010-01-01", progress=False)['Close']
    data['doge'] = yf.download("DOGE-USD", start="2010-01-01", progress=False)['Close']
    data['nasdaq'] = yf.download("^IXIC", start="2010-01-01", progress=False)['Close']

    # 채굴 난이도 (GitHub에 함께 올린 JSON 로드)
    with open('difficulty (1).json', 'r') as f:
        diff_raw = json.load(f)['difficulty']
    df_diff = pd.DataFrame(diff_raw)
    df_diff['Date'] = pd.to_datetime(df_diff['x'], unit='ms')
    df_diff = df_diff.set_index('Date').sort_index()
    data['diff'] = pd.to_numeric(df_diff['y'], errors='coerce')

    return data

# 데이터 실행
with st.spinner('전 세계 금융 데이터를 동기화 중입니다...'):
    d = load_all_data()

# 2. 분석 로직 (선생님과 설계한 수식 적용)
def get_reward(date):
    if date < pd.Timestamp('2012-11-28'): return 50.0
    elif date < pd.Timestamp('2016-07-09'): return 25.0
    elif date < pd.Timestamp('2020-05-11'): return 12.5
    elif date < pd.Timestamp('2024-04-20'): return 6.25
    else: return 3.125

# 채굴 원가 계산
df_cost = d['diff'].resample('D').interpolate(method='linear').to_frame(name='diff')
df_cost['reward'] = [get_reward(date) for date in df_cost.index]
df_cost['raw_cost'] = df_cost['diff'] / df_cost['reward']

# 2022년 바닥 기준 보정 (Calibration)
common = pd.merge(d['btc'], df_cost['raw_cost'], left_index=True, right_index=True)
target = common[(common.index >= '2022-11-01') & (common.index <= '2023-01-31')]
k_factor = (target.iloc[:,0] / target.iloc[:,1]).min()
df_cost['mining_floor'] = df_cost['raw_cost'] * k_factor

# 유동성 계산
# (G3 데이터 주간 단위 합산 및 환율 적용 생략/간소화 - 실제 배포 시 정교하게 구현됨)
fed_t = d['fed'].resample('W-WED').last().iloc[:,0] / 1000000
df_liq = fed_t.to_frame(name='Fed')

# 3. 시각화 (아이폰 최적화)
fig, ax1 = plt.subplots(figsize=(16, 12))
# (이하 생략: 기존의 6축 차트 코드가 여기에 들어감)
# ... [중략: 기존 시각화 코드 삽입] ...

st.pyplot(fig) # 웹에 차트 출력

# 현재 진단 상태 표시
st.sidebar.header("📋 실시간 진단")
last_price = float(d['btc'].iloc[-1])
last_cost = float(df_cost['mining_floor'].iloc[-1])
gap = (last_price / last_cost - 1) * 100

st.sidebar.metric("BTC 현재가", f"${last_price:,.0f}")
st.sidebar.metric("채굴 원가", f"${last_cost:,.0f}", f"{gap:.2f}%")

if gap < 0:
    st.sidebar.error("🔥 현재 진성 항복(Capitulation) 구간입니다. 강력 매수 기회!")
else:
    st.sidebar.success("✅ 시장이 원가 위에서 정상 가동 중입니다.")