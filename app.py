import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import requests
import json
import numpy as np
import warnings
from datetime import datetime

# 시스템 설정
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Grand Master Terminal", layout="wide")

st.title("🏛️ Grand Master Investment Terminal")
st.markdown("---")

# 1. 데이터 로드 및 업데이트 함수 (캐싱 및 견고성 강화)
@st.cache_data(ttl=3600)
def load_all_data():
    def get_fred(s_id):
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s_id}"
        return pd.read_csv(url, index_col=0, parse_dates=True)

    data = {}
    
    # [유동성 지표]
    try:
        data['fed'] = get_fred('WALCL')
        data['credit'] = get_fred('TOTBKCR')
        data['eur_usd'] = get_fred('DEXUSEU')
        data['usd_jpy'] = get_fred('DEXJPUS')
    except Exception as e:
        st.error(f"FRED 데이터 로드 실패: {e}")
        return None

    # [자산 가격] - 구조적 견고함 확보
    def download_crypto(ticker):
        df = yf.download(ticker, start="2019-01-01", progress=False)
        if df.empty:
            return pd.Series()
        # 최근 yfinance의 MultiIndex 이슈 대응
        if isinstance(df.columns, pd.MultiIndex):
            return df['Close'][ticker]
        return df['Close']

    data['btc'] = download_crypto("BTC-USD")
    data['doge'] = download_crypto("DOGE-USD")
    data['nasdaq'] = download_crypto("^IXIC")

    # [채굴 난이도]
    try:
        with open('difficulty (1).json', 'r') as f:
            diff_raw = json.load(f)['difficulty']
        df_diff = pd.DataFrame(diff_raw)
        df_diff['Date'] = pd.to_datetime(df_diff['x'], unit='ms')
        df_diff = df_diff.set_index('Date').sort_index()
        data['diff'] = pd.to_numeric(df_diff['y'], errors='coerce')
    except Exception as e:
        st.warning(f"난이도 데이터(JSON) 로드 실패: {e}")
        data['diff'] = pd.Series()

    return data

# 데이터 실행
d = load_all_data()

if d is not None and not d['btc'].empty:
    # 2. 분석 로직 (기존과 동일하되 안전장치 추가)
    def get_reward(date):
        if date < pd.Timestamp('2012-11-28'): return 50.0
        elif date < pd.Timestamp('2016-07-09'): return 25.0
        elif date < pd.Timestamp('2020-05-11'): return 12.5
        elif date < pd.Timestamp('2024-04-20'): return 6.25
        else: return 3.125

    # 원가 계산
    df_cost = d['diff'].resample('D').interpolate(method='linear').to_frame(name='diff')
    if not df_cost.empty:
        df_cost['reward'] = [get_reward(date) for date in df_cost.index]
        df_cost['raw_cost'] = df_cost['diff'] / df_cost['reward']

        # 2022년 Calibration
        common = pd.merge(d['btc'], df_cost['raw_cost'], left_index=True, right_index=True)
        target = common[(common.index >= '2022-11-01') & (common.index <= '2023-01-31')]
        
        if not target.empty:
            k_factor = (target.iloc[:,0] / target.iloc[:,1]).min()
            df_cost['mining_floor'] = df_cost['raw_cost'] * k_factor
        else:
            df_cost['mining_floor'] = df_cost['raw_cost'] # 폴백(Fallback)

    # 3. 사이드바 진단 (에러가 났던 지점 수정)
    st.sidebar.header("📋 실시간 진단")
    
    # 안전하게 마지막 값 추출
    curr_btc = d['btc'].iloc[-1]
    
    if 'mining_floor' in df_cost.columns:
        curr_cost = df_cost['mining_floor'].iloc[-1]
        gap = (curr_btc / curr_cost - 1) * 100
        
        st.sidebar.metric("BTC 현재가", f"${curr_btc:,.0f}")
        st.sidebar.metric("채굴 원가", f"${curr_cost:,.0f}", f"{gap:.2f}%")
        
        if gap < 0:
            st.sidebar.error("🔥 진성 항복(Capitulation) 구간")
        else:
            st.sidebar.success("✅ 정상 가동 구간")
    
    # 4. 차트 출력
    st.subheader("📊 Grand Master Dashboard")
    # (선생님의 기존 시각화 코드를 여기에 배치)
    st.info(f"마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

else:
    st.error("📉 데이터를 불러오는 데 실패했습니다. GitHub에 JSON 파일이 있는지, 혹은 인터넷 연결을 확인해주세요.")
    if st.button("다시 시도"):
        st.cache_data.clear()
        st.rerun()
