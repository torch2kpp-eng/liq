import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
import yfinance as yf
import json
import numpy as np
import io
import warnings

# 1. 시스템 설정
warnings.filterwarnings("ignore")
st.set_page_config(page_title="GM Stable", layout="wide")

st.title("Ver 8.1 : System Stabilized")
st.caption("무한 로딩 방지 패치 적용됨 | 데이터 타임아웃 기능 활성화")

# 2. 데이터 수집 (안전장치 강화)
@st.cache_data(ttl=3600)
def get_data_safe():
    data = {}
    
    # [A] Upbit (가장 빠름 - 우선 수집)
    try:
        url = "https://api.upbit.com/v1/candles/days?market=USDT-BTC&count=1000"
        r = requests.get(url, timeout=3).json() # 3초 타임아웃
        df = pd.DataFrame(r)
        df['Date'] = pd.to_datetime(df['candle_date_time_utc'])
        df['Price'] = df['trade_price'].astype(float)
        data['btc'] = df.set_index('Date').sort_index()[['Price']]
    except:
        data['btc'] = None

    # [B] FRED (유동성) - 무거운 작업이므로 타임아웃 필수
    def fetch_fred(code):
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"
            # requests로 먼저 받고 타임아웃 걸기
            r = requests.get(url, timeout=2) 
            if r.status_code == 200:
                df = pd.read_csv(io.StringIO(r.text), index_col=0, parse_dates=True)
                return df
        except:
            return None
    
    # 핵심 유동성 지표만 가져옵니다 (속도 최적화)
    data['fed'] = fetch_fred('WALCL')   # 연준 자산
    data['tga'] = fetch_fred('WTREGEN') # TGA 계좌
    data['rrp'] = fetch_fred('RRPONTSYD') # 역레포

    # [C] Nasdaq (Yahoo)
    try:
        # 최근 3년치만 가져와서 속도 향상
        nd = yf.download("^IXIC", period="3y", progress=False)
        if isinstance(nd.columns, pd.MultiIndex):
            s = nd.xs('Close', axis=1, level=0)["^IXIC"]
        else:
            s = nd['Close']
        s.index = s.index.tz_localize(None)
        data['nasdaq'] = s
    except:
        data['nasdaq'] = None

    # [D] Doge (Upbit)
    try:
        url = "https://api.upbit.com/v1/candles/days?market=USDT-DOGE&count=1000"
        r = requests.get(url, timeout=3).json()
        df = pd.DataFrame(r)
        df['Date'] = pd.to_datetime(df['candle_date_time_utc'])
        df['Price'] = df['trade_price'].astype(float)
        data['doge'] = df.set_index('Date').sort_index()[['Price']]
    except:
        data['doge'] = None

    # [E] 난이도 파일
    try:
        with open('difficulty (1).json', 'r') as f:
            d_json = json.load(f)['difficulty']
        df_d = pd.DataFrame(d_json)
        df_d['Date'] = pd.to_datetime(df_d['x'], unit='ms')
        data['diff'] = df_d.set_index('Date').sort_index()['y']
    except:
        data['diff'] = None

    return data

# 로딩 표시
with st.spinner('데이터를 안전하게 가져오는 중... (최대 10초)'):
    dfs = get_data_safe()

if dfs['btc'] is None:
    st.error("비트코인 데이터 연결 실패. 잠시 후 다시 시도하세요.")
    st.stop()

# 3. 데이터 가공 (유동성 간소화)
# G3를 전부 계산하면 터지므로, 'Fed Net Liquidity'를 메인으로 씁니다.
df = dfs['fed'].resample('W-WED').last() if dfs['fed'] is not None else pd.DataFrame()
df.columns = ['Fed']

if dfs['tga'] is not None: 
    tga = dfs['tga'].resample('W-WED').mean()
    df = df.join(tga.rename(columns={tga.columns[0]: 'TGA'}))
else: df['TGA'] = 0

if dfs['rrp'] is not None:
    rrp = dfs['rrp'].resample('W-WED').mean()
    df = df.join(rrp.rename(columns={rrp.columns[0]: 'RRP'}))
else: df['RRP'] = 0

df = df.fillna(method='ffill')

# Fed Net Liquidity 계산
try:
    # 단위 조정 (백만 달러 -> 조 달러)
    df['Net_Liq_Tril'] = (df['Fed'] - df['TGA'] - (df['RRP'] * 1000)) / 1_000_000
    df['Liq_YoY'] = df['Net_Liq_Tril'].pct_change(52) * 100
except:
    df['Liq_YoY'] = 0

# 원가 계산
df_cost = pd.DataFrame(index=dfs['btc'].index)
if dfs['diff'] is not None:
    d_daily = dfs['diff'].resample('D').interpolate()
    common_idx = df_cost.index.intersection(d_daily.index)
    df_cost.loc[common_idx, 'Difficulty'] = d_daily.loc[common_idx]
    
    def get_reward(d):
        if d < pd.Timestamp('2024-04-20'): return 6.25
