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

# 1. 데이터 로드 함수 (이전과 동일하게 유지하되 리턴값 최적화)
@st.cache_data(ttl=3600)
def load_all_data():
    def get_fred(s_id):
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s_id}"
        return pd.read_csv(url, index_col=0, parse_dates=True)

    data = {}
    try:
        data['fed'] = get_fred('WALCL')
        data['credit'] = get_fred('TOTBKCR')
        data['eur_usd'] = get_fred('DEXUSEU')
        data['usd_jpy'] = get_fred('DEXJPUS')
        
        # Crypto/Nasdaq 데이터 로드
        def d_asset(ticker):
            df = yf.download(ticker, start="2019-01-01", progress=False)
            if df.empty: return pd.Series()
            if isinstance(df.columns, pd.MultiIndex): return df['Close'][ticker]
            return df['Close']
            
        data['btc'] = d_asset("BTC-USD")
        data['doge'] = d_asset("DOGE-USD")
        data['nasdaq'] = d_asset("^IXIC")

        with open('difficulty (1).json', 'r') as f:
            diff_raw = json.load(f)['difficulty']
        df_diff = pd.DataFrame(diff_raw)
        df_diff['Date'] = pd.to_datetime(df_diff['x'], unit='ms')
        data['diff'] = df_diff.set_index('Date').sort_index()['y']
        
        return data
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return None

d = load_all_data()

if d:
    # 2. 계산 로직
    # 원가 계산
    df_c = pd.DataFrame(index=pd.date_range(start="2019-01-01", end=datetime.now()))
    diff_daily = d['diff'].resample('D').interpolate(method='linear')
    df_c = df_c.join(diff_daily.to_frame(name='diff'), how='left').fillna(method='ffill')
    
    def get_reward(dt):
        if dt < pd.Timestamp('2020-05-11'): return 12.5
        elif dt < pd.Timestamp('2024-04-20'): return 6.25
        else: return 3.125
        
    df_c['reward'] = [get_reward(date) for date in df_c.index]
    df_c['raw_cost'] = df_c['diff'] / df_c['reward']
    
    # 2022 Calibration (FTX 바닥 기준)
    common = pd.merge(d['btc'], df_c['raw_cost'], left_index=True, right_index=True)
    target = common[(common.index >= '2022-11-01') & (common.index <= '2023-01-31')]
    k = (target.iloc[:,0] / target.iloc[:,1]).min() if not target.empty else 0.00000008 # Fallback
    df_c['floor'] = df_c['raw_cost'] * k

    # 유동성 계산 (G3 간단 모델)
    g3 = d['fed'].resample('W-WED').last().iloc[:,0] / 1000000
    g3_yoy = g3.pct_change(52) * 100

    # 3. 데이터 시프트 (-90일)
    btc_s = d['btc'].copy(); btc_s.index -= pd.Timedelta(days=90)
    nasdaq_s = d['nasdaq'].copy(); nasdaq_s.index -= pd.Timedelta(days=90)
    doge_s = d['doge'].copy(); doge_s.index -= pd.Timedelta(days=90)
    floor_s = df_c['floor'].copy(); floor_s.index -= pd.Timedelta(days=90)

    # 4. 차트 그리기 (중요!)
    start_viz = '2023-01-01'
    dv = g3_yoy[g3_yoy.index >= start_viz]
    
    # 그래프 스타일 설정
    plt.style.use('dark_background')
    fig, ax1 = plt.subplots(figsize=(14, 8), facecolor='black')
    
    # Axis 1: G3 Growth (Yellow)
    ax1.plot(dv.index, dv, color='#FFD700', lw=2, label='G3 Growth (YoY)', alpha=0.8)
    ax1.fill_between(dv.index, dv, 0, where=(dv>=0), color='#FFD700', alpha=0.1)
    ax1.set_ylabel('Liquidity Growth %', color='#FFD700', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#FFD700')
    ax1.set_ylim(-20, 40)

    # Axis 2: BTC Price (Black/White in dark mode)
    ax2 = ax1.twinx()
    ax2.spines['right'].set_position(('axes', 1.05))
    bv = btc_s[btc_s.index >= start_viz]
    fv = floor_s[floor_s.index >= start_viz]
    ax2.plot(bv.index, bv, color='white', lw=2, label='BTC Price (Shifted)')
    ax2.plot(fv.index, fv, color='red', lw=1.5, ls='--', label='Mining Floor')
    ax2.fill_between(bv.index, bv, fv, where=(bv < fv), color='red', alpha=0.3, hatch='///')
    ax2.set_yscale('log')
    ax2.set_ylabel('BTC Price (Log)', color='white', fontsize=12)

    # Axis 3: Nasdaq (Magenta)
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('axes', 1.15))
    nv = nasdaq_s[nasdaq_s.index >= start_viz]
    ax3.plot(nv.index, nv, color='#D62780', lw=1.5, label='Nasdaq (Shifted)', alpha=0.7)
    ax3.set_ylabel('Nasdaq', color='#D62780', fontsize=12)
    ax3.tick_params(axis='y', labelcolor='#D62780')

    # Axis 4: DOGE (Orange)
    ax4 = ax1.twinx()
    ax4.spines['right'].set_position(('axes', 1.25))
    dgv = doge_s[doge_s.index >= start_viz]
    ax4.plot(dgv.index, dgv, color='#FFA500', lw=1.2, label='DOGE (Shifted)', alpha=0.6)
    ax4.set_yscale('log')
    ax4.set_ylabel('DOGE', color='#FFA500', fontsize=12)
    ax4.tick_params(axis='y', labelcolor='#FFA500')

    # 레이아웃 정리
    ax1.set_title("The Grand Master: Liquidity vs Assets Convergence", fontsize=16, pad=20)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
    plt.grid(True, alpha=0.2)
    
    # [핵심] 웹에 차트 출력
    st.pyplot(fig)

    # 사이드바 정보 업데이트
    st.sidebar.header("📋 시장 진단")
    curr_btc = float(d['btc'].iloc[-1])
    curr_floor = float(df_c['floor'].iloc[-1])
    gap = (curr_btc / curr_floor - 1) * 100
    
    st.sidebar.metric("BTC 현재가", f"${curr_btc:,.0f}")
    st.sidebar.metric("채굴 원가", f"${curr_floor:,.0f}", f"{gap:.2f}%")
    
    if gap < 0:
        st.sidebar.error("🔥 진성 항복 구간")
        st.sidebar.write("역사적 저평가 영역입니다.")
    else:
        st.sidebar.success("✅ 정상 가동 중")

else:
    st.error("데이터 로드에 실패했습니다.")
