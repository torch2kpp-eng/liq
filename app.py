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

# ---------------------------------------------------------
# 1. 시스템 설정
# ---------------------------------------------------------
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Grand Master Terminal", layout="wide")

st.title("🏛️ Grand Master Investment Terminal")
st.markdown("---")

# ---------------------------------------------------------
# 2. 데이터 로드 (에러 방지 강화)
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def load_data_safe():
    data = {}
    
    # [A] 자산 가격 (Yahoo Finance) - 가장 중요
    def get_ticker(t):
        try:
            df = yf.download(t, start="2019-01-01", progress=False)
            if df.empty: return pd.Series(dtype='float64')
            if isinstance(df.columns, pd.MultiIndex): return df['Close'][t]
            return df['Close']
        except: return pd.Series(dtype='float64')

    data['btc'] = get_ticker("BTC-USD")
    data['nasdaq'] = get_ticker("^IXIC")
    data['doge'] = get_ticker("DOGE-USD")

    # [B] 채굴 난이도 (JSON)
    try:
        with open('difficulty (1).json', 'r') as f:
            diff_raw = json.load(f)['difficulty']
        df_d = pd.DataFrame(diff_raw)
        df_d['Date'] = pd.to_datetime(df_d['x'], unit='ms')
        data['diff'] = df_d.set_index('Date').sort_index()['y']
    except:
        data['diff'] = pd.Series(dtype='float64')

    # [C] 유동성 (FRED) - 실패 확률 높음 -> 예외처리 필수
    def get_fred(id):
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"
            # 타임아웃 설정으로 무한 로딩 방지
            return pd.read_csv(url, index_col=0, parse_dates=True, timeout=5)
        except:
            return pd.DataFrame()

    data['fed'] = get_fred('WALCL')
    
    return data

# 데이터 로드 실행
d = load_data_safe()

# ---------------------------------------------------------
# 3. 데이터 진단 (디버깅 정보 출력)
# ---------------------------------------------------------
# 데이터가 잘 왔는지 화면에 찍어봅니다.
col1, col2, col3 = st.columns(3)
col1.metric("BTC 데이터 수", f"{len(d['btc'])} rows")
col2.metric("난이도 데이터 수", f"{len(d['diff'])} rows")
col3.metric("FRED(유동성) 데이터 수", f"{len(d['fed'])} rows")

if d['btc'].empty:
    st.error("❌ 비트코인 데이터를 가져오지 못했습니다. 잠시 후 새로고침 해주세요.")
    st.stop()

# ---------------------------------------------------------
# 4. 지표 계산
# ---------------------------------------------------------
# 원가 계산
df_calc = pd.DataFrame(index=d['btc'].index)
if not d['diff'].empty:
    daily_diff = d['diff'].resample('D').interpolate(method='linear')
    df_calc = df_calc.join(daily_diff.rename('diff'), how='left').ffill()
    
    # 반감기 보정
    def get_reward(dt):
        if dt < pd.Timestamp('2020-05-11'): return 12.5
        elif dt < pd.Timestamp('2024-04-20'): return 6.25
        else: return 3.125
    
    df_calc['reward'] = [get_reward(x) for x in df_calc.index]
    df_calc['cost'] = df_calc['diff'] / df_calc['reward']
    
    # Calibration (2022 바닥)
    # 데이터가 겹치는 구간이 있어야 계산 가능
    try:
        aligned = pd.concat([d['btc'], df_calc['cost']], axis=1).dropna()
        subset = aligned[(aligned.index >= '2022-11-01') & (aligned.index <= '2023-01-31')]
        if not subset.empty:
            k = (subset.iloc[:,0] / subset.iloc[:,1]).min()
        else:
            k = 0.00000008 # fallback
    except:
        k = 0.00000008
        
    df_calc['floor'] = df_calc['cost'] * k
else:
    df_calc['floor'] = np.nan

# ---------------------------------------------------------
# 5. 차트 그리기 (BTC를 메인으로!)
# ---------------------------------------------------------
st.subheader("📊 Grand Master Chart (Debug Ver.)")

# 2023년부터 보기
start_date = '2023-01-01'
viz_btc = d['btc'][d['btc'].index >= start_date]

fig, ax1 = plt.subplots(figsize=(12, 7))

# [Layer 1] BTC (무조건 그려짐) - 검은색 선
ax1.plot(viz_btc.index, viz_btc, color='black', lw=2, label='BTC Price', zorder=5)
ax1.set_ylabel('BTC Price ($)', fontweight='bold')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)

# [Layer 2] Mining Floor (원가) - 빨간 점선
if 'floor' in df_calc.columns:
    viz_floor = df_calc['floor'][df_calc['floor'].index >= start_date]
    # 인덱스 매칭
    common_idx = viz_btc.index.intersection(viz_floor.index)
    if not common_idx.empty:
        ax1.plot(common_idx, viz_floor.loc[common_idx], color='red', ls='--', lw=1.5, label='Mining Floor', zorder=4)
        # Gap 채우기
        ax1.fill_between(common_idx, viz_btc.loc[common_idx], viz_floor.loc[common_idx],
                         where=(viz_btc.loc[common_idx] < viz_floor.loc[common_idx]),
                         color='red', alpha=0.3, label='Capitulation')

# [Layer 3] 유동성 (데이터가 있을 때만 그리기)
if not d['fed'].empty:
    try:
        liq = d['fed'].resample('W-WED').last().iloc[:,0].pct_change(52) * 100
        viz_liq = liq[liq.index >= start_date]
        
        ax2 = ax1.twinx()
        ax2.plot(viz_liq.index, viz_liq, color='#FFD700', lw=0, marker=None, label='Liquidity') # 범례용 더미
        ax2.fill_between(viz_liq.index, viz_liq, 0, where=(viz_liq>=0), color='#FFD700', alpha=0.2, zorder=1)
        ax2.fill_between(viz_liq.index, viz_liq, 0, where=(viz_liq<0), color='gray', alpha=0.1, zorder=1)
        ax2.set_ylabel('Liquidity Growth (YoY)', color='#DAA520')
        ax2.set_ylim(-20, 40)
    except:
        st.warning("⚠️ 유동성 지표 계산 중 오류 발생 (데이터 부족)")

# [Layer 4] Nasdaq & DOGE (Shift 적용)
ax3 = ax1.twinx()
# 축 위치 조정 (오른쪽으로 밀기)
ax3.spines['right'].set_position(('axes', 1.15))

if not d['nasdaq'].empty:
    viz_nd = d['nasdaq'].shift(90) # -90일 Shift
    viz_nd = viz_nd[viz_nd.index >= start_date]
    ax3.plot(viz_nd.index, viz_nd, color='#D62780', lw=1.5, alpha=0.6, label='Nasdaq (-90d)')

if not d['doge'].empty:
    viz_dg = d['doge'].shift(90)
    viz_dg = viz_dg[viz_dg.index >= start_date]
    # 도지는 스케일이 다르므로 별도 축 혹은 정규화 필요하지만, 일단 겹쳐봅니다.
    # 여기서는 복잡도를 줄이기 위해 나스닥 축 공유 (경향성만 확인)
    # 만약 스케일 문제가 심하면 ax4 생성
    ax4 = ax1.twinx()
    ax4.spines['right'].set_position(('axes', 1.25))
    ax4.plot(viz_dg.index, viz_dg, color='orange', lw=1.5, alpha=0.6, label='DOGE (-90d)')
    ax4.set_ylabel('DOGE', color='orange')
    ax4.set_yscale('log')

# 범례 및 타이틀
ax1.set_title("Grand Master Chart (Mobile Ver)", fontsize=14)
ax1.legend(loc='upper left', fontsize=8)

# 차트 출력
st.pyplot(fig)

# 사이드바 업데이트
if not d['btc'].empty and 'floor' in df_calc.columns:
    last_price = d['btc'].iloc[-1]
    last_floor = df_calc['floor'].iloc[-1]
    if not np.isnan(last_floor):
        gap = (last_price / last_floor - 1) * 100
        st.sidebar.metric("BTC Price", f"${last_price:,.0f}")
        st.sidebar.metric("Mining Cost", f"${last_floor:,.0f}", f"{gap:.2f}%")
