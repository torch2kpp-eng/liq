import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
import yfinance as yf
import json
import numpy as np
import warnings

# 1. 시스템 설정
warnings.filterwarnings("ignore")
st.set_page_config(page_title="GM Liquidity", layout="wide")

st.title("🏛️ Grand Master: Global Liquidity & Assets")
st.caption("G3 유동성 | 신용 창출 | 비트코인 | 나스닥 (통합 분석)")

# ------------------------------------------------------------------------------
# 2. 데이터 수집 엔진 (FRED + Upbit + Yahoo)
# ------------------------------------------------------------------------------
@st.cache_data(ttl=3600) # 1시간마다 갱신 (FRED 데이터는 자주 안 변함)
def load_all_data():
    data = {}
    
    # [A] FRED 데이터 (유동성) - URL에서 직접 읽기
    fred_urls = {
        'fed': 'WALCL', 'rrp': 'RRPONTSYD', 'tga': 'WTREGEN', 
        'credit': 'TOTBKCR', 'ecb': 'ECBASSETSW', 'boj': 'JPNASSETS', 
        'eur_usd': 'DEXUSEU', 'usd_jpy': 'DEXJPUS'
    }
    
    for key, code in fred_urls.items():
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"
            # 스트림릿 클라우드에서 바로 읽기
            df = pd.read_csv(url, index_col=0, parse_dates=True)
            df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors='coerce')
            data[key] = df.resample('D').interpolate(method='linear')
        except:
            data[key] = None

    # [B] 비트코인/도지 (Upbit - 가장 안정적)
    def get_upbit(market):
        try:
            url = f"https://api.upbit.com/v1/candles/days?market={market}&count=2000" # 넉넉하게
            r = requests.get(url).json()
            df = pd.DataFrame(r)
            df['Date'] = pd.to_datetime(df['candle_date_time_utc'])
            df['Price'] = df['trade_price'].astype(float)
            return df.set_index('Date').sort_index()[['Price']]
        except: return None

    data['btc'] = get_upbit("USDT-BTC")
    data['doge'] = get_upbit("USDT-DOGE")

    # [C] 나스닥 (Yahoo - 실패 시 무시)
    try:
        nasdaq = yf.download("^IXIC", start="2018-01-01", progress=False)
        if isinstance(nasdaq.columns, pd.MultiIndex):
            data['nasdaq'] = nasdaq.xs('Close', axis=1, level=0)["^IXIC"]
        else:
            data['nasdaq'] = nasdaq['Close']
        # 시간대 제거
        data['nasdaq'].index = data['nasdaq'].index.tz_localize(None)
    except:
        data['nasdaq'] = None

    # [D] 난이도 (JSON 파일)
    try:
        with open('difficulty (1).json', 'r') as f:
            d_json = json.load(f)['difficulty']
        df_d = pd.DataFrame(d_json)
        df_d['Date'] = pd.to_datetime(df_d['x'], unit='ms')
        data['diff'] = df_d.set_index('Date').sort_index()['y']
    except:
        data['diff'] = None
        
    return data

# 데이터 로딩
with st.spinner('전 세계 중앙은행 데이터를 수집 중입니다...'):
    dfs = load_all_data()

if dfs['btc'] is None:
    st.error("비트코인 데이터를 가져오지 못했습니다.")
    st.stop()

# ------------------------------------------------------------------------------
# 3. 데이터 가공 및 지표 계산 (선생님의 로직 완벽 이식)
# ------------------------------------------------------------------------------
# 기준 데이터프레임 생성 (주간 단위)
df = dfs['fed'].resample('W-WED').last()
df.columns = ['Fed']

# 데이터 합치기
for k in ['rrp', 'tga', 'credit', 'ecb', 'boj', 'eur_usd', 'usd_jpy']:
    if dfs.get(k) is not None:
        # 환율 데이터는 평균, 나머지는 기말 값
        method = 'mean' if 'usd' in k else 'last'
        r = dfs[k].resample('W-WED').agg(method)
        r.columns = [k.upper()]
        df = df.join(r, how='left')

df = df.fillna(method='ffill').fillna(method='bfill')

# [핵심] G3 유동성 계산
try:
    fed_t = df['Fed'] / 1_000_000
    ecb_t = (df['ECB'] * df['EUR_USD']) / 1_000_000
    boj_t = (df['BOJ'] * 0.0001) / df['USD_JPY']
    df['G3_Tril'] = fed_t + ecb_t + boj_t
    df['G3_YoY'] = df['G3_Tril'].pct_change(52) * 100 # 전년 대비 증감률
except:
    df['G3_YoY'] = 0
    df['G3_Tril'] = 0

# 순유동성 (Fed Net Liquidity)
try:
    tga = df['TGA']
    rrp = df['RRP']
    df['Fed_Net_Tril'] = (df['Fed'] - tga - (rrp * 1000)) / 1_000_000
except:
    df['Fed_Net_Tril'] = 0

# 신용 창출 (Credit)
if 'CREDIT' in df.columns:
    df['Credit_YoY'] = df['CREDIT'].pct_change(52) * 100
else:
    df['Credit_YoY'] = 0

# [원가 계산]
df_cost = pd.DataFrame(index=dfs['btc'].index)
if dfs['diff'] is not None:
    d_daily = dfs['diff'].resample('D').interpolate()
    # 인덱스 맞추기
    common_idx = df_cost.index.intersection(d_daily.index)
    df_cost.loc[common_idx, 'Difficulty'] = d_daily.loc[common_idx]
    
    def get_reward(d):
        if d < pd.Timestamp('2024-04-20'): return 6.25
        else: return 3.125
    
    df_cost['Reward'] = [get_reward(d) for d in df_cost.index]
    df_cost['Cost_Raw'] = df_cost['Difficulty'] / df_cost['Reward']
    
    # Calibration
    try:
        common = dfs['btc'].join(df_cost['Cost_Raw'])
        sub = common[(common.index>='2022-11-01')&(common.index<='2023-01-31')]
        k = (sub['Price']/sub['Cost_Raw']).min() if not sub.empty else 0.00000008
    except: k = 0.00000008
    
    df_cost['Mining_Floor'] = df_cost['Cost_Raw'] * k

# [데이터 시프트 -90일]
btc_s = dfs['btc'].copy(); btc_s.index -= pd.Timedelta(days=90)
cost_s = df_cost[['Mining_Floor']].copy(); cost_s.index -= pd.Timedelta(days=90)
doge_s = dfs['doge'].copy(); doge_s.index -= pd.Timedelta(days=90) if dfs['doge'] is not None else None
if dfs.get('nasdaq') is not None:
    nasdaq_s = dfs['nasdaq'].copy(); nasdaq_s.index -= pd.Timedelta(days=90)
else: nasdaq_s = None


# ------------------------------------------------------------------------------
# 4. 시각화 (Matplotlib -> Streamlit)
# ------------------------------------------------------------------------------
st.markdown("### 📊 The Grand Master Chart (Mobile Optimized)")

# 모바일에서도 잘 보이게 사이즈 조정
fig, ax1 = plt.subplots(figsize=(12, 8)) # 비율 조정
plt.style.use('dark_background') # 다크모드 강제
ax1.set_facecolor('black')
fig.patch.set_facecolor('black')

# 날짜 필터 (2023년부터)
start_dt = '2023-01-01'
d_view = df[df.index >= start_dt]

# [Axis 1] G3 Growth (노란색 영역)
ax1.plot(d_view.index, d_view['G3_YoY'], c='#FFD700', lw=2, label='G3 Liquidity', zorder=1)
ax1.fill_between(d_view.index, d_view['G3_YoY'], 0, where=(d_view['G3_YoY']>=0), color='#FFD700', alpha=0.15)
ax1.set_ylabel('Liquidity Growth %', color='#FFD700')
ax1.tick_params(axis='y', labelcolor='#FFD700')
ax1.set_ylim(-15, 30)
ax1.grid(True, axis='x', alpha=0.2)

# [Axis 2] BTC & Cost (오른쪽 축 1)
ax2 = ax1.twinx()
b_view = btc_s[btc_s.index >= start_dt]
c_view = cost_s[cost_s.index >= start_dt]

# 겹치는 기간만 찾기
idx = b_view.index.intersection(c_view.index)
ax2.plot(b_view.index, b_view['Price'], c='white', lw=2, label='BTC (-90d)', zorder=5)
ax2.plot(c_view.index, c_view['Mining_Floor'], c='red', ls='--', lw=1.5, label='Cost Floor', zorder=4)

# Capitulation Gap 채우기
if not idx.empty:
    b_vals = b_view.loc[idx]['Price']
    c_vals = c_view.loc[idx]['Mining_Floor']
    ax2.fill_between(idx, c_vals, b_vals, where=(b_vals < c_vals), color='red', alpha=0.3, hatch='///')

ax2.set_yscale('log')
ax2.set_ylabel('BTC Price (Log)', color='white')

# [Axis 3] Nasdaq (오른쪽 축 2 - 분홍색)
if nasdaq_s is not None:
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('axes', 1.15)) # 축 위치 이동
    n_view = nasdaq_s[nasdaq_s.index >= start_dt]
    ax3.plot(n_view.index, n_view, c='#D62780', lw=1.5, label='Nasdaq (-90d)', alpha=0.8)
    ax3.set_ylabel('Nasdaq', color='#D62780')
    ax3.tick_params(axis='y', labelcolor='#D62780')

# [Axis 4] Doge (오른쪽 축 3 - 주황색)
if doge_s is not None:
    ax4 = ax1.twinx()
    ax4.spines['right'].set_position(('axes', 1.30)) # 축 위치 이동
    dg_view = doge_s[doge_s.index >= start_dt]
    ax4.plot(dg_view.index, dg_view['Price'], c='#FFA500', lw=1.5, label='DOGE (-90d)', alpha=0.8)
    ax4.set_ylabel('DOGE', color='#FFA500')
    ax4.tick_params(axis='y', labelcolor='#FFA500')
    ax4.set_yscale('log')

# 타이틀 및 레이아웃
plt.title('Global Liquidity vs Asset Prices (Shifted -90d)', color='white', pad=20)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))

# Streamlit에 그림 출력
st.pyplot(fig)

# ------------------------------------------------------------------------------
# 5. 하단 데이터 진단
# ------------------------------------------------------------------------------
st.subheader("📋 핵심 지표 요약 (Latest)")
c1, c2, c3, c4 = st.columns(4)

last_g3 = d_view['G3_YoY'].iloc[-1]
c1.metric("G3 유동성 증감", f"{last_g3:.2f}%", delta_color="normal")

last_btc = dfs['btc']['Price'].iloc[-1]
c2.metric("BTC 현재가", f"${last_btc:,.0f}")

if not c_view.empty:
    last_cost = c_view['Mining_Floor'].iloc[-1]
    gap = (last_btc/last_cost - 1)*100
    c3.metric("채굴 원가", f"${last_cost:,.0f}", f"{gap:.2f}%")

if dfs.get('nasdaq') is not None:
    last_nd = dfs['nasdaq'].iloc[-1]
    c4.metric("Nasdaq", f"{last_nd:,.0f}")
