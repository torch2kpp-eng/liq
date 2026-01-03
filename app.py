import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import yfinance as yf
import json
import numpy as np
import io
import warnings
from datetime import date

# 1. 환경 설정
warnings.filterwarnings("ignore")
st.set_page_config(page_title="GM Terminal Final", layout="wide")

st.title("🏛️ Grand Master: Multi-Axis Final")
st.caption("Ver 8.7 | Upbit KRW 마켓 적용, RRP 단위 및 Halving 비교 수정")

# 2. 데이터 수집 (캐시 적용)
@st.cache_data(ttl=3600)
def fetch_data_final():
    d = {}
    
    # [A] Upbit (KRW 마켓으로 변경)
    def get_upbit(symbol):
        try:
            url = f"https://api.upbit.com/v1/candles/days?market={symbol}&count=1000"
            r = requests.get(url, timeout=10).json()
            if not r:  # 빈 리스트 반환 시
                return pd.Series(dtype=float)
            df = pd.DataFrame(r)
            df['Date'] = pd.to_datetime(df['candle_date_time_utc']).dt.tz_localize(None)
            return df.set_index('Date').sort_index()['trade_price'].astype(float)
        except Exception as e:
            st.warning(f"Upbit {symbol} 데이터 로드 실패: {e}")
            return pd.Series(dtype=float)

    d['btc'] = get_upbit("KRW-BTC")
    d['doge'] = get_upbit("KRW-DOGE")

    # [B] FRED
    def get_fred(series_id):
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
            r = requests.get(url, timeout=10)
            df = pd.read_csv(io.StringIO(r.text), index_col=0, parse_dates=True)
            return df.squeeze().resample('D').interpolate().tz_localize(None)
        except Exception as e:
            st.warning(f"FRED {series_id} 데이터 로드 실패: {e}")
            return pd.Series(dtype=float)

    d['fed'] = get_fred('WALCL')      # Fed Balance Sheet (billions)
    d['tga'] = get_fred('WTREGEN')    # Treasury General Account (billions)
    d['rrp'] = get_fred('RRPONTSYD')  # Reverse Repo (millions)

    # [C] Nasdaq
    try:
        ns = yf.download("^IXIC", period="5y", progress=False, auto_adjust=True)
        close = ns['Close'] if 'Close' in ns.columns else ns.xs('Close', axis=1, level=0)
        d['nasdaq'] = close.tz_localize(None)
    except Exception:
        d['nasdaq'] = pd.Series(dtype=float)

    # [D] Bitcoin Difficulty (로컬 JSON 파일)
    try:
        with open('difficulty (1).json', 'r') as f:
            js = json.load(f)['difficulty']
        df_js = pd.DataFrame(js)
        df_js['Date'] = pd.to_datetime(df_js['x'], unit='ms').dt.tz_localize(None)
        d['diff'] = df_js.set_index('Date').sort_index()['y']
    except Exception as e:
        st.warning(f"Difficulty JSON 로드 실패: {e}")
        d['diff'] = pd.Series(dtype=float)
    
    return d

raw = fetch_data_final()

# 3. 데이터 가공
if not raw['btc'].empty:
    # Liquidity (주간 수요일 기준)
    df_liq = raw['fed'].resample('W-WED').last().to_frame(name='Fed')
    
    if not raw['tga'].empty:
        df_liq['TGA'] = raw['tga'].resample('W-WED').mean()
    if not raw['rrp'].empty:
        df_liq['RRP'] = raw['rrp'].resample('W-WED').mean()

    df_liq = df_liq.fillna(method='ffill')

    # Net Liquidity (trillions of USD) - RRP는 millions → billions로 변환 후 처리
    # Fed & TGA: billions → trillions은 /1000
    # RRP: millions → trillions은 /1_000_000
    df_liq['Net_Tril'] = (
        df_liq['Fed'] / 1000 -
        (df_liq.get('TGA', 0) / 1000) -
        (df_liq.get('RRP', 0) / 1_000_000)
    )
    df_liq['YoY'] = df_liq['Net_Tril'].pct_change(52) * 100

    # Mining Cost Floor
    df_c = pd.DataFrame(index=raw['btc'].index)
    if not raw['diff'].empty:
        df_c['diff'] = raw['diff'].reindex(df_c.index).interpolate()

        # Halving 보상 정확 계산 (2024-04-20 이후 3.125 BTC)
        halving_date = date(2024, 4, 20)
        df_c['reward'] = df_c.index.map(lambda x: 3.125 if x.date() >= halving_date else 6.25)

        df_c['cost_raw'] = df_c['diff'] / df_c['reward']

        # 2022-11-01 ~ 2023-01-31 기간의 최소 BTC/cost_raw 비율로 k 계산
        sub = pd.concat([raw['btc'], df_c['cost_raw']], axis=1).dropna()
        sub.columns = ['btc', 'cost_raw']
        target = sub[(sub.index >= '2022-11-01') & (sub.index <= '2023-01-31')]
        
        if not target.empty:
            k = (target['btc'] / target['cost_raw']).min()
        else:
            k = 0.0000001  # fallback (거의 0에 가까운 값)
            st.info("Mining floor 기준 기간 데이터 부족 → 기본값 사용")

        df_c['floor'] = df_c['cost_raw'] * k
    else:
        df_c['floor'] = pd.Series(dtype=float)

    # -90일 시프트 함수
    def shift_90(s):
        if s.empty:
            return pd.Series(dtype=float)
        new_s = s.copy()
        new_s.index = new_s.index - pd.Timedelta(days=90)
        return new_s

    btc_s = shift_90(raw['btc'])
    floor_s = shift_90(df_c.get('floor', pd.Series(dtype=float)))
    nasdaq_s = shift_90(raw['nasdaq'])
    doge_s = shift_90(raw['doge'])

    # 4. 차트 생성
    st.subheader("📊 Grand Master Integrated Strategy Chart")
    
    start_viz = '2023-01-01'
    
    liq_v = df_liq[df_liq.index >= start_viz]['YoY']
    btc_v = btc_s[btc_s.index >= start_viz]
    fl_v = floor_s[floor_s.index >= start_viz]
    nd_v = nasdaq_s[nasdaq_s.index >= start_viz]
    dg_v = doge_s[doge_s.index >= start_viz]

    fig = go.Figure(
        layout=go.Layout(
            template="plotly_dark",
            height=720,
            xaxis=dict(domain=[0.0, 0.85], showgrid=False),
            yaxis=dict(
                title=dict(text="Liquidity YoY %", font=dict(color="#FFD700")),
                tickfont=dict(color="#FFD700"),
                range=[-30, 50]
            ),
            yaxis2=dict(
                title=dict(text="BTC (Log)", font=dict(color="white")),
                tickfont=dict(color="white"),
                overlaying="y",
                side="right",
                type="log",
                position=0.85
            ),
            yaxis3=dict(
                title=dict(text="Nasdaq", font=dict(color="#D62780")),
                tickfont=dict(color="#D62780"),
                overlaying="y",
                side="right",
                anchor="free",
                position=0.92
            ),
            yaxis4=dict(
                title=dict(text="DOGE (Log)", font=dict(color="orange")),
                tickfont=dict(color="orange"),
                overlaying="y",
                side="right",
                anchor="free",
                position=1.0,
                type="log"
            ),
            legend=dict(orientation="h", y=1.12, x=0.01, bgcolor="rgba(0,0,0,0)"),
            hovermode="x unified",
            margin=dict(l=50, r=120, t=80, b=50)
        )
    )

    # Traces
    fig.add_trace(go.Scatter(
        x=liq_v.index, y=liq_v,
        name="Liquidity YoY %", line=dict(color='#FFD700', width=3),
        fill='tozeroy', fillcolor='rgba(255, 215, 0, 0.15)', yaxis='y'
    ))

    fig.add_trace(go.Scatter(
        x=btc_v.index, y=btc_v,
        name="BTC (-90d)", line=dict(color='white', width=3), yaxis='y2'
    ))

    if not fl_v.empty:
        fig.add_trace(go.Scatter(
            x=fl_v.index, y=fl_v,
            name="Mining Cost Floor", line=dict(color='red', width=2, dash='dot'), yaxis='y2'
        ))

    if not nd_v.empty:
        fig.add_trace(go.Scatter(
            x=nd_v.index, y=nd_v,
            name="Nasdaq (-90d)", line=dict(color='#D62780', width=2), yaxis='y3'
        ))

    if not dg_v.empty:
        fig.add_trace(go.Scatter(
            x=dg_v.index, y=dg_v,
            name="DOGE (-90d)", line=dict(color='orange', width=2), yaxis='y4'
        ))

    st.plotly_chart(fig, use_container_width=True)
    st.success("✅ 시스템 정상 가동: 모든 데이터 및 차트 로딩 완료")

else:
    st.error("❌ 주요 데이터(BTC) 로드 실패. 네트워크 또는 API 상태를 확인 후 다시 시도해주세요.")
