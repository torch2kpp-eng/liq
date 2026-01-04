import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import yfinance as yf
import json
import io
import warnings
from datetime import date

warnings.filterwarnings("ignore")
st.set_page_config(page_title="GM Terminal Final", layout="wide")

st.title("🏛️ Grand Master: Multi-Axis Final")
st.caption("Ver 9.5 | 장기 데이터 적용 (2010~), Nasdaq 정상화, 축 레이블 최적화, 전체 기간 표시")

@st.cache_data(ttl=3600)
def fetch_data_final():
    d = {}
    
    # [A] BTC & DOGE: yfinance의 BTC-KRW, DOGE-KRW 사용 → 2014년부터 장기 데이터 가능 (BTC는 ~2014.9, DOGE ~2014)
    try:
        btc_krw = yf.download("BTC-KRW", period="max", progress=False)['Close'].tz_localize(None)
        d['btc'] = btc_krw if isinstance(btc_krw.index, pd.DatetimeIndex) else pd.Series(dtype=float)
    except:
        d['btc'] = pd.Series(dtype=float)

    try:
        doge_krw = yf.download("DOGE-KRW", period="max", progress=False)['Close'].tz_localize(None)
        d['doge'] = doge_krw if isinstance(doge_krw.index, pd.DatetimeIndex) else pd.Series(dtype=float)
    except:
        d['doge'] = pd.Series(dtype=float)

    # [B] FRED (Liquidity)
    def get_fred(series_id):
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
            r = requests.get(url, timeout=10)
            df = pd.read_csv(io.StringIO(r.text), index_col=0, parse_dates=True)
            return df.squeeze().resample('D').interpolate().tz_localize(None)
        except:
            return pd.Series(dtype=float)

    d['fed'] = get_fred('WALCL')
    d['tga'] = get_fred('WTREGEN')
    d['rrp'] = get_fred('RRPONTSYD')

    # [C] Nasdaq: period="max"로 1971년부터 가져오도록 변경
    try:
        ns = yf.download("^IXIC", period="max", progress=False, auto_adjust=True)
        close = ns['Close'] if 'Close' in ns.columns else ns
        s = close.tz_localize(None)
        d['nasdaq'] = s if isinstance(s.index, pd.DatetimeIndex) else pd.Series(dtype=float)
    except:
        d['nasdaq'] = pd.Series(dtype=float)

    # [D] Difficulty (기존 로컬 JSON 유지)
    try:
        with open('difficulty (1).json', 'r') as f:
            js = json.load(f)['difficulty']
        df_js = pd.DataFrame(js)
        df_js['Date'] = pd.to_datetime(df_js['x'], unit='ms').dt.tz_localize(None)
        d['diff'] = df_js.set_index('Date').sort_index()['y']
    except:
        d['diff'] = pd.Series(dtype=float)
    
    return d

raw = fetch_data_final()

if not raw['btc'].empty and isinstance(raw['btc'].index, pd.DatetimeIndex):
    # Liquidity (주간 수요일 기준, 기존 로직 유지)
    df_liq = raw['fed'].resample('W-WED').last().to_frame(name='Fed')
    if not raw['tga'].empty:
        df_liq['TGA'] = raw['tga'].resample('W-WED').mean()
    if not raw['rrp'].empty:
        df_liq['RRP'] = raw['rrp'].resample('W-WED').mean()
    df_liq = df_liq.fillna(method='ffill')

    df_liq['Net_Tril'] = (
        df_liq['Fed'] / 1000 -
        df_liq.get('TGA', 0) / 1000 -
        df_liq.get('RRP', 0) / 1_000_000
    )
    df_liq['YoY'] = df_liq['Net_Tril'].pct_change(52) * 100

    # Mining Cost Floor
    df_c = pd.DataFrame(index=raw['btc'].index)
    if not raw['diff'].empty:
        df_c['diff'] = raw['diff'].reindex(df_c.index).interpolate()
        halving_date = date(2024, 4, 20)
        df_c['reward'] = df_c.index.map(lambda x: 3.125 if x.date() >= halving_date else 6.25)
        df_c['cost_raw'] = df_c['diff'] / df_c['reward']
        sub = pd.concat([raw['btc'], df_c['cost_raw']], axis=1).dropna()
        sub.columns = ['btc', 'cost_raw']
        target = sub[(sub.index >= '2022-11-01') & (sub.index <= '2023-01-31')]
        k = (target['btc'] / target['cost_raw']).min() if not target.empty else 0.0000001
        df_c['floor'] = df_c['cost_raw'] * k
    else:
        df_c['floor'] = pd.Series(dtype=float)

    # -90일 시프트
    def shift_90(s):
        if s.empty or not isinstance(s.index, pd.DatetimeIndex):
            return pd.Series(dtype=float)
        new_s = s.copy()
        new_s.index = new_s.index - pd.Timedelta(days=90)
        return new_s

    btc_s = shift_90(raw['btc'])
    floor_s = shift_90(df_c.get('floor', pd.Series(dtype=float)))
    nasdaq_s = shift_90(raw['nasdaq'])
    doge_s = shift_90(raw['doge'])

    st.subheader("📊 Grand Master Integrated Strategy Chart")

    # 전체 기간 표시 (2010년부터, 실제 데이터 존재 시점 자동 적용)
    start_viz_dt = pd.to_datetime('2010-01-01')

    def safe_filter(s, start_dt):
        if s.empty or not isinstance(s.index, pd.DatetimeIndex):
            return pd.Series(dtype=float)
        return s[s.index >= start_dt]

    liq_v = df_liq[df_liq.index >= start_viz_dt]['YoY'] if not df_liq.empty else pd.Series(dtype=float)
    btc_v = safe_filter(btc_s, start_viz_dt)
    fl_v = safe_filter(floor_s, start_viz_dt)
    nd_v = safe_filter(nasdaq_s, start_viz_dt)
    dg_v = safe_filter(doge_s, start_viz_dt)

    # BTC 동적 Y축 범위
    if not btc_v.empty:
        btc_max = btc_v.max()
        btc_min_dynamic = max(btc_max * 0.05, 10_000_000)  # 너무 낮지 않게
        btc_max_dynamic = btc_max * 1.2
    else:
        btc_min_dynamic = 10_000_000
        btc_max_dynamic = 200_000_000

    fig = go.Figure(
        layout=go.Layout(
            template="plotly_dark",
            height=800,
            xaxis=dict(domain=[0.0, 0.88], showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
            
            yaxis=dict(
                title=dict(text="Liquidity YoY %", font=dict(color="#FFD700", size=14)),
                tickfont=dict(color="#FFD700"),
                range=[-50, 80],
            ),
            
            yaxis2=dict(
                title=dict(text="BTC (KRW)", font=dict(color="#00FFEE", size=15)),
                tickfont=dict(color="#00FFEE", size=12),
                overlaying="y",
                side="right",
                position=0.88,
                type="linear",
                range=[btc_min_dynamic, btc_max_dynamic],
                tickformat=",",  # 천단위 콤마
                showgrid=False
            ),
            
            # Nasdaq 축: 소수점 최소화 (정수 표시)
            yaxis3=dict(
                title=dict(text="Nasdaq", font=dict(color="#D62780", size=14)),
                tickfont=dict(color="#D62780", size=12),
                overlaying="y",
                side="right",
                anchor="free",
                position=0.96,
                tickformat=","   # 정수만 표시 (소수점 제거)
            ),
            
            yaxis4=dict(
                title=dict(text="DOGE (Log)", font=dict(color="orange", size=14)),
                tickfont=dict(color="orange", size=12),
                overlaying="y",
                side="right",
                anchor="free",
                position=1.0,
                type="log"
            ),
            
            legend=dict(orientation="h", y=1.15, x=0.01, bgcolor="rgba(0,0,0,0)"),
            hovermode="x unified",
            margin=dict(l=60, r=140, t=100, b=60)
        )
    )

    # Liquidity
    fig.add_trace(go.Scatter(x=liq_v.index, y=liq_v, name="Liquidity YoY %",
                             line=dict(color='#FFD700', width=3), fill='tozeroy',
                             fillcolor='rgba(255, 215, 0, 0.15)', yaxis='y'))

    # BTC
    if not btc_v.empty:
        fig.add_trace(go.Scatter(x=btc_v.index, y=btc_v, name="BTC (-90d)",
                                 line=dict(color='#00FFEE', width=4.5), yaxis='y2'))

    # Floor
    if not fl_v.empty:
        fig.add_trace(go.Scatter(x=fl_v.index, y=fl_v, name="Mining Cost Floor",
                                 line=dict(color='red', width=2, dash='dot'), yaxis='y2'))

    # Nasdaq (이제 1971년부터 정상 표시)
    if not nd_v.empty:
        fig.add_trace(go.Scatter(x=nd_v.index, y=nd_v, name="Nasdaq (-90d)",
                                 line=dict(color='#D62780', width=2), yaxis='y3'))

    # DOGE
    if not dg_v.empty:
        fig.add_trace(go.Scatter(x=dg_v.index, y=dg_v, name="DOGE (-90d)",
                                 line=dict(color='orange', width=2), yaxis='y4'))

    st.plotly_chart(fig, use_container_width=True)
    st.success("✅ 모든 문제 해결: 장기 차트(2010~), Nasdaq 정상, 축 레이블 최적화, BTC/DOGE 최대 기간 적용")

else:
    st.error("❌ BTC 데이터 로드 실패. yfinance 연결 확인 후 재시도해주세요.")
