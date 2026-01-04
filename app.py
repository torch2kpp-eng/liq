import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import json
import io
import warnings
import time
import ccxt
from datetime import date

warnings.filterwarnings("ignore")

st.set_page_config(page_title="GM Terminal Final", layout="wide")
st.title("🏛️ Grand Master: Multi-Axis Final")
st.caption("Ver 9.7 | yfinance 대신 ccxt + Bithumb 사용 | 진짜 KRW 가격 | 2017년 말~")

@st.cache_data(ttl=3600, show_spinner="거래소 데이터 불러오는 중...")
def fetch_data_final():
    d = {}
    
    # ── ccxt Bithumb 설정 ─────────────────────────────────────────────
    exchange = ccxt.bithumb({
        'enableRateLimit': True,
    })
    
    def fetch_all_ohlcv(symbol, timeframe='1d', since_year=2010, limit_per_call=500):
        """긴 기간 데이터를 여러 번 호출하여 모두 가져오기"""
        all_data = []
        since = exchange.parse8601(f'{since_year}-01-01T00:00:00Z')
        
        while True:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit_per_call)
                if not ohlcv:
                    break
                all_data.extend(ohlcv)
                # 마지막 캔들 timestamp + 1ms
                since = ohlcv[-1][0] + 1
                time.sleep(exchange.rateLimit / 1000)  # rate limit 방지
            except Exception as e:
                st.warning(f"{symbol} 데이터 가져오기 오류: {str(e)}")
                break
        
        if not all_data:
            return pd.Series(dtype=float)
        
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df['close'].tz_localize(None)

    # BTC/KRW (Bithumb는 2017년 말~ 데이터 존재)
    d['btc'] = fetch_all_ohlcv('BTC/KRW', since_year=2017)
    
    # DOGE/KRW (있을 경우 / 없으면 빈 시리즈)
    try:
        d['doge'] = fetch_all_ohlcv('DOGE/KRW', since_year=2018)
    except:
        st.info("DOGE/KRW 페어가 현재 Bithumb에 없거나 데이터를 불러올 수 없습니다.")
        d['doge'] = pd.Series(dtype=float)

    # ── FRED Liquidity (기존) ────────────────────────────────────────
    def get_fred(series_id):
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
            r = requests.get(url, timeout=12)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text), index_col=0, parse_dates=True)
            return df.squeeze().resample('D').interpolate(method='time').tz_localize(None)
        except Exception as e:
            st.warning(f"FRED {series_id} 실패: {e}")
            return pd.Series(dtype=float)
    
    d['fed'] = get_fred('WALCL')
    d['tga'] = get_fred('WTREGEN')
    d['rrp'] = get_fred('RRPONTSYD')

    # ── Nasdaq (yfinance 유지, 실패 시 빈 값) ─────────────────────────
    try:
        import yfinance as yf
        ns = yf.download("^IXIC", period="max", progress=False)
        close = ns['Close'] if 'Close' in ns.columns else ns
        d['nasdaq'] = close.tz_localize(None) if not close.empty else pd.Series(dtype=float)
    except:
        d['nasdaq'] = pd.Series(dtype=float)

    # ── Difficulty JSON (기존) ────────────────────────────────────────
    try:
        with open('difficulty (1).json', 'r', encoding='utf-8') as f:
            js = json.load(f)['difficulty']
        df_js = pd.DataFrame(js)
        df_js['Date'] = pd.to_datetime(df_js['x'], unit='ms').dt.tz_localize(None)
        d['diff'] = df_js.set_index('Date').sort_index()['y']
    except Exception as e:
        st.warning(f"Difficulty JSON 실패: {e}")
        d['diff'] = pd.Series(dtype=float)

    return d


# ────────────────────────────────────────────────────────────────
raw = fetch_data_final()

if not raw.get('btc', pd.Series()).empty and isinstance(raw['btc'].index, pd.DatetimeIndex):
    # Liquidity 계산 (기존 로직 유지)
    df_liq = raw['fed'].resample('W-WED').last().to_frame(name='Fed')
    if not raw['tga'].empty:
        df_liq['TGA'] = raw['tga'].resample('W-WED').mean()
    if not raw['rrp'].empty:
        df_liq['RRP'] = raw['rrp'].resample('W-WED').mean()
    
    df_liq = df_liq.fillna(method='ffill')
    df_liq['Net_Tril'] = (
        df_liq['Fed'] / 1000 -
        df_liq.get('TGA', pd.Series(0, index=df_liq.index)) / 1000 -
        df_liq.get('RRP', pd.Series(0, index=df_liq.index)) / 1_000_000
    )
    df_liq['YoY'] = df_liq['Net_Tril'].pct_change(52) * 100

    # Mining Cost Floor
    df_c = pd.DataFrame(index=raw['btc'].index)
    if not raw['diff'].empty:
        df_c['diff'] = raw['diff'].reindex(df_c.index).interpolate(method='time')
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

    # -90일 shift
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

    start_viz_dt = pd.to_datetime('2017-01-01')  # Bithumb 데이터 시작 고려

    def safe_filter(s, start_dt):
        if s.empty or not isinstance(s.index, pd.DatetimeIndex):
            return pd.Series(dtype=float)
        return s[s.index >= start_dt]

    liq_v = df_liq[df_liq.index >= start_viz_dt]['YoY'] if not df_liq.empty else pd.Series(dtype=float)
    btc_v = safe_filter(btc_s, start_viz_dt)
    fl_v = safe_filter(floor_s, start_viz_dt)
    nd_v = safe_filter(nasdaq_s, start_viz_dt)
    dg_v = safe_filter(doge_s, start_viz_dt)

    # BTC Y축 동적 범위
    if not btc_v.empty:
        btc_max = btc_v.max()
        btc_min_dynamic = max(btc_max * 0.05, 10_000_000)
        btc_max_dynamic = btc_max * 1.2
    else:
        btc_min_dynamic = 10_000_000
        btc_max_dynamic = 500_000_000_000

    fig = go.Figure(
        layout=go.Layout(
            template="plotly_dark",
            height=800,
            xaxis=dict(domain=[0.0, 0.88], showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
            
            yaxis=dict(title="Liquidity YoY %", title_font_color="#FFD700", tickfont_color="#FFD700", range=[-50, 80]),
            
            yaxis2=dict(title="BTC (KRW)", title_font_color="#00FFEE", tickfont_color="#00FFEE",
                        overlaying="y", side="right", position=0.88, type="linear",
                        range=[btc_min_dynamic, btc_max_dynamic], tickformat=",", showgrid=False),
            
            yaxis3=dict(title="Nasdaq", title_font_color="#D62780", tickfont_color="#D62780",
                        overlaying="y", side="right", anchor="free", position=0.96, tickformat=","),
            
            yaxis4=dict(title="DOGE (Log)", title_font_color="orange", tickfont_color="orange",
                        overlaying="y", side="right", anchor="free", position=1.0, type="log"),
            
            legend=dict(orientation="h", y=1.15, x=0.01, bgcolor="rgba(0,0,0,0)"),
            hovermode="x unified",
            margin=dict(l=60, r=140, t=100, b=60)
        )
    )

    fig.add_trace(go.Scatter(x=liq_v.index, y=liq_v, name="Liquidity YoY %", line_color='#FFD700', width=3,
                             fill='tozeroy', fillcolor='rgba(255,215,0,0.15)', yaxis='y'))
    
    if not btc_v.empty:
        fig.add_trace(go.Scatter(x=btc_v.index, y=btc_v, name="BTC (-90d)", line_color='#00FFEE', width=4.5, yaxis='y2'))
    
    if not fl_v.empty:
        fig.add_trace(go.Scatter(x=fl_v.index, y=fl_v, name="Mining Cost Floor", line_color='red', width=2, dash='dot', yaxis='y2'))
    
    if not nd_v.empty:
        fig.add_trace(go.Scatter(x=nd_v.index, y=nd_v, name="Nasdaq (-90d)", line_color='#D62780', width=2, yaxis='y3'))
    
    if not dg_v.empty:
        fig.add_trace(go.Scatter(x=dg_v.index, y=dg_v, name="DOGE (-90d)", line_color='orange', width=2, yaxis='y4'))

    st.plotly_chart(fig, use_container_width=True)
    st.success("✅ Bithumb 실거래 KRW 가격 데이터 로드 완료!")

else:
    st.error("❌ BTC 데이터 로드 실패 (Bithumb 연결 문제일 가능성 높음)")
    st.info("• requirements.txt에 ccxt>=4.0.0 추가했는지 확인\n• 인터넷 연결 상태 점검\n• Streamlit Cloud 재배포 시도")
