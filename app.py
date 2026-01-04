import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import json
import io
import warnings
import time
import ccxt
import numpy as np
from datetime import date, timedelta

# 1. 환경 설정
warnings.filterwarnings("ignore")
st.set_page_config(page_title="GM Stress Test", layout="wide")

st.title("🏛️ Grand Master: Stress Test Simulator")
st.caption("Ver 19.0 | HY Spread 급등 시나리오 백테스트 | 위기 감지 유효성 검증")

# -----------------------------------------------------------
# [사이드바 설정]
# -----------------------------------------------------------
st.sidebar.header("⚙️ Control Panel")

is_mobile = st.sidebar.checkbox("📱 모바일 모드 (축 공간 최소화)", value=True)

# [NEW] 백테스트 설정
st.sidebar.markdown("---")
st.sidebar.subheader("📉 Crash Simulation")
spike_threshold = st.sidebar.slider(
    "위험 감지 민감도 (Daily Delta bps)", 
    min_value=5, max_value=50, value=15, step=1,
    help="하루에 스프레드가 이 값(bps) 이상 튀어 오르면 '위기'로 간주합니다. (권장: 10~20)"
)
look_forward_days = st.sidebar.slider(
    "반응 관찰 기간 (Days)",
    min_value=1, max_value=30, value=7,
    help="신호 발생 후 며칠 뒤의 가격 등락을 확인할까요?"
)

liq_option = st.sidebar.radio(
    "1. 유동성 지표 (Left Axis)",
    (
        "🇺🇸 Fed Net Liquidity (미국 실질 유동성)",
        "🏛️ G3 Central Bank Assets (본원통화 총량)",
        "🌍 Global M2 (실물 통화량: US+EU+JP)"
    ),
    index=2
)

st.sidebar.markdown("---")
st.sidebar.write("2. Time Shift (Days)")
shift_days = st.sidebar.number_input(
    "자산/지표 이동 (일)", min_value=-365, max_value=365, value=90, step=7
)

st.sidebar.markdown("---")
st.sidebar.write("3. 표시할 자산 (Right Axes)")

ASSETS_CONFIG = [
    {'id': 'hy_spread', 'name': '⚡ HY Spread', 'symbol': 'BAMLH0A0HYM2', 'source': 'fred', 'color': '#E040FB', 'type': 'risk', 'default': True},
    {'id': 'btc',    'name': 'BTC',    'symbol': 'BTC/KRW', 'source': 'bithumb', 'color': '#00FFEE', 'type': 'crypto', 'default': True},
    {'id': 'nasdaq', 'name': 'Nasdaq', 'symbol': 'IXIC', 'source': 'hybrid', 'color': '#D62780', 'type': 'index', 'default': False},
    {'id': 'gold',   'name': 'Gold',   'symbol': 'GC=F', 'source': 'hybrid_metal', 'color': '#FFD700', 'type': 'metal', 'default': False},
    {'id': 'eth',    'name': 'ETH',    'symbol': 'ETH/KRW', 'source': 'bithumb', 'color': '#627EEA', 'type': 'crypto', 'default': False},
]

selected_assets = {}
for asset in ASSETS_CONFIG:
    selected_assets[asset['id']] = st.sidebar.checkbox(f"{asset['name']}", value=asset['default'])

# -----------------------------------------------------------
# 데이터 수집
# -----------------------------------------------------------
def fetch_master_data_logic():
    d = {}
    meta_info = {}
    GLOBAL_START = time.time()
    MAX_EXECUTION_TIME = 30 
    
    START_YEAR = 2018 # [수정] 2019년 사례를 보기 위해 기간 확장
    headers = {'User-Agent': 'Mozilla/5.0'}

    def check_timeout(): return (time.time() - GLOBAL_START > MAX_EXECUTION_TIME)

    def get_fred(id):
        if check_timeout(): return pd.Series(dtype=float)
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"
            r = requests.get(url, headers=headers, timeout=5)
            df = pd.read_csv(io.StringIO(r.text), index_col=0, parse_dates=True)
            s = df.squeeze().apply(pd.to_numeric, errors='coerce')
            return s.resample('D').interpolate(method='time').tz_localize(None)
        except: return pd.Series(dtype=float)

    def get_yahoo(ticker):
        if check_timeout(): return pd.Series(dtype=float)
        try:
            import yfinance as yf
            df = yf.download(ticker, start=f"{START_YEAR}-01-01", progress=False, auto_adjust=True)
            if not df.empty:
                s = df['Close'] if 'Close' in df.columns else df.iloc[:,0]
                if isinstance(s, pd.DataFrame): s = s.iloc[:, 0]
                s = s.squeeze().tz_localize(None)
                if isinstance(s.index, pd.DatetimeIndex):
                    return s.resample('D').interpolate(method='time')
            return pd.Series(dtype=float)
        except: return pd.Series(dtype=float)

    def get_metal_hybrid(symbol):
        if check_timeout(): return pd.Series(dtype=float), "Timeout"
        data = get_yahoo(symbol)
        if not data.empty and len(data) > 10: return data, "Futures"
        backup = "GLD" if "GC" in symbol else "SLV"
        data_b = get_yahoo(backup)
        if not data_b.empty: return data_b, "ETF(Backup)"
        return pd.Series(dtype=float), "Fail"

    bithumb = ccxt.bithumb({'enableRateLimit': True, 'timeout': 3000})
    def fetch_bithumb(symbol_code):
        if check_timeout(): return pd.Series(dtype=float)
        all_data = []
        try:
            since = bithumb.parse8601(f'{START_YEAR}-01-01T00:00:00Z')
            for _ in range(12): 
                if check_timeout(): break
                ohlcv = bithumb.fetch_ohlcv(symbol_code, '1d', since=since, limit=1000)
                if not ohlcv: break
                all_data.extend(ohlcv)
                last_ts = ohlcv[-1][0]
                if last_ts >= (time.time() * 1000) - 86400000: break
                since = last_ts + 1
                time.sleep(0.05)
        except: pass
        if not all_data: return pd.Series(dtype=float)
        df = pd.DataFrame(all_data, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df.drop_duplicates('timestamp').set_index('timestamp')['close'].tz_localize(None)

    status_text = st.empty()
    status_text.text("📡 Initializing Data (2018~)...")
    
    fred_ids = {
        'fed': 'WALCL', 'tga': 'WTREGEN', 'rrp': 'RRPONTSYD',
        'ecb': 'ECBASSETSW', 'boj': 'JPNASSETS',
        'm2_us': 'M2SL', 'm3_eu': 'MABMM301EZM189S', 'm3_jp': 'MABMM301JPM189S',
        'eur_usd': 'DEXUSEU', 'usd_jpy': 'DEXJPUS',
        'nasdaq_fred': 'NASDAQCOM'
    }
    
    for k, v in fred_ids.items():
        if check_timeout(): break
        d[k] = get_fred(v)

    if not d.get('nasdaq_fred', pd.Series()).empty: d['nasdaq'] = d['nasdaq_fred']
    else: d['nasdaq'] = get_yahoo("^IXIC")

    active_ids = [a['id'] for a in ASSETS_CONFIG if selected_assets[a['id']]]
    for asset in ASSETS_CONFIG:
        if asset['id'] not in active_ids: continue
        if check_timeout(): continue
        if asset['id'] == 'nasdaq': continue
        
        if asset['source'] == 'fred': d[asset['id']] = get_fred(asset['symbol'])
        elif asset['source'] == 'hybrid_metal':
            data, src = get_metal_hybrid(asset['symbol'])
            d[asset['id']] = data
            meta_info[asset['id']] = src
        elif asset['source'] == 'yahoo': d[asset['id']] = get_yahoo(asset['symbol'])
        elif asset['source'] == 'bithumb': d[asset['id']] = fetch_bithumb(asset['symbol'])
        
    status_text.empty()
    return d, meta_info

raw, meta = fetch_master_data_logic()

# -----------------------------------------------------------
# [NEW] Stress Test Logic
# -----------------------------------------------------------
def run_stress_test(hy_series, btc_series, threshold_bps, look_forward):
    """
    HY Spread가 threshold_bps 이상 튄 날을 찾고, 그 후 BTC 가격 변화를 추적
    """
    try:
        # 데이터 동기화
        df = pd.concat([hy_series, btc_series], axis=1).dropna()
        df.columns = ['Spread', 'Price']
        
        # 일일 변동폭 (bps) 계산
        df['Spread_Chg_Bps'] = df['Spread'].diff() * 100
        
        # 감지된 날짜들 (Events)
        events = df[df['Spread_Chg_Bps'] >= threshold_bps].index
        
        results = []
        for date in events:
            # look_forward 일 후의 날짜 확인
            target_date = date + timedelta(days=look_forward)
            
            # target_date가 데이터 범위 내에 있는지 확인
            if target_date <= df.index[-1]:
                price_at_signal = df.loc[date]['Price']
                # target_date에 가장 가까운 미래 데이터 찾기 (주말 등 고려)
                future_data = df[df.index >= target_date]
                if not future_data.empty:
                    price_future = future_data.iloc[0]['Price']
                    price_chg_pct = (price_future - price_at_signal) / price_at_signal * 100
                    
                    # 판단 (Spread 급등 -> BTC 하락해야 정상)
                    # 하락했으면 "방어 성공(Success)", 올랐으면 "휩쏘(Whipsaw)"
                    outcome = "🛡️ 방어 성공" if price_chg_pct < 0 else "🎣 휩쏘 (False)"
                    
                    results.append({
                        "Date": date.strftime("%Y-%m-%d"),
                        "Spike (bps)": f"+{df.loc[date]['Spread_Chg_Bps']:.1f}",
                        "BTC Return": f"{price_chg_pct:+.2f}%",
                        "Outcome": outcome
                    })
        
        return pd.DataFrame(results).sort_values("Date", ascending=False)
        
    except Exception as e:
        return pd.DataFrame()

# -----------------------------------------------------------
# Main Logic
# -----------------------------------------------------------
try:
    if not raw.get('fed', pd.Series()).empty:
        # Macro & M2 Logic (Ver 18.4 Logic)
        base_idx = raw['fed'].resample('W-WED').last().index
        df_m = pd.DataFrame(index=base_idx)
        for k in raw:
            if k not in [a['id'] for a in ASSETS_CONFIG] and k != 'diff':
                try: df_m[k] = raw[k].reindex(df_m.index, method='ffill')
                except: continue
        df_m = df_m.fillna(method='ffill')

        # G3 & M2 Calc (Omitted for brevity, same as Ver 18.4)
        # ... (생략: 기존 완벽한 로직 유지) ...
        # (실제 코드에는 Ver 18.4의 로직이 그대로 들어갑니다)
        s_m2_us, s_m3_eu, s_m3_jp = df_m.get('m2_us'), df_m.get('m3_eu'), df_m.get('m3_jp')
        if s_m2_us is not None and s_m3_eu is not None and s_m3_jp is not None:
            m2_us = s_m2_us / 1000
            m3_eu = (s_m3_eu * df_m.get('eur_usd', 1)) / 1e12
            m3_jp = (s_m3_jp / df_m.get('usd_jpy', 1)) / 1e12
            global_m2_sum = m2_us + m3_eu + m3_jp
            df_m['Global_M2_Tril'] = global_m2_sum.interpolate(limit_direction='both')
            df_m['Global_M2_YoY'] = df_m['Global_M2_Tril'].pct_change(52) * 100
        else: df_m['Global_M2_YoY'] = pd.Series(dtype=float)
        
        s_fed, s_ecb, s_boj = df_m.get('fed'), df_m.get('ecb'), df_m.get('boj')
        if s_fed is not None and s_ecb is not None and s_boj is not None:
            fed_t = s_fed / 1000000
            ecb_t = (s_ecb * df_m.get('eur_usd', 1)) / 1000000
            boj_t = (s_boj * 0.0001) / df_m.get('usd_jpy', 1)
            g3_sum = fed_t.fillna(0) + ecb_t.fillna(0) + boj_t.fillna(0)
            df_m['G3_Asset_Tril'] = g3_sum.replace(0, np.nan).interpolate()
            df_m['G3_Asset_YoY'] = df_m['G3_Asset_Tril'].pct_change(52) * 100
        else: df_m['G3_Asset_YoY'] = pd.Series(dtype=float)

        df_m['Fed_Net_Tril'] = (df_m.get('fed',0)/1000 - df_m.get('tga',0)/1000 - df_m.get('rrp',0)/1000000)
        df_m['Fed_Net_YoY'] = df_m['Fed_Net_Tril'].pct_change(52) * 100

        # Shift
        def apply_shift(s, days):
            if s.empty: return pd.Series(dtype=float)
            new_s = s.copy()
            new_s.index = new_s.index - pd.Timedelta(days=days)
            return new_s

        processed = {}
        for asset in ASSETS_CONFIG:
            s = raw.get(asset['id'], pd.Series(dtype=float))
            if isinstance(s.index, pd.DatetimeIndex):
                processed[asset['id']] = apply_shift(s, shift_days)
            else: processed[asset['id']] = pd.Series(dtype=float)

        # Chart Render
        st.subheader(f"📊 Integrated Strategy Chart (Shift: {shift_days}d)")
        start_viz = pd.to_datetime('2021-06-01') 
        def flt(s): return s[s.index >= start_viz] if not s.empty else s

        if "Global M2" in liq_option:
            liq_v, liq_name, liq_color = flt(df_m['Global_M2_YoY']), "Global M2", "#FF4500"
        elif "G3" in liq_option:
            liq_v, liq_name, liq_color = flt(df_m['G3_Asset_YoY']), "G3 Assets", "#FFD700"
        else:
            liq_v, liq_name, liq_color = flt(df_m['Fed_Net_YoY']), "Fed Net", "#00FF7F"

        liq_v = liq_v.replace([np.inf, -np.inf], np.nan).dropna()
        if not liq_v.empty:
            l_min, l_max = liq_v.min(), liq_v.max()
            if pd.isna(l_min) or pd.isna(l_max): l_rng = [-20, 20]
            else: l_rng = [l_min - (l_max-l_min)*0.1, l_max + (l_max-l_min)*0.1]
        else: l_rng = [-20, 20]

        active_assets = [a for a in ASSETS_CONFIG if selected_assets[a['id']]]
        num_active = len(active_assets)
        if is_mobile: tick_fmt, margin, font_size = "s", 0.03, 10
        else: tick_fmt, margin, font_size = ",", 0.05 if num_active > 5 else 0.08, 12
        if num_active == 0: domain_end = 0.95
        else: domain_end = max(0.5, 1.0 - (num_active * margin))

        layout = go.Layout(
            template="plotly_dark", height=600,
            xaxis=dict(domain=[0.0, domain_end], showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
            yaxis=dict(title=dict(text=liq_name, font=dict(color=liq_color, size=font_size)), tickfont=dict(color=liq_color, size=font_size), range=l_rng, showgrid=False),
            legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"),
            hovermode="x", margin=dict(l=30, r=10, t=80, b=50)
        )
        fig = go.Figure(layout=layout)

        if not liq_v.empty:
            h = liq_color.lstrip('#')
            rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
            if shift_days != 0:
                last_date = liq_v.index.max()
                start_date = last_date - pd.Timedelta(days=abs(shift_days))
                fig.add_shape(type="rect", x0=start_date, x1=last_date, y0=l_rng[0], y1=l_rng[1], fillcolor="rgba(200, 200, 200, 0.15)", line=dict(width=0), layer="below")
                fig.add_annotation(x=last_date, y=l_rng[1], text=f"Lag:{abs(shift_days)}d", showarrow=False, yshift=10, xshift=-40, font=dict(color="rgba(255,255,255,0.7)", size=10))
            fig.add_trace(go.Scatter(x=liq_v.index, y=liq_v, name=liq_name, line=dict(color=liq_color, width=3), fill='tozeroy', fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.15)", yaxis='y', hoverinfo='none'))

        current_pos = domain_end
        for i, asset in enumerate(active_assets):
            data = flt(processed[asset['id']])
            data = data.replace([np.inf, -np.inf], np.nan).dropna()
            if data.empty: continue
            axis_name = f'yaxis{i+2}'
            axis_key = f'y{i+2}'
            d_min, d_max = data.min(), data.max()
            if pd.isna(d_min) or pd.isna(d_max) or d_min <= 0: d_min = 0.0001
            
            t_type = "linear"
            if asset['id'] == 'hy_spread': rng = [d_min - 0.5, d_max + 0.5]
            elif asset['id'] == 'doge': 
                t_type = "log"
                log_min, log_max = np.log10(d_min), np.log10(d_max)
                span = log_max - log_min
                rng = [log_min - span*0.1, log_max + span*0.1]
            else:
                span = d_max - d_min
                rng = [d_min - span*0.2, d_max + span*0.1]

            fig.update_layout({
                axis_name: dict(
                    title=dict(text=asset['name'], font=dict(color=asset['color'], size=font_size)),
                    tickfont=dict(color=asset['color'], size=font_size),
                    overlaying="y", side="right", anchor="free", position=current_pos,
                    range=rng, type=t_type, showgrid=False, tickformat=tick_fmt
                )
            })
            fig.add_trace(go.Scatter(x=data.index, y=data, name=asset['name'], line=dict(color=asset['color'], width=2), yaxis=axis_key, hoverinfo='none'))
            current_pos += margin

        st.plotly_chart(fig, use_container_width=True, key="main_chart")

        # ------------------------------------------------------------------
        # [NEW] Stress Test & Backtest Display
        # ------------------------------------------------------------------
        st.markdown("---")
        st.subheader("📉 Crash Simulation (Stress Test)")
        st.caption(f"가정: HY Spread가 하루에 **{spike_threshold} bps 이상 급등**하면 즉시 매도 후 **{look_forward_days}일간 관망**했다면?")

        if 'hy_spread' in raw and 'btc' in raw:
            # 원본 데이터 사용 (Shift 안 된 것)
            res_df = run_stress_test(raw['hy_spread'], raw['btc'], spike_threshold, look_forward_days)
            
            if not res_df.empty:
                # 통계 계산
                total_sigs = len(res_df)
                success_sigs = len(res_df[res_df['Outcome'].str.contains("성공")])
                success_rate = (success_sigs / total_sigs) * 100
                
                c1, c2, c3 = st.columns(3)
                c1.metric("총 위험 신호 발생", f"{total_sigs} 회")
                c2.metric("하락 방어 성공률", f"{success_rate:.1f}%", help="신호 발생 후 실제로 가격이 떨어진 비율")
                
                # 최근 5개 사례만 보여주기 (Expandable)
                st.dataframe(res_df.style.map(lambda x: 'color: #00FF7F' if '성공' in str(x) else ('color: #FF4500' if '휩쏘' in str(x) else ''), subset=['Outcome']), use_container_width=True)
            else:
                st.info(f"설정하신 민감도({spike_threshold} bps)로는 지난 기간 동안 위험 신호가 감지되지 않았습니다. 민감도를 낮춰보세요.")

        # ... (Quant Analytics - 생략되었으나 이전 코드와 동일) ...

    else:
        st.error("❌ 데이터 로드 실패")

except Exception as e:
    st.error(f"⚠️ 시스템 오류: {str(e)}")
