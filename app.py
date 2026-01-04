import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import yfinance as yf
import json
import io
import warnings
from datetime import date

warnings.filterwarnings("ignore", category=FutureWarning)  # 불필요한 경고 좀 더 억제

st.set_page_config(page_title="GM Terminal Final", layout="wide")
st.title("🏛️ Grand Master: Multi-Axis Final")

@st.cache_data(ttl=3600, show_spinner="데이터 불러오는 중...")
def fetch_data_final():
    d = {}
    
    tickers = {
        'btc': "BTC-KRW",
        'doge': "DOGE-KRW",
        'nasdaq': "^IXIC"
    }
    
    for key, ticker in tickers.items():
        try:
            # timeout 추가 + auto_adjust 제거 (KRW 쌍은 영향 없음)
            df = yf.download(
                ticker,
                period="max",
                progress=False,
                timeout=15
            )
            if not df.empty and 'Close' in df.columns:
                s = df['Close'].tz_localize(None)
                d[key] = s
            else:
                d[key] = pd.Series(dtype=float)
                
        except Exception as e:
            st.warning(f"{ticker} 데이터 로드 실패 → {str(e)}")
            d[key] = pd.Series(dtype=float)
    
    # FRED
    def get_fred(series_id):
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
            r = requests.get(url, timeout=12)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text), index_col=0, parse_dates=True)
            return df.squeeze().resample('D').interpolate(method='time').tz_localize(None)
        except Exception as e:
            st.warning(f"FRED {series_id} 불러오기 실패 → {str(e)}")
            return pd.Series(dtype=float)
    
    d['fed']  = get_fred('WALCL')
    d['tga']  = get_fred('WTREGEN')
    d['rrp']  = get_fred('RRPONTSYD')
    
    # Difficulty JSON
    try:
        with open('difficulty (1).json', 'r', encoding='utf-8') as f:
            js = json.load(f)['difficulty']
        df_js = pd.DataFrame(js)
        df_js['Date'] = pd.to_datetime(df_js['x'], unit='ms').dt.tz_localize(None)
        d['diff'] = df_js.set_index('Date').sort_index()['y']
    except Exception as e:
        st.warning(f"Difficulty JSON 로드 실패 → {str(e)}")
        d['diff'] = pd.Series(dtype=float)
    
    return d


# ────────────────────────────────────────────────────────────────
raw = fetch_data_final()

if not raw.get('btc', pd.Series()).empty:
    # ── 이후 기존 로직 그대로 ────────────────────────────────
    # (Liquidity 계산, -90일 shift, 차트 그리기 등)
    # ... (생략) ...
    
    st.success("✅ 데이터 로드 성공! 차트 표시 중...")
    # 여기에 기존 차트 코드 계속
else:
    st.error("❌ BTC 데이터가 여전히 비어있습니다.")
    st.info("가능한 해결책:\n1. requirements.txt에 yfinance==0.2.40 고정\n2. Streamlit Cloud 재배포\n3. 로컬에서는 잘 되는지 먼저 확인")
