import io
import math
import requests
import numpy as np
import pandas as pd
import streamlit as st

# =========================
# liq repo - Global M2 Validator
# =========================

st.set_page_config(page_title="liq | Global M2 Validator", layout="wide")

# -------------------------
# Config
# -------------------------
WEEK_RULE = "W-FRI"

# Core-12 (예시 세트) - 필요하면 이름만 바꿔서 고정하면 됨
CORE12 = [
    "US", "EA", "JP",
    "CN", "GB", "CA", "CH",
    "AU", "IN", "KR",
    "BR", "MX",
]

# 각국 통화 코드(USD 환산용)
CCY = {
    "US": "USD",
    "EA": "EUR",
    "JP": "JPY",
    "CN": "CNY",
    "GB": "GBP",
    "CA": "CAD",
    "CH": "CHF",
    "AU": "AUD",
    "IN": "INR",
    "KR": "KRW",
    "BR": "BRL",
    "MX": "MXN",
}

# -------------------------
# Helpers: robust download
# -------------------------
def http_get(url: str, timeout=30) -> str:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "liq-validator/1.0"})
    r.raise_for_status()
    return r.text

def parse_two_col_csv(text: str, date_col="DATE", value_col=None) -> pd.Series:
    """
    FRED/ECB csvdata는 컬럼명이 다를 수 있어서 유연 파서.
    반환: DatetimeIndex / float Series
    """
    df = pd.read_csv(io.StringIO(text))
    # date column 추정
    if date_col not in df.columns:
        # ECB csvdata는 TIME_PERIOD가 일반적
        if "TIME_PERIOD" in df.columns:
            date_col = "TIME_PERIOD"
        elif "date" in [c.lower() for c in df.columns]:
            date_col = [c for c in df.columns if c.lower() == "date"][0]
        else:
            raise ValueError(f"Cannot find date column in CSV. columns={list(df.columns)}")

    if value_col is None:
        # FRED: VALUE, ECB: OBS_VALUE
        if "VALUE" in df.columns:
            value_col = "VALUE"
        elif "OBS_VALUE" in df.columns:
            value_col = "OBS_VALUE"
        else:
            # fallback: 마지막 컬럼
            value_col = df.columns[-1]

    out = df[[date_col, value_col]].copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out = out.dropna(subset=[date_col]).set_index(date_col).sort_index()
    s = out[value_col].astype(float)
    return s

# -------------------------
# Data sources (Auto)
# -------------------------
def fetch_us_m2_fred() -> pd.Series:
    """
    FRED M2SL (Monthly, SA). API key 없이도 table-data CSV 경로로 실무적으로 수급 가능.
    (FRED series metadata: M2SL)  [oai_citation:3‡FRED](https://fred.stlouisfed.org/data/M2SL?utm_source=chatgpt.com)
    """
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=M2SL"
    txt = http_get(url)
    s = parse_two_col_csv(txt, date_col="DATE", value_col="M2SL")
    s.name = "US_M2_USD_BIL"
    # 단위: Billions of Dollars (USD)
    return s

def fetch_ea_m2_ecb() -> pd.Series:
    """
    ECB Data Portal BSI:
    Monetary aggregate M2, stocks, Euro area, SA
    Series key: BSI.M.U2.Y.V.M20.X.1.U2.2300.Z01.E  [oai_citation:4‡data.ecb.europa.eu](https://data.ecb.europa.eu/data/datasets/BSI/BSI.M.U2.Y.V.M20.X.1.U2.2300.Z01.E?utm_source=chatgpt.com)
    API format=csvdata  [oai_citation:5‡data.ecb.europa.eu](https://data.ecb.europa.eu/help/api/data?utm_source=chatgpt.com)
    """
    series_key = "BSI.M.U2.Y.V.M20.X.1.U2.2300.Z01.E"
    url = f"https://data-api.ecb.europa.eu/service/data/BSI/{series_key}?format=csvdata"
    txt = http_get(url)
    s = parse_two_col_csv(txt, date_col="TIME_PERIOD", value_col="OBS_VALUE")
    s.name = "EA_M2_EUR_MIL"
    # 단위: Millions of Euro
    return s

def fetch_ecb_exr_monthly(base: str, quote: str = "EUR") -> pd.Series:
    """
    ECB EXR 예시에서:
    https://data-api.ecb.europa.eu/service/data/EXR/M.USD.EUR.SP00.A  [oai_citation:6‡data.ecb.europa.eu](https://data.ecb.europa.eu/help/data-examples?utm_source=chatgpt.com)
    를 그대로 확장:
    M.{base}.{quote}.SP00.A (Monthly, spot)
    반환은 'quote 1단위당 base'로 이해하면 됨.
    예: base=USD, quote=EUR -> 1 EUR = x USD (EURUSD)
    """
    key = f"M.{base}.{quote}.SP00.A"
    url = f"https://data-api.ecb.europa.eu/service/data/EXR/{key}?format=csvdata"
    txt = http_get(url)
    s = parse_two_col_csv(txt, date_col="TIME_PERIOD", value_col="OBS_VALUE")
    s.name = f"EXR_{base}{quote}"
    return s

def build_fx_to_usd(ccy: str) -> pd.Series:
    """
    ECB EXR을 이용해 (현지통화 -> USD) 월간 환율을 만든다.
    ECB는 EUR 기준 교차환산이 안정적:
    - EURUSD = USD per EUR (base=USD, quote=EUR)
    - EURCCY = CCY per EUR (base=CCY, quote=EUR)
    그러면 1 CCY = (USD per EUR)/(CCY per EUR) USD
    """
    if ccy == "USD":
        # USD->USD = 1
        idx = pd.date_range("1990-01-01", pd.Timestamp.today(), freq="MS")
        return pd.Series(1.0, index=idx, name="FX_USDUSD")

    eurusd = fetch_ecb_exr_monthly("USD", "EUR")  # 1 EUR = x USD
    eurccy = fetch_ecb_exr_monthly(ccy, "EUR")   # 1 EUR = x CCY

    # align
    df = pd.concat([eurusd, eurccy], axis=1).dropna()
    fx = df.iloc[:, 0] / df.iloc[:, 1]
    fx.name = f"FX_{ccy}USD"
    return fx

# -------------------------
# Manual CSV loader (for non-auto countries)
# -------------------------
def load_manual_m2(country: str, uploaded_file) -> pd.Series:
    """
    업로드 CSV 포맷(권장):
    date,value  (컬럼명 대소문자 무관)
    예: 2024-01-31, 4496.02
    """
    df = pd.read_csv(uploaded_file)
    cols = {c.lower(): c for c in df.columns}
    if "date" not in cols or "value" not in cols:
        raise ValueError(f"[{country}] CSV must have columns: date,value (case-insensitive). got={list(df.columns)}")
    dcol, vcol = cols["date"], cols["value"]
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df[vcol] = pd.to_numeric(df[vcol], errors="coerce")
    df = df.dropna(subset=[dcol]).set_index(dcol).sort_index()
    s = df[vcol].astype(float)
    s.name = f"{country}_M2_LOCAL"
    return s

# -------------------------
# Standardization + Validation
# -------------------------
def to_weekly_last(s: pd.Series) -> pd.Series:
    # 월간/주간 무엇이든 "주간(W-FRI) last"로 표준화
    return s.resample(WEEK_RULE).last().ffill()

def validate_series(name: str, s: pd.Series) -> dict:
    info = {
        "name": name,
        "start": None if s.empty else s.index.min().date(),
        "end": None if s.empty else s.index.max().date(),
        "n": int(s.shape[0]),
        "missing_ratio": float(s.isna().mean()) if len(s) else math.nan,
        "nonpositive_ratio": float((s <= 0).mean()) if len(s) else math.nan,
        "has_gaps_gt_45d": False,
    }
    if len(s) >= 2:
        gaps = s.index.to_series().diff().dt.days.dropna()
        info["has_gaps_gt_45d"] = bool((gaps > 45).any())
    return info

# =========================
# UI
# =========================
st.title("liq | Global M2 (USD) Validator — Core12")

with st.sidebar:
    st.header("1) Core12 구성")
    st.caption("Auto 수급: US(FRED), EA(ECB-BSI), FX(ECB-EXR). 나머지는 CSV 업로드로 주입 후 동일 파이프라인에서 검증.")
    selected = st.multiselect("대상 국가", CORE12, default=CORE12)

    st.header("2) 수동 데이터 업로드(필요 국가만)")
    uploads = {}
    for c in selected:
        # US/EA는 자동 수급이므로 업로드 불필요 (원하면 비교 검증용으로 받게 확장 가능)
        if c in ("US", "EA"):
            continue
        uploads[c] = st.file_uploader(f"{c} M2 CSV (date,value)", type=["csv"], key=f"up_{c}")

    st.header("3) 옵션")
    use_sa_ea = st.checkbox("EA: Seasonally adjusted(Y) 사용", value=True)
    st.caption("EA 시리즈는 SA(Y)로 고정 구현. 필요하면 N(비SA)로 바꿔 비교 가능.")

# -------------------------
# Build dataset
# -------------------------
@st.cache_data(show_spinner=False, ttl=6 * 60 * 60)
def build_all(selected_countries):
    # M2 raw
    m2 = {}

    # Auto
    if "US" in selected_countries:
        m2["US"] = fetch_us_m2_fred()  # USD billions
    if "EA" in selected_countries:
        m2["EA"] = fetch_ea_m2_ecb()   # EUR millions

    # Manual
    # (파일 업로드는 cache 밖에서 합류시키는 게 안전하지만, 여기선 구조 단순화를 우선)
    return m2

m2_auto = build_all(tuple(selected))

# Load manual outside cache (because UploadedFile is not hash-stable)
m2_all = dict(m2_auto)
manual_errors = []
for c in selected:
    if c in ("US", "EA"):
        continue
    if uploads.get(c) is None:
        continue
    try:
        m2_all[c] = load_manual_m2(c, uploads[c])
    except Exception as e:
        manual_errors.append(str(e))

if manual_errors:
    st.error("수동 업로드 오류:\n- " + "\n- ".join(manual_errors))
    st.stop()

# FX build (ECB)
fx_map = {}
fx_errors = []
for c in selected:
    ccy = CCY[c]
    try:
        fx_map[c] = build_fx_to_usd(ccy)  # monthly
    except Exception as e:
        fx_errors.append(f"{c}({ccy}) FX build failed: {e}")

if fx_errors:
    st.error("FX 수급/생성 오류:\n- " + "\n- ".join(fx_errors))
    st.stop()

# Convert M2 -> USD level
m2_usd = {}
meta = []

for c, s in m2_all.items():
    ccy = CCY[c]
    fx = fx_map[c]

    # align monthly -> weekly (convert after aligning in time)
    s_w = to_weekly_last(s)

    if c == "US":
        # US already USD billions
        usd = s_w * 1e9
        unit_note = "FRED M2SL: Billions USD"
    elif c == "EA":
        # ECB: Millions EUR -> EUR -> USD
        # EUR millions * 1e6 * (EUR->USD)
        fx_w = to_weekly_last(fx)
        usd = (s_w * 1e6) * fx_w
        unit_note = "ECB BSI: Millions EUR"
    else:
        # Manual: assume "local currency units" (you must ensure the CSV unit is consistent with official series)
        fx_w = to_weekly_last(fx)
        usd = s_w * fx_w
        unit_note = "Manual CSV: local currency unit (as provided)"

    usd.name = f"{c}_M2_USD"
    m2_usd[c] = usd

    meta.append({
        "country": c,
        "ccy": ccy,
        "m2_unit": unit_note,
        "m2_start": None if s_w.empty else s_w.index.min().date(),
        "m2_end": None if s_w.empty else s_w.index.max().date(),
        "fx_start": None if fx.empty else fx.index.min().date(),
        "fx_end": None if fx.empty else fx.index.max().date(),
    })

# Global sum
df_usd = pd.concat(m2_usd.values(), axis=1).sort_index()
global_m2 = df_usd.sum(axis=1, min_count=1)
global_m2.name = "GLOBAL_M2_USD"

# Validations
val_rows = []
for c, s in m2_usd.items():
    val = validate_series(s.name, s)
    val["country"] = c
    val_rows.append(val)

val_df = pd.DataFrame(val_rows).set_index("country")

# =========================
# Display
# =========================
c1, c2 = st.columns([2, 1], gap="large")

with c1:
    st.subheader("A) Global M2 (USD) — Weekly (W-FRI)")
    st.line_chart(global_m2)

    st.subheader("B) Components (USD) — Weekly (W-FRI)")
    st.line_chart(df_usd)

with c2:
    st.subheader("C) Data/FX coverage")
    st.dataframe(pd.DataFrame(meta))

    st.subheader("D) Sanity checks")
    st.dataframe(val_df)

st.subheader("E) Export (for integration)")
export_df = pd.concat([global_m2, df_usd], axis=1)
csv = export_df.dropna(how="all").to_csv(index=True).encode("utf-8-sig")
st.download_button("Download weekly_global_m2_usd.csv", data=csv, file_name="weekly_global_m2_usd.csv", mime="text/csv")

st.caption(
    "주의: US/EA는 자동 수급이지만, 나머지 국가는 CSV 단위/정의(M2 구성, SA/NSA, 월말잔액/평잔 등)를 "
    "공식 통계 기준으로 맞춰 넣어야 합니다. 이 앱은 파이프라인(환산/리샘플/합산/검증)을 고정해 재현성을 확보합니다."
)