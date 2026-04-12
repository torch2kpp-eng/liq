# -*- coding: utf-8 -*-
# streamlit_app.py
# BTC Lead-Lag Lab (Weekly) + TAB5 Forecast/Backtest + MAIN Predicted BTC line
# v2.16 - Liquidity toggle(Fed Net Liquidity vs G2 M2 USD) + TRUE spaghetti (weekly anchors, dynamic lag per anchor)
#
# Key upgrades:
#  1) Liquidity source toggle applies to all tabs (TAB2/3/4/MAIN/TAB5)
#  2) MAIN predicted history line = "true-history endpoint line"
#     - For each past anchor date t: fit on history, use that anchor's lag xx(t), forecast to t+xx(t)
#     - Endpoint at t+xx(t) becomes the historical predicted line (green dashed)
#  3) Spaghetti = full forecast paths for each anchor (weekly anchors), with dynamic lag per anchor
#     - Exported as long-format CSV (anchor_dt, path_dt, pred_px, lag_weeks, liq_source, combo_type)
#  4) One-file CSV (MAIN/TAB4) now includes:
#     - pred_hist_endpoint_dynamic (true history)
#     - latest_forecast_path (current anchor forecast)
#
# Data sources:
#  - BTC, DXY, multi-assets: Yahoo primary (yfinance), Stooq fallback
#  - Liquidity:
#     A) Fed Net Liquidity (FRED): WALCL - TGA - RRPONTSYD (RRP in billions -> millions)
#     B) G2 M2 USD: US M2SL (FRED, billions USD) + EA M2 (ECB, EUR) * EURUSD (ECB)
#
# Notes:
#  - This app intentionally favors correctness/traceability over speed (spaghetti is expensive).
#  - If ECB endpoints change, G2 mode may error; Fed mode remains available.
#  - Price loader was upgraded in v2.18.1 because Stooq CSV can return non-CSV payloads.

import io
import math
import requests
from urllib.parse import quote_plus
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# =========================
# ENCODING / UI TEXT SAFETY
# =========================
def fix_mojibake_text(text):
    """
    Try to recover UTF-8 Korean text that was accidentally decoded as latin1/cp1252.
    If the text already looks normal, return it unchanged.
    """
    if not isinstance(text, str):
        return text

    suspicious_tokens = ("Ã¬", "Ã«", "Ãª", "Ã­", "Ã£", "Ã", "Ã¢", "Ã")
    if not any(tok in text for tok in suspicious_tokens):
        return text

    def _hangul_score(s):
        return sum(0xAC00 <= ord(ch) <= 0xD7A3 for ch in s)

    base_score = (_hangul_score(text), -sum(text.count(tok) for tok in suspicious_tokens))
    best = text
    best_score = base_score

    for enc in ("latin1", "cp1252"):
        try:
            cand = text.encode(enc).decode("utf-8")
        except Exception:
            continue
        cand_score = (_hangul_score(cand), -sum(cand.count(tok) for tok in suspicious_tokens))
        if cand_score > best_score:
            best = cand
            best_score = cand_score

    return best


def fix_ui_obj(obj):
    if isinstance(obj, str):
        return fix_mojibake_text(obj)
    if isinstance(obj, list):
        return [fix_ui_obj(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(fix_ui_obj(x) for x in obj)
    if isinstance(obj, dict):
        return {fix_ui_obj(k): fix_ui_obj(v) for k, v in obj.items()}
    return obj


def ui_text(text):
    return fix_mojibake_text(text)


def _patch_streamlit_text_apis():
    _markdown = st.markdown
    def markdown(body, *args, **kwargs):
        return _markdown(fix_ui_obj(body), *args, **kwargs)
    st.markdown = markdown

    _caption = st.caption
    def caption(body, *args, **kwargs):
        return _caption(fix_ui_obj(body), *args, **kwargs)
    st.caption = caption

    _title = st.title
    def title(body, *args, **kwargs):
        return _title(fix_ui_obj(body), *args, **kwargs)
    st.title = title

    _header = st.header
    def header(body, *args, **kwargs):
        return _header(fix_ui_obj(body), *args, **kwargs)
    st.header = header

    _subheader = st.subheader
    def subheader(body, *args, **kwargs):
        return _subheader(fix_ui_obj(body), *args, **kwargs)
    st.subheader = subheader

    _write = st.write
    def write(*args, **kwargs):
        return _write(*[fix_ui_obj(a) for a in args], **kwargs)
    st.write = write

    _warning = st.warning
    def warning(body, *args, **kwargs):
        return _warning(fix_ui_obj(body), *args, **kwargs)
    st.warning = warning

    _success = st.success
    def success(body, *args, **kwargs):
        return _success(fix_ui_obj(body), *args, **kwargs)
    st.success = success

    _error = st.error
    def error(body, *args, **kwargs):
        return _error(fix_ui_obj(body), *args, **kwargs)
    st.error = error

    _info = st.info
    def info(body, *args, **kwargs):
        return _info(fix_ui_obj(body), *args, **kwargs)
    st.info = info

    _metric = st.metric
    def metric(label, value=None, delta=None, *args, **kwargs):
        return _metric(fix_ui_obj(label), fix_ui_obj(value), fix_ui_obj(delta), *args, **kwargs)
    st.metric = metric

    _spinner = st.spinner
    def spinner(text="In progress...", *args, **kwargs):
        return _spinner(fix_ui_obj(text), *args, **kwargs)
    st.spinner = spinner

    _tabs = st.tabs
    def tabs(tabs, *args, **kwargs):
        return _tabs(fix_ui_obj(tabs), *args, **kwargs)
    st.tabs = tabs

    _button = st.button
    def button(label, *args, **kwargs):
        return _button(fix_ui_obj(label), *args, **kwargs)
    st.button = button

    _download_button = st.download_button
    def download_button(label, *args, **kwargs):
        return _download_button(fix_ui_obj(label), *args, **kwargs)
    st.download_button = download_button

    _checkbox = st.checkbox
    def checkbox(label, *args, **kwargs):
        return _checkbox(fix_ui_obj(label), *args, **kwargs)
    st.checkbox = checkbox

    _radio = st.radio
    def radio(label, options, *args, **kwargs):
        return _radio(fix_ui_obj(label), fix_ui_obj(options), *args, **kwargs)
    st.radio = radio

    _selectbox = st.selectbox
    def selectbox(label, options, *args, **kwargs):
        return _selectbox(fix_ui_obj(label), fix_ui_obj(options), *args, **kwargs)
    st.selectbox = selectbox

    _multiselect = st.multiselect
    def multiselect(label, options, *args, **kwargs):
        return _multiselect(fix_ui_obj(label), fix_ui_obj(options), *args, **kwargs)
    st.multiselect = multiselect


_patch_streamlit_text_apis()


# =========================
# FIXED CONFIG (NO CONTROLS)
# =========================
APP_VERSION = "v2.18.5-regimefix-fundingdiag"
H19_DIAG_W = 19

START_DATE = "2015-01-01"

TODAY_KST = pd.Timestamp.now(tz="Asia/Seoul").normalize().tz_localize(None)
END_DATE = TODAY_KST.strftime("%Y-%m-%d")

WEEK_RULE = "W-FRI"

# Rolling
WINDOW_WEEKS = 104
STEP_WEEKS = 1
STEP_WEEKS_2D = 2

LAG_MIN_WEEKS = 2
LAG_MAX_WEEKS = 26
MIN_OBS = 30

CORR_HEATMAP_MASK_ABS_BELOW = 0.05
VALID_BEST_ABS_CORR_BELOW = 0.08
BEST_LAG_SMOOTH_WEEKS = 5
TSTAT_HEATMAP_MASK_ABS_BELOW = 2.0

TAB1_INVERT_X = True
TAB2_INVERT_X = False
TAB3_INVERT_X = False

# Price symbols / source preference
# Primary: Yahoo Finance via yfinance
# Fallback: Stooq (kept only as secondary backup; Stooq CSV endpoint can change)
YAHOO_BTC = "BTC-USD"
YAHOO_DXY = "DX-Y.NYB"

STOOQ_BTC = "btcusd"
STOOQ_DXY_PRIMARY = "dx.f"
STOOQ_DXY_FALLBACK = "usd_i"

# FRED series (Fed Net Liquidity)
FRED_WALCL = "WALCL"
FRED_TGA = "WTREGEN"
FRED_RRP = "RRPONTSYD"

# FRED series (US M2, for G2)
FRED_US_M2SL = "M2SL"  # Billions of dollars (levels)

# ECB series keys (EA M2, EURUSD)
# You may need to update these if ECB changes codes.
ECB_EA_M2_KEY = "BSI.M.U2.Y.V.M30.X.1.U2.2300.Z01.E"  # EA M2 (M3? historically used as M2 proxy in some scripts)
ECB_EURUSD_KEY = "EXR.D.USD.EUR.SP00.A"  # EUR per 1 USD? (check orientation); we will infer and invert if needed
ECB_EURUSD_KEY_FALLBACK = "EXR.D.EUR.USD.SP00.A"  # USD per 1 EUR

# LDLI
LDLI_BASE = 100.0
LDLI_SCALE = 10.0  # LDLI_level = BASE + SCALE*cumsum(combo)

# MAIN/TAB4 display windows
PAST_WEEKS_FIXED = 52
PAST_WEEKS_EXTENDED = 208
PAST_WEEKS_LONG = 416

# MAIN regime metrics horizons (weeks)
REGIME_HORIZONS_WEEKS = [4, 13, 26]

# Regime v2
REGIME_SMOOTH_WEEKS = 3
REGIME_DIR_LOOKBACK_WEEKS = 4
REGIME_DIR_DEADBAND = 0.20

# TAB5 defaults
TAB5_DEFAULT_FIT_W = 104

# Spaghetti controls (fixed by your request)
SPAGHETTI_ANCHOR_STEP_WEEKS = 1  # weekly anchors (heavy)
SPAGHETTI_ALPHA = 0.35           # not too faint, still readable
SPAGHETTI_LW = 1.1               # spaghetti linewidth
SPAGHETTI_MAX_ANCHORS = None     # None means use all anchors

# =========================
# Multi-asset mapping (Yahoo primary / Stooq fallback)
# =========================
ASSET_SYMBOLS_YAHOO: Dict[str, List[str]] = {
    "Bitcoin (BTCUSD)": [YAHOO_BTC],
    "Gold (XAUUSD)": ["GC=F"],
    "Silver (XAGUSD)": ["SI=F"],
    "Nasdaq (Composite)": ["^IXIC"],
    "KOSPI": ["^KS11"],

    "Ethereum (ETH)": ["ETH-USD"],
    "Dogecoin (DOGE)": ["DOGE-USD"],
    "Chainlink (LINK)": ["LINK-USD"],
    "Cardano (ADA)": ["ADA-USD"],
}

ASSET_SYMBOLS_STOOQ: Dict[str, List[str]] = {
    "Bitcoin (BTCUSD)": [STOOQ_BTC, "btc.v"],
    "Gold (XAUUSD)": ["xauusd"],
    "Silver (XAGUSD)": ["xagusd"],
    "Nasdaq (Composite)": ["^ndq", "^ndx"],
    "KOSPI": ["^kospi"],

    # Crypto (Stooq often uses *.v)
    "Ethereum (ETH)": ["eth.v", "ethusd", "ethusd_i"],
    "Dogecoin (DOGE)": ["doge.v", "dogeusd", "dogeusd_i"],
    "Chainlink (LINK)": ["link.v", "linkusd", "linkusd_i"],
    "Cardano (ADA)": ["ada.v", "adausd", "adausd_i"],
}


# =========================
# Streamlit base
# =========================
st.set_page_config(page_title=f"BTC Lead-Lag Lab {APP_VERSION}", layout="wide")
st.title(f"BTC Lead-Lag Lab {APP_VERSION} (Weekly)")

with st.sidebar:
    st.markdown("## Settings")
    LIQ_SOURCE = st.selectbox(
        "Liquidity Source",
        [
            "Fed Net Liquidity (FRED)",
            "G2 M2 (US+EA, USD)",
        ],
        index=0,
    )

    ALPHA_MODE = st.selectbox(
        "Alpha / intercept mode",
        [
            "OLS (learn alpha)",
            "ZERO (alpha=0)",
        ],
        index=0,
    )

    with st.expander(ui_text("Alpha Inputs (MVRV / Funding)"), expanded=False):
        USE_BINANCE_FUNDING = st.checkbox(
            "Use Binance funding auto-loader",
            value=True,
            help="Auto-load BTCUSDT funding history from Binance Futures and convert it to weekly funding z-score.",
        )
        FUNDING_SYMBOL = st.text_input("Funding symbol", value="BTCUSDT")
        MVRV_FILE = st.file_uploader(
            "MVRV Z CSV upload (optional)",
            type=["csv"],
            help="Upload a CSV with a date column and an MVRV Z / Z-score value column.",
        )

MVRV_FILE_BYTES = MVRV_FILE.getvalue() if MVRV_FILE is not None else None
MVRV_FILE_NAME = MVRV_FILE.name if MVRV_FILE is not None else None

FORECAST_MODEL = st.selectbox(
        "Forecast model (MAIN/TAB4)",
        [
            "Drivers-only Î (legacy)",
            "ECM on LDLI level (gap + ÎLDLI)",
        ],
        index=1,
        help="Legacy uses only Î(liquidity) and Î(DXY) drivers. ECM uses LDLI level alignment + error-correction (gap) to avoid flat/always-up spaghetti.",
    )

with st.expander(ui_text("ê³ ì  íë¼ë¯¸í°(ìë ¥ê°) ë³´ê¸°"), expanded=False):
    st.code(
        "\n".join([
            f"APP_VERSION = {APP_VERSION}",
            f"START_DATE = {START_DATE}",
            f"END_DATE   = {END_DATE}  # auto: today(Asia/Seoul)",
            f"WEEK_RULE  = {WEEK_RULE}",
            "",
            f"[Rolling] WINDOW_WEEKS={WINDOW_WEEKS}, STEP_WEEKS={STEP_WEEKS}",
            f"[TAB3 2D] STEP_WEEKS_2D={STEP_WEEKS_2D}",
            f"LAGS_WEEKS={LAG_MIN_WEEKS}..{LAG_MAX_WEEKS}",
            f"MIN_OBS={MIN_OBS}",
            "",
            f"[Heatmap masks] CORR_MASK_ABS<{CORR_HEATMAP_MASK_ABS_BELOW}, TSTAT_MASK_ABS<{TSTAT_HEATMAP_MASK_ABS_BELOW}",
            f"[Valid best] VALID_BEST_ABS_CORR_BELOW={VALID_BEST_ABS_CORR_BELOW}, SMOOTH_WEEKS={BEST_LAG_SMOOTH_WEEKS}",
            "",
            f"[LDLI] LDLI_BASE={LDLI_BASE}, LDLI_SCALE={LDLI_SCALE}",
            f"[Forward Overlay] Past fixed={PAST_WEEKS_FIXED}w / {PAST_WEEKS_EXTENDED}w / {PAST_WEEKS_LONG}w, Future=+xx(dynamic)",
            "",
            f"[MAIN Regime v2] z_smooth={REGIME_SMOOTH_WEEKS}w, dir_lookback={REGIME_DIR_LOOKBACK_WEEKS}w, deadband={REGIME_DIR_DEADBAND}",
            f"[MAIN Regime Metrics] horizons={REGIME_HORIZONS_WEEKS}w",
            "",
            f"[TAB5] default fit_window={TAB5_DEFAULT_FIT_W}w",
            "",
            f"[Spaghetti] anchors_step={SPAGHETTI_ANCHOR_STEP_WEEKS}w (weekly), alpha={SPAGHETTI_ALPHA}, lw={SPAGHETTI_LW}",
            f"[Alpha mode] {ALPHA_MODE}",
        ]),
        language="text"
    )


# =========================
# Price loaders (Yahoo primary / Stooq fallback)
# =========================
def _strip_tz_index(idx: pd.Index) -> pd.DatetimeIndex:
    idx = pd.to_datetime(idx)
    tz = getattr(idx, "tz", None)
    if tz is not None:
        idx = idx.tz_convert(None)
    return pd.DatetimeIndex(idx)


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fetch_yahoo_daily(symbol: str) -> pd.Series:
    try:
        import yfinance as yf
    except Exception as e:
        raise RuntimeError(
            "Yahoo loader requires yfinance. Install with: pip install yfinance"
        ) from e

    end_plus_one = (pd.to_datetime(END_DATE) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        df = yf.download(
            symbol,
            start=START_DATE,
            end=end_plus_one,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as e:
        raise RuntimeError(f"Yahoo download failed for symbol={symbol}: {e}") from e

    if df is None or df.empty:
        raise RuntimeError(f"Yahoo returned empty data for symbol={symbol}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    cols = {str(c).strip().lower(): c for c in df.columns}
    close_col = cols.get("adj close") or cols.get("close")
    if close_col is None:
        raise RuntimeError(f"Yahoo schema unexpected for {symbol}: cols={list(df.columns)[:10]}")

    s = df[close_col]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s = pd.to_numeric(s, errors="coerce").dropna()
    s.index = _strip_tz_index(s.index)
    s = s.sort_index()
    s.name = symbol
    return s


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fetch_yahoo_daily_multi(candidates: List[str]) -> Tuple[pd.Series, str]:
    last_err = None
    for sym in candidates:
        try:
            s = fetch_yahoo_daily(sym)
            if s is not None and not s.empty:
                return s, sym
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to load from Yahoo candidates={candidates}. Last error={last_err}")


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fetch_stooq_daily(symbol: str) -> pd.Series:
    sym_q = quote_plus(symbol.strip().lower())
    url = f"https://stooq.com/q/d/l/?s={sym_q}&i=d"
    r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    txt = (r.text or "").strip()

    head = txt[:500].lower()
    if "get your apikey" in head or "apikey" in head and "stooq" in head and ("date" not in head or "close" not in head):
        raise RuntimeError(f"Stooq CSV endpoint requires apikey or returned a non-CSV response for symbol={symbol}")
    if head.startswith("<") or "document.write" in head or ("date" not in head and "close" not in head):
        raise RuntimeError(f"Non-CSV payload from Stooq for symbol={symbol}")

    df = pd.read_csv(io.StringIO(txt))
    cols = {c.lower().strip(): c for c in df.columns}
    date_col = cols.get("date")
    close_col = cols.get("close")
    if date_col is None or close_col is None:
        raise RuntimeError(f"Unexpected Stooq schema for {symbol}: cols={list(df.columns)[:10]}")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)
    s = pd.to_numeric(df[close_col], errors="coerce").dropna()
    s.name = symbol
    return s


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fetch_stooq_daily_multi(candidates: List[str]) -> Tuple[pd.Series, str]:
    last_err = None
    for sym in candidates:
        try:
            s = fetch_stooq_daily(sym)
            if s is not None and not s.empty:
                return s, sym
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to load from Stooq candidates={candidates}. Last error={last_err}")


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fetch_price_daily_multi(yahoo_candidates: List[str], stooq_candidates: Optional[List[str]] = None) -> Tuple[pd.Series, str]:
    errs = []

    if yahoo_candidates:
        try:
            s, used = fetch_yahoo_daily_multi(yahoo_candidates)
            return s, f"Yahoo:{used}"
        except Exception as e:
            errs.append(f"Yahoo={e}")

    if stooq_candidates:
        try:
            s, used = fetch_stooq_daily_multi(stooq_candidates)
            return s, f"Stooq:{used}"
        except Exception as e:
            errs.append(f"Stooq={e}")

    raise RuntimeError(" | ".join(errs) if errs else "No price source candidates provided")


# =========================
# FRED helper
# =========================
@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fetch_fred_fredgraph(series_id: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url)
    date_col = df.columns[0]
    val_col = df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col])
    s = pd.to_numeric(df[val_col], errors="coerce")
    s.index = df[date_col]
    s.name = series_id
    return s.dropna()


# =========================
# ECB helper
# =========================
def _ecb_csv_to_series(csv_text: str, time_col_candidates=("TIME_PERIOD", "TIME", "TIME_PERIOD "), value_col_candidates=("OBS_VALUE", "OBS_VALUE ")):
    df = pd.read_csv(io.StringIO(csv_text))
    # normalize columns
    cols = {c.strip().upper(): c for c in df.columns}
    time_col = None
    val_col = None
    for c in time_col_candidates:
        cc = c.strip().upper()
        if cc in cols:
            time_col = cols[cc]
            break
    for c in value_col_candidates:
        cc = c.strip().upper()
        if cc in cols:
            val_col = cols[cc]
            break
    if time_col is None or val_col is None:
        raise RuntimeError(f"ECB CSV schema unexpected. Columns={list(df.columns)}")

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])
    df = df.sort_values(time_col).set_index(time_col)
    s = pd.to_numeric(df[val_col], errors="coerce").dropna()
    return s


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def fetch_ecb_series(dataset: str, series_key: str) -> pd.Series:
    """
    ECB Data API:
      https://data-api.ecb.europa.eu/service/data/{DATASET}/{SERIES_KEY}?format=csvdata
    """
    url = f"https://data-api.ecb.europa.eu/service/data/{dataset}/{series_key}?format=csvdata"
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    s = _ecb_csv_to_series(r.text)
    s.name = f"{dataset}:{series_key}"
    return s.dropna()


# =========================
# Transform helpers
# =========================
def weekly_log_returns(level_s: pd.Series, week_rule: str = "W-FRI") -> pd.Series:
    level_s = level_s.sort_index().dropna()
    w_level = level_s.resample(week_rule).last()
    w_ret = np.log(w_level).diff()
    return w_ret.replace([np.inf, -np.inf], np.nan).dropna()


def weekly_pct_change(level_s: pd.Series, week_rule: str = "W-FRI") -> pd.Series:
    level_s = level_s.sort_index().dropna()
    w_level = level_s.resample(week_rule).last()
    w_ret = w_level.pct_change()
    return w_ret.replace([np.inf, -np.inf], np.nan).dropna()


def weekly_delta(level_s: pd.Series, week_rule: str = "W-FRI") -> pd.Series:
    level_s = level_s.sort_index().dropna()
    w_level = level_s.resample(week_rule).last()
    w_d = w_level.diff()
    return w_d.replace([np.inf, -np.inf], np.nan).dropna()


def zscore(s: pd.Series) -> pd.Series:
    s = s.dropna()
    mu = float(s.mean())
    sd = float(s.std(ddof=0))
    if sd == 0 or not np.isfinite(sd):
        return s * np.nan
    return (s - mu) / sd


def rolling_zscore(s: pd.Series, window: int = 104, min_periods: int = 26) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std(ddof=0).replace(0.0, np.nan)
    z = (s - mu) / sd
    if z.dropna().empty:
        z = zscore(s)
    return z


def decode_csv_bytes(file_bytes: bytes) -> str:
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "cp949", "latin1"):
        try:
            return file_bytes.decode(enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Unable to decode uploaded CSV bytes: {last_err}")


def detect_date_column(df: pd.DataFrame) -> Optional[str]:
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for key in ["date", "datetime", "time", "timestamp", "week", "dt"]:
        if key in normalized:
            return normalized[key]
    return df.columns[0] if len(df.columns) else None


def detect_value_column(df: pd.DataFrame, preferred: List[str]) -> Optional[str]:
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for key in preferred:
        if key in normalized:
            return normalized[key]
    for c in df.columns:
        if str(c).strip().lower() not in {"date", "datetime", "time", "timestamp", "week", "dt"}:
            if pd.api.types.is_numeric_dtype(df[c]):
                return c
    for c in df.columns:
        if str(c).strip().lower() not in {"date", "datetime", "time", "timestamp", "week", "dt"}:
            return c
    return None


def parse_datetime_series(raw: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(raw):
        dt_ms = pd.to_datetime(raw, unit="ms", errors="coerce")
        if dt_ms.notna().mean() >= 0.5:
            return dt_ms
        return pd.to_datetime(raw, unit="s", errors="coerce")
    return pd.to_datetime(raw, errors="coerce")


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def load_uploaded_mvrv_weekly(file_bytes: Optional[bytes], file_name: Optional[str], week_rule: str = WEEK_RULE) -> pd.Series:
    if not file_bytes:
        return pd.Series(dtype=float, name="mvrv_z")
    csv_text = decode_csv_bytes(file_bytes)
    df = pd.read_csv(io.StringIO(csv_text))
    if df is None or df.empty:
        return pd.Series(dtype=float, name="mvrv_z")

    date_col = detect_date_column(df)
    value_col = detect_value_column(
        df,
        preferred=[
            "mvrv_z", "mvrv z", "mvrv_zscore", "mvrv z-score", "mvrv_z_score",
            "zscore", "z_score", "value", "mvrv"
        ],
    )
    if date_col is None or value_col is None:
        raise RuntimeError(f"MVRV CSV parsing failed: date_col={date_col}, value_col={value_col}, file={file_name}")

    dt = parse_datetime_series(df[date_col])
    vals = pd.to_numeric(df[value_col], errors="coerce")
    s = pd.Series(vals.values, index=dt).dropna()
    s = s[~s.index.isna()]
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s = s.resample(week_rule).last()
    s.name = "mvrv_z"
    return s.dropna()


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fetch_binance_funding_history(symbol: str = "BTCUSDT", start_date: str = START_DATE) -> pd.Series:
    base_url = "https://fapi.binance.com/fapi/v1/fundingRate"
    start_ms = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
    out_rows = []
    cursor = start_ms

    while True:
        params = {"symbol": symbol.upper(), "startTime": cursor, "limit": 1000}
        r = requests.get(base_url, params=params, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        rows = r.json()
        if not rows:
            break
        out_rows.extend(rows)
        last_ms = int(rows[-1]["fundingTime"])
        if len(rows) < 1000:
            break
        cursor = last_ms + 1

    if not out_rows:
        return pd.Series(dtype=float, name="funding_rate")

    df = pd.DataFrame(out_rows)
    df["fundingTime"] = pd.to_datetime(pd.to_numeric(df["fundingTime"], errors="coerce"), unit="ms", utc=True).dt.tz_localize(None)
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    s = pd.Series(df["fundingRate"].values, index=df["fundingTime"]).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s.name = "funding_rate"
    return s.dropna()


def build_funding_weekly_zscore(
    funding_raw: Optional[pd.Series],
    week_rule: str = WEEK_RULE,
    ma_weeks: int = 8,
    z_window: int = 104,
) -> Tuple[pd.Series, pd.Series]:
    if funding_raw is None or len(funding_raw) == 0:
        return pd.Series(dtype=float, name="funding_rate_w"), pd.Series(dtype=float, name="funding_z")
    f = funding_raw.sort_index().dropna()
    funding_w = f.resample(week_rule).mean()
    funding_ma = funding_w.rolling(ma_weeks, min_periods=max(3, ma_weeks // 2)).mean()
    funding_z = rolling_zscore(funding_ma, window=z_window, min_periods=max(26, ma_weeks * 2))
    funding_w.name = "funding_rate_w"
    funding_z.name = "funding_z"
    return funding_w, funding_z


def load_alpha_inputs_weekly(
    mvrv_file_bytes: Optional[bytes],
    mvrv_file_name: Optional[str],
    use_binance_funding: bool,
    funding_symbol: str,
    week_rule: str = WEEK_RULE,
) -> Dict[str, object]:
    mvrv_z = load_uploaded_mvrv_weekly(mvrv_file_bytes, mvrv_file_name, week_rule=week_rule)
    funding_raw = pd.Series(dtype=float, name="funding_rate")
    funding_w = pd.Series(dtype=float, name="funding_rate_w")
    funding_z = pd.Series(dtype=float, name="funding_z")
    funding_error = None

    if use_binance_funding:
        try:
            funding_raw = fetch_binance_funding_history(symbol=funding_symbol, start_date=START_DATE)
            funding_w, funding_z = build_funding_weekly_zscore(funding_raw, week_rule=week_rule)
        except Exception as e:
            funding_error = str(e)

    return {
        "mvrv_z": mvrv_z,
        "funding_raw": funding_raw,
        "funding_rate_w": funding_w,
        "funding_z": funding_z,
        "meta": {
            "mvrv_loaded": bool(not mvrv_z.dropna().empty),
            "mvrv_source": mvrv_file_name or "none",
            "funding_loaded": bool(not funding_z.dropna().empty),
            "funding_symbol": funding_symbol if use_binance_funding else "disabled",
            "funding_source": (f"binance:{funding_symbol.upper()}" if use_binance_funding else "disabled"),
            "funding_error": funding_error,
        },
    }


# =========================
# Stats helpers
# =========================
def _mask_arrays(y: np.ndarray, x: np.ndarray):
    m = np.isfinite(y) & np.isfinite(x)
    return y[m], x[m], m


def corr_fast(y: np.ndarray, x: np.ndarray, min_obs: int) -> float:
    yy, xx, _ = _mask_arrays(y, x)
    if yy.size < min_obs:
        return np.nan
    y0 = yy - yy.mean()
    x0 = xx - xx.mean()
    denom = np.sqrt(np.sum(y0 * y0) * np.sum(x0 * x0))
    if denom == 0:
        return np.nan
    return float(np.sum(y0 * x0) / denom)


def beta_and_tstat_simple(y: np.ndarray, x: np.ndarray, min_obs: int):
    yy, xx, _ = _mask_arrays(y, x)
    n = yy.size
    if n < max(min_obs, 3):
        return np.nan, np.nan

    xbar = xx.mean()
    ybar = yy.mean()
    x0 = xx - xbar
    y0 = yy - ybar

    sxx = np.sum(x0 * x0)
    if sxx == 0:
        return np.nan, np.nan

    beta = float(np.sum(x0 * y0) / sxx)
    alpha = float(ybar - beta * xbar)

    resid = yy - (alpha + beta * xx)
    sse = np.sum(resid * resid)
    if n <= 2:
        return beta, np.nan

    sigma2 = sse / (n - 2)
    se_beta = np.sqrt(sigma2 / sxx) if sigma2 >= 0 else np.nan
    if not np.isfinite(se_beta) or se_beta == 0:
        return beta, np.nan

    t_stat = float(beta / se_beta)
    return beta, t_stat


def ols_2var_with_tstats(y: np.ndarray, x1: np.ndarray, x2: np.ndarray, min_obs: int):
    """
    Multivariate OLS with intercept: y = a + b1*x1 + b2*x2 + e
    Returns dict: a,b1,b2,t1,t2,R2,n,sigma
    """
    m = np.isfinite(y) & np.isfinite(x1) & np.isfinite(x2)
    yy = y[m]
    X1 = x1[m]
    X2 = x2[m]
    n = yy.size
    if n < max(min_obs, 5):
        return {"a": np.nan, "b1": np.nan, "b2": np.nan, "t1": np.nan, "t2": np.nan, "R2": np.nan, "n": n, "sigma": np.nan}

    X = np.column_stack([np.ones(n), X1, X2])
    XtX = X.T @ X
    XtY = X.T @ yy
    XtX_inv = np.linalg.pinv(XtX)
    b = XtX_inv @ XtY

    yhat = X @ b
    resid = yy - yhat
    sse = float(resid.T @ resid)
    sst = float(((yy - yy.mean()) ** 2).sum())
    R2 = 1.0 - (sse / sst) if sst > 0 else np.nan

    k = 3
    dof = n - k
    if dof <= 0:
        return {"a": float(b[0]), "b1": float(b[1]), "b2": float(b[2]), "t1": np.nan, "t2": np.nan, "R2": R2, "n": n, "sigma": np.nan}

    sigma2 = sse / dof
    sigma = float(np.sqrt(sigma2)) if (np.isfinite(sigma2) and sigma2 >= 0) else np.nan

    cov = sigma2 * XtX_inv
    se = np.sqrt(np.diag(cov))
    t1 = float(b[1] / se[1]) if np.isfinite(se[1]) and se[1] != 0 else np.nan
    t2 = float(b[2] / se[2]) if np.isfinite(se[2]) and se[2] != 0 else np.nan

    return {"a": float(b[0]), "b1": float(b[1]), "b2": float(b[2]), "t1": t1, "t2": t2, "R2": R2, "n": n, "sigma": sigma}


def smooth_series(s: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return s.astype(float)
    return s.astype(float).rolling(window, min_periods=max(2, window // 2)).mean()


# =========================
# Rolling computation (1D)
# =========================
def rolling_maps_weekly(y_w: pd.Series, x_w: pd.Series, invert_x: bool, lags_weeks: range, quiet: bool = False):
    df = pd.concat([y_w.rename("y"), x_w.rename("x")], axis=1).dropna()
    df = df.loc[START_DATE:END_DATE]

    if invert_x:
        df["x"] = -df["x"]

    idx = df.index
    max_lag = max(lags_weeks)
    if len(idx) < WINDOW_WEEKS + max_lag + 5:
        raise RuntimeError("ë°ì´í° ê¸¸ì´ê° ë¶ì¡±í©ëë¤. (ê¸°ê°/ìëì°/lag ì¬ê²í  íì)")

    end_positions = list(range(WINDOW_WEEKS, len(idx), STEP_WEEKS))
    end_dates = idx[end_positions]

    corr_raw = pd.DataFrame(index=end_dates, columns=list(lags_weeks), dtype=float)
    corr_masked = pd.DataFrame(index=end_dates, columns=list(lags_weeks), dtype=float)
    beta_map = pd.DataFrame(index=end_dates, columns=list(lags_weeks), dtype=float)
    tstat_map = pd.DataFrame(index=end_dates, columns=list(lags_weeks), dtype=float)
    tstat_masked = pd.DataFrame(index=end_dates, columns=list(lags_weeks), dtype=float)

    y_all = df["y"]
    x_all = df["x"]
    x_shifted = {L: x_all.shift(L) for L in lags_weeks}

    prog = None
    if not quiet:
        prog = st.progress(0)

    for i, end_dt in enumerate(end_dates):
        w_idx = df.loc[:end_dt].tail(WINDOW_WEEKS).index
        y_win = y_all.loc[w_idx].to_numpy(dtype=float)

        for L in lags_weeks:
            x_win = x_shifted[L].loc[w_idx].to_numpy(dtype=float)

            c = corr_fast(y_win, x_win, MIN_OBS)
            corr_raw.loc[end_dt, L] = c

            c_vis = c
            if np.isfinite(c_vis) and abs(c_vis) < CORR_HEATMAP_MASK_ABS_BELOW:
                c_vis = np.nan
            corr_masked.loc[end_dt, L] = c_vis

            b, t = beta_and_tstat_simple(y_win, x_win, MIN_OBS)
            beta_map.loc[end_dt, L] = b
            tstat_map.loc[end_dt, L] = t

            t_vis = t
            if np.isfinite(t_vis) and abs(t_vis) < TSTAT_HEATMAP_MASK_ABS_BELOW:
                t_vis = np.nan
            tstat_masked.loc[end_dt, L] = t_vis

        if prog is not None:
            prog.progress((i + 1) / len(end_dates))

    if prog is not None:
        prog.empty()

    best_lag_raw = corr_raw.idxmax(axis=1)
    best_corr_raw = corr_raw.max(axis=1)
    max_abs_corr = corr_raw.abs().max(axis=1)

    best_lag_valid = best_lag_raw.copy()
    best_lag_valid[max_abs_corr < VALID_BEST_ABS_CORR_BELOW] = np.nan
    best_lag_smooth = smooth_series(best_lag_valid, BEST_LAG_SMOOTH_WEEKS)

    out = pd.DataFrame({
        "best_lag_raw_weeks": best_lag_raw,
        "best_corr_raw": best_corr_raw,
        "max_abs_corr": max_abs_corr,
        "best_lag_valid_weeks": best_lag_valid,
        "best_lag_valid_smooth_weeks": best_lag_smooth,
    })

    return corr_raw, corr_masked, beta_map, tstat_map, tstat_masked, out


# =========================
# TAB3 2D lag-pair + multivariate rolling
# =========================
def rolling_best_pair_and_multivar(
    y: pd.Series,
    liq: pd.Series,
    dxy_ret: pd.Series,
    lags_weeks: range,
    window_weeks: int,
    step_weeks: int,
):
    base = pd.concat([y.rename("y"), liq.rename("liq"), dxy_ret.rename("dxy")], axis=1).dropna()
    base = base.loc[START_DATE:END_DATE]

    idx = base.index
    max_lag = max(lags_weeks)
    if len(idx) < window_weeks + max_lag + 5:
        raise RuntimeError("2D ë¶ìì ìí ë°ì´í° ê¸¸ì´ê° ë¶ì¡±í©ëë¤.")

    end_positions = list(range(window_weeks, len(idx), step_weeks))
    end_dates = idx[end_positions]

    lags_list = list(lags_weeks)
    nL = len(lags_list)
    pair_counts = np.zeros((nL, nL), dtype=int)

    rows = []
    y_all = base["y"]
    liq_all = base["liq"]
    dxy_all = base["dxy"]

    liq_shifted = {L: liq_all.shift(L) for L in lags_list}
    dxy_inv_shifted = {L: (-dxy_all).shift(L) for L in lags_list}

    prog = st.progress(0)
    for i, end_dt in enumerate(end_dates):
        w_idx = base.loc[:end_dt].tail(window_weeks).index
        y_win = y_all.loc[w_idx].to_numpy(dtype=float)

        z_liq = {}
        z_dxy = {}
        for L in lags_list:
            a = liq_shifted[L].loc[w_idx].to_numpy(dtype=float)
            b = dxy_inv_shifted[L].loc[w_idx].to_numpy(dtype=float)

            yy, aa, _ = _mask_arrays(y_win, a)
            if yy.size >= MIN_OBS:
                mu = float(np.mean(aa))
                sd = float(np.std(aa, ddof=0))
                z = (a - mu) / sd if (sd > 0 and np.isfinite(sd)) else (a * np.nan)
            else:
                z = a * np.nan
            z_liq[L] = z

            yy2, bb, _ = _mask_arrays(y_win, b)
            if yy2.size >= MIN_OBS:
                mu2 = float(np.mean(bb))
                sd2 = float(np.std(bb, ddof=0))
                z2 = (b - mu2) / sd2 if (sd2 > 0 and np.isfinite(sd2)) else (b * np.nan)
            else:
                z2 = b * np.nan
            z_dxy[L] = z2

        best_corr = -np.inf
        best_L_liq = None
        best_L_dxy = None

        for L1 in lags_list:
            x1 = z_liq[L1]
            for L2 in lags_list:
                x2 = z_dxy[L2]
                x = x1 + x2
                ccorr = corr_fast(y_win, x, MIN_OBS)
                if np.isfinite(ccorr) and ccorr > best_corr:
                    best_corr = ccorr
                    best_L_liq = L1
                    best_L_dxy = L2

        valid = bool(np.isfinite(best_corr) and abs(best_corr) >= VALID_BEST_ABS_CORR_BELOW)
        if valid and best_L_liq is not None and best_L_dxy is not None:
            pair_counts[lags_list.index(best_L_liq), lags_list.index(best_L_dxy)] += 1

        if best_L_liq is not None and best_L_dxy is not None:
            x1_raw = liq_shifted[best_L_liq].loc[w_idx].to_numpy(dtype=float)
            x2_raw = dxy_inv_shifted[best_L_dxy].loc[w_idx].to_numpy(dtype=float)
            mv = ols_2var_with_tstats(y_win, x1_raw, x2_raw, MIN_OBS)
        else:
            mv = {"a": np.nan, "b1": np.nan, "b2": np.nan, "t1": np.nan, "t2": np.nan, "R2": np.nan, "n": 0, "sigma": np.nan}

        rows.append({
            "end_date": end_dt,
            "best_L_liq": best_L_liq,
            "best_L_dxy": best_L_dxy,
            "best_corr": float(best_corr) if np.isfinite(best_corr) else np.nan,
            "valid": valid,
            "a": mv["a"], "b_liq": mv["b1"], "b_dxyinv": mv["b2"],
            "t_liq": mv["t1"], "t_dxyinv": mv["t2"],
            "R2": mv["R2"], "sigma": mv["sigma"], "n": mv["n"],
        })

        prog.progress((i + 1) / len(end_dates))
    prog.empty()

    best_df = pd.DataFrame(rows).set_index("end_date")
    best_df["best_L_liq_smooth"] = smooth_series(best_df["best_L_liq"], BEST_LAG_SMOOTH_WEEKS)
    best_df["best_L_dxy_smooth"] = smooth_series(best_df["best_L_dxy"], BEST_LAG_SMOOTH_WEEKS)

    return best_df, pair_counts, lags_list


# =========================
# Plotting helpers
# =========================
def plot_heatmap_with_overlay(mat: pd.DataFrame, overlay_y: pd.Series, title: str, ylabel: str):
    z = mat.to_numpy(dtype=float).T
    t = mat.index
    lags = mat.columns.astype(int)

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(
        z,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[0, len(t) - 1, lags.min(), lags.max()],
    )

    tick_n = min(10, len(t))
    tick_pos = np.linspace(0, len(t) - 1, tick_n).astype(int)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([t[i].strftime("%Y-%m") for i in tick_pos])

    ax.set_title(title)
    ax.set_xlabel("Window end date")
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax, label="Value")

    ov = overlay_y.reindex(t)
    ax.plot(np.arange(len(t)), ov.to_numpy(dtype=float), linewidth=2.0, label="best_lag (valid+smoothed)")
    ax.legend(loc="upper right")

    fig.tight_layout()
    return fig


def plot_pair_count_heatmap(pair_counts: np.ndarray, lags_list: List[int], title: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pair_counts, origin="lower", aspect="auto", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("DXY lag (weeks)")
    ax.set_ylabel("Liquidity lag (weeks)")

    ax.set_xticks(range(len(lags_list)))
    ax.set_yticks(range(len(lags_list)))
    ax.set_xticklabels(lags_list, rotation=90, fontsize=8)
    ax.set_yticklabels(lags_list, fontsize=8)

    fig.colorbar(im, ax=ax, label="Count")
    fig.tight_layout()
    return fig


def _shade_segments(ax, dates: pd.DatetimeIndex, mask: np.ndarray, alpha: float = 0.10, hatch: Optional[str] = "////"):
    if mask is None or len(mask) == 0 or len(dates) != len(mask):
        return

    in_seg = False
    seg_start = None
    for i in range(len(dates)):
        if bool(mask[i]) and not in_seg:
            in_seg = True
            seg_start = dates[i]
        if in_seg and (not bool(mask[i]) or i == len(dates) - 1):
            seg_end = dates[i]
            ax.axvspan(seg_start, seg_end, color="gray", alpha=alpha, hatch=hatch)
            in_seg = False
            seg_start = None



def plot_main_overlay_with_predline(
    dates,
    btc_price,
    ldli_level,
    ldli_liq,
    ldli_dxy,
    pred_hist_endpoint: Optional[pd.Series],
    pred_path_latest: Optional[pd.Series],
    spaghetti_paths: Optional[List[pd.Series]],
    title,
    btc_end,
    conflict_mask=None,
    y2_label="LDLI components (shifted)",
    pred_path_latest_adj: Optional[pd.Series] = None,
):
    fig, ax1 = plt.subplots(figsize=(14, 8.0))

    ax1.plot(pd.DatetimeIndex(dates), btc_price, label="BTC Price", linewidth=2.8, color="tab:blue")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("BTC Price")
    ax1.grid(True, alpha=0.25)

    dmin = pd.DatetimeIndex(dates).min()
    dmax = pd.DatetimeIndex(dates).max()

    if spaghetti_paths:
        for ps in spaghetti_paths:
            if ps is None or ps.dropna().empty:
                continue
            psw = ps.loc[(ps.index >= dmin) & (ps.index <= dmax)]
            if psw.dropna().empty:
                continue
            ax1.plot(
                psw.index, psw.values,
                linewidth=float(SPAGHETTI_LW),
                color="tab:green",
                alpha=float(SPAGHETTI_ALPHA),
            )

    if pred_hist_endpoint is not None and not pred_hist_endpoint.dropna().empty:
        s = pred_hist_endpoint.loc[(pred_hist_endpoint.index >= dmin) & (pred_hist_endpoint.index <= dmax)]
        if not s.dropna().empty:
            ax1.plot(
                s.index, s.values,
                linewidth=2.2, linestyle="--",
                label="Predicted BTC (true-history endpoints)",
                color="tab:green", alpha=0.85
            )

    if pred_path_latest is not None and not pred_path_latest.dropna().empty:
        s2 = pred_path_latest.loc[(pred_path_latest.index >= dmin) & (pred_path_latest.index <= dmax)]
        if not s2.dropna().empty:
            ax1.plot(
                s2.index, s2.values,
                linewidth=2.6, linestyle="--",
                label="Latest forecast path (raw)",
                color="tab:blue", alpha=0.55
            )

    if pred_path_latest_adj is not None and not pred_path_latest_adj.dropna().empty:
        s3 = pred_path_latest_adj.loc[(pred_path_latest_adj.index >= dmin) & (pred_path_latest_adj.index <= dmax)]
        if not s3.dropna().empty:
            ax1.plot(
                s3.index, s3.values,
                linewidth=2.6, linestyle=":",
                label="Latest forecast path (adjusted)",
                color="tab:red", alpha=0.9
            )

    if conflict_mask is not None:
        _shade_segments(ax1, dates=pd.DatetimeIndex(dates), mask=np.asarray(conflict_mask, dtype=bool), alpha=0.10, hatch="////")

    if btc_end is not None and len(dates) > 0:
        ax1.axvspan(btc_end, pd.DatetimeIndex(dates).max(), color="gray", alpha=0.12)

    ax2 = ax1.twinx()
    ax2.plot(pd.DatetimeIndex(dates), ldli_level, label="LDLI Total (shifted)", linewidth=2.8, color="tab:orange", alpha=0.95)
    ax2.plot(pd.DatetimeIndex(dates), ldli_liq, label="LDLI Liquidity contrib (shifted)", linewidth=2.0, color="tab:purple", alpha=0.75)
    ax2.plot(pd.DatetimeIndex(dates), ldli_dxy, label="LDLI DXY contrib (shifted)", linewidth=2.0, color="tab:red", alpha=0.75)
    ax2.set_ylabel(y2_label)

    ax1.set_title(title)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")
    fig.tight_layout()
    return fig



# =========================
# Summary blocks
# =========================
def summarize_block(out: pd.DataFrame, tstat_masked: pd.DataFrame):
    valid_ratio = float(np.mean(np.isfinite(out["best_lag_valid_weeks"].values)))
    med_max_abs = float(np.nanmedian(out["max_abs_corr"].values))
    med_best = float(np.nanmedian(out["best_corr_raw"].values))
    sig_cells_ratio = float(np.isfinite(tstat_masked.to_numpy(dtype=float)).mean())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ì í¨ best ë¹ì¨", f"{valid_ratio:.1%}")
    c2.metric("max(|corr|) ì¤ìê°", f"{med_max_abs:.4f}")
    c3.metric("best_corr ì¤ìê°", f"{med_best:.4f}")
    c4.metric("ì ì t-stat ì ë¹ì¨(|t|â¥2)", f"{sig_cells_ratio:.1%}")


def render_model_section(label: str, corr_masked, beta_map, tstat_masked, out):
    st.markdown(f"## {label}")
    summarize_block(out, tstat_masked)

    st.markdown("### 1) Corr heatmap (masked) + best lag overlay")
    st.pyplot(plot_heatmap_with_overlay(
        corr_masked,
        out["best_lag_valid_smooth_weeks"],
        title="corr heatmap (masked) + best lag overlay",
        ylabel="Lag (weeks)"
    ))

    st.markdown("### 2) Beta heatmap + best lag overlay")
    st.pyplot(plot_heatmap_with_overlay(
        beta_map,
        out["best_lag_valid_smooth_weeks"],
        title="beta heatmap + best lag overlay",
        ylabel="Lag (weeks)"
    ))

    st.markdown("### 3) t-stat(beta) heatmap (masked |t|<2) + best lag overlay")
    st.pyplot(plot_heatmap_with_overlay(
        tstat_masked,
        out["best_lag_valid_smooth_weeks"],
        title="t-stat(beta) heatmap (masked |t|<2) + best lag overlay",
        ylabel="Lag (weeks)"
    ))

    st.markdown("### 4) Output table")
    st.dataframe(out.tail(30))


# =========================
# Common loaders
# =========================
@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def load_asset_close(asset_key: str) -> Tuple[pd.Series, str]:
    yahoo_candidates = ASSET_SYMBOLS_YAHOO.get(asset_key, [])
    stooq_candidates = ASSET_SYMBOLS_STOOQ.get(asset_key, [])
    if not yahoo_candidates and not stooq_candidates:
        raise RuntimeError(f"Unknown asset_key={asset_key}")

    s, used = fetch_price_daily_multi(yahoo_candidates, stooq_candidates)
    return s.loc[START_DATE:END_DATE], used


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def load_btc_close() -> pd.Series:
    s, _used = fetch_price_daily_multi([YAHOO_BTC], [STOOQ_BTC, "btc.v"])
    return s.loc[START_DATE:END_DATE]


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def load_dxy_close() -> Tuple[pd.Series, str]:
    s, used = fetch_price_daily_multi([YAHOO_DXY], [STOOQ_DXY_PRIMARY, STOOQ_DXY_FALLBACK])
    return s.loc[START_DATE:END_DATE], used


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def load_netliquidity_daily_millions():
    walcl = fetch_fred_fredgraph(FRED_WALCL).loc[START_DATE:END_DATE]  # Millions
    tga = fetch_fred_fredgraph(FRED_TGA).loc[START_DATE:END_DATE]      # Millions
    rrp = fetch_fred_fredgraph(FRED_RRP).loc[START_DATE:END_DATE]      # Billions daily

    rrp_m = rrp * 1000.0
    daily_idx = pd.date_range(start=START_DATE, end=END_DATE, freq="D")

    walcl_d = walcl.reindex(daily_idx).ffill()
    tga_d = tga.reindex(daily_idx).ffill()
    rrp_d = rrp_m.reindex(daily_idx).ffill()

    netliq = (walcl_d - tga_d - rrp_d).dropna()
    netliq.name = "net_liquidity_millions"

    snapshot = pd.concat(
        [walcl_d.rename("WALCL(M)"), tga_d.rename("TGA(M)"), rrp_d.rename("RRP(M)"), netliq],
        axis=1
    )
    return netliq, snapshot


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def load_g2_m2_usd_daily():
    """
    Build G2 = US + EA in USD.

    - US M2SL: billions USD (levels), typically weekly/monthly -> ffill to daily
    - EA M2: ECB BSI series (units vary; typically millions EUR). We treat as "EUR units" and convert to USD using EURUSD.
      The absolute unit scale doesn't matter for return-based computations, but for DELTA the unit matters (still okay if consistent).
    - FX: EURUSD daily (ECB EXR). We try both orientations; if series is "EUR per USD" we invert.

    Output:
      g2_usd (float, level), snapshot_df (columns: us_m2, ea_m2_eur, eurusd, ea_m2_usd, g2_m2_usd)
    """
    us_m2 = fetch_fred_fredgraph(FRED_US_M2SL)  # Billions USD
    us_m2.name = "US_M2SL_billions_usd"

    # ECB EA monetary aggregate
    ea_m2_eur = fetch_ecb_series("BSI", ECB_EA_M2_KEY)
    ea_m2_eur.name = "EA_M2_EUR_raw"

    # FX (try orientation)
    eurusd = None
    fx_err = None
    for k in [ECB_EURUSD_KEY, ECB_EURUSD_KEY_FALLBACK]:
        try:
            s = fetch_ecb_series("EXR", k)
            eurusd = s
            eurusd.name = f"ECB_EXR_{k}"
            break
        except Exception as e:
            fx_err = e
            continue
    if eurusd is None:
        raise RuntimeError(f"ECB FX fetch failed for both keys. Last error={fx_err}")

    # Determine orientation: we want USD per 1 EUR (EURUSD ~ 1.0~1.5)
    # If series looks like ~0.6~1.2 and is likely EUR per USD, invert.
    med = float(np.nanmedian(eurusd.values)) if len(eurusd) > 0 else np.nan
    if np.isfinite(med) and med < 0.9:  # heuristic
        eurusd = 1.0 / eurusd
        eurusd.name = eurusd.name + "_INVERTED"

    # align to daily
    daily_idx = pd.date_range(start=START_DATE, end=END_DATE, freq="D")
    us_d = us_m2.reindex(daily_idx).ffill()
    ea_d = ea_m2_eur.reindex(daily_idx).ffill()
    fx_d = eurusd.reindex(daily_idx).ffill()

    ea_usd = (ea_d.astype(float) * fx_d.astype(float)).rename("EA_M2_USD_raw")
    g2 = (us_d.astype(float) + ea_usd.astype(float)).rename("G2_M2_USD_level")

    snap = pd.concat([us_d, ea_d, fx_d.rename("EURUSD"), ea_usd, g2], axis=1).dropna()
    return g2.dropna(), snap


def load_liquidity_source_daily(liq_source: str):
    """
    Returns:
      liq_level_daily, snapshot_df, liq_label, unit_label
    """
    if liq_source == "Fed Net Liquidity (FRED)":
        s, snap = load_netliquidity_daily_millions()
        return s, snap, "Fed Net Liquidity (FRED)", "Millions USD"
    if liq_source == "G2 M2 (US+EA, USD)":
        s, snap = load_g2_m2_usd_daily()
        return s, snap, "G2 M2 (US+EA, USD)", "Mixed (US billions + EA converted); level scale varies"
    raise RuntimeError(f"Unknown liquidity source: {liq_source}")


# =========================
# CORE: MAIN/TAB4ê° TAB1~3 ì í ì¤í ìì´ë ëìíëë¡
# =========================
def ensure_tab3_state_ready(base: pd.DataFrame, combo_ret: pd.Series, combo_dlt: pd.Series):
    xx = st.session_state.get("tab3_xx_weeks", None)
    chosen = st.session_state.get("tab3_chosen_combo", None)
    best_corr_last = st.session_state.get("tab3_best_corr_last", None)

    if xx is not None and chosen is not None and best_corr_last is not None:
        return int(xx), str(chosen), float(best_corr_last)

    with st.spinner("MAIN: TAB3 ê°ì´ ìì´ ìµì  lag(xx) ìë ì°ì¶ ì¤..."):
        y = base["btc_wret"]
        lags_weeks = range(LAG_MIN_WEEKS, LAG_MAX_WEEKS + 1)

        _, _, _, _, _, out_r = rolling_maps_weekly(y_w=y, x_w=combo_ret, invert_x=False, lags_weeks=lags_weeks, quiet=True)
        _, _, _, _, _, out_d = rolling_maps_weekly(y_w=y, x_w=combo_dlt, invert_x=False, lags_weeks=lags_weeks, quiet=True)

        last_r = out_r.dropna(subset=["best_corr_raw"]).iloc[-1]
        last_d = out_d.dropna(subset=["best_corr_raw"]).iloc[-1]
        if float(last_r["best_corr_raw"]) >= float(last_d["best_corr_raw"]):
            chosen = "RET"
            out_use = out_r
            best_corr_last = float(last_r["best_corr_raw"])
        else:
            chosen = "DELTA"
            out_use = out_d
            best_corr_last = float(last_d["best_corr_raw"])

        out_use2 = out_use.dropna(subset=["best_lag_raw_weeks"])
        lag_series = out_use2["best_lag_valid_smooth_weeks"]
        lag_latest = lag_series.dropna().iloc[-1] if lag_series.dropna().size > 0 else out_use2["best_lag_raw_weeks"].iloc[-1]
        xx = int(np.clip(int(round(float(lag_latest))), LAG_MIN_WEEKS, LAG_MAX_WEEKS))

    st.session_state["tab3_chosen_combo"] = chosen
    st.session_state["tab3_xx_weeks"] = xx
    st.session_state["tab3_best_corr_last"] = best_corr_last

    return int(xx), str(chosen), float(best_corr_last)


# =========================
# Regime v2 (A + B)
# =========================
def _shift_weeks_index(s: pd.Series, weeks: int, new_name: str):
    out = s.copy()
    out.index = out.index + pd.to_timedelta(int(weeks) * 7, unit="D")
    out.name = new_name
    return out


def label_regime_v2(
    z_liq: pd.Series,
    z_dxy: pd.Series,
    smooth_weeks: int = 3,
    dir_lookback_weeks: int = 4,
    deadband: float = 0.2,
):
    df = pd.concat([z_liq.rename("liq"), z_dxy.rename("dxy")], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype="object"), pd.DataFrame()

    if smooth_weeks and smooth_weeks > 1:
        df["liq"] = df["liq"].rolling(smooth_weeks, min_periods=max(2, smooth_weeks // 2)).mean()
        df["dxy"] = df["dxy"].rolling(smooth_weeks, min_periods=max(2, smooth_weeks // 2)).mean()

    k = max(1, int(dir_lookback_weeks))
    dir_liq = df["liq"].rolling(k, min_periods=max(2, k // 2)).sum()
    dir_dxy = df["dxy"].rolling(k, min_periods=max(2, k // 2)).sum()

    if deadband is not None and deadband > 0:
        dir_liq = dir_liq.where(dir_liq.abs() >= deadband, other=0.0)
        dir_dxy = dir_dxy.where(dir_dxy.abs() >= deadband, other=0.0)

    liq_pos = dir_liq > 0
    liq_neg = dir_liq < 0
    dxy_pos = dir_dxy > 0
    dxy_neg = dir_dxy < 0
    neutral = (dir_liq == 0) | (dir_dxy == 0)

    regime = pd.Series(index=df.index, dtype="object")
    regime[liq_pos & dxy_pos] = "TAILWIND"
    regime[liq_neg & dxy_neg] = "HEADWIND"
    regime[(liq_pos & dxy_neg) | (liq_neg & dxy_pos)] = "CONFLICT"
    regime[neutral] = "NEUTRAL"
    regime = regime.fillna("NEUTRAL")

    diag = pd.DataFrame({
        "z_liq_s": df["liq"],
        "z_dxy_s": df["dxy"],
        "dir_liq_sum": dir_liq,
        "dir_dxy_sum": dir_dxy,
        "regime": regime
    })

    return regime, diag



def add_recent_slice_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        out["recent_52w_flag"] = pd.Series(dtype="int64")
        out["recent_104w_flag"] = pd.Series(dtype="int64")
        return out
    last_dt = pd.DatetimeIndex(out.index).max()
    out["recent_52w_flag"] = (pd.DatetimeIndex(out.index) >= (last_dt - pd.Timedelta(weeks=52))).astype(int)
    out["recent_104w_flag"] = (pd.DatetimeIndex(out.index) >= (last_dt - pd.Timedelta(weeks=104))).astype(int)
    return out


def add_horizon_targets(
    df: pd.DataFrame,
    px_col: str = "btc_close",
    horizon_w: int = H19_DIAG_W,
) -> pd.DataFrame:
    out = df.copy()
    px = pd.to_numeric(out[px_col], errors="coerce")
    out[f"realized_fwd_px_{horizon_w}w"] = px.shift(-horizon_w)
    out[f"realized_fwd_ret_{horizon_w}w"] = np.log(out[f"realized_fwd_px_{horizon_w}w"] / px)
    out[f"realized_sign_{horizon_w}w"] = np.sign(out[f"realized_fwd_ret_{horizon_w}w"])
    return out


def classify_mvrv_state(mvrv_z: pd.Series) -> pd.Series:
    if mvrv_z is None or len(mvrv_z) == 0:
        return pd.Series(dtype="object")
    s = pd.to_numeric(mvrv_z, errors="coerce")
    out = pd.Series(index=s.index, dtype="object")
    out[s <= 0.5] = "alpha_bull"
    out[(s > 0.5) & (s < 3.0)] = "alpha_neutral"
    out[s >= 3.0] = "alpha_bear"
    return out.fillna("alpha_neutral")


def classify_funding_state(funding_z: pd.Series) -> pd.Series:
    if funding_z is None or len(funding_z) == 0:
        return pd.Series(dtype="object")
    s = pd.to_numeric(funding_z, errors="coerce")
    out = pd.Series(index=s.index, dtype="object")
    out[s <= -1.5] = "alpha_bull"
    out[(s > -1.5) & (s < 1.5)] = "alpha_neutral"
    out[s >= 1.5] = "alpha_bear"
    return out.fillna("alpha_neutral")


def compute_alpha_state(
    mvrv_state: Optional[pd.Series],
    funding_state: Optional[pd.Series],
    index: Optional[pd.Index] = None,
) -> pd.Series:
    if mvrv_state is None and funding_state is None:
        return pd.Series("alpha_neutral", index=index, dtype="object")
    if mvrv_state is None:
        return funding_state.fillna("alpha_neutral").astype("object")
    if funding_state is None:
        return mvrv_state.fillna("alpha_neutral").astype("object")
    idx = mvrv_state.index.union(funding_state.index)
    m = mvrv_state.reindex(idx).fillna("alpha_neutral")
    f = funding_state.reindex(idx).fillna("alpha_neutral")
    score = (
        (m == "alpha_bull").astype(int) - (m == "alpha_bear").astype(int) +
        (f == "alpha_bull").astype(int) - (f == "alpha_bear").astype(int)
    )
    out = pd.Series(index=idx, dtype="object")
    out[score >= 1] = "alpha_bull"
    out[score == 0] = "alpha_neutral"
    out[score <= -1] = "alpha_bear"
    return out.fillna("alpha_neutral")


def split_conflict_state(
    regime: pd.Series,
    alpha_state: Optional[pd.Series] = None,
    conflict_flag: Optional[pd.Series] = None,
) -> pd.Series:
    out = regime.copy().astype("object")
    if alpha_state is None:
        alpha_state = pd.Series("alpha_neutral", index=out.index, dtype="object")
    else:
        alpha_state = alpha_state.reindex(out.index).fillna("alpha_neutral")
    if conflict_flag is None:
        conflict_flag = (out == "CONFLICT")
    else:
        conflict_flag = conflict_flag.reindex(out.index).fillna(False).astype(bool)

    out[(conflict_flag) & (alpha_state == "alpha_bull")] = "CONFLICT_BULLISH_ALPHA"
    out[(conflict_flag) & (alpha_state != "alpha_bull")] = "CONFLICT_WEAK_ALPHA"
    return out.fillna("NEUTRAL")


def compute_regime_multipliers(regime_state: pd.Series) -> pd.DataFrame:
    idx = regime_state.index
    path_mult = pd.Series(0.90, index=idx, dtype="float64")
    pos_mult = pd.Series(0.85, index=idx, dtype="float64")
    conf = pd.Series("MID", index=idx, dtype="object")

    path_mult[regime_state == "TAILWIND"] = 1.00
    pos_mult[regime_state == "TAILWIND"] = 1.00
    conf[regime_state == "TAILWIND"] = "HIGH"

    path_mult[regime_state == "NEUTRAL"] = 0.90
    pos_mult[regime_state == "NEUTRAL"] = 0.85
    conf[regime_state == "NEUTRAL"] = "MID"

    path_mult[regime_state == "CONFLICT_BULLISH_ALPHA"] = 0.80
    pos_mult[regime_state == "CONFLICT_BULLISH_ALPHA"] = 0.70
    conf[regime_state == "CONFLICT_BULLISH_ALPHA"] = "MID-LOW"

    path_mult[regime_state == "CONFLICT_WEAK_ALPHA"] = 0.70
    pos_mult[regime_state == "CONFLICT_WEAK_ALPHA"] = 0.55
    conf[regime_state == "CONFLICT_WEAK_ALPHA"] = "LOW"

    path_mult[regime_state == "HEADWIND"] = 0.60
    pos_mult[regime_state == "HEADWIND"] = 0.40
    conf[regime_state == "HEADWIND"] = "LOW"

    return pd.DataFrame({
        "regime_path_multiplier": path_mult,
        "regime_position_multiplier": pos_mult,
        "confidence_bucket": conf,
    }, index=idx)



def add_raw_regime_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """
    one-file exportìì raw regime íë¨ ê·¼ê±°ë¥¼ ì­ì¶ì íê¸° ìí ì§ë¨ ì»¬ë¼ ì¶ê°.
    ê¸°ë ìë ¥ ì»¬ë¼:
      - z_liq_s
      - z_dxy_s
      - dir_liq_sum
      - dir_dxy_sum
      - regime_shifted
      - conflict_flag
    """
    out = df.copy()
    out["dir_liq_s"] = np.sign(pd.to_numeric(out.get("dir_liq_sum"), errors="coerce"))
    out["dir_dxy_s"] = np.sign(pd.to_numeric(out.get("dir_dxy_sum"), errors="coerce"))
    out["dir_sum"] = pd.to_numeric(out["dir_liq_s"], errors="coerce") + pd.to_numeric(out["dir_dxy_s"], errors="coerce")
    out["raw_regime_base"] = out.get("regime_shifted", pd.Series(index=out.index, dtype="object")).fillna("NEUTRAL")
    out["raw_conflict_flag"] = out.get("conflict_flag", pd.Series(index=out.index, dtype="float64")).fillna(0).astype(int)

    def _rule_code(row):
        a = row.get("dir_liq_s", np.nan)
        b = row.get("dir_dxy_s", np.nan)
        if pd.isna(a) or pd.isna(b):
            return np.nan

        def enc(v):
            if v > 0:
                return "1"
            if v < 0:
                return "m1"
            return "0"

        base = str(row.get("raw_regime_base", "NEUTRAL"))
        return f"{base}_{enc(a)}_{enc(b)}"

    out["raw_regime_rule_code"] = out.apply(_rule_code, axis=1)
    return out


def summarize_alpha_merge_status(df: pd.DataFrame, alpha_inputs: dict) -> dict:
    """
    funding / mvrvê° ì¤ì ë¡ mergeëìëì§ one-fileì ë¨ê¸¸ ë©í ìì½.
    """
    meta = alpha_inputs.get("meta", {}) if isinstance(alpha_inputs, dict) else {}
    funding_series = pd.to_numeric(df.get("funding_rate_w"), errors="coerce") if "funding_rate_w" in df.columns else pd.Series(dtype="float64")
    funding_8w_ma = pd.to_numeric(df.get("funding_8w_ma"), errors="coerce") if "funding_8w_ma" in df.columns else pd.Series(dtype="float64")
    funding_z = pd.to_numeric(df.get("funding_z"), errors="coerce") if "funding_z" in df.columns else pd.Series(dtype="float64")
    mvrv_z = pd.to_numeric(df.get("mvrv_z"), errors="coerce") if "mvrv_z" in df.columns else pd.Series(dtype="float64")

    funding_nonnull_count = int(funding_series.notna().sum()) if len(funding_series) else 0
    mvrv_nonnull_count = int(mvrv_z.notna().sum()) if len(mvrv_z) else 0

    funding_last_valid_dt = funding_series.dropna().index.max() if funding_nonnull_count > 0 else pd.NaT
    mvrv_last_valid_dt = mvrv_z.dropna().index.max() if mvrv_nonnull_count > 0 else pd.NaT

    if (funding_nonnull_count > 0) and (mvrv_nonnull_count > 0):
        alpha_ready = "full"
    elif (funding_nonnull_count > 0) or (mvrv_nonnull_count > 0):
        alpha_ready = "partial"
    else:
        alpha_ready = "none"

    return {
        "funding_available_flag": int(funding_nonnull_count > 0),
        "funding_source": str(meta.get("funding_source", "none")),
        "funding_last_valid_dt": funding_last_valid_dt,
        "funding_nonnull_count": funding_nonnull_count,
        "funding_weekly_mean_latest": float(funding_series.dropna().iloc[-1]) if funding_nonnull_count > 0 else np.nan,
        "funding_8w_ma_latest": float(funding_8w_ma.dropna().iloc[-1]) if len(funding_8w_ma.dropna()) > 0 else np.nan,
        "funding_8w_ma_z_latest": float(funding_z.dropna().iloc[-1]) if len(funding_z.dropna()) > 0 else np.nan,
        "funding_state_latest": str(df["funding_state"].dropna().iloc[-1]) if ("funding_state" in df.columns and len(df["funding_state"].dropna()) > 0) else "alpha_neutral",
        "mvrv_available_flag": int(mvrv_nonnull_count > 0),
        "mvrv_source": str(meta.get("mvrv_source", "none")),
        "mvrv_last_valid_dt": mvrv_last_valid_dt,
        "mvrv_nonnull_count": mvrv_nonnull_count,
        "alpha_inputs_ready_flag": alpha_ready,
    }


def compute_current_h19_decision_row(
    btc_end: pd.Timestamp,
    btc_last_px: float,
    current_regime_state: str,
    current_confidence_bucket: str,
    current_path_mult: float,
    current_pos_mult: float,
    pred_path_current_h19_raw: pd.Series,
    pred_path_current_h19_adj: pd.Series,
) -> dict:
    """
    íì¬ anchor ê¸°ì¤ fixed-H19 decision ê°ì one-file exportì ë°ë³µ ì ì¥íê¸° ìí row ë©í ìì±.
    """
    out = {
        "current_anchor_dt": btc_end,
        "current_predicted_ret_19w_raw": np.nan,
        "current_predicted_ret_19w_adj": np.nan,
        "current_predicted_px_19w_raw": np.nan,
        "current_predicted_px_19w_adj": np.nan,
        "current_regime_state": current_regime_state,
        "current_confidence_bucket": current_confidence_bucket,
        "current_regime_path_multiplier": current_path_mult,
        "current_regime_position_multiplier": current_pos_mult,
        "current_suggested_exposure": np.nan,
    }

    if pred_path_current_h19_raw is not None and len(pred_path_current_h19_raw.dropna()) > 0:
        px_raw = float(pred_path_current_h19_raw.dropna().iloc[-1])
        out["current_predicted_px_19w_raw"] = px_raw
        out["current_predicted_ret_19w_raw"] = float(np.log(px_raw / float(btc_last_px)))

    if pred_path_current_h19_adj is not None and len(pred_path_current_h19_adj.dropna()) > 0:
        px_adj = float(pred_path_current_h19_adj.dropna().iloc[-1])
        out["current_predicted_px_19w_adj"] = px_adj
        out["current_predicted_ret_19w_adj"] = float(np.log(px_adj / float(btc_last_px)))
        if np.isfinite(out["current_predicted_ret_19w_adj"]) and np.isfinite(current_pos_mult):
            out["current_suggested_exposure"] = float(np.sign(out["current_predicted_ret_19w_adj"]) * float(current_pos_mult))

    return out


def apply_path_multiplier_to_price_path(
    raw_path: pd.Series,
    anchor_px: float,
    path_multiplier: float,
) -> pd.Series:
    if raw_path is None or len(raw_path) == 0 or not np.isfinite(anchor_px):
        return pd.Series(dtype="float64")
    s = pd.to_numeric(raw_path, errors="coerce").dropna().copy()
    if s.empty:
        return s
    ret = np.log(s / float(anchor_px))
    adj_ret = ret * float(path_multiplier)
    adj_px = float(anchor_px) * np.exp(adj_ret)
    adj_px.name = getattr(raw_path, "name", "adjusted_path")
    return adj_px


def build_horizon_scorecard(
    df: pd.DataFrame,
    horizon_w: int = H19_DIAG_W,
    recent_key: Optional[str] = None,
    group_col: Optional[str] = None,
    pred_ret_col: Optional[str] = None,
    real_ret_col: Optional[str] = None,
) -> pd.DataFrame:
    pred_ret_col = pred_ret_col or f"predicted_fwd_ret_{horizon_w}w"
    real_ret_col = real_ret_col or f"realized_fwd_ret_{horizon_w}w"
    pred_sign_col = pred_ret_col.replace("_ret_", "_sign_")
    real_sign_col = real_ret_col.replace("_ret_", "_sign_")

    cols = [pred_ret_col, real_ret_col]
    if pred_sign_col in df.columns:
        cols.append(pred_sign_col)
    if real_sign_col in df.columns:
        cols.append(real_sign_col)
    if group_col is not None and group_col in df.columns:
        cols.append(group_col)
    if recent_key is not None and recent_key in df.columns:
        cols.append(recent_key)

    tmp = df[cols].copy()
    if recent_key is not None and recent_key in tmp.columns:
        tmp = tmp[tmp[recent_key].fillna(0).astype(int) == 1]
    tmp = tmp.dropna(subset=[pred_ret_col, real_ret_col])
    if tmp.empty:
        return pd.DataFrame()

    if pred_sign_col not in tmp.columns:
        tmp[pred_sign_col] = np.sign(tmp[pred_ret_col])
    if real_sign_col not in tmp.columns:
        tmp[real_sign_col] = np.sign(tmp[real_ret_col])

    if group_col is None or group_col not in tmp.columns:
        grouped_items = [("ALL", tmp)]
    else:
        grouped_items = list(tmp.groupby(group_col, dropna=False))

    rows = []
    for gname, sub in grouped_items:
        n = int(len(sub))
        if n == 0:
            continue
        corr = float(sub[pred_ret_col].corr(sub[real_ret_col])) if n >= 2 else np.nan
        rank_ic = float(sub[pred_ret_col].rank().corr(sub[real_ret_col].rank())) if n >= 2 else np.nan
        sign_acc = float((np.sign(sub[pred_ret_col]) == np.sign(sub[real_ret_col])).mean())
        mae_ret = float((sub[pred_ret_col] - sub[real_ret_col]).abs().mean())
        rmse_ret = float(np.sqrt(np.mean((sub[pred_ret_col] - sub[real_ret_col]) ** 2)))
        up_mask = sub[real_ret_col] > 0
        dn_mask = sub[real_ret_col] < 0
        hit_up = float((np.sign(sub.loc[up_mask, pred_ret_col]) == 1).mean()) if up_mask.any() else np.nan
        hit_down = float((np.sign(sub.loc[dn_mask, pred_ret_col]) == -1).mean()) if dn_mask.any() else np.nan
        rows.append({
            "group": str(gname),
            "n": n,
            "return_corr": corr,
            "rank_ic": rank_ic,
            "sign_acc": sign_acc,
            "mae_ret": mae_ret,
            "rmse_ret": rmse_ret,
            "avg_pred_ret": float(sub[pred_ret_col].mean()),
            "avg_realized_ret": float(sub[real_ret_col].mean()),
            "hit_rate_up": hit_up,
            "hit_rate_down": hit_down,
        })
    return pd.DataFrame(rows).sort_values(["group"]).reset_index(drop=True)


def build_regime_diagnostic_table(
    df: pd.DataFrame,
    group_col: str = "regime_shifted",
    recent_key: Optional[str] = None,
) -> pd.DataFrame:
    needed = [c for c in [group_col, "dir_liq_sum", "dir_dxy_sum", "z_liq_s", "z_dxy_s", "predicted_fwd_ret_19w_adj", "realized_fwd_ret_19w"] if c in df.columns]
    if group_col not in needed:
        return pd.DataFrame()
    tmp = df[needed].copy()
    if recent_key is not None and recent_key in df.columns:
        tmp = tmp[df[recent_key].fillna(0).astype(int) == 1]
    if tmp.empty:
        return pd.DataFrame()

    rows = []
    for gname, sub in tmp.groupby(group_col, dropna=False):
        if sub.empty:
            continue
        row = {"group": str(gname), "n": int(len(sub))}
        for c in ["dir_liq_sum", "dir_dxy_sum", "z_liq_s", "z_dxy_s"]:
            if c in sub.columns:
                row[f"avg_{c}"] = float(pd.to_numeric(sub[c], errors="coerce").mean())
        if "predicted_fwd_ret_19w_adj" in sub.columns and "realized_fwd_ret_19w" in sub.columns:
            valid = sub[["predicted_fwd_ret_19w_adj", "realized_fwd_ret_19w"]].dropna()
            row["valid_signal_n"] = int(len(valid))
            row["avg_pred_ret_adj"] = float(valid["predicted_fwd_ret_19w_adj"].mean()) if not valid.empty else np.nan
            row["avg_realized_ret"] = float(valid["realized_fwd_ret_19w"].mean()) if not valid.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["group"]).reset_index(drop=True)


def build_fixed_horizon_signal_panel(
    px_wclose: pd.Series,
    ldli_level: pd.Series,
    ldli_liq_contrib: pd.Series,
    ldli_dxy_contrib: pd.Series,
    lag_series: pd.Series,
    horizon_w: int = H19_DIAG_W,
    fit_window_w: int = TAB5_DEFAULT_FIT_W,
    alpha_mode: str = "OLS (learn alpha)",
    forecast_model: str = "Drivers only",
) -> pd.DataFrame:
    rows = []
    anchors = list(px_wclose.index)
    max_anchor = px_wclose.index.max() - pd.Timedelta(weeks=horizon_w)
    for anchor_dt in anchors:
        if pd.Timestamp(anchor_dt) > max_anchor:
            continue
        xx_a = lag_series.reindex([anchor_dt]).iloc[0] if anchor_dt in lag_series.index else np.nan
        if not np.isfinite(xx_a):
            continue
        xx_a = int(np.clip(int(round(float(xx_a))), LAG_MIN_WEEKS, LAG_MAX_WEEKS))
        liq_shift_a = _shift_weeks_index(ldli_liq_contrib, xx_a, f"liq_shift_{xx_a}w")
        dxy_shift_a = _shift_weeks_index(ldli_dxy_contrib, xx_a, f"dxy_shift_{xx_a}w")
        try:
            if str(forecast_model).startswith("ECM"):
                fc = forecast_path_ecm_level(
                    btc_px=px_wclose,
                    ldli_level=ldli_level,
                    anchor_dt=anchor_dt,
                    lag_weeks=xx_a,
                    horizon_w=horizon_w,
                    fit_window_w=fit_window_w,
                    alpha_mode=alpha_mode,
                )
                ps = fc["path_series"]
            else:
                fc = forecast_path_from_drivers_driversonly(
                    px_wclose=px_wclose,
                    liq_shifted=liq_shift_a,
                    dxy_shifted=dxy_shift_a,
                    horizon_w=horizon_w,
                    fit_window_w=fit_window_w,
                    end_dt_requested=anchor_dt,
                    alpha_mode=alpha_mode,
                )
                ps = fc["path_series"]
            if ps is None or ps.dropna().empty:
                continue
            pred_px = float(ps.dropna().iloc[-1])
            anchor_px = float(px_wclose.loc[anchor_dt])
            rows.append({
                "anchor_dt": pd.Timestamp(anchor_dt),
                "endpoint_dt_h19": pd.Timestamp(ps.dropna().index[-1]),
                "lag_weeks_h19": int(xx_a),
                "predicted_fwd_px_19w": pred_px,
                "predicted_fwd_ret_19w": float(np.log(pred_px / anchor_px)),
                "predicted_sign_19w": float(np.sign(np.log(pred_px / anchor_px))),
            })
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(index=px_wclose.index)
    out = pd.DataFrame(rows).set_index("anchor_dt").sort_index()
    return out



def compute_forward_metrics(price_w: pd.Series, regime_shifted: pd.Series, horizons: List[int]) -> pd.DataFrame:
    df = pd.concat([price_w.rename("px"), regime_shifted.rename("reg")], axis=1).dropna()
    if df.empty:
        return pd.DataFrame()

    px = df["px"]
    reg = df["reg"]

    rows = []
    for h in horizons:
        fwd = np.log(px.shift(-h) / px)

        fwd_steps = []
        for k in range(1, h + 1):
            fwd_k = np.log(px.shift(-k) / px)
            fwd_steps.append(fwd_k.rename(f"fwd_{k}"))
        mat = pd.concat(fwd_steps, axis=1)

        mae = mat.min(axis=1)
        mfe = mat.max(axis=1)

        tmp = pd.concat([reg, fwd.rename("fwd"), mae.rename("mae"), mfe.rename("mfe")], axis=1).dropna()

        for rname in ["TAILWIND", "CONFLICT", "HEADWIND", "NEUTRAL"]:
            sub = tmp[tmp["reg"] == rname]
            n = int(sub.shape[0])
            if n == 0:
                rows.append({
                    "horizon_w": h, "regime": rname, "n": 0,
                    "win_rate": np.nan, "mean_fwd": np.nan, "median_fwd": np.nan,
                    "avg_gain": np.nan, "avg_loss": np.nan, "payoff": np.nan,
                    "mae_median": np.nan, "mfe_median": np.nan
                })
                continue

            win_rate = float((sub["fwd"] > 0).mean())
            mean_fwd = float(sub["fwd"].mean())
            median_fwd = float(sub["fwd"].median())

            gains = sub.loc[sub["fwd"] > 0, "fwd"]
            losses = sub.loc[sub["fwd"] <= 0, "fwd"]
            avg_gain = float(gains.mean()) if len(gains) > 0 else np.nan
            avg_loss = float(losses.mean()) if len(losses) > 0 else np.nan
            payoff = (avg_gain / abs(avg_loss)) if (np.isfinite(avg_gain) and np.isfinite(avg_loss) and avg_loss != 0) else np.nan

            mae_median = float(sub["mae"].median())
            mfe_median = float(sub["mfe"].median())

            rows.append({
                "horizon_w": h, "regime": rname, "n": n,
                "win_rate": win_rate, "mean_fwd": mean_fwd, "median_fwd": median_fwd,
                "avg_gain": avg_gain, "avg_loss": avg_loss, "payoff": payoff,
                "mae_median": mae_median, "mfe_median": mfe_median
            })

    return pd.DataFrame(rows)


# =========================
# ---- DRIVERS-ONLY FORECAST (FIXED) ----
# =========================
def forecast_path_from_drivers_driversonly(
    px_wclose: pd.Series,
    liq_shifted: pd.Series,
    dxy_shifted: pd.Series,
    horizon_w: int,
    fit_window_w: int,
    end_dt_requested: pd.Timestamp,
    alpha_mode: str = "OLS (learn alpha)",
) -> Dict[str, object]:
    """
    Fix: keep future driver rows by splitting:
      - df_fit: px+drivers (for regression)
      - df_drv: drivers-only (for future horizon)
    """

    df_drv = pd.concat([
        liq_shifted.rename("liq"),
        dxy_shifted.rename("dxy"),
    ], axis=1).dropna()

    df_drv["d1"] = df_drv["liq"].diff()
    df_drv["d2"] = df_drv["dxy"].diff()
    df_drv = df_drv.dropna()

    df_fit = pd.concat([
        px_wclose.rename("px"),
        liq_shifted.rename("liq"),
        dxy_shifted.rename("dxy"),
    ], axis=1).dropna()

    df_fit["r"] = np.log(df_fit["px"]).diff()
    df_fit["d1"] = df_fit["liq"].diff()
    df_fit["d2"] = df_fit["dxy"].diff()
    df_fit = df_fit.dropna()

    if df_fit.empty or df_drv.empty:
        raise RuntimeError("Drivers/Price ë°ì´í°ê° ë¶ì¡±í©ëë¤.")

    if end_dt_requested not in df_fit.index:
        idx = df_fit.index[df_fit.index <= end_dt_requested]
        if len(idx) == 0:
            raise RuntimeError("No fit data up to requested end_dt")
        end_dt_eff = idx[-1]
    else:
        end_dt_eff = end_dt_requested

    pos_fit = df_fit.index.get_loc(end_dt_eff)
    if pos_fit < fit_window_w:
        raise RuntimeError("Not enough history for fit window")

    if end_dt_eff not in df_drv.index:
        idx2 = df_drv.index[df_drv.index <= end_dt_eff]
        if len(idx2) == 0:
            raise RuntimeError("No driver data up to end_dt_eff")
        end_dt_eff_drv = idx2[-1]
    else:
        end_dt_eff_drv = end_dt_eff

    pos_drv = df_drv.index.get_loc(end_dt_eff_drv)
    max_h = int((len(df_drv.index) - 1) - pos_drv)
    h = int(min(int(horizon_w), int(max_h)))
    if h <= 0:
        raise RuntimeError(
            f"Forecast ê°ë¥í ìµë horizon: 0ì£¼ (drivers ê¸°ì¤ max_h=0ì£¼, end_dt_eff={end_dt_eff.date()})"
        )

    w = df_fit.iloc[pos_fit - fit_window_w:pos_fit]
    y = w["r"].to_numpy(float)
    x1 = w["d1"].to_numpy(float)
    x2 = w["d2"].to_numpy(float)
    res = ols_2var_with_tstats(y, x1, x2, min_obs=max(MIN_OBS, 40))

    fut_drv = df_drv.iloc[pos_drv + 1:pos_drv + 1 + h]
    if fut_drv.shape[0] < h:
        raise RuntimeError("Not enough future driver rows for horizon (drivers-only)")

    use_zero_alpha = str(alpha_mode).startswith("ZERO")
    a_eff = 0.0 if use_zero_alpha else float(res.get("a", np.nan))
    if (not use_zero_alpha) and (not np.isfinite(a_eff)):
        raise RuntimeError("OLS alpha is NaN/inf; try ZERO(alpha=0) mode")

    r_hat = (a_eff + res["b1"] * fut_drv["d1"].astype(float) + res["b2"] * fut_drv["d2"].astype(float)).astype(float)

    px0 = float(df_fit.loc[end_dt_eff, "px"])
    path = [px0]
    cur = px0
    for rr in r_hat.values:
        if np.isfinite(rr):
            cur = cur * float(np.exp(rr))
        path.append(cur)

    sigma_w = float(res.get("sigma", np.nan))
    exp_cum_logret = float(np.nansum(r_hat.values)) if r_hat.notna().any() else np.nan
    z = (exp_cum_logret / (sigma_w * math.sqrt(float(h)))) if (np.isfinite(exp_cum_logret) and np.isfinite(sigma_w) and sigma_w > 0) else np.nan

    idx_path = pd.DatetimeIndex([end_dt_eff] + list(fut_drv.index))
    path_s = pd.Series(path, index=idx_path, name="forecast_path")

    return {
        "end_dt_requested": end_dt_requested,
        "end_dt_eff": end_dt_eff,
        "max_h": int(max_h),
        "h": int(h),
        "px0": px0,
        "path_series": path_s,
        "fut_index": fut_drv.index,
        "r_hat": r_hat,
        "exp_logret": exp_cum_logret,
        "sigma_w": sigma_w,
        "z": z,
        "R2": float(res.get("R2", np.nan)),
        "t1": float(res.get("t1", np.nan)),
        "t2": float(res.get("t2", np.nan)),
        "alpha_mode": str(alpha_mode),
                "forecast_model": str(forecast_model),
        "alpha_used": float(a_eff),
    }


# =========================
# ECM (LDLI level) forecast helpers
# =========================
def _ols_fit(y: np.ndarray, X: np.ndarray, add_intercept: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple OLS via least squares.
    Returns: (beta, yhat)
    """
    if add_intercept:
        X_ = np.column_stack([np.ones(len(X)), X])
    else:
        X_ = X
    beta, *_ = np.linalg.lstsq(X_, y, rcond=None)
    yhat = X_ @ beta
    return beta, yhat


def _shift_index_by_weeks(s: pd.Series, weeks: int) -> pd.Series:
    """Shift a Series forward in time by adding `weeks` to the index (keeps values)."""
    if int(weeks) == 0:
        return s.copy()
    out = s.copy()
    out.index = out.index + pd.to_timedelta(int(weeks) * 7, unit="D")
    return out


def fit_ecm_params_from_level(
    btc_px: pd.Series,
    ldli_level: pd.Series,
    lag_weeks: int,
    end_dt: pd.Timestamp,
    fit_window_w: int,
    alpha_mode: str,
) -> Optional[Dict[str, float]]:
    """
    Fit:
      1) Level mapping: log(BTC) ~= c + gamma * LDLI_shifted_level
      2) ECM step model (weekly): Îlog(BTC)_t = a + b_gap * gap_{t-1} + b_dldli * ÎLDLI_t

    LDLI is shifted by `lag_weeks` (index shift) so that LDLI leads BTC on the same x-axis.
    """
    ld_shift = _shift_index_by_weeks(ldli_level, lag_weeks)

    df = pd.concat(
        [
            btc_px.rename("px"),
            ld_shift.rename("ld"),
        ],
        axis=1,
    ).dropna()

    if df.empty:
        return None

    end_dt = pd.to_datetime(end_dt)
    df = df.loc[:end_dt].tail(max(fit_window_w + 10, 40))
    if len(df) < max(24, fit_window_w // 2):
        return None

    log_px = np.log(df["px"].values.astype(float))
    ld = df["ld"].values.astype(float)

    # Level mapping: log_px ~= c + gamma*ld
    beta_lv, _ = _ols_fit(log_px, ld.reshape(-1, 1), add_intercept=True)
    c = float(beta_lv[0])
    gamma = float(beta_lv[1])

    # gap, dld, r
    gap = (log_px - (c + gamma * ld))
    dld = np.diff(ld, prepend=np.nan)
    r = np.diff(log_px, prepend=np.nan)

    # ECM: r_t on gap_{t-1} and dld_t
    x1 = np.roll(gap, 1)
    x1[0] = np.nan
    x2 = dld

    mask = np.isfinite(r) & np.isfinite(x1) & np.isfinite(x2)
    if mask.sum() < 18:
        return None

    y = r[mask]
    X = np.column_stack([x1[mask], x2[mask]])

    add_intercept = not (str(alpha_mode).strip().lower().startswith("alpha=0"))
    beta_ecm, _ = _ols_fit(y, X, add_intercept=add_intercept)

    if add_intercept:
        a = float(beta_ecm[0])
        b_gap = float(beta_ecm[1])
        b_dld = float(beta_ecm[2])
    else:
        a = 0.0
        b_gap = float(beta_ecm[0])
        b_dld = float(beta_ecm[1])

    return {
        "c": c,
        "gamma": gamma,
        "a": a,
        "b_gap": b_gap,
        "b_dld": b_dld,
        "lag_weeks": int(lag_weeks),
        "n_fit": int(mask.sum()),
        "add_intercept": bool(add_intercept),
    }


def forecast_path_ecm_level(
    btc_px: pd.Series,
    ldli_level: pd.Series,
    anchor_dt: pd.Timestamp,
    lag_weeks: int,
    horizon_w: int,
    fit_window_w: int,
    alpha_mode: str,
) -> Dict[str, object]:
    """
    Deterministic ECM path simulation from anchor_dt for `horizon_w` weeks.

    Uses known future LDLI_shifted_level values (index-shift creates values beyond the last BTC point).
    """
    anchor_dt = pd.to_datetime(anchor_dt)

    params = fit_ecm_params_from_level(
        btc_px=btc_px,
        ldli_level=ldli_level,
        lag_weeks=lag_weeks,
        end_dt=anchor_dt,
        fit_window_w=fit_window_w,
        alpha_mode=alpha_mode,
    )
    if params is None:
        return {"path_series": pd.Series(dtype=float), "params": None, "status": "fit_failed"}

    ld_shift = _shift_index_by_weeks(ldli_level, lag_weeks)

    df = pd.concat([btc_px.rename("px"), ld_shift.rename("ld")], axis=1).dropna()
    if df.empty:
        return {"path_series": pd.Series(dtype=float), "params": params, "status": "no_overlap"}

    if anchor_dt not in df.index:
        avail = df.index[df.index <= anchor_dt]
        if len(avail) == 0:
            return {"path_series": pd.Series(dtype=float), "params": params, "status": "anchor_before_data"}
        anchor_dt = avail[-1]

    px_cur = float(df.loc[anchor_dt, "px"])
    ld_cur = float(df.loc[anchor_dt, "ld"])
    log_px_cur = float(np.log(px_cur))

    c = float(params["c"])
    gamma = float(params["gamma"])
    a = float(params["a"])
    b_gap = float(params["b_gap"])
    b_dld = float(params["b_dld"])

    dates: List[pd.Timestamp] = []
    vals: List[float] = []

    dt_cur = anchor_dt

    for _k in range(1, int(horizon_w) + 1):
        dt_next = dt_cur + pd.to_timedelta(7, unit="D")
        if dt_next not in ld_shift.index:
            break

        ld_next = float(ld_shift.loc[dt_next])
        dld_next = ld_next - ld_cur

        gap_cur = log_px_cur - (c + gamma * ld_cur)
        r_next = a + b_gap * gap_cur + b_dld * dld_next

        log_px_next = log_px_cur + float(r_next)
        px_next = float(np.exp(log_px_next))

        dates.append(dt_next)
        vals.append(px_next)

        dt_cur = dt_next
        ld_cur = ld_next
        log_px_cur = log_px_next

    path = pd.Series(vals, index=pd.to_datetime(dates), name="pred_px")
    return {"path_series": path, "params": params, "status": "ok"}


# =========================
# MAIN/TAB4 payload builder
# =========================
@st.cache_data(ttl=60 * 60 * 2, show_spinner=False)

def build_forward_overlay_payload(
    liq_source: str,
    alpha_mode: str,
    forecast_model: str,
    use_binance_funding: bool = True,
    funding_symbol: str = "BTCUSDT",
    mvrv_file_bytes: Optional[bytes] = None,
    mvrv_file_name: Optional[str] = None,
):
    btc_close = load_btc_close()
    dxy_close, dxy_used = load_dxy_close()
    liq_level_daily, liq_snapshot, liq_label, _liq_units = load_liquidity_source_daily(liq_source)

    btc_wclose = btc_close.resample(WEEK_RULE).last().dropna()
    btc_wret = weekly_log_returns(btc_close, WEEK_RULE).rename("btc_wret")
    dxy_wret = weekly_log_returns(dxy_close, WEEK_RULE).rename("dxy_wret")

    liq_week = liq_level_daily.resample(WEEK_RULE).last().dropna()
    if (liq_week <= 0).any():
        liq_ret = weekly_pct_change(liq_level_daily, WEEK_RULE).rename("liq_ret_pct")
    else:
        liq_ret = weekly_log_returns(liq_level_daily, WEEK_RULE).rename("liq_ret_log")
    liq_dlt = weekly_delta(liq_level_daily, WEEK_RULE).rename("liq_delta_level")

    base = pd.concat([btc_wret, btc_wclose.rename("btc_close"), dxy_wret, liq_ret, liq_dlt], axis=1).dropna()
    if len(base) < (WINDOW_WEEKS + LAG_MAX_WEEKS + 10):
        raise RuntimeError("Forward overlay ê³ì°ì íìí ì£¼ê° ìíì´ ë¶ì¡±í©ëë¤.")

    z_dxy_inv = zscore(-base["dxy_wret"]).rename("z(-dxy_ret)")
    z_liq_ret = zscore(base[liq_ret.name]).rename("z(liq_ret)")
    z_liq_dlt = zscore(base["liq_delta_level"]).rename("z(liq_delta)")

    combo_ret = (z_liq_ret + z_dxy_inv).rename("combo_ret")
    combo_dlt = (z_liq_dlt + z_dxy_inv).rename("combo_delta")

    lags_weeks = range(LAG_MIN_WEEKS, LAG_MAX_WEEKS + 1)
    y = base["btc_wret"]

    _, _, _, _, _, out_r = rolling_maps_weekly(y_w=y, x_w=combo_ret, invert_x=False, lags_weeks=lags_weeks, quiet=True)
    _, _, _, _, _, out_d = rolling_maps_weekly(y_w=y, x_w=combo_dlt, invert_x=False, lags_weeks=lags_weeks, quiet=True)

    last_r = out_r.dropna(subset=["best_corr_raw"]).iloc[-1]
    last_d = out_d.dropna(subset=["best_corr_raw"]).iloc[-1]
    if float(last_r["best_corr_raw"]) >= float(last_d["best_corr_raw"]):
        chosen = "RET"
        out_use = out_r
        best_corr_last = float(last_r["best_corr_raw"])
        z_liq_use = z_liq_ret.reindex(base.index)
    else:
        chosen = "DELTA"
        out_use = out_d
        best_corr_last = float(last_d["best_corr_raw"])
        z_liq_use = z_liq_dlt.reindex(base.index)

    lag_s = out_use["best_lag_valid_smooth_weeks"].copy()
    lag_s = lag_s.reindex(base.index).ffill()
    lag_s.name = "xx_dynamic_weeks"

    lag_latest = lag_s.dropna().iloc[-1] if lag_s.dropna().size > 0 else out_use.dropna(subset=["best_lag_raw_weeks"]).iloc[-1]["best_lag_raw_weeks"]
    xx_latest = int(np.clip(int(round(float(lag_latest))), LAG_MIN_WEEKS, LAG_MAX_WEEKS))

    z_dxy_use = z_dxy_inv.reindex(base.index)
    combo_use = (z_liq_use + z_dxy_use).rename("combo_use")

    ldli_level = (LDLI_BASE + LDLI_SCALE * combo_use.cumsum()).rename("LDLI_level")
    ldli_liq_contrib = (LDLI_SCALE * z_liq_use.cumsum()).rename("LDLI_liq_contrib")
    ldli_dxy_contrib = (LDLI_SCALE * z_dxy_use.cumsum()).rename("LDLI_dxy_contrib")

    regime, regime_diag = label_regime_v2(
        z_liq=z_liq_use,
        z_dxy=z_dxy_use,
        smooth_weeks=REGIME_SMOOTH_WEEKS,
        dir_lookback_weeks=REGIME_DIR_LOOKBACK_WEEKS,
        deadband=REGIME_DIR_DEADBAND
    )
    regime = regime.rename("regime")

    ldli_shifted = _shift_weeks_index(ldli_level, xx_latest, f"LDLI_shifted_{xx_latest}w")
    liq_shifted_latest = _shift_weeks_index(ldli_liq_contrib, xx_latest, f"LDLI_liq_contrib_shifted_{xx_latest}w")
    dxy_shifted_latest = _shift_weeks_index(ldli_dxy_contrib, xx_latest, f"LDLI_dxy_contrib_shifted_{xx_latest}w")
    regime_shifted = _shift_weeks_index(regime, xx_latest, f"regime_shifted_{xx_latest}w")
    regime_diag_shifted = _shift_weeks_index(regime_diag, xx_latest, f"regime_diag_shifted_{xx_latest}w")

    btc_end = base.index.max()

    start_disp = btc_end - pd.to_timedelta(PAST_WEEKS_FIXED * 7, unit="D")
    end_disp = btc_end + pd.to_timedelta(xx_latest * 7, unit="D")
    full_idx = pd.date_range(start=start_disp, end=end_disp, freq=WEEK_RULE)

    btc_plot = base["btc_close"].reindex(full_idx)
    ldli_plot = ldli_shifted.reindex(full_idx)
    liq_plot = liq_shifted_latest.reindex(full_idx)
    dxy_plot = dxy_shifted_latest.reindex(full_idx)
    regime_plot = regime_shifted.reindex(full_idx)
    conflict_mask = (regime_plot == "CONFLICT").fillna(False).to_numpy(dtype=bool)

    btc_wclose_full = base["btc_close"].copy()
    regime_on_btc_timeline = regime_shifted.reindex(btc_wclose_full.index).fillna("NEUTRAL")

    metrics = compute_forward_metrics(
        price_w=btc_wclose_full,
        regime_shifted=regime_on_btc_timeline,
        horizons=REGIME_HORIZONS_WEEKS
    )

    if str(forecast_model).startswith("ECM"):
        fc_latest = forecast_path_ecm_level(
            btc_px=btc_wclose_full,
            ldli_level=ldli_level,
            anchor_dt=btc_end,
            lag_weeks=xx_latest,
            horizon_w=xx_latest,
            fit_window_w=TAB5_DEFAULT_FIT_W,
            alpha_mode=alpha_mode,
        )
        pred_path_latest = fc_latest["path_series"]
    else:
        fc_latest = forecast_path_from_drivers_driversonly(
            px_wclose=btc_wclose_full,
            liq_shifted=liq_shifted_latest,
            dxy_shifted=dxy_shifted_latest,
            horizon_w=xx_latest,
            fit_window_w=TAB5_DEFAULT_FIT_W,
            end_dt_requested=btc_end,
            alpha_mode=alpha_mode,
        )
        pred_path_latest = fc_latest["path_series"]

    spaghetti_paths = []
    spaghetti_long_rows = []
    pred_endpoint_map = {}

    anchors = list(base.index)
    if SPAGHETTI_ANCHOR_STEP_WEEKS and SPAGHETTI_ANCHOR_STEP_WEEKS > 1:
        anchors = anchors[::int(SPAGHETTI_ANCHOR_STEP_WEEKS)]
    if SPAGHETTI_MAX_ANCHORS is not None:
        anchors = anchors[-int(SPAGHETTI_MAX_ANCHORS):]

    for anchor_dt in anchors:
        xx_a = lag_s.reindex([anchor_dt]).iloc[0]
        if not np.isfinite(xx_a):
            continue
        xx_a = int(np.clip(int(round(float(xx_a))), LAG_MIN_WEEKS, LAG_MAX_WEEKS))

        liq_shift_a = _shift_weeks_index(ldli_liq_contrib, xx_a, f"liq_shift_{xx_a}w")
        dxy_shift_a = _shift_weeks_index(ldli_dxy_contrib, xx_a, f"dxy_shift_{xx_a}w")

        try:
            if str(forecast_model).startswith("ECM"):
                fc_a = forecast_path_ecm_level(
                    btc_px=btc_wclose_full,
                    ldli_level=ldli_level,
                    anchor_dt=anchor_dt,
                    lag_weeks=xx_a,
                    horizon_w=xx_a,
                    fit_window_w=TAB5_DEFAULT_FIT_W,
                    alpha_mode=alpha_mode,
                )
                ps = fc_a["path_series"]
            else:
                fc_a = forecast_path_from_drivers_driversonly(
                    px_wclose=btc_wclose_full,
                    liq_shifted=liq_shift_a,
                    dxy_shifted=dxy_shift_a,
                    horizon_w=xx_a,
                    fit_window_w=TAB5_DEFAULT_FIT_W,
                    end_dt_requested=anchor_dt,
                    alpha_mode=alpha_mode,
                )
                ps = fc_a["path_series"]
            if ps is None or ps.dropna().empty:
                continue
        except Exception:
            continue

        spaghetti_paths.append(ps)

        for dt, v in ps.items():
            spaghetti_long_rows.append({
                "liq_source": liq_source,
                "alpha_mode": str(alpha_mode),
                "combo_type": chosen,
                "anchor_dt": pd.Timestamp(anchor_dt).date().isoformat(),
                "path_dt": pd.Timestamp(dt).date().isoformat(),
                "pred_px": float(v) if np.isfinite(v) else np.nan,
                "lag_weeks": int(xx_a),
            })

        end_dt = ps.index[-1]
        pred_endpoint_map[end_dt] = float(ps.iloc[-1])

    pred_hist_endpoint = pd.Series(pred_endpoint_map).sort_index()
    pred_hist_endpoint.name = "pred_hist_endpoint_dynamic"

    spaghetti_long = pd.DataFrame(spaghetti_long_rows)

    # === H19 signal / calibration master panel (anchor timeline) ===
    overlay_master_df = pd.DataFrame(index=btc_wclose_full.index)
    overlay_master_df["btc_close"] = btc_wclose_full
    overlay_master_df["ldli_shifted_total"] = ldli_shifted.reindex(overlay_master_df.index)
    overlay_master_df["ldli_shifted_liq"] = liq_shifted_latest.reindex(overlay_master_df.index)
    overlay_master_df["ldli_shifted_dxy"] = dxy_shifted_latest.reindex(overlay_master_df.index)
    overlay_master_df["regime_shifted"] = regime_on_btc_timeline
    if isinstance(regime_diag_shifted, pd.DataFrame) and not regime_diag_shifted.empty:
        for _c in ["z_liq_s", "z_dxy_s", "dir_liq_sum", "dir_dxy_sum"]:
            if _c in regime_diag_shifted.columns:
                overlay_master_df[_c] = pd.to_numeric(regime_diag_shifted[_c], errors="coerce").reindex(overlay_master_df.index)
    overlay_master_df["conflict_flag"] = (overlay_master_df["regime_shifted"] == "CONFLICT").astype(int)
    overlay_master_df = add_raw_regime_diagnostics(overlay_master_df)
    overlay_master_df = add_recent_slice_flags(overlay_master_df)
    overlay_master_df = add_horizon_targets(overlay_master_df, px_col="btc_close", horizon_w=H19_DIAG_W)

    alpha_inputs = load_alpha_inputs_weekly(
        mvrv_file_bytes=mvrv_file_bytes,
        mvrv_file_name=mvrv_file_name,
        use_binance_funding=use_binance_funding,
        funding_symbol=funding_symbol,
        week_rule=WEEK_RULE,
    )
    overlay_master_df["mvrv_z"] = alpha_inputs["mvrv_z"].reindex(overlay_master_df.index)
    overlay_master_df["funding_rate_w"] = alpha_inputs["funding_rate_w"].reindex(overlay_master_df.index)
    overlay_master_df["funding_8w_ma"] = (
        pd.to_numeric(overlay_master_df["funding_rate_w"], errors="coerce")
        .rolling(8, min_periods=4)
        .mean()
    )
    overlay_master_df["funding_z"] = alpha_inputs["funding_z"].reindex(overlay_master_df.index)
    overlay_master_df["mvrv_state"] = classify_mvrv_state(overlay_master_df["mvrv_z"])
    overlay_master_df["funding_state"] = classify_funding_state(overlay_master_df["funding_z"])
    overlay_master_df["alpha_state"] = compute_alpha_state(
        overlay_master_df["mvrv_state"], overlay_master_df["funding_state"], index=overlay_master_df.index
    ).reindex(overlay_master_df.index).fillna("alpha_neutral")

    h19_signals = build_fixed_horizon_signal_panel(
        px_wclose=btc_wclose_full,
        ldli_level=ldli_level,
        ldli_liq_contrib=ldli_liq_contrib,
        ldli_dxy_contrib=ldli_dxy_contrib,
        lag_series=lag_s,
        horizon_w=H19_DIAG_W,
        fit_window_w=TAB5_DEFAULT_FIT_W,
        alpha_mode=alpha_mode,
        forecast_model=forecast_model,
    )
    overlay_master_df = overlay_master_df.join(h19_signals, how="left")
    if "predicted_sign_19w" not in overlay_master_df.columns and "predicted_fwd_ret_19w" in overlay_master_df.columns:
        overlay_master_df["predicted_sign_19w"] = np.sign(overlay_master_df["predicted_fwd_ret_19w"])

    _pred_sig_raw = np.sign(pd.to_numeric(overlay_master_df.get("predicted_fwd_ret_19w"), errors="coerce"))
    _real_sig_raw = np.sign(pd.to_numeric(overlay_master_df.get("realized_fwd_ret_19w"), errors="coerce"))
    overlay_master_df["signal_hit_19w"] = np.where(
        _pred_sig_raw.notna() & _real_sig_raw.notna(),
        (_pred_sig_raw == _real_sig_raw).astype(float),
        np.nan,
    )

    overlay_master_df["regime_state"] = split_conflict_state(
        overlay_master_df["regime_shifted"],
        alpha_state=overlay_master_df["alpha_state"],
        conflict_flag=overlay_master_df["conflict_flag"].astype(bool),
    )
    mult_df = compute_regime_multipliers(overlay_master_df["regime_state"])
    overlay_master_df = overlay_master_df.join(mult_df, how="left")

    overlay_master_df["predicted_fwd_ret_19w_adj"] = (
        pd.to_numeric(overlay_master_df["predicted_fwd_ret_19w"], errors="coerce") *
        pd.to_numeric(overlay_master_df["regime_path_multiplier"], errors="coerce")
    )
    overlay_master_df["predicted_fwd_px_19w_adj"] = overlay_master_df["btc_close"] * np.exp(overlay_master_df["predicted_fwd_ret_19w_adj"])
    overlay_master_df["predicted_sign_19w_adj"] = np.sign(overlay_master_df["predicted_fwd_ret_19w_adj"])
    overlay_master_df["suggested_exposure"] = np.where(
        pd.to_numeric(overlay_master_df["predicted_fwd_ret_19w_adj"], errors="coerce").notna(),
        np.sign(pd.to_numeric(overlay_master_df["predicted_fwd_ret_19w_adj"], errors="coerce")) * pd.to_numeric(overlay_master_df["regime_position_multiplier"], errors="coerce"),
        np.nan,
    )

    current_row = overlay_master_df.iloc[-1].copy()
    current_path_mult = float(current_row.get("regime_path_multiplier", 1.0)) if np.isfinite(current_row.get("regime_path_multiplier", np.nan)) else 1.0
    current_pos_mult = float(current_row.get("regime_position_multiplier", np.nan)) if np.isfinite(current_row.get("regime_position_multiplier", np.nan)) else np.nan
    pred_path_latest_adj = apply_path_multiplier_to_price_path(
        raw_path=pred_path_latest,
        anchor_px=float(btc_wclose_full.iloc[-1]),
        path_multiplier=current_path_mult,
    )

    # Current fixed-H19 forecast for decision panel / exposure
    pred_path_current_h19_raw = pd.Series(dtype="float64")
    pred_path_current_h19_adj = pd.Series(dtype="float64")
    current_pred_ret_19w_raw = np.nan
    current_pred_ret_19w_adj = np.nan
    current_suggested_exposure = np.nan
    try:
        if str(forecast_model).startswith("ECM"):
            fc_current_h19 = forecast_path_ecm_level(
                btc_px=btc_wclose_full,
                ldli_level=ldli_level,
                anchor_dt=btc_end,
                lag_weeks=xx_latest,
                horizon_w=H19_DIAG_W,
                fit_window_w=TAB5_DEFAULT_FIT_W,
                alpha_mode=alpha_mode,
            )
            pred_path_current_h19_raw = fc_current_h19["path_series"]
        else:
            fc_current_h19 = forecast_path_from_drivers_driversonly(
                px_wclose=btc_wclose_full,
                liq_shifted=liq_shifted_latest,
                dxy_shifted=dxy_shifted_latest,
                horizon_w=H19_DIAG_W,
                fit_window_w=TAB5_DEFAULT_FIT_W,
                end_dt_requested=btc_end,
                alpha_mode=alpha_mode,
            )
            pred_path_current_h19_raw = fc_current_h19["path_series"]
        if pred_path_current_h19_raw is not None and not pred_path_current_h19_raw.dropna().empty:
            pred_path_current_h19_adj = apply_path_multiplier_to_price_path(
                raw_path=pred_path_current_h19_raw,
                anchor_px=float(btc_wclose_full.iloc[-1]),
                path_multiplier=current_path_mult,
            )
            current_pred_ret_19w_raw = float(np.log(float(pred_path_current_h19_raw.dropna().iloc[-1]) / float(btc_wclose_full.iloc[-1])))
            if pred_path_current_h19_adj is not None and not pred_path_current_h19_adj.dropna().empty:
                current_pred_ret_19w_adj = float(np.log(float(pred_path_current_h19_adj.dropna().iloc[-1]) / float(btc_wclose_full.iloc[-1])))
                if np.isfinite(current_pred_ret_19w_adj) and np.isfinite(current_pos_mult):
                    current_suggested_exposure = float(np.sign(current_pred_ret_19w_adj) * current_pos_mult)
    except Exception:
        pass

    current_decision = compute_current_h19_decision_row(
        btc_end=btc_end,
        btc_last_px=float(btc_wclose_full.iloc[-1]),
        current_regime_state=str(current_row.get("regime_state", "NEUTRAL")),
        current_confidence_bucket=str(current_row.get("confidence_bucket", "MID")),
        current_path_mult=float(current_path_mult),
        current_pos_mult=float(current_pos_mult) if np.isfinite(current_pos_mult) else np.nan,
        pred_path_current_h19_raw=pred_path_current_h19_raw,
        pred_path_current_h19_adj=pred_path_current_h19_adj,
    )
    current_decision["current_alpha_state"] = str(current_row.get("alpha_state", "alpha_neutral"))
    current_decision["current_path_multiplier"] = current_decision["current_regime_path_multiplier"]
    current_decision["current_position_multiplier"] = current_decision["current_regime_position_multiplier"]
    for _k, _v in current_decision.items():
        overlay_master_df[_k] = _v

    alpha_merge_meta = summarize_alpha_merge_status(overlay_master_df, alpha_inputs)
    for _k, _v in alpha_merge_meta.items():
        overlay_master_df[_k] = _v

    scorecard_h19_full_raw = build_horizon_scorecard(
        overlay_master_df, horizon_w=H19_DIAG_W, group_col=None,
        pred_ret_col="predicted_fwd_ret_19w", real_ret_col="realized_fwd_ret_19w"
    )
    scorecard_h19_recent104_raw = build_horizon_scorecard(
        overlay_master_df, horizon_w=H19_DIAG_W, recent_key="recent_104w_flag", group_col=None,
        pred_ret_col="predicted_fwd_ret_19w", real_ret_col="realized_fwd_ret_19w"
    )
    scorecard_h19_recent52_raw = build_horizon_scorecard(
        overlay_master_df, horizon_w=H19_DIAG_W, recent_key="recent_52w_flag", group_col=None,
        pred_ret_col="predicted_fwd_ret_19w", real_ret_col="realized_fwd_ret_19w"
    )
    scorecard_h19_by_regime_raw = build_horizon_scorecard(
        overlay_master_df, horizon_w=H19_DIAG_W, group_col="regime_state",
        pred_ret_col="predicted_fwd_ret_19w", real_ret_col="realized_fwd_ret_19w"
    )
    scorecard_h19_by_raw_regime_raw = build_horizon_scorecard(
        overlay_master_df, horizon_w=H19_DIAG_W, group_col="regime_shifted",
        pred_ret_col="predicted_fwd_ret_19w", real_ret_col="realized_fwd_ret_19w"
    )

    scorecard_h19_full_adj = build_horizon_scorecard(
        overlay_master_df, horizon_w=H19_DIAG_W, group_col=None,
        pred_ret_col="predicted_fwd_ret_19w_adj", real_ret_col="realized_fwd_ret_19w"
    )
    scorecard_h19_recent104_adj = build_horizon_scorecard(
        overlay_master_df, horizon_w=H19_DIAG_W, recent_key="recent_104w_flag", group_col=None,
        pred_ret_col="predicted_fwd_ret_19w_adj", real_ret_col="realized_fwd_ret_19w"
    )
    scorecard_h19_recent52_adj = build_horizon_scorecard(
        overlay_master_df, horizon_w=H19_DIAG_W, recent_key="recent_52w_flag", group_col=None,
        pred_ret_col="predicted_fwd_ret_19w_adj", real_ret_col="realized_fwd_ret_19w"
    )
    scorecard_h19_by_regime_adj = build_horizon_scorecard(
        overlay_master_df, horizon_w=H19_DIAG_W, group_col="regime_state",
        pred_ret_col="predicted_fwd_ret_19w_adj", real_ret_col="realized_fwd_ret_19w"
    )
    scorecard_h19_by_raw_regime_adj = build_horizon_scorecard(
        overlay_master_df, horizon_w=H19_DIAG_W, group_col="regime_shifted",
        pred_ret_col="predicted_fwd_ret_19w_adj", real_ret_col="realized_fwd_ret_19w"
    )

    regime_diag_full = build_regime_diagnostic_table(overlay_master_df, group_col="regime_shifted", recent_key=None)
    regime_diag_recent104 = build_regime_diagnostic_table(overlay_master_df, group_col="regime_shifted", recent_key="recent_104w_flag")
    regime_diag_recent52 = build_regime_diagnostic_table(overlay_master_df, group_col="regime_shifted", recent_key="recent_52w_flag")

    alpha_meta = alpha_inputs.get("meta", {}) if isinstance(alpha_inputs, dict) else {}


    return {
        "liq_source": liq_source,
        "alpha_mode": str(alpha_mode),
        "liq_label": liq_label,
        "liq_snapshot": liq_snapshot,
        "dxy_used": dxy_used,
        "chosen": chosen,
        "best_corr_last": best_corr_last,
        "xx_latest": xx_latest,
        "xx_series": lag_s,
        "btc_end": btc_end,
        "base": base,
        "liq_ret_name": liq_ret.name,
        "full_idx": full_idx,
        "btc_plot": btc_plot,
        "ldli_plot": ldli_plot,
        "liq_plot": liq_plot,
        "dxy_plot": dxy_plot,
        "conflict_mask": conflict_mask,
        "ldli_shifted_latest": ldli_shifted,
        "liq_shifted_latest": liq_shifted_latest,
        "dxy_shifted_latest": dxy_shifted_latest,
        "ldli_level": ldli_level,
        "ldli_liq_contrib": ldli_liq_contrib,
        "ldli_dxy_contrib": ldli_dxy_contrib,
        "regime": regime,
        "regime_shifted": regime_shifted,
        "regime_diag_shifted": regime_diag_shifted,
        "metrics": metrics,
        "pred_hist_endpoint": pred_hist_endpoint,
        "pred_path_latest": pred_path_latest,
        "pred_path_latest_adj": pred_path_latest_adj,
        "pred_path_current_h19_raw": pred_path_current_h19_raw,
        "pred_path_current_h19_adj": pred_path_current_h19_adj,
        "spaghetti_paths": spaghetti_paths,
        "spaghetti_long": spaghetti_long,
        "fc_latest_debug": fc_latest,
        "overlay_master_df": overlay_master_df,
        "scorecard_h19_full_raw": scorecard_h19_full_raw,
        "scorecard_h19_recent104_raw": scorecard_h19_recent104_raw,
        "scorecard_h19_recent52_raw": scorecard_h19_recent52_raw,
        "scorecard_h19_by_regime_raw": scorecard_h19_by_regime_raw,
        "scorecard_h19_by_raw_regime_raw": scorecard_h19_by_raw_regime_raw,
        "scorecard_h19_full_adj": scorecard_h19_full_adj,
        "scorecard_h19_recent104_adj": scorecard_h19_recent104_adj,
        "scorecard_h19_recent52_adj": scorecard_h19_recent52_adj,
        "scorecard_h19_by_regime_adj": scorecard_h19_by_regime_adj,
        "scorecard_h19_by_raw_regime_adj": scorecard_h19_by_raw_regime_adj,
        "regime_diag_full": regime_diag_full,
        "regime_diag_recent104": regime_diag_recent104,
        "regime_diag_recent52": regime_diag_recent52,
        "current_decision": current_decision,
        "alpha_meta": alpha_meta,
        "alpha_merge_meta": alpha_merge_meta,
    }



def make_display_window(payload: dict, past_weeks: int):
    xx = int(payload["xx_latest"])
    alpha_mode = payload.get("alpha_mode", ALPHA_MODE)
    btc_end = payload["btc_end"]

    start_disp = btc_end - pd.to_timedelta(int(past_weeks) * 7, unit="D")
    end_disp = btc_end + pd.to_timedelta(xx * 7, unit="D")
    full_idx = pd.date_range(start=start_disp, end=end_disp, freq=WEEK_RULE)

    btc_plot = payload["base"]["btc_close"].reindex(full_idx)
    ldli_plot = payload["ldli_shifted_latest"].reindex(full_idx)
    liq_plot = payload["liq_shifted_latest"].reindex(full_idx)
    dxy_plot = payload["dxy_shifted_latest"].reindex(full_idx)

    regime_plot = payload["regime_shifted"].reindex(full_idx)
    conflict_mask = (regime_plot == "CONFLICT").fillna(False).to_numpy(dtype=bool)

    return {
        "full_idx": full_idx,
        "btc_plot": btc_plot,
        "ldli_plot": ldli_plot,
        "liq_plot": liq_plot,
        "dxy_plot": dxy_plot,
        "conflict_mask": conflict_mask,
    }


# =========================
# TAB5 backtest helpers (unchanged)
# =========================
def _max_drawdown(equity: pd.Series) -> float:
    if equity is None or equity.dropna().empty:
        return np.nan
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())


def _sharpe_weekly(returns: pd.Series) -> float:
    r = returns.dropna()
    if r.empty:
        return np.nan
    mu = float(r.mean())
    sd = float(r.std(ddof=0))
    if not np.isfinite(sd) or sd == 0:
        return np.nan
    return float((mu / sd) * np.sqrt(52.0))


def _cagr_from_trade_equity(equity: pd.Series) -> float:
    if equity is None or equity.dropna().empty or len(equity) < 2:
        return np.nan
    e0 = float(equity.iloc[0])
    e1 = float(equity.iloc[-1])
    if not (np.isfinite(e0) and np.isfinite(e1)) or e0 <= 0:
        return np.nan
    days = (equity.index[-1] - equity.index[0]).days
    if days <= 0:
        return np.nan
    yrs = days / 365.25
    return float((e1 / e0) ** (1.0 / yrs) - 1.0)


def walk_forward_backtest_overlapping_driversonly(
    px_wclose: pd.Series,
    liq_shifted_full: pd.Series,
    dxy_shifted_full: pd.Series,
    horizon_w: int,
    fit_window_w: int,
    min_obs: int = 40,
    alpha_mode: str = "OLS (learn alpha)",
) -> pd.DataFrame:
    df_drv = pd.concat([
        liq_shifted_full.rename("liq"),
        dxy_shifted_full.rename("dxy"),
    ], axis=1).dropna()
    df_drv["d1"] = df_drv["liq"].diff()
    df_drv["d2"] = df_drv["dxy"].diff()
    df_drv = df_drv.dropna()

    df_fit = pd.concat([
        px_wclose.rename("px"),
        liq_shifted_full.rename("liq"),
        dxy_shifted_full.rename("dxy"),
    ], axis=1).dropna()
    df_fit["r"] = np.log(df_fit["px"]).diff()
    df_fit["d1"] = df_fit["liq"].diff()
    df_fit["d2"] = df_fit["dxy"].diff()
    df_fit = df_fit.dropna()

    if df_fit.shape[0] < fit_window_w + horizon_w + 30 or df_drv.shape[0] < horizon_w + 30:
        return pd.DataFrame()

    idx = df_fit.index
    rows = []

    for pos_fit in range(fit_window_w, len(idx) - 1):
        end_dt = idx[pos_fit]

        if end_dt not in df_drv.index:
            d_idx = df_drv.index[df_drv.index <= end_dt]
            if len(d_idx) == 0:
                continue
            end_dt_drv = d_idx[-1]
        else:
            end_dt_drv = end_dt

        pos_drv = df_drv.index.get_loc(end_dt_drv)
        if pos_drv + horizon_w >= len(df_drv.index):
            break

        dt_real = df_drv.index[pos_drv + horizon_w]
        if end_dt not in px_wclose.index or dt_real not in px_wclose.index:
            continue

        w = df_fit.iloc[pos_fit - fit_window_w:pos_fit]
        y = w["r"].to_numpy(float)
        x1 = w["d1"].to_numpy(float)
        x2 = w["d2"].to_numpy(float)

        res = ols_2var_with_tstats(y, x1, x2, min_obs=max(min_obs, 40))
        if not np.isfinite(res["a"]) or not np.isfinite(res["b1"]) or not np.isfinite(res["b2"]):
            continue

        fut = df_drv.iloc[pos_drv + 1:pos_drv + 1 + horizon_w]
        if fut.shape[0] < horizon_w:
            break

        use_zero_alpha = str(alpha_mode).startswith("ZERO")
        a_eff = 0.0 if use_zero_alpha else float(res.get("a", np.nan))
        if (not use_zero_alpha) and (not np.isfinite(a_eff)):
            # skip if OLS alpha is invalid
            continue

        r_hat = (a_eff + res["b1"] * fut["d1"].astype(float) + res["b2"] * fut["d2"].astype(float)).astype(float)
        exp_cum_logret = float(np.nansum(r_hat.values)) if r_hat.notna().any() else np.nan
        exp_cum_ret = float(np.exp(exp_cum_logret) - 1.0) if np.isfinite(exp_cum_logret) else np.nan

        px0 = float(px_wclose.loc[end_dt])
        pxh = float(px_wclose.loc[dt_real])
        real_cum_logret = float(np.log(pxh / px0)) if (px0 > 0 and pxh > 0) else np.nan
        real_cum_ret = float(np.exp(real_cum_logret) - 1.0) if np.isfinite(real_cum_logret) else np.nan

        sigma_w = float(res.get("sigma", np.nan))
        sigma_h = (sigma_w * math.sqrt(float(horizon_w))) if (np.isfinite(sigma_w) and sigma_w > 0) else np.nan
        z = (exp_cum_logret / sigma_h) if (np.isfinite(exp_cum_logret) and np.isfinite(sigma_h) and sigma_h > 0) else np.nan

        rows.append({
            "end_dt": end_dt,
            "h": int(horizon_w),
            "dt_real": dt_real,
            "px0": px0,
            "pxh": pxh,

            "exp_logret": exp_cum_logret,
            "exp_ret": exp_cum_ret,
            "real_logret": real_cum_logret,
            "real_ret": real_cum_ret,
            "z_raw": z,

            "R2": float(res.get("R2", np.nan)),
            "t1": float(res.get("t1", np.nan)),
            "t2": float(res.get("t2", np.nan)),
            "sigma_w": sigma_w,
        })

    out = pd.DataFrame(rows).set_index("end_dt")
    return out


def _apply_hysteresis(prev_pos: float, target_pos: float, step: float) -> float:
    if not np.isfinite(prev_pos):
        prev_pos = 0.0
    if not np.isfinite(target_pos):
        target_pos = prev_pos
    if step <= 0:
        return float(target_pos)
    if target_pos > prev_pos + step:
        return float(prev_pos + step)
    if target_pos < prev_pos - step:
        return float(prev_pos - step)
    return float(target_pos)


def apply_position_mode_longtrend(
    bt: pd.DataFrame,
    regime_on_timeline: pd.Series,
    mode: str,
    z_smooth_w: int,
    hysteresis_step: float,
    core_floor_static: float,
    floors_by_regime: Dict[str, float],
) -> pd.DataFrame:
    if bt is None or bt.empty:
        return bt

    out = bt.copy()
    z = out["z_raw"].astype(float)
    z_s = z.rolling(int(z_smooth_w), min_periods=max(1, int(z_smooth_w)//2)).mean() if z_smooth_w and z_smooth_w > 1 else z
    out["z"] = z_s

    reg = regime_on_timeline.reindex(out.index).fillna("NEUTRAL")
    out["regime"] = reg

    if mode == "BASE (pos=1/-1/0)":
        pos = np.where(out["z"].values >= 1.0, 1.0, np.where(out["z"].values <= -1.0, -1.0, 0.0))
        out["pos_target"] = pos
        out["pos"] = out["pos_target"].astype(float)
        return out

    if mode == "A: No-SELL (long/cash)":
        pos = np.where(out["z"].values >= 1.0, 1.0, 0.0)
        out["pos_target"] = pos.astype(float)
    elif mode == "B2: z-sizing long-only (pos=clip(z/2,0,1))":
        out["pos_target"] = (out["z"].astype(float) / 2.0).clip(0.0, 1.0).fillna(0.0)
    elif mode == "B2-Core (static+regime floor + hysteresis)":
        base_pos = (out["z"].astype(float) / 2.0).clip(0.0, 1.0).fillna(0.0)

        floor_vec = out["regime"].map(lambda r: float(floors_by_regime.get(str(r), core_floor_static))).astype(float)
        floor_vec = floor_vec.fillna(float(core_floor_static)).clip(0.0, 1.0)

        floor_vec = np.maximum(floor_vec.values, float(core_floor_static))

        out["pos_target"] = np.maximum(base_pos.values, floor_vec).astype(float)
    else:
        out["pos_target"] = 0.0

    pos_series = []
    prev = float(out["pos_target"].iloc[0]) if len(out) > 0 else 0.0
    for v in out["pos_target"].values:
        prev = _apply_hysteresis(prev, float(v), float(hysteresis_step))
        prev = float(np.clip(prev, 0.0, 1.0))
        pos_series.append(prev)
    out["pos"] = np.array(pos_series, dtype=float)
    return out


def simulate_strategy_nonoverlap(bt: pd.DataFrame, horizon_w: int) -> Tuple[pd.DataFrame, pd.Series]:
    if bt is None or bt.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    trade_dates = []
    cur = bt.index.min()
    end = bt.index.max()
    while cur <= end:
        avail = bt.loc[cur:].index
        if len(avail) == 0:
            break
        dt = avail[0]
        trade_dates.append(dt)
        cur = dt + pd.to_timedelta(7 * horizon_w, unit="D")

    trades = bt.loc[trade_dates].copy()
    trades["strategy_ret"] = trades["pos"].astype(float) * trades["real_ret"].astype(float)
    trades["equity"] = (1.0 + trades["strategy_ret"]).cumprod()
    return trades, trades["equity"].copy()


def backtest_summary(bt: pd.DataFrame, trades: pd.DataFrame, equity: pd.Series) -> Dict[str, float]:
    if bt is None or bt.empty:
        return {}

    sub = bt.dropna(subset=["exp_ret", "real_ret"]).copy()
    corr = float(np.corrcoef(sub["exp_ret"].values, sub["real_ret"].values)[0, 1]) if sub.shape[0] > 5 else np.nan
    mae = float(np.nanmean(np.abs(sub["exp_ret"] - sub["real_ret"]))) if sub.shape[0] > 0 else np.nan
    hit = float(np.mean(np.sign(sub["exp_ret"]) == np.sign(sub["real_ret"]))) if sub.shape[0] > 0 else np.nan
    trade_ratio = float((bt["pos"].astype(float) > 0).mean())

    if trades is None or trades.empty or equity is None or equity.dropna().empty:
        return {
            "n_eval": float(sub.shape[0]),
            "corr_exp_real": corr,
            "mae": mae,
            "hit_rate_sign": hit,
            "trade_ratio": trade_ratio,
            "n_trades": 0.0,
            "total_return": np.nan,
            "CAGR": np.nan,
            "Sharpe": np.nan,
            "max_drawdown": np.nan,
            "win_rate_trades": np.nan,
            "avg_trade_ret": np.nan,
        }

    n_trades = int(trades.shape[0])
    total_return = float(equity.iloc[-1] - 1.0)
    mdd = _max_drawdown(equity)
    sharpe = _sharpe_weekly(trades["strategy_ret"])
    cagr = _cagr_from_trade_equity(equity)
    win = float((trades["strategy_ret"] > 0).mean()) if n_trades > 0 else np.nan
    avg = float(trades["strategy_ret"].mean()) if n_trades > 0 else np.nan

    return {
        "n_eval": float(sub.shape[0]),
        "corr_exp_real": corr,
        "mae": mae,
        "hit_rate_sign": hit,
        "trade_ratio": trade_ratio,
        "n_trades": float(n_trades),
        "total_return": total_return,
        "CAGR": cagr,
        "Sharpe": sharpe,
        "max_drawdown": mdd,
        "win_rate_trades": win,
        "avg_trade_ret": avg,
    }


def plot_backtest_scatter(bt: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    sub = bt.dropna(subset=["exp_ret", "real_ret"]).copy()
    ax.scatter(sub["exp_ret"].values, sub["real_ret"].values, alpha=0.6)
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Expected return (horizon)")
    ax.set_ylabel("Realized return (horizon)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_equity_curve(equity: pd.Series, title: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    if equity is not None and not equity.dropna().empty:
        ax.plot(equity.index, equity.values, linewidth=2.0)
    ax.set_title(title)
    ax.set_xlabel("Trade date")
    ax.set_ylabel("Equity (compounded)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


# =========================
# MAIN TAB
# =========================

def run_main_tab():
    st.subheader("MAIN) BTC vs LDLI (+ True-history predicted + Spaghetti)")

    try:
        payload = build_forward_overlay_payload(LIQ_SOURCE, ALPHA_MODE, FORECAST_MODEL, use_binance_funding=USE_BINANCE_FUNDING, funding_symbol=FUNDING_SYMBOL, mvrv_file_bytes=MVRV_FILE_BYTES, mvrv_file_name=MVRV_FILE_NAME)
    except Exception as e:
        st.error(str(e))
        return

    alpha_meta = payload.get("alpha_meta", {})
    st.caption(
        ui_text(
            f"Alpha inputs | Funding: {'loaded' if alpha_meta.get('funding_loaded', False) else ('error' if alpha_meta.get('funding_error') else 'disabled')} ({alpha_meta.get('funding_symbol', 'NA')})"
            f" | MVRV: {'loaded' if alpha_meta.get('mvrv_loaded', False) else 'not loaded'} ({alpha_meta.get('mvrv_source', 'none')})"
        )
    )
    if alpha_meta.get("funding_error"):
        st.warning(ui_text(f"Funding auto-loader warning: {alpha_meta['funding_error']}"))

    xx = int(payload["xx_latest"])
    st.info(
        f"Liquidity={payload['liq_label']} | DXY={payload['dxy_used']} | combo={payload['chosen']} | latest xx={xx}ì£¼"
        f" | ìµê·¼ best_corr_raw={payload['best_corr_last']:.4f}"
        f" | Spaghetti anchors=weekly (dynamic xx per anchor)"
    )

    alpha_meta = payload.get("alpha_meta", {})
    funding_status = "loaded" if alpha_meta.get("funding_loaded", False) else ("error" if alpha_meta.get("funding_error") else "disabled")
    mvrv_status = "loaded" if alpha_meta.get("mvrv_loaded", False) else "not loaded"
    st.caption(
        ui_text(
            f"Alpha inputs | Funding: {funding_status} ({alpha_meta.get('funding_symbol', 'NA')})"
            f" | MVRV: {mvrv_status} ({alpha_meta.get('mvrv_source', 'none')})"
        )
    )
    if alpha_meta.get("funding_error"):
        st.warning(ui_text(f"Funding auto-loader warning: {alpha_meta['funding_error']}"))

    decision = payload.get("current_decision", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Regime", decision.get("current_regime_state", "NA"))
    c2.metric("Confidence", decision.get("current_confidence_bucket", "NA"))
    c3.metric("Pred 19w (raw)", f"{(np.exp(decision.get('current_pred_ret_19w_raw', np.nan)) - 1.0) * 100.0:.1f}%" if np.isfinite(decision.get("current_pred_ret_19w_raw", np.nan)) else "NA")
    c4.metric("Pred 19w (adj)", f"{(np.exp(decision.get('current_pred_ret_19w_adj', np.nan)) - 1.0) * 100.0:.1f}%" if np.isfinite(decision.get("current_pred_ret_19w_adj", np.nan)) else "NA")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Suggested exposure", f"{decision.get('current_suggested_exposure', np.nan):.2f}" if np.isfinite(decision.get("current_suggested_exposure", np.nan)) else "NA")
    c6.metric("Path multiplier", f"{decision.get('current_path_multiplier', np.nan):.2f}" if np.isfinite(decision.get("current_path_multiplier", np.nan)) else "NA")
    c7.metric("Position multiplier", f"{decision.get('current_position_multiplier', np.nan):.2f}" if np.isfinite(decision.get("current_position_multiplier", np.nan)) else "NA")
    recent52 = payload.get("scorecard_h19_recent52_adj", pd.DataFrame())
    if recent52 is not None and not recent52.empty:
        row = recent52.iloc[0]
        c8.metric("Recent52 sign acc", f"{row['sign_acc'] * 100.0:.1f}%")
    else:
        c8.metric("Recent52 sign acc", "NA")

    disp_52 = make_display_window(payload, PAST_WEEKS_FIXED)
    disp_208 = make_display_window(payload, PAST_WEEKS_EXTENDED)
    disp_416 = make_display_window(payload, PAST_WEEKS_LONG)

    pred_hist = payload.get("pred_hist_endpoint", pd.Series(dtype=float))
    pred_path_latest = payload.get("pred_path_latest", pd.Series(dtype=float))
    pred_path_latest_adj = payload.get("pred_path_latest_adj", pd.Series(dtype=float))
    spaghetti_paths = payload.get("spaghetti_paths", [])

    t52, t208, t416 = st.tabs([f"Past {PAST_WEEKS_FIXED}w", f"Past {PAST_WEEKS_EXTENDED}w", f"Past {PAST_WEEKS_LONG}w"])

    with t52:
        fig = plot_main_overlay_with_predline(
            dates=disp_52["full_idx"],
            btc_price=disp_52["btc_plot"].values,
            ldli_level=disp_52["ldli_plot"].values,
            ldli_liq=disp_52["liq_plot"].values,
            ldli_dxy=disp_52["dxy_plot"].values,
            pred_hist_endpoint=pred_hist,
            pred_path_latest=pred_path_latest,
            pred_path_latest_adj=pred_path_latest_adj,
            spaghetti_paths=spaghetti_paths,
            title=f"BTC vs LDLI + True-history predicted + Spaghetti (latest shift +{xx}w; past={PAST_WEEKS_FIXED}w, future=+{xx}w)",
            btc_end=payload["btc_end"],
            conflict_mask=disp_52["conflict_mask"],
            y2_label=f"LDLI components (shifted +{xx}w)"
        )
        st.pyplot(fig)

    with t208:
        fig2 = plot_main_overlay_with_predline(
            dates=disp_208["full_idx"],
            btc_price=disp_208["btc_plot"].values,
            ldli_level=disp_208["ldli_plot"].values,
            ldli_liq=disp_208["liq_plot"].values,
            ldli_dxy=disp_208["dxy_plot"].values,
            pred_hist_endpoint=pred_hist,
            pred_path_latest=pred_path_latest,
            pred_path_latest_adj=pred_path_latest_adj,
            spaghetti_paths=spaghetti_paths,
            title=f"BTC vs LDLI + True-history predicted + Spaghetti (latest shift +{xx}w; past={PAST_WEEKS_EXTENDED}w, future=+{xx}w)",
            btc_end=payload["btc_end"],
            conflict_mask=disp_208["conflict_mask"],
            y2_label=f"LDLI components (shifted +{xx}w)"
        )
        st.pyplot(fig2)

    with t416:
        fig3 = plot_main_overlay_with_predline(
            dates=disp_416["full_idx"],
            btc_price=disp_416["btc_plot"].values,
            ldli_level=disp_416["ldli_plot"].values,
            ldli_liq=disp_416["liq_plot"].values,
            ldli_dxy=disp_416["dxy_plot"].values,
            pred_hist_endpoint=pred_hist,
            pred_path_latest=pred_path_latest,
            pred_path_latest_adj=pred_path_latest_adj,
            spaghetti_paths=spaghetti_paths,
            title=f"BTC vs LDLI + True-history predicted + Spaghetti (latest shift +{xx}w; past={PAST_WEEKS_LONG}w, future=+{xx}w)",
            btc_end=payload["btc_end"],
            conflict_mask=disp_416["conflict_mask"],
            y2_label=f"LDLI components (shifted +{xx}w)"
        )
        st.pyplot(fig3)

    st.markdown(ui_text("### Regime ì±ê³¼ ìì½ (BTC timeline ê¸°ì¤, regimeë +xxì£¼ shift ì ì©)"))
    met = payload.get("metrics", pd.DataFrame())
    if met is None or met.empty:
        st.warning("ì±ê³¼ ìì½ì ê³ì°í  ë°ì´í°ê° ë¶ì¡±í©ëë¤.")
    else:
        show = met.copy()
        for col in ["mean_fwd", "median_fwd", "avg_gain", "avg_loss", "mae_median", "mfe_median"]:
            show[col] = (np.exp(show[col]) - 1.0) * 100.0
        show["win_rate"] = show["win_rate"] * 100.0
        st.dataframe(show)

    st.markdown("### H19 Scorecard (Adjusted)")
    s1, s2, s3 = st.tabs(["Full sample", "Recent 104w", "By regime"])
    with s1:
        st.dataframe(payload.get("scorecard_h19_full_adj", pd.DataFrame()))
    with s2:
        st.dataframe(payload.get("scorecard_h19_recent104_adj", pd.DataFrame()))
    with s3:
        st.dataframe(payload.get("scorecard_h19_by_regime_adj", pd.DataFrame()))

    st.markdown("### ONE FILE CSV ë¤ì´ë¡ë (MAIN/TAB4 ë¶ìì©, Past 416w + Future +xx)")
    try:
        full_idx = disp_416["full_idx"]
        master = payload.get("overlay_master_df", pd.DataFrame())
        df_one = pd.DataFrame(index=full_idx)
        df_one["btc_close"] = payload["base"]["btc_close"].reindex(full_idx)

        df_one["ldli_shifted_total"] = payload["ldli_shifted_latest"].reindex(full_idx)
        df_one["ldli_shifted_liq"] = payload["liq_shifted_latest"].reindex(full_idx)
        df_one["ldli_shifted_dxy"] = payload["dxy_shifted_latest"].reindex(full_idx)

        df_one["regime_shifted"] = payload["regime_shifted"].reindex(full_idx)
        df_one["conflict_flag"] = (df_one["regime_shifted"] == "CONFLICT").astype(int)

        df_one["pred_hist_endpoint_dynamic"] = payload["pred_hist_endpoint"].reindex(full_idx)
        df_one[f"latest_forecast_path_raw_to_{xx}w"] = payload["pred_path_latest"].reindex(full_idx)
        df_one[f"latest_forecast_path_adj_to_{xx}w"] = payload["pred_path_latest_adj"].reindex(full_idx)

        enrich_cols = [
            "realized_fwd_px_19w", "realized_fwd_ret_19w", "realized_sign_19w",
            "predicted_fwd_px_19w", "predicted_fwd_ret_19w", "predicted_sign_19w",
            "predicted_fwd_px_19w_adj", "predicted_fwd_ret_19w_adj", "predicted_sign_19w_adj",
            "signal_hit_19w", "regime_state", "regime_path_multiplier", "regime_position_multiplier",
            "confidence_bucket", "suggested_exposure", "recent_52w_flag", "recent_104w_flag",
            "alpha_state", "mvrv_state", "funding_state", "endpoint_dt_h19", "lag_weeks_h19",
            "z_liq_s", "z_dxy_s", "dir_liq_sum", "dir_dxy_sum", "dir_liq_s", "dir_dxy_s", "dir_sum",
            "raw_regime_base", "raw_regime_rule_code", "raw_conflict_flag",
            "funding_rate_w", "funding_8w_ma", "funding_z", "mvrv_z",
            "funding_available_flag", "funding_source", "funding_last_valid_dt", "funding_nonnull_count",
            "funding_weekly_mean_latest", "funding_8w_ma_latest", "funding_8w_ma_z_latest", "funding_state_latest",
            "mvrv_available_flag", "mvrv_source", "mvrv_last_valid_dt", "mvrv_nonnull_count", "alpha_inputs_ready_flag",
            "current_anchor_dt", "current_predicted_ret_19w_raw", "current_predicted_ret_19w_adj",
            "current_predicted_px_19w_raw", "current_predicted_px_19w_adj",
            "current_regime_state", "current_confidence_bucket",
            "current_regime_path_multiplier", "current_regime_position_multiplier", "current_suggested_exposure"
        ]
        for c in enrich_cols:
            if c in master.columns:
                df_one[c] = master[c].reindex(full_idx)

        df_one["liq_source"] = payload["liq_source"]
        df_one["alpha_mode"] = payload.get("alpha_mode", "OLS (learn alpha)")
        df_one["combo_type"] = payload["chosen"]
        df_one["xx_latest"] = xx

        preferred_cols = [
            "btc_close",
            "ldli_shifted_total", "ldli_shifted_liq", "ldli_shifted_dxy",
            "regime_shifted", "conflict_flag", "regime_state",
            "z_liq_s", "z_dxy_s", "dir_liq_sum", "dir_dxy_sum", "dir_liq_s", "dir_dxy_s", "dir_sum",
            "raw_regime_base", "raw_regime_rule_code", "raw_conflict_flag",
            "realized_fwd_ret_19w", "predicted_fwd_ret_19w", "predicted_fwd_ret_19w_adj",
            "realized_sign_19w", "predicted_sign_19w", "predicted_sign_19w_adj", "signal_hit_19w",
            "regime_path_multiplier", "regime_position_multiplier", "confidence_bucket", "suggested_exposure",
            "funding_rate_w", "funding_8w_ma", "funding_z", "funding_state",
            "mvrv_z", "mvrv_state", "alpha_state",
            "funding_available_flag", "funding_source", "funding_last_valid_dt", "funding_nonnull_count",
            "funding_weekly_mean_latest", "funding_8w_ma_latest", "funding_8w_ma_z_latest", "funding_state_latest",
            "mvrv_available_flag", "mvrv_source", "mvrv_last_valid_dt", "mvrv_nonnull_count", "alpha_inputs_ready_flag",
            "current_anchor_dt", "current_predicted_ret_19w_raw", "current_predicted_ret_19w_adj",
            "current_predicted_px_19w_raw", "current_predicted_px_19w_adj",
            "current_regime_state", "current_confidence_bucket",
            "current_regime_path_multiplier", "current_regime_position_multiplier", "current_suggested_exposure",
            f"latest_forecast_path_raw_to_{xx}w", f"latest_forecast_path_adj_to_{xx}w",
            "liq_source", "alpha_mode", "combo_type", "xx_latest",
        ]
        existing_cols = [c for c in preferred_cols if c in df_one.columns]
        other_cols = [c for c in df_one.columns if c not in existing_cols]
        df_one = df_one[existing_cols + other_cols]

        st.download_button(
            "Download MAIN/TAB4 one-file CSV",
            data=df_one.reset_index().rename(columns={"index": "date"}).to_csv(index=False).encode("utf-8-sig"),
            file_name=f"MAIN_TAB4_ONEFILE_{payload['liq_source'].replace(' ','_')}_past{PAST_WEEKS_LONG}w_future{xx}w_{APP_VERSION}.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.warning(f"ONE FILE CSV ìì± ì¤ ì¤ë¥: {e}")

    st.markdown("### Liquidity snapshot (raw series alignment)")
    st.dataframe(payload["liq_snapshot"].tail(60))
    st.download_button(
        "Download Liquidity Snapshot CSV",
        data=payload["liq_snapshot"].assign(alpha_mode=payload.get("alpha_mode", "OLS (learn alpha)")).reset_index().to_csv(index=False).encode("utf-8-sig"),
        file_name=f"liquidity_snapshot_{payload['liq_source'].replace(' ','_')}.csv",
        mime="text/csv",
    )

    st.markdown("### Spaghetti (long format)")
    sp_long = payload["spaghetti_long"]
    st.dataframe(sp_long.tail(200))
    st.download_button(
        "Download Spaghetti CSV (long format)",
        data=sp_long.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"spaghetti_long_dynamicLag_weeklyAnchors_{payload['liq_source'].replace(' ','_')}_xxLatest{xx}_{APP_VERSION}.csv",
        mime="text/csv",
    )


def run_tab_dxy():
    st.subheader("TAB1) DXY(ì­ì¶) â BTC (Weekly Returns)")

    btc_close = load_btc_close()
    dxy_close, dxy_used = load_dxy_close()
    st.caption(f"Price data loaded | DXY={dxy_used}")

    btc_wret = weekly_log_returns(btc_close, WEEK_RULE).rename("btc_wret")
    dxy_wret = weekly_log_returns(dxy_close, WEEK_RULE).rename("dxy_wret")

    lags_weeks = range(LAG_MIN_WEEKS, LAG_MAX_WEEKS + 1)
    with st.spinner("Rolling heatmaps ê³ì° ì¤..."):
        corr_raw, corr_masked, beta_map, tstat_map, tstat_masked, out = rolling_maps_weekly(
            y_w=btc_wret,
            x_w=dxy_wret,
            invert_x=TAB1_INVERT_X,
            lags_weeks=lags_weeks,
            quiet=False
        )

    render_model_section("DXY(ì­ì¶) vs BTC", corr_masked, beta_map, tstat_masked, out)


# =========================
# TAB2
# =========================
def run_tab_liquidity():
    st.subheader("TAB2) Liquidity â BTC (Weekly; RETURNS vs DELTA ë¹êµ)")

    btc_close = load_btc_close()
    btc_wret = weekly_log_returns(btc_close, WEEK_RULE).rename("btc_wret")

    liq_level_daily, snapshot, liq_label, unit_label = load_liquidity_source_daily(LIQ_SOURCE)
    st.caption(f"Liquidity source: {liq_label} | Units: {unit_label}")
    st.dataframe(snapshot.tail(5))

    liq_week = liq_level_daily.resample(WEEK_RULE).last().dropna()
    if (liq_week <= 0).any():
        liq_ret = weekly_pct_change(liq_level_daily, WEEK_RULE).rename("liq_ret_pct")
        st.info("Returns input: pct_change (<=0 level detected)")
    else:
        liq_ret = weekly_log_returns(liq_level_daily, WEEK_RULE).rename("liq_ret_log")
        st.info("Returns input: log returns")

    liq_dlt = weekly_delta(liq_level_daily, WEEK_RULE).rename("liq_delta_level")

    lags_weeks = range(LAG_MIN_WEEKS, LAG_MAX_WEEKS + 1)
    with st.spinner("RETURNS/DELTA ë ëª¨ë¸ì ëª¨ë ê³ì° ì¤..."):
        corr_raw_r, corr_masked_r, beta_map_r, tstat_map_r, tstat_masked_r, out_r = rolling_maps_weekly(
            y_w=btc_wret, x_w=liq_ret, invert_x=TAB2_INVERT_X, lags_weeks=lags_weeks, quiet=False
        )
        corr_raw_d, corr_masked_d, beta_map_d, tstat_map_d, tstat_masked_d, out_d = rolling_maps_weekly(
            y_w=btc_wret, x_w=liq_dlt, invert_x=TAB2_INVERT_X, lags_weeks=lags_weeks, quiet=False
        )

    sub1, sub2 = st.tabs(["Liquidity Returns", "Liquidity Delta(Î)"])
    with sub1:
        render_model_section("Liquidity RETURNS vs BTC", corr_masked_r, beta_map_r, tstat_masked_r, out_r)
    with sub2:
        render_model_section("Liquidity DELTA(Î) vs BTC", corr_masked_d, beta_map_d, tstat_masked_d, out_d)


# =========================
# TAB3
# =========================
def run_tab_combo():
    st.subheader("TAB3) Combo(ì ëì± + ë¬ë¬ê°ë) â BTC")
    st.caption("corr-ìµë ë°©ì ì ì§ | 1D Combo + 2D lag-pair(corr-max) + ë¤ë³ë(ì°¸ê³ )")

    btc_close = load_btc_close()
    dxy_close, dxy_used = load_dxy_close()
    liq_level_daily, snapshot, liq_label, unit_label = load_liquidity_source_daily(LIQ_SOURCE)

    st.write(f"Price data loaded | DXY={dxy_used} | Liquidity: {liq_label}")
    st.dataframe(snapshot.tail(5))

    btc_wret = weekly_log_returns(btc_close, WEEK_RULE).rename("btc_wret")
    dxy_wret = weekly_log_returns(dxy_close, WEEK_RULE).rename("dxy_wret")

    liq_week = liq_level_daily.resample(WEEK_RULE).last().dropna()
    if (liq_week <= 0).any():
        liq_ret = weekly_pct_change(liq_level_daily, WEEK_RULE).rename("liq_ret_pct")
        ret_note = "Liquidity returns: pct_change (<=0 level detected)"
    else:
        liq_ret = weekly_log_returns(liq_level_daily, WEEK_RULE).rename("liq_ret_log")
        ret_note = "Liquidity returns: log returns"
    liq_dlt = weekly_delta(liq_level_daily, WEEK_RULE).rename("liq_delta_level")
    st.info(ret_note)

    base = pd.concat([btc_wret, dxy_wret, liq_ret, liq_dlt], axis=1).dropna()
    lags_weeks = range(LAG_MIN_WEEKS, LAG_MAX_WEEKS + 1)

    z_dxy_inv = zscore(-base["dxy_wret"]).rename("z(-dxy_ret)")
    z_liq_ret = zscore(base[liq_ret.name]).rename("z(liq_ret)")
    z_liq_dlt = zscore(base["liq_delta_level"]).rename("z(liq_delta)")

    combo_ret = (z_liq_ret + z_dxy_inv).rename("combo_ret")
    combo_dlt = (z_liq_dlt + z_dxy_inv).rename("combo_delta")

    y = base["btc_wret"]

    with st.spinner("1D Combo(RET/DELTA) ê³ì° ì¤..."):
        corr_raw_cr, corr_masked_cr, beta_map_cr, tstat_map_cr, tstat_masked_cr, out_cr = rolling_maps_weekly(
            y_w=y, x_w=combo_ret, invert_x=TAB3_INVERT_X, lags_weeks=lags_weeks, quiet=False
        )
        corr_raw_cd, corr_masked_cd, beta_map_cd, tstat_map_cd, tstat_masked_cd, out_cd = rolling_maps_weekly(
            y_w=y, x_w=combo_dlt, invert_x=TAB3_INVERT_X, lags_weeks=lags_weeks, quiet=False
        )

    last_cr = out_cr.dropna(subset=["best_corr_raw"]).iloc[-1]
    last_cd = out_cd.dropna(subset=["best_corr_raw"]).iloc[-1]
    if float(last_cr["best_corr_raw"]) >= float(last_cd["best_corr_raw"]):
        chosen = "RET"
        out_use = out_cr
        best_corr_last = float(last_cr["best_corr_raw"])
    else:
        chosen = "DELTA"
        out_use = out_cd
        best_corr_last = float(last_cd["best_corr_raw"])

    out_use2 = out_use.dropna(subset=["best_lag_raw_weeks"])
    lag_series = out_use2["best_lag_valid_smooth_weeks"]
    lag_latest = lag_series.dropna().iloc[-1] if lag_series.dropna().size > 0 else out_use2["best_lag_raw_weeks"].iloc[-1]
    xx = int(np.clip(int(round(float(lag_latest))), LAG_MIN_WEEKS, LAG_MAX_WEEKS))

    st.session_state["tab3_chosen_combo"] = chosen
    st.session_state["tab3_xx_weeks"] = xx
    st.session_state["tab3_best_corr_last"] = best_corr_last

    sub1, sub2 = st.tabs(["1D Combo", "2D Lag-Pair + Multivariate"])
    with sub1:
        s1, s2 = st.tabs(["Combo using Liquidity RETURNS", "Combo using Liquidity DELTA(Î)"])
        with s1:
            render_model_section("1D COMBO (z(Liq RETURNS) + z(-DXY_ret)) vs BTC", corr_masked_cr, beta_map_cr, tstat_masked_cr, out_cr)
        with s2:
            render_model_section("1D COMBO (z(Liq DELTA) + z(-DXY_ret)) vs BTC", corr_masked_cd, beta_map_cd, tstat_masked_cd, out_cd)

        st.success(f"TAB4/MAINì ì¬ì©ë  ìµì  lag(xx): {xx}ì£¼ | ì í combo={chosen} | ìµê·¼ best_corr_raw={best_corr_last:.4f}")

    with sub2:
        st.caption(f"2D/ë¤ë³ëì ê³ì°ë ëë¬¸ì STEP={STEP_WEEKS_2D}ì£¼ë¡ íê°í©ëë¤. (pair ì íì corr-max)")
        with st.spinner("2D lag-pair + ë¤ë³ë(RET/DELTA) ê³ì° ì¤..."):
            best2d_ret, pair_counts_ret, lags_list = rolling_best_pair_and_multivar(
                y=y, liq=base[liq_ret.name], dxy_ret=base["dxy_wret"],
                lags_weeks=lags_weeks, window_weeks=WINDOW_WEEKS, step_weeks=STEP_WEEKS_2D
            )
            best2d_dlt, pair_counts_dlt, _ = rolling_best_pair_and_multivar(
                y=y, liq=base["liq_delta_level"], dxy_ret=base["dxy_wret"],
                lags_weeks=lags_weeks, window_weeks=WINDOW_WEEKS, step_weeks=STEP_WEEKS_2D
            )

        s1, s2 = st.tabs(["2D using Liquidity RETURNS", "2D using Liquidity DELTA(Î)"])
        with s1:
            st.pyplot(plot_pair_count_heatmap(pair_counts_ret, lags_list, "Best (L_liq, L_dxy) selection counts (RET)"))
            st.dataframe(best2d_ret.tail(30))
        with s2:
            st.pyplot(plot_pair_count_heatmap(pair_counts_dlt, lags_list, "Best (L_liq, L_dxy) selection counts (DELTA)"))
            st.dataframe(best2d_dlt.tail(30))


# =========================
# TAB4 (LDLI vs BTC)
# =========================

def run_tab_forward_overlay():
    st.subheader("TAB4) BTC vs LDLI(ì ëì±+ë¬ë¬ê°ë) ì í ì¤ë²ë ì´ (Forward-Look Window)")

    try:
        payload = build_forward_overlay_payload(LIQ_SOURCE, ALPHA_MODE, FORECAST_MODEL, use_binance_funding=USE_BINANCE_FUNDING, funding_symbol=FUNDING_SYMBOL, mvrv_file_bytes=MVRV_FILE_BYTES, mvrv_file_name=MVRV_FILE_NAME)
    except Exception as e:
        st.error(str(e))
        return

    xx = int(payload["xx_latest"])
    st.success(
        f"ì¬ì© lag(latest xx)={xx}ì£¼ | ì í combo={payload['chosen']} | ìµê·¼ best_corr_raw={payload['best_corr_last']:.4f}"
    )

    full_idx = payload["full_idx"]
    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.plot(full_idx, payload["btc_plot"].values, label="BTC Price", linewidth=2.3, color="tab:blue")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("BTC Price")
    ax1.grid(True, alpha=0.25)

    if payload.get("pred_path_latest") is not None and not payload["pred_path_latest"].dropna().empty:
        ax1.plot(payload["pred_path_latest"].index, payload["pred_path_latest"].values, label="Latest path (raw)", linewidth=2.0, linestyle="--", color="tab:blue", alpha=0.6)
    if payload.get("pred_path_latest_adj") is not None and not payload["pred_path_latest_adj"].dropna().empty:
        ax1.plot(payload["pred_path_latest_adj"].index, payload["pred_path_latest_adj"].values, label="Latest path (adjusted)", linewidth=2.2, linestyle=":", color="tab:red", alpha=0.9)

    ax2 = ax1.twinx()
    ax2.plot(full_idx, payload["ldli_plot"].values, label=f"LDLI Level (shifted +{xx}w)", linewidth=2.3, color="tab:orange")
    ax2.set_ylabel(f"LDLI (shifted +{xx}w)")

    ax1.set_title(f"BTC vs LDLI (past=52w fixed, future=+{xx}w) | Liquidity={payload['liq_label']}")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")
    fig.tight_layout()
    st.pyplot(fig)

    st.markdown("### H19 Scorecard (Raw vs Adjusted)")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Raw")
        st.dataframe(payload.get("scorecard_h19_by_regime_raw", pd.DataFrame()))
    with col2:
        st.caption("Adjusted")
        st.dataframe(payload.get("scorecard_h19_by_regime_adj", pd.DataFrame()))

    st.markdown("### H19 Recent 104w / 52w")
    col3, col4 = st.columns(2)
    with col3:
        st.caption("Recent104 Raw / Adjusted")
        st.dataframe(payload.get("scorecard_h19_recent104_raw", pd.DataFrame()))
        st.dataframe(payload.get("scorecard_h19_recent104_adj", pd.DataFrame()))
    with col4:
        st.caption("Recent52 Raw / Adjusted")
        st.dataframe(payload.get("scorecard_h19_recent52_raw", pd.DataFrame()))
        st.dataframe(payload.get("scorecard_h19_recent52_adj", pd.DataFrame()))

    st.markdown("### Raw Regime Validation (TAILWIND/HEADWIND ì ì ê²ì¦ì©)")
    col5, col6 = st.columns(2)
    with col5:
        st.caption("By raw regime - Adjusted H19 scorecard")
        st.dataframe(payload.get("scorecard_h19_by_raw_regime_adj", pd.DataFrame()))
    with col6:
        st.caption("Regime diagnostic means")
        st.dataframe(payload.get("regime_diag_recent104", pd.DataFrame()))

    st.markdown("### Current Decision Audit")
    cur = payload.get("current_decision", {})
    if cur:
        st.dataframe(pd.DataFrame([{
            "anchor_dt": cur.get("current_anchor_dt"),
            "pred_ret_19w_raw": cur.get("current_predicted_ret_19w_raw", cur.get("current_pred_ret_19w_raw")),
            "pred_ret_19w_adj": cur.get("current_predicted_ret_19w_adj", cur.get("current_pred_ret_19w_adj")),
            "pred_px_19w_raw": cur.get("current_predicted_px_19w_raw"),
            "pred_px_19w_adj": cur.get("current_predicted_px_19w_adj"),
            "regime_state": cur.get("current_regime_state"),
            "confidence": cur.get("current_confidence_bucket"),
            "path_mult": cur.get("current_regime_path_multiplier", cur.get("current_path_multiplier")),
            "pos_mult": cur.get("current_regime_position_multiplier", cur.get("current_position_multiplier")),
            "suggested_exposure": cur.get("current_suggested_exposure"),
        }]), use_container_width=True)

    st.markdown("### Funding / Alpha Merge Audit")
    meta = payload.get("alpha_merge_meta", {})
    if meta:
        st.dataframe(pd.DataFrame([meta]), use_container_width=True)

    master = payload.get("overlay_master_df", pd.DataFrame())
    if master is not None and not master.empty:
        st.markdown("### Enriched H19 Master (tail)")
        st.dataframe(master.tail(30))
        st.download_button(
            "Download Enriched H19 Master CSV",
            data=master.reset_index().rename(columns={"index": "date"}).to_csv(index=False).encode("utf-8-sig"),
            file_name=f"overlay_master_h19_{payload['liq_source'].replace(' ','_')}_{APP_VERSION}.csv",
            mime="text/csv",
        )

        score_full = payload.get("scorecard_h19_full_adj", pd.DataFrame())
        score_recent = payload.get("scorecard_h19_recent104_adj", pd.DataFrame())
        score_recent52 = payload.get("scorecard_h19_recent52_adj", pd.DataFrame())
        score_reg = payload.get("scorecard_h19_by_regime_adj", pd.DataFrame())
        score_reg_raw = payload.get("scorecard_h19_by_raw_regime_adj", pd.DataFrame())
        regime_diag_recent104 = payload.get("regime_diag_recent104", pd.DataFrame())
        if score_full is not None and not score_full.empty:
            st.download_button(
                "Download H19 Scorecard Full CSV",
                data=score_full.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"scorecard_h19_full_{APP_VERSION}.csv",
                mime="text/csv",
            )
        if score_recent is not None and not score_recent.empty:
            st.download_button(
                "Download H19 Scorecard Recent104 CSV",
                data=score_recent.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"scorecard_h19_recent104_{APP_VERSION}.csv",
                mime="text/csv",
            )
        if score_recent52 is not None and not score_recent52.empty:
            st.download_button(
                "Download H19 Scorecard Recent52 CSV",
                data=score_recent52.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"scorecard_h19_recent52_{APP_VERSION}.csv",
                mime="text/csv",
            )
        if score_reg is not None and not score_reg.empty:
            st.download_button(
                "Download H19 Scorecard By-Regime CSV",
                data=score_reg.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"scorecard_h19_by_regime_{APP_VERSION}.csv",
                mime="text/csv",
            )
        if score_reg_raw is not None and not score_reg_raw.empty:
            st.download_button(
                "Download H19 Scorecard By-Raw-Regime CSV",
                data=score_reg_raw.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"scorecard_h19_by_raw_regime_{APP_VERSION}.csv",
                mime="text/csv",
            )
        if regime_diag_recent104 is not None and not regime_diag_recent104.empty:
            st.download_button(
                "Download Regime Diagnostic Recent104 CSV",
                data=regime_diag_recent104.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"regime_diag_recent104_{APP_VERSION}.csv",
                mime="text/csv",
            )


# =========================
# TAB5 (Multi-Asset Forecast/Backtest)
# =========================


def run_tab5_multiasset():
    st.subheader("TAB5) Multi-Asset Forecast + Walk-forward Backtest (drivers-only future; long-trend position modes)")

    try:
        payload = build_forward_overlay_payload(LIQ_SOURCE, ALPHA_MODE, FORECAST_MODEL, use_binance_funding=USE_BINANCE_FUNDING, funding_symbol=FUNDING_SYMBOL, mvrv_file_bytes=MVRV_FILE_BYTES, mvrv_file_name=MVRV_FILE_NAME)
    except Exception as e:
        st.error(str(e))
        return

    xx = int(payload["xx_latest"])
    st.info(f"Drivers: combo={payload['chosen']} | latest xx={xx}w | DXY={payload['dxy_used']} | Liquidity={payload['liq_label']}")

    asset_key = st.selectbox("Asset", list(ASSET_SYMBOLS.keys()), index=0)
    horizon_w_sel = st.selectbox("Horizon (weeks)", [4, 11, 13, 26], index=1)
    horizon_w = int(min(int(horizon_w_sel), xx))
    fit_w = st.selectbox("Fit window (weeks)", [78, 104, 156], index=1)

    st.markdown("### TAB5 í¬ì§ì ì¤ì  (ì¥ê¸° ì¶ì¸ í¬ìí)")
    mode = st.selectbox(
        "Strategy / Position mode",
        [
            "B2-Core (static+regime floor + hysteresis)",
            "B2: z-sizing long-only (pos=clip(z/2,0,1))",
            "A: No-SELL (long/cash)",
            "BASE (pos=1/-1/0)",
        ],
        index=0
    )
    z_smooth_w = st.slider("z smoothing window (weeks)", min_value=1, max_value=13, value=3, step=1)
    hysteresis_step = st.slider("position hysteresis step", min_value=0.0, max_value=0.50, value=0.10, step=0.01)

    core_floor_static = st.slider("Static core floor (for B2-Core / fallback)", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    floor_tail = st.slider("TAILWIND floor", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    floor_conf = st.slider("CONFLICT floor", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    floor_head = st.slider("HEADWIND floor", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    floor_neut = st.slider("NEUTRAL floor", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

    floors_by_regime = {
        "TAILWIND": float(floor_tail),
        "CONFLICT": float(floor_conf),
        "HEADWIND": float(floor_head),
        "NEUTRAL": float(floor_neut),
    }

    try:
        s_close, used_sym = load_asset_close(asset_key)
        st.caption(f"{asset_key} loaded via {used_sym}")
    except Exception as e:
        st.error(f"Asset load failed: {e}")
        return

    asset_wclose = s_close.resample(WEEK_RULE).last().dropna()
    if asset_wclose.empty or asset_wclose.shape[0] < (fit_w + horizon_w + 20):
        st.warning("ìì° ì£¼ê° ë°ì´í°ê° ë¶ì¡±í©ëë¤. (ê¸°ê°/fit/horizon ì¡°ì  íì)")
        return

    # Drivers on weekly axis (use latest-shifted drivers)
    liq_shifted_full = payload["liq_shifted_latest"].resample(WEEK_RULE).last().dropna()
    dxy_shifted_full = payload["dxy_shifted_latest"].resample(WEEK_RULE).last().dropna()

    regime_on_asset_timeline = payload["regime_shifted"].resample(WEEK_RULE).last()

    end_dt_req = asset_wclose.index.max()
    try:
        fc = forecast_path_from_drivers_driversonly(
            px_wclose=asset_wclose,
            liq_shifted=liq_shifted_full,
            dxy_shifted=dxy_shifted_full,
            horizon_w=horizon_w,
            fit_window_w=int(fit_w),
            end_dt_requested=end_dt_req,
            alpha_mode=alpha_mode,
        )
    except Exception as e:
        st.error(f"Forecast failed: {e}")
        return

    st.caption(
        f"Forecast anchor: requested end_dt={pd.Timestamp(end_dt_req).date()} â used end_dt_eff={pd.Timestamp(fc['end_dt_eff']).date()} "
        f"| drivers-max_h(at end_dt_eff)={fc.get('max_h', np.nan)}w | h_used={fc.get('h', np.nan)}w"
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    exp_ret = float(np.exp(fc["exp_logret"]) - 1.0) if np.isfinite(fc["exp_logret"]) else np.nan
    c1.metric("z", f"{fc['z']:.2f}" if np.isfinite(fc["z"]) else "N/A")
    c2.metric("Expected return", f"{exp_ret*100:.1f}%" if np.isfinite(exp_ret) else "N/A")
    c3.metric("R2", f"{fc['R2']:.3f}" if np.isfinite(fc["R2"]) else "N/A")
    c4.metric("sigma_w", f"{fc['sigma_w']:.4f}" if np.isfinite(fc["sigma_w"]) else "N/A")
    c5.metric("h_used", f"{int(fc['h'])}" if np.isfinite(fc["h"]) else "N/A")

    fig, ax = plt.subplots(figsize=(12, 4))
    ps = fc["path_series"]
    ax.plot(ps.index, ps.values, linewidth=2.5, label="Forecast path (drivers-only)")
    ax.axhline(float(fc["px0"]), linewidth=1.2, linestyle="--", label="Current")
    ax.set_title(f"{asset_key} | Forecast path (h={int(fc['h'])}w, requested={horizon_w}w)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (model-implied)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    st.pyplot(fig)

    st.markdown("### Walk-forward Backtest (drivers-only future; realized at tât+h)")
    with st.spinner("Backtest ê³ì° ì¤..."):
        bt = walk_forward_backtest_overlapping_driversonly(
            px_wclose=asset_wclose,
            liq_shifted_full=liq_shifted_full,
            dxy_shifted_full=dxy_shifted_full,
            horizon_w=int(horizon_w),
            fit_window_w=int(fit_w),
            min_obs=max(MIN_OBS, 40),
            alpha_mode=payload.get("alpha_mode", ALPHA_MODE),
        )

    if bt is None or bt.empty:
        st.warning("Backtest ê²°ê³¼ê° ë¹ì´ ììµëë¤. (ê³µíµêµ¬ê° ë¶ì¡±)")
        return

    bt = bt.copy()
    bt["exp_ret"] = bt["exp_ret"].astype(float)
    bt["real_ret"] = bt["real_ret"].astype(float)

    bt2 = apply_position_mode_longtrend(
        bt=bt,
        regime_on_timeline=regime_on_asset_timeline,
        mode=mode,
        z_smooth_w=int(z_smooth_w),
        hysteresis_step=float(hysteresis_step),
        core_floor_static=float(core_floor_static),
        floors_by_regime=floors_by_regime,
    )

    trades, equity = simulate_strategy_nonoverlap(bt2, horizon_w=int(horizon_w))
    summ = backtest_summary(bt2, trades, equity)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Eval samples", f"{int(summ.get('n_eval', 0))}")
    c2.metric("Corr(exp,real)", f"{summ.get('corr_exp_real', np.nan):.3f}" if np.isfinite(summ.get("corr_exp_real", np.nan)) else "N/A")
    c3.metric("MAE", f"{summ.get('mae', np.nan):.1%}" if np.isfinite(summ.get("mae", np.nan)) else "N/A")
    c4.metric("Sign hit", f"{summ.get('hit_rate_sign', np.nan):.1%}" if np.isfinite(summ.get("hit_rate_sign", np.nan)) else "N/A")
    c5.metric("Trade ratio", f"{summ.get('trade_ratio', np.nan):.1%}" if np.isfinite(summ.get("trade_ratio", np.nan)) else "N/A")

    c6, c7, c8, c9, c10 = st.columns(5)
    c6.metric("Trades", f"{int(summ.get('n_trades', 0))}")
    c7.metric("Total return", f"{summ.get('total_return', np.nan):.1%}" if np.isfinite(summ.get("total_return", np.nan)) else "N/A")
    c8.metric("CAGR", f"{summ.get('CAGR', np.nan):.1%}" if np.isfinite(summ.get("CAGR", np.nan)) else "N/A")
    c9.metric("Sharpe", f"{summ.get('Sharpe', np.nan):.2f}" if np.isfinite(summ.get("Sharpe", np.nan)) else "N/A")
    c10.metric("Max DD", f"{summ.get('max_drawdown', np.nan):.1%}" if np.isfinite(summ.get("max_drawdown", np.nan)) else "N/A")

    st.pyplot(plot_backtest_scatter(bt2, title=f"{asset_key} | scatter (h={horizon_w}w)"))
    if equity is not None and not equity.dropna().empty:
        st.pyplot(plot_equity_curve(equity, title=f"{asset_key} | equity (mode={mode})"))

    st.markdown("#### Backtest table (tail)")
    st.dataframe(bt2.tail(80))

    st.markdown("### TAB5 One-file CSV ë¤ì´ë¡ë (bt + pos + trades-marks)")
    try:
        df_u = bt2.copy()
        df_u["trade_flag_nonoverlap"] = 0
        if trades is not None and not trades.empty:
            df_u.loc[trades.index, "trade_flag_nonoverlap"] = 1
        df_u["equity_nonoverlap"] = np.nan
        if trades is not None and not trades.empty:
            df_u.loc[trades.index, "equity_nonoverlap"] = trades["equity"].values

        st.download_button(
            "Download TAB5 unified CSV",
            data=df_u.to_csv(index=True).encode("utf-8-sig"),
            file_name=f"tab5_UNIFIED_{asset_key.replace(' ', '_')}_h{horizon_w}w_fit{fit_w}w_xx{xx}w_{APP_VERSION}.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.warning(f"Unified CSV ìì± ì¤ ì¤ë¥: {e}")

    st.download_button(
        "Download TAB5 backtest CSV",
        data=bt2.to_csv(index=True).encode("utf-8-sig"),
        file_name=f"tab5_backtest_{asset_key.replace(' ', '_')}_h{horizon_w}w_fit{fit_w}w_xx{xx}w_{APP_VERSION}.csv",
        mime="text/csv",
    )
    if trades is not None and not trades.empty:
        st.download_button(
            "Download TAB5 trades CSV",
            data=trades.to_csv(index=True).encode("utf-8-sig"),
            file_name=f"tab5_trades_{asset_key.replace(' ', '_')}_h{horizon_w}w_fit{fit_w}w_xx{xx}w_{APP_VERSION}.csv",
            mime="text/csv",
        )


# =========================
# Tabs (keep all)
# =========================
tab_main, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "MAIN (True-history + Spaghetti)",
    "TAB1 DXY(ì­ì¶) vs BTC",
    "TAB2 Liquidity vs BTC",
    "TAB3 Combo + 2D Lag-Pair",
    "TAB4 Forward Overlay (LDLI vs BTC)",
    "TAB5 Multi-Asset Forecast/Backtest",
])

with tab_main:
    run_main_tab()

with tab1:
    run_tab_dxy()

with tab2:
    run_tab_liquidity()

with tab3:
    run_tab_combo()

with tab4:
    run_tab_forward_overlay()

with tab5:
    run_tab5_multiasset()