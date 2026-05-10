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

# ============================================================
# v2.19 Alpha Auto-Fetch: Disk Cache Utilities
# ============================================================
from pathlib import Path
import logging
import time

ALPHA_CACHE_DIR = Path(".cache_alpha")
ALPHA_CACHE_DIR.mkdir(exist_ok=True)
LOG_ALPHA = logging.getLogger("v219_alpha")
logging.basicConfig(level=logging.INFO)


def _alpha_disk_cache_path(name: str) -> Path:
    return ALPHA_CACHE_DIR / f"{name}.parquet"


def _alpha_save_disk(name: str, df: pd.DataFrame) -> None:
    """디스크 캐시 저장 (Parquet)."""
    try:
        if df is not None and not df.empty:
            df.to_parquet(_alpha_disk_cache_path(name))
            LOG_ALPHA.info(f"disk cache saved: {name} ({len(df)} rows)")
    except Exception as e:
        LOG_ALPHA.warning(f"disk cache save failed [{name}]: {e}")


def _alpha_load_disk(name: str, max_age_hours: int = 30) -> Optional[pd.DataFrame]:
    """디스크 캐시 로딩 (max_age_hours 초과 시 None 반환)."""
    p = _alpha_disk_cache_path(name)
    if not p.exists():
        return None
    age_h = (time.time() - p.stat().st_mtime) / 3600
    if age_h > max_age_hours:
        LOG_ALPHA.warning(f"disk cache stale: {name} ({age_h:.1f}h > {max_age_hours}h)")
        return None
    try:
        df = pd.read_parquet(p)
        LOG_ALPHA.info(f"disk cache hit: {name} ({len(df)} rows, age={age_h:.1f}h)")
        return df
    except Exception as e:
        LOG_ALPHA.warning(f"disk cache load failed [{name}]: {e}")
        return None


def apply_pit_cutoff(df: pd.DataFrame, cutoff_days: int) -> pd.DataFrame:
    """
    Point-in-Time 규칙 적용: 오늘 기준 cutoff_days일 이전까지의 데이터만 허용.
    look-ahead bias 방지를 위한 핵심 함수.
    """
    if df is None or df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    cutoff = pd.Timestamp.utcnow().tz_localize(None).normalize() - pd.Timedelta(days=int(cutoff_days))
    return df.loc[df.index <= cutoff].copy()


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

    suspicious_tokens = ("ì", "ë", "ê", "í", "ã", "Â", "â", "Ð")
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


# ============================================================
# v2.19.3 FRED Official API (Diagnostic) — additive only
# ============================================================
# 목적: Streamlit Cloud에서 fred.stlouisfed.org/graph 차단/timeout 시
#       api.stlouisfed.org는 작동하는지 진단.
#
# 기존 함수 (_fetch_fred_via_official_api, _fetch_fred_via_graph_csv,
# _fetch_fred_with_retry, fetch_fred_fredgraph) 변경 없음. 이 진단 함수
# (fetch_fred_official)는 별도 이름이며 기존 fetch path를 건드리지 않는다.
#
# Secret 형식 차이 주의:
#   - 기존 PR #9 (_fetch_fred_via_official_api): st.secrets.get("FRED_API_KEY")
#   - 이번 진단 (_get_fred_api_key):              st.secrets["fred"]["api_key"]
# 사용자가 어느 형식으로 등록했는지에 따라 어느 함수가 키를 찾는지 확인 가능.

def _get_fred_api_key():
    """Streamlit Secrets에서 [fred] api_key (nested) 가져오기.

    secrets.toml 형식:
        [fred]
        api_key = "your_32_char_key_here"

    Returns:
        str | None — 키 발견 못하면 None
    """
    try:
        return st.secrets["fred"]["api_key"]
    except Exception:
        return None


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fetch_fred_official(series_id: str, start_date: str = "1990-01-01") -> pd.Series:
    """공식 FRED API (api.stlouisfed.org) 시리즈 fetch — 진단용.

    https://fred.stlouisfed.org/docs/api/fred/series_observations.html

    Args:
        series_id: FRED series ID (e.g., "WALCL")
        start_date: ISO date (default 1990-01-01)

    Returns:
        pd.Series with date index, sorted ascending

    Raises:
        RuntimeError if API key missing or fetch fails after 3 attempts
    """
    api_key = _get_fred_api_key()
    if not api_key:
        raise RuntimeError(
            "FRED API key not in Streamlit Secrets. "
            "Add: [fred] api_key = 'your_key_here'"
        )

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
    }

    last_err = None
    for attempt in range(3):
        r = None
        try:
            t0 = time.time()
            r = requests.get(url, params=params, timeout=30)
            elapsed = time.time() - t0
            r.raise_for_status()
            data = r.json()

            if "observations" not in data:
                raise ValueError(f"No observations in response: {str(data)[:200]}")

            obs = data["observations"]
            df = pd.DataFrame(obs)
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["value"])

            s = df.set_index("date")["value"]
            s.name = series_id

            LOG_ALPHA.info(
                f"[FRED-API] {series_id} OK ({elapsed:.1f}s, {len(s)} obs, "
                f"{s.index.min().date()}~{s.index.max().date()})"
            )
            return s.sort_index()

        except requests.exceptions.Timeout as e:
            last_err = f"Timeout after 30s: {e}"
            LOG_ALPHA.warning(f"[FRED-API] {series_id} attempt {attempt+1} TIMEOUT")
        except requests.exceptions.HTTPError as e:
            status = r.status_code if r is not None else "?"
            last_err = f"HTTP {status}: {e}"
            LOG_ALPHA.warning(f"[FRED-API] {series_id} attempt {attempt+1} HTTP error: {status}")
            if r is not None and r.status_code == 403:
                raise RuntimeError(f"FRED API 403 (invalid key?): {e}") from e
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            LOG_ALPHA.warning(f"[FRED-API] {series_id} attempt {attempt+1} failed: {e}")

        if attempt < 2:
            time.sleep(2 ** attempt)  # 1s, 2s

    raise RuntimeError(f"FRED API fetch failed for {series_id}: {last_err}")


# ============================================================
# v2.19.4 Migration wrapper: fetch_fred_with_fallback
# ============================================================
# v2.19.3 진단 결과 (2026-05-04):
#   fred.stlouisfed.org/graph/fredgraph.csv — 60s timeout on Streamlit Cloud
#   api.stlouisfed.org/fred/series/observations — 0.2s, all series OK
#
# 이 wrapper는 새 표준 fetch entry point. fallback 체인:
#   1. api.stlouisfed.org (official, 빠르고 안정) — [fred] api_key 필요
#   2. fred.stlouisfed.org/graph (기존 CSV path, key 없거나 official 실패 시)
#
# 기존 fetch_fred_fredgraph()는 보존 (롤백 가능성, fallback 경로용).

def fetch_fred_with_fallback(series_id: str, start_date: str = "1990-01-01") -> pd.Series:
    """FRED fetch with fallback chain.

    Priority:
      1. api.stlouisfed.org (official) — when [fred] api_key in Secrets
      2. fred.stlouisfed.org/graph (CSV) — fallback (existing behavior)

    Streamlit Cloud에서는 1순위만 작동 (CSV는 timeout).
    로컬/Codespaces에서는 둘 다 작동 가능.

    이 함수가 신규 코드의 표준 fetch 함수.

    Args:
        series_id: FRED series ID
        start_date: ISO date for observation_start (only used by official API)

    Returns:
        pd.Series with date index
    """
    api_key = _get_fred_api_key()

    if api_key:
        try:
            return fetch_fred_official(series_id, start_date)
        except Exception as e:
            LOG_ALPHA.warning(f"[FRED] Official API failed for {series_id}: {e}")
            LOG_ALPHA.warning(f"[FRED] Falling back to CSV API...")

    return fetch_fred_fredgraph(series_id)


with st.sidebar:
    st.markdown("## Settings")
    LIQ_SOURCE = st.selectbox(
        "Liquidity Source",
        [
            "Fed Net Liquidity (FRED)",
            "G3 Total Assets (USD)",
            "G3 YoY Change (%)",
            "G2M2 Total (USD)",
            "G2M2 YoY Change (%)",
            # v2.19.8: 컴포넌트별 변화율 (BTC 상관 측정 결과 기반 추가)
            "BOJ 13w Change (%)",
            "BOJ YoY Change (%)",
            "ECB 13w Change (%)",
            "Fed YoY Change (%)",
        ],
        index=0,
        help=(
            "Fed/G3/G2M2: 절대값 또는 YoY. "
            "v2.19.8: 컴포넌트별 변화율 추가. 측정 결과 BTC 13w 상관: "
            "BOJ 13w change=+0.573 (best), BOJ YoY=+0.495, ECB 13w=+0.430, Fed YoY=+0.302. "
            "Fed level (-0.024)보다 압도적. "
            "G3 = Fed + ECB + BOJ (USD-converted). G2M2 = US M2 + EU M3 (Lyn Alden lite)."
        ),
    )

    # v2.19.3 FRED API 진단 expander (additive only, doesn't affect normal flow)
    with st.expander("🔬 FRED API 진단 (v2.19.3)", expanded=False):
        st.caption("api.stlouisfed.org 작동 여부 확인 (graph endpoint timeout 시 대안)")

        # API key 상태 (nested [fred] api_key 형식)
        _diag_api_key_present = _get_fred_api_key() is not None
        if _diag_api_key_present:
            st.success("✅ API key 등록됨 (Secrets [fred] api_key)")
        else:
            st.error(
                "❌ API key 없음.\n\nSecrets에 다음 추가 필요:\n"
                "```toml\n[fred]\napi_key = \"your_key_here\"\n```"
            )

        if _diag_api_key_present:
            st.caption("아래 버튼으로 시리즈별 작동 테스트:")

            _diag_col1, _diag_col2 = st.columns(2)
            with _diag_col1:
                if st.button("Test WALCL", key="diag_walcl"):
                    try:
                        with st.spinner("api.stlouisfed.org → WALCL..."):
                            _diag_t0 = time.time()
                            _diag_s = fetch_fred_official("WALCL", start_date="2020-01-01")
                            _diag_elapsed = time.time() - _diag_t0
                        st.success(f"✅ WALCL ({_diag_elapsed:.1f}s, {len(_diag_s)} obs)")
                        st.caption(f"Latest: ${_diag_s.iloc[-1]:,.0f}M ({_diag_s.index[-1].date()})")
                    except Exception as _diag_e:
                        st.error(f"❌ {type(_diag_e).__name__}: {str(_diag_e)[:200]}")

                if st.button("Test ECBASSETSW", key="diag_ecb"):
                    try:
                        with st.spinner("api.stlouisfed.org → ECBASSETSW..."):
                            _diag_t0 = time.time()
                            _diag_s = fetch_fred_official("ECBASSETSW", start_date="2019-08-01")
                            _diag_elapsed = time.time() - _diag_t0
                        st.success(f"✅ ECBASSETSW ({_diag_elapsed:.1f}s, {len(_diag_s)} obs)")
                        st.caption(f"Latest: €{_diag_s.iloc[-1]:,.0f}M ({_diag_s.index[-1].date()})")
                    except Exception as _diag_e:
                        st.error(f"❌ {type(_diag_e).__name__}: {str(_diag_e)[:200]}")

            with _diag_col2:
                if st.button("Test JPNASSETS", key="diag_boj"):
                    try:
                        with st.spinner("api.stlouisfed.org → JPNASSETS..."):
                            _diag_t0 = time.time()
                            _diag_s = fetch_fred_official("JPNASSETS", start_date="1998-04-01")
                            _diag_elapsed = time.time() - _diag_t0
                        st.success(f"✅ JPNASSETS ({_diag_elapsed:.1f}s, {len(_diag_s)} obs)")
                        st.caption(f"Latest: ¥{_diag_s.iloc[-1]:,.0f}×100M ({_diag_s.index[-1].date()})")
                    except Exception as _diag_e:
                        st.error(f"❌ {type(_diag_e).__name__}: {str(_diag_e)[:200]}")

                if st.button("Test DEXUSEU", key="diag_dexuseu"):
                    try:
                        with st.spinner("api.stlouisfed.org → DEXUSEU..."):
                            _diag_t0 = time.time()
                            _diag_s = fetch_fred_official("DEXUSEU", start_date="2020-01-01")
                            _diag_elapsed = time.time() - _diag_t0
                        st.success(f"✅ DEXUSEU ({_diag_elapsed:.1f}s, {len(_diag_s)} obs)")
                        st.caption(f"Latest: {_diag_s.iloc[-1]:.4f} ({_diag_s.index[-1].date()})")
                    except Exception as _diag_e:
                        st.error(f"❌ {type(_diag_e).__name__}: {str(_diag_e)[:200]}")

            # v2.19.5: Global M2 series 가용성 진단
            st.markdown("---")
            st.caption("**Global M2 시리즈 진단:**")

            _diag_m2_series_to_test = [
                ("M2SL", "US M2 (Billions USD, monthly)"),
                ("MABMM301EZM657S", "EU M3 (Millions EUR, monthly, M2 proxy)"),
                ("MYAGM2JPM189S", "JP M2 (100M JPY, monthly, SA)"),
                ("MYAGM2JPM189N", "JP M2 (NSA, fallback)"),
                ("MABMM301GBM189S", "UK M3 (Millions GBP, monthly)"),
                ("MABMM301CNM189N", "CN M3 (Billions CNY, monthly, NSA)"),
                ("DEXCHUS", "CNY/USD exchange rate"),
                ("DEXUSUK", "USD/GBP exchange rate"),
            ]

            if st.button("Test all M2 series", key="diag_m2_all"):
                _diag_m2_results = []
                for _diag_sid, _diag_desc in _diag_m2_series_to_test:
                    try:
                        _diag_t0 = time.time()
                        _diag_s = fetch_fred_official(_diag_sid, start_date="2015-01-01")
                        _diag_elapsed = time.time() - _diag_t0
                        _diag_latest_val = _diag_s.iloc[-1] if len(_diag_s) > 0 else None
                        _diag_latest_date = _diag_s.index[-1] if len(_diag_s) > 0 else None
                        _diag_m2_results.append({
                            "series": _diag_sid,
                            "status": "✅ OK",
                            "obs": len(_diag_s),
                            "elapsed": f"{_diag_elapsed:.1f}s",
                            "first": str(_diag_s.index[0].date()) if len(_diag_s) > 0 else "N/A",
                            "last": str(_diag_latest_date.date()) if _diag_latest_date is not None else "N/A",
                            "latest_val": f"{_diag_latest_val:,.1f}" if _diag_latest_val is not None else "N/A",
                            "desc": _diag_desc,
                        })
                    except Exception as _diag_e:
                        _diag_err_short = str(_diag_e)[:80]
                        _diag_m2_results.append({
                            "series": _diag_sid,
                            "status": "❌ FAIL",
                            "obs": 0,
                            "elapsed": "-",
                            "first": "-",
                            "last": "-",
                            "latest_val": "-",
                            "desc": _diag_desc + f" | {_diag_err_short}",
                        })

                _diag_df_results = pd.DataFrame(_diag_m2_results)
                st.dataframe(_diag_df_results, use_container_width=True)

                _diag_ok_count = sum(1 for _r in _diag_m2_results if _r["status"] == "✅ OK")
                st.info(f"성공: {_diag_ok_count} / {len(_diag_m2_results)}")

    ALPHA_MODE = st.selectbox(
        "Alpha / intercept mode",
        [
            "OLS (learn alpha)",
            "ZERO (alpha=0)",
        ],
        index=0,
    )

    with st.expander(ui_text("Alpha Inputs (자동 fetch + 옵션 업로드)"), expanded=False):
        st.caption("v2.19: 4개 신호 (Funding/MVRV/Reserve/ETF) 자동 수집")

        # Funding (기존)
        USE_BINANCE_FUNDING = st.checkbox(
            "Funding 자동 수집 (Binance API)",
            value=True,
            help="BTCUSDT funding history를 Binance Futures에서 자동 수집",
        )
        FUNDING_SYMBOL = st.text_input("Funding symbol", value="BTCUSDT")

        # MVRV (자동 + 선택적 업로드)
        USE_AUTO_MVRV = st.checkbox(
            "MVRV Z-Score 자동 수집 (CoinMetrics)",
            value=True,
            help="CoinMetrics Community API → BGeometrics → 디스크 캐시 fallback",
        )
        MVRV_FILE = st.file_uploader(
            "MVRV CSV 수동 업로드 (자동 수집 override)",
            type=["csv"],
            help="업로드 시 자동 수집보다 우선 적용",
        )

        # Reserve (자동만)
        USE_AUTO_RESERVE = st.checkbox(
            "Exchange Reserve 자동 수집 (BGeometrics + proxy)",
            value=True,
            help="BGeometrics → CoinMetrics SplyAct1yr proxy → 디스크 캐시. PIT T-7 적용.",
        )

        # ETF (자동만)
        USE_AUTO_ETF = st.checkbox(
            "ETF Net Flow 자동 수집 (Farside Investors)",
            value=True,
            help="Farside HTML 스크래핑 → SoSoValue Demo API → 디스크 캐시. PIT T-2 적용.",
        )

MVRV_FILE_BYTES = MVRV_FILE.getvalue() if MVRV_FILE is not None else None
MVRV_FILE_NAME = MVRV_FILE.name if MVRV_FILE is not None else None

FORECAST_MODEL = st.selectbox(
        "Forecast model (MAIN/TAB4)",
        [
            "Drivers-only Δ (legacy)",
            "ECM on LDLI level (gap + ΔLDLI)",
        ],
        index=1,
        help="Legacy uses only Δ(liquidity) and Δ(DXY) drivers. ECM uses LDLI level alignment + error-correction (gap) to avoid flat/always-up spaghetti.",
    )

with st.expander(ui_text("고정 파라미터(입력값) 보기"), expanded=False):
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
def fetch_fred_fredgraph(series_id: str) -> pd.Series:
    """FRED CSV fetch with timeout/retry protection.

    v2.19.1 Patch B-fix (2026-05-04):
    Original implementation used pd.read_csv(url) with no timeout guard,
    causing potential infinite hang on Streamlit Cloud → FRED slow connections.
    Now delegates to _fetch_fred_with_retry which has timeout=(10, 60) and
    3-attempt exponential backoff. Cache is held by _fetch_fred_with_retry
    (single source of truth, no duplicate cache).

    All existing call sites (Fed Net Liquidity, G2 M2 legacy, etc.) gain
    timeout protection automatically.
    """
    return _fetch_fred_with_retry(series_id)


# ============================================================
# v2.19.1 Patch B: G3 Liquidity (US + EU + JP)
# ============================================================
# FRED 시리즈 (모두 무료, 인증 불필요):
#   WALCL: Fed Total Assets (millions USD, weekly Wed)
#   ECBASSETSW: ECB Total Assets (millions EUR, weekly Fri)
#   JPNASSETS: BOJ Total Assets (100M JPY, monthly)
#   DEXUSEU: USD per 1 EUR (daily)
#   DEXJPUS: JPY per 1 USD (daily)

FRED_ECBASSETSW = "ECBASSETSW"
FRED_JPNASSETS = "JPNASSETS"
FRED_DEXUSEU = "DEXUSEU"
FRED_DEXJPUS = "DEXJPUS"

G3_START_DATE = "2019-08-01"          # ECBASSETSW 데이터 시작 시점
G3_INDEX_BASE_DATE = "2020-01-03"     # 정규화 인덱스 기준일 (코로나 직전)


# FRED endpoint constants — used by both _fetch_fred_via_official_api and
# _fetch_fred_via_graph_csv. graph endpoint is the legacy public CSV used by
# the FRED website's chart download feature; api endpoint is the dedicated
# programmatic API and is significantly faster + more reliable.
FRED_API_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_GRAPH_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
# Date filter for graph CSV path — reduces payload by limiting history.
# WALCL since 1990s ~1300 weekly rows → since 2019 ~340 rows (~75% reduction).
# G3 needs from 2019-08, but we keep some buffer for YoY (52w) computation.
FRED_GRAPH_COSD = "2018-01-01"


def _fetch_fred_via_official_api(
    series_id: str,
    api_key: str,
    observation_start: str = "2018-01-01",
    timeout: Tuple[int, int] = (10, 30),
) -> pd.Series:
    """Fetch FRED series via official API (api.stlouisfed.org).

    Requires FRED_API_KEY (free, https://fredaccount.stlouisfed.org/apikey).
    Different infrastructure than graph CSV — typically <1s response.
    """
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": observation_start,
    }
    r = requests.get(
        FRED_API_OBSERVATIONS_URL,
        params=params,
        timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    r.raise_for_status()
    payload = r.json()
    obs = payload.get("observations", [])
    if not obs:
        raise RuntimeError(f"FRED API returned empty observations for {series_id}")
    df = pd.DataFrame(obs)
    df["date"] = pd.to_datetime(df["date"])
    s = pd.to_numeric(df["value"], errors="coerce")
    s.index = df["date"]
    s.name = series_id
    return s.dropna()


def _fetch_fred_via_graph_csv(
    series_id: str,
    cosd: str = FRED_GRAPH_COSD,
    timeout: Tuple[int, int] = (10, 60),
) -> pd.Series:
    """Fetch FRED series via legacy graph CSV endpoint with date filter.

    cosd parameter limits history to reduce payload (75% smaller for WALCL).
    Used as fallback when FRED_API_KEY is not configured.
    """
    url = f"{FRED_GRAPH_CSV_URL}?id={series_id}&cosd={cosd}"
    r = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    date_col = df.columns[0]
    val_col = df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col])
    s = pd.to_numeric(df[val_col], errors="coerce")
    s.index = df[date_col]
    s.name = series_id
    return s.dropna()


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def _fetch_fred_with_retry(series_id: str, max_attempts: int = 3) -> pd.Series:
    """FRED fetch with two-tier fallback + retry.

    v2.19.1 Patch B-fix-2 (2026-05-04): Streamlit Cloud reported repeated
    Read timeouts on the legacy graph CSV endpoint, even with 60s timeout.
    Likely cause: graph endpoint (designed for chart downloads) throttles
    or is structurally slow from Cloud's egress IP range.

    Strategy:
      1. If FRED_API_KEY in st.secrets → use api.stlouisfed.org (fast,
         dedicated infrastructure, <1s response typical).
      2. Otherwise → use graph CSV with cosd=2018-01-01 date filter
         (75% smaller payload, may evade throttling).

    Each tier is retried up to max_attempts with exponential backoff (1s/2s/4s).

    Setup for FRED_API_KEY (recommended):
      1. Get free key at https://fredaccount.stlouisfed.org/apikey (1 min).
      2. Streamlit Cloud → app settings → secrets:
           FRED_API_KEY = "your_32_char_key_here"
      3. Redeploy. No code changes needed; this function auto-detects.
    """
    # Detect API key (graceful — no key means fallback to graph CSV)
    api_key: Optional[str] = None
    try:
        if hasattr(st, "secrets"):
            api_key = st.secrets.get("FRED_API_KEY", None)
    except Exception:
        # st.secrets may raise if no secrets.toml configured at all
        api_key = None

    last_err: Optional[Exception] = None

    # ── Tier 1: Official API (preferred) ───────────────────
    if api_key:
        for attempt in range(max_attempts):
            try:
                LOG_ALPHA.info(f"[FRED] {series_id} via official API (attempt {attempt+1})")
                return _fetch_fred_via_official_api(series_id, api_key)
            except Exception as e:
                last_err = e
                LOG_ALPHA.warning(f"[FRED] {series_id} API attempt {attempt+1} failed: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(2 ** attempt)
                continue
        # API failed; fall through to graph CSV (don't give up yet)
        LOG_ALPHA.warning(f"[FRED] {series_id} all API attempts failed, trying graph CSV")

    # ── Tier 2: Graph CSV (fallback or default) ────────────
    for attempt in range(max_attempts):
        try:
            LOG_ALPHA.info(f"[FRED] {series_id} via graph CSV (attempt {attempt+1})")
            return _fetch_fred_via_graph_csv(series_id)
        except Exception as e:
            last_err = e
            LOG_ALPHA.warning(f"[FRED] {series_id} graph attempt {attempt+1} failed: {e}")
            if attempt < max_attempts - 1:
                time.sleep(2 ** attempt)
            continue

    raise RuntimeError(
        f"FRED fetch failed for {series_id} after all attempts "
        f"(API key {'configured' if api_key else 'NOT configured'}): {last_err}"
    )



@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fetch_g3_total_assets(week_rule: str = WEEK_RULE) -> pd.DataFrame:
    """
    G3 (US + EU + JP) Central Bank Total Assets fetch.

    Returns:
        pd.DataFrame with columns:
            - g3_total_usd_m: Total Assets (USD millions)
            - g3_index: Normalized index (2020-01-03 = 100)
            - g3_yoy_pct: YoY change (%)
            - g3_4w_change_pct: 4-week change (%)
            - g3_13w_change_pct: 13-week change (%)
            - fed_usd_m: Fed only (for comparison)
            - ecb_usd_m: ECB in USD
            - boj_usd_m: BOJ in USD

    Notes:
        - All data weekly W-FRI aligned
        - JPNASSETS (monthly) is forward-filled to weekly (PIT-safe)
        - Starts at G3_START_DATE (2019-08)
        - Cache TTL 6 hours
    """
    LOG_ALPHA.info("[G3] fetch_g3_total_assets 시작")

    # ── Step 1: FRED 데이터 fetch ────────────────────────
    # v2.19.4: Migrated to fetch_fred_with_fallback (api.stlouisfed.org first).
    try:
        walcl = fetch_fred_with_fallback(FRED_WALCL, start_date="2002-01-01")
        LOG_ALPHA.info(f"[G3] WALCL fetched: {len(walcl)} obs")
    except Exception as e:
        raise RuntimeError(f"[G3] WALCL fetch failed: {e}") from e

    try:
        ecb = fetch_fred_with_fallback(FRED_ECBASSETSW, start_date="2019-08-01")
        LOG_ALPHA.info(f"[G3] ECBASSETSW fetched: {len(ecb)} obs")
    except Exception as e:
        raise RuntimeError(f"[G3] ECBASSETSW fetch failed: {e}") from e

    try:
        boj = fetch_fred_with_fallback(FRED_JPNASSETS, start_date="1998-04-01")
        LOG_ALPHA.info(f"[G3] JPNASSETS fetched: {len(boj)} obs (monthly)")
    except Exception as e:
        raise RuntimeError(f"[G3] JPNASSETS fetch failed: {e}") from e

    try:
        eur_usd = fetch_fred_with_fallback(FRED_DEXUSEU, start_date="2000-01-01")
        LOG_ALPHA.info(f"[G3] DEXUSEU fetched: {len(eur_usd)} obs")
    except Exception as e:
        raise RuntimeError(f"[G3] DEXUSEU fetch failed: {e}") from e

    try:
        jpy_usd = fetch_fred_with_fallback(FRED_DEXJPUS, start_date="2000-01-01")
        LOG_ALPHA.info(f"[G3] DEXJPUS fetched: {len(jpy_usd)} obs")
    except Exception as e:
        raise RuntimeError(f"[G3] DEXJPUS fetch failed: {e}") from e

    # ── Step 2: 모든 시리즈를 W-FRI 주간으로 정렬 ────────
    # WALCL (Wed): 그 주 W-FRI에 forward fill
    # ECBASSETSW (Fri): 자연스럽게 W-FRI
    # JPNASSETS (월말): 다음 발표까지 forward fill (PIT-safe)
    # 환율: W-FRI 값
    walcl_w = walcl.resample(week_rule).last().ffill()
    ecb_w = ecb.resample(week_rule).last().ffill()
    boj_w = boj.resample(week_rule).last().ffill()  # 월간 → 주간 forward fill
    eur_usd_w = eur_usd.resample(week_rule).last().ffill()
    jpy_usd_w = jpy_usd.resample(week_rule).last().ffill()

    # ── Step 3: USD 환산 ──────────────────────────────
    # WALCL: 이미 USD millions
    fed_usd_m = walcl_w.copy()

    # ECBASSETSW: EUR millions × (USD/EUR) = USD millions
    ecb_usd_m = (ecb_w * eur_usd_w).dropna()

    # JPNASSETS: 100 million JPY 단위 → millions JPY = ×100 → ÷ DEXJPUS = millions USD
    # DEXJPUS: JPY per 1 USD, 따라서 USD = JPY / DEXJPUS
    boj_usd_m = (boj_w * 100 / jpy_usd_w).dropna()

    # ── Step 4: 합성 ─────────────────────────────────
    df = pd.DataFrame({
        "fed_usd_m": fed_usd_m,
        "ecb_usd_m": ecb_usd_m,
        "boj_usd_m": boj_usd_m,
    }).dropna()

    df["g3_total_usd_m"] = df["fed_usd_m"] + df["ecb_usd_m"] + df["boj_usd_m"]

    # ── Step 5: 시작점 cutoff ─────────────────────────
    df = df[df.index >= pd.Timestamp(G3_START_DATE)]

    # ── Step 6: 정규화 인덱스 ──────────────────────────
    base_dt = pd.Timestamp(G3_INDEX_BASE_DATE)
    if base_dt in df.index:
        base_value = df.loc[base_dt, "g3_total_usd_m"]
    else:
        # 가장 가까운 (이후) 인덱스 fallback
        future = df.index[df.index >= base_dt]
        if len(future) > 0:
            nearest = future[0]
        else:
            nearest = df.index[0]
        base_value = df.loc[nearest, "g3_total_usd_m"]
        LOG_ALPHA.info(f"[G3] index base date {base_dt.date()} not exact, using {nearest.date()}")

    df["g3_index"] = df["g3_total_usd_m"] / base_value * 100

    # ── Step 7: 변화율 계산 ────────────────────────────
    df["g3_yoy_pct"] = df["g3_total_usd_m"].pct_change(52) * 100   # 52주 = 1년
    df["g3_4w_change_pct"] = df["g3_total_usd_m"].pct_change(4) * 100   # 4주 (월간 근사)
    df["g3_13w_change_pct"] = df["g3_total_usd_m"].pct_change(13) * 100  # 13주 (분기)

    # v2.19.8: 컴포넌트별 변화율 (BTC와 component-level 상관 측정 결과 BOJ가 가장 강함)
    # 측정 결과 (416주, 2018-05~2026-05): BTC 13w return과의 상관
    #   BOJ 13w change: +0.573 ⭐⭐⭐
    #   BOJ YoY:        +0.495
    #   ECB 13w change: +0.430 (best @ 26w horizon)
    #   Fed YoY:        +0.302
    df["fed_yoy_pct"] = df["fed_usd_m"].pct_change(52) * 100
    df["ecb_yoy_pct"] = df["ecb_usd_m"].pct_change(52) * 100
    df["boj_yoy_pct"] = df["boj_usd_m"].pct_change(52) * 100
    df["fed_13w_change_pct"] = df["fed_usd_m"].pct_change(13) * 100
    df["ecb_13w_change_pct"] = df["ecb_usd_m"].pct_change(13) * 100
    df["boj_13w_change_pct"] = df["boj_usd_m"].pct_change(13) * 100
    df["boj_4w_change_pct"] = df["boj_usd_m"].pct_change(4) * 100

    LOG_ALPHA.info(
        f"[G3] G3 panel built: {len(df)} weekly obs, "
        f"{df.index.min().date()} ~ {df.index.max().date()}"
    )

    return df


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def load_g3_total_assets_daily(metric: str = "level"):
    """
    G3 panel을 daily 시리즈 + snapshot 으로 변환 (load_*_daily 호환).

    Args:
        metric: "level" (g3_total_usd_m) 또는 "yoy" (g3_yoy_pct)

    Returns:
        (daily_series, snapshot_df) — load_g2_m2_usd_daily()와 동일 시그니처
    """
    g3_panel = fetch_g3_total_assets()
    if g3_panel is None or g3_panel.empty:
        raise RuntimeError("[G3] empty panel — fetch failed upstream")

    # v2.19.8: 컴포넌트별 변화율 옵션 추가
    if metric == "level":
        weekly = g3_panel["g3_total_usd_m"].rename("G3_Total_Assets_USD_m")
    elif metric == "yoy":
        weekly = g3_panel["g3_yoy_pct"].rename("G3_YoY_pct")
    elif metric == "fed_yoy":
        weekly = g3_panel["fed_yoy_pct"].rename("Fed_YoY_pct")
    elif metric == "ecb_yoy":
        weekly = g3_panel["ecb_yoy_pct"].rename("ECB_YoY_pct")
    elif metric == "boj_yoy":
        weekly = g3_panel["boj_yoy_pct"].rename("BOJ_YoY_pct")
    elif metric == "ecb_13w":
        weekly = g3_panel["ecb_13w_change_pct"].rename("ECB_13w_change_pct")
    elif metric == "boj_13w":
        weekly = g3_panel["boj_13w_change_pct"].rename("BOJ_13w_change_pct")
    else:
        raise ValueError(f"Unknown G3 metric: {metric}")

    # weekly → daily (forward-fill)
    daily_idx = pd.date_range(start=START_DATE, end=END_DATE, freq="D")
    daily = weekly.reindex(daily_idx).ffill()

    # snapshot: 전체 G3 패널을 daily로
    snap = g3_panel.reindex(daily_idx).ffill()

    return daily.dropna(), snap


# ============================================================
# v2.19.6 Global M2 lite (G2M2: US + EU)
# ============================================================
# v2.19.5 진단 결과: JP/UK/CN M2 시리즈는 FRED에서 stale (2017~2023 멈춤)
# 또는 불가. Lyn Alden 5국 Global M2의 lite 버전: US + EU만.
#
# Data sources (all FRED):
#   M2SL:            US M2, billions USD, monthly
#   MABMM301EZM657S: EU M3 (M2 proxy), millions EUR, monthly
#   DEXUSEU:         USD per 1 EUR, daily

FRED_M2SL = "M2SL"
FRED_EU_M3 = "MABMM301EZM657S"

G2M2_START_DATE = "2015-01-01"
G2M2_INDEX_BASE_DATE = "2020-01-01"


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fetch_g2m2_total(week_rule: str = WEEK_RULE) -> pd.DataFrame:
    """
    G2M2 (Global M2 lite, US + EU) 합성.

    Returns:
        pd.DataFrame with columns:
            - g2m2_total_usd_m: Total M2 (USD millions, weekly)
            - g2m2_index: Normalized index (2020-01 = 100)
            - g2m2_yoy_pct: YoY change (%)
            - g2m2_4w_change_pct: 4-week change
            - g2m2_13w_change_pct: 13-week change
            - m2_us_usd_m: US M2 only
            - m2_eu_usd_m: EU M3 in USD

    Notes:
        - US M2 (M2SL) is monthly, billions USD
        - EU M3 (MABMM301EZM657S) is monthly, millions EUR
        - DEXUSEU is daily, USD per 1 EUR
        - All resampled to W-FRI, forward-fill (PIT-safe)
        - Starts 2015-01-01
    """
    LOG_ALPHA.info("[G2M2] fetch_g2m2_total 시작")

    # Step 1: FRED fetch (all via fetch_fred_with_fallback → api.stlouisfed.org)
    try:
        m2_us_b = fetch_fred_with_fallback(FRED_M2SL, start_date=G2M2_START_DATE)
        LOG_ALPHA.info(f"[G2M2] M2SL fetched: {len(m2_us_b)} obs")
    except Exception as e:
        raise RuntimeError(f"[G2M2] M2SL fetch failed: {e}") from e

    try:
        m2_eu_em = fetch_fred_with_fallback(FRED_EU_M3, start_date=G2M2_START_DATE)
        LOG_ALPHA.info(f"[G2M2] EU M3 fetched: {len(m2_eu_em)} obs")
    except Exception as e:
        raise RuntimeError(f"[G2M2] EU M3 fetch failed: {e}") from e

    try:
        eur_usd = fetch_fred_with_fallback(FRED_DEXUSEU, start_date=G2M2_START_DATE)
        LOG_ALPHA.info(f"[G2M2] DEXUSEU fetched: {len(eur_usd)} obs")
    except Exception as e:
        raise RuntimeError(f"[G2M2] DEXUSEU fetch failed: {e}") from e

    # Step 2: 단위 통일 (모두 USD millions)
    # M2SL: billions USD → millions USD (×1000)
    m2_us_usd_m = m2_us_b * 1000.0
    m2_us_usd_m.name = "m2_us_usd_m"

    # 환율 정렬 위해 W-FRI 주간으로 먼저 변환
    m2_us_w = m2_us_usd_m.resample(week_rule).last().ffill()
    m2_eu_w = m2_eu_em.resample(week_rule).last().ffill()
    eur_usd_w = eur_usd.resample(week_rule).last().ffill()

    # EU M3 (EUR M) × DEXUSEU (USD/EUR) = USD M
    m2_eu_usd_m_w = (m2_eu_w * eur_usd_w).dropna()

    # Step 3: 합성
    df = pd.DataFrame({
        "m2_us_usd_m": m2_us_w,
        "m2_eu_usd_m": m2_eu_usd_m_w,
    }).dropna()

    df["g2m2_total_usd_m"] = df["m2_us_usd_m"] + df["m2_eu_usd_m"]

    # Step 4: 시작일 cutoff
    df = df[df.index >= pd.Timestamp(G2M2_START_DATE)]

    # Step 5: 정규화 인덱스 (2020-01-01 = 100)
    base_dt = pd.Timestamp(G2M2_INDEX_BASE_DATE)
    if base_dt in df.index:
        base_value = df.loc[base_dt, "g2m2_total_usd_m"]
    else:
        idx_after = df.index[df.index >= base_dt]
        if len(idx_after) > 0:
            base_value = df.loc[idx_after[0], "g2m2_total_usd_m"]
            LOG_ALPHA.info(f"[G2M2] index base date adjusted to {idx_after[0].date()}")
        else:
            base_value = df["g2m2_total_usd_m"].iloc[0]
            LOG_ALPHA.info(f"[G2M2] using first available date as base: {df.index[0].date()}")

    df["g2m2_index"] = df["g2m2_total_usd_m"] / base_value * 100

    # Step 6: 변화율 (weekly data, so 52w = YoY)
    df["g2m2_yoy_pct"] = df["g2m2_total_usd_m"].pct_change(52) * 100   # 52주 = 1년
    df["g2m2_4w_change_pct"] = df["g2m2_total_usd_m"].pct_change(4) * 100
    df["g2m2_13w_change_pct"] = df["g2m2_total_usd_m"].pct_change(13) * 100

    LOG_ALPHA.info(
        f"[G2M2] panel built: {len(df)} weekly obs, "
        f"{df.index.min().date()} ~ {df.index.max().date()}"
    )

    return df


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


# ============================================================
# v2.19 Alpha Auto-Fetch: MVRV Z-Score
# ============================================================
@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fetch_mvrv_zscore_auto(week_rule: str = WEEK_RULE) -> pd.Series:
    """
    MVRV Z-Score 자동 수집.

    Fallback 체인:
      1순위: CoinMetrics Community API (CapMrktCurUSD + CapRealUSD → 자체 Z-score 계산)
      2순위: BGeometrics 무료 API
      3순위: 디스크 캐시 (최대 7일된 데이터)

    PIT 규칙: T-7일 컷오프

    Returns:
        pd.Series (name='mvrv_z'), 주간 집계 (W-FRI)
    """

    # ── 1순위: CoinMetrics Community API ─────────────────
    try:
        url = (
            "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
            "?assets=btc&metrics=CapMrktCurUSD,CapRealUSD"
            "&frequency=1d&page_size=10000&pretty=false"
        )
        rows = []
        next_url = url
        for _ in range(20):  # 최대 20페이지
            r = requests.get(next_url, timeout=20)
            r.raise_for_status()
            j = r.json()
            rows.extend(j.get("data", []))
            next_url = j.get("next_page_url")
            if not next_url:
                break

        if not rows:
            raise RuntimeError("CoinMetrics returned empty data")

        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None).dt.normalize()
        df = df.set_index("time").sort_index()
        df["CapMrktCurUSD"] = pd.to_numeric(df["CapMrktCurUSD"], errors="coerce")
        df["CapRealUSD"] = pd.to_numeric(df["CapRealUSD"], errors="coerce")
        df = df.dropna()

        # Glassnode 공식 정의: Z = (MC - RC) / std(MC)
        # std는 inception 이후 expanding std (최소 30일)
        mc = df["CapMrktCurUSD"]
        rc = df["CapRealUSD"]
        std_mc = mc.expanding(min_periods=30).std()
        df["mvrv_z"] = (mc - rc) / std_mc

        out = df[["mvrv_z"]].dropna()
        out = apply_pit_cutoff(out, cutoff_days=7)

        # 주간 집계
        weekly = out.resample(week_rule).last().dropna()
        weekly = weekly[~weekly.index.duplicated(keep="last")]
        s = weekly["mvrv_z"]
        s.name = "mvrv_z"

        _alpha_save_disk("mvrv_zscore", weekly)
        LOG_ALPHA.info(f"[MVRV] CoinMetrics OK ({len(s)} weekly samples)")
        return s

    except Exception as e:
        LOG_ALPHA.warning(f"[MVRV] CoinMetrics primary failed: {e}")

    # ── 2순위: BGeometrics 무료 API ──────────────────────
    # NOTE: BGeometrics 정확한 endpoint는 무료 가입 후
    # api.bgeometrics.com/scalar.html 에서 확인 필요
    # 응답 스키마는 [{"d": "YYYY-MM-DD", "v": float}] 또는 유사 형태로 가정
    try:
        candidate_urls = [
            "https://api.bgeometrics.com/v1/mvrv-zscore",
            "https://api.bgeometrics.com/v1/mvrv-z-score",
            "https://api.bgeometrics.com/v1/mvrv_z_score",
        ]
        df = None
        for url in candidate_urls:
            try:
                r = requests.get(url, timeout=15)
                if r.status_code == 200:
                    payload = r.json()
                    records = payload if isinstance(payload, list) else payload.get("data", [])
                    if records:
                        df = pd.DataFrame(records)
                        break
            except Exception:
                continue

        if df is None or df.empty:
            raise RuntimeError("BGeometrics endpoints all failed")

        # 컬럼 자동 감지
        date_col = next((c for c in ["d", "date", "t", "time"] if c in df.columns), df.columns[0])
        val_col = next((c for c in ["v", "value", "mvrv_z", "mvrv_zscore"] if c in df.columns), df.columns[-1])

        df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
        df = df.rename(columns={val_col: "mvrv_z"}).set_index(date_col).sort_index()
        df = df[["mvrv_z"]].astype(float).dropna()

        out = apply_pit_cutoff(df, cutoff_days=7)
        weekly = out.resample(week_rule).last().dropna()
        s = weekly["mvrv_z"]
        s.name = "mvrv_z"

        _alpha_save_disk("mvrv_zscore", weekly)
        LOG_ALPHA.info(f"[MVRV] BGeometrics fallback OK ({len(s)} weekly samples)")
        return s

    except Exception as e:
        LOG_ALPHA.warning(f"[MVRV] BGeometrics fallback failed: {e}")

    # ── 3순위: 디스크 캐시 ──────────────────────────────
    cached = _alpha_load_disk("mvrv_zscore", max_age_hours=24 * 7)
    if cached is not None and not cached.empty:
        s = cached["mvrv_z"]
        s.name = "mvrv_z"
        LOG_ALPHA.warning(f"[MVRV] all live sources failed, using disk cache ({len(s)} samples)")
        return s

    LOG_ALPHA.error("[MVRV] all sources failed including disk cache")
    return pd.Series(dtype=float, name="mvrv_z")


# ============================================================
# v2.19 Alpha Auto-Fetch: Exchange Reserve
# ============================================================
@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fetch_exchange_reserve_auto(week_rule: str = WEEK_RULE) -> pd.DataFrame:
    """
    거래소 BTC 보유량 자동 수집.

    v2.19 Patch v3: Reorder fallback chain
      1순위: CoinMetrics SplyAct1yr proxy (작동 확인됨)
      2순위: BGeometrics 무료 (작동 미확인)
      3순위: 디스크 캐시

    PIT 규칙: T-7일 컷오프

    중요: 절대량보다 7일 추세 방향이 alpha 신호로 더 의미있음.
    """

    # ── 1순위: CoinMetrics SplyAct1yr proxy ────────────────────
    try:
        LOG_ALPHA.info("[Reserve] CoinMetrics SplyAct1yr fetch start")
        url = (
            "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
            "?assets=btc&metrics=SplyAct1yr&frequency=1d&page_size=10000"
        )
        rows = []
        next_url = url
        for _ in range(20):
            r = requests.get(next_url, timeout=20)
            r.raise_for_status()
            j = r.json()
            rows.extend(j.get("data", []))
            next_url = j.get("next_page_url")
            if not next_url:
                break

        if not rows:
            raise RuntimeError("CoinMetrics SplyAct1yr empty")

        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None).dt.normalize()
        df = df.set_index("time").sort_index()
        df["exchange_reserve_btc"] = pd.to_numeric(df["SplyAct1yr"], errors="coerce")
        df = df[["exchange_reserve_btc"]].dropna()

        out = apply_pit_cutoff(df, cutoff_days=7)
        weekly = out.resample(week_rule).last().dropna()
        weekly["reserve_pct_change_4w"] = weekly["exchange_reserve_btc"].pct_change(4)
        weekly["reserve_z"] = rolling_zscore(weekly["exchange_reserve_btc"], window=104, min_periods=26)

        _alpha_save_disk("exchange_reserve", weekly)
        LOG_ALPHA.info(f"[Reserve] CoinMetrics SplyAct1yr OK ({len(weekly)} weekly samples)")
        return weekly

    except Exception as e:
        LOG_ALPHA.warning(f"[Reserve] CoinMetrics primary failed: {e}")

    # ── 2순위: BGeometrics ──────────────────────────────────────
    try:
        LOG_ALPHA.info("[Reserve] BGeometrics fallback start")
        candidate_urls = [
            "https://api.bgeometrics.com/v1/exchange-reserves",
            "https://api.bgeometrics.com/v1/exchange-reserve",
            "https://api.bgeometrics.com/v1/exchange_reserves",
        ]
        df = None
        for url in candidate_urls:
            try:
                r = requests.get(url, timeout=15)
                if r.status_code == 200:
                    payload = r.json()
                    records = payload if isinstance(payload, list) else payload.get("data", [])
                    if records:
                        df = pd.DataFrame(records)
                        break
            except Exception:
                continue

        if df is None or df.empty:
            raise RuntimeError("BGeometrics endpoints all failed")

        date_col = next((c for c in ["d", "date", "t", "time"] if c in df.columns), df.columns[0])
        val_col = next((c for c in ["v", "value", "reserve", "exchange_reserve"] if c in df.columns), df.columns[-1])

        df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
        df = df.rename(columns={val_col: "exchange_reserve_btc"}).set_index(date_col).sort_index()
        df = df[["exchange_reserve_btc"]].astype(float).dropna()

        out = apply_pit_cutoff(df, cutoff_days=7)
        weekly = out.resample(week_rule).last().dropna()
        weekly["reserve_pct_change_4w"] = weekly["exchange_reserve_btc"].pct_change(4)
        weekly["reserve_z"] = rolling_zscore(weekly["exchange_reserve_btc"], window=104, min_periods=26)

        _alpha_save_disk("exchange_reserve", weekly)
        LOG_ALPHA.info(f"[Reserve] BGeometrics fallback OK ({len(weekly)} weekly samples)")
        return weekly

    except Exception as e:
        LOG_ALPHA.warning(f"[Reserve] BGeometrics failed: {e}")

    # ── 3순위: 디스크 캐시 ───────────────────────────────────
    cached = _alpha_load_disk("exchange_reserve", max_age_hours=24 * 14)
    if cached is not None and not cached.empty:
        LOG_ALPHA.warning("[Reserve] all live sources failed, using disk cache")
        return cached

    LOG_ALPHA.error("[Reserve] all sources failed including disk cache")
    return pd.DataFrame(columns=["exchange_reserve_btc", "reserve_pct_change_4w", "reserve_z"])


# ============================================================
# v2.19 Alpha Auto-Fetch: ETF Net Flow
# ============================================================
@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fetch_etf_netflow_auto(week_rule: str = WEEK_RULE) -> pd.DataFrame:
    """
    미국 스팟 BTC ETF 일일 net flow 자동 수집.

    v2.19 Patch v3: Add Yahoo Finance ETF proxy as new primary source (US-IP-friendly)
      1순위: Yahoo Finance ETF dollar-volume proxy (무료, no auth, US OK)
      2순위: Farside Investors HTML 스크래핑
      3순위: SoSoValue Demo API
      4순위: 디스크 캐시

    PIT 규칙: T-2일 컷오프

    Returns:
        pd.DataFrame with columns: ['etf_netflow_usd_m', 'etf_4w_cumulative', 'etf_z', 'etf_available']
    """
    LOG_ALPHA.info("[ETF] fetch_etf_netflow_auto 시작")

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; BTCMacroPredictionEngine/2.19)"
    }

    # ── 1순위: Yahoo Finance ETF dollar-volume proxy ──────────
    # Individual spot BTC ETF tickers via yfinance (US-IP-friendly).
    # Net flow proxy = sum of daily dollar-volume changes across major spot ETFs.
    try:
        LOG_ALPHA.info("[ETF] Yahoo Finance ETF proxy fetch start")
        import yfinance as yf

        etf_tickers = ["IBIT", "FBTC", "BITB", "ARKB", "BTCO", "EZBC", "BRRR", "HODL", "BTCW", "GBTC"]
        all_etf_data = []

        for ticker in etf_tickers:
            try:
                t = yf.Ticker(ticker)
                hist = t.history(period="2y", interval="1d")
                if hist.empty:
                    continue
                # net flow proxy = (close - prev_close) * volume / 1e6 (millions USD approximation)
                # 더 정확한 net flow는 AUM 변화이지만 무료 데이터로는 제한적
                # 대신: dollar volume change as proxy
                hist["dollar_volume"] = hist["Close"] * hist["Volume"]
                hist["dv_change"] = hist["dollar_volume"].diff() / 1e6  # millions
                hist["ticker"] = ticker
                all_etf_data.append(hist[["dv_change", "ticker"]].dropna())
            except Exception as ee:
                LOG_ALPHA.warning(f"[ETF] {ticker} failed: {ee}")
                continue

        if not all_etf_data:
            raise RuntimeError("Yahoo Finance ETF all failed")

        combined = pd.concat(all_etf_data)
        # Aggregate per date
        daily = combined.groupby(combined.index)["dv_change"].sum()
        daily.index = pd.to_datetime(daily.index).tz_localize(None).normalize()
        daily.name = "etf_netflow_usd_m"
        df = daily.to_frame()

        out = apply_pit_cutoff(df, cutoff_days=2)
        weekly = out.resample(week_rule).sum().dropna(how="all")
        weekly["etf_4w_cumulative"] = weekly["etf_netflow_usd_m"].rolling(4, min_periods=2).sum()
        weekly["etf_z"] = rolling_zscore(weekly["etf_4w_cumulative"], window=52, min_periods=13)
        weekly["etf_available"] = 1

        _alpha_save_disk("etf_netflow", weekly)
        LOG_ALPHA.info(f"[ETF] Yahoo Finance ETF proxy OK ({len(weekly)} weekly samples)")
        LOG_ALPHA.warning(
            "[ETF] Note: Yahoo proxy is dollar-volume-based, "
            "not actual fund flow. Use as directional signal only."
        )
        return weekly

    except Exception as e:
        LOG_ALPHA.warning(f"[ETF] Yahoo proxy failed: {e}")

    # ── 2순위: Farside Investors ──────────────────────────────
    try:
        LOG_ALPHA.info("[ETF] Farside scrape start")
        r = requests.get("https://farside.co.uk/btc/", headers=headers, timeout=20)
        LOG_ALPHA.info(f"[ETF] Farside HTTP status: {r.status_code}, content size: {len(r.text)}")
        r.raise_for_status()

        tables = pd.read_html(r.text)
        if not tables:
            raise RuntimeError("Farside no tables found")

        df_raw = max(tables, key=lambda t: t.shape[0])
        df = df_raw.copy()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[-1] if isinstance(c, tuple) else c for c in df.columns]
        df.columns = [str(c).strip() for c in df.columns]

        date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
        total_col = next((c for c in df.columns if c.lower().startswith("total")), df.columns[-1])

        df = df[[date_col, total_col]].copy()
        df.columns = ["date", "etf_netflow_usd_m"]

        date_pattern = r"\d{1,2}\s+\w+\s+\d{4}|\d{4}-\d{2}-\d{2}"
        df = df[df["date"].astype(str).str.contains(date_pattern, na=False, regex=True)]
        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
        df = df.dropna(subset=["date"]).set_index("date").sort_index()

        def _parse_flow(v):
            s = str(v).strip().replace(",", "")
            if s in ("-", "", "nan", "NaN"):
                return 0.0
            if s.startswith("(") and s.endswith(")"):
                return -float(s[1:-1])
            try:
                return float(s)
            except Exception:
                return np.nan

        df["etf_netflow_usd_m"] = df["etf_netflow_usd_m"].apply(_parse_flow)
        df = df.dropna()

        out = apply_pit_cutoff(df, cutoff_days=2)
        weekly = out.resample(week_rule).sum().dropna(how="all")
        weekly["etf_4w_cumulative"] = weekly["etf_netflow_usd_m"].rolling(4, min_periods=2).sum()
        weekly["etf_z"] = rolling_zscore(weekly["etf_4w_cumulative"], window=52, min_periods=13)
        weekly["etf_available"] = 1

        _alpha_save_disk("etf_netflow", weekly)
        LOG_ALPHA.info(f"[ETF] Farside scrape OK ({len(weekly)} weekly samples)")
        return weekly

    except Exception as e:
        LOG_ALPHA.warning(f"[ETF] Farside scrape failed: {e}")

    # ── 3순위: SoSoValue Demo API ────────────────────────────
    try:
        api_key = st.secrets.get("SOSO_API_KEY", None) if hasattr(st, "secrets") else None
        if not api_key:
            raise RuntimeError("SOSO_API_KEY not configured")

        url = "https://openapi.sosovalue.com/api/v1/etf/historicalInflowChart"
        r = requests.post(
            url,
            json={"type": "us-btc-spot"},
            headers={"x-soso-api-key": api_key, "Content-Type": "application/json"},
            timeout=20
        )
        r.raise_for_status()

        data = r.json().get("data", {})
        records = data.get("list", []) if isinstance(data, dict) else []

        if not records:
            raise RuntimeError("SoSoValue empty response")

        df = pd.DataFrame(records)
        date_col = next((c for c in ["date", "d", "time", "t"] if c in df.columns), df.columns[0])
        flow_col = next((c for c in ["totalNetInflow", "netInflow", "v"] if c in df.columns), df.columns[-1])

        df["date"] = pd.to_datetime(df[date_col]).dt.normalize()
        df["etf_netflow_usd_m"] = pd.to_numeric(df[flow_col], errors="coerce") / 1e6
        df = df.dropna(subset=["etf_netflow_usd_m"]).set_index("date").sort_index()
        df = df[["etf_netflow_usd_m"]]

        out = apply_pit_cutoff(df, cutoff_days=2)
        weekly = out.resample(week_rule).sum().dropna(how="all")
        weekly["etf_4w_cumulative"] = weekly["etf_netflow_usd_m"].rolling(4, min_periods=2).sum()
        weekly["etf_z"] = rolling_zscore(weekly["etf_4w_cumulative"], window=52, min_periods=13)
        weekly["etf_available"] = 1

        _alpha_save_disk("etf_netflow", weekly)
        LOG_ALPHA.warning(f"[ETF] SoSoValue Demo API used ({len(weekly)} samples)")
        return weekly

    except Exception as e:
        LOG_ALPHA.warning(f"[ETF] SoSoValue fallback failed: {e}")

    # ── 4순위: 디스크 캐시 ───────────────────────────────────
    cached = _alpha_load_disk("etf_netflow", max_age_hours=24 * 7)
    if cached is not None and not cached.empty:
        LOG_ALPHA.warning("[ETF] all live sources failed, using disk cache")
        return cached

    LOG_ALPHA.error("[ETF] all sources failed including disk cache")
    return pd.DataFrame(columns=["etf_netflow_usd_m", "etf_4w_cumulative", "etf_z", "etf_available"])


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


# ============================================================
# v2.19 Patch v3: Funding fetcher with US-IP-friendly fallback
# Binance Futures (fapi.binance.com) returns HTTP 451 from
# Streamlit Cloud (AWS US-East-1). Use Bybit/OKX instead.
# ============================================================
@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fetch_funding_rate_universal(
    symbol: str = "BTCUSDT",
    start_date: str = START_DATE,
) -> pd.Series:
    """
    Funding rate fetcher with US-IP-friendly fallback chain.

    Fallback order:
      1. Bybit (no US block, no auth required)
      2. OKX (no US block, no auth required)
      3. Binance (legacy, may fail with 451 from Streamlit Cloud)
      4. Disk cache

    Returns: pd.Series with raw funding rate (8h interval)
    """

    # ── 1순위: Bybit ──────────────────────────────────────────
    try:
        LOG_ALPHA.info("[Funding] Bybit fetch start")
        all_rows = []
        cursor = None
        # Bybit returns max 200 rows per call, use pagination
        for _ in range(50):  # max 50 pages = 10000 rows
            params = {
                "category": "linear",
                "symbol": symbol.upper(),
                "limit": 200,
            }
            if cursor:
                params["cursor"] = cursor

            r = requests.get(
                "https://api.bybit.com/v5/market/funding/history",
                params=params,
                timeout=20,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            r.raise_for_status()
            data = r.json()

            if data.get("retCode") != 0:
                raise RuntimeError(f"Bybit API error: {data.get('retMsg')}")

            result = data.get("result", {})
            rows = result.get("list", [])
            if not rows:
                break
            all_rows.extend(rows)

            cursor = result.get("nextPageCursor")
            if not cursor:
                break

        if not all_rows:
            raise RuntimeError("Bybit returned empty data")

        df = pd.DataFrame(all_rows)
        df["fundingRateTimestamp"] = pd.to_datetime(
            pd.to_numeric(df["fundingRateTimestamp"], errors="coerce"),
            unit="ms", utc=True
        ).dt.tz_localize(None)
        df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
        s = pd.Series(
            df["fundingRate"].values,
            index=df["fundingRateTimestamp"]
        ).sort_index()
        s = s[~s.index.duplicated(keep="last")].dropna()
        s.name = "funding_rate"

        LOG_ALPHA.info(f"[Funding] Bybit OK: {len(s)} rows")
        return s

    except Exception as e:
        LOG_ALPHA.warning(f"[Funding] Bybit failed: {e}")

    # ── 2순위: OKX ────────────────────────────────────────────
    try:
        LOG_ALPHA.info("[Funding] OKX fetch start")
        # OKX uses BTC-USDT-SWAP for perpetual
        okx_symbol = "BTC-USDT-SWAP"
        all_rows = []
        before_ts = None
        for _ in range(50):
            params = {
                "instId": okx_symbol,
                "limit": "100",
            }
            if before_ts:
                params["before"] = str(before_ts)

            r = requests.get(
                "https://www.okx.com/api/v5/public/funding-rate-history",
                params=params,
                timeout=20,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            r.raise_for_status()
            data = r.json()

            if data.get("code") != "0":
                raise RuntimeError(f"OKX error: {data.get('msg')}")

            rows = data.get("data", [])
            if not rows:
                break
            all_rows.extend(rows)

            # OKX pagination: next page uses 'before' = oldest timestamp
            before_ts = int(rows[-1]["fundingTime"])
            if len(rows) < 100:
                break

        if not all_rows:
            raise RuntimeError("OKX returned empty data")

        df = pd.DataFrame(all_rows)
        df["fundingTime"] = pd.to_datetime(
            pd.to_numeric(df["fundingTime"], errors="coerce"),
            unit="ms", utc=True
        ).dt.tz_localize(None)
        df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
        s = pd.Series(
            df["fundingRate"].values,
            index=df["fundingTime"]
        ).sort_index()
        s = s[~s.index.duplicated(keep="last")].dropna()
        s.name = "funding_rate"

        LOG_ALPHA.info(f"[Funding] OKX fallback OK: {len(s)} rows")
        return s

    except Exception as e:
        LOG_ALPHA.warning(f"[Funding] OKX failed: {e}")

    # ── 3순위: Binance (US block 가능성 높음) ─────────────────
    try:
        LOG_ALPHA.info("[Funding] Binance legacy fallback start")
        s = fetch_binance_funding_history(symbol=symbol, start_date=start_date)
        if not s.empty:
            LOG_ALPHA.info(f"[Funding] Binance legacy OK: {len(s)} rows")
            return s
    except Exception as e:
        LOG_ALPHA.warning(f"[Funding] Binance legacy failed: {e}")

    # ── 4순위: 디스크 캐시 ────────────────────────────────────
    cached = _alpha_load_disk("funding_raw", max_age_hours=24 * 7)
    if cached is not None and not cached.empty:
        LOG_ALPHA.warning("[Funding] all live sources failed, using disk cache")
        s = cached.iloc[:, 0] if isinstance(cached, pd.DataFrame) else cached
        s.name = "funding_rate"
        return s

    LOG_ALPHA.error("[Funding] all sources failed including disk cache")
    return pd.Series(dtype=float, name="funding_rate")


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
    use_auto_mvrv: bool = True,        # 신규: 자동 MVRV 사용 여부
    use_auto_reserve: bool = True,     # 신규: 자동 Reserve 사용 여부
    use_auto_etf: bool = True,         # 신규: 자동 ETF 사용 여부
) -> Dict[str, object]:
    """
    Alpha 입력 통합 로딩 함수 (v2.19 alpha layer 자동 활성화 버전).

    우선순위:
      - MVRV: 사용자 업로드 CSV > 자동 fetch (CoinMetrics → BGeometrics → 디스크)
      - Funding: Binance API (기존)
      - Reserve: 자동 fetch (BGeometrics → CoinMetrics proxy → 디스크)
      - ETF: 자동 fetch (Farside → SoSoValue → 디스크)
    """

    # === MVRV: 사용자 업로드 우선, 없으면 자동 ===
    mvrv_z = pd.Series(dtype=float, name="mvrv_z")
    mvrv_source = "none"
    mvrv_error = None

    # 1) 사용자 업로드 시도
    if mvrv_file_bytes is not None:
        try:
            mvrv_z = load_uploaded_mvrv_weekly(mvrv_file_bytes, mvrv_file_name, week_rule=week_rule)
            if not mvrv_z.dropna().empty:
                mvrv_source = mvrv_file_name or "uploaded_csv"
        except Exception as e:
            mvrv_error = f"uploaded_csv_failed: {e}"

    # 2) 업로드 없거나 실패 시 자동 fetch
    if mvrv_z.dropna().empty and use_auto_mvrv:
        try:
            mvrv_z = fetch_mvrv_zscore_auto(week_rule=week_rule)
            if not mvrv_z.dropna().empty:
                mvrv_source = "auto:coinmetrics_or_bgeometrics"
        except Exception as e:
            mvrv_error = f"auto_fetch_failed: {e}"

    # === Funding: 기존 로직 유지 ===
    funding_raw = pd.Series(dtype=float, name="funding_rate")
    funding_w = pd.Series(dtype=float, name="funding_rate_w")
    funding_z = pd.Series(dtype=float, name="funding_z")
    funding_error = None

    if use_binance_funding:
        try:
            # v2.19 Patch v3: Use universal fetcher with Bybit→OKX→Binance fallback
            funding_raw = fetch_funding_rate_universal(
                symbol=funding_symbol,
                start_date=START_DATE,
            )
            if not funding_raw.empty:
                funding_w, funding_z = build_funding_weekly_zscore(
                    funding_raw, week_rule=week_rule
                )
                # 성공한 경우 디스크 캐시에 저장
                _alpha_save_disk("funding_raw", funding_raw.to_frame())
            else:
                funding_error = "all funding sources returned empty"
        except Exception as e:
            funding_error = str(e)

    # === Reserve: 자동 fetch ===
    reserve_df = pd.DataFrame()
    reserve_source = "disabled"
    reserve_error = None

    if use_auto_reserve:
        try:
            reserve_df = fetch_exchange_reserve_auto(week_rule=week_rule)
            if not reserve_df.empty:
                reserve_source = "auto:bgeometrics_or_coinmetrics_proxy"
        except Exception as e:
            reserve_error = str(e)

    # === ETF: 자동 fetch ===
    etf_df = pd.DataFrame()
    etf_source = "disabled"
    etf_error = None
    etf_available = 0

    if use_auto_etf:
        try:
            etf_df = fetch_etf_netflow_auto(week_rule=week_rule)
            if not etf_df.empty:
                etf_source = "auto:farside_or_sosovalue"
                # 최근 ETF 데이터가 있으면 etf_available=1
                latest_etf = etf_df["etf_netflow_usd_m"].dropna()
                if not latest_etf.empty:
                    days_since = (pd.Timestamp.utcnow().tz_localize(None) - latest_etf.index.max()).days
                    etf_available = 1 if days_since <= 14 else 0
        except Exception as e:
            etf_error = str(e)

    return {
        # MVRV (기존 호환)
        "mvrv_z": mvrv_z,
        # Funding (기존 호환)
        "funding_raw": funding_raw,
        "funding_rate_w": funding_w,
        "funding_z": funding_z,
        # 신규: Reserve
        "reserve_df": reserve_df,
        # 신규: ETF
        "etf_df": etf_df,
        "etf_available": etf_available,
        # 메타데이터 (기존 + 신규)
        "meta": {
            "mvrv_loaded": bool(not mvrv_z.dropna().empty),
            "mvrv_source": mvrv_source,
            "mvrv_error": mvrv_error,
            "funding_loaded": bool(not funding_z.dropna().empty),
            "funding_symbol": funding_symbol if use_binance_funding else "disabled",
            "funding_source": (f"binance:{funding_symbol.upper()}" if use_binance_funding else "disabled"),
            "funding_error": funding_error,
            # 신규
            "reserve_loaded": bool(not reserve_df.empty),
            "reserve_source": reserve_source,
            "reserve_error": reserve_error,
            "etf_loaded": bool(not etf_df.empty),
            "etf_source": etf_source,
            "etf_error": etf_error,
            "etf_available": etf_available,
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
        raise RuntimeError("데이터 길이가 부족합니다. (기간/윈도우/lag 재검토 필요)")

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
        raise RuntimeError("2D 분석을 위한 데이터 길이가 부족합니다.")

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
    c1.metric("유효 best 비율", f"{valid_ratio:.1%}")
    c2.metric("max(|corr|) 중앙값", f"{med_max_abs:.4f}")
    c3.metric("best_corr 중앙값", f"{med_best:.4f}")
    c4.metric("유의 t-stat 셀 비율(|t|≥2)", f"{sig_cells_ratio:.1%}")


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
    # v2.19.4: Migrated to fetch_fred_with_fallback (api.stlouisfed.org first).
    walcl = fetch_fred_with_fallback(FRED_WALCL, start_date="2002-01-01").loc[START_DATE:END_DATE]  # Millions
    tga = fetch_fred_with_fallback(FRED_TGA, start_date="2008-01-01").loc[START_DATE:END_DATE]      # Millions
    rrp = fetch_fred_with_fallback(FRED_RRP, start_date="2013-01-01").loc[START_DATE:END_DATE]      # Billions daily

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
    # v2.19.4: Migrated to fetch_fred_with_fallback (api.stlouisfed.org first).
    us_m2 = fetch_fred_with_fallback(FRED_US_M2SL, start_date="1990-01-01")  # Billions USD
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
    if liq_source == "G3 Total Assets (USD)":
        # v2.19.1 Patch B: Fed + ECB + BOJ total assets, USD-converted
        s, snap = load_g3_total_assets_daily(metric="level")
        return s, snap, "G3 Total Assets (USD)", "Millions USD"
    if liq_source == "G3 YoY Change (%)":
        # v2.19.1 Patch B: G3 year-over-year change (driver-style)
        s, snap = load_g3_total_assets_daily(metric="yoy")
        return s, snap, "G3 YoY Change (%)", "Percent (YoY)"

    # v2.19.8: 컴포넌트별 변화율 옵션 (BTC 상관 측정 결과 기반)
    if liq_source == "BOJ 13w Change (%)":
        # 측정 결과: BTC 13w corr=+0.573 (가장 강한 신호)
        s, snap = load_g3_total_assets_daily(metric="boj_13w")
        return s, snap, "BOJ 13w Change (%)", "Percent (13w change)"
    if liq_source == "BOJ YoY Change (%)":
        # 측정 결과: BTC 13w corr=+0.495
        s, snap = load_g3_total_assets_daily(metric="boj_yoy")
        return s, snap, "BOJ YoY Change (%)", "Percent (YoY)"
    if liq_source == "ECB 13w Change (%)":
        # 측정 결과: BTC 26w corr=+0.430
        s, snap = load_g3_total_assets_daily(metric="ecb_13w")
        return s, snap, "ECB 13w Change (%)", "Percent (13w change)"
    if liq_source == "Fed YoY Change (%)":
        # 측정 결과: BTC 13w corr=+0.302 (Fed level -0.024 대비 12배 개선)
        s, snap = load_g3_total_assets_daily(metric="fed_yoy")
        return s, snap, "Fed YoY Change (%)", "Percent (YoY)"

    if liq_source == "G2M2 Total (USD)":
        # v2.19.6: Lyn Alden Global M2 lite (US + EU only)
        g2m2_df = fetch_g2m2_total()
        # weekly DataFrame → daily ffill (downstream code expects daily series)
        daily_idx = pd.date_range(start=START_DATE, end=END_DATE, freq="D")
        s_daily = g2m2_df["g2m2_total_usd_m"].reindex(daily_idx).ffill().dropna()
        s_daily.name = "G2M2_Total_USD_m"
        snap_df = g2m2_df.reindex(daily_idx).ffill()
        return s_daily, snap_df, "G2M2 Total (USD)", "Millions USD"
    if liq_source == "G2M2 YoY Change (%)":
        # v2.19.6: G2M2 year-over-year change (driver-style)
        g2m2_df = fetch_g2m2_total()
        daily_idx = pd.date_range(start=START_DATE, end=END_DATE, freq="D")
        s_daily = g2m2_df["g2m2_yoy_pct"].reindex(daily_idx).ffill().dropna()
        s_daily.name = "G2M2_YoY_pct"
        snap_df = g2m2_df.reindex(daily_idx).ffill()
        return s_daily, snap_df, "G2M2 YoY Change (%)", "Percent (YoY)"
    # NOTE: "G2 M2 (US+EA, USD)" 옵션은 v2.19.1 Patch B에서 폐기됨 (ECB API 변경).
    # load_g2_m2_usd_daily() 함수는 롤백 가능성을 위해 코드에 보존되어 있으나
    # 사이드바 옵션에서는 더 이상 노출되지 않음.
    raise RuntimeError(f"Unknown liquidity source: {liq_source}")


# =========================
# CORE: MAIN/TAB4가 TAB1~3 선행 실행 없이도 동작하도록
# =========================
def ensure_tab3_state_ready(base: pd.DataFrame, combo_ret: pd.Series, combo_dlt: pd.Series):
    xx = st.session_state.get("tab3_xx_weeks", None)
    chosen = st.session_state.get("tab3_chosen_combo", None)
    best_corr_last = st.session_state.get("tab3_best_corr_last", None)

    if xx is not None and chosen is not None and best_corr_last is not None:
        return int(xx), str(chosen), float(best_corr_last)

    with st.spinner("MAIN: TAB3 값이 없어 최적 lag(xx) 자동 산출 중..."):
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


def compute_alpha_state_v219(
    mvrv_state: Optional[pd.Series],
    funding_state: Optional[pd.Series],
    reserve_state: Optional[pd.Series],
    etf_state: Optional[pd.Series],
    etf_available: Optional[pd.Series],
    index: Optional[pd.Index] = None,
) -> pd.Series:
    """
    v2.19 4-signal alpha state.

    score 계산 (v2.19 갭 리뷰 5번 표 기준, ETF 포함 6점 분모 고정):
      mvrv:    +2 / -2 (중요 신호 가중)
      reserve: +1 / -1
      etf:     +1 / -1 (미가용시 0)
      funding: +1 / -1
      halving: +1 / -1 (현재 미구현, score=0)

    State map:
      norm >= 0.55  -> STRONG_BULL
      norm >= 0.20  -> BULLISH
      norm <= -0.55 -> STRONG_BEAR
      norm <= -0.20 -> BEARISH
      else          -> NEUTRAL
    """

    # Index 정렬
    all_indices = []
    for s in [mvrv_state, funding_state, reserve_state, etf_state]:
        if s is not None and not s.empty:
            all_indices.append(s.index)
    if not all_indices:
        return pd.Series("alpha_neutral", index=index or [], dtype="object")

    idx = all_indices[0]
    for other in all_indices[1:]:
        idx = idx.union(other)
    if index is not None:
        idx = idx.union(index)

    # 안전한 reindex
    def _safe_reindex(s):
        if s is None or s.empty:
            return pd.Series("alpha_neutral", index=idx, dtype="object")
        return s.reindex(idx).fillna("alpha_neutral")

    m = _safe_reindex(mvrv_state)
    f = _safe_reindex(funding_state)
    r = _safe_reindex(reserve_state)
    e = _safe_reindex(etf_state)

    # ETF availability
    if etf_available is not None and not etf_available.empty:
        av = pd.to_numeric(etf_available.reindex(idx), errors="coerce").fillna(0).astype(int)
    else:
        av = pd.Series(0, index=idx, dtype=int)

    # Component scores (v2.19 갭 리뷰 5번 표)
    mvrv_score = (m == "alpha_bull").astype(int) * 2 - (m == "alpha_bear").astype(int) * 2
    reserve_score = (r == "alpha_bull").astype(int) - (r == "alpha_bear").astype(int)
    funding_score = (f == "alpha_bull").astype(int) - (f == "alpha_bear").astype(int)
    etf_score_raw = (e == "alpha_bull").astype(int) - (e == "alpha_bear").astype(int)
    etf_score = etf_score_raw * av  # 미가용 시 0

    # Halving: 현재 미구현 → score=0 (향후 추가 가능)
    halving_score = pd.Series(0, index=idx, dtype=int)

    raw_score = mvrv_score + reserve_score + funding_score + etf_score + halving_score
    # ETF 포함 6점 분모 고정 (v2.19 갭 리뷰 결정)
    score_norm = raw_score / 6.0

    # State mapping
    out = pd.Series("NEUTRAL", index=idx, dtype="object")
    out[score_norm >= 0.55] = "STRONG_BULL"
    out[(score_norm >= 0.20) & (score_norm < 0.55)] = "BULLISH"
    out[(score_norm <= -0.20) & (score_norm > -0.55)] = "BEARISH"
    out[score_norm <= -0.55] = "STRONG_BEAR"

    return out


# ============================================================
# v2.19 Phase 1 — State × Regime Position Multiplier Override
# ============================================================
# OOS 분석(2026-05-01) 결과 기반 권장 multiplier
# 산출 근거: sign_acc → multiplier 매핑
#   sign_acc >= 70%: 1.30x (최강)
#   sign_acc 65~70%: 1.30x
#   sign_acc 60~65%: 1.20x
#   sign_acc 55~60%: 1.10x
#   sign_acc 50~55%: 1.00x (중립)
#   sign_acc 45~50%: 0.85x
#   sign_acc 40~45%: 0.60x (FOLLOW_LIGHT NEUTRAL)
#   sign_acc < 40%:  0.40x

# ============================================================
# v2.19.1 Patch A: 19w endpoint 기반 multiplier 비활성화 (2026-05-02)
# ============================================================
# 사유: 2026-05-02 path 분석 결과,
#   - 실제 best_lag 19주 발생 = 414 anchor 중 단 3회 (0.7%)
#   - Path 단위 평가에서 endpoint 결론과 정반대 결론 도출
#   - 예: FOLLOW_LIGHT는 endpoint=함정 / path=정상 (over-corrected)
#   - 예: SHRINK는 endpoint=평범 / path=함정 (under-corrected)
# 따라서 현재 multiplier 방향이 잘못되어 있을 가능성 높음.
# Path-based 재산출 전까지 비활성화 (빈 dict → .get() 모두 None → base 유지).
#
# 원본 multiplier는 코드 보존을 위해 _ARCHIVED_v2_18_5_19w_endpoint_MULTIPLIER_TABLE
# 에 그대로 보존됨 (참조용, 실제 사용 안 함).

PATCH_A_DISABLED_DATE = "2026-05-02"
PATCH_A_REASON = "19w endpoint 평가 기반 → path 평가에서 잘못된 방향 입증"

_ARCHIVED_v2_18_5_19w_endpoint_MULTIPLIER_TABLE = {
    # (alpha_state, regime_state): position_multiplier
    # 표본 수 5개 미만은 None (기존 multiplier 사용)

    # NEUTRAL × 각 regime
    ("NEUTRAL", "DEFENSIVE"):     1.10,  # n=43, sign_acc 53.5%
    ("NEUTRAL", "FOLLOW"):        0.95,  # n=37, sign_acc 48.6%
    ("NEUTRAL", "FOLLOW_LIGHT"):  0.60,  # n=57, sign_acc 42.1% ⚠️ 함정 (endpoint)
    ("NEUTRAL", "SHRINK"):        0.90,  # n=38, sign_acc 47.4%
    ("NEUTRAL", "TRANSITION"):    0.92,  # n=163, sign_acc 48.5%

    # BULLISH × 각 regime
    ("BULLISH", "DEFENSIVE"):     1.20,  # n=17, sign_acc 58.8%
    ("BULLISH", "FOLLOW"):        1.30,  # n=8, sign_acc 75.0% ⭐
    ("BULLISH", "FOLLOW_LIGHT"):  None,  # n=3, 표본 부족
    ("BULLISH", "SHRINK"):        1.30,  # n=7, sign_acc 71.4% ⭐
    ("BULLISH", "TRANSITION"):    1.30,  # n=23, sign_acc 65.2% ⭐

    # BEARISH/STRONG_BEAR/STRONG_BULL: 표본 부족 (n<5)
    # → 기존 alpha_state position_multiplier 사용
}

# v2.19.7 Patch B-1 (2026-05-06): Path-based reconstruction
# 산출 방식:
#   - 5-source × 8년 walk-forward path 분석
#   - Fed (414 anchors) 우선 + 5-src consistency 검증
#   - 매 anchor: path_corr × end_sign_accuracy × severe_loss 종합
#
# 매핑 원칙:
#   end_sa >= 0.65 → 1.20x base    end_sa < 0.40 → 0.25x base
#   path_corr >= 0.20 → 1.10x adj   path_corr < -0.10 → 0.55x adj
#   severe_loss > 20% → ×0.85 페널티
#
# 검증 (8년 시뮬레이션, $100K 시작):
#   OLD (regime only):    $188K, Sharpe 0.53, MDD -25.3%
#   NEW (alpha × regime): $227K, Sharpe 0.59, MDD -25.3%
#   → +$39K with same MDD
# 검증 (5년):
#   OLD: $118K, Sharpe 0.30
#   NEW: $146K, Sharpe 0.48 (Sharpe +0.18 큰 개선)

PATCH_B1_DATE = "2026-05-06"
PATCH_B1_RATIONALE = "5-source × 8년 walk-forward path 분석으로 산출 (Fed 414 anchors 우선)"

STATE_REGIME_MULTIPLIER_TABLE: Dict[Tuple[str, str], Optional[float]] = {
    # (alpha_state, regime_state): position_multiplier
    # 표본 < 5 조합은 None (생략) → 기존 base multiplier 자동 사용

    # NEUTRAL × 5 regimes (가장 풍부한 데이터)
    ("NEUTRAL", "DEFENSIVE"): 1.02,    # n=46, path_corr=+0.20, end_sa=65%
    ("NEUTRAL", "FOLLOW"): 0.60,       # n=37, path_corr=-0.15, end_sa=41% (trap)
    ("NEUTRAL", "FOLLOW_LIGHT"): 0.64, # n=57, path_corr=+0.02, end_sa=53%
    ("NEUTRAL", "SHRINK"): 0.80,       # n=39, path_corr=+0.05, end_sa=56%
    ("NEUTRAL", "TRANSITION"): 0.60,   # n=171, path_corr=-0.02, end_sa=49%

    # BULLISH × 4 regimes (FOLLOW_LIGHT n=3, 표본 부족)
    ("BULLISH", "DEFENSIVE"): 1.00,    # n=18, path_corr=+0.07, end_sa=67%
    ("BULLISH", "FOLLOW"): 0.25,       # n=8, path_corr=-0.19, end_sa=38% (강한 trap)
    ("BULLISH", "SHRINK"): 1.00,       # n=7, path_corr=-0.11, end_sa=57%
    ("BULLISH", "TRANSITION"): 0.70,   # n=26, path_corr=+0.02, end_sa=54%

    # 표본 부족 (None, 생략):
    #   ("BULLISH", "FOLLOW_LIGHT"): n=3
    #   ("BEARISH", "FOLLOW_LIGHT"): n=1
    #   ("BEARISH", "TRANSITION"):   n=1
    # → 기존 regime base multiplier 사용
}


def compute_state_regime_position_multiplier(
    alpha_state: pd.Series,
    regime_state: pd.Series,
    base_position_multiplier: pd.Series,
    blend_weight: float = 0.7,
) -> Tuple[pd.Series, pd.Series]:
    """
    OOS 분석 기반으로 alpha_state × regime 조합별 position multiplier를 적용.

    Args:
        alpha_state: v2.19 alpha state 시리즈
        regime_state: regime state 시리즈
        base_position_multiplier: 기존 regime/alpha 기반 multiplier
        blend_weight: 0~1, OOS 권장값과 기존값을 블렌드. 1.0이면 OOS 권장값 100% 사용.

    Returns:
        (final_multiplier, override_flag) — 두 시리즈 반환
    """
    final = base_position_multiplier.copy()
    override_flag = pd.Series("base", index=base_position_multiplier.index, dtype="object")

    # v2.19.1 Patch A 안전 가드: 비활성화 상태(빈 dict)면 즉시 반환 (모두 base)
    if not STATE_REGIME_MULTIPLIER_TABLE:
        return final, override_flag

    for idx in base_position_multiplier.index:
        a = alpha_state.get(idx, "NEUTRAL")
        r = regime_state.get(idx, None)

        if pd.isna(r) or pd.isna(a):
            continue

        rec = STATE_REGIME_MULTIPLIER_TABLE.get((a, r))
        if rec is None:
            continue

        base_val = base_position_multiplier.loc[idx]
        if pd.isna(base_val):
            continue

        # v2.19.1 Patch A 추가 안전 가드: 비활성화 시 1.0 강제
        if not STATE_REGIME_MULTIPLIER_TABLE:
            continue

        # Blend: blend_weight * OOS권장 + (1-blend_weight) * 기존값
        blended = blend_weight * rec + (1 - blend_weight) * base_val
        final.loc[idx] = blended
        override_flag.loc[idx] = f"oos:{a}x{r}"

    return final, override_flag


def classify_driver_geometry(
    z_liq_s: pd.Series,
    z_dxy_s: pd.Series,
    min_abs_threshold: float = 0.15,
    strength_threshold: float = 0.45,
) -> pd.Series:
    """
    Raw driver geometry classification based on sign alignment and minimum signal strength.
    Output labels are intentionally descriptive rather than directional promises.
    """
    z_liq_s = pd.to_numeric(z_liq_s, errors="coerce")
    z_dxy_s = pd.to_numeric(z_dxy_s, errors="coerce")
    idx = z_liq_s.index.union(z_dxy_s.index)
    z_liq_s = z_liq_s.reindex(idx)
    z_dxy_s = z_dxy_s.reindex(idx)

    min_abs = pd.concat([z_liq_s.abs(), z_dxy_s.abs()], axis=1).min(axis=1)
    strength = z_liq_s.abs().fillna(0.0) + z_dxy_s.abs().fillna(0.0)

    dir_liq = np.sign(z_liq_s)
    dir_dxy = np.sign(z_dxy_s)

    out = pd.Series(index=idx, dtype="object")
    weak_mask = (min_abs < float(min_abs_threshold)) | (strength < float(strength_threshold))
    out[weak_mask] = "LOW_SIGNAL_TRANSITION"

    strong_mask = ~weak_mask
    out[strong_mask & (dir_liq > 0) & (dir_dxy > 0)] = "POS_ALIGN"
    out[strong_mask & (dir_liq < 0) & (dir_dxy < 0)] = "NEG_ALIGN"
    out[strong_mask & (dir_liq > 0) & (dir_dxy < 0)] = "LIQUIDITY_LED_DIVERGENCE"
    out[strong_mask & (dir_liq < 0) & (dir_dxy > 0)] = "DXY_LED_DIVERGENCE"
    return out.fillna("LOW_SIGNAL_TRANSITION")


def map_action_regime(driver_geometry: pd.Series) -> pd.Series:
    """Map diagnostic geometry labels to action regimes used for calibration."""
    dg = driver_geometry.astype("object").fillna("LOW_SIGNAL_TRANSITION")
    out = pd.Series(index=dg.index, dtype="object")
    out[dg == "LIQUIDITY_LED_DIVERGENCE"] = "FOLLOW"
    out[dg == "POS_ALIGN"] = "FOLLOW_LIGHT"
    out[dg == "NEG_ALIGN"] = "DEFENSIVE"
    out[dg == "DXY_LED_DIVERGENCE"] = "SHRINK"
    out[dg == "LOW_SIGNAL_TRANSITION"] = "TRANSITION"
    return out.fillna("TRANSITION")


def compute_regime_multipliers(action_regime: pd.Series) -> pd.DataFrame:
    idx = action_regime.index
    path_mult = pd.Series(0.80, index=idx, dtype="float64")
    pos_mult = pd.Series(0.60, index=idx, dtype="float64")
    conf = pd.Series("LOW", index=idx, dtype="object")

    path_mult[action_regime == "FOLLOW"] = 0.95
    pos_mult[action_regime == "FOLLOW"] = 0.75
    conf[action_regime == "FOLLOW"] = "MID-HIGH"

    path_mult[action_regime == "FOLLOW_LIGHT"] = 0.85
    pos_mult[action_regime == "FOLLOW_LIGHT"] = 0.70
    conf[action_regime == "FOLLOW_LIGHT"] = "MID"

    path_mult[action_regime == "TRANSITION"] = 0.80
    pos_mult[action_regime == "TRANSITION"] = 0.60
    conf[action_regime == "TRANSITION"] = "LOW"

    path_mult[action_regime == "SHRINK"] = 0.70
    pos_mult[action_regime == "SHRINK"] = 0.50
    conf[action_regime == "SHRINK"] = "LOW-MID"

    path_mult[action_regime == "DEFENSIVE"] = 0.65
    pos_mult[action_regime == "DEFENSIVE"] = 0.45
    conf[action_regime == "DEFENSIVE"] = "LOW"

    return pd.DataFrame({
        "regime_path_multiplier": path_mult,
        "regime_position_multiplier": pos_mult,
        "confidence_bucket": conf,
    }, index=idx)


def add_raw_regime_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """
    one-file export에서 raw regime 판단 근거를 역추적하기 위한 진단 컬럼 추가.
    기대 입력 컬럼:
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
    out["driver_geometry"] = classify_driver_geometry(out.get("z_liq_s"), out.get("z_dxy_s"))
    out["action_regime"] = map_action_regime(out["driver_geometry"])
    # backward-compatible alias used by downstream scorecards / panels
    out["regime_state"] = out["action_regime"]
    return out


def summarize_alpha_merge_status(df: pd.DataFrame, alpha_inputs: dict) -> dict:
    """
    funding / mvrv가 실제로 merge되었는지 one-file에 남길 메타 요약.
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
    current_driver_geometry: str,
    current_confidence_bucket: str,
    current_path_mult: float,
    current_pos_mult: float,
    pred_path_current_h19_raw: pd.Series,
    pred_path_current_h19_adj: pd.Series,
) -> dict:
    """
    현재 anchor 기준 fixed-H19 decision 값을 one-file export에 반복 저장하기 위한 row 메타 생성.
    """
    out = {
        "current_anchor_dt": btc_end,
        "current_predicted_ret_19w_raw": np.nan,
        "current_predicted_ret_19w_adj": np.nan,
        "current_predicted_px_19w_raw": np.nan,
        "current_predicted_px_19w_adj": np.nan,
        "current_regime_state": current_regime_state,
        "current_driver_geometry": current_driver_geometry,
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
        raise RuntimeError("Drivers/Price 데이터가 부족합니다.")

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
            f"Forecast 가능한 최대 horizon: 0주 (drivers 기준 max_h=0주, end_dt_eff={end_dt_eff.date()})"
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
      2) ECM step model (weekly): Δlog(BTC)_t = a + b_gap * gap_{t-1} + b_dldli * ΔLDLI_t

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
    use_auto_mvrv: bool = True,
    use_auto_reserve: bool = True,
    use_auto_etf: bool = True,
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
        raise RuntimeError("Forward overlay 계산에 필요한 주간 샘플이 부족합니다.")

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
        use_auto_mvrv=use_auto_mvrv,
        use_auto_reserve=use_auto_reserve,
        use_auto_etf=use_auto_etf,
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

    # === v2.19: Reserve & ETF 통합 ===
    reserve_df = alpha_inputs.get("reserve_df", pd.DataFrame())
    if not reserve_df.empty:
        overlay_master_df["exchange_reserve_btc"] = reserve_df["exchange_reserve_btc"].reindex(overlay_master_df.index)
        overlay_master_df["reserve_pct_change_4w"] = reserve_df.get("reserve_pct_change_4w", pd.Series()).reindex(overlay_master_df.index)
        overlay_master_df["reserve_z"] = reserve_df.get("reserve_z", pd.Series()).reindex(overlay_master_df.index)
    else:
        overlay_master_df["exchange_reserve_btc"] = np.nan
        overlay_master_df["reserve_pct_change_4w"] = np.nan
        overlay_master_df["reserve_z"] = np.nan

    etf_df = alpha_inputs.get("etf_df", pd.DataFrame())
    if not etf_df.empty:
        overlay_master_df["etf_netflow_usd_m"] = etf_df["etf_netflow_usd_m"].reindex(overlay_master_df.index)
        overlay_master_df["etf_4w_cumulative"] = etf_df.get("etf_4w_cumulative", pd.Series()).reindex(overlay_master_df.index)
        overlay_master_df["etf_z"] = etf_df.get("etf_z", pd.Series()).reindex(overlay_master_df.index)
        overlay_master_df["etf_available"] = etf_df.get("etf_available", pd.Series()).reindex(overlay_master_df.index).fillna(0).astype(int)
    else:
        overlay_master_df["etf_netflow_usd_m"] = np.nan
        overlay_master_df["etf_4w_cumulative"] = np.nan
        overlay_master_df["etf_z"] = np.nan
        overlay_master_df["etf_available"] = 0

    # Reserve / ETF state classification (v2.19 갭 리뷰 점수 체계 반영)
    def classify_reserve_state(reserve_pct_change: pd.Series) -> pd.Series:
        """Reserve 4주 변화율 기반 state."""
        s = pd.to_numeric(reserve_pct_change, errors="coerce")
        out = pd.Series(index=s.index, dtype="object")
        out[s <= -0.03] = "alpha_bull"   # 거래소에서 빠지면 강세
        out[s >= 0.03] = "alpha_bear"    # 거래소로 들어오면 약세
        out[(s > -0.03) & (s < 0.03)] = "alpha_neutral"
        return out.fillna("alpha_neutral")

    def classify_etf_state(etf_z: pd.Series, etf_available: pd.Series) -> pd.Series:
        """ETF 4주 누적 z-score 기반 state. 미가용 시 neutral."""
        s = pd.to_numeric(etf_z, errors="coerce")
        av = pd.to_numeric(etf_available, errors="coerce")
        out = pd.Series(index=s.index, dtype="object")
        out[(av == 1) & (s >= 0.5)] = "alpha_bull"
        out[(av == 1) & (s <= -0.5)] = "alpha_bear"
        return out.fillna("alpha_neutral")

    overlay_master_df["reserve_state"] = classify_reserve_state(overlay_master_df["reserve_pct_change_4w"])
    overlay_master_df["etf_state"] = classify_etf_state(overlay_master_df["etf_z"], overlay_master_df["etf_available"])

    # v2.19: 4-signal alpha state (기존 2-signal 호환 유지를 위해 둘 다 계산)
    overlay_master_df["alpha_state_legacy"] = compute_alpha_state(
        overlay_master_df["mvrv_state"], overlay_master_df["funding_state"], index=overlay_master_df.index
    ).reindex(overlay_master_df.index).fillna("alpha_neutral")

    overlay_master_df["alpha_state"] = compute_alpha_state_v219(
        mvrv_state=overlay_master_df["mvrv_state"],
        funding_state=overlay_master_df["funding_state"],
        reserve_state=overlay_master_df["reserve_state"],
        etf_state=overlay_master_df["etf_state"],
        etf_available=overlay_master_df["etf_available"],
        index=overlay_master_df.index,
    ).reindex(overlay_master_df.index).fillna("NEUTRAL")

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

    # New regime redesign: keep raw regime diagnostics, but drive calibration off
    # driver geometry -> action regime instead of legacy tailwind/headwind labels.
    overlay_master_df["driver_geometry"] = classify_driver_geometry(
        overlay_master_df["z_liq_s"], overlay_master_df["z_dxy_s"]
    )
    overlay_master_df["action_regime"] = map_action_regime(overlay_master_df["driver_geometry"])
    overlay_master_df["regime_state"] = overlay_master_df["action_regime"]
    mult_df = compute_regime_multipliers(overlay_master_df["action_regime"])
    overlay_master_df = overlay_master_df.join(mult_df, how="left")

    # === v2.19 Phase 1: State × Regime Position Multiplier Override ===
    overlay_master_df["regime_position_multiplier_base"] = overlay_master_df["regime_position_multiplier"].copy()

    phase1_mult, phase1_flag = compute_state_regime_position_multiplier(
        alpha_state=overlay_master_df["alpha_state"],
        regime_state=overlay_master_df["regime_state"],
        base_position_multiplier=overlay_master_df["regime_position_multiplier"],
        blend_weight=0.7,  # OOS 권장값을 70% 가중
    )
    overlay_master_df["regime_position_multiplier_oos"] = phase1_mult
    overlay_master_df["position_multiplier_source"] = phase1_flag

    # 기본 multiplier를 OOS 보정값으로 교체 (suggested_exposure 자동 반영)
    overlay_master_df["regime_position_multiplier"] = phase1_mult

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
        current_regime_state=str(current_row.get("regime_state", "TRANSITION")),
        current_driver_geometry=str(current_row.get("driver_geometry", "LOW_SIGNAL_TRANSITION")),
        current_confidence_bucket=str(current_row.get("confidence_bucket", "LOW")),
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
        overlay_master_df, horizon_w=H19_DIAG_W, group_col="action_regime",
        pred_ret_col="predicted_fwd_ret_19w_adj", real_ret_col="realized_fwd_ret_19w"
    )
    scorecard_h19_by_driver_geometry_adj = build_horizon_scorecard(
        overlay_master_df, horizon_w=H19_DIAG_W, group_col="driver_geometry",
        pred_ret_col="predicted_fwd_ret_19w_adj", real_ret_col="realized_fwd_ret_19w"
    )
    scorecard_h19_by_raw_regime_adj = build_horizon_scorecard(
        overlay_master_df, horizon_w=H19_DIAG_W, group_col="regime_shifted",
        pred_ret_col="predicted_fwd_ret_19w_adj", real_ret_col="realized_fwd_ret_19w"
    )

    regime_diag_full = build_regime_diagnostic_table(overlay_master_df, group_col="driver_geometry", recent_key=None)
    regime_diag_recent104 = build_regime_diagnostic_table(overlay_master_df, group_col="driver_geometry", recent_key="recent_104w_flag")
    regime_diag_recent52 = build_regime_diagnostic_table(overlay_master_df, group_col="driver_geometry", recent_key="recent_52w_flag")
    action_regime_diag_full = build_regime_diagnostic_table(overlay_master_df, group_col="action_regime", recent_key=None)
    action_regime_diag_recent104 = build_regime_diagnostic_table(overlay_master_df, group_col="action_regime", recent_key="recent_104w_flag")
    action_regime_diag_recent52 = build_regime_diagnostic_table(overlay_master_df, group_col="action_regime", recent_key="recent_52w_flag")

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
        "scorecard_h19_by_driver_geometry_adj": scorecard_h19_by_driver_geometry_adj,
        "scorecard_h19_by_raw_regime_adj": scorecard_h19_by_raw_regime_adj,
        "regime_diag_full": regime_diag_full,
        "regime_diag_recent104": regime_diag_recent104,
        "regime_diag_recent52": regime_diag_recent52,
        "action_regime_diag_full": action_regime_diag_full,
        "action_regime_diag_recent104": action_regime_diag_recent104,
        "action_regime_diag_recent52": action_regime_diag_recent52,
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


# ============================================================
# v2.19 Phase 1 — Validation Dashboard (TAB7)
# ============================================================

def render_phase1_validation_tab(payload: Dict):
    """
    Phase 1 acceptance criteria를 OOS 데이터로 자동 판정.

    검증 항목:
      1. WS-A: Path/Position multiplier 효과 (조건부 분리)
      2. WS-D: State separation (alpha_state별 IC, sign_acc)
      3. v2.19 acceptance criteria 자동 판정
      4. 2D Matrix: alpha × regime
      5. FOLLOW_LIGHT regime 진단
      6. 권장 베팅 조합
    """
    st.subheader("Phase 1 Validation Dashboard")
    st.caption("v2.19 acceptance criteria 자동 판정 — 매 실행 시 재계산")

    # v2.19.1 Patch A: 안전 모드 알림
    if not STATE_REGIME_MULTIPLIER_TABLE:
        st.warning(
            f"⚠️ STATE_REGIME_MULTIPLIER 비활성화 (Patch A, {PATCH_A_DISABLED_DATE}). "
            f"사유: {PATCH_A_REASON}. "
            f"path-based 재산출 전까지 보정 미적용 (모든 조합 = 1.0)."
        )

    overlay_df = payload.get("overlay_master_df")
    if overlay_df is None or overlay_df.empty:
        st.warning("Overlay 데이터가 없습니다.")
        return

    needed = ["realized_fwd_ret_19w", "predicted_fwd_ret_19w",
              "predicted_fwd_ret_19w_adj", "alpha_state", "regime_state"]
    if not all(c in overlay_df.columns for c in needed):
        missing = [c for c in needed if c not in overlay_df.columns]
        st.error(f"필수 컬럼 누락: {missing}")
        return

    v = overlay_df.dropna(subset=["realized_fwd_ret_19w", "predicted_fwd_ret_19w"]).copy()

    if len(v) < 30:
        st.warning(f"OOS 표본이 부족합니다 (n={len(v)}). 30 이상 필요.")
        return

    st.info(f"분석 표본: n = {len(v)}")

    # === Section 1: WS-A Path Multiplier 효과 ===
    st.markdown("### 1️⃣ WS-A: Path Multiplier 조건부 효과")

    raw_pred = v["predicted_fwd_ret_19w"]
    real = v["realized_fwd_ret_19w"]

    base_correct_mask = np.sign(raw_pred) == np.sign(real)
    base_correct = v[base_correct_mask]
    base_wrong = v[~base_correct_mask]

    mae_correct_raw = (base_correct["predicted_fwd_ret_19w"] - base_correct["realized_fwd_ret_19w"]).abs().mean()
    mae_correct_adj = (base_correct["predicted_fwd_ret_19w_adj"] - base_correct["realized_fwd_ret_19w"]).abs().mean()
    mae_wrong_raw = (base_wrong["predicted_fwd_ret_19w"] - base_wrong["realized_fwd_ret_19w"]).abs().mean()
    mae_wrong_adj = (base_wrong["predicted_fwd_ret_19w_adj"] - base_wrong["realized_fwd_ret_19w"]).abs().mean()

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "방향 적중 샘플 MAE 변화",
            f"{(mae_correct_adj - mae_correct_raw)*100:+.2f}%p",
            delta=f"n={len(base_correct)}",
            delta_color="off"
        )
    with col2:
        st.metric(
            "방향 오적중 샘플 MAE 변화",
            f"{(mae_wrong_adj - mae_wrong_raw)*100:+.2f}%p",
            delta=f"n={len(base_wrong)}",
            delta_color="off"
        )

    if (mae_correct_adj < mae_correct_raw) and (mae_wrong_adj < mae_wrong_raw):
        st.success("✅ Path multiplier 양쪽 샘플 모두 개선 → 채택")
    elif mae_wrong_adj < mae_wrong_raw:
        st.info("⚠️ Path multiplier가 오적중 샘플만 개선 → 위험 완화 장치로 채택")
    else:
        st.warning("❌ Path multiplier 효과 검증 실패")

    st.divider()

    # === Section 2: WS-D State Separation ===
    st.markdown("### 2️⃣ WS-D: Alpha State별 OOS 성과")

    states_order = ["STRONG_BEAR", "BEARISH", "NEUTRAL", "BULLISH", "STRONG_BULL"]
    state_rows = []

    for state in states_order:
        sub = v[v["alpha_state"] == state]
        n = len(sub)
        if n == 0:
            state_rows.append({
                "State": state, "n": 0, "Mean Real": None,
                "Sign Acc": None, "IC": None
            })
            continue

        mean_real = sub["realized_fwd_ret_19w"].mean()
        sign_acc = (np.sign(sub["predicted_fwd_ret_19w"]) == np.sign(sub["realized_fwd_ret_19w"])).mean()
        ic = sub["predicted_fwd_ret_19w"].corr(sub["realized_fwd_ret_19w"]) if n > 5 else None

        state_rows.append({
            "State": state, "n": n,
            "Mean Real": f"{mean_real*100:+.1f}%",
            "Sign Acc": f"{sign_acc*100:.1f}%",
            "IC": f"{ic:+.3f}" if ic is not None else "N/A"
        })

    state_df = pd.DataFrame(state_rows)
    st.dataframe(state_df, use_container_width=True, hide_index=True)

    # === Section 3: Acceptance Criteria 자동 판정 ===
    st.markdown("### 3️⃣ v2.19 Acceptance Criteria 판정")

    neutral = v[v["alpha_state"] == "NEUTRAL"]
    if len(neutral) >= 15:
        neutral_ic = neutral["predicted_fwd_ret_19w"].corr(neutral["realized_fwd_ret_19w"])
        neutral_sign = (np.sign(neutral["predicted_fwd_ret_19w"]) == np.sign(neutral["realized_fwd_ret_19w"])).mean()

        st.caption(f"NEUTRAL baseline (n={len(neutral)}): IC={neutral_ic:+.3f}, sign_acc={neutral_sign*100:.1f}%")

        verdicts = []
        for state in ["BULLISH", "BEARISH", "STRONG_BULL", "STRONG_BEAR"]:
            sub = v[v["alpha_state"] == state]
            n = len(sub)

            if n < 15:
                verdicts.append({
                    "State": state, "n": n,
                    "ΔIC": "—", "Δsign_acc": "—",
                    "Verdict": f"⚠️ 보류 (n<15)"
                })
                continue

            state_ic = sub["predicted_fwd_ret_19w"].corr(sub["realized_fwd_ret_19w"])
            state_sign = (np.sign(sub["predicted_fwd_ret_19w"]) == np.sign(sub["realized_fwd_ret_19w"])).mean()

            d_ic = state_ic - neutral_ic
            d_sign = state_sign - neutral_sign

            pass_ic = abs(d_ic) > 0.05
            pass_sign = abs(d_sign) > 0.05

            if pass_ic or pass_sign:
                verdict = "✅ PASS"
            else:
                verdict = "❌ FAIL"

            verdicts.append({
                "State": state, "n": n,
                "ΔIC": f"{d_ic:+.3f}",
                "Δsign_acc": f"{d_sign*100:+.1f}%p",
                "Verdict": verdict
            })

        ver_df = pd.DataFrame(verdicts)
        st.dataframe(ver_df, use_container_width=True, hide_index=True)

        passed = sum(1 for v_ in verdicts if "PASS" in v_["Verdict"])
        if passed >= 1:
            st.success(f"✅ {passed}개 state PASS → Phase 1 통과 가능")
        else:
            st.warning(f"❌ PASS state 없음 → 추가 데이터 필요")
    else:
        st.warning(f"NEUTRAL 샘플 부족 (n={len(neutral)}). baseline 계산 불가.")

    st.divider()

    # === Section 4: 2D Matrix ===
    st.markdown("### 4️⃣ Alpha State × Regime 2D Matrix (Sign Accuracy)")

    regimes = sorted([r for r in v["regime_state"].dropna().unique()])
    matrix_data = []

    for state in states_order:
        row = {"alpha_state": state}
        for reg in regimes:
            sub = v[(v["alpha_state"] == state) & (v["regime_state"] == reg)]
            n = len(sub)
            if n < 5:
                row[reg] = f"(n={n})"
            else:
                sa = (np.sign(sub["predicted_fwd_ret_19w"]) == np.sign(sub["realized_fwd_ret_19w"])).mean()
                row[reg] = f"{sa*100:.1f}% (n={n})"
        matrix_data.append(row)

    matrix_df = pd.DataFrame(matrix_data)
    st.dataframe(matrix_df, use_container_width=True, hide_index=True)

    st.divider()

    # === Section 5: FOLLOW_LIGHT 진단 ===
    st.markdown("### 5️⃣ FOLLOW_LIGHT Regime 진단")

    fl = v[v["regime_state"] == "FOLLOW_LIGHT"]
    if len(fl) > 0:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            sa = (np.sign(fl["predicted_fwd_ret_19w"]) == np.sign(fl["realized_fwd_ret_19w"])).mean()
            st.metric("Sign Acc", f"{sa*100:.1f}%",
                     delta=f"{(sa-0.5)*100:+.1f}%p vs 50%",
                     delta_color="inverse" if sa < 0.5 else "normal")
        with col2:
            mean = fl["realized_fwd_ret_19w"].mean()
            median = fl["realized_fwd_ret_19w"].median()
            st.metric("Mean / Median", f"{mean*100:+.1f}% / {median*100:+.1f}%")
        with col3:
            loss_ratio = (fl["realized_fwd_ret_19w"] < 0).mean()
            st.metric("Loss Ratio", f"{loss_ratio*100:.1f}%",
                     delta="59% 이상이면 함정",
                     delta_color="off")
        with col4:
            severe_loss = (fl["realized_fwd_ret_19w"] < -0.20).mean()
            st.metric(">20% 폭락 비중", f"{severe_loss*100:.1f}%")

        if sa < 0.45:
            st.error(f"⚠️ FOLLOW_LIGHT는 구조적 함정 — 진입 회피 권고 (sign_acc {sa*100:.1f}%)")
        elif sa < 0.50:
            st.warning(f"⚠️ FOLLOW_LIGHT 적중률 미달 — 노출 축소 권고")
        else:
            st.info(f"FOLLOW_LIGHT 적중률 정상 ({sa*100:.1f}%)")

    st.divider()

    # === Section 6: 권장 조합 ===
    st.markdown("### 6️⃣ 권장 베팅 조합 (sign_acc 기준)")

    combos = []
    for state in states_order:
        for reg in regimes:
            sub = v[(v["alpha_state"] == state) & (v["regime_state"] == reg)]
            n = len(sub)
            if n < 5:
                continue
            sa = (np.sign(sub["predicted_fwd_ret_19w"]) == np.sign(sub["realized_fwd_ret_19w"])).mean()
            combos.append({"alpha": state, "regime": reg, "n": n,
                          "sign_acc": sa, "score": (sa - 0.5) * np.sqrt(n)})

    if combos:
        combos_df = pd.DataFrame(combos).sort_values("score", ascending=False)
        combos_df["sign_acc"] = combos_df["sign_acc"].apply(lambda x: f"{x*100:.1f}%")
        combos_df["score"] = combos_df["score"].apply(lambda x: f"{x:+.2f}")

        st.markdown("**Top 5 (적극 베팅)**")
        st.dataframe(combos_df.head(5)[["alpha", "regime", "n", "sign_acc", "score"]],
                     use_container_width=True, hide_index=True)
        st.markdown("**Bottom 3 (회피)**")
        st.dataframe(combos_df.tail(3)[["alpha", "regime", "n", "sign_acc", "score"]],
                     use_container_width=True, hide_index=True)


# =========================
# MAIN TAB
# =========================

def run_main_tab():
    st.subheader("MAIN) BTC vs LDLI (+ True-history predicted + Spaghetti)")

    # === [DEBUG] 4개 alpha 신호 fetch 직접 테스트 ===
    with st.expander("🔧 Debug: Alpha 신호 fetch 직접 테스트", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Funding API 직접 호출"):
                try:
                    test_funding = fetch_funding_rate_universal(symbol="BTCUSDT", start_date=START_DATE)
                    st.success(f"✅ Funding fetch OK: {len(test_funding)} rows")
                    st.write(test_funding.tail(5))
                except Exception as e:
                    st.error(f"❌ Funding fetch 실패: {type(e).__name__}: {e}")

            if st.button("MVRV API 직접 호출"):
                try:
                    test_mvrv = fetch_mvrv_zscore_auto()
                    st.success(f"✅ MVRV fetch OK: {len(test_mvrv)} rows")
                    st.write(test_mvrv.tail(5))
                except Exception as e:
                    st.error(f"❌ MVRV fetch 실패: {type(e).__name__}: {e}")

        with col2:
            if st.button("Reserve API 직접 호출"):
                try:
                    test_reserve = fetch_exchange_reserve_auto()
                    st.success(f"✅ Reserve fetch OK: {len(test_reserve)} rows")
                    st.write(test_reserve.tail(5))
                except Exception as e:
                    st.error(f"❌ Reserve fetch 실패: {type(e).__name__}: {e}")

            if st.button("ETF API 직접 호출"):
                try:
                    test_etf = fetch_etf_netflow_auto()
                    st.success(f"✅ ETF fetch OK: {len(test_etf)} rows")
                    st.write(test_etf.tail(5))
                except Exception as e:
                    st.error(f"❌ ETF fetch 실패: {type(e).__name__}: {e}")

    try:
        payload = build_forward_overlay_payload(LIQ_SOURCE, ALPHA_MODE, FORECAST_MODEL, use_binance_funding=USE_BINANCE_FUNDING, funding_symbol=FUNDING_SYMBOL, mvrv_file_bytes=MVRV_FILE_BYTES, mvrv_file_name=MVRV_FILE_NAME, use_auto_mvrv=USE_AUTO_MVRV, use_auto_reserve=USE_AUTO_RESERVE, use_auto_etf=USE_AUTO_ETF)
    except Exception as e:
        st.error(str(e))
        return

    alpha_meta = payload.get("alpha_meta", {})

    def _alpha_status(loaded: bool, error: Optional[str], src: str) -> str:
        if error:
            return f"⚠️ error ({src})"
        if loaded:
            return f"✅ {src}"
        return f"❌ none"

    st.caption(
        ui_text(
            f"**v2.19 Alpha Layer Status**  \n"
            f"• Funding: {_alpha_status(alpha_meta.get('funding_loaded', False), alpha_meta.get('funding_error'), alpha_meta.get('funding_source', 'NA'))}  \n"
            f"• MVRV: {_alpha_status(alpha_meta.get('mvrv_loaded', False), alpha_meta.get('mvrv_error'), alpha_meta.get('mvrv_source', 'NA'))}  \n"
            f"• Reserve: {_alpha_status(alpha_meta.get('reserve_loaded', False), alpha_meta.get('reserve_error'), alpha_meta.get('reserve_source', 'NA'))}  \n"
            f"• ETF: {_alpha_status(alpha_meta.get('etf_loaded', False), alpha_meta.get('etf_error'), alpha_meta.get('etf_source', 'NA'))} (available={alpha_meta.get('etf_available', 0)})"
        )
    )
    if alpha_meta.get("funding_error"):
        st.warning(ui_text(f"Funding auto-loader warning: {alpha_meta['funding_error']}"))

    # === v2.19 Phase 1: Position Multiplier Override 표시 ===
    _overlay_for_banner = payload.get("overlay_master_df", pd.DataFrame())
    if isinstance(_overlay_for_banner, pd.DataFrame) and "position_multiplier_source" in _overlay_for_banner.columns:
        _src_df = _overlay_for_banner.dropna(subset=["position_multiplier_source"])
        if len(_src_df) > 0:
            _last_valid = _src_df.iloc[-1]
            _src = _last_valid.get("position_multiplier_source", "base")
            _base_mult = _last_valid.get("regime_position_multiplier_base", None)
            _oos_mult = _last_valid.get("regime_position_multiplier_oos", None)
            if _src != "base" and _base_mult is not None and _oos_mult is not None and pd.notna(_base_mult) and pd.notna(_oos_mult):
                st.info(
                    f"**Phase 1 OOS 보정 적용 중**: "
                    f"기존 multiplier {float(_base_mult):.2f}x → OOS 권장 {float(_oos_mult):.2f}x ({_src})"
                )

    xx = int(payload["xx_latest"])
    st.info(
        f"Liquidity={payload['liq_label']} | DXY={payload['dxy_used']} | combo={payload['chosen']} | latest xx={xx}주"
        f" | 최근 best_corr_raw={payload['best_corr_last']:.4f}"
        f" | Spaghetti anchors=weekly (dynamic xx per anchor)"
    )

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

    st.markdown(ui_text("### Regime 성과 요약 (BTC timeline 기준, regime는 +xx주 shift 적용)"))
    met = payload.get("metrics", pd.DataFrame())
    if met is None or met.empty:
        st.warning("성과 요약을 계산할 데이터가 부족합니다.")
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

    st.markdown("### ONE FILE CSV 다운로드 (MAIN/TAB4 분석용, Past 416w + Future +xx)")
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
            "current_regime_state", "current_driver_geometry", "current_confidence_bucket",
            "current_regime_path_multiplier", "current_regime_position_multiplier", "current_suggested_exposure",
            # v2.19 신규 alpha 컬럼 — Reserve & ETF
            "exchange_reserve_btc", "reserve_pct_change_4w", "reserve_z", "reserve_state",
            "etf_netflow_usd_m", "etf_4w_cumulative", "etf_z", "etf_state", "etf_available",
            # v2.19 Phase 1 — State × Regime position multiplier override
            "regime_position_multiplier_base",
            "regime_position_multiplier_oos",
            "position_multiplier_source",
        ]
        for c in enrich_cols:
            if c in master.columns:
                df_one[c] = master[c].reindex(full_idx)

        # v2.19.1 Patch B: G3 Liquidity 컬럼 항상 출력 (LIQ_SOURCE 무관)
        # 8 columns: g3_total_usd_m, g3_index, g3_yoy_pct, g3_4w_change_pct,
        #            g3_13w_change_pct, fed_usd_m, ecb_usd_m, boj_usd_m
        _g3_cols = [
            "g3_total_usd_m", "g3_index", "g3_yoy_pct",
            "g3_4w_change_pct", "g3_13w_change_pct",
            "fed_usd_m", "ecb_usd_m", "boj_usd_m",
        ]
        try:
            _g3_panel = fetch_g3_total_assets()
            for _c in _g3_cols:
                if _c in _g3_panel.columns:
                    df_one[_c] = _g3_panel[_c].reindex(full_idx)
                else:
                    df_one[_c] = np.nan
            LOG_ALPHA.info(f"[G3] columns merged into df_one ({len(_g3_panel)} weekly obs)")
        except Exception as _g3_e:
            LOG_ALPHA.warning(f"[G3] CSV merge failed: {_g3_e}, G3 columns will be NaN")
            for _c in _g3_cols:
                df_one[_c] = np.nan

        # v2.19.6: G2M2 Liquidity 컬럼 항상 출력 (LIQ_SOURCE 무관)
        # 7 columns: g2m2_total_usd_m, g2m2_index, g2m2_yoy_pct,
        #            g2m2_4w_change_pct, g2m2_13w_change_pct, m2_us_usd_m, m2_eu_usd_m
        _g2m2_cols = [
            "g2m2_total_usd_m", "g2m2_index", "g2m2_yoy_pct",
            "g2m2_4w_change_pct", "g2m2_13w_change_pct",
            "m2_us_usd_m", "m2_eu_usd_m",
        ]
        try:
            _g2m2_panel = fetch_g2m2_total()
            for _c in _g2m2_cols:
                if _c in _g2m2_panel.columns:
                    df_one[_c] = _g2m2_panel[_c].reindex(full_idx)
                else:
                    df_one[_c] = np.nan
            LOG_ALPHA.info(f"[G2M2] columns merged into df_one ({len(_g2m2_panel)} weekly obs)")
        except Exception as _g2m2_e:
            LOG_ALPHA.warning(f"[G2M2] CSV merge failed: {_g2m2_e}, G2M2 columns will be NaN")
            for _c in _g2m2_cols:
                df_one[_c] = np.nan

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
            "regime_path_multiplier", "regime_position_multiplier",
            "regime_position_multiplier_base", "regime_position_multiplier_oos", "position_multiplier_source",
            "confidence_bucket", "suggested_exposure",
            "funding_rate_w", "funding_8w_ma", "funding_z", "funding_state",
            "mvrv_z", "mvrv_state",
            "exchange_reserve_btc", "reserve_pct_change_4w", "reserve_z", "reserve_state",
            "etf_netflow_usd_m", "etf_4w_cumulative", "etf_z", "etf_state", "etf_available",
            "alpha_state",
            "funding_available_flag", "funding_source", "funding_last_valid_dt", "funding_nonnull_count",
            "funding_weekly_mean_latest", "funding_8w_ma_latest", "funding_8w_ma_z_latest", "funding_state_latest",
            "mvrv_available_flag", "mvrv_source", "mvrv_last_valid_dt", "mvrv_nonnull_count", "alpha_inputs_ready_flag",
            "current_anchor_dt", "current_predicted_ret_19w_raw", "current_predicted_ret_19w_adj",
            "current_predicted_px_19w_raw", "current_predicted_px_19w_adj",
            "current_regime_state", "current_driver_geometry", "current_confidence_bucket",
            "current_regime_path_multiplier", "current_regime_position_multiplier", "current_suggested_exposure",
            f"latest_forecast_path_raw_to_{xx}w", f"latest_forecast_path_adj_to_{xx}w",
            "liq_source", "alpha_mode", "combo_type", "xx_latest",
            # v2.19.1 Patch B: G3 Liquidity columns
            "g3_total_usd_m", "g3_index", "g3_yoy_pct",
            "g3_4w_change_pct", "g3_13w_change_pct",
            "fed_usd_m", "ecb_usd_m", "boj_usd_m",
            # v2.19.6: G2M2 Liquidity columns
            "g2m2_total_usd_m", "g2m2_index", "g2m2_yoy_pct",
            "g2m2_4w_change_pct", "g2m2_13w_change_pct",
            "m2_us_usd_m", "m2_eu_usd_m",
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
        st.warning(f"ONE FILE CSV 생성 중 오류: {e}")

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
    st.subheader("TAB1) DXY(역축) → BTC (Weekly Returns)")

    btc_close = load_btc_close()
    dxy_close, dxy_used = load_dxy_close()
    st.caption(f"Price data loaded | DXY={dxy_used}")

    btc_wret = weekly_log_returns(btc_close, WEEK_RULE).rename("btc_wret")
    dxy_wret = weekly_log_returns(dxy_close, WEEK_RULE).rename("dxy_wret")

    lags_weeks = range(LAG_MIN_WEEKS, LAG_MAX_WEEKS + 1)
    with st.spinner("Rolling heatmaps 계산 중..."):
        corr_raw, corr_masked, beta_map, tstat_map, tstat_masked, out = rolling_maps_weekly(
            y_w=btc_wret,
            x_w=dxy_wret,
            invert_x=TAB1_INVERT_X,
            lags_weeks=lags_weeks,
            quiet=False
        )

    render_model_section("DXY(역축) vs BTC", corr_masked, beta_map, tstat_masked, out)


# =========================
# TAB2
# =========================
def run_tab_liquidity():
    st.subheader("TAB2) Liquidity → BTC (Weekly; RETURNS vs DELTA 비교)")

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
    with st.spinner("RETURNS/DELTA 두 모델을 모두 계산 중..."):
        corr_raw_r, corr_masked_r, beta_map_r, tstat_map_r, tstat_masked_r, out_r = rolling_maps_weekly(
            y_w=btc_wret, x_w=liq_ret, invert_x=TAB2_INVERT_X, lags_weeks=lags_weeks, quiet=False
        )
        corr_raw_d, corr_masked_d, beta_map_d, tstat_map_d, tstat_masked_d, out_d = rolling_maps_weekly(
            y_w=btc_wret, x_w=liq_dlt, invert_x=TAB2_INVERT_X, lags_weeks=lags_weeks, quiet=False
        )

    sub1, sub2 = st.tabs(["Liquidity Returns", "Liquidity Delta(Δ)"])
    with sub1:
        render_model_section("Liquidity RETURNS vs BTC", corr_masked_r, beta_map_r, tstat_masked_r, out_r)
    with sub2:
        render_model_section("Liquidity DELTA(Δ) vs BTC", corr_masked_d, beta_map_d, tstat_masked_d, out_d)


# =========================
# TAB3
# =========================
def run_tab_combo():
    st.subheader("TAB3) Combo(유동성 + 달러강도) → BTC")
    st.caption("corr-최대 방식 유지 | 1D Combo + 2D lag-pair(corr-max) + 다변량(참고)")

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

    with st.spinner("1D Combo(RET/DELTA) 계산 중..."):
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
        s1, s2 = st.tabs(["Combo using Liquidity RETURNS", "Combo using Liquidity DELTA(Δ)"])
        with s1:
            render_model_section("1D COMBO (z(Liq RETURNS) + z(-DXY_ret)) vs BTC", corr_masked_cr, beta_map_cr, tstat_masked_cr, out_cr)
        with s2:
            render_model_section("1D COMBO (z(Liq DELTA) + z(-DXY_ret)) vs BTC", corr_masked_cd, beta_map_cd, tstat_masked_cd, out_cd)

        st.success(f"TAB4/MAIN에 사용될 최신 lag(xx): {xx}주 | 선택 combo={chosen} | 최근 best_corr_raw={best_corr_last:.4f}")

    with sub2:
        st.caption(f"2D/다변량은 계산량 때문에 STEP={STEP_WEEKS_2D}주로 평가합니다. (pair 선택은 corr-max)")
        with st.spinner("2D lag-pair + 다변량(RET/DELTA) 계산 중..."):
            best2d_ret, pair_counts_ret, lags_list = rolling_best_pair_and_multivar(
                y=y, liq=base[liq_ret.name], dxy_ret=base["dxy_wret"],
                lags_weeks=lags_weeks, window_weeks=WINDOW_WEEKS, step_weeks=STEP_WEEKS_2D
            )
            best2d_dlt, pair_counts_dlt, _ = rolling_best_pair_and_multivar(
                y=y, liq=base["liq_delta_level"], dxy_ret=base["dxy_wret"],
                lags_weeks=lags_weeks, window_weeks=WINDOW_WEEKS, step_weeks=STEP_WEEKS_2D
            )

        s1, s2 = st.tabs(["2D using Liquidity RETURNS", "2D using Liquidity DELTA(Δ)"])
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
    st.subheader("TAB4) BTC vs LDLI(유동성+달러강도) 선행 오버레이 (Forward-Look Window)")

    try:
        payload = build_forward_overlay_payload(LIQ_SOURCE, ALPHA_MODE, FORECAST_MODEL, use_binance_funding=USE_BINANCE_FUNDING, funding_symbol=FUNDING_SYMBOL, mvrv_file_bytes=MVRV_FILE_BYTES, mvrv_file_name=MVRV_FILE_NAME, use_auto_mvrv=USE_AUTO_MVRV, use_auto_reserve=USE_AUTO_RESERVE, use_auto_etf=USE_AUTO_ETF)
    except Exception as e:
        st.error(str(e))
        return

    xx = int(payload["xx_latest"])
    st.success(
        f"사용 lag(latest xx)={xx}주 | 선택 combo={payload['chosen']} | 최근 best_corr_raw={payload['best_corr_last']:.4f}"
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

    st.markdown("### H19 Scorecard (Action Regime, Raw vs Adjusted)")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Raw")
        st.dataframe(payload.get("scorecard_h19_by_regime_raw", pd.DataFrame()))
    with col2:
        st.caption("Adjusted")
        st.dataframe(payload.get("scorecard_h19_by_regime_adj", pd.DataFrame()))

    st.markdown("### H19 Scorecard (Driver Geometry)")
    st.dataframe(payload.get("scorecard_h19_by_driver_geometry_adj", pd.DataFrame()), use_container_width=True)

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

    st.markdown("### Legacy Raw Regime Validation (for back-compat audit)")
    col5, col6 = st.columns(2)
    with col5:
        st.caption("By raw regime - Adjusted H19 scorecard")
        st.dataframe(payload.get("scorecard_h19_by_raw_regime_adj", pd.DataFrame()))
    with col6:
        st.caption("Driver geometry diagnostic means (Recent 104w)")
        st.dataframe(payload.get("regime_diag_recent104", pd.DataFrame()))

    st.markdown("### Action Regime Diagnostics")
    col7, col8 = st.columns(2)
    with col7:
        st.caption("Action regime diagnostic means (Full)")
        st.dataframe(payload.get("action_regime_diag_full", pd.DataFrame()))
    with col8:
        st.caption("Action regime diagnostic means (Recent 52w)")
        st.dataframe(payload.get("action_regime_diag_recent52", pd.DataFrame()))

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
            "driver_geometry": cur.get("current_driver_geometry"),
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
        score_driver = payload.get("scorecard_h19_by_driver_geometry_adj", pd.DataFrame())
        score_reg_raw = payload.get("scorecard_h19_by_raw_regime_adj", pd.DataFrame())
        regime_diag_recent104 = payload.get("regime_diag_recent104", pd.DataFrame())
        action_diag_recent52 = payload.get("action_regime_diag_recent52", pd.DataFrame())
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
# Tabs (keep all)
# =========================
tab_main, tab1, tab2, tab3, tab4, tab7 = st.tabs([
    "MAIN (True-history + Spaghetti)",
    "TAB1 DXY(역축) vs BTC",
    "TAB2 Liquidity vs BTC",
    "TAB3 Combo + 2D Lag-Pair",
    "TAB4 Forward Overlay (LDLI vs BTC)",
    "TAB7 Phase 1 Validation",
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

with tab7:
    try:
        _phase1_payload = build_forward_overlay_payload(
            LIQ_SOURCE, ALPHA_MODE, FORECAST_MODEL,
            use_binance_funding=USE_BINANCE_FUNDING, funding_symbol=FUNDING_SYMBOL,
            mvrv_file_bytes=MVRV_FILE_BYTES, mvrv_file_name=MVRV_FILE_NAME,
            use_auto_mvrv=USE_AUTO_MVRV, use_auto_reserve=USE_AUTO_RESERVE, use_auto_etf=USE_AUTO_ETF,
        )
        render_phase1_validation_tab(_phase1_payload)
    except Exception as _phase1_e:
        st.error(f"Phase 1 Validation 로드 실패: {_phase1_e}")