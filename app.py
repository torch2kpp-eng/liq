# streamlit_app.py
# liq | Global Liquidity (Core12) Validator
#
# Requirements:
#   streamlit, pandas, numpy, requests, python-dateutil
#
# Run:
#   streamlit run streamlit_app.py

import io
import json
import math
import zipfile
from io import BytesIO
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

# =========================
# App Config
# =========================
st.set_page_config(page_title="liq | Global Liquidity Validator", layout="wide")

WEEK_RULE = "W-FRI"

CORE12 = ["US", "EA", "CA", "CH", "JP", "CN", "GB", "AU", "IN", "KR", "BR", "MX"]

CCY = {
    "US": "USD",
    "EA": "EUR",
    "CA": "CAD",
    "CH": "CHF",
    "JP": "JPY",
    "CN": "CNY",
    "GB": "GBP",
    "AU": "AUD",
    "IN": "INR",
    "KR": "KRW",
    "BR": "BRL",
    "MX": "MXN",
}

# Auto sources we implement in this validator
AUTO_COUNTRIES = {"US", "EA", "CA"}  # CA includes BOTH M2 & M2++ auto

# Canada: Valet group code that contains monetary aggregates table
BOC_GROUP_MONETARY = "ATABLE_MONETARY_AGGREGATES"

# =========================
# HTTP helpers
# =========================
def http_get(url: str, params=None, timeout=30) -> requests.Response:
    r = requests.get(
        url,
        params=params,
        timeout=timeout,
        headers={"User-Agent": "liq-validator/1.0"},
    )
    r.raise_for_status()
    return r

def parse_two_col_csv(text: str, date_col_guess=("DATE", "TIME_PERIOD"), value_col_guess=("VALUE", "OBS_VALUE")) -> pd.Series:
    df = pd.read_csv(io.StringIO(text))
    # detect date col
    date_col = None
    for c in date_col_guess:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        # fallback: first column
        date_col = df.columns[0]

    # detect value col
    value_col = None
    for c in value_col_guess:
        if c in df.columns:
            value_col = c
            break
    if value_col is None:
        # fallback: last column
        value_col = df.columns[-1]

    out = df[[date_col, value_col]].copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out = out.dropna(subset=[date_col]).set_index(date_col).sort_index()
    return out[value_col].astype(float)

def to_weekly_last_ffill(s: pd.Series) -> pd.Series:
    if s.empty:
        return s
    s = s.sort_index()
    return s.resample(WEEK_RULE).last().ffill()

# =========================
# FX via ECB (cross through EUR)
# =========================
@st.cache_data(show_spinner=False, ttl=6 * 60 * 60)
def fetch_ecb_exr_monthly(base: str, quote: str = "EUR") -> pd.Series:
    """
    ECB EXR SDMX REST:
      key = M.{base}.{quote}.SP00.A
    Interpreting:
      base=USD, quote=EUR  => 1 EUR = x USD  (EURUSD)
      base=CAD, quote=EUR  => 1 EUR = x CAD  (EURCAD)
    """
    key = f"M.{base}.{quote}.SP00.A"
    url = f"https://data-api.ecb.europa.eu/service/data/EXR/{key}"
    txt = http_get(url, params={"format": "csvdata"}).text
    s = parse_two_col_csv(txt, date_col_guess=("TIME_PERIOD",), value_col_guess=("OBS_VALUE",))
    s.name = f"EXR_{base}{quote}"
    return s

@st.cache_data(show_spinner=False, ttl=6 * 60 * 60)
def fx_to_usd_monthly(ccy: str) -> pd.Series:
    """
    Build LCY->USD cross rate using ECB:
      LCYUSD = (EURUSD) / (EURLCY)
    where:
      EURUSD = USD per EUR
      EURLCY = LCY per EUR
    """
    if ccy == "USD":
        idx = pd.date_range("1990-01-01", pd.Timestamp.today().normalize(), freq="MS")
        return pd.Series(1.0, index=idx, name="FX_USDUSD")

    eurusd = fetch_ecb_exr_monthly("USD", "EUR")   # USD per EUR
    eurccy = fetch_ecb_exr_monthly(ccy, "EUR")     # CCY per EUR

    df = pd.concat([eurusd, eurccy], axis=1).dropna()
    fx = df.iloc[:, 0] / df.iloc[:, 1]
    fx.name = f"FX_{ccy}USD"
    return fx

# =========================
# Auto: US M2 (FRED)
# =========================
@st.cache_data(show_spinner=False, ttl=6 * 60 * 60)
def fetch_us_m2_fred() -> pd.Series:
    # FRED M2SL (Billions USD)
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    txt = http_get(url, params={"id": "M2SL"}).text
    df = pd.read_csv(io.StringIO(txt))
    # columns: DATE, M2SL
    df = df.rename(columns={"DATE": "DATE", "M2SL": "VALUE"})
    s = parse_two_col_csv(df.to_csv(index=False), date_col_guess=("DATE",), value_col_guess=("VALUE",))
    s.name = "US_M2_BIL_USD"
    return s

# =========================
# Auto: EA Broad Money (ECB BSI)
#   - default: M3 (M30)
# =========================
@st.cache_data(show_spinner=False, ttl=6 * 60 * 60)
def fetch_ea_bsi_monthly(series_key: str) -> pd.Series:
    """
    ECB SDMX REST:
      https://data-api.ecb.europa.eu/service/data/BSI/{series_key}?format=csvdata
    Typical:
      M3: ... M30 ...
      M2: ... M20 ...
    Units commonly: Millions of EUR (depends on series)
    """
    url = f"https://data-api.ecb.europa.eu/service/data/BSI/{series_key}"
    txt = http_get(url, params={"format": "csvdata"}).text
    s = parse_two_col_csv(txt, date_col_guess=("TIME_PERIOD",), value_col_guess=("OBS_VALUE",))
    s.name = f"EA_BSI_{series_key}"
    return s

# =========================
# Auto: Canada M2 and M2++ via BoC Valet group
#   - 핵심: seriesDetail 라벨을 보고 코드 자동 식별
# =========================
@st.cache_data(show_spinner=False, ttl=6 * 60 * 60)
def fetch_boc_group(group: str, start_date: str = "2000-01-01") -> dict:
    """
    Valet group endpoint:
      /valet/observations/group/{group}/json?start_date=YYYY-MM-DD
    Returns JSON including:
      - observations: list of dicts with 'd' (date) and series codes
      - seriesDetail: dict mapping series_code -> metadata (label, etc.)
    """
    url = f"https://www.bankofcanada.ca/valet/observations/group/{group}/json"
    j = http_get(url, params={"start_date": start_date}).json()
    return j

def _boc_series_code_by_label(series_detail: dict, must_contain: list[str]) -> str | None:
    """
    Pick series code whose label contains all tokens (case-insensitive).
    """
    tokens = [t.lower() for t in must_contain]
    for code, meta in series_detail.items():
        label = str(meta.get("label", "")).lower()
        if all(t in label for t in tokens):
            return code
    return None

@st.cache_data(show_spinner=False, ttl=6 * 60 * 60)
def fetch_canada_m2_and_m2pp(start_date: str = "2000-01-01") -> tuple[pd.Series, pd.Series, dict]:
    """
    Returns:
      m2_local, m2pp_local, meta
    Note:
      Values are whatever unit BoC provides for the table (often x 1,000,000).
      We'll apply a multiplier in main logic (user-configurable; default 1e6).
    """
    j = fetch_boc_group(BOC_GROUP_MONETARY, start_date=start_date)
    series_detail = j.get("seriesDetail", {})

    # Try to identify seasonally adjusted gross M2 and M2++
    # Labels in BoC tables commonly include:
    #   "M2 (gross) (Seasonally adjusted)"
    #   "M2++ (gross) (Seasonally adjusted)"
    code_m2 = _boc_series_code_by_label(series_detail, ["m2 (gross)", "seasonally"])
    code_m2pp = _boc_series_code_by_label(series_detail, ["m2++ (gross)", "seasonally"])

    if code_m2 is None or code_m2pp is None:
        # Fallback: try without "(gross)" token
        if code_m2 is None:
            code_m2 = _boc_series_code_by_label(series_detail, ["m2", "seasonally"])
        if code_m2pp is None:
            code_m2pp = _boc_series_code_by_label(series_detail, ["m2++", "seasonally"])

    if code_m2 is None or code_m2pp is None:
        raise ValueError(
            f"Could not identify CA M2 / M2++ series codes from Valet group '{BOC_GROUP_MONETARY}'. "
            f"Found codes={len(series_detail)}. You may need to inspect seriesDetail labels."
        )

    obs = j.get("observations", [])
    if not obs:
        raise ValueError("BoC Valet returned no observations.")

    # Build dataframe from observations
    rows = []
    for o in obs:
        d = o.get("d")
        v_m2 = o.get(code_m2, {}).get("v")
        v_m2pp = o.get(code_m2pp, {}).get("v")
        rows.append({"date": d, "M2": v_m2, "M2PP": v_m2pp})

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["M2"] = pd.to_numeric(df["M2"], errors="coerce")
    df["M2PP"] = pd.to_numeric(df["M2PP"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()

    m2 = df["M2"].astype(float)
    m2.name = "CA_M2_LOCAL"
    m2pp = df["M2PP"].astype(float)
    m2pp.name = "CA_M2PP_LOCAL"

    meta = {
        "group": BOC_GROUP_MONETARY,
        "series_code_m2": code_m2,
        "series_label_m2": series_detail.get(code_m2, {}).get("label", ""),
        "series_code_m2pp": code_m2pp,
        "series_label_m2pp": series_detail.get(code_m2pp, {}).get("label", ""),
    }
    return m2, m2pp, meta

# =========================
# Manual CSV loader (for non-auto countries)
# =========================
def load_manual_m2(country: str, uploaded_file) -> pd.Series:
    """
    Required CSV schema (case-insensitive):
      date,value
    Example:
      2024-01-31, 4496.02
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

# =========================
# Validation helpers
# =========================
def validate_series_basic(s: pd.Series) -> dict:
    if s is None or s.empty:
        return {
            "n": 0, "start": None, "end": None,
            "missing_ratio": np.nan,
            "nonpositive_ratio": np.nan,
            "has_gaps_gt_45d": False,
        }
    s2 = s.copy()
    idx = s2.index
    gaps = idx.to_series().diff().dt.days.dropna()
    return {
        "n": int(s2.shape[0]),
        "start": idx.min().date(),
        "end": idx.max().date(),
        "missing_ratio": float(s2.isna().mean()),
        "nonpositive_ratio": float((s2 <= 0).mean()),
        "has_gaps_gt_45d": bool((gaps > 45).any()) if len(gaps) else False,
    }

def log_yoy_weekly(s_weekly: pd.Series, weeks: int = 52) -> pd.Series:
    s = s_weekly.copy()
    out = np.log(s) - np.log(s.shift(weeks))
    out.name = f"{s.name}_logYoY"
    return out

# =========================
# UI
# =========================
st.title("liq | Global Liquidity Validator (Core12)")

with st.sidebar:
    st.header("Scope")
    selected = st.multiselect("Countries", CORE12, default=CORE12)

    st.divider()
    st.header("EA Aggregate")
    ea_use_m3 = st.checkbox("Use EA M3 (recommended)", value=True)
    st.caption("EA: default M3 (broad money). Disable to use M2-like series (M20).")

    st.divider()
    st.header("Canada")
    ca_use_m2pp_in_global = st.checkbox("Use CA M2++ in GLOBAL sum", value=True)
    ca_unit_multiplier = st.selectbox("CA unit multiplier (local)", options=[1.0, 1e3, 1e6, 1e9], index=2)
    st.caption("BoC monetary aggregates often reported in x 1,000,000. Default set to 1e6 for safety.")

    st.divider()
    st.header("Manual uploads (non-auto)")
    manual_scale = st.selectbox("Manual M2 unit multiplier (local)", options=[1.0, 1e3, 1e6, 1e9], index=0)
    uploads = {}
    for c in selected:
        if c in AUTO_COUNTRIES:
            continue
        uploads[c] = st.file_uploader(f"{c} M2 CSV (date,value)", type=["csv"], key=f"up_{c}")

    st.divider()
    strict_mode = st.checkbox("Strict mode (block GLOBAL if missing)", value=True)
    run_btn = st.button("Run / Refresh")

if not run_btn:
    st.info("왼쪽에서 설정 후 Run / Refresh를 누르세요.")
    st.stop()

# =========================
# Build local M2 (monthly) + FX (monthly) + USD weekly outputs
# =========================
errors = []
m2_local = {}      # monthly local series
fx_monthly = {}    # monthly FX_{CCY}USD
meta_rows = []     # for meta.json

# --- US (auto) ---
if "US" in selected:
    try:
        us = fetch_us_m2_fred()  # billions USD
        # convert to USD units
        us_usd = us * 1e9
        m2_local["US"] = us_usd  # treat as local already USD units
        fx_monthly["US"] = fx_to_usd_monthly("USD")
        meta_rows.append({
            "country": "US",
            "ccy": "USD",
            "m2_source": "FRED",
            "m2_series": "M2SL",
            "m2_unit_note": "Billions USD -> multiplied by 1e9",
            "fx_source": "ECB (USDUSD=1)",
            "fx_note": "1.0",
        })
    except Exception as e:
        errors.append(f"US auto fetch failed: {e}")

# --- EA (auto) ---
if "EA" in selected:
    try:
        if ea_use_m3:
            # ECB BSI M3: M30 (Millions EUR, typically)
            ea_key = "BSI.M.U2.Y.V.M30.X.1.U2.2300.Z01.E"
            ea_name = "EA_M3"
        else:
            # ECB BSI M2: M20
            ea_key = "BSI.M.U2.Y.V.M20.X.1.U2.2300.Z01.E"
            ea_name = "EA_M2"

        ea = fetch_ea_bsi_monthly(ea_key)
        # ECB BSI series commonly in Millions of EUR -> convert to EUR units
        ea_eur = ea * 1e6
        m2_local["EA"] = ea_eur
        fx_monthly["EA"] = fx_to_usd_monthly("EUR")
        meta_rows.append({
            "country": "EA",
            "ccy": "EUR",
            "m2_source": "ECB SDMX (BSI)",
            "m2_series": ea_key,
            "m2_unit_note": "Assumed Millions EUR -> multiplied by 1e6",
            "fx_source": "ECB EXR",
            "fx_note": "EUR->USD cross via EURUSD/EURCCY",
            "alias": ea_name,
        })
    except Exception as e:
        errors.append(f"EA auto fetch failed: {e}")

# --- CA (auto, M2 + M2++) ---
ca_meta = {}
ca_m2 = None
ca_m2pp = None
if "CA" in selected:
    try:
        ca_m2_raw, ca_m2pp_raw, ca_meta = fetch_canada_m2_and_m2pp(start_date="2000-01-01")
        # apply unit multiplier (default 1e6)
        ca_m2 = ca_m2_raw * float(ca_unit_multiplier)
        ca_m2pp = ca_m2pp_raw * float(ca_unit_multiplier)

        # store both in local dict (separate keys)
        m2_local["CA_M2"] = ca_m2
        m2_local["CA_M2PP"] = ca_m2pp
        fx_monthly["CA"] = fx_to_usd_monthly("CAD")

        meta_rows.append({
            "country": "CA",
            "ccy": "CAD",
            "m2_source": "BoC Valet group",
            "m2_group": BOC_GROUP_MONETARY,
            "m2_series_m2": ca_meta.get("series_code_m2"),
            "m2_label_m2": ca_meta.get("series_label_m2"),
            "m2_series_m2pp": ca_meta.get("series_code_m2pp"),
            "m2_label_m2pp": ca_meta.get("series_label_m2pp"),
            "m2_unit_note": f"Applied multiplier={ca_unit_multiplier}",
            "fx_source": "ECB EXR",
        })
    except Exception as e:
        errors.append(f"CA auto fetch failed: {e}")

# --- Manual countries ---
for c in selected:
    if c in AUTO_COUNTRIES:
        continue
    try:
        if uploads.get(c) is None:
            if strict_mode:
                errors.append(f"Manual M2 missing for {c} (CSV not uploaded).")
            continue
        s = load_manual_m2(c, uploads[c]) * float(manual_scale)
        m2_local[c] = s
        fx_monthly[c] = fx_to_usd_monthly(CCY[c])
        meta_rows.append({
            "country": c,
            "ccy": CCY[c],
            "m2_source": "Manual CSV upload",
            "m2_unit_note": f"Applied multiplier={manual_scale}",
            "fx_source": "ECB EXR",
        })
    except Exception as e:
        errors.append(f"{c} manual load failed: {e}")

if errors:
    st.error("Errors:\n- " + "\n- ".join(errors))
    if strict_mode:
        st.stop()

# =========================
# Convert to USD weekly (W-FRI)
# =========================
usd_weekly = {}
fx_weekly = {}
local_weekly = {}

# build weekly fx for each currency key in fx_monthly
for k, fxm in fx_monthly.items():
    fx_weekly[k] = to_weekly_last_ffill(fxm)

# convert each local series to USD weekly
for k, s in m2_local.items():
    if s is None or len(s) == 0:
        continue
    s_w = to_weekly_last_ffill(s)
    local_weekly[k] = s_w

    if k == "US":
        # already USD units
        usd_weekly["US"] = s_w
    elif k.startswith("CA_"):
        # CA_M2 or CA_M2PP : use CADUSD
        fx = fx_weekly.get("CA")
        if fx is None or fx.empty:
            continue
        usd_weekly[k] = s_w * fx
    elif k == "EA":
        fx = fx_weekly.get("EA")
        if fx is None or fx.empty:
            continue
        usd_weekly["EA"] = s_w * fx
    else:
        # normal case: country code equals key
        c = k
        fx = fx_weekly.get(c)
        if fx is None or fx.empty:
            continue
        usd_weekly[c] = s_w * fx

# Assemble components for GLOBAL sum
components = {}

# US
if "US" in selected and "US" in usd_weekly:
    components["US"] = usd_weekly["US"]

# EA
if "EA" in selected and "EA" in usd_weekly:
    components["EA"] = usd_weekly["EA"]

# CA: choose M2++ or M2 for global
if "CA" in selected:
    if ca_use_m2pp_in_global and "CA_M2PP" in usd_weekly:
        components["CA"] = usd_weekly["CA_M2PP"]
    elif (not ca_use_m2pp_in_global) and "CA_M2" in usd_weekly:
        components["CA"] = usd_weekly["CA_M2"]

# Others (manual)
for c in selected:
    if c in {"US", "EA", "CA"}:
        continue
    if c in usd_weekly:
        components[c] = usd_weekly[c]

df_components = pd.concat(components.values(), axis=1) if components else pd.DataFrame()
if not df_components.empty:
    df_components.columns = list(components.keys())
    df_components = df_components.sort_index()

# Global sum
global_usd = df_components.sum(axis=1, min_count=len(df_components.columns) if strict_mode else 1)
global_usd.name = "GLOBAL_M2_USD"

# Diagnostics
diag_rows = []
for name, s in usd_weekly.items():
    info = validate_series_basic(s)
    diag_rows.append({"series": name, **info})
diag_df = pd.DataFrame(diag_rows).set_index("series").sort_index()

# YoY (weekly)
global_yoy = log_yoy_weekly(global_usd, 52)

# =========================
# Display
# =========================
col1, col2 = st.columns([2.3, 1.2], gap="large")

with col1:
    st.subheader("A) GLOBAL (USD) — Weekly (W-FRI)")
    st.line_chart(pd.concat([global_usd, global_yoy], axis=1))

    st.subheader("B) Components used in GLOBAL (USD) — Weekly")
    st.line_chart(df_components)

    if "CA" in selected and ("CA_M2" in usd_weekly or "CA_M2PP" in usd_weekly):
        st.subheader("C) Canada comparison (USD) — M2 vs M2++ (Weekly)")
        ca_cmp = pd.DataFrame(index=df_components.index)
        if "CA_M2" in usd_weekly:
            ca_cmp["CA_M2_USD"] = usd_weekly["CA_M2"]
        if "CA_M2PP" in usd_weekly:
            ca_cmp["CA_M2PP_USD"] = usd_weekly["CA_M2PP"]
        st.line_chart(ca_cmp)

with col2:
    st.subheader("D) Diagnostics (weekly series)")
    st.dataframe(diag_df)

    st.subheader("E) Canada Valet mapping")
    if ca_meta:
        st.json(ca_meta)
    else:
        st.caption("Canada not selected or fetch failed.")

# =========================
# One-click Export (ZIP)
# =========================
st.subheader("F) One-click Export (ALL data as ZIP)")

# weekly_usd_wide.csv : MAIN APP input
weekly_usd_wide = pd.concat([global_usd, df_components], axis=1).sort_index()

# monthly_m2_local_wide.csv : validation
monthly_local_wide = pd.DataFrame()
for k, s in m2_local.items():
    monthly_local_wide[k] = s
monthly_local_wide = monthly_local_wide.sort_index()

# monthly_fx_to_usd_wide.csv : validation
monthly_fx_wide = pd.DataFrame()
for k, s in fx_monthly.items():
    monthly_fx_wide[k] = s
monthly_fx_wide = monthly_fx_wide.sort_index()

# diagnostics_summary.csv
diagnostics_summary = diag_df.copy()

# meta.json
meta_obj = {
    "repo": "liq",
    "dataset": "global_liquidity_core12",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "frequency_policy": {
        "m2_raw": "monthly (mixed sources)",
        "fx_raw": "monthly ECB EXR cross rates via EUR",
        "output": f"weekly {WEEK_RULE} last + ffill",
    },
    "selected_countries": selected,
    "currency_map": {k: CCY.get(k, None) for k in selected},
    "canada": {
        "use_m2pp_in_global": ca_use_m2pp_in_global,
        "unit_multiplier": ca_unit_multiplier,
        "valet_group": BOC_GROUP_MONETARY,
        "valet_mapping": ca_meta,
    },
    "ea": {
        "use_m3": ea_use_m3,
        "bsi_key": "M30 (M3)" if ea_use_m3 else "M20 (M2)",
    },
    "notes": [
        "weekly_usd_wide.csv is intended to be uploaded into BTC Lead-Lag Lab as the liquidity input.",
        "monthly_m2_local_wide.csv and monthly_fx_to_usd_wide.csv support reconciliation and audit.",
        "Canada M2/M2++ are auto-identified from BoC Valet group series labels (no hardcoded series codes).",
    ],
    "series_meta": meta_rows,
}

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True).encode("utf-8-sig")

zip_buffer = BytesIO()
with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as z:
    z.writestr("weekly_usd_wide.csv", df_to_csv_bytes(weekly_usd_wide))
    z.writestr("monthly_m2_local_wide.csv", df_to_csv_bytes(monthly_local_wide))
    z.writestr("monthly_fx_to_usd_wide.csv", df_to_csv_bytes(monthly_fx_wide))
    z.writestr("diagnostics_summary.csv", df_to_csv_bytes(diagnostics_summary))
    z.writestr("meta.json", json.dumps(meta_obj, ensure_ascii=False, indent=2).encode("utf-8"))

zip_buffer.seek(0)

st.download_button(
    "Download ALL as ZIP (weekly+monthly+fx+diagnostics+meta)",
    data=zip_buffer,
    file_name="liq_global_liquidity_export.zip",
    mime="application/zip",
)

st.caption(
    "권장 워크플로우: ZIP을 내려받아 (1) 나에게 업로드해 정합성 검증 → "
    "(2) 검증 완료 후 weekly_usd_wide.csv를 BTC Lead-Lag Lab에 업로드해 유동성 입력으로 사용."
)