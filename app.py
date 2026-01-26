# streamlit_app.py
# liq | US+EA Liquidity Validator (M2 unified) | 2010+ only | One-click ZIP export
#
# pip install streamlit pandas numpy requests
# streamlit run streamlit_app.py

import io
import json
import zipfile
from io import BytesIO
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

# =========================
# Fixed policy (as requested)
# =========================
WEEK_RULE = "W-FRI"
CLIP_START = pd.Timestamp("2010-01-01")

# EA: M2 series key (ECB BSI)
EA_M2_KEY = "BSI.M.U2.Y.V.M20.X.1.U2.2300.Z01.E"  # M2
# US: FRED M2SL (billions USD)

st.set_page_config(page_title="liq | US+EA (M2) 2010+", layout="wide")


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


def parse_two_col_csv(text: str, date_candidates=("DATE", "TIME_PERIOD"), value_candidates=("VALUE", "OBS_VALUE")) -> pd.Series:
    df = pd.read_csv(io.StringIO(text))
    date_col = next((c for c in date_candidates if c in df.columns), None) or df.columns[0]
    value_col = next((c for c in value_candidates if c in df.columns), None) or df.columns[-1]

    out = df[[date_col, value_col]].copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out = out.dropna(subset=[date_col]).set_index(date_col).sort_index()
    return out[value_col].astype(float)


def clip_2010(s: pd.Series) -> pd.Series:
    if s is None or s.empty:
        return pd.Series(dtype=float)
    return s.loc[s.index >= CLIP_START].copy()


def to_weekly_last_ffill(s: pd.Series) -> pd.Series:
    if s is None or s.empty:
        return pd.Series(dtype=float)
    return s.sort_index().resample(WEEK_RULE).last().ffill()


def multiply_with_fx_aligned(local_weekly: pd.Series, fx_weekly: pd.Series) -> pd.Series:
    fx_aligned = fx_weekly.reindex(local_weekly.index).ffill()
    return local_weekly * fx_aligned


def validate_series_basic(s: pd.Series) -> dict:
    if s is None or s.empty:
        return {
            "n": 0,
            "start": None,
            "end": None,
            "missing_ratio": np.nan,
            "nonpositive_ratio": np.nan,
            "has_gaps_gt_45d": False,
        }
    gaps = s.index.to_series().diff().dt.days.dropna()
    return {
        "n": int(s.shape[0]),
        "start": s.index.min().date(),
        "end": s.index.max().date(),
        "missing_ratio": float(s.isna().mean()),
        "nonpositive_ratio": float((s <= 0).mean()),
        "has_gaps_gt_45d": bool((gaps > 45).any()) if len(gaps) else False,
    }


# =========================
# FX via ECB
# =========================
@st.cache_data(show_spinner=False, ttl=6 * 60 * 60)
def fetch_ecb_exr_monthly(base: str, quote: str = "EUR") -> pd.Series:
    """
    ECB EXR key:
      M.{base}.{quote}.SP00.A
    Example:
      M.USD.EUR.SP00.A = USD per EUR (EURUSD)
    """
    key = f"M.{base}.{quote}.SP00.A"
    url = f"https://data-api.ecb.europa.eu/service/data/EXR/{key}"
    txt = http_get(url, params={"format": "csvdata"}).text
    s = parse_two_col_csv(txt, date_candidates=("TIME_PERIOD",), value_candidates=("OBS_VALUE",))
    s.name = f"EXR_{base}{quote}"
    return s


@st.cache_data(show_spinner=False, ttl=6 * 60 * 60)
def fx_eur_to_usd_monthly() -> pd.Series:
    # EURUSD: USD per EUR
    eurusd = fetch_ecb_exr_monthly("USD", "EUR")
    eurusd.name = "FX_EURUSD"
    return eurusd


# =========================
# US M2 (FRED)
# =========================
@st.cache_data(show_spinner=False, ttl=6 * 60 * 60)
def fetch_us_m2_usd() -> pd.Series:
    """
    FRED M2SL: Billions USD -> convert to USD by *1e9
    """
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    txt = http_get(url, params={"id": "M2SL"}).text
    df = pd.read_csv(io.StringIO(txt)).rename(columns={"DATE": "DATE", "M2SL": "VALUE"})
    s = parse_two_col_csv(df.to_csv(index=False), date_candidates=("DATE",), value_candidates=("VALUE",))
    s = s * 1e9
    s.name = "US_M2_USD"
    return s


# =========================
# EA M2 (ECB BSI)
# =========================
@st.cache_data(show_spinner=False, ttl=6 * 60 * 60)
def fetch_ea_m2_eur() -> pd.Series:
    """
    ECB BSI endpoint:
      /service/data/BSI/{key_without_BSI_prefix}?format=csvdata
    Note: dataset often in Millions of EUR -> convert by *1e6
    """
    key = EA_M2_KEY.replace("BSI.", "", 1) if EA_M2_KEY.startswith("BSI.") else EA_M2_KEY
    url = f"https://data-api.ecb.europa.eu/service/data/BSI/{key}"
    txt = http_get(url, params={"format": "csvdata"}).text
    s = parse_two_col_csv(txt, date_candidates=("TIME_PERIOD",), value_candidates=("OBS_VALUE",))
    s = s * 1e6  # assumed millions EUR -> EUR
    s.name = "EA_M2_EUR"
    return s


# =========================
# UI
# =========================
st.title("liq | US + EA Liquidity (M2 unified) | 2010+ only")

with st.sidebar:
    st.header("Fixed Policy")
    st.write("- US: M2 (FRED M2SL)")
    st.write("- EA: M2 (ECB BSI M20)")
    st.write(f"- Clip start: {CLIP_START.date()}")
    strict_mode = st.checkbox("Strict (require both US and EA)", value=True)
    run_btn = st.button("Run / Refresh")

if not run_btn:
    st.info("Run / Refresh를 누르세요.")
    st.stop()

errors = []

# =========================
# Load monthly series
# =========================
try:
    us_m2 = clip_2010(fetch_us_m2_usd())
except Exception as e:
    us_m2 = pd.Series(dtype=float)
    errors.append(f"US fetch failed: {e}")

try:
    ea_m2_eur = clip_2010(fetch_ea_m2_eur())
except Exception as e:
    ea_m2_eur = pd.Series(dtype=float)
    errors.append(f"EA fetch failed: {e}")

try:
    eurusd = clip_2010(fx_eur_to_usd_monthly())
except Exception as e:
    eurusd = pd.Series(dtype=float)
    errors.append(f"FX(EURUSD) fetch failed: {e}")

if errors:
    st.error("Errors:\n- " + "\n- ".join(errors))
    if strict_mode:
        st.stop()

# =========================
# Weekly conversion (W-FRI) + FX alignment
# =========================
us_w = to_weekly_last_ffill(us_m2)

ea_w_local = to_weekly_last_ffill(ea_m2_eur)
fx_w = to_weekly_last_ffill(eurusd)

ea_w_usd = multiply_with_fx_aligned(ea_w_local, fx_w)
ea_w_usd.name = "EA_M2_USD"

# Clip again in weekly space (safety)
us_w = clip_2010(us_w)
ea_w_usd = clip_2010(ea_w_usd)

# Align indices (outer join not needed; we control)
df_components = pd.concat(
    {
        "US": us_w,
        "EA": ea_w_usd,
    },
    axis=1
).sort_index()

if strict_mode:
    # require both to compute global
    global_usd = df_components.dropna().sum(axis=1)
else:
    global_usd = df_components.sum(axis=1, min_count=1)

global_usd.name = "GLOBAL_M2_USD"

# Trim to last valid
last_valid = global_usd.last_valid_index()
if last_valid is not None:
    global_usd = global_usd.loc[:last_valid]
    df_components = df_components.loc[:last_valid]

# Diagnostics
diag_df = pd.DataFrame(
    [
        {"series": "US", **validate_series_basic(df_components["US"])},
        {"series": "EA", **validate_series_basic(df_components["EA"])},
        {"series": "GLOBAL", **validate_series_basic(global_usd)},
    ]
).set_index("series")

# =========================
# Display
# =========================
c1, c2 = st.columns([2.2, 1.2], gap="large")

with c1:
    st.subheader("A) GLOBAL (USD) — Weekly (W-FRI), 2010+")
    st.line_chart(global_usd)

    st.subheader("B) Components (USD) — Weekly, 2010+")
    st.line_chart(df_components)

with c2:
    st.subheader("C) Diagnostics")
    st.dataframe(diag_df)

    st.subheader("D) Latest values (USD)")
    if len(global_usd) > 0:
        latest_dt = global_usd.index.max()
        st.write(f"Latest: {latest_dt.date()}")
        st.write(f"US: {df_components.loc[latest_dt, 'US']:.3e}")
        st.write(f"EA: {df_components.loc[latest_dt, 'EA']:.3e}")
        st.write(f"GLOBAL: {global_usd.loc[latest_dt]:.3e}")

# =========================
# One-click Export (ZIP)
# =========================
st.subheader("E) One-click Export (ZIP) — 2010+ only")

weekly_usd_wide = pd.concat([global_usd, df_components], axis=1)
weekly_usd_wide.columns = ["GLOBAL_M2_USD", "US_USD", "EA_USD"]

monthly_m2_local_wide = pd.DataFrame(
    {
        "US_M2_USD_monthly": us_m2,      # already USD monthly
        "EA_M2_EUR_monthly": ea_m2_eur,  # EUR monthly
    }
).sort_index()

monthly_fx_to_usd_wide = pd.DataFrame(
    {
        "FX_EURUSD_monthly": eurusd,     # USD per EUR monthly
    }
).sort_index()

meta_obj = {
    "repo": "liq",
    "dataset": "US_EA_M2_2010plus",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "policy": {
        "US": {"aggregate": "M2", "source": "FRED M2SL", "unit": "USD (billions->*1e9)"},
        "EA": {"aggregate": "M2", "source": "ECB BSI M20", "unit": "EUR (assumed millions->*1e6)"},
        "FX": {"series": "EURUSD", "definition": "USD per EUR (ECB EXR M.USD.EUR.SP00.A)"},
        "clip_start": str(CLIP_START.date()),
        "weekly_rule": WEEK_RULE,
        "weekly_method": "last + ffill",
        "fx_alignment": "FX reindexed to M2 weekly index before multiplication",
    },
    "notes": [
        "weekly_usd_wide.csv is intended for BTC Lead-Lag Lab upload as liquidity input (2010+).",
    ],
}

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True).encode("utf-8-sig")

zip_buffer = BytesIO()
with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as z:
    z.writestr("weekly_usd_wide.csv", df_to_csv_bytes(weekly_usd_wide))
    z.writestr("monthly_m2_local_wide.csv", df_to_csv_bytes(monthly_m2_local_wide))
    z.writestr("monthly_fx_to_usd_wide.csv", df_to_csv_bytes(monthly_fx_to_usd_wide))
    z.writestr("diagnostics_summary.csv", df_to_csv_bytes(diag_df))
    z.writestr("meta.json", json.dumps(meta_obj, ensure_ascii=False, indent=2).encode("utf-8"))

zip_buffer.seek(0)
st.download_button(
    "Download ZIP (2010+ only)",
    data=zip_buffer,
    file_name="liq_us_ea_m2_2010plus_export.zip",
    mime="application/zip",
)