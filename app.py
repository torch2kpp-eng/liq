# streamlit_app.py
# liq | Global Liquidity Validator (Core12) + One-click ZIP export

import io
import json
import zipfile
from io import BytesIO
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

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

AUTO_COUNTRIES = {"US", "EA", "CA"}
BOC_GROUP_DEFAULT = "ATABLE_MONETARY_AGGREGATES"

# -------------------------
# HTTP helpers
# -------------------------
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

def to_weekly_last_ffill(s: pd.Series) -> pd.Series:
    if s is None or s.empty:
        return pd.Series(dtype=float)
    return s.sort_index().resample(WEEK_RULE).last().ffill()

# -------------------------
# FX via ECB
# -------------------------
@st.cache_data(show_spinner=False, ttl=6 * 60 * 60)
def fetch_ecb_exr_monthly(base: str, quote: str = "EUR") -> pd.Series:
    key = f"M.{base}.{quote}.SP00.A"
    url = f"https://data-api.ecb.europa.eu/service/data/EXR/{key}"
    txt = http_get(url, params={"format": "csvdata"}).text
    s = parse_two_col_csv(txt, date_candidates=("TIME_PERIOD",), value_candidates=("OBS_VALUE",))
    s.name = f"EXR_{base}{quote}"
    return s

@st.cache_data(show_spinner=False, ttl=6 * 60 * 60)
def fx_to_usd_monthly(ccy: str) -> pd.Series:
    """
    Return FX_{ccy}USD (USD per 1 CCY).
    - USD: 1
    - EUR: EURUSD directly (USD per EUR) from ECB: M.USD.EUR.SP00.A
    - Others: cross via EUR: (EURUSD)/(EURLCY)
    """
    if ccy == "USD":
        idx = pd.date_range("1990-01-01", pd.Timestamp.today().normalize(), freq="MS")
        return pd.Series(1.0, index=idx, name="FX_USDUSD")

    if ccy == "EUR":
        eurusd = fetch_ecb_exr_monthly("USD", "EUR")  # USD per EUR
        eurusd.name = "FX_EURUSD"
        return eurusd

    eurusd = fetch_ecb_exr_monthly("USD", "EUR")  # USD per EUR
    eurccy = fetch_ecb_exr_monthly(ccy, "EUR")    # CCY per EUR
    df = pd.concat([eurusd, eurccy], axis=1).dropna()
    fx = df.iloc[:, 0] / df.iloc[:, 1]
    fx.name = f"FX_{ccy}USD"
    return fx

# -------------------------
# Auto: US M2 (FRED)
# -------------------------
@st.cache_data(show_spinner=False, ttl=6 * 60 * 60)
def fetch_us_m2_fred_usd() -> pd.Series:
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    txt = http_get(url, params={"id": "M2SL"}).text
    df = pd.read_csv(io.StringIO(txt)).rename(columns={"DATE": "DATE", "M2SL": "VALUE"})
    s = parse_two_col_csv(df.to_csv(index=False), date_candidates=("DATE",), value_candidates=("VALUE",))
    s = s * 1e9
    s.name = "US_M2_USD"
    return s

# -------------------------
# Auto: EA BSI (ECB) - remove BSI. prefix in key
# -------------------------
@st.cache_data(show_spinner=False, ttl=6 * 60 * 60)
def fetch_ea_bsi_monthly(series_key: str) -> pd.Series:
    key = series_key
    if key.startswith("BSI."):
        key = key.replace("BSI.", "", 1)
    url = f"https://data-api.ecb.europa.eu/service/data/BSI/{key}"
    txt = http_get(url, params={"format": "csvdata"}).text
    s = parse_two_col_csv(txt, date_candidates=("TIME_PERIOD",), value_candidates=("OBS_VALUE",))
    s.name = f"EA_BSI_{series_key}"
    return s

# -------------------------
# BoC Valet group + mapping
# -------------------------
@st.cache_data(show_spinner=False, ttl=6 * 60 * 60)
def fetch_boc_group_json(group: str, start_date: str = "2000-01-01") -> dict:
    url = f"https://www.bankofcanada.ca/valet/observations/group/{group}/json"
    return http_get(url, params={"start_date": start_date}).json()

def best_match_series_code(series_detail: Dict[str, dict], include: Tuple[str, ...], exclude: Tuple[str, ...]) -> Optional[str]:
    inc = [t.lower() for t in include]
    exc = [t.lower() for t in exclude]
    best_code, best_score = None, -1
    for code, meta in series_detail.items():
        label = str(meta.get("label", "")).lower()
        if any(t in label for t in exc):
            continue
        score = sum(1 for t in inc if t in label)
        if score > best_score:
            best_score, best_code = score, code
    return best_code if best_score > 0 else None

def boc_extract_two_series(j: dict, code_a: str, code_b: str) -> Tuple[pd.Series, pd.Series]:
    obs = j.get("observations", [])
    rows = []
    for o in obs:
        rows.append({
            "date": o.get("d"),
            "A": o.get(code_a, {}).get("v"),
            "B": o.get(code_b, {}).get("v"),
        })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["A"] = pd.to_numeric(df["A"], errors="coerce")
    df["B"] = pd.to_numeric(df["B"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    return df["A"].astype(float), df["B"].astype(float)

# -------------------------
# Manual CSV loader
# -------------------------
def load_manual_m2(country: str, uploaded_file) -> pd.Series:
    df = pd.read_csv(uploaded_file)
    cols = {c.lower(): c for c in df.columns}
    if "date" not in cols or "value" not in cols:
        raise ValueError(f"[{country}] CSV must have columns: date,value")
    df[cols["date"]] = pd.to_datetime(df[cols["date"]], errors="coerce")
    df[cols["value"]] = pd.to_numeric(df[cols["value"]], errors="coerce")
    df = df.dropna(subset=[cols["date"]]).set_index(cols["date"]).sort_index()
    s = df[cols["value"]].astype(float)
    s.name = f"{country}_M2_LOCAL"
    return s

def validate_series_basic(s: pd.Series) -> dict:
    if s is None or s.empty:
        return {"n": 0, "start": None, "end": None, "missing_ratio": np.nan, "nonpositive_ratio": np.nan, "has_gaps_gt_45d": False}
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
# UI
# =========================
st.title("liq | Global Liquidity Validator (Core12)")

with st.sidebar:
    st.header("Scope")
    selected = st.multiselect("Countries", CORE12, default=["US", "EA", "CA"])

    st.divider()
    strict_mode = st.checkbox("Strict: require ALL selected countries", value=False)

    st.divider()
    st.header("EA (ECB BSI)")
    ea_use_m3 = st.checkbox("Use EA M3 (recommended)", value=True)

    st.divider()
    st.header("Canada (BoC Valet)")
    boc_group = st.text_input("BoC group", value=BOC_GROUP_DEFAULT)
    ca_use_m2pp_in_global = st.checkbox("Use CA M2++ in GLOBAL sum", value=True)
    ca_unit_multiplier = st.selectbox("CA unit multiplier (local)", options=[1.0, 1e3, 1e6, 1e9], index=2)

    st.divider()
    st.header("Manual uploads (non-auto)")
    manual_scale = st.selectbox("Manual M2 unit multiplier (local)", options=[1.0, 1e3, 1e6, 1e9], index=0)
    uploads = {}
    for c in selected:
        if c in AUTO_COUNTRIES:
            continue
        uploads[c] = st.file_uploader(f"{c} M2 CSV (date,value)", type=["csv"], key=f"up_{c}")

    st.divider()
    run_btn = st.button("Run / Refresh")

if not run_btn:
    st.info("왼쪽에서 설정 후 Run / Refresh를 누르세요.")
    st.stop()

warnings = []
errors = []
m2_local_monthly: Dict[str, pd.Series] = {}
fx_monthly: Dict[str, pd.Series] = {}
meta_rows = []

# US
if "US" in selected:
    try:
        m2_local_monthly["US"] = fetch_us_m2_fred_usd()
        fx_monthly["US"] = fx_to_usd_monthly("USD")
        meta_rows.append({"country": "US", "m2_source": "FRED M2SL", "ccy": "USD"})
    except Exception as e:
        errors.append(f"US auto fetch failed: {e}")

# EA
if "EA" in selected:
    try:
        if ea_use_m3:
            ea_key = "BSI.M.U2.Y.V.M30.X.1.U2.2300.Z01.E"
            alias = "EA_M3"
        else:
            ea_key = "BSI.M.U2.Y.V.M20.X.1.U2.2300.Z01.E"
            alias = "EA_M2"

        ea = fetch_ea_bsi_monthly(ea_key) * 1e6  # Millions EUR -> EUR
        m2_local_monthly["EA"] = ea
        fx_monthly["EA"] = fx_to_usd_monthly("EUR")  # now fixed
        meta_rows.append({"country": "EA", "m2_source": "ECB BSI", "ccy": "EUR", "series": ea_key, "alias": alias})
    except Exception as e:
        errors.append(f"EA auto fetch failed: {e}")

# CA
ca_series_table = pd.DataFrame()
ca_meta = {}
if "CA" in selected:
    try:
        j = fetch_boc_group_json(boc_group, start_date="2000-01-01")
        series_detail = j.get("seriesDetail", {})
        if not series_detail:
            raise ValueError(f"No seriesDetail in group={boc_group}")

        ca_series_table = pd.DataFrame(
            [{"code": code, "label": meta.get("label", ""), "units": meta.get("units", ""), "frequency": meta.get("frequency", "")}
             for code, meta in series_detail.items()]
        ).sort_values("label")

        guess_m2pp = best_match_series_code(series_detail, include=("m2++",), exclude=())
        guess_m2 = best_match_series_code(series_detail, include=("m2",), exclude=("m2+", "m2++"))

        codes = ca_series_table["code"].tolist()
        st.sidebar.subheader("CA series mapping")
        ca_m2_code = st.sidebar.selectbox("CA M2 series code", options=codes, index=codes.index(guess_m2) if guess_m2 in codes else 0)
        ca_m2pp_code = st.sidebar.selectbox("CA M2++ series code", options=codes, index=codes.index(guess_m2pp) if guess_m2pp in codes else min(1, len(codes)-1))

        m2, m2pp = boc_extract_two_series(j, ca_m2_code, ca_m2pp_code)
        m2 = m2 * float(ca_unit_multiplier)
        m2pp = m2pp * float(ca_unit_multiplier)
        m2.name = "CA_M2_LOCAL"
        m2pp.name = "CA_M2PP_LOCAL"

        m2_local_monthly["CA_M2"] = m2
        m2_local_monthly["CA_M2PP"] = m2pp
        fx_monthly["CA"] = fx_to_usd_monthly("CAD")

        ca_meta = {
            "group": boc_group,
            "m2_code": ca_m2_code,
            "m2_label": series_detail.get(ca_m2_code, {}).get("label", ""),
            "m2pp_code": ca_m2pp_code,
            "m2pp_label": series_detail.get(ca_m2pp_code, {}).get("label", ""),
            "unit_multiplier": ca_unit_multiplier,
        }
        meta_rows.append({"country": "CA", "m2_source": "BoC Valet", "ccy": "CAD", **ca_meta})
    except Exception as e:
        errors.append(f"CA auto fetch failed: {e}")

# Manual
for c in selected:
    if c in AUTO_COUNTRIES:
        continue
    f = uploads.get(c)
    if f is None:
        msg = f"Manual M2 missing for {c}"
        if strict_mode:
            errors.append(msg)
        else:
            warnings.append(msg + " -> excluded")
        continue
    try:
        m2_local_monthly[c] = load_manual_m2(c, f) * float(manual_scale)
        fx_monthly[c] = fx_to_usd_monthly(CCY[c])
        meta_rows.append({"country": c, "m2_source": "Manual CSV", "ccy": CCY[c], "manual_multiplier": manual_scale})
    except Exception as e:
        errors.append(f"{c} manual load failed: {e}")

if warnings:
    st.warning("Warnings:\n- " + "\n- ".join(warnings))
if errors:
    st.error("Errors:\n- " + "\n- ".join(errors))
    st.stop()

# Convert to USD weekly
fx_weekly = {k: to_weekly_last_ffill(v) for k, v in fx_monthly.items()}
usd_weekly = {}

if "US" in m2_local_monthly:
    usd_weekly["US"] = to_weekly_last_ffill(m2_local_monthly["US"])
if "EA" in m2_local_monthly:
    usd_weekly["EA"] = to_weekly_last_ffill(m2_local_monthly["EA"]) * fx_weekly["EA"]
if "CA_M2" in m2_local_monthly and "CA" in fx_weekly:
    usd_weekly["CA_M2"] = to_weekly_last_ffill(m2_local_monthly["CA_M2"]) * fx_weekly["CA"]
if "CA_M2PP" in m2_local_monthly and "CA" in fx_weekly:
    usd_weekly["CA_M2PP"] = to_weekly_last_ffill(m2_local_monthly["CA_M2PP"]) * fx_weekly["CA"]

for c in selected:
    if c in {"US", "EA", "CA"}:
        continue
    if c in m2_local_monthly:
        usd_weekly[c] = to_weekly_last_ffill(m2_local_monthly[c]) * fx_weekly[c]

# GLOBAL components
components = {}
if "US" in usd_weekly:
    components["US"] = usd_weekly["US"]
if "EA" in usd_weekly:
    components["EA"] = usd_weekly["EA"]
if "CA" in selected:
    if ca_use_m2pp_in_global and "CA_M2PP" in usd_weekly:
        components["CA"] = usd_weekly["CA_M2PP"]
    elif (not ca_use_m2pp_in_global) and "CA_M2" in usd_weekly:
        components["CA"] = usd_weekly["CA_M2"]

for c in selected:
    if c in {"US", "EA", "CA"}:
        continue
    if c in usd_weekly:
        components[c] = usd_weekly[c]

df_components = pd.concat(components.values(), axis=1).sort_index() if components else pd.DataFrame()
if not df_components.empty:
    df_components.columns = list(components.keys())

global_usd = df_components.sum(axis=1, min_count=len(df_components.columns) if strict_mode else 1)
global_usd.name = "GLOBAL_M2_USD"

# Diagnostics
diag_df = pd.DataFrame([{**{"series": k}, **validate_series_basic(s)} for k, s in usd_weekly.items()]).set_index("series").sort_index()

# Display
c1, c2 = st.columns([2.2, 1.2], gap="large")
with c1:
    st.subheader("GLOBAL (USD) — Weekly (W-FRI)")
    st.line_chart(global_usd)
    st.subheader("Components used in GLOBAL (USD) — Weekly")
    st.line_chart(df_components)
    if "CA" in selected and (("CA_M2" in usd_weekly) or ("CA_M2PP" in usd_weekly)):
        st.subheader("Canada (USD) — M2 vs M2++")
        ca_cmp = pd.DataFrame(index=global_usd.index)
        if "CA_M2" in usd_weekly:
            ca_cmp["CA_M2_USD"] = usd_weekly["CA_M2"]
        if "CA_M2PP" in usd_weekly:
            ca_cmp["CA_M2PP_USD"] = usd_weekly["CA_M2PP"]
        st.line_chart(ca_cmp)
    if not ca_series_table.empty:
        st.subheader("Canada seriesDetail (audit)")
        st.dataframe(ca_series_table)
with c2:
    st.subheader("Diagnostics")
    st.dataframe(diag_df)
    if ca_meta:
        st.subheader("Canada mapping (chosen)")
        st.json(ca_meta)

# One-click ZIP export
st.subheader("One-click Export (ALL data as ZIP)")
weekly_usd_wide = pd.concat([global_usd, df_components], axis=1).sort_index()
monthly_local_wide = pd.DataFrame({k: v for k, v in m2_local_monthly.items()}).sort_index()
monthly_fx_wide = pd.DataFrame({k: v for k, v in fx_monthly.items()}).sort_index()

meta_obj = {
    "repo": "liq",
    "dataset": "global_liquidity_export",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "frequency_policy": {"output": f"weekly {WEEK_RULE} last + ffill"},
    "selected_countries": selected,
    "strict_mode": strict_mode,
    "ea_use_m3": ea_use_m3,
    "canada_use_m2pp_in_global": ca_use_m2pp_in_global,
    "meta_rows": meta_rows,
    "notes": [
        "weekly_usd_wide.csv is intended for BTC Lead-Lag Lab upload as liquidity input.",
        "EUR special-case: FX_EURUSD is directly fetched as EXR M.USD.EUR.SP00.A (USD per EUR).",
    ],
}

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True).encode("utf-8-sig")

zip_buffer = BytesIO()
with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as z:
    z.writestr("weekly_usd_wide.csv", df_to_csv_bytes(weekly_usd_wide))
    z.writestr("monthly_m2_local_wide.csv", df_to_csv_bytes(monthly_local_wide))
    z.writestr("monthly_fx_to_usd_wide.csv", df_to_csv_bytes(monthly_fx_wide))
    z.writestr("diagnostics_summary.csv", df_to_csv_bytes(diag_df))
    z.writestr("meta.json", json.dumps(meta_obj, ensure_ascii=False, indent=2).encode("utf-8"))

zip_buffer.seek(0)
st.download_button(
    "Download ALL as ZIP (weekly+monthly+fx+diagnostics+meta)",
    data=zip_buffer,
    file_name="liq_global_liquidity_export.zip",
    mime="application/zip",
)