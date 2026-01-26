# streamlit_app.py
# liq | Global Liquidity Validator (Core12) + One-click ZIP export
#
# pip install streamlit pandas numpy requests python-dateutil
# streamlit run streamlit_app.py

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

AUTO_COUNTRIES = {"US", "EA", "CA"}

BOC_GROUP_DEFAULT = "ATABLE_MONETARY_AGGREGATES"


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
    date_col = None
    for c in date_candidates:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]

    value_col = None
    for c in value_candidates:
        if c in df.columns:
            value_col = c
            break
    if value_col is None:
        value_col = df.columns[-1]

    out = df[[date_col, value_col]].copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out = out.dropna(subset=[date_col]).set_index(date_col).sort_index()
    s = out[value_col].astype(float)
    return s


def to_weekly_last_ffill(s: pd.Series) -> pd.Series:
    if s is None or s.empty:
        return pd.Series(dtype=float)
    return s.sort_index().resample(WEEK_RULE).last().ffill()


# =========================
# FX via ECB (cross through EUR)
# =========================
@st.cache_data(show_spinner=False, ttl=6 * 60 * 60)
def fetch_ecb_exr_monthly(base: str, quote: str = "EUR") -> pd.Series:
    # key = M.{base}.{quote}.SP00.A
    key = f"M.{base}.{quote}.SP00.A"
    url = f"https://data-api.ecb.europa.eu/service/data/EXR/{key}"
    txt = http_get(url, params={"format": "csvdata"}).text
    s = parse_two_col_csv(txt, date_candidates=("TIME_PERIOD",), value_candidates=("OBS_VALUE",))
    s.name = f"EXR_{base}{quote}"
    return s


@st.cache_data(show_spinner=False, ttl=6 * 60 * 60)
def fx_to_usd_monthly(ccy: str) -> pd.Series:
    if ccy == "USD":
        idx = pd.date_range("1990-01-01", pd.Timestamp.today().normalize(), freq="MS")
        return pd.Series(1.0, index=idx, name="FX_USDUSD")

    eurusd = fetch_ecb_exr_monthly("USD", "EUR")  # USD per EUR
    eurccy = fetch_ecb_exr_monthly(ccy, "EUR")    # CCY per EUR
    df = pd.concat([eurusd, eurccy], axis=1).dropna()
    fx = df.iloc[:, 0] / df.iloc[:, 1]
    fx.name = f"FX_{ccy}USD"
    return fx


# =========================
# Auto: US M2 (FRED)
# =========================
@st.cache_data(show_spinner=False, ttl=6 * 60 * 60)
def fetch_us_m2_fred_usd() -> pd.Series:
    # FRED M2SL (Billions USD)
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    txt = http_get(url, params={"id": "M2SL"}).text
    df = pd.read_csv(io.StringIO(txt)).rename(columns={"DATE": "DATE", "M2SL": "VALUE"})
    s = parse_two_col_csv(df.to_csv(index=False), date_candidates=("DATE",), value_candidates=("VALUE",))
    # Billions -> USD
    s = s * 1e9
    s.name = "US_M2_USD"
    return s


# =========================
# Auto: EA Broad Money (ECB BSI)
#   Fix: strip "BSI." prefix in key (flow is already in URL)
# =========================
@st.cache_data(show_spinner=False, ttl=6 * 60 * 60)
def fetch_ea_bsi_monthly(series_key: str) -> pd.Series:
    """
    Use SDMX REST:
      /service/data/BSI/{key}?format=csvdata
    Important: key should NOT include leading "BSI." in this endpoint.
    """
    key = series_key
    if key.startswith("BSI."):
        key = key.replace("BSI.", "", 1)

    url = f"https://data-api.ecb.europa.eu/service/data/BSI/{key}"
    try:
        txt = http_get(url, params={"format": "csvdata"}).text
    except requests.HTTPError:
        # fallback: try with original (in case ECB changes behavior)
        url2 = f"https://data-api.ecb.europa.eu/service/data/BSI/{series_key}"
        txt = http_get(url2, params={"format": "csvdata"}).text

    s = parse_two_col_csv(txt, date_candidates=("TIME_PERIOD",), value_candidates=("OBS_VALUE",))
    s.name = f"EA_BSI_{series_key}"
    return s


# =========================
# Auto: Canada M2 & M2++ via BoC Valet group
#   Fix: show seriesDetail + allow user selection
# =========================
@st.cache_data(show_spinner=False, ttl=6 * 60 * 60)
def fetch_boc_group_json(group: str, start_date: str = "2000-01-01") -> dict:
    url = f"https://www.bankofcanada.ca/valet/observations/group/{group}/json"
    j = http_get(url, params={"start_date": start_date}).json()
    return j


def best_match_series_code(series_detail: Dict[str, dict], include: Tuple[str, ...], exclude: Tuple[str, ...]) -> Optional[str]:
    """
    Score by number of include tokens matched; return best.
    """
    inc = [t.lower() for t in include]
    exc = [t.lower() for t in exclude]

    best_code = None
    best_score = -1
    for code, meta in series_detail.items():
        label = str(meta.get("label", "")).lower()
        if any(t in label for t in exc):
            continue
        score = sum(1 for t in inc if t in label)
        if score > best_score:
            best_score = score
            best_code = code
    if best_score <= 0:
        return None
    return best_code


def boc_extract_two_series(j: dict, code_a: str, code_b: str) -> Tuple[pd.Series, pd.Series]:
    obs = j.get("observations", [])
    if not obs:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    rows = []
    for o in obs:
        d = o.get("d")
        va = o.get(code_a, {}).get("v")
        vb = o.get(code_b, {}).get("v")
        rows.append({"date": d, "A": va, "B": vb})

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["A"] = pd.to_numeric(df["A"], errors="coerce")
    df["B"] = pd.to_numeric(df["B"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()

    a = df["A"].astype(float)
    b = df["B"].astype(float)
    return a, b


# =========================
# Manual CSV loader
# =========================
def load_manual_m2(country: str, uploaded_file) -> pd.Series:
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
    st.header("Strictness")
    strict_mode = st.checkbox(
        "Strict: require ALL selected countries",
        value=False,
        help="OFF: missing manual uploads are excluded with warnings. ON: missing manual uploads cause failure.",
    )

    st.divider()
    st.header("EA (ECB BSI)")
    ea_use_m3 = st.checkbox("Use EA M3 (recommended)", value=True)
    st.caption("M3 key exists on ECB portal; unit is Millions of Euro.  [oai_citation:2‡data.ecb.europa.eu](https://data.ecb.europa.eu/data/datasets/BSI/BSI.M.U2.Y.V.M30.X.1.U2.2300.Z01.E?utm_source=chatgpt.com)")

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

# =========================
# Build monthly local + monthly fx
# =========================
m2_local_monthly: Dict[str, pd.Series] = {}
fx_monthly: Dict[str, pd.Series] = {}
meta_rows = []

# --- US (auto) ---
if "US" in selected:
    try:
        us = fetch_us_m2_fred_usd()
        m2_local_monthly["US"] = us  # already USD
        fx_monthly["US"] = fx_to_usd_monthly("USD")
        meta_rows.append({
            "country": "US", "ccy": "USD",
            "m2_source": "FRED M2SL", "m2_unit": "USD (converted from billions)",
            "fx_source": "USDUSD=1",
        })
    except Exception as e:
        errors.append(f"US auto fetch failed: {e}")

# --- EA (auto) ---
if "EA" in selected:
    try:
        if ea_use_m3:
            ea_key = "BSI.M.U2.Y.V.M30.X.1.U2.2300.Z01.E"
            ea_alias = "EA_M3"
        else:
            ea_key = "BSI.M.U2.Y.V.M20.X.1.U2.2300.Z01.E"
            ea_alias = "EA_M2"

        ea = fetch_ea_bsi_monthly(ea_key)  # Millions EUR typically
        ea_eur = ea * 1e6  # Millions -> EUR
        m2_local_monthly["EA"] = ea_eur
        fx_monthly["EA"] = fx_to_usd_monthly("EUR")
        meta_rows.append({
            "country": "EA", "ccy": "EUR",
            "m2_source": "ECB BSI SDMX",
            "m2_series": ea_key,
            "m2_unit": "EUR (converted from millions)",
            "fx_source": "ECB EXR cross via EUR",
            "alias": ea_alias,
        })
    except Exception as e:
        errors.append(f"EA auto fetch failed: {e}")

# --- CA (auto) ---
ca_series_table = pd.DataFrame()
ca_selected_code_m2 = None
ca_selected_code_m2pp = None
ca_meta = {}

if "CA" in selected:
    try:
        j = fetch_boc_group_json(boc_group, start_date="2000-01-01")
        series_detail = j.get("seriesDetail", {})
        if not series_detail:
            raise ValueError(f"No seriesDetail found in BoC group={boc_group}")

        # Build a table for UI inspection
        ca_series_table = pd.DataFrame(
            [{"code": code, "label": meta.get("label", ""), "units": meta.get("units", ""), "frequency": meta.get("frequency", "")}
             for code, meta in series_detail.items()]
        ).sort_values("label")

        # Auto-guess codes (robust tokens)
        guess_m2pp = best_match_series_code(series_detail, include=("m2++",), exclude=())
        # For M2: include "m2" but exclude m2+ and m2++
        guess_m2 = best_match_series_code(series_detail, include=("m2",), exclude=("m2+", "m2++"))

        # UI: allow explicit selection (guaranteed fix)
        st.sidebar.subheader("CA series mapping (from seriesDetail)")
        codes = ca_series_table["code"].tolist()
        ca_selected_code_m2 = st.sidebar.selectbox("CA M2 series code", options=codes, index=codes.index(guess_m2) if guess_m2 in codes else 0)
        ca_selected_code_m2pp = st.sidebar.selectbox("CA M2++ series code", options=codes, index=codes.index(guess_m2pp) if guess_m2pp in codes else min(1, len(codes)-1))

        a, b = boc_extract_two_series(j, ca_selected_code_m2, ca_selected_code_m2pp)
        a.name = "CA_M2_LOCAL"
        b.name = "CA_M2PP_LOCAL"

        # Apply multiplier
        a = a * float(ca_unit_multiplier)
        b = b * float(ca_unit_multiplier)

        m2_local_monthly["CA_M2"] = a
        m2_local_monthly["CA_M2PP"] = b
        fx_monthly["CA"] = fx_to_usd_monthly("CAD")

        ca_meta = {
            "group": boc_group,
            "chosen_m2_code": ca_selected_code_m2,
            "chosen_m2_label": str(series_detail.get(ca_selected_code_m2, {}).get("label", "")),
            "chosen_m2pp_code": ca_selected_code_m2pp,
            "chosen_m2pp_label": str(series_detail.get(ca_selected_code_m2pp, {}).get("label", "")),
            "unit_multiplier": ca_unit_multiplier,
        }
        meta_rows.append({
            "country": "CA", "ccy": "CAD",
            "m2_source": "BoC Valet group",
            "group": boc_group,
            "m2_code": ca_selected_code_m2,
            "m2pp_code": ca_selected_code_m2pp,
            "m2_unit_multiplier": ca_unit_multiplier,
            "fx_source": "ECB EXR cross via EUR",
        })

    except Exception as e:
        errors.append(f"CA auto fetch failed: {e}")

# --- Manual countries ---
for c in selected:
    if c in AUTO_COUNTRIES:
        continue
    try:
        f = uploads.get(c)
        if f is None:
            msg = f"Manual M2 missing for {c} (CSV not uploaded) -> excluded"
            if strict_mode:
                errors.append(msg.replace("-> excluded", ""))
            else:
                warnings.append(msg)
            continue

        s = load_manual_m2(c, f) * float(manual_scale)
        m2_local_monthly[c] = s
        fx_monthly[c] = fx_to_usd_monthly(CCY[c])
        meta_rows.append({
            "country": c, "ccy": CCY[c],
            "m2_source": "Manual CSV upload",
            "m2_unit_multiplier": manual_scale,
            "fx_source": "ECB EXR cross via EUR",
        })
    except Exception as e:
        errors.append(f"{c} manual load failed: {e}")

if warnings:
    st.warning("Warnings:\n- " + "\n- ".join(warnings))

if errors:
    st.error("Errors:\n- " + "\n- ".join(errors))
    st.stop()

# =========================
# Convert to USD weekly components
# =========================
fx_weekly = {k: to_weekly_last_ffill(v) for k, v in fx_monthly.items()}

usd_weekly = {}

# US
if "US" in m2_local_monthly:
    usd_weekly["US"] = to_weekly_last_ffill(m2_local_monthly["US"])

# EA (EUR->USD)
if "EA" in m2_local_monthly and "EA" in fx_weekly:
    usd_weekly["EA"] = to_weekly_last_ffill(m2_local_monthly["EA"]) * fx_weekly["EA"]

# CA (CAD->USD)
if "CA_M2" in m2_local_monthly and "CA" in fx_weekly:
    usd_weekly["CA_M2"] = to_weekly_last_ffill(m2_local_monthly["CA_M2"]) * fx_weekly["CA"]
if "CA_M2PP" in m2_local_monthly and "CA" in fx_weekly:
    usd_weekly["CA_M2PP"] = to_weekly_last_ffill(m2_local_monthly["CA_M2PP"]) * fx_weekly["CA"]

# Manual (LCY->USD)
for c in selected:
    if c in {"US", "EA", "CA"}:
        continue
    if c in m2_local_monthly and c in fx_weekly:
        usd_weekly[c] = to_weekly_last_ffill(m2_local_monthly[c]) * fx_weekly[c]

# =========================
# Build GLOBAL components (CA uses M2++ by default)
# =========================
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

# diagnostics
diag_rows = []
for k, s in usd_weekly.items():
    info = validate_series_basic(s)
    diag_rows.append({"series": k, **info})
diag_df = pd.DataFrame(diag_rows).set_index("series").sort_index()

# =========================
# Display
# =========================
c1, c2 = st.columns([2.2, 1.2], gap="large")

with c1:
    st.subheader("A) GLOBAL (USD) — Weekly (W-FRI)")
    st.line_chart(global_usd)

    st.subheader("B) Components used in GLOBAL (USD) — Weekly")
    st.line_chart(df_components)

    if "CA" in selected and (("CA_M2" in usd_weekly) or ("CA_M2PP" in usd_weekly)):
        st.subheader("C) Canada comparison (USD) — M2 vs M2++ (Weekly)")
        ca_cmp = pd.DataFrame(index=global_usd.index)
        if "CA_M2" in usd_weekly:
            ca_cmp["CA_M2_USD"] = usd_weekly["CA_M2"]
        if "CA_M2PP" in usd_weekly:
            ca_cmp["CA_M2PP_USD"] = usd_weekly["CA_M2PP"]
        st.line_chart(ca_cmp)

    if not ca_series_table.empty:
        st.subheader("D) Canada seriesDetail (for mapping audit)")
        st.dataframe(ca_series_table)

with c2:
    st.subheader("E) Diagnostics (weekly series)")
    st.dataframe(diag_df)

    if ca_meta:
        st.subheader("F) Canada mapping (chosen)")
        st.json(ca_meta)

# =========================
# One-click Export (ZIP)
# =========================
st.subheader("G) One-click Export (ALL data as ZIP)")

weekly_usd_wide = pd.concat([global_usd, df_components], axis=1).sort_index()

# monthly local wide (as stored)
monthly_local_wide = pd.DataFrame({k: v for k, v in m2_local_monthly.items()}).sort_index()

# monthly fx wide
monthly_fx_wide = pd.DataFrame({k: v for k, v in fx_monthly.items()}).sort_index()

meta_obj = {
    "repo": "liq",
    "dataset": "global_liquidity_export",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "frequency_policy": {
        "m2_raw": "monthly (mixed sources)",
        "fx_raw": "monthly ECB EXR cross via EUR",
        "output": f"weekly {WEEK_RULE} last + ffill",
    },
    "selected_countries": selected,
    "strict_mode": strict_mode,
    "ea": {
        "use_m3": ea_use_m3,
        "series_key": "BSI.M.U2.Y.V.M30.X.1.U2.2300.Z01.E" if ea_use_m3 else "BSI.M.U2.Y.V.M20.X.1.U2.2300.Z01.E",
        "unit_note": "ECB BSI unit is Millions of Euro on portal; converted by *1e6.  [oai_citation:3‡data.ecb.europa.eu](https://data.ecb.europa.eu/data/datasets/BSI/BSI.M.U2.Y.V.M30.X.1.U2.2300.Z01.E?utm_source=chatgpt.com)",
    },
    "canada": {
        "group": boc_group,
        "use_m2pp_in_global": ca_use_m2pp_in_global,
        "unit_multiplier": ca_unit_multiplier,
        "mapping": ca_meta,
    },
    "series_meta_rows": meta_rows,
    "notes": [
        "weekly_usd_wide.csv is intended for BTC Lead-Lag Lab upload as liquidity input.",
        "monthly_m2_local_wide.csv and monthly_fx_to_usd_wide.csv support reconciliation/audit.",
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

st.caption(
    "워크플로우: ZIP 다운로드 → 나에게 업로드(정합성 검증) → 검증 완료 후 weekly_usd_wide.csv를 BTC Lead-Lag Lab에 업로드."
)