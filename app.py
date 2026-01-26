# app.py  (repo: liq)
import io
import json
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="liq - Global Liquidity (M2/M3) Validator", layout="wide")

# ------------------------
# Helpers
# ------------------------
def http_get(url, params=None, headers=None, timeout=30):
    r = requests.get(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r

def to_ts(df, date_col="date", value_col="value"):
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.sort_values(date_col).set_index(date_col)
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    return out[[value_col]]

# ------------------------
# Fetchers (stable ones)
# ------------------------
def fetch_fred_csv(series_id: str) -> pd.DataFrame:
    # FRED CSV direct download (no API key needed for this route in most cases)
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    df = pd.read_csv(io.StringIO(http_get(url, params={"id": series_id}).text))
    df.columns = ["date", "value"]
    return to_ts(df)

def fetch_ecb_sdmx(flow: str, key: str, start_period="2000-01") -> pd.DataFrame:
    # ECB SDMX REST; format=csvdata supported
    # Docs: https://data.ecb.europa.eu/help/api/data  (format=csvdata)
    url = f"https://data-api.ecb.europa.eu/service/data/{flow}/{key}"
    params = {"format": "csvdata", "startPeriod": start_period}
    text = http_get(url, params=params).text
    df = pd.read_csv(io.StringIO(text))
    # ECB CSV has TIME_PERIOD and OBS_VALUE columns
    df = df.rename(columns={"TIME_PERIOD": "date", "OBS_VALUE": "value"})
    return to_ts(df, "date", "value")

def fetch_boe_csv(series_code: str, start="2010-01-01", end="2025-12-31") -> pd.DataFrame:
    # BoE Database CSV export (tabular/columnar). We'll use "TT" tabular for simplicity.
    # Example pattern exists in BoE export pages.
    url = "https://www.bankofengland.co.uk/boeapps/database/fromshowcolumns.asp"
    params = {
        "CSVF": "TT",
        "DAT": "RNG",
        "FromSeries": 1,
        "ToSeries": 1,
        "SeriesCodes": series_code,
        "UsingCodes": "Y",
        "VPD": "Y",
        # date range
        "FD": pd.to_datetime(start).day,
        "FM": pd.to_datetime(start).strftime("%b"),
        "FY": pd.to_datetime(start).year,
        "TD": pd.to_datetime(end).day,
        "TM": pd.to_datetime(end).strftime("%b"),
        "TY": pd.to_datetime(end).year,
    }
    text = http_get(url, params=params).text
    # BoE CSV can include header metadata lines; robust parse:
    raw = pd.read_csv(io.StringIO(text), skiprows=0)
    # Heuristic: find first column named "Date" (case-insensitive)
    date_col = [c for c in raw.columns if str(c).lower() == "date"]
    if not date_col:
        # fallback: first column
        raw = raw.rename(columns={raw.columns[0]: "Date"})
        date_col = ["Date"]
    val_col = [c for c in raw.columns if c != date_col[0]][0]
    df = raw[[date_col[0], val_col]].rename(columns={date_col[0]: "date", val_col: "value"})
    return to_ts(df)

def fetch_boc_valet_series(series_name: str, start_date="2000-01-01") -> pd.DataFrame:
    # Bank of Canada Valet API (no key). CSV/JSON supported.
    # We'll use JSON for parsing stability.
    url = f"https://www.bankofcanada.ca/valet/observations/{series_name}/json"
    params = {"start_date": start_date}
    j = http_get(url, params=params).json()
    obs = j.get("observations", [])
    df = pd.DataFrame({
        "date": [o["d"] for o in obs],
        "value": [o.get(series_name, {}).get("v") for o in obs],
    })
    return to_ts(df)

def fetch_bcb_sgs(series_id: int) -> pd.DataFrame:
    # BCB SGS API (Brazil)
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_id}/dados"
    params = {"formato": "json"}
    j = http_get(url, params=params).json()
    df = pd.DataFrame(j).rename(columns={"data": "date", "valor": "value"})
    # BCB dates are dd/mm/YYYY
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
    return to_ts(df)

def fetch_rba_d3_m3() -> pd.DataFrame:
    # RBA D3 historical xlsx
    url = "https://www.rba.gov.au/statistics/tables/xls/d03hist.xlsx"
    content = http_get(url).content
    xls = pd.ExcelFile(io.BytesIO(content))
    # Sheet names vary. We'll try first sheet and infer columns.
    df0 = xls.parse(xls.sheet_names[0])
    # You will need to adjust depending on the exact table layout.
    # Minimal heuristic: find a "Date" column and an "M3" column.
    cols = {c: str(c).strip() for c in df0.columns}
    df0 = df0.rename(columns=cols)
    date_candidates = [c for c in df0.columns if "date" in c.lower() or "month" in c.lower()]
    m3_candidates = [c for c in df0.columns if c.strip().upper() == "M3" or "M3" == c.strip()]
    if not date_candidates or not m3_candidates:
        raise ValueError("RBA D3 parsing failed: please map Date/M3 columns explicitly.")
    df = df0[[date_candidates[0], m3_candidates[0]]].rename(columns={date_candidates[0]: "date", m3_candidates[0]: "value"})
    return to_ts(df)

# ------------------------
# UI
# ------------------------
st.title("liq — Global Liquidity (M2/M3) Data Validator")

with st.sidebar:
    st.header("Sources (stable first)")
    run = st.button("Fetch & Validate")

    st.caption("Note: KR(ECOS), JP(BOJ), CN(PBoC), IN(RBI), MX(Banxico) are staged for phase-2.")

# Config examples (you will finalize series codes)
CONFIG = {
    "US_M2_FRED": lambda: fetch_fred_csv("M2SL"),
    # ECB example key must be confirmed to the exact aggregate you want (M3 vs M2-like).
    # Put your intended key here after you finalize.
    "EA_M3_ECB": lambda: fetch_ecb_sdmx("BSI", "M.U2.Y.V.M30.X.1.U2.2300.Z01.E", start_period="2000-01"),
    # BoE: pick one "level" series code for M4 (example codes must be selected from BoE database)
    # Replace 'LPMVUBQ' with your chosen M4 level series code.
    "UK_M4_BOE": lambda: fetch_boe_csv("LPMVUBQ", start="2000-01-01", end="2025-12-31"),
    # SNB: API endpoint discovery needed from SNB "API links" on the chart page (phase-1.5)
    # BoC: set your seriesName after list-discovery (e.g., M2++ series code)
    # Brazil: series id to be validated; example placeholder 27842
    "BR_M2_BCB": lambda: fetch_bcb_sgs(27842),
    "AU_M3_RBA": lambda: fetch_rba_d3_m3(),
}

if run:
    results = {}
    errors = {}

    for name, fn in CONFIG.items():
        try:
            ts = fn()
            results[name] = ts
        except Exception as e:
            errors[name] = str(e)

    if errors:
        st.subheader("Errors")
        st.json(errors)

    if results:
        st.subheader("Validation Summary")
        rows = []
        for k, df in results.items():
            rows.append({
                "series": k,
                "start": df.index.min().date(),
                "end": df.index.max().date(),
                "n": int(df["value"].notna().sum()),
                "missing": int(df["value"].isna().sum()),
            })
        st.dataframe(pd.DataFrame(rows))

        st.subheader("Charts")
        for k, df in results.items():
            st.line_chart(df.rename(columns={"value": k}))

        # Simple alignment check (monthly -> weekly carry-forward is a policy choice)
        st.subheader("Aligned (weekly, carry-forward) — optional preview")
        aligned = pd.concat([v.rename(columns={"value": k}) for k, v in results.items()], axis=1).sort_index()
        weekly = aligned.resample("W-FRI").ffill()
        st.line_chart(weekly)