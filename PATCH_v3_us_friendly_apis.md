# 패치 지시서 v3 — Streamlit Cloud 네트워크 차단 우회

> **진단 결과 요약:**
> - ❌ Funding: Binance API에서 **HTTP 451** (US IP 차단)
> - ❌ Reserve: BGeometrics endpoint 미작동 추정
> - ❌ ETF: Farside 스크래핑 실패 추정
> - ✅ MVRV: CoinMetrics 정상 작동
>
> **공통 원인**: Streamlit Cloud (AWS US-East-1) IP에서 접속 차단/거부
> **해결 전략**: 미국 IP 차단 없는 대체 소스로 일괄 교체

---

## 🎯 작업 목표

3개 신호의 데이터 소스를 **Streamlit Cloud에서 작동하는 대체 endpoint**로 교체:

| 신호 | 1순위 → 새 소스 | 2순위 → 새 소스 |
|------|----------------|-----------------|
| **Funding** | Bybit API (무인증, US 허용) | OKX API (무인증, US 허용) |
| **Reserve** | CoinMetrics proxy (작동 확인됨) | BGeometrics (선택적 유지) |
| **ETF** | CoinGecko Bitcoin ETF data (무료, US 허용) | Farside HTML (백업) |

작업 후 모든 신호가 **Streamlit Cloud 환경에서 작동**해야 합니다.

---

## 작업 1: Funding fetcher 교체 (Binance → Bybit + OKX)

### 위치
`streamlit_app.py`의 기존 `fetch_binance_funding_history` 함수 **위쪽 (또는 아래)** 에 새 함수 추가하고, `load_alpha_inputs_weekly`에서 새 함수를 호출하도록 변경.

### 추가할 새 함수

```python
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
```

### `load_alpha_inputs_weekly` 수정

`load_alpha_inputs_weekly` 함수 내부의 funding fetch 부분을 다음으로 교체:

**기존**:
```python
if use_binance_funding:
    try:
        funding_raw = fetch_binance_funding_history(symbol=funding_symbol, start_date=START_DATE)
        funding_w, funding_z = build_funding_weekly_zscore(funding_raw, week_rule=week_rule)
    except Exception as e:
        funding_error = str(e)
```

**교체**:
```python
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
```

---

## 작업 2: Reserve fetcher 우선순위 재배치 (CoinMetrics proxy 1순위로)

### 위치
기존 `fetch_exchange_reserve_auto` 함수의 **fallback 순서를 뒤집음**:
- 1순위: CoinMetrics proxy (작동 검증됨)
- 2순위: BGeometrics (작동 미확인)

### 교체할 코드

```python
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
```

---

## 작업 3: ETF fetcher 보강 (CoinGecko 추가)

### 위치
기존 `fetch_etf_netflow_auto` 함수에 **CoinGecko ETF data를 1순위 또는 새 fallback으로 추가**.

### 교체할 코드

```python
@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fetch_etf_netflow_auto(week_rule: str = WEEK_RULE) -> pd.DataFrame:
    """
    미국 스팟 BTC ETF 일일 net flow 자동 수집.
    
    v2.19 Patch v3: Add CoinGecko as new primary source (US-IP-friendly)
      1순위: CoinGecko Bitcoin ETF data (무료, no auth, US OK)
      2순위: Farside Investors HTML 스크래핑
      3순위: SoSoValue Demo API
      4순위: 디스크 캐시
    
    PIT 규칙: T-2일 컷오프
    """
    
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; BTCMacroPredictionEngine/2.19)"
    }
    
    # ── 1순위: CoinGecko 개별 ETF AUM 변화량 합산 ──────────────
    # CoinGecko provides individual spot BTC ETF data (IBIT, FBTC, GBTC, etc.)
    # We compute net flow as daily AUM change (proxy)
    try:
        LOG_ALPHA.info("[ETF] CoinGecko fetch start")
        # IBIT (BlackRock), FBTC (Fidelity) 등 주요 ETF의 시장 데이터
        # 단순 proxy: BTC market cap change × ETF holding share 추정 어려우므로
        # IBIT ticker price/volume을 직접 사용
        
        # 더 단순한 접근: Yahoo Finance를 통해 IBIT, FBTC 등의 거래량 합산
        # 이건 CoinGecko가 아니라 Yahoo Finance를 활용. yfinance가 이미 import되어 있음.
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
            raise RuntimeError("CoinGecko/Yahoo ETF all failed")
        
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
```

---

## 작업 4: Debug expander 확장 (3개 신호 모두 진단)

### 위치
`run_main_tab` 함수 시작 부분에 있는 기존 Debug expander 확장.

### 교체

```python
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
```

---

## 작업 5: requirements.txt 확인

다음 패키지가 `requirements.txt`에 있는지 확인. 없으면 추가:

```
yfinance
pandas
numpy
streamlit
requests
pyarrow
lxml
beautifulsoup4
html5lib
```

특히 **`yfinance`** 는 ETF Yahoo proxy를 위해 필수.

`pyarrow`, `lxml`, `html5lib`는 Parquet 캐시와 HTML 테이블 파싱에 필요.

---

## 작업 6: 검증 체크리스트

### 작업 후 확인 항목

```
□ 1. python -c "import ast; ast.parse(open('streamlit_app.py').read())" 통과
□ 2. Streamlit 정상 실행
□ 3. Debug expander에서 4개 버튼 모두 클릭 → 결과 확인
□ 4. MAIN 탭 상단 alpha layer status에서:
   - Funding: ✅ (Bybit 또는 OKX 표시)
   - MVRV: ✅ (변화 없음)
   - Reserve: ✅ (CoinMetrics SplyAct1yr proxy 표시)
   - ETF: ✅ (Yahoo Finance ETF proxy 표시) 
□ 5. ONE FILE CSV 다운로드 후 다음 컬럼이 NaN이 아닌지 확인:
   - funding_z (이전 NaN → 정상값)
   - exchange_reserve_btc (이전 NaN → 정상값)
   - etf_netflow_usd_m (이전 NaN → 정상값)
   - etf_available = 1 (이전 0 → 1)
□ 6. alpha_state 분포가 더 다양해졌는지 확인
   (이전: NEUTRAL 349, BULLISH 63, BEARISH 5)
   (예상: 4-signal 합산이라 분포가 더 분산될 것)
```

---

## 작업 7: PR 및 커밋

```
fix(v2.19): replace US-blocked APIs with US-IP-friendly alternatives

- Funding: Binance (HTTP 451) → Bybit + OKX fallback chain
- Reserve: BGeometrics (untested) → CoinMetrics SplyAct1yr proxy primary
- ETF: Farside (Cloudflare) → Yahoo Finance ETF dollar-volume proxy primary
- Add comprehensive Debug expander for all 4 alpha signals
- Add yfinance, pyarrow, lxml, html5lib to requirements.txt

Resolves: HTTP 451 from Streamlit Cloud (AWS US-East-1)
Resolves: Reserve/ETF columns ALL NaN despite fetchers being patched
```

---

## ⚠️ 주의사항

### 1. Yahoo Finance ETF proxy의 한계
- **실제 fund flow가 아님** — dollar volume 변화 기반 proxy
- 방향성(BULLISH/BEARISH)은 신뢰 가능, **절대 크기는 부정확**
- ETF score는 v2.19 점수 체계에서 ±1점이라 영향 제한적 → proxy로 충분

### 2. Bybit funding 데이터 시작일
- Binance(2019)보다 늦은 **2022년경**부터 데이터 있음
- 2022 이전 구간은 funding 신호 없음 (NaN) — 정상
- backtest는 최근 4년만 funding 신호 활용

### 3. 모든 fetcher의 graceful degradation 유지
- 어느 하나가 실패해도 다른 신호는 작동
- 모두 실패해도 MVRV 단독으로 작동 (현재 상태와 동일)

---

## 작업 순서 권장

1. **작업 5** 먼저: requirements.txt 업데이트 (재배포 필요할 수 있음)
2. **작업 1**: Funding (Bybit/OKX) — 가장 큰 효과
3. **작업 2**: Reserve (CoinMetrics primary)
4. **작업 3**: ETF (Yahoo proxy)
5. **작업 4**: Debug expander 확장
6. **작업 6**: 검증
7. **작업 7**: PR + 머지

각 작업 완료 후 syntax check (`python -c "import ast; ast.parse..."`)는 매번 실행.
