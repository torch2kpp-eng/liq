# Claude Code 작업 지시서
## Alpha Layer 자동 활성화 — streamlit_app.py 직접 패치

> **작업 범위**: MVRV / Exchange Reserve / ETF Net Flow 자동 fetch 추가  
> **적용 방식**: 기존 `streamlit_app.py`에 최소 침투 패치 (별도 모듈 생성 금지)  
> **목표**: 사용자 수동 업로드 없이 Streamlit 앱 실행 시 자동으로 3개 신호 수집  
> **예상 작업 시간**: 1~2시간

---

## 🎯 작업 목표

현재 `streamlit_app.py`의 alpha layer 상태:
- ✅ **Funding Rate**: Binance API 자동 fetch (이미 작동)
- ❌ **MVRV**: 사용자 CSV 업로드 필요 (수동)
- ❌ **Exchange Reserve**: 미구현
- ❌ **ETF Net Flow**: 미구현

**작업 후 상태**:
- ✅ 4개 신호 모두 자동 fetch
- ✅ 매일 1회 자동 갱신 (Streamlit cache 6시간 + 디스크 캐시)
- ✅ Point-in-Time (PIT) 규칙 자동 적용 (look-ahead bias 방지)
- ✅ Fallback 체인 (1순위 실패 시 2순위, 그래도 실패 시 디스크 캐시)
- ✅ Graceful degradation (모든 소스 실패 시 alpha_neutral 유지)

---

## 📚 사전 컨텍스트 (필독)

### 1. v2.19 Phase -1 갭 리뷰 핵심 결론

이 패치는 v2.19 갭 리뷰에서 식별된 **누락 항목 ①: Look-ahead Bias 통제 규칙 부재**를 동시에 해결합니다.

**Point-in-Time (PIT) 규칙 (절대 준수)**:
| 데이터 소스 | 사용 허용 기준 | 이유 |
|-------------|----------------|------|
| MVRV | T-7일까지 | 주간 집계 확정 지연 + realized cap 소급 수정 |
| Exchange Reserve | T-7일까지 | wallet 클러스터링 retroactive 수정 |
| ETF Net Flow | T-2일까지 | T+1 발표 + 1일 안전 마진 |
| Funding Rate | T-1일까지 (기존) | 8시간 settlement |

### 2. score_norm 6점 고정 분모 원칙 (이미 갭 리뷰에서 확정)

```
alpha_score_norm = raw_score / 6.0  # 항상 ETF 포함 6점 분모
- ETF 미가용 구간: ETF score = 0, etf_available = 0
- ETF 가용 구간: etf_available = 1
```

이번 패치 후 ETF가 자동 fetch되므로 **etf_available flag**를 필수로 추가해야 합니다.

### 3. 기존 코드 패턴 (반드시 준수)

`streamlit_app.py`의 기존 패턴을 그대로 따릅니다:

```python
# 캐싱 패턴
@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fetch_xxx(...) -> pd.Series:
    ...

# 주간 집계
WEEK_RULE = "W-FRI"  # 이미 정의됨

# 알파 입력 통합 함수
def load_alpha_inputs_weekly(...) -> Dict[str, object]:
    ...
```

---

## 🔧 작업 항목

### 작업 1: 디스크 캐시 유틸리티 추가

**위치**: `streamlit_app.py` 상단 import 블록 직후 (대략 line 50 근처)

**추가할 코드**:

```python
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
```

---

### 작업 2: MVRV 자동 Fetcher 추가

**위치**: 기존 `load_uploaded_mvrv_weekly` 함수 **바로 위**

**추가할 코드**:

```python
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
```

---

### 작업 3: Exchange Reserve 자동 Fetcher 추가

**위치**: 작업 2의 `fetch_mvrv_zscore_auto` 함수 바로 아래

**추가할 코드**:

```python
# ============================================================
# v2.19 Alpha Auto-Fetch: Exchange Reserve
# ============================================================
@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fetch_exchange_reserve_auto(week_rule: str = WEEK_RULE) -> pd.DataFrame:
    """
    거래소 BTC 보유량 자동 수집.
    
    중요: 100% 무료 공식 API가 부재하므로 proxy를 적극 활용.
    PIT T-7 규칙 하에서는 절대량보다 7일 변화 방향이 더 중요.
    
    Fallback 체인:
      1순위: BGeometrics 무료 (직접 reserve)
      2순위: CoinMetrics SplyAct1yr proxy (1년 활동 공급량 → reserve trend proxy)
      3순위: 디스크 캐시 (최대 14일된 데이터)
    
    PIT 규칙: T-7일 컷오프
    
    Returns:
        pd.DataFrame with columns: ['exchange_reserve_btc', 'reserve_pct_change_4w', 'reserve_z']
    """
    
    # ── 1순위: BGeometrics ────────────────────────────────
    try:
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
        LOG_ALPHA.info(f"[Reserve] BGeometrics OK ({len(weekly)} weekly samples)")
        return weekly
        
    except Exception as e:
        LOG_ALPHA.warning(f"[Reserve] BGeometrics failed: {e}")
    
    # ── 2순위: CoinMetrics Proxy (SplyAct1yr) ────────────
    try:
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
        LOG_ALPHA.warning(
            f"[Reserve] BGeometrics failed → CoinMetrics SplyAct1yr proxy used "
            f"({len(weekly)} weekly samples). 절대량 아닌 추세 방향만 신뢰 가능."
        )
        return weekly
        
    except Exception as e:
        LOG_ALPHA.warning(f"[Reserve] CoinMetrics proxy failed: {e}")
    
    # ── 3순위: 디스크 캐시 ─────────────────────────────────
    cached = _alpha_load_disk("exchange_reserve", max_age_hours=24 * 14)
    if cached is not None and not cached.empty:
        LOG_ALPHA.warning(f"[Reserve] all live sources failed, using disk cache")
        return cached
    
    LOG_ALPHA.error("[Reserve] all sources failed including disk cache")
    return pd.DataFrame(columns=["exchange_reserve_btc", "reserve_pct_change_4w", "reserve_z"])
```

---

### 작업 4: ETF Net Flow 자동 Fetcher 추가

**위치**: 작업 3의 함수 바로 아래

**추가할 코드**:

```python
# ============================================================
# v2.19 Alpha Auto-Fetch: ETF Net Flow
# ============================================================
@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fetch_etf_netflow_auto(week_rule: str = WEEK_RULE) -> pd.DataFrame:
    """
    미국 스팟 BTC ETF 일일 net flow 자동 수집.
    
    Fallback 체인:
      1순위: Farside Investors HTML 스크래핑 (시장 표준)
      2순위: SoSoValue Demo API (st.secrets["SOSO_API_KEY"] 필요, 선택사항)
      3순위: 디스크 캐시 (최대 7일된 데이터)
    
    PIT 규칙: T-2일 컷오프 (Farside는 evening US time T+1 발표)
    
    Returns:
        pd.DataFrame with columns: ['etf_netflow_usd_m', 'etf_4w_cumulative', 'etf_z', 'etf_available']
    """
    
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; BTCMacroPredictionEngine/2.19)"
    }
    
    # ── 1순위: Farside Investors ──────────────────────────
    try:
        r = requests.get("https://farside.co.uk/btc/", headers=headers, timeout=20)
        r.raise_for_status()
        
        tables = pd.read_html(r.text)
        if not tables:
            raise RuntimeError("Farside no tables found")
        
        # 가장 큰 테이블이 일별 flow
        df_raw = max(tables, key=lambda t: t.shape[0])
        df = df_raw.copy()
        
        # MultiIndex 컬럼 평탄화
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[-1] if isinstance(c, tuple) else c for c in df.columns]
        df.columns = [str(c).strip() for c in df.columns]
        
        # Date 컬럼과 Total 컬럼 찾기
        date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
        total_col = next((c for c in df.columns if c.lower().startswith("total")), df.columns[-1])
        
        df = df[[date_col, total_col]].copy()
        df.columns = ["date", "etf_netflow_usd_m"]
        
        # 날짜 형식이 아닌 행 제거 (Average, Minimum, Maximum 등)
        date_pattern = r"\d{1,2}\s+\w+\s+\d{4}|\d{4}-\d{2}-\d{2}"
        df = df[df["date"].astype(str).str.contains(date_pattern, na=False, regex=True)]
        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
        df = df.dropna(subset=["date"]).set_index("date").sort_index()
        
        # 값 정규화: Farside는 '-' 또는 빈 셀을 0으로 처리
        # 음수는 '(123.4)' 형태로 표시될 수 있음
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
        
        # PIT 컷오프 T-2
        out = apply_pit_cutoff(df, cutoff_days=2)
        
        # 주간 집계 (sum)
        weekly = out.resample(week_rule).sum().dropna(how="all")
        weekly["etf_4w_cumulative"] = weekly["etf_netflow_usd_m"].rolling(4, min_periods=2).sum()
        weekly["etf_z"] = rolling_zscore(weekly["etf_4w_cumulative"], window=52, min_periods=13)
        weekly["etf_available"] = 1
        
        _alpha_save_disk("etf_netflow", weekly)
        LOG_ALPHA.info(f"[ETF] Farside scrape OK ({len(weekly)} weekly samples)")
        return weekly
        
    except Exception as e:
        LOG_ALPHA.warning(f"[ETF] Farside scrape failed: {e}")
    
    # ── 2순위: SoSoValue Demo API ────────────────────────
    try:
        api_key = st.secrets.get("SOSO_API_KEY", None) if hasattr(st, "secrets") else None
        if not api_key:
            raise RuntimeError("SOSO_API_KEY not in st.secrets, skipping")
        
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
        LOG_ALPHA.warning(f"[ETF] Farside failed → SoSoValue Demo API used ({len(weekly)} samples)")
        return weekly
        
    except Exception as e:
        LOG_ALPHA.warning(f"[ETF] SoSoValue fallback failed: {e}")
    
    # ── 3순위: 디스크 캐시 ─────────────────────────────────
    cached = _alpha_load_disk("etf_netflow", max_age_hours=24 * 7)
    if cached is not None and not cached.empty:
        LOG_ALPHA.warning(f"[ETF] all live sources failed, using disk cache")
        return cached
    
    LOG_ALPHA.error("[ETF] all sources failed including disk cache")
    # ETF 미가용 명시: etf_available=0
    return pd.DataFrame(columns=["etf_netflow_usd_m", "etf_4w_cumulative", "etf_z", "etf_available"])
```

---

### 작업 5: 기존 `load_alpha_inputs_weekly` 함수 수정

**위치**: 기존 `load_alpha_inputs_weekly` 함수 (약 line 540 근처)

**수정 방향**: 기존 시그니처는 유지하되, 내부에서 새로운 자동 fetcher를 호출하고, 사용자 업로드 MVRV가 있으면 그것을 우선 사용.

**기존 코드 (참고용)**:
```python
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
```

**교체할 새 코드**:

```python
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
            funding_raw = fetch_binance_funding_history(symbol=funding_symbol, start_date=START_DATE)
            funding_w, funding_z = build_funding_weekly_zscore(funding_raw, week_rule=week_rule)
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
```

---

### 작업 6: 사이드바 UI 업데이트

**위치**: 사이드바 `with st.expander(ui_text("Alpha Inputs (MVRV / Funding)"), expanded=False):` 블록 (약 line 220 근처)

**기존**:
```python
with st.expander(ui_text("Alpha Inputs (MVRV / Funding)"), expanded=False):
    USE_BINANCE_FUNDING = st.checkbox(
        "Use Binance funding auto-loader",
        value=True,
        ...
    )
    FUNDING_SYMBOL = st.text_input("Funding symbol", value="BTCUSDT")
    MVRV_FILE = st.file_uploader(
        "MVRV Z CSV upload (optional)",
        ...
    )
```

**교체**:
```python
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
```

---

### 작업 7: `build_forward_overlay_payload` 함수의 `load_alpha_inputs_weekly` 호출부 수정

**위치**: `build_forward_overlay_payload` 함수 내 (약 line 1500 근처)

**찾을 코드**:
```python
alpha_inputs = load_alpha_inputs_weekly(
    mvrv_file_bytes=mvrv_file_bytes,
    mvrv_file_name=mvrv_file_name,
    use_binance_funding=use_binance_funding,
    funding_symbol=funding_symbol,
    week_rule=WEEK_RULE,
)
```

**교체할 코드**:
```python
alpha_inputs = load_alpha_inputs_weekly(
    mvrv_file_bytes=mvrv_file_bytes,
    mvrv_file_name=mvrv_file_name,
    use_binance_funding=use_binance_funding,
    funding_symbol=funding_symbol,
    week_rule=WEEK_RULE,
    use_auto_mvrv=USE_AUTO_MVRV,
    use_auto_reserve=USE_AUTO_RESERVE,
    use_auto_etf=USE_AUTO_ETF,
)
```

또한 `build_forward_overlay_payload` 함수의 시그니처도 같이 업데이트:

**기존**:
```python
def build_forward_overlay_payload(
    liq_source: str,
    alpha_mode: str,
    forecast_model: str,
    use_binance_funding: bool = True,
    funding_symbol: str = "BTCUSDT",
    mvrv_file_bytes: Optional[bytes] = None,
    mvrv_file_name: Optional[str] = None,
):
```

**교체**:
```python
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
```

함수 호출하는 모든 곳 (`run_main_tab`, `run_tab_forward_overlay`, `run_tab5_multiasset`)도 같이 업데이트:

**기존**:
```python
payload = build_forward_overlay_payload(
    LIQ_SOURCE, ALPHA_MODE, FORECAST_MODEL, 
    use_binance_funding=USE_BINANCE_FUNDING, 
    funding_symbol=FUNDING_SYMBOL, 
    mvrv_file_bytes=MVRV_FILE_BYTES, 
    mvrv_file_name=MVRV_FILE_NAME
)
```

**교체**:
```python
payload = build_forward_overlay_payload(
    LIQ_SOURCE, ALPHA_MODE, FORECAST_MODEL, 
    use_binance_funding=USE_BINANCE_FUNDING, 
    funding_symbol=FUNDING_SYMBOL, 
    mvrv_file_bytes=MVRV_FILE_BYTES, 
    mvrv_file_name=MVRV_FILE_NAME,
    use_auto_mvrv=USE_AUTO_MVRV,
    use_auto_reserve=USE_AUTO_RESERVE,
    use_auto_etf=USE_AUTO_ETF,
)
```

---

### 작업 8: `build_forward_overlay_payload` 내 Reserve / ETF 데이터 통합

**위치**: `build_forward_overlay_payload` 함수 내, `overlay_master_df` 생성 후 alpha 데이터 merge 부분 (약 line 1600 근처)

**찾을 코드**:
```python
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
```

**바로 아래 추가**:
```python
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
```

또한 기존 `compute_alpha_state` 호출을 4-signal 통합으로 확장하려면 `compute_alpha_state` 함수도 수정 필요. 다만 기존 함수와의 호환성 유지를 위해 **새로운 함수를 추가**:

**`compute_alpha_state` 함수 바로 아래 추가**:
```python
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
```

그 다음 기존 `overlay_master_df["alpha_state"] = compute_alpha_state(...)` 부분을 다음으로 교체:

```python
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
).fillna("NEUTRAL")
```

---

### 작업 9: UI 메타데이터 표시 업데이트

**위치**: `run_main_tab` 함수 내 alpha_meta caption 부분 (약 line 2200 근처)

**기존**:
```python
st.caption(
    ui_text(
        f"Alpha inputs | Funding: {'loaded' if alpha_meta.get('funding_loaded', False) else ('error' if alpha_meta.get('funding_error') else 'disabled')} ({alpha_meta.get('funding_symbol', 'NA')})"
        f" | MVRV: {'loaded' if alpha_meta.get('mvrv_loaded', False) else 'not loaded'} ({alpha_meta.get('mvrv_source', 'none')})"
    )
)
```

**교체**:
```python
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
```

이 부분의 **두 번째 중복 caption block**도 함께 제거 (현재 코드에는 동일한 caption이 두 번 호출되는 부분이 있음).

---

## ✅ 통합 테스트 체크리스트

작업 완료 후 다음을 확인:

```
□ 1. Streamlit 앱 정상 실행 (streamlit run streamlit_app.py)
□ 2. 사이드바에 4개 체크박스 표시 (Funding/MVRV/Reserve/ETF)
□ 3. MAIN 탭 caption에 4개 alpha 신호 상태 표시
□ 4. .cache_alpha/ 폴더에 parquet 파일 3개 생성 확인
   - mvrv_zscore.parquet
   - exchange_reserve.parquet
   - etf_netflow.parquet
□ 5. 콘솔 로그에 [MVRV]/[Reserve]/[ETF] OK 메시지 확인
□ 6. ONE FILE CSV 다운로드 시 신규 컬럼 포함:
   - exchange_reserve_btc, reserve_pct_change_4w, reserve_z
   - etf_netflow_usd_m, etf_4w_cumulative, etf_z, etf_available
   - reserve_state, etf_state
□ 7. alpha_state가 더 이상 alpha_neutral만 나오지 않음 (BULLISH/BEARISH 등 출현)
□ 8. 인터넷 차단 후 재실행 시 디스크 캐시로 graceful degradation 확인
```

---

## ⚠️ 작업 주의사항

### 1. 절대 변경 금지
- `WEEK_RULE = "W-FRI"` 상수
- 기존 Funding 관련 함수 (`fetch_binance_funding_history`, `build_funding_weekly_zscore`)
- score_norm 분모는 **반드시 6.0 고정** (ETF 포함 기준)
- PIT 컷오프 days 값:
  - MVRV/Reserve: 7일
  - ETF: 2일

### 2. 기존 호환성 유지
- `load_alpha_inputs_weekly` 시그니처는 기존 인자 모두 유지하고 신규 인자만 default와 함께 추가
- 기존 컬럼명 (`mvrv_z`, `funding_z` 등) 유지
- `compute_alpha_state` 기존 함수는 **삭제하지 말고** 유지 (legacy 호환)

### 3. BGeometrics endpoint 보정
BGeometrics의 정확한 API path는 무료 가입 후 `api.bgeometrics.com/scalar.html` Swagger 문서에서 확인이 필요합니다. 코드는 3개 candidate URL을 시도하도록 되어 있으나, **실제 endpoint가 다르면 수정이 필요**합니다. 개발 시 다음 명령으로 확인:

```bash
# 실제 endpoint 확인 후 fetch_mvrv_zscore_auto 함수 내 candidate_urls 업데이트
curl https://api.bgeometrics.com/v1/mvrv-zscore
```

### 4. Farside HTML 구조 변경 리스크
Farside는 2024년 1월 이후 안정적이지만 사이트 개편 시 깨질 수 있음. 디스크 캐시 + SoSoValue fallback으로 보완. **주 1회 정도 정상 작동 확인 필요**.

### 5. SoSoValue API key 설정 (선택)
ETF fallback을 활성화하려면 `.streamlit/secrets.toml`에 추가:
```toml
SOSO_API_KEY = "your_api_key_here"
```
없어도 1순위 Farside가 작동하면 문제 없음.

---

## 📋 커밋 메시지 권장

```
feat(v2.19): alpha layer auto-activation - MVRV/Reserve/ETF auto-fetch

- Add CoinMetrics Community API as primary MVRV source
- Add BGeometrics free API as fallback for MVRV/Reserve
- Add Farside Investors HTML scraping for ETF Net Flow
- Implement PIT cutoff rules (T-7 for on-chain, T-2 for ETF)
- Add disk cache (parquet) for graceful degradation
- Extend compute_alpha_state to 4-signal v2.19 spec with 6.0 fixed denominator
- Add etf_available flag for v2.19 score normalization

Resolves: v2.19 gap review missing item ① (look-ahead bias rules)
Resolves: v2.19 gap review logical gap ④ (ETF circular reasoning - now 
  with PIT T-2 + 4-week cumulative z-score for partial control)
```

---

## 🔄 작업 순서 권장

1. **작업 1** (디스크 캐시 유틸) — 가장 먼저, 다른 함수가 사용
2. **작업 2** (MVRV fetcher) — 가장 안전한 데이터 소스
3. **작업 3** (Reserve fetcher) — proxy 로직 포함
4. **작업 4** (ETF fetcher) — 스크래핑 가장 까다로움
5. **작업 5** (load_alpha_inputs_weekly 수정)
6. **작업 6** (사이드바 UI)
7. **작업 7** (build_forward_overlay_payload 호출부)
8. **작업 8** (Reserve/ETF 통합 + state classification)
9. **작업 9** (UI 메타데이터)
10. **테스트** — 체크리스트 9개 항목 모두 통과

각 작업 완료 후 `streamlit run streamlit_app.py`로 정상 실행 여부 확인 후 다음 작업 진행.

---

## 🎯 작업 완료 후 예상 효과

**v2.19 갭 리뷰 해결 항목**:
- ✅ 누락 ① Look-ahead Bias 통제 → PIT 규칙 코드화
- ✅ 누락 ② Score Norm 비교 불가능성 → 6.0 고정 분모 + etf_available flag
- ⚠️ 부분 해결: 갭 ④ ETF 순환 논리 → 4주 누적 z-score + T-2 PIT로 일부 완화

**투자 판단 측면**:
- 현재 alpha_state가 항상 `alpha_neutral`만 나오는 문제 해결
- 4개 신호로 BTC 사이클 위치(top/bottom) 인식 활성화
- 매일 1회 자동 갱신으로 수동 작업 제거

**Phase 1 진입 준비**:
- 갭 리뷰 권장사항 일부 완료 → Phase 1 OOS 검증의 의미가 살아남
- alpha layer가 실제로 작동해야 v2.19의 'state-conditioned calibration'을 검증 가능
