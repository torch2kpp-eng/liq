# Claude Code 패치 지시서 — Patch A + Patch B

**작성일**: 2026-05-02
**대상**: torch2kpp-eng/liq (app.py, branch main)
**현재 버전**: v2.18.5-regimefix-fundingdiag (4,747 lines)
**목표 버전**: v2.19.1-pathsafe-g3liq

## 패치 개요

이번 패치는 두 가지 작업을 한 번에 수행합니다.

### Patch A: 잘못된 OOS Multiplier 즉시 비활성화 ⚠️ 긴급
- **이유**: 19주 endpoint 평가 기반의 STATE_REGIME_MULTIPLIER_TABLE이 path 평가에서 정반대 결론을 보임
- **위험**: NEUTRAL × FOLLOW_LIGHT = 0.63x는 잘못된 보정 (path 평가에선 함정 아님)
- **조치**: 즉시 비활성화, UI에 안전 모드 표시

### Patch B: G3 Liquidity (US + EU + JP) 도입
- **이유**: 사용자 매크로 가설 검증 — "BTC = f(글로벌 유동성, 달러, lag)"
- **확장**: Fed Net Liquidity 외 진짜 G3 자산 합산 옵션 추가
- **데이터**: 모두 FRED API (안정적, 무료)

## 작업 순서

```
1. Patch A 먼저 적용 (안전 확보)
2. 동작 확인 후 Patch B 적용
3. Patch B 검증
4. 둘 다 한 PR로 commit (또는 두 개 PR)
```

## 사전 작업

### Step 0.1: 백업

```bash
cd liq
git checkout main
git pull
git checkout -b feat/v2-19-1-patchAB
cp app.py app.py.backup_$(date +%Y%m%d)
```

### Step 0.2: 현재 상태 확인

다음 라인이 존재하는지 확인:
- `H19_DIAG_W = 19` (line ~459 부근)
- `STATE_REGIME_MULTIPLIER_TABLE` (TAB7 관련 코드 안)
- `LIQ_SOURCE = st.selectbox("Liquidity Source", ...)` (사이드바)

```bash
grep -n "H19_DIAG_W" app.py
grep -n "STATE_REGIME_MULTIPLIER_TABLE" app.py
grep -n "LIQ_SOURCE" app.py | head -5
grep -n "G2 M2" app.py | head -5
```

이 grep 결과가 모두 비어있다면 코드 구조가 다른 것이므로 작업 중단하고 보고.

---

# Patch A: 잘못된 OOS Multiplier 비활성화

## 문제 진단

현재 코드의 `STATE_REGIME_MULTIPLIER_TABLE`은 다음 값들을 가지고 있을 가능성:
- NEUTRAL × FOLLOW_LIGHT: 0.63 (또는 0.60)
- BULLISH × FOLLOW: 1.10 (또는 그 이상)
- 기타 alpha_state × regime_state 조합별 multiplier

**이 모든 값들은 19주 endpoint OOS 분석에 기반**합니다. 그러나 이번 path 분석에서:
- 실제 best_lag 19주는 414 anchor 중 단 3회 발생 (0.7%)
- Path 단위 평가에서 endpoint 결론이 정반대로 뒤집힘
- 특히 FOLLOW_LIGHT는 함정이 아니라 정상 영역
- 진짜 함정은 FOLLOW와 SHRINK (현재 보정 안 됨)

→ 따라서 현재 multiplier는 **잘못된 방향으로 보정 중**일 수 있음.

## Patch A 변경 사항

### A.1 STATE_REGIME_MULTIPLIER_TABLE을 비활성화

`STATE_REGIME_MULTIPLIER_TABLE = { ... }` 정의 부분을 찾아 다음과 같이 수정:

**Before**:
```python
STATE_REGIME_MULTIPLIER_TABLE = {
    ('NEUTRAL', 'DEFENSIVE'): 1.10,
    ('NEUTRAL', 'FOLLOW_LIGHT'): 0.63,
    # ... 기타 조합들
}
```

**After**:
```python
# ============================================================
# v2.19.1 Patch A: 19w endpoint 기반 multiplier 비활성화
# ============================================================
# 사유: 2026-05-02 path 분석 결과,
#   - 실제 best_lag 19주 발생 = 414 anchor 중 단 3회 (0.7%)
#   - Path 단위 평가에서 endpoint 결론과 정반대 결론 도출
#   - 예: FOLLOW_LIGHT는 endpoint=함정 / path=정상
#   - 예: SHRINK는 endpoint=평범 / path=함정
# 따라서 현재 multiplier 방향이 잘못되어 있을 가능성 높음.
# Path-based 재산출 전까지 비활성화 (모든 조합 = 1.0).
#
# 원본 multiplier는 코드 보존을 위해 _ARCHIVED_v2_18_5_19w_endpoint에 저장.

_ARCHIVED_v2_18_5_19w_endpoint_MULTIPLIER_TABLE = {
    # 원래 값들을 여기 그대로 보존 (참조용)
    # 위의 Before 블록 내용을 그대로 옮길 것
}

STATE_REGIME_MULTIPLIER_TABLE = {
    # PATCH A: 모든 조합을 1.0으로 강제 (보정 효과 차단)
    # path 분석 검증 후 새 값으로 교체 예정
}
# 또는 더 명시적으로:
# STATE_REGIME_MULTIPLIER_TABLE = {}  # 빈 dict — 보정 미적용 의미

PATCH_A_DISABLED_DATE = "2026-05-02"
PATCH_A_REASON = "19w endpoint 평가 기반 → path 평가에서 잘못된 방향 입증"
```

### A.2 multiplier 적용 코드에 안전 가드 추가

`STATE_REGIME_MULTIPLIER_TABLE`을 사용하는 함수를 찾아 (TAB7 관련 또는 forecast adjust 관련):

```bash
grep -n "STATE_REGIME_MULTIPLIER_TABLE\[" app.py
grep -n "STATE_REGIME_MULTIPLIER_TABLE.get" app.py
```

각 사용처에 safety guard 추가:

**Before**:
```python
mult = STATE_REGIME_MULTIPLIER_TABLE.get((alpha_state, regime_state), 1.0)
adjusted_position = base_position * mult
```

**After**:
```python
# v2.19.1 Patch A: STATE_REGIME_MULTIPLIER_TABLE 비활성화
# path-based 재산출 전까지 모든 조합 1.0
mult = STATE_REGIME_MULTIPLIER_TABLE.get((alpha_state, regime_state), 1.0)
# 안전 가드: 빈 dict이거나 1.0이 아닌 값이라도 무시
if mult != 1.0 and not STATE_REGIME_MULTIPLIER_TABLE:
    mult = 1.0  # 비활성화 상태일 때 강제 1.0
adjusted_position = base_position * mult
```

### A.3 UI에 안전 모드 표시

TAB7 또는 사이드바 어딘가에 다음 메시지 추가:

```python
# v2.19.1 Patch A 안전 모드 알림
if not STATE_REGIME_MULTIPLIER_TABLE:
    st.warning(
        "⚠️ STATE_REGIME_MULTIPLIER 비활성화 (Patch A, 2026-05-02). "
        "사유: 19w endpoint OOS 평가가 path 평가와 모순. "
        "path-based 재산출 전까지 보정 미적용."
    )
```

이 메시지를 적절한 위치에 (TAB7 상단, 또는 앱 시작 시) 표시.

### A.4 검증 (Patch A 후)

```bash
# 코드 변경 확인
grep -n "PATCH_A_DISABLED_DATE" app.py
grep -n "_ARCHIVED_v2_18_5" app.py

# 앱 실행 (로컬)
streamlit run app.py

# 확인 사항:
# 1. 앱이 정상 실행되는가?
# 2. TAB7에서 "Multiplier 비활성화" 알림이 보이는가?
# 3. forecast 관련 함수에서 multiplier가 1.0으로 적용되는가?
# 4. CSV 출력에서 multiplier 컬럼이 1.0인가?
```

성공 시 git commit:

```bash
git add app.py
git commit -m "Patch A: Disable 19w-endpoint-based STATE_REGIME_MULTIPLIER_TABLE

Reason: Path-based analysis on 2026-05-02 found that
endpoint evaluation conclusions are inverted in path evaluation.
- Real best_lag 19w occurred only 3/414 anchors (0.7%)
- FOLLOW_LIGHT: endpoint=trap / path=normal (over-corrected)
- SHRINK: endpoint=mild / path=trap (under-corrected)

Original table preserved in _ARCHIVED_v2_18_5_19w_endpoint_MULTIPLIER_TABLE.
Will be replaced after path-based recalculation completes."
```

---

# Patch B: G3 Liquidity (US + EU + JP) 도입

## 목표

기존 Fed Net Liquidity 외에 G3 Total Assets 옵션 추가:
- US: WALCL (Fed total assets, weekly)
- EU: ECBASSETSW (ECB total assets, weekly)
- JP: JPNASSETS (BOJ total assets, monthly → forward fill)

모두 USD millions 단위로 환산 후 합산.

## 합의된 설계 명세

| 항목 | 결정 |
|------|------|
| BOJ 월간 처리 | Forward fill (look-ahead 회피) |
| 출력 형태 | 절대값 + 정규화 인덱스 둘 다 |
| 사이드바 옵션 | Fed + G3 Total + G3 YoY (3개) |
| G3 정의 | Total Assets (Net 아님) |
| 환율 처리 | USD 환산 (단순) |
| 데이터 시작점 | 2019-08 이후 (ECBASSETSW 시작점) |

## Patch B 변경 사항

### B.1 새 fetch 함수 추가

기존 `fetch_fred_fredgraph` 함수 근처에 다음 함수 추가:

```python
# ============================================================
# v2.19.1 Patch B: G3 Liquidity (US + EU + JP)
# ============================================================
# FRED 시리즈 (모두 무료, 인증 불필요):
#   WALCL: Fed Total Assets (millions USD, weekly Wed)
#   ECBASSETSW: ECB Total Assets (millions EUR, weekly Fri)
#   JPNASSETS: BOJ Total Assets (100M JPY, monthly)
#   DEXUSEU: USD per 1 EUR (daily)
#   DEXJPUS: JPY per 1 USD (daily)

FRED_WALCL = "WALCL"  # 이미 존재할 수 있음, 중복 정의 주의
FRED_ECBASSETSW = "ECBASSETSW"
FRED_JPNASSETS = "JPNASSETS"
FRED_DEXUSEU = "DEXUSEU"
FRED_DEXJPUS = "DEXJPUS"

G3_START_DATE = "2019-08-01"  # ECBASSETSW 데이터 시작 시점
G3_INDEX_BASE_DATE = "2020-01-03"  # 정규화 인덱스 기준일 (코로나 직전)


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

    # ── Step 1: FRED 데이터 fetch ───────────────────────
    try:
        walcl = fetch_fred_fredgraph(FRED_WALCL)
        LOG_ALPHA.info(f"[G3] WALCL fetched: {len(walcl)} obs")
    except Exception as e:
        raise RuntimeError(f"[G3] WALCL fetch failed: {e}") from e

    try:
        ecb = fetch_fred_fredgraph(FRED_ECBASSETSW)
        LOG_ALPHA.info(f"[G3] ECBASSETSW fetched: {len(ecb)} obs")
    except Exception as e:
        raise RuntimeError(f"[G3] ECBASSETSW fetch failed: {e}") from e

    try:
        boj = fetch_fred_fredgraph(FRED_JPNASSETS)
        LOG_ALPHA.info(f"[G3] JPNASSETS fetched: {len(boj)} obs (monthly)")
    except Exception as e:
        raise RuntimeError(f"[G3] JPNASSETS fetch failed: {e}") from e

    try:
        eur_usd = fetch_fred_fredgraph(FRED_DEXUSEU)
        LOG_ALPHA.info(f"[G3] DEXUSEU fetched: {len(eur_usd)} obs")
    except Exception as e:
        raise RuntimeError(f"[G3] DEXUSEU fetch failed: {e}") from e

    try:
        jpy_usd = fetch_fred_fredgraph(FRED_DEXJPUS)
        LOG_ALPHA.info(f"[G3] DEXJPUS fetched: {len(jpy_usd)} obs")
    except Exception as e:
        raise RuntimeError(f"[G3] DEXJPUS fetch failed: {e}") from e

    # ── Step 2: 모든 시리즈를 W-FRI 주간으로 정렬 ────────
    # WALCL (Wed): 그 주 W-FRI에 forward fill
    # ECBASSETSW (Fri): 자연스럽게 W-FRI
    # JPNASSETS (월말): 다음 발표까지 forward fill
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
    # base_dt에 가장 가까운 W-FRI
    if base_dt in df.index:
        base_value = df.loc[base_dt, "g3_total_usd_m"]
    else:
        # 가장 가까운 인덱스 찾기
        nearest = df.index[df.index >= base_dt][0] if any(df.index >= base_dt) else df.index[0]
        base_value = df.loc[nearest, "g3_total_usd_m"]
        LOG_ALPHA.info(f"[G3] index base date {base_dt} not exact, using {nearest}")

    df["g3_index"] = df["g3_total_usd_m"] / base_value * 100

    # ── Step 7: 변화율 계산 ────────────────────────────
    df["g3_yoy_pct"] = df["g3_total_usd_m"].pct_change(52) * 100  # 52주 = 1년
    df["g3_4w_change_pct"] = df["g3_total_usd_m"].pct_change(4) * 100  # 4주 (월간 근사)
    df["g3_13w_change_pct"] = df["g3_total_usd_m"].pct_change(13) * 100  # 13주 (분기)

    LOG_ALPHA.info(
        f"[G3] G3 panel built: {len(df)} weekly obs, "
        f"{df.index.min().date()} ~ {df.index.max().date()}"
    )

    return df
```

### B.2 사이드바 LIQ_SOURCE 옵션 변경

기존 코드를 찾아:

```python
LIQ_SOURCE = st.selectbox(
    "Liquidity Source",
    [
        "Fed Net Liquidity (FRED)",
        "G2 M2 (US+EA, USD)",  # 기존 — 작동 안 함
    ],
    index=0,
)
```

다음과 같이 변경:

```python
LIQ_SOURCE = st.selectbox(
    "Liquidity Source",
    [
        "Fed Net Liquidity (FRED)",      # 기존 (그대로)
        "G3 Total Assets (USD)",         # 신규 — Fed + ECB + BOJ
        "G3 YoY Change (%)",             # 신규 — G3 변화율 (driver로)
        # "G2 M2 (US+EA, USD)",          # 폐기 (ECB API 변경으로 작동 안 함)
    ],
    index=0,
    help="G3 = Fed + ECB + BOJ Total Assets (USD-converted). G3 YoY는 변화율 driver."
)
```

### B.3 LIQ_SOURCE 분기 로직 변경

기존 `if LIQ_SOURCE == "G2 M2 (US+EA, USD)"` 분기가 있는 곳을 모두 찾아:

```bash
grep -n "G2 M2" app.py
grep -n "LIQ_SOURCE ==" app.py
```

각 분기를 다음과 같이 수정:

**Before**:
```python
if LIQ_SOURCE == "Fed Net Liquidity (FRED)":
    liq_series = compute_fed_net_liquidity()
elif LIQ_SOURCE == "G2 M2 (US+EA, USD)":
    liq_series = compute_g2_m2_usd()  # 이 함수는 작동 안 함
```

**After**:
```python
if LIQ_SOURCE == "Fed Net Liquidity (FRED)":
    liq_series = compute_fed_net_liquidity()
elif LIQ_SOURCE == "G3 Total Assets (USD)":
    g3_panel = fetch_g3_total_assets()
    liq_series = g3_panel["g3_total_usd_m"]
    liq_series.name = "G3_Total_Assets_USD_m"
elif LIQ_SOURCE == "G3 YoY Change (%)":
    g3_panel = fetch_g3_total_assets()
    liq_series = g3_panel["g3_yoy_pct"]
    liq_series.name = "G3_YoY_pct"
else:
    raise ValueError(f"Unknown LIQ_SOURCE: {LIQ_SOURCE}")
```

### B.4 CSV 출력 컬럼 추가

ONE FILE CSV 생성 부분을 찾아 (TAB4 또는 메인 출력 함수):

```bash
grep -n "MAIN_TAB4_ONEFILE" app.py
grep -n "to_csv" app.py | head -10
```

CSV 컬럼 구성에 G3 관련 컬럼 추가:

```python
# v2.19.1 Patch B: G3 컬럼을 항상 출력 (LIQ_SOURCE 무관)
try:
    g3_panel = fetch_g3_total_assets()
    # 기존 weekly_df와 merge
    weekly_df = weekly_df.merge(
        g3_panel[["g3_total_usd_m", "g3_index", "g3_yoy_pct",
                  "g3_4w_change_pct", "g3_13w_change_pct",
                  "fed_usd_m", "ecb_usd_m", "boj_usd_m"]],
        left_index=True, right_index=True, how="left"
    )
    LOG_ALPHA.info(f"[G3] columns merged into weekly_df")
except Exception as e:
    LOG_ALPHA.warning(f"[G3] merge failed: {e}, G3 columns will be NaN")
    for col in ["g3_total_usd_m", "g3_index", "g3_yoy_pct",
                "g3_4w_change_pct", "g3_13w_change_pct",
                "fed_usd_m", "ecb_usd_m", "boj_usd_m"]:
        weekly_df[col] = np.nan
```

### B.5 fetch_fred_fredgraph 함수 강화 (선택)

기존 함수가 다음과 같다면:

```python
@st.cache_data(ttl=60*60*6, show_spinner=False)
def fetch_fred_fredgraph(series_id: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url)
    # ...
```

다음과 같은 retry/timeout 추가 권장:

```python
@st.cache_data(ttl=60*60*6, show_spinner=False)
def fetch_fred_fredgraph(series_id: str) -> pd.Series:
    """FRED CSV API fetch with retry."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    
    last_err = None
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            date_col = df.columns[0]
            val_col = df.columns[1]
            df[date_col] = pd.to_datetime(df[date_col])
            s = pd.to_numeric(df[val_col], errors="coerce")
            s.index = df[date_col]
            s.name = series_id
            return s.dropna()
        except Exception as e:
            last_err = e
            if attempt < 2:
                time.sleep(2 ** attempt)  # exponential backoff
            continue
    raise RuntimeError(f"FRED fetch failed for {series_id} after 3 attempts: {last_err}")
```

### B.6 검증 (Patch B 후)

```bash
# 코드 변경 확인
grep -n "fetch_g3_total_assets" app.py
grep -n "G3 Total Assets" app.py | head -5

# 앱 실행
streamlit run app.py
```

확인 사항:
1. 사이드바에 3개 옵션이 보이는가?
   - Fed Net Liquidity (FRED)
   - G3 Total Assets (USD)
   - G3 YoY Change (%)
2. "G3 Total Assets" 선택 시 모델이 정상 작동하는가?
3. 첫 G3 fetch는 시간 걸림 (수 분), 두 번째부터 cache 사용
4. CSV에 G3 컬럼이 추가되는가?
   - g3_total_usd_m
   - g3_index
   - g3_yoy_pct
   - g3_4w_change_pct
   - g3_13w_change_pct
   - fed_usd_m, ecb_usd_m, boj_usd_m
5. spaghetti CSV 다운로드 가능한가? (LIQ_SOURCE에 G3 표기됨)

### B.7 데이터 검증 (G3 모드)

G3 모드 활성화 후 CSV 다운받아서:

```python
import pandas as pd
df = pd.read_csv("MAIN_TAB4_ONEFILE_G3_Total_Assets...csv")

# 기본 sanity check
print(f"행수: {len(df)}")
print(f"기간: {df['date'].min()} ~ {df['date'].max()}")

# G3 컬럼 확인
print(f"\nG3 컬럼 통계:")
print(df[['g3_total_usd_m', 'g3_index', 'g3_yoy_pct', 
          'fed_usd_m', 'ecb_usd_m', 'boj_usd_m']].describe())

# 절대값 sanity (대략 $20T 이상이어야 정상)
assert df['g3_total_usd_m'].dropna().min() > 15_000_000  # >$15T
assert df['g3_total_usd_m'].dropna().max() < 50_000_000  # <$50T

# 인덱스 sanity (2020-01-03 근처가 100이어야)
base_value = df[df['date'].str.contains('2020-01')]['g3_index'].iloc[0]
assert 99 <= base_value <= 101, f"Base index should be ~100, got {base_value}"
```

## git commit (Patch B)

```bash
git add app.py
git commit -m "Patch B: Add G3 (US+EU+JP) Total Assets liquidity option

New liquidity source options:
- 'G3 Total Assets (USD)': Fed + ECB + BOJ assets, USD-converted
- 'G3 YoY Change (%)': G3 year-over-year change as driver

Removes broken 'G2 M2 (US+EA, USD)' option (ECB API changed).

Data sources (all FRED, free, no auth):
- WALCL: Fed total assets (weekly)
- ECBASSETSW: ECB total assets (weekly)
- JPNASSETS: BOJ total assets (monthly, forward-filled)
- DEXUSEU, DEXJPUS: exchange rates

Specs:
- USD-converted at weekly W-FRI close
- BOJ forward-fill (PIT-safe, no look-ahead)
- Index normalization base: 2020-01-03 = 100
- Data starts 2019-08 (ECBASSETSW availability)

Goal: validate user's macro hypothesis BTC = f(global_liquidity, dollar, lag).
Comparison test: Fed Net Liquidity vs G3 Total Assets (path quality)."
```

---

# 검증 체크리스트 (둘 다 적용 후)

## Patch A 검증
- [ ] STATE_REGIME_MULTIPLIER_TABLE이 비어있거나 모두 1.0
- [ ] forecast 함수에서 multiplier가 1.0으로 적용
- [ ] CSV의 multiplier 관련 컬럼이 1.0
- [ ] UI에 안전 모드 알림 표시
- [ ] 앱 정상 실행

## Patch B 검증
- [ ] 사이드바에 3개 LIQ_SOURCE 옵션
- [ ] G2 M2 옵션 제거됨
- [ ] G3 Total Assets 모드 정상 작동
- [ ] G3 YoY 모드 정상 작동
- [ ] CSV에 G3 관련 컬럼 추가
- [ ] G3 절대값 sanity 통과 (~$20-40T)
- [ ] 첫 fetch 후 cache 작동 (재실행 시 즉각 응답)

## 통합 검증
- [ ] Fed Net Liquidity 모드도 정상 작동 (regression 없음)
- [ ] 기존 alpha 자동 fetch 영향 없음
- [ ] spaghetti CSV 정상 생성
- [ ] TAB6, TAB7 정상 표시
- [ ] 앱 시작 시 에러 없음

# 롤백 절차

만약 문제 발생 시:

```bash
# 백업 파일로 즉시 복구
cd liq
git checkout main
cp app.py.backup_<날짜> app.py
git add app.py
git commit -m "Rollback Patch A+B due to issue"
git push

# 또는 PR을 닫고 main 그대로
git checkout main
git branch -D feat/v2-19-1-patchAB
```

# 예상 문제 및 대응

## 문제 1: FRED 시리즈 중 일부가 갱신 안 됨
**증상**: ECBASSETSW가 최근 1개월 데이터 없음
**대응**: forward fill로 자동 처리. 알림은 LOG에만.

## 문제 2: 환율 데이터 weekend 결측
**증상**: DEXUSEU의 토/일 값이 NaN
**대응**: W-FRI resample(last) + ffill로 자연 해결

## 문제 3: BOJ 단위 잘못 계산
**증상**: BOJ 자산이 너무 크거나 작아 보임
**검증**: 2024-12 BOJ 자산 ≈ 740T JPY ≈ $4.7T (USDJPY=157 기준)
**디버그**: 코드의 `boj_usd_m = boj_w * 100 / jpy_usd_w` 단위 확인
- JPNASSETS 단위는 100 million JPY (10억엔 = 0.1B JPY)
- 따라서 ×100으로 million JPY로 만든 후 / 환율

## 문제 4: 첫 fetch 너무 느림 (수 분)
**원인**: FRED CSV API는 큰 시리즈 fetch 시 느릴 수 있음
**대응**: cache TTL 6시간으로 충분. 두 번째부터 즉각.

## 문제 5: 캐시 충돌
**증상**: G3 모드 변경 후 옛 결과 보임
**대응**: Streamlit Cloud 우측 상단 ⋮ → Clear cache

# 다음 단계 (Patch B 검증 후)

성공 시:
1. Streamlit Cloud에서 G3 모드로 spaghetti CSV 다운로드
2. 사용자가 CSV 첨부하여 채팅에 업로드
3. 제가 path 분석 실행
4. Fed vs G3 비교 보고

검증 문제 시:
1. 정확한 에러 메시지 + 로그 보고
2. 디버깅 함께 진행
3. 필요시 Patch B 롤백, Patch A만 유지

