# 추가 패치 지시서 — Alpha Layer 활성화 마무리

> **이전 패치 결과 진단:**
> - ✅ MVRV 자동 fetch: 정상 (CoinMetrics 209주 데이터)
> - ✅ alpha_state 분화: 정상 (NEUTRAL/BULLISH/BEARISH 출현)
> - ⚠️ Funding: 사이드바 UI는 있으나 실제 데이터 NaN
> - ❌ Reserve/ETF: ONE FILE CSV에 컬럼 자체가 누락 (fetcher는 추가됐으나 export 누락)

---

## 🎯 수정 목표

ONE FILE CSV 다운로드 시 다음 9개 신규 컬럼이 모두 포함되어야 함:

```
exchange_reserve_btc
reserve_pct_change_4w
reserve_z
reserve_state
etf_netflow_usd_m
etf_4w_cumulative
etf_z
etf_state
etf_available
```

또한 Funding fetch 에러 원인 진단.

---

## 작업 1: ONE FILE CSV의 enrich_cols 리스트 보강

**위치**: `streamlit_app.py`의 `run_main_tab` 함수 내부, ONE FILE CSV 생성 블록.

**찾을 코드** (대략 line 2300~2400 근처):

```python
enrich_cols = [
    "realized_fwd_px_19w", "realized_fwd_ret_19w", "realized_sign_19w",
    "predicted_fwd_px_19w", "predicted_fwd_ret_19w", "predicted_sign_19w",
    "predicted_fwd_px_19w_adj", "predicted_fwd_ret_19w_adj", "predicted_sign_19w_adj",
    "signal_hit_19w", "regime_state", "regime_path_multiplier", "regime_position_multiplier",
    "confidence_bucket", "suggested_exposure", "recent_52w_flag", "recent_104w_flag",
    "alpha_state", "mvrv_state", "funding_state", "endpoint_dt_h19", "lag_weeks_h19",
    ...
]
```

**다음 컬럼들을 enrich_cols 리스트에 추가**:

```python
# v2.19 신규 alpha 컬럼 — Reserve & ETF
"exchange_reserve_btc", "reserve_pct_change_4w", "reserve_z", "reserve_state",
"etf_netflow_usd_m", "etf_4w_cumulative", "etf_z", "etf_state", "etf_available",
```

또한 **`preferred_cols` 리스트** (같은 함수 내, 더 아래에 있음)에도 동일 컬럼 추가:

```python
preferred_cols = [
    "btc_close",
    ...
    # 기존 alpha 컬럼들 사이에 추가
    "mvrv_z", "mvrv_state",
    "funding_rate_w", "funding_8w_ma", "funding_z", "funding_state",
    # ↓ 신규 추가 ↓
    "exchange_reserve_btc", "reserve_pct_change_4w", "reserve_z", "reserve_state",
    "etf_netflow_usd_m", "etf_4w_cumulative", "etf_z", "etf_state", "etf_available",
    # ↑ 신규 추가 ↑
    "alpha_state",
    ...
]
```

---

## 작업 2: Funding 에러 디버깅

**현 증상**: 사이드바에 Funding 체크박스 있고 `funding_state` 컬럼도 생성되는데 `funding_rate_w`, `funding_z`가 모든 행 NaN.

**원인 후보**:
1. Streamlit Cloud 환경에서 Binance Futures API (`fapi.binance.com`) 차단
2. `load_alpha_inputs_weekly` 수정 시 funding 부분 호출 경로 깨짐

**진단 방법**: 
`fetch_binance_funding_history` 함수를 직접 호출하는 임시 테스트 추가. `run_main_tab` 함수 시작 부분에 한 줄 추가:

```python
def run_main_tab():
    st.subheader("MAIN) BTC vs LDLI (+ True-history predicted + Spaghetti)")
    
    # === [DEBUG] Funding fetch 진단 ===
    with st.expander("🔧 Debug: Funding fetch 직접 테스트", expanded=False):
        if st.button("Funding API 직접 호출 테스트"):
            try:
                test_funding = fetch_binance_funding_history(symbol="BTCUSDT", start_date=START_DATE)
                st.success(f"Funding fetch OK: {len(test_funding)} rows")
                st.write(test_funding.tail(5))
            except Exception as e:
                st.error(f"Funding fetch 실패: {type(e).__name__}: {e}")
    
    try:
        payload = build_forward_overlay_payload(...)
    ...
```

이걸 추가하면 사이드바 expander에서 직접 버튼을 눌러 진단할 수 있음.

---

## 작업 3: ETF Farside 진단 로그 추가

**위치**: `fetch_etf_netflow_auto` 함수 시작 부분.

**추가**: 함수 시작 시 진단 로그를 더 자세히 출력:

```python
@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fetch_etf_netflow_auto(week_rule: str = WEEK_RULE) -> pd.DataFrame:
    LOG_ALPHA.info("[ETF] fetch_etf_netflow_auto 시작")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; BTCMacroPredictionEngine/2.19)"
    }
    
    # 1순위: Farside Investors
    try:
        LOG_ALPHA.info("[ETF] Farside HTTP 요청 시작")
        r = requests.get("https://farside.co.uk/btc/", headers=headers, timeout=20)
        LOG_ALPHA.info(f"[ETF] Farside HTTP status: {r.status_code}, content size: {len(r.text)}")
        r.raise_for_status()
        ...
```

이렇게 하면 Streamlit Cloud 로그에서 어디서 막히는지 추적 가능.

---

## 작업 4: 작업 완료 후 검증

### 검증 1: Streamlit 실행 후 ONE FILE CSV 다운로드

다음 컬럼들이 **모두** CSV에 존재하는지 확인:
- exchange_reserve_btc, reserve_pct_change_4w, reserve_z, reserve_state
- etf_netflow_usd_m, etf_4w_cumulative, etf_z, etf_state, etf_available

데이터가 NaN이어도 **컬럼 자체가 존재**해야 함.

### 검증 2: Funding 진단

사이드바 expander에서 "Funding API 직접 호출 테스트" 버튼 클릭 후 결과 보고:
- 성공 시: 정상 작동인데 `load_alpha_inputs_weekly` 경로 문제
- 실패 시: 에러 타입과 메시지를 보고 (Streamlit Cloud 환경 이슈 가능성)

---

## 작업 5: 커밋 메시지

```
fix(v2.19): include Reserve/ETF columns in ONE FILE CSV export

- Add reserve/ETF columns to enrich_cols and preferred_cols lists
- Add funding fetch debug expander for Streamlit Cloud diagnosis
- Improve ETF fetcher logging for troubleshooting

Resolves: Reserve/ETF columns missing from CSV despite fetchers being added
```

---

## ⚠️ 중요 주의사항

1. **기존 컬럼 순서나 이름 변경 금지** — 추가만 할 것
2. **enrich_cols와 preferred_cols 두 리스트 모두에 추가** — 한 곳만 추가하면 안 됨
3. **작업 후 streamlit_app.py 문법 체크** (`python -c "import ast; ast.parse(open('streamlit_app.py').read())"`)
4. **PR 생성 후 머지** (이전과 동일 절차)
