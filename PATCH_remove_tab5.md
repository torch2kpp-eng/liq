# Claude Code 작업 지시서 — TAB5 완전 삭제

> **작업 범위**: TAB5 (Multi-Asset Forecast/Backtest) 완전 제거
> **이유**: ASSET_SYMBOLS 미정의로 NameError 발생, 더 이상 사용하지 않음
> **영향**: TAB5 클릭 시 + TAB7 클릭 시 모두 발생 (구조적 문제)

---

## 📋 현 상황

### 에러 메시지 (TAB5/TAB7 클릭 시 동일)
```
NameError: ASSET_SYMBOLS
File "/mount/src/liq/app.py", line ~524 또는 5241
    run_tab5_multiasset()
File "/mount/src/liq/app.py", line ~5034
    asset_key = st.selectbox("Asset", list(ASSET_SYMBOLS.keys()), index=0)
```

### 원인
- `ASSET_SYMBOLS` 변수가 정의되어 있지 않음
- `run_tab5_multiasset` 함수가 TAB5 외에도 어딘가에서 호출되고 있음 → TAB7 클릭 시에도 에러 발생

### 작업 후 기대 상태
- TAB5 자체 제거
- TAB7 정상 작동
- 다른 탭(MAIN, TAB1~4) 영향 없음

---

## 🔧 작업 항목

### 작업 1: 탭 라벨 리스트에서 TAB5 제거

**위치**: `streamlit_app.py`에서 `st.tabs([...])` 정의 부분 찾기

**찾을 코드**:
```python
tabs = st.tabs([
    "MAIN (True-history + Spaghetti)",
    "TAB1 DXY(역축) vs BTC",
    "TAB2 Liquidity vs BTC",
    "TAB3 Combo + 2D Lag-Pair",
    "TAB4 Forward Overlay (LDLI vs BTC)",
    "TAB5 Multi-Asset Forecast/Backtest",  # ← 제거
    "TAB6 Alpha Integration Lab",  # 있다면
    "TAB7 Phase 1 Validation",
])
```

**수정**: `"TAB5 Multi-Asset Forecast/Backtest"` 라인 완전 제거

```python
tabs = st.tabs([
    "MAIN (True-history + Spaghetti)",
    "TAB1 DXY(역축) vs BTC",
    "TAB2 Liquidity vs BTC",
    "TAB3 Combo + 2D Lag-Pair",
    "TAB4 Forward Overlay (LDLI vs BTC)",
    "TAB6 Alpha Integration Lab",  # 있다면
    "TAB7 Phase 1 Validation",
])
```

---

### 작업 2: TAB5 호출부 제거 + 인덱스 재조정

**찾을 코드** (모든 곳):
```python
with tabs[5]:
    run_tab5_multiasset()
```

또는 다른 형태:
```python
run_tab5_multiasset()
```

**처리**:
1. `run_tab5_multiasset()` 호출 자체를 모두 제거
2. **TAB5 이후의 모든 `with tabs[N]:` 인덱스를 1씩 감소**

예시:
```python
# Before:
with tabs[5]:
    run_tab5_multiasset()
with tabs[6]:
    render_alpha_integration_lab(payload)  # TAB6
with tabs[7]:
    render_phase1_validation_tab(payload)  # TAB7

# After:
with tabs[5]:
    render_alpha_integration_lab(payload)  # 이전 TAB6 → 새 인덱스 5
with tabs[6]:
    render_phase1_validation_tab(payload)  # 이전 TAB7 → 새 인덱스 6
```

⚠️ **반드시 grep으로 모든 호출부 찾기**:
```bash
grep -n "run_tab5_multiasset\|tabs\[" streamlit_app.py
```

---

### 작업 3: `run_tab5_multiasset` 함수 정의 자체 제거

`def run_tab5_multiasset():` 또는 `def run_tab5_multiasset(...)` 로 시작하는 함수 정의 전체를 제거.

```python
# 이 블록 전체 제거
def run_tab5_multiasset():
    asset_key = st.selectbox("Asset", list(ASSET_SYMBOLS.keys()), index=0)
    ...
    # 함수 끝까지 모두 제거
```

함수 시작과 끝을 정확히 찾으려면:
- 시작: `def run_tab5_multiasset` 라인
- 끝: 다음 `def `가 시작하는 직전 라인 (또는 같은 들여쓰기 레벨의 다른 정의)

---

### 작업 4: TAB5 전용 상수 제거

`ASSET_SYMBOLS` 등 TAB5에서만 쓰는 상수가 있다면 제거. 단, **다른 탭에서도 쓰는지 먼저 grep 확인**:

```bash
grep -n "ASSET_SYMBOLS" streamlit_app.py
```

- 만약 `run_tab5_multiasset` 안에서만 쓰인다면 → 안전하게 삭제
- 다른 곳에서도 쓰이면 → 그대로 두고 정의만 추가 필요 (다른 작업)

---

### 작업 5: 검증

```bash
# Syntax check
python -c "import ast; ast.parse(open('streamlit_app.py').read())"

# Reference check (모두 0이어야 함)
grep -n "run_tab5_multiasset" streamlit_app.py
grep -n "TAB5 Multi-Asset" streamlit_app.py

# tabs 인덱스 일관성 확인
grep -n "tabs\[" streamlit_app.py
# 출력된 인덱스 값들이 0부터 (총 탭 수 - 1)까지 연속되는지 확인
```

---

### 작업 6: 커밋 메시지

```
fix: remove broken TAB5 (multi-asset) to unblock TAB7 and other tabs

- Remove TAB5 label from st.tabs() list
- Remove run_tab5_multiasset() function definition entirely
- Remove all run_tab5_multiasset() call sites
- Adjust tab index for TAB6/TAB7 (decrement by 1)
- Remove ASSET_SYMBOLS constant (only used by TAB5)

Resolves: NameError: ASSET_SYMBOLS preventing TAB5/TAB7 access
```

---

## ✅ 검증 체크리스트

작업 후 확인:

```
□ 1. python -c "import ast; ast.parse(open('streamlit_app.py').read())" 통과
□ 2. grep "run_tab5_multiasset" → 결과 0건
□ 3. grep "TAB5" → 결과 0건 (또는 주석에만 있음)
□ 4. Streamlit 정상 실행
□ 5. 모든 탭 클릭 가능:
   - MAIN ✅
   - TAB1 DXY ✅
   - TAB2 Liquidity ✅
   - TAB3 Combo ✅
   - TAB4 Forward Overlay ✅
   - TAB6 Alpha Integration Lab (있다면) ✅
   - TAB7 Phase 1 Validation ✅ (이게 핵심)
□ 6. TAB7 클릭 시 6개 섹션 정상 표시
   - WS-A Path multiplier 효과
   - Alpha state별 OOS 성과
   - Acceptance criteria 판정
   - 2D Matrix
   - FOLLOW_LIGHT 진단
   - 권장 베팅 조합
```

---

## ⚠️ 주의사항

### 1. TAB5만 제거. 다른 탭은 절대 건드리지 말 것
- MAIN, TAB1~4는 정상 작동 중 (캡처로 확인됨)
- TAB6 (Alpha Integration Lab)이 있다면 보존
- TAB7 (Phase 1 Validation)은 보존

### 2. 인덱스 재조정 시 한 곳도 빠뜨리지 말 것
- `tabs[5]`, `tabs[6]`, `tabs[7]` 등 모든 인덱스 점검
- with 블록 외에 다른 곳에서 인덱스 사용한다면 그것도 수정

### 3. ASSET_SYMBOLS 외 TAB5 전용 변수도 확인
- 함수 정의 시작부터 끝까지 사용하는 변수들이 다른 곳에서도 쓰이는지 grep으로 확인
- TAB5만 쓰는 변수는 함께 제거

### 4. PR 생성 → 머지 → 자동 재배포

이전과 동일한 절차.

---

## 🚀 작업 순서 권장

1. **작업 5 (검증)을 먼저 실행하여 현 상태 파악** (grep)
2. **작업 1**: 탭 라벨 리스트에서 TAB5 제거
3. **작업 2**: 호출부 제거 + 인덱스 재조정
4. **작업 3**: 함수 정의 제거
5. **작업 4**: TAB5 전용 상수 제거 (확인 후)
6. **작업 5 (재검증)**: syntax check + reference check
7. **PR 생성 + 머지**

---

## 📋 작업 완료 후 알려주실 것

1. ✅ 모든 탭이 정상 클릭 가능한지
2. ✅ TAB7 (Phase 1 Validation)에서 6개 섹션 모두 보이는지
3. ✅ MAIN 탭의 "Phase 1 OOS 보정 적용 중" 표시가 그대로 유지되는지
