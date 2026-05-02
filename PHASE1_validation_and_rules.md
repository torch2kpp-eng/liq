# Claude Code 작업 지시서 — Phase 1 검증 자동화 + 운영 Rule 코드화

> **작업 범위**: 2개 작업 (TAB7 신규 + position multiplier 보강)
> **적용 방식**: streamlit_app.py 직접 패치 (기존 함수 보존)
> **목표**: 
>   1. 매주 데이터 갱신 시 Phase 1 acceptance criteria 자동 재판정
>   2. FOLLOW_LIGHT 함정 + BULLISH 강세 조합을 position multiplier에 반영

---

## 📚 사전 컨텍스트 — 지난 분석에서 발견된 사실

### Fact 1: BULLISH alpha state는 압도적으로 강력
```
BULLISH (n=58):  IC = +0.851,  sign_acc = 63.8%,  mean_real = +14.4%
NEUTRAL (n=338): IC = -0.026,  sign_acc = 47.9%,  mean_real = +11.0%

→ ΔIC = +0.877, Δsign_acc = +15.9%p (acceptance 기준의 17배/3배)
```

### Fact 2: FOLLOW_LIGHT는 구조적 함정
```
FOLLOW_LIGHT (n=61):
  sign_acc = 41.0% (50% 미만, 동전던지기보다 나쁨)
  IC = -0.197 (음수, 모델이 반대 방향을 가리킴)
  중앙값 realized = -13.5%
  손실 비중 = 59% (10주 중 6주는 손실)
  -20% 이상 폭락 비중 = 29.5%
```

### Fact 3: 2D Matrix가 베팅 시점 명확히 알려줌
```
                 sign_acc 매트릭스 (n)
                DEFENSIVE  FOLLOW   FOLLOW_LIGHT  SHRINK    TRANSITION
NEUTRAL           53.5%    48.6%    42.1% ⚠️       47.4%     48.5%
                 (n=43)   (n=37)    (n=57)       (n=38)    (n=163)
BULLISH           58.8%    75.0% ⭐  (n=3)         71.4% ⭐   65.2% ⭐
                 (n=17)   (n=8)                  (n=7)     (n=23)
```

### Fact 4: Path multiplier는 작동 (양쪽 샘플 개선)
```
방향 적중 샘플 (n=199): MAE 24.8% → 24.5% (-0.24%p)
방향 오적중 샘플 (n=199): MAE 57.8% → 54.0% (-3.82%p) ⭐
```

이번 작업의 **즉각 운영 가치**: 현재 FOLLOW_LIGHT regime에서 모델이 70% 노출을 권고 중인데, 통계적으로 50%로 축소해야 함. 이를 코드화.

---

## 🎯 작업 1: TAB7 신규 — Phase 1 Validation Dashboard

### 위치
기존 TAB 구조 끝에 신규 TAB 추가 (또는 기존 마지막 TAB 다음에 추가).

### 작업 1-1: TAB 정의 추가

기존 TAB 정의가 있는 곳 찾아서 추가:

**기존 코드** (대략 main 함수 어딘가):
```python
tabs = st.tabs([
    "MAIN (True-history + Spaghetti)",
    "TAB1 DXY(역축) vs BTC",
    ...
])
```

**수정**:
```python
tabs = st.tabs([
    "MAIN (True-history + Spaghetti)",
    "TAB1 DXY(역축) vs BTC",
    ...,
    "TAB7 Phase 1 Validation",  # 신규
])
```

### 작업 1-2: TAB7 렌더링 함수 추가

`run_main_tab` 함수 근처에 새 함수 추가:

```python
# ============================================================
# v2.19 Phase 1 — Validation Dashboard (TAB7)
# ============================================================

def render_phase1_validation_tab(payload: Dict):
    """
    Phase 1 acceptance criteria를 OOS 데이터로 자동 판정.
    
    검증 항목:
      1. WS-A: Path/Position multiplier 효과 (조건부 분리)
      2. WS-D: State separation (alpha_state별 IC, sign_acc)
      3. 2D Matrix: alpha × regime
      4. FOLLOW_LIGHT regime 진단
      5. v2.19 acceptance criteria 자동 판정
    """
    st.subheader("Phase 1 Validation Dashboard")
    st.caption("v2.19 acceptance criteria 자동 판정 — 매 실행 시 재계산")
    
    overlay_df = payload.get("overlay_master_df")
    if overlay_df is None or overlay_df.empty:
        st.warning("Overlay 데이터가 없습니다.")
        return
    
    # 분석 가능한 행만 필터 (realized + predicted 모두 있어야)
    needed = ["realized_fwd_ret_19w", "predicted_fwd_ret_19w", 
              "predicted_fwd_ret_19w_adj", "alpha_state", "regime_state"]
    if not all(c in overlay_df.columns for c in needed):
        missing = [c for c in needed if c not in overlay_df.columns]
        st.error(f"필수 컬럼 누락: {missing}")
        return
    
    v = overlay_df.dropna(subset=["realized_fwd_ret_19w", "predicted_fwd_ret_19w"]).copy()
    
    if len(v) < 30:
        st.warning(f"OOS 표본이 부족합니다 (n={len(v)}). 30 이상 필요.")
        return
    
    st.info(f"분석 표본: n = {len(v)}")
    
    # === Section 1: WS-A Path Multiplier 효과 ===
    st.markdown("### 1️⃣ WS-A: Path Multiplier 조건부 효과")
    
    raw_pred = v["predicted_fwd_ret_19w"]
    adj_pred = v["predicted_fwd_ret_19w_adj"]
    real = v["realized_fwd_ret_19w"]
    
    # base 방향 적중/오적중 분리
    base_correct_mask = np.sign(raw_pred) == np.sign(real)
    base_correct = v[base_correct_mask]
    base_wrong = v[~base_correct_mask]
    
    mae_correct_raw = (base_correct["predicted_fwd_ret_19w"] - base_correct["realized_fwd_ret_19w"]).abs().mean()
    mae_correct_adj = (base_correct["predicted_fwd_ret_19w_adj"] - base_correct["realized_fwd_ret_19w"]).abs().mean()
    mae_wrong_raw = (base_wrong["predicted_fwd_ret_19w"] - base_wrong["realized_fwd_ret_19w"]).abs().mean()
    mae_wrong_adj = (base_wrong["predicted_fwd_ret_19w_adj"] - base_wrong["realized_fwd_ret_19w"]).abs().mean()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "방향 적중 샘플 MAE 변화",
            f"{(mae_correct_adj - mae_correct_raw)*100:+.2f}%p",
            delta=f"n={len(base_correct)}",
            delta_color="off"
        )
    with col2:
        st.metric(
            "방향 오적중 샘플 MAE 변화",
            f"{(mae_wrong_adj - mae_wrong_raw)*100:+.2f}%p",
            delta=f"n={len(base_wrong)}",
            delta_color="off"
        )
    
    if (mae_correct_adj < mae_correct_raw) and (mae_wrong_adj < mae_wrong_raw):
        st.success("✅ Path multiplier 양쪽 샘플 모두 개선 → 채택")
    elif mae_wrong_adj < mae_wrong_raw:
        st.info("⚠️ Path multiplier가 오적중 샘플만 개선 → 위험 완화 장치로 채택")
    else:
        st.warning("❌ Path multiplier 효과 검증 실패")
    
    st.divider()
    
    # === Section 2: WS-D State Separation ===
    st.markdown("### 2️⃣ WS-D: Alpha State별 OOS 성과")
    
    states_order = ["STRONG_BEAR", "BEARISH", "NEUTRAL", "BULLISH", "STRONG_BULL"]
    state_rows = []
    
    for state in states_order:
        sub = v[v["alpha_state"] == state]
        n = len(sub)
        if n == 0:
            state_rows.append({
                "State": state, "n": 0, "Mean Real": None,
                "Sign Acc": None, "IC": None
            })
            continue
        
        mean_real = sub["realized_fwd_ret_19w"].mean()
        sign_acc = (np.sign(sub["predicted_fwd_ret_19w"]) == np.sign(sub["realized_fwd_ret_19w"])).mean()
        ic = sub["predicted_fwd_ret_19w"].corr(sub["realized_fwd_ret_19w"]) if n > 5 else None
        
        state_rows.append({
            "State": state, "n": n,
            "Mean Real": f"{mean_real*100:+.1f}%",
            "Sign Acc": f"{sign_acc*100:.1f}%",
            "IC": f"{ic:+.3f}" if ic is not None else "N/A"
        })
    
    state_df = pd.DataFrame(state_rows)
    st.dataframe(state_df, use_container_width=True, hide_index=True)
    
    # === Section 3: Acceptance Criteria 자동 판정 ===
    st.markdown("### 3️⃣ v2.19 Acceptance Criteria 판정")
    
    neutral = v[v["alpha_state"] == "NEUTRAL"]
    if len(neutral) >= 15:
        neutral_ic = neutral["predicted_fwd_ret_19w"].corr(neutral["realized_fwd_ret_19w"])
        neutral_sign = (np.sign(neutral["predicted_fwd_ret_19w"]) == np.sign(neutral["realized_fwd_ret_19w"])).mean()
        
        st.caption(f"NEUTRAL baseline (n={len(neutral)}): IC={neutral_ic:+.3f}, sign_acc={neutral_sign*100:.1f}%")
        
        verdicts = []
        for state in ["BULLISH", "BEARISH", "STRONG_BULL", "STRONG_BEAR"]:
            sub = v[v["alpha_state"] == state]
            n = len(sub)
            
            if n < 15:
                verdicts.append({
                    "State": state, "n": n,
                    "ΔIC": "—", "Δsign_acc": "—",
                    "Verdict": f"⚠️ 보류 (n<15)"
                })
                continue
            
            state_ic = sub["predicted_fwd_ret_19w"].corr(sub["realized_fwd_ret_19w"])
            state_sign = (np.sign(sub["predicted_fwd_ret_19w"]) == np.sign(sub["realized_fwd_ret_19w"])).mean()
            
            d_ic = state_ic - neutral_ic
            d_sign = state_sign - neutral_sign
            
            pass_ic = abs(d_ic) > 0.05
            pass_sign = abs(d_sign) > 0.05
            
            if pass_ic or pass_sign:
                verdict = "✅ PASS"
            else:
                verdict = "❌ FAIL"
            
            verdicts.append({
                "State": state, "n": n,
                "ΔIC": f"{d_ic:+.3f}",
                "Δsign_acc": f"{d_sign*100:+.1f}%p",
                "Verdict": verdict
            })
        
        ver_df = pd.DataFrame(verdicts)
        st.dataframe(ver_df, use_container_width=True, hide_index=True)
        
        passed = sum(1 for v_ in verdicts if "PASS" in v_["Verdict"])
        if passed >= 1:
            st.success(f"✅ {passed}개 state PASS → Phase 1 통과 가능")
        else:
            st.warning(f"❌ PASS state 없음 → 추가 데이터 필요")
    else:
        st.warning(f"NEUTRAL 샘플 부족 (n={len(neutral)}). baseline 계산 불가.")
    
    st.divider()
    
    # === Section 4: 2D Matrix ===
    st.markdown("### 4️⃣ Alpha State × Regime 2D Matrix (Sign Accuracy)")
    
    regimes = sorted([r for r in v["regime_state"].dropna().unique()])
    matrix_data = []
    
    for state in states_order:
        row = {"alpha_state": state}
        for reg in regimes:
            sub = v[(v["alpha_state"] == state) & (v["regime_state"] == reg)]
            n = len(sub)
            if n < 5:
                row[reg] = f"(n={n})"
            else:
                sa = (np.sign(sub["predicted_fwd_ret_19w"]) == np.sign(sub["realized_fwd_ret_19w"])).mean()
                row[reg] = f"{sa*100:.1f}% (n={n})"
        matrix_data.append(row)
    
    matrix_df = pd.DataFrame(matrix_data)
    st.dataframe(matrix_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # === Section 5: FOLLOW_LIGHT 진단 ===
    st.markdown("### 5️⃣ FOLLOW_LIGHT Regime 진단")
    
    fl = v[v["regime_state"] == "FOLLOW_LIGHT"]
    if len(fl) > 0:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            sa = (np.sign(fl["predicted_fwd_ret_19w"]) == np.sign(fl["realized_fwd_ret_19w"])).mean()
            st.metric("Sign Acc", f"{sa*100:.1f}%", 
                     delta=f"{(sa-0.5)*100:+.1f}%p vs 50%",
                     delta_color="inverse" if sa < 0.5 else "normal")
        with col2:
            mean = fl["realized_fwd_ret_19w"].mean()
            median = fl["realized_fwd_ret_19w"].median()
            st.metric("Mean / Median", f"{mean*100:+.1f}% / {median*100:+.1f}%")
        with col3:
            loss_ratio = (fl["realized_fwd_ret_19w"] < 0).mean()
            st.metric("Loss Ratio", f"{loss_ratio*100:.1f}%",
                     delta="59% 이상이면 함정",
                     delta_color="off")
        with col4:
            severe_loss = (fl["realized_fwd_ret_19w"] < -0.20).mean()
            st.metric(">20% 폭락 비중", f"{severe_loss*100:.1f}%")
        
        if sa < 0.45:
            st.error(f"⚠️ FOLLOW_LIGHT는 구조적 함정 — 진입 회피 권고 (sign_acc {sa*100:.1f}%)")
        elif sa < 0.50:
            st.warning(f"⚠️ FOLLOW_LIGHT 적중률 미달 — 노출 축소 권고")
        else:
            st.info(f"FOLLOW_LIGHT 적중률 정상 ({sa*100:.1f}%)")
    
    st.divider()
    
    # === Section 6: 권장 조합 ===
    st.markdown("### 6️⃣ 권장 베팅 조합 (sign_acc 기준)")
    
    combos = []
    for state in states_order:
        for reg in regimes:
            sub = v[(v["alpha_state"] == state) & (v["regime_state"] == reg)]
            n = len(sub)
            if n < 5:
                continue
            sa = (np.sign(sub["predicted_fwd_ret_19w"]) == np.sign(sub["realized_fwd_ret_19w"])).mean()
            combos.append({"alpha": state, "regime": reg, "n": n, 
                          "sign_acc": sa, "score": (sa - 0.5) * np.sqrt(n)})
    
    if combos:
        combos_df = pd.DataFrame(combos).sort_values("score", ascending=False)
        combos_df["sign_acc"] = combos_df["sign_acc"].apply(lambda x: f"{x*100:.1f}%")
        combos_df["score"] = combos_df["score"].apply(lambda x: f"{x:+.2f}")
        
        st.markdown("**Top 5 (적극 베팅)**")
        st.dataframe(combos_df.head(5)[["alpha", "regime", "n", "sign_acc", "score"]], 
                     use_container_width=True, hide_index=True)
        st.markdown("**Bottom 3 (회피)**")
        st.dataframe(combos_df.tail(3)[["alpha", "regime", "n", "sign_acc", "score"]], 
                     use_container_width=True, hide_index=True)
```

### 작업 1-3: TAB7 호출

main 함수에서 TAB 렌더링 부분 찾아서:

```python
with tabs[N]:  # N은 마지막 TAB 인덱스
    render_phase1_validation_tab(payload)
```

---

## 🎯 작업 2: Position Multiplier에 State×Regime 보정 추가

### 핵심 아이디어

기존 `regime_position_multiplier`는 regime만 보고 있음. 이를 **alpha_state × regime 조합**으로 확장.

### 위치
`build_forward_overlay_payload` 함수 내, position multiplier 계산 부분 (대략 line 1700~1800).

### 작업 2-1: state×regime override 함수 추가

기존 `compute_alpha_state_v219` 함수 아래에 추가:

```python
# ============================================================
# v2.19 Phase 1 — State × Regime Position Multiplier Override
# ============================================================

# OOS 분석(2026-05-01) 결과 기반 권장 multiplier
# 산출 근거: sign_acc → multiplier 매핑
#   sign_acc >= 70%: 1.30x (최강)
#   sign_acc 65~70%: 1.30x
#   sign_acc 60~65%: 1.20x
#   sign_acc 55~60%: 1.10x
#   sign_acc 50~55%: 1.00x (중립)
#   sign_acc 45~50%: 0.85x
#   sign_acc 40~45%: 0.60x (FOLLOW_LIGHT NEUTRAL)
#   sign_acc < 40%:  0.40x

STATE_REGIME_MULTIPLIER_TABLE = {
    # (alpha_state, regime_state): position_multiplier
    # 표본 수 5개 미만은 None (기존 multiplier 사용)
    
    # NEUTRAL × 각 regime
    ("NEUTRAL", "DEFENSIVE"):     1.10,  # n=43, sign_acc 53.5%
    ("NEUTRAL", "FOLLOW"):        0.95,  # n=37, sign_acc 48.6%
    ("NEUTRAL", "FOLLOW_LIGHT"):  0.60,  # n=57, sign_acc 42.1% ⚠️ 함정
    ("NEUTRAL", "SHRINK"):        0.90,  # n=38, sign_acc 47.4%
    ("NEUTRAL", "TRANSITION"):    0.92,  # n=163, sign_acc 48.5%
    
    # BULLISH × 각 regime  
    ("BULLISH", "DEFENSIVE"):     1.20,  # n=17, sign_acc 58.8%
    ("BULLISH", "FOLLOW"):        1.30,  # n=8, sign_acc 75.0% ⭐
    ("BULLISH", "FOLLOW_LIGHT"):  None,  # n=3, 표본 부족
    ("BULLISH", "SHRINK"):        1.30,  # n=7, sign_acc 71.4% ⭐
    ("BULLISH", "TRANSITION"):    1.30,  # n=23, sign_acc 65.2% ⭐
    
    # BEARISH/STRONG_BEAR/STRONG_BULL: 표본 부족 (n<5)
    # → 기존 alpha_state position_multiplier 사용
}


def compute_state_regime_position_multiplier(
    alpha_state: pd.Series,
    regime_state: pd.Series,
    base_position_multiplier: pd.Series,
    blend_weight: float = 0.7,
) -> Tuple[pd.Series, pd.Series]:
    """
    OOS 분석 기반으로 alpha_state × regime 조합별 position multiplier를 적용.
    
    Args:
        alpha_state: v2.19 alpha state 시리즈
        regime_state: regime state 시리즈
        base_position_multiplier: 기존 regime/alpha 기반 multiplier
        blend_weight: 0~1, OOS 권장값과 기존값을 블렌드. 1.0이면 OOS 권장값 100% 사용.
    
    Returns:
        (final_multiplier, override_flag) — 두 시리즈 반환
    """
    final = base_position_multiplier.copy()
    override_flag = pd.Series("base", index=base_position_multiplier.index, dtype="object")
    
    for idx in base_position_multiplier.index:
        a = alpha_state.get(idx, "NEUTRAL")
        r = regime_state.get(idx, None)
        
        if pd.isna(r) or pd.isna(a):
            continue
        
        rec = STATE_REGIME_MULTIPLIER_TABLE.get((a, r))
        if rec is None:
            continue
        
        base_val = base_position_multiplier.loc[idx]
        if pd.isna(base_val):
            continue
        
        # Blend: blend_weight * OOS권장 + (1-blend_weight) * 기존값
        blended = blend_weight * rec + (1 - blend_weight) * base_val
        final.loc[idx] = blended
        override_flag.loc[idx] = f"oos:{a}x{r}"
    
    return final, override_flag
```

### 작업 2-2: build_forward_overlay_payload에서 호출

기존 코드에서 `regime_position_multiplier`를 만든 직후 (대략 line 1750) 다음 추가:

```python
# === v2.19 Phase 1: State × Regime Position Multiplier Override ===
overlay_master_df["regime_position_multiplier_base"] = overlay_master_df["regime_position_multiplier"].copy()

phase1_mult, phase1_flag = compute_state_regime_position_multiplier(
    alpha_state=overlay_master_df["alpha_state"],
    regime_state=overlay_master_df["regime_state"],
    base_position_multiplier=overlay_master_df["regime_position_multiplier"],
    blend_weight=0.7,  # OOS 권장값을 70% 가중
)
overlay_master_df["regime_position_multiplier_oos"] = phase1_mult
overlay_master_df["position_multiplier_source"] = phase1_flag

# 기본 multiplier를 OOS 보정값으로 교체 (suggested_exposure 자동 반영)
overlay_master_df["regime_position_multiplier"] = phase1_mult
```

### 작업 2-3: UI에 표시

`run_main_tab` 함수의 alpha layer status 영역 아래에 추가:

```python
# === v2.19 Phase 1: Position Multiplier Override 표시 ===
if "position_multiplier_source" in overlay_master_df.columns:
    last_valid = overlay_master_df.dropna(subset=["position_multiplier_source"]).iloc[-1] if len(overlay_master_df.dropna(subset=["position_multiplier_source"])) > 0 else None
    if last_valid is not None:
        src = last_valid.get("position_multiplier_source", "base")
        base_mult = last_valid.get("regime_position_multiplier_base", None)
        oos_mult = last_valid.get("regime_position_multiplier_oos", None)
        
        if src != "base" and base_mult is not None and oos_mult is not None:
            st.info(
                f"**Phase 1 OOS 보정 적용 중**: "
                f"기존 multiplier {base_mult:.2f}x → OOS 권장 {oos_mult:.2f}x ({src})"
            )
```

### 작업 2-4: ONE FILE CSV에 신규 컬럼 추가

`enrich_cols` 리스트에 추가:
```python
"regime_position_multiplier_base",
"regime_position_multiplier_oos",
"position_multiplier_source",
```

---

## ✅ 검증 체크리스트

```
□ 1. python -c "import ast; ast.parse(open('streamlit_app.py').read())" 통과
□ 2. Streamlit 정상 실행
□ 3. TAB7 "Phase 1 Validation" 탭이 보임
□ 4. TAB7에서 6개 Section 모두 정상 표시:
   - WS-A Path multiplier 효과
   - State별 OOS 성과
   - Acceptance criteria 판정
   - 2D Matrix
   - FOLLOW_LIGHT 진단
   - 권장 베팅 조합
□ 5. MAIN 탭에 "Phase 1 OOS 보정 적용 중" info 박스 표시
   - 현재(2026-05-01) 시점은 NEUTRAL × FOLLOW_LIGHT
   - 기존 multiplier ~0.85x → OOS 권장 0.60x로 표시되어야 함
□ 6. ONE FILE CSV에 신규 3개 컬럼 포함:
   - regime_position_multiplier_base
   - regime_position_multiplier_oos
   - position_multiplier_source
□ 7. suggested_exposure가 이전보다 작아짐 (FOLLOW_LIGHT 보정 효과)
```

---

## 📋 커밋 메시지

```
feat(v2.19 Phase 1): add validation dashboard + state×regime multiplier override

Phase 1 Validation Dashboard (TAB7):
- Auto-compute v2.19 acceptance criteria each run
- WS-A: Path multiplier conditional effect (correct vs wrong subsamples)
- WS-D: State separation with NEUTRAL baseline comparison
- 2D matrix: alpha_state × regime sign accuracy
- FOLLOW_LIGHT regime diagnostic
- Recommended bet combos by sign_acc score

State×Regime Position Multiplier Override:
- Replace regime-only multiplier with alpha_state × regime table
- OOS-derived weights (sign_acc → multiplier mapping)
- 0.70 blend weight (OOS recommendation 70%, legacy 30%)
- Key adjustments:
  - NEUTRAL × FOLLOW_LIGHT: 0.60x (was ~0.85x) — defense against trap
  - BULLISH × FOLLOW: 1.30x (was 1.00x) — capitalize on edge
  - BULLISH × SHRINK/TRANSITION: 1.30x — high conviction
- Add diagnostic columns to ONE FILE CSV

Resolves: FOLLOW_LIGHT 41% sign_acc trap (median realized -13.5%)
Resolves: BULLISH × FOLLOW combo underutilized (75% sign_acc)
```

---

## ⚠️ 주의사항

### 1. 기존 동작 보존
- 기존 `regime_position_multiplier` 값은 `regime_position_multiplier_base`에 백업
- OOS 보정값은 `regime_position_multiplier_oos`에 저장
- 메인 컬럼은 OOS 보정값으로 교체 (자동 효과 적용)

### 2. 표본 부족 케이스 처리
- `STATE_REGIME_MULTIPLIER_TABLE`에서 None인 항목은 기존 multiplier 사용
- BEARISH/STRONG_BEAR/STRONG_BULL은 OOS 표본 부족으로 미반영
- BULLISH × FOLLOW_LIGHT (n=3)도 미반영

### 3. blend_weight 조정 가능성
- 초기 `blend_weight=0.7` (OOS 권장 70%)
- 더 보수적으로 가려면 0.5
- 더 적극적으로 가려면 0.9
- 코드 한 줄만 수정

### 4. TAB7 성능
- 매 실행 시 OOS 분석 재계산 → 데이터가 늘어나면 느려질 수 있음
- 필요시 `@st.cache_data` 추가 가능

---

## 🚀 작업 순서

1. **작업 1-1**: TAB 정의에 "Phase 1 Validation" 추가
2. **작업 1-2**: `render_phase1_validation_tab` 함수 추가
3. **작업 1-3**: TAB7 호출 추가
4. **작업 2-1**: `STATE_REGIME_MULTIPLIER_TABLE` + `compute_state_regime_position_multiplier` 함수 추가
5. **작업 2-2**: `build_forward_overlay_payload`에서 호출
6. **작업 2-3**: UI에 보정 표시 추가
7. **작업 2-4**: enrich_cols 업데이트
8. **검증**: syntax check + streamlit 실행 확인
9. **PR 생성 + 머지**

---

## 🎯 작업 완료 후 즉각 효과

### 현재 (2026-05-01) 시점 변화 예상
- **이전**: regime=FOLLOW_LIGHT, alpha=NEUTRAL → suggested_exposure 70%
- **이후**: 동일 조건 → suggested_exposure 약 50%
- **이유**: 이 조합의 OOS sign_acc가 42% (50% 미만 → 노출 축소)

이 변화는 **검증된 통계**에 기반한 것이며, 갭 리뷰의 "정량적 acceptance criteria" 원칙을 처음으로 코드에 적용한 사례입니다.
