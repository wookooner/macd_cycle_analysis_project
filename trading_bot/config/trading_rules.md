# BTC MACD 사이클 트레이딩 규칙 (v1.0)

**근거:** 9년(2017~2025) BTC 데이터, 6,108개 완전 4레벨 체인 통계 분석

---

## 1. 시장 구조 이해

### 사이클 방향 체인 (n_up)
market_state.chain 에서 n_up (0~4) = 현재 UP 방향인 타임프레임 수.

| n_up | 콤보 예시 | 포지션 방향 | 실현 WR | 특이사항 |
|:---:|:---|:---:|:---:|:---|
| 4 | UUUU | LONG | 93.3% | 풀사이즈, 조건 불문 |
| 3 | DUUU, UDUU, UUDU | LONG | ~75% | CVD 반드시 확인 |
| 2 | DDUU, UUDD 등 | 조건부 | ~50% (필터前) | 트리플 필터 필수 |
| 1 | DDDU, DDUD, DUDD | SHORT | ~75% | CVD 반드시 확인 |
| 0 | DDDD | SHORT | ~93% | 풀사이즈, 조건 불문 |

---

## 2. HARD RULES (절대 규칙 — 예외 없음)

### RULE 1: n_up 방향 위반 금지
- n_up=4 또는 3 → SHORT/REVERSE_TO_SHORT 절대 금지
- n_up=0 또는 1 → LONG/REVERSE_TO_LONG 절대 금지
- 위반 시 실적: 전건 손실 (통계 11건 100% 손실)
- **의심되면 HOLD. 방향 모호시 HOLD.**

### RULE 2: 짧은 사이클 진입 금지
- market_state.timeframes["1h"].duration < 5 → HOLD
- 확인 비용이 기대 수익의 ~99%를 잠식, 실현 WR 47.5%
- 이전 사이클 duration ≤ 4였다면 이번도 건너뛰기 (whipsaw 73.8%)

### RULE 3: 고위험 구간 진입 금지
Danger Score = 아래 6인자 합산 (0~13점):
1. |ppo_hist| (analysis_snapshot): <20→+3, <40→+2, <60→+1
2. |dist_MA25| = |price - ma_25| / ma_25 × 100: >3%→+3, >2%→+2, >1.5%→+1
3. 4h position (timeframes["4h"].position_pct): >0.8→+2, >0.5→+1
4. 위험 콤보 (UUUD/DDDU→+2, DUUD/UDDU→+1)
5. 4h 미정렬 + 중반(0.3~0.7): →+1

- Danger ≥ 7 (RED) → **HOLD**
- Danger 5~6 (ORANGE) → size_pct 50% 감소

### RULE 4: 4h/1h 방향 미정렬 시 사이즈 감소
- chain.alignment_4h_1h == False → size_pct 50% 감소
- 4h 정렬 진입 승률 64% / 미정렬 15% (실전 데이터)

### RULE 5: CVD 반대 시 진입 금지
- market_state.timeframes["1h"].analysis_snapshot.cvd 확인
- n_up=3 + CVD ≤ 0 → HOLD (WR 92.6%→60.6% 급락)
- n_up=1 + CVD ≥ 0 → HOLD (WR 90.4%→42.3% 급락)
- n_up=4/0: CVD 방향으로 사이즈 조정 가능 (CVD 상위50% → 수익 2배)

### RULE 6: 청산 후 쿨다운
- 포지션 청산 직후 30분간 신규 진입 금지 (시스템 자동 적용)
- Whipsaw 피해 방지 (-309 USDT, 31일간 실전 데이터)

### RULE 7: 일일 거래 한도
- 시스템이 하루 3건 초과 시 자동 차단 (5건 이상 시 WR 39%)

### RULE 8: 연속 손실 후 중단
- 3연패 후 당일 거래 중단 (시스템 자동 적용)

### RULE 9: 초중반 감속에서 조기 청산 금지
- 사이클 position_pct ≤ 0.6 구간의 ppo_hist 감속: 100% 재가속
- 5연속 감속도 92% 재가속 → **보유 유지**
- 단, Δhist < -50 이면 재가속 확정이 아닌 가짜 바운스 → 즉시 청산

---

## 3. 진입 의사결정 트리

```
Step 1: n_up 확인
  n_up=4 → ENTER_LONG (dur≥5이면 풀사이즈)
  n_up=0 → ENTER_SHORT (dur≥5이면 풀사이즈)
  n_up=3 → Step 2로
  n_up=1 → Step 2로 (방향 반전: SHORT 기준으로 판단)
  n_up=2 → Step 3으로

Step 2: n_up=3(LONG) / n_up=1(SHORT) 조건부 진입
  CVD 방향 불일치 → HOLD
  4h position > 0.8 (말기) → 사이즈 50% 또는 HOLD
  1h MACD음수(ppo<0) + n_up=3 → 최강 구간 (WR 89.7%)
  1h MACD양수(ppo>0) + n_up=1 → 최강 구간 (SHORT 기준)
  그 외 → 정상 진입 (RULE 3~4 적용 후 사이즈 결정)

Step 3: n_up=2 트리플 필터 (모두 충족 시에만 진입)
  LONG 조건: CVD > 0 AND dist_MA25 < 0 AND duration ≥ 8 → ENTER_LONG (WR 89.3%)
  SHORT 조건: CVD < 0 AND dist_MA25 > 0 AND duration ≥ 8 → ENTER_SHORT (WR 84.0%)
  미충족 → HOLD
```

---

## 4. 포지션 사이징

기본 원칙: **AI가 size_pct(0~100)를 반환하면 시스템이 레버리지 고려 후 실제 수량 계산.**

| 조건 | size_pct 기준 |
|:---|:---:|
| n_up=4/0, 조건 양호 | 80~100 |
| n_up=3/1, CVD 정렬, 위험 낮음 | 60~80 |
| n_up=3/1, 위험 보통 | 40~60 |
| n_up=2, 트리플 필터 통과 | 40~60 |
| 4h 미정렬 | ×0.5 추가 적용 |
| Danger ORANGE (5~6) | ×0.5 추가 적용 |

---

## 5. 청산 조건

### 적극 청산 (즉시 CLOSE)
1. **사이클 방향 전환**: 1h 사이클이 반대로 전환되고 n_up이 ≥2 변화 (캐스케이드 붕괴)
   - DDUU → DDDD: 즉시 LONG 청산 (누적5 = -6.46%)
   - UUDD → UUUU: 즉시 SHORT 청산 (누적5 = +6.13%)
2. **Δhist < -50 (UP사이클)**: 가짜 반등 확정 → 즉시 청산

### 보유 유지
1. 사이클 position_pct ≤ 0.6 구간의 감속: 무시 (100% 재가속)
2. 단순 n_up 1단계 변화 (예: 4→3): 포지션 재평가만, 즉시 청산 아님

### 능동 관리 (선택적)
- 1h DOWN 전환 시 일부 청산 → 1h UP 재전환 시 재진입 (+2pp 개선)

---

## 6. 진입 타이밍

- **4h 사이클 초반(position_pct 0~0.2)**: 최강 수익 구간 (p<0.000001)
- **1캔들 확인 즉시 진입**: 모든 n_up에서 기대값 최대 (2캔들 대기는 WR 하락)
- n_up=4/0: 즉시 진입 WR 95~97%
- n_up=3/1: 즉시 진입 WR 84~85%

---

## 7. 회피 콤보 목록 (n_up=2 이하에서도 진입 금지)

| 콤보 | 방향 시도 | 실현 WR | 사유 |
|:---:|:---:|:---:|:---|
| UUUD | SHORT | 38.6% | 상위3 UP, 1h만 DOWN → 구조적 실패 |
| DDDU | LONG | 40.5% | 상위3 DOWN, 1h만 UP → 구조적 실패 |
| UDDU | LONG | 52.6% | 동전던지기 |
| DUUD | SHORT | 48.8% | 동전던지기 |

---

## 8. JSON 응답 형식

반드시 아래 형식으로만 응답:

```json
{
  "action": "HOLD | ENTER_LONG | ENTER_SHORT | CLOSE_LONG | CLOSE_SHORT | REVERSE_TO_LONG | REVERSE_TO_SHORT",
  "direction": "LONG | SHORT | NEUTRAL",
  "size_pct": 0,
  "confidence": 0.0,
  "reasoning": "적용된 규칙과 핵심 지표 수치 명시",
  "sl_pct": 2.0,
  "tp_pct": 4.0,
  "alerts": []
}
```

**reasoning 필수 포함 항목:** n_up 값, combo, CVD 방향, ppo_hist 값, 4h position_pct, Danger Score 계산 근거, 적용된 RULE 번호.

**HOLD 판단 시에도 reasoning 에 이유 명시.**
