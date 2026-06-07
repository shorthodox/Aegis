# AEGIS-1 Master Forensic Report

**Symbol:** GMX/USDT  |  **Generated:** 2026-06-07 16:36:04

---

> **Audit scope:** 15 sections covering Training Pipeline, Meta Model,
> HMM Regime Engine, LSTM Temporal Engine, Live Signal Engine,
> Risk Engine, Execution Filters, Portfolio Engine, Drift Monitor,
> and Post-Trade Performance.
>
> Every conclusion cites supporting metrics.
> Where evidence is weak, uncertainty is explicitly stated.

---


## ❓ The Three Questions

> These three questions answer what matters for P&L.
> Every finding cites the exact file and line where the problem lives.

### Q1 — Why Is Precision Lower Than It Should Be?

> Measured precision is 31.5% — estimated 7.8pp below achievable ceiling of ~39.3%. Primary contributors: HOLD over-representation in labels (−2.5pp), Barrier skew suppresses BUY labels (−1.9pp), Brier score above target (−2.0pp)

#### Precision Waterfall

```
Measured holdout precision :  31.5%

  [✅ FIXED       ]  −2.5pp prec    HOLD over-representation in labels
  [✅ FIXED       ]  −1.9pp prec    Barrier skew suppresses BUY labels
  [🔴 ACTIVE      ]  −2.0pp prec    Brier score above target
  [🔴 ACTIVE      ]  −1.0pp prec    Temperature T=1.209 — overconfidence
  [🔴 ACTIVE      ]  −0.4pp prec    Gate blocking signals that would have won

Achievable precision       :  ~39.3%  (+7.8pp precision)
Recall deficit (BUY side)  :  −2.9pp  (signals not firing)
```

### Q2 — Where Exactly Is Each Problem?

| Issue | File | Lines | Symbol | −Prec | −Recall |
|-------|------|-------|--------|-------|---------|
| Brier score above target | `scripts/retrain_model.py` | 1909-1916 | `_hold_w = clip(_n_dir×0.5 / _n_hold` | 2.0pp | 0.0pp |
| Temperature T=1.209 — overconfidence | `scripts/retrain_model.py` | 1909-1916 | `_hold_w = clip(_n_dir×0.5 / _n_hold` | 1.0pp | 0.0pp |
| Gate blocking signals that would have won | `scripts/retrain_model.py` | 1909-1916 | `_hold_w = clip(_n_dir×0.5 / _n_hold` | 0.4pp | 0.0pp |

**1. Brier score above target**
> 📍 `scripts/retrain_model.py:1909-1916` — `_hold_w = clip(_n_dir×0.5 / _n_hold, 0.10, 0.60)`
> HOLD bars always have meta_y=0 (primary proposes BUY/SELL but true label=HOLD). With HOLD=66%, 60% of meta training targets are al…

**2. Temperature T=1.209 — overconfidence**
> 📍 `scripts/retrain_model.py:1909-1916` — `_hold_w = clip(_n_dir×0.5 / _n_hold, 0.10, 0.60)`
> HOLD bars always have meta_y=0 (primary proposes BUY/SELL but true label=HOLD). With HOLD=66%, 60% of meta training targets are al…

**3. Gate blocking signals that would have won**
> 📍 `scripts/retrain_model.py:1909-1916` — `_hold_w = clip(_n_dir×0.5 / _n_hold, 0.10, 0.60)`
> HOLD bars always have meta_y=0 (primary proposes BUY/SELL but true label=HOLD). With HOLD=66%, 60% of meta training targets are al…

### Q3 — What Is The Best Fix Right Now?

✅ **2 issues already applied** in the codebase (expected gain: +5.3pp once retrained).

**Highest-ROI remaining fix: Brier score above target** (expected +2.0pp):
> 📍 `scripts/retrain_model.py:1909-1916`
> Lower _hold_w floor from 0.10 to 0.05 OR exclude HOLD bars from meta training.

#### BUY Side Gate Trace

🔴 **BUY DISABLED — root cause at Gate: 2. pick_threshold_by_side(BUY) can qualify**

✅ **1. Primary model generates BUY labels**
   - `scripts/retrain_model.py:849-928 (create_triple_barrier_labels)`
   - Check: `BUY label count > 0 in training data`
   - Value: 88 BUY proposals / 440 total directional
   - PASS — primary fires BUY on some bars.

❌ **2. pick_threshold_by_side(BUY) can qualify**
   - `scripts/retrain_model.py:1363-1397`
   - Check: `MAX_SIDE_COVERAGE=0.35×pool(70)=24 ≥ min_fires=35`
   - Value: 24 max fires vs 35 required
   - FAIL (FIXED) — 24 < 35. Deadlock: every quantile rejected before precision is checked. Fix: MAX_SIDE_COVERAGE→0.35 + adaptive effective_min_fires.

❌ **3. hit_buy=True (OOF precision clears target)**
   - `scripts/retrain_model.py:1996-2004`
   - Check: `pick_threshold_by_side(side=2).hit_target → stored as tradeable_buy`
   - Value: tradeable_buy in sidecar = False
   - FAIL — hit_buy=False because Gate 2 deadlock blocked threshold qualification.

✅ **4. buy_fire mask fires BUY holdout signals**
   - `scripts/retrain_model.py:2169-2174`
   - Check: `buy_fire = (meta_prob_h ≥ max(thr_buy, rank_thr)) & (prop_h==2)`
   - Value: buy_h_n = 17 holdout BUY signals fired
   - PASS — 17 BUY holdout trades.

✅ **5. tradeable_buy_holdout = True**
   - `scripts/retrain_model.py:2288-2292`
   - Check: `hit_buy AND buy_h_n > 0 AND buy_h_prec ≥ 0.50`
   - Value: buy_h_n=17, buy_win_rate=70.6%
   - PASS — 17 trades, 70.6% WR.

#### Meta Gate Audit

🔴 **Gate status: HURTING**  (lift: -22.3pp)

| Metric | Value |
|--------|-------|
| Gated-in precision | 31.5% |
| Blocked signals win rate | 53.8% |
| Precision lift from gate | -22.3pp |
| OOF → Holdout gap | +51.8pp |
| thr_buy / thr_sell | 85.096 / 85.017 |
| Blocked signals | 0 (223 would-win / 192 would-lose) |

> ⚠ Gate is DESTROYING 22.3pp of precision. Blocked signals (53.8%) would have beaten gated (31.5%). Meta model is anti-selective. | OOF overfit warning: dev_prec (83.3%) exceeds holdout (31.5%) by 51.8pp.


---

## Section 15 — Executive Summary

**Symbol:** GMX/USDT  |  **Audit:** 2026-06-07 16:36  |  **Confidence Level:** MEDIUM — based on 47 holdout signals; widen to 200+ for HIGH

**Current:** Precision=31.5%  Sharpe=9.79
**Expected after fixes:** Precision≈37.3%  (+5.8pp)

### Top 5 Problems

1. 🟡 **Severe Class Imbalance (HOLD dominates)** — Score: 66/100
   > HOLD=66.8% of labels. Meta model sees 60% zero-labels → calibration distorted.

2. 🟡 **HMM Regime Collapse** — Score: 62/100
   > Max state concentration=100.0%. HMM assigning most bars to one state.


### Top 5 Fixes

1. **Severe Class Imbalance (HOLD dominates)**
   → base_vol_threshold→0.72, symmetric BARRIER skews. Already applied.

2. **HMM Regime Collapse**
   → Re-train HMM (random_state=42 already set). Verify 9 regime features are non-degenerate.


---

## Section 1 — Model Health

| Metric | Value | Status |
|--------|-------|--------|
| CV Accuracy (OOF) | 67.3% | ✓ |
| Dev OOF Precision | 83.3% | ✓ |
| Holdout Precision | 31.5% | ✗ |
| Holdout Coverage | 8.3% | ✓ |
| 95% CI Precision | [23.6%, 40.7%] | — |
| OOF→Holdout Gap | -51.8% | ⚠ degradation |
| Holdout Fired | 111 trades | ✓ |
| SELL Win Rate | 88.5% (26 trades) | ✓ |
| BUY Win Rate | 70.6% (17 trades) | ✓ |
| Sharpe (annualised) | 9.79 | ✓ |
| Max Drawdown | 6845876124.46% | ✗ |
| Profit Factor | 3.00 | ✓ |
| Kelly Fraction | 25.0% | — |
| Expectancy/Trade | +0.4931% | ✓ |
| Meta gate optimizer profile | present | ✓ |
| Optimizer-selected gate | SIDE_EDGE_PERCENTILE | ✓ |
| Optimizer threshold match | YES | ✓ |
| Meta gate summary count | 89 symbols | ✓ |
| Statistical Sig. | p=0.0001 (z=-3.89) | ✓ significant |

### Class Distribution
- HOLD: **66.8%** — ⚠ severe imbalance
- SELL: **19.6%**
- BUY:  **13.6%** — ⚠ minority class

### Issues Detected
- **WARNING** — Class imbalance: 66.8% HOLD labels biases model toward neutrality.


---

## Section 2 — Feature Forensics

⚠ feature_health.json not found



---

## Section 3 — Signal Generation Forensics

**Data window:** 8,736 bars

### All-Bar Prediction Breakdown

| Predicted | Count | Raw Precision |
|-----------|-------|---------------|
| BUY  | 88  | 87.5%  |
| SELL | 352 | 81.0% |
| HOLD | 8,296 | — |

### Signal Rejection Funnel

```
Generated (directional):       440  (100%)
Blocked by Meta Gate:      -  415  (94%)
Blocked by Quality (<55):  -   35
Blocked by HMM:            -   22
Blocked by Confluence:     -   17
Blocked by Fake Breakout:  -   13
Blocked by Portfolio Guard:-    8
Blocked by Safe Mode:      -    4
Blocked by Drift:          -    8
Blocked by Cooldown:       -    4
─────────────────────────────────────
Estimated Executed:             0  (0.0%)
```

**BUY side:** ✗ DISABLED  |  **SELL side:** ✗ DISABLED


---

## Section 4 — Opportunity Cost Analysis

Average hold-signal realized return: **0.6881%/bar**

### Time-to-TP (Upper Barrier) Distribution

| Horizon | % of BUY signals that would have hit TP |
|---------|------------------------------------------|
| 6h | 74.7% |
| 12h | 91.2% |
| 18h | 97.9% |
| 24h | 100.0% |
| 48h | 100.0% |

Median time-to-TP: **3 bars** (3h)

### Opportunity Cost by Filter

| Filter | Blocked | Would Win | Would Lose | Win Rate | Opp. Cost |
|--------|---------|-----------|------------|----------|-----------|
| meta_gate | 415 | 223 | 192 | 53.8% | +285.56% |
| quality | 35 | 21 | 14 | 61.5% | +24.08% ⚠ |
| hmm | 22 | 12 | 10 | 57.6% | +15.14% ⚠ |
| confluence | 17 | 9 | 8 | 54.5% | +11.70% |
| fake_breakout | 13 | 7 | 6 | 60.0% | +8.95% ⚠ |
| portfolio | 8 | 4 | 4 | 57.6% | +5.50% ⚠ |


---

## Section 5 — Meta Model Forensics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| ECE (before cal.) | 0.0814 | <0.10 | ✓ |
| ECE (after cal.)  | 0.0000 | <0.10 | ✓ |
| Brier Score | 0.3328 | <0.25 | ✗ |
| Cal. Temperature | 1.2094 | ~1.0 | ⚠ model overconfident |
| Calibration Type | uncalibrated | isotonic | — |

### Confidence Bucket Analysis (Estimated)

| Bucket | Est. Win Rate | Gap | Status |
|--------|---------------|-----|--------|
| 50-60% | 52.0% | -0.03 | ✓ |
| 60-70% | 60.0% | -0.05 | ⚠ overconfident |
| 70-80% | 31.5% | -0.43 | ⚠ overconfident |
| 80-90% | 41.5% | -0.43 | ⚠ overconfident |
| 90-100% | 80.0% | -0.15 | ⚠ overconfident |

**Confidence inflation detected:** No
**Recommended calibrator (for 30 dev samples):** `isotonic`


---

## Section 6 — HMM Regime Forensics

⚠ HMM model not available.



---

## Section 7 — LSTM Forensics

⚠ LSTM meta not found



---

## Section 8 — Quality Engine Forensics

**Verdict:** VALID — higher quality scores predict better outcomes

| Quality Bucket | Trades | Precision | Expectancy | P.Factor |
|---------------|--------|-----------|------------|---------|
| 0-20 | 0 | 0.0% | +0.0000% | 0.00 |
| 20-40 | 113 | 77.0% | +0.0000% | 2.79 |
| 40-60 | 191 | 81.2% | +0.0000% | 4.06 |
| 60-80 | 172 | 81.4% | +0.0000% | 4.18 |
| 80-100 | 116 | 89.7% | +0.0000% | 8.05 |

**Monotone precision:** ✓ YES — quality is predictive
**Paper trading:** 19 trades, 73.7% WR


---

## Section 9 — Drift Monitor Forensics

**Overall Drift Status:** 🟢 **OK**

| Drift Type | Classification | Detail |
|------------|---------------|--------|
| Feature Drift | 🟢 OK | 0 CRITICAL / 0 DEGRADED / 0 total |
| Confidence Drift | 🟡 WARNING | T=1.209 |
| Prediction Drift | 🔴 CRITICAL | OOF vs holdout gap: +51.80pp |

**Estimated precision loss from feature drift:** ~0.0pp


---

## Section 10 — Portfolio Forensics

**Open positions:** 0/6  |  **Effective leverage:** 0.38×  |  **HHI (concentration):** 0.064

| Symbol | Capital Allocation |
|--------|-------------------|
| VET/USDT | 10.5% |
| ATOM/USDT | 10.5% |
| ADA/USDT | 5.3% |
| SEI/USDT | 5.3% |
| SUI/USDT | 5.3% |
| UNI/USDT | 5.3% |
| FIL/USDT | 5.3% |
| KAVA/USDT | 5.3% |
| SAND/USDT | 5.3% |
| XLM/USDT | 5.3% |
| ARB/USDT | 5.3% |
| STX/USDT | 5.2% |
| DOT/USDT | 5.2% |
| ENA/USDT | 5.2% |
| NEAR/USDT | 5.2% |
| ZEC/USDT | 5.2% |
| THETA/USDT | 5.2% |

**Hidden leverage:** ✓ NO  |  **Over-concentration:** ✓ NO


---

## Section 11 — Risk Engine Forensics

| Metric | Value | Assessment |
|--------|-------|------------|
| ATR Multiplier | 3.0× | ⚠ |
| Win Rate | 73.7% | ✓ |
| Avg Win / Avg Loss | 2.078% / 1.178% | — |
| R:R Ratio | 1.76 | ✓ favourable |
| Kelly Fraction | 58.8% | ⚠ overbetting |
| Avg R-Multiple | 1.33R | ✓ |
| Risk of Ruin | 0.0000% | ✓ low |
| Holdout Sharpe | 9.79 | ✓ |
| Holdout Max DD | 6845876124.46% | ✗ |
| Stop Assessment | TARGETS TOO CLOSE | — |


---

## Section 12 — Live Execution Forensics

**Closed:** 19  |  **Open:** 0  |  **Avg hold:** 0.6h  |  **Avg PnL:** +1.221%

**Confidence discriminates wins from losses:** ✗ NO (WIN conf=0.645 vs LOSS conf=0.636)

### Best Trades
- **ZEC/USDT** BUY  PnL=+10.31%  conf=0.785  exit=TP1_HIT
- **THETA/USDT** BUY  PnL=+3.87%  conf=0.701  exit=TP1_HIT
- **NEAR/USDT** BUY  PnL=+3.15%  conf=0.713  exit=TP1_HIT

### Worst Trades
- **ENA/USDT** SELL  PnL=-3.40%  conf=0.749  exit=SL_HIT
- **VET/USDT** BUY  PnL=-0.79%  conf=0.620  exit=SL_HIT
- **ADA/USDT** BUY  PnL=-0.75%  conf=0.595  exit=SL_HIT

### Exit Reasons
| Reason | Count |
|--------|-------|
| TP1_HIT | 14 |
| SL_HIT | 5 |


---

## Section 13 — Root Cause Engine

**2 root causes identified.**  Combined top-5 impact score: **128/500**

| Rank | Cause | Category | Score | Source | Evidence |
|------|-------|---------|-------|--------|---------|
| 1 | 🟡 **Severe Class Imbalance (HOLD dominates)** | Training Quality | 66/100 | `scripts/retrain_model.py:836` ✅ FIXED | HOLD=66.8% of labels. Meta model sees 60% zero-labels → calibration di… |
| 2 | 🟡 **HMM Regime Collapse** | HMM Failure | 62/100 | — 🔴 ACTIVE | Max state concentration=100.0%. HMM assigning most bars to one state.… |

### Fixes

**1. Severe Class Imbalance (HOLD dominates)**
> 📍 `scripts/retrain_model.py:836` — `base_vol_threshold = 0.80`
> base_vol_threshold→0.72, symmetric BARRIER skews. Already applied.

**2. HMM Regime Collapse**
> Re-train HMM (random_state=42 already set). Verify 9 regime features are non-degenerate.



---

## Section 14 — Automated Improvement Engine

**Base precision:** 31.5%  →  **Expected precision (all fixes):** 37.3%  (+5.8pp)

| # | Action | Prec Gain | Recall Gain | Profit Gain | Confidence | Effort |
|---|--------|-----------|-------------|-------------|------------|--------|
| 1 | Enable BUY side (fix directional asymmetry) | +0.0pp | +8.0pp | +5.0pp | HIGH | MEDIUM |
| 2 | Remove / normalise top-10 drifted features | +3.0pp | +1.5pp | +2.4pp | MEDIUM | LOW |
| 3 | Improve meta model calibration | +2.5pp | +0.5pp | +2.0pp | HIGH | LOW |
| 4 | Redesign triple-barrier labels (reduce HOLD%) | +1.5pp | +4.0pp | +3.0pp | MEDIUM | MEDIUM |
| 5 | Extend lookahead for low-ER tokens | +1.0pp | +2.0pp | +1.5pp | MEDIUM | LOW |
| 6 | Regime-specific meta thresholds | +1.5pp | +1.0pp | +2.5pp | MEDIUM | LOW |
| 7 | Retrain meta model on 60-symbol fleet data | +2.0pp | +0.5pp | +2.5pp | HIGH | HIGH |


---

## Section 16 — Meta Gate Ranking Audit

| Metric | Value |
|--------|-------|
| Selected signals | 111 |
| Rejected signals | 1223 |
| Selected precision | 31.5% |
| Rejected precision | 25.9% |
| Meta gate lift (precision) | +5.6% |
| Selected expectancy | +0.493% |
| Rejected expectancy | +0.439% |
| Selected Sharpe | +30.74 |
| Rejected Sharpe | +19.16 |

**Verdict:** ✅ HELPFUL — Gate selects higher-precision signals than rejected


---

## Section 17 — Hold Pollution Audit

| Strategy | Hold Weight | Brier | PF | Sharpe | Prec | Lift | Notes |
|----------|-------------|-------|----|----|------|------|-------|
| A_current 🔴 CURRENT | 1.00 | 0.330 | 1.20 | 0.45 | 59.0% | -0.02 | Baseline — no mitigation |
| B_reduced  | 0.15 | 0.310 | 1.35 | 0.58 | 62.0% | +0.03 | Partial HOLD downweight — recommended |
| C_excluded ✅ BEST | 0.00 | 0.300 | 1.40 | 0.62 | 64.0% | +0.05 | Total HOLD exclusion — most aggressive |

**Current Strategy Score:** 0.190
**Best Strategy Score:** 0.820
**Potential Improvement:** +0.630

**Recommendation:** Switch from A_current to C_excluded (+0.630 score)


---

## Section 18 — Regime Threshold Audit

**Regime Summary:** 7 enabled, 0 disabled (0.0% disability rate)

| Regime | BUY OK | SELL OK | BUY Thr | SELL Thr | Status | Est. Prec | Est. PF |
|--------|--------|---------|---------|----------|--------|-----------|---------|
| ACCUMULATION         | ✅ | ✅ | 85.1 | 85.0 | ✅ ENABLED | 60.0% | 1.25 |
| CHOPPY               | ✅ | ✅ | 85.1 | 85.0 | ✅ ENABLED | 60.0% | 1.25 |
| COMPRESSION          | ✅ | ✅ | 85.1 | 85.0 | ✅ ENABLED | 60.0% | 1.25 |
| DISTRIBUTION         | ✅ | ✅ | 85.1 | 85.0 | ✅ ENABLED | 60.0% | 1.25 |
| TRENDING_BEAR        | ✅ | ✅ | 85.1 | 85.0 | ✅ ENABLED | 60.0% | 1.25 |
| TRENDING_BULL        | ✅ | ✅ | 85.1 | 85.0 | ✅ ENABLED | 60.0% | 1.25 |
| VOLATILE_EXPANSION   | ✅ | ✅ | 85.1 | 85.0 | ✅ ENABLED | 60.0% | 1.25 |

**Verdict:** ✅ MODERATE — Selective regime blocking


---

## Section 19 — Deep GMX/USDT vs BTC/ETH Comparison

| Metric | SOL | BTC | ETH | SOL vs BTC |
|--------|-----|-----|-----|-----------|
| Meta Threshold | 79.5 | 82.4 | 82.9 | -2.9 |
| Tradeable BUY | False | True | False | — |
| Tradeable SELL | False | True | False | — |
| Holdout Precision | 37.4% | 66.0% | 45.0% | -28.6% |
| Win Rate (PnL) | 48.0% | 72.0% | 52.0% | — |
| Regime Disability | 50% | 20% | 60% | +30% |
| Calibration T | 0.888 | 0.920 | 0.950 | — |

### Top Discriminators (SOL vs BTC)

1. **Regime disability** — gap: +30.00
2. **Meta threshold** — gap: -2.90
3. **Holdout precision** — gap: -0.29

**Root Cause Hypothesis:** SOL fails on Regime disability (gap: 30.00)


---

## Section 20 — AEGIS Gate Lift Engine

**Gate Lift = Selected Precision − Rejected Precision**

| Metric | Value |
|--------|-------|
| Gate Lift (pp) | +5.6% |
| Selected signals | 111 |
| Rejected signals | 1223 |
| Gate coverage | 8.3% |
| Status | HELPFUL (> +1pp) |


---

## Section 21 — AEGIS Gate Self-Preservation (Phase 2)

| Metric | Value |
|--------|-------|
| Gate Status | HELPFUL |
| Trust Score | 55/100 |
| Recommended Action | USE_META_GATE |

**Recommendation:** Unknown status


---

## Section 22 — AEGIS BTC Difference Engine (Phase 7)

Comparing this token against BTC baseline:

| Metric | BTC | This Token | Gap |
|--------|-----|-----------|-----|
| Gate Lift Difference | — | — | TBD (requires BTC comparison data) |
| Precision Gap | — | — | TBD |
| Profit Factor Gap | — | — | TBD |


---

## Section 23 — AEGIS Token Profile (Phase 3)

| Metric | Value |
|--------|-------|
| Precision Target | 56.2% |
| Actual Precision | 31.5% |
| Gap | -24.6% |
| Coverage | 1.9% |
| Gating Strategy | ADAPTIVE_PER_REGIME |
| Gate Trust Score | 55/100 |
| Verdict | 🔴 SIGNIFICANTLY BELOW TARGET |


---
