# AEGIS-1 Master Forensic Report

**Symbol:** FIL/USDT  |  **Generated:** 2026-06-07 16:36:02

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

> Measured precision is 34.0% — estimated 5.4pp below achievable ceiling of ~39.4%. Primary contributors: Brier score above target (−2.0pp), HOLD over-representation in labels (−1.7pp), Barrier skew suppresses BUY labels (−1.3pp)

#### Precision Waterfall

```
Measured holdout precision :  34.0%

  [🔴 ACTIVE      ]  −2.0pp prec    Brier score above target
  [✅ FIXED       ]  −1.7pp prec    HOLD over-representation in labels
  [✅ FIXED       ]  −1.3pp prec    Barrier skew suppresses BUY labels
  [🔴 ACTIVE      ]  −0.4pp prec    Gate blocking signals that would have won

Achievable precision       :  ~39.4%  (+5.4pp precision)
Recall deficit (BUY side)  :  −2.0pp  (signals not firing)
```

### Q2 — Where Exactly Is Each Problem?

| Issue | File | Lines | Symbol | −Prec | −Recall |
|-------|------|-------|--------|-------|---------|
| Brier score above target | `scripts/retrain_model.py` | 1909-1916 | `_hold_w = clip(_n_dir×0.5 / _n_hold` | 2.0pp | 0.0pp |
| Gate blocking signals that would have won | `scripts/retrain_model.py` | 1909-1916 | `_hold_w = clip(_n_dir×0.5 / _n_hold` | 0.4pp | 0.0pp |

**1. Brier score above target**
> 📍 `scripts/retrain_model.py:1909-1916` — `_hold_w = clip(_n_dir×0.5 / _n_hold, 0.10, 0.60)`
> HOLD bars always have meta_y=0 (primary proposes BUY/SELL but true label=HOLD). With HOLD=66%, 60% of meta training targets are al…

**2. Gate blocking signals that would have won**
> 📍 `scripts/retrain_model.py:1909-1916` — `_hold_w = clip(_n_dir×0.5 / _n_hold, 0.10, 0.60)`
> HOLD bars always have meta_y=0 (primary proposes BUY/SELL but true label=HOLD). With HOLD=66%, 60% of meta training targets are al…

### Q3 — What Is The Best Fix Right Now?

✅ **2 issues already applied** in the codebase (expected gain: +3.6pp once retrained).

**Highest-ROI remaining fix: Brier score above target** (expected +2.0pp):
> 📍 `scripts/retrain_model.py:1909-1916`
> Lower _hold_w floor from 0.10 to 0.05 OR exclude HOLD bars from meta training.

#### BUY Side Gate Trace

🔴 **BUY DISABLED — root cause at Gate: 3. hit_buy=True (OOF precision clears target)**

✅ **1. Primary model generates BUY labels**
   - `scripts/retrain_model.py:849-928 (create_triple_barrier_labels)`
   - Check: `BUY label count > 0 in training data`
   - Value: 126 BUY proposals / 467 total directional
   - PASS — primary fires BUY on some bars.

✅ **2. pick_threshold_by_side(BUY) can qualify**
   - `scripts/retrain_model.py:1363-1397`
   - Check: `MAX_SIDE_COVERAGE=0.35×pool(100)=35 ≥ min_fires=35`
   - Value: 35 max fires vs 35 required
   - PASS — 35 ≥ 35.

❌ **3. hit_buy=True (OOF precision clears target)**
   - `scripts/retrain_model.py:1996-2004`
   - Check: `pick_threshold_by_side(side=2).hit_target → stored as tradeable_buy`
   - Value: tradeable_buy in sidecar = False
   - FAIL — hit_buy=False because Gate 2 deadlock blocked threshold qualification.

✅ **4. buy_fire mask fires BUY holdout signals**
   - `scripts/retrain_model.py:2169-2174`
   - Check: `buy_fire = (meta_prob_h ≥ max(thr_buy, rank_thr)) & (prop_h==2)`
   - Value: buy_h_n = 52 holdout BUY signals fired
   - PASS — 52 BUY holdout trades.

✅ **5. tradeable_buy_holdout = True**
   - `scripts/retrain_model.py:2288-2292`
   - Check: `hit_buy AND buy_h_n > 0 AND buy_h_prec ≥ 0.50`
   - Value: buy_h_n=52, buy_win_rate=84.6%
   - PASS — 52 trades, 84.6% WR.

#### Meta Gate Audit

🔴 **Gate status: HURTING**  (lift: -19.8pp)

| Metric | Value |
|--------|-------|
| Gated-in precision | 34.0% |
| Blocked signals win rate | 53.8% |
| Precision lift from gate | -19.8pp |
| OOF → Holdout gap | +53.2pp |
| thr_buy / thr_sell | 80.069 / 80.077 |
| Blocked signals | 0 (172 would-win / 149 would-lose) |

> ⚠ Gate is DESTROYING 19.8pp of precision. Blocked signals (53.8%) would have beaten gated (34.0%). Meta model is anti-selective. | OOF overfit warning: dev_prec (87.2%) exceeds holdout (34.0%) by 53.2pp.


---

## Section 15 — Executive Summary

**Symbol:** FIL/USDT  |  **Audit:** 2026-06-07 16:36  |  **Confidence Level:** MEDIUM — based on 47 holdout signals; widen to 200+ for HIGH

**Current:** Precision=34.0%  Sharpe=22.34
**Expected after fixes:** Precision≈39.8%  (+5.8pp)

### Top 5 Problems

1. 🟡 **Severe Class Imbalance (HOLD dominates)** — Score: 61/100
   > HOLD=61.6% of labels. Meta model sees 60% zero-labels → calibration distorted.


### Top 5 Fixes

1. **Severe Class Imbalance (HOLD dominates)**
   → base_vol_threshold→0.72, symmetric BARRIER skews. Already applied.


---

## Section 1 — Model Health

| Metric | Value | Status |
|--------|-------|--------|
| CV Accuracy (OOF) | 59.0% | ⚠ |
| Dev OOF Precision | 87.2% | ✓ |
| Holdout Precision | 34.0% | ✗ |
| Holdout Coverage | 35.8% | ✓ |
| 95% CI Precision | [29.9%, 38.4%] | — |
| OOF→Holdout Gap | -53.2% | ⚠ degradation |
| Holdout Fired | 479 trades | ✓ |
| SELL Win Rate | 83.8% (142 trades) | ✓ |
| BUY Win Rate | 84.6% (52 trades) | ✓ |
| Sharpe (annualised) | 22.34 | ✓ |
| Max Drawdown | 50.92% | ✗ |
| Profit Factor | 4.66 | ✓ |
| Kelly Fraction | 25.0% | — |
| Expectancy/Trade | +0.6177% | ✓ |
| Meta gate optimizer profile | present | ✓ |
| Optimizer-selected gate | EDGE_SR_VETO | ✓ |
| Optimizer threshold match | YES | ✓ |
| Meta gate summary count | 89 symbols | ✓ |
| Statistical Sig. | p=0.0000 (z=-6.99) | ✓ significant |

### Class Distribution
- HOLD: **61.6%** — ⚠ severe imbalance
- SELL: **22.9%**
- BUY:  **15.6%** — OK

### Issues Detected
- **WARNING** — Class imbalance: 61.6% HOLD labels biases model toward neutrality.


---

## Section 2 — Feature Forensics

⚠ feature_health.json not found



---

## Section 3 — Signal Generation Forensics

**Data window:** 8,736 bars

### All-Bar Prediction Breakdown

| Predicted | Count | Raw Precision |
|-----------|-------|---------------|
| BUY  | 126  | 82.5%  |
| SELL | 341 | 75.1% |
| HOLD | 8,269 | — |

### Signal Rejection Funnel

```
Generated (directional):       467  (100%)
Blocked by Meta Gate:      -  321  (69%)
Blocked by Quality (<55):  -   37
Blocked by HMM:            -   23
Blocked by Confluence:     -   18
Blocked by Fake Breakout:  -   14
Blocked by Portfolio Guard:-    9
Blocked by Safe Mode:      -    4
Blocked by Drift:          -    9
Blocked by Cooldown:       -    4
─────────────────────────────────────
Estimated Executed:            28  (6.0%)
```

**BUY side:** ✗ DISABLED  |  **SELL side:** ✗ DISABLED


---

## Section 4 — Opportunity Cost Analysis

Average hold-signal realized return: **1.8605%/bar**

### Time-to-TP (Upper Barrier) Distribution

| Horizon | % of BUY signals that would have hit TP |
|---------|------------------------------------------|
| 6h | 63.9% |
| 12h | 87.6% |
| 18h | 96.8% |
| 24h | 100.0% |
| 48h | 100.0% |

Median time-to-TP: **5 bars** (5h)

### Opportunity Cost by Filter

| Filter | Blocked | Would Win | Would Lose | Win Rate | Opp. Cost |
|--------|---------|-----------|------------|----------|-----------|
| meta_gate | 321 | 172 | 149 | 53.8% | +597.23% |
| quality | 37 | 22 | 15 | 61.5% | +68.84% ⚠ |
| hmm | 23 | 13 | 10 | 57.6% | +42.79% ⚠ |
| confluence | 18 | 9 | 9 | 54.5% | +33.49% |
| fake_breakout | 14 | 8 | 6 | 60.0% | +26.05% ⚠ |
| portfolio | 9 | 5 | 4 | 57.6% | +16.74% ⚠ |


---

## Section 5 — Meta Model Forensics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| ECE (before cal.) | 0.0716 | <0.10 | ✓ |
| ECE (after cal.)  | 0.0000 | <0.10 | ✓ |
| Brier Score | 0.3328 | <0.25 | ✗ |
| Cal. Temperature | 1.1823 | ~1.0 | ⚠ model overconfident |
| Calibration Type | uncalibrated | isotonic | — |

### Confidence Bucket Analysis (Estimated)

| Bucket | Est. Win Rate | Gap | Status |
|--------|---------------|-----|--------|
| 50-60% | 52.0% | -0.03 | ✓ |
| 60-70% | 60.0% | -0.05 | ⚠ overconfident |
| 70-80% | 34.0% | -0.41 | ⚠ overconfident |
| 80-90% | 44.0% | -0.41 | ⚠ overconfident |
| 90-100% | 80.0% | -0.15 | ⚠ overconfident |

**Confidence inflation detected:** No
**Recommended calibrator (for 196 dev samples):** `isotonic`


---

## Section 6 — HMM Regime Forensics

**States:** 7  |  **Global precision:** 66.0%  |  **Max state concentration:** 39.3%

### Per-Regime Performance

| Regime | Trades | Precision | Sell Prec | Expectancy | P.Factor | Modifier | Rec. |
|--------|--------|-----------|-----------|------------|----------|----------|------|
| TRENDING_BULL | 0 | 0.0% | 0.0% | +0.000% | 0.00 | +0.000 | NEUTRAL (insufficient data) |
| TRENDING_BEAR | 0 | 0.0% | 0.0% | +0.000% | 0.00 | +0.000 | NEUTRAL (insufficient data) |
| ACCUMULATION | 0 | 0.0% | 0.0% | +0.000% | 0.00 | +0.000 | NEUTRAL (insufficient data) |
| DISTRIBUTION | 0 | 0.0% | 0.0% | +0.000% | 0.00 | +0.000 | NEUTRAL (insufficient data) |
| COMPRESSION | 0 | 0.0% | 0.0% | +0.000% | 0.00 | +0.000 | NEUTRAL (insufficient data) |
| VOLATILE_EXPANSION | 0 | 0.0% | 0.0% | +0.000% | 0.00 | +0.000 | NEUTRAL (insufficient data) |
| CHOPPY | 0 | 0.0% | 0.0% | +0.000% | 0.00 | +0.000 | NEUTRAL (insufficient data) |


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
| Confidence Drift | 🟡 WARNING | T=1.182 |
| Prediction Drift | 🔴 CRITICAL | OOF vs holdout gap: +53.22pp |

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
| ATR Multiplier | 1.5× | ✓ |
| Win Rate | 73.7% | ✓ |
| Avg Win / Avg Loss | 2.078% / 1.178% | — |
| R:R Ratio | 1.76 | ✓ favourable |
| Kelly Fraction | 58.8% | ⚠ overbetting |
| Avg R-Multiple | 1.33R | ✓ |
| Risk of Ruin | 0.0000% | ✓ low |
| Holdout Sharpe | 22.34 | ✓ |
| Holdout Max DD | 50.92% | ✗ |
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

**1 root causes identified.**  Combined top-5 impact score: **61/500**

| Rank | Cause | Category | Score | Source | Evidence |
|------|-------|---------|-------|--------|---------|
| 1 | 🟡 **Severe Class Imbalance (HOLD dominates)** | Training Quality | 61/100 | `scripts/retrain_model.py:836` ✅ FIXED | HOLD=61.6% of labels. Meta model sees 60% zero-labels → calibration di… |

### Fixes

**1. Severe Class Imbalance (HOLD dominates)**
> 📍 `scripts/retrain_model.py:836` — `base_vol_threshold = 0.80`
> base_vol_threshold→0.72, symmetric BARRIER skews. Already applied.



---

## Section 14 — Automated Improvement Engine

**Base precision:** 34.0%  →  **Expected precision (all fixes):** 39.8%  (+5.8pp)

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
| Selected signals | 479 |
| Rejected signals | 860 |
| Selected precision | 34.0% |
| Rejected precision | 24.5% |
| Meta gate lift (precision) | +9.5% |
| Selected expectancy | +0.618% |
| Rejected expectancy | +0.084% |
| Selected Sharpe | +33.78 |
| Rejected Sharpe | +6.53 |

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
| ACCUMULATION         | ✅ | ✅ | 80.1 | 80.1 | ✅ ENABLED | 60.0% | 1.25 |
| CHOPPY               | ✅ | ✅ | 80.1 | 80.1 | ✅ ENABLED | 60.0% | 1.25 |
| COMPRESSION          | ✅ | ✅ | 80.1 | 80.1 | ✅ ENABLED | 60.0% | 1.25 |
| DISTRIBUTION         | ✅ | ✅ | 80.1 | 80.1 | ✅ ENABLED | 60.0% | 1.25 |
| TRENDING_BEAR        | ✅ | ✅ | 80.1 | 80.1 | ✅ ENABLED | 60.0% | 1.25 |
| TRENDING_BULL        | ✅ | ✅ | 80.1 | 80.1 | ✅ ENABLED | 60.0% | 1.25 |
| VOLATILE_EXPANSION   | ✅ | ✅ | 80.1 | 80.1 | ✅ ENABLED | 60.0% | 1.25 |

**Verdict:** ✅ MODERATE — Selective regime blocking


---

## Section 19 — Deep FIL/USDT vs BTC/ETH Comparison

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
| Gate Lift (pp) | +9.5% |
| Selected signals | 479 |
| Rejected signals | 860 |
| Gate coverage | 35.8% |
| Status | HELPFUL (> +1pp) |


---

## Section 21 — AEGIS Gate Self-Preservation (Phase 2)

| Metric | Value |
|--------|-------|
| Gate Status | HELPFUL |
| Trust Score | 59/100 |
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
| Precision Target | 57.1% |
| Actual Precision | 34.0% |
| Gap | -23.1% |
| Coverage | 10.4% |
| Gating Strategy | ADAPTIVE_PER_REGIME |
| Gate Trust Score | 59/100 |
| Verdict | 🔴 SIGNIFICANTLY BELOW TARGET |


---
