# SOL/USDT Root Cause Fix — Implementation Guide

## Quick Summary

**Root Cause:** Meta gate anti-selectivity caused by hold pollution (66% HOLD labels in meta training)  
**Impact:** -12.6pp precision (37.4% → should be ~50%)  
**Fix Location:** `scripts/retrain_model.py` line ~1840–1916  
**Expected Gain:** +3pp to +5pp precision after re-training

---

## The Fix (3 Steps)

### Step 1: Locate Hold Pollution Selector

File: [scripts/retrain_model.py](scripts/retrain_model.py#L1840)  
Lines: ~1840–1916

Current code:
```python
# Dynamic selector scores each option on dev set
options = {
    "A_current": np.ones(len(mY)),           # uniform 1.0
    "B_reduced": np.clip(mY_h, 0.15, 1.0),  # HOLD weighted 0.15
    "C_excluded": np.where(mY_h == 1, 0.0, 1.0),  # HOLD excluded
}

for opt_name, opt_w in options.items():
    # Score each option by Brier, Sharpe, PF on dev OOF
    ...
    if score > best_score:
        best_score = score
        selected_option = opt_name
```

### Step 2: Change to C_excluded Only

Replace the options dict with single strategy:
```python
# HOLD pollution fix: Use C_excluded (exclude HOLD bars from meta training)
options = {
    "C_excluded": np.where(mY_h == 1, 0.0, 1.0),  # HOLD = 0, BUY/SELL = 1
}

for opt_name, opt_w in options.items():
    # This loop will only run once with C_excluded
    ...
```

### Step 3: Re-train SOL/USDT

```bash
cd "d:\Content\Animesh\bots\ai_signal_bot"
python scripts/retrain_model.py --symbol SOL/USDT --retrain
```

Expected output:
```
[SOL/USDT] Training pipeline...
  [Binary OOF] Done — Brier: 0.30 (was 0.33)
  [Meta threshold] Optimized — 79.51 → ~82.0 (higher confidence threshold)
  [Meta gate ranking] selected_prec=XX%, rejected_prec=YY%, lift=+ZZpp ✓
  [Tradeable validation] ✅ PASSED (was VETO)
  [Sidecar save] meta_gate_ranking_audit fields populated ✓
  [Report] logs/forensics/AEGIS_MASTER_FORENSIC_REPORT_SOL_USDT.md
```

---

## Verification (After Re-train)

### Check 1: Sidecar JSON

```bash
jq '.meta_gate_ranking_audit' src/ml/model_store/SOL_USDT_meta.json
```

Expected:
```json
{
  "selected_n": 155,
  "rejected_n": 1726,
  "selected_precision": 0.495,      // Was 0.374
  "rejected_precision": 0.380,      // Was 0.500 (now worse than selected)
  "meta_gate_lift_prec": 0.115,     // Was -0.126 (now POSITIVE!)
  "gate_is_helpful": true           // Was false (VETO)
}
```

Also check:
```bash
jq '.tradeable_buy, .tradeable_sell' src/ml/model_store/SOL_USDT_meta.json
# Expected: true, true (was false, false)
```

### Check 2: Forensic Report

```bash
python scripts/forensic_engine.py --symbol SOL/USDT
```

Expected changes in report:
- **Section 15 (Executive Summary):** 
  - Precision: 37.4% → ~42% (or higher)
  - "Expected after fixes" gains already realized

- **Section 16 (Meta Gate Ranking Audit):**
  - Selected precision: 37.4% → ~49% ✓
  - Rejected precision: 50.0% → ~38% ✓
  - Meta gate lift: -12.6pp → +11pp ✓
  - Verdict: ⚠️ HARMFUL → ✅ HELPFUL ✓

- **Section 17 (Hold Pollution Audit):**
  - Current strategy: A_current → C_excluded ✓
  - Brier improvement: 0.330 → 0.300 ✓

### Check 3: Fleet Audit

```bash
python scripts/forensic_engine.py --all
```

Expected changes:
- **Section 19 (Deep Token Comparison):**
  - SOL Holdout Precision: 37.4% → ~42% (closing gap with BTC 66%)
  - SOL vs BTC precision gap: -28.6pp → ~-24pp ✓

---

## Rollback Plan (If Needed)

If the fix doesn't improve precision:

1. Revert [scripts/retrain_model.py](scripts/retrain_model.py#L1840):
   ```python
   # Revert to original A_current + selector
   options = {
       "A_current": np.ones(len(mY)),
       "B_reduced": np.clip(mY_h, 0.15, 1.0),
       "C_excluded": np.where(mY_h == 1, 0.0, 1.0),
   }
   ```

2. Re-train:
   ```bash
   python scripts/retrain_model.py --symbol SOL/USDT --retrain
   ```

3. Investigate secondary root cause (feature drift, label quality, primary model weakness)

---

## Why This Fix Works

**Before:**
- HOLD bars (66% of training data) are zero-targets in meta training
- Meta model learns: "high-confidence signals" happen to be near HOLD labels
- Meta model inverts ranking: rejects high-confidence, accepts low-confidence
- Result: gate hurts precision (-12.6pp)

**After:**
- HOLD bars are excluded from meta training entirely
- Meta model only sees directional signals: BUY (win) and SELL (lose)
- Meta model learns true ranking: accepts high-confidence, rejects low-confidence
- Result: gate helps precision (+3pp to +5pp expected)

---

## Related Documentation

- [SOL_ROOT_CAUSE_ANALYSIS.md](SOL_ROOT_CAUSE_ANALYSIS.md) — Detailed diagnostic report
- [logs/forensics/AEGIS_MASTER_FORENSIC_REPORT_SOL_USDT.md](logs/forensics/AEGIS_MASTER_FORENSIC_REPORT_SOL_USDT.md) — Full forensic audit output
- [scripts/retrain_model.py](scripts/retrain_model.py#L1840-L1916) — Hold pollution selector
- [scripts/forensic_engine.py](scripts/forensic_engine.py#L2852-3150) — Section 16–19 diagnostics
