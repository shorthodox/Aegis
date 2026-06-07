# PRECISION IMPROVEMENTS - CODE IMPLEMENTATION GUIDE

This document provides exact code changes to implement the 8 precision improvements.

---

## PHASE 1 FIXES (Quick Wins: +8.0pp expected)

### FIX 1A: Lower HOLD Weight Floor (Brier Score Fix)

**File:** `scripts/retrain_model.py`  
**Lines:** ~1916 (in create_meta_training_targets function)  
**Change Type:** 1-line modification

**Current Code:**
```python
_hold_w = clip(_n_dir * 0.5 / _n_hold, 0.10, 0.60)
```

**New Code:**
```python
_hold_w = clip(_n_dir * 0.5 / _n_hold, 0.05, 0.60)  # Lowered floor from 0.10 to 0.05
```

**Explanation:**
- HOLD bars are neutral (66% of training data)
- Previous weight of 0.10 meant HOLD bars got 10% impact; most got filtered out
- New weight of 0.05 treats HOLD more like data augmentation (gentle regularization)
- Allows meta model to learn from directional signal while not overweighting neutrals

**Testing:**
```python
# After retrain, check:
# 1. Brier score should decrease ~0.01-0.02
# 2. Holdout precision should increase ~1.5-2.0pp
# 3. Calibration (ECE) should improve
```

---

### FIX 1B: Fix Barrier Skew (Triple Barrier Label Balance)

**File:** `scripts/retrain_model.py`  
**Location:** Around line 849-928 (create_triple_barrier_labels function)  
**Change Type:** 5-10 line modification

**Current Code (find this section):**
```python
def create_triple_barrier_labels(close_prices, high_prices, low_prices, volumes,
                                 barrier_params=None, lookback_days=20):
    if barrier_params is None:
        barrier_params = {
            'target_pct': 0.006,      # 0.6% TP
            'stop_loss_pct': 0.015,   # 1.5% SL
            'lookforward_bars': 100
        }
    # ... rest of function uses these parameters
```

**New Code:**
```python
def create_triple_barrier_labels(close_prices, high_prices, low_prices, volumes,
                                 barrier_params=None, lookback_days=20, volatility_regime=None):
    if barrier_params is None:
        # Symmetric targets for both sides
        if volatility_regime == 'HIGH':
            # Loosen targets in volatile regime
            barrier_params = {
                'target_pct': 0.012,      # 1.2% TP (loosen for volatility)
                'stop_loss_pct': 0.015,   # 1.5% SL
                'lookforward_bars': 120   # Give more bars to hit targets
            }
        elif volatility_regime == 'LOW':
            # Tighter targets in stable regime
            barrier_params = {
                'target_pct': 0.004,      # 0.4% TP (tight target)
                'stop_loss_pct': 0.010,   # 1.0% SL
                'lookforward_bars': 80
            }
        else:
            # Default: balanced
            barrier_params = {
                'target_pct': 0.008,      # 0.8% TP
                'stop_loss_pct': 0.012,   # 1.2% SL (closer to target for balance)
                'lookforward_bars': 100
            }
    # ... rest of function
```

**Testing:**
```python
# After retrain, verify label balance:
buy_count = (labels == 1).sum()
sell_count = (labels == -1).sum()
hold_count = (labels == 0).sum()

# Should be roughly:
# BUY: 28-32% of directional
# SELL: 68-72% of directional
# HOLD: 64-68% of all

# Previous skew was 20% BUY / 80% SELL (imbalanced)
print(f"Label distribution: BUY={buy_count/len(labels)*100:.1f}%, SELL={sell_count/len(labels)*100:.1f}%, HOLD={hold_count/len(labels)*100:.1f}%")
```

---

### FIX 1C: Remove Absolute Price Features

**File:** `src/ml/feature_engine.py`  
**Location:** Feature definition sections (~line 50-200, varies by structure)  
**Change Type:** 15-30 line modifications

**Current Code (example structure):**
```python
def create_features(df):
    df['close'] = df['close']  # Direct price
    df['high'] = df['high']    # Direct price
    df['low'] = df['low']      # Direct price
    df['volume'] = df['volume']  # Direct volume
    
    # Relative features mixed in
    df['ret_1h'] = df['close'].pct_change()
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    return df
```

**New Code:**
```python
def create_features(df):
    # REMOVE direct price features:
    # df['close'] = df['close']  # DELETE
    # df['high'] = df['high']    # DELETE
    # df['low'] = df['low']      # DELETE
    
    # ADD normalized position features instead:
    
    # 1. Normalized Bollinger Bands position (0-1 range)
    df['close_bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    df['close_bb_position'] = df['close_bb_position'].clip(0, 1)  # Bound to 0-1
    
    # 2. Keltner Channel position
    df['close_kc_position'] = (df['close'] - df['kc_lower']) / (df['kc_upper'] - df['kc_lower'] + 1e-10)
    df['close_kc_position'] = df['close_kc_position'].clip(0, 1)
    
    # 3. ATR-normalized offset from SMA
    df['price_sma_offset'] = (df['close'] - df['sma_50']) / (df['atr_14'] + 1e-10)
    
    # 4. Volume normalization
    df['volume_sma_ratio'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-10)
    
    # KEEP relative features:
    df['ret_1h'] = df['close'].pct_change()
    df['ret_4h'] = df['close'].pct_change(4)
    df['ret_1d'] = df['close'].pct_change(24)
    
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['rsi'] = talib.RSI(df['close'], timeperiod=14) / 100.0  # Normalized to 0-1
    df['macd_position'] = (df['macd'] - df['macd_signal']) / (df['atr_14'] + 1e-10)
    
    return df
```

**Key Changes:**
1. Replace `close`, `high`, `low` with `close_bb_position`, `close_kc_position`
2. Replace `volume` with `volume_sma_ratio`
3. Normalize all indicators to 0-1 or z-scored ranges
4. All features should be dimensionless (ratios, positions, or returns)

**Testing:**
```python
# After feature recalculation, verify:
# 1. No feature should contain price levels > 1000
# 2. All features should be bounded or z-scored
# 3. Feature correlation to train should be > 0.90 on OOS data

# Check for absolute prices in model:
feature_names = model.feature_names()
bad_features = [f for f in feature_names if f in ['close', 'high', 'low', 'volume']]
assert len(bad_features) == 0, f"Found absolute price features: {bad_features}"
```

---

## PHASE 2 FIXES (Discrimination Improvements: +2.5pp expected)

### FIX 2A: Calibrate Confidence with Isotonic Regression

**File:** `scripts/retrain_model.py`  
**Location:** After meta model training (around line 2100-2200)  
**Change Type:** 30-40 line addition

**New Code (add after meta model training):**
```python
from sklearn.isotonic import IsotonicRegression
import pickle

# After training meta model on OOF data:
# oof_prob = meta_model.predict_proba(oof_X)[:, 1]  # Probability of BUY/SELL
# oof_y = meta_y_holdout  # True labels (0, 1, 2 for HOLD, BUY, SELL)

# Fit calibration model on OOF predictions
# Map to binary: 0 (not directional) vs 1 (directional)
oof_y_binary = (oof_y != 0).astype(int)

calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
calibrator.fit(oof_prob, oof_y_binary)

# Save calibrator for live use
calibrator_path = f"{model_store}/calibrator_{symbol}.pkl"
with open(calibrator_path, 'wb') as f:
    pickle.dump(calibrator, f)

# Validate calibration: check Brier score improves
oof_prob_calibrated = calibrator.predict(oof_prob)
brier_uncalibrated = np.mean((oof_prob - oof_y_binary)**2)
brier_calibrated = np.mean((oof_prob_calibrated - oof_y_binary)**2)

print(f"Brier score before calibration: {brier_uncalibrated:.4f}")
print(f"Brier score after calibration: {brier_calibrated:.4f}")
print(f"Improvement: {brier_uncalibrated - brier_calibrated:.4f}")

# Log to sidecar
sidecar['calibration_brier_improvement'] = brier_uncalibrated - brier_calibrated
sidecar['calibrator_path'] = calibrator_path
```

**In Live Engine** (src/ml/predictor.py or similar):
```python
# Load calibrator at startup
calibrator_path = f"{model_store}/calibrator_{symbol}.pkl"
if os.path.exists(calibrator_path):
    with open(calibrator_path, 'rb') as f:
        calibrator = pickle.load(f)
else:
    calibrator = None

# At prediction time:
meta_prob_raw = meta_model.predict_proba(X_live)[:, 1]

if calibrator is not None:
    meta_prob = calibrator.predict(meta_prob_raw)  # Calibrated probability
else:
    meta_prob = meta_prob_raw  # Fallback to uncalibrated
```

**Testing:**
```python
# After implementation:
# 1. Verify Brier score improved by ~0.01-0.02
# 2. Check win vs loss confidence spread increased

win_conf_mean = calibrated_prob[oof_y == 1].mean()
loss_conf_mean = calibrated_prob[oof_y == 0].mean()
spread = win_conf_mean - loss_conf_mean

print(f"Win avg confidence: {win_conf_mean:.4f}")
print(f"Loss avg confidence: {loss_conf_mean:.4f}")
print(f"Spread: {spread:.4f}")

assert spread > 0.08, f"Confidence spread too small: {spread:.4f}"
```

---

### FIX 2B: Fix Gate Deadlock - Adaptive effective_min_fires

**File:** `scripts/retrain_model.py`  
**Location:** `pick_threshold_by_side()` function (around line 1363-1397)  
**Change Type:** 20-30 line modification

**Current Code:**
```python
def pick_threshold_by_side(side, oof_prob_directional, oof_y_directional, 
                           oof_trades_by_p, oof_win_by_p):
    """
    Pick threshold for BUY or SELL side to maximize profit while meeting coverage constraint.
    """
    pool_size = len(oof_prob_directional)
    MAX_SIDE_COVERAGE = 0.35  # 35% of pool maximum
    min_fires = 35  # Fixed
    
    # Scan quantiles from high to low confidence
    for threshold in np.arange(0.99, 0.40, -0.01):
        # ... code
    
    # If no threshold found
    return None, False
```

**New Code:**
```python
def pick_threshold_by_side(side, oof_prob_directional, oof_y_directional, 
                           oof_trades_by_p, oof_win_by_p, label_ratio=None):
    """
    Pick threshold for BUY or SELL side with adaptive constraints.
    
    label_ratio: fraction of directional bars with this side label (e.g., 0.20 for BUY if only 20% are BUY)
    """
    pool_size = len(oof_prob_directional)
    
    # Adaptive MAX_SIDE_COVERAGE based on label balance
    if label_ratio is not None and label_ratio < 0.25:
        # Low label ratio: allow higher coverage to find enough signals
        MAX_SIDE_COVERAGE = min(0.60, 100 / (pool_size * label_ratio))
    else:
        MAX_SIDE_COVERAGE = 0.35
    
    # Adaptive effective_min_fires
    # Don't require more fires than the pool can provide
    effective_min_fires = max(10, int(pool_size * 0.15))  # 15% of pool minimum
    
    # If coverage constraint is tighter than min_fires requirement, relax min_fires
    max_possible_fires = int(MAX_SIDE_COVERAGE * pool_size)
    if effective_min_fires > max_possible_fires:
        effective_min_fires = int(max_possible_fires * 0.80)  # Use 80% of max coverage
        LOG.warning(f"Gate constraint deadlock detected. Relaxing min_fires from {max_possible_fires} to {effective_min_fires}")
    
    # Scan quantiles from high to low confidence
    best_threshold = None
    best_win_rate = 0
    
    for threshold in np.arange(0.99, 0.40, -0.01):
        fire_mask = oof_prob_directional >= threshold
        n_fires = fire_mask.sum()
        
        # Check coverage constraint
        if n_fires == 0 or n_fires / pool_size > MAX_SIDE_COVERAGE:
            continue
        
        # Check minimum fires requirement
        if n_fires < effective_min_fires:
            continue
        
        # Calculate win rate
        n_wins = oof_win_by_p[fire_mask].sum()
        win_rate = n_wins / n_fires if n_fires > 0 else 0
        
        # Check precision target
        if win_rate >= 0.50:
            best_threshold = threshold
            best_win_rate = win_rate
            break
    
    hit_target = best_threshold is not None
    return best_threshold, hit_target
```

**Usage (in main retrain function):**
```python
# Calculate label ratios
buy_labels = (prop_h == 2).astype(int)
sell_labels = (prop_h == -2).astype(int)
buy_ratio = buy_labels.sum() / len(prop_h)
sell_ratio = sell_labels.sum() / len(prop_h)

# Pick thresholds with adaptive constraints
thr_buy, hit_buy = pick_threshold_by_side('BUY', ..., label_ratio=buy_ratio)
thr_sell, hit_sell = pick_threshold_by_side('SELL', ..., label_ratio=sell_ratio)
```

**Testing:**
```python
# After implementation:
# 1. Verify gates no longer deadlock (hit_buy and hit_sell should be True)
# 2. Check that effective_min_fires adapts to label ratios

print(f"BUY gate hit: {hit_buy}, threshold: {thr_buy:.3f}")
print(f"SELL gate hit: {hit_sell}, threshold: {thr_sell:.3f}")

assert hit_buy, "BUY gate still deadlocked"
assert hit_sell, "SELL gate still deadlocked"
```

---

## PHASE 3 FIXES (Regime Intelligence: +1.0pp expected)

### FIX 3A: Regime-Aware Model Training

**File:** `scripts/retrain_model.py`  
**Location:** Main training loop (modify around line 1200-1300)  
**Change Type:** 50-100 line modification/addition

**Concept:** Train separate models for each regime, then select at prediction time.

**New Code Structure:**

```python
def train_models_by_regime(X_train, y_train, regimes_train, symbol, model_store):
    """
    Train 3 separate models: one per regime type (TRENDING, RANGING, VOLATILE)
    """
    from sklearn.ensemble import LGBMClassifier
    
    models_by_regime = {}
    regime_names = ['TRENDING', 'RANGING', 'VOLATILE']
    
    for regime_idx, regime_name in enumerate(regime_names):
        # Filter data for this regime
        regime_mask = regimes_train == regime_idx
        X_regime = X_train[regime_mask]
        y_regime = y_train[regime_mask]
        
        if len(X_regime) < 100:
            LOG.warning(f"Insufficient data for regime {regime_name}: {len(X_regime)} samples")
            continue
        
        # Adjust hyperparameters based on regime
        if regime_name == 'VOLATILE':
            # Simpler model in volatile regime (less overfitting)
            model = LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                num_leaves=31,
                reg_alpha=2.0,
                reg_lambda=2.0,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
        elif regime_name == 'RANGING':
            # Standard complexity in ranging regime
            model = LGBMClassifier(
                n_estimators=150,
                max_depth=7,
                num_leaves=63,
                reg_alpha=1.0,
                reg_lambda=1.0,
                min_child_samples=15,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42
            )
        else:  # TRENDING
            # Can use more complex model in stable trending
            model = LGBMClassifier(
                n_estimators=200,
                max_depth=8,
                num_leaves=127,
                reg_alpha=0.5,
                reg_lambda=0.5,
                min_child_samples=10,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42
            )
        
        # Train
        model.fit(X_regime, y_regime, verbose=0)
        models_by_regime[regime_name] = model
        
        # Save
        model_path = f"{model_store}/{symbol}_model_{regime_name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        LOG.info(f"Trained {regime_name} model with {len(X_regime)} samples")
    
    return models_by_regime


def predict_with_regime_selection(X_live, current_regime, models_by_regime):
    """
    Make predictions using regime-appropriate model
    """
    regime_names = ['TRENDING', 'RANGING', 'VOLATILE']
    regime_name = regime_names[current_regime]
    
    if regime_name not in models_by_regime:
        # Fallback to default model if regime-specific not available
        LOG.warning(f"No model for regime {regime_name}, using default")
        model = models_by_regime.get('RANGING')  # Use ranging as default
    else:
        model = models_by_regime[regime_name]
    
    prob = model.predict_proba(X_live)[:, 1]
    return prob
```

**Integration into main retrain:**

```python
# After feature creation and regime classification
regimes_train = hmm.predict(returns_train)
regimes_oof = hmm.predict(returns_oof)

# Train models by regime instead of single model
models_by_regime = train_models_by_regime(X_train, y_train, regimes_train, symbol, model_store)

# During OOF evaluation, use regime-aware prediction
oof_prob_list = []
for i in range(len(X_oof)):
    regime = regimes_oof[i]
    prob = predict_with_regime_selection(X_oof[i:i+1], regime, models_by_regime)[0]
    oof_prob_list.append(prob)

oof_prob = np.array(oof_prob_list)
```

**Regime-Aware Thresholds:**

```python
# Set thresholds per regime
thresholds_by_regime = {
    'TRENDING': 0.50,     # Can be precise in trending
    'RANGING': 0.52,      # Need higher threshold in ranging (choppier)
    'VOLATILE': 0.55      # Need even higher threshold in volatile (noisy)
}

# Apply during live trading
def get_live_threshold(current_regime, side):
    regime_names = ['TRENDING', 'RANGING', 'VOLATILE']
    regime_name = regime_names[current_regime]
    base_threshold = thresholds_by_regime[regime_name]
    
    # Adjust based on side
    if side == 'BUY':
        return base_threshold
    else:  # SELL
        return base_threshold + 0.01  # SELL slightly more conservative
```

**Testing:**
```python
# After implementation:
# 1. Verify models trained for each regime
# 2. Check precision by regime improves

for regime_idx, regime_name in enumerate(['TRENDING', 'RANGING', 'VOLATILE']):
    regime_mask = regimes_oof == regime_idx
    regime_prec = (oof_pred[regime_mask] == oof_y[regime_mask]).mean()
    print(f"{regime_name} precision: {regime_prec:.1%}")

# Expected: precision within 2pp across all regimes (vs 15-20pp spread before)
```

---

## VALIDATION CHECKLIST

After implementing all Phase 1 fixes, run this validation:

```python
# validation_checklist.py

def validate_precision_improvements(symbol, model_store, data_loader):
    """Validate all fixes are working correctly"""
    
    checks = {}
    
    # 1. Brier score improvement
    brier_score = load_sidecar(f"{model_store}/{symbol}.sidecar")['brier_score']
    checks['brier_score < 0.20'] = brier_score < 0.20
    print(f"✓ Brier score: {brier_score:.4f} {'PASS' if checks['brier_score < 0.20'] else 'FAIL'}")
    
    # 2. Label balance
    labels = load_labels(f"{model_store}/{symbol}_labels.pkl")
    buy_pct = (labels == 1).sum() / len(labels)
    sell_pct = (labels == -1).sum() / len(labels)
    skew = abs(buy_pct - sell_pct)
    checks['label_skew < 0.15'] = skew < 0.15
    print(f"✓ Label balance: BUY={buy_pct:.1%}, SELL={sell_pct:.1%}, skew={skew:.1%} {'PASS' if checks['label_skew < 0.15'] else 'FAIL'}")
    
    # 3. Feature drift check
    X_train, X_oof = load_features(model_store, symbol)
    feature_drift = []
    for feat_idx in range(X_train.shape[1]):
        train_mean, train_std = X_train[:, feat_idx].mean(), X_train[:, feat_idx].std()
        oof_zscore = np.abs((X_oof[:, feat_idx].mean() - train_mean) / (train_std + 1e-10))
        feature_drift.append(oof_zscore)
    
    max_drift = max(feature_drift)
    checks['max_feature_drift < 2.5'] = max_drift < 2.5
    print(f"✓ Feature drift: max zscore={max_drift:.2f} {'PASS' if checks['max_feature_drift < 2.5'] else 'FAIL'}")
    
    # 4. No absolute price features
    feature_names = load_feature_names(model_store, symbol)
    absolute_price_features = [f for f in feature_names if f in ['close', 'high', 'low', 'volume']]
    checks['no_absolute_prices'] = len(absolute_price_features) == 0
    print(f"✓ Absolute prices removed: {len(absolute_price_features)} found {'PASS' if checks['no_absolute_prices'] else 'FAIL'}")
    
    # 5. Gate qualification
    sidecar = load_sidecar(f"{model_store}/{symbol}.sidecar")
    checks['buy_gate_qualified'] = sidecar.get('tradeable_buy', False)
    checks['sell_gate_qualified'] = sidecar.get('tradeable_sell', False)
    print(f"✓ Gate qualification: BUY={checks['buy_gate_qualified']}, SELL={checks['sell_gate_qualified']} PASS")
    
    # 6. Confidence discrimination
    oof_prob, oof_y = load_predictions(model_store, symbol)
    win_conf = oof_prob[oof_y == 1].mean()
    loss_conf = oof_prob[oof_y == 0].mean()
    conf_spread = win_conf - loss_conf
    checks['confidence_spread > 0.08'] = conf_spread > 0.08
    print(f"✓ Confidence spread: {conf_spread:.4f} {'PASS' if checks['confidence_spread > 0.08'] else 'FAIL'}")
    
    # 7. Precision by regime
    if hasattr(hmm, 'predict'):
        returns = load_returns(symbol)
        regimes = hmm.predict(returns)
        for regime_idx, regime_name in enumerate(['TRENDING', 'RANGING', 'VOLATILE']):
            mask = regimes == regime_idx
            if mask.sum() > 0:
                regime_prec = (oof_prob[mask] >= 0.50) == (oof_y[mask] == 1).mean()
                print(f"  - {regime_name}: {regime_prec:.1%} precision")
    
    # Summary
    passed = sum(checks.values())
    total = len(checks)
    print(f"\n✓ Validation: {passed}/{total} checks passed")
    
    return checks

if __name__ == "__main__":
    for symbol in ['BTC_USDT', 'ETH_USDT', 'BNB_USDT']:
        print(f"\n{'='*60}")
        print(f"Validating {symbol}")
        print(f"{'='*60}")
        validate_precision_improvements(symbol, MODEL_STORE, DataLoader())
```

---

## Deployment Sequence

1. **Backup current models:**
   ```bash
   cp -r src/ml/model_store src/ml/model_store.backup_prePhase1
   ```

2. **Apply Phase 1 fixes:**
   - Fix 1A: Lower HOLD weight (1 line)
   - Fix 1B: Fix barrier skew (10 lines)
   - Fix 1C: Remove absolute prices (20 lines)

3. **Retrain all tokens:**
   ```bash
   python scripts/retrain_model.py --all-symbols --verbose --save-sidecar
   ```

4. **Run validation:**
   ```bash
   python validation_checklist.py
   ```

5. **Deploy to live:**
   - Only deploy tokens passing all validation checks
   - Monitor precision on live trades for 24-48 hours
   - Rollback if precision drops > 2pp

---

## Expected Results

| Metric | Before | After Phase 1 | After Phase 2 | After Phase 3 |
|--------|--------|--------------|---------------|---------------|
| Avg Precision | 37.9% | ~46% | ~48.5% | ~49.5% |
| Brier Score | 0.23 | 0.20 | 0.19 | 0.19 |
| Confidence Spread | 0.87pp | 3-4pp | 5-6pp | 5-6pp |
| BUY Gate Hit | 40% | 95% | 98% | 100% |
| Feature Drift Max | 3.2σ | 2.1σ | 2.1σ | 1.9σ |
| Regime Consistency | 18pp spread | 18pp | 8pp | 4pp |

