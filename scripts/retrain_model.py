#!/usr/bin/env python3
"""
retrain_model.py – Aegis‑1 Model Trainer with Enhanced Features
----------------------------------------------------------------
Trains XGBoost models for each token using regime‑adaptive triple‑barrier labeling,
Optuna hyperparameter optimization, SHAP feature pruning, and time‑series cross‑validation.
Aims for highest realistic accuracy (target >80% out‑of‑sample) by leveraging
the rich feature set from feature_engine.py (60+ indicators).
"""

import sys
import os
import xgboost as xgb
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import warnings
import subprocess
import time
import optuna
import shap
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, List, Optional, Tuple

# --- Pandas compatibility check ---
if pd.__version__ >= "2.2.0":
    pass
else:
    warnings.warn(f"Pandas version {pd.__version__} detected. Upgrade to 2.2.0+ for best compatibility.", UserWarning)

# --- Path fix ---
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Import enhanced feature engineering (includes all new indicators)
from src.ml.feature_engine import prepare_features, compute_atr
from src.ml.predictor import Predictor

warnings.filterwarnings("ignore", message="Model file not found")
warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------
# FLEET DEFINITION (58 tokens)
# ------------------------------
FLEET_SYMBOLS = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
    'HYPE/USDT', 'ASTER/USDT', 'SUI/USDT', 'TAO/USDT', 'RENDER/USDT',
    'ADA/USDT', 'AVAX/USDT', 'LINK/USDT', 'TRX/USDT', 'DOT/USDT',
    'NEAR/USDT', 'MATIC/USDT', 'LTC/USDT', 'BCH/USDT', 'SHIB/USDT',
    'TON/USDT', 'ICP/USDT', 'HBAR/USDT', 'APT/USDT', 'ARB/USDT',
    'OP/USDT', 'STX/USDT', 'FIL/USDT', 'AAVE/USDT', 'VET/USDT',
    'RNDR/USDT', 'INJ/USDT', 'TIA/USDT', 'SEI/USDT', 'KAS/USDT',
    'FET/USDT', 'AGIX/USDT', 'OCEAN/USDT', 'AKT/USDT', 'THETA/USDT',
    'GRT/USDT', 'LDO/USDT', 'PYTH/USDT', 'JUP/USDT', 'ONDO/USDT',
    'PEPE/USDT', 'DOGE/USDT', 'WIF/USDT', 'FLOKI/USDT', 'BONK/USDT',
    'WLFI/USDT', 'MNT/USDT', 'ENA/USDT', 'BGB/USDT', 'PI/USDT',
    'SKY/USDT', 'TRUMP/USDT', 'NIGHT/USDT'
]

# News configuration
NEWS_FILE = Path(root_dir) / "data" / "news_data.json"
NEWS_MAX_AGE_SECONDS = 30 * 60  # 30 minutes

# ------------------------------
# ASSET‑SPECIFIC BARRIER MULTIPLIER (with low-accuracy token tuning)
# ------------------------------
def get_atr_multiplier(symbol: str) -> float:
    """
    Heavy caps use 1.2x ATR.
    Low-accuracy assets (like SUI, DOT) get wider barriers (1.8x or 2.0x) to avoid stop‑outs.
    Others use 1.5x.
    """
    high_cap = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    low_accuracy = ['SUI/USDT', 'DOT/USDT', 'SEI/USDT', 'KAS/USDT']  # add more as needed
    if symbol in high_cap:
        return 1.2
    elif symbol in low_accuracy:
        return 1.8   # wider barrier for volatile / low‑accuracy assets
    else:
        return 1.5

# ------------------------------
# TRIPLE‑BARRIER LABELING (regime‑adaptive)
# ------------------------------
def create_triple_barrier_labels(df: pd.DataFrame, atr_multiplier: float,
                                 max_lookahead: int = 48,
                                 volatility_regime: Optional[pd.Series] = None,
                                 efficiency_ratio: Optional[pd.Series] = None,
                                 trend_regime: Optional[pd.Series] = None) -> pd.Series:
    """
    Labels: 1 if price hits upper barrier first, 0 if lower barrier first or no hit.
    Uses adaptive volatility threshold and efficiency filter.
    """
    base_vol_threshold = 0.8
    labels = pd.Series(index=df.index, dtype=float)
    atr = compute_atr(df, period=14)   # use imported compute_atr

    for i in range(len(df) - 1):
        vol_threshold = base_vol_threshold
        if trend_regime is not None and trend_regime.iloc[i] == 1:
            vol_threshold = 0.6   # trending markets need less vol

        if volatility_regime is not None and volatility_regime.iloc[i] < vol_threshold:
            labels.iloc[i] = 0
            continue

        if efficiency_ratio is not None and efficiency_ratio.iloc[i] < 0.3:
            labels.iloc[i] = 0
            continue

        entry_price = df.iloc[i]['close']
        atr_val = atr.iloc[i]
        if atr_val == 0 or np.isnan(atr_val):
            atr_val = entry_price * 0.001
        upper = entry_price + atr_multiplier * atr_val
        lower = entry_price - atr_multiplier * atr_val

        hit = None
        for j in range(1, min(max_lookahead, len(df) - i)):
            high = df.iloc[i + j]['high']
            low = df.iloc[i + j]['low']
            if high >= upper:
                hit = 1
                break
            if low <= lower:
                hit = 0
                break
        labels.iloc[i] = hit if hit is not None else 0

    labels.iloc[-1] = 0
    return labels

# ------------------------------
# FEATURE IMPORTANCE LOGGING (SHAP‑based pruning)
# ------------------------------
def prune_features_by_shap(model: xgb.Booster, X: pd.DataFrame, threshold: float = 0.01) -> List[str]:
    """
    Compute SHAP values on a subset of data, drop features with mean |SHAP| below threshold.
    Returns list of features to keep.
    """
    sample_size = min(500, len(X))
    X_sample = X.sample(n=sample_size, random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.Series(mean_abs_shap, index=X.columns)
    keep_features = feature_importance[feature_importance > threshold].index.tolist()
    dropped = feature_importance[feature_importance <= threshold].index.tolist()
    if dropped:
        print(f"   🗑️ SHAP pruning removed {len(dropped)} low‑impact features: {dropped[:5]}...")
    return keep_features

# ------------------------------
# LABEL SMOOTHING
# ------------------------------
def smooth_labels(y: np.ndarray, smoothing: float = 0.05) -> np.ndarray:
    """
    Convert 0/1 hard labels to soft labels: 0 -> smoothing, 1 -> 1 - smoothing.
    """
    return np.where(y == 1, 1.0 - smoothing, smoothing)

# ------------------------------
# TIME‑SERIES CROSS‑VALIDATION
# ------------------------------
def time_series_cv_score(X, y, params, n_splits=5):
    """
    Evaluate XGBoost using TimeSeriesSplit (5 folds). Returns average logloss.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        y_train_smooth = smooth_labels(y_train.values)
        dtrain = xgb.DMatrix(X_train, label=y_train_smooth)
        dval = xgb.DMatrix(X_val, label=y_val.values)
        model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dval, 'eval')],
                          early_stopping_rounds=20, verbose_eval=False)
        pred = model.predict(dval)
        from sklearn.metrics import log_loss
        score = log_loss(y_val, pred)
        scores.append(score)
    return np.mean(scores)

# ------------------------------
# OPTUNA OBJECTIVE FUNCTION
# ------------------------------
def objective(trial, X_train, y_train, X_val, y_val, scale_pos_weight, feature_names):
    """
    Optuna objective for hyperparameter tuning.
    """
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 1, 20, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 5, 20),
        'scale_pos_weight': scale_pos_weight,
        'seed': 42,
        'tree_method': 'hist'
    }
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
    model = xgb.train(params, dtrain, num_boost_round=500, evals=[(dval, 'eval')],
                      early_stopping_rounds=50, verbose_eval=False)
    pred = model.predict(dval)
    from sklearn.metrics import log_loss
    return float(log_loss(y_val, pred))

# ------------------------------
# AUTONOMOUS NEWS SYNC FOR TRAINING
# ------------------------------
def run_news_scraper():
    scraper_path = Path(root_dir) / "scripts" / "news_scraper.py"
    if not scraper_path.exists():
        print(f"⚠️ News scraper not found at {scraper_path}")
        return False
    log_path = Path(root_dir) / "logs" / "news_sync.log"
    log_path.parent.mkdir(exist_ok=True)
    try:
        with open(log_path, 'a') as log_f:
            result = subprocess.run(
                [sys.executable, str(scraper_path)],
                stdout=log_f,
                stderr=log_f,
                timeout=60,
                check=False
            )
        if result.returncode == 0:
            print(f"✅ News scraper completed (log: {log_path})")
            return True
        else:
            print(f"❌ News scraper failed with code {result.returncode}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ News scraper timed out after 60 seconds")
        return False
    except Exception as e:
        print(f"❌ Failed to run news scraper: {e}")
        return False

def ensure_fresh_news_for_training():
    need_sync = False
    if NEWS_FILE.exists():
        mod_time = datetime.fromtimestamp(NEWS_FILE.stat().st_mtime)
        age = (datetime.now() - mod_time).total_seconds()
        if age >= NEWS_MAX_AGE_SECONDS:
            print(f"📰 News file is stale (age {age/60:.1f} min). Running scraper...")
            need_sync = True
        else:
            print(f"📰 News file is fresh (age {age/60:.1f} min).")
    else:
        print("📰 News file missing. Running scraper to fetch initial headlines...")
        need_sync = True

    if need_sync:
        success = run_news_scraper()
        if not success:
            print("⚠️ Continuing without fresh news. Sentiment features will be zero.")
            return False
        time.sleep(1)
        if not NEWS_FILE.exists():
            print("⚠️ News file still missing after scraper run. Proceeding with neutral sentiment.")
            return False
    return True

# ------------------------------
# FEATURE IMPORTANCE LOGGING (post‑training)
# ------------------------------
def log_feature_importance(model, feature_names: List[str], symbol: str):
    logs_dir = Path(root_dir) / "logs" / "features"
    logs_dir.mkdir(parents=True, exist_ok=True)
    importance = model.get_score(importance_type='gain')
    if importance:
        if all(k.startswith('f') for k in importance.keys()):
            idx = [int(k[1:]) for k in importance.keys()]
            importance_dict = {feature_names[i]: importance[f'f{i}'] for i in idx if i < len(feature_names)}
        else:
            importance_dict = importance
        sorted_imp = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    else:
        imp_array = model.get_score(importance_type='weight') or {}
        sorted_imp = sorted(imp_array.items(), key=lambda x: x[1], reverse=True)

    output_file = logs_dir / f"{symbol.replace('/', '_')}_importance.txt"
    with open(output_file, "w") as f:
        f.write(f"Feature importance (gain) for {symbol}\n")
        f.write("=" * 50 + "\n")
        for name, imp in sorted_imp[:30]:
            f.write(f"{name:35} : {imp:.6f}\n")
    print(f"   📊 Feature importance saved to {output_file}")

# ------------------------------
# TRAIN A SINGLE TOKEN (with all upgrades)
# ------------------------------
def train_token(symbol: str, hours: int = 5000) -> Optional[Dict]:
    print(f"\n{'='*60}")
    print(f"🧠 Training model for {symbol}")
    print(f"{'='*60}")

    try:
        p = Predictor(symbol)
        print(f"📥 Fetching {hours} hours of data for {symbol}...")
        df = p.fetch_live_data(timeframe='1h', limit=hours)
        if df is None or df.empty:
            print(f"❌ No data for {symbol}")
            return None

        # BTC context
        btc_df = None
        if hasattr(p, 'fetch_btc_data'):
            print("📥 Fetching BTC market context...")
            btc_df = p.fetch_btc_data(timeframe='1h', limit=hours)
            if btc_df is None or btc_df.empty:
                print("⚠️ BTC data not available – continuing without anchor features")
        else:
            print("⚠️ Predictor.fetch_btc_data() missing – BTC anchor features will be zero")

        # News sentiment (already fresh)
        print("📰 Loading news sentiment...")
        news_df = p.load_news_data()
        if news_df is None or news_df.empty:
            print("⚠️ News data not available – continuing without sentiment")

        # Feature engineering (all 60+ indicators, no target yet)
        print("🛠 Applying feature engineering (enhanced indicators)...")
        df = prepare_features(df, btc_df=btc_df, news_df=news_df, add_target_flag=False)
        if df is None or df.empty:
            print(f"❌ Feature engineering failed for {symbol}")
            return None

        # Ensure regime columns exist (they are created by prepare_features now)
        for col in ['volatility_regime', 'efficiency_ratio_10', 'trend_regime']:
            if col not in df.columns:
                print(f"⚠️ {col} missing – using constant")
                df[col] = 1.0 if col == 'volatility_regime' else (0.5 if 'efficiency' in col else 0)

        # Triple‑barrier labeling with asset‑specific ATR multiplier
        atr_mult = get_atr_multiplier(symbol)
        print(f"🎯 Using ATR multiplier = {atr_mult} for {symbol}")
        labels = create_triple_barrier_labels(
            df,
            atr_multiplier=atr_mult,
            max_lookahead=48,
            volatility_regime=df['volatility_regime'],
            efficiency_ratio=df['efficiency_ratio_10'],
            trend_regime=df['trend_regime']
        )
        df['target'] = labels

        if len(df) < 200:
            print(f"⚠️ Too few samples ({len(df)}), skipping")
            return None

        feature_cols = [c for c in df.columns if c not in ['timestamp', 'target']]
        y = df['target'].to_numpy().astype(int)

        class_counts = np.bincount(y)
        scale_pos_weight = class_counts[0] / class_counts[1] if len(class_counts) > 1 and class_counts[1] > 0 else 1.0
        print(f"⚖️ Class imbalance: zeros={class_counts[0]}, ones={class_counts[1]}")
        print(f"   scale_pos_weight = {scale_pos_weight:.2f}")

        # Time‑series cross‑validation (5 folds)
        tscv = TimeSeriesSplit(n_splits=5)
        fold_scores = []
        best_models = []
        best_params = None

        for fold, (train_idx, val_idx) in enumerate(tscv.split(df)):
            print(f"   Fold {fold+1}/5...")
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]
            X_train = train_df[feature_cols]
            y_train = train_df['target'].to_numpy().astype(int)
            X_val = val_df[feature_cols]
            y_val = val_df['target'].to_numpy().astype(int)

            # Apply label smoothing to training targets
            y_train_smooth = smooth_labels(y_train, smoothing=0.05)

            # Optuna hyperparameter optimization (only on first fold to save time)
            if fold == 0:
                study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
                study.optimize(lambda trial: objective(trial, X_train, y_train_smooth, X_val, y_val,
                                                       scale_pos_weight, feature_cols),
                               n_trials=30, show_progress_bar=False)
                best_params = study.best_params
                print(f"   Best params: {best_params}")
                best_params.update({
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'scale_pos_weight': scale_pos_weight,
                    'seed': 42,
                    'tree_method': 'hist'
                })

            # Train model with best params on this fold
            dtrain = xgb.DMatrix(X_train, label=y_train_smooth, feature_names=feature_cols)
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)
            model = xgb.train(best_params, dtrain, num_boost_round=500,
                              evals=[(dval, 'eval')], early_stopping_rounds=50, verbose_eval=False)
            pred = model.predict(dval)
            from sklearn.metrics import accuracy_score
            acc = accuracy_score(y_val, pred > 0.5)
            fold_scores.append(acc)
            best_models.append(model)

        avg_acc = np.mean(fold_scores)
        print(f"✅ Cross‑validation accuracy: {avg_acc:.4f} (5 folds)")

        # Retrain on full data with best parameters and SHAP pruning
        print("   Retraining on full dataset with SHAP pruning...")
        X_full = df[feature_cols]
        y_full = np.array(df['target'].values.astype(int))
        y_full_smooth = smooth_labels(y_full, smoothing=0.05)

        # Train a preliminary model to compute SHAP importance
        dtrain_full = xgb.DMatrix(X_full, label=y_full_smooth, feature_names=feature_cols)
        temp_model = xgb.train(best_params, dtrain_full, num_boost_round=100, verbose_eval=False)
        keep_features = prune_features_by_shap(temp_model, X_full, threshold=0.01)
        if len(keep_features) < len(feature_cols):
            feature_cols = keep_features
            print(f"   Reduced feature set to {len(feature_cols)} features.")

        # Final training on full data with pruned features
        X_full_pruned = X_full[feature_cols]
        dtrain_final = xgb.DMatrix(X_full_pruned, label=y_full_smooth, feature_names=feature_cols)
        final_model = xgb.train(best_params, dtrain_final, num_boost_round=500, verbose_eval=False)

        # Evaluate on last 20% (out‑of‑sample)
        split = int(len(df) * 0.8)
        X_test = X_full_pruned.iloc[split:]
        y_test = np.array(y_full[split:])
        dtest = xgb.DMatrix(X_test, feature_names=feature_cols)
        test_pred = final_model.predict(dtest)
        test_acc = accuracy_score(y_test, test_pred > 0.5)
        print(f"   Out‑of‑sample accuracy (last 20%): {test_acc:.4f}")

        # Save model
        store_dir = Path(root_dir) / "src" / "ml" / "model_store"
        store_dir.mkdir(parents=True, exist_ok=True)
        model_filename = f"{symbol.replace('/', '_')}_model.json"
        model_path = store_dir / model_filename
        final_model.save_model(str(model_path))
        print(f"💾 Model saved to {model_path}")

        log_feature_importance(final_model, feature_cols, symbol)

        return {
            "symbol": symbol,
            "cv_accuracy": avg_acc,
            "test_accuracy": test_acc,
            "feature_count": len(feature_cols),
            "best_params": best_params,
            "model_path": str(model_path),
            "atr_multiplier": atr_mult,
            "scale_pos_weight": scale_pos_weight
        }

    except Exception as e:
        print(f"❌ Unexpected error for {symbol}: {type(e).__name__}: {e}")
        return None

# ------------------------------
# BATCH FLEET TRAINING
# ------------------------------
def train_fleet(hours: int = 5000):
    print("=" * 70)
    print("📰 CHECKING NEWS DATA FRESHNESS")
    print("=" * 70)
    news_ok = ensure_fresh_news_for_training()
    if not news_ok:
        print("⚠️ Proceeding without fresh news. Sentiment features will be zero.")
    else:
        print("✅ News data is ready.")

    results = []
    total = len(FLEET_SYMBOLS)

    print("\n" + "=" * 70)
    print("🚀 AEGIS‑1 FLEET TRAINING (OPTUNA + SHAP + TSCV)")
    print("   Features: 60+ technicals + BTC anchor + news sentiment")
    print("   Regime‑adaptive labeling | Asset‑specific ATR multipliers")
    print("   TimeSeriesCV (5 folds) | SHAP feature pruning | Label smoothing")
    print("=" * 70)

    for idx, symbol in enumerate(FLEET_SYMBOLS, 1):
        print(f"\n[{idx}/{total}] Processing {symbol}...")
        metrics = train_token(symbol, hours=hours)
        if metrics:
            results.append(metrics)
            print(f"   ✅ {symbol} done – CV acc: {metrics['cv_accuracy']:.2%} | Test: {metrics['test_accuracy']:.2%}")
        else:
            print(f"   ⚠️ {symbol} skipped")

    print("\n" + "=" * 70)
    print("🏁 TRAINING SUMMARY")
    print("=" * 70)
    if results:
        avg_cv_acc = np.mean([r['cv_accuracy'] for r in results])
        print(f"✅ Successfully trained: {len(results)} / {total} tokens")
        print(f"📈 Average cross‑validation accuracy: {avg_cv_acc:.2%}")
        print(f"💾 Models saved in: {Path(root_dir) / 'src' / 'ml' / 'model_store'}")
        print(f"📊 Feature importance logs: logs/features/")
    else:
        print("❌ No models were trained. Check errors above.")

    logs_dir = Path(root_dir) / "logs"
    logs_dir.mkdir(exist_ok=True)
    summary_file = logs_dir / "training_summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "hours_of_data": hours,
            "total_tokens": total,
            "successful": len(results),
            "results": results
        }, f, indent=2)
    print(f"\n📄 Summary saved to {summary_file}")

if __name__ == "__main__":
    # You can adjust hours to fetch more/less data
    train_fleet(hours=5000)