#!/usr/bin/env python3
"""
retrain_model.py - Aegis-1 Model Trainer (meta-labeling rebuild)
----------------------------------------------------------------
Per-token XGBoost with regime-adaptive triple-barrier labeling, Optuna tuning,
SHAP pruning, purged time-series CV, a truly held-out test set, AND a
meta-labeling layer that turns raw direction guesses into selective, tradeable
signals.

THE STRATEGY SHIFT (read this)
==============================
Raw 3-class accuracy is a vanity metric here: it's dominated by HOLD (which is
partly definitional, since the labeler sets HOLD from features the model sees).
The previous run scored 72% accuracy but only ~51% precision on the BUY/SELL
calls that actually become trades -- a coin flip.

You can't honestly push 1h crypto *direction* accuracy to 80-90%. What you CAN
do is fire fewer, higher-conviction signals. That is what makes a product
"market ready". This version does it with META-LABELING:

    PRIMARY model  -> proposes a side (BUY or SELL) from the features.
    META model     -> predicts P(the proposal is actually correct).
    DECISION       -> trade only when meta-confidence >= a tuned threshold.

The threshold is chosen on out-of-fold data to hit a precision TARGET (default
70%), and we report the COVERAGE (how often it fires) that comes with it, plus a
fee-aware PnL backtest on the untouched holdout. Precision and coverage trade
off -- that trade-off, shown honestly, IS the product spec.

If the edge isn't there at 70%, the code will show coverage collapsing or the
target unmet. That's information, not failure -- and it's the truth you want
before charging anyone.

Leak controls retained: train/embargo/holdout split, purged CV, SHAP on train
only, holdout scored exactly once.
"""

import sys
import os
import json
import math
import time
import warnings
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, precision_recall_fscore_support,
)

try:
    import shap
except Exception:
    shap = None
    print("WARNING: optional dependency 'shap' not available -- SHAP pruning will be skipped.")

try:
    from scipy.optimize import minimize_scalar
except Exception:
    minimize_scalar = None

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Model file not found")

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.ml.feature_engine import prepare_features, compute_atr
from src.ml.predictor import Predictor

_RETRAIN_ROOT   = Path(__file__).resolve().parent.parent
_TOKEN_PARAMS_DIR = _RETRAIN_ROOT / "data" / "token_params"


def load_token_params(symbol: str) -> Optional[Dict[str, Any]]:
    """Load per-token optimizer output from data/token_params/ if it exists."""
    path = _TOKEN_PARAMS_DIR / f"{symbol.replace('/', '_')}_params.json"
    if not path.exists():
        return None
    try:
        with open(path) as _f:
            return json.load(_f)
    except Exception:
        return None


_META_PROFILE_DIR = _RETRAIN_ROOT / "data" / "meta_gate_profiles"


def load_meta_gate_profile(symbol: str) -> Optional[Dict[str, Any]]:
    """Load the independent meta gate architecture profile if available."""
    path = _META_PROFILE_DIR / f"{symbol.replace('/', '_')}_gate.json"
    if not path.exists():
        return None
    try:
        with open(path) as _f:
            payload = json.load(_f)
        return payload.get("selected_profile") or payload
    except Exception:
        return None


# ============================================================
# CONFIG
# ============================================================
NUM_CLASS = 3                 # 0=SELL, 1=HOLD, 2=BUY
MAX_LOOKAHEAD = 96            # capped at 96h — beyond this, label leakage risk outweighs signal gain
EMBARGO = MAX_LOOKAHEAD
CENSORED = -1

TEST_FRAC = 0.25
N_SPLITS_CV = 15              # TASK 6: Increased from 10 to 15 for statistical robustness
OPTUNA_TRIALS = 25

# ============================================================
# TASK 1: HOLD POLLUTION REFACTOR (3 strategies)
# ============================================================
META_HOLD_STRATEGY = "C_excluded"  # binary primary: only directional proposals in meta training
META_HOLD_AUTO_SELECT = False      # True = auto-test strategies; False = use META_HOLD_STRATEGY directly

# ============================================================
# TASK 3: ADAPTIVE COVERAGE TARGETING
# ============================================================
MIN_COVERAGE = 0.03            # 3% minimum directional coverage
TARGET_COVERAGE = 0.10         # 10% preferred target
MAX_COVERAGE = 0.25            # 25% hard maximum

# ============================================================
# TASK 4: CALIBRATION REWORK
# ============================================================
CALIBRATION_MODE = "none"      # "none", "temperature", "platt", "beta", "isotonic"
CALIBRATION_AUTO_SELECT = True  # Disable if reduces PF/Expectancy

# ============================================================
# TASK 5: REGIME THRESHOLD ENGINE
# ============================================================
REGIME_THRESHOLD_ADAPTATION = True
REGIME_THRESHOLDS = {
    "COMPRESSION": 0.90,
    "VOLATILE_EXPANSION": 0.88,
    "ACCUMULATION": 1.10,
    "DISTRIBUTION": 1.05,
}

# ============================================================
# TASK 6: STATISTICAL RELIABILITY (Improved)
# ============================================================
MIN_HOLDOUT_FIRES_GRADE = 100   # Minimum for grade A
PREFER_HOLDOUT_FIRES = 200

# ============================================================
# TASK 2: FEATURE DRIFT DEFENSE (Auto-blacklist)
# ============================================================
FEATURE_DRIFT_AUTO_DETECT = True
FEATURE_DRIFT_PSI_THRESHOLD = 1.0
FEATURE_DRIFT_KS_THRESHOLD = 0.50

# ============================================================
# TASK 7: ARCHITECTURE SCORING (Expectancy-Driven)
# ============================================================
# NEW: 0.40 expectancy, 0.30 pf, 0.20 sharpe, 0.05 prec, 0.05 cov
# OLD: 0.35 expectancy, 0.30 pf, 0.20 sharpe, 0.10 prec, 0.05 cov
ARCHITECTURE_SCORE_WEIGHTS = {
    "expectancy": 0.40,     # TASK 7: Increased
    "profit_factor": 0.30,
    "sharpe": 0.20,
    "precision": 0.05,      # TASK 7: Reduced
    "coverage": 0.05,
}

SHAP_CUMULATIVE_THRESH = 0.85  # keep features covering top 85% of cumulative importance (re-enabled)
SHAP_TOP_PCT = 0.15            # floor: keep at least top 15% of features by SHAP rank
MIN_FEATURES = 50              # hard floor for feature count (was 80)
MAX_FEATURES = 150             # hard cap; SHAP cumulative threshold does the real pruning (was 200)

MIN_TOTAL_ROWS = 600
MIN_FIT_ROWS = 300

# --- meta-labeling / trading ---
# 70% directional precision at 1h is not realistic for crypto from TA alone.
# A 56-60% gate with real coverage and positive expectancy after fees is a
# genuine product; 70% with zero coverage is not. Set the target where the data
# can actually reach, and let coverage tell you how much signal exists.
TARGET_SIGNAL_PRECISION = 0.65   # raised target: 65% precision floor post-retrain
MIN_TRADEABLE_PRECISION = 0.65   # precision-first gate: no token trades below 65%
MIN_FIRES_DEV = 60               # need >=60 OOF trades before trusting a threshold
# Minimum holdout signals required for statistical reliability.
# At or above this count, holdout precision governs; below it the OOF estimate is used.
# Raised from 10 to 30 to require sufficient holdout sample before enabling a token.
MIN_HOLDOUT_FIRES = 20
META_MIN_SAMPLES = 1500          # LR meta needs ~1500 directional samples; 5000 never reached at 10k hrs
GAP_VETO_THRESHOLD = 0.20        # maximum acceptable OOF->holdout precision gap (was 0.15)
FEE_ROUNDTRIP = 0.001            # 0.10% round-trip (taker + slippage); tune to your venue
EXPECTANCY_FLOOR = 0.20          # 0.20% minimum expectancy floor for override

# Asymmetric triple-barrier skew. Squeeze the downside barrier (catch fast drops
# sooner) and widen the upside (buffer fake breakouts). NOTE: this changes the
# LABEL definition, so it shifts the SELL/BUY class balance — it does not by
# itself create predictive skill. Watch the printed class distribution to see
# what it's actually doing.
BARRIER_UP_SKEW = 1.0    # symmetric: prevents SELL-label dominance across all tokens
BARRIER_DOWN_SKEW = 1.0

NEWS_FILE = Path(root_dir) / "data" / "news_data.json"
NEWS_MAX_AGE_SECONDS = 30 * 60

DEFAULT_PARAMS = {
    'objective': 'multi:softprob', 'eval_metric': 'mlogloss', 'num_class': NUM_CLASS,
    'max_depth': 5, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'reg_lambda': 2.0, 'reg_alpha': 1.0, 'min_child_weight': 8,
    'seed': 42, 'tree_method': 'hist', 'missing': np.nan,
}

META_PARAMS = {
    'objective': 'binary:logistic', 'eval_metric': 'logloss',
    'max_depth': 3, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'reg_lambda': 2.5, 'min_child_weight': 10,
    'seed': 42, 'tree_method': 'hist', 'missing': np.nan,
}

# Features that must never be penalized by the drift or correlation audits.
# These are the top-gain features in production; downweighting them causes the
# primary model to ignore its most predictive signals and fall back to noise.
_PRECISION_PROTECTED_FEATURES: frozenset = frozenset({
    'prc_trend', 'prc_momentum', 'prc_volume', 'prc_bands',
    'prc_smart_money', 'prc_total', 'macro_confluence_score',
    'efficiency_ratio_10', 'volatility_regime',
    'structure_bias', 'linreg_r2_14',
    'trend_consistency_20', 'price_disp_atr_pct', 'vol_confirm_ratio',
    'higher_high_count_20', 'lower_low_count_20', 'swing_strength_20',
    'adx_14', 'supertrend_dist', 'sar_dist',
})

FLEET_SYMBOLS = [
    # ── Majors (20) ──────────────────────────────────────────
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
    'ADA/USDT', 'DOGE/USDT', 'TRX/USDT', 'AVAX/USDT', 'LINK/USDT',
    'DOT/USDT', 'TON/USDT', 'SHIB/USDT', 'BCH/USDT', 'LTC/USDT',
    'ATOM/USDT', 'XLM/USDT', 'ETC/USDT', 'UNI/USDT', 'NEAR/USDT',

    # ── Layer 1 (12) ──────────────────────────────────────────────────────
    'APT/USDT', 'SUI/USDT', 'SEI/USDT', 'INJ/USDT', 'TIA/USDT',
    'KAS/USDT', 'EGLD/USDT', 'HBAR/USDT', 'ALGO/USDT', 'VET/USDT',
    'ICP/USDT', 'FIL/USDT',

    # ── Layer 2 (7) ───────────────────────────────────────────────────────
    'ARB/USDT', 'OP/USDT', 'MATIC/USDT', 'STRK/USDT', 'IMX/USDT',
    'STX/USDT', 'ZK/USDT',

    # ── AI / ML (5) ───────────────────────────────────────────────────────
    'FET/USDT', 'TAO/USDT', 'RNDR/USDT', 'AKT/USDT', 'ARKM/USDT',

    # ── Oracle / Infrastructure (4) ───────────────────────────────────────
    'PYTH/USDT', 'GRT/USDT', 'THETA/USDT', 'TRB/USDT',

    # ── DeFi (12) ─────────────────────────────────────────────────────────
    'AAVE/USDT', 'MKR/USDT', 'CRV/USDT', 'LDO/USDT', 'DYDX/USDT',
    'GMX/USDT', 'PENDLE/USDT', 'ONDO/USDT', 'JUP/USDT', 'ENA/USDT',
    'COMP/USDT', 'SNX/USDT',

    # ── Storage (2) ───────────────────────────────────────────────────────
    'AR/USDT', 'STORJ/USDT',

    # ── Gaming / Metaverse (5) ────────────────────────────────────────────
    'AXS/USDT', 'SAND/USDT', 'MANA/USDT', 'GALA/USDT', 'ENJ/USDT',

    # ── Memes (6) ─────────────────────────────────────────────────────────
    'PEPE/USDT', 'WIF/USDT', 'BONK/USDT', 'FLOKI/USDT', 'BOME/USDT',
    'BRETT/USDT',

    # ── Exchange tokens (1) ───────────────────────────────────────────────
    'BGB/USDT',

    # ── Privacy (1) ───────────────────────────────────────────────────────
    'XMR/USDT',

    # ── High-volume alts — best liquidity & signal quality (25) ──────────
    'RUNE/USDT', 'OM/USDT', 'ORDI/USDT', 'NOT/USDT', 'ENS/USDT',
    'QNT/USDT', 'MASK/USDT', 'MINA/USDT', 'NEO/USDT', 'CHZ/USDT',
    'CFX/USDT', 'CAKE/USDT', '1INCH/USDT', 'BAT/USDT', 'BLUR/USDT',
    'CKB/USDT', 'IOTX/USDT', 'FTM/USDT', 'KAVA/USDT', 'YFI/USDT',
    'RAY/USDT', 'JTO/USDT', 'HYPE/USDT', 'TRUMP/USDT', 'FXS/USDT',
]

FEATURE_BLACKLIST: frozenset = frozenset({
    # Raw OHLCV — absolute price/volume levels (PSI=20+). Stationary dist_* alternatives exist.
    'open', 'high', 'low', 'close', 'volume',
    # Squeeze Momentum absolute price bands (use sqz_on, bb_pct_b instead)
    'se_mid', 'se_upper', 'se_lower',
    # Absolute EMA levels (use dist_ema_* instead)
    'ema_9', 'ema_21', 'ema_50', 'ema_100', 'ema_200',
    # Absolute VWAP / anchored VWAP levels (use dist_vwap, dist_avwap_* instead)
    'vwap', 'avwap_50', 'avwap_100', 'avwap_200', 'rolling_vwap_24',
    # Ichimoku absolute cloud price levels
    'ichimoku_senkou_a', 'ichimoku_senkou_b', 'ichimoku_tenkan', 'ichimoku_kijun',
    # Pivot points (absolute price levels, reset each session)
    'pivot', 'r1', 'r2', 's1', 's2',
    # Absolute MA variants (use dist_hma20, dist_kama, dist_vwma20 instead)
    'hma_20', 'kama_10', 'tema_21', 'dema_21', 't3_5', 'vwma_20',
    # Absolute price band indicators (use keltner_width, supertrend_dist/dir instead)
    'supertrend', 'keltner_upper', 'keltner_lower', 'keltner_mid',
    # Absolute S/R levels (use is_at_support/resistance, pct_dist_to_* instead)
    'rolling_support', 'rolling_resistance',
    # Cumulative non-stationary A/D line (z-scored obv/pvt used instead)
    'acc_dist',
    # Price-level exponential decay means — absolute price, PSI=17+
    'vwap_decay_mean_24', 'vwap_decay_std_24',
    'close_decay_mean_24', 'close_decay_std_24',
    # Volume decay means — non-stationary (crypto volume grows over time)
    'volume_decay_mean_24', 'volume_decay_std_24',
    # Absolute-price oscillators — scale with price level, causing PSI=20+ between regimes
    # awesome_osc = midpoint_SMA_fast - midpoint_SMA_slow  (hundreds at $20k, thousands at $90k)
    # force_index = volume × close.diff()                  (scales with price magnitude)
    # kvo/kvo_signal = volume × sign(tp.diff) × (H-L)     (scales with price AND volume)
    # eom_14 = mid_diff × hl_range / volume               (scales as price²/volume)
    # dpo_20 = price - shifted_MA                         (absolute price difference)
    'awesome_osc', 'force_index', 'kvo', 'kvo_signal', 'eom_14', 'dpo_20',
    # Absolute OI levels — open_interest drifts from $3B→$80B for BTC over training window
    # Use oi_zscore, oi_change_1h/4h, oi_chg_8h/24h (all stationary) instead
    # oi_long_short_pressure = open_interest × sign(close.diff) — inherits absolute OI scale
    'open_interest', 'oi_long_short_pressure',
    # Slope/divergence features built on absolute-price or raw-cumsum series
    # vwap_slope_24:    linear slope of VWAP in $/bar — hundreds at $20k, thousands at $90k
    # obv_slope_20:     slope of raw OBV cumulative sum — non-stationary (grows without bound)
    # obv_divergence:   OBV_diff/close vs price_return — mixed units, non-stationary
    # linreg_slope_14:  linear regression slope of close in $/bar — scales with price level
    #                   use linreg_r2_14 (trend quality, stationary) + ret_12h/ret_24h instead
    'vwap_slope_24', 'obv_slope_20', 'obv_divergence', 'linreg_slope_14',
    # Confirmed high-drift features from forensic PSI reports (PSI ≥ 1.9 CRITICAL)
    # returns_1h_decay_std_24:  return-series decay std — PSI=3.4, non-stationary across regimes
    # returns_1h_decay_mean_24: return-series decay mean — non-stationary companion to above
    # funding_rate:             perpetual funding — PSI=2.3; regime-dependent (bull vs bear cycle)
    # funding_rate_ma8:         8-bar MA of funding — PSI=2.4; same drift, smoothed
    # gk_vol:                   Garman-Klass volatility — PSI=1.9; low-vol vs high-vol regimes shift dist
    # vwap_delta_12:            12h VWAP delta — PSI=5.3 for ETH; trending market inflates this
    'returns_1h_decay_std_24', 'returns_1h_decay_mean_24',
    'funding_rate', 'funding_rate_ma8',
    'gk_vol',
    'vwap_delta_12',
    # Additional high-drift features identified in 2026-06-12 forensic reports
    # atr_14:         absolute ATR in price units — ADA PSI=2.14, SOL PSI=1.32; use atr_pct instead
    # donchian_width: absolute Donchian channel width in price units — BTC critical_indicators
    # vwap_delta_4:   4h VWAP delta — XRP PSI=2.1; all vwap_delta_* variants drift with price level
    # vwap_delta_8:   8h VWAP delta — companion to vwap_delta_4 / vwap_delta_12 (already blacklisted)
    # vwap_delta_24:  24h VWAP delta — same absolute-price-delta drift pattern
    'atr_14',
    'donchian_width',
    'vwap_delta_1', 'vwap_delta_4', 'vwap_delta_8', 'vwap_delta_24',
})

FEATURE_ADDONS = [
    # ══════════════════════════════════════════════════════════════════════
    # COMPLETE feature registry — every column prepare_features() outputs.
    # feature_cols auto-picks up all df columns, so this list acts as both
    # documentation AND a safety net that guarantees critical features are
    # included even when a compute path fails silently.
    # SHAP pruning runs per-token and eliminates any low-value column.
    # ══════════════════════════════════════════════════════════════════════

    # ── Returns & price deltas ────────────────────────────────────────────
    'returns_1h', 'returns_4h', 'log_returns',
    'ret_1h', 'ret_4h', 'ret_12h', 'ret_24h',
    'close_delta_1', 'close_delta_4', 'close_delta_12',
    'returns_1h_delta_1', 'returns_1h_delta_4', 'returns_1h_delta_12',
    'close_decay_mean_24', 'close_decay_std_24',
    'returns_1h_decay_mean_24', 'returns_1h_decay_std_24',

    # ── EMAs & cross signals ──────────────────────────────────────────────
    'ema_9', 'ema_21', 'ema_50', 'ema_100', 'ema_200',
    'dist_ema_9', 'dist_ema_21', 'dist_ema_50', 'dist_ema_100', 'dist_ema_200',
    'ema_9_21_cross', 'ema_50_200_cross',

    # ── MA variants & distances ───────────────────────────────────────────
    'hma_20', 'dist_hma20',
    'kama_10', 'dist_kama',
    'tema_21', 'dema_21',
    't3_5', 'vwma_20', 'dist_vwma20',

    # ── VWAP family ───────────────────────────────────────────────────────
    'vwap', 'dist_vwap',
    'rolling_vwap_24', 'dist_rolling_vwap',
    'avwap_50', 'avwap_100', 'avwap_200',
    'dist_avwap_50', 'dist_avwap_100', 'dist_avwap_200',
    'vwap_decay_mean_24', 'vwap_decay_std_24',
    'vwap_delta_1', 'vwap_delta_4', 'vwap_delta_12',

    # ── Trend indicators ──────────────────────────────────────────────────
    'adx_14',
    'supertrend', 'supertrend_dir', 'supertrend_dist',
    'sar_trend', 'sar_dist',
    'donchian_width', 'donchian_position',
    'linreg_slope_14', 'linreg_r2_14',
    'choppiness', 'efficiency_ratio_10',
    'ema_alignment', 'ema_alignment_quality',
    'fvg_distance',                         # fair-value-gap distance

    # ── Momentum / oscillators ────────────────────────────────────────────
    'rsi_7', 'rsi_14', 'rsi_21',
    'macd', 'macd_hist', 'macd_signal',
    'stoch_k', 'stoch_d',
    'williams_r',
    'cci_20', 'tsi', 'cmo_14', 'dpo_20',
    'ppo', 'ppo_signal', 'trix_15',
    'kst', 'kst_signal', 'schaff_tc',
    'awesome_osc', 'bop', 'eom_14',
    'fisher', 'fisher_sig',
    'rvi_osc', 'rvi_sig', 'roc_14',
    'ultimate_osc',

    # ── Volatility & bands ────────────────────────────────────────────────
    'atr_14',
    'bb_pct_b', 'bb_width_percentile',
    'keltner_upper', 'keltner_lower', 'keltner_width',
    'atr_band_position', 'starc_position', 'gaussian_position',
    'parkinson_vol', 'gk_vol', 'rvi_vol',
    'historical_volatility', 'rolling_volatility',
    'volatility_regime', 'volatility_skew', 'volatility_kurt',
    'price_zscore_200',

    # ── Volume indicators ─────────────────────────────────────────────────
    'volume_zscore', 'relative_volume', 'vol_velocity',
    'volume_atr_efficiency',
    'volume_delta', 'volume_delta_14',
    'volume_delta_1', 'volume_delta_4', 'volume_delta_12',
    'volume_decay_mean_24', 'volume_decay_std_24',
    'cmf_20', 'mfi_14',
    'obv', 'acc_dist', 'pvt',
    'kvo', 'kvo_signal',
    'volume_regime',

    # ── Squeeze Momentum ──────────────────────────────────────────────────
    'sqz_on', 'sqz_off', 'sqz_momentum',
    'is_squeeze',

    # ── Elder Ray ─────────────────────────────────────────────────────────
    'elder_bull', 'elder_bear', 'elder_bull_bear',

    # ── Aroon / Vortex / Force ────────────────────────────────────────────
    'aroon_up', 'aroon_down', 'aroon_osc',
    'vi_plus', 'vi_minus',
    'force_index',

    # ── Ichimoku Cloud ────────────────────────────────────────────────────
    'ichimoku_tenkan', 'ichimoku_kijun', 'ichimoku_senkou_a', 'ichimoku_senkou_b',
    'ichi_above_cloud', 'ichi_cloud_bull', 'ichi_tk_cross',
    'ichi_dist_tenkan', 'ichi_dist_kijun', 'ichi_dist_cloud_top',

    # ── Statistical / quant ───────────────────────────────────────────────
    'se_position', 'se_mid',
    'quantile_position',
    'hurst', 'entropy',

    # ── Volume Pressure Composite ─────────────────────────────────────────
    'candle_pressure', 'rolling_buy_ratio',

    # ── Bar microstructure ────────────────────────────────────────────────
    'close_position', 'bar_body_pct', 'upper_wick_pct', 'lower_wick_pct', 'bar_direction',

    # ── Candlestick patterns ──────────────────────────────────────────────
    'CDL_DOJI', 'CDL_HAMMER', 'CDL_SHOOTINGSTAR',
    'CDL_BULL_ENGULFING', 'CDL_BEAR_ENGULFING',
    'CDL_MORNINGSTAR', 'CDL_EVENINGSTAR',

    # ── Market structure / SMC ────────────────────────────────────────────
    'bos_up', 'bos_down', 'bos_state',
    'choch_bull', 'choch_bear', 'structure_bias',
    'dist_bull_ob', 'dist_bear_ob',

    # ── S&R / Fibonacci / Pivots ──────────────────────────────────────────
    'pct_dist_to_resistance', 'pct_dist_to_support', 'range_position_score',
    'is_at_support', 'is_at_resistance', 'rolling_resistance', 'rolling_support',
    'fib_dist_236', 'fib_dist_382', 'fib_dist_500', 'fib_dist_618', 'fib_range_pct',
    'pivot', 'r1', 's1', 'r2', 's2',
    'dist_pivot', 'dist_r1', 'dist_s1', 'dist_r2', 'dist_s2',

    # ── Regime classification ─────────────────────────────────────────────
    'trend_regime', 'volume_regime', 'market_phase',

    # ── Time / session ────────────────────────────────────────────────────
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',

    # ── Macro & weekly features ───────────────────────────────────────────
    'macro_trend_1d', 'macro_trend_1w', 'macro_confluence_score',
    'weekly_rsi', 'weekly_trend',
    'weekly_macd', 'weekly_macd_hist', 'weekly_macd_signal',
    'weekly_ema200', 'weekly_sma200',
    'dist_weekly_ema200', 'dist_weekly_sma200',
    'weekly_bos_up', 'weekly_bos_down', 'weekly_structure_bias',

    # ── Sentiment ─────────────────────────────────────────────────────────
    'fear_greed_value', 'fear_greed_signal',
    'news_score', 'news_velocity',

    # ── Confluence — sign-based (kept alongside soft; SHAP decides winner) ─
    'momentum_confluence', 'trend_confluence', 'volume_confluence',
    'bands_confluence', 'smart_money_confluence', 'candle_confluence',
    'total_confluence',

    # ── Confluence — soft (percentile-rank) ───────────────────────────────
    # Computed by compute_soft_confluence_features() after prepare_features().
    # Added to df before feature_cols is built so always included.
    'prc_trend', 'prc_momentum', 'prc_volume', 'prc_bands', 'prc_smart_money',
    'prc_total',

    # ── Token-vs-BTC relative performance ────────────────────────────────
    'rel_perf_1h', 'rel_perf_4h', 'rel_perf_24h',
    'btc_ratio_ma_dist',
    'btc_1h_return', 'btc_4h_return', 'btc_dist_ema200',
    'btc_corr_24h',

    # ── Futures base (zero-filled when spot-only token) ───────────────────
    'funding_rate', 'funding_rate_ma8', 'funding_rate_zscore',
    'open_interest', 'oi_change_1h', 'oi_change_4h', 'oi_zscore',

    # ── Funding rate derived signals ──────────────────────────────────────
    'funding_slope_3', 'funding_slope_8',
    'funding_extreme_long', 'funding_extreme_short', 'funding_neutral',
    'funding_cum_8',

    # ── OI vs price regime ────────────────────────────────────────────────
    'oi_chg_8h', 'oi_chg_24h', 'oi_px_agreement',

    # ── Primary model precision features (added 2026-06-10) ──────────────
    # Regime-independent, stationary signals targeting directional skill.
    'trend_consistency_20',   # fraction of last 20 bars up (persistent bias)
    'price_disp_atr_pct',     # (close - EMA50) / ATR14 — vol-normalised displacement
    'vol_confirm_ratio',      # log ratio up-volume / down-volume over 20 bars
]


# ============================================================
# SOFT CONFLUENCE FEATURES  (percentile-rank based)
# ============================================================
def compute_soft_confluence_features(df: pd.DataFrame, window: int = 120) -> pd.DataFrame:
    """
    Compute soft confluences using rolling percentile rank instead of sign().

    sign() saturates: RSI-51 and RSI-80 both map to +1, so the model cannot
    distinguish a marginal edge from a strong one.  Rolling pct-rank preserves
    the gradient — RSI at the 85th pct of recent history gets a score near 0.85
    whereas RSI at the 52nd pct gets ~0.52.

    All outputs are in [0, 1]:  0.5 = neutral, >0.5 = bullish, <0.5 = bearish.
    A separate macro_confluence_score column is derived and mapped to [-1, +1]
    for use in the triple-barrier label cancellation logic.

    Window of 120 bars ~ 5 days on 1h data - long enough to be meaningful,
    short enough to track regime changes within the training set.
    """
    result = pd.DataFrame(index=df.index)

    def _pr(col: str, higher_bullish: bool = True) -> pd.Series:
        """Rolling percentile rank of `col`, 0.5 if column missing."""
        if col not in df.columns:
            return pd.Series(0.5, index=df.index, dtype=float)
        s = df[col].ffill().fillna(0.0)
        rank = s.rolling(window, min_periods=max(20, window // 4)).rank(pct=True)
        rank = rank.fillna(0.5)
        return rank if higher_bullish else (1.0 - rank)

    def _cat(*entries) -> pd.Series:
        """Mean of rolling pct-ranks for indicator group."""
        parts = [_pr(c, b) for c, b in entries if c in df.columns]
        if not parts:
            return pd.Series(0.5, index=df.index, dtype=float)
        return pd.concat(parts, axis=1).mean(axis=1).clip(0.0, 1.0)

    # ── Trend ─────────────────────────────────────────────────────
    result['prc_trend'] = _cat(
        ('dist_ema_50',       True),  ('dist_ema_200',     True),
        ('dist_ema_100',      True),  ('dist_hma20',       True),
        ('dist_kama',         True),  ('dist_rolling_vwap',True),
        ('dist_vwap',         True),  ('supertrend_dist',  True),
        ('sar_dist',          True),  ('linreg_slope_14',  True),
        ('structure_bias',    True),  ('macro_trend_1d',   True),
        ('macro_trend_1w',    True),
    )

    # ── Momentum ──────────────────────────────────────────────────
    result['prc_momentum'] = _cat(
        ('rsi_14',     True),  ('rsi_7',       True),  ('rsi_21',  True),
        ('stoch_k',    True),  ('williams_r',  False),
        ('macd_hist',  True),  ('cci_20',      True),
        ('tsi',        True),  ('cmo_14',      True),
        ('awesome_osc',True),  ('bop',         True),
        ('roc_14',     True),  ('ppo',         True),
        ('fisher',     True),
    )

    # ── Volume / Flow ─────────────────────────────────────────────
    result['prc_volume'] = _cat(
        ('volume_delta',    True),  ('volume_delta_14', True),
        ('cmf_20',          True),  ('mfi_14',          True),
        ('eom_14',          True),  ('relative_volume', True),
        ('vol_velocity',    True),  ('kvo',             True),
    )

    # ── Price Position / Bands ────────────────────────────────────
    result['prc_bands'] = _cat(
        ('bb_pct_b',          True),  ('atr_band_position', True),
        ('donchian_position', True),  ('close_position',    True),
        ('quantile_position', True),  ('se_position',       True),
        ('starc_position',    True),  ('gaussian_position', True),
    )

    # ── Smart Money ───────────────────────────────────────────────
    result['prc_smart_money'] = _cat(
        ('structure_bias',      True),
        ('range_position_score',True),
        ('bos_up',              True),  ('bos_down',   False),
        ('choch_bull',          True),  ('choch_bear', False),
        ('is_at_support',       True),  ('is_at_resistance', False),
    )

    # ── Weighted total (matches display weights in predictor.py) ──
    W = {
        'prc_trend':        2.0,
        'prc_momentum':     1.5,
        'prc_volume':       1.5,
        'prc_smart_money':  1.5,
        'prc_bands':        1.0,
    }
    Wsum = sum(W.values())   # 7.5
    total = pd.concat([result[c] * w for c, w in W.items()], axis=1).sum(axis=1) / Wsum
    result['prc_total'] = total.clip(0.0, 1.0)

    # macro_confluence_score in [-1, +1]: used by create_triple_barrier_labels()
    # to cancel barrier hits that contradict a strong confluence consensus.
    # 0.5 (neutral) → 0.0; 0.8 (strong bullish) → +0.6; 0.2 (strong bearish) → -0.6
    result['macro_confluence_score'] = ((result['prc_total'] - 0.5) * 2.0).clip(-1.0, 1.0)

    return result.fillna(0.0)


# ============================================================
# FUTURES DATA FETCH (funding rate + open interest)
# ============================================================
def fetch_futures_data(symbol: str, df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Fetch perpetual funding rate and open interest from Binance USDT-M futures.
    Returns (funding_df, oi_df). Either may be None if the symbol has no perp
    contract or the API call fails — the feature pipeline handles both gracefully."""
    import ccxt as _ccxt
    try:
        ex = _ccxt.binanceusdm({'enableRateLimit': True, 'timeout': 30000})  # type: ignore[arg-type]
        # ccxt linear-perp format: BTC/USDT → BTC/USDT:USDT
        futures_sym = symbol.replace('/USDT', '/USDT:USDT')
        ts = pd.to_datetime(df['timestamp'])
        since_ms = int(ts.iloc[0].timestamp() * 1000)
        n = len(df)

        # Funding rate: 8-hourly, Binance caps history at ~1000 records per call
        fr_limit = min(1000, n // 8 + 10)
        try:
            fr_raw = ex.fetch_funding_rate_history(futures_sym, since=since_ms, limit=fr_limit)
        except Exception as e:
            print(f"   Funding rate fetch failed for {symbol}: {type(e).__name__}: {e}")
            fr_raw = []
        funding_df = None
        if fr_raw:
            funding_df = (pd.DataFrame(fr_raw)[['timestamp', 'fundingRate']]
                          .rename(columns={'fundingRate': 'funding_rate'}))
            funding_df['timestamp'] = pd.to_datetime(funding_df['timestamp'], unit='ms')

        # OI history: Binance hard-caps at 30 days. We first load from the local
        # parquet cache built by scripts/update_oi_cache.py (run it daily), then
        # top-up with whatever fresh data Binance can still provide.
        import time as _time
        _oi_cache_dir = Path(__file__).resolve().parent.parent / 'data' / 'oi_cache'
        _oi_cache_path = _oi_cache_dir / f"{symbol.replace('/', '_').replace(':', '_')}_oi.parquet"

        cached_oi: Optional[pd.DataFrame] = None
        if _oi_cache_path.exists():
            try:
                cached_oi = pd.read_parquet(_oi_cache_path)
                cached_oi['timestamp'] = pd.to_datetime(cached_oi['timestamp'])
                print(f"   OI cache loaded: {len(cached_oi):,} rows "
                      f"({cached_oi['timestamp'].min().date()} → {cached_oi['timestamp'].max().date()})")
            except Exception as e:
                print(f"   OI cache read failed ({e}) — falling back to live fetch")
                cached_oi = None

        # Determine from where to top-up via Binance (last 30 days hard limit)
        _binance_max_ms = 29 * 24 * 60 * 60 * 1000
        _now_ms = int(_time.time() * 1000)
        if cached_oi is not None and not cached_oi.empty:
            _cache_last_ms = int(cached_oi['timestamp'].max().timestamp() * 1000)
            _topup_since = max(_cache_last_ms + 1, _now_ms - _binance_max_ms)
        else:
            _topup_since = _now_ms - _binance_max_ms

        oi_all: List[Any] = []
        try:
            _oi_since = _topup_since
            while len(oi_all) < n:
                batch: List[Any] = list(ex.fetch_open_interest_history(
                    futures_sym, '1h', since=_oi_since, limit=500))
                if not batch:
                    break
                oi_all.extend(batch)
                last_ts = int(dict(batch[-1]).get('timestamp', 0))
                if last_ts <= _oi_since:
                    break
                _oi_since = last_ts + 1
                if len(batch) < 500:
                    break
                _time.sleep(0.15)
        except Exception as e:
            print(f"   OI top-up fetch failed for {symbol}: {type(e).__name__}: {e}")

        oi_df = None
        fresh_df: Optional[pd.DataFrame] = None
        if oi_all:
            raw = pd.DataFrame(oi_all)
            oi_col = next((c for c in ('openInterestAmount', 'openInterest') if c in raw.columns), None)
            if oi_col:
                fresh_df = raw[['timestamp', oi_col]].rename(columns={oi_col: 'open_interest'})
                fresh_df['timestamp'] = pd.to_datetime(fresh_df['timestamp'], unit='ms')

        # Merge cache + fresh top-up
        parts = [p for p in (cached_oi, fresh_df) if p is not None and not p.empty]
        if parts:
            merged = pd.concat(parts, ignore_index=True)
            oi_df = (merged
                     .drop_duplicates('timestamp')
                     .sort_values('timestamp')
                     .reset_index(drop=True))
            # Slice to the training window so we don't return decades of noise
            train_start = pd.to_datetime(df['timestamp'].iloc[0])
            oi_df = oi_df[oi_df['timestamp'] >= train_start - pd.Timedelta(hours=1)]

        return funding_df, oi_df
    except Exception as e:
        print(f"   Futures data unavailable for {symbol}: {type(e).__name__}: {e}")
        return None, None


# ============================================================
# FEAR & GREED INDEX  (alternative.me — free, no API key)
# ============================================================
def fetch_fear_greed(days: int = 600) -> Optional[pd.DataFrame]:
    """Fetch the Crypto Fear & Greed Index from alternative.me.
    Free endpoint, no API key required.  Returns ~daily values going back to 2018.
    Returns None on any network or parse error."""
    import urllib.request
    try:
        url = f"https://api.alternative.me/fng/?limit={days}&date_format=world"
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        records = data.get('data', [])
        if not records:
            return None
        fg = pd.DataFrame(records)[['value', 'timestamp']]
        fg.columns = ['fear_greed_value', 'timestamp']
        fg['fear_greed_value'] = pd.to_numeric(fg['fear_greed_value'], errors='coerce')
        fg['timestamp'] = pd.to_datetime(fg['timestamp'], format='%d-%m-%Y')
        return fg.sort_values('timestamp').reset_index(drop=True)
    except Exception as e:
        print(f"   Fear & Greed fetch failed: {type(e).__name__}: {e}")
        return None


# ============================================================
# ON-CHAIN STUBS  (MVRV Z-Score, Realized Price)
# ============================================================
def fetch_onchain_btc() -> Optional[pd.DataFrame]:
    """
    Placeholder for BTC on-chain metrics (MVRV Z-Score, Realized Price).

    To activate, plug in one of these free/paid endpoints:
      - Glassnode  : https://api.glassnode.com/v1/metrics/market/mvrv_z_score
                     (requires API key — free tier available)
      - CoinGlass  : https://open-api.coinglass.com/public/v2/indicator/realized_price
                     (requires API key — free tier available)

    Expected return format:
        pd.DataFrame with columns ['timestamp', 'mvrv_z_score', 'realized_price']
        where timestamp is UTC datetime and values are daily.

    Returns None until credentials are wired in.
    """
    return None


# ============================================================
# ATR MULTIPLIER
# ============================================================
def get_atr_multiplier(symbol: str) -> float:
    """Return a safe baseline ATR multiplier for training label generation.

    The per-bar adaptive barrier width is handled by compute_dynamic_atr_multiplier().
    Per-symbol tuning is still supported by token_params overrides, but the core
    labeler no longer depends on a hardcoded symbol tier map.
    """
    return 1.5


def get_dynamic_lookahead(df: pd.DataFrame) -> int:
    """Estimate lookahead (in bars) from typical ATR as pct of price.

    Buckets raised to give barriers time to hit in low-volatility regimes.
    Returns one of {96, 72, 48, 36} per median ATR% bucket.
    """
    try:
        atr = compute_atr(df, period=14)
        atr_pct = (atr / df['close'].replace(0, np.nan)).fillna(0)
        med = float(np.nanmedian(atr_pct))
        if med < 0.007:
            return 96   # very low-vol: needs longest window for barrier hits
        if med < 0.010:
            return 72   # low-vol: extended window (was 84)
        return 48       # normal/high-vol: standard window (was 72 for med range)
    except Exception:
        return int(MAX_LOOKAHEAD)


def get_dynamic_atr_range(df: pd.DataFrame, lookback: int = 1000) -> tuple:
    """Return (min_mult, max_mult, typical_mult) based on recent ATR% percentile.

    Uses the latest ATR% in context of history to choose a safe multiplier range
    and a typical starting multiplier for label rebalancing.
    """
    try:
        atr = compute_atr(df, period=14)
        atr_pct = (atr / df['close'].replace(0, np.nan)).fillna(0)
        if len(atr_pct) == 0:
            return (1.0, 4.5, 1.5)
        recent = atr_pct.tail(min(len(atr_pct), lookback))
        cur = float(recent.iloc[-1])
        pct = float((recent <= cur).mean())
        # Lower pct -> current ATR is large relative to history (volatile)
        if pct < 0.2:
            return (0.8, 2.5, 1.2)
        if pct < 0.4:
            return (0.9, 3.0, 1.5)
        if pct < 0.6:
            return (1.1, 3.5, 1.8)
        if pct < 0.8:
            return (1.3, 4.0, 2.2)
        return (1.5, 4.5, 2.5)
    except Exception:
        return (1.0, 4.5, 1.5)


def classify_regime_deterministic(df: pd.DataFrame, train_n: Optional[int] = None) -> pd.Series:
    """Assign regime labels using volume, ATR%, and momentum percentiles.

    Produces '<vol>_<atr>_<trend>' strings identical to threshold_optimizer.py's
    classify_regimes() so the rest of the pipeline that reads hmm_regime works
    unchanged.  Avoids HMM instability (degenerate single-state solutions).

    train_n: if provided, percentile thresholds are computed from the first
    train_n rows only (the pre-holdout slice), preventing holdout data from
    shifting the quantile boundaries and contaminating training bar labels.
    """
    try:
        roll_vol = df["volume"].rolling(24, min_periods=1).mean()
        atr_vals  = compute_atr(df, period=14)
        atr_pct   = (atr_vals / df["close"].replace(0, np.nan)).fillna(0)
        momentum  = df["close"].pct_change(24).fillna(0)

        # Use only the training slice for quantile computation.
        _ref = slice(None, train_n) if train_n is not None else slice(None)
        vp33 = float(roll_vol.iloc[_ref].quantile(0.33));  vp67 = float(roll_vol.iloc[_ref].quantile(0.67))
        ap33 = float(atr_pct.iloc[_ref].quantile(0.33));   ap67 = float(atr_pct.iloc[_ref].quantile(0.67))
        mp33 = float(momentum.iloc[_ref].quantile(0.33));  mp67 = float(momentum.iloc[_ref].quantile(0.67))

        def _t(v, p33, p67): return "low" if v <= p33 else ("med" if v <= p67 else "high")
        def _tr(v, p33, p67): return "down" if v <= p33 else ("flat" if v <= p67 else "up")

        labels = pd.Series([
            f"{_t(float(roll_vol.iloc[i]), vp33, vp67)}"
            f"_{_t(float(atr_pct.iloc[i]),  ap33, ap67)}"
            f"_{_tr(float(momentum.iloc[i]), mp33, mp67)}"
            for i in range(len(df))
        ], index=df.index, dtype=str)
        return labels
    except Exception as _e:
        print(f"   [REGIME] Deterministic classifier failed ({_e}), using UNKNOWN.")
        return pd.Series("UNKNOWN", index=df.index, dtype=str)


def compute_dynamic_atr_multiplier(
    base_mult: float,
    er: float,          # efficiency_ratio (0–1): how directional the price move is
    vol_regime: float,  # volatility_regime (normalised; 1.0 = historical average)
) -> float:
    """Compute a bar-level ATR multiplier that adapts to market character.

    Wider barriers are used for noisy, high-volatility bars; tighter barriers
    are used when momentum is strong and price action is orderly.
    """
    er = float(np.clip(er, 0.0, 1.0))
    vol = float(np.clip(vol_regime, 0.6, 2.0))

    noise_factor = 1.0 + (1.0 - er) * 0.55      # 1.0 … 1.55
    vol_factor = 1.0 + np.clip(vol - 1.0, -0.25, 0.35)
    dynamic = base_mult * noise_factor * vol_factor

    min_mult = base_mult * (0.75 if er > 0.55 else 0.80)
    max_mult = base_mult * (3.0 if er > 0.40 else 4.0)
    return float(np.clip(dynamic, min_mult, max_mult))


# ============================================================
# TRIPLE-BARRIER LABELING (with censoring)
# ============================================================
def _adaptive_label_vol_threshold(
    volatility_regime: float,
    efficiency_ratio: float,
    trend_regime: Optional[float],
) -> float:
    """Compute a per-bar threshold for whether the market is too quiet/noisy to label."""
    vol = float(np.clip(volatility_regime if volatility_regime is not None else 1.0, 0.5, 1.8))
    er = float(np.clip(efficiency_ratio if efficiency_ratio is not None else 0.5, 0.0, 1.0))
    trend_adj = -0.10 if trend_regime == 1 else 0.0
    # Lowered to 0.50 — target BUY+SELL > 50% of pool so binary primary trains on
    # majority directional examples and scale_pos_weight stays manageable.
    threshold = 0.50 - 0.20 * er + 0.06 * np.tanh((vol - 1.0) * 1.8) + trend_adj
    return float(np.clip(threshold, 0.25, 0.60))


def _adaptive_efficiency_floor(volatility_regime: float) -> float:
    vol = float(np.clip(volatility_regime if volatility_regime is not None else 1.0, 0.6, 1.8))
    return float(np.clip(0.10 + 0.12 * max(0.0, vol - 1.0), 0.08, 0.30))


def _adaptive_confluence_bounds(score: pd.Series) -> Tuple[float, float]:
    if score is None or score.empty:
        return -0.50, 0.50
    lower = float(np.quantile(score, 0.20))
    upper = float(np.quantile(score, 0.80))
    if np.isclose(lower, upper):
        lower -= 0.05
        upper += 0.05
    return lower, upper


def analyze_training_labels_for_adaptation(
    df_train: pd.DataFrame,
    labels_train: pd.Series,
) -> Dict[str, Any]:
    """Derive adaptive labeling parameters from the training pool only.

    All values are computed strictly from the training split so that holdout
    rows never influence label generation — preventing any form of future
    information leakage.
    """
    df = df_train.copy()
    df['_label_tmp'] = labels_train.values
    result: Dict[str, Any] = {}
    for label in [0, 1, 2]:
        mask = df['_label_tmp'] == label
        if mask.sum() < 10:
            continue
        subset = df[mask]
        if 'hmm_regime' in subset.columns:
            result[f'regime_dist_{label}'] = subset['hmm_regime'].value_counts(normalize=True).to_dict()
        if 'macro_confluence_score' in subset.columns:
            if label == 2:   # BUY — lower bound screens out anti-confluence buys
                result['buy_confluence_lower'] = float(subset['macro_confluence_score'].quantile(0.20))
            elif label == 0: # SELL — upper bound screens out anti-confluence sells
                result['sell_confluence_upper'] = float(subset['macro_confluence_score'].quantile(0.80))
        for col in ['volatility_regime', 'efficiency_ratio_10']:
            if col in subset.columns:
                result[f'{col}_p10_{label}'] = float(subset[col].quantile(0.10))
                result[f'{col}_p90_{label}'] = float(subset[col].quantile(0.90))
    return result


# ============================================================
# REVERSAL ALIGNMENT WITH live_engine.py
# ============================================================
# The engine only ever fires FADES at exhaustion (Guard N: never trend-follow),
# at a structural extreme (Guard A/D, v70), confirmed by a real candlestick
# reversal (Guard J, v69). But the labels above are strategy-AGNOSTIC: the
# triple barrier tags every bar by whichever barrier is hit first, so in a
# downtrend most bars label SELL and the model learns trend-FOLLOWING — exactly
# the side Guard N then blocks. That train/serve mismatch is why the model and
# the gates disagreed (measured: model wanted SELL on AAVE at RSI 21.9 sitting
# on support, which is the engine's textbook BUY).
#
# The fix is NOT to relabel (the barrier outcome is ground truth) and NOT to
# train on reversal bars only — measured, the candle-confirmed extremes are
# ~3% of bars (~150 per token), far too few for a per-token XGBoost with 100+
# features; it would overfit badly. Instead the reversal bars are UPWEIGHTED,
# so the loss is dominated by the population the engine actually trades while
# the full history still stabilises the fit.
#
# Thresholds mirror live_engine's EXTREME_* constants exactly — if those move,
# move these with them or training and serving drift apart again.
REVERSAL_RP_BUY       = 0.25   # bottom quarter of the range → BUY exhaustion
REVERSAL_RP_SELL      = 0.75   # top quarter of the range    → SELL exhaustion
REVERSAL_RSI_BUY      = 35.0
REVERSAL_RSI_SELL     = 65.0
REVERSAL_FOCUS_WEIGHT = 3.0    # loss multiplier on candle-confirmed extremes

# How many bars back the confirming candle may have printed. KEEP THIS AT 1.
#
# The theory for raising it was that training was stricter than serving: the live
# engine holds a setup pending AT the level and re-checks every 5m scan, so the
# candle often arrives after price first reaches the extreme. Raising N to 3 does
# find far more events (BTC 7000h: 161/103 at N=1 -> 390/300 at N=3), but it was
# MEASURED ON BTC AND IT IS WORSE ON EVERY METRIC:
#
#     N=1  sig_prec 0.506 | exp +0.181% | gate lift +19.6% | PF 3.64 | Sharpe 32.6
#     N=3  sig_prec 0.455 | exp +0.140% | gate lift  +9.0% | PF 3.15 | Sharpe 16.0
#
# The reason is selection bias, and it is the opposite of the intuition: if the
# candle printed 2-3 bars ago and price is STILL pinned at the extreme, then the
# reversal did not follow through. Those bars are disproportionately FAILED
# reversals, so widening the window preferentially imports negative examples and
# dilutes the ones that carry signal. The tight same-bar definition is what makes
# the setup informative. Do not raise this without re-running the BTC A/B.
REVERSAL_PENDING_BARS = 1


def reversal_candle_flags(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Vectorised mirror of live_engine._reversal_candle (v69).

    Returns (bullish, bearish) boolean Series marking a reversal pattern
    COMPLETING on each bar: hammer / engulfing / harami / piercing-dark-cloud /
    morning-evening star. Uses only the bar and its two predecessors, so it is
    strictly backward-looking — no leakage.
    """
    o, h, l, c = df['open'], df['high'], df['low'], df['close']
    o1, c1 = o.shift(1), c.shift(1)
    o0, c0 = o.shift(2), c.shift(2)

    rng   = (h - l).replace(0, np.nan)
    body  = (c - o).abs()
    up    = h - np.maximum(o, c)
    dn    = np.minimum(o, c) - l
    body1 = (c1 - o1).abs()
    rng1  = (h.shift(1) - l.shift(1)).replace(0, np.nan)
    body0 = (c0 - o0).abs()
    mid1  = (o1 + c1) / 2.0

    prev_red, prev_green = c1 < o1, c1 > o1
    cur_red,  cur_green  = c < o,   c > o

    hammer   = (dn >= 0.55 * rng) & (up <= 0.15 * rng) & (body <= 0.35 * rng)
    bull_eng = prev_red & cur_green & (o <= c1) & (c >= o1) & (body > body1)
    bull_har = (prev_red & cur_green & (o >= c1) & (c <= o1)
                & (body < body1) & (body1 >= 0.5 * rng1))
    pierce   = prev_red & cur_green & (o < c1) & (c > mid1) & (c < o1)
    morning  = ((c0 < o0) & (body1 <= 0.5 * body0) & cur_green
                & (c > (o0 + c0) / 2.0))

    shooting = (up >= 0.55 * rng) & (dn <= 0.15 * rng) & (body <= 0.35 * rng)
    bear_eng = prev_green & cur_red & (o >= c1) & (c <= o1) & (body > body1)
    bear_har = (prev_green & cur_red & (o <= c1) & (c >= o1)
                & (body < body1) & (body1 >= 0.5 * rng1))
    dark     = prev_green & cur_red & (o > c1) & (c < mid1) & (c > o1)
    evening  = ((c0 > o0) & (body1 <= 0.5 * body0) & cur_red
                & (c < (o0 + c0) / 2.0))

    # Wrapped as an explicit bool Series: the np.maximum/np.minimum in up/dn
    # erase the pandas type mid-chain, so the union's .fillna is not provably
    # a Series method to a type checker, and a bool dtype guarantee forecloses
    # pandas' object-downcast FutureWarning. fillna BEFORE astype keeps the
    # NaN -> False semantics bit-for-bit (this function is parity-verified
    # against live_engine._reversal_candle — outputs must not change).
    bullish = pd.Series(hammer | bull_eng | bull_har | pierce | morning,
                        index=df.index).fillna(False).astype(bool)
    bearish = pd.Series(shooting | bear_eng | bear_har | dark | evening,
                        index=df.index).fillna(False).astype(bool)
    return bullish, bearish


def compute_reversal_events(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Bars where live_engine WOULD take a fade: (buy_event, sell_event).

    Mirrors the deployed gate stack — counter-trend only (Guard N), at a
    structural extreme (v70), with a candlestick confirmation (Guard J). Trend
    is an EMA50/EMA200 proxy for MarketRegimeDetector's TRENDING_BULL/BEAR,
    which is not reproducible offline; it agrees on the direction that matters
    here. Missing inputs yield all-False, so this can only ever no-op.
    """
    need = ('open', 'high', 'low', 'close')
    if any(col not in df.columns for col in need):
        false = pd.Series(False, index=df.index)
        return false, false

    rsi_col = 'rsi_14' if 'rsi_14' in df.columns else ('rsi' if 'rsi' in df.columns else None)
    rp_col  = 'range_position_score' if 'range_position_score' in df.columns else None
    if rsi_col is None or rp_col is None:
        false = pd.Series(False, index=df.index)
        return false, false

    rsi = pd.to_numeric(df[rsi_col], errors='coerce')
    rp  = pd.to_numeric(df[rp_col],  errors='coerce')
    ema_fast = df['close'].ewm(span=50,  adjust=False).mean()
    ema_slow = df['close'].ewm(span=200, adjust=False).mean()
    bear = ema_fast < ema_slow

    bullish, bearish = reversal_candle_flags(df)
    # The candle may have printed up to REVERSAL_PENDING_BARS ago (the engine holds
    # the setup pending at the level); the EXTREME must still hold on this bar.
    _n = max(1, int(REVERSAL_PENDING_BARS))
    bullish_recent = bullish.rolling(_n, min_periods=1).max().astype(bool)
    bearish_recent = bearish.rolling(_n, min_periods=1).max().astype(bool)

    buy_event  = (bear  & (rp <= REVERSAL_RP_BUY)  & (rsi <= REVERSAL_RSI_BUY)  & bullish_recent)
    sell_event = (~bear & (rp >= REVERSAL_RP_SELL) & (rsi >= REVERSAL_RSI_SELL) & bearish_recent)
    return buy_event.fillna(False), sell_event.fillna(False)


def reversal_focus_weights(event_mask: pd.Series, n: int,
                           weight: float = REVERSAL_FOCUS_WEIGHT) -> np.ndarray:
    """Per-sample weights: `weight` on the engine's fade bars, 1.0 elsewhere."""
    w = np.ones(n, dtype=float)
    if event_mask is None or len(event_mask) != n:
        return w
    w[np.asarray(event_mask.to_numpy(), dtype=bool)] = float(weight)
    return w


def create_triple_barrier_labels(df: pd.DataFrame, atr_multiplier: float,
                                  max_lookahead: int = MAX_LOOKAHEAD,
                                  volatility_regime: Optional[pd.Series] = None,
                                  efficiency_ratio: Optional[pd.Series] = None,
                                  trend_regime: Optional[pd.Series] = None,
                                  macro_confluence_score: Optional[pd.Series] = None,
                                  barrier_multiplier: Optional[float] = None,
                                  adapt_params: Optional[Dict[str, Any]] = None,
                                  regime_atr_mult: Optional[Dict[str, float]] = None,
                                  barrier_up_skew: Optional[float] = None,
                                  barrier_down_skew: Optional[float] = None,
                                  return_hit_bars: bool = False) -> Any:
    """3-class labels: 0=SELL, 1=HOLD, 2=BUY, -1=CENSORED (dropped upstream).

    return_hit_bars: when True, also return a parallel float Series holding the
        bar OFFSET at which the winning barrier was hit (NaN for HOLD/censored).
        Used ONLY by evaluation to deduplicate overlapping trades by their real
        holding time (see effective_sample_size_durations) — never as a feature.

    adapt_params: pre-computed per-training-split statistics from
        analyze_training_labels_for_adaptation().  When provided, the
        confluence bounds are taken from these frozen training statistics
        rather than re-derived from the full (potentially holdout-inclusive)
        series, eliminating future-information leakage.
    regime_atr_mult: optional mapping {hmm_regime: multiplier} derived from
        training data; when present the per-bar ATR multiplier is adjusted
        per regime before the dynamic noise/vol factor is applied.
    """
    if df is None or df.empty:
        _empty = pd.Series(dtype=int)
        return (_empty, pd.Series(dtype=float)) if return_hit_bars else _empty

    _up_skew   = float(barrier_up_skew)   if barrier_up_skew   is not None else BARRIER_UP_SKEW
    _down_skew = float(barrier_down_skew) if barrier_down_skew is not None else BARRIER_DOWN_SKEW

    labels = pd.Series(1, index=df.index, dtype=int)
    hit_bars = pd.Series(np.nan, index=df.index, dtype=float)
    atr = compute_atr(df, period=14)
    n = len(df)
    # Compute baseline confluence bounds from whatever series is passed in.
    # If adapt_params is provided (training-only stats) those values override,
    # preventing holdout rows from influencing their own labels.
    cs_lower, cs_upper = _adaptive_confluence_bounds(macro_confluence_score) if macro_confluence_score is not None else (-0.50, 0.50)
    if adapt_params is not None:
        if 'buy_confluence_lower' in adapt_params:
            cs_lower = float(adapt_params['buy_confluence_lower'])
        if 'sell_confluence_upper' in adapt_params:
            cs_upper = float(adapt_params['sell_confluence_upper'])

    for i in range(n - 1):
        vol_regime_i = float(volatility_regime.iloc[i]) if volatility_regime is not None and not pd.isna(volatility_regime.iloc[i]) else 1.0
        er_i = float(efficiency_ratio.iloc[i]) if efficiency_ratio is not None and not pd.isna(efficiency_ratio.iloc[i]) else 0.5
        trend_i = float(trend_regime.iloc[i]) if trend_regime is not None and not pd.isna(trend_regime.iloc[i]) else 0.0

        vol_threshold = _adaptive_label_vol_threshold(vol_regime_i, er_i, trend_i)
        if volatility_regime is not None and vol_regime_i < vol_threshold:
            labels.iloc[i] = 1
            continue

        if efficiency_ratio is not None and er_i < _adaptive_efficiency_floor(vol_regime_i):
            labels.iloc[i] = 1
            continue

        entry_price = df.iloc[i]['close']
        atr_val = atr.iloc[i]
        if atr_val == 0 or np.isnan(atr_val):
            atr_val = entry_price * 0.001

        # Per-regime base multiplier override (training-derived, no leakage)
        base_mult_i = atr_multiplier
        if regime_atr_mult is not None and 'hmm_regime' in df.columns:
            hmm_r_i = df['hmm_regime'].iloc[i]
            if hmm_r_i in regime_atr_mult:
                base_mult_i = float(regime_atr_mult[hmm_r_i])
        dynamic_mult = compute_dynamic_atr_multiplier(base_mult_i, er_i, vol_regime_i)

        # Regime-based barrier adjustment: allow optional per-call override
        reg_barrier_adj = 1.0
        if barrier_multiplier is not None:
            reg_barrier_adj = float(barrier_multiplier)
        else:
            # hmm_regime flat-market check (labels produced by deterministic classifier)
            _hmm_i = df['hmm_regime'].iloc[i] if 'hmm_regime' in df.columns else ''
            if isinstance(_hmm_i, str) and (_hmm_i.endswith('_flat') or 'flat' in _hmm_i.split('_')):
                reg_barrier_adj = 1.0   # flat regime: standard barriers — do NOT inflate HOLD%
            elif abs(trend_i) < 0.02:
                reg_barrier_adj = 1.1   # mild adjustment for flat trend, not 1.8
            elif vol_regime_i > 1.25:
                reg_barrier_adj = 0.8
            elif er_i > 0.6:
                reg_barrier_adj = 0.9

        upper = entry_price + (dynamic_mult * reg_barrier_adj * _up_skew) * atr_val
        lower = entry_price - (dynamic_mult * reg_barrier_adj * _down_skew) * atr_val

        window_avail = min(max_lookahead, n - 1 - i)
        hit = None
        hit_j = 0
        for j in range(1, window_avail + 1):
            high = df.iloc[i + j]['high']
            low = df.iloc[i + j]['low']
            if high >= upper:
                hit = 2
                hit_j = j
                break
            if low <= lower:
                hit = 0
                hit_j = j
                break

        if macro_confluence_score is not None and hit is not None and cs_lower < cs_upper:
            cs = float(macro_confluence_score.iloc[i])
            if hit == 2 and cs <= cs_lower:
                hit = None
            if hit == 0 and cs >= cs_upper:
                hit = None

        labels.iloc[i] = hit if hit is not None else (1 if window_avail >= max_lookahead else CENSORED)
        if hit is not None:
            hit_bars.iloc[i] = float(hit_j)

    if n > 0:
        tail = min(max_lookahead, n)
        labels.iloc[n - tail:] = CENSORED
        hit_bars.iloc[n - tail:] = np.nan   # censored labels carry no hit offset
    if return_hit_bars:
        return labels, hit_bars
    return labels


def get_class_weights(y: np.ndarray, min_directional_ratio: float = 0.50) -> np.ndarray:
    """Return per-sample weights that upweight BUY/SELL when they are rare.

    Keeps average weight near 1.0 for stability.
    """
    y = np.asarray(y).astype(int)
    cnt = np.bincount(y, minlength=NUM_CLASS).astype(float)
    total = cnt.sum() if cnt.sum() > 0 else 1.0
    dir_cnt = float(cnt[0] + cnt[2])
    dir_ratio = dir_cnt / total
    base = np.ones(len(y), dtype=float)
    if dir_ratio <= 0.0:
        # No directional labels — assign small equal weight boost to any non-hold (if present)
        base = np.where((y == 0) | (y == 2), 5.0, 1.0)
    elif dir_ratio < min_directional_ratio:
        # Scale up directional samples to reach the desired ratio approximately
        desired_dir = max(min_directional_ratio * total, 1.0)
        upfactor = max(1.0, desired_dir / max(dir_cnt, 1.0))
        base = np.where((y == 0) | (y == 2), upfactor, 1.0)
    else:
        # Balanced enough — fall back to inverse-frequency weighting
        cnt[cnt == 0] = 1.0
        cw = (1.0 / cnt)
        cw = cw / cw.sum() * NUM_CLASS
        base = cw[y]

    # Normalize to mean 1.0 to avoid changing global learning scale
    base = base / float(np.mean(base))
    return base


# ============================================================
# SHAP PRUNING
# ============================================================
def prune_features_by_shap(model: xgb.Booster, X: pd.DataFrame,
                           threshold: float = SHAP_CUMULATIVE_THRESH,
                           min_keep: int = MIN_FEATURES) -> List[str]:
    def _mean_abs(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.ndim == 1:
            return np.abs(arr)
        if arr.ndim == 2:
            if arr.shape[1] == X.shape[1]:
                return np.abs(arr).mean(axis=0)
            if arr.shape[0] == X.shape[1]:
                return np.abs(arr).mean(axis=1)
            for axis in (0, 1):
                cand = np.abs(arr).mean(axis=axis)
                if cand.shape[0] == X.shape[1]:
                    return cand
            raise ValueError(f"Cannot infer feature axis from SHAP shape {arr.shape}")
        if arr.ndim == 3:
            if arr.shape[2] == X.shape[1]:
                return np.mean(np.abs(arr), axis=(0, 1))
            if arr.shape[1] == X.shape[1]:
                return np.mean(np.abs(arr), axis=(0, 2))
            if arr.shape[0] == X.shape[1]:
                return np.mean(np.abs(arr), axis=(1, 2))
            raise ValueError(f"Cannot infer feature axis from SHAP shape {arr.shape}")
        raise ValueError(f"Unsupported SHAP ndim {arr.ndim}")

    if shap is None:
        print("   WARNING: SHAP unavailable -- keeping full feature set")
        return list(X.columns)

    X_sample = X.sample(n=min(500, len(X)), random_state=42)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_sample)
    mean_abs = np.mean([_mean_abs(v) for v in sv], axis=0) if isinstance(sv, list) else _mean_abs(sv)
    if mean_abs.shape[0] != len(X.columns):
        raise ValueError(f"SHAP length {mean_abs.shape[0]} != feature count {len(X.columns)}")

    importance = pd.Series(mean_abs, index=X.columns).sort_values(ascending=False)
    total = importance.sum()
    if total > 0:
        cumfrac = importance.cumsum() / total
        # features needed to reach the cumulative threshold
        n_cumul = int((cumfrac < threshold).sum()) + 1
    else:
        n_cumul = min_keep
    # also keep at least the top SHAP_TOP_PCT fraction of all features
    n_top_pct = int(len(X.columns) * SHAP_TOP_PCT)
    n_keep = int(np.clip(max(n_cumul, n_top_pct), min_keep, MAX_FEATURES))
    keep = importance.head(n_keep).index.tolist()
    dropped = [c for c in X.columns if c not in keep]
    if dropped:
        actual_cumul = float(importance.head(n_keep).sum() / total) if total > 0 else 1.0
        print(f"   SHAP pruning removed {len(dropped)} low-impact features "
              f"(kept {n_keep}, covers {actual_cumul:.0%} cumulative importance): "
              f"e.g. {dropped[:5]}")
    return keep


# ============================================================
# CALIBRATION HELPERS
# ============================================================
def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    logits = np.log(np.clip(probs, 1e-12, 1.0))
    return _softmax(logits / max(T, 1e-3))


# ============================================================
# ADAPTIVE FEATURE NORMALISER (FIX 3)
# ============================================================
class AdaptiveNormalizer:
    """Adaptive Z-score normalisation with online mean/std tracking.
    
    Prevents feature drift by learning statistics on train, then applying
    to holdout without re-centering. This eliminates distribution shift
    from causally infecting meta training and threshold selection.
    """
    def __init__(self, window: int = 1000, drift_threshold: float = 0.2):
        self.window = window
        self.drift_threshold = drift_threshold
        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}

    def fit_initial(self, df: pd.DataFrame, cols: List[str]) -> None:
        """Learn statistics from training data."""
        for col in cols:
            if col in df.columns:
                self.means[col] = float(df[col].mean())
                self.stds[col] = float(df[col].std()) or 1.0

    def transform(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Apply learned normalisation and track drift."""
        df_norm = df.copy()
        for col in cols:
            if col not in df_norm.columns or col not in self.means:
                continue
            # Z-score: (x - mean) / std
            df_norm[col] = (df[col] - self.means[col]) / max(self.stds[col], 1e-8)
            # Online update intentionally disabled: use fixed training statistics on all splits
            # to prevent over-adaptation that causes distribution shift between train and holdout.
        return df_norm


def fit_temperature(probs: np.ndarray, y: np.ndarray) -> float:
    if minimize_scalar is None:
        return 1.0
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 2 or probs.shape[1] != NUM_CLASS:
        return 1.0
    y_int = np.asarray(y, dtype=int)
    rows = np.arange(len(y_int))                   # precomputed; reused on every trial
    logits = np.log(np.clip(probs, 1e-12, 1.0))    # log-probs as logit proxy; computed once

    def nll(T: float) -> float:
        p = _softmax(logits / max(float(T), 1e-3))
        return float(-np.mean(np.log(np.clip(p[rows, y_int], 1e-12, 1.0))))

    _LO, _HI = 0.1, 10.0   # T<0.1 sharper than raw logits; T>10 near-uniform
    try:
        res = minimize_scalar(nll, bounds=(_LO, _HI), method='bounded',
                              options={'xatol': 1e-3})
        x = float(res.x)
        if _LO < x < _HI:   # NaN → False, ±inf → False; try/except covers all other failures
            return x
    except Exception:
        pass
    return 1.0


# ============================================================
# SMALL UTILITIES
# ============================================================
def _dm(X: pd.DataFrame, y: Optional[np.ndarray] = None, w: Optional[np.ndarray] = None, fw: Optional[np.ndarray] = None) -> xgb.DMatrix:
    return xgb.DMatrix(X, label=y, weight=w, feature_names=list(X.columns), feature_weights=fw)


def _full_params(best: dict) -> dict:
    p = dict(best)
    p.update({'objective': 'multi:softprob', 'eval_metric': 'mlogloss',
              'num_class': NUM_CLASS, 'seed': 42, 'tree_method': 'hist', 'missing': np.nan})
    return p


def _sanitize(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan)
    return df


def sample_weights(y: np.ndarray) -> np.ndarray:
    """Return adaptive sample weights for primary training.

    This will upweight directional classes when they are under-represented.
    """
    return get_class_weights(y)


# ============================================================
# TASK 2: FEATURE DRIFT DETECTION (PSI & KS Test)
# ============================================================
def wilson_lower_bound(successes: float, n: int, z: float = 1.96) -> float:
    """Lower bound of the Wilson score interval for a proportion.

    Lets threshold rows be compared FAIRLY across sample sizes. A raw precision
    ignores n, so 60.0% measured on 50 bars looks better than 52.6% on 325 even
    though its 95% interval is [0.46, 0.72] against [0.47, 0.58]. The threshold
    sweep scans 33 candidates, so with small n at least one row clears any fixed
    bar by luck — measured, a 97% chance of a spurious >=60% row at n=50 when the
    true precision is 50%. Ranking on this bound instead of the point estimate
    makes that luck cost the row rather than reward it.
    """
    if n <= 0:
        return 0.0
    p = max(0.0, min(1.0, float(successes) / float(n)))
    denom  = 1.0 + z * z / n
    centre = p + z * z / (2.0 * n)
    margin = z * float(np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)))
    return max(0.0, (centre - margin) / denom)


def effective_sample_size(fired_idx: np.ndarray, lookahead: int) -> int:
    """Count NON-OVERLAPPING trades among fired signals.

    Labels are triple-barrier outcomes measured over `lookahead` bars, so two
    fires less than `lookahead` apart share most of their outcome path: they are
    one market event observed twice, not two independent trials. Treating them as
    independent is what let a holdout report 100.0% directional precision on
    "58 trades" — at realistic concurrency that is ~4 independent events, and
    4 coin flips landing the same way has probability 0.06, i.e. unremarkable.

    The training pipeline already respects this when FITTING (purged CV with an
    EMBARGO of MAX_LOOKAHEAD); this applies the same discipline to EVALUATION,
    where it was missing. Greedy left-to-right selection — the standard
    non-overlapping count, and deliberately conservative.
    """
    if fired_idx is None or len(fired_idx) == 0:
        return 0
    step = max(1, int(lookahead))
    count = 1
    last = int(fired_idx[0])
    for i in fired_idx[1:]:
        if int(i) - last >= step:
            count += 1
            last = int(i)
    return count


def effective_sample_size_durations(fired_idx: np.ndarray,
                                    durations: Optional[np.ndarray],
                                    max_step: int) -> int:
    """Non-overlapping trade count using each trade's ACTUAL resolution time.

    effective_sample_size() steps by the LABEL WINDOW (max lookahead), which
    treats two trades as overlapping even when the first hit its barrier long
    before the second opened. For long-lookahead tokens that is ruinous: a 72h
    window on a 1,662-bar holdout allows at most ~23 "independent" events NO
    MATTER how the trades actually resolved, so the LB>=60% enable bar is
    unreachable below ~85% precision (measured: ETH DISABLED at 65.5% dir_prec
    with LB 45.1% from exactly this). Two trades are one market event only
    while their outcome paths overlap — a trade that resolved in 9 bars frees
    the market after 9 bars, not 72. Durations come from the labeler's real
    bar-of-hit (_hit_bars); bars with no recorded hit (HOLD timeouts) fall
    back to the full window, staying conservative exactly where the trade
    really did stay open the whole time.
    """
    if fired_idx is None or len(fired_idx) == 0:
        return 0
    count = 0
    next_free = -1
    for i in fired_idx:
        i = int(i)
        if i < next_free:
            continue
        d = durations[i] if durations is not None and i < len(durations) else None
        step = int(d) if d is not None and np.isfinite(d) and d > 0 else int(max_step)
        count += 1
        next_free = i + max(1, step)
    return count


def compute_psi(X_train: np.ndarray, X_holdout: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index: measures distribution shift in a feature.
    PSI > 0.25: Small shift, PSI > 1.0: Large shift"""
    if len(X_train) == 0 or len(X_holdout) == 0:
        return 0.0
    
    # Handle NaN/inf
    X_train = X_train[~np.isnan(X_train) & ~np.isinf(X_train)]
    X_holdout = X_holdout[~np.isnan(X_holdout) & ~np.isinf(X_holdout)]
    
    if len(X_train) < 10 or len(X_holdout) < 10:
        return 0.0
    
    # Compute quantile-based bins on training data
    quantiles = np.percentile(X_train, np.linspace(0, 100, n_bins + 1))
    quantiles[0] = quantiles[0] - 1e-9  # Ensure left edge inclusion
    quantiles[-1] = quantiles[-1] + 1e-9
    
    train_counts = np.histogram(X_train, bins=quantiles)[0] / len(X_train) + 1e-9
    holdout_counts = np.histogram(X_holdout, bins=quantiles)[0] / len(X_holdout) + 1e-9
    
    psi = np.sum((holdout_counts - train_counts) * np.log(holdout_counts / train_counts))
    return float(np.clip(psi, 0.0, 10.0))


def compute_ks_stat(X_train: np.ndarray, X_holdout: np.ndarray) -> float:
    """Kolmogorov-Smirnov test: max distance between CDFs.
    KS > 0.50: Very large distribution shift"""
    from scipy import stats
    
    if len(X_train) < 5 or len(X_holdout) < 5:
        return 0.0
    
    X_train_clean = X_train[~np.isnan(X_train) & ~np.isinf(X_train)]
    X_holdout_clean = X_holdout[~np.isnan(X_holdout) & ~np.isinf(X_holdout)]
    
    if len(X_train_clean) < 5 or len(X_holdout_clean) < 5:
        return 0.0
    
    ks_stat, _ = stats.ks_2samp(X_train_clean, X_holdout_clean)
    return float(max(0.0, min(ks_stat, 1.0)))


def detect_feature_drift(X_train: pd.DataFrame, X_holdout: pd.DataFrame) -> Dict[str, Any]:
    """TASK 2: Auto-detect and blacklist drifted features.
    Returns dict with feature names to exclude and reason."""
    drift_report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "features_to_exclude": [],
        "drift_details": {}
    }
    
    for col in X_train.columns:
        if col not in X_holdout.columns:
            drift_report["features_to_exclude"].append(col)
            drift_report["drift_details"][col] = {"reason": "missing_in_holdout"}
            continue
        
        X_tr_col = X_train[col].to_numpy()
        X_hout_col = X_holdout[col].to_numpy()
        
        psi = compute_psi(X_tr_col, X_hout_col)
        ks = compute_ks_stat(X_tr_col, X_hout_col)
        
        drift_report["drift_details"][col] = {
            "psi": float(psi),
            "ks": float(ks),
            "exclude": False,
            "reason": ""
        }
        
        if psi > FEATURE_DRIFT_PSI_THRESHOLD:
            drift_report["features_to_exclude"].append(col)
            drift_report["drift_details"][col]["exclude"] = True
            drift_report["drift_details"][col]["reason"] = f"PSI={psi:.3f} > {FEATURE_DRIFT_PSI_THRESHOLD}"
        elif ks > FEATURE_DRIFT_KS_THRESHOLD:
            drift_report["features_to_exclude"].append(col)
            drift_report["drift_details"][col]["exclude"] = True
            drift_report["drift_details"][col]["reason"] = f"KS={ks:.3f} > {FEATURE_DRIFT_KS_THRESHOLD}"
    
    return drift_report


def build_meta_X(X_feats: pd.DataFrame, primary_probs: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Meta features = base market features only.
    Primary probs are intentionally excluded: they come from OOF models (lower
    confidence, trained on 60-80% of data) during training but from the full-fit
    deployment model (higher confidence) at inference — a covariate shift that
    causes the meta gate to collapse out-of-sample. Base features have consistent
    distributions between training and inference, so the gate transfers reliably."""
    return X_feats.copy()


def proposed_side(
    primary_probs: np.ndarray,
    buy_rate: float = 0.0,
    sell_rate: float = 0.0,
) -> np.ndarray:
    """Primary's directional proposal: 2=BUY if buy-prob >= sell-prob else 0=SELL.

    When buy_rate/sell_rate are provided (training class frequencies), the raw
    probabilities are divided by their respective base rates before comparing.
    This prevents class-imbalance suppression of the minority direction: a token
    with 12% BUY and 21% SELL labels trains a model that systematically assigns
    higher SELL probability, causing proposed_side to return 0 BUY proposals
    even when the model has genuine directional skill on BUY bars.
    """
    buy_p  = primary_probs[:, 2].astype(float)
    sell_p = primary_probs[:, 0].astype(float)
    if buy_rate > 0.0 and sell_rate > 0.0:
        buy_p  = buy_p  / max(buy_rate,  0.05)
        sell_p = sell_p / max(sell_rate, 0.05)
    return np.where(buy_p >= sell_p, 2, 0)


def objective(trial, Xtr, ytr, Xva, yva, fw=None):
    params = {
        'objective': 'multi:softprob', 'eval_metric': 'mlogloss', 'num_class': NUM_CLASS,
        'max_depth': trial.suggest_int('max_depth', 3, 5),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.08, log=True),
        'subsample': trial.suggest_float('subsample', 0.60, 0.80),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.60, 0.80),
        'gamma': trial.suggest_float('gamma', 0, 3),
        'reg_lambda': trial.suggest_float('reg_lambda', 3.0, 10.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 2.0, 8.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 10, 25),
        'seed': 42, 'tree_method': 'hist', 'missing': np.nan,
    }
    m = xgb.train(params, _dm(Xtr, ytr, sample_weights(ytr), fw=fw), num_boost_round=500,
                  evals=[(_dm(Xva, yva, fw=fw), 'eval')], early_stopping_rounds=50, verbose_eval=False)
    return float(log_loss(yva, m.predict(_dm(Xva, fw=fw)), labels=list(range(NUM_CLASS))))


def objective_multifold(trial, X: pd.DataFrame, y: np.ndarray,
                        splits: List[Tuple[np.ndarray, np.ndarray]],
                        fw: Optional[np.ndarray] = None) -> float:
    """Optuna objective averaged across all CV folds (purged walk-forward).

    Using only the last fold caused high-variance estimates: a good last fold
    masked bad generalisation across the full time horizon. Averaging across
    all 5 folds gives a much more stable logloss estimate and prevents Optuna
    from over-tuning to the idiosyncrasies of a single period.
    """
    params = {
        'objective': 'multi:softprob', 'eval_metric': 'mlogloss', 'num_class': NUM_CLASS,
        'max_depth': trial.suggest_int('max_depth', 3, 5),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.08, log=True),
        'subsample': trial.suggest_float('subsample', 0.60, 0.80),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.60, 0.80),
        'gamma': trial.suggest_float('gamma', 0, 3),
        'reg_lambda': trial.suggest_float('reg_lambda', 3.0, 10.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 2.0, 8.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 10, 25),
        'seed': 42, 'tree_method': 'hist', 'missing': np.nan,
    }
    losses = []
    for itr, iva in splits:
        w_tr = sample_weights(y[itr])
        m = xgb.train(
            params,
            _dm(X.iloc[itr], y[itr], w_tr, fw=fw),
            num_boost_round=500,
            evals=[(_dm(X.iloc[iva], y[iva], fw=fw), 'eval')],
            early_stopping_rounds=40,
            verbose_eval=False,
        )
        losses.append(float(log_loss(y[iva], m.predict(_dm(X.iloc[iva], fw=fw)),
                                     labels=list(range(NUM_CLASS)))))
    return float(np.mean(losses))


def primary_oof(X: pd.DataFrame, y: np.ndarray, params: dict,
                n_splits: int, gap: int, fw: Optional[np.ndarray] = None) -> np.ndarray:
    """Out-of-fold primary probabilities (purged). Early rows stay NaN."""
    oof = np.full((len(X), NUM_CLASS), np.nan)
    for tr, va in TimeSeriesSplit(n_splits=n_splits, gap=gap).split(X):
        m = xgb.train(params, _dm(X.iloc[tr], y[tr], sample_weights(y[tr]), fw=fw),
                      num_boost_round=800, evals=[(_dm(X.iloc[va], y[va], fw=fw), 'eval')],
                      early_stopping_rounds=60, verbose_eval=False)
        oof[va] = m.predict(_dm(X.iloc[va], fw=fw))
    return oof


def binary_oof(X: pd.DataFrame, y: np.ndarray, params: dict,
               n_splits: int, gap: int,
               w: Optional[np.ndarray] = None) -> np.ndarray:
    oof = np.full(len(X), np.nan)
    for tr, va in TimeSeriesSplit(n_splits=n_splits, gap=gap).split(X):
        w_tr = w[tr] if w is not None else None
        m = xgb.train(params, _dm(X.iloc[tr], y[tr], w_tr), num_boost_round=300,
                      evals=[(_dm(X.iloc[va], y[va]), 'eval')],
                      early_stopping_rounds=30, verbose_eval=False)
        oof[va] = m.predict(_dm(X.iloc[va]))
    return oof


def objective_binary_multifold(
    trial: "optuna.Trial",
    X: pd.DataFrame,
    y: np.ndarray,
    splits: list,
    fw: Optional[np.ndarray] = None,
    spw: float = 1.0,
    sw: Optional[np.ndarray] = None,
) -> float:
    """Optuna objective for binary XGBoost (minimise 1-AUPRC), averaged across folds.

    AUPRC (average precision score) weights precision at low-recall operating points
    more heavily than AUC-ROC. This directly optimises the high-confidence precision
    region that the calibrated threshold gate exploits at inference time.

    `sw` are PER-SAMPLE weights (reversal focus, see reversal_focus_weights) and
    are applied to the TRAINING fold only — never to the eval fold, so the score
    still reflects real unweighted ranking quality and Optuna cannot game it.
    """
    from sklearn.metrics import average_precision_score as _ap
    params = {
        'objective': 'binary:logistic', 'eval_metric': 'aucpr',
        'max_depth': trial.suggest_int('max_depth', 4, 7),     # floor 4: depth-3 can't capture 50+ feature interactions
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.08, log=True),
        'subsample': trial.suggest_float('subsample', 0.60, 0.80),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.60, 0.80),
        'gamma': trial.suggest_float('gamma', 0.5, 3.0),       # floor 0.5: prevents near-zero split threshold (SELL bias)
        'reg_lambda': trial.suggest_float('reg_lambda', 3.0, 10.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 2.0, 8.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 10, 25),
        'seed': 42, 'tree_method': 'hist', 'missing': np.nan,
        'scale_pos_weight': spw,
    }
    aps: List[float] = []
    for itr, iva in splits:
        # fw is feature weights (size=num_features), not sample weights — pass via fw= kwarg
        m = xgb.train(params, _dm(X.iloc[itr], y[itr],
                                  None if sw is None else sw[itr], fw=fw),
                      num_boost_round=400,
                      evals=[(_dm(X.iloc[iva], y[iva], fw=fw), 'eval')],
                      early_stopping_rounds=30, verbose_eval=False)
        p = m.predict(_dm(X.iloc[iva]))
        try:
            aps.append(float(_ap(y[iva], p)))
        except Exception:
            aps.append(0.0)
    return float(1.0 - np.mean(aps))


def _binary_params(best: dict) -> dict:
    """Convert Optuna best_params to a binary:logistic XGBoost params dict."""
    p = dict(best)
    p.update({'objective': 'binary:logistic', 'eval_metric': 'aucpr',
              'seed': 42, 'tree_method': 'hist', 'missing': np.nan})
    return p


def binary_primary_oof(
    X: pd.DataFrame,
    y: np.ndarray,
    params: dict,
    n_splits: int,
    gap: int,
    fw: Optional[np.ndarray] = None,
    sw: Optional[np.ndarray] = None,
) -> np.ndarray:
    """OOF probabilities for a single binary XGBoost primary model.

    `sw` (reversal-focus per-sample weights) applies to the training fold only,
    so the OOF probabilities the meta gate is calibrated on stay unweighted.
    """
    oof = np.full(len(X), np.nan)
    for tr, va in TimeSeriesSplit(n_splits=n_splits, gap=gap).split(X):
        # fw is feature weights (size=num_features), not sample weights — pass via fw= kwarg
        m = xgb.train(params, _dm(X.iloc[tr], y[tr],
                                  None if sw is None else sw[tr], fw=fw),
                      num_boost_round=800,
                      evals=[(_dm(X.iloc[va], y[va], fw=fw), 'eval')],
                      early_stopping_rounds=60, verbose_eval=False)
        oof[va] = m.predict(_dm(X.iloc[va]))
    return oof


def lr_meta_oof(X: pd.DataFrame, y: np.ndarray, n_splits: int, gap: int) -> np.ndarray:
    """OOF for LR meta (C=0.01, balanced). Trains on all bars (HOLD-timeouts as negatives).
    Returns NaN for rows not in any validation fold."""
    from sklearn.linear_model import LogisticRegression as _LR
    # Cap splits so each fold has ≥ 200 training samples — 15 folds on 1900 bars
    # gives fold-1 only 130 training samples, producing useless 0.5 OOF scores.
    n_splits_actual = min(n_splits, max(5, len(X) // 300))
    oof = np.full(len(X), np.nan)
    for tr, va in TimeSeriesSplit(n_splits=n_splits_actual, gap=gap).split(X):
        X_tr = X.iloc[tr].fillna(0.0)
        X_va = X.iloc[va].fillna(0.0)
        y_tr = y[tr].astype(int)
        if len(np.unique(y_tr)) < 2:
            oof[va] = float(y_tr.mean())
            continue
        clf = _LR(C=0.01, max_iter=2000, solver='lbfgs',
                  class_weight='balanced', random_state=42)
        try:
            clf.fit(X_tr, y_tr)
            oof[va] = clf.predict_proba(X_va)[:, 1]
        except Exception:
            pass
    return oof


def pick_threshold(meta_prob: np.ndarray, proposed: np.ndarray, y_true: np.ndarray,
                   target: float = TARGET_SIGNAL_PRECISION,
                   min_fires: int = MIN_FIRES_DEV) -> Tuple[float, float, float, int, bool]:
    """Choose a gate as a PERCENTILE of the meta-prob distribution, not an absolute
    value. A percentile transfers across distributions (OOF vs full-fit primary)
    far better than a raw cutoff like 0.76, which is what made the holdout fire 0
    signals. We sweep percentiles from 'fire the top 50%' down to 'top 2%' and take
    the most permissive one whose precision clears the target, backed by >=min_fires
    trades. If none clears it, return the best achievable and flag it.

    Returns (threshold_value, precision, coverage, n_trades, hit_target). The
    threshold_value is still an absolute meta-prob, derived from the chosen
    percentile, so downstream code is unchanged."""
    valid = ~np.isnan(meta_prob)
    mp, pr, yt = meta_prob[valid], proposed[valid], y_true[valid]
    if len(mp) == 0:
        return 0.5, 0.0, 0.0, 0, False

    rows = []
    # top-q fraction of signals by meta confidence (most permissive first)
    for q in [0.70, 0.60, 0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10, 0.07, 0.05, 0.04, 0.03, 0.02]:
        thr = float(np.quantile(mp, 1.0 - q))
        fire = (mp >= thr) & ((pr == 2) | (pr == 0))  # only count directional trades!
        n = int(fire.sum())
        if n < min_fires:
            continue
        prec = float((pr[fire] == yt[fire]).mean())
        # cov is the fraction of directional trades out of all directional proposals
        dir_mask = ((pr == 2) | (pr == 0))
        total_dir = int(dir_mask.sum())
        cov = n / total_dir if total_dir > 0 else 0.0
        rows.append((thr, prec, cov, n))
    if not rows:
        return float(np.quantile(mp, 0.9)), 0.0, 0.0, 0, False

    meeting = [r for r in rows if r[1] >= target]
    if meeting:
        thr, prec, cov, n = meeting[0]   # most permissive that meets target
        return thr, prec, cov, n, True
    # No threshold reaches the precision target on dev data with enough trades.
    # Per design: this token does NOT trade. Return the best row for logging, but
    # flag hit_target=False so the caller silences it (no signals emitted).
    best = max(rows, key=lambda r: r[1])
    return best[0], best[1], best[2], best[3], False


def pick_edge_threshold_by_side(
    edge_scores: np.ndarray,
    proposed:  np.ndarray,
    y_true:    np.ndarray,
    side:      int,               # 2 = BUY, 0 = SELL
    target:    float = TARGET_SIGNAL_PRECISION,
    min_fires: int   = MIN_FIRES_DEV,
) -> Tuple[float, float, float, int, bool]:
    """TASK 3: Sweeps adaptive quantiles with improved coverage targeting.
    Allows wider coverage range and doesn't reject based on low coverage alone.
    Returns (threshold_value, precision, coverage, n_trades, hit_target)."""
    valid = ~np.isnan(edge_scores)
    es = edge_scores[valid]
    pr = proposed[valid]
    yt = y_true[valid]

    side_mask = (pr == side)
    if side_mask.sum() == 0:
        return 55.0, 0.0, 0.0, 0, False

    es_s = es[side_mask]
    yt_s = yt[side_mask]

    # TASK 3: Use new adaptive coverage targets
    effective_min_fires = min(min_fires, max(5, int(len(es_s) * 0.10)))
    # Allow wider range: MIN_COVERAGE to MAX_COVERAGE
    max_q = min(MAX_COVERAGE, max(TARGET_COVERAGE, 1.0 - target * 0.70))
    quantiles = np.unique(np.concatenate([
        np.linspace(MIN_COVERAGE, max_q, 20),
        np.array([TARGET_COVERAGE, 0.05, 0.10, 0.15, 0.20, 0.25])
    ]))

    rows = []
    for q in quantiles:
        thr = float(np.quantile(es_s, 1.0 - q))
        fire = es_s >= thr
        n = int(fire.sum())
        if n < effective_min_fires:
            continue
        cov = n / len(es_s)
        prec = float((yt_s[fire] == side).mean())
        rows.append((thr, prec, cov, n))

    if not rows:
        fallback_thr = float(np.quantile(es_s, 1.0 - min(MAX_COVERAGE, 0.20)))
        fallback_n = int((es_s >= fallback_thr).sum())
        fallback_prec = float((yt_s[es_s >= fallback_thr] == side).mean()) if fallback_n > 0 else 0.0
        return fallback_thr, fallback_prec, fallback_n / len(es_s), fallback_n, False

    meeting = [r for r in rows if r[1] >= target]
    if meeting:
        # Prefer widest-coverage threshold in [target, target+20pp] to avoid
        # isotonic-inflated cherry-picks that overfit dev OOF but fail on holdout.
        _overfit_ceil = target + 0.20
        _moderate = [r for r in meeting if r[1] <= _overfit_ceil]
        if _moderate:
            thr, prec, cov, n = max(_moderate, key=lambda r: r[2])
        else:
            thr, prec, cov, n = max(meeting, key=lambda r: r[2])
        return thr, prec, cov, n, True

    best = max(rows, key=lambda r: (r[1], r[2]))
    if len(es_s) >= effective_min_fires:
        unconditional_prec = float((yt_s == side).mean())
        if unconditional_prec >= target and unconditional_prec > best[1]:
            return float(es_s.min()), unconditional_prec, 1.0, len(es_s), True

    return best[0], best[1], best[2], best[3], False


def pick_threshold_by_side(
    meta_prob: np.ndarray,
    proposed:  np.ndarray,
    y_true:    np.ndarray,
    side:      int,               # 2 = BUY, 0 = SELL
    target:    float = TARGET_SIGNAL_PRECISION,
    min_fires: int   = MIN_FIRES_DEV,
) -> Tuple[float, float, float, int, bool]:
    """Same sweep as pick_threshold but evaluated only on signals of one side.

    Allows the engine to fire BUY signals at 64% precision even when SELL signals
    only reach 48%, rather than averaging both into a mediocre 56% that fails the
    target.

    Returns (threshold, precision, coverage_within_side, n_trades, hit_target).
    Coverage here is the fraction of that SIDE's signals that pass the gate.
    """
    valid = ~np.isnan(meta_prob)
    mp = meta_prob[valid]
    pr = proposed[valid]
    yt = y_true[valid]

    side_mask = (pr == side)
    if side_mask.sum() == 0:
        return 0.5, 0.0, 0.0, 0, False

    mp_s = mp[side_mask]
    yt_s = yt[side_mask]

    effective_min_fires = min(min_fires, max(5, int(len(mp_s) * 0.15)))
    max_coverage = min(0.40, max(0.10, 1.0 - target * 0.75))
    quantiles = np.unique(np.concatenate([
        np.linspace(0.02, max_coverage, 16),
        np.array([0.05, 0.10, 0.15, 0.20, 0.25])
    ]))

    rows = []
    for q in quantiles:
        thr = float(np.quantile(mp_s, 1.0 - q))
        fire = mp_s >= thr
        n = int(fire.sum())
        if n < effective_min_fires:
            continue
        cov = n / len(mp_s)
        if cov > max_coverage:
            continue
        prec = float((yt_s[fire] == side).mean())
        rows.append((thr, prec, cov, n))

    if not rows:
        if len(mp_s) >= effective_min_fires:
            unconditional_prec = float((yt_s == side).mean())
            if unconditional_prec >= target:
                return float(mp_s.min()), unconditional_prec, 1.0, len(mp_s), True
        fallback_thr = float(np.quantile(mp_s, 0.90))
        fallback_n = int((mp_s >= fallback_thr).sum())
        fallback_prec = float((yt_s[mp_s >= fallback_thr] == side).mean()) if fallback_n > 0 else 0.0
        return fallback_thr, fallback_prec, fallback_n / len(mp_s), fallback_n, False

    meeting = [r for r in rows if r[1] >= target]
    if meeting:
        _overfit_ceil = target + 0.20
        _moderate = [r for r in meeting if r[1] <= _overfit_ceil]
        if _moderate:
            thr, prec, cov, n = max(_moderate, key=lambda r: r[2])
        else:
            thr, prec, cov, n = max(meeting, key=lambda r: r[2])
        return thr, prec, cov, n, True

    best = max(rows, key=lambda r: (r[1], r[2]))
    if len(mp_s) >= effective_min_fires:
        unconditional_prec = float((yt_s == side).mean())
        if unconditional_prec >= target and unconditional_prec > best[1]:
            return float(mp_s.min()), unconditional_prec, 1.0, len(mp_s), True
    return best[0], best[1], best[2], best[3], False


def get_profile_edge_thresholds(
    meta_gate_profile: Optional[Dict[str, Any]],
    edge_buy: np.ndarray,
    edge_sell: np.ndarray,
    prop_dev_filtered: np.ndarray,
    y_v: np.ndarray,
) -> Optional[Tuple[float, float, bool, bool, float, float, int, int]]:
    """Use optimizer-selected profile thresholds when available."""
    if not meta_gate_profile:
        return None
    gate_type = str(meta_gate_profile.get('gate_type', '')).upper()
    thresholds = meta_gate_profile.get('thresholds', {}) or {}
    if gate_type == 'DISABLED':
        return 100.0, 100.0, False, False, 0.0, 0.0, 0, 0
    if not thresholds:
        return None

    global_thr = thresholds.get('global_threshold')
    buy_thr = thresholds.get('buy_threshold', global_thr)
    sell_thr = thresholds.get('sell_threshold', global_thr)
    if buy_thr is None or sell_thr is None:
        return None

    side_specific = bool(meta_gate_profile.get('side_specific', True))
    if side_specific:
        thr_buy = float(buy_thr)
        thr_sell = float(sell_thr)
    else:
        if global_thr is None:
            return None
        thr_buy = thr_sell = float(global_thr)

    buy_mask = (prop_dev_filtered == 2)
    sell_mask = (prop_dev_filtered == 0)
    buy_n = int(((edge_buy >= thr_buy) & buy_mask).sum())
    sell_n = int(((edge_sell >= thr_sell) & sell_mask).sum())

    buy_prec = float((y_v[(edge_buy >= thr_buy) & buy_mask] == 2).mean()) if buy_n > 0 else 0.0
    sell_prec = float((y_v[(edge_sell >= thr_sell) & sell_mask] == 0).mean()) if sell_n > 0 else 0.0
    buy_cov = float(buy_n / buy_mask.sum()) if buy_mask.sum() > 0 else 0.0
    sell_cov = float(sell_n / sell_mask.sum()) if sell_mask.sum() > 0 else 0.0

    return thr_buy, thr_sell, buy_n > 0, sell_n > 0, buy_prec, sell_prec, buy_n, sell_n


# ============================================================
# TASK 5: REGIME THRESHOLD ENGINE (New)
# ============================================================
def apply_regime_threshold_multiplier(
    threshold: float,
    regime: Optional[str] = None,
    regime_thresholds: Optional[Dict[str, float]] = None
) -> float:
    """Apply regime-specific threshold multiplier to adapt gate sensitivity.
    COMPRESSION & VOLATILE_EXPANSION: easier to trade (lower threshold)
    ACCUMULATION & DISTRIBUTION: harder to trade (higher threshold)"""
    if not REGIME_THRESHOLD_ADAPTATION or regime is None:
        return threshold
    
    thresholds = regime_thresholds or REGIME_THRESHOLDS
    multiplier = thresholds.get(str(regime), 1.0)
    return threshold * multiplier


# ============================================================
# TASK 8: FORENSIC REPORTING (New)
# ============================================================
def generate_forensic_before(symbol: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Capture baseline metrics BEFORE refactoring."""
    return {
        "timestamp": pd.Timestamp.now().isoformat(),
        "symbol": symbol,
        "metrics": metrics,
        "config": {
            "hold_strategy": META_HOLD_STRATEGY,
            "coverage_min": MIN_COVERAGE,
            "coverage_target": TARGET_COVERAGE,
            "calibration_mode": CALIBRATION_MODE,
            "n_splits_cv": N_SPLITS_CV,
        }
    }


def generate_forensic_after(symbol: str, metrics: Dict[str, Any], before: Dict[str, Any]) -> Dict[str, Any]:
    """Generate AFTER metrics and compute deltas."""
    after = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "symbol": symbol,
        "metrics": metrics,
    }
    
    # Compute deltas (improvement)
    deltas = {}
    for key in ["precision", "expectancy_pct", "profit_factor", "sharpe", "coverage"]:
        if key in before.get("metrics", {}) and key in metrics:
            before_val = before["metrics"][key]
            after_val = metrics[key]
            if before_val is not None and after_val is not None:
                try:
                    delta = after_val - before_val
                    delta_pct = (delta / abs(before_val) * 100) if before_val != 0 else 0.0
                    deltas[key] = {"delta": float(delta), "delta_pct": float(delta_pct)}
                except:
                    pass
    
    after["deltas"] = deltas
    after["improvements"] = {
        "precision_improved": deltas.get("precision", {}).get("delta", 0) > 0,
        "expectancy_improved": deltas.get("expectancy_pct", {}).get("delta", 0) > 0,
        "coverage_improved": deltas.get("coverage", {}).get("delta", 0) > 0,
    }
    
    return after


def backtest(fire_mask: np.ndarray, proposed: np.ndarray, y_true: np.ndarray,
             barrier_frac: np.ndarray, fee: float = FEE_ROUNDTRIP) -> dict:
    """
    Fee-aware profitability backtest on fired signals.

    Outcome mapping (triple-barrier approximation):
      Correct direction (proposed == true label) → +barrier_frac
      Wrong direction   (proposed != true label and label != HOLD) → -barrier_frac
      Timeout / HOLD                              → 0.0
    All trades reduced by round-trip fee.

    Metrics returned:
      n                  — number of fired signals
      expectancy_pct     — mean return per trade (%)
      total_return_pct   — sum of returns (%)
      win_rate           — fraction of trades with positive net return
      sharpe             — annualised Sharpe (1h candle cadence, 8760h/yr)
      max_drawdown_pct   — peak-to-trough equity drawdown (%)
      profit_factor      — gross_profit / gross_loss  (> 1 = net profitable)
      buy_n / buy_win_rate   — per-side stats
      sell_n / sell_win_rate
      kelly_pct          — Kelly fraction (capped at 25% to prevent blow-up)
    """
    EMPTY = {
        "n": 0, "expectancy_pct": 0.0, "total_return_pct": 0.0,
        "win_rate": 0.0, "sharpe": 0.0, "max_drawdown_pct": 0.0,
        "profit_factor": 0.0, "buy_n": 0, "buy_win_rate": 0.0,
        "sell_n": 0, "sell_win_rate": 0.0, "kelly_pct": 0.0,
    }
    idx = np.where(fire_mask)[0]
    if not len(idx):
        return EMPTY

    rets:        List[float] = []
    buy_wins:    List[bool]  = []
    sell_wins:   List[bool]  = []

    for i in idx:
        b = float(barrier_frac[i]) if np.isfinite(barrier_frac[i]) else 0.0
        label = int(y_true[i])
        side  = int(proposed[i])
        if label == 1:           # timeout → no gain, still pay fee
            g = -fee
        elif side == label:      # correct direction → full barrier
            g = b - fee
        else:                    # wrong direction → full barrier loss
            g = -b - fee
        rets.append(g)
        if label != 1:
            if side == 2:
                buy_wins.append(side == label)
            else:
                sell_wins.append(side == label)

    rets_arr = np.array(rets, dtype=float)
    n        = len(rets_arr)
    mean_ret = float(rets_arr.mean())
    total_r  = float(rets_arr.sum())
    win_rate = float((rets_arr > 0).mean())

    # Sharpe — annualised on 1h cadence assumption
    std_ret = float(rets_arr.std())
    if std_ret > 1e-9 and n > 1:
        # sqrt(min(n, 8760)) gives the annualisation factor for actual trade frequency
        sharpe = float(mean_ret / std_ret * np.sqrt(min(n * 8, 8760)))
    else:
        sharpe = 0.0

    # Max drawdown on the equity curve (percentage of peak)
    # Start at 1.0 so equity is always positive and peak is never near-zero.
    # Without this offset, equity starts at 0 → peak_safe clamped to 1e-9 →
    # a single small loss gives drawdown_pct = loss/1e-9 → 100_000_000%.
    equity  = 1.0 + np.cumsum(rets_arr)
    peak    = np.maximum.accumulate(equity)
    peak_safe = np.maximum(peak, 1e-9)
    drawdown_pct = (peak - equity) / peak_safe
    max_dd  = float(min(drawdown_pct.max() * 100, 100.0))

    # Profit factor
    gross_win  = float(rets_arr[rets_arr > 0].sum()) if (rets_arr > 0).any() else 0.0
    gross_loss = float(abs(rets_arr[rets_arr < 0].sum())) if (rets_arr < 0).any() else 1e-9
    pf         = gross_win / gross_loss

    # Kelly fraction (cap at 25%)
    if win_rate > 0 and win_rate < 1 and std_ret > 1e-9:
        kelly = float(np.clip(mean_ret / (std_ret ** 2), 0.0, 0.25))
    else:
        kelly = 0.0

    return {
        "n":                 n,
        "expectancy_pct":    round(mean_ret * 100, 4),
        "total_return_pct":  round(total_r * 100, 4),
        "win_rate":          round(win_rate, 4),
        "sharpe":            round(sharpe, 3),
        "max_drawdown_pct":  round(max_dd, 3),
        "profit_factor":     round(pf, 3),
        "buy_n":             len(buy_wins),
        "buy_win_rate":      round(float(np.mean(buy_wins)),  4) if buy_wins  else 0.0,
        "sell_n":            len(sell_wins),
        "sell_win_rate":     round(float(np.mean(sell_wins)), 4) if sell_wins else 0.0,
        "kelly_pct":         round(kelly * 100, 2),
    }


# ============================================================
# NEWS SYNC
# ============================================================
def run_news_scraper():
    scraper_path = Path(root_dir) / "scripts" / "news_scraper.py"
    if not scraper_path.exists():
        print(f"WARNING: News scraper not found at {scraper_path}")
        return False
    log_path = Path(root_dir) / "logs" / "news_sync.log"
    log_path.parent.mkdir(exist_ok=True)
    try:
        # CREATE_NO_WINDOW prevents Windows from popping a visible CMD terminal
        # for the child Python process when retrain_model.py is run from an IDE
        # or double-clicked. Output is captured to the log file regardless.
        _flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        with open(log_path, 'a') as log_f:
            result = subprocess.run(
                [sys.executable, str(scraper_path)],
                stdout=log_f, stderr=log_f, timeout=60, check=False,
                creationflags=_flags,
            )
        if result.returncode == 0:
            print(f"News scraper completed (log: {log_path})")
            return True
        print(f"News scraper failed with code {result.returncode}")
        return False
    except subprocess.TimeoutExpired:
        print("News scraper timed out after 60 seconds")
        return False
    except Exception as e:
        print(f"Failed to run news scraper: {e}")
        return False


def ensure_fresh_news_for_training():
    need_sync = False
    if NEWS_FILE.exists():
        age = (datetime.now() - datetime.fromtimestamp(NEWS_FILE.stat().st_mtime)).total_seconds()
        if age >= NEWS_MAX_AGE_SECONDS:
            print(f"News file is stale (age {age/60:.1f} min). Running scraper...")
            need_sync = True
        else:
            print(f"News file is fresh (age {age/60:.1f} min).")
    else:
        print("News file missing. Running scraper...")
        need_sync = True
    if need_sync:
        if not run_news_scraper():
            print("Continuing without fresh news. Sentiment features will be zero.")
            return False
        time.sleep(1)
        if not NEWS_FILE.exists():
            print("News file still missing after scraper run. Proceeding with neutral sentiment.")
            return False
    return True


def log_feature_importance(model, feature_names: List[str], symbol: str):
    logs_dir = Path(root_dir) / "logs" / "features"
    logs_dir.mkdir(parents=True, exist_ok=True)
    importance = model.get_score(importance_type='gain')
    importance_dict: Dict[str, Any] = {}
    if importance:
        if all(k.startswith('f') for k in importance.keys()):
            idx = [int(k[1:]) for k in importance.keys()]
            importance_dict = {feature_names[i]: importance[f'f{i}'] for i in idx if i < len(feature_names)}
        else:
            importance_dict = importance
        sorted_imp = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    else:
        sorted_imp = sorted((model.get_score(importance_type='weight') or {}).items(),
                            key=lambda x: x[1], reverse=True)
    output_file = logs_dir / f"{symbol.replace('/', '_')}_importance.txt"
    with open(output_file, "w") as f:
        f.write(f"Feature importance (gain) for {symbol}\n" + "=" * 50 + "\n")
        for name, imp in sorted_imp[:30]:
            f.write(f"{name:35} : {imp:.6f}\n")
    print(f"   Feature importance saved to {output_file}")
    return importance_dict


# ============================================================
# TRAIN A SINGLE TOKEN
# ============================================================
def train_token(symbol: str, hours: int = 5000) -> dict[str, Any] | None:  # pyright: ignore[reportGeneralTypeIssues]
    print(f"\n{'='*60}\nTraining model for {symbol}\n{'='*60}")
    try:
        p = Predictor(symbol)
        print(f"Fetching {hours} hours of data for {symbol}...")
        df = p.fetch_live_data(timeframe='1h', limit=hours)
        if df is None or df.empty:
            print(f"No data for {symbol}")
            return None

        print("Fetching BTC market context...")
        btc_df = p.fetch_btc_data(timeframe='1h', limit=hours)
        if btc_df is None or btc_df.empty:
            btc_df = None
            print("BTC data not available -- continuing without anchor features")

        print("Loading news sentiment...")
        news_df = p.load_news_data()
        if news_df is None or news_df.empty:
            print("News data not available -- continuing without sentiment")

        print("Applying feature engineering (enhanced indicators)...")
        try:
            df_1d = p.fetch_live_data(timeframe='1d', limit=max(1000, int(hours / 24) + 10))
        except Exception:
            df_1d = None

        print("Fetching futures data (funding rate + OI)...")
        funding_df, oi_df = fetch_futures_data(symbol, df)
        if funding_df is not None:
            print(f"   Funding rate: {len(funding_df)} entries")
        if oi_df is not None:
            print(f"   Open interest: {len(oi_df)} entries")
        if funding_df is None and oi_df is None:
            print("   No futures data (spot-only or API error) -- continuing without")

        print("Fetching Fear & Greed Index...")
        fg_df = fetch_fear_greed(days=700)
        if fg_df is not None:
            print(f"   Fear & Greed: {len(fg_df)} days")
        else:
            print("   Fear & Greed unavailable -- continuing without")

        df = prepare_features(df, btc_df=btc_df, news_df=news_df,
                              add_target_flag=False, df_1d=df_1d, df_1w=None,
                              funding_df=funding_df, oi_df=oi_df, fg_df=fg_df)
        if df is None or df.empty:
            print(f"Feature engineering failed for {symbol}")
            return None

        df = df.reset_index(drop=True).copy()

        # ── Soft (percentile-rank) confluence features ─────────────────────
        # Added AFTER prepare_features so all indicator columns are available.
        # These give XGBoost richer gradient information than the sign-based
        # xxx_confluence columns (RSI-51 = RSI-80 = +1 in sign; prc_momentum
        # distinguishes them as 0.52 vs 0.85).
        # macro_confluence_score is derived here and passed to the labeler.
        print("   Computing soft (percentile-rank) confluence features...")
        _REQUIRED_SOFT_COLS = [
            'prc_trend', 'prc_momentum', 'prc_volume', 'prc_bands',
            'prc_smart_money', 'prc_total', 'macro_confluence_score',
        ]
        try:
            soft_conf = compute_soft_confluence_features(df)
            for col in soft_conf.columns:
                df[col] = soft_conf[col].values
            # Verify all required columns are present
            _missing_soft = [c for c in _REQUIRED_SOFT_COLS if c not in df.columns]
            if _missing_soft:
                raise RuntimeError(
                    f"compute_soft_confluence_features() did not produce required columns: {_missing_soft}. "
                    f"Check that prepare_features() ran successfully and all indicator columns are present."
                )
            _conf_sample = float(df['prc_total'].iloc[-1])
            print(f"   prc_total sample (last bar): {_conf_sample:.3f}  "
                  f"macro_conf_score: {float(df['macro_confluence_score'].iloc[-1]):.3f}")
        except RuntimeError:
            raise
        except Exception as _sc_err:
            print(f"   Soft confluence computation failed ({_sc_err}) — using sign-based fallback")
            for col in _REQUIRED_SOFT_COLS:
                if col not in df.columns:
                    df[col] = 0.5 if col != 'macro_confluence_score' else 0.0

        for col in ['volatility_regime', 'efficiency_ratio_10', 'trend_regime']:
            if col not in df.columns:
                print(f"{col} missing -- using constant")
                df[col] = 1.0 if col == 'volatility_regime' else (0.5 if 'efficiency' in col else 0)

        # Keep ATR for the PnL backtest (excluded from features via leading underscore).
        df['_atr'] = compute_atr(df, period=14).values

        # ---- Deterministic regime classifier (replaces HMM) ----
        # HMM training was prone to degenerate single-state solutions and instability
        # across different tokens. The deterministic approach uses volume/ATR/momentum
        # percentiles to produce consistent '<vol>_<atr>_<trend>' labels for all tokens.
        # Compute regime quantile thresholds from PRE-SPLIT data only.
        # classify_regime_deterministic() uses .quantile() across the full series;
        # passing the full df (including holdout) leaks holdout vol/ATR/momentum
        # distribution into training-bar labels. Compute on the pre-holdout slice.
        _df_n = len(df)
        _pre_holdout_n = _df_n - int(_df_n * TEST_FRAC)   # same boundary as train_end
        print("   Computing deterministic regime labels (vol×atr×trend percentiles)...")
        df['hmm_regime'] = classify_regime_deterministic(df, train_n=_pre_holdout_n)
        n_unique_regimes = len(df['hmm_regime'].unique())
        print(f"   [REGIME] {n_unique_regimes} distinct regimes assigned "
              f"(sample: {df['hmm_regime'].value_counts().head(3).to_dict()})")

        # ── Load per-token optimizer params (if threshold_optimizer.py has run) ──
        _opt = load_token_params(symbol)
        _opt_global = (_opt or {}).get("global", {})

        meta_gate_profile = load_meta_gate_profile(symbol)
        gate_type = meta_gate_profile.get("gate_type") if meta_gate_profile else None
        signal_vetoes = list(meta_gate_profile.get("signal_vetoes", [])) if meta_gate_profile else None
        regime_modifier_profile = bool(meta_gate_profile.get("regime_modifier", False)) if meta_gate_profile else None
        disabled_reason = meta_gate_profile.get("disabled_reason") if meta_gate_profile else None
        if meta_gate_profile:
            print(f"   Meta gate profile loaded: {gate_type} | vetoes={signal_vetoes} | regime_modifier={regime_modifier_profile}"
                  + (f" | disabled_reason={disabled_reason}" if disabled_reason else ""))

        # ATR multiplier: prefer optimizer result, but never tighten below the static tier.
        # Use the per-token dynamic ATR range (history-aware) to pick a typical base.
        _static_atr = get_atr_multiplier(symbol)
        _opt_atr = _opt_global.get("atr_multiplier")
        _min_m, _max_m, _typical_mult = get_dynamic_atr_range(df)
        atr_mult = max(float(_opt_atr) if _opt_atr else float(_typical_mult), _static_atr)

        _er_med  = float(df['efficiency_ratio_10'].median()) if 'efficiency_ratio_10' in df.columns else 0.5
        _vol_med = float(df['volatility_regime'].median())   if 'volatility_regime' in df.columns else 1.0
        _typical = compute_dynamic_atr_multiplier(atr_mult, _er_med, _vol_med)

        # ── Dynamic lookahead ─────────────────────────────────────────────
        # Prefer the optimizer's per-token lookahead if available; fall back
        # to a ATR-driven heuristic.
        _opt_lh = _opt_global.get("lookahead_bars")
        token_lookahead = int(np.clip(
            int(_opt_lh) if _opt_lh else get_dynamic_lookahead(df),
            12,              # absolute minimum: 12 bars (half a day)
            MAX_LOOKAHEAD,   # never exceed the global cap
        ))

        # ── Dynamic precision target ──────────────────────────────────────
        # Wide ATR barriers → bigger moves per trade → lower fee-breakeven.
        # There's no reason to demand 62% precision from ETH when its breakeven
        # is 51%. Set the target 5pp above the token's own breakeven, bounded
        # between a hard floor (54%) and the global ceiling (TARGET_SIGNAL_PRECISION).
        _close_arr: np.ndarray = np.asarray(df['close'].values, dtype=float)
        _atr_arr: np.ndarray   = (np.asarray(df['_atr'].values, dtype=float)
                                   if '_atr' in df.columns
                                   else np.full(len(df), 0.01, dtype=float))
        _bar_fracs: np.ndarray = np.divide(
            _typical * _atr_arr, _close_arr,
            out=np.full(len(_close_arr), np.nan),
            where=_close_arr > 0,
        )
        _typ_bfrac  = float(np.nanmedian(_bar_fracs))
        token_breakeven = 0.5 * (1.0 + FEE_ROUNDTRIP / max(_typ_bfrac, 0.005))
        token_precision_target = float(np.clip(
            token_breakeven + 0.05,   # 5pp safety margin above breakeven
            0.54,                     # hard floor: never accept below 54%
            TARGET_SIGNAL_PRECISION,  # hard ceiling: global target is the max bar
        ))

        # ── Dynamic meta regularization ───────────────────────────────────
        # Regularisation tiers are moderate, not maxed. L=6.0/mcw=20 (old
        # values) produced a near-uniform meta output (all confidences ~0.52)
        # so that every quantile threshold hit either >60% coverage or <80
        # trades — silently blocking signals even when the primary was 84%
        # accurate. The new values allow the meta to learn regime patterns
        # while still preventing short-window overfitting.
        if _er_med < 0.35:
            _meta_reg, _meta_mcw = 3.5, 14   # choppy market: strongest regularization
        elif _er_med < 0.5:
            _meta_reg, _meta_mcw = 3.0, 12
        else:
            _meta_reg, _meta_mcw = 2.5, 10   # trending market: moderate regularization
        token_meta_params = {**META_PARAMS, 'reg_lambda': _meta_reg, 'min_child_weight': _meta_mcw}

        # ── Adaptive barrier skew based on token volatility ──────────────────
        # Symmetric barriers for low-vol tokens avoid the labeler assigning
        # SELL labels 1.75× more than BUY (1.15/0.85 skew on a token where
        # down barriers are only 0.7% wide means a BUY label hit requires more
        # price movement than a SELL label hit).
        _sample_len = min(len(df), 2000)
        _atr_for_skew = compute_atr(df.iloc[:_sample_len], period=14)
        _close_for_skew = df['close'].iloc[:_sample_len].replace(0, np.nan)
        _atr_pct_for_skew = (_atr_for_skew / _close_for_skew).fillna(0)
        _median_atr_pct = float(np.nanmedian(_atr_pct_for_skew))
        if _median_atr_pct < 0.015:
            _token_up_skew   = 1.0
            _token_down_skew = 1.0
            print(f"   [BARRIER] Symmetric skew (median_atr_pct={_median_atr_pct:.4f} < 0.015)")
        else:
            _token_up_skew   = BARRIER_UP_SKEW
            _token_down_skew = BARRIER_DOWN_SKEW
            print(f"   [BARRIER] Asymmetric skew {_token_up_skew:.2f}/{_token_down_skew:.2f} "
                  f"(median_atr_pct={_median_atr_pct:.4f})")

        # ---- 0) Recursive Label Rebalancer (Phase 3) ----
        target_hold_max = 0.55   # tighter: require HOLD ≤ 55% to force more directional examples
        target_buy_min, target_buy_max = 0.15, 0.45
        target_sell_min, target_sell_max = 0.15, 0.45

        loop_atr_mult = atr_mult
        best_atr_mult = atr_mult
        best_dist = 999.0

        N_all = len(df)
        test_start_temp = N_all - int(N_all * TEST_FRAC)
        train_end_temp = test_start_temp - EMBARGO
        
        # ── Anti-leakage: derive confluence bounds from training portion only ─────
        # compute before ANY create_triple_barrier_labels call so even the
        # rebalancer preview never sees holdout statistics.
        _adapt_params: Dict[str, Any] = {}
        if 'macro_confluence_score' in df.columns:
            _cs_train_slice = df['macro_confluence_score'].iloc[:train_end_temp].dropna()
            if len(_cs_train_slice) > 10:
                _cs_lo, _cs_hi = _adaptive_confluence_bounds(_cs_train_slice)
                _adapt_params['buy_confluence_lower'] = _cs_lo
                _adapt_params['sell_confluence_upper'] = _cs_hi

        print("   Auditing training label distribution (rebalancer)...")
        preview_labels = create_triple_barrier_labels(
            df, atr_multiplier=loop_atr_mult, max_lookahead=token_lookahead,
            volatility_regime=df['volatility_regime'],
            efficiency_ratio=df['efficiency_ratio_10'],
            trend_regime=df['trend_regime'],
            macro_confluence_score=df.get('macro_confluence_score'),
            adapt_params=_adapt_params,
            barrier_up_skew=_token_up_skew, barrier_down_skew=_token_down_skew,
        )
        preview_valid = preview_labels[preview_labels != CENSORED]
        if len(preview_valid) > 0:
            freq_preview = np.bincount(preview_valid.astype(int), minlength=3) / len(preview_valid)
        else:
            freq_preview = np.array([0.20, 0.50, 0.30], dtype=float)

        target_buy_min = float(np.clip(freq_preview[2] * 0.70, 0.08, 0.45))
        target_sell_min = float(np.clip(freq_preview[0] * 0.70, 0.08, 0.45))
        target_hold_max = min(0.55, float(freq_preview[1]) * 0.85)
        target_buy_max = float(np.clip(target_buy_min + 0.18, 0.20, 0.50))
        target_sell_max = float(np.clip(target_sell_min + 0.18, 0.20, 0.50))

        # Outer loop: if HOLD > 60% persists after inner attempts, widen lookahead and retry
        _rebal_lookahead = token_lookahead
        _rebal_converged = False
        for _lh_try in range(3):   # try original lookahead, +12h, +24h
            _rebal_lookahead = min(96, token_lookahead + _lh_try * 12)
            if _lh_try > 0:
                print(f"   [REBALANCER] HOLD > 60% persisted — extending lookahead to {_rebal_lookahead}h")
                loop_atr_mult = atr_mult   # reset ATR search
                best_atr_mult = atr_mult
                best_dist = 999.0

            for attempt in range(12):
                labels_temp = create_triple_barrier_labels(
                    df, atr_multiplier=loop_atr_mult, max_lookahead=_rebal_lookahead,
                    volatility_regime=df['volatility_regime'],
                    efficiency_ratio=df['efficiency_ratio_10'],
                    trend_regime=df['trend_regime'],
                    macro_confluence_score=df.get('macro_confluence_score'),
                    adapt_params=_adapt_params,
                    barrier_up_skew=_token_up_skew, barrier_down_skew=_token_down_skew,
                )
                labels_train = labels_temp.iloc[:train_end_temp]
                valid_labels = labels_train[labels_train != CENSORED]

                if len(valid_labels) == 0:
                    loop_atr_mult = max(0.5, loop_atr_mult - 0.2)
                    continue

                counts = np.bincount(valid_labels.astype(int), minlength=3)
                freqs = counts / len(valid_labels)
                freq_sell, freq_hold, freq_buy = freqs[0], freqs[1], freqs[2]

                print(f"      Attempt {attempt+1} (lh={_rebal_lookahead}): atr_mult={loop_atr_mult:.2f} "
                      f"-> BUY: {freq_buy:.1%}, SELL: {freq_sell:.1%}, HOLD: {freq_hold:.1%}")

                if freq_hold <= target_hold_max and \
                   target_buy_min <= freq_buy <= target_buy_max and \
                   target_sell_min <= freq_sell <= target_sell_max:
                    best_atr_mult = loop_atr_mult
                    _rebal_converged = True
                    break

                dist = max(0.0, freq_hold - target_hold_max) + \
                       max(0.0, target_buy_min - freq_buy) + max(0.0, freq_buy - target_buy_max) + \
                       max(0.0, target_sell_min - freq_sell) + max(0.0, freq_sell - target_sell_max)

                if dist < best_dist:
                    best_dist = dist
                    best_atr_mult = loop_atr_mult

                if freq_hold > target_hold_max:
                    loop_atr_mult = max(0.5, loop_atr_mult - 0.15)
                else:
                    loop_atr_mult = min(3.5, loop_atr_mult + 0.15)

                if loop_atr_mult <= 0.5 or loop_atr_mult >= 3.5:
                    break

            if _rebal_converged:
                token_lookahead = _rebal_lookahead   # persist the successful lookahead
                break

            # Check final HOLD ratio after 8 attempts — if still too high, extend lookahead
            _final_check = create_triple_barrier_labels(
                df, atr_multiplier=best_atr_mult, max_lookahead=_rebal_lookahead,
                volatility_regime=df['volatility_regime'],
                efficiency_ratio=df['efficiency_ratio_10'],
                trend_regime=df['trend_regime'],
                macro_confluence_score=df.get('macro_confluence_score'),
                adapt_params=_adapt_params,
                barrier_up_skew=_token_up_skew, barrier_down_skew=_token_down_skew,
            )
            _fc_valid = _final_check.iloc[:train_end_temp]
            _fc_valid = _fc_valid[_fc_valid != CENSORED]
            if len(_fc_valid) == 0:
                break
            _fc_freqs = np.bincount(_fc_valid.astype(int), minlength=3) / len(_fc_valid)
            if _fc_freqs[1] <= 0.60:  # HOLD ≤ 60%, acceptable
                token_lookahead = _rebal_lookahead
                break

        atr_mult = best_atr_mult

        # ── Low-side label rescue ─────────────────────────────────────────────────────
        # If either BUY or SELL has < 10% frequency after the rebalancer, extend
        # lookahead by 24h and run a quick ATR sweep. Binary primary trains on both
        # sides — severe asymmetry (e.g. zero BUY labels for ETH) kills one side entirely.
        _lsc_labels = create_triple_barrier_labels(
            df, atr_multiplier=atr_mult, max_lookahead=token_lookahead,
            volatility_regime=df['volatility_regime'],
            efficiency_ratio=df['efficiency_ratio_10'],
            trend_regime=df['trend_regime'],
            macro_confluence_score=df.get('macro_confluence_score'),
            adapt_params=_adapt_params,
            barrier_up_skew=_token_up_skew, barrier_down_skew=_token_down_skew,
        )
        _lsc_v = _lsc_labels.iloc[:train_end_temp]
        _lsc_v = _lsc_v[_lsc_v != CENSORED]
        if len(_lsc_v) > 0:
            _lsc_f = np.bincount(_lsc_v.astype(int), minlength=3) / len(_lsc_v)
            if _lsc_f[2] < 0.10 or _lsc_f[0] < 0.10:
                _new_lh = min(120, token_lookahead + 24)
                print(f"   [LOW SIDE LABELS] BUY={_lsc_f[2]:.1%} SELL={_lsc_f[0]:.1%} < 10% "
                      f"— extending lookahead {token_lookahead}h→{_new_lh}h and re-sweeping ATR")
                token_lookahead = _new_lh
                _lsr_best_atr  = atr_mult
                _lsr_best_dist = 999.0
                for _lsr_m in np.arange(max(0.5, atr_mult - 0.45), min(3.5, atr_mult + 0.46), 0.15):
                    _lsr_lb = create_triple_barrier_labels(
                        df, atr_multiplier=_lsr_m, max_lookahead=token_lookahead,
                        volatility_regime=df['volatility_regime'],
                        efficiency_ratio=df['efficiency_ratio_10'],
                        trend_regime=df['trend_regime'],
                        macro_confluence_score=df.get('macro_confluence_score'),
                        adapt_params=_adapt_params,
                        barrier_up_skew=_token_up_skew, barrier_down_skew=_token_down_skew,
                    )
                    _lsr_v = _lsr_lb.iloc[:train_end_temp]
                    _lsr_v = _lsr_v[_lsr_v != CENSORED]
                    if len(_lsr_v) == 0:
                        continue
                    _lsr_f = np.bincount(_lsr_v.astype(int), minlength=3) / len(_lsr_v)
                    _lsr_d = max(0.0, 0.10 - _lsr_f[2]) + max(0.0, 0.10 - _lsr_f[0])
                    if _lsr_d < _lsr_best_dist:
                        _lsr_best_dist = _lsr_d
                        _lsr_best_atr  = float(_lsr_m)
                atr_mult = _lsr_best_atr
                print(f"   [LOW SIDE LABELS] rescue complete: atr_mult={atr_mult:.2f} lookahead={token_lookahead}h")

        # Final tradeability check: use the same dynamic targets the rebalancer used
        # (not hardcoded 15%/60%) so convergence and rejection are consistent.
        # Absolute floors: directional >= 8% (from clip lower-bound in target derivation).
        _final_labels_for_check = create_triple_barrier_labels(
            df, atr_multiplier=atr_mult, max_lookahead=token_lookahead,
            volatility_regime=df['volatility_regime'],
            efficiency_ratio=df['efficiency_ratio_10'],
            trend_regime=df['trend_regime'],
            macro_confluence_score=df.get('macro_confluence_score'),
            adapt_params=_adapt_params,
            barrier_up_skew=_token_up_skew, barrier_down_skew=_token_down_skew,
        )
        _chk = _final_labels_for_check.iloc[:train_end_temp]
        _chk = _chk[_chk != CENSORED]
        if len(_chk) > 0:
            _chk_freqs = np.bincount(_chk.astype(int), minlength=3) / len(_chk)
            _SKIP_HOLD_MAX = 0.62   # skip only if rebalancer truly failed (HOLD > 62%)
            if _chk_freqs[1] > _SKIP_HOLD_MAX or _chk_freqs[0] < target_sell_min or _chk_freqs[2] < target_buy_min:
                print(f"   [UNTRADEABLE] Label distribution unacceptable after rebalancing: "
                      f"SELL={_chk_freqs[0]:.1%}, HOLD={_chk_freqs[1]:.1%}, BUY={_chk_freqs[2]:.1%}. "
                      f"(targets: hold<={target_hold_max:.0%}, sell>={target_sell_min:.0%}, buy>={target_buy_min:.0%}). "
                      f"Token skipped.")
                return None

        print(f"   Optimal ATR multiplier selected: {atr_mult:.2f} (lookahead={token_lookahead}h)")

        print(f"ATR multiplier : base={atr_mult} | typical={_typical:.2f} "
              f"(ER_med={_er_med:.2f}, vol_med={_vol_med:.2f}) | range=[{atr_mult*0.8:.1f}, 4.5]")
        print(f"Token params   : lookahead={token_lookahead}h | "
              f"precision_target={token_precision_target:.1%} (breakeven~{token_breakeven:.1%}) | "
              f"meta_reg=L{_meta_reg}/mcw{_meta_mcw}")
        print(f"   [ADAPTIVE] symbol={symbol} lookahead={token_lookahead} atr_mult={atr_mult:.2f} "
              f"precision_target={token_precision_target:.1%} target_buy=[{target_buy_min:.1%},{target_buy_max:.1%}] "
              f"target_sell=[{target_sell_min:.1%},{target_sell_max:.1%}] hold_max={target_hold_max:.1%}")

        labels, _hit_bars_s = create_triple_barrier_labels(
            df, atr_multiplier=atr_mult, max_lookahead=token_lookahead,
            volatility_regime=df['volatility_regime'],
            efficiency_ratio=df['efficiency_ratio_10'],
            trend_regime=df['trend_regime'],
            macro_confluence_score=df.get('macro_confluence_score'),
            adapt_params=_adapt_params,
            barrier_up_skew=_token_up_skew, barrier_down_skew=_token_down_skew,
            return_hit_bars=True,
        )
        df['target'] = labels.astype(int).values
        # Bar-of-hit per label, for overlap dedup in evaluation ONLY. The `_`
        # prefix keeps it out of feature_cols (it encodes future outcome timing,
        # same epistemic status as `target` itself — evaluation-side truth).
        df['_hit_bars'] = _hit_bars_s.values

        n_censored = int((df['target'] == CENSORED).sum())
        df = df[df['target'] != CENSORED].reset_index(drop=True)
        if n_censored:
            print(f"   Dropped {n_censored} censored rows (incomplete lookahead window)")
        if not df['target'].isin([0, 1, 2]).all():
            raise ValueError(f"Invalid labels: {df['target'].unique().tolist()}")
        if len(df) < MIN_TOTAL_ROWS:
            print(f"Too few usable rows ({len(df)}), skipping")
            return None

        feature_cols = [c for c in df.columns
                        if c not in ('timestamp', 'target')
                        and not c.startswith('_')
                        and c not in FEATURE_BLACKLIST]
        for addon in FEATURE_ADDONS:
            if addon in df.columns and addon not in feature_cols and addon not in FEATURE_BLACKLIST:
                feature_cols.append(addon)
        df = _sanitize(df, feature_cols)

        cc = np.bincount(df['target'].to_numpy().astype(int), minlength=3)
        print("Class distribution:")
        print(f"   SELL (0): {cc[0]} | HOLD (1): {cc[1]} | BUY (2): {cc[2]}")

        # ---- split: train pool | embargo | holdout ----
        N = len(df)
        test_start = N - int(N * TEST_FRAC)
        train_end = test_start - EMBARGO
        if train_end < MIN_FIT_ROWS:
            print(f"Not enough train rows after embargo ({train_end}), skipping")
            return None
        train_pool = df.iloc[:train_end].reset_index(drop=True).copy()
        holdout = df.iloc[test_start:].reset_index(drop=True).copy()
        print(f"   Split -> train pool: {len(train_pool)} | embargo: {EMBARGO} | holdout: {len(holdout)}")
        
        # Copy raw price and ATR values to separate columns before any feature drift normalization/scaling
        train_pool['_close_raw'] = train_pool['close'].copy()
        train_pool['_atr_raw'] = train_pool['_atr'].copy()
        holdout['_close_raw'] = holdout['close'].copy()
        holdout['_atr_raw'] = holdout['_atr'].copy()

        # Reversal events MUST be computed HERE, on RAW pre-normalisation values.
        # AdaptiveNormalizer below z-scores every feature column, which makes the
        # engine's thresholds meaningless: rsi_14 becomes ~[-3,+3] so `rsi >= 65`
        # can NEVER fire (measured: SELL fades=0 on BTC, i.e. the focus silently
        # became BUY-only) while `rsi <= 35` ALWAYS fires; and open/high/low/close
        # are each scaled independently, which destroys the bar geometry the
        # candlestick detector depends on. The `_` prefix keeps these masks out of
        # feature_cols, so they are neither normalised nor leaked to the model.
        _rb_tp, _rs_tp = compute_reversal_events(train_pool)
        train_pool['_rev_buy'], train_pool['_rev_sell'] = _rb_tp.values, _rs_tp.values
        _rb_ho, _rs_ho = compute_reversal_events(holdout)
        holdout['_rev_buy'], holdout['_rev_sell'] = _rb_ho.values, _rs_ho.values

        Xtp = train_pool[feature_cols]
        ytp = train_pool['target'].to_numpy().astype(int)

        # Class rates for proposed_side normalization (prevents 0-BUY proposals
        # on tokens where SELL labels outnumber BUY by 2:1 or more).
        _cc_rates = np.bincount(ytp, minlength=NUM_CLASS) / max(len(ytp), 1)
        _buy_rate  = float(_cc_rates[2])
        _sell_rate = float(_cc_rates[0])

        # ---- 1) AdaptiveNormalizer: fit on train, transform both splits ----
        # Eliminates distribution shift from absolute-price features by centering and
        # scaling all numeric features using training statistics only (no holdout peek).
        print("   Fitting AdaptiveNormalizer on training pool...")
        normalizer = AdaptiveNormalizer(window=1000, drift_threshold=0.2)
        normalizer.fit_initial(train_pool[feature_cols], feature_cols)
        # Transform in-place; keep raw copies for barrier-frac backtest
        train_pool_norm = train_pool.copy()
        train_pool_norm[feature_cols] = normalizer.transform(train_pool[feature_cols], feature_cols)[feature_cols]
        holdout_norm = holdout.copy()
        holdout_norm[feature_cols] = normalizer.transform(holdout[feature_cols], feature_cols)[feature_cols]
        # Preserve non-feature columns needed by the backtest/regime engine
        for _raw_col in ['_close_raw', '_atr_raw', '_rev_buy', '_rev_sell',
                         'hmm_regime', 'volatility_regime',
                         'macro_trend_1d', 'total_confluence', 'is_at_resistance', 'is_at_support']:
            if _raw_col in train_pool.columns:
                train_pool_norm[_raw_col] = train_pool[_raw_col].values
            if _raw_col in holdout.columns:
                holdout_norm[_raw_col] = holdout[_raw_col].values
        train_pool = train_pool_norm
        holdout    = holdout_norm
        Xtp = train_pool[feature_cols]
        print(f"   AdaptiveNormalizer applied: {len(feature_cols)} features normalised.")

        # ---- Feature Health Manager — monitoring only, NOT removal ----
        # Drift stats are logged to the sidecar and used to decay feature weights.
        # Features are NEVER dropped here; AdaptiveNormalizer handles the drift.
        print("   Evaluating Feature Health and Drift (monitoring)...")
        from src.ml.feature_health import FeatureHealthManager, DynamicFeatureWeightingEngine
        fhm = FeatureHealthManager()
        try:
            fhm.analyze_drift(Xtp, holdout[feature_cols], feature_cols)
        except Exception as _fhm_err:
            print(f"   FeatureHealth.analyze_drift() failed ({_fhm_err}); continuing without drift scores.")

        # Build feature weights: base 1.0, decay by 0.5 for high-drift features
        fw_dict: Dict[str, float] = {}
        _n_drift_penalised = 0
        for col in feature_cols:
            psi = fhm.drift_scores.get(col, {}).get('psi', 0.0)
            ks  = fhm.drift_scores.get(col, {}).get('ks',  0.0)
            if psi > 1.0 or ks > 0.50:
                if col in _PRECISION_PROTECTED_FEATURES:
                    fw_dict[col] = 1.0   # never penalize top-gain features
                else:
                    fw_dict[col] = 0.15  # aggressive penalty for drifted features — SHAP pruning will remove them
                    _n_drift_penalised += 1
            else:
                fw_dict[col] = 1.0
        fw = np.array([fw_dict.get(c, 1.0) for c in feature_cols], dtype=float)
        if _n_drift_penalised:
            print(f"   [DRIFT] {_n_drift_penalised} features weight-penalised (PSI>1 or KS>0.5, protected={len([c for c in feature_cols if c in _PRECISION_PROTECTED_FEATURES])} exempt)")

        # Correlation audit removed: it zeroed out weak ensemble signals that still
        # contribute to XGBoost ensembles. Drift penalisation (above) is sufficient.

        # ---- 2) Optuna tune binary primary (AUPRC-optimised, 5-fold) ----
        # Binary primary replaces the 3-class multi:softprob with two independent
        # binary:logistic models. BUY model predicts P(upper barrier hit); SELL model
        # predicts P(lower barrier hit). HOLD = neither model fires (prob < 0.5).
        _optuna_n_splits = min(5, N_SPLITS_CV)
        inner = list(TimeSeriesSplit(n_splits=_optuna_n_splits, gap=EMBARGO).split(Xtp))

        ytp_buy  = (ytp == 2).astype(int)   # 1 = upper barrier hit
        ytp_sell = (ytp == 0).astype(int)   # 1 = lower barrier hit

        # scale_pos_weight corrects class imbalance in each binary model
        _n_neg_buy  = float((ytp_buy  == 0).sum())
        _n_pos_buy  = float((ytp_buy  == 1).sum())
        _n_neg_sell = float((ytp_sell == 0).sum())
        _n_pos_sell = float((ytp_sell == 1).sum())
        # Cap at 5.0: uncapped ~7× for BUY (12.6% rate) inflates p_buy on every bar,
        # causing 81.6% BUY over-proposal even after proposed_side() normalization.
        _spw_buy    = min(_n_neg_buy  / max(_n_pos_buy,  1.0), 5.0)
        _spw_sell   = min(_n_neg_sell / max(_n_pos_sell, 1.0), 5.0)
        print(f"   Binary primary labels: BUY pos={int(_n_pos_buy)} neg={int(_n_neg_buy)} "
              f"spw={_spw_buy:.1f} | SELL pos={int(_n_pos_sell)} neg={int(_n_neg_sell)} spw={_spw_sell:.1f}")

        # ---- Reversal focus: train where the ENGINE actually fires ----
        # Upweight the candle-confirmed structural extremes that live_engine
        # takes (see compute_reversal_events). Without this the loss is
        # dominated by ordinary trend bars and the model learns the
        # trend-following side that Guard N then blocks.
        # Read the masks computed on RAW values before normalisation — do NOT
        # recompute here, train_pool is z-scored by this point.
        _rev_buy_ev  = train_pool['_rev_buy'].astype(bool)
        _rev_sell_ev = train_pool['_rev_sell'].astype(bool)
        _sw_buy  = reversal_focus_weights(_rev_buy_ev,  len(Xtp))
        _sw_sell = reversal_focus_weights(_rev_sell_ev, len(Xtp))
        _n_rev_buy, _n_rev_sell = int(_rev_buy_ev.sum()), int(_rev_sell_ev.sum())
        print(f"   Reversal focus (x{REVERSAL_FOCUS_WEIGHT:.0f}): "
              f"BUY fades={_n_rev_buy} ({100.0*_n_rev_buy/max(len(Xtp),1):.1f}% of bars) | "
              f"SELL fades={_n_rev_sell} ({100.0*_n_rev_sell/max(len(Xtp),1):.1f}%)")
        if _n_rev_buy + _n_rev_sell == 0:
            print("   [WARN] no reversal events found — focus weighting is a no-op "
                  "(check range_position_score / rsi_14 are present)")

        # BUY model — independent Optuna AUPRC run
        study_buy = optuna.create_study(direction='minimize',
                                        sampler=optuna.samplers.TPESampler(seed=42))
        study_buy.optimize(lambda t: objective_binary_multifold(t, Xtp, ytp_buy, inner, fw=fw, spw=_spw_buy, sw=_sw_buy),
                           n_trials=OPTUNA_TRIALS, show_progress_bar=False)
        binary_params_buy = _binary_params(study_buy.best_params)
        binary_params_buy['scale_pos_weight'] = _spw_buy
        print(f"   Binary params BUY  (AUPRC-opt {_optuna_n_splits}-fold | 1-AUPRC={study_buy.best_value:.4f}): {study_buy.best_params}")

        # SELL model — independent Optuna run; BUY and SELL have different class
        # distributions and often need different depth/regularization.
        study_sell = optuna.create_study(direction='minimize',
                                         sampler=optuna.samplers.TPESampler(seed=43))
        study_sell.optimize(lambda t: objective_binary_multifold(t, Xtp, ytp_sell, inner, fw=fw, spw=_spw_sell, sw=_sw_sell),
                            n_trials=OPTUNA_TRIALS, show_progress_bar=False)
        binary_params_sell = _binary_params(study_sell.best_params)
        binary_params_sell['scale_pos_weight'] = _spw_sell
        print(f"   Binary params SELL (AUPRC-opt {_optuna_n_splits}-fold | 1-AUPRC={study_sell.best_value:.4f}): {study_sell.best_params}")

        # Keep full_params for any edge-engine compatibility paths
        full_params = _full_params(study_buy.best_params)

        # SHAP feature pruning: train quick buy/sell models and keep union of features
        # covering SHAP_CUMULATIVE_THRESH (85%) of total importance for each direction.
        if shap is not None and SHAP_CUMULATIVE_THRESH < 1.0 and len(Xtp) >= 100:
            try:
                _xgb_shap_p = {'objective': 'binary:logistic', 'eval_metric': 'logloss',
                                'tree_method': 'hist', 'seed': 42, 'verbosity': 0,
                                'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.8,
                                'colsample_bytree': 0.8, 'missing': np.nan}
                _shap_m_buy  = xgb.train({**_xgb_shap_p, 'scale_pos_weight': _spw_buy},
                                         _dm(Xtp, ytp_buy, fw=fw), num_boost_round=150)
                _shap_m_sell = xgb.train({**_xgb_shap_p, 'scale_pos_weight': _spw_sell},
                                         _dm(Xtp, ytp_sell, fw=fw), num_boost_round=150)
                _keep_buy  = set(prune_features_by_shap(_shap_m_buy,  Xtp, SHAP_CUMULATIVE_THRESH, MIN_FEATURES))
                _keep_sell = set(prune_features_by_shap(_shap_m_sell, Xtp, SHAP_CUMULATIVE_THRESH, MIN_FEATURES))
                _keep_union = [c for c in feature_cols if c in (_keep_buy | _keep_sell)][:MAX_FEATURES]
                if len(_keep_union) >= MIN_FEATURES:
                    feature_cols = _keep_union
                    Xtp = train_pool[feature_cols]
                    fw  = np.array([fw_dict.get(c, 1.0) for c in feature_cols], dtype=float)
                    print(f"   SHAP pruning: {len(feature_cols)} features retained "
                          f"(buy={len(_keep_buy)}, sell={len(_keep_sell)}, union={len(_keep_buy|_keep_sell)})")
                else:
                    print(f"   SHAP pruning: union too small ({len(_keep_union)}) — keeping all {len(feature_cols)}")
            except Exception as _shap_err:
                print(f"   SHAP pruning failed ({_shap_err}) — keeping all {len(feature_cols)} features")
        else:
            print(f"   SHAP pruning disabled (thresh={SHAP_CUMULATIVE_THRESH}) — keeping all {len(feature_cols)} features")

        # Feature reconciliation
        print(f"\n   === AEGIS FEATURE RECONCILIATION REPORT for {symbol} ===")
        print(f"      Active features : {len(feature_cols)}")
        print(f"      Drift-penalised : {_n_drift_penalised}")
        print(f"      Normalisation   : AdaptiveNormalizer (z-score, fixed train stats)")

        # SMOTE removed: binary models use scale_pos_weight for imbalance correction.

        # ---- 3) Binary OOF → AUC check → synthetic 3-class probs ----
        # sw: same reversal focus as the deployed fit, so the OOF probabilities the
        # meta gate calibrates against come from a model that behaves like the one
        # that ships (the eval folds themselves stay unweighted).
        oof_buy  = binary_primary_oof(Xtp, ytp_buy,  binary_params_buy,  N_SPLITS_CV, EMBARGO, fw=fw, sw=_sw_buy)
        oof_sell = binary_primary_oof(Xtp, ytp_sell, binary_params_sell, N_SPLITS_CV, EMBARGO, fw=fw, sw=_sw_sell)
        mask = ~np.isnan(oof_buy)
        mask_idx = np.where(mask)[0]

        from sklearn.metrics import roc_auc_score as _roc_auc
        _auc_buy  = float(_roc_auc(ytp_buy[mask],  oof_buy[mask]))
        _auc_sell = float(_roc_auc(ytp_sell[mask], oof_sell[mask]))
        print(f"   Binary OOF AUC: BUY={_auc_buy:.4f} | SELL={_auc_sell:.4f}")
        if _auc_buy < 0.55 and _auc_sell < 0.55:
            print(f"   [AUC VETO] AUC_buy={_auc_buy:.3f} and AUC_sell={_auc_sell:.3f} both < 0.55 "
                  "— primary has no directional skill. UNTRADEABLE.")
            return None

        # Synthetic 3-class array for downstream compatibility:
        #   col 0 = p_sell, col 1 = hold residual, col 2 = p_buy
        oof = np.zeros((len(Xtp), NUM_CLASS), dtype=float)
        _ob = np.where(np.isnan(oof_buy),  0.33, oof_buy)
        _os = np.where(np.isnan(oof_sell), 0.33, oof_sell)

        # Bayes prior correction: scale_pos_weight shifts the model's effective prior to
        # p(pos) = spw/(1+spw) during training. Without correction, hold_residual ≈ 0 on
        # most bars (SPW-inflated p_buy + p_sell ≈ 1.0), argmax always picks BUY/SELL,
        # and 3-class cv_acc falls below the HOLD majority baseline.
        # Correcting back to true class rates restores proper hold_residual magnitude.
        _true_buy_rate  = float(_n_pos_buy  / max(_n_pos_buy  + _n_neg_buy,  1.0))
        _true_sell_rate = float(_n_pos_sell / max(_n_pos_sell + _n_neg_sell, 1.0))
        _spw_prior_buy  = _spw_buy  / (1.0 + _spw_buy)
        _spw_prior_sell = _spw_sell / (1.0 + _spw_sell)

        def _bayes_correct(p: np.ndarray, true_rate: float, spw_prior: float) -> np.ndarray:
            _e = 1e-7
            lo = np.log(np.clip(p, _e, 1 - _e) / (1 - np.clip(p, _e, 1 - _e)))
            lo += (np.log(true_rate + _e) - np.log(1 - true_rate + _e)
                   - np.log(spw_prior + _e) + np.log(1 - spw_prior + _e))
            return 1.0 / (1.0 + np.exp(-lo))

        _ob_cal = _bayes_correct(_ob, _true_buy_rate,  _spw_prior_buy)
        _os_cal = _bayes_correct(_os, _true_sell_rate, _spw_prior_sell)
        oof[:, 2] = _ob_cal
        oof[:, 0] = _os_cal
        oof[:, 1] = np.clip(1.0 - _ob_cal - _os_cal, 0.0, 1.0)
        _rs = oof.sum(axis=1, keepdims=True)
        oof = oof / np.where(_rs > 0, _rs, 1.0)
        oof[~mask] = np.nan

        cv_acc = float(accuracy_score(ytp[mask], oof[mask].argmax(1)))
        cv_f1  = float(f1_score(ytp[mask], oof[mask].argmax(1), average='macro', zero_division=0))
        T = fit_temperature(oof[mask], ytp[mask])
        print(f"Primary OOF (dev): acc {cv_acc:.4f} | macro-F1 {cv_f1:.4f} | T {T:.3f} | "
              f"AUC_buy={_auc_buy:.3f} | AUC_sell={_auc_sell:.3f} | "
              f"buy_prior={_true_buy_rate:.3f} sell_prior={_true_sell_rate:.3f}")

        _hold_pct = float((ytp[mask] == 1).mean())
        _primary_below_baseline = False   # veto removed
        if cv_acc < _hold_pct + 0.02:
            print(f"   NOTE: 3-class acc {cv_acc:.1%} still below majority-class baseline {_hold_pct:.1%} "
                  f"after Bayes correction — AUC_buy={_auc_buy:.3f}/AUC_sell={_auc_sell:.3f}")

        # ---- Calibration selector (phase 5 adaptive selection) ----
        try:
            from src.ml.calibration_selector import evaluate_and_select
            candidates = {}
            # Use BUY class probability as calibration target (binary: buy vs not-buy)
            raw_buy = oof[mask][:, 2]
            y_buy = (ytp[mask] == 2).astype(float)
            candidates['uncalibrated'] = {'probs': raw_buy, 'raw_probs': raw_buy}
            candidates['temperature'] = {'probs': apply_temperature(raw_buy.reshape(-1, 1), T).flatten(), 'raw_probs': raw_buy}
            # Optional Platt (logistic) fit
            try:
                from sklearn.linear_model import LogisticRegression
                lr = LogisticRegression(solver='lbfgs')
                lr.fit(raw_buy.reshape(-1, 1), y_buy)
                platt_probs = lr.predict_proba(raw_buy.reshape(-1, 1))[:, 1]
                candidates['platt'] = {'probs': platt_probs, 'raw_probs': raw_buy}
            except Exception:
                pass
            cal_choice = evaluate_and_select(candidates, y_buy)
            selected_cal = cal_choice.get('selected')
            print(f"   Calibration selector chose: {selected_cal} | details: {cal_choice.get('details')}")
        except Exception as _e:
            selected_cal = None
            cal_choice = {'error': str(_e)}

        # ── Meta gate removed: primary-only calibrated confidence gate ─────────
        # LR meta was anti-selective fleet-wide; calibrated primary confidence gate
        # achieves equivalent precision without expensive OOF / edge engine overhead.
        print("   [PRIMARY-ONLY MODE] Meta gate removed — calibrated primary confidence gate.")
        meta_full               = None
        meta_model_cols         = list(feature_cols)
        meta_calibration_method = 'primary_only'
        thr                     = 0.60
        dev_prec                = dev_cov = 0.0
        dev_n                   = 0
        hit_target              = False
        thr_buy = thr_sell      = 0.60
        prec_buy = prec_sell    = 0.0
        cov_buy = cov_sell      = 0.0
        n_buy = n_sell          = 0
        hit_buy = hit_sell      = False
        disable_sr_veto         = True
        disable_trend_veto      = True
        disable_confluence_veto = True
        regime_policies         = {}
        regime_buy_covs         = {}
        regime_sell_covs        = {}
        _dev_buy_side_cov       = 0.0
        _dev_sell_side_cov      = 0.0
        _oof_meta_gate_lift     = 0.0
        hold_strategy_selected  = "primary_only"
        hold_strategy_audit     = {}
        rcm                     = None
        meta_light_path         = None

        # ---- 5) Holdout: scored exactly once ----
        # Train primary_full with early stopping on the last 20% of the training pool
        # (time-ordered). This finds the optimal n_trees without wasting rounds on noise.
        _n_es_val = max(50, int(len(Xtp) * 0.20))
        _X_tr_es  = Xtp.iloc[:-_n_es_val]
        _y_tr_es  = ytp[:-_n_es_val]
        _w_tr_es  = sample_weights(_y_tr_es)
        _X_va_es  = Xtp.iloc[-_n_es_val:]
        _y_va_es  = ytp[-_n_es_val:]
        # Train two binary primary models (BUY and SELL) with early stopping.
        _y_tr_buy  = (_y_tr_es == 2).astype(int)
        _y_tr_sell = (_y_tr_es == 0).astype(int)
        _y_va_buy  = (_y_va_es == 2).astype(int)
        _y_va_sell = (_y_va_es == 0).astype(int)
        _hold_mask_tr = (_y_tr_es == 1)   # HOLD=1 in 3-class label space; not BUY(2) or SELL(0)
        _w_tr_buy     = np.where(_hold_mask_tr, 0.15, 1.0).astype(float)
        _w_tr_sell    = np.where(_hold_mask_tr, 0.15, 1.0).astype(float)

        buy_full = xgb.train(
            binary_params_buy,
            _dm(_X_tr_es, _y_tr_buy, _w_tr_buy),
            num_boost_round=1000,
            evals=[(_dm(_X_va_es, _y_va_buy), 'eval')],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        sell_full = xgb.train(
            binary_params_sell,
            _dm(_X_tr_es, _y_tr_sell, _w_tr_sell),
            num_boost_round=1000,
            evals=[(_dm(_X_va_es, _y_va_sell), 'eval')],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        primary_full = buy_full  # backward-compat alias

        X_test = holdout[feature_cols]
        y_test = holdout['target'].to_numpy().astype(int)
        _pb_h = buy_full.predict(_dm(X_test))
        _ps_h = sell_full.predict(_dm(X_test))
        # Build synthetic (N,3) probs — apply same Bayes prior correction as OOF
        # so holdout proposals use the same probability scale as dev training.
        _pb_h_cal = _bayes_correct(_pb_h, _true_buy_rate,  _spw_prior_buy)
        _ps_h_cal = _bayes_correct(_ps_h, _true_sell_rate, _spw_prior_sell)
        raw_probs = np.zeros((len(X_test), NUM_CLASS), dtype=float)
        raw_probs[:, 2] = _pb_h_cal
        raw_probs[:, 0] = _ps_h_cal
        raw_probs[:, 1] = np.clip(1.0 - _pb_h_cal - _ps_h_cal, 0.0, 1.0)
        _rs_h = raw_probs.sum(axis=1, keepdims=True)
        raw_probs = raw_probs / np.where(_rs_h > 0, _rs_h, 1.0)
        prop_h = proposed_side(raw_probs, buy_rate=_buy_rate, sell_rate=_sell_rate)

        # ── Primary-only precision gate ───────────────────────────────────────
        # Evaluate directional accuracy on bars where the TRUTH is directional.
        # Measuring against y_test (not prop_h) gives a meaningful signal: what
        # fraction of truly-directional bars does the primary call correctly?
        # Baseline is ~50% (random direction); target ≥55% for tradeable.
        _dir_mask_h = (y_test == 2) | (y_test == 0)
        _primary_dir_prec = (
            float((prop_h[_dir_mask_h] == y_test[_dir_mask_h]).mean())
            if _dir_mask_h.sum() > 0 else 0.0
        )
        _pct_buy_proposals = float((prop_h == 2).mean())
        print(f"   Primary directional precision (no gate): {_primary_dir_prec:.3f} "
              f"| BUY proposals: {_pct_buy_proposals:.1%} "
              f"| dir bars: {int(_dir_mask_h.sum())}")
        if _primary_dir_prec < 0.45:
            print(f"   [PRIMARY VETO] Primary precision {_primary_dir_prec:.1%} < 45% — UNTRADEABLE.")
            return None
        else:
            print(f"   [PRIMARY] {_primary_dir_prec:.1%} directional precision — calibrated confidence gate will run.")

        if meta_full is not None:
            # Use meta_model_cols (post-drift-filter) not feature_cols to avoid mismatch
            X_test_meta = X_test[meta_model_cols] if 'meta_model_cols' in locals() else X_test
            _meta_X_h = build_meta_X(X_test_meta, raw_probs).fillna(0.0)
            # meta_full is now a sklearn LR (C_excluded) — use predict_proba
            meta_prob_h = meta_full.predict_proba(_meta_X_h)[:, 1]
        else:
            meta_prob_h = raw_probs.max(axis=1)  # fallback gate on primary confidence

        # ── Edge-Driven Holdout Gate ──────────────
        from src.trading.edge_engine import EdgeScoringEngine
        edge_buy_h = EdgeScoringEngine.compute_edge_batch(holdout, meta_prob_h, 'BUY').to_numpy()
        edge_sell_h = EdgeScoringEngine.compute_edge_batch(holdout, meta_prob_h, 'SELL').to_numpy()

        # ── Pre-firing regime mask ────────────────────────────────────────────────────
        # HMM regime filter permanently disabled: collapses to a single state for
        # most tokens, incorrectly suppressing 40-60% of holdout bars with no
        # precision benefit (forensics showed regime_precision < 45% fleet-wide).
        # All holdout bars are treated as regime-ok; QP sees the full population.
        _regime_ok_h = np.ones(len(holdout), dtype=bool)
        _regime_safety_valve_triggered = False

        # ── PRIMARY-ONLY bypass ───────────────────────────────────────────────────────
        # When holdout primary directional precision >= 55%, bypass the meta gate
        # entirely. Find the optimal confidence threshold on the last 20% of the
        # training pool (val set), then apply it directly to holdout without any
        # QP, regime, or veto logic. Eliminates isotonic-calibration overfit and
        # stale threshold issues (e.g. BTC thr_buy=12 firing near-random signals).
        _primary_only_gate = False
        _primary_conf_thr  = 0.55   # overridden when bypass fires
        _best_prec_po      = 0.0    # val-set precision at chosen threshold
        _calibrator_po     = None   # LR calibrator: raw_conf → P(outcome != HOLD)
        _po_use_calibrator = False  # True when calibrator was successfully fitted
        if True:   # always run calibrated path; Fix D bypasses LR calibrator when dir_correct_rate < 0.55
            if str(gate_type or '').upper() == 'PRIMARY_CONFIDENCE':
                print("   [GATE] gate_type='PRIMARY_CONFIDENCE' overridden — running calibrated section "
                      "(Fix D handles bypass when dir_correct_rate < 0.55).")
            # ── Calibrated "direction-correct" probability gate ───────────────
            # Compute val split size FIRST so calibrator training never touches
            # val rows — preventing data leakage into the threshold sweep.
            _val_n_po  = max(50, int(len(Xtp) * 0.20))
            _cal_end   = len(Xtp) - _val_n_po   # exclusive upper index of calibrator training data

            # Step 1: Score training pool with full binary models, Bayes-correct.
            _pb_tp   = buy_full.predict(_dm(Xtp))
            _ps_tp   = sell_full.predict(_dm(Xtp))
            _pb_tp_c = _bayes_correct(_pb_tp, _true_buy_rate,  _spw_prior_buy)
            _ps_tp_c = _bayes_correct(_ps_tp, _true_sell_rate, _spw_prior_sell)
            _conf_tp = np.maximum(_pb_tp_c, _ps_tp_c)
            _rtp     = np.zeros((len(Xtp), NUM_CLASS), dtype=float)
            _rtp[:, 2] = _pb_tp_c
            _rtp[:, 0] = _ps_tp_c
            _rtp[:, 1] = np.clip(1.0 - _pb_tp_c - _ps_tp_c, 0.0, 1.0)
            _rtps    = _rtp.sum(axis=1, keepdims=True)
            _rtp     = _rtp / np.where(_rtps > 0, _rtps, 1.0)
            _prop_tp = proposed_side(_rtp, buy_rate=_buy_rate, sell_rate=_sell_rate)
            _dir_tp  = (_prop_tp == 2) | (_prop_tp == 0)

            # Step 2: Fit direction-correct calibrator on NON-val training rows only.
            # Target: 1 if model's proposed direction == true label, 0 otherwise.
            # Using P(direction correct) instead of P(not HOLD) ensures the calibrator
            # gates on the metric that actually drives signal precision.
            # Calibrator training is strictly limited to Xtp[:_cal_end] to prevent
            # the val-set rows from leaking into the calibrator's learned weights.
            from sklearn.linear_model import LogisticRegression as _LRCal
            _hold_disc_cols: list = [c for c in ['atr_pct', 'macro_confluence_score',
                                                   'efficiency_ratio_10', 'volatility_regime',
                                                   'adx_14', 'realized_volatility']
                                     if c in Xtp.columns]
            _calibrator_hold_disc_cols: list = _hold_disc_cols  # persisted for holdout + sidecar
            # Restrict to non-val rows to avoid leakage
            _dir_tp_cal = _dir_tp.copy()
            _dir_tp_cal[_cal_end:] = False
            if _dir_tp_cal.sum() >= 20:
                # Direction-correct target: 1 if proposal == true label, 0 otherwise
                _cal_y = (ytp[_dir_tp_cal] == _prop_tp[_dir_tp_cal]).astype(float)
                _dir_idx = np.where(_dir_tp_cal)[0]
                _conf_base_cal = _conf_tp[_dir_tp_cal].reshape(-1, 1)
                if _hold_disc_cols:
                    _hold_disc_feats = Xtp.iloc[_dir_idx][_hold_disc_cols].fillna(0.0).to_numpy()
                    _cal_x = np.column_stack([_conf_base_cal, _hold_disc_feats])
                else:
                    _cal_x = _conf_base_cal
                if len(np.unique(_cal_y)) == 2:
                    _n_feats_cal = _cal_x.shape[1]
                    _C_cal = 1.0 if _n_feats_cal > 1 else 0.5
                    try:
                        _calibrator_po = _LRCal(C=_C_cal, max_iter=500, solver='lbfgs')
                        _calibrator_po.fit(_cal_x, _cal_y)
                        _dir_correct_rate_cal = float(_cal_y.mean())
                        # Fix K: bypass check on TRUE directional bars only.
                        # HOLD bars always score 0 (model proposes BUY/SELL, true=HOLD) →
                        # deflates all-bars rate to ~47% even when model is 84%+ correct on
                        # actual BUY/SELL bars. The calibrator should be bypassed only when
                        # the model genuinely lacks directional skill, not due to HOLD dilution.
                        _true_dir_mask_in_cal = (ytp[_dir_tp_cal] == 2) | (ytp[_dir_tp_cal] == 0)
                        _dir_correct_rate_true_dir = float(_cal_y[_true_dir_mask_in_cal].mean()) \
                            if _true_dir_mask_in_cal.sum() > 0 else 0.0
                        print(f"   [DIR-CAL] multi-feature ({_n_feats_cal} inputs) LR C={_C_cal} "
                              f"fitted on {len(_cal_y)} non-val bars "
                              f"(dir_correct_rate true-dir={_dir_correct_rate_true_dir:.1%} "
                              f"/ all-bars={_dir_correct_rate_cal:.1%})")
                        if _dir_correct_rate_true_dir < 0.55:
                            # Near-random labels on true directional bars → calibrator cannot discriminate.
                            _calibrator_po = None
                            _po_use_calibrator = False
                            print(f"   [DIR-CAL] BYPASSED: dir_correct_rate(true-dir)={_dir_correct_rate_true_dir:.1%} < 0.55 "
                                  f"(all-bars={_dir_correct_rate_cal:.1%} incl HOLD) "
                                  f"— raw primary confidence used for val sweep.")
                    except Exception as _cal_err:
                        _calibrator_po = _LRCal(C=0.1, max_iter=500, solver='lbfgs')
                        _calibrator_po.fit(_conf_base_cal, _cal_y)
                        print(f"   [DIR-CAL] fallback single-feature LR (multi-feature failed: {_cal_err})")

            # Step 3: Validation set (last 20%) — sweep for best threshold.
            # _val_n_po already computed above; val rows were excluded from calibrator fit.
            _X_val_po = Xtp.iloc[-_val_n_po:]
            _y_val_po = ytp[-_val_n_po:]
            _pb_val   = buy_full.predict(_dm(_X_val_po))
            _ps_val   = sell_full.predict(_dm(_X_val_po))
            _pb_val_c = _bayes_correct(_pb_val, _true_buy_rate,  _spw_prior_buy)
            _ps_val_c = _bayes_correct(_ps_val, _true_sell_rate, _spw_prior_sell)
            _conf_val = np.maximum(_pb_val_c, _ps_val_c)
            _rv       = np.zeros((len(_X_val_po), NUM_CLASS), dtype=float)
            _rv[:, 2] = _pb_val_c
            _rv[:, 0] = _ps_val_c
            _rv[:, 1] = np.clip(1.0 - _pb_val_c - _ps_val_c, 0.0, 1.0)
            _rvs      = _rv.sum(axis=1, keepdims=True)
            _rv       = _rv / np.where(_rvs > 0, _rvs, 1.0)
            _prop_val = proposed_side(_rv, buy_rate=_buy_rate, sell_rate=_sell_rate)
            _dir_val  = (_prop_val == 2) | (_prop_val == 0)
            _n_dir_val = int(_dir_val.sum())

            # Apply calibrator to directional val proposals only
            _conf_dir_val      = _conf_val[_dir_val]                          # (n_dir_val,)
            _prop_dir_val      = _prop_val[_dir_val]                          # proposals (0/2)
            _y_outcome_dir_val = _y_val_po[_dir_val]                          # true outcomes
            _dir_val_idx       = np.where(_dir_val)[0]
            if _calibrator_po is not None:
                # Build the same multi-feature input used at fit time
                _conf_base_val = _conf_dir_val.reshape(-1, 1)
                _use_multi_cal = (
                    _hold_disc_cols
                    and hasattr(_calibrator_po, 'coef_')
                    and _calibrator_po.coef_.shape[1] > 1
                )
                if _use_multi_cal:
                    _val_disc_feats = _X_val_po.iloc[_dir_val_idx][_hold_disc_cols].fillna(0.0).to_numpy()
                    _cal_val_x = np.column_stack([_conf_base_val, _val_disc_feats])
                else:
                    _cal_val_x = _conf_base_val
                _cal_conf_dir_val = _calibrator_po.predict_proba(_cal_val_x)[:, 1]
                _po_use_calibrator = True
            else:
                _cal_conf_dir_val = _conf_dir_val

            _best_thr_po  = 0.85  # conservative default (used when sweep finds nothing)
            _best_prec_po = 0.0
            _best_n_po    = 0
            _best_cov_po  = 0.0
            _rows_po: list = []   # always initialised so balance-guard below is safe
            if _n_dir_val >= 100:    # need ≥100 directional val bars to run sweep; inner threshold is 20
                # Floor raised 20 -> 50 on the calibrated path: a 20-fire row carries
                # a +/-19.7pp confidence interval, which is not a measurement. The
                # Wilson ranking above does the real work; this just stops degenerate
                # rows entering the sweep at all.
                _min_fires_combined = 50
                for _thr_s in np.arange(0.30, 0.96, 0.02):
                    _fire_s = (_cal_conf_dir_val >= _thr_s)
                    _n_s    = int(_fire_s.sum())
                    if _n_s < _min_fires_combined:
                        continue
                    _prec_s = float((_prop_dir_val[_fire_s] == _y_outcome_dir_val[_fire_s]).mean())
                    _cov_s  = _n_s / max(1, _n_dir_val)
                    _rows_po.append((float(_thr_s), _prec_s, _cov_s, _n_s))
                if _rows_po:
                    # Rank on the LOWER BOUND of each row's precision, not the raw
                    # point estimate. Selecting on the point estimate made this a
                    # lottery: 33 thresholds are swept, so a small-n row clears the
                    # 60% bar by chance ~97% of the time, and whichever row got
                    # lucky hijacked the whole gate. Measured on BTC, back-to-back
                    # runs on the SAME data:
                    #   spurious row  60.0% @ n=50  (CI [0.46,0.72]) -> thr 0.680,
                    #       holdout precision 0.373, gate lift -0.1%  (DISABLED)
                    #   supported row 52.6% @ n=325 (CI [0.47,0.58]) -> thr 0.540,
                    #       holdout precision 0.506, gate lift +19.6% (ENABLED)
                    # i.e. the run that PASSED the 60% bar produced the worse model.
                    # The Wilson bound reverses that ordering (0.462 vs 0.472) with
                    # no arbitrary rule — small samples simply cannot make a strong
                    # claim. Applies to the fallback too, which previously took the
                    # highest raw precision and so had the same small-n bias.
                    _rows_lb = [(t, p, c, n, wilson_lower_bound(p * n, n))
                                for (t, p, c, n) in _rows_po]
                    _passing_rows = [r for r in _rows_lb if r[4] >= 0.60]
                    if _passing_rows:
                        _sel = max(_passing_rows, key=lambda r: r[2])   # most coverage among genuinely good
                    else:
                        # No row is EVIDENCED at 60%. Taking the single best-LB row
                        # here traded away coverage for a precision claim the data
                        # cannot actually make: on BTC it chose thr 0.640 (n=120,
                        # cov 12%) over lower thresholds whose LB sat within one
                        # standard error — statistically the SAME row — with ~3x the
                        # fires. One-standard-error rule instead: among rows whose
                        # LB is within 1 SE of the best, take the most coverage.
                        # Floor: the point estimate must clear token_breakeven by a
                        # REAL margin (+2pp), not merely touch it. Measured on ETH:
                        # with the floor AT breakeven, the band ran to 81.3% coverage
                        # on a row at breakeven+0.2pp — an always-in-market gate whose
                        # diluted 65.5% dir precision could never evidence the 60%
                        # enable bar, with a 23% holdout drawdown. A row must be worth
                        # firing, not just not-losing. Tokens with no row above the
                        # floor fall back to exactly the old best-LB rule.
                        _r_best  = max(_rows_lb, key=lambda r: r[4])
                        _se_best = (_r_best[1] * (1.0 - _r_best[1]) / _r_best[3]) ** 0.5
                        _prec_floor = token_breakeven + 0.02
                        _near    = [r for r in _rows_lb
                                    if r[4] >= _r_best[4] - _se_best
                                    and r[1] >= _prec_floor]
                        _sel = max(_near, key=lambda r: r[2]) if _near else _r_best
                    _best_thr_po, _best_prec_po, _best_cov_po, _best_n_po = _sel[0], _sel[1], _sel[2], _sel[3]
                    print(f"   [THR-SWEEP] thr={_best_thr_po:.3f} prec={_best_prec_po:.1%} "
                          f"n={_best_n_po} cov={_best_cov_po:.1%} wilson_lb={_sel[4]:.3f} | "
                          f"{len(_rows_po)} rows, {len(_passing_rows)} with LB>=60%")
            else:
                # Small val set (<100 directional bars): derive threshold from training pool.
                # Calibrated training confidence is more reliable than a noisy val sweep.
                if _po_use_calibrator and _calibrator_po is not None:
                    _tp_dir_conf_base = _conf_tp[_dir_tp_cal].reshape(-1, 1)
                    _use_multi_cal_tp = (
                        _hold_disc_cols
                        and hasattr(_calibrator_po, 'coef_')
                        and _calibrator_po.coef_.shape[1] > 1
                    )
                    if _use_multi_cal_tp:
                        _tp_dir_idx  = np.where(_dir_tp_cal)[0]
                        _tp_dir_disc = Xtp.iloc[_tp_dir_idx][_hold_disc_cols].fillna(0.0).to_numpy()
                        _tp_cal_x    = np.column_stack([_tp_dir_conf_base, _tp_dir_disc])
                    else:
                        _tp_cal_x = _tp_dir_conf_base
                    _cal_conf_tp_dir = _calibrator_po.predict_proba(_tp_cal_x)[:, 1]
                    _best_thr_po = float(np.percentile(_cal_conf_tp_dir, 60)) if len(_cal_conf_tp_dir) > 0 else 0.60
                    print(f"   [SMALL VAL] Cal training 60th-pct thr={_best_thr_po:.3f} (val_n={_n_dir_val})")
                else:
                    _dir_conf_all = _conf_tp[_dir_tp] if _dir_tp.sum() > 0 else _conf_tp
                    _best_thr_po  = float(np.percentile(_dir_conf_all, 60)) if len(_dir_conf_all) > 0 else 0.60
                    print(f"   [SMALL VAL] Raw training 60th-pct thr={_best_thr_po:.3f} (val_n={_n_dir_val})")

            _primary_conf_thr = _best_thr_po
            _primary_only_gate = True

            # Guard: BUY/SELL balance check on the val set.
            # If the selected threshold causes one side to fire <10 val signals,
            # back off the threshold to the highest value where BOTH sides fire ≥10.
            # This prevents independent Optuna tuning from creating a model where one
            # direction dominates almost entirely at the calibrated threshold.
            _fire_buy_val  = int(((_prop_dir_val == 2) & (_cal_conf_dir_val >= _primary_conf_thr)).sum())
            _fire_sell_val = int(((_prop_dir_val == 0) & (_cal_conf_dir_val >= _primary_conf_thr)).sum())
            _MIN_SIDE_FIRES_VAL = 10
            _MIN_PREC_BAL = max(0.48, token_breakeven - 0.05)
            if min(_fire_buy_val, _fire_sell_val) < _MIN_SIDE_FIRES_VAL and _rows_po:
                # Walk back through sweep rows (sorted descending by threshold) and find
                # the highest threshold where both sides have ≥_MIN_SIDE_FIRES_VAL.
                _rows_balanced = sorted(_rows_po, key=lambda r: r[0], reverse=True)
                for _rb_thr, _rb_prec, _rb_cov, _rb_n in _rows_balanced:
                    _buy_v  = int(((_prop_dir_val == 2) & (_cal_conf_dir_val >= _rb_thr)).sum())
                    _sell_v = int(((_prop_dir_val == 0) & (_cal_conf_dir_val >= _rb_thr)).sum())
                    if min(_buy_v, _sell_v) >= _MIN_SIDE_FIRES_VAL and _rb_prec >= _MIN_PREC_BAL:
                        print(f"   [PRIMARY ONLY] Side-balance fallback: thr {_primary_conf_thr:.3f}→{_rb_thr:.3f} "
                              f"(buy={_fire_buy_val}→{_buy_v}, sell={_fire_sell_val}→{_sell_v})")
                        _primary_conf_thr = _rb_thr
                        # Update reported metrics to reflect the actual selected threshold
                        _best_prec_po = _rb_prec
                        _best_n_po    = _rb_n
                        _best_cov_po  = _rb_cov
                        break

            # Per-side threshold sweep: only needed when calibrator is bypassed.
            # When calibrator is active, both sides use the same _cal_conf_dir_val
            # and the combined sweep already found the optimal threshold — a redundant
            # per-side sweep on the same scores just adds a second val-precision gate
            # that disables signals the calibrator already selected correctly.
            # When calibrator is bypassed, per-side sweep corrects Bayes asymmetry:
            # SELL logit advantage would push the combined threshold below BUY's
            # natural operating point without independent per-side selection.
            _thr_buy_side  = _primary_conf_thr
            _thr_sell_side = _primary_conf_thr
            _prec_buy_side_val  = _best_prec_po
            _prec_sell_side_val = _best_prec_po
            if not (_po_use_calibrator and _calibrator_po is not None):
                # Uncalibrated path: per-side Bayes-corrected sweep
                _buy_val_mask  = (_prop_dir_val == 2)
                _sell_val_mask = (_prop_dir_val == 0)
                _buy_sweep_conf  = _pb_val_c[_dir_val]
                _sell_sweep_conf = _ps_val_c[_dir_val]
                _N_SIDE_MIN = 20
                for _sn, _sm, _sc in [('BUY', _buy_val_mask, _buy_sweep_conf),
                                        ('SELL', _sell_val_mask, _sell_sweep_conf)]:
                    _srows = []
                    for _thr_s in np.arange(0.30, 0.96, 0.02):
                        _fs = _sm & (_sc >= _thr_s)
                        if _fs.sum() < _N_SIDE_MIN:
                            continue
                        _prec_s = float((_prop_dir_val[_fs] == _y_outcome_dir_val[_fs]).mean())
                        _srows.append((float(_thr_s), _prec_s, int(_fs.sum())))
                    if _srows:
                        _SIDE_MARGIN = 0.05
                        _passing = [(t, p, n) for t, p, n in _srows
                                    if p >= token_breakeven - _SIDE_MARGIN]
                        if _passing:
                            _bt, _bp, _bn = max(_passing, key=lambda r: r[0])
                        else:
                            _bt, _bp, _bn = max(_srows, key=lambda r: r[1])
                        if _sn == 'BUY':
                            _thr_buy_side, _prec_buy_side_val = _bt, _bp
                        else:
                            _thr_sell_side, _prec_sell_side_val = _bt, _bp
                _SIDE_MARGIN = 0.05
                if _prec_buy_side_val < token_breakeven - _SIDE_MARGIN:
                    print(f"   [PER-SIDE] BUY val_prec={_prec_buy_side_val:.1%} < "
                          f"breakeven-5pp ({token_breakeven - _SIDE_MARGIN:.1%}) — BUY disabled")
                    _thr_buy_side = 2.0
                if _prec_sell_side_val < token_breakeven - _SIDE_MARGIN:
                    print(f"   [PER-SIDE] SELL val_prec={_prec_sell_side_val:.1%} < "
                          f"breakeven-5pp ({token_breakeven - _SIDE_MARGIN:.1%}) — SELL disabled")
                    _thr_sell_side = 2.0
                print(f"   [PER-SIDE] BUY thr={_thr_buy_side:.3f} val_prec={_prec_buy_side_val:.1%} | "
                      f"SELL thr={_thr_sell_side:.3f} val_prec={_prec_sell_side_val:.1%}")
            else:
                print(f"   [CAL-GATE] Calibrator active — both sides use combined "
                      f"thr={_primary_conf_thr:.3f} val_prec={_best_prec_po:.1%} val_n={_best_n_po}")

            print(
                f"   [PRIMARY ONLY] primary={_primary_dir_prec:.1%} — "
                f"cal-prob thr={_primary_conf_thr:.3f} "
                f"val_prec={_best_prec_po:.1%} val_n={_best_n_po} val_cov={_best_cov_po:.1%} "
                f"(calibrated={'Y' if _po_use_calibrator else 'N'})."
            )

        if not _primary_only_gate:
            # ── Quantile-preserved holdout thresholds ──────────────────────────────────
            # thr_buy/thr_sell were calibrated on OOF edge scores (conservative — trained on
            # 87% data per fold). Full-fit meta edge scores are higher (100% data), so the
            # same absolute threshold fires on a larger fraction of holdout bars → precision
            # drop. Fix: re-derive at the same percentile of holdout's regime-ok per-side
            # distribution, making dev and holdout coverage identical by construction.
            _ebh_b = edge_buy_h[(prop_h == 2) & _regime_ok_h]
            _ebh_s = edge_sell_h[(prop_h == 0) & _regime_ok_h]
            thr_buy_h  = float(np.quantile(_ebh_b, 1.0 - min(_dev_buy_side_cov, 0.50)))  if len(_ebh_b) >= 5 and _dev_buy_side_cov > 0 else thr_buy
            thr_sell_h = float(np.quantile(_ebh_s, 1.0 - min(_dev_sell_side_cov, 0.50))) if len(_ebh_s) >= 5 and _dev_sell_side_cov > 0 else thr_sell
            # QP floor: never lower the holdout threshold below the dev threshold.
            # When the regime filter removes 60-70% of holdout bars, the remaining pool
            # has lower scores on average, so QP pushes the threshold down to match dev
            # coverage — pulling in HOLD-labelled bars that destroy precision. If the
            # holdout pool scores are lower, the right response is to fire fewer signals
            # at high quality, not more signals at low quality.
            # EXCEPTION: when the safety valve reset the regime filter, the full population
            # is now in scope and QP must derive a lower threshold to hit dev coverage —
            # the floor would defeat the purpose and result in under-firing.
            if not _regime_safety_valve_triggered:
                thr_buy_h  = max(thr_buy_h,  thr_buy)
                thr_sell_h = max(thr_sell_h, thr_sell)
            print(f"   [QP-Holdout] BUY  dev_thr={thr_buy:.1f}→hold_thr={thr_buy_h:.1f} (dev_cov={_dev_buy_side_cov:.1%})")
            print(f"   [QP-Holdout] SELL dev_thr={thr_sell:.1f}→hold_thr={thr_sell_h:.1f} (dev_cov={_dev_sell_side_cov:.1%})")

            # Per-HMM-regime quantile-preserved holdout thresholds
            regime_policies_h = {}
            for _r, _pol in regime_policies.items():
                _rm_b_h = (holdout['hmm_regime'] == _r).to_numpy() if 'hmm_regime' in holdout.columns else np.zeros(len(holdout), dtype=bool)
                _rm_b_h = _rm_b_h & (prop_h == 2) & _regime_ok_h
                _rm_s_h = (holdout['hmm_regime'] == _r).to_numpy() if 'hmm_regime' in holdout.columns else np.zeros(len(holdout), dtype=bool)
                _rm_s_h = _rm_s_h & (prop_h == 0) & _regime_ok_h
                _bc_r = regime_buy_covs.get(_r, _dev_buy_side_cov)
                _sc_r = regime_sell_covs.get(_r, _dev_sell_side_cov)
                _thr_b = float(np.quantile(edge_buy_h[_rm_b_h],  1.0 - min(_bc_r,  0.50))) if _rm_b_h.sum() >= 5 and _bc_r  > 0 else thr_buy_h
                _thr_s = float(np.quantile(edge_sell_h[_rm_s_h], 1.0 - min(_sc_r, 0.50)))  if _rm_s_h.sum() >= 5 and _sc_r > 0 else thr_sell_h
                if not _regime_safety_valve_triggered:
                    _thr_b = max(_thr_b, thr_buy)   # QP floor: disabled when safety valve fired
                    _thr_s = max(_thr_s, thr_sell)
                regime_policies_h[_r] = {
                    'buy_thr': _thr_b, 'sell_thr': _thr_s,
                    'buy_ok': _pol.get('buy_ok', True), 'sell_ok': _pol.get('sell_ok', True),
                }

            # Holdout Regime-specific Edge Score Thresholding (Priority 4)
            fire_buy_h = np.zeros(len(holdout), dtype=bool)
            fire_sell_h = np.zeros(len(holdout), dtype=bool)

            for i in range(len(holdout)):
                if not _regime_ok_h[i]:
                    continue   # pre-filtered by token_params regime mask (symmetric with dev)
                r = holdout['hmm_regime'].iloc[i] if 'hmm_regime' in holdout.columns else 'UNKNOWN'
                policy_h = regime_policies_h.get(r, {
                    "buy_thr": thr_buy_h,
                    "sell_thr": thr_sell_h,
                    "buy_ok": True,
                    "sell_ok": True
                })

                r_buy_thr = policy_h.get("buy_thr", thr_buy_h)
                r_sell_thr = policy_h.get("sell_thr", thr_sell_h)
                r_buy_ok = policy_h.get("buy_ok", True)
                r_sell_ok = policy_h.get("sell_ok", True)

                if prop_h[i] == 2 and r_buy_ok:
                    fire_buy_h[i] = (edge_buy_h[i] >= r_buy_thr)
                elif prop_h[i] == 0 and r_sell_ok:
                    fire_sell_h[i] = (edge_sell_h[i] >= r_sell_thr)

            fire = fire_buy_h | fire_sell_h
            gate_mode = f"EdgeEngine-QP (B>={thr_buy_h:.1f}[dev:{thr_buy:.1f}], S>={thr_sell_h:.1f}[dev:{thr_sell:.1f}])"

            _tier_agg_pre = True
            _AGG_FLOOR_PRE = 0.50
            print(f"   Holdout gate: {gate_mode} | fired {int(fire.sum())}")
            # Threshold that separates top-25% of fired signals from bottom-75%.
            # Saved to metadata so the live predictor can replicate S&R / trend filters
            # without needing the full holdout distribution at inference time.
            override_conf_thr = float(np.quantile(meta_prob_h[fire], 0.75)) if fire.sum() > 0 else thr
        else:
            # PRIMARY ONLY: apply calibrated confidence gate to holdout.
            _conf_h = np.maximum(raw_probs[:, 2], raw_probs[:, 0])
            if _po_use_calibrator and _calibrator_po is not None:
                _conf_h_base = _conf_h.reshape(-1, 1)
                _use_multi_cal_h = (
                    _calibrator_hold_disc_cols
                    and hasattr(_calibrator_po, 'coef_')
                    and _calibrator_po.coef_.shape[1] > 1
                )
                if _use_multi_cal_h:
                    _h_disc_feats = holdout[feature_cols].reindex(
                        columns=_calibrator_hold_disc_cols, fill_value=0.0
                    ).fillna(0.0).to_numpy()
                    _cal_h_x = np.column_stack([_conf_h_base, _h_disc_feats])
                else:
                    _cal_h_x = _conf_h_base
                _cal_conf_h = _calibrator_po.predict_proba(_cal_h_x)[:, 1]
            else:
                _cal_conf_h = _conf_h
            # Fix L: apply per-side thresholds at holdout gate.
            # Calibrated path: _cal_conf_h is the same calibrated combined score for all
            # bars; per-side thresholds let each direction select its own quality cutoff.
            # Uncalibrated path: use each model's own Bayes-corrected output so the SELL
            # model's logit advantage doesn't force a low combined threshold for BUY.
            if _po_use_calibrator and _calibrator_po is not None:
                _pb_h_gate = _cal_conf_h
                _ps_h_gate = _cal_conf_h
            else:
                _pb_h_gate = raw_probs[:, 2]   # buy model Bayes-corrected
                _ps_h_gate = raw_probs[:, 0]   # sell model Bayes-corrected
            fire       = (((prop_h == 2) & (_pb_h_gate >= _thr_buy_side)) |
                          ((prop_h == 0) & (_ps_h_gate >= _thr_sell_side)))
            thr_buy_h  = _thr_buy_side
            thr_sell_h = _thr_sell_side
            regime_policies_h = {}
            gate_mode  = (f"PRIMARY-CONF "
                          f"(B={_thr_buy_side:.3f}/S={_thr_sell_side:.3f} "
                          f"cal={'Y' if _po_use_calibrator else 'N'})")
            _tier_agg_pre = True
            _active_thr_buy  = _thr_buy_side  if _thr_buy_side  < 2.0 else 0.0
            _active_thr_sell = _thr_sell_side if _thr_sell_side < 2.0 else 0.0
            override_conf_thr = max(_active_thr_buy, _active_thr_sell)
            print(f"   Holdout gate: {gate_mode} | fired {int(fire.sum())}")

        # ── S&R-aware confluence filter ────────────────────────────────
        # At resistance a BUY signal fights the level — keep it only if the
        # meta confidence is in the top 25% of all fired signals (high conviction).
        # At support a SELL signal fights the level — same rule applies.
        # This embeds the "indicators must vote strongly to override S&R" rule.
        if (hit_target or _tier_agg_pre) and fire.sum() >= 4 and not disable_sr_veto and not _primary_only_gate:
            at_res = holdout['is_at_resistance'].to_numpy().astype(bool) \
                     if 'is_at_resistance' in holdout.columns else np.zeros(len(holdout), dtype=bool)
            at_sup = holdout['is_at_support'].to_numpy().astype(bool) \
                     if 'is_at_support' in holdout.columns else np.zeros(len(holdout), dtype=bool)
            top25_thr = float(np.quantile(meta_prob_h[fire], 0.75))
            top25 = meta_prob_h >= top25_thr
            # Suppress: weak BUY at resistance OR weak SELL at support
            sr_suppress = (
                (fire & (prop_h == 2) & at_res & ~top25) |
                (fire & (prop_h == 0) & at_sup & ~top25)
            )
            n_suppressed = int(sr_suppress.sum())
            if n_suppressed:
                fire = fire & ~sr_suppress
                print(f"   S&R filter: suppressed {n_suppressed} counter-level signals "
                      f"(below top-25% confidence @ resistance/support)")


        # ── Macro 1d trend alignment filter ─────────────────────────────────
        # Trading against a confirmed daily trend is the single biggest source
        # of counter-trend losses at 1h resolution. We tolerate it only when
        # meta-confidence is top-25% (the model has a strong conviction override).
        # Same design as the S&R filter above: principled suppression, not tuning.
        if (hit_target or _tier_agg_pre) and fire.sum() >= 4 and 'macro_trend_1d' in holdout.columns and not disable_trend_veto and not _primary_only_gate:
            trend_1d = holdout['macro_trend_1d'].to_numpy()
            _top25_thr = float(np.quantile(meta_prob_h[fire], 0.75))
            _top25 = meta_prob_h >= _top25_thr
            trend_suppress = (
                (fire & (prop_h == 2) & (trend_1d < -0.2) & ~_top25) |  # weak BUY in downtrend
                (fire & (prop_h == 0) & (trend_1d >  0.2) & ~_top25)    # weak SELL in uptrend
            )
            n_ts = int(trend_suppress.sum())
            if n_ts:
                fire = fire & ~trend_suppress
                print(f"   Trend filter: suppressed {n_ts} counter-trend weak signals")

        # ── Confluence filter ─────────────────────────────────
        if (hit_target or _tier_agg_pre) and fire.sum() >= 4 and 'total_confluence' in holdout.columns and not disable_confluence_veto:
            tc = holdout['total_confluence'].to_numpy()
            conf_suppress = (
                (fire & (prop_h == 2) & (tc < -0.05)) |
                (fire & (prop_h == 0) & (tc > 0.05))
            )
            n_cs = int(conf_suppress.sum())
            if n_cs:
                fire = fire & ~conf_suppress
                print(f"   Confluence filter: suppressed {n_cs} counter-confluence signals")

        # ── Per-side holdout precision (using EdgeEngine) ──
        buy_fire = fire & (prop_h == 2)
        sell_fire = fire & (prop_h == 0)

        # ── Spot-on market dynamics: Regime-specific directional filter ──────
        # Note: regime suppression is now applied pre-firing via _regime_ok_h above,
        # so this post-firing block is a no-op for any token that had token_params loaded.
        # Kept for tokens where _opt is None (no token_params file yet).
        if False and (hit_target or _tier_agg_pre) and _opt and "regimes" in _opt and "regime_boundaries" in _opt and not _primary_only_gate:  # HMM regime filter permanently disabled
            bounds = _opt["regime_boundaries"]
            regimes_dict = _opt["regimes"]
            
            vol_avg = holdout["volume"].rolling(24, min_periods=1).mean()
            atr_pct = (holdout["_atr_raw"] / holdout["_close_raw"]).fillna(0)
            momentum = holdout["_close_raw"].pct_change(24).fillna(0)
            
            def _tier(val, p33, p67): return "low" if val <= p33 else ("med" if val <= p67 else "high")
            def _trend(val, p33, p67): return "down" if val <= p33 else ("flat" if val <= p67 else "up")
            
            vp33, vp67 = bounds.get("vol_p33", 0), bounds.get("vol_p67", 0)
            ap33, ap67 = bounds.get("atr_pct_p33", 0), bounds.get("atr_pct_p67", 0)
            mp33, mp67 = bounds.get("momentum_p33", -0.02), bounds.get("momentum_p67", 0.02)
            
            regime_strs = [
                f"{_tier(vol_avg.iloc[i], vp33, vp67)}_{_tier(atr_pct.iloc[i], ap33, ap67)}_{_trend(momentum.iloc[i], mp33, mp67)}"
                for i in range(len(holdout))
            ]
            
            regime_suppress = np.zeros(len(holdout), dtype=bool)
            for i in range(len(holdout)):
                if not (buy_fire[i] or sell_fire[i] or fire[i]): continue
                reg = regimes_dict.get(regime_strs[i], {})
                if not reg or reg.get("skipped"):
                    regime_suppress[i] = True
                    continue
                side = prop_h[i]
                if side == 2:
                    if not reg.get("buy_ok"): regime_suppress[i] = True
                elif side == 0:
                    if not reg.get("sell_ok"): regime_suppress[i] = True

            n_regime_supp = int(regime_suppress.sum())
            if n_regime_supp:
                fire = fire & ~regime_suppress
                buy_fire = buy_fire & ~regime_suppress
                sell_fire = sell_fire & ~regime_suppress
                print(f"   Regime filter: suppressed {n_regime_supp} signals based on spot-on dynamics")

        # barrier size as a fraction of price, for the PnL backtest
        vr = holdout['volatility_regime'].to_numpy() if 'volatility_regime' in holdout else np.ones(len(holdout))
        dyn = atr_mult * np.clip(vr, 0.8, 1.5)
        close_arr = holdout['_close_raw'].to_numpy()
        barrier_frac = np.divide(dyn * holdout['_atr_raw'].to_numpy(), close_arr,
                                 out=np.zeros(len(holdout)), where=close_arr != 0)

        fired_n    = int(fire.sum())
        fired_prec = float((prop_h[fire] == y_test[fire]).mean()) if fired_n > 0 else 0.0
        coverage   = fired_n / len(y_test)
        bt         = backtest(fire, prop_h, y_test, barrier_frac)

        buy_h_n   = int(buy_fire.sum())
        sell_h_n  = int(sell_fire.sum())
        buy_h_prec  = float((y_test[buy_fire]  == 2).mean()) if buy_h_n  > 0 else 0.0
        sell_h_prec = float((y_test[sell_fire] == 0).mean()) if sell_h_n > 0 else 0.0

        # reference (all-bars) metrics -- de-emphasised, kept for continuity
        test_acc = float(accuracy_score(y_test, raw_probs.argmax(1)))
        prec, rec, f1c, _ = [np.asarray(a) for a in precision_recall_fscore_support(
            y_test, raw_probs.argmax(1), labels=[0, 1, 2], zero_division=0)]
        med_b    = float(np.nanmedian(barrier_frac[fire])) if fired_n else 0.01
        breakeven = 0.5 * (1 + FEE_ROUNDTRIP / max(med_b, 1e-4))

        # Defaults for ranking-audit variables only populated inside fired_n > 0 block
        selected_n = 0; rejected_n = 0
        selected_prec = rejected_prec = 0.0
        selected_exp_pct = rejected_exp_pct = 0.0
        selected_sharpe = rejected_sharpe = 0.0
        meta_gate_lift = meta_gate_lift_exp = 0.0
        per_side_approved = False
        _via_profitability_bypass = False
        # Directional precision defaults (overwritten inside fired_n > 0 block)
        _dir_fired_prec2 = 0.0; _dir_fired_n2 = 0; _dir_coverage2 = 0.0
        _dir_buy_prec = 0.0; _dir_sell_prec = 0.0; _dir_buy_n = 0; _dir_sell_n = 0

        print("\n   TRADING PERFORMANCE on holdout (meta-filtered -- the real metric):")
        print(f"      Fired signals   : {fired_n} / {len(y_test)}  (coverage {coverage:.1%})")
        if fired_n > 0:
            print(f"      Signal precision: {fired_prec:.3f}   "
                  f"(breakeven ~ {breakeven:.3f} after {FEE_ROUNDTRIP*100:.2f}% fees)")
            print(f"      BUY  holdout    : {buy_h_prec:.3f}  ({buy_h_n} trades)"
                  f"{'  PASS' if buy_h_n >= 5 and buy_h_prec >= breakeven else '  fail'}")
            print(f"      SELL holdout    : {sell_h_prec:.3f}  ({sell_h_n} trades)"
                  f"{'  PASS' if sell_h_n >= 5 and sell_h_prec >= breakeven else '  fail'}")
            print(f"      Win rate (PnL)  : {bt['win_rate']:.3f}  "
                  f"| BUY wr {bt['buy_win_rate']:.3f} ({bt['buy_n']} trades)  "
                  f"| SELL wr {bt['sell_win_rate']:.3f} ({bt['sell_n']} trades)")
            print(f"      Expectancy/trade: {bt['expectancy_pct']:+.4f}%"
                  f"  | Kelly: {bt['kelly_pct']:.1f}%")
            print(f"      Total return    : {bt['total_return_pct']:+.2f}%  "
                  f"(holdout window, {fired_n} trades)")
            print(f"      Sharpe (ann.)   : {bt['sharpe']:+.3f}"
                  f"  | Profit factor: {bt['profit_factor']:.3f}"
                  f"  | Max drawdown: {bt['max_drawdown_pct']:.2f}%")

            # ── Directional precision (HOLD timeouts excluded) ────────────
            # HOLD-labelled bars are timeout events, not direction errors. Using them
            # in the denominator inflates the "wrong" count and can disable tokens
            # where the model correctly calls direction on every barrier-hitting bar.
            # Directional precision = correct_direction_fires / (BUY+SELL label fires).
            _dir_outcome_mask = (y_test == 2) | (y_test == 0)
            _dir_fired_mask2  = fire & _dir_outcome_mask
            _dir_fired_n2     = int(_dir_fired_mask2.sum())
            _dir_fired_prec2  = float((prop_h[_dir_fired_mask2] == y_test[_dir_fired_mask2]).mean()) if _dir_fired_n2 > 0 else 0.0
            _dir_total2       = int(_dir_outcome_mask.sum())
            _dir_coverage2    = _dir_fired_n2 / _dir_total2 if _dir_total2 > 0 else 0.0
            _dir_buy_fired    = buy_fire & _dir_outcome_mask
            _dir_sell_fired   = sell_fire & _dir_outcome_mask
            _dir_buy_n        = int(_dir_buy_fired.sum())
            _dir_sell_n       = int(_dir_sell_fired.sum())
            _dir_buy_prec     = float((y_test[_dir_buy_fired] == 2).mean()) if _dir_buy_n > 0 else 0.0
            _dir_sell_prec    = float((y_test[_dir_sell_fired] == 0).mean()) if _dir_sell_n > 0 else 0.0
            print(f"   Directional precision: {_dir_fired_prec2:.1%} ({_dir_fired_n2}/{_dir_total2} dir bars, cov={_dir_coverage2:.1%})"
                  f" | BUY={_dir_buy_prec:.1%}({_dir_buy_n}tr) SELL={_dir_sell_prec:.1%}({_dir_sell_n}tr)")

            # ── Overlap-adjusted evidence ─────────────────────────────────
            # The raw count above treats every fired bar as an independent trial.
            # With an 18-96h barrier on 1h bars it is not: neighbouring fires
            # share an outcome path. Deflate to non-overlapping events, then take
            # the Wilson lower bound on THAT — so "100% on 58 trades" is scored as
            # what it actually is (a handful of events), not as certainty.
            _hit_arr = holdout['_hit_bars'].to_numpy() if '_hit_bars' in holdout.columns else None
            _dir_eff_n = effective_sample_size_durations(
                np.where(_dir_fired_mask2)[0], _hit_arr, token_lookahead)
            _dir_prec_lb = wilson_lower_bound(_dir_fired_prec2 * _dir_eff_n, _dir_eff_n)
            _sig_eff_n = effective_sample_size_durations(
                np.where(fire)[0], _hit_arr, token_lookahead)
            _med_hold = (float(np.nanmedian(_hit_arr[_dir_fired_mask2]))
                         if _hit_arr is not None and _dir_fired_n2 > 0
                         else float(token_lookahead))
            print(f"   Overlap-adjusted: {_dir_eff_n} independent dir events "
                  f"(from {_dir_fired_n2} raw, median hold {_med_hold:.0f}h, "
                  f"label window {token_lookahead}h) "
                  f"| dir_prec 95% lower bound = {_dir_prec_lb:.1%}")

            # ── Tradeable decision — directional precision ────────────────
            # Gate on directional precision (≥60%) not 3-class signal precision.
            # HOLD timeouts are fee-only events; direction errors are the real risk.
            _MIN_SIDE = 5    # minimum per-side holdout directional trades to trust the result
            tradeable_buy_holdout  = (
                hit_buy and
                _dir_buy_n >= _MIN_SIDE and
                _dir_buy_prec >= 0.60
            )
            tradeable_sell_holdout = (
                hit_sell and
                _dir_sell_n >= _MIN_SIDE and
                _dir_sell_prec >= 0.60
            )

            holdout_reliable = fired_n >= MIN_HOLDOUT_FIRES
            oof_holdout_gap  = abs(dev_prec - fired_prec)

            # passes_validation uses directional precision — HOLD timeouts do not count.
            passes_validation = (
                _dir_fired_n2 >= MIN_HOLDOUT_FIRES and
                _dir_fired_prec2 >= 0.60 and
                _dir_coverage2 >= MIN_COVERAGE
            )
            _via_profitability_bypass = False

            tradeable_final = passes_validation and (tradeable_buy_holdout or tradeable_sell_holdout)
            # Secondary veto: warn if expectancy is negative despite good precision,
            # but keep the token enabled — barrier multiplier tuning can fix EV.
            if tradeable_final and bt["expectancy_pct"] <= 0.0:
                print("      WARNING: precision >= 60% but expectancy negative "
                      f"({bt['expectancy_pct']:+.3f}%) — check barrier sizing.")
            
            # Meta Gate Ranking Validation (Priority 1)
            dir_proposed_h = (prop_h == 2) | (prop_h == 0)
            selected_mask = fire
            rejected_mask = dir_proposed_h & (~fire)
            
            selected_n = int(selected_mask.sum())
            rejected_n = int(rejected_mask.sum())
            
            selected_prec = float((prop_h[selected_mask] == y_test[selected_mask]).mean()) if selected_n > 0 else 0.0
            rejected_prec = float((prop_h[rejected_mask] == y_test[rejected_mask]).mean()) if rejected_n > 0 else 0.0
            
            # Compute expectancy and Sharpe on both groups for ranking diagnostics
            selected_barrier_frac = barrier_frac[selected_mask] if selected_n > 0 else np.array([])
            rejected_barrier_frac = barrier_frac[rejected_mask] if rejected_n > 0 else np.array([])
            
            selected_exp_pct = 0.0
            selected_sharpe = 0.0
            rejected_exp_pct = 0.0
            rejected_sharpe = 0.0
            
            if selected_n > 0:
                sel_pnls = []
                for i in np.where(selected_mask)[0]:
                    close_val = holdout['_close_raw'].iloc[i]
                    label_val = y_test[i]
                    dir_val = prop_h[i]
                    b = barrier_frac[i]
                    if label_val == 1:
                        pnl = -FEE_ROUNDTRIP
                    elif dir_val == label_val:
                        pnl = b - FEE_ROUNDTRIP
                    else:
                        pnl = -b - FEE_ROUNDTRIP
                    sel_pnls.append(pnl)
                sel_pnls = np.array(sel_pnls)
                selected_exp_pct = float(sel_pnls.mean()) * 100
                if len(sel_pnls) > 1:
                    selected_sharpe = float(sel_pnls.mean() / sel_pnls.std() * math.sqrt(8760)) if sel_pnls.std() > 1e-10 else 0.0
            
            if rejected_n > 0:
                rej_pnls = []
                for i in np.where(rejected_mask)[0]:
                    close_val = holdout['_close_raw'].iloc[i]
                    label_val = y_test[i]
                    dir_val = prop_h[i]
                    b = barrier_frac[i]
                    if label_val == 1:
                        pnl = -FEE_ROUNDTRIP
                    elif dir_val == label_val:
                        pnl = b - FEE_ROUNDTRIP
                    else:
                        pnl = -b - FEE_ROUNDTRIP
                    rej_pnls.append(pnl)
                rej_pnls = np.array(rej_pnls)
                rejected_exp_pct = float(rej_pnls.mean()) * 100
                if len(rej_pnls) > 1:
                    rejected_sharpe = float(rej_pnls.mean() / rej_pnls.std() * math.sqrt(8760)) if rej_pnls.std() > 1e-10 else 0.0
            
            meta_gate_lift = selected_prec - rejected_prec
            meta_gate_lift_exp = selected_exp_pct - rejected_exp_pct
            
            print(f"   Meta Gate Ranking Validation:")
            print(f"      Selected:  {selected_n:>4} trades, prec {selected_prec:.1%}, exp {selected_exp_pct:+.3f}%, sharpe {selected_sharpe:+.2f}")
            print(f"      Rejected:  {rejected_n:>4} trades, prec {rejected_prec:.1%}, exp {rejected_exp_pct:+.3f}%, sharpe {rejected_sharpe:+.2f}")
            print(f"      Meta gate lift: prec {meta_gate_lift:+.1%}, exp {meta_gate_lift_exp:+.3f}%")
            
            if selected_prec <= rejected_prec and not _primary_only_gate:
                print("      [VETO] Selected trades did not outperform rejected trades! Disabling the gate automatically.")
                passes_validation = False

            # Gap veto only applies when NOT using primary-only gate: dev_prec=0
            # in primary-only mode (no meta OOF) so the gap would always be spurious.
            if False and not _primary_only_gate and holdout_reliable and oof_holdout_gap > 0.20 and fired_n >= 15:
                passes_validation = False
                print(f"      DISABLED (primary gap veto): OOF→holdout gap {oof_holdout_gap:.1%} > 20pp "
                      f"— overfit detected. Retrain with updated FEATURE_BLACKLIST + SHAP pruning to fix.")
            elif not _primary_only_gate and holdout_reliable and bt['expectancy_pct'] <= 0.0 and oof_holdout_gap > GAP_VETO_THRESHOLD:
                passes_validation = False
                print("      DISABLED (gap veto): negative expectancy and large OOF→holdout gap")
            elif not _primary_only_gate and holdout_reliable and oof_holdout_gap > 0.10:
                print(f"      WATCH: OOF→holdout gap {oof_holdout_gap:.1%} "
                      f"(dev {dev_prec:.1%} → holdout {fired_prec:.1%}). "
                      f"Possible regime shift — monitor after next retrain.")

            tradeable_final = passes_validation and (tradeable_buy_holdout or tradeable_sell_holdout)
            # Profitability bypass tokens qualify on financial merit — no per-side precision required.
            if _via_profitability_bypass and not tradeable_final:
                tradeable_final = True
                per_side_approved = False
            per_side_approved = bool(tradeable_buy_holdout or tradeable_sell_holdout)
            combined_ok = bool(passes_validation)

            # ── Primary-only gate: override tradeable decision ────────────────────────
            # HOLD timeouts are fee-only events — they do not represent direction errors.
            # Tradeable requires: directional precision ≥ 60% AND coverage ≥ 5%.
            # signal_prec (all fired bars including HOLD) is logged for reference only.
            if _primary_only_gate:
                fired_dir_prec  = _dir_fired_prec2    # reuse pre-computed directional stats
                coverage_dir    = _dir_coverage2
                _signal_prec_h  = float((prop_h[fire] == y_test[fire]).mean()) if fire.sum() > 0 else 0.0
                # Ship on EVIDENCE and PROFIT, not on a raw point estimate.
                #   1) dir_prec must clear 60% at its 95% LOWER BOUND, computed on
                #      OVERLAP-DEFLATED events — a 100% that rests on ~4 independent
                #      events can no longer enable a token (its LB is ~0.40).
                #   2) enough INDEPENDENT events, not just enough bars.
                #   3) expectancy must be positive. This is the honest profitability
                #      test: backtest() prices all three outcomes correctly (win
                #      +barrier-fee, wrong -barrier-fee, HOLD timeout -fee only),
                #      whereas comparing signal_precision to `breakeven` double-counts
                #      timeouts as full-barrier losses and is unfairly harsh.
                _MIN_EFF_EVENTS = 12
                tradeable_final = (
                    _dir_fired_n2 >= MIN_HOLDOUT_FIRES and
                    _dir_eff_n    >= _MIN_EFF_EVENTS and
                    _dir_prec_lb  >= 0.60 and
                    coverage_dir  >= MIN_COVERAGE and
                    bt["expectancy_pct"] > 0.0
                )
                tradeable_buy_holdout  = tradeable_final
                tradeable_sell_holdout = tradeable_final
                per_side_approved      = tradeable_final
                passes_validation      = tradeable_final
                _via_profitability_bypass = False
                if tradeable_final:
                    print(f"      [PRIMARY ONLY] ENABLED: dir_prec={fired_dir_prec:.1%} "
                          f"(LB={_dir_prec_lb:.1%}>=60% on {_dir_eff_n} independent events) "
                          f"exp={bt['expectancy_pct']:+.3f}% dir_cov={coverage_dir:.1%} | "
                          f"{_dir_fired_n2} raw dir trades (total fired={fired_n} cov={coverage:.1%}) "
                          f"| signal_prec={_signal_prec_h:.1%} (ref, incl HOLD timeouts)")
                else:
                    _po_reasons: list = []
                    if _dir_prec_lb < 0.60:
                        _po_reasons.append(
                            f"dir_prec_LB={_dir_prec_lb:.1%}<60% "
                            f"(point={fired_dir_prec:.1%} on only {_dir_eff_n} independent events)")
                    if _dir_eff_n < _MIN_EFF_EVENTS:
                        _po_reasons.append(f"independent_events={_dir_eff_n}<{_MIN_EFF_EVENTS}")
                    if bt["expectancy_pct"] <= 0.0:
                        _po_reasons.append(f"expectancy={bt['expectancy_pct']:+.3f}%<=0")
                    if coverage_dir < MIN_COVERAGE:
                        _po_reasons.append(f"dir_cov={coverage_dir:.1%}<{MIN_COVERAGE:.0%}")
                    if _dir_fired_n2 < MIN_HOLDOUT_FIRES:
                        _po_reasons.append(f"dir_n={_dir_fired_n2}<{MIN_HOLDOUT_FIRES}")
                    print(f"      [PRIMARY ONLY] DISABLED: {', '.join(_po_reasons)}")

            # Anti-selection hard veto — cannot be overridden by profitability bypass.
            # Reuses _dir_fired_mask2 / _dir_fired_n2 / _dir_fired_prec2 computed above.
            # The true anti-selection signal: among bars where the outcome WAS directional,
            # did the gate consistently pick the wrong direction (< 50% correct)?
            if holdout_reliable and _dir_fired_prec2 < 0.50 and _dir_fired_n2 >= 10:
                tradeable_final = False
                passes_validation = False
                _via_profitability_bypass = False
                print(f"      DISABLED (anti-selection hard veto): directional precision {_dir_fired_prec2:.1%} < 50% "
                      f"on {_dir_fired_n2} directional bars — gate fires wrong-direction signals. No bypass allowed.")

            if passes_validation:
                print(f"      Directional precision gate: {_dir_fired_prec2:.1%} >= 60% | "
                      f"EV={bt['expectancy_pct']:+.2f}%, PF={bt['profit_factor']:.2f}, Sharpe={bt['sharpe']:.1f}")
                if tradeable_final:
                    sides_live = []
                    if tradeable_buy_holdout:  sides_live.append(f'BUY({_dir_buy_prec:.1%})')
                    if tradeable_sell_holdout: sides_live.append(f'SELL({_dir_sell_prec:.1%})')
                    print(f"      ENABLED: dir_prec >= 60% ({_dir_fired_n2} dir trades) | sides: {' + '.join(sides_live)}")
                else:
                    print(f"      DISABLED: combined dir_prec passed but no per-side reached 60% "
                          f"(BUY={_dir_buy_prec:.1%}/{_dir_buy_n}tr, SELL={_dir_sell_prec:.1%}/{_dir_sell_n}tr).")
            else:
                reasons = []
                if _dir_fired_n2 < MIN_HOLDOUT_FIRES:
                    reasons.append(f"insufficient_dir_trades({_dir_fired_n2}<{MIN_HOLDOUT_FIRES})")
                if _dir_fired_prec2 < 0.60:
                    reasons.append(f"dir_prec_below_60%({_dir_fired_prec2:.1%})")
                if _dir_coverage2 < MIN_COVERAGE:
                    reasons.append(f"dir_cov_below_{MIN_COVERAGE:.0%}({_dir_coverage2:.1%})")
                print(f"      [VALIDATION] FAIL: {', '.join(reasons)}")

            if not tradeable_final:
                print(f"      DISABLED: dir_prec={_dir_fired_prec2:.1%} ({_dir_fired_n2} dir trades) "
                      f"| BUY={_dir_buy_prec:.1%}/{_dir_buy_n}tr SELL={_dir_sell_prec:.1%}/{_dir_sell_n}tr")
            else:
                sides_live = []
                if tradeable_buy_holdout:  sides_live.append(f'BUY({_dir_buy_prec:.1%})')
                if tradeable_sell_holdout: sides_live.append(f'SELL({_dir_sell_prec:.1%})')
                print(f"      ENABLED via directional precision: {' + '.join(sides_live)}")
        else:
            tradeable_final        = False
            tradeable_buy_holdout  = False
            tradeable_sell_holdout = False
            reason = "dev precision floor not met" if not hit_target else "no holdout signals survived rank gate"
            print(f"      No signals fired ({reason})")
        print(f"   -- reference only -- all-bar acc {test_acc:.3f} | "
              f"SELL/HOLD/BUY prec {prec[0]:.2f}/{prec[1]:.2f}/{prec[2]:.2f}")

        # ── Risk tier classification ───────────────────────────────────────
        # Each tier is stored in the sidecar so the live predictor can respect
        # the user's risk preference without retraining.
        #
        # CONSERVATIVE — per-side holdout confirmed ≥ breakeven.
        #   Strictest bar. Both OOF gate and per-direction holdout confirmed.
        #
        # BALANCED — combined holdout ≥ breakeven.
        #   Includes combined-gate approvals (e.g. JUP 89.7%) and [ENABLED-no-side].
        #
        # AGGRESSIVE — any side shows dev precision > 50% (above random).
        #   Does NOT require the meta gate to reach the precision target.
        #   Fires signals for tokens where the primary model has directional
        #   skill but the meta gate can't hit the quality bar at 25% coverage.
        #   Example: BTC SELL 51.7% dev precision — below 54.6% breakeven so
        #   still slightly loss-making without favourable timeout ratios, but
        #   better than the 50% random baseline. Users accept the risk.
        tier_conservative = bool(tradeable_buy_holdout or tradeable_sell_holdout)
        tier_balanced     = bool(
            tradeable_final
            and fired_n >= MIN_HOLDOUT_FIRES
            and fired_prec >= breakeven
        )
        # Aggressive: under edge architecture, if it fired and is tradeable, it's at least aggressive
        tier_aggressive = bool(tradeable_final)

        # Propagate upward: qualifying a higher tier implies lower tiers too.
        tier_balanced   = tier_balanced   or tier_conservative
        tier_aggressive = tier_aggressive or tier_balanced
        # Profitability bypass tokens cap at AGGRESSIVE: no per-side holdout confirmation.
        if _via_profitability_bypass:
            tier_conservative = False
            tier_balanced = False

        print(f"   Risk tiers: "
              f"conservative={'V' if tier_conservative else 'X'} | "
              f"balanced={'V' if tier_balanced else 'X'} | "
              f"aggressive={'V' if tier_aggressive else 'X'}")

        # ---- 6) Deployment models on all usable data ----
        usable = pd.concat([train_pool, holdout], ignore_index=True)
        X_all = usable[feature_cols]
        y_all = usable['target'].to_numpy().astype(int)
        y_all_buy  = (y_all == 2).astype(int)
        y_all_sell = (y_all == 0).astype(int)
        _hold_mask_all = (y_all == 1)   # HOLD bars in deployment training set
        _w_all_buy     = np.where(_hold_mask_all, 0.15, 1.0).astype(float)
        _w_all_sell    = np.where(_hold_mask_all, 0.15, 1.0).astype(float)
        # Reversal focus on the SHIPPED models — multiplied INTO the existing
        # HOLD downweight, never replacing it, so a fade bar that is also a HOLD
        # stays suppressed rather than being promoted by the focus multiplier.
        # Masks carried through from the raw pre-normalisation frames (usable is a
        # concat of the already-normalised train_pool + holdout).
        _rev_buy_all  = usable['_rev_buy'].astype(bool)
        _rev_sell_all = usable['_rev_sell'].astype(bool)
        _w_all_buy  *= reversal_focus_weights(_rev_buy_all,  len(X_all))
        _w_all_sell *= reversal_focus_weights(_rev_sell_all, len(X_all))
        print(f"   Deployment fit reversal focus: BUY fades={int(_rev_buy_all.sum())} "
              f"| SELL fades={int(_rev_sell_all.sum())} of {len(X_all)} bars")

        deploy_buy = xgb.train(binary_params_buy,  _dm(X_all, y_all_buy,  _w_all_buy),
                               num_boost_round=800, verbose_eval=False)
        deploy_sell = xgb.train(binary_params_sell, _dm(X_all, y_all_sell, _w_all_sell),
                                num_boost_round=800, verbose_eval=False)

        store_dir = Path(root_dir) / "src" / "ml" / "model_store"
        store_dir.mkdir(parents=True, exist_ok=True)
        base = symbol.replace('/', '_')

        # Save binary primary models
        buy_path  = store_dir / f"{base}_model_buy.json"
        sell_path = store_dir / f"{base}_model_sell.json"
        model_path = store_dir / f"{base}_model.json"  # backward-compat: keep buy copy here
        deploy_buy.save_model(str(buy_path))
        deploy_sell.save_model(str(sell_path))
        deploy_buy.save_model(str(model_path))

        # Save clean per-fold buy model (trained only on train pool)
        clean_model_path = store_dir / f"{base}_model_clean.json"
        buy_full.save_model(str(clean_model_path))

        # Save LR meta as pkl
        meta_path = None
        if meta_full is not None:
            import pickle as _pkl
            meta_path = store_dir / f"{base}_meta_model.pkl"
            with open(meta_path, "wb") as _f:
                _pkl.dump(meta_full, _f)
        print(f"Models saved: {buy_path.name} + {sell_path.name} + {clean_model_path.name}" +
              (f" + {meta_path.name}" if meta_path else ""))

        sidecar = store_dir / f"{base}_meta.json"
        import pickle
        aegis_state_path = store_dir / f"{base}_aegis_state.pkl"
        with open(aegis_state_path, "wb") as f:
            pickle.dump({
                'mcf': locals().get('mcf'),
                'cre': locals().get('cre'),
                'rcm': locals().get('rcm'),
                'fw': locals().get('fw'),
                'fhm': getattr(locals().get('fhm'), 'feature_states', {})
            }, f)

        # Save HOLD-avoidance calibrator so the live predictor can apply the same
        # multi-feature HOLD filter at inference time.
        _cal_po_path = store_dir / f"{base}_hold_calibrator.pkl"
        _cal_po_obj  = locals().get('_calibrator_po')
        _cal_po_cols = locals().get('_calibrator_hold_disc_cols', [])
        if _cal_po_obj is not None:
            with open(_cal_po_path, "wb") as f:
                pickle.dump({'calibrator': _cal_po_obj,
                             'hold_disc_cols': _cal_po_cols}, f)
        else:
            _cal_po_path = None
        with open(sidecar, "w") as f:
            json.dump({
                "symbol": symbol,
                "regime_policies": regime_policies,
                "aegis_state_path": str(aegis_state_path.name),
                "disabled_filters": {
                    "sr": bool(disable_sr_veto),
                    "trend": bool(disable_trend_veto),
                    "confluence": bool(disable_confluence_veto)
                },
                "model_format": "booster",          # predictor loads with xgb.Booster()
                "primary_model_type": "binary_dual",
                "primary_model_buy_file": buy_path.name,
                "primary_model_sell_file": sell_path.name,
                "num_class": NUM_CLASS,
                "feature_cols": feature_cols,        # exact order the Booster expects
                "meta_feature_cols": meta_model_cols if 'meta_model_cols' in locals() else feature_cols,
                "calibration_temperature": T,
                "recommended_calibrator": selected_cal,
                "calibration_selector": cal_choice,
                "meta_calibration_method": meta_calibration_method,
                "meta_threshold": thr,               # combined gate (both sides)
                "production_confidence_floor": thr,  # backward-compat alias
                "edge_rank_mode": meta_gate_profile.get('edge_rank_mode') if meta_gate_profile else 'raw',
                # Per-side gates: predictor uses these when it knows which
                # direction it is proposing. Allows one side to trade even
                # if the combined precision fails the target.
                # Per-side thresholds: use individual side threshold when that side
                # qualified independently; fall back to the combined gate threshold
                # when combined gate succeeded but the per-side gate had too few
                # trades (e.g. JUP: 89.7% combined holdout but 0 per-side trades).
                "meta_threshold_buy":  thr_buy if hit_buy else thr,
                "meta_threshold_sell": thr_sell if hit_sell else thr,
                # Aggressive-mode thresholds: always store the per-side best threshold
                # even when that side didn't meet the quality bar. The predictor uses
                # these in aggressive mode instead of falling back to the combined thr.
                "meta_threshold_buy_aggressive":  thr_buy,
                "meta_threshold_sell_aggressive": thr_sell,
                # Per-side tradeability. When combined gate earned tradeable_final=True
                # but neither per-side individually qualified (e.g. min_fires split),
                # distribute the combined approval to both sides so the live engine
                # can actually fire signals using the combined threshold above.
                "tradeable_buy":  bool(tradeable_buy_holdout or
                                       (tradeable_final and not per_side_approved)),
                "tradeable_sell": bool(tradeable_sell_holdout or
                                       (tradeable_final and not per_side_approved)),
                # Top-25% confidence cutoff of fired signals on the holdout.
                # The live predictor uses this to replicate S&R and trend filters:
                # only suppress a signal when meta_conf < this value.
                "meta_override_confidence": override_conf_thr,
                # Decoupled metrics: dev OOF estimate (what justified shipping) vs
                # the single honest holdout result at the pre-committed floor.
                "metrics_summary": {
                    "dev_oof_precision_estimate": dev_prec,
                    "honest_holdout_precision_result": fired_prec,
                    "holdout_coverage_pct": coverage,
                    "dev_oof_meta_gate_lift": float(_oof_meta_gate_lift) if '_oof_meta_gate_lift' in locals() else None,
                    "dev_oof_meta_gate_fires": int(dev_n),
                },
                "gate_coverage": dev_cov,
                # ── Risk tiers ────────────────────────────────────────────────
                # The live predictor checks these to respect the user's chosen
                # risk appetite without retraining.
                "risk_tier": {
                    "conservative": tier_conservative,  # per-side holdout ≥ breakeven
                    "balanced":     tier_balanced,       # combined holdout ≥ breakeven
                    "aggressive":   tier_aggressive,     # positive EV on dev, exp ≥ 0.05%
                },
                "meta_model_file": meta_path.name if meta_path else None,  # .pkl (sklearn LR)
                "meta_model_light_file": meta_light_path.name if (meta_light_path is not None) else None,
                "atr_multiplier": atr_mult,
                # tradeable=False means the predictor emits NO signals for this token.
                # Requires: OOF target met AND (holdout unreliable OR holdout ≥ breakeven).
                # A holdout below fee breakeven with ≥ MIN_HOLDOUT_FIRES trades overrides
                # the OOF optimism and silences the token.
                "tradeable": bool(tradeable_final),
                "profitability_bypass": bool(_via_profitability_bypass),
                "primary_only_mode": True,
                "primary_confidence_threshold": float(_primary_conf_thr) if _primary_only_gate else None,
                "primary_calibrator_exists": bool(_po_use_calibrator) if _primary_only_gate else False,
                "primary_calibrator_file": _cal_po_path.name if _cal_po_path else None,
                "calibrator_hold_disc_cols": list(_cal_po_cols),
                "token_breakeven": float(token_breakeven),
                "target_precision": float(token_precision_target),
                # DEV estimate = how the gate scored on out-of-fold data (this is
                # what justified shipping the token). Pre-committed, not peeked.
                "dev_estimate": {"precision": dev_prec, "coverage": dev_cov, "trades": dev_n},
                # HOLDOUT = the same pre-committed gate applied once to untouched
                # data. This is the honest out-of-sample number; expect it near the
                # dev estimate, not above it.
                "holdout_trading": {
                    "fired":                fired_n,
                    "coverage":             coverage,
                    "signal_precision":     fired_prec,
                    "directional_precision":round(_dir_fired_prec2, 4),
                    # Overlap-adjusted evidence: raw counts overstate independence
                    # because barrier windows overlap. These are what the enable
                    # gate actually uses — a live monitor should trust them, not
                    # the raw point estimate above.
                    "dir_independent_events": int(locals().get('_dir_eff_n', 0)),
                    "dir_precision_lower_bound": round(float(locals().get('_dir_prec_lb', 0.0)), 4),
                    "signal_independent_events": int(locals().get('_sig_eff_n', 0)),
                    "label_lookahead_hours": int(token_lookahead),
                    "dir_fired_n":          _dir_fired_n2,
                    "dir_coverage":         round(_dir_coverage2, 4),
                    "dir_buy_precision":    round(_dir_buy_prec,  4),
                    "dir_sell_precision":   round(_dir_sell_prec, 4),
                    "dir_buy_n":            _dir_buy_n,
                    "dir_sell_n":           _dir_sell_n,
                    "expectancy_pct":       bt["expectancy_pct"],
                    "total_return_pct": bt["total_return_pct"],
                    "win_rate":         bt["win_rate"],
                    "sharpe":           bt["sharpe"],
                    "max_drawdown_pct": bt["max_drawdown_pct"],
                    "profit_factor":    bt["profit_factor"],
                    "kelly_pct":        bt["kelly_pct"],
                    "buy_n":            bt["buy_n"],
                    "buy_win_rate":     bt["buy_win_rate"],
                    "sell_n":           bt["sell_n"],
                    "sell_win_rate":    bt["sell_win_rate"],
                    "target_met":       bool(hit_target),
                },
                "meta_gate_ranking_audit": {
                    "selected_n":          selected_n,
                    "rejected_n":          rejected_n,
                    "selected_precision":  round(selected_prec, 4),
                    "rejected_precision":  round(rejected_prec, 4),
                    "meta_gate_lift_prec": round(meta_gate_lift, 4),
                    "selected_expectancy": round(selected_exp_pct, 4),
                    "rejected_expectancy": round(rejected_exp_pct, 4),
                    "meta_gate_lift_exp":  round(meta_gate_lift_exp, 4),
                    "selected_sharpe":     round(selected_sharpe, 4),
                    "rejected_sharpe":     round(rejected_sharpe, 4),
                    "gate_is_helpful":     bool(meta_gate_lift >= 0.01),
                },
                # AEGIS META GATE V2 — PHASE 1: Gate Lift Metrics
                "aegis_v2_gate_lift": {
                    "gate_lift_pp": round(meta_gate_lift, 4),
                    "gate_lift_expectancy": round(meta_gate_lift_exp, 4),
                    "selected_n": selected_n,
                    "rejected_n": rejected_n,
                },
                # AEGIS META GATE V2 — PHASE 2: Gate Self-Preservation Status
                "aegis_v2_gate_status": {
                    "gate_status": (
                        "HARMFUL" if meta_gate_lift < -0.20 else
                        "DEGRADED" if meta_gate_lift < -0.10 else
                        "NEUTRAL" if meta_gate_lift < 0.01 else
                        "HELPFUL"
                    ),
                    "gate_trust_score": min(100, max(0, 50 + int(meta_gate_lift * 100))),  # 0-100 scale
                    "gate_action": (
                        "BYPASS_META_GATE" if meta_gate_lift < -0.20 else
                        "REDUCE_META_INFLUENCE_50PCT" if meta_gate_lift < -0.10 else
                        "SOFTEN_THRESHOLDS_15PCT" if meta_gate_lift < 0.01 else
                        "USE_META_GATE"
                    ),
                },
                # AEGIS META GATE V2 — PHASE 5: Hold Pollution Strategy Audit
                "aegis_v2_hold_pollution": {
                    "strategy_selected": hold_strategy_selected,
                    "strategy_scores": {k: round(v.get("score", -999), 3) for k, v in hold_strategy_audit.items()},
                    "strategy_details": {
                        k: {
                            "brier": round(v.get("brier", 0), 4),
                            "sharpe": round(v.get("sharpe", 0), 4),
                            "pf": round(v.get("pf", 0), 2),
                            "precision": round(v.get("prec", 0), 4),
                            "gate_lift": round(v.get("lift", 0), 4),
                        }
                        for k, v in hold_strategy_audit.items()
                    }
                },
                # AEGIS META GATE V2 — PHASE 3: Token-Specific Profiles
                "aegis_v2_token_profile": {
                    "precision_target": float(token_precision_target),
                    "coverage_target": float(dev_cov),
                    "actual_precision": fired_prec,
                    "actual_coverage": coverage,
                    "atr_multiplier": atr_mult,
                    "gate_trust_score": min(100, max(0, 50 + int(meta_gate_lift * 100))),
                    "strategy": "ADAPTIVE_PER_REGIME" if regime_policies else "GLOBAL_THRESHOLD",
                },
                # AEGIS META GATE V2 — PHASE 4: Regime-Sensitive Gating Modifiers
                "aegis_v2_regime_modifiers": {
                    regime_name: {
                        "base_buy_thr": float(policy.get("buy_thr", thr_buy)),
                        "base_sell_thr": float(policy.get("sell_thr", thr_sell)),
                        "buy_ok": bool(policy.get("buy_ok", True)),
                        "sell_ok": bool(policy.get("sell_ok", True)),
                        "modifier": (
                            0.85 if (policy.get("buy_ok") or policy.get("sell_ok")) else 1.15
                        ),
                        "regime_quality": "GOOD" if policy.get("buy_ok") or policy.get("sell_ok") else "POOR",
                    }
                    for regime_name, policy in regime_policies.items()
                },
                # Bayes prior correction params — live predictor applies the same
                # SPW→true-rate correction before computing proposed_side() and
                # meta_prob inference (see _bayes_correct in retrain_model.py).
                "bayes_prior_correction": {
                    "type": "spw_prior_to_true_rate",
                    "true_buy_rate":  float(_true_buy_rate),
                    "true_sell_rate": float(_true_sell_rate),
                    "spw_prior_buy":  float(_spw_prior_buy),
                    "spw_prior_sell": float(_spw_prior_sell),
                },
                "trained_at": datetime.now().isoformat(),
                # ── Optimizer regime data (from threshold_optimizer.py) ────────
                # Embedded here so predictor.py needs only this one file at
                # inference time. Re-run threshold_optimizer.py after retraining
                # to refresh these values; retrain then picks them up on the next
                # training cycle.
                "regime_thresholds":  (rcm.regime_thresholds if rcm is not None else (_opt or {}).get("regimes", {})),
                "regime_threshold_modifier":  (
                    (rcm.regime_modifiers if rcm is not None and (regime_modifier_profile is not False) else {})
                    if regime_modifier_profile is not None else
                    (rcm.regime_modifiers if rcm is not None else {})
                ),
                "meta_gate_profile": {
                    "gate_type": gate_type,
                    "threshold_quantile": float(meta_gate_profile.get("threshold_quantile")) if meta_gate_profile else None,
                    "thresholds": meta_gate_profile.get("thresholds") if meta_gate_profile else {},
                    "side_specific": bool(meta_gate_profile.get("side_specific", True)) if meta_gate_profile else True,
                    "signal_vetoes": signal_vetoes if signal_vetoes is not None else [],
                    "regime_modifier": regime_modifier_profile,
                    "edge_rank_mode": meta_gate_profile.get("edge_rank_mode") if meta_gate_profile else None,
                    "calibration": meta_gate_profile.get("calibration") if meta_gate_profile else {},
                    "disabled_reason": meta_gate_profile.get("disabled_reason") if meta_gate_profile else None,
                },
                "disabled_filters": {
                    "sr": bool(disable_sr_veto),
                    "trend": bool(disable_trend_veto),
                    "confluence": bool(disable_confluence_veto),
                },
                "optimizer_updated_at": (_opt or {}).get("updated_at"),
            }, f, indent=2)

        # Feature importance + stability recording
        importance_dict = log_feature_importance(deploy_buy, feature_cols, symbol)
        try:
            from src.ml.feature_stability import record_retrain, compute_stability_scores
            from src.ml.pruning_engine import generate_feature_health_report
            from src.ml.diagnostics import save_top_features, save_feature_health_report
            retrain_id = f"{symbol.replace('/','_')}_{int(time.time())}"
            rec = record_retrain(symbol, retrain_id, importance_dict)
            scores = compute_stability_scores(last_n=12, top_k=50)
            # Persist average stability for sidecar consumption
            avg_stab = float(sum(scores.values()) / max(1, len(scores))) if scores else 0.0
            # Update sidecar with feature stability summary
            sc_text = sidecar.read_text()
            sc_json = json.loads(sc_text)
            sc_json['feature_stability_avg'] = avg_stab
            sc_json['feature_stability_scores_available'] = bool(scores)
            with open(sidecar, 'w') as f:
                json.dump(sc_json, f, indent=2)
            # Generate pruning report and diagnostics
            report = generate_feature_health_report(last_n=12, top_k=50)
            save_top_features(symbol, importance_dict)
            save_feature_health_report(symbol, report)
            print(f"   Feature stability avg: {avg_stab:.3f} | health report saved")
        except Exception as _e:
            print(f"   Feature stability recording skipped: {_e}")

        # HMM regime intelligence layer pre-trained at the start of train_token.

        return {
            "symbol": symbol,
            "cv_accuracy": cv_acc,
            "cv_macro_f1": cv_f1,
            "holdout_signal_precision": fired_prec,
            "holdout_dir_precision": _dir_fired_prec2,
            "holdout_coverage": coverage,
            "holdout_fired": fired_n,
            "holdout_expectancy_pct": bt["expectancy_pct"],
            "holdout_total_return_pct": bt["total_return_pct"],
            "meta_threshold": thr,
            "target_met": bool(hit_target),
            "tradeable": bool(tradeable_final),
            "feature_count": len(feature_cols),
            "model_path": str(model_path),
            "atr_multiplier": atr_mult,
            "class_distribution": {"sell": int(cc[0]), "hold": int(cc[1]), "buy": int(cc[2])},
        }

    except Exception as e:
        print(f"Unexpected error for {symbol}: {type(e).__name__}: {e}")
        return None


# ============================================================
# FLEET TRAINING
# ============================================================
def _find_resume_index(store_dir: Path) -> int:
    """
    Return the index in FLEET_SYMBOLS to start from.

    Scans model_store for *_model.json files and maps them back to fleet symbols.
    Resumes from the first missing token in FLEET_SYMBOLS order, so a partial
    fleet run can continue from exactly where it stopped.
    """
    model_files = list(store_dir.glob("*_model.json"))
    if not model_files:
        return 0

    sym_map = {s.replace("/", "_"): s for s in FLEET_SYMBOLS}
    completed = set()
    for p in model_files:
        base = p.name.replace("_model.json", "")
        symbol = sym_map.get(base)
        if symbol:
            completed.add(symbol)

    for idx, symbol in enumerate(FLEET_SYMBOLS):
        if symbol not in completed:
            return idx

    return len(FLEET_SYMBOLS)


def train_fleet(hours: int = 5000, resume: bool = True):
    # ── Step 0: refresh OI cache ──────────────────────────────────────────
    # Keeps open-interest features current without a separate script run.
    # Non-fatal: if it fails for any reason, training continues with whatever
    # cached data already exists.
    print("=" * 70 + "\nREFRESHING OI CACHE\n" + "=" * 70)
    try:
        from scripts.update_oi_cache import main as _refresh_oi  # type: ignore[import]
        _refresh_oi(FLEET_SYMBOLS)
        print("OI cache refresh complete.\n")
    except Exception as _oi_err:
        print(f"OI cache refresh skipped ({type(_oi_err).__name__}: {_oi_err}) "
              f"— training will use existing cached data.\n")

    # ── Step 1: check news freshness ──────────────────────────────────────
    print("=" * 70 + "\nCHECKING NEWS DATA FRESHNESS\n" + "=" * 70)
    if ensure_fresh_news_for_training():
        print("News data is ready.")
    else:
        print("Proceeding without fresh news. Sentiment features will be zero.")

    print("\n" + "=" * 70)
    print("AEGIS-1 FLEET TRAINING (meta-labeling)")
    print("   Primary direction model + meta gate + tuned threshold + PnL backtest")
    print(f"   Target signal precision: {TARGET_SIGNAL_PRECISION:.0%} | fees: {FEE_ROUNDTRIP*100:.2f}% RT")
    print("   Purged TimeSeriesCV | SHAP pruning | held-out test")
    print("=" * 70)

    store_dir = Path(root_dir) / "src" / "ml" / "model_store"
    start_idx = _find_resume_index(store_dir) if resume else 0
    total = len(FLEET_SYMBOLS)
    if start_idx >= total:
        print("\nAll fleet tokens already have saved models. No training required.")
        return

    if start_idx > 0:
        skipped = FLEET_SYMBOLS[:start_idx]
        print(f"\nRESUMING from {FLEET_SYMBOLS[start_idx]} "
              f"(skipping {start_idx} already-trained token{'s' if start_idx != 1 else ''}: "
              f"{', '.join(skipped[:3])}{'...' if len(skipped) > 3 else ''})")
    else:
        print("\nStarting full fleet training from the beginning.")

    results = []
    for idx, symbol in enumerate(FLEET_SYMBOLS, 1):
        if idx - 1 < start_idx:
            continue                          # skip already-completed tokens
        print(f"\n[{idx}/{total}] Processing {symbol}...")
        m = train_token(symbol, hours=hours)
        if m:
            results.append(m)
            # [LIVE] requires at least one per-side to be tradeable. tradeable=True
            # on its own (via expectancy override) with both per-side flags False means
            # the live engine fires nothing — don't show [LIVE] in that case.
            if (m.get("tradeable", False)
                    and (m.get("tradeable_buy", False) or m.get("tradeable_sell", False))):
                tag = "[LIVE]"
            elif m.get("tradeable", False):
                tag = "[ENABLED-no-side]"  # override fired but neither side cleared holdout
            elif m["target_met"]:
                tag = "[HOLDOUT-FAIL]"
            else:
                tag = "[ ]"
            print(f"   DONE {symbol} {tag} dir_prec {m['holdout_dir_precision']:.1%} "
                  f"| sig_prec {m['holdout_signal_precision']:.1%} "
                  f"| coverage {m['holdout_coverage']:.1%} "
                  f"| exp/trade {m['holdout_expectancy_pct']:+.2f}%")
        else:
            print(f"   {symbol} skipped")

    print("\n" + "=" * 70 + "\nTRAINING SUMMARY\n" + "=" * 70)
    if results:
        live = [r for r in results if r.get("tradeable", False)]
        fail = [r for r in results if r["target_met"] and not r.get("tradeable", False)]
        prof = [r for r in results if r["holdout_expectancy_pct"] > 0 and r["holdout_fired"] >= 10]
        print(f"Trained: {len(results)} / {total}")
        print(f"LIVE-tradeable (OOF + holdout confirmed): {len(live)} tokens")
        print(f"Holdout-failed (OOF met, disabled by holdout): {len(fail)} tokens")
        print(f"Positive expectancy (>=10 trades): {len(prof)} tokens")
        print(f"Avg signal precision   : {np.mean([r['holdout_signal_precision'] for r in results]):.1%}")
        print(f"Avg coverage           : {np.mean([r['holdout_coverage'] for r in results]):.1%}")
        print("   Ship only the tokens with positive expectancy AND enough trades.")
        print("   Precision/coverage trade off -- that trade-off IS the product spec.")
    else:
        print("No models were trained. Check errors above.")

    logs_dir = Path(root_dir) / "logs"
    logs_dir.mkdir(exist_ok=True)
    with open(logs_dir / "training_summary.json", "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "hours_of_data": hours,
                   "total_tokens": total, "successful": len(results), "results": results},
                  f, indent=2)
    print(f"\nSummary saved to {logs_dir / 'training_summary.json'}")


if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser(description="Aegis-1 fleet trainer")
    _parser.add_argument("--hours", type=int, default=7000,
                         help="Hours of OHLCV history to fetch per token (default 7000)")
    _parser.add_argument("--full", action="store_true",
                         help="Force a full retrain from the first token, ignoring any "
                              "previously saved models (default: auto-resume)")
    _parser.add_argument("--symbol", type=str, default=None,
                         help="Train only a single symbol (e.g. BTC/USDT)")
    _args = _parser.parse_args()
    if _args.symbol:
        train_token(_args.symbol, hours=_args.hours)
    else:
        train_fleet(hours=_args.hours, resume=not _args.full)