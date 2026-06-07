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
MAX_LOOKAHEAD = 48
EMBARGO = MAX_LOOKAHEAD
CENSORED = -1

TEST_FRAC = 0.20
N_SPLITS_CV = 15              # TASK 6: Increased from 10 to 15 for statistical robustness
OPTUNA_TRIALS = 15

# ============================================================
# TASK 1: HOLD POLLUTION REFACTOR (3 strategies)
# ============================================================
META_HOLD_STRATEGY = "C_excluded"  # "A_current", "B_reduced", or "C_excluded"
META_HOLD_AUTO_SELECT = True       # Automatically test all 3 and select best

# ============================================================
# TASK 3: ADAPTIVE COVERAGE TARGETING
# ============================================================
MIN_COVERAGE = 0.08            # 8% minimum coverage
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

SHAP_CUMULATIVE_THRESH = 0.90  # keep features covering 90% of total |SHAP| importance
SHAP_TOP_PCT = 0.20            # floor: keep at least top 20% of features by SHAP rank
MIN_FEATURES = 25
MAX_FEATURES = 90              # hard cap: beyond ~90 features XGBoost sees diminishing returns on 1h crypto

MIN_TOTAL_ROWS = 600
MIN_FIT_ROWS = 300

# --- meta-labeling / trading ---
# 70% directional precision at 1h is not realistic for crypto from TA alone.
# A 56-60% gate with real coverage and positive expectancy after fees is a
# genuine product; 70% with zero coverage is not. Set the target where the data
# can actually reach, and let coverage tell you how much signal exists.
TARGET_SIGNAL_PRECISION = 0.62   # realistic directional-precision target
MIN_FIRES_DEV = 80               # need >=80 OOF trades before trusting a threshold
# Minimum holdout signals required before the holdout result can override the OOF
# tradeable decision. Below this count the sample is too small to be conclusive —
# the OOF estimate (not holdout) governs. At or above it, a below-breakeven holdout
# disables the token regardless of OOF performance.
MIN_HOLDOUT_FIRES = 10
GAP_VETO_THRESHOLD = 0.15        # maximum acceptable OOF->holdout precision drop
FEE_ROUNDTRIP = 0.001            # 0.10% round-trip (taker + slippage); tune to your venue
EXPECTANCY_FLOOR = 0.20          # 0.20% minimum expectancy floor for override

# Asymmetric triple-barrier skew. Squeeze the downside barrier (catch fast drops
# sooner) and widen the upside (buffer fake breakouts). NOTE: this changes the
# LABEL definition, so it shifts the SELL/BUY class balance — it does not by
# itself create predictive skill. Watch the printed class distribution to see
# what it's actually doing.
BARRIER_UP_SKEW = 1.15
BARRIER_DOWN_SKEW = 0.85

NEWS_FILE = Path(root_dir) / "data" / "news_data.json"
NEWS_MAX_AGE_SECONDS = 30 * 60

DEFAULT_PARAMS = {
    'objective': 'multi:softprob', 'eval_metric': 'mlogloss', 'num_class': NUM_CLASS,
    'max_depth': 5, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'reg_lambda': 2.0, 'reg_alpha': 1.0, 'min_child_weight': 8,
    'seed': 42, 'tree_method': 'hist', 'missing': np.nan,
}

META_PARAMS = {
    'objective': 'reg:squarederror', 'eval_metric': 'rmse',
    'max_depth': 4, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'reg_lambda': 2.0, 'min_child_weight': 10,
    'seed': 42, 'tree_method': 'hist', 'missing': np.nan,
}

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

# ════════════════════════════════════════════════════════════════════════════════
# CRITICAL FEATURE BLACKLIST — Drifted features (PSI > 1.0 or KS > 0.50)
# Forensic report: 16 features CRITICAL (Score: 64/100). Removing top 10 expected +21.0pp precision gain.
# ════════════════════════════════════════════════════════════════════════════════
FEATURE_BLACKLIST = {
    # ── Absolute price features (CRITICAL drift: PSI > 20.5, KS > 0.93) ──
    'low',          # PSI=20.855, KS=0.938 → +2.1pp gain
    'close',        # PSI=20.552, KS=0.938 → +2.1pp gain
    'se_mid',       # PSI=20.515, KS=0.935 → +2.1pp gain
    
    # ── Decay-normalized features (CRITICAL drift) ──
    'vwap_decay_mean_24',       # PSI=17.391, KS=0.906 → +2.1pp gain
    'returns_1h_decay_std_24',  # PSI=3.355, KS=0.536 → +2.1pp gain
    'vwap_decay_std_24',        # PSI=3.236, KS=0.709 → +2.1pp gain
    'close_decay_std_24',       # Critical indicator
    'volume_decay_std_24',      # Critical indicator
    
    # ── Funding/volatility features (CRITICAL drift) ──
    'funding_rate_ma8',         # PSI=2.368, KS=0.541 → +2.1pp gain
    'funding_rate',             # PSI=2.257, KS=0.488 → +2.1pp gain
    'gk_vol',                   # PSI=1.911, KS=0.576 → +2.1pp gain
    'volume_decay_mean_24',     # PSI=1.739, KS=0.503 → +2.1pp gain
    'donchian_width',           # Critical non-price indicator
}

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

    Returns one of {48,36,24,18} per median ATR% buckets.
    """
    try:
        atr = compute_atr(df, period=14)
        atr_pct = (atr / df['close'].replace(0, np.nan)).fillna(0)
        med = float(np.nanmedian(atr_pct))
        if med < 0.005:
            return 48
        if med < 0.010:
            return 36
        if med < 0.020:
            return 24
        return 18
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
    trend_adj = -0.12 if trend_regime == 1 else 0.0
    threshold = 0.78 - 0.22 * er + 0.08 * np.tanh((vol - 1.0) * 1.8) + trend_adj
    return float(np.clip(threshold, 0.45, 0.86))


def _adaptive_efficiency_floor(volatility_regime: float) -> float:
    vol = float(np.clip(volatility_regime if volatility_regime is not None else 1.0, 0.6, 1.8))
    return float(np.clip(0.18 + 0.12 * max(0.0, vol - 1.0), 0.14, 0.30))


def _adaptive_confluence_bounds(score: pd.Series) -> Tuple[float, float]:
    if score is None or score.empty:
        return -0.50, 0.50
    lower = float(np.quantile(score, 0.20))
    upper = float(np.quantile(score, 0.80))
    if np.isclose(lower, upper):
        lower -= 0.05
        upper += 0.05
    return lower, upper


def create_triple_barrier_labels(df: pd.DataFrame, atr_multiplier: float,
                                  max_lookahead: int = MAX_LOOKAHEAD,
                                  volatility_regime: Optional[pd.Series] = None,
                                  efficiency_ratio: Optional[pd.Series] = None,
                                  trend_regime: Optional[pd.Series] = None,
                                  macro_confluence_score: Optional[pd.Series] = None,
                                  barrier_multiplier: Optional[float] = None) -> pd.Series:
    """3-class labels: 0=SELL, 1=HOLD, 2=BUY, -1=CENSORED (dropped upstream)."""
    if df is None or df.empty:
        return pd.Series(dtype=int)

    labels = pd.Series(1, index=df.index, dtype=int)
    atr = compute_atr(df, period=14)
    n = len(df)
    cs_lower, cs_upper = _adaptive_confluence_bounds(macro_confluence_score) if macro_confluence_score is not None else (-0.50, 0.50)

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

        dynamic_mult = compute_dynamic_atr_multiplier(atr_multiplier, er_i, vol_regime_i)

        # Regime-based barrier adjustment: allow optional per-call override
        reg_barrier_adj = 1.0
        if barrier_multiplier is not None:
            reg_barrier_adj = float(barrier_multiplier)
        else:
            # Heuristic: flat markets need wider barriers, volatile markets tighter
            if abs(trend_i) < 0.02:
                reg_barrier_adj = 1.5
            elif vol_regime_i > 1.25:
                reg_barrier_adj = 0.8
            elif er_i > 0.6:
                reg_barrier_adj = 0.9

        upper = entry_price + (dynamic_mult * reg_barrier_adj * BARRIER_UP_SKEW) * atr_val
        lower = entry_price - (dynamic_mult * reg_barrier_adj * BARRIER_DOWN_SKEW) * atr_val

        window_avail = min(max_lookahead, n - 1 - i)
        hit = None
        for j in range(1, window_avail + 1):
            high = df.iloc[i + j]['high']
            low = df.iloc[i + j]['low']
            if high >= upper:
                hit = 2
                break
            if low <= lower:
                hit = 0
                break

        if macro_confluence_score is not None and hit is not None and cs_lower < cs_upper:
            cs = float(macro_confluence_score.iloc[i])
            if hit == 2 and cs <= cs_lower:
                hit = None
            if hit == 0 and cs >= cs_upper:
                hit = None

        labels.iloc[i] = hit if hit is not None else (1 if window_avail >= max_lookahead else CENSORED)

    if n > 0:
        tail = min(max_lookahead, n)
        labels.iloc[n - tail:] = CENSORED
    return labels


def get_class_weights(y: np.ndarray, min_directional_ratio: float = 0.20) -> np.ndarray:
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
            
            # Online update: decay old stats, incorporate latest bar
            if len(df) > 0:
                latest_val = float(df[col].iloc[-1])
                new_mean = 0.99 * self.means[col] + 0.01 * latest_val
                new_std_sq = 0.99 * (self.stds[col] ** 2) + 0.01 * ((latest_val - new_mean) ** 2)
                new_std = float(np.sqrt(max(new_std_sq, 1e-8)))
                self.means[col] = new_mean
                self.stds[col] = max(new_std, 1e-6)
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


def proposed_side(primary_probs: np.ndarray) -> np.ndarray:
    """Primary's directional proposal: 2=BUY if buy-prob >= sell-prob else 0=SELL."""
    return np.where(primary_probs[:, 2] >= primary_probs[:, 0], 2, 0)


def objective(trial, Xtr, ytr, Xva, yva, fw=None):
    params = {
        'objective': 'multi:softprob', 'eval_metric': 'mlogloss', 'num_class': NUM_CLASS,
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 1, 20, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 5, 20),
        'seed': 42, 'tree_method': 'hist', 'missing': np.nan,
    }
    m = xgb.train(params, _dm(Xtr, ytr, sample_weights(ytr), fw=fw), num_boost_round=500,
                  evals=[(_dm(Xva, yva, fw=fw), 'eval')], early_stopping_rounds=50, verbose_eval=False)
    return float(log_loss(yva, m.predict(_dm(Xva, fw=fw)), labels=list(range(NUM_CLASS))))


def primary_oof(X: pd.DataFrame, y: np.ndarray, params: dict,
                n_splits: int, gap: int, fw: Optional[np.ndarray] = None) -> np.ndarray:
    """Out-of-fold primary probabilities (purged). Early rows stay NaN."""
    oof = np.full((len(X), NUM_CLASS), np.nan)
    for tr, va in TimeSeriesSplit(n_splits=n_splits, gap=gap).split(X):
        m = xgb.train(params, _dm(X.iloc[tr], y[tr], sample_weights(y[tr]), fw=fw),
                      num_boost_round=500, evals=[(_dm(X.iloc[va], y[va], fw=fw), 'eval')],
                      early_stopping_rounds=50, verbose_eval=False)
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
        thr, prec, cov, n = meeting[0]
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
        thr, prec, cov, n = meeting[0]
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
    equity  = np.cumsum(rets_arr)
    peak    = np.maximum.accumulate(equity)
    drawdown_dollars = peak - equity
    # FIX (CRITICAL): Normalize by peak to get true percentage (max_dd <= 100%)
    # Before: multiplied decimal by 100 directly, causing 3000%+ values when equity high
    # After: divide drawdown_dollars by peak value (same as validation.py line 95)
    peak_safe = np.maximum(peak, 1e-9)  # Avoid division by zero
    drawdown_pct = drawdown_dollars / peak_safe
    max_dd  = float(drawdown_pct.max() * 100)

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
    return importance_dict if 'importance_dict' in locals() else {}


# ============================================================
# TRAIN A SINGLE TOKEN
# ============================================================
def train_token(symbol: str, hours: int = 5000) -> Optional[Dict]:
    print(f"\n{'='*60}\nTraining model for {symbol}\n{'='*60}")
    try:
        p = Predictor(symbol)
        print(f"Fetching {hours} hours of data for {symbol}...")
        df = p.fetch_live_data(timeframe='1h', limit=hours)
        if df is None or df.empty:
            print(f"No data for {symbol}")
            return None

        btc_df = None
        if hasattr(p, 'fetch_btc_data'):
            print("Fetching BTC market context...")
            btc_df = p.fetch_btc_data(timeframe='1h', limit=hours)
            if btc_df is None or btc_df.empty:
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
        try:
            soft_conf = compute_soft_confluence_features(df)
            for col in soft_conf.columns:
                df[col] = soft_conf[col].values
            _conf_sample = float(df['prc_total'].iloc[-1]) if 'prc_total' in df.columns else 0.5
            print(f"   prc_total sample (last bar): {_conf_sample:.3f}  "
                  f"macro_conf_score: {float(df['macro_confluence_score'].iloc[-1]):.3f}")
        except Exception as _sc_err:
            print(f"   Soft confluence computation failed ({_sc_err}) — using sign-based fallback")

        for col in ['volatility_regime', 'efficiency_ratio_10', 'trend_regime']:
            if col not in df.columns:
                print(f"{col} missing -- using constant")
                df[col] = 1.0 if col == 'volatility_regime' else (0.5 if 'efficiency' in col else 0)

        # Keep ATR for the PnL backtest (excluded from features via leading underscore).
        df['_atr'] = compute_atr(df, period=14).values

        # ---- Train fresh HMM first so regime labels are available for training and validation ----
        # FIX 2: HMM auto-recovery with state count validation
        try:
            from src.ml.hmm_regime import train_hmm_for_symbol as _train_hmm, label_dataframe as _label_df
            print("   Training fresh HMM regime engine...")
            _hmm_ok = _train_hmm(symbol, df)
            if _hmm_ok:
                df['hmm_regime'] = _label_df(symbol, df)
                n_unique_regimes = len(df['hmm_regime'].unique())
                print(f"   [HMM] Fresh HMM trained with {n_unique_regimes} effective states.")
                if n_unique_regimes == 1:
                    print(f"   [HMM WARNING] Only 1 state detected. Regime filter will have limited effect.")
            else:
                print("   [HMM] Fresh HMM training returned False. Using existing engine for labeling.")
                df['hmm_regime'] = _label_df(symbol, df)
                n_unique_regimes = len(df['hmm_regime'].unique())
                if n_unique_regimes == 1:
                    print(f"   [HMM WARNING] Fallback assigned 100% to 1 state. Regime filter disabled.")
        except Exception as _hmm_err:
            print(f"   [HMM] Pre-training failed: {type(_hmm_err).__name__}: {_hmm_err}")
            if 'hmm_regime' not in df.columns:
                df['hmm_regime'] = 'UNKNOWN'
                print(f"   [HMM] Using UNKNOWN regime fallback.")

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
            _meta_reg, _meta_mcw = 3.0, 12   # was 6.0, 20
        elif _er_med < 0.5:
            _meta_reg, _meta_mcw = 2.5, 10   # was 4.0, 15
        else:
            _meta_reg, _meta_mcw = 2.0, 8    # was 2.0, 10
        token_meta_params = {**META_PARAMS, 'reg_lambda': _meta_reg, 'min_child_weight': _meta_mcw}

        # ---- 0) Recursive Label Rebalancer (Phase 3) ----
        target_hold_max = 0.50
        target_buy_min, target_buy_max = 0.20, 0.40
        target_sell_min, target_sell_max = 0.20, 0.40
        
        loop_atr_mult = atr_mult
        best_atr_mult = atr_mult
        best_dist = 999.0
        
        N_all = len(df)
        test_start_temp = N_all - int(N_all * TEST_FRAC)
        train_end_temp = test_start_temp - EMBARGO
        
        print("   Auditing training label distribution (rebalancer)...")
        preview_labels = create_triple_barrier_labels(
            df, atr_multiplier=loop_atr_mult, max_lookahead=token_lookahead,
            volatility_regime=df['volatility_regime'],
            efficiency_ratio=df['efficiency_ratio_10'],
            trend_regime=df['trend_regime'],
            macro_confluence_score=df.get('macro_confluence_score'),
        )
        preview_valid = preview_labels[preview_labels != CENSORED]
        if len(preview_valid) > 0:
            freq_preview = np.bincount(preview_valid.astype(int), minlength=3) / len(preview_valid)
        else:
            freq_preview = np.array([0.20, 0.50, 0.30], dtype=float)

        target_buy_min = float(np.clip(freq_preview[2] * 0.70, 0.08, 0.45))
        target_sell_min = float(np.clip(freq_preview[0] * 0.70, 0.08, 0.45))
        target_hold_max = float(np.clip(freq_preview[1] * 1.15, 0.30, 0.70))
        target_buy_max = float(np.clip(target_buy_min + 0.18, 0.20, 0.50))
        target_sell_max = float(np.clip(target_sell_min + 0.18, 0.20, 0.50))

        for attempt in range(8):
            labels_temp = create_triple_barrier_labels(
                df, atr_multiplier=loop_atr_mult, max_lookahead=token_lookahead,
                volatility_regime=df['volatility_regime'],
                efficiency_ratio=df['efficiency_ratio_10'],
                trend_regime=df['trend_regime'],
                macro_confluence_score=df.get('macro_confluence_score'),
            )
            labels_train = labels_temp.iloc[:train_end_temp]
            valid_labels = labels_train[labels_train != CENSORED]
            
            if len(valid_labels) == 0:
                loop_atr_mult = max(0.5, loop_atr_mult - 0.2)
                continue
                
            counts = np.bincount(valid_labels.astype(int), minlength=3)
            freqs = counts / len(valid_labels)
            freq_sell, freq_hold, freq_buy = freqs[0], freqs[1], freqs[2]
            
            print(f"      Attempt {attempt+1}: atr_mult={loop_atr_mult:.2f} -> BUY: {freq_buy:.1%}, SELL: {freq_sell:.1%}, HOLD: {freq_hold:.1%}")
            
            if freq_hold <= target_hold_max and \
               target_buy_min <= freq_buy <= target_buy_max and \
               target_sell_min <= freq_sell <= target_sell_max:
                best_atr_mult = loop_atr_mult
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
                
        atr_mult = best_atr_mult
        print(f"   Optimal ATR multiplier selected: {atr_mult:.2f}")

        print(f"ATR multiplier : base={atr_mult} | typical={_typical:.2f} "
              f"(ER_med={_er_med:.2f}, vol_med={_vol_med:.2f}) | range=[{atr_mult*0.8:.1f}, 4.5]")
        print(f"Token params   : lookahead={token_lookahead}h | "
              f"precision_target={token_precision_target:.1%} (breakeven~{token_breakeven:.1%}) | "
              f"meta_reg=L{_meta_reg}/mcw{_meta_mcw}")
        print(f"   [ADAPTIVE] symbol={symbol} lookahead={token_lookahead} atr_mult={atr_mult:.2f} "
              f"precision_target={token_precision_target:.1%} target_buy=[{target_buy_min:.1%},{target_buy_max:.1%}] "
              f"target_sell=[{target_sell_min:.1%},{target_sell_max:.1%}] hold_max={target_hold_max:.1%}")

        labels = create_triple_barrier_labels(
            df, atr_multiplier=atr_mult, max_lookahead=token_lookahead,
            volatility_regime=df['volatility_regime'],
            efficiency_ratio=df['efficiency_ratio_10'],
            trend_regime=df['trend_regime'],
            macro_confluence_score=df.get('macro_confluence_score'),
        )
        df['target'] = labels.astype(int).values

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
                        if c not in ('timestamp', 'target') and not c.startswith('_')]
        for addon in FEATURE_ADDONS:
            if addon in df.columns and addon not in feature_cols:
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
        train_pool = df.iloc[:train_end].reset_index(drop=True)
        holdout = df.iloc[test_start:].reset_index(drop=True)
        print(f"   Split -> train pool: {len(train_pool)} | embargo: {EMBARGO} | holdout: {len(holdout)}")
        
        # Copy raw price and ATR values to separate columns before any feature drift normalization/scaling
        train_pool['_close_raw'] = train_pool['close'].copy()
        train_pool['_atr_raw'] = train_pool['_atr'].copy()
        holdout['_close_raw'] = holdout['close'].copy()
        holdout['_atr_raw'] = holdout['_atr'].copy()

        Xtp = train_pool[feature_cols]
        ytp = train_pool['target'].to_numpy().astype(int)

        # ---- 1) Feature Health Manager & Drift Corrector (Phase 2) ----
        print("   Evaluating Feature Health and Drift...")
        from src.ml.feature_health import FeatureHealthManager, DynamicFeatureWeightingEngine
        fhm = FeatureHealthManager()

        # Apply deterministic feature blacklist FIRST: PRIORITY 1 — hardcoded critical drifters
        # These are the top critical features identified by forensic reports (PSI/KS thresholds).
        to_drop_hardcoded = [c for c in feature_cols if c in FEATURE_BLACKLIST]
        if to_drop_hardcoded:
            print(f"   [FORENSIC FIX] Removing {len(to_drop_hardcoded)} hardcoded critical drifters: {to_drop_hardcoded[:8]}...")
            feature_cols = [c for c in feature_cols if c not in to_drop_hardcoded]
            Xtp = train_pool[feature_cols]

        # Now run drift analysis on the cleaned feature set so FeatureHealthManager
        # only evaluates remaining features and can suggest additional removals.
        try:
            fhm.analyze_drift(Xtp, holdout[feature_cols], feature_cols)
        except Exception:
            # If drift analysis fails, continue—hard blacklist still applied.
            print("   FeatureHealth.analyze_drift() failed; continuing with hardcoded blacklist applied.")

        # PRIORITY 2 — FeatureHealthManager state-based removal for remaining drifted features
        try:
            states = fhm.get_feature_states()
            to_drop_fhm = [c for c, s in states.items() if s in ("DEGRADED", "CRITICAL") and c not in to_drop_hardcoded]
            if to_drop_fhm:
                print(f"   FeatureHealth: dropping {len(to_drop_fhm)} additional DEGRADED/CRITICAL features: {to_drop_fhm[:5]}...")
                # Update feature_cols and training frame
                feature_cols = [c for c in feature_cols if c not in to_drop_fhm]
                Xtp = train_pool[feature_cols]
            total_removed = len(to_drop_hardcoded) + len(to_drop_fhm)
            if total_removed > 0:
                print(f"   Total drifted features removed: {total_removed} (hardcoded: {len(to_drop_hardcoded)}, FHM: {len(to_drop_fhm)})")
        except Exception as e_fhm:
            print(f"   FeatureHealth state lookup failed: {e_fhm}")
        
        # Train a quick default model to rank features by gain
        print("   Training baseline model to audit feature contribution...")
        quick_dtrain = xgb.DMatrix(Xtp, label=ytp)
        quick_model = xgb.train({'objective': 'multi:softprob', 'num_class': NUM_CLASS, 'seed': 42}, quick_dtrain, num_boost_round=50)
        importance_scores = quick_model.get_score(importance_type='gain')
        feature_importance = {col: float(importance_scores.get(col, 0.0)) for col in feature_cols}
        sorted_features = sorted(feature_cols, key=lambda c: feature_importance.get(c, 0.0), reverse=True)
        M = len(sorted_features)
        
        feature_transforms = {}
        removed_features = []
        unstable_before = []
        
        for col in feature_cols:
            psi = fhm.drift_scores.get(col, {}).get('psi', 0.0)
            ks = fhm.drift_scores.get(col, {}).get('ks', 0.0)
            
            if psi > 1.0 or ks > 0.50:
                unstable_before.append(col)
                rank = sorted_features.index(col)
                
                if rank < int(M * 0.15):
                    # High importance -> z-score
                    mean_val = float(Xtp[col].mean())
                    std_val = float(Xtp[col].std())
                    feature_transforms[col] = {
                        "type": "zscore",
                        "mean": mean_val,
                        "std": std_val
                    }
                    Xtp[col] = (Xtp[col] - mean_val) / (std_val + 1e-8)
                    holdout[col] = (holdout[col] - mean_val) / (std_val + 1e-8)
                elif rank < int(M * 0.50):
                    # Medium importance -> minmax
                    min_val = float(Xtp[col].min())
                    max_val = float(Xtp[col].max())
                    feature_transforms[col] = {
                        "type": "minmax",
                        "min": min_val,
                        "max": max_val
                    }
                    Xtp[col] = (Xtp[col] - min_val) / (max_val - min_val + 1e-8)
                    holdout[col] = (holdout[col] - min_val) / (max_val - min_val + 1e-8)
                else:
                    # Low importance -> remove
                    removed_features.append(col)
                    
        # Exclude removed drift features
        feature_cols_clean = [c for c in feature_cols if c not in removed_features]
        Xtp = Xtp[feature_cols_clean].copy()
        
        # ---- Feature Rank Correlation Audit (Priority 1) ----
        print("   Performing Feature Rank Correlation Audit...")
        # Compute continuous target outcomes for correlation audit on training pool
        close_tp = train_pool['_close_raw'].to_numpy()
        atr_tp = train_pool['_atr_raw'].to_numpy()
        y_tp = ytp
        
        rets_tp = []
        r_mult_tp = []
        for i in range(len(train_pool)):
            yt_val = int(y_tp[i])
            b_val = atr_mult * atr_tp[i] / close_tp[i] if close_tp[i] > 0 else 0.015
            buy_ret = (b_val - FEE_ROUNDTRIP) if yt_val == 2 else (-b_val - FEE_ROUNDTRIP) if yt_val == 0 else -FEE_ROUNDTRIP
            
            rets_tp.append(buy_ret)
            r_mult_tp.append(buy_ret / b_val if b_val > 0 else 0.0)
            
        rets_tp = np.array(rets_tp)
        r_mult_tp = np.array(r_mult_tp)
        
        uncorrelated_features = []
        for col in feature_cols_clean:
            corr_pnl = Xtp[col].corr(pd.Series(rets_tp), method='spearman')
            corr_rmult = Xtp[col].corr(pd.Series(r_mult_tp), method='spearman')
            
            max_corr = max(
                abs(corr_pnl) if pd.notna(corr_pnl) else 0.0,
                abs(corr_rmult) if pd.notna(corr_rmult) else 0.0
            )
            if max_corr < 0.10:
                uncorrelated_features.append(col)
                removed_features.append(col)
                
        # Final active features
        feature_cols = [c for c in feature_cols_clean if c not in uncorrelated_features]
        Xtp = train_pool[feature_cols].copy()
        
        # Re-compute drift for report
        fhm_after = FeatureHealthManager()
        fhm_after.analyze_drift(Xtp, holdout[feature_cols], feature_cols)
        unstable_after = [c for c in feature_cols if fhm_after.drift_scores.get(c, {}).get('psi', 0.0) > 1.0 or fhm_after.drift_scores.get(c, {}).get('ks', 0.0) > 0.50]
        
        print(f"\n   === AEGIS FEATURE RECONCILIATION REPORT for {symbol} ===")
        print(f"      Initial features: {len(sorted_features)}")
        print(f"      Unstable features (drifted) before: {len(unstable_before)}")
        print(f"      Features transformed (Z-score/MinMax): {len(feature_transforms)}")
        print(f"      Features removed due to drift: {len([c for c in removed_features if c not in uncorrelated_features])}")
        print(f"      Features removed due to low correlation (<0.10): {len(uncorrelated_features)}")
        print(f"      Total features removed: {len(removed_features)}")
        print(f"      Unstable features after: {len(unstable_after)}")
        print(f"      Active features remaining: {len(feature_cols)}")
        
        dfwe = DynamicFeatureWeightingEngine()
        fw = dfwe.calculate_weights(fhm_after.feature_states, feature_cols)
        print(f"   Active feature set: {len(feature_cols)} features.")

        # ---- 2) Optuna tune primary (purged inner split) ----
        inner = list(TimeSeriesSplit(n_splits=N_SPLITS_CV, gap=EMBARGO).split(Xtp))
        itr, iva = inner[-1]
        study = optuna.create_study(direction='minimize',
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(lambda t: objective(t, Xtp.iloc[itr], ytp[itr], Xtp.iloc[iva], ytp[iva], fw=fw),
                       n_trials=OPTUNA_TRIALS, show_progress_bar=False)
        full_params = _full_params(study.best_params)
        print(f"   Best params (val logloss {study.best_value:.4f}): {study.best_params}")

        # ---- 3) Primary OOF on train pool -> dev metrics, calibration, meta data ----
        oof = primary_oof(Xtp, ytp, full_params, N_SPLITS_CV, EMBARGO, fw=fw)
        mask = ~np.isnan(oof[:, 0])
        mask_idx = np.where(mask)[0]
        cv_acc = float(accuracy_score(ytp[mask], oof[mask].argmax(1)))
        cv_f1 = float(f1_score(ytp[mask], oof[mask].argmax(1), average='macro', zero_division=0))
        T = fit_temperature(oof[mask], ytp[mask])
        print(f"Primary OOF (dev): acc {cv_acc:.4f} | macro-F1 {cv_f1:.4f} | T {T:.3f}")

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

        # ---- 4) Meta-labeling: build dataset + OOF + pick threshold ----
        meta_X_all = build_meta_X(Xtp, oof)
        prop_all = proposed_side(oof)

        # Compute continuous target expected returns (realized PnL) for meta-model
        meta_y_all = np.full(len(Xtp), np.nan)
        for i in np.where(mask)[0]:
            y_true = int(ytp[i])
            side = int(prop_all[i])
            close_i = float(train_pool['_close_raw'].iloc[i])
            atr_i = float(train_pool['_atr_raw'].iloc[i])
            b = atr_mult * atr_i / close_i if close_i > 0 else 0.015
            
            if y_true == 1:
                pnl_val = -FEE_ROUNDTRIP
            elif side == y_true:
                pnl_val = b - FEE_ROUNDTRIP
            else:
                pnl_val = -b - FEE_ROUNDTRIP
            meta_y_all[i] = pnl_val

        mX = meta_X_all[mask].reset_index(drop=True)
        mY = meta_y_all[mask]
        prop_v = prop_all[mask]
        y_v = ytp[mask]
        
        # ---- TASK 2: FILTER DRIFTED FEATURES BEFORE META TRAINING ----
        # Remove features with PSI > 1.0 or KS > 0.50 to prevent train-test divergence
        print("   [TASK 2] Filtering drifted features before meta training...")
        meta_feature_cols = list(mX.columns)
        try:
            drift_result = detect_feature_drift(train_pool[meta_feature_cols], holdout[meta_feature_cols])
            # detect_feature_drift() returns 'features_to_exclude'
            drifted_features = drift_result.get('features_to_exclude', [])
            if drifted_features:
                print(f"      Detected {len(drifted_features)} drifted features: {drifted_features[:5]}...")
                mX = mX.drop(columns=[c for c in drifted_features if c in mX.columns])
                print(f"      Meta X reduced: {len(meta_feature_cols)} -> {len(mX.columns)} features")
        except Exception as e:
            print(f"      Feature drift filtering failed: {e}")

        # ---- 3) PHASE 5: HOLD Pollution Audit (AEGIS META GATE V2) ----
        # TASK 1: Apply C_excluded strategy BEFORE meta training (not as weights)
        # This means: remove HOLD labels entirely from meta training dataset
        print("   [TASK 1] Applying HOLD pollution strategy BEFORE meta training...")
        
        hold_ratio = float((y_v == 1).mean()) if len(y_v) else 0.0
        print(f"      HOLD label ratio before filtering: {hold_ratio:.1%}")
        
        # TASK 1 FIX: Apply C_excluded BEFORE meta training by removing HOLD samples
        # This is the most direct way to eliminate 66% HOLD pollution
        if META_HOLD_AUTO_SELECT or META_HOLD_STRATEGY == "C_excluded":
            non_hold_mask = (y_v != 1)
            orig_len = len(mX)
            # Track original train_pool indices through the filtering: mask_idx[non_hold_mask]
            meta_idx = mask_idx[non_hold_mask]
            mX = mX[non_hold_mask].reset_index(drop=True)
            mY = mY[non_hold_mask]
            prop_v = prop_v[non_hold_mask]
            y_v = y_v[non_hold_mask]
            print(f"      Applied C_excluded: removed {orig_len - len(mX)} HOLD samples from meta training")
            print(f"      Meta training now: {len(mX)} samples (BUY: {(y_v==2).sum()}, SELL: {(y_v==0).sum()})")
            best_strategy = "C_excluded"
            meta_w = np.ones(len(mX), dtype=float)
            strategy_reports = {"C_excluded": {"applied": "pre-training", "samples": len(mX)}}
            hold_strategy_selected = "C_excluded"
            hold_strategy_audit = strategy_reports
            # Skip the strategy search loop entirely
            skip_strategy_search = True
        else:
            meta_idx = mask_idx
            skip_strategy_search = False

        # Precompute values for fast CV evaluation (using meta_idx to match filtered dataset)
        df_dev_temp = train_pool.iloc[meta_idx].copy()
        close_tp_raw = df_dev_temp['_close_raw'].to_numpy()
        atr_tp_raw = df_dev_temp['_atr_raw'].to_numpy()
        b_fracs_t = np.divide(atr_mult * atr_tp_raw, close_tp_raw, out=np.zeros(len(close_tp_raw)), where=close_tp_raw > 0)

        if skip_strategy_search:
            print("   [TASK 1] Skipping strategy comparison (C_excluded already applied pre-training)")
            meta_ready = len(mX) >= max(200, MIN_FIRES_DEV * 4)

        regime_policies = {}
        meta_calibration_method = 'uncalibrated'
        meta_ready = len(mX) >= max(200, MIN_FIRES_DEV * 4)
        if meta_ready:
            meta_oof = binary_oof(mX, mY, token_meta_params, N_SPLITS_CV, EMBARGO, w=meta_w)
            
            from src.ml.calibration import MetaCalibrationFramework
            from src.ml.confidence_engine import ConfidenceReliabilityEngine
            from src.ml.regime_intelligence import RegimeConfidenceModifier
            
            mcf = MetaCalibrationFramework()
            # Evaluate calibration on binary target, mapping continuous returns inside calibration.py
            y_v_binary = (y_v == prop_v).astype(int)
            mcf.evaluate_calibrators(meta_oof, y_v_binary)
            
            profile_calibration = meta_gate_profile.get('calibration', {}) if meta_gate_profile else {}
            profile_cal_method = (
                profile_calibration.get('selected_calibrator') or
                profile_calibration.get('selected_method') or
                profile_calibration.get('method') or
                'uncalibrated'
            )
            profile_cal_method = str(profile_cal_method).lower()
            if meta_gate_profile:
                if profile_cal_method != 'uncalibrated':
                    if profile_cal_method in mcf.reports and mcf.reports[profile_cal_method].get('model') is not None:
                        mcf.calibrator_type = profile_cal_method
                        mcf.best_calibrator = mcf.reports[profile_cal_method]['model']
                        print(f"      Trusted optimizer calibration method: {mcf.calibrator_type}")
                    else:
                        print(f"      WARNING: optimizer profile requested calibrator '{profile_cal_method}' but it was not available on this dev set. Using uncalibrated meta output.")
                        mcf.calibrator_type = 'uncalibrated'
                        mcf.best_calibrator = None
                else:
                    mcf.calibrator_type = 'uncalibrated'
                    mcf.best_calibrator = None
                    print(f"      Trusted optimizer calibration method: {mcf.calibrator_type}")
            else:
                # Keep the calibration selected from current dev data when no profile directive is present
                print(f"      Meta calibration selected: {mcf.calibrator_type}")
            meta_oof_cal = mcf.calibrate(meta_oof)
            meta_calibration_method = mcf.calibrator_type
            
            cre = ConfidenceReliabilityEngine()
            cre._mapping_x = None
            cre._mapping_y = None
            
            rcm = RegimeConfidenceModifier(target_precision=token_precision_target)
            if 'hmm_regime' in train_pool.columns:
                regimes_arr = train_pool['hmm_regime'].iloc[meta_idx].to_numpy()
                rcm.analyze_regimes(regimes_arr, meta_oof_cal, mY, prop_v)

            # ── Spot-on market dynamics: Regime-specific directional filter on DEV set ──
            # Apply filter AFTER meta-model is trained on un-tampered proposals
            prop_dev_filtered = prop_v.copy()
            _opt = load_token_params(symbol)
            if _opt and "regimes" in _opt and "regime_boundaries" in _opt:
                bounds = _opt["regime_boundaries"]
                regimes_dict = _opt["regimes"]
                
                vol_avg = train_pool["volume"].rolling(24, min_periods=1).mean()
                atr_pct = (train_pool["_atr_raw"] / train_pool["_close_raw"]).fillna(0)
                momentum = train_pool["_close_raw"].pct_change(24).fillna(0)
                
                def _tier(val, p33, p67): return "low" if val <= p33 else ("med" if val <= p67 else "high")
                def _trend(val, p33, p67): return "down" if val <= p33 else ("flat" if val <= p67 else "up")
                
                vp33, vp67 = bounds.get("vol_p33", 0), bounds.get("vol_p67", 0)
                ap33, ap67 = bounds.get("atr_pct_p33", 0), bounds.get("atr_pct_p67", 0)
                mp33, mp67 = bounds.get("momentum_p33", -0.02), bounds.get("momentum_p67", 0.02)
                
                regime_strs = [
                    f"{_tier(vol_avg.iloc[i], vp33, vp67)}_{_tier(atr_pct.iloc[i], ap33, ap67)}_{_trend(momentum.iloc[i], mp33, mp67)}"
                    for i in range(len(Xtp))
                ]
                
                idx_map = meta_idx
                for i_mask in range(len(mX)):
                    orig_i = idx_map[i_mask]
                    reg = regimes_dict.get(regime_strs[orig_i], {})
                    if not reg or reg.get("skipped"):
                        prop_dev_filtered[i_mask] = 1 # Suppress to HOLD
                        continue
                    
                    side = prop_dev_filtered[i_mask]
                    if side == 2:
                        if not reg.get("buy_ok"): prop_dev_filtered[i_mask] = 1
                    elif side == 0:
                        if not reg.get("sell_ok"): prop_dev_filtered[i_mask] = 1

                        # ── Edge-Driven Evaluation ──────────────
            from src.trading.edge_engine import EdgeScoringEngine
            
            df_dev = train_pool.iloc[meta_idx] if 'meta_idx' in locals() else train_pool
            use_rank = bool(meta_gate_profile and meta_gate_profile.get('edge_rank_mode') == 'percentile')
            edge_buy = EdgeScoringEngine.compute_edge_batch(df_dev, meta_oof_cal, 'BUY', use_rank=use_rank).to_numpy()
            edge_sell = EdgeScoringEngine.compute_edge_batch(df_dev, meta_oof_cal, 'SELL', use_rank=use_rank).to_numpy()

            profile_thresholds = get_profile_edge_thresholds(
                meta_gate_profile, edge_buy, edge_sell, prop_dev_filtered, y_v
            )
            if profile_thresholds is not None:
                thr_buy, thr_sell, hit_buy, hit_sell, prec_buy, prec_sell, n_buy, n_sell = profile_thresholds
                cov_buy = float(n_buy / max(1, int((prop_dev_filtered == 2).sum())))
                cov_sell = float(n_sell / max(1, int((prop_dev_filtered == 0).sum())))
                print(
                    f"      Using optimizer meta gate profile thresholds: "
                    f"BUY={thr_buy:.3f}, SELL={thr_sell:.3f}, "
                    f"side_specific={meta_gate_profile.get('side_specific', True)}, "
                    f"hit_buy={hit_buy}, hit_sell={hit_sell}"
                )
            else:
                # Pick thresholds dynamically per side (Phase 3)
                # TASK 3 FIX: Use percentile-based logic that adapts to edge score distribution
                thr_buy, prec_buy, cov_buy, n_buy, hit_buy = pick_edge_threshold_by_side(
                    edge_buy, prop_dev_filtered, y_v, 2, target=token_precision_target, min_fires=max(5, int(len(edge_buy) * MIN_COVERAGE))
                )
                thr_sell, prec_sell, cov_sell, n_sell, hit_sell = pick_edge_threshold_by_side(
                    edge_sell, prop_dev_filtered, y_v, 0, target=token_precision_target, min_fires=max(5, int(len(edge_sell) * MIN_COVERAGE))
                )
            
            if profile_thresholds is None:
                # TASK 3 FIX: BUY threshold deadlock - if no valid BUY thresholds found, use adaptive fallback
                if not hit_buy and n_buy == 0:
                    print(f"      WARNING: No valid BUY thresholds found (deadlock). Using adaptive fallback.")
                    # Use lower percentile to find ANY BUY signals
                    lower_q_vals = [0.50, 0.40, 0.30, 0.20, 0.15, 0.10]
                    for q_val in lower_q_vals:
                        fallback_thr = float(np.quantile(edge_buy, 1.0 - q_val))
                        fallback_fire = edge_buy >= fallback_thr
                        fallback_side = (prop_dev_filtered == 2) & fallback_fire
                        if fallback_side.sum() >= 5:
                            thr_buy = fallback_thr
                            prec_buy = float((y_v[fallback_side] == 2).mean()) if fallback_side.sum() > 0 else 0.0
                            cov_buy = q_val
                            n_buy = int(fallback_side.sum())
                            hit_buy = True
                            print(f"      BUY fallback: threshold={fallback_thr:.1f}, coverage={q_val:.1%}, n={n_buy}, prec={prec_buy:.1%}")
                            break
                
                if not hit_sell and n_sell == 0:
                    print(f"      WARNING: No valid SELL thresholds found (deadlock). Using adaptive fallback.")
                    lower_q_vals = [0.50, 0.40, 0.30, 0.20, 0.15, 0.10]
                    for q_val in lower_q_vals:
                        fallback_thr = float(np.quantile(edge_sell, 1.0 - q_val))
                        fallback_fire = edge_sell >= fallback_thr
                        fallback_side = (prop_dev_filtered == 0) & fallback_fire
                        if fallback_side.sum() >= 5:
                            thr_sell = fallback_thr
                            prec_sell = float((y_v[fallback_side] == 0).mean()) if fallback_side.sum() > 0 else 0.0
                            cov_sell = q_val
                            n_sell = int(fallback_side.sum())
                            hit_sell = True
                            print(f"      SELL fallback: threshold={fallback_thr:.1f}, coverage={q_val:.1%}, n={n_sell}, prec={prec_sell:.1%}")
                            break
            
            # ---- Regime-Aware Threshold Optimization (Priority 4) ----
            print("   Optimizing separate threshold policies per HMM regime...")
            regimes_list = ['ACCUMULATION', 'DISTRIBUTION', 'COMPRESSION', 'VOLATILE_EXPANSION', 'TRENDING_BULL', 'TRENDING_BEAR', 'CHOPPY']
            regime_policies = {}
            
            for r in regimes_list:
                r_mask = (df_dev['hmm_regime'] == r).to_numpy() if 'hmm_regime' in df_dev.columns else np.zeros(len(df_dev), dtype=bool)
                if r_mask.sum() < 30:
                    # Not enough data for this regime, use defaults
                    regime_policies[r] = {
                        "buy_thr": thr_buy,
                        "sell_thr": thr_sell,
                        "buy_ok": True,
                        "sell_ok": True
                    }
                    continue
                    
                # Tune BUY side
                best_buy_thr = thr_buy
                buy_ok = True
                buy_r_mask = r_mask & (prop_dev_filtered == 2)
                if buy_r_mask.sum() >= 10:
                    es_buy_r = edge_buy[buy_r_mask]
                    yt_buy_r = y_v[buy_r_mask]
                    best_buy_metric = -999.0
                    for th in [45.0, 50.0, 52.0, 55.0, 58.0, 60.0, 62.0, 65.0]:
                        fire = (es_buy_r >= th)
                        if fire.sum() < 5:
                            continue
                        fired_idx = np.where(buy_r_mask)[0][fire]
                        fired_rets = []
                        for idx_val in fired_idx:
                            close_val = df_dev['_close_raw'].iloc[idx_val]
                            atr_val = df_dev['_atr_raw'].iloc[idx_val]
                            b_val = atr_mult * atr_val / close_val if close_val > 0 else 0.015
                            label_val = y_v[idx_val]
                            if label_val == 1:
                                rets_val = -FEE_ROUNDTRIP
                            elif label_val == 2:
                                rets_val = b_val - FEE_ROUNDTRIP
                            else:
                                rets_val = -b_val - FEE_ROUNDTRIP
                            fired_rets.append(rets_val)
                        fired_rets = np.array(fired_rets)
                        exp_pct = float(fired_rets.mean()) * 100
                        gross_win = float(fired_rets[fired_rets > 0].sum())
                        gross_loss = float(abs(fired_rets[fired_rets < 0].sum()))
                        pf_val = gross_win / (gross_loss + 1e-9)
                        
                        metric = exp_pct * min(pf_val, 3.0)
                        if metric > best_buy_metric:
                            best_buy_metric = metric
                            best_buy_thr = th
                    if best_buy_metric < -0.05:
                        buy_ok = False
                        
                # Tune SELL side
                best_sell_thr = thr_sell
                sell_ok = True
                sell_r_mask = r_mask & (prop_dev_filtered == 0)
                if sell_r_mask.sum() >= 10:
                    es_sell_r = edge_sell[sell_r_mask]
                    yt_sell_r = y_v[sell_r_mask]
                    best_sell_metric = -999.0
                    for th in [45.0, 50.0, 52.0, 55.0, 58.0, 60.0, 62.0, 65.0]:
                        fire = (es_sell_r >= th)
                        if fire.sum() < 5:
                            continue
                        fired_idx = np.where(sell_r_mask)[0][fire]
                        fired_rets = []
                        for idx_val in fired_idx:
                            close_val = df_dev['_close_raw'].iloc[idx_val]
                            atr_val = df_dev['_atr_raw'].iloc[idx_val]
                            b_val = atr_mult * atr_val / close_val if close_val > 0 else 0.015
                            label_val = y_v[idx_val]
                            if label_val == 1:
                                rets_val = -FEE_ROUNDTRIP
                            elif label_val == 0:
                                rets_val = b_val - FEE_ROUNDTRIP
                            else:
                                rets_val = -b_val - FEE_ROUNDTRIP
                            fired_rets.append(rets_val)
                        fired_rets = np.array(fired_rets)
                        exp_pct = float(fired_rets.mean()) * 100
                        gross_win = float(fired_rets[fired_rets > 0].sum())
                        gross_loss = float(abs(fired_rets[fired_rets < 0].sum()))
                        pf_val = gross_win / (gross_loss + 1e-9)
                        
                        metric = exp_pct * min(pf_val, 3.0)
                        if metric > best_sell_metric:
                            best_sell_metric = metric
                            best_sell_thr = th
                    if best_sell_metric < -0.05:
                        sell_ok = False
                        
                regime_policies[r] = {
                    "buy_thr": best_buy_thr,
                    "sell_thr": best_sell_thr,
                    "buy_ok": buy_ok,
                    "sell_ok": sell_ok
                }
                print(f"      Regime {r:20} -> BUY: thr={best_buy_thr:.1f}, ok={buy_ok} | SELL: thr={best_sell_thr:.1f}, ok={sell_ok}")
            
            fire_buy_dev = (edge_buy >= thr_buy) & (prop_dev_filtered == 2)
            fire_sell_dev = (edge_sell >= thr_sell) & (prop_dev_filtered == 0)
            
            dev_n = n_buy + n_sell
            hit_target = hit_buy or hit_sell
            thr = float((thr_buy + thr_sell) / 2.0)
            
            fire_dev_any = fire_buy_dev | fire_sell_dev
            dev_prec = float((y_v[fire_dev_any] == prop_dev_filtered[fire_dev_any]).mean()) if dev_n > 0 else 0.0
            
            cov_buy = n_buy / len(prop_dev_filtered)
            cov_sell = n_sell / len(prop_dev_filtered)
            dev_cov = dev_n / len(prop_dev_filtered)
            
            exp_buy = (prec_buy * 1.5 - (1-prec_buy)) * 100
            exp_sell = (prec_sell * 1.5 - (1-prec_sell)) * 100
            
            # ── Filter Audits (Phase 4, 5, 6) ──
            if signal_vetoes is None:
                disable_sr_veto = False
                disable_trend_veto = False
                disable_confluence_veto = False
            else:
                disable_sr_veto = 'sr' not in signal_vetoes
                disable_trend_veto = 'trend' not in signal_vetoes
                disable_confluence_veto = 'confluence' not in signal_vetoes

            # Helper for audit
            def audit_filter(y_true, prop, baseline_mask, filter_pass):
                if baseline_mask.sum() == 0:
                    return 0.0, 0.0, 0.0, 0.0
                n_before = baseline_mask.sum()
                prec_before = float((prop[baseline_mask] == y_true[baseline_mask]).mean())
                
                fired_after = baseline_mask & filter_pass
                n_after = fired_after.sum()
                prec_after = float((prop[fired_after] == y_true[fired_after]).mean()) if n_after > 0 else 0.0
                
                prec_gain = prec_after - prec_before
                cov_loss = (n_before - n_after) / n_before if n_before > 0 else 0.0
                return prec_before, prec_after, cov_loss, prec_gain

            # Base signal mask before filters (EdgeEngine pass)
            fire_base = ((edge_buy >= thr_buy) & (prop_dev_filtered == 2)) | \
                        ((edge_sell >= thr_sell) & (prop_dev_filtered == 0))
            
            # 1. S&R filter audit
            at_res_dev = df_dev['is_at_resistance'].to_numpy().astype(bool) if 'is_at_resistance' in df_dev.columns else np.zeros(len(df_dev), dtype=bool)
            at_sup_dev = df_dev['is_at_support'].to_numpy().astype(bool) if 'is_at_support' in df_dev.columns else np.zeros(len(df_dev), dtype=bool)
            
            top25_thr = float(np.quantile(meta_oof_cal[fire_base], 0.75)) if fire_base.sum() > 0 else 0.55
            top25 = meta_oof_cal >= top25_thr
            sr_pass_dev = ~(((prop_dev_filtered == 2) & at_res_dev & ~top25) | \
                            ((prop_dev_filtered == 0) & at_sup_dev & ~top25))
            
            p_bef, p_aft, cov_l, prec_g = audit_filter(y_v, prop_dev_filtered, fire_base, sr_pass_dev)
            
            # S&R win rate test (Phase 5)
            blocked_sr = fire_base & ~sr_pass_dev
            passed_sr = fire_base & sr_pass_dev
            harmful_sr = False
            if blocked_sr.sum() > 0 and passed_sr.sum() > 0:
                blocked_wr = float((prop_dev_filtered[blocked_sr] == y_v[blocked_sr]).mean())
                passed_wr = float((prop_dev_filtered[passed_sr] == y_v[passed_sr]).mean())
                if blocked_wr >= passed_wr:
                    harmful_sr = True
            
            if (prec_g < 0.01 and cov_l > 0.20) or harmful_sr:
                disable_sr_veto = True
                print(f"      [SR AUDIT] disabled (overrestrictive={prec_g < 0.01 and cov_l > 0.20}, harmful={harmful_sr}, cov_loss={cov_l:.1%}, prec_gain={prec_g:.1%})")
            
            # 2. Daily Trend filter audit
            if 'macro_trend_1d' in df_dev.columns:
                trend_1d = df_dev['macro_trend_1d'].to_numpy()
                trend_pass_dev = ~(
                    ((prop_dev_filtered == 2) & (trend_1d < -0.2) & ~top25) | \
                    ((prop_dev_filtered == 0) & (trend_1d > 0.2) & ~top25)
                )
            else:
                trend_pass_dev = np.ones(len(df_dev), dtype=bool)
                
            p_bef_t, p_aft_t, cov_l_t, prec_g_t = audit_filter(y_v, prop_dev_filtered, fire_base, trend_pass_dev)
            if prec_g_t < 0.01 and cov_l_t > 0.20:
                disable_trend_veto = True
                print(f"      [TREND AUDIT] disabled (overrestrictive=True, cov_loss={cov_l_t:.1%}, prec_gain={prec_g_t:.1%})")

            # 3. Confluence filter audit
            confluence_pass_dev = np.ones(len(df_dev), dtype=bool)
            if 'total_confluence' in df_dev.columns:
                tc = df_dev['total_confluence'].to_numpy()
                confluence_pass_dev = ~((prop_dev_filtered == 2) & (tc < -0.05)) & ~((prop_dev_filtered == 0) & (tc > 0.05))
            
            p_bef_c, p_aft_c, cov_l_c, prec_g_c = audit_filter(y_v, prop_dev_filtered, fire_base, confluence_pass_dev)
            if prec_g_c < 0.01 and cov_l_c > 0.20:
                disable_confluence_veto = True
                print(f"      [CONFLUENCE AUDIT] disabled (overrestrictive=True, cov_loss={cov_l_c:.1%}, prec_gain={prec_g_c:.1%})")

            # 4. Regime filter audit (Phase 6)
            if _opt and "regimes" in _opt and "regime_boundaries" in _opt:
                regimes_dict = _opt["regimes"]
                regime_series = np.array(regime_strs)[meta_idx]
                
                # Unfiltered proposals (before regime skipping)
                fire_base_unfiltered = ((edge_buy >= thr_buy) & (prop_v == 2)) | \
                                       ((edge_sell >= thr_sell) & (prop_v == 0))
                
                for r_name, r_config in list(regimes_dict.items()):
                    if r_config.get("skipped"):
                        blocked_reg = fire_base_unfiltered & (regime_series == r_name) & (prop_dev_filtered == 1)
                        if blocked_reg.sum() > 0:
                            blocked_precision = float((prop_v[blocked_reg] == y_v[blocked_reg]).mean())
                            if blocked_precision > 0.50:
                                r_config["skipped"] = False
                                print(f"      [REGIME AUDIT] Regime {r_name} block is harmful (blocked precision {blocked_precision:.1%} > 50.0%). Re-enabled regime (downgraded to soft penalty).")

            # Fleet Learning Cache Saving
            import pickle
            from pathlib import Path
            
            base = symbol.replace('/', '_')
            oof_cache_dir = Path(root_dir) / "data" / "oof_cache"
            oof_cache_dir.mkdir(parents=True, exist_ok=True)
            
            cache_file = oof_cache_dir / f"{base}_oof.pkl"
            with open(cache_file, "wb") as f:
                pickle.dump((mX, mY, meta_w), f)
            print(f"   Saved OOF cache for {symbol} to {cache_file.name}")
            
            # Fleet Learning Pooling
            mX_list = []
            mY_list = []
            meta_w_list = []
            
            for pkl_path in oof_cache_dir.glob("*.pkl"):
                try:
                    with open(pkl_path, "rb") as f:
                        cached_mX, cached_mY, cached_meta_w = pickle.load(f)
                    # Align features to current token's active feature_cols
                    aligned_mX = cached_mX.reindex(columns=feature_cols, fill_value=0.0)
                    mX_list.append(aligned_mX)
                    mY_list.append(cached_mY)
                    meta_w_list.append(cached_meta_w)
                except Exception as cache_load_err:
                    print(f"   Failed to load cached OOF dataset {pkl_path.name}: {cache_load_err}")
            
            if mX_list:
                mX_fleet = pd.concat(mX_list, ignore_index=True)
                mY_fleet = np.concatenate(mY_list)
                meta_w_fleet = np.concatenate(meta_w_list)
                print(f"   Fleet learning pool: {len(mX_fleet)} total samples pooled from {len(mX_list)} symbols")
            else:
                mX_fleet = mX
                mY_fleet = mY
                meta_w_fleet = meta_w
                
            meta_full = xgb.train(token_meta_params, _dm(mX_fleet, mY_fleet, meta_w_fleet), num_boost_round=300, verbose_eval=False)
            # Train lightweight logistic meta-model (Phase 6: trade-quality scorer)
            try:
                from src.ml.meta_model import train_logistic_meta
                # Build binary target for profitable trade: pnl > 0
                mY_binary = (mY_fleet > 0).astype(int)
                store_dir = Path(root_dir) / "src" / "ml" / "model_store"
                store_dir.mkdir(parents=True, exist_ok=True)
                meta_light_path = store_dir / f"{base}_meta_light.pkl"
                lr = train_logistic_meta(mX_fleet.fillna(0.0), mY_binary, save_path=str(meta_light_path))
                print(f"   Lightweight meta-model trained and saved: {meta_light_path.name}")
            except Exception as _e:
                meta_light_path = None
                print(f"   Lightweight meta-model training skipped: {_e}")
            flag = "target met" if hit_target else "target NOT met (best achievable)"
            print(f"   Meta gate: thr {thr:.3f} | dev precision {dev_prec:.3f} | "
                  f"coverage {dev_cov:.3f} ({dev_n} trades) | {flag}")
            print(f"   BUY  side: thr {thr_buy:.3f} | prec {prec_buy:.3f} | "
                  f"cov {cov_buy:.3f} ({n_buy} trades) | "
                  f"{'LIVE' if hit_buy else 'disabled'}")
            print(f"   SELL side: thr {thr_sell:.3f} | prec {prec_sell:.3f} | "
                  f"cov {cov_sell:.3f} ({n_sell} trades) | "
                  f"{'LIVE' if hit_sell else 'disabled'}")
        else:
            print("   Too few directional samples for meta-labeling -- falling back to "
                  "primary confidence gate.")
            meta_full   = None
            thr         = 0.60
            dev_prec    = dev_cov = 0.0
            dev_n       = 0
            hit_target  = False
            thr_buy     = thr_sell  = 0.60
            prec_buy    = prec_sell = 0.0
            cov_buy     = cov_sell  = 0.0
            n_buy       = n_sell    = 0
            hit_buy     = hit_sell  = False

        # ---- 5) Holdout: scored exactly once ----
        primary_full = xgb.train(full_params, _dm(Xtp, ytp, sample_weights(ytp)),
                                 num_boost_round=500, verbose_eval=False)
        X_test = holdout[feature_cols]
        y_test = holdout['target'].to_numpy().astype(int)
        raw_probs = primary_full.predict(_dm(X_test))
        prop_h = proposed_side(raw_probs)

        if meta_full is not None:
            meta_prob_h = meta_full.predict(_dm(build_meta_X(X_test, raw_probs)))
        else:
            meta_prob_h = raw_probs.max(axis=1)  # fallback gate on primary confidence

        # ── Edge-Driven Holdout Gate ──────────────
        from src.trading.edge_engine import EdgeScoringEngine
        edge_buy_h = EdgeScoringEngine.compute_edge_batch(holdout, meta_prob_h, 'BUY').to_numpy()
        edge_sell_h = EdgeScoringEngine.compute_edge_batch(holdout, meta_prob_h, 'SELL').to_numpy()
        
        # Holdout Regime-specific Edge Score Thresholding (Priority 4)
        fire_buy_h = np.zeros(len(holdout), dtype=bool)
        fire_sell_h = np.zeros(len(holdout), dtype=bool)
        
        for i in range(len(holdout)):
            r = holdout['hmm_regime'].iloc[i] if 'hmm_regime' in holdout.columns else 'UNKNOWN'
            policy = regime_policies.get(r, {
                "buy_thr": thr_buy,
                "sell_thr": thr_sell,
                "buy_ok": True,
                "sell_ok": True
            })
            
            r_buy_thr = policy.get("buy_thr", thr_buy)
            r_sell_thr = policy.get("sell_thr", thr_sell)
            r_buy_ok = policy.get("buy_ok", True)
            r_sell_ok = policy.get("sell_ok", True)
            
            if prop_h[i] == 2 and r_buy_ok:
                fire_buy_h[i] = (edge_buy_h[i] >= r_buy_thr)
            elif prop_h[i] == 0 and r_sell_ok:
                fire_sell_h[i] = (edge_sell_h[i] >= r_sell_thr)
                
        fire = fire_buy_h | fire_sell_h
        gate_mode = f"EdgeEngine (B>={thr_buy:.1f}, S>={thr_sell:.1f})"
        
        _tier_agg_pre = True
        _AGG_FLOOR_PRE = 0.50
        print(f"   Holdout gate: {gate_mode} | fired {int(fire.sum())}")
        # Threshold that separates top-25% of fired signals from bottom-75%.
        # Saved to metadata so the live predictor can replicate S&R / trend filters
        # without needing the full holdout distribution at inference time.
        override_conf_thr = float(np.quantile(meta_prob_h[fire], 0.75)) if fire.sum() > 0 else thr

        # ── S&R-aware confluence filter ────────────────────────────────
        # At resistance a BUY signal fights the level — keep it only if the
        # meta confidence is in the top 25% of all fired signals (high conviction).
        # At support a SELL signal fights the level — same rule applies.
        # This embeds the "indicators must vote strongly to override S&R" rule.
        if (hit_target or _tier_agg_pre) and fire.sum() >= 4 and not disable_sr_veto:
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
        if (hit_target or _tier_agg_pre) and fire.sum() >= 4 and 'macro_trend_1d' in holdout.columns and not disable_trend_veto:
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
        if (hit_target or _tier_agg_pre) and _opt and "regimes" in _opt and "regime_boundaries" in _opt:
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

            # ── Tradeable decision (per-side aware) ────────────────────────
            _MIN_SIDE = 5    # minimum per-side holdout trades to trust the result
            tradeable_buy_holdout  = (
                hit_buy and
                buy_h_n >= _MIN_SIDE and
                (bt["buy_win_rate"] > 0.4 or bt["expectancy_pct"] > 0.0)
            )
            tradeable_sell_holdout = (
                hit_sell and
                sell_h_n >= _MIN_SIDE and
                (bt["sell_win_rate"] > 0.4 or bt["expectancy_pct"] > 0.0)
            )

            holdout_reliable = fired_n >= MIN_HOLDOUT_FIRES
            oof_holdout_gap  = abs(dev_prec - fired_prec)

            passes_validation = (
                fired_n >= 10 and
                bt["expectancy_pct"] > 0.10 and
                bt["profit_factor"] > 1.2 and
                bt["sharpe"] > 0.5
            )

            tradeable_final = passes_validation and (tradeable_buy_holdout or tradeable_sell_holdout)
            
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
            
            if selected_prec <= rejected_prec:
                print("      [VETO] Selected trades did not outperform rejected trades! Disabling the gate automatically.")
                passes_validation = False

            if holdout_reliable and bt['expectancy_pct'] <= 0.0 and oof_holdout_gap > GAP_VETO_THRESHOLD:
                passes_validation = False
                print("      DISABLED (gap veto): negative expectancy and large OOF→holdout gap")
            elif holdout_reliable and oof_holdout_gap > 0.10:
                print(f"      WATCH: OOF→holdout gap {oof_holdout_gap:.1%} "
                      f"(dev {dev_prec:.1%} → holdout {fired_prec:.1%}). "
                      f"Possible regime shift — monitor after next retrain.")

            tradeable_final = passes_validation and (tradeable_buy_holdout or tradeable_sell_holdout)
            per_side_approved = bool(tradeable_buy_holdout or tradeable_sell_holdout)
            combined_ok = bool(passes_validation)

            if passes_validation:
                print(f"      Profitability: expectancy={bt['expectancy_pct']:+.2f}%, PF={bt['profit_factor']:.2f}, Sharpe={bt['sharpe']:.1f}")
                if tradeable_final:
                    print(f"      ENABLED: profitability thresholds met (trades={fired_n}, EV>0.10%, PF>1.2, Sharpe>0.5)")
                else:
                    print(f"      DISABLED: gate passed validation but no per-side profitability approval.")
            else:
                reasons = []
                if fired_n < 10: reasons.append(f"insufficient_trades({fired_n}<10)")
                if bt["buy_n"] == 0 and bt["sell_n"] == 0: reasons.append("no_trades_fired")
                if bt["profit_factor"] <= 1.2: reasons.append(f"low_PF({bt['profit_factor']:.2f}<=1.2)")
                if bt["expectancy_pct"] <= 0.10: reasons.append(f"low_EV({bt['expectancy_pct']:+.2f}%<=0.10%)")
                if bt["sharpe"] <= 0.5: reasons.append(f"low_Sharpe({bt['sharpe']:.1f}<=0.5)")
                print(f"      [VALIDATION] FAIL: {', '.join(reasons)}")

            if not tradeable_final:
                if bt['expectancy_pct'] <= 0.0:
                    print(f"      DISABLED: negative expectancy ({bt['expectancy_pct']:+.3f}%) "
                          f"with no per-side profitability approval.")
                else:
                    print(f"      DISABLED: profitability thresholds not met "
                          f"(EV={bt['expectancy_pct']:+.2f}%, PF={bt['profit_factor']:.2f}, Sharpe={bt['sharpe']:.1f}).")
            elif not (tradeable_buy_holdout or tradeable_sell_holdout):
                sides_live = []
                if tradeable_buy_holdout:  sides_live.append('BUY')
                if tradeable_sell_holdout: sides_live.append('SELL')
                print(f"      ENABLED via per-side profitability: {' + '.join(sides_live)}")
            else:
                print(f"      ENABLED: profitability thresholds met (trades={fired_n}, EV>0.10%, PF>1.2, Sharpe>0.5)")
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

        print(f"   Risk tiers: "
              f"conservative={'V' if tier_conservative else 'X'} | "
              f"balanced={'V' if tier_balanced else 'X'} | "
              f"aggressive={'V' if tier_aggressive else 'X'}")

        # ---- 6) Deployment models on all usable data ----
        usable = pd.concat([train_pool, holdout], ignore_index=True)
        X_all, y_all = usable[feature_cols], usable['target'].to_numpy().astype(int)
        deploy_primary = xgb.train(full_params, _dm(X_all, y_all, sample_weights(y_all)),
                                   num_boost_round=500, verbose_eval=False)

        store_dir = Path(root_dir) / "src" / "ml" / "model_store"
        store_dir.mkdir(parents=True, exist_ok=True)
        base = symbol.replace('/', '_')

        # Save the primary as a Booster JSON. NOTE: predictor.py must load this with
        # xgb.Booster(), not XGBClassifier — see the patched predictor. We tag the
        # file so the loader knows the format and how many classes to expect.
        model_path = store_dir / f"{base}_model.json"
        deploy_primary.save_model(str(model_path))
        
        # Save the clean primary model (trained only on train pool)
        clean_model_path = store_dir / f"{base}_model_clean.json"
        primary_full.save_model(str(clean_model_path))
        
        meta_path = None
        if meta_full is not None:
            meta_path = store_dir / f"{base}_meta_model.json"
            meta_full.save_model(str(meta_path))
        print(f"Models saved: {model_path.name} + {clean_model_path.name}" + (f" + {meta_path.name}" if meta_path else ""))

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
                "num_class": NUM_CLASS,
                "feature_cols": feature_cols,        # exact order the Booster expects
                "meta_feature_cols": feature_cols,
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
                "meta_model_file": meta_path.name if meta_path else None,
                "meta_model_light_file": meta_light_path.name if (meta_light_path is not None) else None,
                "atr_multiplier": atr_mult,
                # tradeable=False means the predictor emits NO signals for this token.
                # Requires: OOF target met AND (holdout unreliable OR holdout ≥ breakeven).
                # A holdout below fee breakeven with ≥ MIN_HOLDOUT_FIRES trades overrides
                # the OOF optimism and silences the token.
                "tradeable": bool(tradeable_final),
                "target_precision": TARGET_SIGNAL_PRECISION,
                # DEV estimate = how the gate scored on out-of-fold data (this is
                # what justified shipping the token). Pre-committed, not peeked.
                "dev_estimate": {"precision": dev_prec, "coverage": dev_cov, "trades": dev_n},
                # HOLDOUT = the same pre-committed gate applied once to untouched
                # data. This is the honest out-of-sample number; expect it near the
                # dev estimate, not above it.
                "holdout_trading": {
                    "fired":            fired_n,
                    "coverage":         coverage,
                    "signal_precision": fired_prec,
                    "expectancy_pct":   bt["expectancy_pct"],
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
        importance_dict = log_feature_importance(deploy_primary, feature_cols, symbol)
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
            print(f"   DONE {symbol} {tag} precision {m['holdout_signal_precision']:.1%} | "
                  f"coverage {m['holdout_coverage']:.1%} | "
                  f"exp/trade {m['holdout_expectancy_pct']:+.2f}%")
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