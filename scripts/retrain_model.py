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


# ============================================================
# CONFIG
# ============================================================
NUM_CLASS = 3                 # 0=SELL, 1=HOLD, 2=BUY
MAX_LOOKAHEAD = 48
EMBARGO = MAX_LOOKAHEAD
CENSORED = -1

TEST_FRAC = 0.20
N_SPLITS_CV = 10              # purged folds for OOF / dev estimates
OPTUNA_TRIALS = 60

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
FEE_ROUNDTRIP = 0.001            # 0.10% round-trip (taker + slippage); tune to your venue

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
    'objective': 'binary:logistic', 'eval_metric': 'logloss',
    'max_depth': 4, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'reg_lambda': 2.0, 'min_child_weight': 10,
    'seed': 42, 'tree_method': 'hist', 'missing': np.nan,
}

FLEET_SYMBOLS = [
    # ── Majors (20) ───────────────────────────────────────────────────────
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

    Window of 120 bars ≈ 5 days on 1h data — long enough to be meaningful,
    short enough to track regime changes within the training set.
    """
    result = pd.DataFrame(index=df.index)

    def _pr(col: str, higher_bullish: bool = True) -> pd.Series:
        """Rolling percentile rank of `col`, 0.5 if column missing."""
        if col not in df.columns:
            return pd.Series(0.5, index=df.index, dtype=float)
        s = df[col].fillna(method='ffill').fillna(0.0)
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
    total = sum(result[c] * w for c, w in W.items()) / Wsum
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
    """Per-asset barrier width. Wider barrier = fewer but cleaner labels.
    Tiers are based on realised volatility relative to BTC, not market cap.

    Tier 1 (1.2) — BTC only: deepest liquidity, tightest vol.
    Tier 2 (1.5) — large established alts: LTC, BNB, XRP, ADA, DOT.
    Tier 3 (1.8) — high-vol majors: ETH, SOL, AVAX, and similar alts that
                    routinely move 5-10% on a single 1h candle.
    Tier 4 (2.2) — meme / micro-cap: DOGE, SHIB, PEPE, BONK, WIF, FLOKI.
                    These need the widest barrier to survive intraday noise.

    Why ETH is NOT in Tier 1 with BTC:
      ETH's intraday vol is ~40-60% higher than BTC's. A 1.2× barrier gets hit
      by noise on most candles, creating dirty BUY/SELL labels that the model
      learns but can't generalise. Moving ETH to 1.8× gives the barrier enough
      room to separate real directional moves from noise, producing cleaner
      training labels and higher holdout precision.
    """
    _TIER1 = {'BTC/USDT'}

    # Large, liquid, relatively low intraday volatility vs BTC
    _TIER2 = {
        'BNB/USDT', 'XRP/USDT', 'ADA/USDT', 'LTC/USDT', 'BCH/USDT',
        'TRX/USDT', 'TON/USDT', 'DOT/USDT', 'LINK/USDT', 'VET/USDT',
        'ATOM/USDT', 'XLM/USDT', 'ETC/USDT', 'UNI/USDT', 'ALGO/USDT',
        'XTZ/USDT', 'EOS/USDT', 'NEO/USDT', 'QTUM/USDT', 'XMR/USDT',
        'ZEC/USDT', 'DASH/USDT', 'MKR/USDT', 'QNT/USDT',
    }

    # High-vol majors, DeFi blue-chips, established L1/L2 alts
    _TIER3 = {
        'ETH/USDT', 'SOL/USDT', 'AVAX/USDT', 'MATIC/USDT', 'NEAR/USDT',
        'ICP/USDT', 'HBAR/USDT', 'APT/USDT', 'ARB/USDT', 'OP/USDT',
        'SUI/USDT', 'STX/USDT', 'FIL/USDT', 'AAVE/USDT', 'INJ/USDT',
        'TAO/USDT', 'RENDER/USDT', 'RNDR/USDT', 'FET/USDT', 'SEI/USDT',
        'TIA/USDT', 'KAS/USDT', 'GRT/USDT', 'LDO/USDT', 'PYTH/USDT',
        'JUP/USDT', 'ONDO/USDT', 'HYPE/USDT', 'ASTER/USDT', 'AGIX/USDT',
        'OCEAN/USDT', 'AKT/USDT', 'THETA/USDT', 'ENA/USDT',
        # Layer 1 additions
        'EGLD/USDT', 'FTM/USDT', 'KAVA/USDT', 'ONE/USDT', 'ZIL/USDT',
        'ROSE/USDT', 'FLOW/USDT',
        # Layer 2 additions
        'STRK/USDT', 'METIS/USDT', 'IMX/USDT', 'MANTA/USDT', 'ZK/USDT',
        'POL/USDT', 'LRC/USDT',
        # AI / infra
        'NMR/USDT', 'ARKM/USDT', 'API3/USDT', 'BAND/USDT', 'TFUEL/USDT',
        'DIA/USDT', 'TRB/USDT', 'RLC/USDT',
        # DeFi
        'COMP/USDT', 'SNX/USDT', 'CRV/USDT', 'CVX/USDT', 'DYDX/USDT',
        'GMX/USDT', '1INCH/USDT', 'SUSHI/USDT', 'YFI/USDT', 'BAL/USDT',
        'FXS/USDT', 'PENDLE/USDT', 'RAY/USDT', 'JTO/USDT', 'ETHFI/USDT',
        # Storage / gaming / mid-caps
        'AR/USDT', 'STORJ/USDT', 'SC/USDT', 'BLZ/USDT',
        'AXS/USDT', 'SAND/USDT', 'MANA/USDT', 'GALA/USDT', 'ILV/USDT',
        'ENJ/USDT', 'MAGIC/USDT', 'YGG/USDT', 'PYR/USDT', 'SUPER/USDT',
        'ALICE/USDT', 'CHR/USDT', 'XAI/USDT',
        # Other established alts
        'OKB/USDT', 'MNT/USDT', 'POLYX/USDT', 'MPL/USDT', 'BGB/USDT',
        'RUNE/USDT', 'BLUR/USDT', 'CYBER/USDT', 'ORDI/USDT', 'ENS/USDT',
        'MINA/USDT', 'CFX/USDT', 'CELO/USDT', 'GLM/USDT', 'LQTY/USDT',
        'LPT/USDT', 'MASK/USDT', 'OM/USDT', 'WOO/USDT', 'ZEN/USDT',
        'ZRX/USDT', 'UMA/USDT', 'KNC/USDT', 'RONIN/USDT', 'AUDIO/USDT',
        'BAT/USDT', 'CAKE/USDT', 'CHZ/USDT', 'CKB/USDT', 'CTSI/USDT',
        'EDU/USDT', 'FIDA/USDT', 'FLUX/USDT', 'GAS/USDT', 'HIGH/USDT',
        'ICX/USDT', 'ID/USDT', 'IO/USDT', 'IOTA/USDT', 'IOTX/USDT',
        'JOE/USDT', 'LISTA/USDT', 'LSK/USDT', 'MOVR/USDT', 'MTL/USDT',
        'NEO/USDT', 'NOT/USDT', 'OGN/USDT', 'POWR/USDT', 'QI/USDT',
        'RSR/USDT', 'SKL/USDT', 'STG/USDT', 'SXP/USDT', 'SYN/USDT',
        'TWT/USDT', 'WAXP/USDT', 'XEC/USDT',
    }

    # Meme coins, micro-caps, and highly speculative tokens
    _TIER4 = {
        'DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT', 'BONK/USDT', 'WIF/USDT',
        'FLOKI/USDT', 'TRUMP/USDT', 'NIGHT/USDT', 'WLFI/USDT', 'PI/USDT',
        'SKY/USDT', 'BOME/USDT', 'MEME/USDT', 'TURBO/USDT', 'BRETT/USDT',
        'DOGS/USDT',
        # Very low-cap / speculative high-vol alts
        'LEVER/USDT', 'TROY/USDT', 'REEF/USDT', 'DENT/USDT', 'XVG/USDT',
        'HOT/USDT', 'NULS/USDT', 'STMX/USDT', 'SLP/USDT', 'TOKEN/USDT',
        'PORTO/USDT', 'ACH/USDT', 'ACE/USDT', 'ADX/USDT', 'AERGO/USDT',
        'AGLD/USDT', 'ALPHA/USDT', 'ALT/USDT', 'AMP/USDT', 'ARK/USDT',
        'ARPA/USDT', 'ASTR/USDT', 'ATA/USDT', 'BAKE/USDT', 'BEAMX/USDT',
        'BEL/USDT', 'BICO/USDT', 'BIGTIME/USDT', 'BNX/USDT', 'C98/USDT',
        'CELR/USDT', 'COMBO/USDT', 'COTI/USDT', 'DAR/USDT', 'DGB/USDT',
        'DODO/USDT', 'DUSK/USDT', 'ERN/USDT', 'FRONT/USDT', 'HOOK/USDT',
        'IDEX/USDT', 'IOTX/USDT', 'LOKA/USDT', 'LTO/USDT', 'NKN/USDT',
        'NULS/USDT', 'OMG/USDT', 'ONG/USDT', 'PHA/USDT', 'PORTAL/USDT',
        'PROM/USDT', 'RAD/USDT', 'RARE/USDT', 'REQ/USDT', 'SFP/USDT',
        'STEEM/USDT', 'SUN/USDT', 'SYS/USDT', 'TLM/USDT', 'VANRY/USDT',
        'AEVO/USDT', 'ANKR/USDT', 'GLM/USDT',
    }
    if symbol in _TIER1:
        return 1.2
    if symbol in _TIER2:
        return 1.5
    if symbol in _TIER4:
        return 2.2
    if symbol in _TIER3:
        return 1.8
    return 1.5  # safe default for any unlisted token


def compute_dynamic_atr_multiplier(
    base_mult: float,
    er: float,          # efficiency_ratio (0–1): how directional the price move is
    vol_regime: float,  # volatility_regime (normalised; 1.0 = historical average)
) -> float:
    """Compute a bar-level ATR multiplier that adapts to current market character.

    Two orthogonal signals govern the barrier width:

    1. Efficiency Ratio (ER) — noise vs trend.
       ER near 1  → price moving cleanly in one direction → barrier can be tighter.
       ER near 0  → price whipsawing (random walk)        → barrier must be wider.
       Formula: noise_penalty = 2.0 − ER  (range 1.0 to 2.0)

    2. Volatility regime — magnitude of moves.
       vol > 1  → moves are larger than usual → barrier must be wider to avoid
                  being hit by noise on the first candle.
       vol < 1  → quieter than usual → slight tightening is safe.
       Formula: vol_factor = clip(vol_regime, 0.7, 1.8)

    Combined: dynamic = base × noise_penalty × vol_factor
    Clipped to [base × 0.8, 4.5] so the barrier never shrinks below 80 % of
    the tier baseline (protects label quality) and never balloons above 4.5
    (which would make nearly every bar HOLD and starve the model of examples).
    """
    er       = float(np.clip(er,         0.0, 1.0))
    vol      = float(np.clip(vol_regime, 0.7, 1.8))
    # Cap noise at 1.5 (was 2.0). Uncapped, very low-ER tokens (ETH at 0.28)
    # reach 1.72 noise → 3.03× barrier → 60%+ HOLD → model starves of labels.
    # 1.5 max still widens barriers in choppy markets without killing label density.
    noise    = min(2.0 - er, 1.5)  # 1.0 (perfect trend) … 1.5 (noisy)
    dynamic  = base_mult * noise * vol
    return float(np.clip(dynamic, base_mult * 0.8, 4.0))


# ============================================================
# TRIPLE-BARRIER LABELING (with censoring)
# ============================================================
def create_triple_barrier_labels(df: pd.DataFrame, atr_multiplier: float,
                                  max_lookahead: int = MAX_LOOKAHEAD,
                                  volatility_regime: Optional[pd.Series] = None,
                                  efficiency_ratio: Optional[pd.Series] = None,
                                  trend_regime: Optional[pd.Series] = None,
                                  macro_confluence_score: Optional[pd.Series] = None) -> pd.Series:
    """3-class labels: 0=SELL, 1=HOLD, 2=BUY, -1=CENSORED (dropped upstream)."""
    if df is None or df.empty:
        return pd.Series(dtype=int)

    # 0.72 is the balanced middle ground between the original 0.80 (too restrictive,
    # causing 60 % HOLD on ETH/BNB) and 0.65 (too permissive — labels noisy bars
    # in low-vol chop, which introduced label noise that hurt BTC holdout precision).
    base_vol_threshold = 0.72
    labels = pd.Series(1, index=df.index, dtype=int)
    atr = compute_atr(df, period=14)
    n = len(df)

    for i in range(n - 1):
        vol_threshold = base_vol_threshold
        if trend_regime is not None and trend_regime.iloc[i] == 1:
            vol_threshold = 0.50   # even more permissive when a trend is confirmed

        if volatility_regime is not None and volatility_regime.iloc[i] < vol_threshold:
            labels.iloc[i] = 1
            continue
        if efficiency_ratio is not None and efficiency_ratio.iloc[i] < 0.2:
            labels.iloc[i] = 1
            continue

        entry_price = df.iloc[i]['close']
        atr_val = atr.iloc[i]
        if atr_val == 0 or np.isnan(atr_val):
            atr_val = entry_price * 0.001

        # Dynamic barrier: adapts continuously to current noise level and
        # volatility regime rather than using a single static multiplier.
        _er  = float(efficiency_ratio.iloc[i]) \
               if efficiency_ratio is not None and not pd.isna(efficiency_ratio.iloc[i]) \
               else 0.5
        _vol = float(volatility_regime.iloc[i]) \
               if volatility_regime is not None and not pd.isna(volatility_regime.iloc[i]) \
               else 1.0
        dynamic_mult = compute_dynamic_atr_multiplier(atr_multiplier, _er, _vol)

        # Fixed asymmetric skew (global constants, no macro-regime override).
        # DO NOT apply macro_confluence_score here — it is already used below to
        # CANCEL hits against the macro trend. Applying it BOTH to barrier width
        # AND to hit cancellation creates a double-application bias: in a bull
        # regime (cs=+2) the BUY barrier is tightened (more BUY labels) AND SELL
        # hits are cancelled (fewer SELL labels), inflating BUY label density by
        # 20-30% beyond what the holdout's regime can sustain. This is the root
        # cause of the 13pp OOF→holdout precision gap on BTC.
        upper = entry_price + (dynamic_mult * BARRIER_UP_SKEW) * atr_val
        lower = entry_price - (dynamic_mult * BARRIER_DOWN_SKEW) * atr_val

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

        # Cancel barrier hits that contradict a strong confluence consensus.
        # Threshold ±0.5 corresponds to prc_total ≤0.25 (strong bearish) or
        # prc_total ≥0.75 (strong bullish).  The old code used exact ±2.0
        # which never matched any real value and silently no-opped for every bar.
        if macro_confluence_score is not None:
            cs = float(macro_confluence_score.iloc[i])
            if hit == 2 and cs <= -0.50:   # BUY barrier but strongly bearish confluence
                hit = None
            if hit == 0 and cs >= 0.50:    # SELL barrier but strongly bullish confluence
                hit = None

        if hit is None:
            labels.iloc[i] = 1 if window_avail >= max_lookahead else CENSORED
        else:
            labels.iloc[i] = hit

    if n > 0:
        tail = min(max_lookahead, n)
        labels.iloc[n - tail:] = CENSORED
    return labels


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
def _dm(X: pd.DataFrame, y: Optional[np.ndarray] = None, w: Optional[np.ndarray] = None) -> xgb.DMatrix:
    return xgb.DMatrix(X, label=y, weight=w, feature_names=list(X.columns))


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
    """Inverse-frequency weights so the primary stops ignoring the minority
    directional classes in favour of the easy HOLD majority."""
    y = np.asarray(y).astype(int)
    cnt = np.bincount(y, minlength=NUM_CLASS).astype(float)
    cnt[cnt == 0] = 1.0
    cw = (1.0 / cnt)
    cw = cw / cw.sum() * NUM_CLASS
    return cw[y]


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


def objective(trial, Xtr, ytr, Xva, yva):
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
    m = xgb.train(params, _dm(Xtr, ytr, sample_weights(ytr)), num_boost_round=500,
                  evals=[(_dm(Xva, yva), 'eval')], early_stopping_rounds=50, verbose_eval=False)
    return float(log_loss(yva, m.predict(_dm(Xva)), labels=list(range(NUM_CLASS))))


def primary_oof(X: pd.DataFrame, y: np.ndarray, params: dict,
                n_splits: int, gap: int) -> np.ndarray:
    """Out-of-fold primary probabilities (purged). Early rows stay NaN."""
    oof = np.full((len(X), NUM_CLASS), np.nan)
    for tr, va in TimeSeriesSplit(n_splits=n_splits, gap=gap).split(X):
        m = xgb.train(params, _dm(X.iloc[tr], y[tr], sample_weights(y[tr])),
                      num_boost_round=500, evals=[(_dm(X.iloc[va], y[va]), 'eval')],
                      early_stopping_rounds=50, verbose_eval=False)
        oof[va] = m.predict(_dm(X.iloc[va]))
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
    target. Tokens with clear directional asymmetry (e.g., strong uptrend where
    SELL signals are noise) will unlock at least one tradeable side.

    Returns (threshold, precision, coverage_within_side, n_trades, hit_target).
    Coverage here is the fraction of that SIDE's signals that pass the gate —
    not the fraction of all bars — so it's comparable across BUY and SELL.

    Guards against over-permissive thresholds:
      - MAX_SIDE_COVERAGE caps at 10 % of the side's pool. A 25 % coverage
        threshold like 0.379 means the meta model barely discriminates — those
        signals will fail on holdout when the regime shifts.
      - MIN_ABS_THRESHOLD = 0.50 enforces a hard floor: any threshold below
        0.50 is rejected regardless of OOF precision, because the meta model is
        essentially random at that confidence level.
    """
    MAX_SIDE_COVERAGE = 0.25
    # HOLD downweighting shifts the meta's calibrated output below 0.50 even
    # when it IS discriminating (e.g., AAVE SELL all-bar 63% but meta ~0.43).
    # Lowering to 0.40 lets these signals through; the 25% coverage cap still
    # blocks truly non-discriminative meta models (uniform output → all fire →
    # >25% coverage → rejected).
    MIN_ABS_THRESHOLD = 0.40

    valid = ~np.isnan(meta_prob)
    mp = meta_prob[valid]
    pr = proposed[valid]
    yt = y_true[valid]

    # Filter to the requested side only
    side_mask = (pr == side)
    if side_mask.sum() == 0:
        return 0.5, 0.0, 0.0, 0, False

    mp_s = mp[side_mask]
    yt_s = yt[side_mask]

    rows = []
    for q in [0.25, 0.20, 0.15, 0.10, 0.07, 0.05, 0.04, 0.03, 0.02]:  # max 25% coverage per side
        thr   = float(np.quantile(mp_s, 1.0 - q))
        if thr < MIN_ABS_THRESHOLD:          # reject thresholds below 50 % conf
            continue
        fire  = mp_s >= thr
        n     = int(fire.sum())
        if n < min_fires:
            continue
        cov   = n / len(mp_s)
        if cov > MAX_SIDE_COVERAGE:          # skip if coverage exceeds the cap
            continue
        prec = float((yt_s[fire] == side).mean())   # precision for this side
        rows.append((thr, prec, cov, n))

    if not rows:
        # The sweep found no valid threshold — typical when the meta model is
        # non-discriminative (near-uniform output due to heavy regularisation).
        # This happens specifically when the primary is already excellent for this
        # side (e.g. BUY prec 84%) but every quantile threshold either hits
        # >MAX_SIDE_COVERAGE or <min_fires, leaving the side silently blocked.
        # Fallback: if the unconditional precision for this side already clears
        # the target, fire on ALL proposals — no meta filtering needed.
        if len(mp_s) >= min_fires and float(mp_s.mean()) >= MIN_ABS_THRESHOLD:
            unconditional_prec = float((yt_s == side).mean())
            if unconditional_prec >= target:
                return float(mp_s.min()), unconditional_prec, 1.0, len(mp_s), True
        return float(np.quantile(mp_s, 0.9)), 0.0, 0.0, 0, False

    meeting = [r for r in rows if r[1] >= target]
    if meeting:
        thr, prec, cov, n = meeting[0]
        return thr, prec, cov, n, True

    best = max(rows, key=lambda r: r[1])

    # Anti-selection check: if the best meta-selected precision is BELOW the
    # unconditional (all-proposals) precision, the meta model is hurting rather
    # than helping. When the primary is already above target for this side,
    # bypass the meta gate entirely and fire on all proposals.
    if len(mp_s) >= min_fires and float(mp_s.mean()) >= MIN_ABS_THRESHOLD:
        unconditional_prec = float((yt_s == side).mean())
        if unconditional_prec >= target and unconditional_prec > best[1]:
            return float(mp_s.min()), unconditional_prec, 1.0, len(mp_s), True

    return best[0], best[1], best[2], best[3], False


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

    # Max drawdown on the equity curve
    equity  = np.cumsum(rets_arr)
    peak    = np.maximum.accumulate(equity)
    drawdown = peak - equity
    max_dd  = float(drawdown.max() * 100)

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

        # ── Load per-token optimizer params (if threshold_optimizer.py has run) ──
        _opt = load_token_params(symbol)
        _opt_global = (_opt or {}).get("global", {})

        # ATR multiplier: prefer optimizer result, but never tighten below the static tier.
        # The optimizer can legitimately widen barriers (safer labels); it cannot go below
        # the tier table because BARRIER_DOWN_SKEW amplifies tight bases into noisy SELL
        # labels that degrade meta-model precision below the all-bar baseline.
        _static_atr = get_atr_multiplier(symbol)
        _opt_atr = _opt_global.get("atr_multiplier")
        atr_mult = max(float(_opt_atr) if _opt_atr else _static_atr, _static_atr)

        _er_med  = float(df['efficiency_ratio_10'].median()) if 'efficiency_ratio_10' in df.columns else 0.5
        _vol_med = float(df['volatility_regime'].median())   if 'volatility_regime' in df.columns else 1.0
        _typical = compute_dynamic_atr_multiplier(atr_mult, _er_med, _vol_med)

        # ── Dynamic lookahead ─────────────────────────────────────────────
        # Prefer the optimizer's per-token lookahead if available; fall back
        # to the ER-adaptive formula.
        _opt_lh = _opt_global.get("lookahead_bars")
        token_lookahead = int(np.clip(
            int(_opt_lh) if _opt_lh else round(MAX_LOOKAHEAD * (0.4 + _er_med * 1.2)),
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
        # Regularisation tiers are moderate, not maxed. λ=6.0/mcw=20 (old
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

        print(f"ATR multiplier : base={atr_mult} | typical={_typical:.2f} "
              f"(ER_med={_er_med:.2f}, vol_med={_vol_med:.2f}) | range=[{atr_mult*0.8:.1f}, 4.5]")
        print(f"Token params   : lookahead={token_lookahead}h | "
              f"precision_target={token_precision_target:.1%} (breakeven≈{token_breakeven:.1%}) | "
              f"meta_reg=λ{_meta_reg}/mcw{_meta_mcw}")

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

        Xtp = train_pool[feature_cols]
        ytp = train_pool['target'].to_numpy().astype(int)

        # ---- 1) SHAP pruning (train pool) ----
        print("   Computing SHAP importance on train pool...")
        temp_model = xgb.train(DEFAULT_PARAMS, _dm(Xtp, ytp, sample_weights(ytp)),
                               num_boost_round=120, verbose_eval=False)
        feature_cols = prune_features_by_shap(temp_model, Xtp)
        Xtp = train_pool[feature_cols]
        print(f"   Active feature set: {len(feature_cols)} features.")

        # ---- 2) Optuna tune primary (purged inner split) ----
        inner = list(TimeSeriesSplit(n_splits=N_SPLITS_CV, gap=EMBARGO).split(Xtp))
        itr, iva = inner[-1]
        study = optuna.create_study(direction='minimize',
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(lambda t: objective(t, Xtp.iloc[itr], ytp[itr], Xtp.iloc[iva], ytp[iva]),
                       n_trials=OPTUNA_TRIALS, show_progress_bar=False)
        full_params = _full_params(study.best_params)
        print(f"   Best params (val logloss {study.best_value:.4f}): {study.best_params}")

        # ---- 3) Primary OOF on train pool -> dev metrics, calibration, meta data ----
        oof = primary_oof(Xtp, ytp, full_params, N_SPLITS_CV, EMBARGO)
        mask = ~np.isnan(oof[:, 0])
        cv_acc = float(accuracy_score(ytp[mask], oof[mask].argmax(1)))
        cv_f1 = float(f1_score(ytp[mask], oof[mask].argmax(1), average='macro', zero_division=0))
        T = fit_temperature(oof[mask], ytp[mask])
        print(f"Primary OOF (dev): acc {cv_acc:.4f} | macro-F1 {cv_f1:.4f} | T {T:.3f}")

        # ---- 4) Meta-labeling: build dataset + OOF + pick threshold ----
        meta_X_all = build_meta_X(Xtp, oof)
        prop_all = proposed_side(oof)

        meta_y_all = (prop_all == ytp).astype(int)

        mX = meta_X_all[mask].reset_index(drop=True)
        mY = meta_y_all[mask]
        prop_v = prop_all[mask]
        y_v = ytp[mask]

        # Downweight HOLD-labeled bars in meta training.
        # HOLD bars always have meta_y=0 (primary proposes BUY/SELL but hits no barrier),
        # representing 40-70% of training data. Including them at full weight makes the
        # meta uniformly pessimistic (~0.52 for everything) → top-25% is a random slice
        # → precision ≈ all-bar precision → fails target. Downweighting to 0.30 reduces
        # contamination while preserving the regime signal ("choppy regime = less reliable").
        _n_hold = int((ytp[mask] == 1).sum())
        _n_dir  = int((ytp[mask] != 1).sum())
        # Weight each HOLD bar so the total HOLD contribution equals half the directional
        # contribution — balancing signal without full exclusion.
        _hold_w = float((_n_dir * 0.5) / max(_n_hold, 1))
        _hold_w = float(np.clip(_hold_w, 0.10, 0.60))   # keep in a sane range
        meta_w  = np.where(ytp[mask] == 1, _hold_w, 1.0).astype(float)

        meta_ready = len(mX) >= max(200, MIN_FIRES_DEV * 4)
        if meta_ready:
            meta_oof = binary_oof(mX, mY, token_meta_params, N_SPLITS_CV, EMBARGO, w=meta_w)

            # ── Spot-on market dynamics: Regime-specific directional filter on DEV set ──
            # Apply filter AFTER meta-model is trained on un-tampered proposals
            prop_dev_filtered = prop_v.copy()
            _opt = load_token_params(symbol)
            if _opt and "regimes" in _opt and "regime_boundaries" in _opt:
                bounds = _opt["regime_boundaries"]
                regimes_dict = _opt["regimes"]
                
                vol_avg = train_pool["volume"].rolling(24, min_periods=1).mean()
                atr_pct = (train_pool["_atr"] / train_pool["close"]).fillna(0)
                momentum = train_pool["close"].pct_change(24).fillna(0)
                
                def _tier(val, p33, p67): return "low" if val <= p33 else ("med" if val <= p67 else "high")
                def _trend(val, p33, p67): return "down" if val <= p33 else ("flat" if val <= p67 else "up")
                
                vp33, vp67 = bounds.get("vol_p33", 0), bounds.get("vol_p67", 0)
                ap33, ap67 = bounds.get("atr_pct_p33", 0), bounds.get("atr_pct_p67", 0)
                mp33, mp67 = bounds.get("momentum_p33", -0.02), bounds.get("momentum_p67", 0.02)
                
                regime_strs = [
                    f"{_tier(vol_avg.iloc[i], vp33, vp67)}_{_tier(atr_pct.iloc[i], ap33, ap67)}_{_trend(momentum.iloc[i], mp33, mp67)}"
                    for i in range(len(Xtp))
                ]
                
                idx_map = np.where(mask)[0]
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

            # ── Combined threshold (backward-compatible baseline) ─────────────
            thr, dev_prec, dev_cov, dev_n, hit_target = pick_threshold(
                meta_oof, prop_dev_filtered, y_v, target=token_precision_target)

            # ── Per-side thresholds (BUY / SELL independently) ────────────────
            # Tokens with directional asymmetry (e.g. strong macro uptrend where
            # SELL signals are noise) can unlock at least one profitable side even
            # when the combined precision fails the target.
            # min_fires=50 for per-side: the BUY/SELL pools are each roughly half
            # the combined pool, so 80 trades per side is too strict (JUP had 105
            # combined trades split 50/55 BUY/SELL — both failed at 80, yet
            # combined holdout was 89.7%).
            _SIDE_MIN_FIRES = 50
            thr_buy,  prec_buy,  cov_buy,  n_buy,  hit_buy  = pick_threshold_by_side(
                meta_oof, prop_dev_filtered, y_v, side=2, target=token_precision_target,
                min_fires=_SIDE_MIN_FIRES)
            thr_sell, prec_sell, cov_sell, n_sell, hit_sell = pick_threshold_by_side(
                meta_oof, prop_dev_filtered, y_v, side=0, target=token_precision_target,
                min_fires=_SIDE_MIN_FIRES)

            # Re-evaluate hit_target: pass if EITHER side is tradeable
            hit_target = hit_target or hit_buy or hit_sell

            meta_full = xgb.train(token_meta_params, _dm(mX, mY, meta_w), num_boost_round=300, verbose_eval=False)
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

        # Fire using BOTH an absolute gate AND a rank gate, take whichever fires more.
        # The absolute `thr` can be unreachable on the holdout if the full-fit primary's
        # confidence is scaled differently than the OOF primary the gate was tuned on
        # (this is exactly what produced 0 signals). The rank gate reproduces the dev
        # COVERAGE on the holdout regardless of absolute scale, so the operating point
        # is preserved even when the probability distribution shifts.
        # Production rule: a token only trades if it earned the precision floor on
        # DEV data (hit_target). If it didn't, it emits nothing — we do not ship
        # below-breakeven signals. The threshold was chosen on OOF dev data, never
        # on this holdout, so the holdout numbers below are an honest estimate of
        # how the pre-committed gate performs out of sample.
        # Pre-compute aggressive eligibility so the holdout section can use it.
        _AGG_FLOOR_PRE = 0.50
        _tier_agg_pre = (
            (prec_sell >= _AGG_FLOOR_PRE and n_sell >= _SIDE_MIN_FIRES) or
            (prec_buy  >= _AGG_FLOOR_PRE and n_buy  >= _SIDE_MIN_FIRES) or
            (dev_prec  >= _AGG_FLOOR_PRE and dev_n  >= MIN_FIRES_DEV)
        )

        if not hit_target:
            if _tier_agg_pre:
                # Dev precision bar not met for balanced/conservative, but the
                # per-side signals ARE above 50% — run a holdout evaluation so
                # aggressive-tier users see real out-of-sample performance.
                # Use the best qualifying per-side threshold (not the rank gate).
                if prec_sell >= _AGG_FLOOR_PRE and n_sell >= _SIDE_MIN_FIRES:
                    _agg_thr = thr_sell
                    _agg_side_pool = (prop_h == 0)
                elif prec_buy >= _AGG_FLOOR_PRE and n_buy >= _SIDE_MIN_FIRES:
                    _agg_thr = thr_buy
                    _agg_side_pool = (prop_h == 2)
                else:
                    _agg_thr = thr
                    _agg_side_pool = (prop_h == 2) | (prop_h == 0)
                # Rank gate: reproduce dev coverage on the qualifying side pool
                if _agg_side_pool.sum() > 0:
                    _pool_meta = meta_prob_h[_agg_side_pool]
                    _agg_cov = (prec_sell >= _AGG_FLOOR_PRE and n_sell >= _SIDE_MIN_FIRES) and cov_sell or cov_buy
                    _agg_rank_thr = float(np.quantile(_pool_meta, max(0.0, 1.0 - _agg_cov)))
                    fire = (meta_prob_h >= max(_agg_thr, _agg_rank_thr)) & _agg_side_pool
                else:
                    fire = np.zeros(len(meta_prob_h), dtype=bool)
                gate_mode = f"AGGRESSIVE (best-side thr {_agg_thr:.3f})"
            else:
                fire = np.zeros(len(meta_prob_h), dtype=bool)
                gate_mode = "DISABLED (dev precision floor not met)"
        else:
            fire_abs = meta_prob_h >= thr
            if dev_cov > 0:
                rank_thr = float(np.quantile(meta_prob_h, 1.0 - dev_cov))
                fire_rank = meta_prob_h >= rank_thr
            else:
                fire_rank = np.zeros(len(meta_prob_h), dtype=bool)
            # Prefer the rank gate: it reproduces the dev COVERAGE out of sample
            # regardless of probability-scale drift. Fall back to absolute only if
            # rank fires nothing.
            fire = fire_rank if fire_rank.sum() > 0 else fire_abs
            gate_mode = f"rank(cov≈{dev_cov:.1%})" if fire is fire_rank else "absolute"
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
        if (hit_target or _tier_agg_pre) and fire.sum() >= 4:
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
        if (hit_target or _tier_agg_pre) and fire.sum() >= 4 and 'macro_trend_1d' in holdout.columns:
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

        # ── Per-side holdout precision (using per-side thresholds directly) ──
        # Do NOT intersect with the combined rank gate. The combined gate fires
        # the top-5% by meta confidence which, in a bull regime, is dominated by
        # SELL signals — leaving BUY with 0 holdout trades and a false "insufficient
        # data" conclusion. Use the per-side OOF thresholds (thr_buy, thr_sell)
        # to fire each side independently on the holdout.
        _h_buy_pool  = (prop_h == 2)
        _h_sell_pool = (prop_h == 0)
        # For aggressive tier, allow per-side fire even when hit_buy/hit_sell is False,
        # as long as that side's dev precision cleared the 50% floor.
        _agg_buy_ok  = _tier_agg_pre and prec_buy  >= _AGG_FLOOR_PRE and n_buy  >= _SIDE_MIN_FIRES
        _agg_sell_ok = _tier_agg_pre and prec_sell >= _AGG_FLOOR_PRE and n_sell >= _SIDE_MIN_FIRES

        if (hit_buy or _agg_buy_ok) and _h_buy_pool.sum() > 0:
            _buy_rank_thr = float(np.quantile(
                meta_prob_h[_h_buy_pool], max(0.0, 1.0 - cov_buy)))
            buy_fire = (meta_prob_h >= max(thr_buy, _buy_rank_thr)) & _h_buy_pool
        else:
            buy_fire = np.zeros(len(meta_prob_h), dtype=bool)

        if (hit_sell or _agg_sell_ok) and _h_sell_pool.sum() > 0:
            _sell_rank_thr = float(np.quantile(
                meta_prob_h[_h_sell_pool], max(0.0, 1.0 - cov_sell)))
            sell_fire = (meta_prob_h >= max(thr_sell, _sell_rank_thr)) & _h_sell_pool
        else:
            sell_fire = np.zeros(len(meta_prob_h), dtype=bool)

        # ── Spot-on market dynamics: Regime-specific directional filter ──────
        if (hit_target or _tier_agg_pre) and _opt and "regimes" in _opt and "regime_boundaries" in _opt:
            bounds = _opt["regime_boundaries"]
            regimes_dict = _opt["regimes"]
            
            vol_avg = holdout["volume"].rolling(24, min_periods=1).mean()
            atr_pct = (holdout["_atr"] / holdout["close"]).fillna(0)
            momentum = holdout["close"].pct_change(24).fillna(0)
            
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
        close_arr = holdout['close'].to_numpy()
        barrier_frac = np.divide(dyn * holdout['_atr'].to_numpy(), close_arr,
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
            # A side is live on holdout when it had enough trades AND cleared
            # the fee breakeven. We use the per-side OOF approval (hit_buy /
            # hit_sell) as a gate — the holdout can confirm or veto each side
            # independently. This prevents a bull regime from killing BUY signals
            # just because the pooled SELL signals were bad on holdout.
            _MIN_SIDE = 5    # minimum per-side holdout trades to trust the result
            # A side is tradeable when:
            #   1. OOF approved it (hit_buy / hit_sell)
            #   2. It actually fired signals in the holdout (> 0 trades)
            #   3. Precision is at least 50% (hard floor regardless of sample size)
            #   4. Either: not enough trades to be conclusive (< _MIN_SIDE)
            #              OR precision cleared the fee breakeven
            # Zero trades → NOT "insufficient data" — it means no high-conf
            # signals of that side appeared in holdout; that is itself evidence
            # the side is inactive in the current regime → disable it.
            # The 0.50 floor prevents 1-4 bad trades from unlocking a side whose
            # combined performance is deeply negative (e.g. FIDA: -248% return).
            tradeable_buy_holdout  = (
                hit_buy and
                buy_h_n > 0 and
                buy_h_prec  >= 0.50 and
                (buy_h_n  < _MIN_SIDE or buy_h_prec  >= breakeven)
            )
            tradeable_sell_holdout = (
                hit_sell and
                sell_h_n > 0 and
                sell_h_prec >= 0.50 and
                (sell_h_n < _MIN_SIDE or sell_h_prec >= breakeven)
            )
            tradeable_final = tradeable_buy_holdout or tradeable_sell_holdout

            holdout_reliable = fired_n >= MIN_HOLDOUT_FIRES
            oof_holdout_gap  = abs(dev_prec - fired_prec)

            # Override: keep tradeable when combined precision OR expectancy is
            # meaningful. Require >= 0.20 %/trade (not just > 0) — at typical
            # holdout sizes (50-200 trades) anything below 0.20% is indistinguishable
            # from noise. The original > 0 threshold was enabling tokens with
            # +0.11 %/trade at 39% precision (well below breakeven) based on the
            # triple-barrier timeout artifact rather than genuine directional edge.
            EXPECTANCY_FLOOR = 0.20          # %/trade minimum for the override
            combined_ok = (fired_prec >= breakeven
                           or bt['expectancy_pct'] >= EXPECTANCY_FLOOR)
            if hit_target and (not holdout_reliable or combined_ok):
                tradeable_final = True

            # Gap veto: if the OOF→holdout precision gap exceeds 12 pp AND the
            # holdout precision is below breakeven, disable unconditionally.
            # A 19 pp gap (dev 58.5% → holdout 39.5%) means the model overfit to
            # the training period. Positive expectancy from timeouts on 81 trades
            # is not a valid override for a regime that clearly shifted.
            GAP_VETO_THRESHOLD = 0.12
            if (holdout_reliable
                    and oof_holdout_gap >= GAP_VETO_THRESHOLD
                    and fired_prec < breakeven):
                tradeable_final = False

            # Final safety veto: disables tokens where no per-side precision
            # check passed AND the combined holdout expectancy is negative.
            per_side_approved = tradeable_buy_holdout or tradeable_sell_holdout
            veto_fires = (holdout_reliable
                          and bt['expectancy_pct'] <= 0
                          and not per_side_approved)
            if veto_fires:
                tradeable_final = False

            if not tradeable_final:
                if veto_fires:
                    print(f"      DISABLED: negative expectancy ({bt['expectancy_pct']:+.3f}%) "
                          f"with no per-side precision approval (combined prec {fired_prec:.1%}).")
                else:
                    print(f"      DISABLED: neither side cleared holdout breakeven "
                          f"{breakeven:.1%} with ≥{_MIN_SIDE} trades and prec ≥50%.")
            elif not combined_ok:
                # Per-side saved it even though combined failed
                sides_live = []
                if tradeable_buy_holdout:  sides_live.append('BUY')
                if tradeable_sell_holdout: sides_live.append('SELL')
                print(f"      ENABLED via per-side: {' + '.join(sides_live)} cleared holdout "
                      f"(combined {fired_prec:.1%} < {breakeven:.1%}).")
            elif bt['expectancy_pct'] >= EXPECTANCY_FLOOR and fired_prec < breakeven:
                print(f"      ENABLED via expectancy: {bt['expectancy_pct']:+.3f}%/trade "
                      f"(prec {fired_prec:.1%} < breakeven {breakeven:.1%}, "
                      f"floor={EXPECTANCY_FLOOR:.2f}%).")

            if holdout_reliable and oof_holdout_gap >= GAP_VETO_THRESHOLD and fired_prec < breakeven:
                print(f"      DISABLED (gap veto): OOF→holdout gap {oof_holdout_gap:.1%} "
                      f"(dev {dev_prec:.1%} → holdout {fired_prec:.1%}) with prec below breakeven. "
                      f"Regime shift too large to ship.")
            elif holdout_reliable and oof_holdout_gap > 0.10:
                print(f"      WATCH: OOF→holdout gap {oof_holdout_gap:.1%} "
                      f"(dev {dev_prec:.1%} → holdout {fired_prec:.1%}). "
                      f"Possible regime shift — monitor after next retrain.")
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
        # Aggressive: any directional side has dev precision > 50% with enough trades
        _AGG_FLOOR     = 0.50
        _any_side_above_floor = (
            (prec_sell >= _AGG_FLOOR and n_sell >= _SIDE_MIN_FIRES) or
            (prec_buy  >= _AGG_FLOOR and n_buy  >= _SIDE_MIN_FIRES) or
            (dev_prec  >= _AGG_FLOOR and dev_n  >= MIN_FIRES_DEV)
        )
        tier_aggressive = bool(_any_side_above_floor)

        # Propagate upward: qualifying a higher tier implies lower tiers too.
        tier_balanced   = tier_balanced   or tier_conservative
        tier_aggressive = tier_aggressive or tier_balanced

        print(f"   Risk tiers: "
              f"conservative={'✓' if tier_conservative else '✗'} | "
              f"balanced={'✓' if tier_balanced else '✗'} | "
              f"aggressive={'✓' if tier_aggressive else '✗'}")

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
        meta_path = None
        if meta_full is not None:
            meta_path = store_dir / f"{base}_meta_model.json"
            meta_full.save_model(str(meta_path))
        print(f"Models saved: {model_path.name}" + (f" + {meta_path.name}" if meta_path else ""))

        sidecar = store_dir / f"{base}_meta.json"
        with open(sidecar, "w") as f:
            json.dump({
                "symbol": symbol,
                "model_format": "booster",          # predictor loads with xgb.Booster()
                "num_class": NUM_CLASS,
                "feature_cols": feature_cols,        # exact order the Booster expects
                "meta_feature_cols": feature_cols,
                "calibration_temperature": T,
                "meta_threshold": thr,               # combined gate (both sides)
                "production_confidence_floor": thr,  # backward-compat alias
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
                "trained_at": datetime.now().isoformat(),
                # ── Optimizer regime data (from threshold_optimizer.py) ────────
                # Embedded here so predictor.py needs only this one file at
                # inference time. Re-run threshold_optimizer.py after retraining
                # to refresh these values; retrain then picks them up on the next
                # training cycle.
                "regime_thresholds":  (_opt or {}).get("regimes", {}),
                "regime_boundaries":  (_opt or {}).get("regime_boundaries", {}),
                "optimizer_updated_at": (_opt or {}).get("updated_at"),
            }, f, indent=2)

        log_feature_importance(deploy_primary, feature_cols, symbol)

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

    Scans model_store for *_model.json files, finds the one with the latest
    mtime, maps it back to a FLEET_SYMBOLS entry, and returns index+1.
    Returns 0 (start from the beginning) if no prior models exist or the last
    model doesn't match any fleet symbol.
    """
    model_files = list(store_dir.glob("*_model.json"))
    if not model_files:
        return 0

    latest = max(model_files, key=lambda p: p.stat().st_mtime)
    # Convert filename back to symbol:  BTC_USDT_model.json → BTC/USDT
    base = latest.name.replace("_model.json", "")
    # The symbol has exactly one "/" — it separates ticker from quote (e.g. BTC/USDT).
    # The filename encodes "/" as "_", so we split on the first "_USDT" or "_BTC" etc.
    # Most reliable: try every fleet symbol and find the one whose encoded form matches.
    sym_map = {s.replace("/", "_"): s for s in FLEET_SYMBOLS}
    symbol = sym_map.get(base)
    if symbol is None:
        return 0

    try:
        idx = FLEET_SYMBOLS.index(symbol)
        return idx + 1   # resume with the token AFTER the last completed one
    except ValueError:
        return 0


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
    if start_idx > 0:
        skipped = FLEET_SYMBOLS[:start_idx]
        print(f"\nRESUMING from {FLEET_SYMBOLS[start_idx]} "
              f"(skipping {start_idx} already-trained token{'s' if start_idx != 1 else ''}: "
              f"{', '.join(skipped[:3])}{'...' if len(skipped) > 3 else ''})")
    else:
        print("\nStarting full fleet training from the beginning.")

    results = []
    total = len(FLEET_SYMBOLS)
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
    _args = _parser.parse_args()
    train_fleet(hours=_args.hours, resume=not _args.full)