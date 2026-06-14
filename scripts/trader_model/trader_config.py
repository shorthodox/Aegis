"""
trader_config.py — Universal Trader Model Configuration

Train on 10 diverse tokens; deploy on 60.
3 mindsets: scalping / intraday / swing
3 risk profiles: conservative / balanced / aggressive
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT               = Path(__file__).resolve().parent.parent.parent
TRADER_MODEL_STORE = ROOT / 'src' / 'ml' / 'trader_model_store'
TRADER_STATE_PATH  = ROOT / 'data' / 'trader_signals_state.json'
TRADER_RECORD_PATH = ROOT / 'data' / 'trader_track_record.json'
TRAINING_SUMMARY   = ROOT / 'logs' / 'training_summary.json'

# ── 10 Training Tokens ─────────────────────────────────────────────────────────
# Chosen for diversity across: volatility, momentum regime, sector, correlation
TRAINING_TOKENS = [
    'BTC/USDT',   # large-cap, low volatility baseline
    'BNB/USDT',   # exchange token, buy-burn dynamics
    'SOL/USDT',   # fast L1, high momentum + retail
    'DOGE/USDT',  # meme/retail, high intraday vol
    'LINK/USDT',  # oracle sector, trend-following
    'AVAX/USDT',  # L1 competitor, medium vol
    'XRP/USDT',   # payment rail, news-sensitive
    'DOT/USDT',   # cross-chain hub, ranging tendency
    'ATOM/USDT',  # DeFi/IBC hub, medium vol
    'ADA/USDT',   # large-cap, accumulation/distribution patterns
]

# ── 60 Deployment Tokens ──────────────────────────────────────────────────────
DEPLOYMENT_TOKENS = [
    'BTC/USDT',  'ETH/USDT',  'XRP/USDT',  'BNB/USDT',  'SOL/USDT',
    'DOGE/USDT', 'ADA/USDT',  'BCH/USDT',  'LINK/USDT', 'LTC/USDT',
    'AVAX/USDT', 'HBAR/USDT', 'SUI/USDT',  'SHIB/USDT', 'TON/USDT',
    'DOT/USDT',  'UNI/USDT',  'NEAR/USDT', 'PEPE/USDT', 'AAVE/USDT',
    'ICP/USDT',  'ETC/USDT',  'ONDO/USDT', 'ENA/USDT',  'POL/USDT',
    'ALGO/USDT', 'RENDER/USDT','ATOM/USDT','QNT/USDT',  'ARB/USDT',
    'APT/USDT',  'FIL/USDT',  'JUP/USDT',  'WLD/USDT',  'TRX/USDT',
    'XLM/USDT',  'INJ/USDT',  'OP/USDT',   'STX/USDT',  'FTM/USDT',
    'RUNE/USDT', 'SEI/USDT',  'KAVA/USDT', 'SNX/USDT',  'SAND/USDT',
    'MANA/USDT', 'GRT/USDT',  'CHZ/USDT',  'BAT/USDT',  'ENJ/USDT',
    'ZEC/USDT',  'VET/USDT',  'CRV/USDT',  'IMX/USDT',  'THETA/USDT',
    'EGLD/USDT', 'ONE/USDT',  'ZIL/USDT',  'DASH/USDT', 'FLOW/USDT',
]

# ── Trading Modes ──────────────────────────────────────────────────────────────
MODES = {
    'scalping': {
        'timeframe':        '5m',
        'label_bars':       4,        # 20-min forward return window
        'label_threshold':  0.003,    # 0.3% → BUY/SELL, else HOLD
        'cooldown_minutes': 30,
        'description':      'Fast 5-minute entries, quick exits (20-60 min hold)',
        'max_hold_bars':    12,
        'candles_fetch':    500,
        'min_confidence_override': 0.55,
        'model_key':        'scalping',  # model file prefix in trader_model_store
    },
    'scalping_15m': {
        'timeframe':        '15m',
        'label_bars':       4,        # 1h forward return window
        'label_threshold':  0.004,    # 0.4% → BUY/SELL, else HOLD
        'cooldown_minutes': 60,
        'description':      '15-minute scalp entries, 1-2h hold',
        'max_hold_bars':    8,
        'candles_fetch':    500,
        'min_confidence_override': 0.52,
        'model_key':        'scalping',  # reuse 5m scalping model weights
    },
    'intraday': {
        'timeframe':        '1h',
        'label_bars':       8,        # 8h forward return window
        'label_threshold':  0.008,    # 0.8%
        'cooldown_minutes': 240,
        'description':      'Intraday 1h setups, same-session exits (4-12h hold)',
        'max_hold_bars':    24,
        'candles_fetch':    500,
    },
    'swing': {
        'timeframe':        '4h',
        'label_bars':       6,        # 24h forward return window
        'label_threshold':  0.015,    # 1.5%
        'cooldown_minutes': 1440,
        'description':      'Multi-day swing setups, 2-5 day holds',
        'max_hold_bars':    30,
        'candles_fetch':    400,
    },
}

# ── Risk Profiles ──────────────────────────────────────────────────────────────
RISK_PROFILES = {
    'conservative': {
        'min_confidence': 0.75,
        'max_signals':    8,
        'position_pct':   1.0,    # 1% of portfolio per trade
        'description':    'High precision, fewer signals, tight stop loss',
        'sl_atr_mult':    1.0,
        'tp1_atr_mult':   1.5,
        'tp2_atr_mult':   2.5,
        'tp3_atr_mult':   4.0,
    },
    'balanced': {
        'min_confidence': 0.70,
        'max_signals':    15,
        'position_pct':   2.0,
        'description':    'Balanced precision vs signal coverage',
        'sl_atr_mult':    1.5,
        'tp1_atr_mult':   2.0,
        'tp2_atr_mult':   3.0,
        'tp3_atr_mult':   5.0,
    },
    'aggressive': {
        'min_confidence': 0.55,
        'max_signals':    25,
        'position_pct':   3.0,
        'description':    'More signals, higher potential returns, higher drawdown risk',
        'sl_atr_mult':    2.0,
        'tp1_atr_mult':   2.5,
        'tp2_atr_mult':   4.0,
        'tp3_atr_mult':   6.0,
    },
}

# ── 25 Strategy Names ──────────────────────────────────────────────────────────
# Scalping (5m): 10 strategies
# Intraday (1h): 10 strategies
# Swing (4h):    5 strategies
STRATEGY_NAMES = [
    # ─ Scalping ────────────────────────────────────────────────────────────────
    'vwap_bounce',
    'bb_squeeze_breakout',
    'ema_cross_fast',
    'macd_zero_cross',
    'volume_spike_entry',
    'rsi_momentum_cross',
    'order_flow_imbalance',
    'support_resistance_retest',
    'opening_range_breakout',
    'momentum_burst_candles',
    # ─ Intraday ────────────────────────────────────────────────────────────────
    'trend_continuation_pullback',
    'ema21_bounce',
    'vwap_deviation_reversion',
    'rsi_divergence',
    'macd_histogram_reversal',
    'bb_squeeze_expansion',
    'oi_price_divergence',
    'news_momentum_catalyst',
    'session_high_low_break',
    'ema_golden_death_cross',
    # ─ Swing ───────────────────────────────────────────────────────────────────
    'supply_demand_zone_retest',
    'fibonacci_retracement_entry',
    'higher_timeframe_alignment',
    'market_structure_break',
    'liquidation_cascade_fade',
]

# ── Feature engine indicator names (from src/ml/feature_engine.py) ───────────
# These 20 normalized indicators are appended to the 25 strategy scores,
# giving the model 45 total features.
FEATURE_ENGINE_NAMES = [
    'fe_rsi_14',
    'fe_rsi_7',
    'fe_macd_line',
    'fe_macd_hist',
    'fe_adx_14',
    'fe_atr_pct',
    'fe_bb_width',
    'fe_bb_pct_b',
    'fe_cmf_20',
    'fe_mfi_14',
    'fe_stoch_rsi_k',
    'fe_stoch_rsi_d',
    'fe_vwap_dev',
    'fe_ema20_slope',
    'fe_ema50_slope',
    'fe_vol_ratio',
    'fe_price_zscore',
    'fe_vol_regime',
    'fe_efficiency_ratio',
    'fe_williams_r',
]

ALL_FEATURE_NAMES = STRATEGY_NAMES + FEATURE_ENGINE_NAMES

# ── Training data fetch sizes ──────────────────────────────────────────────────
TRAINING_BARS = {
    '5m':  2000,
    '1h':  1500,
    '4h':  750,
}

# Minimum confidence for any signal (overridden by risk profile)
ABSOLUTE_MIN_CONFIDENCE = 0.50
