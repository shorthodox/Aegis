"""
strategy_features.py — 25 Scalping/Intraday/Swing Strategy Feature Calculators

Each strategy returns a float in [-1.0, 1.0]:
  +1.0 = very strong BUY signal
   0.0 = neutral
  -1.0 = very strong SELL signal

All functions are pure (no side-effects) and operate on a OHLCV DataFrame.
Features are later percentile-ranked within a rolling window before model input,
making them token-agnostic (works on any token not seen during training).
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from typing import Dict, Optional

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.ml.feature_engine import (
    compute_rsi, compute_macd, compute_vwap, compute_atr,
    compute_adx, compute_bollinger_width, compute_cmf,
    compute_mfi, compute_stoch_rsi, compute_efficiency_ratio,
    compute_price_zscore, compute_volatility_regime, compute_williams_r,
    compute_bollinger_pct_b,
)


def _clamp(x: float) -> float:
    return float(np.clip(x, -1.0, 1.0))


def _percentile_rank(series: pd.Series, lookback: int = 100) -> pd.Series:
    """Rolling percentile rank (0-1) — normalizes a feature within its own history."""
    return series.rolling(lookback, min_periods=20).apply(
        lambda x: (x[:-1] <= x[-1]).sum() / max(len(x) - 1, 1), raw=True
    ).fillna(0.5)


# ═══════════════════════════════════════════════════════════════════════════════
#  SCALPING STRATEGIES (5m timeframe)
# ═══════════════════════════════════════════════════════════════════════════════

def strat_vwap_bounce(df: pd.DataFrame) -> pd.Series:
    """Price deviation from VWAP + volume confirmation.  Buy below, sell above."""
    vwap = compute_vwap(df)
    close = df['close']
    vol   = df['volume']
    vol_ma = vol.rolling(20).mean()
    dev = (close - vwap) / (vwap + 1e-9)          # negative = price below VWAP
    vol_surge = (vol / (vol_ma + 1e-9)).clip(0, 3) / 3.0  # 0-1 normalized
    # Bounce signal: price below VWAP + high volume → BUY; above + high vol → SELL
    score = -dev * 5.0 * vol_surge                  # invert so below VWAP = positive
    return score.apply(_clamp).rename('vwap_bounce')


def strat_bb_squeeze_breakout(df: pd.DataFrame) -> pd.Series:
    """Bollinger Band squeeze (low width) followed by directional breakout."""
    close  = df['close']
    mid    = close.rolling(20).mean()
    std    = close.rolling(20).std()
    upper  = mid + 2 * std
    lower  = mid - 2 * std
    width  = (upper - lower) / (mid + 1e-9)
    width_pct = _percentile_rank(width, 100)        # low percentile = squeeze
    in_squeeze = (width_pct < 0.25).astype(float)
    direction = np.sign(close - mid)                 # +1 above mid, -1 below
    score = in_squeeze * direction * (1 - width_pct) * 4
    return score.apply(_clamp).rename('bb_squeeze_breakout')


def strat_ema_cross_fast(df: pd.DataFrame) -> pd.Series:
    """3/8 EMA crossover with momentum confirmation."""
    close  = df['close']
    ema3   = close.ewm(span=3,  adjust=False).mean()
    ema8   = close.ewm(span=8,  adjust=False).mean()
    ema21  = close.ewm(span=21, adjust=False).mean()
    cross  = ema3 - ema8
    prev   = cross.shift(1)
    # Cross just happened + price above longer EMA
    trend_filter = np.sign(close - ema21)
    cross_signal = np.sign(cross) * (np.sign(cross) != np.sign(prev)).astype(float)
    momentum = (cross / (close.abs() + 1e-9)) * 1000
    score = cross_signal * trend_filter * momentum.abs().clip(0, 1) * 2
    return score.apply(_clamp).rename('ema_cross_fast')


def strat_macd_zero_cross(df: pd.DataFrame) -> pd.Series:
    """MACD line crosses zero from below (buy) or above (sell)."""
    _, _, hist = compute_macd(df['close'])
    macd_line, _, _ = compute_macd(df['close'])
    cross_up   = ((macd_line > 0) & (macd_line.shift(1) <= 0)).astype(float)
    cross_down = ((macd_line < 0) & (macd_line.shift(1) >= 0)).astype(float)
    hist_slope = (hist - hist.shift(1)).apply(np.sign)
    score = (cross_up - cross_down) + hist_slope * 0.3
    return score.apply(_clamp).rename('macd_zero_cross')


def strat_volume_spike_entry(df: pd.DataFrame) -> pd.Series:
    """Volume > 2× 20-bar average with directional candle body."""
    vol    = df['volume']
    vol_ma = vol.rolling(20).mean()
    spike  = (vol / (vol_ma + 1e-9)).clip(0, 5)
    body   = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-9)
    score  = body * (spike / 5.0) * 2
    return score.apply(_clamp).rename('volume_spike_entry')


def strat_rsi_momentum_cross(df: pd.DataFrame) -> pd.Series:
    """RSI crossing 50 from below (buy) or above (sell) with momentum."""
    rsi    = compute_rsi(df['close'], 14)
    cross_up   = ((rsi > 50) & (rsi.shift(1) <= 50)).astype(float)
    cross_down = ((rsi < 50) & (rsi.shift(1) >= 50)).astype(float)
    # Oversold bounce or overbought rejection
    oversold  = ((rsi < 35) & (rsi > rsi.shift(1))).astype(float) * 0.5
    overbought = ((rsi > 65) & (rsi < rsi.shift(1))).astype(float) * 0.5
    score = (cross_up - cross_down) + oversold - overbought
    return score.apply(_clamp).rename('rsi_momentum_cross')


def strat_order_flow_imbalance(df: pd.DataFrame) -> pd.Series:
    """Buy/sell volume imbalance inferred from candle structure."""
    high, low, close, open_ = df['high'], df['low'], df['close'], df['open']
    total_range = (high - low + 1e-9)
    buy_vol_proxy  = (close - low)  / total_range  # fraction of range that's "green"
    sell_vol_proxy = (high - close) / total_range
    imbalance = (buy_vol_proxy - sell_vol_proxy).rolling(3).mean()  # smooth 3 bars
    score = (imbalance - 0.5) * 4   # center and amplify
    return score.apply(_clamp).rename('order_flow_imbalance')


def strat_support_resistance_retest(df: pd.DataFrame) -> pd.Series:
    """Price testing rolling 20-bar S/R with reversal candle signal."""
    close = df['close']
    high  = df['high']
    low   = df['low']
    roll_high = high.rolling(20).max().shift(1)
    roll_low  = low.rolling(20).min().shift(1)
    dist_high = (roll_high - close) / (close + 1e-9)
    dist_low  = (close - roll_low)  / (close + 1e-9)
    # Near resistance with bearish body → sell; near support with bullish body → buy
    body = (close - df['open']) / (high - low + 1e-9)
    near_resistance = (dist_high < 0.005).astype(float)
    near_support    = (dist_low  < 0.005).astype(float)
    score = near_support * body * 4 - near_resistance * body * 4
    return score.apply(_clamp).rename('support_resistance_retest')


def strat_opening_range_breakout(df: pd.DataFrame) -> pd.Series:
    """Break of the most recent 3-bar opening range (ORB)."""
    high  = df['high']
    low   = df['low']
    close = df['close']
    orb_high = high.rolling(3).max().shift(1)
    orb_low  = low.rolling(3).min().shift(1)
    orb_mid  = (orb_high + orb_low) / 2
    vol    = df['volume']
    vol_ma = vol.rolling(20).mean()
    vol_factor = (vol / (vol_ma + 1e-9)).clip(0.5, 3) / 3
    break_up   = ((close > orb_high) & (close.shift(1) <= orb_high.shift(1))).astype(float)
    break_down = ((close < orb_low)  & (close.shift(1) >= orb_low.shift(1))).astype(float)
    score = (break_up - break_down) * vol_factor * 2
    return score.apply(_clamp).rename('opening_range_breakout')


def strat_momentum_burst_candles(df: pd.DataFrame) -> pd.Series:
    """3+ consecutive bullish/bearish candles with increasing volume."""
    close = df['close']
    vol   = df['volume']
    diff = close.diff()
    direction = diff.gt(0).astype(float) - diff.lt(0).astype(float)  # +1 / 0 / -1
    streak = direction.rolling(3).sum() / 3.0
    vol_increasing = ((vol > vol.shift(1)) & (vol.shift(1) > vol.shift(2))).astype(float)
    score = streak * vol_increasing * 2
    return score.apply(_clamp).rename('momentum_burst_candles')


# ═══════════════════════════════════════════════════════════════════════════════
#  INTRADAY STRATEGIES (1h timeframe)
# ═══════════════════════════════════════════════════════════════════════════════

def strat_trend_continuation_pullback(df: pd.DataFrame) -> pd.Series:
    """ADX > 25 with price pulling back to EMA21 in trend direction."""
    close  = df['close']
    ema21  = close.ewm(span=21, adjust=False).mean()
    adx    = compute_adx(df, 14)
    trend  = np.sign(close - ema21)
    pullback = (trend > 0) & (close < ema21 * 1.002)  # near or below EMA in uptrend
    pullback |= (trend < 0) & (close > ema21 * 0.998)
    strong_trend = (adx > 25).astype(float)
    dist = (close - ema21) / (ema21 + 1e-9)
    score = -dist * 10 * strong_trend  # price below EMA21 in uptrend → buy
    return score.apply(_clamp).rename('trend_continuation_pullback')


def strat_ema21_bounce(df: pd.DataFrame) -> pd.Series:
    """Price bouncing off EMA21 with volume confirmation."""
    close  = df['close']
    ema21  = close.ewm(span=21, adjust=False).mean()
    ema50  = close.ewm(span=50, adjust=False).mean()
    trend  = np.sign(ema21 - ema50)  # bullish if ema21 > ema50
    dist   = (close - ema21) / (close + 1e-9)
    body   = (close - df['open']) / (df['high'] - df['low'] + 1e-9)
    vol    = df['volume']
    vol_ma = vol.rolling(20).mean()
    vol_ok = (vol > vol_ma * 0.8).astype(float)
    # Bullish trend + price near EMA21 + bullish candle → buy
    score = -dist * 20 * trend * body * vol_ok
    return score.apply(_clamp).rename('ema21_bounce')


def strat_vwap_deviation_reversion(df: pd.DataFrame) -> pd.Series:
    """Price > 1.5 std away from VWAP → mean reversion signal."""
    vwap   = compute_vwap(df)
    close  = df['close']
    dev    = close - vwap
    dev_std = dev.rolling(50).std()
    z_score = dev / (dev_std + 1e-9)
    # Extreme deviation → expect reversion (fade the move)
    score = -z_score / 3.0   # z > +1.5 → sell score; z < -1.5 → buy score
    return score.apply(_clamp).rename('vwap_deviation_reversion')


def strat_rsi_divergence(df: pd.DataFrame) -> pd.Series:
    """Price makes new high/low but RSI doesn't → reversal signal."""
    close   = df['close']
    rsi     = compute_rsi(close, 14)
    n = 10
    price_hh = (close > close.rolling(n).max().shift(1)).astype(float)
    price_ll = (close < close.rolling(n).min().shift(1)).astype(float)
    rsi_hh   = (rsi  > rsi.rolling(n).max().shift(1)).astype(float)
    rsi_ll   = (rsi  < rsi.rolling(n).min().shift(1)).astype(float)
    # Bearish divergence: price HH but RSI doesn't → sell
    bear_div = price_hh * (1 - rsi_hh)
    # Bullish divergence: price LL but RSI doesn't → buy
    bull_div = price_ll * (1 - rsi_ll)
    score = (bull_div - bear_div)
    return score.apply(_clamp).rename('rsi_divergence')


def strat_macd_histogram_reversal(df: pd.DataFrame) -> pd.Series:
    """MACD histogram changes sign after 2+ bars in one direction."""
    _, _, hist = compute_macd(df['close'])
    direction  = hist.gt(0).astype(float) - hist.lt(0).astype(float)  # +1 / 0 / -1
    streak     = direction.rolling(2).sum()
    reversal   = ((direction != direction.shift(1)) & (streak.abs() >= 1)).astype(float)
    score      = reversal * direction * 0.8 + direction * 0.2
    return score.apply(_clamp).rename('macd_histogram_reversal')


def strat_bb_squeeze_expansion(df: pd.DataFrame) -> pd.Series:
    """BB width transitions from squeeze to expansion with directional bias."""
    close  = df['close']
    mid    = close.rolling(20).mean()
    std    = close.rolling(20).std()
    upper  = mid + 2 * std
    lower  = mid - 2 * std
    width  = (upper - lower) / (mid + 1e-9)
    width_pct    = _percentile_rank(width, 100)
    was_squeezed = (width_pct.shift(3) < 0.25)
    expanding    = (width_pct > width_pct.shift(1)).astype(float)
    direction    = np.sign(close - mid)
    score = was_squeezed.astype(float) * expanding * direction * 2
    return score.apply(_clamp).rename('bb_squeeze_expansion')


def strat_oi_price_divergence(df: pd.DataFrame,
                               oi_col: Optional[str] = 'open_interest') -> pd.Series:
    """OI rising while price falls = long squeeze coming (sell); OI falling + price rising = buy."""
    close = df['close']

    def _sign(s: pd.Series) -> pd.Series:
        return s.gt(0).astype(float) - s.lt(0).astype(float)

    if oi_col in df.columns:
        oi_delta = _sign(df[oi_col] - df[oi_col].shift(3))
    else:
        oi_delta = _sign(df['volume'] - df['volume'].rolling(10).mean())
    price_delta = _sign(close - close.shift(3))
    # Divergence: OI up + price down → bearish; OI down + price up → bullish
    divergence  = -(oi_delta * price_delta)
    score       = divergence * 0.6
    return score.apply(_clamp).rename('oi_price_divergence')


def strat_news_momentum_catalyst(df: pd.DataFrame,
                                  sentiment_col: Optional[str] = 'sentiment') -> pd.Series:
    """News sentiment boosting signal confidence when sentiment is strong."""
    if sentiment_col in df.columns:
        sent = df[sentiment_col].clip(-1, 1)
    else:
        # Proxy: strong directional candle with high volume suggests news catalyst
        body   = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-9)
        vol    = df['volume']
        vol_ma = vol.rolling(20).mean()
        surge  = (vol / (vol_ma + 1e-9)).clip(0, 4) / 4
        sent   = body * surge
    score = sent
    return score.apply(_clamp).rename('news_momentum_catalyst')


def strat_session_high_low_break(df: pd.DataFrame) -> pd.Series:
    """Break of the previous 8-bar session high/low (Asian/London range proxy)."""
    high  = df['high']
    low   = df['low']
    close = df['close']
    session_high = high.rolling(8).max().shift(1)
    session_low  = low.rolling(8).min().shift(1)
    vol    = df['volume']
    vol_ma = vol.rolling(20).mean()
    vol_factor = (vol / (vol_ma + 1e-9)).clip(0.5, 3) / 3
    break_up   = ((close > session_high) & (close.shift(1) <= session_high.shift(1))).astype(float)
    break_down = ((close < session_low)  & (close.shift(1) >= session_low.shift(1))).astype(float)
    score = (break_up - break_down) * vol_factor * 2
    return score.apply(_clamp).rename('session_high_low_break')


def strat_ema_golden_death_cross(df: pd.DataFrame) -> pd.Series:
    """EMA50 / EMA200 cross: golden = buy, death = sell. Includes pre-cross proximity."""
    close  = df['close']
    ema50  = close.ewm(span=50,  adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    diff   = (ema50 - ema200) / (ema200 + 1e-9)
    golden = ((ema50 > ema200) & (ema50.shift(1) <= ema200.shift(1))).astype(float)
    death  = ((ema50 < ema200) & (ema50.shift(1) >= ema200.shift(1))).astype(float)
    cross_event = (golden - death) * 0.8
    trend_strength = diff.clip(-0.05, 0.05) / 0.05   # proximity to cross
    score = cross_event + trend_strength * 0.2
    return score.apply(_clamp).rename('ema_golden_death_cross')


# ═══════════════════════════════════════════════════════════════════════════════
#  SWING STRATEGIES (4h timeframe)
# ═══════════════════════════════════════════════════════════════════════════════

def strat_supply_demand_zone_retest(df: pd.DataFrame) -> pd.Series:
    """Price revisiting a high-volume node (supply/demand zone)."""
    close  = df['close']
    vol    = df['volume']
    high   = df['high']
    low    = df['low']
    # Find high-volume price levels via rolling max volume bars
    vol_threshold = vol.rolling(50).quantile(0.80)
    hv_price = close.where(vol > vol_threshold).ffill()
    dist = (close - hv_price) / (hv_price + 1e-9)
    # Near HV zone (within 0.5%)
    near_zone = (dist.abs() < 0.005)
    body = (close - df['open']) / (high - low + 1e-9)
    score = near_zone.astype(float) * body * 3
    return score.apply(_clamp).rename('supply_demand_zone_retest')


def strat_fibonacci_retracement_entry(df: pd.DataFrame) -> pd.Series:
    """Price at key Fibonacci levels (38.2%, 50%, 61.8%) of recent swing."""
    close = df['close']
    n = 30  # swing lookback
    swing_high = close.rolling(n).max()
    swing_low  = close.rolling(n).min()
    rng = swing_high - swing_low
    fib_382 = swing_high - 0.382 * rng
    fib_500 = swing_high - 0.500 * rng
    fib_618 = swing_high - 0.618 * rng
    trend  = np.sign(close - close.rolling(n).mean())
    # Distance from nearest fib level
    dist_382 = (close - fib_382).abs() / (close + 1e-9)
    dist_500 = (close - fib_500).abs() / (close + 1e-9)
    dist_618 = (close - fib_618).abs() / (close + 1e-9)
    min_dist = pd.concat([dist_382, dist_500, dist_618], axis=1).min(axis=1)
    at_fib   = (min_dist < 0.008).astype(float)
    score = at_fib * trend * 2
    return score.apply(_clamp).rename('fibonacci_retracement_entry')


def strat_higher_timeframe_alignment(df: pd.DataFrame) -> pd.Series:
    """4h + daily trend alignment: daily EMA200 direction matches 4h EMA50."""
    close   = df['close']
    ema50   = close.ewm(span=50,  adjust=False).mean()
    ema200  = close.ewm(span=200, adjust=False).mean()
    # Simulate daily trend: 24-bar EMA on 4h data = 4-day average
    ema_daily = close.ewm(span=24, adjust=False).mean()
    ema4h_trend   = np.sign(ema50 - ema200)
    daily_trend   = np.sign(close - ema_daily)
    alignment     = (ema4h_trend == daily_trend).astype(float) * ema4h_trend
    score = alignment * 0.8
    return score.apply(_clamp).rename('higher_timeframe_alignment')


def strat_market_structure_break(df: pd.DataFrame) -> pd.Series:
    """Break of previous swing high (BOS up) or swing low (BOS down)."""
    close = df['close']
    high  = df['high']
    low   = df['low']
    n = 20
    prev_swing_high = high.rolling(n).max().shift(n // 2)
    prev_swing_low  = low.rolling(n).min().shift(n // 2)
    bos_up   = ((close > prev_swing_high) & (close.shift(1) <= prev_swing_high.shift(1))).astype(float)
    bos_down = ((close < prev_swing_low)  & (close.shift(1) >= prev_swing_low.shift(1))).astype(float)
    vol    = df['volume']
    vol_ma = vol.rolling(20).mean()
    vol_ok = (vol > vol_ma * 1.2).astype(float)
    score  = (bos_up - bos_down) * vol_ok * 2
    return score.apply(_clamp).rename('market_structure_break')


def strat_liquidation_cascade_fade(df: pd.DataFrame) -> pd.Series:
    """Rapid OI decline + sharp price move → fade the move (reversal)."""
    close  = df['close']
    vol    = df['volume']
    atr    = compute_atr(df, 14)
    # Detect sharp moves: candle body > 2× ATR
    candle_body = (close - df['open']).abs()
    sharp_move  = (candle_body > 2.0 * atr).astype(float)
    # Volume surge (proxy for liquidation)
    vol_ma      = vol.rolling(20).mean()
    vol_surge   = (vol > vol_ma * 3).astype(float)
    # Direction of the sharp move (fade = opposite)
    move_dir    = np.sign(close - df['open'])
    # Cascade = sharp move + vol surge → expect reversal
    score = sharp_move * vol_surge * (-move_dir) * 0.8
    return score.apply(_clamp).rename('liquidation_cascade_fade')


# ═══════════════════════════════════════════════════════════════════════════════
#  MASTER FEATURE BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

_STRATEGY_FN_MAP = {
    # Scalping
    'vwap_bounce':                    strat_vwap_bounce,
    'bb_squeeze_breakout':            strat_bb_squeeze_breakout,
    'ema_cross_fast':                 strat_ema_cross_fast,
    'macd_zero_cross':                strat_macd_zero_cross,
    'volume_spike_entry':             strat_volume_spike_entry,
    'rsi_momentum_cross':             strat_rsi_momentum_cross,
    'order_flow_imbalance':           strat_order_flow_imbalance,
    'support_resistance_retest':      strat_support_resistance_retest,
    'opening_range_breakout':         strat_opening_range_breakout,
    'momentum_burst_candles':         strat_momentum_burst_candles,
    # Intraday
    'trend_continuation_pullback':    strat_trend_continuation_pullback,
    'ema21_bounce':                   strat_ema21_bounce,
    'vwap_deviation_reversion':       strat_vwap_deviation_reversion,
    'rsi_divergence':                 strat_rsi_divergence,
    'macd_histogram_reversal':        strat_macd_histogram_reversal,
    'bb_squeeze_expansion':           strat_bb_squeeze_expansion,
    'oi_price_divergence':            strat_oi_price_divergence,
    'news_momentum_catalyst':         strat_news_momentum_catalyst,
    'session_high_low_break':         strat_session_high_low_break,
    'ema_golden_death_cross':         strat_ema_golden_death_cross,
    # Swing
    'supply_demand_zone_retest':      strat_supply_demand_zone_retest,
    'fibonacci_retracement_entry':    strat_fibonacci_retracement_entry,
    'higher_timeframe_alignment':     strat_higher_timeframe_alignment,
    'market_structure_break':         strat_market_structure_break,
    'liquidation_cascade_fade':       strat_liquidation_cascade_fade,
}


def compute_all_strategies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 25 strategy scores for a OHLCV DataFrame.

    Returns a DataFrame with one column per strategy, rows aligned to df.index.
    Each value is raw [-1, 1].  Call percentile_rank_features() afterward to
    normalize for the universal model.
    """
    results = {}
    for name, fn in _STRATEGY_FN_MAP.items():
        try:
            results[name] = fn(df)
        except Exception:
            results[name] = pd.Series(0.0, index=df.index)
    return pd.DataFrame(results, index=df.index)


def compute_feature_engine_subset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 20 core technical indicators from feature_engine.py.
    Prefixed with 'fe_' to avoid name collisions with strategy scores.
    All values are already in a bounded/normalized range or will be
    percentile-ranked alongside strategy features.
    """
    close = df['close']
    idx   = df.index
    out: Dict[str, pd.Series] = {}

    def _safe(fn, *args, **kwargs) -> pd.Series:
        try:
            return fn(*args, **kwargs)
        except Exception:
            return pd.Series(0.0, index=idx)

    # Momentum oscillators
    out['fe_rsi_14']       = _safe(compute_rsi, close, 14) / 100.0          # 0–1
    out['fe_rsi_7']        = _safe(compute_rsi, close, 7)  / 100.0
    try:
        macd_line, _, macd_hist = compute_macd(close)
    except Exception:
        macd_line = pd.Series(0.0, index=idx)
        macd_hist = pd.Series(0.0, index=idx)
    atr = _safe(compute_atr, df, 14)
    out['fe_macd_line']    = (macd_line  / (close + 1e-9)).clip(-0.05, 0.05) / 0.05
    out['fe_macd_hist']    = (macd_hist  / (close + 1e-9)).clip(-0.05, 0.05) / 0.05

    # Trend strength
    out['fe_adx_14']       = _safe(compute_adx, df, 14) / 100.0

    # Volatility
    out['fe_atr_pct']      = (atr / (close + 1e-9)).clip(0, 0.1) / 0.1      # 0–1
    out['fe_bb_width']     = _percentile_rank(_safe(compute_bollinger_width, df, 20, 2))
    try:
        out['fe_bb_pct_b'] = _safe(compute_bollinger_pct_b, df).clip(0, 1)
    except Exception:
        out['fe_bb_pct_b'] = pd.Series(0.5, index=idx)

    # Volume / flow
    out['fe_cmf_20']       = _safe(compute_cmf,  df, 20).clip(-1, 1)
    out['fe_mfi_14']       = _safe(compute_mfi,  df, 14) / 100.0
    stk, std = pd.Series(50.0, index=idx), pd.Series(50.0, index=idx)
    try:
        stk, std = compute_stoch_rsi(close)
    except Exception:
        pass
    out['fe_stoch_rsi_k']  = stk.clip(0, 100) / 100.0
    out['fe_stoch_rsi_d']  = std.clip(0, 100) / 100.0

    # VWAP deviation (mean-reversion signal)
    vwap = _safe(compute_vwap, df)
    out['fe_vwap_dev']     = ((close - vwap) / (vwap + 1e-9)).clip(-0.05, 0.05) / 0.05

    # EMA slopes
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    out['fe_ema20_slope']  = ((ema20 - ema20.shift(5)) / (ema20.shift(5) + 1e-9)).clip(-0.05, 0.05) / 0.05
    out['fe_ema50_slope']  = ((ema50 - ema50.shift(5)) / (ema50.shift(5) + 1e-9)).clip(-0.05, 0.05) / 0.05

    # Volume regime
    vol_ma = df['volume'].rolling(20).mean()
    out['fe_vol_ratio']    = (df['volume'] / (vol_ma + 1e-9)).clip(0, 5) / 5.0

    # Price z-score
    out['fe_price_zscore'] = _safe(compute_price_zscore, df, 200).clip(-3, 3) / 3.0

    # Volatility & trend quality
    out['fe_vol_regime']        = _safe(compute_volatility_regime, df).clip(0, 3) / 3.0
    out['fe_efficiency_ratio']  = _safe(compute_efficiency_ratio, close, 10).clip(0, 1)
    out['fe_williams_r']        = (_safe(compute_williams_r, df, 14) + 100) / 100.0  # 0–1

    return pd.DataFrame(out, index=idx)


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 45 features = 25 strategy scores + 20 feature-engine indicators.
    Use this in place of compute_all_strategies() during training and inference.
    """
    strategies = compute_all_strategies(df)
    indicators = compute_feature_engine_subset(df)
    return pd.concat([strategies, indicators], axis=1)


def percentile_rank_features(features: pd.DataFrame, lookback: int = 100) -> pd.DataFrame:
    """
    Normalize each column to its rolling percentile rank [0, 1].
    This makes the feature space token-agnostic.
    """
    ranked = {}
    for col in features.columns:
        ranked[col] = _percentile_rank(features[col], lookback)
    return pd.DataFrame(ranked, index=features.index)
