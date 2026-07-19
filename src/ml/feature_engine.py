import pandas as pd
import numpy as np
from typing import Any, Optional, Tuple, List, Dict
from src.ml.delltandecay import add_delta_and_decay_features
from src.ml.regime_tools import encode_regime_series, compute_regime_metrics

# ------------------------------------------------------------------
# Data Cleaning (fix inf/NaN issues for XGBoost)
# ------------------------------------------------------------------
def clean_infinite_values(df: pd.DataFrame, max_value: float = 1e6, fill_method: str = 'zero') -> pd.DataFrame:
    """
    Replace infinite values with NaN, then fill NaNs.
    Prevents XGBoost error: "Input data contains `inf` or a value too large, while `missing` is not set"

    Args:
        df: Input dataframe
        max_value: Clip absolute values above this threshold
        fill_method: 'zero' (fill 0) or 'mean' (fill with column mean)

    Returns:
        Cleaned dataframe
    """
    df = df.copy()

    # Replace inf with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Clip extreme values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].clip(-max_value, max_value)

    # Fill NaNs
    if fill_method == 'mean':
        # NOTE: column mean is computed over the whole series. For leakage-strict
        # use during training, prefer fill_method='zero' (the default), since a
        # global mean technically peeks at future rows.
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    else:  # zero
        df[numeric_cols] = df[numeric_cols].fillna(0)

    return df

# ------------------------------------------------------------------
# Core Indicators (reusable)
# ------------------------------------------------------------------

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df['high'], df['low'], df['close']
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def compute_macd(series: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def compute_vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df['high'] + df['low'] + df['close']) / 3
    return (tp * df['volume']).cumsum() / df['volume'].cumsum()

def compute_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tp = (df['high'] + df['low'] + df['close']) / 3
    mf = tp * df['volume']
    pos_mf = mf.where(tp > tp.shift(1), 0.0).rolling(period).sum()
    neg_mf = mf.where(tp < tp.shift(1), 0.0).rolling(period).sum()
    mfr = pos_mf / (neg_mf + 1e-9)
    return 100 - (100 / (1 + mfr))

def compute_stoch_rsi(series: pd.Series, period: int = 14, k: int = 3, d: int = 3) -> Tuple[pd.Series, pd.Series]:
    rsi = compute_rsi(series, period)
    min_rsi = rsi.rolling(period).min()
    max_rsi = rsi.rolling(period).max()
    stoch_rsi = (rsi - min_rsi) / (max_rsi - min_rsi + 1e-9)
    fast_k = stoch_rsi.rolling(k).mean() * 100
    slow_d = fast_k.rolling(d).mean()
    return fast_k, slow_d

def compute_cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-9)
    mf_volume = mf_multiplier * df['volume']
    return mf_volume.rolling(period).sum() / df['volume'].rolling(period).sum()

def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close']
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr = compute_atr(df, period=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / (atr + 1e-9))
    minus_di = 100 * (minus_dm.rolling(period).mean() / (atr + 1e-9))
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    return dx.rolling(period).mean()

def compute_efficiency_ratio(series: pd.Series, period: int = 10) -> pd.Series:
    price_change = series.diff(period).abs()
    volatility = series.diff().abs().rolling(period).sum()
    return price_change / (volatility + 1e-9)

def compute_bollinger_width(df: pd.DataFrame, period: int = 20, std_dev: int = 2) -> pd.Series:
    middle = df['close'].rolling(period).mean()
    std = df['close'].rolling(period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return (upper - lower) / (middle + 1e-9)

def compute_bollinger_width_percentile(df: pd.DataFrame, lookback: int = 252) -> pd.Series:
    width = compute_bollinger_width(df, period=20, std_dev=2)
    percentile = width.rolling(lookback).apply(
        lambda x: (x[-1] <= x).sum() / len(x) if len(x) > 0 else 0.5, raw=True
    )
    return percentile.fillna(0.5)

def compute_fvg_distance(df: pd.DataFrame) -> pd.Series:
    """Fair Value Gap distance (replaced inf with 1e6 to avoid XGBoost errors)."""
    fvg_high = pd.Series(index=df.index, dtype=float)
    fvg_low = pd.Series(index=df.index, dtype=float)
    for i in range(2, len(df)-1):
        if df['high'].iloc[i-2] < df['low'].iloc[i] and df['high'].iloc[i-1] < df['low'].iloc[i]:
            fvg_low.iloc[i] = df['low'].iloc[i]
            fvg_high.iloc[i] = df['high'].iloc[i-2]
        elif df['low'].iloc[i-2] > df['high'].iloc[i] and df['low'].iloc[i-1] > df['high'].iloc[i]:
            fvg_high.iloc[i] = df['high'].iloc[i]
            fvg_low.iloc[i] = df['low'].iloc[i-2]
    distance = pd.Series(index=df.index, dtype=float).fillna(0.0)
    current_price = df['close']
    MAX_DIST = 1e6  # Use large finite value instead of inf
    # NOTE: scan only backward (j <= i). Scanning forward (i+50) would let a FVG
    # formed in the future affect the present row -> lookahead. Fixed here.
    for i in range(len(df)):
        best_dist = MAX_DIST
        for j in range(max(0, i-50), i + 1):
            if not pd.isna(fvg_low.iloc[j]) and not pd.isna(fvg_high.iloc[j]):
                if current_price.iloc[i] < fvg_low.iloc[j]:
                    dist = (fvg_low.iloc[j] - current_price.iloc[i]) / (current_price.iloc[i] + 1e-9)
                elif current_price.iloc[i] > fvg_high.iloc[j]:
                    dist = (current_price.iloc[i] - fvg_high.iloc[j]) / (current_price.iloc[i] + 1e-9)
                else:
                    dist = 0.0
                if dist < best_dist:
                    best_dist = dist
        distance.iloc[i] = best_dist if best_dist != MAX_DIST else 0.0
    return distance

def compute_volume_volatility_efficiency(df: pd.DataFrame) -> pd.Series:
    atr = compute_atr(df, period=14)
    return df['volume'] / (atr + 1e-9)

def compute_price_zscore(df: pd.DataFrame, ema_period: int = 200) -> pd.Series:
    ema = df['close'].ewm(span=ema_period, adjust=False).mean()
    std = df['close'].rolling(ema_period).std()
    return (df['close'] - ema) / (std + 1e-9)

def compute_volatility_regime(df: pd.DataFrame, atr_period: int = 14, lookback: int = 100) -> pd.Series:
    atr = compute_atr(df, atr_period)
    atr_mean = atr.rolling(lookback, min_periods=1).mean()
    return atr / (atr_mean + 1e-9)

# ------------------------------------------------------------------
# Additional structural feature helpers
# ------------------------------------------------------------------

def compute_ema_slope(series: pd.Series, period: int = 14) -> pd.Series:
    ema = series.ewm(span=period, adjust=False).mean()
    return ((ema - ema.shift(1)) / (ema.shift(1).replace(0, np.nan) + 1e-9)).fillna(0.0)


def compute_ema_alignment_score(df: pd.DataFrame, periods: list = [9, 21, 50, 100, 200]) -> pd.Series:
    emas = [df['close'].ewm(span=p, adjust=False).mean() for p in periods]
    score = pd.Series(0.0, index=df.index)
    for i in range(len(periods) - 1):
        score += (emas[i] > emas[i + 1]).astype(float)
    return (score / float(len(periods) - 1)).fillna(0.0)


def compute_ema_stack_distance(df: pd.DataFrame, periods: list = [9, 21, 50, 100, 200]) -> pd.Series:
    emas = [df['close'].ewm(span=p, adjust=False).mean() for p in periods]
    close = df['close'].replace(0, np.nan)
    distances = [((close / ema) - 1).abs() for ema in emas]
    return pd.concat(distances, axis=1).mean(axis=1).fillna(0.0)


def compute_multi_timeframe_trend_agreement(df: pd.DataFrame,
                                           df_1d: Optional[pd.DataFrame],
                                           df_1w: Optional[pd.DataFrame]) -> pd.Series:
    agreement = pd.Series(0.5, index=df.index)
    parts = []
    if df_1d is not None and not df_1d.empty:
        d1 = df_1d['close'].pct_change(5).iloc[-1] if len(df_1d) >= 6 else 0.0
        parts.append(1.0 if d1 > 0 else 0.0)
    if df_1w is not None and not df_1w.empty:
        d7 = df_1w['close'].pct_change(2).iloc[-1] if len(df_1w) >= 3 else 0.0
        parts.append(1.0 if d7 > 0 else 0.0)
    if parts:
        agreement[:] = float(np.mean(parts))
    return agreement.fillna(0.5)


def compute_atr_percentile(df: pd.DataFrame, atr_period: int = 14, lookback: int = 100) -> pd.Series:
    atr = compute_atr(df, atr_period)
    return atr.rolling(lookback, min_periods=1).apply(
        lambda x: float((x <= x.iloc[-1]).sum()) / len(x) if len(x) else 0.5,
        raw=False,
    ).fillna(0.5)


def compute_atr_expansion(df: pd.DataFrame, atr_period: int = 14, lookback: int = 20) -> pd.Series:
    atr = compute_atr(df, atr_period)
    atr_ma = atr.rolling(lookback, min_periods=1).mean()
    return ((atr / (atr_ma + 1e-9)) - 1.0).fillna(0.0)


def compute_realized_volatility(df: pd.DataFrame, period: int = 24, annualize: bool = True) -> pd.Series:
    log_ret = np.log(df['close'] / df['close'].shift(1)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    hv = log_ret.rolling(period, min_periods=1).std()
    if annualize:
        hv = hv * np.sqrt(365 * 24)
    return hv.fillna(0.0)


def compute_volatility_ratio(df: pd.DataFrame, current_period: int = 24, lookback: int = 100) -> pd.Series:
    current = compute_realized_volatility(df, period=current_period, annualize=False)
    past = current.rolling(lookback, min_periods=1).mean()
    return ((current / (past + 1e-9)) - 1.0).fillna(0.0)


def compute_structure_counts(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    hh = (df['high'] > df['high'].shift(1)).astype(float).rolling(lookback, min_periods=1).sum()
    ll = (df['low'] < df['low'].shift(1)).astype(float).rolling(lookback, min_periods=1).sum()
    return pd.DataFrame({'higher_high_count': hh.fillna(0.0), 'lower_low_count': ll.fillna(0.0)})


def compute_swing_strength(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    momentum = df['close'] - df['close'].shift(lookback)
    avg_range = (df['high'] - df['low']).rolling(lookback, min_periods=1).mean()
    return (momentum / (avg_range + 1e-9)).fillna(0.0)


def compute_rsi_slope(series: pd.Series, period: int = 14) -> pd.Series:
    return compute_linear_regression_slope(compute_rsi(series, period), period)


def compute_rsi_acceleration(series: pd.Series, period: int = 14) -> pd.Series:
    return compute_rsi_slope(series, period).diff().fillna(0.0)


def compute_macd_hist_velocity(df: pd.DataFrame) -> pd.Series:
    _, _, macd_hist = compute_macd(df['close'])
    return macd_hist.diff().fillna(0.0)


def compute_macd_divergence(df: pd.DataFrame) -> pd.Series:
    macd, macd_signal, _ = compute_macd(df['close'])
    return (macd - macd_signal).fillna(0.0)


def compute_stoch_slope(df: pd.DataFrame) -> pd.Series:
    fast_k, _ = compute_stoch_rsi(df['close'], 14)
    return compute_linear_regression_slope(fast_k, 7)


def compute_vwap_slope(df: pd.DataFrame, period: int = 24) -> pd.Series:
    vwap = compute_vwap(df)
    return compute_linear_regression_slope(vwap, period)


def compute_obv_slope(df: pd.DataFrame, period: int = 20) -> pd.Series:
    obv = (df['close'].diff().fillna(0).apply(np.sign) * df['volume']).cumsum()
    return compute_linear_regression_slope(obv, period)


def compute_obv_divergence(df: pd.DataFrame, period: int = 24) -> pd.Series:
    obv = (df['close'].diff().fillna(0).apply(np.sign) * df['volume']).cumsum()
    price_ret = df['close'].pct_change(period)
    obv_change = obv.diff(period)
    return ((price_ret - (obv_change / (df['close'].replace(0, np.nan) + 1e-9))).fillna(0.0))


def compute_market_regime(adx: pd.Series,
                          vol_regime: pd.Series,
                          ema_alignment: pd.Series,
                          macro_trend_1d: Optional[pd.Series] = None) -> pd.Series:
    regime = pd.Series('RANGE', index=adx.index)
    macro = macro_trend_1d if macro_trend_1d is not None else pd.Series(0.0, index=adx.index)
    bull = (adx > 25) & (ema_alignment > 0.6) & (macro > 0.0)
    bear = (adx > 25) & (ema_alignment < 0.4) & (macro < 0.0)
    high_vol = (vol_regime > 1.2)
    low_vol = (vol_regime < 0.8)
    regime[bull] = 'TRENDING_BULL'
    regime[bear] = 'TRENDING_BEAR'
    regime[high_vol & ~bull & ~bear] = 'HIGH_VOL'
    regime[low_vol & ~bull & ~bear] = 'LOW_VOL'
    regime[(~bull & ~bear & ~high_vol & ~low_vol)] = 'RANGE'
    return regime

# ==================== NEW INDICATORS ====================

def compute_aroon(df: pd.DataFrame, period: int = 25) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Aroon Up, Aroon Down, Aroon Oscillator."""
    high = df['high']
    low = df['low']
    aroon_up = 100 * high.rolling(period+1).apply(lambda x: x.argmax() / period, raw=True)
    aroon_down = 100 * low.rolling(period+1).apply(lambda x: x.argmin() / period, raw=True)
    aroon_osc = aroon_up - aroon_down
    return aroon_up, aroon_down, aroon_osc

def compute_keltner_channels(df: pd.DataFrame, ema_period: int = 20, atr_period: int = 10, multiplier: float = 1.5) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Keltner Channels: middle = EMA, upper/lower = middle ± multiplier * ATR."""
    middle = df['close'].ewm(span=ema_period, adjust=False).mean()
    atr = compute_atr(df, atr_period)
    upper = middle + multiplier * atr
    lower = middle - multiplier * atr
    width = (upper - lower) / (middle + 1e-9)
    return upper, lower, width

def compute_choppiness_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Choppiness Index: 0-100, high = ranging, low = trending."""
    high = df['high'].rolling(period).max()
    low = df['low'].rolling(period).min()
    sum_atr = compute_atr(df, period).rolling(period).sum()
    choppiness = 100 * np.log10(sum_atr / (high - low + 1e-9)) / np.log10(period)
    return choppiness.clip(0, 100)

def compute_ichimoku(df: pd.DataFrame, tenkan_period=9, kijun_period=26, senkou_b=52) -> pd.DataFrame:
    """
    Ichimoku Cloud components — LEAKAGE-SAFE.

    The classic 'chikou span' (close shifted -26) is intentionally OMITTED: as a
    model feature it would expose the close price 26 bars in the future at the
    current row, which is direct lookahead leakage.

    senkou_a / senkou_b use .shift(+kijun_period), which moves PAST values forward.
    That is correct and NOT a leak: the current row only sees data from 26 bars ago.
    """
    tenkan = (df['high'].rolling(tenkan_period).max() + df['low'].rolling(tenkan_period).min()) / 2
    kijun = (df['high'].rolling(kijun_period).max() + df['low'].rolling(kijun_period).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(kijun_period)
    senkou_b = ((df['high'].rolling(senkou_b).max() + df['low'].rolling(senkou_b).min()) / 2).shift(kijun_period)
    ichimoku = pd.DataFrame({
        'ichimoku_tenkan': tenkan,
        'ichimoku_kijun': kijun,
        'ichimoku_senkou_a': senkou_a,
        'ichimoku_senkou_b': senkou_b,
        # 'ichimoku_chikou' deliberately removed (future leak).
    })
    return ichimoku

def compute_linear_regression_slope(series: pd.Series, period: int = 14) -> pd.Series:
    """Slope of linear regression over rolling window."""
    def slope(y):
        x = np.arange(len(y))
        if len(y) < 2:
            return 0.0
        slope_ = np.polyfit(x, y, 1)[0]
        return slope_
    return series.rolling(period).apply(slope, raw=True)

def compute_r_squared(series: pd.Series, period: int = 14) -> pd.Series:
    """R-squared of linear regression over rolling window."""
    def r2(y):
        x = np.arange(len(y))
        if len(y) < 2:
            return 0.0
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = intercept + slope * x
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        return 1 - (ss_res / (ss_tot + 1e-9))
    return series.rolling(period).apply(r2, raw=True)

def compute_rolling_correlation_with_btc(df: pd.DataFrame, btc_df: pd.DataFrame, period: int = 24) -> pd.Series:
    """Rolling correlation of close prices with BTC."""
    if btc_df is None or btc_df.empty:
        return pd.Series(0.0, index=df.index)
    merged = df[['timestamp', 'close']].merge(btc_df[['timestamp', 'close']], on='timestamp', suffixes=('', '_btc'))
    corr = merged['close'].rolling(period).corr(merged['close_btc']).fillna(0)
    return corr

def compute_historical_volatility(df: pd.DataFrame, period: int = 24, annualize: bool = True) -> pd.Series:
    """Annualized historical volatility (standard deviation of log returns)."""
    log_ret = np.log((df['close'] / df['close'].shift(1)).fillna(1))
    log_ret = pd.Series(log_ret, index=df.index).fillna(0)
    hv = log_ret.rolling(period).std()
    if annualize:
        hv = hv * np.sqrt(365 * 24)  # hourly to annual
    return hv.fillna(0)

def compute_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Williams %R (-100 to 0)."""
    highest_high = df['high'].rolling(period).max()
    lowest_low = df['low'].rolling(period).min()
    return -100 * (highest_high - df['close']) / (highest_high - lowest_low + 1e-9)

def compute_ultimate_oscillator(df: pd.DataFrame, short=7, medium=14, long=28) -> pd.Series:
    """Ultimate Oscillator (0-100)."""
    bp = df['close'] - pd.concat([df['low'], df['close'].shift(1)], axis=1).min(axis=1)
    tr = pd.concat([df['high'] - df['low'],
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))], axis=1).max(axis=1)
    avg7 = bp.rolling(short).sum() / tr.rolling(short).sum()
    avg14 = bp.rolling(medium).sum() / tr.rolling(medium).sum()
    avg28 = bp.rolling(long).sum() / tr.rolling(long).sum()
    uo = 100 * (4*avg7 + 2*avg14 + avg28) / (4+2+1)
    return uo.fillna(50)

def compute_vortex_indicator(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series]:
    """Vortex Indicator: VI+ and VI-."""
    high = df['high']
    low = df['low']
    close = df['close']
    tr = compute_atr(df, period=1)
    vm_plus = abs(high - low.shift(1))
    vm_minus = abs(low - high.shift(1))
    vi_plus = vm_plus.rolling(period).sum() / tr.rolling(period).sum()
    vi_minus = vm_minus.rolling(period).sum() / tr.rolling(period).sum()
    return vi_plus, vi_minus

def compute_force_index(df: pd.DataFrame, period: int = 13) -> pd.Series:
    """Elder's Force Index = volume * (close - close.shift(1))."""
    force = df['volume'] * df['close'].diff()
    return force.rolling(period).mean()

def compute_accumulation_distribution(df: pd.DataFrame) -> pd.Series:
    """Accumulation/Distribution Line."""
    clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-9)
    return (clv * df['volume']).cumsum()


def compute_volume_delta(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series]:
    """CVD proxy: bar-level directional volume — (close-open)/(high-low)*volume.
    Positive = buyer-dominated bar, negative = seller-dominated. Rolling sum tracks
    cumulative positioning pressure over the window."""
    bar_range = df['high'] - df['low'] + 1e-9
    delta = (df['close'] - df['open']) / bar_range * df['volume']
    return delta, delta.rolling(period, min_periods=1).sum()


def compute_bar_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Continuous bar microstructure features — more information than binary candlestick flags."""
    bar_range = df['high'] - df['low'] + 1e-9
    body = (df['close'] - df['open']).abs()
    hi_body = df[['close', 'open']].max(axis=1)
    lo_body = df[['close', 'open']].min(axis=1)
    return pd.DataFrame({
        'close_position':  (df['close'] - df['low']) / bar_range,        # 0=closed at low, 1=at high
        'bar_body_pct':    body / bar_range,                              # body as share of range
        'upper_wick_pct':  (df['high'] - hi_body) / bar_range,          # upper wick share
        'lower_wick_pct':  (lo_body - df['low']) / bar_range,           # lower wick share
        'bar_direction':   np.sign(df['close'] - df['open']).astype(float),  # +1 bull / -1 bear
    }, index=df.index)


def compute_rolling_vwap(df: pd.DataFrame, period: int = 24) -> Tuple[pd.Series, pd.Series]:
    """Rolling 24-bar VWAP and distance of close from it.
    Less biased than a cumulative session VWAP that resets arbitrarily."""
    tp = (df['high'] + df['low'] + df['close']) / 3
    rvwap = (tp * df['volume']).rolling(period, min_periods=1).sum() / \
            df['volume'].rolling(period, min_periods=1).sum()
    return rvwap, (df['close'] / (rvwap + 1e-9)) - 1


def compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical hour-of-day and day-of-week encoding.
    Crypto liquidity and volatility differ sharply by session (Asia/EU/US)
    and within the week — this lets the model see the clock."""
    ts = pd.to_datetime(df['timestamp'])
    h, d = ts.dt.hour, ts.dt.dayofweek
    return pd.DataFrame({
        'hour_sin': np.sin(2 * np.pi * h / 24),
        'hour_cos': np.cos(2 * np.pi * h / 24),
        'dow_sin':  np.sin(2 * np.pi * d / 7),
        'dow_cos':  np.cos(2 * np.pi * d / 7),
    }, index=df.index)


def add_futures_features(df: pd.DataFrame,
                         funding_df: Optional[pd.DataFrame] = None,
                         oi_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Merge perpetual funding rate and open interest onto hourly df (backward asof join).
    Falls back to zero-fill when data is unavailable so the pipeline always produces
    the same columns regardless of whether futures data was fetched."""
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    if funding_df is not None and not funding_df.empty:
        fr = funding_df.copy()
        fr['timestamp'] = pd.to_datetime(fr['timestamp'])
        df = pd.merge_asof(df.sort_values('timestamp'),
                           fr.sort_values('timestamp')[['timestamp', 'funding_rate']],
                           on='timestamp', direction='backward')
        df['funding_rate'] = df['funding_rate'].fillna(0.0)
        df['funding_rate_ma8'] = df['funding_rate'].rolling(8, min_periods=1).mean()
        lb = 24 * 7
        df['funding_rate_zscore'] = (
            (df['funding_rate'] - df['funding_rate'].rolling(lb, min_periods=8).mean()) /
            (df['funding_rate'].rolling(lb, min_periods=8).std() + 1e-9)
        )
    else:
        df[['funding_rate', 'funding_rate_ma8', 'funding_rate_zscore']] = 0.0

    if oi_df is not None and not oi_df.empty:
        oi = oi_df.copy()
        oi['timestamp'] = pd.to_datetime(oi['timestamp'])
        df = pd.merge_asof(df.sort_values('timestamp'),
                           oi.sort_values('timestamp')[['timestamp', 'open_interest']],
                           on='timestamp', direction='backward')
        df['open_interest'] = df['open_interest'].ffill().fillna(0.0)
        df['oi_change_1h'] = df['open_interest'].pct_change(1).fillna(0.0)
        df['oi_change_4h'] = df['open_interest'].pct_change(4).fillna(0.0)
        lb = 24 * 7
        df['oi_zscore'] = (
            (df['open_interest'] - df['open_interest'].rolling(lb, min_periods=24).mean()) /
            (df['open_interest'].rolling(lb, min_periods=24).std() + 1e-9)
        )
        df['oi_acceleration'] = df['oi_change_1h'].diff().fillna(0.0)
        df['oi_price_divergence'] = (
            df['close'].pct_change(4).fillna(0.0) - df['oi_change_4h'].fillna(0.0)
        )
        df['oi_long_short_pressure'] = (df['open_interest'] * np.sign(df['close'].diff().fillna(0.0))).fillna(0.0)
    else:
        df[['open_interest', 'oi_change_1h', 'oi_change_4h', 'oi_zscore',
             'oi_acceleration', 'oi_price_divergence', 'oi_long_short_pressure']] = 0.0

    return df


# ==================================================================
# EXTENDED INDICATOR LIBRARY
# ==================================================================

# ── Moving Average Variants ────────────────────────────────────────
def compute_hma(series: pd.Series, period: int = 20) -> pd.Series:
    """Hull Moving Average — reduced lag."""
    def _wma(s, p):
        weights = np.arange(1, p + 1, dtype=float)
        return s.rolling(p).apply(lambda x: np.dot(x, weights[-len(x):]) / weights[-len(x):].sum(), raw=True)
    return _wma(2 * _wma(series, period // 2) - _wma(series, period), max(1, int(np.sqrt(period))))


def compute_kama(series: pd.Series, period: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    """Kaufman Adaptive Moving Average — adapts speed to market efficiency."""
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    vals: np.ndarray = np.asarray(series, dtype=float)
    kama = vals.copy()
    for i in range(period, len(vals)):
        direction = abs(vals[i] - vals[i - period])
        chunk: np.ndarray = vals[i - period:i + 1]
        volatility = float(np.abs(chunk[1:] - chunk[:-1]).sum()) + 1e-9
        er = direction / volatility
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (vals[i] - kama[i - 1])
    return pd.Series(kama, index=series.index)


def compute_tema(series: pd.Series, period: int = 21) -> pd.Series:
    """Triple Exponential Moving Average."""
    e1 = series.ewm(span=period, adjust=False).mean()
    e2 = e1.ewm(span=period, adjust=False).mean()
    e3 = e2.ewm(span=period, adjust=False).mean()
    return 3 * e1 - 3 * e2 + e3


def compute_dema(series: pd.Series, period: int = 21) -> pd.Series:
    """Double Exponential Moving Average."""
    e1 = series.ewm(span=period, adjust=False).mean()
    return 2 * e1 - e1.ewm(span=period, adjust=False).mean()


def compute_t3(series: pd.Series, period: int = 5, vfactor: float = 0.7) -> pd.Series:
    """Tillson T3 — smoother and less noisy than EMA."""
    c1, c2 = -(vfactor ** 3), 3 * vfactor ** 2 + 3 * vfactor ** 3
    c3, c4 = -6 * vfactor ** 2 - 3 * vfactor - 3 * vfactor ** 3, 1 + 3 * vfactor + vfactor ** 3 + 3 * vfactor ** 2
    e = [series]
    for _ in range(5):
        e.append(e[-1].ewm(span=period, adjust=False).mean())
    return c1 * e[6] + c2 * e[5] + c3 * e[4] + c4 * e[3]


def compute_vwma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Volume-Weighted Moving Average."""
    return (df['close'] * df['volume']).rolling(period).sum() / df['volume'].rolling(period).sum()


# ── Trend Indicators ──────────────────────────────────────────────
def compute_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> Tuple[pd.Series, pd.Series]:
    """SuperTrend — ATR-based trend direction. Returns (level, direction +1/-1)."""
    atr = compute_atr(df, period).values
    hl2 = ((df['high'] + df['low']) / 2).values
    close = df['close'].values
    n = len(df)

    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    st = np.full(n, np.nan)
    direction = np.full(n, 1, dtype=int)   # 1 = upper-band (bearish); loop updates from i=1

    for i in range(1, n):
        upper[i] = upper[i] if upper[i] < upper[i-1] or close[i-1] > upper[i-1] else upper[i-1]
        lower[i] = lower[i] if lower[i] > lower[i-1] or close[i-1] < lower[i-1] else lower[i-1]
        if np.isnan(st[i-1]):
            direction[i] = 1
        elif st[i-1] == upper[i-1]:
            direction[i] = -1 if close[i] > upper[i] else 1
        else:
            direction[i] = 1 if close[i] < lower[i] else -1
        st[i] = lower[i] if direction[i] == -1 else upper[i]

    return pd.Series(st, index=df.index), pd.Series(direction, index=df.index)


def compute_parabolic_sar(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> Tuple[pd.Series, pd.Series]:
    """Parabolic SAR. Returns (sar, trend +1/-1)."""
    high, low = df['high'].values, df['low'].values
    n = len(df)
    sar = np.full(n, np.nan)
    trend = np.zeros(n, dtype=int)
    if n < 2:
        return pd.Series(sar, index=df.index), pd.Series(trend, index=df.index)

    bull, af, ep = True, step, high[0]
    sar[0], trend[0] = low[0], 1

    for i in range(1, n):
        if bull:
            sar[i] = min(sar[i-1] + af * (ep - sar[i-1]), low[i-1], low[max(0, i-2)])
            if low[i] < sar[i]:
                bull, sar[i], ep, af, trend[i] = False, ep, low[i], step, -1
            else:
                trend[i] = 1
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + step, max_step)
        else:
            sar[i] = max(sar[i-1] + af * (ep - sar[i-1]), high[i-1], high[max(0, i-2)])
            if high[i] > sar[i]:
                bull, sar[i], ep, af, trend[i] = True, ep, high[i], step, 1
            else:
                trend[i] = -1
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + step, max_step)

    return pd.Series(sar, index=df.index), pd.Series(trend, index=df.index)


def compute_donchian_channels(df: pd.DataFrame, period: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Donchian Channels (shift-1 to avoid lookahead)."""
    upper = df['high'].shift(1).rolling(period).max()
    lower = df['low'].shift(1).rolling(period).min()
    return upper, lower, (upper + lower) / 2


# ── Momentum / Oscillator Indicators ─────────────────────────────
def compute_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Commodity Channel Index."""
    tp = (df['high'] + df['low'] + df['close']) / 3
    ma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - ma) / (0.015 * mad + 1e-9)


def compute_tsi(series: pd.Series, fast: int = 13, slow: int = 25) -> pd.Series:
    """True Strength Index — double-smoothed momentum."""
    diff = series.diff()
    s1 = diff.ewm(span=slow, adjust=False).mean().ewm(span=fast, adjust=False).mean()
    s2 = diff.abs().ewm(span=slow, adjust=False).mean().ewm(span=fast, adjust=False).mean()
    return 100 * s1 / (s2 + 1e-9)


def compute_cmo(series: pd.Series, period: int = 14) -> pd.Series:
    """Chande Momentum Oscillator (–100 to +100)."""
    diff = series.diff()
    gains = diff.clip(lower=0).rolling(period).sum()
    losses = diff.clip(upper=0).abs().rolling(period).sum()
    return 100 * (gains - losses) / (gains + losses + 1e-9)


def compute_dpo(series: pd.Series, period: int = 20) -> pd.Series:
    """Detrended Price Oscillator — removes trend to expose cycles."""
    return series - series.rolling(period).mean().shift(period // 2 + 1)


def compute_ppo(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    """Percentage Price Oscillator."""
    ema_f = series.ewm(span=fast, adjust=False).mean()
    ema_s = series.ewm(span=slow, adjust=False).mean()
    ppo = 100 * (ema_f - ema_s) / (ema_s + 1e-9)
    return ppo, ppo.ewm(span=signal, adjust=False).mean()


def compute_trix(series: pd.Series, period: int = 15) -> pd.Series:
    """TRIX — 1-period % change of triple-smoothed EMA."""
    e3 = series.ewm(span=period, adjust=False).mean()
    e3 = e3.ewm(span=period, adjust=False).mean()
    e3 = e3.ewm(span=period, adjust=False).mean()
    return e3.pct_change() * 100


def compute_kst(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Know Sure Thing (KST) — multi-period momentum composite."""
    rc1 = series.pct_change(10).rolling(10).mean()
    rc2 = series.pct_change(15).rolling(10).mean()
    rc3 = series.pct_change(20).rolling(10).mean()
    rc4 = series.pct_change(30).rolling(15).mean()
    kst = rc1 + 2 * rc2 + 3 * rc3 + 4 * rc4
    return kst, kst.rolling(9).mean()


def compute_schaff_trend_cycle(series: pd.Series, period: int = 10,
                               fast: int = 23, slow: int = 50) -> pd.Series:
    """Schaff Trend Cycle — MACD run through a double Stochastic."""
    macd = series.ewm(span=fast, adjust=False).mean() - series.ewm(span=slow, adjust=False).mean()

    def _stoch(s, p):
        lo, hi = s.rolling(p).min(), s.rolling(p).max()
        return 100 * (s - lo) / (hi - lo + 1e-9)

    d1 = _stoch(macd, period).ewm(span=3, adjust=False).mean()
    return _stoch(d1, period).ewm(span=3, adjust=False).mean()


def compute_awesome_oscillator(df: pd.DataFrame, fast: int = 5, slow: int = 34) -> pd.Series:
    """Awesome Oscillator — midpoint SMA difference."""
    mid = (df['high'] + df['low']) / 2
    return mid.rolling(fast).mean() - mid.rolling(slow).mean()


def compute_bop(df: pd.DataFrame) -> pd.Series:
    """Balance of Power — (close-open)/(high-low)."""
    return (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-9)


def compute_eom(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Ease of Movement — price change per unit of volume."""
    mid_diff = ((df['high'] + df['low']) / 2).diff()
    hl_range = (df['high'] - df['low']).clip(lower=1e-9)
    volume   = df['volume'].clip(lower=1e-9)
    return (mid_diff * hl_range / volume).rolling(period).mean()


def compute_fisher_transform(df: pd.DataFrame, period: int = 9) -> Tuple[pd.Series, pd.Series]:
    """Fisher Transform — normalises price to Gaussian distribution."""
    hl2 = (df['high'] + df['low']) / 2
    hi = hl2.rolling(period).max()
    lo = hl2.rolling(period).min()
    v = (2 * (hl2 - lo) / (hi - lo + 1e-9) - 1).clip(-0.999, 0.999)
    fish = 0.5 * ((1 + v) / (1 - v + 1e-9)).apply(np.log)
    return fish, fish.shift(1)


def compute_rvi(df: pd.DataFrame, period: int = 10) -> Tuple[pd.Series, pd.Series]:
    """Relative Vigor Index."""
    co = df['close'] - df['open']
    hl = df['high'] - df['low']
    num = (co + 2*co.shift(1) + 2*co.shift(2) + co.shift(3)) / 6
    den = (hl + 2*hl.shift(1) + 2*hl.shift(2) + hl.shift(3)) / 6
    rvi = num.rolling(period).sum() / (den.rolling(period).sum() + 1e-9)
    sig = (rvi + 2*rvi.shift(1) + 2*rvi.shift(2) + rvi.shift(3)) / 6
    return rvi, sig


def compute_rate_of_change(series: pd.Series, period: int = 14) -> pd.Series:
    """Rate of Change (%)."""
    return series.pct_change(period) * 100


# ── Volatility Indicators ─────────────────────────────────────────
def compute_bollinger_pct_b(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.Series:
    """Bollinger %B — where close sits within the bands (0=lower, 1=upper)."""
    ma = df['close'].rolling(period).mean()
    std = df['close'].rolling(period).std()
    return (df['close'] - (ma - std_dev * std)) / (2 * std_dev * std + 1e-9)


def compute_parkinson_volatility(df: pd.DataFrame, period: int = 24) -> pd.Series:
    """Parkinson volatility — high/low range estimator (more efficient than close-to-close)."""
    log_hl2 = (df['high'] / df['low'].clip(lower=1e-9)).apply(np.log) ** 2
    return (log_hl2.rolling(period).mean() / (4 * np.log(2))) ** 0.5


def compute_garman_klass_volatility(df: pd.DataFrame, period: int = 24) -> pd.Series:
    """Garman-Klass volatility — uses full OHLC (most efficient classical estimator).
    Inner expression clipped to 0 before root: on gap-open bars where the O→C move
    dominates the H-L range the raw GK term can go negative, producing NaN."""
    log_hl = (df['high'] / df['low'].clip(lower=1e-9)).apply(np.log) ** 2
    log_co = (df['close'] / df['open'].clip(lower=1e-9)).apply(np.log) ** 2
    gk = (0.5 * log_hl - (2 * np.log(2) - 1) * log_co).rolling(period).mean().clip(lower=0)
    return gk ** 0.5


def compute_atr_bands(df: pd.DataFrame, period: int = 14, multiplier: float = 2.0) -> Tuple[pd.Series, pd.Series]:
    """ATR Bands around rolling MA."""
    ma = df['close'].rolling(period).mean()
    atr = compute_atr(df, period)
    return ma + multiplier * atr, ma - multiplier * atr


def compute_starc_bands(df: pd.DataFrame, ma_period: int = 15,
                        atr_period: int = 5, multiplier: float = 2.0) -> Tuple[pd.Series, pd.Series]:
    """STARC Bands — SMA ± multiplier * ATR."""
    sma = df['close'].rolling(ma_period).mean()
    atr = compute_atr(df, atr_period)
    return sma + multiplier * atr, sma - multiplier * atr


def compute_relative_volatility_index(series: pd.Series, std_period: int = 10, rsi_period: int = 14) -> pd.Series:
    """Relative Volatility Index — RSI applied to rolling std-dev."""
    return compute_rsi(series.rolling(std_period).std().fillna(0), rsi_period)


def compute_gaussian_channel(series: pd.Series, period: int = 20,
                              multiplier: float = 2.0) -> Tuple[pd.Series, pd.Series]:
    """Gaussian Channel — EWM-smoothed midline ± std envelope."""
    mid = series.ewm(span=period, adjust=False).mean()
    std = series.rolling(period, min_periods=1).std()
    return mid + multiplier * std, mid - multiplier * std


# ── Volume Indicators ─────────────────────────────────────────────
def compute_pvt(df: pd.DataFrame) -> pd.Series:
    """Price Volume Trend — running sum of volume * % price change."""
    return (df['close'].pct_change() * df['volume']).fillna(0).cumsum()


def compute_klinger_oscillator(df: pd.DataFrame,
                               fast: int = 34, slow: int = 55) -> Tuple[pd.Series, pd.Series]:
    """Klinger Volume Oscillator."""
    tp = (df['high'] + df['low'] + df['close']) / 3
    vf = df['volume'] * np.where(tp.diff() > 0, 1, -1) * (df['high'] - df['low'])
    vf = pd.Series(vf, index=df.index)
    kvo = vf.ewm(span=fast, adjust=False).mean() - vf.ewm(span=slow, adjust=False).mean()
    return kvo, kvo.ewm(span=13, adjust=False).mean()


# ── Market Structure Indicators ───────────────────────────────────
def compute_pivot_points(df_1d: Optional[pd.DataFrame],
                         df_1h: pd.DataFrame) -> pd.DataFrame:
    """Standard daily pivot points mapped backward onto hourly bars."""
    cols = ['pivot', 'r1', 's1', 'r2', 's2']
    empty = pd.DataFrame(0.0, index=df_1h.index, columns=cols)
    if df_1d is None or df_1d.empty:
        return empty
    try:
        d = df_1d.copy()
        d['timestamp'] = pd.to_datetime(d['timestamp'])
        d = d.sort_values('timestamp')
        d['pivot'] = (d['high'].shift(1) + d['low'].shift(1) + d['close'].shift(1)) / 3
        d['r1'] = 2 * d['pivot'] - d['low'].shift(1)
        d['s1'] = 2 * d['pivot'] - d['high'].shift(1)
        d['r2'] = d['pivot'] + d['high'].shift(1) - d['low'].shift(1)
        d['s2'] = d['pivot'] - (d['high'].shift(1) - d['low'].shift(1))
        h = df_1h.copy()
        h['timestamp'] = pd.to_datetime(h['timestamp'])
        merged = pd.merge_asof(h[['timestamp']].sort_values('timestamp'),
                               d[['timestamp'] + cols].sort_values('timestamp'),
                               on='timestamp', direction='backward')
        merged.index = df_1h.index
        return merged[cols].fillna(0.0)
    except Exception:
        return empty


def compute_fibonacci_levels(df: pd.DataFrame, lookback: int = 100) -> pd.DataFrame:
    """Auto Fibonacci from recent swing high/low (shift-1 to avoid lookahead)."""
    sh = df['high'].shift(1).rolling(lookback).max()
    sl = df['low'].shift(1).rolling(lookback).min()
    rng = sh - sl + 1e-9
    c = df['close']
    return pd.DataFrame({
        'fib_dist_236': (c - (sh - 0.236 * rng)) / (c + 1e-9),
        'fib_dist_382': (c - (sh - 0.382 * rng)) / (c + 1e-9),
        'fib_dist_500': (c - (sh - 0.500 * rng)) / (c + 1e-9),
        'fib_dist_618': (c - (sh - 0.618 * rng)) / (c + 1e-9),
        'fib_range_pct': rng / (c + 1e-9),
    }, index=df.index)


def compute_bos_choch(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Break of Structure (BOS) and Change of Character (CHoCH).

    BOS up/down are ONE-BAR events (1 only on the candle where price crosses
    the lookback extreme, 0 every other bar). bos_state is the continuous
    version (+1 = above high, -1 = below low, 0 = inside range).
    CHoCH fires when a BOS occurs against the prevailing structural trend,
    signalling a potential reversal rather than trend continuation.
    """
    close = df['close']
    sh = df['high'].shift(1).rolling(lookback).max()   # previous lookback high
    sl = df['low'].shift(1).rolling(lookback).min()    # previous lookback low

    above = close > sh
    below = close < sl

    # Event: fired only on the bar where price first crosses the level.
    # shift(fill_value=False) keeps bool dtype — plain .shift().fillna(False)
    # yields an OBJECT series of Python bools, and ~ on those returns -2/-1
    # (bitwise int inversion), silently breaking the logic (and warning on 3.12).
    bos_up   = (above & ~above.shift(1, fill_value=False)).astype(int)
    bos_down = (below & ~below.shift(1, fill_value=False)).astype(int)

    # Continuous state (+1 / 0 / -1)
    bos_state = above.astype(int) - below.astype(int)

    # Structural bias: higher-high / lower-low slope over lookback
    structure_bias = pd.Series(np.sign(close - close.shift(lookback)), index=df.index).fillna(0)

    # CHoCH: BOS that contradicts the prevailing structural bias
    prior_bias = structure_bias.shift(1).fillna(0)
    choch_bull = (bos_up   == 1) & (prior_bias < 0)   # break up after downtrend
    choch_bear = (bos_down == 1) & (prior_bias > 0)   # break down after uptrend

    return pd.DataFrame({
        'bos_up':          bos_up,
        'bos_down':        bos_down,
        'bos_state':       bos_state,
        'choch_bull':      choch_bull.astype(int),
        'choch_bear':      choch_bear.astype(int),
        'structure_bias':  structure_bias,
    }, index=df.index)


def compute_order_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """Approximate order blocks: large-range candles before significant moves."""
    rng = df['high'] - df['low']
    rng_ma = rng.rolling(20, min_periods=1).mean()
    large_bull = (rng > 1.5 * rng_ma) & (df['close'] > df['open'])
    large_bear = (rng > 1.5 * rng_ma) & (df['close'] < df['open'])
    ob_bull = df['open'].where(large_bull).shift(1).ffill().fillna(df['close'])
    ob_bear = df['open'].where(large_bear).shift(1).ffill().fillna(df['close'])
    c = df['close']
    return pd.DataFrame({
        'dist_bull_ob': (c - ob_bull) / (c + 1e-9),
        'dist_bear_ob': (ob_bear - c) / (c + 1e-9),
    }, index=df.index)


# ── Statistical / Quant Indicators ───────────────────────────────
def compute_standard_error_bands(series: pd.Series, period: int = 21) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Standard Error Bands — regression midline ± 2 × standard error."""
    def _lr_end(arr):
        x = np.arange(len(arr))
        m, b = np.polyfit(x, arr, 1)
        return m * (len(arr) - 1) + b

    def _se(arr):
        x = np.arange(len(arr))
        m, b = np.polyfit(x, arr, 1)
        return np.std(arr - (m * x + b), ddof=1) + 1e-9

    lr = series.rolling(period).apply(_lr_end, raw=True)
    se = series.rolling(period).apply(_se, raw=True)
    return lr + 2 * se, lr, lr - 2 * se


def compute_quantile_bands(series: pd.Series, period: int = 100,
                           q_lo: float = 0.1, q_hi: float = 0.9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Rolling quantile bands."""
    return (series.rolling(period).quantile(q_hi),
            series.rolling(period).quantile(0.5),
            series.rolling(period).quantile(q_lo))


def compute_hurst_exponent(series: pd.Series, period: int = 100) -> pd.Series:
    """Hurst Exponent via variance-ratio (fast). >0.5=trending, <0.5=mean-reverting."""
    rets = series.pct_change().fillna(0)

    def _hurst(arr):
        if len(arr) < 10:
            return 0.5
        try:
            v1 = np.var(np.diff(arr)) + 1e-12
            v4 = np.var(arr[4:] - arr[:-4]) / 4 + 1e-12
            return float(np.clip(0.5 * np.log(v4 / v1) / np.log(4) + 0.5, 0.0, 1.0))
        except Exception:
            return 0.5

    return rets.rolling(period, min_periods=20).apply(_hurst, raw=True).fillna(0.5)


def compute_entropy(series: pd.Series, period: int = 20, bins: int = 10) -> pd.Series:
    """Shannon Entropy of returns — high = random/unpredictable, low = structured."""
    def _ent(arr):
        try:
            counts, _ = np.histogram(arr, bins=bins)
            p = counts / (counts.sum() + 1e-9)
            p = p[p > 0]
            return float(-np.sum(p * np.log(p)))
        except Exception:
            return float(np.log(bins))

    return series.pct_change().fillna(0).rolling(period, min_periods=10).apply(_ent, raw=True)


# ── Squeeze Momentum (LazyBear) ───────────────────────────────────
def compute_squeeze_momentum(
    df: pd.DataFrame,
    bb_len: int = 20, bb_mult: float = 2.0,
    kc_len: int = 20, kc_mult: float = 1.5,
) -> pd.DataFrame:
    """
    Identifies consolidation (BB inside KC) and momentum direction.
    squeeze_on  = 1 when BB is inside KC (spring loading before move)
    squeeze_off = 1 when BB just broke wider than KC (expansion starting)
    sqz_momentum = normalised delta-momentum (positive = bullish thrust)
    """
    c, h, l = df['close'], df['high'], df['low']
    tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)

    bb_mid   = c.rolling(bb_len).mean()
    bb_std   = c.rolling(bb_len).std()
    bb_upper = bb_mid + bb_mult * bb_std
    bb_lower = bb_mid - bb_mult * bb_std

    kc_mid   = c.ewm(span=kc_len, adjust=False).mean()
    kc_atr   = tr.rolling(kc_len).mean()
    kc_upper = kc_mid + kc_mult * kc_atr
    kc_lower = kc_mid - kc_mult * kc_atr

    sqz_on  = ((bb_upper < kc_upper) & (bb_lower > kc_lower)).astype(float)
    sqz_off = ((bb_upper > kc_upper) & (bb_lower < kc_lower)).astype(float)

    # Delta value: close vs midpoint of recent range
    highest_h = h.rolling(kc_len).max()
    lowest_l  = l.rolling(kc_len).min()
    delta = c - ((highest_h + lowest_l) / 2 + kc_mid) / 2
    momentum_raw = delta.ewm(span=kc_len, adjust=False).mean()
    momentum_norm = momentum_raw / (kc_atr + 1e-9)   # normalise by KC ATR

    return pd.DataFrame({
        'sqz_on':       sqz_on,
        'sqz_off':      sqz_off,
        'sqz_momentum': momentum_norm.clip(-3, 3),
    }, index=df.index)


# ── Elder Ray Index ────────────────────────────────────────────────
def compute_elder_ray(df: pd.DataFrame, period: int = 13) -> pd.DataFrame:
    """
    Elder Ray: Bull Power = High - EMA(close), Bear Power = Low - EMA(close).
    Positive bull power + rising EMA → healthy uptrend.
    Negative bear power + falling EMA → healthy downtrend.
    Both are normalised by ATR for cross-token comparability.
    """
    ema = df['close'].ewm(span=period, adjust=False).mean()
    atr = compute_atr(df, 14)
    bull = (df['high'] - ema) / (atr + 1e-9)
    bear = (df['low']  - ema) / (atr + 1e-9)
    return pd.DataFrame({
        'elder_bull': bull.clip(-5, 5),
        'elder_bear': bear.clip(-5, 5),
        'elder_bull_bear': (bull + bear).clip(-5, 5),  # net pressure
    }, index=df.index)


# ── Ichimoku Extended signals ──────────────────────────────────────
def compute_ichimoku_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Additional Ichimoku signal columns derived from the cloud components.
    Requires ichimoku_tenkan / kijun / senkou_a / senkou_b already in df.
    All derived in a leakage-safe, backward-only manner.
    """
    c = df['close']
    tenkan = df.get('ichimoku_tenkan', pd.Series(np.nan, index=df.index))
    kijun  = df.get('ichimoku_kijun',  pd.Series(np.nan, index=df.index))
    sa     = df.get('ichimoku_senkou_a', pd.Series(np.nan, index=df.index))
    sb     = df.get('ichimoku_senkou_b', pd.Series(np.nan, index=df.index))

    cloud_top = pd.concat([sa, sb], axis=1).max(axis=1)
    cloud_bot = pd.concat([sa, sb], axis=1).min(axis=1)

    above_cloud = np.where(c > cloud_top, 1.0, np.where(c < cloud_bot, -1.0, 0.0))
    cloud_bull  = np.where(sa > sb, 1.0, -1.0)
    tk_diff     = tenkan - kijun
    tk_cross    = np.where(
        (tk_diff > 0) & (tk_diff.shift(1) <= 0), 1.0,
        np.where((tk_diff < 0) & (tk_diff.shift(1) >= 0), -1.0, 0.0)
    )
    dist_tenkan = ((c - tenkan) / (c + 1e-9)).clip(-0.2, 0.2)
    dist_kijun  = ((c - kijun)  / (c + 1e-9)).clip(-0.2, 0.2)
    dist_cloud_top = ((c - cloud_top) / (c + 1e-9)).clip(-0.3, 0.3)

    return pd.DataFrame({
        'ichi_above_cloud':    pd.Series(above_cloud, index=df.index),
        'ichi_cloud_bull':     pd.Series(cloud_bull,  index=df.index),
        'ichi_tk_cross':       pd.Series(tk_cross,    index=df.index),
        'ichi_dist_tenkan':    dist_tenkan,
        'ichi_dist_kijun':     dist_kijun,
        'ichi_dist_cloud_top': dist_cloud_top,
    }, index=df.index)


# ── Crypto Funding / OI Micro-structure ───────────────────────────
def compute_funding_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derived features from perpetual funding rate.
    Extreme positive funding = over-leveraged longs → reversal risk.
    Extreme negative funding = over-leveraged shorts → squeeze risk.
    """
    if 'funding_rate' not in df.columns:
        return pd.DataFrame(index=df.index)
    fr = df['funding_rate'].fillna(0)
    return pd.DataFrame({
        'funding_slope_3':      fr.diff(3).clip(-0.002, 0.002),   # momentum
        'funding_slope_8':      fr.diff(8).clip(-0.003, 0.003),
        'funding_extreme_long': (fr >  0.001).astype(float),       # >0.1% → crowded long
        'funding_extreme_short':(fr < -0.001).astype(float),       # <-0.1% → crowded short
        'funding_neutral':      (fr.abs() < 0.00005).astype(float),# near-zero
        'funding_cum_8':        fr.rolling(8).sum().clip(-0.01, 0.01),   # 8-bar cumulative
    }, index=df.index)


def compute_oi_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Open Interest trend vs price direction — crypto-specific regime signal.
    OI rising + price rising   = healthy trend (smart money supporting move)
    OI rising + price falling  = bearish (trapped longs, potential cascade)
    OI falling + price moving  = position unwinding (exhaustion)
    """
    if 'open_interest' not in df.columns:
        return pd.DataFrame(index=df.index)
    oi = df['open_interest'].ffill().fillna(0)
    px = df['close']

    oi_chg_8  = oi.pct_change(8).clip(-0.5, 0.5).fillna(0)
    px_chg_8  = px.pct_change(8).clip(-0.3, 0.3).fillna(0)
    oi_chg_24 = oi.pct_change(24).clip(-0.8, 0.8).fillna(0)

    # Agreement: +1 both rising, -1 diverging (OI up, price down = bearish), 0 neutral
    oi_px_agreement = pd.Series(
        np.where((oi_chg_8 > 0.02) & (px_chg_8 > 0),  1.0,   # bullish confluence
        np.where((oi_chg_8 > 0.02) & (px_chg_8 < 0), -1.0,   # bearish divergence
        np.where((oi_chg_8 < -0.02), -0.5 * np.sign(px_chg_8), 0.0))),
        index=df.index,
    )
    return pd.DataFrame({
        'oi_chg_8h':       oi_chg_8,
        'oi_chg_24h':      oi_chg_24,
        'oi_px_agreement': oi_px_agreement,
    }, index=df.index)


# ── Volume Pressure Composite ──────────────────────────────────────
def compute_volume_pressure(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Normalised buying vs selling pressure.
    - candle_pressure: close position within bar × volume (scaled by avg volume)
    - rolling_buy_ratio: fraction of bars where close > open in last `period` bars
    - vol_price_trend: OBV-like but normalised to avoid unit issues
    """
    bar_range  = (df['high'] - df['low']).replace(0, np.nan)
    close_pos  = ((df['close'] - df['low']) / bar_range).fillna(0.5)  # 0=low, 1=high
    avg_vol    = df['volume'].rolling(period, min_periods=5).mean().replace(0, np.nan)
    vol_norm   = (df['volume'] / avg_vol).fillna(1.0)

    # Net pressure: positive when closing near high on above-avg volume
    candle_pressure = (close_pos * 2 - 1) * vol_norm   # [-vol_norm, +vol_norm]
    cp_smooth = candle_pressure.rolling(period, min_periods=5).mean().clip(-3, 3)

    # Rolling win rate: how often the last N candles closed higher
    bull_candle = (df['close'] > df['open']).astype(float)
    rolling_buy_ratio = bull_candle.rolling(period, min_periods=5).mean()

    return pd.DataFrame({
        'candle_pressure': cp_smooth,
        'rolling_buy_ratio': rolling_buy_ratio,
    }, index=df.index)


# ── Multi-period Trend Alignment ──────────────────────────────────
def compute_trend_alignment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agreement score across multiple EMA timeframes.
    When short/medium/long EMAs all agree on direction → stronger trend signal.
    Returns a [-1, +1] composite where +1 = all EMAs bullishly stacked.
    """
    c = df['close']
    votes = []
    for period in (9, 21, 50, 100, 200):
        ema = c.ewm(span=period, adjust=False).mean()
        votes.append(np.sign(c - ema))
    stacked = pd.concat(votes, axis=1)
    alignment = stacked.mean(axis=1)   # in (-1, +1)
    # Trend quality: how many agree (0=split, 1=all same direction)
    quality = stacked.abs().mean(axis=1)
    return pd.DataFrame({
        'ema_alignment':         alignment.clip(-1, 1),
        'ema_alignment_quality': quality,
    }, index=df.index)


# ── Anchored VWAP ─────────────────────────────────────────────────
def compute_anchored_vwap(df: pd.DataFrame, lookback: int = 100) -> pd.DataFrame:
    """Anchored VWAP at three lookback windows (50 / 100 / 200 bars).
    Each window simulates anchoring to the swing high/low that occurred
    lookback bars ago.  All computation is backward-only — no lookahead."""
    tp  = (df['high'] + df['low'] + df['close']) / 3
    tv  = tp * df['volume']
    vol = df['volume']
    c   = df['close']
    avwap = {}
    for lb in (lookback // 2, lookback, min(200, lookback * 2)):
        key = f'avwap_{lb}'
        avwap[key] = tv.rolling(lb, min_periods=1).sum() / vol.rolling(lb, min_periods=1).sum()
        avwap[f'dist_{key}'] = (c / (avwap[key] + 1e-9)) - 1
    return pd.DataFrame(avwap, index=df.index)


# ── Weekly Timeframe Indicators ───────────────────────────────────
def compute_weekly_features(df_1h: pd.DataFrame,
                             df_1d: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """MACD, RSI, 200 EMA/SMA, BOS/CHoCH and trend direction on weekly candles,
    resampled from daily data and mapped backward to the hourly index.
    All joins are backward (direction='backward') — no lookahead."""
    _cols = [
        'weekly_macd', 'weekly_macd_signal', 'weekly_macd_hist',
        'weekly_rsi', 'weekly_ema200', 'weekly_sma200',
        'dist_weekly_ema200', 'dist_weekly_sma200',
        'weekly_bos_up', 'weekly_bos_down', 'weekly_structure_bias',
        'weekly_trend',
    ]
    empty = pd.DataFrame(0.0, index=df_1h.index, columns=_cols)
    if df_1d is None or df_1d.empty:
        return empty
    try:
        d = df_1d.copy()
        d['timestamp'] = pd.to_datetime(d['timestamp'])
        # Strip timezone so merge_asof never throws TypeError when df_1d and
        # df_1h have different tz-awareness (e.g. one UTC-aware, one naive).
        if d['timestamp'].dt.tz is not None:
            d['timestamp'] = d['timestamp'].dt.tz_convert(None)

        w = (d.sort_values('timestamp')
              .set_index('timestamp')
              .resample('W')
              .agg({'open': 'first', 'high': 'max', 'low': 'min',
                    'close': 'last', 'volume': 'sum'})
              .dropna(subset=['close'])
              .reset_index())
        if len(w) < 14:
            return empty

        c      = w['close']
        ema12  = c.ewm(span=12, adjust=False).mean()
        ema26  = c.ewm(span=26, adjust=False).mean()
        w_macd = ema12 - ema26
        w_sig  = w_macd.ewm(span=9, adjust=False).mean()
        # fillna(50) before storing: RSI is NaN for the first period-1 bars;
        # 0 would signal "maximally oversold" to the model — 50 is neutral.
        w_rsi  = compute_rsi(c, 14).fillna(50.0)
        w_e200 = c.ewm(span=200, adjust=False).mean()
        w_s200 = c.rolling(200, min_periods=1).mean()
        bos    = compute_bos_choch(w, lookback=min(20, len(w) // 4))
        w_trend = np.sign(c - w_s200)  # pandas aligns by index; no .values dance

        # Binance 1d candle timestamps are open times (Sunday 00:00 UTC).
        # resample('W') labels the weekly bar at Sunday 00:00, but the bar's
        # close='last' is Sunday's daily close which completes at 23:59 UTC.
        # Shift +1 day so the completed weekly bar is only available from
        # Monday 00:00 onward — eliminates the full-day Sunday lookahead.
        available_from = w['timestamp'] + pd.Timedelta(days=1)

        wf = pd.DataFrame({
            'timestamp':             available_from.to_numpy(),
            'weekly_macd':           w_macd.to_numpy(),
            'weekly_macd_signal':    w_sig.to_numpy(),
            'weekly_macd_hist':      (w_macd - w_sig).to_numpy(),
            'weekly_rsi':            w_rsi.to_numpy(),
            'weekly_ema200':         w_e200.to_numpy(),
            'weekly_sma200':         w_s200.to_numpy(),
            'dist_weekly_ema200':    (c / (w_e200 + 1e-9) - 1).to_numpy(),
            'dist_weekly_sma200':    (c / (w_s200 + 1e-9) - 1).to_numpy(),
            'weekly_bos_up':         bos['bos_up'].to_numpy(),
            'weekly_bos_down':       bos['bos_down'].to_numpy(),
            'weekly_structure_bias': bos['structure_bias'].to_numpy(),
            'weekly_trend':          w_trend.to_numpy(),
        })

        h = df_1h[['timestamp']].copy()
        h['timestamp'] = pd.to_datetime(h['timestamp'])
        if h['timestamp'].dt.tz is not None:
            h['timestamp'] = h['timestamp'].dt.tz_convert(None)

        h_sorted = h.sort_values('timestamp')
        merged = pd.merge_asof(h_sorted,
                               wf.sort_values('timestamp'),
                               on='timestamp', direction='backward')
        merged.index = h_sorted.index
        return merged[_cols].reindex(df_1h.index).fillna(0.0)
    except Exception as e:
        logger.warning(f"compute_weekly_features failed: {e}")
        return empty


# ── Fear & Greed Index ────────────────────────────────────────────
def add_fear_greed_features(df: pd.DataFrame,
                             fg_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Merge daily Fear & Greed Index onto the hourly frame.
    fg_df must have columns ['timestamp', 'fear_greed_value'].
    Falls back to neutral (50) when data is unavailable."""
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if fg_df is None or fg_df.empty:
        df['fear_greed_value']  = 50.0
        df['fear_greed_signal'] = 0.0
        return df
    try:
        fg = fg_df[['timestamp', 'fear_greed_value']].copy()
        fg['timestamp'] = pd.to_datetime(fg['timestamp'])
        merged = pd.merge_asof(df.sort_values('timestamp'),
                               fg.sort_values('timestamp'),
                               on='timestamp', direction='backward')
        merged['fear_greed_value']  = merged['fear_greed_value'].fillna(50.0)
        merged['fear_greed_signal'] = (merged['fear_greed_value'] - 50.0) / 50.0
        df['fear_greed_value']  = merged['fear_greed_value'].values
        df['fear_greed_signal'] = merged['fear_greed_signal'].values
    except Exception:
        df['fear_greed_value']  = 50.0
        df['fear_greed_signal'] = 0.0
    return df


# ── Category Confluence (pre-aggregated voter groups) ─────────────
def compute_category_confluence(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate all indicator categories into six [-1, +1] consensus scores
    plus a weighted total.  Each score represents the net bullish (+1) /
    bearish (-1) vote from that indicator family.  XGBoost sees these as
    high-level 'voter group' features alongside the raw indicators.

    Design rules (lookahead-safe):
      - Only uses columns that are already in df (guard with 'if col in df').
      - All position indicators are mapped [0,1]→[-1,+1] before averaging.
      - BOS/CHoCH events are one-bar signals so they decay naturally via rolling.
    """
    get = df.get  # shorthand

    def _centered(*pairs) -> Optional[pd.Series]:
        """Mean of sign(col - center) for (col, center) pairs."""
        parts = [np.sign((df[c] - v).to_numpy(dtype=float))
                 for c, v in pairs if c in df.columns]
        if not parts:
            return None
        return pd.Series(np.nanmean(np.stack(parts, axis=1), axis=1), index=df.index)

    def _pos(*cols: str) -> Optional[pd.Series]:
        """Map [0,1] position columns to [-1,+1] and average."""
        # Clip each part before averaging so outliers (e.g. bb_pct_b outside [0,1])
        # don't skew the mean; the per-part clamp makes the final average also in [-1,+1].
        parts = [np.clip(df[c].to_numpy(dtype=float) * 2 - 1, -1.0, 1.0)
                 for c in cols if c in df.columns]
        if not parts:
            return None
        return pd.Series(np.nanmean(np.stack(parts, axis=1), axis=1), index=df.index)

    scores: Dict[str, pd.Series] = {}

    # ── MOMENTUM ─────────────────────────────────────────────────────
    mom = _centered(
        ('rsi_7', 50), ('rsi_14', 50), ('rsi_21', 50),
        ('stoch_k', 50), ('williams_r', -50),
        ('macd_hist', 0), ('cci_20', 0), ('tsi', 0),
        ('cmo_14', 0), ('awesome_osc', 0), ('bop', 0),
        ('kst', 0), ('trix_15', 0), ('roc_14', 0),
        ('dpo_20', 0), ('ppo', 0), ('schaff_tc', 50),
        ('fisher', 0), ('rvi_osc', 0),
    )
    if mom is not None:
        scores['momentum_confluence'] = mom.clip(-1, 1)

    # ── TREND ────────────────────────────────────────────────────────
    trend = _centered(
        ('dist_ema_50', 0), ('dist_ema_200', 0),
        ('dist_hma20', 0), ('dist_kama', 0),
        ('dist_rolling_vwap', 0), ('dist_vwap', 0),
        ('supertrend_dist', 0), ('sar_dist', 0),
        ('linreg_slope_14', 0), ('structure_bias', 0),
        ('macro_trend_1d', 0), ('macro_trend_1w', 0),
    )
    if trend is not None:
        scores['trend_confluence'] = trend.clip(-1, 1)

    # ── VOLUME ───────────────────────────────────────────────────────
    vol = _centered(
        ('volume_delta', 0), ('volume_delta_14', 0),
        ('cmf_20', 0), ('mfi_14', 50),
        ('eom_14', 0), ('kvo', 0), ('rvi_osc', 0),
    )
    if vol is not None:
        scores['volume_confluence'] = vol.clip(-1, 1)

    # ── BANDS / PRICE POSITION ────────────────────────────────────────
    bands = _pos(
        'bb_pct_b', 'atr_band_position', 'donchian_position',
        'close_position', 'quantile_position', 'se_position',
        'starc_position', 'gaussian_position',
    )
    if bands is not None:
        scores['bands_confluence'] = bands

    # ── SMART MONEY / MARKET STRUCTURE ───────────────────────────────
    smc_parts: List[pd.Series] = []
    for bull_col in ('bos_up', 'choch_bull', 'is_at_support'):
        if bull_col in df.columns:
            weight = 1.5 if 'choch' in bull_col else 1.0
            smc_parts.append(df[bull_col].astype(float) * weight)
    for bear_col in ('bos_down', 'choch_bear', 'is_at_resistance'):
        if bear_col in df.columns:
            weight = 1.5 if 'choch' in bear_col else 1.0
            smc_parts.append(-df[bear_col].astype(float) * weight)
    rps = _pos('range_position_score')
    if rps is not None:
        smc_parts.append(rps)
    if 'structure_bias' in df.columns:
        smc_parts.append(df['structure_bias'].astype(float))
    if smc_parts:
        scores['smart_money_confluence'] = (
            pd.concat(smc_parts, axis=1).mean(axis=1).clip(-1, 1)
        )

    # ── CANDLESTICK PATTERNS ─────────────────────────────────────────
    cdl_parts: List[pd.Series] = []
    for bull_pat in ('CDL_HAMMER', 'CDL_BULL_ENGULFING', 'CDL_MORNINGSTAR'):
        if bull_pat in df.columns:
            cdl_parts.append(df[bull_pat].astype(float))
    for bear_pat in ('CDL_SHOOTINGSTAR', 'CDL_BEAR_ENGULFING', 'CDL_EVENINGSTAR'):
        if bear_pat in df.columns:
            cdl_parts.append(-df[bear_pat].astype(float))
    if 'bar_direction' in df.columns:
        cdl_parts.append(df['bar_direction'].astype(float) * 0.3)  # bar direction has lower weight
    if cdl_parts:
        scores['candle_confluence'] = (
            pd.concat(cdl_parts, axis=1).mean(axis=1).clip(-1, 1)
        )

    # ── TOTAL (weighted) ─────────────────────────────────────────────
    # Trend and smart money get higher weight; candles lower weight.
    w_map = {
        'trend_confluence':       2.0,
        'momentum_confluence':    1.5,
        'volume_confluence':      1.5,
        'smart_money_confluence': 1.5,
        'bands_confluence':       1.0,
        'candle_confluence':      0.5,
    }
    if scores:
        cat_df = pd.DataFrame(scores, index=df.index)
        weights = np.array([w_map.get(c, 1.0) for c in cat_df.columns])
        scores['total_confluence'] = (
            (cat_df * weights).sum(axis=1) / weights.sum()
        ).clip(-1, 1)

    if not scores:
        return pd.DataFrame(index=df.index)
    return pd.DataFrame(scores, index=df.index).fillna(0)


# ------------------------------------------------------------------
# Support & Resistance (lookahead-bias free) and Macro Regime
# ------------------------------------------------------------------
def add_support_resistance_features(df: pd.DataFrame, window: int = 24) -> pd.DataFrame:
    """
    Add lookahead-bias-free support & resistance features computed from
    historical highs/lows shifted by one bar and rolled over `window`.

    - rolling_resistance: historical rolling max of high.shift(1)
    - rolling_support: historical rolling min of low.shift(1)
    - pct_dist_to_resistance: (rolling_resistance - close) / close
    - pct_dist_to_support: (close - rolling_support) / close
    - range_position_score: (close - support) / (resistance - support)
    """
    df = df.copy()
    highs_shift = df['high'].shift(1)
    lows_shift = df['low'].shift(1)

    rolling_resistance = highs_shift.rolling(window=window, min_periods=1).max()
    rolling_support = lows_shift.rolling(window=window, min_periods=1).min()

    pct_dist_to_resistance = (rolling_resistance - df['close']) / (df['close'] + 1e-9)
    pct_dist_to_support = (df['close'] - rolling_support) / (df['close'] + 1e-9)
    denom = (rolling_resistance - rolling_support).replace(0, np.nan)
    range_position_score = ((df['close'] - rolling_support) / (denom + 1e-9)).clip(0.0, 1.0).fillna(0.5)

    output = pd.DataFrame({
        'rolling_resistance': rolling_resistance,
        'rolling_support': rolling_support,
        'pct_dist_to_resistance': pct_dist_to_resistance.fillna(0.0),
        'pct_dist_to_support': pct_dist_to_support.fillna(0.0),
        'range_position_score': range_position_score,
    }, index=df.index)
    return output

def compute_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorised candlestick pattern library — 1-, 2- and 3-candle patterns.

    All patterns use only the current bar and prior bars (via .shift()), so
    there is no lookahead bias and no repainting.  The original seven columns
    (CDL_DOJI … CDL_EVENINGSTAR) keep their exact definitions for model
    compatibility; everything below them is additive.

    Reversal patterns are context-gated where the classic definition demands
    it: a hammer only counts toward the bullish reversal score when it prints
    after a short-term downtrend (and mirrored for bearish).  The trend proxy
    uses only PRIOR closes (shift(1) vs its own rolling mean).

    Aggregate outputs consumed by the live engine's reversal-confirmation gate:
      cdl_bull_reversal_score — weighted sum of bullish reversal patterns
      cdl_bear_reversal_score — weighted sum of bearish reversal patterns
    Weights reflect pattern reliability: 3-candle (1.5) > 2-candle (1.0)
    > harami (0.75) > 1-candle (0.5).
    """
    o, h, l, c = df['open'], df['high'], df['low'], df['close']
    body = abs(c - o)
    range_hl = h - l
    upper_shadow = h - np.maximum(o, c)
    lower_shadow = np.minimum(o, c) - l
    body_frac = body / (range_hl + 1e-9)
    avg_body = body.rolling(14, min_periods=5).mean()

    is_green = c > o
    is_red = o > c
    prev_green = is_green.shift(1)
    prev_red = is_red.shift(1)

    # Short-term trend context from PRIOR bars only (no lookahead):
    # the close going INTO this bar vs its own 5-bar mean.
    prior_close = c.shift(1)
    prior_ma5 = prior_close.rolling(5, min_periods=3).mean()
    downtrend = (prior_close < prior_ma5).fillna(False)
    uptrend = (prior_close > prior_ma5).fillna(False)

    # ── 1-candle patterns ─────────────────────────────────────────────
    is_doji = body_frac < 0.1
    is_hammer = (lower_shadow > 2 * body) & (upper_shadow < 0.2 * body)
    is_shootingstar = (upper_shadow > 2 * body) & (lower_shadow < 0.2 * body)

    dragonfly_doji = is_doji & (lower_shadow > 0.6 * range_hl) & (upper_shadow < 0.1 * range_hl)
    gravestone_doji = is_doji & (upper_shadow > 0.6 * range_hl) & (lower_shadow < 0.1 * range_hl)

    # Same shapes with the trend context that gives them reversal meaning
    inv_hammer = is_shootingstar & downtrend      # bullish after decline
    hanging_man = is_hammer & uptrend             # bearish after rally

    bull_marubozu = is_green & (body_frac > 0.9)  # full-body conviction bars
    bear_marubozu = is_red & (body_frac > 0.9)

    spinning_top = (body_frac < 0.3) & (upper_shadow > body) & (lower_shadow > body) & ~is_doji

    # ── 2-candle patterns ─────────────────────────────────────────────
    bullish_engulfing = prev_red & is_green & (c > o.shift(1)) & (o < c.shift(1))
    bearish_engulfing = prev_green & is_red & (c < o.shift(1)) & (o > c.shift(1))

    prev_body_mid = (o.shift(1) + c.shift(1)) / 2.0
    # Crypto trades 24/7 so the classic "gap open" never occurs; the standard
    # adaptation is open at-or-below the prior close (piercing) / at-or-above
    # (dark cloud).
    piercing = (
        prev_red & is_green
        & (o <= c.shift(1)) & (c > prev_body_mid) & (c < o.shift(1))
    )
    dark_cloud = (
        prev_green & is_red
        & (o >= c.shift(1)) & (c < prev_body_mid) & (c > o.shift(1))
    )

    prev_large_body = body.shift(1) > avg_body.shift(1)
    body_inside_prev = (
        (np.maximum(o, c) < np.maximum(o.shift(1), c.shift(1)))
        & (np.minimum(o, c) > np.minimum(o.shift(1), c.shift(1)))
    )
    bull_harami = prev_red & prev_large_body & body_inside_prev & is_green
    bear_harami = prev_green & prev_large_body & body_inside_prev & is_red

    lows_match = (l - l.shift(1)).abs() <= (0.001 * c)
    highs_match = (h - h.shift(1)).abs() <= (0.001 * c)
    tweezer_bottom = lows_match & prev_red & is_green & downtrend
    tweezer_top = highs_match & prev_green & is_red & uptrend

    # ── 3-candle patterns ─────────────────────────────────────────────
    prev2_red = is_red.shift(2)
    prev2_green = is_green.shift(2)
    prev1_small_body = (body.shift(1) / (range_hl.shift(1) + 1e-9)) < 0.3

    morning_star = prev2_red & prev1_small_body & is_green & (c > o.shift(2) - (body.shift(2) * 0.5))
    evening_star = prev2_green & prev1_small_body & is_red & (c < o.shift(2) + (body.shift(2) * 0.5))

    solid_body = body_frac > 0.5
    opens_inside_prev = (o > np.minimum(o.shift(1), c.shift(1))) & (o < np.maximum(o.shift(1), c.shift(1)))
    three_white_soldiers = (
        is_green & is_green.shift(1) & is_green.shift(2)
        & solid_body & solid_body.shift(1) & solid_body.shift(2)
        & (c > c.shift(1)) & (c.shift(1) > c.shift(2))
        & opens_inside_prev & opens_inside_prev.shift(1)
    )
    three_black_crows = (
        is_red & is_red.shift(1) & is_red.shift(2)
        & solid_body & solid_body.shift(1) & solid_body.shift(2)
        & (c < c.shift(1)) & (c.shift(1) < c.shift(2))
        & opens_inside_prev & opens_inside_prev.shift(1)
    )

    # Harami / engulfing followed by a confirming close beyond the first bar
    three_inside_up = bull_harami.shift(1, fill_value=False) & is_green & (c > o.shift(2))
    three_inside_down = bear_harami.shift(1, fill_value=False) & is_red & (c < o.shift(2))
    three_outside_up = bullish_engulfing.shift(1, fill_value=False) & is_green & (c > c.shift(1))
    three_outside_down = bearish_engulfing.shift(1, fill_value=False) & is_red & (c < c.shift(1))

    # ── Weighted reversal aggregates ──────────────────────────────────
    # Only patterns with genuine reversal semantics contribute; marubozu is
    # continuation and doji/spinning-top are direction-less, so all three are
    # excluded.  1- and 2-candle patterns require the matching prior trend;
    # 3-candle patterns embed the prior move in their own structure.
    def _w(cond: pd.Series, weight: float) -> pd.Series:
        return cond.fillna(False).astype(float) * weight

    bull_reversal_score = (
        _w(morning_star, 1.5) + _w(three_white_soldiers, 1.5)
        + _w(three_inside_up, 1.5) + _w(three_outside_up, 1.5)
        + _w(bullish_engulfing & downtrend, 1.0) + _w(piercing & downtrend, 1.0)
        + _w(tweezer_bottom, 1.0)
        + _w(bull_harami & downtrend, 0.75)
        + _w(is_hammer & downtrend, 0.5) + _w(inv_hammer, 0.5)
        + _w(dragonfly_doji & downtrend, 0.5)
    )
    bear_reversal_score = (
        _w(evening_star, 1.5) + _w(three_black_crows, 1.5)
        + _w(three_inside_down, 1.5) + _w(three_outside_down, 1.5)
        + _w(bearish_engulfing & uptrend, 1.0) + _w(dark_cloud & uptrend, 1.0)
        + _w(tweezer_top, 1.0)
        + _w(bear_harami & uptrend, 0.75)
        + _w(is_shootingstar & uptrend, 0.5) + _w(hanging_man, 0.5)
        + _w(gravestone_doji & uptrend, 0.5)
    )

    return pd.DataFrame({
        # Original seven — definitions unchanged (model compatibility)
        'CDL_DOJI': is_doji.astype(int),
        'CDL_HAMMER': is_hammer.astype(int),
        'CDL_SHOOTINGSTAR': is_shootingstar.astype(int),
        'CDL_BULL_ENGULFING': bullish_engulfing.astype(int),
        'CDL_BEAR_ENGULFING': bearish_engulfing.astype(int),
        'CDL_MORNINGSTAR': morning_star.astype(int),
        'CDL_EVENINGSTAR': evening_star.astype(int),
        # 1-candle additions
        'CDL_DRAGONFLY_DOJI': dragonfly_doji.astype(int),
        'CDL_GRAVESTONE_DOJI': gravestone_doji.astype(int),
        'CDL_INV_HAMMER': inv_hammer.astype(int),
        'CDL_HANGING_MAN': hanging_man.astype(int),
        'CDL_BULL_MARUBOZU': bull_marubozu.astype(int),
        'CDL_BEAR_MARUBOZU': bear_marubozu.astype(int),
        'CDL_SPINNING_TOP': spinning_top.astype(int),
        # 2-candle additions
        'CDL_PIERCING': piercing.astype(int),
        'CDL_DARK_CLOUD': dark_cloud.astype(int),
        'CDL_BULL_HARAMI': bull_harami.astype(int),
        'CDL_BEAR_HARAMI': bear_harami.astype(int),
        'CDL_TWEEZER_BOTTOM': tweezer_bottom.astype(int),
        'CDL_TWEEZER_TOP': tweezer_top.astype(int),
        # 3-candle additions
        'CDL_3_WHITE_SOLDIERS': three_white_soldiers.astype(int),
        'CDL_3_BLACK_CROWS': three_black_crows.astype(int),
        'CDL_3_INSIDE_UP': three_inside_up.astype(int),
        'CDL_3_INSIDE_DOWN': three_inside_down.astype(int),
        'CDL_3_OUTSIDE_UP': three_outside_up.astype(int),
        'CDL_3_OUTSIDE_DOWN': three_outside_down.astype(int),
        # Weighted reversal aggregates (live-engine confirmation gate)
        'cdl_bull_reversal_score': bull_reversal_score,
        'cdl_bear_reversal_score': bear_reversal_score,
    }, index=df.index)


def add_macro_regime_features(df: pd.DataFrame, df_1d: Optional[pd.DataFrame], df_1w: Optional[pd.DataFrame], macro_state: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Map 1d and 1w macro directional anchors down to the base dataframe `df`.

    Produces:
    - macro_trend_1d: 1.0 if close_1d > ema50_1d else -1.0
    - macro_trend_1w: 1.0 if close_1w > ema50_1w else -1.0
    - macro_confluence_score: sum of the two (2, 0, or -2)

    Uses merge_asof with direction='backward' to map lower-frequency macro signals
    to high-frequency timestamps WITHOUT lookahead (each row only sees the most
    recent macro bar at or before it).
    """
    df = df.copy()
    # Default neutral values
    df['macro_trend_1d'] = 0.0
    df['macro_trend_1w'] = 0.0
    df['macro_confluence_score'] = 0.0

    try:
        # Derive weekly macro anchors from available daily macro history when 1w is unavailable.
        _wk_from_daily = False
        if (df_1w is None or df_1w.empty) and df_1d is not None and not df_1d.empty:
            _wk_from_daily = True
            weekly = df_1d.copy()
            weekly['timestamp'] = pd.to_datetime(weekly['timestamp'])
            weekly = weekly.sort_values('timestamp').set_index('timestamp')
            weekly = weekly['close'].rolling(window=7, min_periods=1).last().reset_index()
            weekly = weekly.dropna(subset=['close'])
            if not weekly.empty:
                df_1w = weekly[['timestamp', 'close']].copy()

        if (df_1d is None or df_1d.empty) and (df_1w is None or df_1w.empty):
            if macro_state:
                df['macro_trend_1d'] = float(macro_state.get('trend_1d', 0.0))
                df['macro_trend_1w'] = float(macro_state.get('trend_1w', 0.0))
                df['macro_confluence_score'] = float(macro_state.get('confluence_score', 0.0))
            return df

        macro_rows = []
        if df_1d is not None and not df_1d.empty:
            t1 = df_1d.copy()
            t1['timestamp'] = pd.to_datetime(t1['timestamp'])
            t1 = t1.sort_values('timestamp')
            t1['ema50_1d']  = t1['close'].ewm(span=50,  adjust=False).mean()
            t1['ema200_1d'] = t1['close'].ewm(span=200, adjust=False).mean()
            t1['sma200_1d'] = t1['close'].rolling(200, min_periods=1).mean()
            t1['macro_trend_1d']    = (t1['close'] > t1['ema50_1d']).astype(float).replace({0.0: -1.0, 1.0: 1.0})
            t1['macro_trend_200d']  = (t1['close'] > t1['ema200_1d']).astype(float).replace({0.0: -1.0, 1.0: 1.0})
            t1['dist_daily_ema200'] = (t1['close'] / (t1['ema200_1d'] + 1e-9)) - 1
            t1['dist_daily_sma200'] = (t1['close'] / (t1['sma200_1d'] + 1e-9)) - 1
            # A daily bar is stamped at its OPEN, but its close/EMA are only
            # known at its CLOSE — shift availability +1 day so an intraday 1h
            # row sees only COMPLETED daily bars (no lookahead). The still-
            # forming bar lands in the future; merge_asof(backward) skips it.
            t1['timestamp'] = t1['timestamp'] + pd.Timedelta(days=1)
            macro_rows.append(t1[['timestamp', 'macro_trend_1d', 'macro_trend_200d',
                                   'dist_daily_ema200', 'dist_daily_sma200']])

        if df_1w is not None and not df_1w.empty:
            t2 = df_1w.copy()
            t2['timestamp'] = pd.to_datetime(t2['timestamp'])
            t2 = t2.sort_values('timestamp')
            t2['ema50_1w'] = t2['close'].ewm(span=50, adjust=False).mean()
            t2['macro_trend_1w'] = (t2['close'] > t2['ema50_1w']).astype(float).replace({0.0: -1.0, 1.0: 1.0})
            # Same availability shift as the daily block: derived-from-daily
            # weekly rows are daily-stamped (+1 day); a real weekly bar is
            # stamped at its open and completes 7 days later.
            t2['timestamp'] = t2['timestamp'] + pd.Timedelta(days=1 if _wk_from_daily else 7)
            macro_rows.append(t2[['timestamp', 'macro_trend_1w']])

        if not macro_rows:
            if macro_state:
                df['macro_trend_1d'] = float(macro_state.get('trend_1d', 0.0))
                df['macro_trend_1w'] = float(macro_state.get('trend_1w', 0.0))
                df['macro_confluence_score'] = float(macro_state.get('confluence_score', 0.0))
            return df

        # Build macro_df by merging available macro rows on timestamp
        from functools import reduce
        macro_df = reduce(
            lambda left, right: pd.merge_asof(
                left.sort_values('timestamp'),
                right.sort_values('timestamp'),
                on='timestamp',
                direction='backward'
            ),
            macro_rows
        )

        for col in ('macro_trend_1d', 'macro_trend_1w', 'macro_trend_200d',
                    'dist_daily_ema200', 'dist_daily_sma200'):
            if col not in macro_df.columns:
                macro_df[col] = 0.0

        macro_df['macro_confluence_score'] = (
            macro_df['macro_trend_1d'].fillna(0.0) +
            macro_df['macro_trend_1w'].fillna(0.0)
        )

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        macro_df = macro_df.sort_values('timestamp')
        df = df.sort_values('timestamp')
        _macro_cols = ['timestamp', 'macro_trend_1d', 'macro_trend_1w',
                       'macro_trend_200d', 'dist_daily_ema200', 'dist_daily_sma200',
                       'macro_confluence_score']
        # THE BUG THAT ZEROED THE DAY-TREND FOR THE PROJECT'S HISTORY: the
        # neutral defaults pre-created at the top collide with the real merged
        # columns — pandas suffixes both sides to _x/_y, the plain-name read
        # below KeyErrors, and the except at the bottom silently returned the
        # zeros. Every model was trained with macro_trend_1d/1w as constants
        # (v58 revived the ENGINE side only). Drop the defaults so the merge
        # lands on clean names.
        df = df.drop(columns=[c for c in ('macro_trend_1d', 'macro_trend_1w',
                                          'macro_confluence_score')
                              if c in df.columns])
        merged = pd.merge_asof(
            df,
            macro_df[[c for c in _macro_cols if c in macro_df.columns]],
            on='timestamp',
            direction='backward'
        )

        for col in ('macro_trend_1d', 'macro_trend_1w', 'macro_trend_200d',
                    'dist_daily_ema200', 'dist_daily_sma200', 'macro_confluence_score'):
            merged[col] = merged[col].fillna(0.0)

        return merged
    except Exception as _macro_err:
        # Never silent again — the silent version hid the collision above for
        # the project's entire history.
        print(f"   [MACRO] add_macro_regime_features failed "
              f"({type(_macro_err).__name__}: {_macro_err}) — macro columns neutral 0.0")
        df['macro_trend_1d'] = df.get('macro_trend_1d', 0.0)
        df['macro_trend_1w'] = df.get('macro_trend_1w', 0.0)
        df['macro_confluence_score'] = df.get('macro_confluence_score', 0.0)
        return df

# ------------------------------------------------------------------
# Regime Classification (Adaptive Strategy)
# ------------------------------------------------------------------
def classify_trend_regime(adx: pd.Series, threshold_trend: float = 25, threshold_weak: float = 20) -> pd.Series:
    """0 = weak/choppy, 1 = trending"""
    return ((adx > threshold_trend) | (adx > threshold_weak)).astype(int)

def classify_volume_regime(volume_zscore: pd.Series, high_threshold: float = 2.0, low_threshold: float = -1.0) -> pd.Series:
    """-1 = low volume, 0 = normal, 1 = high volume"""
    regime = np.zeros_like(volume_zscore, dtype=int)
    regime[volume_zscore > high_threshold] = 1
    regime[volume_zscore < low_threshold] = -1
    return pd.Series(regime, index=volume_zscore.index)

def classify_market_phase(trend: pd.Series, vol_regime: pd.Series) -> pd.Series:
    """
    Combine trend and volatility into 4 phases:
    0 = trending high vol, 1 = trending low vol, 2 = ranging high vol, 3 = ranging low vol, 4 = normal
    """
    high_vol = (vol_regime > 1.2).astype(int)
    low_vol = (vol_regime < 0.8).astype(int)
    normal_vol = ((vol_regime >= 0.8) & (vol_regime <= 1.2)).astype(int)
    phase = np.zeros_like(trend, dtype=int)
    phase[(trend == 1) & (high_vol == 1)] = 0
    phase[(trend == 1) & (low_vol == 1)] = 1
    phase[(trend == 0) & (high_vol == 1)] = 2
    phase[(trend == 0) & (low_vol == 1)] = 3
    phase[((trend == 1) | (trend == 0)) & (normal_vol == 1)] = 4
    return pd.Series(phase, index=trend.index)

# ------------------------------------------------------------------
# Market Anchor & News Features
# ------------------------------------------------------------------
def add_btc_anchor_features(df: pd.DataFrame, btc_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    _anchor_cols = [
        'btc_1h_return', 'btc_4h_return', 'btc_dist_ema200',
        # Relative-performance features (token vs BTC).
        # These are the most important additions for high-vol alts like ETH:
        # the model can now see whether the token is leading/lagging BTC,
        # which is a strong short-term momentum signal.
        'rel_perf_1h',   # token 1h return − BTC 1h return (alpha vs BTC)
        'rel_perf_4h',   # token 4h return − BTC 4h return
        'rel_perf_24h',  # token 24h return − BTC 24h return
        'btc_ratio_ma_dist',  # (token/BTC ratio) distance from its 24h MA
                              # — captures mean-reversion / breakout in the ratio
    ]
    if btc_df is None or btc_df.empty:
        for c in _anchor_cols:
            df[c] = 0.0
        return df

    btc = btc_df.copy()
    btc['timestamp'] = pd.to_datetime(btc['timestamp'])
    btc['btc_1h_return'] = btc['close'].pct_change(1)
    btc['btc_4h_return'] = btc['close'].pct_change(4)
    btc['btc_24h_return'] = btc['close'].pct_change(24)
    btc['btc_ema_200']   = btc['close'].ewm(span=200, adjust=False).mean()
    btc['btc_dist_ema200'] = (btc['close'] / btc['btc_ema_200']) - 1

    keep = ['timestamp', 'btc_1h_return', 'btc_4h_return', 'btc_24h_return', 'btc_dist_ema200', 'close']
    df_ts = pd.to_datetime(df['timestamp'])
    btc_ts = pd.to_datetime(btc['timestamp'])

    merged = df.assign(timestamp=df_ts).merge(
        btc[keep].assign(timestamp=btc_ts).rename(columns={'close': '_btc_close'}),
        on='timestamp', how='left'
    )

    # Token returns for relative-performance calculation
    tok_1h  = merged['close'].pct_change(1)
    tok_4h  = merged['close'].pct_change(4)
    tok_24h = merged['close'].pct_change(24)

    merged['btc_1h_return']  = merged['btc_1h_return'].fillna(0)
    merged['btc_4h_return']  = merged['btc_4h_return'].fillna(0)
    merged['btc_dist_ema200'] = merged['btc_dist_ema200'].fillna(0)

    # Alpha: how much the token outperforms / underperforms BTC.
    # For BTC itself these are identically zero — set them explicitly to avoid
    # the feature consuming SHAP budget or creating rolling-window artefacts.
    _is_btc = merged['close'].equals(merged['_btc_close'].fillna(merged['close']))
    if _is_btc:
        merged['rel_perf_1h']      = 0.0
        merged['rel_perf_4h']      = 0.0
        merged['rel_perf_24h']     = 0.0
        merged['btc_ratio_ma_dist'] = 0.0
    else:
        merged['rel_perf_1h']  = (tok_1h  - merged['btc_1h_return']).fillna(0)
        merged['rel_perf_4h']  = (tok_4h  - merged['btc_4h_return']).fillna(0)
        merged['rel_perf_24h'] = (tok_24h - merged['btc_24h_return'].fillna(0)).fillna(0)

        # Token/BTC price ratio momentum (mean-reversion / breakout signal)
        btc_close = merged['_btc_close'].replace(0, np.nan)
        ratio = merged['close'] / btc_close
        ratio_ma24 = ratio.rolling(24, min_periods=4).mean()
        merged['btc_ratio_ma_dist'] = ((ratio - ratio_ma24) / (ratio_ma24 + 1e-9)).fillna(0)

    for c in _anchor_cols:
        df[c] = merged[c].values

    return df

def add_news_features(df: pd.DataFrame, news_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if news_df is None or news_df.empty:
        df['news_score'] = 0.0
        df['news_velocity'] = 0.0
        return df
    news = news_df.copy()
    news.set_index('timestamp', inplace=True)
    news_hourly = news.resample('1h').agg({'sentiment': 'mean'}).fillna(0)
    df = df.merge(news_hourly, left_on='timestamp', right_index=True, how='left')
    df['news_score'] = df['sentiment'].fillna(0).rolling(4, min_periods=1).mean()
    df['news_velocity'] = df['news_score'].diff(4).fillna(0)
    df.drop('sentiment', axis=1, inplace=True, errors='ignore')
    return df

# ------------------------------------------------------------------
# Target Creation (for supervised learning)
# ------------------------------------------------------------------
def add_target(df: pd.DataFrame, forward_hours: int = 24, atr_multiplier: float = 2.0) -> pd.DataFrame:
    """
    Create triple-barrier targets:
    2 = Buy (upper barrier hit first),
    0 = Sell (lower barrier hit first),
    1 = Hold (no barrier hit before the time barrier).
    """
    df = df.copy()
    if 'atr_14' not in df.columns:
        df['atr_14'] = compute_atr(df, 14)

    upper_barrier = df['close'] + atr_multiplier * df['atr_14']
    lower_barrier = df['close'] - atr_multiplier * df['atr_14']

    targets = np.full(len(df), np.nan)
    for i in range(len(df) - forward_hours):
        if pd.isna(upper_barrier.iloc[i]) or pd.isna(lower_barrier.iloc[i]):
            continue

        window = df.iloc[i + 1: i + forward_hours + 1]
        hit_up = window[window['high'] >= upper_barrier.iloc[i]]
        hit_down = window[window['low'] <= lower_barrier.iloc[i]]

        up_idx = hit_up.index[0] if not hit_up.empty else None
        down_idx = hit_down.index[0] if not hit_down.empty else None

        if up_idx is not None and down_idx is not None:
            if up_idx < down_idx:
                targets[i] = 2
            elif down_idx < up_idx:
                targets[i] = 0
            else:
                targets[i] = 1
        elif up_idx is not None:
            targets[i] = 2
        elif down_idx is not None:
            targets[i] = 0
        else:
            targets[i] = 1

    df['target'] = pd.Series(targets, index=df.index)
    return df

# ------------------------------------------------------------------
# Main Feature Engineering Pipeline (Regime-Adaptive)
# ------------------------------------------------------------------
# Any feature whose construction references future bars must be listed here so it
# can never silently re-enter the training set.
LOOKAHEAD_BLOCKLIST = ['ichimoku_chikou']


def prepare_features(df: pd.DataFrame,
                     btc_df: Optional[pd.DataFrame] = None,
                     news_df: Optional[pd.DataFrame] = None,
                     add_target_flag: bool = False,
                     forward_hours: int = 1,
                     df_1d: Optional[pd.DataFrame] = None,
                     df_1w: Optional[pd.DataFrame] = None,
                     macro_state: Optional[Dict[str, Any]] = None,
                     funding_df: Optional[pd.DataFrame] = None,
                     oi_df: Optional[pd.DataFrame] = None,
                     fg_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    compiled_features: Dict[str, pd.Series] = {}

    # ----- Basic price & volume -----
    compiled_features['returns_1h'] = df['close'].pct_change()
    compiled_features['returns_4h'] = df['close'].pct_change(4)
    compiled_features['log_returns'] = pd.Series(np.log(df['close'] / df['close'].shift(1)), index=df.index)

    # ----- Price Momentum & EMA Cluster -----
    ema_periods = [9, 21, 50, 100, 200]
    for p in ema_periods:
        ema = df['close'].ewm(span=p, adjust=False).mean()
        compiled_features[f'ema_{p}'] = ema
        compiled_features[f'dist_ema_{p}'] = (df['close'] / ema) - 1
        compiled_features[f'ema_slope_{p}'] = compute_ema_slope(df['close'], period=p)

    compiled_features['ema_9_21_cross'] = (compiled_features['ema_9'] > compiled_features['ema_21']).astype(int)
    compiled_features['ema_50_200_cross'] = (compiled_features['ema_50'] > compiled_features['ema_200']).astype(int)
    compiled_features['ema_stack_alignment'] = compute_ema_alignment_score(df, periods=ema_periods)
    compiled_features['ema_stack_distance'] = compute_ema_stack_distance(df, periods=ema_periods)

    if df_1d is not None or df_1w is not None:
        compiled_features['multi_tf_trend_agreement'] = compute_multi_timeframe_trend_agreement(df, df_1d, df_1w)
    else:
        compiled_features['multi_tf_trend_agreement'] = pd.Series(0.5, index=df.index)

    # ----- Oscillators (Momentum) -----
    compiled_features['rsi_14'] = compute_rsi(df['close'], 14)
    compiled_features['rsi_slope_14'] = compute_rsi_slope(df['close'], 14)
    compiled_features['rsi_acceleration_14'] = compute_rsi_acceleration(df['close'], 14)
    compiled_features['mfi_14'] = compute_mfi(df, 14)
    compiled_features['stoch_k'], compiled_features['stoch_d'] = compute_stoch_rsi(df['close'], 14)
    compiled_features['stoch_slope'] = compute_stoch_slope(df)
    compiled_features['macd'], compiled_features['macd_signal'], compiled_features['macd_hist'] = compute_macd(df['close'])
    compiled_features['macd_hist_velocity'] = compute_macd_hist_velocity(df)
    compiled_features['macd_divergence'] = compute_macd_divergence(df)

    # ----- Volume & Liquidity -----
    compiled_features['vwap'] = compute_vwap(df)
    # dist_vwap: use 24-bar rolling VWAP for stationarity.
    # The cumulative VWAP drifts with price history (PSI≈17 across bull/bear cycles).
    # A 24-bar rolling percentage distance is stationary and meaningful.
    _tp24 = (df['high'] + df['low'] + df['close']) / 3
    _vwap24 = ((_tp24 * df['volume']).rolling(24, min_periods=6).sum()
               / df['volume'].rolling(24, min_periods=6).sum())
    compiled_features['dist_vwap'] = (
        (df['close'] - _vwap24) / _vwap24.replace(0, np.nan)
    ).fillna(0.0)
    compiled_features['cmf_20'] = compute_cmf(df, 20)
    compiled_features['vol_velocity'] = df['volume'].pct_change(3)
    compiled_features['volume_zscore'] = (df['volume'] - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).std()
    compiled_features['relative_volume'] = df['volume'] / (df['volume'].rolling(20, min_periods=1).mean() + 1e-9)
    compiled_features['volume_acceleration'] = compiled_features['relative_volume'].diff().fillna(0.0)
    compiled_features['rolling_volatility'] = compiled_features['log_returns'].rolling(24, min_periods=1).std()
    compiled_features['vwap_slope_24'] = compute_vwap_slope(df, period=24)
    _obv_raw = df['close'].diff().apply(np.sign).mul(df['volume']).fillna(0).cumsum()
    _obv_mean = _obv_raw.rolling(100, min_periods=20).mean()
    _obv_std  = _obv_raw.rolling(100, min_periods=20).std().replace(0, np.nan).ffill().fillna(1.0)
    compiled_features['obv'] = ((_obv_raw - _obv_mean) / _obv_std).fillna(0.0)
    compiled_features['obv_slope_20'] = compute_obv_slope(df, period=20)
    compiled_features['obv_divergence'] = compute_obv_divergence(df, period=24)
    compiled_features['acc_dist'] = compute_accumulation_distribution(df)

    # ----- Volume Delta (CVD proxy) -----
    compiled_features['volume_delta'], compiled_features['volume_delta_14'] = compute_volume_delta(df, period=14)

    # ----- Bar Microstructure -----
    try:
        for col, s in compute_bar_structure(df).items():
            compiled_features[str(col)] = s
    except Exception:
        pass

    # ----- Rolling VWAP (24-bar) -----
    try:
        compiled_features['rolling_vwap_24'], compiled_features['dist_rolling_vwap'] = compute_rolling_vwap(df, period=24)
    except Exception:
        pass

    # ----- Time / Session Features -----
    try:
        for col, s in compute_time_features(df).items():
            compiled_features[str(col)] = s
    except Exception:
        pass

    # ----- Multi-period RSI -----
    compiled_features['rsi_7'] = compute_rsi(df['close'], 7)
    compiled_features['rsi_21'] = compute_rsi(df['close'], 21)

    # ----- Volatility & Statistics -----
    compiled_features['atr_14'] = compute_atr(df, 14)
    compiled_features['atr_pct'] = (compiled_features['atr_14'] / df['close'].replace(0, np.nan)).fillna(0.0)
    compiled_features['atr_percentile'] = compute_atr_percentile(df, atr_period=14, lookback=100)
    compiled_features['atr_expansion'] = compute_atr_expansion(df, atr_period=14, lookback=20)
    compiled_features['realized_volatility'] = compute_realized_volatility(df, period=24)
    compiled_features['volatility_ratio'] = compute_volatility_ratio(df, current_period=24, lookback=100)
    compiled_features['volatility_skew'] = df['close'].rolling(24).skew()
    compiled_features['volatility_kurt'] = df['close'].rolling(24).kurt()
    compiled_features['historical_volatility'] = compute_historical_volatility(df, period=24)

    # ----- Advanced 2026 Features -----
    compiled_features['adx_14'] = compute_adx(df, 14)
    compiled_features['efficiency_ratio_10'] = compute_efficiency_ratio(df['close'], period=10)
    compiled_features['bb_width_percentile'] = compute_bollinger_width_percentile(df, lookback=252)
    compiled_features['is_squeeze'] = (compiled_features['bb_width_percentile'] < 0.1).astype(int)
    compiled_features['fvg_distance'] = compute_fvg_distance(df)
    compiled_features['volume_atr_efficiency'] = compute_volume_volatility_efficiency(df)
    compiled_features['price_zscore_200'] = compute_price_zscore(df, 200)
    compiled_features['volatility_regime'] = compute_volatility_regime(df, atr_period=14, lookback=100)
    compiled_features['higher_high_count_20'] = compute_structure_counts(df, lookback=20)['higher_high_count']
    compiled_features['lower_low_count_20'] = compute_structure_counts(df, lookback=20)['lower_low_count']
    compiled_features['swing_strength_20'] = compute_swing_strength(df, lookback=20)
    compiled_features['structure_bias'] = (compiled_features['higher_high_count_20'] - compiled_features['lower_low_count_20']) / 20.0

    # ----- Return Lags -----
    for h in [1, 4, 12, 24]:
        compiled_features[f'ret_{h}h'] = df['close'].pct_change(h)

    # ----- Primary Model Precision Features (high-signal, regime-independent) -----
    # trend_consistency_20: fraction of last 20 bars where close > open.
    # Captures persistent directional bias without depending on absolute price level.
    _up_bars = (df['close'] > df['open']).astype(float)
    compiled_features['trend_consistency_20'] = _up_bars.rolling(20, min_periods=5).mean().fillna(0.5)

    # price_disp_atr_pct: (close - EMA_50) / ATR_14.
    # Normalises trend displacement by current volatility. Stationary across tokens
    # and regimes — a +3 ATR displacement means the same thing whether the token is BTC or DOGE.
    _atr14_tmp = compute_atr(df, 14)
    _ema50_tmp = df['close'].ewm(span=50, adjust=False).mean()
    compiled_features['price_disp_atr_pct'] = (
        (df['close'] - _ema50_tmp) / (_atr14_tmp.replace(0, np.nan) + 1e-9)
    ).clip(-10, 10).fillna(0.0)

    # vol_confirm_ratio: ratio of total volume on up-closes vs down-closes over 20 bars.
    # > 1.0 = buyers dominate (bullish), < 1.0 = sellers dominate (bearish).
    # Complements CMF (which uses intra-bar position) by using net close direction.
    _vol_up   = (df['volume'] * (df['close'] > df['close'].shift(1)).astype(float)).rolling(20, min_periods=5).sum()
    _vol_down = (df['volume'] * (df['close'] < df['close'].shift(1)).astype(float)).rolling(20, min_periods=5).sum()
    compiled_features['vol_confirm_ratio'] = (
        (_vol_up / (_vol_down + 1e-9)).clip(0.1, 10.0).apply(np.log)  # log-scale for XGBoost symmetry
    ).fillna(0.0)

    # ==================== NEW FEATURES ====================
    compiled_features['aroon_up'], compiled_features['aroon_down'], compiled_features['aroon_osc'] = compute_aroon(df, period=25)
    compiled_features['keltner_upper'], compiled_features['keltner_lower'], compiled_features['keltner_width'] = compute_keltner_channels(df)
    compiled_features['choppiness'] = compute_choppiness_index(df, period=14)
    ichimoku = compute_ichimoku(df)
    for col in ichimoku.columns:
        compiled_features[col] = ichimoku[col]
    compiled_features['linreg_slope_14'] = compute_linear_regression_slope(df['close'], 14)
    compiled_features['linreg_r2_14'] = compute_r_squared(df['close'], 14)
    compiled_features['williams_r'] = compute_williams_r(df, 14)
    compiled_features['ultimate_osc'] = compute_ultimate_oscillator(df)
    compiled_features['vi_plus'], compiled_features['vi_minus'] = compute_vortex_indicator(df, period=14)
    compiled_features['force_index'] = compute_force_index(df, period=13)

    # ----- Support & Resistance structural features -----
    try:
        sr_features = add_support_resistance_features(df[['high', 'low', 'close']].copy(), window=24)
        for col in sr_features.columns:
            compiled_features[col] = sr_features[col]

        compiled_features['is_at_support'] = (compiled_features['pct_dist_to_support'] <= 0.015).astype(int)
        compiled_features['is_at_resistance'] = (compiled_features['pct_dist_to_resistance'] <= 0.015).astype(int)
    except Exception:
        pass

    # ----- Candlestick Patterns -----
    cdl_patterns = compute_candlestick_patterns(df[['open', 'high', 'low', 'close']])
    for col in cdl_patterns.columns:
        compiled_features[col] = cdl_patterns[col]

    # ==================== EXTENDED INDICATOR LIBRARY ====================

    # --- MA Variants ---
    try:
        compiled_features['hma_20'] = compute_hma(df['close'], 20)
        compiled_features['dist_hma20'] = (df['close'] / (compiled_features['hma_20'] + 1e-9)) - 1
        compiled_features['kama_10'] = compute_kama(df['close'], 10)
        compiled_features['dist_kama'] = (df['close'] / (compiled_features['kama_10'] + 1e-9)) - 1
        compiled_features['tema_21'] = compute_tema(df['close'], 21)
        compiled_features['dema_21'] = compute_dema(df['close'], 21)
        compiled_features['t3_5'] = compute_t3(df['close'], 5)
        compiled_features['vwma_20'] = compute_vwma(df, 20)
        compiled_features['dist_vwma20'] = (df['close'] / (compiled_features['vwma_20'] + 1e-9)) - 1
    except Exception:
        pass

    # --- Trend ---
    try:
        compiled_features['supertrend'], compiled_features['supertrend_dir'] = compute_supertrend(df)
        compiled_features['supertrend_dist'] = (df['close'] - compiled_features['supertrend']) / (df['close'] + 1e-9)
    except Exception:
        pass
    try:
        _sar, _sar_trend = compute_parabolic_sar(df)
        compiled_features['sar_trend'] = _sar_trend
        compiled_features['sar_dist'] = (df['close'] - _sar) / (df['close'] + 1e-9)
    except Exception:
        pass
    try:
        _du, _dl, _dm = compute_donchian_channels(df, 20)
        compiled_features['donchian_width'] = (_du - _dl) / (_dm + 1e-9)
        compiled_features['donchian_position'] = (df['close'] - _dl) / (_du - _dl + 1e-9)
    except Exception:
        pass

    # --- Oscillators ---
    try:
        compiled_features['cci_20'] = compute_cci(df, 20)
        compiled_features['tsi'] = compute_tsi(df['close'])
        compiled_features['cmo_14'] = compute_cmo(df['close'], 14)
        compiled_features['dpo_20'] = compute_dpo(df['close'], 20)
        compiled_features['ppo'], compiled_features['ppo_signal'] = compute_ppo(df['close'])
        compiled_features['trix_15'] = compute_trix(df['close'], 15)
        compiled_features['kst'], compiled_features['kst_signal'] = compute_kst(df['close'])
        compiled_features['schaff_tc'] = compute_schaff_trend_cycle(df['close'])
        compiled_features['awesome_osc'] = compute_awesome_oscillator(df)
        compiled_features['bop'] = compute_bop(df)
        compiled_features['eom_14'] = compute_eom(df, 14)
        compiled_features['fisher'], compiled_features['fisher_sig'] = compute_fisher_transform(df)
        compiled_features['rvi_osc'], compiled_features['rvi_sig'] = compute_rvi(df)
        compiled_features['roc_14'] = compute_rate_of_change(df['close'], 14)
    except Exception:
        pass

    # --- Volatility ---
    try:
        compiled_features['bb_pct_b'] = compute_bollinger_pct_b(df, 20)
        compiled_features['parkinson_vol'] = compute_parkinson_volatility(df, 24)
        compiled_features['gk_vol'] = compute_garman_klass_volatility(df, 24)
        _ab_u, _ab_l = compute_atr_bands(df, 14, 2.0)
        compiled_features['atr_band_position'] = (df['close'] - _ab_l) / (_ab_u - _ab_l + 1e-9)
        _sb_u, _sb_l = compute_starc_bands(df)
        compiled_features['starc_position'] = (df['close'] - _sb_l) / (_sb_u - _sb_l + 1e-9)
        compiled_features['rvi_vol'] = compute_relative_volatility_index(df['close'])
        _gc_u, _gc_l = compute_gaussian_channel(df['close'])
        compiled_features['gaussian_position'] = (df['close'] - _gc_l) / (_gc_u - _gc_l + 1e-9)
    except Exception:
        pass

    # --- Volume ---
    try:
        _pvt_raw = compute_pvt(df)
        _pvt_mean = _pvt_raw.rolling(100, min_periods=20).mean()
        _pvt_std  = _pvt_raw.rolling(100, min_periods=20).std().replace(0, np.nan).ffill().fillna(1.0)
        compiled_features['pvt'] = ((_pvt_raw - _pvt_mean) / _pvt_std).fillna(0.0)
        compiled_features['kvo'], compiled_features['kvo_signal'] = compute_klinger_oscillator(df)
    except Exception:
        pass

    # --- Market Structure ---
    try:
        for col, s in compute_fibonacci_levels(df, lookback=100).items():
            compiled_features[str(col)] = s
    except Exception:
        pass
    try:
        for col, s in compute_bos_choch(df, lookback=20).items():
            compiled_features[str(col)] = s
    except Exception:
        pass
    try:
        for col, s in compute_order_blocks(df).items():
            compiled_features[str(col)] = s
    except Exception:
        pass

    # --- Statistical / Quant ---
    try:
        _se_u, _se_m, _se_l = compute_standard_error_bands(df['close'], 21)
        compiled_features['se_position'] = (df['close'] - _se_l) / (_se_u - _se_l + 1e-9)
        compiled_features['se_mid'] = _se_m
    except Exception:
        pass
    try:
        _qu, _qm, _ql = compute_quantile_bands(df['close'], 100)
        compiled_features['quantile_position'] = (df['close'] - _ql) / (_qu - _ql + 1e-9)
    except Exception:
        pass
    try:
        compiled_features['hurst'] = compute_hurst_exponent(df['close'], 100)
    except Exception:
        pass
    try:
        compiled_features['entropy'] = compute_entropy(df['close'], 20)
    except Exception:
        pass

    # ═══════════════════════════════════════════════════════════════
    # CRYPTO-SPECIFIC INDICATOR EXTENSIONS
    # All wrapped in try/except so one failure never kills the pipeline.
    # ═══════════════════════════════════════════════════════════════

    # --- Squeeze Momentum (BB inside Keltner = spring-loaded consolidation) ---
    try:
        for col, s in compute_squeeze_momentum(df).items():
            compiled_features[str(col)] = s
    except Exception:
        pass

    # --- Elder Ray (buying / selling pressure relative to EMA) ---
    try:
        for col, s in compute_elder_ray(df).items():
            compiled_features[str(col)] = s
    except Exception:
        pass

    # --- Ichimoku extended signal columns (cloud position, TK cross, distances) ---
    # These depend on ichimoku_tenkan / kijun / senkou_a / senkou_b which are
    # already in compiled_features from the earlier compute_ichimoku() call.
    # We use a temporary df that has both the original OHLCV AND the ichi cols.
    try:
        _ichi_tmp = df.copy()
        for _ichi_col in ('ichimoku_tenkan', 'ichimoku_kijun', 'ichimoku_senkou_a', 'ichimoku_senkou_b'):
            if _ichi_col in compiled_features:
                _ichi_tmp[_ichi_col] = compiled_features[_ichi_col].values
        for col, s in compute_ichimoku_signals(_ichi_tmp).items():
            compiled_features[str(col)] = s
    except Exception:
        pass

    # --- Volume Pressure Composite ---
    try:
        for col, s in compute_volume_pressure(df).items():
            compiled_features[str(col)] = s
    except Exception:
        pass

    # --- Multi-period EMA Trend Alignment ---
    try:
        for col, s in compute_trend_alignment(df).items():
            compiled_features[str(col)] = s
    except Exception:
        pass

    # --- Funding rate derived signals (crypto-specific) ---
    # These rely on funding_rate which is added by add_futures_features() later.
    # We store a placeholder here and patch after the bulk concat below.
    # (Actual computation happens after the bulk concat when funding_rate exists.)

    # Defensive guard: never let a known lookahead feature into the frame.
    for bad in LOOKAHEAD_BLOCKLIST:
        compiled_features.pop(bad, None)

    # Bulk-attach all generated feature columns in one allocation pass.
    df = pd.concat([df, pd.DataFrame(compiled_features, index=df.index)], axis=1)

    # ----- Pivot Points (from daily OHLC) -----
    try:
        pivots = compute_pivot_points(df_1d, df)
        for col in pivots.columns:
            df[col] = pivots[col].values
            df[f'dist_{col}'] = (df['close'] - df[col]) / (df['close'] + 1e-9)
    except Exception:
        pass

    # ----- Market Anchor (BTC) -----
    df = add_btc_anchor_features(df, btc_df)
    if btc_df is not None and not btc_df.empty:
        df['btc_corr_24h'] = compute_rolling_correlation_with_btc(df, btc_df, period=24)

    # ----- News Sentiment -----
    df = add_news_features(df, news_df)

    # ----- Futures Features (funding rate + open interest) -----
    if funding_df is not None or oi_df is not None:
        try:
            df = add_futures_features(df, funding_df, oi_df)
        except Exception:
            pass

    # ----- Crypto microstructure: funding / OI derived signals -----
    # Computed AFTER add_futures_features so funding_rate / open_interest exist.
    try:
        for col, s in compute_funding_signals(df).items():
            df[col] = s.values
    except Exception:
        pass
    try:
        for col, s in compute_oi_signals(df).items():
            df[col] = s.values
    except Exception:
        pass

    # ----- Macro Regime Features (map 1d/1w onto base timeframe) -----
    try:
        df = add_macro_regime_features(df, df_1d, df_1w, macro_state=macro_state)
    except Exception:
        pass

    # ----- Regime Classification -----
    df['trend_regime'] = classify_trend_regime(df['adx_14'], threshold_trend=25, threshold_weak=20)
    df['volume_regime'] = classify_volume_regime(df['volume_zscore'], high_threshold=2.0, low_threshold=-1.0)
    df['market_phase'] = classify_market_phase(df['trend_regime'], df['volatility_regime'])
    df['multi_tf_trend_agreement'] = compiled_features.get('multi_tf_trend_agreement', pd.Series(0.5, index=df.index))
    df['market_regime'] = compute_market_regime(
        df['adx_14'],
        df['volatility_regime'],
        compiled_features['ema_stack_alignment'],
        df.get('macro_trend_1d') if 'macro_trend_1d' in df.columns else None,
    )
    # Safe encoding: one-hot + numeric code + unknown handling
    try:
        onehot, vocab = encode_regime_series(df['market_regime'])
        # merge one-hot columns into df
        for c in onehot.columns:
            df[c] = onehot[c].values
        # derive a numeric regime confidence mapping if not present
        if 'regime_confidence' not in df.columns:
            mapping = {
                'TRENDING_BULL': 0.90,
                'TRENDING_BEAR': 0.90,
                'HIGH_VOL': 0.75,
                'LOW_VOL': 0.75,
                'RANGE': 0.60,
            }
            df['regime_confidence'] = df['market_regime'].map(mapping).fillna(0.50)
    except Exception:
        # Fall back to previous conservative behaviour
        df['regime_confidence'] = df['market_regime'].map({
            'TRENDING_BULL': 0.90,
            'TRENDING_BEAR': 0.90,
            'HIGH_VOL': 0.75,
            'LOW_VOL': 0.75,
            'RANGE': 0.60,
        }).fillna(0.50)
        for regime in ['TRENDING_BULL', 'TRENDING_BEAR', 'RANGE', 'HIGH_VOL', 'LOW_VOL']:
            df[f'regime_{regime.lower()}'] = (df['market_regime'] == regime).astype(int)

    # Compute rolling regime performance metrics (historical-only)
    try:
        regime_metrics = compute_regime_metrics(df, timestamp_col='timestamp', symbol_col='symbol',
                                                regime_col='market_regime', outcome_col='returns_1h',
                                                window_days=90)
        for col in regime_metrics.columns:
            df[col] = regime_metrics[col].values
    except Exception:
        # If metrics computation fails, fill conservative defaults
        df['regime_quality_score'] = 0.0
        df['regime_expectancy'] = 0.0
        df['regime_win_rate'] = 0.0
        df['regime_pf'] = 0.0
        df['regime_sharpe'] = 0.0

    # ----- Target (optional) -----
    if add_target_flag:
        df = add_target(df, forward_hours=forward_hours)

    # ----- Delta & Decay features: short-term deltas and exponentially-weighted aggregates -----
    try:
        df = add_delta_and_decay_features(df, cols=['close', 'volume', 'vwap', 'returns_1h'], half_life=24.0)
        # Convert absolute-price decay means to price-relative distances.
        # close_decay_mean_24 is an exponentially-weighted close price — it drifts
        # proportionally with price level (PSI > 12 at BTC 20k vs 60k).
        # Replacing with (close - decay_mean) / close gives a stationary signal.
        _close = df['close'].replace(0, np.nan)
        if 'close_decay_mean_24' in df.columns:
            df['close_decay_mean_24'] = ((_close - df['close_decay_mean_24']) / _close).fillna(0.0)
        if 'close_decay_std_24' in df.columns:
            df['close_decay_std_24'] = (df['close_decay_std_24'] / _close).fillna(0.0)
        if 'vwap_decay_mean_24' in df.columns:
            df['vwap_decay_mean_24'] = ((_close - df['vwap_decay_mean_24']) / _close).fillna(0.0)
        if 'vwap_decay_std_24' in df.columns:
            df['vwap_decay_std_24'] = (df['vwap_decay_std_24'] / _close).fillna(0.0)
    except Exception:
        pass

    # ----- Anchored VWAP (50 / 100 / 200 bar anchors) -----
    try:
        avwap = compute_anchored_vwap(df, lookback=100)
        for col in avwap.columns:
            df[col] = avwap[col].values
    except Exception:
        pass

    # ----- Weekly Timeframe (MACD, RSI, 200 EMA/SMA, BOS/CHoCH) -----
    try:
        wf = compute_weekly_features(df, df_1d)
        if not wf.empty:
            for col in wf.columns:
                df[col] = wf[col].values
    except Exception:
        pass

    # ----- Fear & Greed Index -----
    try:
        df = add_fear_greed_features(df, fg_df)
    except Exception:
        pass

    # ----- Category Confluence (runs last — needs every other feature present) -----
    # Computes six pre-aggregated voter-group scores (momentum, trend, volume,
    # bands, smart-money, candlestick) plus a weighted total.  These composite
    # features let XGBoost see category-level consensus directly, mimicking the
    # "majority of indicators must agree" rule without hard-coding it as logic.
    try:
        conf = compute_category_confluence(df)
        if not conf.empty:
            for col in conf.columns:
                df[col] = conf[col].values
    except Exception:
        pass

    # ----- Final guard: drop any blocklisted lookahead column that slipped through -----
    df = df.drop(columns=[c for c in LOOKAHEAD_BLOCKLIST if c in df.columns], errors='ignore')

    # ----- Fill NaN boundaries introduced by indicators / multi-timeframe joins -----
    # CRITICAL: forward-fill ONLY. The previous version used .ffill().bfill();
    # the .bfill() pulled FUTURE values backward into NaN gaps across the whole
    # frame (including across the eventual train/test boundary), which is
    # lookahead leakage. Forward-fill only carries past info forward. Any leading
    # NaNs that remain are dropped below or zero-filled, never back-filled.
    try:
        df = df.sort_values('timestamp').ffill()
    except Exception:
        df = df.ffill()

    # ----- Cleanup: drop rows with essential NaNs (warm-up period at the start) -----
    required_cols = ['atr_14', 'rsi_14', 'adx_14', 'volume_atr_efficiency', 'volatility_regime']
    if add_target_flag:
        required_cols.append('target')
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    # ----- Purge indicator warm-up: drop the leading rows where the WIDEST rolling
    # window has not yet saturated. The 252-bar Bollinger-width percentile is the
    # longest lookback; before bar 252 it self-fills to 0.5 (a constant, not real
    # signal), and the 200-span EMAs / z-scores are still settling. Training on
    # those rows teaches the model on placeholder values. We drop them outright
    # rather than zero-filling. This is a one-time trim at the START of the series,
    # so it costs no recent data and creates no lookahead.
    MAX_WARMUP = 252  # = longest rolling lookback used in this pipeline
    if len(df) > MAX_WARMUP * 2:           # only trim if we can spare it
        df = df.iloc[MAX_WARMUP:].reset_index(drop=True)

    # ----- Clean any remaining infinite values for XGBoost (zero-fill: leakage-safe) -----
    df = clean_infinite_values(df, max_value=1e6, fill_method='zero')

    return df