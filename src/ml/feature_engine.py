import pandas as pd
import numpy as np
from typing import Optional, Tuple, List

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
    for i in range(len(df)):
        best_dist = float('inf')
        for j in range(max(0, i-50), min(len(df), i+50)):
            if not pd.isna(fvg_low.iloc[j]) and not pd.isna(fvg_high.iloc[j]):
                if current_price.iloc[i] < fvg_low.iloc[j]:
                    dist = (fvg_low.iloc[j] - current_price.iloc[i]) / (current_price.iloc[i] + 1e-9)
                elif current_price.iloc[i] > fvg_high.iloc[j]:
                    dist = (current_price.iloc[i] - fvg_high.iloc[j]) / (current_price.iloc[i] + 1e-9)
                else:
                    dist = 0.0
                if dist < best_dist:
                    best_dist = dist
        distance.iloc[i] = best_dist if best_dist != float('inf') else 0.0
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
    """Add Ichimoku Cloud components."""
    tenkan = (df['high'].rolling(tenkan_period).max() + df['low'].rolling(tenkan_period).min()) / 2
    kijun = (df['high'].rolling(kijun_period).max() + df['low'].rolling(kijun_period).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(kijun_period)
    senkou_b = ((df['high'].rolling(senkou_b).max() + df['low'].rolling(senkou_b).min()) / 2).shift(kijun_period)
    chikou = df['close'].shift(-kijun_period)
    ichimoku = pd.DataFrame({
        'ichimoku_tenkan': tenkan,
        'ichimoku_kijun': kijun,
        'ichimoku_senkou_a': senkou_a,
        'ichimoku_senkou_b': senkou_b,
        'ichimoku_chikou': chikou
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
    if btc_df is None or btc_df.empty:
        df['btc_1h_return'] = 0.0
        df['btc_4h_return'] = 0.0
        df['btc_dist_ema200'] = 0.0
        return df
    btc = btc_df.copy()
    btc['timestamp'] = pd.to_datetime(btc['timestamp'])
    btc['btc_1h_return'] = btc['close'].pct_change()
    btc['btc_4h_return'] = btc['close'].pct_change(4)
    btc['btc_ema_200'] = btc['close'].ewm(span=200, adjust=False).mean()
    btc['btc_dist_ema200'] = (btc['close'] / btc['btc_ema_200']) - 1
    keep = ['timestamp', 'btc_1h_return', 'btc_4h_return', 'btc_dist_ema200']
    df = df.merge(btc[keep], on='timestamp', how='left')
    df[['btc_1h_return', 'btc_4h_return', 'btc_dist_ema200']] = df[['btc_1h_return', 'btc_4h_return', 'btc_dist_ema200']].fillna(0)
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
def add_target(df: pd.DataFrame, forward_hours: int = 1) -> pd.DataFrame:
    """
    Create binary target: 1 if future close > current close, else 0.
    forward_hours: number of hours to look ahead (1 = next candle).
    """
    future_return = df['close'].shift(-forward_hours) / df['close'] - 1
    df['target'] = (future_return > 0).astype(int)
    return df

# ------------------------------------------------------------------
# Main Feature Engineering Pipeline (Regime-Adaptive)
# ------------------------------------------------------------------
def prepare_features(df: pd.DataFrame,
                     btc_df: Optional[pd.DataFrame] = None,
                     news_df: Optional[pd.DataFrame] = None,
                     add_target_flag: bool = False,
                     forward_hours: int = 1) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # ----- Basic price & volume -----
    df['returns_1h'] = df['close'].pct_change()
    df['returns_4h'] = df['close'].pct_change(4)
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    # ----- Price Momentum & EMA Cluster -----
    for p in [9, 21, 50, 100, 200]:
        df[f'ema_{p}'] = df['close'].ewm(span=p, adjust=False).mean()
        df[f'dist_ema_{p}'] = (df['close'] / df[f'ema_{p}']) - 1

    df['ema_9_21_cross'] = (df['ema_9'] > df['ema_21']).astype(int)
    df['ema_50_200_cross'] = (df['ema_50'] > df['ema_200']).astype(int)

    # ----- Oscillators (Momentum) -----
    df['rsi_14'] = compute_rsi(df['close'], 14)
    df['mfi_14'] = compute_mfi(df, 14)
    df['stoch_k'], df['stoch_d'] = compute_stoch_rsi(df['close'], 14)
    df['macd'], df['macd_signal'], df['macd_hist'] = compute_macd(df['close'])

    # ----- Volume & Liquidity -----
    df['vwap'] = compute_vwap(df)
    df['dist_vwap'] = (df['close'] / df['vwap']) - 1
    df['cmf_20'] = compute_cmf(df, 20)
    df['vol_velocity'] = df['volume'].pct_change(3)
    df['volume_zscore'] = (df['volume'] - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).std()
    df['obv'] = df['close'].diff().apply(np.sign).mul(df['volume']).fillna(0).cumsum()
    df['acc_dist'] = compute_accumulation_distribution(df)

    # ----- Volatility & Statistics -----
    df['atr_14'] = compute_atr(df, 14)
    df['volatility_skew'] = df['close'].rolling(24).skew()
    df['volatility_kurt'] = df['close'].rolling(24).kurt()
    df['historical_volatility'] = compute_historical_volatility(df, period=24)

    # ----- Advanced 2026 Features -----
    df['adx_14'] = compute_adx(df, 14)
    df['efficiency_ratio_10'] = compute_efficiency_ratio(df['close'], period=10)
    df['bb_width_percentile'] = compute_bollinger_width_percentile(df, lookback=252)
    df['is_squeeze'] = (df['bb_width_percentile'] < 0.1).astype(int)
    df['fvg_distance'] = compute_fvg_distance(df)
    df['volume_atr_efficiency'] = compute_volume_volatility_efficiency(df)
    df['price_zscore_200'] = compute_price_zscore(df, 200)
    df['volatility_regime'] = compute_volatility_regime(df, atr_period=14, lookback=100)

    # ----- Return Lags -----
    for h in [1, 4, 12, 24]:
        df[f'ret_{h}h'] = df['close'].pct_change(h)

    # ==================== NEW FEATURES ====================
    # Aroon
    df['aroon_up'], df['aroon_down'], df['aroon_osc'] = compute_aroon(df, period=25)
    # Keltner Channels
    df['keltner_upper'], df['keltner_lower'], df['keltner_width'] = compute_keltner_channels(df)
    # Choppiness Index
    df['choppiness'] = compute_choppiness_index(df, period=14)
    # Ichimoku
    ichimoku = compute_ichimoku(df)
    for col in ichimoku.columns:
        df[col] = ichimoku[col]
    # Linear regression slope & R²
    df['linreg_slope_14'] = compute_linear_regression_slope(df['close'], 14)
    df['linreg_r2_14'] = compute_r_squared(df['close'], 14)
    # Williams %R
    df['williams_r'] = compute_williams_r(df, 14)
    # Ultimate Oscillator
    df['ultimate_osc'] = compute_ultimate_oscillator(df)
    # Vortex
    df['vi_plus'], df['vi_minus'] = compute_vortex_indicator(df, period=14)
    # Force Index
    df['force_index'] = compute_force_index(df, period=13)

    # ----- Market Anchor (BTC) -----
    df = add_btc_anchor_features(df, btc_df)
    if btc_df is not None and not btc_df.empty:
        df['btc_corr_24h'] = compute_rolling_correlation_with_btc(df, btc_df, period=24)

    # ----- News Sentiment -----
    df = add_news_features(df, news_df)

    # ----- Regime Classification -----
    df['trend_regime'] = classify_trend_regime(df['adx_14'], threshold_trend=25, threshold_weak=20)
    df['volume_regime'] = classify_volume_regime(df['volume_zscore'], high_threshold=2.0, low_threshold=-1.0)
    df['market_phase'] = classify_market_phase(df['trend_regime'], df['volatility_regime'])

    # ----- Target (optional) -----
    if add_target_flag:
        df = add_target(df, forward_hours=forward_hours)

    # ----- Cleanup: drop rows with essential NaNs -----
    required_cols = ['atr_14', 'rsi_14', 'adx_14', 'volume_atr_efficiency', 'volatility_regime']
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    return df