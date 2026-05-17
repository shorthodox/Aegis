import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

def compute_exp_decay_weights(length: int, half_life: float) -> np.ndarray:
    """Return exponential decay weights of given length and half-life (in same units as index spacing)."""
    if length <= 1:
        return np.array([1.0])
    # alpha such that weight(t) = exp(-alpha * t) and half-life -> exp(-alpha * hl) = 0.5
    alpha = np.log(2) / float(max(half_life, 1e-9))
    idx = np.arange(length)[::-1]
    w = np.exp(-alpha * idx)
    return w / w.sum()


def add_delta_and_decay_features(df: pd.DataFrame, cols: Optional[List[str]] = None, half_life: float = 24.0) -> pd.DataFrame:
    """
    Add delta (difference) and exponential-decay aggregated features for selected numeric columns.
    - `cols`: list of columns to compute deltas for (defaults to ['close','volume','vwap','returns_1h']).
    - `half_life`: half-life (in rows) for decay weights; default 24 (hours if 1h timeframe).
    This helps the model learn short-term momentum (deltas) and decayed context (recent importance).
    """
    if df is None or df.empty:
        return df
    df = df.copy()
    if cols is None:
        cols = ['close', 'volume', 'vwap', 'returns_1h']

    n = len(df)
    weights = compute_exp_decay_weights(min(256, n), half_life)

    for c in cols:
        if c not in df.columns:
            continue
        # simple deltas at multiple horizons
        df[f'{c}_delta_1'] = df[c].diff(1).fillna(0)
        df[f'{c}_delta_4'] = df[c].diff(4).fillna(0)
        df[f'{c}_delta_12'] = df[c].diff(12).fillna(0)

        # decayed mean and std over recent window using exponential weights
        window = min(len(weights), n)
        if window > 0:
            # use rolling apply with weighted functions when possible
            try:
                arr = df[c].to_numpy()
                # padded rolling: compute decayed mean for each position using last `window` values
                decayed_means = np.zeros(n)
                decayed_stds = np.zeros(n)
                for i in range(n):
                    start = max(0, i - window + 1)
                    seg = arr[start:i+1]
                    w = weights[-len(seg):]
                    if seg.size == 0:
                        decayed_means[i] = 0.0
                        decayed_stds[i] = 0.0
                    else:
                        decayed_means[i] = float(np.sum(seg * w))
                        decayed_stds[i] = float(np.sqrt(np.sum(w * (seg - decayed_means[i])**2)))
                df[f'{c}_decay_mean_{int(half_life)}'] = decayed_means
                df[f'{c}_decay_std_{int(half_life)}'] = decayed_stds
            except Exception:
                df[f'{c}_decay_mean_{int(half_life)}'] = df[c].ewm(span=half_life, adjust=False).mean().fillna(0)
                df[f'{c}_decay_std_{int(half_life)}'] = df[c].ewm(span=half_life, adjust=False).std().fillna(0)

    return df


def smooth_probability_matrix(probs: np.ndarray, span: float = 6.0) -> np.ndarray:
    """
    Smooth a (n_samples, n_classes) probability matrix along the time axis using EWMA.
    `span` controls the degree of smoothing (smaller -> less smoothing).
    """
    if probs is None:
        return probs
    try:
        dfp = pd.DataFrame(probs)
        sm = dfp.ewm(span=span, adjust=False).mean().to_numpy()
        return sm
    except Exception:
        return probs


def adjust_threshold_by_technical_and_fundamental(base_threshold: float,
                                                  vol_regime: Optional[object] = None,
                                                  news_score: float = 0.0,
                                                  efficiency_ratio: float = 0.0,
                                                  btc_anchor: float = 0.0) -> float:
    """
    Adjust an entry threshold based on market technicals and fundamentals.
    - `vol_regime`: higher values (>1) indicate higher volatility -> require slightly higher threshold
    - `news_score`: positive sentiment lowers threshold for buys (more permissive)
    - `efficiency_ratio`: trending markets (higher) can lower threshold
    - `btc_anchor`: if strongly positive, bias buy threshold down; if negative, increase
    Returns clipped threshold in [0.05, 0.99].
    """
    t = float(base_threshold)
    # volatility effect
    # Handle both numeric regimes (e.g., volatility ratio) and categorical strings ('high','low')
    try:
        if vol_regime is not None:
            if isinstance(vol_regime, str):
                v = vol_regime.lower()
                if v == 'high' or v == 'high_volatility' or v == 'high_vol':
                    t += 0.03
                elif v == 'low' or v == 'low_volatility' or v == 'low_vol':
                    t -= 0.02
            else:
                # numeric
                vnum = float(vol_regime)
                if vnum > 1.25:
                    t += 0.03
                elif vnum < 0.85:
                    t -= 0.02
    except Exception:
        pass

    # news sentiment (positive -> more permissive for buys)
    t -= 0.025 * float(np.tanh(news_score))

    # efficiency_ratio: strong trend -> allow slightly lower threshold
    try:
        t -= 0.02 * float(np.clip(efficiency_ratio, -1.0, 1.0))
    except Exception:
        pass

    # BTC anchor: strong positive -> slightly more permissive for buys
    t -= 0.015 * float(np.tanh(btc_anchor))

    # Clip
    t = float(np.clip(t, 0.05, 0.99))
    return t
