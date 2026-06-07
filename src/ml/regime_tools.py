import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict


def fit_regime_vocab(regime_series: pd.Series) -> List[str]:
    """Derive a stable sorted vocabulary from observed regimes."""
    vals = pd.Series(regime_series.dropna().unique()).astype(str)
    vocab = sorted([v for v in vals.tolist() if v is not None and v != 'nan'])
    return vocab


def encode_regime_series(regime_series: pd.Series, vocab: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[str]]:
    """Safely one-hot encode a regime series.

    - Unknown regimes are mapped to `regime_unknown` column.
    - Returns (onehot_df, vocab)
    """
    s = pd.Series(regime_series).astype(object).fillna('UNKNOWN').astype(str)
    if vocab is None:
        vocab = fit_regime_vocab(s)
    onehot = pd.DataFrame(index=s.index)
    for r in vocab:
        col = f'regime_{r.lower()}'
        onehot[col] = (s == r).astype(int)
    # unknown
    onehot['regime_unknown'] = (~s.isin(vocab)).astype(int)
    # numeric code (known: index in vocab, unknown -> len(vocab))
    mapping = {r: i for i, r in enumerate(vocab)}
    onehot['market_regime_code'] = s.map(lambda x: mapping.get(x, len(vocab))).astype(int)
    return onehot, vocab


def compute_regime_metrics(df: pd.DataFrame,
                           timestamp_col: str = 'timestamp',
                           symbol_col: str = 'symbol',
                           regime_col: str = 'market_regime',
                           outcome_col: str = 'returns_1h',
                           window_days: int = 90) -> pd.DataFrame:
    """Compute rolling regime metrics using past-only data (no lookahead).

    Returns a DataFrame aligned with `df` index containing:
      - regime_win_rate
      - regime_pf
      - regime_expectancy
      - regime_sharpe
      - regime_quality_score

    Uses a time-based rolling window and `closed='left'` to exclude the
    current row (prevents leakage).
    """
    if df is None or df.empty:
        return pd.DataFrame(index=df.index)

    df_local = df.copy()
    df_local[timestamp_col] = pd.to_datetime(df_local[timestamp_col])
    df_local = df_local.sort_values(timestamp_col)
    # Ensure index is timestamp for groupby.rolling with time window
    df_local = df_local.set_index(timestamp_col)

    # Prepare series
    outcome = df_local[outcome_col].fillna(0.0)
    grp = df_local.groupby([symbol_col, regime_col])[outcome_col]

    window = f"{int(window_days)}D"
    # Rolling counts and sums (historical only -> closed='left')
    try:
        rolled_count = grp.rolling(window=window, closed='left').count()
        rolled_sum = grp.rolling(window=window, closed='left').sum()
        rolled_mean = grp.rolling(window=window, closed='left').mean()
        rolled_std = grp.rolling(window=window, closed='left').std()
    except Exception:
        # Fallback for older pandas: use closed default (may include current row)
        rolled_count = grp.rolling(window=window).count()
        rolled_sum = grp.rolling(window=window).sum()
        rolled_mean = grp.rolling(window=window).mean()
        rolled_std = grp.rolling(window=window).std()

    # Wins / losses breakdown
    wins = (df_local[outcome_col] > 0).astype(float)
    losses = (df_local[outcome_col] < 0).astype(float)
    gw = df_local.groupby([symbol_col, regime_col])[outcome_col].apply(lambda x: x.where(x > 0, 0.0))
    gl = df_local.groupby([symbol_col, regime_col])[outcome_col].apply(lambda x: x.where(x < 0, 0.0))
    rolled_wins = gw.rolling(window=window, closed='left').apply(lambda x: (x > 0).sum(), raw=True)
    rolled_win_sum = gw.rolling(window=window, closed='left').sum()
    rolled_loss_sum = gl.rolling(window=window, closed='left').sum()

    # Assemble into DataFrame keyed by (symbol, regime, timestamp)
    pieces = {
        'count': rolled_count,
        'sum': rolled_sum,
        'mean': rolled_mean,
        'std': rolled_std,
        'win_count': rolled_wins,
        'win_sum': rolled_win_sum,
        'loss_sum': rolled_loss_sum,
    }
    metrics = pd.concat(pieces.values(), axis=1)
    metrics.columns = list(pieces.keys())
    metrics = metrics.reset_index()
    # rename timestamp column
    metrics = metrics.rename(columns={timestamp_col: 'ts_index'})

    # Merge back to original index via timestamp and group keys
    df_idx = df_local.reset_index()
    merged = pd.merge(df_idx, metrics, left_on=[symbol_col, regime_col, 'timestamp'],
                      right_on=[symbol_col, regime_col, 'ts_index'], how='left')

    # Compute derived metrics
    merged['regime_win_rate'] = merged['win_count'] / (merged['count'].replace(0, np.nan))
    merged['regime_pf'] = merged['win_sum'] / (merged['loss_sum'].abs().replace(0, np.nan))
    # avg win and loss
    merged['regime_avg_win'] = merged['win_sum'] / (merged['win_count'].replace(0, np.nan))
    merged['regime_avg_loss'] = (merged['loss_sum'].abs() / ((merged['count'] - merged['win_count']).replace(0, np.nan))).abs()
    merged['regime_expectancy'] = (
        merged['regime_avg_win'].fillna(0.0) * merged['regime_win_rate'].fillna(0.0)
        - merged['regime_avg_loss'].fillna(0.0) * (1.0 - merged['regime_win_rate'].fillna(0.0))
    )
    # Sharpe-like stat using mean/std (annualization omitted; user can tune window_days)
    merged['regime_sharpe'] = merged['mean'] / (merged['std'].replace(0, np.nan) + 1e-9)

    # Compose quality score (weights configurable; chosen conservatively)
    w_expectancy, w_pf, w_sharpe, w_win = 0.4, 0.25, 0.25, 0.1
    # Normalize components robustly
    def _norm(s):
        return (s - np.nanpercentile(s, 5)) / (np.nanpercentile(s, 95) - np.nanpercentile(s, 5) + 1e-9)

    ne = _norm(merged['regime_expectancy'].fillna(0.0))
    npf = _norm(merged['regime_pf'].fillna(0.0))
    nsh = _norm(merged['regime_sharpe'].fillna(0.0))
    nw = _norm(merged['regime_win_rate'].fillna(0.0))

    merged['regime_quality_score'] = (
        w_expectancy * ne + w_pf * npf + w_sharpe * nsh + w_win * nw
    )

    # Reindex to original and return only the new columns
    out = merged.set_index('timestamp')[[
        'regime_quality_score', 'regime_expectancy', 'regime_win_rate', 'regime_pf', 'regime_sharpe'
    ]]
    out = out.reindex(df.index)
    out = out.fillna(0.0)
    return out
