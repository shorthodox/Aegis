"""
Validation utilities for walk-forward and robustness testing.
Provides:
- compute_fold_metrics(trades, equity_series)
- aggregate_fold_metrics(list_of_metrics)
- bootstrap_trades(all_trades_df, B=5000)
- monte_carlo_shuffle(all_trades_df, R=2000)
- leave_one_fold_out_check(fold_trade_dfs)
- compute_robustness_score(aggregated_metrics, params)

Data assumptions:
- trades: pandas DataFrame with columns ['pnl', 'timestamp', 'side', 'regime']
- equity_series: pandas Series indexed by timestamp with cumulative PnL

These implementations prioritize clarity and correctness; parameters are configurable.
"""
from __future__ import annotations

import math
import json
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

EPS = 1e-9

# ===================== NEW: Validation diagnostics (FIX CRITICAL) =======================

def _validate_fold_inputs(fold_trade_dfs: List[pd.DataFrame]) -> Dict[str, Any]:
    """Pre-flight checks to prevent silent concatenation failures.
    
    Returns:
        Dict with 'status', 'reasons', and 'can_proceed' keys.
        If status is 'validation_failed', can_proceed will be False.
    """
    issues = []
    
    if not fold_trade_dfs:
        issues.append("No folds provided (empty list)")
    else:
        total_trades = sum(len(df) for df in fold_trade_dfs)
        if total_trades == 0:
            issues.append("All folds are empty (0 total trades)")
        
        if len(fold_trade_dfs) == 1:
            issues.append("Only 1 fold provided (LOFO cannot be computed)")
        
        required_cols = {'pnl'}
        for i, df in enumerate(fold_trade_dfs):
            if isinstance(df, pd.DataFrame):
                missing = required_cols - set(df.columns)
                if missing:
                    issues.append(f"Fold {i} missing columns: {missing}")
            else:
                issues.append(f"Fold {i} is not a DataFrame (type: {type(df).__name__})")
    
    if issues:
        return {
            'status': 'validation_failed',
            'reasons': issues,
            'can_proceed': False
        }
    else:
        return {
            'status': 'validation_passed',
            'reasons': [],
            'can_proceed': True
        }

# ========================================================================================



def compute_profit_factor(trades: pd.DataFrame) -> float:
    wins = trades.loc[trades['pnl'] > 0, 'pnl'].sum()
    losses = trades.loc[trades['pnl'] < 0, 'pnl'].sum()
    losses_abs = abs(losses)
    if losses_abs <= 0:
        return float('inf') if wins > 0 else 0.0
    return float(wins / losses_abs)


def compute_expectancy_pct(trades: pd.DataFrame) -> float:
    # Expectancy normalized by average absolute move per trade, expressed as percent
    # expectancy_pct = 100 * mean(pnl) / mean(abs(pnl))
    if len(trades) == 0:
        return 0.0
    mean_pnl = trades['pnl'].mean()
    mean_abs = trades['pnl'].abs().mean() + EPS
    return float(100.0 * mean_pnl / mean_abs)


def compute_sharpe(trades: pd.DataFrame) -> float:
    # Simple per-trade Sharpe: mean(pnl)/std(pnl) * sqrt(N)
    # If pnl is raw returns, caller can adapt. This approximates Sharpe-like behaviour.
    if len(trades) < 2:
        return 0.0
    mean_pnl = trades['pnl'].mean()
    std_pnl = trades['pnl'].std(ddof=0) + EPS
    sharpe = mean_pnl / std_pnl * math.sqrt(len(trades))
    return float(sharpe)


def compute_precision(trades: pd.DataFrame) -> float:
    if len(trades) == 0:
        return 0.0
    wins = (trades['pnl'] > 0).sum()
    total = len(trades)
    return float(wins / total)


# ---------------------------- Equity & drawdown -------------------------------

def compute_equity_series(trades: pd.DataFrame, start_balance: float = 0.0) -> pd.Series:
    # trades expected to have 'timestamp' and 'pnl'; returns cumulative series indexed by timestamp
    if 'timestamp' not in trades.columns:
        # fallback: index order
        cum = trades['pnl'].cumsum() + start_balance
        return pd.Series(cum.values, index=np.arange(len(cum)))
    df = trades.sort_values('timestamp')
    cum = (df['pnl'].cumsum() + start_balance)
    return pd.Series(cum.values, index=pd.to_datetime(df['timestamp']))


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdowns = (running_max - equity) / running_max
    dd = drawdowns.max()
    return float(dd * 100.0)  # percent


def ulcer_index(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown_pct = ((running_max - equity) / running_max) * 100.0
    ui = math.sqrt((drawdown_pct ** 2).mean())
    return float(ui)


def recovery_factor(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    cumret = float((equity.iloc[-1] - equity.iloc[0]) if len(equity) > 1 else 0.0)
    dd = max_drawdown(equity) / 100.0
    if dd <= 0:
        return float('inf') if cumret > 0 else 0.0
    return float(cumret / dd)


# ---------------------------- Fold-level metrics ------------------------------

def compute_fold_metrics(trades: pd.DataFrame, equity: pd.Series = None) -> Dict[str, Any]:
    if trades is None:
        trades = pd.DataFrame(columns=['pnl', 'timestamp', 'side', 'regime'])

    metrics: Dict[str, Any] = {}
    metrics['n_trades'] = int(len(trades))
    metrics['profit_factor'] = compute_profit_factor(trades)
    metrics['expectancy_pct'] = compute_expectancy_pct(trades)
    metrics['sharpe'] = compute_sharpe(trades)
    metrics['precision'] = compute_precision(trades)
    # coverage might be computed at fold level externally

    if equity is None:
        equity = compute_equity_series(trades)

    metrics['cum_return'] = float(equity.iloc[-1] - equity.iloc[0]) if len(equity) > 0 else 0.0
    metrics['max_drawdown_pct'] = max_drawdown(equity)
    metrics['ulcer_index'] = ulcer_index(equity)
    metrics['recovery_factor'] = recovery_factor(equity)

    # simple mean and std for pnl
    metrics['pnl_mean'] = float(trades['pnl'].mean()) if len(trades) else 0.0
    metrics['pnl_std'] = float(trades['pnl'].std(ddof=0)) if len(trades) else 0.0

    return metrics


# ---------------------------- Aggregation ------------------------------------

def aggregate_fold_metrics(fold_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    # fold_metrics: list of dicts returned by compute_fold_metrics
    df = pd.DataFrame(fold_metrics)
    if df.empty:
        return {}

    agg: Dict[str, Any] = {}
    for col in ['profit_factor', 'expectancy_pct', 'sharpe', 'precision', 'cum_return']:
        vals = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        agg[f'{col}_mean'] = float(vals.mean()) if len(vals) else 0.0
        agg[f'{col}_std'] = float(vals.std(ddof=0)) if len(vals) > 1 else 0.0
        agg[f'{col}_cv'] = float(agg[f'{col}_std'] / (abs(agg[f'{col}_mean']) + EPS))
        agg[f'{col}_stability'] = float(1.0 / (1.0 + agg[f'{col}_cv']))

    # drawdown aggregates
    dd_vals = df['max_drawdown_pct'].dropna()
    agg['max_drawdown_pct'] = float(dd_vals.max()) if len(dd_vals) else 0.0
    agg['ulcer_index_mean'] = float(df['ulcer_index'].mean()) if 'ulcer_index' in df else 0.0
    agg['recovery_factor_mean'] = float(df['recovery_factor'].mean()) if 'recovery_factor' in df else 0.0

    agg['n_folds'] = int(len(df))
    agg['total_trades'] = int(df['n_trades'].sum())

    return agg


# ---------------------------- Bootstrap & Monte Carlo ------------------------

def bootstrap_trades(all_trades: pd.DataFrame, B: int = 2000, seed: int = 42) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(all_trades)
    if n == 0:
        return {'pf': np.array([]), 'exp': np.array([]), 'sharpe': np.array([])}

    pnl = all_trades['pnl'].values
    pf_dist = np.empty(B)
    exp_dist = np.empty(B)
    sharpe_dist = np.empty(B)

    for b in range(B):
        idx = rng.integers(0, n, n)
        sample = pd.DataFrame({'pnl': pnl[idx]})
        pf_dist[b] = compute_profit_factor(sample)
        exp_dist[b] = compute_expectancy_pct(sample)
        sharpe_dist[b] = compute_sharpe(sample)

    return {'pf': pf_dist, 'exp': exp_dist, 'sharpe': sharpe_dist}


def monte_carlo_shuffle(all_trades: pd.DataFrame, R: int = 1000, seed: int = 42) -> Dict[str, Any]:
    # Shuffle pnl values across timestamps (breaks time-dependence)
    rng = np.random.default_rng(seed)
    n = len(all_trades)
    if n == 0:
        return {'pf': np.array([]), 'exp': np.array([])}

    pnl = all_trades['pnl'].values
    pf_perm = np.empty(R)
    exp_perm = np.empty(R)

    for r in range(R):
        perm = pnl.copy()
        rng.shuffle(perm)
        sample = pd.DataFrame({'pnl': perm})
        pf_perm[r] = compute_profit_factor(sample)
        exp_perm[r] = compute_expectancy_pct(sample)

    return {'pf_perm': pf_perm, 'exp_perm': exp_perm}


# ---------------------------- Leave-one-fold-out & concentration -------------

def leave_one_fold_out_check(fold_trade_dfs: List[pd.DataFrame]) -> Dict[str, Any]:
    # For each exclusion compute aggregated PF and expectancy
    total_cum = sum((df['pnl'].sum() for df in fold_trade_dfs))
    contributions = [(df['pnl'].sum() / (total_cum + EPS)) if total_cum != 0 else 0.0 for df in fold_trade_dfs]
    max_contrib = max(contributions) if contributions else 0.0

    # LOFO impact
    pf_list = []
    exp_list = []
    for i in range(len(fold_trade_dfs)):
        fold_indices = [j for j in range(len(fold_trade_dfs)) if j != i]
        
        # FIX (CRITICAL): Check if list is empty before concatenating
        # This can happen if only 1 fold exists, causing "No objects to concatenate" error
        if not fold_indices:
            print(f"[WARNING] Cannot compute LOFO for fold {i}: only 1 fold exists")
            pf_list.append(np.nan)
            exp_list.append(np.nan)
            continue
        
        fold_to_concat = [fold_trade_dfs[j] for j in fold_indices]
        pool = pd.concat(fold_to_concat, ignore_index=True)  # Now safe!
        pf_list.append(compute_profit_factor(pool))
        exp_list.append(compute_expectancy_pct(pool))

    pf_array = np.array(pf_list)
    exp_array = np.array(exp_list)

    # if removing any fold causes PF<=1 or Exp<=0 -> flag
    pf_at_risk = bool(np.any(pf_array <= 1.0))
    exp_at_risk = bool(np.any(exp_array <= 0.0))

    return {
        'max_contribution': float(max_contrib),
        'pf_lofo': pf_array,
        'exp_lofo': exp_array,
        'pf_at_risk': pf_at_risk,
        'exp_at_risk': exp_at_risk,
    }


# ---------------------------- Robustness score -------------------------------

def compute_robustness_score(aggregated: Dict[str, Any], params: Dict[str, Any] = None) -> Dict[str, Any]:
    # Default weights and thresholds
    if params is None:
        params = {}
    W1 = params.get('W1', 0.20)
    W2 = params.get('W2', 0.18)
    W3 = params.get('W3', 0.15)
    W4 = params.get('W4', 0.12)
    W5 = params.get('W5', 0.08)
    W6 = params.get('W6', 0.05)
    W7 = params.get('W7', 0.08)
    W8 = params.get('W8', 0.04)
    W9 = params.get('W9', 0.05)
    P1 = params.get('P1', 0.15)
    P2 = params.get('P2', 0.10)

    # Normalizations
    pf_bar = aggregated.get('profit_factor_mean', 0.0)
    exp_bar = aggregated.get('expectancy_pct_mean', 0.0)
    sharpe_bar = aggregated.get('sharpe_mean', 0.0)

    nPF = float(np.clip((pf_bar - 1.0) / (5.0 - 1.0), 0.0, 1.0))
    nExp = float(np.clip(exp_bar / 5.0, 0.0, 1.0))
    nSharpe = float(np.clip(sharpe_bar / 3.0, 0.0, 1.0))

    stabPF = aggregated.get('profit_factor_stability', 0.0)
    stabExp = aggregated.get('expectancy_pct_stability', 0.0)
    stabPrec = aggregated.get('precision_stability', 0.0)

    dd_score = 1.0 - float(np.clip(aggregated.get('max_drawdown_pct', 0.0) / 50.0, 0.0, 1.0))
    ui_score = 1.0 - float(np.clip(aggregated.get('ulcer_index_mean', 0.0) / 50.0, 0.0, 1.0))

    # recovery mapping
    recov = aggregated.get('recovery_factor_mean', 0.0)
    recov_scaled = 1.0 - math.exp(-recov / 1.0) if recov > 0 else 0.0

    regime_penalty = aggregated.get('regime_vol', 0.0)  # expect precomputed
    sample_penalty = aggregated.get('sample_penalty', 0.0)

    raw_score = (W1 * nPF + W2 * nExp + W3 * nSharpe + W4 * stabPF + W5 * stabExp + W6 * stabPrec
                 + W7 * dd_score + W8 * ui_score + W9 * recov_scaled)

    penalized = raw_score - P1 * regime_penalty - P2 * sample_penalty
    R = float(np.clip(penalized, 0.0, 1.0))

    return {
        'raw_score': raw_score,
        'penalized_score': penalized,
        'robustness': R,
        'components': {
            'nPF': nPF, 'nExp': nExp, 'nSharpe': nSharpe,
            'stabPF': stabPF, 'stabExp': stabExp, 'stabPrec': stabPrec,
            'dd_score': dd_score, 'ui_score': ui_score, 'recov_scaled': recov_scaled,
            'regime_penalty': regime_penalty, 'sample_penalty': sample_penalty,
        }
    }


# ---------------------------- High-level runner ---------------------------------

def validate_architecture_from_folds(fold_trade_dfs: List[pd.DataFrame], regimes: List[str] = None,
                                      bootstrap_B: int = 2000, mc_R: int = 1000) -> Dict[str, Any]:
    """
    High-level orchestration: given per-fold trade DataFrames, compute fold metrics,
    aggregate, run bootstrap + MC, LOFO, compute robustness score and returns report.
    Each fold df must include columns: 'pnl', 'timestamp', 'side', 'regime'
    """
    preflight = _validate_fold_inputs(fold_trade_dfs)
    if not preflight['can_proceed']:
        return {
            'status': 'validation_failed',
            'reasons': preflight['reasons'],
            'validation_mode': 'preflight_failed',
            'fold_metrics': [],
            'aggregated': {},
            'bootstrap': {},
            'monte_carlo': {},
            'lofo': {},
            'robustness': 0.0,
        }

    single_fold = len(fold_trade_dfs) == 1
    fold_metrics = []
    all_trades = []
    for df in fold_trade_dfs:
        eq = compute_equity_series(df)
        m = compute_fold_metrics(df, eq)
        fold_metrics.append(m)
        all_trades.append(df)

    aggregated = aggregate_fold_metrics(fold_metrics)

    # regime volatility: compute per-regime expectancy if regimes provided
    if regimes is not None and len(regimes) > 0:
        reg_expectancies = []
        total_trades = 0
        regime_failures = []
        
        for r in regimes:
            # FIX (CRITICAL): Filter and check for empty results before concatenating
            # This prevents "No objects to concatenate" error when regime matches nothing
            regime_dfs = [df.loc[df['regime'] == r, :] for df in fold_trade_dfs]
            
            # Only concatenate non-empty DataFrames
            non_empty = [df for df in regime_dfs if len(df) > 0]
            if non_empty:
                pool = pd.concat(non_empty, ignore_index=True)
                reg_expectancies.append(compute_expectancy_pct(pool))
                total_trades += len(pool)
            else:
                print(f"[WARNING] No trades found for regime '{r}' across all folds")
                regime_failures.append(r)
                reg_expectancies.append(np.nan)
        
        if len(reg_expectancies) > 1:
            # Filter out NaN values for std/mean calculation
            valid_exp = [x for x in reg_expectancies if not np.isnan(x)]
            if valid_exp:
                regime_vol = float(np.std(valid_exp) / (abs(np.mean(valid_exp)) + EPS))
            else:
                regime_vol = 0.0
        else:
            regime_vol = 0.0
    else:
        regime_vol = 0.0

    aggregated['regime_vol'] = regime_vol
    aggregated['sample_penalty'] = float(max(0.0, (200 - aggregated.get('total_trades', 0)) / 200.0))

    # Bootstrap
    combined = pd.concat(fold_trade_dfs, ignore_index=True) if len(fold_trade_dfs) else pd.DataFrame(columns=['pnl'])
    
    # FIX (CRITICAL): Defensive check to ensure concatenation result is not empty
    # This prevents crashes in downstream bootstrap/MC operations
    if combined.empty:
        print("[WARNING] Combined trades DataFrame is empty after concatenation")
        return {
            'status': 'validation_failed',
            'reason': 'no_trades_after_concatenation',
            'robustness': 0.0,
            'components': {},
            'aggregated_metrics': {},
        }
    
    boot = bootstrap_trades(combined, B=bootstrap_B)
    pf_boot = boot['pf']
    exp_boot = boot['exp']

    ci_pf = (float(np.percentile(pf_boot, 2.5)), float(np.percentile(pf_boot, 97.5))) if len(pf_boot) else (0.0, 0.0)
    ci_exp = (float(np.percentile(exp_boot, 2.5)), float(np.percentile(exp_boot, 97.5))) if len(exp_boot) else (0.0, 0.0)
    p_pf_gt1 = float((pf_boot > 1.0).mean()) if len(pf_boot) else 0.0
    p_exp_gt0 = float((exp_boot > 0.0).mean()) if len(exp_boot) else 0.0

    # Monte Carlo
    mc = monte_carlo_shuffle(combined, R=mc_R)
    mc_pf = mc['pf_perm']
    mc_exp = mc['exp_perm']
    p_mc_pf_ge_obs = float((mc_pf >= aggregated.get('profit_factor_mean', 0.0)).mean()) if len(mc_pf) else 1.0

    # LOFO
    if single_fold:
        lofo = {
            'status': 'single_fold_validation_skipped',
            'reason': 'LOFO validation is not meaningful with a single fold',
            'pf_lofo': np.array([np.nan]),
            'exp_lofo': np.array([np.nan]),
            'pf_at_risk': False,
            'exp_at_risk': False,
            'max_contribution': 1.0,
        }
    else:
        lofo = leave_one_fold_out_check(fold_trade_dfs)

    # Robustness
    robustness = compute_robustness_score(aggregated)

    report = {
        'fold_metrics': fold_metrics,
        'aggregated': aggregated,
        'bootstrap': {
            'pf_ci': ci_pf,
            'exp_ci': ci_exp,
            'p_pf_gt1': p_pf_gt1,
            'p_exp_gt0': p_exp_gt0,
        },
        'monte_carlo': {
            'p_mc_pf_ge_obs': p_mc_pf_ge_obs,
        },
        'lofo': lofo,
        'robustness': robustness,
    }
    return report


if __name__ == '__main__':
    print('validation.py loaded. Use validate_architecture_from_folds() in scripts.')
