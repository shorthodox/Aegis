#!/usr/bin/env python3
"""
walk_forward_runner.py — simple walk-forward harness for meta gate optimizer

Runs multiple rolling train/eval splits and records per-fold candidate metrics
for stability analysis. Outputs a JSON summary per symbol in data/meta_gate_profiles.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from scripts.meta_gate_optimizer import (
    _fit_local_model,
    _evaluate_architecture,
    PROFILE_DIR,
)
from src.ml.predictor import Predictor
from src.ml.feature_engine import prepare_features

PROFILE_DIR.mkdir(parents=True, exist_ok=True)


def run_walk_forward(symbol: str, n_splits: int = 5) -> Dict[str, Any]:
    p = Predictor(symbol)
    df_1h = p.fetch_live_data(timeframe='1h', limit=4000)
    if df_1h is None or len(df_1h) < 1000:
        raise RuntimeError('Not enough data for walk-forward')

    features = prepare_features(df_1h)
    n = len(features)
    fold_size = int(n / (n_splits + 1))

    results: List[Dict[str, Any]] = []

    for k in range(n_splits):
        train_end = fold_size * (k + 1)
        eval_start = train_end
        eval_end = min(eval_start + fold_size, n)
        if eval_end - eval_start < 50:
            break

        feat_tr = features.iloc[:train_end].reset_index(drop=True)
        feat_ev = features.iloc[eval_start:eval_end].reset_index(drop=True)

        from scripts.retrain_model import create_triple_barrier_labels
        df_full = p.fetch_live_data(timeframe='1h', limit=eval_end)
        if df_full is None:
            results.append({'fold': k, 'error': 'no data'})
            continue
        df_tr = df_full.iloc[:train_end].reset_index(drop=True)
        df_ev_slice = df_full.iloc[train_end:eval_end].reset_index(drop=True)

        labels_tr = np.asarray(create_triple_barrier_labels(df_tr, p.meta.get('atr_multiplier', 1.0)), dtype=np.intp)
        labels_ev = np.asarray(create_triple_barrier_labels(df_ev_slice, p.meta.get('atr_multiplier', 1.0)), dtype=np.intp)

        train_n = len(feat_tr)
        try:
            proposed_all, dir_conf_all, probs_all, _ = _fit_local_model(
                pd.concat([feat_tr, feat_ev], axis=0).reset_index(drop=True),
                np.concatenate([labels_tr, labels_ev]),
                train_n,
            )
        except Exception as exc:
            results.append({'fold': k, 'error': str(exc)})
            continue

        try:
            res = _evaluate_architecture(
                symbol, feat_ev, pd.Series(),
                proposed_all[train_n:], dir_conf_all[train_n:], probs_all[train_n:],
                labels_ev, np.full(len(feat_ev), 1.0),
                train_edge_scores=np.array([]), train_correct=np.array([]),
            )
            results.append({'fold': k, 'result': res})
        except Exception as exc:
            results.append({'fold': k, 'error': str(exc)})

    out = PROFILE_DIR / f"{symbol.replace('/', '_')}_walkforward.json"
    with open(out, 'w') as fh:
        json.dump({'symbol': symbol, 'generated_at': datetime.now(timezone.utc).isoformat(), 'folds': results}, fh, default=str, indent=2)
    return {'symbol': symbol, 'n_folds': len(results), 'out': str(out)}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('symbol')
    parser.add_argument('--splits', type=int, default=5)
    args = parser.parse_args()
    print(run_walk_forward(args.symbol, n_splits=args.splits))
