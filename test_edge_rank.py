#!/usr/bin/env python3
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import numpy as np
from src.trading.edge_engine import EdgeScoringEngine
from scripts.meta_gate_optimizer import compute_edge_scores

# minimal DataFrame for edge rank test
df = pd.DataFrame({
    'adx': [10, 20, 30, 40, 15],
    'volume_zscore': [0, 1, 2, -1, 0.5],
    'rsi_14': [50, 60, 30, 45, 70],
    'macd_hist': [0.01, -0.02, 0.03, 0.0, 0.02],
    'relative_volume': [1.0, 1.5, 0.5, 1.2, 2.0],
    'oi_zscore': [0.0, 0.5, -0.5, 0.1, 0.2],
    'cmf_20': [0.0, 0.1, -0.1, 0.05, 0.2],
    'oi_price_divergence': [0.01, -0.02, 0.0, 0.03, -0.01],
    'atr_pct': [0.015, 0.02, 0.01, 0.025, 0.018],
    'macro_trend_1d': [0.1, -0.2, 0.05, 0.3, -0.1],
    'hmm_regime': ['TRENDING_BULL','CHOPPY','VOLATILE_EXPANSION','RANGING','ACCUMULATION'],
    'volatility_regime': ['HIGH_VOL','LOW_VOL','MEDIUM_VOL','COMPRESSION','LOW_VOL'],
})

probs = np.array([0.6, 0.7, 0.2, 0.8, 0.4])

# Test compute_edge_rank_batch
print("Testing compute_edge_rank_batch...")
res = EdgeScoringEngine.compute_edge_rank_batch(df, probs, 'BUY')
print("Result columns:", res.columns.tolist())
print("edge_percentile_100:", res['edge_percentile_100'].tolist())
print()

# Test compute_edge_scores with use_rank=True
print("Testing compute_edge_scores with use_rank=True...")
proposed = np.array([2, 2, 0, 2, 0])
scores = compute_edge_scores(df, proposed, probs, use_rank=True)
print("Scores:", scores.tolist())
print()

print("✓ All smoke tests passed!")
