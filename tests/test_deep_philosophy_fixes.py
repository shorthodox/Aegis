"""
tests/test_deep_philosophy_fixes.py — Unit test verifying:
1. Volume absorption (rel_vol_24h) and Liquidity Sweep (sweep_wick_ratio) feature generation.
2. TRAP regime exclusion during triple-barrier training label generation.
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.ml.feature_engine import prepare_features
from scripts.retrain_model import create_triple_barrier_labels


def test_volume_and_sweep_feature_generation():
    df = pd.DataFrame({
        'timestamp': pd.date_range('2026-01-01', periods=300, freq='1h'),
        'open':      [100.0] * 300,
        'high':      [105.0] * 300,
        'low':       [95.0]  * 300,
        'close':     [102.0] * 300,
        'volume':    [1000.0] * 300,
    })

    feats = prepare_features(df)
    assert 'rel_vol_24h' in feats.columns, "rel_vol_24h feature must be generated"
    assert 'sweep_wick_ratio' in feats.columns, "sweep_wick_ratio feature must be generated"
    assert not feats['rel_vol_24h'].isna().any()
    assert not feats['sweep_wick_ratio'].isna().any()


def test_trap_regime_exclusion():
    df = pd.DataFrame({
        'open':           [100.0] * 10,
        'high':           [115.0] * 10,
        'low':            [99.5]  * 10,
        'close':          [100.0] * 10,
        'range_position': [0.20]  * 10,
        'market_regime':  ['TRAP'] * 10,
    })

    labels = create_triple_barrier_labels(df, atr_multiplier=1.0, max_lookahead=5)
    # TRAP regime bars MUST all evaluate to 1 (HOLD)
    assert (labels.iloc[:-5] == 1).all(), f"TRAP regime bars must be excluded (HOLD), got {labels.iloc[0]}"

    print("DEEP PHILOSOPHY FIXES UNIT TESTS PASSED SUCCESSFULLY!")


if __name__ == '__main__':
    test_volume_and_sweep_feature_generation()
    test_trap_regime_exclusion()
