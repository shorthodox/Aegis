"""
tests/test_ema_short_horizon.py — Unit test verifying:
1. EMA 21/50 slope momentum and stack alignment feature generation.
2. 12h-18h optimal short-horizon dynamic lookahead calibration.
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
from scripts.retrain_model import get_dynamic_lookahead


def test_ema_short_horizon_features_and_lookahead():
    df = pd.DataFrame({
        'timestamp': pd.date_range('2026-01-01', periods=300, freq='1h'),
        'open':      [100.0] * 300,
        'high':      [105.0] * 300,
        'low':       [95.0]  * 300,
        'close':     [102.0] * 300,
        'volume':    [1000.0] * 300,
    })

    feats = prepare_features(df)
    assert 'ema_21_slope_3' in feats.columns, "ema_21_slope_3 feature must be generated"
    assert 'dist_ema_21_50' in feats.columns, "dist_ema_21_50 feature must be generated"
    assert 'ema_stack_bullish' in feats.columns, "ema_stack_bullish feature must be generated"
    assert not feats['ema_21_slope_3'].isna().any()

    lh = get_dynamic_lookahead(df)
    assert 12 <= lh <= 18, f"Expected short horizon lookahead between 12 and 18, got {lh}"

    print("EMA SHORT HORIZON UNIT TESTS PASSED SUCCESSFULLY!")


if __name__ == '__main__':
    test_ema_short_horizon_features_and_lookahead()
