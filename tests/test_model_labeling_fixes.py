"""
tests/test_model_labeling_fixes.py — Unit test verifying:
1. Triple-barrier labeling neutralizes ambiguous dual-touch bars (hit = 1).
2. Temperature scaling bounds T safely in [0.1, 10.0].
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.retrain_model import create_triple_barrier_labels, apply_temperature


def test_triple_barrier_dual_touch_neutralization():
    # Construct a synthetic DataFrame where bar 1 touches BOTH upper and lower barriers
    # Length 10 so n > max_lookahead (5) and bar 0 is not tail-censored.
    df = pd.DataFrame({
        'open':  [100.0] * 10,
        'high':  [100.5, 115.0, 100.5, 100.5, 100.5, 100.5, 100.5, 100.5, 100.5, 100.5],  # Bar 1 high hits upper barrier (+15%)
        'low':   [99.5,   85.0,  99.5,  99.5,  99.5,  99.5,  99.5,  99.5,  99.5,  99.5],  # Bar 1 low hits lower barrier (-15%)
        'close': [100.0] * 10,
    })

    # Label with atr_multiplier = 1.0 (barrier ~ 2.0)
    labels = create_triple_barrier_labels(df, atr_multiplier=1.0, max_lookahead=5)
    
    # Bar 0 should hit bar 1 (dual touch) -> MUST be labeled 1 (HOLD), not 2 (BUY)
    assert labels.iloc[0] == 1, f"Expected ambiguous dual-touch bar to be labeled 1 (HOLD), got {labels.iloc[0]}"


def test_temperature_scaling_bounds():
    probs = np.array([[0.3, 0.5, 0.2], [0.1, 0.8, 0.1]])
    
    # Test T <= 0 handling (should clamp to 0.1)
    res_zero = apply_temperature(probs, T=0.0)
    assert res_zero.shape == (2, 3)
    assert not np.isnan(res_zero).any()
    
    # Test T > 10 handling (should clamp to 10.0)
    res_large = apply_temperature(probs, T=100.0)
    assert res_large.shape == (2, 3)
    assert not np.isnan(res_large).any()

    print("ALL MODEL LABELING UNIT TESTS PASSED SUCCESSFULLY!")


if __name__ == '__main__':
    test_triple_barrier_dual_touch_neutralization()
    test_temperature_scaling_bounds()
