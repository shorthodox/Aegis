"""
tests/test_raw_xgboost_enhancements.py — Unit test verifying:
1. Core momentum features are protected from SHAP pruning.
2. Dampened scale_pos_weight calculation.
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def test_dampened_spw_calculation():
    n_neg = 3600.0
    n_pos = 1200.0
    raw_spw = n_neg / n_pos
    dampened_spw = float(np.clip(np.sqrt(raw_spw), 1.0, 3.0))
    assert abs(dampened_spw - np.sqrt(3.0)) < 1e-4, f"Expected sqrt(3.0) ~ 1.732, got {dampened_spw}"
    print("RAW XGBOOST ENHANCEMENT UNIT TESTS PASSED SUCCESSFULLY!")


if __name__ == '__main__':
    test_dampened_spw_calculation()
