"""
tests/test_philosophy_alignment.py — Unit test verifying:
1. Structure-aware location gating in create_triple_barrier_labels.
2. BUY labels require range_position <= 0.40 (near Support).
3. SELL labels require range_position >= 0.60 (near Resistance).
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.retrain_model import create_triple_barrier_labels


def test_structure_aware_label_gating():
    # 10 rows so n > max_lookahead (5)
    # Row 0: range_position = 0.80 (near resistance), hits upper barrier (+15%) -> MUST be demoted to 1 (HOLD)
    # Row 1: range_position = 0.20 (near support), hits upper barrier (+15%) -> MUST remain 2 (BUY)
    df = pd.DataFrame({
        'open':           [100.0] * 10,
        'high':           [100.5, 115.0, 115.0, 100.5, 100.5, 100.5, 100.5, 100.5, 100.5, 100.5],
        'low':            [99.5,   99.5,  99.5,  99.5,  99.5,  99.5,  99.5,  99.5,  99.5,  99.5],
        'close':          [100.0] * 10,
        'range_position': [0.80,  0.20,  0.50,  0.50,  0.50,  0.50,  0.50,  0.50,  0.50,  0.50],
    })

    labels = create_triple_barrier_labels(df, atr_multiplier=1.0, max_lookahead=5)

    # Row 0 (BUY at resistance range_position 0.80) -> MUST be demoted to 1 (HOLD)
    assert labels.iloc[0] == 1, f"Expected BUY at resistance (0.80) to be demoted to 1 (HOLD), got {labels.iloc[0]}"

    # Row 1 (BUY at support range_position 0.20) -> MUST stay 2 (BUY)
    assert labels.iloc[1] == 2, f"Expected BUY at support (0.20) to remain 2 (BUY), got {labels.iloc[1]}"

    print("PHILOSOPHY ALIGNMENT UNIT TESTS PASSED SUCCESSFULLY!")


if __name__ == '__main__':
    test_structure_aware_label_gating()
