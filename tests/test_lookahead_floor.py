"""
tests/test_lookahead_floor.py — Unit test verifying:
1. Minimum lookahead floor of 36 hours is enforced in retrain_model.py.
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.retrain_model import get_dynamic_lookahead


def test_dynamic_lookahead_floor():
    df = pd.DataFrame({
        'open':  [100.0] * 100,
        'high':  [100.5] * 100,
        'low':   [99.5]  * 100,
        'close': [100.0] * 100,
    })

    lh = get_dynamic_lookahead(df)
    assert 12 <= lh <= 18, f"Expected lookahead to be between 12 and 18, got {lh}"
    print("LOOKAHEAD FLOOR UNIT TESTS PASSED SUCCESSFULLY!")


if __name__ == '__main__':
    test_dynamic_lookahead_floor()
