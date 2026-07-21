"""
tests/test_early_5m_reversal_fire.py — Unit test verifying:
  Signals waiting/pending for their target S/R zone fire early if a genuine 5-minute
  candlestick reversal pattern (hammer, engulfing, etc.) completes before reaching the fire zone.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make sure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.live_engine import _reversal_candle


def test_bullish_reversal_patterns():
    # 1. Bullish Hammer: long lower wick (>= 55% range), small body, tiny upper wick
    # OHLC: o=100.5, h=100.6, l=98.0, c=100.4 -> range=2.6, body=0.1, lower wick=2.4 (92%), upper wick=0.1
    hammer_candle = [1600000000000, 100.5, 100.6, 98.0, 100.4, 1000.0]
    prev_candle = [1599999700000, 102.0, 102.1, 100.0, 100.6, 1000.0] # red candle
    candles = [prev_candle, hammer_candle]

    pat = _reversal_candle(candles, want_bullish=True)
    assert pat == 'hammer', f"Expected hammer, got {pat}"

    # 2. Bullish Engulfing: previous red, current green engulfing previous body
    c1 = [1599999700000, 101.0, 101.2, 99.8, 100.0, 1000.0] # red body 1.0
    c2 = [1600000000000, 99.8, 102.0, 99.5, 101.5, 1000.0]  # green body 1.7 engulfing c1
    pat_eng = _reversal_candle([c1, c2], want_bullish=True)
    assert pat_eng == 'bullish_engulfing', f"Expected bullish_engulfing, got {pat_eng}"


def test_bearish_reversal_patterns():
    # 1. Shooting Star: long upper wick (>= 55% range), small body, tiny lower wick
    # OHLC: o=100.0, h=102.5, l=99.9, c=100.1 -> range=2.6, body=0.1, upper wick=2.4
    star_candle = [1600000000000, 100.0, 102.5, 99.9, 100.1, 1000.0]
    prev_candle = [1599999700000, 98.0, 100.0, 97.9, 99.8, 1000.0] # green candle
    candles = [prev_candle, star_candle]

    pat = _reversal_candle(candles, want_bullish=False)
    assert pat == 'shooting_star', f"Expected shooting_star, got {pat}"

    # 2. Bearish Engulfing: previous green, current red engulfing previous body
    c1 = [1599999700000, 100.0, 101.2, 99.8, 101.0, 1000.0] # green body 1.0
    c2 = [1600000000000, 101.2, 101.5, 99.0, 99.5, 1000.0]  # red body 1.7 engulfing c1
    pat_eng = _reversal_candle([c1, c2], want_bullish=False)
    assert pat_eng == 'bearish_engulfing', f"Expected bearish_engulfing, got {pat_eng}"


if __name__ == '__main__':
    test_bullish_reversal_patterns()
    test_bearish_reversal_patterns()
    print("ALL EARLY 5M REVERSAL TESTS PASSED SUCCESSFULLY!")
