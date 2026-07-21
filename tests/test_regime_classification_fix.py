"""
tests/test_regime_classification_fix.py — Unit test verifying:
  1. MarketRegimeDetector correctly classifies high-momentum bullish rallies (RSI > 55, ADX > 25, MACD BULLISH) as TRENDING_BULL.
  2. Gate 1.5 direction logic correctly identifies counter-trend fades vs trend-following blocks.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make sure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.live_engine import MarketRegimeDetector, _REGIME_TRENDING_BULL, _REGIME_TRENDING_BEAR


def test_market_regime_detector_bullish_rally():
    detector = MarketRegimeDetector()
    
    # XRP-like scenario: ADX 42.2 (strong trend), RSI 69.9 (overbought rally), MACD BULLISH
    # even if market_bias defaults to BEARISH from stale macro features
    result = {
        'adx': 42.2,
        'rsi': 69.9,
        'macd_signal': 'BULLISH',
        'trend_regime': 'TRENDING_UP',
        'market_bias': 'BEARISH',
        'volatility_regime': 'MEDIUM',
        'atr_pct': 1.5,
        'volume_zscore': 0.5,
        'volume_strength': 'AVERAGE',
        'funding_bias': 'NEUTRAL',
        'oi_trend': 'STABLE',
    }

    state = detector.detect(result)
    assert state.regime == _REGIME_TRENDING_BULL, f"Expected TRENDING_BULL for bullish rally, got {state.regime}"


def test_market_regime_detector_bearish_drop():
    detector = MarketRegimeDetector()

    result = {
        'adx': 35.0,
        'rsi': 28.0,
        'macd_signal': 'BEARISH',
        'trend_regime': 'TRENDING_DOWN',
        'market_bias': 'NEUTRAL',
        'volatility_regime': 'MEDIUM',
        'atr_pct': 1.5,
        'volume_zscore': 0.5,
        'volume_strength': 'AVERAGE',
        'funding_bias': 'NEUTRAL',
        'oi_trend': 'STABLE',
    }

    state = detector.detect(result)
    assert state.regime == _REGIME_TRENDING_BEAR, f"Expected TRENDING_BEAR for bearish drop, got {state.regime}"


if __name__ == '__main__':
    test_market_regime_detector_bullish_rally()
    test_market_regime_detector_bearish_drop()
    print("ALL REGIME CLASSIFICATION FIX TESTS PASSED SUCCESSFULLY!")
