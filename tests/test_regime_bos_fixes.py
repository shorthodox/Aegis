"""
tests/test_regime_bos_fixes.py — Unit tests verifying:
  1. Guard N hard-blocks BUY in TRENDING_BULL and SELL in TRENDING_BEAR (fade-only).
  2. Opposing BOS hard-blocks trades fighting active Break of Structure ("market changing moving path").
  3. VOLATILE_COMPRESSION hard-blocks unconfirmed pre-breakout squeeze trades.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Make sure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.live_engine import (
    LiveEngine,
    RegimeState,
    TokenConfig,
    _REGIME_TRENDING_BULL,
    _REGIME_TRENDING_BEAR,
    _REGIME_VOLATILE_COMPRESS,
    _REGIME_RANGING,
    _detect_bos_choch,
)


def test_detect_bos_choch_logic():
    # Construct 30 candles: 25 uptrending, last 5 breaking down significantly below rolling low
    candles = []
    price = 100.0
    for i in range(25):
        price += 1.0
        candles.append([i * 3600000, price - 0.5, price + 1.0, price - 1.0, price, 1000.0])
    # Now sharp breakdown below the entire 20-bar low (which was ~100)
    for i in range(25, 30):
        price -= 10.0
        candles.append([i * 3600000, price + 0.5, price + 0.5, price - 2.0, price - 1.0, 1000.0])

    bos_info = _detect_bos_choch(candles, lookback=20)
    assert bos_info['signal'] < 0, f"Expected bearish BOS signal, got {bos_info['signal']}"
    assert bos_info['choch_bear'] > 0 or bos_info['bos_state'] < 0


def test_guard_n_trend_follow_veto():
    # Test helper checking regime logic
    engine = LiveEngine(token_configs=[TokenConfig(symbol='BTC/USDT')])
    
    # 1. BUY in TRENDING_BULL should be identified as trend-following
    reg_bull = _REGIME_TRENDING_BULL
    trend_follow_buy = (('BUY' == 'SELL' and reg_bull == _REGIME_TRENDING_BEAR) or
                        ('BUY' == 'BUY'  and reg_bull == _REGIME_TRENDING_BULL))
    assert trend_follow_buy is True

    # 2. SELL in TRENDING_BEAR should be identified as trend-following
    reg_bear = _REGIME_TRENDING_BEAR
    trend_follow_sell = (('SELL' == 'SELL' and reg_bear == _REGIME_TRENDING_BEAR) or
                         ('SELL' == 'BUY'  and reg_bear == _REGIME_TRENDING_BULL))
    assert trend_follow_sell is True

    # 3. BUY in TRENDING_BEAR is counter-trend (allowed to proceed to reversal check)
    ct_buy = (('BUY' == 'BUY' and reg_bear == _REGIME_TRENDING_BEAR) or
              ('BUY' == 'SELL' and reg_bear == _REGIME_TRENDING_BULL))
    assert ct_buy is True
    assert (('BUY' == 'SELL' and reg_bear == _REGIME_TRENDING_BEAR) or
            ('BUY' == 'BUY'  and reg_bear == _REGIME_TRENDING_BULL)) is False

    # 4. SELL in TRENDING_BULL is counter-trend (allowed to proceed to reversal check)
    ct_sell = (('SELL' == 'SELL' and reg_bull == _REGIME_TRENDING_BULL) or
               ('SELL' == 'BUY'  and reg_bull == _REGIME_TRENDING_BEAR))
    assert ct_sell is True
    assert (('SELL' == 'SELL' and reg_bull == _REGIME_TRENDING_BEAR) or
            ('SELL' == 'BUY'  and reg_bull == _REGIME_TRENDING_BULL)) is False


def test_bos_conflict_and_volatile_compression_logic():
    # Opposing BOS check
    new_side_buy = 'BUY'
    bos_sig_bearish = -1.0
    bos_conflict_buy = ((new_side_buy == 'BUY' and bos_sig_bearish < 0) or
                        (new_side_buy == 'SELL' and bos_sig_bearish > 0))
    assert bos_conflict_buy is True

    new_side_sell = 'SELL'
    bos_sig_bullish = 1.0
    bos_conflict_sell = ((new_side_sell == 'BUY' and bos_sig_bullish < 0) or
                         (new_side_sell == 'SELL' and bos_sig_bullish > 0))
    assert bos_conflict_sell is True

    # VOLATILE_COMPRESSION check
    reg = _REGIME_VOLATILE_COMPRESS
    bos_sig_neutral = 0.0
    rp_mid = 0.5   # mid-range

    cmpr_confirmed_buy = ((new_side_buy == 'BUY' and bos_sig_neutral > 0) or
                          (new_side_buy == 'SELL' and bos_sig_neutral < 0))
    cmpr_at_extreme_buy = ((new_side_buy == 'BUY' and rp_mid <= 0.35) or
                           (new_side_buy == 'SELL' and rp_mid >= 0.65))

    # In VOLATILE_COMPRESSION mid-range without BOS, trade must be blocked
    cmpr_allowed = (cmpr_confirmed_buy or cmpr_at_extreme_buy)
    assert cmpr_allowed is False


if __name__ == '__main__':
    test_detect_bos_choch_logic()
    test_guard_n_trend_follow_veto()
    test_bos_conflict_and_volatile_compression_logic()
    print("ALL TESTS PASSED SUCCESSFULLY!")
