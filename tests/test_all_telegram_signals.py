"""
tests/test_all_telegram_signals.py — Unit test verifying:
  Telegram formatting and dispatching for all 5 signal categories:
  1. Live Fired Entry
  2. Armed / Waiting for Level
  3. Tradable / Under Observation (Paper)
  4. Unfired / Blocked
  5. Position Exit (WIN / LOSS)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make sure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.notifications.formatter import (
    format_entry_telegram,
    format_pending_telegram,
    format_observation_telegram,
    format_blocked_telegram,
    format_exit_telegram,
)


def test_telegram_formatters_all_signals():
    # 1. Live Fired Entry
    sig_live = {
        'symbol': 'BTC/USDT',
        'direction': 'BUY',
        'current_price': 65000.0,
        'stop_loss': 64000.0,
        'take_profit_1': 66500.0,
        'confidence': 0.85,
        'mode': 'scalping',
    }
    msg_live = format_entry_telegram(sig_live)
    assert 'AEGIS BUY — BTC/USDT' in msg_live
    assert 'Entry:' in msg_live

    # 2. Armed / Waiting for Level
    sig_pending = {
        'symbol': 'ETH/USDT',
        'direction': 'SELL',
        'pending_target': 3500.0,
    }
    msg_pending = format_pending_telegram(sig_pending)
    assert 'AEGIS WATCHING — ETH/USDT' in msg_pending
    assert 'At resistance:' in msg_pending or '3500' in msg_pending

    # 3. Tradable / Under Observation (Paper)
    sig_obs = {
        'symbol': 'SOL/USDT',
        'direction': 'BUY',
        'current_price': 140.0,
        'stop_loss': 135.0,
        'take_profit_1': 150.0,
        'paper_reason': 'internal paper validation (class #2)',
    }
    msg_obs = format_observation_telegram(sig_obs)
    assert 'TRADABLE · UNDER OBSERVATION' in msg_obs
    assert 'SOL/USDT' in msg_obs

    # 4. Unfired / Blocked
    sig_blocked = {
        'symbol': 'XRP/USDT',
        'direction': 'SELL',
        'current_price': 1.13,
        'structure_reason': 'SELL in TRENDING_BEAR prohibited (fade-only engine)',
    }
    msg_blocked = format_blocked_telegram(sig_blocked)
    assert 'UNFIRED · BLOCKED' in msg_blocked
    assert 'XRP/USDT' in msg_blocked
    assert 'SELL in TRENDING_BEAR' in msg_blocked

    # 5. Position Exit
    msg_exit = format_exit_telegram('BTC/USDT', 'BUY', 'WIN', 3.5, 3600, 'tp1_hit')
    assert 'AEGIS CLOSED' in msg_exit
    assert 'WIN' in msg_exit
    assert '+3.50%' in msg_exit


if __name__ == '__main__':
    test_telegram_formatters_all_signals()
    print("ALL TELEGRAM FORMATTER TESTS PASSED SUCCESSFULLY!")
