"""Regression tests for the v82c scalp-engine guards.

Motivated by the recorded scalp losses of 2026-07-23, where ZIL/USDT was shorted
eight consecutive times in nine minutes at 9-14 % model confidence, every entry
higher than the last, all eight stopped out, while the token rallied 8.6 %.
Re-entry gaps reached 32 seconds against a nominal 10/15-minute cooldown, and
17 of the 18 recorded losses were shorts taken this way.

Root cause: ScalpBot called TraderEngine.scan_all_tokens(force_fire=True), and
force_fire wrapped the whole gate block — cooldown, confidence floor,
confluence, ATR floor, volume floor and RSI-extreme all switched off together.
"""
import json
from datetime import datetime, timedelta, timezone

import pytest

from scripts.trader_model import signal_manager as sm


@pytest.fixture(autouse=True)
def _tmp_state(tmp_path, monkeypatch):
    """Point the cooldown tracker at a throwaway state file."""
    monkeypatch.setattr(sm, 'TRADER_STATE_PATH', tmp_path / 'state.json')
    yield


def _age_last_signal(symbol, mode, minutes):
    """Backdate the signal clock so only the loss logic is under test."""
    state = sm._load_state()
    state[f'{symbol}_{mode}'] = (
        datetime.now(timezone.utc) - timedelta(minutes=minutes)
    ).isoformat()
    sm._save_state(state)


# ── the force_fire blast radius ──────────────────────────────────────────────

def test_scalpbot_does_not_force_fire():
    """The single line that disabled every gate at once.

    ScalpBot moved from live_engine.py to scripts/engine/scalp.py when the
    engine was split; the guard is unchanged, only its address.
    """
    src = (sm._ROOT / 'scripts' / 'engine' / 'scalp.py').read_text(encoding='utf-8')
    scan = src[src.index('class ScalpBot'):]
    assert 'force_fire=True' not in scan, (
        'ScalpBot is passing force_fire=True again — that disables the '
        'confidence floor, confluence, ATR, volume and RSI gates'
    )


def test_force_fire_cannot_disable_quality_gates():
    """force_fire may waive the cooldown and nothing else."""
    src = (sm._ROOT / 'scripts' / 'trader_model' / 'trader_engine.py').read_text(
        encoding='utf-8')
    assert 'if on_cooldown and not force_fire:' in src
    # the old form wrapped every gate in one conditional
    assert 'if not force_fire:' not in src, (
        'quality gates are back inside a force_fire conditional'
    )


# ── cooldown basics ──────────────────────────────────────────────────────────

def test_fresh_symbol_is_not_on_cooldown():
    assert sm.is_on_cooldown('ZIL/USDT', 'scalping') is False


def test_cooldown_blocks_immediate_refire():
    sm.record_signal('ZIL/USDT', 'scalping')
    assert sm.is_on_cooldown('ZIL/USDT', 'scalping') is True


def test_cooldown_expires():
    sm.record_signal('ZIL/USDT', 'scalping')
    _age_last_signal('ZIL/USDT', 'scalping', 999)
    assert sm.is_on_cooldown('ZIL/USDT', 'scalping') is False


def test_cooldown_is_per_mode():
    sm.record_signal('ZIL/USDT', 'scalping')
    assert sm.is_on_cooldown('ZIL/USDT', 'scalping') is True
    assert sm.is_on_cooldown('ZIL/USDT', 'scalping_15m') is False


# ── post-loss escalation ─────────────────────────────────────────────────────

def test_loss_streak_increments_and_resets_on_a_win():
    for expected in (1, 2):
        assert sm.record_loss('ZIL/USDT', 'scalping') == expected
    sm.record_win('ZIL/USDT', 'scalping')
    assert sm.get_loss_streak('ZIL/USDT', 'scalping') == 0


def test_a_loss_lengthens_the_cooldown():
    base = sm._base_cooldown_minutes('scalping')
    assert sm._effective_cooldown_minutes('ZIL/USDT', 'scalping') == base
    sm.record_loss('ZIL/USDT', 'scalping')
    assert sm._effective_cooldown_minutes('ZIL/USDT', 'scalping') > base


def test_escalated_cooldown_is_capped():
    for _ in range(12):
        sm.record_loss('ZIL/USDT', 'scalping')
    assert (sm._effective_cooldown_minutes('ZIL/USDT', 'scalping')
            <= sm.LOSS_COOLDOWN_CAP)


def test_three_losses_hard_block_the_symbol():
    for _ in range(sm.LOSS_STREAK_BLOCK):
        sm.record_loss('ZIL/USDT', 'scalping')
    assert sm.is_loss_blocked('ZIL/USDT', 'scalping') is True
    # still blocked well after the ordinary cooldown would have lapsed
    _age_last_signal('ZIL/USDT', 'scalping', 60)
    assert sm.is_on_cooldown('ZIL/USDT', 'scalping') is True


def test_hard_block_lifts_after_its_window():
    for _ in range(sm.LOSS_STREAK_BLOCK):
        sm.record_loss('ZIL/USDT', 'scalping')
    state = sm._load_state()
    state[sm._loss_key('ZIL/USDT', 'scalping')]['at'] = (
        datetime.now(timezone.utc)
        - timedelta(minutes=sm.LOSS_BLOCK_MINUTES + 1)
    ).isoformat()
    sm._save_state(state)
    _age_last_signal('ZIL/USDT', 'scalping', sm.LOSS_BLOCK_MINUTES + 1)
    assert sm.is_loss_blocked('ZIL/USDT', 'scalping') is False
    assert sm.is_on_cooldown('ZIL/USDT', 'scalping') is False


def test_a_win_clears_the_hard_block():
    for _ in range(sm.LOSS_STREAK_BLOCK):
        sm.record_loss('ZIL/USDT', 'scalping')
    assert sm.is_loss_blocked('ZIL/USDT', 'scalping') is True
    sm.record_win('ZIL/USDT', 'scalping')
    assert sm.is_loss_blocked('ZIL/USDT', 'scalping') is False


# ── the ZIL sequence itself ──────────────────────────────────────────────────

def test_the_recorded_zil_sequence_is_now_blocked():
    """Replay the real timestamps; count how many of the 8 shorts survive."""
    entries = [  # (HH:MM:SS, mode) from data/trader_track_record.json
        ('10:19:29', 'scalping_15m'), ('10:21:21', 'scalping'),
        ('10:23:30', 'scalping_15m'), ('10:25:03', 'scalping_15m'),
        ('10:25:44', 'scalping'),     ('10:27:00', 'scalping'),
        ('10:27:32', 'scalping_15m'), ('10:28:29', 'scalping'),
    ]
    base = datetime(2026, 7, 23, tzinfo=timezone.utc)
    t0   = base + timedelta(hours=10, minutes=19, seconds=29)

    fired = 0
    for hhmmss, mode in entries:
        h, m, s = (int(x) for x in hhmmss.split(':'))
        now = base + timedelta(hours=h, minutes=m, seconds=s)
        # emulate "now" by backdating the stored clocks relative to this entry
        state = sm._load_state()
        shift = (datetime.now(timezone.utc) - now).total_seconds()
        for k, v in list(state.items()):
            if isinstance(v, str):
                state[k] = (datetime.fromisoformat(v) + timedelta(seconds=shift)).isoformat()
            elif isinstance(v, dict) and 'at' in v:
                v['at'] = (datetime.fromisoformat(v['at']) + timedelta(seconds=shift)).isoformat()
        sm._save_state(state)

        if sm.is_on_cooldown('ZIL/USDT', mode):
            continue
        fired += 1
        sm.record_signal('ZIL/USDT', mode)
        sm.record_loss('ZIL/USDT', mode)   # every one of these stopped out

        # undo the shift so stored times stay absolute for the next iteration
        state = sm._load_state()
        for k, v in list(state.items()):
            if isinstance(v, str):
                state[k] = (datetime.fromisoformat(v) - timedelta(seconds=shift)).isoformat()
            elif isinstance(v, dict) and 'at' in v:
                v['at'] = (datetime.fromisoformat(v['at']) - timedelta(seconds=shift)).isoformat()
        sm._save_state(state)

    assert fired <= 2, f'{fired} of 8 ZIL shorts still fire (was 8)'
