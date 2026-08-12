"""A losing position may only be closed by its published stop.

This is a track-record integrity rule before it is a trading rule. A signal
ships with an entry and a stop; a subscriber holds to that stop. Every exit that
booked a loss BEFORE price reached it recorded a result the subscriber did not
take, at a price they were never shown — so the published loss count was higher
than the strategy as published produces, and nobody reading the board could
reconcile the two.

Two paths did it, both while price was still inside the stop: the model-reversal
exit and the 24h MAX_HOLD timeout. A third, _purge_subquality_positions, did it
at startup.
"""
import types

import pytest

from scripts.engine.exits import (
    STOP_OWNS_THE_LOSS_SIDE, _is_underwater, ExitsMixin,
)
from scripts.engine.models import Position
from scripts.engine.quality import SignalQualityFilter


ENTRY, STOP = 100.0, 99.30          # a 0.70% stop, the deployed band cap


def _pos(direction='LONG', entry=ENTRY, stop=STOP, **kw):
    return Position(
        symbol='TEST/USDT', direction=direction,
        side='BUY' if direction == 'LONG' else 'SELL',
        entry_price=entry, position_value=500.0, stop_loss=stop,
        signal_id='t1', entry_time='2026-08-12T00:00:00+00:00',
        meta_confidence=kw.pop('edge', 45.0), atr_multiplier=1.5, atr=0.9,
        take_profit_1=entry * 1.015, **kw)


# ── the predicate ────────────────────────────────────────────────────────────

@pytest.mark.parametrize('direction,price,under', [
    ('LONG',  99.50, True),    # inside the stop, losing
    ('LONG',  99.95, True),    # a hair down — still a loss
    ('LONG', 100.50, False),   # in profit
    ('SHORT', 100.50, True),
    ('SHORT',  99.50, False),
])
def test_underwater_is_direction_correct(direction, price, under):
    assert _is_underwater(_pos(direction), price) is under


def test_costs_count_as_underwater():
    """A move smaller than the round trip is not a win; closing it books a loss."""
    p = _pos()
    assert _is_underwater(p, ENTRY * 1.0005, cost_pct=0.10) is True
    assert _is_underwater(p, ENTRY * 1.0050, cost_pct=0.10) is False


def test_degenerate_prices_do_not_report_underwater():
    """A missing price must not be read as a loss and trigger a hold."""
    assert _is_underwater(_pos(), 0.0) is False
    assert _is_underwater(_pos(entry=0.0), 100.0) is False


# ── the rule is on ───────────────────────────────────────────────────────────

def test_the_flag_is_on():
    assert STOP_OWNS_THE_LOSS_SIDE is True, (
        'the published stop is the only level allowed to close a losing '
        'position; turning this off re-opens the loss-count inflation'
    )


def test_max_hold_holds_a_loser_and_still_retires_a_winner():
    """Both halves matter: the timeout keeps its job on trades it may close."""
    import inspect
    src = inspect.getsource(ExitsMixin._manage_exit)
    i_hold = src.index('MAX_HOLD_SECONDS')
    seg = src[i_hold:i_hold + 900]
    assert 'STOP_OWNS_THE_LOSS_SIDE' in seg and '_is_underwater' in seg, (
        'MAX_HOLD_EXPIRED must not book a loss the stop has not reached'
    )
    assert "_close('MAX_HOLD_EXPIRED')" in seg, (
        'the timeout must still retire a flat or winning zombie'
    )


def test_reversal_cannot_close_a_position_that_is_under_water():
    import inspect
    src = inspect.getsource(ExitsMixin._manage_exit)
    i = src.index('REVERSAL_HOLD')
    # the guard has to precede the close it is guarding
    assert src.index('STOP_OWNS_THE_LOSS_SIDE', i - 400) < src.index(
        "_close('MODEL_REVERSAL_TP')", i), (
        'the reversal exit must be gated before it can book a loss inside the stop'
    )


# ── the startup purge ────────────────────────────────────────────────────────

def _purge_targets(positions):
    """Mirror of the comprehension in _purge_subquality_positions."""
    return [
        sym for sym, pos in positions.items()
        if not (pos.signal_strength or '').strip()
        and not (pos.entry_mode or '').strip()
        and pos.meta_confidence < SignalQualityFilter.MIN_QUALITY_SCORE
    ]


def test_a_current_generation_position_survives_a_restart():
    """The bug: low model edge is NORMAL now — REQUIRE_MODEL_FIRE is False, so
    the desk opens on structure and the edge may sit anywhere."""
    cur = _pos(edge=45.0)
    cur.signal_strength, cur.entry_mode = 'NORMAL', 'range_fade'
    assert _purge_targets({'TEST/USDT': cur}) == []


def test_a_genuinely_pre_gate_position_is_still_purged():
    old = _pos(edge=45.0)
    old.signal_strength, old.entry_mode = '', ''
    assert _purge_targets({'TEST/USDT': old}) == ['TEST/USDT']


def test_a_high_edge_pre_gate_position_is_left_alone():
    old = _pos(edge=88.0)
    old.signal_strength, old.entry_mode = '', ''
    assert _purge_targets({'TEST/USDT': old}) == []
