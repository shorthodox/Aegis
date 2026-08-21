"""A stop must not stand in the shadow of a level just above it.

Reported 2026-08-21 with SAND/USDT on screen: "look at the resistance above the
target resistance, sl is below it... if the market went to it sl will hit".

    entry 0.047640
    stop  0.048748     2.33% above entry
    Res   0.049500     0.79 ATR ABOVE the stop  (ATR% 2.00)

_clear_levels pushes the stop past levels IN ITS WAY, but its reach was
stop + STOP_BUFFER_ATR (0.55 ATR). A level 0.79 ATR further out was never a
candidate, so the stop was left parked underneath it. Price runs to test that
level, collects the stop on the way, and the trade is closed by a move that said
nothing about the thesis.

Same defect as the AAVE far-level case, one step further out: that one was a
level BETWEEN price and the stop, this one is just BEYOND it.

STOP_SHADOW_ATR (1.00) is the reach, set from a sweep rather than picked — the
reported geometry needs 0.79 and the cost knee is just past 1.0.
"""
import pytest

from src.trading import trader_gate as TG
from src.trading.trader_gate import ACTION_REJECT, TraderGate

from scripts.tests.test_trader_gate import mk, run, TURNED_DOWN


def test_the_reach_covers_the_reported_geometry():
    """SAND: the level sat 0.79 ATR beyond the stop."""
    assert TG.STOP_SHADOW_ATR >= 0.79, (
        'the clearance no longer reaches the level that took the SAND stop'
    )


def test_the_reach_is_wider_than_the_old_one_buffer():
    assert TG.STOP_SHADOW_ATR > TG.STOP_BUFFER_ATR


# -- the mechanism -----------------------------------------------------------

def _clear(side, price, stop, atr, levels):
    return TraderGate._clear_levels(side, price, stop, atr, levels, {})


def test_a_level_just_overhead_pushes_the_stop_above_it():
    """A SHORT's stop with a resistance 0.79 ATR overhead must move beyond it."""
    atr = 1.0
    stop = 110.0
    lvl = stop + 0.79 * atr                       # 110.79
    out = _clear('SELL', 100.0, stop, atr, [(lvl, 4)])
    assert out > lvl, (
        f'stop {out:.4f} is still under the {lvl} level it was standing beneath'
    )
    assert out == pytest.approx(lvl + TG.STOP_BUFFER_ATR * atr, abs=1e-9)


def test_the_mirror_holds_for_a_long():
    atr = 1.0
    stop = 90.0
    lvl = stop - 0.79 * atr                       # 89.21
    out = _clear('BUY', 100.0, stop, atr, [(lvl, 4)])
    assert out < lvl
    assert out == pytest.approx(lvl - TG.STOP_BUFFER_ATR * atr, abs=1e-9)


def test_a_level_beyond_the_shadow_is_left_alone():
    """The reach is bounded, or every distant level drags the stop into the
    MAX_STOP_ATR reject."""
    atr = 1.0
    stop = 110.0
    far = stop + (TG.STOP_SHADOW_ATR + 0.5) * atr
    assert _clear('SELL', 100.0, stop, atr, [(far, 4)]) == stop


def test_a_level_already_cleared_does_not_move_the_stop():
    """Idempotent: clearing a stop that is already clear changes nothing."""
    atr = 1.0
    stop = 110.0
    lvl = 105.0                                    # well inside, already beyond
    once = _clear('SELL', 100.0, stop, atr, [(lvl, 4)])
    assert _clear('SELL', 100.0, once, atr, [(lvl, 4)]) == pytest.approx(once, abs=1e-9)


def test_the_stop_only_ever_moves_away_from_price():
    """_clear_levels must never TIGHTEN a stop."""
    atr = 1.0
    for lv in (104.0, 108.0, 110.5, 111.0):
        assert _clear('SELL', 100.0, 110.0, atr, [(lv, 4)]) >= 110.0
        assert _clear('BUY', 100.0, 90.0, atr, [(200.0 - lv, 4)]) <= 90.0


# -- the cost is still bounded by the existing guards ------------------------

def test_a_stop_pushed_too_far_is_still_refused():
    """Widening the reach must not smuggle a stop past MAX_STOP_ATR."""
    d = mk(price=100.0, atr=1.0, support=94.0, resistance=101.0,
           rsi=72.0, **TURNED_DOWN)
    plan = run(d, regime='RANGING',
               levels=[(101.0, 4), (101.9, 4), (102.8, 4), (103.7, 4), (94.0, 4)])
    if plan.action == ACTION_REJECT:
        assert plan.stage in ('invalidation', 'payoff', 'setup', 'allocation')
    else:
        assert abs(plan.stop - plan.entry) / 1.0 <= TG.MAX_STOP_ATR + 1e-9


def test_nothing_else_moved():
    assert TG.STOP_BUFFER_ATR == 0.55
    assert TG.MAX_STOP_ATR == 3.00
    assert TG.AT_LEVEL_ATR == 0.35
    assert TG.STOP_SHADOW_ATR == 1.00
