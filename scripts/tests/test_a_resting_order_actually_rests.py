"""A resting order must survive a scan that fails to re-derive it.

Reported 2026-08-23: "armed setups are arriving and disappearing without getting
open even though they are genuine... no signals in 24 hours".

The counterfactual log answered it directly. 571 closed working orders:

    outcome         SUPERSEDED 568   FILLED_AT_LEVEL 2   EXPIRED 1
    median age      0.08 bars  (ONE scan, ~5 minutes)
    lived < 1 bar   551/571  (96%)
    reached expiry  1/571
    came within 0.50 ATR   45
    would have filled      22   <- only 2 did

So the "resting order with a clock" was fiction. Any scan whose plan came back
REJECT popped the order, and the gate is a per-scan snapshot whose inputs jitter
— model opposition flickers, regime confidence dips, rp crosses a line. None of
that is news about a thesis already committed to, but all of it cancelled the
order and started the clock again.

An order now ends for its OWN reasons only: the clock runs out, or price goes
through the invalidation it was placed with.
"""
import time

import pytest

import scripts.live_engine as LE
from src.trading.trader_gate import WORK_EXPIRY_BARS


class _Eng:
    """Only what _tend_working_orders touches."""
    def __init__(self):
        self._working_orders = {}
        self._working_levels = {}
        self._working_stops = {}
        self.closed = []
        self._tend_working_orders = LE.LiveEngine._tend_working_orders.__get__(self)

    def _wo_close(self, symbol, side, outcome, now):
        self.closed.append((symbol, side, outcome))


def _armed(side='SELL', level=100.0, stop=101.0, age_bars=0.0):
    e = _Eng()
    key = f'X/USDT|{side}'
    e._working_orders[key] = time.time() - age_bars * 3600
    e._working_levels[key] = level
    e._working_stops[key] = stop
    return e, key


# -- it survives the churn that was killing it --------------------------------

def test_a_young_order_is_left_alone():
    e, key = _armed(age_bars=0.5)
    assert e._tend_working_orders('X/USDT', 98.0, time.time()) == ''
    assert key in e._working_orders, 'the order was retired for no reason'
    assert e.closed == []


def test_the_level_and_stop_survive_with_it():
    e, key = _armed(age_bars=2.0, level=100.0, stop=101.0)
    e._tend_working_orders('X/USDT', 98.0, time.time())
    assert e._working_levels[key] == 100.0
    assert e._working_stops[key] == 101.0


def test_a_reject_does_not_cancel_a_live_order():
    """568 of 571 orders died exactly here."""
    import inspect
    src = inspect.getsource(LE.LiveEngine._run_trader_gate)
    i = src.index('if plan.action not in (ACTION_ENTER, ACTION_WORK)')
    seg = src[i:i + 900]
    assert '_working_orders.pop' not in seg, (
        'the reject path pops the resting order again — this is the churn that '
        'gave 568 SUPERSEDED against 2 fills'
    )


# -- it still ends, for its own reasons ---------------------------------------

def test_the_clock_still_retires_it():
    e, key = _armed(age_bars=WORK_EXPIRY_BARS + 1)
    why = e._tend_working_orders('X/USDT', 98.0, time.time())
    assert 'expired' in why
    assert key not in e._working_orders
    assert e.closed == [('X/USDT', 'SELL', 'EXPIRED')]


def test_a_breached_invalidation_retires_it_early():
    """Price through the stop before ever reaching the level: the thesis is
    dead and the order should not wait out its clock."""
    e, key = _armed(side='SELL', level=100.0, stop=101.0, age_bars=1.0)
    why = e._tend_working_orders('X/USDT', 101.5, time.time())
    assert 'invalidated' in why
    assert key not in e._working_orders
    assert e.closed == [('X/USDT', 'SELL', 'INVALIDATED')]


def test_the_mirror_holds_for_a_long():
    e, key = _armed(side='BUY', level=100.0, stop=99.0, age_bars=1.0)
    why = e._tend_working_orders('X/USDT', 98.5, time.time())
    assert 'invalidated' in why
    assert key not in e._working_orders


def test_price_short_of_the_stop_does_not_retire_it():
    e, key = _armed(side='SELL', level=100.0, stop=101.0, age_bars=1.0)
    assert e._tend_working_orders('X/USDT', 100.9, time.time()) == ''
    assert key in e._working_orders


def test_every_retirement_frees_all_three_stores():
    """A leak here re-pins a stale level onto the next order for that symbol."""
    for price, why in ((98.0, 'expired'), (101.5, 'invalidated')):
        age = WORK_EXPIRY_BARS + 1 if why == 'expired' else 1.0
        e, key = _armed(age_bars=age, stop=101.0)
        e._tend_working_orders('X/USDT', price, time.time())
        assert key not in e._working_orders
        assert key not in e._working_levels
        assert key not in e._working_stops


def test_a_retirement_ends_the_scan():
    """Otherwise the gate re-arms the same setup in the same breath and the
    clock never means anything."""
    import inspect
    src = inspect.getsource(LE.LiveEngine._run_trader_gate)
    i = src.index('_tend_working_orders')
    seg = src[i:i + 500]
    assert 'return True' in seg, 'a retirement no longer ends the scan'


def test_both_sides_are_tended_not_just_the_planned_one():
    """The plan's side is whatever THIS scan derived; a live order on the other
    side still has a clock."""
    e = _Eng()
    for side, stop in (('BUY', 99.0), ('SELL', 101.0)):
        k = f'X/USDT|{side}'
        e._working_orders[k] = time.time() - (WORK_EXPIRY_BARS + 1) * 3600
        e._working_levels[k] = 100.0
        e._working_stops[k] = stop
    e._tend_working_orders('X/USDT', 100.0, time.time())
    e._tend_working_orders('X/USDT', 100.0, time.time())
    assert e._working_orders == {}, 'a side was never tended'
