"""An armed card must last as long as the order it describes.

Reported 2026-08-23, twice: "armed signals are arriving and disappearing without
getting open", then after the lifecycle fix, "armed signal come and disappear...
still the same issue".

The first fix (cef6f460) was real but only half the problem. It stopped the
ORDER being cancelled every time the gate failed to re-derive its plan - 568 of
571 orders had been dying that way at a median age of one scan. What it did not
touch is the CARD.

pending_entry / pending_side / pending_target / suggested_sl / suggested_tp were
only ever stamped onto the published signal inside the ACTION_WORK branch. On any
scan where a live order existed but the gate returned REJECT, the engine took the
early return and none of those fields were written - so the card reverted to
HOLD while the order went on resting.

The result is worse than the original bug, because now the book and the screen
disagree: the order is alive, the desk is told nothing is there, and on the next
scan that happens to re-derive WORK the card reappears. That is exactly "armed
signal come and disappear".

The gate is a per-scan snapshot and its inputs jitter - model opposition
flickers, regime confidence dips, rp crosses a line. The order survives that by
design. The card now survives it too, rebuilt from state captured at arm time
with the expiry clock still ticking.
"""
import inspect

import scripts.engine.engine as E


SRC = inspect.getsource(E.LiveEngine)


def _engine():
    eng = E.LiveEngine.__new__(E.LiveEngine)
    eng._working_orders = {}
    eng._working_levels = {}
    eng._working_stops = {}
    eng._working_meta = {}
    return eng


def _armed(eng, sym='INJ/USDT', side='SELL', now=1000.0):
    key = f'{sym}|{side}'
    eng._working_orders[key] = now
    eng._working_meta[key] = {
        'side': side, 'level': 5.256, 'stop': 5.4995, 'target': 4.8184,
        'setup': 'RANGE_FADE', 'reason': 'fade at resistance', 'expiry_bars': 8.0,
    }
    return key


# -- the card is rebuilt ------------------------------------------------------

def test_a_resting_order_still_publishes_an_armed_card():
    eng = _engine()
    key = _armed(eng)
    sig = {'signal': 'HOLD'}
    eng._republish_working_card('INJ/USDT', key, sig, 1000.0)
    assert sig['pending_entry'] is True, 'the armed card vanished while the order rested'
    assert sig['pending_side'] == 'SELL'
    assert sig['pending_target'] == 5.256
    assert sig['direction'] == 'SHORT'
    assert sig['working_order'] is True
    assert sig['fire'] is False


def test_the_card_carries_the_orders_own_stop_and_target():
    """Not the model's lean - a long's stop under a short setup is worse than
    no card at all."""
    eng = _engine()
    key = _armed(eng)
    sig = {}
    eng._republish_working_card('INJ/USDT', key, sig, 1000.0)
    assert sig['suggested_sl'] == 5.4995
    assert sig['suggested_tp'] == 4.8184


def test_the_expiry_clock_keeps_ticking():
    eng = _engine()
    key = _armed(eng, now=1000.0)
    sig = {}
    eng._republish_working_card('INJ/USDT', key, sig, 1000.0 + 3 * 3600)
    assert sig['expires_in_bars'] == 5.0, (
        'the countdown restarted, so the card claims more time than the order has'
    )


def test_a_long_reads_as_long():
    eng = _engine()
    key = _armed(eng, side='BUY')
    sig = {}
    eng._republish_working_card('INJ/USDT', key, sig, 1000.0)
    assert sig['direction'] == 'LONG'


def test_no_meta_means_no_card_rather_than_a_wrong_one():
    eng = _engine()
    eng._working_orders['X/USDT|BUY'] = 1000.0
    sig = {'signal': 'HOLD'}
    eng._republish_working_card('X/USDT', 'X/USDT|BUY', sig, 1000.0)
    assert 'pending_entry' not in sig


# -- it is actually wired into the reject path -------------------------------

def test_the_reject_branch_republishes_instead_of_going_silent():
    i = SRC.index('# \u2500\u2500 REJECT')
    j = SRC.index('plan.action == ACTION_WORK', i)
    branch = SRC[i:j]
    assert '_republish_working_card' in branch, (
        'a rejecting scan still returns without re-stamping the card, so the '
        'armed signal blinks out while its order is still resting'
    )
    assert '_live_key' in branch


def test_a_symbol_with_no_live_order_still_publishes_no_trade():
    """The republish must not swallow the ordinary no-trade path."""
    i = SRC.index('# \u2500\u2500 REJECT')
    j = SRC.index('plan.action == ACTION_WORK', i)
    branch = SRC[i:j]
    assert '_publish_no_trade' in branch
    assert 'if _live_key is None:' in branch


# -- the store is released with the order -------------------------------------

def test_the_meta_store_is_released_wherever_the_order_is():
    """A leak here would keep publishing an armed card for a dead order."""
    assert SRC.count("_working_meta', {}).pop") + SRC.count("_working_meta or {}).pop") >= 4, (
        'the card state outlives the order on some retirement path'
    )


def test_expiry_releases_it():
    i = SRC.index('def _tend_working_orders')
    j = SRC.index('async def', i)
    body = SRC[i:j]
    assert '_working_meta' in body, 'expiry and invalidation leave the card behind'
