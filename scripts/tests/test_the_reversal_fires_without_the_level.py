"""Three strong 5m candles fire a resting order even if the level never comes.

Asked 2026-08-24: "even if market won't reach the level, and 3 strong candles
are seen in 5 min chart... signal should fire immediately."

THE THING THAT MAKES THIS HARD is not the trigger, it is the stop.

A resting order's stop is anchored to the level it waits at. Filling at market
without that level inherits the whole gap as risk. Live case, ENA/USDT:

    resting level 0.1494 · order stop 0.14746 · price 0.1605
    -> filled at market on the order's own stop = 8.13% risk

against MAX_STOP_PCT of 1.30. It would be refused on width every single time,
so a trigger built that way would do nothing at all.

So an early fill is stopped on THE REVERSAL'S OWN STRUCTURE: the low of the
three confirming 5m candles, plus a buffer. That is still a structural stop —
just the structure the entry is actually being taken from rather than one
percent away. The target stays the order's, and both MAX_STOP_PCT and MIN_NET_R
are applied to the result, so a reversal that does not pay from here is refused
rather than taken because the candles looked good.

Trigger A (price reaches the level, ordinary 3-of-4 confirmation) is untouched
and still requires the level.
"""
import inspect

import scripts.engine.engine as E
from src.trading.trader_gate import MAX_STOP_PCT, MIN_NET_R


SRC = inspect.getsource(E.LiveEngine)


def _eng():
    eng = E.LiveEngine.__new__(E.LiveEngine)
    eng._working_orders = {}
    eng._working_levels = {}
    eng._working_stops = {}
    eng._working_meta = {}
    return eng


def _arm(eng, sym='ENA/USDT', side='BUY', level=0.1494, stop=0.14746,
         target=0.1835, now=1000.0):
    key = f'{sym}|{side}'
    eng._working_orders[key] = now
    eng._working_levels[key] = level
    eng._working_stops[key] = stop
    eng._working_meta[key] = {
        'side': side, 'level': level, 'stop': stop, 'target': target,
        'setup': 'TREND_PULLBACK', 'reason': 'pullback to support',
        'expiry_bars': 8.0, 'entry': level, 'invalidation': stop,
        'risk_atr': 1.0, 'r_gross': 2.3, 'r_net': 2.31, 'size_factor': 0.17,
    }
    return key


# a strong up-turn whose own swing low sits just under the market
STRONG_UP = {'ltf_bull': True, 'ltf_bear': False,
             'ltf_bull_strong': True, 'ltf_bear_strong': False,
             'ltf_low': 0.1600, 'ltf_high': 0.1620}
WEAK_UP = {'ltf_bull': True, 'ltf_bear': False,
           'ltf_bull_strong': False, 'ltf_bear_strong': False,
           'ltf_low': 0.1600, 'ltf_high': 0.1620}


def test_three_strong_candles_fire_without_the_level():
    eng = _eng()
    _arm(eng)
    plan = eng._resting_fill_plan('ENA/USDT', 0.1605, STRONG_UP, 2000.0)
    assert plan is not None, (
        'three consecutive 5m candles turned and the order still did not fire'
    )
    assert plan.side == 'BUY'
    assert plan.entry == 0.1605, 'it must be taken at the market, not at the level'


def test_the_stop_comes_from_the_reversal_not_the_order():
    """The whole point — inheriting the order stop is an 8.13% risk here."""
    eng = _eng()
    _arm(eng)
    plan = eng._resting_fill_plan('ENA/USDT', 0.1605, STRONG_UP, 2000.0)
    assert plan.stop > 0.155, (
        f'stop {plan.stop} is still anchored to the far level, which makes the '
        f'trade unaffordable and it will be refused on width'
    )
    assert plan.stop < 0.1605
    risk_pct = abs(plan.entry - plan.stop) / plan.entry * 100
    assert risk_pct <= MAX_STOP_PCT


def test_it_keeps_the_orders_target():
    eng = _eng()
    _arm(eng, target=0.1835)
    plan = eng._resting_fill_plan('ENA/USDT', 0.1605, STRONG_UP, 2000.0)
    assert plan.target == 0.1835


def test_a_reversal_that_does_not_pay_is_refused():
    """MIN_NET_R still bites. A target barely above the entry cannot clear it."""
    eng = _eng()
    _arm(eng, target=0.1610)
    plan = eng._resting_fill_plan('ENA/USDT', 0.1605, STRONG_UP, 2000.0)
    assert plan is None


def test_a_stop_wider_than_the_cap_is_refused():
    eng = _eng()
    _arm(eng)
    far = dict(STRONG_UP, ltf_low=0.1400)      # swing far below -> huge risk
    plan = eng._resting_fill_plan('ENA/USDT', 0.1605, far, 2000.0)
    assert plan is None


def test_two_candles_are_not_enough_away_from_the_level():
    """The ordinary 3-of-4 read does not fire an early entry — only the strong
    consecutive read does."""
    eng = _eng()
    _arm(eng)
    assert eng._resting_fill_plan('ENA/USDT', 0.1605, WEAK_UP, 2000.0) is None


def test_a_missing_swing_does_not_fire():
    eng = _eng()
    _arm(eng)
    no_swing = dict(STRONG_UP)
    no_swing.pop('ltf_low')
    assert eng._resting_fill_plan('ENA/USDT', 0.1605, no_swing, 2000.0) is None


def test_the_strong_read_must_match_the_side():
    eng = _eng()
    _arm(eng, side='BUY')
    wrong = {'ltf_bull': False, 'ltf_bear': True,
             'ltf_bull_strong': False, 'ltf_bear_strong': True,
             'ltf_low': 0.1600, 'ltf_high': 0.1620}
    assert eng._resting_fill_plan('ENA/USDT', 0.1605, wrong, 2000.0) is None


# -- trigger A is untouched ---------------------------------------------------

def test_the_ordinary_fill_still_requires_the_level():
    """Removing the level guard here would fire every order at any price."""
    eng = _eng()
    _arm(eng)
    ordinary = {'ltf_bull': True, 'ltf_bear': False,
                'ltf_bull_strong': False, 'ltf_bear_strong': False}
    assert eng._resting_fill_plan('ENA/USDT', 0.1605, ordinary, 2000.0) is None
    at_level = eng._resting_fill_plan('ENA/USDT', 0.1494, ordinary, 2000.0)
    assert at_level is not None
    assert at_level.stop == 0.14746, 'a fill AT the level keeps the order stop'


def test_the_two_triggers_are_distinguishable_in_the_card():
    src = inspect.getsource(E.LiveEngine._resting_fill_plan)
    assert 'without waiting for the' in src
    assert 'filled at its' in src


def test_the_refusal_is_logged_with_its_numbers():
    """A refusal here must be attributable, not silent."""
    src = inspect.getsource(E.LiveEngine._resting_fill_plan)
    assert 'EARLY REVERSAL REFUSED' in src
    assert 'does not pay from' in src


def test_the_five_minute_swing_is_published():
    src = inspect.getsource(E.LiveEngine._ltf_confirmation)
    assert "out['ltf_low']" in src and "out['ltf_high']" in src
