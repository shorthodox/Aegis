"""Two things asked for on 2026-08-23.

1 - "why are the signals which signal quality score 5, 0 10 or anything like
     this are being an armed signal"

Because the ACTION_WORK branch returns BEFORE the MIN_FIRE_QUALITY floor, which
only sits on the ENTER path. So a setup the engine scored 5/100 was armed,
published as a PENDING card with a countdown, and then refused the instant price
reached its level and it tried to enter. The card could never become a trade.

The floor itself is not moving: at 45 the fleet measured 38.5% WR against 78.6%
at 60 (p=0.022). What was wrong is ARMING against a bar the order could not
clear.

2 - "i want them to fire even though they won't hit the target support/resistance
     if there are 5 5 min reversal confirmation candles, they have to be strong"

The far tier - taking an armed setup away from its level - now requires FIVE
CONSECUTIVE 5m candles closing the same way, not the ordinary three of four.
Entering early gives up the price the setup was built on, so it is held to a
strictly higher bar than a fill at the level.

Measured over 30 tokens x ~1000 five-minute bars, share of bars where each rule
is satisfied in either direction:

    3 of 4 (ordinary)   52.5%    ~151 per token per day
    4 of 5              29.8%     ~86
    5 of 6              16.1%     ~46
    4 of 4               9.3%     ~27
    5 of 5 (this)        4.2%     ~12

Strict, but not the dead bar a single 42-token snapshot first suggested.
"""
import inspect

import scripts.engine.engine as E
from src.trading import trader_gate as TG


# -- the strong read ----------------------------------------------------------

def test_the_strong_window_is_five():
    assert E.LiveEngine.ENTRY_5M_STRONG == 5


def test_the_strong_read_is_strictly_harder_than_the_ordinary_one():
    assert E.LiveEngine.ENTRY_5M_STRONG > max(3, E.LiveEngine.ENTRY_5M_WINDOW - 1)


def test_the_strong_read_demands_every_candle():
    src = inspect.getsource(E.LiveEngine._ltf_confirmation)
    assert "out['ltf_bull_strong'] = all(" in src, (
        'the strong read is not all-of-window, so a doji or a counter candle '
        'still confirms'
    )
    assert "out['ltf_bear_strong'] = all(" in src


def test_the_keys_default_to_false():
    src = inspect.getsource(E.LiveEngine._ltf_confirmation)
    assert "'ltf_bull_strong': False" in src
    assert "'ltf_bear_strong': False" in src


def test_enough_candles_are_fetched_for_it():
    src = inspect.getsource(E.LiveEngine._ltf_confirmation)
    assert 'max(self.ENTRY_5M_WINDOW, self.ENTRY_5M_STRONG)' in src, (
        'the fetch is sized for the short window, so the strong read runs off '
        'the end and never fires'
    )


# -- the far tier requires it -------------------------------------------------

def test_the_far_tier_requires_the_strong_read():
    src = inspect.getsource(TG.TraderGate.evaluate)
    assert 'ltf_bull_strong' in src and 'ltf_bear_strong' in src
    assert '_full = (EARLY_ENTRY_ON_REVERSAL and ok and _ltf_turned and _strong' in src


def test_the_near_tier_keeps_the_ordinary_bar():
    """A fill AT the level is not held to the away-from-level bar."""
    src = inspect.getsource(TG.TraderGate.evaluate)
    i = src.index('EARLY_ENTRY_ON_LTF and ok and _ltf_turned')
    j = src.index('EARLY_ENTRY_MAX_ATR', i)
    assert '_strong' not in src[i:j]


def test_the_card_says_five_candles():
    src = inspect.getsource(TG.TraderGate.evaluate)
    assert 'five consecutive 5m candles' in src


# -- quality gates arming, not just entry -------------------------------------

def test_a_low_quality_setup_is_not_armed():
    src = inspect.getsource(E.LiveEngine)
    i = src.index('if plan.action == ACTION_WORK:')
    j = src.index("key = f'{symbol}|{plan.side}'", i)
    guard = src[i:j]
    assert 'MIN_FIRE_QUALITY' in guard, (
        'a setup below the fire floor is still armed, so it shows a pending '
        'card with a countdown that can never become a trade'
    )
    assert 'NOT ARMED' in guard


def test_the_arm_guard_releases_the_order_stores():
    src = inspect.getsource(E.LiveEngine)
    i = src.index('if plan.action == ACTION_WORK:')
    j = src.index("key = f'{symbol}|{plan.side}'", i)
    guard = src[i:j]
    for store in ('_working_orders', '_working_levels', '_working_stops'):
        assert store in guard, f'{store} leaks when arming is refused on quality'


def test_the_floor_itself_did_not_move():
    """45 measured 38.5% WR against 78.6% at 60 (p=0.022)."""
    from scripts.engine import config as C
    assert C.MIN_FIRE_QUALITY == 60.0
