"""A resting order must fill when price reaches its level.

Reported 2026-08-24: "Since last night we got a lot of armed signals, none of em
fired... they were genuine, even though if they fired, i guess they are not
stored in volume to showcase."

Both halves were true.

1 - AN ORDER COULD ONLY BE FILLED BY THE GATE RE-DERIVING IT FROM SCRATCH

There was no fill path. A working order was filled only if TraderGate.evaluate
independently returned ACTION_ENTER on some later scan. So an order rested AT its
level while the gate refused the symbol for reasons that had nothing to do with
the order - the setup stage moving on, exhaustion refused fleet-wide, rp drifting
- and it expired unfilled with price sitting right on it.

Measured on the volume, TAO/USDT 2026-08-23:

    level 244.85, touched at 2.17 bars, closest approach 0.0125 ATR,
    19 scans, outcome EXPIRED

and the same night, per scan: 221 NO TRADE (setup), 0 fires.

A resting limit does not re-ask whether the thesis is still fashionable. The
thesis was settled when the order was placed and its price, stop, target and
payoff were frozen with it. What must still hold at fill time is only that the
invalidation is intact, price has reached the level, and the 5m tape confirms -
the last being a HARD requirement by desk decision.

Everything downstream still runs: the quality floor, the entry-stretch veto, the
portfolio guard. Those are about risk and capacity, not about re-litigating a
decision already made.

2 - THE ARMED BOOK LIVED ONLY IN MEMORY

_working_orders / _working_levels / _working_stops / _working_meta were instance
attributes and nothing else. Every deploy cancelled every resting order outright
- no expiry, no invalidation, not even a counterfactual row - so armed signals
vanished for reasons the log could never account for. An order carries an 8-bar
clock; a container does not survive eight bars of deploys.
"""
import inspect

import scripts.engine.engine as E
from scripts.engine import config as ECFG


SRC = inspect.getsource(E.LiveEngine)

TURNED_DOWN = {'ltf_bull': False, 'ltf_bear': True}
QUIET = {'ltf_bull': False, 'ltf_bear': False}


def _eng():
    eng = E.LiveEngine.__new__(E.LiveEngine)
    eng._working_orders = {}
    eng._working_levels = {}
    eng._working_stops = {}
    eng._working_meta = {}
    return eng


def _arm(eng, sym='TAO/USDT', side='SELL', level=244.85, stop=246.99, now=1000.0):
    key = f'{sym}|{side}'
    eng._working_orders[key] = now
    eng._working_levels[key] = level
    eng._working_stops[key] = stop
    eng._working_meta[key] = {
        'side': side, 'level': level, 'stop': stop, 'target': 218.8,
        'setup': 'EXHAUSTION_REVERSAL', 'reason': 'fade at resistance',
        'expiry_bars': 8.0, 'entry': level, 'invalidation': stop,
        'risk_atr': 1.0, 'r_gross': 2.0, 'r_net': 1.78, 'size_factor': 0.17,
    }
    return key


# -- the fill -----------------------------------------------------------------

def test_price_at_the_level_fills_the_order():
    eng = _eng()
    _arm(eng)
    plan = eng._resting_fill_plan('TAO/USDT', 244.85, TURNED_DOWN, 2000.0)
    assert plan is not None, 'price reached the level and the order still did not fill'
    assert plan.side == 'SELL'
    assert plan.level == 244.85


def test_the_filled_plan_carries_the_frozen_thesis():
    """Stop, target and payoff were settled when the order was placed."""
    eng = _eng()
    _arm(eng)
    plan = eng._resting_fill_plan('TAO/USDT', 244.85, TURNED_DOWN, 2000.0)
    assert plan.stop == 246.99
    assert plan.target == 218.8
    assert round(plan.r_net, 2) == 1.78
    assert plan.size_factor == 0.17
    assert plan.setup == 'EXHAUSTION_REVERSAL'


def test_a_quiet_5m_does_not_fill():
    """The one thing that is genuinely news at fill time, and a hard rule."""
    eng = _eng()
    _arm(eng)
    assert eng._resting_fill_plan('TAO/USDT', 244.85, QUIET, 2000.0) is None


def test_the_5m_must_confirm_the_orders_own_side():
    eng = _eng()
    _arm(eng, side='SELL')
    wrong_way = {'ltf_bull': True, 'ltf_bear': False}
    assert eng._resting_fill_plan('TAO/USDT', 244.85, wrong_way, 2000.0) is None


def test_price_short_of_the_level_does_not_fill():
    eng = _eng()
    _arm(eng, level=244.85, stop=246.99)
    assert eng._resting_fill_plan('TAO/USDT', 200.0, TURNED_DOWN, 2000.0) is None


def test_a_long_fills_from_below_not_from_far_above():
    eng = _eng()
    _arm(eng, sym='X/USDT', side='BUY', level=100.0, stop=98.0)
    assert eng._resting_fill_plan('X/USDT', 100.0, {'ltf_bull': True}, 2000.0) is not None
    assert eng._resting_fill_plan('X/USDT', 120.0, {'ltf_bull': True}, 2000.0) is None


def test_no_resting_order_means_no_fill():
    assert _eng()._resting_fill_plan('X/USDT', 100.0, TURNED_DOWN, 2000.0) is None


def test_the_fill_runs_before_the_gate():
    assert '_resting_fill_plan(symbol, price, confirm, now)' in SRC
    i = SRC.index('_resting_fill_plan(symbol, price, confirm, now)')
    j = SRC.index('TraderGate.evaluate(', i)
    assert j > i, 'the gate is consulted before the resting order is checked'


def test_downstream_risk_gates_still_apply():
    """The quality floor and the rest are NOT bypassed - only the thesis
    re-derivation is."""
    assert 'MIN_FIRE_QUALITY' in SRC
    i = SRC.index('_resting_fill_plan(symbol, price, confirm, now)')
    assert SRC.index('LOW_QUALITY_REFUSED', i) > i


# -- the book survives a deploy -----------------------------------------------

def test_the_armed_book_has_a_volume_path():
    assert ECFG.WORKING_ORDERS_PATH.parent == ECFG.STATE_DIR


def test_the_book_is_saved_on_every_change():
    """Arm, expire, invalidate, fill, and refuse-to-arm."""
    assert SRC.count("_save_working_orders', lambda: None)()") >= 5, (
        'some path changes the armed book without persisting it, so a deploy '
        'silently reverts it'
    )


def test_the_book_is_restored_at_startup():
    assert '_load_working_orders()' in SRC
    src = inspect.getsource(E.LiveEngine._load_working_orders)
    assert 'WORK_EXPIRY_BARS' in src, (
        'orders that expired while the engine was down are restored as live'
    )


def test_a_restore_failure_is_not_fatal():
    src = inspect.getsource(E.LiveEngine._load_working_orders)
    assert 'except Exception' in src
    assert 'empty book' in src


def test_restore_drops_what_expired_while_down():
    import json
    import time as _t
    eng = E.LiveEngine.__new__(E.LiveEngine)
    stale = _t.time() - (E.WORK_EXPIRY_BARS + 2) * 3600
    fresh = _t.time() - 3600
    blob = {
        'orders': {'A/USDT|BUY': stale, 'B/USDT|SELL': fresh},
        'levels': {'A/USDT|BUY': 1.0, 'B/USDT|SELL': 2.0},
        'stops': {}, 'meta': {},
    }
    ECFG.WORKING_ORDERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ECFG.WORKING_ORDERS_PATH.write_text(json.dumps(blob), encoding='utf-8')
    try:
        eng._load_working_orders()
        assert 'A/USDT|BUY' not in eng._working_orders, 'an expired order came back alive'
        assert 'B/USDT|SELL' in eng._working_orders
        assert 'A/USDT|BUY' not in eng._working_levels
    finally:
        ECFG.WORKING_ORDERS_PATH.unlink(missing_ok=True)


# ── a resting fill is still bound by the book ────────────────────────────────
# The fill path returns a plan DIRECTLY and never calls TraderGate.evaluate,
# which is where MAX_OPEN_TOTAL and MAX_PER_CLUSTER are enforced. So for a day
# every fill on this path opened with no exposure limit at all.
#
# Measured 2026-08-26: eight correlated LONGs open at once against a cap of five
# total and two per cluster. BTC fell 1.64%, alts amplified it, and the entire
# book stopped together — ENA -3.58, XRP -2.96, SUI -3.71, HBAR -2.32,
# INJ -4.62, TRB -1.49, ENA -2.32. Eight positions expressing one bet, which is
# precisely what the cluster cap exists to prevent.

def test_the_fill_path_consults_the_portfolio_guard():
    assert 'self.portfolio_guard.can_open(' in SRC, (
        'a resting order can still fill with the book already full, so one '
        'market move takes out an unlimited number of correlated positions'
    )


def test_the_guard_runs_before_any_candidate_is_built():
    src = inspect.getsource(E.LiveEngine._resting_fill_plan)
    guard = src.index('can_open(')
    build = src.index("for side in ('BUY', 'SELL')")
    assert guard < build, 'exposure is checked after the plan is already made'


def test_a_refused_fill_returns_none():
    src = inspect.getsource(E.LiveEngine._resting_fill_plan)
    i = src.index('if not _ok:')
    assert 'return None' in src[i:i + 80]


def test_the_wallet_is_synced_before_the_check():
    """can_open counts from the guard's own view, which is stale unless the
    live book is pushed into it first."""
    src = inspect.getsource(E.LiveEngine._resting_fill_plan)
    sync = src.index('sync_from_wallet(')
    check = src.index('can_open(')
    assert sync < check


def test_a_guard_failure_does_not_block_the_fill():
    """The guard is a risk cap, not a data dependency. If it raises, the fill
    proceeds as before rather than the desk silently stopping."""
    src = inspect.getsource(E.LiveEngine._resting_fill_plan)
    i = src.index('sync_from_wallet(')
    assert 'except Exception:' in src[i:i + 700]


def test_the_caps_themselves_are_unchanged():
    from scripts.engine.portfolio import PortfolioGuard
    assert PortfolioGuard.MAX_OPEN_TOTAL == 5
    assert PortfolioGuard.MAX_PER_CLUSTER == 2
