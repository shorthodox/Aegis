"""A resting order the market never comes back to is not a trade.

Reported 2026-08-21 with ZK/USDT on screen: "if it won't hit target resistance,
trade won't open... this is resulting in no signals anymore". Confirmed live at
the time: 2 armed, 0 open, both TREND_PULLBACK SELLs resting at a resistance
price never rallied into, expiring after WORK_EXPIRY_BARS without ever becoming
trades.

    PENDING  SELL @ 0.04826119    1.85R net, size 0.25
    PENDING  SELL @ 0.0094214815  1.96R net, size 0.25   <- the ZK one

The fix takes the trade at the market when the 5m tape has already turned our
way, rather than waiting for a touch that may not come. _ltf_confirmation
already computes exactly that print: ltf_bull/ltf_bear are true when at least 3
of the last ENTRY_5M_WINDOW 5m candles closed our way (need = max(3, window-1)).

The requested safety condition -- "make sure stoploss is always above the
resistance according to the structure" -- holds by construction and is asserted
below: the stop is anchored to the LEVEL (`stop = level +/- buf`, then
_clear_levels), never to price. The worse entry is paid for rather than waved
through, because `risk = abs(price - stop)` is already measured from price, so
MAX_STOP_ATR and stage 3's MIN_NET_R floor both bite on the real entry.
"""
import pytest

from src.trading import trader_gate as TG
from src.trading.trader_gate import (
    ACTION_ENTER, ACTION_REJECT, ACTION_WORK, TraderGate,
)

from scripts.tests.test_trader_gate import mk, run, TURNED_DOWN


TURNED = {'ltf_bull': False, 'ltf_bear': True}      # the 5m has turned DOWN
QUIET = {'ltf_bull': False, 'ltf_bear': False}      # the tape says nothing


def _short_below_resistance(**kw):
    """A bear rallying toward resistance, price just short of the level.

    0.42 ATR short of it: OUTSIDE AT_LEVEL_ATR (0.35), so it cannot take the
    ordinary at-the-level path, and INSIDE EARLY_ENTRY_MAX_ATR (0.50), so it is
    the early fill under test. The fixture used to
    sit 1.53 ATR away, which since 2026-08-21 is correctly too far to fill
    early — the give-up is the whole point of these setups and is now bounded.
    See test_a_fill_far_from_the_level_goes_back_to_working_the_order.

    rp in a TRENDING_BEAR is PULLBACK_RP_SHORT territory -> TREND_PULLBACK
    SELL, which is exactly what was left armed in production.
    """
    d = mk(price=100.0, atr=1.5, support=94.0, resistance=100.63,
           rsi=55.0, **TURNED_DOWN)
    d.update(kw)
    return d


LEVELS = [(100.63, 4), (94.0, 4), (106.0, 3)]


def _plan(confirm, **kw):
    return run(_short_below_resistance(), regime='TRENDING_BEAR',
               levels=LEVELS, confirm=confirm, **kw)


# -- the regression ----------------------------------------------------------

def test_a_turned_tape_enters_instead_of_resting():
    plan = _plan(TURNED)
    assert plan.action == ACTION_ENTER, (
        f'still resting ({plan.action}: {plan.reason}) — this is the order that '
        f'expires unfilled and produces no signal'
    )


def test_the_card_says_why_it_did_not_wait():
    plan = _plan(TURNED)
    assert any('5m tape turned' in n for n in (plan.notes or [])), (
        'an early fill must say it was early, or the card implies a level touch '
        'that never happened'
    )


def test_a_quiet_tape_still_rests_at_the_level():
    """Without the turn, nothing changes — this is not "enter always"."""
    plan = _plan(QUIET)
    assert plan.action == ACTION_WORK


def test_the_flag_restores_the_old_behaviour(monkeypatch):
    monkeypatch.setattr(TG, 'EARLY_ENTRY_ON_LTF', False)
    assert _plan(TURNED).action == ACTION_WORK


# -- the requested safety condition ------------------------------------------

def test_the_stop_is_still_beyond_the_resistance():
    """The whole condition attached to the request. An early SHORT fills BELOW
    the resistance, so the stop must still sit ABOVE it, not above the entry."""
    plan = _plan(TURNED)
    assert plan.action == ACTION_ENTER
    assert plan.stop > 100.63, (
        f'stop {plan.stop:.6g} is not clear of the 100.63 resistance — an early '
        f'entry has moved the invalidation off the structure'
    )
    assert plan.stop > plan.entry


def test_the_early_entry_is_worse_than_the_level_and_that_is_priced_in():
    """A short filled at 100.0 instead of 100.63 is a worse entry with a wider
    stop. It must be measured, not hidden: risk is taken from the fill."""
    plan = _plan(TURNED)
    assert plan.entry == pytest.approx(100.0, abs=1e-9), 'filled at the market'
    risk_atr = abs(plan.stop - plan.entry) / 1.5
    assert risk_atr == pytest.approx(plan.risk_atr, abs=0.02), (
        'the published risk does not match the distance from the actual fill'
    )
    assert risk_atr <= TG.MAX_STOP_ATR


def test_a_trade_that_only_pays_at_the_level_is_refused_not_taken_early():
    """The floor still has to clear on the worse entry. Push the objective in
    until the early fill no longer pays and it must be REJECTED, not filled."""
    d = _short_below_resistance()
    plan = run(d, regime='TRENDING_BEAR', confirm=TURNED,
               levels=[(100.63, 4), (99.4, 4), (106.0, 3)])
    if plan.action == ACTION_ENTER:
        r_net = plan.r_net
        assert r_net >= TG.MIN_NET_R, (
            f'entered early at {r_net:.2f}R net, under the {TG.MIN_NET_R} floor'
        )
    else:
        assert plan.action == ACTION_REJECT


def test_beyond_reach_is_still_refused():
    """The turn does not extend REACH_ATR — a level 4 ATR away is still not a
    trade, however hard the 5m is printing."""
    d = mk(price=100.0, atr=1.5, support=94.0, resistance=112.0,
           rsi=55.0, **TURNED_DOWN)
    plan = run(d, regime='TRENDING_BEAR', confirm=TURNED,
               levels=[(112.0, 4), (94.0, 4)])
    assert plan.action == ACTION_REJECT


def test_the_turn_is_required_on_top_of_confirmation_not_instead_of_it():
    """A counter-trend setup still needs its two independent prints; the 5m turn
    is an extra requirement for filling early, never a bypass of _confirmation."""
    flat = mk(price=100.0, atr=1.5, support=94.0, resistance=102.3, rsi=55.0)
    flat['rsi_slope'] = 0.0                      # no curl, no rejection candle
    plan = run(flat, regime='RANGING', levels=LEVELS, confirm=TURNED)
    assert plan.action != ACTION_ENTER, (
        'a RANGE_FADE filled early on one print — the two-print rule for '
        'counter-trend entries has been bypassed'
    )


def test_the_runtime_knob_is_registered():
    import main
    assert 'early_entry_on_ltf' in main._TUNABLES
    mod, attr, kind = main._TUNABLES['early_entry_on_ltf'][:3]
    assert (mod, attr, kind) == ('src.trading.trader_gate',
                                 'EARLY_ENTRY_ON_LTF', 'bool')


def test_no_threshold_moved():
    assert TG.AT_LEVEL_ATR == 0.35
    assert TG.REACH_ATR == 2.50
    assert TG.WORK_EXPIRY_BARS == 8
    assert TG.MAX_STOP_ATR == 3.0
    assert TG.STOP_BUFFER_ATR > 0


# -- the fill must be CLOSE to the level -------------------------------------
# 2026-08-21: "instead of increasing the SL level we should increase the entry
# level... as close to resistance/support as possible".
#
# SAND/USDT filled 0.67 ATR short of its resistance and sat at -0.25% while the
# level it aimed at was +1.08% away. The trade was right about direction and lost
# money on the gap. Early entry originally reached to REACH_ATR (2.50 ATR), which
# is most of the move it was trying to sell.
#
# EARLY_ENTRY_MAX_ATR bounds the give-up to less than STOP_BUFFER_ATR, so the
# entry can never sit further from the level than the invalidation sits beyond
# it. Beyond that the order goes back to WORKING at the level.

def _at(dist_atr, atr=1.5):
    """A bear rallying toward resistance, `dist_atr` short of the level."""
    price = 100.0
    res = price + dist_atr * atr
    d = mk(price=price, atr=atr, support=price - 6.0, resistance=res,
           rsi=55.0, **TURNED_DOWN)
    return d, [(res, 4), (price - 6.0, 4), (res + 4.0, 3)]


def test_a_fill_close_to_the_level_is_taken():
    d, lv = _at(0.30)
    plan = run(d, regime='TRENDING_BEAR', levels=lv, confirm=TURNED)
    assert plan.action == ACTION_ENTER


def test_a_fill_far_from_the_level_goes_back_to_working_the_order():
    """The SAND shape: 0.67 ATR short of the level is too far to buy the gap."""
    d, lv = _at(0.67)
    plan = run(d, regime='TRENDING_BEAR', levels=lv, confirm=TURNED)
    assert plan.action == ACTION_WORK, (
        f'filled {0.67} ATR from the level ({plan.action}) — this is the gap that '
        f'made SAND lose money while being right about direction'
    )


@pytest.mark.parametrize('dist', [0.60, 1.00, 1.50, 2.40])
def test_everything_beyond_the_bound_rests_at_the_level(dist):
    d, lv = _at(dist)
    plan = run(d, regime='TRENDING_BEAR', levels=lv, confirm=TURNED)
    assert plan.action != ACTION_ENTER


def test_the_give_up_is_smaller_than_the_stop_buffer():
    """The invariant worth keeping: an early entry can never sit further from
    the level than the stop sits beyond it."""
    assert TG.EARLY_ENTRY_MAX_ATR < TG.STOP_BUFFER_ATR


def test_the_bound_is_well_inside_the_working_order_reach():
    assert TG.EARLY_ENTRY_MAX_ATR < TG.REACH_ATR


def test_the_stop_widening_was_reverted():
    """The entry fix replaces it — see the MAX_STOP_PCT comment."""
    assert TG.MAX_STOP_PCT == 1.30
