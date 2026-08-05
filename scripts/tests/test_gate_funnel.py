"""The gate must be able to fire. Pins the funnel open.

A signal drought does not announce itself — the engine keeps scanning, keeps
publishing HOLDs with reasons, and looks healthy. Measured over a 35,640-case
sweep of the conditions this fleet actually sees, the fire rate was ZERO:

    rejected at payoff    45.0%
    rejected at setup     44.6%
    rejected at location  10.4%
    reached ENTER or WORK  0.0%

The cause was an inconsistency between two constants that were tuned
separately. _pick_target took the NEAREST level at least MIN_TARGET_ATR (1.5)
away, while clearing MIN_NET_R (1.6) with a stop at the MIN_STOP_ATR floor
needs an objective roughly 2.7 ATR out. Every level between 1.5 and 2.7 ATR was
therefore selectable and then guaranteed to fail stage 3 — the gate picked the
2 ATR level, priced the trade at ~1.2R, rejected it, and never looked at the
3 ATR level behind it that would have paid.

These tests fail if the arithmetic drifts apart again, whichever constant moves.
"""
import itertools

import pytest

from src.trading import trader_gate as TG
from src.trading.trader_gate import TraderGate
from scripts.engine.models import RegimeState

PRICE = 100.0
MKT = dict(drift_blocked=False, drift_severity='OK', news_locked=False,
           news_label='', spread_pct=0.02, tide_dir='FLAT', tide_strength=0.0)
BOOK = dict(open_total=0, max_open=5, cluster_long=0, cluster_short=0,
            max_per_cluster=2)


def _ladder(atr, k=(1, 2, 3, 4, 6)):
    return [(PRICE + i * atr, 3) for i in k] + [(PRICE - i * atr, 3) for i in k]


def _scenario(rp, rsi, atr_pct, reg, conf, mw, md, cdl=1.5, ltf=True, slope=0.5):
    atr = PRICE * atr_pct / 100.0
    result = {
        'range_position': rp, 'rsi': rsi, 'atr': atr, 'atr_pct': atr_pct,
        'price': PRICE, 'p_buy': 0.53, 'p_sell': 0.47,
        'macro_weekly': mw, 'macro_daily': md,
        'resistance_broken_recent': False, 'support_broken_recent': False,
        'support': PRICE - 0.1 * atr, 'resistance': PRICE + 0.1 * atr,
        'cdl_bull_reversal': cdl, 'cdl_bear_reversal': cdl, 'rsi_slope': slope,
    }
    regime = RegimeState(reg, conf, True, [], 0.10)
    return TraderGate.evaluate(result, regime, market=MKT, book=BOOK,
                               levels=_ladder(atr),
                               confirm={'ltf_bull': ltf, 'ltf_bear': ltf})


# ── the constants must be consistent with each other ─────────────────────────

def test_min_target_atr_is_only_a_noise_floor_not_the_selector():
    """MIN_TARGET_ATR cannot on its own satisfy MIN_NET_R — so the selector
    must keep looking past it, not stop there."""
    atr_pct = 0.91                      # median 1h ATR for this fleet
    risk_pct = TG.MIN_STOP_ATR * atr_pct
    reward_pct = TG.MIN_TARGET_ATR * atr_pct
    r_net = ((reward_pct - TG.ROUND_TRIP_COST_PCT) /
             (risk_pct + TG.ROUND_TRIP_COST_PCT))
    assert r_net < TG.MIN_NET_R, (
        'MIN_TARGET_ATR now clears MIN_NET_R on its own; if that is intended, '
        'the reach-past logic in _pick_target is dead code and should go'
    )


def test_pick_target_reaches_past_objectives_that_cannot_pay():
    atr = 1.0
    risk = TG.MIN_STOP_ATR * atr        # the tightest stop the gate allows
    result = {'support': 0.0, 'resistance': 0.0}
    levels = _ladder(atr)
    tgt = TraderGate._pick_target('BUY', PRICE, atr, levels, result, risk)
    reward_pct = (tgt - PRICE) / PRICE * 100.0
    risk_pct = risk / PRICE * 100.0
    r_net = ((reward_pct - TG.ROUND_TRIP_COST_PCT) /
             (risk_pct + TG.ROUND_TRIP_COST_PCT))
    assert r_net >= TG.MIN_NET_R, (
        f'target {tgt} pays only {r_net:.2f}R — the selector stopped at the '
        f'first level past MIN_TARGET_ATR instead of the first that pays'
    )
    # and it must be the NEAREST qualifying level, not a fib fantasy
    assert tgt <= PRICE + 4 * atr


def test_pick_target_still_takes_the_nearest_when_risk_is_unknown():
    """Back-compat: callers that do not pass risk keep the old behaviour."""
    atr = 1.0
    tgt = TraderGate._pick_target('BUY', PRICE, atr, _ladder(atr),
                                  {'support': 0.0, 'resistance': 0.0})
    assert tgt == pytest.approx(PRICE + 2 * atr)


def test_pick_target_is_symmetric_for_shorts():
    atr = 1.0
    risk = TG.MIN_STOP_ATR * atr
    tgt = TraderGate._pick_target('SELL', PRICE, atr, _ladder(atr),
                                  {'support': 0.0, 'resistance': 0.0}, risk)
    assert tgt < PRICE
    reward_pct = (PRICE - tgt) / PRICE * 100.0
    risk_pct = risk / PRICE * 100.0
    r_net = ((reward_pct - TG.ROUND_TRIP_COST_PCT) /
             (risk_pct + TG.ROUND_TRIP_COST_PCT))
    assert r_net >= TG.MIN_NET_R


# ── the funnel as a whole ────────────────────────────────────────────────────

def _sweep():
    fired = total = 0
    stages = {}
    for rp, rsi, atr_pct, (reg, conf), (mw, md) in itertools.product(
            (0.05, 0.15, 0.25, 0.5, 0.75, 0.85, 0.95),
            (25, 32, 45, 55, 68, 75),
            (0.5, 0.91, 1.5),
            (('TRENDING_BULL', .9), ('TRENDING_BEAR', .9),
             ('RANGING', .6), ('VOLATILE_COMPRESSION', .7)),
            ((1, 1), (-1, -1), (-1, 1))):
        total += 1
        p = _scenario(rp, rsi, atr_pct, reg, conf, mw, md)
        if p.action in (TG.ACTION_ENTER, TG.ACTION_WORK):
            fired += 1
        else:
            stages[p.stage] = stages.get(p.stage, 0) + 1
    return total, fired, stages


def test_the_gate_can_actually_fire():
    total, fired, _ = _sweep()
    assert fired > 0, 'the gate fired on NOTHING — the funnel is closed'
    assert fired / total > 0.10, (
        f'fire rate collapsed to {fired/total*100:.1f}% of {total} scenarios'
    )


def test_payoff_is_no_longer_the_binding_constraint():
    """Payoff should reject setups that genuinely do not pay, not most of them."""
    total, _, stages = _sweep()
    payoff = stages.get('payoff', 0)
    assert payoff / total < 0.20, (
        f'payoff is rejecting {payoff/total*100:.1f}% of all scenarios — the '
        f'target selector and MIN_NET_R have drifted apart again'
    )


# ── confirmation must not treat missing data as evidence ─────────────────────

@pytest.mark.parametrize('sentinel', [-1.0, -1])
def test_unavailable_candle_data_is_not_a_confirmation(sentinel):
    """-1.0 means "could not look", not "saw a rejection candle"."""
    result = {'cdl_bull_reversal': sentinel, 'cdl_bear_reversal': sentinel,
              'rsi_slope': 0.0}
    ok, why = TraderGate._confirmation(result, 'BUY', TG.SETUP_RANGE_FADE,
                                       {'ltf_bull': False, 'ltf_bear': False})
    assert ok is False, f'missing candle data counted as confirmation: {why}'


def test_unavailable_data_cannot_confirm_both_directions_at_once():
    result = {'cdl_bull_reversal': -1.0, 'cdl_bear_reversal': -1.0,
              'rsi_slope': 0.0}
    confirm = {'ltf_bull': False, 'ltf_bear': False}
    buy_ok, _ = TraderGate._confirmation(result, 'BUY', TG.SETUP_RANGE_FADE, confirm)
    sell_ok, _ = TraderGate._confirmation(result, 'SELL', TG.SETUP_RANGE_FADE, confirm)
    assert not (buy_ok and sell_ok)


def test_a_real_pattern_still_confirms():
    result = {'cdl_bull_reversal': 2.0, 'cdl_bear_reversal': 0.0, 'rsi_slope': 0.4}
    ok, why = TraderGate._confirmation(result, 'BUY', TG.SETUP_RANGE_FADE,
                                       {'ltf_bull': False, 'ltf_bear': False})
    assert ok is True, why
