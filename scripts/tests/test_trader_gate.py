"""
test_trader_gate.py — the desk's playbook, stage by stage.

The anchor case is the real one: on 2026-07-20 the engine opened eight alt
SHORTs inside a 55-minute window (SUI, VET, ATOM, APT, ETC, XLM, ICP, AVAX),
every closed one hit its stop, and model confidence on the losers ranged 17.9
to 100.0.  `test_the_losing_basket_*` reconstructs that tape and asserts the new
gate refuses it — at three independent stages, so no single tuning change can
let it back through.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.trading.trader_gate import (  # noqa: E402
    ACTION_ENTER, ACTION_REJECT, ACTION_WORK,
    SETUP_BREAK_RETEST, SETUP_EXHAUSTION_REVERSAL, SETUP_NONE,
    SETUP_RANGE_FADE, SETUP_TREND_PULLBACK,
    MAX_STOP_ATR, MIN_NET_R, MIN_STOP_ATR, WORK_EXPIRY_BARS,
    TraderGate,
)
from src.trading import trader_gate as TG  # noqa: E402

# v87 percent budget band — read the switch, do not assume it
_BAND_ON = TG.MIN_STOP_PCT > 0 and TG.MAX_STOP_PCT > 0


class Regime:
    def __init__(self, name, conf=0.75):
        self.regime = name
        self.confidence = conf


def mk(price=100.0, atr=1.0, support=99.8, resistance=110.0, rsi=50.0,
       regime_conf=0.75, **kw):
    """A signal dict with the fields the gate actually reads."""
    rp = ((price - support) / (resistance - support)) if resistance > support else 0.5
    d = {
        'price': price, 'entry_price': price,
        'atr': atr, 'atr_pct': atr / price * 100.0,
        'support': support, 'resistance': resistance,
        'range_position': max(0.0, min(1.0, rp)),
        'rsi': rsi, 'rsi_slope': 0.0,
        'p_buy': 0.34, 'p_sell': 0.33, 'p_hold': 0.33,
        'volume_zscore': 0.2, 'relative_volume': 1.1,
    }
    d.update(kw)
    return d


FLAT_MARKET = {'tide_dir': 'FLAT', 'tide_strength': 0.0}
EMPTY_BOOK = {'open_total': 0, 'max_open': 5, 'cluster_long': 0, 'cluster_short': 0, 'max_per_cluster': 2}
BOTH_CONFIRM = {'ltf_bull': True, 'ltf_bear': True}

# Result-side confirmation prints.  A counter-trend setup needs TWO independent
# ones, so tests that expect an immediate ENTER on a fade must supply them; the
# 5m alignment in BOTH_CONFIRM is only ever one of the two.
TURNED_UP = {'cdl_bull_reversal': True, 'rsi_slope': 0.4}
TURNED_DOWN = {'cdl_bear_reversal': True, 'rsi_slope': -0.4}


def run(result, regime='RANGING', market=None, book=None, levels=None,
        confirm=None, regime_conf=0.75):
    return TraderGate.evaluate(
        result, Regime(regime, regime_conf),
        dict(FLAT_MARKET, **(market or {})),
        dict(EMPTY_BOOK, **(book or {})),
        levels if levels is not None else [],
        confirm if confirm is not None else dict(BOTH_CONFIRM),
    )


# ═══════════════════════════════════════════════════════════════════════════
# The losing basket
# ═══════════════════════════════════════════════════════════════════════════

def test_the_losing_basket_is_refused_as_a_setup():
    """Shorting a rally mid-range is not a setup at all.

    The 8 shorts were counter-trend entries taken because the model liked them,
    at locations that were merely 'high-ish' rather than stretched.  Stage 1
    refuses them before any threshold is consulted.
    """
    mid = run(mk(price=106.0, support=99.8, resistance=110.0, rsi=60.0),
              regime='TRENDING_BULL')
    assert mid.action == ACTION_REJECT
    assert mid.stage == 'setup'
    assert mid.setup == SETUP_NONE
    assert 'mid-range' in mid.reason

    # ...and the same short pushed right up to the highs, still unstretched:
    # being high in the range is not the same as being exhausted.
    high = run(mk(price=109.5, support=99.8, resistance=110.0, rsi=60.0),
               regime='TRENDING_BULL')
    assert high.action == ACTION_REJECT
    assert high.setup == SETUP_NONE
    assert 'not stretched' in high.reason


def test_the_losing_basket_is_refused_by_the_tide_even_when_stretched():
    """Even a textbook-stretched short is refused against a strong BTC tide.

    This is the second, independent stop: the basket's defining feature was that
    every one of them fought the same rising tape.
    """
    plan = run(mk(price=109.5, support=99.8, resistance=110.0, rsi=75.0),
               regime='TRENDING_BULL',
               market={'tide_dir': 'UP', 'tide_strength': 0.8},
               levels=[(110.0, 4), (99.8, 3), (95.0, 3)])
    assert plan.action == ACTION_REJECT
    assert plan.stage == 'allocation'
    assert 'tide' in plan.reason


def test_the_losing_basket_is_refused_by_correlation():
    """The third stop: two correlated shorts already open ends the cluster."""
    plan = run(mk(price=109.5, support=99.8, resistance=110.0, rsi=75.0),
               regime='TRENDING_BULL',
               book={'cluster_short': 2},
               levels=[(110.0, 4), (99.8, 3), (95.0, 3)])
    assert plan.action == ACTION_REJECT
    assert plan.stage == 'allocation'
    assert 'one thesis' in plan.reason


def test_high_model_confidence_cannot_buy_permission():
    """Conviction is not permission — the losers' confidence ran to 100."""
    hot = mk(price=108.0, support=99.8, resistance=110.0, rsi=60.0,
             p_sell=0.95, p_buy=0.02, edge_score=100.0, meta_confidence=1.0,
             fire=True, side='SELL')
    plan = run(hot, regime='TRENDING_BULL')
    assert plan.action == ACTION_REJECT, 'a 100-confidence model bought its way in'


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1 · setup classification
# ═══════════════════════════════════════════════════════════════════════════

def test_bull_pullback_to_support_is_a_long():
    plan = run(mk(price=102.0, support=99.8, resistance=110.0), regime='TRENDING_BULL',
               levels=[(101.9, 4), (110.0, 3)])
    assert plan.setup == SETUP_TREND_PULLBACK
    assert plan.side == 'BUY'


def test_bear_rally_into_resistance_is_a_short():
    plan = run(mk(price=108.0, support=99.8, resistance=110.0), regime='TRENDING_BEAR',
               levels=[(108.1, 4), (99.8, 3)])
    assert plan.setup == SETUP_TREND_PULLBACK
    assert plan.side == 'SELL'


def test_range_edge_is_a_fade_only_at_the_edge():
    at_edge = run(mk(price=100.0, support=99.8, resistance=110.0), regime='RANGING',
                  levels=[(99.8, 4), (110.0, 4)])
    assert at_edge.setup == SETUP_RANGE_FADE and at_edge.side == 'BUY'

    mid = run(mk(price=105.0, support=99.8, resistance=110.0), regime='RANGING')
    assert mid.setup == SETUP_NONE
    assert mid.stage == 'setup'


def test_exhaustion_reversal_needs_the_rsi_stretch_not_just_the_location():
    loc_only = run(mk(price=109.5, support=99.8, resistance=110.0, rsi=55.0),
                   regime='TRENDING_BULL')
    assert loc_only.setup == SETUP_NONE, 'faded a bull without exhaustion'
    # The refusal must say what is actually wrong. "mid-range" at rp 0.98 is a
    # lie the user reads on the signal card.
    assert 'not stretched' in loc_only.reason
    assert 'mid-range' not in loc_only.reason

    stretched = run(mk(price=109.5, support=99.8, resistance=110.0, rsi=75.0),
                    regime='TRENDING_BULL', levels=[(110.0, 4), (99.8, 3)])
    assert stretched.setup == SETUP_EXHAUSTION_REVERSAL
    assert stretched.side == 'SELL'


def test_broken_resistance_retested_is_no_longer_a_long():
    """BREAK_RETEST is retired — it was counter-location by construction.

    This used to assert the retest of a broken resistance produced a BUY. The
    geometry here is rp ~0.90, i.e. a long at the top of its range, which is the
    trade this desk does not take. The setup bought at rp >= 0.70 and sold at
    rp <= 0.30 in every case, and conditioning it on the higher timeframe did
    not help: in a TRENDING_BEAR the weekly agrees WITH a short, so OP/USDT and
    SUI/USDT were still sold at the BOTTOM of their range on 2026-08-07.

    A long is taken at support, a short at resistance. See
    scripts/tests/test_location_vs_htf.py.
    """
    plan = run(mk(price=109.0, support=99.8, resistance=110.0,
                  resistance_broken_recent=True),
               regime='RANGING', levels=[(108.9, 5), (120.0, 3)])
    assert plan.setup != SETUP_BREAK_RETEST
    assert plan.side != 'BUY', 'a long at the top of the range is back'


def test_a_low_confidence_trend_label_is_treated_as_a_range():
    """'TRENDING_BULL at 0.31 confidence' is a guess, not a trend.

    Under the old system that label authorised continuation trades; here it
    falls through to rangebound handling, so a mid-range location yields nothing.
    """
    plan = run(mk(price=105.0, support=99.8, resistance=110.0),
               regime='TRENDING_BULL', regime_conf=0.31)
    assert plan.setup == SETUP_NONE
    assert 'no trend' in plan.reason


def test_model_leaning_hard_the_other_way_vetoes_the_structure():
    plan = run(mk(price=100.0, support=99.8, resistance=110.0,
                  p_buy=0.05, p_sell=0.80),
               regime='RANGING', levels=[(99.8, 4), (110.0, 4)])
    assert plan.action == ACTION_REJECT
    assert plan.stage == 'setup'
    assert 'disagree' in plan.reason


def test_a_neutral_model_does_not_block_a_good_setup():
    plan = run(mk(price=100.0, support=99.8, resistance=110.0,
                  p_buy=0.33, p_sell=0.34, **TURNED_UP),
               regime='RANGING', levels=[(99.8, 4), (110.0, 4)])
    assert plan.action == ACTION_ENTER


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2 · invalidation
# ═══════════════════════════════════════════════════════════════════════════

def test_stop_sits_beyond_the_level_never_on_it():
    plan = run(mk(price=100.0, support=99.8, resistance=110.0, **TURNED_UP),
               regime='RANGING', levels=[(99.8, 4), (110.0, 4)])
    assert plan.action == ACTION_ENTER
    assert plan.stop < plan.level, 'stop was parked on or above the level it defends'


def test_a_stop_inside_the_noise_band_is_widened_not_accepted():
    """The basket died on ~1.1% stops in ~1% ATR tape — one bar of noise.

    v87: MIN_STOP_ATR still governs the INVALIDATION, which is what stage 3 prices
    the trade on, so the widening it performs is asserted here as before. What it
    no longer governs is the stop actually placed: the percent budget band is
    applied last and, by the user's explicit choice, is allowed to pull that stop
    back inside the noise band this floor exists to escape. The two numbers are
    asserted separately because they now answer different questions.
    """
    plan = run(mk(price=100.0, atr=1.0, support=99.95, resistance=110.0, **TURNED_UP),
               regime='RANGING', levels=[(99.95, 4), (110.0, 4)])
    assert plan.action == ACTION_ENTER
    # the invalidation is still pushed out to the noise floor
    assert abs(plan.entry - plan.invalidation) / 1.0 >= MIN_STOP_ATR - 1e-9

    if _BAND_ON:
        # ...and the placed stop is the budget, which here is INSIDE that floor
        risk_pct = abs(plan.entry - plan.stop) / plan.entry * 100.0
        assert TG.MIN_STOP_PCT - 1e-9 <= risk_pct <= TG.MAX_STOP_PCT + 1e-9
        assert plan.risk_atr < MIN_STOP_ATR, (
            'the band is on but the placed stop still sits outside the noise '
            'floor — check MAX_STOP_PCT against the fleet ATR')
    else:
        assert plan.risk_atr >= MIN_STOP_ATR - 1e-9


def test_the_stop_clears_a_second_level_it_was_not_derived_from():
    """The real ETC/USDT short: SL 6.6692 under a 6.670 resistance.

    `_pick_level` took the nearer 6.645 shelf, the noise-band floor then re-derived
    the stop from price alone, and it landed 0.0008 UNDER the heavier level — the
    one price was always going to overshoot on a reversal or retest from above on a
    break.  The stop has to finish on the far side of BOTH.
    """
    r = mk(price=6.63, atr=0.0438, support=6.41, resistance=6.67, rsi=70.6,
           **TURNED_DOWN)
    plan = run(r, regime='RANGING', levels=[(6.645, 3), (6.67, 4), (6.41, 4)])
    assert plan.action in (ACTION_ENTER, ACTION_WORK)
    assert plan.stop > 6.67, f'stop {plan.stop:.4f} is parked under the 6.67 resistance'


def test_a_far_level_does_not_drag_the_stop_out():
    """Clearing is bounded — only levels the stop actually leans on move it.

    Otherwise a resistance 3 ATR overhead would blow every short's stop past
    MAX_STOP_ATR and the setup would be rejected for structure it never used.
    """
    plan = run(mk(price=100.0, atr=1.0, support=90.0, resistance=103.0, rsi=72.0,
                  **TURNED_DOWN),
               regime='RANGING', levels=[(100.2, 3), (103.0, 4), (90.0, 4)])
    assert plan.action in (ACTION_ENTER, ACTION_WORK)
    assert plan.stop < 103.0, 'a level 3 ATR away should not anchor the stop'
    assert plan.risk_atr <= MAX_STOP_ATR


def test_an_absurdly_distant_invalidation_is_refused():
    plan = run(mk(price=100.0, atr=0.2, support=99.0, resistance=110.0),
               regime='RANGING', levels=[(99.0, 4), (110.0, 4)])
    assert plan.action == ACTION_REJECT
    assert plan.stage == 'invalidation'
    assert plan.risk_atr > MAX_STOP_ATR or 'too far' in plan.reason


def test_no_level_to_lean_on_is_refused():
    r = mk(price=100.0, support=99.8, resistance=110.0)
    r['support'] = 0.0
    r['resistance'] = 0.0
    r['range_position'] = 0.1
    plan = run(r, regime='RANGING', levels=[])
    assert plan.action == ACTION_REJECT
    assert plan.stage == 'invalidation'


# ═══════════════════════════════════════════════════════════════════════════
# Stage 3 · payoff
# ═══════════════════════════════════════════════════════════════════════════

def test_a_cramped_setup_is_refused_however_good_it_looks():
    """Target too close for the risk — the stage the old system never had."""
    plan = run(mk(price=100.0, atr=1.0, support=99.4, resistance=102.0,
                  edge_score=95.0, fire=True),
               regime='RANGING', levels=[(99.4, 5), (102.0, 5)])
    assert plan.action == ACTION_REJECT
    assert plan.stage == 'payoff'
    assert plan.r_net < MIN_NET_R or 'does not pay' in plan.reason


def test_payoff_is_measured_net_of_costs():
    plan = run(mk(price=100.0, support=99.8, resistance=110.0), regime='RANGING',
               levels=[(99.8, 4), (110.0, 4)])
    assert plan.r_net < plan.r_gross, 'costs were not charged against the payoff'
    assert plan.r_net >= MIN_NET_R


def test_a_noise_level_cannot_serve_as_the_objective():
    """A level 0.3 ATR ahead is not somewhere you get paid."""
    plan = run(mk(price=100.0, atr=1.0, support=99.8, resistance=100.3),
               regime='RANGING', levels=[(99.8, 4), (100.3, 4)])
    assert plan.action == ACTION_REJECT
    assert plan.stage in ('payoff', 'setup')


# ═══════════════════════════════════════════════════════════════════════════
# Stage 4 · trigger — what replaces PENDING
# ═══════════════════════════════════════════════════════════════════════════

def test_away_from_the_level_becomes_a_working_order_with_a_clock():
    plan = run(mk(price=101.2, atr=1.0, support=99.8, resistance=112.0),
               regime='RANGING', levels=[(99.8, 4), (112.0, 4)])
    assert plan.action == ACTION_WORK
    assert plan.expiry_bars == WORK_EXPIRY_BARS, 'a working order without a clock is PENDING again'
    assert plan.invalidation > 0
    assert plan.entry == pytest.approx(plan.level), 'working order must rest AT the level'


def test_out_of_reach_is_dropped_not_queued():
    plan = run(mk(price=106.0, atr=1.0, support=99.8, resistance=120.0),
               regime='RANGING')
    assert plan.action == ACTION_REJECT
    assert plan.stage in ('setup', 'trigger')


def test_a_lost_level_is_not_re_leaned_on():
    """Price has dropped clean through its support.

    A trader does not buy the level that just failed — they look for the next
    one down, and if there is none they have no trade.  The gate reaches the
    same place structurally: `_pick_level` only returns levels on the correct
    side of price, so the failed support is no longer a candidate.
    """
    plan = run(mk(price=99.0, atr=1.0, support=99.8, resistance=110.0, **TURNED_UP),
               regime='RANGING', levels=[(99.8, 4), (110.0, 4)])
    assert plan.action == ACTION_REJECT
    assert plan.stage == 'invalidation'

    # ...and with a genuine next level down, that one becomes the thesis.
    lower = run(mk(price=99.0, atr=1.0, support=99.8, resistance=110.0, **TURNED_UP),
                regime='RANGING', levels=[(99.8, 4), (97.5, 4), (110.0, 4)])
    assert lower.level == pytest.approx(97.5)
    assert lower.stop < 97.5


def test_a_fade_needs_two_confirmations_a_pullback_needs_one():
    """The eight shorts were all 'at resistance'; none had turned."""
    one = {'ltf_bull': True, 'ltf_bear': False}
    fade = run(mk(price=100.0, support=99.8, resistance=110.0, rsi_slope=0.0),
               regime='RANGING', levels=[(99.8, 4), (110.0, 4)], confirm=one)
    assert fade.action == ACTION_WORK, 'a counter-trend fade entered on a single print'

    pull = run(mk(price=102.0, support=99.8, resistance=110.0, rsi_slope=0.0),
               regime='TRENDING_BULL', levels=[(101.9, 4), (110.0, 4)], confirm=one)
    assert pull.action == ACTION_ENTER


# ═══════════════════════════════════════════════════════════════════════════
# Stage 5 · allocation
# ═══════════════════════════════════════════════════════════════════════════

def test_setup_class_sets_the_base_size():
    pull = run(mk(price=102.0, support=99.8, resistance=110.0), regime='TRENDING_BULL',
               levels=[(101.9, 4), (110.0, 4)])
    fade = run(mk(price=100.0, support=99.8, resistance=110.0), regime='RANGING',
               levels=[(99.8, 4), (110.0, 4)])
    assert pull.size_factor > fade.size_factor, \
        'the measured-negative setup was not sized below the measured-positive one'


def test_a_full_book_refuses_the_sixth_setup():
    plan = run(mk(price=100.0, support=99.8, resistance=110.0), regime='RANGING',
               book={'open_total': 5}, levels=[(99.8, 4), (110.0, 4)])
    assert plan.action == ACTION_REJECT
    assert plan.stage == 'allocation'


def test_second_correlated_position_is_scaled_down():
    solo = run(mk(price=100.0, support=99.8, resistance=110.0, **TURNED_UP),
               regime='RANGING', levels=[(99.8, 4), (110.0, 4)])
    second = run(mk(price=100.0, support=99.8, resistance=110.0, **TURNED_UP),
                 regime='RANGING', book={'cluster_long': 1},
                 levels=[(99.8, 4), (110.0, 4)])
    assert second.action == ACTION_ENTER
    assert second.size_factor < solo.size_factor


# ═══════════════════════════════════════════════════════════════════════════
# Stage 0 · fitness
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize('market,fragment', [
    ({'drift_blocked': True}, 'drift'),
    ({'news_locked': True, 'news_label': 'FOMC'}, 'FOMC'),
    ({'spread_pct': 0.9}, 'spread'),
    ({'atr_normal_pct': 0.3}, 'shock'),   # 1.0% ATR against a 0.3% normal = 3.3x
])
def test_unfit_markets_are_refused_before_anything_else(market, fragment):
    plan = run(mk(price=100.0, support=99.8, resistance=110.0), regime='RANGING',
               market=market, levels=[(99.8, 4), (110.0, 4)])
    assert plan.action == ACTION_REJECT
    assert plan.stage == 'fitness'
    assert fragment in plan.reason


def test_a_liquidity_trap_is_never_traded():
    plan = run(mk(price=100.0, support=99.8, resistance=110.0), regime='LIQUIDITY_TRAP',
               levels=[(99.8, 4), (110.0, 4)])
    assert plan.action == ACTION_REJECT
    assert plan.stage == 'fitness'


def test_a_calm_but_liquid_market_is_still_tradeable():
    """Low volatility is not a dead market — that veto starved the old engine."""
    plan = run(mk(price=100.0, atr=0.5, support=99.9, resistance=110.0),
               regime='RANGING', levels=[(99.9, 4), (110.0, 4)])
    assert plan.stage != 'fitness'


# ═══════════════════════════════════════════════════════════════════════════
# Contract
# ═══════════════════════════════════════════════════════════════════════════

def test_every_plan_carries_an_audit_trail():
    for plan in (
        run(mk(price=100.0, support=99.8, resistance=110.0), regime='RANGING',
            levels=[(99.8, 4), (110.0, 4)]),
        run(mk(price=105.0, support=99.8, resistance=110.0), regime='RANGING'),
    ):
        assert plan.reason, 'a verdict with no reason is the old system'
        assert plan.notes
        assert plan.stage
        assert isinstance(plan.as_dict(), dict)


def test_an_entered_plan_is_internally_consistent():
    plan = run(mk(price=100.0, support=99.8, resistance=110.0, **TURNED_UP),
               regime='RANGING', levels=[(99.8, 4), (110.0, 4)])
    assert plan.action == ACTION_ENTER
    assert plan.side == 'BUY'
    assert plan.stop < plan.entry < plan.target, 'geometry is inverted'
    assert 0.0 < plan.size_factor <= 1.0
    assert plan.fired and not plan.working


def test_a_short_plan_is_internally_consistent():
    plan = run(mk(price=108.0, support=99.8, resistance=110.0), regime='TRENDING_BEAR',
               levels=[(108.1, 4), (99.8, 4), (95.0, 3)])
    assert plan.action in (ACTION_ENTER, ACTION_WORK)
    assert plan.side == 'SELL'
    assert plan.stop > plan.entry > plan.target, 'geometry is inverted'
