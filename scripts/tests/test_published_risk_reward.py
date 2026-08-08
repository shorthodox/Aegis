"""The headline Risk/Reward must be the number the trade was approved on.

A subscriber reads one number to judge a signal. It was not the number the desk
cleared the trade with:

  * _build_signal_entry quotes |price - tp2| / risk. calculate_stops sets
    tp2 = price + 2.0R, so that ratio is 2.00 BY CONSTRUCTION whenever the
    ladder is not compressed — a constant wearing the costume of a measurement.
  * When the plan's objective sits between MIN_NET_R (1.60) and 2.0R, the
    v85 compression branch pulls tp2 back to 2/3 of the span. The published
    figure then lands BELOW the floor the gate had just enforced: a trade
    approved at 2.5R net could advertise 1.18.
  * On an ENTER the whole level set is republished from the position, because
    _build_signal_entry's values can belong to the opposite direction. risk_reward
    was the one field left out of that republish.

TradePlan.r_net already carries the honest figure — reward and risk to the real
objective, round trip taken off the win and added to the loss. These tests pin
that it is what gets published.
"""
import inspect
import random

import pytest

from scripts.engine.risk import DynamicRiskEngine
from src.trading import trader_gate as TG


# ── the defect: a rung ratio is not a measurement ────────────────────────────

def test_tp2_ratio_is_no_longer_a_constant_but_is_still_not_the_approved_payoff():
    """The rung ratio was a constant; now it is arbitrary. Neither is the R:R.

    It used to be exactly 2.0 by construction, because tp2 was price + 2.0R — a
    constant wearing the costume of a measurement. The ladder is priced in
    percent of entry now, so the ratio varies with whatever the stop happens to
    be, which is not an improvement for reporting: it still is not the number
    the gate approved the trade on. Either way the headline R:R must come from
    plan.r_net, which is what the tests below assert.
    """
    r = DynamicRiskEngine()
    rng = random.Random(5)
    seen = set()
    for _ in range(200):
        price = rng.uniform(0.01, 60_000)
        atr = price * rng.uniform(0.003, 0.05)
        for side in ('BUY', 'SELL'):
            out = r.calculate_stops(price, side, atr,
                                    support=price * 0.90, resistance=price * 1.10)
            if out['risk'] <= 0 or not out['tp2']:
                continue
            seen.add(round(abs(price - out['tp2']) / out['risk'], 6))
    assert len(seen) > 1, 'the rung ratio is a constant again'
    # It now varies with the stop, which is the point: the same rung reports a
    # different "R:R" per token, so it can never be the figure a subscriber
    # judges the trade by. plan.r_net is.
    assert max(seen) - min(seen) > 0.1, (
        f'the rung ratio barely moves ({sorted(seen)[:3]}) — if it has become '
        f'stable again, check whether it is being published as the R:R'
    )


def test_tp2_ratio_can_fall_below_the_floor_the_gate_enforced():
    """The compressed case: published < MIN_NET_R while the gate cleared it."""
    r = DynamicRiskEngine()
    price, atr = 100.0, 1.0
    risk = 2.0 * atr                       # inside [MIN_STOP_ATR, MAX_STOP_ATR]
    target_R = 1.7                         # clears MIN_NET_R once costs are applied
    out = r.calculate_stops(
        price, 'BUY', atr,
        support=price * 0.90, resistance=price * 1.30,
        sl_override=price - risk, tp_override=price + target_R * risk,
    )
    published = abs(price - out['tp2']) / out['risk']
    assert published < TG.MIN_NET_R, (
        f'expected the rung-derived ratio ({published:.2f}) to understate the '
        f'objective; the compression branch may have changed'
    )


# ── the fix ──────────────────────────────────────────────────────────────────

def _enter_block() -> str:
    """Source of the ENTER branch that republishes the position's levels."""
    src = inspect.getsource(
        __import__('scripts.engine.engine', fromlist=['LiveEngine']).LiveEngine
        ._run_trader_gate)
    return src[src.index('# ── ENTER'):]


def test_headline_rr_is_published_from_the_plan():
    block = _enter_block()
    assert "sig['risk_reward']" in block, (
        'the ENTER branch republishes every other level from the position but '
        'not risk_reward — it will keep whatever _build_signal_entry derived '
        "from the model's side, which may be the opposite direction"
    )
    assert 'plan.r_net' in block, 'the headline R:R is not the approved figure'


def test_ladder_ratios_are_kept_but_named_honestly():
    block = _enter_block()
    for key in ("sig['rr_to_tp2']", "sig['rr_to_tp5']"):
        assert key in block, f'{key} missing — the ladder figures should stay visible'
    assert "sig['risk_reward_gross']" in block


def test_plan_carries_a_cost_adjusted_number():
    """r_net must actually net the round trip, not just rename r_gross."""
    fields = {f for f in TG.TradePlan.__dataclass_fields__}
    assert {'r_net', 'r_gross'} <= fields
    src = inspect.getsource(TG.TraderGate.evaluate)
    assert 'ROUND_TRIP_COST_PCT' in src, 'the payoff stage stopped pricing costs'
    # costs off the win, onto the loss
    assert 'reward_pct - ROUND_TRIP_COST_PCT' in src
    assert 'risk_pct + ROUND_TRIP_COST_PCT' in src


def test_cost_constant_matches_the_wallet():
    """A gate that prices costs differently from the book reports a fake edge."""
    from scripts.live_engine import VirtualWallet
    assert TG.ROUND_TRIP_COST_PCT == pytest.approx(
        VirtualWallet.round_trip_cost_pct()), (
        'trader_gate and VirtualWallet disagree on the round-trip cost — the '
        'gate would approve trades the book then cannot pay for'
    )
