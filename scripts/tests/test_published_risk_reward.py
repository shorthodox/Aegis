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
from src.trading.trader_gate import MAX_STOP_PCT, MIN_STOP_PCT

# v87: the stop is clamped to a percent-of-entry band. Read the switch rather
# than assuming it, so these tests keep their meaning whichever way it is set.
_BAND_ON = MIN_STOP_PCT > 0 and MAX_STOP_PCT > 0


# ── the defect: a rung ratio is not a measurement ────────────────────────────

def test_tp2_ratio_is_a_constant_again_and_still_is_not_the_approved_payoff():
    """The rung ratio has been a constant, then arbitrary, and is now constant
    again for a THIRD reason. None of the three is the R:R.

    It was exactly 2.0 by construction when tp2 was price + 2.0R — a constant
    wearing the costume of a measurement. Pricing the ladder in percent of entry
    made it vary with whatever the stop happened to be. v87 then priced the STOP
    in percent too (TraderGate.MIN_STOP_PCT), and a fixed percent rung over a
    fixed percent stop is once again a pure constant: TP2 / MAX_STOP_PCT, the
    same number on every token at every price.

    That is worth knowing for a reason beyond reporting. It means the geometry no
    longer adapts to a token's volatility at all — BTC at 0.4% ATR and a
    small-cap at 2% ATR now get the identical stop in percent, which is ~1.75 ATR
    for one and ~0.35 ATR for the other. The ratio being stable is the visible
    symptom; the lost ATR adaptation is the thing to weigh.

    What the test still guards is unchanged: whatever this number is, it is not
    the figure a subscriber judges the trade by. plan.r_net is.
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
    if _BAND_ON:
        expect = DynamicRiskEngine.TP_LADDER_PCT[1] / MAX_STOP_PCT
        assert seen == {round(expect, 6)}, (
            f'with a percent stop the rung ratio must be exactly TP2/MAX_STOP_PCT '
            f'= {expect:.6f}; got {sorted(seen)[:4]}')
    else:
        assert len(seen) > 1, 'the rung ratio is a constant again'
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
