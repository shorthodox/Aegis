"""The risk budget must not pull a stop back under a level it was placed beyond.

AAVE/USDT, 2026-08-20 — a STRONG SELL, quality 72, that lost:

    entry 98.79   published stop 100.07
    resistances nearby: 98.69 (entered against) and 100.46

_clear_levels did its job and placed the stop at ~101.07, beyond the FAR
resistance. Stage 5b's risk-budget band then pulled it to 100.0743 — the
MAX_STOP_PCT cap to four decimal places — which sits 0.39 BELOW the 100.46 level.
Price ran up to test that level, collected the stop, and then fell as the setup
had said it would.

Reported exactly: "when there are more than 1 resistance nearby, entry is made
according to the nearer one but if the second resistance is getting hit then sl
is booked."

The band's own comment argued a tightened stop leaves the trade's ratio "better".
It does not. It makes the loss smaller and far more likely — a worse trade
wearing a flattering number.

Risk is size x distance. When structure needs a wider stop the honest lever is
SIZE: keep the stop where the thesis dies, scale the position by the ratio the
budget was exceeded by, and the dollar risk is identical with the stop out of the
traffic. Below MIN_SIZE_FACTOR the setup is unaffordable and is refused rather
than fitted with a decorative stop.

This is the same failure as the historical "0.700% stop on ten different tokens":
a budget silently overwriting structure.
"""
import pytest

from src.trading import trader_gate as TG
from src.trading.trader_gate import ACTION_REJECT

from scripts.tests.test_trader_gate import mk, run, TURNED_DOWN


def _aave(**kw):
    """The reported trade's GEOMETRY: two resistances with the far one outside
    the percent budget.

    RSI is 70 rather than the card's 62.7, and the regime RANGING rather than
    TRENDING_BULL, purely so the setup classifies and the plan reaches stage 5b —
    which is the stage under test. At the live values stage 1 refuses it first
    ("a bull at the highs is a trend working, not a top") and every assertion
    below would skip while appearing to pass. The prices, ATR and level spacing
    are the reported ones untouched.
    """
    d = mk(price=98.65, atr=1.1064, support=88.09, resistance=98.69,
           rsi=70.0, **TURNED_DOWN)
    d.update(kw)
    return d


AAVE_LEVELS = [(98.69, 4), (100.46, 4), (88.09, 3)]
FAR_RES = 100.46


def _plan(**kw):
    p = run(_aave(), regime='RANGING', levels=AAVE_LEVELS, **kw)
    assert p.action != ACTION_REJECT, (
        f'fixture no longer reaches stage 5b ({p.stage}: {p.reason}) — these '
        f'tests would silently stop covering the band'
    )
    return p


def test_the_stop_clears_the_far_resistance():
    plan = _plan()
    assert plan.stop > FAR_RES, (
        f'stop {plan.stop:.4f} sits under the 100.46 resistance — the move that '
        f'tests that level collects it, which is the reported loss'
    )


def test_the_stop_is_not_simply_the_percent_cap():
    """The signature of the bug: a stop exactly equal to MAX_STOP_PCT."""
    plan = _plan()
    cap = plan.entry * (1 + TG.MAX_STOP_PCT / 100.0)
    assert abs(plan.stop - cap) > 1e-3, (
        'the stop is the percent cap again — structure was overwritten by budget'
    )


def test_size_pays_for_the_wider_stop():
    """Dollar risk must stay inside the budget; only the lever changes."""
    plan = _plan()
    risk_pct = abs(plan.stop - plan.entry) / plan.entry * 100.0
    assert risk_pct > TG.MAX_STOP_PCT, 'precondition: this stop exceeds the budget'
    budgeted = risk_pct * plan.size_factor
    assert budgeted <= TG.MAX_STOP_PCT * 1.02, (
        f'size {plan.size_factor:.3f} x risk {risk_pct:.2f}% = {budgeted:.2f}% '
        f'exceeds the {TG.MAX_STOP_PCT}% budget — the wider stop was not paid for'
    )


def test_the_reasoning_is_recorded():
    plan = _plan()
    assert any('NOT tightening' in n for n in plan.notes), (
        'the decision to keep the structural stop is invisible in the plan'
    )


# ── the helper, directly ─────────────────────────────────────────────────────

def test_it_finds_a_level_between_the_two_stops():
    lv = TG.TraderGate._level_between('SELL', 100.07, 101.07,
                                      [(100.46, 4)], {'resistance': 98.69})
    assert lv == 100.46


def test_it_returns_nothing_when_tightening_is_safe():
    """No level in the gap means the budget may tighten freely."""
    lv = TG.TraderGate._level_between('SELL', 100.07, 101.07,
                                      [(103.0, 4)], {'resistance': 98.69})
    assert lv == 0.0


def test_it_mirrors_for_a_long():
    """A BUY's stop sits below price; tightening RAISES it past support."""
    lv = TG.TraderGate._level_between('BUY', 99.0, 97.0,
                                      [(98.0, 4)], {'support': 99.8})
    assert lv == 98.0


def test_an_unaffordable_setup_is_refused_not_fudged():
    """A stop so far out that budgeted size falls under MIN_SIZE_FACTOR must be
    declined — not given a tighter stop it cannot defend."""
    far = 98.65 * (1 + TG.MAX_STOP_PCT / 100.0 * 8)     # absurdly distant level
    plan = run(_aave(), regime='RANGING',
               levels=[(98.69, 4), (far, 4), (88.09, 3)])
    if plan.action == ACTION_REJECT:
        return                                           # refused: correct
    risk_pct = abs(plan.stop - plan.entry) / plan.entry * 100.0
    assert risk_pct * plan.size_factor <= TG.MAX_STOP_PCT * 1.02
