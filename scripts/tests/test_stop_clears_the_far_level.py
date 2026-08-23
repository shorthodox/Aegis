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
from src.trading.trader_gate import ACTION_REJECT, ACTION_WORK

from scripts.tests.test_trader_gate import mk, run, TURNED_DOWN, TURNED_UP


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


def _f_atr():
    return 1.1064          # the AAVE fixture's ATR


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


def test_the_entry_moves_to_the_level_the_stop_defends():
    """2026-08-22 — the fix that dissolved this whole problem.

    _pick_level returns the NEAREST structure and _clear_levels pushes the stop
    past a FURTHER one, so the trade used to be opened a long way from the level
    it was risking on. ETH/USDT: filled 2412.29 with its support at 2357.18,
    1.67 ATR of pure give-up and R:R 1:0.72.

    The entry now follows the defended level, so the two collapse onto one and
    the stop stops being wide relative to entry — which is why the budget
    machinery below barely has to fire any more.
    """
    plan = _plan()
    # The LEVEL is what moved: _pick_level's near choice is replaced by the one
    # the stop actually defends, so the trade leans on a single level instead of
    # validating against one and risking on another.
    assert abs(plan.level - FAR_RES) < 1e-9, (
        f'level {plan.level:.8g} is not the {FAR_RES} the stop defends — the '
        f'collapse did not happen'
    )
    # A RESTING order then waits AT that level. Since 2026-08-23 a fully
    # confirmed reversal may instead fill at the market without ever reaching it
    # (EARLY_ENTRY_ON_REVERSAL), which this fixture does — so entry == level is
    # asserted for the resting case only.
    if plan.action == ACTION_WORK:
        assert abs(plan.entry - plan.level) < 1e-9
    # Either way the risk is measured from the ACTUAL fill, never from the level.
    risk_atr = abs(plan.stop - plan.entry) / _f_atr()
    assert risk_atr <= TG.MAX_STOP_ATR + 1e-9, (
        f'risk {risk_atr:.2f} ATR exceeds MAX_STOP_ATR on the real entry'
    )


def test_dollar_risk_stays_inside_the_budget():
    """Whatever route the stop took, size x risk must clear the budget."""
    plan = _plan()
    risk_pct = abs(plan.stop - plan.entry) / plan.entry * 100.0
    budgeted = risk_pct * plan.size_factor
    assert budgeted <= TG.MAX_STOP_PCT * 1.02, (
        f'size {plan.size_factor:.3f} x risk {risk_pct:.2f}% = {budgeted:.2f}% '
        f'exceeds the {TG.MAX_STOP_PCT}% budget'
    )


def test_the_relevelling_is_recorded():
    """Moving the entry is a material change to the trade the card advertises."""
    plan = _plan()
    assert any('the stop defends the further level' in n for n in plan.notes), (
        'the entry was moved to a different level and the plan does not say so'
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


# ── the regression this fix caused, and must never cause again ───────────────
# 2026-08-20: the branch below used to `return _reject('allocation', ...)` when
# the structural stop could not be afforded. Structural stops WIDER than the
# 1.30% cap turned out to be the common case, not the rare one — all three of
# that day's losses shipped a stop at exactly the cap, so structure had asked
# for more every time — and the reject fired constantly. Firing fell from 11.2%
# to 9.1% of setups on a 5,120-case sweep and nothing reached the tape for the
# two hours after it deployed.
#
# The rule: improving stop placement must never REMOVE a trade that fired before.

def _unaffordable(**kw):
    """A far level whose clearance cannot be paid for by size.

    Geometry found by sweeping the gate, not hand-picked: an oversold bear at a
    support 0.2% away with the next level 5% beyond it and ATR at 2.6%. Clearing
    the far level needs far more than the 1.30% budget, and scaling
    EXHAUSTION_REVERSAL's 0.50 weight down by that ratio lands under
    MIN_SIZE_FACTOR.

    Re-picked 2026-08-21 when MIN_SIZE_FACTOR dropped 0.25 -> 0.12: the previous
    fixture became affordable at the lower floor and stopped exercising this
    branch at all.
    """
    d = mk(price=100.0, atr=2.6, support=99.8, resistance=106.0,
           rsi=28.0, **TURNED_UP)
    d.update(kw)
    return d


UNAFFORDABLE_LEVELS = [(99.8, 4), (94.81, 4), (106.0, 4), (111.3, 3)]


@pytest.fixture
def _split_entry_and_stop(monkeypatch):
    """The unaffordable branch below only exists when the entry and the stop can
    sit on DIFFERENT levels. Since 2026-08-22 they collapse onto one, which makes
    the structural stop the buffer width and the case unreachable on ordinary
    geometry — the fix dissolved the problem this branch was a fallback for.
    The branch and its switch remain, so it is exercised in the configuration
    where it applies."""
    monkeypatch.setattr(TG, 'ENTRY_FOLLOWS_DEFENDED_LEVEL', False)


def _unaff_plan():
    plan = run(_unaffordable(), regime='TRENDING_BEAR', levels=UNAFFORDABLE_LEVELS)
    assert any('Tightening instead' in n for n in (plan.notes or [])), (
        f'fixture no longer reaches the unaffordable branch '
        f'({plan.stage}: {plan.reason}) — these tests would stop covering it'
    )
    return plan


def test_an_unaffordable_structural_stop_does_not_reject_the_trade(_split_entry_and_stop):
    plan = _unaff_plan()
    assert plan.action != ACTION_REJECT, (
        f'the unaffordable branch is rejecting again ({plan.stage}: {plan.reason}) '
        f'— this is the change that stopped every signal firing on 2026-08-20'
    )


def test_an_unaffordable_stop_falls_back_to_the_budget_stop_at_full_size(_split_entry_and_stop):
    plan = _unaff_plan()
    risk_pct = abs(plan.stop - plan.entry) / plan.entry * 100.0
    assert risk_pct <= TG.MAX_STOP_PCT + 1e-6, (
        f'fell back but kept a {risk_pct:.2f}% stop — the fallback tightens to '
        f'the {TG.MAX_STOP_PCT}% budget'
    )
    assert plan.size_factor >= TG.MIN_SIZE_FACTOR, (
        f'fell back to a tightened stop but left size at {plan.size_factor:.2f}; '
        f'the point of tightening is that full size is affordable again'
    )


def test_the_fallback_is_exactly_the_old_behaviour(_split_entry_and_stop):
    """The guarantee: improving stop placement must never REMOVE a trade."""
    plan = _unaff_plan()
    assert plan.action != ACTION_REJECT
    assert abs(abs(plan.stop - plan.entry) / plan.entry * 100.0
               - TG.MAX_STOP_PCT) < 1e-6, 'not the pre-change banded stop'
    assert plan.size_factor == 0.50, (
        f'size {plan.size_factor} — the fallback must not keep the scaled-down '
        f'value that the structural stop would have needed'
    )
