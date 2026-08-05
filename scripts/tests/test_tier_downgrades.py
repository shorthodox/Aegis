"""The conviction meter and the S/R flags must actually reach the risk tier.

Both downgrades were documented in three places and implemented in none:

    scripts/engine/config.py   UWGS runs "for the chart breakdown, the risk
                               tier, and the four genuinely protective HARD
                               vetoes"
    web/.../chart.html         "when this meter disagrees, the signal is tagged
                               RISKY"
    engine.py (_process_symbol) "location flags become a tier downgrade below
                               (not a block)"

v83 derives the tier from plan.r_net and plan.size_factor alone, so the meter
and the flags were computed, published to the chart, and then ignored.

Reference failure, ADA/USDT 2026-08-05: the meter read BUY 0.3, SELL 29.5,
HOLD 56.5 — hold decided, against a BUY — and the signal was published as
STRONG / LOW RISK.

The plan still decides WHETHER to trade. This only decides how loudly.
"""
import inspect

import pytest

from scripts.live_engine import LiveEngine


SRC = inspect.getsource(LiveEngine._run_trader_gate)
CODE = '\n'.join(l for l in SRC.splitlines() if not l.strip().startswith('#'))


def _tier(r_net, size_factor, scores=None, sr_loc_poor=False, side='BUY'):
    """Mirror of the engine's tier rule, so the policy itself is testable."""
    tier = ('STRONG' if r_net >= 2.5 and size_factor >= 0.85
            else 'NORMAL' if r_net >= 2.0 or size_factor >= 0.7
            else 'RISKY')
    if scores:
        top = max(scores, key=lambda k: float(scores.get(k) or 0.0))
        if top != side.lower():
            tier = 'RISKY'
    if sr_loc_poor and tier != 'RISKY':
        tier = {'STRONG': 'NORMAL', 'NORMAL': 'RISKY'}.get(tier, tier)
    return tier


# ── the policy ───────────────────────────────────────────────────────────────

def test_the_ada_signal_would_no_longer_publish_as_strong():
    """BUY 0.3 / SELL 29.5 / HOLD 56.5 against a BUY, on strong geometry."""
    ada = {'buy': 0.3, 'sell': 29.5, 'hold': 56.5}
    assert _tier(3.0, 0.9, scores=ada, side='BUY') == 'RISKY'
    # without the meter it would have gone out as STRONG
    assert _tier(3.0, 0.9) == 'STRONG'


@pytest.mark.parametrize('scores,side,expected', [
    ({'buy': 70.0, 'sell': 10.0, 'hold': 20.0}, 'BUY',  'STRONG'),  # meter agrees
    ({'buy': 10.0, 'sell': 70.0, 'hold': 20.0}, 'BUY',  'RISKY'),   # meter says sell
    ({'buy': 10.0, 'sell': 20.0, 'hold': 70.0}, 'BUY',  'RISKY'),   # meter says hold
    ({'buy': 10.0, 'sell': 70.0, 'hold': 20.0}, 'SELL', 'STRONG'),  # meter agrees
    ({'buy': 70.0, 'sell': 10.0, 'hold': 20.0}, 'SELL', 'RISKY'),
])
def test_meter_disagreement_forces_risky(scores, side, expected):
    assert _tier(3.0, 0.9, scores=scores, side=side) == expected


@pytest.mark.parametrize('r_net,sf,start,after', [
    (3.0, 0.9, 'STRONG', 'NORMAL'),
    (2.1, 0.5, 'NORMAL', 'RISKY'),
    (1.0, 0.3, 'RISKY',  'RISKY'),   # already at the bottom
])
def test_poor_sr_location_demotes_one_notch(r_net, sf, start, after):
    assert _tier(r_net, sf) == start
    assert _tier(r_net, sf, sr_loc_poor=True) == after


def test_a_missing_meter_changes_nothing():
    """No signal_scores (UWGS disabled) must not silently downgrade everything."""
    assert _tier(3.0, 0.9, scores=None) == 'STRONG'
    assert _tier(3.0, 0.9, scores={}) == 'STRONG'


def test_downgrade_never_promotes():
    """A downgrade may only lower the tier, never raise it."""
    order = {'RISKY': 0, 'NORMAL': 1, 'STRONG': 2}
    for r_net, sf in ((3.0, 0.9), (2.1, 0.5), (1.0, 0.2)):
        base = _tier(r_net, sf)
        for scores in ({'buy': 90.0, 'sell': 1.0, 'hold': 1.0},
                       {'buy': 1.0, 'sell': 90.0, 'hold': 1.0},
                       {'buy': 1.0, 'sell': 1.0, 'hold': 90.0}):
            for poor in (False, True):
                got = _tier(r_net, sf, scores=scores, sr_loc_poor=poor)
                assert order[got] <= order[base], (
                    f'tier was promoted from {base} to {got}'
                )


# ── the wiring ───────────────────────────────────────────────────────────────

def test_engine_reads_the_meter_when_setting_the_tier():
    assert "result.get('signal_scores')" in CODE, (
        'the tier no longer consults the conviction meter — chart.html still '
        'promises the subscriber that it does'
    )
    assert "tier = 'RISKY'" in CODE


def test_engine_reads_the_sr_location_flag():
    assert "result.get('sr_loc_poor')" in CODE, (
        'the S/R location flag is computed and published but no longer reaches '
        'the tier, contrary to the note in _process_symbol'
    )


def test_the_plan_still_decides_whether_to_trade():
    """The downgrade must not turn into a veto — it only relabels."""
    assert 'plan.r_net' in CODE and 'plan.size_factor' in CODE
    # no early return between the tier block and the position being opened
    tier_at = CODE.index('_tier_notes')
    open_at = CODE.index('self._open_position')
    assert tier_at < open_at
    between = CODE[tier_at:open_at]
    assert 'return True' not in between, (
        'the tier downgrade became a block — it is only meant to relabel'
    )
