"""A banked rung must not be handed all the way back.

Break-even goes on at TP1 and the trailing stop only starts at TP2, so a
position that tagged TP1 and reversed gave the entire move back to entry with
nothing protecting it in between.

Observed on BCH/USDT 2026-08-06: short from 214.70, TP1 212.09 tagged and
banked, price back up to 213.40, stop still sitting at break-even 214.70. The
runner was on its way to returning the whole rung.

The give-back ratchet closes the remainder once price hands back
TP_GIVEBACK_PCT of the rung's own span — entry→TP1 for the first rung,
TP1→TP2 for the second, and so on — and only ever tightens.

This is not the deleted TP1_RECROSS. That had a ZERO-width buffer (any tick
back through a tagged TP) and ran when TP1 was 0.7R against a 1.0R stop, so it
capped every winner at +0.7R. TP1 is 1.0R now and the buffer is a parameter.
"""
import pytest

from scripts.engine.risk import DynamicRiskEngine


# the BCH signal, as published
ENTRY = 214.70
TP1   = 212.091143
TP2   = 209.482286
ATR   = ENTRY * 0.0064          # ATR% 0.64 from the dashboard


def _level(tp, prev, direction, pct=None, min_atr=None, atr=ATR):
    """Mirror of the ratchet's level maths, so the policy is testable alone."""
    pct = DynamicRiskEngine.TP_GIVEBACK_PCT if pct is None else pct
    min_atr = (DynamicRiskEngine.TP_GIVEBACK_MIN_ATR if min_atr is None else min_atr)
    span = abs(tp - prev)
    leash = max(span * pct, min_atr * atr)
    return tp - leash if direction == 'LONG' else tp + leash


# ── the reported case ────────────────────────────────────────────────────────

def test_bch_would_now_close_instead_of_returning_to_break_even():
    lvl = _level(TP1, ENTRY, 'SHORT')
    assert lvl > TP1, 'the protective level must sit behind the rung'
    assert lvl < ENTRY, 'and well in front of break-even'
    # price came back to 213.40 — that is past the give-back level
    assert 213.40 >= lvl, (
        f'BCH at 213.40 would still not close (level {lvl:.4f}) — the runner '
        f'keeps riding back toward entry'
    )


def test_the_gap_this_fills_is_real():
    """Between TP1 and TP2 nothing else protects the runner."""
    # break-even is at ENTRY; the give-back level is much closer to the rung
    lvl = _level(TP1, ENTRY, 'SHORT')
    assert abs(lvl - TP1) < abs(ENTRY - TP1) / 2


# ── the ratchet ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize('direction', ['LONG', 'SHORT'])
def test_level_sits_between_the_rung_and_the_previous_one(direction):
    entry, tp1 = (100.0, 105.0) if direction == 'LONG' else (100.0, 95.0)
    lvl = _level(tp1, entry, direction)
    lo, hi = sorted((entry, tp1))
    assert lo < lvl < hi


@pytest.mark.parametrize('direction', ['LONG', 'SHORT'])
def test_a_later_rung_tightens_the_leash(direction):
    if direction == 'LONG':
        entry, tp1, tp2 = 100.0, 105.0, 110.0
        first = _level(tp1, entry, direction)
        second = _level(tp2, tp1, direction)
        assert second > first, 'TP2 must ratchet the level upward for a LONG'
    else:
        entry, tp1, tp2 = 100.0, 95.0, 90.0
        first = _level(tp1, entry, direction)
        second = _level(tp2, tp1, direction)
        assert second < first, 'TP2 must ratchet the level downward for a SHORT'


def test_a_zero_width_span_is_ignored_not_divided_by():
    """A degenerate rung must not divide by zero, and must not be protected.

    _level() is the raw maths; _manage_exit additionally SKIPS any rung whose
    leash reaches its own span, which is what a zero span always does. See
    test_a_rung_narrower_than_the_leash_defers_to_break_even.
    """
    lvl = _level(105.0, 105.0, 'LONG', min_atr=0.0)
    assert lvl == pytest.approx(105.0)


def test_a_rung_narrower_than_the_leash_defers_to_break_even():
    """The regression that produced OP/USDT's early close.

    The leash is a fraction of the rung span, and moving the ladder to
    percentages shrank every span 2-3x — so the leash collapsed to 0.16 ATR, a
    sixth of a bar, and closed the runner immediately after TP1. The ATR floor
    fixes the width; this fixes what happens when the floor is WIDER than the
    rung, where a protective level would land past the entry and be worse than
    the break-even stop already there.
    """
    import inspect
    from scripts.live_engine import LiveEngine
    src = inspect.getsource(LiveEngine._manage_exit)
    assert '_leash >= _span' in src, (
        'a leash wider than its rung would place the give-back past the '
        'previous rung — for TP1 that is past the entry'
    )
    assert DynamicRiskEngine.TP_GIVEBACK_MIN_ATR > 0, (
        'the ATR floor is off again; the leash will collapse with the rung span'
    )


def test_the_first_rung_defers_but_a_wider_rung_still_protects():
    """The ratchet must still do its job somewhere, or it is dead weight."""
    atr = 0.0894 * 1.07 / 100
    pcts = DynamicRiskEngine.TP_LADDER_PCT
    entry = 0.0894
    lvls = [entry * (1 - p / 100) for p in pcts]
    prev = [entry] + lvls[:-1]
    applies = []
    for tp, pv in zip(lvls, prev):
        span = abs(tp - pv)
        leash = max(span * DynamicRiskEngine.TP_GIVEBACK_PCT,
                    DynamicRiskEngine.TP_GIVEBACK_MIN_ATR * atr)
        applies.append(leash < span)
    assert not applies[0], 'the TP1 rung should defer to break-even'
    assert any(applies), 'no rung protects at all — the ratchet is dead weight'


def test_min_atr_floor_can_widen_a_narrow_rung():
    """The dial that stops a tight rung producing a noise-width stop."""
    narrow = _level(100.5, 100.0, 'LONG', pct=0.05, min_atr=0.0, atr=1.0)
    floored = _level(100.5, 100.0, 'LONG', pct=0.05, min_atr=0.25, atr=1.0)
    assert abs(100.5 - floored) > abs(100.5 - narrow)


# ── the width is the whole question ──────────────────────────────────────────

def test_the_configured_leash_is_reported_in_atr():
    """Not an assertion about the right value — a guard that it stays visible.

    At 5% of an entry→TP1 span the leash is a fraction of one ATR, which is
    inside a single 1h bar's range. That is a deliberate, tunable choice; this
    test just fails loudly if someone sets a value that is effectively zero.
    """
    span = abs(TP1 - ENTRY)
    leash = max(span * DynamicRiskEngine.TP_GIVEBACK_PCT,
                DynamicRiskEngine.TP_GIVEBACK_MIN_ATR * ATR)
    assert leash > 0, 'a zero leash is the deleted TP1_RECROSS'
    assert leash < span, 'a leash wider than the rung protects nothing'


def test_giveback_state_is_cleared_when_a_position_closes():
    import inspect
    from scripts.live_engine import LiveEngine
    src = inspect.getsource(LiveEngine._manage_exit)
    close_block = src[src.index('def _close('):src.index('def _partial(')] \
        if 'def _partial(' in src else src
    assert '_giveback_stop' in close_block, (
        'the ratchet level survives a close — the next position on this symbol '
        'would inherit a stop from the previous trade'
    )


def test_ratchet_is_wired_into_the_exit_path():
    import inspect
    from scripts.live_engine import LiveEngine
    src = inspect.getsource(LiveEngine._manage_exit)
    assert 'TP_GIVEBACK' in src
    assert 'TP_GIVEBACK_PCT' in src
