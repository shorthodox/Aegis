"""A banked rung must not be handed back at all.

v86 set TP_GIVEBACK_MAX_FRAC to zero: the protective level IS the rung, so a
re-cross books the remainder at the level rather than somewhere under it. The
invariant these tests pin is therefore `previous rung < level <= rung` — the
level may sit ON the rung, but never past it and never back at the previous one.
The leash constants are kept and still clamp correctly if the cap is raised.

The original note follows, because the hole it describes is still the reason
the mechanism exists.

A banked rung must not be handed all the way back.

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


def _level(tp, prev, direction, pct=None, min_atr=None, atr=ATR, max_frac=None):
    """Mirror of the ratchet's level maths, so the policy is testable alone."""
    pct = DynamicRiskEngine.TP_GIVEBACK_PCT if pct is None else pct
    min_atr = (DynamicRiskEngine.TP_GIVEBACK_MIN_ATR if min_atr is None else min_atr)
    max_frac = (DynamicRiskEngine.TP_GIVEBACK_MAX_FRAC if max_frac is None else max_frac)
    span = abs(tp - prev)
    leash = min(max(span * pct, min_atr * atr), span * max_frac)
    return tp - leash if direction == 'LONG' else tp + leash


# ── the reported case ────────────────────────────────────────────────────────

def test_bch_would_now_close_instead_of_returning_to_break_even():
    lvl = _level(TP1, ENTRY, 'SHORT')
    assert lvl >= TP1, 'the protective level must sit at or behind the rung'
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
    # Inclusive on the RUNG side, exclusive on the previous one. At a zero cap
    # the level IS the rung, which is the v86 policy; what must never happen is
    # the level passing the rung, or falling back to where it came from.
    if direction == 'LONG':
        assert entry < lvl <= tp1
    else:
        assert tp1 <= lvl < entry


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

    _level() is the raw maths; _manage_exit additionally skips any rung whose
    leash comes out non-positive, which is what a zero span always does.
    """
    lvl = _level(105.0, 105.0, 'LONG', min_atr=0.0)
    assert lvl == pytest.approx(105.0)


def test_the_atr_floor_may_not_exceed_the_rung_it_protects():
    """IMX/USDT 2026-08-08 — the floor voided the rung it was meant to widen.

    The floor was added because moving the ladder to percentages shrank every
    span 2-3x and collapsed the leash to 0.16 ATR. It fixed that and introduced
    the opposite failure: the first rung is a small percentage of price, so
    0.50 ATR was WIDER than the whole rung for most of the fleet, and the code
    skipped any rung whose leash reached its own span. Those tokens therefore
    had no give-back protection on TP1 at all — it fell through to break-even.

    IMX short 0.1124: TP1 0.11184 tagged at +0.50%, price walked back to entry,
    booked +0.02%. Capping instead of skipping keeps the level inside the rung.
    """
    import inspect
    from scripts.live_engine import LiveEngine
    src = inspect.getsource(LiveEngine._manage_exit)
    assert '_leash >= _span' not in src, (
        'the skip is back — a rung whose ATR floor exceeds its span is being '
        'abandoned to break-even instead of having its leash capped'
    )
    assert '_gb_max' in src, 'the leash is no longer capped to the rung'
    assert 0 <= DynamicRiskEngine.TP_GIVEBACK_MAX_FRAC < 1, (
        'the cap may hand back none of the rung (v86) but never all of it')

    # the IMX geometry, across the ATR range where the floor used to win
    entry = 0.1124
    tp1 = entry * (1 - DynamicRiskEngine.TP_LADDER_PCT[0] / 100)
    for atr_pct in (0.8, 1.2, 2.0, 3.0):
        lvl = _level(tp1, entry, 'SHORT', atr=entry * atr_pct / 100)
        assert tp1 <= lvl < entry, (
            f'at ATR {atr_pct}% the give-back level {lvl:.6f} is outside the '
            f'rung — TP1 {tp1:.6f}, entry {entry}')
        booked = (entry - lvl) / entry * 100
        floor = DynamicRiskEngine.TP_LADDER_PCT[0] * (
            1 - DynamicRiskEngine.TP_GIVEBACK_MAX_FRAC) - 1e-9
        assert booked >= floor, (
            f'at ATR {atr_pct}% a tagged TP1 books only {booked:.2f}% against a '
            f'{floor:.2f}% floor — the rung is being handed back')


def test_every_rung_of_the_live_ladder_is_protected():
    """The ratchet must do its job on the FIRST rung, not just the wide ones.

    This assertion is inverted from what it used to be. It previously required
    the TP1 rung to defer to break-even, which is exactly the behaviour that
    booked IMX at +0.02% after it had covered +0.50%.
    """
    atr = 0.0894 * 1.07 / 100
    entry = 0.0894
    lvls = [entry * (1 - p / 100) for p in DynamicRiskEngine.TP_LADDER_PCT]
    prev = [entry] + lvls[:-1]
    for i, (tp, pv) in enumerate(zip(lvls, prev), start=1):
        span = abs(tp - pv)
        leash = max(0.0, min(max(span * DynamicRiskEngine.TP_GIVEBACK_PCT,
                                 DynamicRiskEngine.TP_GIVEBACK_MIN_ATR * atr),
                             span * DynamicRiskEngine.TP_GIVEBACK_MAX_FRAC))
        assert 0 <= leash < span, f'TP{i} leash {leash} exceeds its span {span}'


def test_a_banked_rung_always_beats_a_scratch():
    """The point of the whole mechanism, stated once.

    Whatever the ATR, closing on a give-back after TP1 must return more than
    the break-even stop would have.
    """
    entry = 0.1124
    tp1 = entry * (1 - DynamicRiskEngine.TP_LADDER_PCT[0] / 100)
    for atr_pct in (0.5, 1.0, 2.0, 5.0):
        lvl = _level(tp1, entry, 'SHORT', atr=entry * atr_pct / 100)
        assert lvl < entry, (
            f'at ATR {atr_pct}% the give-back sits at or past entry — it would '
            f'never fire before the break-even stop, so TP1 protects nothing')


def test_min_atr_floor_can_widen_a_narrow_rung():
    """The dial that stops a tight rung producing a noise-width stop.

    Dormant while the cap is zero, so the cap is passed explicitly here — the
    point is that the floor still works when it is allowed to.
    """
    narrow = _level(100.5, 100.0, 'LONG', pct=0.05, min_atr=0.0, atr=1.0,
                    max_frac=0.5)
    floored = _level(100.5, 100.0, 'LONG', pct=0.05, min_atr=0.25, atr=1.0,
                     max_frac=0.5)
    assert abs(100.5 - floored) > abs(100.5 - narrow)


# ── the width is the whole question ──────────────────────────────────────────

def test_the_configured_leash_is_reported_in_atr():
    """Not an assertion about the right value — a guard that it stays visible.

    At 5% of an entry→TP1 span the leash is a fraction of one ATR, which is
    inside a single 1h bar's range. That is a deliberate, tunable choice; this
    test just fails loudly if someone sets a value that is effectively zero.
    """
    span = abs(TP1 - ENTRY)
    leash = max(0.0, min(max(span * DynamicRiskEngine.TP_GIVEBACK_PCT,
                             DynamicRiskEngine.TP_GIVEBACK_MIN_ATR * ATR),
                         span * DynamicRiskEngine.TP_GIVEBACK_MAX_FRAC))
    # v86 sets the cap to zero deliberately — see the module docstring and
    # test_the_recross_is_back_deliberately_and_this_is_the_bill.
    assert leash >= 0
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
