"""A signal may not fire on quality the engine itself scores as poor.

Measured on the live fleet, 2026-08-16, across 44 scored symbols:

    book HELD by     COMP 0, DOGE 18, LINK 35, TAO 38.7, ATOM 56
    turned away      TRX 100, INJ 76, PENDLE 71, UNI 66, AAVE 61

Quality was not merely ignored, it was ANTI-correlated with firing: the five
worst signals on the board held every slot in a book of five while the five best
were refused at the allocation cap. A signal scoring 0/100 was an open position.

The cause is that nothing on the live entry path ever read the score. `fire` is
`edge_score >= thr` (src/ml/predictor.py:990), and engine.py's own comment says
edge_score is a PERCENTILE RANK against the bar's own lookback — in a window
where every bar is poor, the least-poor bar ranks 100. quality_score is the
absolute measure and was spent only on POSITION SIZE.

A floor did exist and was lost rather than retired: the pre-v83 guard chain ran
G3_MIN_QUALITY (scripts/backtest_forensic.py:378; the 2026-07-01 forensic audits
log "edge=62.5 < MIN_QUALITY_SCORE=70.0"). The TraderGate rewrite replaced Guards
A..T wholesale and no stage picked the floor back up. The constant survived —
SignalQualityFilter.MIN_QUALITY_SCORE = 60.0, "biggest WR lever" — with its only
live readers an exit check and a pre-gate position check.

These tests pin the floor's EXISTENCE and its attribution, not a win rate. n=0
trades have been taken under it; see docs/SYSTEMS_REVIEW.md §0 on what a small
sample can and cannot support.
"""
import pytest

from scripts.engine import config as _cfg


def test_the_floor_exists_and_is_a_real_bar():
    assert hasattr(_cfg, 'MIN_FIRE_QUALITY'), (
        'the entry quality floor is gone again — this is the third time a '
        'quality bar has been dropped by a rewrite rather than retired'
    )
    assert 0 < _cfg.MIN_FIRE_QUALITY <= 100


def test_the_floor_is_above_the_fleet_median():
    """A floor at or below the median is not a floor, it is decoration.

    The median is not a constant. Measured 2026-08-16 it was 30; after the
    reversal-penalty exemptions it moved to 40 (p75 50, max 68), which is why
    60 stopped being a bar and became a wall — it retained 3 of 44 symbols for a
    book of five. Re-measure before moving this again; do not assume the
    distribution the last calibration saw.
    """
    assert _cfg.MIN_FIRE_QUALITY > 40


def test_the_floor_does_not_starve_a_book_of_five():
    """The other direction. At 60 the floor retained ~11% of 44 scored symbols —
    about five at once, which is MAX_OPEN. Much higher and the book cannot fill.

    The repo has form: a 35,640-scenario sweep once found stacked floors
    rejecting 45% of everything at payoff, contributing to a 0% fire rate.
    """
    assert _cfg.MIN_FIRE_QUALITY <= 70


# ── the decision, reduced to what it actually is ─────────────────────────────

def _fires(quality, model_fire=True, side='BUY', hard=()):
    """Mirror of the commit point in engine._process_symbol."""
    floor = _cfg.MIN_FIRE_QUALITY
    if hard:
        return False
    if floor > 0 and model_fire and side in ('BUY', 'SELL') and quality < floor:
        return False
    return model_fire and side in ('BUY', 'SELL')


def test_the_worst_of_the_book_is_still_refused():
    """Four of the five that held the book on 2026-08-16 stay out.

    ATOM is deliberately NOT in this list any more. At quality 56 it cleared the
    floor when it dropped 60 -> 45 on 2026-08-17, and pretending otherwise would
    make this file describe a system that does not exist. ATOM is the price of
    that decision, named in config.py and asserted below so the trade-off stays
    visible instead of dissolving into a number.
    """
    for sym, q in (('COMP', 0.0), ('DOGE', 18.0), ('LINK', 35.0), ('TAO', 38.7)):
        assert not _fires(q), f'{sym} at quality {q} fired again'


def test_atom_at_56_is_knowingly_admitted_by_the_lower_floor():
    """The trade the 60 -> 45 drop buys. ATOM's entry geometry — 1.69 ATR from
    the level it leaned on — is what started this investigation, and at 45 its
    context score no longer stops it. The location veto is a separate control
    and is what should catch it; this asserts only that the QUALITY floor does
    not, so nobody later reads the drop as free."""
    assert _fires(56.0)
    assert _cfg.MIN_FIRE_QUALITY <= 56.0


def test_the_five_that_were_turned_away_all_clear_the_floor():
    for sym, q in (('TRX', 100.0), ('INJ', 76.0), ('PENDLE', 71.0),
                   ('UNI', 66.0), ('AAVE', 61.0)):
        assert _fires(q), f'{sym} at quality {q} was blocked by the floor'


def test_the_floor_cannot_manufacture_a_fire():
    """It may only SUPPRESS. High quality with no model fire is still no trade."""
    assert not _fires(95.0, model_fire=False)
    assert not _fires(95.0, side='FLAT')


def test_a_hard_veto_still_wins_over_high_quality():
    assert not _fires(95.0, hard=('FAR_FROM_SR',))


def test_the_refusal_is_counted():
    """A threshold on a live funnel that is not counted cannot be tuned, and the
    0%-fire-rate incident is what happens when one is not."""
    from scripts.engine.engine import LOW_QUALITY_REFUSED
    assert isinstance(LOW_QUALITY_REFUSED, dict)
    assert 'count' in LOW_QUALITY_REFUSED
