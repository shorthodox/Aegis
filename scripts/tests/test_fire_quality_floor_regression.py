"""The 45 floor was measured and it lost money. Pin the revert.

Timeline, from the live track record on 2026-08-19:

    before the drop   n=14   WR 78.6%   avg +0.901%   total +12.62%
    after the drop    n=26   WR 38.5%   avg -0.342%   total  -8.89%
    Fisher exact two-sided p = 0.0219

Not a bad run — a real regression, and it starts at the commit that lowered
MIN_FIRE_QUALITY from 60 to 45. The book also went 85% short (22 of 26) with most
losses closing near the 1.30% stop: a flood of marginal shorts getting stopped.

The justification for 45 does not survive re-measurement. It was "60 retains only
3 of 44 tokens, a book of five cannot fill". That reading was taken minutes after
the reversal-penalty exemptions merged, before the new scores had propagated. The
same floor measured on 2026-08-19 retains 10 of 44 (22.7%) against MAX_OPEN = 5.
The funnel was never starved; the measurement was premature.
"""
import pytest

from scripts.engine import config as _cfg


def test_the_floor_is_back_at_sixty():
    assert _cfg.MIN_FIRE_QUALITY == 60.0, (
        'the quality floor moved off 60 again — 45 was measured at 38.5% WR '
        'against 78.6% at 60, p=0.0219. Do not lower it without a fresh '
        'retention measurement AND an outcome sample.'
    )


def test_it_is_never_silently_dropped_below_the_measured_harm_point():
    """45 is the value that produced the losing sample. Nothing may sit at or
    under it without deliberate re-measurement."""
    assert _cfg.MIN_FIRE_QUALITY > 45.0


def test_the_scorer_exemptions_were_not_reverted_at_the_same_time():
    """Only ONE variable moved in the revert.

    The reversal-penalty exemptions (lstm_exhaustion skipped, macd_conflict
    halved) landed the same day as the 45 floor. Reverting both at once would
    make the next sample uninterpretable — if the win rate does not recover, the
    exemptions are the remaining suspect and must be testable on their own.
    """
    import inspect
    from scripts.engine.quality import SignalQualityFilter
    src = inspect.getsource(SignalQualityFilter.score_signal)
    assert 'is_reversal_strict or is_structural_reversal' in src
    assert '_pen = 6 if' in src, 'the halved macd_conflict penalty is gone'
