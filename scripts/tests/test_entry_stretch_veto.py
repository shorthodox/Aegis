"""A trade may not fire into strength.

The ask was for a trend veto — one more hard requirement, using "a genuine sign
of the trend". Measured over 10,980 simulated signals on 30 tokens against the
v86 ladder and a 1.3% stop, trend agreement is the wrong instrument. Against a
56.6% unfiltered baseline, where breakeven needs 60.9%:

    EMA50 slope agrees              55.6%      full EMA stack        55.8%
    right side of the EMA200        54.9%      ADX > 25              56.6%
    ADX > 20 AND slope agrees       55.6%      efficiency + slope    54.9%

Every one of them is at or below the baseline. Agreeing with the trend means
entering after the move.

The inverse separates, and monotonically, which is why it reads as signal:

    z < -0.1  keeps 48%  56.2%       z < -0.9  keeps 28%  57.9%
    z < -0.3  keeps 43%  56.0%       z < -1.5  keeps 12%  58.2%
    z < -0.5  keeps 39%  56.3%       z < -2.0  keeps  5%  59.9%
    z < -0.6  keeps 36%  56.8%

Note the shape. Between roughly 39% and 48% retention the filter is WORSE than
no filter — the edge lives in the tail, so a threshold picked to keep half the
book costs half the signals AND lowers the hit rate. -0.9 is the loosest
setting that buys anything real.

It does not reach 60.9% alone, and this file does not pretend otherwise.
"""
import inspect

import pytest

from scripts.engine.engine import LiveEngine


Z = LiveEngine.ENTRY_STRETCH_Z_MAX


def _stretch(bb_pct_b: float, side: str) -> float:
    """Mirror of the veto's maths: signed so negative is 'against the trade'."""
    z = (bb_pct_b - 0.5) * 4.0
    return z if side == 'BUY' else -z


def _passes(bb_pct_b: float, side: str) -> bool:
    return _stretch(bb_pct_b, side) <= Z


# ── the rule ─────────────────────────────────────────────────────────────────

def test_the_threshold_is_negative():
    """A positive threshold would demand the opposite trade — buying strength,
    which is the half of the book that measured worst."""
    assert Z is None or Z < 0


def test_a_long_may_only_fire_below_its_own_mean():
    assert _passes(0.10, 'BUY'), 'a long at the lower band was refused'
    assert _passes(0.20, 'BUY')
    assert not _passes(0.50, 'BUY'), 'a long fired at the mean'
    assert not _passes(0.90, 'BUY'), 'a long fired into the upper band'


def test_a_short_may_only_fire_above_its_own_mean():
    assert _passes(0.90, 'SELL'), 'a short at the upper band was refused'
    assert _passes(0.80, 'SELL')
    assert not _passes(0.50, 'SELL'), 'a short fired at the mean'
    assert not _passes(0.10, 'SELL'), 'a short fired into the lower band'


def test_the_two_sides_are_mirror_images():
    """An asymmetry here would be a sign bug, and it would silently bias the
    book toward one direction."""
    for pb in (0.05, 0.15, 0.3, 0.45, 0.55, 0.7, 0.85, 0.95):
        assert _stretch(pb, 'BUY') == pytest.approx(-_stretch(1.0 - pb, 'BUY'))
        assert _passes(pb, 'BUY') == _passes(1.0 - pb, 'SELL')


# ── it is a hard veto, not a tier vote ───────────────────────────────────────

def _gate_src() -> str:
    return inspect.getsource(LiveEngine._run_trader_gate)


def test_it_refuses_the_trade_rather_than_tagging_it():
    src = _gate_src()
    block = src[src.index('Entry-stretch veto'):src.index('# ── ENTER')]
    assert 'return False' in block, (
        'the stretch check tags or logs but does not refuse — it was asked for '
        'as a hard requirement')
    assert "tier = 'RISKY'" not in block


def test_it_runs_before_the_position_is_opened():
    src = _gate_src()
    assert src.index('Entry-stretch veto') < src.index('# ── ENTER'), (
        'the veto sits after the ENTER branch, so it can never stop anything')


def test_a_missing_band_reading_fails_open():
    """A token whose feature build lacks %B must not be silently unable to
    trade — that is a data outage presenting as a strategy."""
    src = _gate_src()
    block = src[src.index('Entry-stretch veto'):src.index('# ── ENTER')]
    assert '_bpb is not None' in block


def test_the_veto_can_be_switched_off():
    """It is one measured filter worth ~1.3pp, not a law."""
    src = _gate_src()
    block = src[src.index('Entry-stretch veto'):src.index('# ── ENTER')]
    assert '_z_max is not None' in block, (
        'setting ENTRY_STRETCH_Z_MAX to None must disable the veto')


# ── the measurement must stay attached to the number ─────────────────────────

def test_the_threshold_sits_where_the_measurement_supports_it():
    """Anything looser than about -0.6 measured WORSE than no filter at all."""
    assert Z is None or Z <= -0.6, (
        f'ENTRY_STRETCH_Z_MAX is {Z} — in the -0.1 to -0.5 band the filter '
        f'discards signals and lowers the hit rate. Read the table in '
        f'engine.py before loosening it further')


def test_bb_pct_b_actually_reaches_the_gate():
    """The veto reads result['bb_pct_b']; it was a feature column only."""
    src = inspect.getsource(__import__('src.ml.predictor', fromlist=['Predictor']))
    assert '"bb_pct_b"' in src, (
        'bb_pct_b is no longer published, so the veto reads None on every '
        'signal and fails open forever')
