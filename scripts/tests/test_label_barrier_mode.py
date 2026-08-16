"""The model should be graded on the race it is paid for.

create_triple_barrier_labels placed its barriers at +/- k x ATR. The live
ladder puts its first rung at a fixed PERCENT of entry, so the two are only
the same distance by coincidence. Measured on real candles at k=1.5, the
training barrier sat at this multiple of the rung the trade needs:

    BTC   0.58x        ARKM  1.41x
    ETH   0.81x        JUP   1.33x
    DOGE  0.86x

A 2.4x spread across the fleet, so sixty models were graded on sixty different
races and none of them was the traded one. A model can be genuinely skilful at
its own barrier race and contribute almost nothing to this one.

'pct' mode grades it on the ladder's own first rung. Deliberately symmetric
even though the trade is not: asymmetric barriers would skew the label
distribution toward whichever side sits nearer and hand the model a directional
bias with no market in it. Direction is the prediction problem; the target/stop
ratio is a risk decision and belongs to the gate's payoff stage.

Switching modes changes what every model learns, so it applies on the next
retrain only and the sidecar records which mode produced each model.
"""
import numpy as np
import pandas as pd
import pytest

import scripts.retrain_model as R


def _frame(n=400, seed=3, drift=0.0, vol=0.004):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({
        'timestamp': pd.date_range('2026-01-01', periods=n, freq='h'),
        'open': o, 'high': np.maximum(o, c) * 1.002,
        'low': np.minimum(o, c) * 0.998, 'close': c,
        'volume': rng.uniform(500, 2000, n),
    })


def _labels(df, mode, **kw):
    return R.create_triple_barrier_labels(df, atr_multiplier=1.5,
                                          max_lookahead=48, barrier_mode=mode, **kw)


# ── the switch ───────────────────────────────────────────────────────────────

def test_atr_is_still_the_default():
    """Existing models were trained under it; changing the default silently
    would make every sidecar's precision figure incomparable."""
    assert R.LABEL_BARRIER_MODE == 'atr'


def test_the_pct_barrier_tracks_the_live_first_rung():
    """A constant copied by hand would drift the moment the ladder moved."""
    from scripts.engine.risk import DynamicRiskEngine
    assert R.LABEL_BARRIER_PCT == pytest.approx(DynamicRiskEngine.TP_LADDER_PCT[0])


def test_an_unknown_mode_is_refused_rather_than_silently_ignored():
    with pytest.raises(ValueError, match='barrier_mode'):
        _labels(_frame(), 'atrr')


def test_a_non_positive_pct_is_refused():
    with pytest.raises(ValueError, match='positive'):
        _labels(_frame(), 'pct', barrier_pct=0.0)


# ── the two modes really are different races ─────────────────────────────────

def test_the_modes_produce_different_labels():
    df = _frame()
    a, p = _labels(df, 'atr'), _labels(df, 'pct')
    assert not a.equals(p), (
        'atr and pct labelling agree exactly — the mode is not reaching the '
        'barrier computation')


def test_a_wider_pct_barrier_produces_more_holds():
    """The sanity check on the unit: a barrier further away is reached less
    often inside the lookahead, so more bars resolve to HOLD."""
    df = _frame()
    near = _labels(df, 'pct', barrier_pct=0.5)
    far = _labels(df, 'pct', barrier_pct=6.0)
    assert (far == 1).sum() > (near == 1).sum()


def test_the_pct_barrier_does_not_depend_on_volatility():
    """The whole point. Two tokens with different ATR must get the SAME
    barrier distance in pct mode, and different ones in atr mode."""
    quiet, wild = _frame(vol=0.002, seed=7), _frame(vol=0.02, seed=7)
    # in pct mode the barrier is a fixed fraction of entry, so the label
    # distribution shifts only because the PATH changed, not the target
    for mode in ('atr', 'pct'):
        lq, lw = _labels(quiet, mode), _labels(wild, mode)
        assert len(lq) == len(lw)
    # a direct read of the geometry: pct mode ignores atr_multiplier entirely
    df = _frame()
    assert _labels(df, 'pct').equals(
        R.create_triple_barrier_labels(df, atr_multiplier=99.0,
                                       max_lookahead=48, barrier_mode='pct')), (
        'pct mode is still consulting the ATR multiplier')
    assert not _labels(df, 'atr').equals(
        R.create_triple_barrier_labels(df, atr_multiplier=99.0,
                                       max_lookahead=48, barrier_mode='atr'))


# ── symmetry ─────────────────────────────────────────────────────────────────

def test_pct_barriers_are_symmetric_by_default():
    """Asymmetric barriers would skew the label distribution toward the nearer
    side and hand the model a directional bias with no market in it."""
    assert R.BARRIER_UP_SKEW == R.BARRIER_DOWN_SKEW


def test_the_skews_still_apply_in_pct_mode():
    """The unit changed; the machinery around it did not."""
    df = _frame()
    even = _labels(df, 'pct')
    skewed = R.create_triple_barrier_labels(
        df, atr_multiplier=1.5, max_lookahead=48, barrier_mode='pct',
        barrier_up_skew=0.4, barrier_down_skew=2.5)
    assert not even.equals(skewed)
    # a much nearer upper barrier must produce more BUY labels
    assert (skewed == 2).sum() > (even == 2).sum()


# ── provenance ───────────────────────────────────────────────────────────────

def test_the_sidecar_records_which_race_the_model_was_graded_on():
    """Precision from an atr model and a pct model are not comparable, and a
    mixed fleet would otherwise be indistinguishable."""
    import inspect
    src = inspect.getsource(R)
    i = src.index('_sidecar_payload = {')
    block = src[i:i + 1200]
    assert 'label_barrier_mode' in block
    assert 'label_barrier_pct' in block


def test_the_contract_does_not_reject_the_new_keys():
    """validate_for_training runs on the payload before it is written."""
    from scripts.engine.contract import REQUIRED_KEYS
    assert 'label_barrier_mode' not in REQUIRED_KEYS, (
        'making the new key required would refuse every existing sidecar')
