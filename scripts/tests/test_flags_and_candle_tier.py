"""Two more things get a vote on the risk tier.

Both were already being computed every scan and thrown away.

  1. Candle confirmation. The 150k-bar study that measured the fade edge found
     candle confirmation is one of the few things that reliably helps. The
     scores were computed, published to the chart, and read by the reversal
     arbiter — but never allowed to speak about the tier, so a fire with no
     confirming candle went out looking identical to one with three.

  2. Flags and pennants. New geometry, added to feature_engine so training and
     serving both see it. A bull flag resolves upward more often than not, so
     selling into one is taking the other side of the pattern.

Neither blocks a trade. The plan still decides whether to trade; these decide
how loudly it is announced — the same contract the meter and the S/R flags got.
"""
import inspect

import numpy as np
import pandas as pd
import pytest

from src.ml.feature_engine import (FLAG_COLS, FLAG_MAX_RETRACE,
                                   FLAG_POLE_MIN_ATR, compute_flag_patterns)


# ── geometry ─────────────────────────────────────────────────────────────────

def _frame(closes, wick=0.15):
    c = np.asarray(closes, dtype=float)
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({
        'open': o, 'high': np.maximum(o, c) + wick,
        'low': np.minimum(o, c) - wick, 'close': c,
        'volume': np.full(len(c), 1000.0),
    })


FLAT = list(np.full(20, 100.0))
POLE_UP   = list(np.linspace(101, 112, 8))
POLE_DOWN = list(np.linspace(99, 88, 8))

CASES = {
    # a pole, then a shallow channel drifting AGAINST it
    'bull_flag':    POLE_UP   + [111.4, 111.0, 110.6, 110.2, 109.8, 109.4],
    'bear_flag':    POLE_DOWN + [88.6, 89.0, 89.4, 89.8, 90.2, 90.6],
    # a pole, then converging boundaries — highs falling into rising lows
    'bull_pennant': POLE_UP   + [109.0, 111.5, 109.6, 111.0, 110.1, 110.6],
    'bear_pennant': POLE_DOWN + [91.0, 88.5, 90.4, 89.0, 89.9, 89.4],
}


def _last(tail):
    return compute_flag_patterns(_frame(FLAT + tail)).iloc[-1]


@pytest.mark.parametrize('name,tail', sorted(CASES.items()))
def test_each_pattern_is_detected(name, tail):
    row = _last(tail)
    assert row[name] == 1, f'{name} not detected'


@pytest.mark.parametrize('name,tail', sorted(CASES.items()))
def test_only_one_pattern_fires_at_a_time(name, tail):
    """A flag and a pennant are told apart by the sign of the lower boundary's
    slope, so they must never both be true — that would double-count."""
    row = _last(tail)
    lit = [c for c in ('bull_flag', 'bear_flag', 'bull_pennant', 'bear_pennant')
           if row[c]]
    assert lit == [name], lit


@pytest.mark.parametrize('name,tail', sorted(CASES.items()))
def test_bias_carries_the_direction(name, tail):
    assert _last(tail)['flag_bias'] == (1.0 if name.startswith('bull') else -1.0)


def test_the_trigger_is_the_boundary_not_the_close():
    """A flag is traded on the break of its boundary, never inside it."""
    for tail in CASES.values():
        assert _last(tail)['flag_breakout_dist_atr'] > 0


# ── what must NOT be a flag ──────────────────────────────────────────────────

def test_chop_is_not_a_flag():
    """Loose detection fires on any pause after any move, which is most bars."""
    row = _last(list(100 + np.sin(np.arange(14)) * 0.4))
    assert row['flag_bias'] == 0


def test_a_full_retrace_is_a_reversal_not_a_flag():
    """Giving back the whole pole is the opposite of a continuation."""
    row = _last(POLE_UP + [110, 108, 106, 104, 102, 100])
    assert row['flag_bias'] == 0


def test_a_drift_is_not_a_pole():
    """Without FLAG_POLE_MIN_ATR any slow grind would qualify."""
    row = _last(list(np.linspace(100, 100.4, 8)) + [100.38, 100.36, 100.34,
                                                    100.32, 100.30, 100.28])
    assert row['flag_bias'] == 0


def test_a_sprawling_consolidation_is_a_range_not_a_flag():
    row = _last(POLE_UP + [112, 105, 113, 104, 112, 106])
    assert row['flag_bias'] == 0


def test_a_short_frame_returns_neutral_rather_than_raising():
    out = compute_flag_patterns(_frame([100, 101, 102]))
    assert list(out.columns) == FLAG_COLS
    assert (out['flag_bias'] == 0).all()


def test_no_lookahead():
    """A bar's value may not change when later bars arrive."""
    full = _frame(FLAT + CASES['bull_flag'])
    cut = full.iloc[:-3]
    assert (compute_flag_patterns(full).iloc[len(cut) - 1]['flag_bias']
            == compute_flag_patterns(cut).iloc[-1]['flag_bias'])


def test_the_detector_is_vectorised():
    """rolling().apply() with a polyfit would be correct and unusably slow over
    a 150k-bar training frame."""
    src = inspect.getsource(compute_flag_patterns)
    assert '.apply(' not in src, 'a per-row apply crept into the hot path'


# ── the feature contract ─────────────────────────────────────────────────────

def test_flags_reach_the_model_frame():
    from src.ml import feature_engine as FE
    src = inspect.getsource(FE.prepare_features)
    assert 'compute_flag_patterns' in src, (
        'the flag columns never reach prepare_features, so the next retrain '
        'will not see them')


def test_new_columns_cannot_break_train_serve_parity():
    """Parity flags what a model EXPECTS and the frame LACKS. Adding columns is
    safe — models trained before these exist simply reindex them away."""
    from src.ml.feature_engine import check_feature_parity
    fatal, benign = check_feature_parity(
        available=['rsi', 'atr'] + FLAG_COLS, expected=['rsi', 'atr'])
    assert fatal == [] and benign == []


# ── the tier rules ───────────────────────────────────────────────────────────

def _tier_block() -> str:
    from scripts.engine.engine import LiveEngine
    src = inspect.getsource(LiveEngine._run_trader_gate)
    return src[src.index('# ── ENTER'):]


def test_candle_confirmation_votes_on_the_tier():
    block = _tier_block()
    assert 'cdl_bull_reversal' in block and 'cdl_bear_reversal' in block, (
        'the candle scores are computed and published but still have no say in '
        'the tier')
    assert 'plan.side' in block


def test_the_unavailable_sentinel_is_not_read_as_a_score():
    """cdl_* is -1.0 when the feature build predates the pattern library, and
    bool(-1.0) is True. Tagging every signal on a stale frame would be worse
    than not tagging at all."""
    block = _tier_block()
    assert '>= 0.0' in block, (
        'the -1.0 unavailable sentinel is being treated as a real score')


def test_an_opposing_flag_votes_on_the_tier():
    block = _tier_block()
    assert 'flag_bias' in block and 'flag_available' in block


def test_an_agreeing_flag_is_not_a_promotion():
    """Flags fire with the trend, and trend-following fires are already tagged.
    Letting a flag upgrade one would undo that in a second place."""
    block = _tier_block()
    flag_part = block[block.index('flag_available'):]
    for upgrade in ("tier = 'STRONG'", "'NORMAL': 'STRONG'", "'RISKY': 'NORMAL'"):
        assert upgrade not in flag_part, f'an agreeing flag promotes via {upgrade}'


def test_the_measurement_that_justifies_the_flag_rule():
    """Documents the study, so the rule is not tuned away without the numbers.

    60 tokens, 58,646 1h bars, 1,031 flag/pennant formations, forward 6h,
    signed in the pattern's own direction against a +0.0091% baseline drift:

        AT FORMATION          n     mean      win    vs baseline
          bullish           499   +0.0887%   51.7%    +0.0796%
          bearish           532   +0.1144%   50.4%    +0.1235%

        ON THE BREAKOUT       n     mean      win
          bullish           189   +0.0196%   42.3%
          bearish           282   -0.2406%   39.4%
          both              471   -0.1362%   40.6%

    Two things follow, and they point opposite ways.

    The formation direction carries a small but consistent edge on both sides,
    which is what the tier rule leans on: a fire against a flag is a fire
    against a measurable drift. It is small, which is why it is a tier vote and
    not a veto.

    Trading the BREAK loses — 40.6% win rate, negative on both sides. By the
    time the boundary goes the move is spent. So flags must NOT become an entry
    trigger here, however standard that is in the textbooks. A 14-symbol sample
    said the formation edge was negative too; the full fleet reversed it, so
    the small-sample version should not be reintroduced from memory.
    """
    from scripts.engine.engine import LiveEngine
    src = inspect.getsource(LiveEngine._run_trader_gate)
    assert 'flag_breakout' not in src, (
        'the flag BREAKOUT is being traded — measured at 40.6% win rate and '
        'negative expectancy across 471 breaks')


def test_neither_criterion_blocks_a_trade():
    """Both are tier votes. The plan decides whether to trade."""
    block = _tier_block()
    tail = block[block.index('# ── criterion 1'):]
    for blocker in ('return None', "result['fire'] = False", 'continue'):
        assert blocker not in tail, (
            f'a tier criterion is vetoing the trade via {blocker!r} — that is '
            f'the gate\'s job, not the announcer\'s')
