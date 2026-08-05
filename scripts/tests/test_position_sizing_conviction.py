"""Size must rise with conviction, and conviction must be a conviction measure.

Two defects met in the sizing path and cancelled each other's symptoms, which is
why neither showed up as an obvious number:

  1. _open_position guarded dynamic sizing with `quality_score > 0`. A zero
     score therefore did NOT take the floor — it fell through to
     wallet.position_size(), which is min(balance * 10%, max_position): the
     MAXIMUM default. The curve inverted at exactly the wrong end.

         quality  5 ->  200 USDT
         quality 45 ->  315 USDT
         quality 65 ->  455 USDT
         quality  0 -> 1000 USDT   <- largest position of the set

  2. _process_symbol overwrote quality_score with edge_score before it reached
     that code. edge_score is a PERCENTILE RANK of the bar against its own
     lookback, so it carries no absolute information: in a poor window the
     least-poor bar ranks 100 and takes full size. It is also already the FIRE
     decision, so spending it again on size double-counts one number.

While the at-support override floored edge_score at 65, defect 2 kept defect 1
hidden — scores were rarely 0. Removing that fabricated floor exposed it, and
zero-edge signals jumped from ~318 to ~700 USDT.
"""
import inspect

import pytest

from scripts.engine.models import RegimeState
from scripts.engine.risk import DynamicRiskEngine


BAL = 10_000.0


def _regime(max_pct: float = 0.10) -> RegimeState:
    return RegimeState(regime='RANGING', confidence=0.5, trade_allowed=True,
                       preferred_strategies=[], max_position_pct=max_pct)


# ── 1. the curve must not invert ─────────────────────────────────────────────

def test_size_is_monotonic_in_conviction():
    r = DynamicRiskEngine()
    reg = _regime()
    sizes = [r.calculate_position_size(BAL, q, reg, 1.5)
             for q in (0, 5, 20, 45, 65, 85, 100)]
    assert sizes == sorted(sizes), f'sizing is not monotonic in quality: {sizes}'


def test_zero_conviction_sizes_at_the_floor():
    r = DynamicRiskEngine()
    floor = BAL * DynamicRiskEngine.MIN_POSITION_PCT
    assert r.calculate_position_size(BAL, 0.0, _regime(), 1.5) == pytest.approx(floor)


def test_zero_conviction_is_never_the_largest_position():
    """The exact inversion the guard produced."""
    r = DynamicRiskEngine()
    reg = _regime()
    zero = r.calculate_position_size(BAL, 0.0, reg, 1.5)
    for q in (5, 20, 45, 65, 85, 100):
        assert zero <= r.calculate_position_size(BAL, q, reg, 1.5), (
            f'a 0-conviction signal is sized above a {q}-conviction one'
        )


def test_flat_fallback_is_larger_than_the_floor():
    """Documents WHY the old guard was dangerous, so it is not reintroduced.

    wallet.position_size() is the maximum default allocation, not a small safe
    one. Routing low-conviction signals to it is an escalation, not a fallback.
    """
    from scripts.engine.portfolio import VirtualWallet
    w = VirtualWallet(BAL, 1_000.0)
    floor = BAL * DynamicRiskEngine.MIN_POSITION_PCT
    assert w.position_size() > floor * 2, (
        'the flat fallback is no longer much larger than the floor; if that '
        'changed, the reasoning in _open_position should be revisited'
    )


def test_open_position_no_longer_gates_sizing_on_a_nonzero_score():
    from scripts.live_engine import LiveEngine
    src = inspect.getsource(LiveEngine._open_position)
    body = '\n'.join(l for l in src.splitlines() if not l.strip().startswith('#'))
    assert 'quality_score > 0' not in body, (
        'dynamic sizing is gated on a non-zero score again — a 0 will fall '
        'through to wallet.position_size(), the MAXIMUM default'
    )
    assert 'if regime is not None:' in body


# ── 2. the conviction number must be a conviction measure ────────────────────

def test_engine_sizes_from_the_context_score_not_the_edge_percentile():
    from scripts.live_engine import LiveEngine
    src = inspect.getsource(LiveEngine._process_symbol)
    body = '\n'.join(l for l in src.splitlines() if not l.strip().startswith('#'))
    assert "result['quality_score'] = round(quality_score, 1)" in body, (
        'quality_score is being overwritten again. edge_score is a percentile '
        'rank and is already the fire decision; sizing on it double-counts one '
        'number and makes size track the bar\'s neighbours, not the setup'
    )
    assert "result['quality_score'] = round(_edge, 1)" not in body


def test_the_context_score_is_what_reaches_the_gate():
    """ctx_quality must actually be the context score its name claims."""
    from scripts.live_engine import LiveEngine
    src = inspect.getsource(LiveEngine._process_symbol)
    body = '\n'.join(l for l in src.splitlines() if not l.strip().startswith('#'))
    # the value handed to _run_trader_gate is the same `quality_score` that
    # _resolve_market_context returned from score_signal()
    assert 'regime, quality_score)' in body
    assert 'quality_score, fake_breakout = _ctx' in body


def test_score_signal_actually_produces_a_spread():
    """A sizing input that is constant would be no better than the percentile."""
    from scripts.engine.quality import SignalQualityFilter
    qf = SignalQualityFilter()
    reg = _regime()
    strong = {
        'adx': 34.0, 'volume_zscore': 2.0, 'rsi': 48.0,
        'confluence': {'total': 7.5, 'momentum': 7.0},
        'funding_bias': 'SHORTS_PAYING', 'oi_trend': 'INCREASING',
        'market_bias': 'BULLISH', 'macd_signal': 'BULLISH',
        'macro_daily': 1.0, 'macro_weekly': 1.0, 'edge_score': 80.0,
    }
    weak = {
        'adx': 12.0, 'volume_zscore': -1.4, 'rsi': 72.0,
        'confluence': {'total': 4.9, 'momentum': 5.0},
        'funding_bias': 'LONGS_PAYING', 'oi_trend': 'DECREASING',
        'market_bias': 'BEARISH', 'macd_signal': 'BEARISH',
        'macro_daily': -1.0, 'macro_weekly': -1.0, 'edge_score': 10.0,
    }
    s_strong, _ = qf.score_signal(strong, reg, 'BUY')
    s_weak, _ = qf.score_signal(weak, reg, 'BUY')
    assert s_strong > s_weak + 20, (
        f'score_signal does not separate a strong setup from a weak one '
        f'({s_strong} vs {s_weak}) — it would be a poor sizing input'
    )
    assert 0.0 <= s_weak and s_strong <= 100.0


def test_setup_conviction_still_reaches_size_via_the_plan():
    """Sizing on context quality must not discard the measured per-setup edge."""
    from src.trading import trader_gate as TG
    assert TG.SETUP_RISK_WEIGHT[TG.SETUP_TREND_PULLBACK] > \
           TG.SETUP_RISK_WEIGHT[TG.SETUP_EXHAUSTION_REVERSAL]
    from scripts.live_engine import LiveEngine
    src = inspect.getsource(LiveEngine._open_position)
    assert 'size_factor' in src, (
        'plan.size_factor no longer reaches sizing — the measured per-setup '
        'edge (TREND_PULLBACK +0.069R vs EXHAUSTION_REVERSAL -0.064R) is lost'
    )
