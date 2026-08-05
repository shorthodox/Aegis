"""Two conviction numbers that were not measurements.

Both defects shared a shape: a number the engine spends as conviction was
produced by something that could not fail, and then flowed into
calculate_position_size().

  1. predict_signal's at-support / at-resistance override tested momentum with
     `macd_hist > -0.05`. macd_hist is `macd - macd_signal` on raw close, so it
     carries the asset's price units. On a sub-dollar token the histogram never
     approaches -0.05, so the test was a constant True and the override fired on
     structural location alone. It also floored edge_score at 65 -- inventing a
     percentile rank the model never produced.

  2. _process_symbol chose the scale of that number by inspecting its VALUE
     (`if _edge <= 1.0: _edge *= 100`). edge_score is a 0-100 percentile, so a
     real bottom-percentile reading of 0.8 was multiplied into 80.

The fleet is mostly sub-dollar tokens, which is precisely where (1) is a
constant.
"""
import numpy as np
import pandas as pd
import pytest

from src.ml.feature_engine import compute_macd


# ── 1. the momentum test must be able to fail at any price ───────────────────

PRICES = [
    ('STRK-like', 0.0259),
    ('SAND-like', 0.30),
    ('ATOM-like', 4.50),
    ('ETH-like',  3000.0),
    ('BTC-like',  60000.0),
]


def _series(p0: float, seed: int = 3, n: int = 600) -> pd.Series:
    rng = np.random.default_rng(seed)
    steps = rng.normal(-0.0004, 0.015, n)   # mild downtrend, 1.5% hourly vol
    return pd.Series(p0 * np.exp(np.cumsum(steps)))


@pytest.mark.parametrize('name,p0', PRICES)
def test_raw_macd_threshold_is_price_dependent(name, p0):
    """Documents the defect: the OLD test's outcome depends on the price tag."""
    close = _series(p0)
    _, _, hist = compute_macd(close)
    frac_true = float((hist.dropna() > -0.05).mean())
    if p0 < 1.0:
        assert frac_true == 1.0, (
            f'{name}: expected the raw threshold to be a constant on cheap '
            f'tokens; got {frac_true:.3f}'
        )
    elif p0 > 1000.0:
        assert frac_true < 0.75, (
            f'{name}: raw threshold should be discriminating on expensive '
            f'tokens; got {frac_true:.3f}'
        )


@pytest.mark.parametrize('name,p0', PRICES)
def test_atr_normalised_macd_threshold_is_scale_free(name, p0):
    """The fix: normalised by ATR, the same threshold means the same thing.

    Across four orders of magnitude in price the pass-rate must land in the same
    band, otherwise the token's price tag is still deciding.
    """
    close = _series(p0)
    _, _, hist = compute_macd(close)
    # ATR proxy consistent with atr_14's magnitude on this series
    atr = close.diff().abs().rolling(14).mean()
    norm = (hist / atr).replace([np.inf, -np.inf], np.nan).dropna()
    frac_true = float((norm > -0.05).mean())
    assert 0.20 < frac_true < 0.90, (
        f'{name}: normalised momentum test still price-dependent '
        f'(pass rate {frac_true:.3f})'
    )


def test_normalised_threshold_pass_rates_agree_across_price_tiers():
    """The real assertion: cheap and expensive tokens get the same treatment."""
    rates = []
    for _, p0 in PRICES:
        close = _series(p0)
        _, _, hist = compute_macd(close)
        atr = close.diff().abs().rolling(14).mean()
        norm = (hist / atr).replace([np.inf, -np.inf], np.nan).dropna()
        rates.append(float((norm > -0.05).mean()))
    spread = max(rates) - min(rates)
    assert spread < 0.15, (
        f'pass rate still varies by {spread:.3f} across price tiers: '
        f'{[round(r, 3) for r in rates]}'
    )


def test_predictor_no_longer_fabricates_an_edge_score():
    """edge_score must never be floored to a value the model did not produce."""
    import inspect

    from src.ml.predictor import Predictor
    src = _code_only(inspect.getsource(Predictor.predict_signal))
    assert 'max(edge_score, 65.0)' not in src, (
        'edge_score is being floored again — it is a percentile rank that the '
        'engine spends as conviction via quality_score -> position size'
    )
    assert 'macd_hist_atr' in src, (
        'the momentum confirmation is reading raw macd_hist again — on a '
        'sub-dollar token that test cannot fail'
    )


# ── 2. the 0-1 vs 0-100 scale guess ──────────────────────────────────────────

def _quality_from(result: dict) -> float:
    """Mirror of the engine's scale selection, so the rule is testable."""
    if result.get('edge_score') is not None:
        edge = float(result.get('edge_score') or 0.0)
    else:
        edge = float(result.get('meta_confidence') or 0.0) * 100.0
    return round(max(0.0, min(edge, 100.0)), 1)


@pytest.mark.parametrize('result,expected', [
    # edge_score is ALREADY 0-100 and must pass through untouched
    ({'edge_score': 0.0},   0.0),
    ({'edge_score': 0.8},   0.8),     # the case the old value-guess inflated to 80
    ({'edge_score': 1.0},   1.0),     # boundary the old rule also multiplied
    ({'edge_score': 65.0},  65.0),
    ({'edge_score': 100.0}, 100.0),
    # meta_confidence is 0-1 and is the only field that needs scaling
    ({'meta_confidence': 0.42}, 42.0),
    ({'meta_confidence': 1.0},  100.0),
    # clamped
    ({'edge_score': 140.0}, 100.0),
    ({'edge_score': -5.0},  0.0),
])
def test_conviction_scale_is_chosen_by_field_not_by_value(result, expected):
    assert _quality_from(result) == expected


def _code_only(src: str) -> str:
    """Strip comments and docstring-ish lines.

    Without this the assertion matches the comment that EXPLAINS the old rule,
    which is exactly the sort of test that passes for the wrong reason.
    """
    out = []
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        out.append(line.split('  #')[0])
    return '\n'.join(out)


def test_engine_uses_the_field_based_rule():
    import inspect

    from scripts.live_engine import LiveEngine
    src = _code_only(inspect.getsource(LiveEngine._process_symbol))
    assert 'if _edge <= 1.0' not in src, (
        'the engine is guessing the conviction scale from the value again; a '
        'bottom-percentile edge of 0.8 becomes 80 and gets sized accordingly'
    )
    assert "result.get('edge_score') is not None" in src


def test_zero_conviction_sizes_at_the_floor_not_at_random():
    """A 0-conviction signal must size small -- and that must be deliberate."""
    from scripts.live_engine import DynamicRiskEngine, RegimeState

    r = DynamicRiskEngine()
    regime = RegimeState(regime='RANGING', confidence=0.5, trade_allowed=True,
                         preferred_strategies=[], max_position_pct=0.10)
    floor = 10_000.0 * DynamicRiskEngine.MIN_POSITION_PCT
    assert r.calculate_position_size(10_000.0, 0.0, regime, 1.5) == pytest.approx(floor)
    # and a fabricated 65 would have sized 32x larger than the measured 0
    sized_65 = r.calculate_position_size(10_000.0, 65.0, regime, 1.5)
    assert sized_65 > floor * 2, 'the fabricated edge floor was not cosmetic'
