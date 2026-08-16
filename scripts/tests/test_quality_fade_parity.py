"""A mean-reversion setup must be able to reach the fire floor.

The user's strategy is mean-reversion, but the scorer is built around trend
agreement, so a fade could not reach MIN_FIRE_QUALITY on merit. Points a
with-trend signal can earn that a fade structurally cannot:

    strong_bull/bear_confluence  +20      macd_aligned          +8
    htf_both_aligned             +15      bias_aligned          +5
    lstm_continuation            +10      ------------------------
                                          headroom a fade never sees  +58

Against +28 of compensation at best (reversal_setup +18, strong candle +10).

Three of the penalties were already exempted for reversals — ranging (-15) and
the two HTF-opposing tiers (-20 / -10) — with a comment saying that penalising a
reversal for being counter-trend is double-counting. Two never got that
treatment, and MIN_FIRE_QUALITY = 60 turned the omission into a filter that
silently retires the strategy:

  * lstm_exhaustion (-8) is INVERTED, not merely double-counted. "Momentum is
    decaying" is a defect in a continuation trade and the whole thesis of a
    mean-reversion one. Now skipped for reversals.

  * macd_conflict (-12) is genuinely mixed. An opposing MACD is expected when
    fading, but it also carries real information about being EARLY, which a fade
    cannot dismiss. Halved to -6 rather than skipped.

These tests pin the exemptions and the boundary. They do NOT claim fading is
profitable — the 150k-bar measurement says fleet-wide it is -0.064R against
+0.069R for trend-following, and fades work per-token. The claim here is only
that a fade should be SCORED on its merits rather than disqualified by
arithmetic before the floor ever sees it.
"""
import pytest

from scripts.engine.config import MIN_FIRE_QUALITY
from scripts.engine.quality import SignalQualityFilter


class _Regime:
    def __init__(self, regime='RANGING', confidence=0.8):
        self.regime = regime
        self.confidence = confidence


def _fade(**over):
    """A textbook mean-reversion SELL: at range resistance, RSI exhausted,
    momentum rolling over, higher timeframes still bullish (as they always are
    when you fade a rally)."""
    r = {
        'price': 110.0, 'support': 100.0, 'resistance': 111.0,
        # ADX 28: if you are fading a trend, there is a trend there to measure.
        # This is the single line that decides the outcome — see
        # test_the_remaining_gap_is_evidence_not_direction below.
        'rsi': 74.0, 'adx': 28.0, 'volume_zscore': 0.9,
        'edge_score': 62.0, 'confluence': {'total': 5.0},
        'macd_signal': 'BULLISH',            # opposing — you are fading it
        'market_bias': 'BULLISH',
        'macro_weekly': 1.0, 'macro_daily': 1.0,   # HTF against the fade
        'lstm_available': True,
        'lstm_continuation_prob': 0.20,      # momentum decaying = the thesis
        'lstm_exhaustion_prob': 0.80,
        'lstm_vol_expansion_prob': 0.5,
        'cdl_bull_reversal': 0.0, 'cdl_bear_reversal': 1.8,
        'funding_bias': 'NEUTRAL', 'oi_trend': 'STABLE',
    }
    r.update(over)
    return r


def _score(result, side='SELL', regime='RANGING'):
    q, reasons = SignalQualityFilter().score_signal(result, _Regime(regime), side)
    return q, reasons


# ── the exemptions ───────────────────────────────────────────────────────────

def test_exhaustion_is_not_charged_against_a_reversal():
    _, reasons = _score(_fade())
    assert not any(r.startswith('lstm_exhaustion') for r in reasons), (
        'a fade was penalised for the decaying momentum it exists to trade'
    )


def test_exhaustion_is_still_charged_against_a_continuation_trade():
    """The exemption must be about the SETUP, not a blanket removal."""
    mid = _fade(price=105.0, rsi=52.0, cdl_bear_reversal=0.0)   # mid-range, no reversal
    _, reasons = _score(mid)
    assert any(r.startswith('lstm_exhaustion') for r in reasons), (
        'the penalty was removed outright instead of exempted for reversals'
    )


def test_macd_conflict_is_halved_for_a_reversal_not_removed():
    """Being early is real information a fade cannot dismiss."""
    _, reasons = _score(_fade())
    assert any(r.startswith('macd_conflict') for r in reasons), (
        'macd_conflict was skipped entirely — a fade can still be early'
    )


def test_the_halving_is_worth_exactly_six_points():
    with_conflict, _ = _score(_fade())
    no_conflict, _   = _score(_fade(macd_signal='NEUTRAL'))
    assert no_conflict - with_conflict == pytest.approx(6.0), (
        f'reversal macd_conflict should cost 6, cost {no_conflict - with_conflict}'
    )


# ── the point of the whole change ────────────────────────────────────────────

def test_a_textbook_fade_can_reach_the_fire_floor():
    """Measured: 43 before the exemptions, 72 after."""
    q, reasons = _score(_fade())
    assert q >= MIN_FIRE_QUALITY, (
        f'a fade at resistance with RSI 74, a bearish engulfing and decaying '
        f'momentum scored {q}, below the {MIN_FIRE_QUALITY} floor — the strategy '
        f'is disqualified by arithmetic before the floor sees it.\n{reasons}'
    )


def test_the_remaining_gap_is_evidence_not_direction():
    """The honest boundary, and the reason this file does not simply lower the
    floor for fades.

    Drop ADX below the trending threshold and the same setup scores 57 — refused.
    That is NOT the counter-trend penalty reappearing; it is the floor declining a
    setup with thin evidence, which is exactly its job. A fade of a REAL trend
    picks up adx_trending (+15) and clears at 72, because a trend worth fading
    registers as a trend.

    Both numbers are measured, not asserted from theory.
    """
    weak_adx, _ = _score(_fade(adx=22.0, volume_zscore=0.2, edge_score=55.0))
    real_trend, _ = _score(_fade())
    assert weak_adx == pytest.approx(57.0)
    assert real_trend == pytest.approx(72.0)
    assert weak_adx < MIN_FIRE_QUALITY <= real_trend


def test_the_floor_still_refuses_a_weak_fade():
    """Parity, not a free pass. Strip the evidence and it must fail."""
    weak = _fade(rsi=57.0, price=104.0,          # not at the extreme
                 cdl_bear_reversal=0.0,           # no reversal candle
                 edge_score=10.0, volume_zscore=-1.2, adx=12.0)
    q, _ = _score(weak)
    assert q < MIN_FIRE_QUALITY, f'a fade with no evidence scored {q}'


def test_the_exemptions_did_not_inflate_trend_following():
    """The gap should close from the fade side only."""
    trend = {
        'price': 105.0, 'support': 100.0, 'resistance': 120.0,
        'rsi': 58.0, 'adx': 34.0, 'volume_zscore': 1.2,
        'edge_score': 70.0, 'confluence': {'total': 7.5},
        'macd_signal': 'BULLISH', 'market_bias': 'BULLISH',
        'macro_weekly': 1.0, 'macro_daily': 1.0,
        'lstm_available': True, 'lstm_continuation_prob': 0.80,
        'lstm_exhaustion_prob': 0.10, 'lstm_vol_expansion_prob': 0.5,
        'cdl_bull_reversal': 0.0, 'cdl_bear_reversal': 0.0,
        'funding_bias': 'NEUTRAL', 'oi_trend': 'STABLE',
    }
    q, reasons = _score(trend, side='BUY', regime='TRENDING_BULL')
    assert not any(r.startswith('lstm_exhaustion') or r.startswith('macd_conflict')
                   for r in reasons), 'a with-trend signal should hit neither penalty'
    assert q >= MIN_FIRE_QUALITY
