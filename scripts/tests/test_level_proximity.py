"""Regression tests for the v82e level-proximity fix.

The engine had two contradictory notions of "at a level":

  * src/trading/gate_scorer.py measured real ATR distance and raised
    FAR_FROM_SR past AT_LEVEL_ATR — but that veto only downgraded the tier to
    RISKY, it never blocked.
  * scripts/live_engine.py chose entry_mode (which sets the STOP WIDTH) from
    range_position, the exact proxy gate_scorer's own comment rejects.

Measured on 8,484 1h bars across 12 tokens: 58.7% of bars sit more than 1 ATR
from the nearest 24h level, median gap 1.17 ATR, and only 15.6% are within
0.35 ATR. That is why 6 of the 7 signals open on 2026-08-01 were tagged RISKY.
"""
import pytest

from scripts.live_engine import LiveEngine, _HARD_VETOES
import src.trading.gate_scorer as gate_scorer


# ── the veto now blocks ──────────────────────────────────────────────────────

def test_far_from_sr_is_a_hard_veto():
    assert 'FAR_FROM_SR' in _HARD_VETOES, (
        'FAR_FROM_SR is back to a tier downgrade — the engine will again fire '
        'trades it has itself measured as not at their level'
    )


def test_no_valid_sr_is_deliberately_not_hard():
    """A weaker, differently-shaped condition; still only downgrades."""
    assert 'NO_VALID_SR' not in _HARD_VETOES


def test_protective_vetoes_are_still_hard():
    for v in ('MODEL_DRIFT_CRITICAL', 'DEAD_MARKET', 'EXTREME_VOLATILITY'):
        assert v in _HARD_VETOES


# ── the ATR gap helper ───────────────────────────────────────────────────────

def _r(price, atr_pct, support=0.0, resistance=0.0):
    return {'price': price, 'atr_pct': atr_pct,
            'support': support, 'resistance': resistance}


def test_gap_is_measured_to_the_level_the_side_leans_on():
    gap = LiveEngine._level_gap_atr
    # BUY leans on support, SELL on resistance
    r = _r(100.0, 1.0, support=99.0, resistance=104.0)
    assert gap(r, 'BUY') == pytest.approx(1.0)
    assert gap(r, 'SELL') == pytest.approx(4.0)


def test_gap_is_negative_once_price_trades_through_the_level():
    """Premise gone: this is fading a level that already failed."""
    assert LiveEngine._level_gap_atr(
        _r(100.0, 1.0, resistance=99.0), 'SELL') == pytest.approx(-1.0)


def test_gap_returns_none_when_it_cannot_be_measured():
    """None must fall back, never be treated as 'at the level'."""
    gap = LiveEngine._level_gap_atr
    assert gap(_r(100.0, 1.0), 'SELL') is None          # no level
    assert gap(_r(0.0, 1.0, resistance=104.0), 'SELL') is None   # no price
    assert gap(_r(100.0, 0.0, resistance=104.0), 'SELL') is None  # no ATR
    assert gap({'price': 'x', 'atr_pct': 1.0, 'support': 9}, 'BUY') is None
    assert gap(_r(100.0, 1.0, support=99.0), 'FLAT') is None


def test_gap_prefers_absolute_atr_over_atr_pct():
    r = {'price': 100.0, 'atr': 2.0, 'atr_pct': 1.0, 'resistance': 104.0}
    assert LiveEngine._level_gap_atr(r, 'SELL') == pytest.approx(2.0)


# ── the two real signals from 2026-08-01 ─────────────────────────────────────

INJ = _r(4.905, 0.86, support=4.760, resistance=4.957)      # RISKY, losing
PENDLE = _r(1.458, 0.88, support=1.392, resistance=1.463)   # NORMAL, winning


def test_the_losing_risky_signal_would_now_be_blocked():
    gap = LiveEngine._level_gap_atr(INJ, 'SELL')
    assert gap == pytest.approx(1.23, abs=0.02)
    assert gap > gate_scorer.AT_LEVEL_ATR, 'INJ would still fire'


def test_the_winning_signal_would_still_fire():
    gap = LiveEngine._level_gap_atr(PENDLE, 'SELL')
    assert gap == pytest.approx(0.39, abs=0.02)
    assert gap <= gate_scorer.AT_LEVEL_ATR, 'PENDLE would be blocked'


def test_the_two_thresholds_are_ordered():
    """Blocking is looser than the tight-stop bar, so there is a middle band:
    <=0.35 ATR at_level (tight stop), 0.35-1.0 mid_range (wider), >1.0 blocked."""
    assert LiveEngine.AT_LEVEL_ATR < gate_scorer.AT_LEVEL_ATR


# ── the proxy that used to decide stop width ─────────────────────────────────

def test_entry_mode_no_longer_keys_off_range_position_alone():
    import inspect
    src = inspect.getsource(LiveEngine._process_symbol)
    i = src.index("_entry_mode = 'model_at_level'")
    window = src[max(0, i - 700):i]
    assert '_level_gap_atr' in window, (
        'entry_mode is choosing the stop width from range_position again'
    )
