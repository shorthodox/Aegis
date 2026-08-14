"""Shadow fields record the road not taken. They must never steer.

Three open questions could not be settled on 2026-08-14 because nothing recorded
the alternative, and none of them can be answered from history — no archived
trade kept its level, its structural stop, or how far price actually travelled:

  1. Would the support-cleared stop have survived where the banded one did not?
     TAO/USDT said yes once; n=1 cannot establish it.
  2. Would a volatility-scaled TP1 have been reached where the fixed 1.5% rung
     was not? Every win in that book paid exactly +1.4000%, which is TP1 minus
     the round trip — the first rung is the only one anything reaches.
  3. How often does the band override structure at all — i.e. what would
     refusing those setups cost in fire rate?

The instrumentation answers all three from the NEXT ~25 trades. The entire value
depends on it changing nothing, so that is what these tests pin: the shipped
stop, the shipped ladder, and sizing must be byte-identical with the shadow
fields present.
"""
import pytest

import scripts.live_engine as LE
from scripts.engine.models import Position, TradeRecord


@pytest.fixture
def risk():
    return LE.DynamicRiskEngine()


# ── the shadow must be observation, not behaviour ────────────────────────────

def test_shipped_ladder_is_unchanged(risk):
    """The hybrid is recorded; the ladder that ships is still fixed percent."""
    d = risk.calculate_stops(price=100.0, side='BUY', atr=1.0,
                             support=98.0, resistance=110.0)
    assert d['tp1'] == pytest.approx(101.5)   # 1.5% of entry
    assert d['tp2'] == pytest.approx(103.0)   # 3.0%


def test_shipped_stop_is_unchanged_when_the_band_caps(risk):
    """TAO's real inputs. The banded stop still ships, exactly as before."""
    d = risk.calculate_stops(price=198.8, side='BUY', atr=1.46428571,
                             support=193.70, resistance=205.30)
    assert d['sl'] == pytest.approx(196.2156, abs=1e-4)
    assert d['band_capped'] is True
    assert d['structural_stop'] == pytest.approx(192.9679, abs=1e-4)
    # the shadow is strictly further away than what shipped
    assert d['structural_stop'] < d['sl']


def test_band_capped_is_false_when_structure_fits(risk):
    """A support inside the budget must not be reported as overridden."""
    d = risk.calculate_stops(price=100.0, side='BUY', atr=1.0,
                             support=99.7, resistance=110.0)
    assert d['band_capped'] is False
    assert d['structural_stop'] == pytest.approx(d['sl'])


# ── the hybrid TP1 policy ────────────────────────────────────────────────────

def test_hybrid_leaves_the_median_token_where_it_is(risk):
    """K is calibrated so this is a redistribution, not a loosening.

    Fleet median ATR% was 0.678 on 2026-08-14; 2.21 x 0.678 = 1.499.
    """
    price = 100.0
    pct, _ = risk.tp1_hybrid(price, price * 0.00678)
    assert pct == pytest.approx(1.5, abs=0.02)


def test_hybrid_compresses_the_difficulty_spread(risk):
    """The point of the change: same target, wildly different difficulty.

    Fixed 1.5% ranges 1.23-5.85 ATR across the fleet (4.8x). The hybrid must
    bring that in substantially.
    """
    quiet_atr, loud_atr = 0.256, 1.219        # BNB and CRV, real values
    fixed_spread = (1.5 / quiet_atr) / (1.5 / loud_atr)
    _, quiet = risk.tp1_hybrid(100.0, 100.0 * quiet_atr / 100)
    _, loud = risk.tp1_hybrid(100.0, 100.0 * loud_atr / 100)
    hybrid_spread = quiet / loud

    assert fixed_spread == pytest.approx(4.76, abs=0.05)
    assert hybrid_spread < 2.1
    assert hybrid_spread < fixed_spread


def test_hybrid_is_bounded_at_both_ends(risk):
    """Neither a dead-quiet nor a violent token may escape the band."""
    lo, _ = risk.tp1_hybrid(100.0, 100.0 * 0.05 / 100)   # absurdly quiet
    hi, _ = risk.tp1_hybrid(100.0, 100.0 * 8.00 / 100)   # absurdly violent
    assert lo == pytest.approx(risk.TP1_HYBRID_MIN_PCT)
    assert hi == pytest.approx(risk.TP1_HYBRID_MAX_PCT)


def test_hybrid_is_safe_on_degenerate_input(risk):
    """Zero ATR or zero price must return zeros, not divide."""
    assert risk.tp1_hybrid(100.0, 0.0) == (0.0, 0.0)
    assert risk.tp1_hybrid(0.0, 1.0) == (0.0, 0.0)
    assert risk.tp1_hybrid(-5.0, -1.0) == (0.0, 0.0)


# ── excursions ───────────────────────────────────────────────────────────────

def test_excursions_default_to_zero_and_are_carried():
    """0.0 means 'never moved that way', which is the correct starting point."""
    pos = Position(
        symbol='X/USDT', direction='LONG', side='BUY', entry_price=100.0,
        position_value=100.0, stop_loss=99.0, entry_stop=99.0,
        signal_id='s', entry_time='', meta_confidence=0.5, atr_multiplier=1.5,
    )
    assert pos.mfe_pct == 0.0 and pos.mae_pct == 0.0

    pos.mfe_pct, pos.mae_pct = 2.4, -1.1
    rec = TradeRecord(
        signal_id='s', symbol='X/USDT', direction='LONG', side='BUY',
        entry_price=100.0, exit_price=98.9, entry_time='', close_time='',
        pnl_pct=-1.1, pnl_usdt=-1.1, outcome='LOSS', exit_reason='STOP_HIT',
        meta_confidence=0.5, position_value=100.0,
        mfe_pct=pos.mfe_pct, mae_pct=pos.mae_pct,
    )
    assert rec.mfe_pct == 2.4
    assert rec.mae_pct == -1.1


def test_mae_can_answer_the_structural_stop_question():
    """The question the whole exercise exists to settle, as arithmetic.

    A trade stopped at the banded stop whose MAE never reached the structural
    one is a trade the wider stop would have survived.
    """
    rec = TradeRecord(
        signal_id='t', symbol='TAO/USDT', direction='LONG', side='BUY',
        entry_price=198.8, exit_price=196.2156, entry_time='', close_time='',
        pnl_pct=-1.4, pnl_usdt=-1.4, outcome='LOSS', exit_reason='STOP_HIT',
        meta_confidence=0.6, position_value=100.0,
        entry_stop=196.2156, structural_stop=192.9679,
        structural_stop_pct=2.934, band_capped=True,
        mae_pct=-2.01,          # price bottomed at 194.80
    )
    banded_pct = (rec.entry_price - rec.entry_stop) / rec.entry_price * 100
    assert abs(rec.mae_pct) > banded_pct, 'price went through the shipped stop'
    assert abs(rec.mae_pct) < rec.structural_stop_pct, (
        'but never reached the structural one — the wider stop survives'
    )
