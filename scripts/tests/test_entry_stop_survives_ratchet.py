"""A trade that ratchets to break-even must still report the risk it was taken with.

CRV/USDT, 2026-08-14. A real +1.40% win — TP1 tagged at the v87 1.5% rung, less
0.10% round-trip cost. It published:

    entry 0.2513    stop 0.2513    distance 0.0000%

Not a bad stop. Not a data error. The break-even ratchet at exits.py:346 sets
`pos.stop_loss = pos.entry_price` the moment TP1 is tagged, and the track record
wrote that mutated value as though it were the risk the trade was opened with.

The win rate survives this. Expectancy does not, and expectancy is where the
published record is heading. Worse, the downstream consumer guarded against the
division rather than the cause:

    if ep > 0 and sl > 0 and abs(ep - sl) > 1e-9:      # aegis_forensics.py

which silently DROPS every ratcheted trade from the R sample. Every dropped
trade is a winner, because only winners reach TP1 — so avg_r was computed over
losers and unratcheted winners, biased downward by construction. A guard that
turns a wrong number into a quietly missing one is not a fix.

entry_stop is set once at open and never mutated. These tests pin that.
"""
import pytest

from scripts.engine.models import Position, TradeRecord


def _pos(**kw):
    base = dict(
        symbol='CRV/USDT', direction='LONG', side='BUY',
        entry_price=0.2513, position_value=100.0,
        stop_loss=0.24954, entry_stop=0.24954,
        signal_id='t1', entry_time='2026-08-14T04:57:51+00:00',
        meta_confidence=0.6, atr_multiplier=1.5,
    )
    base.update(kw)
    return Position(**base)


def test_ratchet_moves_stop_loss_but_not_entry_stop():
    """The exact CRV sequence: TP1 tagged, stop to break-even."""
    pos = _pos()
    assert pos.stop_loss == pos.entry_stop        # identical at open

    pos.stop_loss = pos.entry_price               # exits.py:346

    assert pos.stop_loss == 0.2513, 'live stop should ratchet to break-even'
    assert pos.entry_stop == 0.24954, 'entry stop must NOT move'
    assert pos.entry_stop != pos.stop_loss


def test_R_is_finite_after_the_ratchet():
    """The defect, stated as arithmetic.

    Risk from stop_loss is zero after the ratchet, so R is undefined — which is
    how a real +1.40% win came to be published as a zero-risk trade. Risk from
    entry_stop is 0.70% of entry, so R is a finite ~2.0.
    """
    pos = _pos()
    pos.stop_loss = pos.entry_price               # ratchet
    pnl_pct = 1.40

    risk_from_live = abs(pos.entry_price - pos.stop_loss) / pos.entry_price
    assert risk_from_live == 0.0, 'precondition: the live stop yields zero risk'

    risk_from_entry = abs(pos.entry_price - pos.entry_stop) / pos.entry_price
    assert risk_from_entry == pytest.approx(0.0070, abs=1e-4)

    r_multiple = (pnl_pct / 100.0) / risk_from_entry
    assert r_multiple == pytest.approx(2.0, abs=0.05)
    assert 0.0 < r_multiple < 100.0, 'R must be finite and sane'


def test_trade_record_carries_both_stops():
    """The record must keep the ratcheted stop AND the risk actually taken."""
    rec = TradeRecord(
        signal_id='t1', symbol='CRV/USDT', direction='LONG', side='BUY',
        entry_price=0.2513, exit_price=0.2551,
        entry_time='2026-08-14T04:57:51+00:00', close_time='2026-08-14T05:30:00+00:00',
        pnl_pct=1.40, pnl_usdt=1.40, outcome='WIN', exit_reason='TP1_PARTIAL',
        meta_confidence=0.6, position_value=100.0,
        stop_loss=0.2513,      # ratcheted
        entry_stop=0.24954,    # as opened
    )
    assert rec.stop_loss == rec.entry_price, 'ratcheted stop preserved for display'
    assert rec.entry_stop != rec.entry_price, 'entry stop preserved for R'


def test_entry_stop_defaults_to_zero_for_legacy_records():
    """Pre-existing records have no entry_stop; 0.0 means UNKNOWN, not zero risk.

    Consumers must decline to compute R rather than divide by it. The archived
    CRV row is exactly this case — its entry stop was overwritten before the
    field existed and is not recoverable.
    """
    rec = TradeRecord(
        signal_id='old', symbol='X/USDT', direction='LONG', side='BUY',
        entry_price=1.0, exit_price=1.01, entry_time='', close_time='',
        pnl_pct=1.0, pnl_usdt=1.0, outcome='WIN', exit_reason='TP1',
        meta_confidence=0.5, position_value=100.0,
    )
    assert rec.entry_stop == 0.0

    sl = rec.entry_stop or rec.stop_loss
    assert not (rec.entry_price > 0 and sl > 0), 'legacy record must be skipped, not divided by'
