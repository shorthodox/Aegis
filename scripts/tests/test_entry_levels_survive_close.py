"""The level a trade leaned on must outlive the trade.

TAO/USDT and BCH/USDT, 2026-08-14. Both called direction correctly, both were
stopped out, and price then went where the signal said. The question that
decides whether that was bad luck or bad geometry is one line long:

    was the stop inside the level, or beyond it?

It could not be answered from the record. Position has carried entry_support
and entry_resistance since v82 — TradeRecord dropped them, so the moment a
trade closed the level was gone. TAO was only answerable because a live chart
still displayed 193.70 and the arithmetic happened to reproduce its stop to four
decimal places. BCH was not answerable at all.

Same defect as entry_stop, and the same shape as everything else catalogued in
SYSTEMS_REVIEW.md: the record keeps the numbers that are convenient to write
rather than the ones needed to audit the decision.

These fields are also the input to a question that is currently unanswerable:
"how many past setups had their invalidation beyond the stop budget?" — i.e. the
fire-rate cost of refusing those trades. Without the levels, that cannot be
computed from history at all.
"""
import pytest

from scripts.engine.models import Position, TradeRecord


def _pos(**kw):
    base = dict(
        symbol='TAO/USDT', direction='LONG', side='BUY',
        entry_price=198.8, position_value=100.0,
        stop_loss=196.2156, entry_stop=196.2156,
        entry_support=193.70, entry_resistance=205.30,
        signal_id='t1', entry_time='2026-08-14T13:04:24+00:00',
        meta_confidence=0.6, atr_multiplier=1.5,
    )
    base.update(kw)
    return Position(**base)


def test_levels_are_not_mutated_by_the_ratchet():
    """Whatever the exit logic does to the stop, the level stays put."""
    pos = _pos()
    pos.stop_loss = pos.entry_price          # break-even ratchet, exits.py:346

    assert pos.entry_support == 193.70
    assert pos.entry_resistance == 205.30


def test_record_can_answer_was_the_stop_inside_the_level():
    """The whole point: a closed record must support the audit question.

    TAO's real numbers. The stop sat at 196.2156 with support at 193.70, so the
    stop was 2.5844 above entry-side invalidation — inside the level, which is
    where the market collects it.
    """
    rec = TradeRecord(
        signal_id='t1', symbol='TAO/USDT', direction='LONG', side='BUY',
        entry_price=198.8, exit_price=194.8,
        entry_time='2026-08-14T13:04:24+00:00', close_time='2026-08-14T14:56:12+00:00',
        pnl_pct=-2.1126, pnl_usdt=-2.11, outcome='LOSS', exit_reason='STOP_HIT',
        meta_confidence=0.6, position_value=100.0,
        stop_loss=196.2156, entry_stop=196.2156,
        entry_support=193.70, entry_resistance=205.30,
    )

    assert rec.entry_stop > rec.entry_support, (
        'stop sits ABOVE support — inside the level it leans on'
    )
    stop_dist = (rec.entry_price - rec.entry_stop) / rec.entry_price
    level_dist = (rec.entry_price - rec.entry_support) / rec.entry_price
    assert stop_dist == pytest.approx(0.0130, abs=1e-4)
    assert level_dist == pytest.approx(0.0257, abs=1e-4)
    assert stop_dist < level_dist, 'the stop was nearer than the invalidation'


def test_zero_means_unknown_not_absent():
    """Legacy rows carry 0.0. Consumers must skip them, not read them as 'no level'.

    Every archived row from before this field existed is 0.0 — the levels were
    never persisted and are genuinely unrecoverable. Treating 0.0 as a real
    support would make every one of them look like a stop beyond its level.
    """
    rec = TradeRecord(
        signal_id='old', symbol='X/USDT', direction='LONG', side='BUY',
        entry_price=100.0, exit_price=99.0, entry_time='', close_time='',
        pnl_pct=-1.0, pnl_usdt=-1.0, outcome='LOSS', exit_reason='STOP_HIT',
        meta_confidence=0.5, position_value=100.0, entry_stop=99.0,
    )
    assert rec.entry_support == 0.0
    assert rec.entry_resistance == 0.0

    auditable = rec.entry_support > 0 and rec.entry_stop > 0
    assert not auditable, 'a row without a level must be excluded from the audit'
