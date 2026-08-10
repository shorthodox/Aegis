"""The win rate reports the stop. This is the number that reports the signal.

Measured over 13,560 paths on 30 tokens, the hit rate tracks stop/(target+stop)
to within about a point at every pair tried — including 0.50%/1.40%, which
predicts 73.7% and is the geometry behind the live book's 75%. So a headline
win rate can be moved to 80% by widening the stop to 4%, losing no signals and
earning nothing.

edge = measured hit - mean(stop / (target + stop))

These tests pin the two properties that make it useful: it must be flat when
only the stop moves, and it must rise when the signals actually get better.
"""
import pytest

from scripts.engine.edge_metric import (by_group, geometric_hit_rate, measure,
                                        _reached_target)


def _sig(target_pct, stop_pct, reached, entry=100.0, side='LONG', **kw):
    tp1 = entry * (1 + target_pct / 100) if side == 'LONG' else entry * (1 - target_pct / 100)
    sl = entry * (1 - stop_pct / 100) if side == 'LONG' else entry * (1 + stop_pct / 100)
    base = dict(entry_price=entry, take_profit_1=tp1, stop_loss=sl,
                tp_hits=1 if reached else 0, outcome='WIN' if reached else 'LOSS',
                direction=side)
    base.update(kw)
    return base


# ── the geometry ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize('tgt,stop,expected', [
    (1.0, 1.3, 0.565), (0.5, 1.4, 0.737), (1.0, 1.0, 0.500),
    (1.0, 2.33, 0.700), (1.0, 4.0, 0.800), (2.0, 1.0, 0.333),
])
def test_the_driftless_prediction(tgt, stop, expected):
    assert geometric_hit_rate(tgt, stop) == pytest.approx(expected, abs=0.002)


def test_the_old_ladder_predicts_the_live_win_rate():
    """0.5% target against a 1.4% stop is 73.7%. The board showed 75% and lost
    money — the win rate was the stop distance."""
    assert geometric_hit_rate(0.5, 1.4) == pytest.approx(0.737, abs=0.002)


def test_a_degenerate_barrier_is_not_a_coin_flip():
    """Returning 0.5 for missing geometry would understate the edge on exactly
    the trades that have no data."""
    assert geometric_hit_rate(0.0, 1.0) is None
    assert geometric_hit_rate(1.0, 0.0) is None
    assert geometric_hit_rate(-1.0, 1.0) is None


# ── the property that makes it worth having ──────────────────────────────────

def test_widening_the_stop_does_not_manufacture_edge():
    """The whole point. A book that hits its target exactly as often as
    geometry says must read zero edge at ANY stop distance."""
    for stop in (0.5, 1.0, 1.3, 2.33, 4.0, 9.0):
        g = geometric_hit_rate(1.0, stop)
        n = 1000
        wins = round(g * n)
        sigs = ([_sig(1.0, stop, True)] * wins
                + [_sig(1.0, stop, False)] * (n - wins))
        out = measure(sigs)
        assert abs(out['edge_pp']) < 0.15, (
            f'stop {stop}% reported {out["edge_pp"]}pp of edge on a book that '
            f'is exactly average — the metric is tracking the stop')
        # ...while the headline win rate swings enormously
    assert measure([_sig(1.0, 4.0, True)] * 800 + [_sig(1.0, 4.0, False)] * 200
                   )['measured_hit'] == pytest.approx(80.0)


def test_a_genuinely_better_book_reads_positive():
    sigs = [_sig(1.0, 1.3, True)] * 700 + [_sig(1.0, 1.3, False)] * 300
    out = measure(sigs)
    assert out['measured_hit'] == pytest.approx(70.0)
    assert out['geometric_hit'] == pytest.approx(56.52, abs=0.05)
    assert out['edge_pp'] > 13


def test_a_worse_book_reads_negative():
    sigs = [_sig(1.0, 1.3, True)] * 400 + [_sig(1.0, 1.3, False)] * 600
    assert measure(sigs)['edge_pp'] < -15


def test_mixed_geometries_are_averaged_per_trade():
    """Tokens run different stops, so the baseline is a mean of each trade's
    own geometry, not the geometry of the average trade."""
    sigs = [_sig(1.0, 1.0, True)] * 100 + [_sig(1.0, 4.0, True)] * 100
    out = measure(sigs)
    assert out['geometric_hit'] == pytest.approx((50.0 + 80.0) / 2, abs=0.1)


# ── data hygiene ─────────────────────────────────────────────────────────────

def test_open_positions_are_excluded():
    sigs = [_sig(1.0, 1.3, True), dict(_sig(1.0, 1.3, False), outcome='OPEN')]
    assert measure(sigs)['n'] == 1


def test_a_record_with_no_outcome_is_skipped_not_counted_as_a_miss():
    """Counting unknowns as failures would manufacture a negative edge."""
    bad = {'entry_price': 100.0, 'take_profit_1': 101.0, 'stop_loss': 98.7}
    out = measure([_sig(1.0, 1.3, True)] * 10 + [bad] * 10)
    assert out['n'] == 10 and out['skipped'] == 10


def test_a_record_with_no_levels_is_skipped():
    out = measure([_sig(1.0, 1.3, True)] * 5
                  + [{'tp_hits': 0, 'outcome': 'LOSS'}] * 5)
    assert out['n'] == 5 and out['skipped'] == 5


@pytest.mark.parametrize('reason,expected', [
    ('TP1_PARTIAL', True), ('TP_GIVEBACK', True), ('STOP_HIT', False),
    ('SL_HIT', False), ('', None), ('MODEL_REVERSAL', None),
])
def test_exit_reason_is_the_fallback_when_tp_hits_is_absent(reason, expected):
    assert _reached_target({'exit_reason': reason}) is expected


def test_tp_hits_wins_over_exit_reason():
    """tp_hits counts rungs actually banked; the reason string is a label."""
    assert _reached_target({'tp_hits': 1, 'exit_reason': 'STOP_HIT'}) is True


def test_an_empty_book_says_so_rather_than_dividing_by_zero():
    assert measure([])['n'] == 0


def test_a_small_sample_is_labelled_unreadable():
    out = measure([_sig(1.0, 1.3, True)] * 5)
    assert 'too few' in out['verdict']


def test_the_verdict_names_the_zero_edge_case():
    g = geometric_hit_rate(1.0, 1.3)
    n = 400
    sigs = ([_sig(1.0, 1.3, True)] * round(g * n)
            + [_sig(1.0, 1.3, False)] * (n - round(g * n)))
    assert 'reporting the stop' in measure(sigs)['verdict']


# ── grouping ─────────────────────────────────────────────────────────────────

def test_grouping_ranks_by_edge_not_by_win_rate():
    """A token on a wide stop wins more often and may still be the worse one."""
    sigs = ([dict(_sig(1.0, 4.0, True), symbol='WIDE/USDT')] * 80
            + [dict(_sig(1.0, 4.0, False), symbol='WIDE/USDT')] * 20
            + [dict(_sig(1.0, 1.0, True), symbol='TIGHT/USDT')] * 65
            + [dict(_sig(1.0, 1.0, False), symbol='TIGHT/USDT')] * 35)
    out = by_group(sigs, 'symbol')
    assert out['WIDE/USDT']['measured_hit'] > out['TIGHT/USDT']['measured_hit']
    assert out['TIGHT/USDT']['edge_pp'] > out['WIDE/USDT']['edge_pp']
    assert list(out)[0] == 'TIGHT/USDT', 'ranked by win rate instead of edge'
