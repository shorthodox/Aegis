"""The shadow book measures the alternatives without ever trading them.

The exit study said the live give-back rule beats every runner variant, but it
said so on synthetic entries — every bar, both directions, a population with no
directional edge by construction. If the engine's selection carries drift the
ranking could flip, and only the engine's own signals can settle that.

These tests pin the two properties that make the answer trustworthy: the
shadows must be arithmetically right, and they must be incapable of touching
the live book.
"""
import inspect
import json

import pytest

from scripts.engine.shadow_exits import POLICIES, ShadowBook


ENTRY = 100.0
# a 0.5 / 1.5 / 2 / 3 / 3.5 percent ladder on a long, and a 1.5% stop
TPS = [100.5, 101.5, 102.0, 103.0, 103.5]
SL = 98.5


@pytest.fixture
def book(tmp_path):
    return ShadowBook(cost_pct=0.10, store=tmp_path / 'shadow.json')


def _open(book, tid='t1', direction='LONG', entry=ENTRY, sl=SL, tps=None):
    book.open(tid, 'TEST/USDT', direction, entry, sl, list(tps or TPS))
    return tid


def _walk(book, prices):
    for p in prices:
        book.tick({'TEST/USDT': p})


# ── it must not be able to trade ─────────────────────────────────────────────

def test_the_shadow_book_cannot_place_an_order():
    """The whole design rests on this being observation only."""
    src = inspect.getsource(ShadowBook)
    for forbidden in ('.open_trade(', '.close_trade(', 'self.wallet',
                      'open_positions', '.position_size(', 'create_order'):
        assert forbidden not in src, (
            f'ShadowBook calls {forbidden!r} — a study that can reach the '
            f'book is not a study')


def test_hooks_are_wrapped_so_a_study_bug_cannot_block_a_real_trade():
    from scripts.engine.positions import PositionsMixin
    src = inspect.getsource(PositionsMixin._open_position)
    i = src.index('shadow_book.open')
    assert 'try:' in src[max(0, i - 400):i], (
        'the open hook is unguarded — an exception in the shadow study would '
        'stop a live position from being registered')


# ── the arithmetic ───────────────────────────────────────────────────────────

def test_a_trade_that_never_moves_books_nothing_but_costs(book):
    _open(book)
    _walk(book, [100.0] * 5)
    book.record_live('t1', 0.0, 'FLAT')
    _walk(book, [100.0])
    # still open — no policy has exited, so nothing is written yet
    assert book.rows == []


def test_a_straight_run_to_the_top_pays_every_policy(book):
    _open(book)
    _walk(book, [100.4, 100.6, 101.6, 102.1, 103.1, 103.6])
    book.record_live('t1', 1.20, 'TP_GIVEBACK')
    _walk(book, [103.6])
    assert len(book.rows) == 1
    row = book.rows[0]
    for name, res in row['policies'].items():
        assert res['pct'] > 0.5, f'{name} booked {res["pct"]} on a clean run'


def test_the_giveback_policy_closes_the_runner_where_the_live_rule_does(book):
    """TP1 at +0.5%, give-back a fifth of the rung -> the remainder exits +0.40%."""
    _open(book)
    _walk(book, [100.6, 100.40])          # tag TP1, then hand back to the level
    # The row is not written yet, and must not be: the runner policies are
    # still holding, which is the disagreement being measured. Read the sim.
    live_sim = book._open['t1'].sims['live_rule']
    assert live_sim.done and live_sim.exit_reason == 'STOP'
    # 15% banked at +0.5%, 85% handed back at +0.40% (cost is charged on write)
    assert live_sim.exit_pct == pytest.approx(0.15 * 0.5 + 0.85 * 0.40, abs=1e-6)


def test_a_runner_policy_survives_the_dip_the_giveback_closes_on(book):
    """The disagreement the whole study exists to price."""
    _open(book)
    _walk(book, [100.6, 100.40])
    row_before = len(book.rows)
    # live rule is out; the wide-trail runner is still holding
    assert book._open['t1'].sims['live_rule'].done
    assert not book._open['t1'].sims['bank40_trail_wide'].done
    assert row_before == 0, 'the row was written before every policy finished'


def test_shadows_outlive_the_live_close(book):
    """Marking a shadow flat at the live exit would assume the answer."""
    _open(book)
    _walk(book, [100.6, 100.40])
    book.record_live('t1', 0.40, 'TP_GIVEBACK')
    assert 't1' in book._open, 'the trade was finalised while a policy was still open'
    _walk(book, [101.6, 102.1, 103.1, 103.6])      # the run the live rule missed
    assert len(book.rows) == 1
    row = book.rows[0]
    assert row['policies']['bank40_trail_wide']['pct'] > row['live_pct'], (
        'the runner rode the continuation and still did not beat the live exit')


def test_the_stop_is_checked_before_the_rungs(book):
    """Between two observations the order is unknowable; assuming the
    favourable one flatters every policy that carries a stop."""
    src = inspect.getsource(ShadowBook._advance)
    assert src.index('s.stop') < src.index("enumerate(tr.ladder)")


def test_a_stop_out_is_recorded_as_a_loss(book):
    _open(book)
    _walk(book, [99.0, 98.4])
    book.record_live('t1', -1.5, 'STOP_HIT')
    _walk(book, [98.4])
    row = book.rows[0]
    for name, res in row['policies'].items():
        assert res['pct'] < 0, f'{name} booked {res["pct"]} on a stop-out'
        assert res['reason'] == 'STOP'


@pytest.mark.parametrize('direction,mult', [('LONG', 1), ('SHORT', -1)])
def test_both_directions_measure_the_same_geometry(book, direction, mult):
    tps = [ENTRY * (1 + mult * p / 100) for p in (0.5, 1.5, 2.0, 3.0, 3.5)]
    sl = ENTRY * (1 - mult * 1.5 / 100)
    _open(book, direction=direction, sl=sl, tps=tps)
    _walk(book, [ENTRY * (1 + mult * 0.6 / 100), ENTRY * (1 + mult * 0.40 / 100)])
    live = book._open['t1'].sims['live_rule']
    assert live.done and live.exit_pct == pytest.approx(0.15 * 0.5 + 0.85 * 0.40, abs=1e-6)


# ── policy parameters scale with the trade ───────────────────────────────────

def test_policy_stops_scale_with_a_compressed_ladder():
    """The objective cap compresses the ladder, so absolute percentages would
    mean different things on different trades."""
    src = inspect.getsource(ShadowBook._advance)
    assert 'tp1' in src and '* tp1' in src, (
        'policy parameters are absolute again — they must be multiples of the '
        "trade's own TP1 so a compressed ladder scales them")


def test_the_live_rule_is_simulated_too_as_a_control():
    """Without it, a difference could be the simulator rather than the policy."""
    assert 'live_rule' in POLICIES
    assert POLICIES['live_rule'].get('giveback') is True


# ── output ───────────────────────────────────────────────────────────────────

def test_the_summary_reports_each_policy_against_live(book, tmp_path):
    _open(book)
    _walk(book, [100.6, 100.40])
    book.record_live('t1', 0.40, 'TP_GIVEBACK')
    _walk(book, [101.6, 102.1, 103.1, 103.6])
    s = book.summary()
    assert s['n'] == 1
    assert 'live_actual' in s
    for name in POLICIES:
        assert 'vs_control' in s['policies'][name]
    # the control must sit at exactly zero against itself
    assert s['policies']['live_rule']['vs_control'] == 0.0
    # and the drift between the simulated control and the real engine is
    # reported, because a large one voids the whole comparison
    assert 'control_tracks_reality' in s


def test_results_survive_a_restart(book, tmp_path):
    _open(book)
    _walk(book, [100.6, 100.40])
    book.record_live('t1', 0.40, 'TP_GIVEBACK')
    _walk(book, [101.6, 102.1, 103.1, 103.6])
    again = ShadowBook(cost_pct=0.10, store=tmp_path / 'shadow.json')
    assert len(again.rows) == 1
    saved = json.loads((tmp_path / 'shadow.json').read_text(encoding='utf-8'))
    assert saved['summary']['n'] == 1
    assert 'no live order' in saved['note']


def test_a_malformed_position_is_ignored_not_raised(book):
    book.open('bad', 'X/USDT', 'LONG', 0.0, 0.0, [])
    book.open('bad2', 'X/USDT', 'LONG', 100.0, 98.0, [0, 0, 0, 0, 0])
    assert book._open == {}
