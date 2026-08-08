"""Cut losers on a reversal; do not cut winners short of their first target.

The measured problem, from the closed book:

    winners realised  +0.19R .. +0.40R    6-28 % of their target
    losers  realised  -1.08R .. -1.11R    the full stop

    payoff 0.39 : 1  ->  needs a 72 % win rate to break even
    actual 60 %      ->  -0.32 % per trade after costs

Nothing in the TP ladder can produce that: TP1 sits at 1.0R and those winners
exited at a third of it, so they never reached the first rung. The model-reversal
exit did it — it closed on any opposing signal clearing the SAME edge floor
required to ENTER (60), at any PnL.

The asymmetry is structural. The stop is ~1R away and gets hit in a bar or two;
a reversal needs a full re-score. So losers reach the stop and winners get
reversed out early — the engine cut winners and let losers run.

The reversal keeps its real job (abandon a dead thesis BEFORE the stop) and
loses the one it did badly. In profit but short of TP1, the position is
protected instead of closed: stop to break-even, so the bad case is a scratch
rather than a small win and the good case is still open.
"""
import time
from datetime import datetime, timezone

import pytest

from scripts.live_engine import (
    DriftMonitor, DynamicRiskEngine, LiveEngine, PerformanceTracker,
    Position, VirtualWallet,
)

SYM = 'TEST/USDT'


@pytest.fixture
def engine(tmp_path):
    e = LiveEngine.__new__(LiveEngine)
    e.wallet = VirtualWallet(10_000.0, 1_000.0, track_record_path=tmp_path / 'tr.json')
    e.risk_engine = DynamicRiskEngine()
    e.perf_tracker = PerformanceTracker()
    e.drift_monitor = DriftMonitor()
    e.live_prices, e._open_time, e._peak_price = {}, {}, {}
    e._tp1_hit, e._tp2_hit, e._tp3_hit, e._tp4_hit = {}, {}, {}, {}
    e._giveback_stop = {}
    e._last_close_time, e._last_close_side = {}, {}
    e._last_close_reason, e._last_loss_time = {}, {}
    e.last_signals = {}
    e._save_track_record = lambda: None
    e.MIN_HOLD_SECONDS = 0
    e.adaptive_orchestrator = type('O', (), {
        '__getattr__': lambda s, n: (lambda *a, **k: None)})()
    return e


def _pos(direction='LONG', **kw):
    long_ = direction == 'LONG'
    base = dict(
        symbol=SYM, direction=direction, side='BUY' if long_ else 'SELL',
        entry_price=100.0, position_value=1000.0, initial_value=1000.0,
        stop_loss=99.0 if long_ else 101.0, signal_id='sig-1',
        entry_time=datetime.now(timezone.utc).isoformat(),
        meta_confidence=0.0, atr_multiplier=1.8, atr=0.5,
        take_profit_1=101.0 if long_ else 99.0,
        take_profit_2=102.0 if long_ else 98.0,
        take_profit_3=105.0 if long_ else 95.0,
        take_profit_4=108.0 if long_ else 92.0,
        take_profit_5=112.0 if long_ else 88.0,
    )
    base.update(kw)
    return Position(**base)


def _reverse(engine, pos, price, edge=80.0):
    """Feed one scan where the model has flipped against the position."""
    engine.wallet.open_positions[SYM] = pos
    engine._open_time.setdefault(SYM, time.time() - 10_000)
    engine._peak_price.setdefault(SYM, pos.entry_price)
    engine.live_prices[SYM] = price
    opposing = 'SELL' if pos.direction == 'LONG' else 'BUY'
    result = {'atr': pos.atr, 'side': opposing, 'fire': True, 'edge_score': edge}
    engine._manage_exit(SYM, pos, result, price)
    return engine.wallet.open_positions.get(SYM)


# ── a winner below TP1 is protected, not closed ──────────────────────────────

@pytest.mark.parametrize('direction,price', [('LONG', 100.4), ('SHORT', 99.6)])
def test_reversal_in_profit_below_tp1_does_not_close(engine, direction, price):
    """The exact case that produced 0.19R-0.40R winners."""
    pos = _pos(direction)
    still_open = _reverse(engine, pos, price)
    assert still_open is not None, (
        'a profitable position short of TP1 was closed on a reversal — this is '
        'the behaviour that made winners a third of losers'
    )
    assert not engine.wallet.trade_history, 'nothing should have been booked'


@pytest.mark.parametrize('direction', ['LONG', 'SHORT'])
def test_reversal_in_profit_moves_the_stop_to_break_even(engine, direction):
    pos = _pos(direction)
    price = 100.4 if direction == 'LONG' else 99.6
    _reverse(engine, pos, price)
    assert pos.stop_loss == pytest.approx(pos.entry_price), (
        'the downside was not capped — protection must make the bad case a '
        'scratch, not a small win'
    )


def test_protection_never_loosens_an_already_tighter_stop(engine):
    """A stop already better than break-even must not be pushed back."""
    pos = _pos('LONG', stop_loss=100.5)          # already locked in profit
    _reverse(engine, pos, 100.8)
    assert pos.stop_loss == pytest.approx(100.5)


# ── a loser is still cut ─────────────────────────────────────────────────────

@pytest.mark.parametrize('direction,price', [('LONG', 99.5), ('SHORT', 100.5)])
def test_reversal_at_a_loss_still_closes(engine, direction, price):
    """The reversal's real job: abandon a dead thesis BEFORE the stop."""
    pos = _pos(direction)
    still_open = _reverse(engine, pos, price)
    assert still_open is None, 'a losing position was not cut on a reversal'
    assert engine.wallet.trade_history[-1].exit_reason == 'MODEL_REVERSAL_TP'


def test_reversal_at_break_even_still_closes(engine):
    """Flat is not profit; there is nothing to protect."""
    pos = _pos('LONG')
    assert _reverse(engine, pos, 100.0) is None


def test_a_profit_smaller_than_the_round_trip_is_not_protected(engine):
    """Below the cost of trading, the 'profit' is not real."""
    pos = _pos('LONG')
    cost = engine.wallet.round_trip_cost_pct()          # 0.10 %
    price = 100.0 * (1 + cost / 100.0 * 0.5)            # half the round trip
    assert _reverse(engine, pos, price) is None


# ── after TP1 the old behaviour stands ───────────────────────────────────────

def test_reversal_after_tp1_still_closes(engine):
    """Once the risk is earned, a reversal is a legitimate exit.

    Which mechanism books it is not the point, and it changed: with TP1 at 101
    and the position back to 100.8, a fifth of the rung has already been handed
    back, so the give-back ratchet now closes it before the reversal branch is
    reached — at its own level rather than at market. That is the better of the
    two exits, so this asserts the position closed in profit rather than
    pinning the label.
    """
    pos = _pos('LONG')
    engine._tp1_hit[SYM] = True
    assert _reverse(engine, pos, 100.8) is None
    closed = engine.wallet.trade_history[-1]
    assert closed.exit_reason in ('MODEL_REVERSAL_TP', 'TP_GIVEBACK'), closed.exit_reason
    assert closed.exit_price > pos.entry_price, (
        'a reversal after TP1 booked at or below entry — the banked rung was '
        'handed all the way back, which is the IMX +0.02% defect')


# ── the payoff arithmetic this exists to fix ─────────────────────────────────

def test_the_payoff_ratio_that_motivated_this():
    """Documents why, so the threshold is not tuned away without the numbers."""
    wins = [0.88, 0.48, 0.18]
    losses = [-1.14, -1.49]
    aw = sum(wins) / len(wins)
    al = sum(losses) / len(losses)
    wr = len(wins) / (len(wins) + len(losses))
    payoff = abs(aw / al)
    needed = abs(al) / (aw + abs(al))
    assert payoff < 0.5, payoff
    assert needed > wr, (
        f'break-even needs {needed:.0%} but the engine ran at {wr:.0%} — if this '
        f'ever inverts, the reversal-protect rule can be revisited'
    )
