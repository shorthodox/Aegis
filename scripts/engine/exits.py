"""Position management: the TP ladder, break-even shift, trailing stop and
reversal flip.

v85 lives here in spirit: the ladder banks against levels the plan
actually priced, so what is published as the target is what the trade is
managed to.

Extracted verbatim from the single-file live_engine.py; the bodies are
unchanged, only the class they hang off moved. LiveEngine composes this
mixin, so `self` is the full engine.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any
from typing import Dict
from typing import Optional
import time

from scripts.engine.models import Position
from scripts.engine.quality import SignalQualityFilter


class ExitsMixin:
    """_manage_exit .. _manage_exit — see module docstring."""

    def _manage_exit(self, symbol: str, pos: Position,
                     result: Dict[str, Any], price: float,
                     price_only: bool = False) -> None:
        """Evaluate every exit rule for one open position.

        price_only=True skips the model-reversal branch (section 8). The fast
        exit monitor passes it: reversal is a MODEL decision that belongs to the
        scan cycle, and re-evaluating it against a stale last_signals entry
        every few seconds would fire it on data the scan already acted on.
        Everything else — the TP ladder, the trail, break-even and the stop —
        is purely price-driven and is exactly what the monitor exists to catch
        between scans.
        """
        live_px     = self.live_prices.get(symbol, 0.0)
        check_price = live_px if live_px > 0 else price

        now  = time.time()
        held = now - self._open_time.get(symbol, 0)

        # Current ATR: use live result first, then fall back to the ATR stored at
        # entry (pos.atr), then a price-based estimate.
        atr = (float(result.get('atr') or 0)
               or pos.atr
               or pos.entry_price * 0.015)

        # ── Update peak price for trailing-stop tracking ──────────────────────
        # LONG: track the highest price; SHORT: track the lowest price (trough).
        if pos.direction == 'LONG':
            self._peak_price[symbol] = max(
                self._peak_price.get(symbol, pos.entry_price), check_price)
        else:
            self._peak_price[symbol] = min(
                self._peak_price.get(symbol, pos.entry_price), check_price)

        peak = self._peak_price[symbol]

        # The give-back ratchet's state, initialised defensively. LiveEngine is
        # routinely built with __new__ and hand-set attributes (the
        # characterisation harness and several exit tests do exactly that), so a
        # mixin that assumes __init__ ran breaks callers that never asked it to.
        if getattr(self, '_giveback_stop', None) is None:
            self._giveback_stop = {}

        # ── Helper: full close (removes position, cleans all TP-hit state) ────
        def _close(reason: str, exit_px: Optional[float] = None) -> None:
            rec = self.wallet.close_trade(
                symbol, exit_px if exit_px is not None else check_price, reason)
            if rec:
                self._last_close_time[symbol]   = now
                self._last_close_side[symbol]   = pos.side
                self._last_close_reason[symbol] = reason
                for d in (self._tp1_hit, self._tp2_hit,
                          self._tp3_hit, self._tp4_hit, self._peak_price,
                          self._giveback_stop):
                    d.pop(symbol, None)
                # Whole-trade view: rec.outcome already accounts for banked
                # TP partials (close_trade); aggregate PnL across all slices
                # of this signal_id so logs/alerts report the REAL result.
                _slices   = [t for t in self.wallet.trade_history
                             if t.signal_id == rec.signal_id]
                _tot_usdt = sum(t.pnl_usdt for t in _slices)
                _tot_val  = sum(t.position_value for t in _slices)
                if _tot_usdt < 0:                       # whole trade lost -> bench the token
                    self._last_loss_time[symbol] = now
                _tot_pct  = (_tot_usdt / _tot_val * 100) if _tot_val > 0 else rec.pnl_pct
                tag = rec.outcome
                _slice_note = (f' (final slice {rec.pnl_pct:+.2f}%)'
                               if len(_slices) > 1 else '')
                print(f'[{symbol}] {reason} {tag} {_tot_pct:+.2f}%{_slice_note} @ '
                      f'{(exit_px or check_price):.6g}')
                self.perf_tracker.record_outcome(
                    symbol        = symbol,
                    regime        = self.last_signals.get(symbol, {}).get('regime', 'UNKNOWN'),
                    outcome       = rec.outcome,
                    pnl_pct       = round(_tot_pct, 3),
                    quality_score = float(self.last_signals.get(symbol, {}).get('quality_score', 0)),
                )
                self.drift_monitor.record(symbol, rec.outcome)
                self.drift_monitor.save_state()
                drift_sev = self.drift_monitor.severity(symbol)
                if drift_sev in ('WARNING', 'CRITICAL'):
                    live_wr   = self.drift_monitor._live_win_rate(symbol)
                    benchmark = self.drift_monitor._benchmarks.get(symbol, 0.60)
                    print(f'[{symbol}] DRIFT {drift_sev}: '
                          f'live_wr={live_wr:.1%} benchmark={benchmark:.1%} '
                          f'(drop={((benchmark - (live_wr or 0)) * 100):.1f}pp)')
                self._save_track_record()
                try:
                    self.adaptive_orchestrator.record_trade(asdict(rec))
                except Exception:
                    pass
                try:
                    from scripts.notifications.dispatcher import get_notifier
                    _hold = int(time.time() - self._open_time.get(symbol, time.time()))
                    get_notifier().send_exit(
                        symbol=symbol, direction=pos.side, outcome=tag,
                        pnl_pct=round(_tot_pct, 3), hold_seconds=_hold,
                        exit_reason=reason,
                    )
                except Exception:
                    pass

        # ── Helper: partial close (keeps position open for next TP levels) ────
        def _partial(reason: str, pct: float, exit_px: Optional[float] = None) -> None:
            px  = exit_px if exit_px is not None else check_price
            rec = self.wallet.partial_close_trade(symbol, px, reason, pct)
            if rec:
                print(f'[{symbol}] {reason} {rec.pnl_pct:+.2f}% '
                      f'(closed {pct*100:.0f}% @ {px:.6g}) '
                      f'remaining≈{pos.position_value:.0f} USDT')
                self._save_track_record()
                # TP hits are outcome events — keep subscribers' Telegram in
                # sync with the position as profit is banked, not just at the
                # final close.
                try:
                    from scripts.notifications.dispatcher import get_notifier
                    _hold = int(time.time() - self._open_time.get(symbol, time.time()))
                    get_notifier().send_exit(
                        symbol=symbol, direction=pos.side, outcome=rec.outcome,
                        pnl_pct=rec.pnl_pct, hold_seconds=_hold,
                        exit_reason=f'{reason} ({pct*100:.0f}% closed, position still open)',
                    )
                except Exception:
                    pass

        # ── 1. Maximum hold time (zombie guard) ──────────────────────────────
        if held >= self.MAX_HOLD_SECONDS:
            _close('MAX_HOLD_EXPIRED')
            return

        # ── 2. TP5 hit — close all remaining size ────────────────────────────
        # By this point TPs 1-4 have already taken 80 %; this closes the last 20 %.
        if pos.take_profit_5 > 0:
            tp5_hit = (
                (pos.direction == 'LONG'  and check_price >= pos.take_profit_5) or
                (pos.direction == 'SHORT' and check_price <= pos.take_profit_5)
            )
            tp5_via_peak = self._tp4_hit.get(symbol, False) and (
                (pos.direction == 'LONG'  and peak >= pos.take_profit_5) or
                (pos.direction == 'SHORT' and peak <= pos.take_profit_5)
            )
            if tp5_hit:
                _close('TP5_HIT')
                return
            if tp5_via_peak:
                _close('TP5_HIT', exit_px=pos.take_profit_5)
                return

        # ── 3. TP4 hit — 20 % partial close ──────────────────────────────────
        if pos.take_profit_4 > 0 and not self._tp4_hit.get(symbol, False):
            tp4_hit = (
                (pos.direction == 'LONG'  and check_price >= pos.take_profit_4) or
                (pos.direction == 'SHORT' and check_price <= pos.take_profit_4)
            )
            # v78 fix: detect TP hits via peak from start (not just after previous TP)
            # This prevents missing TP levels when price spikes through them and reverses
            tp4_via_peak = (
                (pos.direction == 'LONG'  and peak >= pos.take_profit_4) or
                (pos.direction == 'SHORT' and peak <= pos.take_profit_4)
            )
            if tp4_hit or tp4_via_peak:
                exit_px = pos.take_profit_4 if tp4_via_peak else None
                _partial('TP4_PARTIAL', self.risk_engine.TP_CLOSE_PCTS[3], exit_px)
                self._tp4_hit[symbol] = True

        # ── 4. TP3 hit — 20 % partial close ──────────────────────────────────
        if pos.take_profit_3 > 0 and not self._tp3_hit.get(symbol, False):
            tp3_hit = (
                (pos.direction == 'LONG'  and check_price >= pos.take_profit_3) or
                (pos.direction == 'SHORT' and check_price <= pos.take_profit_3)
            )
            # v78 fix: detect TP hits via peak from start (not just after TP2)
            # Ensures TP3 closes even if price rapidly moves through it
            tp3_via_peak = (
                (pos.direction == 'LONG'  and peak >= pos.take_profit_3) or
                (pos.direction == 'SHORT' and peak <= pos.take_profit_3)
            )
            if tp3_hit or tp3_via_peak:
                exit_px = pos.take_profit_3 if tp3_via_peak else None
                _partial('TP3_PARTIAL', self.risk_engine.TP_CLOSE_PCTS[2], exit_px)
                self._tp3_hit[symbol] = True

        # ── 5. TP2 hit — 20 % partial close, activate trailing stop ──────────
        # Peak-based detection now applies from the start (v78 fix) to catch TP hits
        # even when price spikes and reverses between scan cycles.
        if pos.take_profit_2 > 0 and not self._tp2_hit.get(symbol, False):
            tp2_hit = (
                (pos.direction == 'LONG'  and check_price >= pos.take_profit_2) or
                (pos.direction == 'SHORT' and check_price <= pos.take_profit_2)
            )
            # v78: Always use peak-based detection, not just after TP1.
            # This catches rapid TP hits that would otherwise be missed.
            tp2_via_peak = (
                (pos.direction == 'LONG'  and peak >= pos.take_profit_2) or
                (pos.direction == 'SHORT' and peak <= pos.take_profit_2)
            )
            if tp2_hit or tp2_via_peak:
                exit_px = pos.take_profit_2 if tp2_via_peak else None
                _partial('TP2_PARTIAL', self.risk_engine.TP_CLOSE_PCTS[1], exit_px)
                self._tp2_hit[symbol] = True
                # Stop is already at break-even from TP1; the trail takes over
                # here and can only ratchet above it.
                print(f'[{symbol}] TP2_HIT — TRAILING on '
                      f'(dist=ATR×{self.risk_engine.TRAIL_MULTIPLIER}, '
                      f'floor=TP1 {pos.take_profit_1:.6g})')

        # ── 6. Trailing stop — active after TP2 is hit ───────────────────────
        # Trail distance = ATR × TRAIL_MULTIPLIER, floor = TP1.
        #
        # v82: the floor used to be TP2, which made this block a no-op — the
        # stop sat exactly at TP2 until price ran a full ATR beyond it, so any
        # dip right after TP2 exited there and no runner ever reached TP3.
        # Flooring at TP1 keeps ≥1R locked on the remainder while giving it the
        # room to actually get to the structural target.
        if self._tp2_hit.get(symbol, False):
            trail_dist = self.risk_engine.TRAIL_MULTIPLIER * atr
            if pos.direction == 'LONG':
                trail_stop = max(pos.take_profit_1, peak - trail_dist)
                if check_price <= trail_stop:
                    _close('TRAILING_STOP', exit_px=trail_stop)
                    return
            else:  # SHORT
                trail_stop = min(pos.take_profit_1, peak + trail_dist)
                if check_price >= trail_stop:
                    _close('TRAILING_STOP', exit_px=trail_stop)
                    return

        # ── 7. TP1 hit — partial close, move SL to break-even ────────────────
        # v78 fix: peak-based detection catches TP1 hits even when price spikes
        # through and reverses rapidly between scan cycles.
        #
        # Break-even at TP1 is a deliberate WIN-RATE choice, not an expectancy
        # one: it converts a would-be full loss into a scratch that still keeps
        # the TP1 slice, which is what holds the published win rate in a range
        # subscribers find credible.  It costs expectancy on trades that dip to
        # entry and then recover — an acceptable price while the model's live
        # edge is being re-measured after the regime-detector repair.
        #
        # This is NOT the deleted TP1_RECROSS.  Break-even exits at ENTRY if the
        # move fully reverses; the re-cross exited the whole position at TP1 on
        # any tick back through it, capping every winner.  Keep this, never that.
        if pos.take_profit_1 > 0 and not self._tp1_hit.get(symbol, False):
            tp1_hit = (
                (pos.direction == 'LONG'  and check_price >= pos.take_profit_1) or
                (pos.direction == 'SHORT' and check_price <= pos.take_profit_1)
            )
            # v78: Always use peak-based detection to catch rapid TP hits
            tp1_via_peak = (
                (pos.direction == 'LONG'  and peak >= pos.take_profit_1) or
                (pos.direction == 'SHORT' and peak <= pos.take_profit_1)
            )
            if tp1_hit or tp1_via_peak:
                _partial('TP1_PARTIAL', self.risk_engine.TP_CLOSE_PCTS[0])
                self._tp1_hit[symbol]    = True
                self._peak_price[symbol] = check_price   # reset peak tracking from TP1
                pos.stop_loss = pos.entry_price          # break-even
                print(f'[{symbol}] TP1_HIT @ {check_price:.6g} — banked '
                      f'{self.risk_engine.TP_CLOSE_PCTS[0]*100:.0f}%, '
                      f'SL to break-even ({pos.entry_price:.6g})')

        # ── 7a. Give-back ratchet — protect a rung that has been banked ──────
        # Once a rung is tagged, the remainder may hand back only part of THAT
        # RUNG'S SPAN before it is closed: entry→TP1 for the first, TP1→TP2 for
        # the second, and so on. The level ratchets — it only ever moves in the
        # trade's favour, so a later rung tightens it and nothing loosens it.
        #
        # This fills a real hole. Break-even goes on at TP1 and the trail only
        # starts at TP2, so a position that banked TP1 and reversed handed the
        # entire move back to entry with nothing in between. Observed on
        # BCH/USDT 2026-08-06: short from 214.70, TP1 212.09 tagged, price back
        # to 213.40 with the stop still sitting at break-even 214.70.
        #
        # It is NOT the deleted TP1_RECROSS, which closed on any tick back
        # through a tagged TP — a zero-width buffer — when TP1 was 0.7R against
        # a 1.0R stop, capping every winner at +0.7R. TP1 is 1.0R now, and the
        # buffer is configurable. See DynamicRiskEngine.TP_GIVEBACK_PCT for why
        # the width matters more than the mechanism.
        _rungs = (
            (self._tp4_hit.get(symbol, False), pos.take_profit_4, pos.take_profit_3),
            (self._tp3_hit.get(symbol, False), pos.take_profit_3, pos.take_profit_2),
            (self._tp2_hit.get(symbol, False), pos.take_profit_2, pos.take_profit_1),
            (self._tp1_hit.get(symbol, False), pos.take_profit_1, pos.entry_price),
        )
        _gb_pct = self.risk_engine.TP_GIVEBACK_PCT
        _gb_min = self.risk_engine.TP_GIVEBACK_MIN_ATR * atr
        for _hit, _tp, _prev in _rungs:
            if not _hit or _tp <= 0 or _prev <= 0:
                continue
            _span = abs(_tp - _prev)
            if _span <= 0:
                continue
            _leash = max(_span * _gb_pct, _gb_min)
            # A leash that reaches its own span would put the protective level
            # at or past the PREVIOUS rung — for the first rung that is the entry
            # itself, i.e. worse than the break-even stop already sitting there.
            # Skip and let break-even handle it; the ratchet is for the wider
            # rungs further up.
            if _leash >= _span:
                continue
            # back from the rung, toward the previous one
            _level = _tp - _leash if pos.direction == 'LONG' else _tp + _leash
            _cur = self._giveback_stop.get(symbol)
            if _cur is None:
                self._giveback_stop[symbol] = _level
            elif pos.direction == 'LONG':
                self._giveback_stop[symbol] = max(_cur, _level)
            else:
                self._giveback_stop[symbol] = min(_cur, _level)
            break                      # highest rung tagged wins; it is the tightest

        _gb = self._giveback_stop.get(symbol)
        if _gb is not None:
            _breached = ((pos.direction == 'LONG'  and check_price <= _gb) or
                         (pos.direction == 'SHORT' and check_price >= _gb))
            if _breached:
                print(f'[{symbol}] TP_GIVEBACK — handed back '
                      f'{_gb_pct*100:.0f}% of the last rung (level {_gb:.6g}), '
                      f'closing the remainder')
                _close('TP_GIVEBACK', exit_px=_gb)
                return

        # ── 7b–7e. TP re-cross exits — DELETED in v82 ────────────────────────
        # There used to be four blocks here (TP1/TP2/TP3/TP4_RECROSS) that
        # closed 100 % of the remaining position the instant price ticked back
        # through a TP level it had already tagged.  TP1_RECROSS was the single
        # most expensive line in this engine:
        #
        #   TP1 was 0.7R, so reaching TP1 and then ticking back — which is
        #   almost every trade on a 1h chart — closed the whole position at
        #   +0.7R.  The stop was a full -1.0R.  Break-even win rate was
        #   1/1.7 = 58.8 % and the engine ran at 60 %, i.e. an edge of +0.02R,
        #   which fees then buried.  It also made TP2-TP5 unreachable: the live
        #   log shows only ever two exit reasons, SL_HIT and TP1.
        #
        # Profit protection after TP2 is the trailing stop's job (section 6),
        # which ratchets instead of capping.  Between TP1 and TP2 the original
        # structural stop stands — a trade is allowed to breathe there.
        #
        # Do not reintroduce these without re-running the payoff arithmetic.

        # ── 8. Model-reversal exit (dynamic exit on opposing signal) ─────────
        side = result.get('side', 'FLAT')
        fire = bool(result.get('fire', False)) and not price_only
        opposite = (
            (pos.direction == 'LONG'  and side == 'SELL' and fire) or
            (pos.direction == 'SHORT' and side == 'BUY'  and fire)
        )
        if opposite:
            # Require minimum hold unless TP1 is already secured.
            _reversal_min = 0 if self._tp1_hit.get(symbol, False) else self.MIN_HOLD_SECONDS
            if held >= _reversal_min:
                # Guard against low-conviction one-candle noise without depending on
                # consecutive-cycle counting (which breaks when the model outputs FLAT
                # between reversal cycles, clearing _signal_history each time).
                # Instead, require the reversal signal to meet the same quality floor
                # used for entry (edge_score >= MIN_QUALITY_SCORE = 70).
                #
                # v82: a reversal after TP1 used to book at `pos.take_profit_1`
                # regardless of where price actually was — a free fill at a level
                # the market had already left.  It now exits at the live price
                # like any other market exit; the TP1 slice is already banked.
                _tp1_secured = self._tp1_hit.get(symbol, False)
                _rev_edge    = float(result.get('edge_score', 0.0))
                if _tp1_secured:
                    _close('MODEL_REVERSAL_TP')
                    return

                # ── a winner below TP1 is not closed here ────────────────────
                # This branch used to close on any opposing signal clearing the
                # SAME edge floor required to ENTER (60), at any PnL. Measured
                # over the closed book that is what produced the payoff ratio:
                #
                #     winners realised  +0.19R .. +0.40R   (6-28 % of target)
                #     losers  realised  -1.08R .. -1.11R   (the full stop)
                #
                #     payoff 0.39 : 1  ->  needs a 72 % win rate to break even
                #     actual win rate 60 %  ->  -0.32 % per trade after costs
                #
                # The asymmetry is structural, not bad luck. The stop sits ~1R
                # away and gets hit in a bar or two, while a reversal needs a
                # full re-score — so losers reach the stop and winners get
                # reversed out early. The engine was cutting winners and letting
                # losers run, which is the one arrangement that cannot be fixed
                # by a better model.
                #
                # So the reversal keeps its real job — abandoning a dead thesis
                # BEFORE the stop — and loses the job it was doing badly. A
                # position already in profit has not yet earned its risk if it
                # is short of TP1 (1R); closing it there books a fraction of R
                # against full-R losses. It is protected instead: the stop comes
                # to break-even, so the bad case is a scratch rather than a
                # small win, and the good case is still open.
                _pnl_pct = ((check_price - pos.entry_price) / pos.entry_price * 100.0
                            if pos.direction == 'LONG'
                            else (pos.entry_price - check_price) / pos.entry_price * 100.0)
                _cost = self.wallet.round_trip_cost_pct()
                if _pnl_pct > _cost:
                    _moved = False
                    if pos.direction == 'LONG' and pos.stop_loss < pos.entry_price:
                        pos.stop_loss, _moved = pos.entry_price, True
                    elif pos.direction == 'SHORT' and pos.stop_loss > pos.entry_price:
                        pos.stop_loss, _moved = pos.entry_price, True
                    print(f'[{symbol}] REVERSAL_PROTECT {pos.direction}→{side}: '
                          f'+{_pnl_pct:.2f}% but TP1 not reached — '
                          f'{"stop to break-even" if _moved else "stop already at/above break-even"}, '
                          f'holding for the first target')
                    return

                if _rev_edge >= SignalQualityFilter.MIN_QUALITY_SCORE:
                    _close('MODEL_REVERSAL_TP')
                    return
                print(f'[{symbol}] REVERSAL_GATE deferred {pos.direction}→{side}: '
                      f'reversal edge={_rev_edge:.1f} < '
                      f'{SignalQualityFilter.MIN_QUALITY_SCORE:.0f} (tp1_secured=False)')

        # ── 9. Stop loss / break-even SL ─────────────────────────────────────
        # Before TP2: uses the original ATR-based structural SL.
        # After  TP2: pos.stop_loss was moved to entry_price (break-even), and
        #             the trailing stop in section 6 is usually tighter still.
        if pos.stop_loss > 0:
            sl_hit = (
                (pos.direction == 'LONG'  and check_price <= pos.stop_loss) or
                (pos.direction == 'SHORT' and check_price >= pos.stop_loss)
            )
            if sl_hit:
                _close('STOP_HIT')
