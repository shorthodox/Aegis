"""Opening positions and persisting what happened.

_build_signal_entry is the single place the published signal dict is
shaped — main.py and the chart both read its output, so a field added
here is a field the UI can rely on.

Extracted verbatim from the single-file live_engine.py; the bodies are
unchanged, only the class they hang off moved. LiveEngine composes this
mixin, so `self` is the full engine.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from datetime import timezone
from typing import Any
from typing import Dict
from typing import Optional
import asyncio
import json
import time
import uuid

from scripts.engine.config import ALPHA_TIMEFRAMES as _ALPHA_TIMEFRAMES
from scripts.engine.config import ALPHA_TRACK_RECORD_PATH
from scripts.engine.config import ROOT as _ROOT
from scripts.engine.config import TRACK_RECORD_PATH
from scripts.engine.models import Position
from scripts.engine.models import RegimeState
from scripts.engine.risk import DynamicRiskEngine
from scripts.engine.state import _fs_save_track_record
from src.trading.trader_gate import TradePlan


def _edge_over_geometry(records):
    """Never let the study break the write that persists the book."""
    try:
        from scripts.engine.edge_metric import by_group, measure
        out = measure(records)
        out['by_symbol'] = by_group(records, 'symbol')
        return out
    except Exception as e:                       # pragma: no cover - defensive
        return {'n': 0, 'error': repr(e)}


class _NoShadow:
    """Stand-in for an engine built without a ShadowBook (tests, tools)."""
    def open(self, *a, **k):
        return None

    def record_live(self, *a, **k):
        return None

    def tick(self, *a, **k):
        return None


_NO_SHADOW = _NoShadow()


class PositionsMixin:
    """_open_position .. _save_track_record — see module docstring."""

    def _open_position(
        self,
        symbol:        str,
        result:        Dict[str, Any],
        price:         float,
        regime:        Optional[RegimeState] = None,
        quality_score: float                 = 0.0,
        risk_tier:     str                   = '',
        entry_mode:    str                   = '',
        gate_warnings: Optional[list]        = None,
        plan:          Optional[TradePlan]   = None,   # v83 TraderGate
    ) -> None:
        side = result.get('side', 'FLAT')
        if side not in ('BUY', 'SELL'):
            return

        direction = 'LONG' if side == 'BUY' else 'SHORT'
        meta_conf = float(result.get('edge_score', result.get('meta_confidence', 0)))
        atr_mult  = float(result.get('atr_multiplier', 1.5))
        atr       = float(result.get('atr', price * 0.015))
        atr_pct   = float(result.get('atr_pct', atr / price * 100 if price > 0 else 1.5))

        # Apply HMM ATR multiplier: VOLATILE_EXPANSION widens stops (1.5×),
        # TRENDING tightens them (0.9×), etc.
        _hmm_atr  = float(result.get('hmm_atr_mult', 1.0))
        _hmm_pscl = float(result.get('hmm_position_scale', 1.0))
        atr_mult  = round(atr_mult * _hmm_atr, 3)

        # ── Dynamic position sizing (replaces fixed wallet.position_size()) ───
        # The `quality_score > 0` condition used to sit in this test, and it
        # inverted the sizing curve at exactly the wrong end. A zero-conviction
        # signal did not take the floor — it fell through to
        # wallet.position_size(), which is min(balance * 10 %, max_position),
        # i.e. the MAXIMUM default allocation:
        #
        #     quality  5 -> 200 USDT      (dynamic, quality-scaled)
        #     quality 45 -> 315 USDT
        #     quality 65 -> 455 USDT
        #     quality  0 -> 1000 USDT     <- the largest position of the set
        #
        # so the worst-scoring setups were sized five times the merely-weak
        # ones. calculate_position_size() already clamps to
        # [MIN_POSITION_PCT, MAX_POSITION_PCT] and returns the floor for a
        # quality of 0, which is the behaviour that was wanted; it just was not
        # being reached. Only a missing regime justifies the flat fallback now.
        if regime is not None:
            pos_value = self.risk_engine.calculate_position_size(
                balance       = self.wallet.balance,
                quality_score = quality_score,
                regime        = regime,
                atr_pct       = atr_pct,
            )
            # HMM position scale: reduces size in choppy/volatile/distribution regimes
            pos_value = round(pos_value * _hmm_pscl, 2)
            # Cap at wallet max_position_usdt
            pos_value = min(pos_value, self.wallet.max_position_usdt)

            # Per-symbol exposure reduction if recent loss streak
            if self.perf_tracker.should_reduce_exposure(symbol):
                pos_value *= 0.5
                print(f'[{symbol}] REDUCE_EXPOSURE — recent loss streak, '
                      f'halved position to {pos_value:.0f} USDT')
        else:
            pos_value = self.wallet.position_size()

        pos_value = max(pos_value, 1.0)   # safety floor

        # ── v74 tide dial: never fight the market's direction at full size ───
        # BTC leads the alt tape intraday; a position against the 4h tide is
        # statistically half the trade it looks like, so it gets half the size.
        _tide = str(result.get('btc_tide', 'FLAT') or 'FLAT')
        if plan is None and ((side == 'SELL' and _tide == 'UP')
                             or (side == 'BUY' and _tide == 'DOWN')):
            # v83: when a plan is present its allocation stage has ALREADY priced
            # the tide (and correlation, and setup class) into size_factor below.
            # Applying this legacy halving on top would charge for the tide twice.
            pos_value = round(pos_value * 0.5, 2)
            print(f'[{symbol}] TIDE_HALF {side}: BTC 4h tide is {_tide} — half size')

        # v83: the desk's allocation stage is the single sizing authority when a
        # plan exists — setup class, tide and correlation are already folded in.
        if plan is not None and plan.size_factor > 0:
            pos_value = max(1.0, round(pos_value * plan.size_factor, 2))
            print(f'[{symbol}] PLAN SIZE {side}: {plan.setup} x{plan.size_factor:.2f} '
                  f'-> {pos_value:.0f} USDT')

        # ── ATR + Structure hybrid stop/TP calculation ───────────────────────
        # SL anchored to the gate's invalidation level (support for a LONG,
        # resistance for a SHORT); TP ladder blends RR-multiples with the
        # structural target and fib extensions.
        # RISKY tier gets the TIGHTEST cap — a failed low-conviction trade must be
        # a small loss, not the 2.2-2.5x bleed the old loose-entry logic gave it
        # (user: "risky setups' stop losses are high, tighten them"). STRONG/NORMAL
        # keep the structural cap; a loose-but-not-RISKY entry still gets the wider
        # stop so a genuine at-level setup isn't noise-stopped.
        _em_l = (entry_mode or '').lower()
        if (risk_tier or '').upper() == 'RISKY':
            _sl_cap = self.risk_engine.RISKY_SL_CAP_ATR
        elif 'mid_range' in _em_l:
            _sl_cap = 2.5
        elif any(t in _em_l for t in ('early entry', 'unconfirmed', 'wide stop', 'ambiguous', 'pullback', 'pre_breakout')):
            _sl_cap = 2.2
        else:
            _sl_cap = self.risk_engine.ATR_SL_MULTIPLIER
        # v74: anchor the stop to the TESTED LEVEL this entry waited for.
        # Guard M (hard since v73) only fires at/tag-rejecting a structural
        # level — that level IS the thesis invalidation. When it sits closer
        # than the rolling S/R, use it: the stop hugs where the fade is
        # actually wrong instead of paying for 2+ ATR of room the thesis
        # never asked for. calculate_stops adds the wick buffer and clamps
        # to [SL_FLOOR_ATR, cap], so this can only tighten, never degenerate.
        _sl_support    = float(result.get('support', 0) or 0)
        _sl_resistance = float(result.get('resistance', 0) or 0)
        _pend_lvl = float((result.get('at_pending_level') or {}).get('level', 0) or 0)
        if _pend_lvl > 0:
            if side == 'BUY' and _pend_lvl < price:
                _sl_support = max(_sl_support, _pend_lvl)
            elif side == 'SELL' and _pend_lvl > price:
                _sl_resistance = (min(_sl_resistance, _pend_lvl)
                                  if _sl_resistance > 0 else _pend_lvl)

        stops = self.risk_engine.calculate_stops(
            price=price, side=side, atr=atr,
            support    = _sl_support,
            resistance = _sl_resistance,
            sl_cap_atr = _sl_cap,
            sl_override = (plan.stop if plan is not None else 0.0),
            tp_override = (plan.target if plan is not None else 0.0),
            gate_stop_source = (plan.stop_source if plan is not None else ''),
        )

        stop_loss = stops['sl']
        tp1       = stops['tp1']
        tp2       = stops['tp2']
        tp3       = stops['tp3']
        tp4       = stops['tp4']
        tp5       = stops['tp5']
        rr        = stops['risk_reward']

        # ── Risk-based position sizing ───────────────────────────────────────
        # The hybrid SL varies per trade, so scale the (quality/regime-sized)
        # notional inversely with the actual risk leg to keep $-at-risk roughly
        # constant: a tighter structural stop → larger size at the same dollar
        # risk (this is how the improved RR turns into higher return). Bounded to
        # [0.5×, 2×] and re-capped to the wallet max so it can never blow up.
        _ref_risk    = self.risk_engine.ATR_SL_MULTIPLIER * atr
        _actual_risk = float(stops.get('risk', 0) or _ref_risk) or _ref_risk
        _risk_scale  = max(0.5, min(_ref_risk / _actual_risk, 2.0))
        pos_value    = round(pos_value * _risk_scale, 2)
        pos_value    = min(pos_value, self.wallet.max_position_usdt)
        pos_value    = max(pos_value, 1.0)

        # ── Affordability gate (item 3, 2026-08-15) ──────────────────────────
        # If clearing the level costs more than the risk budget allows, the trade
        # is not affordable — refuse it, rather than taking it with a stop we
        # already know is in the wrong place.
        #
        # This is the defect behind 15 losses out of 15 exiting at STOP_HIT. The
        # stop was placed by the budget band, not by structure, so it sat BETWEEN
        # entry and the level the thesis leaned on: TAO/USDT stopped at 1.30%
        # with support 2.57% away, price bottomed at 194.80 (never reaching
        # support) and reversed. The trade was right about direction and was
        # taken out by the move that tested its own level.
        #
        # Deliberately NOT fixed by widening the stop past the band. Sizing only
        # partly compensates (the 0.5x floor binds), and the two available
        # harness measurements disagree on whether a wider stop pays per unit of
        # risk. Refusing is the option that does not bet on that disagreement:
        # it takes fewer trades rather than taking the same trades with more
        # risk. band_capped/support_seen come from the shadow instrumentation,
        # so this reads the same signal the analysis was built on.
        _would_refuse = bool(stops.get('band_capped') and stops.get('support_seen'))
        if _would_refuse:
            print(f'[{symbol}] BUDGET_SHADOW {side} — clearing the level needs '
                  f'{stops.get("structural_stop_pct", 0):.2f}% vs a {self.risk_engine.budget_cap_pct():.2f}% '
                  f'budget; TAKING THE TRADE (item 3 held off, see risk.py)')
        if (self.risk_engine.REFUSE_UNAFFORDABLE_INVALIDATION and _would_refuse):
            print(f'[{symbol}] BUDGET_REJECTED {side}')
            if symbol in self.last_signals:
                self.last_signals[symbol]['fire']            = False
                self.last_signals[symbol]['signal']          = 'HOLD'
                self.last_signals[symbol]['budget_blocked']  = True
            return

        # ── Risk/Reward gate ─────────────────────────────────────────────────
        # Reward is measured to TP3 (first full-trend target).
        # Trades below the minimum RR are rejected to protect track record quality.
        if not stops['valid_rr']:
            print(f'[{symbol}] RR_REJECTED {side} — '
                  f'RR={rr:.2f} < min={self.risk_engine.MIN_RISK_REWARD} '
                  f'(ATR={atr:.4g})')
            if symbol in self.last_signals:
                self.last_signals[symbol]['fire']       = False
                self.last_signals[symbol]['signal']     = 'HOLD'
                self.last_signals[symbol]['rr_blocked'] = True
            return

        # SHADOW — recorded, never applied. See RiskEngine.tp1_hybrid.
        _tp1h_pct, _tp1h_atr = self.risk_engine.tp1_hybrid(price, atr)

        pos = Position(
            symbol          = symbol,
            direction       = direction,
            side            = side,
            entry_price     = round(price, 8),
            position_value  = round(pos_value, 2),
            initial_value   = round(pos_value, 2),   # v82: fixed base for TP partial sizing
            stop_loss       = round(stop_loss, 8),
            # Same value as stop_loss at open, and that is the point: from here
            # the ratchet moves stop_loss and this stays put, so R stays honest.
            entry_stop      = round(stop_loss, 8),
            signal_id       = str(uuid.uuid4()),
            entry_time      = datetime.now(timezone.utc).isoformat(),
            meta_confidence = round(meta_conf, 4),
            atr_multiplier  = self.risk_engine.ATR_SL_MULTIPLIER,
            atr             = round(atr, 8),
            take_profit_1   = round(tp1, 8),
            take_profit_2   = round(tp2, 8),
            take_profit_3   = round(tp3, 8),
            take_profit_4   = round(tp4, 8),
            take_profit_5   = round(tp5, 8),
            signal_strength = risk_tier,
            entry_mode      = entry_mode,
            quality_score   = round(quality_score, 1),
            gate_warnings   = list(gate_warnings or []),
            entry_support   = float(result.get('support', 0) or 0),
            entry_resistance= float(result.get('resistance', 0) or 0),
            structural_stop     = float(stops.get('structural_stop', 0) or 0),
            structural_stop_pct = float(stops.get('structural_stop_pct', 0) or 0),
            band_capped         = bool(stops.get('band_capped', False)),
            tp1_hybrid_pct      = _tp1h_pct,
            tp1_hybrid_atr      = _tp1h_atr,
            stop_source         = str(stops.get('stop_source', 'unknown')),
            pre_band_stop       = float(getattr(plan, 'pre_band_stop', 0.0) or 0.0),
            support_present     = bool(stops.get('support_seen', False)),
            would_refuse_unaffordable = _would_refuse,
        )
        self.wallet.open_trade(pos)
        # Shadow accounting — observation only, and deliberately wrapped: a bug
        # in the study must never be able to stop a real position from opening.
        try:
            self.shadow_book.open(
                trade_id=pos.signal_id, symbol=symbol, direction=pos.direction,
                entry=pos.entry_price, stop_loss=pos.stop_loss,
                take_profits=[pos.take_profit_1, pos.take_profit_2, pos.take_profit_3,
                              pos.take_profit_4, pos.take_profit_5])
        except Exception as _e:
            print(f'[ShadowBook] open hook failed for {symbol}: {_e!r}')
        self._open_time[symbol]    = time.time()
        self._tp1_hit[symbol]      = False
        self._tp2_hit[symbol]      = False
        self._tp3_hit[symbol]      = False
        self._tp4_hit[symbol]      = False
        self._peak_price[symbol]   = price

        regime_label = regime.regime if regime else 'UNKNOWN'
        print(
            f'[{symbol}] OPEN {direction} @ {price:.6g} | '
            f'conf={meta_conf:.3f} quality={quality_score:.0f} regime={regime_label} '
            f'mode={entry_mode or "n/a"}\n'
            f'         ATR={atr:.4g}  SL={stop_loss:.6g}  RR={rr:.2f}\n'
            f'         TP1={tp1:.6g}  TP2={tp2:.6g}  TP3={tp3:.6g}  '
            f'TP4={tp4:.6g}  TP5={tp5:.6g}  size={pos_value:.0f} USDT'
        )
        self._save_track_record()
        try:
            from scripts.notifications.dispatcher import get_notifier
            get_notifier().send_entry({
                'symbol':           symbol,
                'direction':        side,
                'confidence':       meta_conf / 100.0,  # edge_score is 0-100; formatters expect 0-1
                'confluence_score': 0.0,
                'current_price':    price,
                'mode':             'live',
                'timeframe':        '1h',
                'top_strategies':   [],
                'atr':              atr,
                'risk_reward':      rr,
                'stop_loss':        stop_loss,
                'take_profit_1':    tp1,
                'take_profit_2':    tp2,
                'take_profit_3':    tp3,
                'take_profit_4':    tp4,
                'take_profit_5':    tp5,
                'guidance':         {},
                'timestamp':        pos.entry_time,
            })
        except Exception:
            pass

    @staticmethod
    def _build_signal_entry(
        symbol:        str,
        result:        Dict[str, Any],
        price:         float,
        regime:        Optional[RegimeState] = None,
        quality_score: float                 = 0.0,
        fake_breakout: bool                  = False,
        open_pos:      Optional[Position]    = None,   # v82: publish frozen levels
    ) -> Dict[str, Any]:
        side     = result.get('side', 'FLAT')
        conf     = float(result.get('edge_score', result.get('meta_confidence', 0)))
        thr      = float(result.get('meta_threshold', 65.0))
        fire     = bool(result.get('fire', False))
        atr      = float(result.get('atr', price * 0.015))
        atr_mult = float(result.get('atr_multiplier', 1.5))
        atr_pct  = float(result.get('atr_pct', atr / price * 100 if price > 0 else 1.5))

        if not fire:
            strength = 'NEUTRAL'
        elif conf >= thr * 1.15 and quality_score >= 70.0:
            strength = f'STRONG_{side}'
        else:
            strength = side

        entry: Dict[str, Any] = {
            'symbol':          symbol,
            'signal':          side,
            'signal_strength': strength,
            'fire':            fire,
            'direction':       'LONG' if side == 'BUY' else ('SHORT' if side == 'SELL' else 'NEUTRAL'),
            'price':           price,
            'entry_price':     price,
            'atr':             round(atr, 8),
            'atr_multiplier':  atr_mult,
            'meta_confidence': round(conf, 4),
            'threshold':       round(thr, 4),
            'tradeable':       result.get('tradeable', True),
            # State flags ALWAYS present so the Firestore merge=True push can
            # never ghost a stale True from a previous scan (measured: tokens
            # kept paper_only/pending_entry forever once set, because a fresh
            # entry simply lacked the key and merge kept the old value).
            'pending_entry':   False,
            'paper_only':      False,
            'evaluating':      False,
            'p_buy':           round(float(result.get('p_buy',  0)), 4),
            'p_sell':          round(float(result.get('p_sell', 0)), 4),
            'p_hold':          round(float(result.get('p_hold', 0)), 4),
            'signal_id':       str(uuid.uuid4()) if fire else f'{symbol.replace("/","_")}_{side}',
            'data_timestamp':  datetime.now(timezone.utc).isoformat(),
            'timestamp':       datetime.now(timezone.utc).isoformat(),
            'timeframe':       '1h',
            # Adaptive intelligence fields
            'regime':              regime.regime        if regime else 'UNKNOWN',
            'regime_confidence':   regime.confidence    if regime else 0.0,
            'quality_score':       round(quality_score, 1),
            'is_fake_breakout':    fake_breakout,
            # Tier inputs the chart explains to the subscriber. This dict is a
            # whitelist, not a copy of `result` — a field omitted here simply
            # never reaches the panel, however faithfully the engine computed it.
            'flag_available':         bool(result.get('flag_available', False)),
            'flag_bias':              float(result.get('flag_bias') or 0.0),
            'flag_pattern':           str(result.get('flag_pattern') or ''),
            'flag_breakout_dist_atr': float(result.get('flag_breakout_dist_atr') or 0.0),
            # -1.0 is the predictor's "pattern library unavailable" sentinel and
            # must survive as-is; 0.0 would read as "scored, and found nothing".
            'cdl_bull_reversal':   float(result.get('cdl_bull_reversal', -1.0)
                                         if result.get('cdl_bull_reversal') is not None else -1.0),
            'cdl_bear_reversal':   float(result.get('cdl_bear_reversal', -1.0)
                                         if result.get('cdl_bear_reversal') is not None else -1.0),
            'risk_score':          round(max(0.0, 100.0 - quality_score), 1),
            'volatility_score':    round(min(atr_pct / 5.0 * 100.0, 100.0), 1),
        }

        # ── ATR-based TP/SL levels for the active direction ──────────────────
        # Use DynamicRiskEngine class constants directly (static method — no self).
        # calculate_stops() is called as an unbound helper via a throw-away instance;
        # it is a pure function so this is safe and cheap.
        _re = DynamicRiskEngine()
        _stops: Dict[str, float] = (
            _re.calculate_stops(
                price=price, side=side, atr=atr,
                support    = float(result.get('support', 0) or 0),
                resistance = float(result.get('resistance', 0) or 0),
            )
            if side in ('BUY', 'SELL') and atr > 0 and price > 0
            else {}
        )

        # v82: when a position is actually OPEN on this symbol, publish ITS
        # frozen levels rather than a fresh re-computation off the live price.
        # The recomputed set drifted with price and skipped the RISKY sl_cap
        # override, so the dashboard showed a materially different trade from
        # the one being tracked (observed: position bar entry 6.19 / TP1 6.2662
        # against a signal panel reading entry 6.12 / TP1 6.1733 on one symbol).
        if open_pos is not None and open_pos.entry_price > 0:
            entry['entry_price']    = open_pos.entry_price
            entry['position_entry'] = open_pos.entry_price
            entry['levels_frozen']  = True
            _stops = {
                'sl':  open_pos.stop_loss,
                'tp1': open_pos.take_profit_1, 'tp2': open_pos.take_profit_2,
                'tp3': open_pos.take_profit_3, 'tp4': open_pos.take_profit_4,
                'tp5': open_pos.take_profit_5,
            }
            side = open_pos.side          # levels belong to the live direction
            price = open_pos.entry_price  # RR below must measure from the real entry
        else:
            entry['levels_frozen'] = False

        if side == 'BUY':
            entry['suggested_sl'] = _stops.get('sl') if _stops else None
            entry['suggested_tp'] = _stops.get('tp1') if _stops else None
            entry['tp2']          = _stops.get('tp2')
            entry['tp3']          = _stops.get('tp3')
            entry['tp4']          = _stops.get('tp4')
            entry['tp5']          = _stops.get('tp5')
        elif side == 'SELL':
            entry['suggested_sl'] = _stops.get('sl') if _stops else None
            entry['suggested_tp'] = _stops.get('tp1') if _stops else None
            entry['tp2']          = _stops.get('tp2')
            entry['tp3']          = _stops.get('tp3')
            entry['tp4']          = _stops.get('tp4')
            entry['tp5']          = _stops.get('tp5')
        else:
            entry['suggested_tp'] = None
            entry['suggested_sl'] = None
            entry['tp2'] = entry['tp3'] = entry['tp4'] = entry['tp5'] = None

        # Expected MOVE projection — a magnitude, not an expectancy.
        #
        # It is |confluence - neutral| scaled by ATR: "how far this tape tends to
        # travel when confluence leans this hard", in percent of price. Three
        # things it is NOT, all of which it has been read as:
        #   * not an expected value — there is no probability and no cost term,
        #     so it must never be compared against the round trip;
        #   * not directional — the abs() means it is always >= 0, for a short
        #     as much as a long. Anything colouring it by sign is dead code;
        #   * not a gate — nothing reads it. The number that decides whether a
        #     trade pays is TradePlan.r_net, computed in trader_gate's payoff
        #     stage against MIN_NET_R with the round trip priced in.
        # Published as `expected_move_pct` and labelled "Exp. Move" on the chart
        # for exactly that reason.
        _conf_data  = result.get('confluence') or {}
        _conf_total = float(_conf_data.get('total', 5.0))
        _conf_raw   = abs(_conf_total - 5.0) / 5.0
        entry['expected_move_pct'] = round(_conf_raw * atr_pct * 3.0, 2)

        # Risk/Reward ratio.
        #
        # v82: the headline number was measured to TP5 and advertised figures
        # like "1 : 15.17" — against a TP5 the UI's own probability panel prices
        # at 6 % and which has never once filled in recorded history.  It also
        # did NOT match the MIN_RISK_REWARD gate despite the old comment saying
        # so (calculate_stops measures reward to the structural target).
        # `risk_reward` is now quoted to TP2, the first objective the position
        # is actually managed toward; the stretch figure is kept alongside it
        # and clearly named.
        sl_val  = entry.get('suggested_sl') or 0
        tp2_val = entry.get('tp2') or 0
        tp5_val = entry.get('tp5') or 0
        _risk   = abs(price - sl_val) if (price > 0 and sl_val) else 0
        entry['risk_reward'] = (
            round(abs(price - tp2_val) / _risk, 2) if (_risk > 0 and tp2_val) else 0)
        entry['risk_reward_tp5'] = (
            round(abs(price - tp5_val) / _risk, 2) if (_risk > 0 and tp5_val) else 0)
        entry['atr_sl_multiplier'] = _re.ATR_SL_MULTIPLIER
        entry['min_risk_reward']   = _re.MIN_RISK_REWARD

        # Forward all market context fields from predictor
        _CONTEXT_KEYS = (
            'market_bias', 'bias_strength', 'trend_regime', 'volatility_regime',
            'atr_pct', 'support', 'resistance', 'pivot',
            'r1', 'r2', 's1', 's2', 'range_position',
            'resistance_broken_recent', 'support_broken_recent',
            'broken_resistance_level', 'broken_support_level',
            'cdl_bull_reversal', 'cdl_bear_reversal', 'cdl_patterns_active',
            'bull_tp1', 'bull_tp2', 'bull_tp3',
            'bear_tp1', 'bear_tp2', 'bear_tp3',
            'confluence',
            'rsi', 'rsi_slope', 'rsi_acceleration', 'macd_signal', 'cci', 'adx', 'supertrend',
            'macro_daily', 'macro_weekly',
            'volume_strength', 'volume_zscore',
            'funding_rate', 'funding_bias', 'oi_trend', 'oi_change_1h_pct', 'oi_zscore',
            'session', 'session_note', 'fear_greed',
            # primary model outputs
            'edge_score', 'edge_rank', 'signal_strength_score',
            'p_buy', 'p_sell', 'p_hold',
            # HMM regime intelligence fields
            'hmm_regime', 'hmm_confidence', 'hmm_state_id',
            'hmm_transition_risk', 'hmm_stability', 'hmm_available',
            'hmm_conf_adjustment', 'hmm_atr_mult', 'hmm_position_scale',
            'hmm_transition_warning',
            # LSTM temporal intelligence fields
            'lstm_continuation_prob', 'lstm_vol_expansion_prob',
            'lstm_exhaustion_prob', 'lstm_available',
            # UWGS — weighted per-direction gate score
            'signal_scores', 'gate_breakdown', 'vetoes', 'sr_quality',
        )
        for k in _CONTEXT_KEYS:
            if k in result:
                entry[k] = result[k]

        entry['target_support']    = float(result.get('target_support') or result.get('support') or result.get('s1') or 0)
        entry['target_resistance'] = float(result.get('target_resistance') or result.get('resistance') or result.get('r1') or 0)

        return entry

    def _alpha_open_position(self, key: str, symbol: str,
                              result: Dict[str, Any], price: float, tf: str) -> None:
        side = result.get('side', 'FLAT')
        if side not in ('BUY', 'SELL'):
            return
        atr      = float(result.get('atr', price * 0.015) or price * 0.015)
        atr_mult = float(result.get('atr_multiplier', 1.5))
        step     = atr * atr_mult
        if side == 'BUY':
            stop_loss = round(price - step, 8)
            tp1       = round(price + step, 8)
            tp2       = round(price + step * 2, 8)
            tp3       = round(price + step * 3.5, 8)
        else:
            stop_loss = round(price + step, 8)
            tp1       = round(price - step, 8)
            tp2       = round(price - step * 2, 8)
            tp3       = round(price - step * 3.5, 8)
        pos = Position(
            symbol          = key,
            direction       = 'LONG' if side == 'BUY' else 'SHORT',
            side            = side,
            entry_price     = price,
            position_value  = self.alpha_wallet.position_size(),
            stop_loss       = stop_loss,
            entry_stop      = stop_loss,   # immutable R denominator — see models.py
            entry_support   = float(result.get('support', 0) or 0),
            entry_resistance= float(result.get('resistance', 0) or 0),
            signal_id       = str(uuid.uuid4()),
            entry_time      = datetime.now(timezone.utc).isoformat(),
            meta_confidence = float(result.get('edge_score', result.get('meta_confidence', 0))),
            atr_multiplier  = atr_mult,
            take_profit_1   = tp1,
            take_profit_2   = tp2,
            take_profit_3   = tp3,
        )
        self.alpha_wallet.open_trade(pos)
        self._alpha_open_time[key] = time.time()
        print(f'[Alpha] OPEN {side} {symbol} {tf} @ {price:.4g} SL={stop_loss:.4g}')

    async def _process_alpha_timeframe(
        self, symbol: str, tf: str, sem: asyncio.Semaphore
    ) -> None:
        async with sem:
            loop = asyncio.get_event_loop()
            try:
                pred   = self.predictors[symbol]
                result: Dict[str, Any] = await asyncio.wait_for(
                    loop.run_in_executor(
                        self._executor,
                        lambda p=pred, t=tf: p.predict_realtime(
                            risk_tier=self.risk_tier, timeframe=t),
                    ),
                    timeout=120,
                )
            except Exception:
                return
            if not isinstance(result, dict):
                return

            price = float(self.live_prices.get(symbol, 0) or result.get('price', 0) or 0)
            if price <= 0:
                return

            regime = self.regime_detector.detect(result)
            key    = f'{symbol}|{tf}'
            sig = self._build_signal_entry(
                symbol, result, price, regime=regime,
                quality_score=min(float(result.get('edge_score', 0.0)), 100.0),
                open_pos=self.alpha_wallet.open_positions.get(key),
            )
            sig['timeframe'] = tf
            sig['pair']      = symbol

            self.alpha_signals[key] = sig

            existing = self.alpha_wallet.open_positions.get(key)
            fire     = bool(result.get('fire', False))
            side     = result.get('side', 'FLAT')

            if existing:
                cur = self.live_prices.get(symbol, price)
                sl_hit = (existing.direction == 'LONG' and cur <= existing.stop_loss) or \
                         (existing.direction == 'SHORT' and cur >= existing.stop_loss)
                reversal = fire and ((existing.side == 'BUY' and side == 'SELL') or
                                     (existing.side == 'SELL' and side == 'BUY'))
                if sl_hit:
                    self.alpha_wallet.close_trade(key, cur, 'SL_HIT')
                    self._alpha_last_close_time[key] = time.time()
                    self._alpha_last_close_side[key] = existing.side
                    self._save_alpha_track_record()
                elif reversal:
                    self.alpha_wallet.close_trade(key, cur, 'SIGNAL_REVERSAL')
                    self._alpha_last_close_time[key] = time.time()
                    self._alpha_last_close_side[key] = existing.side
                    self._save_alpha_track_record()
                    self._alpha_open_position(key, symbol, result, price, tf)
            elif fire and price > 0:
                cooldown = time.time() - self._alpha_last_close_time.get(key, 0)
                if cooldown >= 1800:
                    self._alpha_open_position(key, symbol, result, price, tf)

    def _save_alpha_track_record(self) -> None:
        try:
            import os as _os
            ALPHA_TRACK_RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)

            open_records = []
            for key, p in self.alpha_wallet.open_positions.items():
                sym, tf = key.rsplit('|', 1) if '|' in key else (key, '1h')
                cur     = self.live_prices.get(sym, p.entry_price) or p.entry_price
                if p.direction == 'LONG':
                    pnl_pct = (cur - p.entry_price) / p.entry_price * 100 if p.entry_price else 0.0
                else:
                    pnl_pct = (p.entry_price - cur) / p.entry_price * 100 if p.entry_price else 0.0
                open_records.append({
                    'signal_id':      p.signal_id,
                    'symbol':         sym,
                    'timeframe':      tf,
                    'direction':      p.direction,
                    'side':           p.side,
                    'entry_price':    p.entry_price,
                    'current_price':  round(cur, 8),
                    'exit_price':     None,
                    'entry_time':     p.entry_time,
                    'close_time':     None,
                    'pnl_pct':        round(pnl_pct, 4),
                    'pnl_usdt':       round(pnl_pct / 100 * p.position_value, 4),
                    'outcome':        'OPEN',
                    'exit_reason':    None,
                    'meta_confidence': p.meta_confidence,
                    'position_value': p.position_value,
                    'initial_value':  p.initial_value,   # v82: TP partial-sizing base
                    'stop_loss':      p.stop_loss,
                    'take_profit_1':  p.take_profit_1,
                    'take_profit_2':  p.take_profit_2,
                    'take_profit_3':  p.take_profit_3,
                    'take_profit_4':  p.take_profit_4,
                    'take_profit_5':  p.take_profit_5,
                    'atr':            p.atr,
                    'signal_strength': '',
                })

            history_records = []
            for rec in self.alpha_wallet.trade_history:
                d   = asdict(rec)
                raw = d.get('symbol', '')
                if '|' in raw:
                    d['symbol'], d['timeframe'] = raw.rsplit('|', 1)
                else:
                    d['timeframe'] = '1h'
                history_records.append(d)

            all_records = sorted(
                history_records + open_records,
                key=lambda r: r.get('entry_time') or '',
                reverse=True,
            )[:500]

            wins   = sum(1 for r in history_records if r.get('outcome') == 'WIN')
            losses = sum(1 for r in history_records if r.get('outcome') == 'LOSS')
            total  = wins + losses

            payload: Dict[str, Any] = {
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'mode':         'alpha',
                'timeframes':   _ALPHA_TIMEFRAMES,
                'summary': {
                    'balance':         round(self.alpha_wallet.balance, 2),
                    'initial_capital': self.alpha_wallet.initial_capital,
                    'total_trades':    total,
                    'wins':            wins,
                    'losses':          losses,
                    'win_rate':        round(wins / total, 3) if total else 0.0,
                    'open_positions':  len(self.alpha_wallet.open_positions),
                },
                'signals': all_records,
            }

            tmp = ALPHA_TRACK_RECORD_PATH.with_suffix('.tmp')
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2, default=str)
            _os.replace(tmp, ALPHA_TRACK_RECORD_PATH)
        except Exception as e:
            print(f'[AlphaEngine] alpha_track_record save failed: {e}')

    def _save_track_record(self) -> None:
        try:
            import os as _os, shutil as _shutil
            TRACK_RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)

            # ── Aggregate partial-TP slices into ONE record per trade ─────────
            # Each TP hit and the final close append a SEPARATE TradeRecord with
            # the same signal_id.  Serialising all of them made the public record
            # keep only the FIRST slice (TP1, a tiny +0.5%), mark the trade
            # "closed", and drop the still-open remainder + every later TP — so a
            # trade that ran to TP3 showed as a closed +0.5% and the avg PnL
            # looked weak.  Collapse each signal_id's slices into a single
            # whole-trade record: summed PnL (usdt), position-weighted %, correct
            # WIN/LOSS/OPEN, plus tp_hits and banked profit for display.
            from collections import defaultdict as _defaultdict
            _slices_by_id: Dict[str, list] = _defaultdict(list)
            for t in self.wallet.trade_history:
                _slices_by_id[t.signal_id].append(t)
            _open_ids = {p.signal_id for p in self.wallet.open_positions.values()}

            open_records = []
            for p in self.wallet.open_positions.values():
                cur = self.live_prices.get(p.symbol, p.entry_price) or p.entry_price
                if p.direction == 'LONG':
                    _rem_pct = (cur - p.entry_price) / p.entry_price * 100 if p.entry_price else 0.0
                else:
                    _rem_pct = (p.entry_price - cur) / p.entry_price * 100 if p.entry_price else 0.0
                _rem_usdt  = _rem_pct / 100 * p.position_value
                _banked    = _slices_by_id.get(p.signal_id, [])
                _bank_usdt = sum(t.pnl_usdt for t in _banked)
                _bank_val  = sum(t.position_value for t in _banked)
                _tot_val   = _bank_val + p.position_value
                _tot_usdt  = _bank_usdt + _rem_usdt
                _agg_pct   = (_tot_usdt / _tot_val * 100) if _tot_val else _rem_pct
                open_records.append({
                    'signal_id':       p.signal_id,
                    'symbol':          p.symbol,
                    'direction':       p.direction,
                    'side':            p.side,
                    'entry_price':     p.entry_price,
                    'current_price':   round(cur, 8),
                    'exit_price':      None,
                    'entry_time':      p.entry_time,
                    'close_time':      None,
                    'pnl_pct':         round(_agg_pct, 4),
                    'pnl_usdt':        round(_tot_usdt, 4),
                    'banked_usdt':     round(_bank_usdt, 4),
                    'tp_hits':         len(_banked),
                    'outcome':         'OPEN',
                    'exit_reason':     None,
                    'meta_confidence': p.meta_confidence,
                    'position_value':  round(_tot_val, 2),
                    'signal_strength': p.signal_strength,
                    'entry_mode':      p.entry_mode,
                    'atr':             p.atr,
                    'atr_multiplier':  p.atr_multiplier,
                    'stop_loss':       p.stop_loss,
                    'take_profit_1':   p.take_profit_1,
                    'take_profit_2':   p.take_profit_2,
                    'take_profit_3':   p.take_profit_3,
                    'take_profit_4':   p.take_profit_4,
                    'take_profit_5':   p.take_profit_5,
                })

            closed_records = []
            for _sid, _slices in _slices_by_id.items():
                if _sid in _open_ids:
                    continue   # remainder still open — folded into open_records above
                _tot_usdt = sum(t.pnl_usdt for t in _slices)
                _tot_val  = sum(t.position_value for t in _slices)
                _agg_pct  = (_tot_usdt / _tot_val * 100) if _tot_val else 0.0
                _final    = next((t for t in reversed(_slices)
                                  if 'PARTIAL' not in (t.exit_reason or '')), _slices[-1])
                _rec = asdict(_final)
                _rec.update({
                    'pnl_pct':        round(_agg_pct, 4),
                    'pnl_usdt':       round(_tot_usdt, 4),
                    'tp_hits':        sum(1 for t in _slices if 'PARTIAL' in (t.exit_reason or '')),
                    'outcome':        'WIN' if _tot_usdt > 0 else 'LOSS',
                    'position_value': round(_tot_val, 2),
                })
                closed_records.append(_rec)

            wallet_records = closed_records + open_records
            wallet_ids = {r.get('signal_id') for r in wallet_records if r.get('signal_id')}

            # Merge: preserve any on-disk records not currently in the wallet so
            # records are never lost due to restarts or wallet resets.
            # Dedup by position key (symbol + entry_minute + direction) to avoid
            # accumulating duplicates from the two parallel tracking systems.
            def _pos_key(r: dict) -> tuple:
                dr = r.get('direction', '') or r.get('side', '')
                return (r.get('symbol', ''), (r.get('entry_time') or '')[:16], dr)

            wallet_pos_keys = {_pos_key(r) for r in wallet_records}
            orphan_records: list = []
            if TRACK_RECORD_PATH.exists():
                try:
                    with open(TRACK_RECORD_PATH, 'r', encoding='utf-8') as _f:
                        _old = json.load(_f)
                    for r in _old.get('signals', []):
                        sid = r.get('signal_id')
                        if (sid and sid not in wallet_ids) and _pos_key(r) not in wallet_pos_keys:
                            # Never resurrect OPEN ghosts: an open record the
                            # wallet no longer tracks is a stale duplicate
                            # (restart artifact or parallel-writer leftover).
                            # The wallet is the single source of truth for
                            # open positions; only closed history is preserved.
                            if r.get('outcome') == 'OPEN':
                                continue
                            orphan_records.append(r)
                            wallet_pos_keys.add(_pos_key(r))
                except Exception:
                    pass

            all_records = sorted(
                wallet_records + orphan_records,
                key=lambda r: r.get('entry_time') or '',
                reverse=True,
            )[:500]

            self.portfolio_guard.sync_from_wallet(self.wallet.open_positions)
            payload: Dict[str, Any] = {
                'generated_at':      datetime.now(timezone.utc).isoformat(),
                'engine_version':    self.GATE_VERSION,
                'summary':           self.wallet.summary,
                'signals':           all_records,
                'performance':       self.perf_tracker.get_performance_summary(),
                'drift':             self.drift_monitor.get_summary(),
                'portfolio':         self.portfolio_guard.get_summary(),
                # Edge over geometry — see scripts/engine/edge_metric.py. The
                # win rate above is a report on the stop distance; this is the
                # part of it the signals earned. Written alongside rather than
                # replacing it, because the win rate is what subscribers read.
                'edge':              _edge_over_geometry(all_records),
            }

            tmp = TRACK_RECORD_PATH.with_suffix('.tmp')
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2, default=str)
            _os.replace(tmp, TRACK_RECORD_PATH)

            # Mirror to Firestore so the record survives the next Railway redeploy.
            # Fire-and-forget on a daemon thread: a slow/hung Firestore network
            # call must never block the scan loop (a hang isn't caught by
            # try/except and would freeze the engine).
            import threading as _threading
            _threading.Thread(
                target=_fs_save_track_record, args=(payload,), daemon=True
            ).start()

            # Sync to web/ so the static file server and main.py fallback stay current
            _web = _ROOT / 'web' / 'track_record.json'
            _web.parent.mkdir(parents=True, exist_ok=True)
            _shutil.copy2(TRACK_RECORD_PATH, _web)
        except Exception as e:
            print(f'[LiveEngine] track_record save failed: {e}')
