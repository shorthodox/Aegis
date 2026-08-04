"""Structure and confirmation gates.

These run AFTER the model has a side and ask whether the tape agrees:
is price actually at the level, did the retest hold, do the lower
timeframes confirm. They consume closed bars only, so a verdict never
repaints.

Extracted verbatim from the single-file live_engine.py; the bodies are
unchanged, only the class they hang off moved. LiveEngine composes this
mixin, so `self` is the full engine.
"""
from __future__ import annotations

from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

from scripts.engine.indicators import _detect_bos_choch
from scripts.engine.indicators import _detect_divergence
from scripts.engine.indicators import _detect_volume_events
from scripts.engine.indicators import _reversal_candle


class GatesMixin:
    """_structure_gate .. _confirmation_gate — see module docstring."""

    async def _structure_gate(
        self, symbol: str, side: str, price: float, result: Dict[str, Any]
    ) -> Tuple[str, str]:
        """
        AGGRESSIVE entry gate (v42): fire AT support/resistance IMMEDIATELY.
        Lower-timeframe confirmation is a BONUS, never a requirement. Mid-range
        fires with a WIDER stop (model conviction); wrong-side breakouts fire on
        retest / sustained momentum / pre-break. The only non-fire outcomes are
        'approaching' (WAIT -> pending until price reaches the level) and
        degenerate S/R (WARN). Risk is managed by the hybrid SL + RR gate
        downstream, NOT by discarding signals.
        Returns (verdict, detail): 'PASS' | 'WARN' | 'WAIT'.
        """
        support    = float(result.get('support', 0) or 0)
        resistance = float(result.get('resistance', 0) or 0)
        atr_g      = float(result.get('atr', 0) or 0) or price * 0.005
        range_pos_fwd = result.get('range_position')

        if not (0 < support < resistance) or price <= 0:
            return 'WARN', 'S/R data degenerate — firing with caution'

        # Live swing S/R for accurate location; write back so the chart and the
        # hybrid SL/TP anchor to the same levels the gate judged.
        swing_sr = await self._swing_sr(symbol, price, atr_g)
        if swing_sr:
            support, resistance = swing_sr
            result['support']    = support
            result['resistance'] = resistance
            range_pos = max(0.0, min(1.0, (price - support) / (resistance - support)))
        elif range_pos_fwd is not None:
            range_pos = float(range_pos_fwd)
        else:
            range_pos = max(0.0, min(1.0, (price - support) / (resistance - support)))

        bullish = (side == 'BUY')
        AT_SUPPORT    = range_pos <= self.STRUCT_SUPPORT_ZONE     # 0.35
        AT_RESISTANCE = range_pos >= self.STRUCT_RESISTANCE_ZONE  # 0.65
        correct_level  = AT_SUPPORT    if bullish else AT_RESISTANCE
        breakout_level = AT_RESISTANCE if bullish else AT_SUPPORT

        # Lower-timeframe candles — confirmation is a BONUS, not a blocker.
        raw_5m  = await self._fetch_candles(
            symbol, '5m', max(self.STRUCT_5M_WINDOW, self.STRUCT_RETEST_LOOKBACK) + 2)
        raw_15m = await self._fetch_candles(symbol, '15m', self.STRUCT_15M_WINDOW + 2)
        closed_5m  = raw_5m[:-1]  if len(raw_5m)  >= 2 else []
        closed_15m = raw_15m[:-1] if len(raw_15m) >= 2 else []

        def _n_trending(candles: list, n: int) -> int:
            return sum(1 for c in candles[-n:]
                       if (float(c[4]) > float(c[1])) == bullish and float(c[4]) != float(c[1]))

        n5  = _n_trending(closed_5m,  self.STRUCT_5M_WINDOW)  if closed_5m  else 0
        n15 = _n_trending(closed_15m, self.STRUCT_15M_WINDOW) if closed_15m else 0
        confirmed        = (n5 >= self.STRUCT_5M_MIN and n15 >= self.STRUCT_15M_MIN)
        partly_confirmed = (n5 >= self.STRUCT_5M_MIN)
        # v44: a COUNTER-TREND reversal (SELL in a bull / BUY in a bear, ADX-trending)
        # must not fire on a still-running move — require at least 2 of the 5m candles
        # to have TURNED in the signal direction first (shorting a rising candle is
        # how the counter-trend losses happen). Aligned/range reversals are unaffected.
        _mbias = str(result.get('market_bias', 'NEUTRAL') or 'NEUTRAL').upper()
        _adx   = float(result.get('adx', 0) or 0)
        counter_trend = _adx > 25 and (
            (not bullish and _mbias == 'BULLISH') or (bullish and _mbias == 'BEARISH'))

        tolerance = max(price * 0.0015, atr_g * 0.25)   # defined ONCE — used by both cases
        recent    = closed_5m[-self.STRUCT_RETEST_LOOKBACK:] if closed_5m else []
        _lname    = 'support' if bullish else 'resistance'

        # Volume Absorption & Liquidity Sweep Detection
        _vol_absorption = True
        _rel_vol5 = 1.0
        if closed_5m and len(closed_5m) >= 5:
            _vols = [float(c[5]) for c in closed_5m if len(c) > 5]
            if len(_vols) >= 5:
                _avg_v = sum(_vols[:-1]) / max(1, len(_vols) - 1)
                _rel_vol5 = (_vols[-1] / _avg_v) if _avg_v > 0 else 1.0
                _vol_absorption = (_rel_vol5 >= 1.10)

        _liq_sweep = False
        if recent and (0 < support < resistance):
            if bullish:
                _liq_sweep = any(float(c[3]) < support and float(c[4]) >= support for c in recent)
            else:
                _liq_sweep = any(float(c[2]) > resistance and float(c[4]) <= resistance for c in recent)
        if _liq_sweep:
            result['liquidity_sweep'] = True

        # Minimum Risk-to-Reward (RR) Headroom Check (Min 1.4:1)
        if 0 < support < resistance:
            _sl_est_dist = (0.5 * atr_g) + (abs(price - support) if bullish else abs(resistance - price))
            _sl_est_dist = max(0.7 * atr_g, min(_sl_est_dist, 1.8 * atr_g))
            if bullish:
                _headroom = (resistance - price) / max(1e-6, _sl_est_dist)
            else:
                _headroom = (price - support) / max(1e-6, _sl_est_dist)
            result['rr_headroom'] = round(_headroom, 2)
            if _headroom < 1.4:
                return 'WAIT', f'insufficient RR headroom ({_headroom:.2f}:1 < 1.40:1) to opposing level'

        # ── CASE 1: CORRECT LEVEL (BUY at support / SELL at resistance) ────────
        # Primary setup — fire at/near the level ONLY with 5m 3-candle reversal confirmation.
        if correct_level:
            level = support if bullish else resistance
            dist  = abs(price - level)
            
            # Counter-trend RSI & volume absorption gate
            _rsi_val = float(result.get('rsi', 50) or 50)
            if counter_trend:
                _rsi_ok = (_rsi_val <= 35) if bullish else (_rsi_val >= 65)
                if not _rsi_ok and not _vol_absorption:
                    return 'WAIT', f'counter-trend {_lname}_reversal requires RSI extreme (RSI {_rsi_val:.1f}) or volume absorption'

            # v46: REJECTION FAST-PATH — price TAGGED the level (a recent wick reached
            # it) and has now moved MORE than STRUCT_REJECTION_PCT (10%) of the range
            # back off it: a SELL >10% BELOW the resistance it hit, a BUY >10% ABOVE
            # the support it hit. The rejection itself IS the confirmation, so it
            # fires immediately — bypassing the counter-trend 2-candle wait and the
            # "far from level" pending.
            _rng  = resistance - support
            _tag  = (any((float(c[2]) >= resistance - tolerance) if not bullish
                         else (float(c[3]) <= support + tolerance) for c in recent)
                     if recent else False)
            _back = (resistance - price) if not bullish else (price - support)
            if _rng > 0 and _tag and _back > self.STRUCT_REJECTION_PCT * _rng:
                return 'PASS', (f'{_lname}_rejection — tagged {level:.6g}, now '
                                f'{100 * _back / _rng:.0f}% back off it (confirmed reversal)')
            tested = (any(((float(c[3]) <= level + tolerance) if bullish
                           else (float(c[2]) >= level - tolerance)) for c in recent)
                      if recent else False)
            prox = (atr_g * self.STRUCT_LEVEL_PROXIMITY_ATR) if atr_g else price * 0.01
            near = dist <= prox
            if tested or near:
                _pat = _reversal_candle(closed_5m, want_bullish=bullish) if closed_5m else None
                if confirmed or _liq_sweep:
                    return 'PASS', f'{_lname}_reversal confirmed ({"sweep+" if _liq_sweep else ""}{_pat or f"5m {n5}/{self.STRUCT_5M_WINDOW}"}, 15m {n15}/{self.STRUCT_15M_WINDOW}) @ {level:.6g}'
                if partly_confirmed or _pat is not None:
                    return 'PASS', f'{_lname}_reversal 5m-confirmed ({_pat or f"5m {n5}/{self.STRUCT_5M_WINDOW}"}) @ {level:.6g}'
                # Require proper 5m 3-candle reversal confirmation before firing
                return 'WAIT', f'{_lname}_reversal unconfirmed @ {level:.6g} — waiting for 5m 3-candle reversal confirmation ({n5}/3 turned)'
            # Still far from the level -> PENDING (wait for a closer entry)
            return 'WAIT', f'{_lname}_reversal far (dist {dist:.6g}) — waiting for a closer entry'

        # ── CASE 2: BREAKOUT LEVEL (BUY at resistance / SELL at support) ───────
        # Fire on retest or sustained momentum past level; do NOT buy below resistance
        if breakout_level:
            level  = resistance if bullish else support
            beyond = (price > level) if bullish else (price < level)
            if beyond and self._retest_held(recent, side, level, tolerance):
                return 'PASS', f'breakout_retest {"confirmed" if confirmed else "unconfirmed"} @ {level:.6g}'
            _recent5 = closed_5m[-self.STRUCT_BREAKOUT_MIN_HOLD:] if closed_5m else []
            _bc = (lambda c: float(c[4]) > level) if bullish else (lambda c: float(c[4]) < level)
            _held = (len(_recent5) >= self.STRUCT_BREAKOUT_MIN_HOLD and all(_bc(c) for c in _recent5))
            _ext = (price - level) if bullish else (level - price)
            _overext = atr_g > 0 and _ext > atr_g * self.STRUCT_BREAKOUT_MAX_EXT_ATR
            if _held and not _overext:
                return 'PASS', f'breakout_continuation {"confirmed" if confirmed else "momentum"} @ {level:.6g}'
            # An unconfirmed wrong-side entry below resistance / above support must wait
            if beyond:
                return 'WAIT', f'post_break @ {level:.6g} — waiting for a retest-hold'
            return 'WAIT', f'breakout_unconfirmed @ {level:.6g} — approaching {_lname} without breakout confirmation'

        # ── CASE 3: MID-RANGE — fire with a WIDER stop (model conviction) ──────
        target = resistance if bullish else support
        if len(closed_5m) >= 2:
            _chg = float(closed_5m[-1][4]) - float(closed_5m[-2][4])
            moving_toward = (_chg > 0) if bullish else (_chg < 0)
        else:
            moving_toward = False
        # v43: mid-range fires ONLY when 5m momentum is already moving toward the
        # target (a reversal underway). A mid-range signal with NO momentum ->
        # PENDING (wait for the move or for price to reach the level). Cuts the
        # weakest mid-range fills without losing the signal.
        if moving_toward:
            return 'PASS', f'mid_range moving to {("resistance" if bullish else "support")} {target:.6g} — wide stop'
        return 'WAIT', f'mid_range no-momentum (range_pos={range_pos:.2f}) — waiting for the move'

    @staticmethod
    def _retest_held(candles: list, side: str, level: float, tol: float) -> bool:
        """
        True when the level broke and a pullback held it.  For BUY: price is
        currently above old resistance AND some closed 5m candle opened above
        the level, dipped its low into the level zone (± tol), and closed back
        above it — the broken level acting as support.  Mirror for SELL.
        """
        if not candles or level <= 0:
            return False
        last_close = float(candles[-1][4])
        if side == 'BUY':
            if last_close <= level:
                return False
            return any(
                float(c[1]) > level and float(c[3]) <= level + tol and float(c[4]) > level
                for c in candles
            )
        if last_close >= level:
            return False
        return any(
            float(c[1]) < level and float(c[2]) >= level - tol and float(c[4]) < level
            for c in candles
        )

    async def _confirmation_gate(
        self, symbol: str, side: str, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Score structure + momentum + volume AGREEMENT with the signal on the 1h
        (model) timeframe.  Three independent families vote bullish (+) / bearish
        (−) / neutral (0):

          • Break of Structure / CHoCH   (_detect_bos_choch)
          • RSI + MACD divergence        (_detect_divergence, ±2 when aligned)
          • Volume climax + absorption   (_detect_volume_events)

        The votes are summed and re-signed to the trade direction:
        `agreement > 0` supports the trade, `< 0` fights it.  Returns a dict with
        verdict CONFIRM / NEUTRAL / CONFLICT, the score, a human reason, and the
        raw sub-signals (surfaced for the UI / track record).  Never fetches or
        touches the model; fail-open (NEUTRAL) on thin data so it can only ever
        add an advisory objection, never a silent block.
        """
        bullish = (side == 'BUY')
        raw = await self._fetch_candles(
            symbol, '1h', self.CONFIRM_BOS_LOOKBACK + 5)
        closed = raw[:-1] if len(raw) >= 2 else []
        if len(closed) < self.CONFIRM_MIN_BARS:
            return {'verdict': 'NEUTRAL', 'score': 0.0,
                    'reason': f'insufficient 1h data ({len(closed)} bars)',
                    'signals': {}}

        # Use a dynamic lookback so BOS/CHoCH can be detected with available history
        _dyn_lb = min(self.CONFIRM_BOS_LOOKBACK, max(1, len(closed) - 2))
        bos = _detect_bos_choch(closed, lookback=_dyn_lb)
        div = _detect_divergence(closed, k=self.CONFIRM_PIVOT_K)
        vol = _detect_volume_events(closed, window=self.CONFIRM_VOL_WINDOW,
                                    climax_z=self.CONFIRM_CLIMAX_Z,
                                    absorb_z=self.CONFIRM_ABSORB_Z)

        # Directional votes: bullish (+) / bearish (−).  `agree()` re-signs a
        # vote to the TRADE direction (+ supports the trade, − fights it).
        def agree(vote: float) -> float:
            return vote if bullish else -vote

        v_choch  = float(bos['choch_bull'] - bos['choch_bear'])   # character change (reversal)
        v_div    = float(div['rsi'] + div['macd'])                # −2 … +2
        v_climax = float(vol['climax'])
        v_absorb = float(vol['absorption'])

        # BOS *continuation* is CONFIRM-ONLY: a bullish break confirms a long /
        # a bearish break confirms a short, but it must NEVER count against a
        # reversal.  A short at resistance is FADING prior up-structure — that
        # bullish BOS is the setup, not a contradiction.  So continuation adds
        # to agreement only when it already points the trade's way.
        bos_state = float(bos['bos_state'])
        bos_confirm = 1.0 if (bos_state != 0 and (bos_state > 0) == bullish) else 0.0

        agreement = (agree(v_choch) + agree(v_div) + agree(v_climax)
                     + agree(v_absorb) + bos_confirm)

        # Human-readable evidence, framed relative to the trade direction.
        def _label(vote: float, name: str) -> Optional[str]:
            if vote == 0:
                return None
            vote_bull = vote > 0
            tag = 'confirms' if (vote_bull == bullish) else 'opposes'
            return f'{name} {tag} ({"bull" if vote_bull else "bear"})'

        parts = [p for p in (
            _label(v_choch,  'CHoCH'),
            _label(v_div,    'divergence'),
            _label(v_climax, 'climax'),
            _label(v_absorb, 'absorption'),
        ) if p]
        if bos_confirm:
            parts.append(f'BOS confirms ({"bull" if bullish else "bear"})')
        reason = '; '.join(parts) if parts else 'no structure/momentum/volume signal'

        if agreement <= -self.CONFIRM_CONFLICT_THRESHOLD:
            verdict = 'CONFLICT'
        elif agreement >= self.CONFIRM_CONFIRM_THRESHOLD:
            verdict = 'CONFIRM'
        else:
            verdict = 'NEUTRAL'

        return {
            'verdict': verdict,
            'score':   round(agreement, 2),
            'reason':  reason,
            'signals': {
                'choch':      v_choch,
                'bos_state':  bos_state,
                'divergence': v_div,
                'rsi_div':    float(div['rsi']),
                'macd_div':   float(div['macd']),
                'climax':     v_climax,
                'absorption': v_absorb,
                'vol_z':      round(float(vol['vol_z']), 2),
            },
        }
