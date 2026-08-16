"""Multi-layer signal quality scoring.

Uses only fields already present in the predictor's result dict — no extra API
calls.

The recurring theme in this file is not double-penalising a reversal. A genuine
exhaustion turn necessarily misses every trend-following bonus AND would take
the opposing-HTF penalty, which dragged real setups (ADA SELL at resistance with
RSI 97.7 scored 43) far below the floor. Reversals therefore earn an explicit
bonus and are exempted from the trend-opposition penalties. Given the strategy
is mean-reversion, that exemption is load-bearing — check it before changing any
threshold here.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from scripts.engine.config import REGIME_LIQUIDITY_TRAP, REGIME_RANGING
from scripts.engine.models import RegimeState

__all__ = ["SignalQualityFilter"]


class SignalQualityFilter:
    """
    Multi-layer quality scoring.  Uses only fields from the result dict — no
    extra API calls.

    score_signal() → (float quality 0-100, list[str] reasons)
    is_fake_breakout() → bool
    """

    MIN_QUALITY_SCORE = 60.0  # v43: edge floor 55->60 — cut the coin-flip signals (biggest WR lever) without gutting count

    # Normalised range-position boundaries used by score_signal() when
    # identifying structural support and resistance reversals.
    STRUCT_SUPPORT_ZONE = 0.35
    STRUCT_RESISTANCE_ZONE = 0.65

    def score_signal(
        self,
        result: Dict[str, Any],
        regime: RegimeState,
        side: str,
    ) -> Tuple[float, List[str]]:
        """
        Score a potential entry signal.  Returns (quality_score, reasons).
        quality_score range: 0 – 100 (clipped).
        """
        score: float = 0.0
        reasons: List[str] = []

        # ── helper: safe float read ───────────────────────────────────────────
        def _f(k: str, default: float = 0.0) -> float:
            v = result.get(k, default)
            try:
                return float(v) if v is not None else default
            except (TypeError, ValueError):
                return default

        _conf_data  = result.get('confluence') or {}
        conf_total  = float(_conf_data.get('total', 5.0))
        adx         = _f('adx', 20.0)
        vol_zscore  = _f('volume_zscore', 0.0)
        meta_conf   = _f('edge_score', 0.0) or _f('meta_confidence', 0.0)
        rsi         = _f('rsi', 50.0)
        funding_bias= str(result.get('funding_bias', 'NEUTRAL') or 'NEUTRAL').upper()
        oi_trend    = str(result.get('oi_trend', 'STABLE') or 'STABLE').upper()
        market_bias = str(result.get('market_bias', 'NEUTRAL') or 'NEUTRAL').upper()

        # ── Reversal-setup detection ──────────────────────────────────────────
        # A counter-trend signal AT the structural extreme with an EXHAUSTED
        # oscillator (overbought SELL at resistance / oversold BUY at support) is
        # a legitimate turn, not a weak trend entry.  Without this the scorer
        # double-penalises reversals — they necessarily miss the trend-following
        # confluence/bias bonuses AND take the HTF-oppose penalty — which drags
        # genuine exhaustion signals far below the 70 floor (ADA SELL @
        # resistance, RSI 97.7, scored 43).  We reward the setup and skip the
        # HTF penalty for it (being counter-trend is the definition of a
        # reversal — penalising it for that is double-counting).
        _rev_sup   = _f('support', 0.0)
        _rev_res   = _f('resistance', 0.0)
        _rev_price = _f('price', 0.0) or _f('entry_price', 0.0)
        if 0 < _rev_sup < _rev_res and _rev_price > 0:
            _rev_rp = max(0.0, min(1.0, (_rev_price - _rev_sup) / (_rev_res - _rev_sup)))
        else:
            _rev_rp = float(result.get('range_position') or 0.5)
        # Strict reversal (requires both extreme location + exhausted oscillator)
        is_reversal_strict = (
            (side == 'SELL' and _rev_rp >= 0.65 and rsi >= 68) or
            (side == 'BUY'  and _rev_rp <= 0.35 and rsi <= 32)
        )
        # Structural reversal: price sits at the support/resistance zone (weaker
        # than strict reversal but still a valid fade candidate). Used to avoid
        # double-penalising legitimate counter-trend setups.
        is_structural_reversal = (
            (side == 'BUY'  and _rev_rp <= self.STRUCT_SUPPORT_ZONE) or
            (side == 'SELL' and _rev_rp >= self.STRUCT_RESISTANCE_ZONE)
        )

        # ── Positive contributions ────────────────────────────────────────────

        # +20: strong confluence lean in signal direction
        if side == 'BUY'  and conf_total > 6.5:
            score += 20; reasons.append('strong_bull_confluence')
        elif side == 'SELL' and conf_total < 3.5:
            score += 20; reasons.append('strong_bear_confluence')

        # +15: trending market (ADX confirms momentum)
        if adx > 25:
            score += 15; reasons.append(f'adx_trending({adx:.1f})')

        # +10: strong volume conviction
        if vol_zscore > 1.5:
            score += 10; reasons.append(f'strong_volume(z={vol_zscore:.1f})')

        # +10: regime is confident
        if regime.confidence > 0.7:
            score += 10; reasons.append(f'regime_confident({regime.regime})')

        # +10: primary model edge score is high (top-quartile signal, 0-100 scale)
        if meta_conf > 75.0:
            score += 10; reasons.append(f'high_edge_score({meta_conf:.1f})')

        # RSI zone scoring — graduated, not binary.
        # Rewards fresh-momentum entries; penalises late/overbought entries.
        # is_fake_breakout() hard-blocks RSI > 70 (BUY) / < 30 (SELL) so the
        # -8 penalty here is a belt-and-suspenders quality deduction.
        if side == 'BUY':
            if rsi < 55:
                score += 10; reasons.append(f'rsi_ideal({rsi:.1f})')    # fresh momentum
            elif rsi < 65:
                score += 5;  reasons.append(f'rsi_ok({rsi:.1f})')       # acceptable
            elif rsi < 70:
                pass                                                       # neutral — no bonus
            else:
                score -= 8;  reasons.append(f'rsi_overbought({rsi:.1f})')
        elif side == 'SELL':
            if rsi > 45:
                score += 10; reasons.append(f'rsi_ideal({rsi:.1f})')
            elif rsi > 35:
                score += 5;  reasons.append(f'rsi_ok({rsi:.1f})')
            elif rsi > 30:
                pass
            else:
                score -= 8;  reasons.append(f'rsi_oversold({rsi:.1f})')

        # +10: funding bias aligns with direction
        funding_align = (
            (side == 'BUY'  and funding_bias == 'SHORTS_PAYING') or
            (side == 'SELL' and funding_bias == 'LONGS_PAYING')  or
            funding_bias == 'NEUTRAL'
        )
        if funding_align:
            score += 10; reasons.append('funding_aligned')

        # +5: OI trend supports direction
        oi_align = (
            (side == 'BUY'  and oi_trend == 'INCREASING') or
            (side == 'SELL' and oi_trend == 'DECREASING') or
            oi_trend == 'STABLE'
        )
        if oi_align:
            score += 5; reasons.append('oi_aligned')

        # +5: market_bias matches direction
        bias_align = (
            (side == 'BUY'  and market_bias == 'BULLISH') or
            (side == 'SELL' and market_bias == 'BEARISH')
        )
        if bias_align:
            score += 5; reasons.append('bias_aligned')

        # +18: strict reversal setup — offsets the trend-following bonuses a
        # genuine exhaustion turn necessarily misses. Structural reversals (at
        # support/resistance zone) get a smaller bonus so they can still reach
        # the quality floor when appropriate.
        if is_reversal_strict:
            score += 18; reasons.append(f'reversal_setup(rp={_rev_rp:.2f},rsi={rsi:.0f})')
        elif is_structural_reversal:
            score += 8; reasons.append(f'structural_reversal(rp={_rev_rp:.2f})')

        # ── Penalties ─────────────────────────────────────────────────────────

        # -20: no-trade zone
        if regime.regime == REGIME_LIQUIDITY_TRAP:
            score -= 20; reasons.append('liquidity_trap_penalty')

        # -15: ranging market — no directional edge, signals are noise.  Skipped
        # for a reversal at the range extreme: SELL at range resistance / BUY at
        # range support IS the high-probability range trade, not noise.
        if regime.regime == REGIME_RANGING and not (is_reversal_strict or is_structural_reversal):
            score -= 15; reasons.append('ranging_market_penalty')

        # -10: low volume — no conviction behind the move
        if vol_zscore < -0.8:
            score -= 10; reasons.append(f'low_volume(z={vol_zscore:.1f})')

        # ── HTF macro bias (weekly + daily EMA50 trend) ──────────────────────
        # macro_weekly / macro_daily are binary: +1.0 (above EMA50) / -1.0 (below).
        # Threshold 0.5 separates the two states cleanly.
        #
        # Scoring tiers:
        #   +15: both weekly AND daily align with signal (strongest confirmation)
        #   +10: weekly aligns, daily opposing (pullback/bounce setup — ideal entry)
        #   -20: both weekly AND daily oppose signal (worst setup — nearly certain loss)
        #   -10: weekly opposes, daily neutral/aligned (counter-trend caution)
        #
        # Hard block (weekly+daily both opposing) is also enforced in Gate 1.7
        # of _process_symbol; the -20 here is belt-and-suspenders quality deduction.
        macro_daily  = _f('macro_daily',  0.0)
        macro_weekly = _f('macro_weekly', 0.0)
        if macro_weekly != 0.0 or macro_daily != 0.0:
            _w_bull = macro_weekly > 0.5
            _w_bear = macro_weekly < -0.5
            _d_bull = macro_daily  > 0.5
            _d_bear = macro_daily  < -0.5
            # Opposing-HTF penalties are skipped for a genuine reversal at the
            # extreme — being counter-trend is what a reversal IS, so penalising
            # it here on top of the missed trend bonuses is double-counting.
            if side == 'BUY':
                if _w_bull and _d_bull:
                    score += 15; reasons.append('htf_both_bullish')
                elif _w_bull and _d_bear:
                    score += 10; reasons.append('htf_pullback_buy(w+/d-)')
                elif _w_bear and _d_bear and not (is_reversal_strict or is_structural_reversal):
                    score -= 20; reasons.append('htf_both_bearish')
                elif _w_bear and not (is_reversal_strict or is_structural_reversal):
                    score -= 10; reasons.append('htf_weekly_bearish')
            elif side == 'SELL':
                if _w_bear and _d_bear:
                    score += 15; reasons.append('htf_both_bearish')
                elif _w_bear and _d_bull:
                    score += 10; reasons.append('htf_bounce_sell(w-/d+)')
                elif _w_bull and _d_bull and not (is_reversal_strict or is_structural_reversal):
                    score -= 20; reasons.append('htf_both_bullish')
                elif _w_bull and not (is_reversal_strict or is_structural_reversal):
                    score -= 10; reasons.append('htf_weekly_bullish')

        # ── LSTM temporal intelligence bonuses / penalties ─────────────────────
        # Only applied when LSTM models are available for this symbol.
        # Neutral defaults (0.5) are returned when models are absent, so
        # the thresholds below never fire for untrained symbols.
        lstm_avail = bool(result.get('lstm_available', False))
        if lstm_avail:
            lstm_cont = _f('lstm_continuation_prob', 0.5)
            lstm_vol  = _f('lstm_vol_expansion_prob', 0.5)
            lstm_exh  = _f('lstm_exhaustion_prob',    0.5)

            # +10: LSTM says momentum will persist (strong continuation signal)
            if lstm_cont > 0.65:
                score += 10; reasons.append(f'lstm_continuation({lstm_cont:.2f})')

            # -8: LSTM detects sequence exhaustion (momentum likely decaying).
            #
            # Skipped for a reversal at the extreme, and this one is not merely
            # double-counting — it is INVERTED. Decaying momentum is a defect in
            # a continuation trade and the entire thesis of a mean-reversion one.
            # Scoring "the move is running out of steam" as a fault on a trade
            # taken BECAUSE the move is running out of steam penalises the setup
            # for containing its own premise.
            elif lstm_cont < 0.32 and not (is_reversal_strict or is_structural_reversal):
                score -= 8; reasons.append(f'lstm_exhaustion({lstm_exh:.2f})')

            # +8: Volatility expansion expected — good for breakout trades
            if lstm_vol > 0.70:
                score += 8; reasons.append(f'lstm_vol_expansion({lstm_vol:.2f})')

        # ── Candlestick reversal pattern alignment ────────────────────────────
        # cdl_bull/bear_reversal is the max weighted 1/2/3-candle reversal
        # pattern score over the last 3 closed bars (feature_engine, trend-
        # context gated).  -1.0 = data unavailable → no adjustment.
        # A strong aligned pattern (>= 1.5, i.e. a 3-candle formation) is the
        # highest-reliability entry timing evidence available to this scorer;
        # an opposing pattern printing right at entry is a direct warning.
        _cdl_bull = _f('cdl_bull_reversal', -1.0)
        _cdl_bear = _f('cdl_bear_reversal', -1.0)
        if _cdl_bull >= 0.0 and _cdl_bear >= 0.0:
            _aligned  = _cdl_bull if side == 'BUY' else _cdl_bear
            _opposing = _cdl_bear if side == 'BUY' else _cdl_bull
            if _aligned >= 1.5:
                score += 10; reasons.append(f'strong_reversal_pattern({_aligned:.1f})')
            elif _aligned > 0.0:
                score += 5;  reasons.append(f'reversal_pattern({_aligned:.1f})')
            if _opposing >= 1.0 and _opposing > _aligned:
                score -= 12; reasons.append(f'opposing_pattern({_opposing:.1f})')

        # ── MACD momentum alignment / conflict ────────────────────────────────
        # macd_signal is 'BULLISH' | 'BEARISH' | 'NEUTRAL' — same field already
        # consumed by MarketRegimeDetector, so no extra computation.
        # MACD (EMA crossover) is independent of RSI (price-change velocity),
        # making it a genuinely uncorrelated confirmation source.
        # Penalty for conflict (-12) > bonus for alignment (+8): conservative
        # asymmetry — blocking a false entry matters more than confirming a valid one.
        macd_sig = str(result.get('macd_signal', 'NEUTRAL') or 'NEUTRAL').upper()
        if macd_sig != 'NEUTRAL':
            if (side == 'BUY'  and macd_sig == 'BULLISH') or \
               (side == 'SELL' and macd_sig == 'BEARISH'):
                score += 8;  reasons.append(f'macd_aligned({macd_sig})')
            elif ((side == 'BUY'  and macd_sig == 'BEARISH') or
                  (side == 'SELL' and macd_sig == 'BULLISH')):
                # Halved rather than skipped for a reversal at the extreme.
                #
                # Unlike the ranging, HTF and exhaustion penalties, this one is
                # not purely double-counting. An opposing MACD is EXPECTED when
                # fading — you are trading against momentum by construction, so
                # charging the full -12 penalises the setup for its own premise.
                # But it also carries real information a fade cannot dismiss:
                # being early. The distinction the other exemptions can ignore,
                # this one cannot, so it is reduced, not removed.
                _pen = 6 if (is_reversal_strict or is_structural_reversal) else 12
                score -= _pen; reasons.append(f'macd_conflict({macd_sig})')

        return round(max(0.0, min(score, 100.0)), 1), reasons

    def is_fake_breakout(self, result: Dict[str, Any], side: str) -> bool:
        """
        Detect potential false breakout / exhaustion conditions.
        Returns True if the breakout looks fake and should be blocked.
        """
        try:
            def _f(k: str, default: float = 0.0) -> float:
                v = result.get(k, default)
                try:
                    return float(v) if v is not None else default
                except (TypeError, ValueError):
                    return default

            vol_zscore  = _f('volume_zscore', 0.0)
            rsi         = _f('rsi', 50.0)
            _conf_data  = result.get('confluence') or {}
            conf_mom    = float(_conf_data.get('momentum', 5.0))
            conf_total  = float(_conf_data.get('total', 5.0))

            # Clearly fake breakout: volume clearly below average AND technicals
            # actively contradict the direction (not just neutral — clearly opposed).
            # vol_zscore < -0.5 = clearly below-average volume (not just non-elevated).
            # conf_mom < 4.0 / > 6.0 = technicals clearly opposed, not just neutral.
            if vol_zscore < -0.5:
                if side == 'BUY'  and conf_mom < 4.0 and conf_total < 4.5:
                    return True
                if side == 'SELL' and conf_mom > 6.0 and conf_total > 5.5:
                    return True

            # RSI at classic overbought/oversold extremes — momentum structurally exhausted
            if side == 'BUY'  and rsi > 70:
                return True
            if side == 'SELL' and rsi < 30:
                return True

            # RSI deceleration: elevated RSI that is now rolling over is the classic
            # "0.5-1% up then reverses" pattern — momentum was already peaking when
            # the signal fired.  rsi_slope < 0 means RSI is falling despite being
            # elevated; rsi_acceleration < 0 means it is speeding up downward.
            rsi_slope = _f('rsi_slope', 0.0)
            rsi_accel = _f('rsi_acceleration', 0.0)
            if side == 'BUY' and rsi > 60 and rsi_slope < -0.3 and rsi_accel < 0:
                return True   # overbought + decelerating = entry is likely too late
            if side == 'SELL' and rsi < 40 and rsi_slope > 0.3 and rsi_accel > 0:
                return True   # oversold + recovering = entry is likely too late

            # LSTM exhaustion: sequence analysis suggests momentum is collapsing
            if bool(result.get('lstm_available', False)):
                lstm_cont = _f('lstm_continuation_prob', 0.5)
                if lstm_cont < 0.22:
                    return True  # very low continuation → likely exhausted move

            return False
        except Exception:
            return False
