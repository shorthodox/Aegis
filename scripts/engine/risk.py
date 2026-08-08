"""Position sizing and the ATR+structure stop/target geometry.

This is where v82, v84 and v85 live. Three rules, each of which cost a
measurable amount of money to learn, and none of which should be changed
without re-reading the comment attached to it:

  * v82 — the stop must match the geometry the model was TRAINED on. A stop
    tighter than the label's barrier band means the model can be right about the
    barrier and still be stopped out en route.
  * v84 — the wick buffer is a FLOOR, not an average. A stop resting on the
    level it defends is taken out both when price overshoots and reverses and
    when it breaks through and retests.
  * v85 — when TraderGate supplies the stop and the objective it priced the
    trade on, both are used VERBATIM. Re-deriving either means the R:R the gate
    approved is not the R:R the trade actually has.

All methods are pure functions (no state) — safe to call from async context.
"""
from __future__ import annotations

from typing import Dict

from scripts.engine.models import RegimeState

__all__ = ["DynamicRiskEngine"]


class DynamicRiskEngine:
    """
    Volatility-aware position sizing and ATR-based stop/take-profit calculation.
    All methods are pure functions (no state) — safe to call from async context.

    ATR configuration
    -----------------
    All TP/SL distances are expressed as multiples of the current 1-hour ATR(14).
    Changing a multiplier here automatically adapts every new position opened.
    """

    # ── Position sizing ───────────────────────────────────────────────────────
    BASE_POSITION_PCT = 0.07   # 7 % of balance as the base allocation
    MIN_POSITION_PCT  = 0.02   # floor: never risk less than 2 %
    MAX_POSITION_PCT  = 0.10   # ceiling: never risk more than 10 % per trade

    # ── ATR + Structure hybrid risk parameters ────────────────────────────────
    ATR_PERIOD        = 14     # lookback period for ATR calculation
    ATR_SL_MULTIPLIER = 1.8    # SL distance CAP = ATR × this (also the pure-ATR fallback when structure is missing)
    # Hybrid SL: anchor the stop just beyond the invalidation level (support for a
    # LONG, resistance for a SHORT) + a wick buffer, then clamp the resulting risk
    # leg to [FLOOR, CAP]×ATR so it is never noise-tight and never wider than the
    # old fixed 1.8×ATR stop. Entries are near the level (gate v36), so this
    # usually TIGHTENS risk → higher RR, without being inside the sweep zone.
    STRUCT_SL_BUFFER_ATR = 0.5   # place the stop this far ATR beyond the level's wick
    SL_FLOOR_ATR         = 0.7   # risk is never tighter than this (spread/noise floor)
    # RISKY-tier setups (counter-trend reversals, far-from-level, fake-breakout —
    # the low-conviction trades) get a TIGHT SL cap so a failed thesis is a small
    # loss, not the 2.2-2.5x ATR bleed the old "loose entry -> wider stop" logic
    # gave them. Risk less on the least-certain trades. Tradeoff: a tighter stop
    # is hit more often, but on a low-quality setup a quick small loss beats a big
    # one. STRONG/NORMAL keep the normal/structural cap (they earned the room).
    # v82: 1.2 -> 1.8, i.e. RISKY no longer gets a TIGHTER stop than anything
    # else.  The training label is a symmetric first-touch barrier at roughly
    # ±1.8×ATR over 12–18 h (retrain_model.py:1229), so a 1.2×ATR stop sits
    # INSIDE the band the label lets price wander through: the model can be
    # right about the barrier and still be stopped out en route.  Since almost
    # every live signal is tagged RISKY, that tight cap was the dominant path.
    # Tight-stop-and-near-target is the one combination to never run; the stop
    # now matches the geometry the model was actually trained on.
    RISKY_SL_CAP_ATR     = 1.8   # SL cap in ATR for RISKY-tier signals

    # TP ladder — COMPRESSED into a reachable region (2026-07-04).  The old
    # ladder (2.8 / 4.5 / 6.5 / 9.5) left a huge TP2→TP3 gap and put TP3-TP5 so
    # far out they almost never filled: price hit TP1+TP2 (~1.5%) and reversed
    # long before 4.5×ATR.  Spacing is now even and inside a plausible swing so
    # partials actually bank at each level.  SL stays 1.8×ATR; RR is validated to
    # TP5 (4.5/1.8 = 2.5) so trade acceptance is unchanged — only the interior
    # rungs moved closer.
    # v82: TP1 raised 0.7R -> 1.0R.  At 0.7R the ONLY reachable win was +0.7R
    # against a -1.0R stop, so break-even WR was 1/1.7 = 58.8 % — the engine ran
    # at 60 % and therefore made nothing before fees.  Measured on the live log:
    # mean win 0.556 % / mean loss 0.788 % = 0.705, i.e. exactly TP1_MULTIPLIER.
    # TP1 is now a 1:1 rung; the payoff comes from TP2+ riding the trail.
    TP1_MULTIPLIER    = 1.0    # 15 % partial close — first bank at 1R, no longer the de-facto exit
    TP2_MULTIPLIER    = 1.6    # 20 % partial close + activate trailing stop; also the trail FLOOR, so a higher TP2 locks more on every runner (up from 1.3)
    TP3_MULTIPLIER    = 2.2    # 20 % partial close — small step past TP2 (was 4.5, a near-unreachable gap)
    TP4_MULTIPLIER    = 3.3    # 20 % partial close — reachable stretch target
    TP5_MULTIPLIER    = 4.5    # close remaining position — full-trend target + RR anchor (4.5/1.8 = 2.5)

    # ── The ladder is priced in PERCENT OF ENTRY, not in R ────────────────────
    # The R-multiple ladder above is kept for reference and for anything that
    # still reads the multipliers; the rungs the engine actually places come
    # from TP_LADDER_PCT.
    #
    # Why: the rungs were derived from the stop, and the stop is ATR-derived, so
    # a wide-stop trade got a first objective a long way off in the only unit the
    # market pays in. Measured over the closed book, every loss had its first
    # real target 2.3-3.5 % away against a stop of 1.0-1.65 %:
    #
    #     SUI  risk 1.01 %   target 2.56 %      DOGE risk 1.11 %   target 2.33 %
    #     ONDO risk 1.65 %   target 3.33 %      ARB  risk 1.60 %   target 3.49 %
    #
    # The trade had to travel two to three times further to win than to lose, so
    # a reversal anywhere between entry and the first bank turned a position that
    # had been in profit into a full stop.
    #
    # KNOWN TRADE-OFF, deliberately accepted: a fixed percentage against an
    # ATR-derived stop means each rung lands at a DIFFERENT R per token — TP1 is
    # 0.50R on SUI but 0.30R on ONDO. v82 records that a TP1 at 0.7R made the
    # break-even win rate 58.8 %. Two things keep that from repeating: TP1 closes
    # only 15 % and its real job is arming break-even early, and the ladder still
    # ends at 2.1-3.5R where the trade is actually paid.
    TP_LADDER_PCT = (0.5, 1.5, 2.0, 3.0, 3.5)   # TP1 … TP5, percent of entry
    TP_MIN_GAP_PCT = 0.05                        # rungs must stay strictly apart

    # RETIRED: the TP2 % cap was for the former wide TP2 (2.8×ATR); with the
    # compressed ladder TP2 is only 1.3×ATR (modest at any volatility), so the
    # cap only kinked the ladder in high vol and was removed from calculate_stops.
    TP2_MAX_PCT       = 1.5    # (unused) former cap on TP2 distance, % of entry
    TP2_MIN_TP1_RATIO = 1.4    # ordering guard: TP2 distance ≥ 1.4× TP1

    MIN_RISK_REWARD   = 2.0    # Reward / Risk using TP5 (full-trend target) as reward; below this is rejected

    TRAIL_MULTIPLIER  = 1.0    # trailing stop distance = ATR × this (widened to match wider SL)

    # ── Give-back ratchet ─────────────────────────────────────────────────────
    # Once a TP rung is tagged, how much of THAT RUNG'S SPAN the remainder may
    # hand back before it is closed. The span is measured rung-to-rung:
    # entry→TP1 for the first, TP1→TP2 for the second, and so on.
    #
    # This is not the deleted TP1_RECROSS. That closed on any tick back through
    # a tagged TP — a zero-width buffer — and with TP1 at 0.7R against a 1.0R
    # stop it capped every winner at +0.7R. Two things changed: TP1 is 1.0R now
    # (v82), so a give-back exit books a full 1R rather than 0.7R, and the
    # buffer is no longer zero.
    #
    # Sizing it IS the question, and it was chosen by measurement rather than
    # taste. An entry→TP1 span runs ~1.9 ATR, so the leash in ATR is roughly
    # 1.9 × TP_GIVEBACK_PCT:
    #
    #     0.05 -> 0.09 ATR   ~8 % of one 1h bar. Noise tags it; TP3-TP5 stop
    #                        being reachable, which is the capped-runner problem
    #                        the recross was deleted for.
    #     0.35 -> 0.66 ATR   wide enough to sit outside ordinary bar noise, far
    #                        tighter than break-even. <- chosen
    #     0.50 -> 0.95 ATR   about a full bar; fires only on a real reversal.
    #     0.65 -> 1.23 ATR   wider than the BCH reversal that prompted this, so
    #                        it would not have closed it.
    #
    # No value both protects the rung and preserves v82's "runner reaches TP3"
    # behaviour — they are genuinely in conflict, and 0.35 is where the trade is
    # struck: keep the banked rung, accept a slightly lower TP3+ hit rate.
    TP_GIVEBACK_PCT     = 0.35  # fraction of the rung span the runner may hand back
    TP_GIVEBACK_MIN_ATR = 0.50  # ...but never a leash tighter than this in ATR
    #
    # The floor is not optional. The leash is a fraction of the RUNG SPAN, and
    # moving the ladder to percentages shrank every span by two to three times,
    # so the leash collapsed with it:
    #
    #     TP1 ~1R (1.0-1.65 % of price)  ->  leash 0.35-0.58 %  = 0.35-0.55 ATR
    #     TP1 0.5 % of price             ->  leash 0.175 %      = 0.16 ATR
    #
    # A sixth of one bar is noise. OP/USDT 2026-08-08 tagged TP1, banked its
    # 15 %, and had the remaining 85 % closed immediately after for +0.18 % net —
    # the capped-runner failure the recross was deleted for, reintroduced by a
    # change made two commits away from this constant.
    #
    # A floor alone is not enough either: on a 0.5 % rung, 0.5 ATR can be WIDER
    # than the whole span, which would put the give-back level past the entry —
    # worse than the break-even stop that is already there. _manage_exit
    # therefore skips any rung whose leash would reach its own span, and lets
    # break-even do the job for that rung. The ratchet earns its keep on the
    # later, wider rungs.

    # ── Partial-close percentages (must sum to 1.0) ───────────────────────────
    # Fractions of the ORIGINAL allocation (v82 — see partial_close_trade; they
    # used to be applied to the shrinking remainder, so each rung silently
    # banked less than its nominal share).  Front-loaded onto the two
    # "significant objective" targets (TP2/TP3); the 20 % runner rides the trail.
    TP_CLOSE_PCTS = (0.15, 0.25, 0.25, 0.15, 0.20)  # TP1 … TP5

    @classmethod
    def _ladder(cls, price: float, side: str, target: float = 0.0) -> tuple:
        """The five take-profit rungs, as percentages of entry.

        Returns (tp1..tp5) ordered away from `price` in the trade's direction.
        Percentages come from TP_LADDER_PCT and are held strictly monotonic by
        TP_MIN_GAP_PCT, so a mis-edited table cannot put two rungs at the same
        price — that would make one partial close a no-op and silently change
        the size of the runner.

        `target` is the STRUCTURAL objective the gate cleared the trade on
        (plan.target). When it is nearer than the top rung the whole ladder is
        scaled to land on it. That is v85's rule surviving the move to
        percentages: the payoff stage rejects a setup whose objective is too far
        to be real, so placing a rung BEYOND that objective would re-invent the
        thing the floor exists to reject, and would advertise a target the gate
        never vouched for. When the objective is further than the top rung
        nothing is scaled — banking earlier than the objective is conservative,
        and it is the whole point of a percentage ladder.
        """
        pcts = [float(p) for p in cls.TP_LADDER_PCT]
        if target > 0 and price > 0:
            tgt_pct = abs(target - price) / price * 100.0
            top = pcts[-1]
            if 0 < tgt_pct < top:
                scale = tgt_pct / top
                pcts = [p * scale for p in pcts]

        out, last = [], 0.0
        for p in pcts:
            p = max(p, last + cls.TP_MIN_GAP_PCT)
            last = p
            out.append(price * (1.0 + p / 100.0) if side == 'BUY'
                       else price * (1.0 - p / 100.0))
        return tuple(out)

    def calculate_position_size(
        self,
        balance:       float,
        quality_score: float,
        regime:        RegimeState,
        atr_pct:       float,   # already × 100, e.g. 2.5 means 2.5 %
    ) -> float:
        """
        Returns a USDT position value for this trade.

        Sizing logic
        ------------
        1. Base = BASE_POSITION_PCT × balance
        2. Scale by quality conviction: quality_score / 100
        3. Cap by regime.max_position_pct
        4. Halve in high-volatility conditions (atr_pct > 4 %)
        5. Clamp to [MIN, MAX] × balance
        """
        if balance <= 0:
            return 0.0

        base = balance * self.BASE_POSITION_PCT

        # Quality scaling: 55 points → 55 % of base; 100 points → 100 % of base
        quality_factor = max(0.0, min(quality_score / 100.0, 1.0))
        sized = base * quality_factor

        # Regime ceiling
        regime_cap = balance * max(regime.max_position_pct, self.MIN_POSITION_PCT)
        sized = min(sized, regime_cap)

        # Volatility discount: halve size when market is unusually wide
        if atr_pct > 4.0:
            sized *= 0.5

        # Hard clamp
        floor   = balance * self.MIN_POSITION_PCT
        ceiling = balance * self.MAX_POSITION_PCT
        return round(max(floor, min(sized, ceiling)), 2)

    def calculate_stops(
        self,
        price:      float,
        side:       str,    # 'BUY' | 'SELL'
        atr:        float,
        support:    float = 0.0,   # invalidation level for a LONG / downside target for a SHORT
        resistance: float = 0.0,   # invalidation level for a SHORT / upside target for a LONG
        sl_cap_atr: float   = 0.0,   # v42: SL-cap override in ATR (0 -> ATR_SL_MULTIPLIER)
        sl_override: float  = 0.0,   # v83: TraderGate's structural stop — used verbatim
        tp_override: float  = 0.0,   # v85: TraderGate's structural TARGET — used verbatim
        **_kwargs,      # absorbs legacy keyword args for backward compatibility
    ) -> Dict[str, float]:
        """
        ATR + Structure HYBRID TP/SL.

        Stop Loss (hybrid)
        ------------------
        Anchored just beyond the invalidation level (support for a LONG,
        resistance for a SHORT) + STRUCT_SL_BUFFER_ATR wick buffer, then the risk
        leg is clamped to [SL_FLOOR_ATR, ATR_SL_MULTIPLIER]×ATR.  Falls back to a
        pure ATR_SL_MULTIPLIER×ATR stop when the level is missing/degenerate.

        The wick buffer is a FLOOR, not an average: the stop always finishes at
        least STRUCT_SL_BUFFER_ATR×ATR beyond the level, even when the clamp
        would have pulled it back inside, and even when it arrived as an
        `sl_override`.  A stop resting on the level is taken out both when price
        overshoots and reverses and when it breaks through and retests — the two
        most common things a level does.

        Take Profit ladder (R = risk leg; Range = resistance−support)
        ------------------------------------------------------------
          TP1  = 1.0R                            (1:1 first bank — v82, was 0.7R)
          TP2  = 2R                              (1:2, first significant objective)
          TP3  = the major structural level      (resistance for a LONG / support
                 for a SHORT) — the HTF target / liquidity pool
          TP4  = 1.618 fib extension of Range     (measured move / extended trend)
          TP5  = 2.618 fib extension (display)    — actually a TRAILING exit; the
                 runner rides the trailing stop, this is just the anchor
        Levels are forced strictly monotonic (≥0.3R apart).

        Risk/Reward Validation
        ----------------------
        Reward is measured to TP3 (the structural target) — so a setup whose real
        target is too close for the risk is rejected (valid_rr = False).

        Returns
        -------
        dict with: sl, tp1–tp5, risk, reward, risk_reward, valid_rr, atr
        """
        if price <= 0 or atr <= 0:
            return {
                'sl': 0.0, 'tp1': 0.0, 'tp2': 0.0, 'tp3': 0.0,
                'tp4': 0.0, 'tp5': 0.0,
                'risk': 0.0, 'reward': 0.0, 'risk_reward': 0.0,
                'valid_rr': False, 'atr': atr,
            }

        support    = float(support or 0.0)
        resistance = float(resistance or 0.0)
        buf   = self.STRUCT_SL_BUFFER_ATR * atr
        floor = self.SL_FLOOR_ATR * atr
        cap   = (sl_cap_atr if sl_cap_atr and sl_cap_atr > 0 else self.ATR_SL_MULTIPLIER) * atr

        if side == 'BUY':
            if sl_override and 0 < sl_override < price:
                # v83: TraderGate already placed this stop beyond the level the
                # setup leans on and bounded it to [MIN_STOP_ATR, MAX_STOP_ATR].
                # Take it verbatim — re-deriving it here would mean the R:R the
                # gate approved the trade on is not the R:R the trade actually
                # has, which is the one number the payoff stage must not lie about.
                sl = sl_override
                # ...with one exception: the gate leans on the structure IT was
                # handed, which is not always the rolling S/R published here. A
                # stop parked at or just above the support is the one place the
                # market reliably collects it — price undercuts the level by a
                # wick and turns, or breaks it and retests from below. Clear it.
                # Bounded to levels actually in the way (at/inside the stop, or
                # within one buffer of it), so a distant support can never blow
                # out the stop the gate priced the trade on.
                if 0 < support < price and sl >= support - buf:
                    sl = min(sl, support - buf)
                risk = price - sl
            else:
                # Hybrid SL: just below support + buffer, clamped to [floor, cap].
                risk = ((price - support) + buf) if (0 < support < price) else cap
                risk = max(floor, min(risk, cap))
                sl   = price - risk
                if 0 < support < price:
                    sl = min(sl, support - buf)
            # Rungs are percentages of entry — see TP_LADDER_PCT.
            # The objective still governs: it CAPS the ladder (see _ladder) and
            # it remains the number R:R is quoted against, so the gate's payoff
            # test and the published figure keep meaning the same thing. Banking
            # earlier than the objective does not invalidate the approval — it
            # just takes the money on the way.
            _tgt = tp_override if tp_override > price else 0.0
            tp1, tp2, tp3, tp4, tp5 = self._ladder(price, 'BUY', _tgt)
            reward = _tgt - price if _tgt else (
                (resistance - price) if resistance > price else 3.5 * risk)
        else:  # SELL / SHORT
            if sl_override and sl_override > price:
                sl = sl_override            # v83 — see the BUY branch above
                if resistance > price and sl <= resistance + buf:
                    sl = max(sl, resistance + buf)
                risk = sl - price
            else:
                risk = ((resistance - price) + buf) if (resistance > price) else cap
                risk = max(floor, min(risk, cap))
                sl   = price + risk
                if resistance > price:
                    sl = max(sl, resistance + buf)
            _tgt = tp_override if 0 < tp_override < price else 0.0   # see BUY branch
            tp1, tp2, tp3, tp4, tp5 = self._ladder(price, 'SELL', _tgt)
            reward = price - _tgt if _tgt else (
                (price - support) if 0 < support < price else 3.5 * risk)

        rr       = round(reward / risk, 3) if risk > 0 else 0.0
        valid_rr = rr >= self.MIN_RISK_REWARD

        return {
            'sl':          round(sl,  8),
            'tp1':         round(tp1, 8),
            'tp2':         round(tp2, 8),
            'tp3':         round(tp3, 8),
            'tp4':         round(tp4, 8),
            'tp5':         round(tp5, 8),
            'risk':        round(risk,   8),
            'reward':      round(reward, 8),
            'risk_reward': rr,
            'valid_rr':    valid_rr,
            'atr':         round(atr, 8),
        }
