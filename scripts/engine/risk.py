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

    # RETIRED: the TP2 % cap was for the former wide TP2 (2.8×ATR); with the
    # compressed ladder TP2 is only 1.3×ATR (modest at any volatility), so the
    # cap only kinked the ladder in high vol and was removed from calculate_stops.
    TP2_MAX_PCT       = 1.5    # (unused) former cap on TP2 distance, % of entry
    TP2_MIN_TP1_RATIO = 1.4    # ordering guard: TP2 distance ≥ 1.4× TP1

    MIN_RISK_REWARD   = 2.0    # Reward / Risk using TP5 (full-trend target) as reward; below this is rejected

    TRAIL_MULTIPLIER  = 1.0    # trailing stop distance = ATR × this (widened to match wider SL)

    # ── Partial-close percentages (must sum to 1.0) ───────────────────────────
    # Fractions of the ORIGINAL allocation (v82 — see partial_close_trade; they
    # used to be applied to the shrinking remainder, so each rung silently
    # banked less than its nominal share).  Front-loaded onto the two
    # "significant objective" targets (TP2/TP3); the 20 % runner rides the trail.
    TP_CLOSE_PCTS = (0.15, 0.25, 0.25, 0.15, 0.20)  # TP1 … TP5

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
            # Structural strong target = the major resistance (else an R-multiple).
            # v85: when TraderGate supplied the objective it priced the trade on,
            # that objective IS the target — same reasoning as sl_override above.
            # The payoff stage rejects anything under MIN_NET_R measured to THIS
            # level, so deriving tp3 from a different structure set meant the R:R
            # the trade was approved on was not the R:R it was given.
            _tgt = tp_override if tp_override > price else 0.0
            tp3  = _tgt or (resistance if resistance > price else price + 3.5 * risk)
            rng  = (resistance - support) if (0 < support < resistance) else (tp3 - price)
            tp1  = price + self.TP1_MULTIPLIER * risk
            tp2  = price + 2.0 * risk
            tp4  = (support + 1.618 * rng) if support > 0 else price + 5.0 * risk
            tp5  = (support + 2.618 * rng) if support > 0 else price + 7.0 * risk
            # Force strictly ascending, ≥0.3R apart.
            tp2 = max(tp2, tp1 + 0.3 * risk)
            if _tgt:
                # The objective is FIXED — it is the level the payoff floor
                # cleared.  The gate approves setups from 1.6R gross, so the
                # fixed 2.0R rung routinely lands PAST the target; letting the
                # monotonic clamp push tp3 out to clear it would re-invent the
                # "target too far to be real" that stage 3 exists to reject.
                # Compress the banking rungs inside the objective instead.
                _span = tp3 - price
                if tp2 >= tp3:
                    tp2 = price + (2.0 / 3.0) * _span
                if tp1 >= tp2:
                    tp1 = price + (1.0 / 3.0) * _span
            else:
                tp3 = max(tp3, tp2 + 0.3 * risk)
            tp4 = max(tp4, tp3 + 0.3 * risk)
            tp5 = max(tp5, tp4 + 0.3 * risk)
            # RR to the REAL structural target (not the guard-extended tp3), so a
            # cramped setup whose resistance is too close is honestly rejected.
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
            _tgt = tp_override if 0 < tp_override < price else 0.0   # v85 — see BUY
            tp3  = _tgt or (support if 0 < support < price else price - 3.5 * risk)
            rng  = (resistance - support) if (0 < support < resistance) else (price - tp3)
            tp1  = price - self.TP1_MULTIPLIER * risk
            tp2  = price - 2.0 * risk
            tp4  = (resistance - 1.618 * rng) if resistance > 0 else price - 5.0 * risk
            tp5  = (resistance - 2.618 * rng) if resistance > 0 else price - 7.0 * risk
            # Force strictly descending, ≥0.3R apart.
            tp2 = min(tp2, tp1 - 0.3 * risk)
            if _tgt:
                _span = price - tp3          # v85 — see the BUY branch above
                if tp2 <= tp3:
                    tp2 = price - (2.0 / 3.0) * _span
                if tp1 <= tp2:
                    tp1 = price - (1.0 / 3.0) * _span
            else:
                tp3 = min(tp3, tp2 - 0.3 * risk)
            tp4 = min(tp4, tp3 - 0.3 * risk)
            tp5 = min(tp5, tp4 - 0.3 * risk)
            # RR to the REAL structural target (not the guard-extended tp3).
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
