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

# v87 risk-budget band. Read from TraderGate rather than restated here: the gate
# places the stop and this module consumes it, so a second copy of the numbers is
# a copy that can drift out of step — which is exactly how the shadow book ended
# up scoring against a give-back leash production had already abandoned.
#
# The MODULE is imported, not the two names. `from ... import MIN_STOP_PCT` binds
# a VALUE COPY at import time, so rebinding trader_gate.MIN_STOP_PCT afterwards
# would leave this module reading the old number — the same trap live_engine.py
# documents at its own import block. Reading the attribute per call keeps one
# switch rather than two.
from src.trading import trader_gate as _trader_gate

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
    # v86: TP1 1.0 / TP2 2.0. The upper rungs keep the old spacing (+1.0, +0.5,
    # +1.0, +0.5) so only the first two moved.
    #
    # This is a payoff-ratio change, not a profit-taking one, and it is paid
    # for in win rate. Measured over 14,280 paths on a 1.3% stop: 71.5% of
    # trades reach +0.5% before stopping out, 56.2% reach +1.0%. So the first
    # rung is hit on roughly four trades in five as often. What it buys is the
    # size of the win — median booked result goes from +0.40% to +0.90% against
    # losses that run 1.1-1.9%, which is the ratio that made a 75% win rate
    # lose money.
    #
    # v87: TP1 1.0 -> 1.5, the ladder scaled with it. TP1 was the ONLY reachable
    # win: with TP_GIVEBACK_MAX_FRAC at zero the whole remainder books at the
    # first rung on any tick back through it, so every winner on the public book
    # closed at +0.90% (1.00% rung less the 0.10% round trip) against losses of
    # 1.1-1.6%. Measured on 19,140 trades over 30k real 1h bars, drift calibrated
    # so the current geometry reproduces the observed 63.6% live win rate:
    #
    #     TP1 1.0%  exp +0.222%/trade  WR 63.6%  avg win 0.90%  avg loss -1.34%
    #     TP1 1.25% exp +0.241%/trade  WR 58.9%  avg win 1.34%  avg loss -1.34%
    #     TP1 1.5%  exp +0.265%/trade  WR 55.4%  avg win 1.55%  avg loss -1.34%
    #     TP1 2.0%  exp +0.280%/trade  WR 50.2%  avg win 1.88%  avg loss -1.34%
    #
    # KNOWN CONDITIONAL, accepted by the user on the measured menu: this is a bet
    # that the live edge is real. The same sweep at ZERO drift ranks the old
    # ladder first (TP1 1.0% +0.034 vs 1.5% -0.006), because with no edge a
    # nearer rung simply banks more often. The edge over the geometric break-even
    # win rate held at +8.6 to +9.6pp across every rung tested, which is what
    # makes the ranking hold — but it rests on a 22-trade sample. If the live win
    # rate settles below ~52% the arithmetic flips back; watch it.
    #
    # Two things NOT changed, both measured worse at every setting tested:
    # widening TP_GIVEBACK_MAX_FRAC (0.35 -> exp +0.152, 1.00 -> +0.125) and
    # tightening the stop (1.1 ATR -> +0.143, 0.7 ATR -> +0.078; the fixed
    # round-trip cost is a larger share of a smaller risk leg).
    TP_LADDER_PCT = (1.5, 3.0, 3.75, 5.25, 6.0)   # TP1 … TP5, percent of entry

    # ── SHADOW: volatility-scaled TP1 (NOT APPLIED) ──────────────────────────
    # The ladder above is a fixed percent of entry, which makes TP1 a constant
    # distance and therefore a WILDLY varying difficulty. Measured across the
    # 2026-08-14 book, TP1 at 1.5% ranged from 1.23 ATR (CRV, ATR 1.219%) to
    # 5.85 ATR (BNB, ATR 0.256%) — a 4.8x spread in how hard the same target is
    # to reach. The observable consequence: every win in that book paid exactly
    # +1.4000%, because the one rung anything reaches is the first one, so TP1
    # behaves as a ceiling rather than an objective.
    #
    # K is set so the MEDIAN token's TP1 is unchanged (median fleet ATR% 0.678,
    # 1.5 / 0.678 = 2.21). This is therefore not a disguised loosening — it
    # redistributes difficulty rather than reducing it, compressing the spread
    # from 4.8x to ~1.9x. The bounds stop both failure modes: without the floor
    # a very quiet token gets a target inside the spread, without the cap a very
    # volatile one gets the 2.3-3.5% objectives the percent ladder was
    # introduced to eliminate.
    #
    # Recorded on every trade and applied to none. The question it settles from
    # the next ~25 trades: for trades whose fixed TP1 was never reached, did MFE
    # reach the hybrid rung?
    # Item 3 (2026-08-15): HELD OFF, deliberately, and shadowed instead.
    #
    # The gate enters within AT_LEVEL_ATR (0.35 ATR ~ 0.24% at median ATR) of
    # ITS level, while the affordability bar is ~0.96% at the same ATR. A
    # correctly-matched setup therefore ALWAYS fits, which means this gate never
    # fires on one — every refusal it produces comes from risk.py reaching for a
    # `result['support']` the gate never leaned on.
    #
    # So it is not an affordability problem with a gate as its fix. It is two
    # components disagreeing about which level the trade is about, and a guard
    # that refuses the trade rather than resolving the disagreement. That is the
    # seventh instance of this week's pattern — TRACK_RECORD_PATH meaning two
    # files, (default) vs default, band_capped meaning two things on two paths,
    # and now "the level" meaning one thing to the entry logic and another to
    # the stop logic.
    #
    # The real fix is making the gate and risk.py agree on one level. Genuinely
    # unaffordable setups may still exist, but they are invisible until the
    # mismatch is gone. Shadowed as would_refuse_unaffordable so the next ~25
    # signals show the refusal rate AND, via stop_source/support_present,
    # whether each case is a mismatch or a real one.
    REFUSE_UNAFFORDABLE_INVALIDATION = False

    TP1_HYBRID_K       = 2.21   # x ATR
    TP1_HYBRID_MIN_PCT = 0.90
    TP1_HYBRID_MAX_PCT = 2.20

    @staticmethod
    def budget_cap_pct() -> float:
        """The percent-of-entry ceiling the budget band enforces, for messages."""
        return float(getattr(_trader_gate, 'MAX_STOP_PCT', 0.0) or 0.0)

    @classmethod
    def tp1_hybrid(cls, price: float, atr: float) -> tuple:
        """Shadow only. Returns (pct_of_entry, distance_in_ATR)."""
        if price <= 0 or atr <= 0:
            return 0.0, 0.0
        atr_pct = atr / price * 100.0
        pct = min(max(cls.TP1_HYBRID_K * atr_pct,
                      cls.TP1_HYBRID_MIN_PCT), cls.TP1_HYBRID_MAX_PCT)
        return round(pct, 6), round(pct / atr_pct, 6)
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
    # The ATR floor above was added so a narrow rung could not produce a
    # noise-width stop. It did that, and it also quietly switched the TP1 rung
    # OFF for most of the fleet: entry→TP1 is 0.5% of price, so 0.50 ATR
    # exceeds the whole span for any token with ATR% above ~1%, and a leash
    # wider than its rung was skipped. IMX/USDT 2026-08-08 is the bill —
    # short 0.1124, TP1 0.11184 tagged at +0.50%, no ratchet, price walked back
    # to break-even and it booked +0.02%.
    #
    # So the floor is a floor, not a licence to exceed the rung. A banked rung
    # is banked: once TP1 is tagged the runner may hand back at most
    # TP_GIVEBACK_MAX_FRAC of it and is then closed AT that level, which is
    # always inside the rung and therefore always better than the break-even
    # stop sitting at entry. On the IMX geometry that books ~+0.40% instead of
    # +0.02%. The cost is a lower TP3+ hit rate, which is the same trade struck
    # above — this just stops the trade being silently voided by the floor.
    # The three interact as: clamp the ATR floor between a small proportional
    # buffer and a hard fraction of the rung. On the current 0.5% first rung the
    # CAP is what binds for essentially every token, which is the intent — a
    # banked rung is given back by about a fifth and no more. The other two stay
    # live for wider rungs and very low-ATR tokens, and matter again if the
    # ladder is ever re-spaced.
    # v86: the cap is ZERO — a banked rung is handed back at the rung itself,
    # not below it. The fifth-of-a-rung leash was measurable in the live book:
    # against a 0.5% first rung, wins clustered at +0.31-0.33% while losses ran
    # -1.13% to -1.85%. A 75% win rate lost money on that geometry.
    #
    # With the cap at zero the other two constants no longer bind, and they are
    # kept rather than deleted because they are the dials that come back if the
    # re-cross proves too tight — raise MAX_FRAC first, it is the one that
    # decides how much of a rung may be returned.
    # v88 (2026-08-15): THE CAP IS NO LONGER ZERO. Raised 0.00 -> 0.35.
    #
    # A zero cap collapses the leash to nothing, which puts the protective level
    # exactly ON the rung — so the first tick back through TP1 books the whole
    # remainder at TP1. The ladder above TP1 is then unreachable BY CONSTRUCTION,
    # and every winner pays the same number. Measured on the live book:
    #
    #     TIA/USDT  exit 0.3046605  == take_profit_1 0.3046605   +1.3997%
    #     CRV/USDT                                               +1.4023%
    #     ATOM/USDT                                              +1.4000%
    #
    # Three tokens, three identical results, all exit_reason=TP_GIVEBACK. That is
    # a mechanism, not a market. The v86 note above is the same finding one
    # iteration earlier ("wins clustered at +0.31-0.33%") and the v87 note in
    # trader_gate records it again ("a ZERO leash made TP1 the only reachable
    # win, every win exactly +0.90%"). Zeroing the cap has now produced this
    # pathology three times; the correct reading is that the cap must not be
    # zero, not that the number needs another tune.
    #
    # THE MEASUREMENT DISAGREES, AND THAT IS UNDERSTOOD. The harness ranked
    # widening this worse (0.35 -> exp +0.152, 1.00 -> +0.125) against a zero
    # cap. But that sweep was run in a world where TP1 was the only rung ever
    # reached, so it compared "always bank TP1" with "sometimes bank TP1,
    # sometimes give it back" — it never priced the outcome this change exists
    # to allow, which is the remainder RUNNING to TP2+. mfe_pct now records how
    # far past TP1 price actually travelled, so the next ~25 trades settle it on
    # evidence rather than on a sweep that could not see the alternative.
    #
    # Restore the old behaviour with MAX_FRAC = 0.00.
    TP_GIVEBACK_PCT      = 0.10  # proportional buffer, live again
    TP_GIVEBACK_MIN_ATR  = 0.50  # ATR floor, live again
    TP_GIVEBACK_MAX_FRAC = 0.35  # a banked rung may be given back by at most a third
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

    @staticmethod
    def _budget_band(price: float, risk: float) -> float:
        """Clamp a stop distance into the v87 risk-budget band.

        The band is a percent of price, not a multiple of ATR, because that is
        how it was specified. Both bounds must be positive for it to apply, so
        setting either to 0 in TraderGate disables it here too and the stop
        returns to being purely structural — there is one switch, not two.

        Returns the risk unchanged when the band is off or the inputs are
        degenerate; callers re-derive `sl` from the value they get back.
        """
        if price <= 0 or risk <= 0:
            return risk
        lo_pct = float(getattr(_trader_gate, 'MIN_STOP_PCT', 0.0) or 0.0)
        hi_pct = float(getattr(_trader_gate, 'MAX_STOP_PCT', 0.0) or 0.0)
        if not (lo_pct > 0 and hi_pct > 0):
            return risk
        return min(max(risk, price * lo_pct / 100.0), price * hi_pct / 100.0)

    @classmethod
    def _ladder(cls, price: float, side: str, target: float = 0.0) -> tuple:
        """The five take-profit rungs, as percentages of entry.

        Returns (tp1..tp5) ordered away from `price` in the trade's direction.
        Percentages come from TP_LADDER_PCT and are held strictly monotonic by
        TP_MIN_GAP_PCT, so a mis-edited table cannot put two rungs at the same
        price — that would make one partial close a no-op and silently change
        the size of the runner.

        `target` is the STRUCTURAL objective the gate cleared the trade on
        (plan.target). No rung may sit beyond it — the payoff stage rejects a
        setup whose objective is too far to be real, so a rung past that
        objective would re-invent the thing the floor exists to reject and would
        advertise a target the gate never vouched for.

        It used to enforce that by scaling ALL FIVE rungs onto the objective,
        which quietly moved the one rung that must not move. TP1 exists to bank
        something before a reversal can take the trade back to entry; at 0.5%
        that is its whole job. A 1.6% objective scaled the ladder by 0.47 and
        put TP1 at 0.23%, so the engine was banking a fifth of a percent and
        calling it the first rung. Measured over 2026-08-08/09: 11 of 13 closed
        signals ran a compressed ladder, TP1 landing between 0.23% and 0.48%.

        So the cap truncates instead of scaling. Every rung that fits keeps its
        published percentage, and only the rungs that do NOT fit are distributed
        between the last one that does and the objective. TP1 is 0.5% and TP2 is
        1.5% on any trade whose objective reaches them — which was all 13.
        """
        pcts = [float(p) for p in cls.TP_LADDER_PCT]
        if target > 0 and price > 0:
            tgt_pct = abs(target - price) / price * 100.0
            if 0 < tgt_pct < pcts[-1]:
                fits = [p for p in pcts if p < tgt_pct]
                # Hold only as many rungs fixed as leave room to space the rest
                # at TP_MIN_GAP_PCT. Without this the spare rungs are packed
                # tighter than the gap, and the monotonic clamp below then walks
                # the top rung PAST the objective — the very thing this cap is
                # here to prevent (see test_plan_target_handoff).
                while fits and (tgt_pct - fits[-1]) < (len(pcts) - len(fits)) * cls.TP_MIN_GAP_PCT:
                    fits.pop()
                if fits:
                    # what fits keeps its published percentage; the remainder is
                    # spread evenly from the last fitting rung to the objective
                    spare = len(pcts) - len(fits)
                    lo = fits[-1]
                    step = (tgt_pct - lo) / spare
                    pcts = fits + [lo + step * (i + 1) for i in range(spare)]
                else:
                    # The objective does not reach even TP1. Scaling the whole
                    # ladder onto it — what this did — is the same defect the
                    # truncating cap above was written to end, just relocated: a
                    # 1.2% objective against a 1.5% TP1 scales by 0.2 and puts
                    # TP1 at 0.30%, which is once again banking a fifth of a
                    # percent and calling it the first rung. Moving TP1 out to
                    # 1.5% widens the band of objectives that land in this branch,
                    # so it can no longer be the loose end it was at 0.5%.
                    #
                    # Anchor TP1 as far out as the objective can carry it —
                    # leaving exactly enough room to space the rest at the
                    # minimum gap — and spread the remaining rungs to the
                    # objective. On that 1.2% objective TP1 is 1.00% rather than
                    # 0.30%, and no rung sits beyond the target the gate vouched
                    # for, which is the invariant this cap exists to hold.
                    _n   = len(pcts)
                    _gap = cls.TP_MIN_GAP_PCT
                    _lo  = max(_gap, min(pcts[0], tgt_pct - (_n - 1) * _gap))
                    _step = (tgt_pct - _lo) / (_n - 1) if _n > 1 else 0.0
                    pcts = [_lo + _step * i for i in range(_n)]

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
        gate_stop_source: str = '',  # provenance from TraderGate when overriding
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
        _shadow_structural = 0.0      # see the SHADOW block below
        _shadow_capped     = False
        # Which mechanism placed the shipped stop. Never left ambiguous: on the
        # override path the gate already decided, so its value is inherited
        # rather than guessed at.
        _stop_source       = 'unknown'
        # SIDE-AWARE on purpose. The support-clearing guard for a LONG consumes
        # `support`; `resistance` being present tells you nothing about whether
        # that guard had its input. An OR here would have recorded "support was
        # available" on exactly the trades where it was not — the same class of
        # ambiguity stop_source exists to remove.
        _support_seen      = bool((support if str(side).upper() == 'BUY' else resistance) or 0)                              and float(support if str(side).upper() == 'BUY' else resistance) > 0
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
                _stop_source = gate_stop_source or 'unknown'
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
                    _stop_source = 'structure'   # the level cleared it after all
                risk = price - sl
            else:
                # Hybrid SL: just below support + buffer, clamped to [floor, cap].
                risk = ((price - support) + buf) if (0 < support < price) else cap
                risk = max(floor, min(risk, cap))
                sl   = price - risk
                if 0 < support < price:
                    sl = min(sl, support - buf)
                # v87 budget band — on THIS path only. The override path above is
                # still taken verbatim: TraderGate bands its own stop at stage 3b
                # and prices r_net against it, so re-banding here would be a
                # second opinion on a number that has already been decided.
                #
                # CORRECTED 2026-08-14. This comment previously claimed "the
                # support-clearing exception is deliberately allowed to widen past
                # the band". IT IS NOT. The band runs AFTER the support-clearing
                # min() above and overrides it, so a stop that was just moved below
                # the level is pulled straight back inside it. TAO/USDT: structural
                # stop 192.9679 (below support 193.70) became 196.2156 (1.30% cap),
                # and price bottomed at 194.80 — through the shipped stop, never
                # reaching the structural one.
                #
                # The behaviour is NOT being changed here, because sizing partly
                # compensates (positions.py:208-211 rescales inversely with risk)
                # and the two available harness measurements disagree on whether a
                # wider stop is better per unit of risk. The shadow fields below
                # record what the structural stop would have been so the next ~25
                # trades can settle it. See docs/ENTRY_AND_STOP_ANALYSIS.md.
                # SHADOW (v88): record what the support-cleared stop WOULD have
                # been before the band overrides it. Observation only — the
                # banded stop below still ships and behaviour is unchanged.
                # Exists because the band-vs-structure question cannot be settled
                # from history: no archived record kept the level, so the only
                # way to price the decision is to instrument it going forward.
                _shadow_structural = sl
                _stop_source = 'structure' if (0 < support < price) else 'atr_floor'
                risk = self._budget_band(price, price - sl)
                sl   = price - risk
                _shadow_capped = abs(sl - _shadow_structural) > 1e-9
                if _shadow_capped:
                    _stop_source = 'risk_band'
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
                _shadow_structural = sl                      # SHADOW — see BUY branch
                _stop_source = 'structure' if resistance > price else 'atr_floor'
                risk = self._budget_band(price, sl - price)  # see the BUY branch
                sl   = price + risk
                _shadow_capped = abs(sl - _shadow_structural) > 1e-9
                if _shadow_capped:
                    _stop_source = 'risk_band'
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
            # SHADOW — observation only, never consumed by sizing or exits.
            # 0.0 / False means "the band was not in play on this path" (the
            # gate-override branch bands upstream), NOT "structure agreed".
            'structural_stop':     round(_shadow_structural, 8),
            'structural_stop_pct': (round(abs(price - _shadow_structural) / price * 100.0, 6)
                                    if (_shadow_structural and price > 0) else 0.0),
            'band_capped':         bool(_shadow_capped),
            # Explicit provenance. band_capped alone is ambiguous across paths —
            # False can mean "structure agreed" OR "the band was not in play
            # here" — and a comment explaining that difference is exactly the
            # pattern that let TRACK_RECORD_PATH mean two files.
            'stop_source':         _stop_source,
            'support_seen':        bool(_support_seen),
        }
