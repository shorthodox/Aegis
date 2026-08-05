"""
trader_gate.py — the desk

Replaces the Guard A..T veto pile in `_process_symbol` with the checklist an
experienced discretionary trader actually runs.  The difference is structural,
not cosmetic:

    OLD   16 independent vetoes, each free to say "no" for its own reason, with
          later patches ("trust model", `_atr_relaxed`, `_conv_relaxed`,
          `_trust_warns`) selectively *disabling* earlier ones.  Whether a fire
          meant anything depended on which patch won.  Nothing in the chain ever
          asked what the trade WAS, only whether it was allowed.

    NEW   one ordered playbook.  Each stage answers a question a trader answers
          in the same order, and the output is a PLAN — side, setup type, entry,
          invalidation, target, expected payoff, size — or a documented refusal.
          No stage relaxes another; a stage that fails ends the evaluation.

The stages
----------
  0  Fitness      Is this market tradeable at all right now?  (drift, news,
                  liquidity, volatility shock)  — the only genuinely protective
                  vetoes from the old system, kept and consolidated.
  1  Setup        WHAT is the trade?  Classified from regime + location into
                  TREND_PULLBACK / BREAK_RETEST / RANGE_FADE /
                  EXHAUSTION_REVERSAL, or NONE.  The setup determines the SIDE;
                  structure leads and the model confirms (see below).
  2  Invalidation WHERE am I wrong?  The stop sits beyond the level the setup
                  leans on — not at a fixed ATR multiple.  A stop inside the
                  noise band, or wider than the payoff can carry, ends it here.
  3  Payoff       Does the trade PAY?  Net R:R to the first real structural
                  objective, after round-trip costs.  Below the floor, the trade
                  is refused however good the model score looks.  This is the
                  stage the old system never had: it sized conviction, never
                  payoff.
  4  Trigger      Is it time?  At the level with confirmation -> ENTER.  Within
                  reach -> WORK (a resting order AT the level, with a hard
                  invalidation and a bar-count expiry).  Already through the
                  level, or out of reach -> REJECT.  Nothing waits forever; this
                  is what replaces the open-ended PENDING queue.
  5  Allocation   HOW MUCH?  Correlation cluster, book depth, tide alignment and
                  setup class scale the risk.  Refuses the 9th expression of one
                  bet instead of sizing it like the first.

Why the side comes from the setup, not the model
------------------------------------------------
The measured failure this module exists to fix was a basket of eight alt SHORTs
opened inside one 55-minute window and stopped out together, with model
confidence on the losers spanning 17.9 to 100.0 — conviction had no relationship
to outcome.  Permission-by-conviction is what produced that.  Here the structure
picks the side and the model may only VETO it (`MODEL_OPPOSE_MARGIN`) or scale
it; `REQUIRE_MODEL_FIRE` restores the old model-first behaviour if needed.

Pure aggregation: no API calls, no engine imports, no I/O.  Everything it needs
arrives in the four input dicts, which makes every branch here directly testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ── Setup taxonomy ────────────────────────────────────────────────────────────
SETUP_TREND_PULLBACK      = 'TREND_PULLBACK'       # with the trend, entering on a retracement
SETUP_BREAK_RETEST        = 'BREAK_RETEST'         # broken level retested from the other side
SETUP_RANGE_FADE          = 'RANGE_FADE'           # range extreme, no trend to fight
SETUP_EXHAUSTION_REVERSAL = 'EXHAUSTION_REVERSAL'  # counter-trend, only at a stretched extreme
SETUP_NONE                = 'NONE'

# ── Plan actions ──────────────────────────────────────────────────────────────
ACTION_ENTER  = 'ENTER'    # take it now, at market
ACTION_WORK   = 'WORK'     # resting order at the level; expires
ACTION_REJECT = 'REJECT'   # no trade, with a reason

# ── Stage 1 · where each setup is allowed to exist ────────────────────────────
# range_position: 0.0 = at support, 1.0 = at resistance.
PULLBACK_RP_LONG   = 0.45   # a bull pullback must give back at least this much of the range
PULLBACK_RP_SHORT  = 0.55   # mirror for a bear rally
EXTREME_RP_LOW     = 0.20   # counter-trend reversals live only in the outer fifth
EXTREME_RP_HIGH    = 0.80
RANGE_EDGE_LOW     = 0.30   # a range fade needs the edge, not "the lower half"
RANGE_EDGE_HIGH    = 0.70
EXHAUSTION_RSI_HI  = 68.0   # fading a bull needs the move actually stretched
EXHAUSTION_RSI_LO  = 32.0

# ── Stage 1b · location vs the higher timeframe ───────────────────────────────
# Location says which side is ON OFFER; the weekly and daily say whether a side
# that fights its own location is PERMITTED at all.
#
# Every setup except BREAK_RETEST already agrees with its location — the fades
# sell highs and buy lows, the pullbacks buy dips in a bull and sell rallies in a
# bear. BREAK_RETEST is the one that deliberately does the opposite: it buys at
# the top of the range on the argument that broken resistance is now support.
# That argument is only good while the higher timeframe agrees. When it does
# not, the same picture is just a long into resistance underneath a bearish
# weekly — which is the trade this desk kept taking and losing.
#
# Measured case, ADA/USDT 2026-08-05: resistance_broken_recent with rp 0.85 and
# a VOLATILE_COMPRESSION label (so `bear` was False) returned BREAK_RETEST BUY.
# Weekly BEAR, daily BULL, RSI 62. The HTF check existed but was advisory and
# printed "1 of 2 opposes this long" while the long went on anyway.
HTF_MIN_BIAS = 0.5   # |macro_weekly| / |macro_daily| above this is a real lean,
                     # not noise; the fields are +1.0 above EMA50, -1.0 below
MIN_REGIME_CONF    = 0.45   # below this the regime label is a guess; treat as rangebound

TRENDING_BULL = 'TRENDING_BULL'
TRENDING_BEAR = 'TRENDING_BEAR'
_RANGE_REGIMES = {'RANGING', 'ACCUMULATION', 'DISTRIBUTION',
                  'VOLATILE_COMPRESSION', 'COMPRESSION'}
_DEAD_REGIMES  = {'LIQUIDITY_TRAP'}

# ── Stage 2 · invalidation geometry ───────────────────────────────────────────
STOP_BUFFER_ATR = 0.55   # the stop sits this far BEYOND the level — outside the wick noise
                         # that repeatedly took out stops parked exactly on it
MIN_STOP_ATR    = 1.50   # a stop nearer than this to entry is inside one bar's noise; the
                         # 8-short basket died on ~1.1% stops in ~1% ATR tape
                         # v85: was 0.90.  The floor is really a COST decision, not just a
                         # noise one: the round trip is a fixed % of price, so it lands as
                         # ROUND_TRIP_COST_PCT / risk_pct in R.  At 0.90 ATR on 1h tape
                         # (median ATR 0.91% of price) every trade paid 0.147R in fees
                         # before the market moved.  Measured over 28,650 zero-edge 1h
                         # entries the live ladder returned -0.174R/trade at 0.90 and
                         # -0.110R at 1.50 — the floor was the largest single leak in the
                         # exit geometry.  It buys fewer fires: clearing MIN_NET_R now
                         # needs an objective ~2.7 ATR out rather than ~1.7.
MAX_STOP_ATR    = 3.00   # beyond this the payoff maths can never clear the floor

# ── Stage 3 · payoff ──────────────────────────────────────────────────────────
MIN_NET_R          = 1.60  # net of costs, to the FIRST real objective — not to a fib fantasy
MIN_TARGET_ATR     = 1.50  # a level closer than this is noise, not an objective
ROUND_TRIP_COST_PCT = 0.10 # 0.04% taker + 0.01% slippage per side; keep in step with
                           # VirtualWallet.round_trip_cost_pct() and retrain_model.py

# ── Stage 4 · trigger ─────────────────────────────────────────────────────────
AT_LEVEL_ATR     = 0.35   # price is "at" the level within this many ATR
REACH_ATR        = 2.50   # further than this and it is not a trade yet — drop it, do not queue
WORK_EXPIRY_BARS = 8      # a resting order the market ignores for 8 bars is a dead thesis
MODEL_OPPOSE_MARGIN = 0.12  # model may veto the structure only when it leans this hard the
                            # other way (raw p_buy/p_sell); a neutral model does not block

# ── Stage 5 · allocation ──────────────────────────────────────────────────────
SETUP_RISK_WEIGHT: Dict[str, float] = {
    SETUP_TREND_PULLBACK:      1.00,  # measured +0.069R/trade — the paid setup
    SETUP_BREAK_RETEST:        0.85,
    SETUP_RANGE_FADE:          0.70,
    SETUP_EXHAUSTION_REVERSAL: 0.50,  # measured -0.064R fleet-wide — smallest size it can have
}
COUNTER_TIDE_FACTOR = 0.50   # against the BTC tide is half the trade it looks like
STRONG_TIDE         = 0.65   # tide this strong refuses counter-tide trades outright
CLUSTER_SECOND_FACTOR = 0.60 # the second expression of one cluster thesis is not a fresh bet
MIN_SIZE_FACTOR     = 0.25   # below this the trade is not worth its execution cost

# ── Stage 0 · fitness ─────────────────────────────────────────────────────────
DEAD_ATR_PCT     = 0.15   # genuinely flatlined tape
EXTREME_ATR_PCT  = 6.00   # absolute volatility shock
EXTREME_ATR_MULT = 3.00   # ...or this multiple of the token's own normal ATR
SPREAD_MAX_PCT   = 0.15   # book this wide is not a market you can pay to enter
LOW_VOL_Z        = -1.60

# Toggle: when True the model must independently fire, restoring model-first v80
# permissioning on top of the playbook.  Default False — structure leads.
REQUIRE_MODEL_FIRE = False


@dataclass
class TradePlan:
    """The desk's output. A plan, or a documented refusal."""
    action:  str = ACTION_REJECT
    side:    str = 'FLAT'                 # 'BUY' | 'SELL' | 'FLAT'
    setup:   str = SETUP_NONE

    entry:        float = 0.0
    stop:         float = 0.0
    target:       float = 0.0             # first REAL structural objective
    level:        float = 0.0             # the level the thesis leans on
    invalidation: float = 0.0             # a WORK order beyond this price is dead

    risk_atr:     float = 0.0             # stop distance in ATR
    r_gross:      float = 0.0
    r_net:        float = 0.0             # after round-trip costs — the number that decides
    size_factor:  float = 0.0             # 0..1 multiplier on the normal risk allocation
    expiry_bars:  int   = 0               # WORK only

    stage:        str = ''                # stage that produced the verdict
    reason:       str = ''                # one-line human summary
    notes:        List[str] = field(default_factory=list)   # full audit trail

    @property
    def fired(self) -> bool:
        return self.action == ACTION_ENTER

    @property
    def working(self) -> bool:
        return self.action == ACTION_WORK

    def as_dict(self) -> Dict[str, Any]:
        return {
            'action': self.action, 'side': self.side, 'setup': self.setup,
            'entry': self.entry, 'stop': self.stop, 'target': self.target,
            'level': self.level, 'invalidation': self.invalidation,
            'risk_atr': round(self.risk_atr, 2),
            'r_gross': round(self.r_gross, 2), 'r_net': round(self.r_net, 2),
            'size_factor': round(self.size_factor, 2),
            'expiry_bars': self.expiry_bars,
            'stage': self.stage, 'reason': self.reason, 'notes': list(self.notes),
        }


def _f(d: Any, k: str, default: float = 0.0) -> float:
    """Float accessor tolerant of missing keys, None, strings and NaN."""
    try:
        v = float((d or {}).get(k, default))
        return default if v != v else v
    except (TypeError, ValueError, AttributeError):
        return default


def _reject(stage: str, reason: str, notes: List[str],
            side: str = 'FLAT', setup: str = SETUP_NONE) -> TradePlan:
    return TradePlan(action=ACTION_REJECT, side=side, setup=setup,
                     stage=stage, reason=reason, notes=notes + [f'{stage}: {reason}'])


class TraderGate:
    """The staged playbook. `evaluate` is the whole public surface."""

    # ── Stage 0 ───────────────────────────────────────────────────────────────
    @staticmethod
    def _fitness(result: Dict[str, Any], regime_name: str,
                 market: Dict[str, Any]) -> Optional[str]:
        """Is this market tradeable right now?  Returns a refusal reason or None.

        These are the four vetoes from the old system that were genuinely
        protective rather than doctrinal.  Everything else that used to veto
        here now shapes the plan downstream instead of killing it.
        """
        if market.get('drift_blocked') or str(market.get('drift_severity', '')).upper() == 'CRITICAL':
            return 'model drift is critical on this symbol'
        if market.get('news_locked'):
            return f"scheduled macro event ({market.get('news_label') or 'news lock'})"
        if regime_name in _DEAD_REGIMES:
            return f'{regime_name} — no reliable structure to trade against'

        atr_pct = _f(result, 'atr_pct', 1.5)
        if atr_pct < DEAD_ATR_PCT:
            return f'flatlined tape (ATR {atr_pct:.2f}%)'
        if atr_pct >= EXTREME_ATR_PCT:
            return f'volatility shock (ATR {atr_pct:.2f}%)'
        atr_norm = _f(market, 'atr_normal_pct', 0.0)
        if atr_norm > 0 and atr_pct >= EXTREME_ATR_MULT * atr_norm:
            return (f'volatility shock (ATR {atr_pct:.2f}% is '
                    f'{atr_pct / atr_norm:.1f}x its normal {atr_norm:.2f}%)')

        spread = _f(market, 'spread_pct', 0.0)
        if spread > SPREAD_MAX_PCT:
            return f'book spread {spread:.2f}% — cannot pay to get in'
        if _f(result, 'volume_zscore', 0.0) <= LOW_VOL_Z and _f(result, 'relative_volume', 1.0) < 0.6:
            return 'volume has collapsed — no participation'
        return None

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    @classmethod
    def _classify(cls, result: Dict[str, Any], regime_name: str,
                  regime_conf: float) -> Tuple[str, str, str]:
        """WHAT is this trade?  Returns (setup, side, rationale).

        The setup decides the side.  A trend is only a trend when the regime
        detector is actually confident about it; below MIN_REGIME_CONF the label
        is noise and the market is treated as rangebound, which is the honest
        reading and stops "TRENDING_BULL at 0.31 confidence" from authorising
        continuation trades.
        """
        rp  = _f(result, 'range_position', 0.5)
        rsi = _f(result, 'rsi', _f(result, 'rsi_14', 50.0))

        trending = regime_name in (TRENDING_BULL, TRENDING_BEAR) and regime_conf >= MIN_REGIME_CONF
        bull = trending and regime_name == TRENDING_BULL
        bear = trending and regime_name == TRENDING_BEAR

        # --- BREAK_RETEST · highest quality when it is genuinely present -------
        # A level broke recently and price has come back to it.  Polarity flip:
        # broken resistance is support, broken support is resistance.  It ranks
        # first because it is the one setup where the level has just PROVEN
        # itself, and it is valid in any regime.
        if bool(result.get('resistance_broken_recent')) and rp >= RANGE_EDGE_HIGH and not bear:
            return (SETUP_BREAK_RETEST, 'BUY',
                    'resistance broke and price returned to it — broken resistance is support')
        if bool(result.get('support_broken_recent')) and rp <= RANGE_EDGE_LOW and not bull:
            return (SETUP_BREAK_RETEST, 'SELL',
                    'support broke and price returned to it — broken support is resistance')

        # --- Trending market ---------------------------------------------------
        if bull:
            if rp <= PULLBACK_RP_LONG:
                return (SETUP_TREND_PULLBACK, 'BUY',
                        f'uptrend pulled back into support (rp {rp:.2f}) — buying the dip in a bull')
            if rp >= EXTREME_RP_HIGH:
                if rsi >= EXHAUSTION_RSI_HI:
                    return (SETUP_EXHAUSTION_REVERSAL, 'SELL',
                            f'uptrend stretched at the top of its range (rp {rp:.2f}, RSI {rsi:.0f})')
                # At the highs but not stretched: an uptrend at its own highs is
                # a trend working, not a reversal. Shorting it needs exhaustion.
                return (SETUP_NONE, 'FLAT',
                        f'uptrend at its highs but not stretched (rp {rp:.2f}, RSI {rsi:.0f} '
                        f'< {EXHAUSTION_RSI_HI:.0f}) — a bull at the highs is a trend working, '
                        f'not a top')
            return (SETUP_NONE, 'FLAT',
                    f'uptrend mid-range (rp {rp:.2f}) — no edge here, chasing or fading blind')

        if bear:
            if rp >= PULLBACK_RP_SHORT:
                return (SETUP_TREND_PULLBACK, 'SELL',
                        f'downtrend rallied into resistance (rp {rp:.2f}) — selling the bounce in a bear')
            if rp <= EXTREME_RP_LOW:
                if rsi <= EXHAUSTION_RSI_LO:
                    return (SETUP_EXHAUSTION_REVERSAL, 'BUY',
                            f'downtrend stretched at the bottom of its range (rp {rp:.2f}, RSI {rsi:.0f})')
                return (SETUP_NONE, 'FLAT',
                        f'downtrend at its lows but not stretched (rp {rp:.2f}, RSI {rsi:.0f} '
                        f'> {EXHAUSTION_RSI_LO:.0f}) — a bear at the lows is a trend working, '
                        f'not a bottom')
            return (SETUP_NONE, 'FLAT',
                    f'downtrend mid-range (rp {rp:.2f}) — no edge here, chasing or fading blind')

        # --- Rangebound (incl. a low-confidence trend label) -------------------
        if regime_name in _RANGE_REGIMES or not trending:
            if rp <= RANGE_EDGE_LOW:
                return (SETUP_RANGE_FADE, 'BUY',
                        f'range low (rp {rp:.2f}) with no trend to fight — fading the edge')
            if rp >= RANGE_EDGE_HIGH:
                return (SETUP_RANGE_FADE, 'SELL',
                        f'range high (rp {rp:.2f}) with no trend to fight — fading the edge')
            return (SETUP_NONE, 'FLAT',
                    f'mid-range (rp {rp:.2f}) with no trend — nothing to lean on')

        return (SETUP_NONE, 'FLAT', f'{regime_name} offers no recognised setup')

    # ── Stage 1b ──────────────────────────────────────────────────────────────
    @staticmethod
    def _htf_opposes(side: str, result: Dict[str, Any]) -> str:
        """Which higher timeframes lean against `side`. '' when none do.

        Either timeframe is enough. A weekly that disagrees is not outvoted by a
        daily that agrees — the weekly is the slower, more expensive thing to be
        wrong about, and the daily flips inside it all the time.
        """
        w = _f(result, 'macro_weekly', 0.0)
        d = _f(result, 'macro_daily', 0.0)
        opposing = []
        if side == 'BUY':
            if w <= -HTF_MIN_BIAS:
                opposing.append('weekly')
            if d <= -HTF_MIN_BIAS:
                opposing.append('daily')
        else:
            if w >= HTF_MIN_BIAS:
                opposing.append('weekly')
            if d >= HTF_MIN_BIAS:
                opposing.append('daily')
        return ' and '.join(opposing)

    @classmethod
    def _counter_location_refusal(cls, side: str, rp: float, rsi: float,
                                  result: Dict[str, Any]) -> Optional[str]:
        """Refuse a side that fights BOTH its location and the higher timeframe.

        Only fires when price is at the structural extreme that opposes `side`.
        A long in the lower half or a short in the upper half is never touched
        here — those agree with their location and are somebody else's problem.
        """
        if side == 'BUY' and rp >= RANGE_EDGE_HIGH:
            opposed_by = cls._htf_opposes('BUY', result)
            if opposed_by:
                return (f'long into the top of the range (rp {rp:.2f}) while the '
                        f'{opposed_by} lean bearish — at resistance against the '
                        f'higher timeframe the only trade on offer is the short')
            if rsi >= EXHAUSTION_RSI_HI:
                return (f'long into the top of the range (rp {rp:.2f}) with RSI '
                        f'{rsi:.0f} already stretched — that is buying an '
                        f'overbought high, not a retest')
        if side == 'SELL' and rp <= RANGE_EDGE_LOW:
            opposed_by = cls._htf_opposes('SELL', result)
            if opposed_by:
                return (f'short into the bottom of the range (rp {rp:.2f}) while '
                        f'the {opposed_by} lean bullish — at support against the '
                        f'higher timeframe the only trade on offer is the long')
            if rsi <= EXHAUSTION_RSI_LO:
                return (f'short into the bottom of the range (rp {rp:.2f}) with RSI '
                        f'{rsi:.0f} already washed out — that is selling an '
                        f'oversold low, not a retest')
        return None

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    @staticmethod
    def _pick_level(side: str, setup: str, price: float, atr: float,
                    levels: Sequence[Tuple[float, int]],
                    result: Dict[str, Any]) -> float:
        """The level this setup leans on — the one that, if lost, means it failed.

        For a BUY that is the nearest structure AT OR BELOW price; for a SELL the
        nearest AT OR ABOVE.  "At or" matters: at the moment of a clean tag price
        sits a hair through the level, and demanding strictly-beyond discarded
        exactly the entries the setup was waiting for.
        """
        tol = AT_LEVEL_ATR * atr
        cands = [lv for lv, _t in (levels or []) if lv > 0]
        # The engine's own rolling S/R is a valid fallback when deep structure is sparse.
        sup = _f(result, 'support')
        res = _f(result, 'resistance')
        if sup > 0:
            cands.append(sup)
        if res > 0:
            cands.append(res)
        if not cands:
            return 0.0

        if side == 'BUY':
            below = [lv for lv in cands if lv <= price + tol]
            return max(below) if below else 0.0
        below_res = [lv for lv in cands if lv >= price - tol]
        return min(below_res) if below_res else 0.0

    @staticmethod
    def _clear_levels(side: str, price: float, stop: float, atr: float,
                      levels: Sequence[Tuple[float, int]],
                      result: Dict[str, Any]) -> float:
        """Push the stop past any level it has NOT cleared.

        `_pick_level` returns the NEAREST structure, and the noise-band floor
        below re-derives the stop from price alone — either can leave the stop
        parked a hair short of a second, heavier level.  That is the worst place
        on the chart to stand: price routinely overshoots a level by a wick and
        turns (the stop is collected on a move that proved the thesis right), or
        breaks it clean and flips it to the other side (the stop is collected on
        the retest).  Both outcomes are the trade being wrong about WHERE, not
        about WHAT.

        Only levels genuinely in the way are considered — on the stop's side of
        price, and at or inside the stop plus one buffer.  A level further out
        than that is not something this stop leans on, so it can never drag a
        bounded stop into the MAX_STOP_ATR reject.
        """
        buf = STOP_BUFFER_ATR * atr
        tol = AT_LEVEL_ATR * atr
        cands = [lv for lv, _t in (levels or []) if lv > 0]
        for _key in ('support', 'resistance'):
            lv = _f(result, _key)
            if lv > 0:
                cands.append(lv)

        if side == 'BUY':
            # Levels at or below price that sit at/above the stop, or within one
            # buffer beneath it.  Clearing the LOWEST clears every one above it.
            blocking = [lv for lv in cands if lv <= price + tol and lv >= stop - buf]
            return min(stop, min(blocking) - buf) if blocking else stop
        blocking = [lv for lv in cands if lv >= price - tol and lv <= stop + buf]
        return max(stop, max(blocking) + buf) if blocking else stop

    @staticmethod
    def _pick_target(side: str, price: float, atr: float,
                     levels: Sequence[Tuple[float, int]],
                     result: Dict[str, Any], risk: float = 0.0) -> float:
        """The first objective in the trade's direction that can actually PAY.

        "Real" = at least MIN_TARGET_ATR away.  A level 0.3 ATR ahead is noise
        the trade cannot be paid out of, and counting it as the target is how a
        losing setup passes an R:R test on paper.

        But "real" is not sufficient, and taking the NEAREST real level was
        closing the funnel completely. MIN_TARGET_ATR (1.5) and MIN_NET_R (1.6)
        are not consistent with each other: with a stop at the MIN_STOP_ATR
        floor of 1.5 ATR, clearing 1.6R net after the round trip needs an
        objective roughly 2.7 ATR out. Every level between 1.5 and 2.7 ATR is
        therefore selectable and then guaranteed to fail stage 3 — the gate
        would pick the 2 ATR level, price the trade at ~1.2R, reject it, and
        never look at the 3 ATR level sitting right behind it that would have
        paid. Measured over a 35,640-scenario sweep of the conditions this
        fleet sees, that rejected 45% of everything at payoff and contributed
        to a 0% fire rate.
        (v85 spotted the arithmetic — "clearing MIN_NET_R now needs an objective
        ~2.7 ATR out rather than ~1.7" — but only the floors were moved; the
        selector kept taking the nearest.)

        So: skip past the objectives that cannot pay and take the first one that
        can. It is still the NEAREST qualifying level, so this does not reach
        for a fib fantasy — it reaches for the first level the trade could
        actually be paid out of. When `risk` is unknown (0) the old
        distance-only behaviour is kept.
        """
        floor_dist = MIN_TARGET_ATR * atr
        cands = [lv for lv, _t in (levels or []) if lv > 0]
        sup = _f(result, 'support')
        res = _f(result, 'resistance')
        if sup > 0:
            cands.append(sup)
        if res > 0:
            cands.append(res)

        if side == 'BUY':
            ahead = sorted(lv for lv in cands if lv >= price + floor_dist)
        else:
            ahead = sorted((lv for lv in cands if lv <= price - floor_dist),
                           reverse=True)
        if not ahead:
            return 0.0
        if risk <= 0 or price <= 0:
            return ahead[0]

        risk_pct = risk / price * 100.0
        for lv in ahead:
            reward_pct = abs(lv - price) / price * 100.0
            r_net = ((reward_pct - ROUND_TRIP_COST_PCT) /
                     (risk_pct + ROUND_TRIP_COST_PCT)) if risk_pct > 0 else 0.0
            if r_net >= MIN_NET_R:
                return lv
        # Nothing ahead pays. Return the nearest anyway so stage 3 rejects with
        # the real numbers attached rather than "no structural objective".
        return ahead[0]

    # ── Stage 4 helper ────────────────────────────────────────────────────────
    @staticmethod
    def _confirmation(result: Dict[str, Any], side: str, setup: str,
                      confirm: Dict[str, Any]) -> Tuple[bool, str]:
        """Has something actually TURNED, or is price merely sitting at a level?

        Reversal-class setups demand a print — a rejection candle, a lower-
        timeframe momentum flip, or RSI curling back.  Continuation setups are
        cheaper to confirm because the trend is already the evidence; they need
        momentum to stop going against them, not to reverse outright.
        """
        # cdl_bull/bear_reversal use -1.0 for "data unavailable" and 0.0 for
        # "looked, found nothing" (feature_engine, and score_signal guards it
        # with `if _cdl_bull >= 0.0 and _cdl_bear >= 0.0`). bool(-1.0) is True,
        # so an unavailable read was counting as a rejection candle — and since
        # both fields go to -1.0 together, as a BULLISH and a BEARISH one at the
        # same time. That handed every setup a free confirmation precisely when
        # the engine could see least, which is the opposite of what the two-print
        # requirement below exists for.
        bull_cdl = _f(result, 'cdl_bull_reversal', -1.0) > 0.0
        bear_cdl = _f(result, 'cdl_bear_reversal', -1.0) > 0.0
        ltf_up   = bool(confirm.get('ltf_bull'))     # engine's 5m alignment check
        ltf_down = bool(confirm.get('ltf_bear'))
        slope    = _f(result, 'rsi_slope', 0.0)

        if side == 'BUY':
            signals = [(bull_cdl, 'bullish rejection candle'),
                       (ltf_up, '5m momentum turned up'),
                       (slope > 0, 'RSI curling up')]
        else:
            signals = [(bear_cdl, 'bearish rejection candle'),
                       (ltf_down, '5m momentum turned down'),
                       (slope < 0, 'RSI curling down')]
        hits = [why for ok, why in signals if ok]

        # A counter-trend entry is the one that must not be taken on hope: it
        # needs two independent prints.  This is the direct fix for the eight
        # shorts that were all "at resistance" and none of which had turned.
        need = 2 if setup in (SETUP_EXHAUSTION_REVERSAL, SETUP_RANGE_FADE) else 1
        if len(hits) >= need:
            return True, ' + '.join(hits)
        return False, (f'needs {need} confirmation(s), has {len(hits)}'
                       + (f' ({hits[0]})' if hits else ''))

    # ── Public entry point ────────────────────────────────────────────────────
    @classmethod
    def evaluate(
        cls,
        result:  Dict[str, Any],
        regime:  Any,
        market:  Dict[str, Any],
        book:    Optional[Dict[str, Any]] = None,
        levels:  Optional[Sequence[Tuple[float, int]]] = None,
        confirm: Optional[Dict[str, Any]] = None,
    ) -> TradePlan:
        """Run the playbook and return a TradePlan.

        result  : predictor/engine signal dict (range_position, atr, atr_pct, rsi,
                  support, resistance, p_buy/p_sell, *_broken_recent, cdl_*, ...).
        regime  : RegimeState-like with .regime / .confidence, or None.
        market  : drift_blocked, drift_severity, news_locked, news_label,
                  spread_pct, atr_normal_pct, tide_dir ('UP'|'DOWN'|'FLAT'),
                  tide_strength (0..1).
        book    : open_total, max_open, cluster_long, cluster_short,
                  max_per_cluster — the engine's current exposure, already counted.
        levels  : [(price, touches)] deep structure from `_structural_levels`.
        confirm : ltf_bull / ltf_bear — the engine's lower-timeframe checks.
        """
        market  = market or {}
        book    = book or {}
        confirm = confirm or {}
        notes: List[str] = []

        price = _f(result, 'price') or _f(result, 'entry_price')
        atr   = _f(result, 'atr')
        if price <= 0 or atr <= 0:
            return _reject('fitness', 'no usable price/ATR', notes)

        regime_name = str(getattr(regime, 'regime', None)
                          or result.get('hmm_regime') or 'UNKNOWN').upper()
        regime_conf = float(getattr(regime, 'confidence', 0.0) or 0.0)

        # ── Stage 0 · fitness ────────────────────────────────────────────────
        unfit = cls._fitness(result, regime_name, market)
        if unfit:
            return _reject('fitness', unfit, notes)
        notes.append(f'fitness: tradeable ({regime_name} @ {regime_conf:.0%} conf, ATR '
                     f'{_f(result, "atr_pct", 0):.2f}%)')

        # ── Stage 1 · what is the trade ──────────────────────────────────────
        setup, side, why = cls._classify(result, regime_name, regime_conf)
        if setup == SETUP_NONE or side == 'FLAT':
            return _reject('setup', why, notes)
        notes.append(f'setup: {setup} {side} — {why}')

        # ── Stage 1b · does the higher timeframe permit this side here? ──────
        # Applied to the OUTPUT of _classify rather than inside it, so a setup
        # added later cannot route around it. In practice only BREAK_RETEST can
        # trip this — every other setup already trades with its location.
        _rp  = _f(result, 'range_position', 0.5)
        _rsi = _f(result, 'rsi', _f(result, 'rsi_14', 50.0))
        _loc_refusal = cls._counter_location_refusal(side, _rp, _rsi, result)
        if _loc_refusal:
            return _reject('location', _loc_refusal, notes, side, setup)
        notes.append(f'location: {side} permitted at rp {_rp:.2f} '
                     f'(weekly {_f(result, "macro_weekly", 0.0):+.0f}, '
                     f'daily {_f(result, "macro_daily", 0.0):+.0f})')

        # The model does not grant permission; it may only object.  A model
        # leaning hard the other way is real information, so it vetoes; a
        # neutral or mildly-disagreeing model does not override structure.
        p_buy, p_sell = _f(result, 'p_buy'), _f(result, 'p_sell')
        oppose = (p_sell - p_buy) if side == 'BUY' else (p_buy - p_sell)
        if oppose > MODEL_OPPOSE_MARGIN:
            return _reject('setup',
                           f'model leans the other way by {oppose:.2f} '
                           f'(> {MODEL_OPPOSE_MARGIN}) — structure and model disagree',
                           notes, side, setup)
        if REQUIRE_MODEL_FIRE and not (result.get('fire') and result.get('side') == side):
            return _reject('setup', 'model did not independently fire this side', notes, side, setup)
        notes.append(f'model: does not object (oppose margin {oppose:+.2f})')

        # ── Stage 2 · where am I wrong ───────────────────────────────────────
        level = cls._pick_level(side, setup, price, atr, levels or [], result)
        if level <= 0:
            return _reject('invalidation',
                           f'no structural level on the {"support" if side == "BUY" else "resistance"} '
                           f'side to lean on', notes, side, setup)

        buf = STOP_BUFFER_ATR * atr
        stop = (level - buf) if side == 'BUY' else (level + buf)

        if abs(price - stop) < MIN_STOP_ATR * atr:
            # Tighten-to-fit is how the old system produced stops inside the noise
            # band; push the stop out to the floor instead and let stage 3 decide
            # whether the trade still pays with an honest stop.
            stop = (price - MIN_STOP_ATR * atr) if side == 'BUY' else (price + MIN_STOP_ATR * atr)
            notes.append(f'invalidation: level {level:.8g} is inside the noise band — '
                         f'stop widened to the {MIN_STOP_ATR} ATR floor')

        # Whichever of the two placements won, it must still stand clear of every
        # level between it and price — see `_clear_levels`.  This runs last so the
        # noise-band floor cannot re-park a widened stop underneath a level.
        cleared = cls._clear_levels(side, price, stop, atr, levels or [], result)
        if abs(cleared - stop) > 1e-12:
            notes.append(f'invalidation: stop moved {stop:.8g} -> {cleared:.8g} to stand '
                         f'clear of the level it was sitting {"under" if side == "SELL" else "on top of"}')
            stop = cleared

        risk = abs(price - stop)
        risk_atr = risk / atr if atr > 0 else 0.0
        if risk_atr > MAX_STOP_ATR:
            return _reject('invalidation',
                           f'invalidation is {risk_atr:.1f} ATR away (> {MAX_STOP_ATR}) — '
                           f'too far behind the level to be worth defending',
                           notes, side, setup)
        notes.append(f'invalidation: level {level:.8g}, stop {stop:.8g} '
                     f'({risk_atr:.2f} ATR beyond it)')

        # ── Stage 3 · does it pay ────────────────────────────────────────────
        # `risk` is already known here (stage 2 placed the stop), so the target
        # can be chosen against the floor it has to clear instead of a fixed
        # distance that is inconsistent with it.
        target = cls._pick_target(side, price, atr, levels or [], result, risk)
        if target <= 0:
            return _reject('payoff',
                           f'no structural objective at least {MIN_TARGET_ATR} ATR ahead — '
                           f'nowhere to be paid', notes, side, setup)

        reward_pct = abs(target - price) / price * 100.0
        risk_pct   = risk / price * 100.0
        r_gross    = (reward_pct / risk_pct) if risk_pct > 0 else 0.0
        # Costs come off the win and go onto the loss — the way they actually land.
        r_net = ((reward_pct - ROUND_TRIP_COST_PCT) /
                 (risk_pct + ROUND_TRIP_COST_PCT)) if risk_pct > 0 else 0.0
        if r_net < MIN_NET_R:
            return _reject('payoff',
                           f'net R:R {r_net:.2f} below the {MIN_NET_R} floor '
                           f'(risk {risk_pct:.2f}%, reward {reward_pct:.2f}%, costs '
                           f'{ROUND_TRIP_COST_PCT}%) — the trade does not pay',
                           notes, side, setup)
        notes.append(f'payoff: target {target:.8g}, {r_gross:.2f}R gross / {r_net:.2f}R net')

        # ── Stage 4 · is it time ─────────────────────────────────────────────
        dist_atr = abs(price - level) / atr
        # NB: no "price is already through the stop" check here — `_pick_level`
        # only ever returns a level on the correct side of price (within the
        # AT_LEVEL_ATR tag tolerance), and the buffer is wider than that
        # tolerance, so the stop is unconditionally beyond price by construction.
        # A market that has genuinely lost its level surfaces as either the NEXT
        # level down being chosen, or no level at all — both handled in stage 2.
        ok, cwhy = cls._confirmation(result, side, setup, confirm)
        if dist_atr <= AT_LEVEL_ATR and ok:
            action, expiry = ACTION_ENTER, 0
            notes.append(f'trigger: at the level ({dist_atr:.2f} ATR) and confirmed — {cwhy}')
        elif dist_atr <= REACH_ATR:
            # A resting order, not a queue entry: it has a price, an invalidation
            # and a clock.  This is what replaces PENDING.
            action, expiry = ACTION_WORK, WORK_EXPIRY_BARS
            notes.append(f'trigger: working an order at {level:.8g} ({dist_atr:.2f} ATR away), '
                         f'expires in {WORK_EXPIRY_BARS} bars'
                         + ('' if ok else f' — {cwhy}'))
        else:
            return _reject('trigger',
                           f'level is {dist_atr:.1f} ATR away (> {REACH_ATR}) — not a trade yet',
                           notes, side, setup)

        # ── Stage 5 · how much ───────────────────────────────────────────────
        size = SETUP_RISK_WEIGHT.get(setup, 0.5)
        notes.append(f'allocation: {setup} base weight {size:.2f}')

        tide_dir = str(market.get('tide_dir', 'FLAT') or 'FLAT').upper()
        tide_str = _f(market, 'tide_strength', 0.0)
        against_tide = (side == 'SELL' and tide_dir == 'UP') or (side == 'BUY' and tide_dir == 'DOWN')
        if against_tide:
            if tide_str >= STRONG_TIDE:
                return _reject('allocation',
                               f'{side} against a strong BTC {tide_dir} tide '
                               f'({tide_str:.0%}) — this is the basket trade that bled',
                               notes, side, setup)
            size *= COUNTER_TIDE_FACTOR
            notes.append(f'allocation: against the BTC {tide_dir} tide — halved')

        max_open = int(book.get('max_open', 5) or 5)
        if int(book.get('open_total', 0) or 0) >= max_open:
            return _reject('allocation', f'book already holds {max_open} positions — '
                                         f'this is not one of the best {max_open}', notes, side, setup)

        max_cluster = int(book.get('max_per_cluster', 2) or 2)
        # Counted per side, because the caller cannot know which side the
        # playbook will choose until stage 1 has run.
        same_dir = int(book.get('cluster_long' if side == 'BUY' else 'cluster_short', 0) or 0)
        if same_dir >= max_cluster:
            return _reject('allocation',
                           f'{same_dir} correlated {side} positions already open in this cluster — '
                           f'one thesis, not {same_dir + 1} bets', notes, side, setup)
        if same_dir >= 1:
            size *= CLUSTER_SECOND_FACTOR
            notes.append(f'allocation: {same_dir} correlated {side} already open — scaled down')

        if size < MIN_SIZE_FACTOR:
            return _reject('allocation',
                           f'risk allocation fell to {size:.2f} (< {MIN_SIZE_FACTOR}) — '
                           f'too small to be worth its costs', notes, side, setup)

        invalidation = stop
        plan = TradePlan(
            action=action, side=side, setup=setup,
            entry=(price if action == ACTION_ENTER else level),
            stop=stop, target=target, level=level, invalidation=invalidation,
            risk_atr=risk_atr, r_gross=r_gross, r_net=r_net,
            size_factor=round(size, 3), expiry_bars=expiry,
            stage='allocation',
            reason=f'{setup} {side} @ {level:.8g} — {r_net:.2f}R net, size {size:.2f}',
            notes=notes,
        )
        return plan
