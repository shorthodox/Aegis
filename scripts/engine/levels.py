"""Market structure: candles, swing/HTF support & resistance, the daily
bias, the trendline channel and the BTC tide.

Everything here is READ-ONLY market description. No method decides a
trade; they supply the levels the gate leans on. Keeping that separation
is what lets the S/R logic be changed without touching permissioning.

Extracted verbatim from the single-file live_engine.py; the bodies are
unchanged, only the class they hang off moved. LiveEngine composes this
mixin, so `self` is the full engine.
"""
from __future__ import annotations

from typing import List
from typing import Optional
from typing import Tuple
import asyncio
import time

from scripts.engine.indicators import _confirmed_pivots
from scripts.engine.market_data import _fetch_ohlcv_sync


class LevelsMixin:
    """_fetch_candles .. _sr_levels — see module docstring."""

    async def _fetch_candles(self, symbol: str, timeframe: str, limit: int) -> List:
        """
        Return up to `limit` closed OHLCV candles for `symbol` on `timeframe`.
        Results are cached for _candle_cache_ttl seconds to avoid hammering the
        exchange on every scan cycle.  Returns [] on any error (gate fails open).
        """
        # Key MUST include limit: a shallow fetch (e.g. 1h/40 for breakout quality,
        # or 1h/500 for the gate) and a DEEP one (1h/1500 for chart S/R) share the
        # same symbol+timeframe, so keying without limit let a shallow result poison
        # the deep caller — silently truncating the deep S/R scan to 40 bars.
        cache_key = f'{symbol}|{timeframe}|{limit}'
        now = time.time()
        entry = self._candle_cache.get(cache_key)
        if entry and (now - entry['ts']) < self._candle_cache_ttl:
            return entry['candles']

        loop = asyncio.get_event_loop()
        try:
            candles = await asyncio.wait_for(
                loop.run_in_executor(
                    self._executor,
                    lambda: _fetch_ohlcv_sync(symbol, timeframe, limit),
                ),
                timeout=8,
            )
        except Exception:
            candles = []

        if candles:
            self._candle_cache[cache_key] = {'candles': candles, 'ts': now}
        return candles or []

    @staticmethod
    def _flip_confirmed(candles: list, level: float, break_up: bool, tol: float) -> bool:
        """True only when a break beyond `level` has been RETEST-CONFIRMED — not
        merely a single close beyond it (which is often a liquidity sweep that
        reverses).  Sequence enforced (Break -> Retest -> Confirmation):
          1. a candle CLOSED beyond the level in the break direction, then
          2. price came back and RETESTED it (a later wick returned to the level),
          3. it HELD — the latest close is still beyond the level, and
          4. it did NOT FAIL — no candle closed back across the level after the
             break (that would cancel the flip; the level keeps its old role).
        break_up=True checks a resistance->support flip (broke UP); False checks a
        support->resistance flip (broke DOWN).  Uses closed candles only, so it is
        non-repainting.
        """
        if len(candles) < 4:
            return False
        brk = -1
        for i, c in enumerate(candles):
            cl = float(c[4])
            if (cl > level + tol) if break_up else (cl < level - tol):
                brk = i
                break
        if brk < 0 or brk >= len(candles) - 1:
            return False
        after = candles[brk + 1:]
        # RETEST: a wick returned to the level after the break.
        retested = any((float(c[3]) <= level + tol) if break_up else (float(c[2]) >= level - tol)
                       for c in after)
        if not retested:
            return False
        # FAILED: any close back across the level after the break cancels the flip.
        failed = any((float(c[4]) < level - tol) if break_up else (float(c[4]) > level + tol)
                     for c in after)
        if failed:
            return False
        # HELD: the most recent close is still on the break side.
        last_close = float(candles[-1][4])
        return (last_close > level) if break_up else (last_close < level)

    @staticmethod
    def _breakout_quality(candles: list, level: float, break_up: bool,
                          atr: float) -> Tuple[bool, list]:
        """Rate the breakout that pushed through `level` (on the level's own
        timeframe).  A real break has a decisive BODY (not just a wick) and
        ABOVE-AVERAGE VOLUME; a wick-only or low-volume push is a weak break
        (often a sweep).  Returns (is_strong, weak_reasons).  Missing data →
        (True, []) so we never penalise on absence of information.
        """
        weak: list = []
        if len(candles) < 6 or atr <= 0:
            return True, weak
        brk = None
        for c in candles:                     # last candle that CLOSED beyond
            cl = float(c[4])
            if (cl > level) if break_up else (cl < level):
                brk = c
        if brk is None or len(brk) < 6:
            return True, weak
        body = abs(float(brk[4]) - float(brk[1]))
        if body < atr * 0.5:                  # mostly wick, small body
            weak.append('wick_only')
        vols = [float(c[5]) for c in candles if len(c) > 5]
        if vols:
            avg = sum(vols) / len(vols)
            if float(brk[5]) < avg:           # below-average volume
                weak.append('low_volume')
        return (len(weak) == 0), weak

    @staticmethod
    def _dynamic_k(atr: float, price: float) -> int:
        """Pivot half-window sized to THIS token's own volatility (ATR%).  A
        choppier / higher-ATR% token needs a wider window so intrabar noise isn't
        mistaken for structure; a calm token resolves real swings in fewer bars.
        Every token therefore gets a swing sensitivity matched to how it moves.
        """
        if price <= 0 or atr <= 0:
            return 5
        atr_pct = atr / price * 100.0
        if atr_pct >= 3.0:
            return 7
        if atr_pct >= 1.5:
            return 6
        if atr_pct >= 0.8:
            return 5
        return 4

    @staticmethod
    def _dynamic_rsi_bounds(rsi_series: list) -> Tuple[float, float]:
        """Per-token counter-trend reversal RSI thresholds, calibrated to the
        token's OWN recent RSI distribution rather than a fixed 42/58.  A token
        that oscillates 40-60 needs only mild extremes to reverse; one that swings
        20-80 must reach deeper.  Uses the 25th/75th percentiles of recent RSI,
        clamped to a sane band so a trending token still needs a genuine extreme.
        Returns (long_threshold, short_threshold).
        """
        vals = [v for v in rsi_series if v == v]        # drop NaNs
        if len(vals) < 20:
            return 42.0, 58.0
        s = sorted(vals)
        p25 = s[int(len(s) * 0.25)]
        p75 = s[int(len(s) * 0.75)]
        long_t  = min(45.0, max(25.0, p25))             # oversold for this token
        short_t = max(55.0, min(75.0, p75))             # overbought for this token
        return long_t, short_t

    async def _daily_bias(self, symbol: str) -> Tuple[float, float]:
        """Daily + weekly HTF trend as ±1.0 flags for the macro-bias tiers.

        The model forwards macro_daily / macro_weekly as 0.0 for EVERY token:
        add_macro_regime_features (feature_engine) pre-creates a 0.0 default
        column and then merge_asof-collides it with the real value, renaming
        both to macro_trend_1d_x / _y — the next line reads the plain name,
        throws KeyError, and a bare except silently resets the trend to 0.0.
        So the whole HTF-bias system (the +15/-20 tiers in score_signal and
        Guard F) has been dormant since inception, reading a dead constant.

        Rather than touch the shared feature/training pipeline, recompute the
        two trend flags here, from daily candles, purely as gate inputs:

            macro_daily  = +1 if price > EMA50(daily)  else -1   (medium trend)
            macro_weekly = +1 if price > EMA200(daily) else -1   (long trend)

        EMA50/EMA200 on daily closes map exactly onto the tier semantics the
        existing code already encodes: price above the long EMA but below the
        medium one IS the "weekly bull / daily bear" pullback-buy the tiers
        reward. Closed daily candles only (non-repainting). Fails open to
        (0.0, 0.0) — the exact dormant state the tiers guard against with
        `if macro_weekly != 0.0 or macro_daily != 0.0` — so a missing feed
        simply leaves the bias off instead of firing a wrong direction.
        """
        try:
            raw    = await self._fetch_candles(symbol, '1d', 260)
            closed = raw[:-1] if len(raw) >= 2 else raw     # drop today's forming bar
            if len(closed) < 60:
                return 0.0, 0.0
            closes = [float(c[4]) for c in closed]

            def _ema_last(vals: List[float], span: int) -> float:
                k = 2.0 / (span + 1.0)
                e = vals[0]
                for v in vals[1:]:
                    e = v * k + e * (1.0 - k)
                return e

            px  = closes[-1]
            md  = 1.0 if px > _ema_last(closes, 50)  else -1.0
            mw  = 1.0 if px > _ema_last(closes, 200) else -1.0
            return md, mw
        except Exception:
            return 0.0, 0.0

    async def _trendline_channel(self, symbol: str, price: float,
                                 atr: float = 0.0) -> Optional[dict]:
        """ADDITIVE confirmation: run the Trendline & Trend Channel detector on
        HTF (4h) structure and return its analysis dict for display. It never
        gates or changes the signal — it is context + a confidence score only.
        Primary detection is on the 4h candles; `price` is the live 1h/execution
        price used for the distance-to-boundary. Returns None on any error."""
        try:
            raw = await self._fetch_candles(symbol, '4h', 300)
            if not raw or len(raw) < 40:
                return None
            # atr passed here is the 1h ATR; the detector recomputes its own 4h
            # ATR internally (atr=0) so tolerances match the detection timeframe.
            return self._tlc_detector.analyze(raw, price=price, atr=0.0)
        except Exception:
            return None

    async def _swing_sr(self, symbol: str, price: float,
                        atr: float = 0.0) -> Optional[Tuple[float, float]]:
        """Nearest SIGNIFICANT S/R around price, with RETEST-CONFIRMED role
        reversal — a broken level does NOT flip until a retest confirms it.

        support:
          nearest swing LOW below price (natural support), OR a swing HIGH below
          price that price broke ABOVE *and* a retest CONFIRMED as new support
          (old resistance -> support). An unconfirmed break is ignored — the
          support falls back to the last truly-held level.
        resistance:
          nearest swing HIGH above price (natural resistance), OR a swing LOW
          above price whose breakdown was retest-CONFIRMED as new resistance
          (old support -> resistance).

        Guards keep levels meaningful: k=5 swings only (no micro-wiggles), and a
        level must sit >= ~1 ATR (or 0.6% of price) from price. Non-repainting.
        Returns (support, resistance) or None when levels are sparse (caller
        falls back to the rolling 24h levels).
        """
        try:
            if price <= 0:
                return None
            # Deep history (shared 1500 cache with the chart's _sr_levels) and the
            # SAME MAJOR-swing window (_dynamic_k + 3) as the chart, so the level the
            # gate repoints to IS the level drawn on the chart. A nearer window (k+1)
            # surfaced minor swings sitting BETWEEN the real support and resistance,
            # which then read as "at resistance" and fired MID-RANGE (SUI 0.7443).
            # Frequency is preserved elsewhere: the zone is admitted by the 24h
            # range_position, not by requiring price to be glued to this level.
            raw_1h    = await self._fetch_candles(symbol, '1h', 1500)
            closed_1h = raw_1h[:-1] if len(raw_1h) >= 2 else raw_1h
            if len(closed_1h) < 50:
                return None
            highs = [float(c[2]) for c in closed_1h]
            lows  = [float(c[3]) for c in closed_1h]
            k = self._dynamic_k(atr, price)   # MAJOR swings only — match the chart, no phantom mid-range levels
            swing_highs = {highs[i] for i in _confirmed_pivots(highs, k, True)}
            swing_lows  = {lows[i]  for i in _confirmed_pivots(lows,  k, False)}
            # Per-token: the zone scales purely by THIS token's ATR (its own
            # volatility), not a fixed % that is too wide for a calm token and too
            # tight for a volatile one. The % is only a fallback when ATR is 0.
            gap    = atr if atr > 0 else price * 0.006
            tol    = (atr * 0.35) if atr > 0 else price * 0.0025
            recent = closed_1h[-40:]

            # Support candidates BELOW price: swing lows are always valid support;
            # a swing high below price only counts once its break-up is confirmed.
            sup = [l for l in swing_lows if l < price - gap]
            sup += [h for h in swing_highs
                    if h < price - gap and self._flip_confirmed(recent, h, True, tol)]
            # Resistance candidates ABOVE price: swing highs are always valid;
            # a swing low above price only counts once its break-down is confirmed.
            res = [h for h in swing_highs if h > price + gap]
            res += [l for l in swing_lows
                    if l > price + gap and self._flip_confirmed(recent, l, False, tol)]

            if sup and res:
                return (max(sup), min(res))
        except Exception:
            pass
        return None

    async def _important_levels(self, symbol: str, price: float,
                                atr: float = 0.0) -> List[Tuple[float, int]]:
        """The few levels price has REPEATEDLY reacted to — nothing else.

        The engine had two S/R systems and neither could answer "is price at a
        level right now?":

          * `rolling_support/resistance` (what the gate and range_position use)
            is just the 24h rolling high/low — a rolling extreme, never tested,
            sliding every hour. In any rally every token sits near its own 24h
            high BY CONSTRUCTION, so it read "at resistance" on 79% of the fleet
            at once. A level 79% of assets stand on is not a level.

          * `_swing_sr` finds real pivots but EXCLUDES everything within 1 ATR of
            price (the `gap` filter), so it can never report the level price is
            actually standing on — only the next one beyond a blind spot.
            Measured: 0 of 62 tokens were ever "at" a swing level.

        A level is IMPORTANT when price has come back to it and reacted, more
        than once. So: confirmed (non-repainting) pivots -> CLUSTER the ones
        within LEVEL_MERGE_ATR of each other into a single level, where the
        cluster size is the TOUCH COUNT -> keep only levels touched at least
        LEVEL_MIN_TOUCHES times -> then keep only the LEVEL_TOP_K most-touched.

        That last cap is the one that matters. Unbounded, this returns 10-24
        levels per token, so price is always within 0.6 ATR of *something* and
        "at a level" fires on 88% of the fleet — noise again. Capped at the 4
        strongest it returns ~4 levels (chart-like) and "at a level" lands at a
        realistic 27%: price is usually BETWEEN levels, which is the truth.

        Levels are bidirectional (a broken resistance becomes support), so highs
        and lows are pooled. Returns [(level_price, touch_count)] sorted by
        touches, strongest first; empty when structure is too sparse to trust.
        """
        try:
            if price <= 0 or atr <= 0:
                return []
            raw    = await self._fetch_candles(symbol, '1h', 1000)
            closed = raw[:-1] if len(raw) >= 2 else raw
            if len(closed) < 50:
                return []
            highs = [float(c[2]) for c in closed]
            lows  = [float(c[3]) for c in closed]
            k     = self._dynamic_k(atr, price)
            pivots = ([highs[i] for i in _confirmed_pivots(highs, k, True)] +
                      [lows[i]  for i in _confirmed_pivots(lows,  k, False)])
            if not pivots:
                return []

            # Cluster nearby pivots into one level; cluster size = touch count.
            tol: float = atr * self.LEVEL_MERGE_ATR
            merged: List[List[float]] = []          # [running_mean, count]
            for p in sorted(pivots):
                if merged and abs(p - merged[-1][0]) <= tol:
                    lvl, n = merged[-1]
                    merged[-1] = [(lvl * n + p) / (n + 1), n + 1]
                else:
                    merged.append([p, 1])

            levels = [(lvl, int(n)) for lvl, n in merged
                      if n >= self.LEVEL_MIN_TOUCHES]
            levels.sort(key=lambda t: -t[1])        # most-tested first
            return levels[:self.LEVEL_TOP_K]        # ONLY the important ones
        except Exception:
            return []

    async def _structural_levels(self, symbol: str, price: float,
                                 atr: float = 0.0) -> List[Tuple[float, int]]:
        """The full S/R structure from DEEP, MULTI-TIMEFRAME history.

        `_important_levels` scans only 1h/1000 (~41 days) and keeps the 4
        MOST-TOUCHED levels — which are the big old highs/lows. A near, freshly
        relevant level (a broken support now acting as resistance on the 4h/1d)
        never makes that top-4, so a pending SELL was sent to wait at a
        resistance 4-9% overhead while the level it should actually watch sat
        <0.5% away (HBAR: waited at 0.0705 / +4.3%, real resistance 0.0678 /
        +0.3%). This pools CONFIRMED (non-repainting) pivots across 1h + 4h + 1d
        — weeks on the 1h to ~4 years on the 1d, the "study the whole history"
        the levels need — clusters them within LEVEL_MERGE_ATR, and keeps every
        level touched >= PENDING_TARGET_MIN_TOUCHES. The caller then picks the
        NEAREST significant level on the side it needs, not the most-touched far
        one. Cheap: 3 cached fetches, one per timeframe. Returns [(level,
        touches)] sorted by price; [] when structure is too sparse.
        """
        try:
            if price <= 0 or atr <= 0:
                return []
            pivots: List[float] = []
            for tf, lim, k in (('1h', 1500, 3), ('4h', 1500, 3), ('1d', 1500, 2)):
                raw    = await self._fetch_candles(symbol, tf, lim)
                closed = raw[:-1] if len(raw) >= 2 else raw
                if len(closed) < 2 * k + 5:
                    continue
                hs = [float(c[2]) for c in closed]
                ls = [float(c[3]) for c in closed]
                pivots += [hs[i] for i in _confirmed_pivots(hs, k, True)]
                pivots += [ls[i] for i in _confirmed_pivots(ls, k, False)]
            if not pivots:
                return []
            tol: float = atr * self.LEVEL_MERGE_ATR
            merged: List[List[float]] = []              # [running_mean, count]
            for p in sorted(pivots):
                if merged and abs(p - merged[-1][0]) <= tol:
                    lvl, n = merged[-1]
                    merged[-1] = [(lvl * n + p) / (n + 1), n + 1]
                else:
                    merged.append([p, 1])
            # Important levels use the same minimum-touch threshold as the
            # rest of the structure gate.  PENDING_TARGET_MIN_TOUCHES was
            # never defined on LiveEngine; using LEVEL_MIN_TOUCHES keeps the
            # pending-target filter consistent and avoids dropping all level
            # discovery behind an attribute error.
            return [(lvl, int(n)) for lvl, n in merged
                    if n >= self.LEVEL_MIN_TOUCHES]
        except Exception:
            return []

    @staticmethod
    def _level_context(levels: List[Tuple[float, int]], price: float, atr: float,
                       at_level_atr: float) -> Optional[Tuple[float, int, str, float]]:
        """Nearest important level to price → (level, touches, role, dist_atr).

        role is what the level acts as FOR PRICE RIGHT NOW: a level above price
        is RESISTANCE, one below is SUPPORT. Returns None when no important level
        is within `at_level_atr` — i.e. price is genuinely between levels, and
        location has no directional opinion to offer.
        """
        if not levels or price <= 0 or atr <= 0:
            return None

        def _dist_to_price(t: Tuple[float, int]) -> float:
            return abs(t[0] - price)

        lvl, touches = min(levels, key=_dist_to_price)
        dist = (lvl - price) / atr              # +ve => level sits ABOVE price
        if abs(dist) > at_level_atr:
            return None
        return (lvl, touches, 'RESISTANCE' if dist > 0 else 'SUPPORT', abs(dist))

    async def _btc_tide(self) -> str:
        """Market tide: BTC close vs its 4h EMA50 — 'UP', 'DOWN' or 'FLAT'.

        The one portfolio-level fact no per-symbol gate can see: alts run
        ~0.8 intraday correlation to BTC, so a short book opened into a
        rising BTC bleeds together regardless of per-signal quality
        (measured 2026-07-19: 8 alt shorts red on a BTC bounce while the
        long book sat green). _open_position half-sizes any position that
        fights the tide. Cached 15 min; fails to 'FLAT' (no scaling) so a
        feed hiccup never distorts sizing.
        """
        now = time.time()
        if self._tide_val and now - self._tide_ts < 900:
            return self._tide_val
        try:
            raw = await self._fetch_candles('BTC/USDT', '4h', 60)
            closes = [float(c[4]) for c in (raw[:-1] if len(raw) >= 2 else raw)]
            if len(closes) < 50:
                return self._tide_val or 'FLAT'
            ema = closes[0]
            k = 2.0 / (50 + 1)
            for c in closes:
                ema = c * k + ema * (1 - k)
            self._tide_val = 'UP' if closes[-1] > ema else 'DOWN'
            self._tide_ts  = now
            # v83: how HARD the tide runs, for TraderGate's allocation stage.
            # Direction alone cannot distinguish "BTC is 0.2% over its EMA"
            # (noise — a counter-tide trade is fine) from "BTC is 4% over and
            # climbing" (the tape that drowned the 8-short basket). Distance
            # from the EMA in %, saturating at 3%, which on BTC 4h is a
            # decisively one-way market. Direction semantics are unchanged, so
            # the legacy Guard T path behaves exactly as before.
            self._tide_strength = min(1.0, abs(closes[-1] - ema) / ema * 100.0 / 3.0)
        except Exception:
            return self._tide_val or 'FLAT'
        return self._tide_val

    async def _htf_sr(self, symbol: str, price: float,
                      atr: float = 0.0) -> Optional[Tuple[float, float]]:
        """Nearest HIGHER-TIMEFRAME (4h + 1d) swing S/R around price.

        Returns (htf_support, htf_resistance) — the nearest confirmed swing LOW
        below price and swing HIGH above price pooled across the 4h and 1d
        timeframes (the big, slow levels price actually respects). These are used
        as a CONFLUENCE layer (Gate 1.8) and drawn on the chart; they do NOT gate
        firing (that stays on the 1h zone, so the rate holds). None when sparse.
        Non-repainting (_confirmed_pivots needs bars on both sides); cached fetch.
        """
        try:
            if price <= 0:
                return None
            highs_all: list = []
            lows_all:  list = []
            # (timeframe, bars, pivot half-window). ~33 days of 4h + ~4 months of 1d.
            for tf, limit, k in (('4h', 250, 4), ('1d', 120, 3)):
                raw = await self._fetch_candles(symbol, tf, limit)
                closed = raw[:-1] if len(raw) >= 2 else raw
                if len(closed) < 2 * k + 5:
                    continue
                hs = [float(c[2]) for c in closed]
                ls = [float(c[3]) for c in closed]
                highs_all += [hs[i] for i in _confirmed_pivots(hs, k, True)]
                lows_all  += [ls[i] for i in _confirmed_pivots(ls, k, False)]
            gap = (atr * 0.5) if atr > 0 else price * 0.003
            res = [h for h in highs_all if h > price + gap]
            sup = [l for l in lows_all  if l < price - gap]
            if sup and res:
                return (max(sup), min(res))
        except Exception:
            pass
        return None

    def _level_state(self, recent: list, level: float, natural: str, tol: float) -> str:
        """Break -> Retest -> Confirmation state machine for one S/R level, from
        closed 1h candles.  `natural` is the level's original role ('resistance'
        for a swing high, 'support' for a swing low).  States:
          NORMAL             price never closed beyond it — original role intact
          PENDING_BREAKOUT   resistance closed-through, no retest yet (don't flip)
          PENDING_BREAKDOWN  support closed-through, no retest yet
          WAITING_RETEST     price has pulled back TO the level, unresolved
          CONFIRMED          break + retest + hold — role has flipped
          FAILED             price closed back across without confirming — reverts
        """
        break_up = (natural == 'resistance')
        beyond   = (lambda cl: cl > level + tol) if break_up else (lambda cl: cl < level - tol)
        if not any(beyond(float(c[4])) for c in recent):
            return 'NORMAL'
        if self._flip_confirmed(recent, level, break_up, tol):
            return 'CONFIRMED'
        last_close = float(recent[-1][4])
        if beyond(last_close):
            return 'PENDING_BREAKOUT' if break_up else 'PENDING_BREAKDOWN'
        if abs(last_close - level) <= tol:
            return 'WAITING_RETEST'
        return 'FAILED'

    @staticmethod
    def _state_display(state: str, natural: str) -> Tuple[str, str, bool]:
        """Map a level state to (role, colour, dashed) for the chart.
          NORMAL / FAILED  -> grey (original role restored)
          PENDING_*        -> yellow dashed
          WAITING_RETEST   -> orange
          CONFIRMED        -> flipped role: support=green, resistance=red
        """
        if state == 'CONFIRMED':
            role  = 'support' if natural == 'resistance' else 'resistance'
            return role, ('green' if role == 'support' else 'red'), False
        if state in ('PENDING_BREAKOUT', 'PENDING_BREAKDOWN'):
            return natural, 'yellow', True
        if state == 'WAITING_RETEST':
            return natural, 'orange', False
        return natural, 'grey', False   # NORMAL, FAILED

    async def _sr_levels(self, symbol: str, price: float, atr: float) -> list:
        """Labelled S/R levels near price for the chart: each significant swing
        pivot with its Break->Retest->Confirmation state, effective role, colour
        and dashed flag.  Capped and deduped to avoid clutter."""
        out: list = []
        try:
            if price <= 0:
                return out
            raw_1h    = await self._fetch_candles(symbol, '1h', 1500)  # DEEP history (~9 wks) so the major HTF swings are in view, not just ~12 days
            closed_1h = raw_1h[:-1] if len(raw_1h) >= 2 else raw_1h
            if len(closed_1h) < 50:
                return out
            highs = [float(c[2]) for c in closed_1h]
            lows  = [float(c[3]) for c in closed_1h]
            tol    = (atr * 0.35) if atr > 0 else price * 0.0025   # per-token (ATR)
            recent = closed_1h[-40:]
            # WIDE reach so a big FAR level (a deep floor / high ceiling well away
            # from price) is eligible. The old 8-ATR band hid them — which is why
            # AAVE's ~84-85 support never drew. Bounded to a sane % of price.
            band   = (atr * 25) if atr > 0 else price * 0.35       # per-token (ATR)
            gap    = (atr * 1.2) if atr > 0 else price * 0.008     # clear of price (no bar-adjacent wiggles)
            # MAJOR swings only (wider window than the gate) — the few big obvious
            # levels, not every micro-pivot.
            k     = self._dynamic_k(atr, price) + 3
            merge = max(atr, price * 0.007)   # collapse a congestion cluster into ONE line
            # Confirmed swing highs above / lows below price, within reach. Sorted so
            # [0] is the NEAREST to price and [-1] is the most EXTREME (highest high
            # ceiling / lowest low floor) of the deep window.
            res_pivots = sorted({highs[i] for i in _confirmed_pivots(highs, k, True)
                                 if gap <= (highs[i] - price) <= band})
            sup_pivots = sorted({lows[i]  for i in _confirmed_pivots(lows,  k, False)
                                 if gap <= (price - lows[i]) <= band}, reverse=True)
            # Draw the NEAREST level (immediate context) AND the most EXTREME major
            # level (the deep floor / high ceiling) on each side — big obvious lines,
            # near and far, without the mid clutter. Deduped within `merge`.
            picked: list = []
            for cands, nat in ((res_pivots, 'resistance'), (sup_pivots, 'support')):
                for lvl in ([cands[0], cands[-1]] if cands else []):
                    if any(abs(lvl - p[0]) <= merge for p in picked):
                        continue
                    picked.append((lvl, nat))
            for lvl, nat in picked:
                state = self._level_state(recent, lvl, nat, tol)
                role, color, dashed = self._state_display(state, nat)
                out.append({'price': round(lvl, 8), 'state': state,
                            'role': role, 'color': color, 'dashed': dashed})
            # Overlay the higher-timeframe (4h+1d) S/R as distinct purple lines —
            # the big levels the confluence gate (1.8) judges against. Skipped if a
            # 1h level already sits within `merge` (avoid drawing two lines on top).
            try:
                _htf = await self._htf_sr(symbol, price, atr)
            except Exception:
                _htf = None
            if _htf is not None:
                _hs, _hr = _htf
                for _lvl, _role in ((_hr, 'htf_resistance'), (_hs, 'htf_support')):
                    if _lvl > 0 and not any(abs(_lvl - d['price']) <= merge for d in out):
                        out.append({'price': round(_lvl, 8), 'state': 'HTF',
                                    'role': _role, 'color': 'purple', 'dashed': True})
            out.sort(key=lambda d: d['price'])
        except Exception:
            pass
        return out
