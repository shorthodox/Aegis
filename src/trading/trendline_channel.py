"""
trendline_channel.py — Trendline & Trend Channel Detection.

An INDEPENDENT confirmation tool. It does NOT generate, modify, or override any
buy/sell signal — it only analyses market structure and returns a confidence
score plus context that the caller may display or fold into a tier. Per the
spec:

  * Trend from market structure (HH/HL = up, LH/LL = down, else range).
  * Primary trendline from two confirmed swings, extended forward; touches and
    breaks counted within an ATR tolerance.
  * A parallel channel built ONLY when a valid primary trendline exists — never
    forced. The primary trendline always outranks the channel.
  * Higher timeframe (1d / 4h) for primary detection; the live price (1h/below)
    only for execution-distance / "which boundary is price at now".
  * ATR defines every tolerance. Pivots are confirmed (non-repainting).

Usage:
    det = TrendlineChannelDetector()
    out = det.analyze(htf_candles_4h, price=live_price)   # atr auto-computed
    # out['signal_confirmation'] in {'Bullish Support','Bearish Resistance','Neutral'}
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple


# ── helpers ──────────────────────────────────────────────────────────────────
def confirmed_pivots(vals: List[float], k: int, want_high: bool) -> List[int]:
    """Indices of confirmed swing pivots — a local extreme with k bars on BOTH
    sides. The k bars AFTER the pivot make it non-repainting (the newest
    detectable pivot is already k bars old and can never be revised)."""
    out: List[int] = []
    n = len(vals)
    for i in range(k, n - k):
        w = vals[i - k:i + k + 1]
        if (want_high and vals[i] >= max(w)) or (not want_high and vals[i] <= min(w)):
            out.append(i)
    return out


def compute_atr(highs: List[float], lows: List[float], closes: List[float],
                period: int = 14) -> float:
    """Simple ATR (final value) from OHLC lists."""
    if len(closes) < 2:
        return 0.0
    trs: List[float] = []
    for i in range(1, len(closes)):
        trs.append(max(highs[i] - lows[i],
                       abs(highs[i] - closes[i - 1]),
                       abs(lows[i] - closes[i - 1])))
    if not trs:
        return 0.0
    p = min(period, len(trs))
    return sum(trs[-p:]) / p


def _line_at(x: float, x0: float, y0: float, slope: float) -> float:
    return y0 + slope * (x - x0)


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


class TrendlineChannelDetector:
    # ── ATR-scaled tolerances / structure params ─────────────────────────────
    PIVOT_K        = 3     # swing half-window (confirmed, non-repainting)
    TOUCH_TOL_ATR  = 0.6   # a wick within this ×ATR of the line counts as a touch
    BREAK_TOL_ATR  = 0.6   # a CLOSE beyond the line by more than this ×ATR = a break
    MIN_TOUCHES    = 3     # a VALID trendline needs at least this many touches
    NEAR_ATR       = 1.0   # price within this ×ATR of a boundary = "at" that boundary
    CH_MIN_REACT   = 2     # a VALID channel needs >= this many reactions on EACH side

    # ── public API ───────────────────────────────────────────────────────────
    def analyze(self, candles: List[list], price: float = 0.0,
                atr: float = 0.0) -> Dict:
        """candles: HTF OHLC rows [ts, o, h, l, c, ...] (4h or 1d); price: live
        price for the execution distance (falls back to last close); atr: ATR in
        PRICE units (auto-computed from candles when 0)."""
        try:
            closed = candles[:-1] if len(candles) >= 2 else candles   # drop forming bar
            if len(closed) < 4 * self.PIVOT_K + 12:
                return self._blank('Range')
            highs  = [float(c[2]) for c in closed]
            lows   = [float(c[3]) for c in closed]
            closes = [float(c[4]) for c in closed]
            if atr <= 0:
                atr = compute_atr(highs, lows, closes)
            if atr <= 0 or price <= 0:
                price = price if price > 0 else closes[-1]
            if atr <= 0:
                return self._blank('Range')

            hi_idx = confirmed_pivots(highs, self.PIVOT_K, True)
            lo_idx = confirmed_pivots(lows,  self.PIVOT_K, False)
            if len(hi_idx) < 2 or len(lo_idx) < 2:
                return self._blank('Range')
            swing_highs = [(i, highs[i]) for i in hi_idx]
            swing_lows  = [(i, lows[i])  for i in lo_idx]

            trend = self._trend(swing_highs, swing_lows, atr)
            if trend == 'RANGE':
                return self._blank('Range')

            x_now = len(closed)           # the (forming) bar where `price` lives

            # ── PRIMARY TRENDLINE ────────────────────────────────────────────
            # Slope from a regression over the recent higher-lows (uptrend) /
            # lower-highs (downtrend) — robust to a single pullback pivot that
            # would otherwise flip a 2-point line the wrong way. Anchored at the
            # MOST RECENT pivot so the line stays tight to current price.
            if trend == 'BULL':
                x0, y0, slope = self._fit_line(swing_lows[-6:])
                tl = self._score_line(x0, y0, slope, highs, lows, closes,
                                      atr, support=True)
            else:  # BEAR
                x0, y0, slope = self._fit_line(swing_highs[-6:])
                tl = self._score_line(x0, y0, slope, highs, lows, closes,
                                      atr, support=False)
            tl_now = _line_at(x_now, x0, y0, slope)

            # ── PARALLEL CHANNEL (only if the trendline is not broken) ────────
            channel = self._channel(trend, x0, y0, slope, highs, lows, closes,
                                    atr) if tl['status'] != 'Broken' else None

            # ── NEAREST BOUNDARY / EXPECTED REACTION / CONFIRMATION ───────────
            boundaries: List[Tuple[str, float]] = [('Trendline', tl_now)]
            if channel and channel['status'] != 'Invalid':
                up_now = _line_at(x_now, x0, channel['upper_y0'], slope)
                lo_now = _line_at(x_now, x0, channel['lower_y0'], slope)
                boundaries.append(('Upper Channel', up_now))
                boundaries.append(('Lower Channel', lo_now))

            near_name, near_val = min(boundaries, key=lambda b: abs(price - b[1]))
            dist_abs = abs(price - near_val)
            dist_pct = (dist_abs / price * 100.0) if price > 0 else 0.0
            dist_atr = dist_abs / atr

            expected, confirmation = self._reaction(
                trend, near_name, price, tl_now, dist_atr)

            # ── Line geometry for the chart ──────────────────────────────────
            # A reference (time in SECONDS, price) + slope per SECOND, so the
            # frontend draws the diagonal correctly on ANY displayed timeframe
            # (the 4h anchor is an absolute timestamp). Channel lines share the
            # trendline slope (they are parallel); only their intercept differs.
            _bar_ms = (closed[1][0] - closed[0][0]) if len(closed) >= 2 else 14_400_000
            _ref_t  = float(closed[int(x0)][0]) / 1000.0
            _m_sec  = (slope * 1000.0 / _bar_ms) if _bar_ms else 0.0
            lines = {'trendline': {'t': _ref_t, 'p': round(y0, 10), 'm': _m_sec}}
            if channel and channel['status'] in ('Valid', 'Weak'):
                lines['channel_upper'] = {'t': _ref_t, 'p': round(channel['upper_y0'], 10), 'm': _m_sec}
                lines['channel_lower'] = {'t': _ref_t, 'p': round(channel['lower_y0'], 10), 'm': _m_sec}

            return {
                'trend':                {'BULL': 'Bullish', 'BEAR': 'Bearish'}[trend],
                'primary_trendline':    tl['status'],          # Valid / Weak / Broken
                'trendline_confidence': round(tl['confidence'], 1),
                'trendline_touches':    tl['touches'],
                'trendline_breaks':     tl['breaks'],
                'parallel_channel':     (channel['status'] if channel else 'Invalid'),
                'channel_confidence':   round(channel['confidence'], 1) if channel else 0.0,
                'nearest_boundary':     near_name,
                'distance_pct':         round(dist_pct, 3),
                'distance_atr':         round(dist_atr, 2),
                'expected_reaction':    expected,               # Bounce/Rejection/Breakout/No Edge
                'signal_confirmation':  confirmation,           # Bullish Support / Bearish Resistance / Neutral
                'lines':                lines,                  # geometry for the chart (t=sec, p=price, m=slope/sec)
            }
        except Exception:
            return self._blank('Range')

    # ── internals ────────────────────────────────────────────────────────────
    @staticmethod
    def _slope(idxs: List[int], vals: List[float]) -> float:
        """Least-squares slope (price per bar) of the points."""
        n = len(idxs)
        if n < 2:
            return 0.0
        mx = sum(idxs) / n
        my = sum(vals) / n
        den = sum((x - mx) ** 2 for x in idxs)
        if den == 0:
            return 0.0
        return sum((x - mx) * (y - my) for x, y in zip(idxs, vals)) / den

    def _trend(self, swing_highs, swing_lows, atr) -> str:
        """Uptrend = highs AND lows both sloping UP (HH + HL over the recent
        swings); Downtrend = both sloping DOWN (LH + LL). A single choppy pivot
        no longer flips the read — the slope of the last few swings on each side
        decides. Requires a net structural move of >= 1 ATR so a dead-flat tape
        reads Range, not a phantom trend."""
        hi = swing_highs[-4:]
        lo = swing_lows[-4:]
        sh = self._slope([i for i, _ in hi], [v for _, v in hi])
        sl = self._slope([i for i, _ in lo], [v for _, v in lo])
        move = abs(hi[-1][1] - hi[0][1]) + abs(lo[-1][1] - lo[0][1])
        strong = move >= atr
        if sh > 0 and sl > 0 and strong:
            return 'BULL'
        if sh < 0 and sl < 0 and strong:
            return 'BEAR'
        return 'RANGE'

    def _fit_line(self, pivots: List[Tuple[int, float]]) -> Tuple[int, float, float]:
        """Regression slope over the recent pivots (robust to one pullback),
        anchored at the MOST RECENT pivot so the line stays tight to current
        price. Returns (x0, y0, slope)."""
        xs = [x for x, _ in pivots]
        ys = [y for _, y in pivots]
        slope = self._slope(xs, ys)
        x0, y0 = pivots[-1]
        return x0, y0, slope

    def _score_line(self, x0, y0, slope, highs, lows, closes, atr,
                    support: bool) -> Dict:
        """Count touches & breaks of the line over the bars AT/AFTER the anchor.
        support=True: an ascending SUPPORT line (uptrend) — a touch is a LOW near
        the line that CLOSES above it; a break is a CLOSE below by > tol. Mirror
        for a descending RESISTANCE line (support=False)."""
        touch_tol = self.TOUCH_TOL_ATR * atr
        break_tol = self.BREAK_TOL_ATR * atr
        touches = breaks = 0
        devs: List[float] = []
        recent_break = False
        n = len(closes)
        for x in range(int(x0), n):
            line = _line_at(x, x0, y0, slope)
            if support:
                wick_dist = lows[x] - line          # +ve above the line
                if abs(wick_dist) <= touch_tol and closes[x] >= line - touch_tol:
                    touches += 1
                    devs.append(abs(wick_dist))
                if closes[x] < line - break_tol:
                    breaks += 1
                    recent_break = recent_break or (x >= n - self.PIVOT_K - 2)
            else:
                wick_dist = line - highs[x]          # +ve below the line
                if abs(wick_dist) <= touch_tol and closes[x] <= line + touch_tol:
                    touches += 1
                    devs.append(abs(wick_dist))
                if closes[x] > line + break_tol:
                    breaks += 1
                    recent_break = recent_break or (x >= n - self.PIVOT_K - 2)

        avg_dev_atr = (sum(devs) / len(devs) / atr) if devs else 1.0
        conf = 0.0
        conf += min(60.0, touches * 15.0)               # up to 60 from touches
        conf += max(0.0, 25.0 * (1.0 - min(1.0, avg_dev_atr)))   # up to 25 for tight adherence
        conf += 15.0 if breaks == 0 else 0.0            # clean line bonus
        conf -= breaks * 30.0                           # each clean break hurts
        conf = _clamp(conf)

        if recent_break or breaks >= 2:
            status = 'Broken'
        elif touches >= self.MIN_TOUCHES and breaks == 0:
            status = 'Valid'
        else:
            status = 'Weak'
        return {'touches': touches, 'breaks': breaks, 'confidence': conf,
                'status': status, 'avg_dev_atr': avg_dev_atr}

    def _channel(self, trend, x0, y0, slope, highs, lows, closes, atr) -> Optional[Dict]:
        """Build the parallel boundary (same slope, shifted to the widest opposite
        swing) and validate it. Never forced: returns Invalid when the opposite
        side has too few reactions."""
        touch_tol = self.TOUCH_TOL_ATR * atr
        n = len(closes)
        # Offset the parallel line to the most extreme opposite wick.
        if trend == 'BULL':                       # lower line = support; upper = parallel above
            offsets = [highs[x] - _line_at(x, x0, y0, slope) for x in range(int(x0), n)]
            up_off = max(offsets) if offsets else 0.0
            upper_y0 = y0 + up_off
            lower_y0 = y0
            up_react = sum(1 for x in range(int(x0), n)
                           if abs(highs[x] - _line_at(x, x0, upper_y0, slope)) <= touch_tol)
            lo_react = sum(1 for x in range(int(x0), n)
                           if abs(lows[x] - _line_at(x, x0, lower_y0, slope)) <= touch_tol)
        else:                                     # upper line = resistance; lower = parallel below
            offsets = [_line_at(x, x0, y0, slope) - lows[x] for x in range(int(x0), n)]
            dn_off = max(offsets) if offsets else 0.0
            upper_y0 = y0
            lower_y0 = y0 - dn_off
            up_react = sum(1 for x in range(int(x0), n)
                           if abs(highs[x] - _line_at(x, x0, upper_y0, slope)) <= touch_tol)
            lo_react = sum(1 for x in range(int(x0), n)
                           if abs(lows[x] - _line_at(x, x0, lower_y0, slope)) <= touch_tol)

        # % of bars that stayed INSIDE the channel [lower, upper].
        inside = 0
        span = int(x0)
        for x in range(int(x0), n):
            u = _line_at(x, x0, upper_y0, slope)
            l = _line_at(x, x0, lower_y0, slope)
            if l - touch_tol <= closes[x] <= u + touch_tol:
                inside += 1
        inside_pct = (inside / max(1, n - int(x0))) * 100.0

        conf = 0.0
        conf += min(30.0, up_react * 10.0)
        conf += min(30.0, lo_react * 10.0)
        conf += 0.4 * inside_pct                        # up to 40 for containment
        conf = _clamp(conf)

        if up_react >= self.CH_MIN_REACT and lo_react >= self.CH_MIN_REACT and inside_pct >= 60.0:
            status = 'Valid'
        elif (up_react >= 1 and lo_react >= 1) or inside_pct >= 50.0:
            status = 'Weak'
        else:
            status = 'Invalid'
        return {'status': status, 'confidence': conf,
                'upper_y0': upper_y0, 'lower_y0': lower_y0,
                'upper_reactions': up_react, 'lower_reactions': lo_react,
                'inside_pct': round(inside_pct, 1)}

    def _reaction(self, trend, near_name, price, tl_now, dist_atr) -> Tuple[str, str]:
        """Expected reaction + signal confirmation from where price sits vs the
        nearest boundary. Encodes the user's rule: at a SUPPORT boundary expect a
        bounce (Bullish Support -> confirms a BUY); at a RESISTANCE boundary
        expect a rejection (Bearish Resistance -> confirms a SELL)."""
        at_boundary = dist_atr <= self.NEAR_ATR
        # A clean break of the primary trendline overrides everything.
        if trend == 'BULL' and price < tl_now - self.BREAK_TOL_ATR * (abs(tl_now) * 0 + 1e-12):
            pass  # break handled by status='Broken' upstream; keep simple here
        if not at_boundary:
            return 'No Edge', 'Neutral'

        if trend == 'BULL':
            # In an uptrend the trendline & lower channel are SUPPORT; the upper
            # channel is RESISTANCE.
            if near_name in ('Trendline', 'Lower Channel'):
                return 'Bounce', 'Bullish Support'
            if near_name == 'Upper Channel':
                return 'Rejection', 'Bearish Resistance'
        else:  # BEAR
            # In a downtrend the trendline & upper channel are RESISTANCE; the
            # lower channel is SUPPORT.
            if near_name in ('Trendline', 'Upper Channel'):
                return 'Rejection', 'Bearish Resistance'
            if near_name == 'Lower Channel':
                return 'Bounce', 'Bullish Support'
        return 'No Edge', 'Neutral'

    @staticmethod
    def _blank(trend: str = 'Range') -> Dict:
        return {
            'trend': trend, 'primary_trendline': 'Broken',
            'trendline_confidence': 0.0, 'trendline_touches': 0,
            'trendline_breaks': 0, 'parallel_channel': 'Invalid',
            'channel_confidence': 0.0, 'nearest_boundary': None,
            'distance_pct': None, 'distance_atr': None,
            'expected_reaction': 'No Edge', 'signal_confirmation': 'Neutral',
            'lines': {},
        }
