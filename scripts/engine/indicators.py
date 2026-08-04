"""Confirmation indicators — pure-Python, closed-bar, non-repainting.

These operate on a list of OHLCV candles [ts, open, high, low, close, volume]
and are used ONLY by the post-model confirmation gate — never by the ML model
(the model is pinned to its saved feature_cols at inference, so nothing here can
change a prediction).  Every function is fed CLOSED bars (caller drops the
forming candle) and looks strictly backward, so there is no repainting /
lookahead: a decision made now uses only data that existed now.

Ported unchanged from the single-file live_engine.py. `_reversal_candle` in
particular is mirrored, vectorised, inside retrain_model.py
(reversal_candle_flags) so training labels agree with what the live gate will
accept — if you change the pattern rules here, change them there in the same
commit or the two silently disagree.

Names keep their leading underscore because tests and retrain_model.py refer to
them by those names; they are re-exported from scripts/live_engine.py.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

__all__ = [
    "_closes", "_ema_last", "_rsi_series", "_macd_line",
    "_detect_bos_choch", "_confirmed_pivots", "_range_pos",
    "_reversal_candle", "_detect_divergence", "_detect_volume_events",
]


def _closes(candles: List) -> List[float]:
    return [float(c[4]) for c in candles]


def _ema_last(values: List[float], span: int) -> List[float]:
    """Full EMA series (adjust=False), matching feature_engine.compute_macd."""
    if not values:
        return []
    k = 2.0 / (span + 1.0)
    out = [values[0]]
    for v in values[1:]:
        out.append(v * k + out[-1] * (1.0 - k))
    return out


def _rsi_series(closes: List[float], period: int = 14) -> List[float]:
    """Wilder-free simple-MA RSI — identical formula to compute_rsi (rolling mean)."""
    n = len(closes)
    rsi = [50.0] * n
    if n <= period:
        return rsi
    gains, losses = [0.0], [0.0]
    for i in range(1, n):
        d = closes[i] - closes[i - 1]
        gains.append(d if d > 0 else 0.0)
        losses.append(-d if d < 0 else 0.0)
    for i in range(period, n):
        avg_gain = sum(gains[i - period + 1:i + 1]) / period
        avg_loss = sum(losses[i - period + 1:i + 1]) / period
        rs = avg_gain / (avg_loss + 1e-9)
        rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _macd_line(closes: List[float], fast: int = 12, slow: int = 26) -> List[float]:
    """MACD line (fast EMA − slow EMA), matching compute_macd."""
    if len(closes) < 2:
        return [0.0] * len(closes)
    ef, es = _ema_last(closes, fast), _ema_last(closes, slow)
    return [a - b for a, b in zip(ef, es)]


def _detect_bos_choch(candles: List, lookback: int = 20, recent: int = 3) -> Dict[str, float]:
    """
    Faithful port of feature_engine.compute_bos_choch evaluated at the last
    closed bar.  Returns a single directional 'signal' in {-1, 0, +1}:
    a fresh CHoCH dominates (character change = reversal), then a fresh BOS,
    then the standing bos_state (inside/above/below the rolling range).
    """
    n = len(candles)
    if n < lookback + 2:
        return {'signal': 0.0, 'bos_state': 0.0, 'structure_bias': 0.0,
                'choch_bull': 0.0, 'choch_bear': 0.0}
    highs = [float(c[2]) for c in candles]
    lows  = [float(c[3]) for c in candles]
    closes = _closes(candles)

    def _above(i: int) -> bool:  # close above previous `lookback` high
        return closes[i] > max(highs[i - lookback:i])

    def _below(i: int) -> bool:
        return closes[i] < min(lows[i - lookback:i])

    def _bias(i: int) -> float:  # sign(close - close lookback ago)
        d = closes[i] - closes[i - lookback]
        return 1.0 if d > 0 else (-1.0 if d < 0 else 0.0)

    last = n - 1
    bos_state = (1.0 if _above(last) else 0.0) - (1.0 if _below(last) else 0.0)
    structure_bias = _bias(last)

    fresh_choch_bull = fresh_choch_bear = 0.0
    fresh_bos_up = fresh_bos_down = 0.0
    for i in range(max(lookback + 1, n - recent), n):
        up, dn = _above(i), _below(i)
        prev_up = _above(i - 1) if i - 1 >= lookback else False
        prev_dn = _below(i - 1) if i - 1 >= lookback else False
        bos_up   = up and not prev_up
        bos_down = dn and not prev_dn
        prior_bias = _bias(i - 1)
        if bos_up:
            fresh_bos_up = 1.0
            if prior_bias < 0:
                fresh_choch_bull = 1.0
        if bos_down:
            fresh_bos_down = 1.0
            if prior_bias > 0:
                fresh_choch_bear = 1.0

    if   fresh_choch_bear: signal = -1.0
    elif fresh_choch_bull: signal =  1.0
    elif fresh_bos_down:   signal = -1.0
    elif fresh_bos_up:     signal =  1.0
    else:                  signal = float(bos_state)
    return {'signal': signal, 'bos_state': bos_state, 'structure_bias': structure_bias,
            'choch_bull': fresh_choch_bull, 'choch_bear': fresh_choch_bear}


def _confirmed_pivots(vals: List[float], k: int, want_high: bool) -> List[int]:
    """
    Indices of confirmed swing pivots — a local extreme with k bars on BOTH
    sides.  Requiring k bars AFTER the pivot is what makes it non-repainting:
    the most recent detectable pivot is already k bars old, so it can never be
    revised by a future bar.
    """
    out = []
    for i in range(k, len(vals) - k):
        w = vals[i - k:i + k + 1]
        if (want_high and vals[i] >= max(w)) or (not want_high and vals[i] <= min(w)):
            out.append(i)
    return out


def _range_pos(result: Dict[str, Any]) -> float:
    """range_position (0 = at support, 1 = at resistance) with 0.0 PRESERVED.

    The old `... or 0.5` idiom silently
    rewrote a genuine 0.0 — price sitting at the absolute bottom of its range,
    the most extreme oversold-at-support reading there is — into 0.5, i.e. dead
    centre mid-range, because 0.0 is falsy in Python. That made the very best
    fade setups invisible to every location check: measured live, FIL printed
    edge_score 94 with RSI 11.1 at rp 0.00 and was blocked by Guard L's
    confluence net=-3 even though its reversal exemption (rp <= 0.35 AND
    rsi <= 32) matched on both counts. Missing/garbage still falls back to 0.5.
    """
    v = result.get('range_position')
    if v is None:
        return 0.5
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.5


def _reversal_candle(candles: List, want_bullish: bool) -> Optional[str]:
    """Detect a candlestick REVERSAL pattern COMPLETING at the last closed candle.

    A reversal entry must be confirmed by a real pattern — hammer, engulfing,
    harami, piercing/dark-cloud, or a morning/evening star — NOT by "3 random
    green candles", which is noise in a downtrend. want_bullish=True checks the
    bullish set (confirms a BUY reversal at support); False checks the bearish
    mirror (confirms a SELL reversal at resistance). Returns the pattern name or
    None. `candles`: raw OHLC rows [ts,o,h,l,c,...], oldest-first, CLOSED only.

    Mirrored vectorised in retrain_model.reversal_candle_flags — outputs must
    not change without changing that too.
    """
    if len(candles) < 2:
        return None
    o = lambda c: float(c[1]); h = lambda c: float(c[2])
    lo = lambda c: float(c[3]); cl = lambda c: float(c[4])
    c2, c1 = candles[-1], candles[-2]
    o2, h2, l2, x2 = o(c2), h(c2), lo(c2), cl(c2)
    o1, h1, l1, x1 = o(c1), h(c1), lo(c1), cl(c1)
    rng2 = max(h2 - l2, 1e-12); body2 = abs(x2 - o2)
    up2  = h2 - max(o2, x2);    dn2 = min(o2, x2) - l2
    body1 = abs(x1 - o1); rng1 = max(h1 - l1, 1e-12)
    prev_red, prev_green = x1 < o1, x1 > o1
    cur_red,  cur_green  = x2 < o2, x2 > o2
    mid1 = (o1 + x1) / 2.0

    if want_bullish:
        # Hammer: preceding red candle, long lower wick (>= 55% range), small body, tiny upper wick.
        if prev_red and dn2 >= 0.55 * rng2 and up2 <= 0.15 * rng2 and body2 <= 0.35 * rng2:
            return 'hammer'
        # Bullish engulfing: a red candle then a larger green one engulfing its body.
        if prev_red and cur_green and o2 <= x1 and x2 >= o1 and body2 > body1:
            return 'bullish_engulfing'
        # Bullish harami: a big red then a small green inside its body.
        if prev_red and cur_green and o2 >= x1 and x2 <= o1 and body2 < body1 and body1 >= 0.5 * rng1:
            return 'bullish_harami'
        # Piercing line: red then green opening below and closing past the midpoint.
        if prev_red and cur_green and o2 < x1 and mid1 < x2 < o1:
            return 'piercing'
        # Morning star (3): big red, a small-body star, then a green closing past the first's midpoint.
        if len(candles) >= 3:
            o0, x0 = o(candles[-3]), cl(candles[-3])
            if x0 < o0 and body1 <= 0.5 * abs(x0 - o0) and cur_green and x2 > (o0 + x0) / 2.0:
                return 'morning_star'
    else:
        # Shooting star: preceding green candle, long upper wick (>= 55% range), small body, tiny lower wick.
        if prev_green and up2 >= 0.55 * rng2 and dn2 <= 0.15 * rng2 and body2 <= 0.35 * rng2:
            return 'shooting_star'
        if prev_green and cur_red and o2 >= x1 and x2 <= o1 and body2 > body1:
            return 'bearish_engulfing'
        if prev_green and cur_red and o2 <= x1 and x2 >= o1 and body2 < body1 and body1 >= 0.5 * rng1:
            return 'bearish_harami'
        # Dark cloud cover: green then red opening above and closing past the midpoint.
        if prev_green and cur_red and o2 > x1 and o1 < x2 < mid1:
            return 'dark_cloud'
        if len(candles) >= 3:
            o0, x0 = o(candles[-3]), cl(candles[-3])
            if x0 > o0 and body1 <= 0.5 * abs(x0 - o0) and cur_red and x2 < (o0 + x0) / 2.0:
                return 'evening_star'
    return None


def _detect_divergence(candles: List, k: int = 3, rsi_period: int = 14) -> Dict[str, float]:
    """
    RSI and MACD divergence against price, using the last two CONFIRMED swing
    pivots.  Bearish (-1): price higher-high but oscillator lower-high.
    Bullish (+1): price lower-low but oscillator higher-low.  Reported per
    oscillator; the gate sums them (aligned RSI+MACD divergence = ±2).
    """
    out = {'rsi': 0.0, 'macd': 0.0}
    n = len(candles)
    if n < rsi_period + 2 * k + 4:
        return out
    highs = [float(c[2]) for c in candles]
    lows  = [float(c[3]) for c in candles]
    closes = _closes(candles)
    rsi  = _rsi_series(closes, rsi_period)
    macd = _macd_line(closes)

    hi_piv = _confirmed_pivots(highs, k, True)
    lo_piv = _confirmed_pivots(lows,  k, False)
    # warmup: never compare against an un-warmed oscillator (RSI is a flat 50
    # default for the first `period` bars; MACD's slow EMA needs ~26 to settle).
    for osc_name, osc, warmup in (('rsi', rsi, rsi_period), ('macd', macd, 26)):
        vote = 0.0
        if len(hi_piv) >= 2:
            a, b = hi_piv[-2], hi_piv[-1]
            if a >= warmup and highs[b] > highs[a] and osc[b] < osc[a]:
                vote = -1.0                      # bearish divergence
        if len(lo_piv) >= 2 and vote == 0.0:
            a, b = lo_piv[-2], lo_piv[-1]
            if a >= warmup and lows[b] < lows[a] and osc[b] > osc[a]:
                vote = 1.0                       # bullish divergence
        out[osc_name] = vote
    return out


def _detect_volume_events(candles: List, window: int = 20,
                          climax_z: float = 2.0, absorb_z: float = 1.5) -> Dict[str, float]:
    """
    Volume climax + absorption on the last closed bar.

    Climax   — a volume z-score spike marks EXHAUSTION, so it points AGAINST the
               bar's own direction (blow-off top on a green bar = bearish −1;
               capitulation on a red bar = bullish +1).
    Absorption — a high-volume bar with a small body and a long rejection wick:
               large size soaked up by the opposite side.  Long upper wick that
               closes weak = sellers absorbing buyers (bearish −1); long lower
               wick that closes strong = buyers absorbing sellers (bullish +1).
    """
    out = {'climax': 0.0, 'absorption': 0.0, 'vol_z': 0.0}
    n = len(candles)
    if n < window + 1:
        return out
    vols = [float(c[5]) for c in candles]
    ref = vols[-window - 1:-1]                       # prior `window` bars (exclude last)
    mean = sum(ref) / len(ref)
    var  = sum((v - mean) ** 2 for v in ref) / len(ref)
    std  = var ** 0.5
    # Degenerate volume (flat or zero) — can't assess an anomaly, report nothing.
    if std <= 1e-9 or mean <= 0:
        return out
    z = (vols[-1] - mean) / std
    out['vol_z'] = z

    o, h, l, c = (float(candles[-1][1]), float(candles[-1][2]),
                  float(candles[-1][3]), float(candles[-1][4]))
    rng = max(h - l, 1e-9)
    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    close_pos  = (c - l) / rng                       # 0 = closed on low, 1 = on high

    if z >= climax_z:
        out['climax'] = -1.0 if c > o else (1.0 if c < o else 0.0)

    if z >= absorb_z and body <= 0.5 * rng:
        if upper_wick >= 1.5 * body and upper_wick >= 0.45 * rng and close_pos <= 0.4:
            out['absorption'] = -1.0             # bearish absorption at highs
        elif lower_wick >= 1.5 * body and lower_wick >= 0.45 * rng and close_pos >= 0.6:
            out['absorption'] = 1.0              # bullish absorption at lows
    return out
