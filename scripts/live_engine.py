#!/usr/bin/env python3
"""
live_engine.py — Aegis-1 Live Signal Engine  (Glass-Box Adaptive)
============================================================================
Loads trained XGBoost models from the model store, runs Predictor.predict_realtime()
for every tradeable symbol on a configurable interval, manages a virtual paper-trading
wallet ($10 000 default), and writes data/track_record.json which main.py WebSocket
clients consume in real time.

New in this version
-------------------
    MarketRegimeDetector  — classifies market micro-structure from result dict fields
    SignalQualityFilter   — multi-layer quality scoring before any trade is issued
    DynamicRiskEngine     — volatility-aware position sizing and ATR-projected stop/TP
    PerformanceTracker    — meta-labeling, self-healing safe-mode, per-symbol win rates

Exported for main.py
--------------------
    LiveEngine      – async engine class
    TokenConfig     – per-symbol config dataclass
    automated_setup – reads tradeable models, returns run config
"""

import asyncio
import json
import os
import sys
import time
import uuid
import warnings
# Silence the Python 3.12+ pandas DeprecationWarning about bitwise '~' on bool —
# it floods the logs (obscuring the heartbeat) with no actionable signal. The
# one real occurrence (BOS features) is fixed at source; this catches any others.
warnings.filterwarnings(
    'ignore',
    message=r".*Bitwise inversion '~' on bool is deprecated.*",
    category=DeprecationWarning,
)
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

# ── project root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.trading.gate_scorer import WeightedGateScorer
from src.trading import econ_calendar
from src.trading.trendline_channel import TrendlineChannelDetector

# ── Decision architecture: MODEL-FIRST, UWGS as confirmation ──────────────────
# The ML model (predictor.predict_realtime) is the SOLE authority for signal
# side + fire — this is the ~80%-WR decision that earlier worked. UWGS
# (src/trading/gate_scorer.py) is computed for the chart breakdown, the risk
# tier, and the four genuinely protective HARD vetoes only. It no longer picks
# the side: demoting the proven model to 14/100 points and letting a
# location-weighted composite (plus a MODEL_DISAGREES veto) override it is what
# collapsed signal quality. When True, UWGS runs in this confirmation-only role;
# False disables the UWGS computation entirely (model + hard vetoes still fire).
USE_WEIGHTED_SCORER = True

# Only these UWGS vetoes may BLOCK a model signal — the genuinely protective
# ones. Everything else UWGS reports (FAR_FROM_SR, NO_VALID_SR) becomes a RISKY
# tier downgrade so the model's signal still fires (flagged), never silenced;
# MODEL_DISAGREES is ignored outright (meaningless once the model decides).
# The scheduled-news lock is handled separately (its label is dynamic).
_HARD_VETOES = frozenset({'MODEL_DRIFT_CRITICAL', 'DEAD_MARKET', 'EXTREME_VOLATILITY'})

MODEL_STORE       = _ROOT / 'src' / 'ml' / 'model_store'

# ── Persistent runtime STATE directory ────────────────────────────────────────
# Runtime state (track record, wallet, drift/perf) is WRITTEN LIVE and must
# survive redeploys.  On an ephemeral platform (Railway/Render/Fly with no
# volume) the container filesystem is wiped on every deploy, which is why the
# track record kept resetting.  Point state at a persistent volume via
# AEGIS_STATE_DIR (e.g. a Railway Volume mounted at /data); falls back to the
# in-repo data/ dir for local dev.  Config artifacts the model LOADS
# (token_params, regime_stats, model_store) stay in the image, unaffected.
_STATE_DIR = Path(os.environ.get('AEGIS_STATE_DIR') or (_ROOT / 'data'))
try:
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    _STATE_DIR = _ROOT / 'data'

TRACK_RECORD_PATH       = _STATE_DIR / 'track_record.json'
ALPHA_TRACK_RECORD_PATH = _STATE_DIR / 'alpha_track_record.json'
_ALPHA_TIMEFRAMES       = ['15m', '30m', '4h', '1d']
_PERF_STATE_PATH  = _STATE_DIR / 'perf_state.json'
_DRIFT_STATE_PATH = _STATE_DIR / 'drift_state.json'

# ── Durable track record via Firestore ──────────────────────────────────────
# Railway's filesystem is EPHEMERAL: every git push rebuilds the image and the
# container starts on a blank disk, so a file-only track record vanishes on each
# redeploy (and the AEGIS_STATE_DIR volume only helps if it's actually mounted,
# which kept not being the case).  We therefore mirror the record to Firestore —
# an external store already wired into this project — so it survives ANY
# redeploy regardless of volume config.  All ops are best-effort: if Firestore
# is unreachable the engine silently falls back to local-file behaviour (no
# regression).  The record is only ever wiped by the explicit admin reset — a
# normal deploy never clears it.
_FS_STATE_COLLECTION = 'engine_state'
_FS_STATE_DOC        = 'track_record'
# State generation: bump this to force a ONE-TIME wipe of the durable track
# record on the next deploy. On boot the engine ignores any restored record
# whose generation != this, starting fresh — regardless of what an older engine
# wrote to Firestore in the meantime. gen 2: wipe records produced by the pre-v14
# (sell-into-support / loose-reversal) gates.
_STATE_GENERATION    = 2
# Circuit breaker: the backend Firebase project may have no Firestore database,
# in which case EVERY Firestore call fails/retries and can stall the scan loop.
# After the first failure we trip this and skip all Firestore ops thereafter,
# so a missing/unreachable datastore never blocks signal generation.
_FS_DOWN = False

def _fs_state_client():
    """Best-effort Firestore client for durable state (None if unavailable or the
    circuit breaker has tripped)."""
    if _FS_DOWN:
        return None
    try:
        import firebase_admin
        from firebase_admin import credentials as _creds, firestore as _fs
        cred_path = _ROOT / 'config' / 'serviceAccountKey.json'
        if not cred_path.exists():
            return None
        if not firebase_admin._apps:
            firebase_admin.initialize_app(_creds.Certificate(str(cred_path)))
        return _fs.client()
    except Exception:
        return None

def _fs_save_track_record(payload: dict) -> None:
    """Mirror the track record to Firestore (best-effort, never raises)."""
    try:
        db = _fs_state_client()
        if db is None:
            return
        # Firestore's per-doc limit is ~1 MB; store only the restore-critical
        # subset (the capped signals list + summary), which the wallet reads back.
        slim = {
            'signals':      payload.get('signals', []),
            'summary':      payload.get('summary', {}),
            'gate_version': payload.get('engine_version', ''),
            'generated_at': payload.get('generated_at', ''),
            'generation':   _STATE_GENERATION,
        }
        db.collection(_FS_STATE_COLLECTION).document(_FS_STATE_DOC).set(slim)
    except Exception:
        global _FS_DOWN
        _FS_DOWN = True   # trip breaker — stop retrying a broken datastore

def _fs_clear_track_record() -> None:
    """Delete the durable track record from Firestore.  Used by the admin reset so
    a deliberate wipe is NOT resurrected by the hydrate on the next redeploy."""
    try:
        db = _fs_state_client()
        if db is None:
            return
        db.collection(_FS_STATE_COLLECTION).document(_FS_STATE_DOC).delete()
    except Exception:
        pass

def _fs_load_track_record() -> Optional[dict]:
    """Fetch the durable track record from Firestore (None if absent/unreachable)."""
    global _FS_DOWN
    if _FS_DOWN:            # breaker already tripped — don't retry a broken datastore
        return None
    try:
        db = _fs_state_client()
        if db is None:
            return None
        # snap typed Any: _fs_state_client() returns the SYNC firestore client, so
        # .get() yields a DocumentSnapshot — but the stubs resolve to the AsyncClient
        # (Awaitable[DocumentSnapshot]), falsely flagging .exists/.to_dict(). This is
        # a sync call; do NOT await it.
        snap: Any = db.collection(_FS_STATE_COLLECTION).document(_FS_STATE_DOC).get()
        if not snap.exists:
            return None
        return snap.to_dict()
    except Exception:
        _FS_DOWN = True   # trip breaker — stop retrying a broken datastore
        return None

def _hydrate_track_record_from_firestore() -> None:
    """On boot, if the local track record is missing/empty (ephemeral FS after a
    redeploy), restore it from Firestore so history is never lost.  Writes the
    same file VirtualWallet._load_history reads, so restore is transparent."""
    try:
        if TRACK_RECORD_PATH.exists() and TRACK_RECORD_PATH.stat().st_size > 2:
            return  # local state already present — nothing to restore
        data = _fs_load_track_record()
        if not data or not data.get('signals'):
            return
        # Generation guard: ignore any record from an older state generation so a
        # bump wipes stale history exactly once, no matter what an older engine
        # wrote to Firestore before this deploy took over.
        if int(data.get('generation', 1)) != _STATE_GENERATION:
            print(f'[state] Firestore record is generation '
                  f'{data.get("generation", 1)} != {_STATE_GENERATION} — starting fresh (one-time wipe)')
            _fs_clear_track_record()
            return
        TRACK_RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(TRACK_RECORD_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        print(f'[state] restored {len(data.get("signals", []))} track-record entries from Firestore')
    except Exception as e:
        print(f'[state] Firestore hydrate skipped: {e}')

# Shared exchange for lightweight index-price fetches (reuses the same instance
# as Predictor once the class is loaded to avoid creating a second connection).
_spot_ex = None
_spot_ex_lock = __import__('threading').Lock()

def _fetch_spot_price(symbol: str) -> float:
    """Thread-safe single-symbol spot price fetch. Returns 0.0 on any error."""
    global _spot_ex
    try:
        import ccxt as _ccxt
        # Double-checked locking: fast path avoids lock when already initialised.
        if _spot_ex is None:
            _new = _ccxt.binance({'enableRateLimit': True, 'timeout': 8000})  # type: ignore[arg-type]
            (_new.options or {})['defaultType'] = 'spot'  # type: ignore[index]
            with _spot_ex_lock:
                if _spot_ex is None:
                    _spot_ex = _new
        with _spot_ex_lock:
            ticker = _spot_ex.fetch_ticker(symbol)
        return float(ticker.get('last') or ticker.get('close') or 0)
    except Exception:
        return 0.0


# Shared futures exchange for multi-timeframe OHLCV candle fetches.
# Binance USDM perpetuals — same symbols as the main trading fleet.
_usdm_ex      = None
_usdm_ex_lock = __import__('threading').Lock()

_perp_markets = None
_perp_markets_lock = __import__('threading').Lock()


def _usdm_perp_symbol(symbol: str) -> str:
    """'BASE/USDT' -> ccxt USDM swap notation 'BASE/USDT:USDT'."""
    return symbol if ':' in symbol else symbol.replace('/USDT', '/USDT:USDT')


def _has_usdm_perp(symbol: str) -> bool:
    """Does this symbol actually EXIST as a Binance USD-M perpetual?

    PEPE/USDT and SHIB/USDT do NOT — their perps are listed as 1000PEPE/1000SHIB
    (the token is too cheap to quote 1:1). Without this check the OHLCV fetch
    below raised BadSymbol on the perp and fell straight through to the SPOT
    book, so the engine produced a COMPLETE signal — entry, SL, TP, track-record
    row — for an instrument it does not trade, on a token the chart (which
    correctly requests the perp) cannot render at all. That is how PEPE fired a
    SHORT while its chart read "Binance API blocked or symbol not found".

    Fails OPEN (returns True) when the market list can't be loaded, so a network
    blip never benches the whole fleet.
    """
    global _perp_markets
    if _perp_markets is None:
        try:
            import ccxt as _ccxt
            _ex = _ccxt.binanceusdm({'enableRateLimit': True, 'timeout': 8000})  # type: ignore[arg-type]
            _mk = set((_ex.load_markets() or {}).keys())
            with _perp_markets_lock:
                _perp_markets = _mk
        except Exception:
            return True
    return _usdm_perp_symbol(symbol) in (_perp_markets or ())


def _fetch_ohlcv_sync(symbol: str, timeframe: str, limit: int) -> list:
    """
    Thread-safe OHLCV fetch.  Primary source is Binance USDM perpetuals; on any
    failure — network, rate-limit, or a geo-block on the futures endpoint — it
    falls back to Binance spot so lower-timeframe confirmation keeps working.
    5m/15m candle DIRECTION and level tags are effectively identical across the
    perp and spot books, so the fallback is a faithful proxy for the structure
    gate.  Returns [] only when BOTH sources fail.
    """
    global _usdm_ex, _spot_ex
    # Primary: USDM perpetuals.  ccxt's binanceusdm market id is the unified
    # SWAP notation 'BASE/USDT:USDT' — passing the plain spot symbol 'BASE/USDT'
    # raises BadSymbol and returns nothing.  This mismatch (the fleet configures
    # symbols as 'BASE/USDT') is why the structure gate NEVER received perp
    # candles and always fell through to the fail-open SKIP — it was never a
    # geo-block.  Convert exactly as the predictor does (predictor.py: futures_sym).
    try:
        import ccxt as _ccxt
        if _usdm_ex is None:
            _new = _ccxt.binanceusdm({'enableRateLimit': True, 'timeout': 8000})  # type: ignore[arg-type]
            with _usdm_ex_lock:
                if _usdm_ex is None:
                    _usdm_ex = _new
        perp_sym = _usdm_perp_symbol(symbol)
        with _usdm_ex_lock:
            candles = _usdm_ex.fetch_ohlcv(perp_sym, timeframe, limit=limit) or []
        if candles:
            return candles
    except Exception:
        pass
    # The spot fallback exists for a TRANSIENT futures failure (rate limit, geo
    # block) on a token that genuinely HAS a perp — there the two books track each
    # other and spot is a faithful proxy. It must NEVER cover for a token with no
    # perp at all: that silently SWAPS THE INSTRUMENT and manufactures a signal on
    # a market we do not trade (the PEPE case). No perp => no candles => no signal.
    if not _has_usdm_perp(symbol):
        return []
    # Fallback: Binance spot (shared instance with the index-price fetcher).
    try:
        import ccxt as _ccxt
        if _spot_ex is None:
            _new = _ccxt.binance({'enableRateLimit': True, 'timeout': 8000})  # type: ignore[arg-type]
            (_new.options or {})['defaultType'] = 'spot'  # type: ignore[index]
            with _spot_ex_lock:
                if _spot_ex is None:
                    _spot_ex = _new
        with _spot_ex_lock:
            return _spot_ex.fetch_ohlcv(symbol, timeframe, limit=limit) or []
    except Exception:
        return []


def _fetch_bids_asks_all() -> Dict[str, float]:
    """Best bid/ask book spread (%) keyed by 'BASE/USDT' for the whole USDM fleet in
    ONE call (bookTicker). Feeds the UWGS dead-market veto. Best-effort; never raises."""
    global _usdm_ex
    out: Dict[str, float] = {}
    try:
        import ccxt as _ccxt
        if _usdm_ex is None:
            _new = _ccxt.binanceusdm({'enableRateLimit': True, 'timeout': 8000})  # type: ignore[arg-type]
            with _usdm_ex_lock:
                if _usdm_ex is None:
                    _usdm_ex = _new
        with _usdm_ex_lock:
            raw = _usdm_ex.fetch_bids_asks()          # /fapi/v1/ticker/bookTicker
        items = raw.values() if isinstance(raw, dict) else (raw or [])
        for t in items:
            if not isinstance(t, dict):
                continue
            sym = str(t.get('symbol') or '')
            bid = float(t.get('bid') or 0)
            ask = float(t.get('ask') or 0)
            if sym and bid > 0 and ask > 0:
                mid = (bid + ask) / 2.0
                out[sym.split(':')[0]] = (ask - bid) / mid * 100.0 if mid > 0 else 0.0
    except Exception:
        pass
    return out


# =============================================================================
# Confirmation indicators  (pure-Python, closed-bar, non-repainting)
# =============================================================================
# These operate on a list of OHLCV candles [ts, open, high, low, close, volume]
# and are used ONLY by the post-model confirmation gate in live_engine — never
# by the ML model (the model is pinned to its saved feature_cols at inference,
# so nothing here can change a prediction).  Every function is fed CLOSED bars
# (caller drops the forming candle) and looks strictly backward, so there is no
# repainting / lookahead: a decision made now uses only data that existed now.

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


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class TokenConfig:
    symbol: str
    mode:   str = 'balanced'


@dataclass
class Position:
    symbol:          str
    direction:       str    # LONG | SHORT
    side:            str    # BUY  | SELL
    entry_price:     float
    position_value:  float  # USDT allocated
    stop_loss:       float
    signal_id:       str
    entry_time:      str
    meta_confidence: float
    atr_multiplier:  float
    atr:             float = 0.0   # ATR at entry (used for trailing stop distance)
    take_profit_1:   float = 0.0   # TP1: 0.7× ATR from entry — 20% partial close
    take_profit_2:   float = 0.0   # TP2: 1.6× ATR from entry — 20% partial + trailing-stop floor
    take_profit_3:   float = 0.0   # TP3: 2.2× ATR from entry — 20% partial close
    take_profit_4:   float = 0.0   # TP4: 3.3× ATR from entry — 20% partial close
    take_profit_5:   float = 0.0   # TP5: 4.5× ATR from entry — close remainder (RR anchor)
    signal_strength: str   = ''    # risk tier at entry: STRONG | NORMAL | RISKY
    entry_mode:      str   = ''    # structure-gate verdict detail at entry
                                   # (support_reversal / breakout_* / GATE_SKIPPED: …)
    quality_score:   float = 0.0   # SignalQualityFilter score AT ENTRY — displayed
                                   # for open positions instead of a live re-score
    gate_warnings:   list  = field(default_factory=list)  # advisory-gate ledger AT
                                   # ENTRY — keeps the chart gate breakdown complete
                                   # for open positions (rebuilt away otherwise)
    entry_support:    float = 0.0  # S/R the STRUCTURE GATE judged AT ENTRY — shown
    entry_resistance: float = 0.0  # for open positions so the chart's S/R lines
                                   # reflect the entry structure, not a live re-score
                                   # that drifts after entry (a breakdown short can
                                   # look like a naive "sell at support" otherwise)


@dataclass
class TradeRecord:
    signal_id:       str
    symbol:          str
    direction:       str
    side:            str
    entry_price:     float
    exit_price:      Optional[float]
    entry_time:      str
    close_time:      Optional[str]
    pnl_pct:         float
    pnl_usdt:        float
    outcome:         str              # OPEN | WIN | LOSS
    exit_reason:     Optional[str]
    meta_confidence: float
    position_value:  float
    signal_strength: str   = ''
    stop_loss:       float = 0.0
    take_profit_1:   float = 0.0
    take_profit_2:   float = 0.0
    take_profit_3:   float = 0.0
    take_profit_4:   float = 0.0
    take_profit_5:   float = 0.0
    atr:             float = 0.0


# =============================================================================
# Regime detection
# =============================================================================

@dataclass
class RegimeState:
    """Snapshot of the current market micro-structure for one symbol."""
    regime:              str         # one of the canonical labels below
    confidence:          float       # 0.0 – 1.0
    trade_allowed:       bool
    preferred_strategies: List[str]
    max_position_pct:    float       # fraction of balance, e.g. 0.10


# Canonical regime labels
_REGIME_TRENDING_BULL      = 'TRENDING_BULL'
_REGIME_TRENDING_BEAR      = 'TRENDING_BEAR'
_REGIME_RANGING            = 'RANGING'
_REGIME_ACCUMULATION       = 'ACCUMULATION'
_REGIME_DISTRIBUTION       = 'DISTRIBUTION'
_REGIME_VOLATILE_EXPANSION = 'VOLATILE_EXPANSION'
_REGIME_VOLATILE_COMPRESS  = 'VOLATILE_COMPRESSION'
_REGIME_LIQUIDITY_TRAP     = 'LIQUIDITY_TRAP'


class MarketRegimeDetector:
    """
    Classifies market micro-structure from the fields already present in the
    result dict produced by Predictor.predict_realtime().  No extra API calls.

    Key inputs consumed (all are present in the standard result dict):
        adx, trend_regime, volatility_regime, atr_pct, market_bias,
        funding_rate, funding_bias, oi_trend, volume_zscore, rsi,
        macd_signal, volume_strength
    """

    def detect(self, result: Dict[str, Any]) -> RegimeState:
        """Return a RegimeState from a predict_realtime result dict."""
        try:
            return self._detect(result)
        except Exception:
            # Fail-safe: return a permissive neutral regime so a bug here never
            # blocks ALL trades silently.
            return RegimeState(
                regime               = _REGIME_RANGING,
                confidence           = 0.4,
                trade_allowed        = True,
                preferred_strategies = ['TREND_FOLLOW', 'RANGE_TRADE'],
                max_position_pct     = 0.08,
            )

    def _detect(self, result: Dict[str, Any]) -> RegimeState:
        adx             = float(result.get('adx', 20.0) or 20.0)
        trend_regime    = str(result.get('trend_regime', 'RANGING') or 'RANGING')
        vol_regime      = str(result.get('volatility_regime', 'MEDIUM') or 'MEDIUM').upper()
        atr_pct         = float(result.get('atr_pct', 1.5) or 1.5)       # already × 100
        market_bias     = str(result.get('market_bias', 'NEUTRAL') or 'NEUTRAL').upper()
        funding_bias    = str(result.get('funding_bias', 'NEUTRAL') or 'NEUTRAL').upper()
        oi_trend        = str(result.get('oi_trend', 'STABLE') or 'STABLE').upper()
        vol_zscore      = float(result.get('volume_zscore', 0.0) or 0.0)
        rsi             = float(result.get('rsi', 50.0) or 50.0)
        macd_signal     = str(result.get('macd_signal', 'NEUTRAL') or 'NEUTRAL').upper()
        volume_strength = str(result.get('volume_strength', 'AVERAGE') or 'AVERAGE').upper()

        is_trending  = adx > 25
        is_ranging   = adx < 20
        is_volatile  = (vol_regime == 'HIGH' or atr_pct > 3.0)
        is_quiet     = (vol_regime == 'LOW'  and atr_pct < 1.2)
        is_bull_momentum = (rsi > 52) or (macd_signal == 'BULLISH') or ('UP' in trend_regime) or (market_bias == 'BULLISH')
        is_bear_momentum = (rsi < 48) or (macd_signal == 'BEARISH') or ('DOWN' in trend_regime) or (market_bias == 'BEARISH')

        is_bullish   = is_bull_momentum and not (rsi < 45 and market_bias == 'BEARISH')
        is_bearish   = is_bear_momentum and not (rsi > 55 and market_bias == 'BULLISH')
        low_volume   = (volume_strength == 'BELOW_AVERAGE' or vol_zscore < -0.5)
        high_oi      = oi_trend == 'INCREASING'
        low_oi       = oi_trend == 'DECREASING'
        longs_paying = funding_bias == 'LONGS_PAYING'
        shorts_paying= funding_bias == 'SHORTS_PAYING'

        # ── 1. Liquidity trap: low volume, choppy, no trending structure ─────
        if low_volume and is_ranging and is_quiet:
            return RegimeState(
                regime               = _REGIME_LIQUIDITY_TRAP,
                confidence           = 0.75,
                trade_allowed        = False,
                preferred_strategies = [],
                max_position_pct     = 0.0,
            )

        # ── 2. Volatile expansion ─────────────────────────────────────────────
        if is_volatile and atr_pct > 4.0:
            conf = min(0.9, 0.6 + (atr_pct - 4.0) * 0.05)
            return RegimeState(
                regime               = _REGIME_VOLATILE_EXPANSION,
                confidence           = round(conf, 3),
                trade_allowed        = True,
                preferred_strategies = ['BREAKOUT', 'MOMENTUM'],
                max_position_pct     = 0.06,  # reduced size in expansion
            )

        # ── 3. Volatile compression: quiet market after expansion ─────────────
        if is_quiet and not is_trending:
            return RegimeState(
                regime               = _REGIME_VOLATILE_COMPRESS,
                confidence           = 0.65,
                trade_allowed        = True,
                preferred_strategies = ['RANGE_TRADE', 'MEAN_REVERT'],
                max_position_pct     = 0.07,
            )

        # ── 4. Accumulation: ranging + increasing OI + shorts paying ─────────
        if is_ranging and high_oi and shorts_paying and not is_bearish:
            conf = 0.55 + (0.15 if rsi < 55 else 0.0) + (0.10 if vol_zscore > 0.5 else 0.0)
            return RegimeState(
                regime               = _REGIME_ACCUMULATION,
                confidence           = round(min(conf, 0.85), 3),
                trade_allowed        = True,
                preferred_strategies = ['RANGE_BUY', 'BREAKOUT_LONG'],
                max_position_pct     = 0.10,
            )

        # ── 5. Distribution: ranging + increasing OI + longs paying ──────────
        if is_ranging and high_oi and longs_paying and not is_bullish:
            conf = 0.55 + (0.15 if rsi > 45 else 0.0) + (0.10 if vol_zscore > 0.5 else 0.0)
            return RegimeState(
                regime               = _REGIME_DISTRIBUTION,
                confidence           = round(min(conf, 0.85), 3),
                trade_allowed        = True,
                preferred_strategies = ['RANGE_SELL', 'BREAKOUT_SHORT'],
                max_position_pct     = 0.10,
            )

        # ── 6. Trending bull ──────────────────────────────────────────────────
        if is_trending and is_bullish:
            conf = 0.60
            conf += 0.10 if adx > 35 else 0.0
            conf += 0.10 if macd_signal == 'BULLISH' else 0.0
            conf += 0.10 if vol_zscore > 1.0 else 0.0
            conf += 0.10 if 'UP' in trend_regime else 0.0
            return RegimeState(
                regime               = _REGIME_TRENDING_BULL,
                confidence           = round(min(conf, 0.95), 3),
                trade_allowed        = True,
                preferred_strategies = ['TREND_FOLLOW', 'MOMENTUM', 'PULLBACK_LONG'],
                max_position_pct     = 0.13,
            )

        # ── 7. Trending bear ──────────────────────────────────────────────────
        if is_trending and is_bearish:
            conf = 0.60
            conf += 0.10 if adx > 35 else 0.0
            conf += 0.10 if macd_signal == 'BEARISH' else 0.0
            conf += 0.10 if vol_zscore > 1.0 else 0.0
            conf += 0.10 if 'DOWN' in trend_regime else 0.0
            return RegimeState(
                regime               = _REGIME_TRENDING_BEAR,
                confidence           = round(min(conf, 0.95), 3),
                trade_allowed        = True,
                preferred_strategies = ['TREND_FOLLOW', 'MOMENTUM', 'PULLBACK_SHORT'],
                max_position_pct     = 0.13,
            )

        # ── 8. Default: ranging / neutral ─────────────────────────────────────
        return RegimeState(
            regime               = _REGIME_RANGING,
            confidence           = 0.50,
            trade_allowed        = True,
            preferred_strategies = ['RANGE_TRADE', 'MEAN_REVERT', 'SUPPORT_BUY'],
            max_position_pct     = 0.08,
        )


# =============================================================================
# Signal quality filter
# =============================================================================

class SignalQualityFilter:
    """
    Multi-layer quality scoring.  Uses only fields from the result dict — no
    extra API calls.

    score_signal() → (float quality 0-100, list[str] reasons)
    is_fake_breakout() → bool
    """

    MIN_QUALITY_SCORE = 60.0  # v43: edge floor 55->60 — cut the coin-flip signals (biggest WR lever) without gutting count

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
                pass;                                                      # neutral — no bonus
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
        if regime.regime == _REGIME_LIQUIDITY_TRAP:
            score -= 20; reasons.append('liquidity_trap_penalty')

        # -15: ranging market — no directional edge, signals are noise.  Skipped
        # for a reversal at the range extreme: SELL at range resistance / BUY at
        # range support IS the high-probability range trade, not noise.
        if regime.regime == _REGIME_RANGING and not (is_reversal_strict or is_structural_reversal):
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

            # -8: LSTM detects sequence exhaustion (momentum likely decaying)
            elif lstm_cont < 0.32:
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
            elif (side == 'BUY'  and macd_sig == 'BEARISH') or \
                 (side == 'SELL' and macd_sig == 'BULLISH'):
                score -= 12; reasons.append(f'macd_conflict({macd_sig})')

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


# =============================================================================
# Dynamic risk engine
# =============================================================================

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
    RISKY_SL_CAP_ATR     = 1.2   # SL cap in ATR for RISKY-tier signals

    # TP ladder — COMPRESSED into a reachable region (2026-07-04).  The old
    # ladder (2.8 / 4.5 / 6.5 / 9.5) left a huge TP2→TP3 gap and put TP3-TP5 so
    # far out they almost never filled: price hit TP1+TP2 (~1.5%) and reversed
    # long before 4.5×ATR.  Spacing is now even and inside a plausible swing so
    # partials actually bank at each level.  SL stays 1.8×ATR; RR is validated to
    # TP5 (4.5/1.8 = 2.5) so trade acceptance is unchanged — only the interior
    # rungs moved closer.
    TP1_MULTIPLIER    = 0.7    # 20 % partial close — early lock, nudged up from 0.55 so the first bank isn't tiny
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
    # Front-loaded onto the two "significant objective" targets (TP2/TP3), 20 %
    # runner rides TP5's trailing exit.
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

        Take Profit ladder (R = risk leg; Range = resistance−support)
        ------------------------------------------------------------
          TP1  = 1R                              (1:1, quick partial)
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
            # Hybrid SL: just below support + buffer, clamped to [floor, cap].
            risk = ((price - support) + buf) if (0 < support < price) else cap
            risk = max(floor, min(risk, cap))
            sl   = price - risk
            # Structural strong target = the major resistance (else an R-multiple).
            tp3  = resistance if resistance > price else price + 3.5 * risk
            rng  = (resistance - support) if (0 < support < resistance) else (tp3 - price)
            tp1  = price + 1.0 * risk
            tp2  = price + 2.0 * risk
            tp4  = (support + 1.618 * rng) if support > 0 else price + 5.0 * risk
            tp5  = (support + 2.618 * rng) if support > 0 else price + 7.0 * risk
            # Force strictly ascending, ≥0.3R apart.
            tp2 = max(tp2, tp1 + 0.3 * risk)
            tp3 = max(tp3, tp2 + 0.3 * risk)
            tp4 = max(tp4, tp3 + 0.3 * risk)
            tp5 = max(tp5, tp4 + 0.3 * risk)
            # RR to the REAL structural target (not the guard-extended tp3), so a
            # cramped setup whose resistance is too close is honestly rejected.
            reward = (resistance - price) if resistance > price else 3.5 * risk
        else:  # SELL / SHORT
            risk = ((resistance - price) + buf) if (resistance > price) else cap
            risk = max(floor, min(risk, cap))
            sl   = price + risk
            tp3  = support if 0 < support < price else price - 3.5 * risk
            rng  = (resistance - support) if (0 < support < resistance) else (price - tp3)
            tp1  = price - 1.0 * risk
            tp2  = price - 2.0 * risk
            tp4  = (resistance - 1.618 * rng) if resistance > 0 else price - 5.0 * risk
            tp5  = (resistance - 2.618 * rng) if resistance > 0 else price - 7.0 * risk
            # Force strictly descending, ≥0.3R apart.
            tp2 = min(tp2, tp1 - 0.3 * risk)
            tp3 = min(tp3, tp2 - 0.3 * risk)
            tp4 = min(tp4, tp3 - 0.3 * risk)
            tp5 = min(tp5, tp4 - 0.3 * risk)
            # RR to the REAL structural target (not the guard-extended tp3).
            reward = (price - support) if 0 < support < price else 3.5 * risk

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


# =============================================================================
# Performance tracker  (meta-labeling + self-healing)
# =============================================================================

@dataclass
class _OutcomeRecord:
    symbol:        str
    regime:        str
    outcome:       str    # 'WIN' | 'LOSS'
    pnl_pct:       float
    quality_score: float
    ts:            float  = field(default_factory=time.time)


class PerformanceTracker:
    """
    Lightweight in-memory performance tracking with self-healing safe-mode.

    Tracks:
        - Per-symbol recent outcomes (last 20 trades per symbol)
        - Global recent outcomes (last 30 trades) for safe-mode detection
        - Per-regime win/loss tallies

    No disk persistence — resets on restart intentionally so safe-mode doesn't
    carry stale data from a different market session.
    """

    SYMBOL_WINDOW  = 20   # how many recent trades to consider per symbol
    GLOBAL_WINDOW  = 30   # global window for safe-mode check
    SAFE_MODE_LOSS = 3    # consecutive global losses to activate safe-mode
    REDUCE_STREAK  = 3    # consecutive per-symbol losses to halve position

    def __init__(self) -> None:
        self._by_symbol: Dict[str, Deque[_OutcomeRecord]] = {}
        self._global:    Deque[_OutcomeRecord]             = deque(maxlen=self.GLOBAL_WINDOW)

    def record_outcome(
        self,
        symbol:        str,
        regime:        str,
        outcome:       str,
        pnl_pct:       float,
        quality_score: float,
    ) -> None:
        """Store a completed trade outcome and immediately persist to disk."""
        rec = _OutcomeRecord(
            symbol        = symbol,
            regime        = regime,
            outcome       = outcome,
            pnl_pct       = pnl_pct,
            quality_score = quality_score,
        )
        if symbol not in self._by_symbol:
            self._by_symbol[symbol] = deque(maxlen=self.SYMBOL_WINDOW)
        self._by_symbol[symbol].append(rec)
        self._global.append(rec)
        self.save_state()

    def get_symbol_win_rate(self, symbol: str) -> float:
        """Win rate from the most recent SYMBOL_WINDOW closed trades on this symbol."""
        history = list(self._by_symbol.get(symbol, []))
        if not history:
            return 0.50   # assume neutral when no data
        wins = sum(1 for r in history if r.outcome == 'WIN')
        return round(wins / len(history), 3)

    def should_reduce_exposure(self, symbol: str) -> bool:
        """
        Return True if the last REDUCE_STREAK trades on this symbol were all losses.
        Signals that the model is underperforming on this token — halve position size.
        """
        history = list(self._by_symbol.get(symbol, []))
        if len(history) < self.REDUCE_STREAK:
            return False
        return all(r.outcome == 'LOSS' for r in list(history)[-self.REDUCE_STREAK:])

    def safe_mode_active(self) -> bool:
        """
        Return True if the last SAFE_MODE_LOSS global trades were all losses.
        When in safe-mode, the edge-score floor is raised from 70 → 80 (Gate 3.5).
        """
        recent = list(self._global)
        if len(recent) < self.SAFE_MODE_LOSS:
            return False
        return all(r.outcome == 'LOSS' for r in recent[-self.SAFE_MODE_LOSS:])

    def get_performance_summary(self) -> Dict[str, Any]:
        """Return a summary dict suitable for dashboard display."""
        total   = len(self._global)
        wins    = sum(1 for r in self._global if r.outcome == 'WIN')
        losses  = total - wins
        pnls    = [r.pnl_pct for r in self._global]
        avg_pnl = round(sum(pnls) / len(pnls), 3) if pnls else 0.0
        return {
            'total_recent':    total,
            'wins':            wins,
            'losses':          losses,
            'win_rate':        round(wins / total, 3) if total else 0.0,
            'avg_pnl_pct':     avg_pnl,
            'safe_mode':       self.safe_mode_active(),
            'per_symbol_wr':   {
                sym: self.get_symbol_win_rate(sym)
                for sym in self._by_symbol
            },
        }

    # ── Disk persistence — survives server restarts ───────────────────────────

    def save_state(self) -> None:
        """Persist recent outcomes so safe-mode and reduce-exposure survive restarts."""
        try:
            import os as _os
            payload = {
                'saved_at': datetime.now(timezone.utc).isoformat(),
                'global': [
                    {'symbol': r.symbol, 'regime': r.regime, 'outcome': r.outcome,
                     'pnl_pct': r.pnl_pct, 'quality_score': r.quality_score, 'ts': r.ts}
                    for r in self._global
                ],
                'by_symbol': {
                    sym: [
                        {'symbol': r.symbol, 'regime': r.regime, 'outcome': r.outcome,
                         'pnl_pct': r.pnl_pct, 'quality_score': r.quality_score, 'ts': r.ts}
                        for r in hist
                    ]
                    for sym, hist in self._by_symbol.items()
                },
            }
            _PERF_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = _PERF_STATE_PATH.with_suffix('.tmp')
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(payload, f)
            _os.replace(tmp, _PERF_STATE_PATH)
        except Exception as e:
            print(f'[PerformanceTracker] save_state failed: {e}')

    def load_state(self) -> None:
        """Restore recent outcome history from disk."""
        if not _PERF_STATE_PATH.exists():
            return
        try:
            with open(_PERF_STATE_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)

            now = time.time()
            max_age = 7 * 24 * 3600  # only restore outcomes from last 7 days

            for raw in data.get('global', []):
                if now - float(raw.get('ts', 0)) > max_age:
                    continue
                self._global.append(_OutcomeRecord(
                    symbol=raw['symbol'], regime=raw['regime'],
                    outcome=raw['outcome'], pnl_pct=float(raw['pnl_pct']),
                    quality_score=float(raw['quality_score']), ts=float(raw['ts']),
                ))

            for sym, raws in data.get('by_symbol', {}).items():
                self._by_symbol[sym] = deque(maxlen=self.SYMBOL_WINDOW)
                for raw in raws:
                    if now - float(raw.get('ts', 0)) > max_age:
                        continue
                    self._by_symbol[sym].append(_OutcomeRecord(
                        symbol=raw['symbol'], regime=raw['regime'],
                        outcome=raw['outcome'], pnl_pct=float(raw['pnl_pct']),
                        quality_score=float(raw['quality_score']), ts=float(raw['ts']),
                    ))

            n = len(self._global)
            if n:
                print(f'[PerformanceTracker] Restored {n} recent outcomes from disk.')
                if self.safe_mode_active():
                    print('[PerformanceTracker] WARNING: safe-mode is still active from last session.')
        except Exception as e:
            print(f'[PerformanceTracker] load_state failed (starting fresh): {e}')


# =============================================================================
# Drift monitor  — benchmark-aware precision tracking
# =============================================================================

class DriftMonitor:
    """
    Compares live win rate against the training benchmark precision stored in
    each token's *_meta.json sidecar.  This closes the most important gap in
    the current system: knowing WHEN a model has degraded, not just THAT it lost
    N trades in a row.

    Severity levels
    ---------------
    OK       — live win rate within 10 pp of benchmark
    WARNING  — live win rate 10–20 pp below benchmark (add confidence penalty)
    CRITICAL — live win rate > 20 pp below benchmark (block new entries)
    UNKNOWN  — not enough live trades yet to judge (< MIN_SAMPLE)

    The benchmark is loaded from meta.json at engine startup.  If the meta file
    is missing or has no precision figure, the symbol is given a neutral 0.60
    default — conservative enough not to create false CRITICAL states.
    """

    MIN_SAMPLE       = 8     # need at least 8 live trades before issuing a verdict
    WARNING_DROP_PP  = 10    # 10 percentage-point drop triggers WARNING
    CRITICAL_DROP_PP = 20    # 20 pp drop triggers CRITICAL

    def __init__(self) -> None:
        self._benchmarks:    Dict[str, float] = {}    # symbol → training precision
        self._live_window:   Dict[str, Deque[bool]] = {}   # symbol → rolling outcomes
        self._loaded = False

    # ── Benchmark loading ─────────────────────────────────────────────────────

    def load_benchmarks(self) -> None:
        """Read per-token training precision from model_store meta.json files."""
        loaded = 0
        if not MODEL_STORE.exists():
            return
        for meta_file in MODEL_STORE.glob('*_meta.json'):
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                sym = meta.get('symbol', '')
                if not sym:
                    continue
                # Prefer directional precision (honest gate), then signal precision, then OOF estimate
                ht = meta.get('holdout_trading', {})
                prec = (
                    float(ht.get('directional_precision', 0))
                    or float(ht.get('signal_precision', 0))
                    or float(meta.get('dev_estimate', {}).get('precision', 0))
                    or 0.60
                )
                self._benchmarks[sym] = max(prec, 0.50)  # floor at 50% (random baseline)
                loaded += 1
            except Exception:
                pass
        self._loaded = True
        print(f'[DriftMonitor] Loaded benchmarks for {loaded} symbols.')

    # ── Live outcome recording ────────────────────────────────────────────────

    def record(self, symbol: str, outcome: str) -> None:
        """Record a WIN or LOSS outcome for drift tracking."""
        if symbol not in self._live_window:
            self._live_window[symbol] = deque(maxlen=30)
        self._live_window[symbol].append(outcome == 'WIN')

    # ── State persistence (survives restarts) ─────────────────────────────────

    def save_state(self) -> None:
        try:
            import os as _os
            payload = {
                'saved_at': datetime.now(timezone.utc).isoformat(),
                'windows': {
                    sym: [int(b) for b in hist]
                    for sym, hist in self._live_window.items()
                },
            }
            _DRIFT_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = _DRIFT_STATE_PATH.with_suffix('.tmp')
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(payload, f)
            _os.replace(tmp, _DRIFT_STATE_PATH)
        except Exception:
            pass

    def load_state(self) -> None:
        if not _DRIFT_STATE_PATH.exists():
            return
        try:
            with open(_DRIFT_STATE_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for sym, wins in data.get('windows', {}).items():
                self._live_window[sym] = deque([bool(w) for w in wins], maxlen=30)
        except Exception:
            pass

    # ── Drift scoring ─────────────────────────────────────────────────────────

    def _live_win_rate(self, symbol: str) -> Optional[float]:
        hist = list(self._live_window.get(symbol, []))
        if len(hist) < self.MIN_SAMPLE:
            return None
        return round(sum(hist) / len(hist), 3)

    def severity(self, symbol: str) -> str:
        """Return 'OK', 'WARNING', 'CRITICAL', or 'UNKNOWN'."""
        live_wr = self._live_win_rate(symbol)
        if live_wr is None:
            return 'UNKNOWN'
        benchmark = self._benchmarks.get(symbol, 0.60)
        drop_pp   = (benchmark - live_wr) * 100.0
        if drop_pp >= self.CRITICAL_DROP_PP:
            return 'CRITICAL'
        if drop_pp >= self.WARNING_DROP_PP:
            return 'WARNING'
        return 'OK'

    def confidence_penalty(self, symbol: str) -> float:
        """
        Extra confidence threshold added when the model is drifting.
        This raises the bar for new entries proportionally to how far
        the live win rate has fallen below the training benchmark.
        """
        live_wr = self._live_win_rate(symbol)
        if live_wr is None:
            return 0.0
        benchmark = self._benchmarks.get(symbol, 0.60)
        drop = max(0.0, benchmark - live_wr)
        # Penalty: 0.03 per 10 pp of drop, capped at 0.10
        return round(min(drop * 0.3, 0.10), 3)

    def is_blocked(self, symbol: str) -> bool:
        """True when drift is CRITICAL — new entries are suppressed entirely."""
        return self.severity(symbol) == 'CRITICAL'

    def get_summary(self) -> Dict[str, Any]:
        summary = {}
        for sym in self._live_window:
            live_wr = self._live_win_rate(sym)
            benchmark = self._benchmarks.get(sym, 0.60)
            summary[sym] = {
                'benchmark': benchmark,
                'live_wr':   live_wr,
                'severity':  self.severity(sym),
                'n_trades':  len(self._live_window[sym]),
            }
        return summary


# =============================================================================
# Portfolio guard  — correlation-aware position limits
# =============================================================================

class PortfolioGuard:
    """
    Prevents the engine from opening multiple correlated positions simultaneously.

    Problem it solves:
    ==================
    BTC, ETH, SOL, AVAX, BNB almost always move together.  When the market turns,
    the engine can fire BUY on all five within the same scan cycle.  This creates
    5× correlated exposure — not 5 independent bets.  A single BTC flush wipes
    all five positions at once.

    Solution:
    =========
    Tokens are grouped into correlation clusters.  Within each cluster, only
    MAX_PER_CLUSTER positions can be open at the same time.  The highest-quality
    signal in the cluster wins.

    Portfolio-wide limits:
    - MAX_OPEN_TOTAL:       hard limit on simultaneous open positions
    - MAX_CAPITAL_DEPLOYED: max fraction of wallet balance in open positions

    Clusters are defined statically (good enough — crypto correlations are
    structurally stable within tiers).  Dynamic clustering via PCA on live
    price returns is a Phase 2 improvement.
    """

    # v74 — the caps are BINDING again (user decision 2026-07-19). The old
    # rationale ("independent recommendations, caps only mute opportunities")
    # died on contact with a real session: 16 simultaneous positions opened
    # into one tape — 8 alt shorts that were ONE bet expressed eight times —
    # and bled together on a BTC bounce. Crypto alts run ~0.8 correlation to
    # BTC intraday, so an uncapped book is levered beta, not breadth. The
    # public track record IS the product; it must be built from the best few
    # setups, not everything the model will sign. Capital cap stays open
    # (position values are notional per-signal).
    MAX_PER_CLUSTER    = 2     # one thesis, max two expressions per cluster
    MAX_OPEN_TOTAL     = 5     # the book holds the 5 best setups, not all 58
    MAX_CAPITAL_PCT    = 100.0 # no capital cap — position values are notional per-signal

    # Static correlation clusters (tightest first)
    _CLUSTERS: Dict[str, List[str]] = {
        'MAJORS':    ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
        'L1_FAST':   ['SOL/USDT', 'AVAX/USDT', 'APT/USDT', 'SUI/USDT', 'NEAR/USDT'],
        'L2':        ['ARB/USDT', 'OP/USDT', 'STRK/USDT', 'IMX/USDT', 'ZK/USDT'],
        'DEFI_BLUE': ['AAVE/USDT', 'UNI/USDT', 'CRV/USDT', 'COMP/USDT', 'LDO/USDT'],
        'AI_INFRA':  ['FET/USDT', 'TAO/USDT', 'ARKM/USDT', 'GRT/USDT'],
        'MEME':      ['PEPE/USDT', 'WIF/USDT', 'DOGE/USDT', 'SHIB/USDT', 'BONK/USDT',
                      'FLOKI/USDT', 'BOME/USDT', 'BRETT/USDT'],
        'XRP_ALTS':  ['XRP/USDT', 'XLM/USDT', 'ADA/USDT', 'TRX/USDT'],
        'LAYER1_MID': ['DOT/USDT', 'ATOM/USDT', 'ALGO/USDT', 'EGLD/USDT',
                       'ICP/USDT', 'FIL/USDT', 'HBAR/USDT'],
    }

    def __init__(self) -> None:
        # Build reverse map: symbol → cluster name
        self._sym_to_cluster: Dict[str, str] = {}
        for cluster, syms in self._CLUSTERS.items():
            for s in syms:
                self._sym_to_cluster[s] = cluster
        # Runtime state (populated from wallet)
        self._open_positions: Dict[str, str] = {}   # symbol → direction ('LONG'/'SHORT')
        self._deployed_usdt:  float          = 0.0  # sum of all open position_values

    def sync_from_wallet(self, open_positions: Dict[str, Any]) -> None:
        """Sync open positions from the VirtualWallet on every scan cycle."""
        self._open_positions = {
            sym: pos.direction
            for sym, pos in open_positions.items()
        }
        self._deployed_usdt = sum(pos.position_value for pos in open_positions.values())

    def _cluster_open_count(self, cluster: str) -> int:
        cluster_syms = set(self._CLUSTERS.get(cluster, []))
        return sum(1 for sym in self._open_positions if sym in cluster_syms)

    def _total_open(self) -> int:
        return len(self._open_positions)

    def can_open(self, symbol: str, wallet_balance: float,
                 position_value: float) -> Tuple[bool, str]:
        """
        Returns (allowed, reason).
        Reason is a human-readable string for the log — critical for debugging.
        """
        # Hard cap on total open positions
        if self._total_open() >= self.MAX_OPEN_TOTAL:
            return False, f'MAX_OPEN_TOTAL={self.MAX_OPEN_TOTAL} reached'

        # Cluster correlation cap
        cluster = self._sym_to_cluster.get(symbol)
        if cluster:
            n_in_cluster = self._cluster_open_count(cluster)
            if n_in_cluster >= self.MAX_PER_CLUSTER:
                open_in_cluster = [s for s in self._open_positions
                                   if self._sym_to_cluster.get(s) == cluster]
                return (False,
                        f'CLUSTER_CAP: {cluster} already has {n_in_cluster} '
                        f'open ({", ".join(open_in_cluster)})')

        # Capital deployment cap — sum actual deployed + new position vs balance.
        if wallet_balance > 0:
            deployed_pct = (self._deployed_usdt + position_value) / wallet_balance
            if deployed_pct >= self.MAX_CAPITAL_PCT:
                return (False,
                        f'CAPITAL_CAP: {deployed_pct:.0%} would be deployed '
                        f'(max {self.MAX_CAPITAL_PCT:.0%})')

        return True, 'OK'

    def get_summary(self) -> Dict[str, Any]:
        cluster_counts: Dict[str, int] = {}
        for sym in self._open_positions:
            c = self._sym_to_cluster.get(sym, 'OTHER')
            cluster_counts[c] = cluster_counts.get(c, 0) + 1
        return {
            'open_total':    self._total_open(),
            'max_allowed':   self.MAX_OPEN_TOTAL,
            'cluster_counts': cluster_counts,
        }


# =============================================================================
# Virtual wallet  (paper trading, $10 000 default)
# =============================================================================

class VirtualWallet:
    """Risk 10 % of balance per trade, capped at max_position_usdt."""

    def __init__(self, initial_capital: float, max_position_usdt: float = 1_000.0,
                 track_record_path: Optional[Path] = None):
        self.initial_capital    = initial_capital
        self.balance            = initial_capital
        self.max_position_usdt  = max_position_usdt
        self._track_record_path = track_record_path or TRACK_RECORD_PATH
        self.open_positions:    Dict[str, Position]  = {}
        self.trade_history:     List[TradeRecord]     = []
        self._load_history()

    def _load_history(self) -> None:
        """Restore closed trade history, balance, and open positions from disk on restart."""
        if not self._track_record_path.exists():
            return
        try:
            with open(self._track_record_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            signals   = data.get('signals', [])
            closed    = [s for s in signals if s.get('outcome') in ('WIN', 'LOSS')]
            open_sigs = [s for s in signals if s.get('outcome') == 'OPEN']

            # ── Closed trades → trade_history + adjust balance ─────────────────
            restored = 0
            seen_ids: set = set()
            for s in closed:
                sid = s.get('signal_id', '')
                if not sid or sid in seen_ids:
                    continue
                seen_ids.add(sid)
                try:
                    rec = TradeRecord(
                        signal_id       = sid,
                        symbol          = s['symbol'],
                        direction       = s.get('direction', 'LONG'),
                        side            = s.get('side', 'BUY'),
                        entry_price     = float(s.get('entry_price', 0)),
                        exit_price      = float(s['exit_price']) if s.get('exit_price') else None,
                        entry_time      = s.get('entry_time', ''),
                        close_time      = s.get('close_time'),
                        pnl_pct         = float(s.get('pnl_pct', 0)),
                        pnl_usdt        = float(s.get('pnl_usdt', 0)),
                        outcome         = s['outcome'],
                        exit_reason     = s.get('exit_reason'),
                        meta_confidence = float(s.get('meta_confidence', 0)),
                        position_value  = float(s.get('position_value', 0)),
                        signal_strength = s.get('signal_strength', ''),
                        stop_loss       = float(s.get('stop_loss', 0)),
                        take_profit_1   = float(s.get('take_profit_1', 0)),
                        take_profit_2   = float(s.get('take_profit_2', 0)),
                        take_profit_3   = float(s.get('take_profit_3', 0)),
                        take_profit_4   = float(s.get('take_profit_4', 0)),
                        take_profit_5   = float(s.get('take_profit_5', 0)),
                        atr             = float(s.get('atr', 0)),
                    )
                    self.trade_history.append(rec)
                    self.balance += float(s.get('pnl_usdt', 0))
                    restored += 1
                except Exception:
                    continue
            if restored:
                print(f'[VirtualWallet] Restored {restored} closed trades from disk. '
                      f'Balance: ${self.balance:,.2f}')

            # ── Open positions → restore into wallet so portfolio guard is correct ─
            open_restored = 0
            for s in open_sigs:
                sym = s.get('symbol', '')
                if not sym or sym in self.open_positions:
                    continue
                try:
                    side = s.get('side', '') or ('BUY' if s.get('direction') == 'LONG' else 'SELL')
                    pos  = Position(
                        symbol          = sym,
                        direction       = s.get('direction', 'LONG' if side == 'BUY' else 'SHORT'),
                        side            = side,
                        entry_price     = float(s.get('entry_price', 0)),
                        position_value  = float(s.get('position_value', 0)),
                        stop_loss       = float(s.get('stop_loss', 0)),
                        signal_id       = s.get('signal_id', ''),
                        entry_time      = s.get('entry_time', ''),
                        meta_confidence = float(s.get('meta_confidence', 0)),
                        atr_multiplier  = float(s.get('atr_multiplier', 1.2)),
                        atr             = float(s.get('atr', 0)),
                        take_profit_1   = float(s.get('take_profit_1', 0)),
                        take_profit_2   = float(s.get('take_profit_2', 0)),
                        take_profit_3   = float(s.get('take_profit_3', 0)),
                        take_profit_4   = float(s.get('take_profit_4', 0)),
                        take_profit_5   = float(s.get('take_profit_5', 0)),
                        signal_strength = s.get('signal_strength', ''),
                        entry_mode      = s.get('entry_mode', ''),
                    )
                    self.open_positions[sym] = pos
                    open_restored += 1
                except Exception:
                    continue
            if open_restored:
                print(f'[VirtualWallet] Restored {open_restored} open positions from disk.')
        except Exception as e:
            print(f'[VirtualWallet] History load error (starting fresh): {e}')

    def position_size(self) -> float:
        return min(self.balance * 0.10, self.max_position_usdt)

    def open_trade(self, pos: Position) -> None:
        self.open_positions[pos.symbol] = pos

    def close_trade(self, symbol: str, exit_price: float,
                    exit_reason: str) -> Optional[TradeRecord]:
        """Full close — removes the position and books PnL on remaining size."""
        pos = self.open_positions.pop(symbol, None)
        if pos is None:
            return None

        if pos.direction == 'LONG':
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100
        else:
            pnl_pct = (pos.entry_price - exit_price) / pos.entry_price * 100

        pnl_usdt = round(pos.position_value * pnl_pct / 100, 2)
        self.balance += pnl_usdt

        # Whole-trade outcome: include profits already banked by TP partial
        # closes of the same position (same signal_id).  A trade that banked
        # TP1–TP4 and then closed its remainder at break-even is a WIN — the
        # final slice alone would mislabel it a LOSS.
        _banked_usdt = sum(
            t.pnl_usdt for t in self.trade_history
            if t.signal_id == pos.signal_id and 'PARTIAL' in (t.exit_reason or '')
        )
        _trade_outcome = 'WIN' if (pnl_usdt + _banked_usdt) > 0 else 'LOSS'

        rec = TradeRecord(
            signal_id       = pos.signal_id,
            symbol          = symbol,
            direction       = pos.direction,
            side            = pos.side,
            entry_price     = pos.entry_price,
            exit_price      = round(exit_price, 8),
            entry_time      = pos.entry_time,
            close_time      = datetime.now(timezone.utc).isoformat(),
            pnl_pct         = round(pnl_pct, 3),
            pnl_usdt        = pnl_usdt,
            outcome         = _trade_outcome,
            exit_reason     = exit_reason,
            meta_confidence = pos.meta_confidence,
            position_value  = pos.position_value,
            signal_strength = pos.signal_strength,
            stop_loss       = pos.stop_loss,
            take_profit_1   = pos.take_profit_1,
            take_profit_2   = pos.take_profit_2,
            take_profit_3   = pos.take_profit_3,
            take_profit_4   = pos.take_profit_4,
            take_profit_5   = pos.take_profit_5,
            atr             = pos.atr,
        )
        self.trade_history.append(rec)
        return rec

    def partial_close_trade(
        self,
        symbol:       str,
        exit_price:   float,
        exit_reason:  str,
        close_pct:    float,   # fraction of current position_value to close, e.g. 0.20
    ) -> Optional[TradeRecord]:
        """Partial close — reduces position_value, books PnL on the closed slice.

        The position stays open in open_positions so subsequent TP levels can
        still fire.  close_pct is applied to the *current* position_value (which
        shrinks each time this is called) so each call correctly closes the
        intended fraction of the original allocation.
        """
        pos = self.open_positions.get(symbol)
        if pos is None:
            return None

        close_value = round(pos.position_value * close_pct, 2)
        if close_value <= 0:
            return None

        if pos.direction == 'LONG':
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100
        else:
            pnl_pct = (pos.entry_price - exit_price) / pos.entry_price * 100

        pnl_usdt = round(close_value * pnl_pct / 100, 2)
        self.balance  += pnl_usdt
        pos.position_value = round(pos.position_value - close_value, 2)

        rec = TradeRecord(
            signal_id       = pos.signal_id,
            symbol          = symbol,
            direction       = pos.direction,
            side            = pos.side,
            entry_price     = pos.entry_price,
            exit_price      = round(exit_price, 8),
            entry_time      = pos.entry_time,
            close_time      = datetime.now(timezone.utc).isoformat(),
            pnl_pct         = round(pnl_pct, 3),
            pnl_usdt        = pnl_usdt,
            outcome         = 'WIN' if pnl_pct > 0 else 'LOSS',
            exit_reason     = exit_reason,
            meta_confidence = pos.meta_confidence,
            position_value  = close_value,
            signal_strength = pos.signal_strength,
            stop_loss       = pos.stop_loss,
            take_profit_1   = pos.take_profit_1,
            take_profit_2   = pos.take_profit_2,
            take_profit_3   = pos.take_profit_3,
            take_profit_4   = pos.take_profit_4,
            take_profit_5   = pos.take_profit_5,
            atr             = pos.atr,
        )
        self.trade_history.append(rec)
        return rec

    @property
    def summary(self) -> Dict[str, Any]:
        # Exclude TP partial-close records (exit_reason contains 'PARTIAL') from
        # the signal count — they represent slices of an open position, not closed
        # trades, and inflate wins/losses/total when the position is still running.
        _full = [t for t in self.trade_history if 'PARTIAL' not in (t.exit_reason or '')]
        won   = sum(1 for t in _full if t.outcome == 'WIN')
        lost  = sum(1 for t in _full if t.outcome == 'LOSS')
        total = won + lost
        pnl_u = round(self.balance - self.initial_capital, 2)
        return {
            'initial_capital': self.initial_capital,
            'balance':         round(self.balance, 2),
            'total_pnl_usdt':  pnl_u,
            'total_pnl_pct':   round(pnl_u / self.initial_capital * 100, 3),
            'total_trades':    total,
            'won':             won,
            'lost':            lost,
            'win_rate':        round(won / total, 3) if total else 0.0,
            'open_positions':  len(self.open_positions),
        }


# =============================================================================
# Live engine
# =============================================================================

class LiveEngine:
    """
    Prediction loop that scores every tradeable symbol every scan_interval_seconds.

    Signal flow
    -----------
    1. Predictor.predict_realtime() → dict with fire/side/edge_score/price/atr
    2. MarketRegimeDetector.detect()  → RegimeState (HMM-informed)
    3. SignalQualityFilter.score_signal() → (quality_score, reasons)
    4. Two-tier entry gates (see below)
    7. If price hits ATR stop → STOP_HIT exit
    8. After TP1 hit → activate trailing stop at 0.5× ATR below peak (LONG)
    9. After every cycle → write data/track_record.json
    """

    MAX_CONCURRENT        = 8
    HOURS_CONTEXT         = 300
    MIN_HOLD_SECONDS      = 3_600    # 1 h minimum hold before model-reversal exit
    COOLDOWN_SECONDS      = 300    # 5 min post-close cooldown (any outcome)
    FLIP_COOLDOWN_SECONDS = 600    # 10 min cooldown when the new signal flips direction
    GATE_VERSION = 'model-first-v80 (NEAREST S/R TARGET SELECTION & SUPPORT REVERSAL IMMEDIATE FIRING — (1) Target Support/Resistance now ALWAYS selects the nearest support below price for BUY/LONG and nearest resistance above price for SELL/SHORT from all pooled structure, swing, and HTF S/R levels. Far HTF override removed. (2) Support tag-and-reject/reversal for LONG setups immediately marks _came_from_m=True so bouncing off nearest support fires the signal without waiting.)'

    # ── Structure gate (Gate 1.6) ─────────────────────────────────────────
    # Zone boundaries on range_position (0 = rolling support, 1 = resistance).
    # STRICT (tightened 0.35/0.65 → 0.20/0.80): a BUY only counts as "at support"
    # in the bottom 20% of the range, a SELL "at resistance" in the top 20% — so
    # entries are genuinely NEAR the level, not a third of the way in. Signals
    # outside the tight zone are held PENDING (v35) and wait for a real S/R touch,
    # so this tightens LOCATION without discarding signals.
    STRUCT_SUPPORT_ZONE    = 0.35   # at/below → support zone (near support)
    STRUCT_RESISTANCE_ZONE = 0.65   # at/above → resistance zone (near resistance)
    # Lower-timeframe confirmation: candles must already be moving in the
    # signal direction.  Reversal entries at the correct level tolerate one
    # consolidation candle on 5m (3 of 4); breakout entries require ALL of
    # them (4/4 + 2/2) — momentum has to be unanimous to justify buying
    # into resistance / selling into support.
    STRUCT_5M_WINDOW  = 4
    STRUCT_5M_MIN     = 3
    ENTRY_5M_WINDOW   = 4  # v81: alias for Guard M/J 5m confirmation window (same as STRUCT_5M_WINDOW)
    STRUCT_15M_WINDOW = 2
    STRUCT_15M_MIN    = 2
    # Break-and-retest scan depth (closed 5m candles ≈ one hour)
    STRUCT_RETEST_LOOKBACK = 12
    # Confirmed breakout CONTINUATION (a break that runs and never retests):
    #   last N closed 5m candles must ALL hold beyond the broken level, and price
    #   must not have run more than MAX_EXT ATRs past it (no late chase).
    STRUCT_BREAKOUT_MIN_HOLD    = 3
    STRUCT_BREAKOUT_MAX_EXT_ATR = 2.5
    # Reversal entries fire close to the level: allowed when a wick has TAGGED the
    # level OR price is within this many ATRs of it. Back to 0.9 (0.5 was too tight
    # and muted most S/R reversals — the PRIMARY setup — leaving the fire rate low).
    STRUCT_LEVEL_PROXIMITY_ATR = 1.5
    # Rejection fast-path: once price has TAGGED a level and moved back more than
    # this fraction of the S/R range off it (a SELL >10% below resistance it hit /
    # a BUY >10% above support it hit), the reversal fires immediately — the
    # rejection is the confirmation (bypasses the counter-trend 2-candle wait).
    STRUCT_REJECTION_PCT = 0.10
    # (Removed) STRUCT_SR_ZONE_PCT — the old "15% of the S/R range" mid-range
    # tolerance. It was dead since v33/v34 (the swing zone-opening that used it was
    # deleted); location is governed solely by STRUCT_SUPPORT_ZONE/RESISTANCE_ZONE
    # on range_position now. Removed so nobody thinks it still fires anything.
    # Counter-trend REVERSAL (buy at support in a bear / sell at resistance in a
    # bull) must be at a genuine RSI extreme — a "reversal" with mid RSI is a
    # bounce that resumes (the falling-knife longs / squeezed shorts). Long
    # reversal needs RSI <= LONG floor; short reversal needs RSI >= SHORT ceiling.
    REVERSAL_RSI_LONG  = 42.0
    REVERSAL_RSI_SHORT = 58.0

    # v72 — TRUST THE MODEL (user decision 2026-07-19). The retrained models are
    # reversal-focused and holdout-validated (dir_prec lower bound >= 60% on
    # independent events), so most doctrine/structure guards no longer hard-block
    # a model fire: Guards A(soft)/D/L/K/F/I/N/B DEMOTE the signal to RISKY
    # with a named warning instead of vetoing it.
    # v73 — TWO GUARDS ARE HARD AGAIN by explicit user decision ("support/
    # resistance gate and 3 candle confirmation a hard gate, other will be
    # fine"): Guard M (fires ONLY at/tag-rejecting a tested level; wrong-half
    # locations rejected) and Guard J (no entry without the 5m reversal candle,
    # fails closed) block regardless of this flag. Capital-safety gates always
    # hard-block: the dead-market ATR hard floor, NO_TRADE_REGIME, LOSS_COOLDOWN,
    # safe mode, the portfolio guard, the RR gate, drift block, the no-perp
    # bench. Set False to restore every hard block (full strict doctrine).
    TRUST_MODEL_FIRE = True

    # v79 — HTF-ANCHORED LEVEL DISCIPLINE (user, 2026-07-20): "in a bear market
    # the engine is making lower lows the support — that is wrong; it should be
    # near that purple support zone." A structural 1h shelf only counts as a
    # real entry level when it sits within HTF_ZONE_CONFIRM_ATR of the 4h/1d
    # level on the entry side; otherwise the pending target IS the HTF level
    # and the signal waits for the zone. The v75 off-level candle fire is only
    # honoured within OFF_LEVEL_MAX_ATR of the target zone — a 5m bounce candle
    # in mid-air no longer fires a fade.
    HTF_ZONE_CONFIRM_ATR = 1.0
    OFF_LEVEL_MAX_ATR    = 1.5

    # Model-first hard floor: below this ATR% the market is too flat to trade
    # (stops sit inside 1h tick noise). 0.5% matches the legacy Gate 2 value.
    MIN_FIRE_ATR_PCT = 0.5

    # v70 — the ABSOLUTE dead-market floor. MIN_FIRE_ATR_PCT is a flat, cross-
    # sectional line, so it conflates "this token is dead" with "the whole tape
    # is quiet": measured live, ARKM printed edge_score 100/100, conviction
    # 0.61/0.39, RSI 22.2 AT support (rp 0.167) — a textbook oversold-at-support
    # fade — and was hard-blocked for ATR 0.472% vs 0.5%, a 0.028pp miss. The
    # floor's real question is "can the move cover costs": at 0.47% ATR, TP1
    # (0.55 ATR) is ~0.26%, comfortably above fees+spread, so that block was not
    # protecting anything. Below MIN_FIRE_ATR_HARD_PCT the market genuinely is
    # untradeable (BTC sat at 0.18% in this tape) and stays hard-blocked; the
    # band BETWEEN the two floors fires ONLY at a structural extreme and is
    # tagged RISKY (thin_atr). A reversal FROM an extreme is precisely where
    # volatility expands, so requiring the PRIOR 14h to be lively is backwards.
    MIN_FIRE_ATR_HARD_PCT = 0.35

    # "Structural extreme" = the exhaustion point this engine is built to fade:
    # price pinned at the edge of its range AND RSI stretched. Shared by the
    # Guard A and Guard D relaxations so both mean the same thing.
    EXTREME_RP_BUY   = 0.25   # bottom quarter of the range → BUY exhaustion
    EXTREME_RP_SELL  = 0.75   # top quarter of the range    → SELL exhaustion
    EXTREME_RSI_BUY  = 35.0
    EXTREME_RSI_SELL = 65.0

    # RETIRED (v54): a raw p_buy−p_sell conviction floor is incompatible with how
    # these models express confidence. edge_score is a PERCENTILE RANK of the
    # model's edge vs its own history, not the raw probability spread — the
    # 3-class models sit near 50/50 on buy-vs-sell (p_hold holds the mass), which
    # is exactly why calibration uses percentile edge. A 10pt floor blocked 100%
    # of real model fires (LDO 0.9pt, FIL 1.4pt) while the NON-firing tokens
    # carried the big spreads (SOL 32.8pt, AVAX 35.0pt). The model's own
    # edge>=threshold IS the conviction check. Kept only for reference.
    MIN_MODEL_CONVICTION = 0.0   # unused — see Guard D REMOVED in _process_symbol

    # Guard E advisory: UWGS score_hold has a structural floor, so it edges out the
    # model's side on almost every signal. Only flag RISKY when the composite
    # disagrees by a REAL margin — otherwise every trade would be tagged RISKY and
    # the tier would carry no information (the old `ranging_regime` trap).
    COMPOSITE_HOLD_MARGIN = 15.0

    # ── Important levels (Guard K) ───────────────────────────────────────────
    # A level is important only when price has come back and reacted to it more
    # than once. See _important_levels() for why both previous S/R systems were
    # unusable. LEVEL_TOP_K is the load-bearing dial: uncapped, this yields
    # 10-24 levels per token and price is always within 0.6 ATR of one of them
    # ("at a level" on 88% of the fleet = noise). Capped at the 4 most-tested it
    # yields ~4 levels and a realistic 27%. Measured across 60 tokens:
    #   top-K   levels/token   "at a level"   model fires kept / vetoed
    #     inf        24.3           88%              11 / 19
    #      6          6.0           32%              22 /  8
    #      4          4.0           27%              23 /  7
    LEVEL_MERGE_ATR   = 0.5   # pivots within 0.5 ATR are the SAME level
    LEVEL_MIN_TOUCHES = 2     # touched once is not a level, it is an accident
    LEVEL_TOP_K       = 4     # keep ONLY the strongest levels — the whole point
    AT_LEVEL_ATR      = 0.6   # price is "at" a level within this many ATR

    # ── Guard M: hold every signal until price is AT the level ───────────────
    # A signal no longer fires mid-range. It is HELD (pending) until price is
    # within PENDING_NEAR_PCT of the level it should reverse at — a BUY at the
    # nearest important SUPPORT below, a SELL at the nearest important RESISTANCE
    # above — measured in PRICE, not ATR (the user's spec: "wait for that price
    # to be reached as near as possible"). Only then does the 3x5m confirmation
    # (Guard J) run and the held signal fire. If there is no tested level on the
    # correct side (price at the extreme of its range), it stays pending rather
    # than firing into open air. The scan re-evaluates every cycle, so a pending
    # signal fires the moment price reaches its level and confirms.
    #
    # MEASURED WARNING: on a 28-signal book only ~5 sit within 0.3% of their
    # level; the other ~23 go pending. This is a deliberate ~80% cut in immediate
    # fires — precisely the "at a level is mandatory" design that collapsed the
    # rate in v25/v38. The pending queue is what rescues it: the setups are not
    # discarded, they wait. Expect a quieter, spikier feed by design.
    PENDING_NEAR_PCT  = 1.2   # price within 1.2% of the level counts as "at" it

    # Regimes where entry is unconditionally blocked.  RANGING was demoted to
    # an advisory warning: the structure gate (BUY at support / SELL at
    # resistance) plus candlestick reversal confirmation are exactly the
    # setups that remain valid inside a range, so a blanket ban both starved
    # the engine of signals and contradicted those gates.
    NO_TRADE_REGIMES: set = {_REGIME_LIQUIDITY_TRAP}
    TRUST_MODEL_FIRE: bool = True

    def __init__(
        self,
        token_configs:         List[TokenConfig],
        capital:               float        = 10_000.0,
        max_position_usdt:     float        = 1_000.0,
        scan_interval_seconds: int           = 300,
        risk_tier:             str           = "balanced",
        proxy_url:             Optional[str] = None,
    ):
        self.scan_interval_seconds = scan_interval_seconds
        self.risk_tier = risk_tier
        # Restore the track record from Firestore BEFORE the wallet loads, so a
        # redeploy on Railway's ephemeral disk doesn't start from zero.
        _hydrate_track_record_from_firestore()
        self.wallet    = VirtualWallet(capital, max_position_usdt)
        self._executor = ThreadPoolExecutor(
            max_workers=self.MAX_CONCURRENT, thread_name_prefix='aegis_pred')

        self.predictors:   Dict[str, Any]   = {}
        self.last_signals: Dict[str, Any]   = {}
        self.live_prices:  Dict[str, float] = {}

        self._open_time:        Dict[str, float] = {}
        self._last_close_time:  Dict[str, float] = {}
        self._last_loss_time:   Dict[str, float] = {}   # last LOSING close per token (post-loss cooldown)
        self._last_close_side:  Dict[str, str]   = {}
        self._pending_alert: Dict[str, float]    = {}      # 'SYM|SIDE' -> last time it was PENDING (Telegram dedup)
        self._blocked_alert: Dict[str, float]    = {}      # 'SYM|SIDE' -> last time it was BLOCKED (Telegram dedup)
        self._armed_pending_setups: Dict[str, Dict[str, Any]] = {}  # Persistent queue of ARMED signals waiting for level / 5m turn
        self._last_close_reason: Dict[str, str]  = {}   # reason of the most recent close (for reversal-flip throw)
        self._spreads:          Dict[str, float] = {}   # symbol → book spread % (UWGS dead-market veto)
        self._news_lock:        Tuple[bool, str] = (False, '')   # (locked?, label) — scheduled macro event

        # Signal direction history for stability gate (symbol → last N directions)
        self._signal_history:   Dict[str, Deque[str]] = {}

        # OHLCV candle cache for multi-timeframe reversal gate
        # key = "SYMBOL|timeframe" → {'candles': list, 'ts': float}
        self._candle_cache: Dict[str, Dict] = {}
        self._candle_cache_ttl = 240  # seconds; refresh every 4 min (< 5m candle period)

        # Partial-TP hit tracking (one flag per level per symbol)
        self._tp1_hit:    Dict[str, bool]  = {}   # break-even triggered after TP1
        self._tp2_hit:    Dict[str, bool]  = {}   # trailing stop activated after TP2
        self._tp3_hit:    Dict[str, bool]  = {}
        self._tp4_hit:    Dict[str, bool]  = {}
        self._peak_price: Dict[str, float] = {}   # highest (LONG) or lowest (SHORT) seen since entry

        self.bootstrap_done  = 0
        self.bootstrap_total = len(token_configs)

        # Alpha mode — multi-timeframe scanning (Pro only)
        self.alpha_mode    = False
        self.alpha_signals: Dict[str, Any] = {}
        self.alpha_wallet  = VirtualWallet(10_000.0, 1_000.0, ALPHA_TRACK_RECORD_PATH)
        self._alpha_open_time:       Dict[str, float] = {}
        self._tide_val: str   = ''    # cached BTC 4h tide ('UP'/'DOWN'), see _btc_tide
        self._tide_ts:  float = 0.0
        self._alpha_last_close_time: Dict[str, float] = {}
        self._alpha_last_close_side: Dict[str, str]   = {}

        # Adaptive intelligence modules
        self.regime_detector = MarketRegimeDetector()
        self.quality_filter  = SignalQualityFilter()
        self.risk_engine     = DynamicRiskEngine()
        self._tlc_detector   = TrendlineChannelDetector()   # additive trendline/channel confirmation
        self.perf_tracker    = PerformanceTracker()
        self.drift_monitor   = DriftMonitor()
        self.portfolio_guard = PortfolioGuard()

        # Restore persisted state so protection survives server restarts
        self.perf_tracker.load_state()
        self.drift_monitor.load_state()
        self.drift_monitor.load_benchmarks()

        # Sync engine aux-state from wallet's restored open positions
        for sym, pos in self.wallet.open_positions.items():
            try:
                et = datetime.fromisoformat(pos.entry_time.replace('Z', '+00:00'))
                self._open_time[sym] = et.timestamp()
            except Exception:
                self._open_time[sym] = time.time()
            self._tp1_hit[sym]    = False
            self._tp2_hit[sym]    = False
            self._tp3_hit[sym]    = False
            self._tp4_hit[sym]    = False
            self._peak_price[sym] = pos.entry_price

        self._load_predictors([c.symbol for c in token_configs])

    # ── initialisation ────────────────────────────────────────────────────────

    def _load_predictors(self, symbols: List[str]) -> None:
        from src.ml.predictor import Predictor
        loaded = tradeable = 0
        # Models BENCHED by their own meta: `risk_tier` disables the tier this
        # engine runs (`self.risk_tier`).  predict_signal() early-returns
        # fire=False / edge_score=0 for these on EVERY scan — they can never
        # fire, no matter the market or any engine gate.  (Currently ATOM, BTC,
        # COMP, SNX, VET: they failed the holdout precision floor.)  Before this,
        # the binary-pair branch below force-set tradeable=True and logged them
        # as live, which was a lie — and every scan still paid for a full
        # predict_realtime (350h of feature engineering) just to have the tier
        # gate throw it away.  Track them so _process_symbol can skip the work.
        self._benched: set = set()
        for sym in symbols:
            try:
                p = Predictor(sym)
                base = sym.replace('/', '_')
                buy_path  = MODEL_STORE / f"{base}_model_buy.json"
                sell_path = MODEL_STORE / f"{base}_model_sell.json"

                # No USD-M perpetual => this token cannot be traded on the market
                # the product reports, and its chart cannot render. Bench it rather
                # than let the spot fallback manufacture perp-labelled signals.
                if not _has_usdm_perp(sym):
                    self._benched.add(sym)
                    self.predictors[sym] = p
                    loaded += 1
                    print(f'[LiveEngine] BENCHED {sym} — no Binance USD-M perpetual '
                          f'(perp is listed as 1000{sym.split("/")[0]}); it can only be '
                          f'served from SPOT, so it can never fire (monitor-only)')
                    continue

                # Binary dual-model pair: check BEFORE risk_tier benching so a
                # dual-direction retrained token isn't permanently benched by
                # an older meta.json tier flag. If both side models exist, mark
                # it tradeable regardless of the legacy `risk_tier` setting.
                if buy_path.exists() and sell_path.exists():
                    p.meta['tradeable']      = True
                    p.meta['tradeable_buy']  = True
                    p.meta['tradeable_sell'] = True
                    self.predictors[sym] = p
                    loaded += 1
                    tradeable += 1
                    print(f'[LiveEngine] Loaded binary dual model for {sym} (tradeable)')
                    continue

                # Legacy risk_tier benching: if meta explicitly disables the
                # current engine `risk_tier`, mark symbol as benched (monitor-only).
                _tiers = p.meta.get('risk_tier') or {}
                _benched = bool(_tiers) and not bool(_tiers.get(self.risk_tier, False))
                if _benched:
                    self._benched.add(sym)
                    self.predictors[sym] = p
                    loaded += 1
                    print(f'[LiveEngine] BENCHED {sym} — meta risk_tier disables '
                          f'"{self.risk_tier}"; it can never fire (monitor-only)')
                    continue

                # Fallback: legacy single-direction model
                if p.model is not None:
                    self.predictors[sym] = p
                    loaded += 1
                    _is_tradeable = p.meta.get('tradeable', False)
                    if _is_tradeable:
                        tradeable += 1
                    print(f'[LiveEngine] Loaded legacy model for {sym} (tradeable={_is_tradeable})')
                else:
                    print(f'[LiveEngine] No model found for {sym}')
            except Exception as e:
                print(f'[LiveEngine] Failed to load {sym}: {e}')
        self.bootstrap_total = max(loaded, 1)
        if self._benched:
            print(f'[LiveEngine] {len(self._benched)} benched (never fire): '
                  f'{", ".join(sorted(self._benched))}')
        print(f'[LiveEngine] {loaded} predictors loaded '
              f'({tradeable} tradeable + {loaded - tradeable} monitor-only) '
              f'from {len(symbols)} configured symbols.')

    # ── main loop ─────────────────────────────────────────────────────────────

    def _purge_subquality_positions(self) -> None:
        """Close any restored open positions whose edge score is below MIN_QUALITY_SCORE.

        These positions pre-date the quality gate and should not occupy wallet slots.
        Closed at the current live price (or entry price if live price unavailable).
        """
        to_purge = [
            sym for sym, pos in list(self.wallet.open_positions.items())
            if pos.meta_confidence < SignalQualityFilter.MIN_QUALITY_SCORE
        ]
        for sym in to_purge:
            pos   = self.wallet.open_positions[sym]
            price = self.live_prices.get(sym, pos.entry_price)
            self.wallet.close_trade(sym, price, 'QUALITY_GATE_CLEANUP')
            self._last_close_time[sym] = time.time()
            self._last_close_side[sym] = pos.side
            print(f'[LiveEngine] PURGED {sym} — edge={pos.meta_confidence:.1f} < '
                  f'{SignalQualityFilter.MIN_QUALITY_SCORE:.0f} (pre-gate position)')
        if to_purge:
            self._save_track_record()

    async def run(self) -> None:
        print(f'[LiveEngine] Starting — interval={self.scan_interval_seconds}s '
              f'symbols={len(self.predictors)} | {self.GATE_VERSION}')
        asyncio.create_task(self._ws_price_ticker())
        await asyncio.sleep(2)   # let WebSocket populate live_prices first
        self._purge_subquality_positions()
        self._save_track_record()   # push restored positions to web immediately
        while True:
            t0 = time.time()
            # HARD GUARD: a single transient error (one bad candle fetch, a math
            # edge case on one of 63 tokens, a Firestore hiccup) must NEVER kill
            # the whole engine.  Before this, any unhandled exception here escaped
            # run(), ended the coroutine, and froze the engine until the next
            # redeploy (the recurring "signals stopped firing" stalls).  Now every
            # cycle is isolated: log the traceback and keep scanning.
            try:
                # Once per cycle (market-wide): refresh the news-lock window and the
                # book spreads that feed the UWGS dead-market / news vetoes.
                self._news_lock = econ_calendar.is_locked()
                await self._refresh_spreads()
                await self._scan_all()
                self._save_track_record()
                if self.alpha_mode:
                    self._save_alpha_track_record()
                await self._push_signals_to_firestore()
                # Heartbeat — makes "is the engine alive?" answerable from the
                # Railway logs at a glance, and surfaces per-cycle fire counts.
                _n_fired = sum(1 for s in self.last_signals.values() if s.get('fire'))
                _n_open  = len(self.wallet.open_positions)
                print(f'[LiveEngine] heartbeat {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z '
                      f'| cycle {time.time()-t0:0.1f}s | fired={_n_fired} '
                      f'open={_n_open} | {self.GATE_VERSION.split("(")[0].strip()}')
            except Exception as _cycle_err:
                import traceback as _tb
                print(f'[LiveEngine] scan cycle error (engine stays alive): '
                      f'{_cycle_err!r}')
                _tb.print_exc()
            sleep = max(0.0, self.scan_interval_seconds - (time.time() - t0))
            await asyncio.sleep(sleep)

    async def _push_signals_to_firestore(self) -> None:
        """Push last_signals to Firestore 'signals' collection after each scan.
        Runs the SYNCHRONOUS Firestore work in a worker thread so a slow/broken
        datastore can never block the scan loop, and is skipped once the circuit
        breaker trips (e.g. the project has no Firestore database)."""
        if self.bootstrap_done < self.bootstrap_total or _FS_DOWN:
            return  # skip during warmup, or if Firestore is known-down
        await asyncio.get_event_loop().run_in_executor(self._executor, self._push_signals_sync)

    def _push_signals_sync(self) -> None:
        global _FS_DOWN
        try:
            import firebase_admin
            from firebase_admin import credentials as _creds, firestore as _fs

            cred_path = _ROOT / 'config' / 'serviceAccountKey.json'
            if not cred_path.exists():
                return

            if not firebase_admin._apps:
                firebase_admin.initialize_app(_creds.Certificate(str(cred_path)))

            db = _fs.client()
            batch = db.batch()
            batch_count = 0

            for sym, sig in self.last_signals.items():
                is_fired = bool(sig.get('fire')) and sig.get('signal', 'HOLD') not in ('HOLD', 'FLAT', '')
                is_pending = bool(sig.get('pending_entry'))
                if not (is_fired or is_pending):
                    continue
                doc_id = sym.replace('/', '_')
                doc_ref = db.collection('signals').document(doc_id)
                # Sanitise: remove None values (Firestore rejects them)
                payload = {k: v for k, v in sig.items() if v is not None}
                batch.set(doc_ref, payload)
                batch_count += 1
                if batch_count >= 500:  # Firestore batch limit
                    batch.commit()
                    batch = db.batch()
                    batch_count = 0

            if batch_count > 0:
                batch.commit()

            # Always save local JSON fallback state for REST API endpoints
            try:
                state_path = _ROOT / 'data' / 'trader_signals_state.json'
                state_path.parent.mkdir(parents=True, exist_ok=True)
                state_path.write_text(_json.dumps(self.last_signals, indent=2, default=str))
            except Exception:
                pass

        except Exception as _e:
            _FS_DOWN = True   # trip breaker — stop pushing to a broken datastore
            print(f'[FirebasePush] disabled after error: {_e}')

    async def _ws_price_ticker(self) -> None:
        """
        Real-time price feed via Binance all-market mini-ticker WebSocket stream.
        Reconnects automatically on any error.
        """
        import json as _json

        # Include all meta-store tokens in the ticker subscription so every
        # known symbol gets a live price even if its ML model wasn't loaded.
        import glob as _g
        _meta_syms = [
            Path(f).stem.replace('_meta', '').replace('_USDT', '/USDT')
            for f in _g.glob(str(MODEL_STORE / '*_USDT_meta.json'))
        ]
        all_syms = list(set(list(self.predictors.keys()) + list(self._INDEX_SYMBOLS) + _meta_syms))
        sym_map: Dict[str, str] = {
            s.replace('/', '').lower(): s for s in all_syms
        }

        ws_url = 'wss://stream.binance.com:9443/ws/!miniTicker@arr'

        while True:
            try:
                import websockets                                         # type: ignore[import]
                async with websockets.connect(                           # type: ignore[attr-defined]
                    ws_url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    async for raw in ws:
                        try:
                            tickers = _json.loads(raw)
                            if not isinstance(tickers, list):
                                continue
                            for t in tickers:
                                key = t.get('s', '').lower()
                                ccxt_sym = sym_map.get(key)
                                if ccxt_sym:
                                    price = float(t.get('c') or 0)
                                    if price > 0:
                                        self.live_prices[ccxt_sym] = price
                        except Exception:
                            pass
            except Exception:
                await asyncio.sleep(3)

    _INDEX_SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'BNB/USDT']

    async def _scan_all(self) -> None:
        sem = asyncio.Semaphore(self.MAX_CONCURRENT)
        tasks = [self._process_symbol(sym, pred, sem)
                 for sym, pred in self.predictors.items()]
        await asyncio.gather(*tasks, return_exceptions=True)

        if self.alpha_mode:
            alpha_tasks = [
                self._process_alpha_timeframe(sym, tf, sem)
                for sym in self.predictors
                for tf in _ALPHA_TIMEFRAMES
            ]
            await asyncio.gather(*alpha_tasks, return_exceptions=True)

        await self._fetch_index_prices()

        # Preserve persistent armed setups in last_signals across indicator wobbles
        for sym in list(self.last_signals.keys()):
            self._sync_armed_pending_state(sym)

        self._notify_new_pending()
        self.bootstrap_done = len(self.predictors)

    def _sync_armed_pending_state(self, symbol: str) -> None:
        """Ensure persistent armed setup state is preserved in last_signals unless fired/expired/invalidated."""
        if symbol not in self.last_signals:
            return
        sig = self.last_signals[symbol]
        if sig.get('fire'):
            self._armed_pending_setups.pop(symbol, None)
            sig['pending_entry'] = False
            return

        if symbol in self._armed_pending_setups:
            armed = self._armed_pending_setups[symbol]
            armed_age = time.time() - armed.get('armed_time', 0.0)
            target = armed.get('target')
            price = float(sig.get('price', 0.0) or 0.0)
            side = armed.get('side', '')

            # Invalidation: expired (>12h) or price moved past target level by > 3%
            is_expired = armed_age >= 12 * 3600
            is_invalid = False
            if price > 0 and target and target > 0:
                if side == 'BUY' and price < target * 0.97:
                    is_invalid = True
                elif side == 'SELL' and price > target * 1.03:
                    is_invalid = True

            if is_expired or is_invalid:
                self._armed_pending_setups.pop(symbol, None)
                sig['pending_entry'] = False
            else:
                sig['pending_entry']  = True
                sig['pending_side']   = side
                sig['pending_target'] = target
                sig['pending_reason'] = armed.get('reason', 'waiting for level / 5m turn')
                sig['structure_reason'] = f"pending — {sig['pending_reason']}"

    def _notify_new_pending(self) -> None:
        """Telegram heads-up when a symbol NEWLY enters PENDING — ONE alert per
        pending episode, flickers tolerated.

        A signal held by Guard M (cleared the JACKDLM direction gates, waiting
        for price to reach its S/R level + 3x5m confirm) is announced once. The
        earlier version reset the "already alerted" set to the current pending
        set every scan, so a signal that dropped out of pending for a SINGLE
        scan — which happens all the time when conviction/edge wobbles across a
        threshold in a choppy tape — looked new on its next scan and re-alerted,
        spamming the same signal. Now each (symbol, side) records the last time
        it was pending; a fresh pending only alerts when that key has been ABSENT
        for >= PENDING_ALERT_COOLDOWN. The timestamp is refreshed EVERY pending
        scan, so a continuous or flickering pending fires exactly one alert, and
        a genuinely new setup (gone for hours, then re-armed) alerts again.
        Dispatcher applies its own quiet-hours + pending budget on top. Never
        raises.
        """
        try:
            now = time.time()
            current: Dict[str, dict] = {}          # 'SYM|SIDE' -> sig
            for sym, sig in self.last_signals.items():
                if not (isinstance(sig, dict) and sig.get('pending_entry')):
                    continue
                side = str(sig.get('pending_side') or '').upper()
                if side in ('BUY', 'SELL'):
                    current[f'{sym}|{side}'] = sig
            to_alert = []
            for key, sig in current.items():
                if now - self._pending_alert.get(key, 0.0) >= self.PENDING_ALERT_COOLDOWN:
                    to_alert.append((key, sig))
                self._pending_alert[key] = now     # refresh so a flicker/continuous pending won't re-alert
            if not to_alert:
                return
            from scripts.notifications.dispatcher import get_notifier
            _notifier = get_notifier()
            _iso = datetime.now(timezone.utc).isoformat()
            for key, sig in to_alert:
                sym = key.rsplit('|', 1)[0]
                _notifier.send_pending({
                    'symbol':         sym,
                    'direction':      str(sig.get('pending_side') or '').upper(),
                    'pending_target': sig.get('pending_target'),
                    'timestamp':      _iso,
                })
        except Exception:
            pass

    def _notify_blocked(self, symbol: str, side: str, price: float, reason: str) -> None:
        """Dispatch a Telegram notification for an UNFIRED · BLOCKED model lean (deduplicated)."""
        try:
            key = f"{symbol}|{side}"
            now = time.time()
            if now - self._blocked_alert.get(key, 0.0) >= 1800:
                self._blocked_alert[key] = now
                from scripts.notifications.dispatcher import get_notifier
                get_notifier().send_blocked({
                    'symbol': symbol,
                    'direction': side,
                    'current_price': price,
                    'structure_reason': reason,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                })
        except Exception:
            pass

    async def _refresh_spreads(self) -> None:
        """Refresh best bid/ask book spreads (%) for the fleet — one bulk call per
        cycle, off the event loop. Feeds the UWGS dead-market veto. Never raises."""
        try:
            loop = asyncio.get_event_loop()
            books = await asyncio.wait_for(
                loop.run_in_executor(self._executor, _fetch_bids_asks_all),
                timeout=10,
            )
            if books:
                self._spreads = books
        except Exception:
            pass

    async def _fetch_index_prices(self) -> None:
        """Fetch spot prices for market overview symbols not covered by the tradeable fleet."""
        missing = [s for s in self._INDEX_SYMBOLS if s not in self.live_prices]
        if not missing:
            return
        loop = asyncio.get_event_loop()
        for sym in missing:
            try:
                ticker = await asyncio.wait_for(
                    loop.run_in_executor(
                        self._executor,
                        lambda s=sym: _fetch_spot_price(s),
                    ),
                    timeout=10,
                )
                if ticker and ticker > 0:
                    self.live_prices[sym] = ticker
            except Exception:
                pass

    async def _process_symbol(
        self, symbol: str, predictor: Any, sem: asyncio.Semaphore
    ) -> None:
        # BENCHED model: its meta's risk_tier disables the tier we run, so
        # predict_signal() would early-return fire=False / edge=0 anyway. Skip
        # the call entirely — it costs a full 350h feature build per scan for a
        # result that can never fire. Still surfaced as a monitor-only token.
        if symbol in getattr(self, '_benched', ()):
            self.last_signals.setdefault(symbol, {}).update({
                'symbol':  symbol,
                'signal':  'HOLD',
                'fire':    False,
                'benched': True,
                'price':   self.live_prices.get(symbol, 0.0),
            })
            self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
            return

        async with sem:
            loop = asyncio.get_event_loop()
            try:
                result: Dict[str, Any] = await asyncio.wait_for(
                    loop.run_in_executor(
                        self._executor,
                        lambda p=predictor: p.predict_realtime(risk_tier=self.risk_tier),
                    ),
                    timeout=120,
                )
            except Exception:
                self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                return
            if not isinstance(result, dict):
                return

            _model_price = float(result.get('price', 0) or 0)
            # Prefer the live WebSocket tick for both display and entry pricing.
            # The WS ticker updates self.live_prices every second; the model price
            # is a 1h candle close that can be stale by minutes, causing an instant
            # phantom loss at entry. Only fall back to the model price if WS is unavailable.
            _ws_price = float(self.live_prices.get(symbol, 0) or 0)
            price = _ws_price if _ws_price > 0 else _model_price
            if _model_price > 0 and _ws_price == 0:
                # WS not yet seeded for this symbol — seed it with the model price
                self.live_prices[symbol] = _model_price

            # ── Adaptive intelligence layer ───────────────────────────────────
            # Step 1: HMM regime (probabilistic, from predictor's result dict)
            # The HMM ran inside predict_realtime() and attached hmm_* fields.
            # We extract them here and let them sharpen the MarketRegimeDetector.
            _hmm_regime     = result.get('hmm_regime', 'UNKNOWN')
            _hmm_available  = bool(result.get('hmm_available', False))
            _hmm_conf_adj   = float(result.get('hmm_conf_adjustment', 0.0))
            _hmm_atr_mult   = float(result.get('hmm_atr_mult', 1.0))
            _hmm_pos_scale  = float(result.get('hmm_position_scale', 1.0))
            _hmm_trade_ok   = bool(result.get('hmm_trade_allowed', True))
            _hmm_trans_risk = float(result.get('hmm_transition_risk', 0.0))

            # If HMM says no-trade (e.g. COMPRESSION pre-breakout or DISTRIBUTION)
            # suppress the signal immediately — don't waste the quality scoring pass.
            if _hmm_available and not _hmm_trade_ok:
                if symbol in self.last_signals:
                    self.last_signals[symbol]['fire']         = False
                    self.last_signals[symbol]['signal']       = 'HOLD'
                    self.last_signals[symbol]['hmm_blocked']  = True
                self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                return

            # Step 2: Rule-based regime classifier (existing, now HMM-informed)
            # If HMM has a confident read, override the heuristic detector's label.
            regime = self.regime_detector.detect(result)
            if _hmm_available and float(result.get('hmm_confidence', 0)) > 0.5:
                # Map HMM label to the existing RegimeState taxonomy
                _HMM_TO_INTERNAL = {
                    'TRENDING_BULL':      _REGIME_TRENDING_BULL,
                    'TRENDING_BEAR':      _REGIME_TRENDING_BEAR,
                    'CHOPPY':             _REGIME_RANGING,
                    'VOLATILE_EXPANSION': _REGIME_VOLATILE_EXPANSION,
                    'COMPRESSION':        _REGIME_VOLATILE_COMPRESS,
                    'ACCUMULATION':       _REGIME_ACCUMULATION,
                    'DISTRIBUTION':       _REGIME_DISTRIBUTION,
                }
                _internal = _HMM_TO_INTERNAL.get(_hmm_regime)
                if _internal:
                    regime = RegimeState(
                        regime               = _internal,
                        confidence           = float(result.get('hmm_confidence', 0.5)),
                        trade_allowed        = _hmm_trade_ok,
                        preferred_strategies = regime.preferred_strategies,
                        max_position_pct     = regime.max_position_pct * _hmm_pos_scale,
                    )

            new_side      = result.get('side', 'FLAT')

            # Track signal direction history for the stability gate.
            # Only directional signals accumulate; FLAT/HOLD is ignored so that
            # BUY→FLAT→BUY doesn't reset the counter (volatile markets produce
            # intermittent FLAT cycles that previously erased valid setups).
            # A genuine direction flip (BUY→SELL) is handled by the stability
            # check itself: all(s == new_side) will fail if the deque has the
            # opposite side, so no explicit clear is needed.
            if new_side in ('BUY', 'SELL'):
                if symbol not in self._signal_history:
                    self._signal_history[symbol] = deque(maxlen=self.SIGNAL_STABILITY_WINDOW)
                self._signal_history[symbol].append(new_side)

            # ── Revive the HTF daily/weekly trend bias ────────────────────────────
            # macro_daily / macro_weekly arrive 0.0 from the model (a swallowed
            # KeyError zeroes them at the source — see _daily_bias). Recompute them
            # from daily candles and inject BEFORE scoring, so the existing HTF-bias
            # tiers in score_signal() (+15/+10 aligned, -20/-10 against) and Guard F
            # (hard-block when BOTH daily AND weekly oppose) come alive. This is the
            # directional focus the engine was always meant to have: trade WITH the
            # higher-timeframe trend; a genuine reversal at a level stays exempt from
            # the penalty (score_signal's `is_reversal` branch).
            _macro_d, _macro_w = await self._daily_bias(symbol)
            result['macro_daily']  = _macro_d
            result['macro_weekly'] = _macro_w

            quality_score, _quality_reasons = self.quality_filter.score_signal(
                result, regime, new_side)
            fake_breakout = self.quality_filter.is_fake_breakout(result, new_side)

            # ── MODEL-FIRST decision: the ML model picks side + fire ──────────────
            # The model's side/fire (from predict_realtime) is the authority — the
            # ~80%-WR decision. UWGS is computed only for (a) the chart breakdown,
            # (b) the risk tier, and (c) the four genuinely protective hard vetoes.
            # It cannot change the side or manufacture a fire. Location signals
            # (FAR_FROM_SR / NO_VALID_SR) downgrade the tier to RISKY; they never
            # block. MODEL_DISAGREES is ignored (the model IS the decision now).
            _model_side  = result.get('side', 'FLAT')       # 'BUY' | 'SELL' | 'FLAT'
            _model_fire  = bool(result.get('fire', False))
            _ctx_quality = quality_score                    # context score (0-100) → UWGS input
            _loc_poor    = False
            if USE_WEIGHTED_SCORER:
                result['is_fake_breakout'] = fake_breakout
                result['price'] = price
                _hist = self._signal_history.get(symbol)
                _stab = (sum(1 for x in _hist if x == _model_side) / len(_hist)) if _hist else 0.5
                _uwgs = WeightedGateScorer.score(
                    result, regime,
                    {
                        'drift_blocked':  self.drift_monitor.is_blocked(symbol),
                        'drift_severity': self.drift_monitor.severity(symbol),
                        'stability_frac': _stab,
                        'quality_score':  _ctx_quality,
                        'portfolio_ok':   True,
                        'spread_pct':     self._spreads.get(symbol, 0.0),
                        'news_locked':    self._news_lock[0],
                        'news_label':     self._news_lock[1],
                    },
                )
                # Informational — chart breakdown + S/R quality (does NOT decide side)
                result['signal_scores'] = {'buy':  _uwgs['score_buy'],
                                           'sell': _uwgs['score_sell'],
                                           'hold': _uwgs['score_hold']}
                result['gate_breakdown'] = _uwgs['breakdown']
                result['sr_quality']     = _uwgs['sr_quality']
                # Confirmation: keep only the protective hard vetoes; location
                # flags become a tier downgrade below (not a block).
                _news_lbl = self._news_lock[1]
                _hard = [v for v in _uwgs['vetoes']
                         if v in _HARD_VETOES or v == 'NEWS_LOCK'
                         or (_news_lbl and v == _news_lbl)]
                _loc_poor = ('FAR_FROM_SR' in _uwgs['vetoes']
                             or 'NO_VALID_SR' in _uwgs['vetoes'])
                result['vetoes']      = _hard
                result['sr_loc_poor'] = _loc_poor
                # MODEL decides; a hard veto can only SUPPRESS its fire.
                if _hard:
                    result['side'] = 'FLAT'
                    result['fire'] = False
                else:
                    result['side'] = _model_side
                    result['fire'] = _model_fire and _model_side in ('BUY', 'SELL')
                result['tradeable'] = True
                # Sizing / tier conviction = the MODEL's edge (its own 0-100
                # quality), not the UWGS composite. edge_score is 0-100;
                # meta_confidence is 0-1 — normalise.
                _edge = float(result.get('edge_score',
                                         result.get('meta_confidence', 0)) or 0)
                if _edge <= 1.0:
                    _edge *= 100.0
                result['quality_score'] = round(_edge, 1)
                new_side      = result['side']
                quality_score = result['quality_score']

            # Build signal entry with enriched fields
            self.last_signals[symbol] = self._build_signal_entry(
                symbol, result, price, regime=regime,
                quality_score=quality_score, fake_breakout=fake_breakout)

            # ── v76: the cockpit shows COMMITTED fires only ───────────────────
            # _build_signal_entry publishes the model's RAW intent, but the
            # guard chain (with awaited fetches inside) only settles seconds
            # later — so every vetoed, pending or parked signal flashed through
            # the cockpit as a live fire and then vanished. Demote the initial
            # publish to "evaluating"; the fire point commits fire=True only
            # AFTER a position (real or paper) actually opened. An already-open
            # position — REAL or PAPER — keeps its fired state untouched
            # (v76.1: the paper exemption was missing, so every paper fire
            # demoted back to a dot on the next scan and the cockpit emptied).
            _paper_open = f'{symbol}|risky' in self.alpha_wallet.open_positions
            if (self.last_signals[symbol].get('fire')
                    and symbol not in self.wallet.open_positions
                    and not _paper_open):
                self.last_signals[symbol]['evaluating']      = False
            elif _paper_open:
                # v77.3: an open PAPER position IS an active (paper) signal —
                # its card must not fade out of the cockpit when the model's
                # live lean drifts. Display follows the paper book while open.
                _pp = self.alpha_wallet.open_positions.get(f'{symbol}|risky')
                self.last_signals[symbol]['paper_only'] = True
                if _pp is not None:
                    self.last_signals[symbol]['fire']            = True
                    self.last_signals[symbol]['signal']          = _pp.side
                    self.last_signals[symbol]['direction']       = _pp.direction
                    self.last_signals[symbol]['signal_strength'] = _pp.side
                    self.last_signals[symbol]['evaluating']      = False

            existing = self.wallet.open_positions.get(symbol)

            # ── v74: manage the RISKY paper book (alpha wallet, SYMBOL|risky) ──
            # RISKY-tier fires trade on paper until the tagged population proves
            # itself (see the fire path). Exits mirror the alpha scanner's rules:
            # SL, TP3 full take, opposing model fire, and the 24h zombie guard.
            _rk_key = f'{symbol}|risky'
            _rk_pos = self.alpha_wallet.open_positions.get(_rk_key)
            if _rk_pos is not None:
                _rk_px = float(self.live_prices.get(symbol, 0) or price or 0)
                if _rk_px > 0:
                    _rk_long = _rk_pos.direction == 'LONG'
                    _rk_sl   = ((_rk_px <= _rk_pos.stop_loss) if _rk_long
                                else (_rk_px >= _rk_pos.stop_loss))
                    _rk_tp   = (_rk_pos.take_profit_3 > 0 and
                                ((_rk_px >= _rk_pos.take_profit_3) if _rk_long
                                 else (_rk_px <= _rk_pos.take_profit_3)))
                    _rk_rev  = (bool(result.get('fire'))
                                and result.get('side') in ('BUY', 'SELL')
                                and result.get('side') != _rk_pos.side)
                    _rk_old  = (time.time() - self._alpha_open_time.get(_rk_key, time.time())
                                >= self.MAX_HOLD_SECONDS)
                    _rk_why  = ('SL_HIT' if _rk_sl else 'TP3_HIT' if _rk_tp else
                                'SIGNAL_REVERSAL' if _rk_rev else
                                'MAX_HOLD_EXPIRED' if _rk_old else '')
                    if _rk_why:
                        self.alpha_wallet.close_trade(_rk_key, _rk_px, _rk_why)
                        self._alpha_last_close_time[_rk_key] = time.time()
                        self._save_alpha_track_record()
                        print(f'[{symbol}] RISKY-PAPER {_rk_why} @ {_rk_px:.6g}')

            # Labelled S/R levels + Break->Retest->Confirmation states for the chart.
            # ONLY for symbols a user actually views on the chart — a firing signal
            # or an open position.  Computing this for all 63 tokens every scan (an
            # extra 1h fetch + pivot/state pass each) was heavy load that slowed the
            # scan loop; HOLD tokens fall back to the single support/resistance pair.
            _pending_view = bool(self.last_signals.get(symbol, {}).get('pending_entry'))
            if result.get('fire') or existing is not None or _pending_view:
                _atr_view = float(result.get('atr', 0) or 0)
                try:
                    _srl = await self._sr_levels(symbol, price, _atr_view)
                    if _srl:
                        self.last_signals[symbol]['sr_levels'] = _srl
                except Exception:
                    pass
                # ADDITIVE confirmation only — Trendline & Trend Channel context.
                # Never gates or changes the signal; it just attaches structure
                # analysis + a confidence score (see TrendlineChannelDetector).
                try:
                    _tlc = await self._trendline_channel(symbol, price, _atr_view)
                    if _tlc:
                        self.last_signals[symbol]['trendline_channel'] = _tlc
                except Exception:
                    pass

            _reversal_flip = False
            if existing:
                self._manage_exit(symbol, existing, result, price)
                # If the model just reversed us OUT of the position (a good opposite
                # signal cleared the reversal exit's quality floor + min-hold), let
                # the fire path below THROW the new opposite signal in the same scan
                # rather than holding it back a full flip-cooldown. The reversal was
                # already quality-gated, so a good signal shouldn't be silenced just
                # because we sat on the other side a moment ago.
                _reversal_flip = (
                    symbol not in self.wallet.open_positions
                    and self._last_close_reason.get(symbol) == 'MODEL_REVERSAL_TP'
                    and bool(result.get('fire'))
                    and bool(result.get('tradeable', False))
                    and price > 0
                )
                # After exit management: show the quality the signal FIRED at, not
                # the live re-score (which decays as conditions change after entry
                # and made healthy positions look like quality-6 garbage). Use the
                # stored entry quality_score, falling back to meta_confidence for
                # positions opened before this field existed. Skipped entirely when
                # flipping — the fresh opposite signal built above must stand.
                if not _reversal_flip and symbol in self.last_signals:
                    # v77.3: an open REAL position IS the active signal — the
                    # card must not drop out of the cockpit/setup rooms when
                    # the model's live lean fades mid-trade (measured: 5 open
                    # BUYs on the record but only 2 still showing in the BUY
                    # room). While the position is open, display follows the
                    # BOOK: side, fired state and strength come from the trade.
                    self.last_signals[symbol]['fire']       = True
                    self.last_signals[symbol]['signal']     = existing.side
                    self.last_signals[symbol]['direction']  = existing.direction
                    self.last_signals[symbol]['evaluating'] = False
                    _tier_open = str(existing.signal_strength or '').upper()
                    self.last_signals[symbol]['signal_strength'] = (
                        f'STRONG_{existing.side}' if _tier_open == 'STRONG'
                        else existing.side)
                    _entry_q = existing.quality_score or existing.meta_confidence
                    self.last_signals[symbol]['quality_score'] = round(_entry_q, 1)
                    # Re-attach the entry-time gate context so the chart's gate
                    # breakdown (tier, structure verdict, advisory ledger) stays
                    # complete for open positions — these live on the fire path and
                    # would otherwise be rebuilt away on every subsequent scan.
                    if existing.signal_strength:
                        self.last_signals[symbol]['risk_tier'] = existing.signal_strength
                    if existing.entry_mode:
                        self.last_signals[symbol]['entry_mode'] = existing.entry_mode
                    if existing.gate_warnings:
                        self.last_signals[symbol]['gate_warnings'] = existing.gate_warnings
                    # Show the model EDGE the signal FIRED at, not the live
                    # re-score. edge_score/meta_confidence decay after entry, so an
                    # open position that fired at edge >= 70 would otherwise render
                    # Gate 3 (Model Edge >= 70) as a red FAIL on a healthy trade —
                    # contradicting the fired signal. Position.meta_confidence holds
                    # the entry edge.
                    if existing.meta_confidence:
                        self.last_signals[symbol]['edge_score']      = round(existing.meta_confidence, 1)
                        self.last_signals[symbol]['meta_confidence']  = round(existing.meta_confidence, 1)
                    # Show LIVE, role-reversed S/R so the chart updates as price
                    # moves: nearest pivot ABOVE price = resistance, nearest BELOW
                    # = support. Once price trades above a level it flips to
                    # support (and vice versa). v16's role-reversal makes this
                    # correct even for breakdown shorts, so we no longer freeze the
                    # entry snapshot (which left resistance drawn below the price).
                    _live_sr = await self._swing_sr(symbol, price, float(result.get('atr', 0) or 0))
                    if _live_sr:
                        self.last_signals[symbol]['support']    = _live_sr[0]
                        self.last_signals[symbol]['resistance'] = _live_sr[1]
            if (not existing or _reversal_flip) and result.get('fire') \
                    and result.get('tradeable', False) and price > 0:
                now               = time.time()
                cooldown_elapsed  = now - self._last_close_time.get(symbol, 0)
                last_side         = self._last_close_side.get(symbol, '')
                is_flip           = (last_side != '' and last_side != new_side)
                required_cooldown = (
                    self.FLIP_COOLDOWN_SECONDS if is_flip else self.COOLDOWN_SECONDS
                )

                # A reversal flip has already passed the quality floor + min-hold in
                # _manage_exit, so the flip-cooldown would only silence a signal we
                # have already validated — bypass it and throw the reversal now.
                # Post-LOSS cooldown takes precedence over everything (even a
                # reversal flip): a token that just lost is benched so we don't
                # revenge-fire it while the thesis that failed is still in play.
                _loss_ago = now - self._last_loss_time.get(symbol, 0)
                if _loss_ago < self.LOSS_COOLDOWN_SECONDS:
                    print(f'[{symbol}] MODEL SKIP {new_side}: LOSS_COOLDOWN — lost '
                          f'{_loss_ago / 3600:.1f}h ago (< {self.LOSS_COOLDOWN_SECONDS / 3600:.0f}h), '
                          f'not re-firing a just-lost token')
                    if symbol in self.last_signals:
                        self.last_signals[symbol]['fire']          = False
                        self.last_signals[symbol]['signal']        = 'HOLD'
                        self.last_signals[symbol]['loss_cooldown'] = True
                elif _reversal_flip or cooldown_elapsed >= required_cooldown:
                    # Advisory-gate warning ledger (see ADVISORY_WARNING_BUDGET).
                    # Advisory gates append here instead of returning; the
                    # budget check runs after all gates have been evaluated.
                    _gate_warnings: List[str] = []

                    # ── MODEL-FIRST fire path (SOLE decision path) ────────────────
                    # The model chose side + fire and cleared the four protective
                    # hard vetoes above. A SHORT list of structural guards still
                    # applies here (the v10 baseline that "was doing it great" was
                    # model-decides + a LIGHT S/R/regime gate, NOT model-decides +
                    # nothing): an ATR floor, and a reversal-aware counter-trend
                    # gate. Everything else only sets the risk tier. Returns before
                    # the legacy cascade below (now dead code behind the flag).
                    if USE_WEIGHTED_SCORER:
                        _edge_q = float(result.get('quality_score', 0.0) or 0.0)  # model edge 0-100
                        _srq    = float(result.get('sr_quality', 0.0) or 0.0)
                        _poor   = bool(result.get('sr_loc_poor'))
                        _rp_now = _range_pos(result)
                        # v72 trust-model ledger: guards that would have vetoed this
                        # fire append their reason here instead; anything present
                        # caps the tier at RISKY (see tier assignment below).
                        _trust_warns: List[str] = []

                        # ── Structural extreme (v70) ──────────────────────────────
                        # The exhaustion point this engine exists to fade: price
                        # pinned at the edge of its range AND RSI stretched the same
                        # way. Guards A and D both relax ONLY here, so a marginal
                        # threshold miss can no longer kill the setups the strategy
                        # is built on, while mid-range signals are untouched.
                        _rsi_now = float(result.get('rsi', result.get('rsi_14', 50)) or 50)
                        _edge_raw = float(result.get('edge_score', 0.0) or 0.0)
                        _at_extreme = (
                            (new_side == 'BUY'  and _rp_now <= self.EXTREME_RP_BUY
                             and _rsi_now <= self.EXTREME_RSI_BUY) or
                            (new_side == 'SELL' and _rp_now >= self.EXTREME_RP_SELL
                             and _rsi_now >= self.EXTREME_RSI_SELL))
                        _atr_relaxed = _conv_relaxed = False

                        # ── Guard A: ATR floor — a flat market is untradeable ─────
                        # Hard-blocks only genuinely dead tape. Between the hard floor
                        # and MIN_FIRE_ATR_PCT a structural extreme fires, tagged RISKY.
                        _atr_pct_now = float(result.get('atr_pct', 0.0) or 0.0)
                        if 0 < _atr_pct_now < self.MIN_FIRE_ATR_PCT:
                            if (_atr_pct_now >= self.MIN_FIRE_ATR_HARD_PCT
                                    and (_at_extreme or self.TRUST_MODEL_FIRE)):
                                _atr_relaxed = True
                                print(f'[{symbol}] MODEL RELAX {new_side}: ATR_FLOOR '
                                      f'atr={_atr_pct_now:.2f}% < {self.MIN_FIRE_ATR_PCT}% '
                                      f'({"structural extreme" if _at_extreme else "trust-model"} '
                                      f'rp={_rp_now:.2f} rsi={_rsi_now:.1f}) — firing tagged RISKY')
                            else:
                                _atr_why = (
                                    f'below the {self.MIN_FIRE_ATR_HARD_PCT}% hard floor '
                                    f'— dead market'
                                    if _atr_pct_now < self.MIN_FIRE_ATR_HARD_PCT else
                                    f'< {self.MIN_FIRE_ATR_PCT}% and not at a '
                                    f'structural extreme (rp={_rp_now:.2f} rsi={_rsi_now:.1f})')
                                print(f'[{symbol}] MODEL BLOCK {new_side}: ATR_FLOOR '
                                      f'atr={_atr_pct_now:.2f}% {_atr_why}')
                                if symbol in self.last_signals:
                                    self.last_signals[symbol]['fire']         = False
                                    self.last_signals[symbol]['signal']       = 'HOLD'
                                    self.last_signals[symbol]['atr_blocked']  = True
                                self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                                return

                        # ── Guard C: no-trade regime & Volatile Compression / BOS gates ───────────
                        _reg0 = regime.regime if regime else 'UNKNOWN'
                        if _reg0 in self.NO_TRADE_REGIMES:
                            print(f'[{symbol}] MODEL BLOCK {new_side}: NO_TRADE_REGIME={_reg0}')
                            if symbol in self.last_signals:
                                self.last_signals[symbol]['fire']           = False
                                self.last_signals[symbol]['signal']         = 'HOLD'
                                self.last_signals[symbol]['regime_blocked'] = True
                            self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                            return

                        # Check BOS (Break of Structure) & CHoCH to catch "market changing moving path"
                        _bos_sig = 0.0
                        try:
                            _c1h_bos = await self._fetch_candles(symbol, '1h', self.CONFIRM_BOS_LOOKBACK + 5)
                            _closed1h_bos = _c1h_bos[:-1] if len(_c1h_bos) >= 2 else []
                            if _closed1h_bos:
                                _bos_info = _detect_bos_choch(_closed1h_bos, lookback=self.CONFIRM_BOS_LOOKBACK)
                                _bos_sig = float(_bos_info.get('signal', 0.0) or 0.0)
                        except Exception:
                            _bos_sig = 0.0

                        # 1. Opposing BOS Veto: Market is changing moving path in opposite direction
                        _bos_conflict = ((new_side == 'BUY'  and _bos_sig < 0) or
                                         (new_side == 'SELL' and _bos_sig > 0))
                        if _bos_conflict:
                            _why_bos = f'{new_side} opposes active BOS/CHoCH (market changing moving path)'
                            print(f'[{symbol}] MODEL BLOCK {new_side}: BOS_CONFLICT — market changing moving path (BOS={_bos_sig:+.0f} opposes {new_side})')
                            self._notify_blocked(symbol, new_side, price, _why_bos)
                            if symbol in self.last_signals:
                                self.last_signals[symbol]['fire']           = False
                                self.last_signals[symbol]['signal']         = 'HOLD'
                                self.last_signals[symbol]['bos_blocked']    = True
                                self.last_signals[symbol]['structure_reason'] = _why_bos
                            self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                            return

                        # 2. VOLATILE_COMPRESSION pre-breakout squeeze gate:
                        # In VOLATILE_COMPRESSION, market is coiling pre-breakout. Unconfirmed buys/sells
                        # without BOS in signal direction or extreme S/R location are bad signals.
                        if _reg0 == _REGIME_VOLATILE_COMPRESS:
                            _cmpr_confirmed = ((new_side == 'BUY'  and _bos_sig > 0) or
                                               (new_side == 'SELL' and _bos_sig < 0))
                            _rp_now0 = _range_pos(result)
                            _cmpr_at_extreme = ((new_side == 'BUY'  and _rp_now0 <= self.STRUCT_SUPPORT_ZONE) or
                                                (new_side == 'SELL' and _rp_now0 >= self.STRUCT_RESISTANCE_ZONE))
                            if not (_cmpr_confirmed or _cmpr_at_extreme):
                                print(f'[{symbol}] MODEL BLOCK {new_side}: VOLATILE_COMPRESSION — coiling pre-breakout squeeze requires BOS confirmation in signal direction')
                                if symbol in self.last_signals:
                                    self.last_signals[symbol]['fire']           = False
                                    self.last_signals[symbol]['signal']         = 'HOLD'
                                    self.last_signals[symbol]['regime_blocked'] = True
                                    self.last_signals[symbol]['structure_reason'] = (
                                        f'{new_side} in VOLATILE_COMPRESSION without BOS confirmation or extreme location')
                                self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                                return

                        # ── Guard O: exhaustion symmetry — never chase capitulation ──
                        # A SELL at oversold shorts the squeeze zone; a BUY at over-
                        # bought buys the blow-off. The doctrine fades exhaustion or
                        # stands aside — it never chases it. HARD in every regime:
                        # Guard B's RSI-extreme check only engages in trending
                        # regimes, which is how a SELL fired at RSI 28 in COMPRESSION
                        # (measured 2026-07-19 — the SELL models lean trend-following
                        # because the training window was 60-71% bear, so their fade
                        # focus got a third of the BUY side's reinforcement).
                        if ((new_side == 'SELL' and _rsi_now <= self.EXTREME_RSI_BUY) or
                                (new_side == 'BUY'  and _rsi_now >= self.EXTREME_RSI_SELL)):
                            print(f'[{symbol}] MODEL BLOCK {new_side}: EXHAUSTION_CHASE — '
                                  f'{new_side} at RSI {_rsi_now:.0f} chases '
                                  f'{"capitulation" if new_side == "SELL" else "a blow-off"}; '
                                  f'the fade of this bar is the {"BUY" if new_side == "SELL" else "SELL"}')
                            if symbol in self.last_signals:
                                self.last_signals[symbol]['fire']               = False
                                self.last_signals[symbol]['signal']             = 'HOLD'
                                self.last_signals[symbol]['exhaustion_blocked'] = True
                                self.last_signals[symbol]['structure_reason']   = (
                                    f'{new_side} at RSI {_rsi_now:.0f} — exhaustion chase')
                            self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                            return

                        # ── Guard D: directional conviction floor ─────────────────
                        # Does the model actually PREFER this side, or is it a coin
                        # flip that the direction-blind edge percentile waved through?
                        _pb = float(result.get('p_buy',  0.0) or 0.0)
                        _ps = float(result.get('p_sell', 0.0) or 0.0)
                        _dir_mass = _pb + _ps
                        _conv = ((_pb - _ps) / _dir_mass) if _dir_mass > 0 else 0.0
                        _want = _conv if new_side == 'BUY' else -_conv
                        # At its OWN top confidence and at a structural extreme the
                        # model has already proved the side through its calibrated
                        # meta gate — a flat 0.10 there double-counts and kills the
                        # exact fades this engine is built on (FIL: edge 100, RSI
                        # 18.2, rp 0.00, blocked for 0.095 vs 0.100).
                        _conv_floor = self.MIN_DIR_CONVICTION
                        if _at_extreme and _edge_raw >= self.EXTREME_EDGE_MIN:
                            _conv_floor   = self.MIN_DIR_CONVICTION_EXTREME
                            _conv_relaxed = _want < self.MIN_DIR_CONVICTION
                        if _want < _conv_floor:
                            if self.TRUST_MODEL_FIRE:
                                _conv_relaxed = True
                                print(f'[{symbol}] TRUST_MODEL {new_side}: conviction '
                                      f'{_want:+.3f} < {_conv_floor:.2f} '
                                      f'(p_buy {_pb*100:.1f} vs p_sell {_ps*100:.1f}) '
                                      f'— firing tagged RISKY')
                            else:
                                print(f'[{symbol}] MODEL BLOCK {new_side}: directional conviction '
                                      f'{_want:+.3f} < {_conv_floor:.2f} '
                                      f'(p_buy {_pb*100:.1f} vs p_sell {_ps*100:.1f}) — '
                                      f'the model does not prefer this side, the edge '
                                      f'percentile does')
                                if symbol in self.last_signals:
                                    self.last_signals[symbol]['fire']             = False
                                    self.last_signals[symbol]['signal']           = 'HOLD'
                                    self.last_signals[symbol]['conviction_blocked'] = True
                                    self.last_signals[symbol]['dir_conviction']   = round(_want, 3)
                                    self.last_signals[symbol]['structure_reason'] = (
                                        f'no directional conviction ({_want:+.2f})')
                                self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                                return
                        result['dir_conviction'] = round(_want, 3)
                        if _conv_relaxed:
                            print(f'[{symbol}] MODEL RELAX {new_side}: conviction '
                                  f'{_want:+.3f} < {self.MIN_DIR_CONVICTION:.2f} but edge '
                                  f'{_edge_raw:.0f} at a structural extreme '
                                  f'(rp={_rp_now:.2f} rsi={_rsi_now:.1f}) — '
                                  f'firing tagged RISKY')

                        # ── Guard L: directional confluence ───────────────────────
                        # Reconstruct the SIGN the direction-blind edge score threw
                        # away: tally the signed evidence already in the result dict
                        # FOR the model's side and require a net majority to agree.
                        _side_sign = 1 if new_side == 'BUY' else -1

                        def _sgn(v) -> int:
                            try:    v = float(v)
                            except (TypeError, ValueError): return 0
                            return 1 if v > 0 else (-1 if v < 0 else 0)

                        def _lbl(v) -> int:
                            v = str(v or '').upper()
                            return 1 if 'BULL' in v else (-1 if 'BEAR' in v else 0)

                        _rp_conf = _range_pos(result)
                        _loc_vote = 1 if _rp_conf <= 0.35 else (-1 if _rp_conf >= 0.65 else 0)
                        _dir_ev = {
                            'model':      _sgn(_pb - _ps)                 * _side_sign,
                            'bias':       _lbl(result.get('market_bias')) * _side_sign,
                            'macd':       _lbl(result.get('macd_signal')) * _side_sign,
                            'supertrend': _lbl(result.get('supertrend'))  * _side_sign,
                            'htf_daily':  _sgn(result.get('macro_daily'))  * _side_sign,
                            'htf_weekly': _sgn(result.get('macro_weekly')) * _side_sign,
                            'rsi_slope':  _sgn(result.get('rsi_slope'))    * _side_sign,
                            'location':   _loc_vote                        * _side_sign,
                        }
                        _dir_net = sum(_dir_ev.values())
                        result['dir_confluence'] = {'net': _dir_net, 'evidence': _dir_ev}

                        # Reversal at a level is counter-trend by definition — exempt
                        # (it proves itself through Guards J/B/K), mirroring Guard F.
                        _rsi_conf = float(result.get('rsi', result.get('rsi_14', 50)) or 50)
                        _conf_reversal = (
                            (new_side == 'SELL' and _rp_conf >= 0.65) or
                            (new_side == 'BUY'  and _rp_conf <= 0.35))
                        if not _conf_reversal and _dir_net < self.MIN_DIR_CONFLUENCE:
                            _opp = [k for k, v in _dir_ev.items() if v < 0]
                            if self.TRUST_MODEL_FIRE:
                                _trust_warns.append('weak_confluence')
                                print(f'[{symbol}] TRUST_MODEL {new_side}: DIR_CONFLUENCE '
                                      f'net={_dir_net:+d} < {self.MIN_DIR_CONFLUENCE} '
                                      f'(opposing: {",".join(_opp) or "none"}) '
                                      f'— firing tagged RISKY')
                            else:
                                print(f'[{symbol}] MODEL BLOCK {new_side}: DIR_CONFLUENCE '
                                      f'net={_dir_net:+d} < {self.MIN_DIR_CONFLUENCE} — the '
                                      f'signed evidence does not back this direction '
                                      f'(opposing: {",".join(_opp) or "none"})')
                                if symbol in self.last_signals:
                                    self.last_signals[symbol]['fire']              = False
                                    self.last_signals[symbol]['signal']            = 'HOLD'
                                    self.last_signals[symbol]['confluence_blocked'] = True
                                    self.last_signals[symbol]['dir_confluence']    = result['dir_confluence']
                                    self.last_signals[symbol]['structure_reason']  = (
                                        f'direction not confirmed (net {_dir_net:+d})')
                                self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                                return

                        # ── Guard M: hold until price is AT the level ─────────────
                        # No mid-range fires. Wait (PENDING) until price reaches the
                        # level it should reverse at — nearest important SUPPORT below
                        # for a BUY, nearest important RESISTANCE above for a SELL —
                        # within PENDING_NEAR_PCT (price %, not ATR). Only then does
                        # the 3x5m confirmation (Guard J, next) run and the signal
                        # fire. No tested level on the correct side => stay pending,
                        # never fire into open air. Re-evaluated every scan.
                        # Target = ALWAYS the NEAREST support below price (for BUY)
                        # or NEAREST resistance above price (for SELL).
                        _atr_m  = float(result.get('atr', 0) or 0)
                        _all_candidates: List[float] = []

                        try:
                            _struct_lvls = await self._structural_levels(symbol, price, _atr_m)
                            _all_candidates.extend([l for l, _t in _struct_lvls])
                        except Exception:
                            pass

                        _sup_raw = float(result.get('support', 0) or 0)
                        _res_raw = float(result.get('resistance', 0) or 0)
                        if _sup_raw > 0: _all_candidates.append(_sup_raw)
                        if _res_raw > 0: _all_candidates.append(_res_raw)

                        if isinstance(result.get('sr_levels'), list):
                            for _srlv in result.get('sr_levels', []):
                                if isinstance(_srlv, dict) and _srlv.get('price'):
                                    _all_candidates.append(float(_srlv['price']))

                        try:
                            _htf_m = await self._htf_sr(symbol, price, _atr_m)
                            if _htf_m:
                                _htf_sup, _htf_res = _htf_m
                                if _htf_sup and _htf_sup > 0: _all_candidates.append(_htf_sup)
                                if _htf_res and _htf_res > 0: _all_candidates.append(_htf_res)
                        except Exception:
                            pass

                        if new_side == 'BUY':
                            _sups = [l for l in _all_candidates if l < price]
                            _target_m = max(_sups) if _sups else (_sup_raw if 0 < _sup_raw < price else None)
                            _role_m   = 'support'      # BUY waits at nearest SUPPORT below
                        else:
                            _reses = [l for l in _all_candidates if l > price]
                            _target_m = min(_reses) if _reses else (_res_raw if _res_raw > price else None)
                            _role_m   = 'resistance'   # SELL waits at nearest RESISTANCE above

                        _near_pct_m = (abs(price - _target_m) / price * 100.0
                                       if _target_m and price > 0 else None)
                        _at_level_m = (_target_m is not None and _near_pct_m is not None
                                       and _near_pct_m <= self.PENDING_NEAR_PCT)

                        # Has price ALREADY tagged the level and receded? (came FROM
                        # the level rather than toward it — e.g. market reversed from support for a BUY)
                        # If so, the reversal has begun; fire now instead of waiting for a second touch.
                        _came_from_m = False
                        if _target_m is not None and not _at_level_m:
                            try:
                                _lookback = getattr(self, 'PENDING_TAG_LOOKBACK', 6)
                                _tol_pct  = getattr(self, 'PENDING_TAG_TOL_PCT', 1.5)
                                _rawm = await self._fetch_candles(
                                    symbol, '1h', _lookback + 2)
                                _cm = (_rawm[:-1] if len(_rawm) >= 2 else [])[-_lookback:]
                                _tol = max(float(_tol_pct) / 100.0, 0.015)
                                if _cm and new_side == 'SELL':
                                    _rhigh = max(float(c[2]) for c in _cm)     # recent high
                                    _came_from_m = (_rhigh >= _target_m * (1 - _tol) and price < _target_m)
                                elif _cm and new_side == 'BUY':
                                    _rlow = min(float(c[3]) for c in _cm)      # recent low
                                    _came_from_m = (_rlow <= _target_m * (1 + _tol) and price > _target_m)
                            except Exception:
                                _came_from_m = False

                        # ZONE AWARENESS: v81 relaxation — allow fires at S/R zones even
                        # if not exactly at the tested level. A BUY in the STRUCT_SUPPORT_ZONE
                        # or a SELL in the STRUCT_RESISTANCE_ZONE can proceed if 5m momentum
                        # confirms the reversal. Only block if in the OPPOSITE zone (e.g. BUY
                        # in resistance zone without a break) — the true wrong-side entry.
                        _rp_m = result.get('range_position')
                        _in_correct_zone = False
                        if _rp_m is not None:
                            _rp_m = float(_rp_m)
                            _in_correct_zone = ((new_side == 'BUY' and _rp_m <= self.STRUCT_SUPPORT_ZONE) or
                                               (new_side == 'SELL' and _rp_m >= self.STRUCT_RESISTANCE_ZONE))
                        
                        # Only hard-block if NOT at level AND NOT coming from level AND in opposite zone
                        if (_rp_m is not None and not _at_level_m and not _came_from_m 
                                and not _in_correct_zone):
                            _opp_role = 'support' if new_side == 'SELL' else 'resistance'
                            print(f'[{symbol}] MODEL BLOCK {new_side}: WRONG_ZONE — '
                                  f'range_position {_rp_m:.2f} is in the {_opp_role} zone; '
                                  f'a {new_side} must be in the {_role_m} zone (rp <= 0.35 for BUY, >= 0.65 for SELL)')
                            if symbol in self.last_signals:
                                self.last_signals[symbol]['fire']             = False
                                self.last_signals[symbol]['signal']           = 'HOLD'
                                self.last_signals[symbol]['location_blocked']  = True
                                self.last_signals[symbol]['pending_entry']     = False
                                self.last_signals[symbol]['structure_reason']  = (
                                    f'{new_side} in the {_opp_role} zone (rp {_rp_m:.2f})')
                            self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                            return

                        # Before placing into pending, check if short-term 5m
                        # reversal momentum has already printed — if 3 of the
                        # last ENTRY_5M_WINDOW closed 5m candles align with the
                        # signal direction or a reversal candlestick pattern is
                        # present, allow immediate firing as an off-level 5m
                        # momentum reversal (bypass pending).
                        _has_5m_reversal = False
                        try:
                            _raw5_pre = await self._fetch_candles(symbol, '5m', self.ENTRY_5M_WINDOW + 2)
                            _c5m_pre = _raw5_pre[:-1] if len(_raw5_pre) >= 2 else []
                            if len(_c5m_pre) >= self.ENTRY_5M_WINDOW:
                                _want_up_pre = (new_side == 'BUY')
                                _pat_pre = _reversal_candle(_c5m_pre, want_bullish=_want_up_pre)
                                _closed_dir_pre = [
                                    (float(c[4]) > float(c[1])) if _want_up_pre else (float(c[4]) < float(c[1]))
                                    for c in _c5m_pre[-self.ENTRY_5M_WINDOW:]
                                ]
                                _n5_pre = sum(_closed_dir_pre)
                                _has_5m_reversal = (_pat_pre is not None) or (_n5_pre >= max(3, self.ENTRY_5M_WINDOW - 1))
                        except Exception:
                            _has_5m_reversal = False

                        # v81: RELAXED PENDING LOGIC — fire at zone with 5m confirmation
                        # If price is in the CORRECT S/R ZONE and 5m momentum confirms, fire now.
                        # Only go pending if NOT at level AND NOT in zone AND NOT coming from level.
                        _price_in_zone = ((new_side == 'BUY' and _rp_m <= self.STRUCT_SUPPORT_ZONE) or
                                         (new_side == 'SELL' and _rp_m >= self.STRUCT_RESISTANCE_ZONE)) if _rp_m is not None else False
                        
                        _should_wait = (_target_m is None or 
                                       (not _at_level_m and not _came_from_m and not _has_5m_reversal and not _price_in_zone))
                        
                        if _should_wait:
                            _why_m = (f'no tested {_role_m} to wait for'
                                      if _target_m is None else
                                      f'approaching {_role_m} {_target_m:.6g} '
                                      f'({_near_pct_m:.2f}% away, needs <= {self.PENDING_NEAR_PCT}% '
                                      f'or a tag+reject or 3x5m reversal)')
                            print(f'[{symbol}] MODEL PENDING {new_side}: {_why_m} — '
                                  f'holding for price to reach the {_role_m} level')
                            self._armed_pending_setups[symbol] = {
                                'side': new_side,
                                'target': round(_target_m, 10) if _target_m else None,
                                'reason': _why_m,
                                'armed_time': time.time(),
                            }
                            if symbol in self.last_signals:
                                self.last_signals[symbol]['fire']            = False
                                self.last_signals[symbol]['signal']          = 'HOLD'
                                self.last_signals[symbol]['pending_entry']    = True
                                self.last_signals[symbol]['pending_side']     = new_side
                                self.last_signals[symbol]['pending_target']   = (
                                    round(_target_m, 10) if _target_m else None)
                                self.last_signals[symbol]['pending_reason']   = _why_m
                                self.last_signals[symbol]['structure_reason'] = f'pending — {_why_m}'
                            self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                            return

                        # AT the level, or just tagged-and-rejected it — record and
                        # fall through to Guard J (5m pattern + momentum confirmation).
                        if _target_m is not None:
                            result['at_pending_level'] = {
                                'level': round(_target_m, 10), 'role': _role_m,
                                'dist_pct': round(_near_pct_m, 3) if _near_pct_m is not None else None,
                                'trigger': 'at_level' if _at_level_m else 'tag_reject'}
                        self._armed_pending_setups.pop(symbol, None)
                        if symbol in self.last_signals:
                            self.last_signals[symbol]['pending_entry'] = False

                        # ── Guard J: 5m momentum alignment & pattern confirmation ──
                        _c5m: list = []
                        try:
                            _raw5 = await self._fetch_candles(
                                symbol, '5m', self.ENTRY_5M_WINDOW + 2)
                            _c5m = _raw5[:-1] if len(_raw5) >= 2 else []   # closed only
                        except Exception:
                            _c5m = []

                        _want_up = (new_side == 'BUY')
                        _pat = None
                        if len(_c5m) >= self.ENTRY_5M_WINDOW:
                            _pat = _reversal_candle(_c5m, want_bullish=_want_up)
                            _closed_dir = [
                                (float(c[4]) > float(c[1])) if _want_up else (float(c[4]) < float(c[1]))
                                for c in _c5m[-self.ENTRY_5M_WINDOW:]
                            ]
                            _n5 = sum(_closed_dir)
                            # Strictly require either a valid reversal pattern OR
                            # at least 3 directional 5m closes (no TRUST_MODEL_FIRE bypass)
                            _confirmed = (_pat is not None) or (_n5 >= 3)
                            _why5m = (f'5m {_pat or "directional"} pattern + {_n5}/{self.ENTRY_5M_WINDOW} directional candles confirmed'
                                      if _confirmed else
                                      f'needs 5m {"bullish" if _want_up else "bearish"} '
                                      f'reversal pattern or 3+ directional candles (pattern={_pat}, dir={_n5}/{self.ENTRY_5M_WINDOW})')
                        else:
                            _confirmed = False
                            _why5m = '5m confirmation unavailable (feed down)'

                        if not _confirmed:
                            print(f'[{symbol}] MODEL WAIT {new_side}: {_why5m} — '
                                  f'not entering without a confirmed reversal candle + 5m momentum')
                            if symbol in self.last_signals:
                                self.last_signals[symbol]['fire']            = False
                                self.last_signals[symbol]['signal']          = 'HOLD'
                                self.last_signals[symbol]['momentum_blocked'] = True
                                self.last_signals[symbol]['structure_reason'] = _why5m
                            self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                            return
                        result['reversal_pattern'] = _pat   # the pattern that confirmed entry

                        # ── Guard K: structural veto at an IMPORTANT level ────────
                        # THE gate the engine never had: does the model's side agree
                        # with where price actually is? Never fire INTO a level price
                        # has repeatedly respected — a BUY at resistance or a SELL at
                        # support — unless that level has already been broken.
                        #
                        # This is a VETO AT a level, NOT a requirement to BE at one.
                        # That distinction is the whole design. v25 and v38 both made
                        # "at a level" mandatory to fire and both collapsed the rate to
                        # zero, because real levels are far apart and price is usually
                        # BETWEEN them. Here, mid-range keeps firing untouched; only
                        # the wrong side AT a level is blocked. Measured across 60
                        # tokens: 23 of 30 model fires survive, 7 vetoed.
                        _lvl_ctx = None
                        try:
                            _atr_k = float(result.get('atr', 0) or 0)
                            _levels = await self._important_levels(symbol, price, _atr_k)
                            _lvl_ctx = self._level_context(
                                _levels, price, _atr_k, self.AT_LEVEL_ATR)
                            result['important_levels'] = [
                                {'level': round(l, 10), 'touches': t} for l, t in _levels]
                        except Exception:
                            _lvl_ctx = None

                        # v81: ZONE-BASED LOCATION GATE — allow fires at S/R zones,
                        # block only opposite-zone entries (e.g. BUY in RESISTANCE_ZONE without a break).
                        _rp_k = _range_pos(result)
                        _at_support_zone = (_rp_k <= self.STRUCT_SUPPORT_ZONE)
                        _at_resist_zone = (_rp_k >= self.STRUCT_RESISTANCE_ZONE)
                        _wrong_zone = (
                            (new_side == 'BUY'  and _at_resist_zone and not bool(result.get('resistance_broken_recent'))) or
                            (new_side == 'SELL' and _at_support_zone and not bool(result.get('support_broken_recent')))
                        )
                        if _wrong_zone:
                            _opp_role = 'resistance' if new_side == 'BUY' else 'support'
                            print(f'[{symbol}] MODEL BLOCK {new_side}: WRONG_ZONE — '
                                  f'range_position {_rp_k:.2f} is in the {_opp_role} zone and NOT broken; '
                                  f'a {new_side} at this zone without a break is a bad entry (rp={_rp_k:.2f})')
                            if symbol in self.last_signals:
                                self.last_signals[symbol]['fire']             = False
                                self.last_signals[symbol]['signal']           = 'HOLD'
                                self.last_signals[symbol]['location_blocked']  = True
                                self.last_signals[symbol]['pending_entry']     = False
                                self.last_signals[symbol]['structure_reason']  = f'{new_side} at {_opp_role} zone without break (rp {_rp_k:.2f})'
                            self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                            return

                        _at_level = False
                        if _lvl_ctx is not None:
                            _lv, _touches, _role, _dist = _lvl_ctx
                            _at_level = True
                            result['at_level']       = {'level': _lv, 'touches': _touches,
                                                        'role': _role, 'dist_atr': round(_dist, 2)}
                            # A break already confirmed by the predictor flips the level's
                            # role: a broken resistance is support, so a BUY through it is
                            # a breakout, not a wall. That keeps the dynamic case the user
                            # asked for — buy AT resistance once resistance is taken out.
                            _res_broken = bool(result.get('resistance_broken_recent'))
                            _sup_broken = bool(result.get('support_broken_recent'))
                            _wrong_side = (
                                (_role == 'RESISTANCE' and new_side == 'BUY'  and not _res_broken) or
                                (_role == 'SUPPORT'    and new_side == 'SELL' and not _sup_broken)
                            )
                            if _wrong_side:
                                if self.TRUST_MODEL_FIRE:
                                    _trust_warns.append('wrong_side_at_level')
                                    print(f'[{symbol}] TRUST_MODEL {new_side}: fires into unbroken '
                                          f'{_role.lower()} {_lv:.6g} ({_touches}x, {_dist:.2f} ATR) '
                                          f'— firing tagged RISKY')
                                else:
                                    print(f'[{symbol}] MODEL BLOCK {new_side}: WRONG_SIDE_AT_LEVEL — '
                                          f'{_role.lower()} {_lv:.6g} touched {_touches}x is '
                                          f'{_dist:.2f} ATR away and NOT broken; a {new_side} '
                                          f'fires straight into it')
                                    if symbol in self.last_signals:
                                        self.last_signals[symbol]['fire']            = False
                                        self.last_signals[symbol]['signal']          = 'HOLD'
                                        self.last_signals[symbol]['level_blocked']   = True
                                        self.last_signals[symbol]['structure_reason'] = (
                                            f'{new_side} into {_role.lower()} '
                                            f'({_touches}x, {_dist:.2f} ATR)')
                                    self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                                    return


                        # This gate was a DESIGN ERROR and is what took the engine to
                        # zero signals. `edge_score` is a PERCENTILE RANK of the
                        # model's edge against its OWN history — it is NOT the raw
                        # p_buy/p_sell spread. These 3-class models keep buy-vs-sell
                        # near 50/50 (p_hold absorbs the mass); that is precisely WHY
                        # the calibration uses percentile edge instead of raw
                        # probability. Measured live: the only token whose model fired
                        # (LDO, edge 74.4 >= its 60 threshold) has a 0.9pt raw spread,
                        # and FIL's was 1.4pt — so a 10pt conviction floor blocked
                        # 100% of real model fires, while the tokens that did NOT fire
                        # carried the HIGH raw spreads (SOL 32.8pt, AVAX 35.0pt).
                        # The model's own threshold IS the conviction check. Genuine
                        # technical disagreement is caught by Guard E below (which is
                        # what should have held FIL: HOLD 49.3 > BUY 40.1).

                        # ── Guard E: composite disagreement → ADVISORY, not a block ───
                        # Downgraded from a hard veto in v54. UWGS `score_hold` has a
                        # STRUCTURAL FLOOR — (1-f)*54*0.7 + (46 - qual_gain) — so HOLD
                        # outscores the model's side whenever the location/regime gates
                        # are weak, i.e. most of the time. Blocking on "HOLD > side"
                        # therefore re-installs UWGS as the decider (the very thing that
                        # collapsed signals before) rather than catching a rare
                        # disagreement. Measured: the ONLY token whose model fired (LDO,
                        # edge 74.4) was killed here — UWGS scored its SELL 3.1 vs HOLD
                        # 62.8. So it now TAGS the trade RISKY and lets the model's call
                        # stand; the user can filter RISKY out.
                        _sc      = result.get('signal_scores') or {}
                        _side_sc = float(_sc.get('buy' if new_side == 'BUY' else 'sell', 0) or 0)
                        _hold_sc = float(_sc.get('hold', 0) or 0)
                        _composite_hold = (bool(_sc) and
                                           (_hold_sc - _side_sc) >= self.COMPOSITE_HOLD_MARGIN)
                        if _composite_hold:
                            print(f'[{symbol}] COMPOSITE_HOLD {new_side}: hold={_hold_sc:.1f} vs '
                                  f'{new_side.lower()}={_side_sc:.1f} (gap '
                                  f'{_hold_sc - _side_sc:.1f} >= {self.COMPOSITE_HOLD_MARGIN}) '
                                  f'— technicals oppose, tagged RISKY')

                        # ── Guard F: HTF macro veto (legacy Gate 1.7) ─────────────
                        # Weekly AND daily EMA50 trend both clearly opposing.
                        # EXEMPTION: a genuine reversal AT a level (a SELL into
                        # resistance while overbought, a BUY into support while
                        # oversold) is what a counter-trend entry IS — the user's
                        # exact "bull + at resistance + model sells → fire" case.
                        # It is exempt from the hard block here (mirrors the
                        # is_reversal branch in score_signal) and instead proves
                        # itself through Guards J/B and is tagged RISKY. A plain
                        # trend-continuation entry against both HTF trends is still
                        # hard-blocked.
                        _htf_w = float(result.get('macro_weekly', 0.0) or 0.0)
                        _htf_d = float(result.get('macro_daily',  0.0) or 0.0)
                        _htf_rp  = _range_pos(result)
                        _htf_rsi = float(result.get('rsi', result.get('rsi_14', 50)) or 50)
                        _htf_reversal = (
                            (new_side == 'SELL' and _htf_rp >= 0.65) or
                            (new_side == 'BUY'  and _htf_rp <= 0.35))
                        if (_htf_w != 0.0 or _htf_d != 0.0) and not _htf_reversal and (
                                (new_side == 'BUY'  and _htf_w < -0.5 and _htf_d < -0.5) or
                                (new_side == 'SELL' and _htf_w >  0.5 and _htf_d >  0.5)):
                            if self.TRUST_MODEL_FIRE:
                                _trust_warns.append('htf_opposing')
                                print(f'[{symbol}] TRUST_MODEL {new_side}: HTF weekly={_htf_w:+.1f} '
                                      f'daily={_htf_d:+.1f} both opposing — firing tagged RISKY')
                            else:
                                print(f'[{symbol}] MODEL BLOCK {new_side}: HTF_VETO '
                                      f'weekly={_htf_w:+.1f} daily={_htf_d:+.1f} both opposing')
                                if symbol in self.last_signals:
                                    self.last_signals[symbol]['fire']        = False
                                    self.last_signals[symbol]['signal']      = 'HOLD'
                                    self.last_signals[symbol]['htf_blocked'] = True
                                self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                                return

                        # ── Guard I: HTF (4h/1d) S/R is DIRECTIONAL ───────────────
                        # A 4h SUPPORT is a BUY level; a 4h RESISTANCE is a SELL
                        # level. Never trade INTO the big structure: a SELL with a
                        # 4h support sitting right below gets bounced off that floor
                        # (its TPs are unreachable beneath it), and a BUY with a 4h
                        # resistance right overhead runs into the wall. Requires
                        # HTF_MIN_ROOM_ATR of room to the OPPOSING level. The legacy
                        # Gate 1.8 used the same distance but only blocked at an RSI
                        # extreme, so LDO (SHORT @0.3209, 4h support ~0.31, RSI 65)
                        # slipped through. Strict + RSI-independent now.
                        try:
                            _htf = await self._htf_sr(
                                symbol, price, float(result.get('atr', 0) or 0))
                        except Exception:
                            _htf = None
                        _htf_wall = False   # advisory flag → RISKY tier below
                        if _htf is not None:
                            _htf_sup, _htf_res = _htf
                            _atr_h = float(result.get('atr', 0) or 0) or price * 0.005
                            if symbol in self.last_signals:
                                self.last_signals[symbol]['htf_support']    = _htf_sup
                                self.last_signals[symbol]['htf_resistance'] = _htf_res
                            # Room to the OPPOSING HTF level, in ATR.
                            if new_side == 'BUY' and _htf_res > 0:
                                _room_atr = (_htf_res - price) / _atr_h
                                _lvl_desc = f'4h/1d resistance {_htf_res:.6g} overhead'
                            elif new_side == 'SELL' and _htf_sup > 0:
                                _room_atr = (price - _htf_sup) / _atr_h
                                _lvl_desc = f'4h/1d support {_htf_sup:.6g} below'
                            else:
                                _room_atr, _lvl_desc = 99.0, ''

                            _rsi_h  = float(result.get('rsi', 50) or 50)
                            _exhaust = ((new_side == 'BUY'  and _rsi_h > 72) or
                                        (new_side == 'SELL' and _rsi_h < 28))
                            _block_why = ''
                            if _room_atr < self.HTF_NO_ROOM_ATR:
                                # No room to even reach TP1 (0.55 ATR) before the wall.
                                _block_why = (f'{_lvl_desc} only {_room_atr:.2f} ATR away '
                                              f'(< {self.HTF_NO_ROOM_ATR}) — TP1 unreachable')
                            elif _room_atr < self.HTF_EXHAUSTION_ATR and _exhaust:
                                # v45 exhaustion chase: into the wall at an RSI extreme.
                                _block_why = (f'{_lvl_desc} {_room_atr:.2f} ATR away at RSI '
                                              f'{_rsi_h:.0f} — exhaustion chase into the wall')
                            if _block_why:
                                if self.TRUST_MODEL_FIRE:
                                    _htf_wall = True
                                    _trust_warns.append('htf_wall_close')
                                    print(f'[{symbol}] TRUST_MODEL {new_side}: HTF_WALL — '
                                          f'{_block_why} — firing tagged RISKY')
                                else:
                                    print(f'[{symbol}] MODEL BLOCK {new_side}: HTF_WALL — {_block_why}')
                                    if symbol in self.last_signals:
                                        self.last_signals[symbol]['fire']        = False
                                        self.last_signals[symbol]['signal']      = 'HOLD'
                                        self.last_signals[symbol]['htf_blocked'] = True
                                        self.last_signals[symbol]['structure_reason'] = _block_why
                                    self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                                    return
                            # ADVISORY: trading toward a nearby HTF level — fires, tagged
                            # RISKY, so the rate holds (this is what the old model did).
                            if _room_atr < self.HTF_ADVISORY_ATR:
                                _htf_wall = True
                                print(f'[{symbol}] HTF_CONFLUENCE {new_side}: {_lvl_desc} '
                                      f'{_room_atr:.2f} ATR away — tagged RISKY')

                        # ── Guard B: reversal-aware counter-trend gate ────────────
                        # A BUY in a bear trend / SELL in a bull trend is a
                        # counter-trend REVERSAL. It fires ONLY when ALL THREE hold:
                        #   (1) price has REACHED the level (within REVERSAL_PROX_ATR
                        #       of the real support/resistance — not just the wide
                        #       zone), so it waits instead of firing ~2 ATR early,
                        #   (2) RSI is at an extreme (oversold / overbought), and
                        #   (3) ≥REVERSAL_5M_MIN of the last REVERSAL_5M_WINDOW closed
                        #       5m candles have TURNED in the signal direction —
                        #       the actual reversal confirmation.
                        # Any condition unmet → WAIT (block this scan; re-evaluated
                        # next scan as price approaches the level and the turn forms).
                        # With-trend signals and non-trending regimes are unaffected.
                        _reg = regime.regime if regime else 'UNKNOWN'
                        # ── Guard N: FADE THE TREND — never follow it ─────────────
                        # User's strategy: the ONLY valid trend entries are counter-
                        # trend reversals at exhaustion — SELL a TRENDING_BULL when
                        # OVERBOUGHT, BUY a TRENDING_BEAR when OVERSOLD. A trend-
                        # FOLLOWING entry (SELL in a bear / BUY in a bull) is the
                        # "wrong way to enter" that was losing, so it is BLOCKED here.
                        # The reversal entries fall through to Guard B below, which
                        # already demands the level tag + RSI extreme + 5m turn — i.e.
                        # the "when overbought / oversold" part. In a RANGING regime
                        # neither trend branch triggers, so location-gated sell-at-
                        # resistance / buy-at-support (Guards K/M) still fire.
                        _trend_follow = (
                            (new_side == 'SELL' and _reg == _REGIME_TRENDING_BEAR) or
                            (new_side == 'BUY'  and _reg == _REGIME_TRENDING_BULL))
                        if _trend_follow:
                            _why_tf = f'{new_side} in {_reg} follows trend (fade-only strategy)'
                            print(f'[{symbol}] MODEL BLOCK {new_side}: TREND_FOLLOW — '
                                  f'{new_side} in {_reg} follows the trend; only counter-'
                                  f'trend reversals at exhaustion are taken (BUY in BEAR at support / SELL in BULL at resistance)')
                            self._notify_blocked(symbol, new_side, price, _why_tf)
                            if symbol in self.last_signals:
                                self.last_signals[symbol]['fire']                = False
                                self.last_signals[symbol]['signal']              = 'HOLD'
                                self.last_signals[symbol]['trend_follow_blocked'] = True
                                self.last_signals[symbol]['structure_reason']    = _why_tf
                            self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                            return
                        _ct_buy  = (new_side == 'BUY'  and _reg == _REGIME_TRENDING_BEAR)
                        _ct_sell = (new_side == 'SELL' and _reg == _REGIME_TRENDING_BULL)
                        if _ct_buy or _ct_sell:
                            _sup = float(result.get('support', 0) or 0)
                            _res = float(result.get('resistance', 0) or 0)
                            _atr_abs = (_atr_pct_now / 100.0) * price if price > 0 else 0.0
                            _lvl    = _sup if _ct_buy else _res
                            _sr_ok  = _atr_abs > 0 and _lvl > 0

                            # Fetch the 5m candles ONCE — used for both the level TAG
                            # (lows/highs) and the reversal confirmation.
                            _closed5: list = []
                            try:
                                _c5 = await self._fetch_candles(
                                    symbol, '5m',
                                    max(self.REVERSAL_TAG_LOOKBACK, self.REVERSAL_5M_WINDOW) + 2)
                                _closed5 = _c5[:-1] if len(_c5) >= 2 else _c5   # closed only
                            except Exception:
                                _closed5 = []

                            # (1) TAG — did price REACH the level recently? Measured on
                            #     candle LOWS (BUY) / HIGHS (SELL), NOT the current price,
                            #     because the confirmation candles have since lifted price
                            #     off the level. Falls back to current-price proximity /
                            #     the range-position zone when candles or S/R are missing.
                            if _sr_ok and _closed5:
                                _tag_tol = self.REVERSAL_PROX_ATR * _atr_abs
                                _tag_win = _closed5[-self.REVERSAL_TAG_LOOKBACK:]
                                if _ct_buy:
                                    _tagged = any((float(c[3]) - _lvl) <= _tag_tol for c in _tag_win)
                                else:
                                    _tagged = any((_lvl - float(c[2])) <= _tag_tol for c in _tag_win)
                            elif _sr_ok:
                                _dist0  = (price - _lvl) if _ct_buy else (_lvl - price)
                                _tagged = _dist0 <= self.REVERSAL_PROX_ATR * _atr_abs
                            else:
                                if 0 < _sup < _res and price > 0:
                                    _loc_rp = max(0.0, min(1.0, (price - _sup) / (_res - _sup)))
                                else:
                                    _loc_rp = _rp_now
                                _tagged = ((_loc_rp <= self.STRUCT_SUPPORT_ZONE) if _ct_buy
                                           else (_loc_rp >= self.STRUCT_RESISTANCE_ZONE))

                            # (2) NOT CHASING — current price must still be within
                            #     REVERSAL_MAX_CHASE_ATR of the level, so the fill is
                            #     never "way too wide" from support/resistance even
                            #     after the confirmation candles.
                            if _sr_ok:
                                _dist = (price - _lvl) if _ct_buy else (_lvl - price)
                                _not_chasing = _dist <= self.REVERSAL_MAX_CHASE_ATR * _atr_abs
                                _dist_desc   = f'{_dist / _atr_abs:.2f} ATR from ' + (
                                               'support' if _ct_buy else 'resistance')
                            else:
                                _not_chasing = True
                                _dist_desc   = 'distance unknown'

                            # (3) RSI extreme (per-token dynamic bounds, fallback 42/58)
                            _rsi_now = float(result.get('rsi', 50) or 50)
                            _rsi_lo, _rsi_hi = self.REVERSAL_RSI_LONG, self.REVERSAL_RSI_SHORT
                            try:
                                _rc = await self._fetch_candles(symbol, '1h', 120)
                                _rc = _rc[:-1] if len(_rc) >= 2 else _rc
                                if len(_rc) >= 30:
                                    _rsi_lo, _rsi_hi = self._dynamic_rsi_bounds(
                                        _rsi_series([float(c[4]) for c in _rc]))
                            except Exception:
                                pass
                            _rsi_extreme = (_rsi_now <= _rsi_lo) if _ct_buy else (_rsi_now >= _rsi_hi)

                            # (4) 5m reversal confirmation — ≥3 of the last 4 closed 5m
                            #     candles turned in the signal direction.
                            _n5 = -1   # -1 = candle feed unavailable → don't hard-block on it
                            if _closed5:
                                _last5 = _closed5[-self.REVERSAL_5M_WINDOW:]
                                if len(_last5) >= self.REVERSAL_5M_WINDOW:
                                    _n5 = sum(1 for c in _last5
                                              if (float(c[4]) > float(c[1])) == _ct_buy
                                              and float(c[4]) != float(c[1]))
                            _confirmed_5m = (_n5 < 0) or (_n5 >= self.REVERSAL_5M_MIN)

                            if not (_tagged and _not_chasing and _rsi_extreme and _confirmed_5m):
                                _lvln = 'support' if _ct_buy else 'resistance'
                                if not _tagged:
                                    _why = f'price has not reached {_lvln} yet ({_dist_desc})'
                                elif not _not_chasing:
                                    _why = f'too far from {_lvln} to enter ({_dist_desc} > {self.REVERSAL_MAX_CHASE_ATR} ATR) — not chasing'
                                elif not _rsi_extreme:
                                    _lim = _rsi_lo if _ct_buy else _rsi_hi
                                    _why = (f'RSI {_rsi_now:.0f} not '
                                            + ('oversold' if _ct_buy else 'overbought')
                                            + f' ({"<=" if _ct_buy else ">="}{_lim:.0f})')
                                else:
                                    _why = (f'5m reversal unconfirmed '
                                            f'({_n5}/{self.REVERSAL_5M_MIN} of last '
                                            f'{self.REVERSAL_5M_WINDOW} bars turned)')
                                if self.TRUST_MODEL_FIRE:
                                    _trust_warns.append('reversal_unconfirmed')
                                    print(f'[{symbol}] TRUST_MODEL {new_side}: counter-trend in '
                                          f'{_reg} — {_why} — firing tagged RISKY')
                                else:
                                    print(f'[{symbol}] MODEL WAIT {new_side}: counter-trend in '
                                          f'{_reg} — {_why}')
                                    if symbol in self.last_signals:
                                        self.last_signals[symbol]['fire']            = False
                                        self.last_signals[symbol]['signal']          = 'HOLD'
                                        self.last_signals[symbol]['regime_blocked']  = True
                                        self.last_signals[symbol]['structure_reason'] = (
                                            f'counter-trend {new_side} in {_reg}: {_why}')
                                    self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                                    return

                        # ── Guard T (v77): trade WITH the day, never against it ───
                        # User doctrine update 2026-07-19: "if today is market
                        # bearish then let only bear signals be fired." The BTC 4h
                        # tide is the day's direction; a fire against it — a BUY
                        # into a falling day, a SELL into a rising one — goes to
                        # the PAPER book tagged against_tide, where the counter-day
                        # class must prove itself before touching the record. FLAT
                        # or unknown tide gates nothing. This subsumes fade-vs-
                        # follow: BOTH fire real when they agree with the day.
                        # The models cannot learn this themselves yet — the daily/
                        # weekly macro features are silently ZEROED in TRAINING by
                        # the merge_asof collision (fixed engine-side only, v58) —
                        # so the engine enforces it until the feature pipeline is
                        # fixed and the fleet retrained.
                        result['btc_tide'] = await self._btc_tide()
                        _tide_now = str(result['btc_tide'] or 'FLAT')
                        # Exempt validated reversal fades from the against-tide
                        # papering. A setup qualifies if it's a strict structural
                        # extreme (_at_extreme) or a structural reversal at the
                        # support/resistance zone confirmed by short-term BOS.
                        _is_extreme_fade = False
                        try:
                            _is_extreme_fade = (
                                _at_extreme or (
                                    (new_side == 'BUY' and _rp_now <= self.STRUCT_SUPPORT_ZONE and _bos_sig > 0) or
                                    (new_side == 'SELL' and _rp_now >= self.STRUCT_RESISTANCE_ZONE and _bos_sig < 0)
                                )
                            )
                        except Exception:
                            _is_extreme_fade = _at_extreme

                        if ((new_side == 'BUY' and _tide_now == 'DOWN') or
                                (new_side == 'SELL' and _tide_now == 'UP')):
                            if not _is_extreme_fade:
                                _trust_warns.append('against_tide')
                                print(f'[{symbol}] AGAINST_TIDE {new_side}: BTC 4h tide is '
                                      f'{_tide_now} — papered until this class proves itself')
                            else:
                                print(f'[{symbol}] AGAINST_TIDE EXEMPTION {new_side}: Reversal fade at S/R level — firing live')

                        # ── Guard G: safe mode (legacy Gate 3.5) ──────────────────
                        # After 3 consecutive global losses, raise the edge floor to
                        # protect capital during a drawdown.
                        if self.perf_tracker.safe_mode_active() and _edge_q < 80.0:
                            print(f'[{symbol}] MODEL BLOCK {new_side}: SAFE_MODE '
                                  f'edge={_edge_q:.1f} < 80 (loss streak — elevated floor)')
                            if symbol in self.last_signals:
                                self.last_signals[symbol]['fire']   = False
                                self.last_signals[symbol]['signal'] = 'HOLD'
                            self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                            return

                        # ── Guard H: portfolio guard (legacy Gate 4) ──────────────
                        # Concurrency / correlation-cluster / capital limits. Dropped
                        # by model-first, which is how 7 positions opened at once.
                        self.portfolio_guard.sync_from_wallet(self.wallet.open_positions)
                        _pos_est = self.risk_engine.calculate_position_size(
                            self.wallet.balance, _edge_q, regime,
                            _atr_pct_now if _atr_pct_now > 0 else 1.5)
                        _pg_ok, _pg_why = self.portfolio_guard.can_open(
                            symbol, self.wallet.balance, _pos_est)
                        if not _pg_ok:
                            # v76.1: the cap governs the TRACKED BOOK, not the feed.
                            # A quality signal that arrives with the book full stays
                            # live for subscribers and is graded on PAPER — it does
                            # not vanish (that emptied the whole cockpit).
                            print(f'[{symbol}] BOOK_FULL {new_side}: {_pg_why} — '
                                  f'signal stays live, tracked on paper')
                            if symbol in self.last_signals:
                                self.last_signals[symbol]['portfolio_blocked'] = True

                        # entry_mode: audit trail + SL cap. A signal far from its
                        # level or mid-range gets a wider stop; at-level is tight.
                        if result.get('off_level_fire'):
                            _entry_mode = 'model_off_level_reversal'   # v75: candle fired it, not the level
                        elif _poor:
                            _entry_mode = 'model_far_from_level'
                        elif abs(_rp_now - 0.5) >= 0.3:
                            _entry_mode = 'model_at_level'
                        else:
                            _entry_mode = 'model_mid_range'

                        # Risk tier: model edge gated by S/R confirmation. A poor
                        # location, a fake breakout, or a counter-trend reversal
                        # (genuine but statistically the riskiest) caps at RISKY —
                        # it fires, flagged, so the user can filter RISKY out.
                        _warn: List[str] = []
                        if _poor:               _warn.append('far_from_level')
                        if fake_breakout:       _warn.append('fake_breakout')
                        if _ct_buy or _ct_sell: _warn.append('counter_trend_reversal')
                        if _htf_wall:           _warn.append('htf_wall_ahead')
                        if _composite_hold:     _warn.append('composite_hold')
                        # v70 relaxations: these fired BELOW a normal floor because the
                        # setup was at a structural extreme — genuine, but the thinnest
                        # of their kind, so they can never rate better than RISKY.
                        if _atr_relaxed:        _warn.append('thin_atr')
                        if _conv_relaxed:       _warn.append('thin_conviction')
                        # v75: fired by the 5m candle while pending, without the level
                        # tag — flagged so the record can grade this population.
                        if result.get('off_level_fire'):
                            _warn.append('off_level_reversal')
                        # v72 trust-model: every guard that would have vetoed this fire
                        # demoted itself into this ledger instead — all cap at RISKY.
                        _warn.extend(_trust_warns)
                        # Guard K survivors: price is AT an important level and the
                        # model's side agrees with it (a BUY off support / a SELL off
                        # resistance, or a confirmed break through it). That is real
                        # structural backing — the only setup allowed to reach STRONG.
                        # Mid-range fires are still legitimate (Guard K deliberately
                        # does not block them) but they have NO level behind them, so
                        # they cap at NORMAL however good the model's edge looks.
                        if _warn:
                            _risk_tier = 'RISKY'
                        elif (_at_level and _edge_q >= 80
                              and _srq >= 0.55 and _ctx_quality >= 65):
                            _risk_tier = 'STRONG'
                        elif _edge_q >= 68:
                            _risk_tier = 'NORMAL'
                        else:
                            _risk_tier = 'RISKY'

                        if symbol in self.last_signals:
                            self.last_signals[symbol]['risk_tier']     = _risk_tier
                            self.last_signals[symbol]['entry_mode']    = _entry_mode
                            self.last_signals[symbol]['gate_warnings'] = _warn

                        # ── v74/v75/v76.1: what goes to PAPER vs the real book ────
                        # PAPER (alpha wallet, key SYMBOL|risky):
                        #   - fires a doctrine guard would have VETOED outright
                        #     (_trust_warns — trend-follow, weak confluence, HTF
                        #     opposing, ...): measured, the bleeding class; and
                        #   - quality fires that arrive with the BOOK FULL (the cap
                        #     protects the tracked record, never the feed).
                        # REAL book: everything else — including quality-tagged
                        # RISKY fires (thin ATR, off-level candle fires, counter-
                        # trend reversals). Paper fires DISPLAY as live fires with
                        # paper_only=True: the feed is the product; the books are
                        # accounting.
                        if _trust_warns or not _pg_ok:
                            _rk_key = f'{symbol}|risky'
                            _rk_cd  = time.time() - self._alpha_last_close_time.get(_rk_key, 0)
                            if (_rk_key not in self.alpha_wallet.open_positions
                                    and _rk_cd >= 1800):
                                self._alpha_open_position(_rk_key, symbol, result, price, 'risky')
                            if symbol in self.last_signals:
                                self.last_signals[symbol]['paper_only'] = True
                                # display commit: fired once the paper position exists
                                if _rk_key in self.alpha_wallet.open_positions:
                                    self.last_signals[symbol]['fire']            = True
                                    self.last_signals[symbol]['signal']          = new_side
                                    self.last_signals[symbol]['direction']       = (
                                        'LONG' if new_side == 'BUY' else 'SHORT')
                                    # restore the side chip — the evaluating
                                    # demotion left it NEUTRAL (measured: a
                                    # NEUTRAL-chip card sitting in the SELL room)
                                    self.last_signals[symbol]['signal_strength'] = new_side
                                    self.last_signals[symbol]['evaluating']      = False
                            _why_paper = 'book_full' if (_pg_ok is False and not _trust_warns) \
                                         else ','.join(_trust_warns)
                            print(f'[{symbol}] MODEL FIRE {new_side} tier={_risk_tier} -> PAPER '
                                  f'({_why_paper}) edge={_edge_q:.0f}'
                                  f'{" warn=" + ",".join(_warn) if _warn else ""}')
                            try:
                                from scripts.notifications.dispatcher import get_notifier
                                get_notifier().send_observation({
                                    'symbol': symbol,
                                    'direction': new_side,
                                    'current_price': price,
                                    'stop_loss': float(result.get('stop_loss', 0) or 0),
                                    'take_profit_1': float(result.get('take_profit_1', 0) or 0),
                                    'paper_reason': _why_paper,
                                    'timestamp': datetime.now(timezone.utc).isoformat(),
                                })
                            except Exception:
                                pass
                            self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                            return

                        # (btc_tide already on result — set by Guard T above; the
                        # _open_position half-sizing is belt-and-braces now, since
                        # against-tide fires never reach the real book.)
                        print(f'[{symbol}] MODEL FIRE {new_side} tier={_risk_tier} '
                              f'edge={_edge_q:.0f} srq={_srq:.2f} mode={_entry_mode} '
                              f'scores={result.get("signal_scores", {})}'
                              f'{" warn=" + ",".join(_warn) if _warn else ""}')
                        self._open_position(symbol, result, price, regime, _edge_q,
                                            risk_tier=_risk_tier, entry_mode=_entry_mode,
                                            gate_warnings=_warn)
                        # v76 display commit: fire shows in the cockpit only if the
                        # position actually opened (the RR gate inside _open_position
                        # can still reject — that path must never display as fired).
                        if symbol in self.wallet.open_positions and symbol in self.last_signals:
                            _conf_c = float(result.get('edge_score',
                                                       result.get('meta_confidence', 0)) or 0)
                            _thr_c  = float(result.get('meta_threshold', 65.0) or 65.0)
                            self.last_signals[symbol]['fire']            = True
                            self.last_signals[symbol]['signal']          = new_side
                            self.last_signals[symbol]['evaluating']      = False
                            self.last_signals[symbol]['signal_strength'] = (
                                f'STRONG_{new_side}'
                                if (_conf_c >= _thr_c * 1.15 and _edge_q >= 70.0)
                                else new_side)
                        self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                        return

                    # ── Gate 0: drift monitor — block critically degraded models ─
                    if self.drift_monitor.is_blocked(symbol):
                        live_wr   = self.drift_monitor._live_win_rate(symbol)
                        benchmark = self.drift_monitor._benchmarks.get(symbol, 0.60)
                        print(f'[{symbol}] DRIFT_BLOCKED {new_side}: '
                              f'live_wr={live_wr:.1%} vs benchmark={benchmark:.1%} (CRITICAL)')
                        if symbol in self.last_signals:
                            self.last_signals[symbol]['fire']          = False
                            self.last_signals[symbol]['signal']        = 'HOLD'
                            self.last_signals[symbol]['drift_blocked'] = True
                        self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                        return

                    # ── Gate 1 (CRITICAL): genuinely untradeable regime ──────
                    if regime.regime in self.NO_TRADE_REGIMES:
                        print(f'[{symbol}] NO_TRADE_REGIME={regime.regime} — blocked')
                        if symbol in self.last_signals:
                            self.last_signals[symbol]['fire']           = False
                            self.last_signals[symbol]['signal']         = 'HOLD'
                            self.last_signals[symbol]['regime_blocked'] = True
                        self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                        return

                    # ── Gate 1.2 RETIRED: ranging regime is no longer a risk ──
                    # A range used to be penalised because entries could land
                    # mid-range or chase a fading trend.  The hard structure
                    # gate (1.6) now forces every entry to a confirmed reversal
                    # at support/resistance (or a confirmed breakout) — which
                    # in a range is the intended high-probability setup, not a
                    # risk.  Penalising it here guaranteed every structure-valid
                    # signal carried a warning and could never reach STRONG/
                    # NORMAL tier.  Genuinely bad regimes are still hard-blocked
                    # by Gate 1 (no-trade) and Gate 1.5 (direction-regime veto).

                    # ── Gate 1.5: direction-regime veto (REVERSAL-AWARE) ─────
                    # A counter-trend signal is the model's JOB when it sits at
                    # the structural extreme: SELL at resistance / BUY at support
                    # is a reversal attempt — catching the turn is the entire
                    # point of the prediction.  Do NOT veto it here; the
                    # structure gate (1.6) then demands the actual 5m/15m turn
                    # before it fires, so a genuine exhaustion reversal survives
                    # while an unconfirmed one still waits.
                    # Only veto counter-trend signals with NO structural basis —
                    # mid-trend or at the WRONG extreme.  Shorting a bull mid-move
                    # (not at resistance) is the classic way a high-conviction
                    # model gets wrecked; that stays blocked.
                    # (The old blanket veto killed exactly the reversals the
                    #  engine exists to catch — ADA SELL @ resistance, RSI 97.7,
                    #  edge 93.8, muted purely for being counter-trend, 2026-07-04.)
                    _rv_sup = float(result.get('support', 0) or 0)
                    _rv_res = float(result.get('resistance', 0) or 0)
                    _rv_rp  = result.get('range_position')
                    if _rv_rp is not None:
                        _rv_range_pos = float(_rv_rp)
                    elif 0 < _rv_sup < _rv_res and price > 0:
                        _rv_range_pos = max(0.0, min(1.0,
                                            (price - _rv_sup) / (_rv_res - _rv_sup)))
                    else:
                        _rv_range_pos = 0.5   # location unknown → no reversal exception
                    _sell_at_res = _rv_range_pos >= self.STRUCT_RESISTANCE_ZONE
                    _buy_at_sup  = _rv_range_pos <= self.STRUCT_SUPPORT_ZONE
                    if (new_side == 'BUY' and regime.regime == _REGIME_TRENDING_BEAR
                            and not _buy_at_sup):
                        print(f'[{symbol}] REGIME_VETO blocked BUY: BEAR trend and not at '
                              f'support (rp={_rv_range_pos:.2f}) — no reversal basis')
                        if symbol in self.last_signals:
                            self.last_signals[symbol]['fire']           = False
                            self.last_signals[symbol]['signal']         = 'HOLD'
                            self.last_signals[symbol]['regime_blocked'] = True
                        self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                        return
                    if (new_side == 'SELL' and regime.regime == _REGIME_TRENDING_BULL
                            and not _sell_at_res):
                        print(f'[{symbol}] REGIME_VETO blocked SELL: BULL trend and not at '
                              f'resistance (rp={_rv_range_pos:.2f}) — no reversal basis')
                        if symbol in self.last_signals:
                            self.last_signals[symbol]['fire']           = False
                            self.last_signals[symbol]['signal']         = 'HOLD'
                            self.last_signals[symbol]['regime_blocked'] = True
                        self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                        return

                    # ── Gate 1.6 (CRITICAL): entry location + breakout rules ──
                    # Entries are taken AT structure only:
                    #   BUY at support / SELL at resistance — confirmed by the
                    #   lower timeframes already turning in the signal direction
                    #   (≥3 of last 4 closed 5m candles + 2 of 2 closed 15m).
                    # The single exception is an imminent or confirmed break:
                    #   BUY at resistance (SELL at support) requires EITHER
                    #   full momentum into the level (4/4 5m + 2/2 15m trending)
                    #   OR a completed break-and-retest where the crossed level
                    #   held — old resistance acting as support (mirror for
                    #   SELL).  Mid-range entries are blocked outright.
                    # Fails open when S/R or candle data is unavailable.
                    _sg_verdict, _sg_detail = await self._structure_gate(
                        symbol, new_side, price, result)
                    if _sg_verdict == 'BLOCK':
                        print(f'[{symbol}] STRUCTURE_GATE blocked {new_side}: {_sg_detail}')
                        if symbol in self.last_signals:
                            self.last_signals[symbol]['fire']              = False
                            self.last_signals[symbol]['signal']            = 'HOLD'
                            self.last_signals[symbol]['structure_blocked'] = True
                            self.last_signals[symbol]['structure_reason']  = _sg_detail
                        self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                        return
                    if _sg_verdict == 'WAIT':
                        # Mid-range: the signal is VALID but not yet at its level. Do
                        # NOT discard it — hold it as PENDING. The engine keeps
                        # re-evaluating every scan; when price reaches the target S/R
                        # (resistance for a SELL, support for a BUY) the structure
                        # gate returns PASS/WARN and it fires with the 5m/15m
                        # confirmation. Marked pending (not blocked) so it surfaces as
                        # "waiting for <level>" rather than being thrown away.
                        print(f'[{symbol}] STRUCTURE_GATE pending {new_side}: {_sg_detail}')
                        if symbol in self.last_signals:
                            self.last_signals[symbol]['fire']            = False
                            self.last_signals[symbol]['signal']          = 'HOLD'
                            self.last_signals[symbol]['pending_entry']   = True
                            self.last_signals[symbol]['pending_side']    = new_side
                            self.last_signals[symbol]['pending_reason']  = _sg_detail
                            self.last_signals[symbol]['pending_target']  = (
                                float(result.get('resistance', 0) or 0) if new_side == 'SELL'
                                else float(result.get('support', 0) or 0))
                            self.last_signals[symbol]['structure_blocked'] = False
                        self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                        return
                    # Reaching here means the location is confirmed (PASS/WARN/SKIP) —
                    # clear any stale pending marker from an earlier mid-range scan.
                    if symbol in self.last_signals and self.last_signals[symbol].get('pending_entry'):
                        self.last_signals[symbol]['pending_entry'] = False
                    if _sg_verdict == 'SKIP':
                        # A silent fail-open is how resistance longs slipped
                        # through undetected — every skip is logged, AND it
                        # counts as an advisory warning: an entry whose
                        # structural location could not be verified is, by
                        # definition, higher risk (so it can never be STRONG).
                        print(f'[{symbol}] STRUCTURE_GATE skipped for {new_side}: {_sg_detail}')
                        _gate_warnings.append(f'structure_unverified({_sg_detail})')
                    elif _sg_verdict == 'WARN':
                        # Confirmed reversal at a slightly early location (gap to
                        # the level).  Not blocked — carried as one advisory
                        # objection so it can fire as MODERATE, not muted.
                        print(f'[{symbol}] STRUCTURE_GATE suboptimal for {new_side}: {_sg_detail}')
                        _gate_warnings.append(f'structure_suboptimal({_sg_detail})')
                    _entry_mode = (_sg_detail if _sg_verdict in ('PASS', 'WARN')
                                   else f'GATE_SKIPPED: {_sg_detail}')
                    if symbol in self.last_signals:
                        self.last_signals[symbol]['entry_mode'] = _entry_mode
                        # Surface the swing S/R the structure gate judged against
                        # (it may have overridden the model's 24h high/low) so the
                        # chart's Support/Resistance lines match the gate's levels.
                        self.last_signals[symbol]['support']    = float(result.get('support', 0) or 0)
                        self.last_signals[symbol]['resistance'] = float(result.get('resistance', 0) or 0)

                    # ── Gate 1.75 (ADVISORY): structure + momentum + volume ───
                    # confirmation on the 1h signal timeframe.  BOS/CHoCH, RSI+
                    # MACD divergence and volume climax/absorption each vote on
                    # direction; only a NET contradiction objects (advisory
                    # warning → tier RISKY → held under the no-high-risk rule).
                    # Agreement passes silently and is surfaced for the UI.
                    try:
                        _conf = await self._confirmation_gate(symbol, new_side, result)
                    except Exception as _cf_err:
                        # A bug in the confirmation helpers must never mute a
                        # signal — degrade to NEUTRAL (no objection) and log.
                        print(f'[{symbol}] CONFIRMATION_GATE error (ignored): {_cf_err}')
                        _conf = {'verdict': 'NEUTRAL', 'score': 0.0,
                                 'reason': 'confirmation gate error', 'signals': {}}
                    if symbol in self.last_signals:
                        self.last_signals[symbol]['confirmation'] = _conf
                    if _conf['verdict'] == 'CONFLICT':
                        print(f'[{symbol}] CONFIRMATION_CONFLICT {new_side} '
                              f'score={_conf["score"]:+.0f}: {_conf["reason"]}')
                        _gate_warnings.append(f'confirmation_conflict({_conf["reason"]})')
                    elif _conf['verdict'] == 'CONFIRM':
                        print(f'[{symbol}] CONFIRMATION_OK {new_side} '
                              f'score={_conf["score"]:+.0f}: {_conf["reason"]}')

                    # ── Gate 1.76 (CRITICAL): counter-trend reversal must prove ──
                    # the turn.  A reversal AGAINST a trending regime (buy at
                    # support in a bear / sell at resistance in a bull) is the
                    # riskiest entry: the 5m/15m momentum check is satisfied by a
                    # brief bounce, then the trend resumes — the falling-knife
                    # longs and squeezed shorts that produced the -2.5%/-5% losses.
                    # For these, demand BOTH: (a) the confirmation gate is not
                    # actively opposing (no CONFLICT), and (b) RSI is at a genuine
                    # extreme (oversold for a long / overbought for a short).  A
                    # "reversal" with mid RSI or opposing confirmation is a bounce,
                    # not a turn.  Trend-FOLLOWING breakouts and range reversals
                    # (non-trending regime) are unaffected.
                    _ct_reversal = 'reversal' in (_entry_mode or '').lower() and (
                        (new_side == 'BUY'  and regime.regime == _REGIME_TRENDING_BEAR) or
                        (new_side == 'SELL' and regime.regime == _REGIME_TRENDING_BULL)
                    )
                    if _ct_reversal:
                        _rsi_now = float(result.get('rsi', 50) or 50)
                        # Per-token reversal RSI thresholds from the token's OWN
                        # recent RSI distribution (fallback to the fixed 42/58).
                        _rsi_lo, _rsi_hi = self.REVERSAL_RSI_LONG, self.REVERSAL_RSI_SHORT
                        try:
                            _rc = await self._fetch_candles(symbol, '1h', 120)
                            _rc = _rc[:-1] if len(_rc) >= 2 else _rc
                            if len(_rc) >= 30:
                                _rsi_lo, _rsi_hi = self._dynamic_rsi_bounds(
                                    _rsi_series([float(c[4]) for c in _rc]))
                        except Exception:
                            pass
                        _rsi_ok  = ((_rsi_now <= _rsi_lo) if new_side == 'BUY'
                                    else (_rsi_now >= _rsi_hi))
                        # ADVISORY (not a hard block): S/R reversals are the PRIMARY
                        # setup, so a counter-trend reversal without an RSI extreme
                        # or with opposing confirmation still fires — it is carried
                        # as an advisory objection (higher risk tier), and only a
                        # confirmation CONFLICT *plus* a non-extreme RSI (both bad)
                        # blocks it, which is the genuine falling-knife case.
                        _conf_opp = (_conf['verdict'] == 'CONFLICT')
                        if _conf_opp or not _rsi_ok:
                            _lim = _rsi_lo if new_side == 'BUY' else _rsi_hi
                            _why = ('confirmation opposes' if _conf_opp
                                    else f'RSI {_rsi_now:.0f} not extreme '
                                         f'({"<=" if new_side=="BUY" else ">="}{_lim:.0f})')
                            _gate_warnings.append(f'counter_trend_reversal({_why})')
                        if _conf_opp and not _rsi_ok:
                            print(f'[{symbol}] COUNTER_TREND_REVERSAL blocked {new_side} '
                                  f'in {regime.regime}: confirmation opposes AND RSI not '
                                  f'extreme — falling knife')
                            if symbol in self.last_signals:
                                self.last_signals[symbol]['fire']              = False
                                self.last_signals[symbol]['signal']            = 'HOLD'
                                self.last_signals[symbol]['structure_blocked'] = True
                                self.last_signals[symbol]['structure_reason']  = (
                                    'counter-trend reversal: confirmation opposes and RSI not extreme')
                            self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                            return

                    # ── Gate 1.8 (ADVISORY): higher-timeframe (4h+1d) S/R confluence ─
                    # Trade WITH the big structure, not into it. Pool the nearest
                    # 4h+1d swing S/R and downgrade a BUY that has a major HTF
                    # resistance right overhead (little room to run) or a SELL with
                    # major HTF support right below. Advisory only — adds a warning
                    # (tier down / higher risk), never a hard block, so the 1h fire
                    # rate is unchanged. The levels are surfaced for the chart too.
                    try:
                        _htf = await self._htf_sr(
                            symbol, price, float(result.get('atr', 0) or 0))
                    except Exception:
                        _htf = None
                    if _htf is not None and symbol in self.last_signals:
                        _htf_sup, _htf_res = _htf
                        self.last_signals[symbol]['htf_support']    = _htf_sup
                        self.last_signals[symbol]['htf_resistance'] = _htf_res
                        _atr_h   = float(result.get('atr', 0) or 0) or price * 0.005
                        _rsi_h   = float(result.get('rsi', 50) or 50)
                        _room_up = _htf_res - price
                        _room_dn = price - _htf_sup
                        # HARD block (v45): EXHAUSTION into the HTF wall — buying a
                        # nearby 4h/1d resistance while already OVERBOUGHT (or selling
                        # into a nearby 4h/1d support while OVERSOLD). The first target
                        # sits at/beyond the wall, so it's a low-probability chase at
                        # the top/bottom of an extended move (AAVE LONG, RSI 89, TP1
                        # above the 4H/D resistance).
                        if new_side == 'BUY' and _room_up < 2.5 * _atr_h and _rsi_h > 72:
                            print(f'[{symbol}] HTF_WALL_BLOCK BUY: 4h/1d resistance '
                                  f'{_htf_res:.6g} only {_room_up:.6g} overhead '
                                  f'(<2.5 ATR) at RSI {_rsi_h:.0f} — buying into the wall')
                            self.last_signals[symbol]['fire']        = False
                            self.last_signals[symbol]['signal']      = 'HOLD'
                            self.last_signals[symbol]['htf_blocked'] = True
                            self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                            return
                        if new_side == 'SELL' and _room_dn < 2.5 * _atr_h and _rsi_h < 28:
                            print(f'[{symbol}] HTF_WALL_BLOCK SELL: 4h/1d support '
                                  f'{_htf_sup:.6g} only {_room_dn:.6g} below '
                                  f'(<2.5 ATR) at RSI {_rsi_h:.0f} — selling into the floor')
                            self.last_signals[symbol]['fire']        = False
                            self.last_signals[symbol]['signal']      = 'HOLD'
                            self.last_signals[symbol]['htf_blocked'] = True
                            self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                            return
                        # ADVISORY (unchanged): mild overhead/below without an RSI extreme.
                        if new_side == 'BUY' and _room_up < 1.0 * _atr_h:
                            _gate_warnings.append(f'htf_resistance_overhead({_htf_res:.6g})')
                            print(f'[{symbol}] HTF_CONFLUENCE BUY into 4h/1d '
                                  f'resistance {_htf_res:.6g} (room {_room_up:.6g} < {_atr_h:.6g})')
                        elif new_side == 'SELL' and _room_dn < 1.0 * _atr_h:
                            _gate_warnings.append(f'htf_support_below({_htf_sup:.6g})')
                            print(f'[{symbol}] HTF_CONFLUENCE SELL into 4h/1d '
                                  f'support {_htf_sup:.6g} (room {_room_dn:.6g} < {_atr_h:.6g})')

                    # ── Gate 1.65 RETIRED as a warning: reversal is confirmed ─
                    # by the structure gate's 5m+15m directional check, which
                    # is stronger and fresher evidence than a 1h candlestick
                    # pattern.  Demanding a formal 1h pattern on top of that
                    # added a near-permanent second warning that blocked the
                    # low-risk tiers.  The candlestick reversal score still
                    # feeds score_signal() (+10 strong / +5 weak / -12
                    # opposing), so it promotes STRONG tier and lifts the
                    # quality score — it just no longer forces RISKY on its own.

                    # ── Gate 1.7: HTF macro trend veto (REVERSAL-AWARE) ─────
                    # Hard block when weekly EMA50 AND daily EMA50 both oppose the
                    # signal — BUT, like Gate 1.5, NOT when the signal is a
                    # reversal at the structural extreme (SELL at resistance / BUY
                    # at support).  Catching an exhaustion turn against the macro
                    # trend is a legitimate (higher-risk) reversal — the structure
                    # gate still requires the actual 5m/15m turn, and score_signal
                    # applies the HTF quality penalty so it rates as MODERATE, not
                    # STRONG.  Only mid-trend counter-macro entries (no structural
                    # basis) are hard-vetoed here — those are the "nearly certain
                    # loss" case.  (The old blanket veto killed reversals like the
                    # ADA SELL @ resistance, RSI 97.7, alongside Gate 1.5.)
                    _htf_w = float(result.get('macro_weekly', 0.0) or 0.0)
                    _htf_d = float(result.get('macro_daily',  0.0) or 0.0)
                    _htf_data_ok = (_htf_w != 0.0 or _htf_d != 0.0)
                    if _htf_data_ok:
                        if (new_side == 'BUY' and _htf_w < -0.5 and _htf_d < -0.5
                                and not _buy_at_sup):
                            print(f'[{symbol}] HTF_VETO blocked BUY: weekly={_htf_w:+.1f} '
                                  f'daily={_htf_d:+.1f} both bearish and not at support')
                            if symbol in self.last_signals:
                                self.last_signals[symbol]['fire']        = False
                                self.last_signals[symbol]['signal']      = 'HOLD'
                                self.last_signals[symbol]['htf_blocked'] = True
                            self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                            return
                        if (new_side == 'SELL' and _htf_w > 0.5 and _htf_d > 0.5
                                and not _sell_at_res):
                            print(f'[{symbol}] HTF_VETO blocked SELL: weekly={_htf_w:+.1f} '
                                  f'daily={_htf_d:+.1f} both bullish and not at resistance')
                            if symbol in self.last_signals:
                                self.last_signals[symbol]['fire']        = False
                                self.last_signals[symbol]['signal']      = 'HOLD'
                                self.last_signals[symbol]['htf_blocked'] = True
                            self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                            return

                    # ── Gate 2: ATR floor — stops would be inside tick noise ──
                    _fe_atr_pct = float(result.get('atr_pct', 0.0))
                    if _fe_atr_pct < 0.5:
                        print(f'[{symbol}] ATR_TOO_LOW blocked {new_side} '
                              f'atr_pct={_fe_atr_pct:.2f}%')
                        if symbol in self.last_signals:
                            self.last_signals[symbol]['fire']        = False
                            self.last_signals[symbol]['signal']      = 'HOLD'
                            self.last_signals[symbol]['atr_blocked'] = True
                        self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                        return

                    # ── Gate 3: model edge score floor (drift-adjusted) ──────
                    # confidence_penalty() returns 0.0–0.10 in probability space;
                    # ×100 converts to edge-score points (floor raised 0–10 pts).
                    # WARNING drift (10–20 pp below benchmark) thus proportionally
                    # tightens the gate without a hard block — CRITICAL still blocks
                    # entirely via Gate 0 (drift_monitor.is_blocked).
                    _model_quality = min(float(result.get('edge_score', 0.0)), 100.0)
                    _drift_pen     = self.drift_monitor.confidence_penalty(symbol) * 100.0
                    _edge_floor    = SignalQualityFilter.MIN_QUALITY_SCORE + _drift_pen
                    if _model_quality < _edge_floor:
                        _drift_note = f' (drift+{_drift_pen:.1f}pts)' if _drift_pen > 0 else ''
                        print(f'[{symbol}] QUALITY_GATE blocked {new_side}: '
                              f'edge={_model_quality:.1f} < {_edge_floor:.1f}{_drift_note}')
                        if symbol in self.last_signals:
                            self.last_signals[symbol]['fire']            = False
                            self.last_signals[symbol]['signal']          = 'HOLD'
                            self.last_signals[symbol]['quality_blocked'] = True
                        self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                        return

                    # ── Gate 3b (ADVISORY): contextual quality filter ─────────
                    # score_signal() checks: ADX trend, volume conviction, regime
                    # confidence, RSI zone, funding alignment, OI alignment, market
                    # bias, RANGING penalty (-15), low-volume penalty (-10), macro
                    # conflict penalty (-15), candlestick reversal alignment.
                    _CONTEXT_FLOOR = 70.0
                    # When HMM detects elevated regime-transition risk, raise the
                    # context floor proportionally.  A transition risk > 0.50 means
                    # the current regime may flip mid-trade; entries at these points
                    # have systematically lower expected win rates.
                    # Max adjustment: +20 pts at trans_risk = 1.0 → floor = 90.
                    if _hmm_available and _hmm_trans_risk > 0.50:
                        _trans_adj    = min((_hmm_trans_risk - 0.50) * 40.0, 20.0)
                        _CONTEXT_FLOOR = _CONTEXT_FLOOR + _trans_adj
                    if quality_score < _CONTEXT_FLOOR:
                        _qr_str = ', '.join(_quality_reasons[:4]) if _quality_reasons else 'n/a'
                        _gate_warnings.append(
                            f'context(ctx={quality_score:.1f}<{_CONTEXT_FLOOR:.0f} [{_qr_str}])')

                    # ── Gate 3c (ADVISORY): fake breakout / exhaustion ────────
                    if fake_breakout:
                        _gate_warnings.append('fake_breakout')

                    # ── Gate 3.5: safe-mode quality escalation ───────────────
                    # When last 3 global trades are all losses, raise the quality
                    # floor to 80 to protect capital during drawdowns.
                    if self.perf_tracker.safe_mode_active() and _model_quality < 80.0:
                        print(f'[{symbol}] SAFE-MODE blocked {new_side}: '
                              f'edge={_model_quality:.1f} < 80 (elevated floor)')
                        if symbol in self.last_signals:
                            self.last_signals[symbol]['fire']   = False
                            self.last_signals[symbol]['signal'] = 'HOLD'
                        self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                        return

                    # ── Gate 3.8 (ADVISORY): signal stability ─────────────────
                    # Prefer N consecutive same-direction model outputs before entry.
                    # Very high conviction signals (edge >= SIGNAL_BYPASS_EDGE) skip
                    # this check — they're strong enough to act on immediately.
                    _sig_hist = list(self._signal_history.get(symbol, []))
                    _is_stable = (
                        len(_sig_hist) >= self.SIGNAL_STABILITY_WINDOW and
                        all(s == new_side for s in _sig_hist)
                    )
                    if not _is_stable and _model_quality < self.SIGNAL_BYPASS_EDGE:
                        _gate_warnings.append(
                            f'stability({len(_sig_hist)}/{self.SIGNAL_STABILITY_WINDOW} cycles)')

                    # ── Gate 4: portfolio capital limits ──────────────────────
                    # Use the model's edge_score (0-100) directly for position sizing.
                    self.portfolio_guard.sync_from_wallet(self.wallet.open_positions)
                    _pos_est = self.risk_engine.calculate_position_size(
                        self.wallet.balance, _model_quality, regime,
                        _fe_atr_pct if _fe_atr_pct > 0 else 1.5)
                    _pg_allowed, _pg_reason = self.portfolio_guard.can_open(
                        symbol, self.wallet.balance, _pos_est)
                    if not _pg_allowed:
                        print(f'[{symbol}] PORTFOLIO_GUARD blocked {new_side}: {_pg_reason}')
                        if symbol in self.last_signals:
                            self.last_signals[symbol]['fire']              = False
                            self.last_signals[symbol]['signal']            = 'HOLD'
                            self.last_signals[symbol]['portfolio_blocked'] = True
                        self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                        return

                    # (The old Gate 4.5 multi-timeframe timing check was
                    #  superseded by Gate 1.6's 5m/15m direction confirmation —
                    #  every entry now proves its lower-timeframe alignment at
                    #  the structure gate, so a second overlapping check would
                    #  only double-count the same candles.)

                    # ── RSI exhaustion / deceleration (ADVISORY) ─────────────
                    # Warn when RSI momentum has already peaked (late signal).
                    _rsi_val   = float(result.get('rsi', 50) or 50)
                    _rsi_slope = float(result.get('rsi_slope', 0) or 0)
                    _rsi_accel = float(result.get('rsi_acceleration', 0) or 0)
                    _exhausted = False
                    _exhaust_reason = ''
                    if new_side == 'BUY':
                        if _rsi_val >= 72 and _rsi_slope <= 0:
                            _exhausted = True
                            _exhaust_reason = f'RSI top rsi={_rsi_val:.1f} slope={_rsi_slope:.4f}'
                        elif _rsi_accel < -0.08:
                            _exhausted = True
                            _exhaust_reason = f'RSI decel buy accel={_rsi_accel:.4f}'
                    elif new_side == 'SELL':
                        if _rsi_val <= 28 and _rsi_slope >= 0:
                            _exhausted = True
                            _exhaust_reason = f'RSI bottom rsi={_rsi_val:.1f} slope={_rsi_slope:.4f}'
                        elif _rsi_accel > 0.08:
                            _exhausted = True
                            _exhaust_reason = f'RSI decel sell accel={_rsi_accel:.4f}'
                    if _exhausted:
                        _gate_warnings.append(f'exhaustion({_exhaust_reason})')

                    # ── Coordinative advisory policy — block only HIGH risk ───
                    # The advisory gates VOTE; they do not each veto.  A signal
                    # tolerates up to ADVISORY_WARNING_BUDGET (3) objections and
                    # still publishes (LOW / MODERATE risk); one more makes it
                    # HIGH risk and it is held:
                    #   0 objections   → STRONG  (LOW risk)
                    #   1-3 objections → NORMAL  (MODERATE risk) — still fires
                    #   4+ objections  → RISKY   (HIGH risk) — held
                    _n_warn = len(_gate_warnings)
                    if _n_warn > self.ADVISORY_WARNING_BUDGET:
                        print(f'[{symbol}] HIGH_RISK_HOLD blocked {new_side}: '
                              f'{_n_warn} advisory warnings (> budget '
                              f'{self.ADVISORY_WARNING_BUDGET}) — HIGH risk; only '
                              f'LOW/MODERATE fire [{"; ".join(_gate_warnings)}]')
                        if symbol in self.last_signals:
                            self.last_signals[symbol]['fire']          = False
                            self.last_signals[symbol]['signal']        = 'HOLD'
                            self.last_signals[symbol]['risky_blocked'] = True
                            self.last_signals[symbol]['risk_tier']     = 'RISKY'
                            self.last_signals[symbol]['gate_warnings'] = _gate_warnings
                        self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                        return

                    # ── Tier label: STRONG is reserved, not a default ─────────
                    # STRONG (published as "LOW RISK / STRONG SIGNAL") must mean a
                    # genuinely high-conviction, trend-ALIGNED setup — otherwise a
                    # clean-but-ordinary or counter-trend entry that loses stains
                    # the STRONG badge (the "STRONG BUY that lost" case).  STRONG
                    # requires: zero advisory objections AND top conviction
                    # (edge≥85 or quality≥80) AND not counter-trend.  A
                    # counter-trend reversal (BUY in a bear regime / SELL in a
                    # bull regime — allowed at the S/R extreme by Gate 1.5/1.7) is
                    # inherently a moderate-risk contrarian bet, so it caps at
                    # NORMAL no matter how clean it looks.
                    _counter_trend = (
                        (new_side == 'BUY'  and regime.regime == _REGIME_TRENDING_BEAR) or
                        (new_side == 'SELL' and regime.regime == _REGIME_TRENDING_BULL)
                    )
                    if (_n_warn == 0 and not _counter_trend
                            and (_model_quality >= self.SIGNAL_BYPASS_EDGE
                                 or quality_score >= 80.0)):
                        _risk_tier = 'STRONG'   # LOW risk — clean, high-conviction, trend-aligned
                    else:
                        _risk_tier = 'NORMAL'   # MODERATE risk

                    # ── Model approved — open the position ────────────────────
                    # predictor.predict_realtime() already applied:
                    #   meta_threshold_buy/sell, hold_calibrator, regime_thresholds
                    # Trusting the model's fire=True directly.
                    _warn_note = (f' warnings={len(_gate_warnings)} '
                                  f'[{"; ".join(_gate_warnings)}]'
                                  if _gate_warnings else '')
                    print(f'[{symbol}] MODEL PASS {new_side} tier={_risk_tier} '
                          f'edge={_model_quality:.1f} atr={_fe_atr_pct:.2f}% '
                          f'regime={regime.regime}{_warn_note}')
                    if symbol in self.last_signals:
                        self.last_signals[symbol]['gate_warnings'] = _gate_warnings
                        self.last_signals[symbol]['risk_tier']     = _risk_tier
                    self._open_position(symbol, result, price, regime, _model_quality,
                                        risk_tier=_risk_tier, entry_mode=_entry_mode,
                                        gate_warnings=_gate_warnings)

                elif is_flip:
                    print(f'[{symbol}] FLIP-FLOP BLOCKED {last_side}→{new_side} '
                          f'({int((required_cooldown - cooldown_elapsed)/60)} min remaining)')
                    if symbol in self.last_signals:
                        self.last_signals[symbol]['fire']            = False
                        self.last_signals[symbol]['signal']          = 'HOLD'
                        self.last_signals[symbol]['cooldown_blocked'] = True
                else:
                    if symbol in self.last_signals:
                        self.last_signals[symbol]['fire']            = False
                        self.last_signals[symbol]['signal']          = 'HOLD'
                        self.last_signals[symbol]['cooldown_blocked'] = True

            self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)

    # ── multi-timeframe candle fetch ──────────────────────────────────────────

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
            return [(lvl, int(n)) for lvl, n in merged
                    if n >= self.PENDING_TARGET_MIN_TOUCHES]
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

    # ── structure gate (Gate 1.6) ─────────────────────────────────────────────

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

        # ── CASE 1: CORRECT LEVEL (BUY at support / SELL at resistance) ────────
        # Primary setup — fire immediately at/near the level.
        if correct_level:
            level = support if bullish else resistance
            dist  = abs(price - level)
            # v46: REJECTION FAST-PATH — price TAGGED the level (a recent wick reached
            # it) and has now moved MORE than STRUCT_REJECTION_PCT (10%) of the range
            # back off it: a SELL >10% BELOW the resistance it hit, a BUY >10% ABOVE
            # the support it hit. The rejection itself IS the confirmation, so it
            # fires immediately — bypassing the counter-trend 2-candle wait and the
            # "far from level" pending. (User: sell once price has come back >10% off
            # a resistance it reached; buy the mirror off support.)
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
                if confirmed:
                    return 'PASS', f'{_lname}_reversal confirmed (5m {n5}/{self.STRUCT_5M_WINDOW}, 15m {n15}/{self.STRUCT_15M_WINDOW}) @ {level:.6g}'
                if partly_confirmed:
                    return 'PASS', f'{_lname}_reversal 5m-confirmed (5m {n5}/{self.STRUCT_5M_WINDOW}) @ {level:.6g}'
                # v44: counter-trend reversal needs >=2 5m candles turned — else WAIT
                # (pending) so we don't short a rising candle / buy a falling one.
                if counter_trend and n5 < 2:
                    return 'WAIT', (f'counter-trend {_lname}_reversal @ {level:.6g} — '
                                    f'waiting for 2+ 5m candles to turn ({n5}/2)')
                return 'WARN', f'{_lname}_reversal unconfirmed @ {level:.6g} — tighter SL'
            # v43: still far from the level -> PENDING (wait for a closer entry)
            # instead of firing an early entry with a wide stop. Pending keeps the
            # signal and fires it when price reaches the level (better fill, higher WR).
            return 'WAIT', f'{_lname}_reversal far (dist {dist:.6g}) — waiting for a closer entry'

        # ── CASE 2: BREAKOUT LEVEL (BUY at resistance / SELL at support) ───────
        # Fire on retest, sustained momentum, or pre-break; else fire tagged weak.
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
            if abs(level - price) <= atr_g * 0.5:
                return 'PASS', f'pre_breakout (dist {abs(level - price):.6g} to {level:.6g}) — early entry, wide stop'
            # v43: an UNCONFIRMED wrong-side breakout (no retest-hold, no sustained
            # momentum, not a pre-break) -> PENDING, not a blind fire into the level.
            # Pending waits for the correct level (support for a BUY / resistance for
            # a SELL), so it converts to a clean reversal entry instead of a knife.
            if beyond:
                return 'WAIT', f'post_break @ {level:.6g} — waiting for a retest-hold'
            return 'WAIT', f'breakout_unconfirmed @ {level:.6g} — waiting for retest/momentum'

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

    # ── confirmation gate (Gate 1.7) ──────────────────────────────────────────
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
        raw = await self._fetch_candles(symbol, '1h', self.CONFIRM_1H_BARS + 2)
        closed = raw[:-1] if len(raw) >= 2 else []
        if len(closed) < self.CONFIRM_MIN_BARS:
            return {'verdict': 'NEUTRAL', 'score': 0.0,
                    'reason': f'insufficient 1h data ({len(closed)} bars)',
                    'signals': {}}

        bos = _detect_bos_choch(closed, lookback=self.CONFIRM_BOS_LOOKBACK)
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

    # ── trade management ──────────────────────────────────────────────────────

    def _manage_exit(self, symbol: str, pos: Position,
                     result: Dict[str, Any], price: float) -> None:
        live_px     = self.live_prices.get(symbol, 0.0)
        check_price = live_px if live_px > 0 else price

        now  = time.time()
        held = now - self._open_time.get(symbol, 0)

        # Current ATR: use live result first, then fall back to the ATR stored at
        # entry (pos.atr), then a price-based estimate.
        atr = (float(result.get('atr') or 0)
               or pos.atr
               or pos.entry_price * 0.015)

        # ── Update peak price for trailing-stop tracking ──────────────────────
        # LONG: track the highest price; SHORT: track the lowest price (trough).
        if pos.direction == 'LONG':
            self._peak_price[symbol] = max(
                self._peak_price.get(symbol, pos.entry_price), check_price)
        else:
            self._peak_price[symbol] = min(
                self._peak_price.get(symbol, pos.entry_price), check_price)

        peak = self._peak_price[symbol]

        # ── Helper: full close (removes position, cleans all TP-hit state) ────
        def _close(reason: str, exit_px: Optional[float] = None) -> None:
            rec = self.wallet.close_trade(
                symbol, exit_px if exit_px is not None else check_price, reason)
            if rec:
                self._last_close_time[symbol]   = now
                self._last_close_side[symbol]   = pos.side
                self._last_close_reason[symbol] = reason
                for d in (self._tp1_hit, self._tp2_hit,
                          self._tp3_hit, self._tp4_hit, self._peak_price):
                    d.pop(symbol, None)
                # Whole-trade view: rec.outcome already accounts for banked
                # TP partials (close_trade); aggregate PnL across all slices
                # of this signal_id so logs/alerts report the REAL result.
                _slices   = [t for t in self.wallet.trade_history
                             if t.signal_id == rec.signal_id]
                _tot_usdt = sum(t.pnl_usdt for t in _slices)
                _tot_val  = sum(t.position_value for t in _slices)
                if _tot_usdt < 0:                       # whole trade lost -> bench the token
                    self._last_loss_time[symbol] = now
                _tot_pct  = (_tot_usdt / _tot_val * 100) if _tot_val > 0 else rec.pnl_pct
                tag = rec.outcome
                _slice_note = (f' (final slice {rec.pnl_pct:+.2f}%)'
                               if len(_slices) > 1 else '')
                print(f'[{symbol}] {reason} {tag} {_tot_pct:+.2f}%{_slice_note} @ '
                      f'{(exit_px or check_price):.6g}')
                self.perf_tracker.record_outcome(
                    symbol        = symbol,
                    regime        = self.last_signals.get(symbol, {}).get('regime', 'UNKNOWN'),
                    outcome       = rec.outcome,
                    pnl_pct       = round(_tot_pct, 3),
                    quality_score = float(self.last_signals.get(symbol, {}).get('quality_score', 0)),
                )
                self.drift_monitor.record(symbol, rec.outcome)
                self.drift_monitor.save_state()
                drift_sev = self.drift_monitor.severity(symbol)
                if drift_sev in ('WARNING', 'CRITICAL'):
                    live_wr   = self.drift_monitor._live_win_rate(symbol)
                    benchmark = self.drift_monitor._benchmarks.get(symbol, 0.60)
                    print(f'[{symbol}] DRIFT {drift_sev}: '
                          f'live_wr={live_wr:.1%} benchmark={benchmark:.1%} '
                          f'(drop={((benchmark - (live_wr or 0)) * 100):.1f}pp)')
                self._save_track_record()
                try:
                    from scripts.notifications.dispatcher import get_notifier
                    _hold = int(time.time() - self._open_time.get(symbol, time.time()))
                    get_notifier().send_exit(
                        symbol=symbol, direction=pos.side, outcome=tag,
                        pnl_pct=round(_tot_pct, 3), hold_seconds=_hold,
                        exit_reason=reason,
                    )
                except Exception:
                    pass

        # ── Helper: partial close (keeps position open for next TP levels) ────
        def _partial(reason: str, pct: float, exit_px: Optional[float] = None) -> None:
            px  = exit_px if exit_px is not None else check_price
            rec = self.wallet.partial_close_trade(symbol, px, reason, pct)
            if rec:
                print(f'[{symbol}] {reason} {rec.pnl_pct:+.2f}% '
                      f'(closed {pct*100:.0f}% @ {px:.6g}) '
                      f'remaining≈{pos.position_value:.0f} USDT')
                self._save_track_record()
                # TP hits are outcome events — keep subscribers' Telegram in
                # sync with the position as profit is banked, not just at the
                # final close.
                try:
                    from scripts.notifications.dispatcher import get_notifier
                    _hold = int(time.time() - self._open_time.get(symbol, time.time()))
                    get_notifier().send_exit(
                        symbol=symbol, direction=pos.side, outcome=rec.outcome,
                        pnl_pct=rec.pnl_pct, hold_seconds=_hold,
                        exit_reason=f'{reason} ({pct*100:.0f}% closed, position still open)',
                    )
                except Exception:
                    pass

        # ── 1. Maximum hold time (zombie guard) ──────────────────────────────
        if held >= self.MAX_HOLD_SECONDS:
            _close('MAX_HOLD_EXPIRED')
            return

        # ── 2. TP5 hit — close all remaining size ────────────────────────────
        # By this point TPs 1-4 have already taken 80 %; this closes the last 20 %.
        if pos.take_profit_5 > 0:
            tp5_hit = (
                (pos.direction == 'LONG'  and check_price >= pos.take_profit_5) or
                (pos.direction == 'SHORT' and check_price <= pos.take_profit_5)
            )
            tp5_via_peak = self._tp4_hit.get(symbol, False) and (
                (pos.direction == 'LONG'  and peak >= pos.take_profit_5) or
                (pos.direction == 'SHORT' and peak <= pos.take_profit_5)
            )
            if tp5_hit:
                _close('TP5_HIT')
                return
            if tp5_via_peak:
                _close('TP5_HIT', exit_px=pos.take_profit_5)
                return

        # ── 3. TP4 hit — 20 % partial close ──────────────────────────────────
        if pos.take_profit_4 > 0 and not self._tp4_hit.get(symbol, False):
            tp4_hit = (
                (pos.direction == 'LONG'  and check_price >= pos.take_profit_4) or
                (pos.direction == 'SHORT' and check_price <= pos.take_profit_4)
            )
            tp4_via_peak = self._tp3_hit.get(symbol, False) and (
                (pos.direction == 'LONG'  and peak >= pos.take_profit_4) or
                (pos.direction == 'SHORT' and peak <= pos.take_profit_4)
            )
            if tp4_hit or tp4_via_peak:
                exit_px = pos.take_profit_4 if tp4_via_peak else None
                _partial('TP4_PARTIAL', self.risk_engine.TP_CLOSE_PCTS[3], exit_px)
                self._tp4_hit[symbol] = True

        # ── 4. TP3 hit — 20 % partial close ──────────────────────────────────
        if pos.take_profit_3 > 0 and not self._tp3_hit.get(symbol, False):
            tp3_hit = (
                (pos.direction == 'LONG'  and check_price >= pos.take_profit_3) or
                (pos.direction == 'SHORT' and check_price <= pos.take_profit_3)
            )
            tp3_via_peak = self._tp2_hit.get(symbol, False) and (
                (pos.direction == 'LONG'  and peak >= pos.take_profit_3) or
                (pos.direction == 'SHORT' and peak <= pos.take_profit_3)
            )
            if tp3_hit or tp3_via_peak:
                exit_px = pos.take_profit_3 if tp3_via_peak else None
                _partial('TP3_PARTIAL', self.risk_engine.TP_CLOSE_PCTS[2], exit_px)
                self._tp3_hit[symbol] = True

        # ── 5. TP2 hit — 20 % partial close, activate trailing stop ──────────
        # Peak-based detection applies only after TP1 is confirmed (real move).
        if pos.take_profit_2 > 0 and not self._tp2_hit.get(symbol, False):
            tp2_hit = (
                (pos.direction == 'LONG'  and check_price >= pos.take_profit_2) or
                (pos.direction == 'SHORT' and check_price <= pos.take_profit_2)
            )
            tp2_via_peak = self._tp1_hit.get(symbol, False) and (
                (pos.direction == 'LONG'  and peak >= pos.take_profit_2) or
                (pos.direction == 'SHORT' and peak <= pos.take_profit_2)
            )
            if tp2_hit or tp2_via_peak:
                exit_px = pos.take_profit_2 if tp2_via_peak else None
                _partial('TP2_PARTIAL', self.risk_engine.TP_CLOSE_PCTS[1], exit_px)
                self._tp2_hit[symbol] = True
                print(f'[{symbol}] TRAILING activated @ TP2 — '
                      f'trail distance = ATR×{self.risk_engine.TRAIL_MULTIPLIER}')

        # ── 6. Trailing stop — active after TP2 is hit ───────────────────────
        # Trail distance = ATR × TRAIL_MULTIPLIER.
        # Floor = TP2 price so the stop never gives back more than one ATR of
        # TP2 profit once the trailing level is above TP2.
        if self._tp2_hit.get(symbol, False):
            trail_dist = self.risk_engine.TRAIL_MULTIPLIER * atr
            if pos.direction == 'LONG':
                trail_stop = max(pos.take_profit_2, peak - trail_dist)
                if check_price <= trail_stop:
                    _close('TRAILING_STOP', exit_px=trail_stop)
                    return
            else:  # SHORT
                trail_stop = min(pos.take_profit_2, peak + trail_dist)
                if check_price >= trail_stop:
                    _close('TRAILING_STOP', exit_px=trail_stop)
                    return

        # ── 7. TP1 hit — 20 % partial close, move SL to break-even ──────────
        # Break-even: update pos.stop_loss to entry price so that if price
        # reverses all the way back, we exit at entry (no loss on the position).
        if pos.take_profit_1 > 0 and not self._tp1_hit.get(symbol, False):
            tp1_hit = (
                (pos.direction == 'LONG'  and check_price >= pos.take_profit_1) or
                (pos.direction == 'SHORT' and check_price <= pos.take_profit_1)
            )
            if tp1_hit:
                _partial('TP1_PARTIAL', self.risk_engine.TP_CLOSE_PCTS[0])
                self._tp1_hit[symbol]    = True
                self._peak_price[symbol] = check_price   # reset peak tracking from TP1
                # Break-even: SL moves to entry — guarantees no loss on remaining position
                pos.stop_loss = pos.entry_price
                print(f'[{symbol}] TP1_HIT @ {check_price:.6g} — '
                      f'SL moved to break-even ({pos.entry_price:.6g})')

        # ── 7b. TP1 re-cross exit — protect profit after TP1 is secured ─────
        # After TP1 hit (SL at break-even), if price crosses back below TP1
        # (LONG) or above TP1 (SHORT) AND the model is no longer confirming the
        # original direction, close at the TP1 level.  This locks in the partial
        # TP1 gain instead of waiting for price to grind all the way to break-even.
        # Inactive after TP2 — the trailing stop manages profit protection then.
        if (self._tp1_hit.get(symbol, False) and
                not self._tp2_hit.get(symbol, False)):
            _model_side  = str(result.get('side', 'FLAT') or 'FLAT').upper()
            _orig_dir_sig = 'BUY' if pos.direction == 'LONG' else 'SELL'
            _tp1_recross = (
                (pos.direction == 'LONG'  and check_price < pos.take_profit_1) or
                (pos.direction == 'SHORT' and check_price > pos.take_profit_1)
            )
            if _tp1_recross and _model_side != _orig_dir_sig:
                _close('TP1_RECROSS', exit_px=pos.take_profit_1)
                return

        # ── 8. Model-reversal exit (dynamic exit on opposing signal) ─────────
        side = result.get('side', 'FLAT')
        fire = bool(result.get('fire', False))
        opposite = (
            (pos.direction == 'LONG'  and side == 'SELL' and fire) or
            (pos.direction == 'SHORT' and side == 'BUY'  and fire)
        )
        if opposite:
            # Require minimum hold unless TP1 is already secured.
            _reversal_min = 0 if self._tp1_hit.get(symbol, False) else self.MIN_HOLD_SECONDS
            if held >= _reversal_min:
                # Guard against low-conviction one-candle noise without depending on
                # consecutive-cycle counting (which breaks when the model outputs FLAT
                # between reversal cycles, clearing _signal_history each time).
                # Instead, require the reversal signal to meet the same quality floor
                # used for entry (edge_score >= MIN_QUALITY_SCORE = 70).
                # After TP1 (SL at break-even, profit secured): close immediately.
                _tp1_secured = self._tp1_hit.get(symbol, False)
                _rev_edge    = float(result.get('edge_score', 0.0))
                if _tp1_secured or _rev_edge >= SignalQualityFilter.MIN_QUALITY_SCORE:
                    _close('MODEL_REVERSAL_TP')
                    return
                print(f'[{symbol}] REVERSAL_GATE deferred {pos.direction}→{side}: '
                      f'reversal edge={_rev_edge:.1f} < '
                      f'{SignalQualityFilter.MIN_QUALITY_SCORE:.0f} (tp1_secured=False)')

        # ── 9. Stop loss / break-even SL ─────────────────────────────────────
        # Before TP1: uses the original ATR-based SL.
        # After  TP1: pos.stop_loss was moved to entry_price (break-even).
        if pos.stop_loss > 0:
            sl_hit = (
                (pos.direction == 'LONG'  and check_price <= pos.stop_loss) or
                (pos.direction == 'SHORT' and check_price >= pos.stop_loss)
            )
            if sl_hit:
                _close('STOP_HIT')

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
        if regime is not None and quality_score > 0:
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
        if (side == 'SELL' and _tide == 'UP') or (side == 'BUY' and _tide == 'DOWN'):
            pos_value = round(pos_value * 0.5, 2)
            print(f'[{symbol}] TIDE_HALF {side}: BTC 4h tide is {_tide} — half size')

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

        pos = Position(
            symbol          = symbol,
            direction       = direction,
            side            = side,
            entry_price     = round(price, 8),
            position_value  = round(pos_value, 2),
            stop_loss       = round(stop_loss, 8),
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
        )
        self.wallet.open_trade(pos)
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

    # ── signal entry builder (for dashboard / last_signals) ───────────────────

    @staticmethod
    def _build_signal_entry(
        symbol:        str,
        result:        Dict[str, Any],
        price:         float,
        regime:        Optional[RegimeState] = None,
        quality_score: float                 = 0.0,
        fake_breakout: bool                  = False,
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

        # Expected move projection
        _conf_data  = result.get('confluence') or {}
        _conf_total = float(_conf_data.get('total', 5.0))
        _conf_raw   = abs(_conf_total - 5.0) / 5.0
        entry['expected_move_pct'] = round(_conf_raw * atr_pct * 3.0, 2)

        # Risk/Reward ratio: reward measured to TP5 (full-trend target), matching
        # the MIN_RISK_REWARD gate in _open_position() / calculate_stops().
        sl_val  = entry.get('suggested_sl') or 0
        tp5_val = entry.get('tp5') or 0
        if price > 0 and sl_val and tp5_val:
            _risk   = abs(price - sl_val)
            _reward = abs(price - tp5_val)
            entry['risk_reward'] = round(_reward / _risk, 2) if _risk > 0 else 0
        else:
            entry['risk_reward'] = 0
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

        return entry

    # ── Alpha Mode: multi-timeframe scanning ─────────────────────────────────

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
            sig = self._build_signal_entry(
                symbol, result, price, regime=regime,
                quality_score=min(float(result.get('edge_score', 0.0)), 100.0),
            )
            sig['timeframe'] = tf
            sig['pair']      = symbol

            key = f'{symbol}|{tf}'
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

    # ── track record persistence ──────────────────────────────────────────────

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

    async def shutdown(self) -> None:
        self._save_track_record()
        self._executor.shutdown(wait=False)
        print('[LiveEngine] Shutdown complete.')


# =============================================================================
# ScalpBot — 5m + 15m raw-prediction scalping engine (no gates)
# =============================================================================

_SCALP_RECORD_PATH = _STATE_DIR / 'scalp_trades.json'

class ScalpBot:
    """
    Lightweight scalping bot that fires on raw model prediction (no gates).
    Uses the existing TraderEngine models (5m scalping + 15m via same weights).
    Runs every 60 seconds independently from the main AEGIS scan cycle.
    """

    SCAN_INTERVAL = 60  # seconds
    POSITION_USDT = 100.0  # fixed notional per trade

    def __init__(self) -> None:
        self._open: Dict[str, Dict[str, Any]] = {}   # key = "SYM_mode"
        self._history: List[Dict[str, Any]] = []
        self._engine: Optional[Any] = None
        self._last_scan: float = 0.0
        self._firestore_db: Optional[Any] = None
        self._load_record()

    # ── persistence ──────────────────────────────────────────────────────────

    def _load_record(self) -> None:
        if _SCALP_RECORD_PATH.exists():
            try:
                data = json.loads(_SCALP_RECORD_PATH.read_text())
                self._history = [t for t in data.get('trades', [])
                                 if t.get('outcome') in ('WIN', 'LOSS')]
            except Exception:
                pass

    def _save_record(self) -> None:
        all_trades = self._history + list(self._open.values())
        won  = sum(1 for t in self._history if t.get('outcome') == 'WIN')
        lost = sum(1 for t in self._history if t.get('outcome') == 'LOSS')
        payload = {
            'last_updated': datetime.now(timezone.utc).isoformat(),
            'total_trades': won + lost,
            'won': won, 'lost': lost,
            'win_rate': round(won / (won + lost), 3) if (won + lost) else 0.0,
            'trades': all_trades,
        }
        import os as _os
        tmp = str(_SCALP_RECORD_PATH) + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(payload, f, indent=2, default=str)
        _os.replace(tmp, _SCALP_RECORD_PATH)

    # ── firebase ──────────────────────────────────────────────────────────────

    def _get_db(self) -> Optional[Any]:
        if self._firestore_db is not None:
            return self._firestore_db
        try:
            import firebase_admin
            from firebase_admin import credentials as _creds, firestore as _fs
            cred_path = _ROOT / 'config' / 'serviceAccountKey.json'
            if not cred_path.exists():
                return None
            if not firebase_admin._apps:
                firebase_admin.initialize_app(_creds.Certificate(str(cred_path)))
            self._firestore_db = _fs.client()
            return self._firestore_db
        except Exception:
            return None

    def _push_signal(self, sig: Dict[str, Any]) -> None:
        db = self._get_db()
        if db is None:
            return
        try:
            doc_id = sig['symbol'].replace('/', '_') + '_' + sig['mode']
            db.collection('scalp_signals').document(doc_id).set(
                {k: v for k, v in sig.items() if v is not None}
            )
        except Exception:
            pass

    # ── scan ──────────────────────────────────────────────────────────────────

    def _load_engine(self) -> bool:
        if self._engine is not None:
            return True
        try:
            from scripts.trader_model.trader_engine import TraderEngine
            self._engine = TraderEngine()
            self._engine.load_models()
            if not self._engine.model_store.loaded_modes:
                self._engine = None
                return False
            print('[ScalpBot] Trader models loaded:', self._engine.model_store.loaded_modes)
            return True
        except Exception as e:
            print(f'[ScalpBot] Failed to load trader engine: {e}')
            return False

    def _check_exits(self, live_prices: Dict[str, float]) -> None:
        closed_keys = []
        for key, pos in self._open.items():
            sym   = pos['symbol']
            price = live_prices.get(sym, 0.0)
            if price <= 0:
                continue
            entry = pos['entry_price']
            sl    = pos['sl']
            tp    = pos['tp']
            direction = pos['direction']

            hit_sl = (direction == 'BUY' and price <= sl) or (direction == 'SELL' and price >= sl)
            hit_tp = (direction == 'BUY' and price >= tp) or (direction == 'SELL' and price <= tp)

            if hit_sl or hit_tp:
                pnl_pct = ((price - entry) / entry * 100) if direction == 'BUY' \
                          else ((entry - price) / entry * 100)
                outcome = 'WIN' if hit_tp else 'LOSS'
                closed = {**pos,
                          'exit_price': round(price, 8),
                          'exit_time':  datetime.now(timezone.utc).isoformat(),
                          'pnl_pct':    round(pnl_pct, 3),
                          'pnl_usdt':   round(self.POSITION_USDT * pnl_pct / 100, 2),
                          'outcome':    outcome,
                          'exit_reason': 'TP' if hit_tp else 'SL'}
                self._history.append(closed)
                closed_keys.append(key)
                print(f'[ScalpBot] {outcome} {direction} {sym} {pnl_pct:+.2f}% ({closed["exit_reason"]})')
        for k in closed_keys:
            del self._open[k]
        if closed_keys:
            self._save_record()

    def scan(self, live_prices: Dict[str, float]) -> None:
        """Run one scalp scan cycle. Call from async loop via run_in_executor."""
        now = time.time()
        if now - self._last_scan < self.SCAN_INTERVAL:
            return
        self._last_scan = now

        self._check_exits(live_prices)

        if not self._load_engine():
            return
        engine = self._engine
        if engine is None:
            return

        try:
            signals = engine.scan_all_tokens(
                modes=['scalping', 'scalping_15m'],
                risk_profile='aggressive',
                force_fire=True,
            )
        except Exception as e:
            print(f'[ScalpBot] scan error: {e}')
            return

        for sig in signals:
            sym       = sig['symbol']
            mode      = sig['mode']
            direction = sig['direction']
            price     = float(sig.get('current_price', 0))
            if price <= 0:
                continue

            key = f'{sym}_{mode}'
            if key in self._open:
                continue  # already in a position for this symbol+mode

            # Use directional probability from binary model when available;
            # fall back to combined confidence from 3-class model.
            dir_conf = (
                sig.get('p_buy',  sig.get('confidence', 0)) if direction == 'BUY'
                else sig.get('p_sell', sig.get('confidence', 0))
            )

            atr_est = price * 0.008  # ~0.8% ATR estimate for scalping
            sl = round(price - atr_est, 8) if direction == 'BUY' else round(price + atr_est, 8)
            tp = round(price + atr_est * 1.5, 8) if direction == 'BUY' else round(price - atr_est * 1.5, 8)

            entry = {
                'signal_id':   sig.get('signal_id', str(uuid.uuid4())[:8]),
                'symbol':      sym,
                'mode':        mode,
                'timeframe':   sig.get('timeframe', '5m'),
                'direction':   direction,
                'confidence':  round(float(dir_conf), 4),
                'p_buy':       round(float(sig.get('p_buy',  0)), 4),
                'p_sell':      round(float(sig.get('p_sell', 0)), 4),
                'p_hold':      round(float(sig.get('p_hold', 0)), 4),
                'entry_price': round(price, 8),
                'sl':          sl,
                'tp':          tp,
                'entry_time':  datetime.now(timezone.utc).isoformat(),
                'position_usdt': self.POSITION_USDT,
                'outcome':     'OPEN',
            }
            self._open[key] = entry
            self._save_record()
            self._push_signal({**entry, 'fire': True})
            print(f'[ScalpBot] FIRE {direction} {sym} @ {price:.6g} | {mode} | conf={entry["confidence"]:.2%}')


# =============================================================================
# Setup helpers
# =============================================================================

def automated_setup(_: Path, args: Any):
    """
    Scan MODEL_STORE for all available symbols (up to 60).
    Binary dual-model pairs (*_model_buy.json + *_model_sell.json) are auto-tradeable.
    Legacy single-model symbols use the tradeable flag from *_meta.json.
    """
    tradeable_configs:     List[TokenConfig] = []
    non_tradeable_configs: List[TokenConfig] = []

    if MODEL_STORE.exists():
        # Detect binary dual-model pairs (new training pipeline) — auto-tradeable
        binary_syms: set = set()
        for buy_file in MODEL_STORE.glob('*_model_buy.json'):
            base = buy_file.name.replace('_model_buy.json', '')
            if (MODEL_STORE / f'{base}_model_sell.json').exists():
                sym = base.replace('_', '/', 1)
                binary_syms.add(sym)

        seen: set = set()
        # Add binary pairs first (highest priority — directly tradeable)
        for sym in sorted(binary_syms):
            seen.add(sym)
            tradeable_configs.append(TokenConfig(symbol=sym))

        # Scan meta.json for any remaining legacy symbols
        meta_files = sorted(MODEL_STORE.glob('*_meta.json'),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        for meta_file in meta_files:
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                sym = meta.get('symbol', '')
                if not sym or sym in seen:
                    continue
                seen.add(sym)
                tc = TokenConfig(symbol=sym)
                if meta.get('tradeable', False):
                    tradeable_configs.append(tc)
                else:
                    non_tradeable_configs.append(tc)
            except Exception:
                pass

    TARGET    = 60
    configs   = tradeable_configs[:TARGET]
    remaining = TARGET - len(configs)
    if remaining > 0:
        configs += non_tradeable_configs[:remaining]

    if not configs:
        print('[automated_setup] No models found — falling back to BTC/USDT.')
        configs = [TokenConfig(symbol='BTC/USDT')]

    capital      = float(getattr(args, 'capital',      10_000.0))
    max_pos      = float(getattr(args, 'max_position',  1_000.0))
    scan_seconds = int(getattr(args,   'scan_seconds',    300))
    proxy        = getattr(args, 'proxy', None)

    t = len(tradeable_configs[:TARGET])
    m = len(configs) - t
    print(f'[automated_setup] {len(configs)} symbols ({t} tradeable + {m} monitor-only) | '
          f'capital={capital} | max_pos={max_pos} | scan={scan_seconds}s')
    return configs, capital, max_pos, scan_seconds, proxy


# =============================================================================
# CLI entry point  —  rich terminal dashboard
# =============================================================================

def _build_terminal_dashboard(engine: 'LiveEngine') -> None:
    """
    Live-updating rich terminal dashboard (shown when run directly).

    Layout  (updates every 2 s)
    ──────────────────────────────────────────────────────────────────
    [HEADER]   Capital · Balance · Total PnL · Win-rate · Warmup bar
    [TOKEN GRID]  All symbols — price, signal, bias, RSI, regime,
                  confidence, quality score, open-position P&L
    [OPEN TRADES] Active virtual positions (entry, live PnL, SL)
    [CLOSED (5)]  Last 5 closed trades with outcome badge
    """
    from rich.console import Group
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.live import Live
    from rich import box

    def _px(p: float) -> str:
        if p <= 0:      return '—'
        if p < 0.001:   return f'{p:.6f}'
        if p < 1:       return f'{p:.4f}'
        if p < 100:     return f'{p:.3f}'
        return f'{p:.2f}'

    def _pc(v: float) -> str:
        return 'green' if v > 0 else ('red' if v < 0 else 'dim white')

    def _dir(d: str) -> tuple:
        if d == 'LONG':  return '▲', 'bold green'
        if d == 'SHORT': return '▼', 'bold red'
        return '–', 'dim white'

    def _badge(outcome: str) -> Text:
        if outcome == 'WIN':  return Text(' WIN ',  style='bold black on green')
        if outcome == 'LOSS': return Text(' LOSS ', style='bold white on red')
        if outcome == 'OPEN': return Text(' OPEN ', style='bold black on cyan')
        return Text(outcome, style='dim')

    def _signal_cell(sig: dict) -> str:
        side     = sig.get('signal', 'FLAT')
        fire     = sig.get('fire', False)
        strength = sig.get('signal_strength', '')
        if not fire or side in ('FLAT', 'HOLD'):
            if sig.get('pending_entry'):
                _ps = str(sig.get('pending_side', '') or '')
                return f'[yellow]⏳ ARMED {_ps}[/]'.rstrip()
            return '[dim]·[/]'
        _p = ' [dim](paper)[/]' if sig.get('paper_only') else ''
        if 'STRONG' in strength:
            return ('[bold green]🔥 STRONG BUY[/]' if side == 'BUY'
                    else '[bold red]🔥 STRONG SELL[/]') + _p
        return ('[green]BUY[/]' if side == 'BUY' else '[red]SELL[/]') + _p

    def _regime_short(r: str) -> str:
        r = r or 'UNKNOWN'
        _MAP = {
            _REGIME_TRENDING_BULL:      '[green]↑BULL[/]',
            _REGIME_TRENDING_BEAR:      '[red]↓BEAR[/]',
            _REGIME_RANGING:            '[dim]RANGE[/]',
            _REGIME_ACCUMULATION:       '[cyan]ACCUM[/]',
            _REGIME_DISTRIBUTION:       '[yellow]DIST[/]',
            _REGIME_VOLATILE_EXPANSION: '[bold red]EXPND[/]',
            _REGIME_VOLATILE_COMPRESS:  '[dim]CMPR[/]',
            _REGIME_LIQUIDITY_TRAP:     '[bold red]TRAP[/]',
        }
        return _MAP.get(r, f'[dim]{r[:5]}[/]')

    def _quality_cell(q: float) -> str:
        if q >= 75:   return f'[bold green]{q:.0f}[/]'
        if q >= 55:   return f'[yellow]{q:.0f}[/]'
        if q > 0:     return f'[dim red]{q:.0f}[/]'
        return '[dim]—[/]'

    def _build_renderable() -> Group:
        """Build the full dashboard as a Group (header + grid + footer) for Live."""
        wallet  = engine.wallet
        live_px = engine.live_prices
        signals = engine.last_signals

        pnl_u   = round(wallet.balance - wallet.initial_capital, 2)
        pnl_pct = round(pnl_u / wallet.initial_capital * 100, 2)
        pc      = 'green' if pnl_u >= 0 else 'red'
        s       = wallet.summary
        wr      = round(s['win_rate'] * 100, 1) if s['total_trades'] else 0.0
        warmup  = engine.bootstrap_done >= engine.bootstrap_total
        if warmup:
            status = '[bold green]● LIVE[/]'
        else:
            _ready = len(engine.last_signals)
            status = f'[yellow]⏳ WARMING UP  {_ready}/{engine.bootstrap_total} tokens ready[/]'
        now_utc = datetime.now(timezone.utc).strftime('%Y-%m-%d  %H:%M:%S UTC')
        perf    = engine.perf_tracker.get_performance_summary()
        safe_tag = '[bold red]  ⚠ SAFE-MODE[/]' if perf['safe_mode'] else ''

        header = Panel(
            f"  {status}{safe_tag}   │   "
            f"Capital [bold cyan]${wallet.initial_capital:,.0f}[/]   │   "
            f"Balance [bold cyan]${wallet.balance:,.2f}[/]   │   "
            f"PnL [{pc}]{pnl_u:+.2f} USDT  ({pnl_pct:+.2f}%)[/]   │   "
            f"Closed [white]{s['total_trades']}[/] "
            f"([green]{s['won']}W[/]/[red]{s['lost']}L[/])   │   "
            f"WR [bold]{wr:.1f}%[/]   │   "
            f"Open [bold cyan]{len(wallet.open_positions)}[/]   │   "
            f"[dim]{now_utc}[/]",
            title="[bold]  AEGIS-1  Signal Engine  [/]",
            border_style='cyan',
        )

        tradeable_syms = {
            sym for sym, pred in engine.predictors.items()
            if getattr(pred, 'meta', {}).get('tradeable', False)
        }

        _all_syms = sorted(engine.predictors.keys()) if engine.predictors else sorted(signals.keys())
        # Show ALL tokens; sort: open positions first, then active signals, then HOLD by alpha
        def _row_priority(s: str) -> int:
            if s in wallet.open_positions:
                return 0
            sig_s = signals.get(s, {})
            if sig_s.get('fire') and sig_s.get('signal', 'HOLD') not in ('HOLD', 'FLAT', ''):
                return 1
            return 2
        all_syms = sorted(_all_syms, key=lambda s: (_row_priority(s), s))

        _n_firing = sum(1 for s in _all_syms
                        if signals.get(s, {}).get('fire')
                        and signals.get(s, {}).get('signal', 'HOLD') not in ('HOLD', 'FLAT', ''))
        _n_open   = len(wallet.open_positions)

        grid = Table(
            box=box.SIMPLE_HEAVY,
            border_style='bright_black',
            show_header=True,
            header_style='bold white',
            expand=True,
            title=f'[bold dim]ALL TOKENS — {_n_firing} firing · {_n_open} open · {len(_all_syms)} monitored · refreshes every 1s[/]',
        )
        grid.add_column('#',        justify='right',  width=3,  style='dim')
        grid.add_column('Symbol',   justify='left',   width=14)
        grid.add_column('Price',    justify='right',  width=11)
        grid.add_column('Signal',   justify='center', width=13)
        grid.add_column('Dir%',     justify='right',  width=6)
        grid.add_column('Quality',  justify='right',  width=7)
        grid.add_column('RSI',      justify='right',  width=5)
        grid.add_column('Regime',   justify='center', width=7)
        grid.add_column('SL',       justify='right',  width=11)
        grid.add_column('TP',       justify='right',  width=11)
        grid.add_column('Position', justify='center', width=16)
        for idx, sym in enumerate(all_syms, 1):
            sig          = signals.get(sym, {})
            price        = float(live_px.get(sym, sig.get('price', 0) or 0))
            rsi          = sig.get('rsi', None)
            regime       = sig.get('regime', 'UNKNOWN')
            quality      = float(sig.get('quality_score', 0))
            is_tradeable = sym in tradeable_syms
            _pred_meta   = engine.predictors[sym].meta if sym in engine.predictors else {}
            dir_prec     = float(_pred_meta.get('holdout_trading', {}).get('directional_precision', 0))

            pos = wallet.open_positions.get(sym)

            # ── SL / TP cells ──────────────────────────────────────────────
            if pos:
                sl_val = pos.stop_loss
                tp_val = pos.take_profit_1
                sl_cell = f'[red]{_px(sl_val)}[/]'
                tp_cell = f'[green]{_px(tp_val)}[/]'
            elif sig.get('fire') and sig.get('suggested_sl'):
                sl_val = float(sig.get('suggested_sl') or 0)
                tp_val = float(sig.get('suggested_tp') or 0)
                sl_cell = f'[dim red]{_px(sl_val)}[/]'
                tp_cell = f'[dim green]{_px(tp_val)}[/]'
            else:
                sl_cell = '[dim]—[/]'
                tp_cell = '[dim]—[/]'

            # ── Position P&L cell ──────────────────────────────────────────
            if pos:
                cur = price or pos.entry_price
                ppct = ((cur - pos.entry_price) / pos.entry_price * 100) if pos.direction == 'LONG' \
                       else ((pos.entry_price - cur) / pos.entry_price * 100)
                arrow, ds = _dir(pos.direction)
                trail_flag = '[bold yellow]T[/] ' if engine._tp1_hit.get(sym) else ''
                pos_cell   = f'{trail_flag}[{ds}]{arrow}[/] [{_pc(ppct)}]{ppct:+.2f}%[/]'
            else:
                pos_cell = '[dim]—[/]'

            # ── RSI cell ───────────────────────────────────────────────────
            if rsi is None:
                rsi_cell = '[dim]—[/]'
            else:
                rsi_f = float(rsi)
                if rsi_f >= 70:   rsi_cell = f'[red]{rsi_f:.0f}[/]'
                elif rsi_f <= 30: rsi_cell = f'[green]{rsi_f:.0f}[/]'
                else:             rsi_cell = f'{rsi_f:.0f}'

            # ── Dir% cell ──────────────────────────────────────────────────
            if dir_prec > 0:
                dp_pct   = dir_prec * 100
                dp_col   = 'green' if dp_pct >= 70 else ('yellow' if dp_pct >= 60 else 'red')
                dir_cell = f'[{dp_col}]{dp_pct:.0f}%[/]' if is_tradeable else f'[dim]{dp_pct:.0f}%[/]'
            else:
                dir_cell = '[dim]—[/]'

            sym_cell = f'[bold]{sym}[/]' if is_tradeable else f'[dim]{sym}[/]'
            px_cell  = f'[bold white]{_px(price)}[/]' if is_tradeable else f'[dim]{_px(price)}[/]'

            # ── Signal cell: show conflict when model opposes open position ──
            _raw_side = sig.get('side', 'FLAT')
            _sig_fire  = bool(sig.get('fire'))
            if is_tradeable and pos and _sig_fire and (
                (pos.direction == 'LONG'  and _raw_side == 'SELL') or
                (pos.direction == 'SHORT' and _raw_side == 'BUY')
            ):
                # Model flipped against the open position — warn the user.
                # The position SL/TP are correct for the ORIGINAL direction;
                # "BUY" here means the model now wants the opposite.
                _flip_label = 'BUY↑' if _raw_side == 'BUY' else 'SELL↓'
                sig_cell = f'[bold yellow]⚡ {_flip_label}[/]'
            elif is_tradeable:
                sig_cell = _signal_cell(sig)
            else:
                sig_cell = '[dim]watch[/]'

            grid.add_row(
                f'[dim]{idx}[/]' if not is_tradeable else str(idx),
                sym_cell,
                px_cell,
                sig_cell,
                dir_cell,
                _quality_cell(quality) if is_tradeable else '[dim]—[/]',
                rsi_cell,
                _regime_short(regime),
                sl_cell,
                tp_cell,
                pos_cell,
            )

        # ── Footer: open positions detail + last 5 closed ─────────────────
        open_lines: List[str] = []
        for pos in sorted(wallet.open_positions.values(), key=lambda p: p.entry_time):
            cur  = float(live_px.get(pos.symbol, pos.entry_price) or pos.entry_price)
            ppct = ((cur - pos.entry_price) / pos.entry_price * 100) if pos.direction == 'LONG' \
                   else ((pos.entry_price - cur) / pos.entry_price * 100)
            pu   = round(pos.position_value * ppct / 100, 2)
            arr, ds = _dir(pos.direction)
            col      = _pc(ppct)
            trail_tag = ' [bold yellow][T][/]' if engine._tp1_hit.get(pos.symbol) else ''
            open_lines.append(
                f"  [{ds}]{arr} {pos.symbol}[/]  "
                f"entry [white]{_px(pos.entry_price)}[/] → now [bold white]{_px(cur)}[/]  "
                f"[{col}]{ppct:+.2f}%  {pu:+.2f} USDT[/]  "
                f"SL [red]{_px(pos.stop_loss)}[/]  TP [green]{_px(pos.take_profit_1)}[/]  "
                f"edge [cyan]{pos.meta_confidence:.3f}[/]{trail_tag}  "
                f"[dim]{pos.entry_time[11:16]} UTC[/]"
            )

        closed_lines: List[str] = []
        for rec in sorted(wallet.trade_history,
                          key=lambda t: t.close_time or '', reverse=True)[:5]:
            arr, ds = _dir(rec.direction)
            col = _pc(rec.pnl_pct)
            closed_lines.append(
                f"  {_badge(rec.outcome)}  [{ds}]{arr} {rec.symbol}[/]  "
                f"[{col}]{rec.pnl_pct:+.2f}%  {rec.pnl_usdt:+.2f} USDT[/]  "
                f"[dim]{(rec.exit_reason or '').replace('_',' ')}[/]"
            )

        open_body   = "\n".join(open_lines)   if open_lines   else "  [dim]No open positions[/]"
        closed_body = "\n".join(closed_lines) if closed_lines else "  [dim]No closed trades yet[/]"

        footer = Panel(
            f"[bold cyan]● OPEN[/]  ({len(wallet.open_positions)})\n{open_body}\n\n"
            f"[bold white]✔ RECENT CLOSED[/]  ({len(wallet.trade_history)} total)\n{closed_body}",
            border_style='dim',
        )

        return Group(header, grid, footer)

    async def _run_with_display() -> None:
        scalp_bot = ScalpBot()
        scan_task  = asyncio.create_task(engine.run())

        async def _scalp_loop() -> None:
            executor = __import__('concurrent.futures', fromlist=['ThreadPoolExecutor']).ThreadPoolExecutor(max_workers=1)
            loop = asyncio.get_event_loop()
            while True:
                try:
                    await loop.run_in_executor(executor, scalp_bot.scan, engine.live_prices.copy())
                except Exception:
                    pass
                await asyncio.sleep(10)

        scalp_task = asyncio.create_task(_scalp_loop())

        with Live(_build_renderable(), screen=True, refresh_per_second=1) as live:
            try:
                while not scan_task.done():
                    try:
                        live.update(_build_renderable())
                    except Exception:
                        pass
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                pass
            finally:
                scalp_task.cancel()
                scan_task.cancel()
                await engine.shutdown()

    asyncio.run(_run_with_display())


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Aegis-1 Live Signal Engine')
    parser.add_argument('--capital',      type=float, default=10_000.0)
    parser.add_argument('--max-position', type=float, default=1_000.0, dest='max_position')
    parser.add_argument('--scan-seconds', type=int,   default=300,     dest='scan_seconds')
    parser.add_argument('--proxy',        type=str,   default=None)
    parser.add_argument('--no-ui',        action='store_true',
                        help='Disable rich terminal UI (plain log output)')
    cli_args = parser.parse_args()

    _configs, _cap, _maxp, _scan, _proxy = automated_setup(Path('.'), cli_args)
    _engine = LiveEngine(
        token_configs         = _configs,
        capital               = _cap,
        max_position_usdt     = _maxp,
        scan_interval_seconds = _scan,
        risk_tier             = 'balanced',
        proxy_url             = _proxy,
    )

    if cli_args.no_ui:
        asyncio.run(_engine.run())
    else:
        _build_terminal_dashboard(_engine)
