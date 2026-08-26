"""Exchange access: prices, candles, book spreads.

All fetches are best-effort and thread-safe — they run inside the scan loop's
thread pool. Exchange instances are module-level singletons behind locks so a
63-token scan does not open 63 connections.

The two rules encoded here, both of which cost real money to learn:

  * A token with no USD-M perp must produce NO candles, never spot candles. The
    spot fallback exists for a transient futures failure on a token that has a
    perp; letting it cover for a token with no perp silently swaps the
    instrument and manufactures a complete signal on a market we do not trade.
  * ccxt's binanceusdm wants SWAP notation ('BASE/USDT:USDT'). Passing the plain
    spot symbol raises BadSymbol and yields nothing, which is what starved the
    structure gate of perp candles for so long.
"""
from __future__ import annotations

import threading
from typing import Dict

__all__ = [
    "_fetch_spot_price", "_usdm_perp_symbol", "_has_usdm_perp",
    "_fetch_ohlcv_sync", "_fetch_bids_asks_all",
]

# Shared exchange for lightweight index-price fetches (reuses the same instance
# as Predictor once the class is loaded to avoid creating a second connection).
_spot_ex = None
_spot_ex_lock = threading.Lock()

# Shared futures exchange for multi-timeframe OHLCV candle fetches.
# Binance USDM perpetuals — same symbols as the main trading fleet.
_usdm_ex      = None
_usdm_ex_lock = threading.Lock()

_perp_markets = None
_perp_markets_lock = threading.Lock()


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


# ── Perp price, for confirming a stop ────────────────────────────────────────
# The live tick feed is Binance SPOT (engine._ws_price_ticker), because the
# FUTURES websocket is not reachable from this host — measured from the Railway
# container, fstream.binance.com delivers 0 messages in 18s while the spot stream
# delivers 753 symbols. That is why the ticker is spot and must stay spot.
#
# But everything else about a trade is PERP: candles, levels and therefore the
# stop all come from ccxt.binanceusdm, and the chart tells the subscriber these
# are "Binance USD-M Futures — the exact market AEGIS trades".
#
# A stop is a THRESHOLD, and near a threshold the basis decides the outcome.
# Measured across 20 tokens: median |perp-spot| 0.064%, p90 0.150%, max 0.196%.
# Against a ~1.30% stop that is 5-15% of the whole distance — enough to fire a
# stop on spot that the perp chart never printed. GMX closed 0.103% through its
# stop on a day its basis was 0.094%.
#
# Futures REST is fine from here (749 rows in 0.09s), so a stop is confirmed
# against the perp before it closes. Stops are rare, so this costs almost
# nothing, and the short TTL collapses a burst across symbols.
_PERP_PX: Dict[str, tuple] = {}
_PERP_PX_TTL = 3.0


def fetch_perp_price(symbol: str) -> float:
    """Last PERP price for 'BASE/USDT', or 0.0 if it cannot be established.

    0.0 means UNKNOWN, never "no". The caller must treat it as "could not
    confirm" and fall back to its existing behaviour — a data failure must never
    be able to hold a position open past its stop.
    """
    import time as _t
    key = symbol.replace('/', '').upper()
    hit = _PERP_PX.get(key)
    now = _t.time()
    if hit and (now - hit[1]) < _PERP_PX_TTL:
        return hit[0]
    try:
        import urllib.request, json as _j
        url = f'https://fapi.binance.com/fapi/v1/ticker/price?symbol={key}'
        with urllib.request.urlopen(url, timeout=6) as r:
            px = float((_j.load(r) or {}).get('price') or 0.0)
        if px > 0:
            _PERP_PX[key] = (px, now)
        return px
    except Exception:
        return 0.0
