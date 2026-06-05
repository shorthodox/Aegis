#!/usr/bin/env python3
"""
update_oi_cache.py — Append fresh OI data to per-symbol parquet caches.

Binance hard-caps /futures/data/openInterestHist at 30 days. Running this
script daily (or more often) builds an indefinitely-growing local history
that the retrainer can use instead of the 30-day live window.

Usage:
    python scripts/update_oi_cache.py              # all fleet symbols
    python scripts/update_oi_cache.py BTC/USDT     # one symbol
    python scripts/update_oi_cache.py BTC/USDT ETH/USDT SOL/USDT
"""

import sys
import time
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / 'data' / 'oi_cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Keep in sync with FLEET_SYMBOLS in scripts/retrain_model.py
FLEET_SYMBOLS = ["BTC/USDT"]

# 30-day hard limit Binance enforces (we use 29 to avoid edge races)
_BINANCE_OI_MAX_MS = 29 * 24 * 60 * 60 * 1000


_exchange: Optional[Any] = None   # created once, reused for all 200 symbols


def _get_exchange() -> Optional[Any]:
    """Return the shared binanceusdm exchange instance, creating it on first call.

    Creating one object per symbol wastes ~0.5 s of init overhead per call and
    spams the ccxt internals with redundant market-list fetches.  A single
    cached instance is correct because ccxt exchange objects are stateless
    across calls (state is per-request, not per-instance).
    """
    global _exchange
    if _exchange is not None:
        return _exchange
    try:
        import ccxt as _ccxt
    except ImportError:
        print('  [ERROR] ccxt not installed — run: pip install ccxt')
        return None
    try:
        _exchange = _ccxt.binanceusdm({'enableRateLimit': True, 'timeout': 15000})  # type: ignore[arg-type]
    except AttributeError:
        print('  [ERROR] ccxt version has no binanceusdm — run: pip install -U ccxt')
    except Exception as e:
        print(f'  [ERROR] binanceusdm init failed: {e}')
    return _exchange


def _cache_path(symbol: str) -> Path:
    safe = symbol.replace('/', '_').replace(':', '_')
    return CACHE_DIR / f'{safe}_oi.parquet'


def load_cache(symbol: str) -> Optional[pd.DataFrame]:
    path = _cache_path(symbol)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def save_cache(symbol: str, df: pd.DataFrame) -> None:
    df = (df.copy()
            .assign(timestamp=lambda d: pd.to_datetime(d['timestamp']))
            .drop_duplicates('timestamp')
            .sort_values('timestamp')
            .reset_index(drop=True))
    df.to_parquet(_cache_path(symbol), index=False)


def _fetch_binance_oi(symbol: str, since_ms: int) -> Optional[pd.DataFrame]:
    """Pull 1-h OI from Binance starting at since_ms, respecting the 30-day cap.

    Returns None when the symbol has no perpetual contract or all retries fail.
    Retries transient errors (rate-limits, timeouts) with exponential backoff.
    """
    ex = _get_exchange()
    if ex is None:
        return None

    # Derive futures symbol before touching the network — a bad format fails fast
    if '/USDT' not in symbol:
        print(f'  [SKIP] {symbol}: not a /USDT pair, no OI to fetch')
        return None
    futures_sym = symbol.replace('/USDT', '/USDT:USDT')

    # Clamp to Binance's hard 30-day limit
    earliest_allowed = int(time.time() * 1000) - _BINANCE_OI_MAX_MS
    current_since = max(since_ms, earliest_allowed)

    _NO_PERP_PHRASES = (
        'invalid symbol', 'symbol not found', 'no data',
        'does not exist', 'not support',
        'does not have market symbol',  # ccxt message for spot-only tokens
        'market symbol',                # broader match for same class of error
        'no market',
    )
    _MAX_RETRIES = 3

    all_rows: list = []
    while True:
        chunk = None
        last_err = None

        for attempt in range(_MAX_RETRIES):
            try:
                chunk = list(ex.fetch_open_interest_history(
                    futures_sym, '1h', since=current_since, limit=500))
                last_err = None
                break
            except Exception as e:
                last_err = e
                err_lower = str(e).lower()
                # Permanent failure: symbol has no perp contract — stop immediately
                if any(p in err_lower for p in _NO_PERP_PHRASES):
                    print(f'  [SKIP] {symbol}: no perpetual contract on Binance USDT-M')
                    return pd.DataFrame(all_rows) if all_rows else None
                # Transient error (rate-limit, timeout, 5xx) — wait and retry
                time.sleep(2.0 ** attempt)   # 1 s, 2 s, 4 s

        if last_err is not None:
            print(f'  [WARN] {symbol}: OI fetch stopped after {_MAX_RETRIES} retries — {last_err}')
            break

        if not chunk:
            break

        all_rows.extend(chunk)

        # ccxt returns plain dicts at runtime; the type checker sees `object`
        # because the library ships no stubs — suppress and handle defensively.
        try:
            last_ts = int(chunk[-1]['timestamp'] or 0)  # type: ignore[index]
        except (KeyError, TypeError, ValueError):
            break

        if last_ts <= current_since:
            break
        current_since = last_ts + 1

        if len(chunk) < 500:
            break

        time.sleep(0.2)   # polite inter-page pause

    if not all_rows:
        return None

    raw = pd.DataFrame(all_rows)
    oi_col = next((c for c in ('openInterestAmount', 'openInterest') if c in raw.columns), None)
    if not oi_col:
        return None

    return (raw[['timestamp', oi_col]]
            .rename(columns={oi_col: 'open_interest'})
            .assign(timestamp=lambda d: pd.to_datetime(d['timestamp'], unit='ms'),
                    open_interest=lambda d: d['open_interest'].astype(float))
            .drop_duplicates('timestamp')
            .sort_values('timestamp')
            .reset_index(drop=True))


def update_symbol(symbol: str) -> dict:
    existing = load_cache(symbol)

    if existing is not None and not existing.empty:
        last_ts = existing['timestamp'].max()
        since_ms = int(last_ts.timestamp() * 1000) + 1
        print(f'  cache has {len(existing):,} rows up to {last_ts.strftime("%Y-%m-%d %H:%M")} — fetching delta')
    else:
        since_ms = int(time.time() * 1000) - _BINANCE_OI_MAX_MS
        print(f'  no cache yet — fetching last 29 days')

    new_data = _fetch_binance_oi(symbol, since_ms)

    if new_data is None or new_data.empty:
        print(f'  no new data (symbol may lack a perp contract or be too new)')
        total = len(existing) if existing is not None else 0
        return {'symbol': symbol, 'new_rows': 0, 'total_rows': total}

    combined = (pd.concat([existing, new_data], ignore_index=True)
                if existing is not None and not existing.empty
                else new_data)
    combined = (combined
                .drop_duplicates('timestamp')
                .sort_values('timestamp')
                .reset_index(drop=True))
    save_cache(symbol, combined)

    days = len(combined) // 24
    print(f'  +{len(new_data):,} new rows → {len(combined):,} total ({days} days of OI history)')
    return {'symbol': symbol, 'new_rows': len(new_data), 'total_rows': len(combined)}


def main(symbols: Optional[List[str]] = None) -> None:
    if symbols is None:
        symbols = FLEET_SYMBOLS

    print(f'Updating OI cache for {len(symbols)} symbol(s)')
    print(f'Cache dir: {CACHE_DIR}\n')

    results = []
    for i, sym in enumerate(symbols, 1):
        print(f'[{i}/{len(symbols)}] {sym}')
        try:
            r = update_symbol(sym)
        except Exception as e:
            print(f'  ERROR: {e}')
            r = {'symbol': sym, 'new_rows': 0, 'total_rows': 0}
        results.append(r)
        time.sleep(0.4)

    print(f'\n{"─" * 52}')
    ok = sum(1 for r in results if r['total_rows'] > 0)
    added = sum(r['new_rows'] for r in results)
    print(f'Done: {ok}/{len(symbols)} symbols cached, {added:,} new rows added.')


if __name__ == '__main__':
    args = sys.argv[1:]
    if args:
        syms = [a if '/' in a else a.replace('USDT', '/USDT') for a in args]
        main(syms)
    else:
        main()
