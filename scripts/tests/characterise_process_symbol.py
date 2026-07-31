"""Characterisation harness for LiveEngine._process_symbol.

_process_symbol is 2,059 lines — 27 % of live_engine.py, 211 `if`s, 43 `return`s,
14 levels of nesting. It is being extracted into phases, and this module is the
safety net: it drives the function over a matrix of predictor outputs and
snapshots everything observable, so any behavioural drift during the refactor
shows up as a diff rather than as a silent change in live trading.

This is deliberately NOT a unit test of the gates. Collaborators that touch the
network or heavy feature builds (_fetch_candles, _daily_bias, _structure_gate,
the S/R helpers) are pinned at the boundary with fixed returns, so a run is
deterministic and depends only on _process_symbol's own decision logic.

Usage
-----
    python -m scripts.tests.characterise_process_symbol --write   # record baseline
    python -m scripts.tests.characterise_process_symbol           # compare
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import sys
from contextlib import redirect_stdout
from collections import deque
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.live_engine import (            # noqa: E402
    DriftMonitor, DynamicRiskEngine, LiveEngine, MarketRegimeDetector,
    PerformanceTracker, PortfolioGuard, SignalQualityFilter, VirtualWallet,
)

SNAPSHOT_PATH = Path(__file__).parent / 'process_symbol_baseline.json'
SYMBOL = 'TEST/USDT'


# ── boundary stubs ───────────────────────────────────────────────────────────

class _StubPredictor:
    """Stands in for src.ml.predictor.Predictor."""

    def __init__(self, result: Dict[str, Any]):
        self._result = result
        self.meta: Dict[str, Any] = {'tradeable': True}

    def predict_realtime(self, risk_tier: str = 'balanced', timeframe: str = '1h'):
        return dict(self._result)


class _Recorder:
    """Absorbs orchestrator / notifier calls without side effects.

    evaluate_signal must echo its argument back: the engine assigns the return
    value straight onto last_signals[symbol], so a None here silently blanks the
    published signal (and the real orchestrator returns the dict).
    """

    _ECHO = {'evaluate_signal'}

    def __init__(self) -> None:
        self.calls: List[str] = []

    def __getattr__(self, name):
        def _f(*a, **k):
            self.calls.append(name)
            return a[0] if (name in self._ECHO and a) else None
        return _f


def _sync_executor():
    """Run 'executor' work inline so a snapshot is single-threaded + ordered."""
    class _E:
        def submit(self, fn, *a, **k):
            fut: asyncio.Future = asyncio.Future()
            try:
                fut.set_result(fn(*a, **k))
            except Exception as e:      # pragma: no cover - surfaced in snapshot
                fut.set_exception(e)
            return fut
    return _E()


def _candles(n: int, bullish: bool, base: float = 100.0) -> List[list]:
    """OHLCV series ending in a confirmed reversal pattern.

    The engine refuses to enter without one (`_reversal_candle`), so a flat or
    empty series parks every case at "5m confirmation unavailable" and the
    snapshot never reaches the entry path. Bullish tail = red candle then a
    hammer; bearish tail = green candle then a shooting star.
    """
    out: List[list] = []
    ts = 1_700_000_000_000
    for i in range(max(n, 5)):
        p = base + (i % 3) * 0.05
        out.append([ts + i * 60_000, p, p + 0.08, p - 0.08, p + 0.02, 1_000.0])
    if bullish:
        # penultimate: red;  last: hammer (long lower wick, small body)
        out[-2] = [ts, base + 0.30, base + 0.32, base - 0.05, base - 0.02, 1500.0]
        out[-1] = [ts, base - 0.01, base + 0.02, base - 0.60, base + 0.00, 2200.0]
    else:
        # penultimate: green; last: shooting star (long upper wick, small body)
        out[-2] = [ts, base - 0.30, base + 0.05, base - 0.32, base + 0.02, 1500.0]
        out[-1] = [ts, base + 0.01, base + 0.60, base - 0.02, base + 0.00, 2200.0]
    return out


def _build_engine(tmp: Path, macro: tuple = (0.0, 0.0),
                  bullish_reversal: bool = True) -> LiveEngine:
    """A LiveEngine with real pure collaborators and pinned I/O boundaries."""
    eng = LiveEngine.__new__(LiveEngine)

    eng.wallet = VirtualWallet(10_000.0, 1_000.0, track_record_path=tmp / 'tr.json')
    eng.alpha_wallet = VirtualWallet(10_000.0, 1_000.0,
                                     track_record_path=tmp / 'alpha.json')
    eng.risk_engine = DynamicRiskEngine()
    eng.quality_filter = SignalQualityFilter()
    eng.regime_detector = MarketRegimeDetector()
    eng.portfolio_guard = PortfolioGuard()
    eng.perf_tracker = PerformanceTracker()
    eng.drift_monitor = DriftMonitor()
    eng.adaptive_orchestrator = _Recorder()

    eng.risk_tier = 'balanced'
    eng.last_signals = {}
    eng.live_prices = {SYMBOL: 100.0}
    eng.alpha_signals = {}
    eng.alpha_mode = False
    eng._executor = None                       # patched per-run below
    eng._news_lock = (False, '')       # (locked?, label) — see LiveEngine.__init__
    eng._signal_history = {}
    eng._spreads = {}
    eng._armed_pending_setups = {}
    eng._last_close_time = {}
    eng._last_close_side = {}
    eng._last_close_reason = {}
    eng._last_loss_time = {}
    eng._alpha_open_time = {}
    eng._alpha_last_close_time = {}
    eng._open_time = {}
    eng._peak_price = {}
    eng._tp1_hit, eng._tp2_hit, eng._tp3_hit, eng._tp4_hit = {}, {}, {}, {}
    eng.bootstrap_done = 0
    eng.bootstrap_total = 1

    # ── pin the I/O boundary ─────────────────────────────────────────────────
    async def _no_candles(symbol=None, timeframe='1h', limit=200, *a, **k):
        return _candles(limit or 200, bullish_reversal)

    async def _daily_bias(*a, **k):
        return macro

    async def _btc_tide(*a, **k):
        return 'FLAT'

    async def _structure_gate(*a, **k):
        return ('PASS', 'at_level')

    async def _confirmation_gate(*a, **k):
        return {'verdict': 'OK', 'detail': ''}

    async def _levels(*a, **k):
        return []

    async def _htf_sr(*a, **k):
        return None

    async def _trendline(*a, **k):
        return None

    eng._fetch_candles = _no_candles                      # type: ignore[assignment]
    eng._daily_bias = _daily_bias                         # type: ignore[assignment]
    eng._btc_tide = _btc_tide                             # type: ignore[assignment]
    eng._structure_gate = _structure_gate                 # type: ignore[assignment]
    eng._confirmation_gate = _confirmation_gate           # type: ignore[assignment]
    eng._sr_levels = _levels                              # type: ignore[assignment]
    eng._swing_sr = _levels                               # type: ignore[assignment]
    eng._important_levels = _levels                       # type: ignore[assignment]
    eng._structural_levels = _levels                      # type: ignore[assignment]
    eng._htf_sr = _htf_sr                                 # type: ignore[assignment]
    eng._trendline_channel = _trendline                   # type: ignore[assignment]
    eng._save_track_record = lambda: None                 # type: ignore[assignment]
    eng._save_alpha_track_record = lambda: None           # type: ignore[assignment]
    eng._notify_blocked = lambda *a, **k: None            # type: ignore[assignment]
    eng._register_armed_pending_setup = lambda *a, **k: None  # type: ignore[assignment]
    return eng


# ── input matrix ─────────────────────────────────────────────────────────────

def _base_result(**over) -> Dict[str, Any]:
    r: Dict[str, Any] = {
        'symbol': SYMBOL, 'fire': False, 'side': 'FLAT',
        'price': 100.0, 'atr': 1.0, 'atr_pct': 1.0, 'atr_multiplier': 1.5,
        'edge_score': 0.0, 'meta_confidence': 0.0, 'meta_threshold': 55.0,
        'p_buy': 0.2, 'p_sell': 0.2, 'p_hold': 0.6,
        'support': 97.0, 'resistance': 106.0, 'range_position': 0.5,
        'rsi': 50.0, 'rsi_slope': 0.0, 'rsi_acceleration': 0.0,
        'adx': 25.0, 'trend_regime': 'RANGING', 'volatility_regime': 'MEDIUM',
        'market_bias': 'NEUTRAL', 'funding_bias': 'NEUTRAL', 'oi_trend': 'STABLE',
        'volume_zscore': 0.0, 'volume_strength': 'AVERAGE', 'macd_signal': 'NEUTRAL',
        'supertrend': 'NEUTRAL', 'tradeable': True, 'quality_score': 75.0,
    }
    r.update(over)
    return r


def _long(**over) -> Dict[str, Any]:
    """A BUY that clears conviction + indicator support, so it reaches the
    deep gates instead of dying at the first guard."""
    d = dict(fire=True, side='BUY', edge_score=85.0,
             p_buy=0.72, p_hold=0.18, p_sell=0.10,
             market_bias='BULLISH', macd_signal='BULLISH',
             supertrend='BULLISH', rsi_slope=0.8)
    d.update(over)
    return _base_result(**d)


def _short(**over) -> Dict[str, Any]:
    d = dict(fire=True, side='SELL', edge_score=85.0,
             p_sell=0.72, p_hold=0.18, p_buy=0.10,
             market_bias='BEARISH', macd_signal='BEARISH',
             supertrend='BEARISH', rsi_slope=-0.8)
    d.update(over)
    return _base_result(**d)


# (result, macro_daily, macro_weekly) — macro drives the HTF votes/gates
CASES: Dict[str, Any] = {
    'flat_no_fire':         (_base_result(),                                  0.0,  0.0),
    'buy_aligned':          (_long(rsi=45.0),                                 1.0,  1.0),
    'sell_aligned':         (_short(rsi=55.0),                                -1.0, -1.0),
    'buy_at_support':       (_long(range_position=0.05, rsi=28.0),            1.0,  1.0),
    'sell_at_resistance':   (_short(range_position=0.95, rsi=72.0),          -1.0, -1.0),
    # price sitting ON the level (<=0.35% away) — the only geometry this
    # mean-reversion engine actually fires from, so these reach the entry path
    'buy_tagging_support':  (_long(range_position=0.04, rsi=27.0,
                                   support=99.8, resistance=110.0),           1.0,  1.0),
    'sell_tagging_res':     (_short(range_position=0.96, rsi=73.0,
                                    support=90.0, resistance=100.2),         -1.0, -1.0),
    'buy_tagging_htf_conf': (_long(range_position=0.04, rsi=27.0,
                                   support=99.8, resistance=110.0),          -1.0, -1.0),
    'buy_tagging_wide_atr': (_long(range_position=0.04, rsi=27.0, atr=3.0,
                                   atr_pct=3.0, support=99.8,
                                   resistance=115.0),                         1.0,  1.0),
    'buy_htf_conflict':     (_long(rsi=45.0),                                -1.0, -1.0),
    'sell_htf_conflict':    (_short(rsi=55.0),                                1.0,  1.0),
    'buy_low_edge':         (_long(edge_score=30.0),                          1.0,  1.0),
    'buy_low_atr':          (_long(atr_pct=0.2),                              1.0,  1.0),
    'buy_atr_below_soft':   (_long(atr_pct=0.4),                              1.0,  1.0),
    'buy_no_conviction':    (_base_result(fire=True, side='BUY',
                                          edge_score=85.0),                   1.0,  1.0),
    'buy_weak_indicators':  (_base_result(fire=True, side='BUY', edge_score=85.0,
                                          p_buy=0.72, p_hold=0.18, p_sell=0.10),
                                                                              0.0,  0.0),
    'buy_trend_bear':       (_long(adx=45.0, trend_regime='STRONG_DOWN',
                                   rsi=32.0),                                -1.0, -1.0),
    'sell_trend_bull':      (_short(adx=45.0, trend_regime='STRONG_UP',
                                    rsi=68.0),                                1.0,  1.0),
    'buy_liquidity_trap':   (_long(adx=8.0, atr_pct=0.4, volatility_regime='LOW',
                                   volume_strength='BELOW_AVERAGE',
                                   volume_zscore=-1.2),                       1.0,  1.0),
    'buy_vol_expansion':    (_long(atr_pct=6.0, volatility_regime='HIGH'),    1.0,  1.0),
    'sell_low_quality':     (_short(edge_score=72.0, quality_score=40.0,
                                    volume_zscore=-1.5),                     -1.0, -1.0),
    'buy_overbought':       (_long(rsi=82.0, range_position=0.95),            1.0,  1.0),
    'sell_oversold':        (_short(rsi=18.0, range_position=0.05),          -1.0, -1.0),
    'buy_wide_atr':         (_long(atr=3.0, atr_pct=3.0),                     1.0,  1.0),
}


# ── snapshot ─────────────────────────────────────────────────────────────────

def _observable(eng: LiveEngine, out: str) -> Dict[str, Any]:
    """Everything a caller of _process_symbol can see afterwards."""
    sig = eng.last_signals.get(SYMBOL, {})
    if not isinstance(sig, dict):
        sig = {'_repr': repr(sig)}
    keys = (
        'signal', 'signal_strength', 'fire', 'direction', 'paper_only',
        'pending_entry', 'evaluating', 'entry_price', 'suggested_sl',
        'suggested_tp', 'tp2', 'tp3', 'tp4', 'tp5', 'risk_reward',
        'quality_score', 'regime', 'rr_blocked', 'regime_blocked',
        'levels_frozen', 'risk_tier', 'entry_mode',
    )
    pos = eng.wallet.open_positions.get(SYMBOL)
    apos = eng.alpha_wallet.open_positions.get(f'{SYMBOL}|risky')

    def _p(p):
        if p is None:
            return None
        return {'side': p.side, 'entry': p.entry_price, 'sl': p.stop_loss,
                'tp1': p.take_profit_1, 'tp5': p.take_profit_5,
                'value': p.position_value, 'tier': p.signal_strength}

    # decision lines only — drop anything with a timestamp/uuid in it
    lines = [l for l in out.splitlines() if l.startswith(f'[{SYMBOL}]')]
    return {
        'signal':     {k: sig.get(k) for k in keys if k in sig},
        'position':   _p(pos),
        'paper':      _p(apos),
        'decisions':  lines,
        'bootstrap':  eng.bootstrap_done,
    }


async def _run_case(name: str, spec: Any, tmp: Path) -> Dict[str, Any]:
    result, _md, _mw = spec
    eng = _build_engine(tmp, macro=(_md, _mw),
                        bullish_reversal=(result.get('side') != 'SELL'))
    pred = _StubPredictor(result)
    sem = asyncio.Semaphore(1)

    loop = asyncio.get_running_loop()
    real_rie = loop.run_in_executor

    def _inline(executor, fn, *args):
        fut = loop.create_future()
        try:
            fut.set_result(fn(*args))
        except Exception as e:
            fut.set_exception(e)
        return fut

    loop.run_in_executor = _inline          # type: ignore[assignment]
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            await eng._process_symbol(SYMBOL, pred, sem)
    except Exception as e:
        loop.run_in_executor = real_rie     # type: ignore[assignment]
        return {'ERROR': f'{type(e).__name__}: {e}'}
    loop.run_in_executor = real_rie         # type: ignore[assignment]
    return _observable(eng, buf.getvalue())


async def capture(tmp: Path) -> Dict[str, Any]:
    snap: Dict[str, Any] = {}
    for name, res in CASES.items():
        snap[name] = await _run_case(name, res, tmp)
    return snap


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--write', action='store_true',
                    help='record the current behaviour as the baseline')
    args = ap.parse_args()

    import tempfile
    tmp = Path(tempfile.mkdtemp())
    snap = asyncio.run(capture(tmp))

    if args.write:
        SNAPSHOT_PATH.write_text(json.dumps(snap, indent=2, sort_keys=True,
                                            default=str), encoding='utf-8')
        errs = [k for k, v in snap.items() if 'ERROR' in v]
        print(f'baseline written: {SNAPSHOT_PATH.name} ({len(snap)} cases'
              f'{", " + str(len(errs)) + " raising" if errs else ""})')
        for k in errs:
            print(f'  ! {k}: {snap[k]["ERROR"]}')
        return 0

    if not SNAPSHOT_PATH.exists():
        print('no baseline — run with --write first')
        return 2

    base = json.loads(SNAPSHOT_PATH.read_text(encoding='utf-8'))
    cur = json.loads(json.dumps(snap, sort_keys=True, default=str))
    drift = [k for k in sorted(set(base) | set(cur)) if base.get(k) != cur.get(k)]
    if not drift:
        print(f'OK — {len(cur)} cases identical to baseline')
        return 0
    print(f'BEHAVIOUR DRIFT in {len(drift)} case(s): {", ".join(drift)}')
    for k in drift:
        b, c = base.get(k), cur.get(k)
        for field in sorted(set(b or {}) | set(c or {})):
            if (b or {}).get(field) != (c or {}).get(field):
                print(f'  [{k}] {field}:')
                print(f'      baseline: {(b or {}).get(field)}')
                print(f'      current : {(c or {}).get(field)}')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
