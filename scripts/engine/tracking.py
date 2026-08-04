"""Performance tracking and model-drift monitoring.

PerformanceTracker answers "are we losing right now" from live outcomes.
DriftMonitor answers the harder question — "is this model still the model we
trained" — by comparing the live win rate against the precision recorded in the
token's sidecar at training time.

DriftMonitor is therefore the second consumer of the retrain->live contract
(the first being the predictor's thresholds). It reads the sidecar directly
rather than through engine.contract.Sidecar on purpose: it must degrade to a
neutral 0.60 benchmark for a token whose sidecar is old or partial, whereas
Sidecar.load() refuses such a file outright. Keep that difference — a missing
benchmark should not bench a token, it should just make drift unknowable.
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, Optional

from scripts.engine.config import DRIFT_STATE_PATH, MODEL_STORE, PERF_STATE_PATH

__all__ = ["PerformanceTracker", "DriftMonitor", "_OutcomeRecord"]


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
            PERF_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = PERF_STATE_PATH.with_suffix('.tmp')
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(payload, f)
            os.replace(tmp, PERF_STATE_PATH)
        except Exception as e:
            print(f'[PerformanceTracker] save_state failed: {e}')

    def load_state(self) -> None:
        """Restore recent outcome history from disk."""
        if not PERF_STATE_PATH.exists():
            return
        try:
            with open(PERF_STATE_PATH, 'r', encoding='utf-8') as f:
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
            payload = {
                'saved_at': datetime.now(timezone.utc).isoformat(),
                'windows': {
                    sym: [int(b) for b in hist]
                    for sym, hist in self._live_window.items()
                },
            }
            DRIFT_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = DRIFT_STATE_PATH.with_suffix('.tmp')
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(payload, f)
            os.replace(tmp, DRIFT_STATE_PATH)
        except Exception:
            pass

    def load_state(self) -> None:
        if not DRIFT_STATE_PATH.exists():
            return
        try:
            with open(DRIFT_STATE_PATH, 'r', encoding='utf-8') as f:
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
