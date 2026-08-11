#!/usr/bin/env python3
"""live_engine.py — Aegis-1 Live Signal Engine (Glass-Box Adaptive)

The engine now lives in `scripts/engine/`. This module is the stable import
surface: `main.py` and ~27 other modules import from `scripts.live_engine`, and
everything they used to get is still here, bound to the same names.

Where things went
-----------------
    engine/contract.py    typed sidecar — the retrain <-> live handshake
    engine/config.py      paths, flags, regime labels, hard vetoes
    engine/indicators.py  closed-bar, non-repainting TA helpers
    engine/market_data.py OHLCV / price / spread fetchers
    engine/models.py      TokenConfig, Position, TradeRecord, RegimeState
    engine/state.py       durable track record (local file + Firestore mirror)
    engine/regime.py      MarketRegimeDetector
    engine/quality.py     SignalQualityFilter
    engine/risk.py        DynamicRiskEngine — sizing, stops, the TP ladder
    engine/tracking.py    PerformanceTracker, DriftMonitor
    engine/portfolio.py   PortfolioGuard, VirtualWallet
    engine/levels.py      LevelsMixin  — swing/HTF S/R, daily bias, BTC tide
    engine/gates.py       GatesMixin   — structure + confirmation gates
    engine/exits.py       ExitsMixin   — TP ladder, break-even, trail, flip
    engine/positions.py   PositionsMixin — entries, signal shape, persistence
    engine/engine.py      LiveEngine   — the scan loop
    engine/scalp.py       ScalpBot
    engine/setup.py       automated_setup
    engine/dashboard.py   terminal UI

How it got here
---------------
This file was 8,198 lines, of which one class was 5,200 and one method 2,023.
Two things made it that size, and only one of them was structure:

  * 1,755 lines were the v80..v82 guard chain, sitting behind
    `if USE_TRADER_GATE:` — a branch that returns before reaching it. It had
    been unreachable in production since v83 shipped, was pinned by a
    characterisation baseline that covered nothing else, and still contained a
    NameError on `_quality_reasons` that could not be observed because
    _scan_all gathers with return_exceptions=True. It was deleted; `git log`
    is the rollback.
  * The rest was one class doing market structure, gating, exits, entries and
    orchestration at once. Those are now mixins composed by LiveEngine, with
    the method bodies unchanged.

Each move was checked by differential test rather than by reading — see
scripts/tests/test_engine_extraction_parity.py, which runs old against new over
randomised inputs and requires identical output.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# ── import bootstrap ─────────────────────────────────────────────────────────
# Run as `python scripts/live_engine.py`, sys.path[0] is scripts/ and the repo
# root is absent, so `import scripts.*` cannot resolve from here. It does not
# fail cleanly: pywin32.pth puts site-packages/win32 on the path, that directory
# holds a `scripts/` folder with no __init__.py, and Python binds the name to
# that namespace package. The error then reads "No module named 'scripts.engine'"
# — the parent resolved, just not to this repo.
#
# engine/config.py does the same insert, but too late to matter: resolution of
# `scripts.engine.config` fails before its body ever runs. The root has to be on
# the path before the first `scripts.*` import below, which means here.
_BOOTSTRAP_ROOT = Path(__file__).resolve().parent.parent
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

# ── configuration, paths and feature flags ───────────────────────────────────
# The underscore-prefixed aliases are the names this module has always exported.
# Callers reach for them (main.py imports TRACK_RECORD_PATH), so they stay bound
# here even though the definitions moved.
#
# READ THIS BEFORE FLIPPING A FLAG AT RUNTIME. The names below are VALUE copies
# taken at import time. Rebinding `live_engine.USE_TRADER_GATE` changes this
# module's name and nothing else — the engine reads the flag through
# `scripts.engine.config`, so the patch is silently ignored. Patch
# `scripts.engine.config` instead; it is the single mutable source of truth.
# (Paths and regime labels are constants, so the copies are harmless.)
from scripts.engine.config import (
    ALPHA_TIMEFRAMES as _ALPHA_TIMEFRAMES,
    ALPHA_TRACK_RECORD_PATH,
    DRIFT_STATE_PATH as _DRIFT_STATE_PATH,
    FS_STATE_COLLECTION as _FS_STATE_COLLECTION,
    FS_STATE_DOC as _FS_STATE_DOC,
    HARD_VETOES as _HARD_VETOES,
    MODEL_STORE,
    PERF_STATE_PATH as _PERF_STATE_PATH,
    REGIME_ACCUMULATION as _REGIME_ACCUMULATION,
    REGIME_DISTRIBUTION as _REGIME_DISTRIBUTION,
    REGIME_LIQUIDITY_TRAP as _REGIME_LIQUIDITY_TRAP,
    REGIME_RANGING as _REGIME_RANGING,
    REGIME_TRENDING_BEAR as _REGIME_TRENDING_BEAR,
    REGIME_TRENDING_BULL as _REGIME_TRENDING_BULL,
    REGIME_VOLATILE_COMPRESS as _REGIME_VOLATILE_COMPRESS,
    REGIME_VOLATILE_EXPANSION as _REGIME_VOLATILE_EXPANSION,
    ROOT as _ROOT,
    SCALP_RECORD_PATH as _SCALP_RECORD_PATH,
    STATE_DIR as _STATE_DIR,
    STATE_GENERATION as _STATE_GENERATION,
    TRACK_RECORD_PATH,
    USE_TRADER_GATE,
    USE_WEIGHTED_SCORER,
)

# ── durable state ────────────────────────────────────────────────────────────
from scripts.engine.state import (
    _fs_clear_track_record,
    _fs_load_track_record,
    _fs_save_track_record,
    _fs_state_client,
    _hydrate_track_record_from_firestore,
)

# ── market data ──────────────────────────────────────────────────────────────
from scripts.engine.market_data import (
    _fetch_bids_asks_all,
    _fetch_ohlcv_sync,
    _fetch_spot_price,
    _has_usdm_perp,
    _usdm_perp_symbol,
)

# ── indicators ───────────────────────────────────────────────────────────────
from scripts.engine.indicators import (
    _closes,
    _confirmed_pivots,
    _detect_bos_choch,
    _detect_divergence,
    _detect_volume_events,
    _ema_last,
    _macd_line,
    _range_pos,
    _reversal_candle,
    _rsi_series,
)

# ── domain types and components ──────────────────────────────────────────────
from scripts.engine.models import Position, RegimeState, TokenConfig, TradeRecord
from scripts.engine.regime import MarketRegimeDetector
from scripts.engine.quality import SignalQualityFilter
from scripts.engine.risk import DynamicRiskEngine
from scripts.engine.tracking import DriftMonitor, PerformanceTracker, _OutcomeRecord
from scripts.engine.portfolio import PortfolioGuard, VirtualWallet

# ── the engine itself ────────────────────────────────────────────────────────
from scripts.engine.levels import LevelsMixin
from scripts.engine.gates import GatesMixin
from scripts.engine.exits import ExitsMixin
from scripts.engine.positions import PositionsMixin
from scripts.engine.engine import LiveEngine
from scripts.engine.scalp import ScalpBot
from scripts.engine.setup import automated_setup
from scripts.engine.dashboard import _build_terminal_dashboard

__all__ = [
    # engine + entry points
    "LiveEngine", "ScalpBot", "automated_setup", "_build_terminal_dashboard",
    # mixins (exported so a caller can see what LiveEngine is made of)
    "LevelsMixin", "GatesMixin", "ExitsMixin", "PositionsMixin",
    # domain types
    "TokenConfig", "Position", "TradeRecord", "RegimeState",
    # components
    "MarketRegimeDetector", "SignalQualityFilter", "DynamicRiskEngine",
    "PerformanceTracker", "DriftMonitor", "PortfolioGuard", "VirtualWallet",
    # paths + flags
    "TRACK_RECORD_PATH", "ALPHA_TRACK_RECORD_PATH", "MODEL_STORE",
    "USE_TRADER_GATE", "USE_WEIGHTED_SCORER", "_HARD_VETOES",
    # regime labels (tests import these directly)
    "_REGIME_TRENDING_BULL", "_REGIME_TRENDING_BEAR", "_REGIME_RANGING",
    "_REGIME_ACCUMULATION", "_REGIME_DISTRIBUTION", "_REGIME_LIQUIDITY_TRAP",
    "_REGIME_VOLATILE_EXPANSION", "_REGIME_VOLATILE_COMPRESS",
    # durable state
    "_fs_clear_track_record", "_fs_save_track_record", "_fs_load_track_record",
    "_hydrate_track_record_from_firestore",
    # indicators (retrain_model.py mirrors _reversal_candle; tests import these)
    "_reversal_candle", "_detect_bos_choch", "_detect_divergence",
    "_detect_volume_events", "_range_pos", "_rsi_series", "_ema_last",
    "_macd_line", "_closes", "_confirmed_pivots",
    # market data
    "_fetch_ohlcv_sync", "_fetch_spot_price", "_fetch_bids_asks_all",
    "_has_usdm_perp", "_usdm_perp_symbol",
]


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
