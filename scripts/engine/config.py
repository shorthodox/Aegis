"""Engine-wide constants, paths and feature flags.

Everything here was module-level state in the old single-file live_engine.py.
The comments are kept verbatim where they record *why* a value is what it is --
most of them are the write-up of an incident, and they are the reason the value
does not get casually changed back.

Nothing in this module imports from the rest of the engine, so it is safe to
import from anywhere without a cycle.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ── project root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

__all__ = [
    "ROOT", "MODEL_STORE", "STATE_DIR",
    "TRACK_RECORD_PATH", "ALPHA_TRACK_RECORD_PATH",
    "PERF_STATE_PATH", "DRIFT_STATE_PATH", "SCALP_RECORD_PATH",
    "ALPHA_TIMEFRAMES",
    "USE_TRADER_GATE", "USE_WEIGHTED_SCORER", "HARD_VETOES",
    "FS_STATE_COLLECTION", "FS_STATE_DOC", "STATE_GENERATION",
    "REGIME_TRENDING_BULL", "REGIME_TRENDING_BEAR", "REGIME_RANGING",
    "REGIME_ACCUMULATION", "REGIME_DISTRIBUTION",
    "REGIME_VOLATILE_EXPANSION", "REGIME_VOLATILE_COMPRESS",
    "REGIME_LIQUIDITY_TRAP",
]

ROOT = _ROOT

# ── v83: the desk replaces the Guard A..T veto pile ───────────────────────────
# `src/trading/trader_gate.py` runs one ordered playbook — fitness, setup,
# invalidation, payoff, trigger, allocation — and returns a PLAN rather than a
# boolean.  It supersedes everything from "Guard A: ATR floor" through
# "Guard H: portfolio guard", including the PENDING queue (Guards M + J), which
# becomes a resting order with a hard invalidation and an 8-bar expiry.
#
# What forced the rewrite: on 2026-07-20 the old chain opened eight alt SHORTs
# inside one 55-minute window; every closed one hit its stop, and the model
# confidence on the losers ran from 17.9 to 100.0.  No guard in the chain asked
# whether the trade paid, or whether it was the same bet for the eighth time —
# the two questions a trader asks first.  Sixteen vetoes that each said "no" for
# their own reason, with later patches selectively disabling earlier ones, could
# not be tuned into asking them.
USE_TRADER_GATE = True

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
# v82e: FAR_FROM_SR promoted from "tier downgrade" to HARD veto.
#
# gate_scorer measures, correctly and in ATR, whether price is actually at the
# level the trade leans on, and raises FAR_FROM_SR past AT_LEVEL_ATR (1.0).
# That veto did not block — it only tagged the signal RISKY and fired anyway.
#
# Measured on 8,484 1h bars across 12 tokens: 58.7% sit MORE than 1 ATR from
# the nearest level (median gap 1.17 ATR), and only 15.6% are within 0.35 ATR.
# So the majority of fires were entries taken nowhere near their own structure,
# and the RISKY tag was not a risk rating — it was the engine reporting that
# the setup's premise was absent, then taking the trade regardless. On the
# 2026-08-01 book that was 6 of 7 signals.
#
# NO_VALID_SR is deliberately NOT promoted: it is a weaker, differently-shaped
# condition (srq below floor, dead-centre range position, or cramped RR) and
# still downgrades the tier rather than blocking.
HARD_VETOES = frozenset({'MODEL_DRIFT_CRITICAL', 'DEAD_MARKET',
                         'EXTREME_VOLATILITY', 'FAR_FROM_SR'})

MODEL_STORE = _ROOT / 'src' / 'ml' / 'model_store'

# ── Persistent runtime STATE directory ────────────────────────────────────────
# Runtime state (track record, wallet, drift/perf) is WRITTEN LIVE and must
# survive redeploys.  On an ephemeral platform (Railway/Render/Fly with no
# volume) the container filesystem is wiped on every deploy, which is why the
# track record kept resetting.  Point state at a persistent volume via
# AEGIS_STATE_DIR (e.g. a Railway Volume mounted at /data); falls back to the
# in-repo data/ dir for local dev.  Config artifacts the model LOADS
# (token_params, regime_stats, model_store) stay in the image, unaffected.
STATE_DIR = Path(os.environ.get('AEGIS_STATE_DIR') or (_ROOT / 'data'))
try:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    STATE_DIR = _ROOT / 'data'

TRACK_RECORD_PATH       = STATE_DIR / 'track_record.json'
ALPHA_TRACK_RECORD_PATH = STATE_DIR / 'alpha_track_record.json'
ALPHA_TIMEFRAMES        = ['15m', '30m', '4h', '1d']
PERF_STATE_PATH         = STATE_DIR / 'perf_state.json'
DRIFT_STATE_PATH        = STATE_DIR / 'drift_state.json'
SCALP_RECORD_PATH       = STATE_DIR / 'scalp_trades.json'

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
FS_STATE_COLLECTION = 'engine_state'
FS_STATE_DOC        = 'track_record'
# State generation: bump this to force a ONE-TIME wipe of the durable track
# record on the next deploy. On boot the engine ignores any restored record
# whose generation != this, starting fresh — regardless of what an older engine
# wrote to Firestore in the meantime. gen 2: wipe records produced by the pre-v14
# (sell-into-support / loose-reversal) gates.
STATE_GENERATION = 2

# ── Regime labels ─────────────────────────────────────────────────────────────
REGIME_TRENDING_BULL      = 'TRENDING_BULL'
REGIME_TRENDING_BEAR      = 'TRENDING_BEAR'
REGIME_RANGING            = 'RANGING'
REGIME_ACCUMULATION       = 'ACCUMULATION'
REGIME_DISTRIBUTION       = 'DISTRIBUTION'
REGIME_VOLATILE_EXPANSION = 'VOLATILE_EXPANSION'
REGIME_VOLATILE_COMPRESS  = 'VOLATILE_COMPRESSION'
REGIME_LIQUIDITY_TRAP     = 'LIQUIDITY_TRAP'
