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

# ── Minimum signal quality to FIRE ────────────────────────────────────────────
# An absolute floor on SignalQualityFilter.score_signal() — the 0-100 number the
# UI prints as "Signal Quality Score". Below it the signal is not taken.
#
# Nothing enforced this on the live entry path. `fire` is decided by
# `edge_score >= thr` (src/ml/predictor.py:990), and engine.py's own comment
# explains why that cannot stand in for quality: edge_score is a PERCENTILE RANK
# of the bar against its own lookback, so in a window where every bar is poor the
# least-poor bar ranks 100. quality_score is the absolute measure — ADX, volume
# conviction, regime confidence, RSI zone, funding and OI alignment, HTF macro,
# candles, MACD — and it was computed, published, and spent only on POSITION SIZE.
#
# A floor did exist and was lost rather than retired. The pre-v83 guard chain ran
# G3_MIN_QUALITY (see scripts/backtest_forensic.py:378 and the 2026-07-01 forensic
# audits, which log "edge=62.5 < MIN_QUALITY_SCORE=70.0"). The v83 TraderGate
# rewrite replaced Guards A..T wholesale and no stage picked the floor back up.
# SignalQualityFilter.MIN_QUALITY_SCORE = 60.0 survived as a constant — described
# in-line as "cut the coin-flip signals (biggest WR lever)" — but the only live
# readers left are an exit check and a pre-gate position check. Nothing on entry.
#
# What that cost, measured on the live fleet 2026-08-16 (44 scored symbols):
#
#     book HELD by        COMP 0, DOGE 18, LINK 35, TAO 38.7, ATOM 56
#     turned away         TRX 100, INJ 76, PENDLE 71, UNI 66, AAVE 61
#
# The five worst signals on the board held every slot while the five best were
# refused — not on merit, but because the book cap is arrival-ordered (see the
# allocation stage in trader_gate.py). Quality was anti-correlated with firing.
#
# Lowered 60 -> 45 on 2026-08-17, by decision, and the measurement supports it.
#
# 60 was inherited from SignalQualityFilter.MIN_QUALITY_SCORE, chosen in v43 by
# moving 55 -> 60. It was calibrated against a fleet whose quality ran median 30
# / p75 40 / max 100, where it retained ~11% — about five symbols, exactly
# MAX_OPEN. That fleet no longer exists. After the reversal-penalty exemptions
# the distribution moved to median 40 / p75 50 / **max 68**, and a floor of 60
# retained 3 of 44 symbols (6.8%) — a book of five that cannot fill. A floor the
# funnel cannot clear is not conservative, it is just off.
#
#     >= 40   23/44  (52.3%)        >= 55    7/44  (15.9%)
#     >= 45   15/44  (34.1%)        >= 60    3/44  ( 6.8%)
#     >= 50   13/44  (29.5%)
#
# 45 retains ~34% — comfortably above MAX_OPEN without admitting the median bar.
#
# Know what this admits. It is a real loosening, not a rounding: the 45-59 band
# is 12 symbols (TRX 58, ETC 56, ADA 55, ENA 55, HBAR 54, DOT/ICP/PENDLE/SOL/VET
# 50, SUI 48, TRB 45). ATOM at 56 — one of the positions whose entry geometry
# started this whole investigation — now clears. So does a thin-evidence fade at
# 57, which the previous floor deliberately refused (see
# test_quality_fade_parity). Those are the trades this number buys.
#
# REVERTED TO 60 ON 2026-08-19. The 45 experiment was measured and it lost money.
#
#     before the drop   n=14   WR 78.6%   avg +0.901%   total +12.62%
#     after the drop    n=26   WR 38.5%   avg -0.342%   total  -8.89%
#     Fisher exact two-sided p = 0.0219 — not a bad run, a real regression.
#
# The book also went 85% short (22 of 26) and most losses closed near the 1.30%
# stop, i.e. a flood of marginal shorts that got stopped out.
#
# The reason given for dropping to 45 does not survive re-measurement. It was
# "60 retains only 3 of 44 tokens, a book of five cannot fill" — but that reading
# was taken minutes after the reversal-penalty exemptions merged, before the new
# scores had propagated across the fleet. Measured again on 2026-08-19 the same
# floor of 60 retains 10 of 44 (22.7%), against MAX_OPEN of 5. The funnel was
# never starved; the measurement was premature.
#
# Only ONE variable moved in this revert. The scorer exemptions stay exactly as
# they are, so if the win rate does not recover, the exemptions are the remaining
# suspect and can be tested on their own. Changing both at once would tell us
# nothing, which is the whole reason the note below existed.
#
# This is a THRESHOLD ON A LIVE FUNNEL, so it is counted, not just applied:
# LOW_QUALITY_REFUSED tallies every fire it blocks and the engine logs each one.
# If the fire rate collapses, that counter is the evidence — lower the floor
# rather than removing it, and never stack it with a new veto in the same change.
MIN_FIRE_QUALITY = 60.0

# ── Kill switch ───────────────────────────────────────────────────────────────
# When True the desk opens nothing new. Existing positions are still MANAGED —
# stops, take-profits and exits all keep running — because pausing entries and
# abandoning open risk are completely different things, and only the first is
# ever what someone means by "pause".
#
# Checked in _run_trader_gate, the single path that opens a position. Settable at
# runtime from /control, so a bad tape can be stopped from a phone without a
# deploy. It is deliberately NOT persisted as a code change: if the process
# restarts, the saved runtime override re-applies it.
TRADING_PAUSED = False

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


def _assert_state_dir_is_persistent() -> None:
    """Refuse to boot in production if state would land on an ephemeral disk.

    This is the guard that should have existed from the start. On 2026-08-13 the
    published track record was found empty: no Railway volume had ever been
    attached, AEGIS_STATE_DIR was unset, and STATE_DIR silently resolved to
    /app/data on the container overlay — wiped by every one of the five
    redeploys that week. The Firestore mirror below was the supposed safety net
    and had never written a single document, because it targets a database that
    does not exist. Both mechanisms were broken at once and nothing said so.

    Nothing above this line can detect that: mkdir(exist_ok=True) CREATES the
    missing directory, so an unmounted path looks identical to a mounted one,
    and the `except` clause quietly falls back to the in-repo data/ dir. A
    writable path is not a persistent path, and only is_mount() tells them
    apart.

    Local dev and CI are unaffected — the check only runs where RAILWAY_ENVIRONMENT
    is set, and in-repo data/ is the correct target everywhere else.
    """
    if not os.environ.get('RAILWAY_ENVIRONMENT'):
        return

    # Deliberate, documented escape hatch: lets the service boot without a
    # volume during an incident. It is loud on purpose — if this is set in
    # steady state, the guard is off and the track record is unprotected.
    if os.environ.get('AEGIS_ALLOW_EPHEMERAL_STATE') == '1':
        print('[state] WARNING: AEGIS_ALLOW_EPHEMERAL_STATE=1 — mount guard '
              'DISABLED. Runtime state will be destroyed on the next redeploy.',
              flush=True)
        return

    if not os.environ.get('AEGIS_STATE_DIR'):
        raise RuntimeError(
            'AEGIS_STATE_DIR is not set in a Railway environment, so runtime '
            f'state would be written to {STATE_DIR} on the ephemeral container '
            'filesystem and destroyed on the next redeploy. Attach a volume and '
            'set AEGIS_STATE_DIR to its mount path (see docs/MIGRATION_PLAN.md). '
            'To boot anyway and accept the data loss, set '
            'AEGIS_ALLOW_EPHEMERAL_STATE=1.'
        )

    if not STATE_DIR.is_mount():
        raise RuntimeError(
            f'AEGIS_STATE_DIR={STATE_DIR} exists but is NOT a mount point — it '
            'is a plain directory on the container overlay, and everything '
            'written to it is destroyed on the next redeploy. Attach a Railway '
            'volume at this exact path. To boot anyway and accept the data '
            'loss, set AEGIS_ALLOW_EPHEMERAL_STATE=1.'
        )


_assert_state_dir_is_persistent()

TRACK_RECORD_PATH       = STATE_DIR / 'track_record.json'
ALPHA_TRACK_RECORD_PATH = STATE_DIR / 'alpha_track_record.json'
ALPHA_TIMEFRAMES        = ['15m', '30m', '4h', '1d']
PERF_STATE_PATH         = STATE_DIR / 'perf_state.json'
DRIFT_STATE_PATH        = STATE_DIR / 'drift_state.json'
# Would a resting limit at the level have filled? Observation only — see
# LiveEngine._wo_observe. On the VOLUME because the answer takes days to
# accumulate and a deploy must not reset it.
WORKING_ORDER_LOG_PATH  = STATE_DIR / 'working_order_counterfactual.jsonl'
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
# Bumped 2 -> 3 on 2026-08-19 to wipe the track record on deploy.
# A bump is now a real one-time wipe: _hydrate_track_record_from_firestore
# compares this against the generation stamped in the LOCAL file and empties it,
# which is what finally defeats _save_track_record's orphan-preservation. Bump
# this again whenever the record must be cleared without an admin key.
STATE_GENERATION = 3

# ── Regime labels ─────────────────────────────────────────────────────────────
REGIME_TRENDING_BULL      = 'TRENDING_BULL'
REGIME_TRENDING_BEAR      = 'TRENDING_BEAR'
REGIME_RANGING            = 'RANGING'
REGIME_ACCUMULATION       = 'ACCUMULATION'
REGIME_DISTRIBUTION       = 'DISTRIBUTION'
REGIME_VOLATILE_EXPANSION = 'VOLATILE_EXPANSION'
REGIME_VOLATILE_COMPRESS  = 'VOLATILE_COMPRESSION'
REGIME_LIQUIDITY_TRAP     = 'LIQUIDITY_TRAP'
