"""The engine loop.

Owns the scan cycle, the predictor fleet, the wallet and the paper book,
and hands each symbol's decision to TraderGate. The market-structure,
gate, exit and position code lives in the mixins this class composes —
see engine/levels.py, gates.py, exits.py, positions.py.

_process_symbol used to be 2,023 lines; 1,736 of those were the v80..v82
guard chain, unreachable behind `if USE_TRADER_GATE:` since v83 and now
deleted. What is left resolves context and calls the desk.

Extracted verbatim from the single-file live_engine.py.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any
from typing import Deque
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
import asyncio
import time

from scripts.engine.config import ALPHA_TIMEFRAMES as _ALPHA_TIMEFRAMES
from scripts.engine.config import ALPHA_TRACK_RECORD_PATH
from scripts.engine.config import HARD_VETOES as _HARD_VETOES
from scripts.engine.config import MODEL_STORE
from scripts.engine.config import REGIME_ACCUMULATION as _REGIME_ACCUMULATION
from scripts.engine.config import REGIME_DISTRIBUTION as _REGIME_DISTRIBUTION
from scripts.engine.config import REGIME_LIQUIDITY_TRAP as _REGIME_LIQUIDITY_TRAP
from scripts.engine.config import REGIME_RANGING as _REGIME_RANGING
from scripts.engine.config import REGIME_TRENDING_BEAR as _REGIME_TRENDING_BEAR
from scripts.engine.config import REGIME_TRENDING_BULL as _REGIME_TRENDING_BULL
from scripts.engine.config import REGIME_VOLATILE_COMPRESS as _REGIME_VOLATILE_COMPRESS
from scripts.engine.config import REGIME_VOLATILE_EXPANSION as _REGIME_VOLATILE_EXPANSION
from scripts.engine.config import ROOT as _ROOT
# Read the runtime flags THROUGH the config module, not as `from ... import`.
# `from config import USE_TRADER_GATE` copies the value at import time, so
# flipping it later — which tests do, and which is the documented way to fall
# back to the pre-v83 behaviour — would rebind a name nobody reads. Going via
# the module means scripts/engine/config.py is the single mutable source of
# truth and a patch there is honoured everywhere, immediately.
from scripts.engine import config as _cfg
from scripts.engine.exits import ExitsMixin
from scripts.engine.gates import GatesMixin
from scripts.engine.indicators import _range_pos
from scripts.engine.levels import LevelsMixin
from scripts.engine.market_data import _fetch_bids_asks_all
from scripts.engine.market_data import _fetch_spot_price
from scripts.engine.market_data import _has_usdm_perp
from scripts.engine.models import RegimeState
from scripts.engine.models import TokenConfig
from scripts.engine.portfolio import PortfolioGuard
from scripts.engine.portfolio import VirtualWallet
from scripts.engine.positions import PositionsMixin
from scripts.engine.quality import SignalQualityFilter
from scripts.engine.regime import MarketRegimeDetector
from scripts.engine.risk import DynamicRiskEngine
from scripts.engine.state import _hydrate_track_record_from_firestore
from scripts.engine.tracking import DriftMonitor
from scripts.engine.tracking import PerformanceTracker
from src.ml.adaptive import AdaptiveOrchestrator
from src.trading import econ_calendar
from src.trading.gate_scorer import WeightedGateScorer
from src.trading.trader_gate import ACTION_ENTER
from src.trading.trader_gate import ACTION_WORK
from src.trading.trader_gate import TraderGate
from src.trading.trendline_channel import TrendlineChannelDetector

class LiveEngine(LevelsMixin, GatesMixin, ExitsMixin, PositionsMixin):
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
    # How often the exit monitor re-prices open positions. This is the real
    # granularity of every stop and TP in the engine; the scan interval (300s)
    # governs ENTRIES only. Cheap loop — no inference, no network.
    EXIT_CHECK_SECONDS    = 3
    HOURS_CONTEXT         = 300
    MIN_HOLD_SECONDS      = 3_600    # 1 h minimum hold before model-reversal exit
    MAX_HOLD_SECONDS      = 24 * 3_600  # 24 h zombie guard for open positions
    COOLDOWN_SECONDS      = 300    # 5 min post-close cooldown (any outcome)
    FLIP_COOLDOWN_SECONDS = 600    # 10 min cooldown when the new signal flips direction
    LOSS_COOLDOWN_SECONDS = 4 * 3_600  # 4 h post-loss cooldown before re-entry
    SIGNAL_STABILITY_WINDOW = 3    # directional samples required by the stability gate
    # Edge threshold for bypassing the consecutive-direction stability check.
    # Keep this on LiveEngine because the gate reads it through ``self``.
    SIGNAL_BYPASS_EDGE = 85.0
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
    # Closed 1h bars used by the BOS/CHoCH confirmation gate below.  This is
    # an engine setting because the gate uses it for both candle retrieval and
    # the detector's lookback argument.
    CONFIRM_BOS_LOOKBACK = 20
    # Parameters for the divergence and volume-event detectors used by the
    # confirmation gate.  Keep these as class attributes because the gate reads
    # its tunable settings through ``self``.
    CONFIRM_PIVOT_K    = 3
    CONFIRM_VOL_WINDOW = 20
    CONFIRM_CLIMAX_Z   = 2.0
    CONFIRM_ABSORB_Z   = 1.5
    # Minimum number of closed 1h bars required before confirmation analysis.
    # This is read through ``self`` by _confirmation_gate.
    CONFIRM_MIN_BARS = CONFIRM_BOS_LOOKBACK + 2
    # Agreement thresholds used by _confirmation_gate.  These are class
    # attributes because that method reads them through ``self``.
    CONFIRM_CONFLICT_THRESHOLD = 2.0
    CONFIRM_CONFIRM_THRESHOLD = 2.0
    # Confirmed breakout CONTINUATION (a break that runs and never retests):
    #   last N closed 5m candles must ALL hold beyond the broken level, and price
    #   must not have run more than MAX_EXT ATRs past it (no late chase).
    STRUCT_BREAKOUT_MIN_HOLD    = 3
    STRUCT_BREAKOUT_MAX_EXT_ATR = 2.5
    # Reversal entries fire close to the level: allowed when a wick has TAGGED the
    # level OR price is within this many ATRs of it. Back to 0.9 (0.5 was too tight
    # and muted most S/R reversals — the PRIMARY setup — leaving the fire rate low).
    STRUCT_LEVEL_PROXIMITY_ATR = 0.35
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
    # Counter-trend reversal candle settings used by Guard B.  Keep the tag
    # lookback wider than the confirmation window so a prior wick touch remains
    # valid while the latest closed candles confirm the turn.
    REVERSAL_TAG_LOOKBACK = 12
    REVERSAL_5M_WINDOW    = 4
    REVERSAL_5M_MIN       = 3
    # ATR tolerance used when checking whether a closed 5m candle tagged the
    # counter-trend reversal level.  This is distinct from the maximum chase
    # distance allowed for the eventual entry.
    REVERSAL_PROX_ATR     = 0.35
    # Maximum distance from the tagged support/resistance level at which a
    # counter-trend reversal may still enter. This is separate from the tag
    # tolerance because price may touch a level and then move away before the
    # confirmation candles complete.
    REVERSAL_MAX_CHASE_ATR = 2.0

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

    # Distance from the opposing higher-timeframe level at which an RSI-extreme
    # entry is considered an exhaustion chase.  Keep this separate from the
    # absolute no-room threshold: a trade may have enough space for TP1 while
    # still being too close to the wall to enter on stretched momentum.
    HTF_EXHAUSTION_ATR = 1.0

    # Guard I thresholds for the opposing 4h/1d support or resistance.
    # Entries with less than TP1's projected distance to that wall are hard
    # blocked; entries inside the wider advisory band are tagged RISKY.
    HTF_NO_ROOM_ATR  = 0.55
    HTF_ADVISORY_ATR = 1.50

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
    # Minimum model edge percentile required before the directional-conviction
    # floor is relaxed at a structural extreme.  ``edge_score`` is normalised
    # to the model's 0-100 percentile scale before Guard D reaches this check.
    EXTREME_EDGE_MIN = 80.0

    # RETIRED (v54): a raw p_buy−p_sell conviction floor is incompatible with how
    # these models express confidence. edge_score is a PERCENTILE RANK of the
    # model's edge vs its own history, not the raw probability spread — the
    # 3-class models sit near 50/50 on buy-vs-sell (p_hold holds the mass), which
    # is exactly why calibration uses percentile edge. A 10pt floor blocked 100%
    # of real model fires (LDO 0.9pt, FIL 1.4pt) while the NON-firing tokens
    # carried the big spreads (SOL 32.8pt, AVAX 35.0pt). The model's own
    # edge>=threshold IS the conviction check. Kept only for reference.
    MIN_MODEL_CONVICTION = 0.0   # unused — see Guard D REMOVED in _process_symbol

    # Guard D directional-conviction thresholds.  These are class settings
    # because the gate reads them through ``self`` in _process_symbol.
    MIN_DIR_CONVICTION         = 0.10
    MIN_DIR_CONVICTION_EXTREME = 0.05

    # Guard L directional-confluence floor. Evidence is signed for the model's
    # selected side, so non-reversal entries need a net positive consensus.
    MIN_DIR_CONFLUENCE         = 1

    # Minimum number of core technical indicators that must agree with the
    # model's side (macd, supertrend, market_bias, htf_daily, htf_weekly).
    # If fewer indicators support the side, block the model fire unless the
    # setup is a confirmed reversal at a level or TRUST_MODEL_FIRE is enabled.
    MIN_INDICATOR_SUPPORT      = 2

    # Maximum number of advisory warnings allowed before a signal becomes
    # HIGH risk and is held.  This is read by _process_symbol via self.
    ADVISORY_WARNING_BUDGET    = 3

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
    AT_LEVEL_ATR      = 0.35  # price is "at" a level within this many ATR (thin entry gap)

    # ── Guard M: hold every signal until price is AT the level ───────────────
    # A signal no longer fires mid-range. It is HELD (pending) until price is
    # within PENDING_NEAR_PCT of the level it should reverse at — a BUY at the
    # nearest important SUPPORT below, a SELL at the nearest important RESISTANCE
    # above — measured in PRICE, not ATR.
    PENDING_NEAR_PCT  = 0.35  # price within 0.35% of the level counts as "at" it

    # Minimum time a pending setup must be absent before its notification can
    # be emitted again.  Keep this on LiveEngine because the notifier uses it
    # as an instance attribute during each scan.
    PENDING_ALERT_COOLDOWN = 4 * 60 * 60

    # Regimes where entry is unconditionally blocked.  RANGING was demoted to
    # an advisory warning: the structure gate (BUY at support / SELL at
    # resistance) plus candlestick reversal confirmation are exactly the
    # setups that remain valid inside a range, so a blanket ban both starved
    # the engine of signals and contradicted those gates.
    NO_TRADE_REGIMES: set = {_REGIME_LIQUIDITY_TRAP}
    # Temporary strict deployment mode: disable trust-model relaxations so
    # structure/indicator guards hard-block weak or contradictory model fires.
    TRUST_MODEL_FIRE: bool = False

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
        # v83 working orders (TraderGate): 'SYMBOL|SIDE|LEVEL' -> first time this
        # exact resting order was offered. A setup the market ignores for
        # WORK_EXPIRY_BARS is a dead thesis, not a queue entry — this is the
        # clock that PENDING never had.
        self._working_orders: Dict[str, float] = {}
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
        self._tide_strength: float = 0.0   # 0..1, how hard that tide runs (v83)
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
        self.adaptive_orchestrator = AdaptiveOrchestrator()

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
                    if getattr(self, 'adaptive_orchestrator', None) is not None:
                        p.attach_adaptive_orchestrator(self.adaptive_orchestrator)
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
                    if getattr(self, 'adaptive_orchestrator', None) is not None:
                        p.attach_adaptive_orchestrator(self.adaptive_orchestrator)
                    self.predictors[sym] = p
                    loaded += 1
                    print(f'[LiveEngine] BENCHED {sym} — meta risk_tier disables '
                          f'"{self.risk_tier}"; it can never fire (monitor-only)')
                    continue

                # Fallback: legacy single-direction model
                if p.model is not None:
                    if getattr(self, 'adaptive_orchestrator', None) is not None:
                        p.attach_adaptive_orchestrator(self.adaptive_orchestrator)
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

    async def _exit_monitor_loop(self) -> None:
        """Check stops/TPs against the live feed, independent of the scan cycle.

        Until v82d the ONLY call to _manage_exit was inside _process_symbol, so
        an open position's stop was evaluated once per scan_interval_seconds
        (300s) and only after that symbol's turn through a semaphore of
        MAX_CONCURRENT predict_realtime() calls — each a 350h feature build —
        across every tradeable token.  Prices meanwhile stream in continuously
        from _ws_price_ticker, which writes live_prices and checks nothing.

        The result was that a stop was a suggestion: price ran past it, and
        _close('STOP_HIT') then filled at whatever the market was doing minutes
        later.  Measured on the four closed trades of 2026-08-01, every loss
        overshot its own stop — by 0.02, 0.14, 0.85 and 0.28 percentage points,
        1.28pp of avoidable loss across four trades, on stops of 1.1-2.3%.

        This loop closes that window.  It is deliberately cheap: no inference,
        no network, no feature build — just the arithmetic already in
        _manage_exit against the newest price.
        """
        while True:
            try:
                await asyncio.sleep(self.EXIT_CHECK_SECONDS)
                if not self.wallet.open_positions:
                    continue
                # snapshot: _manage_exit mutates open_positions on a close
                for sym, pos in list(self.wallet.open_positions.items()):
                    px = float(self.live_prices.get(sym, 0.0) or 0.0)
                    if px <= 0:
                        continue
                    # last_signals supplies ATR only; the reversal branch is
                    # suppressed by price_only so a stale entry cannot act.
                    ctx = self.last_signals.get(sym) or {}
                    if not isinstance(ctx, dict):
                        ctx = {}
                    self._manage_exit(sym, pos, ctx, px, price_only=True)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # Never let a bad tick kill the monitor — an unsupervised book
                # is strictly worse than a logged error.
                print(f'[LiveEngine] exit monitor error (loop stays alive): {e!r}')

    async def run(self) -> None:
        print(f'[LiveEngine] Starting — interval={self.scan_interval_seconds}s '
              f'symbols={len(self.predictors)} | {self.GATE_VERSION}')
        asyncio.create_task(self._ws_price_ticker())
        asyncio.create_task(self._exit_monitor_loop())
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
                import json as _json
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
        if _cfg.USE_TRADER_GATE:
            # v83: `_armed_pending_setups` is legacy PENDING state that the desk
            # never populates, and the desk republishes the working-order fields
            # authoritatively on every scan. Leaving this running would give two
            # writers to the same display keys — exactly the class of coupling
            # that made the old chain unreadable.
            return
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

    def _register_armed_pending_setup(self, symbol: str, side: str, target: Optional[float], reason: str) -> None:
        """Register or refresh a persistent armed setup for a waiting signal."""
        self._armed_pending_setups[symbol] = {
            'side':        side,
            'target':      round(target, 10) if target else None,
            'reason':      reason,
            'armed_time':  time.time(),
        }

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

    def _handle_benched_symbol(self, symbol: str) -> bool:
        """Check if symbol is benched and update signals. Returns True if benched."""
        if symbol not in getattr(self, '_benched', ()):
            return False
        self.last_signals.setdefault(symbol, {}).update({
            'symbol':  symbol,
            'signal':  'HOLD',
            'fire':    False,
            'benched': True,
            'price':   self.live_prices.get(symbol, 0.0),
        })
        self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
        return True

    def _resolve_price(self, result: Dict[str, Any], symbol: str) -> float:
        """Resolve price: prefer WS tick over stale model price."""
        _model_price = float(result.get('price', 0) or 0)
        _ws_price = float(self.live_prices.get(symbol, 0) or 0)
        price = _ws_price if _ws_price > 0 else _model_price
        if _model_price > 0 and _ws_price == 0:
            self.live_prices[symbol] = _model_price
        return price

    def _extract_hmm_fields(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract HMM regime fields from prediction result."""
        return {
            'regime':        result.get('hmm_regime', 'UNKNOWN'),
            'available':     bool(result.get('hmm_available', False)),
            'conf_adj':      float(result.get('hmm_conf_adjustment', 0.0)),
            'atr_mult':      float(result.get('hmm_atr_mult', 1.0)),
            'pos_scale':     float(result.get('hmm_position_scale', 1.0)),
            'trade_ok':      bool(result.get('hmm_trade_allowed', True)),
            'trans_risk':    float(result.get('hmm_transition_risk', 0.0)),
        }

    def _check_hmm_trade_veto(self, symbol: str, hmm: Dict[str, Any]) -> bool:
        """Check if HMM blocks trading. Returns True if blocked."""
        if not hmm['available'] or hmm['trade_ok']:
            return False
        if symbol in self.last_signals:
            self.last_signals[symbol]['fire']         = False
            self.last_signals[symbol]['signal']       = 'HOLD'
            self.last_signals[symbol]['hmm_blocked']  = True
        self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
        return True

    def _update_regime_from_hmm(self, regime, result: Dict[str, Any], hmm: Dict[str, Any]):
        """Update regime based on HMM confidence."""
        if not hmm['available'] or float(result.get('hmm_confidence', 0)) <= 0.5:
            return regime
        _HMM_TO_INTERNAL = {
            'TRENDING_BULL':      _REGIME_TRENDING_BULL,
            'TRENDING_BEAR':      _REGIME_TRENDING_BEAR,
            'CHOPPY':             _REGIME_RANGING,
            'VOLATILE_EXPANSION': _REGIME_VOLATILE_EXPANSION,
            'COMPRESSION':        _REGIME_VOLATILE_COMPRESS,
            'ACCUMULATION':       _REGIME_ACCUMULATION,
            'DISTRIBUTION':       _REGIME_DISTRIBUTION,
        }
        _internal = _HMM_TO_INTERNAL.get(hmm['regime'])
        if _internal:
            return RegimeState(
                regime               = _internal,
                confidence           = float(result.get('hmm_confidence', 0.5)),
                trade_allowed        = hmm['trade_ok'],
                preferred_strategies = regime.preferred_strategies,
                max_position_pct     = regime.max_position_pct * hmm['pos_scale'],
            )
        return regime

    def _record_signal_direction(self, symbol: str, new_side: str) -> None:
        """Append a directional read to the stability-gate history.

        Only BUY/SELL accumulate; FLAT/HOLD is ignored so BUY->FLAT->BUY does
        not erase the counter (volatile markets produce intermittent FLAT
        cycles). A genuine flip needs no explicit clear — the stability check
        is all(s == new_side), which fails once the deque holds the other side.
        """
        if new_side not in ('BUY', 'SELL'):
            return
        if symbol not in self._signal_history:
            self._signal_history[symbol] = deque(maxlen=self.SIGNAL_STABILITY_WINDOW)
        self._signal_history[symbol].append(new_side)

    async def _apply_htf_macro_bias(self, symbol: str, result: Dict[str, Any]) -> None:
        """Recompute and inject the daily/weekly trend bias.

        macro_daily / macro_weekly arrive 0.0 from the model (a swallowed
        KeyError zeroes them at the source — see _daily_bias). They are
        recomputed from daily candles and injected BEFORE scoring so the HTF
        tiers in score_signal() (+15/+10 aligned, -20/-10 against) and Guard F
        (hard-block when BOTH daily and weekly oppose) come alive. Trade WITH
        the higher-timeframe trend; a genuine reversal at a level stays exempt
        via score_signal's `is_reversal` branch.
        """
        result['macro_daily'], result['macro_weekly'] = await self._daily_bias(symbol)

    @staticmethod
    def _level_gap_atr(result: Dict[str, Any], side: str) -> Optional[float]:
        """Distance from price to the level this side leans on, in ATR.

        BUY leans on support, SELL on resistance. Returns None when price, ATR
        or the level is missing, so callers can fall back rather than treat an
        unknown as "at the level". A NEGATIVE value means price has already
        traded through the level — the setup's premise is gone.
        """
        try:
            px = float(result.get('price') or result.get('entry_price') or 0.0)
            atr = float(result.get('atr') or 0.0)
            if atr <= 0:
                atr_pct = float(result.get('atr_pct') or 0.0)
                atr = px * atr_pct / 100.0
            if px <= 0 or atr <= 0:
                return None
            if side == 'BUY':
                lvl = float(result.get('support') or 0.0)
                return (px - lvl) / atr if lvl > 0 else None
            if side == 'SELL':
                lvl = float(result.get('resistance') or 0.0)
                return (lvl - px) / atr if lvl > 0 else None
        except (TypeError, ValueError):
            return None
        return None

    def _manage_paper_position(
        self, symbol: str, result: Dict[str, Any], price: float
    ) -> None:
        """v74: manage the RISKY paper book (alpha wallet, SYMBOL|risky).

        RISKY-tier fires trade on paper until the tagged population proves
        itself (see the fire path). Exits mirror the alpha scanner's rules: SL,
        TP3 full take, an opposing model fire, and the 24h zombie guard.
        """
        key = f'{symbol}|risky'
        pos = self.alpha_wallet.open_positions.get(key)
        if pos is None:
            return
        px = float(self.live_prices.get(symbol, 0) or price or 0)
        if px <= 0:
            return

        is_long = pos.direction == 'LONG'
        hit_sl  = (px <= pos.stop_loss) if is_long else (px >= pos.stop_loss)
        hit_tp  = (pos.take_profit_3 > 0 and
                   ((px >= pos.take_profit_3) if is_long
                    else (px <= pos.take_profit_3)))
        reversed_ = (bool(result.get('fire'))
                     and result.get('side') in ('BUY', 'SELL')
                     and result.get('side') != pos.side)
        expired = (time.time() - self._alpha_open_time.get(key, time.time())
                   >= self.MAX_HOLD_SECONDS)

        why = ('SL_HIT' if hit_sl else 'TP3_HIT' if hit_tp else
               'SIGNAL_REVERSAL' if reversed_ else
               'MAX_HOLD_EXPIRED' if expired else '')
        if not why:
            return
        self.alpha_wallet.close_trade(key, px, why)
        self._alpha_last_close_time[key] = time.time()
        self._save_alpha_track_record()
        print(f'[{symbol}] RISKY-PAPER {why} @ {px:.6g}')

    async def _resolve_market_context(
        self, symbol: str, result: Dict[str, Any]
    ) -> Optional[Tuple[float, Any, Dict[str, Any], str, float, bool]]:
        """Turn a raw prediction into the context every downstream gate reads.

        Returns (price, regime, hmm, new_side, quality_score, fake_breakout),
        or None when the HMM vetoes the symbol outright and the scan should
        stop here.

        score_signal()'s `reasons` half is deliberately dropped: its only reader
        was the legacy chain's advisory Gate 3b, which is gone. That gate read a
        name (`_quality_reasons`) bound nowhere in the module, so it would have
        raised NameError had anything reached it — nothing did, because the
        USE_TRADER_GATE branch above returns first. If a future gate wants the
        reasons, return them here rather than reaching for a name that does not
        exist.
        """
        price = self._resolve_price(result, symbol)

        # ── Adaptive intelligence layer ───────────────────────────────────
        # Step 1: HMM regime (probabilistic, from predictor's result dict).
        # The HMM ran inside predict_realtime() and attached hmm_* fields;
        # they are extracted here and used to sharpen MarketRegimeDetector.
        hmm = self._extract_hmm_fields(result)

        # If HMM says no-trade (e.g. COMPRESSION pre-breakout or DISTRIBUTION)
        # suppress immediately — don't spend the quality-scoring pass.
        if self._check_hmm_trade_veto(symbol, hmm):
            return None

        # Step 2: rule-based regime classifier, then let a confident HMM read
        # override the heuristic label.
        regime = self.regime_detector.detect(result)
        regime = self._update_regime_from_hmm(regime, result, hmm)

        new_side = result.get('side', 'FLAT')
        self._record_signal_direction(symbol, new_side)
        await self._apply_htf_macro_bias(symbol, result)

        quality_score, _reasons = self.quality_filter.score_signal(
            result, regime, new_side)
        fake_breakout = self.quality_filter.is_fake_breakout(result, new_side)
        return price, regime, hmm, new_side, quality_score, fake_breakout

    async def _ltf_confirmation(self, symbol: str) -> Dict[str, bool]:
        """Has the 5m tape turned?  {'ltf_bull': ..., 'ltf_bear': ...}.

        One of the independent prints TraderGate's trigger stage counts. Reuses
        the ENTRY_5M_WINDOW the old Guard J used, so 'the 5m turned' means the
        same thing it always did — but it now contributes evidence rather than
        holding a veto.
        """
        out = {'ltf_bull': False, 'ltf_bear': False}
        try:
            raw = await self._fetch_candles(symbol, '5m', self.ENTRY_5M_WINDOW + 2)
            closed = raw[:-1] if len(raw) >= 2 else raw
            window = closed[-self.ENTRY_5M_WINDOW:]
            if len(window) < self.ENTRY_5M_WINDOW:
                return out
            ups = sum(1 for c in window if float(c[4]) > float(c[1]))
            need = max(3, self.ENTRY_5M_WINDOW - 1)
            out['ltf_bull'] = ups >= need
            out['ltf_bear'] = (len(window) - ups) >= need
        except Exception:
            pass
        return out

    def _cluster_exposure(self, symbol: str) -> Tuple[int, int]:
        """(longs, shorts) already open in THIS symbol's correlation cluster.

        Counted across the real book AND the paper book: eight correlated shorts
        are one bet however they are accounted for, and the 2026-07-20 basket
        was booked to paper.
        """
        pg = self.portfolio_guard
        cluster = pg._sym_to_cluster.get(symbol)
        if not cluster:
            return 0, 0
        members = set(pg._CLUSTERS.get(cluster, []))
        longs = shorts = 0
        for sym, pos in self.wallet.open_positions.items():
            if sym in members:
                longs += pos.direction == 'LONG'
                shorts += pos.direction == 'SHORT'
        for key, pos in self.alpha_wallet.open_positions.items():
            if key.split('|')[0] in members:
                longs += pos.direction == 'LONG'
                shorts += pos.direction == 'SHORT'
        return longs, shorts

    async def _run_trader_gate(
        self, symbol: str, result: Dict[str, Any], price: float,
        regime: Any, ctx_quality: float,
    ) -> bool:
        """v83 · run the desk playbook. Returns True once the symbol is handled.

        This is the whole decision path when USE_TRADER_GATE is on: it replaces
        Guards A..T and the PENDING queue. The gate itself is pure; everything
        async or stateful (structure, tide, book, the working-order clock) is
        assembled here and handed in.
        """
        atr = float(result.get('atr', 0) or 0)

        # A token that just lost is benched — the failed thesis is usually still
        # in play, and re-firing it is the revenge trade. Kept from the old chain
        # because it is a trader's rule, not a guard's.
        now = time.time()
        if now - self._last_loss_time.get(symbol, 0) < self.LOSS_COOLDOWN_SECONDS:
            self._publish_no_trade(symbol, 'benched — lost on this token within the last '
                                           f'{self.LOSS_COOLDOWN_SECONDS // 3600}h')
            return True
        if now - self._last_close_time.get(symbol, 0) < self.COOLDOWN_SECONDS:
            self._publish_no_trade(symbol, 'cooling off after the last close')
            return True

        try:
            levels = await self._structural_levels(symbol, price, atr)
        except Exception:
            levels = []
        tide = await self._btc_tide()
        confirm = await self._ltf_confirmation(symbol)
        longs, shorts = self._cluster_exposure(symbol)

        result['price'] = price
        plan = TraderGate.evaluate(
            result, regime,
            market={
                'drift_blocked':   self.drift_monitor.is_blocked(symbol),
                'drift_severity':  self.drift_monitor.severity(symbol),
                'news_locked':     self._news_lock[0],
                'news_label':      self._news_lock[1],
                'spread_pct':      self._spreads.get(symbol, 0.0),
                'tide_dir':        tide,
                'tide_strength':   self._tide_strength,
            },
            book={
                'open_total':      len(self.wallet.open_positions),
                'max_open':        self.portfolio_guard.MAX_OPEN_TOTAL,
                'cluster_long':    longs,
                'cluster_short':   shorts,
                'max_per_cluster': self.portfolio_guard.MAX_PER_CLUSTER,
            },
            levels=levels,
            confirm=confirm,
        )

        sig = self.last_signals.get(symbol)
        if isinstance(sig, dict):
            sig['trade_plan']       = plan.as_dict()
            sig['structure_reason'] = plan.reason
            sig['gate_stage']       = plan.stage
            sig['setup_type']       = plan.setup

        # ── REJECT ───────────────────────────────────────────────────────────
        if plan.action not in (ACTION_ENTER, ACTION_WORK):
            self._working_orders.pop(f'{symbol}|{plan.side}', None)
            self._publish_no_trade(symbol, plan.reason)
            print(f'[{symbol}] NO TRADE ({plan.stage}): {plan.reason}')
            return True

        # ── WORK · a resting order, with a clock ─────────────────────────────
        # The clock is the entire difference from PENDING. A setup the market
        # has ignored for WORK_EXPIRY_BARS is a thesis that did not happen, and
        # it is retired rather than re-offered every scan forever.
        if plan.action == ACTION_WORK:
            key = f'{symbol}|{plan.side}'
            first_seen = self._working_orders.setdefault(key, now)
            age_bars = (now - first_seen) / 3600.0        # 1h engine timeframe
            if age_bars > plan.expiry_bars:
                self._working_orders.pop(key, None)
                self._publish_no_trade(
                    symbol, f'setup expired — {plan.setup} {plan.side} at '
                            f'{plan.level:.8g} went untriggered for {plan.expiry_bars} bars')
                print(f'[{symbol}] SETUP EXPIRED: {plan.setup} {plan.side} '
                      f'after {age_bars:.1f} bars')
                return True
            if isinstance(sig, dict):
                sig['fire']           = False
                sig['signal']         = 'HOLD'
                sig['working_order']  = True
                sig['pending_entry']  = True     # UI compatibility: same card slot
                sig['pending_side']   = plan.side
                sig['pending_target'] = plan.level
                sig['pending_reason'] = plan.reason
                sig['expires_in_bars'] = round(plan.expiry_bars - age_bars, 1)
                # `_build_signal_entry` drew these off the MODEL's side earlier
                # in the scan. The desk can choose the other side, and a card
                # showing a long's stop under a short setup is worse than no
                # card — republish from the plan the user is actually being told
                # about.
                sig['suggested_sl'] = round(plan.stop, 8)
                sig['suggested_tp'] = round(plan.target, 8)
                sig['direction']    = 'LONG' if plan.side == 'BUY' else 'SHORT'
            print(f'[{symbol}] WORKING {plan.side} @ {plan.level:.8g} — {plan.reason} '
                  f'({plan.expiry_bars - age_bars:.1f} bars left)')
            return True

        # ── ENTER ────────────────────────────────────────────────────────────
        self._working_orders.pop(f'{symbol}|{plan.side}', None)
        result['side']      = plan.side
        result['fire']      = True
        result['btc_tide']  = tide
        # The plan's own level is the invalidation the stop was built from; hand
        # it to _open_position so the frozen record shows the real thesis.
        result['at_pending_level'] = {'level': plan.level, 'role': plan.setup}

        tier = ('STRONG' if plan.r_net >= 2.5 and plan.size_factor >= 0.85
                else 'NORMAL' if plan.r_net >= 2.0 or plan.size_factor >= 0.7
                else 'RISKY')
        print(f'[{symbol}] PLAN ENTER {plan.side} {plan.setup} @ {price:.8g} '
              f'stop {plan.stop:.8g} target {plan.target:.8g} '
              f'{plan.r_net:.2f}R net size x{plan.size_factor:.2f} tier={tier}')

        self._open_position(symbol, result, price, regime, ctx_quality,
                            risk_tier=tier, entry_mode=plan.setup.lower(),
                            gate_warnings=[], plan=plan)

        _pos = self.wallet.open_positions.get(symbol)
        if _pos is not None and isinstance(sig, dict):
            sig['fire']            = True
            sig['signal']          = plan.side
            sig['direction']       = 'LONG' if plan.side == 'BUY' else 'SHORT'
            sig['signal_strength'] = plan.side
            sig['evaluating']      = False
            sig['risk_tier']       = tier
            sig['entry_mode']      = plan.setup.lower()
            sig['working_order']   = False
            sig['pending_entry']   = False
            # Publish the levels the POSITION actually holds, not the ones
            # _build_signal_entry derived from the model's side earlier in this
            # same scan — those can belong to the opposite direction entirely.
            sig['entry_price']     = _pos.entry_price
            sig['suggested_sl']    = _pos.stop_loss
            # v85: the headline TP is the objective the plan was priced on
            # (take_profit_3), not the 1.0R first-bank rung — publishing TP1 here
            # re-advertised the 1:1 that the track-record fix removed, and it is
            # the number a subscriber acts on.  TP1 stays visible as `tp1`.
            sig['suggested_tp']    = _pos.take_profit_3 or _pos.take_profit_1
            sig['tp1']             = _pos.take_profit_1
            sig['tp2'], sig['tp3'] = _pos.take_profit_2, _pos.take_profit_3
            sig['tp4'], sig['tp5'] = _pos.take_profit_4, _pos.take_profit_5
            # The headline R:R is the number the trade was APPROVED on.
            #
            # It was being left at whatever _build_signal_entry computed earlier
            # in this scan — from the model's own side and levels, which the note
            # above says can belong to the opposite direction entirely. Every
            # other level in this block is republished from the position; this
            # one was missed.
            #
            # It was also structurally uninformative. _build_signal_entry quotes
            # |price - tp2| / risk, and calculate_stops sets tp2 = price + 2.0R,
            # so the ratio is 2.00 by construction whenever the ladder is not
            # compressed. When the plan's objective sits between 1.6R and 2.0R
            # the compression branch pulls tp2 back to 2/3 of the span, and the
            # published figure drops BELOW the MIN_NET_R floor the gate just
            # enforced — a trade approved at 2.5R net could advertise 1.18.
            #
            # plan.r_net is the figure stage 3 actually cleared: reward and risk
            # measured to the real objective, with the round trip taken off the
            # win and added to the loss. Quote that, and keep the ladder ratios
            # alongside it under names that say what they are.
            sig['risk_reward']       = round(plan.r_net, 2)
            sig['risk_reward_gross'] = round(plan.r_gross, 2)
            _risk_leg = abs(_pos.entry_price - _pos.stop_loss)
            if _risk_leg > 0:
                sig['rr_to_tp2'] = round(
                    abs(_pos.take_profit_2 - _pos.entry_price) / _risk_leg, 2)
                sig['rr_to_tp5'] = round(
                    abs(_pos.take_profit_5 - _pos.entry_price) / _risk_leg, 2)
            sig['levels_frozen']   = True
        return True

    def _publish_no_trade(self, symbol: str, reason: str) -> None:
        """Mark a symbol as not trading, with the desk's reason attached.

        Every refusal carries its own explanation — the old chain's `HOLD` with
        no attribution is what made 16 interacting guards impossible to debug.
        """
        sig = self.last_signals.get(symbol)
        if isinstance(sig, dict):
            sig['fire']             = False
            sig['signal']           = 'HOLD'
            sig['evaluating']       = False
            sig['working_order']    = False
            sig['pending_entry']    = False
            sig['structure_reason'] = reason

    async def _process_symbol(
        self, symbol: str, predictor: Any, sem: asyncio.Semaphore
    ) -> None:
        # BENCHED model: its meta's risk_tier disables the tier we run, so
        # predict_signal() would early-return fire=False / edge=0 anyway. Skip
        # the call entirely — it costs a full 350h feature build per scan for a
        # result that can never fire. Still surfaced as a monitor-only token.
        if self._handle_benched_symbol(symbol):
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

            _ctx = await self._resolve_market_context(symbol, result)
            if _ctx is None:            # HMM no-trade veto
                return
            price, regime, hmm, new_side, quality_score, fake_breakout = _ctx

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
            if _cfg.USE_WEIGHTED_SCORER:
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
                
                # ── 3rd-Party Arbiter: Reversal vs. Conflicting Model Check ─────────────
                _rp_arbiter = _range_pos(result)
                _cdl_bull = bool(result.get('cdl_bull_reversal'))
                _cdl_bear = bool(result.get('cdl_bear_reversal'))
                
                # Model says SELL, but price is in Support Zone and technicals *might*
                # confirm a bullish bounce. Only override the model if the reversal
                # candle/pattern AND core indicators (MACD + trend bias) ALSO agree.
                macd_sig = str(result.get('macd_signal', 'NEUTRAL') or 'NEUTRAL').upper()
                supertrend = str(result.get('supertrend', '') or '').upper()
                market_bias = str(result.get('market_bias', '') or '').upper()

                if _model_side == 'SELL' and _rp_arbiter <= self.STRUCT_SUPPORT_ZONE and not bool(result.get('support_broken_recent')):
                    # require both a reversal signal AND MACD+trend confirmation
                    if (_cdl_bull or float(result.get('rsi', 50) or 50) < 42) and macd_sig == 'BULLISH' and (('BULL' in supertrend) or (market_bias == 'BULLISH')):
                        _model_side = 'BUY'
                        _model_fire = True
                        result['reversal_override'] = True
                        print(f'[{symbol}] 3RD-PARTY ARBITER: Overriding SELL→BUY at support: reversal + MACD/trend confirm')
                    else:
                        # Do not aggressively flip the model on weak or isolated patterns
                        if (_cdl_bull or float(result.get('rsi', 50) or 50) < 42):
                            print(f'[{symbol}] ARBITER SKIP: bullish pattern present but MACD/trend not confirming (macd={macd_sig}, supertrend={supertrend}, bias={market_bias})')
                # Model says BUY, but price is in Resistance Zone and technicals confirm bearish rejection
                elif _model_side == 'BUY' and _rp_arbiter >= self.STRUCT_RESISTANCE_ZONE and not bool(result.get('resistance_broken_recent')):
                    if (_cdl_bear or float(result.get('rsi', 50) or 50) > 58) and macd_sig == 'BEARISH' and (('BEAR' in supertrend) or (market_bias == 'BEARISH')):
                        _model_side = 'SELL'
                        _model_fire = True
                        result['reversal_override'] = True
                        print(f'[{symbol}] 3RD-PARTY ARBITER: Overriding BUY→SELL at resistance: reversal + MACD/trend confirm')
                    else:
                        if (_cdl_bear or float(result.get('rsi', 50) or 50) > 58):
                            print(f'[{symbol}] ARBITER SKIP: bearish pattern present but MACD/trend not confirming (macd={macd_sig}, supertrend={supertrend}, bias={market_bias})')

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
                # Pick the scale from WHICH field supplied the number, not from
                # its value. `if _edge <= 1.0: _edge *= 100` guessed, and it
                # guesses wrong in the one direction that matters: edge_score is
                # a 0-100 percentile, so a genuine bottom-percentile bar of 0.8
                # was inflated to 80 and sized as high conviction, while a true
                # 0.0 stayed 0. meta_confidence is the 0-1 field and is the only
                # one that needs scaling.
                if result.get('edge_score') is not None:
                    _edge = float(result.get('edge_score') or 0.0)      # already 0-100
                else:
                    _edge = float(result.get('meta_confidence') or 0.0) * 100.0
                _edge = max(0.0, min(_edge, 100.0))
                result['quality_score'] = round(_edge, 1)
                new_side      = result['side']
                quality_score = result['quality_score']

            # Build signal entry with enriched fields
            self.last_signals[symbol] = self._build_signal_entry(
                symbol, result, price, regime=regime,
                quality_score=quality_score, fake_breakout=fake_breakout,
                open_pos=self.wallet.open_positions.get(symbol))

            try:
                self.adaptive_orchestrator.record_signal(self.last_signals[symbol])
                self.last_signals[symbol] = self.adaptive_orchestrator.evaluate_signal(
                    self.last_signals[symbol])
            except Exception:
                pass

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

            self._manage_paper_position(symbol, result, price)

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
            # ── v83 · the desk decides ────────────────────────────────────────
            # Deliberately NOT conditioned on result['fire']. Under the playbook
            # the SETUP picks the side and the model may only object (see
            # MODEL_OPPOSE_MARGIN in trader_gate.py) — requiring the model to
            # fire first would restore exactly the permission-by-conviction that
            # produced the 2026-07-20 basket, where the losers' confidence ran
            # from 17.9 to 100.0. Flip REQUIRE_MODEL_FIRE in trader_gate.py to
            # restore model-first permissioning without touching this path.
            #
            # This is the whole decision path. The v80..v82 guard chain that used
            # to follow — 1,736 lines behind `if USE_TRADER_GATE:` — was deleted
            # once it had been dead in production for the whole of v83..v85. It
            # was unreachable (the branch above returns unconditionally), pinned
            # only by a characterisation baseline that tested nothing else, and
            # it still contained a NameError on `_quality_reasons` that could not
            # be observed because _scan_all swallows exceptions. `git log` is the
            # rollback; a flag that guards code nobody runs is not a safety net.
            if (not existing or _reversal_flip) and price > 0 \
                    and result.get('tradeable', False):
                await self._run_trader_gate(symbol, result, price,
                                            regime, quality_score)
            # Counted exactly once per symbol, on every path through the
            # playbook — the desk's own helpers deliberately do not touch it.
            self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)

    # ── multi-timeframe candle fetch ──────────────────────────────────────────

















    # ── structure gate (Gate 1.6) ─────────────────────────────────────────────


    # ── confirmation gate (Gate 1.7) ──────────────────────────────────────────

    # ── trade management ──────────────────────────────────────────────────────



    # ── signal entry builder (for dashboard / last_signals) ───────────────────


    # ── Alpha Mode: multi-timeframe scanning ─────────────────────────────────




    # ── track record persistence ──────────────────────────────────────────────


    async def shutdown(self) -> None:
        self._save_track_record()
        self._executor.shutdown(wait=False)
        print('[LiveEngine] Shutdown complete.')
