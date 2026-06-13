#!/usr/bin/env python3
"""
live_engine.py — Aegis-1 Live Signal Engine  (Institutional-Grade Adaptive)
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
import sys
import time
import uuid
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

MODEL_STORE       = _ROOT / 'src' / 'ml' / 'model_store'
TRACK_RECORD_PATH = _ROOT / 'data' / 'track_record.json'
_PERF_STATE_PATH  = _ROOT / 'data' / 'perf_state.json'
_DRIFT_STATE_PATH = _ROOT / 'data' / 'drift_state.json'

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
    take_profit_1:   float = 0.0   # TP1: 1× ATR step from entry
    take_profit_2:   float = 0.0   # TP2: 2× ATR step — hard ceiling
    take_profit_3:   float = 0.0   # TP3: 3.5× ATR step — extended target


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
    signal_strength: str = ''


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
        is_bullish   = market_bias == 'BULLISH'
        is_bearish   = market_bias == 'BEARISH'
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

    MIN_QUALITY_SCORE = 55.0  # minimum points required to open a position

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
        meta_conf   = _f('meta_confidence', 0.0)
        rsi         = _f('rsi', 50.0)
        funding_bias= str(result.get('funding_bias', 'NEUTRAL') or 'NEUTRAL').upper()
        oi_trend    = str(result.get('oi_trend', 'STABLE') or 'STABLE').upper()
        market_bias = str(result.get('market_bias', 'NEUTRAL') or 'NEUTRAL').upper()

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

        # +10: meta model confidence is high
        if meta_conf > 0.75:
            score += 10; reasons.append(f'high_meta_conf({meta_conf:.3f})')

        # +10: RSI not in extreme exhaustion zone for the proposed direction
        rsi_ok = (
            (side == 'BUY'  and rsi < 75) or
            (side == 'SELL' and rsi > 25)
        )
        if rsi_ok:
            score += 10; reasons.append(f'rsi_ok({rsi:.1f})')
        else:
            reasons.append(f'rsi_extreme({rsi:.1f})')

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

        # ── Penalties ─────────────────────────────────────────────────────────

        # -20: no-trade zone
        if regime.regime == _REGIME_LIQUIDITY_TRAP:
            score -= 20; reasons.append('liquidity_trap_penalty')

        # -15: conflicting macro signals (weekly disagrees with daily)
        macro_daily  = _f('macro_daily', 0.0)
        macro_weekly = _f('macro_weekly', 0.0)
        macro_conflict = (
            (side == 'BUY'  and macro_daily < -0.15 and macro_weekly < -0.10) or
            (side == 'SELL' and macro_daily >  0.15 and macro_weekly >  0.10)
        )
        if macro_conflict:
            score -= 15; reasons.append('conflicting_macro')

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

            # Low-volume breakout: no real conviction behind the move
            if vol_zscore < 0.5:
                # Check if momentum diverges from the signal direction
                # (price breaking out but momentum indicators lagging/opposing)
                if side == 'BUY'  and conf_mom < 5.0 and conf_total < 6.0:
                    return True
                if side == 'SELL' and conf_mom > 5.0 and conf_total > 4.0:
                    return True

            # RSI divergence: strong signal but momentum is near extreme opposite
            if side == 'BUY'  and rsi > 82:
                return True
            if side == 'SELL' and rsi < 18:
                return True

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
    Volatility-aware position sizing and stop/take-profit calculation.
    All methods are pure functions (no state) — safe to call from async context.
    """

    BASE_POSITION_PCT = 0.10   # 10 % of balance as the base allocation
    MIN_POSITION_PCT  = 0.02   # floor: never risk less than 2 %
    MAX_POSITION_PCT  = 0.15   # ceiling: never risk more than 15 %

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
        price:              float,
        side:               str,    # 'BUY' | 'SELL'
        atr:                float,
        atr_mult:           float,
        quality_score:      float,
        lstm_vol_expansion: float = 0.5,   # P(ATR expansion) from LSTM  [0, 1]
        lstm_continuation:  float = 0.5,   # P(continuation)  from LSTM  [0, 1]
    ) -> Dict[str, float]:
        """
        Returns a dict with sl, tp1, tp2, tp3, trailing_trigger.

        ATR multiplier for stop loss
        ----------------------------
        - quality > 70  → 1.5× (tighter stop — higher conviction entry)
        - quality > 50  → 2.0×
        - quality ≤ 50  → 2.5× (wider stop — lower conviction, more room)

        TP levels use risk_distance (price to SL) as the unit:
        TP1 = 1.0 × risk   (quick partial)
        TP2 = 2.0 × risk   (main target)
        TP3 = 3.5 × risk   (extended runner)
        """
        if price <= 0 or atr <= 0:
            return {'sl': 0.0, 'tp1': 0.0, 'tp2': 0.0, 'tp3': 0.0, 'trailing_trigger': 0.0}

        if quality_score > 70:
            sl_mult = 1.5
        elif quality_score > 50:
            sl_mult = 2.0
        else:
            sl_mult = 2.5

        # LSTM vol-expansion adjustment: widen stop to survive the incoming
        # volatility spike without getting flushed by noise.
        if lstm_vol_expansion > 0.70:
            sl_mult *= 1.20   # +20 % ATR room for imminent expansion
        # LSTM high-continuation: tighten stop on confirmed momentum conviction.
        elif lstm_continuation > 0.72:
            sl_mult *= 0.90   # −10 % — move is clean, let price run tight

        risk_distance = sl_mult * atr

        if side == 'BUY':
            sl  = price - risk_distance
            tp1 = price + 1.0 * risk_distance
            tp2 = price + 2.0 * risk_distance
            tp3 = price + 3.5 * risk_distance
        else:  # SELL / SHORT
            sl  = price + risk_distance
            tp1 = price - 1.0 * risk_distance
            tp2 = price - 2.0 * risk_distance
            tp3 = price - 3.5 * risk_distance

        return {
            'sl':               round(sl,  8),
            'tp1':              round(tp1, 8),
            'tp2':              round(tp2, 8),
            'tp3':              round(tp3, 8),
            'trailing_trigger': round(tp1, 8),   # start trailing once TP1 is reached
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
    SAFE_MODE_LOSS = 5    # consecutive global losses to activate safe-mode
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
        When in safe-mode, low-quality signals (< 70) are blocked to protect capital.
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

    MAX_PER_CLUSTER    = 2     # max 2 concurrent open positions per correlation cluster
    MAX_OPEN_TOTAL     = 6     # hard cap: never more than 6 open at once across all tokens
    MAX_CAPITAL_PCT    = 0.45  # never deploy more than 45 % of balance simultaneously

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

    def sync_from_wallet(self, open_positions: Dict[str, Any]) -> None:
        """Sync open positions from the VirtualWallet on every scan cycle."""
        self._open_positions = {
            sym: pos.direction
            for sym, pos in open_positions.items()
        }

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

        # Capital deployment cap
        if wallet_balance > 0:
            deployed_pct = (self._total_open() * position_value) / wallet_balance
            if deployed_pct >= self.MAX_CAPITAL_PCT:
                return (False,
                        f'CAPITAL_CAP: {deployed_pct:.0%} already deployed '
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

    def __init__(self, initial_capital: float, max_position_usdt: float = 1_000.0):
        self.initial_capital   = initial_capital
        self.balance           = initial_capital
        self.max_position_usdt = max_position_usdt
        self.open_positions:   Dict[str, Position]  = {}
        self.trade_history:    List[TradeRecord]     = []
        self._load_history()

    def _load_history(self) -> None:
        """Restore closed trade history and balance from disk so restarts don't lose data."""
        if not TRACK_RECORD_PATH.exists():
            return
        try:
            with open(TRACK_RECORD_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            closed = [s for s in data.get('signals', [])
                      if s.get('outcome') in ('WIN', 'LOSS')]
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
                    )
                    self.trade_history.append(rec)
                    self.balance += float(s.get('pnl_usdt', 0))
                    restored += 1
                except Exception:
                    continue
            if restored:
                print(f'[VirtualWallet] Restored {restored} closed trades from disk. '
                      f'Balance: ${self.balance:,.2f}')
        except Exception as e:
            print(f'[VirtualWallet] History load error (starting fresh): {e}')

    def position_size(self) -> float:
        return min(self.balance * 0.10, self.max_position_usdt)

    def open_trade(self, pos: Position) -> None:
        self.open_positions[pos.symbol] = pos

    def close_trade(self, symbol: str, exit_price: float,
                    exit_reason: str) -> Optional[TradeRecord]:
        pos = self.open_positions.pop(symbol, None)
        if pos is None:
            return None

        if pos.direction == 'LONG':
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100
        else:
            pnl_pct = (pos.entry_price - exit_price) / pos.entry_price * 100

        pnl_usdt = round(pos.position_value * pnl_pct / 100, 2)
        self.balance += pnl_usdt

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
            position_value  = pos.position_value,
        )
        self.trade_history.append(rec)
        return rec

    @property
    def summary(self) -> Dict[str, Any]:
        won   = sum(1 for t in self.trade_history if t.outcome == 'WIN')
        lost  = sum(1 for t in self.trade_history if t.outcome == 'LOSS')
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
    1. Predictor.predict_realtime() → dict with fire/side/meta_confidence/price/atr
    2. MarketRegimeDetector.detect()  → RegimeState
    3. SignalQualityFilter.score_signal() → (quality_score, reasons)
    4. If quality_score < 55 or fake breakout detected → block entry
    5. If fire=True and no open position → open paper trade (VirtualWallet)
    6. If fire=True and opposite position → MODEL_REVERSAL_TP exit, then re-enter
    7. If price hits ATR stop → STOP_HIT exit
    8. After TP1 hit → activate trailing stop at 0.5× ATR below peak (LONG)
    9. After every cycle → write data/track_record.json
    """

    MAX_CONCURRENT        = 8
    HOURS_CONTEXT         = 300
    MIN_HOLD_SECONDS      = 7_200    # 2 h minimum hold before model-reversal exit
    COOLDOWN_SECONDS      = 14_400   # 4 h post-close cooldown
    FLIP_COOLDOWN_SECONDS = 28_800   # 8 h extra cooldown when new signal flips direction
    MAX_HOLD_SECONDS      = 172_800  # 48 h zombie guard
    CONFLUENCE_BUY_MIN    = 5.5
    CONFLUENCE_SELL_MAX   = 4.5

    # Regimes where entry is unconditionally blocked
    NO_TRADE_REGIMES: set = {_REGIME_LIQUIDITY_TRAP}

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
        self.wallet    = VirtualWallet(capital, max_position_usdt)
        self._executor = ThreadPoolExecutor(
            max_workers=self.MAX_CONCURRENT, thread_name_prefix='aegis_pred')

        self.predictors:   Dict[str, Any]   = {}
        self.last_signals: Dict[str, Any]   = {}
        self.live_prices:  Dict[str, float] = {}

        self._open_time:        Dict[str, float] = {}
        self._last_close_time:  Dict[str, float] = {}
        self._last_close_side:  Dict[str, str]   = {}

        # Trailing stop tracking
        self._tp1_hit:    Dict[str, bool]  = {}
        self._peak_price: Dict[str, float] = {}   # highest (LONG) or lowest (SHORT) seen since entry

        self.bootstrap_done  = 0
        self.bootstrap_total = len(token_configs)

        # Adaptive intelligence modules
        self.regime_detector = MarketRegimeDetector()
        self.quality_filter  = SignalQualityFilter()
        self.risk_engine     = DynamicRiskEngine()
        self.perf_tracker    = PerformanceTracker()
        self.drift_monitor   = DriftMonitor()
        self.portfolio_guard = PortfolioGuard()

        # Restore persisted state so protection survives server restarts
        self.perf_tracker.load_state()
        self.drift_monitor.load_state()
        self.drift_monitor.load_benchmarks()

        self._load_predictors([c.symbol for c in token_configs])

    # ── initialisation ────────────────────────────────────────────────────────

    def _load_predictors(self, symbols: List[str]) -> None:
        from src.ml.predictor import Predictor
        loaded = tradeable = 0
        for sym in symbols:
            try:
                p = Predictor(sym)
                if p.model is not None:
                    self.predictors[sym] = p
                    loaded += 1
                    if p.meta.get('tradeable', False):
                        tradeable += 1
            except Exception:
                pass
        self.bootstrap_total = max(loaded, 1)
        print(f'[LiveEngine] {loaded} predictors loaded '
              f'({tradeable} tradeable + {loaded - tradeable} monitor-only) '
              f'from {len(symbols)} configured symbols.')

    # ── main loop ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        print(f'[LiveEngine] Starting — interval={self.scan_interval_seconds}s '
              f'symbols={len(self.predictors)}')
        asyncio.create_task(self._ws_price_ticker())
        while True:
            t0 = time.time()
            await self._scan_all()
            self._save_track_record()
            sleep = max(0.0, self.scan_interval_seconds - (time.time() - t0))
            await asyncio.sleep(sleep)

    async def _ws_price_ticker(self) -> None:
        """
        Real-time price feed via Binance all-market mini-ticker WebSocket stream.
        Reconnects automatically on any error.
        """
        import json as _json

        all_syms = list(set(list(self.predictors.keys()) + list(self._INDEX_SYMBOLS)))
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
        await self._fetch_index_prices()
        self.bootstrap_done = len(self.predictors)

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

            price = float(result.get('price', 0) or 0)
            if price > 0:
                self.live_prices[symbol] = price
            else:
                price = self.live_prices.get(symbol, 0)

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
            quality_score = 0.0
            quality_reasons: List[str] = []
            fake_breakout = False

            if result.get('fire') and new_side in ('BUY', 'SELL'):
                quality_score, quality_reasons = self.quality_filter.score_signal(
                    result, regime, new_side)
                fake_breakout = self.quality_filter.is_fake_breakout(result, new_side)

                # HMM quality bonus/penalty: ±10 points based on regime alignment
                if _hmm_available:
                    if _hmm_regime in ('TRENDING_BULL',) and new_side == 'BUY':
                        quality_score += 10.0
                        quality_reasons.append('hmm_trend_bull_bonus')
                    elif _hmm_regime in ('TRENDING_BEAR',) and new_side == 'SELL':
                        quality_score += 10.0
                        quality_reasons.append('hmm_trend_bear_bonus')
                    elif _hmm_regime in ('CHOPPY', 'VOLATILE_EXPANSION'):
                        quality_score -= 10.0
                        quality_reasons.append(f'hmm_regime_penalty({_hmm_regime})')
                    elif _hmm_regime == 'DISTRIBUTION' and new_side == 'BUY':
                        quality_score -= 8.0
                        quality_reasons.append('hmm_distribution_buy_penalty')
                    elif _hmm_regime == 'ACCUMULATION' and new_side == 'SELL':
                        quality_score -= 8.0
                        quality_reasons.append('hmm_accumulation_sell_penalty')
                    # High transition risk: penalise regime instability
                    if _hmm_trans_risk > 0.35:
                        quality_score -= 5.0
                        quality_reasons.append(f'hmm_transition_risk({_hmm_trans_risk:.0%})')
                    quality_score = round(max(0.0, min(quality_score, 100.0)), 1)

            # Build signal entry with enriched fields
            self.last_signals[symbol] = self._build_signal_entry(
                symbol, result, price, regime=regime,
                quality_score=quality_score, fake_breakout=fake_breakout)

            existing = self.wallet.open_positions.get(symbol)
            if existing:
                self._manage_exit(symbol, existing, result, price)
            elif result.get('fire') and result.get('tradeable', False) and price > 0:
                now               = time.time()
                cooldown_elapsed  = now - self._last_close_time.get(symbol, 0)
                last_side         = self._last_close_side.get(symbol, '')
                is_flip           = (last_side != '' and last_side != new_side)
                required_cooldown = (
                    self.FLIP_COOLDOWN_SECONDS if is_flip else self.COOLDOWN_SECONDS
                )

                if cooldown_elapsed >= required_cooldown:
                    # ── Quality gate ──────────────────────────────────────────
                    if regime.regime in self.NO_TRADE_REGIMES:
                        print(f'[{symbol}] NO_TRADE_REGIME={regime.regime} — blocked')
                        if symbol in self.last_signals:
                            self.last_signals[symbol]['fire']            = False
                            self.last_signals[symbol]['signal']          = 'HOLD'
                            self.last_signals[symbol]['regime_blocked']  = True
                        self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                        return

                    if fake_breakout:
                        print(f'[{symbol}] FAKE_BREAKOUT blocked {new_side} '
                              f'quality={quality_score:.0f}')
                        if symbol in self.last_signals:
                            self.last_signals[symbol]['fire']                 = False
                            self.last_signals[symbol]['signal']               = 'HOLD'
                            self.last_signals[symbol]['fake_breakout_blocked'] = True
                        self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                        return

                    if quality_score < self.quality_filter.MIN_QUALITY_SCORE:
                        print(f'[{symbol}] QUALITY_BLOCKED {new_side} '
                              f'score={quality_score:.0f}/100 '
                              f'(min={self.quality_filter.MIN_QUALITY_SCORE}) '
                              f'reasons={quality_reasons}')
                        if symbol in self.last_signals:
                            self.last_signals[symbol]['fire']              = False
                            self.last_signals[symbol]['signal']            = 'HOLD'
                            self.last_signals[symbol]['confluence_blocked'] = True
                        self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                        return

                    # Self-healing safe-mode: block low-quality signals when on a loss streak
                    if (self.perf_tracker.safe_mode_active() and
                            quality_score < 70):
                        print(f'[{symbol}] SAFE_MODE blocked {new_side} '
                              f'score={quality_score:.0f} (need ≥70 in safe-mode)')
                        if symbol in self.last_signals:
                            self.last_signals[symbol]['fire']          = False
                            self.last_signals[symbol]['signal']        = 'HOLD'
                            self.last_signals[symbol]['safe_mode_blocked'] = True
                        self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                        return

                    # ── Drift gate: block symbol if model is critically degraded ─
                    if self.drift_monitor.is_blocked(symbol):
                        drift_penalty = self.drift_monitor.confidence_penalty(symbol)
                        live_wr       = self.drift_monitor._live_win_rate(symbol)
                        benchmark     = self.drift_monitor._benchmarks.get(symbol, 0.60)
                        print(f'[{symbol}] DRIFT_CRITICAL blocked {new_side} — '
                              f'live_wr={live_wr:.1%} benchmark={benchmark:.1%} '
                              f'(>{self.drift_monitor.CRITICAL_DROP_PP}pp below benchmark)')
                        if symbol in self.last_signals:
                            self.last_signals[symbol]['fire']          = False
                            self.last_signals[symbol]['signal']        = 'HOLD'
                            self.last_signals[symbol]['drift_blocked'] = True
                        self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                        return

                    # Apply HMM regime confidence adjustment to quality gate
                    # _hmm_conf_adj is a signed float (negative = raise threshold)
                    # We convert it to a quality-score equivalent for the gate check.
                    _hmm_quality_penalty = round(-_hmm_conf_adj * 100, 1)  # e.g. +0.05 adj → -5 quality
                    _effective_min_quality = self.quality_filter.MIN_QUALITY_SCORE + _hmm_quality_penalty
                    if quality_score < _effective_min_quality:
                        print(f'[{symbol}] HMM_QUALITY_GATE {new_side} '
                              f'score={quality_score:.0f} < '
                              f'min={_effective_min_quality:.0f} '
                              f'(hmm_regime={_hmm_regime})')
                        if symbol in self.last_signals:
                            self.last_signals[symbol]['fire']          = False
                            self.last_signals[symbol]['signal']        = 'HOLD'
                            self.last_signals[symbol]['hmm_blocked']   = True
                        self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                        return

                    # Apply drift confidence penalty (WARNING level) to quality gate
                    drift_penalty = self.drift_monitor.confidence_penalty(symbol)
                    if drift_penalty > 0 and quality_score < (self.quality_filter.MIN_QUALITY_SCORE + drift_penalty * 100):
                        print(f'[{symbol}] DRIFT_WARNING quality penalty {drift_penalty:.2f} '
                              f'— adjusted min={self.quality_filter.MIN_QUALITY_SCORE + drift_penalty * 100:.0f} '
                              f'score={quality_score:.0f}')
                        if symbol in self.last_signals:
                            self.last_signals[symbol]['fire']          = False
                            self.last_signals[symbol]['signal']        = 'HOLD'
                            self.last_signals[symbol]['drift_blocked'] = True
                        self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                        return

                    # ── Portfolio guard: correlation and capital limits ────────────
                    self.portfolio_guard.sync_from_wallet(self.wallet.open_positions)
                    # Estimate position value for this signal
                    _atr_pct_est = float(result.get('atr_pct', 1.5) or 1.5)
                    _pos_est     = self.risk_engine.calculate_position_size(
                        self.wallet.balance, quality_score, regime, _atr_pct_est)
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

                    # ── Legacy confluence gate (keep existing dual-path logic) ─
                    _conf_data  = result.get('confluence') or {}
                    _conf_total = float(_conf_data.get('total', 5.0))
                    _conf_trend = float(_conf_data.get('trend', 5.0))
                    _conf_mom   = float(_conf_data.get('momentum', 5.0))

                    def _c10(v: float) -> float:
                        if abs(v) <= 1.05:
                            return (v + 1.0) / 2.0 * 10.0
                        return min(10.0, max(0.0, v))

                    _ct  = _c10(_conf_total)
                    _ctr = _c10(_conf_trend)
                    _cm  = _c10(_conf_mom)

                    _pass_total = (
                        (new_side == 'BUY'  and _ct >= self.CONFLUENCE_BUY_MIN) or
                        (new_side == 'SELL' and _ct <= self.CONFLUENCE_SELL_MAX)
                    )
                    _trend_mom_avg = (_ctr * 2.0 + _cm * 1.5) / 3.5
                    _pass_override = (
                        (new_side == 'BUY'  and _trend_mom_avg >= 6.5 and _ct >= 3.5) or
                        (new_side == 'SELL' and _trend_mom_avg <= 3.5 and _ct <= 6.5)
                    )
                    _conf_ok = _pass_total or _pass_override

                    if _conf_ok:
                        reason = 'override(trend+mom)' if _pass_override and not _pass_total else 'total'
                        print(f'[{symbol}] CONF PASS {new_side} '
                              f'total={_ct:.1f} trend={_ctr:.1f} mom={_cm:.1f} [{reason}] '
                              f'quality={quality_score:.0f} regime={regime.regime}')
                        self._open_position(symbol, result, price, regime, quality_score)
                    else:
                        print(f'[{symbol}] CONF BLOCKED {new_side} '
                              f'total={_ct:.1f}/10 trend={_ctr:.1f} mom={_cm:.1f} '
                              f'(need total BUY≥{self.CONFLUENCE_BUY_MIN} '
                              f'or trend+mom override)')
                        if symbol in self.last_signals:
                            self.last_signals[symbol]['fire']              = False
                            self.last_signals[symbol]['signal']            = 'HOLD'
                            self.last_signals[symbol]['confluence_blocked'] = True

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

    # ── trade management ──────────────────────────────────────────────────────

    def _manage_exit(self, symbol: str, pos: Position,
                     result: Dict[str, Any], price: float) -> None:
        live_px     = self.live_prices.get(symbol, 0.0)
        check_price = live_px if live_px > 0 else price

        now  = time.time()
        held = now - self._open_time.get(symbol, 0)
        atr  = float(result.get('atr', pos.entry_price * 0.015) or pos.entry_price * 0.015)

        # ── Update peak price for trailing stop tracking ──────────────────────
        if pos.direction == 'LONG':
            current_peak = self._peak_price.get(symbol, pos.entry_price)
            self._peak_price[symbol] = max(current_peak, check_price)
        else:   # SHORT: track trough (lowest point since entry)
            current_trough = self._peak_price.get(symbol, pos.entry_price)
            self._peak_price[symbol] = min(current_trough, check_price)

        def _close(reason: str) -> None:
            rec = self.wallet.close_trade(symbol, check_price, reason)
            if rec:
                self._last_close_time[symbol] = now
                self._last_close_side[symbol] = pos.side
                # Clean up trailing stop state
                self._tp1_hit.pop(symbol, None)
                self._peak_price.pop(symbol, None)
                tag = 'WIN' if rec.pnl_pct > 0 else 'LOSS'
                print(f'[{symbol}] {reason} {tag} {rec.pnl_pct:+.2f}% @ {check_price:.6g}')
                # Record outcome in performance tracker (persists to disk inside)
                self.perf_tracker.record_outcome(
                    symbol        = symbol,
                    regime        = self.last_signals.get(symbol, {}).get('regime', 'UNKNOWN'),
                    outcome       = rec.outcome,
                    pnl_pct       = rec.pnl_pct,
                    quality_score = float(self.last_signals.get(symbol, {}).get('quality_score', 0)),
                )
                # Feed drift monitor so it can compare against training benchmark
                self.drift_monitor.record(symbol, rec.outcome)
                self.drift_monitor.save_state()

                drift_sev = self.drift_monitor.severity(symbol)
                if drift_sev in ('WARNING', 'CRITICAL'):
                    live_wr    = self.drift_monitor._live_win_rate(symbol)
                    benchmark  = self.drift_monitor._benchmarks.get(symbol, 0.60)
                    print(f'[{symbol}] DRIFT {drift_sev}: '
                          f'live_wr={live_wr:.1%} vs benchmark={benchmark:.1%} '
                          f'(drop={((benchmark - (live_wr or 0)) * 100):.1f}pp)')

                # Persist immediately so the outcome survives a crash
                self._save_track_record()

        # ── 1. Maximum hold time (zombie guard) ──────────────────────────────
        if held >= self.MAX_HOLD_SECONDS:
            _close('MAX_HOLD_EXPIRED')
            return

        # ── 2. Trailing stop (active after TP1 is hit) ────────────────────────
        if self._tp1_hit.get(symbol, False):
            trail_atr = 0.5 * atr
            peak      = self._peak_price.get(symbol, pos.entry_price)
            if pos.direction == 'LONG':
                trail_stop = peak - trail_atr
                if check_price <= trail_stop:
                    _close('TRAILING_STOP')
                    return
            else:  # SHORT
                trail_stop = peak + trail_atr
                if check_price >= trail_stop:
                    _close('TRAILING_STOP')
                    return

        # ── 3. TP1 hit — activate trailing stop from here ─────────────────────
        if pos.take_profit_1 > 0 and not self._tp1_hit.get(symbol, False):
            tp1_hit = (
                (pos.direction == 'LONG'  and check_price >= pos.take_profit_1) or
                (pos.direction == 'SHORT' and check_price <= pos.take_profit_1)
            )
            if tp1_hit:
                self._tp1_hit[symbol]    = True
                self._peak_price[symbol] = check_price   # reset peak to TP1 price
                print(f'[{symbol}] TP1_HIT @ {check_price:.6g} — trailing stop activated')
                # Do NOT close here — let the trailing stop manage the rest
                # (matches the design intent: partial exit via trailing)

        # ── 4. TP2 hit — hard ceiling ─────────────────────────────────────────
        if pos.take_profit_2 > 0:
            tp2_hit = (
                (pos.direction == 'LONG'  and check_price >= pos.take_profit_2) or
                (pos.direction == 'SHORT' and check_price <= pos.take_profit_2)
            )
            if tp2_hit:
                _close('TP2_HIT')
                return

        # ── 5. Model-reversal TP (dynamic exit) ──────────────────────────────
        side = result.get('side', 'FLAT')
        fire = bool(result.get('fire', False))
        opposite = (
            (pos.direction == 'LONG'  and side == 'SELL' and fire) or
            (pos.direction == 'SHORT' and side == 'BUY'  and fire)
        )
        if opposite and held >= self.MIN_HOLD_SECONDS:
            _close('MODEL_REVERSAL_TP')
            return

        # ── 6. ATR-based stop loss ────────────────────────────────────────────
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
    ) -> None:
        side = result.get('side', 'FLAT')
        if side not in ('BUY', 'SELL'):
            return

        direction = 'LONG' if side == 'BUY' else 'SHORT'
        meta_conf = float(result.get('meta_confidence', 0))
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

        # ── Dynamic stop/TP calculation ───────────────────────────────────────
        if regime is not None and quality_score > 0:
            stops = self.risk_engine.calculate_stops(
                price              = price,
                side               = side,
                atr                = atr,
                atr_mult           = atr_mult,
                quality_score      = quality_score,
                lstm_vol_expansion = float(result.get('lstm_vol_expansion_prob', 0.5)),
                lstm_continuation  = float(result.get('lstm_continuation_prob',  0.5)),
            )
            stop_loss = stops['sl']
            tp1       = stops['tp1']
            tp2       = stops['tp2']
            tp3       = stops['tp3']
        else:
            step      = atr_mult * atr
            stop_loss = (price - step) if direction == 'LONG' else (price + step)
            if direction == 'LONG':
                tp1 = float(result.get('bull_tp1') or round(price + 1.0 * step, 8))
                tp2 = float(result.get('bull_tp2') or round(price + 2.0 * step, 8))
                tp3 = float(result.get('bull_tp3') or round(price + 3.5 * step, 8))
            else:
                tp1 = float(result.get('bear_tp1') or round(price - 1.0 * step, 8))
                tp2 = float(result.get('bear_tp2') or round(price - 2.0 * step, 8))
                tp3 = float(result.get('bear_tp3') or round(price - 3.5 * step, 8))

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
            atr_multiplier  = atr_mult,
            take_profit_1   = round(tp1, 8),
            take_profit_2   = round(tp2, 8),
            take_profit_3   = round(tp3, 8),
        )
        self.wallet.open_trade(pos)
        self._open_time[symbol]    = time.time()
        self._tp1_hit[symbol]      = False
        self._peak_price[symbol]   = price

        regime_label = regime.regime if regime else 'UNKNOWN'
        print(f'[{symbol}] OPEN {direction} @ {price} | '
              f'conf={meta_conf:.3f} quality={quality_score:.0f} '
              f'regime={regime_label} '
              f'SL={stop_loss:.6g} TP1={tp1:.6g} TP2={tp2:.6g} '
              f'size={pos_value:.0f} USDT')
        # Persist the new open position immediately
        self._save_track_record()

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
        conf     = float(result.get('meta_confidence', 0))
        thr      = float(result.get('threshold', 0.6))
        fire     = bool(result.get('fire', False))
        atr      = float(result.get('atr', price * 0.015))
        atr_mult = float(result.get('atr_multiplier', 1.5))
        atr_pct  = float(result.get('atr_pct', atr / price * 100 if price > 0 else 1.5))

        if not fire:
            strength = 'NEUTRAL'
        elif conf >= thr * 1.15:
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

        # TP/SL levels for the active direction
        step = atr_mult * atr
        if side == 'BUY':
            entry['suggested_tp'] = result.get('bull_tp1', round(price + step, 8))
            entry['suggested_sl'] = round(price - step, 8)
        elif side == 'SELL':
            entry['suggested_tp'] = result.get('bear_tp1', round(price - step, 8))
            entry['suggested_sl'] = round(price + step, 8)
        else:
            entry['suggested_tp'] = None
            entry['suggested_sl'] = None

        # Expected move projection
        _conf_data  = result.get('confluence') or {}
        _conf_total = float(_conf_data.get('total', 5.0))
        _conf_raw   = abs(_conf_total - 5.0) / 5.0
        entry['expected_move_pct'] = round(_conf_raw * atr_pct * 3.0, 2)

        # Risk/reward ratio (TP1 vs SL)
        sl_val = entry.get('suggested_sl', 0)
        tp_val = entry.get('suggested_tp', 0)
        if price > 0 and sl_val and tp_val:
            risk   = abs(price - sl_val)
            reward = abs(price - tp_val)
            entry['risk_reward'] = round(reward / risk, 2) if risk > 0 else 0
        else:
            entry['risk_reward'] = 0

        # Forward all market context fields from predictor
        _CONTEXT_KEYS = (
            'market_bias', 'bias_strength', 'trend_regime', 'volatility_regime',
            'atr_pct', 'support', 'resistance', 'pivot',
            'r1', 'r2', 's1', 's2',
            'bull_tp1', 'bull_tp2', 'bull_tp3',
            'bear_tp1', 'bear_tp2', 'bear_tp3',
            'confluence',
            'rsi', 'macd_signal', 'cci', 'adx', 'supertrend',
            'macro_daily', 'macro_weekly',
            'volume_strength', 'volume_zscore',
            'funding_rate', 'funding_bias', 'oi_trend', 'oi_change_1h_pct', 'oi_zscore',
            'session', 'session_note', 'fear_greed',
            'scalper_view', 'day_trader_view', 'swing_view',
            # HMM regime intelligence fields
            'hmm_regime', 'hmm_confidence', 'hmm_state_id',
            'hmm_transition_risk', 'hmm_stability', 'hmm_available',
            'hmm_conf_adjustment', 'hmm_atr_mult', 'hmm_position_scale',
            'hmm_transition_warning',
            # LSTM temporal intelligence fields
            'lstm_continuation_prob', 'lstm_vol_expansion_prob',
            'lstm_exhaustion_prob', 'lstm_available',
        )
        for k in _CONTEXT_KEYS:
            if k in result:
                entry[k] = result[k]

        return entry

    # ── track record persistence ──────────────────────────────────────────────

    def _save_track_record(self) -> None:
        try:
            TRACK_RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)

            open_records = [
                {
                    'signal_id':       p.signal_id,
                    'symbol':          p.symbol,
                    'direction':       p.direction,
                    'side':            p.side,
                    'entry_price':     p.entry_price,
                    'exit_price':      None,
                    'entry_time':      p.entry_time,
                    'close_time':      None,
                    'pnl_pct':         0.0,
                    'pnl_usdt':        0.0,
                    'outcome':         'OPEN',
                    'exit_reason':     None,
                    'meta_confidence': p.meta_confidence,
                    'position_value':  p.position_value,
                    'signal_strength': '',
                    'stop_loss':       p.stop_loss,
                    'take_profit_1':   p.take_profit_1,
                    'take_profit_2':   p.take_profit_2,
                    'take_profit_3':   p.take_profit_3,
                }
                for p in self.wallet.open_positions.values()
            ]

            all_records = sorted(
                [asdict(t) for t in self.wallet.trade_history] + open_records,
                key=lambda r: r.get('entry_time') or '',
                reverse=True,
            )[:500]

            self.portfolio_guard.sync_from_wallet(self.wallet.open_positions)
            payload: Dict[str, Any] = {
                'generated_at':      datetime.now(timezone.utc).isoformat(),
                'summary':           self.wallet.summary,
                'signals':           all_records,
                'performance':       self.perf_tracker.get_performance_summary(),
                'drift':             self.drift_monitor.get_summary(),
                'portfolio':         self.portfolio_guard.get_summary(),
            }

            import os as _os
            tmp = TRACK_RECORD_PATH.with_suffix('.tmp')
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2, default=str)
            _os.replace(tmp, TRACK_RECORD_PATH)
        except Exception as e:
            print(f'[LiveEngine] track_record save failed: {e}')

    async def shutdown(self) -> None:
        self._save_track_record()
        self._executor.shutdown(wait=False)
        print('[LiveEngine] Shutdown complete.')


# =============================================================================
# Setup helpers
# =============================================================================

def automated_setup(_: Path, args: Any):
    """
    Scan MODEL_STORE for all available symbols (up to 32).
    Tradeable symbols are loaded first; non-tradeable models fill the remainder.
    """
    tradeable_configs:     List[TokenConfig] = []
    non_tradeable_configs: List[TokenConfig] = []

    if MODEL_STORE.exists():
        meta_files = sorted(MODEL_STORE.glob('*_meta.json'),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        seen: set = set()
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

    TARGET    = 32
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
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text
    from rich.layout import Layout
    from rich import box

    console = Console()

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

    def _bias_cell(bias: str) -> str:
        if bias == 'BULLISH':  return '[green]BULL[/]'
        if bias == 'BEARISH':  return '[red]BEAR[/]'
        return '[dim]NEUT[/]'

    def _signal_cell(sig: dict) -> str:
        side     = sig.get('signal', 'FLAT')
        fire     = sig.get('fire', False)
        strength = sig.get('signal_strength', '')
        if not fire or side in ('FLAT', 'HOLD'):
            return '[dim]·[/]'
        if 'STRONG' in strength:
            return '[bold green]🔥 STRONG BUY[/]' if side == 'BUY' \
              else '[bold red]🔥 STRONG SELL[/]'
        return '[green]BUY[/]' if side == 'BUY' else '[red]SELL[/]'

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

    def _build_layout() -> Layout:
        wallet  = engine.wallet
        live_px = engine.live_prices
        signals = engine.last_signals

        pnl_u   = round(wallet.balance - wallet.initial_capital, 2)
        pnl_pct = round(pnl_u / wallet.initial_capital * 100, 2)
        pc      = 'green' if pnl_u >= 0 else 'red'
        s       = wallet.summary
        wr      = round(s['win_rate'] * 100, 1) if s['total_trades'] else 0.0
        warmup  = engine.bootstrap_done >= engine.bootstrap_total
        status  = '[bold green]● LIVE[/]' if warmup \
                  else f'[yellow]⏳ WARMUP {engine.bootstrap_done}/{engine.bootstrap_total}[/]'
        now_utc = datetime.now(timezone.utc).strftime('%Y-%m-%d  %H:%M:%S UTC')
        perf    = engine.perf_tracker.get_performance_summary()
        safe_tag = '[bold red] ⚠ SAFE-MODE[/]' if perf['safe_mode'] else ''

        header = Panel(
            f"  {status}{safe_tag}   │   "
            f"Capital [bold cyan]${wallet.initial_capital:,.0f}[/]   │   "
            f"Balance [bold cyan]${wallet.balance:,.2f}[/]   │   "
            f"PnL [{pc}]{pnl_u:+.2f} USDT  ({pnl_pct:+.2f}%)[/]   │   "
            f"Closed [white]{s['total_trades']}[/] "
            f"([green]{s['won']}W[/]/[red]{s['lost']}L[/])   │   "
            f"Win-Rate [bold]{wr:.1f}%[/]   │   "
            f"Open [bold cyan]{len(wallet.open_positions)}[/]   │   "
            f"[dim]{now_utc}[/]",
            title="[bold]  AEGIS-1   Institutional-Grade Adaptive Signal Engine  [/]",
            border_style='cyan',
        )

        grid = Table(
            title=f'[bold]TOKEN STATUS  ({len(signals)} symbols)[/]',
            box=box.SIMPLE_HEAVY,
            border_style='bright_black',
            show_header=True,
            header_style='bold white',
            expand=True,
        )
        grid.add_column('#',        justify='right',  width=3,   style='dim')
        grid.add_column('Symbol',   justify='left',   min_width=12)
        grid.add_column('Price',    justify='right',  min_width=10)
        grid.add_column('Signal',   justify='center', min_width=12)
        grid.add_column('Bias',     justify='center', width=6)
        grid.add_column('Regime',   justify='center', width=7)
        grid.add_column('Quality',  justify='right',  width=7)
        grid.add_column('RSI',      justify='right',  width=6)
        grid.add_column('Dir%',     justify='right',  width=6)
        grid.add_column('Funding',  justify='right',  width=8)
        grid.add_column('Session',  justify='center', width=10)
        grid.add_column('Position', justify='center', min_width=14)

        tradeable_syms = {
            sym for sym, pred in engine.predictors.items()
            if getattr(pred, 'meta', {}).get('tradeable', False)
        }

        for idx, (sym, sig) in enumerate(sorted(signals.items()), 1):
            price   = float(live_px.get(sym, sig.get('price', 0) or 0))
            rsi     = sig.get('rsi', None)
            bias    = sig.get('market_bias', '')
            session = sig.get('session', '')[:8]
            funding = sig.get('funding_rate', None)
            regime  = sig.get('regime', 'UNKNOWN')
            quality = float(sig.get('quality_score', 0))
            is_tradeable = sym in tradeable_syms
            _pred_meta = engine.predictors[sym].meta if sym in engine.predictors else {}
            dir_prec = float(_pred_meta.get('holdout_trading', {}).get('directional_precision', 0))

            pos = wallet.open_positions.get(sym)
            if pos:
                cur = price or pos.entry_price
                if pos.direction == 'LONG':
                    ppct = (cur - pos.entry_price) / pos.entry_price * 100
                else:
                    ppct = (pos.entry_price - cur) / pos.entry_price * 100
                arrow, ds = _dir(pos.direction)
                # Show trailing indicator if TP1 has been hit
                trail_flag = '[bold yellow] T[/]' if engine._tp1_hit.get(sym) else ''
                pos_cell   = f'[{ds}]{arrow}[/] [{_pc(ppct)}]{ppct:+.2f}%[/]{trail_flag}'
            else:
                pos_cell = '[dim]—[/]'

            if rsi is None:
                rsi_cell = '[dim]—[/]'
            else:
                rsi_f = float(rsi)
                if rsi_f >= 70:   rsi_cell = f'[red]{rsi_f:.0f}[/]'
                elif rsi_f <= 30: rsi_cell = f'[green]{rsi_f:.0f}[/]'
                else:             rsi_cell = f'[white]{rsi_f:.0f}[/]'

            if funding is None:
                fund_cell = '[dim]—[/]'
            else:
                ff    = float(funding)
                col_f = 'red' if ff > 0.01 else ('green' if ff < -0.01 else 'dim white')
                fund_cell = f'[{col_f}]{ff:+.4f}%[/]'

            sym_cell  = f'[bold]{sym}[/]'  if is_tradeable else f'[dim]{sym}[/]'
            px_cell   = f'[bold white]{_px(price)}[/]' if is_tradeable else f'[dim]{_px(price)}[/]'
            if dir_prec > 0:
                dp_pct  = dir_prec * 100
                dp_col  = 'green' if dp_pct >= 70 else ('yellow' if dp_pct >= 60 else 'red')
                dir_cell = f'[{dp_col}]{dp_pct:.0f}%[/]' if is_tradeable else f'[dim]{dp_pct:.0f}%[/]'
            else:
                dir_cell = '[dim]—[/]'

            grid.add_row(
                f'[dim]{idx}[/]' if not is_tradeable else str(idx),
                sym_cell,
                px_cell,
                _signal_cell(sig) if is_tradeable else '[dim]watch[/]',
                _bias_cell(bias),
                _regime_short(regime),
                _quality_cell(quality) if is_tradeable else '[dim]—[/]',
                rsi_cell,
                dir_cell,
                fund_cell,
                f'[dim]{session}[/]',
                pos_cell,
            )

        open_lines: List[str] = []
        for pos in sorted(wallet.open_positions.values(), key=lambda p: p.entry_time):
            cur  = float(live_px.get(pos.symbol, pos.entry_price) or pos.entry_price)
            ppct = ((cur - pos.entry_price) / pos.entry_price * 100) if pos.direction == 'LONG' \
                   else ((pos.entry_price - cur) / pos.entry_price * 100)
            pu   = round(pos.position_value * ppct / 100, 2)
            arr, ds = _dir(pos.direction)
            col  = _pc(ppct)
            trail_tag = ' [T]' if engine._tp1_hit.get(pos.symbol) else ''
            open_lines.append(
                f"  [{ds}]{arr} {pos.symbol}[/]  "
                f"entry [white]{_px(pos.entry_price)}[/] → "
                f"now [bold white]{_px(cur)}[/]  "
                f"[{col}]{ppct:+.2f}%  {pu:+.2f} USDT[/]  "
                f"SL [dim]{_px(pos.stop_loss)}[/]  "
                f"conf [cyan]{pos.meta_confidence:.3f}[/]"
                f"{trail_tag}  "
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

        layout = Layout()
        layout.split_column(
            Layout(header, name='hdr',    size=3),
            Layout(grid,   name='grid',   ratio=1),
            Layout(footer, name='footer', size=max(4 + len(open_lines) + len(closed_lines), 6)),
        )
        return layout

    async def _run_with_display() -> None:
        scan_task = asyncio.create_task(engine.run())
        with Live(_build_layout(), console=console,
                  refresh_per_second=0.5, screen=True) as live:
            try:
                while not scan_task.done():
                    live.update(_build_layout())
                    await asyncio.sleep(2)
            except asyncio.CancelledError:
                pass
            finally:
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
