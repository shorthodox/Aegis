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
TRACK_RECORD_PATH       = _ROOT / 'data' / 'track_record.json'
ALPHA_TRACK_RECORD_PATH = _ROOT / 'data' / 'alpha_track_record.json'
_ALPHA_TIMEFRAMES       = ['15m', '30m', '4h', '1d']
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


# Shared futures exchange for multi-timeframe OHLCV candle fetches.
# Binance USDM perpetuals — same symbols as the main trading fleet.
_usdm_ex      = None
_usdm_ex_lock = __import__('threading').Lock()

def _fetch_ohlcv_sync(symbol: str, timeframe: str, limit: int) -> list:
    """Thread-safe OHLCV fetch from Binance USDM perpetuals. Returns [] on error."""
    global _usdm_ex
    try:
        import ccxt as _ccxt
        if _usdm_ex is None:
            _new = _ccxt.binanceusdm({'enableRateLimit': True, 'timeout': 8000})  # type: ignore[arg-type]
            with _usdm_ex_lock:
                if _usdm_ex is None:
                    _usdm_ex = _new
        with _usdm_ex_lock:
            return _usdm_ex.fetch_ohlcv(symbol, timeframe, limit=limit) or []
    except Exception:
        return []


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
    take_profit_1:   float = 0.0   # TP1: 1.0× ATR from entry — 20% partial close
    take_profit_2:   float = 0.0   # TP2: 1.8× ATR from entry — 20% partial close
    take_profit_3:   float = 0.0   # TP3: 2.7× ATR from entry — 20% partial close
    take_profit_4:   float = 0.0   # TP4: 3.8× ATR from entry — 20% partial close
    take_profit_5:   float = 0.0   # TP5: 5.2× ATR from entry — close remainder
    signal_strength: str   = ''    # risk tier at entry: STRONG | NORMAL | RISKY


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

    MIN_QUALITY_SCORE = 70.0  # minimum points required to open a position

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

        # ── Penalties ─────────────────────────────────────────────────────────

        # -20: no-trade zone
        if regime.regime == _REGIME_LIQUIDITY_TRAP:
            score -= 20; reasons.append('liquidity_trap_penalty')

        # -15: ranging market — no directional edge, signals are noise
        if regime.regime == _REGIME_RANGING:
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
            if side == 'BUY':
                if _w_bull and _d_bull:
                    score += 15; reasons.append('htf_both_bullish')
                elif _w_bull and _d_bear:
                    score += 10; reasons.append('htf_pullback_buy(w+/d-)')
                elif _w_bear and _d_bear:
                    score -= 20; reasons.append('htf_both_bearish')
                elif _w_bear:
                    score -= 10; reasons.append('htf_weekly_bearish')
            elif side == 'SELL':
                if _w_bear and _d_bear:
                    score += 15; reasons.append('htf_both_bearish')
                elif _w_bear and _d_bull:
                    score += 10; reasons.append('htf_bounce_sell(w-/d+)')
                elif _w_bull and _d_bull:
                    score -= 20; reasons.append('htf_both_bullish')
                elif _w_bull:
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

    # ── ATR-based risk parameters ─────────────────────────────────────────────
    ATR_PERIOD        = 14     # lookback period for ATR calculation
    ATR_SL_MULTIPLIER = 1.8    # SL distance = ATR × this — 1.2 was inside normal 1H candle noise; 1.8× clears it

    TP1_MULTIPLIER    = 0.55   # 20 % partial close — fires early to lock break-even fast (intentionally tight)
    TP2_MULTIPLIER    = 2.8    # 20 % partial close — activate trailing stop
    TP3_MULTIPLIER    = 4.5    # 20 % partial close — used for RR validation (RR = 4.5/1.8 = 2.5)
    TP4_MULTIPLIER    = 6.5    # 20 % partial close
    TP5_MULTIPLIER    = 9.5    # close remaining position — extended runner target

    MIN_RISK_REWARD   = 2.0    # Reward / Risk using TP3 as target; trades below this are rejected

    TRAIL_MULTIPLIER  = 1.0    # trailing stop distance = ATR × this (widened to match wider SL)

    # ── Partial-close percentages (must sum to 1.0) ───────────────────────────
    TP_CLOSE_PCTS = (0.20, 0.20, 0.20, 0.20, 0.20)  # TP1 … TP5

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
        price:  float,
        side:   str,    # 'BUY' | 'SELL'
        atr:    float,
        **_kwargs,      # absorbs legacy keyword args for backward compatibility
    ) -> Dict[str, float]:
        """
        Pure ATR-based TP/SL calculation — no fixed percentages.

        Stop Loss
        ---------
        SL distance = ATR × ATR_SL_MULTIPLIER (default 1.8).
        Wide enough to survive intra-candle noise; adapts automatically when
        volatility expands or contracts.

        Take Profit Levels (TP1–TP5)
        -----------------------------
        Each TP is Entry ± (ATR × multiplier).  Five levels support scaled
        partial-profit exits: 20 % of position closed at each TP.

        Risk/Reward Validation
        ----------------------
        Reward is measured to TP3 (the first "full trend" target).
        Trades where Reward / Risk < MIN_RISK_REWARD are flagged invalid
        and should be rejected by the caller.

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

        # SL distance is fixed at ATR × SL_MULTIPLIER — no quality adjustments.
        # Volatility already informs the distance implicitly through ATR itself.
        risk = self.ATR_SL_MULTIPLIER * atr

        if side == 'BUY':
            sl  = price - risk
            tp1 = price + self.TP1_MULTIPLIER * atr
            tp2 = price + self.TP2_MULTIPLIER * atr
            tp3 = price + self.TP3_MULTIPLIER * atr
            tp4 = price + self.TP4_MULTIPLIER * atr
            tp5 = price + self.TP5_MULTIPLIER * atr
        else:  # SELL / SHORT
            sl  = price + risk
            tp1 = price - self.TP1_MULTIPLIER * atr
            tp2 = price - self.TP2_MULTIPLIER * atr
            tp3 = price - self.TP3_MULTIPLIER * atr
            tp4 = price - self.TP4_MULTIPLIER * atr
            tp5 = price - self.TP5_MULTIPLIER * atr

        # RR measured to TP3 — the first target that represents a full trend move.
        reward     = self.TP3_MULTIPLIER * atr
        rr         = round(reward / risk, 3) if risk > 0 else 0.0
        valid_rr   = rr >= self.MIN_RISK_REWARD

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

    MAX_PER_CLUSTER    = 2     # max 2 concurrent open positions per correlation cluster
    MAX_OPEN_TOTAL     = 6     # hard cap: never more than 6 open at once across all tokens
    MAX_CAPITAL_PCT    = 0.40  # never deploy more than 40 % of balance simultaneously

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
    5. If all gates pass and no open position → open paper trade (VirtualWallet)
    6. If fire=True and opposite position → MODEL_REVERSAL_TP exit, then re-enter
    7. If price hits ATR stop → STOP_HIT exit
    8. After TP1 hit → activate trailing stop at 0.5× ATR below peak (LONG)
    9. After every cycle → write data/track_record.json

    Entry gate hierarchy
    --------------------
    CRITICAL — any single failure blocks the entry:
      · predictor fire/tradeable (model + meta-labeler + calibration)
      · cooldown / flip-flop cooldown
      · Gate 0    drift monitor (live WR critically below benchmark)
      · Gate 1    no-trade regime (liquidity trap)
      · Gate 1.5  direction-regime veto (LONG in bear / SHORT in bull regime)
      · Gate 1.6  structure gate: LONG at support / SHORT at resistance with
                  5m+15m direction confirmation; BUY at resistance (SELL at
                  support) only on unanimous breakout momentum or a confirmed
                  break-and-retest hold; mid-range entries blocked
      · Gate 1.7  HTF macro veto (weekly AND daily EMA50 both opposing)
      · Gate 2    ATR floor (stops inside tick noise)
      · Gate 3    model edge floor (edge_score >= 70, drift-adjusted)
      · Gate 3.5  safe mode (edge >= 80 after 3 consecutive losses)
      · Gate 4    portfolio guard (capital / concurrency limits)

    ADVISORY — each failure appends a warning; entry blocked only when more
    than ADVISORY_WARNING_BUDGET (2) advisory gates object:
      · Gate 1.2  ranging regime
      · Gate 1.65 candlestick reversal confirmation (last 3 closed 1h bars)
      · Gate 3b   context quality score < 70 (HMM-transition adjusted)
      · Gate 3c   fake breakout / exhaustion heuristics
      · Gate 3.8  signal stability (consecutive same-direction cycles)
      · RSI exhaustion / deceleration
    Warnings are logged and forwarded to the dashboard via gate_warnings.
    """

    MAX_CONCURRENT        = 8
    HOURS_CONTEXT         = 300
    MIN_HOLD_SECONDS      = 3_600    # 1 h minimum hold before model-reversal exit
    COOLDOWN_SECONDS      = 1_800    # 30 min post-close cooldown
    FLIP_COOLDOWN_SECONDS = 3_600    # 1 h extra cooldown when new signal flips direction
    MAX_HOLD_SECONDS      = 86_400   # 24 h zombie guard
    CONFLUENCE_BUY_MIN    = 6.0   # need mildly bullish consensus (≥60th pct)
    CONFLUENCE_SELL_MAX   = 4.0   # need mildly bearish consensus (≤40th pct)

    # Signal stability: require N consecutive same-direction signals before entry.
    # Signals with edge_score >= SIGNAL_BYPASS_EDGE skip this gate (high conviction).
    SIGNAL_STABILITY_WINDOW = 2
    SIGNAL_BYPASS_EDGE      = 85.0

    # Two-tier gating.  CRITICAL gates hard-block on their own (drift monitor,
    # no-trade regime, direction-regime veto, HTF macro veto, ATR floor, edge
    # floor, safe mode, portfolio guard).  ADVISORY gates (ranging regime,
    # structure location, reversal pattern, context score, fake breakout,
    # stability, MTF timing, RSI exhaustion) append a warning instead of
    # blocking — a signal that clears every critical gate tolerates up to
    # ADVISORY_WARNING_BUDGET advisory objections before it is blocked
    # (user-confirmed 2026-07-02: tolerate 2, block at 3+).
    ADVISORY_WARNING_BUDGET = 2

    # RISKY signals (passed the budget but carrying warnings) must also clear
    # an expected-value margin in R-multiples before firing — see the
    # risk-tier block in _process_symbol.  0 = breakeven; 0.25 demands a
    # meaningfully positive edge before risking the public track record.
    RISKY_EV_MARGIN = 0.25

    # ── Structure gate (Gate 1.6) ─────────────────────────────────────────
    # Zone boundaries on range_position (0 = rolling support, 1 = resistance)
    STRUCT_SUPPORT_ZONE    = 0.35   # at/below → support zone
    STRUCT_RESISTANCE_ZONE = 0.65   # at/above → resistance zone
    # Lower-timeframe confirmation: candles must already be moving in the
    # signal direction.  Reversal entries at the correct level tolerate one
    # consolidation candle on 5m (3 of 4); breakout entries require ALL of
    # them (4/4 + 2/2) — momentum has to be unanimous to justify buying
    # into resistance / selling into support.
    STRUCT_5M_WINDOW  = 4
    STRUCT_5M_MIN     = 3
    STRUCT_15M_WINDOW = 2
    STRUCT_15M_MIN    = 2
    # Break-and-retest scan depth (closed 5m candles ≈ one hour)
    STRUCT_RETEST_LOOKBACK = 12

    # Regimes where entry is unconditionally blocked.  RANGING was demoted to
    # an advisory warning: the structure gate (BUY at support / SELL at
    # resistance) plus candlestick reversal confirmation are exactly the
    # setups that remain valid inside a range, so a blanket ban both starved
    # the engine of signals and contradicted those gates.
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
        self._alpha_last_close_time: Dict[str, float] = {}
        self._alpha_last_close_side: Dict[str, str]   = {}

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
        for sym in symbols:
            try:
                p = Predictor(sym)
                base = sym.replace('/', '_')
                buy_path  = MODEL_STORE / f"{base}_model_buy.json"
                sell_path = MODEL_STORE / f"{base}_model_sell.json"
                if buy_path.exists() and sell_path.exists():
                    # Binary dual-model pair — force tradeable regardless of meta.json
                    p.meta['tradeable']      = True
                    p.meta['tradeable_buy']  = True
                    p.meta['tradeable_sell'] = True
                    self.predictors[sym] = p
                    loaded += 1
                    tradeable += 1
                    print(f'[LiveEngine] Loaded binary dual model for {sym} (tradeable)')
                elif p.model is not None:
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
              f'symbols={len(self.predictors)}')
        asyncio.create_task(self._ws_price_ticker())
        await asyncio.sleep(2)   # let WebSocket populate live_prices first
        self._purge_subquality_positions()
        self._save_track_record()   # push restored positions to web immediately
        while True:
            t0 = time.time()
            await self._scan_all()
            self._save_track_record()
            if self.alpha_mode:
                self._save_alpha_track_record()
            await self._push_signals_to_firestore()
            sleep = max(0.0, self.scan_interval_seconds - (time.time() - t0))
            await asyncio.sleep(sleep)

    async def _push_signals_to_firestore(self) -> None:
        """Push last_signals to Firestore 'signals' collection after each scan."""
        if self.bootstrap_done < self.bootstrap_total:
            return  # skip during warmup
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
                if not sig.get('fire') or sig.get('signal', 'HOLD') in ('HOLD', 'FLAT', ''):
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

        except Exception as _e:
            print(f'[FirebasePush] {_e}')

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

            quality_score, _quality_reasons = self.quality_filter.score_signal(
                result, regime, new_side)
            fake_breakout = self.quality_filter.is_fake_breakout(result, new_side)

            # Build signal entry with enriched fields
            self.last_signals[symbol] = self._build_signal_entry(
                symbol, result, price, regime=regime,
                quality_score=quality_score, fake_breakout=fake_breakout)

            existing = self.wallet.open_positions.get(symbol)
            if existing:
                self._manage_exit(symbol, existing, result, price)
                # After exit management: overwrite last_signals quality with the
                # entry confidence so the dashboard shows entry quality, not the
                # current (potentially degraded) context score.
                if symbol in self.last_signals:
                    self.last_signals[symbol]['quality_score'] = round(existing.meta_confidence, 1)
            elif result.get('fire') and result.get('tradeable', False) and price > 0:
                now               = time.time()
                cooldown_elapsed  = now - self._last_close_time.get(symbol, 0)
                last_side         = self._last_close_side.get(symbol, '')
                is_flip           = (last_side != '' and last_side != new_side)
                required_cooldown = (
                    self.FLIP_COOLDOWN_SECONDS if is_flip else self.COOLDOWN_SECONDS
                )

                if cooldown_elapsed >= required_cooldown:
                    # Advisory-gate warning ledger (see ADVISORY_WARNING_BUDGET).
                    # Advisory gates append here instead of returning; the
                    # budget check runs after all gates have been evaluated.
                    _gate_warnings: List[str] = []

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

                    # ── Gate 1.2 (ADVISORY): ranging market ──────────────────
                    # No directional edge in a range unless the entry sits at
                    # structure with a reversal pattern — exactly what the
                    # structure + reversal gates verify.  Warn, don't block.
                    if regime.regime == _REGIME_RANGING:
                        _gate_warnings.append(f'ranging_regime(conf={regime.confidence:.2f})')

                    # ── Gate 1.5: Hard direction-regime veto ─────────────────
                    # meta_override_confidence bypasses the predictor's daily
                    # trend-filter feature (trend_1d), but must NEVER override
                    # the live regime classifier which reflects current price
                    # action (HMM + rule-based).  Opening a LONG in a confirmed
                    # BEAR regime — or a SHORT in a confirmed BULL — is the
                    # single most common way a high-conviction model gets wrecked.
                    if new_side == 'BUY' and regime.regime == _REGIME_TRENDING_BEAR:
                        print(f'[{symbol}] REGIME_VETO blocked BUY: ↓BEAR prevents LONG entry')
                        if symbol in self.last_signals:
                            self.last_signals[symbol]['fire']           = False
                            self.last_signals[symbol]['signal']         = 'HOLD'
                            self.last_signals[symbol]['regime_blocked'] = True
                        self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                        return
                    if new_side == 'SELL' and regime.regime == _REGIME_TRENDING_BULL:
                        print(f'[{symbol}] REGIME_VETO blocked SELL: ↑BULL prevents SHORT entry')
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
                    if _sg_verdict == 'PASS' and symbol in self.last_signals:
                        self.last_signals[symbol]['entry_mode'] = _sg_detail

                    # ── Gate 1.65 (ADVISORY): candlestick reversal pattern ───
                    # Entries should be timed at a trend turn, not mid-trend.
                    # The predictor forwards the max weighted reversal-pattern
                    # score over the last 3 closed 1h bars (1/2/3-candle
                    # patterns, trend-context gated in feature_engine).
                    # -1.0 = pattern data unavailable → gate fails open.
                    _bull_rev = float(result.get('cdl_bull_reversal', -1.0))
                    _bear_rev = float(result.get('cdl_bear_reversal', -1.0))
                    _rev_score = _bull_rev if new_side == 'BUY' else _bear_rev
                    if _rev_score == 0.0:
                        _gate_warnings.append(
                            f'reversal(no {"bullish" if new_side == "BUY" else "bearish"} '
                            f'pattern in last 3 bars)')

                    # ── Gate 1.7: HTF macro trend hard veto ─────────────────
                    # Hard block when weekly EMA50 trend AND daily EMA50 trend
                    # both clearly oppose the signal direction.
                    # - Weekly bearish + daily bearish → BUY blocked (no longs in macro downtrend)
                    # - Weekly bullish + daily bullish → SELL blocked (no shorts in macro uptrend)
                    # When only the weekly opposes (daily is recovering/pulling back),
                    # we allow the trade but the quality penalty in score_signal()
                    # already suppresses it below 70 in most cases.
                    _htf_w = float(result.get('macro_weekly', 0.0) or 0.0)
                    _htf_d = float(result.get('macro_daily',  0.0) or 0.0)
                    _htf_data_ok = (_htf_w != 0.0 or _htf_d != 0.0)
                    if _htf_data_ok:
                        if new_side == 'BUY' and _htf_w < -0.5 and _htf_d < -0.5:
                            print(f'[{symbol}] HTF_VETO blocked BUY: '
                                  f'weekly={_htf_w:+.1f} daily={_htf_d:+.1f} both bearish')
                            if symbol in self.last_signals:
                                self.last_signals[symbol]['fire']        = False
                                self.last_signals[symbol]['signal']      = 'HOLD'
                                self.last_signals[symbol]['htf_blocked'] = True
                            self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                            return
                        if new_side == 'SELL' and _htf_w > 0.5 and _htf_d > 0.5:
                            print(f'[{symbol}] HTF_VETO blocked SELL: '
                                  f'weekly={_htf_w:+.1f} daily={_htf_d:+.1f} both bullish')
                            if symbol in self.last_signals:
                                self.last_signals[symbol]['fire']        = False
                                self.last_signals[symbol]['signal']      = 'HOLD'
                                self.last_signals[symbol]['htf_blocked'] = True
                            self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                            return

                    # ── Gate 2: ATR floor — stops would be inside tick noise ──
                    _fe_atr_pct = float(result.get('atr_pct', 0.0))
                    if _fe_atr_pct < 0.8:
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

                    # ── Advisory warning budget — final check ─────────────────
                    # All critical gates passed.  Block only when more than
                    # ADVISORY_WARNING_BUDGET advisory gates objected.
                    if len(_gate_warnings) > self.ADVISORY_WARNING_BUDGET:
                        print(f'[{symbol}] ADVISORY_GATES blocked {new_side}: '
                              f'{len(_gate_warnings)} warnings '
                              f'[{"; ".join(_gate_warnings)}]')
                        if symbol in self.last_signals:
                            self.last_signals[symbol]['fire']             = False
                            self.last_signals[symbol]['signal']           = 'HOLD'
                            self.last_signals[symbol]['advisory_blocked'] = True
                            self.last_signals[symbol]['gate_warnings']    = _gate_warnings
                        self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                        return

                    # ── Risk-tier classification + expected-value hold ────────
                    # STRONG: zero advisory objections and top-tier conviction.
                    # RISKY:  passed the budget but carries 1-2 objections.
                    # NORMAL: everything else.
                    # A RISKY entry must also justify itself economically:
                    # using the symbol's LIVE win rate (drift monitor, falling
                    # back to the training benchmark), penalised 5 pp per
                    # warning, the expected value in R-multiples must clear
                    # RISKY_EV_MARGIN.  EV_R = p×RR − (1−p), RR to TP3.
                    # Symbols whose live performance can't support the added
                    # risk get HELD instead — the public track record is the
                    # product; a marginal risky trade isn't worth staining it.
                    _n_warn = len(_gate_warnings)
                    if _n_warn == 0 and (_model_quality >= self.SIGNAL_BYPASS_EDGE
                                         or quality_score >= 80.0):
                        _risk_tier = 'STRONG'
                    elif _n_warn >= 1:
                        _risk_tier = 'RISKY'
                    else:
                        _risk_tier = 'NORMAL'

                    if _risk_tier == 'RISKY':
                        _p_live = self.drift_monitor._live_win_rate(symbol)
                        _p_base = (_p_live if _p_live is not None
                                   else self.drift_monitor._benchmarks.get(symbol, 0.55))
                        _p_adj  = max(0.05, _p_base - 0.05 * _n_warn)
                        _rr_est = (self.risk_engine.TP3_MULTIPLIER
                                   / self.risk_engine.ATR_SL_MULTIPLIER)
                        _ev_r   = _p_adj * _rr_est - (1.0 - _p_adj)
                        if _ev_r < self.RISKY_EV_MARGIN:
                            print(f'[{symbol}] RISKY_EV_HOLD blocked {new_side}: '
                                  f'EV={_ev_r:+.2f}R < {self.RISKY_EV_MARGIN}R '
                                  f'(p={_p_adj:.2f} after {_n_warn} warnings, '
                                  f'RR={_rr_est:.1f}) '
                                  f'[{"; ".join(_gate_warnings)}]')
                            if symbol in self.last_signals:
                                self.last_signals[symbol]['fire']          = False
                                self.last_signals[symbol]['signal']        = 'HOLD'
                                self.last_signals[symbol]['risky_ev_hold'] = True
                                self.last_signals[symbol]['gate_warnings'] = _gate_warnings
                            self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                            return

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
                                        risk_tier=_risk_tier)

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
        cache_key = f'{symbol}|{timeframe}'
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

    # ── structure gate (Gate 1.6) ─────────────────────────────────────────────

    async def _structure_gate(
        self, symbol: str, side: str, price: float, result: Dict[str, Any]
    ) -> Tuple[str, str]:
        """
        Hard entry-location gate: LONG at support, SHORT at resistance — each
        confirmed by the lower timeframes already turning in the signal
        direction.  The single exception is a level break:

          BUY at resistance / SELL at support is allowed only when
            (a) momentum into the level is unanimous — every one of the last
                STRUCT_5M_WINDOW closed 5m candles and STRUCT_15M_WINDOW
                closed 15m candles trends in the signal direction, OR
            (b) the level already broke and a pullback HELD it: a 5m candle
                opened beyond the level, tagged it, and closed back beyond —
                old resistance acting as support (mirror for SELL).

        Mid-range entries are blocked: signals belong at structure.

        Returns (verdict, detail): 'PASS' | 'BLOCK' | 'SKIP' (data missing —
        gate fails open so a network hiccup never mutes the engine).
        """
        range_pos  = result.get('range_position')
        support    = float(result.get('support', 0) or 0)
        resistance = float(result.get('resistance', 0) or 0)
        if range_pos is None or not (0 < support < resistance) or price <= 0:
            return 'SKIP', 'S/R data unavailable'
        range_pos = float(range_pos)

        raw_5m  = await self._fetch_candles(
            symbol, '5m', max(self.STRUCT_5M_WINDOW, self.STRUCT_RETEST_LOOKBACK) + 2)
        raw_15m = await self._fetch_candles(symbol, '15m', self.STRUCT_15M_WINDOW + 2)
        closed_5m  = raw_5m[:-1]  if len(raw_5m)  >= 2 else []   # [-1] is forming
        closed_15m = raw_15m[:-1] if len(raw_15m) >= 2 else []
        if len(closed_5m) < self.STRUCT_5M_WINDOW or len(closed_15m) < self.STRUCT_15M_WINDOW:
            return 'SKIP', 'candle data unavailable'

        bullish = (side == 'BUY')

        def _n_trending(candles: list, n: int) -> int:
            """Candles among the last n that closed in the signal direction."""
            return sum(
                1 for c in candles[-n:]
                if (float(c[4]) > float(c[1])) == bullish and float(c[4]) != float(c[1])
            )

        n5  = _n_trending(closed_5m,  self.STRUCT_5M_WINDOW)
        n15 = _n_trending(closed_15m, self.STRUCT_15M_WINDOW)
        _counts = (f'5m {n5}/{self.STRUCT_5M_WINDOW}, '
                   f'15m {n15}/{self.STRUCT_15M_WINDOW} '
                   f'{"bullish" if bullish else "bearish"}')

        at_support    = range_pos <= self.STRUCT_SUPPORT_ZONE
        at_resistance = range_pos >= self.STRUCT_RESISTANCE_ZONE
        correct_level  = at_support    if bullish else at_resistance
        breakout_level = at_resistance if bullish else at_support
        level_name     = 'support' if bullish else 'resistance'

        if correct_level:
            # Reversal at the right level — the turn must already be underway.
            if n5 >= self.STRUCT_5M_MIN and n15 >= self.STRUCT_15M_MIN:
                return 'PASS', f'{level_name}_reversal ({_counts})'
            return 'BLOCK', (f'at {level_name} but reversal unconfirmed '
                             f'({_counts}; need {self.STRUCT_5M_MIN}/'
                             f'{self.STRUCT_5M_WINDOW} + {self.STRUCT_15M_MIN}/'
                             f'{self.STRUCT_15M_WINDOW})')

        if breakout_level:
            # Wrong side of the range — only an imminent or confirmed break
            # justifies the entry.
            if n5 >= self.STRUCT_5M_WINDOW and n15 >= self.STRUCT_15M_WINDOW:
                return 'PASS', f'breakout_momentum ({_counts})'
            level = resistance if bullish else support
            atr   = float(result.get('atr', 0) or 0)
            tol   = max(price * 0.001, atr * 0.25)
            if self._retest_held(
                    closed_5m[-self.STRUCT_RETEST_LOOKBACK:], side, level, tol):
                return 'PASS', f'breakout_retest (level {level:.6g} held)'
            return 'BLOCK', (f'{"BUY at resistance" if bullish else "SELL at support"} '
                             f'without breakout confirmation ({_counts}; no retest-hold)')

        return 'BLOCK', (f'mid-range entry (range_pos={range_pos:.2f}) — '
                         f'entries only at support/resistance')

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
                self._last_close_time[symbol] = now
                self._last_close_side[symbol] = pos.side
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

        # ── ATR-based stop/TP calculation ────────────────────────────────────
        # All levels derived from a single ATR value — no fixed percentages.
        stops = self.risk_engine.calculate_stops(price=price, side=side, atr=atr)

        stop_loss = stops['sl']
        tp1       = stops['tp1']
        tp2       = stops['tp2']
        tp3       = stops['tp3']
        tp4       = stops['tp4']
        tp5       = stops['tp5']
        rr        = stops['risk_reward']

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
            f'conf={meta_conf:.3f} quality={quality_score:.0f} regime={regime_label}\n'
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
            _re.calculate_stops(price=price, side=side, atr=atr)
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

        # Risk/Reward ratio: reward measured to TP3 (first full-trend target),
        # which matches the MIN_RISK_REWARD gate used in _open_position().
        sl_val  = entry.get('suggested_sl') or 0
        tp3_val = entry.get('tp3') or 0
        if price > 0 and sl_val and tp3_val:
            _risk   = abs(price - sl_val)
            _reward = abs(price - tp3_val)
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

            open_records = []
            for p in self.wallet.open_positions.values():
                cur = self.live_prices.get(p.symbol, p.entry_price) or p.entry_price
                if p.direction == 'LONG':
                    _pnl_pct  = (cur - p.entry_price) / p.entry_price * 100 if p.entry_price else 0.0
                else:
                    _pnl_pct  = (p.entry_price - cur) / p.entry_price * 100 if p.entry_price else 0.0
                _pnl_usdt = _pnl_pct / 100 * p.position_value
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
                    'pnl_pct':         round(_pnl_pct, 4),
                    'pnl_usdt':        round(_pnl_usdt, 4),
                    'outcome':         'OPEN',
                    'exit_reason':     None,
                    'meta_confidence': p.meta_confidence,
                    'position_value':  p.position_value,
                    'signal_strength': p.signal_strength,
                    'atr':             p.atr,
                    'atr_multiplier':  p.atr_multiplier,
                    'stop_loss':       p.stop_loss,
                    'take_profit_1':   p.take_profit_1,
                    'take_profit_2':   p.take_profit_2,
                    'take_profit_3':   p.take_profit_3,
                    'take_profit_4':   p.take_profit_4,
                    'take_profit_5':   p.take_profit_5,
                })

            wallet_records = [asdict(t) for t in self.wallet.trade_history] + open_records
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

_SCALP_RECORD_PATH = _ROOT / 'data' / 'scalp_trades.json'

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
