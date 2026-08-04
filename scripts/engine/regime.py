"""Market regime classification from the predictor's result dict.

No extra API calls — every input is already present in what
Predictor.predict_realtime() returns.

The v82 note inside _detect() is worth reading before touching this file: a
paste error silently disabled regime detection for the entire fleet for weeks,
and the only visible symptom was that every symbol reported RANGING with
confidence 0.4. detect()'s fail-safe is what hid it, so keep the fail-safe but
be aware it can mask a real defect — the differential test in
scripts/tests/test_engine_extraction_parity.py exercises _detect() directly for
exactly that reason.
"""
from __future__ import annotations

from typing import Any, Dict

from scripts.engine.config import (
    REGIME_ACCUMULATION, REGIME_DISTRIBUTION, REGIME_LIQUIDITY_TRAP,
    REGIME_RANGING, REGIME_TRENDING_BEAR, REGIME_TRENDING_BULL,
    REGIME_VOLATILE_COMPRESS, REGIME_VOLATILE_EXPANSION,
)
from scripts.engine.models import RegimeState

__all__ = ["MarketRegimeDetector"]


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
                regime               = REGIME_RANGING,
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
        bull_score = (
            int(rsi > 52)
            + int(macd_signal == 'BULLISH')
            + int('UP' in trend_regime)
            + int(market_bias == 'BULLISH')
        )
        bear_score = (
            int(rsi < 48)
            + int(macd_signal == 'BEARISH')
            + int('DOWN' in trend_regime)
            + int(market_bias == 'BEARISH')
        )

        is_bullish = bull_score > bear_score
        is_bearish = bear_score > bull_score

        if bull_score == bear_score:
            is_bullish = 'UP' in trend_regime and 'DOWN' not in trend_regime
            is_bearish = 'DOWN' in trend_regime and 'UP' not in trend_regime

        # Avoid flipping the regime when the opposing directional signal is
        # strong and the other side is only weakly present.
        if rsi < 45 and market_bias == 'BEARISH':
            is_bullish = False
        if rsi > 55 and market_bias == 'BULLISH':
            is_bearish = False

        # v82 REPAIR: commit 0f1e3e32 ("hotfix: immediate TP recross closures")
        # pasted a copy of the _manage_exit TP-recross ladder into the middle of
        # this method, overwriting the five flag assignments below.  Those names
        # are read further down, and `pos`/`check_price`/`_close` do not exist in
        # this scope, so _detect() raised NameError on EVERY call and detect()'s
        # fail-safe swallowed it — the engine has been running with a hardcoded
        # RANGING / conf 0.4 / max_position_pct 0.08 regime for every symbol on
        # every scan since 2026-07-25.  Restored from 0f1e3e32~1.
        low_volume   = (volume_strength == 'BELOW_AVERAGE' or vol_zscore < -0.5)
        high_oi      = oi_trend == 'INCREASING'
        low_oi       = oi_trend == 'DECREASING'
        longs_paying = funding_bias == 'LONGS_PAYING'
        shorts_paying= funding_bias == 'SHORTS_PAYING'

        # ── 1. Liquidity trap: low volume, choppy, no trending structure ─────
        if low_volume and is_ranging and is_quiet:
            return RegimeState(
                regime               = REGIME_LIQUIDITY_TRAP,
                confidence           = 0.75,
                trade_allowed        = False,
                preferred_strategies = [],
                max_position_pct     = 0.0,
            )

        # ── 2. Volatile expansion ─────────────────────────────────────────────
        if is_volatile and atr_pct > 4.0:
            conf = min(0.9, 0.6 + (atr_pct - 4.0) * 0.05)
            return RegimeState(
                regime               = REGIME_VOLATILE_EXPANSION,
                confidence           = round(conf, 3),
                trade_allowed        = True,
                preferred_strategies = ['BREAKOUT', 'MOMENTUM'],
                max_position_pct     = 0.06,  # reduced size in expansion
            )

        # ── 3. Volatile compression: quiet market after expansion ─────────────
        if is_quiet and not is_trending:
            return RegimeState(
                regime               = REGIME_VOLATILE_COMPRESS,
                confidence           = 0.65,
                trade_allowed        = True,
                preferred_strategies = ['RANGE_TRADE', 'MEAN_REVERT'],
                max_position_pct     = 0.07,
            )

        # ── 4. Accumulation: ranging + increasing OI + shorts paying ─────────
        if is_ranging and high_oi and shorts_paying and not is_bearish:
            conf = 0.55 + (0.15 if rsi < 55 else 0.0) + (0.10 if vol_zscore > 0.5 else 0.0)
            return RegimeState(
                regime               = REGIME_ACCUMULATION,
                confidence           = round(min(conf, 0.85), 3),
                trade_allowed        = True,
                preferred_strategies = ['RANGE_BUY', 'BREAKOUT_LONG'],
                max_position_pct     = 0.10,
            )

        # ── 5. Distribution: ranging + increasing OI + longs paying ──────────
        if is_ranging and high_oi and longs_paying and not is_bullish:
            conf = 0.55 + (0.15 if rsi > 45 else 0.0) + (0.10 if vol_zscore > 0.5 else 0.0)
            return RegimeState(
                regime               = REGIME_DISTRIBUTION,
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
                regime               = REGIME_TRENDING_BULL,
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
                regime               = REGIME_TRENDING_BEAR,
                confidence           = round(min(conf, 0.95), 3),
                trade_allowed        = True,
                preferred_strategies = ['TREND_FOLLOW', 'MOMENTUM', 'PULLBACK_SHORT'],
                max_position_pct     = 0.13,
            )

        # ── 8. Default: ranging / neutral ─────────────────────────────────────
        return RegimeState(
            regime               = REGIME_RANGING,
            confidence           = 0.50,
            trade_allowed        = True,
            preferred_strategies = ['RANGE_TRADE', 'MEAN_REVERT', 'SUPPORT_BUY'],
            max_position_pct     = 0.08,
        )
