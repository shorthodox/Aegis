import pandas as pd
import numpy as np
from typing import Dict, Any, Union

class EdgeScoringEngine:
    """
    Computes an Edge Score (0-100) combining multiple factors to replace the boolean
    meta-confidence threshold gate.

    Formula:
    0.30 * regime_edge + 0.25 * quality_score + 0.15 * trend_strength + 
    0.10 * vol_expansion + 0.10 * liquidity_score + 0.10 * meta_confidence
    """

    WEIGHTS = {
        'regime': 0.30,
        'quality': 0.25,
        'trend': 0.15,
        'volatility': 0.10,
        'liquidity': 0.10,
        'confidence': 0.10,
    }

    REGIME_EDGE_MAP = {
        'TRENDING_BULL': 85.0,
        'TRENDING_BEAR': 85.0,
        'ACCUMULATION': 70.0,
        'DISTRIBUTION': 70.0,
        'VOLATILE_EXPANSION': 65.0,
        'COMPRESSION': 50.0,
        'CHOPPY': 35.0,
        'RANGING': 45.0,
        'UNKNOWN': 50.0
    }

    @classmethod
    def compute_edge_batch(cls, df: pd.DataFrame, meta_probs: np.ndarray, side: str) -> pd.Series:
        """
        Compute edge score across a DataFrame (used in training/backtesting).
        """
        n = len(df)
        if n == 0:
            return pd.Series(dtype=float)

        # 1. Regime Edge (use HMM labels if available, else fallback to 'RANGING')
        regime_col = 'hmm_regime' if 'hmm_regime' in df.columns else 'trend_regime'
        if regime_col in df.columns:
            regime_scores = df[regime_col].map(cls.REGIME_EDGE_MAP).fillna(50.0)
        else:
            regime_scores = pd.Series(50.0, index=df.index)

        # 2. Quality Score Proxy
        # Real-time uses SignalQualityFilter. Here we approximate for historical backtest.
        quality = pd.Series(50.0, index=df.index)
        
        # Add points for ADX > 25
        if 'adx' in df.columns:
            quality += np.where(df['adx'] > 25, 15.0, 0.0)
        
        # Add points for volume conviction
        if 'volume_zscore' in df.columns:
            quality += np.where(df['volume_zscore'] > 1.5, 10.0, 0.0)
            
        # Add points for high meta conf
        quality += np.where(meta_probs > 0.75, 10.0, 0.0)
        
        # Add points for RSI
        if 'rsi_14' in df.columns:
            rsi = df['rsi_14']
            if side == 'BUY':
                quality += np.where(rsi < 75, 10.0, 0.0)
            elif side == 'SELL':
                quality += np.where(rsi > 25, 10.0, 0.0)
                
        # Clamp quality
        quality = quality.clip(0, 100)

        # 3. Trend Strength
        if 'adx' in df.columns:
            trend = (df['adx'] / 50.0 * 100.0).clip(0, 100)
        elif 'efficiency_ratio_10' in df.columns:
            trend = (df['efficiency_ratio_10'] * 100.0).clip(0, 100)
        else:
            trend = pd.Series(50.0, index=df.index)

        # 4. Volatility Expansion
        if 'atr_pct' in df.columns:
            # Map typical ATR % (0-5%) to 0-100
            vol = (df['atr_pct'] * 20.0).clip(0, 100)
        else:
            vol = pd.Series(50.0, index=df.index)

        # 5. Liquidity Score
        if 'volume_zscore' in df.columns:
            # map -2 to +3 zscore -> 0 to 100
            liq = ((df['volume_zscore'] + 2.0) * 20.0).clip(0, 100)
        else:
            liq = pd.Series(50.0, index=df.index)

        # 6. Meta Confidence
        conf = pd.Series(meta_probs * 100.0, index=df.index).clip(0, 100)

        # Combine
        edge_score = (
            cls.WEIGHTS['regime'] * regime_scores +
            cls.WEIGHTS['quality'] * quality +
            cls.WEIGHTS['trend'] * trend +
            cls.WEIGHTS['volatility'] * vol +
            cls.WEIGHTS['liquidity'] * liq +
            cls.WEIGHTS['confidence'] * conf
        )

        return edge_score.clip(0, 100)

    @classmethod
    def compute_edge_realtime(cls, result: Dict[str, Any], meta_prob: float, side: str, quality_score: float = 50.0) -> float:
        """
        Compute edge score for a single realtime prediction.
        """
        # 1. Regime Edge
        regime = result.get('hmm_regime', result.get('trend_regime', 'UNKNOWN'))
        regime_score = cls.REGIME_EDGE_MAP.get(regime, 50.0)

        # 2. Quality Score is passed in directly from live_engine (SignalQualityFilter) or defaults to 50

        # 3. Trend Strength
        adx = float(result.get('adx', 20.0) or 20.0)
        trend = min(100.0, max(0.0, adx / 50.0 * 100.0))

        # 4. Volatility Expansion
        atr_pct = float(result.get('atr_pct', 1.5) or 1.5)
        vol = min(100.0, max(0.0, atr_pct * 20.0))

        # 5. Liquidity
        vol_z = float(result.get('volume_zscore', 0.0) or 0.0)
        liq = min(100.0, max(0.0, (vol_z + 2.0) * 20.0))

        # 6. Confidence
        conf = min(100.0, max(0.0, meta_prob * 100.0))

        edge_score = (
            cls.WEIGHTS['regime'] * regime_score +
            cls.WEIGHTS['quality'] * quality_score +
            cls.WEIGHTS['trend'] * trend +
            cls.WEIGHTS['volatility'] * vol +
            cls.WEIGHTS['liquidity'] * liq +
            cls.WEIGHTS['confidence'] * conf
        )
        return float(min(100.0, max(0.0, edge_score)))
