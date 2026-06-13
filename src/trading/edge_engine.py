import pandas as pd
import numpy as np
from typing import Dict, Any, Union

class EdgeScoringEngine:
    """
    Computes an Edge Score (0-100) combining multiple factors to replace the boolean
    meta-confidence threshold gate.

    Formula:
    edge_score = quality + regime + trend + confluence + volume + volatility + market_structure
    Range: 0-100
    """

    @classmethod
    def compute_edge_batch(cls, df: pd.DataFrame, meta_probs: np.ndarray, side: str, use_rank: bool = False) -> pd.Series:
        """
        Compute edge score across a DataFrame (used in training/backtesting).
        """
        n = len(df)
        if n == 0:
            return pd.Series(dtype=float)

        def _get_s(col, default_val=0.0):
            return df[col] if col in df.columns else pd.Series(default_val, index=df.index)

        # 1. Base Quality Score (0-100)
        quality = pd.Series(50.0, index=df.index)
        adx = _get_s('adx') if 'adx' in df.columns else _get_s('adx_14')
        quality += np.where(adx > 25, 15.0, 0.0)
        vol_z = _get_s('volume_zscore')
        quality += np.where(vol_z > 1.5, 10.0, 0.0)
        # Scale quality by meta probability: P=1.0→+15, P=0.5→0, P=0.0→-15.
        # Proportional scaling preserves the spread between high/low confidence signals
        # so threshold search has meaningful discrimination.
        quality += np.clip((np.asarray(meta_probs, dtype=float) - 0.5) * 30.0, -15.0, 15.0)
        if 'rsi_14' in df.columns:
            rsi = df['rsi_14']
            if side == 'BUY':
                quality += np.where(rsi < 70, 10.0, 0.0)
            elif side == 'SELL':
                quality += np.where(rsi > 30, 10.0, 0.0)
        quality_score = quality.clip(0, 100)

        # 2. Regime Component (-15 to +15 points)
        regime_col = 'hmm_regime' if 'hmm_regime' in df.columns else 'trend_regime'
        regime_map = {
            'TRENDING_BULL': 15.0,
            'TRENDING_BEAR': 15.0,
            'VOLATILE_EXPANSION': 15.0,
            'COMPRESSION': 5.0,
            'ACCUMULATION': -5.0,
            'DISTRIBUTION': -5.0,
            'CHOPPY': -15.0,
            'RANGING': 0.0,
            'UNKNOWN': 0.0
        }
        if regime_col in df.columns:
            regime_component = df[regime_col].map(regime_map).fillna(0.0)
        else:
            regime_component = pd.Series(0.0, index=df.index)

        # 3. Trend Component (-15 to +15 points)
        macro_trend_1d = _get_s('macro_trend_1d')
        if side == 'BUY':
            trend_val = macro_trend_1d * 10.0
        else:
            trend_val = -macro_trend_1d * 10.0
        trend_val += np.where(adx > 25, 5.0, 0.0)
        trend_component = trend_val.clip(-15.0, 15.0)

        # 4. Confluence Component (-10 to +10 points)
        total_confluence = _get_s('total_confluence')
        if side == 'BUY':
            confluence_component = (total_confluence * 10.0).clip(-10.0, 10.0)
        else:
            confluence_component = (-total_confluence * 10.0).clip(-10.0, 10.0)

        # 5. Volume Component (-10 to +10 points)
        volume_component = (vol_z * 5.0).clip(-10.0, 10.0)

        # 6. Volatility Component (-10 to +10 points)
        if '_atr_raw' in df.columns and '_close_raw' in df.columns:
            atr_pct = df['_atr_raw'] / df['_close_raw']
        elif 'atr_pct' in df.columns:
            atr_pct = df['atr_pct']
        elif '_atr' in df.columns and 'close' in df.columns:
            atr_pct = df['_atr'] / df['close']
        else:
            atr_pct = pd.Series(0.015, index=df.index)
        volatility_component = (atr_pct * 400.0 - 5.0).clip(-10.0, 10.0)

        # 7. Market Structure Component (-10 to +10 points)
        at_sup = _get_s('is_at_support')
        at_res = _get_s('is_at_resistance')
        if side == 'BUY':
            structure_component = (at_sup * 10.0 - at_res * 10.0).clip(-10.0, 10.0)
        else:
            structure_component = (at_res * 10.0 - at_sup * 10.0).clip(-10.0, 10.0)

        # Total Edge Score
        edge_score = (
            quality_score +
            regime_component +
            trend_component +
            confluence_component +
            volume_component +
            volatility_component +
            structure_component
        )
        if use_rank:
            rank_df = cls.compute_edge_rank_batch(df, meta_probs, side)
            return rank_df['edge_percentile_100']
        return edge_score.clip(0, 100)

    @classmethod
    def compute_signal_strength_scores(cls, df: pd.DataFrame, meta_probs: np.ndarray, side: str) -> pd.DataFrame:
        n = len(df)
        if n == 0:
            return pd.DataFrame(
                columns=[
                    'regime_score', 'trend_score', 'momentum_score', 'volume_score',
                    'orderflow_score', 'volatility_score', 'signal_strength_score',
                    'edge_rank', 'edge_percentile', 'edge_percentile_100'
                ],
                index=df.index,
            )

        def _get_s(col, default_val=0.0):
            return df[col] if col in df.columns else pd.Series(default_val, index=df.index)

        regime_col = _get_s('hmm_regime', 'UNKNOWN') if 'hmm_regime' in df.columns else _get_s('trend_regime', 'UNKNOWN')
        regime_map = {
            'TRENDING_BULL': 1.0,
            'TRENDING_BEAR': 1.0,
            'VOLATILE_EXPANSION': 0.95,
            'COMPRESSION': 0.75,
            'ACCUMULATION': 0.65,
            'DISTRIBUTION': 0.65,
            'CHOPPY': 0.30,
            'RANGING': 0.45,
            'UNKNOWN': 0.50,
        }
        regime_score = regime_col.map(regime_map).fillna(0.50)

        volatility_regime = _get_s('volatility_regime', 'UNKNOWN').astype(str).str.upper()
        volatility_regime_map = {
            'VOLATILE_EXPANSION': 0.95,
            'HIGH_VOL': 0.95,
            'MEDIUM_VOL': 0.80,
            'LOW_VOL': 0.65,
            'COMPRESSION': 0.75,
            'UNKNOWN': 0.60,
        }
        volatility_bonus = volatility_regime.map(volatility_regime_map).fillna(0.60)
        regime_score = ((regime_score + volatility_bonus) / 2.0).clip(0.0, 1.0)

        macro_trend_1d = _get_s('macro_trend_1d', 0.0)
        adx = _get_s('adx') if 'adx' in df.columns else _get_s('adx_14')
        trend_strength = np.clip(np.abs(macro_trend_1d) / 0.4, 0.0, 1.0)
        adx_strength = np.clip(adx / 40.0, 0.0, 1.0)
        trend_score = ((trend_strength + adx_strength) / 2.0).clip(0.0, 1.0)

        rsi = _get_s('rsi_14', 50.0)
        macd_hist = _get_s('macd_hist', 0.0)
        rsi_score = np.clip(np.abs(rsi - 50.0) / 50.0, 0.0, 1.0)
        macd_score = np.clip(np.abs(macd_hist) / 0.05, 0.0, 1.0)
        momentum_score = ((rsi_score + macd_score) / 2.0).clip(0.0, 1.0)

        vol_z = _get_s('volume_zscore', 0.0)
        rel_vol = _get_s('relative_volume', 1.0)
        volume_z_score = np.clip((vol_z + 1.5) / 3.0, 0.0, 1.0)
        volume_rel_score = np.clip((rel_vol - 1.0) / 3.0 + 0.5, 0.0, 1.0)
        volume_score = ((volume_z_score + volume_rel_score) / 2.0).clip(0.0, 1.0)

        oi_z = _get_s('oi_zscore', 0.0)
        cmf_20 = _get_s('cmf_20', 0.0)
        oi_price_divergence = _get_s('oi_price_divergence', 0.0)
        oi_score = np.clip(0.5 + oi_z / 6.0, 0.0, 1.0)
        cmf_score = np.clip(0.5 + cmf_20 / 0.2, 0.0, 1.0)
        divergence_score = np.clip(1.0 - np.abs(oi_price_divergence) / 0.05, 0.0, 1.0)
        orderflow_score = ((oi_score + cmf_score + divergence_score) / 3.0).clip(0.0, 1.0)

        if '_atr_raw' in df.columns and '_close_raw' in df.columns:
            atr_pct = df['_atr_raw'] / df['_close_raw']
        elif 'atr_pct' in df.columns:
            atr_pct = _get_s('atr_pct', 0.015)
        elif '_atr' in df.columns and 'close' in df.columns:
            atr_pct = _get_s('_atr', 0.015) / _get_s('close', 1.0)
        else:
            atr_pct = pd.Series(0.015, index=df.index)
        volatility_score = np.clip((atr_pct * 100.0 - 1.5) / 6.0 + 0.5, 0.0, 1.0)

        signal_strength_score = (
            0.18 * regime_score +
            0.18 * trend_score +
            0.18 * momentum_score +
            0.15 * volume_score +
            0.15 * orderflow_score +
            0.16 * volatility_score
        ).clip(0.0, 1.0)

        edge_rank = np.clip(signal_strength_score * np.clip(np.asarray(meta_probs, dtype=float), 0.0, 1.0), 0.0, 1.0)
        edge_percentile = pd.Series(edge_rank, index=df.index)
        edge_percentile = edge_percentile.rank(method='average', pct=True).fillna(0.0)
        edge_percentile_100 = edge_percentile * 100.0

        return pd.DataFrame({
            'regime_score': regime_score,
            'trend_score': trend_score,
            'momentum_score': momentum_score,
            'volume_score': volume_score,
            'orderflow_score': orderflow_score,
            'volatility_score': volatility_score,
            'signal_strength_score': signal_strength_score,
            'edge_rank': edge_rank,
            'edge_percentile': edge_percentile,
            'edge_percentile_100': edge_percentile_100,
        }, index=df.index)

    @classmethod
    def compute_edge_rank_batch(cls, df: pd.DataFrame, meta_probs: np.ndarray, side: str) -> pd.DataFrame:
        return cls.compute_signal_strength_scores(df, meta_probs, side)

    @classmethod
    def compute_edge_realtime(cls, result: Dict[str, Any], meta_prob: float, side: str, quality_score: float = 50.0) -> float:
        """
        Compute edge score for a single realtime prediction.
        """
        # 1. Base Quality Score
        adx = float(result.get('adx', result.get('adx_14', 20.0)) or 20.0)
        vol_z = float(result.get('volume_zscore', 0.0) or 0.0)
        
        # Adjust quality_score based on realtime inputs
        q = quality_score
        if adx > 25:
            q += 15.0
        if vol_z > 1.5:
            q += 10.0
        q += float(np.clip((meta_prob - 0.5) * 30.0, -15.0, 15.0))
            
        rsi = float(result.get('rsi_14', 50.0) or 50.0)
        if side == 'BUY':
            if rsi < 70: q += 10.0
        elif side == 'SELL':
            if rsi > 30: q += 10.0
            
        quality_score_val = min(100.0, max(0.0, q))

        # 2. Regime Component (-15 to +15)
        regime = result.get('hmm_regime', result.get('trend_regime', 'UNKNOWN'))
        regime_map = {
            'TRENDING_BULL': 15.0,
            'TRENDING_BEAR': 15.0,
            'VOLATILE_EXPANSION': 15.0,
            'COMPRESSION': 5.0,
            'ACCUMULATION': -5.0,
            'DISTRIBUTION': -5.0,
            'CHOPPY': -15.0,
            'RANGING': 0.0,
            'UNKNOWN': 0.0
        }
        regime_component = regime_map.get(regime, 0.0)

        # 3. Trend Component (-15 to +15)
        macro_trend_1d = float(result.get('macro_trend_1d', 0.0) or 0.0)
        if side == 'BUY':
            trend_val = macro_trend_1d * 10.0
        else:
            trend_val = -macro_trend_1d * 10.0
        if adx > 25:
            trend_val += 5.0
        trend_component = min(15.0, max(-15.0, trend_val))

        # 4. Confluence Component (-10 to +10)
        total_confluence = float(result.get('total_confluence', 0.0) or 0.0)
        if side == 'BUY':
            confluence_component = min(10.0, max(-10.0, total_confluence * 10.0))
        else:
            confluence_component = min(10.0, max(-10.0, -total_confluence * 10.0))

        # 5. Volume Component (-10 to +10)
        volume_component = min(10.0, max(-10.0, vol_z * 5.0))

        # 6. Volatility Component (-10 to +10)
        atr_pct = float(result.get('atr_pct', 0.015) or 0.015)
        volatility_component = min(10.0, max(-10.0, atr_pct * 400.0 - 5.0))

        # 7. Market Structure Component (-10 to +10)
        at_sup = float(result.get('is_at_support', 0.0) or 0.0)
        at_res = float(result.get('is_at_resistance', 0.0) or 0.0)
        if side == 'BUY':
            structure_component = min(10.0, max(-10.0, at_sup * 10.0 - at_res * 10.0))
        else:
            structure_component = min(10.0, max(-10.0, at_res * 10.0 - at_sup * 10.0))

        # Total
        edge_score = (
            quality_score_val +
            regime_component +
            trend_component +
            confluence_component +
            volume_component +
            volatility_component +
            structure_component
        )
        return float(min(100.0, max(0.0, edge_score)))
