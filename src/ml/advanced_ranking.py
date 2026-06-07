from typing import Dict
import math


def compute_composite_edge(probability: float,
                           signal_strength: float = 0.5,
                           edge_rank: float = 0.5,
                           regime_quality: float = 0.5,
                           feature_stability: float = 0.5,
                           volatility_weight: float = 0.5,
                           trend_strength: float = 0.5,
                           weights: Dict[str, float] = None) -> float:
    """Combine components into a normalized composite edge score [0,1].

    All components expected in [0,1]. Uses simple weighted average with
    defaults chosen to modestly prefer probability and regime.
    """
    comps = {
        'prob': probability,
        'strength': signal_strength,
        'rank': edge_rank,
        'regime': regime_quality,
        'stability': feature_stability,
        'vol': volatility_weight,
        'trend': trend_strength,
    }
    if weights is None:
        weights = {'prob': 3.0, 'strength': 1.0, 'rank': 1.0, 'regime': 1.5, 'stability': 1.0, 'vol': 0.5, 'trend': 0.5}
    num = 0.0
    den = 0.0
    for k, v in comps.items():
        w = weights.get(k, 1.0)
        num += w * float(max(0.0, min(1.0, v)))
        den += w
    out = float(num / (den + 1e-12))
    # small non-linear squash to spread extremes
    return float(max(0.0, min(1.0, (math.tanh((out - 0.5) * 2.0) + 1.0) / 2.0)))
