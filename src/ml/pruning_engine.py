from pathlib import Path
from typing import Dict, Any, List
import json

ROOT = Path(__file__).resolve().parents[2]
HISTORY = ROOT / 'artifacts' / 'feature_stability' / 'history.jsonl'


def generate_feature_health_report(last_n: int = 10, top_k: int = 50,
                                   healthy_freq: float = 0.6,
                                   dead_freq: float = 0.1) -> Dict[str, Any]:
    """Analyze recent retrain history and recommend feature categories.

    Returns dict with keys: healthy, weak, dead, details
    """
    if not HISTORY.exists():
        return {'healthy': [], 'weak': [], 'dead': [], 'details': {}}
    records = [json.loads(l) for l in open(HISTORY).read().strip().splitlines()[-last_n:]]
    counts: Dict[str, int] = {}
    gains: Dict[str, list] = {}
    for r in records:
        imp = r.get('feature_importances', {})
        top = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:top_k]
        for f, g in top:
            counts[f] = counts.get(f, 0) + 1
            gains.setdefault(f, []).append(float(g))

    n = max(1, len(records))
    healthy, weak, dead = [], [], []
    details = {}
    for f in set(counts.keys()):
        freq = counts.get(f, 0) / n
        mean_gain = sum(gains.get(f, [])) / max(1, len(gains.get(f, [])))
        details[f] = {'freq': freq, 'mean_gain': mean_gain}
        if freq >= healthy_freq and mean_gain > 0:
            healthy.append(f)
        elif freq <= dead_freq and mean_gain < 1e-6:
            dead.append(f)
        else:
            weak.append(f)

    return {'healthy': healthy, 'weak': weak, 'dead': dead, 'details': details}
