import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import datetime

ROOT = Path(__file__).resolve().parents[2]
HISTORY = ROOT / 'artifacts' / 'feature_stability' / 'history.jsonl'
HISTORY.parent.mkdir(parents=True, exist_ok=True)


def record_retrain(symbol: str, retrain_id: str, feature_importances: Dict[str, float],
                   shap_summary: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Append retrain feature stats to history and return the record saved.

    feature_importances: mapping feature -> gain
    shap_summary: optional mapping feature -> mean_abs_shap
    """
    rec = {
        'symbol': symbol,
        'retrain_id': retrain_id,
        'timestamp': datetime.datetime.utcnow().isoformat(),
        'feature_importances': feature_importances,
        'shap_summary': shap_summary or {},
    }
    with open(HISTORY, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def compute_stability_scores(last_n: int = 10, top_k: int = 50) -> Dict[str, float]:
    """Compute a simple stability score for features from the last N retrains.

    Returns mapping feature -> stable_score in [0,1].
    """
    import json
    if not HISTORY.exists():
        return {}
    lines = open(HISTORY).read().strip().splitlines()
    if not lines:
        return {}
    records = [json.loads(l) for l in lines[-last_n:]]
    counts: Dict[str, int] = {}
    gains: Dict[str, List[float]] = {}
    shap_pos: Dict[str, int] = {}
    for r in records:
        imp = r.get('feature_importances', {})
        # rank by gain
        top = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:top_k]
        for f, g in top:
            counts[f] = counts.get(f, 0) + 1
            gains.setdefault(f, []).append(float(g))
        shap = r.get('shap_summary', {}) or {}
        for f, v in shap.items():
            if float(v) > 0:
                shap_pos[f] = shap_pos.get(f, 0) + 1

    features = set(list(counts.keys()) + list(gains.keys()) + list(shap_pos.keys()))
    scores: Dict[str, float] = {}
    for f in features:
        freq = counts.get(f, 0) / float(max(1, len(records)))
        mean_gain = float(sum(gains.get(f, [])) / max(1, len(gains.get(f, [])))) if gains.get(f) else 0.0
        shap_consistency = shap_pos.get(f, 0) / float(max(1, len(records)))
        # combine: freq (0.5), normalized mean_gain (0.3), shap_consistency (0.2)
        # normalize mean_gain via log1p then scale
        import math
        ng = math.log1p(mean_gain) / (1.0 + math.log1p(mean_gain)) if mean_gain > 0 else 0.0
        score = 0.5 * freq + 0.3 * ng + 0.2 * shap_consistency
        scores[f] = max(0.0, min(1.0, score))
    return scores
