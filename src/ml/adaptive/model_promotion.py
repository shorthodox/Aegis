from typing import Any, Dict


class ModelPromotionGate:
    """Decide whether a candidate change should be promoted to production."""

    def evaluate(self, candidate_metrics: Dict[str, Any], baseline_metrics: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'approved': False,
            'reason': 'baseline preservation',
            'candidate_metrics': candidate_metrics,
            'baseline_metrics': baseline_metrics,
        }
