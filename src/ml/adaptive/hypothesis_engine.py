from typing import Any, Dict, List


class HypothesisEngine:
    """Manage candidate trading hypotheses and their evidence."""

    def __init__(self):
        self.hypotheses: List[Dict[str, Any]] = []

    def propose(self, description: str, evidence: Dict[str, Any]) -> Dict[str, Any]:
        hypothesis = {
            'id': f'hyp_{len(self.hypotheses) + 1}',
            'description': description,
            'evidence': evidence,
            'status': 'proposed',
        }
        self.hypotheses.append(hypothesis)
        return hypothesis

    def list_hypotheses(self) -> List[Dict[str, Any]]:
        return self.hypotheses
