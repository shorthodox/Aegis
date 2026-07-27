from typing import Any, Dict, List


class SimilaritySearchIndex:
    """Find historical trades that resemble a candidate signal."""

    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def index_trades(self, trades: List[Dict[str, Any]]) -> None:
        self.records = [t for t in trades if isinstance(t, dict)]

    def _feature_vector(self, record: Dict[str, Any]) -> List[float]:
        return [
            float(record.get('confidence', 0.0) or 0.0),
            float(record.get('quality_score', 0.0) or 0.0),
            float(record.get('hmm_confidence', 0.0) or 0.0),
            float(record.get('p_buy', 0.0) or 0.0),
            float(record.get('p_sell', 0.0) or 0.0),
            float(record.get('confluence_score', 0.0) or 0.0),
        ]

    def find_similar(self, signal: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.records:
            return []
        target = self._feature_vector(signal)
        scored: List[Dict[str, Any]] = []
        for record in self.records:
            vector = self._feature_vector(record)
            distance = sum((a - b) ** 2 for a, b in zip(target, vector)) ** 0.5
            scored.append({
                'distance': round(distance, 4),
                'signal_id': record.get('signal_id'),
                'symbol': record.get('symbol'),
                'outcome': record.get('outcome') or record.get('status'),
                'confidence': record.get('confidence'),
                'quality_score': record.get('quality_score'),
                'record_type': record.get('record_type', 'trade'),
            })
        scored.sort(key=lambda row: row['distance'])
        return scored[:min(top_k, len(scored))]
