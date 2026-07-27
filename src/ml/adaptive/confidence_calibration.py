from typing import Any, Dict, List


class ConfidenceCalibrator:
    """Track and calibrate model confidence against realized outcomes."""

    def __init__(self):
        self.history: List[Dict[str, Any]] = []

    def record(self, signal: Dict[str, Any], outcome: str) -> None:
        self.history.append({
            'confidence': float(signal.get('confidence', 0.0) or 0.0),
            'outcome': outcome,
            'positive': outcome in ('WIN', 'OPEN', 'signal'),
        })
        if len(self.history) > 2000:
            self.history = self.history[-2000:]

    def calibrate(self, raw_confidence: float) -> float:
        if not self.history:
            return raw_confidence

        bucket = [h for h in self.history
                  if int(min(10, max(0, h['confidence'] * 10))) == int(min(10, max(0, raw_confidence * 10)))]
        if not bucket:
            return raw_confidence

        positive_rate = sum(1.0 for h in bucket if h['positive']) / len(bucket)
        return float((raw_confidence * 0.6) + (positive_rate * 0.4))

    def summary(self) -> Dict[str, Any]:
        return {'observations': len(self.history)}
