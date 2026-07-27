from typing import Any, Dict, List


class MistakeMemory:
    """Store failure fingerprints to reduce repeat mistakes."""

    def __init__(self):
        self.fingerprints: List[Dict[str, Any]] = []

    def record_mistake(self, trade: Dict[str, Any]) -> None:
        self.fingerprints.append({
            'trade_id': trade.get('trade_id'),
            'symbol': trade.get('symbol'),
            'reason': trade.get('exit_reason'),
            'context': trade.get('signal_metadata', {}),
        })

    def find_closest(self, signal: Dict[str, Any]) -> List[Dict[str, Any]]:
        return self.fingerprints[:3]
