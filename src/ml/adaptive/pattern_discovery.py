from typing import Any, Dict, List


class PatternDiscoveryEngine:
    """Discover recurring success and failure patterns from trade history."""

    def __init__(self):
        self.patterns: List[Dict[str, Any]] = []

    def discover(self, trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.patterns = []
        for trade in trades:
            if trade.get('pnl_pct', 0) < 0:
                self.patterns.append({
                    'pattern': 'loss_cluster',
                    'symbol': trade.get('symbol'),
                    'features': trade.get('signal_metadata', {}),
                    'confidence': 0.5,
                })
        return self.patterns
