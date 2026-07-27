from typing import Any, Dict, List


class ShadowTradingManager:
    """Run alternate candidate strategies in paper/shadow mode."""

    def __init__(self):
        self.shadow_results: List[Dict[str, Any]] = []

    def submit(self, signal: Dict[str, Any]) -> None:
        self.shadow_results.append(signal)

    def summary(self) -> Dict[str, Any]:
        return {'shadow_count': len(self.shadow_results)}
