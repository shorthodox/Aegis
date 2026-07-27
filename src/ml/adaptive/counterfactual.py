from typing import Any, Dict, List


class CounterfactualSimulator:
    """Generate alternate trade outcome scenarios for historical signals."""

    VARIANT_DEFINITIONS = [
        {'name': 'entry_earlier', 'description': 'Simulate entering one bar earlier.'},
        {'name': 'entry_later', 'description': 'Simulate entering one bar later.'},
        {'name': 'higher_tp', 'description': 'Simulate a more aggressive take-profit target.'},
        {'name': 'wider_sl', 'description': 'Simulate a wider stop loss.'},
    ]

    def simulate_variants(self, trade: Dict[str, Any]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for variant in self.VARIANT_DEFINITIONS:
            results.append({
                'trade_id': trade.get('trade_id'),
                'variant_name': variant['name'],
                'description': variant['description'],
                'predicted_outcome': 'UNKNOWN',
                'delta_pnl_pct': 0.0,
            })
        return results

    def explain_variant(self, variant: Dict[str, Any]) -> str:
        return f"Counterfactual {variant['variant_name']}: {variant['description']}"
