from typing import Any, Dict, List, Optional


class TradeInvestigator:
    """Analyze completed trades and generate structured failure diagnostics."""

    def __init__(self):
        self.diagnosis_rules = []

    def analyze_trade(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """Return a structured diagnosis for a closed trade."""
        findings: List[Dict[str, Any]] = []

        if trade.get('pnl_pct') is None:
            return {'trade_id': trade.get('trade_id'), 'findings': findings}

        if trade.get('pnl_pct', 0.0) < 0:
            findings.append({
                'issue': 'negative_outcome',
                'description': 'Trade finished at a loss.',
                'confidence': 1.0,
            })

        if trade.get('exit_reason') == 'SL_HIT':
            findings.append({
                'issue': 'stopped_out',
                'description': 'Stop loss was reached before take profit.',
                'confidence': 0.8,
            })

        return {
            'trade_id': trade.get('trade_id'),
            'findings': findings,
            'primary_cause': findings[0] if findings else None,
        }

    def score_trade(self, trade: Dict[str, Any]) -> float:
        """Compute a simple fault score for use by the adaptive layer."""
        diagnosis = self.analyze_trade(trade)
        return float(len(diagnosis.get('findings', [])))
