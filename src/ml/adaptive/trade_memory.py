import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TradeMemoryRecord:
    trade_id: Optional[str]
    signal_id: Optional[str]
    symbol: str
    mode: str
    direction: str
    status: str
    entry_time: str
    exit_time: Optional[str]
    entry_price: float
    exit_price: Optional[float]
    pnl_pct: Optional[float]
    pnl_usdt: Optional[float]
    features: Dict[str, Any]
    signal_metadata: Dict[str, Any]
    investigation: Optional[Dict[str, Any]] = None
    counterfactuals: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None


class TradeMemoryStore:
    """Persist and query every trade and signal experiment."""

    def __init__(self, path: Optional[Path] = None):
        self.path = path or Path(__file__).resolve().parent.parent.parent / 'data' / 'adaptive_trade_memory.jsonl'
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record_signal(self, signal: Dict[str, Any]) -> None:
        payload = {
            'record_type': 'signal',
            'signal_id': signal.get('signal_id'),
            'symbol': signal.get('symbol'),
            'mode': signal.get('mode'),
            'direction': signal.get('direction'),
            'status': signal.get('status', 'SIGNAL'),
            'timestamp': signal.get('timestamp') or signal.get('data_timestamp'),
            'confidence': signal.get('confidence'),
            'quality_score': signal.get('quality_score'),
            'hmm_regime': signal.get('regime'),
            'payload': signal,
        }
        with self.path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(payload, default=str) + '\n')

    def record_trade(self, record: TradeMemoryRecord) -> None:
        payload = asdict(record)
        if not payload.get('trade_id'):
            payload['trade_id'] = payload.get('signal_id') or payload.get('symbol') or 'unknown'
        with self.path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(payload, default=str) + '\n')

    def load_all(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open('r', encoding='utf-8') as f:
            return [json.loads(line) for line in f if line.strip()]

    def query_by_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        return [row for row in self.load_all() if row.get('symbol') == symbol]
