from typing import Any, Dict, List


class KnowledgeGraph:
    """Store and query relationships between market state, trades, and outcomes."""

    def __init__(self):
        self.nodes: List[Dict[str, Any]] = []
        self.edges: List[Dict[str, Any]] = []

    def add_relation(self, source: str, target: str, relation: str, weight: float = 1.0) -> None:
        self.nodes.append({'id': source})
        self.nodes.append({'id': target})
        self.edges.append({
            'source': source,
            'target': target,
            'relation': relation,
            'weight': weight,
        })

    def query_relations(self, entity: str) -> List[Dict[str, Any]]:
        return [edge for edge in self.edges if edge['source'] == entity or edge['target'] == entity]
