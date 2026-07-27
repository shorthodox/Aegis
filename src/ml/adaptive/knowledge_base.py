from typing import Any, Dict, List


class KnowledgeBase:
    """Persist the evolving intelligence knowledge artifacts."""

    def __init__(self):
        self.entries: List[Dict[str, Any]] = []

    def add_entry(self, entry: Dict[str, Any]) -> None:
        self.entries.append(entry)

    def list_entries(self) -> List[Dict[str, Any]]:
        return self.entries
