#!/usr/bin/env python3
"""Generate training_summary.json from existing model artifacts.

This script scans src/ml/model_store and logs/features to reconstruct a
summary of available trained models and related logs without retraining.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_STORE = ROOT_DIR / "src" / "ml" / "model_store"
FEATURE_LOGS = ROOT_DIR / "logs" / "features"
SUMMARY_FILE = ROOT_DIR / "logs" / "training_summary.json"

SUPPORTED_MODEL_EXTENSIONS = [".json", ".bin"]


def _canonical_symbol_from_model_file(model_path: Path) -> str:
    stem = model_path.stem
    if stem.endswith("_model"):
        stem = stem[: -len("_model")]
    return stem.replace("_", "/")


def _load_existing_summary(summary_path: Path) -> Optional[Dict]:
    if not summary_path.exists():
        return None
    try:
        return json.loads(summary_path.read_text())
    except Exception:
        return None


def _scan_model_store(model_store_path: Path) -> Dict[str, Dict]:
    results: Dict[str, Dict] = {}
    if not model_store_path.exists():
        return results

    for p in model_store_path.iterdir():
        if not p.is_file() or p.suffix.lower() not in SUPPORTED_MODEL_EXTENSIONS:
            continue
        symbol = _canonical_symbol_from_model_file(p)
        existing = results.get(symbol)
        if existing is None or (existing.get("preferred_extension") != ".json" and p.suffix.lower() == ".json"):
            results[symbol] = {
                "symbol": symbol,
                "model_file": str(p.relative_to(ROOT_DIR)),
                "model_path": str(p),
                "model_size_bytes": p.stat().st_size,
                "modified_at": datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
                "model_extension": p.suffix.lower(),
                "preferred_extension": p.suffix.lower(),
                "importance_log_exists": False,
                "importance_log": None,
            }
    return results


def _scan_feature_logs(feature_logs_path: Path, discovered: Dict[str, Dict]) -> None:
    if not feature_logs_path.exists():
        return

    for p in feature_logs_path.iterdir():
        if not p.is_file() or p.suffix.lower() not in {".txt", ".log"}:
            continue
        symbol = p.stem.replace("_importance", "").replace("_", "/")
        match = discovered.get(symbol)
        if match is None:
            # If symbol is not present in model store yet, still include it as a reference
            discovered[symbol] = {
                "symbol": symbol,
                "model_file": None,
                "model_path": None,
                "model_size_bytes": None,
                "modified_at": None,
                "model_extension": None,
                "preferred_extension": None,
                "importance_log_exists": True,
                "importance_log": str(p.relative_to(ROOT_DIR)),
            }
        else:
            match["importance_log_exists"] = True
            match["importance_log"] = str(p.relative_to(ROOT_DIR))


def _build_summary() -> Dict:
    discovered_models = _scan_model_store(MODEL_STORE)
    _scan_feature_logs(FEATURE_LOGS, discovered_models)

    existing_summary = _load_existing_summary(SUMMARY_FILE)
    hours_of_data = existing_summary.get("hours_of_data") if existing_summary else None
    total_tokens = existing_summary.get("total_tokens") if existing_summary else len(discovered_models)

    symbols = sorted(discovered_models.keys())
    summary_records: List[Dict] = []
    for symbol in symbols:
        record = discovered_models[symbol]
        summary_records.append({
            "symbol": record["symbol"],
            "model_file": record["model_file"],
            "model_size_bytes": record["model_size_bytes"],
            "modified_at": record["modified_at"],
            "importance_log_exists": record["importance_log_exists"],
            "importance_log": record["importance_log"],
        })

    missing_models = []
    if existing_summary and isinstance(existing_summary.get("results"), list):
        expected_symbols = [item.get("symbol") for item in existing_summary["results"] if isinstance(item, dict) and item.get("symbol")]
        missing_models = sorted([sym for sym in expected_symbols if sym not in discovered_models])
    
    if not existing_summary:
        missing_models = []

    return {
        "generated_at": datetime.now().isoformat(),
        "generated_from": {
            "model_store": str(MODEL_STORE),
            "feature_logs": str(FEATURE_LOGS),
        },
        "total_models_found": len(discovered_models),
        "total_tokens_expected": total_tokens,
        "hours_of_data": hours_of_data,
        "missing_models": missing_models,
        "models": summary_records,
    }


def main() -> None:
    summary = _build_summary()
    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2))
    print(f"✅ Generated summary at {SUMMARY_FILE}")
    print(f"   Models found: {summary['total_models_found']}")
    if summary["missing_models"]:
        print(f"   Missing models: {len(summary['missing_models'])} tokens")


if __name__ == "__main__":
    main()
