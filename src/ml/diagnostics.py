from pathlib import Path
import pandas as pd
import json

ROOT = Path(__file__).resolve().parents[2]
DIAG_DIR = ROOT / 'artifacts' / 'diagnostics'
DIAG_DIR.mkdir(parents=True, exist_ok=True)


def save_top_features(symbol: str, importance: dict, n: int = 50):
    out = DIAG_DIR / f"{symbol.replace('/','_')}_top_features.csv"
    df = pd.DataFrame(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:n], columns=['feature', 'gain'])
    df.to_csv(out, index=False)
    return str(out)


def save_feature_health_report(symbol: str, report: dict):
    out = DIAG_DIR / f"{symbol.replace('/','_')}_feature_health.json"
    with open(out, 'w') as f:
        json.dump(report, f, indent=2)
    return str(out)
