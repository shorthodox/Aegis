import json
import sys
from pathlib import Path

def load(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception as e:
        print(f'Failed to read {path}: {e}')
        return None

def find_records(data, symbol):
    out = []
    if not data:
        return out
    for r in data.get('signals', []):
        if r.get('symbol') == symbol:
            out.append(r)
    return out

def main():
    if len(sys.argv) < 2:
        print('Usage: python scripts/tools/inspect_entry.py <SYMBOL>')
        print('Example: python scripts/tools/inspect_entry.py "ETC/USDT"')
        sys.exit(1)
    symbol = sys.argv[1]
    repo_root = Path(__file__).resolve().parents[2]
    candidate_paths = [
        repo_root / 'data' / 'track_record.json',
        repo_root / 'web' / 'trader_track_record.json',
    ]
    found = False
    for p in candidate_paths:
        d = load(p)
        if not d:
            continue
        recs = find_records(d, symbol)
        if not recs:
            continue
        print(f'Found {len(recs)} records for {symbol} in {p}')
        for r in recs:
            print('---')
            print(json.dumps(r, indent=2, default=str))
        found = True
    if not found:
        print(f'No records for {symbol} in {candidate_paths}')

if __name__ == '__main__':
    main()
