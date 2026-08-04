"""Boot configuration: which tokens the engine will actually run.

Reads the model store and returns the tradeable fleet plus run
parameters. A token is included on the trainer's say-so — see
engine/contract.py for why that decision belongs to the sidecar and not
to the engine.

Extracted verbatim from the single-file live_engine.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import List
import json

from scripts.engine.config import MODEL_STORE
from scripts.engine.models import TokenConfig

def automated_setup(_: Path, args: Any):
    """
    Scan MODEL_STORE for all available symbols (up to 60).
    Binary dual-model pairs (*_model_buy.json + *_model_sell.json) are auto-tradeable.
    Legacy single-model symbols use the tradeable flag from *_meta.json.
    """
    tradeable_configs:     List[TokenConfig] = []
    non_tradeable_configs: List[TokenConfig] = []

    if MODEL_STORE.exists():
        # Detect binary dual-model pairs (new training pipeline) — auto-tradeable
        binary_syms: set = set()
        for buy_file in MODEL_STORE.glob('*_model_buy.json'):
            base = buy_file.name.replace('_model_buy.json', '')
            if (MODEL_STORE / f'{base}_model_sell.json').exists():
                sym = base.replace('_', '/', 1)
                binary_syms.add(sym)

        seen: set = set()
        # Add binary pairs first (highest priority — directly tradeable)
        for sym in sorted(binary_syms):
            seen.add(sym)
            tradeable_configs.append(TokenConfig(symbol=sym))

        # Scan meta.json for any remaining legacy symbols
        meta_files = sorted(MODEL_STORE.glob('*_meta.json'),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        for meta_file in meta_files:
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                sym = meta.get('symbol', '')
                if not sym or sym in seen:
                    continue
                seen.add(sym)
                tc = TokenConfig(symbol=sym)
                if meta.get('tradeable', False):
                    tradeable_configs.append(tc)
                else:
                    non_tradeable_configs.append(tc)
            except Exception:
                pass

    TARGET    = 60
    configs   = tradeable_configs[:TARGET]
    remaining = TARGET - len(configs)
    if remaining > 0:
        configs += non_tradeable_configs[:remaining]

    if not configs:
        print('[automated_setup] No models found — falling back to BTC/USDT.')
        configs = [TokenConfig(symbol='BTC/USDT')]

    capital      = float(getattr(args, 'capital',      10_000.0))
    max_pos      = float(getattr(args, 'max_position',  1_000.0))
    scan_seconds = int(getattr(args,   'scan_seconds',    300))
    proxy        = getattr(args, 'proxy', None)

    t = len(tradeable_configs[:TARGET])
    m = len(configs) - t
    print(f'[automated_setup] {len(configs)} symbols ({t} tradeable + {m} monitor-only) | '
          f'capital={capital} | max_pos={max_pos} | scan={scan_seconds}s')
    return configs, capital, max_pos, scan_seconds, proxy
