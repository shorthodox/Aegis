#!/usr/bin/env python3
"""Non-interactive test to exercise MarketDataFetcher.start/stop for resource cleanup.

This loads the `live_engine.py` module by file path so imports work even when
`scripts` is not a package.
"""
import asyncio
import importlib.util
import sys
from pathlib import Path

module_path = Path(__file__).resolve().parent / 'live_engine.py'
spec = importlib.util.spec_from_file_location('live_engine_module', str(module_path))
live_engine = importlib.util.module_from_spec(spec)
sys.modules['live_engine_module'] = live_engine
spec.loader.exec_module(live_engine)

MarketDataFetcher = live_engine.MarketDataFetcher

async def main():
    fetcher = MarketDataFetcher()
    try:
        await fetcher.start()
    except Exception as e:
        print('Fetcher start error:', e)
    finally:
        await fetcher.stop()


if __name__ == '__main__':
    asyncio.run(main())
