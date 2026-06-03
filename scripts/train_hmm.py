#!/usr/bin/env python3
"""
train_hmm.py — Standalone HMM regime model trainer
====================================================
Trains a 7-state GaussianHMM for every symbol that has a trained XGBoost model
in model_store.  Run this ONCE after retraining XGBoost models, or whenever you
want to refresh HMM regime intelligence without a full retrain.

Usage
-----
    python -m scripts.train_hmm                      # all symbols
    python -m scripts.train_hmm --symbol BTC/USDT    # single symbol
    python -m scripts.train_hmm --hours 5000         # custom history length

The script is non-destructive: if an HMM model already exists and the new
training would fail, the old model is kept intact.
"""
import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.ml.hmm_regime import HMMRegimeEngine, MODEL_STORE


def train_one(symbol: str, hours: int = 5000) -> bool:
    print(f"\n{'─'*50}\nHMM training: {symbol}\n{'─'*50}")
    try:
        from src.ml.predictor import Predictor
        p = Predictor(symbol)
        print(f"Fetching {hours}h of data…")
        df = p.fetch_live_data(timeframe='1h', limit=hours)
        if df is None or df.empty:
            print(f"  No data for {symbol}")
            return False

        from src.ml.feature_engine import prepare_features
        btc_df = None
        try:
            btc_df = p.fetch_btc_data(timeframe='1h', limit=hours)
        except Exception:
            pass
        df = prepare_features(df, btc_df=btc_df, add_target_flag=False)
        if df is None or df.empty:
            print(f"  Feature engineering failed for {symbol}")
            return False

        engine = HMMRegimeEngine(symbol)
        if engine.train(df):
            engine.save()
            return True
        return False
    except Exception as e:
        print(f"  Error for {symbol}: {type(e).__name__}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Train HMM regime models')
    parser.add_argument('--symbol', type=str, default=None,
                        help='Single symbol (default: all model_store symbols)')
    parser.add_argument('--hours', type=int, default=5000,
                        help='Hours of OHLCV data to use (default: 5000)')
    args = parser.parse_args()

    if args.symbol:
        symbols = [args.symbol]
    else:
        # Discover from model_store
        symbols = []
        for meta_file in sorted(MODEL_STORE.glob('*_meta.json')):
            try:
                meta = json.loads(meta_file.read_text())
                sym  = meta.get('symbol', '')
                if sym:
                    symbols.append(sym)
            except Exception:
                pass
        if not symbols:
            print("No symbols found in model_store. Run retrain_model.py first.")
            return
        print(f"Found {len(symbols)} symbols in model_store.")

    ok = failed = 0
    for sym in symbols:
        if train_one(sym, args.hours):
            ok += 1
        else:
            failed += 1

    print(f"\n{'='*50}")
    print(f"HMM training complete: {ok} OK, {failed} failed")
    print(f"Models saved to: {MODEL_STORE}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
