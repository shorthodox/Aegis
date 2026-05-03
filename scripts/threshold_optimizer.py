

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings("ignore")

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.ml.predictor import Predictor
from src.ml.feature_engine import prepare_features, compute_atr, compute_volatility_regime

BACKTEST_DIR = Path(r"D:\Content\Animesh\bots\ai_signal_bot\logs\backtests")
BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_BALANCE = 1000
EXCHANGE_FEE_PCT = 0.001
SLIPPAGE_PCT = 0.001
MAX_HOLD_CANDLES = 48
MIN_TRADES_FOR_VALID = 10          # reduced from 20 for regime-specific
MIN_PROFIT_FACTOR = 1.2            # reduced slightly for flexibility
DATA_HOURS = 3000

THRESHOLD_BUCKETS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.75, 1.00]

MODE_CONFIG = {
    "entry_threshold_buy": 0.70,
    "entry_threshold_sell": 0.30,
    "risk_per_trade": 2.0,
    "atr_sl": 1.5,
    "atr_tp": 2.0,
}

VOLUME_WINDOW = 100                # rolling window for volume percentile
VOLUME_PERCENTILE_HIGH = 70
VOLUME_PERCENTILE_LOW = 30
VOLATILITY_HIGH_THRESH = 1.2
VOLATILITY_LOW_THRESH = 0.8

MIN_REGIME_CANDLES = 100           # reduced from 200

# ------------------------------------------------------------------
# Simulation functions (same as before)
# ------------------------------------------------------------------
def compute_cost_adjusted_expected_return(prob, atr, entry, atr_sl, atr_tp, fee, slip):
    tp_move = (atr_tp * atr) / entry
    sl_move = (atr_sl * atr) / entry
    expected_raw = prob * tp_move - (1 - prob) * sl_move
    total_costs = 2 * (fee + slip)
    return expected_raw - total_costs

def simulate_trades_for_threshold(df, min_net_profit_pct):
    buy_thresh = MODE_CONFIG["entry_threshold_buy"]
    atr_sl = MODE_CONFIG["atr_sl"]
    atr_tp = MODE_CONFIG["atr_tp"]
    risk_per_trade = MODE_CONFIG["risk_per_trade"] / 100.0
    min_net_profit = min_net_profit_pct / 100.0

    trades = []
    capital = INITIAL_BALANCE
    position_units = 0.0
    entry_price = 0.0
    entry_atr = 0.0
    direction = None

    for i in range(len(df) - MAX_HOLD_CANDLES - 1):
        row = df.iloc[i]
        prob = row['prob']
        if pd.isna(prob):
            continue

        if position_units != 0:
            exit_price = None
            for j in range(1, MAX_HOLD_CANDLES + 1):
                future = df.iloc[i + j]
                if direction == 'long':
                    if future['high'] >= entry_price + atr_tp * entry_atr and future['low'] <= entry_price - atr_sl * entry_atr:
                        exit_price = entry_price - atr_sl * entry_atr
                        break
                    elif future['high'] >= entry_price + atr_tp * entry_atr:
                        exit_price = entry_price + atr_tp * entry_atr
                        break
                    elif future['low'] <= entry_price - atr_sl * entry_atr:
                        exit_price = entry_price - atr_sl * entry_atr
                        break
            if exit_price is not None:
                exit_slipped = exit_price * (1 - SLIPPAGE_PCT)
                gross_pnl = position_units * (exit_slipped - entry_price)
                exit_fee = abs(position_units) * exit_slipped * EXCHANGE_FEE_PCT
                exit_slip = abs(position_units) * exit_slipped * SLIPPAGE_PCT
                net_pnl = gross_pnl - exit_fee - exit_slip
                trade_return = (net_pnl / (entry_price * position_units)) * 100 if (entry_price * position_units) > 0 else 0
                trades.append({"actual_return_pct": trade_return, "was_profitable": net_pnl > 0})
                capital += net_pnl
                position_units = 0
                continue

        if position_units == 0:
            if prob > buy_thresh:
                direction = 'long'
            else:
                continue

            entry_atr = row['atr_14']
            if entry_atr == 0 or np.isnan(entry_atr):
                entry_atr = row['close'] * 0.001

            expected_net = compute_cost_adjusted_expected_return(
                prob, entry_atr, row['close'],
                atr_sl, atr_tp,
                EXCHANGE_FEE_PCT, SLIPPAGE_PCT
            )
            if expected_net < min_net_profit:
                continue

            sl_distance = atr_sl * entry_atr
            risk_amount = capital * risk_per_trade
            position_units = risk_amount / sl_distance if sl_distance > 0 else 0
            max_units = capital / row['close']
            position_units = min(position_units, max_units)
            if position_units <= 0:
                continue

            entry_price = row['close']
            entry_value = position_units * entry_price
            entry_cost = entry_value * (EXCHANGE_FEE_PCT + SLIPPAGE_PCT)
            capital -= entry_cost

    return pd.DataFrame(trades) if trades else pd.DataFrame()

def find_optimal_threshold(df_regime, label):
    best_thresh = None
    best_return = -np.inf
    for thresh in THRESHOLD_BUCKETS:
        trades_df = simulate_trades_for_threshold(df_regime, thresh)
        if len(trades_df) < MIN_TRADES_FOR_VALID:
            continue
        total_return = trades_df['actual_return_pct'].sum()
        gross_profit = trades_df.loc[trades_df['was_profitable'], 'actual_return_pct'].sum()
        gross_loss = abs(trades_df.loc[~trades_df['was_profitable'], 'actual_return_pct'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999 if gross_profit > 0 else 0)
        if profit_factor >= MIN_PROFIT_FACTOR and total_return > best_return:
            best_return = total_return
            best_thresh = thresh
    if best_thresh is not None:
        print(f"      {label} -> optimal {best_thresh}% (trades={len(trades_df)}, return={best_return:.2f}%)")
    return best_thresh

def model_exists(symbol):
    model_path = Path(root_dir) / "src" / "ml" / "model_store" / f"{symbol.replace('/', '_')}_model.json"
    return model_path.exists()

def optimize_token(symbol):
    print(f"\n🔍 Processing {symbol}...")
    if not model_exists(symbol):
        print(f"   ⚠️ Model not found – skipping")
        return {"symbol": symbol, "error": "Model missing"}

    predictor = Predictor(symbol)
    df = predictor.fetch_live_data(limit=DATA_HOURS)
    if df is None or df.empty:
        return {"symbol": symbol, "error": "No data"}

    btc_df = predictor.fetch_btc_data(limit=DATA_HOURS)
    news_df = predictor.load_news_data()
    if btc_df is None or btc_df.empty:
        btc_df = pd.DataFrame({'timestamp': df['timestamp'], 'close': 0.0})
    if news_df is None or news_df.empty:
        news_df = pd.DataFrame({'timestamp': df['timestamp'], 'sentiment': 0.0})

    df = prepare_features(df, btc_df=btc_df, news_df=news_df)
    if df is None or df.empty:
        return {"symbol": symbol, "error": "Feature engineering failed"}

    df['prob'] = predictor.predict(df)
    df['atr_14'] = compute_atr(df, 14)

    # Rolling volume percentile (no lookahead)
    df['volume_rank'] = df['volume'].rolling(VOLUME_WINDOW).apply(lambda x: (x.rank(pct=True).iloc[-1]) if len(x) == VOLUME_WINDOW else 0.5, raw=False)
    df['volume_regime'] = 'normal'
    df.loc[df['volume_rank'] >= VOLUME_PERCENTILE_HIGH/100, 'volume_regime'] = 'high'
    df.loc[df['volume_rank'] <= VOLUME_PERCENTILE_LOW/100, 'volume_regime'] = 'low'

    if 'volatility_regime' not in df.columns:
        df['volatility_regime'] = compute_volatility_regime(df)
    df['volatility_label'] = 'normal'
    df.loc[df['volatility_regime'] > VOLATILITY_HIGH_THRESH, 'volatility_label'] = 'high'
    df.loc[df['volatility_regime'] < VOLATILITY_LOW_THRESH, 'volatility_label'] = 'low'

    regimes = []
    for vol_label in ['low', 'normal', 'high']:
        for vol_label2 in ['low', 'normal', 'high']:
            regime_df = df[(df['volatility_label'] == vol_label) & (df['volume_regime'] == vol_label2)]
            if len(regime_df) < MIN_REGIME_CANDLES:
                print(f"   {vol_label}/{vol_label2}: only {len(regime_df)} candles (<{MIN_REGIME_CANDLES}) – skipped")
                continue
            opt = find_optimal_threshold(regime_df, f"{vol_label}/{vol_label2}")
            if opt is not None:
                regimes.append({
                    "volatility": vol_label,
                    "volume": vol_label2,
                    "optimal_threshold_pct": opt
                })
            else:
                print(f"   {vol_label}/{vol_label2}: no valid threshold (trades <{MIN_TRADES_FOR_VALID} or PF too low)")

    # If no regime-specific thresholds, try overall dataset
    if not regimes:
        print(f"   No regime thresholds found – trying overall dataset...")
        overall_opt = find_optimal_threshold(df, "overall")
        if overall_opt is not None:
            regimes.append({
                "volatility": "any",
                "volume": "any",
                "optimal_threshold_pct": overall_opt,
                "note": "fallback (overall)"
            })

    return {"symbol": symbol, "regime_thresholds": regimes}

def run_adaptive_optimization():
    full_fleet = [
        'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
        'HYPE/USDT', 'ASTER/USDT', 'SUI/USDT', 'TAO/USDT', 'RENDER/USDT',
        'ADA/USDT', 'AVAX/USDT', 'LINK/USDT', 'TRX/USDT', 'DOT/USDT',
        'NEAR/USDT', 'MATIC/USDT', 'LTC/USDT', 'BCH/USDT', 'SHIB/USDT',
        'TON/USDT', 'ICP/USDT', 'HBAR/USDT', 'APT/USDT', 'ARB/USDT',
        'OP/USDT', 'STX/USDT', 'FIL/USDT', 'AAVE/USDT', 'VET/USDT',
        'RNDR/USDT', 'INJ/USDT', 'TIA/USDT', 'SEI/USDT', 'KAS/USDT',
        'FET/USDT', 'AGIX/USDT', 'OCEAN/USDT', 'AKT/USDT', 'THETA/USDT',
        'GRT/USDT', 'LDO/USDT', 'PYTH/USDT', 'JUP/USDT', 'ONDO/USDT',
        'PEPE/USDT', 'DOGE/USDT', 'WIF/USDT', 'FLOKI/USDT', 'BONK/USDT',
        'WLFI/USDT', 'MNT/USDT', 'ENA/USDT', 'BGB/USDT', 'PI/USDT',
        'SKY/USDT', 'TRUMP/USDT', 'NIGHT/USDT'
    ]

    print("=" * 80)
    print("📈 ADAPTIVE THRESHOLD OPTIMIZER (Improved)")
    print("=" * 80)
    print(f"   Symbols: {len(full_fleet)}")
    print(f"   Data hours: {DATA_HOURS}")
    print(f"   Min regime candles: {MIN_REGIME_CANDLES}, Min trades: {MIN_TRADES_FOR_VALID}")
    print("=" * 80)

    results = []
    for symbol in full_fleet:
        res = optimize_token(symbol)
        if "error" not in res and res.get("regime_thresholds"):
            results.append(res)
            print(f"   ✅ {symbol}: {len(res['regime_thresholds'])} thresholds found")
        else:
            print(f"   ❌ {symbol}: {res.get('error', 'No thresholds')}")

    # Save JSON
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "data_hours": DATA_HOURS,
            "min_trades": MIN_TRADES_FOR_VALID,
            "min_profit_factor": MIN_PROFIT_FACTOR,
            "min_regime_candles": MIN_REGIME_CANDLES
        },
        "results": results
    }
    out_file = BACKTEST_DIR / "adaptive_thresholds.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n📁 Results saved to {out_file}")

    # Print table
    print("\n" + "=" * 100)
    print("📊 PER‑TOKEN REGIME‑AWARE THRESHOLDS (expected_net % before entry)")
    print("=" * 100)
    print(f"{'Symbol':<12} | {'Regime (vol/vol)':<18} | {'Threshold (%)':<12}")
    print("-" * 100)
    for res in results:
        symbol = res['symbol']
        for regime in res['regime_thresholds']:
            regime_str = f"vol={regime['volatility']}, vol={regime['volume']}"
            thresh = regime['optimal_threshold_pct']
            print(f"{symbol:<12} | {regime_str:<18} | {thresh:>6.2f}%")
        print("-" * 100)

if __name__ == "__main__":
    run_adaptive_optimization()