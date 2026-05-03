#!/usr/bin/env python3
# scripts/rule_based_strategy.py
# Aegis-1 Rule‑Based Strategy – Uses indicators from feature_engine.py

import pandas as pd
import numpy as np
import ccxt
import sys
import os
import time
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# --- Path fix ---
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.ml.feature_engine import prepare_features

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
SYMBOL = "BTC/USDT"
INITIAL_BALANCE = 1000.0
DAYS = 365
SLIPPAGE_PCT = 0.001      # 0.1%
EXCHANGE_FEE_PCT = 0.001  # 0.1%
MAX_LEVERAGE = 1.0        # spot only
MIN_SL_DISTANCE_PCT = 0.005
LIQUIDATION_THRESHOLD = 0.01
TP_RATIO = 1.5            # asymmetric RR

# Minimum hourly bars required
MIN_HOURLY_BARS = 252

# -------------------------------------------------------------------
# Data fetching (with pagination and resampling)
# -------------------------------------------------------------------
def fetch_ohlcv(symbol: str, limit: int = 5000) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data with pagination, return hourly data."""
    config: Dict[str, Any] = {
        'enableRateLimit': True,
        'defaultType': 'spot',
        'options': {
            'adjustForTimeDifference': True,
            'recvWindow': 60000,
        }
    }
    # ccxt Exchange constructors accept a single config dict argument.
    exchange = ccxt.binance(config)

    # Use 1h timeframe directly
    timeframe = '1h'
    all_bars = []
    remaining = limit
    since = int(time.time() * 1000) - (limit * 60 * 60 * 1000)
    chunk_size = 1000

    while remaining > 0:
        try:
            bars = exchange.fetch_ohlcv(symbol, timeframe, since, min(chunk_size, remaining))
            if not bars:
                break
            all_bars.extend(bars)
            remaining -= len(bars)
            since = bars[-1][0] + 1
            time.sleep(0.5)
        except Exception as e:
            print(f"⚠️ Pagination error for {symbol}: {e}")
            break

    if not all_bars:
        return None

    df = pd.DataFrame(all_bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    # Already hourly, no resampling needed
    if len(df) > limit:
        df = df.iloc[-limit:]
    df.reset_index(inplace=True)

    if len(df) < MIN_HOURLY_BARS:
        print(f"⚠️ Insufficient data for {symbol}: got {len(df)} bars, need {MIN_HOURLY_BARS}.")
        return None
    return df

# -------------------------------------------------------------------
# Rule‑based strategy logic
# -------------------------------------------------------------------
def generate_signal(row: pd.Series) -> int:
    """
    Returns 1 for BUY, 0 for HOLD.
    Rules (example – adjust to your own logic):
      - Trend: price above EMA200 and EMA50 > EMA200
      - Momentum: RSI > 50 and MACD histogram > 0
      - Volatility: ATR not extremely high (volatility_regime < 1.5)
      - Volume: volume_zscore > 0.5 (increasing volume)
    """
    # Trend conditions
    price_above_ema200 = row['close'] > row['ema_200']
    ema50_above_ema200 = row['ema_50'] > row['ema_200']
    trend_ok = price_above_ema200 and ema50_above_ema200

    # Momentum conditions
    rsi_ok = row['rsi_14'] > 50
    macd_ok = row['macd_hist'] > 0

    # Volatility condition (avoid extreme volatility)
    vol_ok = row['volatility_regime'] < 1.5

    # Volume condition (increasing volume)
    vol_zscore_ok = row['volume_zscore'] > 0.5

    if trend_ok and rsi_ok and macd_ok and vol_ok and vol_zscore_ok:
        return 1
    else:
        return 0

# -------------------------------------------------------------------
# Backtest simulation (equity‑based)
# -------------------------------------------------------------------
def run_backtest(df: pd.DataFrame) -> Dict[str, Any]:
    capital = INITIAL_BALANCE
    position_units = 0.0
    entry_price = 0.0
    entry_atr = 0.0
    entry_idx = 0
    trades = 0
    wins = 0
    losses = 0
    gross_profit = 0.0
    gross_loss = 0.0
    peak_capital = capital
    max_drawdown = 0.0
    equity_curve = [(df['timestamp'].iloc[0], capital)]

    for i in range(1, len(df)):
        row = df.iloc[i]
        timestamp = row['timestamp']

        if capital < INITIAL_BALANCE * LIQUIDATION_THRESHOLD:
            print(f"⚠️ LIQUIDATION: equity fell below 1% of initial")
            break

        # --- Exit logic (if in position) ---
        if position_units != 0:
            sl = entry_price - 1.5 * row['atr_14']      # ATR multiplier = 1.5
            tp = entry_price + 1.5 * TP_RATIO * row['atr_14']
            exit_price = None

            # Break‑even trigger
            if row['close'] >= entry_price + entry_atr:
                sl = entry_price

            if row['low'] <= sl:
                exit_price = sl
            elif row['high'] >= tp:
                exit_price = tp
            elif (i - entry_idx) >= 24:   # max hold 24h
                exit_price = row['close']

            if exit_price is not None:
                # Apply slippage on exit
                exit_price_with_slippage = exit_price * (1 - SLIPPAGE_PCT)
                gross_pnl = position_units * (exit_price_with_slippage - entry_price)
                exit_fee = abs(position_units) * exit_price_with_slippage * EXCHANGE_FEE_PCT
                exit_slippage = abs(position_units) * exit_price_with_slippage * SLIPPAGE_PCT
                net_pnl = gross_pnl - exit_fee - exit_slippage

                if net_pnl < 0 and abs(net_pnl) > capital:
                    net_pnl = -capital
                capital += net_pnl
                position_units = 0
                trades += 1
                if net_pnl > 0:
                    wins += 1
                    gross_profit += net_pnl
                else:
                    losses += 1
                    gross_loss += abs(net_pnl)

                if capital > peak_capital:
                    peak_capital = capital
                dd = (peak_capital - capital) / peak_capital * 100 if peak_capital > 0 else 0
                if dd > max_drawdown:
                    max_drawdown = dd
                equity_curve.append((timestamp, capital))
                continue

        # --- Entry logic ---
        if position_units == 0:
            signal = generate_signal(row)
            if signal == 1:
                entry_price = row['close']
                entry_atr = row['atr_14']
                if entry_atr == 0 or np.isnan(entry_atr):
                    entry_atr = entry_price * 0.001
                # Risk per trade = 2% of capital
                risk_pct = 2.0
                sl_distance = 1.5 * entry_atr
                min_sl = entry_price * MIN_SL_DISTANCE_PCT
                if sl_distance < min_sl:
                    sl_distance = min_sl
                risk_amount = capital * (risk_pct / 100)
                position_units = risk_amount / sl_distance
                max_units = capital / entry_price   # spot only
                position_units = min(position_units, max_units)
                if position_units <= 0:
                    continue
                trade_value = position_units * entry_price
                entry_slippage = trade_value * SLIPPAGE_PCT
                entry_fee = trade_value * EXCHANGE_FEE_PCT
                total_cost = entry_slippage + entry_fee
                if total_cost > capital:
                    continue
                capital -= total_cost
                entry_idx = i
                equity_curve.append((timestamp, capital))

    final_return = (capital / INITIAL_BALANCE - 1) * 100
    win_rate = (wins / trades * 100) if trades > 0 else 0
    profit_factor = gross_profit / max(gross_loss, 0.01)

    print(f"\n📊 RESULTS for {SYMBOL}")
    print(f"   Trades: {trades}")
    print(f"   Return: {final_return:.2f}%")
    print(f"   Max Drawdown: {max_drawdown:.2f}%")
    print(f"   Win Rate: {win_rate:.1f}%")
    print(f"   Profit Factor: {profit_factor:.2f}")

    return {
        "return_pct": final_return,
        "max_drawdown_pct": max_drawdown,
        "win_rate": win_rate,
        "total_trades": trades,
        "profit_factor": profit_factor,
        "equity_curve": equity_curve
    }

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    # Fetch data for the main symbol
    print(f"📡 Fetching data for {SYMBOL}...")
    df = fetch_ohlcv(SYMBOL, limit=24 * DAYS)
    if df is None or df.empty:
        print("❌ Failed to fetch data.")
        return

    # Fetch BTC data for anchor features
    print("📡 Fetching BTC data...")
    btc_df = fetch_ohlcv('BTC/USDT', limit=24 * DAYS)
    if btc_df is None or btc_df.empty:
        print("⚠️ BTC data missing. Creating dummy.")
        btc_df = pd.DataFrame({'timestamp': df['timestamp'], 'close': 0.0})
        btc_df['open'] = btc_df['high'] = btc_df['low'] = btc_df['close']
        btc_df['volume'] = 0

    # Load news (if exists)
    news_path = Path(root_dir) / "data" / "news_data.json"
    news_df = None
    if news_path.exists():
        try:
            with open(news_path, "r") as f:
                data = json.load(f)
            news_df = pd.DataFrame(data)
            news_df['timestamp'] = pd.to_datetime(news_df['timestamp'])
            news_df = news_df.sort_values('timestamp')
            if 'sentiment' in news_df.columns:
                news_df = news_df[['timestamp', 'sentiment']]
            else:
                news_df = None
        except Exception as e:
            print(f"⚠️ Could not load news: {e}")

    # Feature engineering
    print("🛠 Applying feature engineering...")
    df = prepare_features(df, btc_df=btc_df, news_df=news_df)
    if df is None or df.empty:
        print("❌ Feature engineering failed.")
        return

    # Run backtest
    results = run_backtest(df)

    # Save equity curve
    curve_file = Path(root_dir) / "logs" / f"{SYMBOL.replace('/', '_')}_rule_based.csv"
    curve_file.parent.mkdir(exist_ok=True)
    eq_df = pd.DataFrame(results["equity_curve"], columns=["timestamp", "balance"])
    eq_df.to_csv(curve_file, index=False)
    print(f"\n💾 Equity curve saved to {curve_file}")

if __name__ == "__main__":
    main()