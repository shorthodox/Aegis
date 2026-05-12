import sys, os, time
import xgboost as xgb
import numpy as np
import pandas as pd
from pathlib import Path

# --- THE PATH FIX ---
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path: sys.path.insert(0, root_dir)

from src.ml.feature_engine import prepare_features
from src.ml.predictor import Predictor

# 🏆 THE 2026 PRO LIST (Top 100 excluding Stablecoins)
SYMBOLS_TO_TRAIN = [
    'BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'BNB/USDT', 'SOL/USDT', 'TRX/USDT', 
    'DOGE/USDT', 'HYPE/USDT', 'ADA/USDT', 'BCH/USDT', 'LINK/USDT', 'XMR/USDT', 
    'XLM/USDT', 'CC/USDT', 'ZEC/USDT', 'LTC/USDT', 'AVAX/USDT', 'HBAR/USDT', 
    'SUI/USDT', 'SHIB/USDT', 'TON/USDT', 'CRO/USDT', 'TAO/USDT', 'WLFI/USDT', 
    'MNT/USDT', 'DOT/USDT', 'UNI/USDT', 'PI/USDT', 'SKY/USDT', 'OKB/USDT', 
    'NEAR/USDT', 'ASTER/USDT', 'PEPE/USDT', 'AAVE/USDT', 'ICP/USDT', 'ETC/USDT', 
    'BGB/USDT', 'ONDO/USDT', 'DEXE/USDT', 'KCS/USDT', 'ENA/USDT', 'POL/USDT', 
    'ALGO/USDT', 'KAS/USDT', 'RENDER/USDT', 'ATOM/USDT', 'QNT/USDT', 'WLD/USDT', 
    'GT/USDT', 'MORPHO/USDT', 'ARB/USDT', 'APT/USDT', 'FIL/USDT', 'FLR/USDT', 
    'TRUMP/USDT', 'JUP/USDT', 'NIGHT/USDT', 'SUN/USDT'
]

TARGET_LOOKAHEAD = 24


def train_fleet(timeframe='5m'):
    print(f"🚀 --- STARTING PRO BATCH TRAINING: {len(SYMBOLS_TO_TRAIN)} TOKENS ---")
    model_store = Path(root_dir) / "src" / "ml" / "model_store"
    model_store.mkdir(parents=True, exist_ok=True)

    for i, symbol in enumerate(SYMBOLS_TO_TRAIN):
        print(f"\n[{i+1}/{len(SYMBOLS_TO_TRAIN)}] 🧠 Training {symbol}...")
        try:
            p = Predictor(symbol)
            df = p.fetch_live_data(timeframe=timeframe, limit=5000)
            if df is None:
                raise ValueError(f"No data fetched for {symbol}")

            df = prepare_features(df, add_target_flag=True, forward_hours=TARGET_LOOKAHEAD)
            if df is None:
                raise ValueError(f"Failed to prepare features for {symbol}")

            df = df.dropna(subset=['target']).reset_index(drop=True)

            X = df.drop(columns=['timestamp', 'target'], errors='ignore')
            y = df['target'].astype(int)
            
            # ✅ Robust data validation before training
            if not isinstance(X, pd.DataFrame):
                raise ValueError("X is not a DataFrame")
            if X.isnull().any().any():
                raise ValueError("X contains NaN values")
            if not all(np.isfinite(X.values.flatten())):
                raise ValueError("X contains infinite values")

            # Using your 72.67% Optuna Parameters updated for multiclass with inf handling
            model = xgb.XGBClassifier(
                n_estimators=186,
                max_depth=4,
                learning_rate=0.1199,
                subsample=0.838,
                colsample_bytree=0.765,
                gamma=0.3506,
                objective='multi:softprob',
                num_class=3,
                eval_metric='mlogloss',
                tree_method='hist',  # More stable with missing values
                missing=np.nan,  # Tell XGBoost how to handle missing values
                verbosity=0
            )
            
            model.fit(X, y)
            
            # Save to model_store as JSON for fleet consistency
            filename = f"{symbol.replace('/', '_')}_model.json"
            model_path = model_store / filename
            model.save_model(str(model_path))
            
            print(f"✅ Saved: {filename}")
            
            # 🛡️ API Protection Delay
            time.sleep(1.2)

        except Exception as e:
            print(f"⚠️ Failed to train {symbol}: {e}")

if __name__ == "__main__":
    # You can change the timeframe here depending on your strategy
    train_fleet(timeframe='5m')