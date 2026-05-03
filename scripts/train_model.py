import sys, os, time
import xgboost as xgb
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

def train_fleet(timeframe='5m'):
    print(f"🚀 --- STARTING PRO BATCH TRAINING: {len(SYMBOLS_TO_TRAIN)} TOKENS ---")
    
    for i, symbol in enumerate(SYMBOLS_TO_TRAIN):
        print(f"\n[{i+1}/{len(SYMBOLS_TO_TRAIN)}] 🧠 Training {symbol}...")
        try:
            p = Predictor(symbol)
            df = p.fetch_live_data(timeframe=timeframe, limit=5000)
            df = prepare_features(df)
            
            X = df.drop(columns=['timestamp', 'target'], errors='ignore')
            y = df['target']

            # Using your 72.67% Optuna Parameters
            model = xgb.XGBClassifier(
                n_estimators=186, max_depth=4, learning_rate=0.1199,
                subsample=0.838, colsample_bytree=0.765, gamma=0.3506,
                objective='binary:logistic', eval_metric='logloss'
            )
            
            model.fit(X, y)
            
            # Save to model_store
            filename = f"{symbol.replace('/', '_')}_model.bin"
            model_path = os.path.join(root_dir, "src", "ml", "model_store", filename)
            model.save_model(model_path)
            
            print(f"✅ Saved: {filename}")
            
            # 🛡️ API Protection Delay
            time.sleep(1.2) 

        except Exception as e:
            print(f"⚠️ Failed to train {symbol}: {e}")

if __name__ == "__main__":
    # You can change the timeframe here depending on your strategy
    train_fleet(timeframe='5m')