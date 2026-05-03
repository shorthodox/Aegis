import sys
import os
from pathlib import Path

# --- THE PATH FIX (MUST BE FIRST) ---
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# --- NOW THE REST OF THE IMPORTS ---
import optuna
import xgboost as xgb
import pandas as pd
from src.ml.feature_engine import prepare_features
from src.ml.predictor import Predictor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def objective(trial):
    # 1. Define the range for 'perfect' settings
    param = {
        'verbosity': 0,
        'objective': 'binary:logistic',
        # Optuna will try different combinations of these:
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
    }

    # 2. Fetch data (Using your Predictor)
    p = Predictor("BTC/USDT")
    # Fetching a larger chunk (5000 hours) gives Optuna better "training ground"
    df = p.fetch_live_data(timeframe='1h', limit=5000)
    if df is None:
        raise ValueError("Failed to fetch live data; df is None")
    df = prepare_features(df)
    
    X = df.drop(columns=['timestamp', 'target'], errors='ignore')
    y = df['target']
    
    # Split: Train on 80%, Test on 20%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 3. Train the trial model
    model = xgb.XGBClassifier(**param)
    model.fit(X_train, y_train)
    
    # 4. We want to maximize Accuracy
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    
    return float(accuracy)

if __name__ == "__main__":
    print("🚀 Starting Hyperparameter Optimization (50 Trials)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50) 

    print("\n🏆 OPTIMIZATION COMPLETE!")
    print(f"Best Accuracy Found: {study.best_value:.4f}")
    print("-" * 30)
    print("USE THESE PARAMETERS IN YOUR TRAIN SCRIPT:")
    for key, value in study.best_params.items():
        print(f"'{key}': {value},")
    print("-" * 30)