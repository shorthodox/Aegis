#!/usr/bin/env python3
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.ml.predictor import Predictor
from src.ml.feature_engine import prepare_features, compute_atr
from src.ml.feature_health import FeatureHealthManager
from scripts.retrain_model import FEATURE_ADDONS, create_triple_barrier_labels, CENSORED, FEE_ROUNDTRIP

def run_feature_scoring():
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
    hours = 1200
    
    print(f"Starting fleet feature selection over {len(symbols)} major symbols...")
    
    feature_data = {f: [] for f in FEATURE_ADDONS}
    
    for symbol in symbols:
        print(f"Fetching and preparing data for {symbol}...")
        p = Predictor(symbol)
        df = p.fetch_live_data(timeframe='1h', limit=hours)
        if df is None or df.empty:
            print(f"Failed to fetch data for {symbol}")
            continue
            
        # Add basic context
        btc_df = None if symbol == "BTC/USDT" else Predictor("BTC/USDT").fetch_live_data(timeframe='1h', limit=hours)
        df = prepare_features(df, btc_df=btc_df, add_target_flag=False)
        if df is None or df.empty:
            continue
            
        df["_atr"] = compute_atr(df, period=14).values
        labels = create_triple_barrier_labels(
            df, atr_multiplier=1.5, max_lookahead=24,
            volatility_regime=df.get('volatility_regime'),
            efficiency_ratio=df.get('efficiency_ratio_10'),
            trend_regime=df.get('trend_regime'),
            macro_confluence_score=df.get('macro_confluence_score')
        )
        df['target'] = labels.astype(int).values
        df = df[df['target'] != CENSORED].reset_index(drop=True)
        
        N = len(df)
        test_start = N - int(N * 0.20)
        train_pool = df.iloc[:test_start].reset_index(drop=True)
        holdout = df.iloc[test_start:].reset_index(drop=True)
        
        # Target returns (expectancy proxy)
        close_tp = train_pool['close'].to_numpy()
        atr_tp = train_pool['_atr'].to_numpy()
        y_tp = train_pool['target'].to_numpy()
        
        rets_tp = []
        for i in range(len(train_pool)):
            yt_val = int(y_tp[i])
            b_val = 1.5 * atr_tp[i] / close_tp[i] if close_tp[i] > 0 else 0.015
            buy_ret = (b_val - FEE_ROUNDTRIP) if yt_val == 2 else (-b_val - FEE_ROUNDTRIP) if yt_val == 0 else -FEE_ROUNDTRIP
            rets_tp.append(buy_ret)
        rets_tp = np.array(rets_tp)
        
        # Analyze drift
        fhm = FeatureHealthManager()
        available_features = [f for f in FEATURE_ADDONS if f in train_pool.columns and f in holdout.columns]
        fhm.analyze_drift(train_pool, holdout, available_features)
        
        # Compute correlations
        for f in FEATURE_ADDONS:
            if f in train_pool.columns and train_pool[f].std() > 1e-8:
                corr = float(train_pool[f].corr(pd.Series(rets_tp), method='spearman'))
                psi = float(fhm.drift_scores.get(f, {}).get('psi', 0.0))
                ks = float(fhm.drift_scores.get(f, {}).get('ks', 0.0))
                
                feature_data[f].append({
                    "symbol": symbol,
                    "corr": corr,
                    "psi": psi,
                    "ks": ks
                })
                
    # Compile scoring
    rows = []
    for f in FEATURE_ADDONS:
        records = feature_data[f]
        if not records:
            continue
            
        corrs = [r["corr"] for r in records]
        psis = [r["psi"] for r in records]
        kss = [r["ks"] for r in records]
        
        # Predictive power: median absolute correlation
        predictive_power = float(np.median([abs(c) for c in corrs if pd.notna(c)])) if corrs else 0.0
        
        # Stability: 1 - median KS
        stability = float(max(0.0, 1.0 - np.median(kss))) if kss else 0.0
        
        # Drift resistance: 1 - min(1.0, median(psi) / 2.0)
        drift_resistance = float(max(0.0, 1.0 - min(1.0, np.median(psis) / 2.0))) if psis else 0.0
        
        # Consistency: sign agreement across symbols
        signs = [np.sign(c) for c in corrs if pd.notna(c) and abs(c) > 1e-4]
        consistency = float(max(signs.count(1), signs.count(-1)) / len(signs)) if signs else 0.0
        
        # Coverage
        coverage = float(len(records) / len(symbols))
        
        # Composite score
        score = (
            0.40 * predictive_power +
            0.30 * stability +
            0.20 * consistency +
            0.10 * drift_resistance
        )
        
        # Decision rules:
        # PSI > 1.0 on most symbols -> REMOVE
        # Fail (missing/no variance) on > 50% of tokens -> REMOVE
        # Median PSI > 1.0 -> REMOVE
        median_psi = float(np.median(psis)) if psis else 9.9
        
        decision = "KEEP"
        if median_psi > 1.0:
            decision = "REMOVE (high drift)"
        elif coverage < 0.50:
            decision = "REMOVE (low coverage)"
        elif score < 0.10:
            decision = "REMOVE (low score)"
            
        rows.append({
            "feature": f,
            "stability_score": round(stability, 4),
            "drift_score": round(median_psi, 4),
            "predictive_power": round(predictive_power, 4),
            "cross_token_consistency": round(consistency, 4),
            "token_coverage": round(coverage, 4),
            "score": round(score, 4),
            "keep_decision": decision
        })
        
    df_scores = pd.DataFrame(rows).sort_values("score", ascending=False)
    
    # Save files
    csv_path_root = Path(root_dir) / "fleet_feature_score.csv"
    csv_path_artifacts = Path(root_dir) / "C:/Users/bkukr/.gemini/antigravity-ide/brain/5c43f861-2b52-4bd7-ac90-68e2709542f9/fleet_feature_score.csv"
    
    df_scores.to_csv(csv_path_root, index=False)
    try:
        df_scores.to_csv(csv_path_artifacts, index=False)
    except Exception:
        pass
        
    print(f"Feature scores generated and saved to {csv_path_root}")
    print(df_scores.head(15))

if __name__ == "__main__":
    run_feature_scoring()
