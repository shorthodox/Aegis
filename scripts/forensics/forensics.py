import sys
import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb




from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.ml.predictor import Predictor
from src.ml.feature_engine import prepare_features
from temp_old_retrain_utf8 import fetch_futures_data, fetch_fear_greed, get_atr_multiplier, create_triple_barrier_labels

def main():
    symbol = 'BTC/USDT'
    hours = 7000
    
    print("Loading datasets...")
    p = Predictor(symbol)
    df = p.fetch_live_data(timeframe='1h', limit=hours)
    btc_df = p.fetch_btc_data(timeframe='1h', limit=hours)
    news_df = p.load_news_data()
    df_1d = p.fetch_live_data(timeframe='1d', limit=max(1000, int(hours / 24) + 10))
    
    funding_df, oi_df = fetch_futures_data(symbol, df)
    fg_df = fetch_fear_greed(days=700)
    
    print("Applying feature engine...")
    df_feats = prepare_features(
        df, btc_df=btc_df, news_df=news_df, add_target_flag=False, 
        df_1d=df_1d, funding_df=funding_df, oi_df=oi_df, fg_df=fg_df
    )
    
    print("Generating triple barrier labels...")
    atr_mult = get_atr_multiplier(symbol)
    labels = create_triple_barrier_labels(
        df_feats, atr_mult, max_lookahead=18,
        volatility_regime=df_feats.get('parkinson_vol'), 
        efficiency_ratio=df_feats.get('kaufman_er'),
        trend_regime=df_feats.get('macro_trend_1d'),
        macro_confluence_score=df_feats.get('macro_confluence_score')
    )
    df_feats['target'] = labels
    
    # Drop censored
    df_clean = df_feats[df_feats['target'] != -1].copy()
    
    report = []
    report.append("# AEGIS-1 BUY Precision Forensic Report\n")
    
    # Section 1: Class Distribution
    report.append("## SECTION 1 — CLASS DISTRIBUTION ANALYSIS")
    dist = df_clean['target'].value_counts().sort_index()
    report.append(f"SELL (0): {dist.get(0, 0)}")
    report.append(f"HOLD (1): {dist.get(1, 0)}")
    report.append(f"BUY (2): {dist.get(2, 0)}")
    report.append(f"Imbalance ratio BUY/HOLD: {dist.get(2, 0)/dist.get(1, 1):.3f}")
    
    # Section 2: Label Quality Audit
    report.append("\n## SECTION 2 — LABEL QUALITY AUDIT")
    report.append(f"Base ATR Multiplier used: {atr_mult}")
    hit_rate = (dist.get(0,0) + dist.get(2,0)) / len(df_clean)
    report.append(f"Total Barrier Hit Frequency: {hit_rate:.2%}")
    report.append(f"BUY vs SELL symmetry: {dist.get(2,0) / max(dist.get(0,1), 1):.2f} (1.0 is symmetric)")
    
    # Check barriers by regime
    if 'macro_confluence_score' in df_clean.columns:
        report.append("\nRegime Distribution of Labels:")
        gb = df_clean.groupby('macro_confluence_score')['target'].value_counts().unstack().fillna(0)
        report.append(gb.to_string())

    # Section 3: Lookahead Horizon
    report.append("\n## SECTION 3 — LOOKAHEAD HORIZON ANALYSIS")
    horizons = [6, 12, 18, 24, 36, 48]
    for h in horizons:
        l = create_triple_barrier_labels(
            df_feats, atr_mult, max_lookahead=h,
            volatility_regime=df_feats.get('parkinson_vol'), 
            efficiency_ratio=df_feats.get('kaufman_er'),
            trend_regime=df_feats.get('macro_trend_1d'),
            macro_confluence_score=df_feats.get('macro_confluence_score')
        )
        vc = l[l != -1].value_counts()
        report.append(f"Horizon {h}h -> BUY: {vc.get(2,0)}, HOLD: {vc.get(1,0)}, SELL: {vc.get(0,0)}")
        
    # Section 5: Feature Dominance Analysis
    report.append("\n## SECTION 5 — FEATURE DOMINANCE ANALYSIS")
    model_path = "src/ml/model_store/BTC_USDT_model.json"
    if os.path.exists(model_path):
        model = xgb.Booster()
        model.load_model(model_path)
        features = model.feature_names
        
        missing = [f for f in features if f not in df_clean.columns]
        if missing:
            report.append(f"Missing features in re-run: {missing}")
            # mock missing
            for m in missing: df_clean[m] = 0.5
            
        X = df_clean[features]
        preds = model.predict(xgb.DMatrix(X))
        
        # We need confusion matrix for regime analysis
        df_clean['pred'] = np.argmax(preds, axis=1)
        
        report.append("Confusion Matrix on entire dataset:")
        report.append(str(confusion_matrix(df_clean['target'], df_clean['pred'])))
        report.append(classification_report(df_clean['target'], df_clean['pred']))
        
        # Section 4: Regime Analysis
        report.append("\n## SECTION 4 — REGIME ANALYSIS")
        hmm_path = "src/ml/model_store/BTC_USDT_hmm.pkl"
        if os.path.exists(hmm_path):
            with open(hmm_path, 'rb') as f:
                hmm = None
            # Assuming HMM labels are somehow accessible, maybe we can mock this or check if they are in features
            pass
        else:
            report.append("HMM model not found. Using macro_confluence_score for regimes.")
            for reg, group in df_clean.groupby('macro_confluence_score'):
                prec = len(group[(group['pred'] == 2) & (group['target'] == 2)]) / max(1, len(group[group['pred'] == 2]))
                recall = len(group[(group['pred'] == 2) & (group['target'] == 2)]) / max(1, len(group[group['target'] == 2]))
                report.append(f"Regime {reg}: BUY Prec={prec:.3f}, Recall={recall:.3f} (Fired={len(group[group['pred'] == 2])})")
        
    # Write report
    with open("scripts/forensics/REPORT_DRAFT.md", "w") as f:
        f.write("\n".join(report))
        
    print("Report written to scripts/forensics/REPORT_DRAFT.md")

if __name__ == "__main__":
    main()
