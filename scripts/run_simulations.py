import sys
import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
from pathlib import Path

# Add project root to path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.ml.predictor import Predictor
from src.trading.edge_engine import EdgeScoringEngine
from scripts.retrain_model import (
    TEST_FRAC, EMBARGO, CENSORED, MAX_LOOKAHEAD,
    compute_soft_confluence_features, compute_atr, backtest,
    create_triple_barrier_labels
)

def run():
    symbol = "BTC/USDT"
    base = symbol.replace("/", "_")
    model_store = Path(root_dir) / "src" / "ml" / "model_store"
    
    # Load sidecar meta
    sidecar_path = model_store / f"{base}_meta.json"
    if not sidecar_path.exists():
        print(f"Error: Sidecar meta not found at {sidecar_path}")
        return
    meta = json.loads(sidecar_path.read_text())
    
    # Load predictor
    p = Predictor(symbol, use_clean_model=True)
    
    # Fetch 8760h of data (1 year)
    print("Fetching historical data...")
    df = p.get_features_with_context(hours=8760)
    if df is None or df.empty:
        print("Error: Could not fetch data.")
        return
    df = df.reset_index(drop=True)
    
    # Compute soft confluence and _atr
    try:
        from scripts.retrain_model import compute_soft_confluence_features
        sc = compute_soft_confluence_features(df)
        for col in sc.columns:
            df[col] = sc[col].values
    except Exception as e:
        print("Confluence feature compilation failed, using existing columns:", e)
        
    df["_atr"] = compute_atr(df, period=14).values
    
    # Compute target labels
    print("Computing target labels...")
    atr_mult = float(meta.get("atr_multiplier", 1.5))
    # Lookahead: load from optimizer params if available, else default to 48
    token_lookahead = int(meta.get("lookahead", MAX_LOOKAHEAD))
    
    labels = create_triple_barrier_labels(
        df, atr_multiplier=atr_mult, max_lookahead=token_lookahead,
        volatility_regime=df['volatility_regime'],
        efficiency_ratio=df['efficiency_ratio_10'],
        trend_regime=df['trend_regime'],
        macro_confluence_score=df.get('macro_confluence_score'),
    )
    df['target'] = labels.astype(int).values
    df = df[df['target'] != CENSORED].reset_index(drop=True)
    
    # Split into train pool and holdout
    N = len(df)
    test_start = N - int(N * TEST_FRAC)
    holdout = df.iloc[test_start:].reset_index(drop=True)
    print(f"Total usable bars: {N} | Holdout bars: {len(holdout)}")
    
    # Generate predictions on holdout
    print("Generating predictions on holdout...")
    feature_cols = meta.get("feature_cols")
    X_test = holdout[feature_cols]
    
    proba = p.predict_proba(holdout)
    prop_h = np.where(proba[:, 2] >= proba[:, 0], 2, 0)
    y_test = holdout['target'].to_numpy().astype(int)
    
    if p.meta_model is not None:
        from scripts.retrain_model import build_meta_X
        meta_prob_h = p.meta_model.predict(xgb.DMatrix(build_meta_X(X_test, proba)))
    else:
        meta_prob_h = proba.max(axis=1)
        
    # Apply AEGIS calibration
    if getattr(p, 'aegis_state', None) is not None:
        mcf = p.aegis_state.get('mcf')
        cre = p.aegis_state.get('cre')
        if mcf:
            meta_prob_h = mcf.calibrate(meta_prob_h)
        if cre:
            meta_prob_h = cre.adjust_confidence_array(meta_prob_h)
            
    # Calculate barrier fraction
    close_arr = holdout['close'].to_numpy()
    atr_arr = holdout['_atr'].to_numpy()
    barrier_frac = np.divide(atr_mult * atr_arr, close_arr, out=np.zeros(len(holdout)), where=close_arr != 0)
    
    # ── SIMULATION A: No Meta Gate (Only Technical Filters) ──
    print("\n--- Running Simulation A: No Meta Gate (Technical Filters Only) ---")
    if "prc_total" in holdout.columns:
        prc_val = holdout["prc_total"].values
        quality = np.where(prop_h == 2, prc_val * 100.0, (1.0 - prc_val) * 100.0)
    else:
        quality = np.full(len(holdout), 60.0)
    quality_pass = quality >= 55.0
    
    # HMM pass
    hmm_pass = np.ones(len(holdout), dtype=bool)
    if 'hmm_regime' in holdout.columns:
        hmm_pass = (holdout['hmm_regime'] != 'CHOPPY').to_numpy()
        
    # Confluence pass
    conf_pass = np.ones(len(holdout), dtype=bool)
    if 'prc_total' in holdout.columns:
        prc = holdout['prc_total'].values
        conf_pass = np.where(prop_h == 2, prc > 0.52, prc < 0.48)
        
    # Drift pass
    drift_pass = np.ones(len(holdout), dtype=bool)
    
    # Base directional proposals
    primary_map = proba.argmax(1)
    dir_mask = primary_map != 1
    
    fire_a = dir_mask & quality_pass & hmm_pass & conf_pass & drift_pass
    bt_a = backtest(fire_a, prop_h, y_test, barrier_frac)
    report_bt("Simulation A (No Meta Gate)", fire_a, prop_h, y_test, bt_a)
    
    # ── SIMULATION B: Current Meta Gate (Edge Score >= 55.0) ──
    print("\n--- Running Simulation B: Current Meta Gate (Edge Score >= 55.0) ---")
    edge_buy_b = EdgeScoringEngine.compute_edge_batch(holdout, meta_prob_h, 'BUY').to_numpy()
    edge_sell_b = EdgeScoringEngine.compute_edge_batch(holdout, meta_prob_h, 'SELL').to_numpy()
    
    # Print statistics of edge scores where prop_h == 2 and prop_h == 0
    edge_buy_proposed = edge_buy_b[prop_h == 2]
    edge_sell_proposed = edge_sell_b[prop_h == 0]
    
    print("Edge Score Stats (BUY proposed):")
    if len(edge_buy_proposed) > 0:
        print(f"  Count: {len(edge_buy_proposed)} | Min: {edge_buy_proposed.min():.2f} | 25%: {np.percentile(edge_buy_proposed, 25):.2f} | 50%: {np.percentile(edge_buy_proposed, 50):.2f} | 75%: {np.percentile(edge_buy_proposed, 75):.2f} | 90%: {np.percentile(edge_buy_proposed, 90):.2f} | Max: {edge_buy_proposed.max():.2f}")
    else:
        print("  No BUY proposals")
        
    print("Edge Score Stats (SELL proposed):")
    if len(edge_sell_proposed) > 0:
        print(f"  Count: {len(edge_sell_proposed)} | Min: {edge_sell_proposed.min():.2f} | 25%: {np.percentile(edge_sell_proposed, 25):.2f} | 50%: {np.percentile(edge_sell_proposed, 50):.2f} | 75%: {np.percentile(edge_sell_proposed, 75):.2f} | 90%: {np.percentile(edge_sell_proposed, 90):.2f} | Max: {edge_sell_proposed.max():.2f}")
    else:
        print("  No SELL proposals")
        
    fire_buy_b = (edge_buy_b >= 55.0) & (prop_h == 2)
    fire_sell_b = (edge_sell_b >= 55.0) & (prop_h == 0)
    fire_b = (fire_buy_b | fire_sell_b) & quality_pass & hmm_pass & conf_pass & drift_pass
    
    bt_b = backtest(fire_b, prop_h, y_test, barrier_frac)
    report_bt("Simulation B (Current Meta Gate)", fire_b, prop_h, y_test, bt_b)
    
    # ── SIMULATION C: Meta Gate with Adaptive Thresholds ──
    print("\n--- Running Simulation C: Meta Gate with Adaptive Thresholds ---")
    for edge_thr in [45.0, 50.0, 52.0, 55.0, 58.0, 60.0]:
        fire_buy_c = (edge_buy_b >= edge_thr) & (prop_h == 2)
        fire_sell_c = (edge_sell_b >= edge_thr) & (prop_h == 0)
        fire_c = (fire_buy_c | fire_sell_c) & quality_pass & hmm_pass & conf_pass & drift_pass
        bt_c = backtest(fire_c, prop_h, y_test, barrier_frac)
        print(f"Threshold: {edge_thr:.1f} | Fired: {fire_c.sum()} | Precision: {bt_c['win_rate']:.1%} | Expectancy: {bt_c['expectancy_pct']:+.3f}% | PF: {bt_c['profit_factor']:.2f}")

def report_bt(name, fire, prop_h, y_test, bt):
    fired_n = int(fire.sum())
    fired_prec = float((prop_h[fire] == y_test[fire]).mean()) if fired_n > 0 else 0.0
    print(f"Result for {name}:")
    print(f"  Trades Fired : {fired_n} / {len(y_test)}")
    print(f"  Precision    : {fired_prec:.1%}")
    print(f"  Win Rate (PnL): {bt['win_rate']:.1%}")
    print(f"  Expectancy   : {bt['expectancy_pct']:+.4f}%")
    print(f"  Profit Factor: {bt['profit_factor']:.3f}")
    print(f"  BUY count    : {bt['buy_n']} (Win Rate: {bt['buy_win_rate']:.1%})")
    print(f"  SELL count   : {bt['sell_n']} (Win Rate: {bt['sell_win_rate']:.1%})")

if __name__ == "__main__":
    run()
