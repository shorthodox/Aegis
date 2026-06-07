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
    model_store = Path(root_dir) / "src" / "ml" / "model_store"
    trained_symbols = []
    
    # Find all trained symbols by looking for sidecar json files
    for p in model_store.glob("*_meta.json"):
        # Convert ADA_USDT_meta.json -> ADA/USDT
        sym = p.name.replace("_meta.json", "").replace("_", "/")
        # Special check: e.g., if there are multiple underscores (like standard tickers), handle them.
        # Here we assume standard USDT pairs
        if not sym.endswith("/USDT"):
            continue
        trained_symbols.append(sym)
        
    print(f"Found {len(trained_symbols)} trained symbols: {trained_symbols}")
    
    report = {}
    
    for symbol in trained_symbols:
        base = symbol.replace("/", "_")
        print(f"\nAuditing {symbol}...")
        
        # Load sidecar meta
        sidecar_path = model_store / f"{base}_meta.json"
        meta = json.loads(sidecar_path.read_text())
        
        # Load predictor
        p = Predictor(symbol)
        
        # Fetch data
        df = p.get_features_with_context(hours=5000)
        if df is None or df.empty:
            print(f"Skipping {symbol}: no data")
            continue
        df = df.reset_index(drop=True)
        
        # Compute soft confluence and _atr
        try:
            from scripts.retrain_model import compute_soft_confluence_features
            sc = compute_soft_confluence_features(df)
            for col in sc.columns:
                df[col] = sc[col].values
        except Exception:
            pass
        df["_atr"] = compute_atr(df, period=14).values
        
        # Compute target labels
        atr_mult = float(meta.get("atr_multiplier", 1.5))
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
        
        # Predict
        feature_cols = meta.get("feature_cols")
        X_test = holdout[feature_cols]
        proba = p.predict_proba(holdout)
        primary_map = proba.argmax(1)
        prop_h = np.where(proba[:, 2] >= proba[:, 0], 2, 0)
        
        if p.meta_model is not None:
            from scripts.retrain_model import build_meta_X
            meta_prob_h = p.meta_model.predict(xgb.DMatrix(build_meta_X(X_test, proba)))
        else:
            meta_prob_h = proba.max(axis=1)
            
        if getattr(p, 'aegis_state', None) is not None:
            mcf = p.aegis_state.get('mcf')
            cre = p.aegis_state.get('cre')
            if mcf: meta_prob_h = mcf.calibrate(meta_prob_h)
            if cre: meta_prob_h = cre.adjust_confidence_array(meta_prob_h)
            
        # Funnel tracing
        dir_mask = primary_map != 1
        total_dir = int(dir_mask.sum())
        
        # Load dynamic thresholds and disabled filters
        thr_buy = float(meta.get("meta_threshold_buy", meta.get("meta_threshold", 55.0)))
        thr_sell = float(meta.get("meta_threshold_sell", meta.get("meta_threshold", 55.0)))
        
        disabled_filters = meta.get("disabled_filters", {})
        disable_sr = disabled_filters.get("sr", False)
        disable_trend = disabled_filters.get("trend", False)
        disable_confluence = disabled_filters.get("confluence", False)
        
        # 1. EdgeEngine (using threshold from meta)
        edge_buy = EdgeScoringEngine.compute_edge_batch(holdout, meta_prob_h, 'BUY').to_numpy()
        edge_sell = EdgeScoringEngine.compute_edge_batch(holdout, meta_prob_h, 'SELL').to_numpy()
        edge_pass = ((edge_buy >= thr_buy) & (prop_h == 2)) | ((edge_sell >= thr_sell) & (prop_h == 0))
        
        # 2. Quality
        if "prc_total" in holdout.columns:
            prc_val = holdout["prc_total"].values
            quality = np.where(prop_h == 2, prc_val * 100.0, (1.0 - prc_val) * 100.0)
        else:
            quality = np.full(len(holdout), 60.0)
        quality_pass = quality >= 55.0
        
        # 3. S&R + Trend filter
        at_res = holdout['is_at_resistance'].to_numpy().astype(bool) if 'is_at_resistance' in holdout.columns else np.zeros(len(holdout), dtype=bool)
        at_sup = holdout['is_at_support'].to_numpy().astype(bool) if 'is_at_support' in holdout.columns else np.zeros(len(holdout), dtype=bool)
        
        fire_edge = dir_mask & edge_pass
        override_thr = float(np.quantile(meta_prob_h[fire_edge], 0.75)) if fire_edge.sum() > 0 else 0.55
        top25 = meta_prob_h >= override_thr
        
        sr_pass = np.ones(len(holdout), dtype=bool)
        if not disable_sr:
            sr_pass = sr_pass & ~((dir_mask & (prop_h == 2) & at_res & ~top25) | (dir_mask & (prop_h == 0) & at_sup & ~top25))
        if 'macro_trend_1d' in holdout.columns and not disable_trend:
            trend_1d = holdout['macro_trend_1d'].to_numpy()
            sr_pass = sr_pass & ~(
                (dir_mask & (prop_h == 2) & (trend_1d < -0.2) & ~top25) |
                (dir_mask & (prop_h == 0) & (trend_1d > 0.2) & ~top25)
            )
            
        # 4. Regime filter (replicate live regime checks)
        regime_pass = np.ones(len(holdout), dtype=bool)
        regime_thresholds = meta.get("regime_thresholds", {})
        bounds = meta.get("regime_boundaries", {})
        if regime_thresholds and bounds:
            vol_avg = holdout["volume"].rolling(24, min_periods=1).mean()
            atr_pct = (holdout["_atr"] / holdout["close"]).fillna(0)
            momentum = holdout["close"].pct_change(24).fillna(0)
            
            def _tier(val, p33, p67): return "low" if val <= p33 else ("med" if val <= p67 else "high")
            def _trend(val, p33, p67): return "down" if val <= p33 else ("flat" if val <= p67 else "up")
            
            vp33, vp67 = bounds.get("vol_p33", 0), bounds.get("vol_p67", 0)
            ap33, ap67 = bounds.get("atr_pct_p33", 0), bounds.get("atr_pct_p67", 0)
            mp33, mp67 = bounds.get("momentum_p33", -0.02), bounds.get("momentum_p67", 0.02)
            
            for i in range(len(holdout)):
                r_str = f"{_tier(vol_avg.iloc[i], vp33, vp67)}_{_tier(atr_pct.iloc[i], ap33, ap67)}_{_trend(momentum.iloc[i], mp33, mp67)}"
                reg = regime_thresholds.get(r_str, {})
                if not reg:
                    continue
                if reg.get("skipped"):
                    regime_pass[i] = False
                    continue
                
                side = prop_h[i]
                p_sell, p_hold, p_buy = float(proba[i, 0]), float(proba[i, 1]), float(proba[i, 2])
                if side == 2:
                    if (not reg.get("buy_ok")
                            or p_buy < reg.get("buy_threshold", 0.0)
                            or (p_buy - p_sell) < reg.get("buy_margin", 0.0)
                            or p_hold > reg.get("buy_max_hold", 1.0)):
                        regime_pass[i] = False
                elif side == 0:
                    if (not reg.get("sell_ok")
                            or p_sell < reg.get("sell_threshold", 0.0)
                            or (p_sell - p_buy) < reg.get("sell_margin", 0.0)
                            or p_hold > reg.get("sell_max_hold", 1.0)):
                        regime_pass[i] = False
            
        # 5. Confluence filter
        confluence_pass = np.ones(len(holdout), dtype=bool)
        if 'total_confluence' in holdout.columns and not disable_confluence:
            tc = holdout['total_confluence'].to_numpy()
            confluence_pass = ~((prop_h == 2) & (tc < -0.05)) & ~((prop_h == 0) & (tc > 0.05))
            
        # 6. Drift filter
        drift_pass = np.ones(len(holdout), dtype=bool)
        n_critical = 0
        if getattr(p, 'aegis_state', None) is not None and isinstance(p.aegis_state, dict):
            n_critical = sum(
                1 for s in (p.aegis_state.get("fhm") or {}).values()
                if s == "CRITICAL"
            )
        if n_critical > 10:
            drift_pass[int(len(holdout) * 0.85):] = False
            
        # 7. Portfolio & Cooldown filters
        portfolio_pass = np.ones(len(holdout), dtype=bool)
        cooldown_pass = np.ones(len(holdout), dtype=bool)
        MAX_POSITIONS = 6
        COOLDOWN_BARS = 4
        last_fire = -COOLDOWN_BARS - 1
        open_count = 0
        
        for i in range(len(holdout)):
            if dir_mask[i] and edge_pass[i] and quality_pass[i] and sr_pass[i] and \
               regime_pass[i] and confluence_pass[i] and drift_pass[i]:
                # Check cooldown
                if (i - last_fire) < COOLDOWN_BARS:
                    cooldown_pass[i] = False
                else:
                    # Check portfolio
                    if open_count >= MAX_POSITIONS:
                        portfolio_pass[i] = False
                    else:
                        open_count = max(0, open_count - 1) + 1
                        last_fire = i

        # Cascade counting
        funnel = []
        rem = dir_mask.copy()
        
        funnel.append(("Directional", int(rem.sum())))
        rem = rem & edge_pass
        funnel.append(("EdgeEngine", int(rem.sum())))
        rem = rem & quality_pass
        funnel.append(("Quality", int(rem.sum())))
        rem = rem & sr_pass
        funnel.append(("S&R", int(rem.sum())))
        rem = rem & regime_pass
        funnel.append(("Regime", int(rem.sum())))
        rem = rem & confluence_pass
        funnel.append(("Confluence", int(rem.sum())))
        rem = rem & drift_pass
        funnel.append(("Drift", int(rem.sum())))
        rem = rem & portfolio_pass
        funnel.append(("Portfolio", int(rem.sum())))
        rem = rem & cooldown_pass
        funnel.append(("Final", int(rem.sum())))
        
        report[symbol] = {name: val for name, val in funnel}
        print(f"  {symbol} Funnel: {funnel}")
        
    output_path = Path(root_dir) / "logs" / "forensics" / "filter_loss_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFunnel loss report saved to {output_path}")

if __name__ == "__main__":
    run()
