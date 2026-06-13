import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.meta_gate_optimizer import (
    _fit_local_model, compute_edge_scores, MIN_BARS, TRAIN_FRAC
)
from scripts.retrain_model import create_triple_barrier_labels, FEE_ROUNDTRIP, get_atr_multiplier
from src.ml.calibration import MetaCalibrationFramework
from src.ml.feature_engine import prepare_features
from src.ml.predictor import Predictor
import numpy as np
import pandas as pd

SYMBOL = 'NEAR/USDT'

def main():
    print(f"Diagnostics for {SYMBOL}")
    p = Predictor(SYMBOL)
    df_1h = p.fetch_live_data(timeframe='1h', limit=4000)
    if df_1h is None or len(df_1h) < MIN_BARS:
        print('Not enough data to run diagnostics')
        return
    btc_df = p.fetch_btc_data(timeframe='1h', limit=4000)
    news_df = p.load_news_data()
    df_1d = p.fetch_live_data(timeframe='1d', limit=300)
    fund_df, oi_df = None, None

    features = prepare_features(df_1h, btc_df=btc_df, news_df=news_df, df_1d=df_1d, funding_df=fund_df, oi_df=oi_df)
    features = features.reset_index(drop=True)
    n_feat = len(features)
    train_n = int(n_feat * TRAIN_FRAC)

    # ATR multiplier
    base_atr_mult = float(p.meta.get('atr_multiplier', get_atr_multiplier(SYMBOL)))

    # labels
    labels_all = np.asarray(create_triple_barrier_labels(df_1h.iloc[-n_feat:].reset_index(drop=True), atr_multiplier=base_atr_mult), dtype=np.intp)

    # Build feature matrix same as optimizer
    feature_cols = p.meta.get('feature_cols')
    if feature_cols:
        missing = [c for c in feature_cols if c not in features.columns]
        if missing:
            features = pd.concat([features, pd.DataFrame(0.0, index=features.index, columns=missing)], axis=1)
        feat_df = features[feature_cols].copy()
    else:
        drop = [c for c in ('timestamp', 'target') if c in features.columns]
        feat_df = features.drop(columns=drop, errors='ignore')
        feat_df = feat_df[[c for c in feat_df.columns if not c.startswith('_')]]

    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # fit local model and get OOF and full predictions
    proposed_all, dir_conf_all, probs_all, _ = _fit_local_model(feat_df, labels_all, train_n)

    # train mask
    train_mask = labels_all[:train_n] != 1
    train_edge_scores = np.asarray(compute_edge_scores(
        features.iloc[:train_n].reset_index(drop=True),
        proposed_all[:train_n],
        dir_conf_all[:train_n],
        use_rank=True,
    ), dtype=float)
    train_correct = (labels_all[:train_n][train_mask] == proposed_all[:train_n][train_mask]).astype(int)

    # holdout
    proposed_ev = proposed_all[train_n:]
    ev_labels = labels_all[train_n:]
    hold_mask = ev_labels != 1
    hold_precision = float((ev_labels[hold_mask] == proposed_ev[hold_mask]).mean()) if hold_mask.any() else 0.0

    train_precision = float(train_correct.mean()) if len(train_correct) else 0.0

    print(f"Train rows: {train_n}, Holdout rows: {n_feat-train_n}")
    print(f"Train valid rows: {train_mask.sum()}, Holdout valid rows: {hold_mask.sum()}")
    print(f"Train precision: {train_precision:.4f}")
    print(f"Holdout precision: {hold_precision:.4f}")

    # label distribution
    from collections import Counter
    print('Train label dist:', Counter(labels_all[:train_n]))
    print('Holdout label dist:', Counter(ev_labels))

    # calibration report
    train_edge_raw = train_edge_scores[train_mask]
    trainer = MetaCalibrationFramework()
    report = trainer.evaluate_calibrators(train_edge_raw / 100.0, train_correct, threshold=0.50)
    print('Calibration candidate keys:', list(report.keys()))
    for k, v in report.items():
        print(k, '->', {kk: v.get(kk) for kk in ('ece','brier','precision','coverage')})

if __name__ == '__main__':
    main()
