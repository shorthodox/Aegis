"""
drift_monitor.py — Drift-Aware Feature Weighting Engine
========================================================
Computes final feature weights for XGBoost retraining:

    final_weight = shap_importance × stability_score × drift_penalty

High-drift features contribute less (penalty < 1.0).
Low-drift features contribute more (penalty = 1.0, stability bonus up to 1.2).

Applied during retraining via XGBoost feature_weights parameter.
Does NOT touch live prediction logic.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class DriftAwareFeatureWeightingEngine:
    """
    Combines three orthogonal signals into a per-feature training weight:

    1. SHAP importance  — how much each feature actually drove predictions
                          (measured on a quick train pass before HPO)
    2. Stability score  — how consistently the feature performs across
                          time-series folds (1.0 = stable, 0.0 = erratic)
    3. Drift penalty    — computed by FeatureHealthManager from PSI/KS/JS
                          (1.0 = no drift, 0.15 = CRITICAL drift)

    The product is re-normalised so the mean weight stays at 1.0 — no
    overall scale drift vs. the original uniform-weight baseline.
    """

    def __init__(
        self,
        shap_floor:       float = 0.05,   # minimum SHAP importance (before normalisation)
        stability_bonus:  float = 0.20,   # max upward bonus for perfectly stable features
        weight_floor:     float = 0.05,   # absolute floor on any feature's final weight
        weight_cap:       float = 3.00,   # absolute cap to prevent one feature dominating
    ) -> None:
        self.shap_floor      = shap_floor
        self.stability_bonus = stability_bonus
        self.weight_floor    = weight_floor
        self.weight_cap      = weight_cap

        # Populated during compute()
        self._final_weights:    Optional[np.ndarray] = None
        self._components:       Optional[pd.DataFrame] = None

    # ── Core computation ─────────────────────────────────────────────────────

    def compute(
        self,
        features:         List[str],
        shap_importance:  Optional[np.ndarray]  = None,   # raw |SHAP| mean per feature
        drift_penalties:  Optional[np.ndarray]  = None,   # from FeatureHealthManager
        fold_importances: Optional[np.ndarray]  = None,   # shape (n_folds, n_features)
    ) -> np.ndarray:
        """
        Compute final_weight for each feature.

        Parameters
        ----------
        features         : list of feature names (length n)
        shap_importance  : absolute SHAP values, shape (n,). If None → uniform.
        drift_penalties  : drift penalty per feature in [0.15, 1.0]. If None → 1.0.
        fold_importances : per-fold SHAP importance matrix for stability calc.
                           If None, stability_score = 1.0 for all features.

        Returns
        -------
        weights : np.ndarray shape (n,), mean ≈ 1.0, floor=weight_floor, cap=weight_cap.
        """
        n = len(features)

        # ── 1. SHAP importance → normalised [0, 1] ───────────────────────────
        if shap_importance is not None and len(shap_importance) == n:
            si = np.asarray(shap_importance, dtype=float)
            si = np.clip(si, 0.0, None)
            total = si.sum()
            if total > 0:
                si = si / total         # normalise to [0, 1], sum = 1
            else:
                si = np.ones(n) / n
            # Apply floor so rare-but-valid features still get a nonzero weight
            si = np.clip(si * n, self.shap_floor, None)   # scale back to mean ≈ 1
        else:
            si = np.ones(n, dtype=float)

        # ── 2. Stability score ────────────────────────────────────────────────
        if fold_importances is not None and fold_importances.shape == (fold_importances.shape[0], n):
            # Coefficient of variation across folds for each feature.
            # Low CV → stable → high score.  High CV → unstable → low score.
            fold_arr = np.asarray(fold_importances, dtype=float)
            means    = fold_arr.mean(axis=0)
            stds     = fold_arr.std(axis=0)
            cv       = stds / (np.abs(means) + 1e-8)
            # Map CV to [0, 1]: CV=0 → 1.0, CV=2 → 0.0 (clamp)
            stability = 1.0 - np.clip(cv / 2.0, 0.0, 1.0)
        else:
            stability = np.ones(n, dtype=float)

        # Apply stability bonus: stable features get up to +stability_bonus
        stability_score = 1.0 + self.stability_bonus * stability   # range [1.0, 1.2]

        # ── 3. Drift penalty ──────────────────────────────────────────────────
        if drift_penalties is not None and len(drift_penalties) == n:
            dp = np.clip(np.asarray(drift_penalties, dtype=float), 0.0, 1.0)
        else:
            dp = np.ones(n, dtype=float)

        # ── 4. Combine ────────────────────────────────────────────────────────
        raw = si * stability_score * dp
        raw = np.clip(raw, self.weight_floor, self.weight_cap)

        # Renormalise: mean = 1.0 so XGBoost feature_weights are scale-neutral
        mean_w = raw.mean()
        if mean_w > 1e-9:
            weights = raw / mean_w
        else:
            weights = np.ones(n, dtype=float)

        # Store diagnostics
        self._final_weights = weights
        self._components = pd.DataFrame({
            'feature':         features,
            'shap_importance': si,
            'stability_score': stability_score,
            'drift_penalty':   dp,
            'final_weight':    weights,
        })

        return weights

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def print_summary(self, top_n: int = 10) -> None:
        if self._components is None:
            print("   [DriftMonitor] No weights computed yet.")
            return
        df = self._components.sort_values('final_weight')
        print(f"   [DriftMonitor] Feature weights (mean={self._final_weights.mean():.3f})")
        print("   Bottom weights (most penalised):")
        for _, row in df.head(top_n).iterrows():
            print(f"      {row['feature']:<40} w={row['final_weight']:.3f}  "
                  f"shap={row['shap_importance']:.3f}  "
                  f"stab={row['stability_score']:.3f}  "
                  f"drift={row['drift_penalty']:.3f}")

    def get_components(self) -> Optional[pd.DataFrame]:
        return self._components


def build_drift_aware_weights(
    features:        List[str],
    fhm,                                     # FeatureHealthManager instance
    shap_importance: Optional[np.ndarray] = None,
    fold_importances: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Convenience wrapper: build DriftAwareFeatureWeightingEngine, pull drift
    penalties from the FeatureHealthManager, and return final_weight array.

    Suitable for drop-in replacement of DynamicFeatureWeightingEngine in
    retrain_model.py.
    """
    engine = DriftAwareFeatureWeightingEngine()
    drift_penalties = fhm.get_all_penalties(features)
    return engine.compute(
        features=features,
        shap_importance=shap_importance,
        drift_penalties=drift_penalties,
        fold_importances=fold_importances,
    )
