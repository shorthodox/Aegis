"""
calibration.py — Meta Model Calibration Framework
===================================================
Evaluates four calibration methods and selects the best by ECE.

Methods evaluated:
  1. Isotonic Regression  — monotonic, non-parametric
  2. Platt Scaling        — logistic sigmoid re-mapping
  3. Temperature Scaling  — single learnable softening parameter
  4. Beta Calibration     — two-parameter Beta CDF mapping

Per method computes: ECE, Brier, Precision, Coverage
Automatically selects best calibrator (target ECE < 0.10, prefer < 0.05).
"""

import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

_ROOT      = Path(__file__).resolve().parent.parent.parent
_MODEL_DIR = _ROOT / 'src' / 'ml' / 'model_store'


# ── Metrics ──────────────────────────────────────────────────────────────────

def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    bins   = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    ece    = 0.0
    for i in range(n_bins):
        m = (binids == i)
        if m.any():
            ece += (m.sum() / len(y_prob)) * abs(y_prob[m].mean() - y_true[m].mean())
    return float(ece)


def _brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


def _precision_coverage(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
) -> Tuple[float, float]:
    fired = y_prob >= threshold
    n = fired.sum()
    if n == 0:
        return 0.0, 0.0
    prec = float(y_true[fired].mean())
    cov  = float(n / len(y_prob))
    return prec, cov


# ── Temperature Scaling ───────────────────────────────────────────────────────

class TemperatureScaling:
    """Single scalar T that divides the log-odds: p_cal = σ(logit(p) / T)."""

    def __init__(self) -> None:
        self.T: float = 1.0

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> 'TemperatureScaling':
        from scipy.optimize import minimize_scalar
        logits = np.log(np.clip(y_prob, 1e-9, 1 - 1e-9)) - \
                 np.log(np.clip(1 - y_prob, 1e-9, 1 - 1e-9))

        def nll(T: float) -> float:
            scaled = 1.0 / (1.0 + np.exp(-logits / max(T, 1e-3)))
            return -float(np.mean(
                y_true * np.log(np.clip(scaled, 1e-9, 1.0)) +
                (1 - y_true) * np.log(np.clip(1 - scaled, 1e-9, 1.0))
            ))

        try:
            res = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
            self.T = float(res.x) if 0.1 < res.x < 10.0 else 1.0
        except Exception:
            self.T = 1.0
        return self

    def predict(self, y_prob: np.ndarray) -> np.ndarray:
        logits = np.log(np.clip(y_prob, 1e-9, 1 - 1e-9)) - \
                 np.log(np.clip(1 - y_prob, 1e-9, 1 - 1e-9))
        scaled = 1.0 / (1.0 + np.exp(-logits / max(self.T, 1e-3)))
        return np.clip(scaled, 0.0, 1.0)


# ── Beta Calibration ─────────────────────────────────────────────────────────

class BetaCalibration:
    """
    Two-parameter Beta CDF mapping: p_cal = I(p; a, b) where a, b > 0.
    Implemented via log-odds regression on log(p) and log(1-p).
    Reference: Kull, de Menezes e Silva Filho, Flach (2017).
    """

    def __init__(self) -> None:
        self._lr: Optional[Any] = None

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> 'BetaCalibration':
        if not _SKLEARN:
            return self
        p   = np.clip(y_prob, 1e-9, 1 - 1e-9)
        X   = np.column_stack([np.log(p), -np.log(1 - p)])
        try:
            lr = LogisticRegression(C=1e6, solver='lbfgs', max_iter=500)
            lr.fit(X, y_true.astype(int))
            self._lr = lr
        except Exception:
            pass
        return self

    def predict(self, y_prob: np.ndarray) -> np.ndarray:
        if self._lr is None:
            return y_prob.copy()
        p = np.clip(y_prob, 1e-9, 1 - 1e-9)
        X = np.column_stack([np.log(p), -np.log(1 - p)])
        return np.clip(self._lr.predict_proba(X)[:, 1], 0.0, 1.0)


# ── Framework ─────────────────────────────────────────────────────────────────

class MetaCalibrationFramework:
    """
    Evaluates Isotonic, Platt, Temperature, and Beta calibration methods.
    Selects the best by ECE and exposes calibrate() for inference.
    Target: ECE < 0.10 (preferred ECE < 0.05).
    """

    def __init__(self) -> None:
        self.best_calibrator: Optional[Any]       = None
        self.calibrator_type: str                  = 'uncalibrated'
        self.reports:         Dict[str, Dict]      = {}
        self._ece_before: float = 1.0
        self._ece_after:  float = 1.0

    # ── Evaluation ───────────────────────────────────────────────────────────

    def evaluate_calibrators(
        self,
        y_prob: np.ndarray,
        y_true: np.ndarray,
        threshold: float = 0.50,
    ) -> Dict[str, Any]:
        valid        = ~np.isnan(y_prob)
        y_prob_clean = y_prob[valid].copy()
        y_true_clean = y_true[valid].copy()

        if len(y_prob_clean) < 20:
            return {}

        # If input values are outside [0, 1] (e.g. continuous expected returns),
        # map them to pseudo-probabilities using a sigmoid function scaled by standard deviation.
        self._is_mapped = False
        if y_prob_clean.min() < 0.0 or y_prob_clean.max() > 1.0:
            self._scale = max(1e-4, np.std(y_prob_clean))
            y_prob_clean = 1.0 / (1.0 + np.exp(-y_prob_clean / self._scale))
            self._is_mapped = True

        results: Dict[str, Dict] = {}

        def _record(name: str, preds: np.ndarray, model: Any) -> None:
            prec, cov = _precision_coverage(y_true_clean, preds, threshold)
            results[name] = {
                'ece':       _ece(y_true_clean, preds),
                'brier':     _brier(y_true_clean, preds),
                'precision': prec,
                'coverage':  cov,
                'model':     model,
            }

        # Baseline
        _record('uncalibrated', y_prob_clean, None)
        self._ece_before = results['uncalibrated']['ece']

        # 1. Isotonic Regression — skipped on tiny dev sets.
        # With < 50 samples isotonic memorises the training distribution perfectly
        # (ECE≈0 on dev, collapse on holdout). Platt/temperature generalise better.
        if _SKLEARN and len(y_prob_clean) >= 50:
            try:
                iso = IsotonicRegression(out_of_bounds='clip')
                iso.fit(y_prob_clean, y_true_clean)
                _record('isotonic', np.clip(iso.predict(y_prob_clean), 0.0, 1.0), iso)
            except Exception:
                pass

        # 2. Platt Scaling
        if _SKLEARN:
            try:
                lr = LogisticRegression(C=1e6, solver='lbfgs', max_iter=500)
                lr.fit(y_prob_clean.reshape(-1, 1), y_true_clean.astype(int))
                preds_platt = lr.predict_proba(y_prob_clean.reshape(-1, 1))[:, 1]
                _record('platt', preds_platt, lr)
            except Exception:
                pass

        # 3. Temperature Scaling
        try:
            ts = TemperatureScaling()
            ts.fit(y_prob_clean, y_true_clean)
            _record('temperature', ts.predict(y_prob_clean), ts)
        except Exception:
            pass

        # 4. Beta Calibration
        if _SKLEARN:
            try:
                bc = BetaCalibration()
                bc.fit(y_prob_clean, y_true_clean)
                _record('beta', bc.predict(y_prob_clean), bc)
            except Exception:
                pass

        # Select best by ECE (must be strictly better than baseline)
        best_name = 'uncalibrated'
        best_ece  = results['uncalibrated']['ece']

        for name, res in results.items():
            if name == 'uncalibrated':
                continue
            if res['model'] is not None and res['ece'] < best_ece:
                best_ece  = res['ece']
                best_name = name

        self.reports         = results
        self.calibrator_type = best_name
        self.best_calibrator = results[best_name]['model']
        self._ece_after      = best_ece

        # Summary
        baseline_ece = results['uncalibrated']['ece']
        print(f"   [Calibration] Selected: {best_name.upper()} | "
              f"ECE: {best_ece:.4f} (baseline: {baseline_ece:.4f}) | "
              f"{'V target met' if best_ece < 0.10 else 'X ECE > 0.10'}")
        for name, res in sorted(results.items(), key=lambda x: x[1]['ece']):
            if name != 'uncalibrated':
                print(f"      {name:<14} ECE={res['ece']:.4f}  "
                      f"Brier={res['brier']:.4f}  "
                      f"Prec={res['precision']:.3f}  "
                      f"Cov={res['coverage']:.3f}")

        return results

    # ── Inference ────────────────────────────────────────────────────────────

    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        cal   = np.full_like(y_prob, np.nan, dtype=np.float64)
        valid = ~np.isnan(y_prob)
        if not valid.any():
            return cal

        p = y_prob[valid].copy()

        if self.calibrator_type != 'uncalibrated' and getattr(self, '_is_mapped', False):
            scale = getattr(self, '_scale', 0.02)
            p = 1.0 / (1.0 + np.exp(-p / scale))

        if self.calibrator_type == 'isotonic' and self.best_calibrator is not None:
            cal[valid] = np.clip(
                self.best_calibrator.predict(p).astype(np.float64), 0.0, 1.0)
        elif self.calibrator_type == 'platt' and self.best_calibrator is not None:
            cal[valid] = self.best_calibrator.predict_proba(
                p.reshape(-1, 1))[:, 1]
        elif self.calibrator_type == 'temperature' and self.best_calibrator is not None:
            cal[valid] = self.best_calibrator.predict(p)
        elif self.calibrator_type == 'beta' and self.best_calibrator is not None:
            cal[valid] = self.best_calibrator.predict(p)
        else:
            cal[valid] = p

        return cal

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, symbol: str) -> Path:
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path = _MODEL_DIR / f"{symbol.replace('/', '_')}_best_calibrator.pkl"
        with open(path, 'wb') as f:
            pickle.dump({
                'type':        self.calibrator_type,
                'model':       self.best_calibrator,
                'ece_before':  self._ece_before,
                'ece_after':   self._ece_after,
            }, f, protocol=4)
        return path

    @classmethod
    def load(cls, symbol: str) -> Optional['MetaCalibrationFramework']:
        path = _MODEL_DIR / f"{symbol.replace('/', '_')}_best_calibrator.pkl"
        if not path.exists():
            return None
        try:
            with open(path, 'rb') as f:
                payload = pickle.load(f)
            inst = cls()
            inst.calibrator_type  = payload['type']
            inst.best_calibrator  = payload['model']
            inst._ece_before      = payload.get('ece_before', 1.0)
            inst._ece_after       = payload.get('ece_after',  1.0)
            return inst
        except Exception:
            return None
