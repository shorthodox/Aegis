import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon
from typing import Dict, List, Optional

# State thresholds (composite drift score)
_HEALTHY   = 0.10
_WARNING   = 0.25
_DEGRADED  = 0.50
# >= _DEGRADED → CRITICAL

_STATE_PENALTIES = {
    'HEALTHY':  1.00,
    'WARNING':  0.80,
    'DEGRADED': 0.50,
    'CRITICAL': 0.15,
}

_ROOT = Path(__file__).resolve().parent.parent.parent
_LOG_DIR = _ROOT / 'logs' / 'feature_health'


@dataclass
class FeatureMetrics:
    name: str
    psi: float
    ks: float
    js: float
    mean_drift: float          # abs(train_mean - eval_mean) / (|train_mean| + 1e-8)
    std_drift: float           # abs(train_std  - eval_std)  / (train_std  + 1e-8)
    score: float               # composite: 0.5×PSI + 0.3×JS + 0.2×KS
    state: str                 # HEALTHY / WARNING / DEGRADED / CRITICAL
    drift_penalty: float       # multiplier applied to SHAP weight


@dataclass
class FeatureHealthReport:
    symbol: str
    generated_at: str
    n_features: int
    summary: Dict[str, int]    # counts per state
    top_drifters: List[Dict]   # top-10 by composite score
    features: Dict[str, Dict]  # full per-feature record


class FeatureHealthManager:
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.feature_states: Dict[str, str]          = {}
        self.drift_scores:   Dict[str, Dict]         = {}
        self._metrics:       Dict[str, FeatureMetrics] = {}

    # ── Internal statistics ──────────────────────────────────────────────────

    def _psi(self, expected: np.ndarray, actual: np.ndarray) -> float:
        expected = expected[~np.isnan(expected)]
        actual   = actual[~np.isnan(actual)]
        if len(expected) == 0 or len(actual) == 0:
            return 0.0
        bins = np.histogram_bin_edges(np.concatenate([expected, actual]), bins=self.n_bins)
        e_cnt, _ = np.histogram(expected, bins=bins)
        a_cnt, _ = np.histogram(actual,   bins=bins)
        e_pct = np.clip(e_cnt / len(expected), 1e-6, None)
        a_pct = np.clip(a_cnt / len(actual),   1e-6, None)
        return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))

    def _js(self, expected: np.ndarray, actual: np.ndarray) -> float:
        expected = expected[~np.isnan(expected)]
        actual   = actual[~np.isnan(actual)]
        if len(expected) == 0 or len(actual) == 0:
            return 0.0
        bins = np.histogram_bin_edges(np.concatenate([expected, actual]), bins=self.n_bins)
        e_cnt, _ = np.histogram(expected, bins=bins)
        a_cnt, _ = np.histogram(actual,   bins=bins)
        e_pct = e_cnt / max(len(expected), 1)
        a_pct = a_cnt / max(len(actual),   1)
        return float(jensenshannon(e_pct, a_pct))

    def _ks(self, expected: np.ndarray, actual: np.ndarray) -> float:
        expected = expected[~np.isnan(expected)]
        actual   = actual[~np.isnan(actual)]
        if len(expected) == 0 or len(actual) == 0:
            return 0.0
        return float(ks_2samp(expected, actual).statistic)

    @staticmethod
    def _mean_drift(train: np.ndarray, eval_: np.ndarray) -> float:
        t = train[~np.isnan(train)]
        e = eval_[~np.isnan(eval_)]
        if len(t) == 0 or len(e) == 0:
            return 0.0
        return float(abs(t.mean() - e.mean()) / (abs(t.mean()) + 1e-8))

    @staticmethod
    def _std_drift(train: np.ndarray, eval_: np.ndarray) -> float:
        t = train[~np.isnan(train)]
        e = eval_[~np.isnan(eval_)]
        if len(t) == 0 or len(e) == 0:
            return 0.0
        t_std = t.std()
        e_std = e.std()
        return float(abs(t_std - e_std) / (t_std + 1e-8))

    @staticmethod
    def _classify(score: float) -> str:
        if score < _HEALTHY:
            return 'HEALTHY'
        if score < _WARNING:
            return 'WARNING'
        if score < _DEGRADED:
            return 'DEGRADED'
        return 'CRITICAL'

    # ── Public API ───────────────────────────────────────────────────────────

    def analyze_drift(
        self,
        df_train: pd.DataFrame,
        df_eval:  pd.DataFrame,
        features: List[str],
    ) -> Dict:
        for feat in features:
            if feat not in df_train.columns or feat not in df_eval.columns:
                continue
            tv = df_train[feat].values.astype(float)
            ev = df_eval[feat].values.astype(float)

            psi   = self._psi(tv, ev)
            js    = self._js(tv, ev)
            ks    = self._ks(tv, ev)
            mdrift = self._mean_drift(tv, ev)
            sdrift = self._std_drift(tv, ev)

            score = psi * 0.5 + js * 0.3 + ks * 0.2
            state = self._classify(score)

            self.drift_scores[feat] = {
                'psi': round(psi, 4),
                'js':  round(js,  4),
                'ks':  round(ks,  4),
                'mean_drift': round(mdrift, 4),
                'std_drift':  round(sdrift, 4),
                'score': round(score, 4),
            }
            self.feature_states[feat] = state
            self._metrics[feat] = FeatureMetrics(
                name=feat, psi=psi, ks=ks, js=js,
                mean_drift=mdrift, std_drift=sdrift,
                score=score, state=state,
                drift_penalty=_STATE_PENALTIES[state],
            )
        return self.drift_scores

    def get_feature_states(self) -> Dict[str, str]:
        return self.feature_states

    def get_drift_penalty(self, feat: str) -> float:
        m = self._metrics.get(feat)
        return m.drift_penalty if m else 1.0

    def get_all_penalties(self, features: List[str]) -> np.ndarray:
        return np.array([self.get_drift_penalty(f) for f in features])

    def build_report(self, symbol: str) -> FeatureHealthReport:
        summary: Dict[str, int] = {'HEALTHY': 0, 'WARNING': 0, 'DEGRADED': 0, 'CRITICAL': 0}
        features_dict: Dict[str, Dict] = {}

        for feat, m in self._metrics.items():
            summary[m.state] = summary.get(m.state, 0) + 1
            features_dict[feat] = asdict(m)

        top_drifters = sorted(
            [asdict(m) for m in self._metrics.values()],
            key=lambda x: x['score'], reverse=True,
        )[:10]

        return FeatureHealthReport(
            symbol=symbol,
            generated_at=datetime.now().isoformat(),
            n_features=len(self._metrics),
            summary=summary,
            top_drifters=top_drifters,
            features=features_dict,
        )

    def save_report(self, symbol: str) -> Path:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        report = self.build_report(symbol)
        path = _LOG_DIR / f"{symbol.replace('/', '_')}_feature_health.json"
        with open(path, 'w') as f:
            json.dump(asdict(report), f, indent=2)
        print(f"   [FeatureHealth] Report saved -> {path.name}")
        return path

    def print_summary(self) -> None:
        states = {'HEALTHY': 0, 'WARNING': 0, 'DEGRADED': 0, 'CRITICAL': 0}
        for s in self.feature_states.values():
            states[s] = states.get(s, 0) + 1
        print(f"   [FeatureHealth] {states['HEALTHY']} HEALTHY | "
              f"{states['WARNING']} WARNING | "
              f"{states['DEGRADED']} DEGRADED | "
              f"{states['CRITICAL']} CRITICAL")
        # Print top drifters
        top = sorted(self._metrics.values(), key=lambda m: m.score, reverse=True)[:5]
        if top:
            print("   [FeatureHealth] Top drifters:")
            for m in top:
                print(f"      {m.name:<35} score={m.score:.4f} [{m.state}] "
                      f"PSI={m.psi:.3f} KS={m.ks:.3f} mean_drift={m.mean_drift:.3f}")


class DynamicFeatureWeightingEngine:
    """Legacy interface — delegates to FeatureHealthManager state-based penalties."""

    def __init__(self, base_weight: float = 1.0):
        self.base_weight = base_weight

    def calculate_weights(
        self,
        feature_states: Dict[str, str],
        features: List[str],
    ) -> np.ndarray:
        weights = []
        for feat in features:
            state = feature_states.get(feat, 'HEALTHY')
            weights.append(self.base_weight * _STATE_PENALTIES.get(state, 1.0))
        return np.array(weights)
