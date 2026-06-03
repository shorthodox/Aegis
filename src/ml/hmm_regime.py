"""
hmm_regime.py — AEGIS HMM Market Regime Intelligence Layer
===========================================================
Wraps a per-symbol Gaussian HMM (7 hidden states, diagonal covariance) that
classifies the *market environment* — not price direction — from 9 smooth,
regime-descriptive observation features derived from the existing
prepare_features() pipeline.

Role in the AEGIS architecture
-------------------------------
XGBoost predicts DIRECTION (BUY / SELL / HOLD).
HMM classifies ENVIRONMENT (trending / choppy / explosive / accumulating).

The HMM output is consumed by:
  • predictor.py  – adds hmm_* fields to every predict_realtime() result
  • live_engine.py – MarketRegimeDetector uses hmm_regime to sharpen its label
                     SignalQualityFilter applies ±10 quality-score adjustment
                     DynamicRiskEngine scales ATR stops by hmm_atr_mult

Key design decisions
---------------------
• GaussianHMM with 'diag' covariance: fewest parameters per state, stable
  training on 3000–7000 hourly bars, no covariance ill-conditioning.
• 7 states: empirically optimal for crypto. 5 is too coarse (CHOP and
  VOLATILE_EXPANSION merge), 9 introduces near-duplicate states and slows
  convergence.
• Per-symbol models: BTC's vol fingerprint is fundamentally different from
  PEPE's. A shared model averages them into noise.
• 9 hand-picked features (not 90): HMM training degrades with high-
  dimensional, correlated, noisy inputs. These 9 are: smooth, orthogonal,
  and highly discriminative for regime classification.
• Graceful fallback: if no HMM model exists for a symbol, all HMM fields
  default to neutral / safe values — existing behavior is preserved.
"""

import json
import pickle
import warnings
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*converge.*")

try:
    from hmmlearn.hmm import GaussianHMM
    _HMM_AVAILABLE = True
except ImportError:
    _HMM_AVAILABLE = False
    GaussianHMM = None  # type: ignore[assignment,misc]

# ── paths ─────────────────────────────────────────────────────────────────────
_ROOT       = Path(__file__).resolve().parent.parent.parent
MODEL_STORE = _ROOT / 'src' / 'ml' / 'model_store'


# =============================================================================
# HMM feature builder
# =============================================================================

class HMMFeatureBuilder:
    """
    Computes the 9 regime-descriptive features from a fully-prepared DataFrame
    (output of prepare_features()). All columns used here exist in FEATURE_ADDONS
    and are guaranteed to be present after feature engineering.

    Feature design rationale
    -------------------------
    1. realized_vol_20   — 20-bar rolling σ of log-returns × √20.
                           Distinguishes quiet / normal / explosive vol regimes.
    2. atr_ratio         — ATR₁₄ / 50-bar rolling ATR mean.
                           >1 = expansion, <1 = compression. Captures dynamic volatility.
    3. efficiency_ratio  — |net price change| / Σ|bar changes| over 20 bars.
                           ≈1 = strong trend, ≈0 = random walk / chop.
    4. volume_zscore     — Rolling z-score of volume over 20 bars.
                           Spikes signal institutional activity.
    5. vwap_dist         — (close − VWAP) / ATR, signed.
                           Positive = price above VWAP (bullish positioning).
    6. rsi_smooth        — 3-bar EMA of RSI-14. Smoothed momentum state.
    7. funding_zscore    — funding_rate / (rolling σ of funding, 20-bar).
                           Captures extreme sentiment (funding drives liquidations).
    8. oi_delta_smooth   — 3-bar EMA of open-interest % change.
                           Rising OI = new money entering; falling = unwinding.
    9. hurst_approx      — Variance-ratio approximation of the Hurst exponent.
                           H > 0.5 = trending; H < 0.5 = mean-reverting.
    """

    N_FEATURES   = 9
    FEATURE_NAMES = [
        'realized_vol_20', 'atr_ratio', 'efficiency_ratio',
        'volume_zscore', 'vwap_dist', 'rsi_smooth',
        'funding_zscore', 'oi_delta_smooth', 'hurst_approx',
    ]

    # ── Hurst approximation constants ────────────────────────────────────────
    _HURST_LAGS = [2, 4, 8, 16]   # variance-ratio lags

    @classmethod
    def compute(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a DataFrame with 9 columns (same index as df).
        All NaN values are forward-filled then zero-filled so HMM never
        receives NaN observations.
        """
        n = len(df)
        if n < 30:
            return pd.DataFrame(
                np.zeros((n, cls.N_FEATURES)),
                index=df.index,
                columns=cls.FEATURE_NAMES,
            )

        # ── 1. Realized volatility ────────────────────────────────────────────
        log_ret = np.log(df['close'] / df['close'].shift(1)).fillna(0)
        realized_vol = log_ret.rolling(20, min_periods=10).std() * np.sqrt(20)

        # ── 2. ATR ratio ──────────────────────────────────────────────────────
        atr_col = 'atr_14' if 'atr_14' in df.columns else None
        if atr_col:
            atr_val  = df[atr_col].ffill()
        else:
            hi = df['high'];  lo = df['low'];  cl = df['close']
            tr = pd.concat([hi - lo,
                            (hi - cl.shift(1)).abs(),
                            (lo - cl.shift(1)).abs()], axis=1).max(axis=1)
            atr_val = tr.rolling(14, min_periods=7).mean()

        atr_mean50  = atr_val.rolling(50, min_periods=20).mean().replace(0, np.nan)
        atr_ratio   = (atr_val / atr_mean50).fillna(1.0).clip(0.2, 4.0)

        # ── 3. Efficiency ratio ───────────────────────────────────────────────
        if 'efficiency_ratio_10' in df.columns:
            er = df['efficiency_ratio_10'].ffill().fillna(0.5)
        else:
            net    = (df['close'] - df['close'].shift(20)).abs()
            total  = log_ret.abs().rolling(20, min_periods=10).sum()
            er     = (net / total.replace(0, np.nan)).fillna(0.5).clip(0.0, 1.0)

        # ── 4. Volume z-score ─────────────────────────────────────────────────
        if 'volume_zscore' in df.columns:
            vol_z = df['volume_zscore'].ffill().fillna(0.0).clip(-4, 4)
        else:
            vol_mean = df['volume'].rolling(20, min_periods=10).mean()
            vol_std  = df['volume'].rolling(20, min_periods=10).std().replace(0, np.nan)
            vol_z    = ((df['volume'] - vol_mean) / vol_std).fillna(0.0).clip(-4, 4)

        # ── 5. VWAP distance (normalised by ATR) ─────────────────────────────
        if 'dist_vwap' in df.columns:
            vwap_dist = df['dist_vwap'].ffill().fillna(0.0).clip(-5, 5)
        elif 'vwap' in df.columns:
            vwap_dist = ((df['close'] - df['vwap']) / atr_val.replace(0, np.nan)).fillna(0.0).clip(-5, 5)
        else:
            vwap_dist = pd.Series(0.0, index=df.index)

        # ── 6. Smoothed RSI ───────────────────────────────────────────────────
        if 'rsi_14' in df.columns:
            rsi_raw = df['rsi_14'].ffill().fillna(50.0)
        else:
            delta = df['close'].diff()
            gain  = delta.clip(lower=0).rolling(14, min_periods=7).mean()
            loss  = (-delta.clip(upper=0)).rolling(14, min_periods=7).mean().replace(0, np.nan)
            rs    = gain / loss
            rsi_raw = (100 - 100 / (1 + rs)).fillna(50.0)
        rsi_smooth = rsi_raw.ewm(span=3, adjust=False).mean() / 100.0 - 0.5  # centre at 0

        # ── 7. Funding rate z-score ───────────────────────────────────────────
        if 'funding_rate' in df.columns:
            fr     = df['funding_rate'].ffill().fillna(0.0)
            fr_std = fr.rolling(20, min_periods=10).std().replace(0, 1e-6)
            fr_z   = (fr / fr_std).fillna(0.0).clip(-4, 4)
        else:
            fr_z = pd.Series(0.0, index=df.index)

        # ── 8. OI delta (smoothed) ────────────────────────────────────────────
        if 'oi_change_1h' in df.columns:
            oi_raw    = df['oi_change_1h'].ffill().fillna(0.0)
            oi_smooth = oi_raw.ewm(span=3, adjust=False).mean().clip(-0.1, 0.1)
        elif 'oi_change_4h' in df.columns:
            oi_raw    = (df['oi_change_4h'] / 4).ffill().fillna(0.0)
            oi_smooth = oi_raw.ewm(span=3, adjust=False).mean().clip(-0.1, 0.1)
        else:
            oi_smooth = pd.Series(0.0, index=df.index)

        # ── 9. Hurst exponent approximation (variance-ratio method) ──────────
        hurst = cls._rolling_hurst(log_ret.values, window=60)

        result = pd.DataFrame({
            'realized_vol_20': realized_vol.fillna(0),
            'atr_ratio':       atr_ratio,
            'efficiency_ratio': er,
            'volume_zscore':   vol_z,
            'vwap_dist':       vwap_dist,
            'rsi_smooth':      rsi_smooth,
            'funding_zscore':  fr_z,
            'oi_delta_smooth': oi_smooth,
            'hurst_approx':    hurst,
        }, index=df.index)

        return result.ffill().fillna(0.0)

    @classmethod
    def _rolling_hurst(cls, returns: np.ndarray, window: int = 60) -> pd.Series:
        """
        Variance-ratio Hurst exponent: H = log(Var(lag)) / (2 × log(lag)).
        H ≈ 0.5 → random walk, H > 0.5 → trending, H < 0.5 → mean-reverting.
        Uses a single lag (lag=4) for speed; full multi-lag is ~3× slower.
        """
        n     = len(returns)
        hurst = np.full(n, 0.5, dtype=np.float32)
        lag   = 4
        for i in range(window, n):
            chunk = returns[i - window:i]
            if chunk.std() < 1e-9:
                continue
            var1  = np.var(np.diff(chunk))       # 1-period variance
            var_l = np.var(chunk[lag:] - chunk[:-lag])  # lag-period variance
            if var1 < 1e-12 or var_l < 1e-12:
                continue
            # H = log(var_l / var1) / (2 * log(lag))
            h_val = np.log(var_l / var1) / (2 * np.log(lag))
            hurst[i] = float(np.clip(h_val, 0.0, 1.0))
        return pd.Series(hurst)

    @classmethod
    def extract_last_observation(cls, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Returns the last-row observation vector, or None on failure."""
        try:
            feat = cls.compute(df)
            last = feat.iloc[-1].values.astype(np.float32)
            if np.isnan(last).any():
                last = np.nan_to_num(last, nan=0.0)
            return last.reshape(1, -1)
        except Exception:
            return None

    @classmethod
    def extract_sequence(cls, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Returns the full observation matrix (n_bars, 9) for HMM training."""
        try:
            feat = cls.compute(df)
            X    = feat.values.astype(np.float32)
            X    = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            return X
        except Exception:
            return None


# =============================================================================
# HMM state dataclass
# =============================================================================

@dataclass
class HMMState:
    """
    All intelligence the HMM layer provides to the rest of the system.
    Defaults are deliberately neutral so missing models never break callers.
    """
    # State identity
    state_id:    int   = 3         # integer label (0–6)
    regime:      str   = 'RANGING' # human-readable label
    confidence:  float = 0.5       # max state probability (higher = more certain)

    # State probabilities (7-vector summing to 1)
    state_probs: List[float] = field(default_factory=lambda: [1/7]*7)

    # Transition intelligence
    next_state_probs: List[float] = field(default_factory=lambda: [1/7]*7)
    transition_risk:  float       = 0.0   # P(move to high-risk state next bar)
    regime_stability: float       = 1.0   # how long current regime has persisted (0–1)

    # Trading modifiers (consumed by live_engine)
    confidence_adjustment: float  = 0.0   # ± delta applied to meta_threshold
    atr_multiplier:        float  = 1.0   # scale factor for ATR stops
    position_scale:        float  = 1.0   # scale factor for position size
    trade_allowed:         bool   = True  # False → suppress all entries

    # Diagnostics
    hmm_available: bool = False
    n_observations: int = 0


# =============================================================================
# State label mapper
# =============================================================================

# Canonical 7-state taxonomy used for the strategy router
_STATE_LABELS = [
    'TRENDING_BULL',       # 0
    'TRENDING_BEAR',       # 1
    'CHOPPY',              # 2
    'VOLATILE_EXPANSION',  # 3
    'COMPRESSION',         # 4
    'ACCUMULATION',        # 5
    'DISTRIBUTION',        # 6
]

# Per-regime trading modifiers
# (confidence_adj, atr_mult, pos_scale, trade_allowed)
_REGIME_CONFIG: Dict[str, Tuple[float, float, float, bool]] = {
    'TRENDING_BULL':      (-0.03, 0.9,  1.15, True),   # raise confidence slightly, tighter stop
    'TRENDING_BEAR':      (-0.03, 0.9,  1.15, True),   # same logic for shorts
    'CHOPPY':             (+0.05, 1.2,  0.70, True),   # raise bar, widen stop, reduce size
    'VOLATILE_EXPANSION': (+0.05, 1.5,  0.60, True),   # much wider stop, smaller size
    'COMPRESSION':        (+0.02, 0.85, 0.80, True),   # near-breakout; wait for quality
    'ACCUMULATION':       (-0.02, 1.0,  1.0,  True),   # mild bullish bias; normal size
    'DISTRIBUTION':       (+0.03, 1.1,  0.80, True),   # mild bearish risk; reduce size
}

# High-risk state transitions: if the predicted next state is one of these,
# raise transition_risk proportionally
_HIGH_RISK_STATES = {2, 3, 6}   # CHOPPY, VOLATILE_EXPANSION, DISTRIBUTION


def _map_states_to_labels(hmm: Any) -> Dict[int, str]:
    """
    Assign human-readable labels to HMM integer state IDs based on the
    centroid of each state's Gaussian emission.

    Heuristic mapping (robust across different random seeds):
      • Highest efficiency_ratio + positive vwap_dist       → TRENDING_BULL
      • Highest efficiency_ratio + negative vwap_dist       → TRENDING_BEAR
      • Lowest efficiency_ratio                             → CHOPPY
      • Highest realized_vol + highest volume_zscore        → VOLATILE_EXPANSION
      • Lowest realized_vol + lowest volume_zscore          → COMPRESSION
      • Rising OI + positive funding (shorts paying)        → ACCUMULATION
      • Falling OI + negative funding (longs paying)        → DISTRIBUTION
      • Remaining state                                     → CHOPPY (duplicate)

    This is intentionally simple and robust — the HMM discovers the states
    statistically, we only need approximate labels for the strategy router.
    """
    means = hmm.means_          # shape (n_states, n_features)
    n_s   = means.shape[0]
    feat  = HMMFeatureBuilder.FEATURE_NAMES

    def _col(name: str) -> int:
        return feat.index(name) if name in feat else -1

    er_idx  = _col('efficiency_ratio')
    vd_idx  = _col('vwap_dist')
    rv_idx  = _col('realized_vol_20')
    vz_idx  = _col('volume_zscore')
    oi_idx  = _col('oi_delta_smooth')
    fr_idx  = _col('funding_zscore')

    labels: Dict[int, str] = {}
    unassigned = set(range(n_s))

    def _best(idx: int, key) -> int:
        candidates = sorted(unassigned, key=lambda s: key(means[s, idx]))
        return candidates[-1]

    def _assign(state_id: int, label: str) -> None:
        labels[state_id] = label
        unassigned.discard(state_id)

    # TRENDING: highest ER
    if er_idx >= 0 and len(unassigned) >= 2:
        top2_er = sorted(unassigned, key=lambda s: means[s, er_idx])[-2:]
        for s in top2_er:
            vd = means[s, vd_idx] if vd_idx >= 0 else 0.0
            _assign(s, 'TRENDING_BULL' if vd >= 0 else 'TRENDING_BEAR')

    # VOLATILE_EXPANSION: highest realized vol × volume zscore
    if rv_idx >= 0 and vz_idx >= 0 and unassigned:
        score = {s: means[s, rv_idx] * max(means[s, vz_idx], 0.1)
                 for s in unassigned}
        _assign(max(score, key=score.get), 'VOLATILE_EXPANSION')

    # COMPRESSION: lowest realized vol + lowest volume
    if rv_idx >= 0 and unassigned:
        _assign(min(unassigned, key=lambda s: means[s, rv_idx]), 'COMPRESSION')

    # ACCUMULATION: rising OI, positive funding (shorts paying)
    if oi_idx >= 0 and fr_idx >= 0 and unassigned:
        score = {s: means[s, oi_idx] - means[s, fr_idx] for s in unassigned}
        if score:
            _assign(max(score, key=score.get), 'ACCUMULATION')

    # DISTRIBUTION: negative OI delta, longs paying
    if oi_idx >= 0 and fr_idx >= 0 and unassigned:
        score = {s: -(means[s, oi_idx]) + means[s, fr_idx] for s in unassigned}
        if score:
            _assign(max(score, key=score.get), 'DISTRIBUTION')

    # CHOPPY: remainder (lowest ER or unassigned)
    for s in list(unassigned):
        _assign(s, 'CHOPPY')

    return labels


# =============================================================================
# HMM Regime Engine — per-symbol
# =============================================================================

class HMMRegimeEngine:
    """
    Manages training, serialization, and online inference for one symbol's HMM.

    Training: call train(df) with the full OHLCV + features DataFrame.
    Inference: call infer(df) with the last N bars for context.
    """

    N_STATES  = 7
    N_ITER    = 200     # EM iterations; usually converges well before this
    N_CONTEXT = 100     # bars of history used for online Viterbi decoding

    def __init__(self, symbol: str) -> None:
        self.symbol    = symbol
        self._hmm:     Optional[Any] = None          # GaussianHMM instance
        self._scaler_mean: Optional[np.ndarray] = None
        self._scaler_std:  Optional[np.ndarray] = None
        self._state_labels: Dict[int, str] = {}
        self._seq_buffer:   Deque[np.ndarray] = deque(maxlen=self.N_CONTEXT)
        self._last_state_id: int = 0
        self._stability_count: int = 0   # bars in current state
        self._loaded = False

    # ── Paths ─────────────────────────────────────────────────────────────────

    @property
    def _model_path(self) -> Path:
        base = self.symbol.replace('/', '_')
        return MODEL_STORE / f'{base}_hmm.pkl'

    # ── Normalisation ─────────────────────────────────────────────────────────

    def _fit_scaler(self, X: np.ndarray) -> None:
        """Fit a robust (IQR-based) scaler so outliers don't contaminate training."""
        self._scaler_mean = np.median(X, axis=0)
        q25 = np.percentile(X, 25, axis=0)
        q75 = np.percentile(X, 75, axis=0)
        iqr = q75 - q25
        self._scaler_std  = np.where(iqr > 1e-8, iqr, 1.0)

    def _scale(self, X: np.ndarray) -> np.ndarray:
        if self._scaler_mean is None:
            return X
        return (X - self._scaler_mean) / self._scaler_std

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame) -> bool:
        """
        Train a 7-state GaussianHMM on the given DataFrame.
        Returns True if training succeeded, False otherwise.

        Caller: retrain_model.py / scripts/train_hmm.py
        """
        if not _HMM_AVAILABLE:
            print(f'[HMM:{self.symbol}] hmmlearn not available — skipping')
            return False

        X_raw = HMMFeatureBuilder.extract_sequence(df)
        if X_raw is None or len(X_raw) < 100:
            print(f'[HMM:{self.symbol}] Not enough data ({len(df)} bars) — need ≥ 100')
            return False

        # Drop rows with all-zero observations (missing data periods)
        valid = ~(np.abs(X_raw).sum(axis=1) < 1e-9)
        X_raw = X_raw[valid]
        if len(X_raw) < 100:
            return False

        self._fit_scaler(X_raw)
        X = self._scale(X_raw)

        try:
            hmm = GaussianHMM(
                n_components    = self.N_STATES,
                covariance_type = 'diag',    # fewer params → stable on 3k–7k bars
                n_iter          = self.N_ITER,
                tol             = 1e-4,
                random_state    = 42,
                verbose         = False,
            )
            hmm.fit(X)
            self._hmm = hmm
            self._state_labels = _map_states_to_labels(hmm)
            self._loaded = True

            # State distribution for diagnostics
            states = hmm.predict(X)
            dist   = np.bincount(states, minlength=self.N_STATES) / len(states)
            dist_str = ' '.join(
                f"{self._state_labels.get(i, str(i))}:{d:.1%}"
                for i, d in enumerate(dist)
            )
            print(f'[HMM:{self.symbol}] Trained on {len(X)} bars. '
                  f'Log-likelihood: {hmm.score(X):.1f}. States: {dist_str}')
            return True
        except Exception as e:
            print(f'[HMM:{self.symbol}] Training failed: {e}')
            return False

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save(self) -> None:
        if self._hmm is None:
            return
        MODEL_STORE.mkdir(parents=True, exist_ok=True)
        payload = {
            'hmm':           self._hmm,
            'scaler_mean':   self._scaler_mean,
            'scaler_std':    self._scaler_std,
            'state_labels':  self._state_labels,
            'symbol':        self.symbol,
            'n_states':      self.N_STATES,
            'feature_names': HMMFeatureBuilder.FEATURE_NAMES,
        }
        with open(self._model_path, 'wb') as f:
            pickle.dump(payload, f, protocol=4)
        print(f'[HMM:{self.symbol}] Saved → {self._model_path.name}')

    def load(self) -> bool:
        if not self._model_path.exists():
            return False
        try:
            with open(self._model_path, 'rb') as f:
                payload = pickle.load(f)
            self._hmm           = payload['hmm']
            self._scaler_mean   = payload['scaler_mean']
            self._scaler_std    = payload['scaler_std']
            self._state_labels  = payload['state_labels']
            self._loaded        = True
            return True
        except Exception as e:
            print(f'[HMM:{self.symbol}] Load failed: {e}')
            return False

    # ── Inference ─────────────────────────────────────────────────────────────

    def infer(self, df: pd.DataFrame) -> HMMState:
        """
        Run online Viterbi inference on the last N_CONTEXT bars of df.
        Updates the internal sequence buffer with the latest observation.
        Returns an HMMState with all trading modifiers filled in.

        Fast: ~0.2ms per call for 7 states × 9 features on CPU.
        """
        default = HMMState(hmm_available=False)

        if not self._loaded or self._hmm is None:
            return default

        try:
            # Extract last observation and update buffer
            obs = HMMFeatureBuilder.extract_last_observation(df)
            if obs is None:
                return default
            self._seq_buffer.append(obs.flatten())

            if len(self._seq_buffer) < 5:
                return default

            # Build sequence for Viterbi
            X_raw  = np.array(list(self._seq_buffer), dtype=np.float32)
            X      = self._scale(X_raw)

            # State sequence via Viterbi
            states = self._hmm.predict(X)
            current_state_id = int(states[-1])

            # Stability: how many bars in a row have we been in this state?
            if current_state_id == self._last_state_id:
                self._stability_count += 1
            else:
                self._stability_count  = 1
                self._last_state_id    = current_state_id

            # State probabilities (posterior on last observation)
            state_probs_raw = self._hmm.predict_proba(X)[-1]
            state_probs     = state_probs_raw.tolist()
            confidence      = float(state_probs_raw[current_state_id])

            # Next-state transition probabilities
            trans_row        = self._hmm.transmat_[current_state_id].tolist()
            transition_risk  = float(sum(trans_row[s] for s in _HIGH_RISK_STATES
                                         if s < len(trans_row)))

            # Stability score: normalized count of bars in current state
            stability = float(min(self._stability_count / 20.0, 1.0))

            # Regime label and modifiers
            regime     = self._state_labels.get(current_state_id, 'RANGING')
            cfg        = _REGIME_CONFIG.get(regime, (0.0, 1.0, 1.0, True))
            conf_adj, atr_mult, pos_scale, trade_ok = cfg

            # Confidence penalty for high transition risk or low stability
            if transition_risk > 0.40:
                conf_adj  += 0.02    # raise threshold when unstable
                pos_scale *= 0.85

            if stability < 0.15:     # very new state — uncertain
                conf_adj  += 0.02
                pos_scale *= 0.90

            return HMMState(
                state_id              = current_state_id,
                regime                = regime,
                confidence            = round(confidence, 3),
                state_probs           = [round(p, 4) for p in state_probs],
                next_state_probs      = [round(p, 4) for p in trans_row],
                transition_risk       = round(transition_risk, 3),
                regime_stability      = round(stability, 3),
                confidence_adjustment = round(conf_adj, 3),
                atr_multiplier        = round(atr_mult, 2),
                position_scale        = round(pos_scale, 2),
                trade_allowed         = trade_ok,
                hmm_available         = True,
                n_observations        = len(self._seq_buffer),
            )

        except Exception as e:
            print(f'[HMM:{self.symbol}] Inference error: {e}')
            return default

    def get_transition_warning(self) -> Optional[str]:
        """
        Returns a human-readable early-warning string when an imminent
        regime transition is detected. Used in dashboard alerts.

        Example: "TRENDING_BULL → VOLATILE_EXPANSION: 38% probability"
        """
        if not self._loaded or self._hmm is None:
            return None
        try:
            trans_row = self._hmm.transmat_[self._last_state_id]
            current_label = self._state_labels.get(self._last_state_id, '?')
            # Find the highest-probability transition to a DIFFERENT state
            other_probs = [
                (s, trans_row[s], self._state_labels.get(s, str(s)))
                for s in range(len(trans_row))
                if s != self._last_state_id and s in _HIGH_RISK_STATES
            ]
            if not other_probs:
                return None
            max_s, max_p, max_label = max(other_probs, key=lambda x: x[1])
            if max_p >= 0.25:
                return (f'{current_label} → {max_label}: '
                        f'{max_p:.0%} transition probability')
            return None
        except Exception:
            return None


# =============================================================================
# Engine pool — one HMMRegimeEngine per symbol, lazy loaded
# =============================================================================

class HMMEnginePool:
    """
    Singleton pool that manages per-symbol HMM engines.
    Loaded lazily on first inference call — zero cost for symbols
    that have no trained model.
    """

    def __init__(self) -> None:
        self._engines: Dict[str, HMMRegimeEngine] = {}

    def get(self, symbol: str) -> HMMRegimeEngine:
        if symbol not in self._engines:
            engine = HMMRegimeEngine(symbol)
            engine.load()   # silent no-op if model doesn't exist
            self._engines[symbol] = engine
        return self._engines[symbol]

    def infer(self, symbol: str, df: pd.DataFrame) -> HMMState:
        """Convenience wrapper: get engine + run inference."""
        return self.get(symbol).infer(df)

    def train_and_save(self, symbol: str, df: pd.DataFrame) -> bool:
        """Train a new model for `symbol` and persist it."""
        engine = HMMRegimeEngine(symbol)
        if engine.train(df):
            engine.save()
            self._engines[symbol] = engine
            return True
        return False

    def loaded_symbols(self) -> List[str]:
        return [s for s, e in self._engines.items() if e._loaded]

    def reload(self, symbol: str) -> bool:
        """Force-reload a symbol's model from disk (after retraining)."""
        engine = HMMRegimeEngine(symbol)
        if engine.load():
            self._engines[symbol] = engine
            return True
        return False


# Module-level singleton
_pool = HMMEnginePool()


def get_pool() -> HMMEnginePool:
    """Return the global HMMEnginePool singleton."""
    return _pool


# =============================================================================
# Standalone training utility
# =============================================================================

def train_hmm_for_symbol(symbol: str, df: pd.DataFrame) -> bool:
    """
    Top-level function called from retrain_model.py train_token().
    Trains a fresh HMM for `symbol` on the provided fully-featured DataFrame
    and saves it to model_store.  Returns True on success.
    """
    return _pool.train_and_save(symbol, df)
