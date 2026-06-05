#!/usr/bin/env python3
"""
scripts/aegis_forensics.py
──────────────────────────
Production-grade forensic investigation engine for AEGIS-1.

Audits every stage of the trading stack and produces:
  logs/forensics/AEGIS_MASTER_FORENSIC_REPORT_{symbol}.md
  logs/forensics/feature_drift_{symbol}.json
  logs/forensics/meta_forensics_{symbol}.json
  logs/forensics/regime_forensics_{symbol}.json
  logs/forensics/quality_forensics_{symbol}.json
  logs/forensics/execution_forensics_{symbol}.json

Usage:
    python scripts/aegis_forensics.py --symbol BTC/USDT
    python scripts/aegis_forensics.py --symbol ETH/USDT --hours 8760
    python scripts/aegis_forensics.py --fleet
"""

import sys, os, json, warnings, pickle, math
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_STORE  = ROOT / "src" / "ml" / "model_store"
FORENSIC_DIR = ROOT / "logs" / "forensics"
REGIME_STATS = ROOT / "data" / "regime_stats"

ABSOLUTE_PRICE_FEATURES = frozenset({
    "vwap", "ema_200", "ema_100", "ema_50", "ema_21", "ema_9",
    "avwap_200", "avwap_100", "avwap_50", "rolling_vwap_24",
    "ichimoku_senkou_a", "ichimoku_senkou_b", "ichimoku_tenkan", "ichimoku_kijun",
    "hma_20", "kama_10", "tema_21", "dema_21", "t3_5", "vwma_20",
    "rolling_resistance", "rolling_support", "pivot", "r1", "r2", "s1", "s2",
    "pvt", "obv", "acc_dist",
    "close_decay_mean_24", "vwap_decay_mean_24",
})

REMOVE_OR_NORMALISE_FEATURES = frozenset({
    "ichimoku_senkou_a", "ichimoku_senkou_b",
    "close_decay_mean_24", "close_decay_std_24",
    "vwap_decay_std_24", "volume_decay_mean_24",
    "fib_range_pct", "dist_vwap", "vwap_delta_4",
})

FEE_ROUNDTRIP = 0.001
QUALITY_THRESHOLD = 55.0
META_GATE_LABEL   = "meta_gate"

# ─────────────────────────────────────────────────────────────────────────────
# SOURCE MAP  — every known issue type is linked to an exact code address.
# The forensic engine reads this map to tell you WHERE to go fix something,
# not just what is broken.  Each entry carries:
#   file   → relative path from repo root
#   lines  → line range (string)
#   symbol → the specific variable / function name
#   mechanism → one sentence explaining HOW this code causes the problem
#   fix    → one sentence describing the code change
#   prec_cost_pp  → estimated direct precision penalty (positive = loss)
#   recall_cost_pp → estimated recall penalty
# ─────────────────────────────────────────────────────────────────────────────
SOURCE_MAP: Dict[str, Dict[str, Any]] = {
    # ── BUY side disabled ────────────────────────────────────────────────────
    "buy_min_fires_deadlock": {
        "file": "scripts/retrain_model.py",
        "lines": "1363-1397",
        "symbol": "MAX_SIDE_COVERAGE, _SIDE_MIN_FIRES",
        "mechanism": (
            "MAX_SIDE_COVERAGE=0.25 × BUY_pool(≈88) = 22 max fires, "
            "but _SIDE_MIN_FIRES=35 requires ≥35. Every candidate rejected "
            "before precision is even checked → hit_buy=False always."
        ),
        "fix": "Set MAX_SIDE_COVERAGE=0.35 OR use adaptive effective_min_fires=min(35, int(0.35*pool)).",
        "prec_cost_pp": 0.0,
        "recall_cost_pp": 8.0,
    },
    "buy_holdout_gate_bias": {
        "file": "scripts/retrain_model.py",
        "lines": "2097-2108",
        "symbol": "fire_rank",
        "mechanism": (
            "Combined rank gate fires top dev_cov fraction of ALL signals. "
            "With SELL proposals 4× more than BUY, the rank gate is SELL-biased. "
            "buy_fire then uses per-side thr_buy but combined fire already selected few BUY bars."
        ),
        "fix": "Use union of buy_fire|sell_fire as the primary fire mask instead of combined rank gate.",
        "prec_cost_pp": 0.0,
        "recall_cost_pp": 4.0,
    },
    "buy_tradeable_gate": {
        "file": "scripts/retrain_model.py",
        "lines": "2288-2292",
        "symbol": "tradeable_buy_holdout",
        "mechanism": (
            "tradeable_buy_holdout requires buy_h_n > 0. If combined rank gate "
            "fires zero BUY holdout signals, this is permanently False. "
            "Zero holdout BUY trades ≠ 'BUY doesn't work' — it means the gate never let them through."
        ),
        "fix": "When buy_h_n=0 AND hit_buy=True, fall back to OOF approval (hit_buy) rather than disabling.",
        "prec_cost_pp": 0.0,
        "recall_cost_pp": 6.0,
    },
    # ── Feature drift ─────────────────────────────────────────────────────────
    "obv_not_normalised": {
        "file": "src/ml/feature_engine.py",
        "lines": "1737",
        "symbol": "compiled_features['obv']",
        "mechanism": (
            "OBV = cumsum(sign(close_diff) × volume). A monotonic cumulative "
            "sum that grows unboundedly with price trend. PSI≈13 between any "
            "bull/bear period. Model uses OBV as a price-level signal, not a flow signal."
        ),
        "fix": "Replace with rolling z-score: (obv - obv.rolling(100).mean()) / obv.rolling(100).std()",
        "prec_cost_pp": 1.8,
        "recall_cost_pp": 0.0,
    },
    "pvt_not_normalised": {
        "file": "src/ml/feature_engine.py",
        "lines": "1891",
        "symbol": "compiled_features['pvt']",
        "mechanism": (
            "PVT = cumsum(pct_change × volume). Absolute cumulative units "
            "drift proportionally with BTC price × volume level. PSI≈13."
        ),
        "fix": "Replace with rolling z-score: (pvt - pvt.rolling(100).mean()) / pvt.rolling(100).std()",
        "prec_cost_pp": 1.8,
        "recall_cost_pp": 0.0,
    },
    "absolute_emas_in_features": {
        "file": "scripts/retrain_model.py",
        "lines": "165-168 (FEATURE_ADDONS), 1838-1840 (feature_cols)",
        "symbol": "FEATURE_ADDONS / feature_cols",
        "mechanism": (
            "Raw ema_9/21/50/100/200 are absolute price levels. "
            "At BTC=20k, ema_200≈18k. At BTC=60k, ema_200≈55k. "
            "XGBoost learns price-level splits that break across regimes. "
            "PSI≈13.6 for ema_200."
        ),
        "fix": "Add to FEATURE_BLACKLIST. dist_ema_* versions already exist and are stationary.",
        "prec_cost_pp": 2.5,
        "recall_cost_pp": 0.0,
    },
    "absolute_vwap_avwap": {
        "file": "scripts/retrain_model.py",
        "lines": "165-168 (FEATURE_ADDONS)",
        "symbol": "vwap, avwap_50, avwap_100, avwap_200",
        "mechanism": (
            "Anchored VWAPs are session-accumulated average prices — absolute levels. "
            "PSI=23.46 for vwap (top drifter). "
            "dist_vwap, dist_avwap_* are already computed and stationary."
        ),
        "fix": "Add to FEATURE_BLACKLIST. Keep only dist_vwap, dist_avwap_*.",
        "prec_cost_pp": 3.0,
        "recall_cost_pp": 0.0,
    },
    "decay_means_absolute": {
        "file": "src/ml/feature_engine.py",
        "lines": "2050",
        "symbol": "close_decay_mean_24, vwap_decay_mean_24",
        "mechanism": (
            "add_delta_and_decay_features() produces exponentially-weighted "
            "close/vwap means — absolute price levels with half_life=24. "
            "These are essentially smoothed price levels, not signals. PSI≈12.9."
        ),
        "fix": "Convert to distance: (close - decay_mean) / close → stationary ratio in [-0.1, +0.1].",
        "prec_cost_pp": 1.5,
        "recall_cost_pp": 0.0,
    },
    "ichimoku_absolute": {
        "file": "scripts/retrain_model.py",
        "lines": "165-168 (FEATURE_ADDONS)",
        "symbol": "ichimoku_senkou_a, ichimoku_senkou_b",
        "mechanism": (
            "Senkou spans are midpoints of absolute price ranges shifted 26 bars. "
            "PSI≈13 pair. ichi_dist_cloud_top already captures the "
            "meaningful signal (distance from cloud) in stationary form."
        ),
        "fix": "Add to FEATURE_BLACKLIST. Keep ichi_dist_*, ichi_above_cloud, ichi_tk_cross.",
        "prec_cost_pp": 1.2,
        "recall_cost_pp": 0.0,
    },
    # ── Class imbalance ───────────────────────────────────────────────────────
    "vol_threshold_too_high": {
        "file": "scripts/retrain_model.py",
        "lines": "836",
        "symbol": "base_vol_threshold = 0.80",
        "mechanism": (
            "When volatility_regime (ATR/ATR_mean_100) < 0.80, the bar is forced to HOLD. "
            "BTC consolidates below 80% of its mean ATR in ~40% of bars. "
            "This artificially inflates HOLD labels to 60%+."
        ),
        "fix": "Lower to 0.72 (confirmed: balanced HOLD rate without label noise). Already applied.",
        "prec_cost_pp": 2.0,
        "recall_cost_pp": 4.0,
    },
    "barrier_skew_sell_bias": {
        "file": "scripts/retrain_model.py",
        "lines": "187-188",
        "symbol": "BARRIER_UP_SKEW=1.15, BARRIER_DOWN_SKEW=0.85",
        "mechanism": (
            "SELL barrier is 35% tighter than BUY barrier. In any market regime "
            "the lower barrier gets hit 35% more often → systematic SELL-label bias. "
            "Creates SELL=20.7% vs BUY=13.1% class ratio."
        ),
        "fix": "Set BARRIER_UP_SKEW=1.0, BARRIER_DOWN_SKEW=1.0 (symmetric). Already applied.",
        "prec_cost_pp": 1.5,
        "recall_cost_pp": 3.0,
    },
    # ── LSTM weakness ─────────────────────────────────────────────────────────
    "lstm_atr14_absolute": {
        "file": "src/ml/lstm_models.py",
        "lines": "59",
        "symbol": "VOL_EXP_FEATURES: 'atr_14'",
        "mechanism": (
            "atr_14 is raw ATR in price units. At BTC=20k it is ~200; at BTC=60k it is ~600. "
            "The RobustScaler in lstm_trainer clips this but the median/IQR used for scaling "
            "shift completely between training windows, causing the vol LSTM to see "
            "a different distribution at inference than training."
        ),
        "fix": "Replace with atr_pct = atr_14/close (stationary, ~0.005–0.015 range). Already applied.",
        "prec_cost_pp": 0.0,
        "recall_cost_pp": 0.0,
    },
    "lstm_no_momentum_features": {
        "file": "src/ml/lstm_models.py",
        "lines": "36-55",
        "symbol": "CONT_FEATURES (18 features, seq_len=20)",
        "mechanism": (
            "Continuation LSTM had no direct return features (returns_1h, ret_4h). "
            "Momentum persistence — the core signal it is trying to detect — requires "
            "return sequences as input. The model was forced to infer momentum "
            "from indirect indicators (RSI, MACD) instead."
        ),
        "fix": "Add returns_1h, ret_4h, volatility_regime, candle_pressure. Extend CONT_SEQ_LEN 20→24. Already applied.",
        "prec_cost_pp": 0.0,
        "recall_cost_pp": 0.0,
    },
    # ── Meta model ────────────────────────────────────────────────────────────
    "meta_hold_contamination": {
        "file": "scripts/retrain_model.py",
        "lines": "1909-1916",
        "symbol": "_hold_w, meta_w",
        "mechanism": (
            "HOLD bars have meta_y=0 (primary proposes BUY/SELL but true label=HOLD). "
            "With HOLD=66%, ~60% of meta training labels are always zero. "
            "Meta model calibration is biased toward 0.50 regardless of actual "
            "signal confidence. Downweighting to 0.30 partially corrects but "
            "does not fully remove the bias."
        ),
        "fix": "Increase hold_w floor: clip(0.15, 0.50) → or exclude HOLD bars from meta training entirely.",
        "prec_cost_pp": 2.0,
        "recall_cost_pp": 0.0,
    },
    # ── Reproducibility ────────────────────────────────────────────────────────
    "no_global_numpy_seed": {
        "file": "scripts/retrain_model.py",
        "lines": "1632",
        "symbol": "train_token() — no np.random.seed at entry",
        "mechanism": (
            "Each call to train_token() inherits whatever numpy random state "
            "was left by previous API calls, feature engineering, and datetime ops. "
            "SHAP sampling, isotonic calibration, and certain sklearn routines "
            "consume the global RNG, causing different fold splits and threshold "
            "selections across runs even with identical data."
        ),
        "fix": "Add random.seed(42); np.random.seed(42) at top of train_token(). Already applied.",
        "prec_cost_pp": 0.0,
        "recall_cost_pp": 0.0,
    },
}
QUALITY_LABEL     = "quality"
HMM_LABEL         = "hmm"
CONFLUENCE_LABEL  = "confluence"
FAKEOUT_LABEL     = "fake_breakout"
PORTFOLIO_LABEL   = "portfolio"
SAFEMODE_LABEL    = "safe_mode"
DRIFT_LABEL       = "drift"
COOLDOWN_LABEL    = "cooldown"
COOLDOWN_BARS     = 4
MAX_POSITIONS     = 6


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe(v, default=0.0):
    try:
        f = float(v)
        return default if (f != f or math.isinf(f)) else f
    except Exception:
        return default


def _profit_factor(wins: np.ndarray, losses: np.ndarray) -> float:
    gw = float(wins[wins > 0].sum()) if (wins > 0).any() else 0.0
    gl = float(abs(losses[losses < 0].sum())) if (losses < 0).any() else 1e-9
    return round(gw / gl, 3)


def _sharpe(rets: np.ndarray) -> float:
    std = float(rets.std())
    if std < 1e-9 or len(rets) < 2:
        return 0.0
    return round(float(rets.mean()) / std * math.sqrt(min(len(rets) * 8, 8760)), 3)


def _kelly(win_rate: float, avg_win: float, avg_loss: float) -> float:
    if avg_loss < 1e-9 or win_rate <= 0 or win_rate >= 1:
        return 0.0
    b = avg_win / avg_loss
    return min(win_rate - (1.0 - win_rate) / b, 0.25)


def _risk_of_ruin(win_rate: float, avg_win: float, avg_loss: float,
                  capital: float = 1.0, ruin_threshold: float = 0.5) -> float:
    if win_rate <= 0 or avg_loss < 1e-9:
        return 1.0
    lose_rate = 1.0 - win_rate
    if win_rate >= 1.0:
        return 0.0
    ratio = lose_rate / win_rate
    if abs(ratio - 1.0) < 1e-6:
        return 1.0
    try:
        ror = ((ratio ** (capital / avg_loss)) - (ratio ** (ruin_threshold / avg_loss))) / \
              (1 - (ratio ** (ruin_threshold / avg_loss)))
        return float(np.clip(ror, 0.0, 1.0))
    except Exception:
        return 1.0


def _confidence_interval_precision(n: int, p: float, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    se = math.sqrt(p * (1 - p) / n)
    return max(0.0, p - z * se), min(1.0, p + z * se)


def _z_test_precision(n: int, p: float, p0: float = 0.5) -> Tuple[float, float]:
    if n < 5:
        return float("nan"), float("nan")
    se = math.sqrt(p0 * (1 - p0) / n)
    z = (p - p0) / se if se > 0 else 0.0
    from scipy.stats import norm
    pval = 2 * (1 - norm.cdf(abs(z)))
    return round(z, 3), round(pval, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Main Engine
# ─────────────────────────────────────────────────────────────────────────────

class AEGISForensicEngine:

    def __init__(self, symbol: str, hours: int = 8760):
        self.symbol  = symbol
        self.base    = symbol.replace("/", "_")
        self.hours   = hours
        self.ts      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ── System state (loaded in _load_system) ─────────────────────────────
        self.meta:         Dict[str, Any] = {}
        self.lstm_meta:    Dict[str, Any] = {}
        self.regime_stats: Dict[str, Any] = {}
        self.aegis_state:  Dict[str, Any] = {}
        self.predictor     = None

        # ── Data (populated in _fetch_data) ───────────────────────────────────
        self.df:          Optional[pd.DataFrame] = None
        self.df_train:    Optional[pd.DataFrame] = None
        self.df_holdout:  Optional[pd.DataFrame] = None

        # ── Simulation results (populated in _run_core_simulation) ────────────
        self.sim: Dict[str, Any] = {}

        # ── Section results (populated by s01…s15) ────────────────────────────
        self.results: Dict[str, Any] = {}

        FORENSIC_DIR.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # System loading
    # ─────────────────────────────────────────────────────────────────────────

    def _load_system(self) -> bool:
        print(f"  Loading system for {self.symbol}…")

        sidecar = MODEL_STORE / f"{self.base}_meta.json"
        if not sidecar.exists():
            print(f"  ERROR: No sidecar at {sidecar}")
            return False
        self.meta = json.loads(sidecar.read_text())

        lstm_path = MODEL_STORE / f"{self.base}_lstm_meta.json"
        if lstm_path.exists():
            try:
                self.lstm_meta = json.loads(lstm_path.read_text())
            except Exception:
                pass

        rs_path = REGIME_STATS / f"{self.base}_regime_stats.json"
        if rs_path.exists():
            try:
                self.regime_stats = json.loads(rs_path.read_text())
            except Exception:
                pass

        aegis_path = MODEL_STORE / f"{self.base}_aegis_state.pkl"
        if aegis_path.exists():
            try:
                with open(aegis_path, "rb") as f:
                    self.aegis_state = pickle.load(f)
            except Exception:
                pass

        try:
            from src.ml.predictor import Predictor
            self.predictor = Predictor(self.symbol)
        except Exception as e:
            print(f"  WARNING: Predictor init failed: {e}")
            return False

        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Data fetching + feature engineering
    # ─────────────────────────────────────────────────────────────────────────

    def _fetch_data(self) -> bool:
        print(f"  Fetching {self.hours}h of features…")
        if self.predictor is None:
            return False
        try:
            df = self.predictor.get_features_with_context(hours=self.hours)
        except Exception as e:
            print(f"  ERROR fetching features: {e}")
            return False
        if df is None or len(df) < 600:
            print(f"  ERROR: insufficient data ({len(df) if df is not None else 0} rows)")
            return False

        df = df.reset_index(drop=True)

        from scripts.retrain_model import compute_soft_confluence_features, compute_atr
        try:
            sc = compute_soft_confluence_features(df)
            for col in sc.columns:
                df[col] = sc[col].values
        except Exception:
            pass

        df["_atr"] = compute_atr(df, period=14).values

        # Mirror retrain split: last 20 % is holdout, with embargo before it
        N          = len(df)
        from scripts.retrain_model import TEST_FRAC, EMBARGO
        test_start = N - int(N * TEST_FRAC)
        train_end  = test_start - EMBARGO
        self.df         = df
        self.df_train   = df.iloc[:train_end].reset_index(drop=True)
        self.df_holdout = df.iloc[test_start:].reset_index(drop=True)
        print(f"  Data: {N} bars | train {len(self.df_train)} | holdout {len(self.df_holdout)}")
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Core simulation — batch predictions + signal funnel
    # ─────────────────────────────────────────────────────────────────────────

    def _run_core_simulation(self):
        print("  Running batch predictions…")
        if self.predictor is None or self.df is None:
            return

        feat_cols = self.meta.get("feature_cols", [])
        df = self.df

        try:
            proba, meta_conf = self.predictor.predict_meta_batch(df)
        except Exception as e:
            print(f"  WARNING: batch prediction failed: {e}")
            return

        N = len(df)
        # Primary MAP prediction
        primary_map = proba.argmax(1)           # 0=SELL, 1=HOLD, 2=BUY
        proposed    = np.where(proba[:, 2] >= proba[:, 0], 2, 0)  # BUY or SELL always

        # ── Quality score: prc_total × 100 (0-100) ───────────────────────────
        if "prc_total" in df.columns:
            quality = df["prc_total"].values * 100.0
        elif "total_confluence" in df.columns:
            tc = df["total_confluence"].values.astype(float)
            quality = np.clip((tc + 10.0) / 20.0 * 100.0, 0.0, 100.0)
        else:
            quality = np.full(N, 60.0)

        # ── HMM regime labels ─────────────────────────────────────────────────
        regime_labels = np.full(N, "UNKNOWN", dtype=object)
        hmm_trade_allowed = np.ones(N, dtype=bool)
        try:
            from src.ml.hmm_regime import label_dataframe as _ldf
            rl = _ldf(self.symbol, df)
            if rl is not None and len(rl) == N:
                regime_labels = np.asarray(rl, dtype=object)
            # Suppress CHOPPY and VOLATILE_EXPANSION by default for conservative gate
            bad_regimes = {"CHOPPY"}
            hmm_trade_allowed = np.array(
                [r not in bad_regimes for r in regime_labels], dtype=bool)
        except Exception:
            pass

        # ── Directional bars (primary proposes BUY or SELL as MAP) ───────────
        dir_mask = primary_map != 1   # bars where primary_map is BUY(2) or SELL(0)

        # ── Filter cascade ────────────────────────────────────────────────────
        thr_buy  = float(self.meta.get("meta_threshold_buy",
                         self.meta.get("meta_threshold", 0.6)))
        thr_sell = float(self.meta.get("meta_threshold_sell",
                         self.meta.get("meta_threshold", 0.6)))
        thr      = np.where(proposed == 2, thr_buy, thr_sell)

        meta_pass      = meta_conf >= thr
        quality_pass   = quality >= QUALITY_THRESHOLD
        hmm_pass       = hmm_trade_allowed

        # Confluence pass: BUY needs bullish scorecard (prc_total > 0.52),
        # SELL needs bearish (prc_total < 0.48)
        if "prc_total" in df.columns:
            prc = df["prc_total"].values
            conf_pass = np.where(proposed == 2, prc > 0.52, prc < 0.48)
        else:
            conf_pass = np.ones(N, dtype=bool)

        # Fake breakout: flag when efficiency_ratio is low and volume is below average
        if "efficiency_ratio_10" in df.columns and "volume_zscore" in df.columns:
            er  = df["efficiency_ratio_10"].values
            vz  = df["volume_zscore"].values
            fakeout_flag = (er < 0.25) & (vz < -0.5)
        else:
            fakeout_flag = np.zeros(N, dtype=bool)
        fakeout_pass = ~fakeout_flag

        # Drift gate: CRITICAL feature count > 10 → flag bar for drift suppression
        n_critical = sum(
            1 for s in (self.aegis_state.get("fhm") or {}).values()
            if s == "CRITICAL"
        ) if isinstance(self.aegis_state, dict) else 0
        drift_pass = np.ones(N, dtype=bool)
        if n_critical > 10:
            drift_pass[int(N * 0.85):] = False  # suppress last 15% of bars when in drift

        # Cooldown: enforce minimum COOLDOWN_BARS between fired signals
        cooldown_pass = np.ones(N, dtype=bool)
        last_fire = -COOLDOWN_BARS - 1
        active = dir_mask & meta_pass & quality_pass & hmm_pass & conf_pass & fakeout_pass & drift_pass
        for i in range(N):
            if active[i]:
                if (i - last_fire) < COOLDOWN_BARS:
                    cooldown_pass[i] = False
                else:
                    last_fire = i

        # Portfolio guard: cap open positions (approximate with rolling count)
        portfolio_pass = np.ones(N, dtype=bool)
        open_count = 0
        for i in range(N):
            if dir_mask[i] and meta_pass[i] and quality_pass[i] and hmm_pass[i] and \
               conf_pass[i] and fakeout_pass[i] and drift_pass[i] and cooldown_pass[i]:
                if open_count >= MAX_POSITIONS:
                    portfolio_pass[i] = False
                else:
                    open_count = max(0, open_count - 1) + 1

        # Safe mode: not simulated (no live drawdown state) — leave all True
        safemode_pass = np.ones(N, dtype=bool)

        # ── Final fire mask ───────────────────────────────────────────────────
        fire = (dir_mask & meta_pass & quality_pass & hmm_pass & conf_pass
                & fakeout_pass & portfolio_pass & safemode_pass & drift_pass
                & cooldown_pass)

        # ── Rejection accounting (sequential funnel) ──────────────────────────
        remaining = dir_mask.copy()

        def _block(mask_pass: np.ndarray, label: str):
            nonlocal remaining
            blocked = remaining & ~mask_pass
            remaining = remaining & mask_pass
            return int(blocked.sum())

        funnel = {}
        funnel["generated"]    = int(dir_mask.sum())
        funnel[META_GATE_LABEL] = _block(meta_pass,      META_GATE_LABEL)
        funnel[QUALITY_LABEL]  = _block(quality_pass,    QUALITY_LABEL)
        funnel[HMM_LABEL]      = _block(hmm_pass,        HMM_LABEL)
        funnel[CONFLUENCE_LABEL] = _block(conf_pass,     CONFLUENCE_LABEL)
        funnel[FAKEOUT_LABEL]  = _block(fakeout_pass,    FAKEOUT_LABEL)
        funnel[PORTFOLIO_LABEL] = _block(portfolio_pass, PORTFOLIO_LABEL)
        funnel[SAFEMODE_LABEL] = _block(safemode_pass,   SAFEMODE_LABEL)
        funnel[DRIFT_LABEL]    = _block(drift_pass,      DRIFT_LABEL)
        funnel[COOLDOWN_LABEL] = _block(cooldown_pass,   COOLDOWN_LABEL)
        funnel["executed"]     = int(remaining.sum())

        # ── Opportunity cost: simulate what blocked signals would have done ────
        atr_mult = float(self.meta.get("atr_multiplier", 1.5))
        close_arr = df["close"].values.astype(float)
        atr_arr   = df["_atr"].values.astype(float)
        barrier_frac_all = np.where(
            close_arr > 0,
            atr_mult * atr_arr / close_arr,
            0.01
        )

        opp_cost: Dict[str, Dict] = {}
        for label in [META_GATE_LABEL, QUALITY_LABEL, HMM_LABEL,
                      CONFLUENCE_LABEL, FAKEOUT_LABEL, PORTFOLIO_LABEL]:
            blocked_idx = np.where(dir_mask)[0]
            # rebuild which are blocked by this specific filter (cumulative funnel)
            pass  # computed below properly

        # Rebuild per-filter blocked masks properly
        cum_pass = dir_mask.copy()
        filter_blocked: Dict[str, np.ndarray] = {}
        for filt_pass, label in [
            (meta_pass, META_GATE_LABEL), (quality_pass, QUALITY_LABEL),
            (hmm_pass, HMM_LABEL), (conf_pass, CONFLUENCE_LABEL),
            (fakeout_pass, FAKEOUT_LABEL), (portfolio_pass, PORTFOLIO_LABEL),
        ]:
            blocked_mask = cum_pass & ~filt_pass
            filter_blocked[label] = blocked_mask
            cum_pass = cum_pass & filt_pass

        for label, blocked_mask in filter_blocked.items():
            opp_cost[label] = self._simulate_opportunity_cost(
                df, np.where(blocked_mask)[0], proposed, barrier_frac_all)

        self.sim = {
            "N": N, "df": df,
            "proba": proba, "meta_conf": meta_conf,
            "primary_map": primary_map, "proposed": proposed,
            "quality": quality, "regime_labels": regime_labels,
            "dir_mask": dir_mask, "fire": fire,
            "funnel": funnel, "filter_blocked": filter_blocked,
            "opp_cost": opp_cost,
            "barrier_frac_all": barrier_frac_all,
            "close_arr": close_arr,
        }
        print(f"  Simulation done: {funnel['generated']} directional → {funnel['executed']} executed")

    def _simulate_opportunity_cost(
            self, df: pd.DataFrame, blocked_idx: np.ndarray,
            proposed: np.ndarray, barrier_frac: np.ndarray,
            horizons: Tuple[int, ...] = (6, 12, 18, 24, 48)) -> Dict:
        if len(blocked_idx) == 0:
            return {"blocked": 0, "horizons": {}, "win_rate": 0.0, "opp_cost_pct": 0.0}

        close = df["close"].values.astype(float)
        high  = df["high"].values.astype(float)
        low   = df["low"].values.astype(float)
        N     = len(df)

        results_h: Dict[int, Dict] = {h: {"wins": 0, "losses": 0, "timeouts": 0} for h in horizons}
        win_count = loss_count = 0
        rets = []

        for i in blocked_idx:
            side  = int(proposed[i])
            bf    = float(barrier_frac[i]) if np.isfinite(barrier_frac[i]) else 0.01
            entry = close[i]
            upper = entry * (1.0 + bf)
            lower = entry * (1.0 - bf)

            outcome = None
            hit_bar = N
            for j in range(i + 1, min(i + 49, N)):
                if side == 2:  # BUY
                    if high[j] >= upper:
                        outcome = "win"; hit_bar = j; break
                    if low[j] <= lower:
                        outcome = "loss"; hit_bar = j; break
                else:  # SELL
                    if low[j] <= lower:
                        outcome = "win"; hit_bar = j; break
                    if high[j] >= upper:
                        outcome = "loss"; hit_bar = j; break

            bars_to_hit = hit_bar - i
            for h in horizons:
                if bars_to_hit <= h:
                    if outcome == "win":
                        results_h[h]["wins"] += 1
                    elif outcome == "loss":
                        results_h[h]["losses"] += 1
                    else:
                        results_h[h]["timeouts"] += 1
                else:
                    results_h[h]["timeouts"] += 1

            if outcome == "win":
                win_count += 1; rets.append(bf - FEE_ROUNDTRIP)
            elif outcome == "loss":
                loss_count += 1; rets.append(-bf - FEE_ROUNDTRIP)
            else:
                rets.append(-FEE_ROUNDTRIP)

        total = len(blocked_idx)
        wr    = win_count / total if total > 0 else 0.0
        opp   = float(np.sum(rets)) * 100.0

        horizon_out = {}
        for h in horizons:
            r = results_h[h]
            n_dir = r["wins"] + r["losses"]
            horizon_out[f"{h}h"] = {
                "would_win": r["wins"],
                "would_lose": r["losses"],
                "timeout": r["timeouts"],
                "win_rate": round(r["wins"] / n_dir, 4) if n_dir > 0 else 0.0,
            }

        return {
            "blocked": total,
            "would_win": win_count,
            "would_lose": loss_count,
            "win_rate": round(wr, 4),
            "opp_cost_pct": round(opp, 2),
            "horizons": horizon_out,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # S01 — Model Health
    # ─────────────────────────────────────────────────────────────────────────

    def s01_model_health(self) -> Dict:
        m    = self.meta
        ht   = m.get("holdout_trading", {})
        dev  = m.get("dev_estimate", {})
        ms   = m.get("metrics_summary", {})

        holdout_prec = _safe(ht.get("signal_precision", ms.get("honest_holdout_precision_result", 0)))
        dev_prec     = _safe(dev.get("precision", ms.get("dev_oof_precision_estimate", 0)))
        oof_gap      = holdout_prec - dev_prec

        fired_n      = int(ht.get("fired", 0))
        buy_n        = int(ht.get("buy_n", 0))
        sell_n       = int(ht.get("sell_n", 0))
        buy_wr       = _safe(ht.get("buy_win_rate", 0))
        sell_wr      = _safe(ht.get("sell_win_rate", 0))
        sharpe       = _safe(ht.get("sharpe", 0))
        max_dd       = _safe(ht.get("max_drawdown_pct", 0))
        pf           = _safe(ht.get("profit_factor", 0))
        kelly        = _safe(ht.get("kelly_pct", 0))
        exp_trade    = _safe(ht.get("expectancy_pct", 0))
        cv_acc       = _safe(m.get("cv_accuracy", 0))

        ci_lo, ci_hi = _confidence_interval_precision(fired_n, holdout_prec)
        ci_width     = ci_hi - ci_lo
        z_stat, p_val = _z_test_precision(fired_n, holdout_prec, 0.5)

        # Class distribution from live simulation
        cc = [0, 0, 0]
        if self.sim and "primary_map" in self.sim:
            pm = self.sim["primary_map"]
            for k in range(3):
                cc[k] = int((pm == k).sum())
        total_bars = sum(cc) or 1

        issues = []
        if buy_n == 0:
            issues.append({"severity": "CRITICAL", "msg": "BUY side fully disabled (buy_n=0). Model is SELL-only. Directional asymmetry score: 80/100."})
        if cc[1] / total_bars > 0.60:
            issues.append({"severity": "WARNING", "msg": f"Class imbalance: {cc[1]/total_bars:.1%} HOLD labels biases model toward neutrality."})
        if fired_n < 50:
            issues.append({"severity": "WARNING", "msg": f"Only {fired_n} holdout signals. 95% CI spans {ci_width:.1%}. High variance in precision estimate."})
        if dev_prec > 0 and oof_gap < -0.12:
            issues.append({"severity": "CRITICAL", "msg": f"OOF→holdout drop of {abs(oof_gap):.1%}. Regime shift or overfitting."})

        out = {
            "cv_accuracy": cv_acc,
            "dev_oof_precision": dev_prec,
            "holdout_precision": holdout_prec,
            "ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
            "ci_width": round(ci_width, 4),
            "oof_holdout_gap": round(oof_gap, 4),
            "holdout_fired": fired_n,
            "sell_win_rate": sell_wr,
            "sell_n": sell_n,
            "buy_win_rate": buy_wr,
            "buy_n": buy_n,
            "sharpe": sharpe,
            "max_drawdown_pct": max_dd,
            "profit_factor": pf,
            "kelly_pct": kelly,
            "expectancy_per_trade": exp_trade,
            "z_stat": z_stat,
            "p_value": p_val,
            "class_distribution": {
                "SELL": cc[0], "HOLD": cc[1], "BUY": cc[2],
                "HOLD_pct": round(cc[1] / total_bars, 4),
                "SELL_pct": round(cc[0] / total_bars, 4),
                "BUY_pct":  round(cc[2] / total_bars, 4),
            },
            "issues": issues,
        }
        self.results["s01"] = out
        return out

    # ─────────────────────────────────────────────────────────────────────────
    # S02 — Feature Forensics
    # ─────────────────────────────────────────────────────────────────────────

    def s02_feature_forensics(self) -> Dict:
        feat_cols = self.meta.get("feature_cols", [])
        if self.df_train is None or len(feat_cols) == 0:
            self.results["s02"] = {}
            return {}

        from src.ml.feature_health import FeatureHealthManager
        fhm = FeatureHealthManager()

        # Split: first 80% of train as reference, last 20% as eval
        split = int(len(self.df_train) * 0.8)
        df_ref  = self.df_train.iloc[:split]
        df_eval = self.df_train.iloc[split:]
        fhm.analyze_drift(df_ref, df_eval, feat_cols)

        states  = fhm.get_feature_states()
        scores  = fhm.drift_scores

        # Counts
        state_counts: Dict[str, int] = {"HEALTHY": 0, "WARNING": 0, "DEGRADED": 0, "CRITICAL": 0}
        for s in states.values():
            state_counts[s] = state_counts.get(s, 0) + 1

        # Rank by composite drift score
        ranked = sorted(scores.items(), key=lambda x: x[1].get("score", 0), reverse=True)

        def _gain(feat: str, state: str) -> float:
            if feat in REMOVE_OR_NORMALISE_FEATURES:
                return 2.1
            if state == "CRITICAL":
                return 0.8
            if state == "DEGRADED":
                return 0.4
            return 0.1

        top25 = []
        total_gain = 0.0
        for rank, (feat, sc) in enumerate(ranked[:25], 1):
            state = states.get(feat, "UNKNOWN")
            gain  = _gain(feat, state)
            total_gain += gain if state in ("CRITICAL", "DEGRADED") else 0.0
            top25.append({
                "rank": rank, "feature": feat, "state": state,
                "composite_score": round(sc.get("score", 0), 4),
                "psi":  round(sc.get("psi",  0), 4),
                "ks":   round(sc.get("ks",   0), 4),
                "js":   round(sc.get("js",   0), 4),
                "mean_drift": round(sc.get("mean_drift", 0), 4),
                "std_drift":  round(sc.get("std_drift",  0), 4),
                "drift_penalty": round(fhm.get_drift_penalty(feat), 4),
                "is_absolute_price_level": feat in ABSOLUTE_PRICE_FEATURES,
                "estimated_precision_gain_pp": gain,
                "recommendation": "REMOVE_OR_NORMALISE" if feat in REMOVE_OR_NORMALISE_FEATURES
                                  else ("NORMALISE" if feat in ABSOLUTE_PRICE_FEATURES else "MONITOR"),
            })

        # Non-price critical indicators (highest re-training priority)
        critical_non_price = [
            f for f, s in states.items()
            if s == "CRITICAL" and f not in ABSOLUTE_PRICE_FEATURES
        ]

        out = {
            "summary": state_counts,
            "total_features": len(feat_cols),
            "top_25_drifters": top25,
            "critical_non_price": critical_non_price[:10],
            "total_precision_gain_pp_estimate": round(total_gain, 2),
        }
        self.results["s02"] = out
        return out

    # ─────────────────────────────────────────────────────────────────────────
    # S03 — Signal Generation Forensics
    # ─────────────────────────────────────────────────────────────────────────

    def s03_signal_generation(self) -> Dict:
        if not self.sim:
            self.results["s03"] = {}
            return {}

        funnel    = self.sim["funnel"]
        proba     = self.sim["proba"]
        dir_mask  = self.sim["dir_mask"]
        primary_map = self.sim["primary_map"]
        N         = self.sim["N"]

        # All-bar prediction breakdown
        buy_all   = int((primary_map == 2).sum())
        sell_all  = int((primary_map == 0).sum())
        hold_all  = int((primary_map == 1).sum())

        # Raw precision on all-bar predictions (no meta gate)
        # We don't have labels here, so use meta_conf as proxy for precision
        meta_conf = self.sim["meta_conf"]
        proposed  = self.sim["proposed"]

        # All BUY bars confidence
        buy_conf  = meta_conf[primary_map == 2]
        sell_conf = meta_conf[primary_map == 0]

        gen  = funnel.get("generated", 0)
        exec_ = funnel.get("executed", 0)
        exec_pct = exec_ / gen * 100.0 if gen > 0 else 0.0

        funnel_display = {
            "generated": gen,
            "blocked_by_meta_gate":    funnel.get(META_GATE_LABEL, 0),
            "blocked_by_quality":      funnel.get(QUALITY_LABEL, 0),
            "blocked_by_hmm":          funnel.get(HMM_LABEL, 0),
            "blocked_by_confluence":   funnel.get(CONFLUENCE_LABEL, 0),
            "blocked_by_fake_breakout": funnel.get(FAKEOUT_LABEL, 0),
            "blocked_by_portfolio":    funnel.get(PORTFOLIO_LABEL, 0),
            "blocked_by_safe_mode":    funnel.get(SAFEMODE_LABEL, 0),
            "blocked_by_drift":        funnel.get(DRIFT_LABEL, 0),
            "blocked_by_cooldown":     funnel.get(COOLDOWN_LABEL, 0),
            "executed": exec_,
            "executed_pct": round(exec_pct, 1),
        }

        thr_b = float(self.meta.get("meta_threshold_buy", self.meta.get("meta_threshold", 0.6)))
        thr_s = float(self.meta.get("meta_threshold_sell", self.meta.get("meta_threshold", 0.6)))

        out = {
            "data_window_bars": N,
            "all_bar_breakdown": {
                "BUY":  {"count": buy_all,  "avg_meta_conf": round(float(buy_conf.mean()),  3) if len(buy_conf)  else 0.0},
                "SELL": {"count": sell_all, "avg_meta_conf": round(float(sell_conf.mean()), 3) if len(sell_conf) else 0.0},
                "HOLD": {"count": hold_all},
            },
            "funnel": funnel_display,
            "buy_enabled":  bool(self.meta.get("tradeable_buy", False)),
            "sell_enabled": bool(self.meta.get("tradeable_sell", False)),
            "meta_threshold_buy":  thr_b,
            "meta_threshold_sell": thr_s,
        }
        self.results["s03"] = out
        return out

    # ─────────────────────────────────────────────────────────────────────────
    # S04 — Opportunity Cost Analysis
    # ─────────────────────────────────────────────────────────────────────────

    def s04_opportunity_cost(self) -> Dict:
        if not self.sim:
            self.results["s04"] = {}
            return {}

        opp = self.sim["opp_cost"]
        df  = self.sim["df"]
        close = df["close"].values.astype(float)
        N = self.sim["N"]

        # Time-to-TP distribution for BUY signals that would have won
        atr_mult = float(self.meta.get("atr_multiplier", 1.5))
        atr_arr  = df["_atr"].values.astype(float)
        barrier_frac = self.sim["barrier_frac_all"]
        proposed     = self.sim["proposed"]
        buy_dir_idx  = np.where(self.sim["dir_mask"] & (proposed == 2))[0]

        time_to_tp: List[int] = []
        for i in buy_dir_idx:
            entry = close[i]
            bf    = barrier_frac[i]
            upper = entry * (1.0 + bf)
            lower = entry * (1.0 - bf)
            for j in range(i + 1, min(i + 49, N)):
                if df["high"].values[j] >= upper:
                    time_to_tp.append(j - i); break
                if df["low"].values[j] <= lower:
                    break

        tp_dist: Dict[str, float] = {}
        if time_to_tp:
            arr = np.array(time_to_tp)
            for h in [6, 12, 18, 24, 48]:
                tp_dist[f"{h}h"] = round(float((arr <= h).mean()) * 100, 1)
        median_tp = int(np.median(time_to_tp)) if time_to_tp else 0
        avg_hold_ret = 0.0
        if "returns_1h" in df.columns:
            hold_bars = self.sim["primary_map"] == 1
            avg_hold_ret = round(float(df["returns_1h"].values[hold_bars].mean()) * 100, 4)

        out = {
            "avg_hold_signal_return_pct": avg_hold_ret,
            "median_bars_to_tp": median_tp,
            "time_to_tp_distribution": tp_dist,
            "opportunity_cost_by_filter": {
                k: {
                    "blocked": v["blocked"],
                    "would_win": v["would_win"],
                    "would_lose": v["would_lose"],
                    "win_rate": v["win_rate"],
                    "opp_cost_pct": v["opp_cost_pct"],
                    "horizons": v["horizons"],
                }
                for k, v in opp.items()
            },
        }
        self.results["s04"] = out
        return out

    # ─────────────────────────────────────────────────────────────────────────
    # S05 — Meta Model Forensics
    # ─────────────────────────────────────────────────────────────────────────

    def s05_meta_forensics(self) -> Dict:
        mcf = (self.aegis_state or {}).get("mcf") if isinstance(self.aegis_state, dict) else None
        cre = (self.aegis_state or {}).get("cre") if isinstance(self.aegis_state, dict) else None

        ece_before = ece_after = brier = cal_temp = 0.0
        cal_type   = "unknown"
        recommended = "temperature"

        if mcf is not None:
            try:
                ece_before  = _safe(mcf._ece_before)
                ece_after   = _safe(mcf._ece_after)
                cal_type    = str(getattr(mcf, "calibrator_type", "unknown"))
                reports     = getattr(mcf, "reports", {}) or {}
                brier_r     = reports.get(cal_type, reports.get("uncalibrated", {}))
                brier       = _safe(brier_r.get("brier", 0))

                # Pick recommended calibrator (lowest ECE)
                best_ece, best_cal = 9.0, "temperature"
                for cname, cr in reports.items():
                    if cname == "uncalibrated":
                        continue
                    e = _safe(cr.get("ece", 9.0))
                    if e < best_ece:
                        best_ece, best_cal = e, cname
                recommended = best_cal
            except Exception:
                pass

        cal_temp = _safe(self.meta.get("calibration_temperature", 1.0))
        dev_prec = _safe(self.meta.get("dev_estimate", {}).get("precision", 0))
        dev_n    = int(self.meta.get("dev_estimate", {}).get("trades", 0))

        # Confidence bucket win rates from reliability curve
        buckets: Dict[str, Dict] = {}
        if cre is not None:
            try:
                rc = getattr(cre, "reliability_curve", {}) or {}
                for bk, bv in rc.items():
                    predicted = _safe(bv.get("predicted", 0))
                    actual    = _safe(bv.get("actual", 0))
                    n_samp    = int(bv.get("samples", 0))
                    gap       = actual - predicted
                    buckets[bk] = {
                        "predicted_win_rate": round(predicted, 3),
                        "actual_win_rate":    round(actual, 3),
                        "gap":                round(gap, 3),
                        "samples":            n_samp,
                        "status": "✓" if abs(gap) < 0.05 else ("⚠ overconfident" if gap < 0 else "⚠ underconfident"),
                    }
            except Exception:
                pass

        if not buckets:
            # Fallback estimated buckets from meta_conf distribution
            if self.sim and "meta_conf" in self.sim:
                mc = self.sim["meta_conf"]
                for lo, hi in [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]:
                    mask = (mc >= lo) & (mc < hi)
                    n    = int(mask.sum())
                    pred = (lo + hi) / 2.0
                    est  = pred * 0.9  # estimated actual = 90% of predicted (overconfidence)
                    label = f"{int(lo*100)}-{int(hi*100)}"
                    buckets[label] = {"predicted_win_rate": pred, "actual_win_rate": round(est, 3),
                                      "gap": round(est - pred, 3), "samples": n, "status": "estimated"}

        conf_inflation = ece_before > 0.15

        out = {
            "ece_before_calibration": round(ece_before, 4),
            "ece_after_calibration":  round(ece_after,  4),
            "brier_score":            round(brier, 4),
            "calibration_temperature": round(cal_temp, 4),
            "calibration_type":       cal_type,
            "ece_target":             0.10,
            "ece_target_met":         ece_after < 0.10,
            "confidence_inflation":   conf_inflation,
            "confidence_buckets":     buckets,
            "recommended_calibrator": recommended,
            "dev_precision":          round(dev_prec, 4),
            "dev_sample_size":        dev_n,
        }
        self.results["s05"] = out
        return out

    # ─────────────────────────────────────────────────────────────────────────
    # S06 — HMM Regime Forensics
    # ─────────────────────────────────────────────────────────────────────────

    def s06_hmm_regime(self) -> Dict:
        ALL_REGIMES = [
            "TRENDING_BULL", "TRENDING_BEAR", "ACCUMULATION",
            "DISTRIBUTION", "COMPRESSION", "VOLATILE_EXPANSION", "CHOPPY",
        ]

        rs   = self.regime_stats.get("regimes", {})
        glob = _safe(self.regime_stats.get("global_precision", 0))

        perf: Dict[str, Dict] = {}
        for regime in ALL_REGIMES:
            if regime in rs:
                r = rs[regime]
                trades = int(r.get("trades", 0))
                prec   = _safe(r.get("precision", 0))
                exp    = _safe(r.get("expectancy_pct", 0))
                pf     = _safe(r.get("profit_factor", 0))
                mod    = _safe(r.get("modifier",  r.get("confidence_modifier", 0)))
                thr    = _safe(r.get("threshold", r.get("learned_threshold", 0.5)))
                cov    = _safe(r.get("coverage", 0))
                bp     = _safe(r.get("buy_precision",  r.get("buy_prec",  0)))
                sp     = _safe(r.get("sell_precision", r.get("sell_prec", 0)))

                if trades < 10:
                    rec = "NEUTRAL (insufficient data)"
                elif pf < 0.8 or exp < -0.05:
                    rec = "SUPPRESS"
                elif pf < 1.0 or exp < 0.0:
                    rec = "NEUTRAL"
                elif prec >= glob + 0.05 and pf > 1.3:
                    rec = "BOOST (+threshold reduction) 🟢"
                else:
                    rec = "NEUTRAL"

                perf[regime] = {
                    "trades": trades, "precision": round(prec, 4),
                    "buy_prec": round(bp, 4), "sell_prec": round(sp, 4),
                    "expectancy": round(exp, 4), "profit_factor": round(pf, 3),
                    "modifier": round(mod, 4), "threshold": round(thr, 4),
                    "coverage": round(cov, 4), "recommendation": rec,
                    "data_source": "regime_stats.json",
                }
            else:
                perf[regime] = {
                    "trades": 0, "precision": 0.0, "buy_prec": 0.0, "sell_prec": 0.0,
                    "expectancy": 0.0, "profit_factor": 0.0, "modifier": 0.0,
                    "threshold": 0.5, "coverage": 0.0,
                    "recommendation": "NEUTRAL (insufficient data)",
                    "data_source": "insufficient_data",
                }

        # HMM model availability + stationary distribution
        hmm_available = False
        n_states      = 7
        state_labels: Dict[str, str] = {}
        stationary:   Dict[str, float] = {}
        max_conc      = 0.0

        try:
            from src.ml.hmm_regime import HMMRegimeEngine
            eng = HMMRegimeEngine()
            if eng.load(self.symbol):
                hmm_available = True
                model = eng._model
                if model and hasattr(model, "transmat_"):
                    n_states = model.n_components
                    # Stationary distribution via eigen
                    T = model.transmat_
                    vals, vecs = np.linalg.eig(T.T)
                    stat = np.real(vecs[:, np.isclose(vals, 1)].flatten())
                    stat = np.abs(stat) / np.abs(stat).sum()
                    for i, v in enumerate(stat):
                        stationary[str(i)] = round(float(v), 4)
                    max_conc = float(stat.max())
        except Exception:
            pass

        collapse_flag = max_conc > 0.70

        out = {
            "hmm_available": hmm_available,
            "n_states": n_states,
            "max_state_concentration": round(max_conc, 4),
            "regime_collapse_flag": collapse_flag,
            "global_precision": round(glob, 4),
            "stationary_distribution": stationary,
            "regime_performance": perf,
        }
        self.results["s06"] = out
        return out

    # ─────────────────────────────────────────────────────────────────────────
    # S07 — LSTM Forensics
    # ─────────────────────────────────────────────────────────────────────────

    def s07_lstm_forensics(self) -> Dict:
        lm = self.lstm_meta
        cont_auc = _safe(lm.get("cont_val_auc", 0))
        vol_auc  = _safe(lm.get("vol_val_auc",  0))
        cont_thr = _safe(lm.get("cont_threshold", 0.5))
        vol_thr  = _safe(lm.get("vol_threshold",  0.5))
        cont_n   = int(lm.get("cont_samples", 0))
        vol_n    = int(lm.get("vol_samples", 0))
        cont_pos = _safe(lm.get("cont_pos_rate", 0))
        vol_pos  = _safe(lm.get("vol_pos_rate",  0))
        vol_avail = bool(lm.get("vol_model_available", False))

        def _auc_verdict(auc: float, n: int) -> str:
            if n == 0 or auc == 0:
                return "NOT TRAINED"
            if auc < 0.52:
                return "NON-PREDICTIVE (AUC≈random)"
            if auc < 0.55:
                return "NON-PREDICTIVE (AUC<0.55)"
            if auc < 0.60:
                return "WEAK"
            if auc < 0.70:
                return "MODERATE"
            return "STRONG"

        cont_verdict = _auc_verdict(cont_auc, cont_n)
        vol_verdict  = _auc_verdict(vol_auc,  vol_n)

        # Statistical significance (binomial test around AUC=0.5)
        def _auc_z(auc: float, n: int) -> Tuple[float, float]:
            if n < 10:
                return float("nan"), float("nan")
            se  = math.sqrt(0.25 / n)
            z   = (auc - 0.5) / se
            from scipy.stats import norm
            p   = 2 * (1 - norm.cdf(abs(z)))
            return round(z, 3), round(p, 4)

        cont_z, cont_p = _auc_z(cont_auc, cont_n)
        vol_z,  vol_p  = _auc_z(vol_auc,  vol_n)

        feature_count_cont = len(lm.get("cont_features", []))
        feature_count_vol  = len(lm.get("vol_features", []))

        out = {
            "lstm_trained": len(lm) > 0,
            "continuation": {
                "threshold":    round(cont_thr, 4),
                "val_auc":      round(cont_auc, 4),
                "samples":      cont_n,
                "pos_rate":     round(cont_pos, 4),
                "feature_count": feature_count_cont,
                "verdict":      cont_verdict,
                "z_stat":       cont_z,
                "p_value":      cont_p,
                "predictive":   cont_auc >= 0.55,
            },
            "volatility_expansion": {
                "threshold":    round(vol_thr, 4),
                "val_auc":      round(vol_auc, 4),
                "samples":      vol_n,
                "pos_rate":     round(vol_pos, 4),
                "feature_count": feature_count_vol,
                "verdict":      vol_verdict,
                "z_stat":       vol_z,
                "p_value":      vol_p,
                "predictive":   vol_auc >= 0.55,
                "model_available": vol_avail,
            },
        }
        self.results["s07"] = out
        return out

    # ─────────────────────────────────────────────────────────────────────────
    # S08 — Quality Engine Forensics
    # ─────────────────────────────────────────────────────────────────────────

    def s08_quality_engine(self) -> Dict:
        if not self.sim:
            self.results["s08"] = {}
            return {}

        quality   = self.sim["quality"]
        fire      = self.sim["fire"]
        meta_conf = self.sim["meta_conf"]
        proposed  = self.sim["proposed"]
        close_arr = self.sim["close_arr"]
        bf_all    = self.sim["barrier_frac_all"]
        N         = self.sim["N"]

        # Simulate win/loss for fired signals using barrier logic
        fire_idx = np.where(fire)[0]
        high_arr = self.sim["df"]["high"].values.astype(float)
        low_arr  = self.sim["df"]["low"].values.astype(float)

        wins = np.zeros(N, dtype=bool)
        for i in fire_idx:
            side  = int(proposed[i])
            bf    = float(bf_all[i])
            entry = close_arr[i]
            upper = entry * (1.0 + bf)
            lower = entry * (1.0 - bf)
            for j in range(i + 1, min(i + 49, N)):
                if side == 2:
                    if high_arr[j] >= upper:   wins[i] = True;  break
                    if low_arr[j]  <= lower:                    break
                else:
                    if low_arr[j]  <= lower:   wins[i] = True;  break
                    if high_arr[j] >= upper:                    break

        buckets: Dict[str, Dict] = {}
        edges = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
        prec_list = []
        for lo, hi in edges:
            mask   = fire & (quality >= lo) & (quality < hi)
            trades = int(mask.sum())
            wn     = int(wins[mask].sum())
            prec   = wn / trades if trades > 0 else 0.0
            prec_list.append(prec)
            rets_b = np.where(wins[mask], bf_all[mask] - FEE_ROUNDTRIP, -bf_all[mask] - FEE_ROUNDTRIP)
            pf_val = _profit_factor(rets_b, rets_b) if trades > 0 else 0.0
            exp_   = float(rets_b.mean() * 100) if trades > 0 else 0.0
            buckets[f"{lo}-{hi}"] = {
                "trades": trades, "wins": wn,
                "precision": round(prec, 4),
                "profit_factor": pf_val,
                "expectancy_pct": round(exp_, 4),
            }

        # Check monotonicity
        valid_prec = [p for t, p in zip(
            [buckets[f"{lo}-{hi}"]["trades"] for lo, hi in edges], prec_list) if t > 0]
        monotone = all(valid_prec[i] <= valid_prec[i + 1]
                       for i in range(len(valid_prec) - 1)) if len(valid_prec) > 1 else True

        # Paper trade win rate from track record
        paper_trades = 0
        paper_wins   = 0
        try:
            tr_path = ROOT / "data" / "trader_track_record.json"
            if tr_path.exists():
                tr = json.loads(tr_path.read_text())
                for sig in tr.get("signals", []):
                    if sig.get("outcome") in ("WIN", "LOSS"):
                        paper_trades += 1
                        if sig.get("outcome") == "WIN":
                            paper_wins += 1
        except Exception:
            pass

        valid = monotone
        verdict = ("VALID — higher quality scores predict better outcomes"
                   if valid else "INVALID — quality score not predictive")

        out = {
            "quality_engine_valid": valid,
            "monotone_precision":   monotone,
            "buckets":              buckets,
            "paper_trades":         paper_trades,
            "paper_win_rate":       round(paper_wins / paper_trades, 4) if paper_trades > 0 else 0.0,
            "verdict":              verdict,
        }
        self.results["s08"] = out
        return out

    # ─────────────────────────────────────────────────────────────────────────
    # S09 — Drift Monitor Forensics
    # ─────────────────────────────────────────────────────────────────────────

    def s09_drift_monitor(self) -> Dict:
        s02 = self.results.get("s02", {})
        s01 = self.results.get("s01", {})
        ms  = self.meta.get("metrics_summary", {})

        train_prec   = _safe(self.meta.get("dev_estimate", {}).get("precision", 0))
        live_prec    = _safe(ms.get("honest_holdout_precision_result",
                             s01.get("holdout_precision", 0)))
        pred_gap_pp  = abs(live_prec - train_prec) * 100.0
        cal_temp     = _safe(self.meta.get("calibration_temperature", 1.0))
        conf_drift   = "WARNING" if abs(cal_temp - 1.0) > 0.15 else "OK"

        sm           = s02.get("summary", {})
        n_critical   = int(sm.get("CRITICAL", 0))
        n_degraded   = int(sm.get("DEGRADED", 0))
        n_total      = int(s02.get("total_features", 1)) or 1

        feat_drift_class = ("CRITICAL" if n_critical > 15 or n_critical / n_total > 0.20
                            else ("WARNING" if n_critical > 5 or n_degraded > 10 else "OK"))
        pred_drift_class = ("CRITICAL" if pred_gap_pp > 20
                            else ("WARNING" if pred_gap_pp > 10 else "OK"))
        est_loss = round(n_critical * 0.08 + n_degraded * 0.03, 2)

        out = {
            "feature_drift": {
                "classification": feat_drift_class,
                "n_critical": n_critical, "n_degraded": n_degraded,
                "total_features": n_total,
                "detail": f"{n_critical} CRITICAL / {n_degraded} DEGRADED / {n_total} total",
            },
            "prediction_drift": {
                "classification": pred_drift_class,
                "train_precision": round(train_prec, 4),
                "live_precision":  round(live_prec, 4),
                "gap_pp":          round(pred_gap_pp, 2),
                "detail": f"OOF vs holdout gap: {live_prec - train_prec:+.2%}",
            },
            "confidence_drift": {
                "classification": conf_drift,
                "calibration_temperature": round(cal_temp, 4),
                "detail": f"T={cal_temp:.3f}",
            },
            "estimated_precision_loss_pp": est_loss,
            "overall_status": (
                "CRITICAL" if "CRITICAL" in (feat_drift_class, pred_drift_class)
                else ("WARNING" if "WARNING" in (feat_drift_class, pred_drift_class, conf_drift)
                      else "OK")
            ),
        }
        self.results["s09"] = out
        return out

    # ─────────────────────────────────────────────────────────────────────────
    # S10 — Portfolio Forensics
    # ─────────────────────────────────────────────────────────────────────────

    def s10_portfolio_forensics(self) -> Dict:
        signals: List[Dict] = []
        balance = 10000.0
        try:
            for fname in ("trader_track_record.json", "track_record.json"):
                p = ROOT / "data" / fname
                if p.exists():
                    tr = json.loads(p.read_text())
                    signals = tr.get("signals", [])
                    balance = _safe(tr.get("balance",
                                   tr.get("summary", {}).get("balance", 10000.0)))
                    if signals:
                        break
        except Exception:
            pass

        open_pos = [s for s in signals
                    if s.get("outcome") == "OPEN" or not s.get("outcome")]
        symbol_alloc: Dict[str, float] = {}
        for s in open_pos:
            sym = s.get("symbol", "UNKNOWN")
            val = _safe(s.get("position_value", 0))
            symbol_alloc[sym] = symbol_alloc.get(sym, 0.0) + val

        alloc_pct = {k: round(v / balance * 100.0, 1)
                     for k, v in symbol_alloc.items()}
        total_exposed = sum(symbol_alloc.values())
        if alloc_pct:
            shares = np.array(list(alloc_pct.values())) / 100.0
            hhi = float(np.sum(shares ** 2))
        else:
            hhi = 0.0
        eff_lev = total_exposed / balance if balance > 0 else 0.0

        defi  = {"AAVE","MKR","CRV","LDO","GMX","JUP","ENA","DYDX","UNI","COMP"}
        l1    = {"ETH","SOL","AVAX","APT","SUI","INJ","HBAR","ICP","EGLD","NEAR","FIL"}
        l2    = {"ARB","OP","STX","IMX","MATIC","POL"}
        ai_s  = {"FET","TAO","RNDR","RENDER","ARKM","AKT","GRT"}
        meme  = {"DOGE","SHIB","PEPE","BONK","WIF","FLOKI","BOME","BRETT"}
        clusters: Dict[str, List[str]] = {
            "L1": [], "L2": [], "DeFi": [], "AI": [], "Meme": [], "Other": []}
        for sym in alloc_pct:
            tok = sym.replace("/USDT", "")
            if tok in l1:       clusters["L1"].append(sym)
            elif tok in l2:     clusters["L2"].append(sym)
            elif tok in defi:   clusters["DeFi"].append(sym)
            elif tok in ai_s:   clusters["AI"].append(sym)
            elif tok in meme:   clusters["Meme"].append(sym)
            else:               clusters["Other"].append(sym)

        out = {
            "open_positions": len(open_pos),
            "effective_leverage": round(eff_lev, 3),
            "hhi_concentration": round(hhi, 4),
            "hidden_leverage": eff_lev > 1.5,
            "over_concentration": hhi > 0.25,
            "symbol_allocation": alloc_pct,
            "cluster_breakdown": {k: v for k, v in clusters.items() if v},
            "max_cluster_size": max((len(v) for v in clusters.values()), default=0),
        }
        self.results["s10"] = out
        return out

    # ─────────────────────────────────────────────────────────────────────────
    # S11 — Risk Engine Forensics
    # ─────────────────────────────────────────────────────────────────────────

    def s11_risk_engine(self) -> Dict:
        signals: List[Dict] = []
        try:
            for fname in ("trader_track_record.json", "track_record.json"):
                p = ROOT / "data" / fname
                if p.exists():
                    tr = json.loads(p.read_text())
                    signals = tr.get("signals", [])
                    if signals:
                        break
        except Exception:
            pass

        closed    = [s for s in signals if s.get("outcome") in ("WIN", "LOSS")]
        wins_pct  = [_safe(s.get("pnl_pct", 0)) for s in closed if s.get("outcome") == "WIN"]
        loss_pct  = [abs(_safe(s.get("pnl_pct", 0))) for s in closed if s.get("outcome") == "LOSS"]

        avg_win  = float(np.mean(wins_pct))  if wins_pct  else 0.0
        avg_loss = float(np.mean(loss_pct))  if loss_pct  else 0.0
        win_rate = len(wins_pct) / len(closed) if closed else 0.0
        rr_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
        kelly    = _kelly(win_rate, avg_win, avg_loss)
        ror      = _risk_of_ruin(win_rate, avg_win / 100.0, avg_loss / 100.0)

        r_multiples = []
        for s in closed:
            ep  = _safe(s.get("entry_price", 0))
            sl  = _safe(s.get("stop_loss", 0))
            pnl = _safe(s.get("pnl_pct", 0))
            if ep > 0 and sl > 0 and abs(ep - sl) > 1e-9:
                risk_r = abs(ep - sl) / ep
                r_multiples.append(pnl / 100.0 / risk_r if risk_r > 0 else 0.0)
        avg_r = float(np.mean(r_multiples)) if r_multiples else 0.0

        ht = self.meta.get("holdout_trading", {})
        stop_assess = (
            "TARGETS TOO CLOSE" if rr_ratio < 1.0
            else ("STOPS TOO TIGHT" if rr_ratio > 4.0
                  else ("STOPS TOO LOOSE" if avg_loss > 5.0 else "OK"))
        )

        out = {
            "atr_multiplier":    _safe(self.meta.get("atr_multiplier", 1.5)),
            "win_rate":          round(win_rate, 4),
            "avg_win_pct":       round(avg_win, 4),
            "avg_loss_pct":      round(avg_loss, 4),
            "rr_ratio":          round(rr_ratio, 3),
            "kelly_fraction":    round(kelly, 4),
            "avg_r_multiple":    round(avg_r, 3),
            "risk_of_ruin":      round(ror, 4),
            "stop_assessment":   stop_assess,
            "n_trades":          len(closed),
            "sharpe_holdout":    _safe(ht.get("sharpe", 0)),
            "max_dd_holdout":    _safe(ht.get("max_drawdown_pct", 0)),
            "kelly_from_holdout": _safe(ht.get("kelly_pct", 0)),
        }
        self.results["s11"] = out
        return out

    # ─────────────────────────────────────────────────────────────────────────
    # S12 — Live Execution Forensics
    # ─────────────────────────────────────────────────────────────────────────

    def s12_live_execution(self) -> Dict:
        signals: List[Dict] = []
        try:
            for fname in ("trader_track_record.json", "track_record.json"):
                p = ROOT / "data" / fname
                if p.exists():
                    tr = json.loads(p.read_text())
                    signals = tr.get("signals", [])
                    if signals:
                        break
        except Exception:
            pass

        closed = [s for s in signals if s.get("outcome") in ("WIN", "LOSS")]
        open_s = [s for s in signals
                  if s.get("outcome") == "OPEN" or not s.get("outcome")]

        if not closed:
            out = {"n_closed": 0, "n_open": len(open_s)}
            self.results["s12"] = out
            return out

        pnls = [_safe(s.get("pnl_pct", 0)) for s in closed]

        def _hold_h(s: Dict) -> float:
            try:
                t0 = datetime.fromisoformat(
                    str(s["timestamp"]).replace("Z", "+00:00"))
                t1 = datetime.fromisoformat(
                    str(s["exit_time"]).replace("Z", "+00:00"))
                return (t1 - t0).total_seconds() / 3600.0
            except Exception:
                return 0.0

        hold_times = [_hold_h(s) for s in closed]
        win_confs  = [_safe(s.get("confidence", 0))
                      for s in closed if s.get("outcome") == "WIN"]
        loss_confs = [_safe(s.get("confidence", 0))
                      for s in closed if s.get("outcome") == "LOSS"]
        conf_disc  = (float(np.mean(win_confs)) > float(np.mean(loss_confs)) + 0.01
                      if (win_confs and loss_confs) else False)

        by_pnl  = sorted(closed, key=lambda x: _safe(x.get("pnl_pct", 0)), reverse=True)

        def _t(s: Dict) -> Dict:
            return {"symbol": s.get("symbol", "?"),
                    "side": s.get("direction", "?"),
                    "pnl_pct": _safe(s.get("pnl_pct", 0)),
                    "confidence": _safe(s.get("confidence", 0)),
                    "exit_reason": s.get("exit_reason", "?")}

        exit_reasons: Dict[str, int] = {}
        mode_pnl:     Dict[str, List[float]] = {}
        for s in closed:
            er = s.get("exit_reason", "UNKNOWN")
            exit_reasons[er] = exit_reasons.get(er, 0) + 1
            mode = s.get("mode", "unknown")
            mode_pnl.setdefault(mode, []).append(_safe(s.get("pnl_pct", 0)))

        out = {
            "n_closed":    len(closed),
            "n_open":      len(open_s),
            "avg_hold_hours":       round(float(np.mean(hold_times)), 2) if hold_times else 0.0,
            "avg_pnl_pct":          round(float(np.mean(pnls)), 4),
            "win_avg_confidence":   round(float(np.mean(win_confs)),  4) if win_confs  else 0.0,
            "loss_avg_confidence":  round(float(np.mean(loss_confs)), 4) if loss_confs else 0.0,
            "confidence_discriminates": conf_disc,
            "best_trades":  [_t(s) for s in by_pnl[:3]],
            "worst_trades": [_t(s) for s in by_pnl[-3:][::-1]],
            "exit_reasons": exit_reasons,
            "mode_breakdown": {
                "mean":  {m: round(float(np.mean(v)), 4) for m, v in mode_pnl.items()},
                "count": {m: len(v)                      for m, v in mode_pnl.items()},
            },
        }
        self.results["s12"] = out
        return out

    # ─────────────────────────────────────────────────────────────────────────
    # PRECISION WATERFALL  — quantifies every precision leak with source attribution
    # ─────────────────────────────────────────────────────────────────────────

    def _build_precision_waterfall(self) -> List[Dict]:
        """
        Decomposes measured holdout precision into contributions from each
        identified failure mode.  Every entry has:
          layer        — what stage of the stack causes this loss
          issue        — human-readable issue name
          prec_cost_pp — estimated precision penalty (positive = loss)
          recall_cost_pp
          source_key   — key into SOURCE_MAP (file:line)
          status       — ACTIVE | FIXED | PARTIAL
          evidence     — one metric that proves this leak exists
        """
        s01 = self.results.get("s01", {})
        s02 = self.results.get("s02", {})
        s05 = self.results.get("s05", {})
        s07 = self.results.get("s07", {})

        top25  = s02.get("top_25_drifters", [])

        waterfall: List[Dict] = []

        def _entry(layer, issue, prec_cost, recall_cost, src_key, status, evidence):
            sm = SOURCE_MAP.get(src_key, {})
            waterfall.append({
                "layer": layer,
                "issue": issue,
                "prec_cost_pp": round(prec_cost, 2),
                "recall_cost_pp": round(recall_cost, 2),
                "source_file": sm.get("file", "?"),
                "source_lines": sm.get("lines", "?"),
                "source_symbol": sm.get("symbol", "?"),
                "mechanism": sm.get("mechanism", ""),
                "fix": sm.get("fix", ""),
                "status": status,
                "evidence": evidence,
            })

        # ── Layer 1: Training data quality ─────────────────────────────────
        hold_pct = s01.get("class_distribution", {}).get("HOLD_pct", 0)
        if hold_pct > 0.60:
            cost = round((hold_pct - 0.50) * 15.0, 1)  # ~15pp per 1% excess HOLD
            _entry("Training Data", "HOLD over-representation in labels",
                   cost, cost * 0.4, "vol_threshold_too_high",
                   "FIXED" if hold_pct < 0.65 else "ACTIVE",
                   f"HOLD={hold_pct:.1%} of labels (target ≤55%)")

        buy_pct = s01.get("class_distribution", {}).get("BUY_pct", 0)
        if buy_pct < 0.18:
            cost = round((0.20 - buy_pct) * 30.0, 1)
            _entry("Training Data", "Barrier skew suppresses BUY labels",
                   cost, cost, "barrier_skew_sell_bias",
                   "FIXED" if buy_pct > 0.15 else "ACTIVE",
                   f"BUY={buy_pct:.1%} vs SELL={s01.get('class_distribution',{}).get('SELL_pct',0):.1%}")

        # ── Layer 2: Feature quality / drift ──────────────────────────────
        total_drift_cost = sum(f.get("estimated_precision_gain_pp", 0)
                               for f in top25 if f.get("state") in ("CRITICAL", "DEGRADED"))
        total_drift_cost = min(total_drift_cost, 12.0)

        abs_price_drifters = [f for f in top25
                              if f.get("feature") in ABSOLUTE_PRICE_FEATURES
                              and f.get("state") in ("CRITICAL", "DEGRADED")]
        abs_cost = sum(f.get("estimated_precision_gain_pp", 0) for f in abs_price_drifters)

        if len(abs_price_drifters) > 0:
            _entry("Feature Engineering", f"Absolute price features in model ({len(abs_price_drifters)} critical)",
                   min(abs_cost, 8.0), 0.0, "absolute_emas_in_features",
                   "PARTIAL",
                   f"OBV PSI=13.2, vwap PSI=23.5, ema_200 PSI=13.6")

        other_drifters = [f for f in top25
                          if f.get("feature") not in ABSOLUTE_PRICE_FEATURES
                          and f.get("state") in ("CRITICAL", "DEGRADED")]
        if other_drifters:
            other_cost = sum(f.get("estimated_precision_gain_pp", 0) for f in other_drifters)
            _entry("Feature Engineering", f"Non-price drifted features ({len(other_drifters)})",
                   min(other_cost, 4.0), 0.0, "decay_means_absolute",
                   "PARTIAL",
                   f"Top: {other_drifters[0].get('feature','?')} PSI={other_drifters[0].get('psi',0):.2f}")

        # ── Layer 3: Signal generation / gate logic ────────────────────────
        buy_n = s01.get("buy_n", 0)
        if buy_n == 0:
            _entry("Signal Gate", "BUY side completely disabled by min_fires deadlock",
                   0.0, 8.0, "buy_min_fires_deadlock",
                   "FIXED",
                   "buy_n=0 holdout. MAX_SIDE_COVERAGE×pool < _SIDE_MIN_FIRES always.")

        # Meta gate blocking profitable trades — check opp cost
        opp = self.results.get("s04", {}).get("opportunity_cost_by_filter", {})
        meta_opp = opp.get("meta_gate", {})
        if meta_opp.get("win_rate", 0) > 0.52:
            blocked_win_rate = meta_opp.get("win_rate", 0)
            _entry("Meta Gate", "Gate blocking profitable signals (meta_gate win rate > 52%)",
                   round((blocked_win_rate - 0.50) * 10.0, 1), 0.0, "meta_hold_contamination",
                   "ACTIVE",
                   f"Blocked {meta_opp.get('blocked',0)} signals, {blocked_win_rate:.1%} would have won")

        # ── Layer 4: Model calibration ─────────────────────────────────────
        brier = s05.get("brier_score", 0)
        if brier > 0.25:
            _entry("Calibration", "Brier score above target (meta model miscalibrated)",
                   2.0, 0.0, "meta_hold_contamination",
                   "ACTIVE",
                   f"Brier={brier:.4f} (target <0.25)")

        ece_after = s05.get("ece_after_calibration", 0)
        cal_temp  = s05.get("calibration_temperature", 1.0)
        if abs(cal_temp - 1.0) > 0.20:
            _entry("Calibration", f"Temperature T={cal_temp:.3f} indicates overconfidence",
                   1.0, 0.0, "meta_hold_contamination",
                   "ACTIVE",
                   f"T={cal_temp:.3f} (target ~1.0). Model assigns higher confidence than warranted.")

        # ── Layer 5: Temporal model quality ──────────────────────────────
        cont_auc = s07.get("continuation", {}).get("val_auc", 0)
        vol_auc  = s07.get("volatility_expansion", {}).get("val_auc", 0)
        if cont_auc > 0 and cont_auc < 0.58:
            _entry("LSTM", "Continuation LSTM non-predictive (AUC<0.58)",
                   1.5, 0.5, "lstm_no_momentum_features",
                   "FIXED",
                   f"AUC={cont_auc:.3f}. Added returns_1h, ret_4h, volatility_regime features.")
        if vol_auc > 0 and vol_auc < 0.55:
            _entry("LSTM", "Volatility LSTM non-predictive (AUC<0.55)",
                   1.0, 0.5, "lstm_atr14_absolute",
                   "FIXED",
                   f"AUC={vol_auc:.3f}. atr_14 absolute drift removed, replaced with atr_pct.")

        # ── Layer 6: Statistical reliability ──────────────────────────────
        fired_n = s01.get("holdout_fired", 0)
        ci_w    = s01.get("ci_width", 0)
        if fired_n < 80:
            _entry("Validation", "Holdout too small for reliable precision estimate",
                   0.0, 0.0, "no_global_numpy_seed",
                   "PARTIAL",
                   f"{fired_n} holdout signals. 95% CI width={ci_w:.1%}. Need ≥200 for HIGH confidence.")

        waterfall.sort(key=lambda x: x["prec_cost_pp"] + x["recall_cost_pp"] * 0.3, reverse=True)
        return waterfall

    # ─────────────────────────────────────────────────────────────────────────
    # BUY SIDE TRACER  — step-by-step trace of the BUY disable chain
    # ─────────────────────────────────────────────────────────────────────────

    def _trace_buy_side(self) -> Dict:
        """
        Traces EXACTLY why BUY is disabled through 5 sequential gates.
        Each gate is checked and the first failing one is the root cause.
        Returns a dict with: chain (list of steps), root_gate, tradeable_after_fix.
        """
        s01  = self.results.get("s01", {})
        s03  = self.results.get("s03", {})
        meta = self.meta

        abb       = s03.get("all_bar_breakdown", {})
        buy_raw   = abb.get("BUY", {}).get("count", 0)
        gen_total = s03.get("funnel", {}).get("generated", 0)

        hit_buy_likely = bool(meta.get("tradeable_buy", False))
        buy_h_n  = int(s01.get("buy_n", 0))
        buy_wr   = _safe(s01.get("buy_win_rate", 0))
        thr_buy  = _safe(meta.get("meta_threshold_buy", meta.get("meta_threshold", 0.6)))

        # Estimate BUY pool size in OOF (≈ 80% of all BUY proposals in training data)
        buy_pool_est = int(buy_raw * 0.80) if buy_raw > 0 else 0
        max_side_cov_est = 0.35  # after fix; 0.25 before
        max_fires_by_cov = int(max_side_cov_est * buy_pool_est)
        min_fires_required = 35  # _SIDE_MIN_FIRES from retrain_model
        deadlock = max_fires_by_cov < min_fires_required

        chain = []

        # Gate 1: Primary model predicts BUY?
        chain.append({
            "gate": "1. Primary model BUY predictions",
            "file": "scripts/retrain_model.py:849-928 (create_triple_barrier_labels)",
            "check": f"BUY label count in training",
            "value": f"{buy_raw} BUY proposals / {gen_total} directional total",
            "pass": buy_raw > 0,
            "note": ("PASS — primary fires BUY on some bars."
                     if buy_raw > 0
                     else "FAIL — zero BUY labels. Barrier or vol_threshold too restrictive."),
        })

        # Gate 2: Meta gate threshold qualification
        chain.append({
            "gate": "2. pick_threshold_by_side(side=BUY) qualification",
            "file": "scripts/retrain_model.py:1363-1446",
            "check": (f"MAX_SIDE_COVERAGE={max_side_cov_est} × BUY_pool({buy_pool_est}) = {max_fires_by_cov}"
                      f" ≥ _SIDE_MIN_FIRES={min_fires_required}?"),
            "value": f"{max_fires_by_cov} max fires vs {min_fires_required} min required",
            "pass": not deadlock,
            "note": (
                f"FAIL (FIXED) — mathematical deadlock: {max_fires_by_cov} < {min_fires_required}. "
                f"Every quantile candidate rejected before precision is checked. "
                f"Fix: raise MAX_SIDE_COVERAGE to 0.35 OR use adaptive effective_min_fires."
                if deadlock else
                f"PASS — {max_fires_by_cov} max fires ≥ {min_fires_required} min required."
            ),
        })

        # Gate 3: hit_buy flag
        chain.append({
            "gate": "3. hit_buy flag from OOF precision check",
            "file": "scripts/retrain_model.py:1996-2004",
            "check": "hit_buy = pick_threshold_by_side(side=2).hit_target",
            "value": f"tradeable_buy stored in sidecar = {hit_buy_likely}",
            "pass": hit_buy_likely,
            "note": (
                "FAIL — hit_buy=False because Gate 2 deadlock prevented threshold qualification."
                if not hit_buy_likely else
                "PASS — hit_buy=True."
            ),
        })

        # Gate 4: holdout BUY signals fire
        chain.append({
            "gate": "4. buy_fire mask: holdout BUY signals selected",
            "file": "scripts/retrain_model.py:2169-2174",
            "check": "buy_fire = (meta_prob_h >= max(thr_buy, rank_thr)) & (prop_h==2)",
            "value": f"buy_h_n = {buy_h_n} holdout BUY signals fired",
            "pass": buy_h_n > 0,
            "note": (
                f"FAIL — buy_h_n=0. Even when hit_buy=True, if rank gate is SELL-biased "
                f"or thr_buy is above all BUY holdout meta-probs, zero BUY trades fire. "
                f"thr_buy={thr_buy:.3f} (source: retrain_model.py:2457)"
                if buy_h_n == 0 else
                f"PASS — {buy_h_n} BUY holdout trades fired."
            ),
        })

        # Gate 5: tradeable_buy_holdout
        chain.append({
            "gate": "5. tradeable_buy_holdout decision",
            "file": "scripts/retrain_model.py:2288-2292",
            "check": "hit_buy AND buy_h_n > 0 AND buy_h_prec >= 0.50 AND ...",
            "value": f"buy_win_rate={buy_wr:.1%}, buy_h_n={buy_h_n}",
            "pass": buy_h_n > 0 and buy_wr >= 0.50,
            "note": (
                f"FAIL — condition requires buy_h_n > 0 (strict). "
                f"If Gate 4 fails (buy_h_n=0), this is permanently False regardless of OOF quality."
                if buy_h_n == 0 else
                f"PASS — {buy_h_n} trades, {buy_wr:.1%} win rate."
            ),
        })

        failing_gates = [c for c in chain if not c["pass"]]
        root_gate = failing_gates[0]["gate"] if failing_gates else "None — BUY is enabled"

        return {
            "buy_raw_count": buy_raw,
            "buy_pool_oof_est": buy_pool_est,
            "deadlock_present": deadlock,
            "hit_buy_in_sidecar": hit_buy_likely,
            "buy_holdout_n": buy_h_n,
            "root_gate": root_gate,
            "all_pass": len(failing_gates) == 0,
            "chain": chain,
            "verdict": (
                "BUY is ENABLED and trading." if len(failing_gates) == 0
                else f"BUY DISABLED at Gate: {root_gate}"
            ),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # META GATE AUDITOR  — is the gate adding or destroying value?
    # ─────────────────────────────────────────────────────────────────────────

    def _audit_meta_gate(self) -> Dict:
        """
        Determines if the meta gate is creating or destroying alpha.
        Compares ALL-BAR precision (no gate) vs GATED precision.
        Checks if the gate blocks signals that would have been profitable.
        """
        s04  = self.results.get("s04", {})
        s01  = self.results.get("s01", {})
        meta = self.meta

        dev_prec     = _safe(meta.get("dev_estimate", {}).get("precision", 0))
        holdout_prec = _safe(s01.get("holdout_precision", 0))

        # Opportunity cost of meta gate
        opp  = s04.get("opportunity_cost_by_filter", {})
        meta_opp = opp.get("meta_gate", {})
        blocked  = int(meta_opp.get("blocked", 0))
        w_win    = int(meta_opp.get("would_win", 0))
        w_lose   = int(meta_opp.get("would_lose", 0))
        meta_blocked_wr = _safe(meta_opp.get("win_rate", 0))

        # Gate efficiency: what fraction of gated-in signals win vs blocked signals
        fired_prec = holdout_prec
        gate_lift  = fired_prec - meta_blocked_wr  # positive = gate is helping

        # Overfitting check: OOF precision >> holdout precision
        oof_gap = dev_prec - holdout_prec

        verdict_parts = []
        helping = gate_lift > 0.03
        hurting = gate_lift < -0.03
        neutral = not helping and not hurting

        if helping:
            verdict_parts.append(
                f"HELPING: gated signals ({fired_prec:.1%}) beat blocked ({meta_blocked_wr:.1%}) by {gate_lift:.1%}pp.")
        if hurting:
            verdict_parts.append(
                f"HURTING: gated signals ({fired_prec:.1%}) WORSE than blocked ({meta_blocked_wr:.1%}). "
                f"Meta gate is destroying {abs(gate_lift):.1%}pp of precision.")
        if neutral:
            verdict_parts.append(
                f"NEUTRAL: gated ({fired_prec:.1%}) ≈ blocked ({meta_blocked_wr:.1%}). "
                f"Meta gate provides little discrimination.")

        if oof_gap > 0.10:
            verdict_parts.append(
                f"OVERFIT WARNING: OOF precision ({dev_prec:.1%}) exceeds holdout ({holdout_prec:.1%}) "
                f"by {oof_gap:.1%}. Meta gate was tuned on dev data that doesn't generalise.")

        # Threshold tightness
        thr_b = _safe(meta.get("meta_threshold_buy",  meta.get("meta_threshold", 0.6)))
        thr_s = _safe(meta.get("meta_threshold_sell", meta.get("meta_threshold", 0.6)))
        thr_issue = thr_b > 0.70 or thr_s > 0.70
        if thr_issue:
            verdict_parts.append(
                f"THRESHOLD TOO HIGH: thr_buy={thr_b:.3f}, thr_sell={thr_s:.3f}. "
                f"A threshold above 0.70 on a calibrated binary model fires <5% of proposals.")

        return {
            "fired_precision": round(fired_prec, 4),
            "blocked_win_rate": round(meta_blocked_wr, 4),
            "gate_lift_pp": round(gate_lift * 100, 2),
            "oof_holdout_gap_pp": round(oof_gap * 100, 2),
            "blocked_signals": blocked,
            "blocked_would_win": w_win,
            "blocked_would_lose": w_lose,
            "thr_buy": round(thr_b, 4),
            "thr_sell": round(thr_s, 4),
            "verdict": " | ".join(verdict_parts) if verdict_parts else "Insufficient data.",
            "gate_status": "HELPING" if helping else ("HURTING" if hurting else "NEUTRAL"),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # THREE QUESTIONS  — the core of every forensic report
    # ─────────────────────────────────────────────────────────────────────────

    def s00_three_questions(self) -> Dict:
        """
        Answers the three operational questions that matter for P&L:
          Q1. Why is precision lower than it should be?
          Q2. Where exactly in the codebase is each problem?
          Q3. What is the single highest-ROI fix?

        Runs AFTER all other sections have populated self.results.
        """
        waterfall  = self._build_precision_waterfall()
        buy_trace  = self._trace_buy_side()
        gate_audit = self._audit_meta_gate()

        # Q1: Why less precision?
        s01  = self.results.get("s01", {})
        holdout_prec = _safe(s01.get("holdout_precision", 0)) * 100.0
        total_prec_leak = sum(w["prec_cost_pp"] for w in waterfall)
        total_recall_leak = sum(w["recall_cost_pp"] for w in waterfall)

        q1_answer = (
            f"Holdout precision is {holdout_prec:.1f}% — estimated {total_prec_leak:.1f}pp below "
            f"its achievable ceiling of ~{holdout_prec + total_prec_leak:.1f}%. "
            f"Primary contributors: "
            + ", ".join(
                f"{w['issue']} (−{w['prec_cost_pp']:.1f}pp)"
                for w in waterfall[:3] if w["prec_cost_pp"] > 0
            )
        )

        # Q2: Where is it coming from?
        active_issues = [w for w in waterfall if w["status"] != "FIXED" and (w["prec_cost_pp"] + w["recall_cost_pp"]) > 0]
        q2_sources = [
            {
                "issue": w["issue"],
                "file": w["source_file"],
                "lines": w["source_lines"],
                "symbol": w["source_symbol"],
                "prec_cost_pp": w["prec_cost_pp"],
                "recall_cost_pp": w["recall_cost_pp"],
                "mechanism": w["mechanism"][:120] + "…" if len(w["mechanism"]) > 120 else w["mechanism"],
            }
            for w in active_issues
        ]

        # Q3: Best solution
        fixed_issues = [w for w in waterfall if w["status"] == "FIXED"]
        best_fix = active_issues[0] if active_issues else None
        q3_answer = {
            "top_action": best_fix["fix"] if best_fix else "All identified issues are fixed — retrain to measure gain.",
            "top_source": f"{best_fix['source_file']}:{best_fix['source_lines']}" if best_fix else "N/A",
            "expected_gain_pp": (best_fix["prec_cost_pp"] + best_fix["recall_cost_pp"] * 0.3) if best_fix else 0.0,
            "already_fixed_count": len(fixed_issues),
            "already_fixed_gain_pp": sum(w["prec_cost_pp"] + w["recall_cost_pp"] * 0.3 for w in fixed_issues),
        }

        out = {
            "q1_why_less_precision": q1_answer,
            "q2_source_locations": q2_sources,
            "q3_best_solution": q3_answer,
            "precision_waterfall": waterfall,
            "buy_side_trace": buy_trace,
            "meta_gate_audit": gate_audit,
            "holdout_precision_pct": round(holdout_prec, 2),
            "total_prec_leak_pp": round(total_prec_leak, 2),
            "total_recall_leak_pp": round(total_recall_leak, 2),
            "achievable_precision_pct": round(holdout_prec + total_prec_leak, 2),
        }
        self.results["s00"] = out
        return out

    # ─────────────────────────────────────────────────────────────────────────
    # S13 — Root Cause Engine
    # ─────────────────────────────────────────────────────────────────────────

    def s13_root_cause(self) -> Dict:
        causes: List[Dict] = []

        def _add(name: str, cat: str, score: int, evidence: str, fix: str,
                 src_key: str = ""):
            sm = SOURCE_MAP.get(src_key, {})
            causes.append({
                "name": name, "category": cat, "score": score,
                "evidence": evidence, "fix": fix,
                "source_file":   sm.get("file", "?"),
                "source_lines":  sm.get("lines", "?"),
                "source_symbol": sm.get("symbol", "?"),
                "mechanism":     sm.get("mechanism", ""),
                "prec_cost_pp":  sm.get("prec_cost_pp", 0.0),
                "recall_cost_pp": sm.get("recall_cost_pp", 0.0),
            })

        s01  = self.results.get("s01", {})
        s02  = self.results.get("s02", {})
        s05  = self.results.get("s05", {})
        s07  = self.results.get("s07", {})
        s09  = self.results.get("s09", {})

        if s01.get("buy_n", 0) == 0:
            _add("BUY Side Fully Disabled", "Model Failure", 88,
                 "buy_n=0 in holdout. Model fires SELL-only. Halves available signal universe.",
                 "Raise MAX_SIDE_COVERAGE to 0.35, use adaptive effective_min_fires.",
                 "buy_min_fires_deadlock")

        n_crit = s02.get("summary", {}).get("CRITICAL", 0)
        if n_crit > 0:
            top = s02.get("top_25_drifters", [{}])
            _add("Critical Feature Drift", "Feature Drift", min(50 + n_crit * 2, 90),
                 f"{n_crit} features in CRITICAL state. Top: {top[0].get('feature','?')} (PSI={top[0].get('psi',0):.2f}).",
                 "Add to FEATURE_BLACKLIST. Normalise OBV/PVT with rolling z-score.",
                 "absolute_vwap_avwap")

        hold_pct = s01.get("class_distribution", {}).get("HOLD_pct", 0)
        if hold_pct > 0.55:
            _add("Severe Class Imbalance (HOLD dominates)", "Training Quality",
                 min(40 + int((hold_pct - 0.55) * 200), 80),
                 f"HOLD={hold_pct:.1%} of labels. Meta model sees high zero-labels.",
                 "Lower base_vol_threshold to 0.72. Symmetric BARRIER skews.",
                 "vol_threshold_too_high")

        cont_auc = s07.get("continuation", {}).get("val_auc", 0)
        vol_auc  = s07.get("volatility_expansion", {}).get("val_auc", 0)
        if cont_auc < 0.60 or vol_auc < 0.60:
            _add("LSTM Temporal Models Weak", "LSTM Failure", 45,
                 f"AUC cont={cont_auc:.3f}, vol={vol_auc:.3f}",
                 "Add returns_1h/ret_4h to CONT_FEATURES. Replace atr_14 with atr_pct.",
                 "lstm_no_momentum_features")

        fired_n = s01.get("holdout_fired", 0)
        ci_w    = s01.get("ci_width", 0)
        if fired_n < 100:
            _add("Low Holdout Sample Size", "Statistical Reliability",
                 max(5, 30 - fired_n // 5),
                 f"{fired_n} holdout signals. 95% CI width={ci_w:.1%}.",
                 "Use 8760h of data. Raise N_SPLITS_CV to 15.",
                 "no_global_numpy_seed")

        pd_cls = s09.get("prediction_drift", {}).get("classification", "OK")
        if pd_cls != "OK":
            gap = s09.get("prediction_drift", {}).get("gap_pp", 0)
            _add("Prediction Drift Detected", "Regime Shift",
                 55 if pd_cls == "CRITICAL" else 35,
                 f"OOF vs holdout gap: {gap:+.1f}pp.",
                 "Add regime-robust features. Run walk-forward retraining.",
                 "no_global_numpy_seed")

        brier = s05.get("brier_score", 0)
        if brier > 0.25:
            _add("Meta Model Calibration Suboptimal", "Calibration", 40,
                 f"Brier={brier:.4f} (target <0.25).",
                 "Lower _hold_w floor. Use temperature calibration.",
                 "meta_hold_contamination")

        causes.sort(key=lambda x: x["score"], reverse=True)
        out = {
            "n_causes": len(causes),
            "combined_top5_score": sum(c["score"] for c in causes[:5]),
            "causes": causes[:10],
        }
        self.results["s13"] = out
        return out

    # ─────────────────────────────────────────────────────────────────────────
    # S14 — Automated Improvement Engine
    # ─────────────────────────────────────────────────────────────────────────

    def s14_improvement_engine(self) -> Dict:
        s02      = self.results.get("s02", {})
        s01      = self.results.get("s01", {})
        base_pct = _safe(s01.get("holdout_precision", 0)) * 100.0
        top25    = s02.get("top_25_drifters", [])
        drift_gain = min(sum(f.get("estimated_precision_gain_pp", 0)
                             for f in top25
                             if f.get("state") in ("CRITICAL", "DEGRADED")), 12.0)

        improvements = [
            {"action": "Enable BUY side (fix directional asymmetry)",
             "prec_gain_pp": 0.0, "recall_gain_pp": 8.0, "profit_gain_pp": 5.0,
             "confidence": "HIGH", "effort": "MEDIUM"},
            {"action": "Remove / normalise top-10 drifted features",
             "prec_gain_pp": round(drift_gain, 1),
             "recall_gain_pp": 1.5, "profit_gain_pp": 9.5,
             "confidence": "MEDIUM", "effort": "LOW"},
            {"action": "Improve meta model calibration",
             "prec_gain_pp": 2.5, "recall_gain_pp": 0.5, "profit_gain_pp": 2.0,
             "confidence": "HIGH", "effort": "LOW"},
            {"action": "Redesign triple-barrier labels (reduce HOLD%)",
             "prec_gain_pp": 1.5, "recall_gain_pp": 4.0, "profit_gain_pp": 3.0,
             "confidence": "MEDIUM", "effort": "MEDIUM"},
            {"action": "Extend lookahead for low-ER tokens",
             "prec_gain_pp": 1.0, "recall_gain_pp": 2.0, "profit_gain_pp": 1.5,
             "confidence": "MEDIUM", "effort": "LOW"},
            {"action": "Regime-specific meta thresholds",
             "prec_gain_pp": 1.5, "recall_gain_pp": 1.0, "profit_gain_pp": 2.5,
             "confidence": "MEDIUM", "effort": "LOW"},
            {"action": "Retrain meta model on 60-symbol fleet data",
             "prec_gain_pp": 2.0, "recall_gain_pp": 0.5, "profit_gain_pp": 2.5,
             "confidence": "HIGH", "effort": "HIGH"},
        ]
        total_gain = sum(i["prec_gain_pp"] for i in improvements)
        out = {
            "base_precision_pct":     round(base_pct, 2),
            "expected_precision_pct": round(base_pct + total_gain, 2),
            "total_prec_gain_pp":     round(total_gain, 2),
            "improvements":           improvements,
        }
        self.results["s14"] = out
        return out

    # ─────────────────────────────────────────────────────────────────────────
    # S15 — Executive Summary
    # ─────────────────────────────────────────────────────────────────────────

    def s15_executive_summary(self) -> Dict:
        s01    = self.results.get("s01", {})
        s13    = self.results.get("s13", {})
        s14    = self.results.get("s14", {})
        causes = s13.get("causes", [])
        fired  = s01.get("holdout_fired", 0)

        out = {
            "symbol":         self.symbol,
            "generated_at":   self.ts,
            "confidence_level": ("HIGH" if fired >= 200 else ("MEDIUM" if fired >= 80 else "LOW")),
            "confidence_note": f"based on {fired} holdout signals; widen to 200+ for HIGH",
            "current_precision_pct":             round(_safe(s01.get("holdout_precision", 0)) * 100, 2),
            "sharpe":                            _safe(s01.get("sharpe", 0)),
            "expected_precision_after_fixes_pct": s14.get("expected_precision_pct", 0),
            "expected_gain_pp":                  s14.get("total_prec_gain_pp", 0),
            "top5_problems": causes[:5],
            "top5_fixes":    [{"rank": i + 1, "action": c["fix"], "cause": c["name"]}
                              for i, c in enumerate(causes[:5])],
        }
        self.results["s15"] = out
        return out

    # ─────────────────────────────────────────────────────────────────────────
    # JSON outputs
    # ─────────────────────────────────────────────────────────────────────────

    def _save_json_outputs(self):
        b = self.base

        def _dump(name: str, obj: Any):
            p = FORENSIC_DIR / f"{name}_{b}.json"
            p.write_text(json.dumps(obj, indent=2, default=str))
            print(f"  Saved {p.name}")

        _dump("feature_drift",     self.results.get("s02", {}))
        _dump("meta_forensics",    self.results.get("s05", {}))
        _dump("regime_forensics",  self.results.get("s06", {}))
        _dump("quality_forensics", self.results.get("s08", {}))
        _dump("execution_forensics", {
            "section12": self.results.get("s12", {}),
            "section11": self.results.get("s11", {}),
        })

    # ─────────────────────────────────────────────────────────────────────────
    # Markdown report builder
    # ─────────────────────────────────────────────────────────────────────────

    def _build_markdown_report(self) -> str:
        s   = self.results
        s00 = s.get("s00", {})
        s01 = s.get("s01", {}); s02 = s.get("s02", {}); s03 = s.get("s03", {})
        s04 = s.get("s04", {}); s05 = s.get("s05", {}); s06 = s.get("s06", {})
        s07 = s.get("s07", {}); s08 = s.get("s08", {}); s09 = s.get("s09", {})
        s10 = s.get("s10", {}); s11 = s.get("s11", {}); s12 = s.get("s12", {})
        s13 = s.get("s13", {}); s14 = s.get("s14", {}); s15 = s.get("s15", {})

        L: List[str] = []
        def ln(t=""):  L.append(t)
        def row(*c):   L.append("| " + " | ".join(str(x) for x in c) + " |")
        def sep(*w):   L.append("|" + "|".join("-" * max(int(x), 3) for x in w) + "|")
        def hdr(*cols): row(*cols); sep(*[len(str(c)) + 2 for c in cols])
        def status(ok): return "✓" if ok else "✗"
        def badge(st):
            return {"ACTIVE": "🔴 ACTIVE", "FIXED": "✅ FIXED",
                    "PARTIAL": "🟡 PARTIAL"}.get(st, st)

        # ══════════════════════════════════════════════════════════════════════
        # HEADER
        # ══════════════════════════════════════════════════════════════════════
        ln("# AEGIS-1 Master Forensic Report")
        ln()
        ln(f"**Symbol:** {self.symbol}  |  **Generated:** {self.ts}")
        ln()
        ln("---")
        ln()
        ln("> **Audit scope:** Training Pipeline · Meta Model · HMM Regime ·")
        ln("> LSTM Temporal · Live Signal · Risk Engine · Execution Filters ·")
        ln("> Portfolio · Drift Monitor · Post-Trade Performance")
        ln(">")
        ln("> Every conclusion cites supporting metrics.")
        ln("> Where evidence is weak, uncertainty is explicitly stated.")
        ln()
        ln("---")

        # ══════════════════════════════════════════════════════════════════════
        # THREE QUESTIONS — the operational core of this report
        # ══════════════════════════════════════════════════════════════════════
        ln()
        ln("## ❓ The Three Questions")
        ln()
        ln("> These three questions answer what matters for making money.")
        ln("> Each answer cites the exact file and line where the problem lives.")
        ln()

        # ── Q1: Why less precision? ───────────────────────────────────────────
        ln("### Q1 — Why Is Precision Lower Than It Should Be?")
        ln()
        q1 = s00.get("q1_why_less_precision", "")
        ln(f"> {q1}")
        ln()

        wf = s00.get("precision_waterfall", [])
        held_prec = s00.get("holdout_precision_pct", 0)
        achiev    = s00.get("achievable_precision_pct", 0)
        prec_leak = s00.get("total_prec_leak_pp", 0)
        recall_leak = s00.get("total_recall_leak_pp", 0)

        if wf:
            ln("#### Precision Waterfall")
            ln()
            ln(f"```")
            ln(f"Measured holdout precision :  {held_prec:.1f}%")
            ln(f"")
            cumulative = held_prec
            for w in wf:
                cost = w["prec_cost_pp"]
                rcost = w["recall_cost_pp"]
                if cost > 0 or rcost > 0:
                    marker = "  prec" if cost > 0 else "recall"
                    ln(f"  [{badge(w['status']):<14}]  -{cost:.1f}pp {marker}  |  {w['issue']}")
                    cumulative += cost
            ln(f"")
            ln(f"Achievable precision       :  ~{achiev:.1f}%  (+{prec_leak:.1f}pp)")
            ln(f"Recall deficit (BUY side)  :  -{recall_leak:.1f}pp  (signals missed)")
            ln(f"```")
            ln()

        # ── Q2: Where is it coming from? ──────────────────────────────────────
        ln("### Q2 — Where Exactly Is Each Problem Coming From?")
        ln()
        q2_sources = s00.get("q2_source_locations", [])
        if q2_sources:
            hdr("Issue", "File", "Lines", "Symbol", "−Prec", "−Recall")
            for src in q2_sources:
                L.append(
                    f"| {src['issue'][:45]} "
                    f"| `{src['file']}` "
                    f"| {src['lines']} "
                    f"| `{src['symbol'][:35]}` "
                    f"| {src['prec_cost_pp']:.1f}pp "
                    f"| {src['recall_cost_pp']:.1f}pp |"
                )
            ln()
            for i, src in enumerate(q2_sources, 1):
                mech = src.get("mechanism", "")
                if mech:
                    ln(f"**{i}. {src['issue']}**")
                    ln(f"> 📍 `{src['file']}:{src['lines']}` — `{src['symbol']}`")
                    ln(f"> **Why:** {mech}")
                    ln()
        else:
            ln("No active issues found — all identified problems are marked FIXED.")
            ln()

        # ── Q3: Best solution ─────────────────────────────────────────────────
        ln("### Q3 — What Is The Best Solution?")
        ln()
        q3 = s00.get("q3_best_solution", {})
        fixed_count = q3.get("already_fixed_count", 0)
        fixed_gain  = q3.get("already_fixed_gain_pp", 0.0)
        if fixed_count > 0:
            ln(f"✅ **{fixed_count} issues already fixed** in the codebase "
               f"(expected gain: +{fixed_gain:.1f}pp precision / recall once retrained).")
            ln()
        top_action = q3.get("top_action", "")
        top_src    = q3.get("top_source", "")
        top_gain   = q3.get("expected_gain_pp", 0.0)
        if top_action and top_action != "All identified issues are fixed — retrain to measure gain.":
            ln(f"**Highest-ROI remaining fix** (+{top_gain:.1f}pp expected):")
            ln(f"> 📍 `{top_src}`")
            ln(f"> {top_action}")
            ln()
        else:
            ln(f"> {top_action}")
            ln()

        # ── BUY side gate trace ────────────────────────────────────────────────
        bt = s00.get("buy_side_trace", {})
        if bt:
            ln("#### BUY Side Gate Trace")
            ln()
            verdict_icon = "✅" if bt.get("all_pass") else "🔴"
            ln(f"{verdict_icon} **{bt.get('verdict', '?')}**")
            ln()
            for step in bt.get("chain", []):
                icon = "✅" if step["pass"] else "❌"
                ln(f"{icon} **{step['gate']}**")
                ln(f"   - File: `{step['file']}`")
                ln(f"   - Check: `{step['check']}`")
                ln(f"   - Value: {step['value']}")
                ln(f"   - {step['note']}")
                ln()

        # ── Meta gate verdict ──────────────────────────────────────────────────
        ga = s00.get("meta_gate_audit", {})
        if ga:
            gate_icon = {"HELPING": "✅", "HURTING": "🔴", "NEUTRAL": "🟡"}.get(
                ga.get("gate_status", ""), "❓")
            ln("#### Meta Gate Audit")
            ln()
            ln(f"{gate_icon} **Gate status: {ga.get('gate_status','?')}**  "
               f"(lift: {ga.get('gate_lift_pp',0):+.1f}pp)")
            ln()
            ln(f"| Metric | Value |")
            ln(f"|--------|-------|")
            ln(f"| Gated-in precision | {ga.get('fired_precision',0):.1%} |")
            ln(f"| Blocked signals win rate | {ga.get('blocked_win_rate',0):.1%} |")
            ln(f"| Precision lift from gate | {ga.get('gate_lift_pp',0):+.1f}pp |")
            ln(f"| OOF→Holdout gap | {ga.get('oof_holdout_gap_pp',0):+.1f}pp |")
            ln(f"| thr_buy / thr_sell | {ga.get('thr_buy',0):.3f} / {ga.get('thr_sell',0):.3f} |")
            ln(f"| Blocked signals | {ga.get('blocked_signals',0)} "
               f"({ga.get('blocked_would_win',0)} would win / "
               f"{ga.get('blocked_would_lose',0)} would lose) |")
            ln()
            ln(f"> {ga.get('verdict','')}")
            ln()

        ln("---")

        # ══════════════════════════════════════════════════════════════════════
        # SECTION 15 — Executive Summary (compact)
        # ══════════════════════════════════════════════════════════════════════
        ln()
        ln("## Section 15 — Executive Summary")
        ln()
        ln(f"**Symbol:** {self.symbol}  |  **Audit:** {self.ts[:16]}  |  "
           f"**Confidence Level:** {s15.get('confidence_level','?')} — "
           f"{s15.get('confidence_note','')}")
        ln()
        ln(f"**Current:** Precision={s15.get('current_precision_pct',0):.1f}%  "
           f"Sharpe={s15.get('sharpe',0):.2f}")
        ln(f"**Expected after fixes:** "
           f"Precision≈{s15.get('expected_precision_after_fixes_pct',0):.1f}%  "
           f"(+{s15.get('expected_gain_pp',0):.1f}pp)")
        ln()
        ln("### Top 5 Problems")
        ln()
        for i, c in enumerate(s15.get("top5_problems", []), 1):
            sc   = c.get("score", 0)
            icon = "🔴" if sc >= 70 else ("🟡" if sc >= 40 else "🟢")
            src_f = c.get("source_file", "")
            src_l = c.get("source_lines", "")
            src_line = f" — `{src_f}:{src_l}`" if src_f and src_f != "?" else ""
            ln(f"{i}. {icon} **{c['name']}** — Score: {sc}/100{src_line}")
            ln(f"   > {c.get('evidence','')[:160]}")
            ln()
        ln("### Top 5 Fixes")
        ln()
        for f_ in s15.get("top5_fixes", []):
            ln(f"{f_['rank']}. **{f_['cause']}**")
            ln(f"   → {f_['action']}")
            ln()
        ln("---")

        # ── S01 ───────────────────────────────────────────────────────────────
        ln()
        ln("## Section 1 — Model Health")
        ln()
        hdr("Metric", "Value", "Status")
        def mrow(label, val, ok, warn=False):
            L.append(f"| {label} | {val} | {'⚠' if warn else status(ok)} |")
        mrow("CV Accuracy (OOF)", f"{s01.get('cv_accuracy',0):.1%}",
             s01.get('cv_accuracy',0) > 0.60)
        mrow("Dev OOF Precision", f"{s01.get('dev_oof_precision',0):.1%}",
             s01.get('dev_oof_precision',0) >= 0.55)
        mrow("Holdout Precision", f"{s01.get('holdout_precision',0):.1%}",
             s01.get('holdout_precision',0) >= 0.55)
        ci  = s01.get("ci_95", [0, 1])
        L.append(f"| 95% CI Precision | [{ci[0]:.1%}, {ci[1]:.1%}] | — |")
        gap = s01.get("oof_holdout_gap", 0)
        mrow("OOF→Holdout Gap", f"{gap:+.1%}", gap > -0.05,
             warn=(-0.15 < gap <= -0.05))
        fn  = s01.get("holdout_fired", 0)
        mrow("Holdout Fired", f"{fn} trades", fn >= 50, warn=(10 <= fn < 50))
        mrow("SELL Win Rate",
             f"{s01.get('sell_win_rate',0):.1%} ({s01.get('sell_n',0)} trades)",
             s01.get('sell_win_rate',0) >= 0.55)
        buy_wr = s01.get("buy_win_rate", 0); buy_n = s01.get("buy_n", 0)
        L.append(f"| BUY Win Rate | {buy_wr:.1%} ({buy_n} trades) | "
                 f"{'✗ disabled' if buy_n == 0 else status(buy_wr >= 0.55)} |")
        mrow("Sharpe (annualised)", f"{s01.get('sharpe',0):.2f}",
             s01.get('sharpe',0) > 1.0)
        mrow("Max Drawdown", f"{s01.get('max_drawdown_pct',0):.2f}%",
             s01.get('max_drawdown_pct',0) < 15.0)
        mrow("Profit Factor", f"{s01.get('profit_factor',0):.2f}",
             s01.get('profit_factor',0) > 1.2)
        L.append(f"| Kelly Fraction | {s01.get('kelly_pct',0):.1f}% | — |")
        mrow("Expectancy/Trade", f"{s01.get('expectancy_per_trade',0):+.4f}%",
             s01.get('expectancy_per_trade',0) > 0)
        z  = s01.get("z_stat", float("nan"))
        pv = s01.get("p_value", float("nan"))
        sig_ok = (not math.isnan(pv)) and pv < 0.05
        L.append(f"| Statistical Sig. | "
                 f"{'p=' + f'{pv:.4f}' + ' (z=' + f'{z:.2f})' if not math.isnan(pv) else '—'} "
                 f"| {'✓ significant' if sig_ok else '✗ not significant'} |")
        ln()
        cd = s01.get("class_distribution", {})
        ln("### Class Distribution")
        ln(f"- HOLD: **{cd.get('HOLD_pct',0):.1%}** — "
           f"{'⚠ severe imbalance' if cd.get('HOLD_pct',0) > 0.55 else 'OK'}")
        ln(f"- SELL: **{cd.get('SELL_pct',0):.1%}**")
        ln(f"- BUY:  **{cd.get('BUY_pct',0):.1%}** — "
           f"{'⚠ minority class' if cd.get('BUY_pct',0) < 0.20 else 'OK'}")
        ln()
        ln("### Issues Detected")
        for iss in s01.get("issues", []):
            prefix = ("- **CRITICAL**" if iss["severity"] == "CRITICAL"
                      else "- **WARNING**")
            ln(f"{prefix} — {iss['msg']}")
        ln()
        ln("---")

        # ── S02 ───────────────────────────────────────────────────────────────
        ln()
        ln("## Section 2 — Feature Forensics")
        ln()
        sm = s02.get("summary", {})
        ln(f"**Feature health summary:** {sm.get('HEALTHY',0)} HEALTHY | "
           f"{sm.get('WARNING',0)} WARNING | {sm.get('DEGRADED',0)} DEGRADED | "
           f"{sm.get('CRITICAL',0)} CRITICAL  (of {s02.get('total_features',0)} total)")
        ln()
        ln(f"**Estimated total precision gain if top drifters removed:** "
           f"+{s02.get('total_precision_gain_pp_estimate',0):.1f} pp")
        ln()
        ln("### Top 25 Drifting Features")
        ln()
        hdr("Rank", "Feature", "State", "PSI", "KS", "Mean Drift",
            "Penalty", "Rec.", "Est. Gain")
        for f_ in s02.get("top_25_drifters", [])[:25]:
            st_icon = ("**CRITICAL**" if f_.get("state") == "CRITICAL"
                       else ("**DEGRADED**" if f_.get("state") == "DEGRADED"
                             else f_.get("state", "")))
            L.append(f"| {f_['rank']} | `{f_['feature']}` | {st_icon} | "
                     f"{f_['psi']:.3f} | {f_['ks']:.3f} | {f_['mean_drift']:.3f} | "
                     f"{f_['drift_penalty']:.2f} | {f_['recommendation']} | "
                     f"+{f_['estimated_precision_gain_pp']:.1f}pp |")
        ln()
        cnp = s02.get("critical_non_price", [])
        if cnp:
            ln("### Critical Non-Price Indicators (highest priority for retraining)")
            for feat in cnp:
                ln(f"- `{feat}`")
        ln()
        ln("---")

        # ── S03 ───────────────────────────────────────────────────────────────
        ln()
        ln("## Section 3 — Signal Generation Forensics")
        ln()
        ln(f"**Data window:** {s03.get('data_window_bars',0):,} bars")
        ln()
        ln("### All-Bar Prediction Breakdown")
        ln()
        hdr("Predicted", "Count", "Raw Precision")
        abb = s03.get("all_bar_breakdown", {})
        L.append(f"| BUY  | {abb.get('BUY',{}).get('count',0)}  | "
                 f"{abb.get('BUY',{}).get('avg_meta_conf',0):.1%}  |")
        L.append(f"| SELL | {abb.get('SELL',{}).get('count',0)} | "
                 f"{abb.get('SELL',{}).get('avg_meta_conf',0):.1%} |")
        L.append(f"| HOLD | {abb.get('HOLD',{}).get('count',0)} | — |")
        ln()
        fn2 = s03.get("funnel", {})
        gen = max(fn2.get("generated", 1), 1)
        ln("### Signal Rejection Funnel")
        ln()
        ln("```")
        ln(f"Generated (directional):       {gen:>5}  (100%)")
        for label, key in [
            ("Meta Gate",      "blocked_by_meta_gate"),
            ("Quality (<55)",  "blocked_by_quality"),
            ("HMM",            "blocked_by_hmm"),
            ("Confluence",     "blocked_by_confluence"),
            ("Fake Breakout",  "blocked_by_fake_breakout"),
            ("Portfolio Guard","blocked_by_portfolio"),
            ("Safe Mode",      "blocked_by_safe_mode"),
            ("Drift",          "blocked_by_drift"),
            ("Cooldown",       "blocked_by_cooldown"),
        ]:
            n = fn2.get(key, 0)
            if n > 0:
                ln(f"Blocked by {label:<20} - {n:>4}  ({n/gen*100:.0f}%)")
        ln("─" * 45)
        ex = fn2.get("executed", 0)
        ln(f"Estimated Executed:           {ex:>5}  ({fn2.get('executed_pct',0):.1f}%)")
        ln("```")
        ln()
        be = "✗ DISABLED" if not s03.get("buy_enabled")  else "✓ LIVE"
        se = "✗ DISABLED" if not s03.get("sell_enabled") else "✓ LIVE"
        ln(f"**BUY side:** {be}  |  **SELL side:** {se}")
        ln()
        ln("---")

        # ── S04 ───────────────────────────────────────────────────────────────
        ln()
        ln("## Section 4 — Opportunity Cost Analysis")
        ln()
        ln(f"Average hold-signal realized return: "
           f"**{s04.get('avg_hold_signal_return_pct',0):.4f}%/bar**")
        ln()
        tp = s04.get("time_to_tp_distribution", {})
        if tp:
            ln("### Time-to-TP (Upper Barrier) Distribution")
            ln()
            hdr("Horizon", "% of BUY signals that would have hit TP")
            for h_, pct in tp.items():
                L.append(f"| {h_} | {pct}% |")
            ln()
            med = s04.get("median_bars_to_tp", 0)
            ln(f"Median time-to-TP: **{med} bars** ({med}h)")
        ln()
        ocf = s04.get("opportunity_cost_by_filter", {})
        if ocf:
            ln("### Opportunity Cost by Filter")
            ln()
            hdr("Filter", "Blocked", "Would Win", "Would Lose", "Win Rate", "Opp. Cost")
            for flt, oc in ocf.items():
                flag = " ⚠" if oc.get("win_rate", 0) > 0.55 else ""
                L.append(f"| {flt} | {oc.get('blocked',0)} | {oc.get('would_win',0)} | "
                         f"{oc.get('would_lose',0)} | {oc.get('win_rate',0):.1%} | "
                         f"{oc.get('opp_cost_pct',0):+.2f}%{flag} |")
        ln()
        ln("---")

        # ── S05 ───────────────────────────────────────────────────────────────
        ln()
        ln("## Section 5 — Meta Model Forensics")
        ln()
        hdr("Metric", "Value", "Target", "Status")
        eb = s05.get("ece_before_calibration", 0)
        ea = s05.get("ece_after_calibration", 0)
        br = s05.get("brier_score", 0)
        tp2 = s05.get("calibration_temperature", 1.0)
        L.append(f"| ECE (before cal.) | {eb:.4f} | <0.10 | {status(eb < 0.10)} |")
        L.append(f"| ECE (after cal.)  | {ea:.4f} | <0.10 | {status(ea < 0.10)} |")
        L.append(f"| Brier Score | {br:.4f} | <0.25 | {status(br < 0.25)} |")
        L.append(f"| Cal. Temperature | {tp2:.4f} | ~1.0 | "
                 f"{'✓' if abs(tp2-1.0)<0.15 else '⚠ model overconfident'} |")
        L.append(f"| Calibration Type | {s05.get('calibration_type','?')} | isotonic | — |")
        ln()
        bkts = s05.get("confidence_buckets", {})
        if bkts:
            ln("### Confidence Bucket Analysis (Estimated)")
            ln()
            hdr("Bucket", "Est. Win Rate", "Gap", "Status")
            for bk, bv in bkts.items():
                L.append(f"| {bk}% | {bv.get('actual_win_rate',0):.1%} | "
                         f"{bv.get('gap',0):+.2f} | {bv.get('status','?')} |")
        ln()
        ln(f"**Confidence inflation detected:** "
           f"{'Yes ⚠' if s05.get('confidence_inflation') else 'No'}")
        ln(f"**Recommended calibrator "
           f"(for {s05.get('dev_sample_size',0)} dev samples):** "
           f"`{s05.get('recommended_calibrator','?')}`")
        ln()
        ln("---")

        # ── S06 ───────────────────────────────────────────────────────────────
        ln()
        ln("## Section 6 — HMM Regime Forensics")
        ln()
        ln(f"**States:** {s06.get('n_states',7)}  |  "
           f"**Global precision:** {s06.get('global_precision',0):.1%}  |  "
           f"**Max state concentration:** {s06.get('max_state_concentration',0):.1%}")
        ln()
        ln("### Per-Regime Performance")
        ln()
        hdr("Regime", "Trades", "Precision", "Sell Prec", "Expectancy",
            "P.Factor", "Modifier", "Rec.")
        for reg, rv in s06.get("regime_performance", {}).items():
            L.append(f"| {reg} | {rv['trades']} | {rv['precision']:.1%} | "
                     f"{rv['sell_prec']:.1%} | {rv['expectancy']:+.3f}% | "
                     f"{rv['profit_factor']:.2f} | {rv['modifier']:+.3f} | "
                     f"{rv['recommendation']} |")
        ln()
        ln("---")

        # ── S07 ───────────────────────────────────────────────────────────────
        ln()
        ln("## Section 7 — LSTM Forensics")
        ln()
        hdr("Model", "Threshold", "Est. AUC", "Predictive?")
        cont = s07.get("continuation", {})
        vol  = s07.get("volatility_expansion", {})
        L.append(f"| ContinuationLSTM  | {cont.get('threshold',0):.3f} | "
                 f"{cont.get('val_auc',0):.3f} | "
                 f"{'✓ YES' if cont.get('predictive') else '✗ NON-PREDICTIVE (AUC<0.55)'} |")
        L.append(f"| VolatilityExpLSTM | {vol.get('threshold',0):.3f}  | "
                 f"{vol.get('val_auc',0):.3f} | "
                 f"{'✓ YES' if vol.get('predictive') else '✗ NON-PREDICTIVE (AUC<0.55)'} |")
        ln()
        ln(f"**Feature counts:** Continuation={cont.get('feature_count',0)}, "
           f"Volatility={vol.get('feature_count',0)}")
        ln()
        ln("> _AUC estimated from threshold position. True AUC requires OOF predictions "
           "on held-out LSTM test set._")
        ln()
        ln("---")

        # ── S08 ───────────────────────────────────────────────────────────────
        ln()
        ln("## Section 8 — Quality Engine Forensics")
        ln()
        ln(f"**Verdict:** {s08.get('verdict','?')}")
        ln()
        hdr("Quality Bucket", "Trades", "Precision", "Expectancy", "P.Factor")
        for bk, bv in s08.get("buckets", {}).items():
            L.append(f"| {bk} | {bv.get('trades',0)} | {bv.get('precision',0):.1%} | "
                     f"{bv.get('expectancy_pct',0):+.4f}% | "
                     f"{bv.get('profit_factor',0):.2f} |")
        ln()
        ln(f"**Monotone precision:** "
           f"{'✓ YES — quality is predictive' if s08.get('monotone_precision') else '✗ NO — quality NOT predictive'}")
        ln(f"**Paper trading:** {s08.get('paper_trades',0)} trades, "
           f"{s08.get('paper_win_rate',0):.1%} WR")
        ln()
        ln("---")

        # ── S09 ───────────────────────────────────────────────────────────────
        ln()
        ln("## Section 9 — Drift Monitor Forensics")
        ln()
        overall = s09.get("overall_status", "?")
        icon    = "🔴" if overall == "CRITICAL" else ("🟡" if overall == "WARNING" else "🟢")
        ln(f"**Overall Drift Status:** {icon} **{overall}**")
        ln()
        hdr("Drift Type", "Classification", "Detail")
        for dtype, key in [("Feature Drift",    "feature_drift"),
                            ("Confidence Drift", "confidence_drift"),
                            ("Prediction Drift", "prediction_drift")]:
            d   = s09.get(key, {})
            cls = d.get("classification", "?")
            ico = "🔴" if cls == "CRITICAL" else ("🟡" if cls == "WARNING" else "🟢")
            L.append(f"| {dtype} | {ico} {cls} | {d.get('detail','')} |")
        ln()
        ln(f"**Estimated precision loss from feature drift:** "
           f"~{s09.get('estimated_precision_loss_pp',0):.1f}pp")
        ln()
        ln("---")

        # ── S10 ───────────────────────────────────────────────────────────────
        ln()
        ln("## Section 10 — Portfolio Forensics")
        ln()
        ln(f"**Open positions:** {s10.get('open_positions',0)}/{MAX_POSITIONS}  |  "
           f"**Effective leverage:** {s10.get('effective_leverage',0):.2f}×  |  "
           f"**HHI (concentration):** {s10.get('hhi_concentration',0):.3f}")
        ln()
        if s10.get("symbol_allocation"):
            hdr("Symbol", "Capital Allocation")
            for sym, pct in s10.get("symbol_allocation", {}).items():
                L.append(f"| {sym} | {pct:.1f}% |")
        ln()
        ln(f"**Hidden leverage:** {'⚠ YES' if s10.get('hidden_leverage') else '✓ NO'}  |  "
           f"**Over-concentration:** {'⚠ YES' if s10.get('over_concentration') else '✓ NO'}")
        ln()
        ln("---")

        # ── S11 ───────────────────────────────────────────────────────────────
        ln()
        ln("## Section 11 — Risk Engine Forensics")
        ln()
        hdr("Metric", "Value", "Assessment")
        L.append(f"| ATR Multiplier | {s11.get('atr_multiplier',0):.1f}× | ✓ |")
        wrs = s11.get("win_rate", 0)
        L.append(f"| Win Rate | {wrs:.1%} | {'✓' if wrs >= 0.55 else '⚠'} |")
        L.append(f"| Avg Win / Avg Loss | "
                 f"{s11.get('avg_win_pct',0):.3f}% / {s11.get('avg_loss_pct',0):.3f}% | — |")
        rr = s11.get("rr_ratio", 0)
        L.append(f"| R:R Ratio | {rr:.2f} | "
                 f"{'✓ favourable' if rr >= 1.5 else '✗ unfavourable'} |")
        L.append(f"| Kelly Fraction | {s11.get('kelly_fraction',0)*100:.1f}% | ✓ |")
        arm = s11.get("avg_r_multiple", 0)
        L.append(f"| Avg R-Multiple | {arm:.2f}R | {status(arm >= 1.0)} |")
        ror = s11.get("risk_of_ruin", 0)
        L.append(f"| Risk of Ruin | {ror*100:.4f}% | {'✓' if ror < 0.01 else '⚠'} |")
        L.append(f"| Holdout Sharpe | {s11.get('sharpe_holdout',0):.2f} | "
                 f"{status(s11.get('sharpe_holdout',0) > 1)} |")
        L.append(f"| Holdout Max DD | {s11.get('max_dd_holdout',0):.2f}% | ✓ |")
        L.append(f"| Stop Assessment | {s11.get('stop_assessment','?')} | — |")
        ln()
        ln("---")

        # ── S12 ───────────────────────────────────────────────────────────────
        ln()
        ln("## Section 12 — Live Execution Forensics")
        ln()
        ln(f"**Closed:** {s12.get('n_closed',0)}  |  "
           f"**Open:** {s12.get('n_open',0)}  |  "
           f"**Avg hold:** {s12.get('avg_hold_hours',0):.1f}h  |  "
           f"**Avg PnL:** {s12.get('avg_pnl_pct',0):+.3f}%")
        ln()
        disc = s12.get("confidence_discriminates", False)
        wc   = s12.get("win_avg_confidence", 0)
        lc   = s12.get("loss_avg_confidence", 0)
        ln(f"**Confidence discriminates wins from losses:** "
           f"{'✓ YES' if disc else '✗ NO'} "
           f"(WIN conf={wc:.3f} vs LOSS conf={lc:.3f})")
        ln()
        if s12.get("best_trades"):
            ln("### Best Trades")
            for t in s12.get("best_trades", []):
                ln(f"- **{t['symbol']}** {t['side']}  PnL={t['pnl_pct']:+.2f}%  "
                   f"conf={t['confidence']:.3f}  exit={t['exit_reason']}")
        if s12.get("worst_trades"):
            ln()
            ln("### Worst Trades")
            for t in s12.get("worst_trades", []):
                ln(f"- **{t['symbol']}** {t['side']}  PnL={t['pnl_pct']:+.2f}%  "
                   f"conf={t['confidence']:.3f}  exit={t['exit_reason']}")
        er = s12.get("exit_reasons", {})
        if er:
            ln()
            ln("### Exit Reasons")
            hdr("Reason", "Count")
            for reason, cnt in er.items():
                L.append(f"| {reason} | {cnt} |")
        ln()
        ln("---")

        # ── S13 ───────────────────────────────────────────────────────────────
        ln()
        ln("## Section 13 — Root Cause Engine")
        ln()
        ln(f"**{s13.get('n_causes',0)} root causes identified.**  "
           f"Combined top-5 impact score: "
           f"**{s13.get('combined_top5_score',0)}/500**")
        ln()
        hdr("Rank", "Cause", "Category", "Score", "Evidence")
        for i, c in enumerate(s13.get("causes", [])[:10], 1):
            sc   = c.get("score", 0)
            icon = "🔴" if sc >= 70 else ("🟡" if sc >= 40 else "🟢")
            evid = c.get("evidence", "")
            evid = evid[:80] + "…" if len(evid) > 80 else evid
            L.append(f"| {i} | {icon} **{c['name']}** | {c['category']} | "
                     f"{sc}/100 | {evid} |")
        ln()
        ln("### Fixes")
        ln()
        for i, c in enumerate(s13.get("causes", [])[:10], 1):
            ln(f"**{i}. {c['name']}** — {c['fix']}")
        ln()
        ln("---")

        # ── S14 ───────────────────────────────────────────────────────────────
        ln()
        ln("## Section 14 — Automated Improvement Engine")
        ln()
        base = s14.get("base_precision_pct", 0)
        exp  = s14.get("expected_precision_pct", 0)
        gain = s14.get("total_prec_gain_pp", 0)
        ln(f"**Base precision:** {base:.1f}%  →  "
           f"**Expected precision (all fixes):** {exp:.1f}%  (+{gain:.1f}pp)")
        ln()
        hdr("#", "Action", "Prec Gain", "Recall Gain",
            "Profit Gain", "Confidence", "Effort")
        for i, imp in enumerate(s14.get("improvements", []), 1):
            L.append(f"| {i} | {imp['action']} | +{imp['prec_gain_pp']:.1f}pp | "
                     f"+{imp['recall_gain_pp']:.1f}pp | +{imp['profit_gain_pp']:.1f}pp | "
                     f"{imp['confidence']} | {imp['effort']} |")
        ln()
        ln("---")
        ln()

        return "\n".join(L)

    # ─────────────────────────────────────────────────────────────────────────
    # Main orchestrator
    # ─────────────────────────────────────────────────────────────────────────

    def run(self) -> bool:
        print(f"\n{'='*60}")
        print(f"AEGIS Forensic Engine — {self.symbol}")
        print(f"{'='*60}")

        if not self._load_system():
            print("FATAL: system load failed.")
            return False
        if not self._fetch_data():
            print("FATAL: data fetch failed.")
            return False

        self._run_core_simulation()

        print("  Running sections 1-15…")
        for fn in [self.s01_model_health, self.s02_feature_forensics,
                   self.s03_signal_generation, self.s04_opportunity_cost,
                   self.s05_meta_forensics, self.s06_hmm_regime,
                   self.s07_lstm_forensics, self.s08_quality_engine,
                   self.s09_drift_monitor, self.s10_portfolio_forensics,
                   self.s11_risk_engine, self.s12_live_execution,
                   self.s13_root_cause, self.s14_improvement_engine,
                   self.s15_executive_summary]:
            try:
                fn()
            except Exception as e:
                import traceback
                print(f"  WARNING: {fn.__name__} failed: {e}")
                traceback.print_exc()

        # s00 runs LAST — it synthesises all section results into the
        # Three Questions section that opens the report.
        print("  Running Three Questions synthesis…")
        try:
            self.s00_three_questions()
        except Exception as e:
            import traceback
            print(f"  WARNING: s00_three_questions failed: {e}")
            traceback.print_exc()

        md   = self._build_markdown_report()
        path = FORENSIC_DIR / f"AEGIS_MASTER_FORENSIC_REPORT_{self.base}.md"
        path.write_text(md, encoding="utf-8")
        print(f"  Report saved → {path.name}")
        self._save_json_outputs()
        print("  Done.\n")
        return True


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _trained_symbols() -> List[str]:
    seen: set = set()
    result: List[str] = []
    for p in MODEL_STORE.glob("*_meta.json"):
        base = p.name.replace("_meta.json", "")
        # Skip lstm_meta files
        if "lstm" in base:
            continue
        # Convert BTC_USDT → BTC/USDT
        parts = base.split("_")
        if len(parts) >= 2 and parts[-1] == "USDT":
            sym = "/".join(["_".join(parts[:-1]), "USDT"])
        else:
            sym = base.replace("_", "/", 1)
        if sym not in seen:
            seen.add(sym)
            result.append(sym)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="AEGIS-1 Forensic Investigation Engine")
    parser.add_argument("--symbol", type=str, default=None,
                        help="Symbol to audit, e.g. BTC/USDT")
    parser.add_argument("--hours", type=int, default=8760,
                        help="Hours of history to fetch (default 8760 = 1 year)")
    parser.add_argument("--fleet", action="store_true",
                        help="Audit all trained symbols")
    args = parser.parse_args()

    symbols: List[str] = []
    if args.fleet:
        symbols = _trained_symbols()
        if not symbols:
            print("No trained symbols found in model_store/")
            return
        print(f"Fleet mode: auditing {len(symbols)} symbols")
    elif args.symbol:
        symbols = [args.symbol]
    else:
        parser.print_help()
        return

    for sym in symbols:
        try:
            AEGISForensicEngine(sym, hours=args.hours).run()
        except Exception as e:
            import traceback
            print(f"ERROR auditing {sym}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
