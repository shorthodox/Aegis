"""
src/ml/lstm_trainer.py
──────────────────────
Training pipeline for the AEGIS LSTM temporal intelligence layer.

Overview
--------
Two models per symbol are trained and saved to model_store/:
  {base}_lstm_cont.pt       — ContinuationLSTM weights
  {base}_lstm_vol.pt        — VolatilityExpansionLSTM weights
  {base}_lstm_meta.json     — scalers, thresholds, metrics, feature lists

Label design
------------
  Continuation label  (triple-barrier, bar-direction aware):
      1 = price continues in current bar's direction (TP hit before SL)
      0 = price reverses (SL hit before TP)
      NaN = uncertain / insufficient lookahead  → excluded

  Volatility-expansion label (ATR ratio):
      1 = ATR at t+12 ≥ ATR at t × 1.4   (expansion)
      0 = ATR at t+12 ≤ ATR at t × 1.1   (stable / compression)
      NaN = in between                    → excluded

Overfitting prevention
----------------------
  - Dropout (0.30) + LayerNorm on every LSTM layer
  - Temporal noise augmentation on training sequences (σ = 0.01)
  - Weighted random sampling to balance class imbalance
  - Early stopping on validation BCELoss (patience = 10)
  - Walk-forward hold-out split (last 20 % of time-ordered data)
  - Per-symbol RobustScaler: (x − median) / IQR, clipped ± 5

Checkpoint / resume
-------------------
  A symbol is considered "done" when its *_lstm_meta.json exists.
  find_lstm_trained_symbols() returns the set of completed symbols so
  train_lstm_fleet() can skip them on a resumed run.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

from src.ml.lstm_models import (
    CONT_FEATURES, CONT_SEQ_LEN,
    VOL_EXP_FEATURES, VOL_EXP_SEQ_LEN,
    ContinuationLSTM, VolatilityExpansionLSTM,
)

MODEL_STORE = Path(__file__).parent / "model_store"

# ── Feature-column aliases ─────────────────────────────────────────────────────
# Maps canonical LSTM feature name → ordered list of fallback column names.
# The trainer tries each alias in order and uses the first one found in df.
# If none exist the feature is zero-filled (safe degradation, not a crash).
_ALIASES: Dict[str, List[str]] = {
    "total_confluence":    ["total_confluence"],
    "prc_total":           ["prc_total"],
    "volume_zscore":       ["volume_zscore", "vol_zscore"],
    "volume_delta":        ["volume_delta"],
    "rsi_14":              ["rsi_14", "rsi"],
    "macd_hist":           ["macd_hist", "macd_histogram"],
    "dist_ema_21":         ["dist_ema_21", "ema21_dist"],
    "efficiency_ratio_10": ["efficiency_ratio_10", "efficiency_ratio"],
    "funding_rate_zscore": ["funding_rate_zscore", "funding_zscore"],
    "oi_change_1h":        ["oi_change_1h", "oi_delta_1h"],
    "btc_corr_24h":        ["btc_corr_24h", "btc_corr_1h", "btc_corr"],
    "sqz_on":              ["sqz_on", "squeeze_on", "is_squeeze"],
    "bb_pct_b":            ["bb_pct_b", "bb_b"],
    "close_position":      ["close_position"],
    "vol_velocity":        ["vol_velocity", "volume_velocity"],
    "macd_signal":         ["macd_signal", "macd_sig"],
    "rsi_7":               ["rsi_7"],
    "adx_14":              ["adx_14", "adx"],
    # vol-expansion exclusive
    "atr_14":              ["atr_14", "_atr"],
    "bb_width_percentile": ["bb_width_percentile", "bb_width"],
    "oi_chg_8h":           ["oi_chg_8h", "oi_chg_8", "oi_change_4h"],
}


def _resolve_features(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    """
    Build a float32 DataFrame with exactly the requested columns.
    Uses _ALIASES to find the best available alternative.
    Missing columns → filled with 0.0.
    """
    out: Dict[str, np.ndarray] = {}
    n = len(df)
    for name in feature_names:
        candidates = _ALIASES.get(name, [name])
        found: Optional[str] = None
        for c in candidates:
            if c in df.columns:
                found = c
                break
        if found:
            vals = df[found].values.astype(np.float32)
        else:
            vals = np.zeros(n, dtype=np.float32)
        # sanitise: replace inf/NaN with 0
        vals = np.where(np.isfinite(vals), vals, 0.0)
        out[name] = vals
    return pd.DataFrame(out, index=df.index)


# ── Label generators ───────────────────────────────────────────────────────────

def make_continuation_labels(
    df: pd.DataFrame,
    lookahead:   int   = 12,
    tp_mult:     float = 1.0,   # TP = close ± tp_mult × ATR
    sl_mult:     float = 0.7,   # SL = close ∓ sl_mult × ATR
    atr_col:     str   = "_atr",
) -> np.ndarray:
    """
    Triple-barrier continuation label (bar-direction aware).

    For bar t:
      direction = UP  if close[t] >= open[t]
      direction = DOWN otherwise

    Look forward up to `lookahead` bars:
      If UP:   label 1 when high[t+k] >= TP first, label 0 when low[t+k] <= SL first
      If DOWN: label 1 when low[t+k]  <= TP first, label 0 when high[t+k] >= SL first
    Bars without a decisive hit → NaN (excluded from training).

    Returns float32 array length n, NaN where excluded.
    """
    n = len(df)
    labels = np.full(n, np.nan, dtype=np.float32)

    # Prefer _atr (computed in train_token), fall back to atr_14
    if atr_col in df.columns:
        atr_arr = df[atr_col].values.astype(np.float64)
    elif "atr_14" in df.columns:
        atr_arr = df["atr_14"].values.astype(np.float64)
    else:
        atr_arr = (df["close"].values * 0.015)  # 1.5% proxy

    close_arr = df["close"].values.astype(np.float64)
    open_arr  = df["open"].values.astype(np.float64)
    high_arr  = df["high"].values.astype(np.float64)
    low_arr   = df["low"].values.astype(np.float64)

    for t in range(n - lookahead):
        c_t   = close_arr[t]
        atr_t = max(float(atr_arr[t]), c_t * 0.002)  # floor at 0.2 %
        bullish = close_arr[t] >= open_arr[t]

        tp = c_t + atr_t * tp_mult if bullish else c_t - atr_t * tp_mult
        sl = c_t - atr_t * sl_mult if bullish else c_t + atr_t * sl_mult

        label = np.nan
        for k in range(1, lookahead + 1):
            hi = high_arr[t + k]
            lo = low_arr[t + k]
            if bullish:
                if hi >= tp:
                    label = 1.0; break
                if lo <= sl:
                    label = 0.0; break
            else:
                if lo <= tp:
                    label = 1.0; break
                if hi >= sl:
                    label = 0.0; break
        labels[t] = label

    return labels


def make_vol_expansion_labels(
    df: pd.DataFrame,
    lookahead:     int   = 12,
    expand_thresh: float = 1.4,   # ATR grows ≥ 40 % → expansion
    stable_thresh: float = 1.1,   # ATR grows < 10 % → stable / compression
    atr_col:       str   = "_atr",
) -> np.ndarray:
    """
    Binary volatility-expansion label.

      1 = ATR at t+lookahead ≥ ATR[t] × expand_thresh
      0 = ATR at t+lookahead ≤ ATR[t] × stable_thresh
      NaN = uncertain zone  → excluded from training

    Returns float32 array length n.
    """
    n = len(df)
    labels = np.full(n, np.nan, dtype=np.float32)

    if atr_col in df.columns:
        atr_arr = df[atr_col].values.astype(np.float64)
    elif "atr_14" in df.columns:
        atr_arr = df["atr_14"].values.astype(np.float64)
    else:
        atr_arr = df["close"].values * 0.015

    for t in range(n - lookahead):
        a_now = float(atr_arr[t])
        a_fut = float(atr_arr[t + lookahead])
        if a_now <= 0 or not np.isfinite(a_now) or not np.isfinite(a_fut):
            continue
        ratio = a_fut / a_now
        if ratio >= expand_thresh:
            labels[t] = 1.0
        elif ratio <= stable_thresh:
            labels[t] = 0.0
        # else: uncertain → NaN

    return labels


# ── Robust feature scaler (JSON-serialisable) ──────────────────────────────────

class _RobustScaler:
    """
    Feature-wise RobustScaler: (x − median) / IQR, clipped to [−5, 5].
    Serialisable to/from dict → no sklearn dependency at inference time.
    """

    def __init__(self) -> None:
        self.median_: Optional[np.ndarray] = None
        self.iqr_:    Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "_RobustScaler":
        q25, q75 = np.nanpercentile(X, [25, 75], axis=0)
        self.median_ = np.nanmedian(X, axis=0)
        self.iqr_    = np.maximum(q75 - q25, 1e-8)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.median_ is None:
            return X.copy()
        return np.clip((X - self.median_) / self.iqr_, -5.0, 5.0).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X); return self.transform(X)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "median": self.median_.tolist() if self.median_ is not None else None,
            "iqr":    self.iqr_.tolist()    if self.iqr_    is not None else None,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "_RobustScaler":
        sc = cls()
        if d.get("median") is not None:
            sc.median_ = np.array(d["median"], dtype=np.float64)
        if d.get("iqr") is not None:
            sc.iqr_ = np.array(d["iqr"], dtype=np.float64)
        return sc


# ── Sliding-window sequence dataset ───────────────────────────────────────────

if _TORCH_OK:
    class _SequenceDataset(Dataset):
        def __init__(
            self,
            X:        np.ndarray,  # (n, seq_len, n_feat)
            y:        np.ndarray,  # (n,)
            augment:  bool  = False,
            noise_σ:  float = 0.01,
        ) -> None:
            self.X      = torch.tensor(X, dtype=torch.float32)
            self.y      = torch.tensor(y, dtype=torch.float32)
            self.aug    = augment
            self.noise  = noise_σ

        def __len__(self) -> int:
            return len(self.y)

        def __getitem__(self, idx: int):
            x = self.X[idx]
            if self.aug:
                x = x + torch.randn_like(x) * self.noise
            return x, self.y[idx]
else:
    class _SequenceDataset:  # type: ignore[no-redef]
        pass


# ── Sequence builder ───────────────────────────────────────────────────────────

def _build_sequences(
    feat_df: pd.DataFrame,
    labels:  np.ndarray,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) arrays for LSTM training.
    Returns only rows where labels are finite (not NaN) and seq_len bars exist.
    X shape: (n_valid, seq_len, n_features)
    """
    n = len(feat_df)
    vals = feat_df.values.astype(np.float32)
    vals = np.where(np.isfinite(vals), vals, 0.0)

    X_list, y_list = [], []
    for i in range(seq_len - 1, n):
        lbl = labels[i]
        if not np.isfinite(lbl):
            continue
        seq = vals[i - seq_len + 1 : i + 1]
        X_list.append(seq)
        y_list.append(float(lbl))

    if not X_list:
        empty_X = np.empty((0, seq_len, feat_df.shape[1]), dtype=np.float32)
        return empty_X, np.empty(0, dtype=np.float32)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


# ── Calibration helper ─────────────────────────────────────────────────────────

def _best_threshold(preds: np.ndarray, labels: np.ndarray) -> float:
    """F1-maximising threshold on validation predictions."""
    if len(labels) < 30 or labels.sum() < 5 or (1 - labels).sum() < 5:
        return 0.50
    best_f1, best_thr = 0.0, 0.50
    for thr in np.arange(0.30, 0.75, 0.05):
        p = (preds >= thr).astype(int)
        tp = ((p == 1) & (labels == 1)).sum()
        fp = ((p == 1) & (labels == 0)).sum()
        fn = ((p == 0) & (labels == 1)).sum()
        denom = 2 * tp + fp + fn
        if denom == 0:
            continue
        f1 = 2 * tp / denom
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr


# ── Core training function ─────────────────────────────────────────────────────

def _train_model(
    model:      "nn.Module",
    X_tr:       np.ndarray,
    y_tr:       np.ndarray,
    X_vl:       np.ndarray,
    y_vl:       np.ndarray,
    epochs:     int   = 50,
    batch_size: int   = 256,
    lr:         float = 1e-3,
    patience:   int   = 10,
    device:     str   = "cpu",
) -> Tuple["nn.Module", float]:
    """
    Train a single LSTM model with early stopping.
    Returns (best_model, best_val_loss).
    """
    if not _TORCH_OK:
        raise ImportError("PyTorch is required for LSTM training.")

    model = model.to(device)

    # Weighted sampler: balance positive / negative classes
    counts = np.bincount(y_tr.astype(int), minlength=2).astype(float)
    if counts.min() >= 5:
        sample_weights = np.where(
            y_tr == 1, 1.0 / counts[1], 1.0 / counts[0]
        )
        sampler = WeightedRandomSampler(
            torch.tensor(sample_weights, dtype=torch.float32),
            num_samples=len(y_tr),
            replacement=True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_dl = DataLoader(
        _SequenceDataset(X_tr, y_tr, augment=True),
        batch_size=batch_size, sampler=sampler, shuffle=shuffle,
        num_workers=0, pin_memory=(device != "cpu"),
    )
    val_dl = DataLoader(
        _SequenceDataset(X_vl, y_vl, augment=False),
        batch_size=512, shuffle=False, num_workers=0,
    )

    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss  = math.inf
    best_state: Optional[Dict[str, Any]] = None
    patience_c = 0

    for _ in range(epochs):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        v_losses: List[float] = []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                v_losses.append(criterion(model(xb), yb).item())
        v_loss = float(np.mean(v_losses))

        if v_loss < best_loss - 1e-5:
            best_loss  = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_c = 0
        else:
            patience_c += 1
            if patience_c >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to("cpu").eval()
    return model, best_loss


# ── Per-symbol training entry point ───────────────────────────────────────────

def train_lstm_for_symbol(
    symbol:  str,
    df:      pd.DataFrame,
    verbose: bool = True,
) -> bool:
    """
    Train ContinuationLSTM + VolatilityExpansionLSTM for one symbol.

    Parameters
    ----------
    symbol  : e.g. 'BTC/USDT'
    df      : fully feature-engineered DataFrame (output of retrain_model.py
              train_token after prepare_features + soft-confluence + _atr).
              Must contain OHLCV + all indicator columns.
    verbose : print progress lines

    Writes to model_store/
        {base}_lstm_cont.pt
        {base}_lstm_vol.pt      (only when vol-expansion labels are sufficient)
        {base}_lstm_meta.json

    Returns True on success, False on skip / failure.
    """
    if not _TORCH_OK:
        if verbose:
            print(f"   [LSTM] PyTorch not installed — skipping LSTM for {symbol}")
        return False

    torch.manual_seed(42)
    np.random.seed(42)

    if df is None or len(df) < 300:
        if verbose:
            n = 0 if df is None else len(df)
            print(f"   [LSTM] Insufficient data for {symbol} (n={n})")
        return False

    base      = symbol.replace("/", "_")
    cont_path = MODEL_STORE / f"{base}_lstm_cont.pt"
    vol_path  = MODEL_STORE / f"{base}_lstm_vol.pt"
    meta_path = MODEL_STORE / f"{base}_lstm_meta.json"
    MODEL_STORE.mkdir(parents=True, exist_ok=True)

    device = "cuda" if _TORCH_OK and torch.cuda.is_available() else "cpu"

    try:
        # ── Step 1: build labels ──────────────────────────────────────────────
        cont_labels = make_continuation_labels(df, lookahead=12)
        vol_labels  = make_vol_expansion_labels(df, lookahead=12)

        # ── Step 2: continuation model ───────────────────────────────────────
        feat_c = _resolve_features(df, CONT_FEATURES)
        X_c_raw, y_c = _build_sequences(feat_c, cont_labels, CONT_SEQ_LEN)

        if len(y_c) < 200 or y_c.sum() < 30 or (1 - y_c).sum() < 30:
            if verbose:
                print(f"   [LSTM] Not enough continuation labels for {symbol} "
                      f"(n={len(y_c)}, pos={int(y_c.sum())})")
            return False

        # Fit scaler on flattened training portion (80 %)
        split_c = int(len(y_c) * 0.80)
        scaler_c = _RobustScaler()
        scaler_c.fit(X_c_raw[:split_c].reshape(-1, X_c_raw.shape[2]))
        X_c = X_c_raw.copy()
        for i in range(len(X_c)):
            X_c[i] = scaler_c.transform(X_c[i])

        X_tr_c, X_vl_c = X_c[:split_c], X_c[split_c:]
        y_tr_c, y_vl_c = y_c[:split_c], y_c[split_c:]

        cont_model = ContinuationLSTM()
        cont_model, cont_val_loss = _train_model(
            cont_model, X_tr_c, y_tr_c, X_vl_c, y_vl_c,
            epochs=50, batch_size=256, lr=1e-3, patience=10, device=device,
        )
        cont_model.eval()
        with torch.no_grad():
            cont_val_preds = cont_model(
                torch.tensor(X_vl_c, dtype=torch.float32)
            ).numpy()
        cont_threshold = _best_threshold(cont_val_preds, y_vl_c)

        # Compute AUC on validation
        cont_auc = 0.5
        try:
            from sklearn.metrics import roc_auc_score
            cont_auc = float(roc_auc_score(y_vl_c, cont_val_preds))
        except Exception:
            pass

        # ── Step 3: vol-expansion model ──────────────────────────────────────
        feat_v = _resolve_features(df, VOL_EXP_FEATURES)
        X_v_raw, y_v = _build_sequences(feat_v, vol_labels, VOL_EXP_SEQ_LEN)

        vol_model:     Optional[Any]     = None
        vol_val_loss:  Optional[float]   = None
        vol_threshold: float             = 0.50
        vol_auc:       float             = 0.5
        scaler_v = _RobustScaler()

        if len(y_v) >= 200 and y_v.sum() >= 30 and (1 - y_v).sum() >= 30:
            split_v = int(len(y_v) * 0.80)
            scaler_v.fit(X_v_raw[:split_v].reshape(-1, X_v_raw.shape[2]))
            X_v = X_v_raw.copy()
            for i in range(len(X_v)):
                X_v[i] = scaler_v.transform(X_v[i])

            X_tr_v, X_vl_v = X_v[:split_v], X_v[split_v:]
            y_tr_v, y_vl_v = y_v[:split_v], y_v[split_v:]

            vol_model_inst = VolatilityExpansionLSTM()
            vol_model_inst, vol_val_loss = _train_model(
                vol_model_inst, X_tr_v, y_tr_v, X_vl_v, y_vl_v,
                epochs=50, batch_size=256, lr=1e-3, patience=10, device=device,
            )
            vol_model_inst.eval()
            with torch.no_grad():
                vol_val_preds = vol_model_inst(
                    torch.tensor(X_vl_v, dtype=torch.float32)
                ).numpy()
            vol_threshold = _best_threshold(vol_val_preds, y_vl_v)
            try:
                from sklearn.metrics import roc_auc_score
                vol_auc = float(roc_auc_score(y_vl_v, vol_val_preds))
            except Exception:
                pass
            vol_model = vol_model_inst
        else:
            if verbose:
                print(f"   [LSTM] Insufficient vol-expansion labels for {symbol} "
                      f"(n={len(y_v)}) — vol model skipped")

        # ── Step 4: save ──────────────────────────────────────────────────────
        torch.save(cont_model.state_dict(), str(cont_path))
        if vol_model is not None:
            torch.save(vol_model.state_dict(), str(vol_path))

        meta: Dict[str, Any] = {
            "symbol":            symbol,
            "cont_features":     CONT_FEATURES,
            "vol_features":      VOL_EXP_FEATURES,
            "cont_seq_len":      CONT_SEQ_LEN,
            "vol_seq_len":       VOL_EXP_SEQ_LEN,
            "cont_threshold":    float(cont_threshold),
            "vol_threshold":     float(vol_threshold),
            "cont_val_loss":     float(cont_val_loss),
            "cont_val_auc":      float(cont_auc),
            "vol_val_loss":      float(vol_val_loss) if vol_val_loss is not None else None,
            "vol_val_auc":       float(vol_auc),
            "vol_model_available": vol_model is not None,
            "cont_samples":      int(len(y_c)),
            "cont_pos_rate":     float(y_c.mean()),
            "vol_samples":       int(len(y_v)),
            "vol_pos_rate":      float(y_v.mean()) if len(y_v) > 0 else 0.0,
            "scaler_cont":       scaler_c.to_dict(),
            "scaler_vol":        scaler_v.to_dict() if vol_model is not None else None,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        if verbose:
            print(
                f"   [LSTM] {symbol}: "
                f"cont_auc={cont_auc:.3f} thr={cont_threshold:.2f} "
                f"| vol_auc={vol_auc:.3f} thr={vol_threshold:.2f}"
                + (" [vol skipped]" if vol_model is None else "")
            )
        return True

    except Exception as exc:
        if verbose:
            print(f"   [LSTM] Training failed for {symbol}: "
                  f"{type(exc).__name__}: {exc}")
        return False


# ── Fleet helpers ──────────────────────────────────────────────────────────────

def find_lstm_trained_symbols() -> Set[str]:
    """
    Return the set of symbols that already have a completed LSTM meta file.
    Used by train_lstm_fleet() to skip already-trained symbols on resume.
    """
    trained: Set[str] = set()
    for p in MODEL_STORE.glob("*_lstm_meta.json"):
        try:
            base = p.name.replace("_lstm_meta.json", "")
            # Convert e.g. BTC_USDT → BTC/USDT
            for quote in ("_USDT", "_BTC", "_ETH", "_BNB", "_BUSD"):
                if base.endswith(quote):
                    trained.add(base[: -len(quote)] + "/" + quote[1:])
                    break
        except Exception:
            pass
    return trained


def train_lstm_fleet(
    symbols:    List[str],
    df_getter,              # callable(symbol) → Optional[pd.DataFrame]
    resume:     bool = True,
    verbose:    bool = True,
) -> Dict[str, bool]:
    """
    Train LSTM models for a list of symbols.

    Parameters
    ----------
    symbols    : list of 'TICKER/USDT' strings
    df_getter  : callable that returns the feature DataFrame for a symbol
    resume     : if True, skip symbols that already have *_lstm_meta.json
    verbose    : print per-symbol progress

    Returns dict {symbol: success_bool}.
    """
    already_done = find_lstm_trained_symbols() if resume else set()
    results: Dict[str, bool] = {}

    for sym in symbols:
        if sym in already_done:
            if verbose:
                print(f"   [LSTM] {sym}: already trained — skipping")
            results[sym] = True
            continue

        df = df_getter(sym)
        ok = train_lstm_for_symbol(sym, df, verbose=verbose)
        results[sym] = ok

    return results
