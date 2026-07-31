"""
trader_engine.py — Universal Trader Live Inference Engine

Loads 3 calibrated models (scalping / intraday / swing) trained on 10 tokens,
runs inference on 60 deployment tokens, applies cooldown + confidence gating,
and returns structured signals with beginner guidance.

Called by main.py on a schedule or via the /api/trader/signals endpoint.
"""

import json
import logging
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ccxt
import joblib
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.trader_model.trader_config import (
    DEPLOYMENT_TOKENS, MODES, RISK_PROFILES, ALL_FEATURE_NAMES, STRATEGY_NAMES,
    TRADER_MODEL_STORE, TRADER_RECORD_PATH, ABSOLUTE_MIN_CONFIDENCE,
)
from scripts.trader_model.strategy_features import (
    compute_all_features, percentile_rank_features,
)
from scripts.trader_model.signal_manager import (
    generate_beginner_guidance, is_on_cooldown, record_signal,
)
from src.ml.adaptive import AdaptiveOrchestrator

log = logging.getLogger(__name__)

# Label mapping from calibrated model output (0=SELL, 1=HOLD, 2=BUY)
_LABEL_MAP = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}

# Minimum fraction of 25 strategies that must agree with the ML direction.
# 12/25 = 0.48 — require at least ~50% confluence to prevent taking counter-trend signals.
_MIN_CONFLUENCE = 0.28   # ≥ 7/25 strategies must agree (ranked percentile)


# ── Virtual Wallet ─────────────────────────────────────────────────────────────

class TraderWallet:
    """
    Paper-trades every signal fired by TraderEngine.

    Position key: f"{symbol}__{mode}"  — allows scalping + intraday open simultaneously.
    Writes closed trade outcomes to TRADER_RECORD_PATH.
    """

    INITIAL_CAPITAL = 10_000.0

    def __init__(self):
        self.balance:         float                   = self.INITIAL_CAPITAL
        self.open_positions:  Dict[str, Dict[str, Any]] = {}
        self.trade_history:   List[Dict[str, Any]]    = []
        self._lock = threading.Lock()
        self._load_history()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_history(self) -> None:
        if TRADER_RECORD_PATH.exists():
            try:
                with open(TRADER_RECORD_PATH) as f:
                    data = json.load(f)
                signals = data.get('signals', [])
                closed = [t for t in signals if t.get('outcome') in ('WIN', 'LOSS')]
                self.trade_history = closed
                for t in closed:
                    self.balance += float(t.get('pnl_usdt', 0) or 0)
                # Restore open positions so they survive restarts
                for t in signals:
                    if t.get('outcome') == 'OPEN':
                        key = f"{t.get('symbol', '')}__{t.get('mode', '')}"
                        if key not in self.open_positions:
                            self.open_positions[key] = t
            except Exception:
                pass

    def _save(self) -> None:
        TRADER_RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Merge open + closed into single list
        all_trades = self.trade_history + list(self.open_positions.values())
        won  = sum(1 for t in self.trade_history if t.get('outcome') == 'WIN')
        lost = sum(1 for t in self.trade_history if t.get('outcome') == 'LOSS')
        total = won + lost
        record = {
            'last_updated': datetime.now(timezone.utc).isoformat(),
            'balance':      round(self.balance, 2),
            'initial_capital': self.INITIAL_CAPITAL,
            'total_pnl_usdt': round(self.balance - self.INITIAL_CAPITAL, 2),
            'total_pnl_pct':  round((self.balance - self.INITIAL_CAPITAL) / self.INITIAL_CAPITAL * 100, 3),
            'total_trades': total,
            'won': won,
            'lost': lost,
            'win_rate': round(won / total, 3) if total else 0.0,
            'open_positions': len(self.open_positions),
            'signals': all_trades[-1000:],   # keep last 1 000
        }
        with open(TRADER_RECORD_PATH, 'w') as f:
            json.dump(record, f, indent=2, default=str)
        # Sync to web/ so Firebase hosting serves fresh data
        import shutil as _shutil
        _web = _ROOT / 'web' / 'trader_track_record.json'
        _web.parent.mkdir(parents=True, exist_ok=True)
        _shutil.copy2(TRADER_RECORD_PATH, _web)

    # ── Trade management ──────────────────────────────────────────────────────

    def open_trade(self, signal: Dict[str, Any]) -> None:
        key     = f"{signal['symbol']}__{signal['mode']}"
        profile = RISK_PROFILES.get(signal.get('risk_profile', 'balanced'), RISK_PROFILES['balanced'])
        pos_pct = profile['position_pct'] / 100.0
        pos_val = round(self.balance * pos_pct, 2)

        guidance = signal.get('guidance', {})
        tp       = guidance.get('take_profit', {})

        with self._lock:
            if key in self.open_positions:
                return   # already open for this symbol+mode
            self.open_positions[key] = {
                **{k: signal[k] for k in
                   ('signal_id', 'symbol', 'mode', 'direction', 'confidence',
                    'current_price', 'top_strategies', 'timeframe', 'timestamp')},
                'entry_price':    signal['current_price'],
                'stop_loss':      guidance.get('stop_loss', 0.0),
                'tp1':            tp.get('tp1', 0.0),
                'tp2':            tp.get('tp2', 0.0),
                'position_value': pos_val,
                'outcome':        'OPEN',
                'exit_price':     None,
                'exit_time':      None,
                'exit_reason':    None,
                'pnl_pct':        None,
                'pnl_usdt':       None,
                'risk_profile':   signal.get('risk_profile', 'balanced'),
            }
        log.info(f"[WALLET] OPEN {signal['direction']} {signal['symbol']} "
                 f"mode={signal['mode']} entry={signal['current_price']} "
                 f"SL={guidance.get('stop_loss')} TP1={tp.get('tp1')} size={pos_val:.0f}$")

    def check_exits(self, symbol: str, current_price: float) -> List[Dict[str, Any]]:
        """Check all open positions for `symbol` against current_price. Returns closed records."""
        closed = []
        with self._lock:
            to_close = {k: p for k, p in self.open_positions.items()
                        if p['symbol'] == symbol}
        for key, pos in to_close.items():
            result = self._evaluate_exit(pos, current_price)
            if result is None:
                continue
            with self._lock:
                self.open_positions.pop(key, None)
                self.balance += result['pnl_usdt']
                self.trade_history.append(result)
            closed.append(result)
            log.info(f"[WALLET] CLOSE {result['outcome']} {result['symbol']} "
                     f"mode={result['mode']} pnl={result['pnl_pct']:+.2f}% "
                     f"({result['exit_reason']})")
        if closed:
            self._save()
        return closed

    def _evaluate_exit(self, pos: Dict[str, Any], price: float) -> Optional[Dict[str, Any]]:
        """Return a closed trade dict if TP1 or SL is hit, else None."""
        direction = pos['direction']
        sl        = float(pos.get('stop_loss') or 0)
        tp1       = float(pos.get('tp1') or 0)
        entry     = float(pos['entry_price'])

        hit_tp = hit_sl = False
        if direction == 'BUY':
            if tp1 > 0 and price >= tp1:
                hit_tp = True
            elif sl > 0 and price <= sl:
                hit_sl = True
        else:  # SELL
            if tp1 > 0 and price <= tp1:
                hit_tp = True
            elif sl > 0 and price >= sl:
                hit_sl = True

        if not (hit_tp or hit_sl):
            return None

        exit_price  = tp1 if hit_tp else sl
        if direction == 'BUY':
            pnl_pct = (exit_price - entry) / entry * 100
        else:
            pnl_pct = (entry - exit_price) / entry * 100
        pnl_usdt = round(pos['position_value'] * pnl_pct / 100, 2)

        return {
            **pos,
            'exit_price':  round(exit_price, 8),
            'exit_time':   datetime.now(timezone.utc).isoformat(),
            'exit_reason': 'TP1_HIT' if hit_tp else 'SL_HIT',
            'pnl_pct':     round(pnl_pct, 3),
            'pnl_usdt':    pnl_usdt,
            'outcome':     'WIN' if hit_tp else 'LOSS',
        }

    @property
    def summary(self) -> Dict[str, Any]:
        won   = sum(1 for t in self.trade_history if t.get('outcome') == 'WIN')
        lost  = sum(1 for t in self.trade_history if t.get('outcome') == 'LOSS')
        total = won + lost
        pnl_u = round(self.balance - self.INITIAL_CAPITAL, 2)
        return {
            'balance':         round(self.balance, 2),
            'total_pnl_usdt':  pnl_u,
            'total_pnl_pct':   round(pnl_u / self.INITIAL_CAPITAL * 100, 3),
            'total_trades':    total,
            'won':             won,
            'lost':            lost,
            'win_rate':        round(won / total, 3) if total else 0.0,
            'open_positions':  len(self.open_positions),
        }


# ── Signal dataclass ───────────────────────────────────────────────────────────

@dataclass
class TraderSignal:
    signal_id:        str
    symbol:           str
    mode:             str          # scalping / intraday / swing
    risk_profile:     str          # conservative / balanced / aggressive
    direction:        str          # BUY / SELL
    status:           str          # PENDING / OPEN
    confidence:       float        # 0.0–1.0  (ML model probability for winning direction)
    confluence_score: float        # 0.0–1.0  (fraction of 25 strategies agreeing)
    current_price:    float
    top_strategies:   List[str]
    strategy_scores:  Dict[str, float]
    guidance:         Dict[str, Any]
    timestamp:        str
    timeframe:        str
    pending_since:    Optional[str] = None
    pending_count:    int = 0
    pending_reason:   str = ''
    p_buy:            float = 0.0  # independent BUY probability
    p_sell:           float = 0.0  # independent SELL probability
    p_hold:           float = 0.0  # residual HOLD probability


# ── Model store ────────────────────────────────────────────────────────────────

class TraderModelStore:
    """Loads and caches the calibrated trader models.

    Supports two formats:
    - Binary dual: {key}_model_buy.pkl + {key}_model_sell.pkl
      Run both classifiers; combine into [p_sell, p_hold, p_buy].
    - Legacy 3-class: {key}_model.pkl  (sklearn predict_proba → 3 classes)
    """

    def __init__(self):
        self._models:      Dict[str, Any] = {}   # mode → buy model (or 3-class)
        self._models_sell: Dict[str, Any] = {}   # mode → sell model (binary only)
        self._meta:        Dict[str, Any] = {}
        self._lock         = threading.Lock()

    def load_all(self) -> None:
        for mode_name in MODES:
            self.load(mode_name)

    def load(self, mode_name: str) -> bool:
        model_key = MODES.get(mode_name, {}).get('model_key', mode_name)
        meta_path = TRADER_MODEL_STORE / f"{model_key}_meta.json"

        # ── Prefer binary dual-model pair ─────────────────────────────────────
        buy_path  = TRADER_MODEL_STORE / f"{model_key}_model_buy.pkl"
        sell_path = TRADER_MODEL_STORE / f"{model_key}_model_sell.pkl"
        if buy_path.exists() and sell_path.exists():
            try:
                with self._lock:
                    self._models[mode_name]      = joblib.load(buy_path)
                    self._models_sell[mode_name] = joblib.load(sell_path)
                    if meta_path.exists():
                        with open(meta_path) as f:
                            self._meta[mode_name] = json.load(f)
                log.info(f"Loaded binary dual trader model: {mode_name} (buy+sell, weights: {model_key})")
                return True
            except Exception as e:
                log.error(f"Failed to load binary trader models {mode_name}: {e}")
                return False

        # ── Fall back to legacy 3-class single model ──────────────────────────
        model_path = TRADER_MODEL_STORE / f"{model_key}_model.pkl"
        if not model_path.exists():
            log.warning(f"Trader model not found: {model_path}")
            log.warning(f"Run: python -m scripts.trader_model.train_trader --mode {model_key}")
            return False
        try:
            with self._lock:
                self._models[mode_name] = joblib.load(model_path)
                if meta_path.exists():
                    with open(meta_path) as f:
                        self._meta[mode_name] = json.load(f)
            log.info(f"Loaded legacy 3-class trader model: {mode_name} (weights: {model_key})")
            return True
        except Exception as e:
            log.error(f"Failed to load trader model {mode_name}: {e}")
            return False

    def predict(self, mode_name: str, X: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """
        Returns (predicted_label, max_confidence, proba_array [p_sell, p_hold, p_buy]).
        predicted_label: 0=SELL, 1=HOLD, 2=BUY

        Binary dual mode: runs BUY and SELL classifiers independently, combines
        into a 3-class probability vector so the rest of the pipeline is unchanged.
        """
        with self._lock:
            model_buy  = self._models.get(mode_name)
            model_sell = self._models_sell.get(mode_name)

        if model_buy is None:
            return 1, 0.0, np.array([0.0, 1.0, 0.0])  # default HOLD

        if model_sell is not None:
            # Binary dual mode: independent BUY and SELL probabilities
            p_buy  = float(model_buy.predict_proba(X)[0, 1])
            p_sell = float(model_sell.predict_proba(X)[0, 1])
            p_hold = max(0.0, 1.0 - p_buy - p_sell)
            total  = p_buy + p_sell + p_hold
            proba  = np.array([p_sell / total, p_hold / total, p_buy / total])
        else:
            # Legacy 3-class model
            proba = model_buy.predict_proba(X)[0]

        label = int(np.argmax(proba))
        conf  = float(proba[label])
        return label, conf, proba

    @property
    def loaded_modes(self) -> List[str]:
        return list(self._models.keys())


# ── Exchange ───────────────────────────────────────────────────────────────────

class _ExchangePool:
    _spot: Optional[ccxt.Exchange] = None
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> ccxt.Exchange:
        if cls._spot is None:
            with cls._lock:
                if cls._spot is None:
                    cls._spot = ccxt.binance({'enableRateLimit': True, 'timeout': 10000})  # type: ignore[arg-type]
        assert cls._spot is not None
        return cls._spot


def _fetch_candles(symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
    try:
        ex  = _ExchangePool.get()
        raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not raw or len(raw) < 50:
            return None
        df = pd.DataFrame(raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        log.debug(f"Fetch failed {symbol} {timeframe}: {e}")
        return None



def _top_strategies(scores_row: pd.Series, direction: str, n: int = 3) -> List[str]:
    """Return the top N strategy names that most support the signal direction."""
    if direction == 'BUY':
        # Higher score (closer to 1.0) = more bullish support
        sorted_strats = scores_row.sort_values(ascending=False)
    else:
        # Lower score (closer to 0.0, originally -1.0) = more bearish support
        sorted_strats = scores_row.sort_values(ascending=True)
    return list(sorted_strats.index[:n])


# ── Main engine ────────────────────────────────────────────────────────────────

class TraderEngine:
    """
    Live inference engine for the Universal Trader Model.

    scan_all_tokens() runs one full cycle across 60 tokens × active modes.
    Results are cached in self._active_signals for the API to serve.
    """

    PENDING_CONFIRM_SCANS = 3
    PENDING_MAX_SECONDS = 60 * 30
    PENDING_CONF_DROP_ALLOWANCE = 0.05
    PENDING_MAX_CHASE_PCT = 0.35

    def __init__(self):
        self.model_store    = TraderModelStore()
        self.wallet         = TraderWallet()
        self.adaptive_orchestrator = AdaptiveOrchestrator()
        self._active_signals: List[Dict[str, Any]] = []
        self._token_status:   Dict[str, Dict[str, Any]] = {}   # all 60 tokens, every scan
        self._pending_entries: Dict[str, Dict[str, Any]] = {}
        self._last_scan_time: Optional[str] = None
        self._scan_lock     = threading.Lock()

    def load_models(self) -> None:
        self.model_store.load_all()
        if not self.model_store.loaded_modes:
            log.warning("No trader models loaded. Run train_trader.py first.")

    def _update_token_status(
        self,
        symbol:           str,
        mode:             str,
        direction:        str,
        confidence:       float,
        strategies:       List[str],
        timeframe:        str,
        on_cooldown:      bool,
        price:            float = 0.0,
        confluence_score: float = 0.0,
    ) -> None:
        """Record the latest scan result for a token so the dashboard can show all 60."""
        ts  = datetime.now(timezone.utc).isoformat()
        key = symbol
        with self._scan_lock:
            existing = self._token_status.get(key, {})
            # Keep the best (highest-confidence non-HOLD) signal across modes
            existing_conf = existing.get('confidence', 0.0)
            existing_dir  = existing.get('direction', 'HOLD')
            is_better = (
                direction != 'HOLD' and confidence > existing_conf
            ) or (
                existing_dir == 'HOLD' and direction == 'HOLD'
            )
            if is_better or existing.get('mode') == mode:
                self._token_status[key] = {
                    'symbol':           symbol,
                    'direction':        direction,
                    'confidence':       round(confidence, 4),
                    'confluence_score': round(confluence_score, 3),
                    'mode':             mode,
                    'timeframe':        timeframe,
                    'top_strategy':     strategies[0] if strategies else '',
                    'on_cooldown':      on_cooldown,
                    'price':            round(price, 8),
                    'last_scan':        ts,
                }

    @property
    def token_status(self) -> Dict[str, Dict[str, Any]]:
        with self._scan_lock:
            return dict(self._token_status)

    def _pending_key(self, symbol: str, mode: str) -> str:
        return f"{symbol}__{mode}"

    def _clear_pending(self, key: str) -> None:
        self._pending_entries.pop(key, None)

    def _should_execute_pending(self, pending: Dict[str, Any], signal: Dict[str, Any]) -> bool:
        if pending.get('direction') != signal.get('direction'):
            return False
        if pending.get('stable_scans', 0) < self.PENDING_CONFIRM_SCANS:
            return False
        if signal.get('confidence', 0.0) < pending.get('max_confidence', 0.0) - self.PENDING_CONF_DROP_ALLOWANCE:
            return False
        price = signal.get('current_price', 0.0)
        entry_price = pending.get('entry_price', price)
        if signal.get('direction') == 'BUY':
            return price <= entry_price * (1.0 + self.PENDING_MAX_CHASE_PCT / 100.0)
        return price >= entry_price * (1.0 - self.PENDING_MAX_CHASE_PCT / 100.0)

    def _execute_signal(self, sig_dict: Dict[str, Any]) -> None:
        key = self._pending_key(sig_dict['symbol'], sig_dict['mode'])
        try:
            self.adaptive_orchestrator.record_signal(sig_dict)
            sig_dict = self.adaptive_orchestrator.evaluate_signal(sig_dict)
        except Exception:
            pass
        self.wallet.open_trade(sig_dict)
        self.wallet._save()
        record_signal(sig_dict['symbol'], sig_dict['mode'])
        self._pending_entries.pop(key, None)

    def _create_pending(self, key: str, sig_dict: Dict[str, Any]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._pending_entries[key] = {
            'signal_id':     sig_dict['signal_id'],
            'symbol':        sig_dict['symbol'],
            'mode':          sig_dict['mode'],
            'direction':     sig_dict['direction'],
            'entry_price':   sig_dict['current_price'],
            'max_confidence': sig_dict['confidence'],
            'stable_scans':  1,
            'created_at':    time.time(),
            'updated_at':    time.time(),
            'pending_since': now,
            'pending_reason': 'awaiting confirmation',
            'signal':        sig_dict,
        }

    def _refresh_pending(self, key: str, sig_dict: Dict[str, Any]) -> None:
        pending = self._pending_entries.get(key)
        if not pending:
            self._create_pending(key, sig_dict)
            return
        pending['stable_scans'] = pending.get('stable_scans', 0) + 1
        pending['max_confidence'] = max(pending.get('max_confidence', 0.0), sig_dict.get('confidence', 0.0))
        pending['updated_at'] = time.time()
        pending['signal'] = sig_dict
        pending['pending_reason'] = 'awaiting repeated confirmation'

    def _expire_pending(self) -> None:
        now = time.time()
        for key, pending in list(self._pending_entries.items()):
            if now - pending.get('created_at', now) >= self.PENDING_MAX_SECONDS:
                self._pending_entries.pop(key, None)

    def scan_all_tokens(
        self,
        modes:        Optional[List[str]] = None,
        risk_profile: str = 'balanced',
        max_signals:  Optional[int] = None,
        force_fire:   bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Scan all deployment tokens for trade opportunities.

        Args:
            modes:        list of mode names to scan (default: all loaded modes)
            risk_profile: 'conservative' | 'balanced' | 'aggressive'
            max_signals:  cap on total signals returned

        Returns:
            List of signal dicts sorted by confidence descending.
        """
        if modes is None:
            modes = self.model_store.loaded_modes
        if not modes:
            log.warning("No models loaded — call load_models() first")
            return []

        profile    = RISK_PROFILES.get(risk_profile, RISK_PROFILES['balanced'])
        max_sig    = max_signals or profile['max_signals']
        min_conf   = profile['min_confidence']

        new_signals: List[Dict[str, Any]] = []

        for mode_name in modes:
            if mode_name not in self.model_store.loaded_modes:
                continue
            mode_cfg      = MODES[mode_name]
            tf            = mode_cfg['timeframe']
            limit         = mode_cfg['candles_fetch']
            # Per-mode override allows scalping (noisy 5m) to use a lower floor
            # than the profile default without affecting intraday/swing thresholds.
            effective_min_conf = max(
                mode_cfg.get('min_confidence_override', min_conf),
                ABSOLUTE_MIN_CONFIDENCE,
            )

            for symbol in DEPLOYMENT_TOKENS:
                on_cooldown = is_on_cooldown(symbol, mode_name)

                df = _fetch_candles(symbol, tf, limit)

                # Check exits for any open virtual positions on this symbol
                if df is not None and len(df) > 0:
                    current_px = float(df['close'].iloc[-1])
                    closed_trades = self.wallet.check_exits(symbol, current_px)
                    for closed in closed_trades:
                        try:
                            self.adaptive_orchestrator.record_trade(closed)
                        except Exception:
                            pass

                if df is None or len(df) < 100:
                    self._update_token_status(symbol, mode_name, 'HOLD', 0.0, [], tf, on_cooldown)
                    continue

                # ── Single-pass feature computation ───────────────────────────────
                try:
                    raw_features = compute_all_features(df)
                    ranked       = percentile_rank_features(raw_features, lookback=100)
                    last_ranked  = np.asarray(ranked[ALL_FEATURE_NAMES].iloc[-1], dtype=np.float32)
                    if np.isnan(last_ranked).any():
                        last_ranked = np.nan_to_num(last_ranked, nan=0.5)
                    X = last_ranked.reshape(1, -1)
                except Exception as _fe:
                    log.debug(f"Feature error {symbol}: {_fe}")
                    self._update_token_status(symbol, mode_name, 'HOLD', 0.0, [], tf, on_cooldown)
                    continue

                label, conf, proba = self.model_store.predict(mode_name, X)
                direction     = _LABEL_MAP.get(label, 'HOLD')
                current_price = float(df['close'].iloc[-1])
                p_sell, p_hold, p_buy = float(proba[0]), float(proba[1]), float(proba[2])

                # ── Scalping directional-edge override ────────────────────────────
                # The binary dual model often gives low absolute probabilities in
                # ranging markets (p_buy ≈ 0.05–0.15, p_hold ≈ 0.80), so the normal
                # argmax always picks HOLD. For scalping we fire when one side has a
                # clear edge over the other, even when HOLD dominates absolutely.
                _SCALP_EDGE_RATIO = 1.4   # one side must be ≥40 % stronger
                _SCALP_RAW_MIN    = 0.07  # raw probability floor (7 %)
                if mode_name in ('scalping', 'scalping_15m') and direction == 'HOLD':
                    if (p_buy >= _SCALP_RAW_MIN and p_buy >= p_sell * _SCALP_EDGE_RATIO):
                        direction = 'BUY'
                        conf      = p_buy
                        label     = 2
                    elif (p_sell >= _SCALP_RAW_MIN and p_sell >= p_buy * _SCALP_EDGE_RATIO):
                        direction = 'SELL'
                        conf      = p_sell
                        label     = 0

                # For scalping edge signals, effective_min_conf applies to the raw
                # directional probability instead of the normalised 3-class score.
                _scalp_edge = mode_name in ('scalping', 'scalping_15m') and conf in (p_buy, p_sell)
                _eff_min    = _SCALP_RAW_MIN if _scalp_edge else effective_min_conf

                # Top strategies from STRATEGY_NAMES only (ranked scores, 0-1 percentile)
                ranked_strat_row = ranked[STRATEGY_NAMES].iloc[-1]
                top_strats = _top_strategies(ranked_strat_row, direction if direction != 'HOLD' else 'BUY')

                # Confluence: count ranked strategy percentiles that agree with ML direction.
                if direction == 'BUY':
                    n_agree = int((ranked_strat_row > 0.55).sum())
                elif direction == 'SELL':
                    n_agree = int((ranked_strat_row < 0.45).sum())
                else:
                    n_agree = 0
                confluence_score = round(n_agree / len(STRATEGY_NAMES), 3)

                # All-feature ranked scores for the signal dict (45 features)
                scores_row = ranked[ALL_FEATURE_NAMES].iloc[-1]

                # Always update token status (before any signal gate)
                self._update_token_status(
                    symbol, mode_name, direction, conf, top_strats, tf, on_cooldown,
                    price=current_price, confluence_score=confluence_score,
                )

                # ── Signal gates ──────────────────────────────────────────────────
                if direction == 'HOLD':
                    continue
                # v82c: `force_fire` may ONLY waive the cooldown.  It used to wrap
                # this entire block, so one caller passing force_fire=True
                # (ScalpBot) silently disabled the confidence floor, confluence,
                # ATR floor, volume floor and RSI-extreme checks as well.  That is
                # what produced eight consecutive ZIL shorts into an 8.6 % rally at
                # 9-14 % confidence on 2026-07-23.  The quality gates below are now
                # unconditional and no flag can turn them off.
                if on_cooldown and not force_fire:
                    continue
                if conf < _eff_min:
                    continue
                if confluence_score < _MIN_CONFLUENCE:
                    log.debug(
                        f"[SKIP] {symbol} {mode_name} {direction} conf={conf:.2%} "
                        f"— confluence={n_agree}/25 ({confluence_score:.2f}) < {_MIN_CONFLUENCE}"
                    )
                    continue

                # ── Feature-engine parity gates ───────────────────────────────
                # Mirror the quality assumptions baked into training so live
                # signals arrive in the same market regime the models were
                # optimised for. raw_features uses raw (unnormalised) values.
                _fe_last = raw_features.iloc[-1]

                # ATR floor: training used ATR > 0.8% of price
                # fe_atr_pct is normalised as (atr/close).clip(0,0.1)/0.1
                # → raw 0.08 corresponds to 0.8% ATR
                _fe_atr = float(_fe_last.get('fe_atr_pct', 0.0))
                if _fe_atr < 0.08:
                    log.debug(
                        f"[SKIP] {symbol} {mode_name} {direction} "
                        f"— ATR_TOO_LOW fe_atr_pct={_fe_atr:.4f} (<0.08 = 0.8%)"
                    )
                    continue

                # Volume floor: skip when volume < 50% of rolling average
                # fe_vol_ratio = (vol/vol_ma).clip(0,5)/5 → raw 0.10 = 50%
                _fe_vol = float(_fe_last.get('fe_vol_ratio', 1.0))
                if _fe_vol < 0.10:
                    log.debug(
                        f"[SKIP] {symbol} {mode_name} {direction} "
                        f"— LOW_VOLUME fe_vol_ratio={_fe_vol:.4f} (<0.10 = 50% avg)"
                    )
                    continue

                # RSI extreme: avoid chasing overbought/oversold entries
                # fe_rsi_14 = rsi/100 → raw 0.75 = RSI 75, 0.25 = RSI 25
                _fe_rsi = float(_fe_last.get('fe_rsi_14', 0.5))
                if direction == 'BUY' and _fe_rsi > 0.75:
                    log.debug(
                        f"[SKIP] {symbol} {mode_name} BUY "
                        f"— RSI_OVERBOUGHT fe_rsi_14={_fe_rsi:.3f} (>0.75 = RSI 75)"
                    )
                    continue
                if direction == 'SELL' and _fe_rsi < 0.25:
                    log.debug(
                        f"[SKIP] {symbol} {mode_name} SELL "
                        f"— RSI_OVERSOLD fe_rsi_14={_fe_rsi:.3f} (<0.25 = RSI 25)"
                    )
                    continue

                # ── Counter-trend guard for SCALPING (v82c) ───────────────────
                # Scalping intentionally has no trend filter: its edge is the
                # directional-edge override, which is a RANGING-market tool.  But
                # nothing checked whether the market was actually ranging, so the
                # engine happily faded vertical trends — the ZIL sequence above is
                # exactly that failure.  This blocks counter-trend scalps ONLY
                # when the tape is decisively trending (ADX > 30), leaving the
                # ranging regime the mode is built for untouched.
                if mode_name in ('scalping', 'scalping_15m'):
                    _fe_adx = float(_fe_last.get('fe_adx_14', 0.0))
                    if _fe_adx > 0.30:
                        _slope = float(ranked.iloc[-1].get('fe_ema20_slope', 0.5))
                        if direction == 'SELL' and _slope > 0.70:
                            log.debug(
                                f"[SKIP] {symbol} {mode_name} SELL — SCALP_COUNTER_TREND "
                                f"adx={_fe_adx*100:.0f} slope_rank={_slope:.2f} (>0.70)"
                            )
                            continue
                        if direction == 'BUY' and _slope < 0.30:
                            log.debug(
                                f"[SKIP] {symbol} {mode_name} BUY — SCALP_COUNTER_TREND "
                                f"adx={_fe_adx*100:.0f} slope_rank={_slope:.2f} (<0.30)"
                            )
                            continue

                # Trend alignment + ADX regime: only for intraday/swing modes.
                if mode_name in ('intraday', 'swing'):
                    # ADX < 20 → no directional trend; skip intraday/swing signals
                    # fe_adx_14 = adx/100 → raw 0.20 = ADX 20
                    _fe_adx = float(_fe_last.get('fe_adx_14', 0.5))
                    if _fe_adx < 0.20:
                        log.debug(
                            f"[SKIP] {symbol} {mode_name} {direction} "
                            f"— RANGING_MARKET fe_adx_14={_fe_adx:.3f} (<0.20 = ADX 20)"
                        )
                        continue

                    # Counter-trend: block entries against strong trend
                    # fe_ema20_slope is already ranked (percentile 0-1)
                    _fe_slope = float(ranked.iloc[-1].get('fe_ema20_slope', 0.5))
                    if direction == 'BUY' and _fe_slope < 0.30:
                        log.debug(
                            f"[SKIP] {symbol} {mode_name} BUY "
                            f"— COUNTER_TREND fe_ema20_slope_rank={_fe_slope:.3f} (<0.30)"
                        )
                        continue
                    if direction == 'SELL' and _fe_slope > 0.70:
                        log.debug(
                            f"[SKIP] {symbol} {mode_name} SELL "
                            f"— COUNTER_TREND fe_ema20_slope_rank={_fe_slope:.3f} (>0.70)"
                        )
                        continue

                # Beginner guidance (only for gated signals)
                guidance = generate_beginner_guidance(
                    symbol         = symbol,
                    direction      = direction,
                    mode           = mode_name,
                    risk_profile   = risk_profile,
                    confidence     = conf,
                    current_price  = current_price,
                    top_strategies = top_strats,
                    df             = df,
                )

                sig_id = str(uuid.uuid4())[:8].upper()
                ts     = datetime.now(timezone.utc).isoformat()

                signal = TraderSignal(
                    signal_id        = sig_id,
                    symbol           = symbol,
                    mode             = mode_name,
                    risk_profile     = risk_profile,
                    direction        = direction,
                    confidence       = round(conf, 4),
                    confluence_score = confluence_score,
                    current_price    = current_price,
                    top_strategies   = top_strats,
                    strategy_scores  = {str(k): round(float(v), 3) for k, v in scores_row.items()},
                    guidance         = guidance,
                    timestamp        = ts,
                    timeframe        = tf,
                    p_buy            = round(p_buy,  4),
                    p_sell           = round(p_sell, 4),
                    p_hold           = round(p_hold, 4),
                )

                sig_dict = asdict(signal)
                sig_key = self._pending_key(symbol, mode_name)

                if sig_key in self.wallet.open_positions:
                    sig_dict['status'] = 'OPEN'
                    sig_dict['pending_entry'] = False
                    new_signals.append(sig_dict)
                    continue

                pending = self._pending_entries.get(sig_key)
                if pending is not None and pending.get('direction') != direction:
                    # New direction invalidates the existing pending arm.
                    self._clear_pending(sig_key)
                    pending = None

                if pending is not None:
                    if self._should_execute_pending(pending, sig_dict):
                        sig_dict['status'] = 'OPEN'
                        sig_dict['pending_entry'] = False
                        self._execute_signal(sig_dict)
                        new_signals.append(sig_dict)
                        try:
                            from scripts.notifications.dispatcher import get_notifier
                            get_notifier().send_entry(sig_dict)
                        except Exception:
                            pass
                        log.info(
                            f"[TRADER] EXECUTE {direction} {symbol} | {mode_name} | "
                            f"conf={conf:.2%} | confl={n_agree}/25 | {top_strats[0] if top_strats else '?'}"
                        )
                        continue
                    self._refresh_pending(sig_key, sig_dict)
                    pending = self._pending_entries[sig_key]
                    sig_dict['status'] = 'PENDING'
                    sig_dict['pending_entry'] = True
                    sig_dict['pending_since'] = pending.get('pending_since')
                    sig_dict['pending_count'] = pending.get('stable_scans', 0)
                    sig_dict['pending_reason'] = pending.get('pending_reason', '')
                    new_signals.append(sig_dict)
                    continue

                self._create_pending(sig_key, sig_dict)
                pending = self._pending_entries[sig_key]
                sig_dict['status'] = 'PENDING'
                sig_dict['pending_entry'] = True
                sig_dict['pending_since'] = pending.get('pending_since')
                sig_dict['pending_count'] = pending.get('stable_scans', 0)
                sig_dict['pending_reason'] = pending.get('pending_reason', '')
                new_signals.append(sig_dict)

        # Sort by confidence descending and cap
        self._expire_pending()

        # Preserve any still-active pending arms that weren't refreshed by the current scan.
        existing_pending_ids = {s.get('signal_id') for s in new_signals if s.get('status') == 'PENDING'}
        for pending in self._pending_entries.values():
            if pending.get('signal', {}).get('signal_id') not in existing_pending_ids:
                pending_signal = dict(pending.get('signal', {}))
                pending_signal['status'] = 'PENDING'
                pending_signal['pending_entry'] = True
                pending_signal['pending_since'] = pending.get('pending_since')
                pending_signal['pending_count'] = pending.get('stable_scans', 0)
                pending_signal['pending_reason'] = pending.get('pending_reason', '')
                new_signals.append(pending_signal)

        # Sort by confidence descending and cap open/active signals, but keep pending
        pending_signals = [s for s in new_signals if s.get('status') == 'PENDING']
        active_signals = [s for s in new_signals if s.get('status') != 'PENDING']
        active_signals.sort(key=lambda s: s['confidence'], reverse=True)
        active_signals = active_signals[:max_sig]
        new_signals = active_signals + pending_signals

        with self._scan_lock:
            self._active_signals = new_signals
            self._last_scan_time = datetime.now(timezone.utc).isoformat()

        return new_signals

    # ── Live position monitor ─────────────────────────────────────────────────

    def start_live_monitor(self, interval_seconds: int = 10) -> None:
        """
        Start a background thread that polls live prices for open positions
        and evaluates SL/TP exits every `interval_seconds`.

        Expired trades are appended to self.expired_signals (newest first).
        Calling this more than once is safe — only one monitor thread runs.
        """
        if getattr(self, '_monitor_thread', None) and self._monitor_thread.is_alive():
            return

        self.expired_signals: List[Dict[str, Any]] = []
        self._monitor_interval = interval_seconds

        def _loop() -> None:
            while True:
                try:
                    open_syms = list(self.wallet.open_positions.keys())
                    symbols   = list({k.split('__')[0] for k in open_syms})
                    if symbols:
                        prices = self._fetch_live_prices(symbols)
                        with self._scan_lock:
                            self._live_prices.update(prices)
                        for sym, price in prices.items():
                            closed = self.wallet.check_exits(sym, price)
                            if closed:
                                with self._scan_lock:
                                    for c in closed:
                                        c.setdefault('expired_at',
                                                     datetime.now(timezone.utc).isoformat())
                                        self.expired_signals.insert(0, c)
                                    self.expired_signals = self.expired_signals[:100]
                                # Fire exit notifications (outside lock, best-effort)
                                try:
                                    from scripts.notifications.dispatcher import get_notifier
                                    _notif = get_notifier()
                                    for c in closed:
                                        _entry_ts = c.get('entry_time', '')
                                        _hold = 0
                                        if _entry_ts:
                                            try:
                                                _t = datetime.fromisoformat(
                                                    _entry_ts.replace('Z', '+00:00')
                                                )
                                                _hold = int(
                                                    (datetime.now(timezone.utc) - _t).total_seconds()
                                                )
                                            except Exception:
                                                pass
                                        _notif.send_exit(
                                            symbol=sym,
                                            direction=c.get('direction', c.get('side', '')),
                                            outcome=c.get('outcome', '?'),
                                            pnl_pct=float(c.get('pnl_pct', 0) or 0),
                                            hold_seconds=_hold,
                                            exit_reason=c.get('exit_reason', ''),
                                        )
                                except Exception:
                                    pass
                except Exception as exc:
                    log.debug(f"[monitor] {exc}")
                time.sleep(self._monitor_interval)

        self._live_prices: Dict[str, float] = {}
        self._monitor_thread = threading.Thread(target=_loop, daemon=True, name='trader-monitor')
        self._monitor_thread.start()
        log.info(f"[TraderEngine] Live position monitor started (interval={interval_seconds}s)")

    @staticmethod
    def _fetch_live_prices(symbols: List[str]) -> Dict[str, float]:
        """Fetch last price for a list of symbols via Binance spot ticker."""
        out: Dict[str, float] = {}
        ex = _ExchangePool.get()
        for sym in symbols:
            try:
                t = ex.fetch_ticker(sym)
                p = float(t.get('last') or t.get('close') or 0)
                if p > 0:
                    out[sym] = p
            except Exception:
                pass
        return out

    @property
    def active_signals(self) -> List[Dict[str, Any]]:
        with self._scan_lock:
            return list(self._active_signals)

    @property
    def last_scan_time(self) -> Optional[str]:
        with self._scan_lock:
            return self._last_scan_time


# ── Module-level singleton ─────────────────────────────────────────────────────
_trader_engine: Optional[TraderEngine] = None
_engine_lock   = threading.Lock()


def get_trader_engine() -> TraderEngine:
    """Return (or lazily create) the module-level TraderEngine singleton."""
    global _trader_engine
    if _trader_engine is None:
        with _engine_lock:
            if _trader_engine is None:
                _trader_engine = TraderEngine()
                _trader_engine.load_models()
    assert _trader_engine is not None
    return _trader_engine


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='AEGIS Trader Engine — live scan all 60 tokens')
    parser.add_argument('--mode', choices=['scalping', 'intraday', 'swing', 'all'], default='all')
    parser.add_argument('--risk', choices=['conservative', 'balanced', 'aggressive'], default='balanced')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    engine = get_trader_engine()
    if not engine.model_store.loaded_modes:
        print("\n  No trained models found.")
        print("  Train first:  python -m scripts.trader_model.train_trader\n")
    else:
        modes = list(MODES.keys()) if args.mode == 'all' else [args.mode]
        print(f"\nScanning ALL {len(DEPLOYMENT_TOKENS)} tokens × {modes} | risk={args.risk}\n")

        engine.scan_all_tokens(modes=modes, risk_profile=args.risk)

        status = engine.token_status
        fired  = engine.active_signals

        # ── Full 60-token status table ──────────────────────────────────────
        col = {'BUY': '\033[92m', 'SELL': '\033[91m', 'HOLD': '\033[90m', 'RST': '\033[0m'}
        W = col['RST']
        thr_pct = {'conservative': 75, 'balanced': 70, 'aggressive': 65}.get(args.risk, 65)
        print(f"\n{'─'*88}")
        print(f"  {'TOKEN':<14} {'DIR':<6} {'CONF':>6}  {'CONFL':>7}  {'MODE':<10} {'TOP STRATEGY':<28} {'CD'}")
        print(f"{'─'*88}")
        rows = sorted(
            status.values(),
            key=lambda x: (0 if x['direction'] != 'HOLD' else 1, -x.get('confidence', 0))
        )
        for t in rows:
            d      = t['direction']
            c      = col.get(d, '')
            pct    = f"{t.get('confidence', 0) * 100:5.1f}%"
            n_ag   = round(t.get('confluence_score', 0) * len(STRATEGY_NAMES))
            confl  = f"{n_ag:2d}/25"
            cd     = ' CD' if t.get('on_cooldown') else ''
            strat  = (t.get('top_strategy') or '').replace('_', ' ')[:26]
            print(f"  {c}{t['symbol']:<14} {d:<6}{W} {pct}  {confl}  {t.get('mode','?'):<10} {strat:<28}{cd}")
        print(f"{'─'*88}")
        print(f"\n  Scanned: {len(status)}/60 tokens   "
              f"Fired: {len(fired)} signal(s)  "
              f"(ML ≥ {thr_pct}% AND confluence ≥ {int(_MIN_CONFLUENCE * len(STRATEGY_NAMES))}/25)\n")

        if fired:
            print(f"  {'─'*56}")
            print(f"  ACTIVE SIGNALS ({len(fired)}):")
            print(f"  {'─'*56}")
            for s in fired:
                g  = s.get('guidance', {})
                tp = g.get('take_profit', {})
                n  = round(s.get('confluence_score', 0) * len(STRATEGY_NAMES))
                print(f"  [{s['mode'].upper():8}] {col.get(s['direction'],'')+s['direction']+W:<4}  "
                      f"{s['symbol']:<12} conf={s['confidence']:.0%}  "
                      f"confl={n}/25  "
                      f"{s['top_strategies'][0].replace('_',' ') if s['top_strategies'] else '?'}")
                print(f"             SL={g.get('stop_loss')}  "
                      f"TP1={tp.get('tp1')}  TP2={tp.get('tp2')}  R:R=1:{g.get('risk_reward','?')}\n")
