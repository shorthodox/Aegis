"""
trader_engine.py — Universal Trader Live Inference Engine

Loads 3 calibrated models (scalping / intraday / swing) trained on 10 tokens,
runs inference on 60 deployment tokens, applies cooldown + confidence gating,
and returns structured signals with beginner guidance.

Called by main.py on a schedule or via the /api/trader/signals endpoint.
"""

import json
import logging
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys

import ccxt
import joblib
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.trader_model.trader_config import (
    DEPLOYMENT_TOKENS, MODES, RISK_PROFILES, ALL_FEATURE_NAMES,
    TRADER_MODEL_STORE, TRADER_RECORD_PATH, ABSOLUTE_MIN_CONFIDENCE,
)
from scripts.trader_model.strategy_features import (
    compute_all_features, percentile_rank_features,
)
from scripts.trader_model.signal_manager import (
    generate_beginner_guidance, is_on_cooldown, record_signal,
)

log = logging.getLogger(__name__)

# Label mapping from calibrated model output (0=SELL, 1=HOLD, 2=BUY)
_LABEL_MAP = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}


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
                closed = [t for t in data.get('signals', []) if t.get('outcome') in ('WIN', 'LOSS')]
                self.trade_history = closed
                # Rebuild balance from history
                for t in closed:
                    self.balance += float(t.get('pnl_usdt', 0) or 0)
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
    signal_id:      str
    symbol:         str
    mode:           str          # scalping / intraday / swing
    risk_profile:   str          # conservative / balanced / aggressive
    direction:      str          # BUY / SELL
    confidence:     float        # 0.0–1.0
    current_price:  float
    top_strategies: List[str]
    strategy_scores: Dict[str, float]
    guidance:       Dict[str, Any]
    timestamp:      str
    timeframe:      str


# ── Model store ────────────────────────────────────────────────────────────────

class TraderModelStore:
    """Loads and caches the 3 calibrated trader models."""

    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._meta:   Dict[str, Any] = {}
        self._lock    = threading.Lock()

    def load_all(self) -> None:
        for mode_name in MODES:
            self.load(mode_name)

    def load(self, mode_name: str) -> bool:
        model_path = TRADER_MODEL_STORE / f"{mode_name}_model.pkl"
        meta_path  = TRADER_MODEL_STORE / f"{mode_name}_meta.json"
        if not model_path.exists():
            log.warning(f"Trader model not found: {model_path}")
            log.warning(f"Run: python -m scripts.trader_model.train_trader --mode {mode_name}")
            return False
        try:
            with self._lock:
                self._models[mode_name] = joblib.load(model_path)
                if meta_path.exists():
                    with open(meta_path) as f:
                        self._meta[mode_name] = json.load(f)
            log.info(f"Loaded trader model: {mode_name}")
            return True
        except Exception as e:
            log.error(f"Failed to load trader model {mode_name}: {e}")
            return False

    def predict(self, mode_name: str, X: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """
        Returns (predicted_label, max_confidence, proba_array).
        predicted_label: 0=SELL, 1=HOLD, 2=BUY
        """
        with self._lock:
            model = self._models.get(mode_name)
        if model is None:
            return 1, 0.0, np.array([0.0, 1.0, 0.0])  # default HOLD

        proba = model.predict_proba(X)[0]
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


# ── Feature builder for a single bar ─────────────────────────────────────────

def _build_inference_features(df: pd.DataFrame) -> Optional[np.ndarray]:
    """Compute normalized strategy features for the last row of df."""
    try:
        raw      = compute_all_features(df)
        ranked   = percentile_rank_features(raw, lookback=100)
        last_row = np.asarray(ranked[ALL_FEATURE_NAMES].iloc[-1], dtype=np.float32)
        if np.isnan(last_row).any():
            last_row = np.nan_to_num(last_row, nan=0.5)
        return last_row.reshape(1, -1)
    except Exception as e:
        log.debug(f"Feature build error: {e}")
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

    def __init__(self):
        self.model_store    = TraderModelStore()
        self.wallet         = TraderWallet()
        self._active_signals: List[Dict[str, Any]] = []
        self._token_status:   Dict[str, Dict[str, Any]] = {}   # all 60 tokens, every scan
        self._last_scan_time: Optional[str] = None
        self._scan_lock     = threading.Lock()

    def load_models(self) -> None:
        self.model_store.load_all()
        if not self.model_store.loaded_modes:
            log.warning("No trader models loaded. Run train_trader.py first.")

    def _update_token_status(
        self,
        symbol:      str,
        mode:        str,
        direction:   str,
        confidence:  float,
        strategies:  List[str],
        timeframe:   str,
        on_cooldown: bool,
        price:       float = 0.0,
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
                    'symbol':       symbol,
                    'direction':    direction,
                    'confidence':   round(confidence, 4),
                    'mode':         mode,
                    'timeframe':    timeframe,
                    'top_strategy': strategies[0] if strategies else '',
                    'on_cooldown':  on_cooldown,
                    'price':        round(price, 8),
                    'last_scan':    ts,
                }

    @property
    def token_status(self) -> Dict[str, Dict[str, Any]]:
        with self._scan_lock:
            return dict(self._token_status)

    def scan_all_tokens(
        self,
        modes:        Optional[List[str]] = None,
        risk_profile: str = 'balanced',
        max_signals:  Optional[int] = None,
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
            mode_cfg = MODES[mode_name]
            tf       = mode_cfg['timeframe']
            limit    = mode_cfg['candles_fetch']

            for symbol in DEPLOYMENT_TOKENS:
                on_cooldown = is_on_cooldown(symbol, mode_name)

                df = _fetch_candles(symbol, tf, limit)

                # Check exits for any open virtual positions on this symbol
                if df is not None and len(df) > 0:
                    current_px = float(df['close'].iloc[-1])
                    self.wallet.check_exits(symbol, current_px)

                if df is None or len(df) < 100:
                    self._update_token_status(symbol, mode_name, 'HOLD', 0.0, [], tf, on_cooldown)
                    continue

                X = _build_inference_features(df)
                if X is None:
                    self._update_token_status(symbol, mode_name, 'HOLD', 0.0, [], tf, on_cooldown)
                    continue

                label, conf, _ = self.model_store.predict(mode_name, X)
                direction = _LABEL_MAP.get(label, 'HOLD')
                current_price = float(df['close'].iloc[-1])

                # Strategy scores for status tracking (always)
                raw_scores = compute_all_features(df)
                ranked     = percentile_rank_features(raw_scores, lookback=100)
                scores_row = ranked[ALL_FEATURE_NAMES].iloc[-1]
                top_strats = _top_strategies(scores_row, direction if direction != 'HOLD' else 'BUY')

                # Always update token status (HOLD, BUY, or SELL — before confidence gate)
                self._update_token_status(
                    symbol, mode_name, direction, conf, top_strats, tf, on_cooldown,
                    price=current_price,
                )

                # Skip signal creation if on cooldown or below threshold
                if on_cooldown:
                    continue
                if direction == 'HOLD':
                    continue
                if conf < max(min_conf, ABSOLUTE_MIN_CONFIDENCE):
                    continue

                # Beginner guidance (only for gated signals)
                guidance = generate_beginner_guidance(
                    symbol        = symbol,
                    direction     = direction,
                    mode          = mode_name,
                    risk_profile  = risk_profile,
                    confidence    = conf,
                    current_price = current_price,
                    top_strategies = top_strats,
                    df             = df,
                )

                sig_id = str(uuid.uuid4())[:8].upper()
                ts     = datetime.now(timezone.utc).isoformat()

                signal = TraderSignal(
                    signal_id       = sig_id,
                    symbol          = symbol,
                    mode            = mode_name,
                    risk_profile    = risk_profile,
                    direction       = direction,
                    confidence      = round(conf, 4),
                    current_price   = current_price,
                    top_strategies  = top_strats,
                    strategy_scores = {str(k): round(float(v), 3) for k, v in scores_row.items()},
                    guidance        = guidance,
                    timestamp       = ts,
                    timeframe       = tf,
                )

                sig_dict = asdict(signal)
                new_signals.append(sig_dict)

                # Open virtual position and save to track record
                self.wallet.open_trade(sig_dict)
                self.wallet._save()

                record_signal(symbol, mode_name)

                log.info(
                    f"[TRADER] {direction} {symbol} | {mode_name} | "
                    f"conf={conf:.2%} | {top_strats[0] if top_strats else '?'}"
                )

        # Sort by confidence descending and cap
        new_signals.sort(key=lambda s: s['confidence'], reverse=True)
        new_signals = new_signals[:max_sig]

        with self._scan_lock:
            self._active_signals = new_signals
            self._last_scan_time = datetime.now(timezone.utc).isoformat()

        return new_signals

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
        print(f"\n{'─'*80}")
        print(f"  {'TOKEN':<14} {'DIR':<6} {'CONF':>6}  {'MODE':<10} {'TOP STRATEGY':<30} {'CD'}")
        print(f"{'─'*80}")
        # sort: BUY/SELL by confidence desc, then HOLD
        rows = sorted(
            status.values(),
            key=lambda x: (0 if x['direction'] != 'HOLD' else 1, -x.get('confidence', 0))
        )
        for t in rows:
            d   = t['direction']
            c   = col.get(d, '')
            pct = f"{t.get('confidence',0)*100:5.1f}%"
            cd  = ' CD' if t.get('on_cooldown') else ''
            strat = (t.get('top_strategy') or '').replace('_', ' ')[:28]
            print(f"  {c}{t['symbol']:<14} {d:<6}{W} {pct}  {t.get('mode','?'):<10} {strat:<30}{cd}")
        print(f"{'─'*80}")
        print(f"\n  Scanned: {len(status)}/60 tokens   "
              f"Fired: {len(fired)} signal(s)  "
              f"(threshold ≥ {int(args.risk == 'balanced' and 70 or args.risk == 'conservative' and 75 or 65)}%)\n")

        if fired:
            print(f"  {'─'*50}")
            print(f"  ACTIVE SIGNALS ({len(fired)}):")
            print(f"  {'─'*50}")
            for s in fired:
                g = s.get('guidance', {})
                tp = g.get('take_profit', {})
                print(f"  [{s['mode'].upper():8}] {col.get(s['direction'],'')+s['direction']+W:4}  "
                      f"{s['symbol']:<12} conf={s['confidence']:.0%}  "
                      f"{s['top_strategies'][0].replace('_',' ') if s['top_strategies'] else '?'}")
                print(f"             SL={g.get('stop_loss')}  "
                      f"TP1={tp.get('tp1')}  TP2={tp.get('tp2')}  R:R=1:{g.get('risk_reward','?')}\n")
