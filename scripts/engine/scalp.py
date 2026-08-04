"""The scalping book — a separate, smaller-timeframe strategy.

Kept apart from LiveEngine deliberately: it has its own record, its own
cooldowns and its own risk. It must never pass force_fire=True into the
trader engine — that single argument disabled the confidence floor,
confluence, ATR, volume and RSI gates at once, and
scripts/tests/test_scalp_guards.py asserts it has not come back.

Extracted verbatim from the single-file live_engine.py.
"""
from __future__ import annotations

from datetime import datetime
from datetime import timezone
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
import json
import time
import uuid

from scripts.engine.config import ROOT as _ROOT
from scripts.engine.config import SCALP_RECORD_PATH as _SCALP_RECORD_PATH

class ScalpBot:
    """
    Lightweight scalping bot that fires on raw model prediction (no gates).
    Uses the existing TraderEngine models (5m scalping + 15m via same weights).
    Runs every 60 seconds independently from the main AEGIS scan cycle.
    """

    SCAN_INTERVAL = 60  # seconds
    POSITION_USDT = 100.0  # fixed notional per trade

    def __init__(self) -> None:
        self._open: Dict[str, Dict[str, Any]] = {}   # key = "SYM_mode"
        self._history: List[Dict[str, Any]] = []
        self._engine: Optional[Any] = None
        self._last_scan: float = 0.0
        self._firestore_db: Optional[Any] = None
        self._load_record()

    # ── persistence ──────────────────────────────────────────────────────────

    def _load_record(self) -> None:
        if _SCALP_RECORD_PATH.exists():
            try:
                data = json.loads(_SCALP_RECORD_PATH.read_text())
                self._history = [t for t in data.get('trades', [])
                                 if t.get('outcome') in ('WIN', 'LOSS')]
            except Exception:
                pass

    def _save_record(self) -> None:
        all_trades = self._history + list(self._open.values())
        won  = sum(1 for t in self._history if t.get('outcome') == 'WIN')
        lost = sum(1 for t in self._history if t.get('outcome') == 'LOSS')
        payload = {
            'last_updated': datetime.now(timezone.utc).isoformat(),
            'total_trades': won + lost,
            'won': won, 'lost': lost,
            'win_rate': round(won / (won + lost), 3) if (won + lost) else 0.0,
            'trades': all_trades,
        }
        import os as _os
        tmp = str(_SCALP_RECORD_PATH) + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(payload, f, indent=2, default=str)
        _os.replace(tmp, _SCALP_RECORD_PATH)

    # ── firebase ──────────────────────────────────────────────────────────────

    def _get_db(self) -> Optional[Any]:
        if self._firestore_db is not None:
            return self._firestore_db
        try:
            import firebase_admin
            from firebase_admin import credentials as _creds, firestore as _fs
            cred_path = _ROOT / 'config' / 'serviceAccountKey.json'
            if not cred_path.exists():
                return None
            if not firebase_admin._apps:
                firebase_admin.initialize_app(_creds.Certificate(str(cred_path)))
            self._firestore_db = _fs.client()
            return self._firestore_db
        except Exception:
            return None

    def _push_signal(self, sig: Dict[str, Any]) -> None:
        db = self._get_db()
        if db is None:
            return
        try:
            doc_id = sig['symbol'].replace('/', '_') + '_' + sig['mode']
            db.collection('scalp_signals').document(doc_id).set(
                {k: v for k, v in sig.items() if v is not None}
            )
        except Exception:
            pass

    # ── scan ──────────────────────────────────────────────────────────────────

    def _load_engine(self) -> bool:
        if self._engine is not None:
            return True
        try:
            from scripts.trader_model.trader_engine import TraderEngine
            self._engine = TraderEngine()
            self._engine.load_models()
            if not self._engine.model_store.loaded_modes:
                self._engine = None
                return False
            print('[ScalpBot] Trader models loaded:', self._engine.model_store.loaded_modes)
            return True
        except Exception as e:
            print(f'[ScalpBot] Failed to load trader engine: {e}')
            return False

    def _check_exits(self, live_prices: Dict[str, float]) -> None:
        closed_keys = []
        for key, pos in self._open.items():
            sym   = pos['symbol']
            price = live_prices.get(sym, 0.0)
            if price <= 0:
                continue
            entry = pos['entry_price']
            sl    = pos['sl']
            tp    = pos['tp']
            direction = pos['direction']

            hit_sl = (direction == 'BUY' and price <= sl) or (direction == 'SELL' and price >= sl)
            hit_tp = (direction == 'BUY' and price >= tp) or (direction == 'SELL' and price <= tp)

            if hit_sl or hit_tp:
                pnl_pct = ((price - entry) / entry * 100) if direction == 'BUY' \
                          else ((entry - price) / entry * 100)
                outcome = 'WIN' if hit_tp else 'LOSS'
                closed = {**pos,
                          'exit_price': round(price, 8),
                          'exit_time':  datetime.now(timezone.utc).isoformat(),
                          'pnl_pct':    round(pnl_pct, 3),
                          'pnl_usdt':   round(self.POSITION_USDT * pnl_pct / 100, 2),
                          'outcome':    outcome,
                          'exit_reason': 'TP' if hit_tp else 'SL'}
                self._history.append(closed)
                closed_keys.append(key)
                print(f'[ScalpBot] {outcome} {direction} {sym} {pnl_pct:+.2f}% ({closed["exit_reason"]})')
                # v82c: feed the outcome back into the cooldown tracker so a
                # losing symbol cools off progressively instead of re-arming on
                # the same fixed clock it would use after a win.
                try:
                    from scripts.trader_model.signal_manager import (
                        record_loss, record_win, LOSS_STREAK_BLOCK,
                    )
                    _mode = pos.get('mode', 'scalping')
                    if outcome == 'LOSS':
                        _streak = record_loss(sym, _mode)
                        if _streak >= LOSS_STREAK_BLOCK:
                            print(f'[ScalpBot] LOSS_STREAK {sym} {_mode}: '
                                  f'{_streak} in a row — blocked')
                    else:
                        record_win(sym, _mode)
                except Exception as _e:
                    print(f'[ScalpBot] cooldown update failed for {sym}: {_e}')
        for k in closed_keys:
            del self._open[k]
        if closed_keys:
            self._save_record()

    def scan(self, live_prices: Dict[str, float]) -> None:
        """Run one scalp scan cycle. Call from async loop via run_in_executor."""
        now = time.time()
        if now - self._last_scan < self.SCAN_INTERVAL:
            return
        self._last_scan = now

        self._check_exits(live_prices)

        if not self._load_engine():
            return
        engine = self._engine
        if engine is None:
            return

        try:
            # v82c: force_fire was True here, which disabled EVERY safety gate in
            # the trader engine at once — cooldown, confidence floor, confluence,
            # ATR floor, volume floor and the RSI-extreme check.  The scalp record
            # shows what that produced: on 2026-07-23 ZIL/USDT was shorted eight
            # consecutive times in nine minutes at 9-14 % model confidence, each
            # entry higher than the last, all eight stopped out, while the token
            # rallied 8.6 %.  Re-entry gaps got as short as 32 seconds against a
            # nominal 10/15-minute cooldown.  17 of the 18 recorded losses were
            # shorts taken this way.
            signals = engine.scan_all_tokens(
                modes=['scalping', 'scalping_15m'],
                risk_profile='aggressive',
            )
        except Exception as e:
            print(f'[ScalpBot] scan error: {e}')
            return

        for sig in signals:
            sym       = sig['symbol']
            mode      = sig['mode']
            direction = sig['direction']
            price     = float(sig.get('current_price', 0))
            if price <= 0:
                continue

            key = f'{sym}_{mode}'
            if key in self._open:
                continue  # already in a position for this symbol+mode

            # Use directional probability from binary model when available;
            # fall back to combined confidence from 3-class model.
            dir_conf = (
                sig.get('p_buy',  sig.get('confidence', 0)) if direction == 'BUY'
                else sig.get('p_sell', sig.get('confidence', 0))
            )

            atr_est = price * 0.008  # ~0.8% ATR estimate for scalping
            sl = round(price - atr_est, 8) if direction == 'BUY' else round(price + atr_est, 8)
            tp = round(price + atr_est * 1.5, 8) if direction == 'BUY' else round(price - atr_est * 1.5, 8)

            entry = {
                'signal_id':   sig.get('signal_id', str(uuid.uuid4())[:8]),
                'symbol':      sym,
                'mode':        mode,
                'timeframe':   sig.get('timeframe', '5m'),
                'direction':   direction,
                'confidence':  round(float(dir_conf), 4),
                'p_buy':       round(float(sig.get('p_buy',  0)), 4),
                'p_sell':      round(float(sig.get('p_sell', 0)), 4),
                'p_hold':      round(float(sig.get('p_hold', 0)), 4),
                'entry_price': round(price, 8),
                'sl':          sl,
                'tp':          tp,
                'entry_time':  datetime.now(timezone.utc).isoformat(),
                'position_usdt': self.POSITION_USDT,
                'outcome':     'OPEN',
            }
            self._open[key] = entry
            self._save_record()
            self._push_signal({**entry, 'fire': True})
            print(f'[ScalpBot] FIRE {direction} {sym} @ {price:.6g} | {mode} | conf={entry["confidence"]:.2%}')
