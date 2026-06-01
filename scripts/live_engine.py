#!/usr/bin/env python3
"""
live_engine.py — Aegis-1 Live Signal Engine
============================================
Loads trained XGBoost models from the model store, runs Predictor.predict_realtime()
for every tradeable symbol on a configurable interval, manages a virtual paper-trading
wallet ($10 000 default), and writes data/track_record.json which main.py WebSocket
clients consume in real time.

Exported for main.py
--------------------
    LiveEngine      – async engine class
    TokenConfig     – per-symbol config dataclass
    automated_setup – reads tradeable models, returns run config
"""

import asyncio
import json
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── project root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

MODEL_STORE       = _ROOT / 'src' / 'ml' / 'model_store'
TRACK_RECORD_PATH = _ROOT / 'data' / 'track_record.json'


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class TokenConfig:
    symbol: str
    mode:   str   = 'balanced'


@dataclass
class Position:
    symbol:          str
    direction:       str    # LONG | SHORT
    side:            str    # BUY  | SELL
    entry_price:     float
    position_value:  float  # USDT allocated
    stop_loss:       float
    signal_id:       str
    entry_time:      str
    meta_confidence: float
    atr_multiplier:  float


@dataclass
class TradeRecord:
    signal_id:       str
    symbol:          str
    direction:       str
    side:            str
    entry_price:     float
    exit_price:      Optional[float]
    entry_time:      str
    close_time:      Optional[str]
    pnl_pct:         float
    pnl_usdt:        float
    outcome:         str              # OPEN | WIN | LOSS
    exit_reason:     Optional[str]
    meta_confidence: float
    position_value:  float
    signal_strength: str = ''


# =============================================================================
# Virtual wallet  (paper trading, $10 000 default)
# =============================================================================

class VirtualWallet:
    """Risk 10 % of balance per trade, capped at max_position_usdt."""

    def __init__(self, initial_capital: float, max_position_usdt: float = 1_000.0):
        self.initial_capital   = initial_capital
        self.balance           = initial_capital
        self.max_position_usdt = max_position_usdt
        self.open_positions:   Dict[str, Position]   = {}
        self.trade_history:    List[TradeRecord]      = []

    def position_size(self) -> float:
        return min(self.balance * 0.10, self.max_position_usdt)

    def open_trade(self, pos: Position) -> None:
        self.open_positions[pos.symbol] = pos

    def close_trade(self, symbol: str, exit_price: float,
                    exit_reason: str) -> Optional[TradeRecord]:
        pos = self.open_positions.pop(symbol, None)
        if pos is None:
            return None

        if pos.direction == 'LONG':
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100
        else:
            pnl_pct = (pos.entry_price - exit_price) / pos.entry_price * 100

        pnl_usdt = round(pos.position_value * pnl_pct / 100, 2)
        self.balance += pnl_usdt

        rec = TradeRecord(
            signal_id       = pos.signal_id,
            symbol          = symbol,
            direction       = pos.direction,
            side            = pos.side,
            entry_price     = pos.entry_price,
            exit_price      = round(exit_price, 8),
            entry_time      = pos.entry_time,
            close_time      = datetime.now(timezone.utc).isoformat(),
            pnl_pct         = round(pnl_pct, 3),
            pnl_usdt        = pnl_usdt,
            outcome         = 'WIN' if pnl_pct > 0 else 'LOSS',
            exit_reason     = exit_reason,
            meta_confidence = pos.meta_confidence,
            position_value  = pos.position_value,
        )
        self.trade_history.append(rec)
        return rec

    @property
    def summary(self) -> Dict[str, Any]:
        won   = sum(1 for t in self.trade_history if t.outcome == 'WIN')
        lost  = sum(1 for t in self.trade_history if t.outcome == 'LOSS')
        total = won + lost
        pnl_u = round(self.balance - self.initial_capital, 2)
        return {
            'initial_capital': self.initial_capital,
            'balance':         round(self.balance, 2),
            'total_pnl_usdt':  pnl_u,
            'total_pnl_pct':   round(pnl_u / self.initial_capital * 100, 3),
            'total_trades':    total,
            'won':             won,
            'lost':            lost,
            'win_rate':        round(won / total, 3) if total else 0.0,
            'open_positions':  len(self.open_positions),
        }


# =============================================================================
# Live engine
# =============================================================================

class LiveEngine:
    """
    Prediction loop that scores every tradeable symbol every scan_interval_seconds.

    Signal flow
    -----------
    1. Predictor.predict_realtime() → dict with fire/side/meta_confidence/price/atr
    2. If fire=True and no open position  → open paper trade (VirtualWallet)
    3. If fire=True and opposite position → MODEL_REVERSAL_TP exit, then re-enter
    4. If price hits ATR stop             → STOP_HIT exit
    5. After every cycle  → write data/track_record.json
    """

    MAX_CONCURRENT = 8      # parallel predictor goroutines (semaphore)
    HOURS_CONTEXT  = 300    # bars fed to predictor (300 h ≈ 12.5 days of 1-h data)

    def __init__(
        self,
        token_configs:         List[TokenConfig],
        capital:               float       = 10_000.0,
        max_position_usdt:     float       = 1_000.0,
        scan_interval_seconds: int          = 300,
        risk_tier:             str          = "balanced",
        proxy_url:             Optional[str] = None,
    ):
        self.scan_interval_seconds = scan_interval_seconds
        self.risk_tier = risk_tier
        self.wallet    = VirtualWallet(capital, max_position_usdt)
        self._executor = ThreadPoolExecutor(
            max_workers=self.MAX_CONCURRENT, thread_name_prefix='aegis_pred')

        self.predictors:   Dict[str, Any]   = {}
        self.last_signals: Dict[str, Any]   = {}
        self.live_prices:  Dict[str, float] = {}

        self.bootstrap_done  = 0
        self.bootstrap_total = len(token_configs)

        self._load_predictors([c.symbol for c in token_configs])

    # ── initialisation ────────────────────────────────────────────────────────

    def _load_predictors(self, symbols: List[str]) -> None:
        from src.ml.predictor import Predictor
        loaded = 0
        for sym in symbols:
            try:
                p = Predictor(sym)
                if p.model is not None and p.meta.get('tradeable', False):
                    self.predictors[sym] = p
                    loaded += 1
            except Exception:
                pass
        self.bootstrap_total = max(loaded, 1)
        print(f'[LiveEngine] {loaded} tradeable predictors loaded '
              f'from {len(symbols)} configured symbols.')

    # ── main loop ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        print(f'[LiveEngine] Starting — interval={self.scan_interval_seconds}s '
              f'symbols={len(self.predictors)}')
        while True:
            t0 = time.time()
            await self._scan_all()
            self._save_track_record()
            sleep = max(0.0, self.scan_interval_seconds - (time.time() - t0))
            await asyncio.sleep(sleep)

    async def _scan_all(self) -> None:
        sem = asyncio.Semaphore(self.MAX_CONCURRENT)
        tasks = [self._process_symbol(sym, pred, sem)
                 for sym, pred in self.predictors.items()]
        await asyncio.gather(*tasks, return_exceptions=True)
        # Safety net: ensure bootstrap_done reaches total even if per-symbol
        # increments were skipped due to early returns.
        self.bootstrap_done = len(self.predictors)

    async def _process_symbol(
        self, symbol: str, predictor: Any, sem: asyncio.Semaphore
    ) -> None:
        async with sem:
            loop = asyncio.get_event_loop()
            try:
                result: Dict[str, Any] = await asyncio.wait_for(
                    loop.run_in_executor(
                        self._executor,
                        lambda p=predictor: p.predict_realtime(risk_tier=self.risk_tier),
                    ),
                    timeout=120,
                )
            except Exception:
                self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)
                return
            if not isinstance(result, dict):
                return

            price = float(result.get('price', 0) or 0)
            if price > 0:
                self.live_prices[symbol] = price
            else:
                price = self.live_prices.get(symbol, 0)

            self.last_signals[symbol] = self._build_signal_entry(
                symbol, result, price)

            existing = self.wallet.open_positions.get(symbol)
            if existing:
                self._manage_exit(symbol, existing, result, price)
            elif result.get('fire') and result.get('tradeable', True) and price > 0:
                self._open_position(symbol, result, price)

            self.bootstrap_done = min(self.bootstrap_done + 1, self.bootstrap_total)

    # ── trade management ──────────────────────────────────────────────────────

    def _manage_exit(self, symbol: str, pos: Position,
                     result: Dict[str, Any], price: float) -> None:
        side = result.get('side', 'FLAT')
        fire = bool(result.get('fire', False))

        # Dynamic TP: the meta gate fired the opposite direction
        opposite = (
            (pos.direction == 'LONG'  and side == 'SELL' and fire) or
            (pos.direction == 'SHORT' and side == 'BUY'  and fire)
        )
        if opposite:
            rec = self.wallet.close_trade(symbol, price, 'MODEL_REVERSAL_TP')
            if rec:
                print(f'[{symbol}] TP {rec.outcome} {rec.pnl_pct:+.2f}% '
                      f'MODEL_REVERSAL_TP @ {price}')
            # Immediately open the new position in the reversed direction
            if result.get('tradeable', True) and price > 0:
                self._open_position(symbol, result, price)
            return

        # Safety SL: price crossed the ATR-based hard stop
        if pos.stop_loss > 0:
            sl_hit = (
                (pos.direction == 'LONG'  and price <= pos.stop_loss) or
                (pos.direction == 'SHORT' and price >= pos.stop_loss)
            )
            if sl_hit:
                rec = self.wallet.close_trade(symbol, price, 'STOP_HIT')
                if rec:
                    print(f'[{symbol}] SL {rec.outcome} {rec.pnl_pct:+.2f}% '
                          f'STOP_HIT @ {price}')

    def _open_position(self, symbol: str, result: Dict[str, Any],
                       price: float) -> None:
        side = result.get('side', 'FLAT')
        if side not in ('BUY', 'SELL'):
            return

        direction  = 'LONG' if side == 'BUY' else 'SHORT'
        meta_conf  = float(result.get('meta_confidence', 0))
        atr_mult   = float(result.get('atr_multiplier', 1.5))
        atr        = float(result.get('atr', price * 0.015))

        stop_loss  = (price - atr_mult * atr) if direction == 'LONG' \
                     else (price + atr_mult * atr)
        pos_value  = self.wallet.position_size()

        pos = Position(
            symbol          = symbol,
            direction       = direction,
            side            = side,
            entry_price     = round(price, 8),
            position_value  = round(pos_value, 2),
            stop_loss       = round(stop_loss, 8),
            signal_id       = str(uuid.uuid4()),
            entry_time      = datetime.now(timezone.utc).isoformat(),
            meta_confidence = round(meta_conf, 4),
            atr_multiplier  = atr_mult,
        )
        self.wallet.open_trade(pos)
        print(f'[{symbol}] OPEN {direction} @ {price} | '
              f'conf={meta_conf:.3f} SL={stop_loss:.4f} size={pos_value:.0f} USDT')

    # ── signal entry builder (for dashboard / last_signals) ───────────────────

    @staticmethod
    def _build_signal_entry(symbol: str, result: Dict[str, Any],
                            price: float) -> Dict[str, Any]:
        side = result.get('side', 'FLAT')
        conf = float(result.get('meta_confidence', 0))
        thr  = float(result.get('threshold', 0.6))
        fire = bool(result.get('fire', False))
        atr  = float(result.get('atr', price * 0.015))
        atr_mult = float(result.get('atr_multiplier', 1.5))

        if not fire:
            strength = 'NEUTRAL'
        elif conf >= thr * 1.15:
            strength = f'STRONG_{side}'
        else:
            strength = side

        entry: Dict[str, Any] = {
            'symbol':          symbol,
            'signal':          side,
            'signal_strength': strength,
            'fire':            fire,
            'direction':       'LONG' if side == 'BUY' else ('SHORT' if side == 'SELL' else 'NEUTRAL'),
            'price':           price,
            'entry_price':     price,
            'atr':             round(atr, 8),
            'atr_multiplier':  atr_mult,
            'meta_confidence': round(conf, 4),
            'threshold':       round(thr, 4),
            'tradeable':       result.get('tradeable', True),
            'p_buy':           round(float(result.get('p_buy',  0)), 4),
            'p_sell':          round(float(result.get('p_sell', 0)), 4),
            'p_hold':          round(float(result.get('p_hold', 0)), 4),
            # signal_id is stable while direction unchanged; new UUID only on a real fire.
            # Stable IDs prevent Firestore churn: pushing 24 new UUIDs every 5-min scan
            # was the sole driver of ~288 Firestore writes/day for zero signal change.
            'signal_id':       str(uuid.uuid4()) if fire else f'{symbol.replace("/","_")}_{side}',
            'data_timestamp':  datetime.now(timezone.utc).isoformat(),
            'timestamp':       datetime.now(timezone.utc).isoformat(),
            'timeframe':       '1h',
        }

        # Convenience TP/SL for the active direction
        if side == 'BUY':
            entry['suggested_tp'] = result.get('bull_tp1', round(price + atr_mult * atr, 8))
            entry['suggested_sl'] = round(price - atr_mult * atr, 8)
        elif side == 'SELL':
            entry['suggested_tp'] = result.get('bear_tp1', round(price - atr_mult * atr, 8))
            entry['suggested_sl'] = round(price + atr_mult * atr, 8)
        else:
            entry['suggested_tp'] = None
            entry['suggested_sl'] = None

        # Forward all market context fields from predictor
        _CONTEXT_KEYS = (
            'market_bias', 'bias_strength', 'trend_regime', 'volatility_regime',
            'atr_pct', 'support', 'resistance', 'pivot',
            'r1', 'r2', 's1', 's2',
            'bull_tp1', 'bull_tp2', 'bull_tp3',
            'bear_tp1', 'bear_tp2', 'bear_tp3',
            'confluence',
            'rsi', 'macd_signal', 'cci', 'adx', 'supertrend',
            'macro_daily', 'macro_weekly',
            'volume_strength', 'volume_zscore',
            'funding_rate', 'funding_bias', 'oi_trend', 'oi_change_1h_pct', 'oi_zscore',
            'session', 'session_note', 'fear_greed',
            'scalper_view', 'day_trader_view', 'swing_view',
        )
        for k in _CONTEXT_KEYS:
            if k in result:
                entry[k] = result[k]

        return entry

    # ── track record persistence ──────────────────────────────────────────────

    def _save_track_record(self) -> None:
        try:
            TRACK_RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)

            open_records = [
                {
                    'signal_id':       p.signal_id,
                    'symbol':          p.symbol,
                    'direction':       p.direction,
                    'side':            p.side,
                    'entry_price':     p.entry_price,
                    'exit_price':      None,
                    'entry_time':      p.entry_time,
                    'close_time':      None,
                    'pnl_pct':         0.0,
                    'pnl_usdt':        0.0,
                    'outcome':         'OPEN',
                    'exit_reason':     None,
                    'meta_confidence': p.meta_confidence,
                    'position_value':  p.position_value,
                    'signal_strength': '',
                }
                for p in self.wallet.open_positions.values()
            ]

            all_records = sorted(
                [asdict(t) for t in self.wallet.trade_history] + open_records,
                key=lambda r: r.get('entry_time') or '',
                reverse=True,
            )[:500]

            payload: Dict[str, Any] = {
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'summary':      self.wallet.summary,
                'signals':      all_records,
            }

            with open(TRACK_RECORD_PATH, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2, default=str)
        except Exception as e:
            print(f'[LiveEngine] track_record save failed: {e}')

    async def shutdown(self) -> None:
        self._save_track_record()
        self._executor.shutdown(wait=False)
        print('[LiveEngine] Shutdown complete.')


# =============================================================================
# Setup helpers
# =============================================================================

def automated_setup(_: Path, args: Any):
    """
    Scan MODEL_STORE for tradeable symbols and return engine config.
    Only sidecar JSONs with tradeable=True are included.
    Falls back to BTC/USDT if the store is empty or no models are trained yet.
    """
    configs: List[TokenConfig] = []

    if MODEL_STORE.exists():
        for meta_file in sorted(MODEL_STORE.glob('*_meta.json')):
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                if meta.get('tradeable', False):
                    sym = meta.get('symbol', '')
                    if sym:
                        configs.append(TokenConfig(symbol=sym))
            except Exception:
                pass

    if not configs:
        print('[automated_setup] No tradeable models found — falling back to BTC/USDT.')
        configs = [TokenConfig(symbol='BTC/USDT')]
        
    if len(configs) > 60:
        print(f'[automated_setup] Limiting to 60 tokens (from {len(configs)}) to prevent OOM.')
        configs = configs[:60]

    capital      = float(getattr(args, 'capital',      10_000.0))
    max_pos      = float(getattr(args, 'max_position',  1_000.0))
    scan_seconds = int(getattr(args,   'scan_seconds',    300))
    proxy        = getattr(args, 'proxy', None)

    print(f'[automated_setup] {len(configs)} tradeable symbols | '
          f'capital={capital} | max_pos={max_pos} | scan={scan_seconds}s')
    return configs, capital, max_pos, scan_seconds, proxy


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Aegis-1 Live Signal Engine')
    parser.add_argument('--capital',      type=float, default=10_000.0)
    parser.add_argument('--max-position', type=float, default=1_000.0, dest='max_position')
    parser.add_argument('--scan-seconds', type=int,   default=300,     dest='scan_seconds')
    parser.add_argument('--proxy',        type=str,   default=None)
    cli_args = parser.parse_args()

    _configs, _cap, _maxp, _scan, _proxy = automated_setup(Path('.'), cli_args)
    _engine = LiveEngine(
        token_configs         = _configs,
        capital               = _cap,
        max_position_usdt     = _maxp,
        scan_interval_seconds = _scan,
        proxy_url             = _proxy,
    )
    asyncio.run(_engine.run())
