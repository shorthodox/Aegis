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

# Shared exchange for lightweight index-price fetches (reuses the same instance
# as Predictor once the class is loaded to avoid creating a second connection).
_spot_ex = None
_spot_ex_lock = __import__('threading').Lock()

def _fetch_spot_price(symbol: str) -> float:
    """Thread-safe single-symbol spot price fetch. Returns 0.0 on any error."""
    global _spot_ex
    try:
        import ccxt as _ccxt
        with _spot_ex_lock:
            if _spot_ex is None:
                _spot_ex = _ccxt.binance({'enableRateLimit': True, 'timeout': 8000})
        ticker = _spot_ex.fetch_ticker(symbol)
        return float(ticker.get('last') or ticker.get('close') or 0)
    except Exception:
        return 0.0


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
        loaded = tradeable = 0
        for sym in symbols:
            try:
                p = Predictor(sym)
                if p.model is not None:          # load ALL models, not just tradeable
                    self.predictors[sym] = p
                    loaded += 1
                    if p.meta.get('tradeable', False):
                        tradeable += 1
            except Exception:
                pass
        self.bootstrap_total = max(loaded, 1)
        print(f'[LiveEngine] {loaded} predictors loaded '
              f'({tradeable} tradeable + {loaded - tradeable} monitor-only) '
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

    # Symbols always shown in market overview even if not in tradeable fleet.
    _INDEX_SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'BNB/USDT']

    async def _scan_all(self) -> None:
        sem = asyncio.Semaphore(self.MAX_CONCURRENT)
        tasks = [self._process_symbol(sym, pred, sem)
                 for sym, pred in self.predictors.items()]
        await asyncio.gather(*tasks, return_exceptions=True)
        # Fetch current prices for index/overview symbols not in the tradeable fleet.
        # These are always displayed in the market overview cards on the dashboard.
        await self._fetch_index_prices()
        # Safety net: ensure bootstrap_done reaches total.
        self.bootstrap_done = len(self.predictors)

    async def _fetch_index_prices(self) -> None:
        """Fetch spot prices for market overview symbols not covered by the tradeable fleet."""
        from src.ml.predictor import Predictor
        missing = [s for s in self._INDEX_SYMBOLS if s not in self.live_prices]
        if not missing:
            return
        loop = asyncio.get_event_loop()
        for sym in missing:
            try:
                ticker = await asyncio.wait_for(
                    loop.run_in_executor(
                        self._executor,
                        lambda s=sym: _fetch_spot_price(s),
                    ),
                    timeout=10,
                )
                if ticker and ticker > 0:
                    self.live_prices[sym] = ticker
            except Exception:
                pass

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
    Scan MODEL_STORE for up to 60 symbols.
    Tradeable symbols are loaded first; non-tradeable models fill the remainder
    up to the 60-token cap so the dashboard always shows a full grid.
    Tradeable symbols fire real signals; monitor-only ones show price/context only.
    """
    tradeable_configs:     List[TokenConfig] = []
    non_tradeable_configs: List[TokenConfig] = []

    if MODEL_STORE.exists():
        # Sort by mtime (newest models first) so freshly-trained symbols are preferred
        meta_files = sorted(MODEL_STORE.glob('*_meta.json'),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        seen: set = set()
        for meta_file in meta_files:
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                sym = meta.get('symbol', '')
                if not sym or sym in seen:
                    continue
                seen.add(sym)
                tc = TokenConfig(symbol=sym)
                if meta.get('tradeable', False):
                    tradeable_configs.append(tc)
                else:
                    non_tradeable_configs.append(tc)
            except Exception:
                pass

    # Tradeable first, then fill up to 60 with monitor-only symbols
    TARGET = 60
    configs = tradeable_configs[:TARGET]
    remaining = TARGET - len(configs)
    if remaining > 0:
        configs += non_tradeable_configs[:remaining]

    if not configs:
        print('[automated_setup] No models found — falling back to BTC/USDT.')
        configs = [TokenConfig(symbol='BTC/USDT')]

    capital      = float(getattr(args, 'capital',      10_000.0))
    max_pos      = float(getattr(args, 'max_position',  1_000.0))
    scan_seconds = int(getattr(args,   'scan_seconds',    300))
    proxy        = getattr(args, 'proxy', None)

    t = len(tradeable_configs[:TARGET])
    m = len(configs) - t
    print(f'[automated_setup] {len(configs)} symbols ({t} tradeable + {m} monitor-only) | '
          f'capital={capital} | max_pos={max_pos} | scan={scan_seconds}s')
    return configs, capital, max_pos, scan_seconds, proxy


# =============================================================================
# CLI entry point  —  rich terminal dashboard
# =============================================================================

def _build_terminal_dashboard(engine: 'LiveEngine') -> None:
    """
    Live-updating rich terminal dashboard (shown when run directly).

    Layout  (updates every 2 s)
    ──────────────────────────────────────────────────────────────────
    [HEADER]   Capital · Balance · Total PnL · Win-rate · Warmup bar
    [TOKEN GRID]  All 24 symbols — price, signal, bias, RSI, regime,
                  confidence, open-position P&L indicator
    [OPEN TRADES] Active virtual positions (entry, live PnL, SL)
    [CLOSED (20)] Last 20 closed trades with outcome badge
    """
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text
    from rich.layout import Layout
    from rich import box

    console = Console()

    # ── formatting helpers ────────────────────────────────────────────────────

    def _px(p: float) -> str:
        if p <= 0:       return '—'
        if p < 0.001:    return f'{p:.6f}'
        if p < 1:        return f'{p:.4f}'
        if p < 100:      return f'{p:.3f}'
        return f'{p:.2f}'

    def _pc(v: float) -> str:
        return 'green' if v > 0 else ('red' if v < 0 else 'dim white')

    def _dir(d: str) -> tuple:
        if d == 'LONG':  return '▲', 'bold green'
        if d == 'SHORT': return '▼', 'bold red'
        return '–', 'dim white'

    def _badge(outcome: str) -> Text:
        if outcome == 'WIN':  return Text(' WIN ',  style='bold black on green')
        if outcome == 'LOSS': return Text(' LOSS ', style='bold white on red')
        if outcome == 'OPEN': return Text(' OPEN ', style='bold black on cyan')
        return Text(outcome, style='dim')

    def _bias_cell(bias: str) -> str:
        if bias == 'BULLISH':  return '[green]BULL[/]'
        if bias == 'BEARISH':  return '[red]BEAR[/]'
        return '[dim]NEUT[/]'

    def _signal_cell(sig: dict) -> str:
        side      = sig.get('signal', 'FLAT')
        fire      = sig.get('fire', False)
        strength  = sig.get('signal_strength', '')
        if not fire or side in ('FLAT', 'HOLD'):
            return '[dim]·[/]'
        if 'STRONG' in strength:
            return '[bold green]🔥 STRONG BUY[/]'  if side == 'BUY' \
              else '[bold red]🔥 STRONG SELL[/]'
        return '[green]BUY[/]' if side == 'BUY' else '[red]SELL[/]'

    def _regime_cell(r: str) -> str:
        r = r or ''
        if 'TRENDING_UP'   in r: return '[green]↑TREND[/]'
        if 'TRENDING_DOWN' in r: return '[red]↓TREND[/]'
        if 'TRENDING'      in r: return '[cyan]TREND[/]'
        return '[dim]RANGE[/]'

    def _build_layout() -> Layout:
        wallet   = engine.wallet
        live_px  = engine.live_prices
        signals  = engine.last_signals          # {symbol: entry_dict}

        # ── Header ────────────────────────────────────────────────────────────
        pnl_u    = round(wallet.balance - wallet.initial_capital, 2)
        pnl_pct  = round(pnl_u / wallet.initial_capital * 100, 2)
        pc       = 'green' if pnl_u >= 0 else 'red'
        s        = wallet.summary
        wr       = round(s['win_rate'] * 100, 1) if s['total_trades'] else 0.0
        warmup   = engine.bootstrap_done >= engine.bootstrap_total
        status   = '[bold green]● LIVE[/]' if warmup \
                   else f'[yellow]⏳ WARMUP {engine.bootstrap_done}/{engine.bootstrap_total}[/]'
        now_utc  = datetime.now(timezone.utc).strftime('%Y-%m-%d  %H:%M:%S UTC')

        header = Panel(
            f"  {status}   │   "
            f"Capital [bold cyan]${wallet.initial_capital:,.0f}[/]   │   "
            f"Balance [bold cyan]${wallet.balance:,.2f}[/]   │   "
            f"PnL [{pc}]{pnl_u:+.2f} USDT  ({pnl_pct:+.2f}%)[/]   │   "
            f"Closed [white]{s['total_trades']}[/] "
            f"([green]{s['won']}W[/]/[red]{s['lost']}L[/])   │   "
            f"Win-Rate [bold]{wr:.1f}%[/]   │   "
            f"Open [bold cyan]{len(wallet.open_positions)}[/]   │   "
            f"[dim]{now_utc}[/]",
            title="[bold]  AEGIS-1   Virtual Trading Wallet  [/]",
            border_style='cyan',
        )

        # ── Token status grid — ALL symbols ──────────────────────────────────
        grid = Table(
            title=f'[bold]TOKEN STATUS  ({len(signals)} symbols)[/]',
            box=box.SIMPLE_HEAVY,
            border_style='bright_black',
            show_header=True,
            header_style='bold white',
            expand=True,
        )
        grid.add_column('#',          justify='right',  width=3,  style='dim')
        grid.add_column('Symbol',     justify='left',   min_width=12)
        grid.add_column('Price',      justify='right',  min_width=10)
        grid.add_column('Signal',     justify='center', min_width=12)
        grid.add_column('Bias',       justify='center', width=6)
        grid.add_column('Regime',     justify='center', width=8)
        grid.add_column('RSI',        justify='right',  width=6)
        grid.add_column('Conf',       justify='right',  width=6)
        grid.add_column('Funding',    justify='right',  width=8)
        grid.add_column('Session',    justify='center', width=10)
        grid.add_column('Position',   justify='center', min_width=14)

        tradeable_syms = {
            sym for sym, pred in engine.predictors.items()
            if getattr(pred, 'meta', {}).get('tradeable', False)
        }

        for idx, (sym, sig) in enumerate(sorted(signals.items()), 1):
            price    = float(live_px.get(sym, sig.get('price', 0) or 0))
            conf     = float(sig.get('meta_confidence', 0))
            rsi      = sig.get('rsi', None)
            bias     = sig.get('market_bias', '')
            regime   = sig.get('trend_regime', '')
            session  = sig.get('session', '')[:8]
            funding  = sig.get('funding_rate', None)
            is_tradeable = sym in tradeable_syms

            # live P&L for open position on this symbol
            pos = wallet.open_positions.get(sym)
            if pos:
                cur = price or pos.entry_price
                if pos.direction == 'LONG':
                    ppct = (cur - pos.entry_price) / pos.entry_price * 100
                else:
                    ppct = (pos.entry_price - cur) / pos.entry_price * 100
                arrow, ds = _dir(pos.direction)
                pos_cell = f'[{ds}]{arrow}[/] [{_pc(ppct)}]{ppct:+.2f}%[/]'
            else:
                pos_cell = '[dim]—[/]'

            # RSI coloring
            if rsi is None:
                rsi_cell = '[dim]—[/]'
            else:
                rsi_f = float(rsi)
                if rsi_f >= 70:   rsi_cell = f'[red]{rsi_f:.0f}[/]'
                elif rsi_f <= 30: rsi_cell = f'[green]{rsi_f:.0f}[/]'
                else:             rsi_cell = f'[white]{rsi_f:.0f}[/]'

            # Funding coloring
            if funding is None:
                fund_cell = '[dim]—[/]'
            else:
                ff = float(funding)
                col_f = 'red' if ff > 0.01 else ('green' if ff < -0.01 else 'dim white')
                fund_cell = f'[{col_f}]{ff:+.4f}%[/]'

            # Monitor-only rows are dimmed; tradeable rows show full colour
            sym_cell  = f'[bold]{sym}[/]' if is_tradeable else f'[dim]{sym}[/]'
            px_cell   = f'[bold white]{_px(price)}[/]' if is_tradeable else f'[dim]{_px(price)}[/]'
            conf_cell = (f'[cyan]{conf:.3f}[/]' if conf > 0 else '[dim]—[/]') if is_tradeable \
                        else f'[dim]{conf:.3f}[/]' if conf > 0 else '[dim]—[/]'

            grid.add_row(
                f'[dim]{idx}[/]' if not is_tradeable else str(idx),
                sym_cell,
                px_cell,
                _signal_cell(sig) if is_tradeable else '[dim]watch[/]',
                _bias_cell(bias),
                _regime_cell(regime),
                rsi_cell,
                conf_cell,
                fund_cell,
                f'[dim]{session}[/]',
                pos_cell,
            )

        # ── Open positions detail ─────────────────────────────────────────────
        open_t = Table(
            title='● OPEN POSITIONS',
            box=box.SIMPLE_HEAVY, border_style='cyan',
            show_header=True, header_style='bold cyan', expand=True,
        )
        open_t.add_column('Symbol',     justify='left')
        open_t.add_column('Dir',        justify='center')
        open_t.add_column('Entry',      justify='right')
        open_t.add_column('Live Price', justify='right')
        open_t.add_column('PnL %',      justify='right')
        open_t.add_column('PnL USDT',   justify='right')
        open_t.add_column('Stop-Loss',  justify='right')
        open_t.add_column('Size USDT',  justify='right')
        open_t.add_column('Conf',       justify='right')
        open_t.add_column('Opened',     justify='left')

        for pos in sorted(wallet.open_positions.values(), key=lambda p: p.entry_time):
            cur   = float(live_px.get(pos.symbol, pos.entry_price) or pos.entry_price)
            ppct  = ((cur - pos.entry_price) / pos.entry_price * 100) if pos.direction == 'LONG' \
                    else ((pos.entry_price - cur) / pos.entry_price * 100)
            pu    = round(pos.position_value * ppct / 100, 2)
            arr, ds = _dir(pos.direction)
            open_t.add_row(
                f'[bold]{pos.symbol}[/]',
                f'[{ds}]{arr} {pos.direction}[/]',
                f'[white]{_px(pos.entry_price)}[/]',
                f'[bold]{_px(cur)}[/]',
                f'[{_pc(ppct)}]{ppct:+.2f}%[/]',
                f'[{_pc(pu)}]{pu:+.2f}[/]',
                f'[dim]{_px(pos.stop_loss)}[/]',
                f'[dim]{pos.position_value:.0f}[/]',
                f'[cyan]{pos.meta_confidence:.3f}[/]',
                f'[dim]{pos.entry_time[11:16]} UTC[/]',
            )
        if not wallet.open_positions:
            open_t.add_row(*(['[dim]—[/]'] * 10))

        # ── Closed trades (last 20) ───────────────────────────────────────────
        closed_t = Table(
            title='✔ CLOSED TRADES  (last 20)',
            box=box.SIMPLE_HEAVY, border_style='dim',
            show_header=True, header_style='bold white', expand=True,
        )
        closed_t.add_column('Symbol',   justify='left')
        closed_t.add_column('Dir',      justify='center')
        closed_t.add_column('Entry',    justify='right')
        closed_t.add_column('Exit',     justify='right')
        closed_t.add_column('PnL %',    justify='right')
        closed_t.add_column('PnL USDT', justify='right')
        closed_t.add_column('Reason',   justify='left')
        closed_t.add_column('Outcome',  justify='center')
        closed_t.add_column('Conf',     justify='right')

        for rec in sorted(wallet.trade_history,
                          key=lambda t: t.close_time or '', reverse=True)[:20]:
            arr, ds = _dir(rec.direction)
            closed_t.add_row(
                f'[bold]{rec.symbol}[/]',
                f'[{ds}]{arr} {rec.direction}[/]',
                f'[white]{_px(rec.entry_price)}[/]',
                f'[white]{_px(rec.exit_price or 0)}[/]',
                f'[{_pc(rec.pnl_pct)}]{rec.pnl_pct:+.2f}%[/]',
                f'[{_pc(rec.pnl_usdt)}]{rec.pnl_usdt:+.2f}[/]',
                f'[dim]{(rec.exit_reason or "—").replace("_", " ")}[/]',
                _badge(rec.outcome),
                f'[cyan]{rec.meta_confidence:.3f}[/]',
            )
        if not wallet.trade_history:
            closed_t.add_row(*(['[dim]—[/]'] * 9))

        # ── Assemble layout ───────────────────────────────────────────────────
        layout = Layout()
        layout.split_column(
            Layout(header,   name='hdr',    size=3),
            Layout(grid,     name='grid',   ratio=5),
            Layout(open_t,   name='open',   ratio=2),
            Layout(closed_t, name='closed', ratio=3),
        )
        return layout

    async def _run_with_display() -> None:
        scan_task = asyncio.create_task(engine.run())
        with Live(_build_layout(), console=console,
                  refresh_per_second=0.5, screen=False) as live:
            try:
                while not scan_task.done():
                    live.update(_build_layout())
                    await asyncio.sleep(2)
            except asyncio.CancelledError:
                pass
            finally:
                scan_task.cancel()
                await engine.shutdown()

    asyncio.run(_run_with_display())


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Aegis-1 Live Signal Engine')
    parser.add_argument('--capital',      type=float, default=10_000.0)
    parser.add_argument('--max-position', type=float, default=1_000.0, dest='max_position')
    parser.add_argument('--scan-seconds', type=int,   default=300,     dest='scan_seconds')
    parser.add_argument('--proxy',        type=str,   default=None)
    parser.add_argument('--no-ui',        action='store_true',
                        help='Disable rich terminal UI (plain log output)')
    cli_args = parser.parse_args()

    _configs, _cap, _maxp, _scan, _proxy = automated_setup(Path('.'), cli_args)
    _engine = LiveEngine(
        token_configs         = _configs,
        capital               = _cap,
        max_position_usdt     = _maxp,
        scan_interval_seconds = _scan,
        risk_tier             = 'balanced',
        proxy_url             = _proxy,
    )

    if cli_args.no_ui:
        asyncio.run(_engine.run())
    else:
        _build_terminal_dashboard(_engine)
