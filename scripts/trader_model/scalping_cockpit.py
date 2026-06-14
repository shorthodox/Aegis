#!/usr/bin/env python3
"""
scalping_cockpit.py — AEGIS Scalping Signal Cockpit

Terminal dashboard for scalping + scalping_15m signals from the Universal Trader Model.

Layout (refreshes every 1 s)
────────────────────────────
  [HEADER]    Balance · PnL · Win-rate · scan countdown · UTC clock
  [LIVE GRID] All 60 tokens — price, signal, conf, confl, mode, CD, position
  [EXPIRED]   Closed positions — TP HIT (green) or SL HIT (red)

Monitor loop: checks open position SL/TP every 10 s against live prices.
Scan loop:    runs full scalping scan every SCAN_INTERVAL seconds.

Usage:
    python -m scripts.trader_model.scalping_cockpit
    python -m scripts.trader_model.scalping_cockpit --risk aggressive --scan 60
"""

import argparse
import logging
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rich import box
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from scripts.trader_model.trader_config import MODES, STRATEGY_NAMES, DEPLOYMENT_TOKENS
from scripts.trader_model.trader_engine import TraderEngine, get_trader_engine

log = logging.getLogger(__name__)

# ── Display helpers ────────────────────────────────────────────────────────────

def _px(p: float) -> str:
    if p <= 0:     return '—'
    if p < 0.001:  return f'{p:.6f}'
    if p < 1:      return f'{p:.4f}'
    if p < 100:    return f'{p:.3f}'
    return f'{p:.2f}'


def _age(ts: str) -> str:
    try:
        t = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        s = int((datetime.now(timezone.utc) - t).total_seconds())
        if s < 60:   return f'{s}s'
        if s < 3600: return f'{s // 60}m'
        return f'{s // 3600}h{(s % 3600) // 60}m'
    except Exception:
        return '—'


def _arrow(direction: str) -> str:
    return '▲' if direction == 'BUY' else ('▼' if direction == 'SELL' else '·')


def _dir_style(direction: str) -> str:
    return 'bold green' if direction == 'BUY' else ('bold red' if direction == 'SELL' else 'dim')


def _pnl_style(v: float) -> str:
    return 'green' if v > 0 else ('red' if v < 0 else 'dim white')


def _signal_cell(direction: str, on_cooldown: bool) -> str:
    if direction == 'BUY':
        return '[bold green]🔥 BUY[/]'
    if direction == 'SELL':
        return '[bold red]🔥 SELL[/]'
    if on_cooldown:
        return '[dim yellow]CD[/]'
    return '[dim]·[/]'


def _expired_badge(record: Dict[str, Any]) -> str:
    outcome = record.get('outcome', '?')
    reason  = record.get('exit_reason', '')
    if outcome == 'WIN':
        label = 'TP HIT' if 'TP' in (reason or '').upper() else 'WIN'
        return f'[bold black on green] {label} [/]'
    label = 'SL HIT' if 'SL' in (reason or '').upper() else 'LOSS'
    return f'[bold white on red] {label} [/]'


# ── Cockpit ────────────────────────────────────────────────────────────────────

class ScalpingCockpit:
    """
    Wraps TraderEngine and adds:
      • Background price-monitor thread — checks SL/TP of open positions every N s
      • Background scan thread — re-runs signal scan on a configurable interval
      • Rich Live terminal display with full 60-token live grid
    """

    SCAN_MODES       = ['scalping', 'scalping_15m']
    MONITOR_INTERVAL = 10    # seconds between exit checks
    MAX_EXPIRED_SHOW = 15    # max expired rows in the display

    def __init__(self, risk_profile: str = 'aggressive', scan_interval: int = 60):
        self.engine        = get_trader_engine()
        self.risk_profile  = risk_profile
        self.scan_interval = scan_interval

        self.engine.start_live_monitor(interval_seconds=self.MONITOR_INTERVAL)

        self._next_scan:  float = 0.0   # 0 = scan immediately on first tick
        self._last_scan:  Optional[str] = None
        self._scan_count: int = 0
        self._running     = True
        self._lock        = threading.Lock()

    # ── Background scan loop ──────────────────────────────────────────────────

    def _scan_loop(self) -> None:
        while self._running:
            now = time.time()
            if now < self._next_scan:
                time.sleep(1)
                continue
            try:
                log.info("[cockpit] Running scalping scan…")
                self.engine.scan_all_tokens(
                    modes        = self.SCAN_MODES,
                    risk_profile = self.risk_profile,
                )
                with self._lock:
                    self._last_scan  = datetime.now(timezone.utc).isoformat()
                    self._next_scan  = time.time() + self.scan_interval
                    self._scan_count += 1
            except Exception as exc:
                log.warning(f"[cockpit] Scan error: {exc}")
                time.sleep(15)

    # ── Rich renderable ───────────────────────────────────────────────────────

    def _build(self) -> Group:
        wallet  = self.engine.wallet
        summary = wallet.summary

        expired   = list(getattr(self.engine, 'expired_signals', []))
        live_px   = dict(getattr(self.engine, '_live_prices', {}))

        pnl_u   = summary['total_pnl_usdt']
        pnl_pct = summary['total_pnl_pct']
        pnl_col = 'green' if pnl_u >= 0 else 'red'
        won     = summary['won']
        lost    = summary['lost']
        total   = summary['total_trades']
        wr_pct  = round(summary['win_rate'] * 100, 1) if total else 0.0
        n_open  = summary['open_positions']

        with self._lock:
            last_scan_str = (_age(self._last_scan) + ' ago') if self._last_scan else 'scanning…'
            next_secs     = max(0, int(self._next_scan - time.time()))
            scan_no       = self._scan_count

        next_str = f'{next_secs}s' if next_secs > 0 else '[yellow]scanning…[/]'
        now_str  = datetime.now(timezone.utc).strftime('%Y-%m-%d  %H:%M:%S UTC')

        # ── Header ────────────────────────────────────────────────────────────
        header = Panel(
            f"  [bold green]● LIVE[/]   │   "
            f"Balance [bold cyan]${summary['balance']:,.2f}[/]   │   "
            f"PnL [{pnl_col}]{pnl_u:+.2f} USDT  ({pnl_pct:+.2f}%)[/]   │   "
            f"Trades [white]{total}[/] ([green]{won}W[/]/[red]{lost}L[/])   WR [bold]{wr_pct:.1f}%[/]   │   "
            f"Open [bold cyan]{n_open}[/]   │   "
            f"Scan #{scan_no}  last [dim]{last_scan_str}[/]  next {next_str}   │   "
            f"[dim]{now_str}[/]",
            title="[bold cyan]  AEGIS  SCALPING COCKPIT  [/]",
            border_style='cyan',
        )

        # ── Live Token Grid ────────────────────────────────────────────────────
        token_status = self.engine.token_status
        open_pos     = wallet.open_positions   # key → pos dict

        # Sort: BUY/SELL first (by conf desc), then HOLD tokens (by symbol)
        all_syms = sorted(DEPLOYMENT_TOKENS)
        active   = sorted(
            [s for s in all_syms if token_status.get(s, {}).get('direction') in ('BUY', 'SELL')],
            key=lambda s: -token_status.get(s, {}).get('confidence', 0),
        )
        hold_syms = [s for s in all_syms if s not in active]
        ordered  = active + hold_syms

        n_buy  = sum(1 for s in all_syms if token_status.get(s, {}).get('direction') == 'BUY')
        n_sell = sum(1 for s in all_syms if token_status.get(s, {}).get('direction') == 'SELL')

        grid = Table(
            box=box.SIMPLE_HEAVY,
            border_style='bright_black',
            show_header=True,
            header_style='bold white',
            expand=True,
            title=(
                f'[bold dim]LIVE TOKEN GRID — {len(all_syms)} tokens · '
                f'[green]{n_buy} BUY[/] [dim]|[/] [red]{n_sell} SELL[/] · '
                f'refreshes every {self.scan_interval}s[/]'
            ),
        )
        grid.add_column('#',       justify='right',  width=3,  style='dim')
        grid.add_column('Symbol',  justify='left',   width=14)
        grid.add_column('Price',   justify='right',  width=10)
        grid.add_column('Signal',  justify='center', width=10)
        grid.add_column('Dir',     justify='center', width=6)
        grid.add_column('Conf',    justify='right',  width=6)
        grid.add_column('Confl',   justify='right',  width=6)
        grid.add_column('Mode',    justify='left',   width=10)
        grid.add_column('CD',      justify='center', width=4)
        grid.add_column('Position',justify='center', width=14)

        for idx, sym in enumerate(ordered, 1):
            t         = token_status.get(sym, {})
            direction = t.get('direction', 'HOLD')
            conf      = float(t.get('confidence', 0))
            confl     = float(t.get('confluence_score', 0))
            mode      = t.get('mode', '—')
            on_cd     = bool(t.get('on_cooldown', False))
            t_price   = float(t.get('price', 0))
            price     = float(live_px.get(sym, t_price) or t_price)

            # Position P&L for this symbol
            pos_cell = '[dim]—[/]'
            for key, pos in open_pos.items():
                if pos.get('symbol') == sym:
                    entry = float(pos.get('entry_price', 0))
                    cur   = price if price > 0 else entry
                    if entry > 0:
                        pnl = ((cur - entry) / entry * 100) if pos.get('direction') == 'BUY' \
                              else ((entry - cur) / entry * 100)
                        pc  = _pnl_style(pnl)
                        arr = _arrow(pos.get('direction', ''))
                        pos_cell = f'[{_dir_style(pos.get("direction",""))}]{arr}[/] [{pc}]{pnl:+.2f}%[/]'
                    break

            d_style   = _dir_style(direction)
            n_ag      = round(confl * len(STRATEGY_NAMES))
            cd_cell   = '[yellow]CD[/]' if on_cd else '[dim]—[/]'
            conf_col  = 'green' if conf >= 0.65 else ('yellow' if conf >= 0.55 else 'dim white')
            confl_col = 'green' if confl >= 0.5 else ('yellow' if confl >= 0.36 else 'dim')

            if direction in ('BUY', 'SELL'):
                sym_cell  = f'[bold]{sym}[/]'
                px_cell   = f'[bold white]{_px(price)}[/]'
                conf_cell = f'[{conf_col}]{conf * 100:.0f}%[/]'
                confl_cell= f'[{confl_col}]{n_ag:2d}/25[/]'
                mode_cell = f'[dim]{mode}[/]'
            else:
                sym_cell  = f'[dim]{sym}[/]'
                px_cell   = f'[dim]{_px(price)}[/]'
                conf_cell = f'[dim]{conf * 100:.0f}%[/]' if conf > 0 else '[dim]—[/]'
                confl_cell= f'[dim]{n_ag}/25[/]' if confl > 0 else '[dim]—[/]'
                mode_cell = f'[dim]{mode}[/]'

            grid.add_row(
                f'[dim]{idx}[/]' if direction == 'HOLD' else f'[bold]{idx}[/]',
                sym_cell,
                px_cell,
                _signal_cell(direction, on_cd),
                f'[{d_style}]{_arrow(direction)} {direction}[/]' if direction != 'HOLD' else '[dim]—[/]',
                conf_cell,
                confl_cell,
                mode_cell,
                cd_cell,
                pos_cell,
            )

        # ── Expired / closed ──────────────────────────────────────────────────
        exp_lines: List[str] = []
        for rec in expired[:self.MAX_EXPIRED_SHOW]:
            sym       = rec.get('symbol', '?')
            direction = rec.get('direction', '?')
            mode      = rec.get('mode', '?')
            pnl_p     = float(rec.get('pnl_pct')   or 0)
            pnl_u_v   = float(rec.get('pnl_usdt')  or 0)
            reason    = (rec.get('exit_reason') or '').replace('_', ' ')
            close_ts  = rec.get('exit_time') or rec.get('expired_at', '')
            d_style   = _dir_style(direction)
            p_style   = _pnl_style(pnl_p)
            badge     = _expired_badge(rec)
            exp_lines.append(
                f"  {badge}  [{d_style}]{_arrow(direction)} {sym}[/]  "
                f"[dim]{mode}[/]  "
                f"[{p_style}]{pnl_p:+.2f}%  {pnl_u_v:+.2f} USDT[/]  "
                f"[dim]{reason}  {_age(close_ts)} ago[/]"
            )

        expired_panel = Panel(
            "\n".join(exp_lines) if exp_lines else "  [dim]No expired signals yet[/]",
            title=f"[bold white]EXPIRED / CLOSED[/]  —  {len(expired)} signal(s)",
            border_style='dim',
        )

        return Group(header, grid, expired_panel)

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self) -> None:
        scan_thread = threading.Thread(
            target=self._scan_loop, daemon=True, name='cockpit-scan',
        )
        scan_thread.start()

        try:
            with Live(self._build(), screen=True, refresh_per_second=1) as live:
                while True:
                    try:
                        live.update(self._build())
                    except Exception:
                        pass
                    time.sleep(1)
        except KeyboardInterrupt:
            self._running = False
            print("\n[cockpit] Stopped.")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AEGIS Scalping Cockpit')
    parser.add_argument(
        '--risk',
        choices=['conservative', 'balanced', 'aggressive'],
        default='aggressive',
        help='Risk profile (default: aggressive)',
    )
    parser.add_argument(
        '--scan',
        type=int,
        default=60,
        metavar='SECONDS',
        help='Seconds between signal scans (default: 60)',
    )
    parser.add_argument(
        '--monitor',
        type=int,
        default=10,
        metavar='SECONDS',
        help='Seconds between SL/TP price checks (default: 10)',
    )
    cli = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger('scripts.trader_model').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.ERROR)
    logging.getLogger('ccxt').setLevel(logging.ERROR)

    cockpit = ScalpingCockpit(
        risk_profile  = cli.risk,
        scan_interval = cli.scan,
    )
    cockpit.engine.start_live_monitor(interval_seconds=cli.monitor)
    cockpit.run()
