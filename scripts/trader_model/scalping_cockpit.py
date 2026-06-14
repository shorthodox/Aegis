#!/usr/bin/env python3
"""
scalping_cockpit.py — AEGIS Scalping Signal Cockpit

Terminal dashboard for scalping + scalping_15m signals from the Universal Trader Model.

Layout (refreshes every 1 s)
────────────────────────────
  [HEADER]   Balance · PnL · Win-rate · scan countdown · UTC clock
  [ACTIVE]   Open positions with live price, live P&L%, SL, TP1, age
  [SCAN]     Latest BUY/SELL tokens from last signal scan (non-HOLD only)
  [EXPIRED]  Closed positions — TP HIT (green) or SL HIT (red)

Monitor loop: checks open position SL/TP every 10 s against live prices.
Scan loop:    runs full scalping scan every SCAN_INTERVAL seconds.

Usage:
    python -m scripts.trader_model.scalping_cockpit
    python -m scripts.trader_model.scalping_cockpit --risk aggressive --scan 180
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

from scripts.trader_model.trader_config import MODES, STRATEGY_NAMES
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
    return '▲' if direction == 'BUY' else ('▼' if direction == 'SELL' else '–')


def _dir_style(direction: str) -> str:
    return 'bold green' if direction == 'BUY' else ('bold red' if direction == 'SELL' else 'dim')


def _pnl_style(v: float) -> str:
    return 'green' if v > 0 else ('red' if v < 0 else 'dim white')


def _expired_badge(record: Dict[str, Any]) -> str:
    """Return a Rich markup string badge for an expired signal."""
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
      • Rich Live terminal display
    """

    SCAN_MODES       = ['scalping', 'scalping_15m']
    MONITOR_INTERVAL = 10    # seconds between exit checks
    MAX_EXPIRED_SHOW = 20    # max expired rows in the display

    def __init__(self, risk_profile: str = 'balanced', scan_interval: int = 300):
        self.engine        = get_trader_engine()
        self.risk_profile  = risk_profile
        self.scan_interval = scan_interval

        # Start the engine's built-in live monitor
        self.engine.start_live_monitor(interval_seconds=self.MONITOR_INTERVAL)

        self._next_scan:  float = 0.0   # epoch seconds — 0 = scan immediately
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
                log.info(f"[cockpit] Scan #{self._scan_count} complete")
            except Exception as exc:
                log.warning(f"[cockpit] Scan error: {exc}")
                time.sleep(30)

    # ── Rich renderable ───────────────────────────────────────────────────────

    def _build(self) -> Group:
        wallet  = self.engine.wallet
        summary = wallet.summary

        # Expired list from engine's live monitor
        expired = list(getattr(self.engine, 'expired_signals', []))

        # Live prices from engine's monitor
        live_px: Dict[str, float] = dict(getattr(self.engine, '_live_prices', {}))

        pnl_u   = summary['total_pnl_usdt']
        pnl_pct = summary['total_pnl_pct']
        pnl_col = 'green' if pnl_u >= 0 else 'red'
        won     = summary['won']
        lost    = summary['lost']
        total   = summary['total_trades']
        wr_pct  = round(summary['win_rate'] * 100, 1) if total else 0.0
        n_open  = summary['open_positions']

        with self._lock:
            last_scan_str = (_age(self._last_scan) + ' ago') if self._last_scan else 'pending…'
            next_secs     = max(0, int(self._next_scan - time.time()))

        next_str  = f'{next_secs}s' if next_secs > 0 else '[yellow]scanning…[/]'
        now_str   = datetime.now(timezone.utc).strftime('%Y-%m-%d  %H:%M:%S UTC')

        # ── Header ────────────────────────────────────────────────────────────
        header = Panel(
            f"  [bold green]● LIVE[/]   │   "
            f"Balance [bold cyan]${summary['balance']:,.2f}[/]   │   "
            f"PnL [{pnl_col}]{pnl_u:+.2f} USDT  ({pnl_pct:+.2f}%)[/]   │   "
            f"Trades [white]{total}[/] ([green]{won}W[/]/[red]{lost}L[/])   │   "
            f"WR [bold]{wr_pct:.1f}%[/]   │   "
            f"Open [bold cyan]{n_open}[/]   │   "
            f"Scan  last [dim]{last_scan_str}[/] · next {next_str}   │   "
            f"[dim]{now_str}[/]",
            title="[bold cyan]  AEGIS  SCALPING COCKPIT  [/]",
            border_style='cyan',
        )

        # ── Active positions table ────────────────────────────────────────────
        open_tbl = Table(
            box=box.SIMPLE_HEAVY, border_style='bright_black',
            show_header=True, header_style='bold white', expand=True,
            title=f'[bold]ACTIVE SIGNALS[/]  —  {n_open} open position(s)',
        )
        open_tbl.add_column('#',       justify='right',  width=3)
        open_tbl.add_column('Token',   justify='left',   width=14)
        open_tbl.add_column('Mode',    justify='left',   width=11)
        open_tbl.add_column('Dir',     justify='center', width=8)
        open_tbl.add_column('Conf',    justify='right',  width=6)
        open_tbl.add_column('Entry',   justify='right',  width=11)
        open_tbl.add_column('Current', justify='right',  width=11)
        open_tbl.add_column('P&L%',    justify='right',  width=8)
        open_tbl.add_column('SL',      justify='right',  width=11)
        open_tbl.add_column('TP1',     justify='right',  width=11)
        open_tbl.add_column('Status',  justify='center', width=8)
        open_tbl.add_column('Age',     justify='right',  width=6)

        open_rows = sorted(
            wallet.open_positions.items(),
            key=lambda kv: kv[1].get('timestamp', ''),
            reverse=True,
        )

        if open_rows:
            for i, (key, pos) in enumerate(open_rows, 1):
                sym       = pos.get('symbol', key.split('__')[0])
                direction = pos.get('direction', '?')
                conf      = float(pos.get('confidence', 0))
                mode      = pos.get('mode', '?')
                entry     = float(pos.get('entry_price', 0))
                sl        = float(pos.get('stop_loss') or 0)
                tp1       = float(pos.get('tp1') or 0)
                ts        = pos.get('timestamp', '')

                cur = live_px.get(sym, entry)
                if entry > 0:
                    pnl = ((cur - entry) / entry * 100) if direction == 'BUY' \
                          else ((entry - cur) / entry * 100)
                else:
                    pnl = 0.0

                d_style = _dir_style(direction)
                p_style = _pnl_style(pnl)

                open_tbl.add_row(
                    f'[dim]{i}[/]',
                    f'[bold]{sym}[/]',
                    f'[dim]{mode}[/]',
                    f'[{d_style}]{_arrow(direction)} {direction}[/]',
                    f'[cyan]{conf * 100:.0f}%[/]',
                    f'[white]{_px(entry)}[/]',
                    f'[bold white]{_px(cur)}[/]',
                    f'[{p_style}]{pnl:+.2f}%[/]',
                    f'[red]{_px(sl)}[/]'   if sl  > 0 else '[dim]—[/]',
                    f'[green]{_px(tp1)}[/]' if tp1 > 0 else '[dim]—[/]',
                    Text(' OPEN ', style='bold black on cyan'),
                    _age(ts),
                )
        else:
            open_tbl.add_row(
                *[''] * 3,
                '[dim]No open positions — waiting for next scan[/]',
                *[''] * 8,
            )

        # ── Scan results (BUY/SELL only, not HOLD) ────────────────────────────
        token_status = self.engine.token_status
        actionable   = [t for t in token_status.values()
                        if t.get('direction') in ('BUY', 'SELL')]
        actionable.sort(key=lambda t: -t.get('confidence', 0))

        scan_tbl = Table(
            box=box.SIMPLE, border_style='bright_black',
            show_header=True, header_style='bold dim', expand=True,
            title=(
                f'[bold dim]LAST SCAN[/]  —  {len(token_status)} tokens  │  '
                f'[green]{sum(1 for t in token_status.values() if t.get("direction")=="BUY")} BUY[/]  '
                f'[red]{sum(1 for t in token_status.values() if t.get("direction")=="SELL")} SELL[/]'
            ),
        )
        scan_tbl.add_column('Token',  justify='left',   width=14)
        scan_tbl.add_column('Dir',    justify='center', width=8)
        scan_tbl.add_column('Conf',   justify='right',  width=6)
        scan_tbl.add_column('Confl',  justify='right',  width=6)
        scan_tbl.add_column('Mode',   justify='left',   width=11)
        scan_tbl.add_column('Price',  justify='right',  width=11)
        scan_tbl.add_column('Top Strategy',             width=28)
        scan_tbl.add_column('CD',     justify='center', width=5)

        if actionable:
            for t in actionable[:25]:   # cap to 25 rows so table doesn't scroll off screen
                d       = t.get('direction', 'HOLD')
                d_style = _dir_style(d)
                n_ag    = round(t.get('confluence_score', 0) * len(STRATEGY_NAMES))
                strat   = (t.get('top_strategy') or '').replace('_', ' ')[:26]
                scan_tbl.add_row(
                    f'[bold]{t["symbol"]}[/]',
                    f'[{d_style}]{_arrow(d)} {d}[/]',
                    f'[{d_style}]{t.get("confidence", 0) * 100:.0f}%[/]',
                    f'[dim]{n_ag:2d}/25[/]',
                    f'[dim]{t.get("mode", "?")}[/]',
                    f'[white]{_px(t.get("price", 0))}[/]',
                    f'[dim]{strat}[/]',
                    '[yellow]CD[/]' if t.get('on_cooldown') else '[dim]—[/]',
                )
        else:
            scan_tbl.add_row('[dim]No BUY/SELL signals in last scan[/]', *[''] * 7)

        # ── Expired / closed positions ────────────────────────────────────────
        exp_lines: List[str] = []
        for rec in expired[:self.MAX_EXPIRED_SHOW]:
            sym       = rec.get('symbol', '?')
            direction = rec.get('direction', '?')
            mode      = rec.get('mode', '?')
            pnl_p     = float(rec.get('pnl_pct')   or 0)
            pnl_u_v   = float(rec.get('pnl_usdt')  or 0)
            reason    = (rec.get('exit_reason') or '').replace('_', ' ')
            close_ts  = rec.get('exit_time') or rec.get('expired_at', '')
            p_style   = _pnl_style(pnl_p)
            d_style   = _dir_style(direction)
            badge = _expired_badge(rec)

            exp_lines.append(
                f"  {badge}  [{d_style}]{_arrow(direction)} {sym}[/]  "
                f"[dim]{mode}[/]  "
                f"[{p_style}]{pnl_p:+.2f}%  {pnl_u_v:+.2f} USDT[/]  "
                f"[dim]{reason}  {_age(close_ts)}[/]"
            )

        expired_panel = Panel(
            "\n".join(exp_lines) if exp_lines else "  [dim]No expired signals yet[/]",
            title=f"[bold white]EXPIRED[/]  —  {len(expired)} signal(s) closed",
            border_style='dim',
        )

        return Group(header, open_tbl, scan_tbl, expired_panel)

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
        default='balanced',
        help='Risk profile (default: balanced)',
    )
    parser.add_argument(
        '--scan',
        type=int,
        default=300,
        metavar='SECONDS',
        help='Seconds between signal scans (default: 300)',
    )
    parser.add_argument(
        '--monitor',
        type=int,
        default=10,
        metavar='SECONDS',
        help='Seconds between SL/TP price checks (default: 10)',
    )
    cli = parser.parse_args()

    # Suppress noisy engine logs — let the Rich display speak
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
