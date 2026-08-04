"""Rich terminal UI.

Display only — nothing here may influence a decision. If a number is
wrong on screen, fix the field the engine publishes rather than the
formatting, so the chart and the terminal cannot disagree.

Extracted verbatim from the single-file live_engine.py.
"""
from __future__ import annotations

from datetime import datetime
from datetime import timezone
from typing import List
import asyncio

from scripts.engine.config import REGIME_ACCUMULATION as _REGIME_ACCUMULATION
from scripts.engine.config import REGIME_DISTRIBUTION as _REGIME_DISTRIBUTION
from scripts.engine.config import REGIME_LIQUIDITY_TRAP as _REGIME_LIQUIDITY_TRAP
from scripts.engine.config import REGIME_RANGING as _REGIME_RANGING
from scripts.engine.config import REGIME_TRENDING_BEAR as _REGIME_TRENDING_BEAR
from scripts.engine.config import REGIME_TRENDING_BULL as _REGIME_TRENDING_BULL
from scripts.engine.config import REGIME_VOLATILE_COMPRESS as _REGIME_VOLATILE_COMPRESS
from scripts.engine.config import REGIME_VOLATILE_EXPANSION as _REGIME_VOLATILE_EXPANSION
from scripts.engine.scalp import ScalpBot

def _build_terminal_dashboard(engine: 'LiveEngine') -> None:
    """
    Live-updating rich terminal dashboard (shown when run directly).

    Layout  (updates every 2 s)
    ──────────────────────────────────────────────────────────────────
    [HEADER]   Capital · Balance · Total PnL · Win-rate · Warmup bar
    [TOKEN GRID]  All symbols — price, signal, bias, RSI, regime,
                  confidence, quality score, open-position P&L
    [OPEN TRADES] Active virtual positions (entry, live PnL, SL)
    [CLOSED (5)]  Last 5 closed trades with outcome badge
    """
    from rich.console import Group
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.live import Live
    from rich import box

    def _px(p: float) -> str:
        if p <= 0:      return '—'
        if p < 0.001:   return f'{p:.6f}'
        if p < 1:       return f'{p:.4f}'
        if p < 100:     return f'{p:.3f}'
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

    def _signal_cell(sig: dict) -> str:
        side     = sig.get('signal', 'FLAT')
        fire     = sig.get('fire', False)
        strength = sig.get('signal_strength', '')
        if not fire or side in ('FLAT', 'HOLD'):
            if sig.get('pending_entry'):
                _ps = str(sig.get('pending_side', '') or '')
                return f'[yellow]⏳ ARMED {_ps}[/]'.rstrip()
            
            _qual = float(sig.get('quality_score', 0) or 0)
            _reason = str(sig.get('structure_reason') or sig.get('pending_reason') or sig.get('vetoes') or '').strip()
            
            if _qual >= 60 or sig.get('location_blocked') or sig.get('structure_blocked') or sig.get('rr_blocked') or sig.get('momentum_blocked'):
                if 'WRONG_ZONE' in _reason or 'wrong zone' in _reason.lower():
                    return '[dim yellow]✋ WRONG_ZONE[/]'
                if 'headroom' in _reason.lower() or 'rr' in _reason.lower() or sig.get('rr_blocked'):
                    return '[dim yellow]✋ LOW_RR[/]'
                if 'unconfirmed' in _reason.lower() or '5m' in _reason.lower() or sig.get('momentum_blocked'):
                    return '[dim yellow]⏳ UNCONF_5M[/]'
                if 'far' in _reason.lower() or 'waiting' in _reason.lower():
                    return '[dim yellow]⏳ PENDING[/]'
                if _reason:
                    _short_r = _reason.replace('MODEL BLOCK', '').replace('STRUCTURE_GATE', '').replace('blocked', '').strip()[:14]
                    return f'[dim yellow]✋ {_short_r.upper()}[/]'
                return '[dim yellow]✋ GATED[/]'

            return '[dim]·[/]'
        _p = ' [dim](paper)[/]' if sig.get('paper_only') else ''
        if 'STRONG' in strength:
            return ('[bold green]🔥 STRONG BUY[/]' if side == 'BUY'
                    else '[bold red]🔥 STRONG SELL[/]') + _p
        return ('[green]BUY[/]' if side == 'BUY' else '[red]SELL[/]') + _p

    def _regime_short(r: str) -> str:
        r = r or 'UNKNOWN'
        _MAP = {
            _REGIME_TRENDING_BULL:      '[green]↑BULL[/]',
            _REGIME_TRENDING_BEAR:      '[red]↓BEAR[/]',
            _REGIME_RANGING:            '[dim]RANGE[/]',
            _REGIME_ACCUMULATION:       '[cyan]ACCUM[/]',
            _REGIME_DISTRIBUTION:       '[yellow]DIST[/]',
            _REGIME_VOLATILE_EXPANSION: '[bold red]EXPND[/]',
            _REGIME_VOLATILE_COMPRESS:  '[dim]CMPR[/]',
            _REGIME_LIQUIDITY_TRAP:     '[bold red]TRAP[/]',
        }
        return _MAP.get(r, f'[dim]{r[:5]}[/]')

    def _quality_cell(q: float) -> str:
        if q >= 75:   return f'[bold green]{q:.0f}[/]'
        if q >= 55:   return f'[yellow]{q:.0f}[/]'
        if q > 0:     return f'[dim red]{q:.0f}[/]'
        return '[dim]—[/]'

    def _build_renderable() -> Group:
        """Build the full dashboard as a Group (header + grid + footer) for Live."""
        wallet  = engine.wallet
        live_px = engine.live_prices
        signals = engine.last_signals

        pnl_u   = round(wallet.balance - wallet.initial_capital, 2)
        pnl_pct = round(pnl_u / wallet.initial_capital * 100, 2)
        pc      = 'green' if pnl_u >= 0 else 'red'
        s       = wallet.summary
        wr      = round(s['win_rate'] * 100, 1) if s['total_trades'] else 0.0
        warmup  = engine.bootstrap_done >= engine.bootstrap_total
        if warmup:
            status = '[bold green]● LIVE[/]'
        else:
            _ready = len(engine.last_signals)
            status = f'[yellow]⏳ WARMING UP  {_ready}/{engine.bootstrap_total} tokens ready[/]'
        now_utc = datetime.now(timezone.utc).strftime('%Y-%m-%d  %H:%M:%S UTC')
        perf    = engine.perf_tracker.get_performance_summary()
        safe_tag = '[bold red]  ⚠ SAFE-MODE[/]' if perf['safe_mode'] else ''

        header = Panel(
            f"  {status}{safe_tag}   │   "
            f"Capital [bold cyan]${wallet.initial_capital:,.0f}[/]   │   "
            f"Balance [bold cyan]${wallet.balance:,.2f}[/]   │   "
            f"PnL [{pc}]{pnl_u:+.2f} USDT  ({pnl_pct:+.2f}%)[/]   │   "
            f"Closed [white]{s['total_trades']}[/] "
            f"([green]{s['won']}W[/]/[red]{s['lost']}L[/])   │   "
            f"WR [bold]{wr:.1f}%[/]   │   "
            f"Open [bold cyan]{len(wallet.open_positions)}[/]   │   "
            f"[dim]{now_utc}[/]",
            title="[bold]  AEGIS-1  Signal Engine  [/]",
            border_style='cyan',
        )

        tradeable_syms = {
            sym for sym, pred in engine.predictors.items()
            if getattr(pred, 'meta', {}).get('tradeable', False)
        }

        _all_syms = sorted(engine.predictors.keys()) if engine.predictors else sorted(signals.keys())
        # Show ALL tokens; sort: open positions first, then active signals, then HOLD by alpha
        def _row_priority(s: str) -> int:
            if s in wallet.open_positions:
                return 0
            sig_s = signals.get(s, {})
            if sig_s.get('fire') and sig_s.get('signal', 'HOLD') not in ('HOLD', 'FLAT', ''):
                return 1
            return 2
        all_syms = sorted(_all_syms, key=lambda s: (_row_priority(s), s))

        _n_firing = sum(1 for s in _all_syms
                        if signals.get(s, {}).get('fire')
                        and signals.get(s, {}).get('signal', 'HOLD') not in ('HOLD', 'FLAT', ''))
        _n_open   = len(wallet.open_positions)

        grid = Table(
            box=box.SIMPLE_HEAVY,
            border_style='bright_black',
            show_header=True,
            header_style='bold white',
            expand=True,
            title=f'[bold dim]ALL TOKENS — {_n_firing} firing · {_n_open} open · {len(_all_syms)} monitored · refreshes every 1s[/]',
        )
        grid.add_column('#',        justify='right',  width=3,  style='dim')
        grid.add_column('Symbol',   justify='left',   width=14)
        grid.add_column('Price',    justify='right',  width=11)
        grid.add_column('Signal',   justify='center', width=13)
        grid.add_column('Dir%',     justify='right',  width=6)
        grid.add_column('Quality',  justify='right',  width=7)
        grid.add_column('RSI',      justify='right',  width=5)
        grid.add_column('Regime',   justify='center', width=7)
        grid.add_column('SL',       justify='right',  width=11)
        grid.add_column('TP',       justify='right',  width=11)
        grid.add_column('Position', justify='center', width=16)
        for idx, sym in enumerate(all_syms, 1):
            sig          = signals.get(sym, {})
            price        = float(live_px.get(sym, sig.get('price', 0) or 0))
            rsi          = sig.get('rsi', None)
            regime       = sig.get('regime', 'UNKNOWN')
            quality      = float(sig.get('quality_score', 0))
            is_tradeable = sym in tradeable_syms
            _pred_meta   = engine.predictors[sym].meta if sym in engine.predictors else {}
            dir_prec     = float(_pred_meta.get('holdout_trading', {}).get('directional_precision', 0))

            pos = wallet.open_positions.get(sym)

            # ── SL / TP cells ──────────────────────────────────────────────
            if pos:
                sl_val = pos.stop_loss
                tp_val = pos.take_profit_1
                sl_cell = f'[red]{_px(sl_val)}[/]'
                tp_cell = f'[green]{_px(tp_val)}[/]'
            elif sig.get('fire') and sig.get('suggested_sl'):
                sl_val = float(sig.get('suggested_sl') or 0)
                tp_val = float(sig.get('suggested_tp') or 0)
                sl_cell = f'[dim red]{_px(sl_val)}[/]'
                tp_cell = f'[dim green]{_px(tp_val)}[/]'
            else:
                sl_cell = '[dim]—[/]'
                tp_cell = '[dim]—[/]'

            # ── Position P&L cell ──────────────────────────────────────────
            if pos:
                cur = price or pos.entry_price
                ppct = ((cur - pos.entry_price) / pos.entry_price * 100) if pos.direction == 'LONG' \
                       else ((pos.entry_price - cur) / pos.entry_price * 100)
                arrow, ds = _dir(pos.direction)
                trail_flag = '[bold yellow]T[/] ' if engine._tp1_hit.get(sym) else ''
                pos_cell   = f'{trail_flag}[{ds}]{arrow}[/] [{_pc(ppct)}]{ppct:+.2f}%[/]'
            else:
                pos_cell = '[dim]—[/]'

            # ── RSI cell ───────────────────────────────────────────────────
            if rsi is None:
                rsi_cell = '[dim]—[/]'
            else:
                rsi_f = float(rsi)
                if rsi_f >= 70:   rsi_cell = f'[red]{rsi_f:.0f}[/]'
                elif rsi_f <= 30: rsi_cell = f'[green]{rsi_f:.0f}[/]'
                else:             rsi_cell = f'{rsi_f:.0f}'

            # ── Dir% cell ──────────────────────────────────────────────────
            if dir_prec > 0:
                dp_pct   = dir_prec * 100
                dp_col   = 'green' if dp_pct >= 70 else ('yellow' if dp_pct >= 60 else 'red')
                dir_cell = f'[{dp_col}]{dp_pct:.0f}%[/]' if is_tradeable else f'[dim]{dp_pct:.0f}%[/]'
            else:
                dir_cell = '[dim]—[/]'

            sym_cell = f'[bold]{sym}[/]' if is_tradeable else f'[dim]{sym}[/]'
            px_cell  = f'[bold white]{_px(price)}[/]' if is_tradeable else f'[dim]{_px(price)}[/]'

            # ── Signal cell: show conflict when model opposes open position ──
            _raw_side = sig.get('side', 'FLAT')
            _sig_fire  = bool(sig.get('fire'))
            if is_tradeable and pos and _sig_fire and (
                (pos.direction == 'LONG'  and _raw_side == 'SELL') or
                (pos.direction == 'SHORT' and _raw_side == 'BUY')
            ):
                # Model flipped against the open position — warn the user.
                # The position SL/TP are correct for the ORIGINAL direction;
                # "BUY" here means the model now wants the opposite.
                _flip_label = 'BUY↑' if _raw_side == 'BUY' else 'SELL↓'
                sig_cell = f'[bold yellow]⚡ {_flip_label}[/]'
            elif is_tradeable:
                sig_cell = _signal_cell(sig)
            else:
                sig_cell = '[dim]watch[/]'

            grid.add_row(
                f'[dim]{idx}[/]' if not is_tradeable else str(idx),
                sym_cell,
                px_cell,
                sig_cell,
                dir_cell,
                _quality_cell(quality) if is_tradeable else '[dim]—[/]',
                rsi_cell,
                _regime_short(regime),
                sl_cell,
                tp_cell,
                pos_cell,
            )

        # ── Footer: open positions detail + last 5 closed ─────────────────
        open_lines: List[str] = []
        for pos in sorted(wallet.open_positions.values(), key=lambda p: p.entry_time):
            cur  = float(live_px.get(pos.symbol, pos.entry_price) or pos.entry_price)
            ppct = ((cur - pos.entry_price) / pos.entry_price * 100) if pos.direction == 'LONG' \
                   else ((pos.entry_price - cur) / pos.entry_price * 100)
            pu   = round(pos.position_value * ppct / 100, 2)
            arr, ds = _dir(pos.direction)
            col      = _pc(ppct)
            trail_tag = ' [bold yellow][T][/]' if engine._tp1_hit.get(pos.symbol) else ''
            open_lines.append(
                f"  [{ds}]{arr} {pos.symbol}[/]  "
                f"entry [white]{_px(pos.entry_price)}[/] → now [bold white]{_px(cur)}[/]  "
                f"[{col}]{ppct:+.2f}%  {pu:+.2f} USDT[/]  "
                f"SL [red]{_px(pos.stop_loss)}[/]  TP [green]{_px(pos.take_profit_1)}[/]  "
                f"edge [cyan]{pos.meta_confidence:.3f}[/]{trail_tag}  "
                f"[dim]{pos.entry_time[11:16]} UTC[/]"
            )

        closed_lines: List[str] = []
        for rec in sorted(wallet.trade_history,
                          key=lambda t: t.close_time or '', reverse=True)[:5]:
            arr, ds = _dir(rec.direction)
            col = _pc(rec.pnl_pct)
            closed_lines.append(
                f"  {_badge(rec.outcome)}  [{ds}]{arr} {rec.symbol}[/]  "
                f"[{col}]{rec.pnl_pct:+.2f}%  {rec.pnl_usdt:+.2f} USDT[/]  "
                f"[dim]{(rec.exit_reason or '').replace('_',' ')}[/]"
            )

        open_body   = "\n".join(open_lines)   if open_lines   else "  [dim]No open positions[/]"
        closed_body = "\n".join(closed_lines) if closed_lines else "  [dim]No closed trades yet[/]"

        footer = Panel(
            f"[bold cyan]● OPEN[/]  ({len(wallet.open_positions)})\n{open_body}\n\n"
            f"[bold white]✔ RECENT CLOSED[/]  ({len(wallet.trade_history)} total)\n{closed_body}",
            border_style='dim',
        )

        return Group(header, grid, footer)

    async def _run_with_display() -> None:
        scalp_bot = ScalpBot()
        scan_task  = asyncio.create_task(engine.run())

        async def _scalp_loop() -> None:
            executor = __import__('concurrent.futures', fromlist=['ThreadPoolExecutor']).ThreadPoolExecutor(max_workers=1)
            loop = asyncio.get_event_loop()
            while True:
                try:
                    await loop.run_in_executor(executor, scalp_bot.scan, engine.live_prices.copy())
                except Exception:
                    pass
                await asyncio.sleep(10)

        scalp_task = asyncio.create_task(_scalp_loop())

        with Live(_build_renderable(), screen=True, refresh_per_second=1) as live:
            try:
                while not scan_task.done():
                    try:
                        live.update(_build_renderable())
                    except Exception:
                        pass
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                pass
            finally:
                scalp_task.cancel()
                scan_task.cancel()
                await engine.shutdown()

    asyncio.run(_run_with_display())
