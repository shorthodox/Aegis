"""
formatter.py — Converts raw signal/trade dicts to Discord embed and WhatsApp text.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional


def _px(p: float) -> str:
    if p <= 0:     return "—"
    if p < 0.001:  return f"{p:.6f}"
    if p < 1:      return f"{p:.4f}"
    if p < 100:    return f"{p:.3f}"
    return f"{p:.2f}"


def _age_str(seconds: int) -> str:
    if seconds < 60:   return f"{seconds}s"
    if seconds < 3600: return f"{seconds // 60}m"
    h = seconds // 3600
    m = (seconds % 3600) // 60
    return f"{h}h {m}m"


# ── Entry notifications ────────────────────────────────────────────────────────

def format_entry_discord(sig: Dict[str, Any]) -> Dict[str, Any]:
    """Build a Discord webhook embed payload for a new signal entry."""
    direction = sig.get("direction", "?")
    symbol    = sig.get("symbol", "?")
    sym_short = symbol.replace("/USDT", "")
    conf      = float(sig.get("confidence", 0))
    confl     = float(sig.get("confluence_score", 0))
    price     = float(sig.get("current_price", 0))
    mode      = sig.get("mode", "—")
    tf        = sig.get("timeframe", "—")
    strategies= sig.get("top_strategies") or []
    ts        = sig.get("timestamp", "")
    guidance  = sig.get("guidance") or {}

    entry_z  = guidance.get("entry_zone") or {}
    sl       = guidance.get("stop_loss") or sig.get("stop_loss")
    tp       = guidance.get("take_profit") or {}
    hold_t   = guidance.get("hold_time", "—")
    rationale= guidance.get("rationale", "")
    tp1      = (tp.get("tp1") if tp else None) or sig.get("take_profit_1")
    tp2      = (tp.get("tp2") if tp else None) or sig.get("take_profit_2")
    tp3      = (tp.get("tp3") if tp else None) or sig.get("take_profit_3")

    is_buy = direction == "BUY"
    color  = 0x00C851 if is_buy else 0xFF4444
    arrow  = "▲" if is_buy else "▼"
    emoji  = "🟢" if is_buy else "🔴"
    n_ag   = round(confl * 25)

    fields: list = []
    if entry_z:
        fields.append({"name": "Entry Zone",  "value": f"{_px(float(entry_z.get('low', 0)))} – {_px(float(entry_z.get('high', 0)))}", "inline": True})
    else:
        fields.append({"name": "Entry Price", "value": _px(price), "inline": True})

    fields.append({"name": "Stop Loss",  "value": _px(float(sl or 0)), "inline": True})
    if tp1:
        fields.append({"name": "TP1", "value": _px(float(tp1)), "inline": True})
    if tp2:
        fields.append({"name": "TP2", "value": _px(float(tp2)), "inline": True})
    if tp3:
        fields.append({"name": "TP3", "value": _px(float(tp3)), "inline": True})

    fields.append({"name": "Confidence",  "value": f"{conf * 100:.1f}%",       "inline": True})
    fields.append({"name": "Confluence",  "value": f"{n_ag}/25 strategies",    "inline": True})
    fields.append({"name": "Mode",        "value": f"{mode} ({tf})",           "inline": True})
    if hold_t and hold_t != "—":
        fields.append({"name": "Est. Hold", "value": str(hold_t),              "inline": True})
    if strategies:
        fields.append({"name": "Top Strategies", "value": ", ".join(strategies[:3]), "inline": False})
    if rationale:
        fields.append({"name": "Rationale", "value": rationale[:250],          "inline": False})

    ts_fmt = ts[:19].replace("T", " ") if ts else datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    embed = {
        "title":       f"{emoji} {arrow} {direction}  {sym_short}/USDT",
        "description": "**AEGIS Signal** · Set a unique alert tone so you don't miss this entry!",
        "color":       color,
        "fields":      fields,
        "footer":      {"text": f"AEGIS AI Signal Bot  ·  {ts_fmt} UTC"},
    }
    return {"embeds": [embed]}


def format_entry_whatsapp(sig: Dict[str, Any]) -> str:
    """Build a WhatsApp plain-text message for a new signal entry."""
    direction = sig.get("direction", "?")
    symbol    = sig.get("symbol", "?")
    conf      = float(sig.get("confidence", 0))
    confl     = float(sig.get("confluence_score", 0))
    price     = float(sig.get("current_price", 0))
    mode      = sig.get("mode", "—")
    guidance  = sig.get("guidance") or {}

    sl   = guidance.get("stop_loss") or sig.get("stop_loss")
    tp   = guidance.get("take_profit") or {}
    tp1  = (tp.get("tp1") if tp else None) or sig.get("take_profit_1")
    tp2  = (tp.get("tp2") if tp else None) or sig.get("take_profit_2")
    hold = guidance.get("hold_time", "—")
    ts   = (sig.get("timestamp") or "")[:16].replace("T", " ")

    arrow = "📈" if direction == "BUY" else "📉"
    n_ag  = round(confl * 25)

    lines = [
        f"{arrow} *AEGIS SIGNAL — {direction} {symbol}*",
        f"⚠️ Set a unique notification tone so you don't miss this!",
        f"",
        f"💰 Entry: {_px(price)}",
        f"🛑 Stop Loss: {_px(float(sl or 0))}",
    ]
    if tp1:
        lines.append(f"🎯 TP1: {_px(float(tp1))}")
    if tp2:
        lines.append(f"🎯 TP2: {_px(float(tp2))}")
    lines += [
        f"",
        f"📊 Confidence: {conf * 100:.1f}%",
        f"🤝 Confluence: {n_ag}/25 strategies",
        f"⚙️ Mode: {mode}",
    ]
    if hold and hold != "—":
        lines.append(f"⏱️ Est. Hold: {hold}")
    lines += [
        f"",
        f"🕐 {ts} UTC",
        f"— AEGIS AI Signal Bot",
    ]
    return "\n".join(lines)


# ── Exit notifications ─────────────────────────────────────────────────────────

def format_exit_discord(
    symbol:       str,
    direction:    str,
    outcome:      str,
    pnl_pct:      float,
    hold_seconds: int,
    exit_reason:  str = "",
) -> Dict[str, Any]:
    """Build a Discord webhook payload for a closed position."""
    is_win   = outcome == "WIN"
    color    = 0x00C851 if is_win else 0xFF4444
    badge    = "✅ WIN" if is_win else "❌ LOSS"
    arrow    = "▲" if direction == "BUY" else "▼"
    sym_s    = symbol.replace("/USDT", "")
    reason   = (exit_reason or "").replace("_", " ")
    hold_s   = _age_str(hold_seconds)
    ts_fmt   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    embed = {
        "title":       f"{badge}  {arrow} {sym_s}/USDT CLOSED",
        "description": f"Position closed — {reason}",
        "color":       color,
        "fields": [
            {"name": "PnL",           "value": f"{pnl_pct:+.2f}%", "inline": True},
            {"name": "Direction",     "value": direction,           "inline": True},
            {"name": "Hold Duration", "value": hold_s,              "inline": True},
            {"name": "Exit Reason",   "value": reason or "—",       "inline": True},
        ],
        "footer": {"text": f"AEGIS AI Signal Bot  ·  {ts_fmt} UTC"},
    }
    return {"embeds": [embed]}


def format_exit_whatsapp(
    symbol:       str,
    direction:    str,
    outcome:      str,
    pnl_pct:      float,
    hold_seconds: int,
    exit_reason:  str = "",
) -> str:
    """Build a WhatsApp plain-text message for a closed position."""
    is_win = outcome == "WIN"
    badge  = "✅ WIN" if is_win else "❌ LOSS"
    arrow  = "📈" if direction == "BUY" else "📉"
    reason = (exit_reason or "").replace("_", " ")
    hold_s = _age_str(hold_seconds)
    ts_fmt = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")

    lines = [
        f"{badge} *AEGIS SIGNAL CLOSED — {arrow} {direction} {symbol}*",
        f"",
        f"📈 PnL: {pnl_pct:+.2f}%",
        f"⏱️ Held for: {hold_s}",
        f"🚪 Reason: {reason or '—'}",
        f"",
        f"🕐 {ts_fmt} UTC",
        f"— AEGIS AI Signal Bot",
    ]
    return "\n".join(lines)


# ── Telegram formatting (Markdown) ────────────────────────────────────────────
# Telegram Markdown: *bold*, `code`, plain text — avoid nested symbols

def format_entry_telegram(sig: Dict[str, Any]) -> str:
    """Build a Telegram Markdown message for a new signal entry."""
    direction = sig.get("direction", "?")
    symbol    = sig.get("symbol", "?")
    conf      = float(sig.get("confidence", 0))
    confl     = float(sig.get("confluence_score", 0))
    price     = float(sig.get("current_price", 0))
    mode      = sig.get("mode", "—")
    strategies= sig.get("top_strategies") or []
    guidance  = sig.get("guidance") or {}

    sl   = guidance.get("stop_loss") or sig.get("stop_loss")
    tp   = guidance.get("take_profit") or {}
    tp1  = (tp.get("tp1") if tp else None) or sig.get("take_profit_1")
    tp2  = (tp.get("tp2") if tp else None) or sig.get("take_profit_2")
    hold = guidance.get("hold_time", "—")
    ts   = (sig.get("timestamp") or "")[:16].replace("T", " ")

    arrow  = "BUY" if direction == "BUY" else "SELL"
    flag   = "GREEN" if direction == "BUY" else "RED"
    n_ag   = round(confl * 25)
    dir_icon = "📈" if direction == "BUY" else "📉"

    lines = [
        f"{dir_icon} *AEGIS {arrow} — {symbol}*",
        f"",
        f"💰 *Entry:* `{_px(price)}`",
        f"🛑 *Stop Loss:* `{_px(float(sl or 0))}`",
    ]
    if tp1:
        lines.append(f"🎯 *TP1:* `{_px(float(tp1))}`")
    if tp2:
        lines.append(f"🎯 *TP2:* `{_px(float(tp2))}`")
    lines += [
        f"",
        f"📊 *Confidence:* {conf * 100:.0f}%",
        f"🤝 *Confluence:* {n_ag}/25 strategies",
        f"⚙️ *Mode:* {mode}",
    ]
    if hold and hold != "—":
        lines.append(f"⏱ *Est. Hold:* {hold}")
    if strategies:
        lines.append(f"💡 *Strategies:* {', '.join(strategies[:3])}")
    lines += [
        f"",
        f"🕐 {ts} UTC",
        f"_AEGIS AI Signal Bot_",
    ]
    return "\n".join(lines)


def format_pending_telegram(sig: Dict[str, Any]) -> str:
    """Build a Telegram message for a signal that has ARMED but not yet fired.

    A pending signal has cleared the JACKDLM direction gates; the engine is
    holding it (Guard M) until price reaches its S/R level and 3x5m candles
    confirm. There is no entry / SL / TP yet — those are set the moment it
    actually fires, which is when the normal entry alert goes out.
    """
    direction = sig.get("direction", "?")
    symbol    = sig.get("symbol", "?")
    target    = sig.get("pending_target") or sig.get("target")
    role      = "support" if direction == "BUY" else "resistance"
    dir_icon  = "📈" if direction == "BUY" else "📉"
    arrow     = "LONG" if direction == "BUY" else "SHORT"
    ts        = (sig.get("timestamp") or "")[:16].replace("T", " ")

    lines = [
        f"⏳ *AEGIS WATCHING — {symbol}*",
        f"",
        f"{dir_icon} Armed *{arrow}* — waiting to fire",
    ]
    if target:
        try:
            lines.append(f"🎯 *At {role}:* `{_px(float(target))}`")
        except (TypeError, ValueError):
            pass
    lines += [
        f"",
        f"Entry, SL & TPs are set the moment price reaches the level and 3×5m candles confirm.",
        f"",
        f"🕐 {ts} UTC",
        f"_AEGIS AI Signal Bot_",
    ]
    return "\n".join(lines)


def format_exit_telegram(
    symbol:       str,
    direction:    str,
    outcome:      str,
    pnl_pct:      float,
    hold_seconds: int,
    exit_reason:  str = "",
) -> str:
    """Build a Telegram Markdown message for a closed position."""
    is_win = outcome == "WIN"
    badge  = "✅ WIN" if is_win else "❌ LOSS"
    arrow  = "📈" if direction == "BUY" else "📉"
    reason = (exit_reason or "").replace("_", " ")
    hold_s = _age_str(hold_seconds)
    ts_fmt = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")

    lines = [
        f"{badge} *AEGIS CLOSED — {arrow} {direction} {symbol}*",
        f"",
        f"📈 *PnL:* `{pnl_pct:+.2f}%`",
        f"⏱ *Held:* {hold_s}",
        f"🚪 *Reason:* {reason or '—'}",
        f"",
        f"🕐 {ts_fmt} UTC",
        f"_AEGIS AI Signal Bot_",
    ]
    return "\n".join(lines)
