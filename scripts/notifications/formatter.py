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


def format_observation_telegram(sig: Dict[str, Any]) -> str:
    """Build a Telegram Markdown message for a signal under paper observation."""
    direction = sig.get("direction", "BUY")
    symbol    = sig.get("symbol", "?")
    price     = float(sig.get("current_price") or sig.get("price") or 0)
    sl        = float(sig.get("stop_loss") or 0)
    tp1       = float(sig.get("take_profit_1") or 0)
    reason    = sig.get("paper_reason") or "Internal paper validation"
    dir_icon  = "📈" if direction == "BUY" else "📉"
    ts        = (sig.get("timestamp") or "")[:16].replace("T", " ")

    lines = [
        f"🧪 *TRADABLE · UNDER OBSERVATION — {symbol}*",
        f"",
        f"{dir_icon} Direction: *{direction}*",
        f"💰 *Entry:* `{_px(price)}`",
        f"🛑 *Stop Loss:* `{_px(sl)}`",
    ]
    if tp1 > 0:
        lines.append(f"🎯 *TP1:* `{_px(tp1)}`")
    lines += [
        f"📋 *Status:* {reason}",
        f"",
        f"🕐 {ts} UTC",
        f"_AEGIS AI Signal Bot_",
    ]
    return "\n".join(lines)


def format_blocked_telegram(sig: Dict[str, Any]) -> str:
    """Build a Telegram Markdown message for an unfired/blocked model lean."""
    direction = sig.get("direction", "BUY")
    symbol    = sig.get("symbol", "?")
    price     = float(sig.get("current_price") or sig.get("price") or 0)
    reason    = sig.get("structure_reason") or sig.get("blocked_reason") or "Engine guard block"
    dir_icon  = "📈" if direction == "BUY" else "📉"
    ts        = (sig.get("timestamp") or "")[:16].replace("T", " ")

    lines = [
        f"🚫 *UNFIRED · BLOCKED — {symbol}*",
        f"",
        f"{dir_icon} Model Lean: *{direction}* @ `{_px(price)}`",
        f"🛡 *Guard Reason:* {reason}",
        f"",
        f"The model detected a potential lean but AEGIS engine guards prevented entry.",
        f"",
        f"🕐 {ts} UTC",
        f"_AEGIS AI Signal Bot_",
    ]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Refusals and position lifecycle
#
# The desk refuses far more than it takes — roughly 200 setups a scan — and each
# refusal already carries the reason it was refused. Publishing those is the
# clearest evidence the glass-box claim is real: anyone can read what was turned
# down and why, not just the trades that worked.
# ═══════════════════════════════════════════════════════════════════════════

def format_refusal_discord(
    symbol:    str,
    side:      str,
    reason:    str,
    stage:     str = "",
    setup:     str = "",
    price:     float = 0.0,
    quality:   Optional[float] = None,
) -> Dict[str, Any]:
    """Build a Discord embed for a setup the desk declined to take."""
    sym_s = symbol.replace("/USDT", "")
    side  = (side or "").upper()
    arrow = "▲" if side == "BUY" else ("▼" if side == "SELL" else "•")
    # Amber, not red. A refusal is not a loss — it is the system working, and
    # colouring it like a failure teaches subscribers the wrong thing.
    color = 0xE8A06A
    ts    = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    fields: list = []
    if side in ("BUY", "SELL"):
        fields.append({"name": "Side", "value": f"{arrow} {side}", "inline": True})
    if setup:
        fields.append({"name": "Setup", "value": setup.replace("_", " "), "inline": True})
    if stage:
        fields.append({"name": "Refused at", "value": stage, "inline": True})
    if price:
        fields.append({"name": "Price", "value": _px(price), "inline": True})
    if quality is not None:
        fields.append({"name": "Quality", "value": f"{float(quality):.0f}/100", "inline": True})

    embed = {
        "title":       f"⊘  {sym_s}/USDT NOT TAKEN",
        "description": reason or "no reason recorded",
        "color":       color,
        "fields":      fields,
        "footer":      {"text": f"AEGIS · refused  ·  {ts} UTC"},
    }
    return {"embeds": [embed]}


def format_position_open_discord(
    symbol:    str,
    side:      str,
    entry:     float,
    stop:      float,
    targets:   Optional[list] = None,
    size_usdt: float = 0.0,
    reason:    str = "",
    setup:     str = "",
    r_net:     Optional[float] = None,
) -> Dict[str, Any]:
    """Build a Discord embed for a position that has just opened.

    This is the claim being made BEFORE the outcome is known, which is the whole
    pitch — so it carries the reasoning, not just the numbers.
    """
    sym_s = symbol.replace("/USDT", "")
    side  = (side or "").upper()
    is_buy = side == "BUY"
    arrow = "▲" if is_buy else "▼"
    color = 0x3B82F6            # blue — neutral; the outcome is not known yet
    ts    = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    risk_pct = (abs(entry - stop) / entry * 100.0) if (entry and stop) else 0.0
    fields: list = [
        {"name": "Direction", "value": f"{arrow} {'LONG' if is_buy else 'SHORT'}", "inline": True},
        {"name": "Entry",     "value": _px(entry), "inline": True},
        {"name": "Stop",      "value": f"{_px(stop)}  ({risk_pct:.2f}%)", "inline": True},
    ]
    tps = [t for t in (targets or []) if t]
    if tps:
        fields.append({"name": "Targets",
                       "value": " → ".join(_px(float(t)) for t in tps[:5]),
                       "inline": False})
    if size_usdt:
        fields.append({"name": "Size", "value": f"{size_usdt:.0f} USDT", "inline": True})
    if r_net is not None:
        fields.append({"name": "Net R:R", "value": f"{float(r_net):.2f}", "inline": True})
    if setup:
        fields.append({"name": "Setup", "value": setup.replace("_", " "), "inline": True})

    embed = {
        "title":       f"◆  {sym_s}/USDT OPENED",
        "description": reason or "position opened",
        "color":       color,
        "fields":      fields,
        "footer":      {"text": f"AEGIS · open position  ·  {ts} UTC"},
    }
    return {"embeds": [embed]}
