"""
dispatcher.py — Thread-safe notification dispatcher with rate limiting and quiet hours.

Settings file: data/notification_settings.json
{
  "enabled":             true,
  "telegram_bot_token":  "7123456789:AAHxxxxxx...",
  "telegram_chat_id":    "123456789",
  "discord_webhook_url": "https://discord.com/api/webhooks/...",
  "twilio_account_sid":  "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
  "twilio_auth_token":   "your_auth_token",
  "whatsapp_from":       "+14155238886",
  "whatsapp_to":         "+91XXXXXXXXXX",
  "min_confidence":      0.65,
  "allowed_directions":  ["BUY", "SELL"],
  "allowed_modes":       ["scalping", "scalping_15m", "intraday", "swing"],
  "quiet_hours":         {"start": "23:00", "end": "06:00"},
  "max_alerts_per_hour": 5
}

Note: quiet_hours supports overnight ranges (e.g. 23:00 → 06:00).
      Set both to "" to disable quiet hours.
"""
from __future__ import annotations

import json
import logging
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, Optional

from scripts.notifications.formatter import (
    format_entry_discord,
    format_entry_telegram,
    format_entry_whatsapp,
    format_exit_discord,
    format_exit_telegram,
    format_exit_whatsapp,
    format_pending_telegram,
)
from scripts.notifications.discord_notifier import send_discord
from scripts.notifications.telegram_notifier import send_telegram
from scripts.notifications.whatsapp_notifier import send_whatsapp

log = logging.getLogger(__name__)

_ROOT          = Path(__file__).resolve().parent.parent.parent
_SETTINGS_PATH = _ROOT / "data" / "notification_settings.json"

_DEFAULT_SETTINGS: Dict[str, Any] = {
    "enabled":             True,
    "telegram_bot_token":  "",
    "telegram_chat_id":    "",
    "discord_webhook_url": "",
    "twilio_account_sid":  "",
    "twilio_auth_token":   "",
    "whatsapp_from":       "",
    "whatsapp_to":         "",
    # 0.0 = notify EVERY fired signal. The old 0.65 silently dropped entries
    # firing at edge_score 60-65 (the engine fires at edge >= 60, so confidence
    # = edge/100 = 0.60-0.65 fell UNDER this bar) — "ETH fired but no Telegram".
    # The engine's gate cascade IS the quality filter; the notification must not
    # re-gate a fired signal on a mismatched threshold. Set > 0 only to opt into
    # muting low-edge alerts.
    "min_confidence":      0.0,
    "allowed_directions":  ["BUY", "SELL"],
    "allowed_modes":       ["scalping", "scalping_15m", "intraday", "swing", "live"],
    # Crypto trades 24/7 and subscribers expect every signal live —
    # quiet hours are opt-in (set both to enable, e.g. 23:00/06:00).
    "quiet_hours":         {"start": "", "end": ""},
    # Entry-alert budget per sliding hour. Exit/outcome alerts are
    # CRITICAL (accountability) and never count against this budget.
    # AEGIS is a SIGNAL SERVICE — subscribers pay to receive EVERY signal, so
    # this is set to an effectively-unlimited ceiling (a burst guard against a
    # runaway bug, not a product limit).  The old value of 20 made entry alerts
    # stop after ~20/hr, which read as "the engine stopped firing after 19".
    "max_alerts_per_hour": 200,
    # PENDING (armed-but-not-fired) heads-ups — a signal that cleared the
    # JACKDLM direction gates and is waiting for price to reach its S/R level.
    # Toggle off to silence them; they use their OWN hourly budget so they can
    # never starve real entry alerts.
    "notify_pending":       True,
    "max_pending_per_hour": 30,
}


class NotificationDispatcher:
    """Fire-and-forget notification dispatcher.

    • Loads settings from disk on every call (live config changes without restart).
    • Rate limiting: max N alerts per 60-minute sliding window.
    • Quiet hours: suppresses during configured time window.
    • Sends Discord + WhatsApp in a background thread pool (non-blocking).
    """

    def __init__(self) -> None:
        self._pool        = ThreadPoolExecutor(max_workers=2, thread_name_prefix="notif")
        self._lock        = threading.Lock()
        self._timestamps: Deque[float] = deque()
        # Separate sliding-hour budget for PENDING heads-ups so they never
        # consume the entry budget above.
        self._pending_ts: Deque[float] = deque()

    # ── Settings ──────────────────────────────────────────────────────────────

    def _load_settings(self) -> Dict[str, Any]:
        if not _SETTINGS_PATH.exists():
            return dict(_DEFAULT_SETTINGS)
        try:
            with open(_SETTINGS_PATH) as f:
                data = json.load(f)
            return {**_DEFAULT_SETTINGS, **data}
        except Exception:
            return dict(_DEFAULT_SETTINGS)

    @staticmethod
    def save_settings(settings: Dict[str, Any]) -> None:
        """Write settings dict to disk (called from API endpoint)."""
        _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _SETTINGS_PATH.write_text(json.dumps({**_DEFAULT_SETTINGS, **settings}, indent=2))

    # ── Guards ────────────────────────────────────────────────────────────────

    def _in_quiet_hours(self, qh: Dict[str, str]) -> bool:
        start = qh.get("start", "")
        end   = qh.get("end", "")
        if not start or not end:
            return False
        now_t = datetime.now(timezone.utc).strftime("%H:%M")
        if start > end:  # overnight window, e.g. 23:00 → 06:00
            return now_t >= start or now_t < end
        return start <= now_t < end

    def _rate_limited(self, max_per_hr: int) -> bool:
        now_ts = datetime.now(timezone.utc).timestamp()
        with self._lock:
            cutoff = now_ts - 3600
            while self._timestamps and self._timestamps[0] < cutoff:
                self._timestamps.popleft()
            return len(self._timestamps) >= max_per_hr

    def _passes_signal_filter(self, cfg: Dict[str, Any], sig: Dict[str, Any]) -> bool:
        direction = sig.get("direction", "")
        if direction not in cfg.get("allowed_directions", ["BUY", "SELL"]):
            return False
        conf = float(sig.get("confidence", 0))
        if conf < float(cfg.get("min_confidence", 0.0)):   # 0 default — see _DEFAULT_SETTINGS
            return False
        mode = sig.get("mode", "")
        allowed_modes = cfg.get("allowed_modes", [])
        if mode and allowed_modes and mode not in allowed_modes:
            return False
        return True

    def _should_send(
        self,
        cfg:      Dict[str, Any],
        sig:      Optional[Dict[str, Any]] = None,
        critical: bool = False,
    ) -> bool:
        """critical=True (exit/outcome alerts): only the global enabled flag
        applies — trade outcomes must always reach subscribers, regardless of
        quiet hours or the entry-alert rate budget."""
        if not cfg.get("enabled", True):
            return False
        if critical:
            return True
        if self._in_quiet_hours(cfg.get("quiet_hours") or {}):
            log.debug("[Notif] Quiet hours active — skipping")
            return False
        max_per_hr = int(cfg.get("max_alerts_per_hour", 20))
        if self._rate_limited(max_per_hr):
            log.info(f"[Notif] Rate limit ({max_per_hr}/hr) reached — skipping")
            return False
        if sig and not self._passes_signal_filter(cfg, sig):
            return False
        return True

    def _record_send(self) -> None:
        with self._lock:
            self._timestamps.append(datetime.now(timezone.utc).timestamp())

    # ── Internal send ─────────────────────────────────────────────────────────

    @staticmethod
    def _tg_send_all(tg_text: str) -> None:
        """Send to every connected Telegram chat_id using server-side bot token."""
        import os as _os, json as _json
        token = _os.getenv("TELEGRAM_BOT_TOKEN", "")
        if not token or not tg_text:
            return
        conn_path = _ROOT / "data" / "telegram_connections.json"
        if not conn_path.exists():
            return
        try:
            connections: Dict[str, str] = _json.loads(conn_path.read_text())
        except Exception:
            return
        for _email, chat_id in connections.items():
            if chat_id:
                send_telegram(token, chat_id, tg_text)

    def _do_send(
        self,
        cfg:             Dict[str, Any],
        discord_payload: Optional[Dict[str, Any]],
        tg_text:         Optional[str],
        wa_text:         Optional[str],
    ) -> None:
        # Telegram — server-side bot token, all connected users
        if tg_text:
            self._tg_send_all(tg_text)
        # Discord webhook
        webhook = cfg.get("discord_webhook_url", "")
        if webhook and discord_payload:
            send_discord(webhook, discord_payload)
        # WhatsApp via Twilio
        sid = cfg.get("twilio_account_sid", "")
        tok = cfg.get("twilio_auth_token", "")
        frm = cfg.get("whatsapp_from", "")
        to  = cfg.get("whatsapp_to", "")
        if sid and tok and frm and to and wa_text:
            send_whatsapp(sid, tok, frm, to, wa_text)

    # ── Public API ────────────────────────────────────────────────────────────

    def send_entry(self, sig: Dict[str, Any]) -> None:
        """Dispatch entry notification for a new signal (non-blocking)."""
        cfg = self._load_settings()
        if not self._should_send(cfg, sig):
            return
        self._record_send()   # only entries consume the hourly alert budget
        dp  = format_entry_discord(sig)
        tg  = format_entry_telegram(sig)
        wa  = format_entry_whatsapp(sig)
        self._pool.submit(self._do_send, cfg, dp, tg, wa)

    def send_pending(self, sig: Dict[str, Any]) -> None:
        """Dispatch a heads-up that a signal has ARMED (Guard M) but not fired.

        Informational, Telegram-only: respects the global enable flag, quiet
        hours, and the `notify_pending` toggle, but draws on a SEPARATE hourly
        budget (`max_pending_per_hour`) so a burst of pending signals can never
        consume the entry budget and starve real entry alerts. Edge-triggered
        de-duplication (one alert per pending episode) is the caller's job.
        """
        cfg = self._load_settings()
        if not cfg.get("enabled", True) or not cfg.get("notify_pending", True):
            return
        if self._in_quiet_hours(cfg.get("quiet_hours") or {}):
            return
        max_pend = int(cfg.get("max_pending_per_hour", 30))
        now_ts   = datetime.now(timezone.utc).timestamp()
        with self._lock:
            cutoff = now_ts - 3600
            while self._pending_ts and self._pending_ts[0] < cutoff:
                self._pending_ts.popleft()
            if len(self._pending_ts) >= max_pend:
                log.info(f"[Notif] Pending budget ({max_pend}/hr) reached — skipping")
                return
            self._pending_ts.append(now_ts)
        tg = format_pending_telegram(sig)
        self._pool.submit(self._do_send, cfg, None, tg, None)

    def send_exit(
        self,
        symbol:       str,
        direction:    str,
        outcome:      str,
        pnl_pct:      float,
        hold_seconds: int,
        exit_reason:  str = "",
    ) -> None:
        """Dispatch exit notification when a position closes (non-blocking).

        Exit alerts are CRITICAL: they bypass quiet hours and the hourly
        rate budget so every WIN/LOSS outcome reaches subscribers — the
        public accountability these notifications exist for.
        """
        cfg = self._load_settings()
        if not self._should_send(cfg, critical=True):
            return
        dp = format_exit_discord(symbol, direction, outcome, pnl_pct, hold_seconds, exit_reason)
        tg = format_exit_telegram(symbol, direction, outcome, pnl_pct, hold_seconds, exit_reason)
        wa = format_exit_whatsapp(symbol, direction, outcome, pnl_pct, hold_seconds, exit_reason)
        self._pool.submit(self._do_send, cfg, dp, tg, wa)

    def test_send(self) -> Dict[str, bool]:
        """Send a test ping to each configured channel. Returns {telegram, discord, whatsapp}."""
        cfg = self._load_settings()
        test_sig = {
            "symbol": "BTC/USDT", "direction": "BUY", "confidence": 0.82,
            "confluence_score": 0.72, "current_price": 105000.0,
            "mode": "scalping", "timeframe": "5m",
            "top_strategies": ["vwap_bounce", "ema_cross_fast"],
            "guidance": {
                "entry_zone":  {"low": 104900.0, "high": 105100.0},
                "stop_loss":   103500.0,
                "take_profit": {"tp1": 106500.0, "tp2": 108000.0, "tp3": 110000.0},
                "hold_time":   "20-60 min",
                "rationale":   "Test ping from AEGIS — if you see this, notifications are working!",
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        results: Dict[str, bool] = {}
        # Telegram
        tg_token = cfg.get("telegram_bot_token", "")
        tg_chat  = cfg.get("telegram_chat_id", "")
        if tg_token and tg_chat:
            results["telegram"] = send_telegram(tg_token, tg_chat, format_entry_telegram(test_sig))
        # Discord
        webhook = cfg.get("discord_webhook_url", "")
        if webhook:
            results["discord"] = send_discord(webhook, format_entry_discord(test_sig))
        # WhatsApp
        sid = cfg.get("twilio_account_sid", "")
        tok = cfg.get("twilio_auth_token", "")
        frm = cfg.get("whatsapp_from", "")
        to  = cfg.get("whatsapp_to", "")
        if sid and tok and frm and to:
            results["whatsapp"] = send_whatsapp(sid, tok, frm, to, format_entry_whatsapp(test_sig))
        return results


# ── Module-level singleton ─────────────────────────────────────────────────────

_notifier:     Optional[NotificationDispatcher] = None
_notifier_lock = threading.Lock()


def get_notifier() -> NotificationDispatcher:
    global _notifier
    if _notifier is None:
        with _notifier_lock:
            if _notifier is None:
                _notifier = NotificationDispatcher()
    return _notifier
