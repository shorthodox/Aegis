"""
telegram_notifier.py — Sends signal messages via Telegram Bot API.

No extra packages needed — uses the requests library (already a dependency).

Setup:
    1. Message @BotFather on Telegram → /newbot → get your bot token
    2. Start a chat with your bot (send it /start)
    3. Message @userinfobot → get your numeric Chat ID
    4. Enter both in the dashboard Settings → Notifications
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

_API = "https://api.telegram.org/bot{token}/sendMessage"


def send_telegram(
    bot_token: str,
    chat_id:   str,
    text:      str,
    timeout:   int = 10,
) -> bool:
    """Send a Markdown message via Telegram Bot API. Returns True on success."""
    if not bot_token or not chat_id:
        log.debug("[Telegram] Not configured — skipping")
        return False
    try:
        import requests
        r = requests.post(
            _API.format(token=bot_token),
            json={
                "chat_id":    chat_id,
                "text":       text,
                "parse_mode": "Markdown",
            },
            timeout=timeout,
        )
        data = r.json()
        if data.get("ok"):
            log.info(f"[Telegram] Sent message_id={data['result']['message_id']}")
            return True
        log.warning(f"[Telegram] API error: {data.get('description', data)}")
        return False
    except ImportError:
        log.warning("[Telegram] requests package not installed — pip install requests")
        return False
    except Exception as exc:
        log.warning(f"[Telegram] Send failed: {exc}")
        return False
