"""
discord_notifier.py — Posts signal embeds to a Discord webhook URL.

No bot token required — uses Discord's incoming webhook API directly.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict

log = logging.getLogger(__name__)


def send_discord(webhook_url: str, payload: Dict[str, Any], timeout: int = 10) -> bool:
    """POST a Discord embed payload to the webhook URL. Returns True on success."""
    if not webhook_url:
        return False
    try:
        import requests
        r = requests.post(
            webhook_url,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        if r.status_code in (200, 204):
            log.info("[Discord] Notification sent")
            return True
        log.warning(f"[Discord] HTTP {r.status_code}: {r.text[:200]}")
        return False
    except ImportError:
        log.warning("[Discord] requests package not installed — pip install requests")
        return False
    except Exception as exc:
        log.warning(f"[Discord] Send failed: {exc}")
        return False
