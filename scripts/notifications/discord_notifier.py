"""
discord_notifier.py — Posts signal embeds to a Discord webhook URL.

No bot token required — uses Discord's incoming webhook API directly.

A webhook URL is a BEARER CREDENTIAL: anyone holding one can post as the bot into
that channel, with no auth step. They are read from the environment (see
dispatcher._ENV_KEYS) and must never be committed — this repository is public and
has leaked committed secrets before.
"""
from __future__ import annotations

import json
import logging
import random
import time
from typing import Any, Dict

log = logging.getLogger(__name__)

# Discord's webhook bucket is roughly 30 requests per minute per webhook. These
# are deliberately patient rather than fast: a notification is never worth
# blocking the trading loop for, and the dispatcher already calls this on a
# background thread pool.
MAX_ATTEMPTS = 4
BASE_BACKOFF = 0.75          # seconds; doubled each attempt, plus jitter
MAX_BACKOFF  = 8.0
RETRY_STATUS = {429, 500, 502, 503, 504}


def send_discord(webhook_url: str, payload: Dict[str, Any], timeout: int = 10) -> bool:
    """POST a Discord embed payload to the webhook URL. Returns True on success.

    Retries on rate limits and transient server errors, honouring Discord's own
    `retry_after` when it sends one. Never raises: a failed Discord post must not
    be able to interrupt an exit, a fill, or anything else that matters.
    """
    if not webhook_url:
        return False
    try:
        import requests
    except ImportError:
        log.warning("[Discord] requests package not installed — pip install requests")
        return False

    body = json.dumps(payload)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            r = requests.post(
                webhook_url,
                data=body,
                headers={"Content-Type": "application/json"},
                timeout=timeout,
            )
        except Exception as exc:
            # A timeout or connection error is worth one more try.
            if attempt == MAX_ATTEMPTS:
                log.warning(f"[Discord] send failed after {attempt} attempts: {exc!r}")
                return False
            _sleep(attempt)
            continue

        if r.status_code in (200, 204):
            return True

        if r.status_code not in RETRY_STATUS or attempt == MAX_ATTEMPTS:
            # 4xx other than 429 will never succeed on a retry — a deleted
            # webhook (404) or a malformed embed (400) needs a human, so say
            # which it was rather than retrying into the same wall.
            log.warning(f"[Discord] HTTP {r.status_code}: {r.text[:200]}")
            return False

        # Discord tells us how long to wait on a 429. Believe it over our own
        # backoff, which is only a guess.
        wait = None
        if r.status_code == 429:
            try:
                wait = float((r.json() or {}).get("retry_after", 0)) or None
            except Exception:
                wait = None
            log.info(f"[Discord] rate limited, waiting {wait or 'backoff'}s")
        _sleep(attempt, override=wait)

    return False


def _sleep(attempt: int, override: float | None = None) -> None:
    if override is not None:
        time.sleep(min(override, MAX_BACKOFF))
        return
    # Full jitter — several webhooks retrying in lockstep is how a rate limit
    # becomes a thundering herd.
    ceiling = min(BASE_BACKOFF * (2 ** (attempt - 1)), MAX_BACKOFF)
    time.sleep(random.uniform(0, ceiling))
