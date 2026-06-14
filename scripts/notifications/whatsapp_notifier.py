"""
whatsapp_notifier.py — Sends signal messages via Twilio WhatsApp API.

Prerequisites:
    pip install twilio

    1. Sign up at https://twilio.com — get Account SID + Auth Token
    2. Join the Twilio WhatsApp Sandbox (or buy a WhatsApp-enabled number)
    3. From number format: +14155238886 (Twilio sandbox)
    4. To number format:   +91XXXXXXXXXX (user's WhatsApp-registered mobile)

Tell users to set a UNIQUE notification tone for AEGIS alerts in WhatsApp
Settings → Notifications → Tone — so signals stand out from regular messages.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def send_whatsapp(
    account_sid: str,
    auth_token:  str,
    from_number: str,
    to_number:   str,
    body:        str,
    timeout:     int = 15,
) -> bool:
    """Send a WhatsApp message via Twilio. Returns True on success."""
    if not all([account_sid, auth_token, from_number, to_number]):
        log.debug("[WhatsApp] Credentials not configured — skipping")
        return False
    try:
        from twilio.rest import Client
        client  = Client(account_sid, auth_token)
        from_wa = from_number if from_number.startswith("whatsapp:") else f"whatsapp:{from_number}"
        to_wa   = to_number   if to_number.startswith("whatsapp:")   else f"whatsapp:{to_number}"
        msg     = client.messages.create(from_=from_wa, to=to_wa, body=body)
        log.info(f"[WhatsApp] Sent SID={msg.sid}")
        return True
    except ImportError:
        log.warning("[WhatsApp] twilio package not installed — pip install twilio")
        return False
    except Exception as exc:
        log.warning(f"[WhatsApp] Send failed: {exc}")
        return False
