#!/usr/bin/env python
"""Fire one representative event into each of the four Discord channels.

Run this BEFORE relying on the wiring live. It proves four separate things that
can each fail on their own: the URL is set, Discord accepts it, the embed renders,
and the router sends it to the channel you expect.

    # see what would be sent, send nothing
    railway run --service web -- python scripts/check_discord_channels.py

    # actually post
    railway run --service web -- python scripts/check_discord_channels.py --send

    # one channel only
    railway run --service web -- python scripts/check_discord_channels.py --send --only refusals

Run it through `railway run` so it picks up the AEGIS_DISCORD_WEBHOOK_* variables
from the service. The URLs are credentials and are never printed — only whether
each one is configured.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.notifications.dispatcher import NotificationDispatcher      # noqa: E402
from scripts.notifications.discord_notifier import send_discord          # noqa: E402
from scripts.notifications.formatter import (                            # noqa: E402
    format_entry_discord,
    format_exit_discord,
    format_position_open_discord,
    format_refusal_discord,
)

SYM = "TEST/USDT"


def _samples():
    """One event per channel, shaped exactly as the live callers shape them."""
    return {
        "signals": format_entry_discord({
            "symbol": SYM, "direction": "BUY", "confidence": 0.78,
            "confluence_score": 0.0, "current_price": 1.2345, "mode": "live",
            "timeframe": "1h", "top_strategies": [], "atr": 0.02,
            "risk_reward": 2.31, "stop_loss": 1.2100,
            "take_profit_1": 1.2600, "take_profit_2": 1.2800,
            "take_profit_3": 1.3000, "take_profit_4": 1.3200,
            "take_profit_5": 1.3400, "guidance": {}, "timestamp": "",
        }),
        "records": format_exit_discord(
            SYM, "BUY", "WIN", 1.42, 4260, "TP GIVEBACK",
        ),
        "refusals": format_refusal_discord(
            SYM, "SELL",
            "short at 6% of the whole structure — the local range says a bounce "
            "to fade, the chart says the lows. A short is taken in the upper 80%",
            stage="location", setup="RANGE_FADE", price=1.2345, quality=62,
        ),
        "positions": format_position_open_discord(
            SYM, "BUY", entry=1.2345, stop=1.2100,
            targets=[1.2600, 1.2800, 1.3000, 1.3200, 1.3400],
            size_usdt=1700, reason="three consecutive 5m candles turned and the "
                                   "level was never reached — taken at the market",
            setup="TREND_PULLBACK", r_net=2.31,
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--send", action="store_true",
                    help="actually post (default is a dry run)")
    ap.add_argument("--only", default="", help="one of: signals records refusals positions")
    args = ap.parse_args()

    cfg = NotificationDispatcher._load_settings(NotificationDispatcher)
    samples = _samples()
    channels = [args.only] if args.only else list(NotificationDispatcher._CHANNELS)

    print(f"mode: {'SEND' if args.send else 'DRY RUN — nothing will post'}\n")
    print(f"{'channel':11} {'configured':11} {'title':44} result")
    print("-" * 88)

    failures = 0
    for ch in channels:
        key = NotificationDispatcher._CHANNELS.get(ch)
        if key is None:
            print(f"{ch:11} unknown channel")
            failures += 1
            continue
        url = cfg.get(key, "")
        payload = samples[ch]
        title = payload["embeds"][0]["title"][:42]

        if not url:
            # A missing URL is the single likeliest failure and it is silent in
            # production, so it gets called out here rather than reported as a
            # send failure.
            print(f"{ch:11} {'NO — unset':11} {title:44} skipped")
            failures += 1
            continue
        if not args.send:
            print(f"{ch:11} {'yes':11} {title:44} would post")
            continue

        ok = send_discord(url, payload)
        print(f"{ch:11} {'yes':11} {title:44} {'posted' if ok else 'FAILED'}")
        if not ok:
            failures += 1

    print()
    if failures:
        print(f"{failures} channel(s) not working — fix before relying on this live.")
    else:
        print("all channels OK." if args.send else "all channels configured.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
