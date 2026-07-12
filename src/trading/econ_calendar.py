"""
econ_calendar.py — scheduled high-impact US macro event lock (news lock).

The model was not trained for the volatility spikes around CPI / FOMC / NFP, so the
UWGS extreme-volatility veto (V4) locks trading in a window around them. This module
answers "are we inside a news-lock window right now?" with NO external dependency:

  * NFP  — computed deterministically: first Friday of the month, 08:30 America/New_York.
  * FOMC / CPI / PPI / other dated events — from the built-in `_DEFAULT_EVENTS` list
    (shipped in code so it always deploys), optionally MERGED with a persistent-volume
    override at data/econ_calendar.json. Missing/unparseable override → defaults only
    (fail-open); a wrong/missing date just skips that one lock.

ET→UTC is computed with the US DST rule (2nd Sun Mar … 1st Sun Nov) so it stays correct
without the IANA tzdata package (which Windows lacks by default).
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple

# Optional override file (NOT committed — data/ is gitignored/ephemeral). If present
# on a persistent volume, its events are MERGED with the built-in defaults below, so
# the calendar can be updated live without a redeploy.
_CAL_PATH = Path(__file__).resolve().parent.parent.parent / 'data' / 'econ_calendar.json'

# Built-in high-impact events (America/New_York wall clock). NFP is auto-computed and
# not listed. FOMC decision = 14:00 ET; CPI/PPI = 08:30 ET. VERIFY/EXTEND as the BLS/
# Fed publish new dates — a wrong/missing date just skips that lock (fail-open).
_DEFAULT_EVENTS = [
    {"date": "2026-07-29", "time_et": "14:00", "label": "FOMC"},
    {"date": "2026-09-16", "time_et": "14:00", "label": "FOMC"},
    {"date": "2026-10-28", "time_et": "14:00", "label": "FOMC"},
    {"date": "2026-12-09", "time_et": "14:00", "label": "FOMC"},
]

# Default lock window: minutes before/after the event print.
LOCK_WINDOW_MIN = 30

_cache: Optional[Tuple[float, List[dict]]] = None   # (mtime, parsed events)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The n-th `weekday` (Mon=0 … Sun=6) of month, e.g. 1st Friday, 2nd Sunday."""
    d = date(year, month, 1)
    offset = (weekday - d.weekday()) % 7
    return d + timedelta(days=offset + 7 * (n - 1))


def _us_in_dst(d: date) -> bool:
    """US Eastern DST: 2nd Sunday of March 02:00 … 1st Sunday of November 02:00."""
    start = _nth_weekday(d.year, 3, 6, 2)    # 2nd Sunday March
    end   = _nth_weekday(d.year, 11, 6, 1)   # 1st Sunday November
    return start <= d < end


def _et_to_utc(d: date, hour: int, minute: int) -> datetime:
    """Convert an America/New_York wall-clock time to an aware UTC datetime."""
    off = 4 if _us_in_dst(d) else 5          # EDT = UTC-4, EST = UTC-5
    return datetime(d.year, d.month, d.day, hour, minute, tzinfo=timezone.utc) + timedelta(hours=off)


def _load_json_events() -> List[dict]:
    """Built-in `_DEFAULT_EVENTS` merged with the optional override file (cached on
    its mtime). Returns just the defaults when no override file is present."""
    global _cache
    try:
        mtime = _CAL_PATH.stat().st_mtime
    except OSError:
        return _DEFAULT_EVENTS
    if _cache and _cache[0] == mtime:
        return _cache[1]
    try:
        raw = json.loads(_CAL_PATH.read_text(encoding='utf-8'))
        evs = raw.get('events', raw) if isinstance(raw, dict) else raw
        merged = _DEFAULT_EVENTS + [e for e in evs if isinstance(e, dict) and e.get('date')]
    except Exception:
        merged = _DEFAULT_EVENTS
    _cache = (mtime, merged)
    return merged


def _events_on(d: date) -> List[Tuple[datetime, str]]:
    """All scheduled event times (UTC) on calendar date `d`."""
    out: List[Tuple[datetime, str]] = []
    # NFP — first Friday, 08:30 ET.
    nfp = _nth_weekday(d.year, d.month, 4, 1)
    if nfp == d:
        out.append((_et_to_utc(d, 8, 30), 'NFP'))
    # Dated events from JSON.
    for e in _load_json_events():
        try:
            ed = date.fromisoformat(str(e['date']))
            if ed != d:
                continue
            hh, mm = (int(x) for x in str(e.get('time_et', '08:30')).split(':')[:2])
            out.append((_et_to_utc(ed, hh, mm), str(e.get('label', 'MACRO'))))
        except Exception:
            continue
    return out


def is_locked(now_utc: Optional[datetime] = None,
              window_min: int = LOCK_WINDOW_MIN) -> Tuple[bool, str]:
    """
    True + 'NEWS_LOCK:<label>' when `now_utc` is within ±window_min of a scheduled
    high-impact US macro event; else (False, ''). Best-effort — never raises.
    """
    try:
        now = now_utc or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        # Check yesterday…tomorrow to cover windows that straddle a UTC midnight.
        for delta in (-1, 0, 1):
            for ev_utc, label in _events_on((now + timedelta(days=delta)).date()):
                if abs((now - ev_utc).total_seconds()) <= window_min * 60:
                    return True, f'NEWS_LOCK:{label}'
    except Exception:
        pass
    return False, ''
