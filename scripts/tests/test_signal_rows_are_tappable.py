"""The fired-signals rows must be tappable on a phone.

Reported: tapping a row in "ALL CURRENTLY FIRED SIGNALS" did nothing on mobile.

The row carried an onclick, so it worked with a mouse. The defect was in the CSS:

    .sig-table tr:hover td { background: ...; cursor: pointer; }

`cursor: pointer` existed ONLY inside `:hover`, and a touch screen never matches
:hover. That is not merely a cosmetic pointer — iOS decides whether to synthesise
a click on a non-interactive element largely from signals like this one, so a
<tr> with no pointer and no interactive role often swallows the tap entirely.
Desktop worked, mobile did nothing, and the markup looked correct either way.

The row was also `<tr onclick=...>`: not focusable, unreachable by keyboard, and
announced as a plain row by a screen reader.

Fixed by making the row genuinely interactive rather than incidentally clickable:
pointer on the row itself, touch-action to drop the 300ms delay, a tap highlight,
role/tabindex, Enter and Space handling, and a visible focus ring.
"""
import re
from pathlib import Path

import pytest

_RAW = (Path(__file__).resolve().parent.parent.parent
        / 'web' / 'src' / 'pages' / 'chart.html').read_text(encoding='utf-8', errors='replace')

# Strip CSS/JS block comments before scanning. The comment explaining THIS fix
# quotes the broken rule verbatim, so an uncommented scan matches the explanation
# and reports the bug as still present.
CHART = re.sub(r'/\*.*?\*/', '', _RAW, flags=re.S)


def test_cursor_pointer_is_not_hover_only():
    """The exact regression. Hover never matches on a touch screen."""
    assert not re.search(r':hover[^{]*\{[^}]*cursor:\s*pointer', CHART), (
        'cursor:pointer is behind :hover again — the row becomes untappable on '
        'mobile while still working on desktop, which is the hardest kind of '
        'break to notice'
    )


def test_the_row_declares_the_pointer_itself():
    m = re.search(r'\.sig-table tbody tr \{([^}]*)\}', CHART)
    assert m, 'the row rule is gone'
    assert 'cursor: pointer' in m.group(1)


def test_the_tap_delay_is_removed():
    m = re.search(r'\.sig-table tbody tr \{([^}]*)\}', CHART)
    assert 'touch-action: manipulation' in m.group(1), (
        'without this the row carries the legacy 300ms double-tap delay'
    )


def test_a_tap_gives_feedback():
    """A tap that navigates with no acknowledgement reads as a dead control."""
    m = re.search(r'\.sig-table tbody tr \{([^}]*)\}', CHART)
    assert '-webkit-tap-highlight-color' in m.group(1)
    assert '.sig-table tbody tr:active td' in CHART, 'no pressed state'


def test_the_row_is_reachable_without_a_mouse():
    i = CHART.index('<tr role="link"')
    row = CHART[i:i + 600]
    assert 'tabindex="0"' in row, 'the row cannot be focused'
    assert 'onkeydown' in row, 'Enter/Space do not activate the row'
    assert "event.key==='Enter'" in row and "event.key===' '" in row


def test_the_row_announces_what_it_does():
    i = CHART.index('<tr role="link"')
    row = CHART[i:i + 600]
    assert 'role="link"' in row, 'a screen reader announces this as a plain row'
    assert 'aria-label=' in row


def test_focus_is_visible():
    assert '.sig-table tbody tr:focus-visible' in CHART, (
        'keyboard users get no indication of where they are'
    )
