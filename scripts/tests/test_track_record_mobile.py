"""The track record must read correctly on a narrow phone.

Two problems, reported together.

1. OVERLAP. .tr-panel-header is a flex row with `position: sticky; top: 60px` and
   no flex-wrap. On a narrow screen the status group (live dot, refreshed time,
   refresh link) cannot fit beside the title, the header grows taller than its
   sticky offset accounts for, and a sticky element that outgrows the layout sits
   ON TOP of the body text instead of pushing it down.

   Fixed twice over, deliberately: wrapping lets the header lay itself out
   honestly at any width, and dropping sticky below 700px means a header of ANY
   height can no longer cover anything — a guarantee wrapping alone does not give.

2. DISCLOSURE ORDER. The RISKY-tier notice sat inside the left panel body, below
   the summary tiles and the history note. On a phone a reader met the win rate,
   the signal table and the engine status BEFORE learning the book is RISKY-tier.
   A disclosure that arrives after the numbers it qualifies has already failed.
   It is now the first thing under "not financial advice", above every statistic.
"""
import re
from pathlib import Path

import pytest

_RAW = (Path(__file__).resolve().parent.parent.parent
        / 'web' / 'src' / 'pages' / 'track-record.html').read_text(
            encoding='utf-8', errors='replace')
# Strip comments: the comment explaining each fix quotes the broken CSS verbatim.
PAGE = re.sub(r'/\*.*?\*/|<!--.*?-->', '', _RAW, flags=re.S)


# ── disclosure order ─────────────────────────────────────────────────────────

def test_the_risky_notice_comes_before_any_statistic():
    note = PAGE.index('<p class="tr-risky-note"')
    panels = PAGE.index('<div class="tr-layout">')
    assert note < panels, (
        'the RISKY-tier disclosure is below the panels again — on a phone the '
        'reader meets the win rate and the signal table before the warning'
    )


def test_it_sits_with_the_other_disclaimer():
    assert PAGE.index('Not financial advice') < PAGE.index('<p class="tr-risky-note"')


def test_it_renders_unconditionally():
    """It matters MOST before any signal has closed, when there is no record to
    judge the signals by."""
    i = PAGE.index('<p class="tr-risky-note"')
    tag = PAGE[i:PAGE.index('>', i)]
    assert 'display:none' not in tag.replace(' ', '')
    assert 'hidden' not in tag


# ── narrow-screen layout ─────────────────────────────────────────────────────

def test_the_panel_header_wraps():
    m = re.search(r'\.tr-panel-header \{([^}]*)\}', PAGE)
    assert m and 'flex-wrap: wrap' in m.group(1), (
        'the header cannot wrap, so its contents overflow on a narrow screen'
    )


def test_the_header_is_not_sticky_on_a_phone():
    """The actual overlap fix. A sticky header that outgrows its offset covers
    the content beneath it."""
    m = re.search(r'@media \(max-width: 700px\) \{(.*?)\n        \}', PAGE, re.S)
    assert m, 'the narrow-screen breakpoint is gone'
    assert 'position: static' in m.group(1), (
        'the panel header is sticky again below 700px — it will overlap the '
        'panel body whenever it wraps to a second line'
    )


def test_the_status_group_gets_its_own_line():
    m = re.search(r'@media \(max-width: 700px\) \{(.*?)\n        \}', PAGE, re.S)
    body = m.group(1)
    assert 'margin-left: 0' in body and 'width: 100%' in body, (
        'the status group is still pinned right by margin-left:auto and will be '
        'squeezed against the title'
    )


def test_very_narrow_screens_get_one_column():
    assert '@media (max-width: 380px)' in PAGE, (
        'no breakpoint for ~360px Android widths, where two summary columns clip'
    )
    m = re.search(r'@media \(max-width: 380px\) \{(.*?)\n        \}', PAGE, re.S)
    assert 'grid-template-columns: 1fr' in m.group(1)


def test_the_page_markup_is_balanced():
    o, c = len(re.findall(r'<div\b', _RAW)), _RAW.count('</div>')
    assert o == c, f'<div> {o} vs </div> {c} — moving the note broke the nesting'
