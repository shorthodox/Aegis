"""Fails if _process_symbol's observable behaviour changes.

The refactor of _process_symbol (2,059 lines) is behaviour-preserving by
contract. This test is what enforces that contract: it replays the case matrix
in characterise_process_symbol.py and diffs against the recorded baseline.

If this fails during a refactor, the extraction changed behaviour — fix the
extraction, do not re-record. Re-record ONLY when a behaviour change is
intended, and say so in the commit:

    python -m scripts.tests.characterise_process_symbol --write
"""
import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from scripts.tests import characterise_process_symbol as H


@pytest.fixture(scope='module')
def snapshot():
    return json.loads(json.dumps(
        asyncio.run(H.capture(Path(tempfile.mkdtemp()))),
        sort_keys=True, default=str,
    ))


@pytest.fixture(scope='module')
def baseline():
    if not H.SNAPSHOT_PATH.exists():
        pytest.skip('no baseline recorded')
    return json.loads(H.SNAPSHOT_PATH.read_text(encoding='utf-8'))


def test_no_case_raises(snapshot):
    """Every case must complete — an exception means the harness drifted too."""
    broken = {k: v['ERROR'] for k, v in snapshot.items() if 'ERROR' in v}
    assert not broken, f'cases raised: {broken}'


def test_behaviour_matches_baseline(snapshot, baseline):
    drift = [k for k in sorted(set(baseline) | set(snapshot))
             if baseline.get(k) != snapshot.get(k)]
    if drift:
        detail = []
        for k in drift[:4]:
            b, c = baseline.get(k) or {}, snapshot.get(k) or {}
            for f in sorted(set(b) | set(c)):
                if b.get(f) != c.get(f):
                    detail.append(f'  [{k}] {f}\n    was: {b.get(f)}\n    now: {c.get(f)}')
        pytest.fail(f'_process_symbol behaviour changed in {len(drift)} case(s): '
                    f'{drift}\n' + '\n'.join(detail))


def test_baseline_still_covers_the_entry_path(baseline):
    """Guards the net itself: if no case reaches _open_position, the matrix has
    rotted and the refactor would be running without cover."""
    opened = [k for k, v in baseline.items() if v.get('position')]
    assert opened, 'no case opens a real position — matrix no longer covers entry'
    lines = {l for v in baseline.values() for l in v.get('decisions', [])}
    assert len(lines) >= 12, f'only {len(lines)} distinct decisions exercised'
