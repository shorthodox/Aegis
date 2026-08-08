"""Every page was slow to open. None of it was the page.

Production is one shared vCPU (fly.toml: cpus = 1) running a single uvicorn
worker that hosts BOTH the web app and the 60-token engine. Three things were
competing for that core, and two of them were free to fix:

  * 23 route handlers were declared `async def` while calling Firestore
    SYNCHRONOUSLY. A blocking call in a coroutine freezes the whole event loop
    for the round trip — measured at 42 ms median, 973 ms worst — and while it
    is frozen nothing else is served, including static HTML. FastAPI runs a
    plain `def` handler in its threadpool instead, so the loop stays free. None
    of the 23 awaited anything, so the keyword was pure cost.

  * The engine ran MAX_CONCURRENT = 8 feature-builds with no BLAS/OMP cap, so
    each of those threads could spawn its own native pool. On one core that is
    an order of magnitude of oversubscription, and it pegged the core for the
    length of every scan.

  * Assets were served `no-cache, must-revalidate`, so a ~340 KB dashboard
    asset graph paid a round trip per file on every single navigation.

These tests pin the fixes, not the timings — a latency assertion on CI hardware
would be a flake generator.
"""
import ast
import inspect
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / 'main.py'

# Helpers that reach Firestore synchronously. A coroutine calling any of these
# blocks the event loop for the duration of a network round trip to Google.
_BLOCKING_CALLS = ('db.collection', 'get_user_doc(', 'update_last_login(',
                   'create_user_doc(', 'phone_is_unique(')


def _route_handlers():
    """Every function main.py registers as an HTTP route."""
    src = MAIN.read_text(encoding='utf-8')
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        decorators = [ast.unparse(d) for d in node.decorator_list]
        if not any(d.startswith('app.') for d in decorators):
            continue
        if any('websocket' in d for d in decorators):
            continue        # a websocket handler must stay async
        yield node, ast.get_source_segment(src, node) or ''


# ── the event loop must not be blocked ───────────────────────────────────────

def test_no_async_route_blocks_the_loop_on_firestore():
    """The defect: sync Firestore inside `async def`.

    A handler that blocks may still be correct in isolation — which is why this
    never showed up in a single-request test. It only hurts under concurrency,
    where it serialises every other request behind itself.
    """
    offenders = []
    for node, seg in _route_handlers():
        if not isinstance(node, ast.AsyncFunctionDef):
            continue
        if not any(c in seg for c in _BLOCKING_CALLS):
            continue
        awaits = [n for n in ast.walk(node)
                  if isinstance(n, (ast.Await, ast.AsyncFor, ast.AsyncWith))]
        if not awaits:
            offenders.append(f'main.py:{node.lineno} {node.name}')

    assert not offenders, (
        'these routes call Firestore synchronously inside `async def` and '
        'await nothing — each one freezes the event loop for the whole round '
        'trip, stalling every other request including static pages. Declaring '
        'them `def` hands them to FastAPI\'s threadpool with identical '
        'semantics:\n  ' + '\n  '.join(offenders))


def test_the_converted_routes_are_still_registered_and_sync():
    """Guard against someone 'tidying' the keyword back in."""
    import main
    for name in ('get_dashboard', 'api_signals', 'api_public_signals',
                 'get_trial_status', 'get_me', 'login'):
        fn = getattr(main, name)
        assert not inspect.iscoroutinefunction(fn), (
            f'{name} is async again — if it now genuinely awaits something, '
            f'move its Firestore work to run_in_threadpool instead of '
            f'blocking the loop')


# ── the engine must not out-thread the machine ───────────────────────────────

def test_native_thread_pools_are_capped_before_numpy_is_imported():
    """Uncapped, each engine worker spawns its own OpenMP pool.

    The cap has to be set before the import, so it lives at the top of main.py
    rather than anywhere more tasteful.
    """
    src = MAIN.read_text(encoding='utf-8')
    head = src[:src.index('import asyncio')]
    assert 'OMP_NUM_THREADS' in head, (
        'the BLAS/OpenMP cap moved below the first heavy import, where it no '
        'longer takes effect — numpy reads these at import time')
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        assert os.environ.get(var) == '1', f'{var} is not capped'


def test_concurrency_is_sized_to_the_cpu_not_to_a_constant():
    from scripts.engine.engine import LiveEngine
    assert LiveEngine.MAX_CONCURRENT <= max(2, (os.cpu_count() or 1) * 2), (
        'MAX_CONCURRENT exceeds what this machine can run — on the 1-vCPU '
        'production box a fixed 8 pegged the core for the whole scan and '
        'starved the web server sharing it')
    assert LiveEngine.MAX_CONCURRENT >= 2, 'the engine would scan serially'


# ── assets must be cacheable ─────────────────────────────────────────────────

@pytest.fixture(scope='module')
def client():
    import main
    from fastapi.testclient import TestClient
    return TestClient(main.app)


def test_versioned_assets_are_immutable(client):
    """A ?v= URL names one build, so it never needs revalidating."""
    r = client.get('/web/src/scripts/gatekeeper.js?v=80.0')
    assert r.status_code == 200
    assert 'immutable' in r.headers.get('cache-control', ''), (
        'versioned assets are revalidating again — that is a round trip per '
        'file on every navigation')


def test_html_is_never_cached(client):
    """The flip side: a cached page pins the whole asset graph to an old deploy."""
    for path in ('/', '/dashboard', '/pricing'):
        cc = client.get(path).headers.get('cache-control', '')
        assert 'no-cache' in cc, f'{path} is cacheable — deploys would not take'


def test_unversioned_js_still_revalidates(client):
    """Nothing else would tell the browser it changed."""
    cc = client.get('/web/src/scripts/auth.js').headers.get('cache-control', '')
    assert 'no-cache' in cc


def test_pages_are_compressed(client):
    r = client.get('/dashboard', headers={'Accept-Encoding': 'gzip'})
    assert r.headers.get('content-encoding') == 'gzip', (
        'the 196 KB dashboard is going out uncompressed')


# ── the mount is rooted at web/, which holds more than the site ──────────────

@pytest.mark.parametrize('path', [
    '/web/node_modules/postcss/package.json',
    '/web/package.json',
    '/web/package-lock.json',
])
def test_the_npm_tree_is_not_public(client, path):
    assert client.get(path).status_code == 404, (
        f'{path} is served to the internet — StaticFiles is mounted on web/, '
        f'which contains the npm tree and build config as well as the site')
