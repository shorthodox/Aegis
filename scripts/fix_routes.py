"""
One-shot script: replace all /web/src/pages/*.html paths with clean URLs
across every HTML and JS file in web/src/.

Run from project root: python scripts/fix_routes.py
"""

import os
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ── Page map (order matters: longer/more-specific first) ──────────────
PAGE_MAP = [
    ("/web/src/pages/privacy_policy.html",  "/privacy"),
    ("/web/src/pages/risk_disclosure.html", "/risk-disclosure"),
    ("/web/src/pages/reset-password.html",  "/reset-password"),
    ("/web/src/pages/track-record.html",    "/track-record"),
    ("/web/src/pages/trader-record.html",   "/trader-record"),
    ("/web/src/pages/refund-policy.html",   "/refund-policy"),
    ("/web/src/pages/refund_policy.html",   "/refund-policy"),
    ("/web/src/pages/bot-record.html",      "/bot-record"),
    ("/web/src/pages/conditions.html",      "/conditions"),
    ("/web/src/pages/dashboard.html",       "/dashboard"),
    ("/web/src/pages/reviews.html",         "/reviews"),
    ("/web/src/pages/signals.html",         "/signals"),
    ("/web/src/pages/contact.html",         "/contact"),
    ("/web/src/pages/pricing.html",         "/pricing"),
    ("/web/src/pages/review.html",          "/reviews"),
    ("/web/src/pages/index.html",           "/"),
    ("/web/src/pages/terms.html",           "/terms"),
    ("/web/src/pages/logic.html",           "/logic"),
    ("/web/src/pages/chart.html",           "/chart"),
    ("/web/src/pages/pitch.html",           "/pitch"),
]

# Also fix canonical URLs with full domain
CANONICAL_MAP = [
    (f"https://gatekeeper.sbs{old}", f"https://gatekeeper.sbs{new}")
    for old, new in PAGE_MAP
]

ALL_REPLACEMENTS = CANONICAL_MAP + PAGE_MAP

DIRS = [
    ROOT / "web" / "src" / "pages",
    ROOT / "web" / "src" / "scripts",
]

def process_file(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    original = text
    for old, new in ALL_REPLACEMENTS:
        text = text.replace(old, new)
    if text != original:
        path.write_text(text, encoding="utf-8")
        count = sum(original.count(old) for old, _ in ALL_REPLACEMENTS)
        return count
    return 0

changed = 0
files_changed = 0
for d in DIRS:
    for ext in ("*.html", "*.js"):
        for p in d.glob(ext):
            n = process_file(p)
            if n:
                print(f"  updated  {p.relative_to(ROOT)}")
                files_changed += 1
                changed += n

print(f"\nDone: {files_changed} files updated, ~{changed} replacements.")
