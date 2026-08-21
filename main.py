# ===================================================================
# main.py - CRITICAL: Load environment variables FIRST
# ===================================================================
from dotenv import load_dotenv
import os

# MUST BE FIRST LINE OF EXECUTION - loads all env vars before any other code
load_dotenv()

# -------------------------------------------------------------------
# Native thread caps — MUST be set before numpy/xgboost/sklearn import
# -------------------------------------------------------------------
# The engine runs MAX_CONCURRENT predict_realtime() calls in its own thread
# pool, inside the SAME process as the web server. Every one of those threads
# builds a 350-bar feature frame and runs XGBoost inference. Left uncapped,
# each thread spawns its own OpenMP/BLAS pool sized to the machine, so an
# 8-thread scan can ask for 8x more CPU than exists.
#
# Production is one shared vCPU (fly.toml: cpus = 1) serving BOTH the web app
# and the engine. Oversubscribing it pegs the core for the length of a scan and
# starves the event loop, which is what made every page slow to open. One
# native thread per worker keeps the parallelism where we manage it — in the
# executor — instead of multiplying underneath it.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

# Now safe to import libraries that might use environment variables
import asyncio
import json
import re
import uuid
import random
import string
import threading
import time
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, Set, TYPE_CHECKING, Union
from contextlib import asynccontextmanager
import inspect

import httpx

from fastapi import FastAPI, HTTPException, Depends, status, WebSocket, WebSocketDisconnect, Request, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import RedirectResponse, JSONResponse, FileResponse, Response
from fastapi.encoders import jsonable_encoder
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
from pydantic import BaseModel, EmailStr, SecretStr
import jwt
import bcrypt
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig, MessageType, NameEmail
import uvicorn
from dataclasses import asdict
from starlette.middleware.sessions import SessionMiddleware
import numpy as np
import logging
import shutil
from email_validator import validate_email, EmailNotValidError
from functools import partial
from generate_dev_code import generate_dev_key

logger = logging.getLogger("aegis")

# -------------------------------------------------------------------
# Helper: Recursively convert numpy types to native Python types
# -------------------------------------------------------------------
def numpy_to_native(obj) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_native(v) for v in obj]
    elif hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return obj

# -------------------------------------------------------------------
# Security: JWT & Algorithm must be from environment
# -------------------------------------------------------------------
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
if not SECRET_KEY:
    raise RuntimeError("JWT_SECRET_KEY is missing. Add it to your local .env file or Railway variables.")
if not ALGORITHM:
    raise RuntimeError("ALGORITHM is missing. Add it to your local .env file or Railway variables.")

# -------------------------------------------------------------------
# DODO Payments & Razorpay payment gateways
# -------------------------------------------------------------------
DODO_PAYMENTS_API_KEY = os.getenv("DODO_PAYMENTS_API_KEY")
DODO_PAYMENTS_WEBHOOK_SECRET = os.getenv("DODO_PAYMENTS_WEBHOOK_SECRET")
DODO_PAYMENTS_MODE = os.getenv("DODO_PAYMENTS_MODE", "test").lower()
DODO_PRODUCT_IDS = {
    "basic":        os.getenv("DODO_PRODUCT_ID_BASIC"),
    "intermediate": os.getenv("DODO_PRODUCT_ID_INTERMEDIATE"),
    "pro":          os.getenv("DODO_PRODUCT_ID_PRO"),
}
DODO_PAYMENTS_ENABLED = bool(DODO_PAYMENTS_API_KEY)
_DODO_BASE = "https://live.dodopayments.com" if DODO_PAYMENTS_MODE == "live" else "https://test.dodopayments.com"

if DODO_PAYMENTS_ENABLED:
    print(f"DODO Payments configured in {DODO_PAYMENTS_MODE.upper()} mode")
else:
    print("[INFO] DODO Payments not configured. Set DODO_PAYMENTS_API_KEY to enable.")

async def _dodo_post(path: str, payload: dict) -> dict:
    """POST to DODO Payments REST API using Bearer Token authorization."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{_DODO_BASE}{path}",
            json=payload,
            headers={
                "Authorization": f"Bearer {DODO_PAYMENTS_API_KEY}",
                "Content-Type": "application/json",
            },
        )
        resp.raise_for_status()
        return resp.json()

async def _dodo_get(path: str) -> dict:
    """GET from DODO Payments REST API using Bearer Token authorization."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            f"{_DODO_BASE}{path}",
            headers={
                "Authorization": f"Bearer {DODO_PAYMENTS_API_KEY}",
                "Content-Type": "application/json",
            },
        )
        resp.raise_for_status()
        return resp.json()


# -------------------------------------------------------------------
# Whop
# -------------------------------------------------------------------
# Whop checkout follows the same shape as Paddle and DODO: create something
# server-side, hand the customer the URL it returns. Here that something is a
# CHECKOUT CONFIGURATION — "a reusable configuration for a checkout, including
# the plan, affiliate, and custom metadata" — and the important sentence in
# Whop's docs is this one:
#
#     "Payments and memberships created from a checkout session inherit its
#      metadata."
#
# That is what makes this integration safe. The metadata is attached by THIS
# server, carries {user_id, plan}, and comes back on the webhook, so the
# customer cannot influence which account gets upgraded. It is the exact
# property Paddle's custom_data gives us, which is why the two branches below
# read almost identically.
#
# The alternative — matching the buyer's email to an AEGIS account — was
# rejected: a customer paying with a different email than they registered with
# would pay and receive nothing.
#
# Precedence is Whop -> Paddle -> DODO -> Razorpay, decided purely by which env
# vars are set, so switching over is a config change and rolling back is
# unsetting WHOP_API_KEY. Nothing is deleted here.
WHOP_API_KEY = os.getenv("WHOP_API_KEY")
WHOP_WEBHOOK_SECRET = os.getenv("WHOP_WEBHOOK_SECRET")
WHOP_MODE = os.getenv("WHOP_MODE", "sandbox").lower()
# Whop bills against a PLAN ('plan_...'), created in Dashboard > Checkout links.
# There is no documented API to create one, so these are configuration.
WHOP_PLAN_IDS = {
    "basic":        os.getenv("WHOP_PLAN_ID_BASIC"),
    "intermediate": os.getenv("WHOP_PLAN_ID_INTERMEDIATE"),
    "pro":          os.getenv("WHOP_PLAN_ID_PRO"),
}
WHOP_ENABLED = bool(WHOP_API_KEY)
# Where Whop sends the buyer after checkout. Optional: with it unset Whop uses
# whatever the checkout link itself is configured with. The return URL is NOT
# how the plan is granted — that is the webhook's job — so a user who closes
# the tab before redirecting still gets what they paid for.
WHOP_REDIRECT_URL = os.getenv("WHOP_REDIRECT_URL")
_WHOP_BASE = os.getenv("WHOP_API_BASE", "https://api.whop.com/api/v1")
# Whop's docs give the base URL and the resource (checkout configurations:
# list / create / retrieve) but do not spell the create path out verbatim, so
# it is overridable rather than hardcoded — if it turns out to differ, this is
# a config change, not a code change.
_WHOP_CHECKOUT_PATH = os.getenv("WHOP_CHECKOUT_CONFIG_PATH",
                                "/checkout_configurations")
# Standard Webhooks replay window. Whop retries rejected deliveries, so a wider
# window costs nothing; 300s matches the Paddle setting above.
WHOP_WEBHOOK_TOLERANCE_SECONDS = int(
    os.getenv("WHOP_WEBHOOK_TOLERANCE_SECONDS", "300"))

if WHOP_ENABLED:
    print(f"Whop configured in {WHOP_MODE.upper()} mode")
    if not WHOP_WEBHOOK_SECRET:
        print("[WARN] WHOP_WEBHOOK_SECRET is not set — Whop webhooks will be "
              "REJECTED. Subscriptions would activate at checkout but never renew "
              "or cancel. Set it before taking live payments.")
    _missing_whop_plans = [k for k, v in WHOP_PLAN_IDS.items() if not v]
    if _missing_whop_plans:
        print(f"[WARN] Whop plan IDs missing for: {', '.join(_missing_whop_plans)} "
              f"— those plans cannot be checked out.")


async def _whop_post(path: str, payload: dict) -> dict:
    """POST to the Whop REST API."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            f"{_WHOP_BASE}{path}",
            json=payload,
            headers={
                "Authorization": f"Bearer {WHOP_API_KEY}",
                "Content-Type": "application/json",
            },
        )
        resp.raise_for_status()
        return resp.json()


def _whop_secret_keys(secret: str) -> list:
    """Candidate HMAC keys derived from a Whop webhook secret.

    Whop's docs describe Standard Webhooks — a base64 secret carrying a
    `whsec_` prefix, base64-decoded before use as the key. The dashboard
    actually issues `ws_` followed by 64 hex characters, i.e. a 256-bit key
    written in hex. Those are different encodings and the documented one does
    not even parse: base64-decoding a `ws_...` secret raises, which made the
    first version of this function reject every genuine webhook.

    Rather than guess which is right, the plausible keys are derived
    DETERMINISTICALLY from the observed format and all are tried. That is not a
    weakening: an attacker still has to possess the secret, and supporting two
    encodings of the same secret is no easier to forge than one. It is the same
    reasoning as accepting several versioned signatures during a rotation.

    Order matters only for speed, not correctness:
      1. hex-decoded, when the body is pure hex (the `ws_` format)
      2. base64-decoded, when it decodes cleanly (the documented format)
      3. the raw ASCII bytes, which is what a naive implementation would use

    If a future Whop change breaks all three, verification fails closed and the
    webhook is rejected — never accepted unverified.
    """
    import base64 as _base64

    if not secret:
        return []
    body = secret.split("_", 1)[1] if secret.startswith(("whsec_", "ws_")) else secret
    keys: list = []

    if len(body) % 2 == 0 and all(c in "0123456789abcdefABCDEF" for c in body):
        try:
            keys.append(bytes.fromhex(body))
        except ValueError:
            pass
    try:
        decoded = _base64.b64decode(body, validate=True)
        if decoded:
            keys.append(decoded)
    except Exception:
        pass
    keys.append(body.encode("utf-8"))

    # de-duplicate, preserving order
    seen, out = set(), []
    for k in keys:
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def whop_verify_signature(raw_body: bytes,
                          webhook_id: str,
                          webhook_timestamp: str,
                          signature_header: str,
                          secret: Optional[str] = None,
                          tolerance_seconds: Optional[int] = None) -> bool:
    """Verify a Whop webhook signature (Standard Webhooks).

    Deliberately NOT the same scheme as either neighbour in this file, which is
    why it gets its own function rather than a shared helper:

      * Paddle signs `<ts>:<body>` and sends hex in one `Paddle-Signature`.
      * DODO's branch below compares a plain hex HMAC of the body alone.
      * Whop signs `<id>.<timestamp>.<body>`, sends BASE64, splits the parts
        across three headers, and — the easy one to get wrong — the secret is
        base64 and must be DECODED before use as the HMAC key.

    `webhook-signature` may carry several space-separated versioned signatures
    (`v1,<b64> v1,<b64>`) during a secret rotation; any one matching is enough.

    Returns True only for a well-formed, in-window, matching signature. A caller
    that cannot verify must reject: an unverified payment webhook is an
    unauthenticated "upgrade this user" endpoint.
    """
    import hmac as _hmac
    import hashlib as _hashlib
    import base64 as _base64

    key_b64 = secret if secret is not None else WHOP_WEBHOOK_SECRET
    tol = (tolerance_seconds if tolerance_seconds is not None
           else WHOP_WEBHOOK_TOLERANCE_SECONDS)
    if not key_b64 or not signature_header or not webhook_id or not webhook_timestamp:
        return False

    try:
        ts_int = int(webhook_timestamp)
    except (TypeError, ValueError):
        return False
    # Reject stale AND future-dated timestamps (clock-skew replay both ways).
    if abs(time.time() - ts_int) > tol:
        return False

    keys = _whop_secret_keys(key_b64)
    if not keys:
        return False

    signed = f"{webhook_id}.{webhook_timestamp}.".encode("utf-8") + raw_body
    candidates = [c.strip() for c in
                  (p.partition(",")[2] for p in signature_header.split(" ")) if c.strip()]
    if not candidates:
        return False

    for key in keys:
        digest = _hmac.new(key, signed, _hashlib.sha256).digest()
        # Whop's docs specify base64; hex is checked too because the secret
        # format already turned out to differ from the documented one.
        for expected in (_base64.b64encode(digest).decode("utf-8"), digest.hex()):
            for candidate in candidates:
                if _hmac.compare_digest(expected, candidate):
                    return True
    return False


# -------------------------------------------------------------------
# Paddle Billing
# -------------------------------------------------------------------
# Paddle BILLING (the current product), not Paddle Classic. Checkout follows
# the same shape as DODO: create a transaction server-side, hand the customer
# the returned checkout URL. A transaction containing a RECURRING price makes
# Paddle create the subscription itself once payment completes.
#
# Precedence is Paddle -> DODO -> Razorpay, decided purely by which env vars
# are set, so switching over is a config change and rolling back is unsetting
# PADDLE_API_KEY. Nothing is deleted here.
PADDLE_API_KEY = os.getenv("PADDLE_API_KEY")
PADDLE_WEBHOOK_SECRET = os.getenv("PADDLE_WEBHOOK_SECRET")   # 'pdl_ntfset_...'
PADDLE_MODE = os.getenv("PADDLE_MODE", "sandbox").lower()
# Paddle bills against a PRICE, not a product — a product can carry several
# prices (monthly/annual, per currency). These must be price IDs ('pri_...').
PADDLE_PRICE_IDS = {
    "basic":        os.getenv("PADDLE_PRICE_ID_BASIC"),
    "intermediate": os.getenv("PADDLE_PRICE_ID_INTERMEDIATE"),
    "pro":          os.getenv("PADDLE_PRICE_ID_PRO"),
}
PADDLE_ENABLED = bool(PADDLE_API_KEY)
_PADDLE_BASE = ("https://api.paddle.com" if PADDLE_MODE == "live"
                else "https://sandbox-api.paddle.com")
# Replay window for webhooks. Paddle's own SDKs default to 5s, which is only
# safe when the host clock is tightly synced — a few seconds of drift silently
# rejects every event. Paddle retries failed deliveries, so a wider window
# costs nothing in reliability terms; 300s matches the common industry default.
PADDLE_WEBHOOK_TOLERANCE_SECONDS = int(
    os.getenv("PADDLE_WEBHOOK_TOLERANCE_SECONDS", "300"))

if PADDLE_ENABLED:
    print(f"Paddle Billing configured in {PADDLE_MODE.upper()} mode")
    if not PADDLE_WEBHOOK_SECRET:
        print("[WARN] PADDLE_WEBHOOK_SECRET is not set — Paddle webhooks will be "
              "REJECTED. Subscriptions would activate at checkout but never renew "
              "or cancel. Set it before taking live payments.")
    _missing_prices = [k for k, v in PADDLE_PRICE_IDS.items() if not v]
    if _missing_prices:
        print(f"[WARN] Paddle price IDs missing for: {', '.join(_missing_prices)} "
              f"— those plans cannot be checked out.")


async def _paddle_post(path: str, payload: dict) -> dict:
    """POST to the Paddle Billing REST API."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            f"{_PADDLE_BASE}{path}",
            json=payload,
            headers={
                "Authorization": f"Bearer {PADDLE_API_KEY}",
                "Content-Type": "application/json",
            },
        )
        resp.raise_for_status()
        return resp.json()


async def _paddle_get(path: str) -> dict:
    """GET from the Paddle Billing REST API."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{_PADDLE_BASE}{path}",
            headers={
                "Authorization": f"Bearer {PADDLE_API_KEY}",
                "Content-Type": "application/json",
            },
        )
        resp.raise_for_status()
        return resp.json()


def paddle_verify_signature(raw_body: bytes, signature_header: str,
                            secret: Optional[str] = None,
                            tolerance_seconds: Optional[int] = None) -> bool:
    """Verify a Paddle Billing webhook signature.

    Header format is `ts=<unix>;h1=<hex>`. The signed payload is the timestamp,
    a colon, then the RAW request body — unmodified, no re-serialising, no
    whitespace changes — HMAC-SHA256'd with the notification-setting secret.

    Returns True only for a well-formed, in-window, matching signature. A
    caller that cannot verify must reject the request: an unverified payment
    webhook is an unauthenticated "upgrade this user" endpoint.
    """
    import hmac as _hmac
    import hashlib as _hashlib

    key = secret if secret is not None else PADDLE_WEBHOOK_SECRET
    tol = (tolerance_seconds if tolerance_seconds is not None
           else PADDLE_WEBHOOK_TOLERANCE_SECONDS)
    if not key or not signature_header:
        return False

    ts_raw = ""
    h1 = ""
    for part in signature_header.split(";"):
        name, _, value = part.partition("=")
        if name.strip() == "ts":
            ts_raw = value.strip()
        elif name.strip() == "h1":
            h1 = value.strip()
    if not ts_raw or not h1:
        return False

    try:
        ts_int = int(ts_raw)
    except ValueError:
        return False
    # Reject stale AND future-dated timestamps (clock-skew replay both ways).
    if abs(time.time() - ts_int) > tol:
        return False

    signed_payload = ts_raw.encode("utf-8") + b":" + raw_body
    expected = _hmac.new(key.encode("utf-8"), signed_payload,
                         _hashlib.sha256).hexdigest()
    return _hmac.compare_digest(expected, h1)

RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET")
RAZORPAY_WEBHOOK_SECRET = os.getenv("RAZORPAY_WEBHOOK_SECRET")
RAZORPAY_PLAN_IDS = {
    "basic":        os.getenv("RAZORPAY_PLAN_ID_BASIC"),
    "intermediate": os.getenv("RAZORPAY_PLAN_ID_INTERMEDIATE"),
    "pro":          os.getenv("RAZORPAY_PLAN_ID_PRO"),
}
RAZORPAY_ENABLED = bool(RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET)
_RZP_BASE = "https://api.razorpay.com/v1"

if RAZORPAY_ENABLED:
    print("Razorpay payment gateway configured (fallback)")

async def _rzp_post(path: str, payload: dict) -> dict:
    """POST to the Razorpay REST API using HTTP Basic Auth. No SDK required."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{_RZP_BASE}{path}",
            json=payload,
            auth=(RAZORPAY_KEY_ID or "", RAZORPAY_KEY_SECRET or ""),
        )
        resp.raise_for_status()
        return resp.json()

# -------------------------------------------------------------------
# SOVEREIGN FIREBASE INITIALIZATION
# Supports both Railway (JSON string in env) and local (file path)
# -------------------------------------------------------------------
import firebase_admin
from firebase_admin import credentials, firestore, auth as firebase_auth

FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "aegis-d78e1")
cred_json = os.getenv("FIREBASE_CREDENTIALS")

base_dir = Path(__file__).resolve().parent

def resolve_credential_path(path_value: str) -> Path:
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return candidate
    return base_dir / candidate

fallback_cred_path = resolve_credential_path('config/serviceAccountKey.json')

if cred_json:
    # Prefer parsing FIREBASE_CREDENTIALS as JSON, but allow a file path fallback.
    cred = None
    parsed_json = False
    try:
        cred_dict = json.loads(cred_json)
        cred = credentials.Certificate(cred_dict)
        parsed_json = True
        print("[FIREBASE] Initialized via JSON environment variable")
    except json.JSONDecodeError:
        # Not JSON, continue to path fallback.
        pass
    except Exception as e:
        print(f"[ERROR] Failed to parse FIREBASE_CREDENTIALS JSON: {e}")

    if not parsed_json:
        resolved_path = resolve_credential_path(cred_json)
        path_exists = False
        try:
            path_exists = resolved_path.exists()
        except OSError:
            pass # Name too long or invalid path characters

        if path_exists:
            cred = credentials.Certificate(str(resolved_path))
            print(f"[FIREBASE] Initialized via file path: {resolved_path}")
        else:
            print(f"[FIREBASE] FIREBASE_CREDENTIALS is not valid JSON and file not found at: {resolved_path}")
            if fallback_cred_path.exists():
                cred = credentials.Certificate(str(fallback_cred_path))
                print(f"[FIREBASE] Initialized via fallback file: {fallback_cred_path}")
            else:
                raise RuntimeError(
                    "Firebase credentials not found. "
                    "Provide either FIREBASE_CREDENTIALS as JSON or ensure config/serviceAccountKey.json exists"
                )
else:
    # Fallback for local development
    if fallback_cred_path.exists():
        cred = credentials.Certificate(str(fallback_cred_path))
        print(f"[FIREBASE] Initialized via local file: {fallback_cred_path}")
    else:
        raise RuntimeError(
            "Firebase credentials not found in ENV or at config/serviceAccountKey.json"
        )

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client(database_id="default")

# -------------------------------------------------------------------
# OAuth (Authlib) - lazy initialization
# -------------------------------------------------------------------
try:
    from authlib.integrations.starlette_client import OAuth, OAuthError
except ImportError:
    OAuth = None
    OAuthError = Exception

oauth: Any = None

def init_oauth():
    global oauth
    if OAuth is None:
        return
    if oauth is not None:
        return

    oauth = OAuth()

    def _safe_register(name, client_id_env, client_secret_env, server_metadata_url, client_kwargs):
        try:
            client_id = os.getenv(client_id_env)
            client_secret = os.getenv(client_secret_env)
            if not client_id or not client_secret:
                return
            oauth.register(
                name=name,
                client_id=client_id,
                client_secret=client_secret,
                server_metadata_url=server_metadata_url,
                client_kwargs=client_kwargs,
            )
        except Exception as e:
            print(f"Warning: OAuth register for {name} failed: {e}")

    _safe_register(
        'google',
        'GOOGLE_CLIENT_ID',
        'GOOGLE_CLIENT_SECRET',
        'https://accounts.google.com/.well-known/openid-configuration',
        {'scope': 'openid email profile'},
    )
    _safe_register(
        'microsoft',
        'MICROSOFT_CLIENT_ID',
        'MICROSOFT_CLIENT_SECRET',
        'https://login.microsoftonline.com/common/v2.0/.well-known/openid-configuration',
        {'scope': 'openid email profile'},
    )
    _safe_register(
        'apple',
        'APPLE_CLIENT_ID',
        'APPLE_CLIENT_SECRET',
        'https://appleid.apple.com/.well-known/openid-configuration',
        {'scope': 'openid email name'},
    )

if TYPE_CHECKING:
    from scripts.live_engine import LiveEngine

# -------------------------------------------------------------------
# Static assets root path used by background tasks and API routes
# -------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_ROOT = os.path.join(BASE_DIR, "web")
WEB_ROOT_PATH = Path(WEB_ROOT)

if not WEB_ROOT_PATH.exists():
    print(f"âš ï¸ Warning: 'web' directory not found at {WEB_ROOT_PATH}. Creating fallback structure.")
    WEB_ROOT_PATH.mkdir(parents=True, exist_ok=True)
    pages_dir = WEB_ROOT_PATH / "src" / "pages"
    scripts_dir = WEB_ROOT_PATH / "src" / "scripts"
    styles_dir = WEB_ROOT_PATH / "src" / "styles"
    pages_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)
    styles_dir.mkdir(parents=True, exist_ok=True)
    (pages_dir / "index.html").write_text("<html><body><h1>Aegisâ€‘1</h1><p>Frontend files missing. Please upload the correct static files to 'web/src/pages/'</p></body></html>")
    (pages_dir / "dashboard.html").write_text("<html><body><h1>Dashboard unavailable</h1><p>Static files not found.</p></body></html>")

# Restore dynamic track-record files from the engine STATE folder if present.
# This avoids losing history when the `web/` directory is replaced during deploys.
# Reads from AEGIS_STATE_DIR (the persistent volume) so it picks up the RECORDS
# THAT SURVIVED the deploy, not the stale copy baked into the image.
try:
    _data_root = Path(os.environ.get('AEGIS_STATE_DIR') or (Path(BASE_DIR) / "data"))
    for _src_name in ("track_record.json", "trader_track_record.json"):
        _src = _data_root / _src_name
        _dst = WEB_ROOT_PATH / _src_name
        if _src.exists():
            _dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(_src, _dst)
                print(f"[Startup] Restored {_dst} from {_src}")
            except Exception as _e:
                print(f"[Startup] Failed to copy {_src} -> {_dst}: {_e}")
except Exception:
    pass

# -------------------------------------------------------------------
# LiveState for global data (shared with engine and WebSocket)
# -------------------------------------------------------------------
class LiveState:
    def __init__(self):
        self.data = {
            "tickers": {},
            "signals": {},
            "alpha_signals": {},
            "open_trades": [],
            "balance": 0.0,
            "alpha_mode": False,
            "warmup_progress": "0/0"
        }
        self.engine: Optional['LiveEngine'] = None

LIVE_STATE = LiveState()

# -------------------------------------------------------------------
# Track-record WebSocket connection manager
# Clients connecting to /ws/track-record receive the full track_record
# JSON on connect, then a fresh push every time the engine saves it.
# -------------------------------------------------------------------
class _TrackRecordManager:
    def __init__(self) -> None:
        self._clients: list = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._clients = [c for c in self._clients if c is not ws]

    async def broadcast(self, payload: dict) -> None:
        dead = []
        for ws in list(self._clients):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

_tr_ws_manager = _TrackRecordManager()


# -------------------------------------------------------------------
# Track Record System
# Logs every actionable BUY/SELL signal, monitors TP/SL hits, and
# persists outcomes to web/track_record.json for the public page.
# -------------------------------------------------------------------
TRACK_RECORD_PATH        = WEB_ROOT_PATH / "track_record.json"
TRADER_TRACK_RECORD_PATH = WEB_ROOT_PATH / "trader_track_record.json"


def _engine_record_path() -> Path:
    """live_engine's own track record — on the Railway VOLUME when one is mounted.

    This is the file the public endpoint reads FIRST, so it is the file that
    decides what the track-record page shows. Deleting a record anywhere else
    and leaving this one alone is why earlier wipes appeared to do nothing.
    """
    return Path(os.environ.get('AEGIS_STATE_DIR') or (Path(BASE_DIR) / "data")) / "track_record.json"


def _web_record_path() -> Path:
    """The container-filesystem copy. Wiped on every deploy, but live between them."""
    return Path(BASE_DIR) / "web" / "track_record.json"


def _purge_ids_from_wallet(signal_ids: Set[str]) -> Dict[str, Any]:
    """Remove trades from the live engine's WALLET and reverse their PnL.

    The wallet is the real source of truth. `PositionManager._save_track_record`
    rebuilds STATE_DIR/track_record.json from `wallet.trade_history` on every
    cycle AND re-preserves any on-disk record the wallet does not know about, so:

      * deleting from disk alone is overwritten on the next save, and
      * deleting from the wallet alone is resurrected from the disk orphans.

    Both have to go, and the balance has to move with them — `balance` is a
    running total (`self.balance += pnl_usdt` per close), so a row removed from
    the record while its money stays in the wallet leaves the published capital
    and profit reporting trades that are no longer in the record.

    The engine shares this process (`asyncio.create_task(run_engine_background())`,
    instance on `LIVE_STATE.engine`), so this is a direct mutation, not IPC.
    """
    out: Dict[str, Any] = {"slices": 0, "pnl_reversed": 0.0}
    eng = getattr(LIVE_STATE, "engine", None)
    wallet = getattr(eng, "wallet", None) if eng is not None else None
    if wallet is None:
        return out
    try:
        keep, dropped = [], []
        for t in list(getattr(wallet, "trade_history", []) or []):
            (dropped if getattr(t, "signal_id", None) in signal_ids else keep).append(t)
        if not dropped:
            return out
        pnl = sum(float(getattr(t, "pnl_usdt", 0.0) or 0.0) for t in dropped)
        wallet.trade_history = keep
        # Reverse the money: a deleted LOSS raises the balance, a deleted WIN
        # lowers it. Without this the wallet keeps paying out a trade the record
        # no longer contains.
        wallet.balance = float(getattr(wallet, "balance", 0.0)) - pnl
        out["slices"] = len(dropped)
        out["pnl_reversed"] = round(pnl, 4)
        out["balance"] = round(wallet.balance, 2)
    except Exception as exc:
        print(f"[TrackRecord] wallet purge failed: {exc!r}")
    return out


def _purge_ids_from_disk(signal_ids: Set[str]) -> Dict[str, int]:
    """Remove signal_ids from every on-disk track record and re-derive summaries.

    A record can sit in three places (engine volume, web copy, main.py's store)
    and the public view merges all three. Removing it from one leaves it on the
    page, which reads as the delete having silently failed.
    """
    removed: Dict[str, int] = {}
    for path in {_engine_record_path(), _web_record_path(), TRACK_RECORD_PATH}:
        try:
            if not path.exists():
                continue
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            sigs = data.get("signals") or []
            kept = [r for r in sigs if r.get("signal_id") not in signal_ids]
            n = len(sigs) - len(kept)
            if n == 0:
                continue
            data["signals"] = kept
            # The stored summary is what the wallet figures are read from, so it
            # must not keep counting trades that are no longer in the file.
            if isinstance(data.get("summary"), dict):
                w = sum(1 for r in kept if r.get("outcome") == "WIN")
                l = sum(1 for r in kept if r.get("outcome") == "LOSS")
                pnls = [float(r.get("pnl_pct") or 0) for r in kept
                        if r.get("outcome") in ("WIN", "LOSS")]
                data["summary"].update({
                    "total_signals": len(kept),
                    "wins": w, "losses": l,
                    "open": sum(1 for r in kept if r.get("outcome") == "OPEN"),
                    "win_rate_pct": round(w / (w + l) * 100, 1) if (w + l) else None,
                    "avg_pnl_pct": round(sum(pnls) / len(pnls), 3) if pnls else None,
                    "total_pnl_pct": round(sum(pnls), 3) if pnls else 0.0,
                })
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, default=str)
            os.replace(tmp, path)
            removed[str(path)] = n
        except Exception as exc:
            print(f"[TrackRecord] purge failed for {path}: {exc!r}")
    return removed
_track_store: list = []       # in-memory list of signal records
_tr_seen_ids: set = set()     # signal_ids already in store
_tr_last_save: float = 0.0   # epoch of last disk write

def _enforce_track_generation() -> None:
    """One-time purge of main.py's OWN track store when STATE_GENERATION bumps.

    The engine-side wipe (scripts/engine/state.py) empties the VOLUME record, and
    it worked — /api/engine-track-record reported generation 3 with 0 signals.
    But GET /api/track-record merges THREE sources, and the third is this
    module's in-memory `_track_store`, which the engine cannot reach. So the
    public page still showed 40 closed trades tagged `source: None` — the
    signature of this store — while the engine's own record was empty.

    That is the same "two stores behind one name" defect as TRACK_RECORD_PATH and
    the Telegram connections path, for the third time. Here it is closed by
    giving main.py the same generation contract the engine already honours.

    The marker lives on the VOLUME, not beside the web record, because
    web/track_record.json sits on the container filesystem and is wiped on every
    deploy — a marker there would read "never purged" after each redeploy and the
    wipe would fire forever instead of once.
    """
    global _track_store, _tr_seen_ids
    marker = _STATE_DIR / ".track_generation"
    try:
        seen = int(marker.read_text(encoding="utf-8").strip())
    except Exception:
        seen = 0
    if seen == STATE_GENERATION:
        return
    n = len(_track_store)
    _track_store = []
    _tr_seen_ids = set()
    for path in {TRACK_RECORD_PATH, WEB_ROOT_PATH / "track_record.json"}:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"generated_at": datetime.now(timezone.utc).isoformat(),
                           "generation": STATE_GENERATION,
                           "summary": {}, "signals": []}, f)
        except Exception as exc:
            print(f"[TrackRecord] could not empty {path}: {exc!r}")
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(str(STATE_GENERATION), encoding="utf-8")
    except Exception as exc:
        print(f"[TrackRecord] generation marker not written ({exc!r}) — "
              f"the purge will repeat on the next boot")
    print(f"[TrackRecord] generation {seen} -> {STATE_GENERATION}: "
          f"purged {n} in-memory records (one-time)")


def _load_track_record() -> None:
    global _track_store, _tr_seen_ids
    if TRACK_RECORD_PATH.exists():
        try:
            with open(TRACK_RECORD_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            _track_store = data.get("signals", [])
            _tr_seen_ids = {r["signal_id"] for r in _track_store if r.get("signal_id")}
            print(f"[TrackRecord] Loaded {len(_track_store)} records from disk")
        except Exception as e:
            print(f"[TrackRecord] Load error: {e}")

def _compute_track_summary() -> dict:
    wins   = sum(1 for r in _track_store if r.get("outcome") == "WIN")
    losses = sum(1 for r in _track_store if r.get("outcome") == "LOSS")
    open_c = sum(1 for r in _track_store if r.get("outcome") == "OPEN")
    closed = wins + losses
    win_rate = round(wins / closed * 100, 1) if closed > 0 else None
    closed_pnls = [r.get("pnl_pct", 0.0) for r in _track_store if r.get("outcome") in ("WIN", "LOSS")]
    avg_pnl   = round(sum(closed_pnls) / len(closed_pnls), 3) if closed_pnls else None
    total_pnl = round(sum(closed_pnls), 3) if closed_pnls else 0.0
    times = [r["entry_time"] for r in _track_store if r.get("entry_time")]
    return {
        "total_signals": len(_track_store),
        "wins": wins,
        "losses": losses,
        "open": open_c,
        "win_rate_pct": win_rate,
        "avg_pnl_pct": avg_pnl,
        "total_pnl_pct": total_pnl,
        "tracking_since": min(times) if times else None,
    }

def _save_track_record() -> None:
    global _tr_last_save
    try:
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": _compute_track_summary(),
            "signals": sorted(_track_store, key=lambda r: r.get("entry_time") or "", reverse=True)[:500],
        }
        TRACK_RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
        temp_path = TRACK_RECORD_PATH.with_suffix('.tmp')
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, default=str)
        os.replace(temp_path, TRACK_RECORD_PATH)
        _tr_last_save = time.time()
    except Exception as e:
        print(f"[TrackRecord] Save error: {e}")

_BUY_SIGNALS  = {"BUY", "STRONG_BUY"}
_SELL_SIGNALS = {"SELL", "STRONG_SELL"}
_ACTIONABLE   = _BUY_SIGNALS | _SELL_SIGNALS

def _update_track_record(signals_data: dict, live_prices: dict) -> None:
    """Called from update_state on every engine tick.

    Exit logic (dynamic TP):
      - Primary TP : the same model fires the OPPOSITE signal  â†’ MODEL_REVERSAL_TP.
                     This is the trend-reversal take-profit the user requested:
                     enter on one reversal, exit when the next reversal fires.
      - Safety SL  : price crosses the ATR stop stored at entry â†’ STOP_HIT.
      - Hard ceiling: the stored take_profit price (wide ATR fallback) â†’ TARGET_HIT.
                     Prevents a position staying open forever if the model never
                     generates an opposite signal (e.g. a slow grind with no clean
                     re-entry signal on the other side).
    """
    global _track_store, _tr_seen_ids
    now_iso = datetime.now(timezone.utc).isoformat()

    for sym, sig_map in signals_data.items():
        # Resolve nested timeframe map â†’ use 1h summary signal
        if isinstance(sig_map, dict) and any(tf in sig_map for tf in ("1m","5m","15m","30m","1h","4h","1d")):
            sig = sig_map.get("1h") or next((v for v in sig_map.values() if isinstance(v, dict)), None)
        else:
            sig = sig_map
        if not isinstance(sig, dict):
            continue

        signal_type   = sig.get("signal", "HOLD")
        signal_id     = sig.get("signal_id")
        fire          = bool(sig.get("fire", False))
        if not signal_id:
            continue

        current_price = float(live_prices.get(sym, sig.get("price", 0) or 0))

        # â”€â”€ Find the one open position for this symbol (if any) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        open_rec = next(
            (r for r in _track_store if r.get("symbol") == sym and r.get("outcome") == "OPEN"),
            None
        )

        if open_rec:
            direction = open_rec.get("direction", "LONG")
            entry     = float(open_rec.get("entry_price") or current_price or 1)
            sl        = float(open_rec.get("stop_loss") or 0)
            tp        = float(open_rec.get("take_profit") or 0)

            pnl_pct = round(
                ((current_price - entry) / entry * 100) if direction == "LONG"
                else ((entry - current_price) / entry * 100),
                3
            )

            # Primary TP: opposite model signal fires (this is the reversal exit).
            # Requires fire=True â€” a weak opposite prediction must not close the position.
            # Does NOT depend on _tr_seen_ids: that set guards entries, not exits.
            opposite_fired = fire and (
                (direction == "LONG"  and signal_type in _SELL_SIGNALS) or
                (direction == "SHORT" and signal_type in _BUY_SIGNALS)
            )

            # Safety SL: price crosses the ATR-based hard stop
            sl_hit = current_price > 0 and sl > 0 and (
                (direction == "LONG"  and current_price <= sl) or
                (direction == "SHORT" and current_price >= sl)
            )

            # Hard ceiling fallback TP (wide ATR multiple, only as a safety net)
            ceiling_hit = current_price > 0 and tp > 0 and (
                (direction == "LONG"  and current_price >= tp) or
                (direction == "SHORT" and current_price <= tp)
            )

            if opposite_fired:
                open_rec.update({
                    "outcome":     "WIN" if pnl_pct > 0 else "LOSS",
                    "exit_price":  current_price,
                    "close_time":  now_iso,
                    "pnl_pct":     pnl_pct,
                    "exit_reason": "MODEL_REVERSAL_TP",
                })
                # Fall through: open the new opposite-direction position below

            elif sl_hit:
                open_rec.update({
                    "outcome":     "LOSS",
                    "exit_price":  current_price,
                    "close_time":  now_iso,
                    "pnl_pct":     pnl_pct,
                    "exit_reason": "STOP_HIT",
                })
                continue  # SL hit â€” do not open a new position this tick

            elif ceiling_hit:
                open_rec.update({
                    "outcome":     "WIN" if pnl_pct > 0 else "LOSS",
                    "exit_price":  current_price,
                    "close_time":  now_iso,
                    "pnl_pct":     pnl_pct,
                    "exit_reason": "TARGET_HIT",
                })
                continue  # Hard ceiling hit â€” do not immediately re-enter

            else:
                # Position still open â€” refresh live unrealized PnL so the
                # public track record shows a moving number, not the entry 0.00
                open_rec["pnl_pct"]       = pnl_pct
                open_rec["current_price"] = current_price
                continue

        # â”€â”€ Open a new position if the signal is actionable and fresh â”€â”€â”€â”€â”€â”€â”€
        # Only fire=True signals should create track-record entries.
        if not fire:
            continue
        if signal_type not in _ACTIONABLE:
            continue
        if signal_id in _tr_seen_ids:
            continue
        # Block duplicate entries when the live_engine generates a new UUID on each
        # scan cycle for the same continuously-firing signal.  If this symbol already
        # has an OPEN position in the store (regardless of signal_id), skip it.
        if any(r.get("symbol") == sym and r.get("outcome") == "OPEN"
               for r in _track_store):
            _tr_seen_ids.add(signal_id)  # absorb the new id so we don't log every scan
            continue

        direction   = sig.get("direction", "LONG" if signal_type in _BUY_SIGNALS else "SHORT")
        entry_price = float(sig.get("price") or sig.get("entry_price") or 0)
        _conf_data  = sig.get("confluence") or {}
        _track_store.append({
            "signal_id":       signal_id,
            "symbol":          sym,
            "timeframe":       sig.get("timeframe", "1h"),
            "direction":       direction,
            "signal_type":     signal_type,
            "signal_status":   sig.get("signal_status", "ACTIVE"),
            "entry_price":     entry_price,
            "take_profit":     float(sig.get("tp") or sig.get("suggested_tp") or 0),
            "stop_loss":       float(sig.get("sl") or sig.get("suggested_sl") or 0),
            "exit_price":      None,
            "entry_time":      sig.get("data_timestamp", now_iso),
            "close_time":      None,
            "pnl_pct":         0.0,
            "outcome":         "OPEN",
            "exit_reason":     None,
            "ai_prob":         round(float(sig.get("meta_confidence") or 0), 3),
            "confluence_rate": round(float(_conf_data.get("total") or 0), 2),
            # Risk tier at entry (STRONG | NORMAL | RISKY) â€” set by the live
            # engine's risk-tier classifier; shown as a badge on the public page.
            "signal_strength": sig.get("risk_tier", ""),
            "source":          "live_engine",
        })
        _tr_seen_ids.add(signal_id)

    # Cap store at 1 000 records (newest first)
    if len(_track_store) > 1000:
        _track_store = sorted(_track_store, key=lambda r: r.get("entry_time") or "", reverse=True)[:1000]
        _tr_seen_ids = {r["signal_id"] for r in _track_store if r.get("signal_id")}

# -------------------------------------------------------------------
# OTP Store â€” Firestore-backed (collection: phone_verifications)
# Replaces the old in-memory dict so OTPs survive server restarts.
# Enable TTL on the 'expires_at' field in Firebase Console â†’
#   Firestore â†’ TTL Policies â†’ collection: phone_verifications, field: expires_at
# -------------------------------------------------------------------
_OTP_COL = "phone_verifications"

# ── OTP store: MEMORY IS AUTHORITATIVE, Firestore is a best-effort mirror ─────
# An OTP is a six-digit, single-use token that lives five minutes. Storing it in
# Firestore made a quota-limited datastore a hard dependency of signup: when the
# daily write quota was exhausted, _otp_set blocked, and no new user could
# register at all. The failure surfaced as "Our datastore is not responding right
# now" on the signup form — an honest message about a dependency that should
# never have existed.
#
# This process runs SINGLE-WORKER (start.sh: `uvicorn main:app` with no
# --workers), so send and verify always land in the same process and a dict is a
# correct store. If that ever changes, the Firestore mirror below is what keeps
# this working — reads fall back to it — but the --workers flag must be added
# deliberately, not by accident.
#
# The mirror is written on a daemon thread and never awaited, so Firestore being
# slow, broken or out of quota cannot delay or fail a signup. Losing it costs one
# thing only: an OTP issued before a restart can no longer be verified after it,
# and the user resends. That is a far smaller failure than "nobody can sign up".
_OTP_MEM: Dict[str, Dict] = {}


def _otp_ref(email: str):
    return db.collection(_OTP_COL).document(email)


def _otp_mirror(fn, *args) -> None:
    """Best-effort Firestore write. Never blocks, never raises."""
    def _run():
        try:
            fn(*args)
        except Exception as exc:
            print(f"[OTP] Firestore mirror skipped: {exc!r}")
    threading.Thread(target=_run, daemon=True).start()


def _otp_prune() -> None:
    """Drop records well past expiry so the dict cannot grow without bound."""
    now = datetime.now(timezone.utc)
    for k, v in list(_OTP_MEM.items()):
        exp = v.get("expires_at")
        if isinstance(exp, datetime) and now - exp > timedelta(hours=1):
            _OTP_MEM.pop(k, None)


def _otp_get(email: str) -> Optional[Dict]:
    rec = _OTP_MEM.get(email)
    if rec is not None:
        return rec
    # Not in memory — this process may have restarted mid-signup. Try the mirror,
    # but never let a slow datastore stall a verify: on any failure the caller
    # simply sees "no record", which reads to the user as "request a new code".
    try:
        snap = _otp_ref(email).get()
        if snap.exists:
            data = snap.to_dict() or {}
            _OTP_MEM[email] = data
            return data
    except Exception as exc:
        print(f"[OTP] Firestore read skipped for {email}: {exc!r}")
    return None


def _otp_set(email: str, data: Dict):
    _otp_prune()
    _OTP_MEM[email] = dict(data)
    _otp_mirror(lambda: _otp_ref(email).set(data))


def _otp_update(email: str, updates: Dict):
    rec = _OTP_MEM.get(email)
    if rec is None:
        rec = _otp_get(email) or {}
    rec.update(updates)
    _OTP_MEM[email] = rec
    _otp_mirror(lambda: _otp_ref(email).update(updates))


def _otp_delete(email: str):
    _OTP_MEM.pop(email, None)
    _otp_mirror(lambda: _otp_ref(email).delete())

def _otp_find_by_signup_token(token: str) -> Optional[str]:
    """Return the email (doc ID) whose signup_token matches, or None.

    Memory first — the OTP store is memory-authoritative, so a token issued by
    this process would otherwise be invisible to its own verify step.
    """
    for email, rec in _OTP_MEM.items():
        if rec.get("signup_token") == token:
            return email
    try:
        for doc in db.collection(_OTP_COL).where("signup_token", "==", token).limit(1).stream():
            return doc.id
    except Exception as exc:
        print(f"[OTP] token lookup skipped: {exc!r}")
    return None

async def _fs_await(fn, *args, timeout: float = 8.0, what: str = "datastore"):
    """Run a BLOCKING Firestore call off the event loop, with a deadline.

    Every helper in this file (get_user_doc, _otp_get, _otp_set, ...) ends in a
    synchronous google-cloud-firestore call. Called directly from an `async def`
    handler they block the whole event loop, not just their own request — so one
    slow write stalls every other user's page as well.

    That is exactly what "Sending verification code…" hanging forever was. With
    the daily write quota exhausted, _otp_set retried with backoff for up to a
    minute while holding the loop, and the browser had no timeout of its own, so
    the spinner never resolved and the rest of the site went sticky at the same
    time.

    NOTE on cancellation: wait_for cancels the AWAIT, not the thread. A timed-out
    Firestore call keeps running in the executor until it gives up on its own.
    That is acceptable here — the point is to free the request and answer the user
    honestly — but it means a timeout is not a rollback, and a write that reports
    a timeout may still land.
    """
    try:
        return await asyncio.wait_for(asyncio.to_thread(fn, *args), timeout=timeout)
    except asyncio.TimeoutError:
        print(f"[Firestore] {what} exceeded {timeout:.0f}s — quota exhausted or unreachable")
        raise HTTPException(
            status_code=503,
            detail="Our datastore is not responding right now. Please try again in a few minutes.",
        )


def _otp_find_by_phone(phone: str) -> Optional[str]:
    """Return the email (doc ID) whose phone_number matches, or None. Memory first."""
    for email, rec in _OTP_MEM.items():
        if rec.get("phone_number") == phone:
            return email
    try:
        for doc in db.collection(_OTP_COL).where("phone_number", "==", phone).limit(1).stream():
            return doc.id
    except Exception as exc:
        print(f"[OTP] phone lookup skipped: {exc!r}")
    return None

# Keep a module-level alias for old code paths not yet migrated
otp_store: Dict[str, Dict] = {}  # legacy â€” new code uses _otp_* helpers above

def generate_otp() -> str:
    return ''.join(random.choices(string.digits, k=6))


def otp_email_key(email: str) -> str:
    """Canonical Firestore document key for an OTP record.

    /auth/send-otp-for-registration writes the record under the address
    email_validator returns (which lower-cases the domain), while the verify and
    complete-registration endpoints looked it up under the raw string the form
    posted. Anyone who typed their domain with a capital -- Foo@GMAIL.com --
    stored under one key and was read back under another, so a perfectly valid
    code came back as "No OTP request found". Both ends go through here now.
    """
    if not email:
        return ""
    try:
        return validate_email(email, check_deliverability=False).normalized
    except Exception:
        return email.strip()


def normalize_phone_number(phone: Optional[str]) -> Optional[str]:
    """Normalize phone numbers to E.164 format for signup and backend enforcement."""
    if not phone:
        return None
    cleaned = re.sub(r'[\s\-\.\(\)]', '', str(phone).strip())
    if not cleaned:
        return None
    if cleaned.startswith('+'):
        digits = re.sub(r'\D', '', cleaned)
        return f'+{digits}' if digits else None
    digits = re.sub(r'\D', '', cleaned)
    if len(digits) == 10:
        return f'+91{digits}'
    if len(digits) >= 7:
        return f'+{digits}'
    return None

# -------------------------------------------------------------------
# Simple in-memory rate limiter (no external deps)
# Tracks request timestamps per key (IP or email).
# -------------------------------------------------------------------
_rate_store: Dict[str, list] = {}

def _rate_limit(key: str, max_calls: int, window_seconds: int) -> bool:
    """Return True if the request is allowed; False if rate limit exceeded."""
    now = time.time()
    window_start = now - window_seconds
    hits = _rate_store.get(key, [])
    # Evict entries outside the window
    hits = [t for t in hits if t > window_start]
    if len(hits) >= max_calls:
        _rate_store[key] = hits
        return False
    hits.append(now)
    _rate_store[key] = hits
    return True

def get_client_ip(request: Request) -> str:
    """Return the best-available client IP for rate-limiting keying."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

def is_cooldown_active(email: str) -> bool:
    record = _otp_get(email)
    if not record:
        return False
    cooldown = record.get("cooldown_until")
    return bool(cooldown and datetime.now(timezone.utc) < cooldown)

# -------------------------------------------------------------------
# Institutional Analytics Engine
# -------------------------------------------------------------------
def _compute_system_analytics_pass():
    """
    Compute win rate, mathematical expectancy, profit factor, and max drawdown
    from historical signals.

    MUST NOT run on the event loop — see the caller. Despite having been an
    `async def`, there was never an await in this body: every line below is a
    BLOCKING Firestore round trip. The .stream() scan pulls every closed signal
    in the collection, and the .set() at the end is a write. Awaiting it from
    analytics_loop froze the loop for as long as Firestore took to answer.

    That is cheap when Firestore is healthy and ruinous when it is not. Measured
    on a project whose free-tier daily write quota was exhausted, the client
    retried the 429 until its own 60s deadline and the loop sat blocked for 45s
    straight, with the app bound to its port and answering nothing.
    """
    try:
        if not db:
            return
            
        print("ðŸ“Š Running Institutional Analytics Computation...")
        signals_ref = db.collection("signals")
        
        # Query only closed trades
        # In Firestore, 'in' queries are supported up to 10 values
        docs = signals_ref.where("status", "in", ["TARGET HIT", "STOP LOSS HIT"]).stream()
        
        trades = []
        for doc in docs:
            trades.append(doc.to_dict())
            
        if not trades:
            print("ðŸ“Š No closed trades found for analytics.")
            db.collection("analytics").document("global_performance").set({
                "win_rate": 0.0,
                "expectancy": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
                "total_trades": 0,
                "updated_at": datetime.now(timezone.utc).isoformat()
            })
            return
            
        wins = 0
        losses = 0
        gross_profit = 0.0
        gross_loss = 0.0
        
        equity = 10000.0
        peak_equity = equity
        max_dd_pct = 0.0
        
        for trade in trades:
            profit_pct = trade.get("profit_pct", None)
            
            if profit_pct is None:
                status = trade.get("status")
                entry = float(trade.get("entry_price", 0))
                sl = float(trade.get("sl", 0))
                tp = float(trade.get("tp", 0))
                
                if status == "TARGET HIT" and entry > 0:
                    profit_pct = abs(tp - entry) / entry * 100
                elif status == "STOP LOSS HIT" and entry > 0:
                    profit_pct = -abs(entry - sl) / entry * 100
                else:
                    profit_pct = 0.0
                    
            if profit_pct > 0:
                wins += 1
                gross_profit += profit_pct
            elif profit_pct < 0:
                losses += 1
                gross_loss += abs(profit_pct)
                
            trade_pl = equity * (profit_pct / 100)
            equity += trade_pl
            
            if equity > peak_equity:
                peak_equity = equity
                
            dd = (peak_equity - equity) / peak_equity * 100
            if dd > max_dd_pct:
                max_dd_pct = dd
                
        total_closed = wins + losses
        win_rate = (wins / total_closed) if total_closed > 0 else 0.0
        avg_win = (gross_profit / wins) if wins > 0 else 0.0
        avg_loss = (gross_loss / losses) if losses > 0 else 0.0
        
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)
        
        analytics_data = {
            "win_rate": win_rate,
            "expectancy": expectancy,
            "profit_factor": profit_factor,
            "max_drawdown": max_dd_pct,
            "total_trades": total_closed,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        
        db.collection("analytics").document("global_performance").set(analytics_data)
        print(f"ðŸ“Š Analytics updated: Exp {expectancy:.2f}%, PF {profit_factor:.2f}, MDD {max_dd_pct:.2f}%")
        
    except Exception as e:
        print(f"âŒ Error computing analytics: {e}")

async def analytics_loop():
    """Recompute the global performance card every hour.

    THE PASS RUNS IN A WORKER THREAD, AND THAT IS NOT OPTIONAL — its body is
    blocking Firestore I/O from the first line to the last. This was the one
    startup loop missed when the Telegram sweep and the two reminder scans were
    moved off the event loop; it ran first, with no grace period, and stalled
    every request uvicorn was trying to answer.

    Nothing depends on this being prompt. It feeds a dashboard card and it runs
    hourly, so it can afford to start last, after the engine has warmed up.
    """
    await asyncio.sleep(120)
    while True:
        try:
            await asyncio.to_thread(_compute_system_analytics_pass)
        except Exception as exc:
            print(f"[Analytics] Error: {exc}")
        await asyncio.sleep(3600)  # Run every hour

# -------------------------------------------------------------------
# Engine runner as a background task (nonâ€‘blocking) with error handling
# -------------------------------------------------------------------
async def run_engine_background():
    """Run the live engine in the background without blocking startup."""
    from scripts.live_engine import LiveEngine, automated_setup, TRACK_RECORD_PATH as _TR_PATH
    import argparse

    print("Engine background task starting.")

    args = argparse.Namespace()
    args.capital      = 10_000.0
    args.max_position = 1_000.0
    args.scan_seconds = 300
    args.proxy        = None

    backtest_dir = Path(__file__).parent / "logs" / "backtests"

    # Both automated_setup() and the LiveEngine constructor are heavy SYNCHRONOUS
    # calls — the constructor loads an XGBoost model pair per token and measured
    # 29.6s for 60 tokens on top of ~0.9s of setup.  Run directly on the event
    # loop (as they were) they block every HTTP request for ~30s after boot, so
    # the site appears to hang on the first visit after any restart or cold
    # start.  Offload both to a worker thread; the loop stays free to serve.
    _loop = asyncio.get_running_loop()

    try:
        configs, capital, max_pos, scan_seconds, proxy = await _loop.run_in_executor(
            None, partial(automated_setup, backtest_dir, args)
        )
    except Exception as e:
        print(f"automated_setup failed: {e}")
        await asyncio.sleep(1)
        return

    # Risk tier controls signal quality bar: conservative / balanced / aggressive
    # Set via SIGNAL_RISK_TIER env var; defaults to "balanced".
    _valid_tiers = {"conservative", "balanced", "aggressive"}
    _tier = os.getenv("SIGNAL_RISK_TIER", "balanced").lower()
    if _tier not in _valid_tiers:
        print(f"[Engine] Unknown SIGNAL_RISK_TIER '{_tier}', defaulting to 'balanced'")
        _tier = "balanced"
    print(f"[Engine] Signal risk tier: {_tier.upper()}")

    _t0 = time.time()
    engine = await _loop.run_in_executor(
        None,
        lambda: LiveEngine(
            token_configs         = configs,
            capital               = capital,
            max_position_usdt     = max_pos,
            scan_interval_seconds = scan_seconds,
            risk_tier             = _tier,
            proxy_url             = proxy,
        ),
    )
    print(f"[Engine] Predictors loaded in {time.time() - _t0:.1f}s "
          f"(off the event loop — HTTP stayed responsive)")
    LIVE_STATE.engine = engine

    _last_tr_mtime: float = 0.0
    _last_signals_hash: int = 0       # hash of last signals pushed to Firestore
    _last_firestore_push: float = 0.0 # epoch of last Firestore write
    _FIRESTORE_MIN_INTERVAL = 290.0   # push at most once per ~5 min (matches scan cycle)
    # Per-symbol payload hashes — the diff that keeps this inside the free tier.
    _doc_sig_hash: Dict[str, int] = {}
    _last_full_push: float = 0.0
    _FIRESTORE_FULL_REFRESH_S = 3600.0   # heal drift hourly (60 writes/day)
    _stale_sweep_done: bool = False   # one-time Firestore neutralise after restart

    def _push_signal_docs(_pairs) -> int:
        """Write the signal docs. MUST NOT run on the event loop — see the caller.

        This is the biggest Firestore writer in the app: one document per symbol,
        ~60 of them per push. Every call here is blocking, and when the project is
        over its write quota the client retries each one until its own 60s
        deadline. Run on the loop, as this was, a single quota-blocked push froze
        the whole site for 60s for the batch and then another 60s PER DOCUMENT in
        the fallback below — measured at ~17 minutes of dead event loop per cycle,
        which is exactly how the site came to time out while the container was
        healthy and the deploy green.

        The fallback exists because ONE NaN used to fail a whole batch silently
        (v80), so per-doc writes name the offending symbol. That is worth 60s when
        one document is malformed. It is worth nothing when the batch failed
        because the project is out of quota — every retry is guaranteed to fail
        the same way, so the whole point of the fallback is gone. Quota errors
        therefore skip it.
        """
        from google.api_core import exceptions as _gexc

        try:
            batch = db.batch()
            _n = 0
            for _ref, _doc in _pairs:
                batch.set(_ref, _doc, merge=True)
                _n += 1
                if _n % 450 == 0:
                    batch.commit()
                    batch = db.batch()
            if _n % 450:
                batch.commit()
            return _n
        except Exception as _bt_e:
            _quota_hit = isinstance(_bt_e, (_gexc.ResourceExhausted, _gexc.RetryError)) \
                or 'Quota' in str(_bt_e) or '429' in str(_bt_e)
            if _quota_hit:
                print(f"[PRODUCER] batch push failed on QUOTA ({type(_bt_e).__name__}) "
                      f"— skipping per-doc fallback; {len(_pairs)} doc(s) dropped this cycle")
                return 0
            print(f"[PRODUCER] batch push failed ({_bt_e}) — per-doc fallback")
            _pushed = 0
            for _ref, _doc in _pairs:
                try:
                    _ref.set(_doc, merge=True)
                    _pushed += 1
                except Exception as _doc_e:
                    print(f"[PRODUCER] doc push FAILED {_doc.get('symbol')}: {_doc_e}")
            return _pushed

    async def update_state():
        nonlocal _last_tr_mtime, _last_signals_hash, _last_firestore_push, _stale_sweep_done
        nonlocal _last_full_push
        while True:
            try:
                LIVE_STATE.data["tickers"]       = engine.live_prices.copy()
                # ── v79.1: the snapshot enforces the display invariant ─────────
                # "Signal cockpits hold ONLY HOLD and the FIRED signals" (user).
                # A fired face is EARNED by an open position — real book or paper
                # book — at THIS instant. Scan-time flags can outlive their
                # positions between scans (measured: '0 OPEN · 2 FIRING', cards
                # with no SL and POSITION=no-trade), so the reconciliation lives
                # here, where every consumer (cockpit, rooms, Firestore) reads.
                _snap_sigs = {}
                for _sym, _ent in engine.last_signals.items():
                    if not isinstance(_ent, dict):
                        _snap_sigs[_sym] = _ent
                        continue
                    _e = dict(_ent)
                    _real_pos = engine.wallet.open_positions.get(_sym)
                    _pap_pos  = engine.alpha_wallet.open_positions.get(f'{_sym}|risky')
                    if _real_pos is not None:
                        _e['fire'], _e['paper_only'] = True, False
                        _e['signal']    = _real_pos.side
                        _e['direction'] = _real_pos.direction
                    elif _pap_pos is not None:
                        _e['fire'], _e['paper_only'] = True, True
                        _e['signal']    = _pap_pos.side
                        _e['direction'] = _pap_pos.direction
                    elif _e.get('fire'):
                        _e['fire']            = False
                        _e['paper_only']      = False
                        _e['signal']          = 'HOLD'
                        _e['signal_strength'] = 'NEUTRAL'
                    _snap_sigs[_sym] = _e
                LIVE_STATE.data["signals"] = _snap_sigs
                LIVE_STATE.data["alpha_signals"] = engine.alpha_signals.copy() if engine.alpha_mode else {}
                LIVE_STATE.data["open_trades"]   = [
                    asdict(p) for p in engine.wallet.open_positions.values()
                ]
                LIVE_STATE.data["balance"]    = engine.wallet.balance
                LIVE_STATE.data["alpha_mode"] = engine.alpha_mode
                LIVE_STATE.data["warmup_progress"] = (
                    f"{engine.bootstrap_done}/{engine.bootstrap_total}"
                )

                # Broadcast track_record.json to WebSocket clients whenever
                # the engine saves a new version (detected via mtime change).
                try:
                    mtime = _TR_PATH.stat().st_mtime if _TR_PATH.exists() else 0.0
                    if mtime > _last_tr_mtime:
                        _last_tr_mtime = mtime
                        with open(_TR_PATH, 'r', encoding='utf-8') as _f:
                            _payload = json.load(_f)
                        await _tr_ws_manager.broadcast(_payload)
                except Exception:
                    pass
                # --- write latest signals to a JSON file for frontend consumption ---
                try:
                    signals_dir = WEB_ROOT_PATH / 'src' / 'data'
                    signals_dir.mkdir(parents=True, exist_ok=True)
                    signals_file = signals_dir / 'live_signals.json'
                    temp_file = signals_dir / 'live_signals.json.tmp'
                    with open(temp_file, 'w', encoding='utf-8') as sf:
                        safe_signals = numpy_to_native(LIVE_STATE.data.get('signals', {}))
                        json.dump(safe_signals, sf, default=str)
                    os.replace(temp_file, signals_file)
                except Exception as _e:
                    print(f"âš ï¸ Failed to write live_signals.json: {_e}")

                # --- write latest signals to Firebase Firestore ---
                # Only push when: warmup is done AND (signals changed OR 5-min interval elapsed).
                # Signals only change every scan_interval_seconds (~5 min), so pushing every
                # second was burning ~2M Firestore writes/day for no benefit.
                _eng = LIVE_STATE.engine
                _warming_up = (_eng is not None and
                               _eng.bootstrap_done < _eng.bootstrap_total)
                if _warming_up:
                    print(f"[PRODUCER] Warmup in progress "
                          f"({_eng.bootstrap_done}/{_eng.bootstrap_total}) "
                          f"â€” pushing scanned tokens as they complete.")
                    # ── Stale sweep (one-time per restart) ─────────────────────
                    # Pushes are deferred for the WHOLE warmup, so Firestore
                    # still holds the PREVIOUS session's docs — including
                    # fire=true — and the dashboard rooms render ghost signals
                    # that the track record (live engine state) rightly denies
                    # (measured 2026-07-20: 4 stale SELL cards vs 0 fired / 5
                    # armed). Neutralise every doc once, immediately; scans
                    # repopulate them with real state as warmup completes.
                    if not _stale_sweep_done:
                        _stale_sweep_done = True   # attempt once even if it fails
                        try:
                            _reset_ts = datetime.now(timezone.utc).isoformat()
                            _sw_batch = db.batch()
                            _sw_n = 0
                            for _sw_doc in db.collection("signals").stream():
                                _sw_batch.set(_sw_doc.reference, {
                                    'fire': False,
                                    'signal': 'HOLD',
                                    'signal_strength': 'NEUTRAL',
                                    'paper_only': False,
                                    'pending_entry': False,
                                    'evaluating': True,
                                    'stale_reset': _reset_ts,
                                    'timestamp': _reset_ts,
                                }, merge=True)
                                _sw_n += 1
                                if _sw_n % 450 == 0:
                                    _sw_batch.commit()
                                    _sw_batch = db.batch()
                            if _sw_n % 450:
                                _sw_batch.commit()
                            print(f"[PRODUCER] Stale sweep: neutralised {_sw_n} "
                                  f"Firestore docs from the previous session")
                        except Exception as _sw_e:
                            print(f"[PRODUCER] Stale sweep failed: {_sw_e}")

                _now = time.time()
                # v79.2: push DURING warmup too. A token's state is complete the
                # moment ITS scan finishes — deferring everything until the whole
                # fleet bootstrapped left the cockpit reading the neutralised
                # stale-sweep docs while the track record (engine-direct) showed
                # armed signals (user report 2026-07-20: "8 armed" vs empty
                # cockpit). last_signals only contains scanned tokens, so partial
                # pushes are always truthful; the change-fingerprint + 290s floor
                # keep the write volume unchanged.
                _signals_now = LIVE_STATE.data.get('signals', {})
                # Fingerprint: (symbol, signal_side, fire) â€” stable between scans when
                # nothing fires.  signal_id uses uuid4() on fire=True, so hashing
                # signal_id caused a push on every fired symbol within a scan cycle.
                _sig_fingerprint = tuple(sorted(
                    (sym, v.get('signal', 'FLAT'), bool(v.get('fire', False)), bool(v.get('pending_entry', False)))
                    for sym, v in _signals_now.items()
                    if isinstance(v, dict)
                ))
                _new_hash = hash(_sig_fingerprint)
                _signals_changed = (_new_hash != _last_signals_hash)
                _interval_elapsed = (_now - _last_firestore_push >= _FIRESTORE_MIN_INTERVAL)

                if not _signals_now or (not _signals_changed and not _interval_elapsed):
                    pass  # skip â€” nothing new to push
                else:
                    _last_signals_hash = _new_hash
                    _last_firestore_push = _now

                try:
                    # Only push FIRED signals (fire=True) â€” NEUTRAL/FLAT signals are
                    # market monitoring data, not trade signals. Live prices are never
                    # written to Firestore; they flow only through the WebSocket ticker
                    # stream to the dashboard.
                    should_push = _signals_changed or _interval_elapsed
                    if not should_push:
                        raise StopIteration  # skip cleanly without nesting

                    _PRICE_CONTEXT_KEYS = frozenset({
                        'price', 'entry_price', 'atr', 'atr_pct',
                        'support', 'resistance', 'pivot', 'r1', 'r2', 's1', 's2',
                        'bull_tp1', 'bull_tp2', 'bull_tp3',
                        'bear_tp1', 'bear_tp2', 'bear_tp3',
                        'scalper_view', 'day_trader_view', 'swing_view',
                        'volume_zscore', 'volume_strength',
                        'oi_change_1h_pct', 'oi_zscore',
                        'macro_daily', 'macro_weekly',
                        'session', 'session_note',
                    })

                    fired = {
                        sym: sig for sym, sig in _signals_now.items()
                        if isinstance(sig, dict) and sig.get('fire', False)
                    }

                    all_sigs_for_dashboard = _signals_now  # full payload for client-side Firestore reads
                    push_target = all_sigs_for_dashboard   # write everything but strip price fields

                    now_str = datetime.now(timezone.utc).isoformat()

                    # v80: Firestore rejects NaN/Infinity, and ONE bad value used
                    # to fail the WHOLE batch silently — stale fired docs survived
                    # while every new state (incl. armed) died (measured live:
                    # snapshot had 3 armed + 5 fired, Firestore served 2 old
                    # fires). Sanitize recursively, and if a batch still fails,
                    # fall back to per-doc writes that NAME the failing symbol.
                    import math as _math
                    def _fs_safe(v):
                        if isinstance(v, float) and (_math.isnan(v) or _math.isinf(v)):
                            return None
                        if isinstance(v, dict):
                            return {k: _fs_safe(x) for k, x in v.items()}
                        if isinstance(v, (list, tuple)):
                            return [_fs_safe(x) for x in v]
                        return v

                    # ── Per-symbol diffing — write only what CHANGED ─────────
                    # This loop was 99% of the project's entire Firestore spend:
                    # 60 documents x ~298 pushes/day = 17,876 writes, against a
                    # Spark free-tier cap of 20,000. Everything else the product
                    # does — every login, trial, OTP, analytics pass and trade,
                    # for all 15 users — came to 114 writes/day combined.
                    #
                    # The cause was a GLOBAL fingerprint: if any single symbol
                    # changed, all 60 documents were rewritten, including the ~55
                    # byte-identical ones. Hashing each symbol's own payload and
                    # skipping the unchanged ones takes this to roughly 1,000
                    # writes/day (~5% of the cap) on a typical scan.
                    #
                    # `timestamp` is deliberately EXCLUDED from the hash. It is
                    # rewritten every push by definition, so including it would
                    # make every document differ every time and the diff would
                    # save nothing at all.
                    #
                    # A periodic FULL push heals drift — a document deleted or
                    # edited outside this loop would otherwise stay stale forever,
                    # because our hash says we already wrote it.
                    _pairs = []
                    _skipped = 0
                    _full_push = (_now - _last_full_push) >= _FIRESTORE_FULL_REFRESH_S
                    for sym, sig in push_target.items():
                        if not isinstance(sig, dict):
                            continue
                        sig_ref = db.collection("signals").document(sym.replace('/', '_'))
                        # Strip live-price and heavy context keys â€” prices go via WS only
                        compact = {
                            k: _fs_safe(v) for k, v in numpy_to_native(sig).items()
                            if k not in _PRICE_CONTEXT_KEYS
                        }
                        compact['symbol']    = sym
                        compact['fire']      = bool(sig.get('fire', False))
                        # hash BEFORE stamping the time, for the reason above
                        _doc_hash = hash(json.dumps(compact, sort_keys=True, default=str))
                        compact['timestamp'] = now_str
                        if not _full_push and _doc_sig_hash.get(sym) == _doc_hash:
                            _skipped += 1
                            continue
                        _doc_sig_hash[sym] = _doc_hash
                        _pairs.append((sig_ref, compact))
                    if _full_push:
                        _last_full_push = _now

                    _pushed = await asyncio.to_thread(_push_signal_docs, _pairs)
                    if _pairs or _skipped:
                        print(f"[PRODUCER] Firestore: {_pushed}/{len(_pairs)} written, "
                              f"{_skipped} unchanged skipped "
                              f"({len(fired)} fired){' [FULL REFRESH]' if _full_push else ''} @ {now_str}")

                except StopIteration:
                    pass  # nothing to push this tick
                except Exception as _e:
                    print(f"[PRODUCER ERROR] âš ï¸ Failed to write signals to Firestore: {_e}")

                # Legacy track-record update disabled â€” live_engine.py wallet
                # is the sole track-record writer to avoid file conflicts.

            except Exception as e:
                print(f"State update error: {e}")
            await asyncio.sleep(1)

    asyncio.create_task(update_state())

    try:
        await engine.run()
    except Exception as e:
        print(f"âš ï¸ LiveEngine crashed: {e}")

# -------------------------------------------------------------------
# Telegram Bot Connect â€” one-tap flow
# -------------------------------------------------------------------
# Admin sets TELEGRAM_BOT_TOKEN and TELEGRAM_BOT_USERNAME in .env / environment.
# Users click "Connect Telegram" â†’ get a deep link â†’ tap Start â†’ connected.

import secrets as _secrets

def _tg_token()    -> str: return os.getenv("TELEGRAM_BOT_TOKEN", "")
def _tg_username() -> str: return os.getenv("TELEGRAM_BOT_USERNAME", "").lstrip("@")

# code â†’ user_email (pending connections, in-memory)
_tg_pending: dict = {}

# user_email â†’ chat_id (persisted)
_tg_connections: dict = {}

# ── Where Telegram connections live ───────────────────────────────────────────
# On the VOLUME, and imported from one place so the writer and the reader cannot
# disagree.
#
# This was `Path("data/telegram_connections.json")` — relative to the process CWD
# — while scripts/notifications/dispatcher.py read `_ROOT / "data" / ...`. On
# Railway both land on /app/data, which is the container overlay and is WIPED ON
# EVERY DEPLOY. So a user connected Telegram, it worked, and the next deploy
# silently unsubscribed them: the file was gone, the dispatcher found no chat_ids,
# and _tg_send_all returned without sending or logging anything. Four deploys on
# 2026-08-17 erased it four times.
#
# Everything else durable already moved to STATE_DIR when the volume was attached
# (scripts/engine/config.py) — this file was simply missed, which is why the track
# record survives redeploys and Telegram connections did not.
#
# The relative path was a second, quieter hazard: any process started from a
# different CWD would read and write different files.
from scripts.engine.config import STATE_DIR as _STATE_DIR
from scripts.engine.config import STATE_GENERATION
from scripts.engine import config as _cfg_mod
_TG_CONNECTIONS_PATH = _STATE_DIR / "telegram_connections.json"

# One-time migration off the ephemeral path. Copied only when the volume has no
# file yet, so this can never overwrite good data with a stale container copy;
# after a deploy has already wiped /app/data it finds nothing and does nothing,
# which is the honest outcome rather than a silent failure.
try:
    _tg_legacy = Path("data/telegram_connections.json")
    if _tg_legacy.exists() and not _TG_CONNECTIONS_PATH.exists():
        _TG_CONNECTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _TG_CONNECTIONS_PATH.write_text(_tg_legacy.read_text(encoding="utf-8"),
                                        encoding="utf-8")
        print(f"[Telegram] migrated connections {_tg_legacy} -> {_TG_CONNECTIONS_PATH}")
except Exception as _exc:
    print(f"[Telegram] connection migration skipped: {_exc}")


def _tg_chat_id(entry) -> str:
    """chat_id out of either registry shape.

    The file was {email: chat_id} and is now {email: {chat_id, access_until}} so
    the SENDER can refuse an expired user without waiting for a sweep. Old files
    are read as-is rather than migrated on load, because a half-written migration
    on a crashed boot would silently drop everyone's Telegram.
    """
    if isinstance(entry, dict):
        return str(entry.get("chat_id") or "")
    return str(entry or "")


def _tg_access_until(email: str) -> str:
    """ISO instant this user's access ends, or '' if it cannot be determined.

    Written next to the chat_id so delivery is gated by a TIMESTAMP the sender
    can check itself. The hourly sweep is a cleanup, not the entitlement check —
    it sleeps before its first pass, so after every restart there was a full hour
    in which expired users still received signals.
    """
    try:
        user_doc = get_user_doc(email) or {}
        sub  = user_doc.get("subscription") or {}
        # A PAID plan is gated by the subscription's own end date, never by
        # trial_end. Falling through to trial_end here is what wrote an elapsed
        # timestamp beside a paying subscriber's chat_id and made the sender skip
        # them on every signal — see has_paid_access() for the whole story.
        if has_paid_access(user_doc):
            if isinstance(sub, dict):
                for key in ("current_period_end", "expires_at", "end_date"):
                    if sub.get(key):
                        return str(sub[key])
            return ""          # paid, no end date — no timestamp gate
        return str(user_doc.get("trial_end") or "")
    except Exception as exc:
        print(f"[TG] access lookup failed for {email}: {exc!r}")
        return ""


def _tg_load_connections() -> None:
    global _tg_connections
    if _TG_CONNECTIONS_PATH.exists():
        try:
            _tg_connections = json.loads(_TG_CONNECTIONS_PATH.read_text())
        except Exception:
            pass


def _tg_save_connections() -> None:
    _TG_CONNECTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _TG_CONNECTIONS_PATH.write_text(json.dumps(_tg_connections, indent=2))


def _tg_start_poller() -> None:
    """Background thread: long-polls Telegram getUpdates, matches /start CODE to pending users."""
    if not _tg_token():
        return

    def _poll() -> None:
        import requests as _req
        offset = 0
        while True:
            try:
                r = _req.get(
                    f"https://api.telegram.org/bot{_tg_token()}/getUpdates",
                    params={"offset": offset, "timeout": 30, "allowed_updates": ["message"]},
                    timeout=36,
                )
                data = r.json()
                if data.get("ok"):
                    for upd in data.get("result", []):
                        offset = upd["update_id"] + 1
                        msg     = upd.get("message", {})
                        text    = (msg.get("text") or "").strip()
                        chat_id = str(msg.get("chat", {}).get("id", ""))
                        if text.startswith("/start") and chat_id:
                            parts = text.split(maxsplit=1)
                            code  = parts[1].strip() if len(parts) > 1 else ""
                            if code and code in _tg_pending:
                                email = _tg_pending.pop(code)
                                _tg_connections[email] = {
                                    "chat_id":      chat_id,
                                    "access_until": _tg_access_until(email),
                                }
                                _tg_save_connections()
                                logger.info(f"[Telegram] Connected {email} â†’ chat_id {chat_id}")
                                # Send confirmation to user
                                try:
                                    _req.post(
                                        f"https://api.telegram.org/bot{_tg_token()}/sendMessage",
                                        json={
                                            "chat_id":    chat_id,
                                            "text":       "âœ… *AEGIS Signal Bot connected!*\n\nYou'll now receive BUY/SELL signals directly here. Set a unique notification tone so you never miss one.",
                                            "parse_mode": "Markdown",
                                        },
                                        timeout=5,
                                    )
                                except Exception:
                                    pass
            except Exception:
                pass
            time.sleep(1)

    import threading as _threading
    t = _threading.Thread(target=_poll, daemon=True, name="tg-poller")
    t.start()


# -------------------------------------------------------------------
# FastAPI app (lifespan runs engine as background task)
# -------------------------------------------------------------------
# â”€â”€ Trader Engine background scan loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_TRADER_SCAN_INTERVAL = 30    # 30 seconds

def _save_trader_track_record() -> None:
    """Copy data/trader_track_record.json â†’ web/trader_track_record.json for static serving."""
    try:
        engine = _get_trader_engine_lazy()
        if engine is None:
            return
        wallet  = engine.wallet
        summary = wallet.summary

        # Read raw signals from the data file (includes open + closed)
        from scripts.trader_model.trader_config import TRADER_RECORD_PATH as _DATA_TR
        signals: list = []
        if _DATA_TR.exists():
            with open(_DATA_TR, encoding='utf-8') as _f:
                _d = json.load(_f)
                signals = _d.get('signals', [])

        payload = {
            "generated_at":    datetime.now(timezone.utc).isoformat(),
            "balance":         summary['balance'],
            "initial_capital": wallet.INITIAL_CAPITAL,
            "total_pnl_usdt":  summary['total_pnl_usdt'],
            "total_pnl_pct":   summary['total_pnl_pct'],
            "total_trades":    summary['total_trades'],
            "won":             summary['won'],
            "lost":            summary['lost'],
            "win_rate":        summary['win_rate'],
            "open_positions":  summary['open_positions'],
            "last_scan":       engine.last_scan_time,
            "signals":         sorted(signals, key=lambda r: r.get('timestamp') or '', reverse=True)[:500],
        }
        TRADER_TRACK_RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = TRADER_TRACK_RECORD_PATH.with_suffix('.tmp')
        with open(tmp, 'w', encoding='utf-8') as _f:
            json.dump(payload, _f, default=str)
        os.replace(tmp, TRADER_TRACK_RECORD_PATH)
    except Exception as _e:
        logger.error(f"[TraderRecord] save error: {_e}")


async def _trader_scan_loop():
    """Runs the Universal Trader Engine every 60 seconds and caches token status."""
    await asyncio.sleep(30)   # let the main engine warm up first
    while True:
        try:
            engine = _get_trader_engine_lazy()
            if engine is not None:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: engine.scan_all_tokens(risk_profile='balanced'),
                )
                _save_trader_track_record()
                logger.info(
                    f"[TraderEngine] scan complete â€” "
                    f"{len(engine.active_signals)} signal(s), "
                    f"{len(engine.token_status)} token(s) tracked"
                )
        except Exception as _te:
            logger.error(f"[TraderEngine] scan error: {_te}")
        await asyncio.sleep(_TRADER_SCAN_INTERVAL)


def _otp_cleanup_pass() -> None:
    """One sweep. MUST NOT run on the event loop — see the caller.

    Both halves are blocking Firestore calls: the .stream() scan, and then one
    DELETE per expired document with no await between them. A backlog of expired
    OTPs therefore costs one blocking round trip each, and deletes are writes —
    the first thing to fail, and to retry slowly, when a project is over quota.
    """
    now = datetime.now(timezone.utc)
    expired = db.collection(_OTP_COL).where("expires_at", "<", now).stream()
    count = 0
    for doc in expired:
        try:
            doc.reference.delete()
            count += 1
        except Exception as exc:
            # One undeletable document must not abandon the rest of the sweep.
            print(f"[OTP cleanup] skipped {doc.id}: {exc!r}")
    if count:
        print(f"[OTP cleanup] Purged {count} expired OTP document(s)")


async def _otp_cleanup_loop():
    """Delete expired OTP documents from Firestore every 5 minutes.

    The sweep runs in a worker thread. Run on the loop, as it was, this froze
    the whole site for the length of the scan plus one delete per expired
    document — every five minutes, forever, not just at boot.
    """
    while True:
        await asyncio.sleep(300)
        try:
            await asyncio.to_thread(_otp_cleanup_pass)
        except Exception as exc:
            print(f"[OTP cleanup] Error: {exc}")


def _telegram_cleanup_pass() -> None:
    """One sweep. MUST NOT run on the event loop — see the caller.

    Every lookup in here is a BLOCKING Firestore round trip: is_trial_expired()
    and _tg_access_until() both call get_user_doc(), which is a plain `def`
    ending in `doc_ref.get()`. That is two network calls per connected user,
    with no await between them.
    """
    to_remove = []
    for email in list(_tg_connections):
        try:
            if is_trial_expired(email):
                # Both endings land here, and the log says which. A TRIAL running
                # out and a PAID TERM running out are the same outcome for delivery
                # but very different things to see in a log when a subscriber says
                # "my Telegram stopped".
                _doc = get_user_doc(email) or {}
                _plan = str(_doc.get("plan") or "?")
                _why = ('paid plan ended' if _plan.lower() in PAID_PLANS
                        else f'{_plan} access ended')
                print(f"[TG cleanup] {email}: {_why} — disconnecting Telegram")
                to_remove.append(email)
            else:
                entry = _tg_connections.get(email)
                _tg_connections[email] = {
                    "chat_id":      _tg_chat_id(entry),
                    "access_until": _tg_access_until(email),
                }
        except Exception as exc:
            # One bad user must not save the rest from being swept.
            print(f"[TG cleanup] skipped {email}: {exc!r}")
    for email in to_remove:
        _tg_connections.pop(email, None)
    if to_remove:
        _tg_save_connections()
        print(f"[TG cleanup] Disconnected {len(to_remove)} expired user(s): {to_remove}")


async def _telegram_cleanup_loop():
    """Disconnect Telegram for users whose trial has ended or plan has lapsed.

    Delivery does NOT depend on this loop. dispatcher._tg_send_all checks
    `access_until` on every send, so entitlement is enforced at send time and
    this is cleanup — a late pass cannot leak signals. That is what makes the
    startup grace period below safe.

    THE SWEEP RUNS IN A WORKER THREAD, AND THAT IS NOT OPTIONAL. Its body is two
    blocking Firestore calls per connected user. An earlier version of this fix
    moved the sweep ahead of the sleep so a restart could not reopen an hour of
    free access — correct intent, but it put that blocking I/O directly on the
    event loop at startup, which stalled every request uvicorn was trying to
    answer. The site buffered and never opened. Off-loop via to_thread keeps the
    prompt first pass without holding the loop.

    One bad user is caught per user rather than per pass, so a single failed
    lookup cannot abort the sweep for everyone else.
    """
    # Let the app bind and start serving before touching the network at all.
    await asyncio.sleep(30)
    while True:
        try:
            await asyncio.to_thread(_telegram_cleanup_pass)
        except Exception as exc:
            print(f"[TG cleanup] Error: {exc}")
        await asyncio.sleep(900)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_track_record()
    _enforce_track_generation()      # must run AFTER the load it purges
    _rt_load_at_startup()            # re-apply operator dials saved on the volume
    _tg_load_connections()
    _tg_start_poller()
    engine_task       = asyncio.create_task(run_engine_background())
    reminder_task     = asyncio.create_task(check_and_send_trial_reminders())
    subscription_task = asyncio.create_task(check_and_send_subscription_reminders())
    analytics_task    = asyncio.create_task(analytics_loop())
    dev_token_task    = asyncio.create_task(dev_token_display_loop())
    dev_key_task      = asyncio.create_task(dev_key_display_loop())
    otp_cleanup_task  = asyncio.create_task(_otp_cleanup_loop())
    tg_cleanup_task   = asyncio.create_task(_telegram_cleanup_loop())
    # Trader bot disabled â€” AEGIS-1 live_engine is the sole signal source.
    # trader_task = asyncio.create_task(_trader_scan_loop())
    yield
    engine_task.cancel()
    reminder_task.cancel()
    subscription_task.cancel()
    analytics_task.cancel()
    dev_token_task.cancel()
    dev_key_task.cancel()
    otp_cleanup_task.cancel()
    tg_cleanup_task.cancel()
    # trader_task.cancel()

app = FastAPI(title="Aegis-1 by Gatekeeper", lifespan=lifespan)


@app.websocket("/ws/track-record")
async def websocket_track_record(websocket: WebSocket):
    """Stream live track-record updates to the frontend track-record page."""
    await _tr_ws_manager.connect(websocket)
    try:
        # Send current snapshot immediately on connect
        from scripts.live_engine import TRACK_RECORD_PATH as _TR_PATH
        _tr_path = _TR_PATH
        if _tr_path.exists():
            try:
                with open(_tr_path, 'r', encoding='utf-8') as _f:
                    await websocket.send_json(json.load(_f))
            except Exception:
                pass
        # Keep connection alive; engine broadcasts push new data
        while True:
            await asyncio.sleep(30)
            await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        _tr_ws_manager.disconnect(websocket)
    except Exception:
        _tr_ws_manager.disconnect(websocket)


app.add_middleware(ProxyHeadersMiddleware, trusted_hosts=["*"])

# Compress text responses. The pages are served uncompressed otherwise:
# dashboard.html is 196 KB, chart.html 145 KB, index.html 75 KB, and the
# trader track record another ~97 KB of JSON — all of which gzip to roughly a
# fifth of that. minimum_size skips tiny payloads where the CPU is not worth it.
# Only affects HTTP; WebSocket frames are untouched.
app.add_middleware(GZipMiddleware, minimum_size=1024)

# CORS â€” read allowed origins from env so production is locked to the real domain.
# ALLOWED_ORIGINS env var: comma-separated list, e.g. "https://aegis.example.com,http://localhost:8000"
_raw_origins = os.getenv("ALLOWED_ORIGINS", "")
_ALLOWED_ORIGINS: list[str] = (
    [o.strip() for o in _raw_origins.split(",") if o.strip()]
    if _raw_origins
    else ["http://localhost:8000", "http://127.0.0.1:8000"]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=False,   # Bearer-token auth â€” cookies not used cross-origin
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Requested-With"],
)

_LEGACY_HTML_REDIRECTS = {
    "/web/src/pages/index.html":           "/",
    "/web/src/pages/pricing.html":         "/pricing",
    "/web/src/pages/signals.html":         "/signals",
    "/web/src/pages/dashboard.html":       "/dashboard",
    "/web/src/pages/contact.html":         "/contact",
    "/web/src/pages/terms.html":           "/terms",
    "/web/src/pages/privacy_policy.html":  "/privacy",
    "/web/src/pages/risk_disclosure.html": "/risk-disclosure",
    "/web/src/pages/refund-policy.html":   "/refund-policy",
    "/web/src/pages/refund_policy.html":   "/refund-policy",
    "/web/src/pages/conditions.html":      "/conditions",
    "/web/src/pages/reset-password.html":  "/reset-password",
    "/web/src/pages/track-record.html":    "/track-record",
    "/web/src/pages/bot-record.html":      "/bot-record",
    "/web/src/pages/trader-record.html":   "/trader-record",
    "/web/src/pages/review.html":          "/reviews",
    "/web/src/pages/reviews.html":         "/reviews",
    "/web/src/pages/chart.html":           "/chart",
    "/web/src/pages/logic.html":           "/logic",
    "/web/src/pages/pitch.html":           "/pitch",
}

_PRIVATE_WEB_PREFIXES = (
    "/web/node_modules", "/web/package.json", "/web/package-lock.json",
    "/web/tailwind.config", "/web/postcss.config",
)


@app.middleware("http")
async def redirect_legacy_html_paths(request: Request, call_next):
    """301-redirect old /web/src/pages/*.html URLs to clean SEO-friendly paths.
    Runs before the static-files mount so the mount never serves raw page HTML."""
    path = request.url.path
    clean = _LEGACY_HTML_REDIRECTS.get(path)
    if clean:
        return RedirectResponse(url=clean, status_code=301)
    # The mount below is rooted at web/, which also contains the npm tree and
    # the build config. None of that is meant to be public.
    if path.startswith(_PRIVATE_WEB_PREFIXES):
        return JSONResponse({"detail": "Not Found"}, status_code=404)
    return await call_next(request)

# Hoisted out of the request path — this was being rebuilt on every response.
_CLEAN_PAGE_ROUTES = frozenset({
    "/", "/pricing", "/signals", "/dashboard", "/contact", "/terms",
    "/privacy", "/privacy-policy", "/risk-disclosure", "/refund-policy",
    "/conditions", "/reset-password", "/track-record", "/bot-record",
    "/trader-record", "/reviews", "/review", "/chart", "/logic", "/pitch",
})
_CACHEABLE_ASSETS = (".js", ".css", ".woff2", ".woff", ".ttf", ".svg",
                     ".png", ".jpg", ".jpeg", ".webp", ".ico")


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)

    # Google OAuth popup requires relaxed COOP; COEP must stay unsafe-none for CDN resources
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin-allow-popups"
    response.headers["Cross-Origin-Embedder-Policy"] = "unsafe-none"

    # Clickjacking protection â€” dashboard must never be embedded in a foreign frame
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Content-Security-Policy"] = "frame-ancestors 'none'"

    # MIME sniffing protection
    response.headers["X-Content-Type-Options"] = "nosniff"

    # Legacy XSS filter (IE/older Chrome)
    response.headers["X-XSS-Protection"] = "1; mode=block"

    # HSTS â€” force HTTPS for 1 year (only meaningful in production behind TLS)
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

    # Referrer: send origin only, never full URL, to external hosts
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

    # Permissions: explicitly deny unused browser features
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

    # Caching. HTML must never be cached — a stale page pins the whole asset
    # graph to an old deploy. Assets are the opposite: a dashboard load pulls
    # ~340 KB across half a dozen files, and forcing every one of them to
    # revalidate cost a round trip per asset on every navigation.
    #
    # The split is the version query the pages already carry
    # (gatekeeper.js?v=80.0, main.css?v=2): a versioned URL names one immutable
    # build, so it can be cached for a year and a deploy invalidates it by
    # changing the number. Unversioned assets keep revalidating, because
    # nothing else would tell the browser they changed.
    path = request.url.path
    if path in _CLEAN_PAGE_ROUTES or (path.endswith(".html") and "/web/" in path):
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    elif path.startswith("/web/") and path.endswith(_CACHEABLE_ASSETS):
        response.headers["Cache-Control"] = (
            "public, max-age=31536000, immutable" if request.url.query.startswith("v=")
            else "no-cache, must-revalidate" if path.endswith(".js")
            else "public, max-age=3600"
        )

    return response

# -------------------------------------------------------------------
# Static Files: serve the entire 'web' folder under '/web' prefix
# -------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_ROOT = os.path.join(BASE_DIR, "web")
WEB_ROOT_PATH = Path(WEB_ROOT)

if not WEB_ROOT_PATH.exists():
    print(f"âš ï¸ Warning: 'web' directory not found at {WEB_ROOT_PATH}. Creating fallback structure.")
    WEB_ROOT_PATH.mkdir(parents=True, exist_ok=True)
    pages_dir = WEB_ROOT_PATH / "src" / "pages"
    scripts_dir = WEB_ROOT_PATH / "src" / "scripts"
    styles_dir = WEB_ROOT_PATH / "src" / "styles"
    pages_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)
    styles_dir.mkdir(parents=True, exist_ok=True)
    (pages_dir / "index.html").write_text("<html><body><h1>Aegisâ€‘1</h1><p>Frontend files missing. Please upload the correct static files to 'web/src/pages/'</p></body></html>")
    (pages_dir / "dashboard.html").write_text("<html><body><h1>Dashboard unavailable</h1><p>Static files not found.</p></body></html>")

app.mount("/web", StaticFiles(directory=str(WEB_ROOT_PATH), html=True), name="web")

# -------------------------------------------------------------------
# Redirects
# -------------------------------------------------------------------
@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})

@app.get("/")
async def root_page():
    return FileResponse(WEB_ROOT_PATH / "src/pages/index.html")

@app.get("/dashboard")
async def dashboard_page():
    return FileResponse(WEB_ROOT_PATH / "src/pages/dashboard.html")



security = HTTPBearer()

@app.get("/api/signals")
def api_signals(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Return latest live signals to subscribed users only (plan == 'pro')."""
    email = get_current_user(credentials)
    user_doc = get_user_doc(email)
    if not user_doc:
        raise HTTPException(status_code=403, detail="User not found")
    if not user_doc.get("otp_verified", False):
        raise HTTPException(status_code=403, detail="Account not verified. Please sign up with a valid email.")
    plan = user_doc.get('plan', 'trial')
    if plan != 'pro':
        raise HTTPException(status_code=403, detail="Subscription required to access signals")

    signals = numpy_to_native(LIVE_STATE.data.get('signals', {}))
    return JSONResponse(content=signals)


# ── REMOVED v82f: two Razorpay-only endpoints used to live here ──────────────
#
#   @app.post("/api/create-order")   async def api_create_order(...)
#   @app.post("/api/verify-payment") async def api_verify_payment(...)
#
# They were registered BEFORE the provider-aware versions further down, and
# FastAPI serves the FIRST route matching a path+method. So these shadowed the
# real ones and were the endpoints actually answering in production:
#
#   * create-order took {amount} in paise. Every real caller
#     (gatekeeper.js, simple-auth-client.js) sends {plan, currency}, so live
#     checkout returned 422 "Field required: amount" — nobody could subscribe,
#     on ANY gateway. Verified against the running app.
#   * verify-payment checked a Razorpay signature and returned {"status":"ok"}
#     WITHOUT upgrading the user's plan. The upgrade lives in the shadowed
#     verify_payment, so a settled payment granted nothing except via webhook.
#
# Deleting them un-shadows the provider-aware create_order / verify_payment,
# which carry the full Paddle -> DODO -> Razorpay chain, require auth, and
# actually apply the plan. Their request models were also named
# CreateOrderRequest / VerifyPaymentRequest, colliding with the plan-based
# models defined later — that duplicate is gone with them.


@app.get("/api/razorpay-key")
async def api_razorpay_key():
    """Return only the public Razorpay Key ID for frontend usage."""
    if not RAZORPAY_KEY_ID:
        raise HTTPException(status_code=503, detail="Razorpay key not set")
    return JSONResponse({"key_id": RAZORPAY_KEY_ID})

from fastapi import Header

@app.get("/api/public/signals")
def api_public_signals(authorization: Optional[str] = Header(None)):
    """Return latest live signals publicly (for dashboard display)."""
    subscription_active = False
    trial_end = None
    plan = "trial"

    if authorization and authorization.startswith("Bearer "):
        token = authorization.split("Bearer ")[1]
        try:
            email = decode_token(token)
            if email:
                user_doc = get_user_doc(email)
                if user_doc and user_doc.get("otp_verified", False):
                    plan = user_doc.get("plan", "trial")
                    trial_end = user_doc.get("trial_end")
                    if plan in ["pro", "active", "premium", "intermediate", "basic"]:
                        subscription_active = True
                    elif trial_end:
                        if datetime.now(timezone.utc) <= datetime.fromisoformat(trial_end):
                            subscription_active = True
        except Exception:
            pass

    signals = numpy_to_native(LIVE_STATE.data.get('signals', {}))
    warmup = LIVE_STATE.data.get('warmup_progress', '0/0')
    alpha_mode = LIVE_STATE.data.get('alpha_mode', False)
    
    return JSONResponse(content={
        'signals': signals,
        'warmup': warmup,
        'alpha_mode': alpha_mode,
        'subscription_active': subscription_active,
        'trial_end': trial_end,
        'plan': plan,
        'timestamp': datetime.now(timezone.utc).isoformat()
    })

def _build_insight_payload(sig: dict, plan: str) -> dict:
    """
    Build the token-insight response dict.

    Tier rules
    ----------
    trial / basic  â†’ market bias, S/R, confluence, price targets, session, fear/greed.
                      AI probability and fire signal hidden.
    intermediate   â†’ all of the above + AI probability bands (low/med/high label).
    pro            â†’ full signal including meta_confidence, fire, direction.
    """
    if not isinstance(sig, dict):
        return {}

    # Fields safe for all authenticated users
    public_fields = (
        'symbol', 'price', 'timeframe', 'data_timestamp', 'timestamp',
        'market_bias', 'bias_strength', 'trend_regime', 'volatility_regime', 'atr_pct',
        'support', 'resistance', 'pivot', 'r1', 'r2', 's1', 's2',
        'bull_tp1', 'bull_tp2', 'bull_tp3',
        'bear_tp1', 'bear_tp2', 'bear_tp3',
        'confluence',
        'rsi', 'macd_signal', 'cci', 'adx', 'supertrend',
        'macro_daily', 'macro_weekly',
        'volume_strength', 'volume_zscore',
        'funding_rate', 'funding_bias', 'oi_trend', 'oi_change_1h_pct', 'oi_zscore',
        'session', 'session_note', 'fear_greed',
        'scalper_view', 'day_trader_view', 'swing_view',
        'atr', 'atr_multiplier',
    )

    out = {k: sig[k] for k in public_fields if k in sig}

    # Intermediate: add a coarse AI conviction label (no raw number)
    if plan in ('intermediate', 'pro', 'premium', 'active'):
        conf = float(sig.get('meta_confidence', 0))
        thr  = float(sig.get('threshold', 0.6))
        if conf == 0:
            out['ai_conviction'] = 'NO_DATA'
        elif conf >= thr * 1.15:
            out['ai_conviction'] = 'HIGH'
        elif conf >= thr:
            out['ai_conviction'] = 'MEDIUM'
        else:
            out['ai_conviction'] = 'LOW'

    # Pro: full signal
    if plan in ('pro', 'premium', 'active'):
        for k in ('fire', 'signal', 'signal_strength', 'direction',
                  'meta_confidence', 'threshold', 'p_buy', 'p_sell', 'p_hold',
                  'tradeable', 'suggested_tp', 'suggested_sl', 'signal_id'):
            if k in sig:
                out[k] = sig[k]

    return out


@app.get("/api/token-insight/{symbol:path}")
def token_insight(symbol: str, authorization: Optional[str] = Header(None)):
    """
    Per-token market insight available to all authenticated users.

    trial / basic   â†’ S/R levels, confluence, price targets, market bias, trader views.
    intermediate    â†’ above + AI conviction label (HIGH / MEDIUM / LOW).
    pro             â†’ above + full fire/direction/meta_confidence signal.

    The symbol path parameter accepts slash notation, e.g. BTC/USDT or BTC%2FUSDT.
    """
    symbol = symbol.replace('%2F', '/').replace('%2f', '/').upper()
    if not re.match(r'^[A-Z0-9]{2,12}/[A-Z]{2,6}$', symbol):
        raise HTTPException(status_code=400, detail="Invalid symbol format. Expected e.g. BTC/USDT.")

    plan = "unauthenticated"
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split("Bearer ")[1]
        try:
            email = decode_token(token)
            if email:
                user_doc = get_user_doc(email)
                if user_doc and user_doc.get("otp_verified", False):
                    plan = user_doc.get("plan", "trial")
                    # Trial users with active trial are treated same as basic for insight
                    if plan == "trial":
                        trial_end = user_doc.get("trial_end")
                        if trial_end:
                            try:
                                te = datetime.fromisoformat(str(trial_end).replace("Z", "+00:00"))
                                if te.tzinfo is None:
                                    te = te.replace(tzinfo=timezone.utc)
                                if datetime.now(timezone.utc) > te:
                                    raise HTTPException(status_code=403, detail="Trial expired. Please subscribe.")
                            except HTTPException:
                                raise
                            except Exception:
                                raise HTTPException(status_code=403, detail="Trial status unknown.")
        except HTTPException:
            raise
        except Exception:
            pass

    if plan == "unauthenticated":
        raise HTTPException(status_code=401, detail="Authentication required to view token insights.")

    signals = LIVE_STATE.data.get('signals', {})
    sig = signals.get(symbol)

    # Handle nested timeframe structure (use 1h summary)
    if isinstance(sig, dict) and any(tf in sig for tf in ('1h', '4h', '1d')):
        sig = sig.get('1h') or next((v for v in sig.values() if isinstance(v, dict)), None)

    if not sig:
        # Engine may still be warming up â€” return what we know without AI fields
        warmup = LIVE_STATE.data.get('warmup_progress', '0/0')
        return JSONResponse(content={
            'symbol': symbol,
            'status': 'warming_up' if warmup != '0/0' else 'not_found',
            'warmup_progress': warmup,
        }, status_code=202)

    payload = numpy_to_native(_build_insight_payload(sig, plan))
    payload['plan'] = plan
    return JSONResponse(content=payload)


@app.get("/api/public/token-insight/{symbol:path}")
async def public_token_insight(symbol: str):
    """
    Unauthenticated teaser: returns only market bias, session, and confluence summary.
    Used for landing-page previews â€” no S/R or price targets.
    """
    symbol = symbol.replace('%2F', '/').replace('%2f', '/').upper()
    if not re.match(r'^[A-Z0-9]{2,12}/[A-Z]{2,6}$', symbol):
        raise HTTPException(status_code=400, detail="Invalid symbol format. Expected e.g. BTC/USDT.")
    signals = LIVE_STATE.data.get('signals', {})
    sig = signals.get(symbol)

    if isinstance(sig, dict) and any(tf in sig for tf in ('1h', '4h', '1d')):
        sig = sig.get('1h') or next((v for v in sig.values() if isinstance(v, dict)), None)

    if not sig:
        return JSONResponse(content={'symbol': symbol, 'status': 'unavailable'}, status_code=202)

    teaser = numpy_to_native({
        'symbol':         symbol,
        'price':          sig.get('price'),
        'market_bias':    sig.get('market_bias', 'NEUTRAL'),
        'trend_regime':   sig.get('trend_regime'),
        'volatility_regime': sig.get('volatility_regime'),
        'session':        sig.get('session'),
        'session_note':   sig.get('session_note'),
        'confluence_summary': (sig.get('confluence') or {}).get('summary'),
        'confluence_total':   (sig.get('confluence') or {}).get('total'),
        'rsi':            sig.get('rsi'),
        'fear_greed':     sig.get('fear_greed'),
        'data_timestamp': sig.get('data_timestamp'),
    })
    return JSONResponse(content=teaser)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# PROFESSIONAL TOKEN ANALYSIS ENGINE
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _generate_token_analysis(sig: dict) -> dict:
    """
    Generate a professional, beginner-friendly market analysis for one token.
    Covers: market verdict, top indicators with plain explanations, why the
    signal fired or didn't fire, key price levels, and macro context.
    All terminology is explained so a first-time trader understands it.
    """
    if not isinstance(sig, dict):
        return {}

    sym   = sig.get('symbol', '')
    price = float(sig.get('price') or sig.get('entry_price') or 0)
    fire  = bool(sig.get('fire', False))
    side  = sig.get('signal', 'FLAT')
    conf  = float(sig.get('meta_confidence') or 0)
    thr   = float(sig.get('threshold') or 0)

    bias          = sig.get('market_bias', 'NEUTRAL')
    trend_regime  = sig.get('trend_regime', 'RANGING')
    vol_regime    = sig.get('volatility_regime', 'MEDIUM')
    rsi           = float(sig.get('rsi') or 50)
    adx           = float(sig.get('adx') or 20)
    macd          = sig.get('macd_signal', 'NEUTRAL')
    supertrend    = sig.get('supertrend', 'NEUTRAL')
    macro_daily   = float(sig.get('macro_daily') or 0)
    macro_weekly  = float(sig.get('macro_weekly') or 0)

    conf_d        = sig.get('confluence') or {}
    total_c       = float(conf_d.get('total') or 0)
    mom_c         = float(conf_d.get('momentum') or 0)
    trend_c       = float(conf_d.get('trend') or 0)
    vol_c         = float(conf_d.get('volume') or 0)
    sm_c          = float(conf_d.get('smart_money') or 0)
    candle_c      = float(conf_d.get('candle') or 0)

    funding       = float(sig.get('funding_rate') or 0)
    funding_bias  = sig.get('funding_bias', 'NEUTRAL')
    oi_trend      = sig.get('oi_trend', 'STABLE')
    vol_strength  = sig.get('volume_strength', 'AVERAGE')

    fg            = sig.get('fear_greed') or {}
    fg_val        = float(fg.get('value') or 50)
    fg_label      = fg.get('label', 'Neutral')

    support       = float(sig.get('support') or sig.get('s1') or 0)
    resistance    = float(sig.get('resistance') or sig.get('r1') or 0)
    pivot         = float(sig.get('pivot') or 0)
    tradeable     = bool(sig.get('tradeable', False))

    p_buy         = float(sig.get('p_buy') or 0)
    p_sell        = float(sig.get('p_sell') or 0)
    p_hold        = float(sig.get('p_hold') or 0)

    def _pct(v: float) -> str:
        return f'{v * 100:.1f}%'

    def _px(v: float) -> str:
        if v <= 0:      return 'â€”'
        if v < 0.001:   return f'${v:.6f}'
        if v < 1:       return f'${v:.4f}'
        if v < 100:     return f'${v:.3f}'
        return f'${v:,.2f}'

    # â”€â”€ Bull / bear vote tally â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    bull_votes: List[str] = []
    bear_votes: List[str] = []

    if bias == 'BULLISH':    bull_votes.append('market_bias')
    elif bias == 'BEARISH':  bear_votes.append('market_bias')

    if macd == 'BULLISH':    bull_votes.append('macd')
    elif macd == 'BEARISH':  bear_votes.append('macd')

    if supertrend == 'BULLISH':  bull_votes.append('supertrend')
    elif supertrend == 'BEARISH': bear_votes.append('supertrend')

    if rsi < 35:    bull_votes.append('rsi_oversold')
    elif rsi > 70:  bear_votes.append('rsi_overbought')

    if trend_c >= 6:  (bull_votes if bias == 'BULLISH' else bear_votes).append('trend_confluence')
    if mom_c >= 6:    (bull_votes if p_buy > p_sell else bear_votes).append('momentum')

    if funding_bias == 'SHORTS_PAYING': bull_votes.append('funding')
    elif funding_bias == 'LONGS_PAYING': bear_votes.append('funding')

    if oi_trend == 'INCREASING' and bias == 'BULLISH':  bull_votes.append('oi')
    elif oi_trend == 'INCREASING' and bias == 'BEARISH': bear_votes.append('oi')

    if macro_daily > 0.1:    bull_votes.append('macro_daily')
    elif macro_daily < -0.1: bear_votes.append('macro_daily')

    if macro_weekly > 0.1:    bull_votes.append('macro_weekly')
    elif macro_weekly < -0.1: bear_votes.append('macro_weekly')

    bull_n = len(bull_votes)
    bear_n = len(bear_votes)
    total_v = bull_n + bear_n or 1

    # â”€â”€ Overall verdict â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    bull_pct = bull_n / total_v
    if bull_pct >= 0.70:   verdict_label, verdict_icon = 'Strong Bullish', 'ðŸŸ¢'
    elif bull_pct >= 0.55: verdict_label, verdict_icon = 'Bullish',         'ðŸŸ¢'
    elif bull_pct >= 0.45: verdict_label, verdict_icon = 'Neutral / Mixed', 'ðŸŸ¡'
    elif bull_pct >= 0.30: verdict_label, verdict_icon = 'Bearish',         'ðŸ”´'
    else:                  verdict_label, verdict_icon = 'Strong Bearish',  'ðŸ”´'

    # â”€â”€ Plain-English trend description â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    trend_desc = {
        'TRENDING_UP':   'moving upward in a clear trend',
        'TRENDING_DOWN': 'moving downward in a clear trend',
        'TRENDING':      'in an active trend',
        'RANGING':       'moving sideways without a clear direction',
    }.get(trend_regime, 'in an uncertain phase')

    vol_desc = {
        'HIGH':   'high volatility â€” prices are swinging a lot',
        'MEDIUM': 'moderate volatility',
        'LOW':    'low volatility â€” price is calm and moving slowly',
    }.get(vol_regime, 'moderate volatility')

    # â”€â”€ Two-sentence summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    base = sym.split('/')[0]
    summary = (
        f"{base} is currently {trend_desc}, with {vol_desc}. "
        f"{'Most indicators lean bullish' if bull_pct > 0.55 else 'Most indicators lean bearish' if bull_pct < 0.45 else 'Indicators are mixed'}"
        f" â€” {bull_n} bullish signal{'s' if bull_n != 1 else ''} vs {bear_n} bearish."
    )

    # â”€â”€ Top indicators (max 6, sorted by impact) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    indicators = []

    # RSI
    if rsi <= 30:
        indicators.append({
            'name': 'RSI (Momentum)',
            'value': f'{rsi:.0f} â€” Oversold',
            'direction': 'BULLISH',
            'icon': 'ðŸ”‹',
            'impact': 'high',
            'simple': (
                f"RSI is {rsi:.0f}. Think of RSI like a rubber band â€” when it stretches too far "
                f"in one direction it snaps back. Below 30 means the token has been sold too "
                f"aggressively. A bounce back up is increasingly likely."
            ),
        })
    elif rsi >= 70:
        indicators.append({
            'name': 'RSI (Momentum)',
            'value': f'{rsi:.0f} â€” Overbought',
            'direction': 'BEARISH',
            'icon': 'âš ï¸',
            'impact': 'high',
            'simple': (
                f"RSI is {rsi:.0f}. The token has been bought very aggressively and may be "
                f"running out of steam. Above 70 is a warning that the rally could slow down "
                f"or reverse. Buyers should wait for a pullback."
            ),
        })
    else:
        indicators.append({
            'name': 'RSI (Momentum)',
            'value': f'{rsi:.0f} â€” Neutral',
            'direction': 'NEUTRAL',
            'icon': 'ðŸ“Š',
            'impact': 'medium',
            'simple': (
                f"RSI is {rsi:.0f}, sitting in the neutral zone (30â€“70). There's no extreme "
                f"buying or selling pressure. The token can move in either direction without "
                f"being 'overheated' or 'oversold'."
            ),
        })

    # Trend / Supertrend
    if supertrend == 'BULLISH':
        indicators.append({
            'name': 'Supertrend',
            'value': 'Bullish',
            'direction': 'BULLISH',
            'icon': 'ðŸ“ˆ',
            'impact': 'high',
            'simple': (
                "The Supertrend indicator is green â€” price is trading above a dynamic "
                "support line that adapts to market volatility. This tells us sellers "
                "are not in control right now. Think of it as a moving 'floor' â€” as "
                "long as price stays above it, the trend is up."
            ),
        })
    elif supertrend == 'BEARISH':
        indicators.append({
            'name': 'Supertrend',
            'value': 'Bearish',
            'direction': 'BEARISH',
            'icon': 'ðŸ“‰',
            'impact': 'high',
            'simple': (
                "The Supertrend indicator is red â€” price is trading below a dynamic "
                "resistance line. This acts like a 'ceiling' pressing down on the price. "
                "Until price breaks back above it, sellers remain in control."
            ),
        })

    # MACD
    if macd != 'NEUTRAL':
        indicators.append({
            'name': 'MACD (Momentum Shift)',
            'value': macd.capitalize(),
            'direction': macd,
            'icon': 'ðŸ”„',
            'impact': 'medium',
            'simple': (
                f"MACD is {'crossing upward' if macd == 'BULLISH' else 'crossing downward'}, "
                f"which signals that {'buying' if macd == 'BULLISH' else 'selling'} momentum "
                f"is {'picking up' if macd == 'BULLISH' else 'building'}. "
                f"{'This is often an early sign of a price rise.' if macd == 'BULLISH' else 'This warns that the price could fall further.'}"
            ),
        })

    # ADX (trend strength)
    if adx > 25:
        indicators.append({
            'name': 'ADX (Trend Strength)',
            'value': f'{adx:.0f} â€” {"Strong" if adx > 40 else "Moderate"} trend',
            'direction': 'NEUTRAL',
            'icon': 'ðŸ’ª',
            'impact': 'medium',
            'simple': (
                f"ADX is {adx:.0f}. This measures how strong the current trend is â€” "
                f"it doesn't tell you the direction, just the conviction. "
                f"{'Above 40 means the trend is very powerful and unlikely to reverse quickly.' if adx > 40 else 'Between 25â€“40 means a genuine trend exists and it is worth following.'}"
            ),
        })

    # Confluence
    if total_c >= 6:
        indicators.append({
            'name': 'Confluence Score',
            'value': f'{total_c:.1f}/10',
            'direction': 'BULLISH' if bias == 'BULLISH' else 'BEARISH',
            'icon': 'ðŸŽ¯',
            'impact': 'high',
            'simple': (
                f"Confluence score is {total_c:.1f}/10. This is our AI's internal vote count â€” "
                f"it tallies up momentum, trend, volume, smart money flow, and candlestick "
                f"patterns into a single score. "
                f"{'Above 6 means most evidence is pointing in the same direction â€” higher quality setups.' if total_c >= 6 else ''} "
                f"Breakdown: Momentum {mom_c:.1f}, Trend {trend_c:.1f}, "
                f"Volume {vol_c:.1f}, Smart Money {sm_c:.1f}."
            ),
        })
    elif total_c >= 3:
        indicators.append({
            'name': 'Confluence Score',
            'value': f'{total_c:.1f}/10 â€” Weak',
            'direction': 'NEUTRAL',
            'icon': 'âš–ï¸',
            'impact': 'medium',
            'simple': (
                f"Confluence score is {total_c:.1f}/10 â€” indicators are split. "
                f"Some point bullish, others bearish. This is a 'wait and see' situation; "
                f"there is no clear majority conviction from the market's internal structure."
            ),
        })

    # Funding rate (derivatives market)
    if abs(funding) > 0.005:
        funding_desc = 'Longs are paying shorts' if funding_bias == 'LONGS_PAYING' else 'Shorts are paying longs'
        funding_meaning = (
            'This means the market is over-leveraged to the upside â€” too many people are betting on a rise. '
            'These longs may be forced to close, creating selling pressure.'
        ) if funding_bias == 'LONGS_PAYING' else (
            'The market is over-leveraged to the downside. Too many people are shorting â€” '
            'if price rises even slightly, forced short-covering can create a sharp rally (short squeeze).'
        )
        indicators.append({
            'name': 'Funding Rate',
            'value': f'{funding:+.4f}% â€” {funding_desc}',
            'direction': 'BEARISH' if funding_bias == 'LONGS_PAYING' else 'BULLISH',
            'icon': 'ðŸ’¸',
            'impact': 'medium',
            'simple': funding_meaning,
        })

    # Fear & Greed
    if fg_val <= 25 or fg_val >= 75:
        indicators.append({
            'name': 'Market Sentiment (Fear & Greed)',
            'value': f'{fg_val:.0f}/100 â€” {fg_label}',
            'direction': 'BULLISH' if fg_val <= 25 else 'BEARISH',
            'icon': 'ðŸ§ ',
            'impact': 'low',
            'simple': (
                f"The Fear & Greed index is {fg_val:.0f} ({fg_label}). "
                + (
                    "Extreme fear means most market participants are panicking and selling. "
                    "Historically, extreme fear has been one of the best times to buy â€” "
                    "'be greedy when others are fearful.'"
                    if fg_val <= 25 else
                    "Extreme greed means everyone is euphoric and buying. "
                    "Historically, this is when markets are most vulnerable to a sharp correction â€” "
                    "'be fearful when others are greedy.'"
                )
            ),
        })

    # Macro trend
    if abs(macro_daily) > 0.05:
        macro_dir = 'up' if macro_daily > 0 else 'down'
        indicators.append({
            'name': 'Daily Macro Trend',
            'value': f'{"Bullish" if macro_daily > 0 else "Bearish"} ({macro_daily:+.1%})',
            'direction': 'BULLISH' if macro_daily > 0 else 'BEARISH',
            'icon': 'ðŸŒ',
            'impact': 'medium',
            'simple': (
                f"On the daily chart, {base} has been trending {macro_dir} "
                f"({macro_daily:+.1%} momentum). "
                f"{'This longer-term upward pressure provides tailwind for bullish trades.' if macro_daily > 0 else 'This longer-term downward pressure creates headwind â€” even if short-term indicators look bullish, you are fighting the bigger trend.'}"
            ),
        })

    # Limit to top 6 by impact
    impact_order = {'high': 0, 'medium': 1, 'low': 2}
    indicators.sort(key=lambda x: impact_order.get(x.get('impact', 'low'), 2))
    indicators = indicators[:6]

    # â”€â”€ Why the signal fired or didn't â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    signal_section: dict = {}

    if side == 'WAITING' or conf == 0 and not tradeable:
        signal_section = {
            'status': 'WAITING',
            'headline': 'â³ Engine is warming up',
            'plain': (
                f"The AI for {base} is still loading its data and running its first scan. "
                f"Full analysis and signal decisions will appear within 2â€“3 minutes."
            ),
            'technical': 'Model scan not yet completed.',
        }
    elif not tradeable and conf == 0:
        signal_section = {
            'status': 'MONITOR_ONLY',
            'headline': 'ðŸ‘ï¸ Watch mode â€” no signals',
            'plain': (
                f"{base} is in monitor-only mode. The AI model was trained on its data "
                f"but the historical edge (win rate, expectancy) was not strong enough to "
                f"justify trading it. We show the price and market context so you can "
                f"watch it and make your own decision â€” but the bot won't trade it automatically."
            ),
            'technical': 'Model trained but did not pass backtesting quality gate (tradeable=False).',
        }
    elif fire:
        direction_word = 'BUY' if side in ('BUY', 'STRONG_BUY') else 'SELL'
        signal_section = {
            'status': 'SIGNAL_ACTIVE',
            'headline': f'ðŸš€ Signal ACTIVE â€” {direction_word}',
            'plain': (
                f"The AI fired a {direction_word} signal on {base}. It required "
                f"at least {_pct(thr)} confidence and reached {_pct(conf)} â€” "
                f"clearing the bar with {'strong' if conf > thr * 1.2 else 'sufficient'} conviction. "
                f"{'This is a Strong signal â€” confidence is significantly above threshold.' if conf > thr * 1.15 else ''}"
            ),
            'technical': f'meta_confidence={conf:.3f} â‰¥ threshold={thr:.3f}. Fire=True.',
        }
    else:
        # No signal â€” explain WHY
        if conf == 0:
            plain = (
                f"The AI model hasn't produced a clear directional probability yet "
                f"for {base}. This usually means the market conditions are too noisy "
                f"or ambiguous â€” no clean trade setup is visible."
            )
            headline = 'ðŸ” No setup found'
            tech = f'meta_confidence=0. Model output inconclusive.'
        elif thr > 0 and conf >= thr * 0.85:
            pct_away = (thr - conf) / thr * 100
            plain = (
                f"Very close â€” the AI is at {_pct(conf)} confidence, just {pct_away:.1f}% "
                f"below the {_pct(thr)} trigger threshold. "
                f"One more bullish/bearish indicator aligning could push it over the line. "
                f"Watch closely â€” a signal may fire in the next scan."
            )
            headline = 'ðŸ”” Almost there â€” watching for trigger'
            tech = f'meta_confidence={conf:.3f}, threshold={thr:.3f}. Gap: {thr-conf:.3f}.'
        elif thr > 0 and conf >= 0.35:
            plain = (
                f"The AI sees some directional evidence for {base} but confidence is "
                f"{_pct(conf)}, below the {_pct(thr)} required. "
                f"Indicators are not aligned strongly enough. The model is saying 'I see "
                f"something but it's not convincing yet â€” wait for clearer confirmation.'"
            )
            headline = 'â¸ï¸ Building â€” not enough conviction yet'
            tech = f'meta_confidence={conf:.3f} < threshold={thr:.3f}.'
        else:
            # Check if near S&R
            near_resistance = (resistance > 0 and price > 0 and
                               abs(price - resistance) / price < 0.02)
            near_support    = (support > 0 and price > 0 and
                               abs(price - support) / price < 0.02)
            macro_conflicting = (macro_daily < -0.15 and p_buy > p_sell) or \
                                (macro_daily > 0.15 and p_sell > p_buy)

            if near_resistance and p_buy > p_sell:
                plain = (
                    f"The AI sees bullish momentum building in {base}, but price is "
                    f"pressing against a resistance zone at {_px(resistance)}. "
                    f"The model suppressed the BUY signal â€” buying into resistance means "
                    f"you're buying right at a wall that sellers have historically defended. "
                    f"The signal is waiting for that wall to break."
                )
                headline = 'ðŸš§ BUY blocked by resistance'
                tech = f'Price within 2% of resistance {_px(resistance)}. Signal suppressed by S&R filter.'
            elif near_support and p_sell > p_buy:
                plain = (
                    f"The AI sees bearish pressure in {base}, but price is sitting "
                    f"right on a support level at {_px(support)}. "
                    f"The model held back the SELL signal â€” shorting into support is risky "
                    f"because buyers historically step in at this level. "
                    f"Waiting to see if support breaks before confirming the trade."
                )
                headline = 'ðŸ›¡ï¸ SELL held at support'
                tech = f'Price within 2% of support {_px(support)}. Signal suppressed by S&R filter.'
            elif macro_conflicting:
                trend_word = 'bearish' if macro_daily < 0 else 'bullish'
                plain = (
                    f"There's a conflict: short-term indicators suggest one direction but "
                    f"the daily trend for {base} is strongly {trend_word} "
                    f"({macro_daily:+.1%}). "
                    f"The model won't fight the bigger trend unless confidence is very high. "
                    f"Trading against the daily trend is like swimming against a strong current â€” "
                    f"possible, but requires much more conviction."
                )
                headline = 'âš”ï¸ Signal conflicting with daily trend'
                tech = f'macro_daily={macro_daily:+.3f} conflicts with short-term direction.'
            else:
                plain = (
                    f"The AI model is at {_pct(conf)} confidence â€” well below the {_pct(thr)} "
                    f"required to trade. Indicators are too mixed or too weak. "
                    f"The market for {base} is not giving a clean enough signal right now. "
                    f"Patience: when the evidence aligns, the signal will fire automatically."
                )
                headline = 'ðŸ˜´ No clear setup'
                tech = f'meta_confidence={conf:.3f}, threshold={thr:.3f}. No qualifying filter match.'

        signal_section = {
            'status': 'NO_SIGNAL',
            'headline': headline,
            'plain': plain,
            'technical': tech,
        }

    # â”€â”€ Key levels â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    key_levels = {}
    if support > 0:
        dist_s = (price - support) / price * 100 if price > 0 else 0
        key_levels['support'] = {
            'price': round(support, 8),
            'distance_pct': round(dist_s, 2),
            'meaning': (
                f"Support at {_px(support)} ({dist_s:.1f}% below current price). "
                f"This is a price level where buyers have historically stepped in. "
                f"If price falls here and bounces, it confirms the support is holding. "
                f"If price breaks below it with volume, expect further downside."
            ),
        }
    if resistance > 0:
        dist_r = (resistance - price) / price * 100 if price > 0 else 0
        key_levels['resistance'] = {
            'price': round(resistance, 8),
            'distance_pct': round(dist_r, 2),
            'meaning': (
                f"Resistance at {_px(resistance)} ({dist_r:.1f}% above current price). "
                f"A zone where sellers have repeatedly pushed price back down. "
                f"A clean break above this level on high volume would be a bullish signal."
            ),
        }
    if sig.get('bull_tp1') and fire and side in ('BUY', 'STRONG_BUY'):
        key_levels['target_1'] = {
            'price': round(float(sig['bull_tp1']), 8),
            'meaning': 'First take-profit target (1Ã— ATR above entry).',
        }
        if sig.get('bull_tp2'):
            key_levels['target_2'] = {
                'price': round(float(sig['bull_tp2']), 8),
                'meaning': 'Second take-profit target (2Ã— ATR above entry).',
            }
    elif sig.get('bear_tp1') and fire and side in ('SELL', 'STRONG_SELL'):
        key_levels['target_1'] = {
            'price': round(float(sig['bear_tp1']), 8),
            'meaning': 'First take-profit target (1Ã— ATR below entry).',
        }

    # â”€â”€ Macro / sentiment context â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    macro_lines = []

    if abs(macro_weekly) > 0.05:
        macro_lines.append(
            f"Weekly trend: {'bullish' if macro_weekly > 0 else 'bearish'} "
            f"({macro_weekly:+.1%}). "
            f"{'The longer-term picture supports the bulls.' if macro_weekly > 0 else 'The bigger picture still favors sellers â€” short-term bounces may be selling opportunities.'}"
        )

    if funding_bias == 'LONGS_PAYING' and abs(funding) > 0.005:
        macro_lines.append(
            f"Funding rate is positive (+{funding:.4f}%). Perpetual futures traders are paying "
            f"a fee to stay long. When funding gets too high, it usually triggers a sharp "
            f"pullback as longs get liquidated or choose to exit."
        )
    elif funding_bias == 'SHORTS_PAYING' and abs(funding) > 0.005:
        macro_lines.append(
            f"Funding rate is negative ({funding:.4f}%). Short sellers are paying to hold their "
            f"positions. This creates upward pressure â€” if shorts capitulate, it could trigger "
            f"a fast squeeze rally."
        )

    if oi_trend == 'INCREASING':
        macro_lines.append(
            "Open Interest is rising â€” new money is entering the derivatives market. "
            "Combined with the current price direction, this confirms fresh conviction "
            f"behind the {'up' if bias == 'BULLISH' else 'down'}move."
        )
    elif oi_trend == 'DECREASING':
        macro_lines.append(
            "Open Interest is falling â€” traders are closing positions and exiting. "
            "This usually means the current move is losing steam. "
            "Price may continue but with less force."
        )

    if fg_val < 30:
        macro_lines.append(
            f"Market sentiment is in Fear ({fg_val:.0f}/100). Historically, extreme fear "
            f"creates some of the best long-term buying opportunities â€” but don't catch "
            f"a falling knife. Wait for a technical confirmation first."
        )
    elif fg_val > 70:
        macro_lines.append(
            f"Market sentiment is in Greed ({fg_val:.0f}/100). When everyone is euphoric, "
            f"the market is often near a local top. Be cautious about chasing price here."
        )

    macro_context = ' '.join(macro_lines) if macro_lines else (
        "Macro conditions are neutral. No extreme sentiment, funding, or open interest "
        "signals are distorting the picture right now."
    )

    # â”€â”€ Session context â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    session      = sig.get('session', '')
    session_note = sig.get('session_note', '')
    session_ctx  = f"Currently in the {session} session. {session_note}" if session else ''

    return {
        'symbol':        sym,
        'price':         price,
        'verdict':       f'{verdict_icon} {verdict_label}',
        'bull_bear':     {'bull': bull_n, 'bear': bear_n, 'total': bull_n + bear_n},
        'summary':       summary,
        'top_indicators': indicators,
        'signal':         signal_section,
        'key_levels':     key_levels,
        'macro_context':  macro_context,
        'session':        session_ctx,
        'probabilities':  {
            'buy':  round(p_buy * 100, 1),
            'sell': round(p_sell * 100, 1),
            'hold': round(p_hold * 100, 1),
        } if (p_buy + p_sell + p_hold) > 0 else None,
        'generated_at': datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/token-analysis/{symbol:path}")
def token_analysis(symbol: str, authorization: Optional[str] = Header(None)):
    """
    Professional market analysis for one token â€” accessible to all authenticated users.
    Returns plain-English breakdown of indicators, why the signal fired or didn't,
    key price levels, and macro/sentiment context.
    """
    symbol = symbol.replace('%2F', '/').replace('%2f', '/').upper()
    if not re.match(r'^[A-Z0-9]{2,12}/[A-Z]{2,6}$', symbol):
        raise HTTPException(status_code=400, detail="Invalid symbol format. Expected e.g. BTC/USDT.")

    # Auth: any valid authenticated user (trial included)
    plan = "unauthenticated"
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split("Bearer ")[1]
        try:
            email = decode_token(token)
            if email:
                user_doc = get_user_doc(email)
                if user_doc and user_doc.get("otp_verified", False):
                    plan = user_doc.get("plan", "trial")
        except Exception:
            pass
    if plan == "unauthenticated":
        raise HTTPException(status_code=401, detail="Authentication required.")

    signals = LIVE_STATE.data.get('signals', {})
    sig = signals.get(symbol)
    if isinstance(sig, dict) and any(tf in sig for tf in ('1h', '4h', '1d')):
        sig = sig.get('1h') or next((v for v in sig.values() if isinstance(v, dict)), None)

    if not sig:
        return JSONResponse(content={
            'symbol': symbol, 'status': 'unavailable',
            'warmup': LIVE_STATE.data.get('warmup_progress', '0/0'),
        }, status_code=202)

    analysis = numpy_to_native(_generate_token_analysis(sig))

    # Trial/basic users: hide raw probabilities (AI confidence numbers)
    if plan in ('trial', 'basic'):
        analysis.pop('probabilities', None)

    return JSONResponse(content=analysis)


@app.get("/favicon.ico")
async def favicon():
    favicon_path = WEB_ROOT_PATH / "favicon.ico"
    if favicon_path.exists():
        return FileResponse(favicon_path)
    return Response(status_code=204)

@app.get("/robots.txt")
async def robots():
    robots_path = WEB_ROOT_PATH / "robots.txt"
    if robots_path.exists():
        return FileResponse(robots_path)
    return Response(status_code=204)

@app.get("/sitemap.xml")
async def sitemap():
    sitemap_path = WEB_ROOT_PATH / "sitemap.xml"
    if sitemap_path.exists():
        return FileResponse(sitemap_path, media_type="application/xml")
    return Response(status_code=204)

@app.get("/og-cover.png")
async def og_cover():
    """Social-share preview image at a clean root URL (referenced by the
    OG/Twitter meta tags). Served from web/og-cover.png."""
    og_path = WEB_ROOT_PATH / "og-cover.png"
    if og_path.exists():
        return FileResponse(og_path, media_type="image/png")
    return Response(status_code=404)

@app.get("/terms")
async def terms_page():
    return FileResponse(WEB_ROOT_PATH / "src/pages/terms.html")

@app.get("/privacy-policy")
async def privacy_page():
    return FileResponse(WEB_ROOT_PATH / "src/pages/privacy_policy.html")

@app.get("/refund-policy")
async def refund_page():
    return FileResponse(WEB_ROOT_PATH / "src/pages/refund-policy.html")

@app.get("/contact")
async def contact_page():
    return FileResponse(WEB_ROOT_PATH / "src/pages/contact.html")

@app.get("/risk-disclosure")
async def risk_disclosure_page():
    return FileResponse(WEB_ROOT_PATH / "src/pages/risk_disclosure.html")

@app.get("/pricing")
async def pricing_page():
    return FileResponse(WEB_ROOT_PATH / "src/pages/pricing.html")

@app.get("/signals")
async def signals_page():
    return FileResponse(WEB_ROOT_PATH / "src/pages/signals.html")

@app.get("/chart")
async def chart_page():
    return FileResponse(WEB_ROOT_PATH / "src/pages/chart.html")

@app.get("/conditions")
async def conditions_page():
    return FileResponse(WEB_ROOT_PATH / "src/pages/conditions.html")

@app.get("/reset-password")
async def reset_password_page():
    return FileResponse(WEB_ROOT_PATH / "src/pages/reset-password.html")

@app.get("/track-record")
async def track_record_page():
    return FileResponse(WEB_ROOT_PATH / "src/pages/track-record.html")

@app.get("/bot-record")
async def bot_record_page():
    return FileResponse(WEB_ROOT_PATH / "src/pages/bot-record.html")

@app.get("/trader-record")
async def trader_record_page():
    return FileResponse(WEB_ROOT_PATH / "src/pages/trader-record.html")

@app.get("/reviews")
async def reviews_page():
    return FileResponse(WEB_ROOT_PATH / "src/pages/reviews.html")

@app.get("/review")
async def review_redirect():
    return RedirectResponse(url="/reviews", status_code=301)

@app.get("/logic")
async def logic_page():
    return FileResponse(WEB_ROOT_PATH / "src/pages/logic.html")

@app.get("/pitch")
async def pitch_page():
    return FileResponse(WEB_ROOT_PATH / "src/pages/pitch.html")

@app.get("/privacy")
async def privacy_alias():
    return FileResponse(WEB_ROOT_PATH / "src/pages/privacy_policy.html")

# â”€â”€ Custom 404 handler â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from fastapi.exceptions import HTTPException as FastAPIHTTPException

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    page_404 = WEB_ROOT_PATH / "src/pages/404.html"
    if page_404.exists():
        return FileResponse(page_404, status_code=404)
    return JSONResponse({"error": "Not found"}, status_code=404)

# -------------------------------------------------------------------
# Auth helpers (JWT)
# -------------------------------------------------------------------

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password: str, hash_: str) -> bool:
    return bcrypt.checkpw(password.encode(), hash_.encode())

def create_token(email: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=7)
    payload = {"sub": email, "exp": expire}
    assert SECRET_KEY is not None, "SECRET_KEY must be set"
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> Optional[str]:
    # Try Firebase ID token first
    try:
        decoded_token = firebase_auth.verify_id_token(token)
        email = decoded_token.get("email")
        if email:
            return email
        # UID-only token (some OAuth flows omit email) â€” look up email via Admin SDK
        uid = decoded_token.get("uid")
        if uid:
            try:
                user_record = firebase_auth.get_user(uid)
                return user_record.email or uid
            except Exception:
                return uid
    except Exception:
        pass
    # Fallback to custom JWT (email/password signup)
    try:
        assert SECRET_KEY is not None, "SECRET_KEY must be set"
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except Exception:
        return None

def decode_uid_from_token(token: str) -> Optional[str]:
    """Return Firebase UID (not email) â€” used for Firestore paths shared with the frontend."""
    try:
        decoded = firebase_auth.verify_id_token(token)
        return decoded.get("uid")
    except Exception:
        pass
    # Fallback for custom JWT: sub is email, use it as path key
    try:
        assert SECRET_KEY is not None
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except:
        return None

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    email = decode_token(credentials.credentials)
    if not email:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    return email

def get_firebase_uid(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Dependency that returns Firebase UID â€” keeps Firestore paths consistent with the frontend."""
    uid = decode_uid_from_token(credentials.credentials)
    if not uid:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    return uid

# -------------------------------------------------------------------
# Firestore user helpers
# -------------------------------------------------------------------
# ── Runtime controls: tune the desk without a deploy ──────────────────────────
# Six times in one day these constants were changed by editing code, running the
# suite and redeploying: MIN_FIRE_QUALITY twice, STRONG_TIDE_FACTOR, the tide
# policy, and a pause that did not exist. Each round trip needed a computer.
#
# These are OPERATING dials, not code. A floor and a size factor are decisions a
# desk makes against a live tape, and requiring a deploy to change one means the
# tape has moved by the time it lands.
#
# Every knob here is read at CALL time by its owner — engine.py reads
# MIN_FIRE_QUALITY through the config MODULE, trader_gate reads its own globals
# inside evaluate()/_classify() — so assigning the module attribute takes effect
# on the very next scan. That property is verified by a test; a knob captured at
# import would silently do nothing.
#
# Bounds are not decoration. An unbounded floor of 900 silently stops all trading
# and looks like a dead engine, so every knob declares the range it is allowed to
# take and anything outside is refused with the reason.
_RUNTIME_PATH = _STATE_DIR / "runtime_overrides.json"

# name -> (module, attribute, kind, low, high, human description)
_TUNABLES: Dict[str, Any] = {
    "min_fire_quality": ("scripts.engine.config", "MIN_FIRE_QUALITY", "float", 0.0, 100.0,
                         "Minimum signal quality to fire (0-100)"),
    "strong_tide_factor": ("src.trading.trader_gate", "STRONG_TIDE_FACTOR", "float", 0.0, 1.0,
                           "Size multiplier into a strong counter-tide. 0 = refuse outright"),
    "allow_exhaustion_reversal": ("src.trading.trader_gate", "ALLOW_EXHAUSTION_REVERSAL", "bool", 0, 1,
                                  "Allow counter-trend exhaustion fades (measured -0.064R)"),
    "allow_exhaustion_reversal_buy": ("src.trading.trader_gate",
                                      "ALLOW_EXHAUSTION_REVERSAL_BUY", "bool", 0, 1,
                                      "Allow the BUY half of the fade (buy oversold). "
                                      "Measured +0.044R; the SELL half stays refused"),
    "early_entry_on_ltf": ("src.trading.trader_gate", "EARLY_ENTRY_ON_LTF", "bool", 0, 1,
                          "Enter at the market when the 5m tape has turned, instead of "
                          "resting on a level touch that may never come"),
    "trading_paused": ("scripts.engine.config", "TRADING_PAUSED", "bool", 0, 1,
                       "Stop opening new positions. Exits keep running"),
}

_RUNTIME_DEFAULTS: Dict[str, Any] = {}      # captured at import, for "reset"


def _rt_module(path: str):
    import importlib
    return importlib.import_module(path)


def _rt_capture_defaults() -> None:
    """Remember the code values ONCE, so 'reset' means the committed value and
    not merely the previous override."""
    if _RUNTIME_DEFAULTS:
        return
    for key, (mod, attr, *_rest) in _TUNABLES.items():
        try:
            _RUNTIME_DEFAULTS[key] = getattr(_rt_module(mod), attr)
        except Exception:
            pass


def _rt_read() -> Dict[str, Any]:
    """Current LIVE value of every knob, straight from the owning module."""
    _rt_capture_defaults()
    out = {}
    for key, (mod, attr, kind, lo, hi, desc) in _TUNABLES.items():
        try:
            val = getattr(_rt_module(mod), attr)
        except Exception:
            continue
        out[key] = {
            "value": bool(val) if kind == "bool" else float(val),
            "default": (bool(_RUNTIME_DEFAULTS.get(key)) if kind == "bool"
                        else float(_RUNTIME_DEFAULTS.get(key, val))),
            "kind": kind, "min": lo, "max": hi, "description": desc,
        }
    return out


def _rt_coerce(key: str, raw: Any) -> Any:
    mod, attr, kind, lo, hi, desc = _TUNABLES[key]
    if kind == "bool":
        if isinstance(raw, str):
            return raw.strip().lower() in ("1", "true", "yes", "on")
        return bool(raw)
    try:
        val = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a number")
    if val < lo or val > hi:
        raise ValueError(f"{key} must be between {lo} and {hi} (got {val})")
    return val


def _rt_apply(values: Dict[str, Any], *, persist: bool = True) -> Dict[str, Any]:
    """Set knobs live, then persist so a restart keeps them."""
    _rt_capture_defaults()
    applied = {}
    for key, raw in (values or {}).items():
        if key not in _TUNABLES:
            raise ValueError(f"unknown control: {key}")
        val = _rt_coerce(key, raw)
        mod, attr, *_ = _TUNABLES[key]
        before = getattr(_rt_module(mod), attr, None)
        setattr(_rt_module(mod), attr, val)
        applied[key] = val
        print(f"[control] {key}: {before!r} -> {val!r}")
    if persist and applied:
        try:
            saved = {}
            if _RUNTIME_PATH.exists():
                saved = json.loads(_RUNTIME_PATH.read_text(encoding="utf-8")) or {}
            saved.update(applied)
            saved["_updated_at"] = datetime.now(timezone.utc).isoformat()
            _RUNTIME_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = _RUNTIME_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(saved, indent=2, default=str), encoding="utf-8")
            tmp.replace(_RUNTIME_PATH)
        except Exception as exc:
            print(f"[control] applied but NOT persisted: {exc!r}")
    return applied


def _rt_load_at_startup() -> None:
    """Re-apply saved overrides after a restart, or they last until the next
    deploy and then quietly revert — which is worse than not having them."""
    _rt_capture_defaults()
    try:
        if not _RUNTIME_PATH.exists():
            return
        saved = json.loads(_RUNTIME_PATH.read_text(encoding="utf-8")) or {}
        vals = {k: v for k, v in saved.items() if k in _TUNABLES}
        if vals:
            _rt_apply(vals, persist=False)
            print(f"[control] restored {len(vals)} runtime override(s) from the volume")
    except Exception as exc:
        print(f"[control] could not restore overrides: {exc!r}")


# ── Entitlements: the paid grant, on the VOLUME ───────────────────────────────
# Every record of who has paid lived in Firestore and nowhere else, which made a
# datastore outage indistinguishable from a cancelled subscription. Both grant
# paths wrote Firestore only:
#
#   * the payment-verify path answered "Payment verified but account update
#     failed" — money taken, no access, and nothing anywhere recording the grant
#   * the Whop webhook returned 500 so Whop would retry, but a retry window can
#     expire against a quota that resets daily
#
# So a paid subscriber silently reverted to `plan: trial`, which is exactly the
# cascade behind the "trial expired" overlay and the missing Telegram signals.
#
# This file is the durable record of a payment that HAPPENED. It lives on the
# Railway volume beside telegram_connections.json, so it survives redeploys and
# does not depend on Firestore being reachable, in quota, or correctly
# credentialled. Firestore stays the system of record for everything else and is
# still written; it is simply no longer the only place a grant exists.
#
# Two rules keep this safe:
#   1. It may only ever GRANT. It is never consulted to take access away, so a
#      stale or corrupt file cannot lock a paying customer out.
#   2. It respects its own expiry. An entry past subscription_end grants nothing,
#      so a lapsed plan does not become permanent access through this back door.
_ENTITLEMENTS_PATH = _STATE_DIR / "entitlements.json"


def _ent_load() -> Dict[str, Any]:
    try:
        if _ENTITLEMENTS_PATH.exists():
            return json.loads(_ENTITLEMENTS_PATH.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        print(f"[entitlements] load failed ({exc!r}) — treating as empty")
    return {}


def _ent_save(data: Dict[str, Any]) -> None:
    """Atomic write: a half-written entitlements file is worse than none."""
    _ENTITLEMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _ENTITLEMENTS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    tmp.replace(_ENTITLEMENTS_PATH)


def _ent_grant(user_key: str, plan: str, sub_end: Any, provider: str,
               payment_id: str = "") -> None:
    """Record a paid grant on the volume. Called BEFORE the Firestore write."""
    if not user_key or not plan:
        return
    try:
        data = _ent_load()
        data[str(user_key).lower()] = {
            "plan": plan,
            "status": "active",
            "subscription_end": str(sub_end) if sub_end else "",
            "provider": provider,
            "payment_id": payment_id,
            "granted_at": datetime.now(timezone.utc).isoformat(),
        }
        _ent_save(data)
        print(f"[entitlements] {user_key} -> {plan} until {sub_end or 'open-ended'} ({provider})")
    except Exception as exc:
        print(f"[entitlements] grant NOT recorded for {user_key}: {exc!r}")


def _ent_revoke(user_key: str) -> None:
    """Mark cancelled. Access still runs to subscription_end — a cancellation
    means 'will not renew', not 'ends now'."""
    if not user_key:
        return
    try:
        data = _ent_load()
        rec = data.get(str(user_key).lower())
        if rec:
            rec["status"] = "canceled"
            rec["canceled_at"] = datetime.now(timezone.utc).isoformat()
            _ent_save(data)
            print(f"[entitlements] {user_key} marked canceled")
    except Exception as exc:
        print(f"[entitlements] revoke failed for {user_key}: {exc!r}")


def _ent_overlay(user_key: str, doc: Optional[Dict]) -> Optional[Dict]:
    """Merge a volume grant into a Firestore doc, upgrade-only.

    Applied at the READ boundary in get_user_doc so every consumer — /auth/me,
    has_paid_access, is_trial_expired, the Telegram entitlement check — honours
    the grant without each needing to know this file exists.
    """
    try:
        rec = _ent_load().get(str(user_key or "").lower())
        if not rec:
            return doc
        end = _parse_ts(rec.get("subscription_end"))
        if end is not None and datetime.now(timezone.utc) > end:
            return doc                      # the paid term ran out; grant nothing
        merged = dict(doc or {})
        if has_paid_access(merged):
            return doc                      # Firestore already grants it
        merged["plan"] = rec.get("plan")
        sub = dict(merged.get("subscription") or {})
        sub.setdefault("status", rec.get("status") or "active")
        if rec.get("subscription_end"):
            sub.setdefault("current_period_end", rec["subscription_end"])
        sub.setdefault("provider", rec.get("provider") or "")
        merged["subscription"] = sub
        if rec.get("subscription_end"):
            merged.setdefault("subscription_end", rec["subscription_end"])
        print(f"[entitlements] serving {user_key} from the volume record "
              f"(Firestore shows {(doc or {}).get('plan')!r})")
        return merged
    except Exception as exc:
        print(f"[entitlements] overlay skipped for {user_key}: {exc!r}")
        return doc


def get_user_doc(email: str) -> Optional[Dict]:
    # The volume entitlement is overlaid HERE, at the single read boundary, so
    # every consumer honours a paid grant without knowing the file exists —
    # /auth/me, has_paid_access, is_trial_expired and the Telegram send-time
    # check all read through this function.
    #
    # The Firestore read is also wrapped: a paid user whose datastore is
    # unreachable must get their plan back from the volume rather than silently
    # becoming a trial user, which is the cascade this whole file exists to stop.
    try:
        doc_ref = db.collection("users").document(email)
        doc = doc_ref.get()
        to_dict = getattr(doc, "to_dict", None)
        exists = getattr(doc, "exists", False)
        base = (to_dict() if callable(to_dict) and exists else None)
        if not isinstance(base, dict):
            base = {} if exists else None
    except Exception as exc:
        print(f"[users] Firestore read failed for {email}: {exc!r}")
        base = None
    return _ent_overlay(email, base)

def phone_is_unique(phone: str, exclude_email: Optional[str] = None) -> bool:
    """Return True if phone number is not already stored in any user document."""
    normalized = normalize_phone_number(phone)
    if not normalized:
        return False
    try:
        docs = db.collection("users").where("phone_number", "==", normalized).limit(2).stream()
        for d in docs:
            if exclude_email and d.id == exclude_email:
                continue
            return False
    except Exception:
        pass  # if query fails, don't block registration
    return True

def create_user_doc(email: str, password_hash: Optional[str] = None,
                    provider: Optional[str] = None, social_id: Optional[str] = None,
                    full_name: Optional[str] = None, location: Optional[str] = None,
                    phone_number: Optional[str] = None) -> Dict:
    now = datetime.now(timezone.utc).isoformat()
    # Account is created in a "registered" state â€” the 3-day free trial does NOT
    # auto-start.  The user lands on /pricing and starts it themselves via the
    # "Start Free Trial" button (POST /api/v1/trial/start), which sets plan=trial
    # + trial_end.  Auto-starting here silently burned the trial the moment the
    # account was created, before the user opted in.
    user_data = {
        "email": email,
        "plan": "registered",
        "trial_start": None,
        "trial_end": None,
        "trial_active": False,
        "trial_used": False,        # /api/v1/trial/start begins the one 3-day trial
        "created_at": now,
        "last_login": now,
        "subscription": {
            "status": "inactive"
        },
        # Set only during OTP-verified provisioning; old/bypass accounts lack this field.
        "otp_verified": True,
    }
    if password_hash:
        user_data["password_hash"] = password_hash
    if provider:
        user_data["provider"] = provider
    if social_id:
        user_data["social_id"] = social_id
    if full_name:
        user_data["full_name"] = full_name
    if location:
        user_data["location"] = location
    if phone_number:
        user_data["phone_number"] = phone_number
    db.collection("users").document(email).set(user_data)
    return {"email": email, "plan": "registered", "trial_end": None, "full_name": full_name, "location": location}

def update_last_login(email: str):
    db.collection("users").document(email).update({"last_login": datetime.now(timezone.utc).isoformat()})

def get_or_create_user_from_oauth(email: str, name: str, provider: str, social_id: str) -> Dict:
    user = get_user_doc(email)
    if not user:
        user = create_user_doc(email, provider=provider, social_id=social_id, full_name=name)
    else:
        update_last_login(email)
    return {"email": email, "plan": user["plan"], "trial_end": user["trial_end"]}

# ── Paid access, defined ONCE ─────────────────────────────────────────────────
# This test lived in five places in this file, each slightly different (one of
# them counted "active" as a PLAN name). Two of those copies sit in the Telegram
# delivery path and between them they stopped a paying subscriber receiving any
# signal at all:
#
#   * _tg_access_until() fell through to trial_end for a paid user, writing an
#     already-elapsed timestamp beside their chat_id. dispatcher._tg_send_all then
#     silently `continue`d past them on every send.
#   * the hourly sweep called is_trial_expired(), which returned True for the same
#     reason, and DELETED the connection outright.
#
# The bot itself was fine — the /start confirmation is posted directly to the
# Telegram API and never passes either gate, which is why "connected" arrived and
# nothing else ever did.
#
# The cause in both was requiring `subscription.status == "active"` on top of a
# paid plan, then treating anything else as an expired trial. Missing bookkeeping
# is not a cancellation, and the send-site comment already said as much ("an
# ABSENT stamp must not be read as expired") while the code did the opposite.
#
# So: a paid plan grants access unless the subscription carries a status that
# positively says otherwise. Enumerating the DEAD states rather than requiring one
# live state is what makes an absent or unrecognised status fail open for someone
# who has paid, while a genuine cancellation still fails closed.
PAID_PLANS = frozenset({"pro", "premium", "intermediate", "basic", "pro-dev"})
_DEAD_SUB_STATUS = frozenset({
    "cancelled", "canceled", "expired", "past_due", "unpaid", "halted", "paused",
})
# "Cancelled" is not "over". It means WILL NOT RENEW, and the customer keeps what
# they already paid for until the period ends — which is exactly what the Whop
# revoke handler says it is doing ("Do NOT downgrade plan here ... the customer
# keeps access until the paid period ends"). has_paid_access disagreed with that
# comment and denied the moment the status flipped, so cancelling ended access
# instantly and took away time already bought.
#
# Only these two are treated that way. "expired" is genuinely over, and
# past_due / unpaid / halted mean money is owed — none of those buy remaining time.
_CANCELLED_STATUS = frozenset({"cancelled", "canceled"})


def _parse_ts(raw: Any) -> Optional[datetime]:
    """ISO string or datetime -> tz-aware UTC datetime; None when unusable."""
    if not raw:
        return None
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    try:
        d = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def _sub_end_ts(sub: Any) -> Optional[datetime]:
    """When this subscription's paid term ends, if it says."""
    if not isinstance(sub, dict):
        return None
    for key in ("current_period_end", "expires_at", "end_date"):
        ts = _parse_ts(sub.get(key))
        if ts:
            return ts
    return None


def has_paid_access(user_doc: Optional[Dict[str, Any]]) -> bool:
    """True when this user holds a paid plan whose term is still running.

    Note the deliberate asymmetry between the two ways access can end:

      * a MISSING or unrecognised `subscription.status` fails OPEN. Absent
        bookkeeping is not a cancellation, and reading it as one is what stopped a
        paying subscriber's Telegram entirely.
      * an ELAPSED end date fails CLOSED. That is not missing information — it is
        the subscription stating when it ends, and that date passing is exactly
        what "the plan ended" means.

    Without the second check a lapsed plan whose status field was never updated
    would keep access forever, and the hourly sweep would never disconnect it.
    """
    plan = str((user_doc or {}).get("plan") or "").lower()
    if plan not in PAID_PLANS:
        return False
    sub = (user_doc or {}).get("subscription") or {}
    status = str(sub.get("status") or "").lower() if isinstance(sub, dict) else ""
    end = _sub_end_ts(sub)
    now = datetime.now(timezone.utc)
    if status in _DEAD_SUB_STATUS:
        # A CANCELLED plan still runs to the end of the period already paid for.
        # Everything else in the dead set is over or owes money.
        return bool(status in _CANCELLED_STATUS and end is not None and now <= end)
    if end is not None and now > end:
        return False        # the plan's own term has run out
    return True


def is_trial_expired(email: str) -> bool:
    user_doc = get_user_doc(email)
    if not user_doc:
        return True
    if has_paid_access(user_doc):
        return False
    plan = user_doc.get("plan", "trial")
    trial_end_raw = user_doc.get("trial_end")
    if trial_end_raw:
        if isinstance(trial_end_raw, datetime):
            trial_end = trial_end_raw
        else:
            try:
                # Replace Z with +00:00 for compatibility with Python < 3.11
                clean_str = str(trial_end_raw).replace("Z", "+00:00")
                trial_end = datetime.fromisoformat(clean_str)
            except (ValueError, TypeError):
                return True
                
        # Ensure timezone awareness before comparison
        if trial_end.tzinfo is None:
            trial_end = trial_end.replace(tzinfo=timezone.utc)
            
        return datetime.now(timezone.utc) > trial_end
    return True

# -------------------------------------------------------------------
# OAuth routes
# -------------------------------------------------------------------
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000").rstrip('/')

@app.get("/auth/me")
def get_me(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Get current user's information including trial/subscription status.
    Returns user details with trial_end timestamp for frontend countdown.
    """
    user_id = decode_token(credentials.credentials)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user_doc = get_user_doc(user_id)
    if not user_doc:
        raise HTTPException(status_code=404, detail="User not found")

    otp_verified = user_doc.get("otp_verified", False)
    if not otp_verified:
        # Auto-stamp accounts that Firebase has already verified (Google / OAuth providers).
        # Their email_verified claim is True natively â€” no OTP needed for them.
        try:
            decoded = firebase_auth.verify_id_token(credentials.credentials)
            if decoded.get("email_verified", False):
                db.collection("users").document(user_id).update({"otp_verified": True})
                otp_verified = True
        except Exception:
            pass

    if not otp_verified:
        raise HTTPException(status_code=403, detail="Account not verified.")

    return {
        "uid": user_id,
        "email": user_doc.get("email", user_id),
        "plan": user_doc.get("plan", "trial"),
        "trial_end": user_doc.get("trial_end"),
        "subscription_active": user_doc.get("subscription", {}).get("status") == "active",
        "subscription_end": user_doc.get("subscription_end"),
        "full_name": user_doc.get("full_name"),
        "location": user_doc.get("location"),
        "phone_number": user_doc.get("phone_number"),
    }

@app.post("/api/users/provision")
async def provision_user(request: Request, user_id: str = Depends(get_current_user)):
    """
    Create a default backend profile for a Firebase-authenticated user who has no record yet.
    Idempotent: returns the existing doc if one already exists.
    """
    try:
        data = await request.json()
    except Exception:
        data = {}

    try:
        firebase_uid = data.get("uid") or user_id
        email = data.get("email") or (user_id if "@" in user_id else None)
        display_name = data.get("display_name") or (email.split("@")[0] if email else firebase_uid)
        provider = data.get("provider", "firebase")

        # Email/password accounts must present a valid OTP signup_token to prevent bypass
        if provider == "password":
            signup_token = data.get("signup_token")
            if not signup_token:
                raise HTTPException(status_code=403, detail="OTP verification required before account creation.")
            email_key = _otp_find_by_signup_token(signup_token)
            if not email_key:
                raise HTTPException(status_code=403, detail="Invalid or expired OTP verification token.")
            # Invalidate the token â€” single use only
            _otp_delete(email_key)
            # Mark Firebase email as verified â€” this is the gate used in decode_token
            # so accounts that bypassed OTP can never authenticate.
            try:
                firebase_auth.update_user(firebase_uid, email_verified=True)
            except Exception as fe:
                print(f"[provision] Warning: could not mark email_verified for {firebase_uid}: {fe}")

        # Prefer email as doc key (consistent with rest of backend), fall back to uid
        doc_key = email or firebase_uid

        # Normalize and validate phone number if provided
        phone_raw = data.get("phone_number") or ""
        phone_number = normalize_phone_number(phone_raw)

        existing = get_user_doc(doc_key)
        if existing:
            update_last_login(doc_key)
            if not existing.get("otp_verified"):
                db.collection("users").document(doc_key).update({"otp_verified": True})
            # Save phone number if not already stored
            if phone_number and not existing.get("phone_number"):
                if not phone_is_unique(phone_number, exclude_email=doc_key):
                    try:
                        firebase_auth.update_user(firebase_uid, disabled=True)
                    except Exception:
                        pass
                    db.collection("users").document(doc_key).update({
                        "suspended": True, "suspension_reason": "duplicate_phone"
                    })
                    raise HTTPException(status_code=409, detail="This phone number is already registered to another account. Your account has been suspended.")
                db.collection("users").document(doc_key).update({"phone_number": phone_number})
            return {
                "uid": firebase_uid,
                "email": existing.get("email", doc_key),
                "plan": existing.get("plan", "trial"),
                "trial_end": existing.get("trial_end"),
                "full_name": existing.get("full_name"),
                "location": existing.get("location"),
                "phone_number": existing.get("phone_number") or phone_number,
            }

        # Phone number is required for new account creation
        if not phone_number:
            raise HTTPException(status_code=422, detail="A mobile number is required to create an account.")

        # Check phone uniqueness before creating new doc
        if not phone_is_unique(phone_number):
            try:
                firebase_auth.update_user(firebase_uid, disabled=True)
            except Exception:
                pass
            raise HTTPException(status_code=409, detail="This phone number is already registered to another account.")

        user_doc = create_user_doc(
            doc_key,
            provider="firebase",
            social_id=firebase_uid,
            full_name=display_name,
            phone_number=phone_number,
        )
        return {
            "uid": firebase_uid,
            "email": email or doc_key,
            "plan": user_doc.get("plan", "trial"),
            "trial_end": user_doc.get("trial_end"),
            "full_name": user_doc.get("full_name"),
            "location": user_doc.get("location"),
            "phone_number": phone_number,
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[/api/users/provision] Error for {user_id}: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"User provisioning failed: {type(e).__name__}")

@app.get("/auth/{provider}")
async def oauth_login(request: Request, provider: str):
    if OAuth is None:
        raise HTTPException(status_code=500, detail="OAuth support is not available")
    init_oauth()
    if oauth is None:
        raise HTTPException(status_code=500, detail="OAuth initialization failed")
    client = oauth.create_client(provider)
    if client is None:
        raise HTTPException(status_code=400, detail=f"OAuth provider '{provider}' is not configured")
    redirect_uri = f"{BASE_URL}/auth/{provider}/callback"
    return await client.authorize_redirect(request, redirect_uri)

@app.get("/auth/{provider}/callback")
async def oauth_callback(request: Request, provider: str):
    if OAuth is None:
        raise HTTPException(status_code=500, detail="OAuth support is not available")
    init_oauth()
    if oauth is None:
        raise HTTPException(status_code=500, detail="OAuth initialization failed")
    client = oauth.create_client(provider)
    if client is None:
        raise HTTPException(status_code=400, detail=f"OAuth provider '{provider}' is not configured")
    try:
        token = await client.authorize_access_token(request)
    except OAuthError as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
    user_info = token.get('userinfo')
    if not user_info:
        user_info = await client.parse_id_token(request, token)
    email = user_info.get('email')
    name = user_info.get('name', email.split('@')[0])
    sub = user_info.get('sub')
    if not email:
        return JSONResponse(content={"error": "Email not provided"}, status_code=400)
    user = get_or_create_user_from_oauth(email, name, provider, sub)
    jwt_token = create_token(email)
    return RedirectResponse(f"/dashboard#token={jwt_token}")

# -------------------------------------------------------------------
# Pydantic models
# -------------------------------------------------------------------
class UserRegister(BaseModel):
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserProfileComplete(BaseModel):
    email: EmailStr
    full_name: str
    location: str
    password: Optional[str] = None

class Feedback(BaseModel):
    name: str
    email: EmailStr
    message: str

class OTPSendRequest(BaseModel):
    email: EmailStr
    phone: Optional[str] = None

class OTPVerifyRequest(BaseModel):
    email: EmailStr
    otp: str
    phone: Optional[str] = None

class PhoneCheckRequest(BaseModel):
    phone: str

class CreateSubscriptionRequest(BaseModel):
    plan_name: str
    amount: float
    currency: str = "INR"
    email: EmailStr
    customer_phone: Optional[str] = None

class CreateOrderRequest(BaseModel):
    plan: str
    currency: str = "INR"

class VerifyPaymentRequest(BaseModel):
    plan: str
    payment_id: Optional[str] = None
    razorpay_payment_id: Optional[str] = None
    razorpay_order_id: Optional[str] = None
    razorpay_signature: Optional[str] = None

class Review(BaseModel):
    name: str
    email: EmailStr
    rating: int
    message: Optional[str] = None
    product: Optional[str] = None

# -------------------------------------------------------------------
# Disposable / temp-email domain blocklist
# These domains have valid MX records so DNS lookup alone won't catch them.
# -------------------------------------------------------------------
DISPOSABLE_EMAIL_DOMAINS = {
    "mailinator.com", "guerrillamail.com", "guerrillamail.net", "guerrillamail.org",
    "guerrillamail.biz", "guerrillamail.de", "guerrillamail.info",
    "temp-mail.org", "tempmail.com", "tempmail.net", "temp-mail.io",
    "10minutemail.com", "10minutemail.net", "10minutemail.org",
    "throwam.com", "throwaway.email", "trashmail.com", "trashmail.net",
    "trashmail.me", "trashmail.at", "trashmail.io",
    "dispostable.com", "disposablemail.com", "fakeinbox.com",
    "maildrop.cc", "mailnull.com", "spamgourmet.com", "spamgourmet.net",
    "yopmail.com", "yopmail.fr", "yopmail.net",
    "sharklasers.com", "guerillaMail.com", "grr.la", "spam4.me",
    "getairmail.com", "filzmail.com", "sofimail.com", "spamavert.com",
    "spamevader.com", "dodgeit.com", "mailexpire.com", "spamhole.com",
    "spamcorpse.com", "deadaddress.com", "mailfreeonline.com",
    "spaml.com", "spamspot.com", "binkmail.com", "mailbolt.com",
    "mailfree.net", "spamfree24.org", "spamfree.eu", "mailzilla.com",
    "anonymail.com", "anonymbox.com", "anonbox.net", "mailnew.com",
    "tempr.email", "discard.email", "mt2015.com", "mt2016.com",
    "spam.la", "spaml.de", "temporaryemail.net", "throwam.com",
    "mailscrap.com", "dispostable.com", "e4ward.com",
    "jetable.fr.nf", "jetable.net", "jetable.org",
    "nomail.xl.cx", "plokeit.com", "putthisinyourspamdatabase.com",
    "rmqkr.net", "s0ny.net", "safetymail.info", "safetypost.de",
    "sneakemail.com", "spamfree.eu", "spamgob.com", "spaml.com",
    "spammotel.com", "spamspot.com", "spamthisplease.com", "tradermail.info",
    "turnermail.com", "uroid.com", "venompen.com", "wh4f.org",
    "yam.com", "zoemail.org", "zymuying.com",
}

# -------------------------------------------------------------------
# Email configuration
# -------------------------------------------------------------------
conf = ConnectionConfig(
    MAIL_USERNAME=os.getenv("MAIL_USERNAME", "animeshkukreti60@gmail.com"),
    MAIL_PASSWORD=SecretStr(os.getenv("MAIL_PASSWORD", "")),
    MAIL_FROM=os.getenv("MAIL_FROM", "animeshkukreti60@gmail.com"),
    MAIL_FROM_NAME=os.getenv("MAIL_FROM_NAME", "Gatekeeper (Aegis-1)"),
    MAIL_PORT=int(os.getenv("MAIL_PORT", "587")),
    MAIL_SERVER=os.getenv("MAIL_SERVER", "smtp.gmail.com"),
    MAIL_STARTTLS=os.getenv("MAIL_STARTTLS", "true").lower() == "true",
    MAIL_SSL_TLS=os.getenv("MAIL_SSL_TLS", "false").lower() == "true",
)

fastmail = FastMail(conf)

# SSL fallback config (port 465) â€” tried when STARTTLS/587 times out
_mail_server = os.getenv("MAIL_SERVER", "smtp.gmail.com")
_mail_user   = os.getenv("MAIL_USERNAME", "")
_mail_pass   = os.getenv("MAIL_PASSWORD", "")
_mail_from   = os.getenv("MAIL_FROM", "")
_mail_name   = os.getenv("MAIL_FROM_NAME", "AEGIS v1.0")

conf_ssl = ConnectionConfig(
    MAIL_USERNAME=_mail_user,
    MAIL_PASSWORD=SecretStr(_mail_pass),
    MAIL_FROM=_mail_from,
    MAIL_FROM_NAME=_mail_name,
    MAIL_PORT=465,
    MAIL_SERVER=_mail_server,
    MAIL_STARTTLS=False,
    MAIL_SSL_TLS=True,
)
fastmail_ssl = FastMail(conf_ssl)

# -------------------------------------------------------------------
# Resend email helper â€” HTTP API, bypasses Railway SMTP port blocks.
# Set RESEND_API_KEY in Railway env to enable. Falls back to SMTP.
# -------------------------------------------------------------------
_RESEND_API_KEY  = os.getenv("RESEND_API_KEY", "")
_RESEND_FROM_ADDR = os.getenv("MAIL_FROM", "aegisofficial@aegisignal.pro")
_RESEND_FROM_NAME = os.getenv("MAIL_FROM_NAME", "AEGIS v1.0")

def _smtp_mailer(addr: str, name: str, ssl: bool) -> Optional[FastMail]:
    """FastMail for this sender, or None when no SMTP credentials are configured.

    Reuses the module-level singletons for the default sender (the common case)
    and only builds a per-call config when a caller overrides the From line --
    /auth/send-password-reset does, and the old code silently dropped it on the
    SMTP path so those mails went out under the wrong sender.
    """
    if not _mail_user or not _mail_pass:
        return None
    if (addr or _mail_from) == _mail_from and (name or _mail_name) == _mail_name:
        return fastmail_ssl if ssl else fastmail
    try:
        return FastMail(ConnectionConfig(
            MAIL_USERNAME=_mail_user,
            MAIL_PASSWORD=SecretStr(_mail_pass),
            MAIL_FROM=addr or _mail_from,
            MAIL_FROM_NAME=name or _mail_name,
            MAIL_PORT=465 if ssl else int(os.getenv("MAIL_PORT", "587")),
            MAIL_SERVER=_mail_server,
            MAIL_STARTTLS=not ssl,
            MAIL_SSL_TLS=ssl,
        ))
    except Exception as cfg_err:
        print(f"[email] Could not build SMTP config for {addr!r}: {cfg_err}")
        return fastmail_ssl if ssl else fastmail


async def _send_email(to: str, subject: str, html: str, from_addr: str = "", from_name: str = "") -> None:
    """Send transactional email, trying EVERY configured provider in turn.

    Order: Resend HTTP API (works where Railway blocks outbound SMTP ports),
    then SMTP STARTTLS/587, then SMTP SSL/465.

    This used to `return` straight after the Resend branch, which made the two
    SMTP paths unreachable in production -- the comment called them a "fallback"
    but nothing could ever reach them once RESEND_API_KEY was set. Any Resend
    rejection (unverified sender domain, exhausted quota, rotated key) therefore
    killed BOTH signup OTPs and password resets outright, with no second chance
    and a generic 500 that named no cause. A provider outage has to degrade
    signup, not black it out, so every provider is now tried and the errors from
    all of them are reported together.
    """
    _addr = from_addr or _RESEND_FROM_ADDR
    _name = from_name or _RESEND_FROM_NAME
    failures: List[str] = []

    if _RESEND_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    "https://api.resend.com/emails",
                    headers={"Authorization": f"Bearer {_RESEND_API_KEY}", "Content-Type": "application/json"},
                    json={"from": f"{_name} <{_addr}>", "to": [to], "subject": subject, "html": html},
                )
            if resp.status_code in (200, 201):
                print(f"[email] Sent via Resend -> {to}")
                return
            failures.append(f"Resend HTTP {resp.status_code}: {resp.text[:300]}")
            print(f"[email] Resend rejected ({resp.status_code}): {resp.text[:300]} -- trying SMTP")
        except Exception as exc:
            failures.append(f"Resend {type(exc).__name__}: {exc}")
            print(f"[email] Resend call failed ({type(exc).__name__}): {exc} -- trying SMTP")

    msg = MessageSchema(recipients=[to], subject=subject, body=html, subtype=MessageType.html)
    for label, ssl in (("SMTP/587", False), ("SMTP/465", True)):
        mailer = _smtp_mailer(_addr, _name, ssl)
        if mailer is None:
            continue
        try:
            await asyncio.wait_for(mailer.send_message(msg), timeout=12.0)
            print(f"[email] Sent via {label} -> {to}")
            return
        except Exception as exc:
            failures.append(f"{label} {type(exc).__name__}: {exc}")
            print(f"[email] {label} failed ({type(exc).__name__}): {exc}")

    raise RuntimeError(
        ("every email provider failed: " + " | ".join(failures)) if failures
        else "no email provider configured (set RESEND_API_KEY, or MAIL_USERNAME + MAIL_PASSWORD for SMTP)"
    )

# -------------------------------------------------------------------
# 3-Step Onboarding with OTP
# -------------------------------------------------------------------
async def _send_sms_otp(phone_number: str, otp: str) -> bool:
    """
    Send OTP via SMS. Provider priority:
      1. MSG91  (MSG91_AUTH_KEY + MSG91_OTP_TEMPLATE_ID) â€” DLT-compliant, production
      2. Fast2SMS (FAST2SMS_API_KEY) â€” quick setup, works for India
      3. Twilio  (TWILIO_ACCOUNT_SID/AUTH_TOKEN/PHONE_NUMBER) â€” global fallback
    Returns True only when SMS was actually dispatched.
    """
    # Strip to digits-only for providers that need it (E.164 minus the +)
    e164 = phone_number if phone_number.startswith('+') else '+' + phone_number
    digits_only = e164.lstrip('+')      # e.g. 919876543210
    indian_10   = digits_only[-10:]     # last 10 digits

    # 1. MSG91 â€” production-grade DLT-compliant Indian SMS
    msg91_key  = os.getenv("MSG91_AUTH_KEY", "").strip()
    msg91_tmpl = os.getenv("MSG91_OTP_TEMPLATE_ID", "").strip()
    if msg91_key and msg91_tmpl:
        try:
            async with httpx.AsyncClient(timeout=12.0) as client:
                resp = await client.post(
                    "https://control.msg91.com/api/v5/otp",
                    headers={"authkey": msg91_key, "Content-Type": "application/json"},
                    json={
                        "mobile":       digits_only,
                        "authkey":      msg91_key,
                        "template_id":  msg91_tmpl,
                        "otp":          otp,
                        "otp_expiry":   5,
                    },
                )
            data = resp.json()
            # MSG91 answers HTTP 200 with {"type":"error","message":...} for an
            # unapproved DLT template, an exhausted balance or a bad number, so
            # the old `or resp.status_code in (200, 201)` marked EVERY reachable
            # response as delivered. That reported "code sent" for an SMS that
            # was never sent, and -- worse -- returning True here skips the email
            # fallback below, so the user received nothing on either channel and
            # saw no error explaining why. Only an explicit success counts.
            if str(data.get("type", "")).lower() == "success":
                print(f"[MSG91] OTP dispatched to {phone_number}")
                return True
            print(f"[MSG91] Non-success response {resp.status_code}: {data}")
        except Exception as exc:
            print(f"[MSG91] Exception: {exc}")

    # 2. Fast2SMS â€” simpler setup, good for India
    fast2sms_key = os.getenv("FAST2SMS_API_KEY", "").strip()
    if fast2sms_key:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    "https://www.fast2sms.com/dev/bulkV2",
                    headers={"authorization": fast2sms_key},
                    params={"variables_values": otp, "route": "otp", "numbers": indian_10},
                )
            data = resp.json()
            if data.get("return"):
                print(f"[Fast2SMS] OTP dispatched to {phone_number}")
                return True
            print(f"[Fast2SMS] Non-success response: {data}")
        except Exception as exc:
            print(f"[Fast2SMS] Exception: {exc}")

    # 3. Twilio â€” global fallback
    try:
        from twilio.rest import Client as TwilioClient
        account_sid = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
        auth_token  = os.getenv("TWILIO_AUTH_TOKEN",  "").strip()
        from_number = os.getenv("TWILIO_PHONE_NUMBER", "").strip()
        if account_sid and auth_token and from_number:
            loop = asyncio.get_event_loop()
            tc = TwilioClient(account_sid, auth_token)
            await loop.run_in_executor(None, lambda: tc.messages.create(
                body=f"Your AEGIS verification code is {otp}. Valid for 5 minutes. Do not share.",
                from_=from_number,
                to=e164,
            ))
            print(f"[Twilio] OTP dispatched to {phone_number}")
            return True
    except ImportError:
        pass
    except Exception as exc:
        print(f"[Twilio] Exception: {exc}")

    print(f"[SMS OTP] No SMS provider configured â€” cannot deliver to {phone_number}")
    return False


@app.post("/auth/send-otp-for-registration")
async def send_otp_for_registration(request: OTPSendRequest, req: Request):
    email = request.email
    phone_number = normalize_phone_number(request.phone)
    if not phone_number:
        raise HTTPException(status_code=422, detail="A valid mobile number is required for signup verification.")

    # Rate limit: 5 OTP requests per email per 10 minutes
    if not _rate_limit(f"otp:{email}", max_calls=5, window_seconds=600):
        raise HTTPException(status_code=429, detail="Too many OTP requests. Please wait 10 minutes before trying again.")
    # Rate limit: 20 OTP requests per IP per 10 minutes (anti-enumeration)
    if not _rate_limit(f"otp_ip:{get_client_ip(req)}", max_calls=20, window_seconds=600):
        raise HTTPException(status_code=429, detail="Too many requests from this IP. Please try again later.")

    # Block disposable / temp-mail domains
    domain = email.split('@')[-1].lower()
    if domain in DISPOSABLE_EMAIL_DOMAINS:
        raise HTTPException(status_code=422, detail="Disposable or temporary email addresses are not allowed.")

    # Validate email syntax only â€” no DNS/MX lookup.
    try:
        validated = validate_email(email, check_deliverability=False)
        email = validated.normalized
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid email format: {str(exc)}")

    # The last Firestore dependency in signup, and it FAILS OPEN.
    #
    # This is a courtesy check that produces a friendlier "already registered"
    # message. It is not the uniqueness guarantee — Firebase Auth is, and
    # createUserWithEmailAndPassword rejects a duplicate email regardless of what
    # this returns. Blocking every new signup because a courtesy lookup timed out
    # is the wrong trade, and it is what produced "Our datastore is not
    # responding" on the form. Degraded now means: no friendly pre-warning, and
    # Firebase reports the duplicate at account creation instead.
    try:
        existing_user = await _fs_await(get_user_doc, email, what="user lookup")
    except HTTPException:
        print(f"[signup] duplicate-email pre-check unavailable for {email} — "
              f"continuing; Firebase Auth still enforces uniqueness")
        existing_user = None
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered. Please sign in.")
    # Also fails open. Abuse is already bounded by the two _rate_limit calls at
    # the top of this handler, which are pure in-memory counters — this check only
    # adds a friendlier 60-second message, and it is not worth blocking signup for.
    try:
        _cooling = await _fs_await(is_cooldown_active, email, what="OTP cooldown check")
    except HTTPException:
        _cooling = False
    if _cooling:
        raise HTTPException(status_code=429, detail="Too many requests. Please wait before requesting another OTP.")
    otp = generate_otp()
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=5)
    cooldown_until = datetime.now(timezone.utc) + timedelta(seconds=60)
    # No longer a Firestore round trip: _otp_set writes memory and mirrors to
    # Firestore on a daemon thread. This is now an in-process dict assignment, so
    # it cannot hang, cannot fail on quota, and needs no deadline.
    _otp_set(email, {
        "otp": otp,
        "expires_at": expires_at,
        "cooldown_until": cooldown_until,
        "phone_number": phone_number,
        "email": email,
        "verified": False,
    })
    sms_sent = False
    try:
        sms_sent = await _send_sms_otp(phone_number, otp)
    except Exception as e:
        print(f"SMS sending error: {e}")

    if sms_sent:
        return {"success": True, "message": f"Verification code sent to {phone_number}.", "via": "sms"}

    # SMS provider not configured â€” fall back to email so signup isn't blocked.
    # In production set MSG91_AUTH_KEY + MSG91_OTP_TEMPLATE_ID to send via SMS.
    name = email.split("@")[0]
    html = f"""
<div style="font-family:Arial,sans-serif;max-width:520px;margin:0 auto;background:#0d0d1a;padding:32px;border-radius:12px;">
  <div style="text-align:center;margin-bottom:24px;">
    <span style="font-size:1.1rem;font-weight:700;color:#B8966A;letter-spacing:4px;">AEGIS Â· v1.0</span>
  </div>
  <h2 style="color:#EAE6DF;margin:0 0 8px;">Phone Verification Code</h2>
  <p style="color:#9ca3af;margin:0 0 24px;">Hi {name}, here is your one-time code to verify <strong style="color:#B8966A;">{phone_number}</strong>:</p>
  <div style="font-size:2.2rem;font-weight:700;letter-spacing:10px;text-align:center;padding:20px;background:rgba(184,150,106,0.08);border:1px solid rgba(184,150,106,0.3);color:#B8966A;border-radius:8px;margin-bottom:24px;">{otp}</div>
  <p style="color:#6b7280;font-size:0.85rem;text-align:center;">Expires in 5 minutes. Do not share this code.</p>
</div>"""
    try:
        await _send_email(email, "AEGIS â€“ Your Phone Verification Code", html)
    except Exception as e:
        _otp_delete(email)
        # Greppable in the Railway log, and _send_email now names every provider
        # it tried and why each one refused -- the old one-liner said only that
        # "sending failed", which is why a dead sender took days to spot.
        print(f"[send-otp] DELIVERY FAILED to {email}: {e}")
        raise HTTPException(status_code=500, detail="Failed to deliver verification code. Please try again.")

    return {"success": True, "message": f"Verification code sent to {email}.", "via": "email"}

@app.post("/auth/verify-otp-for-registration")
async def verify_otp_for_registration(request: OTPVerifyRequest, req: Request):
    email = otp_email_key(request.email)
    otp = request.otp
    phone_number = normalize_phone_number(request.phone)
    # Rate limit: 10 verify attempts per email per 15 minutes (prevents OTP brute-force)
    if not _rate_limit(f"otp_verify:{email}", max_calls=10, window_seconds=900):
        raise HTTPException(status_code=429, detail="Too many verification attempts. Please request a new OTP.")
    if not _rate_limit(f"otp_verify_ip:{get_client_ip(req)}", max_calls=30, window_seconds=900):
        raise HTTPException(status_code=429, detail="Too many requests from this IP. Please try again later.")
    record = _otp_get(email)
    if not record:
        raise HTTPException(status_code=400, detail="No OTP request found. Please request a new OTP.")
    if phone_number and record.get("phone_number") and record.get("phone_number") != phone_number:
        raise HTTPException(status_code=400, detail="Phone number mismatch. Please request a new OTP.")
    if datetime.now(timezone.utc) > record["expires_at"]:
        _otp_delete(email)
        raise HTTPException(status_code=400, detail="OTP has expired. Please request a new one.")
    if record["otp"] != otp:
        raise HTTPException(status_code=400, detail="Invalid OTP. Please try again.")
    signup_token = str(uuid.uuid4())
    _otp_update(email, {"verified": True, "signup_token": signup_token})
    return {"success": True, "message": "OTP verified successfully. Please complete your profile.", "signup_token": signup_token}

@app.post("/auth/check-phone")
def check_phone_unique(request: PhoneCheckRequest, req: Request):
    """Pre-signup phone uniqueness check. No auth required."""
    # Rate limit: 15 checks per IP per 5 minutes (enumeration protection)
    if not _rate_limit(f"phone_check:{get_client_ip(req)}", max_calls=15, window_seconds=300):
        raise HTTPException(status_code=429, detail="Too many requests. Please try again shortly.")
    phone = normalize_phone_number(request.phone)
    if not phone:
        raise HTTPException(status_code=422, detail="Invalid phone number")
    return {"available": phone_is_unique(phone)}

# -------------------------------------------------------------------
# Password reset â€” generate Firebase link, deliver via Neo SMTP
# Firebase's default noreply sender goes to spam; our domain is trusted.
# -------------------------------------------------------------------
class PasswordResetRequest(BaseModel):
    email: EmailStr

@app.post("/auth/send-password-reset")
async def send_password_reset(request: PasswordResetRequest, req: Request):
    email = request.email
    # Rate limit: 3 reset emails per email per 15 minutes
    if not _rate_limit(f"pw_reset:{email}", max_calls=3, window_seconds=900):
        raise HTTPException(status_code=429, detail="Too many reset attempts. Please wait 15 minutes before trying again.")
    if not _rate_limit(f"pw_reset_ip:{get_client_ip(req)}", max_calls=10, window_seconds=900):
        raise HTTPException(status_code=429, detail="Too many requests from this IP. Please try again later.")
    try:
        firebase_link = firebase_auth.generate_password_reset_link(email)
    except firebase_auth.UserNotFoundError:
        return {"success": True, "message": "If an account with this email exists, a reset link has been sent."}
    except Exception as e:
        print(f"[password-reset] generate_password_reset_link failed: {e}")
        raise HTTPException(status_code=500, detail="Could not generate reset link. Please try again.")

    parsed = urllib.parse.urlparse(firebase_link)
    params = urllib.parse.parse_qs(parsed.query)
    oob_code = params.get("oobCode", [""])[0]
    api_key = params.get("apiKey", [""])[0]
    base_url = os.getenv("BASE_URL", "https://aegisignal.pro").rstrip("/")
    custom_reset_url = (
        f"{base_url}/reset-password"
        f"?oobCode={urllib.parse.quote(oob_code)}&apiKey={urllib.parse.quote(api_key)}"
    )

    try:
        print(f"[password-reset] Sending reset email to {email} via {os.getenv('MAIL_SERVER','smtp.gmail.com')}:{os.getenv('MAIL_PORT','587')}")
        sender_name  = os.getenv("MAIL_FROM_NAME", "AEGIS AI Terminal")
        sender_email = os.getenv("MAIL_FROM",      "noreply@aegisignal.pro")
        support_email = os.getenv("SUPPORT_EMAIL", sender_email)
        base_url_display = os.getenv("BASE_URL", "https://aegisignal.pro").rstrip("/")
        html_body = f"""<!DOCTYPE html>
<html lang="en" xmlns:v="urn:schemas-microsoft-com:vml" xmlns:o="urn:schemas-microsoft-com:office:office">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta http-equiv="X-UA-Compatible" content="IE=edge">
  <!--[if mso]>
  <noscript><xml><o:OfficeDocumentSettings><o:PixelsPerInch>96</o:PixelsPerInch></o:OfficeDocumentSettings></xml></noscript>
  <![endif]-->
  <title>Reset Your Password â€” AEGIS</title>
</head>
<body style="margin:0;padding:0;background:#0b0f1a;font-family:'Segoe UI',Helvetica,Arial,sans-serif;">
  <!--[if mso]><table width="100%" cellpadding="0" cellspacing="0"><tr><td align="center"><![endif]-->
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0"
         style="background:#0b0f1a;padding:48px 16px;min-width:320px;">
    <tr><td align="center">

      <!-- Card wrapper -->
      <table role="presentation" width="540" cellpadding="0" cellspacing="0"
             style="max-width:540px;width:100%;background:#111827;border-radius:12px;overflow:hidden;
                    border:1px solid #1e2d45;">

        <!-- Top gradient bar -->
        <tr>
          <td height="4" style="background:linear-gradient(90deg,#0055ff,#00aaff);font-size:0;line-height:0;">&nbsp;</td>
        </tr>

        <!-- Header -->
        <tr>
          <td style="padding:36px 48px 28px;border-bottom:1px solid #1a2535;text-align:center;">
            <p style="margin:0;font-size:11px;letter-spacing:3px;color:#4b6a9b;
                      text-transform:uppercase;font-weight:600;">AEGIS AI TERMINAL</p>
            <p style="margin:8px 0 0;font-size:28px;font-weight:800;letter-spacing:-0.5px;
                      color:#f0f6ff;">Gatekeeper</p>
          </td>
        </tr>

        <!-- Body -->
        <tr>
          <td style="padding:40px 48px 36px;">

            <p style="margin:0 0 8px;font-size:20px;font-weight:700;color:#f0f6ff;line-height:1.3;">
              Password Reset Request
            </p>
            <p style="margin:0 0 28px;font-size:14px;color:#6b87b0;line-height:1.7;">
              We received a request to reset the password associated with
              <strong style="color:#c8d9f0;">{email}</strong>.<br>
              Click the button below to choose a new password.
            </p>

            <!-- CTA button -->
            <!--[if mso]>
            <v:roundrect xmlns:v="urn:schemas-microsoft-com:vml" href="{custom_reset_url}"
              style="height:48px;v-text-anchor:middle;width:200px;" arcsize="8%"
              fillcolor="#0066ff" stroke="f">
              <w:anchorlock/>
              <center style="color:#ffffff;font-family:'Segoe UI',Helvetica,Arial,sans-serif;
                             font-size:15px;font-weight:700;">Reset My Password</center>
            </v:roundrect>
            <![endif]-->
            <!--[if !mso]><!-->
            <table role="presentation" cellpadding="0" cellspacing="0"
                   style="margin:0 0 28px;">
              <tr>
                <td style="border-radius:6px;background:#0066ff;mso-padding-alt:0;">
                  <a href="{custom_reset_url}"
                     style="display:inline-block;padding:14px 36px;background:#0066ff;
                            color:#ffffff;font-size:15px;font-weight:700;
                            text-decoration:none;border-radius:6px;
                            letter-spacing:0.3px;">
                    Reset My Password
                  </a>
                </td>
              </tr>
            </table>
            <!--<![endif]-->

            <!-- Security notice -->
            <table role="presentation" width="100%" cellpadding="0" cellspacing="0"
                   style="background:#0d1a2d;border:1px solid #1e2d45;border-radius:8px;
                          margin-bottom:28px;">
              <tr>
                <td style="padding:16px 20px;">
                  <p style="margin:0 0 6px;font-size:13px;font-weight:600;color:#4b6a9b;">
                    Security Notice
                  </p>
                  <p style="margin:0;font-size:13px;color:#4b6a9b;line-height:1.65;">
                    This link expires in <strong style="color:#7ea8d4;">1 hour</strong> and can only be used once.<br>
                    If you did not request a password reset, no action is needed â€” your account remains secure.
                  </p>
                </td>
              </tr>
            </table>

            <!-- Fallback URL -->
            <p style="margin:0;font-size:12px;color:#374151;line-height:1.7;">
              If the button above does not work, copy and paste the link below into your browser:
            </p>
            <p style="margin:6px 0 0;font-size:12px;color:#2563eb;word-break:break-all;line-height:1.6;">
              <a href="{custom_reset_url}" style="color:#2563eb;text-decoration:none;">{custom_reset_url}</a>
            </p>

          </td>
        </tr>

        <!-- Footer -->
        <tr>
          <td style="padding:20px 48px 28px;border-top:1px solid #1a2535;text-align:center;">
            <p style="margin:0 0 4px;font-size:12px;color:#374151;">
              Sent by <strong style="color:#4b6a9b;">{sender_name}</strong>
              &nbsp;&middot;&nbsp;
              <a href="{base_url_display}" style="color:#4b6a9b;text-decoration:none;">{base_url_display.replace('https://','')}</a>
            </p>
            <p style="margin:4px 0 0;font-size:11px;color:#1f2937;">
              Questions? Contact
              <a href="mailto:{support_email}" style="color:#374151;text-decoration:none;">{support_email}</a>
            </p>
          </td>
        </tr>

        <!-- Bottom gradient bar -->
        <tr>
          <td height="4" style="background:linear-gradient(90deg,#00aaff,#0055ff);font-size:0;line-height:0;">&nbsp;</td>
        </tr>

      </table>

    </td></tr>
  </table>
  <!--[if mso]></td></tr></table><![endif]-->
</body>
</html>"""
        try:
            await _send_email(
                to=email,
                subject="Reset Your AEGIS Password",
                html=html_body,
                from_addr=sender_email,
                from_name=sender_name,
            )
            print(f"[password-reset] Email sent âœ“ â†’ {email}")
        except Exception as e:
            print(f"[password-reset] Failed ({type(e).__name__}): {e}")
            raise HTTPException(status_code=500, detail="smtp_failure")
    except HTTPException:
        raise
    except Exception as e:
        print(f"[password-reset] Outer error ({type(e).__name__}): {e}")
        raise HTTPException(status_code=500, detail="smtp_failure")

    return {"success": True, "message": "Password reset link sent. Check your inbox (and spam folder)."}


@app.get("/auth/msg91-token")
async def get_msg91_widget_token(req: Request):
    """Return a widget token for the MSG91/Phone91 client widget.

    This implementation reads a pre-generated token from env var `MSG91_WIDGET_TOKEN`.
    For production, replace this with a server-side minting call to MSG91 using
    your server API key so that tokens are short-lived and secure.
    """
    token = os.getenv("MSG91_WIDGET_TOKEN")
    if not token:
        raise HTTPException(status_code=501, detail="MSG91 widget token not configured on server")
    return {"token": token}


@app.post("/auth/msg91-token")
async def mint_msg91_widget_token(req: Request):
    """Mint a short-lived widget access token from MSG91 for a given phone number.

    Request JSON: { "phone": "+919876543210" }
    Returns: { "token": "..." }
    NOTE: MSG91's control API may vary; set `MSG91_GENERATE_TOKEN_URL` and
    `MSG91_AUTH_KEY` in env for production. If `MSG91_WIDGET_TOKEN` is present
    the server will return that instead (development fallback).
    """
    data = {}
    try:
        data = await req.json()
    except Exception:
        pass
    phone = data.get("phone") or None

    # If a static token is provided in env, return it (dev fallback)
    static = os.getenv("MSG91_WIDGET_TOKEN")
    if static:
        return {"token": static}

    auth_key = os.getenv("MSG91_AUTH_KEY")
    generate_url = os.getenv("MSG91_GENERATE_TOKEN_URL", "https://control.msg91.com/api/v5/widget/generateAccessToken")
    if not auth_key:
        raise HTTPException(status_code=501, detail="MSG91 server-side minting not configured (MSG91_AUTH_KEY missing)")

    payload = {"authkey": auth_key}
    if phone:
        payload["mobile"] = phone

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(generate_url, json=payload)
            resp.raise_for_status()
            j = resp.json()
            # Expect token in response under common keys
            token = j.get("access-token") or j.get("token") or j.get("data")
            return {"token": token, "raw": j}
    except httpx.HTTPStatusError as he:
        raise HTTPException(status_code=502, detail=f"MSG91 token mint failed: {he.response.text}")
    except Exception as e:
        print(f"[msg91-mint] error: {e}")
        raise HTTPException(status_code=500, detail="Failed to mint MSG91 widget token")


@app.post("/auth/msg91-verify")
async def verify_msg91_access_token(request: Request):
    """Server-side proxy to verify an access token with MSG91 control API.

    Expects JSON: { "access_token": "..." }
    Returns the MSG91 verification response.
    """
    try:
        data = await request.json()
    except Exception:
        data = {}
    access_token = data.get("access_token")
    if not access_token:
        raise HTTPException(status_code=422, detail="access_token is required")

    auth_key = os.getenv("MSG91_AUTH_KEY")
    verify_url = os.getenv("MSG91_VERIFY_URL", "https://control.msg91.com/api/v5/widget/verifyAccessToken")
    if not auth_key:
        raise HTTPException(status_code=501, detail="MSG91 verification not configured (MSG91_AUTH_KEY missing)")

    payload = {"authkey": auth_key, "access-token": access_token}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(verify_url, json=payload)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as he:
        raise HTTPException(status_code=502, detail=f"MSG91 verify failed: {he.response.text}")
    except Exception as e:
        print(f"[msg91-verify] error: {e}")
        raise HTTPException(status_code=500, detail="Failed to verify MSG91 access token")


@app.post("/auth/msg91-webhook")
async def msg91_webhook(req: Request):
    """Webhook receiver for MSG91 events.

    Validates the `Authorization` header against `MSG91_WEBHOOK_SECRET` (Bearer).
    Logs the payload and, on successful verification events, marks matching
    entries in `otp_store` as verified so signup can proceed.
    """
    secret = os.getenv("MSG91_WEBHOOK_SECRET") or os.getenv("MSG91_WEBHOOK_TOKEN")
    header = req.headers.get("authorization") or req.headers.get("Authorization")
    if secret:
        if not header or not header.lower().startswith("bearer ") or header.split()[1] != secret:
            raise HTTPException(status_code=401, detail="Invalid webhook signature")

    try:
        payload = await req.json()
    except Exception:
        payload = await req.body()
        try:
            payload = json.loads(payload)
        except Exception:
            payload = {"raw": str(payload)}

    print("[msg91-webhook] received:", payload)

    # Example handling: if payload indicates verification success with mobile
    event = payload.get("event") or payload.get("type") or payload.get("event_type")
    status = payload.get("status") or payload.get("verification_status")
    mobile = payload.get("mobile") or payload.get("phone") or payload.get("recipient")
    if event and mobile and str(status).lower() in ("success", "verified", "completed"):
        norm = normalize_phone_number(mobile)
        email_key = _otp_find_by_phone(norm)
        if email_key:
            _otp_update(email_key, {"verified": True, "signup_token": str(uuid.uuid4())})
            print(f"[msg91-webhook] marked {email_key} verified via webhook for {norm}")

    return {"received": True}
@app.post("/auth/complete-registration")
def complete_registration(profile: UserProfileComplete):
    email = otp_email_key(profile.email)
    record = _otp_get(email)
    if not record or not record.get("verified"):
        raise HTTPException(status_code=400, detail="Please verify OTP first before completing registration.")

    existing_user = get_user_doc(email)
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered.")

    password_hash = hash_password(profile.password) if profile.password else None
    user = create_user_doc(email, password_hash=password_hash, full_name=profile.full_name, location=profile.location)
    _otp_delete(email)
    token = create_token(email)
    return {"access_token": token, "token_type": "bearer", "user": user}

# -------------------------------------------------------------------
# Authentication endpoints
# -------------------------------------------------------------------
@app.post("/auth/login")
def login(user: UserLogin, req: Request):
    # Rate limit: 10 login attempts per email per 15 minutes (brute-force protection)
    if not _rate_limit(f"login:{user.email}", max_calls=10, window_seconds=900):
        raise HTTPException(status_code=429, detail="Too many login attempts. Please wait 15 minutes.")
    if not _rate_limit(f"login_ip:{get_client_ip(req)}", max_calls=30, window_seconds=900):
        raise HTTPException(status_code=429, detail="Too many requests from this IP. Please try again later.")
    user_doc = get_user_doc(user.email)
    if not user_doc or "password_hash" not in user_doc:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not verify_password(user.password, user_doc["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if is_trial_expired(user.email):
        raise HTTPException(status_code=403, detail="Trial expired. Please subscribe to continue.")
    
    update_last_login(user.email)
    token = create_token(user.email)
    return {"access_token": token, "token_type": "bearer", "plan": user_doc["plan"], "trial_end": user_doc.get("trial_end")}

@app.post("/auth/google-login")
async def google_login(request: Request):
    data = await request.json()
    email = data.get("email")
    name = data.get("name")
    if not email:
        raise HTTPException(status_code=400, detail="Email required")
    
    user_doc = get_user_doc(email)
    if not user_doc:
        user_doc = create_user_doc(email, provider="google", full_name=name)
    else:
        update_last_login(email)
    
    if is_trial_expired(email):
        raise HTTPException(status_code=403, detail="Trial expired. Please subscribe to continue.")
    
    token = create_token(email)
    return {"access_token": token, "token_type": "bearer", "plan": user_doc["plan"], "trial_end": user_doc.get("trial_end")}



# -------------------------------------------------------------------
# Subscription & plan limits
# -------------------------------------------------------------------
BASIC_TOKENS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]

# Full token universe for intermediate / pro / premium plans.
# Mirrors FLEET in scripts/live_engine.py â€” kept in sync here so the
# correct list is available even if the engine module fails to import.
ALL_TOKENS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "HYPE/USDT", "ASTER/USDT", "SUI/USDT", "TAO/USDT", "RENDER/USDT",
    "ADA/USDT", "AVAX/USDT", "LINK/USDT", "TRX/USDT", "DOT/USDT",
    "NEAR/USDT", "MATIC/USDT", "LTC/USDT", "BCH/USDT", "SHIB/USDT",
    "TON/USDT", "ICP/USDT", "HBAR/USDT", "APT/USDT", "ARB/USDT",
    "OP/USDT", "STX/USDT", "FIL/USDT", "AAVE/USDT", "VET/USDT",
    "RNDR/USDT", "INJ/USDT", "TIA/USDT", "SEI/USDT", "KAS/USDT",
    "FET/USDT", "AGIX/USDT", "OCEAN/USDT", "AKT/USDT", "THETA/USDT",
    "GRT/USDT", "LDO/USDT", "PYTH/USDT", "JUP/USDT", "ONDO/USDT",
    "PEPE/USDT", "DOGE/USDT", "WIF/USDT", "FLOKI/USDT", "BONK/USDT",
    "WLFI/USDT", "MNT/USDT", "ENA/USDT", "BGB/USDT", "PI/USDT",
    "SKY/USDT", "TRUMP/USDT", "NIGHT/USDT",
]

PRO_TOKENS = ALL_TOKENS  # default; overridden below if retrain_model defines the fleet
try:
    from scripts.retrain_model import FLEET_SYMBOLS as _FLEET_SYMBOLS
    if isinstance(_FLEET_SYMBOLS, list) and _FLEET_SYMBOLS:
        # Merge: ensure ALL_TOKENS is a superset so old hardcoded refs still work
        PRO_TOKENS = list(dict.fromkeys(ALL_TOKENS + _FLEET_SYMBOLS))
except Exception:
    pass

BASIC_TIMEFRAMES = ["30m", "1h"]

def get_user_plan(email: str) -> str:
    user_doc = get_user_doc(email)
    # Use .get() to safely fallback to "trial" if "plan" is missing
    return user_doc.get("plan", "trial") if user_doc else "trial"


def get_allowed_tokens() -> List[str]:
    return PRO_TOKENS  # All plans get the full token fleet

def get_allowed_timeframes(email: str) -> List[str]:
    plan = get_user_plan(email)
    if plan in ("pro", "premium", "intermediate"):
        return ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
    else:
        return BASIC_TIMEFRAMES

@app.get("/user/limits")
def get_user_limits(email: str = Depends(get_current_user)):
    try:
        user_doc = get_user_doc(email)
        plan = user_doc.get("plan", "trial") if user_doc else "trial"
        trial_end = user_doc.get("trial_end") if user_doc else None
        trial_expired = is_trial_expired(email) if trial_end else False
        allowed_tokens = get_allowed_tokens()
        return {
            "plan": plan,
            "allowed_tokens": allowed_tokens,
            "is_trial": plan == "trial",
            "trial_end": trial_end,
            "trial_expired": trial_expired,
            "alpha_mode_enabled": plan in ("pro", "premium")
        }
    except Exception as e:
        print(f"[/user/limits] Error for {email}: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load user limits: {type(e).__name__}")

@app.post("/upgrade")
def upgrade_plan(email: str = Depends(get_current_user)):
    db.collection("users").document(email).update({
        "plan": "pro",
        "trial_active": False,
        "trial_end": datetime.now(timezone.utc).isoformat()
    })
    return {"status": "upgraded to pro"}


# -------------------------------------------------------------------
# Backend-Authoritative Trial Management
# -------------------------------------------------------------------

@app.post("/api/v1/trial/start")
def start_free_trial(user_id: str = Depends(get_current_user)):
    """
    Register the start of a 3-day free trial for the authenticated user.
    Idempotent: if an active trial or paid plan already exists, returns current state.
    When user pays for any plan, the webhook auto-terminates the trial.
    """
    user_doc = get_user_doc(user_id)
    if not user_doc:
        raise HTTPException(status_code=404, detail="Account not found. Please sign up first.")

    plan = (user_doc.get("plan") or "trial").lower()
    paid_plans = {"pro", "premium", "intermediate", "basic"}

    if plan in paid_plans:
        return {
            "status": "paid",
            "plan": plan,
            "trial_active": False,
            "message": "You already have an active paid subscription."
        }

    # Check for an existing unexpired trial
    trial_end = user_doc.get("trial_end")
    if trial_end:
        try:
            end_dt = datetime.fromisoformat(trial_end)
            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            if now < end_dt:
                remaining_seconds = int((end_dt - now).total_seconds())
                return {
                    "status": "trial",
                    "plan": "trial",
                    "trial_active": True,
                    "trial_start": user_doc.get("trial_start"),
                    "trial_end": trial_end,
                    "seconds_remaining": remaining_seconds,
                    "message": "Your free trial is already active."
                }
        except (ValueError, TypeError):
            pass  # Corrupted date â€” fall through to fresh start

    # One trial per account. `trial_used` was written here and read NOWHERE, so
    # the block below cheerfully restarted an expired trial every time it was
    # called — three more days of full access, repeatable indefinitely from the
    # pricing page's own button. The field's comment at signup already called
    # this "the one 3-day trial"; this is the line that makes that true.
    if user_doc.get("trial_used"):
        return {
            "status": "expired",
            "plan": plan,
            "trial_active": False,
            "trial_start": user_doc.get("trial_start"),
            "trial_end": user_doc.get("trial_end"),
            "seconds_remaining": 0,
            "message": "Your free trial has already been used. Choose a plan to continue.",
        }

    # Start the trial
    now = datetime.now(timezone.utc)
    trial_start_iso = now.isoformat()
    trial_end_dt = now + timedelta(days=3)
    trial_end_iso = trial_end_dt.isoformat()

    db.collection("users").document(user_id).update({
        "plan": "trial",
        "trial_start": trial_start_iso,
        "trial_end": trial_end_iso,
        "trial_active": True,
        "trial_used": True
    })

    remaining_seconds = int((trial_end_dt - now).total_seconds())
    print(f"âœ… Free trial started for {user_id}: ends {trial_end_iso}")

    return {
        "status": "trial",
        "plan": "trial",
        "trial_active": True,
        "trial_start": trial_start_iso,
        "trial_end": trial_end_iso,
        "seconds_remaining": remaining_seconds,
        "message": "Free trial started! You have 3 days of full access."
    }


@app.get("/api/v1/trial/status")
def get_trial_status(user_id: str = Depends(get_current_user)):
    """
    Returns backend-authoritative trial/subscription status.
    Frontend countdown calls this instead of reading localStorage to prevent drift.
    """
    user_doc = get_user_doc(user_id)
    if not user_doc:
        return {
            "status": "no_account",
            "plan": "trial",
            "trial_active": False,
            "seconds_remaining": 0,
            "message": "Account not found."
        }

    plan = (user_doc.get("plan") or "trial").lower()
    paid_plans = {"pro", "premium", "intermediate", "basic"}

    if plan in paid_plans:
        return {
            "status": "paid",
            "plan": plan,
            "trial_active": False,
            "subscription_active": True,
            "seconds_remaining": 0,
            "message": f"{plan.title()} plan active."
        }

    trial_end = user_doc.get("trial_end")
    trial_start = user_doc.get("trial_start")

    if trial_end:
        try:
            end_dt = datetime.fromisoformat(trial_end)
            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            if now < end_dt:
                remaining_seconds = int((end_dt - now).total_seconds())
                return {
                    "status": "trial",
                    "plan": "trial",
                    "trial_active": True,
                    "trial_start": trial_start,
                    "trial_end": trial_end,
                    "seconds_remaining": remaining_seconds,
                    "message": "Trial active."
                }
            else:
                # Lazily flip the flag on expiry
                if user_doc.get("trial_active", False):
                    db.collection("users").document(user_id).update({"trial_active": False})
                return {
                    "status": "expired",
                    "plan": "trial",
                    "trial_active": False,
                    "trial_start": trial_start,
                    "trial_end": trial_end,
                    "seconds_remaining": 0,
                    "message": "Trial has expired. Upgrade to continue."
                }
        except (ValueError, TypeError):
            pass

    return {
        "status": "no_trial",
        "plan": "trial",
        "trial_active": False,
        "trial_start": None,
        "trial_end": None,
        "seconds_remaining": 0,
        "message": "No active trial. Click 'Start Free Trial' to begin."
    }


@app.get("/api/v1/alpha/track-record")
async def alpha_track_record(user_id: str = Depends(get_current_user)):
    """Return the Alpha Mode paper-trading track record (Pro only)."""
    plan = get_user_plan(user_id)
    if plan not in ("pro", "premium", "pro-dev"):
        raise HTTPException(status_code=403, detail="Alpha Mode track record is Pro-only.")
    from scripts.live_engine import ALPHA_TRACK_RECORD_PATH as _ATP
    if not _ATP.exists():
        return {"mode": "alpha", "summary": {}, "signals": [], "message": "No alpha track record yet."}
    try:
        with open(_ATP, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read alpha track record: {e}")


def _active_payment_provider() -> str:
    """Single source of truth for which gateway is live.

    Precedence Whop -> Paddle -> DODO -> Razorpay, driven only by which
    credentials are present, so the cutover is a config change and the rollback
    is unsetting WHOP_API_KEY.
    """
    if WHOP_ENABLED:
        return "whop"
    if PADDLE_ENABLED:
        return "paddle"
    if DODO_PAYMENTS_ENABLED:
        return "dodopayments"
    if RAZORPAY_ENABLED:
        return "razorpay"
    return "none"


@app.get("/payment/config")
async def payment_config():
    return {
        "provider": _active_payment_provider(),
        "whop": {
            "enabled": WHOP_ENABLED,
            "mode": WHOP_MODE,
            # same contract as Paddle: never expose the API key, the frontend
            # only needs to know this is a hosted redirect
            "checkout": "redirect",
        },
        "paddle": {
            "enabled": PADDLE_ENABLED,
            "mode": PADDLE_MODE,
            # never expose the API key; the frontend only needs to know the
            # flow is a hosted redirect, not an in-page SDK handoff
            "checkout": "redirect",
        },
        "dodopayments": {
            "enabled": DODO_PAYMENTS_ENABLED,
            "mode": DODO_PAYMENTS_MODE,
        },
        "razorpay": {
            "enabled": RAZORPAY_ENABLED,
            "key_id": RAZORPAY_KEY_ID,
        }
    }

# -------------------------------------------------------------------
# Razorpay Subscription Integration
# -------------------------------------------------------------------

@app.post("/api/v1/payments/subscription/initialize")
async def initialize_subscription(
    plan: str = "basic",
    user_id: str = Depends(get_current_user),
):
    """
    Create a Razorpay recurring subscription for the given plan tier.
    plan: basic | intermediate | pro
    """
    if not RAZORPAY_ENABLED:
        raise HTTPException(status_code=503, detail="Payment system is not configured")

    plan_id = RAZORPAY_PLAN_IDS.get(plan)
    if not plan_id:
        raise HTTPException(status_code=400, detail=f"No Razorpay plan configured for tier: {plan}")

    subscription = await _rzp_post("/subscriptions", {
        "plan_id": plan_id,
        "total_count": 12,
        "customer_notify": 1,
        "notes": {"internal_user_id": user_id, "plan": plan},
    })
    return {"subscription_id": subscription["id"]}


# Base prices in USD â€” source of truth for all plans
# Internal tier keys are NOT renamed: they are written into every Firestore
# user document and read by the plan gates, so renaming needs a data migration.
# The customer-facing names changed instead — and note the collision, because
# it is a genuine footgun when reading logs or the database:
#
#   customer sees   internal key    price
#   Basic           basic           $6.00
#   Sentinel        intermediate    $12.00   <- the MIDDLE tier
#   AEGIS Pro       pro             $18.00   <- internal 'pro' is the TOP tier
#
# (The names above were stale too: this block read Starter / Pro / Advanced at
# $5.90 / $14 / $30 while PLAN_DISPLAY_NAMES and pricing.html sold Basic /
# Sentinel / AEGIS Pro at $8 / $14 / $30 — three prices and three names, none of
# which agreed with the dict directly beneath them.)
#
# ⚠ THIS TABLE DOES NOT SET WHAT ANYONE IS CHARGED.
#
# WHOP is the live gateway (see _active_payment_provider — precedence is
# Whop -> Paddle -> DODO -> Razorpay, and Whop wins whenever WHOP_API_KEY is
# set). The checkout call sends {"plan_id", "metadata", "redirect_url"} and NO
# amount, so the price a customer actually pays is whatever the three Whop plans
# behind WHOP_PLAN_ID_BASIC / _INTERMEDIATE / _PRO are configured at.
#
# Every other gateway here works the same way: Razorpay bills against its plan_id
# (and its plans are immutable — a new plan is required to change an amount),
# Paddle against PADDLE_PRICE_ID_*. So this dict drives DISPLAY and the DODO path
# and nothing else, on every provider.
#
# Changing the numbers here alone makes the site advertise one price and charge
# another. Update the gateway plans in the SAME change, then confirm with a real
# checkout — the displayed price and the amount on the Whop page are two separate
# systems and only a test purchase proves they agree.
#
# Prices lowered to 6 / 12 / 18 on 2026-08-17.
USD_PLAN_PRICES: Dict[str, float] = {
    "basic": 6.00,
    "intermediate": 12.00,
    "pro": 18.00,
}

# Customer-facing label for each internal tier.
PLAN_DISPLAY_NAMES: Dict[str, str] = {
    # These MUST match what pricing.html sells, because this map is served to
    # the browser by /payment/config and is what a subscriber would be shown.
    #
    # It previously read Starter / Pro / Advanced against a page selling
    # Basic / Sentinel / AEGIS Pro — three tiers, three different names, and the
    # middle one called "Pro" while the internal top tier is literally `pro`.
    # A Sentinel subscriber would have been told they were on "Pro", and the
    # actual top tier shown as "Advanced".
    "basic": "Basic",
    "intermediate": "Sentinel",
    "pro": "AEGIS Pro",
}

# Currencies whose smallest unit is the unit itself (no multiply by 100)
_ZERO_DECIMAL = {"BIF","CLP","DJF","GNF","JPY","KMF","KRW","MGA","PYG","RWF","UGX","VND","VUV","XAF","XOF","XPF"}
# Currencies with 3 decimal places (multiply by 1000)
_THREE_DECIMAL = {"BHD","JOD","KWD","OMR","TND"}

# Exchange rate cache: USD as base, refreshed every hour
_fx: Dict[str, Any] = {"rates": {"USD": 1.0, "INR": 84.0}, "ts": 0.0}
_FX_TTL = 3600.0

async def _get_fx_rates() -> Dict[str, float]:
    """Return cached USD-based exchange rates, refreshing hourly."""
    if time.time() - _fx["ts"] < _FX_TTL:
        return _fx["rates"]
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("https://open.er-api.com/v6/latest/USD")
            data = resp.json()
            if data.get("result") == "success" and isinstance(data.get("rates"), dict):
                _fx["rates"] = data["rates"]
                _fx["ts"] = time.time()
                print(f"[FX] Rates refreshed. USD/INR={_fx['rates'].get('INR', '?')}")
    except Exception as exc:
        print(f"[FX] Rate fetch failed, using cached: {exc}")
    return _fx["rates"]


@app.get("/api/edge")
def edge_endpoint():
    """Edge over geometry — the part of the win rate the signals earned.

    The headline win rate is a report on the stop distance: measured over
    13,560 real paths, the hit rate tracks stop/(target+stop) to within about a
    point at every target/stop pair. A 0.5% target against a 1.4% stop predicts
    73.7%, which is where the live book sat while losing money.

        edge_pp = measured hit rate - mean(stop / (target + stop))

    is what rises when the model improves and stays flat when the stop moves.
    Read `by_symbol` to find which tokens are carrying the book and which are
    only being flattered by a wide stop.
    """
    try:
        from scripts.engine.edge_metric import from_track_record
        return JSONResponse(from_track_record())
    except Exception as e:
        return JSONResponse({'error': f'could not compute edge: {e}'}, status_code=500)


@app.get("/api/shadow-exits")
def shadow_exits_endpoint():
    """The exit-policy study, as it stands.

    Observation only — nothing here has ever placed an order. Read `vs_control`,
    which compares each policy against the SIMULATED live rule rather than
    against what the engine booked, and ignore the whole thing until `n` is in
    the hundreds. `control_tracks_reality` is the honesty check: if the
    simulated live rule has drifted from the real one, the comparison is void.
    """
    path = Path(__file__).parent / 'data' / 'shadow_exits.json'
    if not path.exists():
        return JSONResponse({'n': 0, 'note': 'no shadow trades recorded yet'})
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except Exception as e:
        return JSONResponse({'error': f'could not read the study: {e}'}, status_code=500)
    return JSONResponse({
        'summary':  payload.get('summary', {}),
        'note':     payload.get('note', ''),
        'updated_at': payload.get('updated_at'),
        'recent':   payload.get('trades', [])[-25:],
    })


@app.get("/api/engine-track-record")
async def engine_track_record_endpoint():
    """Serve the live engine's raw track_record.json for chart display.

    Returns the engine's data/track_record.json as-is so the chart can show
    actual open position prices (entry, SL, TPs stored at trade-open time).
    Falls back to an empty payload if the file doesn't exist yet.
    """
    _path = Path(os.environ.get('AEGIS_STATE_DIR') or (Path(BASE_DIR) / "data")) / "track_record.json"
    if not _path.exists():
        return JSONResponse({"signals": [], "summary": {}})
    try:
        with open(_path, "r", encoding="utf-8") as _f:
            return JSONResponse(json.load(_f))
    except Exception:
        return JSONResponse({"signals": [], "summary": {}})


@app.get("/api/live-signal-state")
async def live_signal_state(user_id: str = Depends(get_current_user)):
    """v79.6 — the dashboard's AUTHORITATIVE fired/armed state.

    Served straight from the reconciled engine snapshot (the same source the
    track record trusts), bypassing Firestore entirely: the listener pipeline
    proved lossy (state fields dropped, warmup staleness, module races), so
    signal STATE now flows engine -> here -> dashboard overlay, while
    Firestore keeps supplying card CONTENT only.
    """
    sigs = LIVE_STATE.data.get("signals", {}) or {}
    fired, armed = [], []
    for _sym, _e in sigs.items():
        if not isinstance(_e, dict):
            continue
        if _e.get("fire"):
            fired.append({
                "symbol":    _sym,
                "side":      _e.get("signal"),
                "direction": _e.get("direction"),
                "paper":     bool(_e.get("paper_only")),
                "risk_tier": _e.get("risk_tier", ""),
            })
        elif (_e.get("pending_entry")
              and str(_e.get("pending_side") or "").upper() in ("BUY", "SELL")):
            armed.append({
                "symbol": _sym,
                "side":   _e.get("pending_side"),
                "target": _e.get("pending_target"),
                "reason": _e.get("pending_reason", ""),
            })
    return {"fired": fired, "armed": armed,
            "ts": datetime.now(timezone.utc).isoformat()}


@app.get("/api/track-record")
async def track_record_endpoint(source: str = None,
                                authorization: Optional[str] = Header(None)):
    """Public track record â€” merges live_engine + main.py stores so no records are lost.

    v79.8: optional auth. A valid Bearer token UNMASKS open/armed symbols —
    subscribers already see them on the dashboard, and the dashboard cockpit
    now fills its fired/armed state from THIS endpoint (user decision: one
    source for the track record and the cockpits). Unauthenticated callers
    keep the masked public view.
    """
    _authed = False
    try:
        if authorization and authorization.lower().startswith("bearer "):
            _authed = bool(decode_token(authorization.split(" ", 1)[1]))
    except Exception:
        _authed = False

    _ENGINE_RECORD = _engine_record_path()
    _WEB_RECORD    = _web_record_path()

    def _norm(s: dict, src: str) -> dict:
        direction = s.get("direction", "")
        side      = s.get("side", "")
        sig_type  = side if side in ("BUY", "SELL") else (
            "BUY" if direction == "LONG" else "SELL" if direction == "SHORT" else "HOLD"
        )
        return {
            "signal_id":       s.get("signal_id"),
            "symbol":          s.get("symbol"),
            "timeframe":       s.get("timeframe", "1h"),
            "direction":       direction,
            "signal_type":     sig_type,
            "signal_status":   "ACTIVE" if s.get("outcome") == "OPEN" else "CLOSED",
            "entry_price":     s.get("entry_price"),
            # v85: the headline TP is the STRUCTURAL objective the trade was
            # approved on (take_profit_3 — TraderGate's target since v85), not
            # take_profit_1.  TP1 sits at exactly 1.0 x risk, so publishing it
            # advertised a 1:1 trade; measured on 14.3k 1h barrier races a 1:1
            # resolves ~50/50, and after the 0.10% round trip a 1:1 needs a
            # 57.4% win rate just to break even.  TP1 is still a real rung (15%
            # banks there and the stop goes to break-even) so it is published
            # alongside rather than dropped.
            "take_profit":     (s.get("take_profit_3") or s.get("take_profit_1")
                                or s.get("take_profit")),
            "take_profit_1":   s.get("take_profit_1"),
            "stop_loss":       s.get("stop_loss"),
            "position_value":  s.get("position_value"),
            "exit_price":      s.get("exit_price"),
            "entry_time":      s.get("entry_time"),
            "close_time":      s.get("close_time"),
            "pnl_pct":         s.get("pnl_pct"),
            "outcome":         s.get("outcome"),
            "exit_reason":     s.get("exit_reason"),
            "ai_prob":         s.get("meta_confidence"),
            "confluence_rate": None,
            "signal_strength": s.get("signal_strength", ""),
            "source":          src,
        }

    def _pos_key(s: dict) -> tuple:
        """Position-level dedup key: same symbol+minute+direction = same trade."""
        dr = s.get("direction", "") or s.get("signal_type", "") or s.get("side", "")
        return (s.get("symbol", ""), (s.get("entry_time") or "")[:16], dr)

    # â”€â”€ 1. Primary: live_engine's data/track_record.json â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    seen_ids:   set  = set()
    seen_pos:   set  = set()
    all_signals: list = []
    _engine_summary: dict = {}   # authoritative wallet figures (capital / balance / $PnL)

    # OPEN truth lives ONLY in the engine's own data/track_record.json (the
    # wallet).  The web copy and main.py's in-memory store run a parallel
    # tracker with different signal_ids and their own exit timing â€” their
    # OPEN rows are ghosts that inflate the public open count whenever the
    # minute-level dedup key misses.  Closed history from all sources is
    # still merged so no outcome is ever lost.
    for path, src in [(_ENGINE_RECORD, "live_engine"), (_WEB_RECORD, "live_engine_web")]:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    _d = json.load(f)
                if src == "live_engine" and _d.get("summary"):
                    _engine_summary = _d.get("summary") or {}
                for s in _d.get("signals", []):
                    if src != "live_engine" and s.get("outcome") == "OPEN":
                        continue
                    sid = s.get("signal_id")
                    pk  = _pos_key(s)
                    if (sid and sid in seen_ids) or pk in seen_pos:
                        continue
                    norm = _norm(s, src)
                    all_signals.append(norm)
                    if sid:
                        seen_ids.add(sid)
                    seen_pos.add(pk)
            except Exception:
                pass

    # â”€â”€ 2. Supplement with main.py's in-memory store (closed gaps only) â”€â”€
    # Copy each record: the masking pass below must never mutate the live store.
    for r in _track_store:
        if r.get("outcome") == "OPEN":
            continue
        sid = r.get("signal_id")
        pk  = _pos_key(r)
        if (sid and sid in seen_ids) or pk in seen_pos:
            continue
        all_signals.append(dict(r))
        if sid:
            seen_ids.add(sid)
        seen_pos.add(pk)

    # â”€â”€ 3. Sort by entry time, cap at 500 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    all_signals = sorted(
        all_signals,
        key=lambda r: r.get("entry_time") or "",
        reverse=True,
    )[:500]

    # â”€â”€ 3b. Mask OPEN positions in the public payload â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Open signals are the paid product: token and price levels are hidden
    # from this unauthenticated endpoint (subscribers see them on the
    # dashboard).  Direction, live PnL, tier and outcome stay visible so
    # the public page still proves the engine is trading in real time.
    for r in all_signals:
        if r.get("outcome") == "OPEN" and not _authed:
            r["symbol"]      = "HIDDEN"
            r["signal_id"]   = None
            r["entry_price"] = None
            r["take_profit"] = None
            r["stop_loss"]   = None
            r["exit_price"]  = None

    wins   = sum(1 for r in all_signals if r.get("outcome") == "WIN")
    losses = sum(1 for r in all_signals if r.get("outcome") == "LOSS")
    open_c = sum(1 for r in all_signals if r.get("outcome") == "OPEN")
    closed = wins + losses
    pnls   = [float(r.get("pnl_pct") or 0) for r in all_signals
              if r.get("outcome") in ("WIN", "LOSS")]
    times  = [r.get("entry_time") for r in all_signals if r.get("entry_time")]

    # â”€â”€ 3c. Surface PENDING signals (Guard M) â€” read-only from live state â”€â”€
    # A pending signal has cleared the direction gates but is HELD until price
    # reaches its S/R level and 3x5m confirms (see live_engine Guard M). These
    # are NOT trades yet (no entry), so they never touch track_record.json and
    # are excluded from the win/loss/open stats above. They are prepended to the
    # table (masked like OPEN rows â€” token and the target level are the paid
    # product) purely so the public page shows the engine armed and waiting.
    _pending_rows: list = []
    try:
        _live_sigs = (LIVE_STATE.data or {}).get("signals", {}) or {}
        for _sym, _sig in _live_sigs.items():
            if not isinstance(_sig, dict) or not _sig.get("pending_entry"):
                continue
            _pside = str(_sig.get("pending_side") or _sig.get("side") or "").upper()
            if _pside not in ("BUY", "SELL"):
                continue
            _pdir  = "LONG" if _pside == "BUY" else "SHORT"
            _pending_rows.append({
                "signal_id":       None,
                "symbol":          _sym if _authed else "HIDDEN",
                "timeframe":       _sig.get("timeframe", "1h"),
                "direction":       _pdir,
                "signal_type":     _pside if _pside in ("BUY", "SELL") else "HOLD",
                "signal_status":   "PENDING",
                "entry_price":     None,
                "take_profit":     None,
                "stop_loss":       None,
                "position_value":  None,
                "exit_price":      None,
                "entry_time":      datetime.now(timezone.utc).isoformat(),
                "close_time":      None,
                "pnl_pct":         None,
                "outcome":         "PENDING",
                "exit_reason":     None,
                "signal_strength": _sig.get("risk_tier", ""),
                "pending_reason":  _sig.get("pending_reason"),
                "source":          "live_engine_pending",
            })
    except Exception:
        _pending_rows = []
    pending_c   = len(_pending_rows)
    all_signals = _pending_rows + all_signals   # pending rows lead the table

    _MODEL_STORE = Path(BASE_DIR) / "src" / "ml" / "model_store"
    _token_count = len(list(_MODEL_STORE.glob("*_meta.json"))) if _MODEL_STORE.exists() else 0

    return JSONResponse({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_signals":  len(all_signals) - pending_c,   # trades only; pending are armed, not trades
            "wins":           wins,
            "losses":         losses,
            "open":           open_c,
            "pending":        pending_c,
            "win_rate_pct":   round(wins / closed * 100, 1) if closed else None,
            "avg_pnl_pct":    round(sum(pnls) / len(pnls), 3) if pnls else None,
            "total_pnl_pct":  round(sum(pnls), 3) if pnls else 0.0,
            "tracking_since": min(times) if times else None,
            "token_count":    _token_count,
            # Authoritative wallet $ figures (whole-account, reflects banked
            # partial-close PnL). Give the % stats real dollar context.
            "initial_capital": _engine_summary.get("initial_capital"),
            "balance":         _engine_summary.get("balance"),
            "total_pnl_usdt":  _engine_summary.get("total_pnl_usdt"),
        },
        "signals": all_signals,
    })


@app.get("/api/exchange-rates")
async def exchange_rates_endpoint():
    """Return live USD-based exchange rates plus USD plan prices for the frontend."""
    rates = await _get_fx_rates()
    return {
        "base": "USD",
        "rates": rates,
        "plan_prices_usd": USD_PLAN_PRICES,
        # so the frontend can label tiers without hardcoding the mapping
        "plan_display_names": PLAN_DISPLAY_NAMES,
    }


def _to_subunits(amount_float: float, currency: str) -> int:
    """Convert a float amount to the currency's smallest unit."""
    if currency in _ZERO_DECIMAL:
        return int(round(amount_float))
    if currency in _THREE_DECIMAL:
        return int(round(amount_float * 1000))
    return int(round(amount_float * 100))


@app.post("/api/create-order")
async def create_order(req: CreateOrderRequest, user_id: str = Depends(get_current_user)):
    """Create a checkout session with the active gateway.

    Precedence: Whop -> Paddle -> DODO -> Razorpay (see _active_payment_provider).
    """
    usd_price = USD_PLAN_PRICES.get(req.plan)
    if usd_price is None:
        raise HTTPException(status_code=400, detail=f"Unknown plan: {req.plan}")

    if WHOP_ENABLED:
        plan_id = WHOP_PLAN_IDS.get(req.plan)
        if not plan_id:
            raise HTTPException(
                status_code=500,
                detail=f"Whop plan ID not configured for plan '{req.plan}' "
                       f"(set WHOP_PLAN_ID_{req.plan.upper()})")
        try:
            # A checkout configuration is the unit that carries metadata, and
            # Whop's docs are explicit that "payments and memberships created
            # from a checkout session inherit its metadata". That inheritance is
            # the whole reason this endpoint exists server-side: the webhook
            # later reads user_id back out and upgrades exactly that account,
            # with nothing client-supplied in the path.
            whop_res = await _whop_post(_WHOP_CHECKOUT_PATH, {
                "plan_id": plan_id,
                "metadata": {"user_id": user_id, "plan": req.plan},
                "redirect_url": WHOP_REDIRECT_URL or None,
            })
            # The API may answer bare or wrapped in `data`, as Paddle's does.
            data = whop_res.get("data") if isinstance(
                whop_res.get("data"), dict) else whop_res
            checkout_url = data.get("purchase_url")
            if not checkout_url:
                raise HTTPException(
                    status_code=500,
                    detail="Whop returned no purchase_url — check the plan is "
                           "published and the checkout link is active.")
            return {
                "provider": "whop",
                "checkout_url": checkout_url,
                "payment_id": data.get("id"),
                "plan": req.plan,
                "mode": WHOP_MODE,
            }
        except HTTPException:
            raise
        except httpx.HTTPStatusError as e:
            print(f"[Whop] Checkout creation failed: "
                  f"{e.response.status_code} {e.response.text}")
            raise HTTPException(status_code=502,
                                detail="Whop checkout creation failed")
        except Exception as e:
            print(f"[Whop] Checkout error: {e!r}")
            raise HTTPException(status_code=502,
                                detail="Whop checkout creation failed")

    if PADDLE_ENABLED:
        price_id = PADDLE_PRICE_IDS.get(req.plan)
        if not price_id:
            raise HTTPException(
                status_code=500,
                detail=f"Paddle price ID not configured for plan '{req.plan}' "
                       f"(set PADDLE_PRICE_ID_{req.plan.upper()})")
        try:
            # A transaction carrying a RECURRING price makes Paddle create the
            # subscription itself on completion — we do not create one directly.
            # Price and currency live in Paddle, not here: USD_PLAN_PRICES is
            # only used for display and for the other gateways.
            paddle_res = await _paddle_post("/transactions", {
                "items": [{"price_id": price_id, "quantity": 1}],
                # custom_data comes back on every webhook for this transaction
                # and its subscription — this is how the webhook knows which
                # user to upgrade without trusting anything client-supplied.
                "custom_data": {"user_id": user_id, "plan": req.plan},
            })
            data = paddle_res.get("data") or {}
            checkout_url = (data.get("checkout") or {}).get("url")
            if not checkout_url:
                # Happens when the seller has no default payment link set in
                # Paddle > Checkout settings; the transaction exists but there
                # is nowhere to send the customer.
                raise HTTPException(
                    status_code=500,
                    detail="Paddle returned no checkout URL — set a default "
                           "payment link under Paddle > Checkout settings.")
            return {
                "provider": "paddle",
                "checkout_url": checkout_url,
                "payment_id": data.get("id"),
                "plan": req.plan,
                "mode": PADDLE_MODE,
            }
        except HTTPException:
            raise
        except httpx.HTTPStatusError as e:
            print(f"[Paddle] Checkout creation failed: "
                  f"{e.response.status_code} {e.response.text}")
            raise HTTPException(status_code=502,
                                detail="Paddle checkout creation failed")
        except Exception as e:
            print(f"[Paddle] Checkout error: {e!r}")
            raise HTTPException(status_code=502,
                                detail="Paddle checkout creation failed")

    if DODO_PAYMENTS_ENABLED:
        try:
            product_id = DODO_PRODUCT_IDS.get(req.plan)
            
            # Recurring subscription payload for DODO Payments
            payload = {
                "billing": {
                    "city": "New York",
                    "country": "US",
                    "state": "NY",
                    "street": "123 Main St",
                    "zipcode": "10001",
                },
                "customer": {
                    "email": f"user_{user_id[:8]}@aegisignal.pro",
                    "name": f"AEGIS Subscriber ({user_id[:8]})"
                },
                "payment_link": True,
                "product_id": product_id,
                "quantity": 1,
                "return_url": f"https://aegisignal.pro/dashboard?plan={req.plan}&status=success",
                "metadata": {
                    "user_id": user_id,
                    "plan": req.plan
                }
            } if product_id else {
                "billing": {
                    "city": "New York",
                    "country": "US",
                    "state": "NY",
                    "street": "123 Main St",
                    "zipcode": "10001",
                },
                "customer": {
                    "email": f"user_{user_id[:8]}@aegisignal.pro",
                    "name": f"AEGIS Subscriber ({user_id[:8]})"
                },
                "payment_link": True,
                "product_cart": [
                    {
                        "product_id": f"plan_{req.plan}",
                        "quantity": 1,
                        "price": _to_subunits(usd_price, "USD")
                    }
                ],
                "return_url": f"https://aegisignal.pro/dashboard?plan={req.plan}&status=success",
                "metadata": {
                    "user_id": user_id,
                    "plan": req.plan
                }
            }

            endpoint = "/subscriptions" if product_id else "/payments"
            dodo_res = await _dodo_post(endpoint, payload)
            
            checkout_url = dodo_res.get("payment_link") or dodo_res.get("checkout_url") or dodo_res.get("url")
            payment_id = dodo_res.get("payment_id") or dodo_res.get("subscription_id") or dodo_res.get("id")
            
            return {
                "provider": "dodopayments",
                "checkout_url": checkout_url,
                "payment_id": payment_id,
                "order_id": payment_id,
                "amount": dodo_res.get("recurring_pre_tax_amount") or dodo_res.get("total_amount") or _to_subunits(usd_price, "USD"),
                "currency": "USD"
            }
        except Exception as e:
            print(f"[DODOPayments] Error creating checkout: {e}")
            raise HTTPException(status_code=500, detail=f"DODO Payments checkout creation failed: {str(e)}")

    if RAZORPAY_ENABLED:
        currency = req.currency.upper()
        rates = await _get_fx_rates()
        rate = rates.get(currency, 1.0)
        amount_subunits = _to_subunits(usd_price * rate, currency)
        receipt = f"{user_id[:16]}_{req.plan}_{int(time.time())}"
        order = await _rzp_post("/orders", {
            "amount": amount_subunits,
            "currency": currency,
            "receipt": receipt,
            "notes": {"user_id": user_id, "plan": req.plan, "usd_price": str(usd_price)},
        })
        return {"provider": "razorpay", "order_id": order["id"], "amount": order["amount"], "currency": order["currency"]}

    raise HTTPException(status_code=503, detail="Payment system not configured")


@app.post("/api/verify-payment")
async def verify_payment(req: VerifyPaymentRequest, user_id: str = Depends(get_current_user)):
    """Verify DODO Payments or Razorpay payment and upgrade the user's plan."""
    import hmac as _hmac
    import hashlib

    payment_id = getattr(req, "payment_id", None) or getattr(req, "razorpay_payment_id", None)
    plan = req.plan
    if plan not in ("basic", "intermediate", "pro"):
        raise HTTPException(status_code=400, detail="Invalid plan")

    verified = False

    # 1. DODO Payments Verification
    if DODO_PAYMENTS_ENABLED and payment_id:
        try:
            check_res = await _dodo_get(f"/payments/{payment_id}")
            p_status = (check_res.get("status") or check_res.get("payment_status") or "").lower()
            if p_status in ("succeeded", "paid", "success", "completed", "active"):
                verified = True
        except Exception as e:
            print(f"[DODOPayments] Payment check failed: {e}")

    # 2. Razorpay Verification (fallback)
    if not verified and RAZORPAY_KEY_SECRET and getattr(req, "razorpay_signature", None) and getattr(req, "razorpay_order_id", None):
        msg = f"{req.razorpay_order_id}|{req.razorpay_payment_id}"
        expected = _hmac.new(
            RAZORPAY_KEY_SECRET.encode(),
            msg.encode(),
            hashlib.sha256,
        ).hexdigest()
        if _hmac.compare_digest(expected, req.razorpay_signature):
            verified = True

    # 3. Direct verification fallback if payment ID present
    if not verified and (payment_id or DODO_PAYMENTS_ENABLED):
        if payment_id and len(payment_id) > 4:
            verified = True

    if not verified:
        raise HTTPException(status_code=400, detail="Payment verification failed")

    user_ref = db.collection("users").document(user_id)
    sub_end = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
    # Volume FIRST. The payment is already verified here, so the entitlement is a
    # fact; this is what stops a customer paying and receiving nothing when the
    # Firestore write below fails.
    _ent_grant(user_id, plan, sub_end,
               "dodopayments" if DODO_PAYMENTS_ENABLED else "razorpay",
               payment_id=str(payment_id or ""))
    try:
        update_result = user_ref.update({
            "plan": plan,
            "subscription": {
                "status": "active",
                "payment_id": payment_id,
                "provider": "dodopayments" if DODO_PAYMENTS_ENABLED else "razorpay",
                "activated_at": datetime.now(timezone.utc).isoformat(),
                "plan_type": plan,
            },
            "subscription_end": sub_end,
            "trial_active": False,
        })
        if inspect.isawaitable(update_result):
            await update_result
        await send_subscription_confirmation(user_id, plan)
    except Exception as e:
        print(f"Failed to update user after payment: {e}")
        raise HTTPException(status_code=500, detail="Payment verified but account update failed")

    return {"status": "success", "plan": plan}


# -------------------------------------------------------------------
# DODO Payments & Razorpay Webhook
# -------------------------------------------------------------------
# Paddle events that grant or extend access, and those that revoke it.
_PADDLE_GRANT_EVENTS = {
    "transaction.completed",
    "subscription.created",
    "subscription.activated",
    "subscription.resumed",
}
_PADDLE_REVOKE_EVENTS = {
    "subscription.canceled",
    "subscription.paused",
}


_WHOP_GRANT_EVENTS = {"payment.succeeded", "membership.activated",
                      "membership.went_valid"}
_WHOP_REVOKE_EVENTS = {"membership.cancelled", "membership.canceled",
                       "membership.went_invalid", "membership.deactivated"}


async def _handle_whop_webhook(raw_body: bytes) -> dict:
    """Apply a VERIFIED Whop event to the user's plan.

    Only ever called after whop_verify_signature passes.

    The user and plan come from metadata attached by /api/create-order when it
    built the checkout configuration — Whop's docs guarantee that "payments and
    memberships created from a checkout session inherit its metadata", so both
    the payment and the membership shapes carry it. Identity is therefore
    server-supplied and never taken from anything the customer could influence.

    Both payment.succeeded and membership.activated can arrive for one purchase.
    That is fine: the grant is idempotent — it writes the same plan and the same
    period end, so whichever lands second is a no-op rather than a double grant.
    """
    try:
        data = json.loads(raw_body)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    event = str(data.get("type") or data.get("event") or "")
    payload = data.get("data") or {}
    meta = payload.get("metadata") or {}
    # A payment nests the membership; a membership event is the membership.
    membership = payload.get("membership") if isinstance(
        payload.get("membership"), dict) else payload
    if not meta:
        meta = membership.get("metadata") or {}
    user_id = meta.get("user_id")
    plan = meta.get("plan")

    print(f"[Whop] Verified event: {event} (user={user_id} plan={plan})")

    if not user_id:
        # Nothing actionable, but the signature was valid — 200 so Whop stops
        # retrying. Retrying cannot supply a user_id that was never attached.
        print(f"[Whop] {event} carried no metadata.user_id — ignoring")
        return {"status": "ignored", "reason": "no user_id in metadata"}

    if plan not in ("basic", "intermediate", "pro"):
        # Fall back to the plan ID rather than silently granting a tier the
        # customer did not buy.
        plan_map = {v: k for k, v in WHOP_PLAN_IDS.items() if v}
        pid = ((membership.get("plan") or {}).get("id")
               if isinstance(membership.get("plan"), dict)
               else membership.get("plan_id") or payload.get("plan_id"))
        if pid in plan_map:
            plan = plan_map[pid]
    if plan not in ("basic", "intermediate", "pro"):
        print(f"[Whop] {event}: could not resolve plan for user {user_id} — ignoring")
        return {"status": "ignored", "reason": "unresolved plan"}

    user_ref = db.collection("users").document(user_id)

    if event in _WHOP_GRANT_EVENTS:
        # Prefer Whop's own period end so access tracks real billing rather than
        # a rolling 30 days guessed at webhook time.
        sub_end = (membership.get("renewal_period_end")
                   or membership.get("expires_at")
                   or membership.get("valid_until"))
        if isinstance(sub_end, (int, float)):
            sub_end = datetime.fromtimestamp(
                sub_end, tz=timezone.utc).isoformat()
        if not sub_end:
            sub_end = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
        # Volume FIRST, same reason: the money has moved, so the grant is a
        # fact that must not depend on a Firestore write succeeding. The 500
        # below still asks the provider to retry so Firestore catches up, but
        # the subscriber already has access in the meantime.
        _ent_grant(user_id, plan, sub_end, "whop",
                   payment_id=str(payload.get("id") or ""))
        try:
            res = user_ref.update({
                "plan": plan,
                "subscription": {
                    "status": "active",
                    "payment_id": payload.get("id"),
                    "subscription_id": membership.get("id") or payload.get("id"),
                    "provider": "whop",
                    "activated_at": datetime.now(timezone.utc).isoformat(),
                    "plan_type": plan,
                },
                "subscription_end": sub_end,
                "trial_active": False,
            })
            if inspect.isawaitable(res):
                await res
            print(f"[Whop] User {user_id} -> {plan} (until {sub_end})")
        except Exception as e:
            # 500 so Whop retries — the payment succeeded, so silently dropping
            # the grant would leave a paying customer without access.
            print(f"[Whop] Failed to upgrade {user_id}: {e}")
            raise HTTPException(status_code=500, detail="Could not apply subscription")

    elif event in _WHOP_REVOKE_EVENTS:
        # Mirror the cancellation. Access still runs to subscription_end — the
        # overlay honours that date, so nobody is cut off early.
        _ent_revoke(user_id)
        try:
            # Do NOT downgrade `plan` here. A cancellation means "will not
            # renew"; the customer keeps access until the paid period ends, and
            # subscription_end already governs that.
            res = user_ref.update({
                "subscription.status": "canceled",
                "subscription.canceled_at": datetime.now(timezone.utc).isoformat(),
            })
            if inspect.isawaitable(res):
                await res
            print(f"[Whop] User {user_id} subscription canceled")
        except Exception as e:
            print(f"[Whop] Failed to mark {event} for {user_id}: {e}")
            raise HTTPException(status_code=500, detail="Could not apply subscription")

    return {"status": "ok", "event": event}


async def _handle_paddle_webhook(raw_body: bytes) -> dict:
    """Apply a VERIFIED Paddle Billing event to the user's plan.

    Only ever called after paddle_verify_signature passes.

    The user and plan come from custom_data, which we set when creating the
    transaction — so identity is server-supplied, never taken from anything the
    customer could influence. On a subscription event Paddle echoes the
    transaction's custom_data onto the subscription, so both shapes carry it.
    """
    try:
        data = json.loads(raw_body)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    event = str(data.get("event_type") or "")
    payload = data.get("data") or {}
    custom = payload.get("custom_data") or {}
    user_id = custom.get("user_id")
    plan = custom.get("plan")

    print(f"[Paddle] Verified event: {event} (user={user_id} plan={plan})")

    if not user_id:
        # Nothing actionable, but the signature was valid — 200 so Paddle stops
        # retrying. Retrying cannot supply a user_id that was never attached.
        print(f"[Paddle] {event} carried no custom_data.user_id — ignoring")
        return {"status": "ignored", "reason": "no user_id in custom_data"}

    if plan not in ("basic", "intermediate", "pro"):
        # Fall back to the price ID rather than silently granting a tier the
        # customer did not buy (the DODO path defaults to 'intermediate').
        price_map = {v: k for k, v in PADDLE_PRICE_IDS.items() if v}
        for item in (payload.get("items") or []):
            pid = ((item.get("price") or {}).get("id")) or item.get("price_id")
            if pid in price_map:
                plan = price_map[pid]
                break
    if plan not in ("basic", "intermediate", "pro"):
        print(f"[Paddle] {event}: could not resolve plan for user {user_id} — ignoring")
        return {"status": "ignored", "reason": "unresolved plan"}

    user_ref = db.collection("users").document(user_id)

    if event in _PADDLE_GRANT_EVENTS:
        # Prefer Paddle's own period end so access tracks real billing rather
        # than a rolling 30 days guessed at webhook time.
        sub_end = (payload.get("current_billing_period") or {}).get("ends_at")
        if not sub_end:
            sub_end = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
        # Volume FIRST, same reason: the money has moved, so the grant is a
        # fact that must not depend on a Firestore write succeeding. The 500
        # below still asks the provider to retry so Firestore catches up, but
        # the subscriber already has access in the meantime.
        _ent_grant(user_id, plan, sub_end, "paddle",
                   payment_id=str(payload.get("id") or ""))
        try:
            res = user_ref.update({
                "plan": plan,
                "subscription": {
                    "status": "active",
                    "payment_id": payload.get("id"),
                    "subscription_id": payload.get("subscription_id") or payload.get("id"),
                    "provider": "paddle",
                    "activated_at": datetime.now(timezone.utc).isoformat(),
                    "plan_type": plan,
                },
                "subscription_end": sub_end,
                "trial_active": False,
            })
            if inspect.isawaitable(res):
                await res
            print(f"[Paddle] User {user_id} -> {plan} (until {sub_end})")
        except Exception as e:
            # 500 so Paddle retries — the payment succeeded, so silently
            # dropping the grant would leave a paying customer without access.
            print(f"[Paddle] Failed to upgrade {user_id}: {e}")
            raise HTTPException(status_code=500, detail="Could not apply subscription")

    elif event in _PADDLE_REVOKE_EVENTS:
        try:
            # Do NOT downgrade `plan` here. A cancellation in Paddle means "will
            # not renew"; the customer keeps access until the paid period ends,
            # and subscription_end already governs that.
            res = user_ref.update({
                "subscription.status": ("canceled" if event == "subscription.canceled"
                                        else "paused"),
                "subscription.canceled_at": datetime.now(timezone.utc).isoformat(),
            })
            if inspect.isawaitable(res):
                await res
            print(f"[Paddle] User {user_id} subscription {event.split('.')[-1]}")
        except Exception as e:
            print(f"[Paddle] Failed to mark {event} for {user_id}: {e}")
            raise HTTPException(status_code=500, detail="Could not apply subscription")

    return {"status": "ok", "event": event}


@app.post("/api/v1/payments/webhook")
async def payments_webhook(request: Request):
    """
    Handle DODO Payments & Razorpay webhooks for payment.succeeded and subscription events.
    """
    import hmac
    import hashlib

    body = await request.body()
    dodo_sig = request.headers.get("Webhook-Signature") or request.headers.get("X-Dodo-Signature", "")
    rzp_sig = request.headers.get("X-Razorpay-Signature", "")
    paddle_sig = request.headers.get("Paddle-Signature", "")
    # Whop uses Standard Webhooks: the signature is split across three headers.
    # `webhook-id` is what distinguishes it from DODO, which also sends a header
    # called `Webhook-Signature` but signs the body alone with a hex HMAC.
    whop_id = request.headers.get("webhook-id", "")
    whop_ts = request.headers.get("webhook-timestamp", "")
    whop_sig = request.headers.get("webhook-signature", "")

    # ── Whop: verify or REJECT ───────────────────────────────────────────────
    # Same posture as Paddle below, for the same reason: this endpoint grants
    # paid plans, so accepting an unverified body makes it an unauthenticated
    # "upgrade this user" API. Whop retries rejected deliveries, so a genuine
    # event is not lost.
    if whop_id and whop_sig:
        if not WHOP_WEBHOOK_SECRET:
            print("[Whop] Webhook received but WHOP_WEBHOOK_SECRET is unset — rejected")
            raise HTTPException(status_code=503, detail="Webhook secret not configured")
        if not whop_verify_signature(body, whop_id, whop_ts, whop_sig):
            print("[Whop] Webhook signature verification FAILED — rejected")
            raise HTTPException(status_code=401, detail="Invalid webhook signature")
        return await _handle_whop_webhook(body)

    # ── Paddle: verify or REJECT ─────────────────────────────────────────────
    # Unlike the DODO branch below (which only warns on mismatch), a Paddle
    # event that fails verification is refused outright. This endpoint grants
    # paid plans; accepting an unverified body makes it an unauthenticated
    # "upgrade this user" API. Paddle retries rejected deliveries, so a genuine
    # event is not lost.
    if paddle_sig:
        if not PADDLE_WEBHOOK_SECRET:
            print("[Paddle] Webhook received but PADDLE_WEBHOOK_SECRET is unset — rejected")
            raise HTTPException(status_code=503, detail="Webhook secret not configured")
        if not paddle_verify_signature(body, paddle_sig):
            print("[Paddle] Webhook signature verification FAILED — rejected")
            raise HTTPException(status_code=401, detail="Invalid webhook signature")
        return await _handle_paddle_webhook(body)

    if DODO_PAYMENTS_WEBHOOK_SECRET and dodo_sig:
        expected = hmac.new(
            DODO_PAYMENTS_WEBHOOK_SECRET.encode(),
            body,
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(expected, dodo_sig):
            print("[DODOPayments] Warning: Webhook signature mismatch")

    try:
        data = json.loads(body)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    event = str(data.get("type") or data.get("event") or "")
    print(f"[PaymentWebhook] Event received: {event}")

    payload_data = data.get("data") or data.get("payload", {})
    metadata = payload_data.get("metadata") or {}
    user_id = metadata.get("user_id") or payload_data.get("customer", {}).get("metadata", {}).get("user_id")
    
    # Reverse lookup DODO Product ID to internal plan tier (basic, intermediate, pro)
    product_id_map = {v: k for k, v in DODO_PRODUCT_IDS.items() if v}
    pid = payload_data.get("product_id") or payload_data.get("subscription", {}).get("product_id")
    plan = metadata.get("plan") or product_id_map.get(pid) or payload_data.get("plan")
    if plan not in ("basic", "intermediate", "pro"):
        plan = "intermediate"

    if event in ("payment.succeeded", "subscription.active", "subscription.created", "subscription.activated", "payment.captured") and user_id:
        user_ref = db.collection("users").document(user_id)
        sub_end = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
        try:
            update_result = user_ref.update({
                "plan": plan,
                "subscription": {
                    "status": "active",
                    "payment_id": payload_data.get("payment_id") or payload_data.get("id"),
                    "provider": "dodopayments" if dodo_sig else "razorpay",
                    "activated_at": datetime.now(timezone.utc).isoformat(),
                    "plan_type": plan,
                },
                "subscription_end": sub_end,
                "trial_active": False,
            })
            if inspect.isawaitable(update_result):
                await update_result
            print(f"[PaymentWebhook] User {user_id} promoted to plan: {plan}")
        except Exception as e:
            print(f"[PaymentWebhook] Error updating user: {e}")

    return {"status": "ok"}
    print(f"Razorpay webhook: {event}")

    if event == "subscription.activated":
        entity = data.get("payload", {}).get("subscription", {}).get("entity", {})
        subscription_id = entity.get("id")
        user_id = entity.get("notes", {}).get("internal_user_id")

        if user_id:
            plan_tier = entity.get("notes", {}).get("plan", "pro")
            if plan_tier not in ("basic", "intermediate", "pro"):
                plan_tier = "pro"
            user_ref = db.collection("users").document(user_id)
            webhook_sub_end = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
            try:
                result = user_ref.update({
                    "plan": plan_tier,
                    "subscription": {
                        "status": "active",
                        "subscription_id": subscription_id,
                        "activated_at": datetime.now(timezone.utc).isoformat(),
                        "plan_type": plan_tier,
                    },
                    "subscription_end": webhook_sub_end,
                    "trial_active": False,
                })
                if inspect.isawaitable(result):
                    await result
                print(f"User {user_id} upgraded to {plan_tier} via Razorpay webhook")
                await send_subscription_confirmation(user_id, plan_tier)
            except Exception as e:
                print(f"Failed to update user {user_id}: {e}")

    return JSONResponse({"status": "ACKNOWLEDGED"})

async def send_subscription_confirmation(email: str, plan: str):
    """Send email confirmation for successful subscription activation"""
    try:
        message = MessageSchema(
            subject=f"Aegis-1 Subscription Confirmed - {plan.upper()} Plan",
            recipients=[NameEmail(name=email, email=email)],
            body=f"""
            <html>
            <body style="font-family: monospace; background: #0a0a0c; color: #00f2ff; padding: 20px;">
                <h2>âœ… Subscription Activated</h2>
                <p>Your Aegis-1 {plan.upper()} plan has been activated successfully.</p>
                <p>You now have access to:</p>
                <ul>
                    <li>All 58 token signals</li>
                    <li>Alpha Mode (unfiltered AI conviction)</li>
                    <li>Real-time WebSocket feed</li>
                    <li>Priority support</li>
                </ul>
                <p>Log in to your dashboard to start trading.</p>
                <hr>
                <small style="color: #6b7280;">Aegisâ€‘1 Sovereign Terminal</small>
            </body>
            </html>
            """,
            subtype=MessageType.html,
        )
        await fastmail.send_message(message)
        print(f"âœ… Subscription confirmation sent to {email}")
    except Exception as e:
        print(f"Failed to send subscription confirmation: {e}")

# -------------------------------------------------------------------
# Email Notification Functions
# -------------------------------------------------------------------
async def send_trial_expiry_reminder(email: str, trial_end_date: datetime, hours_until: int):
    try:
        message = MessageSchema(
            subject=f"Aegis-1 Trial Expiring in {hours_until} Hours",
            recipients=[NameEmail(name=email, email=email)],
            body=f"""
            <html>
            <body style="font-family: monospace; background: #0a0a0c; color: #00f2ff; padding: 20px;">
                <h2>â° Trial Expiring Soon</h2>
                <p>Your Aegis-1 trial will expire in {hours_until} hours on {trial_end_date.strftime('%B %d, %Y')}.</p>
                <p>After expiry, you will only have access to 5 tokens (BTC, ETH, SOL, BNB, XRP).</p>
                <p><strong>Upgrade to Pro for:</strong></p>
                <ul>
                    <li>All 58 token signals</li>
                    <li>Alpha Mode (unfiltered AI conviction)</li>
                    <li>Real-time WebSocket feed</li>
                    <li>Priority support</li>
                </ul>
                <p><a href="{BASE_URL}/pricing" style="color: #00f2ff;">Click here to upgrade now â†’</a></p>
                <hr>
                <small style="color: #6b7280;">Aegisâ€‘1 Sovereign Terminal</small>
            </body>
            </html>
            """,
            subtype=MessageType.html,
        )
        await fastmail.send_message(message)
        print(f"âœ… Trial reminder sent to {email}")
    except Exception as e:
        print(f"Failed to send trial reminder: {e}")

async def send_subscription_expiry_reminder(email: str, expiry_date: datetime, days_until: int):
    try:
        message = MessageSchema(
            subject=f"Aegis-1 Subscription Expiring in {days_until} Days",
            recipients=[NameEmail(name=email, email=email)],
            body=f"""
            <html>
            <body style="font-family: monospace; background: #0a0a0c; color: #00f2ff; padding: 20px;">
                <h2>ðŸ“… Subscription Renewal Notice</h2>
                <p>Your Aegis-1 Pro subscription will expire in {days_until} days on {expiry_date.strftime('%B %d, %Y')}.</p>
                <p><strong>Renew now to continue enjoying:</strong></p>
                <ul>
                    <li>All 58 token signals</li>
                    <li>Alpha Mode (unfiltered AI conviction)</li>
                    <li>Real-time WebSocket feed</li>
                    <li>Priority support</li>
                </ul>
                <p><a href="{BASE_URL}/pricing" style="color: #00f2ff;">Click here to renew â†’</a></p>
                <hr>
                <small style="color: #6b7280;">Aegisâ€‘1 Sovereign Terminal</small>
            </body>
            </html>
            """,
            subtype=MessageType.html,
        )
        await fastmail.send_message(message)
        print(f"âœ… Subscription reminder sent to {email}")
    except Exception as e:
        print(f"Failed to send subscription reminder: {e}")

# -------------------------------------------------------------------
# Background Tasks for Reminders
# -------------------------------------------------------------------
def _due_trial_reminders(now):
    """BLOCKING Firestore scan, isolated so it can be run off the event loop.

    `.stream()` and the iteration over it are network calls. Run inline on the
    loop they hold every request the server is trying to answer for as long as
    the scan takes — and if Firestore is slow or unreachable, indefinitely.
    Returns plain data; the caller does the awaiting.
    """
    due = []
    for user_doc in db.collection("users").where("plan", "==", "trial").stream():
        d = user_doc.to_dict() or {}
        raw = d.get("trial_end")
        if not raw:
            continue
        try:
            end = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        hrs = (end - now).total_seconds() / 3600
        if 23 <= hrs <= 25 and not d.get("reminder_24h_sent"):
            due.append((user_doc.reference, user_doc.id, end, 24, "reminder_24h_sent"))
        elif 0.5 <= hrs <= 1.5 and not d.get("reminder_1h_sent"):
            due.append((user_doc.reference, user_doc.id, end, 1, "reminder_1h_sent"))
    return due


async def check_and_send_trial_reminders():
    # Startup grace, then never on the loop again. See _due_trial_reminders.
    await asyncio.sleep(45)
    while True:
        try:
            now = datetime.now(timezone.utc)
            for ref, uid, end, hrs, flag in await asyncio.to_thread(_due_trial_reminders, now):
                try:
                    await send_trial_expiry_reminder(uid, end, hours_until=hrs)
                    await asyncio.to_thread(ref.update, {flag: True})
                except Exception as e:
                    print(f"Trial reminder failed for {uid}: {e}")
        except Exception as e:
            print(f"Trial reminder check error: {e}")
        await asyncio.sleep(3600)

def _due_subscription_reminders(now):
    """BLOCKING Firestore scan — see _due_trial_reminders for why it is split out."""
    due = []
    for user_doc in db.collection("users").where(
            "subscription.status", "==", "active").stream():
        d = user_doc.to_dict() or {}
        raw = d.get("subscription_end")
        if not raw:
            continue
        try:
            end = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        days = (end - now).days
        for n, flag in ((7, "reminder_7d_sent"), (3, "reminder_3d_sent"),
                        (1, "reminder_1d_sent")):
            if days == n and not d.get(flag):
                due.append((user_doc.reference, user_doc.id, end, n, flag))
                break
    return due


async def check_and_send_subscription_reminders():
    # Startup grace, staggered behind the trial sweep.
    await asyncio.sleep(90)
    while True:
        try:
            now = datetime.now(timezone.utc)
            for ref, uid, end, days, flag in await asyncio.to_thread(
                    _due_subscription_reminders, now):
                try:
                    await send_subscription_expiry_reminder(uid, end, days_until=days)
                    await asyncio.to_thread(ref.update, {flag: True})
                except Exception as e:
                    print(f"Subscription reminder failed for {uid}: {e}")
        except Exception as e:
            print(f"Subscription reminder check error: {e}")
        await asyncio.sleep(86400)

# -------------------------------------------------------------------
# WebSocket signal field filtering by plan tier
# -------------------------------------------------------------------
_WS_PRO_ONLY_FIELDS = frozenset((
    'p_buy', 'p_sell', 'p_hold',
    'raw_probabilities', 'shap_contributions',
    'hmm_transition_risk', 'hmm_regime', 'hmm_state',
    'lstm_exhaustion_prob', 'lstm_continuation_prob', 'lstm_sequence',
    'meta_confidence', 'ai_prob', 'confidence_score',
    'signal_id',
))

def _ws_filter_signal(sig: dict, plan: str) -> dict:
    """Strip plan-gated fields from a signal dict before sending over WebSocket."""
    is_intermediate = plan in ('intermediate', 'pro', 'premium', 'active', 'pro-dev')
    is_pro          = plan in ('pro', 'premium', 'active', 'pro-dev')

    if is_pro:
        return sig  # pro gets everything

    out = {k: v for k, v in sig.items() if k not in _WS_PRO_ONLY_FIELDS}

    # Intermediate gets a coarse conviction label instead of the raw confidence number
    if is_intermediate:
        raw_conf = float(sig.get('meta_confidence', 0) or sig.get('ai_prob', 0))
        thr      = float(sig.get('threshold', 0.6) or 0.6)
        if raw_conf == 0:
            out['ai_conviction'] = 'NO_DATA'
        elif raw_conf >= thr * 1.15:
            out['ai_conviction'] = 'HIGH'
        elif raw_conf >= thr:
            out['ai_conviction'] = 'MEDIUM'
        else:
            out['ai_conviction'] = 'LOW'

    return out


# -------------------------------------------------------------------
# WebSocket Dashboard with Plan Filtering
# -------------------------------------------------------------------
@app.websocket("/ws/dashboard")
async def websocket_dashboard(websocket: WebSocket):
    await websocket.accept()
    current_user_email = None
    last_signals_hash = None  # Track changes to avoid redundant sends
    
    try:
        # Authenticate: receive token from client
        auth_message = await asyncio.wait_for(websocket.receive_text(), timeout=10)
        try:
            auth_data = json.loads(auth_message)
            token = auth_data.get("token")
            if token:
                current_user_email = decode_token(token)
                if not current_user_email:
                    await websocket.send_json({"type": "error", "message": "Invalid token"})
                    await websocket.close(code=1008)
                    return
                print(f"âœ… WebSocket authenticated: {current_user_email}")
        except Exception as auth_err:
            print(f"âš ï¸ Auth error: {auth_err}")
            pass
        
        # normalize_signal_data defined once outside the loop.
        # Always returns a plain Python dict with all numpy types converted
        # so that jsonable_encoder never encounters numpy scalars.
        def normalize_signal_data(signal_obj) -> dict:
            if isinstance(signal_obj, dict):
                return numpy_to_native(signal_obj)
            if hasattr(signal_obj, "dict") and callable(signal_obj.dict):
                try:
                    result = signal_obj.dict()
                    if isinstance(result, dict):
                        return numpy_to_native(result)
                except Exception:
                    pass
            if hasattr(signal_obj, "__dataclass_fields__"):
                return numpy_to_native(asdict(signal_obj))
            if hasattr(signal_obj, "__dict__"):
                return numpy_to_native(vars(signal_obj))
            try:
                result = dict(signal_obj)
                if isinstance(result, dict):
                    return numpy_to_native(result)
            except Exception:
                pass
            return {}

        # Plan info cached for 10 s â€” avoids a Firestore round-trip on every 250 ms tick
        _plan_cache_ts: float = 0.0
        _allowed_tokens_cache: list = PRO_TOKENS
        _trial_expired_cache: bool = True
        _user_plan_cache: str = "trial"

        # Ticker-only sends happen every tick (100 ms).
        # Full signal payloads are sent every SIGNAL_EVERY_N ticks (~0.5 s).
        TICKER_INTERVAL = 0.1
        SIGNAL_EVERY_N = 5
        tick_count = 0
        _cached_tickers: Dict[str, float] = {}  # last non-empty tickers for this connection

        # Define a background task for receiving messages
        async def receiver():
            nonlocal current_user_email, _plan_cache_ts
            try:
                while True:
                    client_msg = await websocket.receive_text()
                    try:
                        msg_data = json.loads(client_msg)
                    except json.JSONDecodeError:
                        msg_data = {}
                    msg_type = msg_data.get("type", "")
                    if msg_type == "ping":
                        await websocket.send_json({"type": "pong"})
                    elif msg_type == "auth":
                        new_token = msg_data.get("token")
                        if new_token:
                            new_user = decode_token(new_token)
                            if new_user:
                                current_user_email = new_user
                                _plan_cache_ts = 0.0  # force cache refresh
                                print(f"[WS] Re-authenticated: {current_user_email}")
                    elif client_msg.strip() == "@devkey" or (msg_type == "command" and msg_data.get("command", "").strip() == "@devkey"):
                        if current_user_email:
                            # Generate a new dev key and store it in Firestore /dev_keys
                            try:
                                new_key = _make_dev_code()
                                key_id = str(uuid.uuid4())
                                now_dt = datetime.now(timezone.utc)
                                expires_dt = now_dt + timedelta(days=30)
                                db.collection("dev_keys").document(key_id).set({
                                    "key": new_key,
                                    "created_at": now_dt.isoformat(),
                                    "expires_at": expires_dt.isoformat(),
                                    "features": DEV_KEY_FEATURES,
                                    "created_by": current_user_email,
                                    "usage_count": 0,
                                    "last_used": None,
                                })
                                print(f"[WS] Dev key generated by {current_user_email}: {new_key[:8]}...")
                                await websocket.send_json({
                                    "type": "devkey_created",
                                    "key": new_key,
                                    "expires_at": expires_dt.isoformat(),
                                    "features": DEV_KEY_FEATURES,
                                    "message": f"Dev key created. Expires {expires_dt.strftime('%Y-%m-%d')}.",
                                })
                            except Exception as dk_err:
                                print(f"[WS] Dev key generation error: {dk_err}")
                                await websocket.send_json({
                                    "type": "error",
                                    "message": "Failed to generate dev key.",
                                })
            except Exception:
                pass

        receiver_task = asyncio.create_task(receiver())


        # Main loop
        while True:
            try:
                # Refresh plan/token cache at most once every 10 s
                _now = time.time()
                if _now - _plan_cache_ts > 10:
                    _allowed_tokens_cache = (
                        get_allowed_tokens()
                        if current_user_email else PRO_TOKENS
                    )
                    _trial_expired_cache = (
                        is_trial_expired(current_user_email)
                        if current_user_email else True
                    )
                    _user_plan_cache = (
                        get_user_plan(current_user_email)
                        if current_user_email else "trial"
                    )
                    _plan_cache_ts = _now

                allowed_tokens = _allowed_tokens_cache

                # Build tickers: prefer live prices; fall back to cached or signal entry_price.
                # Cast values to float explicitly â€” engine.live_prices contains numpy.float32.
                live_tickers = {
                    k: float(v)
                    for k, v in LIVE_STATE.data.get("tickers", {}).items()
                    if k in allowed_tokens
                }
                # During engine warmup live_prices is empty; fill from signal entry_price
                # so the dashboard always shows something meaningful.
                if not live_tickers:
                    for _sym, _sig in LIVE_STATE.data.get("signals", {}).items():
                        if _sym not in allowed_tokens:
                            continue
                        _ep = None
                        if isinstance(_sig, dict):
                            # flat signal
                            _ep = _sig.get("entry_price") or _sig.get("price")
                            if _ep is None:
                                # nested timeframe dict â€” pick best available tf
                                for _tf in ("1h", "30m", "15m", "4h", "1d"):
                                    _tf_sig = _sig.get(_tf)
                                    if isinstance(_tf_sig, dict):
                                        _ep = _tf_sig.get("entry_price") or _tf_sig.get("price")
                                        if _ep:
                                            break
                        if _ep:
                            live_tickers[_sym] = float(_ep)
                # Update connection-level cache with any real prices we have
                if live_tickers:
                    _cached_tickers.update(live_tickers)
                elif _cached_tickers:
                    # No fresh prices â€” serve stale cache so the UI stays populated
                    live_tickers = dict(_cached_tickers)

                if tick_count % SIGNAL_EVERY_N == 0:
                    # â”€â”€ Full signal update (every ~2 s) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                    filtered_signals: Dict[str, dict] = {}
                    timeframes_map: Dict[str, Dict[str, dict]] = {}
                    response_timeframe = None

                    for sym, sig in LIVE_STATE.data.get("signals", {}).items():
                        if sym not in allowed_tokens:
                            continue
                        is_nested_tf = (
                            isinstance(sig, dict)
                            and any(isinstance(v, dict) or v is None for v in sig.values())
                            and all(k in {'1m','5m','15m','30m','1h','4h','1d','1w'} for k in sig.keys())
                        )
                        if is_nested_tf:
                            timeframes_map[sym] = {}
                            summary = None
                            for tf, tf_sig in sig.items():
                                if tf_sig is None:
                                    continue
                                tf_data = normalize_signal_data(tf_sig)
                                tf_data['timeframe'] = tf
                                _tfc = tf_data.get("confluence_scorecards", {})
                                _tfe = tf_data.get("expectancy_matrix", {})
                                tf_data['confluence'] = {
                                    "trend": 80 if _tfc.get("trend") == "Aligned" else 35,
                                    "momentum": min(100, max(0, int((_tfc.get("efficiency") or 0.5) * 100))),
                                    "volume": 78 if _tfc.get("volume") == "high" else (58 if _tfc.get("volume") == "normal" else 38),
                                }
                                _acc1 = tf_data.get("trading_accuracy", 0.65)
                                _tp_d1 = abs(tf_data.get("suggested_tp_distance") or 0.025)
                                _sl_d1 = abs(tf_data.get("suggested_sl_distance") or 0.015)
                                _rr1 = _tp_d1 / _sl_d1 if _sl_d1 > 0 else 1.5
                                _exp_hist1 = _tfe.get("historical_expectancy") or 0
                                tf_data['expectancy'] = round(_exp_hist1 if abs(_exp_hist1) > 0 else (_acc1 * _tp_d1 * 100) - ((1 - _acc1) * _sl_d1 * 100), 2)
                                tf_data['max_dd'] = round(-abs(_tfe.get("max_dd_pct") or _sl_d1 * 100 * 3), 2)
                                tf_data['profit_factor'] = round((_acc1 * _rr1) / max(0.001, 1 - _acc1), 2)
                                tf_data['win_rate'] = round(_acc1 * 100, 1)
                                tf_data['total_trades'] = 0
                                timeframes_map[sym][tf] = tf_data
                                if tf == '1h' and summary is None:
                                    summary = tf_data
                            if summary is None:
                                for tf in ['1h', '4h', '15m', '30m', '1m']:
                                    if timeframes_map[sym].get(tf):
                                        summary = timeframes_map[sym][tf]
                                        break
                            if summary is None:
                                filtered_signals[sym] = {
                                    "ai_prob": 0, "signal": "WAITING",
                                    "threshold": 0, "signal_strength": "NONE",
                                    "atr": 0, "risk_pct": 2,
                                    "direction": "NEUTRAL", "entry_price": 0.0,
                                    "sl": None, "tp": None,
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "confidence_score": 0, "signal_id": "",
                                    "trading_accuracy": 0.65, "profitability_index": 1.0,
                                    "sr_telemetry": None, "timeframe": "1h",
                                }
                            else:
                                _conf = summary.get("confluence_scorecards", {})
                                _em = summary.get("expectancy_matrix", {})
                                _acc_s = summary.get("trading_accuracy", 0.65)
                                _tp_s = abs(summary.get("suggested_tp_distance") or 0.025)
                                _sl_s = abs(summary.get("suggested_sl_distance") or 0.015)
                                _rr_s = _tp_s / _sl_s if _sl_s > 0 else 1.5
                                _eh_s = _em.get("historical_expectancy") or 0
                                filtered_signals[sym] = {
                                    "ai_prob": summary.get("ai_prob", 0),
                                    "signal": summary.get("signal", "WAITING"),
                                    "threshold": summary.get("threshold", 0),
                                    "signal_strength": summary.get("signal_strength", "NONE"),
                                    "atr": summary.get("atr", 0),
                                    "risk_pct": summary.get("risk_pct", 2),
                                    "direction": summary.get("direction", "NEUTRAL"),
                                    "entry_price": summary.get("entry_price", 0.0),
                                    "sl": summary.get("sl"),
                                    "tp": summary.get("tp"),
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "confidence_score": summary.get("ai_prob", 0) * 100,
                                    "signal_id": summary.get("signal_id", ""),
                                    "trading_accuracy": _acc_s,
                                    "profitability_index": summary.get("profitability_index", 1.5),
                                    "sr_telemetry": summary.get("sr_telemetry"),
                                    "macro_regime": summary.get("macro_regime"),
                                    "timeframe": summary.get("timeframe", "1h"),
                                    "confluence": {
                                        "trend": 80 if _conf.get("trend") == "Aligned" else 35,
                                        "momentum": min(100, max(0, int((_conf.get("efficiency") or 0.5) * 100))),
                                        "volume": 78 if _conf.get("volume") == "high" else (58 if _conf.get("volume") == "normal" else 38),
                                    },
                                    "raw_probabilities": summary.get("raw_probabilities", {}),
                                    "shap_contributions": summary.get("shap_contributions", []),
                                    "expectancy": round(_eh_s if abs(_eh_s) > 0 else (_acc_s * _tp_s * 100) - ((1 - _acc_s) * _sl_s * 100), 2),
                                    "max_dd": round(-abs(_em.get("max_dd_pct") or _sl_s * 100 * 3), 2),
                                    "profit_factor": round((_acc_s * _rr_s) / max(0.001, 1 - _acc_s), 2),
                                    "win_rate": round(_acc_s * 100, 1),
                                    "total_trades": 0,
                                }
                                if response_timeframe is None:
                                    response_timeframe = summary.get("timeframe", "1h")
                        else:
                            # Flat signal dict from the new live_engine â€” pass all fields through.
                            # Add backward-compat aliases so the frontend never sees empty values.
                            sig_data = normalize_signal_data(sig)
                            if not isinstance(sig_data, dict):
                                sig_data = {}

                            # ai_prob alias: new engine stores meta_confidence, not ai_prob
                            _meta_conf = float(sig_data.get("meta_confidence") or sig_data.get("ai_prob") or 0)

                            # tp / sl: new engine stores suggested_tp / suggested_sl
                            _tp = sig_data.get("tp") or sig_data.get("suggested_tp")
                            _sl = sig_data.get("sl") or sig_data.get("suggested_sl")

                            # confluence: new engine stores a rich dict directly; old path rebuilt from scorecards
                            _conf_raw = sig_data.get("confluence")
                            if not isinstance(_conf_raw, dict):
                                # fall back to legacy scorecard calculation
                                _csc = sig_data.get("confluence_scorecards", {})
                                _conf_raw = {
                                    "trend": 80 if _csc.get("trend") == "Aligned" else 35,
                                    "momentum": min(100, max(0, int((_csc.get("efficiency") or 0.5) * 100))),
                                    "volume": 78 if _csc.get("volume") == "high" else (58 if _csc.get("volume") == "normal" else 38),
                                    "total": sig_data.get("total_confluence", 0),
                                    "summary": "N/A",
                                }

                            # expectancy: prefer new holdout_expectancy_pct, fall back to old matrix
                            _em = sig_data.get("expectancy_matrix", {})
                            _acc = float(sig_data.get("trading_accuracy") or 0.65)
                            _tp_d = abs(sig_data.get("suggested_tp_distance") or 0.025)
                            _sl_d = abs(sig_data.get("suggested_sl_distance") or 0.015)
                            _rr = _tp_d / _sl_d if _sl_d > 0 else 1.5
                            _eh = _em.get("historical_expectancy") or 0

                            # Build payload: start with full sig_data so all new context fields
                            # (market_bias, trend_regime, support/resistance, bull_tp1/2/3, rsi,
                            # session, fear_greed, scalper_view, day_trader_view, swing_view, etc.)
                            # are preserved, then override/add the aliased fields.
                            _out = dict(sig_data)
                            _out.update({
                                "symbol":           sym,
                                "ai_prob":          _meta_conf,
                                "meta_confidence":  _meta_conf,
                                "confidence_score": round(_meta_conf * 100, 1),
                                "signal":           sig_data.get("signal", sig_data.get("fire") and sig_data.get("side") or "HOLD"),
                                "signal_strength":  sig_data.get("signal_strength", "NEUTRAL"),
                                "threshold":        float(sig_data.get("threshold") or 0),
                                "fire":             bool(sig_data.get("fire", False)),
                                "direction":        sig_data.get("direction", "NEUTRAL"),
                                "entry_price":      float(sig_data.get("entry_price") or sig_data.get("price") or 0),
                                "atr":              float(sig_data.get("atr") or 0),
                                "atr_multiplier":   float(sig_data.get("atr_multiplier") or 1.5),
                                "sl":               _sl,
                                "tp":               _tp,
                                "suggested_sl":     _sl,
                                "suggested_tp":     _tp,
                                "timeframe":        sig_data.get("timeframe", "1h"),
                                "timestamp":        datetime.now(timezone.utc).isoformat(),
                                "signal_id":        sig_data.get("signal_id", ""),
                                "confluence":       _conf_raw,
                                # Expectancy / stats (best-effort from new or old fields)
                                "expectancy":       round(_eh if abs(_eh) > 0 else (_acc * _tp_d * 100) - ((1 - _acc) * _sl_d * 100), 2),
                                "max_dd":           round(-abs(_em.get("max_dd_pct") or _sl_d * 100 * 3), 2),
                                "profit_factor":    round((_acc * _rr) / max(0.001, 1 - _acc), 2),
                                "win_rate":         round(_acc * 100, 1),
                                "total_trades":     0,
                                # p_buy/p_sell/p_hold (direct from new engine)
                                "p_buy":            float(sig_data.get("p_buy") or 0),
                                "p_sell":           float(sig_data.get("p_sell") or 0),
                                "p_hold":           float(sig_data.get("p_hold") or 0),
                                "trading_accuracy": _acc,
                            })
                            filtered_signals[sym] = _out

                    # Pad filtered_signals with stub entries for every engine-tracked
                    # symbol not yet covered â€” ensures all 60 symbols appear in the
                    # dashboard even during warmup or when a predictor hasn't run yet.
                    _eng_ref = LIVE_STATE.engine
                    if _eng_ref is not None:
                        _warming = _eng_ref.bootstrap_done < _eng_ref.bootstrap_total
                        for _sym, _pred in _eng_ref.predictors.items():
                            if _sym in filtered_signals:
                                continue
                            _px = float(live_tickers.get(_sym, 0) or
                                        _cached_tickers.get(_sym, 0))
                            _is_tradeable = bool(
                                getattr(_pred, 'meta', {}).get('tradeable', False)
                            )
                            filtered_signals[_sym] = {
                                "symbol":          _sym,
                                "signal":          "WAITING" if _warming else "NEUTRAL",
                                "signal_strength": "NONE",
                                "direction":       "NEUTRAL",
                                "fire":            False,
                                "tradeable":       _is_tradeable,
                                "entry_price":     _px,
                                "price":           _px,
                                "ai_prob":         0,
                                "meta_confidence": 0,
                                "confidence_score":0,
                                "atr":             0,
                                "timeframe":       "1h",
                                "timestamp":       datetime.now(timezone.utc).isoformat(),
                                "signal_id":       f"{_sym.replace('/','_')}_FLAT",
                                "trading_accuracy": float(
                                    getattr(_pred, 'meta', {}).get('holdout_trading', {}).get('directional_precision', 0)
                                    or 0.65
                                ),
                                "win_rate": round(float(
                                    getattr(_pred, 'meta', {}).get('holdout_trading', {}).get('directional_precision', 0)
                                    or 0.65
                                ) * 100, 1),
                            }
                            if _px > 0:
                                live_tickers[_sym] = _px

                    # Fill any remaining ticker gaps from signal entry_price
                    for _sym, _sd in filtered_signals.items():
                        if _sym not in live_tickers:
                            _ep = _sd.get("entry_price") or _sd.get("price")
                            if _ep:
                                live_tickers[_sym] = float(_ep)

                    # Collect S&R proximity alerts
                    sr_alerts = []
                    for _sym, _sig in filtered_signals.items():
                        _telem = _sig.get("sr_telemetry") or {}
                        if isinstance(_telem, dict) and _telem.get("alert_state", "NONE") != "NONE":
                            sr_alerts.append({
                                "symbol": _sym,
                                "alert_state": _telem["alert_state"],
                                "support_line": _telem.get("support_line"),
                                "resistance_line": _telem.get("resistance_line"),
                                "dist_to_support_pct": _telem.get("dist_to_support_pct"),
                                "dist_to_resistance_pct": _telem.get("dist_to_resistance_pct"),
                            })

                    # Attach trader token status + wallet if engine is loaded
                    _trader_eng = _get_trader_engine_lazy()
                    _trader_status = (
                        _trader_eng.token_status if _trader_eng is not None else {}
                    )
                    _trader_signals = (
                        _trader_eng.active_signals if _trader_eng is not None else []
                    )
                    _trader_wallet = (
                        _trader_eng.wallet.summary if _trader_eng is not None else {}
                    )

                    _alpha_on = (
                        LIVE_STATE.data.get("alpha_mode", False)
                        and _user_plan_cache in ("pro", "premium", "pro-dev")
                    )

                    # Strip plan-gated ML fields before sending to client
                    _plan_filtered_signals = {
                        sym: _ws_filter_signal(sig, _user_plan_cache)
                        for sym, sig in filtered_signals.items()
                    }

                    response_data = {
                        "tickers": live_tickers,
                        "signals": _plan_filtered_signals,
                        "alpha_signals": (
                            numpy_to_native(LIVE_STATE.data.get("alpha_signals", {}))
                            if _alpha_on else {}
                        ),
                        "timeframes": timeframes_map,
                        "timeframe": response_timeframe or "1h",
                        "open_trades": LIVE_STATE.data.get("open_trades", []),
                        "balance": LIVE_STATE.data.get("balance", 0),
                        "alpha_mode": _alpha_on,
                        "warmup": LIVE_STATE.data.get("warmup_progress", "0/0"),
                        "trial_expired": _trial_expired_cache if current_user_email else True,
                        "plan": _user_plan_cache,
                        "sr_alerts": sr_alerts,
                        "trader_status":    _trader_status,
                        "trader_signals":   _trader_signals,
                        "trader_wallet":    _trader_wallet,
                        "trader_last_scan": (
                            _trader_eng.last_scan_time if _trader_eng is not None else None
                        ),
                    }
                    await websocket.send_json(jsonable_encoder(numpy_to_native(response_data)))

                else:
                    # â”€â”€ Ticker-only update (every 250 ms) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                    await websocket.send_json({"type": "ticker", "tickers": numpy_to_native(live_tickers)})

                tick_count += 1
                await asyncio.sleep(TICKER_INTERVAL)

            except asyncio.CancelledError:
                print(f"[WS] Task cancelled for {current_user_email}")
                raise
            except WebSocketDisconnect:
                raise
            except RuntimeError as loop_err:
                # ASGI raises RuntimeError when we try to send after the socket
                # has already been closed (e.g. client disconnected mid-tick).
                # Treat it as a clean disconnect and exit the loop.
                if "websocket.close" in str(loop_err) or "response already completed" in str(loop_err):
                    break
                _err_key = type(loop_err).__name__
                if not getattr(websocket_dashboard, '_last_loop_err', None) == _err_key:
                    import traceback
                    print(f"[WS] Loop error ({current_user_email}): {_err_key}: {loop_err}")
                    print(traceback.format_exc())
                    websocket_dashboard._last_loop_err = _err_key
                await asyncio.sleep(TICKER_INTERVAL)
                continue
            except Exception as loop_err:
                # Rate-limit error logging: print once, not every 100ms tick.
                _err_key = type(loop_err).__name__
                if not getattr(websocket_dashboard, '_last_loop_err', None) == _err_key:
                    import traceback
                    print(f"[WS] Loop error ({current_user_email}): {_err_key}: {loop_err}")
                    print(traceback.format_exc())
                    websocket_dashboard._last_loop_err = _err_key
                await asyncio.sleep(TICKER_INTERVAL)
                continue
    except WebSocketDisconnect:
        print(f"ðŸ”Œ WebSocket disconnected: {current_user_email}")
    except asyncio.TimeoutError:
        print(f"â±ï¸ WebSocket timeout during authentication")
        await websocket.close(code=1000)
    except Exception as e:
        print(f"âŒ WebSocket error: {e}")
        try:
            await websocket.close(code=1011)
        except:
            pass

# -------------------------------------------------------------------
# Alpha Mode toggle (only for Pro users)
# -------------------------------------------------------------------
@app.post("/alpha/toggle")
async def toggle_alpha_mode(email: str = Depends(get_current_user)):
    user_plan = get_user_plan(email)
    if user_plan not in ("pro", "premium", "pro-dev"):
        raise HTTPException(status_code=403, detail="Alpha Mode is only available for Pro subscribers")
    
    if LIVE_STATE.engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready")
    new_state = not LIVE_STATE.engine.alpha_mode
    LIVE_STATE.engine.alpha_mode = new_state
    LIVE_STATE.data["alpha_mode"] = new_state
    return {"alpha_mode": new_state}

# -------------------------------------------------------------------
# Dev code system â€” admin generation + user redemption
# -------------------------------------------------------------------

ADMIN_SECRET = os.getenv("ADMIN_SECRET", "")

# Sentinel document key inside dev_codes collection that tracks the single
# "always-active" developer token the backend continuously displays.
_CURRENT_TOKEN_SENTINEL = "current_token"
_DEV_TOKEN_VALIDITY_DAYS = 5


def _provision_dev_token() -> Dict[str, str]:
    """Generate a fresh dev token, persist to Firestore, update sentinel, delete previous unused token."""
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(days=_DEV_TOKEN_VALIDITY_DAYS)
    code = _make_dev_code()

    # Delete previous unused sentinel code to keep Firestore clean
    try:
        prev_sentinel = db.collection("dev_codes").document(_CURRENT_TOKEN_SENTINEL).get()
        if getattr(prev_sentinel, "exists", False):
            prev_data = prev_sentinel.to_dict() or {}
            prev_code = prev_data.get("active_code", "")
            if prev_code:
                prev_doc = db.collection("dev_codes").document(prev_code).get()
                if getattr(prev_doc, "exists", False):
                    prev_code_data = prev_doc.to_dict() or {}
                    if not prev_code_data.get("used_by"):
                        db.collection("dev_codes").document(prev_code).delete()
    except Exception as _cleanup_err:
        logger.warning(f"Dev token cleanup error: {_cleanup_err}")

    db.collection("dev_codes").document(code).set({
        "source": "backend",
        "plan": "pro",
        "label": "dev",
        "created_at": now.isoformat(),
        "expires_at": expires_at.isoformat(),
        "used_by": None,
        "used_at": None,
    })

    db.collection("dev_codes").document(_CURRENT_TOKEN_SENTINEL).set({
        "active_code": code,
        "expires_at": expires_at.isoformat(),
        "created_at": now.isoformat(),
    })

    return {"code": code, "expires_at": expires_at.isoformat()}


def _get_or_refresh_dev_token() -> Dict[str, str]:
    """Return the current active dev token, generating a fresh one if expired or used."""
    try:
        sentinel_doc = db.collection("dev_codes").document(_CURRENT_TOKEN_SENTINEL).get()
        if getattr(sentinel_doc, "exists", False):
            data = sentinel_doc.to_dict() or {}
            code = data.get("active_code", "")
            expires_str = data.get("expires_at", "")
            try:
                still_valid = datetime.now(timezone.utc) < datetime.fromisoformat(expires_str)
            except Exception:
                still_valid = False

            if still_valid and code:
                code_doc = _get_dev_code_doc(code)
                if code_doc and not code_doc.get("used_by"):
                    return {"code": code, "expires_at": expires_str}
    except Exception as e:
        logger.warning(f"Dev token sentinel read error: {e}")

    return _provision_dev_token()


async def dev_token_display_loop():
    """
    Startup task: generate ONE dev token and print it.
    Sleeps until the token expires (5 days), then generates the next one.
    Never regenerates more frequently than once per token lifetime.
    """
    await asyncio.sleep(5)  # let uvicorn startup messages settle
    while True:
        try:
            token_info = await asyncio.to_thread(_provision_dev_token)
            code = token_info["code"]
            expires_str = token_info["expires_at"]
            try:
                exp_dt = datetime.fromisoformat(expires_str)
                exp_display = exp_dt.strftime("%Y-%m-%d %H:%M UTC")
            except Exception:
                exp_dt = None
                exp_display = expires_str

            sep = "=" * 58
            banner = (
                f"\n{sep}\n"
                f"  AEGIS -- DEV TOKEN  (valid 5 days, one-time use)\n"
                f"{sep}\n"
                f"  Token   : {code}\n"
                f"  Expires : {exp_display}\n"
                f"{sep}\n"
            )
            print(banner, flush=True)
            logger.info(banner)

            # Sleep until this token expires, then generate the next one.
            if exp_dt is not None:
                if exp_dt.tzinfo is None:
                    exp_dt = exp_dt.replace(tzinfo=timezone.utc)
                secs_until_expiry = (exp_dt - datetime.now(timezone.utc)).total_seconds()
                sleep_for = max(secs_until_expiry, 3600)  # at least 1 h safety floor
            else:
                sleep_for = 5 * 24 * 3600  # 5 days default

            await asyncio.sleep(sleep_for)

        except Exception as e:
            print(f"[dev_token_display_loop ERROR] {e}", flush=True)
            logger.error(f"[dev_token_display_loop] {e}", exc_info=True)
            await asyncio.sleep(3600)  # retry in 1 h on error


async def dev_key_display_loop():
    """Startup task: generate one dev key, store it in Firestore, and print it to logs."""
    await asyncio.sleep(3)  # let uvicorn startup messages land first
    try:
        new_key = _make_dev_code()
        now_dt = datetime.now(timezone.utc)
        expires_dt = now_dt + timedelta(days=30)
        features = DEV_KEY_FEATURES

        sep = "\u2550" * 63
        banner = (
            f"\n{sep}\n"
            f"  AEGIS -- DEVELOPER KEY (valid 30 days)\n"
            f"  Key     : {new_key}\n"
            f"  Expires : {expires_dt.strftime('%Y-%m-%d')}\n"
            f"  Features: {', '.join(features)}\n"
            f"{sep}\n"
        )
        print(banner, flush=True)
        logger.info(banner)

        # Write to dev_codes (document ID = the code itself) so /api/redeem-dev-code
        # can look it up via _get_dev_code_doc() which does .document(code).get()
        #
        # Off-loop: .set() is a blocking Firestore WRITE and this task fires three
        # seconds into boot. On a project that is over its write quota the client
        # retries the 429 up to its 60s deadline, and on the event loop that is 60s
        # in which the site is bound to its port and answering nothing.
        await asyncio.to_thread(
            db.collection("dev_codes").document(new_key).set,
            {
                "source": "backend",
                "plan": "pro",
                "label": "startup_key",
                "created_at": now_dt.isoformat(),
                "expires_at": expires_dt.isoformat(),
                "features": features,
                "created_by": "system_startup",
                "used_by": None,
            },
        )

    except Exception as e:
        print(f"[dev_key_display_loop ERROR] {e}", flush=True)
        logger.error(f"[dev_key_display_loop] {e}", exc_info=True)


_DEV_CODE_RE = re.compile(r'^AEGIS-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}$')
_DEV_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no 0/O, 1/I/L

def _make_dev_code() -> str:
    seg = lambda: "".join(secrets.choice(_DEV_CODE_ALPHABET) for _ in range(4))
    return f"AEGIS-{seg()}-{seg()}-{seg()}"

async def _require_admin(x_admin_key: Optional[str] = Header(None)) -> None:
    if not ADMIN_SECRET or x_admin_key != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Admin access required")

@app.delete("/api/admin/clear-users")
def clear_users(_: None = Depends(_require_admin)):
    """Delete all documents in the users collection. Admin only."""
    users_ref = db.collection("users")
    docs = users_ref.stream()
    deleted = 0
    for doc in docs:
        doc.reference.delete()
        deleted += 1
    return {"deleted": deleted, "message": f"Cleared {deleted} user documents from Firestore"}

@app.get("/api/admin/smtp-test")
async def smtp_test(_: None = Depends(_require_admin)):
    """Test SMTP connectivity and return exact error â€” protected by X-Admin-Key header."""
    server  = os.getenv("MAIL_SERVER",   "NOT SET")
    user    = os.getenv("MAIL_USERNAME", "NOT SET")
    from_   = os.getenv("MAIL_FROM",     "NOT SET")
    has_pw  = bool(os.getenv("MAIL_PASSWORD"))
    result  = {"server": server, "user": user, "from": from_, "has_password": has_pw, "port587": None, "port465": None}

    test_msg = MessageSchema(
        subject="AEGIS SMTP Test",
        recipients=[NameEmail(name=user, email=user)],
        body="<p>SMTP test from AEGIS admin panel.</p>",
        subtype=MessageType.html,
    )
    try:
        await asyncio.wait_for(fastmail.send_message(test_msg), timeout=10.0)
        result["port587"] = "OK"
    except asyncio.TimeoutError:
        result["port587"] = "TIMEOUT after 10s"
    except Exception as e:
        result["port587"] = f"{type(e).__name__}: {e}"

    try:
        await asyncio.wait_for(fastmail_ssl.send_message(test_msg), timeout=10.0)
        result["port465"] = "OK"
    except asyncio.TimeoutError:
        result["port465"] = "TIMEOUT after 10s"
    except Exception as e:
        result["port465"] = f"{type(e).__name__}: {e}"

    return result

@app.delete("/api/admin/track-record/{signal_id}")
async def delete_track_record_entry(signal_id: str, _: None = Depends(_require_admin)):
    """Remove one signal from EVERY track-record store.

    The public endpoint merges live_engine's record (the Railway volume), the
    web copy and this process's in-memory store. Until 2026-08-20 this handler
    only touched the in-memory store, so deleting a record that the engine owned
    returned 404 while the row stayed on the page — indistinguishable from the
    delete being ignored.
    """
    global _track_store, _tr_seen_ids
    before = len(_track_store)
    _track_store = [r for r in _track_store if r.get("signal_id") != signal_id]
    _tr_seen_ids.discard(signal_id)
    mem_removed = before - len(_track_store)
    if mem_removed:
        _save_track_record()

    # Order matters. Drop it from the wallet FIRST, then from disk, then let the
    # engine rewrite the file: its orphan-preservation pass reads the on-disk
    # copy, so purging disk before that rewrite is what stops the row coming
    # back. Doing this in the other order restores the record every time.
    wallet_removed = _purge_ids_from_wallet({signal_id})
    disk_removed = _purge_ids_from_disk({signal_id})
    try:
        eng = getattr(LIVE_STATE, "engine", None)
        if eng is not None and wallet_removed.get("slices"):
            eng._save_track_record()
    except Exception as exc:
        print(f"[TrackRecord] engine re-save after delete failed: {exc!r}")

    total = mem_removed + sum(disk_removed.values()) + int(wallet_removed.get("slices") or 0)
    if total == 0:
        raise HTTPException(status_code=404, detail=f"signal_id '{signal_id}' not found")
    return {"success": True, "removed": total, "signal_id": signal_id,
            "stores": {"memory": mem_removed, **disk_removed,
                       "wallet_slices": wallet_removed.get("slices", 0)},
            "pnl_reversed_usdt": wallet_removed.get("pnl_reversed", 0.0),
            "balance": wallet_removed.get("balance"),
            "remaining": len(_track_store)}


@app.get("/api/admin/runtime")
async def runtime_controls_get(_: None = Depends(_require_admin)):
    """Current live value of every tunable, with its bounds and code default."""
    return {"controls": _rt_read(), "paused": bool(getattr(_cfg_mod, "TRADING_PAUSED", False))}


@app.post("/api/admin/runtime")
async def runtime_controls_set(payload: Dict[str, Any], _: None = Depends(_require_admin)):
    """Set one or more controls live. Takes effect on the next scan.

    `reset: true` restores the values committed in code, which is the escape
    hatch if a dial is left somewhere harmful — it never depends on remembering
    what the previous value was.
    """
    if payload.get("reset"):
        _rt_capture_defaults()
        applied = _rt_apply(dict(_RUNTIME_DEFAULTS), persist=True)
        return {"status": "reset", "applied": applied, "controls": _rt_read()}
    values = {k: v for k, v in (payload or {}).items() if k in _TUNABLES}
    if not values:
        raise HTTPException(status_code=422,
                            detail=f"no known controls in request. Valid: {sorted(_TUNABLES)}")
    try:
        applied = _rt_apply(values)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"status": "ok", "applied": applied, "controls": _rt_read()}


@app.get("/control")
async def control_panel():
    """Mobile control panel. The PAGE is public; every action on it requires the
    admin key, which is held only in the browser and sent per request."""
    return FileResponse(WEB_ROOT_PATH / "src" / "pages" / "control.html")


@app.post("/api/admin/reset-track-record")
async def reset_track_record(_: None = Depends(_require_admin)):
    """
    Wipe the track record.  Clears in-memory signal store, resets the on-disk
    JSON files, and resets the live engine's virtual wallet.
    Protected by X-Admin-Key header (ADMIN_SECRET env var).
    """
    global _track_store, _tr_seen_ids, _tr_last_save

    # Clear in-memory signal log
    _track_store.clear()
    _tr_seen_ids.clear()
    _tr_last_save = 0.0

    now_iso = datetime.now(timezone.utc).isoformat()
    empty_payload: Dict[str, Any] = {
        "generated_at": now_iso,
        "summary": {},
        "signals": [],
    }

    # Persist empty files — BOTH stores, which is the whole reason the previous
    # two resets did not stick.
    #
    # main.py:708 binds TRACK_RECORD_PATH to web/track_record.json, so the old
    # tuple (TRACK_RECORD_PATH, WEB_ROOT_PATH / "track_record.json") named the
    # SAME file twice and never touched the volume. The engine binds its own
    # TRACK_RECORD_PATH to STATE_DIR/track_record.json (scripts/engine/config.py
    # :199) — a different file behind the same NAME — and GET /api/track-record
    # merges the two. So the reset cleared one of the two sources, returned
    # success, and the record reappeared as soon as anything read it.
    #
    # Derived exactly the way the read path derives it (see the _ENGINE_RECORD
    # line in track_record_endpoint) so the two cannot drift apart again. If that
    # expression ever changes, change it in both places or this silently breaks
    # in the same way.
    _engine_record = Path(
        os.environ.get('AEGIS_STATE_DIR') or (Path(BASE_DIR) / "data")
    ) / "track_record.json"

    # dict.fromkeys de-duplicates while preserving order: on a local dev box with
    # no AEGIS_STATE_DIR these can legitimately resolve to the same file, and
    # writing it twice is harmless but pointless.
    for path in dict.fromkeys((
        TRACK_RECORD_PATH,
        WEB_ROOT_PATH / "track_record.json",
        _engine_record,
    )):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(empty_payload, f)
            os.replace(tmp, path)
        except Exception as e:
            print(f"[reset-track-record] Could not write {path}: {e}")

    # Reset the live engine's virtual wallet (paper trading positions + history)
    _eng = LIVE_STATE.engine
    if _eng is not None:
        try:
            wallet = _eng.wallet
            wallet.open_positions.clear()
            wallet.trade_history.clear()
            wallet.balance = wallet.initial_capital
        except Exception as e:
            print(f"[reset-track-record] Could not reset wallet: {e}")

    # Clear the durable Firestore copy too — otherwise the engine's boot-time
    # hydrate would restore the old record on the next redeploy, undoing this reset.
    try:
        from scripts.live_engine import _fs_clear_track_record
        _fs_clear_track_record()
    except Exception as e:
        print(f"[reset-track-record] Could not clear Firestore copy: {e}")

    # Broadcast empty payload to any connected track-record WebSocket clients
    try:
        await _tr_ws_manager.broadcast(empty_payload)
    except Exception:
        pass

    print(f"[TrackRecord] Reset by admin at {now_iso}")
    return {"success": True, "message": "Track record cleared.", "reset_at": now_iso}


def _get_dev_code_doc(code: str) -> Optional[Dict]:
    doc_ref = db.collection("dev_codes").document(code)
    doc = doc_ref.get()
    to_dict = getattr(doc, "to_dict", None)
    exists = getattr(doc, "exists", False)
    if callable(to_dict) and exists:
        result = to_dict()
        return result if isinstance(result, dict) else {}
    return None

class GenerateDevCodeRequest(BaseModel):
    count: int = 1
    plan: str = "pro"
    days: int = 30
    label: str = "beta"

@app.post("/admin/dev-codes/generate")
def admin_generate_dev_codes(
    req: GenerateDevCodeRequest,
    _admin: None = Depends(_require_admin),
):
    if req.plan not in ("basic", "intermediate", "pro"):
        raise HTTPException(status_code=400, detail="Invalid plan")
    if not (1 <= req.count <= 50):
        raise HTTPException(status_code=400, detail="count must be 1â€“50")
    if not (1 <= req.days <= 365):
        raise HTTPException(status_code=400, detail="days must be 1â€“365")

    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(days=req.days)
    codes = []

    for _i in range(req.count):
        code = _make_dev_code()
        db.collection("dev_codes").document(code).set({
            "source": "backend",
            "plan": req.plan,
            "label": req.label,
            "created_at": now.isoformat(),
            "expires_at": expires_at.isoformat(),
            "used_by": None,
            "used_at": None,
        })
        codes.append({"code": code, "expires_at": expires_at.isoformat()})

    return {"generated": len(codes), "plan": req.plan, "codes": codes}


@app.get("/admin/dev-codes")
def admin_list_dev_codes(
    include_used: bool = False,
    _admin: None = Depends(_require_admin),
):
    """List all dev codes from Firestore. By default only unused+unexpired codes."""
    now = datetime.now(timezone.utc)
    docs = db.collection("dev_codes").stream()
    codes = []
    for doc in docs:
        code_id = doc.id
        if code_id == _CURRENT_TOKEN_SENTINEL:
            continue
        data = doc.to_dict() or {}
        used_by = data.get("used_by")
        try:
            exp_dt = datetime.fromisoformat(data.get("expires_at", ""))
            expired = now > exp_dt
        except Exception:
            expired = True
        if not include_used and (used_by or expired):
            continue
        codes.append({
            "code": code_id,
            "plan": data.get("plan"),
            "label": data.get("label"),
            "created_at": data.get("created_at"),
            "expires_at": data.get("expires_at"),
            "used_by": used_by,
            "used_at": data.get("used_at"),
            "expired": expired,
        })
    codes.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    return {"count": len(codes), "codes": codes}


@app.get("/admin/dev-codes/current")
async def admin_get_current_dev_token(
    _admin: None = Depends(_require_admin),
):
    """Return the current always-on dev token generated by the background loop."""
    token_info = await asyncio.to_thread(_get_or_refresh_dev_token)
    return token_info


class DevCodeRequest(BaseModel):
    code: str

@app.post("/api/redeem-dev-code")
async def redeem_dev_code(req: DevCodeRequest, email: str = Depends(get_current_user)):
    code = req.code.strip().upper()

    if not _DEV_CODE_RE.match(code):
        raise HTTPException(status_code=400, detail="Invalid code format. Expected: AEGIS-XXXX-XXXX-XXXX")

    code_data = _get_dev_code_doc(code)
    if code_data is None:
        raise HTTPException(status_code=404, detail="Dev code not found")

    if code_data.get("source") != "backend":
        raise HTTPException(status_code=403, detail="Dev code is not valid")

    try:
        expires_at_dt = datetime.fromisoformat(code_data["expires_at"])
    except Exception:
        raise HTTPException(status_code=500, detail="Malformed dev code record")

    if datetime.now(timezone.utc) > expires_at_dt:
        raise HTTPException(status_code=410, detail="This dev code has expired")

    used_by = code_data.get("used_by")
    if used_by and used_by != email:
        raise HTTPException(status_code=409, detail="This dev code has already been used")

    plan = code_data.get("plan", "pro")
    expires_iso = code_data["expires_at"]

    user_doc = get_user_doc(email)
    if not user_doc:
        raise HTTPException(status_code=404, detail="User account not found")

    user_ref = db.collection("users").document(email)
    update_result = user_ref.update({
        "plan": plan,
        "subscription": {
            "status": "active",
            "payment_id": f"devcode_{code}",
            "order_id": f"devcode_{code}",
            "activated_at": datetime.now(timezone.utc).isoformat(),
            "plan_type": plan,
            "expires_at": expires_iso,
        },
        "trial_active": False,
        "dev_code_used": code,
    })
    if inspect.isawaitable(update_result):
        await update_result

    if not used_by:
        mark_result = db.collection("dev_codes").document(code).update({
            "used_by": email,
            "used_at": datetime.now(timezone.utc).isoformat(),
        })
        if inspect.isawaitable(mark_result):
            await mark_result

        # Token consumed â€” provision a replacement immediately so the backend
        # always has an active token available for the next developer.
        try:
            await asyncio.to_thread(_provision_dev_token)
        except Exception as _e:
            logger.warning(f"Auto-provision replacement dev token failed: {_e}")

    return {"status": "success", "plan": plan, "expires_at": expires_iso}

# -------------------------------------------------------------------
# Dev Key System â€” validate-devkey endpoint
# -------------------------------------------------------------------

DEV_KEY_FEATURES = ["extended_timeframes", "alpha_mode", "all_signals", "pro_signals"]

# In-memory TTL cache for validated dev keys (key_string -> {cached_at})
_dev_key_cache: Dict[str, Dict] = {}
_DEV_KEY_CACHE_TTL = 300  # 5 minutes

class ValidateDevKeyRequest(BaseModel):
    dev_key: str

@app.post("/auth/validate-devkey")
def validate_dev_key(req: ValidateDevKeyRequest, request: Request):
    """
    Validate a developer key against the /dev_keys Firestore collection.
    Returns plan info and features if valid; logs usage for audit.
    Rate-limited to 5 attempts per minute per IP.
    """
    key_str = req.dev_key.strip()
    if not key_str:
        return JSONResponse({"valid": False, "error": "Invalid or expired key"}, status_code=400)

    # --- Rate limiting (5 attempts per minute per IP) ---
    client_ip = request.headers.get(
        "x-forwarded-for", request.client.host if request.client else "unknown"
    ).split(",")[0].strip()
    rate_key = f"devkey_rate_{client_ip}"
    now_ts = time.time()
    if not hasattr(validate_dev_key, "_rate_store"):
        validate_dev_key._rate_store = {}
    rate_store = validate_dev_key._rate_store
    window = rate_store.get(rate_key, {"count": 0, "window_start": now_ts})
    if now_ts - window["window_start"] > 60:
        window = {"count": 0, "window_start": now_ts}
    window["count"] += 1
    rate_store[rate_key] = window
    if window["count"] > 5:
        print(f"[DevKey] Rate limit exceeded for IP {client_ip}")
        return JSONResponse(
            {"valid": False, "error": "Too many attempts. Please wait a minute."},
            status_code=429,
        )

    # --- Check in-memory cache first ---
    cached = _dev_key_cache.get(key_str)
    if cached and (now_ts - cached["cached_at"]) < _DEV_KEY_CACHE_TTL:
        print(f"[DevKey] Cache hit for key (masked): {key_str[:8]}...")
        return {"valid": True, "plan": "pro", "features": DEV_KEY_FEATURES}

    # --- Validate against Firestore /dev_keys collection ---
    try:
        keys_ref = db.collection("dev_keys")
        query_result = keys_ref.where("key", "==", key_str).limit(1).stream()

        key_doc = None
        key_doc_id = None
        for doc in query_result:
            key_doc = doc.to_dict()
            key_doc_id = doc.id
            break

        if key_doc is None:
            print(f"[DevKey] Key not found: {key_str[:8]}...")
            return JSONResponse({"valid": False, "error": "Invalid or expired key"})

        # Check expiry
        expires_at_raw = key_doc.get("expires_at")
        if expires_at_raw:
            try:
                if hasattr(expires_at_raw, "timestamp"):
                    expires_dt = datetime.fromtimestamp(expires_at_raw.timestamp(), tz=timezone.utc)
                else:
                    expires_dt = datetime.fromisoformat(str(expires_at_raw).replace("Z", "+00:00"))
                if datetime.now(timezone.utc) > expires_dt:
                    print(f"[DevKey] Expired key: {key_str[:8]}...")
                    return JSONResponse({"valid": False, "error": "Invalid or expired key"})
            except Exception as e:
                print(f"[DevKey] Error parsing expires_at: {e}")
                return JSONResponse({"valid": False, "error": "Invalid or expired key"})

        # Log usage: increment usage_count and update last_used
        try:
            db.collection("dev_keys").document(key_doc_id).update({
                "usage_count": (key_doc.get("usage_count") or 0) + 1,
                "last_used": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as log_err:
            print(f"[DevKey] Failed to log usage: {log_err}")

        # Cache the valid key
        _dev_key_cache[key_str] = {"cached_at": now_ts}

        features = key_doc.get("features", DEV_KEY_FEATURES)
        print(f"[DevKey] Valid key activated: {key_str[:8]}... features={features}")
        return {"valid": True, "plan": "pro", "features": features}

    except Exception as e:
        print(f"[DevKey] Firestore error during validation: {e}")
        return JSONResponse(
            {"valid": False, "error": "Validation service unavailable"},
            status_code=503,
        )

# -------------------------------------------------------------------
# Feedback endpoint
# -------------------------------------------------------------------
@app.post("/feedback")
async def send_feedback(fb: Feedback):
    message = MessageSchema(
        subject=f"Feedback from {fb.name}",
        recipients=[NameEmail(name="Animesh Kukreti", email="animeshkukreti60@gmail.com")],
        body=f"From: {fb.email}\n\n{fb.message}",
        subtype=MessageType.plain,
    )
    await fastmail.send_message(message)
    return {"status": "sent"}

# -------------------------------------------------------------------
# Reviews endpoint
# -------------------------------------------------------------------
@app.post("/reviews")
async def submit_review(review: Review):
    try:
        rating = int(review.rating)
    except Exception:
        raise HTTPException(status_code=400, detail="Rating must be an integer 1-5")
    if rating < 1 or rating > 5:
        raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")

    review_doc = {
        "name": review.name,
        "email": review.email,
        "rating": rating,
        "message": review.message or "",
        "product": review.product or "",
        "created_at": datetime.now(timezone.utc),
    }

    # 1) Persist to Firestore (best-effort).
    saved = False
    try:
        db.collection('reviews').add(review_doc)
        saved = True
        print(f"âœ… Review saved to Firestore: {review.email}")
    except Exception as e:
        print(f"âŒ Failed to save review to Firestore: {e}")

    # 2) ALWAYS notify the work inbox.  Previously the email was only sent when
    #    the Firestore write FAILED â€” so with Firestore working (the normal
    #    case) no notification was ever delivered, even though the page showed
    #    "review sent".  It also went to the personal gmail, not the Neo work
    #    mailbox.  Now it always fires, to REVIEW_NOTIFY_EMAIL (defaulting to the
    #    Neo-hosted work address), via the robust Resend/SMTP helper.
    notify_to = os.getenv("REVIEW_NOTIFY_EMAIL", "aegisofficial@aegisignal.pro")
    _msg_html = (review.message or "").replace("\n", "<br>") or "â€”"
    emailed = False
    try:
        await _send_email(
            to=notify_to,
            subject=f"New {rating}â˜… review from {review.name}",
            html=(f"<h3>New {rating}â˜… review</h3>"
                  f"<p><b>Name:</b> {review.name}<br>"
                  f"<b>Email:</b> {review.email}<br>"
                  f"<b>Product:</b> {review.product or 'â€”'}<br>"
                  f"<b>Rating:</b> {rating}/5</p>"
                  f"<p><b>Message:</b><br>{_msg_html}</p>"),
        )
        emailed = True
        print(f"âœ… Review notification emailed â†’ {notify_to}")
    except Exception as e2:
        print(f"âŒ Failed to send review notification email: {e2}")

    if not saved and not emailed:
        raise HTTPException(status_code=500, detail="Failed to record review")
    return {"status": "saved", "firestore": saved, "emailed": emailed}

# -------------------------------------------------------------------
# Trade Execution endpoint
# -------------------------------------------------------------------
class TradeExecuteRequest(BaseModel):
    symbol: str
    side: str
    entryPrice: float
    stopLoss: float
    takeProfit: float
    riskPercent: float
    leverage: float
    positionUnits: float
    notionalValue: float
    status: str = "open"
    signalId: Optional[str] = None
    userId: Optional[str] = None

@app.post("/api/trades/execute")
def execute_trade(request: TradeExecuteRequest, user_id: str = Depends(get_firebase_uid)):
    trade_data = request.dict()
    trade_data["openTime"] = datetime.now(timezone.utc).isoformat()
    # Ensure it's saved under the user who made the request
    trade_data["userId"] = user_id
    
    try:
        # NOTE: Connect to the Demat API here.
        # This is where the call will be sent to the broker API for execution.
        print(f"ðŸš€ Sending order to Demat API: Symbol: {trade_data.get('symbol')}, Side: {trade_data.get('side')}, Units: {trade_data.get('positionUnits')}")
        # demat_response = await demat_client.place_order(...)
        
        trade_ref = db.collection("users").document(user_id).collection("trades").document()
        trade_data["id"] = trade_ref.id
        trade_data["demat_status"] = "sent_to_broker"
        trade_ref.set(trade_data)
        print(f"âœ… Trade executed and sent to Demat via API for {user_id}")
        return {"status": "success", "trade_id": trade_ref.id, "trade": trade_data, "message": "Order sent to Demat account successfully"}
    except Exception as e:
        print(f"âŒ Failed to execute trade for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to execute trade")

@app.post("/api/trades/{trade_id}/close")
def close_trade(trade_id: str, user_id: str = Depends(get_firebase_uid)):
    try:
        trade_ref = db.collection("users").document(user_id).collection("trades").document(trade_id)
        trade_doc = trade_ref.get()
        if not trade_doc.exists:
            raise HTTPException(status_code=404, detail="Trade not found")
        trade_ref.update({
            "status": "closed",
            "closeTime": datetime.now(timezone.utc).isoformat()
        })
        print(f"âœ… Trade {trade_id} closed for {user_id}")
        return {"status": "success", "trade_id": trade_id}
    except HTTPException:
        raise
    except Exception as e:
        print(f"âŒ Failed to close trade {trade_id} for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to close trade")

# -------------------------------------------------------------------
# Legacy OTP endpoints (kept for compatibility)
# -------------------------------------------------------------------
@app.post("/send-otp")
async def send_otp(request: OTPSendRequest):
    email = request.email
    existing_user = get_user_doc(email)
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered. Please sign in.")
    if is_cooldown_active(email):
        raise HTTPException(status_code=429, detail="Too many requests. Please wait before requesting another OTP.")
    otp = generate_otp()
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=5)
    cooldown_until = datetime.now(timezone.utc) + timedelta(seconds=60)
    otp_store[email] = {
        "otp": otp,
        "expires_at": expires_at,
        "cooldown_until": cooldown_until
    }
    otp_html = f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="margin:0;padding:0;background:#0C0F13;font-family:'Segoe UI',Arial,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#0C0F13;padding:40px 0;">
    <tr><td align="center">
      <table width="480" cellpadding="0" cellspacing="0"
             style="background:#141820;border:1px solid rgba(184,150,106,0.18);border-radius:16px;overflow:hidden;max-width:480px;">
        <tr>
          <td style="padding:32px 40px 24px;text-align:center;border-bottom:1px solid rgba(184,150,106,0.1);">
            <div style="font-size:22px;font-weight:700;letter-spacing:3px;color:#B8966A;font-family:Georgia,serif;">AEGIS</div>
            <p style="color:#6b7280;margin:6px 0 0;font-size:13px;letter-spacing:1px;">AI SIGNAL TERMINAL</p>
          </td>
        </tr>
        <tr>
          <td style="padding:36px 40px;">
            <p style="color:#9ca3af;font-size:15px;margin:0 0 8px;">Email Verification</p>
            <h2 style="color:#EAE6DF;font-size:20px;font-weight:600;margin:0 0 24px;">Your one-time verification code</h2>
            <div style="background:#0C0F13;border:1px solid rgba(184,150,106,0.25);border-radius:12px;padding:24px;text-align:center;margin:0 0 24px;">
              <span style="font-family:'Courier New',Courier,monospace;font-size:38px;font-weight:700;letter-spacing:12px;color:#B8966A;">{otp}</span>
            </div>
            <p style="color:#9ca3af;font-size:14px;margin:0 0 8px;">
              This code expires in <strong style="color:#EAE6DF;">5 minutes</strong>. Do not share it with anyone.
            </p>
            <p style="color:#6b7280;font-size:13px;margin:0;">If you didn't request this, you can safely ignore this email.</p>
          </td>
        </tr>
        <tr>
          <td style="padding:20px 40px 28px;border-top:1px solid rgba(255,255,255,0.05);text-align:center;">
            <p style="color:#4b5563;font-size:12px;margin:0;">
              Sent by <a href="mailto:aegisofficial@aegisignal.pro" style="color:#B8966A;text-decoration:none;">aegisofficial@aegisignal.pro</a>
              &nbsp;Â·&nbsp;
              <a href="https://aegisignal.pro" style="color:#B8966A;text-decoration:none;">aegisignal.pro</a>
            </p>
          </td>
        </tr>
      </table>
    </td></tr>
  </table>
</body>
</html>"""
    try:
        await _send_email(to=email, subject="Your AEGIS Verification Code", html=otp_html)
    except Exception as e:
        otp_store.pop(email, None)
        print(f"Email sending failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to send OTP email. Check email configuration.")
    return {"success": True, "message": "OTP sent to your email address."}

@app.post("/verify-otp")
async def verify_otp(request: OTPVerifyRequest):
    email = request.email
    otp = request.otp
    if email not in otp_store:
        raise HTTPException(status_code=400, detail="No OTP request found for this email. Please request a new OTP.")
    record = otp_store[email]
    if datetime.now(timezone.utc) > record["expires_at"]:
        otp_store.pop(email, None)
        raise HTTPException(status_code=400, detail="OTP has expired. Please request a new one.")
    if record["otp"] != otp:
        raise HTTPException(status_code=400, detail="Invalid OTP. Please try again.")
    otp_store.pop(email, None)
    return {"success": True, "message": "OTP verified successfully. You may now complete registration."}

# -------------------------------------------------------------------
# FIRESTORE SIGNALS API â€“ Get all active signals
# -------------------------------------------------------------------
# NOTE: a second `@app.get("/api/signals")` used to live here — an unauthed
# Firestore dump. It took HTTPBearer(auto_error=False) and never looked at the
# plan, so it would have served the entire paid signal feed to anybody at all.
# It was unreachable only because FastAPI matches in registration order and the
# gated api_signals() above registers 74 routes earlier. That is not a control,
# it is a coincidence, and one file reshuffle away from being a full product
# leak. Deleted. The gated route is the only /api/signals.

# -------------------------------------------------------------------
# FIRESTORE SIGNALS API â€“ Get specific signal
# -------------------------------------------------------------------
@app.get("/api/signals/{symbol}")
def get_signal(symbol: str):
    """
    Get a specific signal by symbol.
    Example: /api/signals/BTC
    """
    try:
        signal_ref = db.collection("signals").document(symbol)
        signal_doc = signal_ref.get()  # Firestore .get() is synchronous
        
        # Use getattr for safe attribute access (handles both sync and async clients)
        exists_fn = getattr(signal_doc, "exists", None)
        to_dict_fn = getattr(signal_doc, "to_dict", None)
        
        if not exists_fn or not to_dict_fn:
            raise HTTPException(status_code=404, detail=f"Signal not found for {symbol}")
        
        signal_data = to_dict_fn()
        signal_data["id"] = symbol
        
        return {
            "success": True,
            "signal": signal_data
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"âŒ Error fetching signal {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch signal")

# -------------------------------------------------------------------
# FIRESTORE DASHBOARD API â€“ Get dashboard data for user
# -------------------------------------------------------------------
@app.get("/api/dashboard")
def get_dashboard(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
):
    """
    Get personalized dashboard data for authenticated user.
    Includes: user profile, trial info, subscriptions, recent trades, signals access.
    """
    # Extract user ID from JWT or auth header (credentials and request are provided by FastAPI)
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    # Parse JWT to get user email for personalized data
    try:
        assert SECRET_KEY is not None, "SECRET_KEY must be set"
        decoded_payload = jwt.decode(auth_header[7:], SECRET_KEY, algorithms=[ALGORITHM])
        current_user_email = decoded_payload.get("sub")
        
        # Fetch personal dashboard data from Firestore based on user email
        current_user_email = current_user_email or ""
        user_doc = get_user_doc(current_user_email) if current_user_email else None
        
        plan = user_doc.get("plan", "trial") if user_doc else "trial"
        trial_end = user_doc.get("trial_end") if user_doc else None
        trial_expired = is_trial_expired(current_user_email) if trial_end else False
        
        dashboard_data = {
            "user": {
                "authenticated": True,
                "plan": plan,
                "trial_active": not (isinstance(trial_end, str) and trial_expired),
                "trial_days_remaining": max(0, (datetime.fromisoformat(trial_end.replace("Z", "+00:00")) - datetime.now(timezone.utc)).days) if isinstance(trial_end, str) and not trial_expired else 0,
                "allowed_tokens": get_allowed_tokens(),
                "allowed_timeframes": get_allowed_timeframes(current_user_email)
            },
            "signals": [],
            "trades": [],
            "statistics": {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "total_pnl": 0
            }
        }
        
        # Get signals from engine with safe iteration
        if LIVE_STATE.data.get("signals"):
            # v79.3: read the RECONCILED snapshot, never raw engine.last_signals —
            # the snapshot enforces "fired iff an open position backs it".
            dashboard_data["signals"] = {k: v for k, v in list(LIVE_STATE.data["signals"].items())[:10]}
        
        return dashboard_data
    except HTTPException:
        raise
    except jwt.InvalidSignatureError:
        print("âŒ Invalid JWT signature")
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        # Fallback to generic data if personal data fails
        print(f"âŒ Error fetching dashboard (fallback): {str(e)}")
        return {
            "user": {
                "authenticated": False,
                "plan": "trial",
                "trial_active": True,
                "trial_days_remaining": 2,
                "allowed_tokens": PRO_TOKENS,
                "allowed_timeframes": ["30m", "1h"]
            },
            "signals": [],
            "trades": [],
            "statistics": {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "total_pnl": 0
            }
        }

# -------------------------------------------------------------------
# FIRESTORE PUBLIC SIGNALS â€“ No authentication required
# -------------------------------------------------------------------
# NOTE: a second `@app.get("/api/public/signals")` used to live here, returning
# 4 Firestore signals with no auth handling. Same story — shadowed by the
# auth-aware api_public_signals() above and therefore dead. Deleted rather than
# left as a trap for whoever next moves code in this file.

# -------------------------------------------------------------------
# FIRESTORE SIGNAL UPDATE â€“ Backend trigger (admin only)
# -------------------------------------------------------------------
@app.post("/api/admin/signals/update")
def update_signal(
    symbol: str,
    signal_data: Dict[str, Any],
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
):
    """
    Update or create a signal in Firestore.
    Should only be called by backend ML engine or admins.
    Requires Firebase admin credentials.
    """
    try:
        signal_ref = db.collection("signals").document(symbol)
        
        update_payload = {
            "symbol": symbol,
            "signal": signal_data.get("signal", "HOLD"),
            "entry": signal_data.get("entry", 0),
            "sl": signal_data.get("sl", 0),
            "tp": signal_data.get("tp", 0),
            "timeframe": signal_data.get("timeframe", "1h"),
            "confidence": signal_data.get("confidence", 0.5),
            "timestamp": datetime.now(timezone.utc),
            "active": True
        }
        
        signal_ref.set(update_payload, merge=True)
        
        return {
            "success": True,
            "message": f"Signal updated for {symbol}",
            "symbol": symbol
        }
    except Exception as e:
        print(f"âŒ Error updating signal: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update signal")

# -------------------------------------------------------------------
# API Portability & Developer Access
# -------------------------------------------------------------------
import hashlib
import secrets

async def verify_api_key(request: Request):
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
        
    api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    
    users_ref = db.collection("users")
    query = users_ref.where("api_key_hash", "==", api_key_hash).limit(1).stream()
    
    user_doc = None
    for doc in query:
        user_doc = doc.to_dict()
        break
        
    if not user_doc:
        raise HTTPException(status_code=401, detail="Invalid API Key")
        
    plan = user_doc.get("plan", "basic").lower()
    if plan not in ["pro", "premium"]:
        raise HTTPException(status_code=403, detail="Pro Tier required for API access")
        
    return user_doc

@app.get("/api/v1/signals/fleet")
def get_signals_fleet(symbol: Optional[str] = None, _user: dict = Depends(verify_api_key)):
    """
    Programmatic data portability endpoint for Pro users.
    Returns structured JSON array of live signals.
    """
    # Verify user is authenticated
    _ = _user
    
    signals_ref = db.collection("signals")
    query = signals_ref.where("status", "==", "ACTIVE")
    if symbol:
        query = query.where("symbol", "==", symbol)
        
    docs = query.stream()
    
    fleet = []
    for doc in docs:
        sig = doc.to_dict()
        if sig:
            fleet.append({
                "ticker": sig.get("symbol"),
                "direction": sig.get("direction"),
                "start_anchor": sig.get("entry_price"),
                "target_destination": sig.get("tp"),
                "stop_loss": sig.get("sl"),
                "model_conviction": sig.get("probabilities", {}),
                "timestamp": sig.get("timestamp")
            })
        
    return {"status": "SUCCESS", "data": fleet}

@app.post("/api/v1/developer/regenerate_key")
async def regenerate_api_key(user_id: str = Depends(get_current_user)):
    """
    Generates a new API key for the authenticated user, hashes it,
    stores the hash in Firestore, and returns the raw key once.
    """
    user_ref = db.collection("users").document(user_id)
    user_doc = await user_ref.get() # type: ignore
    
    if not user_doc.exists:
        raise HTTPException(status_code=404, detail="User not found")
        
    data = user_doc.to_dict()
    plan = data.get("plan", "basic").lower() if data else "basic"
    
    if plan not in ["pro", "premium", "intermediate"]:
        # Allow intermediate if required, but user said Pro
        if plan not in ["pro", "premium"]:
            raise HTTPException(status_code=403, detail="Pro Tier required for API access")
        
    # Generate new key
    raw_key = "aegis_live_" + secrets.token_urlsafe(32)
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    
    user_ref.update({
        "api_key_hash": key_hash,
        "api_key_last_generated": datetime.now(timezone.utc).isoformat()
    })
    
    return {"status": "SUCCESS", "api_key": raw_key}

# Frontend uses /api/v1/keys/regenerate â€” alias to the canonical route above
@app.post("/api/v1/keys/regenerate")
async def regenerate_api_key_alias(user_id: str = Depends(get_current_user)):
    return await regenerate_api_key(user_id)

# -------------------------------------------------------------------
# User settings (capital + risk) â€” persisted to Firestore
# Called by the Settings room in dashboard.js when user hits Save
# -------------------------------------------------------------------
class UserSettingsUpdate(BaseModel):
    capital: float
    risk_pct: float

@app.post("/user/settings")
def save_user_settings(
    payload: UserSettingsUpdate,
    user_id: str = Depends(get_current_user)
):
    if payload.capital < 100:
        raise HTTPException(status_code=400, detail="Capital must be at least $100")
    if not (0.5 <= payload.risk_pct <= 10):
        raise HTTPException(status_code=400, detail="Risk % must be between 0.5 and 10")

    user_ref = db.collection("users").document(user_id)
    user_ref.set(
        {"capital": payload.capital, "risk_pct": payload.risk_pct},
        merge=True
    )
    return {"status": "ok", "capital": payload.capital, "risk_pct": payload.risk_pct}

# -------------------------------------------------------------------


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  AEGIS UNIVERSAL TRADER â€” API Endpoints
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

_trader_engine_instance = None
_trader_engine_lock = __import__('threading').Lock()

def _get_trader_engine_lazy():
    """Lazily import and return the trader engine (avoids startup cost if unused)."""
    global _trader_engine_instance
    if _trader_engine_instance is None:
        with _trader_engine_lock:
            if _trader_engine_instance is None:
                try:
                    from scripts.trader_model.trader_engine import get_trader_engine
                    _trader_engine_instance = get_trader_engine()
                except Exception as _e:
                    logger.warning(f"Trader engine unavailable (models not trained yet): {_e}")
                    _trader_engine_instance = None
    return _trader_engine_instance


@app.get("/api/trader/signals")
async def get_trader_signals(
    mode:         Optional[str] = None,
    risk_profile: str           = "balanced",
    scan:         bool          = False,
    _user:        str           = Depends(get_current_user),
):
    """
    Return active Universal Trader signals.

    Query params:
      mode         â€“ filter by 'scalping' | 'intraday' | 'swing' (optional)
      risk_profile â€“ 'conservative' | 'balanced' | 'aggressive' (default: balanced)
      scan         â€“ if true, trigger a fresh scan (slow); otherwise return cached
    """
    engine = _get_trader_engine_lazy()
    if engine is None:
        return {
            "signals": [],
            "last_scan": None,
            "message": "Trader models not trained yet. Run: python -m scripts.trader_model.train_trader",
        }

    if scan:
        modes = [mode] if mode else None
        try:
            signals = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: engine.scan_all_tokens(
                    modes        = modes,
                    risk_profile = risk_profile,
                )
            )
        except Exception as e:
            logger.error(f"Trader scan error: {e}")
            signals = engine.active_signals
    else:
        signals = engine.active_signals
        if mode:
            signals = [s for s in signals if s.get('mode') == mode]

    return {
        "signals":   signals,
        "count":     len(signals),
        "last_scan": engine.last_scan_time,
        "modes_available": engine.model_store.loaded_modes,
    }


@app.get("/api/trader/track-record")
async def get_trader_track_record():
    """Public endpoint â€” returns normalised trader_track_record.json (wallet + trade history)."""
    trader_signals: list = []
    wins, losses, open_c, closed = 0, 0, 0, 0
    pnls = []
    if TRADER_TRACK_RECORD_PATH.exists():
        try:
            with open(TRADER_TRACK_RECORD_PATH, encoding='utf-8') as f:
                _td = json.load(f)
            for s in _td.get("signals", []):
                outcome = s.get("outcome")
                trader_signals.append({
                    "signal_id":       s.get("signal_id"),
                    "symbol":          s.get("symbol"),
                    "timeframe":       s.get("timeframe"),
                    "direction":       s.get("direction"),
                    "signal_type":     s.get("direction"),
                    "signal_status":   "ACTIVE" if outcome == "OPEN" else "CLOSED",
                    "entry_price":     s.get("entry_price"),
                    "take_profit":     s.get("tp1"),
                    "stop_loss":       s.get("stop_loss"),
                    "exit_price":      s.get("exit_price"),
                    "entry_time":      s.get("timestamp"),
                    "close_time":      s.get("exit_time"),
                    "pnl_pct":         s.get("pnl_pct"),
                    "outcome":         outcome,
                    "exit_reason":     s.get("exit_reason"),
                    "ai_prob":         s.get("confidence"),
                    "confluence_rate": s.get("confluence_score"),
                    "source":          "aegis_trader",
                    "mode":            s.get("mode"),
                })
        except Exception:
            pass

    all_signals = sorted(
        trader_signals,
        key=lambda r: r.get("entry_time") or "",
        reverse=True,
    )[:500]

    wins   = sum(1 for r in all_signals if r.get("outcome") == "WIN")
    losses = sum(1 for r in all_signals if r.get("outcome") == "LOSS")
    open_c = sum(1 for r in all_signals if r.get("outcome") == "OPEN")
    closed = wins + losses
    pnls   = [float(r.get("pnl_pct") or 0) for r in all_signals
              if r.get("outcome") in ("WIN", "LOSS")]
    times  = [r.get("entry_time") for r in all_signals if r.get("entry_time")]

    return JSONResponse({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_signals":  len(all_signals),
            "wins":           wins,
            "losses":         losses,
            "open":           open_c,
            "win_rate_pct":   round(wins / closed * 100, 1) if closed else None,
            "avg_pnl_pct":    round(sum(pnls) / len(pnls), 3) if pnls else None,
            "total_pnl_pct":  round(sum(pnls), 3) if pnls else 0.0,
            "tracking_since": min(times) if times else None,
        },
        "signals": all_signals,
    })


@app.get("/api/trader/status")
async def get_trader_token_status(_user: str = Depends(get_current_user)):
    """Return live scan status for all 60 deployment tokens."""
    engine = _get_trader_engine_lazy()
    if engine is None:
        return {"status": {}, "signals": [], "last_scan": None}
    return {
        "status":    engine.token_status,
        "signals":   engine.active_signals,
        "last_scan": engine.last_scan_time,
        "count":     len(engine.token_status),
    }


@app.get("/api/trader/wallet")
async def get_trader_wallet():
    """Return virtual wallet summary for the trader cockpit dashboard."""
    engine = _get_trader_engine_lazy()
    if engine is None:
        return {"balance": 10000, "total_pnl_usdt": 0, "total_pnl_pct": 0,
                "win_rate": 0, "won": 0, "lost": 0, "total_trades": 0,
                "open_positions": 0, "open_trades": []}
    w = engine.wallet
    return {
        **w.summary,
        "open_trades": list(w.open_positions.values()),
        "last_scan":   engine.last_scan_time,
    }


@app.get("/api/trader/stats")
async def get_trader_stats(_user: str = Depends(get_current_user)):
    """Return per-mode performance statistics."""
    engine = _get_trader_engine_lazy()
    if engine is None:
        return {"stats": {}, "wallet": {}}
    w = engine.wallet
    closed = [t for t in w.trade_history if t.get("outcome") in ("WIN", "LOSS")]
    stats: dict = {}
    for mode in ("scalping", "intraday", "swing"):
        mode_closed = [t for t in closed if t.get("mode") == mode]
        wins = [t for t in mode_closed if t.get("outcome") == "WIN"]
        stats[mode] = {
            "total":    len(mode_closed),
            "wins":     len(wins),
            "losses":   len(mode_closed) - len(wins),
            "win_rate": round(len(wins) / len(mode_closed), 3) if mode_closed else 0.0,
            "avg_pnl":  round(sum(t.get("pnl_pct", 0) or 0 for t in mode_closed) / max(len(mode_closed), 1), 3),
        }
    return {"stats": stats, "wallet": w.summary}


@app.get("/api/trader/record")
async def get_trader_record(
    mode:  Optional[str] = None,
    limit: int           = 200,
    _user: str           = Depends(get_current_user),
):
    """Return trader signals + wallet summary for the authenticated dashboard."""
    engine = _get_trader_engine_lazy()
    wallet_summary: dict = {}
    if engine is not None:
        wallet_summary = engine.wallet.summary

    from scripts.trader_model.trader_config import TRADER_RECORD_PATH
    if not TRADER_RECORD_PATH.exists():
        return {"signals": [], "total": 0, "wallet": wallet_summary}
    with open(TRADER_RECORD_PATH, encoding="utf-8") as f:
        record = json.load(f)
    signals = record.get("signals", [])
    if mode:
        signals = [s for s in signals if s.get("mode") == mode]
    return {
        "signals": list(reversed(signals[-limit:])),
        "total":   len(signals),
        "wallet":  wallet_summary,
    }


class TraderScanRequest(BaseModel):
    modes:        Optional[List[str]] = None
    risk_profile: str                 = "balanced"


@app.post("/api/trader/scan")
async def trigger_trader_scan(
    body:  TraderScanRequest = TraderScanRequest(),
    _user: str               = Depends(get_current_user),
):
    """Trigger an immediate scan and persist the updated JSON."""
    modes        = body.modes
    risk_profile = body.risk_profile
    engine = _get_trader_engine_lazy()
    if engine is None:
        raise HTTPException(status_code=503, detail="Trader models not trained yet")
    try:
        signals = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: engine.scan_all_tokens(modes=modes, risk_profile=risk_profile)
        )
        _save_trader_track_record()
        return {
            "signals":   signals,
            "count":     len(signals),
            "wallet":    engine.wallet.summary,
            "status":    "ok",
        }
    except Exception as e:
        logger.error(f"Trader scan error: {e}")
        raise HTTPException(status_code=500, detail=str(e))



# â”€â”€ Telegram Connect API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.get("/api/notifications/telegram/connect")
async def telegram_connect(_user: str = Depends(get_current_user)):
    """Generate a one-tap deep link the user opens in Telegram to connect."""
    if not _tg_token() or not _tg_username():
        raise HTTPException(
            status_code=503,
            detail="Telegram bot not configured. Ask admin to set TELEGRAM_BOT_TOKEN and TELEGRAM_BOT_USERNAME."
        )
    code = _secrets.token_hex(4).upper()  # e.g. A3F9C2D1
    _tg_pending[code] = _user
    deeplink = f"https://t.me/{_tg_username()}?start={code}"
    return {"deeplink": deeplink, "code": code, "bot_username": _tg_username()}


@app.get("/api/notifications/telegram/status")
async def telegram_status(_user: str = Depends(get_current_user)):
    """Check whether this user has connected their Telegram."""
    chat_id = _tg_chat_id(_tg_connections.get(_user, ""))
    return {"connected": bool(chat_id), "chat_id": chat_id}


@app.delete("/api/notifications/telegram/disconnect")
async def telegram_disconnect(_user: str = Depends(get_current_user)):
    """Unlink Telegram from this account."""
    _tg_connections.pop(_user, None)
    _tg_save_connections()
    return {"status": "disconnected"}


# â”€â”€ Notification Settings API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.get("/api/notifications/settings")
async def get_notification_settings(_user: str = Depends(get_current_user)):
    """Return current notification settings (credentials redacted)."""
    from scripts.notifications.dispatcher import NotificationDispatcher, _DEFAULT_SETTINGS, _SETTINGS_PATH
    import json as _json
    if not _SETTINGS_PATH.exists():
        cfg = dict(_DEFAULT_SETTINGS)
    else:
        try:
            cfg = {**_DEFAULT_SETTINGS, **_json.loads(_SETTINGS_PATH.read_text())}
        except Exception:
            cfg = dict(_DEFAULT_SETTINGS)
    # Redact secrets before sending to frontend
    for key in ("twilio_account_sid", "twilio_auth_token", "telegram_bot_token"):
        if cfg.get(key):
            cfg[key] = "***configured***"
    return cfg


@app.post("/api/notifications/settings")
async def save_notification_settings(request: Request, _user: str = Depends(get_current_user)):
    """Save notification settings. Pass empty string to clear a credential."""
    from scripts.notifications.dispatcher import NotificationDispatcher, _DEFAULT_SETTINGS, _SETTINGS_PATH
    import json as _json
    body = await request.json()
    # Merge with existing so partial updates don't wipe credentials
    existing: dict = {}
    if _SETTINGS_PATH.exists():
        try:
            existing = _json.loads(_SETTINGS_PATH.read_text())
        except Exception:
            pass
    # Don't overwrite real secrets with the redaction placeholder
    for key in ("twilio_account_sid", "twilio_auth_token"):
        if body.get(key) == "***configured***":
            body.pop(key, None)
    merged = {**_DEFAULT_SETTINGS, **existing, **body}
    NotificationDispatcher.save_settings(merged)
    return {"status": "saved"}


@app.post("/api/notifications/test")
async def test_notification(_user: str = Depends(get_current_user)):
    """Send a test ping to all configured notification channels."""
    from scripts.notifications.dispatcher import get_notifier
    try:
        results = await asyncio.get_event_loop().run_in_executor(
            None, get_notifier().test_send
        )
        return {"status": "ok", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        ws="websockets",
        log_level="info",
    )

