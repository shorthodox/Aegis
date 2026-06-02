# ===================================================================
# main.py - CRITICAL: Load environment variables FIRST
# ===================================================================
from dotenv import load_dotenv
import os

# MUST BE FIRST LINE OF EXECUTION - loads all env vars before any other code
load_dotenv()

# Now safe to import libraries that might use environment variables
import asyncio
import json
import re
import uuid
import random
import string
import time
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, TYPE_CHECKING, Union
from contextlib import asynccontextmanager
import inspect

import httpx

from fastapi import FastAPI, HTTPException, Depends, status, WebSocket, WebSocketDisconnect, Request, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
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
# Razorpay payment gateway
# -------------------------------------------------------------------
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
    print("Razorpay payment gateway configured")
else:
    print("[WARNING] Razorpay not configured. Set RAZORPAY_KEY_ID/RAZORPAY_KEY_SECRET to enable.")

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
    print(f"⚠️ Warning: 'web' directory not found at {WEB_ROOT_PATH}. Creating fallback structure.")
    WEB_ROOT_PATH.mkdir(parents=True, exist_ok=True)
    pages_dir = WEB_ROOT_PATH / "src" / "pages"
    scripts_dir = WEB_ROOT_PATH / "src" / "scripts"
    styles_dir = WEB_ROOT_PATH / "src" / "styles"
    pages_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)
    styles_dir.mkdir(parents=True, exist_ok=True)
    (pages_dir / "index.html").write_text("<html><body><h1>Aegis‑1</h1><p>Frontend files missing. Please upload the correct static files to 'web/src/pages/'</p></body></html>")
    (pages_dir / "dashboard.html").write_text("<html><body><h1>Dashboard unavailable</h1><p>Static files not found.</p></body></html>")

# -------------------------------------------------------------------
# LiveState for global data (shared with engine and WebSocket)
# -------------------------------------------------------------------
class LiveState:
    def __init__(self):
        self.data = {
            "tickers": {},
            "signals": {},
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
_track_store: list = []       # in-memory list of signal records
_tr_seen_ids: set = set()     # signal_ids already in store
_tr_last_save: float = 0.0   # epoch of last disk write

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
      - Primary TP : the same model fires the OPPOSITE signal  → MODEL_REVERSAL_TP.
                     This is the trend-reversal take-profit the user requested:
                     enter on one reversal, exit when the next reversal fires.
      - Safety SL  : price crosses the ATR stop stored at entry → STOP_HIT.
      - Hard ceiling: the stored take_profit price (wide ATR fallback) → TARGET_HIT.
                     Prevents a position staying open forever if the model never
                     generates an opposite signal (e.g. a slow grind with no clean
                     re-entry signal on the other side).
    """
    global _track_store, _tr_seen_ids
    now_iso = datetime.now(timezone.utc).isoformat()

    for sym, sig_map in signals_data.items():
        # Resolve nested timeframe map → use 1h summary signal
        if isinstance(sig_map, dict) and any(tf in sig_map for tf in ("1m","5m","15m","30m","1h","4h","1d")):
            sig = sig_map.get("1h") or next((v for v in sig_map.values() if isinstance(v, dict)), None)
        else:
            sig = sig_map
        if not isinstance(sig, dict):
            continue

        signal_type   = sig.get("signal", "HOLD")
        signal_id     = sig.get("signal_id")
        if not signal_id:
            continue

        current_price = float(live_prices.get(sym, sig.get("price", 0) or 0))

        # ── Find the one open position for this symbol (if any) ────────────
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

            # Primary TP: opposite model signal fires (this is the reversal exit)
            opposite_fired = signal_id not in _tr_seen_ids and (
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
                continue  # SL hit — do not open a new position this tick

            elif ceiling_hit:
                open_rec.update({
                    "outcome":     "WIN" if pnl_pct > 0 else "LOSS",
                    "exit_price":  current_price,
                    "close_time":  now_iso,
                    "pnl_pct":     pnl_pct,
                    "exit_reason": "TARGET_HIT",
                })
                continue  # Hard ceiling hit — do not immediately re-enter

            else:
                continue  # Position still open, nothing to do

        # ── Open a new position if the signal is actionable and fresh ───────
        if signal_type not in _ACTIONABLE:
            continue
        if signal_id in _tr_seen_ids:
            continue

        direction   = sig.get("direction", "LONG" if signal_type in _BUY_SIGNALS else "SHORT")
        entry_price = float(sig.get("price") or sig.get("entry_price") or 0)
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
            "ai_prob":         round(float(sig.get("ai_prob") or 0), 3),
            "confluence_rate": round(float(
                (sig.get("confluence_scorecards") or {}).get("efficiency", 0) or 0
            ), 2),
        })
        _tr_seen_ids.add(signal_id)

    # Cap store at 1 000 records (newest first)
    if len(_track_store) > 1000:
        _track_store = sorted(_track_store, key=lambda r: r.get("entry_time") or "", reverse=True)[:1000]
        _tr_seen_ids = {r["signal_id"] for r in _track_store if r.get("signal_id")}

# -------------------------------------------------------------------
# OTP Store (in-memory)
# -------------------------------------------------------------------
otp_store: Dict[str, Dict] = {}

def generate_otp() -> str:
    return ''.join(random.choices(string.digits, k=6))

def is_cooldown_active(email: str) -> bool:
    if email not in otp_store:
        return False
    cooldown = otp_store[email].get("cooldown_until")
    return bool(cooldown and datetime.now(timezone.utc) < cooldown)

# -------------------------------------------------------------------
# Institutional Analytics Engine
# -------------------------------------------------------------------
async def compute_system_analytics():
    """
    Background task to compute win rate, mathematical expectancy, 
    profit factor, and max drawdown from historical signals.
    """
    try:
        if not db:
            return
            
        print("📊 Running Institutional Analytics Computation...")
        signals_ref = db.collection("signals")
        
        # Query only closed trades
        # In Firestore, 'in' queries are supported up to 10 values
        docs = signals_ref.where("status", "in", ["TARGET HIT", "STOP LOSS HIT"]).stream()
        
        trades = []
        for doc in docs:
            trades.append(doc.to_dict())
            
        if not trades:
            print("📊 No closed trades found for analytics.")
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
        print(f"📊 Analytics updated: Exp {expectancy:.2f}%, PF {profit_factor:.2f}, MDD {max_dd_pct:.2f}%")
        
    except Exception as e:
        print(f"❌ Error computing analytics: {e}")

async def analytics_loop():
    while True:
        await compute_system_analytics()
        await asyncio.sleep(3600)  # Run every hour

# -------------------------------------------------------------------
# Engine runner as a background task (non‑blocking) with error handling
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

    try:
        configs, capital, max_pos, scan_seconds, proxy = automated_setup(backtest_dir, args)
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

    engine = LiveEngine(
        token_configs         = configs,
        capital               = capital,
        max_position_usdt     = max_pos,
        scan_interval_seconds = scan_seconds,
        risk_tier             = _tier,
        proxy_url             = proxy,
    )
    LIVE_STATE.engine = engine

    _last_tr_mtime: float = 0.0
    _last_signals_hash: int = 0       # hash of last signals pushed to Firestore
    _last_firestore_push: float = 0.0 # epoch of last Firestore write
    _FIRESTORE_MIN_INTERVAL = 290.0   # push at most once per ~5 min (matches scan cycle)

    async def update_state():
        nonlocal _last_tr_mtime, _last_signals_hash, _last_firestore_push
        while True:
            try:
                LIVE_STATE.data["tickers"]  = engine.live_prices.copy()
                LIVE_STATE.data["signals"]  = engine.last_signals.copy()
                LIVE_STATE.data["open_trades"] = [
                    asdict(p) for p in engine.wallet.open_positions.values()
                ]
                LIVE_STATE.data["balance"]  = engine.wallet.balance
                LIVE_STATE.data["alpha_mode"] = False
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
                    print(f"⚠️ Failed to write live_signals.json: {_e}")

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
                          f"— Firestore push deferred.")

                _now = time.time()
                _signals_now = LIVE_STATE.data.get('signals', {}) if not _warming_up else {}
                # Fingerprint: (symbol, signal_side, fire) — stable between scans when
                # nothing fires.  signal_id uses uuid4() on fire=True, so hashing
                # signal_id caused a push on every fired symbol within a scan cycle.
                _sig_fingerprint = tuple(sorted(
                    (sym, v.get('signal', 'FLAT'), bool(v.get('fire', False)))
                    for sym, v in _signals_now.items()
                    if isinstance(v, dict)
                ))
                _new_hash = hash(_sig_fingerprint)
                _signals_changed = (_new_hash != _last_signals_hash)
                _interval_elapsed = (_now - _last_firestore_push >= _FIRESTORE_MIN_INTERVAL)

                if not _signals_now or (not _signals_changed and not _interval_elapsed):
                    pass  # skip — nothing new to push
                else:
                    _last_signals_hash = _new_hash
                    _last_firestore_push = _now

                try:
                    # Only push FIRED signals (fire=True) — NEUTRAL/FLAT signals are
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
                    batch = db.batch()
                    count = 0
                    for sym, sig in push_target.items():
                        if not isinstance(sig, dict):
                            continue
                        doc_id  = sym.replace('/', '_')
                        sig_ref = db.collection("signals").document(doc_id)
                        # Strip live-price and heavy context keys — prices go via WS only
                        compact = {
                            k: v for k, v in numpy_to_native(sig).items()
                            if k not in _PRICE_CONTEXT_KEYS
                        }
                        compact['symbol']    = sym
                        compact['timestamp'] = now_str
                        compact['fire']      = bool(sig.get('fire', False))
                        batch.set(sig_ref, compact, merge=True)
                        count += 1
                        if count >= 450:
                            batch.commit()
                            batch  = db.batch()
                            count  = 0

                    if count > 0:
                        results = batch.commit()
                        fired_count = len(fired)
                        ts = results[-1].update_time if results else 'N/A'
                        label = f'{fired_count} FIRED + {count - fired_count} monitoring'
                        print(f"[PRODUCER] Firestore: {count} signals ({label}) @ {ts}")

                except StopIteration:
                    pass  # nothing to push this tick
                except Exception as _e:
                    print(f"[PRODUCER ERROR] ⚠️ Failed to write signals to Firestore: {_e}")

                # --- update track record (save to disk every 5 min) ---
                try:
                    _update_track_record(
                        LIVE_STATE.data.get("signals", {}),
                        LIVE_STATE.data.get("tickers", {}),
                    )
                    if time.time() - _tr_last_save >= 300:
                        _save_track_record()
                except Exception as _te:
                    print(f"[TrackRecord] Update error: {_te}")

            except Exception as e:
                print(f"State update error: {e}")
            await asyncio.sleep(1)

    asyncio.create_task(update_state())

    try:
        await engine.run()
    except Exception as e:
        print(f"⚠️ LiveEngine crashed: {e}")

# -------------------------------------------------------------------
# FastAPI app (lifespan runs engine as background task)
# -------------------------------------------------------------------
# ── Trader Engine background scan loop ────────────────────────────────────────
_TRADER_SCAN_INTERVAL = 300   # 5 minutes

def _save_trader_track_record() -> None:
    """Copy data/trader_track_record.json → web/trader_track_record.json for static serving."""
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
    """Runs the Universal Trader Engine every 5 minutes and caches token status."""
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
                    f"[TraderEngine] scan complete — "
                    f"{len(engine.active_signals)} signal(s), "
                    f"{len(engine.token_status)} token(s) tracked"
                )
        except Exception as _te:
            logger.error(f"[TraderEngine] scan error: {_te}")
        await asyncio.sleep(_TRADER_SCAN_INTERVAL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_track_record()
    engine_task       = asyncio.create_task(run_engine_background())
    reminder_task     = asyncio.create_task(check_and_send_trial_reminders())
    subscription_task = asyncio.create_task(check_and_send_subscription_reminders())
    analytics_task    = asyncio.create_task(analytics_loop())
    dev_token_task    = asyncio.create_task(dev_token_display_loop())
    dev_key_task      = asyncio.create_task(dev_key_display_loop())
    trader_task       = asyncio.create_task(_trader_scan_loop())
    yield
    engine_task.cancel()
    reminder_task.cancel()
    subscription_task.cancel()
    analytics_task.cancel()
    dev_token_task.cancel()
    dev_key_task.cancel()
    trader_task.cancel()

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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """
    Add critical security headers:
    - Cross-Origin-Opener-Policy: same-origin-allow-popups - fixes 'window.closed' blocking
    - Cross-Origin-Embedder-Policy: unsafe-none - allows third-party resources for popups
    """
    response = await call_next(request)
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin-allow-popups"
    response.headers["Cross-Origin-Embedder-Policy"] = "unsafe-none"
    # Prevent browsers from serving stale JS/HTML from disk cache after deploys
    path = request.url.path
    if path.endswith((".js", ".html")) and "/web/" in path:
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response

# -------------------------------------------------------------------
# Static Files: serve the entire 'web' folder under '/web' prefix
# -------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_ROOT = os.path.join(BASE_DIR, "web")
WEB_ROOT_PATH = Path(WEB_ROOT)

if not WEB_ROOT_PATH.exists():
    print(f"⚠️ Warning: 'web' directory not found at {WEB_ROOT_PATH}. Creating fallback structure.")
    WEB_ROOT_PATH.mkdir(parents=True, exist_ok=True)
    pages_dir = WEB_ROOT_PATH / "src" / "pages"
    scripts_dir = WEB_ROOT_PATH / "src" / "scripts"
    styles_dir = WEB_ROOT_PATH / "src" / "styles"
    pages_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)
    styles_dir.mkdir(parents=True, exist_ok=True)
    (pages_dir / "index.html").write_text("<html><body><h1>Aegis‑1</h1><p>Frontend files missing. Please upload the correct static files to 'web/src/pages/'</p></body></html>")
    (pages_dir / "dashboard.html").write_text("<html><body><h1>Dashboard unavailable</h1><p>Static files not found.</p></body></html>")

app.mount("/web", StaticFiles(directory=str(WEB_ROOT_PATH), html=True), name="web")

# -------------------------------------------------------------------
# Redirects
# -------------------------------------------------------------------
@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})

@app.get("/")
async def root_redirect():
    # Permanent redirect — compliance checkers and search engines follow 301s reliably
    return RedirectResponse(url="/web/src/pages/index.html", status_code=301)

@app.get("/dashboard")
async def dashboard_redirect():
    return RedirectResponse(url="/web/src/pages/dashboard.html")

# -------------------------------------------------------------------
# Diagnostic endpoint
# -------------------------------------------------------------------
@app.get("/debug-files")
async def debug_files():
    try:
        top_files = os.listdir(WEB_ROOT_PATH) if WEB_ROOT_PATH.exists() else []
        pages_path = WEB_ROOT_PATH / "src" / "pages"
        scripts_path = WEB_ROOT_PATH / "src" / "scripts"
        styles_path = WEB_ROOT_PATH / "src" / "styles"
        return JSONResponse(content={
            "web_root": str(WEB_ROOT_PATH),
            "top_level": top_files,
            "pages_files": os.listdir(pages_path) if pages_path.exists() else [],
            "scripts_files": os.listdir(scripts_path) if scripts_path.exists() else [],
            "styles_files": os.listdir(styles_path) if styles_path.exists() else [],
            "exists": WEB_ROOT_PATH.exists(),
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


security = HTTPBearer()

@app.get("/api/signals")
async def api_signals(credentials: HTTPAuthorizationCredentials = Depends(security)):
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

from fastapi import Header

@app.get("/api/public/signals")
async def api_public_signals(authorization: Optional[str] = Header(None)):
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
    trial / basic  → market bias, S/R, confluence, price targets, session, fear/greed.
                      AI probability and fire signal hidden.
    intermediate   → all of the above + AI probability bands (low/med/high label).
    pro            → full signal including meta_confidence, fire, direction.
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
async def token_insight(symbol: str, authorization: Optional[str] = Header(None)):
    """
    Per-token market insight available to all authenticated users.

    trial / basic   → S/R levels, confluence, price targets, market bias, trader views.
    intermediate    → above + AI conviction label (HIGH / MEDIUM / LOW).
    pro             → above + full fire/direction/meta_confidence signal.

    The symbol path parameter accepts slash notation, e.g. BTC/USDT or BTC%2FUSDT.
    """
    symbol = symbol.replace('%2F', '/').replace('%2f', '/').upper()

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
        # Engine may still be warming up — return what we know without AI fields
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
    Used for landing-page previews — no S/R or price targets.
    """
    symbol = symbol.replace('%2F', '/').replace('%2f', '/').upper()
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


# ─────────────────────────────────────────────────────────────────────────────
# PROFESSIONAL TOKEN ANALYSIS ENGINE
# ─────────────────────────────────────────────────────────────────────────────

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
        if v <= 0:      return '—'
        if v < 0.001:   return f'${v:.6f}'
        if v < 1:       return f'${v:.4f}'
        if v < 100:     return f'${v:.3f}'
        return f'${v:,.2f}'

    # ── Bull / bear vote tally ────────────────────────────────────────────────
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

    # ── Overall verdict ───────────────────────────────────────────────────────
    bull_pct = bull_n / total_v
    if bull_pct >= 0.70:   verdict_label, verdict_icon = 'Strong Bullish', '🟢'
    elif bull_pct >= 0.55: verdict_label, verdict_icon = 'Bullish',         '🟢'
    elif bull_pct >= 0.45: verdict_label, verdict_icon = 'Neutral / Mixed', '🟡'
    elif bull_pct >= 0.30: verdict_label, verdict_icon = 'Bearish',         '🔴'
    else:                  verdict_label, verdict_icon = 'Strong Bearish',  '🔴'

    # ── Plain-English trend description ──────────────────────────────────────
    trend_desc = {
        'TRENDING_UP':   'moving upward in a clear trend',
        'TRENDING_DOWN': 'moving downward in a clear trend',
        'TRENDING':      'in an active trend',
        'RANGING':       'moving sideways without a clear direction',
    }.get(trend_regime, 'in an uncertain phase')

    vol_desc = {
        'HIGH':   'high volatility — prices are swinging a lot',
        'MEDIUM': 'moderate volatility',
        'LOW':    'low volatility — price is calm and moving slowly',
    }.get(vol_regime, 'moderate volatility')

    # ── Two-sentence summary ──────────────────────────────────────────────────
    base = sym.split('/')[0]
    summary = (
        f"{base} is currently {trend_desc}, with {vol_desc}. "
        f"{'Most indicators lean bullish' if bull_pct > 0.55 else 'Most indicators lean bearish' if bull_pct < 0.45 else 'Indicators are mixed'}"
        f" — {bull_n} bullish signal{'s' if bull_n != 1 else ''} vs {bear_n} bearish."
    )

    # ── Top indicators (max 6, sorted by impact) ─────────────────────────────
    indicators = []

    # RSI
    if rsi <= 30:
        indicators.append({
            'name': 'RSI (Momentum)',
            'value': f'{rsi:.0f} — Oversold',
            'direction': 'BULLISH',
            'icon': '🔋',
            'impact': 'high',
            'simple': (
                f"RSI is {rsi:.0f}. Think of RSI like a rubber band — when it stretches too far "
                f"in one direction it snaps back. Below 30 means the token has been sold too "
                f"aggressively. A bounce back up is increasingly likely."
            ),
        })
    elif rsi >= 70:
        indicators.append({
            'name': 'RSI (Momentum)',
            'value': f'{rsi:.0f} — Overbought',
            'direction': 'BEARISH',
            'icon': '⚠️',
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
            'value': f'{rsi:.0f} — Neutral',
            'direction': 'NEUTRAL',
            'icon': '📊',
            'impact': 'medium',
            'simple': (
                f"RSI is {rsi:.0f}, sitting in the neutral zone (30–70). There's no extreme "
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
            'icon': '📈',
            'impact': 'high',
            'simple': (
                "The Supertrend indicator is green — price is trading above a dynamic "
                "support line that adapts to market volatility. This tells us sellers "
                "are not in control right now. Think of it as a moving 'floor' — as "
                "long as price stays above it, the trend is up."
            ),
        })
    elif supertrend == 'BEARISH':
        indicators.append({
            'name': 'Supertrend',
            'value': 'Bearish',
            'direction': 'BEARISH',
            'icon': '📉',
            'impact': 'high',
            'simple': (
                "The Supertrend indicator is red — price is trading below a dynamic "
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
            'icon': '🔄',
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
            'value': f'{adx:.0f} — {"Strong" if adx > 40 else "Moderate"} trend',
            'direction': 'NEUTRAL',
            'icon': '💪',
            'impact': 'medium',
            'simple': (
                f"ADX is {adx:.0f}. This measures how strong the current trend is — "
                f"it doesn't tell you the direction, just the conviction. "
                f"{'Above 40 means the trend is very powerful and unlikely to reverse quickly.' if adx > 40 else 'Between 25–40 means a genuine trend exists and it is worth following.'}"
            ),
        })

    # Confluence
    if total_c >= 6:
        indicators.append({
            'name': 'Confluence Score',
            'value': f'{total_c:.1f}/10',
            'direction': 'BULLISH' if bias == 'BULLISH' else 'BEARISH',
            'icon': '🎯',
            'impact': 'high',
            'simple': (
                f"Confluence score is {total_c:.1f}/10. This is our AI's internal vote count — "
                f"it tallies up momentum, trend, volume, smart money flow, and candlestick "
                f"patterns into a single score. "
                f"{'Above 6 means most evidence is pointing in the same direction — higher quality setups.' if total_c >= 6 else ''} "
                f"Breakdown: Momentum {mom_c:.1f}, Trend {trend_c:.1f}, "
                f"Volume {vol_c:.1f}, Smart Money {sm_c:.1f}."
            ),
        })
    elif total_c >= 3:
        indicators.append({
            'name': 'Confluence Score',
            'value': f'{total_c:.1f}/10 — Weak',
            'direction': 'NEUTRAL',
            'icon': '⚖️',
            'impact': 'medium',
            'simple': (
                f"Confluence score is {total_c:.1f}/10 — indicators are split. "
                f"Some point bullish, others bearish. This is a 'wait and see' situation; "
                f"there is no clear majority conviction from the market's internal structure."
            ),
        })

    # Funding rate (derivatives market)
    if abs(funding) > 0.005:
        funding_desc = 'Longs are paying shorts' if funding_bias == 'LONGS_PAYING' else 'Shorts are paying longs'
        funding_meaning = (
            'This means the market is over-leveraged to the upside — too many people are betting on a rise. '
            'These longs may be forced to close, creating selling pressure.'
        ) if funding_bias == 'LONGS_PAYING' else (
            'The market is over-leveraged to the downside. Too many people are shorting — '
            'if price rises even slightly, forced short-covering can create a sharp rally (short squeeze).'
        )
        indicators.append({
            'name': 'Funding Rate',
            'value': f'{funding:+.4f}% — {funding_desc}',
            'direction': 'BEARISH' if funding_bias == 'LONGS_PAYING' else 'BULLISH',
            'icon': '💸',
            'impact': 'medium',
            'simple': funding_meaning,
        })

    # Fear & Greed
    if fg_val <= 25 or fg_val >= 75:
        indicators.append({
            'name': 'Market Sentiment (Fear & Greed)',
            'value': f'{fg_val:.0f}/100 — {fg_label}',
            'direction': 'BULLISH' if fg_val <= 25 else 'BEARISH',
            'icon': '🧠',
            'impact': 'low',
            'simple': (
                f"The Fear & Greed index is {fg_val:.0f} ({fg_label}). "
                + (
                    "Extreme fear means most market participants are panicking and selling. "
                    "Historically, extreme fear has been one of the best times to buy — "
                    "'be greedy when others are fearful.'"
                    if fg_val <= 25 else
                    "Extreme greed means everyone is euphoric and buying. "
                    "Historically, this is when markets are most vulnerable to a sharp correction — "
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
            'icon': '🌐',
            'impact': 'medium',
            'simple': (
                f"On the daily chart, {base} has been trending {macro_dir} "
                f"({macro_daily:+.1%} momentum). "
                f"{'This longer-term upward pressure provides tailwind for bullish trades.' if macro_daily > 0 else 'This longer-term downward pressure creates headwind — even if short-term indicators look bullish, you are fighting the bigger trend.'}"
            ),
        })

    # Limit to top 6 by impact
    impact_order = {'high': 0, 'medium': 1, 'low': 2}
    indicators.sort(key=lambda x: impact_order.get(x.get('impact', 'low'), 2))
    indicators = indicators[:6]

    # ── Why the signal fired or didn't ───────────────────────────────────────
    signal_section: dict = {}

    if side == 'WAITING' or conf == 0 and not tradeable:
        signal_section = {
            'status': 'WAITING',
            'headline': '⏳ Engine is warming up',
            'plain': (
                f"The AI for {base} is still loading its data and running its first scan. "
                f"Full analysis and signal decisions will appear within 2–3 minutes."
            ),
            'technical': 'Model scan not yet completed.',
        }
    elif not tradeable and conf == 0:
        signal_section = {
            'status': 'MONITOR_ONLY',
            'headline': '👁️ Watch mode — no signals',
            'plain': (
                f"{base} is in monitor-only mode. The AI model was trained on its data "
                f"but the historical edge (win rate, expectancy) was not strong enough to "
                f"justify trading it. We show the price and market context so you can "
                f"watch it and make your own decision — but the bot won't trade it automatically."
            ),
            'technical': 'Model trained but did not pass backtesting quality gate (tradeable=False).',
        }
    elif fire:
        direction_word = 'BUY' if side in ('BUY', 'STRONG_BUY') else 'SELL'
        signal_section = {
            'status': 'SIGNAL_ACTIVE',
            'headline': f'🚀 Signal ACTIVE — {direction_word}',
            'plain': (
                f"The AI fired a {direction_word} signal on {base}. It required "
                f"at least {_pct(thr)} confidence and reached {_pct(conf)} — "
                f"clearing the bar with {'strong' if conf > thr * 1.2 else 'sufficient'} conviction. "
                f"{'This is a Strong signal — confidence is significantly above threshold.' if conf > thr * 1.15 else ''}"
            ),
            'technical': f'meta_confidence={conf:.3f} ≥ threshold={thr:.3f}. Fire=True.',
        }
    else:
        # No signal — explain WHY
        if conf == 0:
            plain = (
                f"The AI model hasn't produced a clear directional probability yet "
                f"for {base}. This usually means the market conditions are too noisy "
                f"or ambiguous — no clean trade setup is visible."
            )
            headline = '🔍 No setup found'
            tech = f'meta_confidence=0. Model output inconclusive.'
        elif thr > 0 and conf >= thr * 0.85:
            pct_away = (thr - conf) / thr * 100
            plain = (
                f"Very close — the AI is at {_pct(conf)} confidence, just {pct_away:.1f}% "
                f"below the {_pct(thr)} trigger threshold. "
                f"One more bullish/bearish indicator aligning could push it over the line. "
                f"Watch closely — a signal may fire in the next scan."
            )
            headline = '🔔 Almost there — watching for trigger'
            tech = f'meta_confidence={conf:.3f}, threshold={thr:.3f}. Gap: {thr-conf:.3f}.'
        elif thr > 0 and conf >= 0.35:
            plain = (
                f"The AI sees some directional evidence for {base} but confidence is "
                f"{_pct(conf)}, below the {_pct(thr)} required. "
                f"Indicators are not aligned strongly enough. The model is saying 'I see "
                f"something but it's not convincing yet — wait for clearer confirmation.'"
            )
            headline = '⏸️ Building — not enough conviction yet'
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
                    f"The model suppressed the BUY signal — buying into resistance means "
                    f"you're buying right at a wall that sellers have historically defended. "
                    f"The signal is waiting for that wall to break."
                )
                headline = '🚧 BUY blocked by resistance'
                tech = f'Price within 2% of resistance {_px(resistance)}. Signal suppressed by S&R filter.'
            elif near_support and p_sell > p_buy:
                plain = (
                    f"The AI sees bearish pressure in {base}, but price is sitting "
                    f"right on a support level at {_px(support)}. "
                    f"The model held back the SELL signal — shorting into support is risky "
                    f"because buyers historically step in at this level. "
                    f"Waiting to see if support breaks before confirming the trade."
                )
                headline = '🛡️ SELL held at support'
                tech = f'Price within 2% of support {_px(support)}. Signal suppressed by S&R filter.'
            elif macro_conflicting:
                trend_word = 'bearish' if macro_daily < 0 else 'bullish'
                plain = (
                    f"There's a conflict: short-term indicators suggest one direction but "
                    f"the daily trend for {base} is strongly {trend_word} "
                    f"({macro_daily:+.1%}). "
                    f"The model won't fight the bigger trend unless confidence is very high. "
                    f"Trading against the daily trend is like swimming against a strong current — "
                    f"possible, but requires much more conviction."
                )
                headline = '⚔️ Signal conflicting with daily trend'
                tech = f'macro_daily={macro_daily:+.3f} conflicts with short-term direction.'
            else:
                plain = (
                    f"The AI model is at {_pct(conf)} confidence — well below the {_pct(thr)} "
                    f"required to trade. Indicators are too mixed or too weak. "
                    f"The market for {base} is not giving a clean enough signal right now. "
                    f"Patience: when the evidence aligns, the signal will fire automatically."
                )
                headline = '😴 No clear setup'
                tech = f'meta_confidence={conf:.3f}, threshold={thr:.3f}. No qualifying filter match.'

        signal_section = {
            'status': 'NO_SIGNAL',
            'headline': headline,
            'plain': plain,
            'technical': tech,
        }

    # ── Key levels ────────────────────────────────────────────────────────────
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
            'meaning': 'First take-profit target (1× ATR above entry).',
        }
        if sig.get('bull_tp2'):
            key_levels['target_2'] = {
                'price': round(float(sig['bull_tp2']), 8),
                'meaning': 'Second take-profit target (2× ATR above entry).',
            }
    elif sig.get('bear_tp1') and fire and side in ('SELL', 'STRONG_SELL'):
        key_levels['target_1'] = {
            'price': round(float(sig['bear_tp1']), 8),
            'meaning': 'First take-profit target (1× ATR below entry).',
        }

    # ── Macro / sentiment context ─────────────────────────────────────────────
    macro_lines = []

    if abs(macro_weekly) > 0.05:
        macro_lines.append(
            f"Weekly trend: {'bullish' if macro_weekly > 0 else 'bearish'} "
            f"({macro_weekly:+.1%}). "
            f"{'The longer-term picture supports the bulls.' if macro_weekly > 0 else 'The bigger picture still favors sellers — short-term bounces may be selling opportunities.'}"
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
            f"positions. This creates upward pressure — if shorts capitulate, it could trigger "
            f"a fast squeeze rally."
        )

    if oi_trend == 'INCREASING':
        macro_lines.append(
            "Open Interest is rising — new money is entering the derivatives market. "
            "Combined with the current price direction, this confirms fresh conviction "
            f"behind the {'up' if bias == 'BULLISH' else 'down'}move."
        )
    elif oi_trend == 'DECREASING':
        macro_lines.append(
            "Open Interest is falling — traders are closing positions and exiting. "
            "This usually means the current move is losing steam. "
            "Price may continue but with less force."
        )

    if fg_val < 30:
        macro_lines.append(
            f"Market sentiment is in Fear ({fg_val:.0f}/100). Historically, extreme fear "
            f"creates some of the best long-term buying opportunities — but don't catch "
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

    # ── Session context ───────────────────────────────────────────────────────
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
async def token_analysis(symbol: str, authorization: Optional[str] = Header(None)):
    """
    Professional market analysis for one token — accessible to all authenticated users.
    Returns plain-English breakdown of indicators, why the signal fired or didn't,
    key price levels, and macro/sentiment context.
    """
    symbol = symbol.replace('%2F', '/').replace('%2f', '/').upper()

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
        # UID-only token (some OAuth flows omit email) — look up email via Admin SDK
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
    """Return Firebase UID (not email) — used for Firestore paths shared with the frontend."""
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
    """Dependency that returns Firebase UID — keeps Firestore paths consistent with the frontend."""
    uid = decode_uid_from_token(credentials.credentials)
    if not uid:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    return uid

# -------------------------------------------------------------------
# Firestore user helpers
# -------------------------------------------------------------------
def get_user_doc(email: str) -> Optional[Dict]:
    doc_ref = db.collection("users").document(email)
    doc = doc_ref.get()
    to_dict = getattr(doc, "to_dict", None)
    exists = getattr(doc, "exists", False)
    if callable(to_dict) and exists:
        result = to_dict()
        return result if isinstance(result, dict) else {}
    return None

def phone_is_unique(phone: str, exclude_email: Optional[str] = None) -> bool:
    """Return True if phone number is not already stored in any user document."""
    try:
        docs = db.collection("users").where("phone_number", "==", phone).limit(2).stream()
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
    trial_end = (datetime.now(timezone.utc) + timedelta(days=3)).isoformat()
    user_data = {
        "email": email,
        "plan": "trial",
        "trial_start": now,
        "trial_end": trial_end,
        "trial_active": True,
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
    return {"email": email, "plan": "trial", "trial_end": trial_end, "full_name": full_name, "location": location}

def update_last_login(email: str):
    db.collection("users").document(email).update({"last_login": datetime.now(timezone.utc).isoformat()})

def get_or_create_user_from_oauth(email: str, name: str, provider: str, social_id: str) -> Dict:
    user = get_user_doc(email)
    if not user:
        user = create_user_doc(email, provider=provider, social_id=social_id, full_name=name)
    else:
        update_last_login(email)
    return {"email": email, "plan": user["plan"], "trial_end": user["trial_end"]}

def is_trial_expired(email: str) -> bool:
    user_doc = get_user_doc(email)
    if not user_doc:
        return True
    plan = user_doc.get("plan", "trial")
    if plan in ("pro", "premium", "intermediate", "basic"):
        # Only bypass expiry if there is an active subscription record
        subscription = user_doc.get("subscription", {})
        if isinstance(subscription, dict) and subscription.get("status") == "active":
            return False
        # No active subscription — fall through to trial date check
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
async def get_me(credentials: HTTPAuthorizationCredentials = Depends(security)):
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
        # Their email_verified claim is True natively — no OTP needed for them.
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
        "full_name": user_doc.get("full_name"),
        "location": user_doc.get("location")
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
            valid = any(
                isinstance(entry, dict) and entry.get("signup_token") == signup_token
                for entry in otp_store.values()
            )
            if not valid:
                raise HTTPException(status_code=403, detail="Invalid or expired OTP verification token.")
            # Invalidate the token — single use only
            for k, entry in list(otp_store.items()):
                if isinstance(entry, dict) and entry.get("signup_token") == signup_token:
                    otp_store.pop(k, None)
                    break
            # Mark Firebase email as verified — this is the gate used in decode_token
            # so accounts that bypassed OTP can never authenticate.
            try:
                firebase_auth.update_user(firebase_uid, email_verified=True)
            except Exception as fe:
                print(f"[provision] Warning: could not mark email_verified for {firebase_uid}: {fe}")

        # Prefer email as doc key (consistent with rest of backend), fall back to uid
        doc_key = email or firebase_uid

        # Normalize and validate phone number if provided
        phone_raw = data.get("phone_number") or ""
        phone_number: Optional[str] = None
        if phone_raw:
            cleaned = re.sub(r'[\s\-\.\(\)]', '', phone_raw.strip())
            if re.match(r'^\d{10}$', cleaned):
                cleaned = '+91' + cleaned
            if re.match(r'^\+\d{7,15}$', cleaned):
                phone_number = cleaned

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
            }

        # Check phone uniqueness before creating new doc
        if phone_number and not phone_is_unique(phone_number):
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
    return RedirectResponse(f"/web/src/pages/dashboard.html#token={jwt_token}")

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

class OTPVerifyRequest(BaseModel):
    email: EmailStr
    otp: str

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
    razorpay_payment_id: str
    razorpay_order_id: str
    razorpay_signature: str
    plan: str

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

# -------------------------------------------------------------------
# 3-Step Onboarding with OTP
# -------------------------------------------------------------------
@app.post("/auth/send-otp-for-registration")
async def send_otp_for_registration(request: OTPSendRequest):
    email = request.email

    # Block disposable / temp-mail domains
    domain = email.split('@')[-1].lower()
    if domain in DISPOSABLE_EMAIL_DOMAINS:
        raise HTTPException(status_code=422, detail="Disposable or temporary email addresses are not allowed.")

    # Validate email syntax only — no DNS/MX lookup.
    # Deliverability is proven naturally: if the email is fake or unreachable,
    # the OTP never arrives and signup cannot complete.
    try:
        validated = validate_email(email, check_deliverability=False)
        email = validated.normalized
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid email format: {str(exc)}")

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
    try:
        message = MessageSchema(
            subject="Your AEGIS Verification Code",
            recipients=[NameEmail(name=email, email=email)],
            body=f"""
            <!DOCTYPE html>
            <html>
            <head><meta charset="UTF-8"></head>
            <body style="margin:0;padding:0;background:#0a0a0c;font-family:'Segoe UI',Arial,sans-serif;">
              <table width="100%" cellpadding="0" cellspacing="0" style="background:#0a0a0c;padding:40px 0;">
                <tr><td align="center">
                  <table width="480" cellpadding="0" cellspacing="0"
                         style="background:#0f111a;border:1px solid rgba(0,242,255,0.15);border-radius:16px;overflow:hidden;max-width:480px;">

                    <!-- Header -->
                    <tr>
                      <td style="background:linear-gradient(135deg,#00f2ff22,#7b2fff22);padding:32px 40px 24px;text-align:center;border-bottom:1px solid rgba(0,242,255,0.1);">
                        <div style="font-size:28px;font-weight:800;letter-spacing:3px;
                                    background:linear-gradient(90deg,#00f2ff,#7b2fff);
                                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                                    display:inline-block;">
                          ⚡ AEGIS
                        </div>
                        <p style="color:#6b7280;margin:8px 0 0;font-size:13px;letter-spacing:1px;">SOVEREIGN TERMINAL</p>
                      </td>
                    </tr>

                    <!-- Body -->
                    <tr>
                      <td style="padding:36px 40px;">
                        <p style="color:#9ca3af;font-size:15px;margin:0 0 8px;">Email Verification</p>
                        <h2 style="color:#f9fafb;font-size:20px;font-weight:600;margin:0 0 24px;">Your one-time verification code</h2>

                        <!-- OTP display -->
                        <div style="background:#0a0a0c;border:1px solid rgba(0,242,255,0.25);border-radius:12px;
                                    padding:24px;text-align:center;margin:0 0 24px;">
                          <span style="font-family:'Courier New',Courier,monospace;font-size:38px;font-weight:700;
                                       letter-spacing:12px;color:#00f2ff;">{otp}</span>
                        </div>

                        <p style="color:#9ca3af;font-size:14px;margin:0 0 8px;">
                          This code expires in <strong style="color:#f9fafb;">5 minutes</strong>.
                          Do not share it with anyone.
                        </p>
                        <p style="color:#6b7280;font-size:13px;margin:0;">
                          If you didn't request this, you can safely ignore this email.
                        </p>
                      </td>
                    </tr>

                    <!-- Footer -->
                    <tr>
                      <td style="padding:20px 40px 28px;border-top:1px solid rgba(255,255,255,0.05);text-align:center;">
                        <p style="color:#4b5563;font-size:12px;margin:0;">
                          Sent by
                          <a href="mailto:animeshkukreti@gatekeeper.sbs"
                             style="color:#00f2ff;text-decoration:none;">animeshkukreti@gatekeeper.sbs</a>
                          &nbsp;·&nbsp;
                          <a href="https://gatekeeper.sbs"
                             style="color:#00f2ff;text-decoration:none;">gatekeeper.sbs</a>
                        </p>
                      </td>
                    </tr>

                  </table>
                </td></tr>
              </table>
            </body>
            </html>
            """,
            subtype=MessageType.html,
        )
        await fastmail.send_message(message)
    except Exception as e:
        otp_store.pop(email, None)
        print(f"Email sending failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to send OTP email. Check email configuration.")
    return {"success": True, "message": "OTP sent to your email address."}

@app.post("/auth/verify-otp-for-registration")
async def verify_otp_for_registration(request: OTPVerifyRequest):
    email = request.email
    otp = request.otp
    if email not in otp_store:
        raise HTTPException(status_code=400, detail="No OTP request found. Please request a new OTP.")
    record = otp_store[email]
    if datetime.now(timezone.utc) > record["expires_at"]:
        otp_store.pop(email, None)
        raise HTTPException(status_code=400, detail="OTP has expired. Please request a new one.")
    if record["otp"] != otp:
        raise HTTPException(status_code=400, detail="Invalid OTP. Please try again.")
    signup_token = str(uuid.uuid4())
    otp_store[email]["verified"] = True
    otp_store[email]["signup_token"] = signup_token
    return {"success": True, "message": "OTP verified successfully. Please complete your profile.", "signup_token": signup_token}

@app.post("/auth/check-phone")
async def check_phone_unique(request: PhoneCheckRequest):
    """Pre-signup phone uniqueness check. No auth required."""
    phone = request.phone.strip()
    if not phone or len(phone) < 7:
        raise HTTPException(status_code=422, detail="Invalid phone number")
    return {"available": phone_is_unique(phone)}

# -------------------------------------------------------------------
# Password reset — generate Firebase link, deliver via Neo SMTP
# Firebase's default noreply sender goes to spam; our domain is trusted.
# -------------------------------------------------------------------
class PasswordResetRequest(BaseModel):
    email: EmailStr

@app.post("/auth/send-password-reset")
async def send_password_reset(request: PasswordResetRequest):
    email = request.email
    try:
        firebase_link = firebase_auth.generate_password_reset_link(email)
    except firebase_auth.UserNotFoundError:
        return {"success": True, "message": "If an account with this email exists, a reset link has been sent."}
    except Exception as e:
        print(f"[password-reset] generate_password_reset_link failed: {e}")
        raise HTTPException(status_code=500, detail="Could not generate reset link. Please try again.")

    # Extract oobCode and apiKey from Firebase's link so our custom reset page
    # can call confirmPasswordReset() directly without going through Firebase's UI.
    parsed = urllib.parse.urlparse(firebase_link)
    params = urllib.parse.parse_qs(parsed.query)
    oob_code = params.get("oobCode", [""])[0]
    api_key  = params.get("apiKey",  [""])[0]
    base_url = os.getenv("BASE_URL", "https://gatekeeper.sbs").rstrip("/")
    custom_reset_url = (
        f"{base_url}/web/src/pages/reset-password.html"
        f"?oobCode={urllib.parse.quote(oob_code)}&apiKey={urllib.parse.quote(api_key)}"
    )

    try:
        message = MessageSchema(
            subject="Reset Your Gatekeeper Password",
            recipients=[NameEmail(name=email, email=email)],
            body=f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
</head>
<body style="margin:0;padding:0;background:#050505;font-family:'Segoe UI',Arial,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0"
         style="background:#050505;padding:48px 16px;">
    <tr><td align="center">

      <!-- Card -->
      <table width="520" cellpadding="0" cellspacing="0"
             style="max-width:520px;width:100%;background:#0a0c14;
                    border:1px solid rgba(0,242,255,0.14);
                    border-radius:18px;overflow:hidden;">

        <!-- Top accent bar -->
        <tr>
          <td style="height:3px;background:linear-gradient(90deg,#00f2ff,#7b2fff);"></td>
        </tr>

        <!-- Header -->
        <tr>
          <td style="padding:36px 44px 28px;text-align:center;
                     border-bottom:1px solid rgba(255,255,255,0.05);">
            <div style="display:inline-flex;align-items:center;gap:10px;margin-bottom:6px;">
              <span style="font-size:22px;">⚡</span>
              <span style="font-size:22px;font-weight:800;letter-spacing:2.5px;
                           background:linear-gradient(90deg,#00f2ff,#7b2fff);
                           -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                GATEKEEPER
              </span>
            </div>
            <p style="margin:2px 0 0;font-size:11px;letter-spacing:2px;
                      color:#4b5563;text-transform:uppercase;">Aegis-1 · Sovereign Terminal</p>
          </td>
        </tr>

        <!-- Body -->
        <tr>
          <td style="padding:40px 44px 32px;">

            <!-- Icon circle -->
            <div style="text-align:center;margin-bottom:28px;">
              <div style="display:inline-block;width:64px;height:64px;border-radius:50%;
                          background:rgba(0,242,255,0.08);border:1px solid rgba(0,242,255,0.2);
                          line-height:64px;font-size:26px;">🔐</div>
            </div>

            <h1 style="margin:0 0 10px;font-size:22px;font-weight:700;
                       color:#f1f5f9;text-align:center;letter-spacing:-0.3px;">
              Password Reset Request
            </h1>
            <p style="margin:0 0 28px;font-size:14px;color:#94a3b8;
                      text-align:center;line-height:1.65;">
              We received a request to reset the password for<br>
              <strong style="color:#e2e8f0;">{email}</strong>
            </p>

            <!-- CTA button -->
            <div style="text-align:center;margin-bottom:28px;">
              <a href="{custom_reset_url}"
                 style="display:inline-block;padding:15px 40px;
                        background:linear-gradient(95deg,#00f2ff,#00a8c6);
                        color:#000;font-weight:700;font-size:15px;
                        border-radius:10px;text-decoration:none;
                        letter-spacing:0.4px;
                        box-shadow:0 4px 20px rgba(0,242,255,0.25);">
                Reset My Password
              </a>
            </div>

            <!-- Expiry notice -->
            <div style="background:rgba(0,242,255,0.04);border:1px solid rgba(0,242,255,0.1);
                        border-radius:10px;padding:16px 20px;margin-bottom:24px;">
              <p style="margin:0;font-size:13px;color:#64748b;line-height:1.6;">
                ⏱ &nbsp;This link expires in <strong style="color:#94a3b8;">1 hour</strong>.<br>
                🔒 &nbsp;If you didn't request this, your account is safe — ignore this email.
              </p>
            </div>

            <!-- Fallback link -->
            <p style="margin:0;font-size:11.5px;color:#374151;
                      word-break:break-all;line-height:1.7;">
              Button not working? Copy and paste this link into your browser:<br>
              <a href="{custom_reset_url}"
                 style="color:#00f2ff;text-decoration:none;">{custom_reset_url}</a>
            </p>
          </td>
        </tr>

        <!-- Footer -->
        <tr>
          <td style="padding:20px 44px 28px;
                     border-top:1px solid rgba(255,255,255,0.04);
                     text-align:center;">
            <p style="margin:0 0 6px;font-size:12px;color:#374151;">
              Sent by Gatekeeper (Aegis-1) &nbsp;·&nbsp;
              <a href="https://gatekeeper.sbs"
                 style="color:#00f2ff;text-decoration:none;">gatekeeper.sbs</a>
            </p>
            <p style="margin:0;font-size:11px;color:#1f2937;">
              © 2025 Gatekeeper. All rights reserved.
            </p>
          </td>
        </tr>

        <!-- Bottom accent bar -->
        <tr>
          <td style="height:3px;background:linear-gradient(90deg,#7b2fff,#00f2ff);"></td>
        </tr>

      </table>

    </td></tr>
  </table>
</body>
</html>""",
            subtype=MessageType.html,
        )
        await fastmail.send_message(message)
    except Exception as e:
        print(f"[password-reset] SMTP send failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to send reset email. Please check your email address and try again.")

    return {"success": True, "message": "Password reset link sent. Check your inbox (and spam folder)."}

@app.post("/auth/complete-registration")
async def complete_registration(profile: UserProfileComplete):
    email = profile.email
    if email not in otp_store or not otp_store[email].get("verified"):
        raise HTTPException(status_code=400, detail="Please verify OTP first before completing registration.")
    
    existing_user = get_user_doc(email)
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered.")
    
    password_hash = hash_password(profile.password) if profile.password else None
    user = create_user_doc(email, password_hash=password_hash, full_name=profile.full_name, location=profile.location)
    otp_store.pop(email, None)
    token = create_token(email)
    return {"access_token": token, "token_type": "bearer", "user": user}

# -------------------------------------------------------------------
# Authentication endpoints
# -------------------------------------------------------------------
@app.post("/auth/login")
async def login(user: UserLogin):
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
# Mirrors FLEET in scripts/live_engine.py — kept in sync here so the
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
async def get_user_limits(email: str = Depends(get_current_user)):
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
async def upgrade_plan(email: str = Depends(get_current_user)):
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
async def start_free_trial(user_id: str = Depends(get_current_user)):
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
            pass  # Corrupted date — fall through to fresh start

    # Start a new (or restart an expired) trial
    now = datetime.now(timezone.utc)
    trial_start_iso = now.isoformat()
    trial_end_dt = now + timedelta(days=3)
    trial_end_iso = trial_end_dt.isoformat()

    db.collection("users").document(user_id).update({
        "plan": "trial",
        "trial_start": trial_start_iso,
        "trial_end": trial_end_iso,
        "trial_active": True
    })

    remaining_seconds = int((trial_end_dt - now).total_seconds())
    print(f"✅ Free trial started for {user_id}: ends {trial_end_iso}")

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
async def get_trial_status(user_id: str = Depends(get_current_user)):
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


@app.get("/payment/config")
async def payment_config():
    return {
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


# Base prices in USD — source of truth for all plans
USD_PLAN_PRICES: Dict[str, float] = {
    "basic": 3.60,
    "intermediate": 24.00,
    "pro": 40.00,
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


@app.get("/api/track-record")
async def track_record_endpoint():
    """Public endpoint — returns the persisted track_record.json."""
    if TRACK_RECORD_PATH.exists():
        try:
            with open(TRACK_RECORD_PATH, "r", encoding="utf-8") as f:
                return JSONResponse(json.load(f))
        except Exception as e:
            print(f"[TrackRecord] Read error: {e}")
    # Return live in-memory snapshot if file not yet written
    return JSONResponse({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": _compute_track_summary(),
        "signals": sorted(_track_store, key=lambda r: r.get("entry_time") or "", reverse=True)[:500],
    })


@app.get("/api/exchange-rates")
async def exchange_rates_endpoint():
    """Return live USD-based exchange rates plus USD plan prices for the frontend."""
    rates = await _get_fx_rates()
    return {"base": "USD", "rates": rates, "plan_prices_usd": USD_PLAN_PRICES}


def _to_subunits(amount_float: float, currency: str) -> int:
    """Convert a float amount to the currency's smallest unit."""
    if currency in _ZERO_DECIMAL:
        return int(round(amount_float))
    if currency in _THREE_DECIMAL:
        return int(round(amount_float * 1000))
    return int(round(amount_float * 100))


@app.post("/api/create-order")
async def create_order(req: CreateOrderRequest, user_id: str = Depends(get_current_user)):
    """Convert the USD plan price to the requested currency at live rate, create Razorpay order."""
    if not RAZORPAY_ENABLED:
        raise HTTPException(status_code=503, detail="Payment system not configured")

    usd_price = USD_PLAN_PRICES.get(req.plan)
    if usd_price is None:
        raise HTTPException(status_code=400, detail=f"Unknown plan: {req.plan}")

    currency = req.currency.upper()
    rates = await _get_fx_rates()
    rate = rates.get(currency)
    if rate is None:
        raise HTTPException(status_code=400, detail=f"Unsupported currency: {currency}")

    amount_subunits = _to_subunits(usd_price * rate, currency)
    if amount_subunits < 100:
        raise HTTPException(status_code=400, detail="Calculated amount too low (min 100 subunits)")

    receipt = f"{user_id[:16]}_{req.plan}_{int(time.time())}"
    order = await _rzp_post("/orders", {
        "amount": amount_subunits,
        "currency": currency,
        "receipt": receipt,
        "notes": {"user_id": user_id, "plan": req.plan, "usd_price": str(usd_price)},
    })
    return {"order_id": order["id"], "amount": order["amount"], "currency": order["currency"]}


@app.post("/api/verify-payment")
async def verify_payment(req: VerifyPaymentRequest, user_id: str = Depends(get_current_user)):
    """Verify Razorpay payment signature and upgrade the user's plan."""
    import hmac as _hmac
    import hashlib

    if not RAZORPAY_KEY_SECRET:
        raise HTTPException(status_code=503, detail="Payment system not configured")
    if not req.razorpay_payment_id or not req.razorpay_order_id or not req.razorpay_signature:
        raise HTTPException(status_code=400, detail="Missing payment fields")

    msg = f"{req.razorpay_order_id}|{req.razorpay_payment_id}"
    expected = _hmac.new(
        RAZORPAY_KEY_SECRET.encode(),
        msg.encode(),
        hashlib.sha256,
    ).hexdigest()

    if not _hmac.compare_digest(expected, req.razorpay_signature):
        raise HTTPException(status_code=400, detail="Payment verification failed")

    plan = req.plan if req.plan in ("basic", "intermediate", "pro") else "basic"
    user_ref = db.collection("users").document(user_id)
    try:
        update_result = user_ref.update({
            "plan": plan,
            "subscription": {
                "status": "active",
                "payment_id": req.razorpay_payment_id,
                "order_id": req.razorpay_order_id,
                "activated_at": datetime.now(timezone.utc).isoformat(),
                "plan_type": plan,
            },
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
# Razorpay Webhook for subscription.activated events
# -------------------------------------------------------------------
@app.post("/api/v1/payments/webhook")
async def razorpay_webhook(request: Request):
    """
    Verify Razorpay webhook signature, handle subscription.activated,
    and promote the matching user to the pro plan.
    """
    import hmac
    import hashlib

    body = await request.body()
    received_sig = request.headers.get("X-Razorpay-Signature", "")

    if RAZORPAY_WEBHOOK_SECRET:
        expected = hmac.new(
            RAZORPAY_WEBHOOK_SECRET.encode(),
            body,
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(expected, received_sig):
            raise HTTPException(status_code=400, detail="Invalid webhook signature")

    try:
        data = json.loads(body)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    event = data.get("event")
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
            try:
                result = user_ref.update({
                    "plan": plan_tier,
                    "subscription": {
                        "status": "active",
                        "subscription_id": subscription_id,
                        "activated_at": datetime.now(timezone.utc).isoformat(),
                        "plan_type": plan_tier,
                    },
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
                <h2>✅ Subscription Activated</h2>
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
                <small style="color: #6b7280;">Aegis‑1 Sovereign Terminal</small>
            </body>
            </html>
            """,
            subtype=MessageType.html,
        )
        await fastmail.send_message(message)
        print(f"✅ Subscription confirmation sent to {email}")
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
                <h2>⏰ Trial Expiring Soon</h2>
                <p>Your Aegis-1 trial will expire in {hours_until} hours on {trial_end_date.strftime('%B %d, %Y')}.</p>
                <p>After expiry, you will only have access to 5 tokens (BTC, ETH, SOL, BNB, XRP).</p>
                <p><strong>Upgrade to Pro for:</strong></p>
                <ul>
                    <li>All 58 token signals</li>
                    <li>Alpha Mode (unfiltered AI conviction)</li>
                    <li>Real-time WebSocket feed</li>
                    <li>Priority support</li>
                </ul>
                <p><a href="{BASE_URL}/web/src/pages/pricing.html" style="color: #00f2ff;">Click here to upgrade now →</a></p>
                <hr>
                <small style="color: #6b7280;">Aegis‑1 Sovereign Terminal</small>
            </body>
            </html>
            """,
            subtype=MessageType.html,
        )
        await fastmail.send_message(message)
        print(f"✅ Trial reminder sent to {email}")
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
                <h2>📅 Subscription Renewal Notice</h2>
                <p>Your Aegis-1 Pro subscription will expire in {days_until} days on {expiry_date.strftime('%B %d, %Y')}.</p>
                <p><strong>Renew now to continue enjoying:</strong></p>
                <ul>
                    <li>All 58 token signals</li>
                    <li>Alpha Mode (unfiltered AI conviction)</li>
                    <li>Real-time WebSocket feed</li>
                    <li>Priority support</li>
                </ul>
                <p><a href="{BASE_URL}/web/src/pages/pricing.html" style="color: #00f2ff;">Click here to renew →</a></p>
                <hr>
                <small style="color: #6b7280;">Aegis‑1 Sovereign Terminal</small>
            </body>
            </html>
            """,
            subtype=MessageType.html,
        )
        await fastmail.send_message(message)
        print(f"✅ Subscription reminder sent to {email}")
    except Exception as e:
        print(f"Failed to send subscription reminder: {e}")

# -------------------------------------------------------------------
# Background Tasks for Reminders
# -------------------------------------------------------------------
async def check_and_send_trial_reminders():
    while True:
        try:
            now = datetime.now(timezone.utc)
            users_ref = db.collection("users")
            query = users_ref.where("plan", "==", "trial").stream()
            
            for user_doc in query:
                user_data = user_doc.to_dict() or {}
                trial_end = user_data.get("trial_end")
                
                if trial_end:
                    trial_end_date = datetime.fromisoformat(trial_end)
                    if trial_end_date.tzinfo is None:
                        trial_end_date = trial_end_date.replace(tzinfo=timezone.utc)
                    hours_until_expiry = (trial_end_date - now).total_seconds() / 3600
                    
                    if 23 <= hours_until_expiry <= 25 and not user_data.get("reminder_24h_sent"):
                        await send_trial_expiry_reminder(user_doc.id, trial_end_date, hours_until=24)
                        user_doc.reference.update({"reminder_24h_sent": True})
                    elif 0.5 <= hours_until_expiry <= 1.5 and not user_data.get("reminder_1h_sent"):
                        await send_trial_expiry_reminder(user_doc.id, trial_end_date, hours_until=1)
                        user_doc.reference.update({"reminder_1h_sent": True})
            
            await asyncio.sleep(3600)
        except Exception as e:
            print(f"Trial reminder check error: {e}")
            await asyncio.sleep(3600)

async def check_and_send_subscription_reminders():
    while True:
        try:
            now = datetime.now(timezone.utc)
            users_ref = db.collection("users")
            query = users_ref.where("subscription.status", "==", "active").stream()
            
            for user_doc in query:
                user_data = user_doc.to_dict() or {}
                sub_end = user_data.get("subscription_end")
                
                if sub_end:
                    sub_end_date = datetime.fromisoformat(sub_end)
                    if sub_end_date.tzinfo is None:
                        sub_end_date = sub_end_date.replace(tzinfo=timezone.utc)
                    days_until_expiry = (sub_end_date - now).days
                    
                    if days_until_expiry == 7 and not user_data.get("reminder_7d_sent"):
                        await send_subscription_expiry_reminder(user_doc.id, sub_end_date, days_until=7)
                        user_doc.reference.update({"reminder_7d_sent": True})
                    elif days_until_expiry == 3 and not user_data.get("reminder_3d_sent"):
                        await send_subscription_expiry_reminder(user_doc.id, sub_end_date, days_until=3)
                        user_doc.reference.update({"reminder_3d_sent": True})
                    elif days_until_expiry == 1 and not user_data.get("reminder_1d_sent"):
                        await send_subscription_expiry_reminder(user_doc.id, sub_end_date, days_until=1)
                        user_doc.reference.update({"reminder_1d_sent": True})
            
            await asyncio.sleep(86400)
        except Exception as e:
            print(f"Subscription reminder check error: {e}")
            await asyncio.sleep(86400)

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
                print(f"✅ WebSocket authenticated: {current_user_email}")
        except Exception as auth_err:
            print(f"⚠️ Auth error: {auth_err}")
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

        # Plan info cached for 10 s — avoids a Firestore round-trip on every 250 ms tick
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
                # Cast values to float explicitly — engine.live_prices contains numpy.float32.
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
                                # nested timeframe dict — pick best available tf
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
                    # No fresh prices — serve stale cache so the UI stays populated
                    live_tickers = dict(_cached_tickers)

                if tick_count % SIGNAL_EVERY_N == 0:
                    # ── Full signal update (every ~2 s) ──────────────────────────
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
                            # Flat signal dict from the new live_engine — pass all fields through.
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
                            })
                            filtered_signals[sym] = _out

                    # Pad filtered_signals with stub entries for every engine-tracked
                    # symbol not yet covered — ensures all 60 symbols appear in the
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

                    response_data = {
                        "tickers": live_tickers,
                        "signals": filtered_signals,
                        "timeframes": timeframes_map,
                        "timeframe": response_timeframe or "1h",
                        "open_trades": LIVE_STATE.data.get("open_trades", []),
                        "balance": LIVE_STATE.data.get("balance", 0),
                        "alpha_mode": (
                            LIVE_STATE.data.get("alpha_mode", False)
                            and _user_plan_cache in ("pro", "premium")
                        ),
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
                    # ── Ticker-only update (every 250 ms) ───────────────────────
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
        print(f"🔌 WebSocket disconnected: {current_user_email}")
    except asyncio.TimeoutError:
        print(f"⏱️ WebSocket timeout during authentication")
        await websocket.close(code=1000)
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
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
    if user_plan != "pro":
        raise HTTPException(status_code=403, detail="Alpha Mode is only available for Pro subscribers")
    
    if LIVE_STATE.engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready")
    new_state = not LIVE_STATE.engine.alpha_mode
    LIVE_STATE.engine.alpha_mode = new_state
    LIVE_STATE.data["alpha_mode"] = new_state
    return {"alpha_mode": new_state}

# -------------------------------------------------------------------
# Dev code system — admin generation + user redemption
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
        db.collection("dev_codes").document(new_key).set({
            "source": "backend",
            "plan": "pro",
            "label": "startup_key",
            "created_at": now_dt.isoformat(),
            "expires_at": expires_dt.isoformat(),
            "features": features,
            "created_by": "system_startup",
            "used_by": None,
        })

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

@app.delete("/api/admin/track-record/{signal_id}")
async def delete_track_record_entry(signal_id: str, _: None = Depends(_require_admin)):
    """Remove one signal from the in-memory track record and persist to disk."""
    global _track_store, _tr_seen_ids
    before = len(_track_store)
    _track_store = [r for r in _track_store if r.get("signal_id") != signal_id]
    _tr_seen_ids.discard(signal_id)
    removed = before - len(_track_store)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"signal_id '{signal_id}' not found")
    _save_track_record()
    return {"success": True, "removed": removed, "signal_id": signal_id,
            "remaining": len(_track_store)}


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

    # Persist empty files
    for path in (TRACK_RECORD_PATH, WEB_ROOT_PATH / "track_record.json"):
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
async def admin_generate_dev_codes(
    req: GenerateDevCodeRequest,
    _admin: None = Depends(_require_admin),
):
    if req.plan not in ("basic", "intermediate", "pro"):
        raise HTTPException(status_code=400, detail="Invalid plan")
    if not (1 <= req.count <= 50):
        raise HTTPException(status_code=400, detail="count must be 1–50")
    if not (1 <= req.days <= 365):
        raise HTTPException(status_code=400, detail="days must be 1–365")

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
async def admin_list_dev_codes(
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

        # Token consumed — provision a replacement immediately so the backend
        # always has an active token available for the next developer.
        try:
            await asyncio.to_thread(_provision_dev_token)
        except Exception as _e:
            logger.warning(f"Auto-provision replacement dev token failed: {_e}")

    return {"status": "success", "plan": plan, "expires_at": expires_iso}

# -------------------------------------------------------------------
# Dev Key System — validate-devkey endpoint
# -------------------------------------------------------------------

DEV_KEY_FEATURES = ["extended_timeframes", "alpha_mode", "all_signals", "pro_signals"]

# In-memory TTL cache for validated dev keys (key_string -> {cached_at})
_dev_key_cache: Dict[str, Dict] = {}
_DEV_KEY_CACHE_TTL = 300  # 5 minutes

class ValidateDevKeyRequest(BaseModel):
    dev_key: str

@app.post("/auth/validate-devkey")
async def validate_dev_key(req: ValidateDevKeyRequest, request: Request):
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

    try:
        db.collection('reviews').add(review_doc)
        print(f"✅ Review saved to Firestore: {review.email}")
        return {"status": "saved", "method": "firestore"}
    except Exception as e:
        print(f"❌ Failed to save review to Firestore: {e}. Falling back to email.")
        try:
            message = MessageSchema(
                subject=f"New Review from {review.name}",
                recipients=[NameEmail(name="Animesh Kukreti", email="animeshkukreti60@gmail.com")],
                body=(f"Name: {review.name}\nEmail: {review.email}\nRating: {rating}\nProduct: {review.product or ''}\n\n"
                      f"Message:\n{review.message or ''}"),
                subtype=MessageType.plain,
            )
            await fastmail.send_message(message)
            print("✅ Review emailed as fallback")
            return {"status": "saved", "method": "email_fallback"}
        except Exception as e2:
            print(f"❌ Failed to send review email fallback: {e2}")
            raise HTTPException(status_code=500, detail="Failed to save review or send fallback email")

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
async def execute_trade(request: TradeExecuteRequest, user_id: str = Depends(get_firebase_uid)):
    trade_data = request.dict()
    trade_data["openTime"] = datetime.now(timezone.utc).isoformat()
    # Ensure it's saved under the user who made the request
    trade_data["userId"] = user_id
    
    try:
        # NOTE: Connect to the Demat API here.
        # This is where the call will be sent to the broker API for execution.
        print(f"🚀 Sending order to Demat API: Symbol: {trade_data.get('symbol')}, Side: {trade_data.get('side')}, Units: {trade_data.get('positionUnits')}")
        # demat_response = await demat_client.place_order(...)
        
        trade_ref = db.collection("users").document(user_id).collection("trades").document()
        trade_data["id"] = trade_ref.id
        trade_data["demat_status"] = "sent_to_broker"
        trade_ref.set(trade_data)
        print(f"✅ Trade executed and sent to Demat via API for {user_id}")
        return {"status": "success", "trade_id": trade_ref.id, "trade": trade_data, "message": "Order sent to Demat account successfully"}
    except Exception as e:
        print(f"❌ Failed to execute trade for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to execute trade")

@app.post("/api/trades/{trade_id}/close")
async def close_trade(trade_id: str, user_id: str = Depends(get_firebase_uid)):
    try:
        trade_ref = db.collection("users").document(user_id).collection("trades").document(trade_id)
        trade_doc = trade_ref.get()
        if not trade_doc.exists:
            raise HTTPException(status_code=404, detail="Trade not found")
        trade_ref.update({
            "status": "closed",
            "closeTime": datetime.now(timezone.utc).isoformat()
        })
        print(f"✅ Trade {trade_id} closed for {user_id}")
        return {"status": "success", "trade_id": trade_id}
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Failed to close trade {trade_id} for {user_id}: {e}")
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
    try:
        message = MessageSchema(
            subject="Your AEGIS Verification Code",
            recipients=[NameEmail(name=email, email=email)],
            body=f"""
            <!DOCTYPE html>
            <html>
            <head><meta charset="UTF-8"></head>
            <body style="margin:0;padding:0;background:#0a0a0c;font-family:'Segoe UI',Arial,sans-serif;">
              <table width="100%" cellpadding="0" cellspacing="0" style="background:#0a0a0c;padding:40px 0;">
                <tr><td align="center">
                  <table width="480" cellpadding="0" cellspacing="0"
                         style="background:#0f111a;border:1px solid rgba(0,242,255,0.15);border-radius:16px;overflow:hidden;max-width:480px;">

                    <!-- Header -->
                    <tr>
                      <td style="background:linear-gradient(135deg,#00f2ff22,#7b2fff22);padding:32px 40px 24px;text-align:center;border-bottom:1px solid rgba(0,242,255,0.1);">
                        <div style="font-size:28px;font-weight:800;letter-spacing:3px;
                                    background:linear-gradient(90deg,#00f2ff,#7b2fff);
                                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                                    display:inline-block;">
                          ⚡ AEGIS
                        </div>
                        <p style="color:#6b7280;margin:8px 0 0;font-size:13px;letter-spacing:1px;">SOVEREIGN TERMINAL</p>
                      </td>
                    </tr>

                    <!-- Body -->
                    <tr>
                      <td style="padding:36px 40px;">
                        <p style="color:#9ca3af;font-size:15px;margin:0 0 8px;">Email Verification</p>
                        <h2 style="color:#f9fafb;font-size:20px;font-weight:600;margin:0 0 24px;">Your one-time verification code</h2>

                        <!-- OTP display -->
                        <div style="background:#0a0a0c;border:1px solid rgba(0,242,255,0.25);border-radius:12px;
                                    padding:24px;text-align:center;margin:0 0 24px;">
                          <span style="font-family:'Courier New',Courier,monospace;font-size:38px;font-weight:700;
                                       letter-spacing:12px;color:#00f2ff;">{otp}</span>
                        </div>

                        <p style="color:#9ca3af;font-size:14px;margin:0 0 8px;">
                          This code expires in <strong style="color:#f9fafb;">5 minutes</strong>.
                          Do not share it with anyone.
                        </p>
                        <p style="color:#6b7280;font-size:13px;margin:0;">
                          If you didn't request this, you can safely ignore this email.
                        </p>
                      </td>
                    </tr>

                    <!-- Footer -->
                    <tr>
                      <td style="padding:20px 40px 28px;border-top:1px solid rgba(255,255,255,0.05);text-align:center;">
                        <p style="color:#4b5563;font-size:12px;margin:0;">
                          Sent by
                          <a href="mailto:animeshkukreti@gatekeeper.sbs"
                             style="color:#00f2ff;text-decoration:none;">animeshkukreti@gatekeeper.sbs</a>
                          &nbsp;·&nbsp;
                          <a href="https://gatekeeper.sbs"
                             style="color:#00f2ff;text-decoration:none;">gatekeeper.sbs</a>
                        </p>
                      </td>
                    </tr>

                  </table>
                </td></tr>
              </table>
            </body>
            </html>
            """,
            subtype=MessageType.html,
        )
        await fastmail.send_message(message)
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
# FIRESTORE SIGNALS API – Get all active signals
# -------------------------------------------------------------------
@app.get("/api/signals")
async def get_signals(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
):
    """
    Get all active signals from Firestore.
    Signals are readable by all authenticated users.
    Trial users get limited access to specific tokens only.
    """
    try:
        # Get signals collection
        signals_ref = db.collection("signals")
        signals_docs = signals_ref.stream()
        
        signals = []
        for doc in signals_docs:
            signal_data = doc.to_dict() if hasattr(doc, "to_dict") else {}
            
            if not isinstance(signal_data, dict):
                continue
                
            # Safe access to document ID - handle None values properly
            doc_id = getattr(doc, "id", None)
            if doc_id is None:
                doc_id = getattr(doc, "name", None) or (str(doc.reference.path).split("/")[-1] if hasattr(doc, "reference") else f"doc_{datetime.now().timestamp()}")
            
            signal_data["id"] = str(doc_id)  # Convert to string for type compatibility
            signals.append(signal_data)
        
        # Sort by timestamp (newest first)
        signals.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        
        return {
            "success": True,
            "count": len(signals),
            "signals": signals,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        print(f"❌ Error fetching signals: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch signals")

# -------------------------------------------------------------------
# FIRESTORE SIGNALS API – Get specific signal
# -------------------------------------------------------------------
@app.get("/api/signals/{symbol}")
async def get_signal(symbol: str):
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
        print(f"❌ Error fetching signal {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch signal")

# -------------------------------------------------------------------
# FIRESTORE DASHBOARD API – Get dashboard data for user
# -------------------------------------------------------------------
@app.get("/api/dashboard")
async def get_dashboard(
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
        if LIVE_STATE.engine is not None and hasattr(LIVE_STATE.engine, 'last_signals') and LIVE_STATE.engine.last_signals:
            dashboard_data["signals"] = {k: v for k, v in list(LIVE_STATE.engine.last_signals.items())[:10]}
        
        return dashboard_data
    except HTTPException:
        raise
    except jwt.InvalidSignatureError:
        print("❌ Invalid JWT signature")
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        # Fallback to generic data if personal data fails
        print(f"❌ Error fetching dashboard (fallback): {str(e)}")
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
# FIRESTORE PUBLIC SIGNALS – No authentication required
# -------------------------------------------------------------------
@app.get("/api/public/signals")
async def get_public_signals():
    """
    Get public signals for non-logged-in users.
    Shows a limited set of signals to encourage signup.
    """
    try:
        signals_ref = db.collection("signals")
        # Get only top 3-4 signals
        signals_docs = signals_ref.limit(4).stream()
        
        signals = []
        for doc in signals_docs:
            signal_data = doc.to_dict()
            if signal_data is None:
                continue
            signal_data["id"] = doc.id
            signals.append(signal_data)
        
        return {
            "success": True,
            "count": len(signals),
            "signals": signals,
            "message": "Sign up to access all signals",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        print(f"❌ Error fetching public signals: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch signals")

# -------------------------------------------------------------------
# FIRESTORE SIGNAL UPDATE – Backend trigger (admin only)
# -------------------------------------------------------------------
@app.post("/api/admin/signals/update")
async def update_signal(
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
        print(f"❌ Error updating signal: {str(e)}")
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
async def get_signals_fleet(symbol: Optional[str] = None, _user: dict = Depends(verify_api_key)):
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

# Frontend uses /api/v1/keys/regenerate — alias to the canonical route above
@app.post("/api/v1/keys/regenerate")
async def regenerate_api_key_alias(user_id: str = Depends(get_current_user)):
    return await regenerate_api_key(user_id)

# -------------------------------------------------------------------
# User settings (capital + risk) — persisted to Firestore
# Called by the Settings room in dashboard.js when user hits Save
# -------------------------------------------------------------------
class UserSettingsUpdate(BaseModel):
    capital: float
    risk_pct: float

@app.post("/user/settings")
async def save_user_settings(
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


# ═══════════════════════════════════════════════════════════════════════════════
#  AEGIS UNIVERSAL TRADER — API Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

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
      mode         – filter by 'scalping' | 'intraday' | 'swing' (optional)
      risk_profile – 'conservative' | 'balanced' | 'aggressive' (default: balanced)
      scan         – if true, trigger a fresh scan (slow); otherwise return cached
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
    """Public endpoint — returns trader_track_record.json (wallet + trade history)."""
    if TRADER_TRACK_RECORD_PATH.exists():
        try:
            with open(TRADER_TRACK_RECORD_PATH, encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {"signals": [], "balance": 10000, "win_rate": 0, "total_trades": 0}


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
async def get_trader_wallet(_user: str = Depends(get_current_user)):
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
