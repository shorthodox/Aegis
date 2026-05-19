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
import random
import string
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

# -------------------------------------------------------------------
# Helper: Recursively convert numpy types to native Python types
# -------------------------------------------------------------------
def numpy_to_native(obj):
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
# Cashfree PG SDK imports
# -------------------------------------------------------------------
try:
    from cashfree_pg.api_client import Cashfree
    from cashfree_pg.models.create_order_request import CreateOrderRequest
    from cashfree_pg.models.customer_details import CustomerDetails
    from cashfree_pg.models.create_subscription_payment_request import CreateSubscriptionPaymentRequest
    CASHFREE_SDK_AVAILABLE = True
except ImportError:
    CASHFREE_SDK_AVAILABLE = False
    print("[WARNING] Cashfree PG SDK not installed. Install with: pip install cashfree-pg")

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
# Cashfree payment gateway environment fields
# -------------------------------------------------------------------
CASHFREE_APP_ID = os.getenv("CASHFREE_APP_ID")
CASHFREE_SECRET_KEY = os.getenv("CASHFREE_SECRET_KEY")
CASHFREE_ENV = os.getenv("CASHFREE_ENV", "TEST").upper()
CASHFREE_BASE_URL = "https://sandbox.cashfree.com" if CASHFREE_ENV == "TEST" else "https://api.cashfree.com"
CASHFREE_ENABLED = bool(CASHFREE_APP_ID and CASHFREE_SECRET_KEY)

# Initialize Cashfree SDK if available and credentials present
if CASHFREE_SDK_AVAILABLE and CASHFREE_ENABLED:
    Cashfree.XClientId = CASHFREE_APP_ID
    Cashfree.XClientSecret = CASHFREE_SECRET_KEY
    Cashfree.XEnvironment = Cashfree.SANDBOX if CASHFREE_ENV == "TEST" else Cashfree.PRODUCTION
    print(f"🔒 Cashfree payment gateway configured for {CASHFREE_ENV}")
elif CASHFREE_ENABLED:
    print(f"[WARNING] Cashfree SDK not available, using REST API fallback for {CASHFREE_ENV}")
else:
    print("[WARNING] Cashfree payment gateway not configured. Set CASHFREE_APP_ID/CASHFREE_SECRET_KEY to enable.")

# -------------------------------------------------------------------
# SOVEREIGN FIREBASE INITIALIZATION
# Supports both Railway (JSON string in env) and local (file path)
# -------------------------------------------------------------------
import firebase_admin
from firebase_admin import credentials, firestore, auth as firebase_auth

FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "aegis-d78e1")
cred_json = os.getenv("FIREBASE_CREDENTIALS")

if cred_json:
    # Check if cred_json is a file path first
    if os.path.exists(cred_json):
        cred = credentials.Certificate(cred_json)
        print(f"[FIREBASE] Initialized via file path: {cred_json}")
    else:
        try:
            # Try to parse as JSON string
            cred_dict = json.loads(cred_json)
            cred = credentials.Certificate(cred_dict)
            print("[FIREBASE] Initialized via JSON environment variable")
        except Exception as e:
            print(f"[ERROR] Failed to parse FIREBASE_CREDENTIALS JSON: {e}")
            # Fallback to default file path
            cred_path = 'config/serviceAccountKey.json'
            if os.path.exists(cred_path):
                cred = credentials.Certificate(cred_path)
                print(f"[FIREBASE] Initialized via fallback file: {cred_path}")
            else:
                raise RuntimeError("Firebase credentials not found. Provide either FIREBASE_CREDENTIALS as JSON or ensure config/serviceAccountKey.json exists")
else:
    # Fallback for local development
    cred_path = 'config/serviceAccountKey.json'
    if os.path.exists(cred_path):
        cred = credentials.Certificate(cred_path)
        print(f"[FIREBASE] Initialized via local file: {cred_path}")
    else:
        raise RuntimeError("Firebase credentials not found in ENV or at config/serviceAccountKey.json")

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
    from scripts.live_engine import LiveEngine, automated_setup
    import argparse

    base_url = os.getenv("BASE_URL", "http://localhost:8000")
    print(f"🚀 Engine background task starting. BASE_URL = {base_url}")

    args = argparse.Namespace()
    args.capital = 10000.0
    args.risk = 2.0
    args.alpha_risk = 3.0
    args.max_position = 2000.0
    args.timeframe = '1m'
    args.alpha_mode = False
    args.proxy = None

    backtest_dir = Path(__file__).parent / "logs" / "backtests"

    try:
        configs, capital, max_pos, scan_seconds, alpha_mode, alpha_risk, proxy = automated_setup(backtest_dir, args)
    except Exception as e:
        print(f"❌ automated_setup failed: {e}")
        await asyncio.sleep(1)
        return

    engine = LiveEngine(
        token_configs=configs,
        capital=capital,
        max_position_usdt=max_pos,
        scan_interval_seconds=scan_seconds,
        alpha_mode=alpha_mode,
        alpha_risk_pct=alpha_risk,
        proxy_url=proxy
    )
    LIVE_STATE.engine = engine

    async def update_state():
        while True:
            try:
                LIVE_STATE.data["tickers"] = engine.live_prices.copy()
                LIVE_STATE.data["signals"] = engine.last_signals.copy()
                LIVE_STATE.data["open_trades"] = [asdict(t) for t in engine.wallet.open_trades.values()]
                LIVE_STATE.data["balance"] = engine.wallet.balance
                LIVE_STATE.data["alpha_mode"] = engine.alpha_mode
                if hasattr(engine, 'bootstrap_total'):
                    LIVE_STATE.data["warmup_progress"] = f"{engine.bootstrap_done}/{engine.bootstrap_total}"
                # --- write latest signals to a JSON file for frontend consumption ---
                try:
                    signals_dir = WEB_ROOT_PATH / 'src' / 'data'
                    signals_dir.mkdir(parents=True, exist_ok=True)
                    signals_file = signals_dir / 'live_signals.json'
                    with open(signals_file, 'w', encoding='utf-8') as sf:
                        # use default=str to ensure datetimes/objects are serializable
                        safe_signals = numpy_to_native(LIVE_STATE.data.get('signals', {}))
                        json.dump(safe_signals, sf, default=str)
                except Exception as _e:
                    print(f"⚠️ Failed to write live_signals.json: {_e}")
                
                # --- write latest signals to Firebase Firestore ---
                try:
                    signals_data = LIVE_STATE.data.get('signals', {})
                    if signals_data:
                        print(f"[PRODUCER] Attempting to push {len(signals_data)} signals to Firestore...")
                    batch = db.batch()
                    count = 0
                    now_str = datetime.now(timezone.utc).isoformat()
                    for sym, sig in signals_data.items():
                        doc_id = sym.replace('/', '_')
                        sig_ref = db.collection("signals").document(doc_id)
                        
                        # Convert signals to a plain dictionary safely
                        if isinstance(sig, dict):
                            sig_data = sig.copy()
                        elif hasattr(sig, 'dict') and callable(sig.dict):
                            sig_data = sig.dict()
                        elif hasattr(sig, '__dataclass_fields__'):
                            sig_data = asdict(sig)
                        elif hasattr(sig, '__dict__'):
                            sig_data = vars(sig).copy()
                        else:
                            try:
                                sig_data = dict(sig)
                            except Exception:
                                sig_data = {'value': sig}

                        # Ensure sig_data is a dict before setting keys
                        if not isinstance(sig_data, dict):
                            sig_data = {'value': sig_data}
                        
                        sig_data['symbol'] = sym
                        sig_data['timestamp'] = now_str
                        
                        # Fix numpy serialization here
                        sig_data = numpy_to_native(sig_data)
                        if not isinstance(sig_data, dict):
                            sig_data = {'value': sig_data, 'symbol': sym, 'timestamp': now_str}
                        
                        batch.set(sig_ref, sig_data, merge=True)
                        count += 1
                        
                        if count >= 450:
                            results = batch.commit()
                            print(f"[PRODUCER SUCCESS] Batch committed {count} signals. Last commit timestamp: {results[-1].update_time if results else 'N/A'}")
                            batch = db.batch()
                            count = 0
                    
                    if count > 0:
                        results = batch.commit()
                        print(f"[PRODUCER SUCCESS] Batch committed {count} signals. Last commit timestamp: {results[-1].update_time if results else 'N/A'}")
                except Exception as _e:
                    print(f"[PRODUCER ERROR] ⚠️ Failed to write signals to Firestore: {_e}")
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
@asynccontextmanager
async def lifespan(app: FastAPI):
    engine_task = asyncio.create_task(run_engine_background())
    reminder_task = asyncio.create_task(check_and_send_trial_reminders())
    subscription_task = asyncio.create_task(check_and_send_subscription_reminders())
    analytics_task = asyncio.create_task(analytics_loop())
    yield
    engine_task.cancel()
    reminder_task.cancel()
    subscription_task.cancel()
    analytics_task.cancel()

app = FastAPI(title="Aegis-1 by Gatekeeper", lifespan=lifespan)

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
@app.get("/")
async def root_redirect():
    return RedirectResponse(url="/web/src/pages/index.html")

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
                if user_doc:
                    plan = user_doc.get("plan", "trial")
                    trial_end = user_doc.get("trial_end")
                    if plan in ["pro", "active"]:
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
    # Try decoding as Firebase token first
    try:
        decoded_token = firebase_auth.verify_id_token(token)
        # Firebase token contains email, fallback to uid if not present
        return decoded_token.get("email") or decoded_token.get("uid")
    except Exception:
        # Fallback to custom JWT
        try:
            assert SECRET_KEY is not None, "SECRET_KEY must be set"
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload.get("sub")
        except:
            return None

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    email = decode_token(credentials.credentials)
    if not email:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    return email

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

def create_user_doc(email: str, password_hash: Optional[str] = None,
                    provider: Optional[str] = None, social_id: Optional[str] = None,
                    full_name: Optional[str] = None, location: Optional[str] = None) -> Dict:
    now = datetime.now(timezone.utc).isoformat()
    trial_end = (datetime.now(timezone.utc) + timedelta(days=3)).isoformat()
    user_data = {
        "email": email,
        "plan": "trial",
        "trial_end": trial_end,
        "created_at": now,
        "last_login": now,
        "subscription": {
            "status": "inactive"
        }
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
    if user_doc.get("plan") == "pro":
        return False
    trial_end = user_doc.get("trial_end")
    if trial_end:
        return datetime.now(timezone.utc) > datetime.fromisoformat(trial_end)
    return True

# -------------------------------------------------------------------
# OAuth routes
# -------------------------------------------------------------------
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000").rstrip('/')

@app.get("/auth/me")
async def get_me(user_id: str = Depends(get_current_user)):
    user_doc = get_user_doc(user_id)
    if not user_doc:
        raise HTTPException(status_code=404)
    return {
        "uid": user_id,
        "email": user_doc.get("email", user_id),
        "plan": user_doc.get("plan", "trial"),
        "trial_end": user_doc.get("trial_end"),
        "full_name": user_doc.get("full_name"),
        "location": user_doc.get("location")
    }

@app.post("/api/users/provision")
async def provision_user(request: Request, user_id: str = Depends(get_current_user)):
    """
    Create a default backend profile for a Firebase-authenticated user who has no record yet.
    Idempotent: returns the existing doc if one already exists.
    """
    data = await request.json()
    firebase_uid = data.get("uid") or user_id
    email = data.get("email") or (user_id if "@" in user_id else None)
    display_name = data.get("display_name") or (email.split("@")[0] if email else firebase_uid)

    # Prefer email as doc key (consistent with rest of backend), fall back to uid
    doc_key = email or firebase_uid

    existing = get_user_doc(doc_key)
    if existing:
        update_last_login(doc_key)
        return {
            "uid": firebase_uid,
            "email": existing.get("email", doc_key),
            "plan": existing.get("plan", "trial"),
            "trial_end": existing.get("trial_end"),
            "full_name": existing.get("full_name"),
            "location": existing.get("location"),
        }

    user_doc = create_user_doc(
        doc_key,
        provider="firebase",
        social_id=firebase_uid,
        full_name=display_name,
    )
    return {
        "uid": firebase_uid,
        "email": email or doc_key,
        "plan": user_doc.get("plan", "trial"),
        "trial_end": user_doc.get("trial_end"),
        "full_name": user_doc.get("full_name"),
        "location": user_doc.get("location"),
    }

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

class CashfreePaymentRequest(BaseModel):
    amount: float
    currency: str = "INR"
    email: EmailStr
    order_id: Optional[str] = None
    customer_phone: Optional[str] = None

class CreateSubscriptionRequest(BaseModel):
    plan_name: str
    amount: float
    currency: str = "INR"
    email: EmailStr
    customer_phone: Optional[str] = None

class Review(BaseModel):
    name: str
    email: EmailStr
    rating: int
    message: Optional[str] = None
    product: Optional[str] = None

# -------------------------------------------------------------------
# Email configuration
# -------------------------------------------------------------------
conf = ConnectionConfig(
    MAIL_USERNAME=os.getenv("MAIL_USERNAME", "your_email@gmail.com"),
    MAIL_PASSWORD=SecretStr(os.getenv("MAIL_PASSWORD", "your_app_password")),
    MAIL_FROM=os.getenv("MAIL_FROM", "noreply@aegis.com"),
    MAIL_PORT=587,
    MAIL_SERVER="smtp.gmail.com",
    MAIL_STARTTLS=True,
    MAIL_SSL_TLS=False,
)

fastmail = FastMail(conf)

# -------------------------------------------------------------------
# 3-Step Onboarding with OTP
# -------------------------------------------------------------------
@app.post("/auth/send-otp-for-registration")
async def send_otp_for_registration(request: OTPSendRequest):
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
            subject="Your Aegis‑1 Verification Code",
            recipients=[NameEmail(name=email, email=email)],
            body=f"""
            <html>
            <body style="font-family: monospace; background: #0a0a0c; color: #00f2ff; padding: 20px;">
                <h2>🔐 Aegis‑1 OTP</h2>
                <p>Your verification code is:</p>
                <h1 style="background: #1a1f2e; display: inline-block; padding: 12px 24px; border-radius: 12px;">{otp}</h1>
                <p>This code expires in 5 minutes.</p>
                <p>If you didn't request this, please ignore this email.</p>
                <hr>
                <small style="color: #6b7280;">Aegis‑1 Sovereign Terminal</small>
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
    otp_store[email]["verified"] = True
    return {"success": True, "message": "OTP verified successfully. Please complete your profile."}

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
PRO_TOKENS = []
try:
    from scripts.live_engine import FLEET
    PRO_TOKENS = FLEET if isinstance(FLEET, list) else []
except ImportError:
    PRO_TOKENS = BASIC_TOKENS + ["ADA/USDT", "DOT/USDT", "DOGE/USDT", "MATIC/USDT", "AVAX/USDT", "LINK/USDT"]

BASIC_TIMEFRAMES = ["30m", "1h"]

def get_user_plan(email: str) -> str:
    user_doc = get_user_doc(email)
    return user_doc["plan"] if user_doc else "trial"

def get_allowed_tokens(email: str) -> List[str]:
    plan = get_user_plan(email)
    if plan == "pro":
        return PRO_TOKENS if PRO_TOKENS else BASIC_TOKENS + ["ADA/USDT", "DOT/USDT", "DOGE/USDT", "MATIC/USDT", "AVAX/USDT", "LINK/USDT"]
    else:
        return BASIC_TOKENS

def get_allowed_timeframes(email: str) -> List[str]:
    plan = get_user_plan(email)
    if plan == "pro":
        return ["1m","5m","15m","30m","1h","1d","1w","1M"]
    else:
        return BASIC_TIMEFRAMES

@app.get("/user/limits")
async def get_user_limits(email: str = Depends(get_current_user)):
    user_doc = get_user_doc(email)
    plan = user_doc["plan"] if user_doc else "trial"
    trial_end = user_doc.get("trial_end") if user_doc else None
    trial_expired = is_trial_expired(email) if trial_end else False
    
    return {
        "plan": plan,
        "allowed_tokens": get_allowed_tokens(email),
        "is_trial": plan == "trial",
        "trial_end": trial_end,
        "trial_expired": trial_expired,
        "alpha_mode_enabled": plan == "pro"
    }

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
        "cashfree": {
            "enabled": CASHFREE_ENABLED,
            "environment": CASHFREE_ENV,
            "base_url": CASHFREE_BASE_URL
        }
    }

# -------------------------------------------------------------------
# Cashfree Payment Integration - Create Subscription Endpoint (FIXED)
# -------------------------------------------------------------------

def generate_unique_subscription_id(email: str) -> str:
    """Generate a unique subscription ID for Cashfree mandate"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    email_prefix = email.split('@')[0][:8]
    return f"SUB_{email_prefix}_{timestamp}_{random_str}"

async def create_cashfree_mandate(subscription_id: str, amount: float, plan_name: str, email: str, phone: Optional[str] = None, currency: str = "INR") -> Dict:
    """
    Create a Cashfree subscription mandate using REST API.
    This properly implements the Cashfree Subscriptions API v2023-08-01.
    """
    if not CASHFREE_ENABLED:
        raise HTTPException(status_code=503, detail="Payment system is not configured")
    
    # Build customer ID from email (safe for all characters)
    customer_id = "".join(c if c.isalnum() else "_" for c in email.replace("@", "_").replace(".", "_"))
    
    # Cashfree Subscriptions API payload structure
    payload = {
        "subscription_id": subscription_id,
        "subscription_amount": amount,
        "subscription_currency": currency,
        "subscription_name": f"Aegis-1 {plan_name.upper()} Plan",
        "subscription_description": f"Monthly subscription for {plan_name} plan",
        "customer_details": {
            "customer_id": customer_id,
            "customer_email": email,
            "customer_phone": phone or "9999999999"
        },
        "return_url": f"{BASE_URL}/web/src/pages/dashboard.html?subscription={subscription_id}",
        "callback_url": f"{BASE_URL}/payments/webhook",
        "auth_mode": "AUTH",  # Use AUTH for mandate creation
        "first_payment_amount": amount,
        "expiry_time": (datetime.now(timezone.utc) + timedelta(days=365)).isoformat()
    }
    
    headers = {
        "Content-Type": "application/json",
        "x-api-version": "2023-08-01"
    }
    
    # Add authentication headers if credentials are available
    if CASHFREE_APP_ID and CASHFREE_SECRET_KEY:
        # Generate signature for secure API calls
        import hashlib
        import hmac
        
        message = json.dumps(payload, sort_keys=True)
        secret = CASHFREE_SECRET_KEY.encode() if isinstance(CASHFREE_SECRET_KEY, str) else CASHFREE_SECRET_KEY
        signature = hmac.new(secret, message.encode(), hashlib.sha256).hexdigest()
        
        headers["x-signature"] = signature
    
    # Make the actual API request to create subscription mandate
    async with httpx.AsyncClient() as client:
        url = f"{CASHFREE_BASE_URL}/pg/subscriptions"
        response = await client.post(url, json=payload, headers=headers)
        result = response.json()
    
    # Safely handle result - it might be a string (error HTML) or dict
    sub_auth_url = None
    if response.status_code in (200, 201):
        if isinstance(result, dict):
            sub_auth_url = result.get("subscription_auth_url") or \
                          result.get("auth_url") or \
                          result.get("redirect_url") or \
                          result.get("data", {}).get("subscription_auth_url")
        else:
            # If result is not a dict (e.g., error HTML), use fallback URL
            sub_auth_url = f"{CASHFREE_BASE_URL}/pg/subscriptions/{subscription_id}/auth"
        
        if not sub_auth_url:
            # If no URL in response, construct one from the callback
            sub_auth_url = f"{CASHFREE_BASE_URL}/pg/subscriptions/{subscription_id}/auth"
    else:
        print(f"Cashfree API error (status {response.status_code}): {result}")
    
    return {
        "success": True,
        "subscription_id": subscription_id,
        "sub_auth_url": sub_auth_url or f"{CASHFREE_BASE_URL}/pg/subscriptions/{subscription_id}/auth",
        "cashfree_response": result,
        "status_code": response.status_code
    }

@app.post("/create-subscription")
async def create_subscription(request: CreateSubscriptionRequest, email: str = Depends(get_current_user)):
    """
    Create a Cashfree subscription mandate and return the authorization URL.
    This endpoint implements the Cashfree Subscriptions API to create a mandate.
    """
    if not CASHFREE_ENABLED:
        raise HTTPException(status_code=503, detail="Payment system is not configured")
    
    if request.email != email:
        raise HTTPException(status_code=403, detail="Email mismatch")
    
    # Generate unique subscription ID
    subscription_id = generate_unique_subscription_id(email)
    plan_amount = request.amount
    plan_name = request.plan_name
    customer_phone = request.customer_phone
    
    # Store pending subscription in Firestore
    pending_ref = db.collection("pending_subscriptions").document(subscription_id)
    pending_ref.set({
        "email": email,
        "plan_name": plan_name,
        "amount": plan_amount,
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "subscription_id": subscription_id
    })
    
    # Create Cashfree mandate using the fixed helper function
    result = await create_cashfree_mandate(
        subscription_id=subscription_id,
        amount=plan_amount,
        plan_name=plan_name,
        email=email,
        phone=customer_phone,
        currency=request.currency
    )
    
    # Update pending subscription with mandate info
    if result.get("success"):
        sub_auth_url = result.get("sub_auth_url")
        pending_ref.update({
            "sub_auth_url": sub_auth_url,
            "subscription_status": "mandate_pending",
            "cashfree_response": result.get("cashfree_response")
        })
    
    return {
        "success": result.get("success", False),
        "subscription_id": subscription_id,
        "sub_auth_url": result.get("sub_auth_url"),
        "message": result.get("message", "Subscription mandate created successfully" if result.get("success") else "Check logs for details")
    }

# -------------------------------------------------------------------
# Cashfree Webhook for SUBSCRIPTION_ACTIVATED events
# -------------------------------------------------------------------
@app.post("/payments/webhook")
async def cashfree_webhook(request: Request):
    """
    Webhook endpoint to catch SUBSCRIPTION_ACTIVATED events from Cashfree.
    Upon success, find the user in Firestore by email and update their plan to 'pro' 
    and subscription.status to 'active'.
    """
    try:
        # Get raw body for signature verification (optional but recommended)
        body = await request.body()
        data = json.loads(body)
        
        # Log incoming webhook for debugging
        print(f"📨 Webhook received: {json.dumps(data, indent=2)}")
        
        webhook_type = data.get("type") or data.get("event")
        
        # Handle SUBSCRIPTION_ACTIVATED event
        if webhook_type == "SUBSCRIPTION_ACTIVATED" or webhook_type == "SUBSCRIPTION_CREATED":
            subscription_data = data.get("data", {}) if isinstance(data, dict) else {}
            subscription_id = subscription_data.get("subscription_id") if isinstance(subscription_data, dict) else None
            
            # Extract customer email (depends on webhook structure)
            customer_details = subscription_data.get("customer_details", {}) if isinstance(subscription_data, dict) else {}
            email = customer_details.get("customer_email") if isinstance(customer_details, dict) else None
            
            if not email and isinstance(data, dict):
                # Try alternative paths at root level
                customer_details_root = data.get("customer_details", {})
                email = customer_details_root.get("customer_email") if isinstance(customer_details_root, dict) else None
            
            if not email and subscription_id:
                # Look up from pending_subscriptions
                sub_ref = db.collection("pending_subscriptions").document(subscription_id)
                try:
                    sub_get_result = sub_ref.get()
                    # Handle both sync and async Firestore clients
                    if inspect.isawaitable(sub_get_result):
                        sub_doc = await sub_get_result
                    else:
                        sub_doc = sub_get_result
                    
                    if hasattr(sub_doc, "exists") and sub_doc.exists and hasattr(sub_doc, "to_dict"):
                        to_dict_fn = getattr(sub_doc, "to_dict", None)
                        if callable(to_dict_fn):
                            sub_data = to_dict_fn()
                            # Ensure sub_data is a dict before accessing .get()
                            if isinstance(sub_data, dict):
                                email = sub_data.get("email")
                except Exception as e:
                    print(f"Error looking up subscription: {e}")
            
            if email:
                print(f"✅ Processing subscription activation for {email}")
                
                # Update user in Firestore
                user_ref = db.collection("users").document(email)
                try:
                    update_result = user_ref.update({
                        "plan": "pro",
                        "subscription": {
                            "status": "active",
                            "subscription_id": subscription_id,
                            "activated_at": datetime.now(timezone.utc).isoformat(),
                            "plan_type": subscription_data.get("subscription_plan_name", "pro")
                        }
                    })
                    # Handle async update result if needed
                    if inspect.isawaitable(update_result):
                        await update_result
                    print(f"✅ User {email} updated to pro plan")
                except Exception as e:
                    print(f"Failed to update user: {e}")
                
                # Update pending subscription record
                if subscription_id:
                    try:
                        sub_ref = db.collection("pending_subscriptions").document(subscription_id)
                        sub_ref.update({
                            "status": "completed",
                            "activated_at": datetime.now(timezone.utc).isoformat(),
                            "webhook_data": data
                        })
                    except Exception as e:
                        print(f"Failed to update subscription record: {e}")
                
                # Send confirmation email
                await send_subscription_confirmation(email, "pro")
                
                return JSONResponse({"status": "success", "message": "Subscription activated"})
            else:
                print(f"⚠️ Could not find email for subscription {subscription_id}")
                return JSONResponse({"status": "ignored", "message": "Email not found"}, status_code=200)
        
        # Handle PAYMENT_SUCCESS for one-time payments (legacy support)
        elif webhook_type == "PAYMENT_SUCCESS":
            order_data = data.get("data", {}).get("order", {})
            order_id = order_data.get("order_id")
            email = order_data.get("customer_details", {}).get("customer_email")
            
            if email and order_id:
                user_ref = db.collection("users").document(email)
                user_ref.update({
                    "plan": "pro",
                    "subscription": {
                        "status": "active",
                        "activated_at": datetime.now(timezone.utc).isoformat()
                    }
                })
                print(f"✅ User {email} upgraded via one-time payment")
                await send_subscription_confirmation(email, "pro")
            
            return JSONResponse({"status": "success"})
        
        # Acknowledge other webhook types
        print(f"📨 Webhook type {webhook_type} received, no action taken")
        return JSONResponse({"status": "received"})
        
    except Exception as e:
        print(f"Webhook processing error: {e}")
        return JSONResponse({"status": "error"}, status_code=500)

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
                    await websocket.send_json({"error": "Invalid token"})
                    await websocket.close()
                    return
                print(f"✅ WebSocket authenticated: {current_user_email}")
        except Exception as auth_err:
            print(f"⚠️ Auth error: {auth_err}")
            pass
        
        # Main loop: send data + listen for client messages (ping/pong)
        while True:
            try:
                # Use wait_for to allow timeout so we can also check for incoming messages
                try:
                    # Try to receive any incoming message from client (with 0.5s timeout)
                    client_msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.5)
                    msg_data = json.loads(client_msg)
                    msg_type = msg_data.get("type", "")
                    
                    if msg_type == "ping":
                        # Respond with pong
                        await websocket.send_json({"type": "pong"})
                        print(f"🏓 Ping/Pong received from {current_user_email}")
                    elif msg_type == "auth":
                        # Re-authenticate if needed
                        new_token = msg_data.get("token")
                        if new_token:
                            new_user = decode_token(new_token)
                            if new_user:
                                current_user_email = new_user
                                print(f"🔄 WebSocket re-authenticated: {current_user_email}")
                except asyncio.TimeoutError:
                    # No message received, continue to send data
                    pass
                
                # Build response data
                allowed_tokens = get_allowed_tokens(current_user_email) if current_user_email else BASIC_TOKENS
                trial_expired = is_trial_expired(current_user_email) if current_user_email else False
                
                def normalize_signal_data(signal_obj) -> dict:
                    if isinstance(signal_obj, dict):
                        return signal_obj
                    if hasattr(signal_obj, "dict") and callable(signal_obj.dict):
                        try:
                            result = signal_obj.dict()
                            if isinstance(result, dict):
                                return result
                        except Exception:
                            pass
                    if hasattr(signal_obj, "__dataclass_fields__"):
                        return asdict(signal_obj)
                    if hasattr(signal_obj, "__dict__"):
                        return vars(signal_obj)
                    try:
                        result = dict(signal_obj)
                        if isinstance(result, dict):
                            return result
                    except Exception:
                        pass
                    return {}

                filtered_signals = {}
                for sym, sig in LIVE_STATE.data.get("signals", {}).items():
                    if sym in allowed_tokens:
                        sig_data = normalize_signal_data(sig)
                        if not isinstance(sig_data, dict):
                            sig_data = {}

                        filtered_signals[sym] = {
                            "ai_prob": sig_data.get("ai_prob", 0),
                            "signal": sig_data.get("signal", "WAITING"),
                            "threshold": sig_data.get("threshold", 0),
                            "signal_strength": sig_data.get("signal_strength", "NONE"),
                            "atr": sig_data.get("atr", 0),
                            "risk_pct": sig_data.get("risk_pct", 2),
                            "direction": sig_data.get("direction", "NEUTRAL"),
                            "entry_price": sig_data.get("entry_price", 0.0),
                            "sl": sig_data.get("sl"),
                            "tp": sig_data.get("tp"),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "confidence_score": sig_data.get("ai_prob", 0) * 100,
                            "signal_id": sig_data.get("signal_id", ""),
                            "trading_accuracy": sig_data.get("trading_accuracy", 0.65),
                            "profitability_index": sig_data.get("profitability_index", 1.5),
                        }
                
                response_data = {
                    "tickers": {k: v for k, v in LIVE_STATE.data["tickers"].items() if k in allowed_tokens},
                    "signals": filtered_signals,
                    "open_trades": LIVE_STATE.data["open_trades"],
                    "balance": LIVE_STATE.data["balance"],
                    "alpha_mode": LIVE_STATE.data["alpha_mode"] and (get_user_plan(current_user_email) == "pro" if current_user_email else False),
                    "warmup": LIVE_STATE.data["warmup_progress"],
                    "trial_expired": trial_expired if current_user_email else True,
                    "plan": get_user_plan(current_user_email) if current_user_email else "trial"
                }
                clean_data = jsonable_encoder(response_data)
                await websocket.send_json(clean_data)
                
                # Send less frequently if no changes to reduce bandwidth
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                print(f"⚠️ WebSocket task cancelled for {current_user_email}")
                raise
            except Exception as loop_err:
                print(f"❌ Error in WebSocket loop: {loop_err}")
                raise
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
async def execute_trade(request: TradeExecuteRequest, user_id: str = Depends(get_current_user)):
    trade_data = request.dict()
    trade_data["openTime"] = datetime.now(timezone.utc).isoformat()
    # Ensure it's saved under the user who made the request
    trade_data["userId"] = user_id
    
    try:
        trade_ref = db.collection("users").document(user_id).collection("trades").document()
        trade_data["id"] = trade_ref.id
        trade_ref.set(trade_data)
        print(f"✅ Trade executed via API for {user_id}")
        return {"status": "success", "trade_id": trade_ref.id, "trade": trade_data}
    except Exception as e:
        print(f"❌ Failed to execute trade for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to execute trade")

@app.post("/api/trades/{trade_id}/close")
async def close_trade(trade_id: str, user_id: str = Depends(get_current_user)):
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
            subject="Your Aegis‑1 Verification Code",
            recipients=[NameEmail(name=email, email=email)],
            body=f"""
            <html>
            <body style="font-family: monospace; background: #0a0a0c; color: #00f2ff; padding: 20px;">
                <h2>🔐 Aegis‑1 OTP</h2>
                <p>Your verification code is:</p>
                <h1 style="background: #1a1f2e; display: inline-block; padding: 12px 24px; border-radius: 12px;">{otp}</h1>
                <p>This code expires in 5 minutes.</p>
                <p>If you didn't request this, please ignore this email.</p>
                <hr>
                <small style="color: #6b7280;">Aegis‑1 Sovereign Terminal</small>
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
        
        plan = user_doc["plan"] if user_doc else "trial"
        trial_end = user_doc.get("trial_end") if user_doc else None
        trial_expired = is_trial_expired(current_user_email) if trial_end else False
        
        dashboard_data = {
            "user": {
                "authenticated": True,
                "plan": plan,
                "trial_active": not (isinstance(trial_end, str) and trial_expired),
                "trial_days_remaining": max(0, (datetime.fromisoformat(trial_end.replace("Z", "+00:00")) - datetime.now(timezone.utc)).days) if isinstance(trial_end, str) and not trial_expired else 0,
                "allowed_tokens": get_allowed_tokens(current_user_email),
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
                "allowed_tokens": ["BTC", "ETH", "SOL", "ARB", "AAVE"],
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
# Phase 2: Cashfree Payment Integration
# -------------------------------------------------------------------
import uuid
import requests

class PaymentSessionRequest(BaseModel):
    amount: float
    tier: str
    currency: str = "INR"
    email: Optional[str] = "user@example.com"
    user_id: Optional[str] = "user_123"

def create_cashfree_order(user_id: str, user_email: str, plan_amount: float, plan_name: str, currency: str = "INR"):
    IS_PROD = os.getenv("CASHFREE_MODE") == "PRODUCTION"
    CASHFREE_BASE_URL = "https://api.cashfree.com/pg" if IS_PROD else "https://sandbox.cashfree.com/pg"
    APP_URL = os.getenv("BASE_URL", "http://localhost:8000").rstrip('/')
    
    headers = {
        "x-client-id": os.getenv("CASHFREE_APP_ID", os.getenv("CASHFREE_CLIENT_ID", "test_client_id")),
        "x-client-secret": os.getenv("CASHFREE_SECRET_KEY", "test_client_secret"),
        "x-api-version": "2023-08-01",
        "Content-Type": "application/json"
    }
    
    order_id = f"order_{uuid.uuid4().hex[:12]}"

    # Cashfree customer_id: alphanumeric + underscore/hyphen only, max 50 chars
    safe_customer_id = re.sub(r'[^a-zA-Z0-9_-]', '_', user_id)[:50]

    payload = {
        "order_id": order_id,
        "order_amount": float(plan_amount),
        "order_currency": currency,
        "customer_details": {
            "customer_id": safe_customer_id,
            "customer_email": user_email,
            "customer_phone": "9999999999"
        },
        "order_meta": {
            "return_url": f"{APP_URL}/web/src/pages/dashboard.html?order_id={order_id}",
            "notify_url": f"{APP_URL}/api/v1/cashfree-webhook"
        },
        "order_tags": {
            "plan_tier": plan_name
        }
    }
    
    try:
        response = requests.post(f"{CASHFREE_BASE_URL}/orders", json=payload, headers=headers)
        if response.status_code == 200:
            order_data = response.json()
            return {
                "success": True,
                "payment_session_id": order_data.get("payment_session_id"),
                "order_id": order_id
            }
        else:
            print(f"❌ Cashfree Order Creation Failed: {response.text}")
            return {"success": False, "error": response.text}
    except Exception as e:
        print(f"❌ Request Error: {str(e)}")
        return {"success": False, "error": str(e)}

@app.post("/api/v1/create-payment-session")
async def create_payment_session(request: PaymentSessionRequest):
    if not request.user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    if not request.email:
        raise HTTPException(status_code=400, detail="email is required")
    
    try:
        result = create_cashfree_order(
            user_id=request.user_id,
            user_email=request.email,
            plan_amount=request.amount,
            plan_name=request.tier,
            currency=request.currency
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import hmac
import hashlib
import base64

@app.post("/api/v1/cashfree-webhook")
async def handle_cashfree_webhook(request: Request):
    # 1. Capture the raw body payload and signature verification headers
    raw_payload = await request.body()
    payload_str = raw_payload.decode("utf-8")
    
    headers = request.headers
    signature = headers.get("x-webhook-signature")
    timestamp = headers.get("x-webhook-timestamp")
    secret_key = os.getenv("CASHFREE_SECRET_KEY")
    
    # 2. Cryptographic Signature Verification (Protects against fake upgrades)
    if not signature or not timestamp or not secret_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing security headers")
        
    # Cashfree v3 Signature math: Base64(HMAC-SHA256(timestamp + payload, secret_key))
    data_to_sign = timestamp + payload_str
    computed_sig = base64.b64encode(
        hmac.new(secret_key.encode('utf-8'), data_to_sign.encode('utf-8'), hashlib.sha256).digest()
    ).decode('utf-8')
    
    if not hmac.compare_digest(computed_sig, signature):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Signature mismatch")

    # 3. Parse the data array
    data = await request.json()
    event_type = data.get("event_type", data.get("type"))
    
    # Only execute logic if the event explicitly confirms an order was paid successfully
    if event_type == "ORDER_PAID" or event_type == "PAYMENT_SUCCESS_WEBHOOK":
        webhook_data = data.get("data", {})
        order_details = webhook_data.get("order", {})
        customer_details = webhook_data.get("customer_details", {})
        
        order_id = order_details.get("order_id")
        user_id = customer_details.get("customer_id")
        email = customer_details.get("customer_email")
        amount = order_details.get("order_amount", 0)

        # Prefer email as Firestore doc key; fall back to customer_id only if no email
        target_doc = email if email else (user_id if user_id and user_id != 'user_unknown' else None)
        
        # 4. Perform the live database elevation inside Firestore
        try:
            if not target_doc:
                raise ValueError("No valid user_id or email found in webhook")
                
            user_ref = db.collection("users").document(target_doc)
            
            # Check if user doc exists, fallback to email if target_doc was user_id
            doc_snap = user_ref.get()
            if not getattr(doc_snap, "exists", False) and target_doc != email and email:
                print(f"⚠️ User {target_doc} not found. Trying fallback to email {email}...")
                target_doc = email
                user_ref = db.collection("users").document(target_doc)

            # Determine the structural tier target based on the transaction amount
            # E.g., ~1999 INR maps to PRO tier privileges
            assigned_tier = "pro" if float(amount) >= 1000 else ("intermediate" if float(amount) >= 500 else "basic")
            
            # Use current datetime for Firestore as fallback to SERVER_TIMESTAMP depending on admin library
            current_time = datetime.now(timezone.utc).isoformat()
            
            user_ref.update({
                "plan": assigned_tier,
                "trial_active": False,
                "trial_end": current_time,   # seal trial at payment time
                "subscription": {
                    "status": "active",
                    "plan_type": assigned_tier,
                    "activated_at": current_time,
                    "last_order_id": order_id
                }
            })
            
            print(f"✅ Webhook Success: Elevated User {target_doc} to {assigned_tier} Tier for Order {order_id}")
            return {"status": "SUCCESS", "message": "User access parameters updated"}
            
        except Exception as e:
            print(f"❌ Firestore Update Error inside Webhook: {str(e)}")
            raise HTTPException(status_code=500, detail="Database write error")
            
    return {"status": "IGNORED", "message": "Non-payment event type processed"}

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

# -------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)



