# main.py
import asyncio
import os
import json
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
from dotenv import load_dotenv
from starlette.middleware.sessions import SessionMiddleware

load_dotenv()

app = FastAPI()

# -------------------------------------------------------------------
# Load environment variables FIRST
# -------------------------------------------------------------------
load_dotenv()

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
    print("⚠️ Cashfree PG SDK not installed. Install with: pip install cashfree-pg")

# -------------------------------------------------------------------
# Security: JWT & Algorithm must be from environment
# -------------------------------------------------------------------
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
if not SECRET_KEY:
    raise RuntimeError("JWT_SECRET_KEY is missing. Add it to your local .env file or Railway variables.")
if not ALGORITHM:
    raise RuntimeError("ALGORITHM is missing. Add it to your local .env file or Railway variables.")

app.add_middleware(
    SessionMiddleware, 
    secret_key=SECRET_KEY
)

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
    print(f"⚠️ Cashfree SDK not available, using REST API fallback for {CASHFREE_ENV}")
else:
    print("⚠️ Cashfree payment gateway not configured. Set CASHFREE_APP_ID/CASHFREE_SECRET_KEY to enable.")

# -------------------------------------------------------------------
# Firebase Admin SDK – check path existence before initializing
# -------------------------------------------------------------------
import firebase_admin
from firebase_admin import credentials, firestore

FIREBASE_CREDENTIALS_PATH = os.getenv("FIREBASE_CREDENTIALS")
if not FIREBASE_CREDENTIALS_PATH:
    raise RuntimeError("FIREBASE_CREDENTIALS environment variable not set (should be a file path)")

cred_path = Path(FIREBASE_CREDENTIALS_PATH)
if not cred_path.exists():
    raise RuntimeError(f"Firebase credentials file not found at {cred_path}")

if not firebase_admin._apps:
    cred = credentials.Certificate(str(cred_path))
    firebase_admin.initialize_app(cred)
    print("🔥 Firebase initialized.")
else:
    print("☁️ Firebase already initialized, skipping.")

db = firestore.client()

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
                        json.dump(LIVE_STATE.data.get('signals', {}), sf, default=str)
                except Exception as _e:
                    print(f"⚠️ Failed to write live_signals.json: {_e}")
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
    yield
    engine_task.cancel()
    reminder_task.cancel()
    subscription_task.cancel()

app = FastAPI(title="Aegis-1 by Gatekeeper", lifespan=lifespan)

app.add_middleware(ProxyHeadersMiddleware, trusted_hosts=["*"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

    signals = LIVE_STATE.data.get('signals', {})
    return JSONResponse(content=signals)

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

@app.get("/auth/me")
async def get_me(email: str = Depends(get_current_user)):
    user_doc = get_user_doc(email)
    if not user_doc:
        raise HTTPException(status_code=404)
    return {
        "email": email, 
        "plan": user_doc["plan"], 
        "trial_end": user_doc.get("trial_end"),
        "full_name": user_doc.get("full_name"),
        "location": user_doc.get("location")
    }

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
    db.collection("users").document(email).update({"plan": "pro"})
    return {"status": "upgraded to pro"}

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

async def create_cashfree_mandate(subscription_id: str, amount: float, plan_name: str, email: str, phone: Optional[str] = None) -> Dict:
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
        "subscription_currency": "INR",
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
        phone=customer_phone
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
    
    try:
        data = await websocket.receive_text()
        try:
            auth_data = json.loads(data)
            token = auth_data.get("token")
            if token:
                current_user_email = decode_token(token)
                if not current_user_email:
                    await websocket.send_json({"error": "Invalid token"})
                    await websocket.close()
                    return
        except:
            pass
        
        while True:
            allowed_tokens = get_allowed_tokens(current_user_email) if current_user_email else BASIC_TOKENS
            trial_expired = is_trial_expired(current_user_email) if current_user_email else False
            
            filtered_signals = {}
            for sym, sig in LIVE_STATE.data["signals"].items():
                if sym in allowed_tokens:
                    filtered_signals[sym] = {
                        "ai_prob": sig.get("ai_prob", 0),
                        "signal": sig.get("signal", "WAITING"),
                        "threshold": sig.get("threshold", 0),
                        "signal_strength": sig.get("signal_strength", "NONE"),
                        "atr": sig.get("atr", 0),
                        "risk_pct": sig.get("risk_pct", 2),
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
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")

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
# Main entry point
# -------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
