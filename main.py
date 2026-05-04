import asyncio
import os
import json
import random
import string
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, status, WebSocket, WebSocketDisconnect, Request
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

# -------------------------------------------------------------------
# Load environment variables FIRST
# -------------------------------------------------------------------
load_dotenv()

# -------------------------------------------------------------------
# Security: JWT & Algorithm must be from environment
# -------------------------------------------------------------------
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
if not SECRET_KEY:
    raise RuntimeError("JWT_SECRET_KEY environment variable is not set. Please define it in Railway.")

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

# Container for the OAuth instance; providers are registered lazily
oauth: Any = None

def init_oauth():
    """
    Lazily initialize and register OAuth providers. This avoids network
    calls (fetching provider metadata) during module import which can hang.
    Safe to call multiple times.
    """
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
                # Skip registration if credentials not provided
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
# Engine runner as a background task (non‑blocking) with error handling
# -------------------------------------------------------------------
async def run_engine_background():
    """Run the live engine in the background without blocking startup."""
    from scripts.live_engine import LiveEngine, automated_setup
    import argparse

    # Log BASE_URL at engine startup to verify environment variable
    base_url = os.getenv("BASE_URL", "http://localhost:8000")
    print(f"🚀 Engine background task starting. BASE_URL = {base_url}")

    # Build a dummy argparse namespace with default values
    args = argparse.Namespace()
    args.capital = 10000.0
    args.risk = 2.0
    args.alpha_risk = 3.0
    args.max_position = 2000.0
    args.timeframe = '1m'
    args.alpha_mode = False
    args.proxy = None

    # Use relative path consistent with project root (Linux container safe)
    backtest_dir = Path(__file__).parent / "logs" / "backtests"

    try:
        configs, capital, max_pos, scan_seconds, alpha_mode, alpha_risk, proxy = automated_setup(backtest_dir, args)
    except Exception as e:
        print(f"❌ automated_setup failed: {e}")
        # Add a short delay to avoid log flooding if the failure repeats in a loop
        await asyncio.sleep(1)
        return  # Engine cannot start, but FastAPI stays online

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

    # Start the background state updater
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
            except Exception as e:
                print(f"State update error: {e}")
            await asyncio.sleep(1)

    asyncio.create_task(update_state())

    # Run the engine (may throw exceptions, but we catch them here to keep API alive)
    try:
        await engine.run()
    except Exception as e:
        print(f"⚠️ LiveEngine crashed: {e}")
        # Engine stops, but FastAPI continues; you may want to log or attempt restart later

# -------------------------------------------------------------------
# FastAPI app (lifespan runs engine as background task)
# -------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start the engine in a background task (non‑blocking) – allows Railway healthcheck to pass immediately
    asyncio.create_task(run_engine_background())
    yield  # App is now ready to serve requests

app = FastAPI(title="Aegis-1 by Gatekeeper", lifespan=lifespan)

# Add proxy headers middleware (from uvicorn) to handle X-Forwarded-Proto from Railway's load balancer
app.add_middleware(ProxyHeadersMiddleware, trusted_hosts=["*"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Root redirect – send users to the actual frontend (nested inside web/src/pages)
# Use absolute path to construct the static directory and redirect accordingly.
# -------------------------------------------------------------------
# Determine the absolute path of the current file's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# The frontend files are located in web/src/pages (as per the GitHub structure)
FRONTEND_ROOT = os.path.join(BASE_DIR, "web", "src", "pages")
FRONTEND_ROOT_PATH = Path(FRONTEND_ROOT)

# Ensure the directory exists; if not, create it with a fallback index.html
if not FRONTEND_ROOT_PATH.exists():
    print(f"⚠️ Warning: Frontend directory not found at {FRONTEND_ROOT_PATH}. Creating it.")
    FRONTEND_ROOT_PATH.mkdir(parents=True, exist_ok=True)
    (FRONTEND_ROOT_PATH / "index.html").write_text(
        "<html><body><h1>Aegis‑1</h1><p>Frontend files missing. Please upload the correct static files to web/src/pages/</p></body></html>"
    )

# Mount the deep directory to the /web prefix
app.mount("/web", StaticFiles(directory=str(FRONTEND_ROOT_PATH), html=True), name="web")

@app.get("/")
async def root_redirect():
    # Redirect to the actual index.html inside the nested structure
    return RedirectResponse(url="/web/index.html")

# -------------------------------------------------------------------
# Diagnostic endpoint: list files inside the frontend directory
# -------------------------------------------------------------------
@app.get("/debug-files")
async def debug_files():
    try:
        files = os.listdir(FRONTEND_ROOT_PATH)
        return JSONResponse(content={
            "frontend_root": str(FRONTEND_ROOT_PATH),
            "files": files,
            "exists": FRONTEND_ROOT_PATH.exists(),
            "is_dir": FRONTEND_ROOT_PATH.is_dir() if FRONTEND_ROOT_PATH.exists() else False,
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# -------------------------------------------------------------------
# Serve favicon.ico and other root-level static files from the same frontend directory
# -------------------------------------------------------------------
@app.get("/favicon.ico")
async def favicon():
    favicon_path = FRONTEND_ROOT_PATH / "favicon.ico"
    if favicon_path.exists():
        return FileResponse(favicon_path)
    return Response(status_code=204)

@app.get("/robots.txt")
async def robots():
    robots_path = FRONTEND_ROOT_PATH / "robots.txt"
    if robots_path.exists():
        return FileResponse(robots_path)
    return Response(status_code=204)

# -------------------------------------------------------------------
# Auth helpers (JWT)
# -------------------------------------------------------------------
security = HTTPBearer()

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
                    provider: Optional[str] = None, social_id: Optional[str] = None) -> Dict:
    now = datetime.utcnow().isoformat()
    trial_end = (datetime.utcnow() + timedelta(days=3)).isoformat()
    user_data = {
        "email": email,
        "plan": "trial",
        "trial_end": trial_end,
        "created_at": now,
        "last_login": now,
    }
    if password_hash:
        user_data["password_hash"] = password_hash
    if provider:
        user_data["provider"] = provider
    if social_id:
        user_data["social_id"] = social_id
    db.collection("users").document(email).set(user_data)
    return {"email": email, "plan": "trial", "trial_end": trial_end}

def update_last_login(email: str):
    db.collection("users").document(email).update({"last_login": datetime.utcnow().isoformat()})

def get_or_create_user_from_oauth(email: str, name: str, provider: str, social_id: str) -> Dict:
    user = get_user_doc(email)
    if not user:
        user = create_user_doc(email, provider=provider, social_id=social_id)
    else:
        update_last_login(email)
    return {"email": email, "plan": user["plan"], "trial_end": user["trial_end"]}

# -------------------------------------------------------------------
# OAuth routes – Dynamic redirect URI based on BASE_URL
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
    return RedirectResponse(f"/web/dashboard.html#token={jwt_token}")

# -------------------------------------------------------------------
# Pydantic models
# -------------------------------------------------------------------
class UserRegister(BaseModel):
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Feedback(BaseModel):
    name: str
    email: EmailStr
    message: str

class OTPSendRequest(BaseModel):
    email: EmailStr

class OTPVerifyRequest(BaseModel):
    email: EmailStr
    otp: str

# -------------------------------------------------------------------
# Email/Password User API
# -------------------------------------------------------------------
@app.post("/auth/register")
async def register(user: UserRegister):
    existing = get_user_doc(user.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    create_user_doc(user.email, password_hash=hash_password(user.password))
    token = create_token(user.email)
    return {"access_token": token, "token_type": "bearer"}

@app.post("/auth/login")
async def login(user: UserLogin):
    user_doc = get_user_doc(user.email)
    if not user_doc or "password_hash" not in user_doc:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not verify_password(user.password, user_doc["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    plan = user_doc["plan"]
    trial_end = user_doc.get("trial_end")
    if plan == "trial" and trial_end and datetime.utcnow() > datetime.fromisoformat(trial_end):
        raise HTTPException(status_code=403, detail="Trial expired. Upgrade to continue.")
    update_last_login(user.email)
    token = create_token(user.email)
    return {"access_token": token, "token_type": "bearer"}

@app.get("/auth/me")
async def get_me(email: str = Depends(get_current_user)):
    user_doc = get_user_doc(email)
    if not user_doc:
        raise HTTPException(status_code=404)
    return {"email": email, "plan": user_doc["plan"], "trial_end": user_doc.get("trial_end")}

# -------------------------------------------------------------------
# Subscription & plan limits
# -------------------------------------------------------------------
BASIC_TOKENS = ["BTC/USDT", "ETH/USDT", "SHIB/USDT", "LTC/USDT", "DOGE/USDT"]
BASIC_TIMEFRAMES = ["5m", "15m", "1h"]

def get_user_plan(email: str) -> str:
    user_doc = get_user_doc(email)
    return user_doc["plan"] if user_doc else "trial"

def get_allowed_tokens(email: str) -> List[str]:
    plan = get_user_plan(email)
    if plan == "pro":
        from scripts.live_engine import FLEET
        return FLEET
    else:
        return BASIC_TOKENS

def get_allowed_timeframes(email: str) -> List[str]:
    plan = get_user_plan(email)
    if plan == "pro":
        return ["1m","5m","15m","30m","1h","1d","1w","1M"]
    else:
        return BASIC_TIMEFRAMES

@app.post("/upgrade")
async def upgrade_plan(email: str = Depends(get_current_user)):
    db.collection("users").document(email).update({"plan": "pro"})
    return {"status": "upgraded to pro"}

# -------------------------------------------------------------------
# WebSocket dashboard – flat JSON structure (includes atr and risk_pct)
# -------------------------------------------------------------------
@app.websocket("/ws/dashboard")
async def websocket_dashboard(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = {
                "tickers": LIVE_STATE.data["tickers"],
                "signals": {
                    sym: {
                        "ai_prob": sig.get("ai_prob", 0),
                        "signal": sig.get("signal", "WAITING"),
                        "threshold": sig.get("threshold", 0),
                        "signal_strength": sig.get("signal_strength", "NONE"),
                        "atr": sig.get("atr", 0),
                        "risk_pct": sig.get("risk_pct", 2),
                    }
                    for sym, sig in LIVE_STATE.data["signals"].items()
                },
                "open_trades": LIVE_STATE.data["open_trades"],
                "balance": LIVE_STATE.data["balance"],
                "alpha_mode": LIVE_STATE.data["alpha_mode"],
                "warmup": LIVE_STATE.data["warmup_progress"]
            }
            clean_data = jsonable_encoder(data)
            await websocket.send_json(clean_data)
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")

# -------------------------------------------------------------------
# Alpha Mode toggle
# -------------------------------------------------------------------
@app.post("/alpha/toggle")
async def toggle_alpha_mode(email: str = Depends(get_current_user)):
    if LIVE_STATE.engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready")
    new_state = not LIVE_STATE.engine.alpha_mode
    LIVE_STATE.engine.alpha_mode = new_state
    LIVE_STATE.data["alpha_mode"] = new_state
    return {"alpha_mode": new_state}

# -------------------------------------------------------------------
# Email configuration (for OTP and feedback)
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

@app.post("/feedback")
async def send_feedback(fb: Feedback):
    message = MessageSchema(
        subject=f"Feedback from {fb.name}",
        recipients=[NameEmail(name="Owner", email="owner@aegis.com")],
        body=f"From: {fb.email}\n\n{fb.message}",
        subtype=MessageType.plain,
    )
    await fastmail.send_message(message)
    return {"status": "sent"}

# -------------------------------------------------------------------
# OTP Endpoints (in‑memory store, fixed timezone)
# -------------------------------------------------------------------
otp_store: Dict[str, Dict] = {}

def generate_otp() -> str:
    return ''.join(random.choices(string.digits, k=6))

def is_cooldown_active(email: str) -> bool:
    if email not in otp_store:
        return False
    cooldown = otp_store[email].get("cooldown_until")
    return bool(cooldown and datetime.now(timezone.utc) < cooldown)

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

if __name__ == "__main__":
    # Always pull the PORT from the environment in production
    port = int(os.environ.get("PORT", 8080))
    # Host MUST be 0.0.0.0 for Railway to route external traffic
    uvicorn.run("main:app", host="0.0.0.0", port=port)
