# app.py
"""
ZULTX — app with GREEN-PROTOCOL (SQLite auth + JWT + guest session support)
- Signup / Login (password hashing: bcrypt if available, fallback PBKDF2)
- JWT created/verified with HMAC-SHA256 (no external JWT lib required)
- /ask uses JWT user identity (if present) otherwise uses guest session_id
- Trial mode supported (no login required)
"""
import os
import glob
import json
import time
import hmac
import base64
import hashlib
import sqlite3
import traceback
from typing import Any, Optional
from datetime import datetime, timedelta

from fastapi import FastAPI, Query, Body, HTTPException, Request, Header
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# External modules optional: bcrypt (faster) - fallback to PBKDF2 if not installed
try:
    import bcrypt  # type: ignore
    BCRYPT_AVAIL = True
except Exception:
    BCRYPT_AVAIL = False

# Try optional external ask functions (phase_4 preferred)
ASK_FUNC = None
try:
    from phase4 import phase4_ask as ask_func
    ASK_FUNC = ask_func
    print("[ZULTX] Using phase_4.phase4_ask")
except Exception:
    try:
        from phase_3 import ask as ask_func
        ASK_FUNC = ask_func
        print("[ZULTX] Using phase_3.ask")
    except Exception as e:
        print("[ZULTX] No phase_3/phase_4 ask() found, using internal fallback. Error:", e)
        ASK_FUNC = None

# -------------------------
# Configs & DB paths
# -------------------------
BASE_DIR = os.getcwd()
USERS_DB = os.getenv("ZULTX_USERS_DB", "users.db")
LETTERS_DIR = os.getenv("ZULTX_LETTERS_DIR", "letters")
JWT_SECRET = os.getenv("ZULTX_JWT_SECRET", "dev-secret")
JWT_EXP_SECONDS = int(os.getenv("ZULTX_JWT_EXP_SECONDS", 60 * 60 * 24 * 7))  # 7 days

# Ensure folders
os.makedirs(LETTERS_DIR, exist_ok=True)
os.makedirs("feedback", exist_ok=True)
os.makedirs("tips", exist_ok=True)

# default letter
example_path = os.path.join(LETTERS_DIR, "real.txt")
if not os.path.exists(example_path):
    with open(example_path, "w", encoding="utf-8") as f:
        f.write("HEY!\n\nIf you have any complain kindly mail to 'zultx.service@gmail.com'\n\nIf you like ZultX wanna support it!\n\nWith love,\nAura Sharma\n13\nZultX-Owner.")

# -------------------------
# FastAPI init
# -------------------------
app = FastAPI(title="ZULTX — v1.4 (auth)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://zultx.github.io"
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# serve static if present (optional)
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


# -------------------------
# Users DB helpers
# -------------------------
_INIT_USERS_SQL = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT UNIQUE,
    email TEXT,
    password_hash TEXT,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_user_username ON users (username);
"""

def get_users_conn():
    conn = sqlite3.connect(USERS_DB, check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_users_db():
    conn = get_users_conn()
    cur = conn.cursor()
    cur.executescript(_INIT_USERS_SQL)
    conn.commit()
    conn.close()

initialize_users_db()

# -------------------------
# Password hashing helpers
# -------------------------
def hash_password(password: str) -> str:
    if BCRYPT_AVAIL:
        salt = bcrypt.gensalt()
        ph = bcrypt.hashpw(password.encode("utf-8"), salt)
        return ph.decode("utf-8")
    # fallback PBKDF2 - store as iterations$salt$hex
    salt = base64.urlsafe_b64encode(os.urandom(16)).decode("utf-8")
    iterations = 200_000
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), iterations)
    return f"pbkdf2${iterations}${salt}${dk.hex()}"

def verify_password(password: str, stored: str) -> bool:
    if stored.startswith("$2b$") or stored.startswith("$2a$") or stored.startswith("$2y$"):  # bcrypt
        if not BCRYPT_AVAIL:
            return False
        try:
            return bcrypt.checkpw(password.encode("utf-8"), stored.encode("utf-8"))
        except Exception:
            return False
    if stored.startswith("pbkdf2$"):
        try:
            _, iter_s, salt, hex_dk = stored.split("$", 3)
            iterations = int(iter_s)
            dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), iterations)
            return hmac.compare_digest(dk.hex(), hex_dk)
        except Exception:
            return False
    return False

# -------------------------
# Simple JWT (HMAC-SHA256)
# -------------------------
def _b64u(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")

def _b64u_decode(s: str) -> bytes:
    # pad
    s2 = s + "=" * ((4 - len(s) % 4) % 4)
    return base64.urlsafe_b64decode(s2.encode("utf-8"))

def create_jwt(payload: dict, secret: str, exp_seconds: int = JWT_EXP_SECONDS) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    payload2 = dict(payload)
    payload2["exp"] = int(time.time()) + int(exp_seconds)
    header_b = _b64u(json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    payload_b = _b64u(json.dumps(payload2, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    signing_input = f"{header_b}.{payload_b}".encode("utf-8")
    sig = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    sig_b = _b64u(sig)
    return f"{header_b}.{payload_b}.{sig_b}"

def verify_jwt(token: str, secret: str) -> Optional[dict]:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header_b, payload_b, sig_b = parts
        signing_input = f"{header_b}.{payload_b}".encode("utf-8")
        expected_sig = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
        if not hmac.compare_digest(_b64u(expected_sig), sig_b):
            return None
        payload_json = json.loads(_b64u_decode(payload_b).decode("utf-8"))
        if "exp" in payload_json and int(time.time()) > int(payload_json["exp"]):
            return None
        return payload_json
    except Exception:
        return None

# -------------------------
# Auth helpers
# -------------------------
def create_user(username: str, password: str, email: Optional[str] = None) -> dict:
    uid = base64.urlsafe_b64encode(os.urandom(9)).decode("utf-8")
    created_at = datetime.utcnow().isoformat()
    ph = hash_password(password)
    conn = get_users_conn()
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO users (id, username, email, password_hash, created_at) VALUES (?,?,?,?,?)",
                    (uid, username, email, ph, created_at))
        conn.commit()
    except sqlite3.IntegrityError as e:
        conn.close()
        raise ValueError("username_taken")
    conn.close()
    return {"id": uid, "username": username, "email": email, "created_at": created_at}

def get_user_by_username(username: str) -> Optional[dict]:
    conn = get_users_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return dict(row)

def get_user_by_id(uid: str) -> Optional[dict]:
    conn = get_users_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id = ?", (uid,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return dict(row)

def token_for_user(user_row: dict) -> str:
    payload = {"sub": user_row["id"], "username": user_row["username"]}
    return create_jwt(payload, JWT_SECRET)

# -------------------------
# HTTP endpoints: signup / login / me
# -------------------------
@app.post("/signup")
def signup(payload: dict = Body(...)):
    username = (payload.get("username") or "").strip()
    password = payload.get("password") or ""
    email = payload.get("email")
    if not username or not password:
        raise HTTPException(status_code=400, detail="username and password required")
    try:
        u = create_user(username, password, email)
    except ValueError as e:
        if str(e) == "username_taken":
            return JSONResponse({"ok": False, "error": "username_taken"}, status_code=409)
        raise HTTPException(status_code=500, detail=str(e))
    token = token_for_user(u)
    return JSONResponse({"ok": True, "token": token, "user": {"id": u["id"], "username": u["username"], "email": u["email"]}})

@app.post("/login")
def login(payload: dict = Body(...)):
    username = (payload.get("username") or "").strip()
    password = payload.get("password") or ""
    if not username or not password:
        raise HTTPException(status_code=400, detail="username and password required")

    user = get_user_by_username(username)
    if not user:
        return JSONResponse({"ok": False, "error": "invalid_credentials"}, status_code=401)
    if not verify_password(password, user["password_hash"]):
        return JSONResponse({"ok": False, "error": "invalid_credentials"}, status_code=401)
    token = token_for_user(user)
    return JSONResponse({"ok": True, "token": token, "user": {"id": user["id"], "username": user["username"], "email": user["email"]}})

@app.get("/check_username")
def check_username(username: str):
    if not username.strip():
        return {"available": False}
    user = get_user_by_username(username.strip())
    return {"available": user is None}

@app.get("/me")
def me(authorization: Optional[str] = Header(None)):
    if not authorization:
        return JSONResponse({"authenticated": False})
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return JSONResponse({"authenticated": False})
    payload = verify_jwt(parts[1], JWT_SECRET)
    if not payload:
        return JSONResponse({"authenticated": False})
    user = get_user_by_id(payload.get("sub"))
    if not user:
        return JSONResponse({"authenticated": False})
    return JSONResponse({"authenticated": True, "user": {"id": user["id"], "username": user["username"], "email": user["email"]}})

# -------------------------
# Existing site endpoints (letters, tip, feedback etc.)
# -------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return HTMLResponse("<html><body><h1>ZultX</h1><p>UI missing</p></body></html>")

@app.get("/letters")
def list_letters():
    files = sorted([os.path.basename(p) for p in glob.glob(os.path.join(LETTERS_DIR, "*.txt"))])
    return JSONResponse({"letters": files})

@app.get("/letters/{name}")
def get_letter(name: str):
    if ".." in name or not name.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = os.path.join(LETTERS_DIR, name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Letter not found")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return PlainTextResponse(content)

@app.post("/feedback")
def feedback(payload: dict = Body(...)):
    ts = int(time.time() * 1000)
    fname = f"feedback/{ts}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return JSONResponse({"ok": True})

@app.post("/tip")
def tip(payload: dict = Body(...)):
    try:
        amount = int(payload.get("amount", 10))
        if amount <= 0:
            raise ValueError()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid amount")

    ts = int(time.time())
    tn = urllib.parse.quote_plus("Tip ZULTX")
    upi_uri = f"upi://pay?pa={urllib.parse.quote_plus('9358588509@fam')}&pn=ZULTX&tn={tn}&am={amount}&cu=INR"
    qr_payload = urllib.parse.quote_plus(upi_uri)
    qr_url = f"https://chart.googleapis.com/chart?cht=qr&chs=360x360&chl={qr_payload}"

    order = {"id": f"upi_{ts}", "amount": amount * 100, "currency": "INR"}
    try:
        with open(f"tips/{ts}.json", "w", encoding="utf-8") as f:
            json.dump({"order": order, "upi": "9358588509@fam", "upi_uri": upi_uri, "created_at": ts}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return JSONResponse({"ok": True, "order": order, "upi_link": upi_uri, "qr": qr_url})

@app.get("/health/providers")
def test_providers():
    results = {}

    # OpenAI
    try:
        import requests, os
        r = requests.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}
        )
        results["openai"] = r.status_code == 200
    except:
        results["openai"] = False

    # Mistral
    try:
        r = requests.get(
            "https://api.mistral.ai/v1/models",
            headers={"Authorization": f"Bearer {os.getenv('MISTRAL_API_KEY')}"}
        )
        results["mistral"] = r.status_code == 200
    except:
        results["mistral"] = False

    # Anthropic
    try:
        r = requests.post(
      "https://api.anthropic.com/v1/messages",
      headers={
        "x-api-key": os.getenv("ANTHROPIC_API_KEY"),
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
      },
      json={
        "model": "claude-3-haiku-20240307",
        "max_tokens": 5,
        "messages": [{"role": "user", "content": "hi"}]
      }
      )
    except:
        results["anthropic"] = False

    return results
    
    # Google Gemini
    try:
        key = os.getenv("GOOGLE_API_KEY")
        r = requests.post(
    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={key}",
    json={
        "contents":[{"parts":[{"text":"hi"}]}]
    }
)
    except:
        results["gemini"] = False

    return results

@app.get("/health/pinecone")
def test_pinecone():
    import os
    from pinecone import Pinecone

    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        indexes = pc.list_indexes()
        return {"connected": True, "indexes": indexes}
    except Exception as e:
        return {"connected": False, "error": str(e)}

# -------------------------
# Helper: determine requester identity (JWT or guest session)
# -------------------------
def extract_user_and_session(request: Request) -> (Optional[str], Optional[str]):
    """
    Returns (user_id_or_none, session_id_or_none)
    Priority:
      - If Authorization Bearer token present and valid => user_id (from token), session_id MAY still be provided for convo buffer.
      - If no token: session_id from query or headers is used to create guest owner id "guest:<session_id>".
    """
    # prefer Authorization header
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    user_id = None
    if auth:
        parts = auth.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            payload = verify_jwt(parts[1], JWT_SECRET)
            if payload and payload.get("sub"):
                user_id = payload.get("sub")

    # session id
    # client may pass session_id as query param or header X-Session-Id
    query_session = request.query_params.get("session_id")
    header_session = request.headers.get("x-session-id")
    session_id = query_session or header_session or None

    # if no token -> we create guest owner id based on session_id
    owner = None
    if user_id:
        owner = user_id
    else:
        if session_id:
            owner = f"guest:{session_id}"
        else:
            owner = None
    return owner, session_id

# -------------------------
# Normalizer reused from your original app but slightly improved
# -------------------------
import asyncio
def local_fallback_ask_plain(user_input: str) -> str:
    base = (user_input or "").strip().lower()
    if any(word in base for word in ("sad", "depressed", "unhappy", "down")):
        return ("Hey — I'm really sorry you're feeling sad. You're not alone.\n\n"
                "Three quick things that might help right now:\n"
                "1) Take three slow breaths (inhale 4s, hold 2s, exhale 6s).\n"
                "2) Stretch or stand up for 30 seconds.\n"
                "3) Write one small thing you're grateful for.\n\n"
                "If you want, tell me more — I'm here to listen.")
    return f"Hey. I heard: \"{user_input}\". Be kind to yourself — tell me more and I'll help."

async def normalize_result_to_text(result: Any) -> str:
    if result is None:
        return local_fallback_ask_plain("")

    if asyncio.iscoroutine(result):
        try:
            result = await result
        except Exception:
            return str(result)

    # If it's a dict and has 'answer' prefer that
    if isinstance(result, dict) and "answer" in result:
        return str(result.get("answer") or "")

    if hasattr(result, "__aiter__"):
        parts = []
        try:
            async for p in result:
                parts.append(str(p))
        except Exception:
            pass
        return "".join(parts)

    if hasattr(result, "__iter__") and not isinstance(result, (str, bytes, dict)):
        parts = []
        try:
            for p in result:
                parts.append(str(p))
        except Exception:
            pass
        return "".join(parts)

    return str(result)

# -------------------------
# Protected ask endpoint (uses identity extraction)
# -------------------------
@app.get("/ask")
async def ask_get(request: Request,
                  q: str = Query(..., alias="q"),
                  mode: str = Query("friend"),
                  temperature: Optional[float] = Query(None),
                  max_tokens: int = Query(512),
                  memory_mode: str = Query("auto")):
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Missing query")

    try:
        # extract owner (user or guest) and session_id
        owner, session_id = extract_user_and_session(request)

        # Build kwargs for ASK_FUNC/phase4_ask
        kwargs = {
            "user_input": q,
            "session_id": session_id,
            "user_id": owner,
            "memory_mode": memory_mode,
            "mode": mode,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }

        if ASK_FUNC is not None:
            try:
                # call directly
                result = ASK_FUNC(**kwargs)
            except TypeError:
                # fallback positional
                try:
                    result = ASK_FUNC(q, session_id, owner, False)
                except Exception:
                    result = ASK_FUNC(q)
        else:
            result = local_fallback_ask_plain(q)

        text = await normalize_result_to_text(result)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"answer": f"ZULTX error: {str(e)}"}, status_code=500)

    if not isinstance(text, str):
        text = str(text)
    if len(text) > 20000:
        text = text[:20000] + "\n\n...[truncated]"

    return JSONResponse({"answer": text})

# health
@app.get("/health")
def health():
    return JSONResponse({"status": "ok"})
