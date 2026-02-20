# app.py (remade) — ZULTX server with robust conversation persistence
"""
ZULTX — app with GREEN-PROTOCOL (Postgres auth + JWT + guest session support)
This edition fixes the conversations schema to be compatible with phase4.py
and guarantees the in-DB conversation buffer is written for both user and assistant messages.
"""
import os
import glob
import json
import time
import hmac
import base64
import hashlib
import traceback
from typing import Any, Optional, Tuple
from datetime import datetime

from fastapi import FastAPI, Query, Body, HTTPException, Request, Header
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import urllib.parse

# -------------------------
# Required: psycopg2 (no fallback)
# -------------------------
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is required (no fallback). Set DATABASE_URL to your Postgres/Supabase URL.")

try:
    import psycopg2
    import psycopg2.extras
    from psycopg2.pool import SimpleConnectionPool
except Exception as e:
    raise RuntimeError("psycopg2 is required. Install with `pip install psycopg2-binary`") from e

# Optional bcrypt
try:
    import bcrypt  # type: ignore
    BCRYPT_AVAIL = True
except Exception:
    BCRYPT_AVAIL = False

# Prefer phase_4 if available, then fall back to phase_3
ASK_FUNC = None
try:
    # note: user provided phase4 as "phase4" file; import if present
    from phase4 import phase4_ask as phase4_ask_func
    ASK_FUNC = phase4_ask_func
    print("[ZULTX] Using phase_4.phase4_ask()")
except Exception as e:
    try:
        from phase_3 import ask as phase3_ask
        ASK_FUNC = phase3_ask
        print("[ZULTX] Using phase_3.ask()")
    except Exception as e2:
        ASK_FUNC = None
        print("[ZULTX] No external ask() found, using local fallback.")

# -------------------------
# Configs & other paths
# -------------------------
BASE_DIR = os.getcwd()
LETTERS_DIR = os.getenv("ZULTX_LETTERS_DIR", "letters/")
JWT_SECRET = os.getenv("ZULTX_JWT_SECRET", "dev-secret")
JWT_EXP_SECONDS = int(os.getenv("ZULTX_JWT_EXP_SECONDS", 60 * 60 * 24 * 7))  # 7 days

os.makedirs(LETTERS_DIR, exist_ok=True)
os.makedirs("feedback", exist_ok=True)
os.makedirs("tips", exist_ok=True)

# default letter if missing
example_path = os.path.join(LETTERS_DIR, "real.txt")
if not os.path.exists(example_path):
    with open(example_path, "w", encoding="utf-8") as f:
        f.write("HEY!\n\nIf you have any complain kindly mail to 'zultx.service@gmail.com'\n\nIf you like ZultX wanna support it!\n\nWith love,\nAura Sharma\n13\nZultX-Owner.")

# -------------------------
# FastAPI init
# -------------------------
app = FastAPI(title="ZULTX — v1.4 (patched)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# -------------------------
# Postgres pool & init (fail-fast)
# -------------------------
PG_POOL_MIN = int(os.getenv("PG_POOL_MIN", "1"))
PG_POOL_MAX = int(os.getenv("PG_POOL_MAX", "6"))

try:
    PG_POOL = SimpleConnectionPool(PG_POOL_MIN, PG_POOL_MAX, DATABASE_URL)
except Exception as e:
    raise RuntimeError(f"Failed to create Postgres connection pool: {e}") from e

def _get_conn():
    """Get a connection from the pool (caller must put it back with _put_conn)."""
    try:
        conn = PG_POOL.getconn()
        return conn
    except Exception as e:
        raise RuntimeError(f"Failed to acquire DB connection: {e}") from e

def _put_conn(conn):
    try:
        PG_POOL.putconn(conn)
    except Exception:
        try:
            conn.close()
        except Exception:
            pass

# -------------------------
# Schema - users + conversations (shared with phase4)
# -------------------------
# IMPORTANT: We include both owner (used by phase4) and user_id (used by app)
# and both ts and created_at timestamps so both sides are compatible.
# If you already have an older conversations table on Postgres, drop it (or ALTER to add missing columns).
_INIT_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    email TEXT,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_users_username ON users (username);

-- Unified conversations schema:
-- includes: id, session_id, user_id, owner, role, content, created_at, ts
CREATE TABLE IF NOT EXISTS conversations (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT,
    user_id TEXT,
    owner TEXT,
    role TEXT,
    content TEXT,
    created_at TIMESTAMP,
    ts TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_conv_session_ts ON conversations (session_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_conv_session_created_at ON conversations (session_id, created_at DESC);
"""

def initialize_db():
    conn = _get_conn()
    cur = conn.cursor()
    try:
        cur.execute(_INIT_SQL)
        conn.commit()
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        # Fail fast — app should not run without DB
        raise RuntimeError(f"Failed to initialize DB schema: {e}") from e
    finally:
        try:
            cur.close()
        except Exception:
            pass
        _put_conn(conn)

# run init on import; crash on failures
initialize_db()

# -------------------------
# Password hashing helpers
# -------------------------
def hash_password(password: str) -> str:
    if BCRYPT_AVAIL:
        salt = bcrypt.gensalt()
        ph = bcrypt.hashpw(password.encode("utf-8"), salt)
        return ph.decode("utf-8")
    salt = base64.urlsafe_b64encode(os.urandom(16)).decode("utf-8")
    iterations = 200_000
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), iterations)
    return f"pbkdf2${iterations}${salt}${dk.hex()}"

def verify_password(password: str, stored: str) -> bool:
    if stored.startswith("$2b$") or stored.startswith("$2a$") or stored.startswith("$2y$"):
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
# JWT helpers (same logic)
# -------------------------
def _b64u(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")

def _b64u_decode(s: str) -> bytes:
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
# Auth helpers (Postgres-backed)
# -------------------------
def create_user(username: str, password: str, email: Optional[str] = None) -> dict:
    uid = base64.urlsafe_b64encode(os.urandom(9)).decode("utf-8")
    created_at = datetime.utcnow()
    ph = hash_password(password)
    conn = _get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cur.execute(
            "INSERT INTO users (id, username, email, password_hash, created_at) VALUES (%s, %s, %s, %s, %s)",
            (uid, username, email, ph, created_at)
        )
        conn.commit()
    except psycopg2.errors.UniqueViolation:
        conn.rollback()
        raise ValueError("username_taken")
    except Exception as e:
        conn.rollback()
        raise
    finally:
        try:
            cur.close()
        except Exception:
            pass
        _put_conn(conn)
    return {"id": uid, "username": username, "email": email, "created_at": created_at.isoformat()}

def get_user_by_username(username: str) -> Optional[dict]:
    conn = _get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cur.execute("SELECT * FROM users WHERE username = %s", (username,))
        row = cur.fetchone()
        if not row:
            return None
        return dict(row)
    finally:
        try:
            cur.close()
        except Exception:
            pass
        _put_conn(conn)

def get_user_by_id(uid: str) -> Optional[dict]:
    conn = _get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cur.execute("SELECT * FROM users WHERE id = %s", (uid,))
        row = cur.fetchone()
        if not row:
            return None
        return dict(row)
    finally:
        try:
            cur.close()
        except Exception:
            pass
        _put_conn(conn)

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
# Letters / tip / feedback endpoints (unchanged)
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

# -------------------------
# Identity extraction helper
# -------------------------
def extract_user_and_session(request: Request) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (owner_user_or_guest, session_id_or_none)
    Priority:
      - Bearer JWT => owner = user_id (sub)
      - else session_id => owner = 'guest:<session_id>'
      - else owner = None (no persistent memory)
    """
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    user_id = None
    if auth:
        parts = auth.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            payload = verify_jwt(parts[1], JWT_SECRET)
            if payload and payload.get("sub"):
                user_id = payload.get("sub")

    query_session = request.query_params.get("session_id")
    header_session = request.headers.get("x-session-id")
    session_id = query_session or header_session or None

    owner = user_id if user_id else (f"guest:{session_id}" if session_id else None)
    return owner, session_id

# -------------------------
# Conversation persistence helper (keeps schema compatible with phase4)
# -------------------------
def persist_conversation(session_id: Optional[str], owner: Optional[str], role: str, content: str, ts: Optional[datetime] = None):
    """
    Writes a row into conversations. We keep both 'owner' (free-form used by phase4)
    and 'user_id' (real authenticated user id or NULL).
    """
    ts_val = ts or datetime.utcnow()
    # user_id is None for guests of the form "guest:<session_id>"
    user_id_col = None
    if owner and not str(owner).startswith("guest:"):
        user_id_col = owner

    try:
        conn = _get_conn()
        cur = conn.cursor()
        try:
            cur.execute(
                "INSERT INTO conversations (session_id, user_id, owner, role, content, created_at, ts) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                (session_id, user_id_col, owner, role, content, ts_val, ts_val)
            )
            conn.commit()
        except Exception as e:
            try:
                conn.rollback()
            except Exception:
                pass
            # non-fatal — log and continue
            print("[persist_conversation] insert error:", e)
        finally:
            try:
                cur.close()
            except Exception:
                pass
            _put_conn(conn)
    except Exception as e:
        # pool / connectivity problem
        print("[persist_conversation] DB error:", e)

# -------------------------
# Local fallback ask
# -------------------------
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

import asyncio
async def normalize_result_to_text(result: Any) -> str:
    if result is None:
        return local_fallback_ask_plain("")
    if asyncio.iscoroutine(result):
        try:
            result = await result
        except Exception:
            return str(result)
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
# Protected ask endpoint — uses ASK_FUNC if available (phase_4 preferred)
# -------------------------
@app.get("/ask")
async def ask_get(
    request: Request,
    q: str = Query(..., alias="q"),
    mode: str = Query("friend"),
    temperature: Optional[float] = Query(None),
    max_tokens: int = Query(512),
    memory_mode: str = Query("auto")
):
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Missing query")

    try:
        owner, session_id = extract_user_and_session(request)
        source = request.query_params.get("source")
        # Debugging log for production troubleshooting (will appear in Railway logs)
        print(f"[ask] owner={owner} session_id={session_id} memory_mode={memory_mode} using_phase4={ASK_FUNC is not None}")

        # persist user's message (so convo buffer is available to phase4/db)
        try:
            persist_conversation(session_id, owner, "user", q, ts=datetime.utcnow())
        except Exception:
            pass

        # Build kwargs to match phase4_ask signature (it expects user_input, session_id, user_id, ...)
        if source == "suggestion":
            q = f"""
Respond directly and informatively.
Do not greet.
Do not say hello.
Explain user statement when needed.
Answer clearly and concisely.

User request:
{q}
"""
        kwargs = {
            "user_input": q,
            "session_id": session_id,
            "user_id": owner,            # phase4 expects the 'owner' string here (can be guest:... or user id)
            "memory_mode": memory_mode,
            "mode": mode,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }

        if ASK_FUNC is not None:
            # try direct call with kwargs, else fallback to other common signatures
            try:
                result = ASK_FUNC(**kwargs)
            except TypeError:
                try:
                    # maybe phase_3: ask(q, session_id, owner, stream=False)
                    result = ASK_FUNC(q, session_id, owner, False)
                except Exception:
                    try:
                        result = ASK_FUNC(q)
                    except Exception as e:
                        raise
        else:
            result = local_fallback_ask_plain(q)

        text = await normalize_result_to_text(result)

        # persist assistant response
        try:
            persist_conversation(session_id, owner, "assistant", text, ts=datetime.utcnow())
        except Exception:
            pass

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"answer": f"ZULTX error: {str(e)}"}, status_code=500)

    if not isinstance(text, str):
        text = str(text)
    if len(text) > 20000:
        text = text[:20000] + "\n\n...[truncated]"

    return JSONResponse({"answer": text})


@app.get("/suggestions")
async def get_suggestions(request: Request):
    owner, session_id = extract_user_and_session(request)
    
    conn = _get_conn()
    cur = conn.cursor()

    try:
        # 1️⃣ Get user recent topics
        user_topics = []
        if owner:
            cur.execute("""
                SELECT content FROM conversations
                WHERE owner = %s AND role = 'user'
                ORDER BY ts DESC
                LIMIT 10
            """, (owner,))
            rows = cur.fetchall()
            user_topics = [r[0] for r in rows]

        # 2️⃣ Get global trending topics
        cur.execute("""
            SELECT content FROM conversations
            WHERE role = 'user'
            ORDER BY ts DESC
            LIMIT 30
        """)
        global_rows = cur.fetchall()
        global_topics = [r[0] for r in global_rows]

    finally:
        cur.close()
        _put_conn(conn)

    # Clean & compress topics
    def clean_topics(texts):
        cleaned = []
        for t in texts:
            if len(t) > 8 and len(t) < 120:
                cleaned.append(t.strip())
        return cleaned[:5]

    user_topics = clean_topics(user_topics)
    global_topics = clean_topics(global_topics)

    # 3️⃣ If phase4 AI available → generate smart suggestions
    suggestions = []
    if ASK_FUNC:
        prompt = f"""
Generate exactly 3 short homepage suggestion questions for a chat AI.

Rules:
- Each line must start with a relevant emoji.
- Maximum 6 words per suggestion question.
- Make them exciting and very curiosity-driven.
- No numbering.
- No explanations.
- Plain text lines only.
- Sometimes make suggestion questions with trending topics.
- Make clean and short suggestion questions.
- Make new suggest questions each time.
- Only make suggestive curiose question.

Personalize if possible using:
User topics: {user_topics}

Trending topics: {global_topics}
"""
        try:
            result = ASK_FUNC(
                user_input=prompt,
                session_id=None,
                user_id=None,
                stream=False
            )
            text = await normalize_result_to_text(result)
            suggestions = [line.strip() for line in text.split("\n") if line.strip()]
        except:
            pass

    # 4️⃣ Fallback if AI fails
    if not suggestions:
        suggestions = [
            "Explain quantum computing simply",
            "How will AI change the world?",
            "Best study methods for exams",
            "Is multiverse theory real?",
            "How to stay motivated daily?",
            "Will humans colonize Mars?"
        ]

    return {"suggestions": suggestions[:3]}

# health
@app.get("/health")
def health():
    return JSONResponse({"status": "ok"})
