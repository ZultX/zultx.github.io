# phase4.py — Hardened / optimized (drop-in)
"""
ZULTX Phase 4 — Hardened memory orchestration (patched)
- Compatible with Postgres 'conversations' schema (session_id,user_id,owner,role,content,created_at,ts)
  and older/simple schema (session_id,owner,role,content,ts).
- Loads recent messages by session_id when present, else by owner (user_id or guest:...).
- Keeps in-memory buffer keyed by session_id if present, otherwise owner.
- Defensive SQL and row-access logic for both psycopg2 and sqlite3 rows.
"""
import os
import re
import json
import time
import uuid
import math
import threading
import traceback
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

# optional fast vector recall
SKLEARN_AVAIL = False
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAIL = True
except Exception:
    SKLEARN_AVAIL = False

# DB adapters / env
DB_URL = os.getenv("DATABASE_URL")  # Postgres URL if provided
USE_POSTGRES = bool(DB_URL)
PG_AVAILABLE = False
PG_POOL = None
if USE_POSTGRES:
    try:
        import psycopg2
        import psycopg2.extras
        from psycopg2.pool import SimpleConnectionPool
        PG_AVAILABLE = True
        min_conn = int(os.getenv("PG_POOL_MIN", "1"))
        max_conn = int(os.getenv("PG_POOL_MAX", "6"))
        try:
            PG_POOL = SimpleConnectionPool(min_conn, max_conn, DB_URL)
        except Exception as e:
            print("[phase_4] PG pool init failed:", e)
            PG_POOL = None
            PG_AVAILABLE = False
    except Exception as e:
        print("[phase_4] psycopg2 not available:", e)
        PG_AVAILABLE = False

# sqlite fallback
SQLITE_PATH = os.getenv("ZULTX_MEMORY_DB", "zultx_memory.db")
USE_SQLITE = not PG_AVAILABLE

# (skip Pinecone/OpenAI parts — keep as in your file)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
USE_PINECONE = bool(PINECONE_API_KEY and PINECONE_INDEX)
PINECONE_CLIENT = None
_pine_index = None

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USE_OPENAI_EMBED = False
if OPENAI_API_KEY:
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
        USE_OPENAI_EMBED = True
    except Exception as e:
        print("[phase_4] openai import failed:", e)
        USE_OPENAI_EMBED = False

# reasoning core import (phase_3)
try:
    from phase_3 import ask as phase3_ask
except Exception as e:
    phase3_ask = None
    print("[phase_4] WARNING: phase_3.ask not importable:", e)

# CONFIG knobs
MAX_INJECT_TOKENS = int(os.getenv("ZULTX_MAX_INJECT_TOKENS", "1200"))
TFIDF_TOP_K = int(os.getenv("ZULTX_TFIDF_K", "6"))
PROMOTE_TO_CM_SCORE = float(os.getenv("ZULTX_PROMOTE_CM", "0.60"))
PROMOTE_TO_LTM_SCORE = float(os.getenv("ZULTX_PROMOTE_LTM", "0.85"))
CONFIDENCE_PROMOTE_THRESHOLD = float(os.getenv("ZULTX_CONF_PROMOTE", "0.80"))
STM_EXPIRE_DAYS = int(os.getenv("ZULTX_STM_DAYS", "1"))
CM_EXPIRE_DAYS = int(os.getenv("ZULTX_CM_DAYS", "365"))
EM_DEFAULT_DAYS = int(os.getenv("ZULTX_EM_DAYS", "7"))

# Sensitive patterns (same)
SENSITIVE_PATTERNS = [
    re.compile(r"\b\d{10}\b"),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    re.compile(r"\b(?:card(?:-|\s)?num|credit card|visa|mastercard)\b", re.I),
]

# Locks
_db_lock = threading.RLock()
_RECALL_BUILD_LOCK = threading.Lock()
_CONV_LOCK = threading.Lock()

# debounce
_RECALL_DEBOUNCE_SECONDS = int(os.getenv("RECALL_DEBOUNCE_SECONDS", "60"))
_last_recall_build = 0.0
_recall_build_scheduled = False

# DB init SQL (Postgres + SQLite compatibility)
# Keep memories and other tables identical to your original; conversations is flexible (we create both forms)
_POSTGRES_INIT_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    owner TEXT,
    type TEXT NOT NULL,
    content TEXT NOT NULL,
    source TEXT,
    raw_snippet TEXT,
    created_at TIMESTAMP,
    last_used TIMESTAMP,
    expires_at TIMESTAMP,
    confidence DOUBLE PRECISION,
    importance DOUBLE PRECISION,
    frequency INTEGER,
    memory_score DOUBLE PRECISION,
    tags JSONB,
    consent BOOLEAN,
    metadata JSONB
);
CREATE INDEX IF NOT EXISTS idx_owner_type_score ON memories (owner, type, memory_score DESC NULLS LAST, last_used DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS idx_expires_at ON memories (expires_at);

CREATE TABLE IF NOT EXISTS audit_log (
    id TEXT PRIMARY KEY,
    ts TIMESTAMP,
    action TEXT,
    mem_id TEXT,
    payload JSONB
);

CREATE TABLE IF NOT EXISTS queue_pending (
    id TEXT PRIMARY KEY,
    ts TIMESTAMP,
    payload JSONB
);

-- Conversations: allow both owner and user_id plus created_at and ts
CREATE TABLE IF NOT EXISTS conversations (
    session_id TEXT,
    user_id TEXT,
    owner TEXT,
    role TEXT,
    content TEXT,
    created_at TIMESTAMP,
    ts TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_conversations_session_ts ON conversations (session_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_conversations_session_created_at ON conversations (session_id, created_at DESC);
"""

_SQLITE_INIT_SQL = f"""
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    owner TEXT,
    type TEXT NOT NULL,
    content TEXT NOT NULL,
    source TEXT,
    raw_snippet TEXT,
    created_at TEXT,
    last_used TEXT,
    expires_at TEXT,
    confidence REAL,
    importance REAL,
    frequency INTEGER,
    memory_score REAL,
    tags TEXT,
    consent INTEGER,
    metadata TEXT
);
CREATE INDEX IF NOT EXISTS idx_owner_type_score ON memories (owner, type, memory_score DESC, last_used DESC);
CREATE INDEX IF NOT EXISTS idx_expires_at ON memories (expires_at);

CREATE TABLE IF NOT EXISTS audit_log (
    id TEXT PRIMARY KEY,
    ts TEXT,
    action TEXT,
    mem_id TEXT,
    payload TEXT
);

CREATE TABLE IF NOT EXISTS queue_pending (
    id TEXT PRIMARY KEY,
    ts TEXT,
    payload TEXT
);

CREATE TABLE IF NOT EXISTS conversations (
    session_id TEXT,
    user_id TEXT,
    owner TEXT,
    role TEXT,
    content TEXT,
    created_at TEXT,
    ts TEXT
);
CREATE INDEX IF NOT EXISTS idx_conversations_session_ts ON conversations (session_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_conversations_session_created_at ON conversations (session_id, created_at DESC);
"""

# Utilities
def now_ts() -> str:
    return datetime.utcnow().isoformat()

def parse_ts(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        try:
            return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f")
        except Exception:
            return None

def uuid4() -> str:
    return str(uuid.uuid4())

def clamp(v, a=0.0, b=1.0):
    try:
        v = float(v)
    except Exception:
        v = a
    return max(a, min(b, v))

# DB connection helpers (same as your original)
def _pg_getconn():
    if PG_POOL:
        try:
            conn = PG_POOL.getconn()
            conn.autocommit = False
            return conn
        except Exception:
            pass
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = False
    return conn

def _pg_putconn(conn):
    try:
        if PG_POOL and hasattr(psycopg2, "extensions") and isinstance(conn, psycopg2.extensions.connection):
            PG_POOL.putconn(conn)
            return
    except Exception:
        pass
    try:
        conn.close()
    except Exception:
        pass

def get_db_conn():
    if PG_AVAILABLE and DB_URL:
        return _pg_getconn()
    conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn

# initialize DB
def initialize_db():
    with _db_lock:
        conn = get_db_conn()
        try:
            if PG_AVAILABLE and DB_URL:
                cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                cur.execute(_POSTGRES_INIT_SQL)
                conn.commit()
                cur.close()
                _pg_putconn(conn)
            else:
                cur = conn.cursor()
                cur.executescript(_SQLITE_INIT_SQL)
                conn.commit()
                cur.close()
                conn.close()
        except Exception as e:
            print("[phase_4] initialize_db error:", e)
            try:
                if PG_AVAILABLE and DB_URL:
                    _pg_putconn(conn)
                else:
                    conn.close()
            except Exception:
                pass

initialize_db()

# defensive row accessor to avoid tuple/dict mismatch errors
def _col(row: Any, name: str, idx: int):
    try:
        if row is None:
            return None
        if isinstance(row, dict):
            return row.get(name)
        if hasattr(row, '__getitem__') and callable(getattr(row, "keys", None)) and name in row.keys():
            try:
                return row[name]
            except Exception:
                pass
        try:
            return row[idx]
        except Exception:
            return getattr(row, name, None)
    except Exception:
        return None

# SimpleRecall, other memory helpers kept as-is (use your original implementations)
class SimpleRecall:
    def __init__(self):
        self.vectorizer = None
        self.corpus = []
        self.matrix = None
        self.last_build = 0

    def build_from_db(self):
        global _last_recall_build
        with _RECALL_BUILD_LOCK:
            rows = []
            conn = get_db_conn()
            try:
                if PG_AVAILABLE and DB_URL:
                    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                    cur.execute("SELECT id, owner, content FROM memories WHERE type IN ('CM','LTM')")
                    rows = cur.fetchall()
                    cur.close()
                    _pg_putconn(conn)
                else:
                    cur = conn.cursor()
                    rows = cur.execute("SELECT id, owner, content FROM memories WHERE type IN ('CM','LTM')").fetchall()
                    cur.close()
                    conn.close()
                corpus = []
                for r in rows:
                    try:
                        if isinstance(r, dict):
                            mem_id = r.get("id")
                            owner = r.get("owner")
                            content = r.get("content")
                        else:
                            mem_id = _col(r, "id", 0)
                            owner = _col(r, "owner", 1)
                            content = _col(r, "content", 2)
                    except Exception:
                        continue
                    if mem_id is None:
                        continue
                    corpus.append((str(mem_id), owner, content or ""))
                self.corpus = corpus
                texts = [t for (_id, _owner, t) in self.corpus]
                if SKLEARN_AVAIL and texts and len(texts) > 1:
                    try:
                        self.vectorizer = TfidfVectorizer(max_features=20000)
                        self.matrix = self.vectorizer.fit_transform(texts)
                    except Exception as e:
                        print("[phase_4][recall] tfidf build failed:", e)
                        self.vectorizer = None
                        self.matrix = None
                else:
                    self.vectorizer = None
                    self.matrix = None
                self.last_build = time.time()
                _last_recall_build = self.last_build
            except Exception as e:
                print("[phase_4][recall] build error:", e)

    def retrieve(self, query: str, k: int = TFIDF_TOP_K, owner: Optional[str] = None) -> List[Tuple[str, str, float]]:
        if not self.corpus:
            return []
        filtered = []
        for mem_id, mem_owner, content in self.corpus:
            if owner is None:
                if mem_owner is None:
                    filtered.append((mem_id, content))
            else:
                if mem_owner is None or mem_owner == owner:
                    filtered.append((mem_id, content))
        if not filtered:
            return []
        ids, texts = zip(*filtered)
        if self.vectorizer is not None and self.matrix is not None:
            try:
                qv = self.vectorizer.transform([query])
                fm = self.vectorizer.transform(list(texts))
                sims = cosine_similarity(qv, fm)[0]
                idxs = sims.argsort()[::-1][:k]
                out = []
                for i in idxs:
                    out.append((ids[int(i)], texts[int(i)], float(sims[int(i)])))
                return out
            except Exception:
                pass
        q = (query or "").lower()
        scored = []
        for mem_id, t in zip(ids, texts):
            txt = (t or "").lower()
            score = 0.0
            if q and q in txt:
                score += 1.0
            for w in q.split()[:8]:
                if w and w in txt:
                    score += 0.01
            scored.append((mem_id, t, score))
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:k]

_recall_instance = SimpleRecall()

def _schedule_recall_rebuild(debounce_seconds: int = _RECALL_DEBOUNCE_SECONDS):
    global _recall_build_scheduled, _last_recall_build
    with _RECALL_BUILD_LOCK:
        now = time.time()
        if now - _last_recall_build < debounce_seconds:
            return
        if _recall_build_scheduled:
            return
        _recall_build_scheduled = True

    def _worker():
        global _recall_build_scheduled, _last_recall_build
        try:
            _recall_instance.build_from_db()
        except Exception as e:
            print("[phase_4] recall rebuild worker error:", e)
        finally:
            _recall_build_scheduled = False
            _last_recall_build = time.time()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

try:
    _schedule_recall_rebuild(0)
except Exception:
    pass

# Memory scoring & policy & anonymize: keep your existing implementations
def compute_memory_score(importance: float, frequency: int, created_at: Optional[str], confidence: float) -> float:
    importance = clamp(importance)
    confidence = clamp(confidence)
    norm_frequency = min(1.0, math.log2(1 + max(0, frequency)) / 6.0)
    if created_at:
        created_dt = parse_ts(created_at)
        if created_dt:
            age_days = (datetime.utcnow() - created_dt).total_seconds() / 86400.0
            recency = 1.0 / (1.0 + (age_days / 7.0))
        else:
            recency = 0.5
    else:
        recency = 0.5
    score = (0.40 * importance) + (0.30 * norm_frequency) + (0.20 * recency) + (0.10 * confidence)
    return clamp(score)

def policy_allow_store(content: str) -> Tuple[bool, Optional[str]]:
    c = content or ""
    for p in SENSITIVE_PATTERNS:
        if p.search(c):
            return False, "sensitive_pattern"
    return True, None

def anonymize_if_needed(content: str) -> str:
    s = content
    s = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[email]", s)
    s = re.sub(r"\b\d{10}\b", "[phone]", s)
    return s

# Audit + queue: keep as before (use existing functions)
def audit(action: str, mem_id: Optional[str], payload: Dict[str, Any]):
    try:
        with _db_lock:
            conn = get_db_conn()
            try:
                if PG_AVAILABLE and DB_URL:
                    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                    cur.execute("INSERT INTO audit_log (id, ts, action, mem_id, payload) VALUES (%s, %s, %s, %s, %s)",
                                (uuid4(), datetime.utcnow(), action, mem_id, psycopg2.extras.Json(payload)))
                    conn.commit()
                    cur.close()
                    _pg_putconn(conn)
                else:
                    cur = conn.cursor()
                    cur.execute("INSERT INTO audit_log (id, ts, action, mem_id, payload) VALUES (?, ?, ?, ?, ?)",
                                (uuid4(), now_ts(), action, mem_id, json.dumps(payload)))
                    conn.commit()
                    cur.close()
                    conn.close()
            except Exception as e:
                print("[phase_4][audit_error]", e)
                try:
                    if PG_AVAILABLE and DB_URL:
                        _pg_putconn(conn)
                    else:
                        conn.close()
                except Exception:
                    pass
    except Exception:
        print("[phase_4] audit top-level exception")
        traceback.print_exc()

def enqueue_retry(item: Dict[str, Any]):
    try:
        with _db_lock:
            conn = get_db_conn()
            try:
                if PG_AVAILABLE and DB_URL:
                    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                    cur.execute("INSERT INTO queue_pending (id, ts, payload) VALUES (%s, %s, %s)",
                                (uuid4(), datetime.utcnow(), psycopg2.extras.Json(item)))
                    conn.commit()
                    cur.close()
                    _pg_putconn(conn)
                else:
                    cur = conn.cursor()
                    cur.execute("INSERT OR REPLACE INTO queue_pending (id, ts, payload) VALUES (?,?,?)",
                                (uuid4(), now_ts(), json.dumps(item)))
                    conn.commit()
                    cur.close()
                    conn.close()
            except Exception as e:
                print("[phase_4][enqueue_error]", e)
                try:
                    if PG_AVAILABLE and DB_URL:
                        _pg_putconn(conn)
                    else:
                        conn.close()
                except Exception:
                    pass
        audit("queue_enqueue", None, {"item": str(item)[:200]})
    except Exception:
        pass

# Conversations persistence (robust)
# Key design: support both schemas and allow session-less owner writes.
def persist_conversation(session_id: Optional[str], owner: Optional[str], role: str, content: str, ts_iso: Optional[str] = None):
    ts = ts_iso or now_ts()
    # compute user_id column if owner is not guest:...
    user_id_val = None
    if owner and not str(owner).startswith("guest:"):
        user_id_val = owner
    try:
        with _db_lock:
            conn = get_db_conn()
            try:
                if PG_AVAILABLE and DB_URL:
                    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                    cur.execute(
                        "INSERT INTO conversations (session_id, user_id, owner, role, content, created_at, ts) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                        (session_id, user_id_val, owner, role, content, datetime.fromisoformat(ts) if isinstance(ts, str) else ts, datetime.fromisoformat(ts) if isinstance(ts, str) else ts)
                    )
                    conn.commit()
                    cur.close()
                    _pg_putconn(conn)
                else:
                    cur = conn.cursor()
                    cur.execute(
                        "INSERT INTO conversations (session_id, user_id, owner, role, content, created_at, ts) VALUES (?,?,?,?,?,?,?)",
                        (session_id, user_id_val, owner, role, content, ts, ts)
                    )
                    conn.commit()
                    cur.close()
                    conn.close()
            except Exception as e:
                print("[phase_4] persist_conversation error:", e)
                try:
                    if PG_AVAILABLE and DB_URL:
                        _pg_putconn(conn)
                    else:
                        conn.close()
                except Exception:
                    pass
    except Exception:
        pass

def load_recent_messages_from_db(owner: Optional[str], session_id: Optional[str], limit: int = 7):
    """
    Loads recent messages. Priority:
      1) If session_id provided: load rows for that session (ordered by ts/created_at).
      2) Else if owner provided: load rows where user_id=owner OR owner=owner.
      3) Return list of dicts: {"role","content","ts"}
    """
    try:
        conn = get_db_conn()
        results = []
        try:
            if PG_AVAILABLE and DB_URL:
                cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                if session_id:
                    # use COALESCE to prefer ts, fallback to created_at
                    cur.execute(
                        "SELECT role, content, COALESCE(ts, created_at) AS ts FROM conversations WHERE session_id = %s ORDER BY COALESCE(ts, created_at) DESC LIMIT %s",
                        (session_id, limit)
                    )
                elif owner:
                    cur.execute(
                        "SELECT role, content, COALESCE(ts, created_at) AS ts FROM conversations WHERE (user_id = %s OR owner = %s) ORDER BY COALESCE(ts, created_at) DESC LIMIT %s",
                        (owner, owner, limit)
                    )
                else:
                    # nothing to load
                    cur.close()
                    _pg_putconn(conn)
                    return []
                rows = cur.fetchall()
                cur.close()
                _pg_putconn(conn)
            else:
                cur = conn.cursor()
                if session_id:
                    rows = cur.execute(
                        "SELECT role, content, COALESCE(ts, created_at) AS ts FROM conversations WHERE session_id = ? ORDER BY COALESCE(ts, created_at) DESC LIMIT ?",
                        (session_id, limit)
                    ).fetchall()
                elif owner:
                    rows = cur.execute(
                        "SELECT role, content, COALESCE(ts, created_at) AS ts FROM conversations WHERE (user_id = ? OR owner = ?) ORDER BY COALESCE(ts, created_at) DESC LIMIT ?",
                        (owner, owner, limit)
                    ).fetchall()
                else:
                    conn.close()
                    return []
                cur.close()
                conn.close()
            rows = list(rows)
            # debug
            try:
                print(f"[phase_4] load_recent_messages_from_db -> session_id={session_id} owner={owner} rows={len(rows)}")
            except Exception:
                pass
            # rows are in descending order; we want oldest->newest in buffer
            rows.reverse()
            for r in rows:
                if isinstance(r, dict):
                    results.append({"role": r.get("role"), "content": r.get("content"), "ts": r.get("ts")})
                else:
                    role = _col(r, "role", 0)
                    content = _col(r, "content", 1)
                    ts = _col(r, "ts", 2)
                    results.append({"role": role, "content": content, "ts": ts})
        except Exception as e:
            print("[phase_4] load_recent_messages_from_db error:", e)
            try:
                if PG_AVAILABLE and DB_URL:
                    _pg_putconn(conn)
                else:
                    conn.close()
            except Exception:
                pass
        return results
    except Exception:
        return []

# CRUD memories + index refresh trigger (keep your existing implementations)
def insert_memory(mem: Dict[str, Any]) -> str:
    with _db_lock:
        conn = get_db_conn()
        try:
            if PG_AVAILABLE and DB_URL:
                cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                cur.execute("""
                    INSERT INTO memories
                    (id, owner, type, content, source, raw_snippet, created_at, last_used, expires_at,
                     confidence, importance, frequency, memory_score, tags, consent, metadata)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (id) DO UPDATE SET
                      content = EXCLUDED.content,
                      last_used = EXCLUDED.last_used,
                      frequency = EXCLUDED.frequency,
                      memory_score = EXCLUDED.memory_score,
                      tags = EXCLUDED.tags,
                      metadata = EXCLUDED.metadata
                """, (
                    mem.get("id"),
                    mem.get("owner"),
                    mem.get("type"),
                    mem.get("content"),
                    mem.get("source"),
                    mem.get("raw_snippet"),
                    datetime.fromisoformat(mem.get("created_at")) if mem.get("created_at") else None,
                    datetime.fromisoformat(mem.get("last_used")) if mem.get("last_used") else None,
                    datetime.fromisoformat(mem.get("expires_at")) if mem.get("expires_at") else None,
                    float(mem.get("confidence", 0.0)),
                    float(mem.get("importance", 0.0)),
                    int(mem.get("frequency", 1)),
                    float(mem.get("memory_score", 0.0)),
                    psycopg2.extras.Json(mem.get("tags") or []),
                    bool(mem.get("consent", True)),
                    psycopg2.extras.Json(mem.get("metadata") or {})
                ))
                conn.commit()
                cur.close()
                _pg_putconn(conn)
            else:
                cur = conn.cursor()
                cur.execute("""
                    INSERT OR REPLACE INTO memories
                    (id, owner, type, content, source, raw_snippet, created_at, last_used, expires_at,
                     confidence, importance, frequency, memory_score, tags, consent, metadata)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    mem.get("id"),
                    mem.get("owner"),
                    mem.get("type"),
                    mem.get("content"),
                    mem.get("source"),
                    mem.get("raw_snippet"),
                    mem.get("created_at"),
                    mem.get("last_used"),
                    mem.get("expires_at"),
                    float(mem.get("confidence", 0.0)),
                    float(mem.get("importance", 0.0)),
                    int(mem.get("frequency", 1)),
                    float(mem.get("memory_score", 0.0)),
                    json.dumps(mem.get("tags") or []),
                    1 if mem.get("consent", True) else 0,
                    json.dumps(mem.get("metadata") or {})
                ))
                conn.commit()
                cur.close()
                conn.close()
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            raise
    audit("create_memory", mem.get("id"), {"owner": mem.get("owner"), "summary": (mem.get("content") or "")[:140]})
    _schedule_recall_rebuild()
    try:
        if USE_PINECONE and PINECONE_CLIENT and mem.get("type") in ("CM", "LTM"):
            t = threading.Thread(target=_pine_upsert_safe, args=(mem,), daemon=True)
            t.start()
    except Exception:
        pass
    return mem.get("id")

def _pine_upsert_safe(mem):
    try:
        text = mem.get("content", "")
        vec = get_embedding_for_text(text)
        if vec is not None and _pine_index is not None:
            try:
                _pine_index.upsert([(mem.get("id"), vec, {"owner": mem.get("owner") or None, "type": mem.get("type")})])
            except Exception as e:
                audit("pinecone_upsert_failed", mem.get("id"), {"err": str(e)})
    except Exception as e:
        audit("pinecone_embed_err", mem.get("id"), {"err": str(e)})

def update_memory_last_used(mem_id: str):
    with _db_lock:
        conn = get_db_conn()
        try:
            if PG_AVAILABLE and DB_URL:
                cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                cur.execute("SELECT importance, confidence, frequency, created_at FROM memories WHERE id = %s", (mem_id,))
                row = cur.fetchone()
            else:
                cur = conn.cursor()
                cur.execute("SELECT importance, confidence, frequency, created_at FROM memories WHERE id = ?", (mem_id,))
                row = cur.fetchone()
            cur.close()
            if PG_AVAILABLE and DB_URL:
                _pg_putconn(conn)
            else:
                conn.close()
            if not row:
                return
            if isinstance(row, dict):
                importance = row.get("importance")
                confidence = row.get("confidence")
                frequency = row.get("frequency")
                created_at = row.get("created_at")
            else:
                importance = _col(row, "importance", 0)
                confidence = _col(row, "confidence", 1)
                frequency = _col(row, "frequency", 2)
                created_at = _col(row, "created_at", 3)
            importance = float(importance) if importance is not None else 0.5
            confidence = float(confidence) if confidence is not None else 0.5
            frequency = int(frequency) if frequency is not None else 1
            frequency += 1
            new_score = compute_memory_score(importance, frequency, created_at, confidence)
            conn2 = get_db_conn()
            try:
                if PG_AVAILABLE and DB_URL:
                    cur2 = conn2.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                    cur2.execute("UPDATE memories SET last_used = %s, frequency = %s, memory_score = %s WHERE id = %s",
                                 (datetime.utcnow(), frequency, new_score, mem_id))
                    conn2.commit()
                    cur2.close()
                    _pg_putconn(conn2)
                else:
                    cur2 = conn2.cursor()
                    cur2.execute("UPDATE memories SET last_used = ?, frequency = ?, memory_score = ? WHERE id = ?",
                                 (now_ts(), frequency, new_score, mem_id))
                    conn2.commit()
                    cur2.close()
                    conn2.close()
            except Exception:
                try:
                    conn2.rollback()
                except Exception:
                    pass
                try:
                    if PG_AVAILABLE and DB_URL:
                        _pg_putconn(conn2)
                    else:
                        conn2.close()
                except Exception:
                    pass
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            try:
                if PG_AVAILABLE and DB_URL:
                    _pg_putconn(conn)
                else:
                    conn.close()
            except Exception:
                pass
            raise
    audit("update_last_used", mem_id, {"ts": now_ts(), "frequency": frequency, "memory_score": new_score})

def delete_owner_name_memories(owner: Optional[str]):
    with _db_lock:
        conn = get_db_conn()
        try:
            if PG_AVAILABLE and DB_URL:
                cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                if owner is None:
                    cur.execute("DELETE FROM memories WHERE content LIKE 'name:%' AND owner IS NULL")
                else:
                    cur.execute("DELETE FROM memories WHERE content LIKE 'name:%' AND owner = %s", (owner,))
                conn.commit()
                cur.close()
                _pg_putconn(conn)
            else:
                cur = conn.cursor()
                if owner is None:
                    cur.execute("DELETE FROM memories WHERE content LIKE 'name:%' AND owner IS NULL")
                else:
                    cur.execute("DELETE FROM memories WHERE content LIKE 'name:%' AND owner = ?", (owner,))
                conn.commit()
                cur.close()
                conn.close()
        finally:
            pass
    audit("delete_name_memory", None, {"owner": owner})
    _schedule_recall_rebuild()

def get_memory(mem_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db_conn()
    try:
        if PG_AVAILABLE and DB_URL:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute("SELECT * FROM memories WHERE id = %s", (mem_id,))
            row = cur.fetchone()
            cur.close()
            _pg_putconn(conn)
            if not row:
                return None
            r = dict(row)
            if isinstance(r.get("tags"), str):
                try: r["tags"] = json.loads(r["tags"])
                except Exception: r["tags"] = []
            r["tags"] = r.get("tags") or []
            if isinstance(r.get("metadata"), str):
                try: r["metadata"] = json.loads(r["metadata"])
                except Exception: r["metadata"] = {}
            r["metadata"] = r.get("metadata") or {}
            for k in ("created_at", "last_used", "expires_at"):
                if r.get(k) and not isinstance(r.get(k), str):
                    try:
                        r[k] = r[k].isoformat()
                    except Exception:
                        pass
            return r
        else:
            cur = conn.cursor()
            cur.execute("SELECT * FROM memories WHERE id = ?", (mem_id,))
            row = cur.fetchone()
            cur.close()
            conn.close()
            if not row:
                return None
            r = dict(row)
            r["tags"] = json.loads(r["tags"]) if r["tags"] else []
            r["metadata"] = json.loads(r["metadata"]) if r["metadata"] else {}
            return r
    finally:
        pass

def list_memories(limit: int = 50, owner: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_db_conn()
    try:
        out = []
        if PG_AVAILABLE and DB_URL:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            if owner is None:
                cur.execute("SELECT * FROM memories ORDER BY memory_score DESC NULLS LAST, last_used DESC NULLS LAST LIMIT %s", (limit,))
            else:
                cur.execute("SELECT * FROM memories WHERE owner = %s ORDER BY memory_score DESC NULLS LAST, last_used DESC NULLS LAST LIMIT %s", (owner, limit))
            rows = cur.fetchall()
            cur.close()
            _pg_putconn(conn)
            for r in rows:
                rr = dict(r)
                if isinstance(rr.get("tags"), str):
                    try: rr["tags"] = json.loads(rr["tags"])
                    except Exception: rr["tags"] = []
                rr["tags"] = rr.get("tags") or []
                if isinstance(rr.get("metadata"), str):
                    try: rr["metadata"] = json.loads(rr["metadata"])
                    except Exception: rr["metadata"] = {}
                rr["metadata"] = rr.get("metadata") or {}
                for k in ("created_at", "last_used", "expires_at"):
                    if rr.get(k) and not isinstance(rr.get(k), str):
                        try: rr[k] = rr[k].isoformat()
                        except Exception: pass
                out.append(rr)
            return out
        else:
            cur = conn.cursor()
            if owner is None:
                rows = cur.execute("SELECT * FROM memories ORDER BY memory_score DESC, last_used DESC LIMIT ?", (limit,)).fetchall()
            else:
                rows = cur.execute("SELECT * FROM memories WHERE owner = ? ORDER BY memory_score DESC, last_used DESC LIMIT ?", (owner, limit)).fetchall()
            cur.close()
            conn.close()
            for r in rows:
                d = dict(r)
                d["tags"] = json.loads(d["tags"]) if d["tags"] else []
                d["metadata"] = json.loads(d["metadata"]) if d["metadata"] else {}
                out.append(d)
            return out
    finally:
        pass

# Extractor & promotion: keep as in original (rule_based_extract etc.)
def rule_based_extract(user_input: str, assistant_text: str) -> List[Dict[str, Any]]:
    candidates = []
    text = (user_input or "") + "\n\n" + (assistant_text or "")
    for m in re.finditer(r"\bremember(?: that)?\s*[:\-]?\s*(.+?)(?:\.|$|\n)", text, flags=re.I):
        content = m.group(1).strip()
        if content:
            candidates.append({"content": content, "suggested_type": "CM", "importance": 0.9, "confidence": 0.9, "reason": "explicit_remember"})
    for m in re.finditer(r"\b(?:remember|save)\s+(?:permanent|permanently|forever)\s*[:\-]?\s*(.+?)(?:\.|$|\n)", text, flags=re.I):
        content = m.group(1).strip()
        if content:
            candidates.append({"content": content, "suggested_type": "LTM", "importance": 0.95, "confidence": 0.95, "reason": "explicit_permanent"})
    for m in re.finditer(r"\bI (?:prefer|like|love|hate|want)\s+(.*?)(?:\.|$|\n)", text, flags=re.I):
        content = m.group(1).strip()
        if content and len(content) < 200:
            candidates.append({"content": "pref:" + content, "suggested_type": "CM", "importance": 0.7, "confidence": 0.7, "reason": "preference_statement"})
    m = re.search(r"\b(?:my name is|i am)\s+([A-Za-z][a-zA-Z]{1,30})\b", text, flags=re.I)
    if m:
        name = m.group(1).strip().title()
        candidates.append({"content": f"name:{name}", "suggested_type": "CM", "importance": 0.90, "confidence": 0.90, "reason": "self_identify"})
    return candidates

def llm_assisted_extract_stub(user_input: str, assistant_text: str) -> List[Dict[str, Any]]:
    return []

def extract_candidates(user_input: str, assistant_text: str) -> List[Dict[str, Any]]:
    cands = rule_based_extract(user_input, assistant_text)
    cands += llm_assisted_extract_stub(user_input, assistant_text)
    out = []
    for c in cands:
        content = c.get("content", "").strip()
        if not content: continue
        allowed, reason = policy_allow_store(content)
        if not allowed:
            content2 = anonymize_if_needed(content)
            if content2 != content:
                content = content2
                c["confidence"] = min(0.6, float(c.get("confidence", 0.6)))
            else:
                audit("blocked_candidate", None, {"content": content[:200], "reason": reason})
                continue
        out.append({"content": content, "suggested_type": c.get("suggested_type","CM"), "importance": float(c.get("importance",0.5)), "confidence": float(c.get("confidence",0.5)), "reason": c.get("reason","extracted")})
    return out

def decide_storage_for_candidate(candidate: Dict[str, Any]) -> str:
    suggested = candidate.get("suggested_type", "CM")
    importance = clamp(float(candidate.get("importance", 0.5)))
    confidence = clamp(float(candidate.get("confidence", 0.5)))
    mem_score = compute_memory_score(importance, 1, now_ts(), confidence)
    candidate["memory_score"] = mem_score
    if re.search(r"\btoday\b|\btomorrow\b|\bon\b\s+\w+\s+\d{1,2}\b", candidate["content"], flags=re.I):
        return "EM"
    if suggested == "LTM" and mem_score >= PROMOTE_TO_LTM_SCORE and confidence >= 0.85:
        return "LTM"
    if mem_score >= PROMOTE_TO_CM_SCORE and confidence >= CONFIDENCE_PROMOTE_THRESHOLD:
        return "CM"
    return "STM"

# Conversation buffer (in-memory + persisted)
_CONV_BUFFERS: Dict[str, List[Dict[str, str]]] = {}
CONV_BUFFER_LIMIT = int(os.getenv("CONV_BUFFER_LIMIT", "7"))

def _buffer_key(session_id: Optional[str], owner: Optional[str]) -> Optional[str]:
    # prefer session_id as key, else owner string
    if session_id:
        return f"session:{session_id}"
    if owner:
        return f"owner:{owner}"
    return None

def add_to_conversation_buffer(session_id: Optional[str], role: str, content: str, owner: Optional[str] = None):
    key = _buffer_key(session_id, owner)
    if not key:
        return
    ts = now_ts()
    with _CONV_LOCK:
        buf = _CONV_BUFFERS.get(key) or []
        buf.append({"role": role, "content": content, "ts": ts})
        if len(buf) > CONV_BUFFER_LIMIT:
            buf = buf[-CONV_BUFFER_LIMIT:]
        _CONV_BUFFERS[key] = buf
    # persist to DB
    try:
        persist_conversation(session_id, owner, role, content, ts_iso=ts)
    except Exception:
        pass

def get_recent_chat_block(session_id: Optional[str], owner: Optional[str] = None) -> str:
    key = _buffer_key(session_id, owner)
    if not key:
        return ""
    with _CONV_LOCK:
        buf = _CONV_BUFFERS.get(key, [])
    if not buf:
        try:
            rows = load_recent_messages_from_db(owner, session_id, limit=CONV_BUFFER_LIMIT)
            if rows:
                with _CONV_LOCK:
                    _CONV_BUFFERS[key] = rows
                    buf = rows
        except Exception:
            pass
    if not buf:
        return ""
    lines = []
    for m in buf:
        lines.append(f"{m['role'].upper()}: {m['content']}")
    return "-- RECENT CHAT --\n" + "\n".join(lines) + "\n-- END RECENT CHAT --\n\n"

# Retrieval & injection (kept mostly as-is)
def retrieve_relevant_memories(user_input: str, owner: Optional[str], max_tokens_budget: int = MAX_INJECT_TOKENS) -> List[Dict[str, Any]]:
    if owner is None:
        return []
    if _recall_instance.last_build == 0:
        _schedule_recall_rebuild(0)
    recs = _recall_instance.retrieve(user_input or "", k=TFIDF_TOP_K, owner=owner)
    out = []
    token_budget = max_tokens_budget
    for mem_id, content, sim in recs:
        mem = get_memory(mem_id)
        if not mem:
            continue
        mem_owner = mem.get("owner")
        if mem_owner is not None and mem_owner != owner:
            continue
        if mem.get("type") == "EM" and mem.get("expires_at"):
            exp = parse_ts(mem.get("expires_at"))
            if exp and exp < datetime.utcnow():
                continue
        if (mem.get("memory_score") or 0.0) < 0.20:
            continue
        if (mem.get("content") or "").startswith("name:"):
            if not re.search(r'\b(name|who am i|what is my name|my name)\b', (user_input or "").lower()):
                continue
        snippet = mem.get("content", "")
        approx_tokens = max(1, int(len(snippet) / 6))
        if token_budget - approx_tokens < 0:
            break
        token_budget -= approx_tokens
        out.append({"id": mem_id, "content": snippet, "type": mem.get("type"), "memory_score": mem.get("memory_score"), "why_matched": f"tfidf_sim={sim:.3f}", "owner": mem_owner})
        if len(out) >= 6:
            break
    return out

def build_memory_injection_block(memories: List[Dict[str, Any]]) -> str:
    if not memories:
        return ""
    parts = ["-- INJECTED MEMORIES (distilled) --"]
    for m in memories:
        content = m['content'] or ""
        short = (content[:120] + "…") if len(content) > 120 else content
        parts.append(f"- {short}")
    parts.append("-- END INJECTED MEMORIES --\n")
    return "\n".join(parts)

# get_embedding_for_text, phase4_ask: keep your original behavior but wire to new buffer helpers
def get_embedding_for_text(text: str):
    if not text or not USE_OPENAI_EMBED:
        return None
    try:
        resp = openai.Embedding.create(input=[text], model=os.getenv("PHASE4_EMBED_MODEL", "text-embedding-3-small"))
        vec = resp["data"][0]["embedding"]
        return vec
    except Exception as e:
        audit("openai_embed_failed", None, {"err": str(e)})
        return None

def phase4_ask(user_input: str,
               session_id: Optional[str] = None,
               user_id: Optional[str] = None,
               *,
               memory_mode: str = "auto",
               persona: Optional[str] = None,
               mode: Optional[str] = None,
               temperature: Optional[float] = None,
               max_tokens: Optional[int] = None,
               stream: bool = False,
               timeout: int = 30,
               **_kwargs) -> Dict[str, Any]:
    start = time.time()
    # owner here is the identity string (could be user id or 'guest:<sid>')
    owner = user_id if user_id else (f"guest:{session_id}" if session_id else None)

    # debug
    try:
        print(f"[phase4_ask] owner={owner} session_id={session_id} memory_mode={memory_mode}")
    except Exception:
        pass

    if phase3_ask is None:
        return {"answer": "[phase_3 missing] Core unavailable.", "explain": [], "memory_actions": [], "meta": {"latency_ms": int((time.time()-start)*1000), "fallback": True}}

    memory_mode = (memory_mode or "auto").lower()

    # Add user to convo buffer (persisted) using unified key
    if session_id or owner:
        add_to_conversation_buffer(session_id, "user", user_input, owner)

    # STM
    session_stm = _kwargs.get("session_stm") or {}
    stm_block = ""
    if session_stm:
        bullets = [f"{k}: {v}" for k, v in session_stm.items()]
        stm_block = "-- STM --\n" + "\n".join(bullets) + "\n-- END STM --\n\n"

    # determine injections
    if memory_mode == "off":
        inject_memories = []
    elif memory_mode == "manual":
        if re.search(r'\b(watch memory|use memory|remember for this|use my memory)\b', user_input, flags=re.I):
            inject_memories = retrieve_relevant_memories(user_input, owner, max_tokens_budget=MAX_INJECT_TOKENS)
        else:
            inject_memories = []
    else:
        inject_memories = retrieve_relevant_memories(user_input, owner, max_tokens_budget=MAX_INJECT_TOKENS)

    injection = build_memory_injection_block(inject_memories)
    recent_chat_block = get_recent_chat_block(session_id, owner) if (session_id or owner) else ""

    # Prepare final prompt
    prompt_parts = []
    if persona:
        prompt_parts.append(f"[Persona: {persona}]")
    if recent_chat_block:
        prompt_parts.append(recent_chat_block)
    if stm_block:
        prompt_parts.append(stm_block)
    if injection:
        prompt_parts.append(injection)
    prompt_parts.append("[System: Use user preferences and memory to answer. Do not invent personal data.]")
    prompt_parts.append("\nUser: " + (user_input or ""))
    final_prompt = "\n\n".join([p for p in prompt_parts if p])

    # Call reasoning core (phase3)
    try:
        phase3_result = phase3_ask(final_prompt, persona=persona, mode=mode, temperature=temperature, max_tokens=max_tokens, stream=stream, timeout=timeout)
        fallback = False
    except Exception as e:
        try:
            phase3_result = phase3_ask(user_input, persona=persona, mode=mode, temperature=temperature, max_tokens=max_tokens, stream=stream, timeout=timeout)
            fallback = True
        except Exception as e2:
            return {"answer": "ZULTX error: reasoning core failed.", "explain": [], "memory_actions": [], "meta": {"latency_ms": int((time.time()-start)*1000), "fallback": True, "error": str(e2)}}

    answer_text = phase3_result if isinstance(phase3_result, str) else (phase3_result.get("answer") if isinstance(phase3_result, dict) else str(phase3_result))

    # append assistant answer to buffer (persist)
    if session_id or owner:
        add_to_conversation_buffer(session_id, "assistant", answer_text, owner)

    # mark used memories (update last_used) asynchronously
    used_ids = []
    for m in inject_memories:
        mid = m.get("id")
        if not mid:
            continue
        used_ids.append(mid)
        try:
            threading.Thread(target=update_memory_last_used, args=(mid,), daemon=True).start()
        except Exception:
            pass

    # extract + store candidate memories (same as original)
    candidates = extract_candidates(user_input, answer_text)
    memory_actions = []
    for cand in candidates:
        content = cand["content"]
        allowed, block_reason = policy_allow_store(content)
        if not allowed:
            content2 = anonymize_if_needed(content)
            if content2 != content:
                content = content2
                cand["confidence"] = min(0.6, cand.get("confidence", 0.6))
            else:
                audit("blocked_candidate", None, {"content": content[:200], "reason": block_reason})
                continue

        target = decide_storage_for_candidate({**cand})
        mem_id = uuid4()
        created_at = now_ts()
        mem_owner = owner

        if content.startswith("name:"):
            try:
                delete_owner_name_memories(mem_owner)
            except Exception:
                pass

        memory_score = cand.get("memory_score") or compute_memory_score(float(cand.get("importance", 0.5)), 1, created_at, float(cand.get("confidence", 0.5)))
        mem_obj = {
            "id": mem_id,
            "owner": mem_owner,
            "type": target,
            "content": content,
            "source": "extractor",
            "raw_snippet": (content or "")[:800],
            "created_at": created_at,
            "last_used": created_at,
            "expires_at": None,
            "confidence": float(cand.get("confidence", 0.5)),
            "importance": float(cand.get("importance", 0.5)),
            "frequency": 1,
            "memory_score": float(memory_score),
            "tags": [cand.get("reason", "auto")],
            "consent": True,
            "metadata": {"origin": "phase_4_extractor"}
        }

        if target == "STM":
            expires = datetime.utcnow() + timedelta(days=STM_EXPIRE_DAYS)
            mem_obj["expires_at"] = expires.isoformat()
        elif target == "CM":
            expires = datetime.utcnow() + timedelta(days=CM_EXPIRE_DAYS)
            mem_obj["expires_at"] = expires.isoformat()
        elif target == "LTM":
            mem_obj["expires_at"] = None
        elif target == "EM":
            expires = datetime.utcnow() + timedelta(days=EM_DEFAULT_DAYS)
            mem_obj["expires_at"] = expires.isoformat()

        try:
            t = threading.Thread(target=insert_memory, args=(mem_obj,), daemon=True)
            t.start()
            memory_actions.append({"id": mem_id, "action": "queued_bg", "type": target, "owner": mem_owner, "summary": mem_obj["content"][:140]})
        except Exception as e:
            try:
                insert_memory(mem_obj)
                memory_actions.append({"id": mem_id, "action": "created", "type": target, "owner": mem_owner, "summary": mem_obj["content"][:140]})
            except Exception as ee:
                enqueue_retry({"op": "insert_memory", "candidate": mem_obj})
                memory_actions.append({"id": None, "action": "queued", "detail": str(ee)})
                audit("write_failed", None, {"error": str(ee)})

    # cleanup expired memories async
    try:
        threading.Thread(target=cleanup_expired_memories, daemon=True).start()
    except Exception:
        pass

    explain = []
    for m in inject_memories:
        explain.append({"id": m["id"], "content": m["content"], "why": m["why_matched"], "type": m["type"], "owner": m.get("owner")})

    meta = {
        "latency_ms": int((time.time() - start) * 1000),
        "fallback": fallback,
        "used_memory_count": len(used_ids),
        "candidate_count": len(candidates)
    }

    return {"answer": answer_text, "explain": explain, "memory_actions": memory_actions, "meta": meta}

def cleanup_expired_memories():
    with _db_lock:
        conn = get_db_conn()
        try:
            if PG_AVAILABLE and DB_URL:
                cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                cur.execute("DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at < %s", (datetime.utcnow(),))
                deleted = cur.rowcount
                conn.commit()
                cur.close()
                _pg_putconn(conn)
            else:
                cur = conn.cursor()
                now = now_ts()
                cur.execute("DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at < ?", (now,))
                deleted = cur.rowcount
                conn.commit()
                cur.close()
                conn.close()
            if deleted:
                audit("cleanup_expired", None, {"deleted": deleted})
                try:
                    _schedule_recall_rebuild()
                except Exception:
                    pass
        except Exception as e:
            audit("cleanup_failed", None, {"error": str(e)})
            try:
                if PG_AVAILABLE and DB_URL:
                    _pg_putconn(conn)
                else:
                    conn.close()
            except Exception:
                pass

# (Optional) CLI tester left intact for local runs
if __name__ == "__main__":
    print("ZULTX phase_4 hardened tester (sqlite/postgres compatible)")
    while True:
        try:
            ui = input("\nYou: ").strip()
            if ui.lower() in ("exit", "quit"):
                break
            if ui.lower().startswith("listmem"):
                parts = ui.split()
                owner = None
                if len(parts) > 1:
                    owner = parts[1]
                rows = list_memories(50, owner)
                for r in rows:
                    owner_label = (r['owner'][:8] if r['owner'] else 'GLOBAL')
                    print(f"{r['id'][:8]} owner={owner_label} {r['type']} score={r.get('memory_score') or 0:.3f} freq={r.get('frequency')} content={r['content'][:80]}")
                continue
            if ui.lower().startswith("conv "):
                sid = ui.split(maxsplit=1)[1].strip()
                print(get_recent_chat_block(sid))
                continue
            if ui.lower().startswith("forget "):
                target = ui.split(" ", 1)[1].strip()
                mem = get_memory(target)
                if mem:
                    with _db_lock:
                        conn = get_db_conn()
                        try:
                            if PG_AVAILABLE and DB_URL:
                                cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                                cur.execute("DELETE FROM memories WHERE id = %s", (target,))
                                conn.commit()
                                cur.close()
                                _pg_putconn(conn)
                            else:
                                cur = conn.cursor()
                                cur.execute("DELETE FROM memories WHERE id = ?", (target,))
                                conn.commit()
                                cur.close()
                                conn.close()
                            print("Deleted memory", target)
                        except Exception as e:
                            print("delete failed:", e)
                    continue
                print("No memory by that id; use listmem to inspect")
                continue

            res = phase4_ask(ui, session_id="cli_session", user_id="cli_user", memory_mode="auto")
            print("\nZultX:", res["answer"])
            if res.get("explain"):
                print("\nUsed memories:")
                for e in res["explain"]:
                    print(f"- {e['id'][:8]} [{e.get('type')}] {e['content'][:120]} (why:{e['why']})")
            if res.get("memory_actions"):
                print("\nMemory actions:")
                for a in res["memory_actions"]:
                    print("-", a)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("Error:", e)
            traceback.print_exc()
