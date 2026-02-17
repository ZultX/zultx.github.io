# phase_4.py — Hardened / optimized (drop-in)
"""
ZULTX Phase 4 — Hardened memory orchestration
Key improvements:
- Postgres connection pool (psycopg2) when DATABASE_URL is set
- Conversation buffer persisted to DB (survives Railway restarts)
- Debounced recall index rebuild (async background)
- Safe DB resource handling (putconn / close)
- Non-blocking Pinecone/OpenAI calls guarded
- Minimal blocking on hot path
- Configuration knobs at top for speed tuning
"""
import os
import re
import json
import time
import uuid
import math
import threading
import traceback
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
        # small pool; tune min/max as needed
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

# Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
USE_PINECONE = bool(PINECONE_API_KEY and PINECONE_INDEX)
PINECONE_CLIENT = None
_pine_index = None
if USE_PINECONE:
    try:
        import pinecone
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
        PINECONE_CLIENT = pinecone
        _pine_index = pinecone.Index(PINECONE_INDEX)
        print("[phase_4] Pinecone initialized")
    except Exception as e:
        print("[phase_4] Pinecone init failed:", e)
        PINECONE_CLIENT = None
        _pine_index = None
        USE_PINECONE = False

# OpenAI embeddings (optional)
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

# CONFIG knobs (tune for speed)
MAX_INJECT_TOKENS = int(os.getenv("ZULTX_MAX_INJECT_TOKENS", "1200"))
TFIDF_TOP_K = int(os.getenv("ZULTX_TFIDF_K", "8"))  # smaller K => faster
PROMOTE_TO_CM_SCORE = float(os.getenv("ZULTX_PROMOTE_CM", "0.60"))
PROMOTE_TO_LTM_SCORE = float(os.getenv("ZULTX_PROMOTE_LTM", "0.85"))
CONFIDENCE_PROMOTE_THRESHOLD = float(os.getenv("ZULTX_CONF_PROMOTE", "0.80"))
STM_EXPIRE_DAYS = int(os.getenv("ZULTX_STM_DAYS", "1"))
CM_EXPIRE_DAYS = int(os.getenv("ZULTX_CM_DAYS", "365"))
EM_DEFAULT_DAYS = int(os.getenv("ZULTX_EM_DAYS", "7"))

# Sensitive patterns
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

# debounce settings for recall rebuild (seconds)
_RECALL_DEBOUNCE_SECONDS = int(os.getenv("RECALL_DEBOUNCE_SECONDS", "5"))
_last_recall_build = 0.0
_recall_build_scheduled = False

# ---------------------------
# DB Schema (Postgres + SQLite compatibility)
# - added conversations table to persist convo buffers
# ---------------------------
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

CREATE TABLE IF NOT EXISTS conversations (
    session_id TEXT,
    owner TEXT,
    role TEXT,
    content TEXT,
    ts TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_conversations_session_ts ON conversations (session_id, ts DESC);
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
    owner TEXT,
    role TEXT,
    content TEXT,
    ts TEXT
);
CREATE INDEX IF NOT EXISTS idx_conversations_session_ts ON conversations (session_id, ts DESC);
"""

# ---------------------------
# Utilities
# ---------------------------
def now_ts() -> str:
    return datetime.utcnow().isoformat()

def parse_ts(ts: Optional[str]) -> Optional[datetime]:
    if not ts: return None
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

# ---------------------------
# DB connection helpers
# ---------------------------
def _pg_getconn():
    """Get a connection from pool or direct conn (if pool absent)"""
    if PG_POOL:
        try:
            return PG_POOL.getconn()
        except Exception:
            # fallback to direct connect
            pass
    # fallback direct
    conn = psycopg2.connect(DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    return conn

def _pg_putconn(conn):
    if PG_POOL and isinstance(conn, psycopg2.extensions.connection):
        try:
            PG_POOL.putconn(conn)
            return
        except Exception:
            pass
    try:
        conn.close()
    except Exception:
        pass

def get_db_conn():
    """
    Returns:
      - If Postgres available: a psycopg2 connection (must be returned with _pg_putconn if using pool)
      - Else: sqlite3 connection
    IMPORTANT: callers must close cursor and either call _pg_putconn(conn) (for PG) or conn.close() for sqlite.
    """
    if PG_AVAILABLE and DB_URL:
        return _pg_getconn()
    import sqlite3
    conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn

# ---------------------------
# initialize DB (safe)
# ---------------------------
def initialize_db():
    with _db_lock:
        conn = get_db_conn()
        cur = conn.cursor()
        try:
            if PG_AVAILABLE and DB_URL:
                cur.execute(_POSTGRES_INIT_SQL)
                conn.commit()
            else:
                cur.executescript(_SQLITE_INIT_SQL)
                conn.commit()
        except Exception as e:
            print("[phase_4] initialize_db error:", e)
        finally:
            try:
                if PG_AVAILABLE and DB_URL:
                    _pg_putconn(conn)
                else:
                    conn.close()
            except Exception:
                pass

initialize_db()

# ---------------------------
# Recall index (TF-IDF) with debounced rebuild
# ---------------------------
class SimpleRecall:
    def __init__(self):
        self.vectorizer = None
        self.corpus = []  # list of (mem_id, owner, content)
        self.matrix = None
        self.last_build = 0

    def build_from_db(self):
        global _last_recall_build
        with _RECALL_BUILD_LOCK:
            start = time.time()
            rows = []
            conn = get_db_conn()
            cur = conn.cursor()
            try:
                if PG_AVAILABLE and DB_URL:
                    cur.execute("SELECT id, owner, content FROM memories WHERE type IN ('CM','LTM')")
                    rows = cur.fetchall()
                    # RealDictCursor -> dicts
                    self.corpus = [(r["id"], r.get("owner"), r["content"]) for r in rows]
                else:
                    rows = cur.execute("SELECT id, owner, content FROM memories WHERE type IN ('CM','LTM')").fetchall()
                    self.corpus = [(r["id"], r["owner"], r["content"]) for r in rows]
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
            finally:
                try:
                    if PG_AVAILABLE and DB_URL:
                        _pg_putconn(conn)
                    else:
                        conn.close()
                except Exception:
                    pass
            # debug
            # print("[phase_4] recall built in", time.time()-start, "s, corpus size:", len(self.corpus))

    def retrieve(self, query: str, k: int = TFIDF_TOP_K, owner: Optional[str] = None) -> List[Tuple[str, str, float]]:
        # fast path: empty corpus
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
        # cheap substring fallback
        q = query.lower()
        scored = []
        for mem_id, t in zip(ids, texts):
            txt = (t or "").lower()
            score = 0.0
            if q in txt:
                score += 1.0
            # tiny partial match boost
            for w in q.split()[:8]:
                if w and w in txt:
                    score += 0.01
            scored.append((mem_id, t, score))
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:k]

_recall_instance = SimpleRecall()

def _schedule_recall_rebuild(debounce_seconds: int = _RECALL_DEBOUNCE_SECONDS):
    # schedule a rebuild if not recently done
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

# initial build (background safe)
try:
    _schedule_recall_rebuild(0)
except Exception:
    pass

# ---------------------------
# Memory scoring & policy (same as before)
# ---------------------------
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

# ---------------------------
# Audit + queue
# ---------------------------
def audit(action: str, mem_id: Optional[str], payload: Dict[str, Any]):
    try:
        with _db_lock:
            conn = get_db_conn()
            cur = conn.cursor()
            try:
                ts = datetime.utcnow()
                if PG_AVAILABLE and DB_URL:
                    cur.execute("INSERT INTO audit_log (id, ts, action, mem_id, payload) VALUES (%s, %s, %s, %s, %s)",
                                (uuid4(), ts, action, mem_id, psycopg2.extras.Json(payload)))
                else:
                    cur.execute("INSERT INTO audit_log (id, ts, action, mem_id, payload) VALUES (?, ?, ?, ?, ?)",
                                (uuid4(), now_ts(), action, mem_id, json.dumps(payload)))
                conn.commit()
            except Exception as e:
                print("[phase_4][audit_error]", e)
            finally:
                try:
                    if PG_AVAILABLE and DB_URL:
                        _pg_putconn(conn)
                    else:
                        conn.close()
                except Exception:
                    pass
    except Exception:
        # never crash entire process because of audit
        print("[phase_4] audit top-level exception")
        traceback.print_exc()

def enqueue_retry(item: Dict[str, Any]):
    try:
        with _db_lock:
            conn = get_db_conn()
            cur = conn.cursor()
            try:
                ts = datetime.utcnow()
                if PG_AVAILABLE and DB_URL:
                    cur.execute("INSERT INTO queue_pending (id, ts, payload) VALUES (%s, %s, %s)",
                                (uuid4(), ts, psycopg2.extras.Json(item)))
                else:
                    cur.execute("INSERT OR REPLACE INTO queue_pending (id, ts, payload) VALUES (?,?,?)",
                                (uuid4(), now_ts(), json.dumps(item)))
                conn.commit()
            except Exception as e:
                print("[phase_4][enqueue_error]", e)
            finally:
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

# ---------------------------
# Conversations persistence (buffer survives restarts)
# ---------------------------
def persist_conversation(session_id: str, owner: Optional[str], role: str, content: str, ts_iso: Optional[str] = None):
    ts = ts_iso or now_ts()
    try:
        with _db_lock:
            conn = get_db_conn()
            cur = conn.cursor()
            try:
                if PG_AVAILABLE and DB_URL:
                    cur.execute("INSERT INTO conversations (session_id, owner, role, content, ts) VALUES (%s,%s,%s,%s,%s)",
                                (session_id, owner, role, content, ts))
                else:
                    cur.execute("INSERT INTO conversations (session_id, owner, role, content, ts) VALUES (?,?,?,?,?)",
                                (session_id, owner, role, content, ts))
                conn.commit()
            except Exception as e:
                # swallow but log
                print("[phase_4] persist_conversation error:", e)
            finally:
                try:
                    if PG_AVAILABLE and DB_URL:
                        _pg_putconn(conn)
                    else:
                        conn.close()
                except Exception:
                    pass
    except Exception:
        pass

def load_recent_messages_from_db(owner: str, session_id: str, limit: int = 7):
    try:
        conn = get_db_conn()
        cur = conn.cursor()
        results = []
        try:
            if PG_AVAILABLE and DB_URL:
                cur.execute("SELECT role, content, ts FROM conversations WHERE session_id = %s ORDER BY ts DESC LIMIT %s", (session_id, limit))
                rows = cur.fetchall()
                rows = list(rows)
            else:
                rows = cur.execute("SELECT role, content, ts FROM conversations WHERE session_id = ? ORDER BY ts DESC LIMIT ?", (session_id, limit)).fetchall()
            rows.reverse()
            for r in rows:
                # row may be dict or sqlite Row
                if isinstance(r, dict):
                    results.append({"role": r.get("role"), "content": r.get("content"), "ts": r.get("ts")})
                else:
                    results.append({"role": r[0], "content": r[1], "ts": r[2]})
        except Exception as e:
            print("[phase_4] load_recent_messages_from_db error:", e)
        finally:
            if PG_AVAILABLE and DB_URL:
                _pg_putconn(conn)
            else:
                conn.close()
        return results
    except Exception:
        return []

# ---------------------------
# CRUD memories + index refresh trigger
# ---------------------------
def insert_memory(mem: Dict[str, Any]) -> str:
    with _db_lock:
        conn = get_db_conn()
        cur = conn.cursor()
        try:
            if PG_AVAILABLE and DB_URL:
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
            else:
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
        except Exception as e:
            try:
                conn.rollback()
            except Exception:
                pass
            raise
        finally:
            try:
                if PG_AVAILABLE and DB_URL:
                    _pg_putconn(conn)
                else:
                    conn.close()
            except Exception:
                pass

    audit("create_memory", mem.get("id"), {"owner": mem.get("owner"), "summary": (mem.get("content") or "")[:140]})
    # schedule recall rebuild (debounced) — do not block
    _schedule_recall_rebuild()
    # pinecone upsert in background
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
        cur = conn.cursor()
        try:
            if PG_AVAILABLE and DB_URL:
                cur.execute("SELECT importance, confidence, frequency, created_at FROM memories WHERE id = %s", (mem_id,))
                row = cur.fetchone()
            else:
                cur.execute("SELECT importance, confidence, frequency, created_at FROM memories WHERE id = ?", (mem_id,))
                row = cur.fetchone()
            if not row:
                return
            importance = float(row["importance"] if isinstance(row, dict) else row[0] or 0.5)
            confidence = float(row["confidence"] if isinstance(row, dict) else row[1] or 0.5)
            frequency = int(row["frequency"] if isinstance(row, dict) else row[2] or 1) + 1
            created_at = (row["created_at"] if isinstance(row, dict) else row[3])
            new_score = compute_memory_score(importance, frequency, created_at, confidence)
            if PG_AVAILABLE and DB_URL:
                cur.execute("UPDATE memories SET last_used = %s, frequency = %s, memory_score = %s WHERE id = %s",
                            (datetime.utcnow(), frequency, new_score, mem_id))
            else:
                cur.execute("UPDATE memories SET last_used = ?, frequency = ?, memory_score = ? WHERE id = ?",
                            (now_ts(), frequency, new_score, mem_id))
            conn.commit()
        finally:
            try:
                if PG_AVAILABLE and DB_URL:
                    _pg_putconn(conn)
                else:
                    conn.close()
            except Exception:
                pass
    audit("update_last_used", mem_id, {"ts": now_ts(), "frequency": frequency, "memory_score": new_score})

def delete_owner_name_memories(owner: Optional[str]):
    with _db_lock:
        conn = get_db_conn()
        cur = conn.cursor()
        try:
            if PG_AVAILABLE and DB_URL:
                if owner is None:
                    cur.execute("DELETE FROM memories WHERE content LIKE 'name:%' AND owner IS NULL")
                else:
                    cur.execute("DELETE FROM memories WHERE content LIKE 'name:%' AND owner = %s", (owner,))
            else:
                if owner is None:
                    cur.execute("DELETE FROM memories WHERE content LIKE 'name:%' AND owner IS NULL")
                else:
                    cur.execute("DELETE FROM memories WHERE content LIKE 'name:%' AND owner = ?", (owner,))
            conn.commit()
        finally:
            try:
                if PG_AVAILABLE and DB_URL:
                    _pg_putconn(conn)
                else:
                    conn.close()
            except Exception:
                pass
    audit("delete_name_memory", None, {"owner": owner})
    # schedule rebuild
    _schedule_recall_rebuild()

def get_memory(mem_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db_conn()
    cur = conn.cursor()
    try:
        if PG_AVAILABLE and DB_URL:
            cur.execute("SELECT * FROM memories WHERE id = %s", (mem_id,))
            row = cur.fetchone()
            if not row:
                return None
            r = dict(row)
            if isinstance(r.get("tags"), str):
                try:
                    r["tags"] = json.loads(r["tags"])
                except Exception:
                    r["tags"] = []
            r["tags"] = r.get("tags") or []
            if isinstance(r.get("metadata"), str):
                try:
                    r["metadata"] = json.loads(r["metadata"])
                except Exception:
                    r["metadata"] = {}
            r["metadata"] = r.get("metadata") or {}
            for k in ("created_at", "last_used", "expires_at"):
                if r.get(k) and not isinstance(r.get(k), str):
                    try:
                        r[k] = r[k].isoformat()
                    except Exception:
                        pass
            return r
        else:
            cur.execute("SELECT * FROM memories WHERE id = ?", (mem_id,))
            row = cur.fetchone()
            if not row:
                return None
            r = dict(row)
            r["tags"] = json.loads(r["tags"]) if r["tags"] else []
            r["metadata"] = json.loads(r["metadata"]) if r["metadata"] else {}
            return r
    finally:
        try:
            if PG_AVAILABLE and DB_URL:
                _pg_putconn(conn)
            else:
                conn.close()
        except Exception:
            pass

def list_memories(limit: int = 50, owner: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_db_conn()
    cur = conn.cursor()
    try:
        out = []
        if PG_AVAILABLE and DB_URL:
            if owner is None:
                cur.execute("SELECT * FROM memories ORDER BY memory_score DESC NULLS LAST, last_used DESC NULLS LAST LIMIT %s", (limit,))
            else:
                cur.execute("SELECT * FROM memories WHERE owner = %s ORDER BY memory_score DESC NULLS LAST, last_used DESC NULLS LAST LIMIT %s", (owner, limit))
            rows = cur.fetchall()
            for r in rows:
                rr = dict(r)
                if isinstance(rr.get("tags"), str):
                    try:
                        rr["tags"] = json.loads(rr["tags"])
                    except Exception:
                        rr["tags"] = []
                rr["tags"] = rr.get("tags") or []
                if isinstance(rr.get("metadata"), str):
                    try:
                        rr["metadata"] = json.loads(rr["metadata"])
                    except Exception:
                        rr["metadata"] = {}
                rr["metadata"] = rr.get("metadata") or {}
                for k in ("created_at", "last_used", "expires_at"):
                    if rr.get(k) and not isinstance(rr.get(k), str):
                        try:
                            rr[k] = rr[k].isoformat()
                        except Exception:
                            pass
                out.append(rr)
            return out
        else:
            if owner is None:
                rows = cur.execute("SELECT * FROM memories ORDER BY memory_score DESC, last_used DESC LIMIT ?", (limit,)).fetchall()
            else:
                rows = cur.execute("SELECT * FROM memories WHERE owner = ? ORDER BY memory_score DESC, last_used DESC LIMIT ?", (owner, limit)).fetchall()
            for r in rows:
                d = dict(r)
                d["tags"] = json.loads(d["tags"]) if d["tags"] else []
                d["metadata"] = json.loads(d["metadata"]) if d["metadata"] else {}
                out.append(d)
            return out
    finally:
        try:
            if PG_AVAILABLE and DB_URL:
                _pg_putconn(conn)
            else:
                conn.close()
        except Exception:
            pass

# ---------------------------
# Extractor and promotion decision (unchanged logic, same as previous)
# ---------------------------
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

# ---------------------------
# Conversation buffer (in-memory + persisted)
# ---------------------------
_CONV_BUFFERS: Dict[str, List[Dict[str, str]]] = {}
CONV_BUFFER_LIMIT = int(os.getenv("CONV_BUFFER_LIMIT", "7"))

def add_to_conversation_buffer(session_id: str, role: str, content: str, owner: Optional[str] = None):
    if not session_id:
        return
    ts = now_ts()
    with _CONV_LOCK:
        buf = _CONV_BUFFERS.get(session_id) or []
        buf.append({"role": role, "content": content, "ts": ts})
        if len(buf) > CONV_BUFFER_LIMIT:
            buf = buf[-CONV_BUFFER_LIMIT:]
        _CONV_BUFFERS[session_id] = buf
    # persist immediately (cheap single-row insert)
    try:
        persist_conversation(session_id, owner, role, content, ts_iso=ts)
    except Exception:
        pass

def get_recent_chat_block(session_id: str) -> str:
    if not session_id:
        return ""
    with _CONV_LOCK:
        buf = _CONV_BUFFERS.get(session_id, [])
    # if no in-memory buffer, try to load from DB (Railway restart case)
    if not buf:
        # attempt to load persisted rows (owner optional for now)
        try:
            # use empty owner and limit
            rows = load_recent_messages_from_db(None, session_id, limit=CONV_BUFFER_LIMIT)
            if rows:
                with _CONV_LOCK:
                    _CONV_BUFFERS[session_id] = rows
                    buf = rows
        except Exception:
            pass
    if not buf:
        return ""
    lines = []
    for m in buf:
        lines.append(f"{m['role'].upper()}: {m['content']}")
    return "-- RECENT CHAT --\n" + "\n".join(lines) + "\n-- END RECENT CHAT --\n\n"

# ---------------------------
# Retrieval & injection
# ---------------------------
def retrieve_relevant_memories(user_input: str, owner: Optional[str], max_tokens_budget: int = MAX_INJECT_TOKENS) -> List[Dict[str, Any]]:
    # ensure recall index is built at least once (non-blocking)
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
        if owner is None:
            if mem_owner is not None:
                continue
        else:
            if mem_owner is not None and mem_owner != owner:
                continue
        if mem.get("type") == "EM" and mem.get("expires_at"):
            exp = parse_ts(mem.get("expires_at"))
            if exp and exp < datetime.utcnow():
                continue
        if (mem.get("memory_score") or 0.0) < 0.20:
            continue
        if mem["content"].startswith("name:"):
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

# ---------------------------
# Embedding helper (OpenAI) - guarded and synchronous (safe)
# ---------------------------
def get_embedding_for_text(text: str):
    if not text:
        return None
    if not USE_OPENAI_EMBED:
        return None
    try:
        # prefer modern clients if installed; keep minimal fallback
        resp = openai.Embedding.create(input=[text], model=os.getenv("PHASE4_EMBED_MODEL", "text-embedding-3-small"))
        vec = resp["data"][0]["embedding"]
        return vec
    except Exception as e:
        audit("openai_embed_failed", None, {"err": str(e)})
        return None

# ---------------------------
# Main orchestration (phase4_ask)
# ---------------------------
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
    owner = user_id if user_id is not None else session_id
    if phase3_ask is None:
        return {"answer": "[phase_3 missing] Core unavailable.", "explain": [], "memory_actions": [], "meta": {"latency_ms": int((time.time()-start)*1000), "fallback": True}}

    memory_mode = (memory_mode or "auto").lower()

    # Add user to convo buffer (persisted)
    if session_id:
        add_to_conversation_buffer(session_id, "user", user_input, owner)

    # STM / session_stm
    session_stm = _kwargs.get("session_stm") or {}
    stm_block = ""
    if session_stm:
        bullets = [f"{k}: {v}" for k, v in session_stm.items()]
        stm_block = "-- STM --\n" + "\n".join(bullets) + "\n-- END STM --\n\n"

    # determine injections (fast retrieval)
    if memory_mode == "off":
        inject_memories = []
    elif memory_mode == "manual":
        if re.search(r'\b(watch memory|use memory|remember for this|use my memory)\b', user_input, flags=re.I):
            inject_memories = retrieve_relevant_memories(user_input, owner, max_tokens_budget=MAX_INJECT_TOKENS)
        else:
            inject_memories = []
    elif memory_mode == "watch":
        inject_memories = retrieve_relevant_memories(user_input, owner, max_tokens_budget=MAX_INJECT_TOKENS)
    else:
        inject_memories = retrieve_relevant_memories(user_input, owner, max_tokens_budget=MAX_INJECT_TOKENS)

    injection = build_memory_injection_block(inject_memories)
    recent_chat_block = get_recent_chat_block(session_id) if session_id else ""

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

    # Call reasoning core (phase3) - let it run; we don't block long here
    try:
        phase3_result = phase3_ask(final_prompt, persona=persona, mode=mode, temperature=temperature, max_tokens=max_tokens, stream=stream, timeout=timeout)
        fallback = False
    except Exception as e:
        try:
            # fallback try shorter input
            phase3_result = phase3_ask(user_input, persona=persona, mode=mode, temperature=temperature, max_tokens=max_tokens, stream=stream, timeout=timeout)
            fallback = True
        except Exception as e2:
            return {"answer": "ZULTX error: reasoning core failed.", "explain": [], "memory_actions": [], "meta": {"latency_ms": int((time.time()-start)*1000), "fallback": True, "error": str(e2)}}

    answer_text = phase3_result if isinstance(phase3_result, str) else (phase3_result.get("answer") if isinstance(phase3_result, dict) else str(phase3_result))

    # append assistant answer to buffer (persist)
    if session_id:
        add_to_conversation_buffer(session_id, "assistant", answer_text, owner)

    # mark used memories (update last_used)
    used_ids = []
    for m in inject_memories:
        try:
            update_memory_last_used(m["id"])
            used_ids.append(m["id"])
        except Exception:
            pass

    # extract and store candidate memories (non-blocking writes)
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

        # set expiry
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

        # write in background to avoid blocking request
        try:
            t = threading.Thread(target=_insert_memory_bg_safe, args=(mem_obj,), daemon=True)
            t.start()
            memory_actions.append({"id": mem_id, "action": "queued_bg", "type": target, "owner": mem_owner, "summary": mem_obj["content"][:140]})
        except Exception as e:
            # synchronous fallback
            try:
                insert_memory(mem_obj)
                memory_actions.append({"id": mem_id, "action": "created", "type": target, "owner": mem_owner, "summary": mem_obj["content"][:140]})
            except Exception as ee:
                enqueue_retry({"op": "insert_memory", "candidate": mem_obj})
                memory_actions.append({"id": None, "action": "queued", "detail": str(ee)})
                audit("write_failed", None, {"error": str(ee)})

    # cleanup expired memories async (non-blocking)
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

def _insert_memory_bg_safe(mem_obj):
    try:
        insert_memory(mem_obj)
    except Exception as e:
        enqueue_retry({"op": "insert_memory", "candidate": mem_obj})
        audit("write_failed_bg", None, {"error": str(e)})

# ---------------------------
# Cleanup expired memories
# ---------------------------
def cleanup_expired_memories():
    with _db_lock:
        conn = get_db_conn()
        cur = conn.cursor()
        try:
            if PG_AVAILABLE and DB_URL:
                cur.execute("DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at < %s", (datetime.utcnow(),))
                deleted = cur.rowcount
            else:
                now = now_ts()
                cur.execute("DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at < ?", (now,))
                deleted = cur.rowcount
            conn.commit()
            if deleted:
                audit("cleanup_expired", None, {"deleted": deleted})
                try:
                    _schedule_recall_rebuild()
                except Exception:
                    pass
        except Exception as e:
            audit("cleanup_failed", None, {"error": str(e)})
        finally:
            try:
                if PG_AVAILABLE and DB_URL:
                    _pg_putconn(conn)
                else:
                    conn.close()
            except Exception:
                pass

# ---------------------------
# CLI tester (for local debug)
# ---------------------------
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
                # conv <session_id>
                sid = ui.split(maxsplit=1)[1].strip()
                print(get_recent_chat_block(sid))
                continue
            if ui.lower().startswith("forget "):
                target = ui.split(" ", 1)[1].strip()
                mem = get_memory(target)
                if mem:
                    with _db_lock:
                        conn = get_db_conn()
                        cur = conn.cursor()
                        if PG_AVAILABLE and DB_URL:
                            cur.execute("DELETE FROM memories WHERE id = %s", (target,))
                        else:
                            cur.execute("DELETE FROM memories WHERE id = ?", (target,))
                        conn.commit()
                        if PG_AVAILABLE and DB_URL:
                            _pg_putconn(conn)
                        else:
                            conn.close()
                    print("Deleted memory", target)
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
