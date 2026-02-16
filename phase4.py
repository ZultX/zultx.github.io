# phase_4.py
"""
ZULTX Phase 4 — Polished memory orchestration (Postgres + Pinecone)
- Per-user memory isolation (owner/user_id)
- Single conversation buffer (recent N messages)
- STM | CM | LTM | EM with expiries
- TF-IDF fallback (sklearn optional) or substring fallback
- Conservative extraction + explicit "remember permanently"
- Audit logs, retry queue, safe defaults
- PostgreSQL persistence (primary). Optional SQLite fallback (dev).
- Optional Pinecone vector storage (CM/LTM) using OpenAI embeddings if configured.
"""
import os
import re
import json
import time
import uuid
import math
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

# Optional dependencies
SKLEARN_AVAIL = False
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAIL = True
except Exception:
    SKLEARN_AVAIL = False

# DB adapters
USE_POSTGRES = bool(os.getenv("DATABASE_URL"))
DB_URL = os.getenv("DATABASE_URL")

# try postgres driver
PG_AVAILABLE = False
if USE_POSTGRES:
    try:
        import psycopg2
        import psycopg2.extras
        PG_AVAILABLE = True
    except Exception:
        PG_AVAILABLE = False
        print("[phase_4] WARNING: psycopg2 not available; will try SQLite fallback.")

# SQLite fallback if postgres not available
USE_SQLITE_FALLBACK = not PG_AVAILABLE
SQLITE_PATH = os.getenv("ZULTX_MEMORY_DB", "zultx_memory.db")

# Pinecone (optional)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
USE_PINECONE = bool(PINECONE_API_KEY and PINECONE_INDEX)
PINECONE_CLIENT = None
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
        USE_PINECONE = False

# Embedding backend (OpenAI optional)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USE_OPENAI_EMBED = False
if OPENAI_API_KEY:
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
        USE_OPENAI_EMBED = True
    except Exception as e:
        USE_OPENAI_EMBED = False
        print("[phase_4] openai import failed:", e)

# phase_3 importer (reasoning/RAG wrapper)
try:
    from phase_3 import ask as phase3_ask
except Exception as e:
    phase3_ask = None
    print("[phase_4] WARNING: phase_3.ask not importable:", e)
    
from psycopg2.pool import SimpleConnectionPool

PG_POOL = None
if PG_AVAILABLE and DB_URL:
    PG_POOL = SimpleConnectionPool(
        1, 10,  # min, max connections
        DB_URL
    )

# CONFIG (env overrides)
MAX_INJECT_TOKENS = int(os.getenv("ZULTX_MAX_INJECT_TOKENS", "1200"))
TFIDF_TOP_K = int(os.getenv("ZULTX_TFIDF_K", "12"))
PROMOTE_TO_CM_SCORE = float(os.getenv("ZULTX_PROMOTE_CM", "0.60"))
PROMOTE_TO_LTM_SCORE = float(os.getenv("ZULTX_PROMOTE_LTM", "0.85"))
CONFIDENCE_PROMOTE_THRESHOLD = float(os.getenv("ZULTX_CONF_PROMOTE", "0.80"))
STM_EXPIRE_DAYS = int(os.getenv("ZULTX_STM_DAYS", "1"))
CM_EXPIRE_DAYS = int(os.getenv("ZULTX_CM_DAYS", "365"))
LTM_EXPIRE_DAYS = None
EM_DEFAULT_DAYS = int(os.getenv("ZULTX_EM_DAYS", "7"))

# Sensitive patterns
SENSITIVE_PATTERNS = [
    re.compile(r"\b\d{10}\b"),  # simple phone number
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN-like
    re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    re.compile(r"\b(?:card(?:-|\s)?num|credit card|visa|mastercard)\b", re.I),
]

# Locks
_db_lock = threading.Lock()
_CONV_LOCK = threading.Lock()

# ---------------------------
# DB Schema for Postgres (and compatible for SQLite fallback)
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
"""

# SQLite fallback SQL (kept similar to your previous file for dev)
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
"""

# ---------------------------
# DB helpers
# ---------------------------
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

# ---------------------------
# Postgres connection wrapper (or sqlite fallback)
# ---------------------------
def get_db_conn():
    if PG_POOL:
        return PG_POOL.getconn()
    import sqlite3
    conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_db():
    with _db_lock:
        conn = get_db_conn()
        cur = conn.cursor()
        try:
            if PG_AVAILABLE and DB_URL:
                # execute postgres init
                cur.execute(_POSTGRES_INIT_SQL)
                conn.commit()
            else:
                # sqlite fallback
                cur.executescript(_SQLITE_INIT_SQL)
                conn.commit()
        finally:
            cur.close()
            conn.close()

initialize_db()

# ---------------------------
# Recall index (TF-IDF) - same logic but uses DB rows
# ---------------------------
class SimpleRecall:
    def __init__(self):
        self.vectorizer = None
        self.corpus = []  # list of (mem_id, owner, content)
        self.matrix = None

    def build_from_db(self):
        rows = []
        conn = get_db_conn()
        cur = conn.cursor()
        try:
            if PG_AVAILABLE and DB_URL:
                cur.execute("SELECT id, owner, content FROM memories WHERE type IN ('CM','LTM')")
                rows = cur.fetchall()
                # psycopg2 with RealDictCursor returns dict-like rows
                self.corpus = [(r["id"], r["owner"], r["content"]) for r in rows]
            else:
                rows = cur.execute("SELECT id, owner, content FROM memories WHERE type IN ('CM','LTM')").fetchall()
                self.corpus = [(r["id"], r["owner"], r["content"]) for r in rows]
            texts = [t for (_id, _owner, t) in self.corpus]
            if SKLEARN_AVAIL and texts:
                self.vectorizer = TfidfVectorizer(max_features=50000)
                self.matrix = self.vectorizer.fit_transform(texts)
            else:
                self.vectorizer = None
                self.matrix = None
        finally:
            cur.close()
            if PG_POOL:
               PG_POOL.putconn(conn)
            else:
               conn.close()


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
                # re-vectorize the filtered texts (safer)
                fm = self.vectorizer.transform(list(texts))
                sims = cosine_similarity(qv, fm)[0]
                idxs = sims.argsort()[::-1][:k]
                out = []
                for i in idxs:
                    out.append((ids[int(i)], texts[int(i)], float(sims[int(i)])))
                return out
            except Exception:
                pass
        # substring fallback
        q = query.lower()
        scored = []
        for mem_id, t in zip(ids, texts):
            txt = t.lower()
            score = 0.0
            if q in txt:
                score += 1.0
            for w in q.split()[:10]:
                if w and w in txt:
                    score += 0.01
            scored.append((mem_id, t, score))
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:k]

_recall_instance = SimpleRecall()

def refresh_recall_index():
    try:
        _recall_instance.build_from_db()
    except Exception as e:
        audit("recall_index_failed", None, {"err": str(e)})

# initial build
try:
    refresh_recall_index()
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
    with _db_lock:
        conn = get_db_conn()
        cur = conn.cursor()
        try:
            ts = datetime.utcnow()
            if PG_AVAILABLE and DB_URL:
                cur.execute(
                    "INSERT INTO audit_log (id, ts, action, mem_id, payload) VALUES (%s, %s, %s, %s, %s)",
                    (uuid4(), ts, action, mem_id, psycopg2.extras.Json(payload))
                )
            else:
                cur.execute(
                    "INSERT INTO audit_log (id, ts, action, mem_id, payload) VALUES (?, ?, ?, ?, ?)",
                    (uuid4(), now_ts(), action, mem_id, json.dumps(payload))
                )
            conn.commit()
        except Exception as e:
            # best-effort logging to stdout
            print("[phase_4][audit_error]", e)
        finally:
            cur.close()
            conn.close()

def enqueue_retry(item: Dict[str, Any]):
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
            cur.close()
            conn.close()
    audit("queue_enqueue", None, {"item": str(item)[:200]})

# ---------------------------
# CRUD: insert/update/get/list/delete
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
            conn.rollback()
            raise
        finally:
            cur.close()
            conn.close()

    audit("create_memory", mem.get("id"), {"owner": mem.get("owner"), "summary": mem.get("content")[:140]})
    # async refresh
    try:
        threading.Thread(target=refresh_recall_index, daemon=True).start()
    except Exception:
        pass

    # Upsert vector to Pinecone if configured and memory is CM/LTM
    try:
        if USE_PINECONE and PINECONE_CLIENT and mem.get("type") in ("CM", "LTM"):
            # get vector (use openai embeddings if available)
            text = mem.get("content", "")
            vec = get_embedding_for_text(text)
            if vec is not None:
                _pine_index.upsert([(mem.get("id"), vec, {"owner": mem.get("owner") or None, "type": mem.get("type")})])
    except Exception as e:
        audit("pinecone_upsert_failed", mem.get("id"), {"err": str(e)})

    return mem.get("id")

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
            # row may be dict or sqlite Row
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
            cur.close()
            conn.close()
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
            cur.close()
            conn.close()
    audit("delete_name_memory", None, {"owner": owner})

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
            # ensure types for tags/metadata
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
            # convert timestamps to iso
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
        cur.close()
        conn.close()

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
        cur.close()
        conn.close()

# ---------------------------
# Extractor (same as in your code) - rule-based + LLM stub
# ---------------------------
def rule_based_extract(user_input: str, assistant_text: str) -> List[Dict[str, Any]]:
    candidates = []
    text = (user_input or "") + "\n\n" + (assistant_text or "")

    # explicit: "remember: X"
    for m in re.finditer(r"\bremember(?: that)?\s*[:\-]?\s*(.+?)(?:\.|$|\n)", text, flags=re.I):
        content = m.group(1).strip()
        if content:
            candidates.append({
                "content": content,
                "suggested_type": "CM",
                "importance": 0.9,
                "confidence": 0.9,
                "reason": "explicit_remember"
            })

    # explicit permanent save
    for m in re.finditer(r"\b(?:remember|save)\s+(?:permanent|permanently|forever)\s*[:\-]?\s*(.+?)(?:\.|$|\n)", text, flags=re.I):
        content = m.group(1).strip()
        if content:
            candidates.append({
                "content": content,
                "suggested_type": "LTM",
                "importance": 0.95,
                "confidence": 0.95,
                "reason": "explicit_permanent"
            })

    # preferences
    for m in re.finditer(r"\bI (?:prefer|like|love|hate|want)\s+(.*?)(?:\.|$|\n)", text, flags=re.I):
        content = m.group(1).strip()
        if content and len(content) < 200:
            candidates.append({
                "content": "pref:" + content,
                "suggested_type": "CM",
                "importance": 0.7,
                "confidence": 0.7,
                "reason": "preference_statement"
            })

    # name detection
    m = re.search(r"\b(?:my name is|i am)\s+([A-Za-z][a-zA-Z]{1,30})\b", text, flags=re.I)
    if m:
        name = m.group(1).strip()
        name = name.strip().title()
        candidates.append({
            "content": f"name:{name}",
            "suggested_type": "CM",
            "importance": 0.90,
            "confidence": 0.90,
            "reason": "self_identify"
        })

    return candidates

def llm_assisted_extract_stub(user_input: str, assistant_text: str) -> List[Dict[str, Any]]:
    # Placeholder for future LLM-assisted extraction (not called to external models here)
    return []

def extract_candidates(user_input: str, assistant_text: str) -> List[Dict[str, Any]]:
    cands = rule_based_extract(user_input, assistant_text)
    cands += llm_assisted_extract_stub(user_input, assistant_text)
    out = []
    for c in cands:
        content = c.get("content", "").strip()
        if not content:
            continue
        allowed, reason = policy_allow_store(content)
        if not allowed:
            content2 = anonymize_if_needed(content)
            if content2 != content:
                content = content2
                c["confidence"] = min(0.6, float(c.get("confidence", 0.6)))
            else:
                audit("blocked_candidate", None, {"content": content[:200], "reason": reason})
                continue
        out.append({
            "content": content,
            "suggested_type": c.get("suggested_type", "CM"),
            "importance": float(c.get("importance", 0.5)),
            "confidence": float(c.get("confidence", 0.5)),
            "reason": c.get("reason", "extracted")
        })
    return out

# ---------------------------
# Promotion decision
# ---------------------------
def decide_storage_for_candidate(candidate: Dict[str, Any]) -> str:
    suggested = candidate.get("suggested_type", "CM")
    importance = clamp(float(candidate.get("importance", 0.5)))
    confidence = clamp(float(candidate.get("confidence", 0.5)))
    mem_score = compute_memory_score(importance, 1, now_ts(), confidence)
    candidate["memory_score"] = mem_score

    # time-sensitive detection => EM
    if re.search(r"\btoday\b|\btomorrow\b|\bon\b\s+\w+\s+\d{1,2}\b", candidate["content"], flags=re.I):
        return "EM"

    if suggested == "LTM" and mem_score >= PROMOTE_TO_LTM_SCORE and confidence >= 0.85:
        return "LTM"

    if mem_score >= PROMOTE_TO_CM_SCORE and confidence >= CONFIDENCE_PROMOTE_THRESHOLD:
        return "CM"

    return "STM"

# ---------------------------
# Conversation buffer
# ---------------------------
_CONV_BUFFERS: Dict[str, List[Dict[str, str]]] = {}
CONV_BUFFER_LIMIT = 7

def add_to_conversation_buffer(session_id: str, role: str, content: str):
    if not session_id:
        return
    with _CONV_LOCK:
        buf = _CONV_BUFFERS.get(session_id, [])
        buf.append({"role": role, "content": content})
        if len(buf) > CONV_BUFFER_LIMIT:
            buf = buf[-CONV_BUFFER_LIMIT:]
        _CONV_BUFFERS[session_id] = buf

def get_recent_chat_block(session_id: str) -> str:
    with _CONV_LOCK:
        buf = _CONV_BUFFERS.get(session_id, [])
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
    recs = _recall_instance.retrieve(user_input, k=TFIDF_TOP_K, owner=owner)
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
            if not re.search(r'\b(name|who am i|what is my name|my name)\b', user_input.lower()):
                continue
        snippet = mem.get("content", "")
        approx_tokens = max(1, int(len(snippet) / 6))
        if token_budget - approx_tokens < 0:
            break
        token_budget -= approx_tokens
        out.append({
            "id": mem_id,
            "content": snippet,
            "type": mem.get("type"),
            "memory_score": mem.get("memory_score"),
            "why_matched": f"tfidf_sim={sim:.3f}",
            "owner": mem_owner
        })
        if len(out) >= 6:
            break
    return out

def build_memory_injection_block(memories: List[Dict[str, Any]]) -> str:
    if not memories:
        return ""
    parts = ["-- INJECTED MEMORIES (distilled) --"]
    for m in memories:
        content = m['content']
        short = (content[:120] + "…") if len(content) > 120 else content
        parts.append(f"- {short}")
    parts.append("-- END INJECTED MEMORIES --\n")
    return "\n".join(parts)

# ---------------------------
# Embedding helper (OpenAI)
# ---------------------------
def get_embedding_for_text(text: str):
    if not text or not USE_OPENAI_EMBED:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        model = os.getenv("PHASE4_EMBED_MODEL", "text-embedding-3-small")
        resp = client.embeddings.create(
            model=model,
            input=text
        )
        return resp.data[0].embedding
    except Exception as e:
        audit("embed_failed", None, {"err": str(e)})
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

    # add user message to buffer
    if session_id:
        add_to_conversation_buffer(session_id, "user", user_input)

    # transient STM via kwargs
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
    elif memory_mode == "watch":
        inject_memories = retrieve_relevant_memories(user_input, owner, max_tokens_budget=MAX_INJECT_TOKENS)
    else:
        inject_memories = retrieve_relevant_memories(user_input, owner, max_tokens_budget=MAX_INJECT_TOKENS)

    injection = build_memory_injection_block(inject_memories)
    recent_chat_block = get_recent_chat_block(session_id) if session_id else ""

    # Prepare prompt parts
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
    prompt_parts.append("\nUser: " + user_input)
    final_prompt = "\n\n".join([p for p in prompt_parts if p])

    # Call reasoning core
    try:
        phase3_result = phase3_ask(final_prompt, persona=persona, mode=mode, temperature=temperature, max_tokens=max_tokens, stream=stream, timeout=timeout)
        fallback = False
    except Exception as e:
        try:
            phase3_result = phase3_ask(user_input, persona=persona, mode=mode, temperature=temperature, max_tokens=max_tokens, stream=stream, timeout=timeout)
            fallback = True
        except Exception as e2:
            return {"answer": "ZULTX error: reasoning core failed.", "explain": [], "memory_actions": [], "meta": {"latency_ms": int((time.time()-start)*1000), "fallback": True, "error": str(e2)}}

    answer_text = phase3_result if isinstance(phase3_result, str) else phase3_result.get("answer") or str(phase3_result)

    # append assistant answer to buffer
    if session_id:
        add_to_conversation_buffer(session_id, "assistant", answer_text)

    # mark used memories
    used_ids = []
    for m in inject_memories:
        try:
            update_memory_last_used(m["id"])
            used_ids.append(m["id"])
        except Exception:
            pass

    # extract candidates
    candidates = extract_candidates(user_input, answer_text)

    # decide writes
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
            "raw_snippet": content[:800],
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

        try:
            insert_memory(mem_obj)
            memory_actions.append({"id": mem_id, "action": "created", "type": target, "owner": mem_owner, "summary": mem_obj["content"][:140]})
        except Exception as e:
            enqueue_retry({"op": "insert_memory", "candidate": mem_obj})
            memory_actions.append({"id": None, "action": "queued", "detail": str(e)})
            audit("write_failed", None, {"error": str(e)})

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
                    refresh_recall_index()
                except Exception:
                    pass
        except Exception as e:
            audit("cleanup_failed", None, {"error": str(e)})
        finally:
            cur.close()
            conn.close()

# ---------------------------
# CLI for testing (same behavior)
# ---------------------------
if __name__ == "__main__":
    print("ZULTX phase_4 (Postgres + Pinecone) tester")
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
                        cur.close()
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
