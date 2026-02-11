# phase_4.py
"""
ZULTX Phase 4 — Final god-tier memory orchestration (single-file)
Features:
 - phase4_ask(user_input, session_id, user_id, memory_mode, ...) calls phase_3.ask(...) for reasoning
 - Per-user memory isolation (owner/user_id)
 - Conversation buffer (recent 7 messages per session)
 - STM (short expiry), CM (longer expiry), LTM (very long), EM (ephemeral with expiry)
 - Safe defaults, promotion rules, audit logs, retry queue
 - TF-IDF fallback recall (sklearn optional) or substring fallback
 - Conservative extraction + "save permanent" detection
 - Designed for Railway / local use
"""
import os
import re
import json
import time
import sqlite3
import uuid
import math
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

# Optional dependencies (guarded)
SKLEARN_AVAIL = False
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAIL = True
except Exception:
    SKLEARN_AVAIL = False

# Import phase_3 (reasoning/RAG wrapper)
try:
    from phase_3 import ask as phase3_ask
except Exception as e:
    phase3_ask = None
    print("[phase_4] WARNING: phase_3.ask not importable:", e)

# CONFIG (tweakable via env)
DB_PATH = os.getenv("ZULTX_MEMORY_DB", "zultx_memory.db")
MAX_INJECT_TOKENS = int(os.getenv("ZULTX_MAX_INJECT_TOKENS", "1200"))
TFIDF_TOP_K = int(os.getenv("ZULTX_TFIDF_K", "12"))

# Promotion thresholds (conservative)
PROMOTE_TO_CM_SCORE = float(os.getenv("ZULTX_PROMOTE_CM", "0.45"))      # need both score+confidence to promote
PROMOTE_TO_LTM_SCORE = float(os.getenv("ZULTX_PROMOTE_LTM", "0.80"))
CONFIDENCE_PROMOTE_THRESHOLD = float(os.getenv("ZULTX_CONF_PROMOTE", "0.70"))

# Expiry defaults
STM_EXPIRE_DAYS = int(os.getenv("ZULTX_STM_DAYS", "1"))       # short-term: ~1 day
CM_EXPIRE_DAYS = int(os.getenv("ZULTX_CM_DAYS", "365"))      # chat memory: 1 year
LTM_EXPIRE_DAYS = None                                        # LTM: no expiry by default
EM_DEFAULT_DAYS = int(os.getenv("ZULTX_EM_DAYS", "7"))       # ephemeral default: 7 days

# Basic policy: regex patterns for sensitive items we should not store raw
SENSITIVE_PATTERNS = [
    re.compile(r"\b\d{10}\b"),  # simple phone number
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN-like
    re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),  # email
    re.compile(r"\b(?:card(?:-|\s)?num|credit card|visa|mastercard)\b", re.I),
]

# ---------------------------
# SQLite memory store helpers (owner column added)
# ---------------------------
_INIT_SQL = f"""
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    owner TEXT,                 -- owner / user_id (nullable -> global)
    type TEXT NOT NULL,         -- STM|CM|LTM|EM
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

def get_db_conn(path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn

_db_lock = threading.Lock()
def initialize_db():
    with _db_lock:
        conn = get_db_conn()
        cur = conn.cursor()
        cur.executescript(_INIT_SQL)
        conn.commit()
        conn.close()

initialize_db()

# ---------------------------
# Utilities
# ---------------------------
def now_ts() -> str:
    return datetime.utcnow().isoformat()

def parse_ts(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
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
# Memory score computation
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

# ---------------------------
# Simple policy engine
# ---------------------------
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
# TF-IDF semantic recall helper (fallback)
# ---------------------------
class SimpleRecall:
    def __init__(self):
        self.vectorizer = None
        self.corpus = []  # list of (mem_id, owner, content)
        self.matrix = None

    def build_from_db(self):
        with get_db_conn() as conn:
            cur = conn.cursor()
            # Load all CM+LTM into the in-memory index (we'll filter by owner when retrieving)
            cur.execute("SELECT id, owner, content FROM memories WHERE type IN ('CM','LTM')")
            rows = cur.fetchall()
            self.corpus = [(r["id"], r["owner"], r["content"]) for r in rows]
            texts = [t for (_id, _owner, t) in self.corpus]
            if SKLEARN_AVAIL and texts:
                try:
                    self.vectorizer = TfidfVectorizer(max_features=50000)
                    self.matrix = self.vectorizer.fit_transform(texts)
                except Exception:
                    self.vectorizer = None
                    self.matrix = None
            else:
                self.vectorizer = None
                self.matrix = None

    def retrieve(self, query: str, k: int = TFIDF_TOP_K, owner: Optional[str] = None) -> List[Tuple[str, str, float]]:
        if not self.corpus:
            return []
        # Filter corpus indices by owner: owner or global (owner is NULL)
        filtered = []
        for mem_id, mem_owner, content in self.corpus:
            if owner is None or mem_owner is None or mem_owner == owner:
                filtered.append((mem_id, content))
        if not filtered:
            return []
        ids, texts = zip(*filtered)
        if self.vectorizer is not None and self.matrix is not None:
            try:
                qv = self.vectorizer.transform([query])
                # build reduced matrix for filtered entries (fallback: re-vectorize filtered)
                # For practicality, compute similarities by vectorizing filtered texts
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

# ---------------------------
# DB operations + audit
# ---------------------------
def insert_memory(mem: Dict[str, Any]) -> str:
    with _db_lock:
        conn = get_db_conn()
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
            1 if mem.get("consent") else 0,
            json.dumps(mem.get("metadata") or {})
        ))
        conn.commit()
        conn.close()
    audit("create_memory", mem.get("id"), {"owner": mem.get("owner"), "summary": mem.get("content")[:140]})
    # asynchronous recall index refresh
    try:
        threading.Thread(target=refresh_recall_index, daemon=True).start()
    except Exception:
        pass
    return mem.get("id")

def update_memory_last_used(mem_id: str):
    with _db_lock:
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute("UPDATE memories SET last_used = ?, frequency = frequency + 1 WHERE id = ?", (now_ts(), mem_id))
        conn.commit()
        conn.close()
    audit("update_last_used", mem_id, {"ts": now_ts()})

def delete_owner_name_memories(owner: Optional[str]):
    # Delete any existing 'name:' memory for that owner (prevents name collisions)
    with _db_lock:
        conn = get_db_conn()
        cur = conn.cursor()
        if owner is None:
            cur.execute("DELETE FROM memories WHERE content LIKE 'name:%' AND owner IS NULL")
        else:
            cur.execute("DELETE FROM memories WHERE content LIKE 'name:%' AND owner = ?", (owner,))
        conn.commit()
        conn.close()
    audit("delete_name_memory", None, {"owner": owner})

def get_memory(mem_id: str) -> Optional[Dict[str, Any]]:
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM memories WHERE id = ?", (mem_id,))
        row = cur.fetchone()
        if not row:
            return None
        r = dict(row)
        r["tags"] = json.loads(r["tags"]) if r["tags"] else []
        r["metadata"] = json.loads(r["metadata"]) if r["metadata"] else {}
        return r

def list_memories(limit: int = 50, owner: Optional[str] = None) -> List[Dict[str, Any]]:
    with get_db_conn() as conn:
        cur = conn.cursor()
        if owner is None:
            cur.execute("SELECT * FROM memories ORDER BY memory_score DESC, last_used DESC LIMIT ?", (limit,))
        else:
            cur.execute("SELECT * FROM memories WHERE owner = ? ORDER BY memory_score DESC, last_used DESC LIMIT ?", (owner, limit))
        rows = cur.fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["tags"] = json.loads(d["tags"]) if d["tags"] else []
            d["metadata"] = json.loads(d["metadata"]) if d["metadata"] else {}
            out.append(d)
        return out

def audit(action: str, mem_id: Optional[str], payload: Dict[str, Any]):
    with _db_lock:
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute("INSERT INTO audit_log (id, ts, action, mem_id, payload) VALUES (?,?,?,?,?)", (
            uuid4(), now_ts(), action, mem_id, json.dumps(payload)
        ))
        conn.commit()
        conn.close()

def enqueue_retry(item: Dict[str, Any]):
    with _db_lock:
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute("INSERT OR REPLACE INTO queue_pending (id, ts, payload) VALUES (?,?,?)", (uuid4(), now_ts(), json.dumps(item)))
        conn.commit()
        conn.close()
    audit("queue_enqueue", None, {"item": str(item)[:200]})

# ---------------------------
# Extractor (rules + LLM-assist stub)
# ---------------------------
def rule_based_extract(user_input: str, assistant_text: str) -> List[Dict[str, Any]]:
    candidates = []
    text = (user_input or "") + "\n\n" + (assistant_text or "")

    # explicit: "remember: X" (CM by default)
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

    # explicit permanent save: "remember permanently" / "save permanent"
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

    # preferences: "I prefer X"
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

    # name detection (careful): "my name is X" or "I am X" (capitalize detection a bit)
    m = re.search(r"\b(?:my name is|i am)\s+([A-Z][a-zA-Z]{0,1,50}|[A-Za-z0-9 _-]{1,50})", text)
    if m:
        name = m.group(1).strip()
        candidates.append({
            "content": f"name:{name}",
            "suggested_type": "CM",
            "importance": 0.85,
            "confidence": 0.85,
            "reason": "self_identify"
        })

    return candidates

def llm_assisted_extract_stub(user_input: str, assistant_text: str) -> List[Dict[str, Any]]:
    # Railway-safe stub. Integrate external LLM here when you want (careful with privacy).
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
# Promotion decision (STM->CM->LTM)
# ---------------------------
def decide_storage_for_candidate(candidate: Dict[str, Any]) -> str:
    suggested = candidate.get("suggested_type", "CM")
    importance = clamp(float(candidate.get("importance", 0.5)))
    confidence = clamp(float(candidate.get("confidence", 0.5)))
    mem_score = compute_memory_score(importance, 1, now_ts(), confidence)
    candidate["memory_score"] = mem_score

    # time-sensitive detection (EM)
    if re.search(r"\btoday\b|\btomorrow\b|\bon\b\s+\w+\s+\d{1,2}\b", candidate["content"], flags=re.I):
        return "EM"

    # explicit LTM suggestion
    if suggested == "LTM" and mem_score >= PROMOTE_TO_LTM_SCORE and confidence >= 0.8:
        return "LTM"

    # conservative promotion to CM: require both score and confidence
    if mem_score >= PROMOTE_TO_CM_SCORE and confidence >= CONFIDENCE_PROMOTE_THRESHOLD:
        return "CM"

    # fallback to STM
    return "STM"

# ---------------------------
# Conversation buffer (in-memory) — recent messages per session
# ---------------------------
_CONV_BUFFERS: Dict[str, List[Dict[str, str]]] = {}
_CONV_LOCK = threading.Lock()
CONV_BUFFER_LIMIT = 7

def add_to_conversation_buffer(session_id: str, role: str, content: str):
    if not session_id:
        return
    with _CONV_LOCK:
        buf = _CONV_BUFFERS.get(session_id) or []
        buf.append({"role": role, "content": content, "ts": now_ts()})
        # keep only last N
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
# Memory retrieval (owner-aware) and injection
# ---------------------------
def retrieve_relevant_memories(user_input: str, owner: Optional[str], max_tokens_budget: int = MAX_INJECT_TOKENS) -> List[Dict[str, Any]]:
    refresh_recall_index()
    # Get candidate mem_ids from recall (owner-aware)
    recs = _recall_instance.retrieve(user_input, k=TFIDF_TOP_K, owner=owner)
    out = []
    token_budget = max_tokens_budget
    for mem_id, content, sim in recs:
        mem = get_memory(mem_id)
        if not mem:
            continue
        # owner filter: allow only mem.owner == owner or mem.owner is NULL (global)
        mem_owner = mem.get("owner")
        if mem_owner is not None and owner is not None and mem_owner != owner:
            continue
        # discard expired EM
        if mem["type"] == "EM" and mem.get("expires_at"):
            exp = parse_ts(mem.get("expires_at"))
            if exp and exp < datetime.utcnow():
                continue
        # filter low score
        if (mem.get("memory_score") or 0.0) < 0.20:
            continue
        # Skip injecting name memories unless query asks for identity
        if mem["content"].startswith("name:"):
            if not re.search(r'\b(name|who am i|what is my name|my name)\b', user_input.lower()):
                continue
        snippet = mem.get("content", "")
        approx_tokens = max(1, int(len(snippet) / 6))  # compact estimate
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
        # cap injected memories (safety)
        if len(out) >= 6:
            break
    return out

def build_memory_injection_block(memories: List[Dict[str, Any]]) -> str:
    if not memories:
        return ""
    parts = ["-- INJECTED MEMORIES (distilled) --"]
    for m in memories:
        # compact bullet: do not dump raw text; keep distilled content short
        content = m['content']
        # show only short distilled snippet (first 120 chars)
        short = (content[:120] + "…") if len(content) > 120 else content
        parts.append(f"- [{m['id'][:8]}] ({m['type']}) {short}")
    parts.append("-- END INJECTED MEMORIES --\n")
    return "\n".join(parts)

# ---------------------------
# phase4_ask orchestration
# ---------------------------
def phase4_ask(user_input: str,
               session_id: Optional[str] = None,
               user_id: Optional[str] = None,
               *,
               memory_mode: str = "auto",   # "auto" | "manual" | "off" | "watch"
               persona: Optional[str] = None,
               mode: Optional[str] = None,
               temperature: Optional[float] = None,
               max_tokens: Optional[int] = None,
               stream: bool = False,
               timeout: int = 30,
               **_kwargs) -> Dict[str, Any]:
    """
    Returns dict:
      { answer, explain, memory_actions, meta }
    user_id: per-user owner id (string). If None -> owner is session_id if available.
    memory_mode controls when memory is injected:
       - "auto": inject relevant memories automatically
       - "manual": inject only when explicit trigger words or 'watch' phrase
       - "off": never inject memories
       - "watch": force injection (like manual watch)
    """
    start = time.time()
    owner = user_id if user_id is not None else session_id
    if phase3_ask is None:
        return {"answer": "[phase_3 missing] Core unavailable.", "explain": [], "memory_actions": [], "meta": {"latency_ms": int((time.time()-start)*1000), "fallback": True}}

    # sanitize memory_mode
    memory_mode = (memory_mode or "auto").lower()
    # 0) Add user query to conversation buffer (user message)
    if session_id:
        add_to_conversation_buffer(session_id, "user", user_input)

    # Build STM block from session_stm (we support storing a small transient STM passed via _kwargs)
    session_stm = _kwargs.get("session_stm") or {}
    stm_block = ""
    if session_stm:
        bullets = [f"{k}: {v}" for k, v in session_stm.items()]
        stm_block = "-- STM --\n" + "\n".join(bullets) + "\n-- END STM --\n\n"

    # 1) Decide whether to retrieve/inject memories
    inject_memories = []
    if memory_mode == "off":
        inject_memories = []
    elif memory_mode == "manual":
        # only if explicit 'watch' keywords or "watch memory" or "use memory" phrase
        if re.search(r'\b(watch memory|use memory|remember for this|use my memory)\b', user_input, flags=re.I):
            inject_memories = retrieve_relevant_memories(user_input, owner, max_tokens_budget=MAX_INJECT_TOKENS)
    elif memory_mode == "watch":
        inject_memories = retrieve_relevant_memories(user_input, owner, max_tokens_budget=MAX_INJECT_TOKENS)
    else:  # auto
        inject_memories = retrieve_relevant_memories(user_input, owner, max_tokens_budget=MAX_INJECT_TOKENS)

    injection = build_memory_injection_block(inject_memories)
    recent_chat_block = get_recent_chat_block(session_id) if session_id else ""

    # 2) Compose final prompt
    prompt_parts = []
    if persona:
        prompt_parts.append(f"[Persona: {persona}]")
    # include recent chat first (so phase_3 receives context)
    if recent_chat_block:
        prompt_parts.append(recent_chat_block)
    if stm_block:
        prompt_parts.append(stm_block)
    if injection:
        prompt_parts.append(injection)
    prompt_parts.append("[System: Use user preferences and memory to answer. Do not invent personal data.]")
    prompt_parts.append("\nUser: " + user_input)
    final_prompt = "\n\n".join([p for p in prompt_parts if p])

    # 3) Call phase_3.ask (reasoning wrapper)
    try:
        phase3_result = phase3_ask(final_prompt, persona=persona, mode=mode, temperature=temperature, max_tokens=max_tokens, stream=stream, timeout=timeout)
        fallback = False
    except Exception as e:
        # fallback to raw call
        try:
            phase3_result = phase3_ask(user_input, persona=persona, mode=mode, temperature=temperature, max_tokens=max_tokens, stream=stream, timeout=timeout)
            fallback = True
        except Exception as e2:
            return {"answer": "ZULTX error: reasoning core failed.", "explain": [], "memory_actions": [], "meta": {"latency_ms": int((time.time()-start)*1000), "fallback": True, "error": str(e2)}}
   
    # normalize phase3_result
    answer_text = phase3_result if isinstance(phase3_result, str) else phase3_result.get("answer") or str(phase3_result)

    # append assistant answer to convo buffer
    if session_id:
        add_to_conversation_buffer(session_id, "assistant", answer_text)

    # 4) Mark used memories last_used/frequency for injected ones
    used_ids = []
    for m in inject_memories:
        try:
            update_memory_last_used(m["id"])
            used_ids.append(m["id"])
        except Exception:
            pass

    # 5) Extract candidate memories from (user_input, answer_text)
    candidates = extract_candidates(user_input, answer_text)

    # 6) Score, decide storage, set expiries, insert (owner-aware)
    memory_actions = []
    for cand in candidates:
        content = cand["content"]
        allowed, block_reason = policy_allow_store(content)
        if not allowed:
            # try anonymize
            content2 = anonymize_if_needed(content)
            if content2 != content:
                content = content2
                cand["confidence"] = min(0.6, cand.get("confidence", 0.6))
            else:
                audit("blocked_candidate", None, {"content": content[:200], "reason": block_reason})
                continue

        # Decide storage type
        target = decide_storage_for_candidate({**cand})
        mem_id = uuid4()
        created_at = now_ts()

        # Set owner (per-user). If owner is None, it's global (rare). Default to owner variable above
        mem_owner = owner

        # Name uniqueness: if this is name:... ensure we delete previous one for this owner
        if content.startswith("name:"):
            try:
                delete_owner_name_memories(mem_owner)
            except Exception:
                pass

        # compute memory_score if not present
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

        # set expiries depending on type
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

        # attempt insertion transactionally
        try:
            insert_memory(mem_obj)
            memory_actions.append({"id": mem_id, "action": "created", "type": target, "owner": mem_owner, "summary": mem_obj["content"][:140]})
        except Exception as e:
            enqueue_retry({"op": "insert_memory", "candidate": mem_obj})
            memory_actions.append({"id": None, "action": "queued", "detail": str(e)})
            audit("write_failed", None, {"error": str(e)})

    # 7) Clean expired memories (best-effort background)
    try:
        threading.Thread(target=cleanup_expired_memories, daemon=True).start()
    except Exception:
        pass

    # 8) Build explain block for UI
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
        now = now_ts()
        try:
            cur.execute("DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at < ?", (now,))
            deleted = cur.rowcount
            conn.commit()
            if deleted:
                audit("cleanup_expired", None, {"deleted": deleted})
        except Exception as e:
            audit("cleanup_failed", None, {"error": str(e)})
        finally:
            conn.close()

# ---------------------------
# CLI for testing
# ---------------------------
if __name__ == "__main__":
    print("ZULTX phase_4 final tester")
    refresh_recall_index()
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
                    print(f"{r['id'][:8]} owner={r['owner'][:8] if r['owner'] else 'GLOBAL'} {r['type']} score={r['memory_score']:.3f} freq={r['frequency']} content={r['content'][:80]}")
                continue
            if ui.lower().startswith("forget "):
                target = ui.split(" ", 1)[1].strip()
                mem = get_memory(target)
                if mem:
                    with _db_lock:
                        c = get_db_conn().cursor()
                        c.execute("DELETE FROM memories WHERE id = ?", (target,))
                        get_db_conn().commit()
                    print("Deleted memory", target)
                    continue
                print("No memory by that id; use listmem to inspect")
                continue

            # default: simulate session "cli_session" and user "cli_user"
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
