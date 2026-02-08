# phase_3.py
"""
ZULTX Phase 3 — RAG WRAPPER (single-file, drop-in)
- Loads a small document store from ./rag_docs/
- Builds an index (sentence-transformers / openai / tfidf fallback)
- Retrieves top-k chunks for each query
- Injects context and calls phase_2.ask(final_prompt)
- Graceful fallback if no index available
- Usage: from phase_3 import ask
"""

import os
import sys
import glob
import time
import json
import math
import pickle
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

# optional imports (guarded)
try:
    import numpy as np
except Exception:
    np = None

# prefer sentence-transformers if present
SENT_TRANSFORMERS_AVAIL = False
try:
    from sentence_transformers import SentenceTransformer
    SENT_TRANSFORMERS_AVAIL = True
except Exception:
    SENT_TRANSFORMERS_AVAIL = False

# openai embeddings optional
OPENAI_AVAIL = False
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
if OPENAI_API_KEY:
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
        OPENAI_AVAIL = True
    except Exception:
        OPENAI_AVAIL = False

# sklearn tfidf fallback
SKLEARN_AVAIL = False
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAIL = True
except Exception:
    SKLEARN_AVAIL = False

# Import the underlying brain (phase_2) — this should exist
try:
    from phase_2 import ask as base_phase_2_ask
except Exception as e:
    # if phase_2 is missing, keep a fallback to avoid crashing import
    base_phase_2_ask = None
    print("[phase_3] WARNING: phase_2.ask not importable:", e, file=sys.stderr)

# ---------------------------
# CONFIG
# ---------------------------
RAG_DIR = Path(os.getenv("RAG_DOCS_DIR", "rag_docs"))
INDEX_CACHE = Path(os.getenv("RAG_INDEX_CACHE", ".rag_index"))
INDEX_CACHE.mkdir(exist_ok=True)
EMBED_MODEL_NAME = os.getenv("RAG_EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2")
RAG_K = int(os.getenv("RAG_K", "4"))                    # top-k docs/chunks to retrieve
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1200"))   # chunk doc into sizes ~chars
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "4000"))  # limit injected context

# ---------------------------
# UTILS
# ---------------------------
def sha1_of_files(files: List[Path]) -> str:
    h = hashlib.sha1()
    for p in sorted(files, key=lambda x: x.as_posix()):
        h.update(p.name.encode())
        h.update(str(p.stat().st_mtime).encode())
        h.update(str(p.stat().st_size).encode())
    return h.hexdigest()

def read_doc_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return path.read_text(encoding="latin-1")

# ---------------------------
# DOCUMENT LOADING & CHUNKING
# ---------------------------
def load_documents(doc_dir: Path) -> List[Dict[str, Any]]:
    docs = []
    if not doc_dir.exists():
        print(f"[phase_3] rag_docs dir missing: {doc_dir}. Create and add .txt docs.", file=sys.stderr)
        return docs
    patterns = ["**/*.txt", "**/*.md"]
    files = []
    for p in patterns:
        files.extend(doc_dir.glob(p))
    for p in sorted(files):
        txt = read_doc_text(p)
        if not txt.strip():
            continue
        # chunk the doc into overlapping pieces for better retrieval
        start = 0
        L = len(txt)
        while start < L:
            end = min(L, start + CHUNK_SIZE)
            chunk_text = txt[start:end].strip()
            meta = {
                "source": p.name,
                "path": str(p),
                "start": start,
                "end": end,
            }
            docs.append({"text": chunk_text, "meta": meta})
            if end == L:
                break
            start = end - CHUNK_OVERLAP
    return docs

# ---------------------------
# EMBEDDING / INDEX BACKENDS
# ---------------------------
class RagIndex:
    """
    Minimal RAG index wrapper supporting:
    - SentenceTransformer embeddings (faiss optional)
    - OpenAI embeddings (requests)
    - TF-IDF fallback (cosine)
    """
    def __init__(self, docs: List[Dict[str, Any]]):
        self.docs = docs
        self.embeddings = None
        self.backend = None
        self.vectorizer = None

    def build(self):
        # choose backend
        if SENT_TRANSFORMERS_AVAIL:
            self.backend = "sbert"
            print("[phase_3] using sentence-transformers:", EMBED_MODEL_NAME)
            model = SentenceTransformer(EMBED_MODEL_NAME)
            texts = [d["text"] for d in self.docs]
            self.embeddings = np.array(model.encode(texts, show_progress_bar=False, convert_to_numpy=True))
            # normalize
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms[norms==0] = 1.0
            self.embeddings = self.embeddings / norms
        elif OPENAI_AVAIL:
            self.backend = "openai"
            print("[phase_3] using OpenAI embeddings (text-embedding-3-small)")
            texts = [d["text"] for d in self.docs]
            embs = []
            # chunk in small batches
            BATCH = 16
            for i in range(0, len(texts), BATCH):
                batch = texts[i:i+BATCH]
                resp = openai.Embedding.create(input=batch, model="text-embedding-3-small")
                for r in resp["data"]:
                    embs.append(r["embedding"])
            self.embeddings = np.array(embs)
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms[norms==0] = 1.0
            self.embeddings = self.embeddings / norms
        elif SKLEARN_AVAIL:
            self.backend = "tfidf"
            print("[phase_3] using TF-IDF fallback (sklearn)")
            texts = [d["text"] for d in self.docs]
            self.vectorizer = TfidfVectorizer(max_features=50_000)
            X = self.vectorizer.fit_transform(texts)
            # keep dense matrix for quick cosine with query
            self.embeddings = X  # sparse matrix
        else:
            self.backend = "substring"
            print("[phase_3] WARNING: No embedding support found — using substring match fallback.")
            # nothing to precompute

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        meta = {"backend": self.backend}
        with open(path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)
        with open(path / "docs.pkl", "wb") as f:
            pickle.dump(self.docs, f)
        if self.backend == "sbert" or self.backend == "openai":
            if np is None:
                raise RuntimeError("numpy needed for sbert/openai embeddings")
            np.save(path / "embeddings.npy", self.embeddings)
        elif self.backend == "tfidf":
            with open(path / "vectorizer.pkl", "wb") as f:
                pickle.dump(self.vectorizer, f)

    @classmethod
    def load(cls, path: Path):
        if not path.exists():
            raise FileNotFoundError("index path missing")
        with open(path / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        with open(path / "docs.pkl", "rb") as f:
            docs = pickle.load(f)
        idx = cls(docs)
        idx.backend = meta.get("backend")
        if idx.backend in ("sbert", "openai"):
            idx.embeddings = np.load(path / "embeddings.npy")
        elif idx.backend == "tfidf":
            with open(path / "vectorizer.pkl", "rb") as f:
                idx.vectorizer = pickle.load(f)
            # we can reconstruct embeddings lazily: transform on query
        return idx

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        """
        Returns list of (doc, score) ordered desc
        """
        if not self.docs:
            return []
        if self.backend == "sbert":
            model = SentenceTransformer(EMBED_MODEL_NAME)
            q_emb = model.encode([query], convert_to_numpy=True)[0]
            if np is None:
                return []
            q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-9)
            scores = (self.embeddings @ q_emb).astype(float)
            idxs = np.argsort(scores)[::-1][:k]
            return [(self.docs[int(i)], float(scores[int(i)])) for i in idxs]
        elif self.backend == "openai":
            resp = openai.Embedding.create(input=[query], model="text-embedding-3-small")
            q_emb = np.array(resp["data"][0]["embedding"])
            q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-9)
            scores = (self.embeddings @ q_emb).astype(float)
            idxs = np.argsort(scores)[::-1][:k]
            return [(self.docs[int(i)], float(scores[int(i)])) for i in idxs]
        elif self.backend == "tfidf":
            q_v = self.vectorizer.transform([query])
            sims = cosine_similarity(q_v, self.vectorizer.transform([d["text"] for d in self.docs]))[0]
            idxs = list(reversed(sims.argsort()))[:k]
            return [(self.docs[int(i)], float(sims[int(i)])) for i in idxs]
        elif self.backend == "substring":
            # very naive substring ranking
            q = query.lower()
            scored = []
            for d in self.docs:
                t = d["text"].lower()
                score = 0.0
                if q in t:
                    score += 1.0
                # count token overlaps
                for w in q.split()[:10]:
                    if w and w in t:
                        score += 0.01
                scored.append(score)
            idxs = sorted(range(len(scored)), key=lambda i: scored[i], reverse=True)[:k]
            return [(self.docs[i], float(scored[i])) for i in idxs]
        else:
            return []

# ---------------------------
# INDEX BUILD / LOAD
# ---------------------------
def prepare_or_load_index() -> Optional[RagIndex]:
    # load docs
    docs = load_documents(RAG_DIR)
    if not docs:
        print("[phase_3] No docs found in", RAG_DIR)
        return None

    files_hash = sha1_of_files(sorted(RAG_DIR.glob("**/*")))
    cache_folder = INDEX_CACHE / files_hash
    # if cache exists, load
    try:
        if cache_folder.exists():
            idx = RagIndex.load(cache_folder)
            print("[phase_3] Loaded cached RAG index from", cache_folder)
            return idx
    except Exception as e:
        print("[phase_3] failed to load cache:", e)

    # build new index
    idx = RagIndex(docs)
    idx.build()
    try:
        idx.save(cache_folder)
        print("[phase_3] Saved RAG index to", cache_folder)
    except Exception as e:
        print("[phase_3] failed to save cache:", e)
    return idx

# keep a global index instance (lazy)
_GLOBAL_INDEX = None
def get_global_index():
    global _GLOBAL_INDEX
    if _GLOBAL_INDEX is None:
        _GLOBAL_INDEX = prepare_or_load_index()
    return _GLOBAL_INDEX

# ---------------------------
# CONTEXT COMPOSER
# ---------------------------
def build_context_block(retrieved: List[Tuple[Dict[str,Any], float]], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    if not retrieved:
        return ""
    parts = []
    total = 0
    for doc, score in retrieved:
        s = doc["text"].strip()
        meta = doc.get("meta", {})
        src = meta.get("source") or meta.get("path") or "unknown"
        header = f"[source: {src} | score: {score:.3f}]\n"
        piece = header + s + "\n\n"
        if total + len(piece) > max_chars:
            # truncate remaining
            allowed = max_chars - total
            if allowed <= 0:
                break
            piece = piece[:allowed] + "…"
            parts.append(piece)
            total += len(piece)
            break
        parts.append(piece)
        total += len(piece)
    ctx = ("\n-- Retrieved context (for grounding) --\n\n" + "\n".join(parts) + "\n-- End retrieved context --\n\n")
    return ctx

# ---------------------------
# PUBLIC ENTRYPOINT
# ---------------------------
def ask(user_input: str,
        *,
        persona: Optional[str] = None,
        mode: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        rag_k: Optional[int] = None,
        extra_instructions: Optional[List[str]] = None,
        timeout: int = 30,
        **_ignore) -> str:
    """
    The RAG wrapper you import. Behavior:
    - Retrieve top-k docs
    - Build context block
    - Compose final prompt: context + user_input (+ optional extra_instructions)
    - Call phase_2.ask(final_prompt, stream=stream)
    - Return result string (or generator if stream)
    """
    if base_phase_2_ask is None:
        # emergency fallback: local friendly fallback
        return "ZULTX core missing. Try again later."

    idx = get_global_index()
    k = rag_k or RAG_K

    # Retrieve
    retrieved = []
    if idx is not None:
        try:
            retrieved = idx.retrieve(user_input, k=k)
        except Exception as e:
            print("[phase_3] retrieval failed:", e)

    if retrieved:
        context = build_context_block(retrieved, max_chars=MAX_CONTEXT_CHARS)
    else:
        context = ""

    # assemble final prompt: prefer putting adapters intact and use final user prompt injection
    # We'll create an "extra rules" block so phase_2 adapters that respect extra rules can see it.
    # But phase_2.ask's signature may not accept 'extra_rules' — so we embed context into the user_input.
    injected = ""
    if extra_instructions:
        injected += "\n".join(extra_instructions) + "\n\n"

    if context:
        injected += context + "\n"

    final_user_input = injected + user_input

    # If there's no context and we want to be explicit, we do NOT pretend we have facts
    # we just call phase_2.ask directly.
    try:
        result = base_phase_2_ask(final_user_input,
                                  persona=persona,
                                  mode=mode,
                                  temperature=temperature,
                                  max_tokens=max_tokens,
                                  stream=stream,
                                  timeout=timeout)
        return result
    except Exception as e:
        # last-resort graceful fallback to non-RAG call (original user input)
        print("[phase_3] phase_2.ask failed, falling back:", e)
        try:
            return base_phase_2_ask(user_input, persona=persona, mode=mode, stream=stream, timeout=timeout)
        except Exception as e2:
            print("[phase_3] emergency fallback failed too:", e2)
            return "ZULTX error: temporary failure."
