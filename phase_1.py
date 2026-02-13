# phase_1.py
"""
ZULTX Phase 1 — IMMORTAL BRAIN ORCHESTRA (production-ready template)
- Adapter-based model routing & failover
- Intent-aware routing (cheap/small vs heavy/reasoning vs embeddings vs multimodal)
- Streaming support (generators) and non-stream returns
- Backoff + health checks + simple soft-rate-limit
- Env-driven keys (RAILWAY-friendly)
- Add adapters by subclassing ModelAdapter and registering in router
"""
import os
import time
import json
import traceback
import requests
import threading
import math
from typing import Generator, Optional, Dict, Any, List, Callable, Union

# -----------------------
# Config from environment
# -----------------------
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY")   # for Gemini via Google Generative API / Vertex
ANTHROPIC_KEY   = os.getenv("ANTHROPIC_API_KEY")
COHERE_API_KEY  = os.getenv("COHERE_API_KEY")
# etc. Add others like PINECONE_API_KEY, HF_API_KEY, etc.

DEFAULT_TIMEOUT = int(os.getenv("PHASE1_DEFAULT_TIMEOUT", "30"))
MAX_ATTEMPTS = int(os.getenv("PHASE1_MAX_ATTEMPTS", "3"))

# -----------------------
# Base exceptions
# -----------------------
class ModelFailure(Exception):
    pass

# -----------------------
# Intent detection (very small rule-based)
# -----------------------
def detect_intent(prompt: str) -> str:
    """
    Return one of:
      - 'small'      : small replies, greetings, quick chit-chat
      - 'reason'     : heavy reasoning, code, planning, long-form answers
      - 'embed'      : compute embeddings
      - 'multimodal' : image or multimodal prompts (if "image", "photo", "show me", etc.)
    """
    txt = (prompt or "").strip().lower()
    if not txt:
        return "small"
    if any(w in txt for w in ["embedding", "embed", "vectorize", "vector"]):
        return "embed"
    if any(w in txt for w in ["image", "photo", "show me", "generate an image", "describe image", "img:"]):
        return "multimodal"
    # heuristics for heavy tasks
    heavy_tokens = ["explain", "compare", "design", "optimize", "plan", "write", "implement", "debug", "proof"]
    if len(txt) > 300 or sum(1 for t in heavy_tokens if t in txt) >= 1:
        return "reason"
    # default fallback
    return "small"

# -----------------------
# Simple Token Bucket limiter (soft, per-adapter)
# -----------------------
class TokenBucket:
    def __init__(self, rate_per_sec: float, capacity: float):
        self.rate = rate_per_sec
        self.capacity = capacity
        self._tokens = capacity
        self._last = time.time()
        self._lock = threading.Lock()
    def consume(self, amount=1.0) -> bool:
        with self._lock:
            now = time.time()
            delta = now - self._last
            self._last = now
            self._tokens = min(self.capacity, self._tokens + delta * self.rate)
            if self._tokens >= amount:
                self._tokens -= amount
                return True
            return False

# -----------------------
# Abstract Adapter
# -----------------------
class ModelAdapter:
    name: str = "adapter"
    supports_stream: bool = False
    rate_limiter: Optional[TokenBucket] = None

    def __init__(self):
        # default: generous rate
        self.rate_limiter = TokenBucket(rate_per_sec=1.0, capacity=5.0)

    def check_ready(self) -> bool:
        """Optional health-check or quick config check. Return True if adapter is ready."""
        return True

    def generate(self, prompt: str, stream: bool = False, timeout: int = DEFAULT_TIMEOUT) -> Union[str, Generator[str, None, None]]:
        """
        Return string (non-stream) or generator (stream=True).
        Raise ModelFailure on error.
        """
        raise NotImplementedError

    def _consume_slot(self) -> bool:
        if not self.rate_limiter:
            return True
        return self.rate_limiter.consume()

# -----------------------
# Concrete adapters
# (examples; minimal implementations using requests)
# -----------------------
class MistralAdapter(ModelAdapter):
    name = "mistral"
    supports_stream = True

    def __init__(self, api_key: Optional[str]):
        super().__init__()
        self.api_key = api_key
        self.endpoint = "https://api.mistral.ai/v1/chat/completions"

    def check_ready(self):
        return bool(self.api_key)

    def generate(self, prompt, stream=False, timeout=DEFAULT_TIMEOUT):
        if not self.check_ready():
            raise ModelFailure("Mistral API key missing")
        if not self._consume_slot():
            raise ModelFailure("rate_limited")
        payload = {
            "model": "mistral-large-latest",
            "messages": [{"role": "user", "content": prompt}],
            "stream": bool(stream)
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            r = requests.post(self.endpoint, json=payload, headers=headers, timeout=timeout, stream=bool(stream))
            r.raise_for_status()
            if not stream:
                j = r.json()
                # defensive: paths differ by provider
                return j.get("choices", [{}])[0].get("message", {}).get("content", "") or j.get("output", {}).get("text", "")
            # stream generator
            def gen():
                for chunk in r.iter_lines(decode_unicode=True):
                    if chunk:
                        # Mistral stream may be chunked lines; return raw chunk (caller should parse)
                        yield chunk
                # ensure newline at end
            return gen()
        except Exception as e:
            raise ModelFailure(f"Mistral error: {str(e)}")

class OpenAIAdapter(ModelAdapter):
    name = "openai"
    supports_stream = True

    def __init__(self, api_key: Optional[str]):
        super().__init__()
        self.api_key = api_key
        self.endpoint = "https://api.openai.com/v1/chat/completions"

    def check_ready(self):
        return bool(self.api_key)

    def generate(self, prompt, stream=False, timeout=DEFAULT_TIMEOUT):
        if not self.check_ready():
            raise ModelFailure("OpenAI API key missing")
        if not self._consume_slot():
            raise ModelFailure("rate_limited")
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": os.getenv("OPENAI_MODEL","gpt-4o-mini"),
            "messages":[{"role":"user","content":prompt}],
            "temperature": float(os.getenv("OPENAI_TEMPERATURE","0.2")),
            "stream": bool(stream)
        }
        try:
            r = requests.post(self.endpoint, json=payload, headers=headers, timeout=timeout, stream=bool(stream))
            r.raise_for_status()
            if not stream:
                j = r.json()
                return j.get("choices", [{}])[0].get("message", {}).get("content", "")
            def gen():
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    # OpenAI stream yields "data: ..." lines
                    try:
                        s = line.strip()
                        if s.startswith("data: "):
                            s = s[len("data: "):]
                        if s == "[DONE]":
                            break
                        obj = json.loads(s)
                        delta = obj.get("choices",[{}])[0].get("delta", {}).get("content")
                        if delta:
                            yield delta
                    except Exception:
                        yield line
            return gen()
        except Exception as e:
            raise ModelFailure(f"OpenAI error: {str(e)}")

class GoogleGeminiAdapter(ModelAdapter):
    name = "google_gemini"
    supports_stream = False  # Generative API streaming model support varies

    def __init__(self, api_key: Optional[str]):
        super().__init__()
        self.api_key = api_key
        self.endpoint = "https://generativelanguage.googleapis.com/v1beta/models"

    def check_ready(self):
        return bool(self.api_key)

    def generate(self, prompt, stream=False, timeout=DEFAULT_TIMEOUT):
        if not self.check_ready():
            raise ModelFailure("Google API key missing")
        if not self._consume_slot():
            raise ModelFailure("rate_limited")
        # choose model via env
        gmodel = os.getenv("GOOGLE_GEMINI_MODEL","gemini-pro")
        url = f"{self.endpoint}/{gmodel}:generateText?key={self.api_key}"
        payload = {"prompt": {"text": prompt}}
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            j = r.json()
            # output path depends on API; try several common ones
            cand = j.get("candidates") or j.get("output") or []
            if isinstance(cand, list) and len(cand):
                # may be nested
                part = cand[0]
                if isinstance(part, dict):
                    return part.get("content", {}).get("text", "") or part.get("text", "")
                return str(part)
            return j.get("output", {}).get("text", "") or json.dumps(j)
        except Exception as e:
            raise ModelFailure(f"Google Gemini error: {str(e)}")

class AnthropicAdapter(ModelAdapter):
    name = "anthropic"
    supports_stream = False

    def __init__(self, api_key: Optional[str]):
        super().__init__()
        self.api_key = api_key
        self.endpoint = "https://api.anthropic.com/v1/complete"

    def check_ready(self):
        return bool(self.api_key)

    def generate(self, prompt, stream=False, timeout=DEFAULT_TIMEOUT):
        if not self.check_ready():
            raise ModelFailure("Anthropic key missing")
        if not self._consume_slot():
            raise ModelFailure("rate_limited")
        payload = {"model": os.getenv("ANTHROPIC_MODEL","claude-2.1"), "prompt": prompt, "max_tokens": 1000}
        headers = {"x-api-key": self.api_key, "Content-Type":"application/json"}
        try:
            r = requests.post(self.endpoint, json=payload, headers=headers, timeout=timeout)
            r.raise_for_status()
            j = r.json()
            return j.get("completion") or j.get("output") or ""
        except Exception as e:
            raise ModelFailure(f"Anthropic error: {str(e)}")

# Add more adapters (CohereAdapter, HFAdapter, WhisperAdapter, EmbeddingAdapter, etc.)
# For embeddings+vector DB combine Cohere/HF and Pinecone/Milvus adapter layers.

# -----------------------
# Router: select adapters based on intent + health + policy
# -----------------------
class ModelRouter:
    def __init__(self, adapters: List[ModelAdapter]):
        self.adapters = adapters
        # ordering hint: prefer local/high-quality fast models first
        # but router will reorder based on intent & health
    def _sorted_candidates(self, intent: str) -> List[ModelAdapter]:
        # Create prioritized list per intent
        # You can tune the order; this is a sensible default:
        priority = []
        if intent == "embed":
            # embeddings -> dedicated embedding provider (Cohere/HF), else fallback to openai embeddings
            priority = [a for a in self.adapters if a.name in ("cohere","openai","mistral")]
        elif intent == "multimodal":
            priority = [a for a in self.adapters if a.name in ("google_gemini","openai","mistral")]
        elif intent == "reason":
            priority = [a for a in self.adapters if a.name in ("mistral","openai","anthropic","google_gemini")]
        else:  # small chat
            priority = [a for a in self.adapters if a.name in ("openai","mistral","anthropic")]
        # fallback: include all others not in priority at the end
        others = [a for a in self.adapters if a not in priority]
        return priority + others

    def ask(self, prompt: str, stream: bool=False, timeout: int=DEFAULT_TIMEOUT) -> Union[str, Generator[str,None,None]]:
        intent = detect_intent(prompt)
        candidates = self._sorted_candidates(intent)
        errors = []
        for adapter in candidates:
            try:
                if not adapter.check_ready():
                    continue
                # We try attempts with exponential backoff
                attempt = 0
                while attempt < MAX_ATTEMPTS:
                    try:
                        result = adapter.generate(prompt, stream=stream, timeout=timeout)
                        # If adapter returns generator and stream requested, return generator
                        return result
                    except ModelFailure as mf:
                        # soft backoff and try again (unless rate_limited)
                        errstr = str(mf)
                        if "rate_limited" in errstr:
                            # wait a bit longer then move to next adapter quickly
                            time.sleep(0.25 + attempt*0.25)
                            break
                        attempt += 1
                        backoff = min(1 + attempt**2 * 0.25, 5)
                        time.sleep(backoff)
                        continue
            except Exception as e:
                errors.append((adapter.name, str(e)))
                # log and continue
                traceback.print_exc()
                continue
        # If we get here, all adapters failed — return graceful message or stream generator
        fallback = "ZULTX is stabilizing its brain network. Try again in a few seconds."
        if stream:
            def gen():
                for ch in fallback:
                    yield ch
                    time.sleep(0.01)
            return gen()
        return fallback

# -----------------------
# Bootstrap a default router with adapters
# -----------------------
def build_default_router() -> ModelRouter:
    adapters: List[ModelAdapter] = []
    # register adapters with environment keys
    adapters.append(MistralAdapter(MISTRAL_API_KEY))
    adapters.append(OpenAIAdapter(OPENAI_API_KEY))
    adapters.append(GoogleGeminiAdapter(GOOGLE_API_KEY))
    adapters.append(AnthropicAdapter(ANTHROPIC_KEY))
    # Add more: CohereAdapter(COHERE_API_KEY), HFAdapter(HF_KEY), etc.
    # The order here is non-final — router will prioritize per intent
    return ModelRouter(adapters)

# global singleton router
_router = build_default_router()

# -----------------------
# Public ask() used by rest of ZultX
# -----------------------
def ask(prompt: str, *, stream: bool=False, timeout: int=DEFAULT_TIMEOUT):
    """
    Single entrypoint for all layers.
    - prompt: user text
    - stream: if True, return generator yielding text chunks
    - timeout: seconds to wait per-adapter call
    """
    return _router.ask(prompt, stream=stream, timeout=timeout)
