# phase_1.py
"""
ZULTX Phase 1 — IMMORTAL BRAIN ORCHESTRA (final)
- Multi-adapter router (OpenRouter models + Mistral direct + fallbacks)
- Intent + complexity detection (fast / normal / heavy / multimodal / embed)
- Streaming and non-stream returns (generators)
- TokenBucket soft-rate-limits per-adapter
- Health checks, retries, exponential backoff
- Simple metrics & hooks for observability (no external libs)
- Env-driven config (RAILWAY-friendly). Only dependency: requests
"""
from __future__ import annotations
import os
import time
import json
import math
import threading
import traceback
from typing import Generator, Optional, List, Union, Callable, Dict
import requests

# --------------------
# Config (env-friendly)
# --------------------
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
MISTRAL_KEY = os.getenv("MISTRAL_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")  # optional fallback for some use-cases

DEFAULT_TIMEOUT = int(os.getenv("PHASE1_DEFAULT_TIMEOUT", "30"))
MAX_ATTEMPTS = int(os.getenv("PHASE1_MAX_ATTEMPTS", "3"))
BACKOFF_BASE = float(os.getenv("PHASE1_BACKOFF_BASE", "0.35"))
RATE_PER_SEC = float(os.getenv("PHASE1_RATE_PER_SEC", "1.0"))
RATE_CAPACITY = float(os.getenv("PHASE1_RATE_CAPACITY", "5.0"))

# Observability hooks (callable): optional functions you can inject at runtime
# e.g. set Phase1.metrics_hook = lambda event, meta: print(...)
metrics_hook: Optional[Callable[[str, Dict], None]] = None
health_check_hook: Optional[Callable[[], Dict[str, bool]]] = None

def _emit_metric(event: str, meta: Dict = None):
    try:
        if metrics_hook:
            metrics_hook(event, meta or {})
        else:
            # lightweight default log (avoid noisy logs in production)
            print(f"[phase1][metric] {event} {json.dumps(meta or {}, default=str)}")
    except Exception:
        pass

# --------------------
# Exceptions
# --------------------
class ModelFailure(Exception):
    pass

# --------------------
# Utilities
# --------------------
def safe_json_load(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None

def now_s():
    return time.time()

# --------------------
# Intent & Complexity Detection
# --------------------
def detect_intent(prompt: str) -> str:
    txt = (prompt or "").strip().lower()
    if not txt:
        return "small"
    if any(w in txt for w in ["embed", "embedding", "vectorize", "vector"]):
        return "embed"
    if any(w in txt for w in ["image", "photo", "describe image", "generate an image", "img:", "vision"]):
        return "multimodal"
    # heavy keywords
    heavy = ["design", "architecture", "implement", "optimize", "debug", "proof", "step by step", "analysis"]
    score = 0
    for t in heavy:
        if t in txt:
            score += 1
    if len(txt) > 800 or score >= 1:
        return "reason"
    if len(txt) > 250:
        return "long"
    return "small"

def detect_complexity(prompt: str) -> str:
    intent = detect_intent(prompt)
    if intent == "reason":
        return "heavy"
    if intent == "long":
        return "normal"
    if intent in ("multimodal", "embed"):
        return intent
    return "fast"

# --------------------
# Simple Token Bucket (per-adapter)
# --------------------
class TokenBucket:
    def __init__(self, rate_per_sec: float = RATE_PER_SEC, capacity: float = RATE_CAPACITY):
        self.rate = float(rate_per_sec)
        self.capacity = float(capacity)
        self._tokens = float(capacity)
        self._last = now_s()
        self._lock = threading.Lock()

    def consume(self, amount: float = 1.0) -> bool:
        with self._lock:
            now = now_s()
            delta = now - self._last
            self._last = now
            self._tokens = min(self.capacity, self._tokens + delta * self.rate)
            if self._tokens >= amount:
                self._tokens -= amount
                return True
            return False

# --------------------
# Adapter Base
# --------------------
class ModelAdapter:
    name = "adapter"
    supports_stream = False

    def __init__(self):
        self.bucket = TokenBucket()
        self.last_health = 0.0
        self._healthy = True

    def check_ready(self) -> bool:
        return True

    def health(self) -> bool:
        # lightweight health marker; adapters can override with real checks
        return self._healthy

    def mark_unhealthy(self):
        self._healthy = False
        self.last_health = now_s()

    def generate(self, prompt: str, stream: bool = False, timeout: int = DEFAULT_TIMEOUT) -> Union[str, Generator[str, None, None]]:
        raise NotImplementedError

# --------------------
# OpenRouter Adapter (generic)
# --------------------
class OpenRouterAdapter(ModelAdapter):
    name = "openrouter"
    supports_stream = True

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        super().__init__()
        self.model = model_name
        self.key = api_key or OPENROUTER_KEY
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"

    def check_ready(self):
        return bool(self.key and self.model)

    def generate(self, prompt: str, stream: bool = False, timeout: int = DEFAULT_TIMEOUT):
        if not self.check_ready():
            raise ModelFailure("openrouter-missing-key-or-model")
        if not self.bucket.consume():
            raise ModelFailure("rate_limited")

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": bool(stream)
        }
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        try:
            resp = requests.post(self.endpoint, json=payload, headers=headers, timeout=timeout, stream=bool(stream))
            resp.raise_for_status()
            if not stream:
                j = resp.json()
                # defensive: try common locations
                content = None
                try:
                    content = j.get("choices", [{}])[0].get("message", {}).get("content")
                except Exception:
                    pass
                if not content:
                    content = j.get("output", {}).get("text") or j.get("choices", [{}])[0].get("text")
                return content or ""
            # streaming generator
            def gen():
                for chunk in resp.iter_lines(decode_unicode=True):
                    if not chunk:
                        continue
                    s = chunk.strip()
                    # openrouter streaming often uses data: lines
                    if s.startswith("data: "):
                        s = s[len("data: "):]
                    if s == "[DONE]":
                        break
                    try:
                        obj = json.loads(s)
                        delta = obj.get("choices", [{}])[0].get("delta", {}).get("content")
                        if delta:
                            yield delta
                    except Exception:
                        # emit raw chunk as fallback
                        yield s
                return
            return gen()
        except Exception as e:
            self.mark_unhealthy()
            raise ModelFailure(f"openrouter-error: {e}")

# --------------------
# Mistral Direct Adapter
# --------------------
class MistralAdapter(ModelAdapter):
    name = "mistral-direct"
    supports_stream = True

    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self.key = api_key or MISTRAL_KEY
        self.endpoint = "https://api.mistral.ai/v1/chat/completions"

    def check_ready(self):
        return bool(self.key)

    def generate(self, prompt: str, stream: bool = False, timeout: int = DEFAULT_TIMEOUT):
        if not self.check_ready():
            raise ModelFailure("mistral-key-missing")
        if not self.bucket.consume():
            raise ModelFailure("rate_limited")

        payload = {"model": "mistral-large-latest", "messages": [{"role": "user", "content": prompt}], "stream": bool(stream)}
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        try:
            resp = requests.post(self.endpoint, json=payload, headers=headers, timeout=timeout, stream=bool(stream))
            resp.raise_for_status()
            if not stream:
                j = resp.json()
                # Mistral returns choices[0].message.content in new API
                return j.get("choices", [{}])[0].get("message", {}).get("content") or j.get("output", {}).get("text", "") or ""
            def gen():
                for chunk in resp.iter_lines(decode_unicode=True):
                    if not chunk:
                        continue
                    # Mistral streaming chunks may be chunks of JSON or plain text
                    line = chunk.strip()
                    try:
                        j = json.loads(line)
                        # try common delta path
                        delta = j.get("choices", [{}])[0].get("delta", {}).get("content")
                        if delta:
                            yield delta
                        else:
                            # sometimes raw 'content' field
                            c = j.get("choices", [{}])[0].get("message", {}).get("content")
                            if c:
                                yield c
                    except Exception:
                        yield line
                return
            return gen()
        except Exception as e:
            self.mark_unhealthy()
            raise ModelFailure(f"mistral-error: {e}")

# --------------------
# Convenience Adapters (model name wrappers)
# --------------------
class TrinityOpenRouter(OpenRouterAdapter):
    def __init__(self, key=None):
        super().__init__("arcee-ai/trinity-large-preview:free", api_key=key)
        self.name = "trinity-preview"

class StepFlashOpenRouter(OpenRouterAdapter):
    def __init__(self, key=None):
        super().__init__("stepfun/step-3.5-flash:free", api_key=key)
        self.name = "step-3.5-flash"

# --------------------
# Router / Orchestration
# --------------------
class ModelRouter:
    def __init__(self, adapters: List[ModelAdapter]):
        self.adapters = adapters

    def _candidates_for_intent(self, intent: str, complexity: str) -> List[ModelAdapter]:
        # Prioritized lists tuned for zones
        if intent == "embed":
            # embeddings would use dedicated embedding adapters (phase_3 handles embeddings), fallback to OpenRouter/OpenAI
            names = ("openai","openrouter","mistral-direct")
            return [a for a in self.adapters if any(n in a.name for n in names)]
        if intent == "multimodal":
            # prefer OpenRouter trinity (if supports image) then mistral fallback
            ordered = ["trinity-preview","mistral-direct","openrouter","step-3.5-flash"]
            return [a for name in ordered for a in self.adapters if name in a.name]
        # complexity priority
        if complexity == "heavy":
            ordered = ["mistral-direct","trinity-preview","openrouter","step-3.5-flash"]
        elif complexity == "normal":
            ordered = ["trinity-preview","openrouter","step-3.5-flash","mistral-direct"]
        else:  # fast / small
            ordered = ["step-3.5-flash","openrouter","trinity-preview","mistral-direct"]
        # produce adapters in that order (unique)
        out = []
        for n in ordered:
            for a in self.adapters:
                if n in a.name and a not in out:
                    out.append(a)
        # append any other adapters not included
        for a in self.adapters:
            if a not in out:
                out.append(a)
        return out

    def ask(self, prompt: str, stream: bool = False, timeout: int = DEFAULT_TIMEOUT) -> Union[str, Generator[str, None, None]]:
        complexity = detect_complexity(prompt)
        intent = detect_intent(prompt)
        _emit_metric("route_selected", {"intent": intent, "complexity": complexity, "time": now_s()})

        candidates = self._candidates_for_intent(intent, complexity)

        last_err = None
        for adapter in candidates:
            try:
                if not adapter.check_ready():
                    _emit_metric("adapter_skipped_not_ready", {"adapter": adapter.name})
                    continue
                # perform attempts with backoff
                for attempt in range(1, MAX_ATTEMPTS + 1):
                    try:
                        _emit_metric("adapter_attempt", {"adapter": adapter.name, "attempt": attempt})
                        out = adapter.generate(prompt, stream=stream, timeout=timeout)
                        _emit_metric("adapter_success", {"adapter": adapter.name, "attempt": attempt})
                        return out
                    except ModelFailure as mf:
                        last_err = mf
                        # if rate_limited — move to next adapter quickly
                        if "rate_limited" in str(mf).lower():
                            _emit_metric("adapter_rate_limited", {"adapter": adapter.name})
                            break
                        # transient: backoff and retry same adapter
                        backoff = BACKOFF_BASE * (2 ** (attempt - 1))
                        time.sleep(backoff)
                        continue
            except Exception as e:
                last_err = e
                _emit_metric("adapter_unexpected_error", {"adapter": getattr(adapter, "name", "unknown"), "err": str(e)})
                traceback.print_exc()
                # try next adapter
                continue

        # if all failed, return graceful fallback (string or generator)
        msg = "ZULTX brain temporarily unavailable. Try again in a moment."
        _emit_metric("router_all_failed", {"err": str(last_err)})
        if stream:
            def g():
                for ch in msg:
                    yield ch
                    time.sleep(0.003)
            return g()
        return msg

# --------------------
# Bootstrap default router with sensible adapters
# --------------------
def build_default_router() -> ModelRouter:
    adapters: List[ModelAdapter] = []
    # Prioritized: local/free fast -> normal -> heavy direct
    # Step / flash fast model (OpenRouter)
    adapters.append(StepFlashOpenRouter(key=OPENROUTER_KEY))
    adapters.append(TrinityOpenRouter(key=OPENROUTER_KEY))
    # Generic OpenRouter wrapper as fallback (if you want other models)
    adapters.append(OpenRouterAdapter(model_name="openai/gpt-4o-mini", api_key=OPENROUTER_KEY))
    # Mistral direct (heavy reasoning)
    adapters.append(MistralAdapter(api_key=MISTRAL_KEY))
    # Optionally add an OpenAI adapter if key exists (useful as last-resort)
    if OPENAI_KEY:
        class _OpenAIAdapter(ModelAdapter):
            name = "openai"
            supports_stream = True
            def __init__(self, key):
                super().__init__()
                self.key = key
                self.endpoint = "https://api.openai.com/v1/chat/completions"
            def check_ready(self): return bool(self.key)
            def generate(self, prompt, stream=False, timeout=DEFAULT_TIMEOUT):
                if not self.check_ready(): raise ModelFailure("openai-missing")
                if not self.bucket.consume(): raise ModelFailure("rate_limited")
                payload = {"model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"), "messages":[{"role":"user","content":prompt}], "stream": bool(stream)}
                headers = {"Authorization": f"Bearer {self.key}", "Content-Type":"application/json"}
                r = requests.post(self.endpoint, json=payload, headers=headers, timeout=timeout, stream=bool(stream))
                r.raise_for_status()
                if not stream:
                    return r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                def gen():
                    for line in r.iter_lines(decode_unicode=True):
                        if not line: continue
                        s = line.strip()
                        if s.startswith("data: "): s = s[len("data: "):]
                        if s == "[DONE]": break
                        try:
                            obj = json.loads(s)
                            delta = obj.get("choices",[{}])[0].get("delta", {}).get("content")
                            if delta: yield delta
                        except Exception:
                            yield s
                return gen()
        adapters.append(_OpenAIAdapter(OPENAI_KEY))
    return ModelRouter(adapters)

# global singleton
_router = build_default_router()

# --------------------
# Public API: ask()
# --------------------
def ask(prompt: str, *, stream: bool = False, timeout: int = DEFAULT_TIMEOUT) -> Union[str, Generator[str, None, None]]:
    """
    Main entrypoint for ZultX core.
    - prompt: user text (string)
    - stream: if True returns a generator that yields chunks
    - timeout: per-adapter HTTP timeout (seconds)
    """
    return _router.ask(prompt, stream=stream, timeout=timeout)

# --------------------
# Helpers: status & health
# --------------------
def get_adapters_status() -> List[Dict]:
    out = []
    for a in _router.adapters:
        out.append({
            "name": getattr(a, "name", "unknown"),
            "ready": a.check_ready(),
            "healthy": a.health(),
            "last_health_ts": getattr(a, "last_health", None)
        })
    return out

def set_metrics_hook(fn: Optional[Callable[[str, Dict], None]]):
    global metrics_hook
    metrics_hook = fn

def set_health_hook(fn: Optional[Callable[[], Dict[str, bool]]]):
    global health_check_hook
    health_check_hook = fn
    
# --------------------
# << APPEND: Multimodal glue + helpers + safe router rebuild >>
# Paste this at the end of phase_1.py
# --------------------
import base64
from io import BytesIO
from typing import Any

# --- Safe define adapters only if missing (so re-running cell is safe) ---
if "ImageGenAdapter" not in globals():
    class ImageGenAdapter(ModelAdapter):
        name = "imagegen"
        supports_stream = False

        def __init__(self, provider: str = None, api_key: Optional[str] = None, model: Optional[str] = None):
            super().__init__()
            self.provider = (provider or os.getenv("IMAGE_PROVIDER", "openai")).lower()
            self.key = api_key
            self.model = model
            self._init_provider_settings()

        def _init_provider_settings(self):
            prov = self.provider
            if prov == "stability":
                self.key = self.key or os.getenv("STABILITY_KEY")
                self.endpoint = "https://api.stability.ai/v1/generation"
                self.engine = os.getenv("STABILITY_ENGINE", "stable-diffusion-v1-5")
            elif prov == "openrouter":
                self.key = self.key or os.getenv("OPENROUTER_API_KEY")
                self.endpoint = os.getenv("OPENROUTER_IMAGE_ENDPOINT", "https://openrouter.ai/api/v1/images/generate")
                self.model = self.model or os.getenv("OPENROUTER_IMAGE_MODEL", "stability/stable-diffusion-v1")
            else:  # openai default
                self.key = self.key or os.getenv("OPENAI_KEY")
                self.endpoint = "https://api.openai.com/v1/images/generations"
                self.model = self.model or os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")

        def check_ready(self):
            return bool(self.key)

        def generate(self, prompt: str, stream: bool = False, timeout: int = DEFAULT_TIMEOUT):
            if not self.check_ready():
                raise ModelFailure("imagegen-missing-key")
            if not self.bucket.consume():
                raise ModelFailure("rate_limited")
            prov = self.provider
            try:
                if prov == "stability":
                    url = f"{self.endpoint}/{self.engine}/text-to-image"
                    headers = {"Authorization": f"Bearer {self.key}", "Content-Type":"application/json"}
                    payload = {"text_prompts":[{"text": prompt}], "width":512, "height":512}
                    r = requests.post(url, json=payload, headers=headers, timeout=timeout)
                    r.raise_for_status()
                    j = r.json()
                    b64 = j.get("artifacts",[{}])[0].get("base64")
                    if b64:
                        return base64.b64decode(b64)
                    return str(j)
                elif prov == "openrouter":
                    headers = {"Authorization": f"Bearer {self.key}", "Content-Type":"application/json"}
                    payload = {"model": self.model, "prompt": prompt}
                    r = requests.post(self.endpoint, json=payload, headers=headers, timeout=timeout)
                    r.raise_for_status()
                    j = r.json()
                    b64 = (j.get("output",{}) or {}).get("images", [{}])[0].get("b64") or (j.get("data",[{}])[0].get("b64"))
                    if b64:
                        return base64.b64decode(b64)
                    return str(j)
                else:  # openai
                    headers = {"Authorization": f"Bearer {self.key}", "Content-Type":"application/json"}
                    payload = {"model": self.model, "prompt": prompt, "size":"1024x1024"}
                    r = requests.post(self.endpoint, json=payload, headers=headers, timeout=timeout)
                    r.raise_for_status()
                    j = r.json()
                    # try a few fields
                    b64 = j.get("data",[{}])[0].get("b64_json") or j.get("data",[{}])[0].get("b64")
                    if b64:
                        return base64.b64decode(b64)
                    url = j.get("data",[{}])[0].get("url")
                    if url:
                        return requests.get(url, timeout=timeout).content
                    return str(j)
            except Exception as e:
                self.mark_unhealthy()
                raise ModelFailure(f"imagegen-error: {e}")

if "WhisperASRAdapter" not in globals():
    class WhisperASRAdapter(ModelAdapter):
        name = "whisper-asr"
        supports_stream = False

        def __init__(self, api_key: Optional[str] = None):
            super().__init__()
            self.key = api_key or os.getenv("OPENAI_KEY")
            self.endpoint = "https://api.openai.com/v1/audio/transcriptions"

        def check_ready(self):
            return bool(self.key)

        def generate(self, audio_bytes: bytes, stream: bool = False, timeout: int = DEFAULT_TIMEOUT):
            if not self.check_ready():
                raise ModelFailure("whisper-missing-key")
            if not self.bucket.consume():
                raise ModelFailure("rate_limited")
            try:
                # OpenAI expects form-data file
                files = {"file": ("audio.wav", audio_bytes)}
                data = {"model": os.getenv("WHISPER_MODEL","whisper-1")}
                headers = {"Authorization": f"Bearer {self.key}"}
                r = requests.post(self.endpoint, headers=headers, files=files, data=data, timeout=timeout)
                r.raise_for_status()
                return r.json().get("text","")
            except Exception as e:
                self.mark_unhealthy()
                raise ModelFailure(f"whisper-error: {e}")

if "ElevenLabsTTSAdapter" not in globals():
    class ElevenLabsTTSAdapter(ModelAdapter):
        name = "elevenlabs-tts"
        supports_stream = False

        def __init__(self, api_key: Optional[str] = None, voice: Optional[str] = None):
            super().__init__()
            self.key = api_key or os.getenv("ELEVENLABS_KEY")
            self.voice = voice or os.getenv("ELEVENLABS_VOICE", "alloy")
            self.endpoint_base = "https://api.elevenlabs.io/v1"

        def check_ready(self):
            return bool(self.key)

        def generate(self, text: str, stream: bool = False, timeout: int = DEFAULT_TIMEOUT):
            if not self.check_ready():
                raise ModelFailure("elevenlabs-key-missing")
            if not self.bucket.consume():
                raise ModelFailure("rate_limited")
            url = f"{self.endpoint_base}/text-to-speech/{self.voice}"
            headers = {"xi-api-key": self.key, "Content-Type":"application/json"}
            payload = {"text": text, "voice": self.voice, "model":"eleven_monolingual_v1"}
            try:
                r = requests.post(url, json=payload, headers=headers, timeout=timeout)
                r.raise_for_status()
                return r.content
            except Exception as e:
                self.mark_unhealthy()
                raise ModelFailure(f"elevenlabs-tts-error: {e}")

# --- Helper: speak(text) -> bytes (TTS) ---
def speak(text: str, timeout: int = DEFAULT_TIMEOUT) -> bytes:
    """Return audio bytes from first available TTS adapter (ElevenLabs)."""
    for a in (_router.adapters if "_router" in globals() else []):
        if getattr(a, "name", "") == "elevenlabs-tts" and a.check_ready():
            return a.generate(text, stream=False, timeout=timeout)
    raise ModelFailure("no-tts-available")

# --- Safe rebuild of default router (fixes constructor arg mismatch and wires multimodal adapters) ---
def rebuild_router_with_multimodal():
    """Call this to rebuild global _router with multimodal adapters included.
       Safe to call multiple times."""
    adapters: List[ModelAdapter] = []
    # Step/flash fast model (OpenRouter) — pass key via named param 'key' (convenience adapters expect key)
    try:
        adapters.append(StepFlashOpenRouter(key=OPENROUTER_KEY))
    except Exception:
        # fallback instantiate by positional if previous signature differs
        try: adapters.append(StepFlashOpenRouter(OPENROUTER_KEY))
        except Exception: pass

    try:
        adapters.append(TrinityOpenRouter(key=OPENROUTER_KEY))
    except Exception:
        try: adapters.append(TrinityOpenRouter(OPENROUTER_KEY))
        except Exception: pass

    adapters.append(OpenRouterAdapter(model_name="openai/gpt-4o-mini", api_key=OPENROUTER_KEY))
    adapters.append(MistralAdapter(api_key=MISTRAL_KEY))
    # multimodal
    adapters.append(ImageGenAdapter())
    adapters.append(WhisperASRAdapter())
    adapters.append(ElevenLabsTTSAdapter())
    # optional OpenAI fallback
    if OPENAI_KEY:
        class _OpenAIAdapter(ModelAdapter):
            name = "openai"
            supports_stream = True
            def __init__(self, key):
                super().__init__()
                self.key = key
                self.endpoint = "https://api.openai.com/v1/chat/completions"
            def check_ready(self): return bool(self.key)
            def generate(self, prompt, stream=False, timeout=DEFAULT_TIMEOUT):
                if not self.check_ready(): raise ModelFailure("openai-missing")
                if not self.bucket.consume(): raise ModelFailure("rate_limited")
                payload = {"model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"), "messages":[{"role":"user","content":prompt}], "stream": bool(stream)}
                headers = {"Authorization": f"Bearer {self.key}", "Content-Type":"application/json"}
                r = requests.post(self.endpoint, json=payload, headers=headers, timeout=timeout, stream=bool(stream))
                r.raise_for_status()
                if not stream:
                    return r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                def gen():
                    for line in r.iter_lines(decode_unicode=True):
                        if not line: continue
                        s = line.strip()
                        if s.startswith("data: "): s = s[len("data: "):]
                        if s == "[DONE]": break
                        try:
                            obj = json.loads(s)
                            delta = obj.get("choices",[{}])[0].get("delta", {}).get("content")
                            if delta: yield delta
                        except Exception:
                            yield s
                return gen()
        adapters.append(_OpenAIAdapter(OPENAI_KEY))

    # replace global router
    global _router
    _router = ModelRouter(adapters)
    _emit_metric("router_rebuilt", {"adapters": [getattr(a,"name",str(a)) for a in adapters]})
    return _router

# --- Patch ModelRouter.ask to auto-route audio bytes to Whisper (safe monkeypatch) ---
if getattr(ModelRouter, "_mm_patched", False) is False:
    def _patched_ask(self, prompt: Union[str, bytes], stream: bool = False, timeout: int = DEFAULT_TIMEOUT):
        # If raw bytes are provided -> treat as audio and send to whisper-asr adapter
        if isinstance(prompt, (bytes, bytearray)):
            for a in self.adapters:
                if getattr(a, "name", "") == "whisper-asr" and a.check_ready():
                    _emit_metric("route_audio_to_whisper", {"time": now_s()})
                    return a.generate(bytes(prompt), stream=False, timeout=timeout)
            # if no whisper available -> graceful message
            raise ModelFailure("no-asr-available")
        # otherwise, normal text flow (reuse earlier router algorithm but minimal)
        complexity = detect_complexity(prompt if isinstance(prompt, str) else str(prompt))
        intent = detect_intent(prompt if isinstance(prompt, str) else str(prompt))
        _emit_metric("route_selected", {"intent": intent, "complexity": complexity, "time": now_s()})
        candidates = self._candidates_for_intent(intent, complexity)
        last_err = None
        for adapter in candidates:
            try:
                if not adapter.check_ready():
                    _emit_metric("adapter_skipped_not_ready", {"adapter": adapter.name})
                    continue
                for attempt in range(1, MAX_ATTEMPTS + 1):
                    try:
                        _emit_metric("adapter_attempt", {"adapter": adapter.name, "attempt": attempt})
                        out = adapter.generate(prompt, stream=stream, timeout=timeout)
                        _emit_metric("adapter_success", {"adapter": adapter.name, "attempt": attempt})
                        return out
                    except ModelFailure as mf:
                        last_err = mf
                        if "rate_limited" in str(mf).lower():
                            _emit_metric("adapter_rate_limited", {"adapter": adapter.name})
                            break
                        backoff = BACKOFF_BASE * (2 ** (attempt - 1))
                        time.sleep(backoff)
                        continue
            except Exception as e:
                last_err = e
                _emit_metric("adapter_unexpected_error", {"adapter": getattr(adapter, "name", "unknown"), "err": str(e)})
                traceback.print_exc()
                continue
        msg = "ZULTX brain temporarily unavailable. Try again in a moment."
        _emit_metric("router_all_failed", {"err": str(last_err)})
        if stream:
            def g():
                for ch in msg:
                    yield ch
                    time.sleep(0.003)
            return g()
        return msg

    ModelRouter.ask = _patched_ask
    ModelRouter._mm_patched = True

# --- Ensure router is rebuilt with multimodal adapters available ---
if "_router" not in globals() or not isinstance(_router, ModelRouter):
    rebuild_router_with_multimodal()
else:
    # safe rebuild to include multimodal adapters and fix signatures if current router doesn't have them
    try:
        rebuild_router_with_multimodal()
    except Exception:
        pass
