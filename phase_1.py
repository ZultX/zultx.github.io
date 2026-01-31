# phase_1.py
"""
ZULTX Phase 1 — IMMORTAL BRAIN INTERFACE (PRODUCTION)

This file is NEVER rewritten.
Only extended internally if absolutely required.

Responsibilities:
- Base model orchestration (failover)
- Unified ask() interface
- Streaming-safe
- Adapter-safe
- Future-proof (RAG, memory, multimodal)

Railway-ready | Env-based secrets
"""

from typing import Generator, Optional
import os
import time
import traceback
import requests


# -----------------------------
# ENV LOADING (RAILWAY SAFE)
# -----------------------------

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not MISTRAL_API_KEY and not GEMINI_API_KEY:
    print("[ZULTX WARNING] No model API keys found in environment.")


# -----------------------------
# Base Exceptions
# -----------------------------

class ModelFailure(Exception):
    pass


# -----------------------------
# Base Model Interface
# -----------------------------

class BaseModel:
    name: str = "base"

    def generate(
        self,
        prompt: str,
        stream: bool = False,
        timeout: int = 30
    ) -> Generator[str, None, None] | str:
        raise NotImplementedError


# -----------------------------
# MISTRAL MODELS
# -----------------------------

class MistralLarge(BaseModel):
    name = "mistral-large"
    endpoint = "https://api.mistral.ai/v1/chat/completions"

    def generate(self, prompt, stream=False, timeout=30):
        if not MISTRAL_API_KEY:
            raise ModelFailure("Mistral API key missing")

        payload = {
            "model": "mistral-large-latest",
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
        }

        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json",
        }

        try:
            r = requests.post(
                self.endpoint,
                json=payload,
                headers=headers,
                timeout=timeout,
                stream=stream,
            )
            r.raise_for_status()

            if not stream:
                return r.json()["choices"][0]["message"]["content"]

            def gen():
                for line in r.iter_lines():
                    if line:
                        yield line.decode(errors="ignore")

            return gen()

        except Exception as e:
            raise ModelFailure(str(e))


class MistralSmall(MistralLarge):
    name = "mistral-small"

    def generate(self, prompt, stream=False, timeout=30):
        if not MISTRAL_API_KEY:
            raise ModelFailure("Mistral API key missing")

        payload = {
            "model": "mistral-small-latest",
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
        }

        return super().generate(prompt, stream, timeout)


# -----------------------------
# GEMINI MODELS
# -----------------------------

class GeminiPro(BaseModel):
    name = "gemini-pro"
    endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"

    def generate(self, prompt, stream=False, timeout=30):
        if not GEMINI_API_KEY:
            raise ModelFailure("Gemini API key missing")

        payload = {
            "contents": [
                {"parts": [{"text": prompt}]}
            ]
        }

        try:
            r = requests.post(
                f"{self.endpoint}?key={GEMINI_API_KEY}",
                json=payload,
                timeout=timeout
            )
            r.raise_for_status()
            return r.json()["candidates"][0]["content"]["parts"][0]["text"]

        except Exception as e:
            raise ModelFailure(str(e))


class GeminiFlash(GeminiPro):
    name = "gemini-flash"
    endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"


# -----------------------------
# IMMORTAL MODEL ROUTER
# -----------------------------

class BaseModelRouter:
    """
    Silent failover brain.
    User NEVER sees failure.
    """

    def __init__(self):
        self.models = [
            MistralLarge(),
            MistralSmall(),
            GeminiPro(),
            GeminiFlash(),
        ]

    def ask(
        self,
        prompt: str,
        stream: bool = False,
        timeout: int = 30
    ) -> Generator[str, None, None] | str:

        for model in self.models:
            try:
                return model.generate(prompt, stream, timeout)
            except Exception as e:
                self._log_failure(model.name, e)
                continue

        return self._graceful_failure(stream)

    def _log_failure(self, model_name: str, error: Exception):
        print(f"[ZULTX FAILOVER] {model_name} failed")
        print(traceback.format_exc())

    def _graceful_failure(self, stream: bool):
        msg = (
            "ZULTX is stabilizing its core systems.\n"
            "Please try again in a moment."
        )
        if not stream:
            return msg
        for ch in msg:
            yield ch
            time.sleep(0.01)


# -----------------------------
# GLOBAL IMMORTAL ENTRYPOINT
# -----------------------------

_router = BaseModelRouter()


def ask(
    prompt: str,
    *,
    stream: bool = False,
    timeout: int = 30
) -> Generator[str, None, None] | str:
    """
    THE ONLY FUNCTION THE REST OF ZULTX CALLS.
    """
    return _router.ask(prompt, stream, timeout)
