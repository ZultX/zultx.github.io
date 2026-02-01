"""
ZULTX Phase 2 — ADAPTER ORCHESTRA (SOUL ENGINE)

This file composes all adapters safely and deterministically.
It wraps Phase_1.ask() — never replaces it.

Responsibilities:
- Load adapter prompts
- Compose system prompt stack
- Enforce policy order
- Prepare final prompt for model
- Remain model-agnostic (works with failover)

Author: ZULTX Core
"""

from pathlib import Path
from typing import Dict, List, Optional
from phase_1 import ask as base_ask

# -----------------------------
# CONFIG
# -----------------------------

ADAPTER_DIR = Path("prompt/adapters")

# Order MATTERS — this is the soul chain
ADAPTER_ORDER = [
    "core_identity.txt",
    "persona_base.txt",
    "safety_precheck.txt",
    "minor_first_policy.txt",
    "crisis_redirection.txt",
    "identity_boundary.txt",
    "memory_decision.txt",
    "memory_scope.txt",
    "fact_verification.txt",
    "hallucination_dampener.txt",
    "model_governance.txt",
    "latency_streaming.txt",
    "format_adapter.txt",
    "brevity_depth.txt",
    "creativity_modulator.txt",
    "cache_intelligence.txt",
    "tool_boundary.txt",
    "multimodal_governance.txt",
    "safety_postcheck.txt",
    "self_audit.txt",
]

# -----------------------------
# ADAPTER LOADER
# -----------------------------

def load_adapters() -> Dict[str, str]:
    """
    Loads adapter prompt files into memory.
    """
    adapters = {}
    for fname in ADAPTER_ORDER:
        path = ADAPTER_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing adapter: {fname}")
        adapters[fname] = path.read_text(encoding="utf-8").strip()
    return adapters


_ADAPTERS = load_adapters()


# -----------------------------
# PROMPT COMPOSER
# -----------------------------

def compose_system_prompt(
    *,
    persona: Optional[str] = None,
    extra_rules: Optional[List[str]] = None
) -> str:
    """
    Builds the full system prompt from adapters.
    """
    blocks: List[str] = []

    for name in ADAPTER_ORDER:
        text = _ADAPTERS[name]

        # Persona is modular
        if name == "persona_base.txt" and persona:
            text = text + f"\n\nActive persona: {persona}"

        blocks.append(text)

    if extra_rules:
        blocks.append("\n".join(extra_rules))

    return "\n\n---\n\n".join(blocks)


# -----------------------------
# PUBLIC ENTRYPOINT (PHASE 2)
# -----------------------------
def ask(
    user_input: str,
    *,
    persona: Optional[str] = None,
    mode: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    speed: Optional[float] = None,
    stream: bool = False,
    timeout: int = 30,
    **_ignore
):
    """
    THE OFFICIAL ZULTX INTERFACE FROM v1.1 ONWARD.

    - Applies adapter soul
    - Delegates to Phase_1 (immortal brain)
    - Safe for all future upgrades
    """

    system_prompt = compose_system_prompt(
        persona=persona
    )

    final_prompt = (
        f"<<SYSTEM>>\n{system_prompt}\n\n"
        f"<<USER>>\n{user_input}\n\n"
        f"<<ZULTX>>"
    )

    return base_ask(
        final_prompt,
        stream=stream,
        timeout=timeout
)
