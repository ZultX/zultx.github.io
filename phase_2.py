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
import json
import hashlib
# -----------------------------
# CONFIG
# -----------------------------

ADAPTER_DIR = Path("prompt/adapters")


# -----------------------------
# ADAPTER LOADER
# -----------------------------
MANIFEST_PATH = ADAPTER_DIR / "manifest.json"

def load_manifest():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError("manifest.json missing in adapters directory")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

def load_adapters():
    manifest = load_manifest()
    adapters = {}

    for group_name in manifest["order"]:
        files = manifest["groups"].get(group_name, [])
        for fname in files:
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
    phase: str = "full",  # rag | memory | full
    extra_rules: Optional[List[str]] = None
) -> str:
    manifest = load_manifest()
    blocks: List[str] = []

    for group in manifest["order"]:
        # phase gating
        if phase == "rag" and group == "memory":
            continue
        if phase == "memory" and group == "truth":
            continue

        for fname in manifest["groups"][group]:
            text = _ADAPTERS[fname]

            if fname == "persona_base.txt" and persona:
                text += f"\n\nActive persona: {persona}"

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
    phase: str = "full",
    stream: bool = False,
    timeout: int = 30,
    **_
):
    """
    THE OFFICIAL ZULTX INTERFACE FROM v1.1 ONWARD.

    - Applies adapter soul
    - Delegates to Phase_1 (immortal brain)
    - Safe for all future upgrades
    """
    system_prompt = compose_system_prompt(
    persona=persona,
    phase=phase
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
