"""
LLM client abstraction for Violett a.

Supports:
- Ollama (local, OpenAI compatible) — default
- xAI / Grok API (if XAI_API_KEY is set)

Uses the official openai library for maximum compatibility and streaming.
"""

import os
from typing import List, Dict, AsyncGenerator, Optional
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# Configuration (defaults from .env at start)
XAI_API_KEY = os.getenv("XAI_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
XAI_MODEL = os.getenv("XAI_MODEL", "grok-3")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")  # for cloud/ollama-hosted models that require subscription key

# Performance tuning for fast natural responses on laptop
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.68"))
OLLAMA_MAX_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", "380"))

# Runtime current model (allows switch without server restart for same backend group)
_current_model = os.getenv("OLLAMA_MODEL", "gemma4:31b-cloud")

# Known ollama-hosted cloud models (offered in switcher). These still go through local ollama serve as proxy in the default setup.
# For direct OpenAI-compatible providers (Groq, Together, OpenRouter, Fireworks etc.) just set OLLAMA_BASE_URL and OLLAMA_API_KEY + any model name — the key will be used automatically.
KNOWN_CLOUD_MODELS = [
    "gemma4:31b-cloud",
    "nemotron-3-super:cloud",
    "kimi-k2.6:cloud",
    "kimi-k2.5:cloud",
]

def get_backend(model: str | None = None) -> str:
    """Return active backend name. 'cloud' for ollama-hosted, 'ollama' for local pulled."""
    if XAI_API_KEY:
        return "xai"
    m = (model or _current_model).lower()
    if "cloud" in m or ":cloud" in m or m.startswith(("kimi", "minimax", "glm", "qwen")):
        return "cloud"
    return "ollama"

def get_client(async_client: bool = False) -> OpenAI | AsyncOpenAI:
    """Return sync or async OpenAI-compatible client. Uses current runtime model."""
    if XAI_API_KEY:
        base_url = "https://api.x.ai/v1"
        api_key = XAI_API_KEY
        model = XAI_MODEL
    else:
        base_url = OLLAMA_BASE_URL
        # Always prefer real key if provided (works for Ollama local/cloud + any other OpenAI-compatible provider like Groq, Together, OpenRouter etc.)
        # For pure local ollama without key it falls back to "ollama" (which is ignored anyway).
        api_key = OLLAMA_API_KEY or "ollama"
        model = get_model_name()

    ClientClass = AsyncOpenAI if async_client else OpenAI
    return ClientClass(api_key=api_key, base_url=base_url)

def get_model_name() -> str:
    """Current active model (runtime switcheable for same-backend)."""
    return XAI_MODEL if XAI_API_KEY else _current_model

def set_current_model(model: str):
    """Switch model at runtime (only within same backend group recommended; mem0 extraction stays on init model)."""
    global _current_model
    _current_model = (model or "").strip() or _current_model

def get_model_display(model: str | None = None) -> str:
    """Human label for pill / UI."""
    name = model or get_model_name()
    m = name.lower()
    if "e2b" in m:
        return f"{name} (E2B)"
    elif "e4b" in m:
        return f"{name} (E4B)"
    elif "cloud" in m or ":cloud" in m:
        return f"{name} (cloud)"
    return name

async def stream_chat_completion(
    messages: List[Dict[str, str]],
    temperature: float = None,
    max_tokens: Optional[int] = None,
) -> AsyncGenerator[str, None]:
    """
    Stream tokens from the LLM.
    Yields raw text chunks.
    """
    client = get_client(async_client=True)
    model = get_model_name()

    # Use configured defaults for speed + natural length on this hardware
    if temperature is None:
        temperature = OLLAMA_TEMPERATURE
    if max_tokens is None:
        max_tokens = OLLAMA_MAX_TOKENS

    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        # Surface the error in a user-friendly way
        yield f"\n\n[Ошибка подключения к LLM ({get_backend()}): {str(e)}]\nПроверь OLLAMA_BASE_URL / модель (для cloud: kimi-k2.6:cloud и т.д.), или XAI_API_KEY."

def build_messages(
    system_prompt: str,
    history: List[Dict],
    current_user_message: str,
    current_form_instruction: str = "",
    memory_context: str = ""
) -> List[Dict[str, str]]:
    """
    Construct the full prompt for the model.
    Includes system + optional long-term memory block + form instruction + history + user.
    memory_context should be the formatted string from long_term_memory.format_memories_for_prompt()
    """
    msgs = [{"role": "system", "content": system_prompt}]

    if memory_context:
        msgs.append({"role": "system", "content": memory_context})

    if current_form_instruction:
        msgs.append({"role": "system", "content": current_form_instruction})

    for h in history:
        msgs.append({"role": h["role"], "content": h["content"]})

    msgs.append({"role": "user", "content": current_user_message})
    return msgs

# Simple non-streaming helper (useful for testing or special calls)
def chat_completion(
    messages: List[Dict[str, str]],
    temperature: float = None,
    max_tokens: Optional[int] = None,
) -> str:
    client = get_client(async_client=False)
    model = get_model_name()

    if temperature is None:
        temperature = OLLAMA_TEMPERATURE
    if max_tokens is None:
        max_tokens = OLLAMA_MAX_TOKENS

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


# --- Model discovery and safe runtime switching (for UI switcher) ---

def list_local_models() -> list[str]:
    """Discover models pulled locally via ollama (OpenAI /v1/models compat)."""
    try:
        c = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        data = c.models.list()
        names = []
        for m in getattr(data, "data", []):
            mid = getattr(m, "id", None) or str(m)
            if mid and "cloud" not in str(mid).lower():
                names.append(str(mid))
        return sorted(set(names))
    except Exception:
        # Safe fallback (known from this env)
        return ["gemma4:e4b", "gemma4:e2b", "gemma3:4b", "gemma3:1b", "qwen2.5:3b", "gemma2:2b"]


def list_cloud_models() -> list[str]:
    """Known cloud models (nemotron, gemma cloud etc). Offered even if not 'pulled' — ollama proxies on first use."""
    return KNOWN_CLOUD_MODELS[:]


def get_available_models() -> dict:
    """For /api/models : current + grouped lists."""
    curr = get_model_name()
    return {
        "current": curr,
        "current_backend": get_backend(),
        "display": get_model_display(),
        "local": list_local_models(),
        "cloud": list_cloud_models(),
    }


def can_switch_to(new_model: str) -> tuple[bool, str | None]:
    """Same-group only (cloud<->cloud or local<->local). Cross requires server restart (different mem0/ctx etc)."""
    if not new_model:
        return False, "empty model"
    curr_b = get_backend()
    new_b = get_backend(new_model)
    if curr_b == new_b:
        return True, None
    # cross group
    return False, "Невозможно переключаться между cloud и local без перезапуска сервера. Контекст будет потерян."
