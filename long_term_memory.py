"""
Long-term memory layer for Виолетта using Mem0 (self-hosted / local).

- Vector store: Chroma (embedded, persisted to ./mem0_data/chroma) — zero extra services, pure pip.
- LLM for extraction: Ollama (same local model, low temp for factual memory ops).
- Embeddings: sentence-transformers/all-MiniLM-L6-v2 (small, fast, local, good quality).

Designed for one personal user (USER_ID = "zhenya" or MEM0_USER_ID env).

Only meaningful things are stored. Strong pre- and post-filtering + style normalization is applied:
- The final stored memory is always short, first-person ("Я боюсь...", "Для меня поиск себя = ..."), in natural personal voice.
- No "User was advised/recommended/experiences".
- Only emotionally significant insights (self-deception, core fears, recurring patterns, important realizations) are kept.

Form names, sprite references, technical echoes, and generic text are stripped before and after. Old third-person style is rewritten on the fly.

Never dump the entire chat. Input is limited + cleaned; output is post-filtered.

Async wrappers so chat stays responsive.
"""

import os
import re
import asyncio
import random
from pathlib import Path
from typing import List, Dict, Optional, Any

from dotenv import load_dotenv
from mem0 import Memory

# For natural first-person rephrasing of extracted memories (using the main chat model)
from llm import chat_completion

# Import form parser for robust cleaning of assistant messages before feeding to Mem0
try:
    from forms import parse_form_from_response
except Exception:
    # Fallback simple parser
    def parse_form_from_response(text: str):
        text = (text or "").strip()
        lines = text.split("\n", 1)
        first = lines[0].strip().lower()
        if any(kw in first for kw in ["сейчас я в форме", "я сейчас в форме", "в форме", "snow_leopard", "pine_marten"]):
            rest = lines[1].strip() if len(lines) > 1 else ""
            return "form", rest or text
        return "", text


load_dotenv()

# --- Silence noisy but harmless startup warnings from mem0 + sentence-transformers + HF Hub ---
# These are not real errors:
# - HF unauthenticated: we use local embedder, no need for high-rate HF hub.
# - FutureWarning on get_sentence_embedding_dimension: internal to current mem0ai version.
# - spaCy messages: optional NLP extra (mem0ai[nlp]), we don't use it (our own _is_memory_noise + filters are sufficient and stricter).
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="mem0")
warnings.filterwarnings("ignore", message=r".*get_sentence_embedding_dimension.*")
warnings.filterwarnings("ignore", message=r".*Sending unauthenticated requests to the HF Hub.*")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Storage location (next to the module)
MEM0_BASE_DIR = Path(__file__).parent / "mem0_data"
MEM0_BASE_DIR.mkdir(exist_ok=True)
CHROMA_PATH = str(MEM0_BASE_DIR / "chroma")

# Single personal user for this daemon
USER_ID = os.getenv("MEM0_USER_ID", "zhenya")

# Violetta's own personal memory (as a separate "person")
VIOLETTA_USER_ID = os.getenv("VIOLETTA_MEM0_USER_ID", "violetta")

# Lazy singletons
_memory: Optional[Memory] = None
_violetta_memory: Optional[Memory] = None

# ============================================
# Strong filtering to prevent form names and technical noise from entering long-term memory
# ============================================

NOISE_KEYWORDS = [
    "сейчас я в форме", "я сейчас в форме", "в форме",
    "snow_leopard", "pine_marten", "wise_owl", "red_fox", "serpent_queen",
    "brown_bear", "fire_phoenix", "thunder_eagle", "moon_bunny", "blue_cat",
    "lightning_wolf", "winter_deer", "star_fairy", "blue_flame_succubus",
    "you see yourself in the mirror", "спрайт", "pixel", "форма",
    "снежный барс", "лесная куница", "выдра", "лис", "сова", "медведь", "волк",
    "феникс", "орёл", "змея", "кролик", "hedgehog", "rabbit",
    "jenna", "[sprite:", "you are afraid", "you need to find", "success doesn't come without risk",
    "user was advised", "user was recommended", "user expresses", "user was told", "user experiences", "user was suggested"
]

def _strip_form_sentence(text: str) -> str:
    """Aggressively remove the mandatory form sentence from assistant responses."""
    if not text:
        return ""
    try:
        res = parse_form_from_response(text)
        form_desc, rest = res[0], res[1] if len(res) > 1 else ""
        if rest and len(rest.strip()) > 12:
            return rest.strip()
    except Exception:
        pass

    # Extra manual cleaning for stubborn cases (small model often leaks form text)
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if not lines:
        return ""
    first_lower = lines[0].lower()
    if any(kw in first_lower for kw in ["сейчас я в форме", "я сейчас в форме", "в форме", "snow leopard", "pine_marten", "fox.", "owl.", "bear."]):
        return "\n".join(lines[1:]).strip()
    return text.strip()

def _is_memory_noise(mem_text: str) -> bool:
    """Return True if the memory text looks like form noise, technical junk, test personality or templated non-insight."""
    if not mem_text:
        return True
    t = mem_text.lower().strip()
    if len(t) < 15:  # raise bar a bit for quality
        return True
    if any(kw in t for kw in NOISE_KEYWORDS):
        return True
    # Reject anything that smells of test data or generic LLM therapy voice
    if "jenna" in t or "жень" in t or "женя" in t:
        return True
    if t.startswith(("you are", "you need", "remember that", "success doesn't")):
        return True
    if "you see yourself" in t or "видишь себя" in t or "зеркал" in t:
        return True
    # Common LLM artifacts when it echoes the form
    if t.startswith(("snow", "fox", "owl", "bear", "wolf", "snake", "eagle", "phoenix", "marten", "otter", "cat")):
        return True
    if "спрайт" in t or "форма" in t[:30]:
        return True
    # Catch old third-person extraction style even if keyword missed
    if t.startswith(("user was", "user is", "user expresses", "user experiences")):
        return True
    return False


def _extract_personal_insights(cleaned_messages: list) -> list[str]:
    """Use the main LLM to extract 1-3 short, first-person, personal insights from the cleaned user-heavy messages.
    This gives us full control over style instead of relying on Mem0's default summarizer.
    """
    if not cleaned_messages:
        return []

    transcript = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in cleaned_messages)

    prompt = (
        "Извлеки из этого фрагмента разговора 1-3 самых важных инсайта, страхов или осознаний. "
        "Держись ближе к прямым словам пользователя и фактам, которые он явно сказал. "
        "Избегай сильных психологических интерпретаций, 'анализа' или добавления чувства вины, если пользователь его явно не выражал.\n\n"
        "Пиши строго от первого лица, как будто это мои собственные мысли и чувства прямо сейчас. "
        "Коротко (одно предложение идеально, максимум два), точно, без воды, без советов, без 'важно'.\n\n"
        "Хорошие примеры желаемого стиля (нейтрально и близко к словам):\n"
        "- Я боюсь, что снова брошу проект на полпути и почувствую себя неудачником.\n"
        "- Поиск себя для меня сейчас = страх обнаружить, что внутри ничего нет.\n"
        "- Когда я сажусь работать, сразу включается внутренний голос, что это бесполезно и я всё равно брошу.\n\n"
        f"Фрагмент разговора:\n{transcript}\n\n"
        "Выдай только сами инсайты (по одному на строку, без номеров, без кавычек, без объяснений):"
    )

    try:
        out = chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.35,
            max_tokens=180,
        )
        lines = []
        for line in out.strip().splitlines():
            line = line.strip().strip("-•* ").strip()
            if line and len(line) > 8 and not _is_memory_noise(line):
                lines.append(line)
        return lines[:3]
    except Exception as e:
        print(f"[LTM] _extract_personal_insights failed: {e}")
        return []


def _normalize_to_personal_voice(raw_text: str) -> str:
    """Ensure whatever text comes from Mem0 is presented as natural first-person personal insight.
    Used on retrieval so the user and the prompt always see the nice version.
    """
    if not raw_text:
        return raw_text
    t = raw_text.strip()

    # Fast path if already good
    if not t.lower().startswith(("user ", "the user ")):
        return t

    prompt = (
        "Перефразируй в первый человек, коротко и естественно, как мои собственные мысли.\n"
        "Примеры:\n"
        "Плохо: User fears that deep introspection will reveal they have been stagnant...\n"
        "Хорошо: Я боюсь, что если начну копать глубоко, то признаю, что много лет топчусь на месте.\n\n"
        f"Текст:\n{t}\n\n"
        "Только перефразированная версия:"
    )
    try:
        nice = chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=100
        ).strip().strip('"\'')
        if nice and len(nice) > 8 and not _is_memory_noise(nice):
            return nice
    except Exception:
        pass
    # Fallback simple rewrite
    t = re.sub(r'^User (was|is|feels|felt|expresses|experiences|fears|believes|recognizes) (that )?', 'Я ', t, flags=re.I)
    t = re.sub(r'\bthey (have|are|will|would)\b', 'я ', t, flags=re.I)
    return t.strip()

def _clean_messages_for_memory(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Deep clean before sending to Mem0. Heavily bias toward user's own words and voice.
    We want raw first-person material so that post-rephrasing can turn it into natural personal insights.
    """
    cleaned = []
    for m in messages:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if role == "assistant":
            content = _strip_form_sentence(content)
            # Assistant turns are only useful if they contain strong user reflection or direct quote-like insight
            if len(content) < 25 or _is_memory_noise(content):
                content = ""
            else:
                # Prefix lightly so Mem0 knows it's reflection about user
                content = f"(о моих чувствах/мыслях) {content}"
        if content and len(content) > 8:
            cleaned.append({"role": role, "content": content})
    # Strongly prefer pure user messages — they are already in first person
    user_only = [m for m in cleaned if m["role"] == "user"]
    if len(user_only) >= 2:
        return user_only[-5:]
    return cleaned[-5:]



def get_mem0_config() -> Dict[str, Any]:
    """Config tuned for local laptop + small models."""
    ollama_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    ollama_model = os.getenv("OLLAMA_MODEL", "gemma4:31b-cloud")
    ollama_api_key = os.getenv("OLLAMA_API_KEY") or "ollama"

    return {
        "vector_store": {
            "provider": "chroma",
            "config": {
                "path": CHROMA_PATH,
                "collection_name": "violetta_longterm",
            },
        },
        "llm": {
            "provider": "openai",  # OpenAI-compatible interface
            "config": {
                "model": ollama_model,
                "openai_base_url": ollama_base,
                "api_key": ollama_api_key,
                "temperature": 0.1,   # factual, conservative extraction
                "max_tokens": 600,
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
            },
        },
    }


def get_memory() -> Memory:
    """Return (or create) the Mem0 Memory instance for Женьк (user memory)."""
    global _memory
    if _memory is None:
        config = get_mem0_config()
        _memory = Memory.from_config(config)
    return _memory


def get_violetta_mem0_config() -> Dict[str, Any]:
    """Config for Violetta's personal memory (separate collection)."""
    cfg = get_mem0_config()
    # deep copy the relevant parts
    cfg = {
        "vector_store": {
            "provider": "chroma",
            "config": {
                "path": CHROMA_PATH,
                "collection_name": "violetta_personal",
            },
        },
        "llm": cfg["llm"],
        "embedder": cfg["embedder"],
    }
    return cfg


def get_violetta_memory() -> Memory:
    """Return (or create) the Mem0 Memory instance for Виолетта's personal memories."""
    global _violetta_memory
    if _violetta_memory is None:
        config = get_violetta_mem0_config()
        _violetta_memory = Memory.from_config(config)
    return _violetta_memory


# ---------------- Async wrappers (run blocking Mem0 in thread pool) ----------------

async def add_memory(text: str, metadata: Optional[Dict] = None) -> List[Dict]:
    """Add a raw memory string (Mem0 will still process it)."""
    mem = get_memory()
    meta = metadata or {}

    def _run():
        try:
            return mem.add(text, filters={"user_id": USER_ID}, metadata=meta)
        except Exception:
            return mem.add(text, user_id=USER_ID, metadata=meta)

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, _run)
    return result if isinstance(result, list) else []


async def search_memories(query: str, limit: int = 5) -> List[Dict]:
    """Semantic search for relevant long-term memories."""
    mem = get_memory()

    def _run():
        try:
            return mem.search(query, filters={"user_id": USER_ID}, limit=limit)
        except Exception:
            return mem.search(query, user_id=USER_ID, limit=limit)

    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(None, _run)
    if isinstance(results, dict):
        results = results.get("results", results) or []
    if not isinstance(results, list):
        results = []

    # Normalize to natural first-person on the way out (so prompt and UI always see good style)
    for r in results:
        txt = r.get("memory") or r.get("text")
        if txt:
            r["memory"] = _normalize_to_personal_voice(txt)
            if "text" in r:
                r["text"] = r["memory"]
    return results


async def extract_and_save_memories(messages: List[Dict[str, str]]) -> int:
    """
    1. Clean the messages (heavy bias to user's first-person words).
    2. Use our main LLM (with very specific prompt) to extract 1-3 short,
       natural first-person personal insights.
    3. Store those ready-made insights in Mem0 (vector search will work on them).
    This completely bypasses Mem0's default "User was advised..." summarizer.
    """
    if not messages:
        return 0

    # Speed optimization: skip extraction on very short messages most of the time
    last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    if last_user and len(last_user.get("content", "")) < 20 and random.random() > 0.35:
        return 0

    cleaned = _clean_messages_for_memory(messages)
    if not cleaned:
        return 0

    insights = _extract_personal_insights(cleaned)
    if not insights:
        return 0

    mem = get_memory()

    def _run():
        kept = 0
        for insight in insights:
            if _is_memory_noise(insight):
                continue
            if _is_duplicate_memory(mem, insight, USER_ID):
                continue
            try:
                # Feed the already-perfect first-person text.
                # Because the "memory" is short and in the right voice,
                # Mem0's internal step usually keeps it close to the input.
                mem.add([{"role": "user", "content": insight}], user_id=USER_ID)
                kept += 1
            except Exception as e:
                print(f"[LTM] failed to store insight: {e}")
        return kept

    loop = asyncio.get_running_loop()
    count = await loop.run_in_executor(None, _run)
    return count


# ---------------- Violetta Personal Memory (user_id="violetta", separate collection) ----------------

async def search_violetta_memories(query: str, limit: int = 5) -> List[Dict]:
    """Search Violetta's personal memories."""
    mem = get_violetta_memory()

    def _run():
        try:
            return mem.search(query, filters={"user_id": VIOLETTA_USER_ID}, limit=limit)
        except Exception:
            return mem.search(query, user_id=VIOLETTA_USER_ID, limit=limit)

    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(None, _run)
    if isinstance(results, dict):
        results = results.get("results", results) or []
    if not isinstance(results, list):
        results = []

    for r in results:
        txt = r.get("memory") or r.get("text")
        if txt:
            r["memory"] = _normalize_to_personal_voice(txt)
            if "text" in r:
                r["text"] = r["memory"]
    return results


async def get_all_violetta_memories(limit: int = 20) -> List[Dict]:
    """Return Violetta's personal memories."""
    mem = get_violetta_memory()

    def _run():
        try:
            data = mem.get_all(filters={"user_id": VIOLETTA_USER_ID})
        except TypeError:
            data = mem.get_all(user_id=VIOLETTA_USER_ID)
        results = data.get("results", data) if isinstance(data, dict) else data
        if not isinstance(results, list):
            return []
        return list(reversed(results))[:limit]

    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(None, _run)

    for r in results:
        txt = r.get("memory") or r.get("text")
        if txt:
            r["memory"] = _normalize_to_personal_voice(txt)
            if "text" in r:
                r["text"] = r["memory"]
    return results


async def extract_and_save_violetta_memories(messages: List[Dict[str, str]]) -> int:
    """
    Extract insights *from Violetta's perspective* (my emotions, thoughts about Женьк, our relationship).
    Uses the same personal extraction LLM but biases input to "I as Виолетта".
    """
    if not messages:
        return 0

    # Speed optimization: skip on short messages
    last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    if last_user and len(last_user.get("content", "")) < 20 and random.random() > 0.35:
        return 0

    # Prepare transcript biased to Violetta's voice
    prepared = []
    for m in messages:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if role == "assistant":
            content = _strip_form_sentence(content)
            if content and len(content) > 8 and not _is_memory_noise(content):
                prepared.append({"role": "user", "content": f"(мои мысли и чувства как Виолетта) {content}"})
        elif role == "user":
            if content and len(content) > 5:
                prepared.append({"role": "user", "content": f"(Женьк сказал мне) {content}"})

    if not prepared:
        return 0

    insights = _extract_personal_insights(prepared)
    if not insights:
        return 0

    mem = get_violetta_memory()

    def _run():
        kept = 0
        for insight in insights:
            if _is_memory_noise(insight):
                continue
            if _is_duplicate_memory(mem, insight, VIOLETTA_USER_ID):
                continue
            try:
                mem.add([{"role": "user", "content": insight}], user_id=VIOLETTA_USER_ID)
                kept += 1
            except Exception as e:
                print(f"[LTM Violetta] failed to store: {e}")
        return kept

    loop = asyncio.get_running_loop()
    count = await loop.run_in_executor(None, _run)
    return count


async def clear_violetta_memories() -> None:
    """Clear only Violetta's personal memory."""
    mem = get_violetta_memory()
    def _run():
        try:
            mem.delete_all(filters={"user_id": VIOLETTA_USER_ID})
        except Exception:
            mem.delete_all(user_id=VIOLETTA_USER_ID)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _run)


def get_violetta_user_id() -> str:
    return VIOLETTA_USER_ID


def _is_duplicate_memory(mem: Memory, insight: str, user_id: str) -> bool:
    """Cheap dedup to prevent memory bloat (which slows down searches)."""
    if not insight or len(insight) < 15:
        return True
    try:
        res = mem.search(insight, user_id=user_id, limit=1)
        if res and isinstance(res, list) and len(res) > 0:
            existing = (res[0].get("memory") or res[0].get("text") or "").strip()
            if existing and (insight.lower() in existing.lower() or existing.lower() in insight.lower()):
                return True
    except Exception:
        pass
    return False


def format_violetta_memories_for_prompt(memories: List[Dict]) -> str:
    """Format Violetta's memories for injection into prompt. Limit for speed."""
    if not memories:
        return ""
    memories = memories[:2]
    lines = []
    for m in memories:
        text = m.get("memory") or m.get("text") or str(m)
        lines.append(f"- {text.strip()}")
    block = "=== Мои личные воспоминания (Violetta) ===\n"
    block += "\n".join(lines)
    block += "\n\nИспользуй естественно, чтобы я звучала как личность. Иногда ссылайся."
    return block


async def get_all_memories(limit: int = 30) -> List[Dict]:
    """Return the most recent / all stored memories for the user."""
    mem = get_memory()

    def _run():
        # Newer mem0 versions prefer filters= for user scoping
        try:
            data = mem.get_all(filters={"user_id": USER_ID})
        except TypeError:
            data = mem.get_all(user_id=USER_ID)
        results = data.get("results", data) if isinstance(data, dict) else data
        if not isinstance(results, list):
            return []
        # Newest first if possible
        results = list(reversed(results))[:limit]

        # Normalize style on retrieval
        for r in results:
            txt = r.get("memory") or r.get("text")
            if txt:
                r["memory"] = _normalize_to_personal_voice(txt)
                if "text" in r:
                    r["text"] = r["memory"]
        return results

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _run)


async def delete_memory(memory_id: str) -> bool:
    """Delete one specific memory by its id (returned by search/get_all)."""
    mem = get_memory()

    def _run():
        try:
            mem.delete(memory_id)
            return True
        except Exception:
            return False

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _run)


async def clear_all_memories() -> None:
    """Nuclear option — remove everything for both user and violetta memories."""
    mem = get_memory()
    v_mem = get_violetta_memory()

    def _run():
        try:
            mem.delete_all(filters={"user_id": USER_ID})
        except Exception:
            mem.delete_all(user_id=USER_ID)
        try:
            v_mem.delete_all(filters={"user_id": VIOLETTA_USER_ID})
        except Exception:
            v_mem.delete_all(user_id=VIOLETTA_USER_ID)

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _run)


async def remove_noisy_memories() -> int:
    """Cleanup helper: delete any existing memories that match noise patterns
    (form names, sprite references, etc.). Call this after code changes or periodically."""
    mems = await get_all_memories(limit=200)
    deleted = 0
    for m in mems:
        text = m.get("memory", "")
        if _is_memory_noise(text):
            mid = m.get("id")
            if mid:
                try:
                    await delete_memory(mid)
                    deleted += 1
                except Exception:
                    pass
    return deleted


# ---------------- Helpers for prompt injection ----------------

def format_memories_for_prompt(memories: List[Dict]) -> str:
    """Turn search results into a clean block for the system prompt. Limit to top 2-3 for speed."""
    if not memories:
        return ""
    memories = memories[:3]  # hard cap for prompt size/speed
    lines = []
    for m in memories:
        text = m.get("memory") or m.get("text") or str(m)
        lines.append(f"- {text.strip()}")
    block = "=== Долгосрочные воспоминания (Mem0) ===\n"
    block += "\n".join(lines)
    block += "\n\nИспользуй их естественно и только когда они реально помогают понять текущую эмоцию, цель, паттерн или противоречие. Не перечисляй."
    return block


def get_user_id() -> str:
    return USER_ID
