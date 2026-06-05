"""
Custom HTML/JS chat server for Виолетта (Violett a) - no Chainlit.

Uses FastAPI for API + streaming (SSE), serves a beautiful self-contained index.html
with Tailwind CDN + vanilla JS.

Reuses:
- character_prompt.py for SYSTEM_PROMPT and form helpers
- llm.py for streaming completions (Ollama / xAI)
- memory.py for persistent SQLite history + current form
- long_term_memory.py for Mem0 (self-hosted long-term facts, goals, emotional patterns)
- forms.py for sprite selection + URLs (now also biased by long-term memories)

Layout per assistant message (as requested):
- Left: pixel avatar (sprite) + caption below with the "почему" (the full form sentence from LLM)
- Right: the actual response text, in a chat bubble like normal conversation.

User messages on the right.

Live streaming: tokens appear live in the right text. On "final" the left avatar+caption is set/updated (transform effect).

Run with: uvicorn server:app --reload --port 8000
(or use the updated start-*.ps1 / .bat)

The sprites are served statically from /forms/ (the Forms/ dir).
"""

import os
import json
import asyncio
import re
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Body, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Reuse existing modules (they live in the same dir)
from character_prompt import SYSTEM_PROMPT, get_form_instruction, get_initial_form_description
from llm import (
    build_messages,
    stream_chat_completion,
    OLLAMA_TEMPERATURE,
    OLLAMA_MAX_TOKENS,
    get_model_name,
    get_model_display,
    get_backend,
    get_available_models,
    set_current_model,
    can_switch_to,
)
from memory import init_db, get_main_conversation_id, add_message, get_recent_messages, set_current_form, get_current_form
from long_term_memory import (
    search_memories,
    extract_and_save_memories,
    get_all_memories,
    delete_memory,
    format_memories_for_prompt,
    get_user_id as get_ltm_user_id,
    clear_all_memories,
    # Violetta personal memory
    search_violetta_memories,
    extract_and_save_violetta_memories,
    get_all_violetta_memories,
    format_violetta_memories_for_prompt,
    get_violetta_user_id,
    clear_violetta_memories,
)
from forms import (
    get_sprite_path,
    get_random_form,
    get_sprite_url_for_form,
    get_all_form_sprites,
    parse_form_from_response,
    get_form_by_qualities,
    get_form_for_context,
)  # single source for sprite + quality selection + parse (aligned to new prompt block)

# Make sure DB is ready
init_db()
CONVO_ID = get_main_conversation_id()

app = FastAPI(title="Виолетта — твой AI-даэмон")

# CORS for local dev (if opening html directly etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the pixel sprites statically at /forms/ (note: the folder on disk is "Forms" capital F)
FORMS_DIR = Path(__file__).parent / "Forms"
app.mount("/forms", StaticFiles(directory=str(FORMS_DIR), html=False), name="forms")

# ---------- Helpers (parse is also in forms.py, import it) ----------
# We import parse_form_from_response from forms (added there for sharing)
# If import fails for some reason, the one in forms.py now includes it.

# ---------- API ----------

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the custom HTML chat UI."""
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>index.html not found — create it next to server.py</h1>", status_code=500)

@app.get("/api/history")
async def get_history():
    """Return recent messages, enriched with sprite_url for assistant messages that have a form."""
    msgs = get_recent_messages(CONVO_ID, limit=80)
    enriched = []
    for m in msgs:
        item = {
            "role": m["role"],
            "content": m["content"],
            "timestamp": m.get("timestamp"),
        }
        if m["role"] == "assistant" and m.get("form_description"):
            item["form_description"] = m["form_description"]
            item["sprite_url"] = get_sprite_url_for_form(m["form_description"])
        enriched.append(item)
    return enriched

@app.get("/api/forms")
async def get_forms():
    """List all possible forms with their sprite URLs (for the gallery / inspiration buttons)."""
    return get_all_form_sprites()


@app.get("/api/current-form")
async def get_current_form_api():
    """Return the persisted current form (for ambient visual presence layer)."""
    try:
        form_desc = get_current_form()
        if not form_desc:
            # fallback to initial (already imported at top)
            form_desc = get_initial_form_description()
        sprite_url = get_sprite_url_for_form(form_desc or "")
        return {
            "form": form_desc or "",
            "sprite_url": sprite_url,
            "key": (form_desc or "").lower().replace(" ", "_")
        }
    except Exception as e:
        # graceful fallback
        sprite_url = get_sprite_url_for_form("")
        return {"form": "", "sprite_url": sprite_url, "key": "pine_marten", "error": str(e)}


@app.get("/api/model")
async def get_current_model():
    """Return the active model (runtime)."""
    try:
        model = get_model_name()
        return {
            "model": model,
            "display": get_model_display(model),
            "backend": get_backend(model),
        }
    except Exception as e:
        return {"model": "unknown", "display": "unknown", "backend": "unknown", "error": str(e)}


@app.get("/api/models")
async def list_available_models():
    """Grouped models for switcher UI. Discovers locals, offers known clouds."""
    try:
        return get_available_models()
    except Exception as e:
        return {"error": str(e), "current": get_model_name(), "current_backend": get_backend(), "local": [], "cloud": []}


@app.post("/api/model/switch")
async def switch_current_model(payload: dict = Body(...)):
    """Runtime switch (same backend group only). History (SQLite) is preserved automatically."""
    new_model = (payload.get("model") or "").strip()
    if not new_model:
        return {"success": False, "error": "model is required"}
    allowed, msg = can_switch_to(new_model)
    if not allowed:
        return {
            "success": False,
            "error": msg,
            "requires_restart": True,
            "current": get_model_name(),
        }
    old = get_model_name()
    set_current_model(new_model)
    return {
        "success": True,
        "old_model": old,
        "new_model": new_model,
        "backend": get_backend(),
        "display": get_model_display(),
        "message": f"Модель переключена на {new_model}. История чата сохранена.",
    }


# ---------- Long-term memory endpoints (Mem0) ----------
@app.get("/api/memories")
async def get_memories():
    """Return current long-term memories for the user (for UI or debugging)."""
    try:
        mems = await get_all_memories(limit=30)
        return {
            "user_id": get_ltm_user_id(),
            "count": len(mems),
            "memories": mems,
        }
    except Exception as e:
        return {"error": str(e), "memories": []}


@app.delete("/api/memories/{memory_id}")
async def delete_memory_endpoint(memory_id: str):
    """Delete one specific memory by id (id comes from /api/memories)."""
    try:
        ok = await delete_memory(memory_id)
        return {"success": ok, "deleted_id": memory_id}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/memories/clean-noise")
async def clean_noise_memories():
    """Run the aggressive noise filter on existing memories (removes form names etc.)."""
    try:
        from long_term_memory import remove_noisy_memories
        deleted = await remove_noisy_memories()
        remaining = await get_all_memories(5)
        return {"deleted": deleted, "remaining_count": len(remaining)}
    except Exception as e:
        return {"error": str(e)}

class ChatIn(BaseModel):
    message: str

@app.post("/api/chat")
async def chat_endpoint(payload: ChatIn):
    """Send a user message. Returns SSE stream: 'token' events for live text, then 'final' with form + clean text."""
    user_text = payload.message.strip()
    if not user_text:
        return JSONResponse({"error": "empty message"}, status_code=400)

    # Persist user message
    add_message(CONVO_ID, "user", user_text)

    # === Long-term memory (Mem0) — two separate memories ===
    memory_context = ""
    relevant_memories: list = []
    violetta_memories: list = []
    try:
        relevant_memories = await search_memories(user_text, limit=2)
        violetta_memories = []
        low = user_text.lower()
        # Only search Violetta personal memory when user is asking about *her* feelings/reactions (saves tokens and time on normal turns)
        if any(w in low for w in ["ты", "виолетта", "чувствуешь", "твои", "твоя", "твоё", "твоего", "как ты", "что ты"]):
            violetta_memories = await search_violetta_memories(user_text, limit=2)
        parts = []
        if relevant_memories:
            parts.append(format_memories_for_prompt(relevant_memories))
        if violetta_memories:
            parts.append(format_violetta_memories_for_prompt(violetta_memories))
        if parts:
            memory_context = "\n\n".join(parts)
    except Exception as e:
        # Memory is best-effort; never break the chat
        print(f"[LTM] search failed (non-fatal): {e}")

    # Special memory commands (explicit phrases only; bare "память" or similar now treated as normal message + natural memory_context injection)
    low = user_text.lower()
    is_memory_command = False
    special_response = None

    if any(phrase in low for phrase in ["покажи память", "воспоминания", "мои воспоминания", "что ты помнишь"]):
        is_memory_command = True
        try:
            all_mems = await get_all_memories(limit=10)
            v_mems = await get_all_violetta_memories(limit=6)
            parts = []
            if all_mems:
                lines = [f"• {m.get('memory', m)}" for m in all_mems]
                parts.append("О Женьке:\n" + "\n".join(lines))
            if v_mems:
                lines = [f"• {m.get('memory', m)}" for m in v_mems]
                parts.append("Мои воспоминания обо всём этом:\n" + "\n".join(lines))
            if parts:
                special_response = "\n\n".join(parts) + "\n\n(Можешь использовать кнопки или команды для удаления.)"
            else:
                special_response = "Пока у меня нет сохранённых воспоминаний. Мы начнём накапливать их."
        except Exception as e:
            special_response = f"Не смогла достать память (ошибка: {e})."

    elif any(phrase in low for phrase in ["забудь всё", "очисти память", "clear memory", "удали все воспоминания"]):
        is_memory_command = True
        try:
            await clear_all_memories()
            special_response = "Долгосрочная память очищена. Начинаем с чистого листа."
        except Exception as e:
            special_response = f"Не смогла очистить память: {e}"

    # Direct form command support ("стань X", "become Y", gallery click) — use qualities + keyword resolution for reliability
    forced_sprite_url = None
    forced_form_hint = None
    if low.startswith(("стань ", "become ", "прими форму ", "я в форме ", "form ")):
        desired_raw = low.split(" ", 1)[1].strip()
        desired = desired_raw.replace(" ", "_")

        # Prefer quality-based selection (matches the new prompt examples exactly)
        q_key, q_path = get_form_by_qualities([desired_raw, desired.replace("_", " ")])
        sprite_for_direct = get_sprite_path(desired) or get_sprite_path(desired_raw) or q_path

        if sprite_for_direct:
            fname = Path(sprite_for_direct).name
            forced_sprite_url = f"/forms/{fname}"
            # nice sentence stub (LLM will refine with real emotions from context)
            forced_form_hint = f"Пользователь попросил: «{user_text}». В этом ответе обязательно начни с «Сейчас я в форме ... — чувствую твою ...» (используй 2-3 реальных качества из контекста + эту просьбу), и укажи спрайт. Выбери форму, которая лучше всего подходит под текущие эмоции разговора."

    # Build prompt (reuse the exact same logic as before)
    history = get_recent_messages(CONVO_ID, limit=30)
    form_instruction = get_form_instruction()
    llm_messages = build_messages(
        SYSTEM_PROMPT,
        history,
        user_text,
        current_form_instruction=form_instruction,
        memory_context=memory_context,
    )
    if forced_form_hint:
        llm_messages.append({"role": "system", "content": forced_form_hint})

    # Short-circuit for pure memory commands (no need to wake the full model)
    if is_memory_command and special_response:
        add_message(CONVO_ID, "assistant", special_response)

        async def _special_stream():
            yield f"event: token\ndata: {json.dumps({'text': special_response})}\n\n"
            final = {
                "form": "",
                "sprite_url": get_sprite_url_for_form(""),
                "clean_text": special_response,
            }
            yield f"event: final\ndata: {json.dumps(final)}\n\n"

        return StreamingResponse(_special_stream(), media_type="text/event-stream")

    async def event_stream():
        full_response = ""
        try:
            async for token in stream_chat_completion(
                llm_messages,
                temperature=OLLAMA_TEMPERATURE,
                max_tokens=OLLAMA_MAX_TOKENS,
            ):
                full_response += token
                # Live tokens (the client will show raw text while generating, including the form sentence at the start)
                yield f"event: token\ndata: {json.dumps({'text': token})}\n\n"
                await asyncio.sleep(0)  # be nice to event loop

            # Generation done — parse sprite via marker (new) or fallback. Then aggressively clean visible text.
            form_desc, clean_text, ambient_cmds = parse_form_from_response(full_response)
            if ambient_cmds is None:
                ambient_cmds = []

            # Extra safety: strip marker + any leaked form names / animal words / "в форме" from the visible text the user sees
            clean_text = re.sub(r'\[SPRITE:[^\]]+\]', '', clean_text, flags=re.IGNORECASE).strip()
            clean_text = re.sub(r'\[AMBIENT:[^\]]+\]', '', clean_text, flags=re.IGNORECASE).strip()
            # Remove leading lines that are clearly form declarations or animal references
            lines = [l for l in clean_text.splitlines() if l.strip()]
            while lines and any(kw in lines[0].lower() for kw in ["snow", "fox", "owl", "bear", "wolf", "pine", "serpent", "мартен", "барс", "лис", "сова", "в форме", "сейчас я в форме", "я в форме"]):
                lines = lines[1:]
            # Also scrub animal words inside the remaining text (small model leaks them)
            animal_words = ["snow leopard", "pine marten", "wise owl", "red fox", "serpent", "brown bear", "lightning wolf", "fire phoenix", "thunder eagle", "moon bunny", "снежный барс", "лесная куница", "выдра", "лис", "сова", "медведь"]
            for w in animal_words:
                lines = [l.replace(w, "").replace(w.title(), "") for l in lines]
            clean_text = "\n".join(lines).strip()
            if not clean_text:
                clean_text = "..."  # fallback if everything was stripped (rare)

            if form_desc:
                set_current_form(form_desc)
                add_message(CONVO_ID, "assistant", clean_text, form_description=form_desc)
                sprite_url = get_sprite_url_for_form(form_desc)
            else:
                add_message(CONVO_ID, "assistant", full_response)
                sprite_url = get_sprite_url_for_form("")  # random

            # === Long-term memory save (async, non-blocking) ===
            # Save to BOTH memories after response.
            # User memory: about Женьк. Violetta memory: my own feelings/reactions/relationship insights.
            try:
                assistant_clean = (clean_text or full_response).strip()
                recent_turns = [m for m in history[-4:] if m.get("content", "").strip()]
                recent_for_mem = recent_turns + [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_clean},
                ]
                asyncio.create_task(extract_and_save_memories(recent_for_mem))
                # Separate extraction for Violetta's personal perspective (my emotions about him)
                asyncio.create_task(extract_and_save_violetta_memories(recent_for_mem))
            except Exception as e:
                print(f"[LTM] schedule extract failed (non-fatal): {e}")

            # Boost sprite selection with original user_text keywords + long-term memories (qualities/emotions).
            # Now uses BOTH memories so Violetta's own "feelings" can influence the visual reaction too.
            if (not form_desc or "не распознана" in (form_desc or "") or "текущая форма" in (form_desc or "")):
                mem_texts = [m.get("memory", "") for m in relevant_memories] + [m.get("memory", "") for m in (violetta_memories or [])]
                try:
                    _, boosted_path = get_form_for_context(user_text, mem_texts)
                    if boosted_path:
                        boosted = f"/forms/{Path(boosted_path).name}"
                    else:
                        boosted = None
                except Exception:
                    boosted = get_sprite_url_for_form(user_text)
                if boosted:
                    sprite_url = boosted

            # If this was a direct "стань ..." command, prefer the forced sprite we resolved via qualities/keywords
            if forced_sprite_url:
                sprite_url = forced_sprite_url
                # if parse gave weak form_desc, keep a good one (LLM should have produced it thanks to hint)
                if not form_desc or "не распознана" in form_desc or len(form_desc) < 10:
                    form_desc = f"Сейчас я в форме, которую ты попросил — чувствую контекст и твою просьбу."

            # === Parallel ambient / visual body control channel ===
            # LLM can emit hidden [AMBIENT: ...] markers (stripped from text).
            # We yield them as separate 'ambient' SSE events (parallel to token/final).
            # This lets Violetta control her presence (form, opacity, position, dust, burst)
            # independently of the words she "speaks" to the user.
            # Also support set_form via ambient as override (for full autonomy over her visual self).
            effective_form = form_desc or ""
            for ac in (ambient_cmds or []):
                ac_type = (ac.get("type") or "").lower().replace("-", "_")
                ac_val = ac.get("value")
                if ac_type in ("set_form", "form") and ac_val:
                    effective_form = str(ac_val).lower().strip().replace(" ", "_")
                    try:
                        set_current_form(effective_form)
                    except Exception:
                        pass
                    form_desc = effective_form
            if effective_form and effective_form != (form_desc or ""):
                sprite_url = get_sprite_url_for_form(effective_form)

            # Send each ambient command as its own event on the parallel meta channel.
            # Client applies them live via the state manager (no visible text).
            for ac in (ambient_cmds or []):
                try:
                    yield f"event: ambient\ndata: {json.dumps(ac)}\n\n"
                except Exception:
                    pass

            final_payload = {
                "form": form_desc or "",
                "sprite_url": sprite_url,
                "clean_text": clean_text or full_response
            }
            yield f"event: final\ndata: {json.dumps(final_payload)}\n\n"

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# ---------- Run hint ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
