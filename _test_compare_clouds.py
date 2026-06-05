"""
Comparative test for two cloud models for Violetta:
- nemotron-3-super:cloud
- gemma4:31b-cloud

Same deep realistic dialog (project, self-deception/inner voices, fear of abandoning, health background, fatigue, "inner enemy", honest root cause wanted).

Direct client per model (no globals pollution).
Uses full SYSTEM_PROMPT + get_form_instruction().
Same strip + parse as server.
Memory: tries to use search/extract (mem0 will be on whatever .env was at import time for extraction LLM).
Clears memories before each full run for clean compare.
Prints per-turn: user, time, chosen form, visible response, +mem count if any.
At end: memories + short per-model notes.
"""

import os
import time
import re
import asyncio
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from character_prompt import SYSTEM_PROMPT, get_form_instruction
from forms import parse_form_from_response, get_sprite_path
from long_term_memory import (
    search_memories,
    extract_and_save_memories,
    get_all_memories,
    clear_all_memories,
    format_memories_for_prompt,
)

BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
API_KEY = os.getenv("OLLAMA_API_KEY") or "ollama"
TEMP = float(os.getenv("OLLAMA_TEMPERATURE", "0.6"))
MAX_TOK = int(os.getenv("OLLAMA_MAX_TOKENS", "420"))

def make_client(model: str) -> OpenAI:
    key = API_KEY if "cloud" in model.lower() else "ollama"
    return OpenAI(base_url=BASE, api_key=key)

def strip_visible_leaks(text: str) -> str:
    if not text: return ""
    t = re.sub(r'\[SPRITE:[^\]]+\]', '', text, flags=re.IGNORECASE).strip()
    lines = [l for l in t.splitlines() if l.strip()]
    starters = ["snow", "fox", "owl", "bear", "wolf", "pine", "serpent", "мартен", "барс", "лис", "сова", "в форме", "сейчас я в форме", "я в форме"]
    while lines and any(k in lines[0].lower() for k in starters):
        lines = lines[1:]
    words = ["snow leopard", "pine marten", "wise owl", "red fox", "serpent queen", "brown bear", "fire phoenix", "снежный барс", "лесная куница", "выдра", "лис", "сова", "медведь", "феникс"]
    for w in words:
        lines = [l.replace(w, "").replace(w.title(), "") for l in lines]
    return "\n".join(lines).strip() or "..."

def get_form(raw: str) -> str:
    res = parse_form_from_response(raw); k = res[0] if res else "pine_marten"
    return k or "pine_marten"

def chat_with_model(client: OpenAI, model: str, messages: list) -> str:
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=TEMP, max_tokens=MAX_TOK, stream=False
    )
    return (resp.choices[0].message.content or "").strip()

def run_one_model(model: str, user_turns: list):
    print(f"\n{'='*70}")
    print(f"=== MODEL: {model}  |  temp={TEMP} max={MAX_TOK} ===")
    print("Direct client + full prompt + forms + LTM (search+extract)")
    print('='*70)

    # fresh memories for this model run (global mem0 extraction LLM is from .env at start)
    print("(clearing LTM for isolated run)")
    try:
        asyncio.run(clear_all_memories())
    except Exception as e:
        print("  clear note:", e)

    client = make_client(model)
    history = []  # short term for prompt
    full_sys = SYSTEM_PROMPT + "\n\n" + get_form_instruction()

    for i, ut in enumerate(user_turns, 1):
        print(f"\n[Женьк {i}] {ut}")

        # mem context (best effort)
        mem_ctx = ""
        try:
            rel = asyncio.run(search_memories(ut, limit=4))
            mem_ctx = format_memories_for_prompt(rel) if rel else ""
            if mem_ctx: print("  [mem ctx used]")
        except Exception as ex:
            print("  [mem search note:", str(ex)[:70], "]")

        msgs = [
            {"role": "system", "content": full_sys},
        ]
        if mem_ctx:
            msgs.append({"role": "system", "content": mem_ctx})
        for h in history[-5:]:
            msgs.append(h)
        msgs.append({"role": "user", "content": ut})

        t0 = time.time()
        try:
            raw = chat_with_model(client, model, msgs)
            dt = time.time() - t0
        except Exception as e:
            print(f"  ERROR: {e}")
            break

        form = get_form(raw)
        vis = strip_visible_leaks(raw)
        sp = get_sprite_path(form)

        print(f"  [Violetta {dt:.1f}s | form={form} | {os.path.basename(sp) if sp else ''}]")
        print(f"  {vis[:580]}{'...' if len(vis)>580 else ''}")

        history.append({"role": "user", "content": ut})
        history.append({"role": "assistant", "content": vis})

        # extract for LTM (uses mem0's configured LLM)
        try:
            added = asyncio.run(extract_and_save_memories([
                {"role":"user", "content": ut},
                {"role":"assistant", "content": vis}
            ]))
            if added: print(f"  [+{added} LTM]")
        except Exception as ex:
            print("  [extract note:", str(ex)[:60], "]")

        time.sleep(0.15)

    # final memories snapshot
    print("\n--- LTM after this model ---")
    try:
        ms = asyncio.run(get_all_memories(8))
        if not ms:
            print("(no LTM stored)")
        for m in ms:
            txt = (m.get("memory") or m.get("text") or str(m))[:140]
            print("  -", txt)
    except Exception as e:
        print("  LTM read err:", e)

    print(f"\n=== END {model} ===\n")
    return history  # for any cross check

if __name__ == "__main__":
    # Same deep test dialog for both (realistic Женьк voice, no motivation fluff, health note, self-saboteur, meaning/fear)
    TURNS = [
        "Привет. Опять эта тяжесть. Хочу запустить проект, но каждый раз останавливаюсь из-за страха, что брошу.",
        "Понимаю, что важно для меня, но стоит только сесть — сразу мысли 'а вдруг это не то' и я выключаюсь.",
        "Иногда думаю: если не сделаю что-то настоящее сейчас, то потеряю смысл вообще. Но и не могу начать по-настоящему.",
        "Помнишь, я говорил про этот большой проект? Каждый раз когда приближаюсь — появляется голос 'да кому это нужно'.",
        "Я устал от того, что всё время в голове. Хочу чтобы было дело, которое я не брошу через две недели.",
        "Когда сажусь — сразу сопротивление. Не лень, а именно внутренний саботаж. Как будто я сам себе враг.",
        "Хочу быть честным здесь. Не ищу мотивацию в стиле 'просто начни'. Хочу понять корень. И да, здоровье тоже в фоне, но не дави на это.",
    ]

    for m in ["gemma4:31b-cloud", "nemotron-3-super:cloud"]:
        run_one_model(m, TURNS)
        # small pause between models
        time.sleep(1)

    print("Comparative test complete. Look at naturalness, honesty, character consistency, memory reflection in responses, and LTM quality.")
