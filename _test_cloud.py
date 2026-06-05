"""
Direct cloud verification test for current OLLAMA_MODEL (e.g. gemma4:31b-cloud or kimi).

- No local model load (no ollama run, no probe of gemma local tags).
- Uses llm.py get_client / chat_completion (sends correct api_key for cloud).
- Full SYSTEM_PROMPT + internal [SPRITE:] only.
- Memory search before + extract after (tests LTM with cloud LLM for extraction too).
- Realistic Женьк turns on project, self-deception, fear, resistance, health, meaning.
- Prints raw visible responses + sprite for eval.
- After: shows stored memories (should be clean insights only).
"""

import os
import time
import re
import asyncio
from dotenv import load_dotenv

load_dotenv()

from llm import chat_completion, build_messages, get_model_name, get_backend
from character_prompt import SYSTEM_PROMPT, get_form_instruction
from forms import parse_form_from_response, get_sprite_path
from long_term_memory import (
    search_memories,
    extract_and_save_memories,
    get_all_memories,
    clear_all_memories,
    format_memories_for_prompt,
)

def strip_visible_leaks(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    t = re.sub(r'\[SPRITE:[^\]]+\]', '', t, flags=re.IGNORECASE).strip()
    lines = [l for l in t.splitlines() if l.strip()]
    animal_starters = ["snow", "fox", "owl", "bear", "wolf", "pine", "serpent", "мартен", "барс", "лис", "сова", "в форме", "сейчас я в форме", "я в форме", "я сейчас в форме"]
    while lines and any(kw in lines[0].lower() for kw in animal_starters):
        lines = lines[1:]
    animal_words = ["snow leopard", "pine marten", "wise owl", "red fox", "serpent queen", "brown bear", "fire phoenix", "снежный барс", "лесная куница", "выдра", "лис", "сова", "медведь", "феникс"]
    for w in animal_words:
        lines = [l.replace(w, "").replace(w.title(), "") for l in lines]
    t = "\n".join(lines).strip()
    return t or "..."

def get_form_key(raw_text: str) -> str:
    res = parse_form_from_response(raw_text); key = res[0] if res else "pine_marten"
    return key or "pine_marten"

def main():
    model = get_model_name()
    backend = get_backend()
    print(f"\n=== CLOUD TEST — model={model} | backend={backend} ===")
    print("Using full character prompt + [SPRITE: ] marker only. Visible text = natural voice.\n")

    # Fresh for verification (remove prior test noise; real memories would be kept by user normally)
    print("(clearing memories for clean verification run)")
    try:
        asyncio.run(clear_all_memories())
    except Exception as e:
        print("  clear warn:", e)

    # 6-7 realistic Женьк messages (project fear, self-deception voices, resistance as inner enemy, health, honest root, no fake motivation)
    user_turns = [
        "Привет. Опять эта тяжесть. Хочу запустить проект, но каждый раз останавливаюсь из-за страха, что брошу.",
        "Понимаю, что важно, но стоит сесть — сразу 'а вдруг это не то' и выключаюсь.",
        "Если не сделаю что-то настоящее сейчас, потеряю смысл. Но не могу начать по-настоящему.",
        "Помнишь про большой проект? Каждый раз голос 'да кому это нужно' появляется.",
        "Устал от того что всё время в голове. Хочу дело, которое не брошу через две недели.",
        "Когда сажусь — сразу сопротивление. Не лень, а внутренний саботаж. Как будто сам себе враг.",
        "Хочу быть честным. Не ищу 'просто начни'. Хочу понять корень. И да, здоровье тоже в фоне, но не дави.",
    ]

    history = []
    full_sys = SYSTEM_PROMPT + "\n\n" + get_form_instruction()

    for i, user_msg in enumerate(user_turns, 1):
        print(f"\n[Женьк {i}] {user_msg}")

        # memory context (like real server)
        mem_context = ""
        try:
            rel = asyncio.run(search_memories(user_msg, limit=4))
            mem_context = format_memories_for_prompt(rel) if rel else ""
            if mem_context:
                print("  (mem context used)")
        except Exception as ex:
            print("  (mem search:", str(ex)[:80], ")")

        msgs = build_messages(SYSTEM_PROMPT, history[-5:], user_msg, get_form_instruction(), mem_context)

        t0 = time.time()
        try:
            raw = chat_completion(msgs, temperature=0.6, max_tokens=450)
            dt = time.time() - t0
        except Exception as e:
            print(f"  ERROR from cloud: {e}")
            print("  (check if ollama serve running and proxies gemma4:31b-cloud with/without key)")
            break

        form_key = get_form_key(raw)
        visible = strip_visible_leaks(raw)
        sprite_path = get_sprite_path(form_key)

        print(f"  [Violetta {dt:.1f}s | form={form_key} | img={sprite_path}]")
        print(f"  {visible[:520]}{'...' if len(visible) > 520 else ''}")

        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": visible})

        # extract (tests cloud LLM for memory extraction too)
        try:
            added = asyncio.run(extract_and_save_memories([
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": visible}
            ]))
            if added:
                print(f"  (+{added} new memory/memories)")
        except Exception as ex:
            print("  (extract:", str(ex)[:90], ")")

        time.sleep(0.2)

    print("\n=== END DIALOG ===")

    print("\n=== LONG-TERM MEMORIES (post test, should be clean insights only) ===")
    try:
        mems = asyncio.run(get_all_memories(10))
        if not mems:
            print("(no memories stored — possible if extraction conservative or model refused)")
        for m in mems:
            txt = m.get("memory") or m.get("text") or str(m)
            print("  -", txt[:130])
            low = txt.lower()
            if any(x in low for x in ["jenna", "сейчас я в форме", "[sprite", "snow", "pine_marten", "выдра"]):
                print("    !! possible noise")
    except Exception as e:
        print("  LTM error:", e)

    print("\n=== QUICK EVAL (per your 4 criteria) ===")
    print("- Liveliness/natural: look at visible texts above (should feel like direct honest companion, short-ish, RU/EN mix, no templates).")
    print("- Memory: see above list (only real patterns like project sabotage, voices of self-deception, intention to understand root; no form/Jenna noise).")
    print("- Forms: each turn has form=KEY (internal choice via qualities+memories), visible has zero form declarations.")
    print("- Verbosity/template: check lengths and if repetitive phrasing; if too wordy we can lower MAX_TOKENS or temp, or switch model in .env .")
    print("Done. To chat live: use start-violetta.ps1 or uvicorn (ollama serve must stay up for cloud proxy).")

if __name__ == "__main__":
    main()
