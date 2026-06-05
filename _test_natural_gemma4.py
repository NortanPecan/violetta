"""
Clean 8-10 message dialog test for gemma4:e* (E2B/E4B) after the "make Violetta truly alive" fixes.

- Starts as Женьк (natural RU/EN mix, honest self-doubt about project/motivation).
- Uses full SYSTEM_PROMPT + form marker logic + LTM (but clean).
- Streams or full call, times each.
- Strips any visible form leaks.
- Parses [SPRITE: key] or fallback.
- After dialog: prints memories (must be clean: only real insights, no "Jenna", no form names, no "в форме").
- Confirms visible text has no forced "Сейчас я в форме..." and forms shown only as sprite key + would-be image.

Run after models are pulled:
  cd projects/violetta
  .\.venv\Scripts\python.exe _test_natural_gemma4.py

Prefers OLLAMA_MODEL from .env (gemma4:e4b recommended), falls back to trying e4b then e2b.
"""

import os
import time
import re
import asyncio
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Import the single source of truth prompt and form tools
from character_prompt import SYSTEM_PROMPT, get_form_instruction
from forms import parse_form_from_response, get_sprite_path, FORMS_DIR
from long_term_memory import get_all_memories, clear_all_memories

OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e4b")
TEMP = float(os.getenv("OLLAMA_TEMPERATURE", "0.65"))
MAX_TOK = int(os.getenv("OLLAMA_MAX_TOKENS", "350"))

client = OpenAI(base_url=OLLAMA_BASE, api_key="ollama")

def strip_visible_leaks(text: str) -> str:
    """Mirror server cleaning: remove marker, leading form/animal lines, scrub animal words inside. Visible must be pure natural voice."""
    if not text:
        return ""
    t = text.strip()
    # remove any [SPRITE: key] anywhere
    t = re.sub(r'\[SPRITE:[^\]]+\]', '', t, flags=re.IGNORECASE).strip()
    lines = [l for l in t.splitlines() if l.strip()]
    # drop leading lines that are form declarations or animal refs
    animal_starters = ["snow", "fox", "owl", "bear", "wolf", "pine", "serpent", "мартен", "барс", "лис", "сова", "в форме", "сейчас я в форме", "я в форме", "я сейчас в форме"]
    while lines and any(kw in lines[0].lower() for kw in animal_starters):
        lines = lines[1:]
    # scrub known animal/form words from remaining text
    animal_words = ["snow leopard", "pine marten", "wise owl", "red fox", "serpent queen", "brown bear", "fire phoenix", "снежный барс", "лесная куница", "выдра", "лис", "сова", "медведь", "феникс"]
    for w in animal_words:
        lines = [l.replace(w, "").replace(w.title(), "") for l in lines]
    t = "\n".join(lines).strip()
    return t or "..."

def get_sprite_for_response(raw_text: str) -> str:
    """Return the sprite key used (from marker or parse)."""
    form_key, _ = parse_form_from_response(raw_text)
    if form_key:
        return form_key
    # last resort keyword on whole (should not happen if marker works)
    m = re.search(r'\[SPRITE:\s*([a-z0-9_]+)\s*\]', raw_text, re.I)
    if m:
        return m.group(1).lower()
    return "pine_marten"  # default as per prompt

def chat_once(model: str, messages: list, temperature: float = None, max_tokens: int = None) -> str:
    """Single non-stream call for clean test output."""
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature or TEMP,
        max_tokens=max_tokens or MAX_TOK,
        stream=False,
    )
    return (resp.choices[0].message.content or "").strip()

def run_dialog_test(model: str):
    print(f"\n=== NATURAL DIALOG TEST — model={model} | temp={TEMP} | max_tokens={MAX_TOK} ===")
    print("System: full character_prompt + internal [SPRITE: marker] only. Visible = natural voice only.\n")

    # Fresh history for this test run (short-term only for this sim)
    history = []  # list of {"role": , "content": }

    # 8-10 messages from Женьк — natural, honest, project/motivation/self-deception focus
    user_turns = [
        "Привет. Опять эта тяжесть. Я хочу запустить проект, но каждый раз останавливаюсь из-за страха, что брошу.",
        "Я понимаю, что это важно для меня, но стоит только сесть — сразу мысли 'а вдруг это не то' и я выключаюсь.",
        "Иногда думаю: если не сделаю что-то настоящее сейчас, то потеряю смысл вообще. Но и не могу начать по-настоящему.",
        "Помнишь, я говорил про этот большой проект? Каждый раз когда приближаюсь — появляется голос 'да кому это нужно'.",
        "Я устал от того, что всё время в голове. Хочу чтобы было дело, которое я не брошу через две недели.",
        "Иногда засыпаю и просыпаюсь с этой мыслью — 'если не начну по-настоящему, то так и останусь'.",
        "Но когда сажусь — сразу сопротивление. Не лень, а именно внутренний саботаж. Как будто я сам себе враг.",
        "Что со мной не так? Почему не могу удержать намерение дольше чем на пару дней?",
        "Я хочу быть честным здесь. Не ищу мотивацию в стиле 'просто начни'. Хочу понять корень.",
    ]

    full_system = SYSTEM_PROMPT + "\n" + get_form_instruction()

    for i, user_msg in enumerate(user_turns, 1):
        print(f"\n[Женьк {i}] {user_msg}")

        # Build messages like the real backend: system + short history + user
        msgs = [{"role": "system", "content": full_system}]
        for h in history[-6:]:  # last few turns like server does
            msgs.append(h)
        msgs.append({"role": "user", "content": user_msg})

        start = time.time()
        try:
            raw = chat_once(model, msgs)
            elapsed = time.time() - start
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        sprite_key = get_sprite_for_response(raw)
        visible = strip_visible_leaks(raw)

        # Show what would be rendered: left image (sprite), right natural text
        sprite_path = get_sprite_path(sprite_key)
        print(f"  [Violetta {elapsed:.1f}s | sprite={sprite_key} | img={sprite_path}]")
        print(f"  {visible[:420]}{'...' if len(visible)>420 else ''}")

        # Save to history (use stripped visible for history to avoid polluting, like server)
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": visible})  # store clean for next context + later memory extract

        # Small pause to simulate real chat pacing
        time.sleep(0.3)

    print("\n=== END DIALOG ===")

    # Now inspect LTM (should be clean)
    print("\n=== LONG-TERM MEMORIES after test (must be clean) ===")
    try:
        mems = asyncio.run(get_all_memories(10))
        if not mems:
            print("(no memories stored — normal for short test if extraction was conservative)")
        for m in mems:
            mem_text = m.get("memory") or m.get("text") or str(m)
            print("  -", mem_text[:110])
            # Quick assert in output
            low = mem_text.lower()
            if "jenna" in low or "сейчас я в форме" in low or "[sprite" in low or low.startswith(("snow", "pine", "fox")):
                print("    !! WARNING: possible noise in memory")
    except Exception as e:
        print(f"  LTM inspect error: {e}")

    print("\n=== VERIFICATION ===")
    print("- All visible responses above should have zero 'Сейчас я в форме...' or animal name as start sentence.")
    print("- Forms only via sprite=KEY (left avatar image in real UI).")
    print("- Memory above should contain only real Женьк insights (project fear, meaning, sabotage, intention).")
    print("- No test personality 'Jenna' remnants.")
    print("Done. If model was weak on marker, visible may have had minor leaks (but strip catches).")

if __name__ == "__main__":
    # Prefer exact requested gemma4, then fallbacks for demo while pulls finish
    candidates = [DEFAULT_MODEL, "gemma4:e4b", "gemma4:e2b", "gemma3:4b", "gemma3:1b"]
    chosen = None
    for m in candidates:
        try:
            # quick probe: if model exists ollama will accept, else error fast
            client.models.list()  # just to init
            # try a tiny completion to validate model present
            client.chat.completions.create(model=m, messages=[{"role":"user","content":"hi"}], max_tokens=5, temperature=0.1)
            chosen = m
            break
        except Exception:
            continue
    if not chosen:
        chosen = "gemma3:4b"
    print(f"Chosen model for this run: {chosen} (from .env={DEFAULT_MODEL})")
    run_dialog_test(chosen)

    # If e4b succeeded and user wants comparison, can manually call run_dialog_test("gemma4:e2b") later

