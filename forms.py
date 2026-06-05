"""
forms.py - Pixel sprite management for Violett a fluid forms.

This module makes the form visualization system extensible:
- Add new .jpg sprites to the Forms/ folder.
- Add an entry to FORM_SPRITES with a descriptive snake_case key.
- The matching logic will try to associate LLM-generated form descriptions
  (e.g. "Я сейчас в форме снежный барс...") to the best sprite.

The LLM in character_prompt.py still decides the narrative form.
This module only provides the visual representation.
"""

import base64
import os
import random
from functools import lru_cache
from pathlib import Path
from typing import Tuple, Optional

from character_prompt import POSSIBLE_FORMS  # needed for parse fallback scan of known forms

# Base directory for the project (so paths work when running from anywhere)
BASE_DIR = Path(__file__).parent
FORMS_DIR = BASE_DIR / "Forms"

# Main dictionary: snake_case key -> absolute path to sprite
# Keys should be descriptive and match (partially) the names in POSSIBLE_FORMS
# or keywords that appear in the LLM's form description.
FORM_SPRITES: dict[str, str] = {
    # Cute / playful
    "moon_bunny": str(FORMS_DIR / "6HV10.jpg"),           # pink bunny girl
    "blue_cat": str(FORMS_DIR / "eb7Mi.jpg"),             # happy blue cat
    "red_fox": str(FORMS_DIR / "bxkev.jpg"),              # red fox
    "pine_marten": str(FORMS_DIR / "or4u5.jpg"),          # лесная куница / marten / proxy for выдра (otter)
    "shadow_cat": str(FORMS_DIR / "zATaf.jpg"),           # black cat on moon

    # Majestic / wise / night
    "wise_owl": str(FORMS_DIR / "Mf0kp.jpg"),             # owl with halo
    "snow_leopard": str(FORMS_DIR / "Whyzf.jpg"),         # snow leopard
    "lightning_wolf": str(FORMS_DIR / "c1Cnw.jpg"),       # gray wolf with lightning

    # Strong / grounded
    "brown_bear": str(FORMS_DIR / "8DFCj.jpg"),           # brown bear

    # Mystical / elemental
    "winter_deer": str(FORMS_DIR / "u7hrD.jpg"),          # white deer spirit
    "thunder_eagle": str(FORMS_DIR / "D0F9D.jpg"),        # blue eagle
    "star_fairy": str(FORMS_DIR / "qCswC.jpg"),           # fairy with butterfly wings

    # Intense / emotional
    "blue_flame_succubus": str(FORMS_DIR / "7tlQb.jpg"),  # blue flame horned girl
    "fire_phoenix": str(FORMS_DIR / "DOlZB.jpg"),         # fire winged girl
    "serpent_queen": str(FORMS_DIR / "9fT1o.jpg"),        # girl with water snake
}

# Precompute base64 data URLs for all sprites at import time (small images, efficient for markdown embedding)
SPRITE_DATA_URLS: dict[str, str] = {}
for _key, _path in FORM_SPRITES.items():
    with open(_path, "rb") as _f:
        _b64 = base64.b64encode(_f.read()).decode("ascii")
    SPRITE_DATA_URLS[_key] = f"data:image/jpeg;base64,{_b64}"

def get_sprite_data_url(sprite_path: str) -> str:
    """Return data: URL for the sprite (for self-contained markdown <img src=...> in chat layout)."""
    for _key, _p in FORM_SPRITES.items():
        if _p == sprite_path:
            return SPRITE_DATA_URLS[_key]
    # fallback (should not happen)
    with open(sprite_path, "rb") as _f:
        _b64 = base64.b64encode(_f.read()).decode("ascii")
    return f"data:image/jpeg;base64,{_b64}"

def _normalize(text: str) -> str:
    """Lowercase and normalize for matching."""
    return text.lower().replace(" ", "_").replace("-", "_").replace("я сейчас в форме", "")

def get_sprite_path(form_description: str) -> Optional[str]:
    """
    Given the LLM's form sentence (e.g. "Я сейчас в форме снежный барс, потому что..."),
    return the best matching sprite path or None.
    """
    if not form_description:
        return None

    norm = _normalize(form_description)

    # Direct key match
    for key in FORM_SPRITES:
        if key in norm or norm in key:
            return FORM_SPRITES[key]

    # Keyword based matching (easy to extend)
    keyword_map = {
        # rabbit / hedgehog / cautious
        "bunny": "moon_bunny",
        "rabbit": "moon_bunny",
        "зай": "moon_bunny",
        "кролик": "moon_bunny",
        "hedgehog": "moon_bunny",
        "ёж": "moon_bunny",
        "еж": "moon_bunny",

        # bear / powerful protective steady
        "bear": "brown_bear",
        "медведь": "brown_bear",
        "медв": "brown_bear",

        # fox / sly sarcastic tricky clever
        "fox": "red_fox",
        "лис": "red_fox",
        "лиса": "red_fox",
        "sarcastic": "red_fox",
        "сарказм": "red_fox",
        "саркастич": "red_fox",
        "tricky": "red_fox",
        "хитр": "red_fox",
        "clever": "red_fox",
        "black_cat": "shadow_cat",
        "чёрная кошка": "shadow_cat",
        "черная кошка": "shadow_cat",

        # wolf
        "wolf": "lightning_wolf",
        "волк": "lightning_wolf",

        # eagle / free inspired transformative
        "eagle": "thunder_eagle",
        "орёл": "thunder_eagle",
        "орел": "thunder_eagle",
        "inspired": "thunder_eagle",
        "вдохнов": "thunder_eagle",

        # owl / wise calm reflective thoughtful
        "owl": "wise_owl",
        "сова": "wise_owl",
        "wise": "wise_owl",
        "calm": "wise_owl",
        "спокойн": "wise_owl",
        "thoughtful": "wise_owl",
        "задумчив": "wise_owl",
        "reflective": "wise_owl",

        # cat variants
        "cat": "blue_cat",
        "кошка": "blue_cat",
        "кот": "blue_cat",

        # snow_leopard / pride confident strong independent
        "leopard": "snow_leopard",
        "леопард": "snow_leopard",
        "леопарда": "snow_leopard",
        "барс": "snow_leopard",
        "snow": "snow_leopard",
        "snow_leopard": "snow_leopard",
        "pride": "snow_leopard",
        "горд": "snow_leopard",
        "confident": "snow_leopard",
        "уверен": "snow_leopard",
        "strong": "snow_leopard",
        "сильн": "snow_leopard",
        "independent": "snow_leopard",
        "независим": "snow_leopard",
        "determin": "snow_leopard",
        "решительн": "snow_leopard",

        # pine_marten / otter / curiosity playful energetic joyful happy
        "marten": "pine_marten",
        "куница": "pine_marten",
        "выдра": "pine_marten",
        "otter": "pine_marten",
        "curios": "pine_marten",
        "любопыт": "pine_marten",
        "playful": "pine_marten",
        "игрив": "pine_marten",
        "energetic": "pine_marten",
        "энергичн": "pine_marten",
        "joy": "pine_marten",
        "радост": "pine_marten",
        "happy": "pine_marten",
        "весел": "pine_marten",

        # deer
        "deer": "winter_deer",
        "олень": "winter_deer",

        # fairy
        "fairy": "star_fairy",
        "фея": "star_fairy",
        "пикси": "star_fairy",
        "butterfly": "star_fairy",

        # succubus / demon / intense
        "demon": "blue_flame_succubus",
        "демон": "blue_flame_succubus",
        "succubus": "blue_flame_succubus",
        "tense": "blue_flame_succubus",

        # phoenix / fire / transformative
        "fire": "fire_phoenix",
        "огонь": "fire_phoenix",
        "phoenix": "fire_phoenix",
        "transform": "fire_phoenix",
        "свобод": "fire_phoenix",
        "дракон": "fire_phoenix",
        "dragon": "fire_phoenix",

        # snake / serpent / tense skeptical doubt defensive anxious
        "snake": "serpent_queen",
        "змея": "serpent_queen",
        "серпент": "serpent_queen",
        "skeptic": "serpent_queen",
        "скепт": "serpent_queen",
        "doubt": "serpent_queen",
        "сомн": "serpent_queen",
        "defensive": "serpent_queen",
        "защитн": "serpent_queen",
        "anxious": "serpent_queen",
        "тревож": "serpent_queen",
        "cautious": "serpent_queen",
        "напряж": "serpent_queen",
        "паук": "serpent_queen",
        "spider": "serpent_queen",
        "spid": "serpent_queen",
    }

    for keyword, key in keyword_map.items():
        if keyword in norm:
            return FORM_SPRITES[key]

    return None

def get_random_form() -> tuple[str, str]:
    """Return (key, full_path) for a random sprite. Useful for variety or fallback."""
    key = random.choice(list(FORM_SPRITES.keys()))
    return key, FORM_SPRITES[key]

def get_form_by_qualities(qualities: list[str], memories: Optional[list[str]] = None) -> tuple[str, str]:
    """
    Quality-driven sprite selector matching the exact **Примеры маппинга качеств** from the System Prompt.
    Pass 2-3 key emotions/qualities extracted from user message + context.
    If memories (long-term) are provided, they can bias the emotional tone (e.g. long-term anxiety even on a "happy" day).
    Always falls back gracefully (pine_marten default bias, then random).
    """
    if not qualities:
        qualities = []

    qs = [q.lower().strip() for q in qualities if q]

    # Incorporate memory context (simple but effective keyword bias from memory texts)
    if memories:
        mem_text = " ".join(memories).lower()
        if any(w in mem_text for w in ["тревог", "страх", "сомн", "anxious", "doubt", "напряж", "защит"]):
            qs.append("anxious")
        if any(w in mem_text for w in ["горд", "достиг", "прогресс", "решил", "pride", "confident"]):
            qs.append("pride")
        if any(w in mem_text for w in ["цель", "проект", "хочу большой", "мечта", "goal"]):
            qs.append("determined")
        if any(w in mem_text for w in ["устал", "выгорел", "трудно", "tired", "exhausted"]):
            qs.append("tired")

    # Direct match priority per the new prompt block (order matters for overlap)
    # pride, confident, strong, independent → snow_leopard
    pride_keys = {"pride", "confident", "strong", "independent", "горд", "уверен", "сильн", "независим", "determination", "решительн", "гордость"}
    if any(k in qs for k in pride_keys) or any(any(pk in q for pk in pride_keys) for q in qs):
        return "snow_leopard", FORM_SPRITES["snow_leopard"]

    # curiosity, playful, energetic / joyful, happy, energetic → pine_marten (or otter alias)
    playful_keys = {"curiosity", "curious", "playful", "energetic", "joyful", "happy", "joy", "любопыт", "игрив", "энергичн", "радост", "весел", "curios"}
    if any(k in qs for k in playful_keys) or any(any(pk in q for pk in playful_keys) for q in qs):
        return "pine_marten", FORM_SPRITES["pine_marten"]

    # wise, calm, reflective, thoughtful → owl
    wise_keys = {"wise", "calm", "reflective", "thoughtful", "мудр", "спокойн", "задумчив", "рефлексивн", "thought"}
    if any(k in qs for k in wise_keys) or any(any(pk in q for pk in wise_keys) for q in qs):
        return "wise_owl", FORM_SPRITES["wise_owl"]

    # sly, sarcastic, tricky, clever → fox
    sly_keys = {"sly", "sarcastic", "sarcasm", "tricky", "clever", "хитр", "сарказм", "саркастич", "умн"}
    if any(k in qs for k in sly_keys) or any(any(pk in q for pk in sly_keys) for q in qs):
        return "red_fox", FORM_SPRITES["red_fox"]

    # tense, skeptical, doubt, defensive / anxious, cautious, protective → snake (serpent_queen)
    tense_keys = {"tense", "skeptical", "skeptic", "doubt", "defensive", "anxious", "cautious", "protective", "напряж", "скепт", "сомн", "защитн", "тревож", "осторожн"}
    if any(k in qs for k in tense_keys) or any(any(pk in q for pk in tense_keys) for q in qs):
        return "serpent_queen", FORM_SPRITES["serpent_queen"]

    # powerful, protective, steady → bear or wolf (prefer bear for steady/protective)
    power_keys = {"powerful", "protective", "steady", "protect", "защит", "устойчив", "мощн"}
    if any(k in qs for k in power_keys) or any(any(pk in q for pk in power_keys) for q in qs):
        return "brown_bear", FORM_SPRITES["brown_bear"]

    # free, inspired, transformative → phoenix or eagle
    free_keys = {"free", "inspired", "transformative", "transform", "вдохнов", "свобод", "трансформ"}
    if any(k in qs for k in free_keys) or any(any(pk in q for pk in free_keys) for q in qs):
        return "fire_phoenix", FORM_SPRITES["fire_phoenix"]

    # anxious / cautious / protective already caught above in tense, but rabbit/hedgehog bias for pure anxious
    rabbit_keys = {"anxious", "rabbit", "hedgehog", "тревожн", "осторож", "кролик", "ёж"}
    if any(k in qs for k in rabbit_keys):
        return "moon_bunny", FORM_SPRITES["moon_bunny"]

    # Default bias to pine_marten (as per prompt: "Базовая форма по умолчанию: pine_marten")
    if "pine_marten" in FORM_SPRITES:
        return "pine_marten", FORM_SPRITES["pine_marten"]

    return get_random_form()

# Convenience: get display name from path (for UI)
def get_form_display_name(sprite_path: str) -> str:
    """Return a nice name from the filename or key."""
    for key, path in FORM_SPRITES.items():
        if path == sprite_path:
            return key.replace("_", " ").title()
    return Path(sprite_path).stem

def get_sprite_url_for_form(form_description: str) -> str:
    """Return the web URL (for /forms/ mount) for the sprite matching the description, or random."""
    path = get_sprite_path(form_description) or get_random_form()[1]
    fname = os.path.basename(path)
    return f"/forms/{fname}"

def get_all_form_sprites() -> list:
    """For UI list of possible forms."""
    items = []
    for key, path in FORM_SPRITES.items():
        fname = os.path.basename(path)
        items.append({
            "key": key,
            "url": f"/forms/{fname}",
            "name": key.replace("_", " ").title()
        })
    return items


def parse_ambient_commands(text: str) -> list[dict]:
    """
    Extract hidden [AMBIENT: ...] meta-commands for parallel control of visual presence.
    Supports:
      [AMBIENT: set_form red_fox]
      [AMBIENT: set_opacity 0.35]
      [AMBIENT: set_position left-mid]
      [AMBIENT: burst_particles]
      [AMBIENT: random_change]
      [AMBIENT: set_intensity 0.9]
    Returns list of {"type": "set_form", "value": "red_fox"} etc.
    Safe, multiple allowed, case-insensitive.
    """
    if not text:
        return []
    import re
    cmds = []
    for m in re.finditer(r'\[AMBIENT:\s*([^\]]+?)\s*\]', text, re.IGNORECASE):
        inner = m.group(1).strip()
        if not inner:
            continue
        # Normalize separators: "set_form=red_fox" or "set_opacity 0.4" etc.
        parts = re.split(r'[\s=]+', inner)
        cmd = parts[0].strip().lower().replace('-', '_')
        args = [p.strip() for p in parts[1:] if p.strip()]
        if not cmd:
            continue
        if len(args) == 0:
            val = True
        elif len(args) == 1:
            val = args[0]
        else:
            val = args
        cmds.append({"type": cmd, "value": val})
    return cmds


def get_form_for_context(current_text: str, memory_texts: Optional[list[str]] = None) -> tuple[str, str]:
    """
    High-level helper used by server: combines current user message with long-term memory
    to decide the best sprite. First tries to extract simple qualities, then calls
    the quality selector (which now also understands memories).
    """
    qualities = []
    text_lower = (current_text or "").lower()

    # Very lightweight extraction (LLM does the heavy lifting in prompt anyway)
    if any(w in text_lower for w in ["горд", "решил", "достиг", "получилось", "успех", "pride", "finally"]):
        qualities.append("pride")
    if any(w in text_lower for w in ["любопыт", "интерес", "хочу узнать", "play", "curious", "explore"]):
        qualities.append("curiosity")
    if any(w in text_lower for w in ["тревог", "страшно", "сомнева", "не уверен", "напряж", "защища", "anxious", "doubt"]):
        qualities.append("anxious")
    if any(w in text_lower for w in ["устал", "выгорел", "тяжело", "трудно", "tired"]):
        qualities.append("tired")

    # memories will further bias inside get_form_by_qualities
    return get_form_by_qualities(qualities, memories=memory_texts)

# --- parse moved here so server and other modules can reuse without depending on the old Chainlit app.py ---
def parse_form_from_response(text: str) -> tuple[str, str, list]:
    """Extract sprite key via [SPRITE: key] marker (preferred new way) or fallback to old first-sentence logic.
    Also extracts parallel [AMBIENT: ...] hidden meta-commands for the visual body.
    Returns (sprite_key_or_desc, clean_text, ambient_commands_list).
    Ambient commands are stripped from visible text.
    """
    text = text.strip()
    if not text:
        return "pine_marten", text, []

    import re
    ambient_cmds = parse_ambient_commands(text)

    # New preferred way: [SPRITE: key] at the end
    marker = re.search(r'\[SPRITE:\s*([a-zA-Z_]+)\s*\]\s*$', text, re.IGNORECASE)
    if marker:
        sprite_key = marker.group(1).lower().strip()
        clean = re.sub(r'\s*\[SPRITE:[^\]]+\]\s*$', '', text, flags=re.IGNORECASE).strip()
        clean = re.sub(r'\[AMBIENT:[^\]]+\]', '', clean, flags=re.IGNORECASE).strip()
        # If after removing marker the text is empty, keep original without marker
        if not clean:
            clean = text
        return sprite_key, clean, ambient_cmds

    # Fallback for old conversations or if model forgets marker: old first-line logic
    lines = text.split("\n", 1)
    first_line = lines[0].strip()

    form_markers = [
        "я сейчас в форме", "сейчас я в форме", "я в форме",
        "форма:", "*я сейчас", "я принимаю форму",
        "чувствую твою", "— чувствую твою", "— чувствую"
    ]

    lower_first = first_line.lower()
    looks_like_form = (
        any(marker in lower_first for marker in form_markers)
        or first_line.startswith(("*", "Я сейчас", "Сейчас я", "Сейчас я в форме"))
        or "— чувствую твою" in lower_first
        or "чувствую твою" in lower_first
    )

    if looks_like_form:
        form_desc = first_line.strip("* ").strip()
        rest = lines[1].strip() if len(lines) > 1 else ""
        if not rest or len(rest) < 15:
            rest = text
        rest = re.sub(r'\[AMBIENT:[^\]]+\]', '', rest, flags=re.IGNORECASE).strip()
        return form_desc, rest, ambient_cmds

    # Fallback scan for known forms
    beginning = text[:250].lower()
    for form in POSSIBLE_FORMS:
        if form.lower() in beginning:
            return form, text, ambient_cmds

    for key in FORM_SPRITES:
        if key in beginning or key.replace("_", " ") in beginning:
            return key, text, ambient_cmds

    return "pine_marten", text, ambient_cmds  # safe default so sprite always shows something
