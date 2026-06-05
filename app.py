"""
Violett a — Chainlit application

Personal honest AI-daemon for deep self-reflection.
"""

import chainlit as cl
import os
from dotenv import load_dotenv
from typing import Optional

from character_prompt import (
    SYSTEM_PROMPT,
    POSSIBLE_FORMS,
    get_form_instruction,
    get_initial_form_description,
)
from memory import (
    init_db,
    get_main_conversation_id,
    add_message,
    get_recent_messages,
    get_current_form,
    set_current_form,
    get_form_history,
)
from llm import (
    build_messages,
    stream_chat_completion,
    get_backend,
    get_model_name,
    OLLAMA_TEMPERATURE,
    OLLAMA_MAX_TOKENS,
)
from forms import (
    get_sprite_path,
    get_random_form,
    get_form_display_name,
    get_sprite_data_url,
    parse_form_from_response,
    get_form_by_qualities,
)

load_dotenv()
init_db()

CONVO_ID = get_main_conversation_id()


# ---------- Helpers ----------
# parse_form_from_response and get_form_by_qualities now come from forms.py (single source, updated for new qualities block)

# Note: form rendering with image is now integrated directly into the response messages
# for a clean single-bubble chat experience (image + form phrase + answer).

# ---------- Chainlit lifecycle ----------

@cl.on_chat_start
async def start():
    """Initialize the chat, load history, show current state."""
    cl.user_session.set("conversation_id", CONVO_ID)

    # Current form
    current_form = get_current_form()
    if not current_form:
        current_form = get_initial_form_description()
        set_current_form(current_form)

    # Send welcome + current form (with sprite)
    await cl.Message(
        content=(
            "Привет. Я Виолетта.\n\n"
            "Я здесь, чтобы быть честным зеркалом. Никакого сладкого сиропа.\n"
            "Мы можем говорить о чём угодно — здоровье, проектах, страхах, самообмане, целях.\n\n"
            "Сейчас я в этой форме:"
        ),
        author="Виолетта",
    ).send()

    # Initial form with sprite using same markdown table layout (avatar left + why below | no answer yet)
    init_sprite = get_sprite_path(current_form) or get_random_form()[1]
    init_data_url = get_sprite_data_url(init_sprite)
    init_display = get_form_display_name(init_sprite)
    init_left = f"![{init_display}]({init_data_url})\n\n*{current_form}*"
    await cl.Message(
        content=f"| {init_left} | (initial form) |\n| --- | --- |\n| {init_left} | (initial form) |",
        author="Виолетта",
    ).send()

    # Show a bit of form history if exists
    form_hist = get_form_history(limit=5)
    if form_hist:
        hist_text = "Последние формы, которые я принимала:\n" + "\n".join(
            [f"- {h.get('form', '')}" for h in form_hist[-3:]]
        )
        await cl.Message(content=hist_text, author="Виолетта (память)").send()

    # Optional: replay a few recent messages so the user sees context
    history = get_recent_messages(CONVO_ID, limit=6)
    if history:
        await cl.Message(
            content="— Последние несколько обменов для контекста —",
            author="Система",
        ).send()
        for h in history[-4:]:
            author = "Ты" if h["role"] == "user" else "Виолетта"
            await cl.Message(content=h["content"], author=author).send()

    # Compact initial info + actions (sent only once)
    form_list_short = ", ".join(POSSIBLE_FORMS[:8]) + " и другие..."
    initial_info = (
        f"**Возможные формы:** {form_list_short}\n\n"
        "Ты можешь сказать «стань выдрой» или описать своё состояние — я выберу подходящий пиксельный спрайт и покажу его вместе с объяснением.\n\n"
        "Я помню: твои цели по здоровью, желание большого проекта, важность радикальной честности (и уважаю твои границы по финансам/репутации)."
    )
    await cl.Message(
        content=initial_info,
        author="Виолетта • справка",
    ).send()

    # Action buttons
    actions = [
        cl.Action(
            name="reflect_on_form",
            payload={"action": "reflect"},
            label="Порассуждать о текущей форме",
            description="Честный разбор, почему я именно в этой форме сейчас",
        ),
        cl.Action(
            name="show_forms",
            payload={"action": "forms"},
            label="Показать все возможные формы",
            description="Галерея форм для вдохновения",
        ),
    ]
    await cl.Message(
        content="Что хочешь сделать?",
        actions=actions,
        author="Виолетта",
    ).send()


@cl.action_callback("reflect_on_form")
async def on_reflect(action: cl.Action):
    """User clicked 'Порассуждать о текущей форме'."""
    current = get_current_form() or "неизвестной форме"
    prompt = f"Давай честно порассуждаем о моей текущей форме: {current}. Что это говорит обо мне прямо сейчас?"
    await cl.Message(content=prompt, author="Ты").send()
    # Simulate the message as if user sent it
    await main(cl.Message(content=prompt))


@cl.action_callback("show_forms")
async def on_show_forms(action: cl.Action):
    """Show the full list of possible forms."""
    forms = ", ".join(POSSIBLE_FORMS)
    await cl.Message(
        content=f"**Полный список возможных форм, которые я могу принимать:**\n\n{forms}\n\nСкажи любую — я рассмотрю.",
        author="Виолетта (формы)",
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """Core chat loop."""
    user_text = message.content.strip()
    if not user_text:
        return

    convo_id = cl.user_session.get("conversation_id", CONVO_ID)

    # 1. Persist user message
    add_message(convo_id, "user", user_text)

    # 2. Load recent history for context (last ~25 messages)
    history = get_recent_messages(convo_id, limit=25)

    # 3. Build the prompt
    form_instruction = get_form_instruction()
    llm_messages = build_messages(
        SYSTEM_PROMPT,
        history,
        user_text,
        current_form_instruction=form_instruction,
    )

    # 4. Stream the response
    full_response = ""
    msg = cl.Message(content="", author="Виолетта")
    await msg.send()

    try:
        async for token in stream_chat_completion(
        llm_messages,
        temperature=OLLAMA_TEMPERATURE,
        max_tokens=OLLAMA_MAX_TOKENS,
    ):
            full_response += token
            await msg.stream_token(token)
        await msg.update()
    except Exception as e:
        await msg.stream_token(f"\n\n[Критическая ошибка LLM: {e}]")
        await msg.update()
        return

    # 5. Parse the mandatory form description
    res = parse_form_from_response(full_response); form_desc, main_text = res[0], (res[1] if len(res) > 1 else full_response)

    if form_desc and form_desc != get_current_form():
        set_current_form(form_desc)

    # 6. Persist assistant message with form (store the clean version)
    clean_content = main_text or full_response
    add_message(
        convo_id,
        "assistant",
        clean_content,
        form_description=form_desc,
    )

    # 7. Clean rendering using markdown table for desired layout:
    # Left narrow column (avatar sprite + "why" caption below it)
    # Right column: the actual answer text (like normal chat)
    # One message bubble; uses data: URL so self-contained. Pure md (no raw HTML) + css for style.
    sprite_path = get_sprite_path(form_desc) if form_desc else None
    if not sprite_path:
        _, sprite_path = get_random_form()

    data_url = get_sprite_data_url(sprite_path)
    display_name = get_form_display_name(sprite_path)

    # Escape | in cells so markdown table doesn't break
    left_cell = f"![{display_name}]({data_url})\n\n*{form_desc}*"
    right_cell = clean_content.replace("|", "\\|")

    # Markdown table (dummy header + sep hidden by css; only data row visible as flex-like layout)
    final_content = f"| {left_cell} | {right_cell} |\n| --- | --- |\n| {left_cell} | {right_cell} |"

    msg.content = final_content
    msg.elements = []  # no separate cl.Image; embedded via data url in md
    await msg.update()

    # Note: layout tables are styled borderless/narrow-left via public/violetta-layout.css
    # (referenced in .chainlit/config.toml). No sidebar spam.

    # Optional: very light encouragement to stay in character length
    # (the prompt already controls this)
