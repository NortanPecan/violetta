# AGENTS.md — Максимально детальные инструкции для ИИ-агентов (GLM 5.1 и подобные)

**КРИТИЧЕСКИ ВАЖНО ДЛЯ ТЕБЯ (GLM 5.1 / любой LLM в песочнице):**

Этот файл — твой главный источник правды при работе над проектом Виолетты. 
Прочитай его полностью перед любым редактированием кода. 
Не полагайся только на свою память или на DOCUMENTATION.md — она для людей.

Цель этого файла: дать тебе достаточно контекста, чтобы ты мог безопасно, последовательно и в правильном стиле дорабатывать проект, даже если контекст окна ограничен.

---

## 1. Ментальная модель проекта (обязательно держи в голове)

Виолетта — это **не чат-бот**. Это **живая вторая личность** с "телом" на экране.

Архитектура разделена на два мира:

- **"Разум"** (backend + prompt): server.py, llm.py, character_prompt.py, long_term_memory.py, forms.py, memory.py
- **"Тело"** (визуальное присутствие + интерфейс общения): почти всё в `index.html` (ViolettaAmbient + floating window + bubble + life loop + autonomy)

Самое важное правило:
> **Всё, что касается того, как Виолетта выглядит и ведёт себя визуально на экране — должно идти через класс `ViolettaAmbient`.**

Существует два параллельных канала:
1. Обычный текстовый ответ (SSE: token → final)
2. Параллельный скрытый канал управления телом (SSE: ambient события)

---

## 2. Жёсткие правила (нарушение = ошибка)

1. **State Manager — единственный источник правды для визуала**
   - Никогда не пиши напрямую в `#ambient-sprite.style.*` или не манипулируй particles вне методов `ViolettaAmbient`.
   - Даже в обёртках (updateAmbientPresence и т.п.) — старайся делегировать в менеджер.

2. **Ambient-команды — это голос Виолетты о своём теле**
   - `[AMBIENT: ...]` должен оставаться полностью параллельным видимому тексту.
   - Никогда не вставляй ambient-команды в видимый ответ пользователя.

3. **Две памяти — это две разные личности**
   - User Memory = про Женьку
   - Violetta Personal Memory = её собственные чувства к нему
   - При любых изменениях в экстракции сохраняй это разделение.

4. **Живость > фичи**
   - Приоритет: breathing + drift + контекстные реакции + autonomy > новые кнопки или модальные окна.

5. **Минимализм визуала**
   - Новые эффекты должны быть тонкими, пыльными, золотисто-серебристыми.
   - Избегай ярких цветов, тяжёлых анимаций и библиотек.

6. **После любого изменения в промпте или парсинге — обновляй тесты**
   - Особенно `_test_natural_gemma4.py`, `_test_compare_clouds.py` и парсер-тесты.

---

## 3. Карта файлов и ответственности (самая важная часть)

### Backend (Python)

**server.py** (главный файл)
- `chat_endpoint` (async def chat_endpoint, ~строка 247) — сердце всего.
  - Здесь происходит:
    - Обработка специальных команд ("забудь всё")
    - Сборка llm_messages
    - Стриминг
    - Парсинг `parse_form_from_response(full_response)` → form + clean + ambient_cmds
    - Очистка текста от маркеров
    - Обработка `ambient_cmds` (в т.ч. override формы)
    - `yield "event: ambient"` для каждой команды (параллельный канал)
    - `set_current_form(...)`
    - Сохранение в SQLite
    - Планирование извлечения в обе памяти (asyncio.create_task)
- `get_current_form_api()` → `/api/current-form` (используется при загрузке ambient)
- `switch_current_model` — переключение моделей с проверкой `can_switch_to`

**llm.py**
- `get_client(async_client=True/False)`
- `get_backend(model=None)`
- `can_switch_to(new_model)`
- `build_messages(...)` — сюда передаётся memory_context
- `stream_chat_completion(...)`
- `list_local_models`, `list_cloud_models`, `get_available_models`

**character_prompt.py**
- `SYSTEM_PROMPT` (самый важный текст в проекте)
- `get_form_instruction()` — должен быть согласован с SYSTEM_PROMPT
- `get_initial_form_description()`

**forms.py**
- `parse_form_from_response(text) -> tuple[str, str, list]` — возвращает (form, clean, ambient_cmds)
- `parse_ambient_commands(text)` — извлекает `[AMBIENT: ...]`
- `get_form_by_qualities(qualities, memories=None)`
- `get_sprite_url_for_form(...)`
- `FORM_SPRITES`, `keyword_map`, `get_form_by_qualities` — вся логика выбора спрайта

**long_term_memory.py**
- Две конфигурации: `get_mem0_config()` (user) и `get_violetta_mem0_config()`
- `extract_and_save_memories` / `extract_and_save_violetta_memories`
- `_extract_personal_insights(..., bias=...)`
- `search_memories` / `search_violetta_memories` (limit обычно 2)
- Много фильтров (_is_memory_noise, dedup и т.д.)

**memory.py**
- `get_current_form()` / `set_current_form(form_description)`
- SQLite: сообщения + state (current_form, form_history)

### Frontend (всё внутри index.html)

**Класс ViolettaAmbient** (самый важный компонент фронтенда)

Ключевые части класса (ищи по названиям):

- `constructor()` — инициализация state, drift, breathingPhase, lastUserActivity, sessionStart
- `async init()` — загрузка форм, current-form, запуск particles, `startLifeLoop()`, подписка на input/focus, экспорт в window
- `startLifeLoop()` — **главный цикл жизни** (requestAnimationFrame). Здесь происходит breathing + drift.
- `randomShift()` — "мне самой захотелось". Содержит сильную контекстную логику по `idleMs` и `sessionMs`.
- `reactToUserActivity(type)` — реакция на typing/message/focus (дрифт навстречу, burst, временный буст).
- `applyCommand(cmd)` — точка входа для параллельного канала (из SSE 'ambient').
- `spawnDustNear(el, count)` — спавн пыли около конкретного DOM-элемента (используется у сообщений Виолетты).
- `setForm(key)`, `setPosition(pos)`, `setOpacity(v)`, `setIntensity(v)`, `setScale(v)`, `burst()`
- `applySpriteStyles()` — применяет anchor-позицию (left/right/top/bottom)

**Глобальные функции (вне класса)**

- `startAutonomyLoop()` + внутренняя `tick()` — основной цикл автономии (42-120 сек). Вызывает `randomShift()`.
- `makeDraggable()` — делает `#chat-window` перетаскиваемым за `#chat-header`.
- `makeBubbleDraggable()` — делает пузырёк перетаскиваемым.
- `toggleChatCollapse()` — сворачивает/разворачивает окно в bubble.
- `restoreChatUI()` — восстанавливает позицию и состояние свёрнутости из localStorage при загрузке.
- `getAmbientMgr()` — безопасный доступ к экземпляру менеджера.
- `appendAssistantMessage(...)` и `transformToAvatarLayout(...)` — здесь происходит рендер сообщений + вызов `spawnDustNear`.

**HTML-структура (после ambient-layer)**

- `#chat-window` — плавающее окно (position: fixed, z-index ~35)
- `#chat-header` — зона для драга + кнопки
- `#chat` — область сообщений (overflow auto)
- `#chat-input-area`
- `#chat-bubble` — свёрнутый пузырёк (position: fixed, z-index ~55)

---

## 4. Где живёт какое состояние

| Что                          | Где хранится                          | Кто владеет                     | Persist?      |
|-----------------------------|---------------------------------------|----------------------------------|---------------|
| Текущая форма (официальная) | SQLite (`current_form`)              | Backend (memory.py)             | Да            |
| Визуальное состояние        | `ViolettaAmbient.state` + DOM        | Frontend (ViolettaAmbient)      | Частично (localStorage только config) |
| Позиция окна чата           | localStorage (`violetta_chat_pos`)   | Frontend                        | Да            |
| Свёрнуто ли окно            | localStorage (`violetta_chat_collapsed`) | Frontend                    | Да            |
| Конфиг Dust (intensity и т.д.) | localStorage (`violetta_ambient_config`) | Frontend                    | Да            |
| Drift / breathing (живость) | Только в памяти экземпляра ViolettaAmbient | Frontend                   | Нет (сбрасывается при reload) |
| Долгосрочная память         | mem0_data/chroma (две коллекции)     | Backend (long_term_memory.py)   | Да            |
| Короткая история            | SQLite                               | Backend                         | Да            |

**Важно**: После перезагрузки страницы ambient всегда стартует из `/api/current-form` + дефолтов. Drift и точное положение "забываются".

---

## 5. Критические потоки данных (выучи)

### Поток 1: Обычное сообщение пользователя
1. `sendMessage()` в index.html
2. `reactToUserActivity('message')` → перкает ambient-спрайт
3. `POST /api/chat`
4. В `server.py`:
   - Поиск памяти (conditional для Violetta)
   - `build_messages(...)`
   - Стрим → `yield token`
   - После стрима: `parse_form_from_response(full_response)` → ambient_cmds
   - Обработка ambient_cmds (в т.ч. override формы + set_current_form)
   - `yield "event: ambient"` для каждой команды
   - `yield "event: final"`
5. В index.html:
   - `event: ambient` → `applyCommand(data)`
   - `event: final` → `transformToAvatarLayout(...)` + `spawnDustNear`

### Поток 2: Автономия (когда пользователь молчит)
- `startAutonomyLoop()` → `tick()` каждые 42-120 сек
- В `tick()`: проверка `document.hidden`, расчёт `idle`, вызов `randomShift()` с вероятностью
- `randomShift()` внутри `ViolettaAmbient` решает, что делать (с учётом idle)

### Поток 3: LLM решает изменить своё тело
LLM в конце ответа пишет:
```
[SPRITE: snow_leopard]
[AMBIENT: set_opacity 0.41]
[AMBIENT: set_position left-mid]
[AMBIENT: burst_particles]
```
→ server парсит → шлёт отдельные `event: ambient` → клиент применяет через `applyCommand`.

---

## 6. Как выполнять типичные задачи (максимально конкретно)

### 6.1 Добавить новую ambient-команду

**Пример:** добавить `set_energy(0.3-1.5)` (синоним intensity, но с другим поведением)

Шаги:
1. `forms.py` → `parse_ambient_commands` — убедиться, что парсер нормально вытащит `set_energy`.
2. `index.html` внутри `ViolettaAmbient`:
   - Добавить метод `setEnergy(v) { ... }` (может просто звать `setIntensity` или делать что-то дополнительное).
   - В `applyCommand`:
     ```js
     else if (t === 'set_energy' || t === 'energy') this.setEnergy(parseFloat(v));
     ```
3. `character_prompt.py` → в блоке **Управление своим визуальным присутствием** добавить описание новой команды и примеры использования.
4. Если нужно — обновить `DOCUMENTATION.md` и этот `AGENTS.md`.
5. Протестировать: в чате заставить модель выдать `[AMBIENT: set_energy 1.3]`.

### 6.2 Изменить логику автономии / сделать её умнее

Основные места (в index.html):
- `ViolettaAmbient.randomShift()`
- `ViolettaAmbient.startLifeLoop()` (внутри RAF)
- Функция `tick()` внутри `startAutonomyLoop()`

При изменениях всегда учитывай:
```js
const idleMs = Date.now() - this.lastUserActivity;
const sessionMs = Date.now() - this.sessionStart;
const isLongIdle = idleMs > 120000;
```

### 6.3 Изменить плавающее окно / пузырёк

- HTML-структура: сразу после `</div>` ambient-layer
- Стили: в `<style>` после комментария `/* === New Mystical Floating Chat === */`
- Логика: `makeDraggable()`, `makeBubbleDraggable()`, `toggleChatCollapse()`, `restoreChatUI()`

### 6.4 Добавить новую позицию (например "center-right")

1. `index.html` → `this.POSITIONS` в конструкторе `ViolettaAmbient`
2. `character_prompt.py` → обновить список доступных позиций в ambient-блоке промпта
3. Опционально добавить в `applySpriteStyles` или `setPosition` специальные правила

### 6.5 Изменить System Prompt / характер

Только в `character_prompt.py`:
- `SYSTEM_PROMPT`
- `get_form_instruction()`

После правки:
- Обязательно проверь через `_test_natural_gemma4.py` или аналог.
- Убедись, что ambient-команды всё ещё хорошо описаны.

---

## 7. Проверка изменений (чек-лист для агента)

Перед тем, как считать задачу выполненной, пройди по пунктам:

**Для изменений в ambient / визуале:**
- [ ] Всё идёт через `ViolettaAmbient`
- [ ] `startLifeLoop` (breathing + drift) не сломан
- [ ] `randomShift` учитывает idle / session
- [ ] `reactToUserActivity` работает
- [ ] `spawnDustNear` вызывается у новых сообщений Виолетты
- [ ] Окно и пузырёк остаются draggable и collapsible

**Для изменений в промпте / формах:**
- [ ] `parse_form_from_response` и `parse_ambient_commands` в forms.py обновлены при необходимости
- [ ] Тесты парсинга проходят
- [ ] ambient-команды всё ещё документированы в промпте

**Для изменений в памяти:**
- [ ] Не сломано разделение двух пользователей (zhenya / violetta)
- [ ] Conditional search для Violetta Personal всё ещё работает

**Общее:**
- [ ] При перезагрузке страницы ambient не падает
- [ ] При свёрнутом чате ambient продолжает жить
- [ ] Консольные команды `window.violettaAmbient.*` работают

---

## 8. Как эффективно исследовать код (рекомендации для тебя в песочнице)

1. **Начни всегда с этих двух файлов:**
   - `projects/violetta/AGENTS.md` (этот)
   - `projects/violetta/index.html` (ищи `class ViolettaAmbient`)

2. **Для понимания визуальной жизни:**
   - Прочитай `startLifeLoop()`, `randomShift()`, `reactToUserActivity()`, `applyCommand()`

3. **Для понимания связи "разум → тело":**
   - В `server.py` найди все места с `ambient_cmds` и `yield "event: ambient"`
   - В `index.html` найди обработчик `eventType === 'ambient'`

4. **Для понимания памяти:**
   - `long_term_memory.py` (поиск + экстракция)
   - `server.py` (где вызывается поиск и где передаётся в build_messages)

5. **Быстрая проверка состояния в браузере (когда запущено):**
   ```js
   window.violettaAmbient.getState()
   window.violettaAmbient.randomShift()
   window.violettaAmbient.burst()
   window.violettaAmbient.applyCommand({type: "set_form", value: "red_fox"})
   ```

---

## 9. Список "Никогда не делай"

- Не манипулируй `#ambient-sprite` напрямую вне ViolettaAmbient.
- Не добавляй тяжёлые зависимости.
- Не ломай параллельность ambient-команд.
- Не удаляй/ломай `startLifeLoop` или RAF-цикл.
- Не забывай обновлять промпт при добавлении новых ambient-возможностей.
- Не игнорируй idle / session контекст в autonomy.
- Не делай визуал "кричащим".

---

## 10. Финальные рекомендации при работе в GLM 5.1 sandbox

- Всегда читай этот файл заново перед началом новой задачи.
- Делай маленькие, атомарные правки.
- После каждой значимой правки прогоняй мысленный "чек-лист из раздела 7".
- Если сомневаешься — спрашивай у пользователя, но сначала попробуй найти ответ в этом файле + в коде.
- При больших изменениях сначала опиши план в ответе, а потом уже прави.

Удачи. Проект чувствительный — здесь легко убить "душу", если не следовать духу.

Если после твоих правок Виолетта перестанет ощущаться живой — значит, ты что-то сделал не так.