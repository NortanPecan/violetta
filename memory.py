"""
Persistent memory for Violett a using SQLite.

Stores:
- Conversation history (messages)
- Current fluid form + history of forms
- Simple key-value user state

Designed for a single personal user (local daemon).
"""

import sqlite3
import json
import os
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple

DB_PATH = os.getenv("SQLITE_DB_PATH", "violetta_memory.db")

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Create tables if they don't exist."""
    conn = get_conn()
    cur = conn.cursor()

    # One main conversation for the personal daemon (can be extended later)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT DEFAULT 'Разговор с Виолеттой',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            form_description TEXT,           -- only for assistant messages: "Я сейчас в форме ..."
            timestamp TEXT NOT NULL,
            extra TEXT,                      -- JSON for future metadata
            FOREIGN KEY(conversation_id) REFERENCES conversations(id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)

    # Ensure we have at least one conversation
    cur.execute("SELECT id FROM conversations LIMIT 1")
    if not cur.fetchone():
        now = datetime.now(timezone.utc).isoformat()
        cur.execute(
            "INSERT INTO conversations (title, created_at, updated_at) VALUES (?, ?, ?)",
            ("Главный разговор с Виолеттой", now, now)
        )
        conn.commit()

    conn.close()

def get_main_conversation_id() -> int:
    """Return the single main conversation id (create if needed)."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id FROM conversations ORDER BY id LIMIT 1")
    row = cur.fetchone()
    conn.close()
    if row:
        return row["id"]
    # Should never happen after init_db
    raise RuntimeError("No conversation found — run init_db()")

def add_message(
    conversation_id: int,
    role: str,
    content: str,
    form_description: Optional[str] = None,
    extra: Optional[Dict] = None
) -> int:
    """Save a message. Returns new message id."""
    conn = get_conn()
    cur = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()

    cur.execute(
        """
        INSERT INTO messages (conversation_id, role, content, form_description, timestamp, extra)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            conversation_id,
            role,
            content,
            form_description,
            now,
            json.dumps(extra or {}, ensure_ascii=False) if extra else None
        )
    )
    msg_id = cur.lastrowid

    # Update conversation timestamp
    cur.execute(
        "UPDATE conversations SET updated_at = ? WHERE id = ?",
        (now, conversation_id)
    )
    conn.commit()
    conn.close()
    return msg_id

def get_recent_messages(conversation_id: int, limit: int = 30) -> List[Dict]:
    """Return last N messages in chronological order (oldest first for prompt)."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT role, content, form_description, timestamp
        FROM messages
        WHERE conversation_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (conversation_id, limit)
    )
    rows = cur.fetchall()
    conn.close()

    # Reverse to chronological
    messages = []
    for row in reversed(rows):
        msg = {
            "role": row["role"],
            "content": row["content"],
            "timestamp": row["timestamp"]
        }
        if row["form_description"]:
            msg["form_description"] = row["form_description"]
        messages.append(msg)
    return messages

def get_current_form() -> Optional[str]:
    """Return the last known form description (or None)."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT value FROM state WHERE key = 'current_form'")
    row = cur.fetchone()
    conn.close()
    return row["value"] if row else None

def set_current_form(form_description: str):
    """Persist the current form."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO state (key, value) VALUES ('current_form', ?)",
        (form_description,)
    )
    conn.commit()
    conn.close()

    # Also log into form history (simple)
    log_form_change(form_description)

def log_form_change(form_description: str):
    """Keep a lightweight history of forms for reflection (pure Python append for compatibility)."""
    conn = get_conn()
    cur = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()

    cur.execute("SELECT value FROM state WHERE key = 'form_history'")
    row = cur.fetchone()

    history = []
    if row and row["value"]:
        try:
            history = json.loads(row["value"])
            if not isinstance(history, list):
                history = []
        except Exception:
            history = []

    history.append({"form": form_description, "timestamp": now})

    # Keep only last 50 entries
    if len(history) > 50:
        history = history[-50:]

    cur.execute(
        "INSERT OR REPLACE INTO state (key, value) VALUES ('form_history', ?)",
        (json.dumps(history, ensure_ascii=False),)
    )
    conn.commit()
    conn.close()

def get_form_history(limit: int = 15) -> List[Dict]:
    """Return recent form changes."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT value FROM state WHERE key = 'form_history'")
    row = cur.fetchone()
    conn.close()

    if not row:
        return []

    try:
        history = json.loads(row["value"])
        return history[-limit:] if isinstance(history, list) else []
    except Exception:
        return []

def get_all_form_history() -> List[Dict]:
    """Full form history for gallery / export."""
    return get_form_history(limit=1000)

def clear_memory(keep_conversation: bool = True):
    """Dangerous: wipe messages and state (for testing or 'new life')."""
    conn = get_conn()
    cur = conn.cursor()
    if not keep_conversation:
        cur.execute("DELETE FROM conversations")
    cur.execute("DELETE FROM messages")
    cur.execute("DELETE FROM state")
    conn.commit()
    conn.close()
    init_db()
