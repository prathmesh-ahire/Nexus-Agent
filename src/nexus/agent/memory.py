"""
memory.py — NEXUS Memory System
Contains both:
  1. ConversationMemory — in-memory session history (resets on restart)
  2. Persistent Memory — long-term fact storage in JSON on disk (survives restarts)

Originally split across memory.py (Phase 19) and memory.py (Phase 28),
now unified into a single module for cleaner organisation.
"""

import contextlib
import json
import os
import shutil
from datetime import datetime

from nexus import config

# ===========================================================================
# Part 1: Conversation Memory (session-only, in-memory)
# ===========================================================================

class ConversationMemory:
    """
    In-memory conversation history for a single NEXUS session.
    Stores user and assistant messages as a list of dicts and
    provides helpers to retrieve, format, and clear the history.
    """

    def __init__(self):
        self._history = []  # list of {"role": str, "content": str}

    # ------------------------------------------------------------------
    # Add a message to history
    # ------------------------------------------------------------------
    def add_message(self, role, content):
        """
        Append a message to the conversation history.

        Args:
            role:    "user" or "assistant"
            content: The message text.
        """
        if not content or not content.strip():
            return  # don't store empty messages

        self._history.append({
            "role": role,
            "content": content.strip(),
        })

    # ------------------------------------------------------------------
    # Retrieve recent history
    # ------------------------------------------------------------------
    def get_history(self, max_turns=5):
        """
        Return the last N *exchanges* (user + assistant pairs) from history.
        An exchange = one user message + one assistant response = 2 entries.

        Args:
            max_turns: Maximum number of exchanges to return (default 5).

        Returns:
            List of {"role": ..., "content": ...} dicts, in chronological order.
        """
        # Each "turn" is a user+assistant pair = 2 messages
        max_messages = max_turns * 2
        if len(self._history) <= max_messages:
            return list(self._history)
        return list(self._history[-max_messages:])

    # ------------------------------------------------------------------
    # Clear history
    # ------------------------------------------------------------------
    def clear(self):
        """Reset conversation history — start a fresh session."""
        self._history.clear()

    # ------------------------------------------------------------------
    # Format history for the model prompt
    # ------------------------------------------------------------------
    def get_context_string(self, max_turns=5):
        """
        Build a plain-text context string from recent history, suitable
        for prepending to a model prompt. Uses a simple format:

            User: ...
            NEXUS: ...
            User: ...
            NEXUS: ...

        Args:
            max_turns: Maximum number of exchanges to include.

        Returns:
            A formatted string of recent conversation, or empty string
            if there is no history.
        """
        recent = self.get_history(max_turns)
        if not recent:
            return ""

        lines = []
        for msg in recent:
            if msg["role"] == "user":
                lines.append(f"User: {msg['content']}")
            elif msg["role"] == "assistant":
                lines.append(f"NEXUS: {msg['content']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Build messages list for chat completion API
    # ------------------------------------------------------------------
    def get_messages_for_model(self, max_turns=5):
        """
        Return the history as a list of message dicts ready for the
        model's create_chat_completion() messages parameter.
        Does NOT include the system prompt — the caller should prepend that.

        Args:
            max_turns: Maximum number of exchanges to include.

        Returns:
            List of {"role": "user"|"assistant", "content": ...} dicts.
        """
        return self.get_history(max_turns)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def length(self):
        """Number of individual messages stored."""
        return len(self._history)

    @property
    def is_empty(self):
        """True if no messages have been stored yet."""
        return len(self._history) == 0

    def __len__(self):
        return len(self._history)

    def __repr__(self):
        return f"ConversationMemory({len(self._history)} messages)"


# ===========================================================================
# Part 2: Persistent Memory (long-term, stored on disk as JSON)
# ===========================================================================

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Resolve path relative to the project root (nexus-agent/config/)
MEMORY_FILE = str(config.MEMORY_FILE)


# ---------------------------------------------------------------------------
# Load / Save helpers
# ---------------------------------------------------------------------------
def load_memory():
    """
    Read user_memory.json and return its contents as a dict.
    Returns an empty dict with a 'facts' key if the file is missing
    or corrupted.
    """
    if not os.path.isfile(MEMORY_FILE):
        return {"facts": []}

    try:
        with open(MEMORY_FILE, encoding="utf-8") as f:
            data = json.load(f)
        # Ensure the expected structure exists
        if not isinstance(data, dict) or "facts" not in data:
            return {"facts": []}
        return data
    except (json.JSONDecodeError, OSError, ValueError):
        return {"facts": []}


def save_memory(data):
    """
    Write the memory dict to user_memory.json.

    Writes to a temp file first and atomically renames it into place
    (os.replace), so a crash or power loss mid-write can't leave the
    memory file half-written. Keeps one rotating .bak of the previous
    version before each overwrite.

    Args:
        data: Dict with at least a 'facts' key containing a list.
    """
    try:
        # Ensure the config directory exists
        os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)

        tmp_path = MEMORY_FILE + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        if os.path.isfile(MEMORY_FILE):
            with contextlib.suppress(OSError):
                shutil.copy2(MEMORY_FILE, MEMORY_FILE + ".bak")

        os.replace(tmp_path, MEMORY_FILE)
    except OSError as e:
        print(f"Warning: Could not save memory file: {e}")


# ---------------------------------------------------------------------------
# Store and recall facts
# ---------------------------------------------------------------------------
def remember_fact(fact_text):
    """
    Store a user fact with a timestamp.

    Args:
        fact_text: The fact string to remember.

    Returns:
        Confirmation message string.
    """
    if not fact_text or not fact_text.strip():
        return "Please provide a fact to remember. Example: /remember my favourite movie is Avengers Endgame"

    fact_text = fact_text.strip()
    data = load_memory()

    # Check for duplicate facts (case-insensitive)
    for existing in data["facts"]:
        if existing["text"].lower() == fact_text.lower():
            return f"I already know that: \"{fact_text}\""

    # Add the new fact
    data["facts"].append({
        "text": fact_text,
        "stored_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })

    save_memory(data)
    return f"Got it! I'll remember: \"{fact_text}\""


def recall_facts(query=None):
    """
    Return all stored facts, optionally filtered by a keyword query.

    Args:
        query: Optional keyword string to filter facts.
               If None or empty, returns all facts.

    Returns:
        Formatted string listing the stored facts.
    """
    data = load_memory()
    facts = data.get("facts", [])

    if not facts:
        return "No memories stored yet. Use /remember to store facts about yourself."

    # Filter by keyword if a query is provided
    if query and query.strip():
        query_lower = query.strip().lower()
        filtered = [
            f for f in facts
            if query_lower in f["text"].lower()
        ]
        if not filtered:
            return f"No memories matching \"{query}\". Use /recall to see all stored facts."
        facts = filtered

    # Build numbered list
    lines = ["Stored memories:"]
    lines.append("-" * 45)
    for i, fact in enumerate(facts, 1):
        timestamp = fact.get("stored_at", "unknown date")
        lines.append(f"  {i}. {fact['text']}")
        lines.append(f"     (stored: {timestamp})")
    lines.append("-" * 45)
    lines.append(f"Total: {len(facts)} {'memory' if len(facts) == 1 else 'memories'}")

    return "\n".join(lines)


def forget_fact(index_or_keyword):
    """
    Remove a specific stored fact by index number or keyword match.

    Args:
        index_or_keyword: Either a numeric index (1-based) shown by /recall,
                          or a keyword string to match against stored facts.

    Returns:
        Confirmation or error message string.
    """
    if not index_or_keyword or not str(index_or_keyword).strip():
        return ("Please specify which memory to forget.  \n"
                "Use a number: /forget 1  \n"
                "Or a keyword: /forget favourite movie")

    data = load_memory()
    facts = data.get("facts", [])

    if not facts:
        return "No memories stored. Nothing to forget."

    target = str(index_or_keyword).strip()

    # Try numeric index first
    try:
        idx = int(target) - 1  # Convert 1-based to 0-based
        if 0 <= idx < len(facts):
            removed = facts.pop(idx)
            save_memory(data)
            return f"Forgotten: \"{removed['text']}\""
        return f"Invalid memory number. Use /recall to see the list (1-{len(facts)})."
    except ValueError:
        pass  # Not a number — try keyword match

    # Keyword match — find the first fact containing the keyword
    target_lower = target.lower()
    for i, fact in enumerate(facts):
        if target_lower in fact["text"].lower():
            removed = facts.pop(i)
            save_memory(data)
            return f"Forgotten: \"{removed['text']}\""

    return f"No memory matching \"{target}\". Use /recall to see all stored facts."


def clear_all_memories():
    """
    Delete ALL stored facts after confirmation.
    Note: This function is called programmatically — the caller
    should handle confirmation prompting.

    Returns:
        Confirmation message string.
    """
    data = load_memory()
    count = len(data.get("facts", []))

    if count == 0:
        return "No memories stored. Nothing to clear."

    data["facts"] = []
    save_memory(data)
    return f"All {count} memories cleared."


def load_all_facts():
    """
    Return the raw list of stored fact dicts.

    Callers that need a count or want to iterate should use this; recall_facts()
    and list_memories() return display strings, not data.
    """
    facts = load_memory().get("facts", [])
    return facts if isinstance(facts, list) else []


def list_memories():
    """
    Show all stored facts in a numbered list.
    Alias for recall_facts(None).

    Returns:
        Formatted string listing all stored facts.
    """
    return recall_facts(None)


# ---------------------------------------------------------------------------
# Memory injection — retrieve relevant facts for QA context
# ---------------------------------------------------------------------------
def get_relevant_memories(question):
    """
    Retrieve user memories that are relevant to the given question.
    Uses simple keyword overlap to determine relevance.

    Args:
        question: The user's question string.

    Returns:
        A context string to prepend to the model prompt, or empty
        string if no relevant memories are found.
    """
    data = load_memory()
    facts = data.get("facts", [])

    if not facts:
        return ""

    # Extract meaningful words from the question (3+ chars, lowercased)
    question_lower = question.lower()
    question_words = {
        word.strip(".,!?;:'\"()[]")
        for word in question_lower.split()
        if len(word.strip(".,!?;:'\"()[]")) >= 3
    }

    # Filter out common stop words that would cause false matches
    stop_words = {
        "the", "and", "for", "are", "but", "not", "you", "all",
        "can", "had", "her", "was", "one", "our", "out", "has",
        "its", "any", "who", "what", "when", "where", "why", "how",
        "this", "that", "with", "from", "they", "been", "have",
        "will", "each", "make", "like", "just", "about", "over",
        "such", "take", "than", "them", "very", "some", "could",
        "would", "should", "into", "also", "these", "tell", "does",
        "much", "know", "which", "there", "their", "more", "other",
        "most", "only", "your", "then", "many",
    }
    question_words -= stop_words

    if not question_words:
        return ""

    # Find facts that share keywords with the question
    relevant = []
    for fact in facts:
        fact_lower = fact["text"].lower()
        fact_words = {
            word.strip(".,!?;:'\"()[]")
            for word in fact_lower.split()
            if len(word.strip(".,!?;:'\"()[]")) >= 3
        }
        fact_words -= stop_words

        # Check for keyword overlap
        overlap = question_words & fact_words
        if overlap:
            relevant.append(fact["text"])

    if not relevant:
        return ""

    # Build context string
    facts_text = "\n".join(f"- {f}" for f in relevant)
    return (
        "The user has previously told you these personal facts:\n"
        f"{facts_text}\n"
        "Use this information if it's relevant to answering their question.\n\n"
    )
