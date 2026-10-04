"""
router.py — NEXUS Intent Router
Classifies user input into one of the known ACTION types using the AI model,
then dispatches to the appropriate handler module (tools.files, tools.system, etc.).
Includes a fast-path keyword router that bypasses the LLM for obvious intents.
"""

import logging
import re

from nexus.llm.loader import generate
from nexus.utils import extract_filepath

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fast-path keyword routing — bypass LLM for obvious intents
# ---------------------------------------------------------------------------
# Each entry: (set of keywords, action, needs_params)
# Weak signal: only fires when the message is not an informational question.
# "charge"/"charging"/"plugged" were removed -- the specific phrasings that
# actually mean the laptop battery ("am i charging", "charge level") are all
# covered by _BATTERY_PATTERNS below, so the bare words only caused misfires
# such as "I need to charge my phone".
_BATTERY_KEYWORDS = {"battery"}
_PROCESS_KEYWORDS = {"process", "processes", "task manager"}
_GREETING_KEYWORDS = {"hello", "hi", "hey", "good morning", "good evening", "good afternoon", "howdy", "greetings"}
_SUMMARISE_KEYWORDS = {"summarise", "summarize", "summary", "summarisation", "summarization"}
_FOLDER_KEYWORDS = {"folder", "directory", "all files", "all documents", "entire folder"}
_QUESTION_STARTERS = ("what ", "what's", "who ", "who's", "where ", "when ", "why ", "how ",
                       "explain ", "define ", "describe ", "tell me", "give me",
                       "name ", "can you", "could you", "do you know", "is ", "are ", "does ")
_RESET_KEYWORDS = {"reset", "clear", "delete"}
_RESET_CONTEXT = {"training", "lora", "adapter", "custom", "model", "weights"}
_MODEL_INFO_KEYWORDS = {"model info", "model status", "what model", "which model", "model details"}
_SHUTDOWN_KEYWORDS = {"shutdown", "shut down", "turn off", "power off"}
_RESTART_KEYWORDS = {"restart", "reboot", "re-boot"}
_SLEEP_KEYWORDS = {"sleep", "hibernate", "standby"}
_CANCEL_SHUTDOWN_KEYWORDS = {"cancel shutdown", "cancel restart", "abort shutdown", "stop shutdown"}
_CLEAR_HISTORY_KEYWORDS = {"clear chat", "new chat", "clear history", "clear conversation",
                            "reset chat", "forget everything", "start over", "new conversation"}

# Phase 28 — Persistent memory keywords
_REMEMBER_PATTERNS = [
    "remember that", "remember my", "don't forget", "do not forget",
    "store this", "save this fact", "my favourite", "my favorite",
    "remember i ", "remember me", "note that", "keep in mind",
]
_RECALL_PATTERNS = [
    "what do you remember", "what do you know about me",
    "recall my", "recall all", "show memories", "list memories",
    "what have i told you", "my memories", "stored memories",
    "what did i tell you",
]
_FORGET_PATTERNS = [
    "forget my", "forget that", "forget about", "remove memory",
    "delete memory", "clear memories", "clear all memories",
    "forget what i told you",
]
_RENAME_KEYWORDS = {"rename"}
_COPY_KEYWORDS = {"copy"}
_MOVE_KEYWORDS = {"move"}
_DELETE_KEYWORDS = {"delete file", "remove file", "delete the file", "remove the file"}
_LIST_FILES_KEYWORDS = {"list files", "list folder", "show files", "show folder", "dir ", "ls "}
_REINDEX_KEYWORDS = {"reindex", "re-index", "rebuild index", "index my files", "index my documents",
                      "index the folder", "index files", "build index"}
_RAG_QA_KEYWORDS = {"ask about my files", "ask about my documents", "search my files",
                     "search my documents", "find in my files", "find in my documents",
                     "look in my files", "look in my documents", "check my files",
                     "from my files", "from my documents", "in my files", "in my documents",
                     "according to my files", "based on my files", "what do my files say"}

# Phase 26 — Expanded substring patterns for natural-language matching
_BATTERY_PATTERNS = [
    "battery percentage", "battery status", "battery level", "battery left",
    "battery life", "charge level", "charging status", "charge status",
    "how much battery", "how much charge", "what is my battery",
    "what's my battery", "check battery", "check charge", "check charging",
    "is it charging", "am i charging", "is my laptop charging",
    "show battery", "get battery", "battery info", "power status",
]
_PROCESS_PATTERNS = [
    "list processes", "list top", "top 10 processes", "top 5 processes",
    "top processes", "running processes", "running programs", "active programs",
    "active processes", "show running", "what is running", "what's running",
    "show processes", "get processes", "running apps", "running applications",
    "task list", "list running", "check processes", "check running",
    "show tasks", "running tasks", "monitor processes",
]
_SYSTEM_CMD_PATTERNS = [
    "turn off computer", "turn off my computer", "turn off pc", "turn off my pc",
    "turn off laptop", "turn off my laptop", "turn off the computer",
    "power off computer", "power off pc", "power off laptop",
    "shut down computer", "shut down pc", "shut down laptop",
    "shut down my computer", "shut down my pc", "shut down my laptop",
    "restart computer", "restart pc", "restart laptop", "restart my computer",
    "restart my pc", "restart my laptop", "reboot computer", "reboot pc",
    "reboot my computer", "reboot my pc", "reboot laptop",
    "put to sleep", "put computer to sleep", "put pc to sleep",
    "put my computer to sleep", "sleep mode", "go to sleep",
]


# Phrasings that mean "tell me about X" rather than "do X to my machine".
# A message opening this way is asking a question, so bare keyword matches are
# suppressed for it.
_INFORMATIONAL_STARTERS = (
    "how do ", "how does ", "how did ", "how can ", "how would ", "how should ",
    "how many ", "how much ", "how long ", "how often ",
    "what is ", "what are ", "what was ", "what were ", "what does ", "what do ",
    "what did ", "what factors", "what causes", "what happens", "what's the ",
    "what kind", "what type",
    "why ", "when did ", "when was ", "when do ", "where do ", "where is ",
    "who is ", "who was ", "who are ",
    "explain ", "define ", "describe ", "tell me about", "tell me how",
    "difference between", "compare ",
    "can you explain", "could you explain",
)

# If the user says "my <thing>", they usually mean their own machine, which
# outranks the informational reading.
_PERSONAL_MARKERS = (
    "my battery", "my laptop", "my computer", "my pc", "my machine",
    "my system", "my cpu", "my ram", "my disk", "my files", "my documents",
)


def _is_informational(lower):
    """
    True when the message reads as a question about a topic rather than a
    command to act on this machine.

    Strong multi-word patterns are matched before this gate is consulted, so a
    genuine request like "what is my battery percentage" is unaffected.
    """
    if any(marker in lower for marker in _PERSONAL_MARKERS):
        return False
    return lower.startswith(_INFORMATIONAL_STARTERS)


def fast_route(user_input):
    """
    Attempt to classify user input using simple keyword matching.
    This is MUCH faster than calling the LLM for intent detection.

    Priority order (checked top-to-bottom — Phase 26 fix):
      1. Clear history / Model info / Reset training
      2. Cancel shutdown
      3. Battery / Processes / Shutdown / Restart / Sleep  (weak matches gated)
      4. File management (rename, copy, move, delete, list)
      5. RAG (reindex, ask about documents)
      6. Summarise file/folder
      7. Greetings
      8. Questions (catch-all — LAST priority)

    Returns:
        (action, params) tuple if a match is found, or None if the
        input should be sent to the LLM for classification.
    """
    lower = user_input.lower().strip()
    words = set(lower.split())

    # Weak (bare-word) matches are suppressed for informational questions.
    informational = _is_informational(lower)

    # --- 0. Persistent memory (Phase 28) — check BEFORE clear history ---
    # "remember that X" → REMEMBER, not CLEAR_HISTORY
    if any(pat in lower for pat in _REMEMBER_PATTERNS):
        # Extract the fact text: strip the trigger phrase
        fact = user_input
        for pat in _REMEMBER_PATTERNS:
            idx = lower.find(pat)
            if idx >= 0:
                fact = user_input[idx + len(pat):].strip()
                break
        return ("ACTION:REMEMBER", fact if fact else user_input)

    if any(pat in lower for pat in _RECALL_PATTERNS):
        # Strip the trigger phrase so only a real search term is left.
        # "recall my address"      -> filter on "address"
        # "what do you remember"   -> no filter, list everything
        query = ""
        for pat in _RECALL_PATTERNS:
            idx = lower.find(pat)
            if idx >= 0:
                query = user_input[idx + len(pat):].strip()
                break
        query = query.lstrip("?,.:; ").strip()
        for filler in ("about me", "about myself", "so far", "please"):
            if query.lower() == filler:
                query = ""
        return ("ACTION:RECALL", query)

    if any(pat in lower for pat in _FORGET_PATTERNS):
        # Extract what to forget
        fact = user_input
        for pat in _FORGET_PATTERNS:
            idx = lower.find(pat)
            if idx >= 0:
                fact = user_input[idx + len(pat):].strip()
                break
        return ("ACTION:FORGET", fact if fact else user_input)

    # --- 1. Clear conversation history ---
    if any(kw in lower for kw in _CLEAR_HISTORY_KEYWORDS):
        return ("ACTION:CLEAR_HISTORY", "")

    # --- 1. Model info ---
    if any(kw in lower for kw in _MODEL_INFO_KEYWORDS):
        return ("ACTION:MODEL_INFO", "")

    # --- 1. Reset training ---
    if words & _RESET_KEYWORDS and words & _RESET_CONTEXT:
        return ("ACTION:RESET_TRAINING", "")

    # --- 2. Cancel shutdown (before shutdown/restart checks) ---
    if any(kw in lower for kw in _CANCEL_SHUTDOWN_KEYWORDS):
        return ("ACTION:CANCEL_SHUTDOWN", "")

    # --- 3. Battery (substring patterns FIRST, then keyword fallback) ---
    # Checked BEFORE questions so "what is my battery?" → BATTERY, not QA
    if any(pat in lower for pat in _BATTERY_PATTERNS):
        return ("ACTION:BATTERY", "")
    if words & _BATTERY_KEYWORDS and not informational:
        return ("ACTION:BATTERY", "")

    # --- 3. Processes (substring patterns FIRST, then keyword fallback) ---
    # Checked BEFORE questions so "list top 10 processes" → PROCESSES, not QA
    if any(pat in lower for pat in _PROCESS_PATTERNS):
        return ("ACTION:PROCESSES", "")
    if words & _PROCESS_KEYWORDS and not informational:
        return ("ACTION:PROCESSES", "")

    # --- 3. Shutdown / Restart / Sleep ---
    # Check expanded system command patterns first
    if any(pat in lower for pat in _SYSTEM_CMD_PATTERNS):
        if any(kw in lower for kw in ["restart", "reboot", "re-boot"]):
            return ("ACTION:RESTART", "")
        if any(kw in lower for kw in ["sleep", "hibernate", "standby"]):
            return ("ACTION:SLEEP", "")
        return ("ACTION:SHUTDOWN", "")
    # Then check original keywords
    if any(kw in lower for kw in _SHUTDOWN_KEYWORDS) and not informational:
        return ("ACTION:SHUTDOWN", "")
    if any(kw in lower for kw in _RESTART_KEYWORDS) and not informational:
        return ("ACTION:RESTART", "")
    if any(kw in lower for kw in _SLEEP_KEYWORDS) and not informational:
        return ("ACTION:SLEEP", "")

    # --- 4. File management ---
    if words & _RENAME_KEYWORDS and not informational:
        filepath = extract_filepath(user_input)
        return ("ACTION:RENAME_FILE", filepath or "")
    if words & _COPY_KEYWORDS and "copyright" not in lower and not informational:
        filepath = extract_filepath(user_input)
        return ("ACTION:COPY_FILE", filepath or "")
    if words & _MOVE_KEYWORDS and "movie" not in lower and not informational:
        filepath = extract_filepath(user_input)
        return ("ACTION:MOVE_FILE", filepath or "")
    if any(kw in lower for kw in _DELETE_KEYWORDS) and not informational:
        filepath = extract_filepath(user_input)
        return ("ACTION:DELETE_FILE", filepath or "")
    if any(kw in lower for kw in _LIST_FILES_KEYWORDS):
        filepath = extract_filepath(user_input)
        return ("ACTION:LIST_FILES", filepath or "")

    # --- 5. RAG: reindex files ---
    if any(kw in lower for kw in _REINDEX_KEYWORDS):
        filepath = extract_filepath(user_input)
        return ("ACTION:REINDEX", filepath or "")

    # --- 5. RAG: ask about documents (before general QA) ---
    if any(kw in lower for kw in _RAG_QA_KEYWORDS):
        return ("ACTION:RAG_QA", user_input)

    # --- 6. Summarise file or folder ---
    if words & _SUMMARISE_KEYWORDS:
        if any(kw in lower for kw in _FOLDER_KEYWORDS):
            filepath = extract_filepath(user_input)
            return ("ACTION:SUMMARISE_FOLDER", filepath or "")
        filepath = extract_filepath(user_input)
        if filepath:
            return ("ACTION:SUMMARISE_FILE", filepath)
        return None

    # --- 7. Greetings ---
    if lower in _GREETING_KEYWORDS or any(lower.startswith(g) for g in _GREETING_KEYWORDS):
        return ("ACTION:QA", user_input)

    # --- 8. Questions (LAST priority — catch-all for question-phrased inputs) ---
    # This is intentionally LAST so action keywords like battery/processes
    # are matched first even when phrased as questions.
    if any(lower.startswith(q) for q in _QUESTION_STARTERS) or lower.endswith("?"):
        return ("ACTION:QA", user_input)

    # No keyword match — fall through to LLM
    return None


# ---------------------------------------------------------------------------
# System prompt for intent detection
# ---------------------------------------------------------------------------
INTENT_SYSTEM_PROMPT = """\
You are NEXUS, an intent classification engine for an offline AI assistant.
Your ONLY job is to read the user's message and output exactly ONE action line.
Do NOT explain anything. Do NOT add extra text. Output ONLY the action line.

Valid actions and when to use them:

ACTION:SUMMARISE_FILE|<filepath>
  Use when the user wants to summarise, explain, or read a specific file.
  Examples:
    User: "summarise notes.txt"            → ACTION:SUMMARISE_FILE|notes.txt
    User: "explain what is in report.pdf"  → ACTION:SUMMARISE_FILE|report.pdf

ACTION:SUMMARISE_FOLDER|<folderpath>
  Use when the user wants to summarise all files in a folder or directory.
  Examples:
    User: "summarise all files in C:\\docs"       → ACTION:SUMMARISE_FOLDER|C:\\docs
    User: "give me a summary of the folder D:\\notes" → ACTION:SUMMARISE_FOLDER|D:\\notes

ACTION:QA|<question>
  Use when the user asks a general knowledge question, wants an explanation, or requests help with a topic.
  Examples:
    User: "what is machine learning?"        → ACTION:QA|what is machine learning?
    User: "explain python decorators"        → ACTION:QA|explain python decorators

ACTION:BATTERY
  Use when the user asks about battery status, percentage, or charging state.
  Examples:
    User: "what is my battery percentage?"   → ACTION:BATTERY
    User: "am I plugged in?"                 → ACTION:BATTERY

ACTION:PROCESSES
  Use when the user asks about running programs, processes, or what is using memory/CPU.
  Examples:
    User: "show running processes"           → ACTION:PROCESSES
    User: "what programs are running?"       → ACTION:PROCESSES

ACTION:UNKNOWN
  Use ONLY when the input does not match ANY of the above categories.
  Examples:
    User: "asdfghjkl"                        → ACTION:UNKNOWN
    User: "open chrome"                      → ACTION:UNKNOWN

ACTION:RESET_TRAINING
  Use when the user wants to reset, clear, or delete their custom training/LoRA data.
  Examples:
    User: "reset my training"                → ACTION:RESET_TRAINING
    User: "delete custom model"              → ACTION:RESET_TRAINING
    User: "go back to base model"            → ACTION:RESET_TRAINING

ACTION:MODEL_INFO
  Use when the user asks about the current model, its status, or configuration.
  Examples:
    User: "model info"                       → ACTION:MODEL_INFO
    User: "what model are you using"         → ACTION:MODEL_INFO

ACTION:SHUTDOWN
  Use when the user wants to shut down or turn off the computer.
  Examples:
    User: "shut down my computer"            → ACTION:SHUTDOWN
    User: "turn off the PC"                  → ACTION:SHUTDOWN

ACTION:RESTART
  Use when the user wants to restart or reboot the computer.
  Examples:
    User: "restart the computer"             → ACTION:RESTART
    User: "reboot the system"                → ACTION:RESTART

ACTION:SLEEP
  Use when the user wants to put the computer to sleep or hibernate.
  Examples:
    User: "put the computer to sleep"        → ACTION:SLEEP
    User: "hibernate the system"             → ACTION:SLEEP

ACTION:CANCEL_SHUTDOWN
  Use when the user wants to cancel a pending shutdown or restart.
  Examples:
    User: "cancel shutdown"                  → ACTION:CANCEL_SHUTDOWN
    User: "stop the restart"                 → ACTION:CANCEL_SHUTDOWN

Rules:
1. Always respond with ONLY the ACTION line. Nothing else.
2. If the user mentions a file name or path, it is likely SUMMARISE_FILE.
3. If the user mentions a folder or directory, it is likely SUMMARISE_FOLDER.
4. If the user asks a question about a topic (not a file), it is QA.
5. Greeting messages like "hello" or "hi" should map to ACTION:QA|<the greeting>.
6. For file management actions, include the file/folder path after the pipe.

ACTION:RENAME_FILE|<filepath>
  Use when the user wants to rename a file.
  Examples:
    User: "rename notes.txt to notes_old.txt"  → ACTION:RENAME_FILE|notes.txt
    User: "rename my report"                   → ACTION:RENAME_FILE|report

ACTION:COPY_FILE|<filepath>
  Use when the user wants to copy a file to another location.
  Examples:
    User: "copy notes.txt to D:\\backup"        → ACTION:COPY_FILE|notes.txt

ACTION:MOVE_FILE|<filepath>
  Use when the user wants to move a file to another location.
  Examples:
    User: "move report.pdf to D:\\docs"         → ACTION:MOVE_FILE|report.pdf

ACTION:DELETE_FILE|<filepath>
  Use when the user wants to delete or remove a file.
  Examples:
    User: "delete old_notes.txt"               → ACTION:DELETE_FILE|old_notes.txt
    User: "remove temp.txt"                    → ACTION:DELETE_FILE|temp.txt

ACTION:LIST_FILES|<folderpath>
  Use when the user wants to see files in a folder or list directory contents.
  Examples:
    User: "list files in C:\\docs"              → ACTION:LIST_FILES|C:\\docs
    User: "show me what's in my folder"        → ACTION:LIST_FILES|

ACTION:CLEAR_HISTORY
  Use when the user wants to clear the conversation history or start a new chat.
  Examples:
    User: "clear chat"                         → ACTION:CLEAR_HISTORY
    User: "new conversation"                   → ACTION:CLEAR_HISTORY
    User: "forget everything"                  → ACTION:CLEAR_HISTORY

ACTION:RAG_QA|<question>
  Use when the user asks a question specifically about THEIR documents or files.
  Examples:
    User: "what do my files say about revenue?"  → ACTION:RAG_QA|what do my files say about revenue?
    User: "search my documents for project deadlines" → ACTION:RAG_QA|search my documents for project deadlines
    User: "find info about budgets in my files"  → ACTION:RAG_QA|find info about budgets in my files

ACTION:REINDEX|<folderpath>
  Use when the user wants to (re)index their files for document search.
  Examples:
    User: "reindex my files"                    → ACTION:REINDEX
    User: "index files in C:\\docs"              → ACTION:REINDEX|C:\\docs
    User: "rebuild the search index"            → ACTION:REINDEX

"""


# ---------------------------------------------------------------------------
# Action registry -- single source of truth for the action namespace
# (V4.0 Phase 38). Before this, the same ~23 action names were hand-typed
# independently in three places: fast_route()'s returned "ACTION:X" string
# literals, commands.py's own lowercase _TASK_KEYWORDS values, and this
# module's _VALID_ACTIONS set -- which is exactly the kind of three-way
# drift that caused the routing-confusion bugs in docs/feedback.txt.
#
# fast_route()'s per-action keyword/pattern constants above keep their own
# natural-language matching logic (Phase 26's false-positive gating) rather
# than being folded in here, since that gating genuinely doesn't apply to
# /task -- an explicit command has no ambiguity to guard against. What *is*
# shared is the name each action goes by, and the extra keyword phrases
# commands.py's /task recognises for it; both now come from here instead of
# a second, independently hand-typed dict.
# ---------------------------------------------------------------------------
ACTIONS = {
    "SUMMARISE_FILE": {},
    "SUMMARISE_FOLDER": {},
    "QA": {},
    "BATTERY": {"task_keywords": ("battery", "charge", "charging", "power")},
    "PROCESSES": {"task_keywords": ("processes", "process", "tasks", "running",
                                     "running programs", "running apps", "task manager")},
    "RESET_TRAINING": {"task_keywords": ("reset training", "reset model", "clear training")},
    "MODEL_INFO": {"task_keywords": ("model info", "model status")},
    "SHUTDOWN": {"task_keywords": ("shutdown", "shut down", "turn off", "power off")},
    "RESTART": {"task_keywords": ("restart", "reboot")},
    "SLEEP": {"task_keywords": ("sleep", "hibernate", "standby")},
    "CANCEL_SHUTDOWN": {"task_keywords": ("cancel shutdown", "cancel restart")},
    "RENAME_FILE": {"task_keywords": ("rename",)},
    "COPY_FILE": {"task_keywords": ("copy",)},
    "MOVE_FILE": {"task_keywords": ("move",)},
    "DELETE_FILE": {"task_keywords": ("delete", "remove")},
    "LIST_FILES": {"task_keywords": ("list", "list files", "show files", "dir")},
    "CLEAR_HISTORY": {},
    "RAG_QA": {},
    "REINDEX": {},
    "REMEMBER": {},
    "RECALL": {},
    "FORGET": {},
    "UNKNOWN": {},
}

# Derived from ACTIONS above instead of being hand-typed a second time.
_VALID_ACTIONS = {f"ACTION:{name}" for name in ACTIONS}

# Regex alternation of bare action names (no "ACTION:" prefix), likewise
# derived rather than hand-typed a third time.
_ACTION_NAME_PATTERN = "|".join(sorted(ACTIONS))


# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------
def detect_intent(user_input):
    """
    Classify the user's input into an (action, params) tuple.

    fast_route()'s keyword table (built from the ACTIONS registry above) is
    the primary classifier; the LLM is an explicit last-resort fallback for
    whatever fast_route() doesn't recognise -- not a second path that runs
    in parallel and might disagree with it (V4.0 Phase 38).

    Args:
        user_input: Raw string from the user.

    Returns:
        Tuple of (action_string, params_string).
        action_string is one of the ACTION:* constants.
        params_string is the part after '|', or empty string if none.
    """
    # --- Fast path: try keyword matching first (instant, no LLM call) ---
    fast_result = fast_route(user_input)
    if fast_result is not None:
        return fast_result

    # --- Slow path: use LLM for ambiguous inputs ---
    # Call the model with a very short max_tokens — we only need the action line
    raw = generate(
        prompt=user_input,
        max_tokens=80,
        system_prompt=INTENT_SYSTEM_PROMPT,
    )

    if not raw:
        return ("ACTION:UNKNOWN", "")

    # Clean up the response — take only the first line and strip whitespace
    raw = raw.strip()
    first_line = raw.split("\n")[0].strip()

    # Try to extract the ACTION from the response
    # The model might output extra text; find the ACTION:XXX pattern
    action_match = re.search(
        rf"(ACTION:(?:{_ACTION_NAME_PATTERN}))",
        first_line,
        re.IGNORECASE,
    )

    if not action_match:
        return ("ACTION:UNKNOWN", "")

    action = action_match.group(1).upper()

    # Extract params (everything after the '|' separator, if present)
    params = ""
    pipe_idx = first_line.find("|", action_match.end() - 1)
    if pipe_idx != -1:
        params = first_line[pipe_idx + 1:].strip()

    # If the action expects a filepath but model didn't put one in params,
    # try to extract it from the original user input
    if action in ("ACTION:SUMMARISE_FILE", "ACTION:SUMMARISE_FOLDER") and not params:
        extracted = extract_filepath(user_input)
        if extracted:
            params = extracted

    # Also try to extract filepaths for file management actions
    if action in ("ACTION:RENAME_FILE", "ACTION:COPY_FILE", "ACTION:MOVE_FILE",
                  "ACTION:DELETE_FILE", "ACTION:LIST_FILES", "ACTION:REINDEX") and not params:
        extracted = extract_filepath(user_input)
        if extracted:
            params = extracted

    return (action, params)


# ---------------------------------------------------------------------------
# Route dispatcher
# ---------------------------------------------------------------------------
# Module-level conversation memory reference (set by agent.py at startup)
_conversation_memory = None


def set_conversation_memory(memory_instance):
    """
    Set the conversation memory instance used for QA context.
    Called by agent.py during initialisation.
    """
    global _conversation_memory
    _conversation_memory = memory_instance


def _build_qa_context(question):
    """
    Build the (prompt, history_messages) pair ACTION:QA sends to the model.

    Shared by route()'s non-streaming dispatch and route_stream()'s
    streaming dispatch so the two can't drift apart (V4.0 Phase 41).
    """
    history_messages = None
    if _conversation_memory and not _conversation_memory.is_empty:
        history_messages = _conversation_memory.get_messages_for_model(max_turns=5)

    # Inject relevant persistent memories into the prompt (Phase 28)
    memory_context = ""
    try:
        from nexus.agent import memory
        memory_context = memory.get_relevant_memories(question)
    except Exception:
        # Memory lookup is a best-effort enhancement — QA must still work
        # if it fails, but the failure shouldn't vanish silently.
        logger.debug("Memory lookup failed for QA; continuing without it", exc_info=True)

    prompt = memory_context + "Answer the following question in 2-3 sentences. Be brief and direct:\n\n" + question
    return prompt, history_messages


def route(user_input):
    """
    Main entry point: classify the user's input and dispatch to the correct
    handler function. Returns the result string to be displayed to the user.

    Priority order:
      1. Slash commands (/task, /remember, etc.) — instant, no LLM
      2. fast_route() keyword matching — fast, no LLM
      3. LLM intent detection — slowest, used as fallback

    Args:
        user_input: Raw string from the user.

    Returns:
        A response string from the appropriate handler.
    """
    # --- Phase 27: Check for slash commands FIRST (bypass LLM entirely) ---
    from nexus.agent import commands
    slash_result = commands.execute_input(user_input)
    if slash_result is not None:
        return slash_result

    # --- Normal routing: fast_route() → LLM fallback ---
    action, params = detect_intent(user_input)
    return _dispatch_action(action, params, user_input)


def _dispatch_action(action, params, user_input):
    """
    Call the handler for an already-classified action.

    Split out from route() so the streaming path (route_stream(), used by
    the FastAPI/pywebview UI) can reuse the exact same dispatch logic for
    every non-QA action instead of duplicating this chain (V4.0 Phase 41).
    """
    if action == "ACTION:SUMMARISE_FILE":
        # Lazy import to avoid circular imports and missing-module errors
        from nexus.tools import files
        if not params:
            return "Please specify which file you'd like me to summarise."
        return files.summarise_file(params)

    if action == "ACTION:SUMMARISE_FOLDER":
        from nexus.tools import files
        if not params:
            return "Please specify which folder you'd like me to summarise."
        return files.summarise_folder(params)

    if action == "ACTION:QA":
        from nexus.llm.loader import generate as _generate
        question = params if params else user_input
        prompt, history_messages = _build_qa_context(question)
        return _generate(prompt, max_tokens=200, conversation_history=history_messages)

    if action == "ACTION:BATTERY":
        from nexus.tools import system
        return system.get_battery()

    if action == "ACTION:PROCESSES":
        from nexus.tools import system
        return system.get_processes()

    if action == "ACTION:RESET_TRAINING":
        from nexus.llm import loader
        return loader.reset_training()

    if action == "ACTION:MODEL_INFO":
        from nexus.llm import loader
        return loader.get_model_info()

    if action == "ACTION:SHUTDOWN":
        from nexus.tools import system
        return system.shutdown_system()

    if action == "ACTION:RESTART":
        from nexus.tools import system
        return system.restart_system()

    if action == "ACTION:SLEEP":
        from nexus.tools import system
        return system.sleep_system()

    if action == "ACTION:CANCEL_SHUTDOWN":
        from nexus.tools import system
        return system.cancel_shutdown()

    if action == "ACTION:RENAME_FILE":
        from nexus.tools import files
        if not params:
            return "Please specify which file to rename."
        # Try to extract new name from user input
        new_name = _extract_new_name(user_input)
        if not new_name:
            return (
                f"I can rename {params}, but I need the new name. "
                f'Try: rename {params} to <new name>'
            )
        return files.rename_file(params, new_name)

    if action == "ACTION:COPY_FILE":
        from nexus.tools import files
        if not params:
            return "Please specify which file to copy."
        destination = _extract_destination(user_input)
        if not destination:
            return (
                f"I can copy {params}, but I need a destination. "
                f'Try: copy {params} to D:\\backup'
            )
        return files.copy_file(params, destination)

    if action == "ACTION:MOVE_FILE":
        from nexus.tools import files
        if not params:
            return "Please specify which file to move."
        destination = _extract_destination(user_input)
        if not destination:
            return (
                f"I can move {params}, but I need a destination. "
                f'Try: move {params} to D:\\docs'
            )
        return files.move_file(params, destination)

    if action == "ACTION:DELETE_FILE":
        from nexus.tools import files
        if not params:
            return "Please specify which file to delete."
        return files.delete_file(params)

    if action == "ACTION:LIST_FILES":
        from nexus.tools import files
        if not params:
            return "Please specify which folder to list."
        return files.list_files(params)

    if action == "ACTION:CLEAR_HISTORY":
        if _conversation_memory:
            _conversation_memory.clear()
            return "Conversation history cleared. Starting a fresh chat!"
        return "No conversation history to clear."

    if action == "ACTION:REMEMBER":
        from nexus.agent import memory
        fact = params if params else user_input
        return memory.remember_fact(fact)

    if action == "ACTION:RECALL":
        from nexus.agent import memory
        query = params if params else None
        return memory.recall_facts(query)

    if action == "ACTION:FORGET":
        from nexus.agent import memory
        return memory.forget_fact(params if params else "")

    if action == "ACTION:RAG_QA":
        from nexus.tools import rag
        question = params if params else user_input
        return rag.ask(question)

    if action == "ACTION:REINDEX":
        from nexus.tools import rag
        if not params:
            # If no folder specified, try to use the first allowed folder
            from nexus.security.permissions import load_permissions
            perms = load_permissions()
            if perms:
                params = perms[0]
            else:
                return "Please specify which folder to index, e.g. \"reindex files in C:\\Documents\""
        return rag.build_index(params)

    # ACTION:UNKNOWN or anything unrecognised
    return "I did not understand that. Please try rephrasing."


def route_stream(user_input, on_token):
    """
    Like route(), but delivers the reply incrementally via on_token()
    instead of returning the complete string in one call. Used by the
    FastAPI/pywebview UI's WebSocket endpoint to stream model output
    live instead of showing a blocking placeholder (V4.0 Phase 41).

    Only ACTION:QA is actually streamed token-by-token -- it's the only
    action that calls the model for free-form generation. Every other
    action (file ops, system info, slash commands, ...) resolves
    instantly and is delivered to on_token() as a single chunk, via the
    same _dispatch_action() route() uses, so the two entry points can't
    disagree about what a given action does.

    Args:
        user_input: Raw string from the user.
        on_token:   Callable invoked with each text chunk as it's ready.
    """
    from nexus.agent import commands
    slash_result = commands.execute_input(user_input)
    if slash_result is not None:
        on_token(slash_result)
        return

    action, params = detect_intent(user_input)

    if action == "ACTION:QA":
        from nexus.llm.loader import generate_stream
        question = params if params else user_input
        prompt, history_messages = _build_qa_context(question)
        for chunk in generate_stream(prompt, max_tokens=200, conversation_history=history_messages):
            on_token(chunk)
        return

    on_token(_dispatch_action(action, params, user_input))


# ---------------------------------------------------------------------------
# Helper functions for file management param extraction
# ---------------------------------------------------------------------------
def _extract_new_name(text):
    """Try to extract a 'new name' from text like 'rename X to Y'."""
    lower = text.lower()
    for keyword in (" to ", " as ", " into "):
        idx = lower.find(keyword)
        if idx != -1:
            candidate = text[idx + len(keyword):].strip().strip('"').strip("'")
            if candidate:
                return candidate
    return None


def _extract_destination(text):
    """Try to extract a destination path from text like 'copy X to Y'."""
    lower = text.lower()
    for keyword in (" to ", " into "):
        idx = lower.find(keyword)
        if idx != -1:
            candidate = text[idx + len(keyword):].strip().strip('"').strip("'")
            if candidate:
                return candidate
    return None



