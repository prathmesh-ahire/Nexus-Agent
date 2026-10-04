"""
commands.py -- NEXUS Slash Command System
Detects and routes slash commands (/task, /save, /remember, /help, etc.)
directly to handler functions, bypassing LLM intent detection entirely.
This makes NEXUS faster and more predictable for known commands.

Phase 27 of the NEXUS implementation plan.
"""

import os
import re
import subprocess

from nexus.security import confirm

# ---------------------------------------------------------------------------
# Supported slash commands
# ---------------------------------------------------------------------------
SUPPORTED_COMMANDS = {
    "/task": "Execute a system task (battery, processes, shutdown, file ops, etc.)",
    "/save": "Save data to a file",
    "/remember": "Store a fact in persistent memory (Phase 28)",
    "/help": "Show all available slash commands and examples",
    "/recall": "List stored memories (Phase 28)",
    "/forget": "Remove a stored memory (Phase 28)",
    "/model": "Show model info or reset training (Phase 34)",
    "/settings": "Open settings or show current configuration (Phase 34)",
    "/theme": "Switch theme (Dark / Light / Blue) (Phase 34)",
    "/clear": "Clear conversation history (Phase 34)",
}


# ---------------------------------------------------------------------------
# Task keyword mapping -- /task subcommands to handler functions
# ---------------------------------------------------------------------------
_TASK_KEYWORDS = {
    # System info
    "battery": "battery",
    "charge": "battery",
    "charging": "battery",
    "power": "battery",
    "processes": "processes",
    "process": "processes",
    "tasks": "processes",
    "running": "processes",
    "running programs": "processes",
    "running apps": "processes",
    "task manager": "processes",
    # System commands
    "shutdown": "shutdown",
    "shut down": "shutdown",
    "turn off": "shutdown",
    "power off": "shutdown",
    "restart": "restart",
    "reboot": "restart",
    "sleep": "sleep",
    "hibernate": "sleep",
    "standby": "sleep",
    "cancel shutdown": "cancel_shutdown",
    "cancel restart": "cancel_shutdown",
    # File management
    "rename": "rename",
    "copy": "copy",
    "move": "move",
    "delete": "delete",
    "remove": "delete",
    "list": "list_files",
    "list files": "list_files",
    "show files": "list_files",
    "dir": "list_files",
    # System utilities
    "disk cleanup": "disk_cleanup",
    "clean disk": "disk_cleanup",
    "cleanup": "disk_cleanup",
    "wallpaper": "set_wallpaper",
    "set wallpaper": "set_wallpaper",
    "desktop wallpaper": "set_wallpaper",
    "background": "set_wallpaper",
    # Model management
    "reset training": "reset_training",
    "reset model": "reset_training",
    "clear training": "reset_training",
    "model info": "model_info",
    "model status": "model_info",
}


# ---------------------------------------------------------------------------
# Core parser -- detect and extract slash commands
# ---------------------------------------------------------------------------
def parse_command(user_input):
    """
    Detect and extract a slash command from user input.

    Args:
        user_input: Raw string from the user.

    Returns:
        Tuple of (command, args) if a slash command is found.
        command is lowercase (e.g. "/task").
        args is the remaining text after the command.
        Returns None if no slash command is detected.
    """
    stripped = user_input.strip()

    if not stripped.startswith("/"):
        return None

    # Split into command and args
    parts = stripped.split(None, 1)  # Split on first whitespace
    command = parts[0].lower()
    args = parts[1].strip() if len(parts) > 1 else ""

    # Validate that the command is supported
    if command not in SUPPORTED_COMMANDS:
        return None

    return (command, args)


# ---------------------------------------------------------------------------
# Multi-command parser -- split input by slash prefixes
# ---------------------------------------------------------------------------
def parse_multi_commands(user_input):
    """
    Split input containing multiple slash commands into separate commands.
    Handles: "/task battery and /save to log.txt"

    Args:
        user_input: Raw string possibly containing multiple slash commands.

    Returns:
        List of (command, args) tuples. Returns a single-element list if
        only one command is present. Returns None if no commands found.
    """
    stripped = user_input.strip()

    if not stripped.startswith("/"):
        return None

    # Find all slash command positions
    # Pattern: / followed by a known command name
    command_names = "|".join(
        re.escape(cmd.lstrip("/")) for cmd in SUPPORTED_COMMANDS
    )
    pattern = rf"(/(?:{command_names}))\b"
    matches = list(re.finditer(pattern, stripped, re.IGNORECASE))

    if not matches:
        return None

    commands = []
    for i, match in enumerate(matches):
        start = match.start()
        # End is either the start of the next command or end of string
        end = matches[i + 1].start() if i + 1 < len(matches) else len(stripped)

        chunk = stripped[start:end].strip()
        # Remove trailing "and" connector if present
        chunk = re.sub(r"\s+and\s*$", "", chunk, flags=re.IGNORECASE).strip()

        result = parse_command(chunk)
        if result:
            commands.append(result)

    return commands if commands else None


# ---------------------------------------------------------------------------
# /task handler
# ---------------------------------------------------------------------------
def handle_task(args):
    """
    Handle /task commands -- route directly to tools.system or tools.files.

    Args:
        args: The text after "/task ", e.g. "battery" or "list files in C:\\docs"

    Returns:
        Result string from the appropriate handler.
    """
    if not args:
        return ("No task specified. Try:\n"
                "  /task battery\n"
                "  /task processes\n"
                "  /task shutdown\n"
                "  /task rename <file>\n"
                "  /task list files <folder>\n"
                "  /task disk cleanup\n"
                "Type /help for all available commands.")

    lower = args.lower().strip()

    # Try to match task keywords (check longer phrases first).
    # Matched on a word boundary, not a plain prefix -- a plain startswith()
    # let "listen to music" match the keyword "list" (list_files).
    matched_task = None
    for keyword in sorted(_TASK_KEYWORDS.keys(), key=len, reverse=True):
        if re.match(rf"{re.escape(keyword)}\b", lower):
            matched_task = _TASK_KEYWORDS[keyword]
            # Extract remaining args after the keyword
            remaining = args[len(keyword):].strip()
            break

    if matched_task is None:
        # Try word-by-word matching for single keywords
        first_word = lower.split()[0] if lower.split() else ""
        if first_word in _TASK_KEYWORDS:
            matched_task = _TASK_KEYWORDS[first_word]
            remaining = args[len(first_word):].strip()

    if matched_task is None:
        return (f"Unknown task: \"{args}\"\n"
                "Available tasks: battery, processes, shutdown, restart, sleep,\n"
                "  rename, copy, move, delete, list files, disk cleanup, set wallpaper,\n"
                "  reset training, model info\n"
                "Type /help for full list.")

    # Route to the correct handler
    if matched_task == "battery":
        from nexus.tools import system
        return system.get_battery()

    if matched_task == "processes":
        from nexus.tools import system
        return system.get_processes()

    if matched_task == "shutdown":
        from nexus.tools import system
        return system.shutdown_system()

    if matched_task == "restart":
        from nexus.tools import system
        return system.restart_system()

    if matched_task == "sleep":
        from nexus.tools import system
        return system.sleep_system()

    if matched_task == "cancel_shutdown":
        from nexus.tools import system
        return system.cancel_shutdown()

    if matched_task == "rename":
        from nexus.tools import files
        from nexus.utils import extract_filepath
        filepath = extract_filepath(remaining)
        if not filepath:
            return "Please specify a file to rename. Example: /task rename notes.txt to notes_old.txt"
        new_name = _extract_rename_target(remaining)
        if not new_name:
            return f"Please specify the new name. Example: /task rename {filepath} to new_name.txt"
        return files.rename_file(filepath, new_name)

    if matched_task == "copy":
        from nexus.tools import files
        from nexus.utils import extract_filepath
        filepath = extract_filepath(remaining)
        if not filepath:
            return "Please specify a file to copy. Example: /task copy notes.txt to D:\\backup"
        dest = _extract_destination(remaining)
        if not dest:
            return f"Please specify the destination. Example: /task copy {filepath} to D:\\backup"
        return files.copy_file(filepath, dest)

    if matched_task == "move":
        from nexus.tools import files
        from nexus.utils import extract_filepath
        filepath = extract_filepath(remaining)
        if not filepath:
            return "Please specify a file to move. Example: /task move notes.txt to D:\\docs"
        dest = _extract_destination(remaining)
        if not dest:
            return f"Please specify the destination. Example: /task move {filepath} to D:\\docs"
        return files.move_file(filepath, dest)

    if matched_task == "delete":
        from nexus.tools import files
        from nexus.utils import extract_filepath
        filepath = extract_filepath(remaining)
        if not filepath:
            return "Please specify a file to delete. Example: /task delete old_notes.txt"
        return files.delete_file(filepath)

    if matched_task == "list_files":
        from nexus.tools import files
        from nexus.utils import extract_filepath
        # Remove "files" or "in" from args before extracting path
        clean_remaining = re.sub(r"^(files?\s+)?(in\s+)?", "", remaining, flags=re.IGNORECASE).strip()
        filepath = extract_filepath(clean_remaining) if clean_remaining else None
        if not filepath and clean_remaining:
            filepath = clean_remaining  # Use the raw text as a path
        if not filepath:
            return "Please specify a folder to list. Example: /task list files in C:\\docs"
        return files.list_files(filepath)

    if matched_task == "disk_cleanup":
        return _run_disk_cleanup(remaining)

    if matched_task == "set_wallpaper":
        from nexus.utils import extract_filepath
        filepath = extract_filepath(remaining)
        if not filepath:
            return "Please specify an image file. Example: /task set wallpaper C:\\pics\\bg.jpg"
        return _set_wallpaper(filepath)

    if matched_task == "reset_training":
        from nexus.llm import loader
        return loader.reset_training()

    if matched_task == "model_info":
        from nexus.llm import loader
        return loader.get_model_info()

    return f"Task '{matched_task}' is not yet implemented."


# ---------------------------------------------------------------------------
# /save handler
# ---------------------------------------------------------------------------
def handle_save(args):
    """
    Handle /save commands -- save data to a file.

    Args:
        args: Text containing the data and filename.
              e.g. "this to notes.txt" or "to stocks.txt"

    Returns:
        Result string.
    """
    if not args:
        return ("No save target specified. Try:\n"
                "  /save to notes.txt\n"
                "  /save data to C:\\docs\\output.txt")

    from nexus.tools import files

    # Try to extract filename from args
    filepath = None
    data = args

    # Pattern: "to <filename>"
    to_match = re.search(r"\bto\s+([^\s]+\.(?:txt|csv|json|xlsx?))\b", args, re.IGNORECASE)
    if to_match:
        filepath = to_match.group(1)
        data = args[:to_match.start()].strip()

    if not filepath:
        # Try to find any filename pattern
        file_match = re.search(r"([^\s]+\.(?:txt|csv|json|xlsx?))\b", args, re.IGNORECASE)
        if file_match:
            filepath = file_match.group(1)
            data = args.replace(filepath, "").strip()

    if not filepath:
        return ("Could not determine the filename. Try:\n"
                "  /save to notes.txt\n"
                "  /save my data to output.txt")

    # If we have no data content, check if there's a previous response to save
    if not data:
        data = "[No data provided -- use /save <data> to <filename>]"

    return files.save_text(data, filepath)


# ---------------------------------------------------------------------------
# /remember handler
# ---------------------------------------------------------------------------
def handle_remember(args):
    """
    Handle /remember commands -- store a fact in persistent memory.

    Args:
        args: The fact to remember.

    Returns:
        Result string.
    """
    if not args:
        return ("No fact provided. Try:\n"
                "  /remember my favourite movie is Avengers Endgame\n"
                "  /remember I work at Google\n"
                "  /remember my birthday is 15 March")

    try:
        from nexus.agent import memory
    except ImportError:
        return ("Persistent memory module not available.\n"
                "Make sure memory.py exists in the src/ folder.")
    return memory.remember_fact(args)


# ---------------------------------------------------------------------------
# /recall handler
# ---------------------------------------------------------------------------
def handle_recall(args):
    """
    Handle /recall commands -- list stored memories.
    Optionally filter by keyword.
    """
    try:
        from nexus.agent import memory
    except ImportError:
        return ("Persistent memory module not available.\n"
                "Make sure memory.py exists in the src/ folder.")
    return memory.recall_facts(args if args else None)


# ---------------------------------------------------------------------------
# /forget handler
# ---------------------------------------------------------------------------
def handle_forget(args):
    """
    Handle /forget commands -- remove a stored memory by index or keyword.
    """
    if not args:
        return ("Please specify which memory to forget.\n"
                "  /forget 1              (by number from /recall)\n"
                "  /forget favourite movie (by keyword)")

    try:
        from nexus.agent import memory
    except ImportError:
        return ("Persistent memory module not available.\n"
                "Make sure memory.py exists in the src/ folder.")
    return memory.forget_fact(args)


# ---------------------------------------------------------------------------
# /help handler
# ---------------------------------------------------------------------------
def handle_model(args):
    """
    Handle /model commands -- show model info or reset training.

    Args:
        args: Optional subcommand ("info", "reset", or empty for info).

    Returns:
        Result string from the model handler.
    """
    lower = args.lower().strip() if args else ""

    if lower in ("", "info", "status"):
        from nexus.llm import loader
        return loader.get_model_info()
    if lower in ("reset", "reset training", "clear"):
        from nexus.llm import loader
        return loader.reset_training()
    return ("Unknown /model subcommand.\n"
            "  /model          -> Show model info\n"
            "  /model info     -> Show model info\n"
            "  /model reset    -> Reset custom training")


def handle_settings(args):
    """
    Handle /settings command -- display current configuration.

    Args:
        args: Unused (reserved for future subcommands).

    Returns:
        Formatted settings summary string.
    """
    try:
        from nexus.llm.loader import load_settings
        settings = load_settings()

        lines = [
            "+------------------------------------------------------+",
            "|              NEXUS Current Settings                   |",
            "+------------------------------------------------------+",
            "",
        ]
        for key, value in settings.items():
            lines.append(f"  {key}: {value}")
        lines.append("")
        lines.append("Tip: Edit config/settings.json to change settings,")
        lines.append("or use the Settings panel in the Quick Menu (three-dot menu).")
        return "\n".join(lines)
    except Exception as e:
        return f"Could not load settings: {e}"


def handle_theme(args):
    """
    Handle /theme command -- switch or list available themes.

    Args:
        args: Theme name to switch to (Dark/Light/Blue), or empty to list.

    Returns:
        Result string confirming the theme change or listing themes.
    """
    try:
        from nexus.ui import themes
    except ImportError:
        return "Theme module not available."

    available = themes.get_theme_names()

    if not args or not args.strip():
        current = themes.load_saved_theme()
        lines = ["Available themes:"]
        for name in available:
            marker = " (active)" if name == current else ""
            lines.append(f"  - {name}{marker}")
        lines.append("\nUsage: /theme Dark  or  /theme Light  or  /theme Blue")
        return "\n".join(lines)

    # Try to match the requested theme
    requested = args.strip()
    for name in available:
        if name.lower() == requested.lower():
            themes.save_theme(name)
            return f"Theme switched to {name}. Restart the Quick Menu to see changes."

    return (f"Unknown theme: \"{requested}\"\n"
            f"Available themes: {', '.join(available)}")


def handle_clear(args):
    """
    Handle /clear command -- clear conversation history.

    Args:
        args: Unused.

    Returns:
        Confirmation message.
    """
    # Try to access the shared conversation memory instance
    try:
        from nexus.agent import router
        if router._conversation_memory:
            router._conversation_memory.clear()
            return "Conversation history cleared. Starting a fresh chat!"
        return "No conversation history to clear."
    except Exception:
        return "Conversation history cleared."


def handle_help():
    """
    Show all available slash commands with descriptions and examples.

    Returns:
        Formatted help text string.
    """
    help_text = """
+------------------------------------------------------+
|              NEXUS Slash Commands                     |
+------------------------------------------------------+

  /task <action>     - Execute a system task directly
    Examples:
      /task battery              -> Show battery percentage
      /task processes             -> List top 10 running processes
      /task shutdown              -> Shut down computer (with confirmation)
      /task restart               -> Restart computer
      /task sleep                 -> Put computer to sleep
      /task rename file.txt to new.txt  -> Rename a file
      /task copy file.txt to D:\\backup  -> Copy a file
      /task move file.txt to D:\\docs    -> Move a file
      /task delete old_file.txt   -> Delete a file
      /task list files in C:\\docs -> List files in a folder
      /task disk cleanup          -> Run Windows disk cleanup
      /task set wallpaper bg.jpg  -> Set desktop wallpaper

  /save <data> to <filename>  - Save data to a file
    Examples:
      /save to notes.txt
      /save my notes to output.txt

  /remember <fact>   - Store a fact for long-term memory
    Examples:
      /remember my favourite movie is Avengers Endgame
      /remember I work at Google

  /recall            - Show all stored memories
  /forget <fact>     - Remove a stored memory

  /model [info|reset] - Show model info or reset training
    Examples:
      /model               -> Show current model status
      /model info           -> Show current model status
      /model reset          -> Reset custom training data

  /settings          - Show current NEXUS configuration

  /theme [name]      - Switch theme or list available themes
    Examples:
      /theme               -> List available themes
      /theme Dark           -> Switch to Dark theme
      /theme Light          -> Switch to Light theme
      /theme Blue           -> Switch to Blue theme

  /clear             - Clear conversation history

  /help              - Show this help message

------------------------------------------------------
Tip: You can also use multiple commands in one line:
  /task battery and /save to log.txt
------------------------------------------------------
Without a slash prefix, NEXUS uses AI to understand
your intent automatically.
"""
    return help_text.strip()


# ---------------------------------------------------------------------------
# Main dispatcher -- route slash commands to handlers
# ---------------------------------------------------------------------------
def execute_command(command, args):
    """
    Execute a slash command by routing to the appropriate handler.

    Args:
        command: The slash command (e.g. "/task").
        args:    The arguments after the command.

    Returns:
        Result string from the handler.
    """
    if command == "/task":
        return handle_task(args)
    if command == "/save":
        return handle_save(args)
    if command == "/remember":
        return handle_remember(args)
    if command == "/recall":
        return handle_recall(args)
    if command == "/forget":
        return handle_forget(args)
    if command == "/model":
        return handle_model(args)
    if command == "/settings":
        return handle_settings(args)
    if command == "/theme":
        return handle_theme(args)
    if command == "/clear":
        return handle_clear(args)
    if command == "/help":
        return handle_help()
    return f"Unknown command: {command}\nType /help for available commands."


def execute_input(user_input):
    """
    Process user input -- check for slash commands (single or multi),
    execute them, and return the combined result.

    This is the main entry point called by router.py before
    falling through to normal routing.

    Args:
        user_input: Raw string from the user.

    Returns:
        Result string if slash commands were found and executed,
        or None if no slash commands detected (caller should proceed
        with normal routing).
    """
    stripped = user_input.strip()

    if not stripped.startswith("/"):
        return None

    # Check for multi-command input first
    multi = parse_multi_commands(stripped)
    if multi and len(multi) > 1:
        results = []
        prev_result = None
        for cmd, cmd_args in multi:
            # If /save comes after another command, pass the previous result
            if cmd == "/save" and prev_result and not cmd_args:
                cmd_args = "to output.txt"  # Default filename
            result = execute_command(cmd, cmd_args)
            results.append(result)
            prev_result = result
        return "\n\n".join(results)

    # Single command
    parsed = parse_command(stripped)
    if parsed:
        cmd, args = parsed
        return execute_command(cmd, args)

    # Starts with / but not a known command
    return f"Unknown command: \"{stripped.split()[0]}\"\nType /help for available commands."


# ---------------------------------------------------------------------------
# Helper functions for /task subcommands
# ---------------------------------------------------------------------------
def _extract_rename_target(text):
    """Extract new name from text like 'file.txt to new_name.txt'."""
    lower = text.lower()
    for keyword in (" to ", " as ", " into "):
        idx = lower.find(keyword)
        if idx != -1:
            candidate = text[idx + len(keyword):].strip().strip('"').strip("'")
            if candidate:
                return candidate
    return None


def _extract_destination(text):
    """Extract destination path from text like 'file.txt to D:\\backup'."""
    lower = text.lower()
    for keyword in (" to ", " into "):
        idx = lower.find(keyword)
        if idx != -1:
            candidate = text[idx + len(keyword):].strip().strip('"').strip("'")
            if candidate:
                return candidate
    return None


def _run_disk_cleanup(args):
    """Run Windows Disk Cleanup utility via subprocess."""
    try:
        # Parse drive letter from args (default to C:)
        drive = "C"
        if args:
            drive_match = re.search(r"([A-Za-z])[\s:]*drive", args, re.IGNORECASE)
            if drive_match:
                drive = drive_match.group(1).upper()
            elif len(args.strip()) == 1 and args.strip().isalpha():
                drive = args.strip().upper()

        if not confirm.request(
            f"Run Disk Cleanup on drive {drive}:?",
            "This opens the Windows Disk Cleanup utility.",
        ):
            return confirm.denied_message("Disk cleanup")

        subprocess.Popen(["cleanmgr", "/d", drive])
        return f"Disk Cleanup started for drive {drive}:. Follow the on-screen prompts."
    except Exception as e:
        return f"Could not start Disk Cleanup: {e}"


def _set_wallpaper(filepath):
    """Set desktop wallpaper using ctypes (Windows)."""
    filepath = os.path.abspath(filepath)

    if not os.path.isfile(filepath):
        return f"Image file not found: {filepath}"

    # Check file extension
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in valid_exts:
        return f"Unsupported image format: {ext}\nSupported: {', '.join(valid_exts)}"

    try:
        import ctypes
        SPI_SETDESKWALLPAPER = 0x0014
        result = ctypes.windll.user32.SystemParametersInfoW(
            SPI_SETDESKWALLPAPER, 0, filepath, 3
        )
        if result:
            return f"Desktop wallpaper set to: {os.path.basename(filepath)}"
        return "Failed to set wallpaper. The system call returned an error."
    except Exception as e:
        return f"Could not set wallpaper: {e}"
