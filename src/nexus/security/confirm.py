"""
confirm.py — Confirmation gateway for destructive actions.

Destructive operations used to call ``input()`` directly. That worked in the
old CMD interface, but the GUI runs them on a worker thread with no console
attached, so the prompt never reached the user and the action either hung or
raised. Every destructive path now routes through ``request()`` instead, and
the active frontend registers how to ask.

Fails closed: if no handler is registered (headless runs, tests, scripts),
``request()`` denies. Nothing destructive happens without an explicit yes.
"""

from __future__ import annotations

from collections.abc import Callable

# Signature: (title, details) -> bool
ConfirmHandler = Callable[[str, str], bool]

_handler: ConfirmHandler | None = None


def set_handler(handler: ConfirmHandler | None) -> None:
    """Register the frontend's confirmation prompt. Pass None to unregister."""
    global _handler
    _handler = handler


def has_handler() -> bool:
    """True if a frontend has registered a way to ask the user."""
    return _handler is not None


def request(title: str, details: str = "") -> bool:
    """
    Ask the user to approve a destructive action.

    Returns True only on an explicit yes. With no handler registered this
    returns False, so callers must treat a False result as "cancelled".
    """
    if _handler is None:
        return False
    try:
        return bool(_handler(title, details))
    except Exception:
        # A broken frontend handler must never be read as approval.
        return False


def denied_message(action: str) -> str:
    """Standard response when a confirmation was declined or unavailable."""
    if not has_handler():
        return (
            f"{action} needs confirmation, but no confirmation prompt is "
            "available in this context. Run NEXUS through the Quick Menu to "
            "approve destructive actions."
        )
    return f"{action} cancelled."


def console_handler(title: str, details: str = "") -> bool:
    """
    A confirmation handler for terminal contexts (reset.bat, scripts, manual runs).

    Frontends with a console register this explicitly; it is never the default,
    because the GUI runs under pythonw with no stdin and would raise.
    """
    print()
    print(f"  {title}")
    for line in details.splitlines():
        if line.strip():
            print(f"  {line}")
    try:
        return input("  Proceed? (y/n): ").strip().lower() in {"y", "yes"}
    except (EOFError, RuntimeError):
        # No usable stdin -- treat as a refusal rather than crashing.
        return False
