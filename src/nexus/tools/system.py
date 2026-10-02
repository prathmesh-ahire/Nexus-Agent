"""
system.py — NEXUS System Tasks & Task Queue
Provides functions to query system information like battery status
and running processes, and to execute safe system commands (shutdown,
restart, sleep) with confirmation prompts. Uses psutil and subprocess.

Also provides task queue functionality for batching multiple tasks
and executing them sequentially, with persistence to tasks_queue.json.

Originally split across system.py and system.py,
now unified into a single module for cleaner organisation.
"""

import datetime
import json
import os
import subprocess

import psutil

from nexus import config
from nexus.security import confirm

# ===========================================================================
# Part 1: System Information & Commands
# ===========================================================================

# ---------------------------------------------------------------------------
# Battery status
# ---------------------------------------------------------------------------
def get_battery():
    """
    Return a formatted string showing battery percentage and charging status.
    Handles desktop PCs that have no battery gracefully.
    """
    battery = psutil.sensors_battery()

    if battery is None:
        return "No battery detected. This appears to be a desktop PC (always on AC power)."

    percent = battery.percent
    plugged = battery.power_plugged
    status = "Charging" if plugged else "Discharging"

    # Estimate time remaining (only meaningful when discharging)
    time_info = ""
    if not plugged and battery.secsleft not in (
        psutil.POWER_TIME_UNLIMITED,
        psutil.POWER_TIME_UNKNOWN,
    ):
        remaining = str(datetime.timedelta(seconds=battery.secsleft))
        time_info = f" | Estimated time remaining: {remaining}"

    return f"Battery: {percent}% | Status: {status}{time_info}"


# ---------------------------------------------------------------------------
# Running processes (top 10 by memory)
# ---------------------------------------------------------------------------
def get_processes():
    """
    Return a formatted string listing the top 10 processes by RAM usage.
    Handles AccessDenied exceptions for protected system processes.
    """
    processes = []

    for proc in psutil.process_iter(["pid", "name", "memory_info"]):
        try:
            info = proc.info
            mem = info["memory_info"]
            if mem is not None:
                processes.append({
                    "pid": info["pid"],
                    "name": info["name"],
                    "ram_bytes": mem.rss,
                })
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            # Skip processes we can't access
            continue

    # Sort by RAM usage descending and take top 10
    processes.sort(key=lambda p: p["ram_bytes"], reverse=True)
    top10 = processes[:10]

    if not top10:
        return "Could not retrieve process information."

    lines = ["Top 10 Processes by Memory Usage:", ""]
    for i, p in enumerate(top10, 1):
        ram_mb = p["ram_bytes"] / (1024 * 1024)
        lines.append(
            f"  {i:2d}. {p['name']:<30s} | PID: {p['pid']:<8d} | RAM: {ram_mb:.1f} MB"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# System commands — shutdown, restart, sleep (Phase 17)
# ---------------------------------------------------------------------------
def shutdown_system():
    """
    Shut down the computer after a 30-second delay.
    Asks for user confirmation before executing.
    The user can cancel with 'shutdown /a' within 30 seconds.
    """
    if not confirm.request(
        "Shut down this computer?",
        "The computer will shut down in 30 seconds. "
        "You can cancel within that window with 'cancel shutdown'.",
    ):
        return confirm.denied_message("Shutdown")

    try:
        subprocess.run(["shutdown", "/s", "/t", "30"], check=True)
        return ("Shutdown initiated. Your computer will shut down in 30 seconds.\n"
                "To cancel, type 'cancel shutdown' or run: shutdown /a")
    except subprocess.CalledProcessError as e:
        return f"Failed to initiate shutdown: {e}"
    except Exception as e:
        return f"Error: {e}"


def restart_system():
    """
    Restart the computer after a 30-second delay.
    Asks for user confirmation before executing.
    """
    if not confirm.request(
        "Restart this computer?",
        "The computer will restart in 30 seconds. "
        "You can cancel within that window with 'cancel shutdown'.",
    ):
        return confirm.denied_message("Restart")

    try:
        subprocess.run(["shutdown", "/r", "/t", "30"], check=True)
        return ("Restart initiated. Your computer will restart in 30 seconds.\n"
                "To cancel, type 'cancel shutdown' or run: shutdown /a")
    except subprocess.CalledProcessError as e:
        return f"Failed to initiate restart: {e}"
    except Exception as e:
        return f"Error: {e}"


def sleep_system():
    """
    Put the computer to sleep immediately.
    Asks for user confirmation before executing.
    """
    if not confirm.request("Put this computer to sleep?", "The computer will sleep immediately."):
        return confirm.denied_message("Sleep")

    try:
        subprocess.run(
            ["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"],
            check=True,
        )
        return "Sleep command sent."
    except subprocess.CalledProcessError as e:
        return f"Failed to put system to sleep: {e}"
    except Exception as e:
        return f"Error: {e}"


def cancel_shutdown():
    """
    Cancel a pending shutdown or restart using 'shutdown /a'.
    """
    try:
        subprocess.run(["shutdown", "/a"], check=True)
        return "Pending shutdown/restart has been cancelled."
    except subprocess.CalledProcessError:
        return "No pending shutdown or restart to cancel."
    except Exception as e:
        return f"Error: {e}"


# ===========================================================================
# Part 2: Task Queue (originally in system.py)
# ===========================================================================

# ---------------------------------------------------------------------------
# Path to the queue file
# ---------------------------------------------------------------------------
QUEUE_FILE = str(config.QUEUE_FILE)


# ---------------------------------------------------------------------------
# Write / Read queue
# ---------------------------------------------------------------------------
def write_queue(tasks_list):
    """
    Save a list of task dicts to tasks_queue.json.
    Each task: {"action": "ACTION:QA", "params": "what is python", "status": "pending"}
    """
    try:
        with open(QUEUE_FILE, "w", encoding="utf-8") as f:
            json.dump(tasks_list, f, indent=2)
    except OSError as e:
        print(f"[ERROR] Could not write task queue: {e}")


def read_queue():
    """
    Read and return the task queue list from tasks_queue.json.
    Returns an empty list if the file doesn't exist or is invalid.
    """
    if not os.path.isfile(QUEUE_FILE):
        return []

    try:
        with open(QUEUE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return []


def _generate_qa(question):
    """Answer a general question using the model (replaces removed files.answer_question)."""
    from nexus.llm.loader import generate
    if not question or not question.strip():
        return "Please ask a question."
    prompt = "Answer the following question in 2-3 sentences. Be brief and direct:\n\n" + question
    return generate(prompt, max_tokens=200)


# ---------------------------------------------------------------------------
# Execute all pending tasks
# ---------------------------------------------------------------------------
def execute_queue():
    """
    Run all pending tasks in order. Updates status to 'done' after each.
    Clears the queue file after all tasks are complete.
    """
    tasks = read_queue()

    if not tasks:
        print("No tasks in queue.")
        return

    # Lazy import to avoid circular imports
    from nexus.tools import files

    # Map action strings to handler functions
    # Note: these task helpers are local to this module
    action_map = {
        "ACTION:SUMMARISE_FILE": lambda params: files.summarise_file(params),
        "ACTION:SUMMARISE_FOLDER": lambda params: files.summarise_folder(params),
        "ACTION:QA": lambda params: _generate_qa(params),
        "ACTION:BATTERY": lambda params: get_battery(),
        "ACTION:PROCESSES": lambda params: get_processes(),
    }

    total = len(tasks)

    for i, task in enumerate(tasks, 1):
        action = task.get("action", "ACTION:UNKNOWN")
        params = task.get("params", "")

        print(f"\nExecuting task {i} of {total}: [{action}]")
        print("-" * 40)

        handler = action_map.get(action)
        if handler:
            result = handler(params)
            print(result)
        else:
            print("Unknown action — skipped.")

        # Mark as done
        task["status"] = "done"
        write_queue(tasks)  # persist progress

    # Clear the queue after all tasks are done
    clear_queue()
    print("\nAll tasks complete.")


# ---------------------------------------------------------------------------
# Clear the queue
# ---------------------------------------------------------------------------
def clear_queue():
    """Reset the queue to an empty list."""
    write_queue([])
    print("Task queue cleared.")
