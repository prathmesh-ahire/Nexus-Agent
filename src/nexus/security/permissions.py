"""
permissions.py — NEXUS Permission Guard
Controls which folders NEXUS is allowed to access. Every file operation
checks against this whitelist before proceeding. Prevents accidental or
malicious access to sensitive system directories.
"""

import json
import os

from nexus import config

# ---------------------------------------------------------------------------
# Path to the permissions config file
# ---------------------------------------------------------------------------
CONFIG_PATH = str(config.PERMISSIONS_FILE)


# ---------------------------------------------------------------------------
# Load / Save permissions
# ---------------------------------------------------------------------------
def load_permissions():
    """
    Read config/permissions.json and return the list of allowed folder paths.
    Returns an empty list (and prints a warning) if the file is missing or invalid.
    """
    if not os.path.isfile(CONFIG_PATH):
        print("[WARNING] permissions.json not found. No folders are currently allowed.")
        return []

    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("allowed_paths", [])
    except (OSError, json.JSONDecodeError) as e:
        print(f"[WARNING] Could not read permissions.json ({e}). Returning empty list.")
        return []


def save_permissions(paths_list):
    """
    Write the given list of allowed paths back to config/permissions.json.
    Only real, existing directories are saved — invalid paths are silently dropped.
    """
    # Filter to only paths that are actual directories
    valid_paths = [p for p in paths_list if os.path.isdir(p)]

    if len(valid_paths) < len(paths_list):
        skipped = len(paths_list) - len(valid_paths)
        print(f"[WARNING] {skipped} invalid path(s) were removed before saving.")

    data = {"allowed_paths": valid_paths}

    try:
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError as e:
        print(f"[ERROR] Could not save permissions.json: {e}")


# ---------------------------------------------------------------------------
# Access checks
# ---------------------------------------------------------------------------
def is_allowed(filepath):
    """
    Return True if `filepath` falls inside one of the approved directories.
    Blocks directory traversal attempts (paths containing '..').
    Works for both files and folders.
    """
    # Block directory traversal
    if ".." in filepath:
        return False

    # Normalise the incoming path to an absolute form
    filepath = os.path.abspath(filepath)

    allowed_paths = load_permissions()

    if not allowed_paths:
        return False

    for allowed in allowed_paths:
        allowed = os.path.abspath(allowed)
        try:
            # os.path.commonpath returns the longest common sub-path.
            # If that common path equals the allowed path, then filepath
            # is inside (or equal to) the allowed directory.
            common = os.path.commonpath([allowed, filepath])
            if common == allowed:
                return True
        except ValueError:
            # commonpath raises ValueError on Windows if paths are on
            # different drives (e.g. C:\ vs D:\) — not a match.
            continue

    return False


# ---------------------------------------------------------------------------
# Add / remove permissions
# ---------------------------------------------------------------------------
def add_permission(folder_path):
    """
    Add a folder to the allowed list (if it exists and isn't already listed).
    """
    folder_path = os.path.abspath(folder_path)

    if not os.path.isdir(folder_path):
        print(f"[ERROR] Folder does not exist: {folder_path}")
        return False

    current = load_permissions()

    # Normalise existing paths for comparison
    normalised = [os.path.abspath(p) for p in current]

    if folder_path in normalised:
        print(f"[INFO] Folder is already allowed: {folder_path}")
        return True

    current.append(folder_path)
    save_permissions(current)
    print(f"[OK] Access granted for: {folder_path}")
    return True


def remove_permission(folder_path):
    """
    Remove a folder from the allowed list.
    """
    folder_path = os.path.abspath(folder_path)
    current = load_permissions()
    normalised = [os.path.abspath(p) for p in current]

    if folder_path not in normalised:
        print(f"[INFO] Folder was not in the allowed list: {folder_path}")
        return False

    # Remove matching path (compare normalised forms)
    updated = [p for p in current if os.path.abspath(p) != folder_path]
    save_permissions(updated)
    print(f"[OK] Access revoked for: {folder_path}")
    return True
