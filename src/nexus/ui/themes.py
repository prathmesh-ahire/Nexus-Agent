"""
themes.py -- NEXUS Theme System
Defines color themes for the Quick Menu GUI and provides
functions to apply and persist theme selections.

Phase 30 of the NEXUS implementation plan.
"""

import json
import os

from nexus import config

# ---------------------------------------------------------------------------
# Theme definitions
# ---------------------------------------------------------------------------
THEMES = {
    "Dark": {
        "bg_dark":       "#0A1931",
        "bg_medium":     "#1A3D63",
        "bg_chat":       "#0D2137",
        "fg_text":       "#F6FAFD",
        "fg_dim":        "#B3CFE5",
        "accent":        "#4A7FA7",
        "accent_hover":  "#5E9CC7",
        "error_red":     "#f38ba8",
        "success_grn":   "#a6e3a1",
        "thinking_ylw":  "#f9e2af",
        "pill_bg":       "#162D4A",
        "pill_hover":    "#1E4D7A",
        "menu_bg":       "#122340",
        "menu_hover":    "#1A3D63",
        "border":        "#1E4D7A",
    },
    "Light": {
        "bg_dark":       "#F0F4F8",
        "bg_medium":     "#E2E8F0",
        "bg_chat":       "#FFFFFF",
        "fg_text":       "#1A202C",
        "fg_dim":        "#718096",
        "accent":        "#3182CE",
        "accent_hover":  "#2B6CB0",
        "error_red":     "#E53E3E",
        "success_grn":   "#38A169",
        "thinking_ylw":  "#D69E2E",
        "pill_bg":       "#E2E8F0",
        "pill_hover":    "#CBD5E0",
        "menu_bg":       "#FFFFFF",
        "menu_hover":    "#EDF2F7",
        "border":        "#CBD5E0",
    },
    "Blue": {
        "bg_dark":       "#0F172A",
        "bg_medium":     "#1E293B",
        "bg_chat":       "#0F172A",
        "fg_text":       "#F1F5F9",
        "fg_dim":        "#94A3B8",
        "accent":        "#3B82F6",
        "accent_hover":  "#60A5FA",
        "error_red":     "#FB7185",
        "success_grn":   "#4ADE80",
        "thinking_ylw":  "#FACC15",
        "pill_bg":       "#1E293B",
        "pill_hover":    "#334155",
        "menu_bg":       "#1E293B",
        "menu_hover":    "#334155",
        "border":        "#334155",
    },
}

DEFAULT_THEME = "Dark"

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
_SETTINGS_FILE = str(config.SETTINGS_FILE)


# ---------------------------------------------------------------------------
# Get / set theme
# ---------------------------------------------------------------------------
def get_theme(name=None):
    """
    Return the theme dict for the given name.
    If name is None, loads the saved theme from settings.

    Args:
        name: Theme name ('Dark', 'Light', 'Blue') or None for saved.

    Returns:
        Theme dict with color keys.
    """
    if name is None:
        name = load_saved_theme()
    return THEMES.get(name, THEMES[DEFAULT_THEME])


def get_theme_names():
    """Return list of available theme names."""
    return list(THEMES.keys())


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def load_saved_theme():
    """
    Load the saved theme name from config/settings.json.

    Returns:
        Theme name string, or DEFAULT_THEME if not saved.
    """
    if not os.path.isfile(_SETTINGS_FILE):
        return DEFAULT_THEME

    try:
        with open(_SETTINGS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("theme", DEFAULT_THEME)
    except (json.JSONDecodeError, OSError):
        return DEFAULT_THEME


def save_theme(name):
    """
    Save the selected theme name to config/settings.json.

    Args:
        name: Theme name to save.

    Returns:
        True if saved, False on error.
    """
    data = {}
    if os.path.isfile(_SETTINGS_FILE):
        try:
            with open(_SETTINGS_FILE, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            data = {}

    data["theme"] = name

    try:
        with open(_SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return True
    except OSError:
        return False
