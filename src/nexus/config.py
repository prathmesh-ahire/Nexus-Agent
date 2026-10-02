"""
config.py — Single source of truth for filesystem paths and settings.

Every other module imports paths from here. Before this existed, PROJECT_ROOT
was recomputed independently in seven modules and the settings.json path was
duplicated in three, so moving any file silently broke unrelated features.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# config.py lives at <root>/src/nexus/config.py — the project root is two up.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

CONFIG_DIR: Path = PROJECT_ROOT / "config"
MODEL_DIR: Path = PROJECT_ROOT / "model"
DATASETS_DIR: Path = PROJECT_ROOT / "datasets"
LORA_WEIGHTS_DIR: Path = PROJECT_ROOT / "lora-weights"
RAG_INDEX_DIR: Path = PROJECT_ROOT / "rag-index"

SETTINGS_FILE: Path = CONFIG_DIR / "settings.json"
PERMISSIONS_FILE: Path = CONFIG_DIR / "permissions.json"
API_KEYS_FILE: Path = CONFIG_DIR / "api_keys.json"
API_KEYS_TEMPLATE: Path = CONFIG_DIR / "api_keys.template.json"
MEMORY_FILE: Path = CONFIG_DIR / "user_memory.json"
QUEUE_FILE: Path = PROJECT_ROOT / "tasks_queue.json"

DEFAULT_MODEL_NAME = "qwen2.5-3b-instruct-q4_k_m.gguf"

DEFAULT_SETTINGS: dict[str, Any] = {
    "model_path": f"model/{DEFAULT_MODEL_NAME}",
    "max_tokens": 512,
    "n_ctx": 4096,
    "n_threads": 4,
    "auto_threads": True,
    "theme": "dark",
}


# ---------------------------------------------------------------------------
# Settings access
# ---------------------------------------------------------------------------
def load_settings() -> dict[str, Any]:
    """Read settings.json, falling back to defaults for any missing key."""
    settings = dict(DEFAULT_SETTINGS)
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as fh:
            settings.update(json.load(fh))
    except FileNotFoundError:
        pass
    except (json.JSONDecodeError, OSError):
        # A corrupt settings file must not stop the app from starting.
        pass
    return settings


def save_settings(settings: dict[str, Any]) -> None:
    """Write settings.json, creating the config directory if needed."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_FILE, "w", encoding="utf-8") as fh:
        json.dump(settings, fh, indent=4)


def resolve_model_path(settings: dict[str, Any] | None = None) -> Path:
    """Return the absolute path to the GGUF model file."""
    settings = settings if settings is not None else load_settings()
    raw = str(settings.get("model_path", DEFAULT_SETTINGS["model_path"]))
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_setting(key: str, default: Any = None) -> Any:
    """Read a single setting value."""
    return load_settings().get(key, default)


def update_setting(key: str, value: Any) -> None:
    """Write a single setting value, preserving the rest of the file."""
    settings = load_settings()
    settings[key] = value
    save_settings(settings)


def n_threads() -> int:
    """Resolve the inference thread count, honouring the auto_threads flag."""
    settings = load_settings()
    if settings.get("auto_threads", True):
        return max(1, (os.cpu_count() or 4) // 2)
    return int(settings.get("n_threads", 4))
