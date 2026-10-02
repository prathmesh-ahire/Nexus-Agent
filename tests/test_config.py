"""
Tests for centralised path and settings resolution.

Every module used to compute PROJECT_ROOT from its own __file__, so moving a
file silently repointed unrelated features at the wrong directory. These tests
pin the invariant that all derived paths stay under the project root.
"""

import json

import pytest

from nexus import config


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def test_project_root_contains_the_package():
    assert (config.PROJECT_ROOT / "src" / "nexus" / "config.py").is_file()


@pytest.mark.parametrize(
    "path_name",
    [
        "CONFIG_DIR", "MODEL_DIR", "DATASETS_DIR", "LORA_WEIGHTS_DIR",
        "RAG_INDEX_DIR", "SETTINGS_FILE", "PERMISSIONS_FILE", "API_KEYS_FILE",
        "MEMORY_FILE", "QUEUE_FILE",
    ],
)
def test_derived_paths_live_under_project_root(path_name):
    path = getattr(config, path_name)
    assert config.PROJECT_ROOT in path.parents


def test_all_modules_agree_on_the_root():
    """A module computing its own root would drift from config's."""
    from nexus.agent import memory
    from nexus.llm import loader
    from nexus.security import permissions
    from nexus.tools import search, system

    root = str(config.PROJECT_ROOT)
    for value in (
        loader.SETTINGS_PATH,
        loader.LORA_WEIGHTS_PATH,
        memory.MEMORY_FILE,
        permissions.CONFIG_PATH,
        system.QUEUE_FILE,
        search.API_KEYS_FILE,
    ):
        assert str(value).startswith(root), f"{value} escaped the project root"


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
def test_load_settings_returns_defaults_for_missing_keys():
    settings = config.load_settings()
    for key in config.DEFAULT_SETTINGS:
        assert key in settings


def test_corrupt_settings_file_does_not_raise(tmp_path, monkeypatch):
    bad = tmp_path / "settings.json"
    bad.write_text("{ this is not json", encoding="utf-8")
    monkeypatch.setattr(config, "SETTINGS_FILE", bad)
    # A corrupt file must degrade to defaults, not stop the app from starting.
    assert config.load_settings() == config.DEFAULT_SETTINGS


def test_missing_settings_file_falls_back_to_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SETTINGS_FILE", tmp_path / "nope.json")
    assert config.load_settings() == config.DEFAULT_SETTINGS


def test_save_then_load_roundtrip(tmp_path, monkeypatch):
    target = tmp_path / "settings.json"
    monkeypatch.setattr(config, "SETTINGS_FILE", target)
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)

    config.save_settings({"max_tokens": 99, "theme": "blue"})
    loaded = config.load_settings()

    assert loaded["max_tokens"] == 99
    assert loaded["theme"] == "blue"
    # Unspecified keys still come from defaults.
    assert loaded["n_ctx"] == config.DEFAULT_SETTINGS["n_ctx"]


def test_update_setting_preserves_other_keys(tmp_path, monkeypatch):
    target = tmp_path / "settings.json"
    monkeypatch.setattr(config, "SETTINGS_FILE", target)
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)

    config.save_settings({"max_tokens": 42, "theme": "dark"})
    config.update_setting("theme", "light")

    written = json.loads(target.read_text(encoding="utf-8"))
    assert written["theme"] == "light"
    assert written["max_tokens"] == 42


def test_relative_model_path_resolves_against_root():
    resolved = config.resolve_model_path({"model_path": "model/x.gguf"})
    assert resolved.is_absolute()
    assert config.PROJECT_ROOT in resolved.parents


def test_absolute_model_path_is_left_alone():
    absolute = config.PROJECT_ROOT / "elsewhere" / "y.gguf"
    assert config.resolve_model_path({"model_path": str(absolute)}) == absolute


def test_n_threads_is_at_least_one():
    assert config.n_threads() >= 1
