"""
Tests for the FastAPI backend (server.py).

Uses TestClient without the `with` context form, so the app's lifespan
(which loads the real 3B model) never runs -- these tests only exercise
the HTTP layer and the parts of the agent that don't need the model.
"""

import pytest
from fastapi.testclient import TestClient

from nexus.ui import server

client = TestClient(server.app)


@pytest.fixture(autouse=True)
def _ensure_confirm_handler_is_registered():
    """
    server.py registers its WebSocket confirm handler once at import time.
    test_confirm.py's tests reset the (global, shared) confirm handler to
    None in their own teardown, which would otherwise leak into whichever
    test module runs after it in the same session. Reassert it here so
    these tests don't depend on module import/execution order.
    """
    from nexus.security import confirm

    confirm.set_handler(server._confirm_via_ws)


def test_command_happy_path_does_not_need_the_model():
    response = client.post("/api/command", json={"text": "/help"})
    assert response.status_code == 200
    assert "Slash Commands" in response.json()["reply"]


def test_command_missing_model_file_is_a_graceful_chat_error(monkeypatch):
    from nexus.llm import loader

    monkeypatch.setattr(loader, "model", None)

    def _raise():
        raise FileNotFoundError("Model file not found at: model/missing.gguf")

    monkeypatch.setattr(loader, "load_model", _raise)

    # Any input that reaches generate() -- either as QA directly or via the
    # LLM fallback classifier -- must surface the failure as a chat message,
    # not a 500 or an unhandled exception killing the request.
    response = client.post("/api/command", json={"text": "what is machine learning"})
    assert response.status_code == 200
    assert "Error" in response.json()["reply"]


def test_settings_roundtrip(tmp_path, monkeypatch):
    from nexus import config

    target = tmp_path / "settings.json"
    monkeypatch.setattr(config, "SETTINGS_FILE", target)
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)

    response = client.post("/api/settings", json={"max_tokens": 111})
    assert response.status_code == 200
    assert response.json()["max_tokens"] == 111

    response = client.get("/api/settings")
    assert response.json()["max_tokens"] == 111


def test_theme_get_and_set(tmp_path, monkeypatch):
    from nexus import config

    target = tmp_path / "settings.json"
    monkeypatch.setattr(config, "SETTINGS_FILE", target)

    response = client.post("/api/theme", json={"name": "Light"})
    assert response.status_code == 200

    response = client.get("/api/theme")
    assert response.json()["name"] == "Light"
    assert "Dark" in response.json()["available"]


def test_theme_rejects_unknown_name():
    response = client.post("/api/theme", json={"name": "Nonexistent"})
    assert response.status_code == 400


def test_permissions_roundtrip(tmp_path, monkeypatch):
    from nexus.security import permissions

    perm_file = tmp_path / "permissions.json"
    monkeypatch.setattr(permissions, "CONFIG_PATH", str(perm_file))

    allowed = tmp_path / "allowed"
    allowed.mkdir()

    response = client.post("/api/permissions", json={"paths": [str(allowed)]})
    assert response.status_code == 200
    assert str(allowed) in response.json()["paths"]

    response = client.get("/api/permissions")
    assert str(allowed) in response.json()["paths"]


def test_memory_endpoint_returns_a_list(tmp_path, monkeypatch):
    from nexus.agent import memory

    monkeypatch.setattr(memory, "MEMORY_FILE", str(tmp_path / "user_memory.json"))

    response = client.get("/api/memory")
    assert response.status_code == 200
    assert isinstance(response.json()["facts"], list)


# ---------------------------------------------------------------------------
# WebSocket streaming + confirmation gateway
#
# The receive loop must stay free to read a confirm_response while a
# command's worker thread is mid-flight -- an earlier version awaited the
# worker's completion inside the same coroutine that reads incoming
# messages, which deadlocked the moment a destructive action needed
# confirmation. These pin that fix down.
# ---------------------------------------------------------------------------
def test_ws_streams_a_non_llm_command_to_completion():
    with client.websocket_connect("/api/ws") as ws:
        ws.send_json({"type": "command", "text": "what is my battery"})
        chunks = []
        while True:
            msg = ws.receive_json()
            if msg["type"] == "done":
                break
            chunks.append(msg["text"])
        assert "Battery" in "".join(chunks)


def test_ws_confirm_approved_runs_the_destructive_action(tmp_path, monkeypatch):
    from nexus.security import permissions

    monkeypatch.setattr(permissions, "CONFIG_PATH", str(tmp_path / "permissions.json"))
    permissions.save_permissions([str(tmp_path)])

    target = tmp_path / "deleteme.txt"
    target.write_text("x", encoding="utf-8")

    with client.websocket_connect("/api/ws") as ws:
        ws.send_json({"type": "command", "text": f"/task delete {target}"})

        msg = ws.receive_json()
        assert msg["type"] == "confirm"
        ws.send_json({"type": "confirm_response", "id": msg["id"], "answer": True})

        chunks = []
        while True:
            msg = ws.receive_json()
            if msg["type"] == "done":
                break
            chunks.append(msg["text"])
        assert "deleted" in "".join(chunks).lower()

    assert not target.exists()


def test_ws_confirm_denied_blocks_the_destructive_action(tmp_path, monkeypatch):
    from nexus.security import permissions

    monkeypatch.setattr(permissions, "CONFIG_PATH", str(tmp_path / "permissions.json"))
    permissions.save_permissions([str(tmp_path)])

    target = tmp_path / "keepme.txt"
    target.write_text("x", encoding="utf-8")

    with client.websocket_connect("/api/ws") as ws:
        ws.send_json({"type": "command", "text": f"/task delete {target}"})

        msg = ws.receive_json()
        assert msg["type"] == "confirm"
        ws.send_json({"type": "confirm_response", "id": msg["id"], "answer": False})

        chunks = []
        while True:
            msg = ws.receive_json()
            if msg["type"] == "done":
                break
            chunks.append(msg["text"])
        assert "cancelled" in "".join(chunks).lower()

    assert target.exists()
