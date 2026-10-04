"""
server.py -- NEXUS FastAPI Backend

Exposes the existing agent (agent/router.py, agent/commands.py) and its
supporting modules (config, security/permissions, agent/memory, ui/themes,
llm/loader, deps, training/trainer) over HTTP + WebSocket, for the
pywebview-wrapped web frontend that replaces the old Tkinter Quick Menu.

No agent logic lives here -- every endpoint is a thin adapter around a
function that already existed before this file, reused as-is (V4.0 Phase 41).
"""

import asyncio
import os
import threading
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from nexus import config
from nexus import deps as dep_checker
from nexus.agent import memory, router
from nexus.security import confirm, permissions
from nexus.ui import themes


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Load the model in a background thread as the server starts."""
    def worker() -> None:
        from nexus.llm.loader import load_model
        try:
            load_model()
        except Exception as e:
            print(f"[ERROR] Model load failed: {e}")

    threading.Thread(target=worker, daemon=True).start()
    yield


app = FastAPI(title="NEXUS", lifespan=_lifespan)

_WEB_DIR = os.path.join(os.path.dirname(__file__), "web")
if os.path.isdir(_WEB_DIR):
    app.mount("/app", StaticFiles(directory=_WEB_DIR, html=True), name="web")


# ---------------------------------------------------------------------------
# Confirmation gateway -- bridges security/confirm.py's synchronous
# request/response contract onto the single active WebSocket connection.
#
# A destructive action (delete, shutdown, reset training, ...) calls
# confirm.request() from whatever thread is handling that command. That
# thread blocks on a threading.Event while the confirmation prompt is
# pushed to the frontend over the WebSocket; the event loop stays free to
# receive the frontend's answer and unblock it, exactly like the old
# Tkinter UI's worker thread blocked while the main thread's mainloop
# served the Tk dialog.
# ---------------------------------------------------------------------------
_main_loop: asyncio.AbstractEventLoop | None = None
_active_ws: WebSocket | None = None
_pending_confirms: dict[str, tuple[threading.Event, list]] = {}


def _confirm_via_ws(title: str, details: str = "") -> bool:
    if _active_ws is None or _main_loop is None:
        return False  # fail closed -- no frontend connected

    req_id = uuid.uuid4().hex
    event = threading.Event()
    answer_box = [False]
    _pending_confirms[req_id] = (event, answer_box)

    try:
        asyncio.run_coroutine_threadsafe(
            _active_ws.send_json({"type": "confirm", "id": req_id, "title": title, "details": details}),
            _main_loop,
        )
    except Exception:
        _pending_confirms.pop(req_id, None)
        return False

    answered = event.wait(timeout=120)
    _pending_confirms.pop(req_id, None)
    return answer_box[0] if answered else False


confirm.set_handler(_confirm_via_ws)


# ---------------------------------------------------------------------------
# /api/command -- non-streaming request/response
# ---------------------------------------------------------------------------
class CommandRequest(BaseModel):
    text: str


class CommandResponse(BaseModel):
    reply: str


@app.post("/api/command", response_model=CommandResponse)
def post_command(body: CommandRequest) -> CommandResponse:
    try:
        reply = router.route(body.text)
    except Exception as e:
        # Mirrors the old Tkinter UI: errors are shown as a chat message,
        # never a network-level failure, so the frontend has one code path.
        reply = f"Error: {e}"
    return CommandResponse(reply=reply)


# ---------------------------------------------------------------------------
# /api/ws -- streaming chat + confirmation channel
# ---------------------------------------------------------------------------
@app.websocket("/api/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    global _active_ws, _main_loop

    await websocket.accept()
    _active_ws = websocket
    _main_loop = asyncio.get_running_loop()

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "confirm_response":
                pending = _pending_confirms.get(data.get("id"))
                if pending:
                    event, answer_box = pending
                    answer_box[0] = bool(data.get("answer"))
                    event.set()

            elif msg_type == "command":
                _start_command_worker(websocket, _main_loop, data.get("text", ""))

    except WebSocketDisconnect:
        pass
    finally:
        if _active_ws is websocket:
            _active_ws = None


def _start_command_worker(websocket: WebSocket, loop: asyncio.AbstractEventLoop, text: str) -> None:
    """
    Run router.route_stream() on a worker thread and forward tokens to the
    client as they arrive.

    Crucially, this does NOT block the WebSocket's receive loop while the
    command runs -- a destructive action mid-command calls confirm.request(),
    which needs that same receive loop free to read the frontend's
    confirm_response before the worker thread can be unblocked. Blocking
    here would deadlock the worker against itself.
    """
    def on_token(chunk: str) -> None:
        asyncio.run_coroutine_threadsafe(
            websocket.send_json({"type": "token", "text": chunk}), loop
        )

    def worker() -> None:
        try:
            router.route_stream(text, on_token)
        except Exception as e:
            on_token(f"Error: {e}")
        finally:
            asyncio.run_coroutine_threadsafe(websocket.send_json({"type": "done"}), loop)

    threading.Thread(target=worker, daemon=True).start()


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
class SettingsUpdate(BaseModel):
    max_tokens: int | None = None
    n_ctx: int | None = None
    n_threads: int | None = None
    auto_threads: bool | None = None


@app.get("/api/settings")
def get_settings() -> dict:
    return config.load_settings()


@app.post("/api/settings")
def post_settings(body: SettingsUpdate) -> dict:
    settings = config.load_settings()
    settings.update(body.model_dump(exclude_none=True))
    config.save_settings(settings)
    return settings


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
class ThemeUpdate(BaseModel):
    name: str


@app.get("/api/theme")
def get_theme() -> dict:
    return {"name": themes.load_saved_theme(), "available": themes.get_theme_names()}


@app.post("/api/theme")
def post_theme(body: ThemeUpdate) -> dict:
    if body.name not in themes.get_theme_names():
        raise HTTPException(400, f"Unknown theme: {body.name}")
    themes.save_theme(body.name)
    return {"name": body.name}


# ---------------------------------------------------------------------------
# Folder permissions
# ---------------------------------------------------------------------------
class PermissionsUpdate(BaseModel):
    paths: list[str]


@app.get("/api/permissions")
def get_permissions() -> dict:
    return {"paths": permissions.load_permissions()}


@app.post("/api/permissions")
def post_permissions(body: PermissionsUpdate) -> dict:
    permissions.save_permissions(body.paths)
    return {"paths": permissions.load_permissions()}


# ---------------------------------------------------------------------------
# Persistent memory
# ---------------------------------------------------------------------------
class ForgetRequest(BaseModel):
    target: str


@app.get("/api/memory")
def get_memory() -> dict:
    return {"facts": memory.load_all_facts()}


@app.post("/api/memory/forget")
def post_memory_forget(body: ForgetRequest) -> CommandResponse:
    return CommandResponse(reply=memory.forget_fact(body.target))


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------
@app.get("/api/about")
def get_about() -> dict:
    from nexus.llm import loader

    lora_applied = False
    try:
        lora_dir = str(config.LORA_WEIGHTS_DIR)
        lora_applied = os.path.isdir(lora_dir) and bool(os.listdir(lora_dir))
    except OSError:
        pass

    return {
        "version": "V4.0",
        "theme": themes.load_saved_theme(),
        "model_loaded": loader.model is not None,
        "lora_applied": lora_applied,
        "memories": len(memory.load_all_facts()),
    }


@app.post("/api/model/reset")
def post_model_reset() -> CommandResponse:
    from nexus.llm import loader

    return CommandResponse(reply=loader.reset_training())


# ---------------------------------------------------------------------------
# Training (experimental -- see README)
# ---------------------------------------------------------------------------
class TrainRequest(BaseModel):
    folder: str
    quick: bool = False


@app.post("/api/train")
def post_train(body: TrainRequest) -> dict:
    """
    Kick off training in a background thread and return immediately.
    The result is pushed to the active WebSocket as a "train_complete"
    message once training finishes, since it can run for a long time.
    """
    def worker() -> None:
        from nexus.training import trainer
        try:
            result = trainer.train_on_folder(body.folder, quick_mode=body.quick)
        except Exception as e:
            result = f"Training error: {e}"
        if _active_ws is not None and _main_loop is not None:
            asyncio.run_coroutine_threadsafe(
                _active_ws.send_json({"type": "train_complete", "reply": result}), _main_loop
            )

    threading.Thread(target=worker, daemon=True).start()
    return {"status": "started"}


# ---------------------------------------------------------------------------
# RAG -- "Index this folder" / "Ask your documents" (Phase 45)
# ---------------------------------------------------------------------------
class RagIndexRequest(BaseModel):
    folder: str


class RagAskRequest(BaseModel):
    question: str


@app.get("/api/rag/status")
def get_rag_status() -> dict:
    from nexus.tools import rag

    deps = dep_checker.check_dependencies()
    missing = [pip_name for pip_name, _, feature in deps["missing_optional"]
               if feature == "RAG / document search"]
    return {
        "has_index": rag.has_index(),
        "info": rag.get_index_info(),
        "deps_missing": missing,
    }


@app.post("/api/rag/index")
def post_rag_index(body: RagIndexRequest) -> dict:
    """
    Build the RAG index in a background thread; the result is pushed to the
    active WebSocket as a "rag_index_complete" message once it finishes,
    the same long-running-task pattern as /api/train, since embedding a
    large folder can take well beyond a normal request timeout.
    """
    def worker() -> None:
        from nexus.tools import rag
        try:
            result = rag.build_index(body.folder)
        except Exception as e:
            result = f"Indexing error: {e}"
        if _active_ws is not None and _main_loop is not None:
            asyncio.run_coroutine_threadsafe(
                _active_ws.send_json({"type": "rag_index_complete", "reply": result}), _main_loop
            )

    threading.Thread(target=worker, daemon=True).start()
    return {"status": "started"}


@app.post("/api/rag/ask", response_model=CommandResponse)
def post_rag_ask(body: RagAskRequest) -> CommandResponse:
    from nexus.tools import rag
    try:
        reply = rag.ask(body.question)
    except Exception as e:
        reply = f"Error: {e}"
    return CommandResponse(reply=reply)


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
@app.get("/api/deps")
def get_deps() -> dict:
    return dep_checker.check_dependencies()


@app.post("/api/deps/install")
def post_deps_install() -> dict:
    import subprocess
    import sys

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", ".[all]"],
            cwd=str(config.PROJECT_ROOT),
            capture_output=True, text=True, timeout=120,
        )
    except Exception as e:
        return {"ok": False, "message": str(e)}

    if result.returncode == 0:
        return {"ok": True, "message": "All packages installed!"}
    return {"ok": False, "message": result.stderr[:300]}
