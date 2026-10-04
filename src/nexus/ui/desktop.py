"""
desktop.py -- NEXUS pywebview launcher

Wraps the FastAPI backend (ui/server.py) and the web frontend (ui/web/) in
a native, frameless window, replacing the Tkinter Quick Menu as NEXUS's
primary interface.

No global hotkey here on purpose -- the window opens like a normal app
(Start Menu shortcut / `nexus` command) instead of toggling via Ctrl+Alt+N
(V4.0 Phase 43; see the Part K note in docs/NEXUS_TODO.md for why that
scope was cut).
"""

import socket
import threading
import time

import uvicorn
import webview

from nexus.ui.server import app as fastapi_app

HOST = "127.0.0.1"
WIN_WIDTH = 420
WIN_HEIGHT = 560
MIN_SIZE = (360, 460)

# Matches the Dark theme's bg_dark in ui/web/style.css, so there's no flash
# of mismatched colour behind the frameless window before the page paints.
BACKGROUND_COLOR = "#0A1931"


def _free_port() -> int:
    """Pick an unused local port instead of hardcoding one."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


def _wait_until_listening(host: str, port: int, timeout: float = 15.0) -> None:
    """Block until the server is actually accepting connections."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            if s.connect_ex((host, port)) == 0:
                return
        time.sleep(0.1)


class Api:
    """
    Exposed to the frontend as window.pywebview.api.*

    Covers what the old Tkinter title bar's window controls and the
    Settings/Train folder pickers did natively.
    """

    def __init__(self):
        self._window = None
        self._maximized = False

    def set_window(self, window) -> None:
        self._window = window

    def minimize(self) -> None:
        if self._window:
            self._window.minimize()

    def toggle_maximize(self) -> None:
        if not self._window:
            return
        if self._maximized:
            self._window.restore()
        else:
            self._window.maximize()
        self._maximized = not self._maximized

    def close(self) -> None:
        if self._window:
            self._window.destroy()

    def pick_folder(self):
        if not self._window:
            return None
        result = self._window.create_file_dialog(webview.FileDialog.FOLDER)
        return result[0] if result else None


def _run_server(port: int) -> None:
    config = uvicorn.Config(fastapi_app, host=HOST, port=port, log_level="warning")
    uvicorn.Server(config).run()


def main() -> None:
    """Console-script entry point (``nexus`` / ``python -m nexus``)."""
    port = _free_port()
    threading.Thread(target=_run_server, args=(port,), daemon=True).start()
    _wait_until_listening(HOST, port)

    api = Api()
    window = webview.create_window(
        "NEXUS",
        url=f"http://{HOST}:{port}/app/index.html",
        js_api=api,
        width=WIN_WIDTH,
        height=WIN_HEIGHT,
        min_size=MIN_SIZE,
        frameless=True,
        easy_drag=False,  # rely solely on the .pywebview-drag-region title bar
        background_color=BACKGROUND_COLOR,
    )
    api.set_window(window)
    webview.start()


if __name__ == "__main__":
    main()
