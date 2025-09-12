"""Flask-based development server with SSE live reload.

This module provides a simple, reliable, and linear server implementation
for the `uvnote serve` command.
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Callable, Iterable

from flask import Flask, Response, redirect  # type: ignore[import-not-found]


class Broadcaster:
    """Thread-safe broadcaster for SSE messages using per-client queues."""

    def __init__(self) -> None:
        self._clients: set[queue.Queue[str]] = set()
        self._lock = threading.Lock()

    def register(self) -> queue.Queue[str]:
        q: queue.Queue[str] = queue.Queue()
        with self._lock:
            self._clients.add(q)
        return q

    def unregister(self, q: queue.Queue[str]) -> None:
        with self._lock:
            self._clients.discard(q)

    def broadcast(self, message: str) -> None:
        with self._lock:
            for q in list(self._clients):
                try:
                    q.put_nowait(message)
                except Exception:
                    # If a client queue is full or closed, ignore; client may be gone.
                    pass


def sse_stream(broadcaster: Broadcaster) -> Iterable[str]:
    """SSE generator: yields messages for a single client."""
    q = broadcaster.register()
    try:
        yield "data: connected\n\n"
        # Heartbeat every 15s if no messages
        while True:
            try:
                msg = q.get(timeout=15)
                yield f"data: {msg}\n\n"
            except queue.Empty:
                yield ": hb\n\n"
    finally:
        broadcaster.unregister(q)


def create_app(output_dir: Path, index_name: str, broadcaster: Broadcaster) -> Flask:
    """Create a Flask app that serves static files and an SSE endpoint."""

    # Flask resolves relative static_folder against the package root, not CWD.
    # Use an absolute path so we serve the caller's output directory.
    static_root = str(output_dir.resolve())
    app = Flask(__name__, static_folder=static_root, static_url_path="")

    @app.after_request
    def _no_cache(resp):  # type: ignore[override]
        # Avoid caching during development; also disable proxy buffering for SSE.
        # Allow caching for image files to prevent reload issues
        if resp.mimetype and resp.mimetype.startswith("image/"):
            resp.headers.setdefault("Cache-Control", "max-age=3600")
        else:
            resp.headers.setdefault("Cache-Control", "no-cache")
        if resp.mimetype == "text/event-stream":
            resp.headers["X-Accel-Buffering"] = "no"
            resp.headers["Connection"] = "keep-alive"
        return resp

    @app.get("/events")
    def events():
        return Response(sse_stream(broadcaster), mimetype="text/event-stream")

    @app.get("/")
    def root():
        return redirect("/" + index_name, code=302)

    # Static files are served by Flask's static handler at the root path.
    return app
