"""A small, synchronous Chrome DevTools Protocol client.

nw.js (the runtime RPG Maker MZ games ship in) embeds Chromium, so a game started
with --remote-debugging-port speaks CDP.  That gives us three things a screen
scraper cannot: a real screenshot stream (Page.startScreencast), synthetic input
(Input.dispatchKeyEvent), and - when the runtime cooperates - frame stepping via
Emulation.setVirtualTimePolicy.

This module deliberately implements only what the environment needs and raises
CdpError with the underlying message on every failure, because a silent CDP
misunderstanding is the most expensive kind of bug in this project.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from websockets.sync.client import connect as ws_connect

DEFAULT_ENDPOINT = "http://127.0.0.1:9222"


class CdpError(RuntimeError):
    """Any failure talking to, or understood only partially by, the debugger."""


@dataclass
class Target:
    id: str
    type: str
    title: str
    url: str
    web_socket_debugger_url: str | None = None

    @property
    def is_page(self) -> bool:
        return self.type == "page"


def _http_json(url: str, timeout: float = 3.0) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "vpt-speedrun/0.0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def cdp_endpoint_alive(endpoint: str | None = None, *, timeout: float = 1.5) -> bool:
    """True when something answers /json/version at the endpoint."""
    endpoint = (endpoint or DEFAULT_ENDPOINT).rstrip("/")
    try:
        _http_json(endpoint + "/json/version", timeout=timeout)
        return True
    except Exception:
        return False


def cdp_version(endpoint: str | None = None) -> dict[str, Any]:
    endpoint = (endpoint or DEFAULT_ENDPOINT).rstrip("/")
    try:
        return _http_json(endpoint + "/json/version")
    except Exception as exc:
        raise CdpError("no CDP endpoint at %s: %s" % (endpoint, exc)) from exc


def list_targets(endpoint: str | None = None) -> list[Target]:
    """Every debuggable target, pages first."""
    endpoint = (endpoint or DEFAULT_ENDPOINT).rstrip("/")
    try:
        raw = _http_json(endpoint + "/json/list")
    except Exception as exc:
        raise CdpError("cannot list CDP targets at %s: %s" % (endpoint, exc)) from exc
    targets = [
        Target(
            id=str(t.get("id")),
            type=str(t.get("type", "")),
            title=str(t.get("title", "")),
            url=str(t.get("url", "")),
            web_socket_debugger_url=t.get("webSocketDebuggerUrl"),
        )
        for t in raw
    ]
    targets.sort(key=lambda t: (not t.is_page, t.title))
    return targets


@dataclass
class CdpConnection:
    """Blocking request/response plus event subscription over one WebSocket."""

    websocket_url: str
    timeout: float = 10.0
    _ws: Any = None
    _next_id: int = 1
    _events: list[dict[str, Any]] = field(default_factory=list)
    _open: bool = False

    def open(self) -> CdpConnection:
        if self._open:
            return self
        try:
            self._ws = ws_connect(self.websocket_url, open_timeout=self.timeout,
                                  max_size=64 * 1024 * 1024)
        except Exception as exc:
            raise CdpError("cannot open %s: %s" % (self.websocket_url, exc)) from exc
        self._open = True
        return self

    def close(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
        self._ws = None
        self._open = False

    def __enter__(self) -> CdpConnection:
        return self.open()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------- primitives
    def call(self, method: str, params: dict[str, Any] | None = None, *,
             timeout: float | None = None) -> dict[str, Any]:
        """Send a command and wait for its matching reply.

        Events that arrive while waiting are buffered and can be drained with
        pop_events(); nothing is ever silently dropped.
        """
        if not self._open:
            self.open()
        message_id = self._next_id
        self._next_id += 1
        payload = {"id": message_id, "method": method}
        if params:
            payload["params"] = params
        try:
            self._ws.send(json.dumps(payload))
        except Exception as exc:
            raise CdpError("send %s failed: %s" % (method, exc)) from exc

        deadline = time.monotonic() + (timeout if timeout is not None else self.timeout)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CdpError("timeout waiting for %s" % method)
            try:
                raw = self._ws.recv(timeout=remaining)
            except Exception as exc:
                raise CdpError("recv while waiting for %s failed: %s" % (method, exc)) from exc
            if raw is None:
                raise CdpError("debugger closed the connection during %s" % method)
            msg = json.loads(raw)
            if msg.get("id") == message_id:
                if "error" in msg:
                    raise CdpError("%s rejected: %s" % (method, msg["error"]))
                return msg.get("result", {})
            if "id" in msg:
                # A reply to a command we are not waiting for: keep it.
                self._events.append(msg)
            else:
                self._events.append(msg)

    def evaluate(self, expression: str, *, await_promise: bool = False,
                 return_by_value: bool = True, timeout: float | None = None) -> Any:
        """Runtime.evaluate returning the JSON value, raising on exceptions."""
        result = self.call(
            "Runtime.evaluate",
            {
                "expression": expression,
                "returnByValue": return_by_value,
                "awaitPromise": await_promise,
                "userGesture": True,
            },
            timeout=timeout,
        )
        if result.get("exceptionDetails"):
            detail = result["exceptionDetails"]
            raise CdpError("evaluate raised: %s" % json.dumps(detail)[:500])
        remote = result.get("result", {})
        if remote.get("subtype") == "error":
            raise CdpError("evaluate produced an error object: %s" % remote.get("description"))
        return remote.get("value")

    def drain_events(self, *, max_wait: float = 0.0) -> list[dict[str, Any]]:
        """Return buffered events, optionally waiting a little for new ones."""
        events, self._events = self._events, []
        if max_wait > 0:
            deadline = time.monotonic() + max_wait
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    raw = self._ws.recv(timeout=remaining)
                except Exception:
                    break
                if raw is None:
                    break
                events.append(json.loads(raw))
        return events

    def pop_events(self) -> list[dict[str, Any]]:
        events, self._events = self._events, []
        return events
