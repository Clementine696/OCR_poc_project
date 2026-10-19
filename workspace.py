"""Per-tab workspaces: each browser tab's own pickers and switches.

**Added 2026-10-19 at the user's request** -- *the web will be deployed on a
server and there will be other users using it*, with each TAB its own user and
each tab's random and stress tests running and monitored on their own. Until
now every picker on the page was process state: one person choosing a model
moved everyone's next read onto it, a random test switching models between
rounds moved everyone else's reads with it, and a second Run anywhere was
refused because one was already going.

**Two kinds of caller, and the old one is untouched.** A request that names no
workspace -- `compare.py`, `randomtest.py`, `stress.py`, curl, the tests -- is in
the *default* workspace, which IS the module globals every setting has always
lived in. Nothing about those paths changes. A request naming one (the page
sends `X-OCR-Workspace` on every fetch, and `?ws=` on the page load itself) gets
its own copy of every value registered here with `field`, made the first time
the workspace asks for it, from what the process started with (`seal`) rather
than from what the default workspace has been moved to since: a CLI sweep
switching models must not move a tab that has not picked one yet.

**What stays shared** is what describes the servers rather than a choice made
on them -- the endpoint list, each server's engine, the probe cache -- and the
run log, which is one measurement history however many people add to it.

The current workspace is a context variable set once per request. A thread a
request starts does not inherit it, so work handed to a pool is wrapped in
`bind` first.

Imports nothing of this project's but `config`, so `settings` can use it.
"""

import contextlib
import contextvars
import copy
import re
import threading
import time

import config

# A workspace nobody has asked anything of for this long is forgotten; the next
# request naming it starts again from the defaults.
IDLE_SECONDS = config.env_float("WORKSPACE_IDLE_HOURS", 24.0, minimum=0.1) * 3600
# The most kept at once. Past it the longest-idle one goes first.
MAX_WORKSPACES = config.env_int("MAX_WORKSPACES", 500, minimum=1)
# How recently a workspace must have been heard from to count as someone still
# working in it -- `active`, which is what decides whose models an Ollama
# switch may unload.
ACTIVE_SECONDS = config.env_float("WORKSPACE_ACTIVE_MINUTES", 15.0, minimum=0.1) * 60

HEADER = "X-OCR-Workspace"
_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")

_current = contextvars.ContextVar("ocr_workspace", default=None)
_lock = threading.Lock()
_registry = {}
# key -> (initial, fresh). `initial()` is what a new workspace starts with;
# `fresh` values (a running sweep, a counter) are never taken from the template.
_fields = {}
_template = None
# The default workspace's own bookkeeping, since it has no object.
_default = {"seen": 0.0, "busy": 0}
_last_sweep = 0.0


class Workspace:
    def __init__(self, wid: str):
        self.id = wid
        self.values = {}
        self.seen = time.time()
        self.busy = 0
        self.lock = threading.RLock()

    def get(self, key):
        with self.lock:
            if key not in self.values:
                self.values[key] = _initial(key)
            return self.values[key]

    def put(self, key, value):
        with self.lock:
            self.values[key] = value


def field(key: str, initial, fresh: bool = False):
    """Register a per-workspace value. `initial()` gives a new workspace's copy.

    For a setting, `initial` reads the module global, and `seal` keeps what it
    read at startup. For runtime state, `fresh=True`, and it is called every time.
    """
    _fields[key] = (initial, fresh)


def _initial(key):
    initial, fresh = _fields[key]
    if not fresh and _template is not None and key in _template:
        return copy.deepcopy(_template[key])
    return copy.deepcopy(initial())


def seal():
    """Fix what a new workspace starts from: the settings as they are now.

    Called once, after startup has auto-selected the server and models, so a
    tab opens on the same choices the app announced in its log.
    """
    global _template
    _template = {key: copy.deepcopy(initial())
                 for key, (initial, fresh) in _fields.items() if not fresh}


def valid(wid) -> bool:
    return isinstance(wid, str) and bool(_ID.match(wid))


def enter(wid=None):
    """Make `wid` the current request's workspace, creating it if new.

    An absent or malformed id is the default workspace. Returns the workspace,
    or None for the default.
    """
    now = time.time()
    if not valid(wid):
        _default["seen"] = now
        _current.set(None)
        return None
    with _lock:
        ws = _registry.get(wid)
        if ws is None:
            ws = _registry[wid] = Workspace(wid)
        ws.seen = now
        _evict(now)
    _current.set(ws)
    return ws


def _evict(now):
    """Forget idle workspaces. Caller holds `_lock`."""
    global _last_sweep
    if now - _last_sweep < 60 and len(_registry) <= MAX_WORKSPACES:
        return
    _last_sweep = now
    for wid, ws in list(_registry.items()):
        if not ws.busy and now - ws.seen > IDLE_SECONDS:
            del _registry[wid]
    if len(_registry) > MAX_WORKSPACES:
        idle = sorted((ws for ws in _registry.values() if not ws.busy),
                      key=lambda ws: ws.seen)
        for ws in idle[:len(_registry) - MAX_WORKSPACES]:
            del _registry[ws.id]


def current():
    """The current workspace, or None meaning the default one."""
    return _current.get()


def current_id() -> str:
    ws = _current.get()
    return ws.id if ws is not None else ""


def get(key: str, default_value):
    """The current workspace's `key`; `default_value` in the default workspace.

    The caller passes the module global it has always read, so the default
    workspace keeps reading exactly that.
    """
    ws = _current.get()
    return default_value if ws is None else ws.get(key)


def put(key: str, value) -> bool:
    """Set `key` in the current workspace. False in the default one, where the
    caller sets its module global as before."""
    ws = _current.get()
    if ws is None:
        return False
    ws.put(key, value)
    return True


@contextlib.contextmanager
def use(ws):
    """Run a block as workspace `ws` (None = the default)."""
    token = _current.set(ws)
    try:
        yield ws
    finally:
        _current.reset(token)


def bind(fn):
    """`fn`, set to run in the workspace current NOW -- for handing to a thread."""
    ws = _current.get()

    def run(*args, **kwargs):
        with use(ws):
            return fn(*args, **kwargs)
    return run


def hold():
    """Mark the current workspace busy (a sweep is running in it): never
    evicted, and always `active`. Pair with `release`."""
    ws = _current.get()
    with _lock:
        if ws is None:
            _default["busy"] += 1
        else:
            ws.busy += 1


def release():
    ws = _current.get()
    with _lock:
        if ws is None:
            _default["busy"] = max(0, _default["busy"] - 1)
        else:
            ws.busy = max(0, ws.busy - 1)


def active(include_current: bool = True) -> list:
    """Every workspace someone is still working in, the default as None.

    Heard from within `ACTIVE_SECONDS`, or busy. The current one is always
    included unless asked not to be: whoever is asking is by definition there.
    """
    now = time.time()
    cur = _current.get()
    out = []
    with _lock:
        if (_default["busy"] or now - _default["seen"] <= ACTIVE_SECONDS
                or (include_current and cur is None)):
            out.append(None)
        for ws in _registry.values():
            if ws.busy or now - ws.seen <= ACTIVE_SECONDS or (include_current and ws is cur):
                out.append(ws)
    if not include_current:
        out = [ws for ws in out if ws is not cur]
    return out


def find(wid):
    """The workspace with this id if it exists, without creating it."""
    with _lock:
        return _registry.get(wid)


def reset():
    """Forget every workspace and the template. For tests."""
    global _template, _last_sweep
    with _lock:
        _registry.clear()
        _template = None
        _last_sweep = 0.0
        _default.update(seen=0.0, busy=0)
    _current.set(None)
