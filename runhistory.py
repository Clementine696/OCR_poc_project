"""Durable run snapshots, independent of the accuracy ranking CSV."""

import json
import re
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone

import config

PROCESS_ID = uuid.uuid4().hex


def _now():
    return datetime.now(timezone.utc).isoformat()


def _connect():
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(config.LOG_DIR / "run_history.sqlite3"), timeout=10)
    db.row_factory = sqlite3.Row
    db.execute("""CREATE TABLE IF NOT EXISTS runs (
        id TEXT PRIMARY KEY, kind TEXT NOT NULL, started_at TEXT NOT NULL,
        finished_at TEXT, status TEXT NOT NULL, owner TEXT NOT NULL,
        configuration TEXT NOT NULL, plan TEXT, result TEXT, progress TEXT,
        error TEXT NOT NULL DEFAULT '')""")
    db.execute("CREATE INDEX IF NOT EXISTS runs_started ON runs (started_at DESC, id DESC)")
    return db


def start(kind, configuration):
    run_id = uuid.uuid4().hex
    with closing(_connect()) as db, db:
        db.execute("INSERT INTO runs (id, kind, started_at, status, owner, configuration) "
                   "VALUES (?, ?, ?, 'running', ?, ?)",
                   (run_id, kind, _now(), PROCESS_ID, json.dumps(configuration, ensure_ascii=False)))
    return run_id


def save_plan(run_id, plan):
    with closing(_connect()) as db, db:
        db.execute("UPDATE runs SET plan=? WHERE id=?",
                   (json.dumps(plan, ensure_ascii=False), run_id))


def finish(run_id, status, result=None, progress=None, error=""):
    with closing(_connect()) as db, db:
        db.execute("UPDATE runs SET status=?, finished_at=?, result=?, progress=?, error=? "
                   "WHERE id=? AND status='running'",
                   (status, _now(), json.dumps(result, ensure_ascii=False),
                    json.dumps(progress, ensure_ascii=False), error, run_id))


def _decode(row):
    item = dict(row)
    owner = item.pop("owner")
    for key in ("configuration", "plan", "result", "progress"):
        if key in item:
            item[key] = json.loads(item[key]) if item[key] else None
    if item["status"] == "running" and owner != PROCESS_ID:
        item["status"] = "unfinished"
        item["error"] = "No final report is saved yet. This run belongs to another app instance or was interrupted by a restart."
    return item


def list_runs(limit=50, offset=0):
    limit, offset = max(1, min(int(limit), 100)), max(0, int(offset))
    with closing(_connect()) as db:
        total = db.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        rows = db.execute("SELECT id, kind, started_at, finished_at, status, owner, "
                          "configuration, plan, error FROM runs "
                          "ORDER BY started_at DESC, id DESC LIMIT ? OFFSET ?",
                          (limit, offset)).fetchall()
    return {"runs": [_decode(row) for row in rows], "total": total,
            "limit": limit, "offset": offset}


def get(run_id):
    if not re.fullmatch(r"[0-9a-f]{32}", run_id):
        return None
    with closing(_connect()) as db:
        row = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    return _decode(row) if row else None
