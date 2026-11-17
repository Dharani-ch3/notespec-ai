"""SQLite storage: projects, artifacts, chunks, runs and an append-only audit log.

SQLite keeps the demo dependency-free. The queries are plain SQL, so moving to
Postgres means swapping the connection, not rewriting the engine.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id),
    filename TEXT NOT NULL,
    kind TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    text TEXT NOT NULL,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(project_id, sha256)
);
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    artifact_id TEXT NOT NULL REFERENCES artifacts(id),
    ord INTEGER NOT NULL,
    text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_project ON chunks(project_id);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id),
    status TEXT NOT NULL,
    revision INTEGER NOT NULL DEFAULT 1,
    provider TEXT NOT NULL,
    spec_json TEXT,
    answers_json TEXT NOT NULL DEFAULT '{}',
    feedback_json TEXT NOT NULL DEFAULT '[]',
    warnings_json TEXT NOT NULL DEFAULT '[]',
    created_by TEXT NOT NULL,
    approved_by TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL,
    run_id TEXT,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    detail_json TEXT NOT NULL DEFAULT '{}',
    at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_run ON events(run_id);
"""


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Store:
    def __init__(self, path: str) -> None:
        self.path = path
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # One shared connection guarded by a lock: simple and safe for a single-process demo.
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._tx() as db:
            db.execute("PRAGMA foreign_keys = ON")
            if path != ":memory:":
                db.execute("PRAGMA journal_mode = WAL")
            db.executescript(SCHEMA)

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._db
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise

    # ---------- projects ----------

    def create_project(self, name: str) -> dict[str, Any]:
        row = {"id": new_id("prj"), "name": name.strip(), "created_at": now()}
        with self._tx() as db:
            db.execute("INSERT INTO projects VALUES (:id, :name, :created_at)", row)
        return row

    def list_projects(self) -> list[dict[str, Any]]:
        with self._tx() as db:
            rows = db.execute(
                """SELECT p.*, (SELECT COUNT(*) FROM artifacts a WHERE a.project_id = p.id) AS artifacts,
                          (SELECT COUNT(*) FROM runs r WHERE r.project_id = p.id) AS runs
                   FROM projects p ORDER BY created_at DESC"""
            ).fetchall()
        return [dict(r) for r in rows]

    def get_project(self, project_id: str) -> dict[str, Any] | None:
        with self._tx() as db:
            row = db.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        return dict(row) if row else None

    # ---------- artifacts and chunks ----------

    def find_artifact_by_hash(self, project_id: str, sha256: str) -> dict[str, Any] | None:
        with self._tx() as db:
            row = db.execute(
                "SELECT id, filename FROM artifacts WHERE project_id = ? AND sha256 = ?",
                (project_id, sha256),
            ).fetchone()
        return dict(row) if row else None

    def add_artifact(
        self,
        project_id: str,
        filename: str,
        kind: str,
        sha256: str,
        text: str,
        chunks: list[str],
        actor: str,
    ) -> dict[str, Any]:
        art = {
            "id": new_id("art"),
            "project_id": project_id,
            "filename": filename,
            "kind": kind,
            "sha256": sha256,
            "text": text,
            "created_by": actor,
            "created_at": now(),
        }
        with self._tx() as db:
            db.execute(
                "INSERT INTO artifacts VALUES (:id, :project_id, :filename, :kind, :sha256, :text, :created_by, :created_at)",
                art,
            )
            db.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?)",
                [(f"{art['id']}#{i}", project_id, art["id"], i, c) for i, c in enumerate(chunks)],
            )
        art["chunks"] = len(chunks)
        return art

    def list_artifacts(self, project_id: str) -> list[dict[str, Any]]:
        with self._tx() as db:
            rows = db.execute(
                """SELECT a.id, a.filename, a.kind, a.created_by, a.created_at, LENGTH(a.text) AS chars,
                          (SELECT COUNT(*) FROM chunks c WHERE c.artifact_id = a.id) AS chunks
                   FROM artifacts a WHERE project_id = ? ORDER BY created_at""",
                (project_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_chunks(self, project_id: str) -> list[dict[str, Any]]:
        with self._tx() as db:
            rows = db.execute(
                """SELECT c.id, c.artifact_id, c.ord, c.text, a.filename
                   FROM chunks c JOIN artifacts a ON a.id = c.artifact_id
                   WHERE c.project_id = ? ORDER BY a.created_at, c.ord""",
                (project_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        with self._tx() as db:
            row = db.execute(
                """SELECT c.id, c.project_id, c.artifact_id, c.ord, c.text, a.filename
                   FROM chunks c JOIN artifacts a ON a.id = c.artifact_id WHERE c.id = ?""",
                (chunk_id,),
            ).fetchone()
        return dict(row) if row else None

    # ---------- runs ----------

    def create_run(self, project_id: str, provider: str, actor: str) -> dict[str, Any]:
        ts = now()
        run = {
            "id": new_id("run"),
            "project_id": project_id,
            "status": "needs_answers",
            "provider": provider,
            "created_by": actor,
            "created_at": ts,
            "updated_at": ts,
        }
        with self._tx() as db:
            db.execute(
                """INSERT INTO runs (id, project_id, status, provider, created_by, created_at, updated_at)
                   VALUES (:id, :project_id, :status, :provider, :created_by, :created_at, :updated_at)""",
                run,
            )
        return self.get_run(run["id"])  # type: ignore[return-value]

    def update_run(self, run_id: str, **fields: Any) -> dict[str, Any]:
        allowed = {"status", "revision", "provider", "spec", "answers", "feedback", "warnings", "approved_by"}
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"Cannot update {unknown}")
        cols, values = [], []
        for key, value in fields.items():
            if key in {"spec", "answers", "feedback", "warnings"}:
                cols.append(f"{key}_json = ?")
                values.append(json.dumps(value))
            else:
                cols.append(f"{key} = ?")
                values.append(value)
        cols.append("updated_at = ?")
        values.extend([now(), run_id])
        with self._tx() as db:
            db.execute(f"UPDATE runs SET {', '.join(cols)} WHERE id = ?", values)
        return self.get_run(run_id)  # type: ignore[return-value]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._tx() as db:
            row = db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if not row:
            return None
        run = dict(row)
        run["spec"] = json.loads(run.pop("spec_json")) if run.get("spec_json") else None
        for key in ("answers", "feedback", "warnings"):
            run[key] = json.loads(run.pop(f"{key}_json"))
        return run

    def list_runs(self, project_id: str) -> list[dict[str, Any]]:
        with self._tx() as db:
            rows = db.execute(
                "SELECT id, status, revision, provider, created_by, approved_by, created_at, updated_at "
                "FROM runs WHERE project_id = ? ORDER BY created_at DESC",
                (project_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ---------- audit log ----------

    def log(self, project_id: str, actor: str, action: str, run_id: str | None = None, **detail: Any) -> None:
        with self._tx() as db:
            db.execute(
                "INSERT INTO events (project_id, run_id, actor, action, detail_json, at) VALUES (?, ?, ?, ?, ?, ?)",
                (project_id, run_id, actor, action, json.dumps(detail), now()),
            )

    def events(self, project_id: str, run_id: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM events WHERE project_id = ?"
        args: list[Any] = [project_id]
        if run_id:
            sql += " AND (run_id = ? OR run_id IS NULL)"
            args.append(run_id)
        with self._tx() as db:
            rows = db.execute(sql + " ORDER BY id", args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["detail"] = json.loads(d.pop("detail_json"))
            out.append(d)
        return out
