"""In-browser bridge: the REST API's routes as a plain function call.

The GitHub Pages demo runs this engine inside the visitor's browser with Pyodide
(Python compiled to WebAssembly). There is no server there, so the web console sends
its requests to `dispatch()` instead of over HTTP. Same engine code, same JSON.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from pydantic import BaseModel, ValidationError

from .config import Settings
from .core import Conflict, Engine, Forbidden, NotFound
from .gateway import ProviderError, list_providers
from .ingest import IngestError
from .schemas import AnswersIn, ProjectIn, ReviewIn, RunIn, TextArtifactIn

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "clinic_intake"
ACTOR = "demo"

_engine: Engine | None = None


def engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = Engine(Settings(db_path=":memory:"))
    return _engine


def _body(model: type[BaseModel], body: Any) -> Any:
    return model.model_validate(body if body is not None else {})


def _route(method: str, path: str, query: dict[str, str], body: Any) -> Any:
    e = engine()
    m = re.fullmatch

    if method == "GET" and path == "/api/health":
        return {"status": "ok", "provider": "offline", "embedder": "in-browser", "auth": False}
    if method == "GET" and path == "/api/providers":
        return [p for p in list_providers(e.settings) if p["name"] == "offline"]
    if method == "GET" and path == "/api/me":
        return {"actor": ACTOR, "role": "approver"}
    if path == "/api/projects":
        if method == "GET":
            return e.store.list_projects()
        return e.create_project(_body(ProjectIn, body).name, ACTOR)

    if g := m(r"/api/projects/([\w-]+)/artifacts", path):
        pid = g.group(1)
        if method == "GET":
            e.project(pid)
            return e.store.list_artifacts(pid)
        results = []
        for f in (body or {}).get("files", []):
            try:
                results.append(e.ingest_bytes(pid, f["name"], base64.b64decode(f["b64"]), ACTOR))
            except IngestError as exc:
                results.append({"filename": f["name"], "error": str(exc)})
        return results
    if (g := m(r"/api/projects/([\w-]+)/artifacts/text", path)) and method == "POST":
        b = _body(TextArtifactIn, body)
        return e.ingest_text(g.group(1), b.filename, b.text, ACTOR)
    if (g := m(r"/api/projects/([\w-]+)/sample", path)) and method == "POST":
        return [e.ingest_bytes(g.group(1), p.name, p.read_bytes(), ACTOR) for p in sorted(EXAMPLES.iterdir())]
    if (g := m(r"/api/projects/([\w-]+)/search", path)) and method == "GET":
        q = query.get("q", "").strip()
        if not q:
            raise Conflict("Search query is empty")
        return e.search(g.group(1), q, max(1, min(int(query.get("k", 8)), 25)))
    if g := m(r"/api/projects/([\w-]+)/runs", path):
        pid = g.group(1)
        if method == "GET":
            e.project(pid)
            return e.store.list_runs(pid)
        return e.start_run(pid, ACTOR, _body(RunIn, body).provider)

    if (g := m(r"/api/chunks/(.+)", path)) and method == "GET":
        return e.chunk(g.group(1))
    if (g := m(r"/api/runs/([\w-]+)", path)) and method == "GET":
        return e.run(g.group(1))
    if (g := m(r"/api/runs/([\w-]+)/answers", path)) and method == "POST":
        return e.answer(g.group(1), _body(AnswersIn, body).answers, ACTOR)
    if (g := m(r"/api/runs/([\w-]+)/review", path)) and method == "POST":
        b = _body(ReviewIn, body)
        return e.review(g.group(1), b.decision, b.note, ACTOR)
    if (g := m(r"/api/runs/([\w-]+)/events", path)) and method == "GET":
        return e.events(g.group(1))
    if (g := m(r"/api/runs/([\w-]+)/export\.md", path)) and method == "GET":
        return e.export_markdown(g.group(1))
    raise NotFound(f"No route for {method} {path}")


def dispatch(method: str, url: str, body_json: str = "null") -> str:
    """Handle one request. Returns JSON: {"status": int, "data": ...}."""
    parts = urlsplit(url)
    path = unquote(parts.path)
    query = {k: v[-1] for k, v in parse_qs(parts.query).items()}
    try:
        data = _route(method.upper(), path, query, json.loads(body_json))
        status = 200
    except NotFound as exc:
        status, data = 404, {"detail": str(exc)}
    except Forbidden as exc:
        status, data = 403, {"detail": str(exc)}
    except (IngestError, ValidationError) as exc:  # both subclass ValueError, so they come first
        status, data = 422, {"detail": str(exc)}
    except (Conflict, ValueError) as exc:  # ValueError covers bad numbers in the query string
        status, data = 409, {"detail": str(exc)}
    except ProviderError as exc:
        status, data = 400, {"detail": str(exc)}
    return json.dumps({"status": status, "data": data})
