"""REST API and web console.

Auth: set ENGINE_API_KEYS="alice:approver:KEY1,bob:editor:KEY2" and send `X-API-Key`.
Roles are ordered viewer < editor < approver. With no keys set the engine runs in open
demo mode, where every caller is the "demo" user with full rights.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.requests import Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .core import Conflict, Engine, Forbidden, NotFound
from .gateway import ProviderError, list_providers
from .ingest import IngestError
from .schemas import AnswersIn, ProjectIn, ReviewIn, RunIn, TextArtifactIn

STATIC = Path(__file__).parent / "static"
EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "clinic_intake"
ROLE_RANK = {"viewer": 0, "editor": 1, "approver": 2}


@lru_cache
def get_engine() -> Engine:
    return Engine()


class Caller:
    def __init__(self, actor: str, role: str) -> None:
        self.actor, self.role = actor, role


def require(role: str) -> Callable[..., Caller]:
    def dep(
        engine: Annotated[Engine, Depends(get_engine)],
        x_api_key: Annotated[str | None, Header()] = None,
    ) -> Caller:
        keys = engine.settings.api_keys
        if not keys:
            return Caller("demo", "approver")
        if not x_api_key or x_api_key not in keys:
            raise HTTPException(401, "Missing or invalid X-API-Key")
        actor, have = keys[x_api_key]
        if ROLE_RANK[have] < ROLE_RANK[role]:
            raise HTTPException(403, f"This action needs the {role} role (you are {have})")
        return Caller(actor, have)

    return dep


EngineDep = Annotated[Engine, Depends(get_engine)]
Viewer = Annotated[Caller, Depends(require("viewer"))]
Editor = Annotated[Caller, Depends(require("editor"))]
Approver = Annotated[Caller, Depends(require("approver"))]

app = FastAPI(
    title="Knowledge AI Engine",
    version=__version__,
    description="Turn messy notes and documents into a reviewed, cited app spec.",
)


@app.exception_handler(NotFound)
async def _not_found(_: Request, exc: NotFound) -> JSONResponse:
    return JSONResponse({"detail": str(exc)}, status_code=404)


@app.exception_handler(Conflict)
async def _conflict(_: Request, exc: Conflict) -> JSONResponse:
    return JSONResponse({"detail": str(exc)}, status_code=409)


@app.exception_handler(Forbidden)
async def _forbidden(_: Request, exc: Forbidden) -> JSONResponse:
    return JSONResponse({"detail": str(exc)}, status_code=403)


@app.exception_handler(IngestError)
async def _bad_file(_: Request, exc: IngestError) -> JSONResponse:
    return JSONResponse({"detail": str(exc)}, status_code=422)


@app.exception_handler(ProviderError)
async def _provider(_: Request, exc: ProviderError) -> JSONResponse:
    return JSONResponse({"detail": str(exc)}, status_code=400)


# ---------- meta ----------


@app.get("/api/health")
def health(engine: EngineDep) -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "provider": engine.settings.provider,
        "embedder": engine.embedder.name,
        "auth": engine.settings.auth_enabled,
    }


@app.get("/api/providers")
def providers(engine: EngineDep, _: Viewer) -> list[dict]:
    return list_providers(engine.settings)


@app.get("/api/me")
def me(caller: Viewer) -> dict:
    return {"actor": caller.actor, "role": caller.role}


# ---------- projects and artifacts ----------


@app.get("/api/projects")
def projects(engine: EngineDep, _: Viewer) -> list[dict]:
    return engine.store.list_projects()


@app.post("/api/projects", status_code=201)
def create_project(body: ProjectIn, engine: EngineDep, caller: Editor) -> dict:
    return engine.create_project(body.name, caller.actor)


@app.get("/api/projects/{project_id}/artifacts")
def artifacts(project_id: str, engine: EngineDep, _: Viewer) -> list[dict]:
    engine.project(project_id)
    return engine.store.list_artifacts(project_id)


@app.post("/api/projects/{project_id}/artifacts", status_code=201)
async def upload(project_id: str, engine: EngineDep, caller: Editor, files: list[UploadFile] = File(...)) -> list[dict]:
    results = []
    for f in files:
        data = await f.read(engine.settings.max_upload_bytes + 1)
        try:
            results.append(engine.ingest_bytes(project_id, f.filename or "upload.txt", data, caller.actor))
        except IngestError as exc:
            results.append({"filename": f.filename, "error": str(exc)})
    return results


@app.post("/api/projects/{project_id}/artifacts/text", status_code=201)
def add_text(project_id: str, body: TextArtifactIn, engine: EngineDep, caller: Editor) -> dict:
    return engine.ingest_text(project_id, body.filename, body.text, caller.actor)


@app.post("/api/projects/{project_id}/sample", status_code=201)
def load_sample(project_id: str, engine: EngineDep, caller: Editor) -> list[dict]:
    """Load the bundled clinic-intake notes so visitors can try the engine in one click."""
    return [
        engine.ingest_bytes(project_id, p.name, p.read_bytes(), caller.actor) for p in sorted(EXAMPLES.iterdir())
    ]


@app.get("/api/projects/{project_id}/search")
def search(
    project_id: str, engine: EngineDep, _: Viewer, q: str = Query(min_length=1, max_length=500), k: int = Query(8, ge=1, le=25)
) -> list[dict]:
    return engine.search(project_id, q, k)


@app.get("/api/chunks/{chunk_id:path}")
def chunk(chunk_id: str, engine: EngineDep, _: Viewer) -> dict:
    return engine.chunk(chunk_id)


# ---------- runs ----------


@app.get("/api/projects/{project_id}/runs")
def runs(project_id: str, engine: EngineDep, _: Viewer) -> list[dict]:
    engine.project(project_id)
    return engine.store.list_runs(project_id)


@app.post("/api/projects/{project_id}/runs", status_code=201)
def start_run(project_id: str, engine: EngineDep, caller: Editor, body: RunIn | None = None) -> dict:
    return engine.start_run(project_id, caller.actor, body.provider if body else None)


@app.get("/api/runs/{run_id}")
def get_run(run_id: str, engine: EngineDep, _: Viewer) -> dict:
    return engine.run(run_id)


@app.post("/api/runs/{run_id}/answers")
def answer(run_id: str, body: AnswersIn, engine: EngineDep, caller: Editor) -> dict:
    return engine.answer(run_id, body.answers, caller.actor)


@app.post("/api/runs/{run_id}/review")
def review(run_id: str, body: ReviewIn, engine: EngineDep, caller: Approver) -> dict:
    return engine.review(run_id, body.decision, body.note, caller.actor)


@app.get("/api/runs/{run_id}/events")
def events(run_id: str, engine: EngineDep, _: Viewer) -> list[dict]:
    return engine.events(run_id)


@app.get("/api/runs/{run_id}/export.md", response_class=PlainTextResponse)
def export(run_id: str, engine: EngineDep, _: Viewer) -> str:
    return engine.export_markdown(run_id)


# ---------- web console ----------

app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def console() -> FileResponse:
    return FileResponse(STATIC / "index.html")
