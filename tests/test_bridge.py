"""The GitHub Pages demo calls engine.bridge.dispatch instead of HTTP; it must behave like the API."""

import base64
import json
import sys

import pytest

from engine import bridge
from engine.config import Settings
from engine.core import Engine
from engine.store import Store


@pytest.fixture(autouse=True)
def fresh_engine(monkeypatch):
    monkeypatch.setattr(bridge, "_engine", Engine(Settings(db_path=":memory:"), Store(":memory:")))


def call(method, url, body=None):
    out = json.loads(bridge.dispatch(method, url, json.dumps(body)))
    return out["status"], out["data"]


def test_full_flow_through_the_bridge():
    _, project = call("POST", "/api/projects", {"name": "Browser demo"})
    pid = project["id"]
    status, loaded = call("POST", f"/api/projects/{pid}/sample")
    assert status == 200 and len(loaded) == 4

    _, hits = call("GET", f"/api/projects/{pid}/search?q=offline%20wifi&k=3")
    status, chunk = call("GET", "/api/chunks/" + hits[0]["chunk_id"].replace("#", "%23"))
    assert status == 200 and "offline" in chunk["text"].lower()

    _, run = call("POST", f"/api/projects/{pid}/runs", {})
    answers = {q["id"]: "Nurses and the manager; AWS; 7 years; PDF export." for q in run["spec"]["open_questions"]}
    _, run = call("POST", f"/api/runs/{run['id']}/answers", {"answers": answers})
    assert run["status"] == "in_review"
    _, run = call("POST", f"/api/runs/{run['id']}/review", {"decision": "approve"})
    assert run["status"] == "approved"
    _, md = call("GET", f"/api/runs/{run['id']}/export.md")
    assert md.startswith("# Browser demo")


def test_uploads_are_base64_and_errors_map_to_status_codes():
    _, project = call("POST", "/api/projects", {"name": "P"})
    pid = project["id"]
    files = [
        {"name": "notes.md", "b64": base64.b64encode(b"Nurses must log vitals.").decode()},
        {"name": "bad.exe", "b64": base64.b64encode(b"MZ").decode()},
    ]
    status, res = call("POST", f"/api/projects/{pid}/artifacts", {"files": files})
    assert status == 200 and res[0]["chunks"] == 1 and "Unsupported" in res[1]["error"]
    assert call("GET", "/api/runs/run_nope")[0] == 404
    assert call("POST", "/api/projects", {"name": ""})[0] == 422
    assert call("POST", "/api/projects/prj_x/runs", {})[0] == 404
    assert call("GET", "/api/nowhere")[0] == 404


def test_engine_imports_without_httpx(monkeypatch):
    monkeypatch.setitem(sys.modules, "httpx", None)
    assert call("GET", "/api/health")[1]["status"] == "ok"
