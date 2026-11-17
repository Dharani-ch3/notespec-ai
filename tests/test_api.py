from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from engine import api
from engine.config import Settings
from engine.core import Engine
from engine.store import Store


def client_for(settings: Settings) -> TestClient:
    eng = Engine(settings, Store(":memory:"))
    api.app.dependency_overrides[api.get_engine] = lambda: eng
    return TestClient(api.app)


@pytest.fixture(autouse=True)
def _reset():
    yield
    api.app.dependency_overrides.clear()


def test_full_flow_sample_to_approved_spec():
    c = client_for(Settings(db_path=":memory:"))
    assert c.get("/").status_code == 200
    pid = c.post("/api/projects", json={"name": "Clinic"}).json()["id"]
    loaded = c.post(f"/api/projects/{pid}/sample").json()
    assert len(loaded) == 4

    hits = c.get(f"/api/projects/{pid}/search", params={"q": "works offline when wifi drops"}).json()
    assert "offline" in hits[0]["text"].lower()
    assert c.get(f"/api/chunks/{quote(hits[0]['chunk_id'], safe='')}").json()["filename"]

    run = c.post(f"/api/projects/{pid}/runs", json={}).json()
    assert run["status"] == "needs_answers" and run["spec"]["features"]

    # Reviewing is blocked while questions are open.
    r = c.post(f"/api/runs/{run['id']}/review", json={"decision": "approve"})
    assert r.status_code == 409

    # Answer every open question; answers become cited sources.
    qs = run["spec"]["open_questions"]
    run = c.post(
        f"/api/runs/{run['id']}/answers",
        json={"answers": {q["id"]: "Only nurses and the clinic manager; keep for 7 years on AWS." for q in qs}},
    ).json()
    assert run["status"] == "in_review" and run["revision"] == 2

    run = c.post(f"/api/runs/{run['id']}/review", json={"decision": "request_changes", "note": "Staff must be able to print the queue."}).json()
    assert run["revision"] == 3
    assert any("print the queue" in f["description"] for f in run["spec"]["features"])

    run = c.post(f"/api/runs/{run['id']}/review", json={"decision": "approve"}).json()
    assert run["status"] == "approved" and run["approved_by"] == "demo"

    md = c.get(f"/api/runs/{run['id']}/export.md").text
    assert md.startswith("# Clinic") and "## Decisions" in md and "DRAFT" not in md
    actions = [e["action"] for e in c.get(f"/api/runs/{run['id']}/events").json()]
    for a in ["run_started", "context_built", "spec_generated", "questions_answered", "changes_requested", "spec_approved"]:
        assert a in actions


def test_unknown_question_ids_and_bad_files_are_rejected():
    c = client_for(Settings(db_path=":memory:"))
    pid = c.post("/api/projects", json={"name": "P"}).json()["id"]
    res = c.post(f"/api/projects/{pid}/artifacts", files=[("files", ("x.exe", b"MZ", "application/octet-stream"))]).json()
    assert "Unsupported" in res[0]["error"]
    c.post(f"/api/projects/{pid}/artifacts/text", json={"filename": "n", "text": "Who signs off? Nurses must log vitals."})
    run = c.post(f"/api/projects/{pid}/runs", json={}).json()
    assert c.post(f"/api/runs/{run['id']}/answers", json={"answers": {"Q-nope": "x"}}).status_code == 409
    assert c.get("/api/runs/run_missing").status_code == 404


def test_duplicate_upload_is_not_stored_twice():
    c = client_for(Settings(db_path=":memory:"))
    pid = c.post("/api/projects", json={"name": "P"}).json()["id"]
    body = {"filename": "n.md", "text": "Nurses must log vitals."}
    c.post(f"/api/projects/{pid}/artifacts/text", json=body)
    assert c.post(f"/api/projects/{pid}/artifacts/text", json=body).json()["duplicate"] is True
    assert len(c.get(f"/api/projects/{pid}/artifacts").json()) == 1


def test_roles_and_separation_of_duties():
    keys = {"k-view": ("vic", "viewer"), "k-edit": ("eve", "editor"), "k-appr": ("ann", "approver")}
    c = client_for(Settings(db_path=":memory:", api_keys=keys))
    assert c.get("/api/projects").status_code == 401
    assert c.post("/api/projects", json={"name": "P"}, headers={"X-API-Key": "k-view"}).status_code == 403

    pid = c.post("/api/projects", json={"name": "P"}, headers={"X-API-Key": "k-appr"}).json()["id"]
    text = "Nurses must log vitals. Staff log in with SSO. Hosted on AWS."
    c.post(f"/api/projects/{pid}/artifacts/text", json={"filename": "n", "text": text}, headers={"X-API-Key": "k-edit"})

    # The approver starts a run, so the approver cannot also approve it.
    run = c.post(f"/api/projects/{pid}/runs", json={}, headers={"X-API-Key": "k-appr"}).json()
    assert run["status"] == "in_review", run["spec"]["open_questions"]
    r = c.post(f"/api/runs/{run['id']}/review", json={"decision": "approve"}, headers={"X-API-Key": "k-appr"})
    assert r.status_code == 403 and "Separation of duties" in r.json()["detail"]
    # An editor cannot approve at all.
    r = c.post(f"/api/runs/{run['id']}/review", json={"decision": "approve"}, headers={"X-API-Key": "k-edit"})
    assert r.status_code == 403


def test_health_and_providers():
    c = client_for(Settings(db_path=":memory:"))
    assert c.get("/api/health").json()["provider"] == "offline"
    names = {p["name"]: p for p in c.get("/api/providers").json()}
    assert names["offline"]["configured"] and names["offline"]["default"]
    assert set(names) >= {"openai", "anthropic", "ollama"}
