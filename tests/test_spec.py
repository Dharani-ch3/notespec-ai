import json

import pytest

from engine import gateway
from engine.config import Settings
from engine.core import Conflict, Engine
from engine.gateway import ProviderError
from engine.schemas import Spec
from engine.spec import ground, llm_spec, offline_spec, question_id
from engine.store import Store

CHUNKS = [
    {"id": "n#0", "filename": "notes.md", "text": "Nurses must be able to flag urgent symptoms.\nWho approves refunds?"},
    {"id": "n#1", "filename": "notes.md", "text": "Appointment fields: patient, date, status.\nStaff log in with SSO."},
]


def test_offline_spec_extracts_cited_items_and_questions():
    spec = offline_spec("Demo", CHUNKS, {})
    assert any(f.name.startswith("Flag urgent") and f.priority == "must" for f in spec.features)
    appt = next(e for e in spec.entities if e.name == "Appointment")
    assert appt.fields == ["patient", "date", "status"]
    assert any(c.kind == "security" for c in spec.constraints)
    assert [q.question for q in spec.open_questions][0] == "Who approves refunds?"
    for f in spec.features:
        assert f.sources and all(s.chunk_id in {"n#0", "n#1"} for s in f.sources)


def test_answered_questions_are_not_asked_again():
    qid = question_id("Who approves refunds?")
    spec = offline_spec("Demo", CHUNKS, {qid: "The clinic manager"})
    assert qid not in {q.id for q in spec.open_questions}


def test_gap_check_asks_about_hosting_when_notes_never_mention_it():
    spec = offline_spec("Demo", CHUNKS, {})
    assert any("Where will the app run" in q.question for q in spec.open_questions)


class FakeLLM:
    name, model = "fake", "fake-1"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []

    def complete_json(self, system, user):
        self.prompts.append(user)
        return self.outputs.pop(0)


GOOD = {
    "title": "Demo",
    "summary": "Triage app",
    "users": [{"name": "Nurse", "goals": ["Flag urgent symptoms"], "sources": [{"chunk_id": "n#0"}]}],
    "features": [
        {"id": "X", "name": "Urgent flag", "description": "Flag urgent", "priority": "must", "sources": [{"chunk_id": "n#0"}]},
        {"id": "Y", "name": "Invented", "description": "Not in notes", "priority": "must", "sources": [{"chunk_id": "nope#9"}]},
    ],
    "entities": [],
    "screens": [],
    "constraints": [],
    "open_questions": [{"id": "Q1", "question": "Who approves refunds?", "why_it_matters": "", "sources": []}],
}


def test_llm_path_repairs_invalid_json_then_grounds_citations():
    llm = FakeLLM(["not json at all", json.dumps({"title": "x"}), json.dumps(GOOD)])
    result = llm_spec(llm, "Demo", CHUNKS, {}, {}, max_repairs=2)
    assert result.repairs == 2
    assert "previous answer was invalid" in llm.prompts[1]
    names = [f.name for f in result.spec.features]
    assert names == ["Urgent flag"]  # the uncited "Invented" feature was dropped
    assert result.spec.features[0].id == "F1"
    assert result.spec.features[0].sources[0].artifact == "notes.md"
    assert any("Dropped uncited feature" in w for w in result.warnings)
    assert result.spec.open_questions[0].id == question_id("Who approves refunds?")


def test_llm_path_gives_up_after_max_repairs():
    with pytest.raises(ProviderError):
        llm_spec(FakeLLM(["{}", "{}", "{}"]), "Demo", CHUNKS, {}, {}, max_repairs=2)


def test_ground_removes_answered_questions():
    spec = Spec.model_validate(GOOD)
    spec, _ = ground(spec, CHUNKS, {question_id("Who approves refunds?"): "Manager"})
    assert spec.open_questions == []


def test_engine_uses_registered_provider_and_falls_back_on_failure(monkeypatch):
    class Broken:
        name, model = "broken", "x"

        def __init__(self, settings):
            pass

        def complete_json(self, system, user):
            raise ProviderError("model is down")

    monkeypatch.setitem(gateway.PROVIDERS, "broken", Broken)
    eng = Engine(Settings(db_path=":memory:"), Store(":memory:"))
    p = eng.create_project("P", "t")
    eng.ingest_text(p["id"], "notes", CHUNKS[0]["text"], "t")
    run = eng.start_run(p["id"], "t", "broken")
    assert run["spec"]["features"]  # offline extractor stepped in
    assert any("broken failed" in w for w in run["warnings"])
    assert any(e["action"] == "provider_fallback" for e in eng.events(run["id"]))


def test_run_needs_artifacts(engine):
    p = engine.create_project("Empty", "t")
    with pytest.raises(Conflict):
        engine.start_run(p["id"], "t")


def test_open_questions_explain_why_they_matter():
    from engine.spec import DEFAULT_REASON, why_it_matters

    assert "edit rights" in why_it_matters("Who is allowed to edit a patient's intake after they submit it?")
    assert "where the app runs" in why_it_matters("Host on the clinic's existing AWS account?")
    assert "retention" in why_it_matters("How long do we keep the intake records?")
    assert "responsible for" in why_it_matters("Decide whether vitals belong in this app or stay in the EHR.")
    assert "integration" in why_it_matters("Decide whether we push intake answers into the EHR or just export a PDF.")
    assert why_it_matters("What color should the logo be?") == DEFAULT_REASON
