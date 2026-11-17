"""The engine itself. The REST API, MCP server and CLI are thin wrappers around this class.

Run lifecycle:
    ingest -> build context -> generate spec --(open questions?)--> needs_answers
                                   ^                                    | answers
                                   |------------------------------------+
                                   |                                    v
                         request_changes <---- reviewer ---------- in_review --> approved
"""

from __future__ import annotations

from typing import Any

from .config import Settings, get_settings
from .gateway import ProviderError, get_embedder, get_provider, is_configured
from .index import IndexCache
from .ingest import IngestError, chunk, extract_text, safe_filename, sha256
from .schemas import RunStatus, Spec
from .spec import GenerationResult, llm_spec, offline_spec
from .store import Store

# Retrieval queries used to build context for a spec, one per section of the spec.
CONTEXT_QUERIES = [
    "goal purpose of the app what we want to build",
    "users roles staff who uses it needs to",
    "must should need to be able to requirements features",
    "track store record data fields form",
    "screen page dashboard view",
    "security privacy compliance HIPAA login access",
    "offline language tablet integration deadline budget",
    "not sure unclear TBD open question decide",
]


class NotFound(LookupError):
    pass


class Conflict(RuntimeError):
    pass


class Forbidden(PermissionError):
    pass


class Engine:
    def __init__(self, settings: Settings | None = None, store: Store | None = None) -> None:
        self.settings = settings or get_settings()
        self.store = store or Store(self.settings.db_path)
        self.embedder = get_embedder(self.settings)
        self.indexes = IndexCache()

    # ---------- projects ----------

    def create_project(self, name: str, actor: str) -> dict[str, Any]:
        project = self.store.create_project(name)
        self.store.log(project["id"], actor, "project_created", name=name)
        return project

    def project(self, project_id: str) -> dict[str, Any]:
        project = self.store.get_project(project_id)
        if not project:
            raise NotFound(f"Project {project_id} not found")
        return project

    # ---------- ingest ----------

    def ingest_bytes(self, project_id: str, filename: str, data: bytes, actor: str) -> dict[str, Any]:
        self.project(project_id)
        if len(data) > self.settings.max_upload_bytes:
            raise IngestError(f"{filename} is larger than {self.settings.max_upload_bytes // 1024 // 1024} MB")
        filename = safe_filename(filename)
        kind, text = extract_text(filename, data)
        return self._store_text(project_id, filename, kind, text, actor)

    def ingest_text(self, project_id: str, filename: str, text: str, actor: str) -> dict[str, Any]:
        self.project(project_id)
        filename = safe_filename(filename)
        if "." not in filename:
            filename += ".md"
        kind, clean = extract_text(filename, text.encode("utf-8"))
        return self._store_text(project_id, filename, kind, clean, actor)

    def _store_text(self, project_id: str, filename: str, kind: str, text: str, actor: str) -> dict[str, Any]:
        digest = sha256(text)
        dup = self.store.find_artifact_by_hash(project_id, digest)
        if dup:
            return {**dup, "duplicate": True}
        pieces = chunk(text, self.settings.chunk_words, self.settings.chunk_overlap)
        art = self.store.add_artifact(project_id, filename, kind, digest, text, pieces, actor)
        self.store.log(project_id, actor, "artifact_ingested", artifact=art["id"], filename=filename, chunks=len(pieces))
        art.pop("text", None)
        return art

    # ---------- retrieval ----------

    def _index(self, project_id: str):
        return self.indexes.get(project_id, self.store.list_chunks(project_id), self.embedder)

    def search(self, project_id: str, query: str, k: int = 8) -> list[dict[str, Any]]:
        self.project(project_id)
        return [h.as_dict() for h in self._index(project_id).search(query, k)]

    def build_context(self, project_id: str, run: dict[str, Any]) -> list[dict[str, Any]]:
        """Pick the passages the spec is written from, plus answers and reviewer feedback."""
        index = self._index(project_id)
        limit = self.settings.context_chunks
        if len(index.chunks) <= limit:
            chosen = list(index.chunks)
        else:
            seen: dict[str, dict] = {}
            per_query = max(2, limit // len(CONTEXT_QUERIES) + 1)
            for q in CONTEXT_QUERIES:
                for hit in index.search(q, per_query):
                    seen.setdefault(hit.chunk_id, {"id": hit.chunk_id, "filename": hit.filename, "text": hit.text})
            order = {c["id"]: i for i, c in enumerate(index.chunks)}
            chosen = sorted(list(seen.values())[:limit], key=lambda c: order[c["id"]])
        for qid, qa in run["answers"].items():
            chosen.append({"id": f"answers#{qid}", "filename": "answers", "text": f"Q: {qa['question']}\nA: {qa['answer']}"})
        for i, note in enumerate(run["feedback"]):
            chosen.append({"id": f"feedback#{i}", "filename": "reviewer feedback", "text": note["note"]})
        return chosen

    # ---------- runs ----------

    def start_run(self, project_id: str, actor: str, provider: str | None = None) -> dict[str, Any]:
        project = self.project(project_id)
        if not self.store.list_chunks(project_id):
            raise Conflict("Add at least one artifact before generating a spec")
        provider = (provider or self.settings.provider).lower()
        get_provider(provider, self.settings)  # validates the name early
        run = self.store.create_run(project_id, provider, actor)
        self.store.log(project_id, actor, "run_started", run["id"], provider=provider)
        return self._generate(project, run, actor)

    def _generate(self, project: dict, run: dict[str, Any], actor: str) -> dict[str, Any]:
        context = self.build_context(project["id"], run)
        self.store.log(
            project["id"], "engine", "context_built", run["id"],
            passages=len(context), chunk_ids=[c["id"] for c in context][:40],
        )
        answered = {qid: qa["answer"] for qid, qa in run["answers"].items()}
        question_text = {qid: qa["question"] for qid, qa in run["answers"].items()}
        warnings: list[str] = []
        result: GenerationResult
        provider_name = run["provider"]
        llm = None
        if provider_name != "offline":
            if not is_configured(provider_name, self.settings):
                warnings.append(f"{provider_name} is not configured, so the offline extractor was used instead.")
            else:
                llm = get_provider(provider_name, self.settings)
        try:
            if llm is not None:
                result = llm_spec(llm, project["name"], context, answered, question_text, self.settings.max_spec_repairs)
            else:
                result = GenerationResult(spec=offline_spec(project["name"], context, answered))
        except ProviderError as exc:
            warnings.append(f"{provider_name} failed ({exc}); the offline extractor was used instead.")
            self.store.log(project["id"], "engine", "provider_fallback", run["id"], error=str(exc)[:300])
            result = GenerationResult(spec=offline_spec(project["name"], context, answered))

        spec: Spec = result.spec
        warnings.extend(result.warnings)
        status = RunStatus.needs_answers if spec.open_questions else RunStatus.in_review
        updated = self.store.update_run(
            run["id"], spec=spec.model_dump(), status=status.value, warnings=warnings
        )
        self.store.log(
            project["id"], "engine", "spec_generated", run["id"],
            revision=updated["revision"], repairs=result.repairs, status=status.value,
            features=len(spec.features), open_questions=len(spec.open_questions), warnings=len(warnings),
        )
        return updated

    def run(self, run_id: str) -> dict[str, Any]:
        run = self.store.get_run(run_id)
        if not run:
            raise NotFound(f"Run {run_id} not found")
        return run

    def answer(self, run_id: str, answers: dict[str, str], actor: str) -> dict[str, Any]:
        run = self.run(run_id)
        if run["status"] == RunStatus.approved.value:
            raise Conflict("This spec is already approved")
        open_q = {q["id"]: q["question"] for q in (run["spec"] or {}).get("open_questions", [])}
        unknown = set(answers) - set(open_q)
        if unknown:
            raise Conflict(f"Unknown or already answered question ids: {', '.join(sorted(unknown))}")
        clean = {k: v.strip() for k, v in answers.items() if v and v.strip()}
        if not clean:
            raise Conflict("Answers cannot be empty")
        merged = {**run["answers"], **{k: {"question": open_q[k], "answer": v} for k, v in clean.items()}}
        self.store.update_run(run_id, answers=merged, revision=run["revision"] + 1)
        self.store.log(run["project_id"], actor, "questions_answered", run_id, ids=sorted(clean))
        return self._generate(self.project(run["project_id"]), self.run(run_id), actor)

    def review(self, run_id: str, decision: str, note: str, actor: str) -> dict[str, Any]:
        run = self.run(run_id)
        if run["status"] != RunStatus.in_review.value:
            raise Conflict(f"Only specs in review can be reviewed (status is {run['status']})")
        if self.settings.auth_enabled and actor == run["created_by"]:
            raise Forbidden("Separation of duties: the person who started a run cannot approve it")
        if decision == "approve":
            updated = self.store.update_run(run_id, status=RunStatus.approved.value, approved_by=actor)
            self.store.log(run["project_id"], actor, "spec_approved", run_id, revision=run["revision"], note=note)
            return updated
        if not note.strip():
            raise Conflict("Say what should change when requesting changes")
        feedback = [*run["feedback"], {"by": actor, "note": note.strip()}]
        self.store.update_run(run_id, feedback=feedback, revision=run["revision"] + 1)
        self.store.log(run["project_id"], actor, "changes_requested", run_id, note=note)
        return self._generate(self.project(run["project_id"]), self.run(run_id), actor)

    def events(self, run_id: str) -> list[dict[str, Any]]:
        run = self.run(run_id)
        return self.store.events(run["project_id"], run_id)

    def chunk(self, chunk_id: str) -> dict[str, Any]:
        found = self.store.get_chunk(chunk_id)
        if not found:
            raise NotFound(f"Passage {chunk_id} not found")
        return found

    # ---------- export ----------

    def export_markdown(self, run_id: str) -> str:
        run = self.run(run_id)
        spec = run["spec"] or {}

        def cite(item: dict) -> str:
            refs = sorted({s["artifact"] or s["chunk_id"] for s in item.get("sources", [])})
            return f" _(sources: {', '.join(refs)})_" if refs else ""

        status = run["status"].replace("_", " ")
        lines = [f"# {spec.get('title', 'Spec')}", ""]
        if run["status"] != RunStatus.approved.value:
            lines += [f"> **DRAFT** (status: {status}, revision {run['revision']})", ""]
        else:
            lines += [f"> Approved by {run['approved_by']} (revision {run['revision']})", ""]
        lines += [spec.get("summary", ""), ""]
        sections = [
            ("Users", "users", lambda u: f"- **{u['name']}**" + (": " + "; ".join(u["goals"]) if u["goals"] else "") + cite(u)),
            ("Features", "features", lambda f: f"- **{f['id']} {f['name']}** [{f['priority']}]: {f['description']}{cite(f)}"),
            ("Data entities", "entities", lambda e: f"- **{e['name']}**" + (f" ({', '.join(e['fields'])})" if e["fields"] else "") + cite(e)),
            ("Screens", "screens", lambda s: f"- **{s['name']}**: {s['purpose']}{cite(s)}"),
            ("Constraints", "constraints", lambda c: f"- [{c['kind']}] {c['text']}{cite(c)}"),
            ("Open questions", "open_questions", lambda q: f"- {q['question']}"),
        ]
        for heading, key, fmt in sections:
            items = spec.get(key) or []
            if items:
                lines += [f"## {heading}", "", *[fmt(i) for i in items], ""]
        if run["answers"]:
            lines += ["## Decisions", ""]
            for qa in run["answers"].values():
                lines.append(f"- **{qa['question']}** {qa['answer']}")
            lines.append("")
        return "\n".join(lines)
