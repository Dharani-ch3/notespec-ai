"""MCP server: lets Cursor or any MCP client use the engine as a tool.

Run over stdio:   python -m engine.mcp_server
MCP client config (e.g. Cursor):
    {"mcpServers": {"notespec": {"command": "python", "args": ["-m", "engine.mcp_server"],
                                          "cwd": "/path/to/notespec-ai"}}}
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from .core import Engine

ACTOR = "mcp-client"
mcp = MCPServer(
    name="notespec-ai",
    instructions=(
        "Turn project notes into a cited app spec. Typical flow: create_project, add_note (one per document), "
        "generate_spec, then answer_questions until no open questions remain. A human approves the spec in the web console."
    ),
)
_engine: Engine | None = None


def engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = Engine()
    return _engine


def _summary(run: dict) -> dict:
    spec = run.get("spec") or {}
    return {
        "run_id": run["id"],
        "status": run["status"],
        "revision": run["revision"],
        "warnings": run["warnings"],
        "spec": spec,
        "next_step": {
            "needs_answers": "Call answer_questions with {question_id: answer} for the open questions.",
            "in_review": "Ask a human approver to review the spec in the web console.",
            "approved": "Done. Call export_spec for a Markdown copy.",
        }.get(run["status"], ""),
    }


@mcp.tool()
def list_projects() -> list[dict]:
    """List projects with their artifact and run counts."""
    return engine().store.list_projects()


@mcp.tool()
def create_project(name: str) -> dict:
    """Create a project to hold notes and specs. Returns the project id."""
    return engine().create_project(name, ACTOR)


@mcp.tool()
def add_note(project_id: str, title: str, text: str) -> dict:
    """Add a note, meeting transcript, email or document text to a project."""
    return engine().ingest_text(project_id, title, text, ACTOR)


@mcp.tool()
def search_notes(project_id: str, query: str, k: int = 5) -> list[dict]:
    """Hybrid keyword + vector search over a project's notes. Returns passages with their chunk ids."""
    return engine().search(project_id, query, min(max(k, 1), 20))


@mcp.tool()
def generate_spec(project_id: str, provider: str | None = None) -> dict:
    """Draft a cited app spec from all of the project's notes. Open questions are listed instead of guessed."""
    return _summary(engine().start_run(project_id, ACTOR, provider))


@mcp.tool()
def answer_questions(run_id: str, answers: dict[str, str]) -> dict:
    """Answer open questions ({question_id: answer}) and regenerate the spec with the answers as context."""
    return _summary(engine().answer(run_id, answers, ACTOR))


@mcp.tool()
def get_spec(run_id: str) -> dict:
    """Get the latest spec and status for a run."""
    return _summary(engine().run(run_id))


@mcp.tool()
def export_spec(run_id: str) -> str:
    """Export a run's spec as Markdown, with sources for every item."""
    return engine().export_markdown(run_id)


def main() -> None:
    mcp.run("stdio")


if __name__ == "__main__":
    main()
