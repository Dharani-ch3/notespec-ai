import asyncio

from engine import mcp_server
from engine.config import Settings
from engine.core import Engine
from engine.store import Store


def test_mcp_tools_are_registered():
    tools = asyncio.run(mcp_server.mcp.list_tools())
    names = {t.name for t in tools}
    assert {"create_project", "add_note", "search_notes", "generate_spec", "answer_questions", "export_spec"} <= names


def test_mcp_flow_through_tool_functions(monkeypatch):
    monkeypatch.setattr(mcp_server, "_engine", Engine(Settings(db_path=":memory:"), Store(":memory:")))
    pid = mcp_server.create_project("Via MCP")["id"]
    mcp_server.add_note(pid, "kickoff", "Nurses must flag urgent symptoms. Who approves edits?")
    assert mcp_server.search_notes(pid, "urgent")[0]["chunk_id"]
    out = mcp_server.generate_spec(pid)
    assert out["status"] == "needs_answers" and "answer_questions" in out["next_step"]
    answers = {q["id"]: "The clinic manager, on AWS, staff sign in with SSO" for q in out["spec"]["open_questions"]}
    out = mcp_server.answer_questions(out["run_id"], answers)
    assert out["status"] == "in_review"
    assert "# Via MCP" in mcp_server.export_spec(out["run_id"])
