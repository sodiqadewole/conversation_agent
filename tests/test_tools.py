from __future__ import annotations

import json
from datetime import datetime

import pytest

import tools as tools_module
from tools import (
    current_date_time,
    make_limited_knowledge_search_tool,
    search_local_knowledge,
)


def test_current_date_time_returns_requested_timezone() -> None:
    result = current_date_time.invoke({"timezone": "UTC"})

    parsed = datetime.fromisoformat(result)
    assert parsed.utcoffset() is not None
    assert parsed.utcoffset().total_seconds() == 0


def test_current_date_time_defaults_to_host_local_timezone() -> None:
    before = datetime.now().astimezone()
    parsed = datetime.fromisoformat(current_date_time.invoke({}))
    after = datetime.now().astimezone()

    assert parsed.utcoffset() == before.utcoffset()
    assert abs(parsed.timestamp() - before.timestamp()) < 1


def test_current_date_time_rejects_unknown_timezone() -> None:
    with pytest.raises(ValueError, match="Unknown timezone"):
        current_date_time.invoke({"timezone": "Not/A_Zone"})


def test_search_local_knowledge_returns_ranked_matches(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "deployment.md").write_text(
        "Start the local server with vLLM. The server exposes an OpenAI API.",
        encoding="utf-8",
    )
    (tmp_path / "unrelated.txt").write_text(
        "This file discusses gardening.",
        encoding="utf-8",
    )
    monkeypatch.setenv("LOCAL_KNOWLEDGE_DIR", str(tmp_path))

    result = json.loads(search_local_knowledge.invoke(
        {"query": "local server API", "max_results": 2}
    ))

    assert result["status"] == "matched"
    assert result["results"][0]["citation"] == "deployment.md"
    assert result["results"][0]["source_id"].startswith("SRC-")
    assert "unrelated.txt" not in str(result)


def test_search_local_knowledge_reports_not_found(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "notes.md").write_text("Project deployment notes.", encoding="utf-8")
    monkeypatch.setenv("LOCAL_KNOWLEDGE_DIR", str(tmp_path))

    result = json.loads(
        search_local_knowledge.invoke({"query": "What is the weather?"})
    )

    assert result["status"] == "not_found"
    assert result["results"] == []


def test_search_local_knowledge_reports_weak_match(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "notes.md").write_text("alpha is mentioned here.", encoding="utf-8")
    monkeypatch.setenv("LOCAL_KNOWLEDGE_DIR", str(tmp_path))

    result = json.loads(
        search_local_knowledge.invoke(
            {"query": "alpha beta gamma delta"}
        )
    )

    assert result["status"] == "weak"
    assert result["results"][0]["query_coverage"] == 0.25


def test_search_local_knowledge_does_not_follow_symlinks(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path.parent / "private.txt"
    outside.write_text("confidential lookup target", encoding="utf-8")
    (tmp_path / "linked.txt").symlink_to(outside)
    monkeypatch.setenv("LOCAL_KNOWLEDGE_DIR", str(tmp_path))

    result = json.loads(search_local_knowledge.invoke({"query": "confidential"}))

    assert result["status"] == "not_found"


def test_search_local_knowledge_requires_existing_directory(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCAL_KNOWLEDGE_DIR", str(tmp_path / "missing"))

    with pytest.raises(FileNotFoundError, match="Knowledge directory"):
        search_local_knowledge.invoke({"query": "anything"})


def test_limited_search_tool_enforces_per_instance_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_search(query: str, max_results: int) -> str:
        nonlocal calls
        calls += 1
        return json.dumps({"status": "matched", "query": query, "results": []})

    monkeypatch.setattr(tools_module, "_search_local_knowledge", fake_search)
    search_tool = make_limited_knowledge_search_tool(max_searches=2)

    assert json.loads(search_tool.invoke({"query": "first"}))["status"] == "matched"
    assert json.loads(search_tool.invoke({"query": "second"}))["status"] == "matched"
    limited_result = json.loads(search_tool.invoke({"query": "third"}))

    assert limited_result["status"] == "limit_reached"
    assert calls == 2
