from __future__ import annotations

from datetime import datetime

import pytest

from tools import current_date_time, lookup_local_knowledge, search_local_knowledge


def test_current_date_time_returns_requested_timezone() -> None:
    result = current_date_time.invoke({"timezone": "UTC"})

    parsed = datetime.fromisoformat(result)
    assert parsed.utcoffset() is not None
    assert parsed.utcoffset().total_seconds() == 0


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

    result = search_local_knowledge.invoke(
        {"query": "local server API", "max_results": 2}
    )

    assert "[deployment.md]" in result
    assert "unrelated.txt" not in result


def test_lookup_local_knowledge_returns_none_without_meaningful_match(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "notes.md").write_text("Project deployment notes.", encoding="utf-8")
    monkeypatch.setenv("LOCAL_KNOWLEDGE_DIR", str(tmp_path))

    assert lookup_local_knowledge("What is the weather?") is None


def test_search_local_knowledge_does_not_follow_symlinks(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path.parent / "private.txt"
    outside.write_text("confidential lookup target", encoding="utf-8")
    (tmp_path / "linked.txt").symlink_to(outside)
    monkeypatch.setenv("LOCAL_KNOWLEDGE_DIR", str(tmp_path))

    result = search_local_knowledge.invoke({"query": "confidential"})

    assert result.startswith("No local knowledge matched")


def test_search_local_knowledge_requires_existing_directory(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCAL_KNOWLEDGE_DIR", str(tmp_path / "missing"))

    with pytest.raises(FileNotFoundError, match="Knowledge directory"):
        search_local_knowledge.invoke({"query": "anything"})
