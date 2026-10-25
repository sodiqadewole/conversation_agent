from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
import pytest

import agent as agent_module
from agent import (
    AgentSettings,
    add_local_knowledge_context,
    build_agent,
    extract_answer,
)


def test_default_model_is_qwen_vl() -> None:
    assert AgentSettings().model == "local-qwen-vl"


def test_agent_has_expected_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_tools = []

    def capture_agent(*, model, tools, system_prompt):
        captured_tools.extend(tools)
        return object()

    monkeypatch.setattr(agent_module, "create_agent", capture_agent)
    build_agent(AgentSettings())

    assert {tool.name for tool in captured_tools} == {
        "current_date_time",
        "search_local_knowledge",
    }


def test_add_local_knowledge_context_attaches_matching_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_module,
        "lookup_local_knowledge",
        lambda query, max_results: "[facts.md] Verified local fact.",
    )

    prepared, sources = add_local_knowledge_context(
        [{"role": "user", "content": "Tell me the verified fact"}]
    )

    assert prepared[0]["content"].startswith("Tell me the verified fact")
    assert "[facts.md] Verified local fact." in prepared[0]["content"]
    assert "<local_knowledge_reference>" in prepared[0]["content"]
    assert sources == ["facts.md"]


def test_add_local_knowledge_context_preserves_messages_without_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_module, "lookup_local_knowledge", lambda query, max_results: None
    )
    messages = [{"role": "user", "content": "Hello"}]

    assert add_local_knowledge_context(messages) == (messages, [])


def test_extract_answer_returns_last_assistant_text() -> None:
    messages = [
        HumanMessage(content="Hello"),
        AIMessage(content="First"),
        HumanMessage(content="Again"),
        AIMessage(content="Final answer"),
    ]

    assert extract_answer(messages) == "Final answer"


def test_extract_answer_rejects_empty_response() -> None:
    with pytest.raises(RuntimeError, match="no assistant text"):
        extract_answer([AIMessage(content="")])
