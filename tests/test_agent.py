from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
import pytest

import agent as agent_module
from agent import (
    AgentSettings,
    add_local_knowledge_context,
    build_agent,
    deduplicate_local_sources,
    extract_answer,
    extract_response_metrics,
    remove_model_local_sources,
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


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        (
            "Answer body.\n\n**Local sources:** `guide.pdf > Page 1`",
            "Answer body.",
        ),
        (
            "Answer body.\n\nLocal sources: `guide.pdf > Page 1`",
            "Answer body.",
        ),
        ("Answer without citations.", "Answer without citations."),
    ],
)
def test_remove_model_local_sources(answer: str, expected: str) -> None:
    assert remove_model_local_sources(answer) == expected


def test_deduplicate_local_sources_keeps_final_verified_line() -> None:
    answer = (
        "Answer body.\n\n"
        "**Local sources:** model-cited.pdf\n\n"
        "**Local sources:** verified.pdf > Page 4"
    )

    assert deduplicate_local_sources(answer) == (
        "Answer body.\n\n**Local sources:** verified.pdf > Page 4"
    )


def test_extract_response_metrics_aggregates_tool_round_usage() -> None:
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_local_knowledge",
                    "args": {"query": "test"},
                    "id": "call-1",
                }
            ],
            usage_metadata={
                "input_tokens": 20,
                "output_tokens": 5,
                "total_tokens": 25,
            },
        ),
        AIMessage(
            content="Found it.",
            usage_metadata={
                "input_tokens": 30,
                "output_tokens": 10,
                "total_tokens": 40,
            },
            response_metadata={
                "model_name": "local-qwen-vl",
                "finish_reason": "stop",
                "id": "request-123",
                "system_fingerprint": "vllm-test",
            },
        ),
    ]

    metrics = extract_response_metrics(messages, latency_ms=1234.56)

    assert metrics == {
        "latency_ms": 1234.6,
        "input_tokens": 50,
        "output_tokens": 15,
        "total_tokens": 65,
        "model": "local-qwen-vl",
        "finish_reason": "stop",
        "request_id": "request-123",
        "server_fingerprint": "vllm-test",
    }


def test_extract_response_metrics_handles_missing_usage() -> None:
    metrics = extract_response_metrics([AIMessage(content="Hi")], latency_ms=20)

    assert metrics == {"latency_ms": 20}
