from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
import pytest

import agent as agent_module
from agent import (
    AgentSettings,
    append_retrieval_sources_when_uncited,
    build_agent,
    deduplicate_local_sources,
    extract_answer,
    extract_retrieval_citations,
    extract_response_metrics,
    invoke_agent,
    LOCAL_KNOWLEDGE_CUE,
    remove_model_local_sources,
    resolve_verified_citations,
)


def test_default_model_is_qwen_vl() -> None:
    assert AgentSettings().model == "local-qwen-vl"


@pytest.mark.parametrize(
    ("message", "should_search"),
    [
        ("According to the local camera manual, how do I configure motion?", True),
        ("What does the PDF say about motion alerts?", True),
        ("What is the capital of France?", False),
    ],
)
def test_local_document_cue(message: str, should_search: bool) -> None:
    assert bool(LOCAL_KNOWLEDGE_CUE.search(message)) is should_search


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


def test_extract_retrieval_citations_uses_only_actual_tool_results() -> None:
    messages = [
        ToolMessage(
            name="search_local_knowledge",
            tool_call_id="call-1",
            content=(
                '{"status":"matched","results":['
                '{"citation":"manual.pdf > Page 4"},'
                '{"citation":"manual.pdf > Page 5"},'
                '{"citation":"manual.pdf > Page 4"}]}'
            ),
        ),
        ToolMessage(
            name="current_date_time",
            tool_call_id="call-2",
            content="2026-01-01T00:00:00+00:00",
        ),
        ToolMessage(
            name="search_local_knowledge",
            tool_call_id="call-3",
            content='{"status":"not_found","results":[]}',
        ),
    ]

    assert extract_retrieval_citations(messages) == [
        "manual.pdf > Page 4",
        "manual.pdf > Page 5",
    ]


def test_extract_retrieval_citations_returns_empty_without_tool_call() -> None:
    assert extract_retrieval_citations([AIMessage(content="General answer")]) == []


def test_append_retrieval_sources_when_model_omits_inline_citations() -> None:
    messages = [
        ToolMessage(
            name="search_local_knowledge",
            tool_call_id="call-1",
            content=(
                '{"status":"matched","results":['
                '{"citation":"manual.pdf > Page 4"}]}'
            ),
        )
    ]

    assert append_retrieval_sources_when_uncited("Answer.", messages) == (
        "Answer.\n\n**Local sources:** `manual.pdf > Page 4`"
    )


def test_append_retrieval_sources_does_not_duplicate_inline_citation() -> None:
    messages = [
        ToolMessage(
            name="search_local_knowledge",
            tool_call_id="call-1",
            content=(
                '{"status":"matched","results":['
                '{"citation":"manual.pdf > Page 4"}]}'
            ),
        )
    ]

    assert append_retrieval_sources_when_uncited(
        "Answer [manual.pdf > Page 4].", messages
    ) == "Answer [manual.pdf > Page 4]."


def test_resolve_verified_citations_maps_only_retrieved_source_ids() -> None:
    messages = [
        ToolMessage(
            name="search_local_knowledge",
            tool_call_id="call-1",
            content=(
                '{"status":"matched","results":['
                '{"source_id":"SRC-1234abcd",'
                '"citation":"manual.pdf > Page 4"}]}'
            ),
        )
    ]

    assert resolve_verified_citations(
        "Supported claim [SRC-1234abcd]. Unknown [SRC-deadbeef].",
        messages,
    ) == "Supported claim [manual.pdf > Page 4]. Unknown."


def test_resolve_verified_citations_ignores_not_found_results() -> None:
    messages = [
        ToolMessage(
            name="search_local_knowledge",
            tool_call_id="call-1",
            content=(
                '{"status":"not_found","results":['
                '{"source_id":"SRC-1234abcd",'
                '"citation":"manual.pdf > Page 4"}]}'
            ),
        )
    ]

    assert resolve_verified_citations("Claim [SRC-1234abcd].", messages) == (
        "Claim."
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
