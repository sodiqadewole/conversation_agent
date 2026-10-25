"""Streamlit interface for the local LangChain conversational agent."""

from __future__ import annotations

import sqlite3
import time
from typing import TypedDict

import streamlit as st

from agent import (
    AgentSettings,
    ResponseMetrics,
    append_retrieval_sources_when_uncited,
    deduplicate_local_sources,
    extract_answer,
    extract_response_metrics,
    invoke_agent,
    remove_model_local_sources,
    resolve_verified_citations,
)
from knowledge_index import (
    knowledge_poll_interval_seconds,
    synchronize_knowledge_index,
)


class ChatMessage(TypedDict):
    role: str
    content: str


class StoredChatMessage(ChatMessage, total=False):
    metrics: ResponseMetrics


MAX_HISTORY_MESSAGES = 100
KNOWLEDGE_POLL_INTERVAL_SECONDS = knowledge_poll_interval_seconds()

st.set_page_config(page_title="Local LLM Agent", page_icon="💬")
st.markdown(
    """
    <style>
    [data-testid="stMetric"] {
        padding: 0.25rem 0;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.8rem;
    }
    [data-testid="stMetricValue"] {
        font-size: 1rem;
    }
    [data-testid="stMetricValue"] > div {
        font-size: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.fragment(run_every=KNOWLEDGE_POLL_INTERVAL_SECONDS)
def render_knowledge_index_status() -> None:
    """Refresh the index and display its current size."""
    try:
        index_stats = synchronize_knowledge_index()
    except (OSError, RuntimeError, UnicodeError, sqlite3.Error) as error:
        st.error(f"Knowledge index unavailable: {error}")
    else:
        st.caption(
            f"Knowledge index: {index_stats.documents} documents, "
            f"{index_stats.chunks} passages "
            f"(polling every {KNOWLEDGE_POLL_INTERVAL_SECONDS:g}s)"
        )


def render_response_metrics(metrics: ResponseMetrics) -> None:
    """Render latency and token counts in compact metric cards."""
    displayed_metrics: list[tuple[str, str]] = [
        ("Latency", f"{metrics['latency_ms'] / 1000:.2f} s")
    ]
    if "input_tokens" in metrics:
        displayed_metrics.extend(
            [
                ("Input tokens", f"{metrics['input_tokens']:,}"),
                ("Output tokens", f"{metrics['output_tokens']:,}"),
                ("Total tokens", f"{metrics['total_tokens']:,}"),
            ]
        )
    columns = st.columns(len(displayed_metrics))
    for column, (label, value) in zip(columns, displayed_metrics):
        column.metric(label, value)

    metadata: list[str] = []
    for label, key in (
        ("Model", "model"),
        ("Finish", "finish_reason"),
        ("Request", "request_id"),
        ("vLLM", "server_fingerprint"),
    ):
        if key in metrics:
            metadata.append(f"{label}: `{metrics[key]}`")
    if metadata:
        st.caption(" · ".join(metadata))


st.title("Local LLM Agent")
st.caption("A LangChain conversational agent powered by your local vLLM API.")

defaults = AgentSettings.from_environment()
with st.sidebar:
    st.header("Connection")
    base_url = st.text_input("API base URL", value=defaults.base_url)
    model_name = st.text_input("Model name", value=defaults.model)
    api_key = st.text_input("API key", value=defaults.api_key, type="password")
    render_knowledge_index_status()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

messages: list[StoredChatMessage] = st.session_state.messages
for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(
            deduplicate_local_sources(message["content"])
            if message["role"] == "assistant"
            else message["content"]
        )
        if message["role"] == "assistant" and "metrics" in message:
            render_response_metrics(message["metrics"])

if prompt := st.chat_input("Ask the agent anything"):
    request_started_at = time.perf_counter()
    user_message: ChatMessage = {"role": "user", "content": prompt}
    messages.append(user_message)
    with st.chat_message("user"):
        st.markdown(prompt)

    settings = AgentSettings(
        base_url=base_url.strip(),
        api_key=api_key,
        model=model_name.strip(),
    )
    langchain_messages = [
        {"role": message["role"], "content": message["content"]}
        for message in messages[-MAX_HISTORY_MESSAGES:]
    ]
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                result = invoke_agent(settings, langchain_messages)
                latency_ms = (time.perf_counter() - request_started_at) * 1000
                answer = remove_model_local_sources(
                    extract_answer(result["messages"])
                )
                answer = resolve_verified_citations(answer, result["messages"])
                answer = append_retrieval_sources_when_uncited(
                    answer, result["messages"]
                )
                response_metrics = extract_response_metrics(
                    result["messages"],
                    latency_ms=latency_ms,
                )
        except Exception as error:
            st.error(
                "The agent request failed. Confirm that the local API is running "
                f"and the URL, key, and model are correct.\n\n{error}"
            )
        else:
            st.markdown(answer)
            render_response_metrics(response_metrics)
            messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "metrics": response_metrics,
                }
            )
