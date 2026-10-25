"""Streamlit interface for the local LangChain conversational agent."""

from __future__ import annotations

import sqlite3
from typing import TypedDict

import streamlit as st

from agent import (
    AgentSettings,
    add_local_knowledge_context,
    build_agent,
    extract_answer,
)
from knowledge_index import (
    knowledge_poll_interval_seconds,
    synchronize_knowledge_index,
)


class ChatMessage(TypedDict):
    role: str
    content: str


MAX_HISTORY_MESSAGES = 100
KNOWLEDGE_POLL_INTERVAL_SECONDS = knowledge_poll_interval_seconds()

st.set_page_config(page_title="Local LLM Agent", page_icon="💬")


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

messages: list[ChatMessage] = st.session_state.messages
for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask the agent anything"):
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
    langchain_messages, knowledge_sources = add_local_knowledge_context(
        langchain_messages
    )

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                result = build_agent(settings).invoke(
                    {"messages": langchain_messages},
                    config={"recursion_limit": 10},
                )
                answer = extract_answer(result["messages"])
                if knowledge_sources:
                    citations = ", ".join(f"`{source}`" for source in knowledge_sources)
                    answer = f"{answer}\n\n**Local sources:** {citations}"
        except Exception as error:
            st.error(
                "The agent request failed. Confirm that the local API is running "
                f"and the URL, key, and model are correct.\n\n{error}"
            )
        else:
            st.markdown(answer)
            messages.append({"role": "assistant", "content": answer})
