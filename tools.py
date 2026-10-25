"""LangChain tools available to the conversational agent."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import BaseTool, tool

from knowledge_index import search_knowledge_index


@tool
def current_date_time(timezone: str = "UTC") -> str:
    """Return the current date and time in an IANA timezone such as UTC, Europe/London, or America/New_York."""
    try:
        zone = ZoneInfo(timezone)
    except ZoneInfoNotFoundError as error:
        raise ValueError(
            f"Unknown timezone '{timezone}'. Use an IANA timezone such as UTC."
        ) from error
    return datetime.now(zone).isoformat(timespec="seconds")


def lookup_local_knowledge(query: str, max_results: int = 5) -> str | None:
    """Return matching passages from the persistent local knowledge index."""
    return search_knowledge_index(query, max_results)


@tool
def search_local_knowledge(query: str, max_results: int = 5) -> str:
    """Search approved local Markdown and text knowledge files for relevant passages."""
    result = lookup_local_knowledge(query, max_results)
    return result or f"No local knowledge matched: {query}"


AGENT_TOOLS: list[BaseTool] = [current_date_time, search_local_knowledge]
