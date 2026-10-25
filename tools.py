"""LangChain tools available to the conversational agent."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import BaseTool, tool

from knowledge_index import search_knowledge_results


MAX_KNOWLEDGE_SEARCHES_PER_TURN = 3


@tool
def current_date_time(timezone: str = "local") -> str:
    """Return the current date and time in the host's local timezone by default, or an IANA timezone such as UTC or America/Los_Angeles."""
    if timezone.casefold() == "local":
        return datetime.now().astimezone().isoformat(timespec="seconds")
    try:
        zone = ZoneInfo(timezone)
    except ZoneInfoNotFoundError as error:
        raise ValueError(
            f"Unknown timezone '{timezone}'. Use 'local' or an IANA timezone "
            "such as UTC."
        ) from error
    return datetime.now(zone).isoformat(timespec="seconds")


@tool
def search_local_knowledge(query: str, max_results: int = 5) -> str:
    """Search local knowledge by a focused query. Returns JSON with status, passages, citations, BM25 scores and query coverage. If status is weak, retry with a more specific or alternate query. If not_found, try a meaning-preserving query once; do not invent evidence."""
    return _search_local_knowledge(query, max_results)


def _search_local_knowledge(query: str, max_results: int) -> str:
    results = search_knowledge_results(query, max_results)
    if not results:
        return json.dumps(
            {
                "status": "not_found",
                "query": query,
                "message": (
                    "No indexed passages matched this query. Try one alternate, "
                    "focused query if local-document evidence is still needed."
                ),
                "results": [],
            },
            ensure_ascii=False,
        )

    status = (
        "matched"
        if max(result.query_coverage for result in results) >= 0.5
        else "weak"
    )
    message = (
        "Lexical matches found. Assess whether the passages support the specific "
        "question; search again with a focused query if they do not."
        if status == "matched"
        else "Only weak lexical overlap was found. Refine the query; if evidence "
        "remains weak, state that the local sources do not establish the answer."
    )
    return json.dumps(
        {
            "status": status,
            "query": query,
            "message": message,
            "results": [
                {
                    "source_id": (
                        "SRC-"
                        + hashlib.sha256(
                            result.citation.encode("utf-8")
                        ).hexdigest()[:8]
                    ),
                    "citation": result.citation,
                    "passage": result.passage,
                    "bm25_score": result.bm25_score,
                    "matched_terms": result.matched_terms,
                    "query_coverage": result.query_coverage,
                }
                for result in results
            ],
        },
        ensure_ascii=False,
    )


def make_limited_knowledge_search_tool(
    max_searches: int = MAX_KNOWLEDGE_SEARCHES_PER_TURN,
) -> BaseTool:
    """Create a search tool that enforces a per-agent-invocation call limit."""
    if max_searches < 1:
        raise ValueError("max_searches must be at least 1.")
    searches_used = 0

    @tool("search_local_knowledge")
    def limited_search_local_knowledge(query: str, max_results: int = 5) -> str:
        """Search local knowledge and return evidence as structured JSON."""
        nonlocal searches_used
        if searches_used >= max_searches:
            return json.dumps(
                {
                    "status": "limit_reached",
                    "query": query,
                    "message": (
                        f"The limit of {max_searches} knowledge searches for this "
                        "turn has been reached. Answer only from evidence already "
                        "retrieved, or state that the local sources do not "
                        "establish the answer."
                    ),
                    "results": [],
                },
                ensure_ascii=False,
            )
        searches_used += 1
        return _search_local_knowledge(query, max_results)

    return limited_search_local_knowledge
