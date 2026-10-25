# Agentic RAG design

This document describes how the conversational agent retrieves local knowledge
and uses it to answer document-specific questions. The existing index remains
lexical SQLite FTS5/BM25; this change makes retrieval a tool the agent can call
instead of injecting search results into every user message.

## Request flow

1. **Route explicit local-document requests.** A narrow cue check for terms
   such as `manual`, `PDF`, `document`, `local file`, or `knowledge base`
   triggers retrieval for that user message. General questions do not trigger
   a knowledge search. This deterministic gate compensates for the local model
   occasionally ignoring a tool-use instruction on clear document questions.
2. **Search with the request as the initial query.** The
   `search_local_knowledge` tool
   synchronizes the index, runs lexical retrieval, and returns structured JSON.
   A result includes its passage, citation label, BM25 score, matched terms,
   query-term coverage, and a stable `source_id` derived from the citation
   label.
3. **Assess and retry.** Results are labeled `matched`, `weak`, or `not_found`.
   A lexical `matched` result is not by itself proof that the passage answers
   the question: the agent must inspect its content. It can refine or replace
   an unhelpful query. The tool enforces a maximum of three knowledge searches
   per agent invocation, including the initial routed search; attempts after
   the limit return `limit_reached`.
4. **Answer from evidence.** If retrieved passages do not establish the answer,
   the agent must say so rather than guess. For supported claims, the model
   includes the exact returned source ID inline, for example
   `[SRC-a1b2c3d4]`. The Streamlit app resolves only IDs appearing in successful
   knowledge-tool results to their actual citation labels. Unknown or
   model-invented IDs are removed, and no separate model-generated source list
   is displayed. If the model omits inline IDs, the interface falls back to a
   source list made only from citations in successful search results.

## Retrieval and evidence boundaries

The tool uses the existing FTS5/BM25 ranking and page-aware PDF citations
described in [the retrieval algorithm deep dive](./knowledge-retrieval-algorithm.md).
There is no embedding model or vector database in the current design. Filename
attribution and the actual retrieved passage are returned as tool evidence;
the model does not get automatic context for unrelated or general turns.

The statuses describe lexical retrieval, not semantic entailment:

- `matched`: the best matching passages cover at least half of the unique
  searchable query terms. The agent still has to judge whether the text is
  relevant to the question.
- `weak`: matches exist, but best-result query coverage is below one half.
  The agent should retry with a more focused query and abstain if useful
  evidence remains unavailable.
- `not_found`: no indexed passage matched. The agent may try a meaning-
  preserving alternate query before abstaining.
- `limit_reached`: the per-invocation search budget is exhausted.

Source IDs are stable for a citation label, allowing a model to cite evidence
without fabricating a filename or page number. The interface acts as the
validator and displays only source labels returned by the knowledge tool.

## Future work

- [ ] Add semantic or hybrid lexical-plus-vector retrieval. Evaluate it against
  the current FTS5/BM25 baseline before changing default retrieval, and retain
  the evidence and citation validation behavior described above.
