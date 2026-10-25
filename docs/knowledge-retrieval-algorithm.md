# Knowledge Retrieval Algorithm Deep Dive

This document describes how the conversational agent parses local documents,
stores searchable passages, retrieves lexical matches, and ranks them with
SQLite FTS5's BM25 implementation. The system does not use embeddings or a
vector database.

## 1. Retrieval pipeline

The complete path is:

```text
Knowledge files
    |
    v
Parse Markdown, text, or PDF
    |
    v
Split into heading- or page-aware word chunks
    |
    v
Store chunks in SQLite FTS5
    |
    v
Normalize a user question into query terms
    |
    v
Run an OR query over indexed terms
    |
    v
Rank matching chunks with FTS5 BM25
    |
    v
Return the top passages to the language model
```

The implementation is in [`knowledge_index.py`](../knowledge_index.py).

## 2. Document parsing and chunk storage

### 2.1 Supported sources

The index accepts:

- Markdown (`.md`)
- Plain text (`.txt`)
- Text-based PDF (`.pdf`)

Markdown sections retain their nearest heading. PDF pages retain a label such
as `Page 49`. Plain-text chunks have no heading.

Chunking is word-based. Paragraphs are accumulated until adding the next
paragraph would exceed `TARGET_CHUNK_WORDS`. A single oversized paragraph is
split at word boundaries. Chunks never cross Markdown heading or PDF page
boundaries.

### 2.2 Metadata table

The ordinary `documents` table stores one row per source file:

```sql
CREATE TABLE documents (
    path TEXT PRIMARY KEY,
    modified_ns INTEGER NOT NULL,
    size_bytes INTEGER NOT NULL
);
```

The modification time and size form the source signature used to detect added,
changed, and removed files.

### 2.3 FTS5 table

Each parsed passage is inserted into:

```sql
CREATE VIRTUAL TABLE chunks USING fts5(
    path UNINDEXED,
    heading,
    content,
    tokenize='unicode61'
);
```

The columns have different roles:

| Column | Indexed | Purpose |
|---|---:|---|
| `path` | No | Source attribution only |
| `heading` | Yes | Markdown heading or PDF page label |
| `content` | Yes | Extracted passage text |

Because `path` is `UNINDEXED`, a filename cannot make a passage match a query.
Only heading and passage content participate in matching and ranking.

### 2.4 Configuration fingerprint

The index stores a SHA-256 fingerprint of the parser and chunk configuration:

$$
F = \operatorname{SHA256}(\operatorname{CanonicalJSON}(C))
$$

where $C$ contains the schema version, parser version, supported suffixes,
file-size limits, chunking unit, and target chunk size. If the current
fingerprint differs from the stored fingerprint, all generated chunks are
rebuilt from the source files.

## 3. Lexical query construction

Suppose the user asks:

> How do I configure motion detection on the camera?

### 3.1 Term extraction

The application case-folds the question and extracts Unicode word sequences.
Let the raw token sequence be:

$$
R(q) = \operatorname{RegexWords}(\operatorname{CaseFold}(q))
$$

For the example:

```text
["how", "do", "i", "configure", "motion", "detection", "on",
 "the", "camera"]
```

Terms of length one and configured stopwords are removed:

$$
T(q) =
\left[
t \in R(q)
\;\middle|\;
|t| > 1 \land t \notin S
\right]
$$

where $S$ is the stopword set. The result is approximately:

```text
["configure", "motion", "detection", "camera"]
```

This is lexical normalization only. It performs no stemming, lemmatization,
synonym expansion, or semantic encoding.

### 3.2 FTS query

Duplicate terms are removed while preserving their first-seen order. Each term
is quoted, and the terms are joined with `OR`:

$$
Q = \text{"configure"} \lor \text{"motion"}
    \lor \text{"detection"} \lor \text{"camera"}
$$

The generated FTS5 query is:

```text
"configure" OR "motion" OR "detection" OR "camera"
```

A chunk is therefore eligible when at least one query term occurs in its
indexed heading or content.

## 4. Unicode61 tokenization

The table uses SQLite's `unicode61` tokenizer. It:

- Separates text into Unicode-aware tokens.
- Is case-insensitive by default.
- Removes diacritics from Latin characters by default.
- Records term positions and frequencies in an inverted index.

Conceptually, the inverted index maps:

```text
motion       -> [(row 48, positions ...), (row 49, positions ...), ...]
detection    -> [(row 48, positions ...), (row 49, positions ...), ...]
password     -> [(row 16, positions ...), ...]
```

SQLite uses these posting lists to find candidate rows without scanning every
stored chunk.

## 5. BM25 ranking

The query orders candidates using:

```sql
ORDER BY bm25(chunks)
```

SQLite FTS5 uses fixed constants:

$$
k_1 = 1.2
$$

$$
b = 0.75
$$

### 5.1 Main score

For chunk $D$ and query $Q$, SQLite computes:

$$
\operatorname{bm25}(D,Q)
=
-1
\sum_{i=1}^{nPhrase}
\operatorname{IDF}(q_i)
\frac{
f(q_i,D)(k_1+1)
}{
f(q_i,D)
+
k_1
\left(
1-b+b\frac{|D|}{avgdl}
\right)
}
$$

where:

- $nPhrase$ is the number of query phrases.
- $q_i$ is query phrase $i$.
- $f(q_i,D)$ is the weighted frequency of $q_i$ in chunk $D$.
- $|D|$ is the number of indexed tokens in chunk $D$.
- $avgdl$ is the average indexed-token count across all chunk rows.

The leading $-1$ is specific to SQLite FTS5. Traditional BM25 gives better
matches larger positive values. SQLite negates the score so ordinary ascending
SQL order returns the best match first:

$$
\text{better match} \Rightarrow \text{smaller, usually more negative score}
$$

### 5.2 Inverse document frequency

For query phrase $q_i$:

$$
\operatorname{IDF}(q_i)
=
\log
\left(
\frac{
N-n(q_i)+0.5
}{
n(q_i)+0.5
}
\right)
$$

where:

- $N$ is the number of rows in the FTS table. In this application, rows are
  chunks, not source files.
- $n(q_i)$ is the number of chunk rows containing $q_i$ at least once.

A term found in relatively few chunks receives more discriminating weight than
a term found in most chunks.

### 5.3 Column-weighted phrase frequency

SQLite defines phrase frequency as:

$$
f(q_i,D)
=
\sum_{c=1}^{nCol}
w_c \cdot n(q_i,c)
$$

where:

- $nCol$ is the number of FTS columns.
- $w_c$ is the configured weight for column $c$.
- $n(q_i,c)$ is the number of occurrences of $q_i$ in column $c$.

This project calls `bm25(chunks)` without explicit weights, so every indexed
column receives the default weight $1.0$. The `path` column is `UNINDEXED`
and contributes nothing. Therefore:

$$
f(q_i,D)
=
n(q_i,\text{heading})
+
n(q_i,\text{content})
$$

### 5.4 Term-frequency saturation

The term-frequency contribution is:

$$
\operatorname{TF}(q_i,D)
=
\frac{
f(q_i,D)(k_1+1)
}{
f(q_i,D)
+
k_1
\left(
1-b+b\frac{|D|}{avgdl}
\right)
}
$$

Increasing term frequency improves relevance, but the gain diminishes as
$f(q_i,D)$ grows. Mentioning `motion` ten times is useful, but it is not worth
ten times as much as mentioning it once.

### 5.5 Length normalization

The normalization component is:

$$
1-b+b\frac{|D|}{avgdl}
$$

If $|D| = avgdl$, this component equals 1. Shorter chunks receive less length
penalty; longer chunks receive more. With $b=0.75$, document length matters
substantially but does not completely dominate term frequency.

## 6. Worked ranking example

Consider the camera manual query:

```text
motion detection configuration
```

Assume a simplified ten-chunk corpus:

$$
N = 10,\quad avgdl = 800
$$

Assume all three terms occur in two chunks:

$$
n(q_i) = 2
$$

Their IDF is:

$$
\operatorname{IDF}(q_i)
=
\log\left(\frac{10-2+0.5}{2+0.5}\right)
=
\log(3.4)
\approx
1.2238
$$

Use these hypothetical chunk statistics:

| Chunk | Length | `motion` | `detection` | `configuration` |
|---|---:|---:|---:|---:|
| Page 49 passage | 700 | 4 | 3 | 1 |
| Page 50 passage | 900 | 2 | 2 | 1 |

For `motion` on the Page 49 passage:

$$
\operatorname{TF}
=
\frac{
4(1.2+1)
}{
4+1.2
\left(
1-0.75+0.75\frac{700}{800}
\right)
}
\approx
1.7298
$$

$$
\operatorname{IDF}\cdot\operatorname{TF}
\approx
1.2238 \cdot 1.7298
\approx
2.1168
$$

After calculating every term:

| Chunk | Per-term weighted contributions | SQLite BM25 |
|---|---|---:|
| Page 49 | $2.1168 + 1.9760 + 1.2897$ | $-5.3825$ |
| Page 50 | $1.6255 + 1.6255 + 1.1642$ | $-4.4153$ |

Because $-5.3825 < -4.4153$, Page 49 ranks first. This corresponds to the
current manual, where Page 49 begins the motion-detection configuration
instructions and Page 50 continues with detection parameters.

## 7. Snippet extraction

After ranking, the query calls:

```sql
snippet(chunks, 2, '', '', ' ... ', 80)
```

The arguments mean:

1. Read column 2, `content`.
2. Add no opening highlight marker.
3. Add no closing highlight marker.
4. Insert ` ... ` where text is omitted.
5. Return at most 80 FTS tokens.

SQLite selects a fragment intended to contain as many distinct query terms as
possible. It prefers fragments near the start of a column or following `.` or
`:`. The snippet is a presentation result; it does not alter BM25 ranking.

## 8. Context passed to the agent

Automatic retrieval requests at most three passages. Results are formatted as:

```text
[cmip7923wlpr-32r_um_20200917.pdf > Page 49]
Network Camera User Manual ... Set Motion Detection ...
```

The passages are attached to the latest user message inside a local-knowledge
reference block. The language model receives passage text, source filename, and
page attribution. The UI appends the retrieved source labels to the response.

## 9. Latency benchmark

### 9.1 Environment and corpus

Measurements were taken on October 24, 2026, using:

| Component | Value |
|---|---|
| Platform | Linux 7.0.0-34-generic, x86-64 |
| Python | 3.12.3 |
| SQLite | 3.45.1 |
| pypdf | 6.21.2 |
| Source documents | 2 |
| Source size | 1,859,629 bytes |
| PDF pages | 126 |
| Indexed chunks | 127 |

The corpus consisted of the local knowledge README and the 126-page Network
Camera User Manual. The retrieval query was:

```text
motion detection configuration
```

### 9.2 Methodology

- **Fresh index build:** five separate, empty SQLite database files. Each run
  parsed all sources, extracted all PDF pages, chunked content, and built FTS5.
- **Unchanged synchronization:** 200 runs against an up-to-date index. Each run
  scanned source metadata and verified that no document required reindexing.
- **End-to-end retrieval:** 300 runs through `search_knowledge_index()`. Each
  run synchronized sources, opened SQLite, created the FTS query, ranked
  results, extracted three snippets, and formatted the passages.
- **SQLite query only:** 1,000 executions on one open SQLite connection. This
  isolated FTS matching, BM25 ranking, and snippet extraction.
- Percentiles are empirical nearest-rank values. OS filesystem caches were not
  cleared, so fresh builds used new databases but may have benefited from
  cached source-file reads after the first run.

### 9.3 Results

All values are milliseconds.

| Operation | Runs | Min | Median (p50) | Mean | p95 | Max |
|---|---:|---:|---:|---:|---:|---:|
| Fresh full index build | 5 | 1,717.955 | 1,820.422 | 1,829.604 | 1,961.605 | 1,961.605 |
| Unchanged synchronization | 200 | 1.941 | 5.206 | 5.165 | 7.061 | 9.588 |
| End-to-end retrieval | 300 | 4.746 | 9.326 | 9.172 | 12.594 | 16.948 |
| SQLite FTS/BM25 query only | 1,000 | 0.154 | 0.195 | 0.348 | 0.821 | 1.157 |

The dominant indexing cost is PDF text extraction. The raw FTS/BM25 query is
sub-millisecond at this corpus size. Most end-to-end retrieval time comes from
source synchronization, opening the database, and formatting snippets.

These measurements do not include language-model inference, network transport
to vLLM, Streamlit rendering, or model tool-selection latency. They describe
only local knowledge indexing and retrieval.

## 10. Retrieval properties and limitations

### Strengths

- Fully local and deterministic.
- No embedding model or vector service.
- Sub-millisecond median database search at the current corpus size.
- Exact filename, Markdown heading, and PDF page attribution.
- Automatic refresh and configuration-aware rebuilding.

### Limitations

- Lexical matching requires word overlap.
- No stemming or synonym expansion.
- An expression such as `moving-object alert` may not match `motion detection`
  unless another query term overlaps.
- `OR` retrieval favors recall; a chunk matching only one common remaining term
  can be eligible.
- BM25 ranks chunks, not entire source documents.
- Extracted PDF quality depends on the PDF text layer.

An embedding or hybrid retriever could improve semantic recall, but it would
introduce an embedding model, vector storage, additional indexing latency, and
new relevance-tuning decisions.

## 11. References

- [SQLite FTS5 documentation](https://www.sqlite.org/fts5.html)
- [SQLite FTS5 BM25 function](https://www.sqlite.org/fts5.html#the_bm25_function)
- [SQLite FTS5 snippet function](https://www.sqlite.org/fts5.html#the_snippet_function)
- [Okapi BM25 overview](https://en.wikipedia.org/wiki/Okapi_BM25)
