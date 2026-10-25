"""Persistent full-text index for approved local knowledge files."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


DEFAULT_KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"
ALLOWED_KNOWLEDGE_SUFFIXES = {".md", ".pdf", ".txt"}
MAX_KNOWLEDGE_FILES = 100
MAX_FILE_BYTES = 1_000_000
MAX_PDF_FILE_BYTES = 20_000_000
MAX_RESULTS = 10
TARGET_CHUNK_WORDS = 1_000
DEFAULT_POLL_INTERVAL_SECONDS = 20.0
INDEX_FILENAME = ".knowledge_index.sqlite3"
SCHEMA_VERSION = 2
PARSER_VERSION = 3
SEARCH_STOPWORDS = {
    "about",
    "and",
    "are",
    "does",
    "for",
    "from",
    "has",
    "have",
    "how",
    "into",
    "is",
    "it",
    "local",
    "me",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}


@dataclass(frozen=True)
class KnowledgeIndexStats:
    documents: int
    chunks: int


def knowledge_poll_interval_seconds() -> float:
    """Return the configured positive knowledge polling interval."""
    configured_value = os.getenv("KNOWLEDGE_POLL_INTERVAL_SECONDS")
    if configured_value is None:
        return DEFAULT_POLL_INTERVAL_SECONDS
    try:
        interval = float(configured_value)
    except ValueError as error:
        raise RuntimeError(
            "KNOWLEDGE_POLL_INTERVAL_SECONDS must be a number."
        ) from error
    if interval <= 0:
        raise RuntimeError("KNOWLEDGE_POLL_INTERVAL_SECONDS must be positive.")
    return interval


def knowledge_root() -> Path:
    configured_path = os.getenv("LOCAL_KNOWLEDGE_DIR")
    return (
        Path(configured_path).expanduser().resolve()
        if configured_path
        else DEFAULT_KNOWLEDGE_DIR.resolve()
    )


def _index_path(root: Path) -> Path:
    configured_path = os.getenv("LOCAL_KNOWLEDGE_INDEX")
    return (
        Path(configured_path).expanduser().resolve()
        if configured_path
        else root / INDEX_FILENAME
    )


def _connect(root: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(_index_path(root), timeout=10)
    connection.execute("PRAGMA foreign_keys = ON")
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version > SCHEMA_VERSION:
        connection.close()
        raise RuntimeError(
            f"Knowledge index schema {version} is newer than supported schema "
            f"{SCHEMA_VERSION}."
        )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            path TEXT PRIMARY KEY,
            modified_ns INTEGER NOT NULL,
            size_bytes INTEGER NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
            path UNINDEXED,
            heading,
            content,
            tokenize='unicode61'
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS index_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    return connection


def _configuration_fingerprint() -> str:
    if TARGET_CHUNK_WORDS <= 0:
        raise RuntimeError("TARGET_CHUNK_WORDS must be positive.")
    configuration = {
        "allowed_suffixes": sorted(ALLOWED_KNOWLEDGE_SUFFIXES),
        "chunk_unit": "words",
        "max_file_bytes": MAX_FILE_BYTES,
        "max_pdf_file_bytes": MAX_PDF_FILE_BYTES,
        "parser_version": PARSER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "target_chunk_words": TARGET_CHUNK_WORDS,
    }
    serialized = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _split_long_text(text: str) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text)]
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    def flush_current() -> None:
        nonlocal current, current_words
        if current:
            chunks.append("\n\n".join(current))
            current = []
            current_words = 0

    for paragraph in (paragraph for paragraph in paragraphs if paragraph):
        words = paragraph.split()
        if len(words) > TARGET_CHUNK_WORDS:
            flush_current()
            chunks.extend(
                " ".join(words[start : start + TARGET_CHUNK_WORDS])
                for start in range(0, len(words), TARGET_CHUNK_WORDS)
            )
            continue
        if current and current_words + len(words) > TARGET_CHUNK_WORDS:
            flush_current()
        current.append(paragraph)
        current_words += len(words)
    flush_current()
    return chunks


def _parse_chunks(path: Path, text: str) -> list[tuple[str, str]]:
    if path.suffix.casefold() != ".md":
        return [("", chunk) for chunk in _split_long_text(text)]

    chunks: list[tuple[str, str]] = []
    heading = ""
    section_lines: list[str] = []

    def flush_section() -> None:
        section_text = "\n".join(section_lines).strip()
        if section_text:
            chunks.extend((heading, chunk) for chunk in _split_long_text(section_text))

    for line in text.splitlines():
        heading_match = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if heading_match:
            flush_section()
            heading = heading_match.group(1)
            section_lines = []
        else:
            section_lines.append(line)
    flush_section()
    return chunks


def _parse_pdf_chunks(path: Path) -> list[tuple[str, str]]:
    reader = PdfReader(path)
    if reader.is_encrypted and reader.decrypt("") == 0:
        raise ValueError(
            f"PDF is password protected and cannot be indexed: {path.name}"
        )

    chunks: list[tuple[str, str]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if not text or not text.strip():
            continue
        chunks.extend(
            (f"Page {page_number}", chunk) for chunk in _split_long_text(text)
        )
    if not chunks:
        raise ValueError(
            f"PDF contains no extractable text: {path.name}. "
            "Scanned PDFs require OCR before indexing."
        )
    return chunks


def _document_chunks(path: Path) -> list[tuple[str, str]]:
    if path.suffix.casefold() == ".pdf":
        return _parse_pdf_chunks(path)
    return _parse_chunks(path, path.read_text(encoding="utf-8"))


def _knowledge_files(root: Path) -> list[Path]:
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.suffix.casefold() in ALLOWED_KNOWLEDGE_SUFFIXES
        and path.stat().st_size
        <= (MAX_PDF_FILE_BYTES if path.suffix.casefold() == ".pdf" else MAX_FILE_BYTES)
    )
    if len(files) > MAX_KNOWLEDGE_FILES:
        raise RuntimeError(
            f"Knowledge directory contains more than {MAX_KNOWLEDGE_FILES} supported files."
        )
    return files


def synchronize_knowledge_index() -> KnowledgeIndexStats:
    """Create or refresh the index for new, changed, and removed documents."""
    root = knowledge_root()
    if not root.is_dir():
        raise FileNotFoundError(
            f"Knowledge directory does not exist: {root}. "
            "Create it and add .md, .txt, or .pdf files."
        )

    files = _knowledge_files(root)
    current = {
        path.relative_to(root).as_posix(): (path, path.stat()) for path in files
    }
    with _connect(root) as connection:
        fingerprint = _configuration_fingerprint()
        stored_fingerprint_row = connection.execute(
            "SELECT value FROM index_metadata WHERE key = 'configuration_fingerprint'"
        ).fetchone()
        stored_fingerprint = (
            stored_fingerprint_row[0] if stored_fingerprint_row else None
        )
        if stored_fingerprint != fingerprint:
            connection.execute("DELETE FROM chunks")
            connection.execute("DELETE FROM documents")
            connection.execute(
                """
                INSERT INTO index_metadata(key, value)
                VALUES ('configuration_fingerprint', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (fingerprint,),
            )

        indexed = {
            row[0]: (row[1], row[2])
            for row in connection.execute(
                "SELECT path, modified_ns, size_bytes FROM documents"
            )
        }

        for removed_path in indexed.keys() - current.keys():
            connection.execute("DELETE FROM chunks WHERE path = ?", (removed_path,))
            connection.execute("DELETE FROM documents WHERE path = ?", (removed_path,))

        for relative_path, (path, stat) in current.items():
            signature = (stat.st_mtime_ns, stat.st_size)
            if indexed.get(relative_path) == signature:
                continue
            chunks = _document_chunks(path)
            connection.execute("DELETE FROM chunks WHERE path = ?", (relative_path,))
            connection.executemany(
                "INSERT INTO chunks(path, heading, content) VALUES (?, ?, ?)",
                (
                    (relative_path, heading, content)
                    for heading, content in chunks
                ),
            )
            connection.execute(
                """
                INSERT INTO documents(path, modified_ns, size_bytes)
                VALUES (?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    modified_ns = excluded.modified_ns,
                    size_bytes = excluded.size_bytes
                """,
                (relative_path, *signature),
            )

        documents = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunks = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    return KnowledgeIndexStats(documents=documents, chunks=chunks)


def _query_terms(query: str) -> list[str]:
    return [
        term
        for term in re.findall(r"[^\W_]+", query.casefold())
        if len(term) > 1 and term not in SEARCH_STOPWORDS
    ]


def search_knowledge_index(query: str, max_results: int = 5) -> str | None:
    """Search indexed document chunks, refreshing changed files first."""
    terms = _query_terms(query)
    if not terms:
        return None
    if not 1 <= max_results <= MAX_RESULTS:
        raise ValueError(f"max_results must be between 1 and {MAX_RESULTS}.")

    synchronize_knowledge_index()
    root = knowledge_root()
    fts_query = " OR ".join(f'"{term}"' for term in dict.fromkeys(terms))
    with _connect(root) as connection:
        rows = connection.execute(
            """
            SELECT path, heading,
                   snippet(chunks, 2, '', '', ' ... ', 80) AS passage
            FROM chunks
            WHERE chunks MATCH ?
            ORDER BY bm25(chunks)
            LIMIT ?
            """,
            (fts_query, max_results),
        ).fetchall()

    if not rows:
        return None
    return "\n\n".join(
        f"[{path}{f' > {heading}' if heading else ''}] {passage.strip()}"
        for path, heading, passage in rows
    )
