from __future__ import annotations

import os
from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

import knowledge_index
from knowledge_index import search_knowledge_index, synchronize_knowledge_index


def _write_text_pdf(path: Path, text: str) -> None:
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): writer._add_object(font)}
            )
        }
    )
    content = DecodedStreamObject()
    escaped_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    content.set_data(
        f"BT /F1 12 Tf 72 720 Td ({escaped_text}) Tj ET".encode("latin-1")
    )
    page[NameObject("/Contents")] = writer._add_object(content)
    with path.open("wb") as output:
        writer.write(output)


def test_index_parses_markdown_into_heading_aware_chunks(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "profile.md").write_text(
        "# Profile\n\nGeneral introduction.\n\n"
        "## Research Projects\n\nBuilt a multimodal dialogue simulator.",
        encoding="utf-8",
    )
    monkeypatch.setenv("LOCAL_KNOWLEDGE_DIR", str(tmp_path))

    stats = synchronize_knowledge_index()
    result = search_knowledge_index("multimodal dialogue")

    assert stats.documents == 1
    assert stats.chunks == 2
    assert result is not None
    assert "[profile.md > Research Projects]" in result
    assert "Built a multimodal dialogue simulator." in result


def test_filename_is_not_part_of_search_index(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "confidential-roadmap.md").write_text(
        "# Notes\n\nOranges and apples.",
        encoding="utf-8",
    )
    monkeypatch.setenv("LOCAL_KNOWLEDGE_DIR", str(tmp_path))

    assert search_knowledge_index("confidential roadmap") is None


def test_index_refreshes_changed_files(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = tmp_path / "facts.txt"
    document.write_text("The release color is blue.", encoding="utf-8")
    monkeypatch.setenv("LOCAL_KNOWLEDGE_DIR", str(tmp_path))
    assert search_knowledge_index("blue") is not None

    document.write_text("The release color is amber.", encoding="utf-8")
    stat = document.stat()
    os.utime(document, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))

    assert search_knowledge_index("blue") is None
    assert search_knowledge_index("amber") is not None


def test_long_paragraph_is_split_by_word_count(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "long.txt").write_text(
        "one two three four five six seven eight nine ten eleven",
        encoding="utf-8",
    )
    monkeypatch.setenv("LOCAL_KNOWLEDGE_DIR", str(tmp_path))
    monkeypatch.setattr(knowledge_index, "TARGET_CHUNK_WORDS", 5)

    stats = synchronize_knowledge_index()

    assert stats.documents == 1
    assert stats.chunks == 3


def test_chunk_configuration_change_rebuilds_index_automatically(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "facts.txt").write_text(
        "one two three four five six seven eight nine ten eleven twelve",
        encoding="utf-8",
    )
    monkeypatch.setenv("LOCAL_KNOWLEDGE_DIR", str(tmp_path))
    monkeypatch.setattr(knowledge_index, "TARGET_CHUNK_WORDS", 20)
    initial = synchronize_knowledge_index()

    monkeypatch.setattr(knowledge_index, "TARGET_CHUNK_WORDS", 4)
    rebuilt = synchronize_knowledge_index()

    assert initial.chunks == 1
    assert rebuilt.chunks == 3


def test_new_document_is_added_during_next_synchronization(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCAL_KNOWLEDGE_DIR", str(tmp_path))
    initial = synchronize_knowledge_index()
    (tmp_path / "new-source.md").write_text(
        "# New information\n\nThe launch codename is Firefly.",
        encoding="utf-8",
    )

    refreshed = synchronize_knowledge_index()

    assert initial.documents == 0
    assert refreshed.documents == 1
    assert search_knowledge_index("Firefly") is not None


@pytest.mark.parametrize("value", ["0", "-1", "not-a-number"])
def test_poll_interval_must_be_positive_number(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KNOWLEDGE_POLL_INTERVAL_SECONDS", value)

    with pytest.raises(RuntimeError, match="KNOWLEDGE_POLL_INTERVAL_SECONDS"):
        knowledge_index.knowledge_poll_interval_seconds()


def test_pdf_is_extracted_and_indexed_by_page(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_text_pdf(
        tmp_path / "architecture.pdf",
        "Project Firefly uses retrieval augmented generation.",
    )
    monkeypatch.setenv("LOCAL_KNOWLEDGE_DIR", str(tmp_path))

    stats = synchronize_knowledge_index()
    result = search_knowledge_index("retrieval augmented generation")

    assert stats.documents == 1
    assert stats.chunks == 1
    assert result is not None
    assert "[architecture.pdf > Page 1]" in result
    assert "Project Firefly" in result


def test_pdf_without_extractable_text_reports_ocr_requirement(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with (tmp_path / "scan.pdf").open("wb") as output:
        writer.write(output)
    monkeypatch.setenv("LOCAL_KNOWLEDGE_DIR", str(tmp_path))

    with pytest.raises(ValueError, match="require OCR"):
        synchronize_knowledge_index()
