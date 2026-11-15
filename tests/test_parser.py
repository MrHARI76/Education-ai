"""Tests for app/parser.py using small files generated in code."""

from pathlib import Path

import pytest

from app.parser import ParseError, check_extension, chunk_sections, parse_document


def test_pdf_has_one_section_per_page(sample_files: dict[str, Path]) -> None:
    """Each PDF page becomes a section labelled 'page N'."""
    sections, headings = parse_document(sample_files["pdf"], "bio.pdf")
    assert [s.location for s in sections] == ["page 1", "page 2"]
    assert "Photosynthesis" in sections[0].text
    assert headings == []


def test_docx_groups_paragraphs_under_headings(sample_files: dict[str, Path]) -> None:
    """Headings become structure hints and table rows are kept."""
    sections, headings = parse_document(sample_files["docx"], "bio.docx")
    assert headings == ["Photosynthesis", "Light reactions", "Calvin cycle"]
    calvin = [s for s in sections if s.heading == "Calvin cycle"][0]
    assert "Carbon dioxide | Glucose" in calvin.text
    assert all(s.file_type == "docx" for s in sections)


def test_pptx_reads_titles_body_and_notes(sample_files: dict[str, Path]) -> None:
    """Slide titles are headings; speaker notes are prefixed with 'Notes:'."""
    sections, headings = parse_document(sample_files["pptx"], "bio.pptx")
    assert headings == ["Light reactions", "Calvin cycle"]
    assert sections[1].location == "slide 2"
    assert "Notes: Remember to explain Calvin cycle." in sections[1].text


def test_chunks_keep_metadata(sample_files: dict[str, Path]) -> None:
    """Every chunk carries source info and a running chunk_index."""
    sections, _ = parse_document(sample_files["pptx"], "bio.pptx")
    chunks = chunk_sections(sections, chunk_size=200, chunk_overlap=30)
    assert len(chunks) > len(sections)  # small chunk_size forces splitting
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert {"text", "source", "file_type", "location", "heading"} <= chunks[0].keys()
    assert all(len(c["text"]) <= 200 for c in chunks)


def test_too_little_text_is_rejected(tmp_path: Path) -> None:
    """A file with almost no text gives the 'scanned PDF' message."""
    from docx import Document

    doc = Document()
    doc.add_paragraph("Hi")
    path = tmp_path / "tiny.docx"
    doc.save(str(path))
    with pytest.raises(ParseError, match="No readable text"):
        parse_document(path, "tiny.docx")


@pytest.mark.parametrize("name,hint", [("old.doc", ".docx"), ("old.ppt", ".pptx")])
def test_legacy_formats_get_friendly_message(name: str, hint: str) -> None:
    """Old Office formats are rejected with a 'Save As' hint."""
    with pytest.raises(ParseError, match=hint):
        check_extension(name)


def test_unknown_extension_rejected() -> None:
    """Anything other than pdf/docx/pptx is refused."""
    with pytest.raises(ParseError, match="Unsupported"):
        check_extension("notes.txt")


def test_corrupt_file_gives_clear_error(tmp_path: Path) -> None:
    """A broken file raises ParseError instead of crashing."""
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"this is not a pdf")
    with pytest.raises(ParseError):
        parse_document(path, "broken.pdf")
