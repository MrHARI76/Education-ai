"""Shared test setup: sample-file builders and an isolated environment.

Environment variables are set BEFORE the app is imported, so tests use a
temporary data folder and an in-memory Qdrant (no Docker needed).
"""

import os
import tempfile
from pathlib import Path

import pytest

os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="nexus-test-")
os.environ["QDRANT_URL"] = ":memory:"
os.environ["OLLAMA_URL"] = "http://127.0.0.1:9"  # nothing listens here -> "Ollama down"

SAMPLE_PARAGRAPH = (
    "Photosynthesis is the process plants use to turn sunlight, water and carbon dioxide "
    "into glucose and oxygen. Chlorophyll in the chloroplasts absorbs light energy. "
    "The light reactions happen in the thylakoid membranes, while the Calvin cycle "
    "fixes carbon in the stroma. "
)


def make_pdf(path: Path) -> Path:
    """Create a two-page PDF with reportlab."""
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path))
    for page_title in ("Light reactions", "Calvin cycle"):
        text = c.beginText(50, 780)
        text.textLine(page_title)
        for i in range(0, len(SAMPLE_PARAGRAPH), 80):  # wrap long text into lines
            text.textLine(SAMPLE_PARAGRAPH[i : i + 80])
        c.drawText(text)
        c.showPage()
    c.save()
    return path


def make_docx(path: Path) -> Path:
    """Create a Word file with a title, two headings, paragraphs and a table."""
    from docx import Document

    doc = Document()
    doc.add_heading("Photosynthesis", level=0)  # level 0 = "Title" style
    doc.add_heading("Light reactions", level=1)
    doc.add_paragraph(SAMPLE_PARAGRAPH)
    doc.add_heading("Calvin cycle", level=1)
    doc.add_paragraph(SAMPLE_PARAGRAPH)
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text, table.cell(0, 1).text = "Input", "Output"
    table.cell(1, 0).text, table.cell(1, 1).text = "Carbon dioxide", "Glucose"
    doc.save(str(path))
    return path


def make_pptx(path: Path) -> Path:
    """Create a two-slide deck with titles, body text and speaker notes."""
    from pptx import Presentation

    prs = Presentation()
    for title in ("Light reactions", "Calvin cycle"):
        slide = prs.slides.add_slide(prs.slide_layouts[1])  # layout 1 = "Title and Content"
        slide.shapes.title.text = title
        slide.placeholders[1].text = SAMPLE_PARAGRAPH
        slide.notes_slide.notes_text_frame.text = f"Remember to explain {title}."
    prs.save(str(path))
    return path


@pytest.fixture
def sample_files(tmp_path: Path) -> dict[str, Path]:
    """One sample file of each supported type."""
    return {
        "pdf": make_pdf(tmp_path / "bio.pdf"),
        "docx": make_docx(tmp_path / "bio.docx"),
        "pptx": make_pptx(tmp_path / "bio.pptx"),
    }
