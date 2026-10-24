"""Layout-aware text extraction from uploaded documents.

Converts PDF / DOCX / text into clean, structure-preserving Markdown so that
headings, sections, reading order (multi-column) and tables survive ingestion.

- PDF  -> PyMuPDF4LLM (column detection, heading levels, Markdown tables) plus
          RapidOCR for scanned pages and images embedded in text pages.
- DOCX -> python-docx, preserving heading styles and tables in document order.
- TXT/MD -> passed through unchanged.
"""

import io
import logging

logger = logging.getLogger(__name__)


class UnsupportedFileError(ValueError):
    """Raised when a file's type/content cannot be parsed to text."""


# --------------------------------------------------------------------------- #
# PDF (layout-aware + OCR)
# --------------------------------------------------------------------------- #
def _ocr_images(doc, markdown: str) -> str:
    """OCR images embedded in text pages and append any recovered text.

    pymupdf4llm already OCRs pages that have *no* extractable text (fully scanned
    pages). This covers the remaining case: a normal text PDF that also contains
    an image (a screenshot, chart or signature). Each embedded image is rendered
    and run through RapidOCR, and the recognised text is appended so it becomes
    searchable too.
    """
    try:
        import numpy as np
        import pymupdf
        from rapidocr_onnxruntime import RapidOCR
    except Exception as exc:  # OCR deps missing -> skip gracefully
        logger.warning("OCR skipped (dependency unavailable): %s", exc)
        return markdown

    engine = RapidOCR()
    extra: list[str] = []

    for page in doc:
        if not page.get_text().strip():
            continue  # fully scanned pages already OCR'd by pymupdf4llm
        images = page.get_images(full=True)
        if not images:
            continue

        page_texts: list[str] = []
        for img in images:
            xref = img[0]
            try:
                pix = pymupdf.Pixmap(doc, xref)
                if pix.n >= 5:  # CMYK / alpha -> convert to RGB
                    pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )
                result, _ = engine(arr)
            except Exception as exc:  # one bad image shouldn't fail the doc
                logger.warning("OCR failed for image xref=%s: %s", xref, exc)
                continue
            if result:
                page_texts.extend(
                    line[1].strip() for line in result if line[1].strip()
                )

        if page_texts:
            extra.append(
                f"## Image OCR (page {page.number + 1})\n\n" + "\n".join(page_texts)
            )

    if not extra:
        return markdown
    logger.info("OCR recovered text from %d image-bearing page(s)", len(extra))
    return (markdown + "\n\n" + "\n\n".join(extra)).strip()


def _extract_pdf(data: bytes, enable_ocr: bool) -> str:
    import pymupdf
    import pymupdf4llm

    doc = pymupdf.open(stream=data, filetype="pdf")
    try:
        markdown = pymupdf4llm.to_markdown(doc, show_progress=False)
        if enable_ocr:
            markdown = _ocr_images(doc, markdown)
    finally:
        doc.close()
    return markdown.strip()


# --------------------------------------------------------------------------- #
# DOCX (structure-aware: headings + tables in document order)
# --------------------------------------------------------------------------- #
def _extract_docx(data: bytes) -> str:
    from docx import Document as DocxDocument
    from docx.document import Document as _Doc
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table, _Cell
    from docx.text.paragraph import Paragraph

    document = DocxDocument(io.BytesIO(data))

    def iter_block_items(parent):
        if isinstance(parent, _Doc):
            parent_elm = parent.element.body
        elif isinstance(parent, _Cell):
            parent_elm = parent._tc
        else:
            parent_elm = parent
        for child in parent_elm.iterchildren():
            if isinstance(child, CT_P):
                yield Paragraph(child, parent)
            elif isinstance(child, CT_Tbl):
                yield Table(child, parent)

    def table_to_md(table: Table) -> str:
        rows = [
            [c.text.strip().replace("\n", " ") for c in row.cells]
            for row in table.rows
        ]
        if not rows:
            return ""
        header = rows[0]
        md = ["| " + " | ".join(header) + " |"]
        md.append("|" + "|".join([" --- "] * len(header)) + "|")
        for r in rows[1:]:
            md.append("| " + " | ".join(r) + " |")
        return "\n".join(md)

    parts: list[str] = []
    for block in iter_block_items(document):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if not text:
                continue
            style = (block.style.name or "").lower() if block.style else ""
            if style.startswith("heading"):
                level = "".join(ch for ch in style if ch.isdigit()) or "1"
                parts.append(f"{'#' * min(int(level), 6)} {text}")
            elif style == "title":
                parts.append(f"# {text}")
            else:
                parts.append(text)
        elif isinstance(block, Table):
            md = table_to_md(block)
            if md:
                parts.append(md)
    return "\n\n".join(parts).strip()


# --------------------------------------------------------------------------- #
# Detection + dispatch
# --------------------------------------------------------------------------- #
def _looks_like_pdf(data: bytes) -> bool:
    return data[:5] == b"%PDF-"


def _looks_like_zip(data: bytes) -> bool:
    return data[:2] == b"PK"


def extract_text(
    data: bytes, filename: str | None = None, enable_ocr: bool = True
) -> str:
    """Return layout-aware Markdown text from raw file bytes.

    Detection order: content-signature first, then filename extension, then a
    UTF-8 text fallback. Raises UnsupportedFileError when nothing can parse it.
    """
    name = (filename or "").lower()

    try:
        if _looks_like_pdf(data) or name.endswith(".pdf"):
            logger.info("Layout-aware PDF parse (%s)", filename or "upload")
            return _extract_pdf(data, enable_ocr=enable_ocr)

        if name.endswith(".docx") or (_looks_like_zip(data) and name.endswith(".docx")):
            logger.info("Structure-aware DOCX parse (%s)", filename or "upload")
            return _extract_docx(data)

        return data.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise UnsupportedFileError(
            "Unsupported or non-text file. Supported formats: PDF, DOCX, and "
            "UTF-8 text/markdown."
        ) from exc
    except UnsupportedFileError:
        raise
    except Exception as exc:  # parsing error from pymupdf/docx
        raise UnsupportedFileError(
            f"Failed to extract text from file: {exc}"
        ) from exc
