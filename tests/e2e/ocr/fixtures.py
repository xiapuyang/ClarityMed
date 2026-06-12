"""Programmatic fixture generation for OCR e2e tests.

Each helper writes a minimal but *real* file of the requested format
to ``tmp_path``. We avoid checked-in binary fixtures because (a) they
bloat the repo, (b) they go stale silently when a fixture-generator
library version drifts, and (c) tests that read real format files
verify the provider's actual decode path — what we want to guard.

Every helper takes ``tmp_path: Path`` and returns the resulting
``Path``. The on-disk content is the smallest payload the format
spec allows that still embeds the literal ``MARKER_TEXT`` so the
e2e tests can assert "the provider's output contains the planted
string."
"""

from __future__ import annotations

from pathlib import Path

MARKER_TEXT = "claritymed-ocr-marker"


def make_pdf(tmp_path: Path, name: str = "doc.pdf") -> Path:
    """A one-page PDF containing ``MARKER_TEXT``."""
    from reportlab.pdfgen import canvas

    p = tmp_path / name
    c = canvas.Canvas(str(p))
    c.drawString(100, 750, MARKER_TEXT)
    c.save()
    return p


def make_png(tmp_path: Path, name: str = "scan.png") -> Path:
    """A small PNG with ``MARKER_TEXT`` rasterised onto it.

    The image carries readable text in the default font so vision
    LLMs and rapidocr can extract it. Small (320x80) so e2e remains
    fast on CPU OCR backends.
    """
    return _make_text_image(tmp_path / name, fmt="PNG")


def make_jpeg(tmp_path: Path, name: str = "scan.jpg") -> Path:
    return _make_text_image(tmp_path / name, fmt="JPEG")


def make_webp(tmp_path: Path, name: str = "scan.webp") -> Path:
    return _make_text_image(tmp_path / name, fmt="WEBP")


def make_bmp(tmp_path: Path, name: str = "scan.bmp") -> Path:
    return _make_text_image(tmp_path / name, fmt="BMP")


def make_tiff(tmp_path: Path, name: str = "scan.tiff") -> Path:
    return _make_text_image(tmp_path / name, fmt="TIFF")


def make_gif(tmp_path: Path, name: str = "scan.gif") -> Path:
    return _make_text_image(tmp_path / name, fmt="GIF")


def make_docx(tmp_path: Path, name: str = "report.docx") -> Path:
    from docx import Document

    p = tmp_path / name
    doc = Document()
    doc.add_paragraph(MARKER_TEXT)
    doc.save(str(p))
    return p


def make_xlsx(tmp_path: Path, name: str = "labs.xlsx") -> Path:
    from openpyxl import Workbook

    p = tmp_path / name
    wb = Workbook()
    ws = wb.active
    ws["A1"] = MARKER_TEXT
    wb.save(str(p))
    return p


def make_pptx(tmp_path: Path, name: str = "slides.pptx") -> Path:
    from pptx import Presentation
    from pptx.util import Inches

    p = tmp_path / name
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    tb = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(6), Inches(1))
    tb.text_frame.text = MARKER_TEXT
    prs.save(str(p))
    return p


def make_html(tmp_path: Path, name: str = "page.html") -> Path:
    p = tmp_path / name
    p.write_text(f"<html><body><p>{MARKER_TEXT}</p></body></html>", encoding="utf-8")
    return p


def make_rtf(tmp_path: Path, name: str = "note.rtf") -> Path:
    """Minimal RTF document containing ``MARKER_TEXT``."""
    p = tmp_path / name
    p.write_text(
        r"{\rtf1\ansi\deff0 {\fonttbl {\f0 Helvetica;}}\f0\fs24 "
        + MARKER_TEXT
        + r"\par}",
        encoding="utf-8",
    )
    return p


def make_odt(tmp_path: Path, name: str = "note.odt") -> Path:
    """Minimal ODT zip containing ``MARKER_TEXT``.

    ODT requires four files at minimum for pandoc to parse: ``mimetype``,
    ``META-INF/manifest.xml``, ``content.xml``, and ``styles.xml``. The
    earlier fixture omitted ``styles.xml`` and pandoc 3.x rejected the
    package outright (``Could not find styles.xml``) — this version
    includes a stub stylesheet so the document round-trips cleanly.
    """
    import zipfile

    p = tmp_path / name
    office_ns = 'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"'
    text_ns = 'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"'
    style_ns = 'xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0"'
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("mimetype", "application/vnd.oasis.opendocument.text")
        z.writestr(
            "META-INF/manifest.xml",
            '<?xml version="1.0"?>'
            '<manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0">'
            '<manifest:file-entry manifest:full-path="/" '
            'manifest:media-type="application/vnd.oasis.opendocument.text"/>'
            '<manifest:file-entry manifest:full-path="content.xml" manifest:media-type="text/xml"/>'
            '<manifest:file-entry manifest:full-path="styles.xml" manifest:media-type="text/xml"/>'
            "</manifest:manifest>",
        )
        z.writestr(
            "content.xml",
            '<?xml version="1.0"?>'
            f"<office:document-content {office_ns} {text_ns}>"
            "<office:body><office:text><text:p>"
            + MARKER_TEXT
            + "</text:p></office:text></office:body>"
            "</office:document-content>",
        )
        z.writestr(
            "styles.xml",
            '<?xml version="1.0"?>'
            f"<office:document-styles {office_ns} {style_ns}>"
            "<office:styles/>"
            "</office:document-styles>",
        )
    return p


def make_epub(tmp_path: Path, name: str = "book.epub") -> Path:
    """Minimal EPUB zip containing ``MARKER_TEXT``.

    EPUB = zip with mimetype + container.xml + an OPF + at least one
    XHTML content document. We hand-roll the smallest structure pandoc
    accepts so we don't pull yet another library just for this fixture.
    """
    import zipfile

    p = tmp_path / name
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        z.writestr(
            "META-INF/container.xml",
            '<?xml version="1.0"?>'
            '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
            '<rootfiles><rootfile full-path="content.opf" media-type="application/oebps-package+xml"/></rootfiles>'
            "</container>",
        )
        z.writestr(
            "content.opf",
            '<?xml version="1.0"?>'
            '<package xmlns="http://www.idpf.org/2007/opf" version="2.0" unique-identifier="BookId">'
            '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>t</dc:title>'
            '<dc:language>en</dc:language><dc:identifier id="BookId">id</dc:identifier></metadata>'
            '<manifest><item id="c" href="ch.xhtml" media-type="application/xhtml+xml"/></manifest>'
            '<spine><itemref idref="c"/></spine>'
            "</package>",
        )
        z.writestr(
            "ch.xhtml",
            '<?xml version="1.0"?>'
            '<html xmlns="http://www.w3.org/1999/xhtml"><body><p>'
            + MARKER_TEXT
            + "</p></body></html>",
        )
    return p


# --- helpers ---------------------------------------------------------


def _make_text_image(path: Path, *, fmt: str) -> Path:
    """Render ``MARKER_TEXT`` to an image and save in ``fmt``.

    320x80 white background, black text in the default Pillow font.
    Large enough for rapidocr / vision LLMs to recognise the glyphs;
    small enough that the e2e suite stays under a second per case.
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (480, 120), color="white")
    draw = ImageDraw.Draw(img)
    # Pillow's default font is small but legible enough for OCR at
    # 480x120. We avoid loading a system TTF to keep tests
    # cross-platform.
    draw.text((20, 40), MARKER_TEXT, fill="black")
    img.save(str(path), format=fmt)
    return path
