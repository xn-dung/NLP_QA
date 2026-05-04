from dataclasses import dataclass
from pathlib import Path

from docx import Document as DocxDocument
from pypdf import PdfReader

from .config import SUPPORTED_EXTENSIONS


@dataclass(frozen=True)
class RawDocument:
    source: str
    text: str


def load_documents(folder: Path) -> list[RawDocument]:
    folder.mkdir(parents=True, exist_ok=True)
    docs: list[RawDocument] = []

    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        text = _read_file(path)
        if text.strip():
            docs.append(RawDocument(source=str(path.relative_to(folder)), text=text))

    return docs


def _read_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix == ".docx":
        return _read_docx(path)
    return ""


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def _read_docx(path: Path) -> str:
    doc = DocxDocument(str(path))
    return "\n".join(paragraph.text for paragraph in doc.paragraphs)

