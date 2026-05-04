from dataclasses import dataclass

from .document_loader import RawDocument


@dataclass(frozen=True)
class Chunk:
    id: int
    source: str
    text: str


def chunk_documents(
    docs: list[RawDocument],
    chunk_size: int = 900,
    overlap: int = 150,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    next_id = 0

    for doc in docs:
        text = _normalize_text(doc.text)
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Chunk(id=next_id, source=doc.source, text=chunk_text))
                next_id += 1
            if end >= len(text):
                break
            start = max(0, end - overlap)

    return chunks


def _normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)

