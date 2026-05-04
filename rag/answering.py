import re

from .indexes import SearchHit


def extractive_answer(query: str, hits: list[SearchHit], max_sentences: int = 4) -> str:
    query_terms = set(_tokenize(query))
    candidates = []

    for hit in hits:
        for sentence in _sentences(hit.text):
            terms = set(_tokenize(sentence))
            if not terms:
                continue
            overlap = len(query_terms & terms)
            score = overlap + min(len(sentence), 500) / 10000
            candidates.append((score, sentence.strip()))

    selected = []
    seen = set()
    for _, sentence in sorted(candidates, reverse=True):
        key = sentence.lower()
        if key not in seen:
            selected.append(sentence)
            seen.add(key)
        if len(selected) >= max_sentences:
            break

    return " ".join(selected) if selected else "Không tìm thấy câu trả lời đủ rõ trong tài liệu."


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+|\n+", text) if part.strip()]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)
