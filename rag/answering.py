import re

from .indexes import SearchHit


def extractive_answer(query: str, hits: list[SearchHit], max_sentences: int = 3) -> str:
    query_terms = set(_tokenize(query))
    candidates = []

    for hit in hits:
        for sentence in _sentences(hit.text):
            sentence = _clean_sentence(sentence)
            terms = set(_tokenize(sentence))
            if not terms or len(sentence) < 30 or len(sentence) > 700:
                continue
            overlap = len(query_terms & terms)
            if query_terms and overlap == 0:
                continue
            density = overlap / max(len(terms), 1)
            score = overlap + density + hit.score
            candidates.append((score, sentence, hit.source))

    selected = []
    seen = set()
    for _, sentence, source in sorted(candidates, reverse=True):
        key = sentence.lower()
        if key not in seen:
            selected.append(f"- `{source}`: {sentence}")
            seen.add(key)
        if len(selected) >= max_sentences:
            break

    if not selected:
        return "Không tìm thấy câu trả lời đủ rõ trong tài liệu."

    return "\n".join(selected)


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def _clean_sentence(sentence: str) -> str:
    sentence = re.sub(r"\s+", " ", sentence)
    sentence = re.sub(r"\[\s*Page\s+(\d+)\s*\]", r"[Page \1]", sentence)
    return sentence.strip()
