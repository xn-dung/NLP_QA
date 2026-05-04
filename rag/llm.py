import os

import requests

from .indexes import SearchHit
from .web_search import TavilyResult


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def get_gemini_api_key(user_key: str | None = None) -> str:
    if user_key and user_key.strip():
        return user_key.strip()
    return os.getenv("GEMINI_API_KEY", "").strip()


def generate_rag_answer(
    question: str,
    hits: list[SearchHit],
    api_key: str,
    model: str = DEFAULT_GEMINI_MODEL,
    web_results: list[TavilyResult] | None = None,
) -> str:
    has_web = bool(web_results)
    payload = {
        "system_instruction": {
            "parts": [
                {
                    "text": (
                        "Ban la tro ly RAG tra loi bang tieng Viet. Hay tong hop cau tra loi hoan chinh "
                        "tu Document context. Document context co the bang tieng Anh, tieng Viet, hoac ngon ngu khac; "
                        "hay hieu noi dung goc, dich/dien giai chinh xac sang tieng Viet, khong bo qua nguon tieng Anh. "
                        "Neu Web context co du lieu, chi dung no de bo sung thong tin "
                        "lien quan hoac cap nhat, khong thay the noi dung tai lieu neu khong co can cu. "
                        "Khong bia thong tin ngoai context. Neu context khong du, noi ro phan nao chua du. "
                        "Neu document va web mau thuan, neu ro su khac nhau. Cau tra loi nen mach lac, "
                        "co ket luan truc tiep, va dan nguon bang ten file hoac ten trang trong ngoac. "
                        "Khong ket thuc bang cau dang do, ngoac mo, hoac y chua hoan thanh."
                    )
                }
            ]
        },
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            f"Question:\n{question}\n\n"
                            "The user question may be Vietnamese while the document may be English. "
                            "Match the meaning across languages before answering.\n\n"
                            f"Use web search: {'yes' if has_web else 'no'}\n\n"
                            f"Document context:\n{_format_document_context(hits)}\n\n"
                            f"Web context:\n{_format_web_context(web_results or [])}"
                        )
                    }
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 1400,
        },
    }
    response = requests.post(
        GEMINI_API_URL.format(model=model),
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return _extract_gemini_text(data)


def _extract_gemini_text(data: dict) -> str:
    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError("Gemini returned no candidates.")

    parts = candidates[0].get("content", {}).get("parts", [])
    text = "\n".join(part.get("text", "") for part in parts if part.get("text", "")).strip()
    finish_reason = candidates[0].get("finishReason", "")

    if not text:
        raise RuntimeError(f"Gemini returned empty text. finishReason={finish_reason}")

    if finish_reason == "MAX_TOKENS":
        text += "\n\n[Output may be truncated because Gemini reached the token limit.]"

    return text


def _format_document_context(hits: list[SearchHit], max_chars: int = 5000) -> str:
    parts = []
    total = 0

    for hit in hits:
        text = hit.text.strip().replace("\n", " ")
        item = f"[{hit.source}] {text}"
        if total + len(item) > max_chars:
            break
        parts.append(item)
        total += len(item)

    return "\n\n".join(parts) if parts else "No document context."


def _format_web_context(results: list[TavilyResult], max_chars: int = 3500) -> str:
    parts = []
    total = 0

    for result in results:
        item = f"[{result.title}] {result.url}\n{result.content}"
        if total + len(item) > max_chars:
            break
        parts.append(item)
        total += len(item)

    return "\n\n".join(parts) if parts else "No web context."
