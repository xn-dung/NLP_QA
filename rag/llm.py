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
    payload = {
        "system_instruction": {
            "parts": [
                {
                    "text": (
                        "Tra loi bang tieng Viet. Chi duoc dung du lieu trong context da cung cap. "
                        "Neu context khong du, noi ro tai lieu va ket qua web chua du thong tin. "
                        "Tra loi ngan gon, uu tien dung su that va kem ten nguon trong ngoac."
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
                            f"Document context:\n{_format_document_context(hits)}\n\n"
                            f"Web context:\n{_format_web_context(web_results or [])}"
                        )
                    }
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 500,
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
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


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
