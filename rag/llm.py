import os

import requests

from .indexes import SearchHit


DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct:fastest"
HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"


def get_hf_token(user_token: str | None = None) -> str:
    if user_token and user_token.strip():
        return user_token.strip()
    return os.getenv("HF_TOKEN", "").strip()


def generate_rag_answer(
    question: str,
    hits: list[SearchHit],
    token: str,
    model: str = DEFAULT_LLM_MODEL,
) -> str:
    context = _format_context(hits)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You answer in Vietnamese. Use only the provided context. "
                    "If the context is not enough, say that the document does not contain enough information. "
                    "Keep the answer concise and cite source names in parentheses."
                ),
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nContext:\n{context}",
            },
        ],
        "temperature": 0.2,
        "max_tokens": 450,
    }
    response = requests.post(
        HF_CHAT_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def _format_context(hits: list[SearchHit], max_chars: int = 6000) -> str:
    parts = []
    total = 0

    for hit in hits:
        text = hit.text.strip().replace("\n", " ")
        item = f"[{hit.source}] {text}"
        if total + len(item) > max_chars:
            break
        parts.append(item)
        total += len(item)

    return "\n\n".join(parts)
