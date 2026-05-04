import os
from dataclasses import dataclass

import requests


TAVILY_SEARCH_URL = "https://api.tavily.com/search"


@dataclass(frozen=True)
class TavilyResult:
    title: str
    url: str
    content: str
    score: float


def get_tavily_api_key(user_key: str | None = None) -> str:
    if user_key and user_key.strip():
        return user_key.strip()
    return os.getenv("TAVILY_API_KEY", "").strip()


def tavily_search(
    query: str,
    api_key: str,
    topic: str = "general",
    max_results: int = 5,
    search_depth: str = "basic",
) -> list[TavilyResult]:
    response = requests.post(
        TAVILY_SEARCH_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "query": query,
            "topic": topic,
            "search_depth": search_depth,
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": False,
        },
        timeout=45,
    )
    response.raise_for_status()
    data = response.json()
    return [
        TavilyResult(
            title=item.get("title", "").strip(),
            url=item.get("url", "").strip(),
            content=item.get("content", "").strip(),
            score=float(item.get("score", 0.0)),
        )
        for item in data.get("results", [])
    ]
