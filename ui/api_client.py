"""Thin HTTP client for the Prax RAG API, used by the Streamlit UI."""

import json
import os
from collections.abc import Iterator

import requests

API_BASE = os.environ.get("PRAX_API_URL", "http://localhost:8000/api/v1")
TIMEOUT = 120


def health() -> dict:
    r = requests.get(f"{API_BASE}/health/db", timeout=10)
    r.raise_for_status()
    return r.json()


def list_documents() -> dict:
    r = requests.get(f"{API_BASE}/documents", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def ingest_text(text: str, source: str, title: str | None = None) -> dict:
    r = requests.post(
        f"{API_BASE}/ingest",
        json={"text": text, "source": source, "title": title},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def ingest_file(filename: str, data: bytes, title: str | None = None) -> dict:
    files = {"file": (filename, data)}
    payload = {"title": title} if title else {}
    r = requests.post(
        f"{API_BASE}/ingest/file", files=files, data=payload, timeout=TIMEOUT
    )
    r.raise_for_status()
    return r.json()


def delete_document(document_id: int) -> dict:
    r = requests.delete(f"{API_BASE}/documents/{document_id}", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def reset() -> dict:
    r = requests.post(
        f"{API_BASE}/documents/reset", params={"confirm": "true"}, timeout=TIMEOUT
    )
    r.raise_for_status()
    return r.json()


def query_stream(question: str, top_k: int | None = None) -> Iterator[dict]:
    """Yield SSE events ({'type': 'sources'|'token'|'done', ...}) from the API."""
    payload = {"question": question}
    if top_k:
        payload["top_k"] = top_k
    with requests.post(
        f"{API_BASE}/query/stream",
        json=payload,
        stream=True,
        timeout=TIMEOUT,
        headers={"Accept": "text/event-stream"},
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                yield json.loads(line[len("data: ") :])
