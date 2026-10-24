"""Content hashing used for deduplication, versioning and incremental indexing."""

import hashlib
import re

_WS_RE = re.compile(r"\s+")


def content_hash(text: str) -> str:
    """Return a stable SHA-256 of ``text``, normalised for whitespace/case.

    Normalising means documents differing only in spacing or casing are treated
    as identical, which is what deduplication and change-detection want.
    """
    normalized = _WS_RE.sub(" ", text or "").strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
