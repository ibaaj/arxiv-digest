"""Small pure helpers used across the package.

All functions here are deterministic and side-effect-free, which makes
them easy to unit test.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import UTC, datetime

ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"

_NORMALIZE_TABLE = str.maketrans(
    {
        "-": " ",
        "_": " ",
        "/": " ",
        "&": " and ",
        ":": " ",
        ";": " ",
        ",": " ",
        "(": " ",
        ")": " ",
        "[": " ",
        "]": " ",
        "{": " ",
        "}": " ",
        "\n": " ",
        "\t": " ",
    }
)


def utcnow() -> datetime:
    """Return a timezone-aware UTC ``datetime`` for the current moment."""
    return datetime.now(UTC)


def ensure_utc(dt: datetime) -> datetime:
    """Convert ``dt`` to UTC; assume naive inputs are already UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def dt_to_iso(dt: datetime) -> str:
    """Format a ``datetime`` as a fixed-format UTC ISO-8601 string."""
    return ensure_utc(dt).strftime(ISO_FMT)


def dt_from_iso(value: str) -> datetime:
    """Parse a fixed-format UTC ISO-8601 string back to a ``datetime``."""
    return datetime.strptime(value, ISO_FMT).replace(tzinfo=UTC)


def normalize_text(text: str) -> str:
    """Lowercase, replace separators with spaces, and collapse whitespace.

    The normalization is intentionally light: it folds punctuation that
    commonly separates words ("knowledge-graphs" → "knowledge graphs")
    so that substring keyword matching does not miss hyphenated forms.
    """
    text = text.lower().translate(_NORMALIZE_TABLE)
    return re.sub(r"\s+", " ", text).strip()


def unique_preserve_order(items: Iterable[str]) -> list[str]:
    """Deduplicate while preserving insertion order."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def estimate_tokens(text: str) -> int:
    """Heuristic token estimate from text length.

    Uses ``max(chars/4, words*1.3)`` which is close to what tiktoken
    would report for English-and-light-markup text. We only use this
    for cost estimation before sending the request; the actual usage
    from the API is reported separately.
    """
    if not text:
        return 0
    by_chars = len(text) / 4.0
    by_words = len(re.findall(r"\S+", text)) * 1.3
    return int(max(by_chars, by_words))


def parse_arxiv_id_from_abs_url(url: str) -> tuple[str, str | None]:
    """Return ``(arxiv_id, version)`` from an ``/abs/`` URL.

    Example: ``https://arxiv.org/abs/2401.12345v2`` →
    ``("2401.12345", "2")``.
    """
    m = re.search(r"/abs/([^/?#]+)", url)
    if not m:
        raise ValueError(f"Cannot parse arXiv id from url: {url}")
    raw = m.group(1)
    vm = re.match(r"(.+?)v(\d+)$", raw)
    if vm:
        return vm.group(1), vm.group(2)
    return raw, None


def clean_abstract(text: str) -> str:
    """Collapse whitespace inside an abstract."""
    return re.sub(r"\s+", " ", text).strip()


def extract_json(text: str) -> str:
    """Strip code fences and isolate the first JSON object/array.

    The LLM is asked to return JSON only, but real-world models sometimes
    wrap their output in ```` ```json ```` fences or add a short preamble.
    This helper is permissive about both.
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    if text.startswith("{") or text.startswith("["):
        return text
    match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if match:
        return match.group(1)
    raise ValueError("Model output did not contain JSON.")
