"""Deterministic local scoring of papers from a keyword + author config.

The score is a small integer; the weights are exposed in
``[filters]`` of the TOML config so users can tune sensitivity
without editing code.

Scoring rules (all configurable via the config; defaults shown):

* Blocked author match: returns ``-999`` immediately and short-circuits.
* Preferred author match: ``+6`` per matched author.
* Strong keyword hit: ``+4`` each.
* Medium keyword hit: ``+2`` each.
* Weak keyword hit: ``+1`` each, capped at 4 contributions.
* Negative keyword hit: ``-4`` each.
* No strong/medium/author signal and no ``cs.LO`` category: ``-4``.
* Bonus for the ``cs.LO`` category and combinations with ``cs.AI`` /
  ``cs.LG`` (this is hardcoded historical behaviour, configurable via
  ``[filters].category_bonuses`` if present in the future).
"""

from __future__ import annotations

from typing import Any

from .models import LocalScore, Paper
from .utils import normalize_text, unique_preserve_order


def match_keywords(text: str, keywords: list[str]) -> list[str]:
    """Return all keywords whose normalized form is a substring of the normalized text."""
    text_n = normalize_text(text)
    hits: list[str] = []
    for kw in keywords:
        kw_n = normalize_text(kw)
        if kw_n and kw_n in text_n:
            hits.append(kw)
    return unique_preserve_order(hits)


def match_preferred_authors(authors: list[str], seeds: list[str]) -> list[str]:
    """Match author seeds against the paper author list (normalized substrings)."""
    hits: list[str] = []
    author_norms = [normalize_text(a) for a in authors]
    for seed in seeds:
        s = normalize_text(seed)
        if not s:
            continue
        for author_n in author_norms:
            if s == author_n or s in author_n or author_n in s:
                hits.append(seed)
                break
    return unique_preserve_order(hits)


def match_blocked_authors(authors: list[str], blocked: list[str]) -> list[str]:
    """Same matching logic as preferred authors but used to block."""
    return match_preferred_authors(authors, blocked)


def compute_local_score(paper: Paper, cfg: dict[str, Any]) -> LocalScore:
    """Compute the deterministic local score for a paper.

    All configurable thresholds and lexicons live in ``cfg["filters"]``.
    """
    filters = cfg["filters"]
    preferred_authors = filters.get("preferred_authors", [])
    blocked_authors = filters.get("blocked_authors_exact", [])
    strong = filters.get("positive_keywords_strong", [])
    medium = filters.get("positive_keywords_medium", [])
    weak = filters.get("positive_keywords_weak", [])
    negative = filters.get("negative_keywords", [])

    title_text = paper.title
    abstract_text = paper.abstract
    categories_text = " ".join(paper.categories)

    author_hits = match_preferred_authors(paper.authors, preferred_authors)
    blocked_hits = match_blocked_authors(paper.authors, blocked_authors)

    if blocked_hits:
        return LocalScore(
            score=-999,
            author_hits=blocked_hits,
            reasons=[f"blocked author match: {', '.join(blocked_hits[:3])}"],
        )

    strong_hits = unique_preserve_order(
        match_keywords(title_text, strong) + match_keywords(abstract_text, strong)
    )
    medium_hits = unique_preserve_order(
        match_keywords(title_text, medium) + match_keywords(abstract_text, medium)
    )
    weak_hits = unique_preserve_order(
        match_keywords(title_text, weak) + match_keywords(abstract_text, weak)
    )
    negative_hits = unique_preserve_order(
        match_keywords(title_text, negative) + match_keywords(abstract_text, negative)
    )

    score = 0
    reasons: list[str] = []

    if author_hits:
        bonus = 6 * len(author_hits)
        score += bonus
        reasons.append(f"preferred author match (+{bonus})")

    if strong_hits:
        bonus = 4 * len(strong_hits)
        score += bonus
        reasons.append(f"strong topic hits (+{bonus})")

    if medium_hits:
        bonus = 2 * len(medium_hits)
        score += bonus
        reasons.append(f"medium topic hits (+{bonus})")

    if weak_hits:
        bonus = 1 * min(len(weak_hits), 4)
        score += bonus
        reasons.append(f"weak bridge hits (+{bonus})")

    cats = {c.lower() for c in paper.categories}
    if "cs.lo" in cats:
        score += 2
        reasons.append("cs.LO bonus (+2)")
    if "cs.lo" in cats and "cs.ai" in cats:
        score += 2
        reasons.append("cs.AI+cs.LO bonus (+2)")
    if "cs.lo" in cats and "cs.lg" in cats:
        score += 2
        reasons.append("cs.LG+cs.LO bonus (+2)")
    if not strong_hits and not medium_hits and not author_hits and "cs.lo" not in cats:
        score -= 4
        reasons.append("no strong topic signal (-4)")
    if negative_hits:
        penalty = 4 * len(negative_hits)
        score -= penalty
        reasons.append(f"out-of-scope hits (-{penalty})")
    if (
        not strong_hits
        and not medium_hits
        and match_keywords(categories_text, ["cs.LO"])
    ):
        score += 1
        reasons.append("category bridge (+1)")

    return LocalScore(
        score=score,
        author_hits=author_hits,
        keyword_hits_strong=strong_hits,
        keyword_hits_medium=medium_hits,
        keyword_hits_weak=weak_hits,
        negative_hits=negative_hits,
        reasons=reasons,
    )


def compact_local_why(local: LocalScore) -> str:
    """One-line human-readable summary of why a paper got its local score."""
    parts: list[str] = []
    if local.author_hits:
        parts.append(f"author match: {', '.join(local.author_hits[:2])}")
    if local.keyword_hits_strong:
        parts.append(f"strong: {', '.join(local.keyword_hits_strong[:3])}")
    elif local.keyword_hits_medium:
        parts.append(f"medium: {', '.join(local.keyword_hits_medium[:3])}")
    if local.negative_hits:
        parts.append(f"negative: {', '.join(local.negative_hits[:2])}")
    return "; ".join(parts[:3]) if parts else "local topic filter"
