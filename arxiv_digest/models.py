"""Plain dataclasses used across the package."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Paper:
    """A single arXiv paper as we observe it from RSS or the API."""

    arxiv_id: str
    version: str | None
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    primary_category: str | None
    link: str
    pdf_link: str | None
    published: datetime | None
    updated: datetime | None
    source: str
    announce_type: str | None = None


@dataclass
class LocalScore:
    """The deterministic local score and the reasons that produced it."""

    score: int
    author_hits: list[str] = field(default_factory=list)
    keyword_hits_strong: list[str] = field(default_factory=list)
    keyword_hits_medium: list[str] = field(default_factory=list)
    keyword_hits_weak: list[str] = field(default_factory=list)
    negative_hits: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


@dataclass
class FinalDecision:
    """The final keep / drop decision after the optional LLM rerank."""

    arxiv_id: str
    local_score: int
    llm_score: int | None
    final_score: float
    decision: str
    why: str
    tags: list[str]
    local_reasons: list[str]
