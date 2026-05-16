"""Unit tests for the deterministic local scorer."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from arxiv_digest.models import Paper
from arxiv_digest.scoring import (
    compute_local_score,
    match_blocked_authors,
    match_keywords,
    match_preferred_authors,
)


@pytest.fixture
def cfg():
    """A small, deterministic filter config used across the tests."""
    return {
        "filters": {
            "preferred_authors": ["Jane Doe"],
            "blocked_authors_exact": ["Spam Author"],
            "positive_keywords_strong": ["sim-to-real", "robot learning"],
            "positive_keywords_medium": ["imitation learning"],
            "positive_keywords_weak": ["robot"],
            "negative_keywords": ["stock prediction"],
            "broad_candidate_threshold": 0,
            "local_keep_threshold": 6,
            "final_keep_threshold": 25,
            "local_weight": 1.0,
            "llm_weight": 0.35,
        }
    }


def _paper(
    title: str = "",
    abstract: str = "",
    authors: list[str] | None = None,
    categories: list[str] | None = None,
    arxiv_id: str = "2401.00001",
) -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        version=None,
        title=title,
        abstract=abstract,
        authors=authors or [],
        categories=categories or ["cs.LG"],
        primary_category=(categories or ["cs.LG"])[0],
        link=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_link=None,
        published=datetime(2024, 1, 1, tzinfo=UTC),
        updated=datetime(2024, 1, 1, tzinfo=UTC),
        source="api",
    )


class TestMatchKeywords:
    def test_basic_substring(self):
        assert match_keywords("hello world", ["world"]) == ["world"]

    def test_hyphenated_match(self):
        # Normalization treats "-" as space, so both forms match.
        assert match_keywords("sim-to-real transfer", ["sim to real"]) == ["sim to real"]

    def test_no_match(self):
        assert match_keywords("hello", ["world"]) == []

    def test_case_insensitive(self):
        assert match_keywords("HELLO", ["hello"]) == ["hello"]


class TestMatchAuthors:
    def test_exact_match(self):
        assert match_preferred_authors(["Jane Doe"], ["Jane Doe"]) == ["Jane Doe"]

    def test_substring_match(self):
        # Common abbreviation in arXiv listings.
        assert match_preferred_authors(["J. Doe"], ["J. Doe"]) == ["J. Doe"]

    def test_no_match(self):
        assert match_preferred_authors(["Alice"], ["Bob"]) == []

    def test_blocked_uses_same_logic(self):
        assert match_blocked_authors(["Spam Author"], ["Spam Author"]) == ["Spam Author"]


class TestComputeLocalScore:
    def test_blocked_author_short_circuits(self, cfg):
        p = _paper(
            title="A great paper",
            abstract="sim-to-real robot learning imitation learning",
            authors=["Spam Author"],
        )
        score = compute_local_score(p, cfg)
        assert score.score == -999
        assert "blocked" in score.reasons[0].lower()

    def test_no_signal_penalty(self, cfg):
        p = _paper(title="Quantum chess", abstract="An unrelated paper.")
        score = compute_local_score(p, cfg)
        # No matches, no cs.LO bonus → -4 floor reason.
        assert score.score == -4
        assert any("no strong topic signal" in r for r in score.reasons)

    def test_strong_hits(self, cfg):
        p = _paper(
            title="Sim-to-real for robot learning",
            abstract="We study sim to real transfer for manipulators.",
        )
        score = compute_local_score(p, cfg)
        # Two distinct strong keyword hits → +4*2 = +8.
        assert score.score >= 8
        assert score.keyword_hits_strong  # both should be detected

    def test_preferred_author_bonus(self, cfg):
        p = _paper(
            title="Quantum chess",
            abstract="A paper about chess endgames only.",
            authors=["Jane Doe"],
        )
        score = compute_local_score(p, cfg)
        # +6 author bonus; the no-signal penalty is skipped because
        # author_hits is non-empty.
        assert score.score == 6
        assert score.author_hits == ["Jane Doe"]

    def test_negative_keyword_penalty(self, cfg):
        p = _paper(
            title="Robot learning for stock prediction",
            abstract="We use robot learning to predict stocks.",
        )
        score = compute_local_score(p, cfg)
        # robot learning (+4), weak 'robot' (+1), negative (-4)
        # → 4 + 1 - 4 = 1.
        assert score.score == 1
        assert score.negative_hits == ["stock prediction"]

    def test_cs_lo_bonus(self, cfg):
        p = _paper(
            title="Logic stuff",
            abstract="Nothing in scope.",
            categories=["cs.LO"],
        )
        score = compute_local_score(p, cfg)
        # +2 cs.LO bonus; +1 category bridge (no strong/medium, cs.LO
        # present); the no-signal penalty is skipped when cs.LO is
        # present.
        assert score.score == 3

    def test_cs_lo_plus_lg_double_bonus(self, cfg):
        p = _paper(
            title="Logic + ML",
            abstract="Nothing.",
            categories=["cs.LO", "cs.LG"],
        )
        score = compute_local_score(p, cfg)
        # +2 cs.LO, +2 LG+LO combo, +1 category bridge = +5.
        assert score.score == 5

    def test_weak_hits_are_capped(self, cfg):
        # Five separate weak hits should only add +4 (cap).
        cfg2 = dict(cfg)
        cfg2["filters"] = dict(cfg["filters"])
        cfg2["filters"]["positive_keywords_weak"] = ["a", "b", "c", "d", "e"]
        p = _paper(title="a b c d e", abstract="")
        score = compute_local_score(p, cfg2)
        # Five weak hits → cap to 4; then no strong/medium → -4 penalty.
        # Total: 4 - 4 = 0.
        assert score.score == 0
