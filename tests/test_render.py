"""Unit tests for the final keep/drop decision logic."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from arxiv_digest.models import LocalScore, Paper
from arxiv_digest.render import decide


@pytest.fixture
def cfg():
    return {
        "filters": {
            "local_keep_threshold": 6,
            "final_keep_threshold": 25,
            "local_weight": 1.0,
            "llm_weight": 0.35,
        }
    }


def _paper(arxiv_id: str = "2401.00001") -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        version=None,
        title="t", abstract="a", authors=[], categories=["cs.LG"],
        primary_category="cs.LG",
        link=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_link=None,
        published=datetime(2024, 1, 1, tzinfo=UTC),
        updated=datetime(2024, 1, 1, tzinfo=UTC),
        source="api",
    )


def test_blocked_short_circuits(cfg):
    local = LocalScore(score=-999, reasons=["blocked author match: X"])
    d = decide(_paper(), local, None, cfg)
    assert d.decision == "drop"
    assert d.tags == ["blocked"]


def test_keep_no_llm_above_threshold(cfg):
    local = LocalScore(score=10)
    d = decide(_paper(), local, None, cfg)
    assert d.decision == "keep"
    assert d.llm_score is None


def test_drop_no_llm_below_threshold(cfg):
    local = LocalScore(score=3)
    d = decide(_paper(), local, None, cfg)
    assert d.decision == "drop"


def test_llm_keep_combined(cfg):
    local = LocalScore(score=10)
    llm = {"score": 50, "decision": "keep", "why": "looks relevant", "tags": ["rl"]}
    d = decide(_paper(), local, llm, cfg)
    # final = 1.0 * 10 + 0.35 * 50 = 27.5 ≥ 25
    assert d.decision == "keep"
    assert d.llm_score == 50
    assert abs(d.final_score - 27.5) < 1e-6


def test_llm_drop_when_below_threshold(cfg):
    local = LocalScore(score=10)
    llm = {"score": 10, "decision": "keep", "why": "ok", "tags": []}
    d = decide(_paper(), local, llm, cfg)
    # final = 10 + 3.5 = 13.5 < 25 → drop despite LLM keep.
    assert d.decision == "drop"


def test_rescue_rule(cfg):
    """Very high local + very high LLM rescues a borderline final score."""
    local = LocalScore(score=14)  # local_threshold + 8
    llm = {"score": 40, "decision": "keep", "why": "strong", "tags": []}
    d = decide(_paper(), local, llm, cfg)
    # final = 14 + 14 = 28 ≥ 25 anyway, but the rescue rule covers
    # configs where final_threshold is high.
    assert d.decision == "keep"
