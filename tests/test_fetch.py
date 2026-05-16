"""Unit tests for the fetch helpers that are pure (no network)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from arxiv_digest.fetch import filter_new_papers, http_get_text, merge_papers
from arxiv_digest.models import Paper


def _paper(arxiv_id: str, source: str, **kwargs) -> Paper:
    defaults = dict(
        version=None,
        title="t",
        abstract="a",
        authors=[],
        categories=["cs.LG"],
        primary_category="cs.LG",
        link=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_link=None,
        published=datetime(2024, 1, 1, tzinfo=UTC),
        updated=datetime(2024, 1, 1, tzinfo=UTC),
        source=source,
    )
    defaults.update(kwargs)
    return Paper(arxiv_id=arxiv_id, **defaults)


class TestMergePapers:
    def test_api_overrides_rss_metadata(self):
        rss = _paper("2401.001", "rss", title="rss title", authors=["A"])
        api = _paper("2401.001", "api", title="api title", authors=["A", "B"])
        merged = merge_papers([rss], [api])
        assert merged["2401.001"].title == "api title"
        assert merged["2401.001"].authors == ["A", "B"]
        assert merged["2401.001"].source == "api+rss"

    def test_rss_only(self):
        rss = _paper("2401.002", "rss", title="rss only")
        merged = merge_papers([rss])
        assert merged["2401.002"].source == "rss"

    def test_api_only(self):
        api = _paper("2401.003", "api")
        merged = merge_papers([api])
        assert merged["2401.003"].source == "api"

    def test_dedup_by_id(self):
        rss1 = _paper("2401.004", "rss")
        rss2 = _paper("2401.004", "rss")
        merged = merge_papers([rss1, rss2])
        assert len(merged) == 1


class TestFilterNewPapers:
    def test_keeps_papers_after_resume_point(self):
        now = datetime(2024, 1, 10, tzinfo=UTC)
        old = _paper(
            "2401.100", "api",
            published=now - timedelta(days=5), updated=now - timedelta(days=5),
        )
        new = _paper(
            "2401.101", "api",
            published=now - timedelta(hours=1), updated=now - timedelta(hours=1),
        )
        merged = {p.arxiv_id: p for p in [old, new]}
        result = filter_new_papers(merged, since_dt=now - timedelta(days=2))
        ids = [p.arxiv_id for p in result]
        assert ids == ["2401.101"]


class TestHttpGetTextRetries:
    def test_retries_on_5xx_then_succeeds(self, monkeypatch):
        """Retries on a 503 and returns content on the second attempt."""
        from io import BytesIO
        from urllib.error import HTTPError

        attempts = {"n": 0}

        class FakeResp:
            def __init__(self, body: bytes):
                self._body = body
                self.headers = type("H", (), {"get_content_charset": lambda self: "utf-8"})()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return self._body

        def fake_urlopen(req, timeout=None):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise HTTPError(
                    url=req.full_url, code=503, msg="busy",
                    hdrs=None, fp=BytesIO(b""),
                )
            return FakeResp(b"ok body")

        sleeps: list[float] = []
        monkeypatch.setattr("arxiv_digest.fetch.urlopen", fake_urlopen)

        result = http_get_text(
            "https://example.com/x",
            user_agent="test",
            max_attempts=3,
            backoff_base_s=0.01,
            sleep=sleeps.append,
        )
        assert result == "ok body"
        assert attempts["n"] == 2
        assert len(sleeps) == 1

    def test_raises_after_exhausting_retries(self, monkeypatch):
        from io import BytesIO
        from urllib.error import HTTPError

        def fake_urlopen(req, timeout=None):
            raise HTTPError(
                url=req.full_url, code=503, msg="busy",
                hdrs=None, fp=BytesIO(b""),
            )

        monkeypatch.setattr("arxiv_digest.fetch.urlopen", fake_urlopen)
        with pytest.raises(HTTPError):
            http_get_text(
                "https://example.com/x",
                user_agent="test",
                max_attempts=2,
                backoff_base_s=0.01,
                sleep=lambda s: None,
            )
