"""Unit tests for pure helpers in ``arxiv_digest.utils``."""

from __future__ import annotations

from datetime import UTC, datetime, timezone

import pytest

from arxiv_digest.utils import (
    clean_abstract,
    dt_from_iso,
    dt_to_iso,
    ensure_utc,
    estimate_tokens,
    extract_json,
    normalize_text,
    parse_arxiv_id_from_abs_url,
    unique_preserve_order,
)


class TestNormalizeText:
    def test_lowercases(self):
        assert normalize_text("Hello WORLD") == "hello world"

    def test_replaces_hyphens_and_underscores(self):
        assert normalize_text("sim-to-real") == "sim to real"
        assert normalize_text("foo_bar") == "foo bar"

    def test_collapses_whitespace(self):
        assert normalize_text("a  b\tc\nd") == "a b c d"

    def test_strips_punctuation(self):
        assert normalize_text("foo, bar; baz.") == "foo bar baz."

    def test_empty_string(self):
        assert normalize_text("") == ""


class TestUniquePreserveOrder:
    def test_keeps_first_occurrence(self):
        assert unique_preserve_order(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]

    def test_empty(self):
        assert unique_preserve_order([]) == []


class TestEstimateTokens:
    def test_empty(self):
        assert estimate_tokens("") == 0

    def test_proportional_to_length(self):
        short = estimate_tokens("hello world")
        long = estimate_tokens("hello world " * 100)
        assert long > short * 50  # roughly proportional


class TestParseArxivId:
    def test_versioned(self):
        arxiv_id, version = parse_arxiv_id_from_abs_url(
            "https://arxiv.org/abs/2401.12345v2"
        )
        assert arxiv_id == "2401.12345"
        assert version == "2"

    def test_unversioned(self):
        arxiv_id, version = parse_arxiv_id_from_abs_url(
            "https://arxiv.org/abs/2401.12345"
        )
        assert arxiv_id == "2401.12345"
        assert version is None

    def test_invalid_url_raises(self):
        with pytest.raises(ValueError):
            parse_arxiv_id_from_abs_url("https://example.com/no-abs")


class TestExtractJson:
    def test_bare_json_object(self):
        s = '{"foo": 1}'
        assert extract_json(s) == s

    def test_fenced_json(self):
        s = '```json\n{"foo": 1}\n```'
        assert extract_json(s) == '{"foo": 1}'

    def test_fenced_no_lang(self):
        s = '```\n{"foo": 1}\n```'
        assert extract_json(s) == '{"foo": 1}'

    def test_preamble_then_object(self):
        s = 'Here you go:\n{"foo": 1}'
        assert extract_json(s) == '{"foo": 1}'

    def test_no_json_raises(self):
        with pytest.raises(ValueError):
            extract_json("nothing useful here")


class TestDatetimeRoundtrip:
    def test_iso_roundtrip(self):
        dt = datetime(2024, 1, 15, 12, 30, 0, tzinfo=UTC)
        assert dt_from_iso(dt_to_iso(dt)) == dt

    def test_ensure_utc_naive(self):
        naive = datetime(2024, 1, 15, 12, 30, 0)
        assert ensure_utc(naive).tzinfo == UTC

    def test_ensure_utc_aware(self):
        from datetime import timedelta
        aware = datetime(2024, 1, 15, 12, 30, 0, tzinfo=timezone(timedelta(hours=2)))
        result = ensure_utc(aware)
        assert result.tzinfo == UTC
        # 12:30 +02:00 == 10:30 UTC
        assert result.hour == 10
        assert result.minute == 30


class TestCleanAbstract:
    def test_collapses_whitespace(self):
        assert clean_abstract("hello   world\n\n") == "hello world"
