"""Unit tests for config loading and validation."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from arxiv_digest.config import ConfigError, load_config


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_loads_minimal_valid_config(tmp_path):
    path = _write(
        tmp_path,
        """
        [app]
        title = "x"

        [arxiv]
        categories = ["cs.AI"]

        [filters]

        [llm]
        enabled = false

        [output]
        directory = "output"
        """,
    )
    cfg = load_config(path)
    assert cfg["app"]["title"] == "x"
    assert cfg["arxiv"]["categories"] == ["cs.AI"]


def test_missing_section_raises(tmp_path):
    path = _write(
        tmp_path,
        """
        [app]
        title = "x"
        """,
    )
    with pytest.raises(ConfigError, match=r"Missing \[arxiv\]"):
        load_config(path)


def test_invalid_toml_raises(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text("not valid toml = =", encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid TOML"):
        load_config(path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="Config not found"):
        load_config(tmp_path / "nope.toml")


def test_empty_categories_raises(tmp_path):
    path = _write(
        tmp_path,
        """
        [app]
        title = "x"

        [arxiv]
        categories = []

        [filters]

        [llm]
        enabled = false

        [output]
        directory = "output"
        """,
    )
    with pytest.raises(ConfigError, match="categories"):
        load_config(path)


def test_llm_enabled_requires_system_prompt(tmp_path):
    path = _write(
        tmp_path,
        """
        [app]
        title = "x"

        [arxiv]
        categories = ["cs.AI"]

        [filters]

        [llm]
        enabled = true
        model = "gpt-4o-mini"

        [output]
        directory = "output"
        """,
    )
    with pytest.raises(ConfigError, match="system_prompt"):
        load_config(path)


def test_wrong_field_type_raises(tmp_path):
    path = _write(
        tmp_path,
        """
        [app]
        title = "x"

        [arxiv]
        categories = ["cs.AI"]

        [filters]
        positive_keywords_strong = "should be a list"

        [llm]
        enabled = false

        [output]
        directory = "output"
        """,
    )
    with pytest.raises(ConfigError, match="must be a list"):
        load_config(path)
