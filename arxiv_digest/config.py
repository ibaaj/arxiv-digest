"""Config loading and validation.

The config is a single TOML file. We validate the high-level structure
and the few fields the script genuinely needs, so a typo in the config
gives a clear error rather than a ``KeyError`` deep inside the run.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


class ConfigError(RuntimeError):
    """Raised when the config is missing, malformed, or inconsistent."""


_REQUIRED_SECTIONS = ("app", "arxiv", "filters", "llm", "output")

_NUMERIC_FILTER_FIELDS = (
    "broad_candidate_threshold",
    "local_keep_threshold",
    "final_keep_threshold",
    "local_weight",
    "llm_weight",
)


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate a config TOML file.

    Returns the parsed dict on success and raises :class:`ConfigError`
    with a human-readable message on any structural problem.
    """
    if not path.exists():
        raise ConfigError(f"Config not found: {path}")
    try:
        with path.open("rb") as fh:
            cfg = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {path}: {exc}") from exc

    for section in _REQUIRED_SECTIONS:
        if section not in cfg:
            raise ConfigError(f"Missing [{section}] section in {path}")

    # Light type checks on the few fields we actually depend on.
    arxiv = cfg["arxiv"]
    if not isinstance(arxiv.get("categories", []), list) or not arxiv.get("categories"):
        raise ConfigError("[arxiv].categories must be a non-empty list of category strings")

    filters = cfg["filters"]
    for field_name in _NUMERIC_FILTER_FIELDS:
        if field_name in filters and not isinstance(filters[field_name], (int, float)):
            raise ConfigError(
                f"[filters].{field_name} must be a number, got {type(filters[field_name]).__name__}"
            )

    for list_field in (
        "preferred_authors",
        "blocked_authors_exact",
        "positive_keywords_strong",
        "positive_keywords_medium",
        "positive_keywords_weak",
        "negative_keywords",
    ):
        if list_field in filters and not isinstance(filters[list_field], list):
            raise ConfigError(f"[filters].{list_field} must be a list of strings")

    llm = cfg["llm"]
    if llm.get("enabled") and not llm.get("system_prompt"):
        raise ConfigError(
            "[llm].system_prompt is required when [llm].enabled = true. "
            "It describes the topic and selection criteria to the reranker."
        )

    return cfg
