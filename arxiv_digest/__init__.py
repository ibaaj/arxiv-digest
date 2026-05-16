"""arxiv-digest: a configurable daily digest of new arXiv papers.

The package builds a local daily digest by:

1. Fetching new papers from arXiv RSS feeds and the arXiv API (with
   resume/backfill so missed days are recovered).
2. Storing persistent state in SQLite.
3. Scoring papers with a deterministic, configurable keyword + author
   filter.
4. Optionally re-ranking the top local candidates with a single batched
   OpenAI call.
5. Rendering a self-contained HTML report.

The package is topic-agnostic: all keywords, preferred authors, and the
LLM rerank instructions live in a TOML config file.
"""

from .config import ConfigError, load_config
from .models import FinalDecision, LocalScore, Paper

__all__ = ["Paper", "LocalScore", "FinalDecision", "load_config", "ConfigError"]
__version__ = "0.3.0"
