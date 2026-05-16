"""SQLite-backed persistence for runs and papers.

The database keeps:

* ``kv_state``: small key/value table; we use it for the timestamp of
  the last successful run so the next run can resume from there.
* ``runs``: one row per execution, with status and timestamps.
* ``papers``: one row per arXiv id, upserted on every run.

Schema migrations are not handled here; if you change the schema, bump
``schema_version`` in ``kv_state`` and migrate explicitly.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .models import FinalDecision, Paper
from .utils import dt_to_iso, utcnow

_SCHEMA = """
CREATE TABLE IF NOT EXISTS kv_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS papers (
    arxiv_id TEXT PRIMARY KEY,
    version TEXT,
    title TEXT NOT NULL,
    abstract TEXT NOT NULL,
    authors_json TEXT NOT NULL,
    categories_json TEXT NOT NULL,
    primary_category TEXT,
    link TEXT NOT NULL,
    pdf_link TEXT,
    published TEXT,
    updated TEXT,
    first_seen_at TEXT NOT NULL,
    last_run_id INTEGER,
    source TEXT NOT NULL,
    announce_type TEXT,
    local_score INTEGER,
    llm_score INTEGER,
    final_score REAL,
    decision TEXT,
    why TEXT,
    tags_json TEXT,
    local_reasons_json TEXT,
    FOREIGN KEY(last_run_id) REFERENCES runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_papers_last_run_id ON papers(last_run_id);
CREATE INDEX IF NOT EXISTS idx_papers_decision ON papers(decision);
"""


class DB:
    """Thin wrapper around a SQLite connection."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        self.conn.close()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(_SCHEMA)
        self.conn.commit()

    # ---- runs --------------------------------------------------------

    def begin_run(self) -> int:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO runs (started_at, status, notes) VALUES (?, ?, ?)",
            (dt_to_iso(utcnow()), "running", ""),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def finish_run(self, run_id: int, status: str, notes: str = "") -> None:
        self.conn.execute(
            "UPDATE runs SET finished_at = ?, status = ?, notes = ? WHERE run_id = ?",
            (dt_to_iso(utcnow()), status, notes, run_id),
        )
        self.conn.commit()

    # ---- key/value state --------------------------------------------

    def get_state(self, key: str) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM kv_state WHERE key = ?", (key,)
        ).fetchone()
        return None if row is None else str(row[0])

    def set_state(self, key: str, value: str) -> None:
        self.conn.execute(
            """
            INSERT INTO kv_state(key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )
        self.conn.commit()

    # ---- papers ------------------------------------------------------

    def upsert_paper(
        self, run_id: int, paper: Paper, decision: FinalDecision
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO papers (
                arxiv_id, version, title, abstract, authors_json, categories_json,
                primary_category, link, pdf_link, published, updated, first_seen_at,
                last_run_id, source, announce_type, local_score, llm_score, final_score,
                decision, why, tags_json, local_reasons_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(arxiv_id) DO UPDATE SET
                version = excluded.version,
                title = excluded.title,
                abstract = excluded.abstract,
                authors_json = excluded.authors_json,
                categories_json = excluded.categories_json,
                primary_category = excluded.primary_category,
                link = excluded.link,
                pdf_link = excluded.pdf_link,
                published = excluded.published,
                updated = excluded.updated,
                last_run_id = excluded.last_run_id,
                source = excluded.source,
                announce_type = excluded.announce_type,
                local_score = excluded.local_score,
                llm_score = excluded.llm_score,
                final_score = excluded.final_score,
                decision = excluded.decision,
                why = excluded.why,
                tags_json = excluded.tags_json,
                local_reasons_json = excluded.local_reasons_json
            """,
            (
                paper.arxiv_id,
                paper.version,
                paper.title,
                paper.abstract,
                json.dumps(paper.authors, ensure_ascii=False),
                json.dumps(paper.categories, ensure_ascii=False),
                paper.primary_category,
                paper.link,
                paper.pdf_link,
                dt_to_iso(paper.published) if paper.published else None,
                dt_to_iso(paper.updated) if paper.updated else None,
                dt_to_iso(utcnow()),
                run_id,
                paper.source,
                paper.announce_type,
                decision.local_score,
                decision.llm_score,
                decision.final_score,
                decision.decision,
                decision.why,
                json.dumps(decision.tags, ensure_ascii=False),
                json.dumps(decision.local_reasons, ensure_ascii=False),
            ),
        )
        self.conn.commit()
