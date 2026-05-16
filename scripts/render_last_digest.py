#!/usr/bin/env python3
"""Re-render the HTML for the most recent successful run.

Useful when you change the HTML template or styling and want to see
the new layout without re-fetching from arXiv.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from arxiv_digest.config import load_config
from arxiv_digest.db import DB
from arxiv_digest.models import FinalDecision, LocalScore, Paper
from arxiv_digest.render import build_html, write_html
from arxiv_digest.utils import dt_from_iso, utcnow


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.toml", help="Path to TOML config file")
    args = parser.parse_args(argv)

    cfg = load_config(Path(args.config))
    db = DB(Path(cfg["app"].get("state_db", "data/state.db")))

    row = db.conn.execute(
        """
        SELECT run_id, started_at FROM runs
        WHERE status = 'success'
        ORDER BY run_id DESC LIMIT 1
        """
    ).fetchone()
    if row is None:
        print("No successful run found in the database.", file=sys.stderr)
        return 1

    run_id = int(row["run_id"])
    run_started = dt_from_iso(row["started_at"])

    rows = db.conn.execute(
        "SELECT * FROM papers WHERE last_run_id = ?", (run_id,)
    ).fetchall()

    decisions: list[tuple[Paper, FinalDecision, LocalScore]] = []
    for r in rows:
        paper = Paper(
            arxiv_id=r["arxiv_id"],
            version=r["version"],
            title=r["title"],
            abstract=r["abstract"],
            authors=json.loads(r["authors_json"] or "[]"),
            categories=json.loads(r["categories_json"] or "[]"),
            primary_category=r["primary_category"],
            link=r["link"],
            pdf_link=r["pdf_link"],
            published=dt_from_iso(r["published"]) if r["published"] else None,
            updated=dt_from_iso(r["updated"]) if r["updated"] else None,
            source=r["source"],
            announce_type=r["announce_type"],
        )
        local = LocalScore(
            score=int(r["local_score"] or 0),
            reasons=json.loads(r["local_reasons_json"] or "[]"),
        )
        dec = FinalDecision(
            arxiv_id=r["arxiv_id"],
            local_score=int(r["local_score"] or 0),
            llm_score=int(r["llm_score"]) if r["llm_score"] is not None else None,
            final_score=float(r["final_score"] or 0.0),
            decision=r["decision"] or "drop",
            why=r["why"] or "",
            tags=json.loads(r["tags_json"] or "[]"),
            local_reasons=json.loads(r["local_reasons_json"] or "[]"),
        )
        decisions.append((paper, dec, local))

    decisions.sort(
        key=lambda x: (
            x[1].decision != "keep",
            x[1].llm_score is None,
            -(x[1].llm_score if x[1].llm_score is not None else -1),
            -x[1].local_score,
            -x[1].final_score,
        )
    )

    stats = {
        "fetched_total": len(rows),
        "new_candidates": len(rows),
        "llm_reranked": sum(1 for _, dec, _ in decisions if dec.llm_score is not None),
        "resume_since": "render-only",
        "llm_meta": {
            "estimated": {"input_tokens": 0, "output_tokens": 0, "cost_usd": None},
            "actual": {"input_tokens": None, "output_tokens": None, "cost_usd": None},
        },
    }

    html = build_html(decisions, run_started, cfg, stats)
    stem = utcnow().strftime("digest_rendered_%Y%m%d_%H%M%S")
    out = write_html(html, Path(cfg["output"].get("directory", "output")), stem)
    print(f"Rendered layout only: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
