#!/usr/bin/env python3
"""Render a demo HTML digest from a small set of synthetic papers.

This script does not call the arXiv API or the LLM. It builds a
believable digest from a few hand-crafted fake-paper records so you
can preview the layout without exposing real topic of interest.

Usage:

.. code-block:: shell

    python scripts/demo_digest.py --config examples/config.example.toml
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from arxiv_digest.config import load_config
from arxiv_digest.render import build_html, decide, write_html
from arxiv_digest.scoring import compute_local_score
from arxiv_digest.models import Paper


SYNTHETIC_PAPERS = [
    Paper(
        arxiv_id="2601.00101",
        version="1",
        title="Sim-to-real transfer for robotic manipulation with limited demonstrations",
        abstract=(
            "We propose a sim-to-real training pipeline for robot learning of dexterous "
            "manipulation policies, combining domain randomization with a small amount of "
            "real-world imitation learning."
        ),
        authors=["A. Researcher", "B. Coauthor"],
        categories=["cs.RO", "cs.LG"],
        primary_category="cs.RO",
        link="https://arxiv.org/abs/2601.00101",
        pdf_link="https://arxiv.org/pdf/2601.00101.pdf",
        published=datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc),
        updated=datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc),
        source="api",
    ),
    Paper(
        arxiv_id="2601.00102",
        version="1",
        title="Legged locomotion via model-based reinforcement learning",
        abstract=(
            "We train a quadruped policy with model-based RL and trajectory optimization, "
            "evaluating sim-to-real transfer on rough terrain. Behavior cloning warm-starts "
            "the actor-critic loop."
        ),
        authors=["C. Researcher"],
        categories=["cs.RO", "cs.LG", "cs.AI"],
        primary_category="cs.RO",
        link="https://arxiv.org/abs/2601.00102",
        pdf_link="https://arxiv.org/pdf/2601.00102.pdf",
        published=datetime(2026, 5, 15, 13, 0, tzinfo=timezone.utc),
        updated=datetime(2026, 5, 15, 13, 0, tzinfo=timezone.utc),
        source="api",
    ),
    Paper(
        arxiv_id="2601.00103",
        version="1",
        title="Audio codec design for low-latency streaming",
        abstract=(
            "We present a new audio codec with low latency optimization, targeted at "
            "real-time speech recognition pipelines."
        ),
        authors=["D. Author"],
        categories=["cs.LG"],
        primary_category="cs.LG",
        link="https://arxiv.org/abs/2601.00103",
        pdf_link="https://arxiv.org/pdf/2601.00103.pdf",
        published=datetime(2026, 5, 15, 14, 0, tzinfo=timezone.utc),
        updated=datetime(2026, 5, 15, 14, 0, tzinfo=timezone.utc),
        source="api",
    ),
    Paper(
        arxiv_id="2601.00104",
        version="1",
        title="A survey on offline reinforcement learning",
        abstract=(
            "We survey offline reinforcement learning methods, including policy "
            "optimization and exploration without environment interaction."
        ),
        authors=["E. Author", "F. Author"],
        categories=["cs.LG", "cs.AI"],
        primary_category="cs.LG",
        link="https://arxiv.org/abs/2601.00104",
        pdf_link="https://arxiv.org/pdf/2601.00104.pdf",
        published=datetime(2026, 5, 15, 15, 0, tzinfo=timezone.utc),
        updated=datetime(2026, 5, 15, 15, 0, tzinfo=timezone.utc),
        source="api",
    ),
    Paper(
        arxiv_id="2601.00105",
        version="1",
        title="Visuomotor policy distillation from large pretrained models",
        abstract=(
            "We distill visuomotor policies for real-world robot learning by combining "
            "behavior cloning with reward shaping, achieving robust dexterous manipulation."
        ),
        authors=["G. Author"],
        categories=["cs.RO", "cs.LG"],
        primary_category="cs.RO",
        link="https://arxiv.org/abs/2601.00105",
        pdf_link="https://arxiv.org/pdf/2601.00105.pdf",
        published=datetime(2026, 5, 15, 16, 0, tzinfo=timezone.utc),
        updated=datetime(2026, 5, 15, 16, 0, tzinfo=timezone.utc),
        source="api",
    ),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="examples/config.example.toml",
        help="Path to TOML config file",
    )
    parser.add_argument(
        "--out",
        default="output/demo.html",
        help="Where to write the demo digest",
    )
    args = parser.parse_args(argv)

    cfg = load_config(Path(args.config))
    decisions = []
    for paper in SYNTHETIC_PAPERS:
        local = compute_local_score(paper, cfg)
        # No LLM in the demo: pass None as the llm_result.
        dec = decide(paper, local, None, cfg)
        decisions.append((paper, dec, local))

    decisions.sort(
        key=lambda x: (
            x[1].decision != "keep",
            -x[1].local_score,
            -x[1].final_score,
        )
    )

    stats = {
        "fetched_total": len(SYNTHETIC_PAPERS),
        "new_candidates": len(SYNTHETIC_PAPERS),
        "llm_reranked": 0,
        "resume_since": "demo (no fetch)",
        "llm_meta": {
            "estimated": {"input_tokens": 0, "output_tokens": 0, "cost_usd": None},
            "actual": {"input_tokens": None, "output_tokens": None, "cost_usd": None},
        },
    }

    html = build_html(decisions, datetime.now(timezone.utc), cfg, stats)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    # Also use write_html to keep a "latest.html" alongside it, useful
    # for screenshotting from a stable filename.
    write_html(html, out_path.parent, out_path.stem)
    print(f"Wrote demo digest: {out_path}")
    kept = sum(1 for _, d, _ in decisions if d.decision == "keep")
    print(f"  kept: {kept}/{len(decisions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
