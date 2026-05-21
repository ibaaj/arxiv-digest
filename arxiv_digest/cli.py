"""Command-line entry point.

Usage:

.. code-block:: shell

    python -m arxiv_digest --config config.toml
    python -m arxiv_digest --config config.toml --no-open
    python -m arxiv_digest --config config.toml --dry-run
    python -m arxiv_digest --config config.toml --since 2026-04-01T00:00:00Z

If the arXiv API window fetch fails with a rate limit (429) or a
transient server / network error, the run does not abort: it logs a
warning, continues with whatever the RSS feeds returned, and leaves the
resume point unchanged so the missed window is retried on the next run.
"""

from __future__ import annotations

import argparse
import logging
import sys
import webbrowser
from datetime import timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError

from .config import load_config
from .db import DB
from .fetch import (
    fetch_api_window,
    fetch_rss_latest,
    filter_new_papers,
    merge_papers,
)
from .llm import build_cost_meta, llm_rerank
from .render import build_html, decide, write_html
from .scoring import compute_local_score
from .utils import dt_from_iso, dt_to_iso, ensure_utc, utcnow

log = logging.getLogger("arxiv_digest")

DEFAULT_CONFIG_PATH = "config.toml"


def _parse_iso_arg(value: str):
    """Accept ``2026-04-01T00:00:00Z`` and similar ISO-8601 forms."""
    from datetime import datetime
    return ensure_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))


def _choose_resume_since(db: DB, cfg: dict, override_since: str | None):
    if override_since:
        return _parse_iso_arg(override_since)
    saved = db.get_state("last_successful_run_utc")
    if saved:
        return dt_from_iso(saved)
    lookback_days = int(cfg["arxiv"].get("initial_lookback_days", 3))
    return utcnow() - timedelta(days=lookback_days)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def _is_transient_http(exc: HTTPError) -> bool:
    """A 429 or any 5xx is transient: worth degrading rather than aborting."""
    return exc.code == 429 or 500 <= exc.code <= 599


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arxiv-digest",
        description="Build a daily arXiv digest filtered by your topic config.",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to TOML config file")
    parser.add_argument("--since", default=None, help="Override resume point, ISO-8601 UTC")
    parser.add_argument("--no-open", action="store_true", help="Do not open the HTML digest in a browser")
    parser.add_argument("--dry-run", action="store_true", help="Do not update the last successful run state")
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="Skip the arXiv API window fetch entirely and use RSS only",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    _setup_logging(args.verbose)

    cfg = load_config(Path(args.config))
    db = DB(Path(cfg["app"].get("state_db", "data/state.db")))
    run_id = db.begin_run()
    run_started = utcnow()

    try:
        categories = cfg["arxiv"]["categories"]
        user_agent = cfg["app"].get(
            "user_agent", "arxiv-digest/0.3 (set-your-contact-email@example.com)"
        )
        since_dt = _choose_resume_since(db, cfg, args.since)
        overlap_minutes = int(cfg["arxiv"].get("resume_overlap_minutes", 90))
        query_since = since_dt - timedelta(minutes=overlap_minutes)
        now_dt = utcnow()

        log.info("Resume since %s (overlap %d min)", dt_to_iso(since_dt), overlap_minutes)
        log.info("Categories: %s", ", ".join(categories))

        rss_papers = fetch_rss_latest(
            categories,
            user_agent=user_agent,
            pause_s=float(cfg["arxiv"].get("rss_pause_seconds", 1.0)),
        )
        log.info("RSS fetched %d entries", len(rss_papers))

        # The API window fetch is the part that most often hits arXiv's
        # rate limit. If it fails transiently (429 / 5xx / network), keep
        # the RSS results rather than discarding the whole run.
        api_papers: list = []
        api_degraded = False
        if args.no_api:
            log.info("Skipping API window fetch (--no-api); using RSS only.")
            api_degraded = True
        else:
            try:
                api_papers = fetch_api_window(
                    categories,
                    since_dt=query_since,
                    until_dt=now_dt,
                    user_agent=user_agent,
                    page_size=int(cfg["arxiv"].get("api_page_size", 100)),
                    pause_s=float(cfg["arxiv"].get("api_pause_seconds", 3.0)),
                )
                log.info("API fetched %d entries", len(api_papers))
            except HTTPError as exc:
                if not _is_transient_http(exc):
                    raise
                api_degraded = True
                log.warning(
                    "arXiv API rate-limited or unavailable (HTTP %s); continuing "
                    "with RSS results only. Resume point will not advance, so the "
                    "missed window is retried on the next run.",
                    exc.code,
                )
            except URLError as exc:
                api_degraded = True
                log.warning(
                    "arXiv API network error (%s); continuing with RSS results "
                    "only. Resume point will not advance.",
                    exc,
                )

        merged = merge_papers(rss_papers, api_papers)
        all_recent = filter_new_papers(merged, since_dt)
        log.info(
            "Merged %d unique; %d new after resume point", len(merged), len(all_recent)
        )

        local_scored = [(p, compute_local_score(p, cfg)) for p in all_recent]

        broad_threshold = int(cfg["filters"].get("broad_candidate_threshold", 0))
        llm_candidates = [
            (paper, local)
            for paper, local in sorted(local_scored, key=lambda x: x[1].score, reverse=True)
            if local.score >= broad_threshold
        ]
        log.info(
            "LLM candidates above broad threshold (%d): %d",
            broad_threshold, len(llm_candidates),
        )

        llm_cfg = cfg["llm"]
        llm_results: dict = {}
        llm_meta: dict = {
            "estimated": build_cost_meta(0, 0, llm_cfg),
            "actual": build_cost_meta(None, None, llm_cfg),
            "candidate_count": 0,
            "model": llm_cfg.get("model", ""),
        }
        if llm_cfg.get("enabled", False) and llm_candidates:
            llm_results, llm_meta = llm_rerank(llm_candidates, cfg)

        decisions = []
        for paper, local in sorted(local_scored, key=lambda x: x[1].score, reverse=True):
            dec = decide(paper, local, llm_results.get(paper.arxiv_id), cfg)
            decisions.append((paper, dec, local))
            db.upsert_paper(run_id, paper, dec)

        decisions.sort(
            key=lambda x: (
                x[1].decision != "keep",          # keeps first
                x[1].llm_score is None,           # reranked first
                -(x[1].llm_score or -1),          # higher llm first
                -x[1].local_score,                # tie-break
                -x[1].final_score,
            )
        )

        stats = {
            "fetched_total": len(merged),
            "new_candidates": len(all_recent),
            "llm_reranked": len(llm_results),
            "resume_since": dt_to_iso(since_dt),
            "llm_meta": llm_meta,
        }
        html = build_html(decisions, run_started, cfg, stats)
        stem = run_started.strftime("digest_%Y%m%d_%H%M%S")
        html_path = write_html(html, Path(cfg["output"].get("directory", "output")), stem)

        # Only advance the resume point on a complete run. A degraded run
        # (API skipped or rate-limited) must re-query the same window next
        # time, or papers that exist only in the API window would be lost.
        advanced = False
        if not args.dry_run and not api_degraded:
            db.set_state("last_successful_run_utc", dt_to_iso(now_dt))
            advanced = True

        status = "success" if not api_degraded else "partial"
        note_bits = [f"wrote {html_path}"]
        if api_degraded:
            note_bits.append("API degraded; resume point held")
        if not advanced and not api_degraded and args.dry_run:
            note_bits.append("dry-run; resume point held")
        db.finish_run(run_id, status, notes="; ".join(note_bits))

        log.info("Wrote digest: %s", html_path)
        if api_degraded:
            log.warning(
                "Run completed in degraded mode (RSS only). Re-run later to "
                "backfill the API window."
            )
        kept_count = sum(1 for _, d, _ in decisions if d.decision == "keep")
        log.info(
            "Fetched %d | New %d | Kept %d | LLM reranked %d",
            len(merged), len(all_recent), kept_count, len(llm_results),
        )
        est_in = llm_meta["estimated"].get("input_tokens", 0)
        est_out = llm_meta["estimated"].get("output_tokens", 0)
        est_cost = llm_meta["estimated"].get("cost_usd") or 0
        log.info(
            "LLM estimated tokens/cost: %d in / %d out / $%.4f",
            est_in, est_out, est_cost,
        )
        act_in = llm_meta["actual"].get("input_tokens")
        if act_in is not None:
            act_out = llm_meta["actual"].get("output_tokens") or 0
            act_cost = llm_meta["actual"].get("cost_usd") or 0
            log.info(
                "LLM actual tokens/cost:    %d in / %d out / $%.4f",
                act_in, act_out, act_cost,
            )

        if not args.no_open:
            webbrowser.open(html_path.resolve().as_uri())
        return 0
    except Exception as exc:
        db.finish_run(run_id, "failed", notes=str(exc))
        log.exception("Run failed: %s", exc)
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
