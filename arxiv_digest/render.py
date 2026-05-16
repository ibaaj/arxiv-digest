"""Final keep/drop decision and HTML rendering.

The HTML is intentionally minimal and self-contained: one file, inline
CSS, no JavaScript. This makes it easy to share, email, or open
offline.
"""

from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

from .models import FinalDecision, LocalScore, Paper
from .scoring import compact_local_why
from .utils import dt_to_iso


def decide(
    paper: Paper,
    local: LocalScore,
    llm_result: dict[str, Any] | None,
    cfg: dict[str, Any],
) -> FinalDecision:
    """Combine local and LLM scores into a keep/drop decision."""
    filters = cfg["filters"]

    if local.score <= -999:
        return FinalDecision(
            arxiv_id=paper.arxiv_id,
            local_score=local.score,
            llm_score=None,
            final_score=float(local.score),
            decision="drop",
            why=local.reasons[0] if local.reasons else "blocked author",
            tags=["blocked"],
            local_reasons=local.reasons,
        )

    local_threshold = int(filters.get("local_keep_threshold", 6))
    final_threshold = float(filters.get("final_keep_threshold", 25))
    local_weight = float(filters.get("local_weight", 1.0))
    llm_weight = float(filters.get("llm_weight", 0.35))

    if llm_result:
        llm_score = int(llm_result.get("score", 0))
        final_score = local_weight * local.score + llm_weight * llm_score
        decision = llm_result.get("decision", "drop")
        why = llm_result.get("why", "LLM reranker") or compact_local_why(local)
        tags = llm_result.get("tags", [])
        keep = decision == "keep" and final_score >= final_threshold
        # Safety rescue: a paper with very high local AND llm score should
        # still be kept even if the final threshold marginally fails.
        if not keep and local.score >= local_threshold + 8 and llm_score >= 40:
            keep = True
        decision = "keep" if keep else "drop"
    else:
        llm_score = None
        final_score = float(local.score)
        decision = "keep" if local.score >= local_threshold else "drop"
        why = compact_local_why(local)
        tags = []

    return FinalDecision(
        arxiv_id=paper.arxiv_id,
        local_score=local.score,
        llm_score=llm_score,
        final_score=final_score,
        decision=decision,
        why=why,
        tags=tags,
        local_reasons=local.reasons,
    )


def _fmt_cost(value: Any) -> str:
    return "n/a" if value is None else f"${float(value):.4f}"


def _fmt_tokens(value: Any) -> str:
    """Render a token count, falling back to 'n/a' for missing or None values."""
    return "n/a" if value is None else str(value)


def _paper_card(paper: Paper, dec: FinalDecision, local: LocalScore) -> str:
    authors = escape(", ".join(paper.authors) if paper.authors else "Unknown authors")
    categories = escape(", ".join(paper.categories))
    why = escape(dec.why or compact_local_why(local))
    tags = " ".join(f"<span class='tag'>{escape(tag)}</span>" for tag in dec.tags[:4])
    local_reason = escape("; ".join(local.reasons[:4]))
    score_bits = [f"local {dec.local_score}"]
    if dec.llm_score is not None:
        score_bits.append(f"llm {dec.llm_score}")
    score_bits.append(f"final {dec.final_score:.1f}")
    scores = " · ".join(score_bits)
    return f"""
    <article class="card">
      <h2><a href="{escape(paper.link)}" target="_blank" rel="noopener noreferrer">{escape(paper.title)}</a></h2>
      <div class="meta"><strong>Authors:</strong> {authors}</div>
      <div class="meta"><strong>Categories:</strong> {categories}</div>
      <div class="meta"><strong>Why:</strong> <span class="why">{why}</span></div>
      <div class="meta"><strong>Scores:</strong> {escape(scores)}</div>
      <div class="meta"><strong>Local rationale:</strong> {local_reason}</div>
      <div class="tags">{tags}</div>
      <details>
        <summary>Abstract</summary>
        <p>{escape(paper.abstract)}</p>
      </details>
    </article>
    """


_STYLE = """
:root {
  --bg: #fbfbfd;
  --card: #ffffff;
  --fg: #1b1f24;
  --muted: #586069;
  --border: #d0d7de;
  --accent: #0b5fff;
  --tag: #eef4ff;
}
body {
  margin: 0;
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--fg);
  background: var(--bg);
  line-height: 1.45;
}
main { max-width: 980px; margin: 0 auto; padding: 24px; }
h1 { margin-bottom: 0.2rem; }
.subtle { color: var(--muted); }
.summary {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
  margin: 18px 0 24px;
}
.summary .box, .card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 14px 16px;
  box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}
.card { margin-bottom: 16px; }
.card h2 { font-size: 1.05rem; margin: 0 0 0.6rem; }
.card h2 a { color: var(--accent); text-decoration: none; }
.card h2 a:hover { text-decoration: underline; }
.meta { margin: 0.25rem 0; }
.tags { margin: 0.5rem 0; }
.tag {
  display: inline-block;
  background: var(--tag);
  border: 1px solid #c9ddff;
  padding: 2px 8px;
  border-radius: 999px;
  margin: 0 6px 6px 0;
  font-size: 0.82rem;
}
details { margin-top: 0.6rem; }
summary { cursor: pointer; font-weight: 600; }
.section-title { margin-top: 28px; }
footer { margin-top: 28px; color: var(--muted); font-size: 0.94rem; }
code { background: #f6f8fa; padding: 0.1rem 0.3rem; border-radius: 6px; }
"""


def build_html(
    decisions: list[tuple[Paper, FinalDecision, LocalScore]],
    run_started: datetime,
    cfg: dict[str, Any],
    stats: dict[str, Any],
) -> str:
    """Render the digest to a single HTML string."""
    title = cfg["app"].get("title", "arXiv digest")
    kept = [x for x in decisions if x[1].decision == "keep"]
    dropped = [x for x in decisions if x[1].decision != "keep"]
    llm_est = stats.get("llm_meta", {}).get("estimated", {})
    llm_actual = stats.get("llm_meta", {}).get("actual", {})

    cards_kept = "\n".join(_paper_card(p, d, loc) for p, d, loc in kept)
    cards_dropped = "\n".join(
        _paper_card(p, d, loc) for p, d, loc in dropped[: min(len(dropped), 25)]
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>{_STYLE}</style>
</head>
<body>
  <main>
    <h1>{escape(title)}</h1>
    <div class="subtle">Generated at {escape(dt_to_iso(run_started))}</div>

    <section class="summary">
      <div class="box"><strong>Fetched</strong><br>{stats['fetched_total']}</div>
      <div class="box"><strong>New after resume point</strong><br>{stats['new_candidates']}</div>
      <div class="box"><strong>Kept</strong><br>{len(kept)}</div>
      <div class="box"><strong>Dropped</strong><br>{len(dropped)}</div>
      <div class="box"><strong>LLM reranked</strong><br>{stats['llm_reranked']}</div>
      <div class="box"><strong>Resume since</strong><br>{escape(stats['resume_since'])}</div>
      <div class="box"><strong>Est. input/output tokens</strong><br>{_fmt_tokens(llm_est.get('input_tokens'))} / {_fmt_tokens(llm_est.get('output_tokens'))}</div>
      <div class="box"><strong>Est. LLM cost</strong><br>{_fmt_cost(llm_est.get('cost_usd'))}</div>
      <div class="box"><strong>Actual input/output tokens</strong><br>{_fmt_tokens(llm_actual.get('input_tokens'))} / {_fmt_tokens(llm_actual.get('output_tokens'))}</div>
      <div class="box"><strong>Actual LLM cost</strong><br>{_fmt_cost(llm_actual.get('cost_usd'))}</div>
    </section>

    <h2 class="section-title">Read first</h2>
    {cards_kept if cards_kept else '<p>No papers passed the current threshold.</p>'}

    <h2 class="section-title">Dropped / borderline (first 25)</h2>
    {cards_dropped if cards_dropped else '<p>No dropped papers recorded in this run.</p>'}

    <footer>
      <p>Thank you to arXiv for use of its open access interoperability.</p>
      <p>This digest uses arXiv RSS/ATOM feeds and the arXiv API. The local filter is deterministic; the LLM reranker is optional and batched.</p>
      <p>The LLM reranker reports estimated cost from configured token prices; actual usage is shown when returned by the API.</p>
    </footer>
  </main>
</body>
</html>
"""


def write_html(html: str, out_dir: Path, stem: str) -> Path:
    """Write the HTML to ``out_dir/{stem}.html`` and update ``latest.html``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.html"
    path.write_text(html, encoding="utf-8")
    latest = out_dir / "latest.html"
    latest.write_text(html, encoding="utf-8")
    return path
