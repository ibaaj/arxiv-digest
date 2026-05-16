"""Optional batched LLM rerank for the top local candidates.

We keep the LLM step optional and batched (one request per run) so the
script remains useful and cheap by default. The actual interest profile
(strong keywords, preferred authors, etc.) is forwarded verbatim from
the config so the model can use it as context, but the *topic
description* itself lives in ``[llm].system_prompt``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from .config import ConfigError
from .models import LocalScore, Paper
from .scoring import compact_local_why
from .utils import estimate_tokens, extract_json

log = logging.getLogger(__name__)


def usage_value(obj: Any, key: str) -> int | None:
    """Read an integer field from a usage object whether it is dict-like or object-like."""
    if obj is None:
        return None
    value = obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_cost_meta(
    input_tokens: int | None,
    output_tokens: int | None,
    llm_cfg: dict[str, Any],
) -> dict[str, float | int | None]:
    """Build a small dict with tokens and a cost estimate."""
    input_price = float(llm_cfg.get("input_price_per_million", 2.50))
    output_price = float(llm_cfg.get("output_price_per_million", 15.00))
    cost: float | None = None
    if input_tokens is not None and output_tokens is not None:
        cost = (
            (input_tokens / 1_000_000.0) * input_price
            + (output_tokens / 1_000_000.0) * output_price
        )
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost,
    }


def _empty_meta(llm_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "estimated": build_cost_meta(0, 0, llm_cfg),
        "actual": build_cost_meta(None, None, llm_cfg),
        "candidate_count": 0,
        "model": llm_cfg.get("model", ""),
    }


# Forward-compatible: when [llm].api is "responses" we use Responses API,
# otherwise we fall back to chat completions. Default is responses.
def _call_openai(client, model: str, system_prompt: str, user_content: str, api: str):
    if api == "chat":
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
        )
    return client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )


def _extract_text(response, api: str) -> str:
    if api == "chat":
        return response.choices[0].message.content or ""
    raw = getattr(response, "output_text", None)
    return raw if raw else str(response)


def llm_rerank(
    papers: list[tuple[Paper, LocalScore]],
    cfg: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Send the top local candidates to the LLM in one batched request.

    Returns ``(results, meta)`` where ``results`` maps each ``arxiv_id``
    to a dict with ``score``, ``decision``, ``why``, ``tags`` from the
    model, and ``meta`` contains token usage and cost estimates.

    The LLM step is skipped (returning empty results) when
    ``[llm].enabled = false``.
    """
    llm_cfg = cfg["llm"]
    if not llm_cfg.get("enabled", False):
        return {}, _empty_meta(llm_cfg)

    api_key = os.environ.get(llm_cfg.get("api_key_env", "OPENAI_API_KEY"), "")
    if not api_key:
        raise ConfigError(
            f"Missing API key in env var {llm_cfg.get('api_key_env', 'OPENAI_API_KEY')}"
        )
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise ConfigError(
            "The 'openai' package is required when [llm].enabled = true. "
            "Install with: pip install 'openai>=1.0.0'"
        ) from exc

    model = llm_cfg.get("model", "")
    if not model:
        raise ConfigError("[llm].model must be set when [llm].enabled = true")

    max_candidates = int(llm_cfg.get("max_candidates", 100))
    papers = papers[:max_candidates]
    if not papers:
        return {}, _empty_meta(llm_cfg) | {"model": model}

    filters = cfg["filters"]
    system_prompt = llm_cfg["system_prompt"].strip()

    payload = {
        "interest_profile": {
            "preferred_authors": filters.get("preferred_authors", []),
            "positive_keywords_strong": filters.get("positive_keywords_strong", []),
            "positive_keywords_medium": filters.get("positive_keywords_medium", []),
            "negative_keywords": filters.get("negative_keywords", []),
        },
        "papers": [
            {
                "arxiv_id": p.arxiv_id,
                "title": p.title,
                "authors": p.authors,
                "categories": p.categories,
                "abstract": p.abstract,
                "local_score": s.score,
                "local_why": compact_local_why(s),
            }
            for p, s in papers
        ],
    }

    user_content = (
        "Rank these candidate papers. Return JSON only with shape "
        '{"items": [{"arxiv_id": str, "score": int, "decision": "keep"|"drop", '
        '"why": str, "tags": [str, ...]}]}.\n\n'
        + json.dumps(payload, ensure_ascii=False)
    )
    estimated_input_tokens = estimate_tokens(system_prompt) + estimate_tokens(user_content)
    estimated_output_tokens = (
        int(llm_cfg.get("estimated_output_tokens_per_paper", 40)) * len(papers)
    )
    meta: dict[str, Any] = {
        "estimated": build_cost_meta(estimated_input_tokens, estimated_output_tokens, llm_cfg),
        "actual": build_cost_meta(None, None, llm_cfg),
        "candidate_count": len(papers),
        "model": model,
    }

    api_choice = str(llm_cfg.get("api", "responses")).lower()
    if api_choice not in ("responses", "chat"):
        raise ConfigError("[llm].api must be 'responses' or 'chat'")

    log.info(
        "LLM rerank: model=%s api=%s candidates=%d est_in=%d est_out=%d",
        model, api_choice, len(papers), estimated_input_tokens, estimated_output_tokens,
    )

    client = OpenAI(api_key=api_key)
    response = _call_openai(client, model, system_prompt, user_content, api_choice)

    usage = getattr(response, "usage", None)
    actual_input_tokens = usage_value(usage, "input_tokens") or usage_value(usage, "prompt_tokens")
    actual_output_tokens = (
        usage_value(usage, "output_tokens") or usage_value(usage, "completion_tokens")
    )
    meta["actual"] = build_cost_meta(actual_input_tokens, actual_output_tokens, llm_cfg)

    raw = _extract_text(response, api_choice)
    try:
        parsed = json.loads(extract_json(raw))
    except (ValueError, json.JSONDecodeError) as exc:
        log.error("Failed to parse LLM JSON response: %s; raw=%r", exc, raw[:500])
        return {}, meta

    items = parsed.get("items", [])
    out: dict[str, dict[str, Any]] = {}
    for item in items:
        arxiv_id = str(item.get("arxiv_id", "")).strip()
        if not arxiv_id:
            continue
        # Drop the leading 'arXiv:' prefix the model sometimes adds.
        arxiv_id = re.sub(r"^arxiv:\s*", "", arxiv_id, flags=re.IGNORECASE)
        out[arxiv_id] = {
            "score": int(item.get("score", 0)),
            "decision": str(item.get("decision", "drop")).strip().lower(),
            "why": str(item.get("why", "")).strip(),
            "tags": [str(x).strip() for x in item.get("tags", []) if str(x).strip()],
        }
    return out, meta
