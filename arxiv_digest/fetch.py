"""Fetching from arXiv RSS feeds and the arXiv API.

We use the standard library only (``urllib``) so the package has no
hard dependency beyond the optional OpenAI SDK. HTTP failures are
retried with exponential backoff because the arXiv API occasionally
returns 5xx under load.
"""

from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .models import Paper
from .utils import (
    clean_abstract,
    ensure_utc,
    parse_arxiv_id_from_abs_url,
    unique_preserve_order,
)

log = logging.getLogger(__name__)

ATOM_NS = "http://www.w3.org/2005/Atom"
ARXIV_NS = "http://arxiv.org/schemas/atom"
DC_NS = "http://purl.org/dc/elements/1.1/"
OPENSEARCH_NS = "http://a9.com/-/spec/opensearch/1.1/"
NS = {
    "atom": ATOM_NS,
    "arxiv": ARXIV_NS,
    "dc": DC_NS,
    "opensearch": OPENSEARCH_NS,
}

# Retry transient failures. arXiv asks clients to back off on errors.
_RETRY_STATUS = {429, 500, 502, 503, 504}


def http_get_text(
    url: str,
    user_agent: str,
    timeout_s: int = 30,
    max_attempts: int = 4,
    backoff_base_s: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """GET a URL as text with retry on transient failures.

    Retries on ``URLError`` (DNS/network) and HTTP status codes in
    ``_RETRY_STATUS``. Delay between attempts is ``backoff_base_s * 2**n``.
    """
    req = Request(url, headers={"User-Agent": user_agent})
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            with urlopen(req, timeout=timeout_s) as resp:
                charset = resp.headers.get_content_charset() or "utf-8"
                return resp.read().decode(charset, errors="replace")
        except HTTPError as exc:
            last_exc = exc
            if exc.code in _RETRY_STATUS and attempt < max_attempts - 1:
                delay = backoff_base_s * (2 ** attempt)
                log.warning(
                    "HTTP %s on %s (attempt %d/%d); retrying in %.1fs",
                    exc.code, url, attempt + 1, max_attempts, delay,
                )
                sleep(delay)
                continue
            raise
        except URLError as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                delay = backoff_base_s * (2 ** attempt)
                log.warning(
                    "Network error on %s (attempt %d/%d): %s; retrying in %.1fs",
                    url, attempt + 1, max_attempts, exc, delay,
                )
                sleep(delay)
                continue
            raise
    # Should be unreachable, but keep mypy happy.
    raise RuntimeError(f"Exhausted retries fetching {url}") from last_exc


def _parse_rss_description(description: str) -> tuple[str | None, str]:
    import re
    text = re.sub(r"<[^>]+>", "", description or "")
    announce_type: str | None = None
    abstract = text
    m = re.search(r"Announce Type:\s*([^\n]+)", text, flags=re.IGNORECASE)
    if m:
        announce_type = m.group(1).strip()
    m2 = re.search(r"Abstract:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if m2:
        abstract = m2.group(1).strip()
    return announce_type, clean_abstract(abstract)


def parse_rss_feed(xml_text: str, category: str) -> list[Paper]:
    """Parse the arXiv RSS XML for one category into ``Paper`` objects."""
    root = ET.fromstring(xml_text)
    out: list[Paper] = []
    for item in root.findall("./channel/item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        if not title or not link:
            continue
        try:
            arxiv_id, version = parse_arxiv_id_from_abs_url(link)
        except ValueError:
            continue
        description = item.findtext("description") or ""
        announce_type, abstract = _parse_rss_description(description)
        creators = [
            c.text.strip()
            for c in item.findall(f"{{{DC_NS}}}creator")
            if c.text
        ]
        if creators:
            if len(creators) == 1 and "," in creators[0]:
                authors = [a.strip() for a in creators[0].split(",") if a.strip()]
            else:
                authors = creators
        else:
            authors = []
        categories = [c.text.strip() for c in item.findall("category") if c.text]
        pub_text = item.findtext("pubDate")
        published = (
            parsedate_to_datetime(pub_text).astimezone(UTC)
            if pub_text
            else None
        )
        out.append(
            Paper(
                arxiv_id=arxiv_id,
                version=version,
                title=title,
                abstract=abstract,
                authors=authors,
                categories=unique_preserve_order(categories or [category]),
                primary_category=category,
                link=link,
                pdf_link=link.replace("/abs/", "/pdf/") + ".pdf",
                published=published,
                updated=published,
                source="rss",
                announce_type=announce_type,
            )
        )
    return out


def parse_atom_api(xml_text: str) -> list[Paper]:
    """Parse the Atom XML returned by the arXiv API search endpoint."""
    root = ET.fromstring(xml_text)
    out: list[Paper] = []
    for entry in root.findall("atom:entry", NS):
        title = clean_abstract(entry.findtext("atom:title", default="", namespaces=NS))
        link = entry.findtext("atom:id", default="", namespaces=NS).strip()
        if not title or not link:
            continue
        arxiv_id, version = parse_arxiv_id_from_abs_url(link)
        authors: list[str] = []
        for author in entry.findall("atom:author", NS):
            name = author.findtext("atom:name", default="", namespaces=NS).strip()
            if name:
                authors.append(name)
        categories = [c.attrib.get("term", "").strip() for c in entry.findall("atom:category", NS)]
        categories = [c for c in categories if c]
        primary_category_el = entry.find("arxiv:primary_category", NS)
        primary_category = (
            primary_category_el.attrib.get("term") if primary_category_el is not None else None
        )
        pdf_link: str | None = None
        for link_el in entry.findall("atom:link", NS):
            if link_el.attrib.get("title") == "pdf":
                pdf_link = link_el.attrib.get("href")
                break
        published_text = entry.findtext("atom:published", default="", namespaces=NS).strip()
        updated_text = entry.findtext("atom:updated", default="", namespaces=NS).strip()
        published = (
            datetime.fromisoformat(published_text).astimezone(UTC)
            if published_text
            else None
        )
        updated = (
            datetime.fromisoformat(updated_text).astimezone(UTC)
            if updated_text
            else None
        )
        out.append(
            Paper(
                arxiv_id=arxiv_id,
                version=version,
                title=title,
                abstract=clean_abstract(
                    entry.findtext("atom:summary", default="", namespaces=NS)
                ),
                authors=authors,
                categories=categories,
                primary_category=primary_category,
                link=link,
                pdf_link=pdf_link,
                published=published,
                updated=updated,
                source="api",
            )
        )
    return out


def fetch_rss_latest(
    categories: list[str], user_agent: str, pause_s: float = 1.0
) -> list[Paper]:
    """Fetch the current arXiv RSS feed for each requested category."""
    out: list[Paper] = []
    for i, category in enumerate(categories):
        if i:
            time.sleep(pause_s)
        url = f"https://rss.arxiv.org/rss/{category}"
        log.debug("fetching RSS: %s", url)
        xml_text = http_get_text(url, user_agent)
        out.extend(parse_rss_feed(xml_text, category))
    return out


def _format_arxiv_submitted_date(dt: datetime) -> str:
    return ensure_utc(dt).strftime("%Y%m%d%H%M")


def fetch_api_window(
    categories: list[str],
    since_dt: datetime,
    until_dt: datetime,
    user_agent: str,
    page_size: int = 100,
    pause_s: float = 3.0,
) -> list[Paper]:
    """Fetch all papers in a category submitted between two timestamps.

    Pages through the arXiv API until a short batch indicates the end.
    A pause between requests respects arXiv's usage guidance.
    """
    out: list[Paper] = []
    since_s = _format_arxiv_submitted_date(since_dt)
    until_s = _format_arxiv_submitted_date(until_dt)
    for category in categories:
        start = 0
        while True:
            query = f"cat:{category}+AND+submittedDate:[{since_s}+TO+{until_s}]"
            params = {
                "search_query": query,
                "start": start,
                "max_results": page_size,
                "sortBy": "submittedDate",
                "sortOrder": "ascending",
            }
            url = "https://export.arxiv.org/api/query?" + urlencode(params)
            log.debug("fetching API: %s", url)
            xml_text = http_get_text(url, user_agent, timeout_s=60)
            batch = parse_atom_api(xml_text)
            out.extend(batch)
            if len(batch) < page_size:
                break
            start += page_size
            time.sleep(pause_s)
        time.sleep(pause_s)
    return out


def merge_papers(*paper_lists: list[Paper]) -> dict[str, Paper]:
    """Merge multiple lists of ``Paper`` by ``arxiv_id``.

    Records from the API are considered authoritative for metadata;
    fields from RSS only fill gaps. The ``source`` field is updated to
    ``api+rss`` when both have contributed.
    """
    merged: dict[str, Paper] = {}
    for papers in paper_lists:
        for paper in papers:
            if paper.arxiv_id not in merged:
                merged[paper.arxiv_id] = paper
                continue
            cur = merged[paper.arxiv_id]
            if paper.source == "api":
                cur.title = paper.title or cur.title
                cur.abstract = paper.abstract or cur.abstract
                cur.authors = paper.authors or cur.authors
                cur.categories = unique_preserve_order(cur.categories + paper.categories)
                cur.primary_category = paper.primary_category or cur.primary_category
                cur.link = paper.link or cur.link
                cur.pdf_link = paper.pdf_link or cur.pdf_link
                cur.published = paper.published or cur.published
                cur.updated = paper.updated or cur.updated
                cur.version = paper.version or cur.version
                cur.source = "api+rss" if cur.source != "api" else cur.source
            else:
                cur.categories = unique_preserve_order(cur.categories + paper.categories)
                if not cur.authors and paper.authors:
                    cur.authors = paper.authors
                if not cur.abstract and paper.abstract:
                    cur.abstract = paper.abstract
                cur.announce_type = cur.announce_type or paper.announce_type
    return merged


def filter_new_papers(
    papers: dict[str, Paper], since_dt: datetime
) -> list[Paper]:
    """Keep only papers whose published or updated time is at or after ``since_dt``."""
    out: list[Paper] = []
    for paper in papers.values():
        if paper.published and ensure_utc(paper.published) >= since_dt:
            out.append(paper)
            continue
        if paper.updated and ensure_utc(paper.updated) >= since_dt:
            out.append(paper)
    return out
