"""Download RSS feeds and turn their entries into Items."""
from __future__ import annotations

import hashlib
import html
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from urllib.parse import urljoin

import feedparser
import requests

from .config import Source
from .timeutil import from_iso, parse_date, to_iso

USER_AGENT = "Mozilla/5.0 (compatible; MascotNewsbot/1.0; +https://github.com)"
SUMMARY_CHARS = 300
CONTENT_CHARS = 1500

_TAG = re.compile(r"<[^>]+>")
_IMG = re.compile(r"<img[^>]+src=[\"']([^\"']+)[\"']", re.I)
_SPACE = re.compile(r"\s+")


@dataclass
class Item:
    id: str
    source: str
    title: str
    summary: str        # short text, for triage
    content: str        # longer text, for analysis
    link: str
    published: datetime | None
    image: str | None
    official: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["published"] = to_iso(self.published)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Item":
        d = dict(d)
        d["published"] = from_iso(d.get("published"))
        return cls(**d)


def clean_text(value: str | None, limit: int) -> str:
    if not value:
        return ""
    text = html.unescape(_TAG.sub(" ", value))
    text = _SPACE.sub(" ", text).strip()
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0] + "…"
    return text


def _first_image(entry) -> str | None:
    for key in ("media_content", "media_thumbnail"):
        for media in entry.get(key) or []:
            url = media.get("url")
            if url:
                return url
    for enc in entry.get("enclosures") or []:
        if str(enc.get("type", "")).startswith("image") and enc.get("href"):
            return enc["href"]
    blobs = [c.get("value", "") for c in entry.get("content") or []]
    blobs.append(entry.get("summary", ""))
    for blob in blobs:
        m = _IMG.search(blob or "")
        if m:
            return m.group(1)
    return None


def _item_id(source: Source, guid: str, link: str, title: str, summary: str) -> str:
    basis = link or guid or f"{source.url}|{title}|{summary[:200]}"
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]


def parse_feed(content: bytes, source: Source) -> list[Item]:
    """Parse raw feed bytes. Tolerates truncated or slightly malformed XML."""
    parsed = feedparser.parse(content)
    items: list[Item] = []
    for entry in parsed.entries:
        title = clean_text(entry.get("title"), 300)
        if not title:
            continue
        raw_link = (entry.get("link") or "").strip()
        link = urljoin(source.url, raw_link) if raw_link else ""
        guid = str(entry.get("id") or "")
        body_html = ""
        if entry.get("content"):
            body_html = entry["content"][0].get("value", "")
        summary_html = entry.get("summary", "") or body_html
        summary = clean_text(summary_html, SUMMARY_CHARS)
        content_text = clean_text(body_html or summary_html, CONTENT_CHARS)
        published = parse_date(
            entry.get("published") or entry.get("updated"),
            entry.get("published_parsed") or entry.get("updated_parsed"),
        )
        items.append(
            Item(
                id=_item_id(source, guid, link, title, summary),
                source=source.name,
                title=title,
                summary=summary,
                content=content_text,
                link=link or source.link or source.url,
                published=published,
                image=_first_image(entry),
                official=source.official,
            )
        )
    return items


def fetch_source(source: Source, timeout: float = 20) -> list[Item]:
    resp = requests.get(source.url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    resp.raise_for_status()
    items = parse_feed(resp.content, source)
    if not items:
        raise ValueError("ฟีดนี้ไม่มีข่าว หรืออ่านไม่ออก")
    return items


def fetch_all(sources: list[Source], timeout: float = 20) -> dict[str, list[Item] | Exception]:
    """Fetch every enabled source in parallel. Returns {source name: items or the error}."""
    enabled = [s for s in sources if s.enabled]
    results: dict[str, list[Item] | Exception] = {}

    def work(src: Source):
        try:
            return src.name, fetch_source(src, timeout)
        except Exception as exc:  # one broken feed must not stop the run
            return src.name, exc

    with ThreadPoolExecutor(max_workers=8) as pool:
        for name, result in pool.map(work, enabled):
            results[name] = result
    return results
