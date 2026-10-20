"""Persistent state: seen items, news events, digest slots. Stored as one JSON file."""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from .fetch import Item
from .timeutil import from_iso, to_iso

SEEN_KEEP = timedelta(days=3)       # feeds only show recent items, 3 days is plenty
EVENT_KEEP = timedelta(hours=36)    # triage needs the last 24 h of stories as context
TEXT_KEEP = timedelta(hours=6)      # full article text is only needed while a story can alert


@dataclass
class Event:
    id: str
    headline: str
    category: str
    first_seen: datetime
    last_updated: datetime
    urgency: int = 0
    sensitive: bool = False
    promotional: bool = False
    status: str = "new"          # new | notified | digested | failed
    attempts: int = 0
    items: list[Item] = field(default_factory=list)
    analysis: dict | None = None
    notified_at: datetime | None = None

    # ----- derived -----
    @property
    def source_names(self) -> list[str]:
        names: list[str] = []
        for it in self.items:
            if it.source not in names:
                names.append(it.source)
        return names

    @property
    def has_official(self) -> bool:
        return any(it.official for it in self.items)

    @property
    def earliest(self) -> datetime:
        dates = [it.published for it in self.items if it.published]
        return min(dates) if dates else self.first_seen

    @property
    def credibility(self) -> str:
        if self.has_official:
            return "หน่วยงานรัฐยืนยัน"
        if len(self.source_names) >= 2:
            return f"ยืนยัน {len(self.source_names)} แหล่ง"
        return "แหล่งเดียว รอยืนยัน"

    def add_items(self, items: list[Item]) -> None:
        known = {it.id for it in self.items}
        for it in items:
            if it.id not in known:
                self.items.append(it)
                known.add(it.id)

    # ----- (de)serialisation -----
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "headline": self.headline,
            "category": self.category,
            "first_seen": to_iso(self.first_seen),
            "last_updated": to_iso(self.last_updated),
            "urgency": self.urgency,
            "sensitive": self.sensitive,
            "promotional": self.promotional,
            "status": self.status,
            "attempts": self.attempts,
            "items": [it.to_dict() for it in self.items],
            "analysis": self.analysis,
            "notified_at": to_iso(self.notified_at),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Event":
        return cls(
            id=d["id"],
            headline=d["headline"],
            category=d.get("category", "other"),
            first_seen=from_iso(d["first_seen"]),
            last_updated=from_iso(d["last_updated"]),
            urgency=int(d.get("urgency", 0)),
            sensitive=bool(d.get("sensitive", False)),
            promotional=bool(d.get("promotional", False)),
            status=d.get("status", "new"),
            attempts=int(d.get("attempts", 0)),
            items=[Item.from_dict(x) for x in d.get("items", [])],
            analysis=d.get("analysis"),
            notified_at=from_iso(d.get("notified_at")),
        )


class State:
    def __init__(self, path: Path):
        self.path = path
        self.seen: dict[str, str] = {}
        self.events: dict[str, Event] = {}
        self.digests_sent: set[str] = set()
        self.last_digest_at: datetime | None = None
        self.bootstrapped: set[str] = set()   # sources fetched at least once

    @classmethod
    def load(cls, path: Path) -> "State":
        st = cls(path)
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            st.seen = raw.get("seen", {})
            st.events = {e["id"]: Event.from_dict(e) for e in raw.get("events", [])}
            st.digests_sent = set(raw.get("digests_sent", []))
            st.last_digest_at = from_iso(raw.get("last_digest_at"))
            st.bootstrapped = set(raw.get("bootstrapped", []))
        return st

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "seen": self.seen,
            "events": [e.to_dict() for e in sorted(self.events.values(),
                                                    key=lambda e: e.first_seen)],
            "digests_sent": sorted(self.digests_sent)[-60:],
            "last_digest_at": to_iso(self.last_digest_at),
            "bootstrapped": sorted(self.bootstrapped),
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")),
                       encoding="utf-8")
        os.replace(tmp, self.path)

    # ----- seen items -----
    def is_seen(self, item_id: str) -> bool:
        return item_id in self.seen

    def mark_seen(self, item_id: str, now: datetime) -> None:
        self.seen[item_id] = to_iso(now)

    # ----- events -----
    def new_event(self, headline: str, category: str, now: datetime) -> Event:
        ev = Event(id="e_" + uuid.uuid4().hex[:10], headline=headline, category=category,
                   first_seen=now, last_updated=now)
        self.events[ev.id] = ev
        return ev

    def recent_events(self, now: datetime, hours: float = 24,
                      top: int = 50, latest: int = 30) -> list[Event]:
        """Stories the triage model may attach new articles to: the most important
        ones of the last day plus the most recently updated ones."""
        cutoff = now - timedelta(hours=hours)
        recent = [e for e in self.events.values()
                  if e.last_updated >= cutoff and not e.promotional]
        by_score = sorted(recent, key=lambda e: e.urgency, reverse=True)[:top]
        by_time = sorted(recent, key=lambda e: e.last_updated, reverse=True)[:latest]
        picked = {e.id: e for e in by_score + by_time}
        return list(picked.values())

    def prune(self, now: datetime) -> None:
        self.seen = {k: v for k, v in self.seen.items() if now - from_iso(v) < SEEN_KEEP}
        self.events = {k: e for k, e in self.events.items()
                       if now - e.last_updated < EVENT_KEEP}
        for e in self.events.values():
            if e.status != "new" or now - e.last_updated > TEXT_KEEP:
                for it in e.items:  # past alerting: keep only titles and links
                    it.content = ""
                    it.summary = ""
