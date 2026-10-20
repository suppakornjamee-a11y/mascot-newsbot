"""Date parsing and Thai-friendly time formatting."""
from __future__ import annotations

import email.utils
import re
from datetime import datetime, timedelta, timezone

from .config import TZ

_THAI_STYLE = re.compile(
    r"^\s*(\d{1,2})/(\d{1,2})/(\d{4})\s+(\d{1,2}):(\d{2})(?::(\d{2}))?\s*$"
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def parse_date(raw: str | None, struct=None) -> datetime | None:
    """Parse a feed date into an aware UTC datetime.

    Handles feedparser's parsed struct (already UTC), RFC 822 dates, ISO 8601,
    and the TMD style ``15/4/2568 5:17:23`` (Buddhist-era year, Thai time).
    """
    if struct:
        try:
            return datetime(*struct[:6], tzinfo=timezone.utc)
        except (TypeError, ValueError):
            pass
    if not raw:
        return None
    raw = raw.strip()

    m = _THAI_STYLE.match(raw)
    if m:
        day, month, year, hour, minute, second = m.groups()
        year = int(year)
        if year > 2400:  # Buddhist era
            year -= 543
        try:
            local = datetime(year, int(month), int(day), int(hour), int(minute),
                             int(second or 0), tzinfo=TZ)
            return local.astimezone(timezone.utc)
        except ValueError:
            return None

    try:
        dt = email.utils.parsedate_to_datetime(raw)
        if dt is not None:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=TZ)
            return dt.astimezone(timezone.utc)
    except (TypeError, ValueError, IndexError):
        pass

    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=TZ)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def to_iso(dt: datetime | None) -> str | None:
    return dt.astimezone(timezone.utc).isoformat() if dt else None


def from_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def thai_clock(dt: datetime) -> str:
    """'13:58 น.' in Thai time."""
    return dt.astimezone(TZ).strftime("%H:%M น.")


def thai_datetime(dt: datetime) -> str:
    """'21/10 13:58 น.' in Thai time."""
    local = dt.astimezone(TZ)
    return f"{local.day}/{local.month} {local.strftime('%H:%M')} น."


def ago(dt: datetime, now: datetime) -> str:
    """Relative time in Thai: 'เมื่อสักครู่', '7 นาทีที่แล้ว', '2 ชม.ที่แล้ว'."""
    minutes = int((now - dt).total_seconds() // 60)
    if minutes < 2:
        return "เมื่อสักครู่"
    if minutes < 60:
        return f"{minutes} นาทีที่แล้ว"
    hours = minutes // 60
    if hours < 24:
        return f"{hours} ชม.ที่แล้ว"
    return f"{hours // 24} วันที่แล้ว"


def in_hour_window(now: datetime, window: list[int]) -> bool:
    """True when the Thai-time hour of ``now`` is inside [start, end) (may wrap midnight)."""
    if not window or len(window) != 2:
        return False
    start, end = int(window[0]), int(window[1])
    hour = now.astimezone(TZ).hour
    if start == end:
        return False
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end


def due_digest_slot(now: datetime, hours: list[int], sent: set[str],
                    grace: timedelta = timedelta(hours=3)) -> str | None:
    """Return the key of the most recent digest slot that is due and not yet sent."""
    local = now.astimezone(TZ)
    due = None
    for h in sorted(int(x) for x in hours):
        slot = local.replace(hour=h, minute=0, second=0, microsecond=0)
        key = slot.strftime("%Y-%m-%d-%H")
        if slot <= local < slot + grace and key not in sent:
            due = key
    return due
