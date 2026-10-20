"""LINE Messaging API: build Flex messages and push them to your own LINE account."""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime

import requests

from .ai import CATEGORIES
from .state import Event
from .timeutil import ago, thai_clock

PUSH_URL = "https://api.line.me/v2/bot/message/push"

RED = "#C62828"
ORANGE = "#E65100"
BLUE = "#1565C0"
GREY = "#8A8A8A"
INK = "#1F1F1F"
WARN_BG = "#FFF3E0"


class LineError(RuntimeError):
    pass


def _cut(text: str, limit: int) -> str:
    text = (text or "").strip() or "-"
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _https(url: str | None) -> str | None:
    if url and url.startswith("https://") and len(url) <= 1000:
        return url
    if url and url.startswith("http://") and len(url) <= 999:
        return "https://" + url[len("http://"):]
    return None


def _row(label: str, value: str) -> dict:
    return {
        "type": "box", "layout": "horizontal", "spacing": "md",
        "contents": [
            {"type": "text", "text": label, "size": "sm", "color": GREY, "flex": 2},
            {"type": "text", "text": _cut(value, 400), "size": "sm", "color": INK,
             "wrap": True, "flex": 5},
        ],
    }


def urgency_color(score: int) -> str:
    if score >= 90:
        return RED
    if score >= 70:
        return ORANGE
    return BLUE


# ---------------------------------------------------------------- alert
def build_alert_messages(event: Event, analysis: dict, now: datetime,
                         mascot_name: str = "", label: str = "ข่าวด่วน",
                         image: tuple[str, str] | None = None) -> list[dict]:
    """A Flex card with the analysis, the post image (if any), and the caption ready to copy.

    ``image`` is (full-size URL, preview URL), both public HTTPS.
    """
    score = int(analysis.get("urgency", event.urgency))
    category = CATEGORIES.get(event.category, "อื่นๆ")
    first = event.earliest
    headline = analysis.get("headline") or event.headline

    body: list[dict] = [
        {"type": "text", "text": _cut(headline, 200), "weight": "bold", "size": "lg",
         "wrap": True, "color": INK},
        {"type": "text", "size": "xs", "color": GREY, "wrap": True,
         "text": f"ข่าวแรกออก {thai_clock(first)} ({ago(first, now)}) · "
                 f"{len(event.source_names)} แหล่ง"},
        {"type": "separator", "margin": "md"},
        _row("เกิดอะไร", analysis.get("what_happened", "")),
        _row("สาเหตุ", analysis.get("cause", "")),
        _row("เมื่อไหร่", analysis.get("when", "")),
        _row("ที่ไหน", analysis.get("where", "")),
        _row("ผลกระทบ", analysis.get("affected", "")),
    ]
    missing = [m for m in analysis.get("missing_info", []) if str(m).strip()]
    if missing:
        body.append(_row("ยังไม่รู้", " · ".join(missing[:4])))

    warnings = []
    if analysis.get("caution"):
        warnings.append(analysis["caution"])
    if len(event.source_names) < 2 and not event.has_official:
        warnings.append("มีแหล่งข่าวเดียว ควรรอยืนยันก่อนโพสต์")
    if analysis.get("sensitive") or event.sensitive:
        warnings.append("ข่าวอ่อนไหว ตรวจน้ำเสียงและรายละเอียดให้ดี")
    if warnings:
        body.append({
            "type": "box", "layout": "vertical", "backgroundColor": WARN_BG,
            "cornerRadius": "md", "paddingAll": "10px", "margin": "md",
            "contents": [{"type": "text", "text": _cut("ระวัง: " + " / ".join(warnings), 500),
                          "size": "xs", "color": ORANGE, "wrap": True}],
        })

    buttons = []
    used = set()
    for it in event.items:
        uri = _https(it.link)
        if not uri or it.source in used:
            continue
        used.add(it.source)
        buttons.append({"type": "button", "style": "link", "height": "sm",
                        "action": {"type": "uri", "label": _cut(f"อ่าน: {it.source}", 20),
                                   "uri": uri}})
        if len(buttons) == 3:
            break

    bubble = {
        "type": "bubble", "size": "mega",
        "header": {
            "type": "box", "layout": "vertical", "backgroundColor": urgency_color(score),
            "paddingAll": "12px",
            "contents": [
                {"type": "text", "text": f"{label} · คะแนน {score}", "color": "#FFFFFF",
                 "weight": "bold", "size": "sm"},
                {"type": "text", "text": f"{category} · {event.credibility}",
                 "color": "#FFFFFF", "size": "xs"},
            ],
        },
        "body": {"type": "box", "layout": "vertical", "spacing": "md", "contents": body},
    }
    if buttons:
        bubble["footer"] = {"type": "box", "layout": "vertical", "spacing": "none",
                            "contents": buttons}

    alt = _cut(f"{label} ({score}): {headline}", 400)
    caption_msg = {"type": "text", "text": _cut(build_post_text(event, analysis), 4900)}
    messages = [{"type": "flex", "altText": alt, "contents": bubble}]
    if image and _https(image[0]) and _https(image[1]):
        messages.append({"type": "image", "originalContentUrl": image[0],
                         "previewImageUrl": image[1]})
    messages.append(caption_msg)
    return messages


def build_post_text(event: Event, analysis: dict) -> str:
    """The post text, ready to long-press and copy in LINE."""
    parts = [analysis.get("caption", "").strip()]
    tags = " ".join(analysis.get("hashtags", []))
    if tags:
        parts.append(tags)
    parts.append("ที่มา: " + ", ".join(event.source_names))
    return "\n\n".join(p for p in parts if p)


# ---------------------------------------------------------------- digest
def build_digest_message(events: list[Event], now: datetime) -> dict:
    rows: list[dict] = []
    for i, ev in enumerate(events):
        if i:
            rows.append({"type": "separator", "margin": "md"})
        entry = {
            "type": "box", "layout": "vertical", "spacing": "xs", "margin": "md",
            "contents": [
                {"type": "text", "text": _cut(ev.headline, 150), "size": "sm",
                 "weight": "bold", "wrap": True, "color": INK},
                {"type": "text", "size": "xxs", "color": GREY, "wrap": True,
                 "text": f"คะแนน {ev.urgency} · {CATEGORIES.get(ev.category, 'อื่นๆ')} · "
                         f"{len(ev.source_names)} แหล่ง · {ago(ev.earliest, now)}"},
            ],
        }
        link = next((_https(it.link) for it in ev.items if _https(it.link)), None)
        if link:
            entry["action"] = {"type": "uri", "label": "อ่านข่าว", "uri": link}
        rows.append(entry)

    bubble = {
        "type": "bubble", "size": "mega",
        "header": {
            "type": "box", "layout": "vertical", "backgroundColor": BLUE, "paddingAll": "12px",
            "contents": [
                {"type": "text", "text": f"สรุปข่าวรอบ {thai_clock(now)}", "color": "#FFFFFF",
                 "weight": "bold", "size": "sm"},
                {"type": "text", "text": f"{len(events)} เรื่องที่น่าสนใจ แตะเพื่ออ่านข่าวต้นทาง",
                 "color": "#FFFFFF", "size": "xs"},
            ],
        },
        "body": {"type": "box", "layout": "vertical", "contents": rows},
    }
    return {"type": "flex", "altText": _cut(f"สรุปข่าว {len(events)} เรื่อง", 400),
            "contents": bubble}


# ---------------------------------------------------------------- senders
class LineNotifier:
    def __init__(self, token: str, user_id: str, timeout: float = 20):
        self.token = token
        self.user_id = user_id
        self.timeout = timeout
        self.sent = 0

    def push(self, messages: list[dict]) -> None:
        if not 1 <= len(messages) <= 5:
            raise LineError("LINE ส่งได้ครั้งละ 1-5 ข้อความ")
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "X-Line-Retry-Key": str(uuid.uuid4()),  # same key on retry = no duplicate
        }
        body = {"to": self.user_id, "messages": messages}
        last_error = ""
        for attempt in range(3):
            try:
                resp = requests.post(PUSH_URL, headers=headers, json=body, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = str(exc)
            else:
                if resp.status_code == 200:
                    self.sent += 1
                    return
                if resp.status_code == 409:  # already accepted under this retry key
                    self.sent += 1
                    return
                last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
                if resp.status_code < 500 and resp.status_code != 429:
                    break
            time.sleep(2 * (attempt + 1))
        raise LineError(f"ส่ง LINE ไม่สำเร็จ — {last_error}")


class ConsoleNotifier:
    """Dry-run: print what would be sent instead of calling LINE."""

    def __init__(self, echo: bool = True):
        self.echo = echo
        self.batches: list[list[dict]] = []

    def push(self, messages: list[dict]) -> None:
        self.batches.append(messages)
        if not self.echo:
            return
        print("\n" + "=" * 60 + "\n[DRY RUN] ข้อความที่จะส่งเข้า LINE:")
        for msg in messages:
            if msg["type"] == "text":
                print("--- ข้อความ ---\n" + msg["text"])
            else:
                print("--- Flex: " + msg["altText"])
                print(json.dumps(msg["contents"], ensure_ascii=False)[:600] + " …")
