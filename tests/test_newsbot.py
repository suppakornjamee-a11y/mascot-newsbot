"""Offline tests: synthetic feeds, a fake AI and a fake LINE. Run: python -m unittest -v"""
from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import anthropic
import httpx
import httpx2

from newsbot.ai import AIError, ClaudeAnalyst, GeminiAnalyst, TriageGroup
from newsbot.config import TZ, Config, Mascot, Source
from newsbot.fetch import parse_feed
from newsbot.line import ConsoleNotifier, build_alert_messages, build_digest_message
from newsbot.pipeline import run_once
from newsbot.state import State
from newsbot.timeutil import parse_date


def bkk(y, mo, d, h, mi=0) -> datetime:
    return datetime(y, mo, d, h, mi, tzinfo=TZ).astimezone(timezone.utc)


def rfc(dt: datetime) -> str:
    return dt.strftime("%a, %d %b %Y %H:%M:%S +0000")


def rss(items: list[dict]) -> bytes:
    parts = []
    for it in items:
        date = f"<pubDate>{it['date']}</pubDate>" if it.get("date") else ""
        link = f"<link>{it['link']}</link>" if it.get("link") else ""
        parts.append(f"<item><title>{it['title']}</title>{link}{date}"
                     f"<description><![CDATA[{it.get('desc', '')}]]></description></item>")
    return ("<?xml version='1.0' encoding='UTF-8'?><rss version='2.0' "
            "xmlns:media='http://search.yahoo.com/mrss/'><channel><title>t</title>"
            "<link>https://example.com</link><description>d</description>"
            + "".join(parts) + "</channel></rss>").encode("utf-8")


SRC_A = Source("สำนักข่าว A", "https://a.example.com/feed")
SRC_B = Source("สำนักข่าว B", "https://b.example.com/feed")
SRC_TMD = Source("กรมอุตุฯ", "https://tmd.example.go.th/xml", official=True,
                 link="https://tmd.example.go.th/quake")


def claude_config() -> Config:
    cfg = make_config(Path(tempfile.mkdtemp()))
    cfg.ai_provider = "claude"
    cfg.triage_model, cfg.analysis_model = "claude-haiku-5-5", "claude-sonnet-5-5"
    return cfg


def make_config(tmp: Path) -> Config:
    return Config(mascot=Mascot(name="น้องทดสอบ"), sources=[SRC_A, SRC_B, SRC_TMD],
                  state_path=tmp / "state.json", quiet_hours=[0, 6])


class FakeAnalyst:
    """Groups by keyword and scores deterministically, like a tiny rule-based Claude."""

    def __init__(self, flood_score=85):
        self.flood_score = flood_score
        self.triage_calls = 0
        self.analyze_calls = 0
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}

    def triage(self, items, recent, now):
        self.triage_calls += 1
        flood_existing = next((e.id for e in recent if "น้ำท่วม" in e.headline), None)
        flood, groups = [], []
        for i, it in enumerate(items):
            if "น้ำ" in it.title or "ฝน" in it.title:
                flood.append(i)
            elif "รางวัล" in it.title:
                groups.append(TriageGroup([i], None, it.title, "business", 10, False, True))
            elif "เตือน" in it.title:
                groups.append(TriageGroup([i], None, it.title, "disaster", 60, False, False))
            else:
                groups.append(TriageGroup([i], None, it.title, "society", 45, False, False))
        if flood:
            groups.append(TriageGroup(flood, flood_existing, "น้ำท่วมฉับพลัน จ.ทดสอบ",
                                      "disaster", self.flood_score, True, False))
        return groups

    def analyze(self, event, now):
        self.analyze_calls += 1
        return {
            "headline": event.headline, "what_happened": "น้ำท่วมฉับพลันหลายหมู่บ้าน",
            "cause": "ฝนตกหนักต่อเนื่อง", "when": "21 ต.ค. ช่วงบ่าย", "where": "อ.เมือง จ.ทดสอบ",
            "affected": "ชาวบ้านริมน้ำ", "missing_info": ["จำนวนผู้ได้รับผลกระทบ"],
            "urgency": event.urgency, "sensitive": True, "caution": "",
            "caption": "น้องทดสอบรายงาน: น้ำท่วมฉับพลันที่ จ.ทดสอบ ขอให้ทุกคนปลอดภัย",
            "hashtags": ["#น้ำท่วม", "#จังหวัดทดสอบ"],
            "card_kicker": "เตือนภัย · สภาพอากาศ", "card_headline": "น้ำท่วมฉับพลัน จ.ทดสอบ",
            "card_detail": "ฝนตกหนักทำน้ำป่าไหลเข้าหลายหมู่บ้าน ทุกคนในพื้นที่ติดตามประกาศอำเภอ",
            "mascot_mood": "serious",
        }


def feeds_round1(now: datetime) -> dict:
    a = rss([
        {"title": "น้ำท่วมฉับพลัน อ.เมือง จ.ทดสอบ ชาวบ้านอพยพ", "link": "/news/1",
         "date": rfc(now - timedelta(minutes=15)), "desc": "<p>ฝนตกหนัก <b>น้ำป่า</b></p>"},
        {"title": "บริษัทตัวอย่างคว้ารางวัลองค์กรดีเด่น", "link": "https://a.example.com/news/2",
         "date": rfc(now - timedelta(minutes=30))},
    ])
    b = rss([
        {"title": "ฝนถล่มหนัก น้ำป่าทะลักเข้าเมือง จ.ทดสอบ", "link": "https://b.example.com/x/9",
         "date": rfc(now - timedelta(minutes=10))},
        {"title": "ข่าวเก่าเมื่อสองวันก่อน", "link": "https://b.example.com/old",
         "date": rfc(now - timedelta(days=2))},
    ])
    tmd = rss([{"title": "แผ่นดินไหว ประเทศเพื่อนบ้าน ขนาด 3.1", "desc": "รายงานเก่า"}])
    return {SRC_A.name: parse_feed(a, SRC_A), SRC_B.name: parse_feed(b, SRC_B),
            SRC_TMD.name: parse_feed(tmd, SRC_TMD)}


class DateTests(unittest.TestCase):
    def test_formats(self):
        self.assertEqual(parse_date("Tue, 20 Oct 2026 02:10:50 +0000"),
                         datetime(2026, 10, 20, 2, 10, 50, tzinfo=timezone.utc))
        # TMD: Buddhist-era year, Thai local time
        self.assertEqual(parse_date("15/4/2568 5:17:23"),
                         datetime(2025, 4, 14, 22, 17, 23, tzinfo=timezone.utc))
        self.assertEqual(parse_date("2026-10-20T19:21:05.328Z").hour, 19)
        self.assertIsNone(parse_date("not a date"))
        self.assertIsNone(parse_date(None))


class FeedTests(unittest.TestCase):
    def test_parse(self):
        now = bkk(2026, 10, 21, 14, 5)
        items = feeds_round1(now)[SRC_A.name]
        self.assertEqual(items[0].link, "https://a.example.com/news/1")  # relative link joined
        self.assertEqual(items[0].summary, "ฝนตกหนัก น้ำป่า")             # html stripped
        self.assertEqual(items[0].published, now - timedelta(minutes=15))
        tmd = feeds_round1(now)[SRC_TMD.name][0]
        self.assertEqual(tmd.link, "https://tmd.example.go.th/quake")       # fallback link
        self.assertIsNone(tmd.published)

    def test_truncated_feed_still_parses(self):
        data = rss([{"title": "ข่าวหนึ่ง", "link": "https://a.example.com/1"},
                    {"title": "ข่าวสอง", "link": "https://a.example.com/2"}])
        items = parse_feed(data[:-40], SRC_A)  # cut off the closing tags
        self.assertGreaterEqual(len(items), 1)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.cfg = make_config(self.tmp)

    def test_alert_dedupe_and_follow_up(self):
        now = bkk(2026, 10, 21, 14, 5)
        state, ai, line = State(self.cfg.state_path), FakeAnalyst(), ConsoleNotifier(echo=False)
        rep = run_once(self.cfg, state, ai, line, now, fetcher=lambda c: feeds_round1(now))

        self.assertEqual(rep.new_items, 3)            # old item + undated TMD (first run) skipped
        self.assertEqual(len(rep.alerts_sent), 1)     # flood only; PR never alerts
        self.assertEqual(len(line.batches), 1)
        flex, caption = line.batches[0]
        self.assertEqual(flex["type"], "flex")
        self.assertIn("ยืนยัน 2 แหล่ง", json.dumps(flex, ensure_ascii=False))
        self.assertIn("#น้ำท่วม", caption["text"])
        self.assertIn("ที่มา: สำนักข่าว A, สำนักข่าว B", caption["text"])
        state.save()

        # 30 min later, same feeds: nothing new, nothing sent
        later = now + timedelta(minutes=30)
        state = State.load(self.cfg.state_path)
        rep = run_once(self.cfg, state, ai, line, later, fetcher=lambda c: feeds_round1(now))
        self.assertEqual(rep.new_items, 0)
        self.assertEqual(len(line.batches), 1)
        self.assertEqual(ai.triage_calls, 1)

        # a new article on the same flood joins the existing event, no second alert
        extra = feeds_round1(now)
        extra[SRC_B.name] += parse_feed(rss([{
            "title": "ผู้ว่าฯ สั่งเปิดศูนย์พักพิง หลังน้ำท่วม จ.ทดสอบ",
            "link": "https://b.example.com/x/10", "date": rfc(later)}]), SRC_B)
        rep = run_once(self.cfg, state, ai, line, later, fetcher=lambda c: extra)
        self.assertEqual(rep.new_items, 1)
        self.assertEqual(rep.events_updated, 1)
        self.assertEqual(len(line.batches), 1)
        flood = [e for e in state.events.values() if "น้ำท่วม" in e.headline][0]
        self.assertEqual(len(flood.items), 3)
        self.assertEqual(flood.status, "notified")

    def test_undated_items_after_bootstrap_are_new(self):
        now = bkk(2026, 10, 21, 14, 5)
        state, ai, line = State(self.cfg.state_path), FakeAnalyst(), ConsoleNotifier(echo=False)
        run_once(self.cfg, state, ai, line, now, fetcher=lambda c: feeds_round1(now))
        feeds = feeds_round1(now)
        feeds[SRC_TMD.name] += parse_feed(rss([{"title": "ประกาศเตือนพายุฝนฟ้าคะนอง",
                                                "desc": "ฉบับใหม่"}]), SRC_TMD)
        rep = run_once(self.cfg, state, ai, line, now + timedelta(minutes=30),
                       fetcher=lambda c: feeds)
        self.assertEqual(rep.new_items, 1)

    def test_digest_once_per_slot(self):
        morning = bkk(2026, 10, 21, 11, 35)
        state, ai, line = State(self.cfg.state_path), FakeAnalyst(), ConsoleNotifier(echo=False)
        feeds = {SRC_A.name: parse_feed(rss([
            {"title": "ครม.เห็นชอบงบประมาณโครงการถนนสายใหม่", "link": "https://a.example.com/7",
             "date": rfc(morning - timedelta(minutes=20))},
            {"title": "ประกาศเตือนคลื่นลมแรงอ่าวไทย", "link": "https://a.example.com/8",
             "date": rfc(morning - timedelta(minutes=5))},
        ]), SRC_A)}
        rep = run_once(self.cfg, state, ai, line, morning, fetcher=lambda c: feeds)
        self.assertEqual(rep.digest_sent, 0)          # 11:35 is not a digest slot
        self.assertEqual(len(line.batches), 0)        # 45 and 60 < alert threshold

        noon = bkk(2026, 10, 21, 12, 5)
        rep = run_once(self.cfg, state, ai, line, noon, fetcher=lambda c: feeds)
        self.assertEqual(rep.digest_sent, 2)
        digest = line.batches[0][0]
        text = json.dumps(digest, ensure_ascii=False)
        self.assertIn("สรุปข่าวรอบ 12:05", text)
        self.assertLess(text.index("คลื่นลมแรง"), text.index("งบประมาณ"))  # highest score first

        rep = run_once(self.cfg, state, ai, line, noon + timedelta(minutes=30),
                       fetcher=lambda c: feeds)
        self.assertEqual(rep.digest_sent, 0)
        self.assertEqual(len(line.batches), 1)

    def test_quiet_hours(self):
        night = bkk(2026, 10, 21, 2, 0)
        line = ConsoleNotifier(echo=False)
        state = State(self.cfg.state_path)
        rep = run_once(self.cfg, state, FakeAnalyst(85), line, night,
                       fetcher=lambda c: feeds_round1(night))
        self.assertEqual(rep.held_quiet, 1)
        self.assertEqual(len(line.batches), 0)

        # the held story goes out on the first run after quiet hours end
        rep = run_once(self.cfg, state, FakeAnalyst(85), line, bkk(2026, 10, 21, 6, 5),
                       fetcher=lambda c: {})
        self.assertEqual(len(rep.alerts_sent), 1)

        state2 = State(self.tmp / "s2.json")
        rep = run_once(self.cfg, state2, FakeAnalyst(95), line, night,
                       fetcher=lambda c: feeds_round1(night))
        self.assertEqual(len(rep.alerts_sent), 1)     # 95 >= quiet_override

    def test_failed_ai_retries_next_run(self):
        class Broken(FakeAnalyst):
            def triage(self, *a):
                raise RuntimeError("API down")

        now = bkk(2026, 10, 21, 14, 5)
        state, line = State(self.cfg.state_path), ConsoleNotifier(echo=False)
        rep = run_once(self.cfg, state, Broken(), line, now, fetcher=lambda c: feeds_round1(now))
        self.assertTrue(rep.errors)
        rep = run_once(self.cfg, state, FakeAnalyst(), line, now + timedelta(minutes=30),
                       fetcher=lambda c: feeds_round1(now))
        self.assertEqual(rep.new_items, 3)            # items were not lost
        self.assertEqual(len(rep.alerts_sent), 1)

    def test_state_roundtrip(self):
        now = bkk(2026, 10, 21, 14, 5)
        state = State(self.cfg.state_path)
        run_once(self.cfg, state, FakeAnalyst(), ConsoleNotifier(echo=False), now,
                 fetcher=lambda c: feeds_round1(now))
        state.save()
        again = State.load(self.cfg.state_path)
        self.assertEqual(again.seen, state.seen)
        self.assertEqual({e.id: e.to_dict() for e in again.events.values()},
                         {e.id: e.to_dict() for e in state.events.values()})


class LineMessageTests(unittest.TestCase):
    def test_flex_within_limits(self):
        now = bkk(2026, 10, 21, 14, 5)
        state = State(Path(tempfile.mkdtemp()) / "s.json")
        ai, line = FakeAnalyst(), ConsoleNotifier(echo=False)
        cfg = make_config(Path(tempfile.mkdtemp()))
        run_once(cfg, state, ai, line, now, fetcher=lambda c: feeds_round1(now))
        msgs = line.batches[0]
        flex = msgs[0]
        self.assertLessEqual(len(flex["altText"]), 400)
        self.assertLess(len(json.dumps(flex["contents"], ensure_ascii=False).encode()), 30_000)
        buttons = flex["contents"]["footer"]["contents"]
        self.assertEqual(len(buttons), 2)
        for b in buttons:
            self.assertLessEqual(len(b["action"]["label"]), 20)
            self.assertTrue(b["action"]["uri"].startswith("https://"))
        self.assertLessEqual(len(msgs[1]["text"]), 5000)

        digest = build_digest_message(list(state.events.values()), now)
        self.assertLessEqual(len(digest["altText"]), 400)

        def walk(node):  # every text component must have non-empty text
            if isinstance(node, dict):
                if node.get("type") == "text":
                    self.assertTrue(node["text"].strip())
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)
        walk(flex)
        walk(digest)


class AnthropicRequestTests(unittest.TestCase):
    """Checks the real SDK call shape against a mocked HTTP endpoint."""

    def _client(self, handler):
        return anthropic.Anthropic(api_key="test-key", max_retries=0,
                                   http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))

    @staticmethod
    def _message(payload: dict) -> httpx2.Response:
        return httpx2.Response(200, json={
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
            "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 120, "output_tokens": 40},
        })

    def test_triage_request_and_parse(self):
        seen = []

        def handler(request: httpx2.Request):
            seen.append(json.loads(request.content))
            return self._message({"groups": [{
                "item_indexes": [0], "existing_event_id": None, "headline": "ทดสอบ",
                "category": "DISASTER", "urgency": 140, "sensitive": False, "promotional": False}]})

        analyst = ClaudeAnalyst(claude_config(), client=self._client(handler))
        now = bkk(2026, 10, 21, 14, 5)
        items = feeds_round1(now)[SRC_A.name]
        groups = analyst.triage(items, [], now)

        body = seen[0]
        self.assertEqual(body["model"], "claude-haiku-5-5")
        self.assertEqual(body["output_config"]["format"]["type"], "json_schema")
        self.assertEqual(body["output_config"]["effort"], "low")
        self.assertIn("น้ำท่วมฉับพลัน", body["messages"][0]["content"])
        self.assertEqual(groups[0].category, "disaster")   # case-insensitive enum
        self.assertEqual(groups[0].urgency, 100)           # clamped
        self.assertEqual(analyst.usage["input_tokens"], 120)

    def test_effort_rejected_falls_back(self):
        calls = []

        def handler(request: httpx2.Request):
            body = json.loads(request.content)
            calls.append(body)
            if "effort" in body["output_config"]:
                return httpx2.Response(400, json={"type": "error", "error": {
                    "type": "invalid_request_error", "message": "effort is not supported"}})
            return self._message({"groups": []})

        analyst = ClaudeAnalyst(claude_config(), client=self._client(handler))
        self.assertEqual(analyst.triage([], [], bkk(2026, 10, 21, 14)), [])
        self.assertEqual(len(calls), 2)
        self.assertNotIn("effort", calls[1]["output_config"])

    def test_analysis_schema_rules(self):
        from newsbot.ai import ANALYSIS_SCHEMA, TRIAGE_SCHEMA

        banned = {"minimum", "maximum", "minLength", "maxLength", "maxItems", "multipleOf"}

        def check(node):
            if isinstance(node, dict):
                self.assertFalse(banned & node.keys())
                if node.get("type") == "object":
                    self.assertIs(node.get("additionalProperties"), False)
                    self.assertEqual(set(node["required"]), set(node["properties"]))
                for v in node.values():
                    check(v)
            elif isinstance(node, list):
                for v in node:
                    check(v)
        check(TRIAGE_SCHEMA)
        check(ANALYSIS_SCHEMA)


class GeminiRequestTests(unittest.TestCase):
    """Checks the real google-genai SDK call shape against a mocked HTTP endpoint."""

    GROUP = {"item_indexes": [0], "existing_event_id": None, "headline": "ทดสอบ",
             "category": "disaster", "urgency": 88, "sensitive": False, "promotional": False}

    def _analyst(self, handler) -> GeminiAnalyst:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key="test-key", http_options=types.HttpOptions(
            httpx_client=httpx.Client(transport=httpx.MockTransport(handler))))
        return GeminiAnalyst(make_config(Path(tempfile.mkdtemp())), client=client,
                             retry_waits=(0,))

    @staticmethod
    def _reply(payload: dict, finish: str = "STOP") -> httpx.Response:
        return httpx.Response(200, json={
            "candidates": [{"content": {"role": "model", "parts": [
                {"text": json.dumps(payload, ensure_ascii=False)}]}, "finishReason": finish}],
            "usageMetadata": {"promptTokenCount": 300, "candidatesTokenCount": 50,
                              "thoughtsTokenCount": 25},
        })

    @staticmethod
    def _error(code: int, status: str, message: str) -> httpx.Response:
        return httpx.Response(code, json={"error": {"code": code, "message": message,
                                                    "status": status}})

    def test_request_shape_and_parse(self):
        seen = []

        def handler(request: httpx.Request):
            seen.append((str(request.url), json.loads(request.content)))
            return self._reply({"groups": [self.GROUP]})

        analyst = self._analyst(handler)
        now = bkk(2026, 10, 21, 14, 5)
        groups = analyst.triage(feeds_round1(now)[SRC_A.name], [], now)

        url, body = seen[0]
        self.assertIn("models/gemini-3.5-flash-lite:generateContent", url)
        gen = body["generationConfig"]
        self.assertEqual(gen["responseMimeType"], "application/json")
        self.assertIn("groups", gen["responseJsonSchema"]["properties"])
        thinking = gen["thinkingConfig"]
        self.assertEqual(thinking.get("thinkingLevel", thinking.get("thinking_level")), "LOW")
        self.assertIn("บรรณาธิการข่าว", body["systemInstruction"]["parts"][0]["text"])
        self.assertIn("น้ำท่วมฉับพลัน", body["contents"][0]["parts"][0]["text"])
        self.assertEqual({s["threshold"] for s in body["safetySettings"]}, {"BLOCK_ONLY_HIGH"})
        self.assertEqual(groups[0].urgency, 88)
        self.assertEqual(analyst.usage["input_tokens"], 300)
        self.assertEqual(analyst.usage["output_tokens"], 75)   # answer + thinking

    def test_analysis_uses_analysis_model(self):
        seen = []

        def handler(request: httpx.Request):
            seen.append(str(request.url))
            return self._reply(FakeAnalyst().analyze(type("E", (), {
                "headline": "h", "urgency": 80})(), None))

        now = bkk(2026, 10, 21, 14, 5)
        state = State(Path(tempfile.mkdtemp()) / "s.json")
        ev = state.new_event("น้ำท่วม", "disaster", now)
        ev.add_items(feeds_round1(now)[SRC_A.name][:1])
        out = self._analyst(handler).analyze(ev, now)
        self.assertIn("models/gemini-3.8-flash:generateContent", seen[0])
        self.assertEqual(out["mascot_mood"], "serious")     # disaster story: never cheerful

    def test_thinking_level_rejected_falls_back(self):
        calls = []

        def handler(request: httpx.Request):
            body = json.loads(request.content)
            calls.append(body)
            if "thinkingConfig" in body.get("generationConfig", {}):
                return self._error(400, "INVALID_ARGUMENT",
                                   "Thinking level is not supported for this model.")
            return self._reply({"groups": []})

        self.assertEqual(self._analyst(handler).triage([], [], bkk(2026, 10, 21, 14)), [])
        self.assertEqual(len(calls), 2)
        self.assertNotIn("thinkingConfig", calls[1]["generationConfig"])

    def test_overloaded_then_ok(self):
        calls = []

        def handler(request: httpx.Request):
            calls.append(1)
            if len(calls) == 1:
                return self._error(503, "UNAVAILABLE", "The model is overloaded.")
            return self._reply({"groups": []})

        self.assertEqual(self._analyst(handler).triage([], [], bkk(2026, 10, 21, 14)), [])
        self.assertEqual(len(calls), 2)

    def test_quota_exhausted_raises(self):
        from google.genai import errors

        def handler(request: httpx.Request):
            return self._error(429, "RESOURCE_EXHAUSTED", "Quota exceeded")

        with self.assertRaises(errors.ClientError):
            self._analyst(handler).triage([], [], bkk(2026, 10, 21, 14))

    def test_safety_block_raises(self):
        def handler(request: httpx.Request):
            return self._reply({"groups": []}, finish="SAFETY")

        with self.assertRaises(AIError):
            self._analyst(handler).triage([], [], bkk(2026, 10, 21, 14))


class DemoTests(unittest.TestCase):
    def test_demo_sends_top_story_and_leaves_state_alone(self):
        from newsbot.pipeline import run_demo

        tmp = Path(tempfile.mkdtemp())
        cfg = make_config(tmp)
        now = bkk(2026, 10, 21, 3, 15)          # inside quiet hours on purpose
        feeds = feeds_round1(now - timedelta(hours=10))  # 10 h old: beyond normal 6 h window
        line = ConsoleNotifier(echo=False)
        res = run_demo(cfg, FakeAnalyst(flood_score=55), line, now, fetcher=lambda c: feeds)

        self.assertTrue(res.sent)
        self.assertEqual(res.urgency, 55)                 # below the alert threshold, still sent
        self.assertEqual(set(res.sources), {"สำนักข่าว A", "สำนักข่าว B"})
        flex = line.batches[0][0]
        self.assertTrue(flex["altText"].startswith("ตัวอย่าง (55)"))
        self.assertFalse(cfg.state_path.exists())         # real state untouched

    def test_demo_reports_empty_feeds(self):
        from newsbot.pipeline import run_demo

        res = run_demo(make_config(Path(tempfile.mkdtemp())), FakeAnalyst(),
                       ConsoleNotifier(echo=False), bkk(2026, 10, 21, 3), fetcher=lambda c: {})
        self.assertFalse(res.sent)


class FakeImageHost:
    note = ""

    def __init__(self):
        self.files = {}

    def upload(self, path, data):
        self.files[path] = data
        return f"https://img.example.com/{path}"


class PostImageTests(unittest.TestCase):
    def test_alert_carries_post_image(self):
        now = bkk(2026, 10, 21, 14, 5)
        cfg = make_config(Path(tempfile.mkdtemp()))
        host, line = FakeImageHost(), ConsoleNotifier(echo=False)
        rep = run_once(cfg, State(cfg.state_path), FakeAnalyst(), line, now,
                       fetcher=lambda c: feeds_round1(now), images=host)
        self.assertEqual(rep.cards_made, 1)
        flex, image, caption = line.batches[0]
        self.assertEqual(image["type"], "image")
        self.assertTrue(image["originalContentUrl"].endswith(".png"))
        self.assertTrue(image["previewImageUrl"].endswith("-s.jpg"))
        self.assertEqual(caption["type"], "text")
        png = next(v for k, v in host.files.items() if k.endswith(".png"))
        jpg = next(v for k, v in host.files.items() if k.endswith(".jpg"))
        self.assertLess(len(png), 10_000_000)          # LINE limits
        self.assertLess(len(jpg), 1_000_000)

    def test_broken_image_never_blocks_alert(self):
        class Broken(FakeImageHost):
            def upload(self, path, data):
                raise RuntimeError("GitHub down")

        now = bkk(2026, 10, 21, 14, 5)
        cfg = make_config(Path(tempfile.mkdtemp()))
        line = ConsoleNotifier(echo=False)
        rep = run_once(cfg, State(cfg.state_path), FakeAnalyst(), line, now,
                       fetcher=lambda c: feeds_round1(now), images=Broken())
        self.assertEqual(len(rep.alerts_sent), 1)
        self.assertEqual(len(line.batches[0]), 2)        # flex + caption, no image
        self.assertTrue(any("GitHub down" in w for w in rep.warnings))
        self.assertFalse(rep.errors)

    def test_card_renders_long_and_short_text(self):
        from newsbot.card import CardText, card_bytes, render_card

        for size in ((1080, 1080), (1080, 1350)):
            for head in ("ใต้ฝนหนัก 21–25 ต.ค.",
                         "สมาคมโรงแรมชี้ นักท่องเที่ยวภูเก็ต-กระบี่ชะลอตัวก่อนไฮซีซั่น "
                         "เพราะเที่ยวบินตรงลดลงมากกว่าที่คาด ผู้ประกอบการกังวลรายได้ทั้งปี"):
                img = render_card(CardText("เศรษฐกิจ", head, "รายละเอียด " * 30, "21 ต.ค. 2569",
                                           "smile"), size)
                self.assertEqual(img.size, size)
                full, prev = card_bytes(img)
                self.assertTrue(full.startswith(b"\x89PNG"))
                self.assertTrue(prev.startswith(b"\xff\xd8"))

    def test_headline_breaks_between_phrases(self):
        from newsbot.card import font, wrap_balanced

        f = font("head", 80)
        lines = wrap_balanced(f, "ใต้ฝนหนักถึงหนักมาก 21–25 ต.ค.", lambda i: 620, 3, ("phrase",))
        if lines:  # at 80px the first phrase may not fit; then no phrase layout exists
            self.assertNotIn("หนัก\nมาก", "\n".join(lines))
        lines = wrap_balanced(font("head", 64), "ใต้ฝนหนักถึงหนักมาก 21–25 ต.ค.",
                              lambda i: 620, 3, ("phrase",))
        self.assertEqual(lines, ["ใต้ฝนหนักถึงหนักมาก", "21–25 ต.ค."])

    def test_mood_mapping(self):
        from newsbot.card import card_text_from

        now = bkk(2026, 10, 21, 14, 5)
        ev = State(Path(tempfile.mkdtemp()) / "s.json").new_event("x", "economy", now)
        t = card_text_from({"card_headline": "h", "mascot_mood": "EXCITED"}, ev, now, False)
        self.assertEqual(t.mood, "excited")
        self.assertEqual(card_text_from({"mascot_mood": "party"}, ev, now, False).mood, "smile")


class GitHubImageHostTests(unittest.TestCase):
    def _host(self, branch_exists: bool, private: bool = False):
        from newsbot.imagehost import GitHubImageHost

        calls = []

        class Resp:
            def __init__(self, code, data=None):
                self.status_code, self._data, self.text = code, data or {}, ""

            def json(self):
                return self._data

        class Session:
            headers = {}

            def request(self, method, url, timeout=None, json=None):
                calls.append((method, url, json))
                if url == "/repos/o/r".join(["https://api.github.com", ""]):
                    return Resp(200, {"private": private, "default_branch": "main"})
                if url.endswith("/git/ref/heads/cards"):
                    return Resp(200 if branch_exists else 404, {"object": {"sha": "c1"}})
                if url.endswith("/git/ref/heads/main"):
                    return Resp(200, {"object": {"sha": "abc"}})
                return Resp(201, {})

        return GitHubImageHost("o/r", "t", session=Session()), calls

    def test_creates_branch_once_and_returns_raw_url(self):
        host, calls = self._host(branch_exists=False)
        url = host.upload("cards/2026/10/21/1405-e1.png", b"png")
        host.upload("cards/2026/10/21/1405-e1-s.jpg", b"jpg")
        self.assertEqual(url, "https://raw.githubusercontent.com/o/r/cards/cards/2026/10/21/1405-e1.png")
        posts = [c for c in calls if c[0] == "POST"]
        self.assertEqual(posts[0][2], {"ref": "refs/heads/cards", "sha": "abc"})
        self.assertEqual(len(posts), 1)
        puts = [c for c in calls if c[0] == "PUT"]
        self.assertEqual(puts[0][2]["branch"], "cards")
        self.assertEqual(puts[0][2]["content"], "cG5n")   # base64 of b"png"

    def test_private_repo_is_reported(self):
        from newsbot.imagehost import ImageHostError

        host, _ = self._host(branch_exists=True, private=True)
        with self.assertRaises(ImageHostError):
            host.upload("x.png", b"png")


class ConfigTests(unittest.TestCase):
    def test_default_is_gemini_and_switchable(self):
        from newsbot.config import ROOT, load_config

        cfg = load_config()
        self.assertEqual(cfg.ai_provider, "gemini")
        self.assertEqual(cfg.api_key_env, "GEMINI_API_KEY")
        self.assertEqual(cfg.analysis_model, "gemini-3.8-flash")

        text = (ROOT / "config.yaml").read_text(encoding="utf-8").replace(
            "provider: gemini", "provider: claude")
        tmp = Path(tempfile.mkdtemp()) / "config.yaml"
        tmp.write_text(text, encoding="utf-8")
        cfg = load_config(tmp)
        self.assertEqual(cfg.api_key_env, "ANTHROPIC_API_KEY")
        self.assertEqual(cfg.triage_model, "claude-haiku-5-5")


if __name__ == "__main__":
    unittest.main()
