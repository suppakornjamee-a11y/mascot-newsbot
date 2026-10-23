"""Offline tests for the editor agent, the picture agent and the photo card.

Run: python -m unittest -v
"""
from __future__ import annotations

import base64
import io
import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import anthropic
import httpx
import httpx2
from PIL import Image

from newsbot.art import (CHECK_SCHEMA, PICK_SCHEMA, ArtDirector, Background, GeminiPainter,
                         PexelsClient, Photo)
from newsbot.editor import POST_SCHEMA, REVIEW_SCHEMA, NewsEditor
from newsbot.line import ConsoleNotifier, build_alert_messages
from newsbot.pipeline import PostImage, RunReport, make_post_image, run_once
from newsbot.state import State
from tests.test_newsbot import (FakeAnalyst, FakeImageHost, bkk, claude_config, feeds_round1,
                                make_config)

FACTS = {
    "headline": "น้ำท่วมฉับพลัน จ.ทดสอบ", "what_happened": "น้ำท่วมฉับพลันหลายหมู่บ้าน",
    "cause": "ฝนตกหนักต่อเนื่อง", "when": "21 ต.ค. ช่วงบ่าย", "where": "อ.เมือง จ.ทดสอบ",
    "affected": "ชาวบ้านริมน้ำ", "missing_info": ["จำนวนผู้ได้รับผลกระทบ"],
    "urgency": 85, "sensitive": True, "caution": "",
}

DRAFT = {
    "card_topic": "เตือนภัย · สภาพอากาศ", "card_headline": "น้ำท่วมฉับพลัน จ.ทดสอบ",
    "card_detail": "ฝนตกหนักทำน้ำป่าไหลเข้าหลายหมู่บ้าน", "mascot_mood": "excited",
    "caption": "น้ำท่วมฉับพลันที่ จ.ทดสอบ มีผู้อพยพ 300 คน\n\nฝนตกหนักต่อเนื่อง",
    "hashtags": ["น้ำท่วม", "#จังหวัด ทดสอบ"],
    "image_queries": ["flooded street Thailand", " ", "heavy rain"],
    "image_prompt": "A flooded village road under heavy rain",
}

FIXED = {
    "issues": ["ตัวเลขผู้อพยพ 300 คน ไม่มีในข่าวต้นทาง จึงลบออก"],
    "card_topic": DRAFT["card_topic"], "card_headline": DRAFT["card_headline"],
    "card_detail": DRAFT["card_detail"],
    "caption": "น้ำท่วมฉับพลันที่ จ.ทดสอบ\n\nฝนตกหนักต่อเนื่อง",
}


def jpeg(size=(1200, 1200), color=(40, 90, 160)) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", size, color).save(out, "JPEG")
    return out.getvalue()


def flood_event(now):
    state = State(Path(tempfile.mkdtemp()) / "s.json")
    ev = state.new_event("น้ำท่วมฉับพลัน จ.ทดสอบ", "disaster", now)
    ev.add_items(feeds_round1(now)["สำนักข่าว A"][:1] + feeds_round1(now)["สำนักข่าว B"][:1])
    return ev


class ScriptedAI:
    """Answers ask() from a queue, keyed by the schema it was called with."""

    def __init__(self, **answers):
        self.answers = answers           # name -> dict or Exception
        self.calls = []

    def ask(self, *, system, payload, schema, model=None, effort=None, max_tokens=16000,
            images=None):
        name = {id(POST_SCHEMA): "write", id(REVIEW_SCHEMA): "review",
                id(PICK_SCHEMA): "pick", id(CHECK_SCHEMA): "check"}[id(schema)]
        self.calls.append((name, payload, images, model))
        answer = self.answers[name]
        if isinstance(answer, Exception):
            raise answer
        return json.loads(json.dumps(answer))


def schema_is_strict(test: unittest.TestCase, schema: dict) -> None:
    """Rules both structured-output APIs accept: no numeric/length limits, closed objects."""
    banned = {"minimum", "maximum", "minLength", "maxLength", "maxItems", "multipleOf"}

    def check(node):
        if isinstance(node, dict):
            test.assertFalse(banned & node.keys())
            if node.get("type") == "object":
                test.assertIs(node.get("additionalProperties"), False)
                test.assertEqual(set(node["required"]), set(node["properties"]))
            for v in node.values():
                check(v)
        elif isinstance(node, list):
            for v in node:
                check(v)
    check(schema)


# ================================================================ editor
class EditorTests(unittest.TestCase):
    def setUp(self):
        self.now = bkk(2026, 10, 21, 14, 5)
        self.cfg = make_config(Path(tempfile.mkdtemp()))
        self.ev = flood_event(self.now)

    def test_schemas_are_strict(self):
        for schema in (POST_SCHEMA, REVIEW_SCHEMA, PICK_SCHEMA, CHECK_SCHEMA):
            schema_is_strict(self, schema)

    def test_review_fixes_are_applied(self):
        ai = ScriptedAI(write=DRAFT, review=FIXED)
        post = NewsEditor(self.cfg, ai).write(self.ev, FACTS, self.now)

        self.assertEqual([c[0] for c in ai.calls], ["write", "review"])
        # writing on the analysis model, checking on the lighter review model
        self.assertEqual([c[3] for c in ai.calls], [None, self.cfg.review_model])
        self.assertNotIn("300", post["caption"])
        self.assertEqual(post["review"], {"checked": True, "issues": FIXED["issues"]})
        self.assertEqual(post["mascot_mood"], "serious")        # disaster: never cheerful
        self.assertEqual(post["hashtags"], ["#น้ำท่วม", "#จังหวัดทดสอบ"])
        self.assertEqual(post["image_queries"], ["flooded street Thailand", "heavy rain"])
        # the writer sees the facts and the articles; the checker also sees the draft
        self.assertEqual(ai.calls[0][1]["facts"]["cause"], "ฝนตกหนักต่อเนื่อง")
        self.assertEqual(len(ai.calls[0][1]["articles"]), 2)
        self.assertIn("300", ai.calls[1][1]["draft"]["caption"])

    def test_clean_review_keeps_draft(self):
        clean = {"issues": [], "card_topic": "x", "card_headline": "x", "card_detail": "x",
                 "caption": "x"}
        post = NewsEditor(self.cfg, ScriptedAI(write=DRAFT, review=clean)).write(
            self.ev, FACTS, self.now)
        self.assertEqual(post["caption"], DRAFT["caption"])     # nothing flagged: no rewrite
        self.assertEqual(post["review"], {"checked": True, "issues": []})

    def test_failed_review_keeps_draft_and_warns(self):
        ai = ScriptedAI(write=DRAFT, review=RuntimeError("quota"))
        post = NewsEditor(self.cfg, ai).write(self.ev, FACTS, self.now)
        self.assertEqual(post["caption"], DRAFT["caption"])
        self.assertFalse(post["review"]["checked"])
        self.assertIn("quota", post["review"]["error"])

        flex = build_alert_messages(self.ev, {**FACTS, **post}, self.now)[0]
        self.assertIn("ตรวจทานอัตโนมัติไม่สำเร็จ", json.dumps(flex, ensure_ascii=False))

    def test_empty_draft_is_a_failure(self):
        from newsbot.ai import AIError

        with self.assertRaises(AIError):
            NewsEditor(self.cfg, ScriptedAI(write={**DRAFT, "caption": " "})).write(
                self.ev, FACTS, self.now)

    def test_fact_check_can_be_turned_off(self):
        self.cfg.editor_fact_check = False
        ai = ScriptedAI(write=DRAFT)
        post = NewsEditor(self.cfg, ai).write(self.ev, FACTS, self.now)
        self.assertEqual(len(ai.calls), 1)
        self.assertEqual(post["review"], {"checked": False, "issues": []})

    def test_cheerful_mood_kept_for_good_news(self):
        ev = flood_event(self.now)
        ev.category, ev.sensitive = "economy", False
        post = NewsEditor(self.cfg, ScriptedAI(write=DRAFT, review=FIXED)).write(
            ev, {**FACTS, "sensitive": False}, self.now)
        self.assertEqual(post["mascot_mood"], "excited")


class FactsOnlyAnalyst(FakeAnalyst):
    """Like the real analyst now: analyze() returns facts, the editor writes via ask()."""

    def __init__(self, flood_score=85, write=DRAFT, review=FIXED):
        super().__init__(flood_score)
        self.scripted = ScriptedAI(write=write, review=review)

    def analyze(self, event, now):
        self.analyze_calls += 1
        return {**FACTS, "headline": event.headline, "urgency": event.urgency}

    def ask(self, **kw):
        return self.scripted.ask(**kw)


class EditorPipelineTests(unittest.TestCase):
    def setUp(self):
        self.cfg = make_config(Path(tempfile.mkdtemp()))
        self.cfg.digest_hours = []   # a due digest would pick up the story that failed

    def test_alert_carries_the_edited_post(self):
        now = bkk(2026, 10, 21, 14, 5)
        ai, line = FactsOnlyAnalyst(), ConsoleNotifier(echo=False)
        rep = run_once(self.cfg, State(self.cfg.state_path), ai, line, now,
                       fetcher=lambda c: feeds_round1(now))
        self.assertEqual(len(rep.alerts_sent), 1)
        flex, post = line.batches[0]
        self.assertEqual(post["text"].split("\n\n")[0], "น้ำท่วมฉับพลันที่ จ.ทดสอบ")
        self.assertIn("#น้ำท่วม", post["text"])
        text = json.dumps(flex, ensure_ascii=False)
        self.assertIn("แก้แล้ว 1 จุด", text)
        self.assertEqual([c[0] for c in ai.scripted.calls], ["write", "review"])

    def test_editor_failure_retries_next_run(self):
        now = bkk(2026, 10, 21, 14, 5)
        state, line = State(self.cfg.state_path), ConsoleNotifier(echo=False)
        broken = FactsOnlyAnalyst(write=RuntimeError("overloaded"))
        rep = run_once(self.cfg, state, broken, line, now, fetcher=lambda c: feeds_round1(now))
        self.assertTrue(any("เรียบเรียงข่าวไม่สำเร็จ" in e for e in rep.errors))
        self.assertEqual(len(line.batches), 0)

        ai = FactsOnlyAnalyst()
        rep = run_once(self.cfg, state, ai, line, now + timedelta(minutes=30),
                       fetcher=lambda c: feeds_round1(now))
        self.assertEqual(len(rep.alerts_sent), 1)
        self.assertEqual(ai.analyze_calls, 0)          # facts were kept, only the post redone

    def test_no_post_written_while_held_in_quiet_hours(self):
        night = bkk(2026, 10, 21, 2, 0)
        ai = FactsOnlyAnalyst()
        rep = run_once(self.cfg, State(self.cfg.state_path), ai, ConsoleNotifier(echo=False),
                       night, fetcher=lambda c: feeds_round1(night))
        self.assertEqual(rep.held_quiet, 1)
        self.assertEqual(ai.scripted.calls, [])


# ================================================================ picture agent
class FakePexels:
    def __init__(self, photos_by_query: dict, broken_downloads=()):
        self.photos_by_query = photos_by_query
        self.broken = set(broken_downloads)
        self.searches, self.downloads = [], []

    def search(self, query, orientation="square", per_page=8):
        self.searches.append((query, orientation))
        result = self.photos_by_query.get(query, [])
        if isinstance(result, Exception):
            raise result
        return result

    def download(self, url):
        self.downloads.append(url)
        if url in self.broken:
            raise RuntimeError(f"404 {url}")
        return jpeg((400, 300) if "thumb" in url else (1600, 1200))


def photo(n: int, side=2000) -> Photo:
    return Photo(id=str(n), page_url=f"https://www.pexels.com/photo/{n}/",
                 photographer=f"ช่างภาพ {n}", thumb_url=f"https://img/{n}/thumb",
                 image_url=f"https://img/{n}/full", width=side, height=side, alt=f"photo {n}")


class ArtDirectorTests(unittest.TestCase):
    def setUp(self):
        self.now = bkk(2026, 10, 21, 14, 5)
        self.cfg = make_config(Path(tempfile.mkdtemp()))
        self.cfg.card_background = "photo"
        self.ev = flood_event(self.now)
        self.post = {"card_headline": "น้ำท่วมฉับพลัน", "mascot_mood": "serious",
                     "where": "อ.เมือง จ.ทดสอบ",
                     "image_queries": ["flooded street Thailand", "heavy rain"]}

    def test_ai_ranks_and_best_photo_is_used(self):
        pex = FakePexels({"flooded street Thailand": [photo(1), photo(2, side=640)],
                          "heavy rain": [photo(3), photo(1)]})
        ai = ScriptedAI(pick={"suitable": [1, 0], "reason": "ภาพถนนน้ำท่วม ไม่มีหน้าคน"})
        bg, problem = ArtDirector(self.cfg, ai, pex).find(self.ev, self.post)

        self.assertEqual(problem, "")
        self.assertEqual(bg.credit, "ภาพประกอบ: ช่างภาพ 3 / Pexels")   # candidates: 1, 3
        self.assertEqual(bg.link, "https://www.pexels.com/photo/3/")
        self.assertEqual(bg.image.size, (1600, 1200))
        name, payload, images, model = ai.calls[0]
        self.assertEqual(model, self.cfg.review_model)
        self.assertEqual(len(images), 2)                       # 640 px photo was skipped
        self.assertTrue(payload["sad_or_sensitive_story"])
        self.assertEqual(payload["story_location"], "อ.เมือง จ.ทดสอบ")   # for the landmark rule
        self.assertEqual([c["description"] for c in payload["candidates"]],
                         ["photo 1", "photo 3"])
        self.assertEqual(pex.searches[0], ("flooded street Thailand", "square"))

    def test_ai_rejecting_all_photos_gives_plain_card_quietly(self):
        pex = FakePexels({"flooded street Thailand": [photo(1)]})
        ai = ScriptedAI(pick={"suitable": [], "reason": "มีแต่ภาพเด็กยิ้ม"})
        self.assertEqual(ArtDirector(self.cfg, ai, pex).find(self.ev, self.post), (None, ""))
        self.assertNotIn("https://img/1/full", pex.downloads)

    def test_broken_thumbnail_keeps_numbers_aligned(self):
        pex = FakePexels({"flooded street Thailand": [photo(1), photo(2), photo(3)]},
                         broken_downloads={"https://img/2/thumb"})
        ai = ScriptedAI(pick={"suitable": [1], "reason": ""})   # 2nd image shown = photo 3
        bg, _ = ArtDirector(self.cfg, ai, pex).find(self.ev, self.post)
        self.assertEqual(bg.credit, "ภาพประกอบ: ช่างภาพ 3 / Pexels")

    def test_busy_review_model_falls_back_to_analysis_model(self):
        cfg = self.cfg

        class BusyThenOk:
            def __init__(self):
                self.models = []

            def ask(self, *, model=None, **kw):
                self.models.append(model)
                if model == cfg.review_model:
                    raise RuntimeError("503 UNAVAILABLE high demand")
                return {"suitable": [0], "reason": "ok"}

        pex = FakePexels({"flooded street Thailand": [photo(1)]})
        ai = BusyThenOk()
        bg, problem = ArtDirector(self.cfg, ai, pex).find(self.ev, self.post)
        self.assertEqual((bg.credit, problem), ("ภาพประกอบ: ช่างภาพ 1 / Pexels", ""))
        self.assertEqual(ai.models, [cfg.review_model, cfg.analysis_model])

    def test_without_ai_first_photo_that_downloads(self):
        pex = FakePexels({"flooded street Thailand": [photo(1), photo(2)]},
                         broken_downloads={"https://img/1/full"})
        bg, _ = ArtDirector(self.cfg, None, pex).find(self.ev, self.post)
        self.assertEqual(bg.credit, "ภาพประกอบ: ช่างภาพ 2 / Pexels")

    def test_problems_are_reported_not_raised(self):
        self.assertIn("PIXABAY_API_KEY", ArtDirector(self.cfg).find(self.ev, self.post)[1])

        pex = FakePexels({"flooded street Thailand": RuntimeError("Pexels HTTP 429"),
                          "heavy rain": RuntimeError("Pexels HTTP 429")})
        bg, problem = ArtDirector(self.cfg, None, pex).find(self.ev, self.post)
        self.assertIsNone(bg)
        self.assertIn("429", problem)

        pex = FakePexels({"flooded street Thailand": [photo(1)]})
        ai = ScriptedAI(pick=RuntimeError("model overloaded"))
        bg, problem = ArtDirector(self.cfg, ai, pex).find(self.ev, self.post)
        self.assertIsNone(bg)
        self.assertIn("overloaded", problem)

        self.cfg.card_background = "plain"
        self.assertEqual(ArtDirector(self.cfg, None, pex).find(self.ev, self.post), (None, ""))

    def test_category_keywords_when_editor_gave_none(self):
        pex = FakePexels({})
        ArtDirector(self.cfg, None, pex).find(self.ev, {})
        self.assertEqual(pex.searches, [("storm clouds rain", "square")])

    def test_portrait_card_searches_portrait_photos(self):
        self.cfg.card_size = (1080, 1350)
        pex = FakePexels({})
        ArtDirector(self.cfg, None, pex).find(self.ev, self.post)
        self.assertEqual(pex.searches[0][1], "portrait")

    def test_ai_painting(self):
        class Interactions:
            def __init__(self):
                self.calls = []

            def create(self, **kw):
                self.calls.append(kw)
                img = type("Img", (), {"data": base64.b64encode(jpeg((1024, 1024))).decode()})
                return type("Result", (), {"output_image": img})

        client = type("Client", (), {"interactions": Interactions()})()
        self.cfg.card_background = "ai"
        painter = GeminiPainter("gemini-nano-banana-2.1", client=client)
        bg, problem = ArtDirector(self.cfg, None, None, painter).find(
            self.ev, {**self.post, "image_prompt": "A flooded village road"})
        self.assertEqual(problem, "")
        self.assertEqual(bg.credit, "ภาพประกอบสร้างด้วย AI")
        call = client.interactions.calls[0]
        self.assertEqual(call["model"], "gemini-nano-banana-2.1")
        self.assertEqual(call["response_format"]["aspect_ratio"], "1:1")
        self.assertIn("A flooded village road", call["input"])
        self.assertIn("No text", call["input"])
        self.assertIn("somber", call["input"])                 # serious story


class PixabayClientTests(unittest.TestCase):
    HIT = {"id": 42, "pageURL": "https://pixabay.com/photos/flood-42/", "tags": "flood, street",
           "webformatURL": "https://pixabay.com/get/abc_640.jpg",
           "largeImageURL": "https://pixabay.com/get/abc_1280.jpg",
           "imageWidth": 4000, "imageHeight": 3000, "user": "somchai"}

    def _session(self, calls, status=200, error=None):
        hit = self.HIT

        class Resp:
            status_code, text, content = status, "[ERROR 400] Invalid or missing API key", b"jpg"

            def json(self):
                return {"total": 1, "totalHits": 1, "hits": [hit, {"id": 43}]}

        class Session:
            def get(self, url, timeout=None, params=None):
                calls.append((url, params))
                if error:
                    raise error
                return Resp()

        return Session()

    def test_search_request_and_parse(self):
        from newsbot.art import PixabayClient

        calls = []
        photos = PixabayClient("SECRET", session=self._session(calls)).search(
            "flooded street", "portrait")
        url, params = calls[0]
        self.assertEqual(url, "https://pixabay.com/api/")
        self.assertEqual(params["key"], "SECRET")
        self.assertEqual(params["q"], "flooded street")
        self.assertEqual(params["orientation"], "vertical")
        self.assertEqual(params["image_type"], "photo")
        self.assertEqual(params["safesearch"], "true")
        self.assertEqual(len(photos), 1)                         # hit without URLs skipped
        p = photos[0]
        self.assertEqual((p.provider, p.photographer, p.width), ("Pixabay", "somchai", 4000))
        self.assertEqual(p.image_url, "https://pixabay.com/get/abc_1280.jpg")
        self.assertEqual(p.thumb_url, "https://pixabay.com/get/abc_340.jpg")

    def test_results_cached_for_24_hours(self):
        from newsbot.art import PixabayClient, SearchCache

        now = [1_000_000.0]
        path = Path(tempfile.mkdtemp()) / "photo-cache.json"
        calls = []
        client = PixabayClient("K", session=self._session(calls),
                               cache=SearchCache(path, clock=lambda: now[0]))
        client.search("Flood", "square")
        client.search("flood", "square")
        self.assertEqual(len(calls), 1)
        # a new run reads the saved cache
        again = PixabayClient("K", session=self._session(calls),
                              cache=SearchCache(path, clock=lambda: now[0]))
        self.assertEqual(again.search("flood", "square")[0].id, "pixabay-42")
        self.assertEqual(len(calls), 1)
        now[0] += 24 * 3600 + 1
        again.search("flood", "square")
        self.assertEqual(len(calls), 2)

    def test_errors_never_show_the_key(self):
        import requests

        from newsbot.art import ArtError, PixabayClient

        err = requests.ConnectionError("Max retries exceeded with url: /api/?key=SECRET&q=x")
        with self.assertRaises(ArtError) as ctx:
            PixabayClient("SECRET", session=self._session([], error=err)).search("x")
        self.assertNotIn("SECRET", str(ctx.exception))
        self.assertIsNone(ctx.exception.__cause__)
        self.assertTrue(ctx.exception.__suppress_context__)

        with self.assertRaises(ArtError) as ctx:
            PixabayClient("SECRET", session=self._session([], status=400)).search("x")
        self.assertIn("400", str(ctx.exception))
        self.assertNotIn("SECRET", str(ctx.exception))

    def test_sources_from_environment(self):
        from unittest import mock

        from newsbot.art import PexelsClient, PixabayClient

        cfg = make_config(Path(tempfile.mkdtemp()))
        cfg.card_background = "photo"
        env = {"PIXABAY_API_KEY": "a", "PEXELS_API_KEY": "b"}
        with mock.patch.dict("os.environ", env):
            art = ArtDirector.from_env(cfg)
        self.assertEqual([type(s) for s in art.sources], [PixabayClient, PexelsClient])
        self.assertEqual(art.sources[0].cache.path, cfg.state_path.parent / "photo-cache.json")
        with mock.patch.dict("os.environ", {"PIXABAY_API_KEY": "", "PEXELS_API_KEY": ""}):
            self.assertEqual(ArtDirector.from_env(cfg).sources, [])

    def test_second_source_used_when_first_fails(self):
        now = bkk(2026, 10, 21, 14, 5)
        cfg = make_config(Path(tempfile.mkdtemp()))
        cfg.card_background = "photo"
        down = FakePexels({"flood": RuntimeError("Pixabay HTTP 429")})
        p = photo(5)
        p.provider = "Pixabay"
        backup = FakePexels({"flood": [p]})
        bg, problem = ArtDirector(cfg, None, [down, backup]).find(
            flood_event(now), {"image_queries": ["flood", "rain"]})
        self.assertEqual(bg.credit, "ภาพประกอบ: ช่างภาพ 5 / Pixabay")
        self.assertEqual(len(down.searches), 1)                 # gave up on the broken site


class PexelsClientTests(unittest.TestCase):
    def test_search_request_and_parse(self):
        calls = []

        class Resp:
            status_code, text, content = 200, "", b"jpeg"

            def json(self):
                return {"photos": [
                    {"id": 7, "width": 3000, "height": 2000, "url": "https://www.pexels.com/photo/7/",
                     "photographer": "Somchai", "alt": "Flooded road",
                     "src": {"original": "https://images.pexels.com/photos/7/p.jpeg",
                             "medium": "https://images.pexels.com/photos/7/p.jpeg?h=350"}},
                    {"id": 8, "width": 2000, "height": 3000, "src": {
                        "original": "https://images.pexels.com/photos/8/p.jpeg"}},
                    {"id": 9, "src": {}},                                   # unusable
                ]}

        class Session:
            def get(self, url, timeout=None, headers=None, params=None):
                calls.append((url, headers, params))
                return Resp()

        photos = PexelsClient("KEY", session=Session()).search("flood", "square")
        url, headers, params = calls[0]
        self.assertEqual(url, "https://api.pexels.com/v1/search")
        self.assertEqual(headers, {"Authorization": "KEY"})
        self.assertEqual(params["query"], "flood")
        self.assertEqual(params["orientation"], "square")
        self.assertEqual([p.id for p in photos], ["7", "8"])
        self.assertTrue(photos[0].image_url.endswith("&h=1400"))   # landscape: fix the height
        self.assertTrue(photos[1].image_url.endswith("&w=1400"))   # portrait: fix the width
        self.assertEqual(photos[1].photographer, "Pexels")

        PexelsClient("KEY", session=Session()).download("https://images.pexels.com/x.jpeg")
        self.assertIsNone(calls[-1][1])                           # key never sent to the CDN


# ================================================================ photo card + pipeline
class FakeArt:
    def __init__(self, background=None, problem=""):
        self.result = (background, problem)
        self.calls = 0

    def find(self, event, post):
        self.calls += 1
        return self.result


class PhotoCardTests(unittest.TestCase):
    def test_renders_all_sizes_and_lengths(self):
        from newsbot.card import CardText, card_bytes, render_photo_card

        bg = Image.open(io.BytesIO(jpeg((1600, 900))))
        for size in ((1080, 1080), (1080, 1350)):
            for head in ("ข่าวดี", "สมาคมโรงแรมชี้ นักท่องเที่ยวภูเก็ต-กระบี่ชะลอตัวก่อนไฮซีซั่น "
                                   "เพราะเที่ยวบินตรงลดลงมากกว่าที่คาด ผู้ประกอบการกังวลรายได้ทั้งปี"):
                img = render_photo_card(CardText("เศรษฐกิจ", head, "", "21 ต.ค. 2569 · ที่มา: A",
                                                 "serious", True, "ภาพประกอบ: X / Pexels"),
                                        bg, size)
                self.assertEqual(img.size, size)
                full, prev = card_bytes(img)
                self.assertTrue(full.startswith(b"\x89PNG"))
                self.assertLess(len(full), 10_000_000)

    def test_cover_crop_fills_the_card(self):
        from newsbot.card import cover

        for src in ((500, 2000), (3000, 400), (1080, 1080)):
            self.assertEqual(cover(Image.new("RGB", src), 1080, 1350).size, (1080, 1350))

    def test_no_stray_short_line_in_headline(self):
        from newsbot.card import _photo_headline

        f, size, lines = _photo_headline("อุเทนถวาย ยื่น ตร. ขอเจรจาสงบศึก", 578)
        self.assertNotIn("ตร.", lines)
        self.assertLessEqual(len(lines), 3)
        f, size, lines = _photo_headline("ครม. เคาะหมื่นล้าน ช่วยน้ำท่วม", 578)
        self.assertNotIn("ครม.", lines)

    def test_photo_card_used_and_credited(self):
        now = bkk(2026, 10, 21, 14, 5)
        cfg = make_config(Path(tempfile.mkdtemp()))
        host, line = FakeImageHost(), ConsoleNotifier(echo=False)
        bg = Background(Image.open(io.BytesIO(jpeg((1600, 1200), (200, 60, 60)))),
                        "ภาพประกอบ: Somchai / Pexels", "https://www.pexels.com/photo/7/")
        art = FakeArt(bg)
        rep = run_once(cfg, State(cfg.state_path), FakeAnalyst(), line, now,
                       fetcher=lambda c: feeds_round1(now), images=host, art=art)
        self.assertEqual(art.calls, 1)
        self.assertEqual(rep.cards_made, 1)
        flex, image, post = line.batches[0]
        self.assertIn("ภาพประกอบ: Somchai / Pexels", post["text"])
        self.assertIn("Somchai / Pexels", json.dumps(flex, ensure_ascii=False))
        png = next(v for k, v in host.files.items() if k.endswith(".png"))
        card = Image.open(io.BytesIO(png)).convert("RGB")
        self.assertGreater(card.getpixel((540, 300))[0], card.getpixel((540, 300))[2])  # red photo

    def test_problem_becomes_warning_and_plain_card(self):
        now = bkk(2026, 10, 21, 14, 5)
        cfg = make_config(Path(tempfile.mkdtemp()))
        report = RunReport()
        ev = flood_event(now)
        image = make_post_image(cfg, FakeImageHost(), ev, {**FACTS, **DRAFT}, now, True, report,
                                art=FakeArt(None, "ยังไม่ได้ตั้งค่า PEXELS_API_KEY"))
        self.assertIsInstance(image, PostImage)
        self.assertEqual(image.credit, "")
        self.assertEqual(report.warnings, ["ยังไม่ได้ตั้งค่า PEXELS_API_KEY"])

    def test_credit_only_when_image_was_sent(self):
        now = bkk(2026, 10, 21, 14, 5)
        ev = flood_event(now)
        analysis = {**FACTS, **DRAFT}
        local = PostImage("", "", "ภาพประกอบ: X / Pexels")        # saved locally, no URL
        msgs = build_alert_messages(ev, analysis, now, image=None)
        self.assertNotIn("Pexels", msgs[-1]["text"])
        self.assertEqual(len(build_alert_messages(ev, analysis, now, image=local)), 2)


# ================================================================ multimodal requests
class ImageRequestTests(unittest.TestCase):
    def test_gemini_sends_images_before_text(self):
        from google import genai
        from google.genai import types

        from newsbot.ai import GeminiAnalyst

        seen = []

        def handler(request):
            seen.append(json.loads(request.content))
            return httpx.Response(200, json={
                "candidates": [{"content": {"role": "model", "parts": [
                    {"text": json.dumps({"suitable": [0], "reason": "ok"})}]},
                    "finishReason": "STOP"}]})

        client = genai.Client(api_key="k", http_options=types.HttpOptions(
            httpx_client=httpx.Client(transport=httpx.MockTransport(handler))))
        analyst = GeminiAnalyst(make_config(Path(tempfile.mkdtemp())), client=client)
        out = analyst.ask(system="s", payload={"headline": "น้ำท่วม"}, schema=PICK_SCHEMA,
                          images=[b"\xff\xd8one", b"\xff\xd8two"], effort="low")
        self.assertEqual(out["suitable"], [0])
        parts = seen[0]["contents"][0]["parts"]
        inline = [p["inlineData"] for p in parts[:2]]
        self.assertEqual([d.get("mimeType", d.get("mime_type")) for d in inline],
                         ["image/jpeg"] * 2)
        self.assertEqual(base64.urlsafe_b64decode(inline[1]["data"]), b"\xff\xd8two")
        self.assertIn("น้ำท่วม", parts[2]["text"])

    def test_claude_sends_images_before_text(self):
        from newsbot.ai import ClaudeAnalyst

        seen = []

        def handler(request):
            seen.append(json.loads(request.content))
            return httpx2.Response(200, json={
                "id": "m", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5",
                "content": [{"type": "text", "text": json.dumps({"suitable": [], "reason": ""})}],
                "stop_reason": "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1}})

        client = anthropic.Anthropic(api_key="k", max_retries=0, http_client=httpx2.Client(
            transport=httpx2.MockTransport(handler)))
        analyst = ClaudeAnalyst(claude_config(), client=client)
        analyst.ask(system="s", payload={"headline": "น้ำท่วม"}, schema=PICK_SCHEMA,
                    images=[b"\xff\xd8one"])
        body = seen[0]
        content = body["messages"][0]["content"]
        self.assertEqual(content[0]["type"], "image")
        self.assertEqual(content[0]["source"]["media_type"], "image/jpeg")
        self.assertEqual(content[1]["type"], "text")
        self.assertEqual(body["model"], "claude-sonnet-5-5")    # ask() defaults to analysis model


class CloudflarePainterTests(unittest.TestCase):
    def _session(self, calls, status=200, body=None, error=None):
        image = base64.b64encode(jpeg((1024, 1024))).decode()
        body = body if body is not None else {"result": {"image": image}, "success": True,
                                              "errors": [], "messages": []}

        class Resp:
            status_code, text = status, json.dumps(body)

            def json(self):
                return body

        class Session:
            def post(self, url, headers=None, timeout=None, json=None, files=None):
                calls.append({"url": url, "headers": headers, "json": json, "files": files})
                if error:
                    raise error
                return Resp()

        return Session()

    def test_flux1_request_and_image(self):
        from newsbot.art import CloudflarePainter

        calls = []
        painter = CloudflarePainter("ACC", "TOKEN", "@cf/black-forest-labs/flux-1-schnell",
                                    session=self._session(calls))
        data = painter.paint("a flooded Thai village road", "1:1")
        self.assertEqual(Image.open(io.BytesIO(data)).size, (1024, 1024))
        call = calls[0]
        self.assertEqual(call["url"], "https://api.cloudflare.com/client/v4/accounts/ACC/ai/run/"
                                      "@cf/black-forest-labs/flux-1-schnell")
        self.assertEqual(call["headers"], {"Authorization": "Bearer TOKEN"})
        self.assertEqual(call["json"]["prompt"], "a flooded Thai village road")
        self.assertLessEqual(call["json"]["steps"], 8)            # the model's maximum
        self.assertEqual(set(call["json"]), {"prompt", "steps"})  # the API rejects "seed"
        self.assertIsNone(call["files"])

    def test_flux2_sends_multipart_with_size(self):
        from newsbot.art import CloudflarePainter

        calls = []
        CloudflarePainter("ACC", "T", "@cf/black-forest-labs/flux-2-klein-4b",
                          session=self._session(calls)).paint("x", "4:5")
        files = calls[0]["files"]
        self.assertEqual((files["width"][1], files["height"][1]), ("1024", "1280"))
        self.assertIsNone(calls[0]["json"])

    def test_errors_are_readable_and_never_show_the_token(self):
        import requests

        from newsbot.art import ArtError, CloudflarePainter

        quota = {"result": None, "success": False, "errors": [
            {"code": 4006, "message": "you have used up your daily free allocation of 10,000 "
                                      "neurons"}]}
        painter = CloudflarePainter("ACC", "TOKEN", "m", session=self._session([], 429, quota))
        with self.assertRaises(ArtError) as ctx:
            painter.paint("x", "1:1")
        self.assertIn("daily free allocation", str(ctx.exception))

        err = requests.ConnectionError("Authorization: Bearer TOKEN failed")
        painter = CloudflarePainter("ACC", "TOKEN", "m", session=self._session([], error=err))
        with self.assertRaises(ArtError) as ctx:
            painter.paint("x", "1:1")
        self.assertNotIn("TOKEN", str(ctx.exception))

    def test_painter_from_environment(self):
        from unittest import mock

        from newsbot.art import CloudflarePainter

        cfg = make_config(Path(tempfile.mkdtemp()))
        cfg.card_background = "ai"
        env = {"CLOUDFLARE_ACCOUNT_ID": "acc", "CLOUDFLARE_API_TOKEN": "tok",
               "PIXABAY_API_KEY": "px"}
        with mock.patch.dict("os.environ", env):
            art = ArtDirector.from_env(cfg)
        self.assertIsInstance(art.painter, CloudflarePainter)
        self.assertEqual(art.painter.model, "@cf/black-forest-labs/flux-2-klein-4b")
        self.assertEqual(len(art.sources), 1)                     # Pixabay kept as fallback
        with mock.patch.dict("os.environ", {"CLOUDFLARE_ACCOUNT_ID": "", "PIXABAY_API_KEY": ""}):
            self.assertIsNone(ArtDirector.from_env(cfg).painter)


class FakePainter:
    def __init__(self, error=None):
        self.error = error
        self.prompts = []
        self.model = "fake"

    def paint(self, prompt, aspect_ratio):
        self.prompts.append((prompt, aspect_ratio))
        if self.error:
            raise self.error
        return jpeg((1024, 1024), (30, 120, 90))


class AIPaintingTests(unittest.TestCase):
    def setUp(self):
        now = bkk(2026, 10, 21, 14, 5)
        self.cfg = make_config(Path(tempfile.mkdtemp()))
        self.cfg.card_background = "ai"
        self.ev = flood_event(now)
        self.post = {"card_headline": "น้ำท่วมฉับพลัน", "mascot_mood": "serious",
                     "where": "อ.เมือง จ.ทดสอบ", "image_prompt": "A flooded Thai village road",
                     "image_queries": ["flooded street Thailand"]}

    def test_painting_checked_and_used(self):
        ai = ScriptedAI(check={"ok": True, "reason": ""})
        bg, problem = ArtDirector(self.cfg, ai, None, FakePainter()).find(self.ev, self.post)
        self.assertEqual((bg.credit, problem), ("ภาพประกอบสร้างด้วย AI", ""))
        name, payload, images, model = ai.calls[0]
        self.assertEqual((name, model, len(images)), ("check", self.cfg.review_model, 1))
        self.assertEqual(payload["story_location"], "อ.เมือง จ.ทดสอบ")

    def test_rejected_twice_falls_back_to_photo(self):
        painter = FakePainter()
        ai = ScriptedAI(check={"ok": False, "reason": "มีตัวอักษรมั่ว"},
                        pick={"suitable": [0], "reason": ""})
        pex = FakePexels({"flooded street Thailand": [photo(1)]})
        bg, problem = ArtDirector(self.cfg, ai, pex, painter).find(self.ev, self.post)
        self.assertEqual(len(painter.prompts), 2)                 # one retry with a new seed
        self.assertEqual(bg.credit, "ภาพประกอบ: ช่างภาพ 1 / Pexels")
        self.assertIn("ไม่ผ่านการตรวจ", problem)
        self.assertIn("ตัวอักษรมั่ว", problem)

    def test_painter_error_or_missing_falls_back(self):
        pex = FakePexels({"flooded street Thailand": [photo(1)]})
        bg, problem = ArtDirector(self.cfg, None, pex,
                                  FakePainter(RuntimeError("Cloudflare HTTP 500"))).find(
            self.ev, self.post)
        self.assertEqual(bg.credit, "ภาพประกอบ: ช่างภาพ 1 / Pexels")
        self.assertIn("Cloudflare HTTP 500", problem)

        bg, problem = ArtDirector(self.cfg, None, pex, None).find(self.ev, self.post)
        self.assertIn("CLOUDFLARE_ACCOUNT_ID", problem)
        self.assertIsNotNone(bg)

        bg, problem = ArtDirector(self.cfg, None, None, None).find(self.ev, self.post)
        self.assertIsNone(bg)
        self.assertIn("พื้นสีเรียบ", problem)

    def test_check_failure_still_uses_painting(self):
        ai = ScriptedAI(check=RuntimeError("503 UNAVAILABLE"))
        bg, problem = ArtDirector(self.cfg, ai, None, FakePainter()).find(self.ev, self.post)
        self.assertEqual((bg.credit, problem), ("ภาพประกอบสร้างด้วย AI", ""))

    def test_prompt_carries_style_and_mood(self):
        painter = FakePainter()
        ArtDirector(self.cfg, None, None, painter).find(self.ev, self.post)
        prompt, ratio = painter.prompts[0]
        self.assertTrue(prompt.startswith("Semi-realistic digital painting"))   # style first
        self.assertIn("not a photograph", prompt)
        self.assertIn("A flooded Thai village road", prompt)
        self.assertIn("signboards", prompt)
        self.assertIn("No text", prompt)
        self.assertIn("somber", prompt)
        self.assertEqual(ratio, "1:1")

    def test_styles(self):
        from newsbot.art import IMAGE_STYLES, paint_prompt
        from newsbot.config import IMAGE_STYLES as CONFIG_STYLES

        self.assertEqual(IMAGE_STYLES, CONFIG_STYLES)
        self.assertTrue(paint_prompt("x", "flat").startswith("Flat vector"))
        realistic = paint_prompt("x", "realistic")
        self.assertIn("documentary news photograph", realistic)
        self.assertNotIn("not a photograph", realistic)
        self.assertIn("No real or identifiable people", realistic)
        self.assertNotIn("somber", paint_prompt("x", "painting", somber=False))

        self.cfg.card_image_style = "flat"
        painter = FakePainter()
        ArtDirector(self.cfg, None, None, painter).find(self.ev, self.post)
        self.assertTrue(painter.prompts[0][0].startswith("Flat vector"))

    def test_check_follows_style(self):
        from newsbot.art import check_system

        self.assertIn("ภาพถ่ายข่าวจริง", check_system("painting"))
        self.assertNotIn("ภาพถ่ายข่าวจริง", check_system("realistic"))
        self.assertIn("บุคคลจริง", check_system("realistic"))
        self.assertNotIn("{look_rule}", check_system("flat"))

        self.cfg.card_image_style = "realistic"
        seen = []

        class Recorder(ScriptedAI):
            def ask(self, **kw):
                seen.append(kw["system"])
                return super().ask(**kw)

        ArtDirector(self.cfg, Recorder(check={"ok": True, "reason": ""}), None,
                    FakePainter()).find(self.ev, self.post)
        self.assertNotIn("ภาพถ่ายข่าวจริง", seen[0])

    def test_paint_test_makes_every_card_and_survives_errors(self):
        from newsbot.art import IMAGE_STYLES
        from newsbot.pipeline import PAINT_SAMPLES, run_paint_test

        host = FakeImageHost()
        painters = {"@cf/a/good": FakePainter(), "@cf/a/broken": FakePainter(RuntimeError("400"))}
        lines = run_paint_test(self.cfg, painters, host, bkk(2026, 10, 23, 18, 0), IMAGE_STYLES)
        per_model = len(PAINT_SAMPLES) * len(IMAGE_STYLES)
        self.assertEqual(len(lines), 2 * per_model)
        self.assertEqual(len(host.files), per_model)               # only the good painter
        self.assertTrue(all(k.startswith("cards/paint-test/20261023-1800/") for k in host.files))
        self.assertEqual(sum("วาดไม่สำเร็จ" in line for line in lines), per_model)


class QuotaAndTimeTests(unittest.TestCase):
    DAILY = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                       "message": "You exceeded your current quota",
                       "details": [{"violations": [{
                           "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                           "quotaValue": "20"}]}]}}

    def test_daily_quota_stops_without_retrying(self):
        from google import genai
        from google.genai import types

        from newsbot.ai import GeminiAnalyst, QuotaExhausted

        calls = []

        def handler(request):
            calls.append(str(request.url))
            return httpx.Response(429, json=self.DAILY)

        client = genai.Client(api_key="k", http_options=types.HttpOptions(
            httpx_client=httpx.Client(transport=httpx.MockTransport(handler))))
        analyst = GeminiAnalyst(make_config(Path(tempfile.mkdtemp())), client=client,
                                retry_waits=(0, 0))
        with self.assertRaises(QuotaExhausted):
            analyst.ask(system="s", payload={}, schema=PICK_SCHEMA)
        # one call per model, no pointless retries: the analysis model, then its backup
        self.assertEqual(len(calls), 2)
        self.assertIn("gemini-3.8-flash", calls[0])
        self.assertIn("gemini-3.5-flash-lite", calls[1])
        with self.assertRaises(QuotaExhausted):
            analyst.ask(system="s", payload={}, schema=PICK_SCHEMA)
        self.assertEqual(len(calls), 2)                    # the rest of the run skips the API

    def _gemini(self, handler):
        from google import genai
        from google.genai import types

        from newsbot.ai import GeminiAnalyst

        client = genai.Client(api_key="k", http_options=types.HttpOptions(
            httpx_client=httpx.Client(transport=httpx.MockTransport(handler))))
        return GeminiAnalyst(make_config(Path(tempfile.mkdtemp())), client=client,
                             retry_waits=(0,))

    def test_busy_analysis_model_uses_backup(self):
        from newsbot.ai import ANALYSIS_SCHEMA

        urls = []
        facts = {k: FACTS[k] for k in ANALYSIS_SCHEMA["properties"]}

        def handler(request):
            urls.append(str(request.url))
            if "gemini-3.8-flash" in str(request.url):
                return httpx.Response(503, json={"error": {"code": 503, "status": "UNAVAILABLE",
                                                           "message": "high demand"}})
            return httpx.Response(200, json={"candidates": [{"content": {"role": "model",
                "parts": [{"text": json.dumps(facts, ensure_ascii=False)}]},
                "finishReason": "STOP"}]})

        now = bkk(2026, 10, 21, 14, 5)
        analyst = self._gemini(handler)
        out = analyst.analyze(flood_event(now), now)
        self.assertEqual(out["_backup_model"], "gemini-3.5-flash-lite")
        self.assertEqual(sum("3.8-flash" in u for u in urls), 2)   # first try + one retry
        self.assertIn("flash-lite", urls[-1])

        flex = build_alert_messages(flood_event(now), {**out, **DRAFT}, now)[0]
        self.assertIn("โมเดลสำรอง", json.dumps(flex, ensure_ascii=False))

    def test_backup_failing_too_raises(self):
        from google.genai import errors

        def handler(request):
            return httpx.Response(503, json={"error": {"code": 503, "status": "UNAVAILABLE",
                                                       "message": "high demand"}})

        now = bkk(2026, 10, 21, 14, 5)
        with self.assertRaises(errors.ServerError):
            self._gemini(handler).analyze(flood_event(now), now)

    def test_editor_marks_backup_writer(self):
        class BackupAI(ScriptedAI):
            def ask(self, **kw):
                out = super().ask(**kw)
                self.last_fallback = kw["schema"] is POST_SCHEMA
                return out

        now = bkk(2026, 10, 21, 14, 5)
        cfg = make_config(Path(tempfile.mkdtemp()))
        post = NewsEditor(cfg, BackupAI(write=DRAFT, review=FIXED)).write(
            flood_event(now), FACTS, now)
        self.assertEqual(post["_backup_writer"], cfg.review_model)

    def test_default_client_has_a_timeout(self):
        from unittest import mock

        from newsbot.ai import CALL_TIMEOUT_MS, GeminiAnalyst

        with mock.patch.dict("os.environ", {"GEMINI_API_KEY": "k"}):
            analyst = GeminiAnalyst(make_config(Path(tempfile.mkdtemp())))
        self.assertEqual(analyst.client._api_client._http_options.timeout, CALL_TIMEOUT_MS)

    def test_quota_gone_keeps_story_for_next_run(self):
        from newsbot.ai import QuotaExhausted

        class OutOfQuota(FakeAnalyst):
            def analyze(self, event, now):
                self.analyze_calls += 1
                raise QuotaExhausted("โควตาฟรีรายวันของ gemini-3.8-flash หมดแล้ว")

        now = bkk(2026, 10, 21, 14, 5)
        cfg = make_config(Path(tempfile.mkdtemp()))
        cfg.digest_hours = []
        state, ai = State(cfg.state_path), OutOfQuota()
        rep = run_once(cfg, state, ai, ConsoleNotifier(echo=False), now,
                       fetcher=lambda c: feeds_round1(now))
        self.assertEqual(rep.errors, [])                   # a warning, not a failed run
        self.assertTrue(any("โควตา" in w for w in rep.warnings))
        flood = next(e for e in state.events.values() if "น้ำท่วม" in e.headline)
        self.assertEqual((flood.status, flood.attempts), ("new", 0))

        rep = run_once(cfg, state, FakeAnalyst(), ConsoleNotifier(echo=False),
                       now + timedelta(minutes=30), fetcher=lambda c: feeds_round1(now))
        self.assertEqual(len(rep.alerts_sent), 1)          # sent once the quota is back

    def test_out_of_time_saves_work_for_next_run(self):
        import time

        from newsbot.pipeline import OUT_OF_TIME

        now = bkk(2026, 10, 21, 14, 5)
        cfg = make_config(Path(tempfile.mkdtemp()))
        state, ai = State(cfg.state_path), FakeAnalyst()
        rep = run_once(cfg, state, ai, ConsoleNotifier(echo=False), now,
                       fetcher=lambda c: feeds_round1(now), deadline=time.monotonic() - 1)
        self.assertEqual(ai.triage_calls, 0)
        self.assertIn(OUT_OF_TIME, rep.warnings)
        self.assertEqual(rep.errors, [])

        rep = run_once(cfg, state, ai, ConsoleNotifier(echo=False), now + timedelta(minutes=30),
                       fetcher=lambda c: feeds_round1(now))
        self.assertEqual(rep.new_items, 3)                 # nothing was lost
        self.assertEqual(len(rep.alerts_sent), 1)

    def test_out_of_time_mid_run_stops_before_next_story(self):
        import time

        now = bkk(2026, 10, 21, 14, 5)
        cfg = make_config(Path(tempfile.mkdtemp()))
        state = State(cfg.state_path)
        run_once(cfg, state, FakeAnalyst(), ConsoleNotifier(echo=False), now,
                 fetcher=lambda c: {}, deadline=None)
        ev = flood_event(now)
        ev.urgency = 85
        state.events[ev.id] = ev
        ai, line = FakeAnalyst(), ConsoleNotifier(echo=False)
        from newsbot.pipeline import send_alerts

        report = RunReport()
        send_alerts(cfg, state, ai, line, now, report, deadline=time.monotonic() - 1)
        self.assertEqual((ai.analyze_calls, len(line.batches)), (0, 0))
        self.assertEqual(ev.status, "new")


class ConfigTests(unittest.TestCase):
    def test_background_setting(self):
        from newsbot.config import ROOT, load_config

        cfg = load_config()
        self.assertEqual(cfg.card_background, "ai")
        self.assertEqual((cfg.card_image_provider, cfg.card_image_model), ("cloudflare", ""))
        self.assertEqual(cfg.review_model, "gemini-3.5-flash-lite")
        self.assertNotEqual(cfg.review_model, cfg.analysis_model)
        self.assertTrue(cfg.card_ai_pick and cfg.editor_fact_check)

        text = (ROOT / "config.yaml").read_text(encoding="utf-8")
        tmp = Path(tempfile.mkdtemp()) / "config.yaml"
        tmp.write_text(text.replace("background: ai", "background: Plain")
                       .replace("fact_check: true", "fact_check: false"), encoding="utf-8")
        cfg = load_config(tmp)
        self.assertEqual(cfg.card_background, "plain")
        self.assertFalse(cfg.editor_fact_check)

        tmp.write_text(text.replace("background: ai", "background: video"), encoding="utf-8")
        with self.assertRaises(SystemExit):
            load_config(tmp)
        tmp.write_text(text.replace("image_provider: cloudflare", "image_provider: midjourney"),
                       encoding="utf-8")
        with self.assertRaises(SystemExit):
            load_config(tmp)
        self.assertEqual(load_config().card_image_style, "painting")
        tmp.write_text(text.replace("image_style: painting", "image_style: Realistic"),
                       encoding="utf-8")
        self.assertEqual(load_config(tmp).card_image_style, "realistic")
        tmp.write_text(text.replace("image_style: painting", "image_style: anime"),
                       encoding="utf-8")
        with self.assertRaises(SystemExit):
            load_config(tmp)


if __name__ == "__main__":
    unittest.main()
