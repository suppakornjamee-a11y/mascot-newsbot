"""Command line:

    python -m newsbot run              # one full round (use this in cron / GitHub Actions)
    python -m newsbot run --dry-run    # same, but print messages instead of sending to LINE
    python -m newsbot demo             # analyse the most urgent story of the last 24 h and send it
    python -m newsbot test-line        # send a sample alert to check your LINE setup
    python -m newsbot check-sources    # check every RSS feed in config.yaml
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import traceback
from datetime import timedelta

from .config import ROOT, TZ, load_config, load_dotenv, require_env
from .fetch import Item, fetch_all
from .line import ConsoleNotifier, LineNotifier, build_alert_messages
from .state import State
from .timeutil import utcnow


# GitHub stops the job at 15 minutes (timeout-minutes in the workflow) and the state of a
# stopped run is lost. No new AI work starts after this, leaving room for the last step.
RUN_BUDGET_SECONDS = 10 * 60


def annotate(level: str, message: str) -> None:
    """On GitHub Actions, also show the message on the run's summary page."""
    if os.environ.get("GITHUB_ACTIONS") != "true" or not message:
        return
    text = message.replace("%", "%25").replace("\r", "").replace("\n", "%0A")
    print(f"::{level} title=newsbot::{text}", flush=True)


def _notifier(dry_run: bool):
    if dry_run:
        return ConsoleNotifier()
    return LineNotifier(require_env("LINE_CHANNEL_ACCESS_TOKEN"), require_env("LINE_USER_ID"))


def _images(cfg):
    from .imagehost import make_image_host

    return make_image_host(cfg.card_branch) if cfg.card_enabled else None


def cmd_run(args) -> int:
    from .ai import make_analyst
    from .pipeline import run_once

    cfg = load_config()
    require_env(cfg.api_key_env)
    notifier = _notifier(args.dry_run)
    state = State.load(cfg.state_path)
    analyst = make_analyst(cfg)
    now = utcnow()
    report = run_once(cfg, state, analyst, notifier, now, images=_images(cfg),
                      deadline=time.monotonic() + RUN_BUDGET_SECONDS)
    state.save()
    u = analyst.usage
    summary = (report.summary() + f"\nAI ({cfg.ai_provider}): {u['calls']} ครั้ง, "
               f"input {u['input_tokens']:,} / output {u['output_tokens']:,} tokens")
    print(summary)
    # A broken feed is only a warning; failing to reach the AI or LINE is an error.
    level = "error" if report.errors else "warning" if report.warnings else "notice"
    annotate(level, summary)
    return 1 if report.errors else 0


def cmd_demo(args) -> int:
    from .ai import make_analyst
    from .pipeline import run_demo

    cfg = load_config()
    require_env(cfg.api_key_env)
    notifier = _notifier(args.dry_run)
    analyst = make_analyst(cfg)
    result = run_demo(cfg, analyst, notifier, utcnow(), images=_images(cfg))
    if not result.sent:
        msg = f"ส่งตัวอย่างไม่ได้: {result.message}"
        print(msg)
        annotate("error", msg)
        return 1
    u = analyst.usage
    msg = (f"ส่งตัวอย่างแล้ว: {result.headline}\n"
           f"คะแนน {result.urgency} · แหล่ง: {', '.join(result.sources)} "
           f"· เลือกจากข่าว {result.candidates} รายการ\n"
           f"AI ({cfg.ai_provider}): {u['calls']} ครั้ง")
    if result.image_url:
        msg += f"\nรูปโพสต์: {result.image_url}"
    for w in result.warnings:
        msg += f"\nคำเตือน: {w}"
    print(msg)
    annotate("warning" if result.warnings else "notice", msg)
    return 0


def cmd_test_line(args) -> int:
    from .state import Event

    now = utcnow()
    ev = Event(id="e_sample", headline="ตัวอย่าง: ฝนตกหนักทำน้ำท่วมขังหลายเขตใน กทม.",
               category="disaster", first_seen=now, last_updated=now, urgency=78)
    for name in ("แหล่งข่าว A", "แหล่งข่าว B"):
        ev.add_items([Item(id=name, source=name, title=ev.headline, summary="", content="",
                           link="https://www.tmd.go.th/", published=now - timedelta(minutes=7),
                           image=None)])
    analysis = {
        "headline": "[ข้อความทดสอบ] ฝนตกหนัก น้ำท่วมขังหลายเขตใน กทม.",
        "what_happened": "นี่คือข่าวสมมุติสำหรับทดสอบการเชื่อมต่อ LINE เท่านั้น",
        "cause": "ฝนตกหนักต่อเนื่อง (ตัวอย่าง)",
        "when": "วันนี้ช่วงบ่าย (ตัวอย่าง)",
        "where": "กรุงเทพฯ (ตัวอย่าง)",
        "affected": "ผู้ใช้รถใช้ถนนในเขตที่น้ำท่วมขัง (ตัวอย่าง)",
        "missing_info": ["ระยะเวลาที่น้ำจะลด"],
        "urgency": 78, "sensitive": False, "caution": "",
        "caption": "ปิ๊ง! ถ้าเห็นข้อความนี้ แปลว่าระบบส่งข่าวเข้า LINE ได้แล้ว\n\n"
                   "ข้อความนี้คือโพสต์ที่ AI เรียบเรียงให้ กดค้างเพื่อคัดลอกไปโพสต์ได้เลย",
        "hashtags": ["#ทดสอบระบบ"],
        "card_topic": "ทดสอบระบบ", "card_headline": "ระบบส่งข่าวเข้า LINE ได้แล้ว",
        "card_detail": "นี่คือรูปโพสต์ตัวอย่างจากน้องปิ๊ง ข่าวจริงจะใช้หน้าตาแบบนี้",
        "mascot_mood": "excited",
        "image_queries": ["heavy rain city street", "rain umbrella street", "rain clouds"],
        "image_prompt": "Heavy rain over a Bangkok street at dusk, wet road reflecting lights",
        "review": {"checked": True, "issues": []},
    }
    from .pipeline import RunReport, default_art, make_post_image

    cfg = load_config()
    report = RunReport()
    images = _images(cfg)
    # No AI here: test-line only checks LINE, image hosting and the photo key.
    art = default_art(cfg, None) if images is not None else None
    image = make_post_image(cfg, images, ev, analysis, now, urgent=False, report=report, art=art)
    notifier = _notifier(args.dry_run)
    notifier.push(build_alert_messages(ev, analysis, now, cfg.mascot.name, image=image))
    msg = "ส่งข้อความทดสอบแล้ว" + (f" พร้อมรูป {image.url}" if image else " (ไม่มีรูป)")
    if image and image.credit:
        msg += f"\n{image.credit}"
    for w in report.warnings:
        msg += f"\nคำเตือน: {w}"
    print(msg)
    annotate("warning" if report.warnings else "notice", msg)
    return 0


def cmd_check_sources(args) -> int:
    cfg = load_config()
    results = fetch_all(cfg.sources, cfg.fetch_timeout)
    ok = 0
    for src in cfg.sources:
        if not src.enabled:
            print(f"  ปิดอยู่   {src.name}")
            continue
        res = results.get(src.name)
        if isinstance(res, Exception):
            print(f"  ✗ ใช้ไม่ได้ {src.name}: {str(res)[:120]}")
        else:
            ok += 1
            newest = max((i.published for i in res if i.published), default=None)
            when = newest.astimezone(TZ).strftime("%d/%m %H:%M") if newest else "ไม่ระบุเวลา"
            print(f"  ✓ ใช้ได้   {src.name}: {len(res)} ข่าว ล่าสุด {when} — {res[0].title[:50]}")
    print(f"\nใช้ได้ {ok} แหล่ง")
    return 0 if ok else 1


def main(argv=None) -> int:
    load_dotenv(ROOT / ".env")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("newsbot").setLevel(logging.INFO)   # progress lines in the Actions log
    parser = argparse.ArgumentParser(prog="newsbot")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="รันหนึ่งรอบ")
    p.add_argument("--dry-run", action="store_true", help="แสดงผลแทนการส่ง LINE")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("demo", help="ส่งข่าวที่ด่วนที่สุดใน 24 ชม. เข้า LINE เป็นตัวอย่าง")
    p.add_argument("--dry-run", action="store_true", help="แสดงผลแทนการส่ง LINE")
    p.set_defaults(func=cmd_demo)
    p = sub.add_parser("test-line", help="ส่งข้อความทดสอบเข้า LINE")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_test_line)
    p = sub.add_parser("check-sources", help="ตรวจว่าแหล่งข่าวแต่ละแหล่งใช้ได้ไหม")
    p.set_defaults(func=cmd_check_sources)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except SystemExit as exc:
        if isinstance(exc.code, str):
            annotate("error", exc.code)
        raise
    except Exception as exc:
        traceback.print_exc()
        annotate("error", f"{type(exc).__name__}: {exc}"[:1500])
        return 1


if __name__ == "__main__":
    sys.exit(main())
