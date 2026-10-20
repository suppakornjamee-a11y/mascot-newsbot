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
import sys
from datetime import timedelta

from .config import ROOT, TZ, load_config, load_dotenv, require_env
from .fetch import Item, fetch_all
from .line import ConsoleNotifier, LineNotifier, build_alert_messages
from .state import State
from .timeutil import utcnow


def _notifier(dry_run: bool):
    if dry_run:
        return ConsoleNotifier()
    return LineNotifier(require_env("LINE_CHANNEL_ACCESS_TOKEN"), require_env("LINE_USER_ID"))


def cmd_run(args) -> int:
    from .ai import make_analyst
    from .pipeline import run_once

    cfg = load_config()
    require_env(cfg.api_key_env)
    notifier = _notifier(args.dry_run)
    state = State.load(cfg.state_path)
    analyst = make_analyst(cfg)
    now = utcnow()
    report = run_once(cfg, state, analyst, notifier, now)
    state.save()
    print(report.summary())
    u = analyst.usage
    print(f"AI ({cfg.ai_provider}): {u['calls']} ครั้ง, "
          f"input {u['input_tokens']:,} / output {u['output_tokens']:,} tokens")
    # A broken feed is only a warning; failing to reach the AI or LINE is an error.
    return 1 if report.errors else 0


def cmd_demo(args) -> int:
    from .ai import make_analyst
    from .pipeline import run_demo

    cfg = load_config()
    require_env(cfg.api_key_env)
    notifier = _notifier(args.dry_run)
    analyst = make_analyst(cfg)
    result = run_demo(cfg, analyst, notifier, utcnow())
    if not result.sent:
        print(f"ส่งตัวอย่างไม่ได้: {result.message}")
        return 1
    print(f"ส่งตัวอย่างแล้ว: {result.headline}")
    print(f"คะแนน {result.urgency} · แหล่ง: {', '.join(result.sources)} "
          f"· เลือกจากข่าว {result.candidates} รายการ")
    u = analyst.usage
    print(f"AI ({cfg.ai_provider}): {u['calls']} ครั้ง")
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
        "caption": "ถ้าเห็นข้อความนี้ แปลว่าระบบส่งข่าวเข้า LINE ได้แล้ว 🎉\n"
                   "ข้อความนี้คือแคปชันที่กดค้างเพื่อคัดลอกไปโพสต์ได้เลย",
        "hashtags": ["#ทดสอบระบบ"], "image_text": "ทดสอบระบบ",
        "mascot_pose": "happy", "image_idea": "-",
    }
    notifier = _notifier(args.dry_run)
    notifier.push(build_alert_messages(ev, analysis, now))
    print("ส่งข้อความทดสอบแล้ว ลองเปิด LINE ดูได้เลย" if not args.dry_run else "")
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
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
