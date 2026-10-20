"""One run: fetch → triage → alert → digest → save. Designed to be called every 20-30 min."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Protocol

from .ai import Analyst
from .config import Config
from .fetch import Item, fetch_all
from .line import build_alert_messages, build_digest_message
from .state import Event, State
from .timeutil import due_digest_slot, in_hour_window

log = logging.getLogger("newsbot")

TRIAGE_BATCH = 40
MAX_ANALYSIS_ATTEMPTS = 3


class Notifier(Protocol):
    def push(self, messages: list[dict]) -> None: ...


@dataclass
class RunReport:
    fetched: int = 0
    new_items: int = 0
    feed_errors: dict[str, str] = field(default_factory=dict)
    events_created: int = 0
    events_updated: int = 0
    alerts_sent: list[str] = field(default_factory=list)
    held_quiet: int = 0
    digest_sent: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"อ่านข่าว {self.fetched} รายการ ใหม่ {self.new_items}",
            f"เหตุการณ์ใหม่ {self.events_created} อัปเดต {self.events_updated}",
            f"แจ้งด่วน {len(self.alerts_sent)} เรื่อง"
            + (f" (พักไว้ช่วงไม่รบกวน {self.held_quiet})" if self.held_quiet else ""),
            f"สรุปรอบวัน {self.digest_sent} เรื่อง",
        ]
        for name, err in self.feed_errors.items():
            lines.append(f"ฟีดมีปัญหา: {name} — {err}")
        for err in self.errors:
            lines.append(f"ข้อผิดพลาด: {err}")
        return "\n".join(lines)


Fetcher = Callable[[Config], dict[str, list[Item] | Exception]]


def default_fetcher(cfg: Config) -> dict[str, list[Item] | Exception]:
    return fetch_all(cfg.sources, cfg.fetch_timeout)


# ---------------------------------------------------------------- step: collect
def collect_new_items(cfg: Config, state: State, results: dict, now: datetime,
                      report: RunReport) -> list[Item]:
    max_age = timedelta(hours=cfg.max_item_age_hours)
    new: dict[str, Item] = {}
    for source_name, result in results.items():
        if isinstance(result, Exception):
            report.feed_errors[source_name] = str(result)[:200]
            continue
        first_time = source_name not in state.bootstrapped
        for it in result:
            report.fetched += 1
            if state.is_seen(it.id) or it.id in new:
                continue
            if it.published is None:
                # Undated items (e.g. TMD): on a source's first run, treat them as old.
                if first_time:
                    state.mark_seen(it.id, now)
                    continue
            elif now - it.published > max_age or it.published > now + timedelta(hours=1):
                state.mark_seen(it.id, now)
                continue
            new[it.id] = it
        state.bootstrapped.add(source_name)
    report.new_items = len(new)
    return list(new.values())


# ---------------------------------------------------------------- step: triage
def triage_items(state: State, analyst: Analyst, items: list[Item], now: datetime,
                 report: RunReport) -> None:
    for start in range(0, len(items), TRIAGE_BATCH):
        batch = items[start:start + TRIAGE_BATCH]
        try:
            groups = analyst.triage(batch, state.recent_events(now), now)
        except Exception as exc:  # leave the batch unseen so the next run retries it
            report.errors.append(f"คัดกรองข่าวไม่สำเร็จ: {exc}")
            log.exception("triage failed")
            continue

        assigned: set[int] = set()
        for g in groups:
            idxs = [i for i in g.item_indexes if 0 <= i < len(batch) and i not in assigned]
            if not idxs:
                continue
            assigned.update(idxs)
            ev = state.events.get(g.existing_event_id) if g.existing_event_id else None
            if ev is None:
                ev = state.new_event(g.headline or batch[idxs[0]].title, g.category, now)
                ev.promotional = g.promotional
                report.events_created += 1
            else:
                ev.promotional = ev.promotional and g.promotional
                report.events_updated += 1
            ev.add_items([batch[i] for i in idxs])
            ev.urgency = max(ev.urgency, g.urgency)
            ev.sensitive = ev.sensitive or g.sensitive
            ev.last_updated = now

        for i, it in enumerate(batch):  # anything the model skipped becomes its own low event
            if i not in assigned:
                ev = state.new_event(it.title, "other", now)
                ev.add_items([it])
                report.events_created += 1
        for it in batch:
            state.mark_seen(it.id, now)


# ---------------------------------------------------------------- step: alerts
def send_alerts(cfg: Config, state: State, analyst: Analyst, notifier: Notifier,
                now: datetime, report: RunReport) -> None:
    max_age = timedelta(hours=cfg.alert_max_age_hours)
    quiet = in_hour_window(now, cfg.quiet_hours)
    candidates = sorted(
        (e for e in state.events.values()
         if e.status == "new" and not e.promotional
         and e.urgency >= cfg.alert_threshold and now - e.earliest <= max_age),
        key=lambda e: e.urgency, reverse=True,
    )
    sent = 0
    for ev in candidates:
        if sent >= cfg.alert_max_per_run:
            break
        if quiet and ev.urgency < cfg.quiet_override:
            report.held_quiet += 1
            continue
        if ev.analysis and ev.analysis.get("_item_count") == len(ev.items):
            analysis = ev.analysis  # nothing new since the last analysis: reuse it
        else:
            try:
                analysis = analyst.analyze(ev, now)
            except Exception as exc:
                ev.attempts += 1
                if ev.attempts >= MAX_ANALYSIS_ATTEMPTS:
                    ev.status = "failed"
                report.errors.append(f"วิเคราะห์ไม่สำเร็จ ({ev.headline[:40]}): {exc}")
                log.exception("analysis failed")
                continue
            analysis["_item_count"] = len(ev.items)

        ev.analysis = analysis
        ev.sensitive = ev.sensitive or bool(analysis.get("sensitive"))
        if analysis["urgency"] < cfg.alert_threshold:
            # With the full text the story looks less urgent: keep it for the digest.
            ev.urgency = analysis["urgency"]
            continue
        ev.urgency = analysis["urgency"]
        if quiet and ev.urgency < cfg.quiet_override:
            report.held_quiet += 1
            continue

        try:
            notifier.push(build_alert_messages(ev, analysis, now, cfg.mascot.name))
        except Exception as exc:
            report.errors.append(f"ส่ง LINE ไม่สำเร็จ: {exc}")
            log.exception("line push failed")
            break  # LINE is down or misconfigured: stop, retry next run
        ev.status = "notified"
        ev.notified_at = now
        report.alerts_sent.append(ev.headline)
        sent += 1


# ---------------------------------------------------------------- step: digest
def send_digest(cfg: Config, state: State, notifier: Notifier, now: datetime,
                report: RunReport) -> None:
    slot = due_digest_slot(now, cfg.digest_hours, state.digests_sent)
    if not slot:
        return
    since = state.last_digest_at or now - timedelta(hours=12)
    picks = sorted(
        (e for e in state.events.values()
         if e.status == "new" and not e.promotional
         and e.urgency >= cfg.digest_min_score and e.last_updated >= since),
        key=lambda e: e.urgency, reverse=True,
    )[: cfg.digest_max_items]
    if picks:
        try:
            notifier.push([build_digest_message(picks, now)])
        except Exception as exc:
            report.errors.append(f"ส่งสรุปข่าวไม่สำเร็จ: {exc}")
            log.exception("digest push failed")
            return  # slot stays unsent, next run retries
        for ev in picks:
            ev.status = "digested"
        report.digest_sent = len(picks)
    state.digests_sent.add(slot)
    state.last_digest_at = now


# ---------------------------------------------------------------- run
def run_once(cfg: Config, state: State, analyst: Analyst, notifier: Notifier,
             now: datetime, fetcher: Fetcher = default_fetcher) -> RunReport:
    report = RunReport()
    results = fetcher(cfg)
    items = collect_new_items(cfg, state, results, now, report)
    if items:
        triage_items(state, analyst, items, now, report)
    send_alerts(cfg, state, analyst, notifier, now, report)
    send_digest(cfg, state, notifier, now, report)
    state.prune(now)
    return report
