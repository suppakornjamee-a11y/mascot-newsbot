"""Load config.yaml and secrets from the environment (or a local .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

TZ = ZoneInfo("Asia/Bangkok")
ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Source:
    name: str
    url: str
    enabled: bool = True
    official: bool = False
    link: str | None = None  # fallback link for feeds whose items have none


@dataclass
class Mascot:
    name: str = "น้องข่าว"
    personality: str = ""
    voice: str = ""
    sign_off: str = ""


@dataclass
class Config:
    mascot: Mascot
    sources: list[Source]
    alert_threshold: int = 70
    alert_max_per_run: int = 3
    alert_max_age_hours: float = 6
    quiet_hours: list[int] = field(default_factory=lambda: [0, 6])
    quiet_override: int = 90
    digest_hours: list[int] = field(default_factory=lambda: [8, 12, 18])
    digest_min_score: int = 35
    digest_max_items: int = 8
    max_item_age_hours: float = 6
    fetch_timeout: float = 20
    ai_provider: str = "gemini"            # gemini (free tier) | claude (paid)
    triage_model: str = "gemini-3.5-flash-lite"
    triage_effort: str | None = "low"      # Claude effort / Gemini thinking level
    analysis_model: str = "gemini-3.8-flash"
    analysis_effort: str | None = "medium"
    # Fact-check and photo pick. A lighter model with its own free quota: the free tier
    # of the analysis model allows only ~20 requests a day per project.
    review_model: str = "gemini-3.5-flash-lite"
    review_effort: str | None = "medium"
    state_path: Path = ROOT / "data" / "state.json"
    card_enabled: bool = True
    card_size: tuple[int, int] = (1080, 1080)
    card_page_name: str = ""
    card_branch: str = "cards"
    card_background: str = "photo"        # photo (Pixabay/Pexels) | ai (Gemini paints) | plain
    card_ai_pick: bool = True              # let the AI choose among candidate photos
    card_image_provider: str = "cloudflare"   # cloudflare (free allowance) | gemini (paid)
    card_image_model: str = ""                  # empty: the provider's default model
    editor_fact_check: bool = True         # second AI pass checks the post against sources

    @property
    def api_key_env(self) -> str:
        return "ANTHROPIC_API_KEY" if self.ai_provider == "claude" else "GEMINI_API_KEY"


BACKGROUNDS = ("photo", "ai", "plain")
IMAGE_PROVIDERS = ("cloudflare", "gemini")

AI_DEFAULTS = {
    "gemini": {"triage_model": "gemini-3.5-flash-lite", "triage_level": "low",
               "analysis_model": "gemini-3.8-flash", "analysis_level": "medium",
               "review_model": "gemini-3.5-flash-lite", "review_level": "medium"},
    "claude": {"triage_model": "claude-haiku-5-5", "triage_level": "low",
               "analysis_model": "claude-sonnet-5-5", "analysis_level": "medium",
               "review_model": "claude-haiku-5-5", "review_level": "medium"},
}


def load_dotenv(path: Path) -> None:
    """Minimal .env reader: KEY=value lines; existing env vars win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), value)


def load_config(path: Path | None = None) -> Config:
    path = path or ROOT / "config.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    m = raw.get("mascot") or {}
    alerts = raw.get("alerts") or {}
    digest = raw.get("digest") or {}
    fetch = raw.get("fetch") or {}
    ai = raw.get("ai") or {}
    card = raw.get("card") or {}
    editor = raw.get("editor") or {}

    sources = []
    for s in raw.get("sources") or []:
        sources.append(
            Source(
                name=str(s["name"]),
                url=str(s["url"]),
                enabled=bool(s.get("enabled", True)),
                official=bool(s.get("official", False)),
                link=s.get("link"),
            )
        )

    defaults = Config(mascot=Mascot(), sources=[])
    provider = str(ai.get("provider", "gemini")).strip().lower()
    if provider not in AI_DEFAULTS:
        raise SystemExit(f"ai.provider ต้องเป็น gemini หรือ claude (ตอนนี้คือ '{provider}')")
    p = {**AI_DEFAULTS[provider], **(ai.get(provider) or {})}
    background = str(card.get("background", defaults.card_background) or "plain").strip().lower()
    if background not in BACKGROUNDS:
        raise SystemExit(f"card.background ต้องเป็น photo, ai หรือ plain (ตอนนี้คือ '{background}')")
    image_provider = str(card.get("image_provider", defaults.card_image_provider)
                         or "cloudflare").strip().lower()
    if image_provider not in IMAGE_PROVIDERS:
        raise SystemExit(f"card.image_provider ต้องเป็น cloudflare หรือ gemini "
                         f"(ตอนนี้คือ '{image_provider}')")
    return Config(
        mascot=Mascot(
            name=str(m.get("name", defaults.mascot.name)),
            personality=str(m.get("personality", "")),
            voice=str(m.get("voice", "")),
            sign_off=str(m.get("sign_off", "") or ""),
        ),
        sources=sources,
        alert_threshold=int(alerts.get("threshold", defaults.alert_threshold)),
        alert_max_per_run=int(alerts.get("max_per_run", defaults.alert_max_per_run)),
        alert_max_age_hours=float(alerts.get("max_age_hours", defaults.alert_max_age_hours)),
        quiet_hours=list(alerts.get("quiet_hours", defaults.quiet_hours) or []),
        quiet_override=int(alerts.get("quiet_override", defaults.quiet_override)),
        digest_hours=list(digest.get("hours", defaults.digest_hours) or []),
        digest_min_score=int(digest.get("min_score", defaults.digest_min_score)),
        digest_max_items=int(digest.get("max_items", defaults.digest_max_items)),
        max_item_age_hours=float(fetch.get("max_item_age_hours", defaults.max_item_age_hours)),
        fetch_timeout=float(fetch.get("timeout_seconds", defaults.fetch_timeout)),
        ai_provider=provider,
        triage_model=str(p["triage_model"]),
        triage_effort=p.get("triage_level") or None,
        analysis_model=str(p["analysis_model"]),
        analysis_effort=p.get("analysis_level") or None,
        review_model=str(p["review_model"]),
        review_effort=p.get("review_level") or None,
        card_enabled=bool(card.get("enabled", True)),
        card_size=tuple(int(x) for x in (card.get("size") or [1080, 1080]))[:2],
        card_page_name=str(card.get("page_name", "") or ""),
        card_branch=str(card.get("branch", "cards") or "cards"),
        card_background=background,
        card_ai_pick=bool(card.get("ai_pick", defaults.card_ai_pick)),
        card_image_provider=image_provider,
        card_image_model=str(card.get("image_model", "") or "").strip(),
        editor_fact_check=bool(editor.get("fact_check", defaults.editor_fact_check)),
    )


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(
            f"ยังไม่ได้ตั้งค่า {name} — ใส่ในไฟล์ .env หรือ GitHub Secrets (ดู README)"
        )
    return value
