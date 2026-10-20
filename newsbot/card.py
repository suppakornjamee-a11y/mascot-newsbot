"""Draw the news post image in the น้องปิ๊ง brand template (Pillow, no browser needed).

Layout follows the "ตัวอย่างโพสต์ประกาศข่าว" board: light-blue background, white circle
bottom-right, dark "ปิ๊ง! มีข่าวมาบอก" pill, orange kicker, big Mitr headline,
1-2 sentence detail, date + sources, and the mascot with a megaphone.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, features

from .config import ROOT, TZ

ASSETS = ROOT / "assets"
FONTS = ASSETS / "fonts"
MASCOT = ASSETS / "mascot"

BG = "#EEF3F8"
WHITE = "#FFFFFF"
INK = "#1C2230"
TEXT_2 = "#3A4152"
MUTED = "#5A6273"
KICKER = "#D16426"          # ส้มปิ๊ง #FF7A2F at 82% brightness, as in the template
DOT_NORMAL = "#2BC4AE"
DOT_URGENT = "#FF5A4E"

MOODS = ("smile", "excited", "serious")
# Proper Thai shaping (tone marks over vowels) needs libraqm; without it text still renders.
RAQM = features.check("raqm")
LANG = {"language": "th"} if RAQM else {}
TH_MONTHS = ["ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.",
             "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค."]


@dataclass
class CardText:
    kicker: str
    headline: str
    detail: str
    meta: str
    mood: str = "smile"
    urgent: bool = False


# ---------------------------------------------------------------- fonts & text
@lru_cache(maxsize=None)
def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    files = {
        "head": "Mitr-SemiBold.ttf",
        "head-medium": "Mitr-Medium.ttf",
        "body": "IBMPlexSansThai-Regular.ttf",
        "body-medium": "IBMPlexSansThai-Medium.ttf",
        "body-bold": "IBMPlexSansThai-SemiBold.ttf",
    }
    engine = ImageFont.Layout.RAQM if RAQM else ImageFont.Layout.BASIC
    return ImageFont.truetype(str(FONTS / files[name]), size, layout_engine=engine)


def _tokens(text: str) -> list[str]:
    """Split Thai text into words so lines break between words, not inside them."""
    try:
        from pythainlp.tokenize import word_tokenize

        return word_tokenize(text, engine="newmm", keep_whitespace=True)
    except Exception:  # library missing: fall back to spaces only
        parts = []
        for i, chunk in enumerate(text.split(" ")):
            if i:
                parts.append(" ")
            parts.append(chunk)
        return parts


def _length(f: ImageFont.FreeTypeFont, text: str) -> float:
    return f.getlength(text, **LANG)


def _split_long(f, word: str, width: float) -> list[str]:
    """Break a single over-long word without separating Thai vowels/tone marks."""
    marks = set(chr(c) for c in list(range(0x0E31, 0x0E32)) + list(range(0x0E34, 0x0E3B))
                + list(range(0x0E47, 0x0E4F)))
    out, cur = [], ""
    for ch in word:
        if cur and ch not in marks and _length(f, cur + ch) > width:
            out.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        out.append(cur)
    return out


def wrap(f, text: str, widths, mode: str = "word") -> list[str] | None:
    """Greedy wrap. ``widths(i)`` is the max width of line ``i``.

    mode "phrase" breaks only at spaces (how Thai headlines are set) and returns None
    if a phrase is too wide; mode "word" also breaks between Thai words.
    """
    text = " ".join(text.split())
    if mode == "phrase":
        toks = []
        for i, chunk in enumerate(text.split(" ")):
            if i:
                toks.append(" ")
            toks.append(chunk)
    else:
        toks = _tokens(text)
    lines: list[str] = []
    cur = ""
    for tok in toks:
        width = widths(len(lines))
        if not cur and tok.isspace():
            continue
        if _length(f, cur + tok) <= width:
            cur += tok
            continue
        if cur.strip():
            lines.append(cur.rstrip())
            cur = ""
            width = widths(len(lines))
        if tok.isspace():
            continue
        if _length(f, tok) > width:
            if mode == "phrase":
                return None
            pieces = _split_long(f, tok, width)
            lines.extend(pieces[:-1])
            cur = pieces[-1]
        else:
            cur = tok
    if cur.strip():
        lines.append(cur.rstrip())
    return lines


def wrap_balanced(f, text: str, widths, max_lines: int, modes=("phrase", "word")) -> list[str] | None:
    """Headline wrap: prefer breaks at spaces, then even out the line lengths."""
    for mode in modes:
        lines = wrap(f, text, widths, mode)
        if lines is None or len(lines) > max_lines:
            continue
        if len(lines) > 1:  # shrink the measure while the line count stays the same
            k = 0.95
            while k >= 0.55:
                trial = wrap(f, text, lambda i, k=k: widths(i) * k, mode)
                if trial is None or len(trial) != len(lines):
                    break
                lines = trial
                k -= 0.05
        return lines
    return None


def _ellipsize(f, line: str, width: float) -> str:
    while line and _length(f, line + "…") > width:
        line = line[:-1]
    return line.rstrip() + "…"


# ---------------------------------------------------------------- drawing
@lru_cache(maxsize=None)
def _mascot(mood: str, width: int) -> Image.Image | None:
    path = MASCOT / f"{mood if mood in MOODS else 'smile'}.png"
    if not path.exists():
        return None
    im = Image.open(path).convert("RGBA")
    return im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)


def _logo(max_w: int, max_h: int) -> Image.Image | None:
    path = ASSETS / "logo.png"
    if not path.exists():
        return None
    im = Image.open(path).convert("RGBA")
    im.thumbnail((max_w, max_h), Image.LANCZOS)
    return im


def thai_date(dt: datetime) -> str:
    local = dt.astimezone(TZ)
    return f"{local.day} {TH_MONTHS[local.month - 1]} {local.year + 543}"


def render_card(t: CardText, size: tuple[int, int] = (1080, 1080), page_name: str = "") -> Image.Image:
    W, H = size
    im = Image.new("RGBA", (W, H), BG)
    d = ImageDraw.Draw(im)

    # white circle, bottom-right (template: 760px, right -180, bottom -220)
    cx, cy, r = W - 200, H - 160, 380
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=WHITE)

    # top-left pill
    pill_font = font("head-medium", 28)
    label = "ปิ๊ง! ข่าวด่วน" if t.urgent else "ปิ๊ง! มีข่าวมาบอก"
    text_w = _length(pill_font, label)
    px, py, ph = 80, 80, 64
    pw = 30 + 18 + 12 + text_w + 30
    d.rounded_rectangle((px, py, px + pw, py + ph), radius=ph // 2, fill=INK)
    dot_y = py + ph // 2
    d.ellipse((px + 30, dot_y - 9, px + 48, dot_y + 9),
              fill=DOT_URGENT if t.urgent else DOT_NORMAL)
    d.text((px + 60, dot_y), label, font=pill_font, fill=WHITE, anchor="lm", **LANG)

    # top-right: logo image if provided, otherwise the page name
    logo = _logo(220, 64)
    if logo:
        im.alpha_composite(logo, (W - 80 - logo.width, py + (ph - logo.height) // 2))
    elif page_name:
        d.text((W - 80, dot_y), page_name, font=font("head-medium", 26), fill=MUTED,
               anchor="rm", **LANG)

    # mascot, bottom-right (template: 440 x 415 at right 40, bottom 30)
    mascot = _mascot(t.mood, 501)
    mascot_top = H - 452
    if mascot:
        im.alpha_composite(mascot, (W - 492, mascot_top))

    # text column: full width above the mascot, narrower beside it
    left, top, bottom = 80, 230, H - 80
    wide, narrow, narrow_from = 620, W - 492 - 80 - 24, mascot_top + 20

    def fit(head_size: int, detail_size: int, head_max: int, detail_max: int,
            modes=("phrase", "word")):
        kf, hf, bf, mf = (font("body-bold", 24), font("head", head_size),
                          font("body", detail_size), font("body-medium", 22))
        blocks, y = [], top
        # kicker
        blocks.append((kf, [t.kicker], 24 * 1.4, KICKER, y))
        y += 24 * 1.4 + 24

        def widths_from(y0, lh):
            return lambda i: wide if y0 + (i + 1) * lh <= narrow_from else narrow

        h_lh = head_size * 1.15
        h_lines = wrap_balanced(hf, t.headline, widths_from(y, h_lh), head_max, modes)
        if not h_lines:
            return None
        blocks.append((hf, h_lines, h_lh, INK, y))
        y += len(h_lines) * h_lh + 24

        d_lh = detail_size * 1.55
        d_widths = widths_from(y, d_lh)
        d_lines = []
        if t.detail and detail_max:
            d_lines = (wrap_balanced(bf, t.detail, d_widths, detail_max, ("word",))
                       or wrap(bf, t.detail, d_widths))
        if len(d_lines) > detail_max:
            keep = d_lines[:detail_max]
            idx = detail_max - 1
            width = widths_from(y, d_lh)(idx)
            keep[idx] = _ellipsize(bf, keep[idx], width)
            d_lines = keep
        if d_lines:
            blocks.append((bf, d_lines, d_lh, TEXT_2, y))
            y += len(d_lines) * d_lh + 24

        m_lh = 22 * 1.4
        m_lines = wrap(mf, t.meta, widths_from(y, m_lh))[:2]
        blocks.append((mf, m_lines, m_lh, MUTED, y))
        y += len(m_lines) * m_lh
        return blocks if y <= bottom else None

    sizes = ((80, 28, 2, 3), (72, 28, 3, 3), (64, 27, 3, 3), (58, 26, 3, 3),
             (52, 25, 4, 3), (46, 24, 4, 2))
    layout = None
    # Breaking only at spaces reads best in Thai; allow mid-phrase breaks only if no size works.
    for modes in (("phrase",), ("word",)):
        for head_size, detail_size, head_max, detail_max in sizes:
            layout = fit(head_size, detail_size, head_max, detail_max, modes)
            if layout:
                break
        if layout:
            break
    if layout is None:  # very long headline: keep 4 lines at the smallest size, end with …
        hf = font("head", 46)
        lines = wrap(hf, t.headline, lambda i: wide, "word")[:4]
        lines[-1] = _ellipsize(hf, lines[-1], wide)
        t = CardText(t.kicker, " ".join(lines), t.detail, t.meta, t.mood, t.urgent)
        layout = fit(46, 24, 5, 2) or fit(46, 24, 5, 0) or []

    for f, lines, lh, color, y in layout:
        for i, line in enumerate(lines):
            d.text((left, y + i * lh), line, font=f, fill=color, anchor="la", **LANG)

    return im


def card_bytes(img: Image.Image) -> tuple[bytes, bytes]:
    """PNG for posting + a small JPEG preview (LINE needs both)."""
    full = io.BytesIO()
    img.convert("RGB").save(full, "PNG", optimize=True)
    small = img.convert("RGB")
    small.thumbnail((540, 540), Image.LANCZOS)
    prev = io.BytesIO()
    small.save(prev, "JPEG", quality=85, optimize=True)
    return full.getvalue(), prev.getvalue()


def card_text_from(analysis: dict, event, now: datetime, urgent: bool) -> CardText:
    from .ai import CATEGORIES

    kicker = analysis.get("card_kicker") or CATEGORIES.get(event.category, "ข่าว")
    headline = analysis.get("card_headline") or analysis.get("headline") or event.headline
    detail = analysis.get("card_detail") or analysis.get("what_happened", "")
    sources = ", ".join(event.source_names[:3])
    meta = f"{thai_date(now)} · ที่มา: {sources}" if sources else thai_date(now)
    mood = str(analysis.get("mascot_mood", "smile")).lower()
    return CardText(kicker=kicker, headline=headline, detail=detail, meta=meta,
                    mood=mood if mood in MOODS else "smile", urgent=urgent)
