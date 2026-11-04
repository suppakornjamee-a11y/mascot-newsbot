"""Draw the news post image in the น้องปิ๊ง brand template (Pillow, no browser needed).

Two layouts:

* ``render_photo_card``: a photo about the story fills the card, darkened at the bottom,
  with the "ปิ๊ง!" pill, an orange topic tag, a big white headline, a short detail in
  small type, date + sources, the photo credit, and the mascot as a sticker in the
  bottom-right corner.
* ``render_card``: the original flat board ("ตัวอย่างโพสต์ประกาศข่าว"): light-blue
  background, white circle, headline, 1-2 sentence detail and the mascot. Used when
  no suitable photo was found.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps, features

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
ORANGE = "#FF7A2F"          # ส้มปิ๊ง: topic tag on photo cards
PHOTO_META = "#E2E6EE"
PHOTO_DETAIL = "#EEF1F6"
# Brand tones for photos taken from the news (card.source_look: brand): deep navy shadows,
# blue-grey mid-tones and warm light highlights.
BRAND_SHADOW, BRAND_MID, BRAND_LIGHT = "#141B2B", "#3F6E8C", "#FFE7D1"
SOURCE_LOOKS = ("brand", "light")
PHOTO_CREDIT = "#B4BCC8"

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
    credit: str = ""         # photo credit, photo cards only


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


def _glue_numbers(chunks: list[str]) -> list[str]:
    """Keep numbers with their short neighbours: "60 สต.", "ตี 5", "21 ต.ค." stay on one line."""
    out: list[str] = []
    for chunk in chunks:
        prev = out[-1] if out else ""
        prev_num = prev[-1:].isdigit()
        this_num = chunk[:1].isdigit()
        if prev and ((prev_num and (len(chunk) <= 4 or chunk.endswith(".")))
                     or (this_num and len(prev) <= 2)):
            out[-1] = f"{prev} {chunk}"
        else:
            out.append(chunk)
    return out


def wrap(f, text: str, widths, mode: str = "word") -> list[str] | None:
    """Greedy wrap. ``widths(i)`` is the max width of line ``i``.

    mode "phrase" breaks only at spaces (how Thai headlines are set) and returns None
    if a phrase is too wide; mode "word" also breaks between Thai words.
    """
    text = " ".join(text.split())
    if mode == "phrase":
        toks = []
        for i, chunk in enumerate(_glue_numbers(text.split(" "))):
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


def _draw_top_bar(im: Image.Image, d: ImageDraw.ImageDraw, urgent: bool, page_name: str,
                  name_color: str, margin: int = 80) -> None:
    """The dark "ปิ๊ง!" pill top-left, and the logo or page name top-right."""
    W = im.width
    pill_font = font("head-medium", 28)
    label = "ปิ๊ง! ข่าวด่วน" if urgent else "ปิ๊ง! มีข่าวมาบอก"
    text_w = _length(pill_font, label)
    px, py, ph = margin, margin, 64
    pw = 30 + 18 + 12 + text_w + 30
    d.rounded_rectangle((px, py, px + pw, py + ph), radius=ph // 2, fill=INK)
    dot_y = py + ph // 2
    d.ellipse((px + 30, dot_y - 9, px + 48, dot_y + 9),
              fill=DOT_URGENT if urgent else DOT_NORMAL)
    d.text((px + 60, dot_y), label, font=pill_font, fill=WHITE, anchor="lm", **LANG)

    logo = _logo(220, 64)
    if logo:
        im.alpha_composite(logo, (W - margin - logo.width, py + (ph - logo.height) // 2))
    elif page_name:
        d.text((W - margin, dot_y), page_name, font=font("head-medium", 26), fill=name_color,
               anchor="rm", **LANG)


def render_card(t: CardText, size: tuple[int, int] = (1080, 1080), page_name: str = "") -> Image.Image:
    W, H = size
    im = Image.new("RGBA", (W, H), BG)
    d = ImageDraw.Draw(im)

    # white circle, bottom-right (template: 760px, right -180, bottom -220)
    cx, cy, r = W - 200, H - 160, 380
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=WHITE)

    _draw_top_bar(im, d, t.urgent, page_name, MUTED)

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


# ---------------------------------------------------------------- photo card
def cover(photo: Image.Image, W: int, H: int, focus_y: float = 0.42) -> Image.Image:
    """Scale the photo to fill W x H and crop the rest.

    Crops a little above the centre: the bottom of the card is covered by text, and
    subjects in news photos tend to sit in the upper half.
    """
    photo = photo.convert("RGB")
    scale = max(W / photo.width, H / photo.height)
    size = (max(W, round(photo.width * scale)), max(H, round(photo.height * scale)))
    im = photo.resize(size, Image.LANCZOS)
    left = (im.width - W) // 2
    top = round(min(max(im.height * focus_y - H / 2, 0), im.height - H))
    return im.crop((left, top, left + W, top + H))


def restyle_source(photo: Image.Image, size: tuple[int, int], look: str = "brand") -> Image.Image:
    """Give a photo from the news article the card's own look. Returns a W x H image.

    light: zooms in a little (trims the edges, where outlets put watermarks) and evens out
           the colour toward the brand palette.
    brand: zooms in further and re-colours the photo in the brand tones (navy shadows, warm
           highlights), so it reads as the page's own artwork.
    Either way, a photo too small to fill the card sharply is set at the top over a
    blurred, darkened copy of itself.
    """
    W, H = size
    photo = photo.convert("RGB")
    trim = 0.08 if look == "brand" else 0.04
    dx, dy = round(photo.width * trim), round(photo.height * trim)
    photo = photo.crop((dx, dy, photo.width - dx, photo.height - dy))
    photo = ImageEnhance.Contrast(photo).enhance(1.08 if look == "brand" else 1.06)
    if look == "brand":
        duo = ImageOps.colorize(ImageOps.grayscale(photo), black=BRAND_SHADOW, white=BRAND_LIGHT,
                                mid=BRAND_MID)
        photo = Image.blend(photo, duo, 0.5)
    else:
        photo = ImageEnhance.Color(photo).enhance(0.9)
        photo = Image.blend(photo, Image.new("RGB", photo.size, INK), 0.06)

    if min(photo.width / W, photo.height / H) >= 0.8:      # big enough: fill the card
        return cover(photo, W, H)
    back = cover(photo, W, H).filter(ImageFilter.GaussianBlur(28))
    back = Image.blend(back, Image.new("RGB", (W, H), INK), 0.35)
    scale = min(W / photo.width, H * 0.72 / photo.height)
    fg = photo.resize((round(photo.width * scale), round(photo.height * scale)), Image.LANCZOS)
    mask = Image.new("L", fg.size, 255)                      # fade the bottom edge into the blur
    fade = max(1, min(140, fg.height // 4))
    md = ImageDraw.Draw(mask)
    for i in range(fade):
        y = fg.height - fade + i
        md.line([(0, y), (fg.width, y)], fill=round(255 * (1 - (i + 1) / fade)))
    back.paste(fg, ((W - fg.width) // 2, 0), mask)
    return back


def _shade(W: int, H: int, stops: list[tuple[float, float]], color: str = INK) -> Image.Image:
    """A vertical gradient of ``color``; ``stops`` are (y, opacity 0-1) pairs, eased."""
    stops = sorted(stops)
    column = []
    for y in range(H):
        a = stops[-1][1]
        if y <= stops[0][0]:
            a = stops[0][1]
        elif y < stops[-1][0]:
            for (y0, a0), (y1, a1) in zip(stops, stops[1:]):
                if y0 <= y <= y1:
                    k = (y - y0) / max(1, y1 - y0)
                    a = a0 + (a1 - a0) * k * k * (3 - 2 * k)   # smoothstep
                    break
        column.append(round(255 * max(0.0, min(1.0, a))))
    mask = Image.new("L", (1, H))
    mask.putdata(column)
    layer = Image.new("RGBA", (W, H), color)
    layer.putalpha(mask.resize((W, H)))
    return layer


@lru_cache(maxsize=None)
def _mascot_sticker(mood: str, width: int) -> Image.Image | None:
    """The mascot with a white sticker outline and a soft shadow, so it reads on any photo."""
    base = _mascot(mood, width)
    if base is None:
        return None
    ring_r = max(4, round(width * 0.02))
    blur = max(6, round(width * 0.03))
    pad = ring_r + blur * 2
    alpha = Image.new("L", (base.width + 2 * pad, base.height + 2 * pad), 0)
    alpha.paste(base.getchannel("A"), (pad, pad))
    solid = alpha.point(lambda v: 255 if v > 24 else 0)
    ring = solid.filter(ImageFilter.MaxFilter(2 * ring_r + 1))
    shadow = ring.filter(ImageFilter.GaussianBlur(blur)).point(lambda v: round(v * 0.45))
    out = Image.new("RGBA", alpha.size, (0, 0, 0, 0))
    dark = Image.new("RGBA", alpha.size, INK)
    dark.putalpha(shadow)
    out.alpha_composite(dark, (0, round(blur * 0.6)))
    white = Image.new("RGBA", alpha.size, WHITE)
    white.putalpha(ring)
    out.alpha_composite(white)
    out.alpha_composite(base, (pad, pad))
    return out


def _text_with_shadow(im: Image.Image, blocks, blur: int = 8, opacity: float = 0.55) -> None:
    """Draw text blocks with a soft dark halo so they stay readable on bright photos."""
    W, H = im.size
    text_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    td = ImageDraw.Draw(text_layer)
    for f, lines, lh, color, y, x in blocks:
        for i, line in enumerate(lines):
            td.text((x, y + i * lh), line, font=f, fill=color, anchor="la", **LANG)
    halo = text_layer.getchannel("A").filter(ImageFilter.GaussianBlur(blur))
    halo = halo.point(lambda v: round(min(255, v * 1.6) * opacity))
    dark = Image.new("RGBA", (W, H), INK)
    dark.putalpha(halo)
    im.alpha_composite(dark)
    im.alpha_composite(text_layer)


PHOTO_HEAD_SIZES = ((78, 3), (70, 3), (62, 3), (56, 4), (50, 4), (46, 4))


def _has_orphan(f, lines: list[str], width: float) -> bool:
    """A stray short line, e.g. "ตร." or "ครม." alone above or between long ones."""
    if len(lines) < 2:
        return False
    lengths = [_length(f, line) for line in lines]
    return min(lengths) < width * 0.25 or min(lengths) < max(lengths) * 0.4


def _photo_headline(text: str, width: float, sizes=PHOTO_HEAD_SIZES
                    ) -> tuple[ImageFont.FreeTypeFont, int, list[str]]:
    """Largest size whose lines fit; prefer breaking at spaces (phrases), then words,
    and a slightly smaller size over a layout with a stray short line."""
    fallback = None
    for modes in (("phrase",), ("word",)):
        for size, max_lines in sizes:
            f = font("head", size)
            lines = wrap_balanced(f, text, lambda i: width, max_lines, modes)
            if not lines:
                continue
            if not _has_orphan(f, lines, width):
                return f, size, lines
            fallback = fallback or (f, size, lines)
    if fallback:
        return fallback
    size = sizes[-1][0]
    f = font("head", size)
    lines = wrap(f, text, lambda i: width, "word")[:4]
    lines[-1] = _ellipsize(f, lines[-1], width)
    return f, size, lines


def render_photo_card(t: CardText, photo: Image.Image, size: tuple[int, int] = (1080, 1080),
                      page_name: str = "") -> Image.Image:
    """Photo background + topic tag + headline + mascot sticker (see module docstring)."""
    W, H = size
    im = cover(photo, W, H).convert("RGBA")
    margin = 72

    # mascot sticker, bottom-right; its visible left edge limits the text column
    sticker = _mascot_sticker(t.mood if t.mood in MOODS else "smile", 400)
    if sticker:
        sx, sy = W - sticker.width + 12, H - sticker.height + 8
        text_right = sx + round(sticker.width * 0.09)
    else:
        sx = sy = 0
        text_right = W - margin
    text_w = text_right - margin - 20

    # ---- bottom text block, laid out from the bottom up:
    #      topic tag, headline, detail (small), date + sources, credit
    tag_font = font("body-bold", 28)
    detail_font = font("body", 26)
    meta_font, credit_font = font("body-medium", 21), font("body", 19)
    meta_lines = wrap(meta_font, t.meta, lambda i: text_w)[:2] if t.meta else []
    credit_lines = wrap(credit_font, t.credit, lambda i: text_w)[:2] if t.credit else []
    detail_lines: list[str] = []
    if t.detail:
        detail_lines = wrap(detail_font, t.detail, lambda i: text_w)
        if len(detail_lines) > 3:
            detail_lines = detail_lines[:3]
            detail_lines[-1] = _ellipsize(detail_font, detail_lines[-1], text_w)

    meta_lh, credit_lh, detail_lh = 30, 27, 37
    tag_h = 56 if t.kicker else 0
    top_limit = margin + 64 + 48                       # below the "ปิ๊ง!" pill
    credit_y = H - 56 - len(credit_lines) * credit_lh
    meta_y = credit_y - len(meta_lines) * meta_lh - (6 if credit_lines else 0)
    detail_y = meta_y - len(detail_lines) * detail_lh - (14 if detail_lines else 0)
    for start in range(len(PHOTO_HEAD_SIZES)):         # shrink the headline until it all fits
        hf, head_size, head_lines = _photo_headline(t.headline, text_w, PHOTO_HEAD_SIZES[start:])
        head_lh = round(head_size * 1.18)
        head_y = detail_y - len(head_lines) * head_lh - 16
        tag_y = head_y - (tag_h + 22 if tag_h else 0)
        if tag_y >= top_limit:
            break

    # ---- darken: a light veil at the top (for the pill), a deep one behind the text
    im.alpha_composite(_shade(W, H, [(0, 0.55), (240, 0.0)]))
    im.alpha_composite(_shade(W, H, [(tag_y - 300, 0.0), (tag_y + 40, 0.70),
                                      (head_y + head_lh * 2, 0.85), (H, 0.92)]))
    d = ImageDraw.Draw(im)
    _draw_top_bar(im, d, t.urgent, page_name, WHITE, margin)

    if t.kicker:
        label = t.kicker
        while label and _length(tag_font, label) > text_w - 44:
            label = label[:-1]
        tw = _length(tag_font, label)
        d.rounded_rectangle((margin, tag_y, margin + tw + 44, tag_y + tag_h), radius=12,
                            fill=ORANGE)
        d.text((margin + 22, tag_y + tag_h / 2), label, font=tag_font, fill=INK, anchor="lm",
               **LANG)

    blocks = [(hf, head_lines, head_lh, WHITE, head_y, margin)]
    if detail_lines:
        blocks.append((detail_font, detail_lines, detail_lh, PHOTO_DETAIL, detail_y, margin))
    if meta_lines:
        blocks.append((meta_font, meta_lines, meta_lh, PHOTO_META, meta_y, margin))
    if credit_lines:
        blocks.append((credit_font, credit_lines, credit_lh, PHOTO_CREDIT, credit_y, margin))
    _text_with_shadow(im, blocks)

    if sticker:
        im.alpha_composite(sticker, (sx, sy))
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

    kicker = (analysis.get("card_topic") or analysis.get("card_kicker")
              or CATEGORIES.get(event.category, "ข่าว"))
    headline = analysis.get("card_headline") or analysis.get("headline") or event.headline
    detail = analysis.get("card_detail") or analysis.get("what_happened", "")
    sources = ", ".join(event.source_names[:3])
    meta = f"{thai_date(now)} · ที่มา: {sources}" if sources else thai_date(now)
    mood = str(analysis.get("mascot_mood", "smile")).lower()
    return CardText(kicker=kicker, headline=headline, detail=detail, meta=meta,
                    mood=mood if mood in MOODS else "smile", urgent=urgent)
