"""Build น้องปิ๊ง (Ping) mascot PNGs from the SVG in the brand character sheet.

Run once whenever the mascot changes:  python tools/build_mascot.py
Writes assets/mascot/{smile,excited,serious}.svg and .png (transparent background).
Needs `pip install resvg-py` (only for this build step, not at runtime).
"""
from pathlib import Path

BODY = "#FF7A2F"
CREAM = "#FFD9B8"
TEAL = "#149985"
TEAL_DARK = "#0F7A6A"
INK = "#1C2230"
BLUSH = "#FF8FA3"

ARM_WAVE = (f'<path d="M50 200 C 24 182 16 150 26 124" stroke="{BODY}" stroke-width="20" '
            f'stroke-linecap="round" fill="none"/><circle cx="26" cy="116" r="15" fill="{BODY}"/>')
ARM_DOWN = (f'<path d="M48 214 C 30 228 24 248 28 264" stroke="{BODY}" stroke-width="20" '
            f'stroke-linecap="round" fill="none"/><circle cx="28" cy="270" r="15" fill="{BODY}"/>')

FACES = {
    # ยิ้มสดใส: general updates, daily news
    "smile": f"""
<ellipse cx="124" cy="170" rx="11" ry="14" fill="{INK}"/>
<ellipse cx="176" cy="170" rx="11" ry="14" fill="{INK}"/>
<circle cx="128" cy="164" r="4" fill="#FFFFFF"/>
<circle cx="180" cy="164" r="4" fill="#FFFFFF"/>
<ellipse cx="100" cy="196" rx="13" ry="8" fill="{BLUSH}" opacity="0.65"/>
<ellipse cx="200" cy="196" rx="13" ry="8" fill="{BLUSH}" opacity="0.65"/>
<path d="M135 192 Q 150 214 165 192 Z" fill="{INK}"/>""",
    # ตื่นเต้น: good news, launches
    "excited": f"""
<path d="M115 172 Q 124 160 133 172" stroke="{INK}" stroke-width="6" stroke-linecap="round" fill="none"/>
<path d="M167 172 Q 176 160 185 172" stroke="{INK}" stroke-width="6" stroke-linecap="round" fill="none"/>
<ellipse cx="100" cy="196" rx="13" ry="8" fill="{BLUSH}" opacity="0.75"/>
<ellipse cx="200" cy="196" rx="13" ry="8" fill="{BLUSH}" opacity="0.75"/>
<path d="M132 190 Q 150 222 168 190 Z" fill="{INK}"/>""",
    # จริงจัง: important notices, warnings, bad news (scaled from the sheet's bust)
    "serious": f"""
<path d="M111 152 L 138 150" stroke="{INK}" stroke-width="6" stroke-linecap="round"/>
<path d="M189 152 L 162 150" stroke="{INK}" stroke-width="6" stroke-linecap="round"/>
<ellipse cx="124" cy="173" rx="10" ry="11.5" fill="{INK}"/>
<ellipse cx="176" cy="173" rx="10" ry="11.5" fill="{INK}"/>
<path d="M136 207 L 164 207" stroke="{INK}" stroke-width="6" stroke-linecap="round"/>""",
}


def mascot_svg(mood: str) -> str:
    arm = ARM_DOWN if mood == "serious" else ARM_WAVE
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="410" height="346" viewBox="-20 -6 410 346">
<ellipse cx="150" cy="322" rx="96" ry="10" fill="{INK}" opacity="0.08"/>
<ellipse cx="118" cy="308" rx="25" ry="12" fill="{BODY}"/>
<ellipse cx="182" cy="308" rx="25" ry="12" fill="{BODY}"/>
<ellipse cx="118" cy="308" rx="25" ry="12" fill="{INK}" opacity="0.18"/>
<ellipse cx="182" cy="308" rx="25" ry="12" fill="{INK}" opacity="0.18"/>
<path d="M150 74 Q 146 46 160 26" stroke="{INK}" stroke-width="5" stroke-linecap="round" fill="none"/>
<circle cx="160" cy="22" r="11" fill="{TEAL}"/>
<path d="M177 10 Q 185 22 177 34" stroke="{TEAL}" stroke-width="4" stroke-linecap="round" fill="none"/>
<path d="M188 2 Q 200 22 188 42" stroke="{TEAL}" stroke-width="4" stroke-linecap="round" fill="none" opacity="0.5"/>
<path d="M250 205 C 272 208 284 218 288 232" stroke="{BODY}" stroke-width="20" stroke-linecap="round" fill="none"/>
<path d="M150 70 C 230 70 262 140 262 210 C 262 280 214 312 150 312 C 86 312 38 280 38 210 C 38 140 70 70 150 70 Z" fill="{BODY}"/>
<ellipse cx="150" cy="240" rx="72" ry="56" fill="{CREAM}"/>
{arm}
<g transform="translate(296 222) rotate(-25)">
<path d="M0 -9 L 44 -28 L 44 28 L 0 9 Z" fill="{TEAL}"/>
<rect x="42" y="-30" width="9" height="60" rx="4" fill="{TEAL_DARK}"/>
<path d="M62 -20 L 74 -28" stroke="{TEAL}" stroke-width="4" stroke-linecap="round"/>
<path d="M64 0 L 78 0" stroke="{TEAL}" stroke-width="4" stroke-linecap="round"/>
<path d="M62 20 L 74 28" stroke="{TEAL}" stroke-width="4" stroke-linecap="round"/>
</g>
<circle cx="290" cy="234" r="15" fill="{BODY}"/>
{FACES[mood]}
<path d="M150 228 L 126 215 L 126 241 Z" fill="{TEAL}"/>
<path d="M150 228 L 174 215 L 174 241 Z" fill="{TEAL}"/>
<circle cx="150" cy="228" r="7" fill="{TEAL_DARK}"/>
</svg>
"""


def main() -> None:
    import resvg_py

    out = Path(__file__).resolve().parent.parent / "assets" / "mascot"
    out.mkdir(parents=True, exist_ok=True)
    for mood in FACES:
        svg = mascot_svg(mood)
        (out / f"{mood}.svg").write_text(svg, encoding="utf-8")
        png = resvg_py.svg_to_bytes(svg_string=svg, width=1025, height=865)
        (out / f"{mood}.png").write_bytes(bytes(png))
        print("wrote", out / f"{mood}.png")


if __name__ == "__main__":
    main()
