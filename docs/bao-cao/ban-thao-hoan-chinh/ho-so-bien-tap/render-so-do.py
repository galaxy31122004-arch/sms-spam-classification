"""Render every Mermaid block of the report to a trimmed PNG under hinh/.

Mermaid is rendered by headless Microsoft Edge against the mermaid bundle on jsDelivr, then the
white margin is trimmed with Pillow. File name comes from the figure caption above the block,
so hinh-3-2.png belongs to "Hình 3.2".
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageChops

AUDIT = Path(__file__).resolve().parent
REPORT = AUDIT.parent
FIGURES = REPORT / "hinh"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
MERMAID = "https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.min.js"
# The canvas must be larger than every diagram at its natural size: a diagram that touches the
# edge is clipped, not scaled. SCALE renders at twice the pixel density so text stays sharp in print.
CANVAS = (2200, 3000)
SCALE = 2
PADDING = 24

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><script src="{cdn}"></script>
<style>
  html, body {{ margin: 0; padding: 0; background: #ffffff; }}
  .wrap {{ width: {w}px; height: {h}px; display: flex; align-items: center; justify-content: center; }}
  .mermaid {{ font-family: "Times New Roman", serif; }}
  .mermaid svg {{ width: auto; height: auto; }}
</style></head>
<body><div class="wrap"><pre class="mermaid">{code}</pre></div>
<script>mermaid.initialize({{ startOnLoad: true, theme: "neutral",
  fontFamily: "Times New Roman, serif", flowchart: {{ useMaxWidth: false, htmlLabels: true }} }});</script>
</body></html>
"""


def trim(path: Path) -> tuple[int, int]:
    image = Image.open(path).convert("RGB")
    background = Image.new("RGB", image.size, (255, 255, 255))
    box = ImageChops.difference(image, background).getbbox()
    if box:
        left, top, right, bottom = box
        box = (max(0, left - PADDING), max(0, top - PADDING),
               min(image.width, right + PADDING), min(image.height, bottom + PADDING))
        image = image.crop(box)
    image.save(path)
    return image.size


def blocks():
    """Yield (figure id, mermaid source) for every diagram, in chapter order."""
    for chapter in sorted(REPORT.glob("0*.md")):
        text = chapter.read_text(encoding="utf-8")
        for match in re.finditer(r"^\*\*(Hình ([\d.]+))\.[^\n]*\*\*\s*\n+```mermaid\n(.*?)\n```",
                                 text, re.S | re.M):
            yield match.group(2).rstrip("."), match.group(3)


def main() -> None:
    if not EDGE.is_file():
        sys.exit(f"Khong tim thay Edge tai {EDGE}")
    FIGURES.mkdir(exist_ok=True)
    rendered = 0
    with tempfile.TemporaryDirectory() as directory:
        for number, code in blocks():
            page = Path(directory) / f"hinh-{number.replace('.', '-')}.html"
            page.write_text(PAGE.format(cdn=MERMAID, w=CANVAS[0], h=CANVAS[1], code=code), encoding="utf-8")
            target = FIGURES / f"hinh-{number.replace('.', '-')}.png"
            subprocess.run([str(EDGE), "--headless=new", "--disable-gpu", "--hide-scrollbars",
                            f"--window-size={CANVAS[0]},{CANVAS[1]}", "--virtual-time-budget=15000",
                            f"--force-device-scale-factor={SCALE}",
                            f"--screenshot={target}", page.as_uri()],
                           check=True, capture_output=True, timeout=120)
            width, height = trim(target)
            clipped = width >= CANVAS[0] * SCALE or height >= CANVAS[1] * SCALE
            print(f"Hình {number}: {target.name} {width}x{height}" + ("  [BI CAT - tang CANVAS]" if clipped else ""))
            rendered += 1
    print(f"Da render {rendered} so do vao {FIGURES}")


if __name__ == "__main__":
    main()
