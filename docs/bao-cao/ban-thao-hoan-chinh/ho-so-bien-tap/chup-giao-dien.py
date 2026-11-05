"""Render real app.py responses and screenshot them with headless Microsoft Edge."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parents[3]
OUT = AUDIT.parent / "hinh"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
sys.path.insert(0, str(ROOT))
import app as website  # noqa: E402

PAGES = {
    "website-trang-chu": None,
    "website-ket-qua-spam": "URGENT: Your bank account has been locked. Verify your identity now at secure-bank-login.com/verify",
    "website-ket-qua-ham": "Are we still meeting for lunch today?",
}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    client = website.app.test_client()
    for name, message in PAGES.items():
        response = client.get("/") if message is None else client.post("/", data={"message": message})
        html_path = OUT / f"{name}.html"
        html_path.write_text(response.get_data(as_text=True), encoding="utf-8")
        png_path = OUT / f"{name}.png"
        subprocess.run([str(EDGE), "--headless=new", "--disable-gpu", "--hide-scrollbars",
                        "--window-size=1000,900", f"--screenshot={png_path}", html_path.as_uri()],
                       check=True, capture_output=True, timeout=60)
        html_path.unlink()
        print(name, response.status_code, png_path.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
