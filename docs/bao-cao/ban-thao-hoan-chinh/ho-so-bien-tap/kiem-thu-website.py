"""Exercise every route branch of app.py with Flask's test client and record the outcome."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parents[3]
sys.path.insert(0, str(ROOT))
import app as website  # noqa: E402

CASES = [
    ("GET", None),
    ("POST spam", {"message": "URGENT: Your bank account is locked. Verify now at secure-bank-login.com/verify"}),
    ("POST ham", {"message": "Are we still meeting for lunch today?"}),
    ("POST empty", {"message": "   "}),
    ("POST missing field", {}),
    ("POST too long", {"message": "a" * 5001}),
]


def main() -> None:
    client = website.app.test_client()
    records = []
    for name, form in CASES:
        response = client.get("/") if form is None else client.post("/", data=form)
        html = response.get_data(as_text=True)
        label = re.search(r"\b(SPAM|HAM)\b(?=[^<]*</)", html)
        records.append({
            "case": name, "status": response.status_code,
            "shows_empty_error": "Vui lòng nhập nội dung tin nhắn." in html,
            "shows_length_error": "Tin nhắn quá dài" in html,
            "classify_result": website.classify_message(form["message"]) if form and form.get("message", "").strip()
            and len(form["message"].strip()) <= 5000 else None,
            "label_in_html": label.group(1) if label else None,
        })
    payload = {"model_kind": website.MODEL_KIND, "model_dir": str(website.MODEL_DIR.relative_to(ROOT)),
               "cases": records}
    out = AUDIT / "kiem-thu-website.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
