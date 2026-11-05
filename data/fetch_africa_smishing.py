"""Download electricsheepafrica/africa-smishing-sms-phishing via the Hugging Face datasets-server API.

Writes one JSON object per line with the fields augment_smishing.py needs. Resumes from an existing
partial file (whole 100-row pages only) and retries with backoff on HTTP 429/5xx.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

from prepare_data import PROJECT_ROOT

DATASET = "electricsheepafrica%2Fafrica-smishing-sms-phishing"
ROWS_URL = ("https://datasets-server.huggingface.co/rows?dataset=" + DATASET
            + "&config=default&split=train&offset={offset}&length=100")
SIZE_URL = "https://datasets-server.huggingface.co/size?dataset=" + DATASET
KEEP = ("sms_text", "label", "language", "smishing_type", "is_synthetic")
HEADERS = {"User-Agent": "sms-spam-classification-fetch"}


def get_json(url: str, attempts: int = 6) -> dict:
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as response:
                return json.loads(response.read().decode())
        except Exception as error:  # rate limit or transient network failure
            wait = 5 * 2 ** attempt
            print(f"{url[-40:]} failed ({error}); retry in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"giving up on {url}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/raw/africa_smish_raw.jsonl")
    args = parser.parse_args()

    total = get_json(SIZE_URL)["size"]["dataset"]["num_rows"]
    kept = sum(1 for _ in args.output.open(encoding="utf-8")) if args.output.exists() else 0
    if kept % 100:
        raise SystemExit(f"Partial page in {args.output} ({kept} rows); inspect before resuming")
    with args.output.open("a", encoding="utf-8") as sink:
        for offset in range(kept, total, 100):
            rows = get_json(ROWS_URL.format(offset=offset)).get("rows", [])
            if not rows:
                raise SystemExit(f"Empty page at offset {offset}")
            for item in rows:
                sink.write(json.dumps({key: item["row"].get(key) for key in KEEP}, ensure_ascii=False) + "\n")
            print(f"fetched {offset + len(rows)}/{total}", flush=True)
    print(f"Saved {total} rows to {args.output}")


if __name__ == "__main__":
    main()
