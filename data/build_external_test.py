"""Build an external test set from the Mendeley SMS phishing dataset, excluding anything near UCI.

Source: Mishra & Soni (2022), "SMS Phishing Dataset for Machine Learning and Pattern Recognition",
Mendeley Data, DOI 10.17632/f45bkkt8pr.1, CC BY 4.0. Most rows copy the UCI SMS Spam Collection,
so rows that match any UCI message exactly or approximately are dropped. `smishing` maps to `spam`.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import io
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

from prepare_data import DEFAULT_INPUT, PROJECT_ROOT, load_rows, normalize_text, write_csv

NEAR_CUTOFF = 0.8
MAX_BROKEN_CHARS = 2


def letters_key(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def read_mendeley(path: Path) -> list[dict[str, str]]:
    """Read Dataset_5971.csv directly or from the Mendeley zip that contains it."""
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            name = next(item for item in archive.namelist() if item.lower().endswith(".csv"))
            data = archive.read(name)
    else:
        data = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return list(csv.DictReader(io.StringIO(data.decode(encoding), newline="")))
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data/raw/mendeley_sms_phishing_5971.zip",
                        help="Mendeley Dataset_5971.zip or the extracted Dataset_5971.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/splits/test_external.csv")
    args = parser.parse_args()

    uci_raw = load_rows(DEFAULT_INPUT)
    uci_keys = sorted({letters_key(row["raw_text"]) for row in uci_raw})
    uci_key_set = set(uci_keys)
    uci_texts = {row["text"] for row in uci_raw}

    reasons = Counter()
    kept: dict[str, dict[str, str]] = {}
    for row in read_mendeley(args.input):
        label = {"ham": "ham", "spam": "spam", "smishing": "spam"}.get(row["LABEL"].strip().lower())
        raw = (row["TEXT"] or "").strip()
        key = letters_key(raw)
        if label is None or len(key) < 10:
            reasons["invalid_or_too_short"] += 1
        elif raw.count("�") > MAX_BROKEN_CHARS:
            reasons["broken_encoding"] += 1
        elif normalize_text(raw) in uci_texts or key in uci_key_set:
            reasons["exact_uci"] += 1
        elif difflib.get_close_matches(key, uci_keys, n=1, cutoff=NEAR_CUTOFF):
            reasons["near_uci"] += 1
        elif key in kept:
            reasons["duplicate_within" if kept[key]["label"] == label else "conflicting_label"] += 1
            if kept[key]["label"] != label:
                kept[key]["label"] = "conflict"
        else:
            kept[key] = {"label": label, "raw_text": raw, "text": normalize_text(raw),
                         "source_label": row["LABEL"].strip().lower()}
    rows = [row for row in kept.values() if row["label"] != "conflict"]
    write_csv(args.output, [{k: row[k] for k in ("label", "raw_text", "text")} for row in rows])
    print(json.dumps({"dropped": dict(reasons), "kept": dict(Counter(r["label"] for r in rows)),
                      "kept_by_source_label": dict(Counter(r["source_label"] for r in rows)),
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
