"""Normalize the SMS Spam Collection CSV and create deterministic stratified splits."""

from __future__ import annotations

import argparse
import csv
import io
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "spam.csv"
SPLIT_NAMES = ("train", "val", "test")
SPLIT_RATIOS = (0.70, 0.15, 0.15)
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MONEY_RE = re.compile(
    r"([$£€])\s*\d[\d,.]*|\b\d[\d,.]*\s?(?:usd|gbp|eur|dollars?|pounds?|euros?)\b",
    re.IGNORECASE,
)
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d ().-]{5,}\d)(?!\w)")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = URL_RE.sub(" <URL> ", text)

    def replace_money(match: re.Match[str]) -> str:
        symbol = match.group(1) or ""
        return f" {symbol}<MONEY> " if symbol else " <MONEY> "

    text = MONEY_RE.sub(replace_money, text)
    text = PHONE_RE.sub(" <PHONE> ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_rows(path: Path) -> list[dict[str, str]]:
    try:
        content = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        content = path.read_text(encoding="cp1252")
    sample = content[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
    except csv.Error:
        dialect = csv.excel
    rows = list(csv.reader(io.StringIO(content, newline=""), dialect))

    if not rows:
        raise ValueError(f"Dataset is empty: {path}")

    first = [cell.strip().lower() for cell in rows[0]]
    has_header = any(value in {"v1", "label", "class", "category"} for value in first)
    if has_header:
        headers = first
        label_index = next((i for i, value in enumerate(headers) if value in {"v1", "label", "class", "category"}), None)
        text_index = next((i for i, value in enumerate(headers) if value in {"v2", "text", "message", "sms"}), None)
        if label_index is None or text_index is None:
            raise ValueError("CSV header must identify label and message columns (for example v1, v2).")
        data_rows = rows[1:]
    else:
        if len(rows[0]) < 2:
            raise ValueError("Expected a label column and a message column.")
        label_index, text_index = 0, 1
        data_rows = rows

    records: list[dict[str, str]] = []
    for line_number, row in enumerate(data_rows, start=2 if has_header else 1):
        if len(row) <= max(label_index, text_index):
            raise ValueError(f"Malformed CSV row {line_number}: expected label and message columns.")
        label = row[label_index].strip().lower()
        if label not in {"ham", "spam"}:
            raise ValueError(f"Unsupported label {row[label_index]!r} on row {line_number}; expected ham or spam.")
        raw_text = row[text_index].strip()
        if not raw_text:
            raise ValueError(f"Empty message on row {line_number}.")
        records.append({"label": label, "raw_text": raw_text, "text": normalize_text(raw_text)})

    if not records:
        raise ValueError("Dataset has no usable rows.")
    return records


def stratified_split(records: list[dict[str, str]], seed: int) -> dict[str, list[dict[str, str]]]:
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        by_label[record["label"]].append(record)

    splits = {name: [] for name in SPLIT_NAMES}
    rng = random.Random(seed)
    for label in sorted(by_label):
        examples = by_label[label]
        rng.shuffle(examples)
        test_count = round(len(examples) * SPLIT_RATIOS[2])
        val_count = round(len(examples) * SPLIT_RATIOS[1])
        boundaries = (len(examples) - test_count - val_count, len(examples) - test_count)
        chunks = (examples[:boundaries[0]], examples[boundaries[0]:boundaries[1]], examples[boundaries[1]:])
        for name, chunk in zip(SPLIT_NAMES, chunks):
            splits[name].extend(chunk)

    for split in splits.values():
        rng.shuffle(split)
    return splits


def write_csv(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=("label", "raw_text", "text"))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Kaggle spam.csv or equivalent CSV")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Replace existing splits and unlock a new test split")
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Dataset not found: {args.input}. Place Kaggle's spam.csv in data/raw/ or pass --input.")

    split_paths = [args.output_dir / f"{name}.csv" for name in SPLIT_NAMES]
    existing = [path for path in split_paths if path.exists()]
    if existing and not args.force:
        parser.error(f"Split already exists ({existing[0]}). Use --force only when intentionally creating a new split.")

    records = load_rows(args.input)
    splits = stratified_split(records, args.seed)
    for (name, split), path in zip(splits.items(), split_paths):
        write_csv(path, split)
        counts = {label: sum(row["label"] == label for row in split) for label in ("ham", "spam")}
        print(f"{name}: {len(split)} rows ({counts['ham']} ham, {counts['spam']} spam)")
    print(f"Prepared {len(records)} rows from {args.input}; seed={args.seed}; ratios=70/15/15.")


if __name__ == "__main__":
    main()
