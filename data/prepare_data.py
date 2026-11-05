"""Normalize SMS data and split duplicate groups with deterministic class balance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "spam.csv"
SPLIT_NAMES = ("train", "val", "test")
SPLIT_RATIOS = (0.70, 0.15, 0.15)
DOMAIN = r"(?<![\w@.-])(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+"
# Bare domains (no scheme/www): any listed TLD when a path follows, a stricter subset otherwise,
# so casual text such as "ok.me" or "e.g" is left alone.
URL_RE = re.compile(
    r"https?://[^\s<>]+|www\.[^\s<>]+"
    rf"|{DOMAIN}(?:com|net|org|co|uk|me|ly|io|info|biz|xyz|top|online|site|link|gl|ng|ke|za|gh|tz|ug)/[^\s<>]*"
    rf"|{DOMAIN}(?:com|net|org|info|biz|co\.uk)(?![\w-])",
    re.IGNORECASE,
)
AMOUNT = r"\d+(?:[,.]\d+)*"
# ISO 4217 codes seen in SMS data (UK/US/EU plus African mobile-money currencies).
CURRENCY_CODES = r"(?:usd|gbp|eur|ngn|kes|ghs|zar|tzs|ugx|rwf|xof|xaf|zmw|mzn|etb)"
MONEY_RE = re.compile(
    rf"([$£€])\s*{AMOUNT}(?:[km])?(?!\w|[,.]\d)"
    rf"|(?<![\w.,]){CURRENCY_CODES}\s?{AMOUNT}(?:[km])?(?!\w|[,.]\d)"
    rf"|(?<![\w.,]){AMOUNT}\s*(?:{CURRENCY_CODES}|dollars?|pounds?|euros?|quid)\b"
    r"|(?<![\w.,])\d{1,3}(?:\.\d{1,2})?\s*(?:pence|p)\b",
    re.IGNORECASE,
)
# UK phone numbers have 11 domestic digits; restrict the attached rate to 1-3 digits.
PHONE_RATE_RE = re.compile(r"(?<!\w)0[1-9]\d{9}\d{1,3}(?:\.\d{1,2})?p(?=\s*(?:/|per\b))")
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d{7,15}|\+?\d[\d ().-]{5,}\d)(?!\w)")
DATE_RE = re.compile(r"(?:\d{4}[-.]\d{1,2}[-.]\d{1,2}|\d{1,2}[-.]\d{1,2}[-.]\d{2,4})")


def replace_url(match: re.Match[str]) -> str:
    url, suffix = match.group(), ""
    closing = {")": "(", "]": "[", "}": "{"}
    while url:
        last = url[-1]
        if last in ".,!?;:$%\"'" or (last in closing and url.count(last) > url.count(closing[last])):
            suffix = last + suffix
            url = url[:-1]
        else:
            break
    return f" <URL> {suffix}"


def replace_phone(match: re.Match[str]) -> str:
    candidate = match.group()
    digits = sum(character.isdigit() for character in candidate)
    if 7 <= digits <= 15 and not DATE_RE.fullmatch(re.sub(r"\s+", "", candidate)):
        return " <PHONE> "
    return candidate


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("\u00e5\u00a3", "\u00a3")
    text = URL_RE.sub(replace_url, text)
    text = PHONE_RATE_RE.sub(" <PHONE> <MONEY> ", text)

    def replace_money(match: re.Match[str]) -> str:
        symbol = match.group(1) or ""
        return f" {symbol}<MONEY> " if symbol else " <MONEY> "

    text = MONEY_RE.sub(replace_money, text)
    text = PHONE_RE.sub(replace_phone, text)
    text = re.sub(r"<(url|phone|money)>", lambda match: match.group().upper(), text)
    return re.sub(r"\s+", " ", text).strip()


def load_rows(path: Path) -> list[dict[str, str]]:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            with path.open("r", encoding=encoding, newline="") as source:
                content = source.read()
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Cannot decode dataset as UTF-8 or cp1252: {path}")
    sample = content[:4096]
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",\t;").delimiter
    except csv.Error:
        delimiter = ","
    rows = list(csv.reader(io.StringIO(content, newline=""), delimiter=delimiter, doublequote=True))

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
    """Keep each normalized-text group intact; allocate larger groups first."""
    if not records:
        raise ValueError("Cannot split an empty dataset")
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        groups[record["text"]].append(record)

    by_label: dict[str, list[list[dict[str, str]]]] = defaultdict(list)
    for group in groups.values():
        labels = {record["label"] for record in group}
        if len(labels) != 1:
            raise ValueError("A normalized-text duplicate group has conflicting labels; review it before splitting.")
        by_label[group[0]["label"]].append(group)

    splits = {name: [] for name in SPLIT_NAMES}
    rng = random.Random(seed)
    for label in sorted(by_label):
        label_groups = sorted(by_label[label], key=lambda group: (
            -len(group), hashlib.sha256(f"{seed}:{group[0]['text']}".encode("utf-8")).hexdigest()))
        total = sum(len(group) for group in label_groups)
        test_count = round(total * SPLIT_RATIOS[2])
        val_count = round(total * SPLIT_RATIOS[1])
        targets = (total - test_count - val_count, val_count, test_count)
        active = [index for index, target in enumerate(targets) if target > 0]
        if len(label_groups) < len(active):
            raise ValueError(f"Not enough distinct {label} groups to populate the requested splits")
        counts = [0, 0, 0]
        for position, group in enumerate(label_groups):
            empty = [index for index in active if counts[index] == 0]
            candidates = empty if len(label_groups) - position == len(empty) else active
            # Minimize the increase in squared class-count error, scaled by each target.
            destination = min(candidates, key=lambda index: (
                2 * counts[index] + len(group) - 2 * targets[index]) / targets[index])
            splits[SPLIT_NAMES[destination]].extend(group)
            counts[destination] += len(group)

    for split in splits.values():
        rng.shuffle(split)
    return splits


def summarize_splits(splits: dict[str, list[dict[str, str]]]) -> dict:
    """Audit counts and overlap in memory; rows_in_right counts repeated records."""
    summary = {"counts": {}, "overlap": {}}
    for name, rows in splits.items():
        summary["counts"][name] = {"rows": len(rows), **{
            label: sum(row["label"] == label for row in rows) for label in ("ham", "spam")}}
    for field in ("raw_text", "text"):
        pairs = {}
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            seen = {row[field] for row in splits[left]}
            repeated = [row for row in splits[right] if row[field] in seen]
            pairs[f"{left}:{right}"] = {
                "shared_groups": len({row[field] for row in repeated}), "rows_in_right": len(repeated),
                "spam_rows_in_right": sum(row["label"] == "spam" for row in repeated)}
        summary["overlap"][field] = pairs
    return summary


def write_csv(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=("label", "raw_text", "text"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Kaggle spam.csv or equivalent CSV")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Replace existing splits and unlock a new test split")
    parser.add_argument("--dry-run", action="store_true", help="Print duplicate audit and proposed split counts without writing files")
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Dataset not found: {args.input}. Place Kaggle's spam.csv in data/raw/ or pass --input.")

    split_paths = [args.output_dir / f"{name}.csv" for name in SPLIT_NAMES]
    existing = [path for path in split_paths if path.exists()]
    if existing and not args.force and not args.dry_run:
        parser.error(f"Split already exists ({existing[0]}). Use --force only when intentionally creating a new split.")

    records = load_rows(args.input)
    splits = stratified_split(records, args.seed)
    if args.dry_run:
        group_sizes: dict[str, int] = defaultdict(int)
        for record in records:
            group_sizes[record["text"]] += 1
        current = None
        if len(existing) == len(SPLIT_NAMES):
            current = {}
            for name, path in zip(SPLIT_NAMES, split_paths):
                with path.open(encoding="utf-8-sig", newline="") as source:
                    current[name] = list(csv.DictReader(source))
        print(json.dumps({"mode": "dry_run", "seed": args.seed, "group_key": "normalized_text",
                          "source_rows": len(records), "groups": len(group_sizes),
                          "duplicate_groups": sum(size > 1 for size in group_sizes.values()),
                          "duplicate_extra_rows": len(records) - len(group_sizes),
                          "largest_group": max(group_sizes.values()),
                          "current": summarize_splits(current) if current is not None else None,
                          "proposed": summarize_splits(splits)}, indent=2))
        return
    for (name, split), path in zip(splits.items(), split_paths):
        write_csv(path, split)
        counts = {label: sum(row["label"] == label for row in split) for label in ("ham", "spam")}
        print(f"{name}: {len(split)} rows ({counts['ham']} ham, {counts['spam']} spam)")
    print(f"Prepared {len(records)} rows from {args.input}; seed={args.seed}; ratios=70/15/15.")


if __name__ == "__main__":
    main()
