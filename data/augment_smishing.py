"""Re-normalize locked splits and add the africa-smishing dataset to train plus a separate test set.

Membership and labels of train/val/test are never changed; only the derived `text` column is
recomputed from `raw_text` with the current normalize_text. New rows are deduplicated by
normalized text, rows that collide with any existing split or carry conflicting labels are
dropped, and the remaining groups are split per label into train (85%) and test_smishing (15%).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from prepare_data import PROJECT_ROOT, SPLIT_NAMES, normalize_text, write_csv

SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
MANIFEST = SPLIT_DIR / "augmentation_manifest.json"
TEST_FRACTION = 0.15


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return [{"label": row["label"], "raw_text": row["raw_text"], "text": row["text"]}
                for row in csv.DictReader(source)]


def renormalize(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[tuple[str, str]]]:
    updated, changed = [], []
    for row in rows:
        text = normalize_text(row["raw_text"])
        if text != row["text"]:
            changed.append((row["text"], text))
        updated.append({**row, "text": text})
    return updated, changed


def load_new_rows(path: Path) -> list[dict[str, str]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            item = json.loads(line)
            label = {0: "ham", 1: "spam"}.get(item.get("label"))
            raw = (item.get("sms_text") or "").strip()
            if label is None or not raw:
                raise ValueError(f"Invalid label or empty text at {path}:{line_number}")
            rows.append({"label": label, "raw_text": raw, "text": normalize_text(raw)})
    return rows


def template_key(text: str) -> str:
    """Synthetic rows differ mostly by IDs, balances and dates; mask digits to group templates."""
    return re.sub(r"\d", "0", text)


def overlap_counts(splits: dict[str, list[dict[str, str]]]) -> dict[str, int]:
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    return {f"{left}:{right}": len({r["text"] for r in splits[left]} & {r["text"] for r in splits[right]})
            for left, right in pairs}


def group_split(groups: list[list[dict[str, str]]], seed: int) -> tuple[list, list]:
    """Deterministically move whole groups to test until each label reaches its target share."""
    train, test = [], []
    by_label: dict[str, list[list[dict[str, str]]]] = defaultdict(list)
    for group in groups:
        by_label[group[0]["label"]].append(group)
    for label in sorted(by_label):
        ordered = sorted(by_label[label], key=lambda group: hashlib.sha256(
            f"{seed}:{template_key(group[0]['text'])}".encode("utf-8")).hexdigest())
        target = round(sum(map(len, ordered)) * TEST_FRACTION)
        taken = 0
        for group in ordered:
            if taken + len(group) <= target:
                test.extend(group)
                taken += len(group)
            else:
                train.extend(group)
    return train, test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data/raw/africa_smish_raw.jsonl",
                        help="africa-smishing JSONL (sms_text, label) from data/fetch_africa_smishing.py")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--apply", action="store_true", help="Write files; default is a dry run")
    args = parser.parse_args()

    if MANIFEST.exists():
        parser.error(f"{MANIFEST} exists: augmentation already applied. Restore the backup first.")

    originals, splits, changes = {}, {}, {}
    for name in SPLIT_NAMES:
        originals[name] = read_rows(SPLIT_DIR / f"{name}.csv")
        splits[name], changes[name] = renormalize(originals[name])
    existing = {template_key(row["text"]) for rows in splits.values() for row in rows}

    new_rows = load_new_rows(args.input)
    # Group by digit-masked template so near-identical rows never straddle train and test.
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in new_rows:
        groups[template_key(row["text"])].append(row)
    conflicting = [key for key, rows in groups.items() if len({r["label"] for r in rows}) > 1]
    overlapping = [key for key in groups if key in existing]
    dropped = set(conflicting) | set(overlapping)
    # Keep the distinct normalized texts of each template group; exact duplicates add nothing.
    kept = []
    for key, rows in sorted(groups.items()):
        if key not in dropped:
            kept.append(list({row["text"]: row for row in rows}.values()))
    aug_train, aug_test = group_split(kept, args.seed)

    summary = {
        "seed": args.seed, "source": str(args.input), "source_rows": len(new_rows),
        "source_labels": dict(Counter(r["label"] for r in new_rows)),
        "unique_normalized_texts": len({r["text"] for r in new_rows}),
        "template_groups": len(groups), "dropped_conflicting_label_groups": len(conflicting),
        "dropped_overlap_with_splits_groups": len(overlapping),
        "added_train": dict(Counter(r["label"] for r in aug_train)),
        "test_smishing": dict(Counter(r["label"] for r in aug_test)),
        "test_smishing_template_groups": len({template_key(r["text"]) for r in aug_test}),
        "renormalized_rows_changed": {name: len(rows) for name, rows in changes.items()},
        "cross_split_text_overlap_before_renormalize": overlap_counts(originals),
        "cross_split_text_overlap_after_renormalize": overlap_counts(splits),
    }
    print(json.dumps(summary, indent=2))
    for name in SPLIT_NAMES:
        for old, new in changes[name][:5]:
            print(f"[{name}] {old!r}\n    -> {new!r}")
    if not args.apply:
        print("Dry run: no files written. Pass --apply to write.")
        return

    backup = SPLIT_DIR / f"backup_{datetime.now():%Y%m%d_%H%M%S}"
    backup.mkdir()
    for name in SPLIT_NAMES:
        shutil.copy2(SPLIT_DIR / f"{name}.csv", backup / f"{name}.csv")
    write_csv(SPLIT_DIR / "train_base.csv", splits["train"])
    write_csv(SPLIT_DIR / "train.csv", splits["train"] + aug_train)
    write_csv(SPLIT_DIR / "val.csv", splits["val"])
    write_csv(SPLIT_DIR / "test.csv", splits["test"])
    write_csv(SPLIT_DIR / "test_smishing.csv", aug_test)
    summary["backup_dir"] = backup.relative_to(PROJECT_ROOT).as_posix()
    MANIFEST.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Backed up original splits to {backup}; wrote train/train_base/val/test/test_smishing and manifest.")


if __name__ == "__main__":
    main()
