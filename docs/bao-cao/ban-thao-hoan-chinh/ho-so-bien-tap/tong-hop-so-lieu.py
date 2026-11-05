"""Collect every number the updated report cites into one JSON file (read-only on project data)."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parents[3]
sys.path.insert(0, str(ROOT))
from data.prepare_data import load_rows, summarize_splits  # noqa: E402

SPLITS = ROOT / "data/splits"


def read(name: str) -> list[dict[str, str]]:
    with (SPLITS / name).open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def counts(rows):
    labels = Counter(row["label"] for row in rows)
    return {"rows": len(rows), "ham": labels["ham"], "spam": labels["spam"]}


raw = load_rows(ROOT / "data/raw/spam.csv")
files = ("train.csv", "train_base.csv", "val.csv", "test.csv", "test_smishing.csv", "test_external.csv")
split_rows = {name: read(name) for name in files}
results = {}
for run in ("baseline", "augmented"):
    for test in ("test", "test_smishing", "test_external"):
        results[f"{run}_{test}"] = json.loads((ROOT / f"eval/results/{run}_{test}.json").read_text(encoding="utf-8"))
configs = {}
for kind in ("mlp", "lstm", "transformer"):
    configs[f"baseline_{kind}"] = json.loads((ROOT / f"eval/results/baseline/{kind}/config.json").read_text(encoding="utf-8"))
    configs[f"augmented_{kind}"] = json.loads((ROOT / f"models/{kind}/checkpoint/config.json").read_text(encoding="utf-8"))

payload = {
    "raw": counts(raw),
    "splits": {name: counts(rows) for name, rows in split_rows.items()},
    "overlap_base_splits": summarize_splits({"train": split_rows["train_base.csv"],
                                             "val": split_rows["val.csv"], "test": split_rows["test.csv"]})["overlap"],
    "augmentation_manifest": json.loads((SPLITS / "augmentation_manifest.json").read_text(encoding="utf-8")),
    "configs": configs,
    "results": results,
}
out = AUDIT / "tong-hop-so-lieu.json"
out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Saved {out}")
