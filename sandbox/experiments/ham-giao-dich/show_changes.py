import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for name in ("train_base", "val", "test", "test_external"):
    old = list(csv.DictReader((ROOT / "data/splits" / f"{name}.csv").open(encoding="utf-8-sig")))
    new = list(csv.DictReader((HERE / "data/splits" / f"{name}.csv").open(encoding="utf-8-sig")))
    for a, b in zip(old, new):
        if a["text"] != b["text"]:
            print(f"[{name}] {b['label']}: {b['text'][:150]}")
