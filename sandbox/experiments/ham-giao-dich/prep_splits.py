"""Re-normalize the locked splits with normalize_v2 (rows, labels and raw_text unchanged).

Writes into this experiment's data/splits/; the repo's data/splits/ is never touched.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from normalize_v2 import normalize_text  # noqa: E402

OUT = HERE / "data" / "splits"


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=("label", "raw_text", "text"), lineterminator="\n")
        w.writeheader()
        w.writerows({k: r[k] for k in ("label", "raw_text", "text")} for r in data)


def main() -> None:
    for name in ("train_base", "val", "test", "test_external", "test_smishing"):
        data = rows(ROOT / "data" / "splits" / f"{name}.csv")
        changed = codes = 0
        for r in data:
            new = normalize_text(r["raw_text"])
            changed += new != r["text"]
            codes += "<CODE>" in new
            r["text"] = new
        write(OUT / f"{name}.csv", data)
        print(f"{name}: rows={len(data)} text_changed={changed} with_CODE={codes}")
    base = rows(OUT / "train_base.csv")
    extra = rows(HERE / "data" / "ham_extra.csv")
    for r in extra:
        r["text"] = normalize_text(r["raw_text"])
    held = {r["text"] for n in ("val", "test", "test_external", "test_smishing") for r in rows(OUT / f"{n}.csv")}
    leaked = sum(r["text"] in held for r in extra)
    write(OUT / "train_ham.csv", base + extra)
    print(f"train_ham: {len(base)} base + {len(extra)} ham_extra; extra rows matching held-out text={leaked}")


if __name__ == "__main__":
    main()
