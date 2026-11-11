"""train_mix = train_ham + the africa-smishing rows already in the repo's train.csv (not in train_base)."""

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from normalize_v2 import normalize_text  # noqa: E402
from prep_splits import rows, write  # noqa: E402

base_raw = {r["raw_text"] for r in rows(ROOT / "data/splits/train_base.csv")}
extra = [r for r in rows(ROOT / "data/splits/train.csv") if r["raw_text"] not in base_raw]
for r in extra:
    r["text"] = normalize_text(r["raw_text"])
mix = rows(HERE / "data/splits/train_ham.csv") + extra
write(HERE / "data/splits/train_mix.csv", mix)
print(f"smishing rows={len(extra)} (spam={sum(r['label'] == 'spam' for r in extra)}); train_mix={len(mix)}")
