"""Compare orig (current repo checkpoints, old normalization) with base / ham (normalize_v2).

Writes results/summary.json and results/summary.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from compare_demo_models import SETS  # noqa: E402
from data.prepare_data import normalize_text as norm_old  # noqa: E402
from eval.evaluate import calculate_metrics  # noqa: E402
from manual_sets import SET_5  # noqa: E402
from models.common import KINDS, load_checkpoint, predict_probabilities, read_split, transform_texts  # noqa: E402
from normalize_v2 import normalize_text as norm_new  # noqa: E402

torch.set_num_threads(4)
CONFIGS = {
    "orig": {"dirs": {"mlp": ROOT / "eval/results/baseline/mlp", "lstm": ROOT / "eval/results/baseline/lstm",
                      "transformer": ROOT / "models/transformer/demo_checkpoint"},
             "splits": ROOT / "data/splits", "norm": norm_old},
    "base": {"dirs": {k: HERE / "checkpoints/base" / k for k in KINDS}, "splits": HERE / "data/splits", "norm": norm_new},
    "ham": {"dirs": {k: HERE / "checkpoints/ham" / k for k in KINDS}, "splits": HERE / "data/splits", "norm": norm_new},
    "mix": {"dirs": {k: HERE / "checkpoints/mix" / k for k in KINDS}, "splits": HERE / "data/splits", "norm": norm_new},
}
TESTS = ("test", "test_external", "test_smishing")
MANUAL = {**{f"set{k}": v for k, v in SETS.items()}, "set5": SET_5}


def scores(model, encoder, kind, config, texts):
    return predict_probabilities(model, transform_texts(kind, encoder, texts, config["max_len"]), 64)


def main() -> None:
    out = {}
    for name, cfg in CONFIGS.items():
        out[name] = {}
        for kind in KINDS:
            model, encoder, config = load_checkpoint(kind, cfg["dirs"][kind])
            res = {}
            for test in TESTS:
                texts, labels = read_split(cfg["splits"] / f"{test}.csv")
                m = calculate_metrics(labels.astype(int).tolist(), scores(model, encoder, kind, config, texts))
                (tn, fp), (fn, tp) = m["confusion_matrix"]
                res[test] = {"f1": round(m["spam_f1"], 4), "fp": fp, "fn": fn}
            total = {"correct": 0, "n": 0, "fp": 0, "fn": 0}
            per = {}
            for set_name, samples in MANUAL.items():
                probs = scores(model, encoder, kind, config, [cfg["norm"](s[2]) for s in samples])
                ok = 0
                for (label, tag, _), p in zip(samples, probs):
                    pred = "spam" if p >= 0.5 else "ham"
                    ok += pred == label
                    total["fp"] += label == "ham" and pred == "spam"
                    total["fn"] += label == "spam" and pred == "ham"
                    per.setdefault("scores", {})[f"{set_name}:{tag}"] = round(p, 3)
                per[set_name] = f"{ok}/{len(samples)}"
                total["correct"] += ok
                total["n"] += len(samples)
            res["manual"] = {**{k: v for k, v in per.items() if k != "scores"}, **total}
            res["manual_scores"] = per["scores"]
            out[name][kind] = res
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results/summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["| Config | Model | test F1 (FP/FN) | external F1 (FP/FN) | smishing F1 | Manual 76 (FP/FN) | " +
             " | ".join(MANUAL) + " |", "|" + "---|" * (6 + len(MANUAL))]
    for name in CONFIGS:
        for kind in KINDS:
            r = out[name][kind]
            mm = r["manual"]
            lines.append(f"| {name} | {kind} | {r['test']['f1']:.4f} ({r['test']['fp']}/{r['test']['fn']}) | "
                         f"{r['test_external']['f1']:.4f} ({r['test_external']['fp']}/{r['test_external']['fn']}) | "
                         f"{r['test_smishing']['f1']:.4f} | {mm['correct']}/{mm['n']} ({mm['fp']}/{mm['fn']}) | " +
                         " | ".join(mm[s] for s in MANUAL) + " |")
    (HERE / "results/summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
