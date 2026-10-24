"""Benchmark the three scratch models on the locked test split."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.common import KINDS, load_checkpoint, predict_probabilities, read_split, transform_texts


def calculate_metrics(labels: list[int], probabilities: list[float], threshold: float = 0.5) -> dict:
    predictions = [int(probability >= threshold) for probability in probabilities]
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "spam_precision": float(precision_score(labels, predictions, zero_division=0)),
        "spam_recall": float(recall_score(labels, predictions, zero_division=0)),
        "spam_f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if len(set(labels)) == 2 else None,
        "confusion_matrix_labels": ["ham", "spam"],
        "confusion_matrix": matrix.tolist(),
    }


def benchmark_prediction(model, encoder, kind: str, texts: list[str], max_len: int | None, batch_size: int):
    start = time.perf_counter()
    probabilities = []
    for offset in range(0, len(texts), batch_size):
        features = transform_texts(kind, encoder, texts[offset:offset + batch_size], max_len)
        probabilities.extend(predict_probabilities(model, features, batch_size))
    elapsed = time.perf_counter() - start
    return probabilities, round(elapsed * 1000 / len(texts), 4)


def evaluate(test_path: Path, checkpoint_root: Path) -> dict:
    torch.set_num_threads(min(4, torch.get_num_threads()))
    texts, labels = read_split(test_path)
    results = {}
    for kind in KINDS:
        directory = checkpoint_root / kind / "checkpoint"
        if not (directory / "config.json").is_file():
            raise FileNotFoundError(f"Missing {kind} checkpoint: run models/{kind}/train.py first")
        model, encoder, config = load_checkpoint(kind, directory)
        max_len = None if kind == "mlp" else config["max_len"]
        # Warm up model, tokenizer, and CPU kernels; exclude this pass from latency.
        benchmark_prediction(model, encoder, kind, texts[:min(32, len(texts))], max_len, 32)
        probabilities, latency_1 = benchmark_prediction(model, encoder, kind, texts, max_len, 1)
        _, latency_32 = benchmark_prediction(model, encoder, kind, texts, max_len, 32)
        metrics = calculate_metrics(labels.astype(int).tolist(), probabilities, config["decision_threshold"])
        metrics.update({"latency_ms_batch_1": latency_1, "latency_ms_batch_32": latency_32,
                        "parameters": config["parameters"], "train_seconds": config["train_seconds"],
                        "config": {key: config[key] for key in ("input_type", "vocab_size", "max_len",
                                                         "embedding_dim", "hidden_dim", "optimizer",
                                                         "epochs_trained", "best_epoch", "seed",
                                                         "decision_threshold")}})
        results[kind] = metrics
    try:
        test_file = test_path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        test_file = str(test_path)
    return {"dataset": {"test_file": test_file, "examples": len(texts)}, "models": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, default=ROOT / "data/splits/test.csv")
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "models")
    parser.add_argument("--output", type=Path, default=ROOT / "eval/results/evaluation.json")
    args = parser.parse_args()
    payload = evaluate(args.test, args.checkpoint_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({kind: payload["models"][kind]["spam_f1"] for kind in KINDS}, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
