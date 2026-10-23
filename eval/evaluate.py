"""Evaluate the standalone Transformer and the Tier-1/Tier-2 cascade."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRANSFORMER_MODEL = "mrm8488/bert-tiny-finetuned-sms-spam-detection"
sys.path.insert(0, str(PROJECT_ROOT))


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def display_model(model: str) -> str:
    path = Path(model)
    return display_path(path) if path.exists() else model


def read_split(path: Path) -> list[dict[str, str | int]]:
    examples: list[dict[str, str | int]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not {"text", "label"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain label and text columns.")
        for row_number, row in enumerate(reader, start=2):
            label = row["label"].strip().lower()
            if label not in {"ham", "spam"}:
                raise ValueError(f"Unsupported label {label!r} on row {row_number}.")
            raw_text = row.get("raw_text") or row["text"]
            examples.append({"label": int(label == "spam"), "raw_text": raw_text, "text": row["text"]})
    if not examples:
        raise ValueError(f"No examples in {path}.")
    return examples


def resolve_spam_label_id(model, requested: int | None) -> int:
    if requested is not None:
        if requested < 0 or requested >= model.config.num_labels:
            raise ValueError(f"--spam-label-id must be between 0 and {model.config.num_labels - 1}.")
        return requested
    for label_id, name in model.config.id2label.items():
        if str(name).strip().lower() == "spam":
            return int(label_id)
    raise ValueError("Checkpoint does not identify its spam label. Pass --spam-label-id after confirming label order.")


def predict_transformer(model, tokenizer, texts: list[str], spam_label_id: int, torch, batch_size: int, max_length: int):
    probabilities: list[float] = []
    started = time.perf_counter()
    model.eval()
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(
            texts[start:start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.inference_mode():
            logits = model(**batch).logits
            scores = torch.softmax(logits, dim=-1)[:, spam_label_id]
        probabilities.extend(float(score) for score in scores.tolist())
    return probabilities, time.perf_counter() - started


def calculate_metrics(y_true: list[int], y_pred: list[int], probabilities: list[float], elapsed: float) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    matrix = confusion_matrix(y_true, y_pred, labels=(0, 1))
    ham_total = int(matrix[0].sum())
    return {
        "examples": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "spam_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "spam_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "spam_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate_ham": float(matrix[0, 1] / ham_total) if ham_total else None,
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if len(set(y_true)) == 2 else None,
        "confusion_matrix_labels": ["ham", "spam"],
        "confusion_matrix": matrix.tolist(),
        "avg_latency_ms": elapsed * 1000 / len(y_true) if y_true else 0.0,
        "classification_report": classification_report(
            y_true, y_pred, labels=(0, 1), target_names=("ham", "spam"), output_dict=True, zero_division=0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, default=PROJECT_ROOT / "data" / "splits" / "test.csv")
    parser.add_argument("--model", default=DEFAULT_TRANSFORMER_MODEL, help="Hugging Face model ID or local checkpoint path")
    parser.add_argument("--rules", type=Path, default=PROJECT_ROOT / "tier1_rules.yaml")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "eval" / "results" / "evaluation.json")
    parser.add_argument("--spam-label-id", type=int, default=1, help="Spam output index; default 1 was validated on val.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    if args.batch_size < 1 or args.max_length < 1:
        parser.error("--batch-size and --max-length must be positive.")
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from tier1.cascade import load_rules, route_message
        from tier1.model import Tier1Model
    except ImportError as error:
        parser.error(f"Missing evaluation dependency ({error.name}); run eval/setup_env.py first.")

    examples = read_split(args.test)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    spam_label_id = resolve_spam_label_id(model, args.spam_label_id)
    y_true = [int(row["label"]) for row in examples]
    normalized_texts = [str(row["text"]) for row in examples]
    transformer_probabilities, transformer_elapsed = predict_transformer(
        model, tokenizer, normalized_texts, spam_label_id, torch, args.batch_size, args.max_length
    )
    transformer_predictions = [int(probability >= 0.5) for probability in transformer_probabilities]
    transformer_metrics = calculate_metrics(y_true, transformer_predictions, transformer_probabilities, transformer_elapsed)
    transformer_metrics["spam_label_id"] = spam_label_id

    rules = load_rules(args.rules)
    tier1_model = Tier1Model.from_pretrained()
    tier1_started = time.perf_counter()
    raw_texts = [str(row["raw_text"]) for row in examples]
    tier1_scores = tier1_model.predict_many(raw_texts)
    tier1_probabilities = [probability for _, probability in tier1_scores]
    decisions = [
        route_message(text, score, rules)
        for text, (score, _) in zip(raw_texts, tier1_scores)
    ]
    tier1_elapsed = time.perf_counter() - tier1_started

    route_counts = {
        "tier1_spam": sum(decision.tier == "tier1" and decision.label == "spam" for decision in decisions),
        "tier1_ham": sum(decision.tier == "tier1" and decision.label == "ham" for decision in decisions),
        "tier2": sum(decision.tier == "tier2" for decision in decisions),
    }
    decided_indices = [index for index, decision in enumerate(decisions) if decision.tier == "tier1"]
    tier1_predictions = [int(decisions[index].label == "spam") for index in decided_indices]
    if decided_indices:
        tier1_metrics = calculate_metrics(
            [y_true[index] for index in decided_indices],
            tier1_predictions,
            [tier1_probabilities[index] for index in decided_indices],
            tier1_elapsed * len(decided_indices) / len(examples),
        )
    else:
        tier1_metrics = {
            "examples": 0,
            "accuracy": None,
            "spam_precision": None,
            "spam_recall": None,
            "spam_f1": None,
            "false_positive_rate_ham": None,
            "roc_auc": None,
            "confusion_matrix_labels": ["ham", "spam"],
            "confusion_matrix": [[0, 0], [0, 0]],
            "avg_latency_ms": 0.0,
        }
    tier1_metrics["coverage"] = len(decided_indices) / len(examples)
    tier1_metrics["route_counts"] = route_counts
    tier1_metrics["score_method"] = "log-odds of calibrated spam probability"

    tier2_indices = [index for index, decision in enumerate(decisions) if decision.tier == "tier2"]
    cascade_probabilities = tier1_probabilities[:]
    tier2_elapsed = 0.0
    if tier2_indices:
        tier2_probabilities, tier2_elapsed = predict_transformer(
            model,
            tokenizer,
            [normalized_texts[index] for index in tier2_indices],
            spam_label_id,
            torch,
            args.batch_size,
            args.max_length,
        )
        for index, probability in zip(tier2_indices, tier2_probabilities):
            cascade_probabilities[index] = probability
    cascade_predictions = [
        int(decision.label == "spam") if decision.tier == "tier1" else int(cascade_probabilities[index] >= 0.5)
        for index, decision in enumerate(decisions)
    ]
    cascade_metrics = calculate_metrics(
        y_true, cascade_predictions, cascade_probabilities, tier1_elapsed + tier2_elapsed
    )
    cascade_metrics["tier2_rate"] = len(tier2_indices) / len(examples)
    cascade_metrics["route_counts"] = route_counts

    payload = {
        "dataset": {"test_file": display_path(args.test), "examples": len(examples)},
        "models": {
            "transformer": display_model(args.model),
            "tier1": "jngb-labs/sms-spam-classical",
            "tier1_score_method": tier1_metrics["score_method"],
        },
        "transformer": transformer_metrics,
        "tier1": tier1_metrics,
        "cascade": cascade_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "examples": len(examples),
        "transformer_spam_f1": transformer_metrics["spam_f1"],
        "tier1_coverage": tier1_metrics["coverage"],
        "tier2_rate": cascade_metrics["tier2_rate"],
        "cascade_spam_f1": cascade_metrics["spam_f1"],
        "cascade_avg_latency_ms": cascade_metrics["avg_latency_ms"],
        "routes": route_counts,
    }, indent=2))
    print(f"Saved evaluation results to {args.output}")


if __name__ == "__main__":
    main()
