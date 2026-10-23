"""Evaluate the local BERT SMS classifier on the locked test split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_split(path: Path) -> tuple[list[str], list[int]]:
    texts: list[str] = []
    labels: list[int] = []
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not {"text", "label"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain label and text columns.")
        for row_number, row in enumerate(reader, start=2):
            label = row["label"].strip().lower()
            if label not in {"ham", "spam"}:
                raise ValueError(f"Unsupported label {label!r} on row {row_number}.")
            texts.append(row["text"])
            labels.append(int(label == "spam"))
    if not texts:
        raise ValueError(f"No examples in {path}.")
    return texts, labels


def resolve_spam_label_id(model, requested: int | None) -> int:
    if requested is not None:
        if requested < 0 or requested >= model.config.num_labels:
            raise ValueError(f"--spam-label-id must be between 0 and {model.config.num_labels - 1}.")
        return requested
    for label_id, name in model.config.id2label.items():
        if str(name).strip().lower() == "spam":
            return int(label_id)
    raise ValueError("Checkpoint does not identify its spam label. Pass --spam-label-id after confirming label order.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, default=PROJECT_ROOT / "data" / "splits" / "test.csv")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "transformer" / "bert-tiny-sms-spam")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "eval" / "results" / "transformer.json")
    parser.add_argument("--spam-label-id", type=int, help="Required when the checkpoint has no explicit spam label mapping")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    if args.batch_size < 1 or args.max_length < 1:
        parser.error("--batch-size and --max-length must be positive.")

    try:
        import torch
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as error:
        parser.error(f"Missing evaluation dependency ({error.name}); install requirements.txt first.")

    texts, y_true = read_split(args.test)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(args.model), local_files_only=True)
    spam_label_id = resolve_spam_label_id(model, args.spam_label_id)
    probabilities: list[float] = []
    model.eval()
    for start in range(0, len(texts), args.batch_size):
        batch = tokenizer(
            texts[start:start + args.batch_size],
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        with torch.inference_mode():
            logits = model(**batch).logits
            batch_probabilities = torch.softmax(logits, dim=-1)[:, spam_label_id]
        probabilities.extend(batch_probabilities.tolist())
    y_pred = [int(probability >= 0.5) for probability in probabilities]

    matrix = confusion_matrix(y_true, y_pred, labels=(0, 1))
    metrics = {
        "model": str(args.model),
        "test_file": str(args.test),
        "examples": len(y_true),
        "spam_label_id": spam_label_id,
        "threshold": 0.5,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "spam_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "spam_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "spam_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if len(set(y_true)) == 2 else None,
        "confusion_matrix_labels": ["ham", "spam"],
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=(0, 1), target_names=("ham", "spam"), output_dict=True, zero_division=0
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: metrics[key] for key in ("examples", "accuracy", "spam_precision", "spam_recall", "spam_f1", "roc_auc", "confusion_matrix")}, indent=2))
    print(f"Saved full report to {args.output}")


if __name__ == "__main__":
    main()
