"""Render a comparable Markdown report from benchmark results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def render_report(payload: dict) -> str:
    dataset = payload["dataset"]
    lines = [
        "# So sánh mô hình phân loại SMS spam",
        "",
        f"Tập test: `{dataset['test_file']}` ({dataset['examples']} tin nhắn)",
        "",
        "| Mô hình | Precision spam | Recall spam | F1 spam | ROC-AUC | Ma trận nhầm lẫn [[TN, FP], [FN, TP]] | CPU ms/tin (batch 1) | CPU ms/tin (batch 32) | Tham số | Thời gian train (s) | Cấu hình |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for kind in ("mlp", "lstm", "transformer"):
        metrics = payload["models"][kind]
        config = metrics["config"]
        config_text = ", ".join(
            f"{key}={config[key] if config[key] is not None else 'n/a'}"
            for key in ("input_type", "vocab_size", "max_len", "embedding_dim", "hidden_dim",
                        "optimizer", "epochs_trained", "best_epoch", "seed", "decision_threshold")
        )
        auc = "n/a" if metrics["roc_auc"] is None else f"{metrics['roc_auc']:.3f}"
        lines.append(
            f"| {kind.upper() if kind != 'transformer' else 'Transformer'} | "
            f"{metrics['spam_precision']:.3f} | {metrics['spam_recall']:.3f} | {metrics['spam_f1']:.3f} | "
            f"{auc} | `{metrics['confusion_matrix']}` | {metrics['latency_ms_batch_1']:.3f} | "
            f"{metrics['latency_ms_batch_32']:.3f} | {metrics['parameters']:,} | "
            f"{metrics['train_seconds']:.1f} | {config_text} |"
        )
    lines.extend([
        "",
        "Cả ba mô hình được train từ đầu trên cùng train split và chọn checkpoint theo F1 spam trên validation. "
        "Chuẩn hóa văn bản giống nhau; MLP dùng TF-IDF từ/ký tự, còn LSTM và Transformer dùng token ID. "
        "Ngưỡng 0,5 được cố định cho cả ba, không tối ưu riêng theo validation. "
        "Latency CPU tính cả bước vector hóa (TF-IDF hoặc token ID) và inference, không phải thời gian inference thuần; "
        "đã loại lượt warmup đầu tiên. Số liệu được đo trên test split đã khóa.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "eval/results/evaluation.json")
    parser.add_argument("--output", type=Path, default=ROOT / "eval/results/report.md")
    args = parser.parse_args()
    report = render_report(json.loads(args.input.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()
