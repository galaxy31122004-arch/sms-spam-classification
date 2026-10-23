"""Render evaluation JSON as a compact Markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _metric_row(name: str, metrics: dict, coverage: float | None = None) -> str:
    coverage_cell = _percent(coverage) if coverage is not None else "100.0%"
    return (
        f"| {name} | {metrics['examples']} | {coverage_cell} | "
        f"{_percent(metrics['spam_precision'])} | {_percent(metrics['spam_recall'])} | "
        f"{_percent(metrics['spam_f1'])} | {_percent(metrics['false_positive_rate_ham'])} | "
        f"{metrics['avg_latency_ms']:.2f} ms |"
    )


def render_report(payload: dict) -> str:
    dataset = payload["dataset"]
    transformer = payload["transformer"]
    tier1 = payload["tier1"]
    cascade = payload["cascade"]
    route_counts = tier1["route_counts"]
    lines = [
        "# SMS Spam Evaluation",
        "",
        f"Test set: `{dataset['test_file']}` ({dataset['examples']} messages)",
        f"Transformer spam label: `{transformer['spam_label_id']}`",
        "",
        "| Path | Evaluated | Tier-1 coverage | Spam precision | Spam recall | Spam F1 | Ham false-positive rate | Avg latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        _metric_row("Transformer", transformer),
        _metric_row("Tier-1 decided subset", tier1, tier1["coverage"]),
        _metric_row("Cascade", cascade),
        "",
        f"Tier-2 rate: {_percent(cascade['tier2_rate'])}",
        f"Tier-1 routes: {route_counts['tier1_spam']} spam, {route_counts['tier1_ham']} ham, {route_counts['tier2']} to Tier-2.",
        "",
        "Tier-1 is a pre-filter. Messages routed to Tier-2 are classified by the Transformer.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "eval" / "results" / "evaluation.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "eval" / "results" / "report.md")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    report = render_report(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()
