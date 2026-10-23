"""Load the published sklearn classifier and expose calibrated spam scores."""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import joblib
from huggingface_hub import snapshot_download


MODEL_ID = "jngb-labs/sms-spam-classical"


class Tier1Model:
    def __init__(self, pipeline, labels: list[str]):
        if labels != ["ham", "spam"]:
            raise ValueError(f"Unexpected Tier-1 label order: {labels!r}")
        self.pipeline = pipeline
        self.labels = labels

    @classmethod
    def from_pretrained(cls, model_id: str = MODEL_ID, cache_dir: Path | None = None) -> "Tier1Model":
        snapshot = Path(
            snapshot_download(
                repo_id=model_id,
                allow_patterns=("model/classifier.joblib", "src/features.py"),
                cache_dir=str(cache_dir) if cache_dir else None,
            )
        )
        source_dir = str(snapshot / "src")
        if source_dir not in sys.path:
            sys.path.insert(0, source_dir)
        importlib.import_module("features")
        bundle = joblib.load(snapshot / "model" / "classifier.joblib")
        return cls(bundle["pipeline"], list(bundle["labels"]))

    def predict(self, text: str) -> tuple[float, float]:
        return self.predict_many([text])[0]

    def predict_many(self, texts: list[str]) -> list[tuple[float, float]]:
        if not texts:
            return []
        probabilities = self.pipeline.predict_proba(texts)
        spam_index = self.labels.index("spam")
        results = []
        for row in probabilities:
            spam_probability = float(row[spam_index])
            epsilon = 1e-7
            clipped = min(max(spam_probability, epsilon), 1 - epsilon)
            results.append((math.log(clipped / (1 - clipped)), spam_probability))
        return results
