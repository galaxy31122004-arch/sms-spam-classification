"""Shared data, model, and training code for the three scratch benchmarks."""

from __future__ import annotations

import csv
import json
import random
import re
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
TOKEN_RE = re.compile(r"<[A-Z]+>|\w+|[!$%]")
KINDS = ("mlp", "lstm", "transformer")
DECISION_THRESHOLD = 0.5


def read_split(path: Path) -> tuple[list[str], np.ndarray]:
    texts, labels = [], []
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not {"label", "text"}.issubset(reader.fieldnames or ()):
            raise ValueError(f"{path} needs label and text columns")
        for line, row in enumerate(reader, 2):
            label = row["label"].strip().lower()
            if label not in ("ham", "spam") or not row["text"].strip():
                raise ValueError(f"Invalid label or empty text at {path}:{line}")
            texts.append(row["text"])
            labels.append(int(label == "spam"))
    if not texts:
        raise ValueError(f"Empty split: {path}")
    return texts, np.asarray(labels, dtype=np.float32)


def build_vocab(texts: list[str], max_size: int = 10000) -> dict[str, int]:
    from collections import Counter

    counts = Counter(token for text in texts for token in TOKEN_RE.findall(text))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:max_size - 2]:
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode_texts(texts: list[str], vocab: dict[str, int], max_len: int) -> np.ndarray:
    encoded = np.zeros((len(texts), max_len), dtype=np.int64)
    for index, text in enumerate(texts):
        tokens = (TOKEN_RE.findall(text) or ["<UNK>"])[:max_len]
        encoded[index, :len(tokens)] = [vocab.get(token, 1) for token in tokens]
    return encoded


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.4),
                                    nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features).squeeze(-1)


class SequenceClassifier(nn.Module):
    def __init__(self, kind: str, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 128,
                 max_len: int = 100):
        super().__init__()
        self.kind = kind
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if kind == "lstm":
            self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
            self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(hidden_dim * 2, 64), nn.ReLU(), nn.Linear(64, 1))
        elif kind == "transformer":
            self.position = nn.Embedding(max_len, embedding_dim)
            layer = nn.TransformerEncoderLayer(embedding_dim, 4, hidden_dim * 2, dropout=0.1,
                                               batch_first=True, norm_first=False)
            self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
            self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(embedding_dim, 1))
        else:
            raise ValueError(f"Unknown sequence model: {kind}")

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = tokens.ne(0)
        embedded = self.embedding(tokens)
        if self.kind == "lstm":
            lengths = mask.sum(dim=1).clamp(min=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
            _, (hidden, _) = self.encoder(packed)
            pooled = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        else:
            positions = torch.arange(tokens.shape[1], device=tokens.device)
            encoded = self.encoder(embedded + self.position(positions), src_key_padding_mask=~mask)
            pooled = (encoded * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
        return self.head(pooled).squeeze(-1)


def feature_encoder(kind: str, train_texts: list[str], max_vocab: int, max_len: int):
    if kind == "mlp":
        vectorizer = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), max_features=max_vocab,
                                     token_pattern=r"(?u)<[A-Z]+>|\b\w+\b|[!$%]", lowercase=False)),
            ("char", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=5000)),
        ])
        vectorizer.fit(train_texts)
        return vectorizer, {"input_type": "tfidf_word_char", "vocab_size": len(vectorizer.get_feature_names_out()),
                            "max_len": None}
    vocab = build_vocab(train_texts, max_vocab)
    return vocab, {"input_type": "token_ids", "vocab_size": len(vocab), "max_len": max_len}


def transform_texts(kind: str, encoder, texts: list[str], max_len: int | None) -> torch.Tensor:
    if kind == "mlp":
        return torch.from_numpy(encoder.transform(texts).toarray().astype(np.float32))
    if max_len is None or max_len < 1:
        raise ValueError("Sequence models require a positive max_len")
    return torch.from_numpy(encode_texts(texts, encoder, max_len))


def training_loader(features: torch.Tensor, labels: torch.Tensor, batch_size: int, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(features, labels), batch_size=batch_size, shuffle=True, generator=generator)


def create_model(kind: str, config: dict) -> nn.Module:
    if kind == "mlp":
        return MLPClassifier(config["vocab_size"])
    return SequenceClassifier(kind, config["vocab_size"], config["embedding_dim"],
                              config["hidden_dim"], config["max_len"])


def predict_probabilities(model: nn.Module, features: torch.Tensor, batch_size: int) -> list[float]:
    model.eval()
    scores = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            scores.extend(torch.sigmoid(model(features[start:start + batch_size])).tolist())
    return scores


def train(kind: str, output_dir: Path, epochs: int = 15, batch_size: int = 64, seed: int = 42,
          max_vocab: int = 10000, max_len: int = 100, patience: int = 3) -> dict:
    if kind not in KINDS or min(epochs, batch_size, max_vocab, max_len, patience) < 1:
        raise ValueError("Invalid model kind or training setting")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    train_texts, train_labels = read_split(ROOT / "data/splits/train.csv")
    val_texts, val_labels = read_split(ROOT / "data/splits/val.csv")
    encoder, details = feature_encoder(kind, train_texts, max_vocab, max_len)
    x_train = transform_texts(kind, encoder, train_texts, max_len)
    x_val = transform_texts(kind, encoder, val_texts, max_len)
    y_train = torch.from_numpy(train_labels)
    config = {"kind": kind, **details, "embedding_dim": None, "hidden_dim": None,
              "optimizer": "AdamW" if kind == "transformer" else "Adam", "learning_rate": 0.001,
              "batch_size": batch_size, "seed": seed, "epochs_requested": epochs, "patience": patience,
              "decision_threshold": DECISION_THRESHOLD, "train_file": "data/splits/train.csv",
              "val_file": "data/splits/val.csv"}
    if kind != "mlp":
        config.update(embedding_dim=128, hidden_dim=128)
    model = create_model(kind, config)
    positive_weight = torch.tensor([(len(train_labels) - train_labels.sum()) / train_labels.sum()])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer_type = torch.optim.AdamW if kind == "transformer" else torch.optim.Adam
    optimizer = optimizer_type(model.parameters(), lr=config["learning_rate"])
    loader = training_loader(x_train, y_train, batch_size, seed)
    total_steps = max(1, epochs * len(loader))
    warmup_steps = max(1, int(total_steps * 0.1))
    best_f1, stale, best_state = -1.0, 0, None
    start_time = time.perf_counter()
    step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for features, labels in loader:
            if kind == "transformer":
                step += 1
                scale = min(step / warmup_steps, max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps)))
                for group in optimizer.param_groups:
                    group["lr"] = config["learning_rate"] * scale
            optimizer.zero_grad()
            loss = loss_fn(model(features), labels)
            loss.backward()
            optimizer.step()
        val_probs = predict_probabilities(model, x_val, batch_size)
        val_f1 = f1_score(val_labels, np.asarray(val_probs) >= config["decision_threshold"], zero_division=0)
        print(f"{kind} epoch {epoch}: val_spam_f1={val_f1:.4f}", flush=True)
        if val_f1 > best_f1:
            best_f1, stale = float(val_f1), 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            config["best_epoch"] = epoch
        else:
            stale += 1
            if stale >= patience:
                break
    config["epochs_trained"] = epoch
    config["train_seconds"] = round(time.perf_counter() - start_time, 3)
    config["val_spam_f1"] = best_f1
    config["parameters"] = sum(p.numel() for p in model.parameters())
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output_dir / "model.pt")
    joblib.dump(encoder, output_dir / "encoder.joblib")
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config


def load_checkpoint(kind: str, directory: Path):
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if config["kind"] != kind:
        raise ValueError(f"Checkpoint at {directory} is not {kind}")
    model = create_model(kind, config)
    model.load_state_dict(torch.load(directory / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    return model, joblib.load(directory / "encoder.joblib"), config
