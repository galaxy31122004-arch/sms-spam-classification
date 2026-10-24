import unittest

import torch

from eval.evaluate import calculate_metrics
from eval.report import render_report
from models.common import SequenceClassifier, build_vocab, encode_texts, training_loader, transform_texts


class BenchmarkTests(unittest.TestCase):
    def test_vocab_is_train_only_and_padding_is_stable(self):
        vocab = build_vocab(["offer now!", "offer $<MONEY>"], max_size=8)
        self.assertEqual(vocab["<PAD>"], 0)
        self.assertEqual(vocab["<UNK>"], 1)
        encoded = encode_texts(["unknown offer!"], vocab, max_len=5)
        self.assertEqual(encoded.shape, (1, 5))
        self.assertEqual(encoded[0, 0], 1)
        self.assertEqual(encoded[0, -1], 0)
        self.assertEqual(encode_texts(["..."], vocab, max_len=5)[0, 0], 1)

    def test_sequence_models_return_one_logit_per_message(self):
        batch = torch.tensor([[2, 3, 0, 0], [4, 5, 6, 7]])
        for kind in ("lstm", "transformer"):
            with self.subTest(kind=kind):
                model = SequenceClassifier(kind, vocab_size=10, embedding_dim=16, hidden_dim=16)
                self.assertEqual(tuple(model(batch).shape), (2,))

    def test_sequence_encoding_requires_max_len(self):
        with self.assertRaisesRegex(ValueError, "max_len"):
            transform_texts("lstm", {"offer": 2}, ["offer"], None)

    def test_batch_order_is_independent_of_global_torch_rng(self):
        features = torch.arange(10)
        labels = features.float()
        first = [batch[0].tolist() for batch in training_loader(features, labels, 3, seed=42)]
        torch.rand(100)
        second = [batch[0].tolist() for batch in training_loader(features, labels, 3, seed=42)]
        self.assertEqual(first, second)

    def test_metrics_use_spam_as_positive_class(self):
        metrics = calculate_metrics([0, 0, 1, 1], [0.1, 0.8, 0.9, 0.4])
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(metrics["spam_f1"], 0.5)
        adjusted = calculate_metrics([0, 0, 1, 1], [0.1, 0.8, 0.9, 0.4], threshold=0.85)
        self.assertEqual(adjusted["confusion_matrix"], [[2, 0], [1, 1]])

    def test_report_compares_three_models_without_legacy_routes(self):
        metric = {"spam_precision": 0.8, "spam_recall": 0.7, "spam_f1": 0.746,
                  "roc_auc": 0.9, "confusion_matrix": [[8, 1], [1, 3]],
                  "latency_ms_batch_1": 2.0, "latency_ms_batch_32": 0.5,
                  "parameters": 1000, "train_seconds": 10.0,
                  "config": {"input_type": "tfidf_word_char", "vocab_size": 100, "max_len": None,
                             "embedding_dim": None, "hidden_dim": None, "optimizer": "Adam",
                             "epochs_trained": 2, "best_epoch": 1, "seed": 42, "decision_threshold": 0.5}}
        payload = {"dataset": {"test_file": "data/splits/test.csv", "examples": 13},
                   "models": {name: metric for name in ("mlp", "lstm", "transformer")}}
        report = render_report(payload)
        self.assertIn("| MLP |", report)
        self.assertIn("| LSTM |", report)
        self.assertIn("| Transformer |", report)
        self.assertIn("embedding_dim=n/a", report)
        self.assertIn("decision_threshold=0.5", report)
        self.assertIn("vector hóa", report)
        self.assertNotIn("Tier-1", report)


if __name__ == "__main__":
    unittest.main()
