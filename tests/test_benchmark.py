import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import mock_open, patch

import numpy as np
import torch

from eval.evaluate import benchmark_prediction, calculate_metrics, evaluate
from eval.report import render_report
from models.common import (MLPClassifier, SequenceClassifier, build_vocab, encode_texts, read_split,
                           training_loader, transformer_lr_scale, transform_texts)


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

    def test_mlp_returns_one_logit_even_for_single_message(self):
        model = MLPClassifier(input_dim=4)
        for size in (1, 2):
            self.assertEqual(tuple(model(torch.zeros(size, 4)).shape), (size,))

    def test_vocab_capacity_reserves_special_tokens(self):
        with self.assertRaisesRegex(ValueError, "PAD and UNK"):
            build_vocab(["offer now"], max_size=1)
        self.assertEqual(build_vocab(["offer now"], max_size=2), {"<PAD>": 0, "<UNK>": 1})

    def test_single_step_schedule_keeps_nonzero_learning_rate(self):
        self.assertEqual(transformer_lr_scale(1, 1), 1.0)
        self.assertEqual(transformer_lr_scale(10, 100), 1.0)
        self.assertEqual(transformer_lr_scale(100, 100), 0.0)
        self.assertAlmostEqual(transformer_lr_scale(55, 100), 0.5)
        with self.assertRaises(ValueError):
            transformer_lr_scale(0, 10)

    def test_malformed_split_reports_a_validation_error(self):
        with patch("pathlib.Path.open", mock_open(read_data="label,text\nham\n")):
            with self.assertRaisesRegex(ValueError, "Invalid label or empty text"):
                read_split(Path("unused.csv"))

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
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["roc_auc"], 0.75)
        adjusted = calculate_metrics([0, 0, 1, 1], [0.1, 0.8, 0.9, 0.4], threshold=0.85)
        self.assertEqual(adjusted["confusion_matrix"], [[2, 0], [1, 1]])
        self.assertEqual(adjusted["accuracy"], 0.75)

    def test_accuracy_and_fixed_threshold_boundary(self):
        metrics = calculate_metrics([0, 1, 1], [0.49, 0.5, 0.9])
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["confusion_matrix"], [[1, 0], [0, 2]])
        self.assertIsNone(calculate_metrics([0, 0], [0.1, 0.2])["roc_auc"])

    def test_metrics_reject_invalid_inputs(self):
        cases = [([], [], 0.5), ([0], [0.1, 0.2], 0.5), ([2], [0.1], 0.5),
                 ([0], [float("nan")], 0.5), ([0], [float("inf")], 0.5),
                 ([0], [-0.1], 0.5), ([0], [1.1], 0.5), ([0], [0.1], float("nan"))]
        for labels, scores, threshold in cases:
            with self.subTest(labels=labels, scores=scores, threshold=threshold):
                with self.assertRaises(ValueError):
                    calculate_metrics(labels, scores, threshold)

    def test_latency_is_amortized_per_message_including_partial_batch(self):
        with patch("eval.evaluate.time.perf_counter", side_effect=[10.0, 10.006]), \
             patch("eval.evaluate.transform_texts", side_effect=["first", "last"]) as transform, \
             patch("eval.evaluate.predict_probabilities", side_effect=[[0.1, 0.2], [0.3]]):
            scores, latency = benchmark_prediction(None, None, "mlp", ["a", "b", "c"], None, 2)
        self.assertEqual(scores, [0.1, 0.2, 0.3])
        self.assertEqual(latency, 2.0)
        self.assertEqual(transform.call_count, 2)
        for texts, batch_size in (([], 1), (["a"], 0)):
            with self.assertRaises(ValueError):
                benchmark_prediction(None, None, "mlp", texts, None, batch_size)

    def test_evaluator_rejects_a_different_checkpoint_threshold(self):
        with patch("eval.evaluate.read_split", return_value=(["ham", "spam"], np.array([0, 1]))), \
             patch("eval.evaluate.load_checkpoint", return_value=(None, None, {"decision_threshold": 0.8})), \
             patch("eval.evaluate.benchmark_prediction") as predict, \
             patch("pathlib.Path.is_file", return_value=True):
            with self.assertRaisesRegex(ValueError, "fixed decision threshold 0.5"):
                evaluate(Path("unused.csv"), Path("unused-checkpoints"))
        predict.assert_not_called()

    def test_checkpoint_lookup_never_falls_back_to_another_root(self):
        from eval.evaluate import resolve_checkpoint_dir
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, "Missing mlp checkpoint"):
                resolve_checkpoint_dir(root, "mlp")
            (root / "mlp" / "checkpoint").mkdir(parents=True)
            (root / "mlp" / "checkpoint" / "config.json").write_text("{}", encoding="utf-8")
            self.assertEqual(resolve_checkpoint_dir(root, "mlp"), root / "mlp" / "checkpoint")

    def test_evaluator_exports_each_checkpoints_training_settings(self):
        # In-memory fixtures exercise metadata wiring without models or result files.
        checkpoints = []
        for index, kind in enumerate(("mlp", "lstm", "transformer"), 1):
            config = {"kind": kind, "max_len": 100, "parameters": 10, "train_seconds": 1.0,
                      "decision_threshold": 0.5, "learning_rate": index * 0.001,
                      "batch_size": index * 16, "checkpoint_selection": "max_val_spam_f1_first_tie"}
            checkpoints.append((torch.nn.Linear(1, 1), None, config))
        with patch("eval.evaluate.read_split", return_value=(["ham", "spam"], np.array([0, 1]))), \
             patch("eval.evaluate.load_checkpoint", side_effect=checkpoints), \
             patch("eval.evaluate.benchmark_prediction", return_value=([0.1, 0.9], 1.0)), \
             patch("pathlib.Path.is_file", return_value=True):
            payload = evaluate(Path("unused-test.csv"), Path("unused-checkpoints"))
        for index, kind in enumerate(("mlp", "lstm", "transformer"), 1):
            self.assertEqual(payload["models"][kind]["config"]["learning_rate"], index * 0.001)
            self.assertEqual(payload["models"][kind]["config"]["batch_size"], index * 16)
            self.assertEqual(payload["models"][kind]["config"]["checkpoint_selection"], "max_val_spam_f1_first_tie")
            self.assertEqual(payload["models"][kind]["accuracy"], 1.0)
            self.assertEqual(payload["models"][kind]["parameters"], 2)

    def test_report_compares_three_models_without_legacy_routes(self):
        metric = {"accuracy": 0.85, "spam_precision": 0.8, "spam_recall": 0.7, "spam_f1": 0.746,
                  "roc_auc": 0.9, "confusion_matrix": [[8, 1], [1, 3]],
                  "latency_ms_batch_1": 2.0, "latency_ms_batch_32": 0.5,
                  "parameters": 1000, "train_seconds": 10.0,
                  "config": {"input_type": "tfidf_word_char", "vocab_size": 100, "max_len": None,
                             "embedding_dim": None, "hidden_dim": None, "optimizer": "Adam",
                             "epochs_trained": 2, "best_epoch": 1, "seed": 42, "decision_threshold": 0.5,
                             "checkpoint_selection": "max_val_spam_f1_first_tie"}}
        payload = {"dataset": {"test_file": "data/splits/test.csv", "examples": 13},
                   "models": {name: deepcopy(metric) for name in ("mlp", "lstm", "transformer")}}
        for index, config in enumerate(payload["models"].values(), 1):
            config["config"].update(learning_rate=index * 0.001, batch_size=index * 16)
        report = render_report(payload)
        self.assertIn("| MLP |", report)
        self.assertIn("| LSTM |", report)
        self.assertIn("| Transformer |", report)
        self.assertIn("embedding_dim=n/a", report)
        self.assertIn("decision_threshold=0.5", report)
        self.assertIn("| Accuracy |", report)
        self.assertIn("| 0.850 |", report)
        for index, kind in enumerate(("mlp", "lstm", "transformer"), 1):
            row = next(line for line in report.splitlines() if line.startswith(f"| {kind.upper() if kind != 'transformer' else 'Transformer'} |"))
            self.assertIn(f"learning_rate={index * 0.001}", row)
            self.assertIn(f"batch_size={index * 16}", row)
            self.assertIn("checkpoint_selection=max_val_spam_f1_first_tie", row)
        del payload["models"]["mlp"]["accuracy"]
        del payload["models"]["mlp"]["config"]["learning_rate"]
        self.assertIn("| MLP | n/a |", render_report(payload))
        self.assertIn("learning_rate=n/a", render_report(payload))
        self.assertIn("vector hóa", report)
        self.assertNotIn("Tier-1", report)


if __name__ == "__main__":
    unittest.main()
