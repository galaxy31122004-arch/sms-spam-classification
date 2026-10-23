import unittest

from eval.report import render_report


class ReportTests(unittest.TestCase):
    def test_report_summarizes_metrics_and_cascade_routing(self):
        payload = {
            "dataset": {"examples": 10, "test_file": "data/splits/test.csv"},
            "transformer": {
                "examples": 10,
                "spam_label_id": 1,
                "spam_precision": 0.9,
                "spam_recall": 0.8,
                "spam_f1": 0.847,
                "false_positive_rate_ham": 0.1,
                "avg_latency_ms": 8.0,
            },
            "tier1": {
                "examples": 7,
                "coverage": 0.7,
                "spam_precision": 0.95,
                "spam_recall": 0.6,
                "spam_f1": 0.731,
                "false_positive_rate_ham": 0.02,
                "route_counts": {"tier1_spam": 2, "tier1_ham": 5, "tier2": 3},
                "avg_latency_ms": 0.8,
            },
            "cascade": {
                "examples": 10,
                "spam_precision": 0.92,
                "spam_recall": 0.9,
                "spam_f1": 0.91,
                "false_positive_rate_ham": 0.05,
                "tier2_rate": 0.3,
                "avg_latency_ms": 4.25,
            },
        }

        report = render_report(payload)

        self.assertIn("Tier-2 rate", report)
        self.assertIn("30.0%", report)
        self.assertIn("91.0%", report)
        self.assertIn("5 ham", report)


if __name__ == "__main__":
    unittest.main()
