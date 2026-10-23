import math
import unittest

from tier1.model import Tier1Model


class FakePipeline:
    def predict_proba(self, texts):
        return [[0.75, 0.25], [0.1, 0.9]][:len(texts)]


class Tier1ModelTests(unittest.TestCase):
    def test_converts_calibrated_spam_probability_to_log_odds(self):
        model = Tier1Model(FakePipeline(), ["ham", "spam"])

        scores = model.predict_many(["ham", "spam"])

        self.assertAlmostEqual(scores[0][0], math.log(1 / 3))
        self.assertEqual(scores[0][1], 0.25)
        self.assertAlmostEqual(scores[1][0], math.log(9))
        self.assertEqual(scores[1][1], 0.9)

    def test_empty_batch_returns_no_predictions(self):
        model = Tier1Model(FakePipeline(), ["ham", "spam"])

        self.assertEqual(model.predict_many([]), [])


if __name__ == "__main__":
    unittest.main()
