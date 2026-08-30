import unittest

import advanced_metrics as metrics


class AdvancedMetricsTests(unittest.TestCase):
    def test_token_f1(self):
        self.assertEqual(metrics.token_f1("", ""), 1.0)
        self.assertEqual(metrics.token_f1("answer", ""), 0.0)
        self.assertAlmostEqual(metrics.token_f1("red blue", "red green"), 0.5)

    def test_retrieval_metrics(self):
        relevant = ["a", "c"]
        retrieved = ["x", "a", "b", "c"]
        self.assertEqual(metrics.recall_at_k(relevant, retrieved, 2), 0.5)
        self.assertEqual(metrics.reciprocal_rank(relevant, retrieved), 0.5)
        self.assertGreater(metrics.ndcg_at_k(relevant, retrieved, 4), 0.0)
        with self.assertRaises(ValueError):
            metrics.recall_at_k(relevant, retrieved, 0)
        with self.assertRaises(ValueError):
            metrics.ndcg_at_k(relevant, retrieved, 0)

    def test_empty_relevant_set_is_fully_satisfied(self):
        self.assertEqual(metrics.recall_at_k([], ["x"], 1), 1.0)
        self.assertEqual(metrics.reciprocal_rank([], ["x"]), 1.0)
        self.assertEqual(metrics.ndcg_at_k([], ["x"], 1), 1.0)

    def test_citation_precision_recall(self):
        result = metrics.citation_precision_recall(["s1", "s2"], ["s2", "s3"])
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall"], 0.5)
        self.assertEqual(metrics.citation_precision_recall([], []), {"precision": 1.0, "recall": 1.0})

    def test_tool_call_accuracy(self):
        expected = [{"name": "search", "arguments": {"q": "rag"}}]
        self.assertEqual(metrics.tool_call_accuracy(expected, expected), 1.0)
        self.assertEqual(
            metrics.tool_call_accuracy(expected, [{"name": "search", "arguments": {"q": "agents"}}]),
            0.0,
        )
        self.assertEqual(metrics.tool_call_accuracy([], []), 1.0)

    def test_regression_policy_passes_acceptable_candidate(self):
        policy = metrics.RegressionPolicy()
        baseline = {
            "average_score": 0.90,
            "pass_rate": 0.90,
            "average_latency_ms": 100,
            "average_cost_usd": 0.10,
        }
        candidate = {
            "average_score": 0.89,
            "pass_rate": 0.89,
            "average_latency_ms": 110,
            "average_cost_usd": 0.11,
        }
        self.assertEqual(policy.evaluate(baseline, candidate), [])

    def test_regression_policy_reports_quality_and_operational_regressions(self):
        policy = metrics.RegressionPolicy(
            max_average_score_drop=0.01,
            max_pass_rate_drop=0.01,
            max_latency_increase_ratio=0.10,
            max_cost_increase_ratio=0.10,
        )
        baseline = {
            "average_score": 0.90,
            "pass_rate": 0.90,
            "average_latency_ms": 100,
            "average_cost_usd": 0.10,
        }
        candidate = {
            "average_score": 0.80,
            "pass_rate": 0.80,
            "average_latency_ms": 130,
            "average_cost_usd": 0.13,
        }
        violations = policy.evaluate(baseline, candidate)
        self.assertEqual(len(violations), 4)

    def test_regression_policy_validation(self):
        with self.assertRaises(ValueError):
            metrics.RegressionPolicy(max_average_score_drop=-1)
        with self.assertRaises(ValueError):
            metrics.RegressionPolicy(max_latency_increase_ratio=-1)


if __name__ == "__main__":
    unittest.main()
