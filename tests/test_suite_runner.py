import unittest

from suite_runner import run_suite, score_case


class SuiteRunnerTests(unittest.TestCase):
    def test_multiple_case_types(self):
        cases = [
            {
                "id": "retrieval",
                "kind": "retrieval",
                "relevant_ids": ["a"],
                "retrieved_ids": ["a", "b"],
                "k": 2,
            },
            {
                "id": "routing",
                "kind": "routing",
                "expected": "research",
                "actual": "research",
            },
            {
                "id": "failure",
                "kind": "failure",
                "expected": "blocked",
                "actual": "blocked",
            },
            {
                "id": "injection",
                "kind": "prompt_injection",
                "expected": "blocked",
                "actual": "blocked",
            },
        ]
        result = run_suite(cases)
        self.assertEqual(result["case_count"], 4)
        self.assertEqual(result["pass_rate"], 1.0)

    def test_tool_and_factual_support(self):
        tool_score = score_case(
            {
                "kind": "tool_call",
                "expected": [{"name": "search"}],
                "actual": [{"name": "search"}],
            }
        )
        self.assertEqual(tool_score, 1.0)
        factual = score_case(
            {
                "kind": "factual_support",
                "output": "Retries improve reliability.",
                "context": ["Retries improve reliability for workflows."],
            }
        )
        self.assertGreater(factual, 0.5)

    def test_validation(self):
        with self.assertRaises(ValueError):
            run_suite([], pass_threshold=2)
        with self.assertRaises(ValueError):
            score_case({"kind": "unknown"})


if __name__ == "__main__":
    unittest.main()
