import tempfile
import unittest
from pathlib import Path

import evaluate


class EvaluationHarnessTests(unittest.TestCase):
    def test_metrics(self):
        case = {
            "id": "case-1",
            "context": "RAG retrieves source context before generation",
            "output": "RAG retrieves source context before generation",
            "required_terms": ["RAG", "source context"],
            "max_words": 20,
        }
        score = evaluate.evaluate(case)
        self.assertEqual(score.keyword_recall, 1.0)
        self.assertEqual(score.groundedness, 1.0)
        self.assertGreaterEqual(score.overall, 0.9)

    def test_edge_cases(self):
        self.assertEqual(evaluate.keyword_recall("anything", []), 1.0)
        self.assertEqual(evaluate.groundedness("", "context"), 0.0)
        with self.assertRaises(ValueError):
            evaluate.concision("text", 0)

    def test_dataset_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            bad.write_text("{bad", encoding="utf-8")
            with self.assertRaises(ValueError):
                evaluate.load_cases(bad)

    def test_sample_report(self):
        cases = evaluate.load_cases(Path("sample_cases.json"))
        report = evaluate.build_report(cases)
        self.assertEqual(len(report["cases"]), 2)
        self.assertGreaterEqual(report["pass_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
