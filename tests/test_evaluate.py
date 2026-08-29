import tempfile
import unittest
from pathlib import Path

import evaluate


class EvaluationHarnessTests(unittest.TestCase):
    def test_metrics(self):
        case = {
            "id": "case-1",
            "context": "RAG retrieves source context before generation",
            "output": "RAG retrieves source context before generation [doc-1]",
            "required_terms": ["RAG", "source context"],
            "expected_citations": ["[doc-1]"],
            "max_words": 20,
        }
        score = evaluate.evaluate(case)
        self.assertEqual(score.keyword_recall, 1.0)
        self.assertGreater(score.groundedness, 0.7)
        self.assertEqual(score.citation_coverage, 1.0)
        self.assertGreaterEqual(score.overall, 0.8)

    def test_edge_cases(self):
        self.assertEqual(evaluate.keyword_recall("anything", []), 1.0)
        self.assertEqual(evaluate.groundedness("", "context"), 0.0)
        self.assertEqual(evaluate.citation_coverage("answer", []), 1.0)
        with self.assertRaises(ValueError):
            evaluate.concision("text", 0)
        with self.assertRaises(ValueError):
            evaluate.build_report([{"id": "x", "output": "answer"}], threshold=1.1)

    def test_case_validation_variants(self):
        invalid_cases = [
            {},
            {"id": "x", "output": ""},
            {"id": "x", "output": "answer", "required_terms": "bad"},
            {"id": "x", "output": "answer", "expected_citations": "bad"},
            {"id": "x", "output": "answer", "max_words": 0},
            {"id": "x", "output": "answer", "cost_usd": -0.1},
        ]
        for index, case in enumerate(invalid_cases, 1):
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    evaluate.validate_case(case, index)

    def test_dataset_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            bad = folder / "bad.json"
            bad.write_text("{bad", encoding="utf-8")
            with self.assertRaises(ValueError):
                evaluate.load_cases(bad)

            invalid_metric = folder / "invalid.json"
            invalid_metric.write_text(
                '[{"id":"x","output":"answer","latency_ms":-1}]', encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                evaluate.load_cases(invalid_metric)

            empty = folder / "empty.json"
            empty.write_text("[]", encoding="utf-8")
            with self.assertRaises(ValueError):
                evaluate.load_cases(empty)

            non_object = folder / "non-object.json"
            non_object.write_text('["bad"]', encoding="utf-8")
            with self.assertRaises(ValueError):
                evaluate.load_cases(non_object)

            with self.assertRaises(ValueError):
                evaluate.load_cases(folder / "missing.json")

    def test_sample_report(self):
        cases = evaluate.load_cases(Path("sample_cases.json"))
        report = evaluate.build_report(cases)
        self.assertEqual(len(report["cases"]), 2)
        self.assertGreaterEqual(report["pass_rate"], 0.5)
        self.assertEqual(report["threshold"], evaluate.PASS_THRESHOLD)
        self.assertIsNone(report["average_citation_coverage"])

    def test_operational_metrics(self):
        cases = [
            {
                "id": "x",
                "output": "grounded answer",
                "context": "grounded answer",
                "latency_ms": 120,
                "cost_usd": 0.002,
            },
            {
                "id": "y",
                "output": "grounded answer",
                "context": "grounded answer",
                "latency_ms": 180,
                "cost_usd": 0.004,
            },
        ]
        report = evaluate.build_report(cases)
        self.assertEqual(report["average_latency_ms"], 150.0)
        self.assertEqual(report["average_cost_usd"], 0.003)

    def test_baseline_comparison(self):
        baseline = {"average_overall": 0.7, "pass_rate": 0.5}
        candidate = {"average_overall": 0.8, "pass_rate": 1.0}
        comparison = evaluate.compare_reports(baseline, candidate)
        self.assertEqual(comparison["average_overall_delta"], 0.1)
        self.assertEqual(comparison["pass_rate_delta"], 0.5)

    def test_html_report(self):
        report = evaluate.build_report(
            [{"id": "<unsafe>", "output": "answer", "context": "answer"}]
        )
        rendered = evaluate.render_html_report(
            report,
            {"average_overall_delta": 0.1, "pass_rate_delta": 0.0},
        )
        self.assertIn("AI Evaluation Report", rendered)
        self.assertIn("&lt;unsafe&gt;", rendered)
        self.assertNotIn("<unsafe>", rendered)
        self.assertNotIn("Baseline comparison", evaluate.render_html_report(report))


if __name__ == "__main__":
    unittest.main()
