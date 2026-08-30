import tempfile
import unittest
from pathlib import Path

import portfolio_bridge


class PortfolioBridgeTests(unittest.TestCase):
    def setUp(self):
        self.record = {
            "schema_version": "portfolio-evidence/v1",
            "producer": "agent-workflow-engine",
            "query": "Explain grounded RAG",
            "output": "Grounded RAG uses retrieved evidence.",
            "retrieved_ids": ["doc-a", "doc-b"],
            "citations": ["doc-a", "doc-b"],
            "context": ["Evidence A", "Evidence B"],
            "tool_calls": [{"name": "rag.answer", "arguments": {"top_k": 2}}],
            "latency_ms": 15,
        }
        self.expected = {
            "reference": "Grounded RAG uses retrieved evidence.",
            "relevant_ids": ["doc-a"],
            "expected_citations": ["doc-a"],
            "expected_tool_calls": [{"name": "rag.answer", "arguments": {"top_k": 2}}],
            "k": 2,
            "pass_threshold": 0.7,
        }

    def test_evaluate_record(self):
        result = portfolio_bridge.evaluate_record(self.record, self.expected)
        self.assertTrue(result["passed"])
        self.assertEqual(result["metrics"]["answer_token_f1"], 1.0)
        self.assertEqual(result["metrics"]["retrieval_recall_at_k"], 1.0)
        self.assertEqual(result["metrics"]["tool_call_accuracy"], 1.0)
        self.assertGreater(result["metrics"]["citation_precision"], 0)

    def test_validation(self):
        self.assertEqual(portfolio_bridge.validate_record(self.record), [])
        invalid = dict(self.record)
        invalid["schema_version"] = "wrong"
        errors = portfolio_bridge.validate_record(invalid)
        self.assertTrue(errors)
        with self.assertRaises(ValueError):
            portfolio_bridge.evaluate_record(invalid, self.expected)

    def test_minimal_record_without_optional_expectations(self):
        result = portfolio_bridge.evaluate_record(self.record, {"pass_threshold": 0.1})
        self.assertFalse(result["passed"])
        self.assertEqual(result["overall_score"], 0.0)

    def test_read_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            path.write_text('{"schema_version":"portfolio-evidence/v1"}', encoding="utf-8")
            self.assertEqual(
                portfolio_bridge._read_json(path)["schema_version"], "portfolio-evidence/v1"
            )
            bad = Path(directory) / "bad.json"
            bad.write_text("[]", encoding="utf-8")
            with self.assertRaises(ValueError):
                portfolio_bridge._read_json(bad)


if __name__ == "__main__":
    unittest.main()
