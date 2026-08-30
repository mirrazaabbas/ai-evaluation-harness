import tempfile
import unittest
from pathlib import Path

from history import EvaluationHistoryStore, render_history_html


class HistoryTests(unittest.TestCase):
    def test_store_and_render(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EvaluationHistoryStore(Path(directory) / "history.db")
            store.record(
                run_id="r1",
                suite="rag",
                score=0.9,
                pass_rate=1.0,
                latency_ms=12,
                cost_usd=0.01,
                metadata={"commit": "abc"},
            )
            items = store.recent("rag")
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].run_id, "r1")
            report = render_history_html(items)
            self.assertIn("Evaluation History", report)
            self.assertIn("r1", report)

    def test_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EvaluationHistoryStore(Path(directory) / "history.db")
            with self.assertRaises(ValueError):
                store.record(run_id="", suite="x", score=1, pass_rate=1)
            with self.assertRaises(ValueError):
                store.record(run_id="x", suite="x", score=2, pass_rate=1)
            with self.assertRaises(ValueError):
                store.recent("x", 0)


if __name__ == "__main__":
    unittest.main()
