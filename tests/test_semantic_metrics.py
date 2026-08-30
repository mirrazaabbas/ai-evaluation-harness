import json
import unittest

from semantic_metrics import (
    JudgeResult,
    OpenAICompatibleJudge,
    cosine_similarity,
    factual_support_proxy,
    semantic_similarity,
)


class FakeEmbeddingProvider:
    def embed(self, texts):
        mapping = {
            "same": [1.0, 0.0],
            "same too": [1.0, 0.0],
            "different": [0.0, 1.0],
        }
        return [mapping[text] for text in texts]


class SemanticMetricTests(unittest.TestCase):
    def test_cosine_and_semantic_similarity(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)
        self.assertEqual(semantic_similarity("same", "same too", FakeEmbeddingProvider()), 1.0)

    def test_factual_support_proxy(self):
        score = factual_support_proxy(
            "Agents use retries. Agents use approval gates.",
            ["Reliable agents use retries and approval gates."],
        )
        self.assertGreater(score, 0.5)

    def test_structured_judge_adapter(self):
        def transport(request, timeout):
            self.assertEqual(timeout, 4)
            body = json.loads(request.data.decode("utf-8"))
            self.assertEqual(body["response_format"]["type"], "json_object")
            content = json.dumps({"score": 0.8, "explanation": "mostly supported"})
            return json.dumps({"choices": [{"message": {"content": content}}]}).encode()

        judge = OpenAICompatibleJudge(
            model="test",
            api_key="secret",
            endpoint="https://judge.example.test/v1/chat/completions",
            timeout_seconds=4,
            transport=transport,
        )
        result = judge.judge(output="answer", reference="reference", rubric="grounding")
        self.assertEqual(result, JudgeResult(0.8, "mostly supported", "judge-rubric/v1"))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            cosine_similarity([], [])
        with self.assertRaises(ValueError):
            semantic_similarity("", "x", FakeEmbeddingProvider())
        with self.assertRaises(ValueError):
            factual_support_proxy("", [])
        with self.assertRaises(ValueError):
            OpenAICompatibleJudge(model="", api_key="x")


if __name__ == "__main__":
    unittest.main()
