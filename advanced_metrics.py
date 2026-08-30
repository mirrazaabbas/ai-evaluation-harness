"""Advanced deterministic metrics for RAG and agent evaluation.

The functions here are deliberately provider-independent and deterministic so
that they can run in CI without model credentials. They complement the core
benchmark runner in ``evaluate.py``.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text or "")]


def token_f1(reference: str, candidate: str) -> float:
    """Bag-of-tokens F1 for deterministic answer-similarity regression checks."""
    ref = _tokens(reference)
    cand = _tokens(candidate)
    if not ref and not cand:
        return 1.0
    if not ref or not cand:
        return 0.0

    ref_counts: dict[str, int] = {}
    cand_counts: dict[str, int] = {}
    for token in ref:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    for token in cand:
        cand_counts[token] = cand_counts.get(token, 0) + 1

    overlap = sum(min(count, cand_counts.get(token, 0)) for token, count in ref_counts.items())
    precision = overlap / len(cand)
    recall = overlap / len(ref)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def recall_at_k(relevant_ids: list[str], retrieved_ids: list[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be greater than zero.")
    relevant = set(relevant_ids)
    if not relevant:
        return 1.0
    hits = len(relevant.intersection(retrieved_ids[:k]))
    return hits / len(relevant)


def reciprocal_rank(relevant_ids: list[str], retrieved_ids: list[str]) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 1.0
    for rank, item in enumerate(retrieved_ids, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevant_ids: list[str], retrieved_ids: list[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be greater than zero.")
    relevant = set(relevant_ids)
    if not relevant:
        return 1.0

    dcg = 0.0
    for index, item in enumerate(retrieved_ids[:k], start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(index + 1)

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def citation_precision_recall(expected: list[str], actual: list[str]) -> dict[str, float]:
    expected_set = set(expected)
    actual_set = set(actual)
    if not actual_set:
        precision = 1.0 if not expected_set else 0.0
    else:
        precision = len(expected_set & actual_set) / len(actual_set)
    if not expected_set:
        recall = 1.0
    else:
        recall = len(expected_set & actual_set) / len(expected_set)
    return {"precision": precision, "recall": recall}


def tool_call_accuracy(expected: list[dict[str, Any]], actual: list[dict[str, Any]]) -> float:
    """Exact ordered tool-name/argument accuracy for agent regression suites."""
    if not expected and not actual:
        return 1.0
    if len(expected) != len(actual):
        return 0.0
    correct = 0
    for exp, got in zip(expected, actual, strict=True):
        if exp.get("name") == got.get("name") and exp.get("arguments", {}) == got.get("arguments", {}):
            correct += 1
    return correct / len(expected) if expected else 0.0


@dataclass(frozen=True)
class RegressionPolicy:
    """Quality gate for baseline-vs-candidate benchmark summaries."""

    max_average_score_drop: float = 0.02
    max_pass_rate_drop: float = 0.02
    max_latency_increase_ratio: float | None = 0.25
    max_cost_increase_ratio: float | None = 0.25

    def __post_init__(self) -> None:
        for name in ("max_average_score_drop", "max_pass_rate_drop"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative.")
        for name in ("max_latency_increase_ratio", "max_cost_increase_ratio"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} cannot be negative when provided.")

    def evaluate(self, baseline: dict[str, float], candidate: dict[str, float]) -> list[str]:
        violations: list[str] = []
        score_drop = baseline.get("average_score", 0.0) - candidate.get("average_score", 0.0)
        if score_drop > self.max_average_score_drop:
            violations.append(f"average_score dropped by {score_drop:.4f}")

        pass_drop = baseline.get("pass_rate", 0.0) - candidate.get("pass_rate", 0.0)
        if pass_drop > self.max_pass_rate_drop:
            violations.append(f"pass_rate dropped by {pass_drop:.4f}")

        self._check_ratio(
            "average_latency_ms",
            baseline,
            candidate,
            self.max_latency_increase_ratio,
            violations,
        )
        self._check_ratio(
            "average_cost_usd",
            baseline,
            candidate,
            self.max_cost_increase_ratio,
            violations,
        )
        return violations

    @staticmethod
    def _check_ratio(
        metric: str,
        baseline: dict[str, float],
        candidate: dict[str, float],
        allowed: float | None,
        violations: list[str],
    ) -> None:
        if allowed is None or metric not in baseline or metric not in candidate:
            return
        base = baseline[metric]
        current = candidate[metric]
        if base <= 0:
            return
        ratio = (current - base) / base
        if ratio > allowed:
            violations.append(f"{metric} increased by {ratio:.1%}")
