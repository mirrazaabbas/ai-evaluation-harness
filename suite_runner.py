"""Run heterogeneous deterministic benchmark suites for RAG and agent systems."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from advanced_metrics import (
    citation_precision_recall,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    tool_call_accuracy,
)
from semantic_metrics import factual_support_proxy


def score_case(case: dict[str, Any]) -> float:
    kind = case.get("kind")
    if kind == "retrieval":
        relevant = [str(item) for item in case.get("relevant_ids", [])]
        retrieved = [str(item) for item in case.get("retrieved_ids", [])]
        k = int(case.get("k", max(1, len(retrieved))))
        metrics = [
            recall_at_k(relevant, retrieved, k),
            reciprocal_rank(relevant, retrieved),
            ndcg_at_k(relevant, retrieved, k),
        ]
        return sum(metrics) / len(metrics)
    if kind == "citation":
        scores = citation_precision_recall(
            [str(item) for item in case.get("expected", [])],
            [str(item) for item in case.get("actual", [])],
        )
        return (scores["precision"] + scores["recall"]) / 2
    if kind == "tool_call":
        return tool_call_accuracy(
            list(case.get("expected", [])),
            list(case.get("actual", [])),
        )
    if kind == "factual_support":
        return factual_support_proxy(
            str(case.get("output", "")),
            [str(item) for item in case.get("context", [])],
        )
    if kind in {"routing", "failure", "prompt_injection"}:
        return 1.0 if case.get("expected") == case.get("actual") else 0.0
    raise ValueError(f"unsupported benchmark case kind: {kind!r}")


def run_suite(cases: list[dict[str, Any]], *, pass_threshold: float = 0.75) -> dict[str, Any]:
    if not 0 <= pass_threshold <= 1:
        raise ValueError("pass_threshold must be between 0 and 1")
    results = []
    for index, case in enumerate(cases):
        case_id = str(case.get("id", f"case-{index + 1}"))
        score = round(score_case(case), 6)
        results.append(
            {
                "id": case_id,
                "kind": case.get("kind"),
                "score": score,
                "passed": score >= pass_threshold,
            }
        )
    average = sum(item["score"] for item in results) / len(results) if results else 0.0
    pass_rate = sum(1 for item in results if item["passed"]) / len(results) if results else 0.0
    return {
        "case_count": len(results),
        "average_score": round(average, 6),
        "pass_rate": round(pass_rate, 6),
        "pass_threshold": pass_threshold,
        "cases": results,
    }


def _read_cases(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read suite {path}: {exc}") from exc
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError("benchmark suite must contain a JSON array of objects")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a deterministic heterogeneous AI benchmark suite."
    )
    parser.add_argument("suite", type=Path)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_suite(_read_cases(args.suite), pass_threshold=args.threshold)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
