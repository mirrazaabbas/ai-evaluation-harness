"""Evaluation harness for comparing AI-system outputs against quality criteria."""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = BASE_DIR / "sample_cases.json"


@dataclass(frozen=True)
class Score:
    case_id: str
    keyword_recall: float
    groundedness: float
    concision: float
    overall: float


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def keyword_recall(output: str, required: list[str]) -> float:
    if not required:
        return 1.0
    text = output.lower()
    return sum(term.lower() in text for term in required) / len(required)


def groundedness(output: str, context: str) -> float:
    out = words(output)
    ctx = words(context)
    return len(out & ctx) / len(out) if out else 0.0


def concision(output: str, max_words: int) -> float:
    if max_words <= 0:
        raise ValueError("max_words must be greater than zero.")
    count = len(output.split())
    return 1.0 if count <= max_words else max_words / count


def validate_case(case: dict[str, Any], index: int) -> None:
    for field in ("id", "output"):
        if field not in case or not isinstance(case[field], str) or not case[field].strip():
            raise ValueError(f"Case {index}: '{field}' must be a non-empty string.")
    if "required_terms" in case and not isinstance(case["required_terms"], list):
        raise ValueError(f"Case {index}: 'required_terms' must be a list.")
    if "max_words" in case and (not isinstance(case["max_words"], int) or case["max_words"] <= 0):
        raise ValueError(f"Case {index}: 'max_words' must be a positive integer.")


def evaluate(case: dict[str, Any]) -> Score:
    recall = keyword_recall(case["output"], case.get("required_terms", []))
    ground = groundedness(case["output"], case.get("context", ""))
    concise = concision(case["output"], case.get("max_words", 120))
    overall = 0.45 * recall + 0.40 * ground + 0.15 * concise
    return Score(
        case["id"],
        round(recall, 3),
        round(ground, 3),
        round(concise, 3),
        round(overall, 3),
    )


def load_cases(path: Path) -> list[dict[str, Any]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Dataset not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Dataset contains invalid JSON: {exc}") from exc
    if not isinstance(raw, list) or not raw:
        raise ValueError("Dataset must be a non-empty JSON array.")
    for index, case in enumerate(raw, 1):
        if not isinstance(case, dict):
            raise ValueError(f"Case {index} must be a JSON object.")
        validate_case(case, index)
    return raw


def build_report(cases: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [evaluate(case) for case in cases]
    return {
        "cases": [asdict(score) for score in scores],
        "average_overall": round(sum(score.overall for score in scores) / len(scores), 3),
        "pass_rate": round(sum(score.overall >= 0.65 for score in scores) / len(scores), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AI outputs against repeatable quality metrics.")
    parser.add_argument("dataset", nargs="?", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    try:
        report = build_report(load_cases(args.dataset))
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
